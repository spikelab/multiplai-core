"""The ``pi`` agent backend: ``run_agent(..., model="pi:<model-id>@<provider>")``.

Runs one agent invocation through ``pi --mode rpc`` and returns the same
:class:`~multiplai_core.agent_runner.AgentRunResult` the Claude Agent SDK path
returns. Importing this module registers the backend; ``run_agent`` imports it
the first time it sees a ``pi:`` spec.

Adapted from the agent path of DolceEvals ``buildme_pipeline/pi_executor.py``
(DolceTech/DolceEvals@d755fcc): ``parse_pi_spec``, ``TOOL_MAP``,
``_assistant_text`` and ``_files_changed`` come from there. Left out on
purpose: the metering proxy and its ledger (they need ``uvicorn``/``httpx``),
the steering control channel, and the spend cap. The run loop below enforces
the two bounds ``run_agent`` promises, ``max_turns`` and ``timeout_s``, which
is the part of DolceEvals' ``supervisor.py`` this path needs.

How the arguments map onto Pi:

- ``pi:<model-id>@<provider>`` → ``--provider <provider> --model <model-id>``.
  The split is on the LAST ``@``. Both must exist in Pi's own configuration
  (``~/.pi/agent/models.json``), which this package does not manage.
- ``allowed_tools`` → ``-t <pi tools>`` through :data:`TOOL_MAP`. Pi's ``-t``
  is a true allow-list: an unlisted tool does not exist for the run. Names
  with no Pi equivalent are dropped with a warning. ``None``/``[]`` (or an
  allow-list that maps to nothing) → ``--no-tools``, a no-tool call.
  ``disallowed_tools`` removes names from the allow-list.
- ``effort`` → ``--thinking <level>``; ``None`` leaves the flag off, so Pi's
  default applies. ``thinking={"type": "disabled"}`` with no effort →
  ``--thinking off``.
- ``cwd`` → Pi's working directory. ``None`` uses the SDK path's default,
  ``$CLAUDE_CONFIG_DIR/hook-sessions``, so the run cannot edit or pick up
  context files from the caller's project.
- ``files_changed`` → the paths of write/edit tool calls, as the agent passed
  them; bash writes are not reported. Both match the SDK path.
- ``system_prompt`` is put ahead of the prompt in the one message Pi receives.
  The prompt travels as a JSON line on Pi's stdin, so there is no argv limit.
  ``prompt_file_fallback`` is accepted and ignored, as on the SDK path.
- ``read_roots`` is accepted and ignored: Pi has no PreToolUse hook to deny a
  read with, so on this path the read tools are not limited to those folders.
- Usage is summed from the ``usage`` of every assistant ``message_end`` event
  (Pi reports input, cache reads and cache writes separately, the same
  convention as Anthropic). ``cost_usd`` is Pi's own ``cost.total``; when Pi
  reports no cost it is ``0`` and the cost-ledger record carries
  ``"cost_unreported": true`` instead of a price guessed from the model name.

Environment:

    MULTIPLAI_PI_BIN       the pi binary (default ``pi`` on PATH)
    MULTIPLAI_PI_GATE_EXT  path to a Pi extension that gates bash (DolceEvals'
                           ``extensions/gating.ts``); loaded with
                           ``--no-extensions --extension <path>`` when set,
                           off otherwise. The Claude path does not gate bash
                           either.

A missing binary raises :class:`~multiplai_core.agent_runner.AgentRunError`
with an install hint. It never falls back to Claude. A spawn that fails, and a
prompt Pi rejects (``response`` with ``success: false``), raise it too, at once.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any

from ..agent_runner import (
    AgentRunError,
    AgentRunResult,
    AgentRunTimeout,
    AgentUsage,
    ToolCall,
    _hook_session_dir,
    register_agent_backend,
)
from ..aio import hard_timeout
from .events import Event, Kind
from .rpc import PiRpc, SubprocessTransport

logger = logging.getLogger(__name__)

PROVIDER = "pi"
PI_BIN_ENV = "MULTIPLAI_PI_BIN"
GATE_EXT_ENV = "MULTIPLAI_PI_GATE_EXT"

# Claude Code tool names -> Pi tool names. Pi's -t flag is an allowlist that
# *removes* every unlisted tool, so translating the allow-list is also the deny
# mechanism (the complement simply does not exist for the run).
TOOL_MAP = {
    "Read": "read",
    "Write": "write",
    "Edit": "edit",
    "Bash": "bash",
    "Glob": "find",
    "Grep": "grep",
}
_CLAUDE_NAME = {pi: claude for claude, pi in TOOL_MAP.items()}

# Effort names map 1:1 onto pi --thinking levels (pi additionally has
# off/minimal below low).
THINKING_LEVELS = frozenset({"low", "medium", "high", "xhigh", "max"})

# How long to keep reading after `abort` before giving up on Pi settling.
ABORT_GRACE_S = 10.0

# How long writing `abort` may take. A Pi that has stopped reading stdin never
# drains the pipe, so the write is bounded or timeout_s would not be a limit.
ABORT_WRITE_TIMEOUT_S = 5.0

_INSTALL_HINT = (
    "Install the Pi coding agent so `pi` is on PATH and configure the model in "
    f"~/.pi/agent/models.json, or set {PI_BIN_ENV} to the binary's path."
)


class PiSpecError(ValueError):
    """A ``pi:`` model spec that does not have the ``pi:<model-id>@<provider>`` shape."""


def parse_pi_spec(spec: str) -> tuple[str, str]:
    """``pi:<model-id>@<provider>`` -> ``(model_id, provider)``.

    The provider suffix is mandatory, and the split is on the last ``@`` so a
    model ID may itself contain one.
    """
    if not spec.startswith(f"{PROVIDER}:"):
        raise PiSpecError(f"not a pi model spec: {spec!r}")
    model, sep, provider = spec[len(PROVIDER) + 1:].rpartition("@")
    if not sep or not model.strip() or not provider.strip():
        raise PiSpecError(
            f"malformed pi model spec {spec!r}: expected 'pi:<model-id>@<provider>'"
        )
    return model.strip(), provider.strip()


def pi_tools(allowed_tools: list[str] | None, disallowed_tools: list[str] | None = None) -> list[str]:
    """Translate a Claude Code tool allow-list into Pi tool names, de-duplicated."""
    denied = set(disallowed_tools or ())
    tools: list[str] = []
    for name in allowed_tools or ():
        if name in denied:
            continue
        mapped = TOOL_MAP.get(name)
        if mapped is None:
            logger.warning("pi backend: no Pi equivalent for tool %r; dropped", name)
            continue
        if mapped not in tools:
            tools.append(mapped)
    return tools


def build_argv(
    pi_bin: str,
    model: str,
    provider: str,
    *,
    tools: list[str],
    effort: str | None,
    thinking: dict | None = None,
    gate_ext: str | None = None,
) -> list[str]:
    """The ``pi --mode rpc`` command line for one run."""
    argv = [pi_bin, "--mode", "rpc", "--no-session",
            "--provider", provider, "--model", model]
    argv += ["-t", ",".join(tools)] if tools else ["--no-tools"]
    if effort:
        if effort in THINKING_LEVELS:
            argv += ["--thinking", effort]
        else:
            logger.warning("pi backend: effort %r has no Pi --thinking level; omitted", effort)
    elif isinstance(thinking, dict) and thinking.get("type") == "disabled":
        argv += ["--thinking", "off"]
    if gate_ext:
        argv += ["--no-extensions", "--extension", gate_ext]
    return argv


# ------------------------------------------------------------- the run loop


@dataclass
class PiRunOutcome:
    """What one Pi run produced, kept even when the run is cut short."""

    events: list[Event] = field(default_factory=list)
    turns: int = 0
    settled: bool = False
    """Nothing more is coming for this prompt: Pi said ``agent_settled``, or it
    answered ``prompt`` with disposition ``"handled"`` and started no run."""
    settled_aborted: bool | None = None
    """The ``aborted`` flag of ``agent_settled`` (pi 1.1.0+): ``True`` when the
    run ended because it was aborted. ``None`` when Pi sent no flag."""
    aborted_reason: str = ""
    """``"max_turns"`` or ``"timeout"`` when this side ended the run."""
    pipe_error: str = ""
    """Writing to Pi failed (it exited, so its stdin is closed); empty otherwise."""
    rejected: str = ""
    """Pi's error text when it answered ``prompt`` with ``success: false``."""


async def _send_abort(rpc: Any, out: PiRunOutcome) -> None:
    """Send ``abort``, bounded by :data:`ABORT_WRITE_TIMEOUT_S`.

    A failed write is recorded in *out* and a wedged one is given up on; the
    caller closes the process either way.
    """
    try:
        await hard_timeout(rpc.abort(), ABORT_WRITE_TIMEOUT_S)
    except TimeoutError:
        logger.warning("pi backend: abort write did not finish in %.0fs; closing pi",
                       ABORT_WRITE_TIMEOUT_S)
    except OSError as e:
        out.pipe_error = out.pipe_error or f"{type(e).__name__}: {e}"


async def _drive(rpc: Any, prompt: str, out: PiRunOutcome, max_turns: int) -> None:
    """Send *prompt* and read events into *out* until Pi settles or the stream ends.

    ``pi --mode rpc`` is a long-lived server: its stdout does not close when a
    run is over, so ``agent_settled``, not EOF, ends a run. Reaching
    *max_turns* sends ``abort`` and keeps reading for :data:`ABORT_GRACE_S`, so
    the events already in flight still land in the result. A write that fails
    because Pi has exited is recorded in ``out.pipe_error``, not raised. A
    ``response`` to ``prompt`` with ``success: false`` means Pi rejected the
    prompt and starts no run (pi docs/rpc.md), so it ends the run too, with
    Pi's error text in ``out.rejected``. A successful one with
    ``data.disposition == "handled"`` means an extension consumed the prompt:
    no run starts and no ``agent_settled`` follows (pi 1.2.0 docs/rpc.md, "Run
    lifecycle"), so it ends the run as settled, with no turns.
    """
    try:
        await rpc.prompt(prompt)
    except OSError as e:  # BrokenPipeError / ConnectionResetError: Pi exited
        out.pipe_error = f"{type(e).__name__}: {e}"
        return
    stream = rpc.events()
    deadline: float | None = None
    loop = asyncio.get_running_loop()
    while True:
        try:
            if deadline is None:
                event = await anext(stream)
            else:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return
                event = await asyncio.wait_for(anext(stream), timeout=remaining)
        except (TimeoutError, StopAsyncIteration):
            return
        out.events.append(event)
        if event.kind is Kind.TURN_END:
            out.turns += 1
        if event.kind is Kind.AGENT_SETTLED:
            out.settled = True
            aborted = event.raw.get("aborted")
            if isinstance(aborted, bool):
                out.settled_aborted = aborted
            return
        if event.raw_type == "response" and event.raw.get("command") == "prompt":
            if event.raw.get("success") is False:
                out.rejected = str(event.raw.get("error") or "prompt rejected")
                return
            data = event.raw.get("data")
            if isinstance(data, dict) and data.get("disposition") == "handled":
                out.settled = True
                return
        if deadline is None and max_turns and out.turns >= max_turns:
            out.aborted_reason = "max_turns"
            await _send_abort(rpc, out)
            deadline = loop.time() + ABORT_GRACE_S


async def supervise(rpc: Any, prompt: str, *, max_turns: int, timeout_s: float) -> PiRunOutcome:
    """Run one prompt on *rpc* within *max_turns* and *timeout_s*.

    The timeout does not depend on Pi sending events: a model that hangs
    mid-tool-call is cut off on time. It is a hard limit: neither the cut nor
    the ``abort`` that follows waits on a Pi that has stopped reading stdin, so
    this returns within *timeout_s* plus :data:`ABORT_WRITE_TIMEOUT_S`.
    Everything read before the cut stays in the returned outcome.
    """
    out = PiRunOutcome()
    try:
        await hard_timeout(_drive(rpc, prompt, out, max_turns), timeout_s)
    except TimeoutError:
        out.aborted_reason = "timeout"
        await _send_abort(rpc, out)
    return out


# ------------------------------------------------------------------ results


def _assistant_messages(events: list[Event]) -> list[dict]:
    """The final rendition of every assistant message (its ``message_end``)."""
    return [
        ev.raw["message"] for ev in events
        if ev.raw_type == "message_end"
        and isinstance(ev.raw.get("message"), dict)
        and ev.raw["message"].get("role") == "assistant"
    ]


def assistant_text(events: list[Event]) -> str:
    """Concatenated assistant text across the run.

    Pi repeats the same message id across message_start/update/end with
    growing content, so per id the last rendition wins; ids join in first-seen
    order.
    """
    by_id: dict[str, str] = {}
    for ev in events:
        msg = ev.raw.get("message")
        if not (isinstance(msg, dict) and msg.get("role") == "assistant"):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        text = "\n".join(
            c.get("text", "")
            for c in content
            if isinstance(c, dict) and c.get("type") == "text" and c.get("text")
        )
        if not text:
            continue
        key = str(ev.raw.get("id") or msg.get("responseId") or msg.get("timestamp") or ev.seq)
        by_id[key] = text
    return "\n".join(by_id.values()).strip()


def usage_from_events(events: list[Event]) -> tuple[AgentUsage, bool]:
    """Sum the usage of every assistant message; ``(usage, cost_reported)``.

    Each assistant message is counted once, from its ``message_end``.
    ``cost_reported`` is False when no message carried a cost figure, which is
    different from a reported cost of zero only in name: either way
    ``cost_usd`` is 0.
    """
    tin = tout = cr = cw = 0
    cost = 0.0
    cost_reported = False
    for msg in _assistant_messages(events):
        usage = msg.get("usage")
        if not isinstance(usage, dict):
            continue
        tin += int(usage.get("input") or 0)
        tout += int(usage.get("output") or 0)
        cr += int(usage.get("cacheRead") or 0)
        cw += int(usage.get("cacheWrite") or 0)
        c = usage.get("cost")
        total = c.get("total") if isinstance(c, dict) else c
        if isinstance(total, (int, float)) and total > 0:
            cost += float(total)
            cost_reported = True
    return AgentUsage(input_tokens=tin, output_tokens=tout,
                      cache_creation_tokens=cw, cache_read_tokens=cr,
                      cost_usd=cost), cost_reported


# Pi argument name -> Claude Code argument name, per Pi tool (schemas in
# pi-mono packages/coding-agent/src/core/tools/*.ts). A key with no Claude Code
# equivalent (find/grep ``limit`` aside, grep ``literal``) keeps Pi's name.
_CLAUDE_ARGS = {
    "read": {"path": "file_path"},
    "write": {"path": "file_path"},
    "edit": {"path": "file_path", "oldText": "old_string", "newText": "new_string"},
    "grep": {"ignoreCase": "-i", "context": "-C", "limit": "head_limit"},
}
_EDIT_KEYS = {"oldText": "old_string", "newText": "new_string"}


def claude_tool_input(pi_tool: str, args: dict) -> dict:
    """Pi tool arguments under Claude Code's argument names.

    Pi's edit takes ``edits: [{oldText, newText}, ...]``: one edit becomes
    ``old_string``/``new_string`` (Claude Code's Edit), several keep the
    ``edits`` list with each entry's keys renamed. Pi's bash ``timeout`` is in
    seconds and Claude Code's in milliseconds, so the value is converted.
    """
    renames = _CLAUDE_ARGS.get(pi_tool, {})
    out = {renames.get(k, k): v for k, v in args.items()}
    if pi_tool == "edit" and isinstance(out.get("edits"), list):
        edits = [{_EDIT_KEYS.get(k, k): v for k, v in e.items()} if isinstance(e, dict) else e
                 for e in out["edits"]]
        if len(edits) == 1 and isinstance(edits[0], dict) and "old_string" not in out:
            del out["edits"]
            out.update(edits[0])
        else:
            out["edits"] = edits
    if pi_tool == "bash" and isinstance(out.get("timeout"), (int, float)) \
            and not isinstance(out["timeout"], bool):
        out["timeout"] = out["timeout"] * 1000
    return out


def tool_calls_from_events(events: list[Event]) -> tuple[ToolCall, ...]:
    """Every tool call Pi started, in order, with Claude Code tool and argument
    names where one exists (an unmapped tool keeps Pi's name and arguments)."""
    calls: list[ToolCall] = []
    for ev in events:
        if ev.kind is not Kind.TOOL_START or not ev.tool_name:
            continue
        if ev.tool_name in _CLAUDE_NAME:
            calls.append(ToolCall(name=_CLAUDE_NAME[ev.tool_name],
                                  input=claude_tool_input(ev.tool_name, dict(ev.tool_input))))
        else:
            calls.append(ToolCall(name=ev.tool_name, input=dict(ev.tool_input)))
    return tuple(calls)


def final_error(events: list[Event]) -> str | None:
    """Pi's error text when the run's last assistant message ended in error.

    Pi retries a failed model call itself (``auto_retry_*``); when every retry
    fails, the run still settles normally and the last assistant message has
    ``stopReason: "error"``. Returning that run as a success would hand the
    caller empty text with no reason.
    """
    messages = _assistant_messages(events)
    if not messages or messages[-1].get("stopReason") != "error":
        return None
    return str(messages[-1].get("errorMessage") or "model call failed")


_WRITE_TOOLS = {"write", "edit"}


def files_changed(events: list[Event]) -> list[str]:
    """Files the run changed: the path of every write/edit tool call, in order.

    The same contract as the SDK path (``file_path`` args of Write/Edit), so a
    path is reported as the agent passed it. Files written another way (the
    bash tool) are not reported on either path. There is deliberately no
    ``git status`` fallback: it reports paths relative to the repo root rather
    than the run's cwd, covers the whole repo, and so attributes changes made
    by other processes during the run to the agent.
    """
    files: list[str] = []
    for ev in events:
        if ev.kind not in (Kind.TOOL_START, Kind.TOOL_END):
            continue
        if (ev.tool_name or "").lower() not in _WRITE_TOOLS:
            continue
        if ev.target and ev.target not in files:
            files.append(ev.target)
    return files


# --------------------------------------------------------------- the backend


async def _open_transport(argv: list[str], *, cwd: str, env: dict[str, str]) -> Any:
    """Start ``pi --mode rpc``. Tests replace this with an in-process stub.

    Raises :class:`AgentRunError` when Pi cannot be started: the binary is not
    on PATH, or the spawn itself fails (a cwd that does not exist, a file exec
    rejects).
    """
    if shutil.which(argv[0]) is None:
        raise AgentRunError(
            f"pi binary {argv[0]!r} not found. {_INSTALL_HINT}",
            reason="pi binary not found",
        )
    try:
        return await SubprocessTransport.spawn(argv, cwd=cwd, env=env)
    except OSError as e:
        raise AgentRunError(
            f"pi failed to start ({type(e).__name__}: {e}); argv[0]={argv[0]!r} cwd={cwd!r}",
            reason=f"pi failed to start: {type(e).__name__}",
        ) from e


def _record_cost(component: str, label: str, spec: str,
                 usage: AgentUsage, cost_reported: bool) -> None:
    """One cost-ledger record for the run; never raises."""
    if not component or not any((usage.input_tokens, usage.output_tokens,
                                 usage.cache_creation_tokens, usage.cache_read_tokens)):
        return
    try:
        import uuid
        from datetime import datetime

        from ..costing import TokenCounts, append_records, build_record

        record = build_record(
            ts=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            source="pi",
            session=f"pi-{label}",
            model=spec,
            msg_id=f"run-{uuid.uuid4().hex[:12]}",
            tokens=TokenCounts(input=usage.input_tokens, output=usage.output_tokens,
                               cw5m=usage.cache_creation_tokens,
                               cr=usage.cache_read_tokens),
            component=component,
            # Always explicit: the token-derived price is for Claude models and
            # would invent a figure for this one.
            cost_usd=usage.cost_usd,
        )
        if not cost_reported:
            record["cost_unreported"] = True
        append_records([record])
    except Exception:  # the ledger is never worth failing a run
        logger.warning("pi run [%s]: cost ledger write failed (ignored)", label, exc_info=True)


async def _attempt(argv: list[str], prompt: str, *, workdir: str, env: dict[str, str],
                   max_turns: int, timeout_s: float, label: str) -> tuple[PiRunOutcome, str]:
    transport = await _open_transport(argv, cwd=workdir, env=env)
    try:
        outcome = await supervise(PiRpc(transport), prompt,
                                  max_turns=max_turns, timeout_s=timeout_s)
    finally:
        try:
            await transport.close()
        except Exception:
            logger.warning("pi run [%s]: closing pi failed (ignored)", label, exc_info=True)
    tail_fn = getattr(transport, "stderr_tail", None)
    tail = tail_fn() if callable(tail_fn) else ""
    return outcome, tail


async def run_pi_agent(
    prompt: str,
    *,
    system_prompt: str | None = None,
    allowed_tools: list[str] | None = None,
    disallowed_tools: list[str] | None = None,
    max_turns: int = 1,
    model: str | None = None,
    effort: str | None = None,
    thinking: dict | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout_s: float = 600.0,
    max_attempts: int = 1,
    retry_backoff_s: float = 1.5,
    prompt_file_fallback: bool | None = None,
    label: str = "agent",
    component: str = "",
    read_roots: Sequence[str | Path] | None = None,
) -> AgentRunResult:
    """``run_agent`` for a ``pi:<model-id>@<provider>`` spec. See the module docstring."""
    del prompt_file_fallback  # no argv limit on this path (module docstring)
    if read_roots is not None:
        logger.warning("run_pi_agent [%s]: read_roots is not enforced on the pi "
                       "backend; reads are not limited to %s", label, list(read_roots))
    del read_roots  # not enforceable on this path (module docstring)
    spec = model or ""
    try:
        pi_model, pi_provider = parse_pi_spec(spec)
    except PiSpecError as e:
        raise AgentRunError(str(e), reason="malformed pi model spec") from e

    # Same default as the SDK path: an isolated directory, so a run with no
    # cwd neither edits nor picks up context files from the caller's project.
    workdir = str(Path(cwd).resolve()) if cwd else str(_hook_session_dir())
    run_env = {**os.environ, **(env or {}), "_HOOK_CHILD_SESSION": "1"}
    gate_ext = os.environ.get(GATE_EXT_ENV, "").strip() or None
    if gate_ext:
        if not Path(gate_ext).is_file():
            raise AgentRunError(f"{GATE_EXT_ENV}={gate_ext!r} is not a file",
                                reason="pi gating extension not found")
        run_env.setdefault("PIHARNESS_WORKSPACE", workdir)
    argv = build_argv(
        os.environ.get(PI_BIN_ENV, "").strip() or "pi", pi_model, pi_provider,
        tools=pi_tools(allowed_tools, disallowed_tools), effort=effort,
        thinking=thinking, gate_ext=gate_ext,
    )
    message = f"{system_prompt.strip()}\n\n---\n\n{prompt}" if system_prompt else prompt

    logger.info("START pi run [%s] model=%s tools=%s max_turns=%d timeout=%.0fs attempts=%d",
                label, spec, argv[argv.index("-t") + 1] if "-t" in argv else "none",
                max_turns, timeout_s, max_attempts)
    started = time.monotonic()
    last: AgentRunError | None = None
    for attempt in range(max(1, max_attempts)):
        if attempt:
            assert last is not None
            logger.warning("pi run [%s] %s (attempt %d/%d), retrying in %.1fs",
                           label, last.reason, attempt, max_attempts, retry_backoff_s)
            await asyncio.sleep(retry_backoff_s)
        try:
            outcome, stderr_tail = await _attempt(
                argv, message, workdir=workdir, env=run_env,
                max_turns=max_turns, timeout_s=timeout_s, label=label)
        except AgentRunError as e:  # pi did not start (_open_transport)
            e.attempts = attempt + 1
            last = e
            continue
        usage, cost_reported = usage_from_events(outcome.events)
        _record_cost(component, label, spec, usage, cost_reported)
        result = AgentRunResult(
            text=assistant_text(outcome.events),
            turns=outcome.turns,
            usage=usage,
            files_changed=files_changed(outcome.events),
            stderr_tail="",
            tool_calls=tool_calls_from_events(outcome.events),
        )
        error = final_error(outcome.events)
        if outcome.aborted_reason == "timeout":
            last = AgentRunTimeout(
                f"pi run [{label}] timed out after {timeout_s:.0f}s",
                reason=f"timed out after {timeout_s:.0f}s", attempts=attempt + 1,
                stderr_tail=stderr_tail, partial=result)
        elif outcome.rejected:
            last = AgentRunError(
                f"pi run [{label}] pi rejected the prompt: {outcome.rejected[:500]}",
                reason=f"prompt rejected: {outcome.rejected[:200]}", attempts=attempt + 1,
                stderr_tail=stderr_tail, partial=result)
        elif error is not None:
            last = AgentRunError(
                f"pi run [{label}] failed: {error[:500]}",
                reason=f"failed: {error[:200]}", attempts=attempt + 1,
                stderr_tail=stderr_tail, partial=result)
        elif outcome.pipe_error and not outcome.settled:
            last = AgentRunError(
                f"pi run [{label}] lost the pipe to pi ({outcome.pipe_error}); "
                "pi exited early, see stderr_tail",
                reason="pi exited (write to pi failed)", attempts=attempt + 1,
                stderr_tail=stderr_tail, partial=result)
        elif not outcome.settled and outcome.aborted_reason != "max_turns":
            last = AgentRunError(
                f"pi run [{label}] ended before Pi settled (pi exited?)",
                reason="pi exited before the run settled", attempts=attempt + 1,
                stderr_tail=stderr_tail, partial=result)
        else:
            logger.info("DONE pi run [%s] turns=%d text=%d chars files=%d cost=$%.4f "
                        "elapsed=%.1fs%s", label, result.turns, len(result.text),
                        len(result.files_changed), usage.cost_usd,
                        time.monotonic() - started,
                        " (stopped at max_turns)" if outcome.aborted_reason else "")
            return result
    assert last is not None
    logger.warning("FAIL pi run [%s] %s after %d attempt(s) elapsed=%.1fs",
                   label, last.reason, max_attempts, time.monotonic() - started)
    raise last


register_agent_backend(PROVIDER, run_pi_agent)

