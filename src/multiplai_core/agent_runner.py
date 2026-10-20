"""Single SDK agent runner — the one place every background agent invocation goes.

``run_agent()`` wraps ``claude_agent_sdk.query()`` with the machinery that the
model client, buildme, and deep-research each used to reimplement privately:
subprocess isolation flags, unknown-message skipping, hard timeout with
generator teardown, in-memory stderr capture with a compact failure tail,
opt-in retry, and a normalized result carrying text, turns, usage, and files
changed.

Skill-specific policy stays in the skills: trust gating (buildme), usage
accumulation across a run (deep-research), Pydantic structured-output
validation (both), and each caller's public error taxonomy — callers catch
``AgentRunError``/``AgentRunTimeout`` and re-raise their own types.

Every invocation always gets the isolation/hardening bundle:

- ``permission_mode="bypassPermissions"`` — these are unattended subprocesses;
  callers that need a safety boundary gate BEFORE calling (buildme's
  ``--trust-repo``) or constrain tools (``allowed_tools``/``disallowed_tools``).
- ``setting_sources=[]`` + ``extra_args={"setting-sources": ""}`` — the
  second is deliberately redundant on the pinned SDK (belt-and-braces; see
  the inline comment). Without the isolation the child inherits parent
  settings/hooks and spawns runaway subagents (verified 2026-04-20).
- ``debug-to-stderr`` — forces the CLI to emit diagnosable stderr (the SDK
  hardcodes ProcessError stderr to "Check stderr output for details").
  ``_safe_query`` is mandatory while this is on: the CLI emits internal
  message types the bundled SDK parser doesn't recognize.
- ``strict-mcp-config`` — isolates the subprocess from account-level MCP
  integrations; without it the nested CLI attempts non-interactive OAuth and
  exits 1 with no usable stderr (verified root cause 2026-05-19).
- ``env["_HOOK_CHILD_SESSION"]="1"`` — any hook that still loads skips work.
- default ``cwd`` = ``$CLAUDE_CONFIG_DIR/hook-sessions`` — prevents project
  settings.json pickup and keeps child sessions out of the user's history.

``read_roots`` adds one more boundary, on request: a ``PreToolUse`` hook that
denies ``Read``/``Grep``/``Glob``/``LS``/``NotebookRead`` calls whose path
resolves outside the given folders. A hook's deny holds under
``bypassPermissions`` (verified 2026-10-16 with a real call on
claude-agent-sdk 0.2.139).

Every run also logs an ``alive`` heartbeat at INFO while an attempt is in
flight, every ``MULTIPLAI_AGENT_HEARTBEAT_S`` seconds (default 60; ``0`` or
negative disables it). Set it to ``0`` if a caller's log must stay quiet.

Every attempt also writes the CLI's debug stderr to a file under
``<logs_dir>/sdk/`` as the lines arrive, so the file survives the calling
process being killed. ``MULTIPLAI_SDK_DEBUG_KEEP_S`` set
to a positive number deletes a successful attempt's file when the attempt was
faster than that many seconds; ``MULTIPLAI_SDK_DEBUG_LOG=off`` turns the file
off. Files older than the log retention (``retention_days()``, 365 days by
default) are deleted; ``MULTIPLAI_LOG_RETENTION_DAYS=0`` keeps them all.

``model`` may name a backend other than the Claude Agent SDK, as a
``provider:model`` spec (``pi:deepseek/deepseek-v4-flash@fireworks``). An
Anthropic spec (a bare ID, ``anthropic:<id>`` or ``claude:<id>``) runs on the
SDK path above with the provider prefix removed; any other provider goes to the
runner registered for it with :func:`register_agent_backend`. The ``pi``
backend registers itself the first time a ``pi:`` spec is run, so importing
this module never imports :mod:`multiplai_core.pi`.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import itertools
import logging
import os
import re
import time
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Awaitable, Callable, Sequence

from .aio import hard_timeout
from .env import DEFAULT_PROVIDER, parse_model_spec

logger = logging.getLogger(__name__)

# A size hint for callers that split work across several calls (deep-research's
# synthesis does). It is not an OS limit: the SDK writes every prompt to the
# CLI's stdin as a stream-json message, so ``run_agent`` sends any size inline.
MAX_PROMPT_BYTES = 80_000

# CLI stderr capture (debug-to-stderr is on, so raw volume is large). Kept in
# memory only, never on disk; summarized into a compact tail only on failure:
# the [ERROR] lines (which carry the real cause — rate limit, auth, crash), or
# a few trailing raw lines if there were none.
_STDERR_RING_LINES = 50          # trailing context kept for the fallback tail
_STDERR_MAX_ERROR_LINES = 15     # [ERROR] lines surfaced in the summary
_STDERR_ERROR_CAPTURE_CAP = 200  # absolute cap on captured [ERROR] lines

# Liveness heartbeat. Between START and DONE a run emits nothing, so a
# multi-minute call (dream's consolidation pass runs 5-30 min) is
# indistinguishable from a wedged one in the log — the operator's only signal
# is the DONE/FAIL line that may be half an hour away. The heartbeat logs
# elapsed time plus the output accumulated so far, which also distinguishes
# "slow but producing" from "stalled with nothing".
_HEARTBEAT_ENV = "MULTIPLAI_AGENT_HEARTBEAT_S"
_HEARTBEAT_DEFAULT_S = 60.0

# CLI debug stderr on disk, per attempt. The in-memory capture above is only
# summarized when an attempt ends, so a caller that is killed mid-call (a hook
# at its harness ceiling) lost every line: on 2026-09-27 a memory-router call
# stalled for 26s and the only record left was the hook's HOOK_ENTRY line. The
# file is written line by line and flushed, so it outlives a SIGKILL of this
# process, and it holds the CLI's own record of the call: auth check, request
# dispatch with its x-client-request-id, first stream chunk, retries.
_DEBUG_LOG_ENV = "MULTIPLAI_SDK_DEBUG_LOG"      # "off" disables the file
_DEBUG_KEEP_ENV = "MULTIPLAI_SDK_DEBUG_KEEP_S"  # >0: delete faster successes
_DEBUG_KEEP_DEFAULT_S = 0.0  # keep every file
_DEBUG_LOG_SUBDIR = "sdk"
_debug_logs_swept = False
# Per-process sequence number in each file name. Timestamp, label, pid and
# attempt alone collide when two calls with the same label start in the same
# second, and the fast one's cleanup (when MULTIPLAI_SDK_DEBUG_KEEP_S is set)
# would then delete the slow one's file.
_debug_log_seq = itertools.count(1)

# Under permission_mode="bypassPermissions" ``allowed_tools`` is only an
# allow-list: it adds nothing to the deny side, so every default tool stays
# present AND auto-approved. Callers routinely feed UNTRUSTED text (fetched web
# pages, emails, logs) through run_agent, and an injected instruction in that
# text would otherwise get an auto-approved Read → WebFetch to exfiltrate local
# secrets, or a Bash to do worse.
#
# There are two ways to shut that off, and this module uses BOTH:
#
# 1. ``tools`` (the CLI's ``--tools``) sets the *base set* — the tools that
#    exist at all. It replaces the built-in set rather than adding to it, so
#    ``tools=[]`` is "no tools" with no list to keep up to date. This is the
#    real guarantee, and it is why TOOL_UNIVERSE going stale is no longer a
#    security question.
# 2. ``disallowed_tools`` — the complement of the allow-list, below. Belt and
#    braces: it is what still holds if a CLI ever ignores ``--tools``, and it
#    is the only lever on a tool that is somehow present anyway.
#
# TOOL_UNIVERSE is therefore a best-effort enumeration, not a safety floor.
# Treat it as such: adding a name makes the second layer tighter, forgetting
# one no longer opens a hole. Derived 2026-08-06 from the CLI's own generated
# schema list (``@anthropic-ai/claude-code/sdk-tools.d.ts``, CLI 2.1.221) plus
# the harness-only names that file does not carry (Skill, ToolSearch,
# SlashCommand, BashOutput, KillShell, Task). Re-derive it on a CLI bump:
#
#   grep -oE '^export (type|interface) [A-Za-z]+Input' sdk-tools.d.ts \
#     | sed 's/.* //;s/Input$//' | sort -u
#
# (That file names schemas, not tools, so FileRead/FileEdit/FileWrite appear
# there as the schemas behind Read/Edit/Write.)
TOOL_UNIVERSE = (
    # mutation / execution
    "Bash", "BashOutput", "KillShell", "Edit", "Write", "NotebookEdit",
    "MultiEdit", "REPL", "Task", "Agent", "AskUserQuestion", "SlashCommand",
    "ExitPlanMode", "EnterPlanMode", "TodoWrite",
    # read / network / meta — closes the prompt-injection exfiltration chain
    # (untrusted input steering an auto-approved Read → WebFetch of a secret)
    "Read", "NotebookRead", "Grep", "Glob", "LS", "WebFetch", "WebSearch",
    "ToolSearch", "Skill",
    # egress: each of these can carry text off the machine, which makes them
    # exfiltration channels every bit as much as WebFetch
    "Artifact", "SendMessage", "PushNotification", "RemoteTrigger",
    "SendFeedback",
    # background / scheduled execution — a denied Bash is worth little if the
    # run can queue work that gets one later
    "TaskCreate", "TaskGet", "TaskList", "TaskUpdate", "TaskStop",
    "TaskOutput", "Workflow", "ScheduleWakeup", "Monitor",
    "CronCreate", "CronDelete", "CronList",
    # MCP: separately contained by strict-mcp-config + setting_sources=[] (see
    # the options below), listed so the deny-list does not depend on that.
    "Mcp", "ListMcpResources", "ReadMcpResource", "ReadMcpResourceDir",
    "RefreshMcpTools",
    # remaining harness surface
    "EnterWorktree", "ExitWorktree", "ReportFindings", "ProposeSkills",
    "Projects", "ClaudeDesign", "ShowOnboardingRolePicker",
)


def deny_list(allowed_tools: list[str] | None) -> list[str]:
    """Return the deny-list complement of ``allowed_tools``.

    With no allowed tools this is the full ``TOOL_UNIVERSE`` — the fail-closed
    default. With an allow-list it is everything else, so opening ``WebFetch``
    opens exactly ``WebFetch`` and nothing that rides along with it.

    This is the *second* of the two layers described above. The first —
    ``tools``, the base set — is what makes the guarantee independent of this
    list being complete.
    """
    opened = set(allowed_tools or ())
    return [t for t in TOOL_UNIVERSE if t not in opened]


# The read tools ``read_roots`` limits, as the PreToolUse matcher spells them.
_READ_ROOT_MATCHER = "Read|Grep|Glob|LS|NotebookRead"
_PATH_KEYS = ("file_path", "path", "notebook_path")
_GLOB_MAGIC = re.compile(r"[*?\[{]")


def _glob_base(pattern: str) -> str:
    """The leading part of a glob pattern that holds no wildcard."""
    parts = pattern.split("/")
    for i, part in enumerate(parts):
        if _GLOB_MAGIC.search(part):
            parts = parts[:i]
            break
    base = "/".join(parts)
    return base or ("/" if pattern.startswith("/") else "")


def path_allowed(
    tool_name: str,
    tool_input: dict[str, Any],
    roots: Sequence[str | Path],
    cwd: str | Path,
) -> bool:
    """Whether a read tool call stays inside ``roots``.

    Reads ``file_path``, ``path`` and ``notebook_path``, and for ``Glob`` the
    part of ``pattern`` before its first wildcard. Relative paths resolve
    against ``cwd`` (a ``Glob`` pattern against its ``path``, if given);
    ``~`` is expanded and symlinks are followed with ``os.path.realpath``. A
    call with no path means ``cwd``. Only paths inside a resolved root pass;
    a value that is not a string fails.
    """
    base = os.path.realpath(os.fspath(cwd))
    resolved_roots = [os.path.realpath(os.fspath(r)) for r in roots]

    def inside(raw: object, against: str) -> bool:
        if not isinstance(raw, str):
            return False
        target = os.path.realpath(os.path.join(against, os.path.expanduser(raw)))
        return any(
            target == root or target.startswith(root.rstrip(os.sep) + os.sep)
            for root in resolved_roots
        )

    given = [tool_input[k] for k in _PATH_KEYS if tool_input.get(k) is not None]
    if not all(inside(raw, base) for raw in given or [base]):
        return False
    if tool_name == "Glob" and "pattern" in tool_input:
        pattern = tool_input["pattern"]
        if not isinstance(pattern, str):
            return False
        search = tool_input.get("path")
        search_dir = (os.path.join(base, os.path.expanduser(search))
                      if isinstance(search, str) else base)
        return inside(_glob_base(os.path.expanduser(pattern)), search_dir)
    return True


def _deny_outside(
    roots: Sequence[str | Path], cwd: str | Path, label: str,
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """A PreToolUse hook that denies a read tool call outside ``roots``."""

    async def _hook(input_data: dict[str, Any], tool_use_id: Any, context: Any) -> dict[str, Any]:
        name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input") or {}
        if path_allowed(name, tool_input, roots, cwd):
            return {}
        logger.warning(
            "run_agent [%s]: denied %s outside read_roots: %s",
            label, name, {k: tool_input.get(k) for k in (*_PATH_KEYS, "pattern") if k in tool_input},
        )
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"{name} is limited to these folders: "
                    + ", ".join(os.fspath(r) for r in roots)
                ),
            }
        }

    return _hook


def _env_float(name: str, default: float) -> float:
    """Parse a float env var, falling back to the default on garbage.

    The single definition — ``model_client`` imports it from here (the
    client is built on the runner, so this direction cannot cycle).

    This module reads it at **call** time, not import time, so a caller or
    test can set the knob per run.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using default %s", name, raw, default)
        return default


@dataclass(frozen=True)
class AgentUsage:
    """Token/cost metrics reported by the SDK's ResultMessage (zeros if the
    stream ended without one — e.g. failure mid-run or an old SDK)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class ToolCall:
    """One tool call an agent made: the tool's name and a copy of its input.

    Holds only what the agent asked for, never the tool's result.
    """

    name: str
    input: dict


@dataclass(frozen=True)
class AgentRunResult:
    """Normalized result of one agent run."""

    text: str                    # concatenated assistant text, stripped
    turns: int                   # number of AssistantMessages consumed
    usage: AgentUsage
    files_changed: list[str]     # file_path args of Write/Edit ToolUseBlocks
    stderr_tail: str             # compact failure context ("" on success)
    # Every ToolUseBlock in the order it arrived, Write/Edit included. Last and
    # defaulted so callers that build an AgentRunResult keep working.
    tool_calls: tuple[ToolCall, ...] = ()


class AgentRunError(RuntimeError):
    """Raised when an agent run fails beyond the retry budget.

    ``stderr_tail`` holds the compact CLI stderr summary; ``reason`` a short
    human cause ("timed out after 600s" / "failed: <exc>"); ``attempts`` how
    many attempts were made; ``partial`` the result assembled up to the
    failure point (text/turns/files/usage), for callers that degrade to
    partial output instead of failing hard (buildme's agent_call).
    ``debug_log_path`` is the last attempt's full CLI debug log on disk
    (``""`` when the file was off or could not be written).
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str = "",
        attempts: int = 1,
        stderr_tail: str = "",
        partial: AgentRunResult | None = None,
        debug_log_path: str = "",
    ) -> None:
        self.reason = reason or message
        self.attempts = attempts
        self.stderr_tail = stderr_tail
        self.partial = partial
        self.debug_log_path = debug_log_path
        parts = [message]
        if stderr_tail:
            parts.append("--- captured CLI stderr (errors) ---")
            parts.append(stderr_tail)
            parts.append("--- end stderr ---")
        super().__init__("\n".join(parts))


class AgentRunTimeout(AgentRunError):
    """The run exceeded ``timeout_s`` on every attempt."""


class UnknownAgentBackendError(AgentRunError):
    """``model`` names a provider no agent backend is registered for."""


# The CLI's result text for a hit usage limit, e.g. "You've hit your session
# limit · resets 9:40am (UTC)". The CLI (bundled with claude-agent-sdk
# 0.2.168) tells these apart with `startsWith("You've hit your ...")`, so the
# match is anchored the same way: at the start of the text, after at most the
# SDK's "Claude Code returned an error result:" prefix.
_USAGE_LIMIT_RE = re.compile(
    r"\A(?:[^\n:]*error result:\s*)?you(?:'|\u2019)ve hit your (?P<limit>[^\n·]{0,60}?)\s*(?:limit|budget)\b"
    r"(?:\s*·\s*resets\s+(?P<resets>[^()\n·]*(?:\((?!exit code)[^()\n]*\))?))?",
    re.IGNORECASE,
)


class UsageLimitError(AgentRunError):
    """The account's Claude usage limit (session, weekly, ...) was hit.

    Never retry it: the limit resets hours later, and every call until then
    fails the same way after spending a CLI start-up. :func:`run_agent` raises
    it on the first attempt that reports the limit, whatever ``max_attempts``
    says. A caller with its own retry loop must catch it before
    ``AgentRunError`` and stop, keeping whatever it can resume from.

    ``limit`` is the kind the CLI named, lower-cased: ``"session"`` and
    ``"weekly"`` are the common ones, but the set is open (``"opus"``,
    ``"monthly spend"``, ...), and ``""`` when it named none. ``resets`` is when
    it resets, as the CLI wrote it (``"9:40am (UTC)"``), or ``""`` when it did
    not say.
    """

    def __init__(self, message: str, *, limit: str = "", resets: str = "", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.limit = limit
        self.resets = resets


def _usage_limit_match(text: str) -> "re.Match[str] | None":
    return _USAGE_LIMIT_RE.search(text) if text else None


def is_usage_limit(error: BaseException | str) -> bool:
    """Whether *error* is, or was caused by, a hit Claude usage limit.

    *error* is an exception or a message. A message counts when it starts
    with the CLI's limit text (``You've hit your ... limit``), alone or after
    the SDK's ``Claude Code returned an error result:`` prefix; the phrase
    anywhere else (a quote, a stderr line a wrapper appended) does not. An
    exception counts when it, or one reached from it the way Python's
    traceback does (``__cause__``, and ``__context__`` unless suppressed by
    ``raise ... from None``), is a :class:`UsageLimitError`, or is not an
    ``AgentRunError`` and has such a message (the SDK's own error). This lets
    a caller that wraps ``AgentRunError`` in its own type
    (``model_client.SDKQueryError``, a skill's error class) still tell a usage
    limit from a failure worth retrying.
    """
    if isinstance(error, str):
        return _usage_limit_match(error) is not None
    seen: set[int] = set()
    todo: list[BaseException] = [error]
    while todo:
        e = todo.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        if isinstance(e, UsageLimitError):
            return True
        # run_agent raises UsageLimitError itself, so another AgentRunError's
        # text (which carries the CLI's stderr) is not read.
        if not isinstance(e, AgentRunError) and _usage_limit_match(str(e)):
            return True
        if e.__cause__ is not None:
            todo.append(e.__cause__)
        if e.__context__ is not None and not e.__suppress_context__:
            todo.append(e.__context__)
    return False


AgentBackend = Callable[..., Awaitable["AgentRunResult"]]
"""An agent backend: ``async (prompt, *, <run_agent's keywords>) -> AgentRunResult``."""

_AGENT_BACKENDS: dict[str, AgentBackend] = {}

# Backends that register themselves when their module is imported. They are
# imported on first use only, so `import multiplai_core.agent_runner` stays
# free of their code (and of anything they import).
_LAZY_AGENT_BACKENDS: dict[str, str] = {"pi": "multiplai_core.pi.backend"}

# A provider prefix is a plain lowercase word. Requiring it keeps a Bedrock
# model ID such as `anthropic.claude-3-5-sonnet-20240620-v1:0` (dots before the
# colon) on the SDK path, exactly as before backends existed.
_PROVIDER_RE = re.compile(r"[a-z][a-z0-9_-]*")
# A Bedrock ARN (`arn:aws:bedrock:…`) is also a model ID the SDK path takes.
_BARE_ID_PREFIXES = frozenset({"arn"})


def register_agent_backend(provider: str, runner: AgentBackend) -> None:
    """Make :func:`run_agent` send ``<provider>:<model>`` specs to *runner*.

    *runner* is called as ``await runner(prompt, **kwargs)`` with every keyword
    :func:`run_agent` takes, ``model`` holding the full spec with the provider
    read the way dispatch read it: lower-cased, whitespace stripped
    (``PI: deepseek/deepseek-v4-flash@fireworks`` arrives as
    ``pi:deepseek/deepseek-v4-flash@fireworks``). It must return an
    :class:`AgentRunResult`. Registering a provider again replaces its runner.
    The Anthropic provider cannot be replaced: it is the SDK path itself.
    """
    name = (provider or "").strip().lower()
    if not name:
        raise ValueError("agent backend provider name is empty")
    if name in (DEFAULT_PROVIDER, "claude"):
        raise ValueError(f"{name!r} is the built-in Claude Agent SDK backend")
    _AGENT_BACKENDS[name] = runner


def registered_agent_backends() -> tuple[str, ...]:
    """The providers :func:`run_agent` can dispatch to, sorted.

    Includes ``anthropic`` (the SDK path) and the built-in backends that
    register on first use (``pi``), whether or not they have loaded yet.
    """
    return tuple(sorted({DEFAULT_PROVIDER, *_AGENT_BACKENDS, *_LAZY_AGENT_BACKENDS}))


def _backend_provider(model: str) -> tuple[str, str]:
    """Split *model* into ``(provider, model)`` for dispatch.

    A value with no colon, or whose part before the first colon is not a plain
    provider word, is an Anthropic model ID and is returned unchanged.
    """
    head, sep, _ = model.partition(":")
    head = head.strip().lower()
    if not sep or head in _BARE_ID_PREFIXES or not _PROVIDER_RE.fullmatch(head):
        return DEFAULT_PROVIDER, model
    spec = parse_model_spec(model)
    return spec.provider, spec.model


def _agent_backend(provider: str) -> AgentBackend:
    runner = _AGENT_BACKENDS.get(provider)
    if runner is None and provider in _LAZY_AGENT_BACKENDS:
        import importlib

        importlib.import_module(_LAZY_AGENT_BACKENDS[provider])
        runner = _AGENT_BACKENDS.get(provider)
    if runner is None:
        raise UnknownAgentBackendError(
            f"no agent backend for provider {provider!r}; available: "
            f"{', '.join(registered_agent_backends())}",
            reason=f"unknown agent backend {provider!r}",
        )
    return runner


def _summarize_stderr(error_lines: list[str], recent_lines: list[str]) -> str:
    """Build a compact stderr summary for a failed SDK call.

    Prefers CLI ``[ERROR]`` lines (consecutive duplicates collapsed, capped)
    since they carry the actionable cause. Falls back to the last few raw
    lines when the CLI emitted no error-level output.
    """
    if error_lines:
        deduped = [line for line, _ in itertools.groupby(error_lines)]
        return "\n".join(deduped[-_STDERR_MAX_ERROR_LINES:])
    return "\n".join(recent_lines[-_STDERR_RING_LINES:])


def _sweep_debug_logs(sdk_dir: Path) -> None:
    """Delete debug-log files older than the log retention, once per process.

    Does nothing when the retention is ``0`` (keep forever).
    """
    global _debug_logs_swept
    if _debug_logs_swept:
        return
    _debug_logs_swept = True
    from .log_utils import retention_days

    days = retention_days()
    if days <= 0:
        return
    cutoff = time.time() - days * 86400
    for path in sdk_dir.glob("*.log"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


class _DebugLog:
    """One attempt's CLI stderr, written to disk as each line arrives.

    Never raises into the run: a failure to open or write disables the file
    and logs one warning. Writes after :meth:`close` (the CLI subprocess of a
    timed-out attempt can keep emitting) are dropped.
    """

    def __init__(self, path: Path, fh: IO[str], label: str) -> None:
        self.path = path
        self._fh: IO[str] | None = fh
        self._label = label

    @classmethod
    def open(cls, label: str, component: str, attempt: int) -> "_DebugLog | None":
        if os.environ.get(_DEBUG_LOG_ENV, "").strip().lower() == "off":
            return None
        try:
            from .log_utils import _get_logs_dir

            sdk_dir = _get_logs_dir() / _DEBUG_LOG_SUBDIR
            sdk_dir.mkdir(parents=True, exist_ok=True)
            _sweep_debug_logs(sdk_dir)
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            name = re.sub(
                r"[^A-Za-z0-9_.-]+", "_",
                f"{stamp}-{component or 'agent'}-{label}-p{os.getpid()}"
                f"-n{next(_debug_log_seq)}-a{attempt + 1}",
            )
            path = sdk_dir / f"{name}.log"
            # buffering=1: line-buffered, so every line reaches the OS as it
            # is written and survives a kill of this process. "x": a name that
            # still collides fails the open (caught below) rather than two
            # calls sharing one file.
            fh = open(path, "x", encoding="utf-8", buffering=1)
        except Exception:  # noqa: BLE001 — diagnostics must not break a run
            logger.warning(
                "run_agent [%s]: could not open CLI debug log (ignored)",
                label, exc_info=True,
            )
            return None
        return cls(path, fh, label)

    def write(self, line: str) -> None:
        fh = self._fh
        if fh is None:
            return
        try:
            fh.write(line if line.endswith("\n") else line + "\n")
        except Exception:  # noqa: BLE001
            self._fh = None
            logger.warning(
                "run_agent [%s]: CLI debug log write failed; stopped writing %s",
                self._label, self.path,
            )

    def close(self, *, keep: bool) -> None:
        fh, self._fh = self._fh, None
        if fh is not None:
            with contextlib.suppress(Exception):
                fh.close()
        if not keep:
            with contextlib.suppress(OSError):
                self.path.unlink()


def _hook_session_dir() -> Path:
    """Default cwd for agent runs — prevents project settings.json pickup."""
    cfg = Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude")))
    d = cfg / "hook-sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _safe_query(sdk, *, prompt, options):
    """Wrap ``claude_agent_sdk.query()`` to skip unknown message types.

    With ``debug-to-stderr`` enabled (which run_agent sets), the CLI emits
    internal message types (e.g. rate_limit_event) the bundled SDK parser
    doesn't know about — without this wrapper they crash the call with a
    generic ``Command failed with exit code 1``. Mandatory whenever
    ``debug-to-stderr`` is on.
    """
    gen = sdk.query(prompt=prompt, options=options).__aiter__()
    while True:
        try:
            message = await gen.__anext__()
            yield message
        except StopAsyncIteration:
            break
        except Exception as e:  # noqa: BLE001
            if "Unknown message type" in str(e):
                logger.debug("Skipping unknown SDK message type: %s", e)
                continue
            raise


def _record_cost(
    component: str, label: str, model: str | None,
    usage: AgentUsage, session_id: str,
) -> None:
    """Append one SDK-run record to the cost ledger; never raise.

    Undifferentiated ``cache_creation_tokens`` go in the 5m tier (errs low);
    the SDK's own ``cost_usd`` wins over token-derived pricing when present.
    """
    if not component:
        return
    if not any((usage.input_tokens, usage.output_tokens,
                usage.cache_creation_tokens, usage.cache_read_tokens,
                usage.cost_usd)):
        return
    try:
        import uuid
        from datetime import datetime, timezone

        from .costing import TokenCounts, append_records, build_record

        append_records([build_record(
            ts=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            source="sdk",
            session=session_id or f"sdk-{label}",
            model=model or "sdk-default",
            msg_id=f"run-{uuid.uuid4().hex[:12]}",
            tokens=TokenCounts(
                input=usage.input_tokens,
                output=usage.output_tokens,
                cw5m=usage.cache_creation_tokens,
                cr=usage.cache_read_tokens,
            ),
            component=component,
            cost_usd=usage.cost_usd if usage.cost_usd > 0 else None,
        )])
    except Exception:  # noqa: BLE001 — ledger is never worth failing a run
        logger.warning("run_agent [%s]: cost ledger write failed (ignored)",
                       label, exc_info=True)


def _sdk_class(sdk, name: str) -> type | None:
    """Fetch a message/block class off the SDK module, tolerating absence.

    Old SDK versions may lack e.g. ``ResultMessage``; test doubles may expose
    non-type attributes. Return the class only when it is usable with
    ``isinstance``.
    """
    cls = getattr(sdk, name, None)
    return cls if isinstance(cls, type) else None


async def run_agent(
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
    """Run one agent invocation via ``claude_agent_sdk.query()``.

    Args:
        prompt: The user prompt, sent unchanged at any size: the SDK writes it
            to the CLI's stdin, never onto the command line.
        prompt_file_fallback: Deprecated; has no effect. Passing ``True``
            emits a ``DeprecationWarning``. It will be removed in a later
            release.
        allowed_tools: Tool allow-list; ``None``/``[]`` means a no-tools text
            call. Under bypassPermissions this alone removes nothing, so it is
            also forwarded as the SDK's ``tools`` (the *base set* of tools that
            exist at all) and paired with a deny-list — see
            ``disallowed_tools``. Names outside ``TOOL_UNIVERSE`` are passed
            through but logged as a warning.
        disallowed_tools: Tools to remove, as a second layer under the base
            set. Defaults to ``deny_list(allowed_tools)``: everything in
            ``TOOL_UNIVERSE`` that the allow-list did not open. Pass a list to
            override, including ``[]`` to deliberately deny nothing — note this
            opts out of the *second* layer only; ``tools`` still bounds the run
            to the allow-list.
        effort: Reasoning effort; forwarded to the SDK only when set, so old
            SDK versions without the option keep working. ``None`` falls back
            to the default effort (see below).
        thinking: Extended-thinking config, forwarded verbatim to the SDK and
            only when set (same old-SDK tolerance as ``effort``). The case that
            motivated it is ``{"type": "disabled"}``: measured 2026-08-09, a
            cold no-tools call drops 18.4 s → 2.9 s with thinking off, which is
            the difference between fitting a hook budget and not. Latency, not
            quality, is what this knob buys — leave it ``None`` for work where
            reasoning depth matters.
        env: Extra env vars for the child. ``_HOOK_CHILD_SESSION`` always
            wins over this mapping — a caller cannot clear the fork-bomb
            guard.
        timeout_s: Hard wall-clock ceiling per attempt (``hard_timeout`` — a
            wedged CLI subprocess can block ``asyncio.wait_for`` forever).
        max_attempts: Total attempts; >1 turns the bundled CLI's intermittent
            exit-1 into a transparent recovery.
        label: Short identifier used in log lines to tell concurrent calls
            apart.
        component: Cost-ledger component tag (e.g. ``"buildme"``). When set,
            the run's usage is appended to the cost ledger (source ``"sdk"``);
            ledger failures never break the run. Empty = no ledger write.
        read_roots: Folders the read tools may open. When given (even as
            ``[]``), a ``PreToolUse`` hook denies ``Read``, ``Grep``, ``Glob``,
            ``LS`` and ``NotebookRead`` calls whose path resolves outside them
            (see :func:`path_allowed`). It limits those five tools only: a
            caller that also opens ``Bash`` is not bounded by it. ``None``
            (the default) installs no hook.

        model: Model ID or ``provider:model`` spec. ``None`` and Anthropic
            specs (bare, ``anthropic:``, ``claude:``) run on the Claude Agent
            SDK; another provider runs on its registered backend (see
            :func:`register_agent_backend`). ``None`` falls back to the
            default model (see below).

    Raises:
        AgentRunTimeout: every attempt exceeded ``timeout_s``.
        AgentRunError: every attempt failed for another reason, or the SDK is
            not importable. Carries ``.partial`` with whatever was consumed.
        UnknownAgentBackendError: ``model`` names a provider with no backend.
        UsageLimitError: the CLI reported a hit usage limit. Raised on that
            attempt, never retried; see the class.

    With no ``model``, the run uses :func:`multiplai_core.session.default_model`
    (the configured default, else the Claude Code session's model); with no
    ``effort``, :func:`multiplai_core.session.default_effort`. When neither
    finds a value the option is omitted and the CLI uses its own default.
    """
    if prompt_file_fallback:
        warnings.warn(
            "run_agent(prompt_file_fallback=...) has no effect: prompts are sent "
            "on stdin at any size. It will be removed in a later release.",
            DeprecationWarning,
            stacklevel=2,
        )
    if not model or effort is None:
        from .session import default_effort, default_model

        if not model:
            model = default_model()
        if effort is None:
            effort = default_effort()
    if model:
        provider, parsed = _backend_provider(model)
        if provider != DEFAULT_PROVIDER:
            runner = _agent_backend(provider)
            kwargs: dict[str, Any] = dict(
                system_prompt=system_prompt, allowed_tools=allowed_tools,
                disallowed_tools=disallowed_tools, max_turns=max_turns,
                model=f"{provider}:{parsed}", effort=effort, thinking=thinking, cwd=cwd,
                env=env, timeout_s=timeout_s, max_attempts=max_attempts,
                retry_backoff_s=retry_backoff_s,
                prompt_file_fallback=prompt_file_fallback, label=label,
                component=component, read_roots=read_roots,
            )
            return await runner(prompt, **kwargs)
        model = parsed

    try:
        import claude_agent_sdk as sdk
    except ImportError as e:
        raise AgentRunError(
            "claude-agent-sdk is not available in the current runtime",
            reason="claude-agent-sdk not installed",
        ) from e

    assistant_cls = _sdk_class(sdk, "AssistantMessage")
    text_cls = _sdk_class(sdk, "TextBlock")
    tool_use_cls = _sdk_class(sdk, "ToolUseBlock")
    result_cls = _sdk_class(sdk, "ResultMessage")
    system_cls = _sdk_class(sdk, "SystemMessage")
    hook_matcher_cls = getattr(sdk, "HookMatcher", None)
    if read_roots is not None and hook_matcher_cls is None:
        # Fail closed: without hooks the roots cannot be enforced.
        raise AgentRunError(
            f"run_agent [{label}]: read_roots needs claude-agent-sdk's "
            "HookMatcher, which the installed SDK lacks",
            reason="read_roots unsupported by installed SDK",
        )

    effective_tools = list(allowed_tools or [])
    # Guard merged LAST so a caller's env can never clear it: the reversed
    # order ({guard, **env}) let env={"_HOOK_CHILD_SESSION": ""} disable the
    # fork-bomb guard.
    run_env: dict[str, str] = {**(env or {}), "_HOOK_CHILD_SESSION": "1"}
    prompt_bytes = len(prompt.encode("utf-8"))

    read_hooks: dict[str, list] | None = None
    if read_roots is not None:
        hook_cwd = str(cwd) if cwd is not None else str(_hook_session_dir())
        read_hooks = {"PreToolUse": [hook_matcher_cls(
            matcher=_READ_ROOT_MATCHER,
            hooks=[_deny_outside(list(read_roots), hook_cwd, label)],
        )]}

    # A name outside TOOL_UNIVERSE is either a tool added since the list was
    # last derived, or a typo. Both are worth a line in the log: the typo case
    # otherwise fails silently — the tool is never available and the model
    # improvises around a capability it was told it has. Warn, don't raise;
    # TOOL_UNIVERSE is best-effort (see its comment) and must not be able to
    # block a legitimate call.
    unknown = [t for t in effective_tools if t not in TOOL_UNIVERSE]
    if unknown:
        logger.warning(
            "run_agent [%s]: allowed_tools names not in TOOL_UNIVERSE: %s — "
            "typo, or a tool added since the list was derived",
            label, ", ".join(sorted(unknown)),
        )

    logger.info(
        "START run_agent [%s] prompt=%d bytes tools=%s max_turns=%d model=%s "
        "timeout=%.0fs attempts=%d",
        label, prompt_bytes, effective_tools or "none", max_turns,
        model or "default", timeout_s, max_attempts,
    )

    loop = asyncio.get_running_loop()
    run_start = loop.time()
    last_exc: Exception | None = None
    last_reason = ""
    last_tail = ""
    last_partial: AgentRunResult | None = None
    last_session_id = ""
    last_debug_path = ""
    any_attempt_failed = False
    timed_out = False

    for attempt in range(max_attempts):
        # Fresh per-attempt capture: a ring of recent lines for trailing
        # context plus a bounded list of [ERROR] lines so a fatal error is
        # never evicted by a DEBUG burst.
        recent_lines: deque[str] = deque(maxlen=_STDERR_RING_LINES)
        error_lines: list[str] = []
        debug_log = _DebugLog.open(label, component, attempt)
        if debug_log is not None:
            logger.info(
                "run_agent [%s] attempt=%d/%d CLI debug log: %s",
                label, attempt + 1, max_attempts, debug_log.path,
            )

        def _on_stderr(
            line: str, _r=recent_lines, _e=error_lines, _d=debug_log,
        ) -> None:
            if _d is not None:
                _d.write(line)
            _r.append(line)
            if "[ERROR]" in line and len(_e) < _STDERR_ERROR_CAPTURE_CAP:
                _e.append(line)

        opts_kwargs: dict = dict(
            allowed_tools=effective_tools,
            # The base set of tools that exist at all, NOT an addition to
            # the built-in set — this is the lever that makes the tool
            # boundary a guarantee instead of a list someone maintains.
            # `[]` becomes `--tools ""`, i.e. no tools whatsoever, which
            # is the right default for the text-only calls that dominate.
            tools=list(effective_tools),
            max_turns=max_turns,
            permission_mode="bypassPermissions",
            system_prompt=system_prompt,
            model=model,
            env=run_env,
            cwd=str(cwd) if cwd is not None else str(_hook_session_dir()),
            setting_sources=[],
            extra_args={
                # Deliberately redundant: the pinned SDK already emits
                # --setting-sources= from setting_sources=[] (verified on
                # 0.2.129); kept as belt-and-braces against a regression.
                "setting-sources": "",
                "debug-to-stderr": None,
                "strict-mcp-config": None,
            },
            stderr=_on_stderr,
        )
        # Fail-closed: no explicit deny-list means deny everything the
        # allow-list did not open. An explicitly passed list (including
        # []) is forwarded verbatim — a caller can still opt out
        # deliberately.
        opts_kwargs["disallowed_tools"] = (
            deny_list(effective_tools) if disallowed_tools is None
            else list(disallowed_tools)
        )
        if effort is not None:
            opts_kwargs["effort"] = effort
        if thinking is not None:
            opts_kwargs["thinking"] = thinking
        if read_hooks is not None:
            opts_kwargs["hooks"] = read_hooks
        # A TypeError here is a signature mismatch with the installed SDK
        # — deterministic, so retrying cannot help. Convert it immediately
        # to honor the documented Raises contract (AgentRunError/
        # AgentRunTimeout only) instead of letting a raw TypeError escape.
        try:
            options = sdk.ClaudeAgentOptions(**opts_kwargs)
        except TypeError as e:
            raise AgentRunError(
                f"run_agent [{label}] claude-agent-sdk rejected the "
                f"constructed options (SDK signature mismatch): {e}",
                reason=f"ClaudeAgentOptions rejected: {e}",
            ) from e

        chunks: list[str] = []
        text_bytes = 0  # running UTF-8 size of chunks, for log lines
        files_changed: list[str] = []
        tool_calls: list[ToolCall] = []
        turns = 0
        usage = AgentUsage()
        session_id = ""
        attempt_start = loop.time()

        async def _consume() -> None:
            # Hold the generator explicitly and aclose() it in finally: on
            # timeout the task is cancelled fire-and-forget, and an
            # `async for` does not deterministically close its generator
            # on cancellation — the CLI subprocess would linger until GC
            # while a retry spawns a second one.
            nonlocal turns, usage, session_id, text_bytes
            gen = _safe_query(sdk, prompt=prompt, options=options)
            # Two timing lines split a slow call in two: CLI startup
            # (spawn, auth, config) versus the model request itself. A
            # stall with neither line means the CLI never came up.
            cli_ready_logged = False
            try:
                async for message in gen:
                    if not cli_ready_logged:
                        cli_ready_logged = True
                        child = ""
                        if system_cls and isinstance(message, system_cls):
                            data = getattr(message, "data", None) or {}
                            if isinstance(data, dict):
                                child = data.get("session_id") or ""
                        logger.info(
                            "run_agent [%s] CLI ready after %.1fs "
                            "child_session=%s",
                            label, loop.time() - attempt_start,
                            child or "?",
                        )
                    if assistant_cls and isinstance(message, assistant_cls):
                        if turns == 0:
                            logger.info(
                                "run_agent [%s] first reply after %.1fs",
                                label, loop.time() - attempt_start,
                            )
                        turns += 1
                        for block in message.content:
                            if text_cls and isinstance(block, text_cls):
                                chunks.append(block.text)
                                text_bytes += len(block.text.encode("utf-8"))
                            elif tool_use_cls and isinstance(block, tool_use_cls):
                                block_input = getattr(block, "input", None) or {}
                                tool_calls.append(ToolCall(
                                    name=block.name,
                                    input=copy.deepcopy(dict(block_input)),
                                ))
                                if block.name in ("Write", "Edit"):
                                    fp = block_input.get("file_path", "")
                                    if fp and fp not in files_changed:
                                        files_changed.append(fp)
                    elif result_cls and isinstance(message, result_cls):
                        session_id = getattr(message, "session_id", "") or ""
                        u = getattr(message, "usage", None) or {}
                        usage = AgentUsage(
                            input_tokens=u.get("input_tokens", 0) or 0,
                            output_tokens=u.get("output_tokens", 0) or 0,
                            cache_creation_tokens=u.get(
                                "cache_creation_input_tokens", 0
                            ) or 0,
                            cache_read_tokens=u.get(
                                "cache_read_input_tokens", 0
                            ) or 0,
                            cost_usd=getattr(message, "total_cost_usd", 0.0)
                            or 0.0,
                        )
            finally:
                await gen.aclose()

        async def _heartbeat(interval: float, attempt_no: int = attempt) -> None:
            """Log liveness every *interval* seconds until cancelled.

            Reads the same ``chunks``/``turns`` locals ``_consume()``
            mutates, so the line reports real progress rather than only
            that the task is scheduled.
            """
            hb_start = loop.time()
            while True:
                await asyncio.sleep(interval)
                logger.info(
                    "run_agent [%s] alive %.0fs attempt=%d/%d turns=%d "
                    "text=%d bytes",
                    label, loop.time() - hb_start, attempt_no + 1,
                    max_attempts, turns, text_bytes,
                )

        heartbeat_s = _env_float(_HEARTBEAT_ENV, _HEARTBEAT_DEFAULT_S)
        heartbeat_task = (
            asyncio.create_task(_heartbeat(heartbeat_s))
            if heartbeat_s > 0
            else None
        )
        backoff_s = 0.0

        attempt_ok = False
        try:
            await hard_timeout(_consume(), timeout_s)
            attempt_ok = True
            elapsed = loop.time() - run_start
            logger.info(
                "DONE run_agent [%s] attempt=%d/%d turns=%d text=%d bytes "
                "files_changed=%d elapsed=%.1fs",
                label, attempt + 1, max_attempts, turns, text_bytes,
                len(files_changed), elapsed,
            )
            if any_attempt_failed:
                logger.warning(
                    "run_agent [%s] recovered after a failed attempt", label
                )
            _record_cost(component, label, model, usage, session_id)
            return AgentRunResult(
                text="".join(chunks).strip(),
                turns=turns,
                usage=usage,
                files_changed=files_changed,
                tool_calls=tuple(tool_calls),
                stderr_tail="",
            )
        except Exception as e:  # noqa: BLE001
            last_exc = e
            timed_out = isinstance(e, asyncio.TimeoutError)
            # TimeoutError stringifies to "" — describe it explicitly.
            last_reason = (
                f"timed out after {timeout_s:.0f}s"
                if timed_out
                else f"failed: {e}"
            )
            last_tail = _summarize_stderr(error_lines, list(recent_lines))
            last_partial = AgentRunResult(
                text="".join(chunks).strip(),
                turns=turns,
                usage=usage,
                files_changed=files_changed,
                tool_calls=tuple(tool_calls),
                stderr_tail=last_tail,
            )
            last_session_id = session_id
            last_debug_path = str(debug_log.path) if debug_log else ""
            any_attempt_failed = True
            limit_hit = None if timed_out else _usage_limit_match(str(e))
            if limit_hit is not None:
                # Raised from inside the attempt so `finally` still closes the
                # debug log and the heartbeat; no backoff, no next attempt.
                logger.error(
                    "FAIL run_agent [%s] usage limit hit, not retried: %s "
                    "after %d attempt(s) elapsed=%.1fs CLI debug log: %s\n"
                    "--- captured CLI stderr (errors) ---\n%s",
                    label, last_reason, attempt + 1, loop.time() - run_start,
                    last_debug_path or "(none)", last_tail or "(none)",
                )
                _record_cost(component, label, model, last_partial.usage, last_session_id)
                raise UsageLimitError(
                    f"run_agent [{label}] {last_reason} after {attempt + 1} attempt(s)",
                    limit=(limit_hit.group("limit") or "").strip().lower(),
                    resets=(limit_hit.group("resets") or "").strip(),
                    reason=last_reason,
                    attempts=attempt + 1,
                    stderr_tail=last_tail,
                    partial=last_partial,
                    debug_log_path=last_debug_path,
                ) from e
            if attempt + 1 < max_attempts:
                logger.warning(
                    "run_agent [%s] %s (attempt %d/%d), retrying in %.1fs"
                    " — CLI debug log: %s",
                    label, last_reason, attempt + 1, max_attempts,
                    retry_backoff_s, last_debug_path or "(none)",
                )
                backoff_s = retry_backoff_s
        finally:
            if debug_log is not None:
                keep_s = _env_float(_DEBUG_KEEP_ENV, _DEBUG_KEEP_DEFAULT_S)
                debug_log.close(keep=(
                    not attempt_ok
                    or keep_s <= 0
                    or loop.time() - attempt_start >= keep_s
                ))
            # Cancel AND await, for the same reason _consume() explicitly
            # aclose()s its generator: on timeout the attempt is cancelled
            # fire-and-forget, and a heartbeat left pending would keep
            # logging a dead attempt's counters into the next one (or past
            # the end of the call). Awaiting makes the teardown ordered.
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task

        # Backoff sits outside the attempt so the heartbeat is already
        # stopped: an "alive" line during the gap would report the failed
        # attempt's turn count as if the run were still producing.
        if backoff_s:
            await asyncio.sleep(backoff_s)

    elapsed = loop.time() - run_start
    logger.error(
        "FAIL run_agent [%s] %s after %d attempt(s) elapsed=%.1fs "
        "CLI debug log: %s\n"
        "--- captured CLI stderr (errors) ---\n%s",
        label, last_reason, max_attempts, elapsed, last_debug_path or "(none)",
        last_tail or "(none)",
    )
    if last_partial is not None:
        # Failed runs still spent tokens — record what was consumed.
        _record_cost(component, label, model, last_partial.usage, last_session_id)
    err_cls = AgentRunTimeout if timed_out else AgentRunError
    raise err_cls(
        f"run_agent [{label}] {last_reason} after {max_attempts} attempt(s)",
        reason=last_reason,
        attempts=max_attempts,
        stderr_tail=last_tail,
        partial=last_partial,
        debug_log_path=last_debug_path,
    ) from last_exc
