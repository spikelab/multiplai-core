"""Single SDK agent runner — the one place every background agent invocation goes.

``run_agent()`` wraps ``claude_agent_sdk.query()`` with the machinery that the
model client, buildme, and deep-research each used to reimplement privately:
subprocess isolation flags, unknown-message skipping, hard timeout with
generator teardown, in-memory stderr capture with a compact failure tail,
big-prompt tempfile fallback, opt-in retry, and a normalized result carrying
text, turns, usage, and files changed.

Skill-specific policy stays in the skills: trust gating (buildme), usage
accumulation across a run (deep-research), Pydantic structured-output
validation (both), and each caller's public error taxonomy — callers catch
``AgentRunError``/``AgentRunTimeout`` and re-raise their own types.

Every invocation always gets the isolation/hardening bundle:

- ``permission_mode="bypassPermissions"`` — these are unattended subprocesses;
  callers that need a safety boundary gate BEFORE calling (buildme's
  ``--trust-repo``) or constrain tools (``allowed_tools``/``disallowed_tools``).
- ``setting_sources=[]`` + ``extra_args={"setting-sources": ""}`` — both are
  required; without them the child inherits parent settings/hooks and spawns
  runaway subagents (verified 2026-04-20).
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
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .aio import hard_timeout

logger = logging.getLogger(__name__)

# The SDK passes the prompt as a CLI argument to a subprocess. Past ~100KB the
# OS rejects it with E2BIG; we stay conservative to leave room for args/env.
MAX_PROMPT_BYTES = 80_000
# The tempfile fallback needs turns to read the file and then answer.
_PROMPT_FILE_MIN_TURNS = 3
# Child CLIs don't inherit the parent's settings.json, so the default Read
# cap (2000 lines) would truncate a large prompt file.
_PROMPT_FILE_READ_TOKENS = "100000"

# CLI stderr capture (debug-to-stderr is on, so raw volume is large). Kept in
# memory only, never on disk; summarized into a compact tail only on failure:
# the [ERROR] lines (which carry the real cause — rate limit, auth, crash), or
# a few trailing raw lines if there were none.
_STDERR_RING_LINES = 50          # trailing context kept for the fallback tail
_STDERR_MAX_ERROR_LINES = 15     # [ERROR] lines surfaced in the summary
_STDERR_ERROR_CAPTURE_CAP = 200  # absolute cap on captured [ERROR] lines


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
class AgentRunResult:
    """Normalized result of one agent run."""

    text: str                    # concatenated assistant text, stripped
    turns: int                   # number of AssistantMessages consumed
    usage: AgentUsage
    files_changed: list[str]     # file_path args of Write/Edit ToolUseBlocks
    stderr_tail: str             # compact failure context ("" on success)


class AgentRunError(RuntimeError):
    """Raised when an agent run fails beyond the retry budget.

    ``stderr_tail`` holds the compact CLI stderr summary; ``reason`` a short
    human cause ("timed out after 600s" / "failed: <exc>"); ``attempts`` how
    many attempts were made; ``partial`` the result assembled up to the
    failure point (text/turns/files/usage), for callers that degrade to
    partial output instead of failing hard (buildme's agent_call).
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str = "",
        attempts: int = 1,
        stderr_tail: str = "",
        partial: AgentRunResult | None = None,
    ) -> None:
        self.reason = reason or message
        self.attempts = attempts
        self.stderr_tail = stderr_tail
        self.partial = partial
        parts = [message]
        if stderr_tail:
            parts.append("--- captured CLI stderr (errors) ---")
            parts.append(stderr_tail)
            parts.append("--- end stderr ---")
        super().__init__("\n".join(parts))


class AgentRunTimeout(AgentRunError):
    """The run exceeded ``timeout_s`` on every attempt."""


def _summarize_stderr(error_lines: list[str], recent_lines: list[str]) -> str:
    """Build a compact stderr summary for a failed SDK call.

    Prefers CLI ``[ERROR]`` lines (consecutive duplicates collapsed, capped)
    since they carry the actionable cause. Falls back to the last few raw
    lines when the CLI emitted no error-level output.
    """
    if error_lines:
        deduped: list[str] = []
        for line in error_lines:
            if not deduped or deduped[-1] != line:
                deduped.append(line)
        return "\n".join(deduped[-_STDERR_MAX_ERROR_LINES:])
    return "\n".join(recent_lines[-_STDERR_RING_LINES:])


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
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout_s: float = 600.0,
    max_attempts: int = 1,
    retry_backoff_s: float = 1.5,
    prompt_file_fallback: bool = True,
    label: str = "agent",
) -> AgentRunResult:
    """Run one agent invocation via ``claude_agent_sdk.query()``.

    Args:
        prompt: The user prompt. If it exceeds ``MAX_PROMPT_BYTES`` and
            ``prompt_file_fallback`` is on, it is written to a temp file and
            the agent is directed to Read it (E2BIG workaround, no data loss).
        allowed_tools: Tool allow-list; ``None``/``[]`` means a no-tools text
            call. Note allowed_tools is only an allow-list under
            bypassPermissions — pass ``disallowed_tools`` to actually remove
            tools (see model_client._DISALLOWED_TOOLS for the rationale).
        effort: Reasoning effort; forwarded to the SDK only when set, so old
            SDK versions without the option keep working.
        env: Extra env vars merged over the isolation baseline.
        timeout_s: Hard wall-clock ceiling per attempt (``hard_timeout`` — a
            wedged CLI subprocess can block ``asyncio.wait_for`` forever).
        max_attempts: Total attempts; >1 turns the bundled CLI's intermittent
            exit-1 into a transparent recovery.
        label: Short identifier used in log lines to tell concurrent calls
            apart.

    Raises:
        AgentRunTimeout: every attempt exceeded ``timeout_s``.
        AgentRunError: every attempt failed for another reason, or the SDK is
            not importable. Carries ``.partial`` with whatever was consumed.
    """
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

    effective_tools = list(allowed_tools or [])
    run_env: dict[str, str] = {"_HOOK_CHILD_SESSION": "1", **(env or {})}
    prompt_file: str | None = None

    prompt_bytes = len(prompt.encode("utf-8"))
    if prompt_file_fallback and prompt_bytes > MAX_PROMPT_BYTES:
        logger.info(
            "run_agent [%s]: prompt too large for CLI arg (%d bytes), "
            "writing to temp file", label, prompt_bytes,
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", prefix="agent_prompt_", delete=False
        ) as f:
            f.write(prompt)
            prompt_file = f.name
        # The directive must not encourage narration — any text emitted
        # before reading gets captured as output.
        prompt = (
            f"Read the file {prompt_file} using the Read tool. "
            f"It contains your complete instructions and data. "
            f"After reading, follow those instructions exactly and produce "
            f"the requested output. Do not describe what you are doing — "
            f"just read the file and produce the output directly."
        )
        if "Read" not in effective_tools:
            effective_tools.append("Read")
        max_turns = max(max_turns, _PROMPT_FILE_MIN_TURNS)
        run_env["CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS"] = _PROMPT_FILE_READ_TOKENS

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
    any_attempt_failed = False
    timed_out = False

    try:
        for attempt in range(max_attempts):
            # Fresh per-attempt capture: a ring of recent lines for trailing
            # context plus a bounded list of [ERROR] lines so a fatal error is
            # never evicted by a DEBUG burst.
            recent_lines: deque[str] = deque(maxlen=_STDERR_RING_LINES)
            error_lines: list[str] = []

            def _on_stderr(line: str, _r=recent_lines, _e=error_lines) -> None:
                _r.append(line)
                if "[ERROR]" in line and len(_e) < _STDERR_ERROR_CAPTURE_CAP:
                    _e.append(line)

            opts_kwargs: dict = dict(
                allowed_tools=effective_tools,
                max_turns=max_turns,
                permission_mode="bypassPermissions",
                system_prompt=system_prompt,
                model=model,
                env=run_env,
                cwd=str(cwd) if cwd is not None else str(_hook_session_dir()),
                setting_sources=[],
                extra_args={
                    "setting-sources": "",
                    "debug-to-stderr": None,
                    "strict-mcp-config": None,
                },
                stderr=_on_stderr,
            )
            if disallowed_tools is not None:
                opts_kwargs["disallowed_tools"] = list(disallowed_tools)
            if effort is not None:
                opts_kwargs["effort"] = effort
            options = sdk.ClaudeAgentOptions(**opts_kwargs)

            chunks: list[str] = []
            files_changed: list[str] = []
            turns = 0
            usage = AgentUsage()

            async def _consume() -> None:
                # Hold the generator explicitly and aclose() it in finally: on
                # timeout the task is cancelled fire-and-forget, and an
                # `async for` does not deterministically close its generator
                # on cancellation — the CLI subprocess would linger until GC
                # while a retry spawns a second one.
                nonlocal turns, usage
                gen = _safe_query(sdk, prompt=prompt, options=options)
                try:
                    async for message in gen:
                        if assistant_cls and isinstance(message, assistant_cls):
                            turns += 1
                            for block in message.content:
                                if text_cls and isinstance(block, text_cls):
                                    chunks.append(block.text)
                                elif tool_use_cls and isinstance(block, tool_use_cls):
                                    if block.name in ("Write", "Edit"):
                                        fp = (getattr(block, "input", None) or {}).get(
                                            "file_path", ""
                                        )
                                        if fp and fp not in files_changed:
                                            files_changed.append(fp)
                        elif result_cls and isinstance(message, result_cls):
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

            try:
                await hard_timeout(_consume(), timeout_s)
                elapsed = loop.time() - run_start
                logger.info(
                    "DONE run_agent [%s] attempt=%d/%d turns=%d text=%d bytes "
                    "files_changed=%d elapsed=%.1fs",
                    label, attempt + 1, max_attempts, turns,
                    sum(len(c.encode("utf-8")) for c in chunks),
                    len(files_changed), elapsed,
                )
                if any_attempt_failed:
                    logger.warning(
                        "run_agent [%s] recovered after a failed attempt", label
                    )
                return AgentRunResult(
                    text="".join(chunks).strip(),
                    turns=turns,
                    usage=usage,
                    files_changed=files_changed,
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
                    stderr_tail=last_tail,
                )
                any_attempt_failed = True
                if attempt + 1 < max_attempts:
                    logger.warning(
                        "run_agent [%s] %s (attempt %d/%d), retrying in %.1fs",
                        label, last_reason, attempt + 1, max_attempts,
                        retry_backoff_s,
                    )
                    await asyncio.sleep(retry_backoff_s)
    finally:
        if prompt_file:
            try:
                os.unlink(prompt_file)
            except OSError:
                pass

    elapsed = loop.time() - run_start
    logger.error(
        "FAIL run_agent [%s] %s after %d attempt(s) elapsed=%.1fs\n"
        "--- captured CLI stderr (errors) ---\n%s",
        label, last_reason, max_attempts, elapsed, last_tail or "(none)",
    )
    err_cls = AgentRunTimeout if timed_out else AgentRunError
    raise err_cls(
        f"run_agent [{label}] {last_reason} after {max_attempts} attempt(s)",
        reason=last_reason,
        attempts=max_attempts,
        stderr_tail=last_tail,
        partial=last_partial,
    ) from last_exc
