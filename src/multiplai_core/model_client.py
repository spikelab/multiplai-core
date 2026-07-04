"""Model client abstraction for multiplai plugin.

Provides a Protocol-based interface with two implementations:
- AgentSDKClient: uses claude_agent_sdk from the host runtime (zero-config)
- AnthropicAPIClient: uses the anthropic PyPI package with an API key

The create_client() factory tries Agent SDK first, falls back to API key.
"""

import asyncio
import logging
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096

# CLI stderr capture (debug-to-stderr is on, so raw volume is large). We keep
# it all in memory, never on disk, and only surface a compact slice when a
# call actually fails: the `[ERROR]` lines (which carry the real cause —
# rate limit, auth, crash), or a few trailing raw lines if there were none.
_STDERR_RING_LINES = 50          # trailing context kept for the fallback tail
_STDERR_MAX_ERROR_LINES = 15     # [ERROR] lines surfaced in the summary
_STDERR_ERROR_CAPTURE_CAP = 200  # absolute cap on captured [ERROR] lines

# The bundled CLI intermittently exits 1 (verified recurring: diary
# 2026-04-19/28304b42, 2026-04-29/8bcd0f1c). One bounded retry turns a flaky
# failure into a transparent recovery for unattended pipelines like dream.
_SDK_MAX_ATTEMPTS = 2
_SDK_RETRY_BACKOFF_S = 1.5

# These callers want a pure text completion, but claude_agent_sdk.query() runs
# the full agentic loop. allowed_tools=[] does NOT remove tools — under
# permission_mode="bypassPermissions" it's only an allow-list, so every default
# tool stays present and auto-approved. With the original max_turns=1 the model
# would nondeterministically spend its single turn on an exploratory tool call
# (Agent→Explore, a guessed Read path, or ToolSearch loading a deferred tool
# like AskUserQuestion) whose result needs a turn 2 that never comes, so the
# session ends with no text → CLI exit 1. Verified across subprocess transcripts
# 2026-05-24. Enumerating disallowed_tools can't fully fix this: ToolSearch is a
# meta-tool that can load any deferred tool, so there is always something to
# call. The real fix is _SDK_MAX_TURNS > 1 (a stray tool call recovers instead
# of crashing) plus a system-prompt directive to answer directly. disallowed_
# tools is kept as a SAFETY floor only: under bypassPermissions a multi-turn run
# must never be able to mutate the filesystem, shell out, ask a (headless,
# unanswerable) question, or spawn an expensive subagent.
_SDK_MAX_TURNS = 6

# Hard ceiling on a single SDK call. The bundled CLI subprocess can stall
# indefinitely — a network hang on the model call, or the CLI parked waiting on
# stream-json stdin that never closes — and the SDK exposes no timeout. Without
# this guard the `async for` consume loop below blocks forever: the retry/except
# machinery only catches *exceptions* (crashes), never a hang, so a single
# stalled subprocess wedges the whole pipeline (observed: dream hung ~8h on the
# critic pass, 2026-06-20). asyncio.wait_for turns a stall into a TimeoutError
# that the existing retry loop catches and, after _SDK_MAX_ATTEMPTS, surfaces as
# SDKQueryError — callers that tolerate failure (e.g. dream's critic pass) then
# degrade gracefully instead of hanging. Default keeps interactive callers
# (context_manager, session_start) snappy; long-running batch callers raise it
# via env — dream.py sets MULTIPLAI_SDK_CALL_TIMEOUT_S=1800 before import.
_SDK_CALL_TIMEOUT_S = float(os.environ.get("MULTIPLAI_SDK_CALL_TIMEOUT_S", "600"))
_DISALLOWED_TOOLS = [
    "Bash", "BashOutput", "KillShell", "Edit", "Write", "NotebookEdit",
    "Task", "Agent", "AskUserQuestion", "SlashCommand", "ExitPlanMode",
]
_NO_TOOLS_SUFFIX = (
    "\n\nAll information you need is already provided in this message. Do NOT "
    "use any tools, skills, subagents, or tool search, and do NOT ask "
    "questions. Respond directly with only the requested output text."
)


@dataclass(frozen=True)
class ModelResponse:
    """Normalized response from any model client."""
    content: str


class SDKQueryError(RuntimeError):
    """Raised when ``claude_agent_sdk.query()`` fails.

    ``stderr_tail`` holds a filtered summary of the CLI stderr — the
    ``[ERROR]`` lines (or the last few raw lines if there were none) —
    enough to surface rate-limit, auth, or crash details that the SDK's
    generic "exit code 1" would otherwise hide. Captured in memory only;
    nothing is written to disk.
    """

    def __init__(self, message: str, *, stderr_tail: str = "") -> None:
        self._base_message = message
        self.stderr_tail = stderr_tail
        super().__init__(self._format())

    def _format(self) -> str:
        parts = [self._base_message]
        if self.stderr_tail:
            parts.append("--- captured CLI stderr (errors) ---")
            parts.append(self.stderr_tail)
            parts.append("--- end stderr ---")
        else:
            parts.append("(no CLI stderr captured — subprocess likely died before emitting any output)")
        return "\n".join(parts)


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


def _messages_to_prompt(messages: list[dict]) -> str:
    """Flatten the ModelClient messages list into a single prompt string.

    ``claude_agent_sdk.query()`` takes a ``prompt`` string rather than a
    messages list. Plugin callers invoke single-turn user queries, so we
    concatenate every user message. Non-user roles are ignored (the
    system prompt is passed separately via ``ClaudeAgentOptions``).
    """
    user_parts = [m["content"] for m in messages if m.get("role") == "user"]
    return "\n\n".join(user_parts)


def _hook_session_dir() -> Path:
    """cwd for no-tool SDK calls — prevents project settings.json pickup."""
    cfg = Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude")))
    d = cfg / "hook-sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _safe_query(sdk, *, prompt, options):
    """Wrap ``claude_agent_sdk.query()`` to skip unknown message types.

    The SDK message parser raises for message types it doesn't recognize.
    With ``debug-to-stderr`` enabled (which this client sets), the CLI emits
    additional internal message types the bundled SDK parser doesn't know
    about — without this wrapper they crash the call with a generic
    ``Command failed with exit code 1``. Mirrors deep-research/sdk.py and
    buildme's ``_safe_query``. This guard is mandatory whenever
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


@runtime_checkable
class ModelClient(Protocol):
    """Abstract interface for LLM clients."""

    async def query(
        self,
        system: str,
        messages: list[dict],
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 1.0,
    ) -> ModelResponse: ...


class AgentSDKClient:
    """Uses ``claude_agent_sdk.query()`` from the Claude Code host runtime.

    ``claude_agent_sdk.query()`` is an async generator that takes a
    ``prompt`` string and a ``ClaudeAgentOptions`` bundle and yields
    ``AssistantMessage`` objects. This client adapts the ``ModelClient``
    interface (system + messages list) to that shape for single-turn
    queries, captures CLI stderr so SDK-internal failures (rate limits,
    auth, CLI crashes) surface to the caller rather than being silently
    dropped, and attaches the captured tail to any raised exception.

    Requires running inside Claude Code where the host injects
    ``claude_agent_sdk`` into the plugin's Python environment.
    ``max_tokens`` and ``temperature`` parameters are accepted for
    interface parity but are not forwarded — the SDK uses session
    defaults.
    """

    def __init__(self) -> None:
        try:
            import claude_agent_sdk
            self._sdk = claude_agent_sdk
        except ImportError:
            raise ImportError(
                "claude_agent_sdk is not available in the current runtime. "
                "This client requires running inside Claude Code."
            )

    async def query(
        self,
        system: str,
        messages: list[dict],
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 1.0,
    ) -> ModelResponse:
        """Send a single-turn query via the Agent SDK and return normalized text."""
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            TextBlock,
        )

        prompt = _messages_to_prompt(messages)
        system_bytes = len(system.encode("utf-8")) if system else 0
        prompt_bytes = len(prompt.encode("utf-8"))
        logger.info(
            "SDK call start: model=%s system=%d bytes prompt=%d bytes",
            model, system_bytes, prompt_bytes,
        )
        call_start = asyncio.get_event_loop().time()

        last_exc: Exception | None = None
        last_tail = ""
        any_attempt_failed = False
        for attempt in range(_SDK_MAX_ATTEMPTS):
            # Capture stderr in memory only: a ring buffer of recent lines for
            # trailing context, plus a bounded list of [ERROR] lines so a
            # fatal error is never evicted by a DEBUG burst. Summarized into a
            # compact tail only if the call fails.
            recent_lines: deque[str] = deque(maxlen=_STDERR_RING_LINES)
            error_lines: list[str] = []

            def _on_stderr(line: str, _r=recent_lines, _e=error_lines) -> None:
                _r.append(line)
                if "[ERROR]" in line and len(_e) < _STDERR_ERROR_CAPTURE_CAP:
                    _e.append(line)

            options = ClaudeAgentOptions(
                allowed_tools=[],
                disallowed_tools=_DISALLOWED_TOOLS,  # see _DISALLOWED_TOOLS note
                max_turns=_SDK_MAX_TURNS,
                permission_mode="bypassPermissions",
                system_prompt=system + _NO_TOOLS_SUFFIX,
                model=model,
                env={"_HOOK_CHILD_SESSION": "1"},
                cwd=str(_hook_session_dir()),
                setting_sources=[],
                # strict-mcp-config isolates this subprocess from
                # account-level MCP integrations (claude.ai Gmail/Drive/
                # Calendar/etc). Without it the bundled CLI discovers those
                # OAuth integrations and tries to authenticate them in a
                # non-interactive subprocess, collapsing with exit 1 and no
                # usable stderr. Verified root cause 2026-05-19 against
                # mcp-needs-auth-cache.json; see anthropics/
                # claude-agent-sdk-python issues + PLANS doc.
                extra_args={
                    "setting-sources": "",
                    "debug-to-stderr": None,
                    "strict-mcp-config": None,
                },
                stderr=_on_stderr,
            )

            chunks: list[str] = []
            message_count = 0

            async def _consume() -> int:
                # Drain the SDK generator into `chunks`; returns the message count.
                # Factored out so asyncio.wait_for can bound the whole drain —
                # cancelling this coroutine on timeout closes the SDK generator,
                # which terminates the bundled-CLI subprocess.
                count = 0
                async for message in _safe_query(
                    self._sdk, prompt=prompt, options=options
                ):
                    count += 1
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                chunks.append(block.text)
                return count

            try:
                message_count = await asyncio.wait_for(
                    _consume(), timeout=_SDK_CALL_TIMEOUT_S
                )
                response_bytes = sum(len(c.encode("utf-8")) for c in chunks)
                elapsed = asyncio.get_event_loop().time() - call_start
                logger.info(
                    "SDK call OK on attempt %d/%d: messages=%d response=%d bytes elapsed=%.1fs",
                    attempt + 1, _SDK_MAX_ATTEMPTS, message_count, response_bytes, elapsed,
                )
                if any_attempt_failed:
                    logger.warning(
                        "claude_agent_sdk.query() recovered after a failed attempt"
                    )
                return ModelResponse(content="".join(chunks).strip())
            except Exception as e:  # noqa: BLE001
                last_exc = e
                last_tail = _summarize_stderr(error_lines, list(recent_lines))
                any_attempt_failed = True
                # TimeoutError stringifies to "" — describe it explicitly.
                reason = (
                    f"timed out after {_SDK_CALL_TIMEOUT_S:.0f}s"
                    if isinstance(e, asyncio.TimeoutError)
                    else f"failed: {e}"
                )
                if attempt + 1 < _SDK_MAX_ATTEMPTS:
                    logger.warning(
                        "claude_agent_sdk.query() %s (attempt %d/%d), "
                        "retrying in %.1fs",
                        reason,
                        attempt + 1,
                        _SDK_MAX_ATTEMPTS,
                        _SDK_RETRY_BACKOFF_S,
                    )
                    await asyncio.sleep(_SDK_RETRY_BACKOFF_S)

        raise SDKQueryError(
            f"claude_agent_sdk.query() failed after {_SDK_MAX_ATTEMPTS} "
            f"attempts: {last_exc}",
            stderr_tail=last_tail,
        ) from last_exc


class AnthropicAPIClient:
    """Uses the anthropic PyPI package with an explicit API key.

    The underlying ``AsyncAnthropic`` client is created lazily on the
    first call to :meth:`query`, so the ``anthropic`` package need not
    be importable at instantiation time.
    """

    def __init__(self, api_key: str | None) -> None:
        if not api_key:
            raise ValueError(
                "An API key is required for the Anthropic fallback client. "
                "Set CLAUDE_PLUGIN_OPTION_anthropic_api_key or pass api_key directly."
            )
        self._api_key = api_key
        self._client = None  # lazily created on first query

    def _ensure_client(self):
        """Lazily initialize the AsyncAnthropic client on first use."""
        if self._client is None:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def query(
        self,
        system: str,
        messages: list[dict],
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 1.0,
    ) -> ModelResponse:
        """Send a query via the Anthropic API and return a normalized response."""
        client = self._ensure_client()
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        )
        # Empty content list (tool-only turn, refusal, non-text stop) would
        # otherwise IndexError and convert a recoverable empty reply into a
        # total extraction/routing failure.
        text = ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text = block.text
                break
        return ModelResponse(content=text)


def detect_client_type() -> str:
    """Detect which model client backend will be used.

    Returns a human-readable string indicating the selected client type.
    This is a synchronous check suitable for logging at session start.
    """
    try:
        import claude_agent_sdk  # noqa: F401
        return "AgentSDKClient"
    except ImportError:
        key = os.environ.get("CLAUDE_PLUGIN_OPTION_anthropic_api_key", "")
        if key:
            return "AnthropicAPIClient"
        return "none (no SDK or API key)"


async def create_client(*, api_key: str | None = None) -> ModelClient:
    """Create a model client. Tries Agent SDK first, falls back to API key.

    Args:
        api_key: Optional API key override. If not provided, reads from
                 CLAUDE_PLUGIN_OPTION_anthropic_api_key env var.

    Returns:
        A ModelClient instance.

    Raises:
        RuntimeError: If neither Agent SDK nor API key is available.
    """
    try:
        client = AgentSDKClient()
        logger.info("Model client: Agent SDK selected (zero-config)")
        return client
    except ImportError:
        pass

    # Fall back to API key
    key = api_key or os.environ.get("CLAUDE_PLUGIN_OPTION_anthropic_api_key", "")
    if not key:
        raise RuntimeError(
            "Neither the Agent SDK nor an API key is available. "
            "Install claude_agent_sdk or set CLAUDE_PLUGIN_OPTION_anthropic_api_key."
        )

    logger.warning("Model client: Falling back to Anthropic API key authentication")
    return AnthropicAPIClient(key)
