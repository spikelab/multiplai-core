"""Model client abstraction for multiplai plugin.

Provides a Protocol-based interface with two implementations:
- AgentSDKClient: uses claude_agent_sdk from the host runtime (zero-config)
- AnthropicAPIClient: uses the anthropic PyPI package with an API key

The create_client() factory tries Agent SDK first, falls back to API key.
"""

import logging
import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .agent_runner import (  # noqa: F401 — _summarize_stderr re-exported for compat
    AgentRunError,
    _summarize_stderr,
    run_agent,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096

# The bundled CLI intermittently exits 1 (verified recurring). One bounded
# retry turns a flaky failure into a transparent recovery for unattended
# pipelines like dream.
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
# 2026-05-24. The real fix is _SDK_MAX_TURNS > 1 (a stray tool call recovers
# instead of crashing) plus a system-prompt directive to answer directly.
# disallowed_tools is the SAFETY floor: under bypassPermissions a multi-turn
# run must never be able to mutate the filesystem, shell out, spawn a subagent,
# ask a (headless, unanswerable) question — or, because callers routinely feed
# UNTRUSTED text through this client, read local files / fetch URLs. Leaving
# Read+WebFetch enabled would let injected instructions in that text exfiltrate
# local secrets in a single auto-approved multi-turn run. ToolSearch and Skill
# are blocked too so deferred tools can't be loaded back in. The
# _NO_TOOLS_SUFFIX prompt directive is an optimization, not a boundary.
_SDK_MAX_TURNS = 6

# Hard ceiling on a single SDK call. The bundled CLI subprocess can stall
# indefinitely — a network hang on the model call, or the CLI parked waiting on
# stream-json stdin that never closes — and the SDK exposes no timeout. Without
# this guard the `async for` consume loop below blocks forever: the retry/except
# machinery only catches *exceptions* (crashes), never a hang, so a single
# stalled subprocess wedges the whole pipeline (observed: dream hung ~8h on the
# critic pass, 2026-06-20). run_agent's hard timeout turns a stall into a
# TimeoutError that the retry budget catches and, after _SDK_MAX_ATTEMPTS,
# surfaces as SDKQueryError — callers that tolerate failure (e.g. dream's
# critic pass) then degrade gracefully instead of hanging. Default keeps
# interactive callers
# (context_manager, session_start) snappy; long-running batch callers raise it
# via env — e.g. a long-running batch caller sets
# MULTIPLAI_SDK_CALL_TIMEOUT_S=1800 before import.
def _env_float(name: str, default: float) -> float:
    """Parse a float env var, falling back to the default on garbage.

    Read at import time (this value is a module constant), so a malformed
    value must not crash `import multiplai_core` for every consumer — mirror
    the defensive parsing in log_utils.retention_days().
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using default %s", name, raw, default)
        return default


_SDK_CALL_TIMEOUT_S = _env_float("MULTIPLAI_SDK_CALL_TIMEOUT_S", 600.0)
_DISALLOWED_TOOLS = [
    # mutation / execution
    "Bash", "BashOutput", "KillShell", "Edit", "Write", "NotebookEdit",
    "Task", "Agent", "AskUserQuestion", "SlashCommand", "ExitPlanMode",
    # read / network / meta — closes the prompt-injection exfiltration chain
    # (untrusted input steering an auto-approved Read → WebFetch of a secret)
    "Read", "Grep", "Glob", "LS", "WebFetch", "WebSearch", "ToolSearch",
    "Skill",
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


def _messages_to_prompt(messages: list[dict]) -> str:
    """Flatten the ModelClient messages list into a single prompt string.

    ``claude_agent_sdk.query()`` takes a ``prompt`` string rather than a
    messages list. Plugin callers invoke single-turn user queries, so we
    concatenate every user message. Non-user roles are ignored (the
    system prompt is passed separately via ``ClaudeAgentOptions``).

    Accepts both plain-string content and Anthropic content-block lists
    (``[{"type": "text", "text": ...}]``) so the same messages work against
    either backend; non-text blocks are skipped.
    """
    user_parts: list[str] = []
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m["content"]
        if isinstance(content, str):
            user_parts.append(content)
        else:
            user_parts.extend(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
    return "\n\n".join(user_parts)


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
        self._warned_ignored_params = False
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
        # max_tokens/temperature are accepted for interface parity but the SDK
        # uses session defaults — warn once if a caller relies on them (e.g.
        # temperature=0 for determinism) so the silent no-op is visible.
        if not self._warned_ignored_params and (
            temperature != 1.0 or max_tokens != DEFAULT_MAX_TOKENS
        ):
            logger.warning(
                "AgentSDKClient ignores max_tokens/temperature "
                "(got max_tokens=%s, temperature=%s); the SDK uses session "
                "defaults. Use AnthropicAPIClient if you need to control them.",
                max_tokens, temperature,
            )
            self._warned_ignored_params = True

        prompt = _messages_to_prompt(messages)
        system_bytes = len(system.encode("utf-8")) if system else 0
        logger.info(
            "SDK call start: model=%s system=%d bytes prompt=%d bytes",
            model, system_bytes, len(prompt.encode("utf-8")),
        )

        # The timeout/retry knobs are read at call time so a caller (or test)
        # can patch the module globals. prompt_file_fallback stays off: Read
        # is deliberately disallowed on this untrusted-text path.
        try:
            result = await run_agent(
                prompt,
                system_prompt=system + _NO_TOOLS_SUFFIX,
                allowed_tools=[],
                disallowed_tools=_DISALLOWED_TOOLS,  # see _DISALLOWED_TOOLS note
                max_turns=_SDK_MAX_TURNS,
                model=model,
                timeout_s=_SDK_CALL_TIMEOUT_S,
                max_attempts=_SDK_MAX_ATTEMPTS,
                retry_backoff_s=_SDK_RETRY_BACKOFF_S,
                prompt_file_fallback=False,
                label="model_client",
            )
        except AgentRunError as e:
            raise SDKQueryError(
                f"claude_agent_sdk.query() {e.reason} after {e.attempts} attempts",
                stderr_tail=e.stderr_tail,
            ) from (e.__cause__ or e)
        return ModelResponse(content=result.text)


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
        # Concatenate every text block, matching AgentSDKClient's behavior — a
        # response whose text is split around thinking/citation/search blocks
        # must not be truncated to its first segment. An empty content list
        # (tool-only turn, refusal, non-text stop) yields "" rather than
        # IndexError, keeping a recoverable empty reply from becoming a total
        # extraction/routing failure. `.strip()` also mirrors the SDK path.
        parts = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        return ModelResponse(content="".join(parts).strip())


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
