"""Model client abstraction for multiplai plugin.

Provides a Protocol-based interface with two implementations:
- AgentSDKClient: uses claude_agent_sdk from the host runtime (zero-config)
- AnthropicAPIClient: uses the anthropic PyPI package with an API key

The create_client() factory tries Agent SDK first, falls back to API key.
"""

import inspect
import logging
import os
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol, runtime_checkable

from .agent_runner import (  # noqa: F401 — _summarize_stderr re-exported for compat
    AgentRunError,
    _summarize_stderr,
    run_agent,
)
from .env import DEFAULT_PROVIDER, ModelSpec, parse_model_spec
from .plugin_options import option

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
        effort: str | None = None,
        timeout_s: float | None = None,
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

    def __init__(self, *, component: str = "model_client") -> None:
        # Cost-ledger tag for this client's runs (see agent_runner.run_agent).
        self._component = component
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
        effort: str | None = None,
        timeout_s: float | None = None,
    ) -> ModelResponse:
        """Send a single-turn query via the Agent SDK and return normalized text.

        *effort* is the second axis alongside *model*: forwarded to the SDK only
        when set (``run_agent`` omits the option entirely for ``None``), so an
        older SDK without the parameter keeps working.

        *timeout_s* overrides the per-call hard ceiling for **this call only**;
        ``None`` keeps the module default (``_SDK_CALL_TIMEOUT_S``, from
        ``MULTIPLAI_SDK_CALL_TIMEOUT_S``). A caller that must escalate the
        timeout for one oversized request among several concurrent ones needs
        this: the module global is shared, so patching it under
        ``asyncio.gather`` would change the ceiling for every in-flight call.
        """
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
                effort=effort,
                timeout_s=(
                    _SDK_CALL_TIMEOUT_S if timeout_s is None else timeout_s
                ),
                max_attempts=_SDK_MAX_ATTEMPTS,
                retry_backoff_s=_SDK_RETRY_BACKOFF_S,
                prompt_file_fallback=False,
                label="model_client",
                component=self._component,
            )
        except AgentRunError as e:
            raise SDKQueryError(
                f"claude_agent_sdk.query() {e.reason} after {e.attempts} attempts",
                stderr_tail=e.stderr_tail,
            ) from (e.__cause__ or e)
        return ModelResponse(content=result.text)


# Anthropic will not cache a prefix shorter than its per-model minimum (1024
# tokens for Sonnet/Opus, 2048 for Haiku); a breakpoint below it is ignored,
# not an error. We gate on bytes rather than tokens to avoid a tokenizer
# dependency — ~2 bytes/token is a deliberately conservative floor, so we only
# ever skip the breakpoint on prompts that could not have been cached anyway.
MIN_CACHEABLE_SYSTEM_BYTES = 4096


def cacheable_system(system: str) -> str | list[dict[str, object]]:
    """The `system` argument with a cache breakpoint on long, stable prompts.

    The direct-API path sent `system=` as a bare string, which is never
    cached — every call re-paid full input price for the same prompt. The
    Agent-SDK path caches on its own (measured ~100% hit ratio in the ledger);
    this closes the same gap for the fallback client.

    Short prompts pass through as a plain string: below the model minimum the
    breakpoint buys nothing, and the string form keeps the request identical
    to what it was for every existing caller.
    """
    if not system or len(system.encode("utf-8")) < MIN_CACHEABLE_SYSTEM_BYTES:
        return system
    return [{
        "type": "text",
        "text": system,
        "cache_control": {"type": "ephemeral"},
    }]


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
                "Set the anthropic_api_key plugin option or pass api_key directly."
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
        effort: str | None = None,
        timeout_s: float | None = None,
    ) -> ModelResponse:
        """Send a query via the Anthropic API and return a normalized response.

        *effort* is accepted for interface parity and ignored: reasoning effort
        is a Claude Code/Agent-SDK session knob, not a Messages API parameter.

        *timeout_s* is likewise accepted for parity and ignored: it is the
        Agent-SDK path's guard against a wedged CLI subprocess, and the HTTP
        client here has its own transport timeouts. Ignoring it keeps a caller
        that escalates the timeout for one large request working against either
        backend instead of raising ``TypeError`` on the fallback client.
        """
        if effort is not None:
            logger.debug("AnthropicAPIClient ignores effort=%s (not a Messages API param)", effort)
        if timeout_s is not None:
            logger.debug("AnthropicAPIClient ignores timeout_s=%s (SDK-path guard)", timeout_s)
        client = self._ensure_client()
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=cacheable_system(system),
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
        key = option("anthropic_api_key")
        if key:
            return "AnthropicAPIClient"
        return "none (no SDK or API key)"


async def create_client(
    *, api_key: str | None = None, component: str = "model_client"
) -> ModelClient:
    """Create a model client. Tries Agent SDK first, falls back to API key.

    Args:
        api_key: Optional API key override. If not provided, reads the
                 ``anthropic_api_key`` plugin option.
        component: Cost-ledger tag for this client's runs (SDK backend only;
                 e.g. "extraction", "dream").

    Returns:
        A ModelClient instance.

    Raises:
        RuntimeError: If neither Agent SDK nor API key is available.
    """
    try:
        client = AgentSDKClient(component=component)
        logger.info("Model client: Agent SDK selected (zero-config)")
        return client
    except ImportError:
        pass

    # Fall back to API key
    key = api_key or option("anthropic_api_key")
    if not key:
        raise RuntimeError(
            "Neither the Agent SDK nor an API key is available. "
            "Install claude_agent_sdk or set the anthropic_api_key plugin option."
        )

    logger.warning("Model client: Falling back to Anthropic API key authentication")
    return AnthropicAPIClient(key)


# --------------------------------------------------------------------------
# Provider seam
# --------------------------------------------------------------------------
# Reviewer panels want models from *different families*, because Claude and GPT
# reviewers empirically find largely disjoint error sets — a same-family panel
# mostly re-finds the same things. The tier/ceiling machinery in env.py only
# ranks the Claude family, so cross-vendor support cannot be a new branch inside
# create_client(); it has to be a registry that an out-of-tree backend can join.
#
# This is the seam only. No non-Anthropic backend ships here: choosing a
# provider means choosing whose API key and whose bill, which is a human
# decision, not a default. Registering one is three lines from the outside.

ProviderFactory = Callable[..., "ModelClient | Awaitable[ModelClient]"]


class UnknownProviderError(RuntimeError):
    """Raised when a :class:`ModelSpec` names a provider nobody registered."""


_PROVIDERS: dict[str, ProviderFactory] = {}


async def _anthropic_factory(spec: ModelSpec, *, api_key: str | None = None,
                             component: str = "model_client") -> ModelClient:
    """Built-in provider: the existing SDK-then-API-key ladder, unchanged.

    The model in *spec* is deliberately not baked into the client — both
    Anthropic clients take ``model=`` per :meth:`ModelClient.query` call.
    """
    logger.debug("Anthropic provider serving %s", spec.model)
    return await create_client(api_key=api_key, component=component)


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register *factory* as the client builder for provider *name*.

    The factory is called as ``factory(spec, api_key=..., component=...)`` and
    may be sync or async; it must return an object satisfying
    :class:`ModelClient`. Re-registering a name replaces it (so a test can swap
    in a stub and put the original back).
    """
    key = name.strip().lower()
    if not key:
        raise ValueError("provider name is empty")
    if key in _PROVIDERS:
        logger.info("Replacing already-registered provider %r", key)
    _PROVIDERS[key] = factory


def unregister_provider(name: str) -> None:
    """Remove a provider registration. Missing names are a no-op."""
    _PROVIDERS.pop(name.strip().lower(), None)


def registered_providers() -> list[str]:
    """Sorted list of provider names currently registered."""
    return sorted(_PROVIDERS)


async def create_client_for(
    spec: ModelSpec | str,
    *,
    api_key: str | None = None,
    component: str = "model_client",
) -> ModelClient:
    """Build the client that serves *spec*, honoring its provider.

    Accepts a :class:`ModelSpec` or a ``provider:model`` / bare-model string.
    A bare string keeps the Anthropic default, so this is a drop-in for
    :func:`create_client` at any call site that has a model ID in hand.

    Raises:
        UnknownProviderError: the spec names a provider with no registered
            factory — with the registered names in the message, because the
            usual cause is a config typo or a backend nobody installed.
    """
    if isinstance(spec, str):
        spec = parse_model_spec(spec)
    factory = _PROVIDERS.get(spec.provider)
    if factory is None:
        raise UnknownProviderError(
            f"No backend registered for provider {spec.provider!r} "
            f"(from model spec {spec.qualified!r}). "
            f"Registered: {', '.join(registered_providers()) or '(none)'}. "
            "Call register_provider() to add one."
        )
    client = factory(spec, api_key=api_key, component=component)
    if inspect.isawaitable(client):
        client = await client
    return client


register_provider(DEFAULT_PROVIDER, _anthropic_factory)
