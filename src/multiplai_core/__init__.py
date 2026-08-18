"""multiplai-core — shared library for the Multiplai Claude Code plugins.

One source of truth for path resolution, config loading, logging, and the
model client. Every Multiplai plugin imports from here instead of vendoring
its own copy.

The asyncio-heavy modules (``agent_runner``, ``aio``, ``model_client``) are
imported lazily via PEP 562: a hook that only needs a path, an option, or a
``log_event`` no longer pays for ``asyncio`` at import time. Every public
name still resolves through ``from multiplai_core import X`` exactly as
before — the import just happens on first use.
"""

from typing import TYPE_CHECKING

from .banks import (
    BANKS_FILENAME,
    BANK_MODES,
    DEFAULT_SHARED_MODE,
    PERSONAL_BANK,
    PERSONAL_MODE,
    MemoryBank,
    bank_ref,
    is_bank_name,
    load_banks,
    parse_bank_ref,
    personal_bank,
    split_bank_ref,
)
from .config import (
    load_config,
    load_yaml,
    read_memory_files,
    read_session_state,
    save_yaml,
    write_session_state,
)
from .env import (
    CURRENT_MODEL,
    DEFAULT_PROVIDER,
    EFFORT_TIERS,
    KNOWN_EFFORTS,
    ModelSpec,
    env_candidates,
    find_project_root,
    load_env,
    load_multiplai_conf,
    parse_model_spec,
    pick_effort,
    pick_model,
    pick_model_spec,
    resolve_effort,
    resolve_model,
)
from .text import extract_json
from .log_utils import (
    HookRun,
    hook_run,
    log_event,
    resolve_level,
    retention_days,
    setup_logging,
)
# NB: do not re-export the `paths` singleton here — binding the name `paths`
# in the package namespace would shadow the `multiplai_core.paths` submodule.
# Import the singleton explicitly via `from multiplai_core.paths import paths`.
from .paths import Paths, get_paths
from .plugin_options import (
    OPTION_PREFIX,
    option,
    option_bool,
    option_float,
    option_int,
    option_present,
    option_var,
)
from .untrusted import (
    bracket_notice,
    contains_injection,
    defang,
    fence,
    markdown_notice,
)

if TYPE_CHECKING:  # pragma: no cover — for type checkers only; runtime is lazy
    from .agent_runner import (
        MAX_PROMPT_BYTES,
        AgentRunError,
        AgentRunResult,
        AgentRunTimeout,
        TOOL_UNIVERSE,
        AgentUsage,
        deny_list,
        run_agent,
    )
    from .aio import hard_timeout, swallow_task_result
    from .model_client import (
        DEFAULT_MAX_TOKENS,
        DEFAULT_MODEL,
        AgentSDKClient,
        AnthropicAPIClient,
        ModelClient,
        ModelResponse,
        SDKQueryError,
        UnknownProviderError,
        create_client,
        create_client_for,
        detect_client_type,
        register_provider,
        registered_providers,
        unregister_provider,
    )

# Which lazily-imported submodule serves each deferred public name. The module
# names themselves are included so `multiplai_core.agent_runner` keeps working
# after a bare `import multiplai_core` (the eager `from .agent_runner import`
# used to bind the submodule attribute as a side effect).
_LAZY_ATTRS: dict[str, str] = {
    **dict.fromkeys(
        (
            "run_agent", "AgentRunResult", "AgentRunError", "AgentRunTimeout",
            "AgentUsage", "MAX_PROMPT_BYTES", "TOOL_UNIVERSE", "deny_list",
            "agent_runner",
        ),
        "agent_runner",
    ),
    **dict.fromkeys(("hard_timeout", "swallow_task_result", "aio"), "aio"),
    **dict.fromkeys(
        (
            "create_client", "detect_client_type", "ModelClient",
            "ModelResponse", "AgentSDKClient", "AnthropicAPIClient",
            "SDKQueryError", "DEFAULT_MODEL", "DEFAULT_MAX_TOKENS",
            "create_client_for", "register_provider", "unregister_provider",
            "registered_providers", "UnknownProviderError",
            "model_client",
        ),
        "model_client",
    ),
    "costing": "costing",
}


def __getattr__(name: str):
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(f".{module_name}", __name__)
    value = module if name == module_name else getattr(module, name)
    globals()[name] = value  # cache: __getattr__ runs at most once per name
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_LAZY_ATTRS))


__version__ = "0.13.0"

__all__ = [
    # agent runner
    "run_agent",
    "AgentRunResult",
    "AgentRunError",
    "AgentRunTimeout",
    "AgentUsage",
    "MAX_PROMPT_BYTES",
    "TOOL_UNIVERSE",
    "deny_list",
    # paths
    "Paths",
    "get_paths",
    # memory banks
    "BANKS_FILENAME",
    "BANK_MODES",
    "DEFAULT_SHARED_MODE",
    "PERSONAL_BANK",
    "PERSONAL_MODE",
    "MemoryBank",
    "bank_ref",
    "is_bank_name",
    "load_banks",
    "parse_bank_ref",
    "personal_bank",
    "split_bank_ref",
    # plugin options
    "OPTION_PREFIX",
    "option",
    "option_bool",
    "option_float",
    "option_int",
    "option_present",
    "option_var",
    # config
    "load_config",
    "load_yaml",
    "save_yaml",
    "read_memory_files",
    "read_session_state",
    "write_session_state",
    # logging
    "setup_logging",
    "log_event",
    "hook_run",
    "HookRun",
    "resolve_level",
    "retention_days",
    # model client
    "create_client",
    "detect_client_type",
    "ModelClient",
    "ModelResponse",
    "AgentSDKClient",
    "AnthropicAPIClient",
    "SDKQueryError",
    "DEFAULT_MODEL",
    "DEFAULT_MAX_TOKENS",
    # provider seam
    "create_client_for",
    "register_provider",
    "unregister_provider",
    "registered_providers",
    "UnknownProviderError",
    "ModelSpec",
    "parse_model_spec",
    "pick_model_spec",
    "DEFAULT_PROVIDER",
    # async helpers
    "hard_timeout",
    "swallow_task_result",
    # env / config loading
    "load_env",
    "env_candidates",
    "find_project_root",
    "load_multiplai_conf",
    "resolve_model",
    "resolve_effort",
    "pick_model",
    "pick_effort",
    "EFFORT_TIERS",
    "KNOWN_EFFORTS",
    "CURRENT_MODEL",
    # text
    "extract_json",
    # untrusted content
    "defang",
    "fence",
    "contains_injection",
    "markdown_notice",
    "bracket_notice",
    "__version__",
]
