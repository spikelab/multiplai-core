"""multiplai-core — shared library for the Multiplai Claude Code plugins.

One source of truth for path resolution, config loading, logging, and the
model client. Every Multiplai plugin imports from here instead of vendoring
its own copy.
"""

from .agent_runner import (
    MAX_PROMPT_BYTES,
    AgentRunError,
    AgentRunResult,
    AgentRunTimeout,
    AgentUsage,
    run_agent,
)
from .aio import hard_timeout, swallow_task_result
from .config import (
    load_config,
    load_yaml,
    read_memory_files,
    read_session_state,
    save_yaml,
    write_session_state,
)
from .env import (
    env_candidates,
    find_project_root,
    load_env,
    load_multiplai_conf,
    resolve_effort,
    resolve_model,
)
from .text import extract_json
from .log_utils import (
    log_event,
    resolve_level,
    retention_days,
    setup_logging,
)
from .model_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    AgentSDKClient,
    AnthropicAPIClient,
    ModelClient,
    ModelResponse,
    SDKQueryError,
    create_client,
    detect_client_type,
)
# NB: do not re-export the `paths` singleton here — binding the name `paths`
# in the package namespace would shadow the `multiplai_core.paths` submodule.
# Import the singleton explicitly via `from multiplai_core.paths import paths`.
from .paths import Paths, get_paths

__version__ = "0.5.2"

__all__ = [
    # agent runner
    "run_agent",
    "AgentRunResult",
    "AgentRunError",
    "AgentRunTimeout",
    "AgentUsage",
    "MAX_PROMPT_BYTES",
    # paths
    "Paths",
    "get_paths",
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
    # text
    "extract_json",
    "__version__",
]
