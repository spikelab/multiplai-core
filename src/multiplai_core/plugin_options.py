"""Read Claude Code plugin options under the name the harness exports.

Claude Code delivers a plugin's ``userConfig`` values to **hook processes** as
environment variables named ``CLAUDE_PLUGIN_OPTION_<KEY>``, where ``<KEY>`` is
the option key **uppercased** — see the
`plugins reference <https://code.claude.com/docs/en/plugins-reference.md>`_.

Every accessor here takes the **bare option name** as it appears in
``plugin.json`` (``option("enable_skills")``), never a full variable name.
Uppercasing happens in exactly one place, so a call site cannot get the case
wrong — which is the whole point of this module. Reading the variable under the
key's own (lowercase) spelling always misses, and misses *silently*: the option
falls back to its default and the feature simply never runs.

There is deliberately **no lowercase fallback**. Accepting both cases would
keep a dead name alive as though it meant something.

Parsing is tolerant by design: a malformed value logs a warning and yields the
caller's default. These accessors run inside hooks, and a bad config value must
never crash one.
"""

import logging
import os

logger = logging.getLogger(__name__)

__all__ = [
    "OPTION_PREFIX",
    "option",
    "option_bool",
    "option_float",
    "option_int",
    "option_present",
    "option_var",
]

OPTION_PREFIX = "CLAUDE_PLUGIN_OPTION_"

_TRUE = frozenset({"true", "1", "yes", "on"})
_FALSE = frozenset({"false", "0", "no", "off"})


def option_var(name: str) -> str:
    """Return the environment-variable name the harness exports for *name*.

    ``option_var("enable_skills") == "CLAUDE_PLUGIN_OPTION_ENABLE_SKILLS"``.
    Use this when you need the variable itself (to set it in a test harness or
    a subprocess environment) rather than its value.
    """
    return f"{OPTION_PREFIX}{name.upper()}"


def option_present(name: str) -> bool:
    """Whether the harness delivered a non-empty value for *name*.

    Distinguishes "the user configured this" from "this fell back to its
    default" — the distinction that made this bug class invisible for eight
    days, since a dead option and a deliberately-off option look identical.
    """
    return bool(os.environ.get(option_var(name), "").strip())


def option(name: str, default: str = "") -> str:
    """Return option *name* as a string, or *default* when unset or blank.

    A value that is empty after stripping is treated as unset, matching the
    path-resolver cascade in :mod:`multiplai_core.paths`.
    """
    value = os.environ.get(option_var(name), "").strip()
    return value if value else default


def option_bool(name: str, default: bool) -> bool:
    """Return option *name* as a bool, or *default* when unset or malformed.

    Accepts ``true/1/yes/on`` and ``false/0/no/off``, case-insensitively.
    Anything else warns and yields *default*.
    """
    raw = option(name)
    if not raw:
        return default
    lowered = raw.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    logger.warning(
        "Malformed plugin option %s=%r; using default %s", name, raw, default
    )
    return default


def option_int(name: str, default: int) -> int:
    """Return option *name* as an int, or *default* when unset or malformed."""
    raw = option(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Malformed plugin option %s=%r; using default %d", name, raw, default
        )
        return default


def option_float(name: str, default: float) -> float:
    """Return option *name* as a float, or *default* when unset or malformed."""
    raw = option(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "Malformed plugin option %s=%r; using default %s", name, raw, default
        )
        return default
