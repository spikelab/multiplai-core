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

Parsing is tolerant by design: a malformed *value* logs a warning and yields
the caller's default. These accessors run inside hooks, and a bad config value
must never crash one. A malformed *key* is different — it is developer error
in code, not user config, and it raises ``ValueError``: a key that cannot
round-trip through an environment-variable name (``-``, ``.``, spaces) would
otherwise silently read the default forever. Tests catch the raise; the hook
entry points already fail open.
"""

import logging
import os
import re

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

# The shapes that survive the plugin.json key -> uppercase -> env var
# round-trip. Anything else (a `-`, a `.`, a leading digit) builds a name the
# harness never exports, so every read silently returns the default forever.
_VALID_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def option_var(name: str) -> str:
    """Return the environment-variable name the harness exports for *name*.

    ``option_var("enable_skills") == "CLAUDE_PLUGIN_OPTION_ENABLE_SKILLS"``.
    Use this when you need the variable itself (to set it in a test harness or
    a subprocess environment) rather than its value.

    Raises:
        ValueError: *name* does not match ``[A-Za-z_][A-Za-z0-9_]*`` and so
            cannot name an option the harness delivers. Developer error, not
            user config — see the module docstring.
    """
    if not _VALID_KEY.fullmatch(name):
        raise ValueError(
            f"invalid plugin option key {name!r}: must match "
            f"[A-Za-z_][A-Za-z0-9_]* to round-trip through the "
            f"{OPTION_PREFIX}<KEY> environment variable"
        )
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


def _parse_bool(raw: str) -> bool:
    lowered = raw.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError(raw)


def _option_typed(name: str, default, parse):
    """Shared skeleton for the typed accessors.

    One home for the malformed-value policy (warn, yield the default) so a
    change to it cannot reach two of the three accessors and miss the third.
    """
    raw = option(name)
    if not raw:
        return default
    try:
        return parse(raw)
    except ValueError:
        logger.warning(
            "Malformed plugin option %s=%r; using default %s", name, raw, default
        )
        return default


def option_bool(name: str, default: bool) -> bool:
    """Return option *name* as a bool, or *default* when unset or malformed.

    Accepts ``true/1/yes/on`` and ``false/0/no/off``, case-insensitively.
    Anything else warns and yields *default*.
    """
    return _option_typed(name, default, _parse_bool)


def option_int(name: str, default: int) -> int:
    """Return option *name* as an int, or *default* when unset or malformed."""
    return _option_typed(name, default, int)


def option_float(name: str, default: float) -> float:
    """Return option *name* as a float, or *default* when unset or malformed."""
    return _option_typed(name, default, float)
