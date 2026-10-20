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

Fallback to the saved options
-----------------------------

The harness exports those variables to **hooks only**. A script that a skill
runs through Bash gets none of them, so before this fallback every such script
silently ran on defaults (``enable_skills`` read false while the catalogs were
regenerated; the health check reported a router the hooks were not running).

So when the variable is absent or blank, the accessors read the value Claude
Code saved for this plugin in the **user settings file**,
``$CLAUDE_CONFIG_DIR/settings.json`` (``~/.claude/settings.json`` when the
variable is unset), at ``pluginConfigs["<plugin>@<marketplace>"].options``.
The value is stringified the way the harness stringifies it for a hook
(JavaScript ``String(value)``: ``true`` -> ``"true"``, ``30.0`` -> ``"30"``,
``["a", "b"]`` -> ``"a,b"``), so a hook and a skill script read the same string.
A ``null`` or object value counts as unset.

Precedence: the environment variable always wins; the settings file is only
consulted when it is absent or blank.

What is deliberately **not** read:

- **Sensitive options.** Claude Code keeps those in secure storage (the
  keychain, or its credentials file), never in settings. They fall back to the
  default outside a hook.
- **Project and local settings.** Claude Code does not read ``pluginConfigs``
  from them either.
- **Managed (policy) and ``--settings`` sources**, which Claude Code layers over
  the user file. Their layout is platform-specific and a ``--settings`` file is
  invisible to a child process; a hook still receives the effective value
  through its environment.

Which plugin is this? The process's plugin directory is
``$CLAUDE_PLUGIN_ROOT`` when set (hooks), otherwise the directory of the entry
script (``__main__.__file__``, else ``sys.argv[0]``) -- skills invoke scripts by
their path inside the installed plugin. A path under ``site-packages`` is never
used, so ``python -m`` runs identify nothing. That path names a plugin only by
one of two records Claude Code itself keeps under the config dir:

1. ``plugins/installed_plugins.json`` -- the entry whose ``installPath`` is the
   path or one of its parents; its key is exactly the ``pluginConfigs`` key.
2. The cache layout ``plugins/cache/<marketplace>/<plugin>/<version>/`` -- which
   also covers a session still running a superseded version directory.

A checkout outside the config dir (a development clone, a test run) is
**never** matched by plugin name, so the user's real saved options cannot leak
into it; it keeps reading defaults. Settings and plugin identity are parsed once
per process. Nothing here raises on a missing or malformed file: it logs at
debug and the default is used.
"""

import functools
import json
import logging
import os
import re
import sys
from pathlib import Path

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
    return bool(_raw(name))


def option(name: str, default: str = "") -> str:
    """Return option *name* as a string, or *default* when unset or blank.

    A value that is empty after stripping is treated as unset, matching the
    path-resolver cascade in :mod:`multiplai_core.paths`.
    """
    value = _raw(name)
    return value if value else default


def _raw(name: str) -> str:
    """The stripped value for *name*: the hook's environment variable when it
    is non-blank, otherwise the plugin's saved option (see the module
    docstring), otherwise ``""``."""
    value = os.environ.get(option_var(name), "").strip()
    if value:
        return value
    return _saved_options().get(name, "")


def _reset_cache() -> None:
    """Forget the parsed settings and plugin identity (tests use this)."""
    _saved_options.cache_clear()


def _config_dir() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")


def _read_json(path: Path):
    """Parsed JSON at *path*, or ``None`` (logged at debug) when it is missing
    or unreadable. Never raises."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.debug("plugin options: %s does not exist", path)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        logger.debug("plugin options: cannot read settings file %s: %s", path, exc)
    return None


def _resolve(path: Path) -> Path | None:
    try:
        return path.expanduser().resolve()
    except (OSError, RuntimeError):
        return None


def _plugin_path() -> Path | None:
    """A path inside this process's plugin, or ``None``."""
    root = os.environ.get("CLAUDE_PLUGIN_ROOT", "").strip()
    if root:
        return _resolve(Path(root))
    main = getattr(sys.modules.get("__main__"), "__file__", None)
    entry = main or (sys.argv[0] if sys.argv and sys.argv[0] else None)
    if not entry:
        return None
    path = _resolve(Path(entry))
    if path is None or {"site-packages", "dist-packages"} & set(path.parts):
        return None
    return path.parent


def _plugin_id(config: Path, where: Path) -> str | None:
    """The ``<plugin>@<marketplace>`` key *where* belongs to, from Claude
    Code's own records under *config* only (see the module docstring)."""
    installed = _read_json(config / "plugins" / "installed_plugins.json")
    plugins = installed.get("plugins") if isinstance(installed, dict) else None
    best: tuple[int, str] | None = None
    if isinstance(plugins, dict):
        for key, entries in plugins.items():
            for entry in entries if isinstance(entries, list) else []:
                install = entry.get("installPath") if isinstance(entry, dict) else None
                if not isinstance(install, str) or not install:
                    continue
                base = _resolve(Path(install))
                if base is not None and (where == base or base in where.parents):
                    depth = len(base.parts)
                    if best is None or depth > best[0]:
                        best = (depth, key)
    if best is not None:
        return best[1]

    cache = _resolve(config / "plugins" / "cache")
    if cache is not None and cache in where.parents:
        parts = where.relative_to(cache).parts
        if len(parts) >= 3:  # <marketplace>/<plugin>/<version>[/...]
            return f"{parts[1]}@{parts[0]}"
    logger.debug("plugin options: %s is not inside an installed plugin", where)
    return None


def _as_harness_string(value) -> str | None:
    """*value* stringified as the harness does for a hook (JS ``String()``),
    or ``None`` for a value that cannot mean anything as an option."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e21:
            return str(int(value))
        return repr(value)
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return ",".join(value)
    return None


@functools.lru_cache(maxsize=1)
def _saved_options() -> dict[str, str]:
    """This plugin's saved options as stripped, non-empty strings. Parsed
    once per process; empty when anything is missing or malformed."""
    where = _plugin_path()
    if where is None:
        return {}
    config = _resolve(_config_dir())
    if config is None:
        return {}
    plugin_id = _plugin_id(config, where)
    if plugin_id is None:
        return {}
    settings = _read_json(config / "settings.json")
    configs = settings.get("pluginConfigs") if isinstance(settings, dict) else None
    block = configs.get(plugin_id) if isinstance(configs, dict) else None
    saved = block.get("options") if isinstance(block, dict) else None
    if not isinstance(saved, dict):
        logger.debug("plugin options: no saved options for %s", plugin_id)
        return {}
    out = {}
    for key, value in saved.items():
        text = _as_harness_string(value)
        if text is not None and text.strip():
            out[key] = text.strip()
    return out


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
