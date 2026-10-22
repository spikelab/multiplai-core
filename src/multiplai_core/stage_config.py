"""Helpers for a pipeline's per-stage config: ``<out>/<name>.yaml`` over
``multiplai.conf`` over the defaults.

Copied from the review and codebase-walkthrough pipelines' ``config.py``. Each
pipeline keeps its own config dataclass and its own keys; what they shared is
here: reading a ``multiplai.conf`` value, reading the YAML file, coercing an
integer, and checking an effort name. A pipeline's ``load_config`` applies its
sources in this order, each overriding the last:

1. its defaults;
2. ``conf_value(load_multiplai_conf(), "<pipeline>_<key>")``;
3. ``read_yaml_mapping(<out>/<name>.yaml)``;

then drops an unknown effort with ``check_effort``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .env import KNOWN_EFFORTS

log = logging.getLogger(__name__)


def conf_value(conf: dict, key: str) -> str | None:
    """*key* (or its upper-case spelling) from a ``multiplai.conf`` dict, stripped.

    ``None`` when absent, not a string, or blank.
    """
    for k in (key, key.upper()):
        value = conf.get(k)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def read_yaml_mapping(path: str | os.PathLike[str] | None) -> dict:
    """The YAML mapping in *path*, or ``{}``.

    A missing file is ``{}`` silently: most runs have none. A file that cannot
    be read, does not parse, or holds something other than a mapping is ``{}``
    with a warning naming it.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    import yaml  # deferred: a bare `import multiplai_core` stays light for hooks

    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
        log.warning("Ignoring unparseable %s: %s", p, e)
        return {}
    if not isinstance(data, dict):
        log.warning("Ignoring %s: expected a mapping", p)
        return {}
    return data


def coerce_int(data: dict, key: str, path: Any, minimum: int = 1) -> int | None:
    """``data[key]`` as an integer no lower than *minimum*, or ``None``.

    ``None`` when the key is absent or null, and also (with a warning naming
    *path*) when the value is not an integer, so the caller keeps its default.
    A value below *minimum* is raised to it.
    """
    value = data.get(key)
    if value is None:
        return None
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        log.warning("Ignoring %s=%r in %s (want an integer)", key, value, path)
        return None


def check_effort(value: str | None) -> str | None:
    """*value* unchanged when it names a known effort (any case), else ``None``.

    An empty value is ``None`` silently; an unknown one is ``None`` with a
    warning listing the known efforts.
    """
    if not value:
        return None
    if value.lower() not in KNOWN_EFFORTS:
        log.warning(
            "Ignoring unknown effort %r (expected one of %s)",
            value,
            ", ".join(sorted(KNOWN_EFFORTS)),
        )
        return None
    return value
