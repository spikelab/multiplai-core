"""Environment + config loading shared across the Multiplai plugins.

`.env` discovery, `multiplai.conf` parsing, and the model/effort ceiling
resolver were copied (and drifted) across buildme and deep-research; this is
the single source of truth.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

log = logging.getLogger(__name__)


def find_project_root(start: Path | None = None) -> Path | None:
    """Walk upward from *start* looking for the multiplai-kit root.

    A directory qualifies if it contains both a ``.env.example`` AND a
    ``dotfiles/`` directory. Falls back to the first ancestor with a ``.env``.
    """
    current = (start or Path.cwd()).resolve()
    for ancestor in [current, *current.parents]:
        if (ancestor / ".env.example").exists() and (ancestor / "dotfiles").is_dir():
            return ancestor
    for ancestor in [current, *current.parents]:
        if (ancestor / ".env").exists():
            return ancestor
    return None


def env_candidates(start: Path | None = None) -> list[Path]:
    """Ordered ``.env`` locations, most explicit first.

    Covers a plain plugin install with no kit tree: an explicit override
    (``MULTIPLAI_ENV_FILE``), the kit home (``CLAUDE_MULTIPLAI_HOME``), the
    current working directory, and finally the marker/walk-up.
    """
    candidates: list[Path] = []
    explicit = os.environ.get("MULTIPLAI_ENV_FILE")
    if explicit:
        candidates.append(Path(explicit))
    home = os.environ.get("CLAUDE_MULTIPLAI_HOME")
    if home:
        candidates.append(Path(home) / ".env")
    candidates.append(Path.cwd() / ".env")
    root = find_project_root(start)
    if root is not None:
        candidates.append(root / ".env")
    return candidates


def load_env(start: Path | None = None) -> bool:
    """Load ``.env`` into ``os.environ`` from the first candidate that exists.

    Existing environment variables are NOT overridden — explicit env wins.
    Returns True if a file was found and loaded.
    """
    env_file = next((p for p in env_candidates(start) if p.exists()), None)
    if env_file is None:
        log.debug("No .env found in any candidate location — skipping")
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        log.warning(
            "python-dotenv not installed; cannot auto-load %s "
            "(pip install python-dotenv)", env_file,
        )
        return False
    loaded = load_dotenv(env_file, override=False)
    if loaded:
        log.info("Loaded .env from %s", env_file)
    return loaded


def load_multiplai_conf() -> dict:
    """Load ``multiplai.conf`` with optional INI-style section support.

    Returns a dict with global keys at the top level plus a ``_sections`` dict
    for per-skill overrides.
    """
    multiplai_home = os.environ.get("CLAUDE_MULTIPLAI_HOME")
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if multiplai_home:
        conf_path = Path(multiplai_home) / "multiplai.conf"
    elif config_dir:
        conf_path = Path(config_dir).parent / "multiplai.conf"
    else:
        root = find_project_root()
        conf_path = (root / "multiplai.conf") if root else None
    if conf_path is None or not conf_path.exists():
        return {"_sections": {}}

    result: dict[str, str] = {}
    sections: dict[str, dict[str, str]] = {}
    current_section: str | None = None
    for line in conf_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        section_match = re.match(r"^\[([a-zA-Z0-9_-]+)\]\s*$", line)
        if section_match:
            current_section = section_match.group(1)
            sections.setdefault(current_section, {})
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if current_section:
                sections[current_section][key] = value
            else:
                result[key] = value
    result["_sections"] = sections  # type: ignore[assignment]
    return result


_TIERS = {"haiku": 1, "sonnet": 2, "opus": 3}
_EFFORT_TIERS = {"low": 1, "medium": 2, "high": 3, "max": 4}


def _tier(model: str) -> int:
    model_lower = model.lower()
    for name, rank in _TIERS.items():
        if name in model_lower:
            return rank
    return 2  # default to sonnet


def resolve_model(requested: str, ceiling: str | None = None) -> str:
    """Return *requested*, or the ceiling model if requested is above it.

    Ceiling comes from *ceiling* or ``MULTIPLAI_MODEL`` (default sonnet).
    """
    if ceiling is None:
        ceiling = os.environ.get("MULTIPLAI_MODEL", "claude-sonnet-4-6")
    if _tier(requested) > _tier(ceiling):
        log.info("Model ceiling: %s → %s", requested, ceiling)
        return ceiling
    return requested


def _effort_tier(effort: str) -> int:
    return _EFFORT_TIERS.get(effort.lower(), 3)


def resolve_effort(requested: str, ceiling: str | None = None) -> str:
    """Return *requested*, or the ceiling effort if requested is above it."""
    if ceiling is None:
        ceiling = os.environ.get("MULTIPLAI_EFFORT", "high")
    if _effort_tier(requested) > _effort_tier(ceiling):
        log.info("Effort ceiling: %s → %s", requested, ceiling)
        return ceiling
    return requested
