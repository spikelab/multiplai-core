"""Configuration utilities for multiplai plugin.

Loads YAML/JSON config files from the plugin's data directory via path resolver.
Exports shared constants and reusable file-handling helpers used across scripts.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Memory template filenames shipped with the plugin.  Used by both
# setup_check.py and setup_write.py to iterate over the known set.
TEMPLATE_FILENAMES: list[str] = ["me.md", "technical-pref.md", "preferences.md"]

# Expected memory filenames for health checks and audit.
MEMORY_FILENAMES: list[str] = ["me.md", "technical-pref.md", "preferences.md"]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning an empty dict if missing or unreadable.

    Imports ``pyyaml`` on demand so the module can be imported before venv
    bootstrap completes.
    """
    import yaml

    if not path.exists():
        return {}
    try:
        with path.open() as f:
            return yaml.safe_load(f) or {}
    except Exception:
        logger.warning("Could not read %s, starting fresh", path.name)
        return {}


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write *data* to a YAML file, creating parent directories as needed.

    Imports ``pyyaml`` on demand so the module can be imported before venv
    bootstrap completes.
    """
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(data, f, default_flow_style=False)


def read_memory_files(memory_dir: Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    """Read all ``.md`` files from *memory_dir*, returning ``{name: content}``.

    Files whose names appear in *exclude* are skipped.  Missing or
    unreadable files are silently skipped.
    """
    exclude = exclude or set()
    files: dict[str, str] = {}
    if not memory_dir.exists():
        return files
    for md_file in sorted(memory_dir.glob("*.md")):
        if md_file.name in exclude:
            continue
        try:
            files[md_file.name] = md_file.read_text()
        except Exception:
            logger.warning("Failed to read memory file: %s", md_file.name)
    return files


def read_session_state(data_dir: Path) -> dict[str, Any] | None:
    """Read ``session_state.json`` from *data_dir*.

    Returns the parsed dict, or ``None`` if the file is missing or
    unreadable.  Used by both ``session_stop`` and ``session_end``.
    """
    state_file = data_dir / "session_state.json"
    if not state_file.exists():
        return None
    try:
        return json.loads(state_file.read_text())
    except Exception:
        return None


def write_session_state(data_dir: Path, state: dict[str, Any]) -> bool:
    """Atomically write ``session_state.json`` to *data_dir*.

    Writes to a temp file then ``os.replace`` so a crash mid-write never
    leaves a half-written state file. Returns ``True`` on success,
    ``False`` on any OS error (callers treat this as best-effort).
    """
    state_file = data_dir / "session_state.json"
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        tmp = state_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2))
        os.replace(str(tmp), str(state_file))
        return True
    except OSError:
        logger.warning("Could not write %s", state_file.name)
        return False


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a configuration file (JSON or YAML).

    Args:
        config_path: Absolute path to the config file.

    Returns:
        Parsed configuration as a dict.

    Raises:
        FileNotFoundError: With a descriptive message including the expected path.
        ValueError: If the file format is not supported.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}. "
            f"Ensure the file exists in the plugin directory."
        )

    suffix = config_path.suffix.lower()

    if suffix == ".json":
        return json.loads(config_path.read_text())
    elif suffix in (".yaml", ".yml"):
        import yaml
        with config_path.open() as f:
            return yaml.safe_load(f) or {}
    else:
        raise ValueError(
            f"Unsupported config file format: {suffix}. "
            f"Use .json or .yaml/.yml."
        )
