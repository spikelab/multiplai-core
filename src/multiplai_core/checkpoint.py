"""Atomic JSON checkpoint for a pydantic model: written after every stage, read on resume.

Copied from the review and codebase-walkthrough pipelines' ``state.py``, which
differed only in the model class and the file name. Those two are now
arguments.

Nothing here imports pydantic: *model* is anything with ``model_dump_json``,
and *cls* anything with ``model_validate_json``.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


def save_model(model: Any, dir: str | os.PathLike[str], filename: str) -> Path:
    """Write *model* as JSON to ``dir/filename`` and return that path.

    The JSON goes to a temp file in the same directory, then ``os.replace``
    moves it over the old one, so a kill mid-write leaves the previous
    checkpoint whole. The directory is created when missing.
    """
    target_dir = Path(dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
    fd, tmp = tempfile.mkstemp(
        dir=target_dir, prefix=f".{Path(filename).stem}-", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(model.model_dump_json(indent=2))
        os.replace(tmp, path)
    except BaseException:
        # Leave no stray temp file behind; the previous checkpoint is untouched.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def load_model(path: str | os.PathLike[str], cls: type[T]) -> T | None:
    """The model saved at *path*, or ``None`` (logged) when the file is missing or invalid."""
    p = Path(path)
    if not p.is_file():
        log.info("No checkpoint at %s", p)
        return None
    try:
        return cls.model_validate_json(p.read_text(encoding="utf-8"))  # type: ignore[attr-defined]
    except (OSError, ValueError) as e:  # pydantic's ValidationError is a ValueError
        log.error("Unreadable checkpoint %s: %s", p, e)
        return None
