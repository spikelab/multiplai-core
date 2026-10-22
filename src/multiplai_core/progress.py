"""Tailable progress file: one timestamped line per event.

Copied from the review and codebase-walkthrough pipelines' ``progress.py``. The
session running those skills tails the file with a bounded loop until a line
matches ``^\\[[^]]*\\] (DONE|FAILED)``, so the line format is a contract:

    [2026-10-11T09:30:00Z] STAGE find: 5 finders
    [2026-10-11T09:41:12Z] DONE 3 findings

The ``STARTED`` line differs per pipeline (what it names), so each pipeline
adds its own ``started`` in a subclass and writes it with :meth:`line`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path


class ProgressWriter:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _now(self) -> str:
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    def line(self, text: str) -> None:
        """Append ``[<UTC time>Z] <text>``, creating the directory when missing."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"[{self._now()}] {text}\n")

    def stage(self, stage: str, detail: str = "") -> None:
        self.line(f"STAGE {stage}" + (f": {detail}" if detail else ""))

    def done(self, detail: str) -> None:
        self.line(f"DONE {detail}")

    def failed(self, detail: str) -> None:
        self.line(f"FAILED {detail}")
