"""One line per run of a multiplai pipeline, in one file every viewer reads.

`<data_dir>/runs.jsonl` gets one JSON line when a run of review, buildme or
pm-spec ends, however it ends: succeeded, parked, failed, stopped by a limit,
or waiting on a person. The file is only ever appended to. Each line is
written by one `write` call on a descriptor opened with `O_APPEND`, under an
exclusive `flock` on the file, so two runs ending at once never interleave
their lines on one machine.

Every line carries `row_version`. A reader keeps only the versions it knows
(`read_rows`) and counts the rest, so a line written by a newer producer is
never misread by an older reader. `producer` says which skill and plugin
version wrote it (`"multiplai-dev:buildme 0.155.2"`), so runs can be grouped
by the version that produced them.

A row (`make_row`):

    {"row_version": 1,
     "producer": "multiplai-dev:review 0.47.0",
     "kind": "review" | "build" | "epic" | "pm-spec",
     "target": {"repo": "Dolcebot", "branch": "...", "change": "...", "pr": 88, ...},
     "folder": "<path under data_dir>" | null,
     "requested_by": null | {"branch": "...", "child": "..."},
     "started_at": "2026-11-04T01:20:57Z" | null,
     "finished_at": "2026-11-04T03:02:11Z",
     "outcome": "succeeded" | "parked" | "failed" | "stopped" | "waiting",
     "cause": "merge-conflict" | ... | null,
     "spend_usd": 12.5 | null,
     "run": {...}}

`target` and `run` belong to the producer; this module checks only that they
are mappings. `requested_by` is null for a run a person started, and names the
change (and epic child) for a run another pipeline started, such as a review
buildme asked for.

A change's own machine data lives under `changes_root()`:
`<data_dir>/changes/<repo folder>/<branch>/`, the branch being the change's
feature branch (`<TICKET>-<change>`), which exists before the change's first
file is written. `folder_name` cleans a name for one path part, the same way
review names its `reviews/<repo>/` folders.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ROW_VERSION = 1
RUNS_FILE = "runs.jsonl"
CHANGES_DIR = "changes"
KINDS = ("review", "build", "epic", "pm-spec")
OUTCOMES = ("succeeded", "parked", "failed", "stopped", "waiting")


def _data_dir() -> Path:
    from .paths import get_paths

    return Path(get_paths().data_dir)


def runs_path() -> Path:
    """`<data_dir>/runs.jsonl`."""
    return _data_dir() / RUNS_FILE


def changes_root() -> Path:
    """`<data_dir>/changes`: one folder per repository, one per change's branch under it."""
    return _data_dir() / CHANGES_DIR


def folder_name(text: str) -> str:
    """*text* as one path part: every run of characters outside `A-Za-z0-9._-`
    becomes `_`, with leading and trailing `_` dropped. Review's `sanitize_slug`."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return cleaned or "unnamed"


def change_folder(repo: str, branch: str) -> Path:
    """`<data_dir>/changes/<repo>/<branch>/`, each part cleaned by `folder_name`."""
    return changes_root() / folder_name(repo) / folder_name(branch)


def producer(plugin_root: Path, skill: str) -> str:
    """`"<plugin name>:<skill> <version>"` from *plugin_root*'s
    `.claude-plugin/plugin.json`; the version is `unknown` when the file
    cannot be read."""
    name, version = Path(plugin_root).name, "unknown"
    try:
        data = json.loads(
            (Path(plugin_root) / ".claude-plugin" / "plugin.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError):
        data = None
    if isinstance(data, dict):
        name = str(data.get("name") or name)
        version = str(data.get("version") or version)
    return f"{name}:{skill} {version}"


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_row(
    *,
    producer: str,
    kind: str,
    target: Mapping[str, Any],
    outcome: str,
    started_at: str | None = None,
    finished_at: str | None = None,
    cause: str | None = None,
    spend_usd: float | None = None,
    folder: str | None = None,
    requested_by: Mapping[str, Any] | None = None,
    run: Mapping[str, Any] | None = None,
) -> dict:
    """One row (module docstring). Raises ValueError for an unknown *kind* or
    *outcome*: a producer that passes one has a bug its tests should catch."""
    if kind not in KINDS:
        raise ValueError(f"unknown run kind {kind!r}; expected one of {KINDS}")
    if outcome not in OUTCOMES:
        raise ValueError(f"unknown outcome {outcome!r}; expected one of {OUTCOMES}")
    return {
        "row_version": ROW_VERSION,
        "producer": producer,
        "kind": kind,
        "target": dict(target),
        "folder": folder,
        "requested_by": dict(requested_by) if requested_by is not None else None,
        "started_at": started_at,
        "finished_at": finished_at or now(),
        "outcome": outcome,
        "cause": cause,
        "spend_usd": spend_usd,
        "run": dict(run or {}),
    }


def append_row(row: Mapping[str, Any], path: Path | None = None) -> bool:
    """Append *row* as one line to *path* (default `runs_path()`).

    Never raises: a failed write is logged and returns False, because a
    record of the run must not fail the run."""
    path = Path(path) if path is not None else runs_path()
    try:
        line = (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                os.write(fd, line)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    except (OSError, TypeError, ValueError) as e:
        log.warning("could not append a run row to %s: %s", path, e)
        return False
    return True


def read_rows(path: Path | None = None) -> tuple[list[dict], int]:
    """Every row of *path* (default `runs_path()`) whose `row_version` this
    module knows, in file order, and how many lines were left out: unparseable
    (such as a line a killed process cut short), not an object, or of another
    `row_version`. A missing file is no rows."""
    path = Path(path) if path is not None else runs_path()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return [], 0
    except OSError as e:
        log.warning("could not read %s: %s", path, e)
        return [], 0
    rows, skipped = [], 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            skipped += 1
            continue
        if not isinstance(row, dict) or row.get("row_version") != ROW_VERSION:
            skipped += 1
            continue
        rows.append(row)
    return rows, skipped
