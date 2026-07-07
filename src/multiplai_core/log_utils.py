"""Logging utilities for multiplai plugin.

Standard adopted across every Multiplai plugin:

- UTC ISO-8601 line format with component + session id
- ``MULTIPLAI_DEBUG`` / ``MULTIPLAI_LOG_LEVEL`` env-driven level
- Date-rotated per-component logs with configurable retention
- Shared ``hook-errors.log`` for ERROR+ across all components

On top of the standard, ``log_event()`` writes a curated, human-readable
activity stream (``activity.log``) plus a machine-parseable mirror
(``activity.jsonl``). This is the human-in-the-loop view: one narrative
line per meaningful thing the plugin does (context injected, nudge
fired, diary written, learnings captured, catalog rebuilt). It is
written regardless of log level — it is the signal, not the debug noise.

**File-naming convention (one rule for every log in the directory):**

- ``<name>.log`` — the *current* file (no date suffix).
- On the first write of a new UTC day the current file is rotated to
  ``<name>-YYYY-MM-DD.log`` (date infix *before* the extension). The
  ``<name>.log.YYYY-MM-DD`` form produced by stdlib
  ``TimedRotatingFileHandler`` is rejected — editors don't recognise it
  as a log file — and any such legacy files are migrated to the correct
  form opportunistically.

Retention is governed by ``MULTIPLAI_LOG_RETENTION_DAYS`` (default 7,
``0`` = keep forever). It applies uniformly to every rotated
``<name>-DATE.log`` / ``<name>-DATE.jsonl`` file, including the activity
stream.

All log files live under the plugin data directory via the path resolver.
"""

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

# Default rotated-log retention when MULTIPLAI_LOG_RETENTION_DAYS is unset
# or invalid. Matches the documented logging standard.
_DEFAULT_RETENTION_DAYS = 7

# A trailing ``-YYYY-MM-DD`` before the extension marks a rotated file.
_DATED_RE = re.compile(r"-(\d{4}-\d{2}-\d{2})\.(log|jsonl)$")

# The rejected stdlib form: ``<name>.log.YYYY-MM-DD``.
_REJECTED_RE = re.compile(r"^(?P<base>.+)\.log\.(?P<date>\d{4}-\d{2}-\d{2})$")

# Directory sweep (migrate + prune) runs at most once per process.
_swept = False

# Oversize ceiling for append-only logs (hook-errors.log), per the logging
# standard: "truncated to ~100KB when oversized".
_ERROR_LOG_MAX_BYTES = 100 * 1024


def _truncate_oversized(path: Path, max_bytes: int = _ERROR_LOG_MAX_BYTES) -> None:
    """Truncate an append-only log to its most recent tail when oversized.

    Keeps roughly half of *max_bytes* so truncation runs infrequently.
    Rewrites in place (same inode) so concurrent O_APPEND writers keep
    working; a few lines may interleave during the rewrite — acceptable
    for a best-effort error sink. Never raises.
    """
    try:
        if not path.exists() or path.stat().st_size <= max_bytes:
            return
        keep = max_bytes // 2
        with path.open("r+b") as f:
            f.seek(-keep, os.SEEK_END)
            tail = f.read()
            nl = tail.find(b"\n")
            if nl != -1:
                tail = tail[nl + 1:]
            f.seek(0)
            f.write(b"[truncated: exceeded %d bytes]\n" % max_bytes + tail)
            f.truncate()
    except OSError:
        pass


def _get_logs_dir() -> Path:
    """Get logs directory from path resolver (imported lazily)."""
    from .paths import get_paths
    return _pytest_guard(get_paths().logs_dir())


# Per-process redirect target when the pytest guard trips (one dir, so all
# components in a test process land together).
_pytest_redirect: Path | None = None


def _pytest_guard(logs_dir: Path) -> Path:
    """Never write logs into a real workspace from inside pytest.

    Loggers are typically configured at module import time; pytest imports
    modules during collection, before any fixture (including autouse env
    scrubbers) runs, so a leaked WORKSPACE — or the ``~/.multiplai``
    standalone fallback — would silently route test log writes into real
    logs. Under pytest, any logs dir outside the system temp root is
    redirected to a throwaway temp dir.
    """
    if "PYTEST_CURRENT_TEST" not in os.environ and "pytest" not in sys.modules:
        return logs_dir
    import tempfile
    tmp_root = Path(tempfile.gettempdir()).resolve()
    try:
        resolved = logs_dir.resolve()
    except OSError:
        resolved = logs_dir
    if resolved == tmp_root or tmp_root in resolved.parents:
        return logs_dir
    global _pytest_redirect
    if _pytest_redirect is None:
        _pytest_redirect = Path(tempfile.mkdtemp(prefix="multiplai-pytest-logs-"))
    return _pytest_redirect


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def retention_days() -> int:
    """Resolve rotated-log retention from ``MULTIPLAI_LOG_RETENTION_DAYS``.

    Returns the configured day count, ``0`` for "keep forever", or the
    default (7) when unset or unparseable. Negative values fall back to
    the default.
    """
    raw = os.environ.get("MULTIPLAI_LOG_RETENTION_DAYS", "").strip()
    if not raw:
        return _DEFAULT_RETENTION_DAYS
    try:
        n = int(raw)
    except ValueError:
        return _DEFAULT_RETENTION_DAYS
    return n if n >= 0 else _DEFAULT_RETENTION_DAYS


def resolve_level() -> int:
    """Resolve the log level from the environment per the logging standard.

    Precedence:
        1. ``MULTIPLAI_DEBUG`` truthy (1/true/yes/on) → DEBUG
        2. ``MULTIPLAI_LOG_LEVEL`` (DEBUG|INFO|WARNING|ERROR)
        3. INFO (default)
    """
    if os.environ.get("MULTIPLAI_DEBUG", "").strip().lower() in ("1", "true", "yes", "on"):
        return logging.DEBUG
    name = os.environ.get("MULTIPLAI_LOG_LEVEL", "").strip().upper()
    return _LEVELS.get(name, logging.INFO)


def _rotate_dated(base: Path) -> None:
    """Archive *base* to ``<stem>-<its-day>.<ext>`` if it predates today.

    The day a file's content belongs to is taken from its mtime (UTC).
    A non-existent or empty file, or one already written today, is left
    untouched. If the dated target already exists (e.g. two processes
    crossing midnight), the stale content is appended rather than lost.
    Best-effort: never raises.
    """
    try:
        if not base.exists() or base.stat().st_size == 0:
            return
        file_day = datetime.fromtimestamp(
            base.stat().st_mtime, timezone.utc
        ).strftime("%Y-%m-%d")
        if file_day == _utc_today():
            return
        target = base.with_name(f"{base.stem}-{file_day}{base.suffix}")
        if target.exists():
            with base.open("rb") as src, target.open("ab") as dst:
                dst.write(src.read())
            base.unlink()
        else:
            base.rename(target)
    except OSError:
        pass


def _sweep_logs(logs_dir: Path, days: int) -> None:
    """Normalise and prune the logs directory (best-effort, once/process).

    1. Migrate any legacy ``<name>.log.YYYY-MM-DD`` (the rejected stdlib
       form) to the standard ``<name>-YYYY-MM-DD.log``.
    2. When *days* > 0, delete rotated ``<name>-DATE.log`` /
       ``<name>-DATE.jsonl`` files whose mtime is older than the cutoff.
       *days* == 0 keeps rotated files forever (migration still runs).
    """
    try:
        for f in list(logs_dir.glob("*.log.*")):
            m = _REJECTED_RE.match(f.name)
            if not m:
                continue
            target = f.with_name(f"{m['base']}-{m['date']}.log")
            try:
                if target.exists():
                    with f.open("rb") as src, target.open("ab") as dst:
                        dst.write(src.read())
                    f.unlink()
                else:
                    f.rename(target)
            except OSError:
                pass

        if days <= 0:
            return
        cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
        for f in list(logs_dir.glob("*.log")) + list(logs_dir.glob("*.jsonl")):
            if not _DATED_RE.search(f.name):
                continue
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass
    except OSError:
        pass


class _StandardFormatter(logging.Formatter):
    """Emit ``[ts] [component] [session:xxxxxxxx] LEVEL: message``.

    Timestamp is UTC, ISO-8601, always suffixed ``Z``. Session id is the
    first 8 chars of the Claude Code session id, or ``--------`` if
    unknown.
    """

    def __init__(self, session_id: str | None = None):
        super().__init__()
        self.set_session(session_id)

    def set_session(self, session_id: str | None) -> None:
        sid = (session_id or "")[:8]
        self._sid = sid if sid else "--------"

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        line = (
            f"[{ts}] [{record.name}] [session:{self._sid}] "
            f"{record.levelname}: {record.getMessage()}"
        )
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class _DatedRotatingFileHandler(logging.FileHandler):
    """Write ``<name>.log``; rotate to ``<name>-YYYY-MM-DD.log`` on day change.

    Hooks are short-lived, so the common rotation path is at construction:
    a ``<name>.log`` left over from a previous UTC day is archived before
    the stream is (re)opened. Long-lived processes are also covered — the
    emit path re-checks the UTC day and rotates mid-run.

    Retention is not handled here; :func:`_sweep_logs` prunes uniformly
    across the whole directory per ``MULTIPLAI_LOG_RETENTION_DAYS``.
    """

    def __init__(self, base: Path):
        self._base = Path(base)
        _rotate_dated(self._base)
        super().__init__(self._base, encoding="utf-8")
        self._day = _utc_today()

    def emit(self, record: logging.LogRecord) -> None:
        if _utc_today() != self._day:
            self.acquire()
            try:
                if self.stream:
                    self.stream.flush()
                    self.stream.close()
                _rotate_dated(self._base)
                self.stream = self._open()
                self._day = _utc_today()
            finally:
                self.release()
        super().emit(record)


def setup_logging(
    name: str = "multiplai",
    level: int | None = None,
    session_id: str | None = None,
) -> logging.Logger:
    """Set up logging for a multiplai script.

    Configures (idempotently) a stderr handler, a date-rotated per-component
    file handler (``<name>.log`` current, ``<name>-DATE.log`` rotated), and
    a shared ``hook-errors.log`` handler for ERROR+. When *level* is omitted
    it is resolved from the environment via :func:`resolve_level` so
    ``MULTIPLAI_DEBUG=1`` makes every script verbose without code changes.
    """
    logger = logging.getLogger(name)
    resolved = level if level is not None else resolve_level()
    logger.setLevel(resolved)

    if logger.handlers:
        # Already configured. A long-lived process may call setup_logging again
        # for a new session — refresh the session id on the existing formatters
        # so subsequent lines aren't mislabeled with the first session's id.
        if session_id is not None:
            for handler in logger.handlers:
                formatter = handler.formatter
                if isinstance(formatter, _StandardFormatter):
                    formatter.set_session(session_id)
        return logger

    # Don't also bubble records to the root logger: an embedding app with its
    # own root handler would otherwise print every line twice (our handler +
    # root's). We attach our own handlers below, so propagation is redundant.
    logger.propagate = False

    fmt = _StandardFormatter(session_id)

    # Stderr handler for immediate feedback (visible under `claude --debug`
    # and to anything tailing the hook's stderr).
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(resolved)
    stderr_handler.setFormatter(fmt)
    logger.addHandler(stderr_handler)

    try:
        logs_dir = _get_logs_dir()
        logs_dir.mkdir(parents=True, exist_ok=True)

        file_handler = _DatedRotatingFileHandler(logs_dir / f"{name}.log")
        file_handler.setLevel(resolved)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

        # Shared ERROR+ sink across all components (append-only, undated
        # per the logging standard). Enforce the oversize ceiling before
        # binding — nothing else ever truncates this file.
        _truncate_oversized(logs_dir / "hook-errors.log")
        error_handler = logging.FileHandler(
            logs_dir / "hook-errors.log", encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(fmt)
        logger.addHandler(error_handler)

        global _swept
        if not _swept:
            _swept = True
            _sweep_logs(logs_dir, retention_days())
    except Exception:
        logger.debug("Could not set up file logging", exc_info=True)

    return logger


def log_event(
    component: str,
    event: str,
    message: str,
    *,
    session_id: str | None = None,
    level: str = "INFO",
    **fields: object,
) -> None:
    """Append one curated event to the activity log and its JSONL mirror.

    This is the human-in-the-loop signal — what the plugin actually did,
    in plain language. Written regardless of configured log level and
    never raises (a logging failure must not break a hook).

    Writes to the *current* files ``activity.log`` / ``activity.jsonl``
    (no date suffix); the previous day's stream is rotated to
    ``activity-YYYY-MM-DD.{log,jsonl}`` on the first write of a new UTC
    day, consistent with every other log in the directory.

    Args:
        component: Short subsystem tag (e.g. ``context``, ``nudge``,
            ``diary``, ``learnings``, ``catalog``, ``session``).
        event: Stable machine key for the JSONL mirror (e.g.
            ``inject``, ``dream``, ``write``).
        message: Human-readable sentence describing what happened.
        session_id: Claude Code session id (first 8 chars are recorded).
        level: Severity label for the JSONL record (INFO/WARNING/ERROR).
        **fields: Structured key/values appended to the JSONL mirror.
    """
    try:
        logs_dir = _get_logs_dir()
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_path = logs_dir / "activity.log"
        jsonl_path = logs_dir / "activity.jsonl"
        _rotate_dated(log_path)
        _rotate_dated(jsonl_path)

        now = datetime.now(timezone.utc)
        sid = (session_id or "")[:8] or "--------"

        # The human line is the message, verbatim — a clean sentence the
        # call site is responsible for making self-contained. Structured
        # fields enrich the JSONL mirror only (no noisy key=value tail).
        # Time carries a ``Z`` (UTC, unambiguous across timezones) and
        # the 8-char session id is inline so a line is self-traceable
        # (grep one id to replay a whole session) without the JSONL.
        # Non-INFO severities are tagged inline so a WARNING/ERROR is
        # visible in the human log, not only in the JSONL mirror.
        sev = "" if level.upper() == "INFO" else f" [{level.upper()}]"
        human = (
            f"{now.strftime('%H:%M:%S')}Z [{sid}] [{component}]{sev} {message}"
        )
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(human + "\n")

        record: dict[str, object] = {
            "ts": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "component": component,
            "event": event,
            "level": level,
            "session": sid,
            "msg": message,
        }
        record.update(fields)
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

        global _swept
        if not _swept:
            _swept = True
            _sweep_logs(logs_dir, retention_days())
    except Exception:
        # Observability must never break the thing it observes.
        pass
