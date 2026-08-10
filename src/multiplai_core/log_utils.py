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
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# Fallback reference for "how long has this process been alive", bound at import
# of *this* module. It is only a fallback: anything the process imported before
# log_utils is excluded from it, and so is interpreter start itself. The real
# baseline is the kernel's process start time, read once by
# :func:`_startup_ms`. Monotonic: immune to clock steps mid-run.
_PROCESS_T0 = time.monotonic()

# Startup cost is a property of the *process*, not of each hook_run in it, so it
# is measured once and cached. Without this, a second hook_run 300ms later
# reports 300ms more "startup" than the first, and the number silently becomes
# uptime. None until the first hook_run.
_STARTUP_MS: float | None = None

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
    *,
    propagate_loggers: tuple[str, ...] = (),
) -> logging.Logger:
    """Set up logging for a multiplai script.

    Configures (idempotently) a stderr handler, a date-rotated per-component
    file handler (``<name>.log`` current, ``<name>-DATE.log`` rotated), and
    a shared ``hook-errors.log`` handler for ERROR+. When *level* is omitted
    it is resolved from the environment via :func:`resolve_level` so
    ``MULTIPLAI_DEBUG=1`` makes every script verbose without code changes.

    ``propagate_loggers`` names extra loggers (typically packages such as
    ``"multiplai_core"``) whose records should also land in this component's
    ``<name>.log`` and the shared ``hook-errors.log``. Those libraries log to
    their own dotted loggers (e.g. ``multiplai_core.agent_runner``) which
    normally propagate only to the root logger — which has no file handler in a
    hook process, so their failure detail reaches stderr only. Attaching the
    same file + error handlers to the named package loggers captures that
    detail on disk without adding a second stderr stream (interactive stderr is
    left to this component's own logger). Default ``()`` leaves behavior
    unchanged. The attachment is idempotent — a handler already present on a
    package logger is not attached twice.

    Two caveats. Once a package logger has handlers, ``logging.lastResort``
    stops firing for it, so its WARNING+ records no longer reach stderr at all
    in the capturing process — they land only in the files. And because a
    logger that already has handlers makes ``setup_logging`` return early,
    ``propagate_loggers`` takes effect only on the call that first configures
    *name*; passing it for an already-configured *name* is silently a no-op.
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

        # Route named package loggers' records into this component's file and
        # the shared error sink. Same handler objects, no stderr handler — so
        # library failure detail lands on disk without doubling interactive
        # output. Root has no handlers, so no duplication; we leave each
        # package logger's ``propagate`` untouched.
        for pkg_name in propagate_loggers:
            pkg_logger = logging.getLogger(pkg_name)
            pkg_logger.setLevel(resolved)
            for handler in (file_handler, error_handler):
                if handler not in pkg_logger.handlers:
                    pkg_logger.addHandler(handler)

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


# ---------------------------------------------------------------------------
# Hook timing
# ---------------------------------------------------------------------------
# Why this exists, concretely (2026-08-10): a UserPromptSubmit hook was killed
# at its 30s ceiling and the prompt lost its injected context. The logs said
# nothing at all — not "slow", not "failed", *nothing* — because every hook's
# first log line came after the work it was doing. A killed process cannot
# report its own death, so the only way to learn where the budget went is to
# have written a line down *before* spending it.
#
# So the contract is two lines per run, and the pair is the diagnostic:
#
#   HOOK_ENTRY hook=<name> startup_ms=<n>
#   HOOK_EXIT  hook=<name> status=ok ms=<n> startup_ms=<n> stages=a:12,b:4400
#
# An ENTRY with no matching EXIT is a hook that died mid-run — killed by the
# harness timeout, OOM, or a hard crash. That orphan is the signal; without the
# ENTRY line there is nothing to notice. The intended consumer is
# ``log_doctor --hooks`` in multiplai-cc-mktplace, which pairs the two lines on
# (hook, session, pid).
#
# What this cannot see, by construction: a run killed *before* Python reaches
# ``hook_run()`` — during interpreter start, ``uv`` dependency resolution, or
# imports. Such a run writes no line at all, not even the ENTRY, and looks
# exactly like a hook that never fired. ``startup_ms`` bounds that window from
# below (it is the part of it a surviving run can measure) but cannot report on
# a run that never got that far. Closing the gap needs a marker written outside
# this process — the launcher stamping a line before exec — which is where it
# belongs, not here.
#
# The pair is written in the component log's text format rather than the JSONL
# activity stream: an orphan ENTRY is diagnosed by reading what the hook logged
# around it, and activity.log is a curated user-facing narrative that two
# machine lines per run would bury. The cost is that field values are strings
# (``injected=3`` parses back as ``"3"``) and that keys must be sanitized by
# hand — see ``_STAGE_NAME_SANITIZE`` and ``_RESERVED_FIELDS`` below. Both write
# paths share :func:`_emit_hook_line`, so level-independence is implemented
# once, on the same principle as :func:`log_event`.

# Stage names are embedded in a comma/colon-delimited field, so a name carrying
# either delimiter would corrupt the record for every downstream parser.
_STAGE_NAME_SANITIZE = re.compile(r"[,:=\s]+")

# Keys the record format owns. A note() using one of these would emit a second
# `status=` / `ms=` token on the same line, and two parsers would disagree about
# which one is the hook's real status — so they are prefixed, never overwritten.
_RESERVED_FIELDS = frozenset(
    {"hook", "status", "ms", "startup_ms", "session", "stages", "pid", "err"}
)


def _startup_ms() -> float:
    """Milliseconds from process start to the first :func:`hook_run` in it.

    Computed once per process and cached — see ``_STARTUP_MS``. The baseline is
    the kernel's process start time where the platform exposes it
    (``/proc/self/stat``, which covers every environment hooks actually run in);
    elsewhere it degrades to this module's import time, which undercounts by
    interpreter start plus whatever was imported before log_utils.
    """
    global _STARTUP_MS
    if _STARTUP_MS is not None:
        return _STARTUP_MS
    _STARTUP_MS = (time.monotonic() - _PROCESS_T0) * 1000.0
    try:
        stat = Path("/proc/self/stat").read_text(encoding="ascii", errors="replace")
        # Field 22 (1-indexed) is starttime, in clock ticks since boot. Field 2
        # (comm) can contain spaces and parens, so index from the last ')' —
        # everything after it starts at field 3, making starttime index 19.
        start_ticks = float(stat[stat.rindex(")") + 2 :].split()[19])
        uptime = float(
            Path("/proc/uptime").read_text(encoding="ascii").split()[0]
        )
        elapsed = (uptime - start_ticks / os.sysconf("SC_CLK_TCK")) * 1000.0
        if elapsed >= 0:
            _STARTUP_MS = elapsed
    except Exception:
        pass
    return _STARTUP_MS


def _emit_hook_line(logger: logging.Logger, level: int, message: str) -> None:
    """Write one hook-timing line to the component log, whatever the level.

    ENTRY and EXIT are the signal, not debug noise — the same call
    :func:`log_event` makes, for the same reason. Emitting them with
    ``logger.info`` meant ``MULTIPLAI_LOG_LEVEL=WARNING`` erased both, which is
    indistinguishable from the killed-hook symptom the pair exists to diagnose.

    So the component file handler is fed directly (``Handler.handle`` applies
    filters but not the handler's level), while every other sink — stderr, the
    shared ``hook-errors.log`` — is still gated on *level*. An EXIT logged at
    ERROR therefore reaches the shared error sink; a routine one does not.
    """
    try:
        record = logger.makeRecord(
            logger.name, level, "(hook_run)", 0, message, (), None
        )
    except Exception:
        return
    delivered = False
    try:
        handlers = list(logger.handlers)
    except Exception:
        handlers = []
    for handler in handlers:
        try:
            if isinstance(handler, _DatedRotatingFileHandler):
                handler.handle(record)
                delivered = True
            elif record.levelno >= handler.level:
                handler.handle(record)
        except Exception:
            continue
    if not delivered:
        # No component file handler — logs dir unavailable, or a caller-supplied
        # logger. Nothing to bypass the level for; take the ordinary path.
        try:
            logger.log(level, message)
        except Exception:
            pass


class HookRun:
    """Timing recorder for one hook invocation. Created by :func:`hook_run`.

    Stage timings accumulate *by name*: calling ``stage("router")`` twice adds
    both elapsed times together and counts one stage, which is what you want
    for a loop and is never wrong for a single pass.
    """

    def __init__(self) -> None:
        self._t0 = time.monotonic()
        self._stages: dict[str, float] = {}
        self._fields: dict[str, object] = {}

    @contextmanager
    def stage(self, name: str):
        """Time a named phase of the hook.

        Never suppresses an exception: a stage that raises still records its
        elapsed time, so the EXIT line shows how far the run got before it
        failed.
        """
        key = _STAGE_NAME_SANITIZE.sub("_", name.strip()) or "unnamed"
        started = time.monotonic()
        try:
            yield
        finally:
            elapsed = (time.monotonic() - started) * 1000.0
            self._stages[key] = self._stages.get(key, 0.0) + elapsed

    def note(self, **fields: object) -> None:
        """Attach key=value facts to the EXIT line (e.g. ``injected=3``).

        Values are rendered with ``str()`` and stripped of the field
        delimiters, so anything is safe to pass. A key the record format
        already owns (``status``, ``ms``, …) is prefixed ``note_`` rather than
        emitted twice — a line carrying two ``status=`` tokens parses as ``ok``
        or as the note depending on the reader, and both readers are ours.
        """
        for key, value in fields.items():
            safe_key = _STAGE_NAME_SANITIZE.sub("_", str(key))
            if safe_key in _RESERVED_FIELDS:
                safe_key = f"note_{safe_key}"
            try:
                rendered = str(value)
            except Exception:
                rendered = "<unprintable>"
            self._fields[safe_key] = _STAGE_NAME_SANITIZE.sub("_", rendered)

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds since this run started (excludes process startup)."""
        return (time.monotonic() - self._t0) * 1000.0

    def _stages_field(self) -> str:
        return ",".join(
            f"{key}:{value:.0f}" for key, value in self._stages.items()
        )


@contextmanager
def hook_run(
    name: str,
    logger: logging.Logger,
    *,
    session_id: str | None = None,
):
    """Bracket a hook's work with an ENTRY line and an EXIT line.

    Wrap the *whole* body of a hook's ``main()``, as early as the session id is
    known — everything before the ENTRY line is invisible to this and shows up
    only in ``startup_ms``.

    Args:
        name: The hook's component name, matching its log file
            (``context_manager``, ``session_start``, …). Sanitized like a stage
            name — it is the field every downstream group-by keys on.
        logger: The component logger from :func:`setup_logging`.
        session_id: Recorded on *both* lines so a slow — or killed — run is
            traceable to the session that paid for it. The line prefix already
            carries it when ``setup_logging`` was given one; this is for the
            case where it was not, and the ENTRY line is exactly where it
            matters, because that is the line a killed run leaves behind.

    Yields:
        A :class:`HookRun` for per-stage timing (``run.stage("router")``) and
        extra facts (``run.note(injected=3)``).

    Both lines carry ``pid=``: several hooks of the same name can run
    concurrently and append to one log, and without it a reader can only tell
    that some ENTRY went unmatched, not which one.

    The instrumentation never raises and never swallows: an exception in the
    body is logged as ``status=error err=<type>`` at ERROR (so it reaches the
    shared ``hook-errors.log``) and re-raised unchanged.
    """
    safe_name = _STAGE_NAME_SANITIZE.sub("_", str(name).strip()) or "unnamed"
    sid = str(session_id)[:8] if session_id else ""
    pid = os.getpid()

    run = HookRun()
    startup_ms = _startup_ms()

    entry = [f"HOOK_ENTRY hook={safe_name}", f"pid={pid}", f"startup_ms={startup_ms:.0f}"]
    if sid:
        entry.append(f"session={sid}")
    _emit_hook_line(logger, logging.INFO, " ".join(entry))

    status = "ok"
    err = ""
    try:
        yield run
    except BaseException as exc:
        # BaseException, not Exception: SystemExit is how a hook normally ends
        # (sys.exit(0) after emitting its payload), and a KeyboardInterrupt or
        # a cancelled run is exactly the case where the EXIT line matters most.
        #
        # `code in (None, 0)`, not `not code`: SystemExit("") is falsy but
        # exits the process with status 1 (CPython prints the message, empty or
        # not, and exits 1 for any non-int payload), so a truthiness test calls
        # a failed hook healthy. `in (None, 0)` also keeps SystemExit(False)
        # ok, which is right — bool is an int subclass and it exits 0.
        clean_exit = isinstance(exc, SystemExit) and exc.code in (None, 0)
        if not clean_exit:
            status = "error"
            err = type(exc).__name__
        raise
    finally:
        # Build defensively: an exception escaping here would leave an ENTRY
        # with no EXIT, i.e. this instrumentation forging the exact tombstone it
        # exists to make trustworthy. Whatever fails, a line still goes out.
        try:
            parts = [
                f"HOOK_EXIT hook={safe_name}",
                f"status={status}",
                f"ms={run.elapsed_ms:.0f}",
                f"startup_ms={startup_ms:.0f}",
                f"pid={pid}",
            ]
            if sid:
                parts.append(f"session={sid}")
            if err:
                parts.append(f"err={_STAGE_NAME_SANITIZE.sub('_', err)}")
            stages = run._stages_field()
            if stages:
                parts.append(f"stages={stages}")
            parts.extend(f"{k}={v}" for k, v in run._fields.items())
            line = " ".join(parts)
        except Exception:
            line = (
                f"HOOK_EXIT hook={safe_name} status=error ms=0 startup_ms=0 "
                f"pid={pid} err=instrumentation_failure"
            )
            status = "error"
        _emit_hook_line(
            logger, logging.ERROR if status != "ok" else logging.INFO, line
        )
