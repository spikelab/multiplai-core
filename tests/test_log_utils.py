"""Tests for multiplai_core.log_utils — standard logging + activity stream.

Covers env-driven level resolution, the standard line format, session-id
stamping, the shared hook-errors sink, and the log_event() dual-sink
activity stream (human + JSONL, failure-safe, retention).
"""

import json
import logging
import os
import re
import time
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from multiplai_core import log_utils


@pytest.fixture
def logs_dir(tmp_path, monkeypatch, reset_paths_cache):
    """Point the path resolver at a tmp data dir; yield its logs/ dir."""
    monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path / "plugin"))
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path / "data"))
    log_utils._swept = False
    log_utils._STARTUP_MS = None
    monkeypatch.delenv("MULTIPLAI_LOG_RETENTION_DAYS", raising=False)
    yield tmp_path / "data" / "logs"


def _unique(name: str) -> str:
    return f"{name}-{uuid.uuid4().hex[:8]}"


def _today(logs_dir, name: str, ext: str = "log"):
    """The file a component writes today: ``<name>-YYYY-MM-DD.<ext>``."""
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    return logs_dir / f"{name}-{day}.{ext}"


# setup_logging attaches handlers to package loggers that outlive the test:
# multiplai_core always, and the caller's package (this module's).
_PKG_LOGGERS = ("multiplai_core", __name__.split(".")[0], "fakepkg")


@pytest.fixture(autouse=True)
def _restore_package_loggers():
    saved = {n: (list(logging.getLogger(n).handlers), logging.getLogger(n).level)
             for n in _PKG_LOGGERS}
    yield
    for n, (handlers, level) in saved.items():
        lg = logging.getLogger(n)
        for h in list(lg.handlers):
            if h not in handlers:
                lg.removeHandler(h)
        lg.setLevel(level)


# --------------------------------------------------------------------------
# resolve_level
# --------------------------------------------------------------------------

def test_resolve_level_defaults_to_info(monkeypatch):
    monkeypatch.delenv("MULTIPLAI_DEBUG", raising=False)
    monkeypatch.delenv("MULTIPLAI_LOG_LEVEL", raising=False)
    assert log_utils.resolve_level() == logging.INFO


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
def test_resolve_level_debug_env_wins(monkeypatch, val):
    monkeypatch.setenv("MULTIPLAI_DEBUG", val)
    monkeypatch.setenv("MULTIPLAI_LOG_LEVEL", "ERROR")
    assert log_utils.resolve_level() == logging.DEBUG


@pytest.mark.parametrize(
    "name,expected",
    [("DEBUG", logging.DEBUG), ("warning", logging.WARNING), ("ERROR", logging.ERROR)],
)
def test_resolve_level_from_log_level_env(monkeypatch, name, expected):
    monkeypatch.delenv("MULTIPLAI_DEBUG", raising=False)
    monkeypatch.setenv("MULTIPLAI_LOG_LEVEL", name)
    assert log_utils.resolve_level() == expected


def test_resolve_level_invalid_falls_back_to_info(monkeypatch):
    monkeypatch.delenv("MULTIPLAI_DEBUG", raising=False)
    monkeypatch.setenv("MULTIPLAI_LOG_LEVEL", "LOUD")
    assert log_utils.resolve_level() == logging.INFO


# --------------------------------------------------------------------------
# setup_logging
# --------------------------------------------------------------------------

def test_setup_logging_level_from_env(monkeypatch, logs_dir):
    monkeypatch.setenv("MULTIPLAI_DEBUG", "1")
    logger = log_utils.setup_logging(_unique("lvl"))
    assert logger.level == logging.DEBUG
    assert all(h.level == logging.DEBUG for h in logger.handlers
               if not isinstance(h, logging.FileHandler) or "hook-errors" not in str(getattr(h, "baseFilename", "")))


def test_setup_logging_writes_standard_format(logs_dir):
    name = _unique("fmt")
    logger = log_utils.setup_logging(name)
    logger.info("hello world")
    for h in logger.handlers:
        h.flush()

    line = _today(logs_dir, name).read_text().strip().splitlines()[-1]
    assert re.match(
        rf"^\[\d{{4}}-\d{{2}}-\d{{2}}T\d{{2}}:\d{{2}}:\d{{2}}Z\] "
        rf"\[{re.escape(name)}\] \[session:--------\] INFO: hello world$",
        line,
    ), line


def test_setup_logging_stamps_session_id(logs_dir):
    name = _unique("sid")
    logger = log_utils.setup_logging(name, session_id="abcdefdeadbeef")
    logger.info("x")
    for h in logger.handlers:
        h.flush()
    line = _today(logs_dir, name).read_text().strip().splitlines()[-1]
    assert "[session:abcdefde]" in line


def test_errors_go_to_shared_hook_errors_log(logs_dir):
    name = _unique("err")
    logger = log_utils.setup_logging(name)
    logger.error("boom")
    for h in logger.handlers:
        h.flush()
    shared = (logs_dir / "hook-errors.log").read_text()
    assert "ERROR: boom" in shared
    assert f"[{name}]" in shared


@pytest.fixture
def clean_pkg_logger():
    """Package loggers have fixed names, so their handlers and level leak
    across tests. Detach anything a test attaches to ``multiplai_core`` and
    restore its level."""
    pkg = logging.getLogger("multiplai_core")
    saved_level = pkg.level
    yield
    for h in list(pkg.handlers):
        pkg.removeHandler(h)
    pkg.setLevel(saved_level)


def test_setup_logging_captures_propagate_loggers(logs_dir, clean_pkg_logger):
    """propagate_loggers routes a package logger's records into the
    component file AND the shared hook-errors sink."""
    name = _unique("cap")
    log_utils.setup_logging(name, propagate_loggers=("multiplai_core",))
    logging.getLogger("multiplai_core.agent_runner").error("boom")
    for h in logging.getLogger("multiplai_core").handlers:
        h.flush()

    component = _today(logs_dir, name).read_text()
    shared = (logs_dir / "hook-errors.log").read_text()
    assert "ERROR: boom" in component
    assert "ERROR: boom" in shared


def test_setup_logging_captures_multiplai_core_by_default(logs_dir, clean_pkg_logger):
    """A bare setup_logging(name) routes multiplai_core.* INFO lines into the
    component file: core's lines were lost when a caller forgot to list it."""
    name = _unique("defcap")
    log_utils.setup_logging(name)
    logging.getLogger("multiplai_core.agent_runner").info("core line")
    for h in logging.getLogger("multiplai_core").handlers:
        h.flush()

    assert "[multiplai_core.agent_runner]" in _today(logs_dir, name).read_text()
    assert "INFO: core line" in _today(logs_dir, name).read_text()


def _fake_package(tmp_path, monkeypatch):
    """A package ``fakepkg`` whose __init__ sets up logging and whose
    ``fakepkg.mod`` logs to ``getLogger(__name__)``, like a pipeline."""
    pkg = tmp_path / "fakepkg_src" / "fakepkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        "from multiplai_core.log_utils import setup_logging\n"
        "def start(name):\n"
        "    return setup_logging(name)\n"
    )
    (pkg / "mod.py").write_text(
        "import logging\n"
        "log = logging.getLogger(__name__)\n"
    )
    monkeypatch.syspath_prepend(str(pkg.parent))
    import importlib
    import sys
    for mod in ("fakepkg", "fakepkg.mod"):
        sys.modules.pop(mod, None)
    return importlib.import_module("fakepkg"), importlib.import_module("fakepkg.mod")


def test_module_logger_in_the_callers_package_reaches_the_file(
    logs_dir, tmp_path, monkeypatch
):
    fakepkg, mod = _fake_package(tmp_path, monkeypatch)
    name = _unique("pkg")
    fakepkg.start(name)
    mod.log.info("module line")
    for h in logging.getLogger("fakepkg").handlers:
        h.flush()

    text = _today(logs_dir, name).read_text()
    assert "[fakepkg.mod]" in text
    assert "INFO: module line" in text


def test_package_warning_reaches_stderr_formatted(logs_dir, tmp_path, monkeypatch, capsys):
    fakepkg, mod = _fake_package(tmp_path, monkeypatch)
    fakepkg.start(_unique("warn"))
    capsys.readouterr()
    mod.log.info("quiet")
    mod.log.warning("loud")

    err = capsys.readouterr().err
    assert "quiet" not in err
    assert re.search(
        r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\] \[fakepkg\.mod\] "
        r"\[session:[^\]]+\] WARNING: loud$",
        err, re.MULTILINE,
    ), err


def test_package_warning_is_printed_once_with_two_components(
    logs_dir, tmp_path, monkeypatch, capsys
):
    fakepkg, mod = _fake_package(tmp_path, monkeypatch)
    fakepkg.start(_unique("one"))
    fakepkg.start(_unique("two"))
    capsys.readouterr()
    mod.log.warning("once")
    assert capsys.readouterr().err.count("WARNING: once") == 1


def test_packages_argument_names_script_packages(logs_dir, clean_pkg_logger):
    """A script run by path has no package; it names its own."""
    name = _unique("explicit")
    lg = logging.getLogger("fakepkg")
    log_utils.setup_logging(name, packages=("fakepkg",))
    logging.getLogger("fakepkg.lib").info("script lib line")
    for h in lg.handlers:
        h.flush()
    assert "script lib line" in _today(logs_dir, name).read_text()


def test_setup_logging_capture_no_double_write(logs_dir, clean_pkg_logger):
    """Calling setup_logging twice must not attach the handlers twice —
    each package record is written exactly once."""
    name = _unique("dbl")
    log_utils.setup_logging(name, propagate_loggers=("multiplai_core",))
    log_utils.setup_logging(name, propagate_loggers=("multiplai_core",))
    logging.getLogger("multiplai_core.agent_runner").error("solo")
    for h in logging.getLogger("multiplai_core").handlers:
        h.flush()

    component = _today(logs_dir, name).read_text()
    assert component.count("ERROR: solo") == 1


# --------------------------------------------------------------------------
# log_event
# --------------------------------------------------------------------------

def test_log_event_writes_human_and_jsonl(logs_dir):
    log_utils.log_event(
        "context", "inject", "injected 2 memory · 0 skills",
        session_id="sess1234xx", memory=2, files=["a.md", "b.md"],
    )

    # Current file is undated — no date in the filename.
    human = _today(logs_dir, "activity").read_text().strip()
    assert re.match(
        r"^\d{2}:\d{2}:\d{2}Z \[sess1234\] \[context\] "
        r"injected 2 memory · 0 skills$",
        human,
    ), human

    # Structured fields live only in the JSONL mirror.
    assert "memory=2" not in human

    rec = json.loads(_today(logs_dir, "activity", "jsonl").read_text().strip())
    assert rec["component"] == "context"
    assert rec["event"] == "inject"
    assert rec["session"] == "sess1234"
    assert rec["memory"] == 2
    assert rec["files"] == ["a.md", "b.md"]
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", rec["ts"])


def test_log_event_tags_non_info_severity_in_human_line(logs_dir):
    """A WARNING/ERROR event is visible in the human log, not just the JSONL."""
    log_utils.log_event("dream", "apply", "memory write failed",
                         session_id="sess1234xx", level="ERROR")
    human = _today(logs_dir, "activity").read_text().strip()
    assert "[ERROR]" in human
    assert "[dream] [ERROR] memory write failed" in human


def test_log_event_info_has_no_severity_tag(logs_dir):
    """INFO events stay clean — no severity tag."""
    log_utils.log_event("context", "inject", "did a thing", session_id="s")
    human = _today(logs_dir, "activity").read_text().strip()
    assert "[INFO]" not in human


def test_log_event_never_raises(monkeypatch):
    def boom():
        raise RuntimeError("no logs dir")

    monkeypatch.setattr(log_utils, "_get_logs_dir", boom)
    # Must not propagate — observability cannot break the hook.
    log_utils.log_event("x", "y", "z")


def test_log_event_prunes_old_activity_only_when_retention_is_set(logs_dir, monkeypatch):
    monkeypatch.setenv("MULTIPLAI_LOG_RETENTION_DAYS", "7")
    logs_dir.mkdir(parents=True, exist_ok=True)
    old = logs_dir / "activity-2000-01-01.log"
    old.write_text("ancient\n")
    ancient = datetime(2000, 1, 1, tzinfo=UTC).timestamp()
    os.utime(old, (ancient, ancient))

    log_utils._swept = False
    log_utils.log_event("session", "start", "fresh")

    assert not old.exists()
    assert _today(logs_dir, "activity").exists()


def test_default_sweep_keeps_a_year_of_logs(logs_dir):
    logs_dir.mkdir(parents=True, exist_ok=True)
    old = logs_dir / "activity-2000-01-01.log"
    recent = logs_dir / "build-pipeline-2026-01-01.log"
    for p in (old, recent):
        p.write_text("x\n")
    ancient = datetime(2000, 1, 1, tzinfo=UTC).timestamp()
    os.utime(old, (ancient, ancient))
    months = (datetime.now(UTC) - timedelta(days=300)).timestamp()
    os.utime(recent, (months, months))

    log_utils._swept = False
    log_utils.log_event("session", "start", "fresh")
    log_utils.setup_logging(_unique("sweep"))

    assert not old.exists()
    assert recent.read_text() == "x\n"


def test_log_event_appends_multiple_records(logs_dir):
    for i in range(3):
        log_utils.log_event("diary", "write", f"entry {i}")
    lines = _today(logs_dir, "activity", "jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    assert [json.loads(l)["msg"] for l in lines] == ["entry 0", "entry 1", "entry 2"]


# --------------------------------------------------------------------------
# date-rotation + naming convention
# --------------------------------------------------------------------------

def test_log_event_writes_each_day_to_its_own_file(logs_dir, monkeypatch):
    """Each UTC day's events go to activity-<day>.{log,jsonl}; a legacy
    undated activity.log is left exactly as it was."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    legacy = logs_dir / "activity.log"
    legacy.write_text("12:00:00 [old] legacy line\n")

    monkeypatch.setattr(log_utils, "_utc_today", lambda: "2031-01-01")
    log_utils.log_event("session", "start", "day one")
    monkeypatch.setattr(log_utils, "_utc_today", lambda: "2031-01-02")
    log_utils.log_event("session", "start", "day two")

    assert "day one" in (logs_dir / "activity-2031-01-01.log").read_text()
    assert "day two" in (logs_dir / "activity-2031-01-02.log").read_text()
    assert "day two" not in (logs_dir / "activity-2031-01-01.log").read_text()
    assert json.loads((logs_dir / "activity-2031-01-02.jsonl").read_text())["msg"] == "day two"
    assert legacy.read_text() == "12:00:00 [old] legacy line\n"
    assert not (logs_dir / "activity.jsonl").exists()


def test_setup_logging_writes_the_dated_file_and_leaves_a_legacy_one(logs_dir):
    """The handler writes <name>-<today>.log; a <name>.log from an earlier
    release is neither renamed nor appended to."""
    name = _unique("rot")
    logs_dir.mkdir(parents=True, exist_ok=True)
    legacy = logs_dir / f"{name}.log"
    legacy.write_text("[old] line\n")

    logger = log_utils.setup_logging(name)
    logger.info("new line")
    for h in logger.handlers:
        h.flush()

    assert legacy.read_text() == "[old] line\n"
    assert "new line" in _today(logs_dir, name).read_text()


def test_handler_switches_file_when_the_day_changes(logs_dir, monkeypatch):
    name = _unique("daychange")
    monkeypatch.setattr(log_utils, "_utc_today", lambda: "2031-03-01")
    logger = log_utils.setup_logging(name)
    logger.info("before midnight")
    monkeypatch.setattr(log_utils, "_utc_today", lambda: "2031-03-02")
    logger.info("after midnight")
    for h in logger.handlers:
        h.flush()

    day1 = (logs_dir / f"{name}-2031-03-01.log").read_text()
    day2 = (logs_dir / f"{name}-2031-03-02.log").read_text()
    assert "before midnight" in day1 and "after midnight" not in day1
    assert "after midnight" in day2 and "before midnight" not in day2


_WRITER = """
import logging, sys
import multiplai_core.log_utils as lu

name, tag, total, switch_at = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
state = {"i": 0}
lu._utc_today = lambda: "2031-05-01" if state["i"] < switch_at else "2031-05-02"
logger = lu.setup_logging(name, level=logging.INFO)
for i in range(total):
    state["i"] = i
    logger.info("%s line %d", tag, i)
"""


def test_two_processes_across_midnight_lose_no_line(logs_dir, tmp_path):
    """Two writers of one component cross a (patched) UTC midnight at
    different moments. Both days' files together hold every line."""
    import subprocess
    import sys

    script = tmp_path / "writer.py"
    script.write_text(_WRITER)
    name = _unique("race")
    env = {**os.environ}
    total = 1500
    procs = [
        subprocess.Popen([sys.executable, str(script), name, tag, str(total), str(switch)],
                         env=env, stderr=subprocess.DEVNULL)
        for tag, switch in (("A", 500), ("B", 1000))
    ]
    for p in procs:
        assert p.wait(timeout=60) == 0

    lines = []
    for day in ("2031-05-01", "2031-05-02"):
        lines += (logs_dir / f"{name}-{day}.log").read_text().splitlines()
    for tag in ("A", "B"):
        got = {int(m.group(1)) for line in lines
               if (m := re.search(rf"INFO: {tag} line (\d+)$", line))}
        assert got == set(range(total)), f"{tag} lost {total - len(got)} lines"
    assert len(lines) == 2 * total
    assert not (logs_dir / f"{name}.log").exists()


def test_sweep_migrates_rejected_stdlib_form(logs_dir):
    """Legacy <name>.log.YYYY-MM-DD is migrated to <name>-DATE.log."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    bad = logs_dir / "context_manager.log.2026-05-16"
    bad.write_text("legacy rotated content\n")

    log_utils._sweep_logs(logs_dir, log_utils.retention_days())

    assert not bad.exists()
    good = logs_dir / "context_manager-2026-05-16.log"
    assert good.read_text() == "legacy rotated content\n"


# --------------------------------------------------------------------------
# retention
# --------------------------------------------------------------------------

def test_retention_days_default_and_parsing(monkeypatch):
    # Unset, unparseable or negative all mean the default, 365 days.
    monkeypatch.delenv("MULTIPLAI_LOG_RETENTION_DAYS", raising=False)
    assert log_utils.retention_days() == 365
    monkeypatch.setenv("MULTIPLAI_LOG_RETENTION_DAYS", "30")
    assert log_utils.retention_days() == 30
    monkeypatch.setenv("MULTIPLAI_LOG_RETENTION_DAYS", "0")
    assert log_utils.retention_days() == 0
    monkeypatch.setenv("MULTIPLAI_LOG_RETENTION_DAYS", "garbage")
    assert log_utils.retention_days() == 365
    monkeypatch.setenv("MULTIPLAI_LOG_RETENTION_DAYS", "-5")
    assert log_utils.retention_days() == 365


def test_retention_days_reads_multiplai_conf(tmp_path, monkeypatch):
    """A pipeline started with `uv run` has no exported setting; the conf is
    how it keeps logs as long as a hook does."""
    monkeypatch.delenv("MULTIPLAI_LOG_RETENTION_DAYS", raising=False)
    monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
    (tmp_path / "multiplai.conf").write_text('MULTIPLAI_LOG_RETENTION_DAYS="90"\n')
    assert log_utils.retention_days() == 90
    monkeypatch.setenv("MULTIPLAI_LOG_RETENTION_DAYS", "12")
    assert log_utils.retention_days() == 12  # the environment wins


def test_retention_zero_keeps_dated_files_forever(logs_dir, monkeypatch):
    logs_dir.mkdir(parents=True, exist_ok=True)
    old = logs_dir / "activity-2000-01-01.log"
    old.write_text("ancient\n")
    ancient = datetime(2000, 1, 1, tzinfo=UTC).timestamp()
    os.utime(old, (ancient, ancient))
    monkeypatch.setenv("MULTIPLAI_LOG_RETENTION_DAYS", "0")

    log_utils._swept = False
    log_utils.log_event("session", "start", "fresh")

    assert old.exists()  # 0 = keep forever


def test_sweep_prunes_only_dated_files_beyond_cutoff(logs_dir):
    logs_dir.mkdir(parents=True, exist_ok=True)
    keep_current = logs_dir / "context_manager.log"      # undated → never pruned
    keep_recent = logs_dir / "context_manager-2030-01-01.log"
    drop_old = logs_dir / "context_manager-2000-01-01.log"
    for p in (keep_current, keep_recent, drop_old):
        p.write_text("x\n")
    ancient = datetime(2000, 1, 1, tzinfo=UTC).timestamp()
    os.utime(drop_old, (ancient, ancient))

    log_utils._sweep_logs(logs_dir, 7)

    assert keep_current.exists()
    assert keep_recent.exists()
    assert not drop_old.exists()


# --------------------------------------------------------------------------
# the shared error log is append-only
# --------------------------------------------------------------------------

def test_hook_errors_log_is_never_shortened(logs_dir):
    logs_dir.mkdir(parents=True, exist_ok=True)
    err_log = logs_dir / "hook-errors.log"
    big = "".join(f"line-{i:06d}\n" for i in range(30_000))  # ~390KB
    err_log.write_text(big)

    logger = log_utils.setup_logging(_unique("big"))
    logger.error("appended")
    for h in logger.handlers:
        h.flush()

    text = err_log.read_text()
    assert text.startswith(big)
    assert "ERROR: appended" in text[len(big):]


def _stamp(path, when: datetime) -> None:
    ts = when.timestamp()
    os.utime(path, (ts, ts))


def test_hook_errors_log_moves_to_its_dated_name_on_a_new_day(logs_dir):
    logs_dir.mkdir(parents=True, exist_ok=True)
    err_log = logs_dir / "hook-errors.log"
    err_log.write_text("yesterday's error\n")
    _stamp(err_log, datetime(2026, 3, 4, 12, tzinfo=UTC))

    logger = log_utils.setup_logging(_unique("rot"))
    logger.error("today's error")
    for h in logger.handlers:
        h.flush()

    assert (logs_dir / "hook-errors-2026-03-04.log").read_text() == "yesterday's error\n"
    assert "today's error" in err_log.read_text()
    assert "yesterday" not in err_log.read_text()


def test_hook_errors_log_written_today_stays_put(logs_dir):
    logs_dir.mkdir(parents=True, exist_ok=True)
    err_log = logs_dir / "hook-errors.log"
    err_log.write_text("earlier today\n")

    log_utils._rotate_shared_error_log(logs_dir)

    assert err_log.read_text() == "earlier today\n"
    assert not list(logs_dir.glob("hook-errors-*.log"))


def test_hook_errors_rotation_never_overwrites_a_dated_file(logs_dir):
    """Two processes rotating at once: the second finds the dated name taken
    and leaves both files alone."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    dated = logs_dir / "hook-errors-2026-03-04.log"
    dated.write_text("first rotation\n")
    err_log = logs_dir / "hook-errors.log"
    err_log.write_text("second\n")
    _stamp(err_log, datetime(2026, 3, 4, 12, tzinfo=UTC))

    log_utils._rotate_shared_error_log(logs_dir)

    assert dated.read_text() == "first rotation\n"
    assert err_log.read_text() == "second\n"


def test_rotated_hook_errors_follow_retention(logs_dir):
    logs_dir.mkdir(parents=True, exist_ok=True)
    old = logs_dir / "hook-errors-2000-01-01.log"
    old.write_text("x\n")
    _stamp(old, datetime(2000, 1, 1, tzinfo=UTC))

    log_utils._sweep_logs(logs_dir, 365)

    assert not old.exists()


# --------------------------------------------------------------------------
# hook_run
# --------------------------------------------------------------------------


def _lines(logs_dir, name):
    return _today(logs_dir, name).read_text().splitlines()


def test_hook_run_writes_entry_before_the_body(logs_dir):
    """The ENTRY line must be on disk while the body is still running.

    This is the whole point: a hook killed mid-body leaves the ENTRY behind.
    """
    name = _unique("hookrun")
    logger = log_utils.setup_logging(name)

    with log_utils.hook_run(name, logger):
        during = _lines(logs_dir, name)

    assert any("HOOK_ENTRY" in line for line in during)
    assert not any("HOOK_EXIT" in line for line in during)


def test_hook_run_exit_reports_ok_and_duration(logs_dir):
    name = _unique("hookrun")
    logger = log_utils.setup_logging(name)

    with log_utils.hook_run(name, logger, session_id="abcdef1234567890"):
        pass

    exit_line = [ln for ln in _lines(logs_dir, name) if "HOOK_EXIT" in ln][0]
    assert f"hook={name}" in exit_line
    assert "status=ok" in exit_line
    assert re.search(r"\bms=\d+", exit_line)
    assert re.search(r"\bstartup_ms=\d+", exit_line)
    assert "session=abcdef12" in exit_line
    assert "session=abcdef1234567890" not in exit_line


def test_hook_run_records_stage_timings(logs_dir):
    name = _unique("hookrun")
    logger = log_utils.setup_logging(name)

    with log_utils.hook_run(name, logger) as run:
        with run.stage("router"):
            pass
        with run.stage("assemble"):
            pass

    exit_line = [ln for ln in _lines(logs_dir, name) if "HOOK_EXIT" in ln][0]
    stages = re.search(r"stages=(\S+)", exit_line).group(1)
    assert [s.split(":")[0] for s in stages.split(",")] == ["router", "assemble"]


def test_hook_run_accumulates_a_repeated_stage(logs_dir):
    name = _unique("hookrun")
    logger = log_utils.setup_logging(name)

    with log_utils.hook_run(name, logger) as run:
        for _ in range(3):
            with run.stage("chunk"):
                pass

    exit_line = [ln for ln in _lines(logs_dir, name) if "HOOK_EXIT" in ln][0]
    assert re.search(r"stages=chunk:\d+(\s|$)", exit_line)


def test_hook_run_sanitizes_delimiters_in_names_and_notes(logs_dir):
    name = _unique("hookrun")
    logger = log_utils.setup_logging(name)

    with log_utils.hook_run(name, logger) as run:
        with run.stage("read stdin: phase,one"):
            pass
        run.note(picked="a,b:c")

    exit_line = [ln for ln in _lines(logs_dir, name) if "HOOK_EXIT" in ln][0]
    # Field structure survives: each space-separated token is at most one '='.
    for token in exit_line.split("HOOK_EXIT ", 1)[1].split():
        assert token.count("=") <= 1
    assert "read_stdin_phase_one" in exit_line
    assert "picked=a_b_c" in exit_line


def test_hook_run_marks_errors_and_reraises(logs_dir):
    name = _unique("hookrun")
    logger = log_utils.setup_logging(name)

    with pytest.raises(ValueError), log_utils.hook_run(name, logger) as run:
        with run.stage("router"):
            raise ValueError("boom")

    exit_line = [ln for ln in _lines(logs_dir, name) if "HOOK_EXIT" in ln][0]
    assert "status=error" in exit_line
    # The stage that failed still reports how long it ran before failing.
    assert "stages=router:" in exit_line


@pytest.mark.parametrize("code", [0, None, False])
def test_hook_run_treats_clean_sys_exit_as_ok(logs_dir, code):
    """Hooks end with sys.exit(0) after emitting their payload.

    ``None`` and ``False`` are here because CPython exits 0 for both.
    """
    name = _unique("hookrun")
    logger = log_utils.setup_logging(name)

    with pytest.raises(SystemExit), log_utils.hook_run(name, logger):
        raise SystemExit(code)

    exit_line = [ln for ln in _lines(logs_dir, name) if "HOOK_EXIT" in ln][0]
    assert "status=ok" in exit_line


def test_hook_run_treats_nonzero_sys_exit_as_error(logs_dir):
    name = _unique("hookrun")
    logger = log_utils.setup_logging(name)

    with pytest.raises(SystemExit), log_utils.hook_run(name, logger):
        raise SystemExit(2)

    exit_line = [ln for ln in _lines(logs_dir, name) if "HOOK_EXIT" in ln][0]
    assert "status=error" in exit_line


def test_hook_run_survives_a_broken_logger(logs_dir):
    """Observability must never break the thing it observes."""

    class Exploding(logging.Logger):
        def handle(self, *a, **k):
            raise RuntimeError("logger is down")

        def makeRecord(self, *a, **k):
            raise RuntimeError("logger is down")

        def log(self, *a, **k):
            raise RuntimeError("logger is down")

    logger = Exploding("exploding")
    body_ran = False
    with log_utils.hook_run("whatever", logger) as run:
        with run.stage("s"):
            body_ran = True
        run.note(k=1)

    # The body completed and no logging failure escaped the context manager.
    assert body_ran


def test_hook_run_entry_carries_session_and_pid(logs_dir):
    """The ENTRY line is all a killed run leaves — it must be attributable."""
    name = _unique("hookrun")
    logger = log_utils.setup_logging(name)

    with log_utils.hook_run(name, logger, session_id="abcdef1234567890"):
        pass

    entry = [ln for ln in _lines(logs_dir, name) if "HOOK_ENTRY" in ln][0]
    assert "session=abcdef12" in entry
    assert f"pid={os.getpid()}" in entry


def test_hook_run_writes_both_lines_when_level_suppresses_info(logs_dir, monkeypatch):
    """MULTIPLAI_LOG_LEVEL=WARNING must not erase the pair.

    A level that hides the tombstone produces the exact zero-lines symptom
    hook_run exists to diagnose.
    """
    monkeypatch.setenv("MULTIPLAI_LOG_LEVEL", "WARNING")
    name = _unique("hookrun")
    logger = log_utils.setup_logging(name)

    with log_utils.hook_run(name, logger):
        pass

    lines = _lines(logs_dir, name)
    assert any("HOOK_ENTRY" in ln for ln in lines)
    assert any("HOOK_EXIT" in ln for ln in lines)


def test_hook_run_error_names_the_exception_and_reaches_the_error_sink(logs_dir):
    name = _unique("hookrun")
    logger = log_utils.setup_logging(name)

    with pytest.raises(ValueError), log_utils.hook_run(name, logger):
        raise ValueError("boom")

    exit_line = [ln for ln in _lines(logs_dir, name) if "HOOK_EXIT" in ln][0]
    assert "status=error" in exit_line
    assert "err=ValueError" in exit_line

    # The shared ERROR+ sink is where someone greps for failures.
    shared = (logs_dir / "hook-errors.log").read_text()
    assert "HOOK_EXIT" in shared and "err=ValueError" in shared


def test_hook_run_ok_exit_stays_out_of_the_error_sink(logs_dir):
    name = _unique("hookrun")
    logger = log_utils.setup_logging(name)

    with log_utils.hook_run(name, logger):
        pass

    shared = logs_dir / "hook-errors.log"
    assert not shared.exists() or "HOOK_EXIT" not in shared.read_text()


@pytest.mark.parametrize("code", ["", "boom"])
def test_hook_run_treats_falsy_nonzero_sys_exit_as_error(logs_dir, code):
    """A falsy-but-nonzero payload is a failed hook.

    Verified against CPython: ``sys.exit("")`` exits 1 (the message goes to
    stderr, empty or not), so a truthiness test would call it ok. ``False``
    is not in this list — it exits 0, and ``code in (None, 0)`` matches that.
    """
    name = _unique("hookrun")
    logger = log_utils.setup_logging(name)

    with pytest.raises(SystemExit), log_utils.hook_run(name, logger):
        raise SystemExit(code)

    exit_line = [ln for ln in _lines(logs_dir, name) if "HOOK_EXIT" in ln][0]
    assert "status=error" in exit_line


def test_hook_run_note_cannot_shadow_a_reserved_field(logs_dir):
    name = _unique("hookrun")
    logger = log_utils.setup_logging(name)

    with log_utils.hook_run(name, logger) as run:
        run.note(status="degraded", ms=999, outcome="injected")

    exit_line = [ln for ln in _lines(logs_dir, name) if "HOOK_EXIT" in ln][0]
    tail = exit_line.split("HOOK_EXIT ", 1)[1]
    assert tail.count(" status=") == 1
    assert tail.count(" ms=") == 1
    assert "note_status=degraded" in tail
    assert "note_ms=999" in tail
    assert "outcome=injected" in tail


def test_hook_run_startup_ms_does_not_grow_across_runs(logs_dir):
    """startup_ms describes the process, not how long ago it started."""
    name = _unique("hookrun")
    logger = log_utils.setup_logging(name)

    with log_utils.hook_run(name, logger):
        pass
    time.sleep(0.05)
    with log_utils.hook_run(name, logger):
        pass

    values = [
        float(re.search(r"\bstartup_ms=(\d+)", ln).group(1))
        for ln in _lines(logs_dir, name)
        if "HOOK_ENTRY" in ln
    ]
    assert len(values) == 2
    assert values[0] == values[1]


def test_hook_run_still_writes_an_exit_when_the_line_cannot_be_built(logs_dir):
    """An instrumentation failure must not forge a killed-hook tombstone."""
    name = _unique("hookrun")
    logger = log_utils.setup_logging(name)

    with log_utils.hook_run(name, logger) as run:
        # Corrupt the recorder so building the stages field raises.
        run._stages = {"router": object()}

    lines = _lines(logs_dir, name)
    assert any("HOOK_ENTRY" in ln for ln in lines)
    exit_line = [ln for ln in lines if "HOOK_EXIT" in ln][0]
    assert "err=instrumentation_failure" in exit_line


def test_hook_run_sanitizes_the_hook_name(logs_dir):
    name = _unique("hookrun")
    logger = log_utils.setup_logging(name)

    with log_utils.hook_run("context manager=x", logger):
        pass

    entry = [ln for ln in _lines(logs_dir, name) if "HOOK_ENTRY" in ln][0]
    assert "hook=context_manager_x" in entry
    for token in entry.split("HOOK_ENTRY ", 1)[1].split():
        assert token.count("=") <= 1


# --------------------------------------------------------------------------
# hook_run watchdog
# --------------------------------------------------------------------------


def _wait_for_line(logs_dir, name, needle, timeout=5.0):
    import time as _time

    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        path = _today(logs_dir, name)
        if path.exists():
            hits = [ln for ln in path.read_text().splitlines() if needle in ln]
            if hits:
                return hits
        _time.sleep(0.01)
    return []


def test_watchdog_names_the_running_stage_before_the_body_ends(logs_dir):
    """The line must be on disk while the stage is still running.

    That is the case it exists for: the harness kills the hook mid-stage and
    nothing after HOOK_ENTRY would otherwise be written.
    """
    name = _unique("watchdog")
    logger = log_utils.setup_logging(name)

    with log_utils.hook_run(name, logger, session_id="abcdef123456", watchdog_s=0.05) as run:
        with run.stage("transcript"):
            pass
        with run.stage("router"):
            during = _wait_for_line(logs_dir, name, "HOOK_WATCHDOG")

    assert during, "watchdog line never reached the file"
    line = during[0]
    assert f"hook={name}" in line
    assert "stage=router" in line
    assert re.search(r"stages=transcript:\d+", line)
    assert "session=abcdef12" in line
    assert re.search(r"\bms=\d+", line)
    exit_line = [ln for ln in _lines(logs_dir, name) if "HOOK_EXIT" in ln][0]
    assert "watchdog=fired" in exit_line


def test_watchdog_stays_quiet_when_the_body_is_fast(logs_dir):
    import time as _time

    name = _unique("watchdog")
    logger = log_utils.setup_logging(name)

    with log_utils.hook_run(name, logger, watchdog_s=0.2):
        pass
    _time.sleep(0.3)

    lines = _lines(logs_dir, name)
    assert not any("HOOK_WATCHDOG" in ln for ln in lines)
    assert not any("watchdog=" in ln for ln in lines)


def test_watchdog_reports_no_stage_between_stages(logs_dir):
    name = _unique("watchdog")
    logger = log_utils.setup_logging(name)

    with log_utils.hook_run(name, logger, watchdog_s=0.05):
        hits = _wait_for_line(logs_dir, name, "HOOK_WATCHDOG")

    assert hits and "stage=-" in hits[0]


def test_watchdog_runs_the_callback_and_swallows_its_errors(logs_dir):
    name = _unique("watchdog")
    logger = log_utils.setup_logging(name)
    seen = []

    def _probe(run):
        seen.append(run.current_stage)
        logger.warning("probe says dns_ms=12")
        raise RuntimeError("probe blew up")

    with log_utils.hook_run(name, logger, watchdog_s=0.05, on_watchdog=_probe) as run:
        with run.stage("router"):
            hits = _wait_for_line(logs_dir, name, "probe says")

    assert hits
    assert seen == ["router"]
    exit_line = [ln for ln in _lines(logs_dir, name) if "HOOK_EXIT" in ln][0]
    assert "status=ok" in exit_line


def test_current_stage_tracks_nesting():
    run = log_utils.HookRun()
    assert run.current_stage == ""
    with run.stage("outer"):
        assert run.current_stage == "outer"
        with run.stage("inner"):
            assert run.current_stage == "inner"
        assert run.current_stage == "outer"
    assert run.current_stage == ""


def test_watchdog_is_a_reserved_note_field(logs_dir):
    name = _unique("watchdog")
    logger = log_utils.setup_logging(name)

    with log_utils.hook_run(name, logger) as run:
        run.note(watchdog="spoofed")

    exit_line = [ln for ln in _lines(logs_dir, name) if "HOOK_EXIT" in ln][0]
    assert "note_watchdog=spoofed" in exit_line
    assert " watchdog=" not in exit_line
