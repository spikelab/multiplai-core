"""Tests for lib/log_utils.py — standard-compliant logging + activity stream.

Covers env-driven level resolution, the standard line format, session-id
stamping, the shared hook-errors sink, and the log_event() dual-sink
activity stream (human + JSONL, failure-safe, retention).
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime, timedelta, timezone

import pytest

import multiplai_core.log_utils as log_utils
from multiplai_core.paths import _reset_cache


@pytest.fixture
def logs_dir(tmp_path, monkeypatch):
    """Point the path resolver at a tmp data dir; yield its logs/ dir."""
    monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path / "plugin"))
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path / "data"))
    _reset_cache()
    log_utils._swept = False
    monkeypatch.delenv("MULTIPLAI_LOG_RETENTION_DAYS", raising=False)
    yield tmp_path / "data" / "logs"
    _reset_cache()


def _unique(name: str) -> str:
    return f"{name}-{uuid.uuid4().hex[:8]}"


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

    line = (logs_dir / f"{name}.log").read_text().strip().splitlines()[-1]
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
    line = (logs_dir / f"{name}.log").read_text().strip().splitlines()[-1]
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


# --------------------------------------------------------------------------
# log_event
# --------------------------------------------------------------------------

def test_log_event_writes_human_and_jsonl(logs_dir):
    log_utils.log_event(
        "context", "inject", "injected 2 memory · 0 skills",
        session_id="sess1234xx", memory=2, files=["a.md", "b.md"],
    )

    # Current file is undated — no date in the filename.
    human = (logs_dir / "activity.log").read_text().strip()
    assert re.match(
        r"^\d{2}:\d{2}:\d{2}Z \[sess1234\] \[context\] "
        r"injected 2 memory · 0 skills$",
        human,
    ), human

    # Structured fields live only in the JSONL mirror.
    assert "memory=2" not in human

    rec = json.loads((logs_dir / "activity.jsonl").read_text().strip())
    assert rec["component"] == "context"
    assert rec["event"] == "inject"
    assert rec["session"] == "sess1234"
    assert rec["memory"] == 2
    assert rec["files"] == ["a.md", "b.md"]
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", rec["ts"])


def test_log_event_never_raises(monkeypatch):
    def boom():
        raise RuntimeError("no logs dir")

    monkeypatch.setattr(log_utils, "_get_logs_dir", boom)
    # Must not propagate — observability cannot break the hook.
    log_utils.log_event("x", "y", "z")


def test_log_event_prunes_old_activity(logs_dir):
    logs_dir.mkdir(parents=True, exist_ok=True)
    old = logs_dir / "activity-2000-01-01.log"
    old.write_text("ancient\n")
    ancient = datetime(2000, 1, 1, tzinfo=timezone.utc).timestamp()
    os.utime(old, (ancient, ancient))

    log_utils._swept = False
    log_utils.log_event("session", "start", "fresh")

    assert not old.exists()
    assert (logs_dir / "activity.log").exists()


def test_log_event_appends_multiple_records(logs_dir):
    for i in range(3):
        log_utils.log_event("diary", "write", f"entry {i}")
    lines = (logs_dir / "activity.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    assert [json.loads(l)["msg"] for l in lines] == ["entry 0", "entry 1", "entry 2"]


# --------------------------------------------------------------------------
# date-rotation + naming convention
# --------------------------------------------------------------------------

def test_log_event_rotates_prior_day_to_dated_file(logs_dir):
    """A current activity.log from a prior UTC day rotates to
    activity-<that-day>.log; today's write starts a fresh activity.log."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    cur = logs_dir / "activity.log"
    curj = logs_dir / "activity.jsonl"
    cur.write_text("12:00:00 [old] yesterday line\n")
    curj.write_text('{"msg": "yesterday"}\n')
    yday = datetime.now(timezone.utc) - timedelta(days=1)
    stamp = yday.timestamp()
    os.utime(cur, (stamp, stamp))
    os.utime(curj, (stamp, stamp))
    ystr = yday.strftime("%Y-%m-%d")

    log_utils.log_event("session", "start", "today line")

    # Yesterday's content archived under the dated (infix) name.
    assert (logs_dir / f"activity-{ystr}.log").read_text() == \
        "12:00:00 [old] yesterday line\n"
    assert (logs_dir / f"activity-{ystr}.jsonl").read_text() == \
        '{"msg": "yesterday"}\n'
    # The rejected stdlib form must never appear.
    assert not (logs_dir / f"activity.log.{ystr}").exists()
    # Current file holds only today's line.
    assert "today line" in (logs_dir / "activity.log").read_text()
    assert "yesterday line" not in (logs_dir / "activity.log").read_text()


def test_setup_logging_rotates_stale_current_on_init(logs_dir):
    """A <name>.log left from a prior day is archived when the next
    run constructs the handler — no date suffix on the live file."""
    name = _unique("rot")
    logs_dir.mkdir(parents=True, exist_ok=True)
    cur = logs_dir / f"{name}.log"
    cur.write_text("[old] line\n")
    yday = datetime.now(timezone.utc) - timedelta(days=2)
    stamp = yday.timestamp()
    os.utime(cur, (stamp, stamp))

    logger = log_utils.setup_logging(name)
    logger.info("new line")
    for h in logger.handlers:
        h.flush()

    dated = logs_dir / f"{name}-{yday.strftime('%Y-%m-%d')}.log"
    assert dated.read_text() == "[old] line\n"
    assert "new line" in (logs_dir / f"{name}.log").read_text()


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
    monkeypatch.delenv("MULTIPLAI_LOG_RETENTION_DAYS", raising=False)
    assert log_utils.retention_days() == 7
    monkeypatch.setenv("MULTIPLAI_LOG_RETENTION_DAYS", "30")
    assert log_utils.retention_days() == 30
    monkeypatch.setenv("MULTIPLAI_LOG_RETENTION_DAYS", "0")
    assert log_utils.retention_days() == 0
    monkeypatch.setenv("MULTIPLAI_LOG_RETENTION_DAYS", "garbage")
    assert log_utils.retention_days() == 7
    monkeypatch.setenv("MULTIPLAI_LOG_RETENTION_DAYS", "-5")
    assert log_utils.retention_days() == 7


def test_retention_zero_keeps_dated_files_forever(logs_dir, monkeypatch):
    logs_dir.mkdir(parents=True, exist_ok=True)
    old = logs_dir / "activity-2000-01-01.log"
    old.write_text("ancient\n")
    ancient = datetime(2000, 1, 1, tzinfo=timezone.utc).timestamp()
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
    ancient = datetime(2000, 1, 1, tzinfo=timezone.utc).timestamp()
    os.utime(drop_old, (ancient, ancient))

    log_utils._sweep_logs(logs_dir, 7)

    assert keep_current.exists()
    assert keep_recent.exists()
    assert not drop_old.exists()
