"""Tests for multiplai_core.fable_guard and its call in run_agent.

The usage endpoint is never called: ``_fetch`` is replaced by a fake that
returns JSON shaped like the reading of 2026-11-13 (``limits[]`` with a
``weekly_all`` entry and a ``weekly_scoped`` entry for Fable, plus
``five_hour``). The clock is fixed, so state files compare byte for byte.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
import urllib.error
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from _fakes import _FakeAssistantMessage, _FakeTextBlock, _make_mock_sdk

from multiplai_core import fable_guard as fg
from multiplai_core.agent_runner import AgentRunError, run_agent
from multiplai_core.env import CURRENT_MODEL

NOW = datetime(2026, 11, 14, 12, 0, tzinfo=UTC).timestamp()
TOKEN = "FAKE-OAUTH-TOKEN-that-must-never-appear-0123456789"
FABLE = "claude-fable-5-2"
OPUS = CURRENT_MODEL["opus"]


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def reading(
    *,
    fable: float = 0.0,
    fable_elapsed: float = 0.5,
    weekly: float = 0.0,
    weekly_elapsed: float = 0.5,
    five: float | None = None,
    now: float = NOW,
    fable_entry: bool = True,
) -> dict:
    """An endpoint answer. ``*_elapsed`` is the elapsed fraction of the 7-day window."""
    limits = [
        {
            "kind": "weekly_all",
            "percent": weekly,
            "resets_at": _iso(now + (1 - weekly_elapsed) * fg.WEEK_S),
            "scope": None,
            "is_active": True,
        }
    ]
    if fable_entry:
        limits.append(
            {
                "kind": "weekly_scoped",
                "percent": fable,
                "resets_at": _iso(now + (1 - fable_elapsed) * fg.WEEK_S),
                "scope": {"model": {"display_name": "Fable"}},
                "is_active": True,
            }
        )
    return {
        "limits": limits,
        "weekly_scoped_shares": [
            {
                "limit_index": 1,
                "used_percent_of_weekly": 0,
                "allowance_percent_of_weekly": 50,
            }
        ],
        "five_hour": None
        if five is None
        else {"utilization": five, "resets_at": _iso(now + 3600)},
    }


class Env:
    """One guard environment: its own state dir, conf, credentials, clock and endpoint."""

    def __init__(self, monkeypatch, root: Path):
        self.root = root
        self.mp = monkeypatch
        self.state_dir = root / "state"
        self.home = root / "home"
        self.config = root / "claude"
        self.home.mkdir(parents=True)
        self.config.mkdir(parents=True)
        (self.config / ".credentials.json").write_text(
            json.dumps({"claudeAiOauth": {"accessToken": TOKEN}})
        )
        self.answer: dict | Exception = reading()
        self.fetches = 0
        self.tokens_seen: list[str] = []
        self.now = NOW
        self.conf({"FABLE_GUARD": "enforce"})
        self.activate()

    def activate(self):
        self.mp.setattr(fg, "_state_dir", lambda: self.state_dir)
        self.mp.setenv("CLAUDE_MULTIPLAI_HOME", str(self.home))
        self.mp.setenv("CLAUDE_CONFIG_DIR", str(self.config))
        self.mp.setattr(fg.time, "time", lambda: self.now)
        self.mp.setattr(fg, "_fetch", self._fetch)

    def _fetch(self, token):
        self.fetches += 1
        self.tokens_seen.append(token)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer

    def conf(self, keys: dict):
        text = "".join(f"{k}={v}\n" for k, v in keys.items())
        (self.home / "multiplai.conf").write_text(text)

    def state_text(self) -> str:
        return (self.state_dir / "state.json").read_text()

    def log_rows(self) -> list[dict]:
        return fg.read_decisions()


@pytest.fixture
def env(monkeypatch, tmp_path):
    return Env(monkeypatch, tmp_path / "a")


@pytest.fixture
def make_env(monkeypatch, tmp_path):
    """A second environment for the run that never stopped."""

    def _make(name: str) -> Env:
        return Env(monkeypatch, tmp_path / name)

    return _make


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------


class TestRules:
    def test_rule1_fable_percent_at_max(self, env):
        env.answer = reading(fable=85, fable_elapsed=0.95)  # pace 0.89
        d = fg.decide(FABLE, "t")
        assert (d.model_used, d.reason, d.switched) == (OPUS, fg.R_FABLE_MAX, True)

    def test_rule2_fable_pace_over_max_above_min_percent(self, env):
        env.answer = reading(fable=30, fable_elapsed=0.2)  # pace 1.5
        d = fg.decide(FABLE, "t")
        assert (d.model_used, d.reason) == (OPUS, fg.R_FABLE_PACE)

    def test_rule2_needs_min_percent(self, env):
        env.answer = reading(fable=20, fable_elapsed=0.1)  # pace 2.0, under 25%
        d = fg.decide(FABLE, "t")
        assert (d.model_used, d.reason) == (FABLE, fg.R_OK)

    def test_rule3_weekly_pace_over_max_above_half(self, env):
        env.answer = reading(weekly=60, weekly_elapsed=0.5)  # pace 1.2
        d = fg.decide(FABLE, "t")
        assert (d.model_used, d.reason) == (OPUS, fg.R_WEEKLY_PACE)

    def test_rule3_needs_half_the_weekly_allowance(self, env):
        env.answer = reading(weekly=40, weekly_elapsed=0.2)  # pace 2.0, under 50%
        assert fg.decide(FABLE, "t").model_used == FABLE

    def test_rule4_five_hour_at_max(self, env):
        env.answer = reading(five=90)
        d = fg.decide(FABLE, "t")
        assert (d.model_used, d.reason) == (OPUS, fg.R_FIVE_HOUR)

    def test_no_fable_entry_leaves_rules_3_and_4(self, env):
        env.answer = reading(fable_entry=False, weekly=70, weekly_elapsed=0.5)
        d = fg.decide(FABLE, "t")
        assert d.reason == fg.R_WEEKLY_PACE

    def test_thresholds_come_from_the_conf(self, env):
        env.conf({"FABLE_GUARD": "enforce", "FABLE_GUARD_MAX_PERCENT": "60"})
        env.answer = reading(fable=65, fable_elapsed=0.95)
        assert fg.decide(FABLE, "t").reason == fg.R_FABLE_MAX

    def test_fallback_model_from_the_conf(self, env):
        env.conf({"FABLE_GUARD": "enforce", "FABLE_GUARD_FALLBACK": "sonnet"})
        env.answer = reading(fable=90)
        assert fg.decide(FABLE, "t").model_used == CURRENT_MODEL["sonnet"]

    def test_pace_formula(self):
        reset = _iso(NOW + 0.5 * fg.WEEK_S)  # half the window gone
        assert fg.pace(30, reset, NOW) == pytest.approx(0.6)
        assert fg.pace(None, reset, NOW) is None
        assert fg.pace(30, None, NOW) is None


class TestSwitchBack:
    def test_fable_returns_only_below_resume_pace(self, env):
        env.answer = reading(fable=30, fable_elapsed=0.2, weekly=10)  # pace 1.5
        assert fg.decide(FABLE, "t").model_used == OPUS
        assert json.loads(env.state_text())["backed_out"] is True

        # Rule 2 no longer holds (pace 0.86) but the pace is not below 0.8.
        env.now += 61
        env.answer = reading(fable=30, fable_elapsed=0.35, weekly=10, now=env.now)
        d = fg.decide(FABLE, "t")
        assert (d.model_used, d.reason) == (OPUS, fg.R_HOLD)

        # Pace 0.75 for Fable, 0.25 for the weekly limit: back to Fable.
        env.now += 61
        env.answer = reading(
            fable=30, fable_elapsed=0.4, weekly=10, weekly_elapsed=0.4, now=env.now
        )
        d = fg.decide(FABLE, "t")
        assert (d.model_used, d.reason) == (FABLE, fg.R_OK)
        state = json.loads(env.state_text())
        assert state["backed_out"] is False and state["since"] is None

    def test_without_back_out_the_resume_pace_does_not_apply(self, env):
        env.answer = reading(fable=30, fable_elapsed=0.35)  # pace 0.86
        assert fg.decide(FABLE, "t").model_used == FABLE


# ---------------------------------------------------------------------------
# Modes, unreadable endpoint, non-Fable models
# ---------------------------------------------------------------------------


class TestModes:
    def test_off_reads_nothing_and_logs_nothing(self, env):
        env.conf({"FABLE_GUARD": "off"})
        env.answer = reading(fable=95)
        d = fg.decide(FABLE, "t")
        assert d.model_used == FABLE and env.fetches == 0
        assert not fg.log_path().exists() and not fg.state_path().exists()

    def test_monitor_logs_but_keeps_the_model(self, env):
        env.conf({"FABLE_GUARD": "monitor"})
        env.answer = reading(fable=95)
        d = fg.decide(FABLE, "t")
        assert (d.model_used, d.switched) == (FABLE, False)
        [row] = env.log_rows()
        assert row["mode"] == "monitor" and row["reason"] == fg.R_FABLE_MAX
        assert row["used"] == FABLE and row["requested"] == FABLE
        # monitor tracks the back-out the way enforce would
        assert json.loads(env.state_text())["backed_out"] is True

    def test_monitor_is_the_default(self, env):
        env.conf({})
        env.answer = reading(fable=95)
        assert fg.load_settings().mode == "monitor"
        assert fg.decide(FABLE, "t").model_used == FABLE

    def test_enforce_logs_the_model_that_ran(self, env):
        env.answer = reading(fable=95, weekly=40, five=12)
        fg.decide(FABLE, "plan")
        [row] = env.log_rows()
        assert row["task"] == "plan" and row["used"] == OPUS
        assert row["fable_percent"] == 95 and row["weekly_percent"] == 40
        assert row["five_hour"] == 12
        assert set(row) == {
            "ts",
            "task",
            "requested",
            "used",
            "mode",
            "reason",
            "fable_percent",
            "fable_pace",
            "weekly_percent",
            "weekly_pace",
            "five_hour",
        }


class TestUnreadable:
    @pytest.mark.parametrize(
        ("setting", "used", "reason"),
        [
            ("fallback", OPUS, fg.R_UNREADABLE_FALLBACK),
            ("keep", FABLE, fg.R_UNREADABLE_KEEP),
        ],
    )
    def test_http_failure(self, env, setting, used, reason):
        env.conf({"FABLE_GUARD": "enforce", "FABLE_GUARD_ON_UNREADABLE": setting})
        env.answer = urllib.error.URLError("down")
        d = fg.decide(FABLE, "t")
        assert (d.model_used, d.reason) == (used, reason)

    @pytest.mark.parametrize(("setting", "used"), [("fallback", OPUS), ("keep", FABLE)])
    def test_missing_token(self, env, setting, used):
        env.conf({"FABLE_GUARD": "enforce", "FABLE_GUARD_ON_UNREADABLE": setting})
        (env.config / ".credentials.json").unlink()
        assert fg.decide(FABLE, "t").model_used == used
        assert env.fetches == 0

    def test_unparseable_answer_is_unreadable(self, env, monkeypatch):
        monkeypatch.setattr(fg, "parse_usage", lambda data: 1 / 0)
        assert fg.decide(FABLE, "t").reason == fg.R_UNREADABLE_FALLBACK

    def test_fallback_is_the_default(self, env):
        env.answer = ValueError("not json")
        assert fg.decide(FABLE, "t").model_used == OPUS


class TestNonFable:
    @pytest.mark.parametrize("model", [OPUS, "claude-sonnet-5-5", "opus", "haiku"])
    def test_passes_through_unchecked(self, env, model):
        env.answer = reading(fable=99)
        d = fg.decide(model, "t")
        assert d.model_used == model and env.fetches == 0
        assert not fg.log_path().exists()

    @pytest.mark.parametrize(
        "model", ["fable", "claude-fable-5-2", "claude-fable-5-1", "claude-mythos-5-2"]
    )
    def test_fable_names(self, model):
        assert fg.is_fable(model)


# ---------------------------------------------------------------------------
# The token
# ---------------------------------------------------------------------------


class TestToken:
    def test_token_goes_only_into_the_request(self, env, monkeypatch, capsys, caplog):
        caplog.set_level(logging.DEBUG)
        sent = {}

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(reading(fable=95)).encode()

        def _urlopen(request, timeout):
            sent["headers"] = dict(request.header_items())
            return _Resp()

        # The real _fetch, so the request it builds is checked too.
        monkeypatch.setattr(fg, "_fetch", _real_fetch)
        monkeypatch.setattr(fg.urllib.request, "urlopen", _urlopen)

        mock_sdk = _make_mock_sdk([_FakeAssistantMessage([_FakeTextBlock("ok")])])
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            asyncio.run(run_agent("hi", model="fable", label="plan"))
        assert sent["headers"]["Authorization"] == f"Bearer {TOKEN}"
        assert sent["headers"]["Anthropic-beta"] == fg.OAUTH_BETA

        # A failure whose text holds the token must not leak it either.
        def _fails(request, timeout):
            raise urllib.error.URLError(f"refused for {TOKEN}")

        monkeypatch.setattr(fg.urllib.request, "urlopen", _fails)
        env.now += 61
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            asyncio.run(run_agent("hi", model="fable", label="plan"))

        captured = capsys.readouterr()
        assert "FABLE-GUARD:" in captured.err
        assert TOKEN not in captured.err and TOKEN not in captured.out
        assert TOKEN not in caplog.text
        for path in env.state_dir.rglob("*"):
            if path.is_file():
                assert TOKEN not in path.read_text()
        fg.status()
        assert TOKEN not in capsys.readouterr().out


_real_fetch = fg._fetch


# ---------------------------------------------------------------------------
# run_agent
# ---------------------------------------------------------------------------


class TestRunAgent:
    def _sdk(self):
        return _make_mock_sdk([_FakeAssistantMessage([_FakeTextBlock("ok")])])

    def test_enforce_at_fable_85_runs_on_opus(self, env, capsys):
        env.answer = reading(fable=85)
        mock_sdk = self._sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            result = asyncio.run(run_agent("hi", model="fable", label="plan"))
        assert mock_sdk.ClaudeAgentOptions.call_args.kwargs["model"] == OPUS
        assert result.model == OPUS
        err = capsys.readouterr().err
        assert f"FABLE-GUARD: plan runs on {OPUS} instead of fable: " in err
        assert len(fg.log_path().read_text().splitlines()) == 1

    def test_monitor_keeps_the_requested_model(self, env, capsys):
        env.conf({"FABLE_GUARD": "monitor"})
        env.answer = reading(fable=85)
        mock_sdk = self._sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            result = asyncio.run(run_agent("hi", model="fable", label="plan"))
        assert mock_sdk.ClaudeAgentOptions.call_args.kwargs["model"] == "fable"
        assert result.model == "fable"
        assert "FABLE-GUARD" not in capsys.readouterr().err
        assert len(env.log_rows()) == 1

    def test_non_fable_model_is_not_checked(self, env):
        env.answer = reading(fable=99)
        mock_sdk = self._sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            result = asyncio.run(run_agent("hi", model="opus", label="plan"))
        assert result.model == "opus" and env.fetches == 0

    def test_guard_failure_keeps_the_model(self, env, monkeypatch):
        monkeypatch.setattr(fg, "decide", lambda m, t: 1 / 0)
        mock_sdk = self._sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            result = asyncio.run(run_agent("hi", model=FABLE, label="plan"))
        assert result.model == FABLE

    def test_checked_once_per_call(self, env):
        env.answer = reading(fable=85)
        mock_sdk = _make_mock_sdk(
            [
                _FakeAssistantMessage([_FakeTextBlock("a")]),
                _FakeAssistantMessage([_FakeTextBlock("b")]),
            ]
        )
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            asyncio.run(run_agent("hi", model=FABLE, label="plan", max_turns=5))
        assert len(env.log_rows()) == 1


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class TestCache:
    def test_reading_reused_inside_the_ttl(self, env):
        env.answer = reading(fable=10)
        fg.decide(FABLE, "t")
        env.now += 30
        fg.decide(FABLE, "t")
        assert env.fetches == 1
        env.now += 31
        fg.decide(FABLE, "t")
        assert env.fetches == 2

    def test_ttl_from_the_conf(self, env):
        env.conf({"FABLE_GUARD": "enforce", "FABLE_GUARD_TTL_S": "5"})
        fg.decide(FABLE, "t")
        env.now += 6
        fg.decide(FABLE, "t")
        assert env.fetches == 2


# ---------------------------------------------------------------------------
# Stop and resume rows S1-S9 (plan-fable-5-2-adaptive-2026-11-13.md). Each case
# compares the state with a run that never stopped.
# ---------------------------------------------------------------------------


def _clean_run(make_env, steps) -> Env:
    """Run *steps* (``(answer, now_offset)`` pairs) in a fresh environment."""
    clean = make_env("clean")
    for answer, offset in steps:
        clean.now = NOW + offset
        clean.answer = answer
        fg.decide(FABLE, "t")
    return clean


class TestResumeRows:
    def test_S1_killed_before_os_replace(self, make_env):
        steps = [(reading(fable=90), 0), (reading(fable=10), 120)]
        clean = _clean_run(make_env, steps)
        clean_state = clean.state_text()

        e = make_env("stopped")
        e.answer = reading(fable=90)
        fg.decide(FABLE, "t")
        # The kill: a temp file written, never renamed, now two hours old,
        # plus one written a minute ago by a writer still running.
        old = e.state_dir / "state.json.999-deadbeef.tmp"
        old.write_text('{"backed_out": false}')
        os.utime(old, (time.time() - 7200, time.time() - 7200))
        recent = e.state_dir / "state.json.998-cafebabe.tmp"
        recent.write_text("{}")
        e.now, e.answer = NOW + 120, reading(fable=10)
        fg.decide(FABLE, "t")
        assert e.state_text() == clean_state
        assert not old.exists() and recent.exists()

    def test_S2_corrupt_state_is_absent(self, make_env):
        clean = _clean_run(make_env, [(reading(fable=90), 0)])
        e = make_env("stopped")
        e.state_dir.mkdir(parents=True)
        (e.state_dir / "state.json").write_text('{"backed_out": tru')
        e.answer = reading(fable=90)
        assert fg.decide(FABLE, "t").model_used == OPUS
        assert e.fetches == 1
        assert e.state_text() == clean.state_text()

    def test_S3_cut_log_line(self, make_env):
        clean = _clean_run(make_env, [(reading(fable=90), 0)])
        clean_rows = clean.log_rows()
        e = make_env("stopped")
        e.state_dir.mkdir(parents=True)
        fg.log_path().write_text('{"ts": "2026-11-14T11:00:00Z", "reas')
        e.answer = reading(fable=90)
        fg.decide(FABLE, "t")
        assert e.log_rows() == clean_rows
        lines = fg.log_path().read_text().splitlines()
        assert json.loads(lines[-1]) == clean_rows[0]
        assert fg.log_path().read_text().endswith("\n")

    def test_S4_conf_edited_between_steps(self, make_env):
        tight = {"FABLE_GUARD": "enforce", "FABLE_GUARD_MAX_PERCENT": "60"}
        clean = make_env("clean")
        clean.conf(tight)
        clean.answer = reading(fable=70, fable_elapsed=0.95)
        clean_d = fg.decide(FABLE, "t")

        e = make_env("stopped")
        e.answer = reading(fable=70, fable_elapsed=0.95)
        assert fg.decide(FABLE, "t").model_used == FABLE
        e.conf(tight)
        assert fg.decide(FABLE, "t") == clean_d
        assert e.state_text() == clean.state_text()

        # backed_out is re-evaluated under the new values
        e.conf({"FABLE_GUARD": "enforce", "FABLE_GUARD_RESUME_PACE": "2"})
        assert fg.decide(FABLE, "t").model_used == FABLE

    def test_S5_state_deleted_between_steps(self, make_env):
        clean = _clean_run(make_env, [(reading(fable=10), 120)])
        e = make_env("stopped")
        e.answer = reading(fable=90)
        fg.decide(FABLE, "t")
        (e.state_dir / "state.json").unlink()
        e.now, e.answer = NOW + 120, reading(fable=10)
        assert fg.decide(FABLE, "t").model_used == FABLE
        assert e.state_text() == clean.state_text()

    def test_S6_unreadable_twice(self, make_env):
        clean = make_env("clean")
        clean.answer = reading(fable=10)
        fg.decide(FABLE, "t")
        clean_state = clean.state_text()

        e = make_env("stopped")
        e.answer = reading(fable=10)
        fg.decide(FABLE, "t")
        # Inside the TTL a failing endpoint is never asked: the cache is used.
        e.answer = urllib.error.URLError("down")
        e.now += 30
        assert fg.decide(FABLE, "t").reason == fg.R_OK
        assert e.fetches == 1
        # Past the TTL each call asks once and applies ON_UNREADABLE alone.
        for i in (1, 2):
            e.now += 61
            assert fg.decide(FABLE, "t").reason == fg.R_UNREADABLE_FALLBACK
            assert e.fetches == 1 + i
        state = json.loads(e.state_text())
        assert state["reading"] == json.loads(clean_state)["reading"]
        assert state["backed_out"] is False

    def test_S7_weekly_window_reset(self, make_env):
        after = NOW + 2 * fg.WEEK_S / 7  # two days later, past the reset
        # Fable pace 0.83: above the resume pace, so only the reset brings Fable back.
        fresh = reading(fable=5, fable_elapsed=0.06, weekly=5, now=after)
        clean = _clean_run(make_env, [(fresh, after - NOW)])

        e = make_env("stopped")
        e.answer = reading(fable=90, fable_elapsed=0.9)  # resets in 0.7 days
        assert fg.decide(FABLE, "t").model_used == OPUS
        e.now, e.answer = after, fresh
        d = fg.decide(FABLE, "t")
        assert (d.model_used, d.reason) == (FABLE, fg.R_RESET)
        assert e.fetches == 2
        assert e.state_text() == clean.state_text()
        assert e.log_rows()[-1]["reason"] == "window reset"

    def test_S8_step_killed_after_choosing_opus(self, make_env):
        good = reading(fable=10, fable_elapsed=0.5, weekly=10)
        clean = _clean_run(make_env, [(reading(fable=90), 0), (good, 120)])

        e = make_env("stopped")
        e.answer = reading(fable=90)
        failing = _make_mock_sdk(fail=RuntimeError("killed"))
        with (
            patch.dict(sys.modules, {"claude_agent_sdk": failing}),
            pytest.raises(AgentRunError),
        ):
            asyncio.run(run_agent("hi", model=FABLE, label="t"))
        e.now, e.answer = NOW + 120, good
        ok = _make_mock_sdk([_FakeAssistantMessage([_FakeTextBlock("ok")])])
        with patch.dict(sys.modules, {"claude_agent_sdk": ok}):
            result = asyncio.run(run_agent("hi", model=FABLE, label="t"))
        assert result.model == FABLE
        assert e.state_text() == clean.state_text()
        assert [r["used"] for r in e.log_rows()] == [OPUS, FABLE]

    def test_S9_concurrent_decisions(self, make_env):
        clean = _clean_run(make_env, [(reading(fable=90), 0)])
        e = make_env("stopped")
        e.answer = reading(fable=90)
        results: list[fg.Decision] = []
        barrier = threading.Barrier(16)

        def worker():
            barrier.wait()
            results.append(fg.decide(FABLE, "t"))

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert {r.model_used for r in results} == {OPUS}
        assert e.state_text() == clean.state_text()
        assert len(e.log_rows()) == 16
        assert not list(e.state_dir.glob("*.tmp"))


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_prints_percentages_paces_and_next_decision(self, env, capsys):
        env.answer = reading(fable=30, fable_elapsed=0.2, weekly=29)
        assert fg.main(["status"]) == 0
        out = capsys.readouterr().out
        assert "fable:          30.0%  pace 1.50" in out
        assert "weekly (all):   29.0%  pace 0.58" in out
        assert f"runs on {OPUS} ({fg.R_FABLE_PACE})" in out
        assert not fg.state_path().exists() and not fg.log_path().exists()

    def test_counts_decisions_and_skips_cut_lines(self, env, capsys):
        env.answer = reading(fable=90)
        fg.decide(FABLE, "t")
        with fg.log_path().open("a") as f:
            f.write('{"cut')
        fg.status()
        out = capsys.readouterr().out
        assert "decisions, last 7 days: 1" in out
        assert f"1  {fg.R_FABLE_MAX}" in out

    def test_unreadable(self, env, capsys):
        env.answer = urllib.error.URLError("down")
        fg.status()
        assert "endpoint not readable" in capsys.readouterr().out

    def test_bad_arguments(self, capsys):
        assert fg.main([]) == 2
