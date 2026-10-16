"""Tests for multiplai_core.session — the default model and effort."""

import json
import logging
import os
import subprocess
import sys

import pytest

import multiplai_core.session as session
from multiplai_core.env import CURRENT_MODEL
from multiplai_core.session import (
    default_effort,
    default_model,
    default_model_source,
    default_effort_source,
    pin_session_settings,
    session_effort,
    session_model,
)

SID = "11111111-2222-3333-4444-555555555555"


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    d = tmp_path / "claude-config"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(d))
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", SID)
    return d


def _assistant(model, *, sidechain=False):
    return {"type": "assistant", "isSidechain": sidechain,
            "message": {"model": model, "role": "assistant", "content": []}}


def _write_transcript(config_dir, entries, project="-Users-x-some-dir"):
    path = config_dir / "projects" / project / f"{SID}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    return path


@pytest.fixture(autouse=True)
def _reset_logged(monkeypatch):
    monkeypatch.setattr(session, "_logged", set())
    monkeypatch.setattr(session, "_warned_no_model", False)


class TestSessionModel:
    def test_env_override_wins(self, config_dir, monkeypatch):
        _write_transcript(config_dir, [_assistant("claude-from-transcript")])
        monkeypatch.setenv("MULTIPLAI_SESSION_MODEL", "claude-pinned")
        assert session_model() == "claude-pinned"

    def test_reads_the_last_assistant_model(self, config_dir):
        _write_transcript(config_dir, [
            {"type": "user", "message": {"content": "hi"}},
            _assistant("claude-first"),
            {"type": "user", "message": {"content": "again"}},
            _assistant("claude-last"),
            {"type": "user", "message": {"content": "tool result"}},
        ])
        assert session_model() == "claude-last"

    def test_skips_synthetic_and_sidechain_turns(self, config_dir):
        _write_transcript(config_dir, [
            _assistant("claude-real"),
            _assistant("claude-subagent", sidechain=True),
            _assistant("<synthetic>"),
        ])
        assert session_model() == "claude-real"

    def test_finds_a_model_far_back_in_a_large_file(self, config_dir):
        filler = [{"type": "user", "message": {"content": "x" * 5000}}] * 100
        _write_transcript(config_dir, [_assistant("claude-early"), *filler])
        assert session_model() == "claude-early"

    def test_any_project_directory_matches(self, config_dir):
        _write_transcript(config_dir, [_assistant("claude-m")], project="odd[dir]name")
        assert session_model() == "claude-m"

    def test_missing_transcript_gives_none(self, config_dir):
        assert session_model() is None

    def test_no_session_id_gives_none(self, config_dir, monkeypatch):
        _write_transcript(config_dir, [_assistant("claude-m")])
        monkeypatch.delenv("CLAUDE_CODE_SESSION_ID")
        assert session_model() is None

    def test_garbage_lines_are_ignored(self, config_dir):
        path = _write_transcript(config_dir, [_assistant("claude-ok")])
        with path.open("a") as fh:
            fh.write('{"type": "assistant", "message": {"model": \n')
            fh.write("not json at all\n")
        assert session_model() == "claude-ok"


class TestSessionEffort:
    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_EFFORT", "low")
        monkeypatch.setenv("MULTIPLAI_SESSION_EFFORT", "max")
        assert session_effort() == "max"

    def test_claude_effort(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_EFFORT", "medium")
        assert session_effort() == "medium"

    def test_unknown_effort_gives_none(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_EFFORT", "turbo")
        assert session_effort() is None

    def test_unset_gives_none(self):
        assert session_effort() is None


class TestPin:
    def test_pins_both_into_the_environment(self, config_dir, monkeypatch):
        _write_transcript(config_dir, [_assistant("claude-m")])
        monkeypatch.setenv("CLAUDE_EFFORT", "xhigh")
        assert pin_session_settings() == ("claude-m", "xhigh")
        assert os.environ["MULTIPLAI_SESSION_MODEL"] == "claude-m"
        assert os.environ["MULTIPLAI_SESSION_EFFORT"] == "xhigh"

    def test_nothing_found_pins_nothing(self, config_dir):
        assert pin_session_settings() == (None, None)
        assert "MULTIPLAI_SESSION_MODEL" not in os.environ
        assert "MULTIPLAI_SESSION_EFFORT" not in os.environ

    def test_pinned_value_wins_over_a_fresh_transcript_read(self, config_dir):
        """S13: the session moved to another model after the parent pinned."""
        path = _write_transcript(config_dir, [_assistant("claude-at-start")])
        pin_session_settings()
        with path.open("a") as fh:
            fh.write(json.dumps(_assistant("claude-switched")) + "\n")
        assert session_model() == "claude-at-start"
        assert default_model() == "claude-at-start"

    def test_child_process_sees_the_pinned_value(self, config_dir, monkeypatch):
        """S13 across a process boundary; MULTIPLAI_MODEL still wins over it."""
        path = _write_transcript(config_dir, [_assistant("claude-at-start")])
        pin_session_settings()
        with path.open("a") as fh:
            fh.write(json.dumps(_assistant("claude-switched")) + "\n")
        code = "from multiplai_core.session import default_model; print(default_model())"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env={**os.environ}, check=True)
        assert out.stdout.strip() == "claude-at-start"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, check=True,
                             env={**os.environ, "MULTIPLAI_MODEL": "claude-test-default"})
        assert out.stdout.strip() == "claude-test-default"


def _conf(tmp_path, monkeypatch, text):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / "multiplai.conf").write_text(text)
    monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(home))


class TestDefaults:
    @pytest.fixture(autouse=True)
    def _session(self, monkeypatch):
        monkeypatch.setenv("MULTIPLAI_SESSION_MODEL", "claude-test-session")
        monkeypatch.setenv("MULTIPLAI_SESSION_EFFORT", "low")

    def test_env_wins(self, monkeypatch):
        monkeypatch.setenv("MULTIPLAI_MODEL", "claude-test-default")
        monkeypatch.setenv("MULTIPLAI_EFFORT", "high")
        assert default_model_source() == ("claude-test-default", "MULTIPLAI_MODEL")
        assert default_effort_source() == ("high", "MULTIPLAI_EFFORT")

    def test_env_wins_over_conf(self, tmp_path, monkeypatch):
        _conf(tmp_path, monkeypatch, 'MULTIPLAI_MODEL="claude-conf"\nMULTIPLAI_EFFORT="max"\n')
        monkeypatch.setenv("MULTIPLAI_MODEL", "claude-env")
        assert default_model() == "claude-env"
        assert default_effort() == "max"

    def test_conf_only(self, tmp_path, monkeypatch):
        _conf(tmp_path, monkeypatch, 'MULTIPLAI_MODEL="claude-opus-4-6"\nMULTIPLAI_EFFORT="medium"\n')
        # A dated ID passes through exactly as written.
        assert default_model_source() == ("claude-opus-4-6", "multiplai.conf")
        assert default_effort_source() == ("medium", "multiplai.conf")

    def test_family_alias_expands(self, monkeypatch):
        monkeypatch.setenv("MULTIPLAI_MODEL", "sonnet")
        assert default_model() == CURRENT_MODEL["sonnet"]

    def test_empty_string_counts_as_unset(self, tmp_path, monkeypatch):
        _conf(tmp_path, monkeypatch, 'MULTIPLAI_MODEL=""\nMULTIPLAI_EFFORT=""\n')
        monkeypatch.setenv("MULTIPLAI_MODEL", "")
        monkeypatch.setenv("MULTIPLAI_EFFORT", "  ")
        assert default_model_source() == ("claude-test-session", "MULTIPLAI_SESSION_MODEL")
        assert default_effort_source() == ("low", "MULTIPLAI_SESSION_EFFORT")

    def test_unset_falls_to_the_session(self):
        assert default_model() == "claude-test-session"
        assert default_effort() == "low"

    def test_model_set_effort_unset_keeps_the_session_effort(self, monkeypatch):
        monkeypatch.setenv("MULTIPLAI_MODEL", "claude-test-default")
        assert default_model() == "claude-test-default"
        assert default_effort() == "low"

    def test_unknown_multiplai_effort_falls_to_the_session(self, monkeypatch, caplog):
        monkeypatch.setenv("MULTIPLAI_EFFORT", "turbo")
        with caplog.at_level(logging.WARNING, logger="multiplai_core.session"):
            assert default_effort() == "low"
        assert "turbo" in caplog.text

    def test_transcript_source_is_named(self, config_dir, monkeypatch):
        monkeypatch.delenv("MULTIPLAI_SESSION_MODEL")
        monkeypatch.delenv("MULTIPLAI_SESSION_EFFORT")
        monkeypatch.setenv("CLAUDE_EFFORT", "high")
        _write_transcript(config_dir, [_assistant("claude-m")])
        assert default_model_source() == ("claude-m", "session transcript")
        assert default_effort_source() == ("high", "CLAUDE_EFFORT")


class TestLogging:
    def test_each_result_is_logged_once(self, monkeypatch, caplog):
        monkeypatch.setenv("MULTIPLAI_SESSION_MODEL", "claude-test-session")
        with caplog.at_level(logging.INFO, logger="multiplai_core.session"):
            default_model()
            default_model()
        lines = [r for r in caplog.records if "claude-test-session" in r.getMessage()]
        assert len(lines) == 1
        assert "MULTIPLAI_SESSION_MODEL" in lines[0].getMessage()

    def test_no_model_anywhere_warns_once(self, caplog):
        with caplog.at_level(logging.WARNING, logger="multiplai_core.session"):
            assert default_model() is None
            assert default_model() is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "CLI's default" in warnings[0].getMessage()
