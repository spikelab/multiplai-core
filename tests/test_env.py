"""Tests for multiplai_core.env."""

import os

import pytest

from multiplai_core import env as env_mod
from multiplai_core.env import (
    CURRENT_MODEL,
    env_candidates,
    find_project_root,
    load_multiplai_conf,
    pick_model,
    resolve_effort,
    resolve_model,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("MULTIPLAI_ENV_FILE", "CLAUDE_MULTIPLAI_HOME", "CLAUDE_CONFIG_DIR",
              "MULTIPLAI_MODEL", "MULTIPLAI_EFFORT"):
        monkeypatch.delenv(k, raising=False)


class TestResolveModel:
    def test_downgrades_above_ceiling(self):
        assert resolve_model("claude-opus-4", "claude-sonnet-4-6") == "claude-sonnet-4-6"

    def test_keeps_at_or_below_ceiling(self):
        assert resolve_model("claude-haiku-4", "claude-sonnet-4-6") == "claude-haiku-4"

    def test_ceiling_from_env(self, monkeypatch):
        monkeypatch.setenv("MULTIPLAI_MODEL", "claude-haiku-4-5")
        assert resolve_model("claude-opus-4") == "claude-haiku-4-5"

    def test_default_ceiling_sonnet(self):
        assert resolve_model("claude-opus-4") == "claude-sonnet-4-6"


class TestResolveEffort:
    def test_downgrades(self):
        assert resolve_effort("max", "medium") == "medium"

    def test_keeps_below(self):
        assert resolve_effort("low", "high") == "low"

    def test_default_ceiling_high(self):
        assert resolve_effort("max") == "high"


class TestEnvCandidates:
    def test_explicit_first(self, monkeypatch, tmp_path):
        f = tmp_path / "custom.env"
        monkeypatch.setenv("MULTIPLAI_ENV_FILE", str(f))
        assert env_candidates()[0] == f

    def test_home_included(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        assert (tmp_path / ".env") in env_candidates()

    def test_cwd_included(self):
        assert (env_mod.Path.cwd() / ".env") in env_candidates()


class TestFindProjectRoot:
    def test_kit_marker(self, tmp_path):
        (tmp_path / ".env.example").write_text("")
        (tmp_path / "dotfiles").mkdir()
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        assert find_project_root(sub) == tmp_path

    def test_env_fallback(self, tmp_path):
        (tmp_path / ".env").write_text("")
        sub = tmp_path / "a"
        sub.mkdir()
        assert find_project_root(sub) == tmp_path

    def test_none_when_absent(self, tmp_path):
        assert find_project_root(tmp_path) is None


class TestLoadMultiplaiConf:
    def test_sections_and_globals(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        (tmp_path / "multiplai.conf").write_text(
            'MULTIPLAI_MODEL="claude-opus-4-6"\n'
            "[deep-research]\n"
            "MODEL=sonnet\n"
        )
        conf = load_multiplai_conf()
        assert conf["MULTIPLAI_MODEL"] == "claude-opus-4-6"
        assert conf["_sections"]["deep-research"]["MODEL"] == "sonnet"

    def test_missing_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        assert load_multiplai_conf() == {"_sections": {}}

    def test_dotted_section_name(self, monkeypatch, tmp_path):
        # Dotted task keys must parse as section names — pick_model's per-task
        # override channel depends on this.
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        (tmp_path / "multiplai.conf").write_text(
            "[deep-research.parse]\nMODEL=sonnet\n"
        )
        conf = load_multiplai_conf()
        assert conf["_sections"]["deep-research.parse"]["MODEL"] == "sonnet"


def _write_conf(tmp_path, text):
    (tmp_path / "multiplai.conf").write_text(text)


class TestPickModel:
    def test_default_tier_opus(self, monkeypatch, tmp_path):
        # Ceiling of opus allows the opus family through verbatim.
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, 'MULTIPLAI_MODEL="claude-opus-4-6"\n')
        assert pick_model() == CURRENT_MODEL["opus"]

    def test_explicit_sonnet_tier(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, 'MULTIPLAI_MODEL="claude-opus-4-6"\n')
        assert pick_model("sonnet") == CURRENT_MODEL["sonnet"]

    def test_task_override_wins_over_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(
            tmp_path,
            'MULTIPLAI_MODEL="claude-opus-4-6"\n'
            "[deep-research.parse]\nMODEL=sonnet\n",
        )
        assert pick_model("opus", task="deep-research.parse") == CURRENT_MODEL["sonnet"]

    def test_missing_task_section_falls_back_to_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, 'MULTIPLAI_MODEL="claude-opus-4-6"\n')
        assert pick_model("opus", task="unconfigured.task") == CURRENT_MODEL["opus"]

    def test_ceiling_caps_the_tier(self, monkeypatch, tmp_path):
        # A sonnet ceiling downgrades an opus request to the ceiling string.
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, 'MULTIPLAI_MODEL="claude-sonnet-4-6"\n')
        assert pick_model("opus") == "claude-sonnet-4-6"

    def test_override_accepts_full_model_id(self, monkeypatch, tmp_path):
        # A dated ID in the override normalizes to its family.
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(
            tmp_path,
            'MULTIPLAI_MODEL="claude-opus-4-6"\n'
            "[buildme.implementer]\nMODEL=claude-opus-4-8\n",
        )
        assert pick_model("sonnet", task="buildme.implementer") == CURRENT_MODEL["opus"]

    def test_unknown_default_tier_falls_back_to_opus(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, 'MULTIPLAI_MODEL="claude-opus-4-6"\n')
        assert pick_model("bogus") == CURRENT_MODEL["opus"]
