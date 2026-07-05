"""Tests for multiplai_core.env."""

import os

import pytest

from multiplai_core import env as env_mod
from multiplai_core.env import (
    env_candidates,
    find_project_root,
    load_multiplai_conf,
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
