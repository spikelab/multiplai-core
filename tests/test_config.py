"""Tests for multiplai_core.config — YAML/JSON helpers and session state."""

import json

import pytest

from multiplai_core import config


# ---------------------------------------------------------------------------
# load_yaml / save_yaml
# ---------------------------------------------------------------------------

def test_load_yaml_missing_returns_empty(tmp_path):
    assert config.load_yaml(tmp_path / "nope.yaml") == {}


def test_save_then_load_yaml_roundtrip(tmp_path):
    path = tmp_path / "sub" / "cfg.yaml"
    data = {"model": "claude-sonnet-4-6", "effort": 3, "nested": {"a": 1}}
    config.save_yaml(path, data)
    assert path.exists()  # parent dir created
    assert config.load_yaml(path) == data


def test_load_yaml_malformed_returns_empty(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("::: not : valid : yaml :::\n\t- broken")
    assert config.load_yaml(path) == {}


# ---------------------------------------------------------------------------
# read_memory_files
# ---------------------------------------------------------------------------

def test_read_memory_files_missing_dir(tmp_path):
    assert config.read_memory_files(tmp_path / "absent") == {}


def test_read_memory_files_reads_md_and_excludes(tmp_path):
    (tmp_path / "a.md").write_text("alpha")
    (tmp_path / "b.md").write_text("beta")
    (tmp_path / "skip.md").write_text("nope")
    (tmp_path / "ignore.txt").write_text("not markdown")

    out = config.read_memory_files(tmp_path, exclude={"skip.md"})
    assert out == {"a.md": "alpha", "b.md": "beta"}


# ---------------------------------------------------------------------------
# session state read/write
# ---------------------------------------------------------------------------

def test_read_session_state_missing_returns_none(tmp_path):
    assert config.read_session_state(tmp_path) is None


def test_write_then_read_session_state_roundtrip(tmp_path):
    state = {"session_id": "abc123", "turns": 7}
    assert config.write_session_state(tmp_path, state) is True
    assert config.read_session_state(tmp_path) == state


def test_write_session_state_is_atomic_no_tmp_left(tmp_path):
    config.write_session_state(tmp_path, {"x": 1})
    assert not (tmp_path / "session_state.json.tmp").exists()


def test_read_session_state_malformed_returns_none(tmp_path):
    (tmp_path / "session_state.json").write_text("{ not json")
    assert config.read_session_state(tmp_path) is None


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

def test_load_config_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        config.load_config(tmp_path / "missing.json")


def test_load_config_json(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"k": "v"}))
    assert config.load_config(path) == {"k": "v"}


def test_load_config_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("k: v\nn: 2\n")
    assert config.load_config(path) == {"k": "v", "n": 2}


def test_load_config_unsupported_format_raises(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("k = 'v'")
    with pytest.raises(ValueError):
        config.load_config(path)
