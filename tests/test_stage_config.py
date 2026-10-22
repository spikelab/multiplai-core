"""stage_config helpers, ported from walkthrough's test_config.py, plus the order a
pipeline's load_config applies its sources in, an unknown effort, and a
non-integer concurrency."""

import logging
from dataclasses import dataclass
from pathlib import Path

import pytest

from multiplai_core import (
    check_effort,
    coerce_int,
    conf_value,
    load_multiplai_conf,
    read_yaml_mapping,
)


# A pipeline's load_config, built only from the helpers, the way review's and
# walkthrough's config.py use them. The tests below pin the order of its sources.
@dataclass
class Cfg:
    concurrency: int = 4
    max_turns: int = 60
    explore_model: str | None = None
    write_model: str | None = None
    effort: str | None = None


def load_config(run_dir: Path | None) -> Cfg:
    cfg = Cfg()
    conf = load_multiplai_conf()
    for key in ("explore_model", "write_model"):
        setattr(cfg, key, conf_value(conf, f"walkthrough_{key}"))
    cfg.effort = conf_value(conf, "walkthrough_effort")
    path = (run_dir / "walkthrough.yaml") if run_dir else None
    data = read_yaml_mapping(path)
    for key in ("explore_model", "write_model", "effort"):
        if data.get(key):
            setattr(cfg, key, str(data[key]))
    for key in ("concurrency", "max_turns"):
        value = coerce_int(data, key, path)
        if value is not None:
            setattr(cfg, key, value)
    cfg.effort = check_effort(cfg.effort)
    return cfg


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(h))
    return h


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    return d


def test_defaults_without_conf_or_yaml(home, run_dir):
    cfg = load_config(run_dir)
    assert (cfg.concurrency, cfg.explore_model, cfg.effort, cfg.max_turns) == (
        4,
        None,
        None,
        60,
    )


def test_load_config_applies_defaults_then_conf_then_yaml(home, run_dir):
    (home / "multiplai.conf").write_text(
        "walkthrough_explore_model = sonnet\nwalkthrough_write_model = opus\nwalkthrough_effort = medium\n"
    )
    cfg = load_config(run_dir)
    assert (cfg.explore_model, cfg.write_model, cfg.effort, cfg.concurrency) == (
        "sonnet",
        "opus",
        "medium",
        4,
    )
    (run_dir / "walkthrough.yaml").write_text(
        "explore_model: haiku\neffort: high\nconcurrency: 2\n"
    )
    cfg = load_config(run_dir)
    # yaml beats conf where it sets a key; conf still beats the default where yaml is silent.
    assert (cfg.explore_model, cfg.write_model, cfg.effort, cfg.concurrency) == (
        "haiku",
        "opus",
        "high",
        2,
    )


def test_bad_integers_and_an_unknown_effort_are_dropped(home, run_dir):
    (run_dir / "walkthrough.yaml").write_text(
        "concurrency: 0\nmax_turns: lots\neffort: turbo\n"
    )
    cfg = load_config(run_dir)
    assert cfg.concurrency == 1  # clamped
    assert cfg.max_turns == 60  # not an integer: ignored
    assert cfg.effort is None  # unknown effort: dropped


def test_an_unknown_effort_in_conf_is_dropped_too(home, run_dir):
    (home / "multiplai.conf").write_text("walkthrough_effort = ludicrous\n")
    assert load_config(run_dir).effort is None


def test_unparseable_yaml_keeps_conf(home, run_dir):
    (home / "multiplai.conf").write_text("walkthrough_write_model = sonnet\n")
    (run_dir / "walkthrough.yaml").write_text("explore_model: [unclosed\n")
    assert load_config(run_dir).write_model == "sonnet"


# --- the helpers on their own ---------------------------------------------------


def test_conf_value_reads_either_case_and_strips():
    assert conf_value({"review_effort": "  high "}, "review_effort") == "high"
    assert conf_value({"REVIEW_EFFORT": "low"}, "review_effort") == "low"
    assert conf_value({"review_effort": "   "}, "review_effort") is None
    assert conf_value({"review_effort": 3}, "review_effort") is None
    assert conf_value({}, "review_effort") is None


def test_read_yaml_mapping_missing_or_none_is_empty_and_silent(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert read_yaml_mapping(tmp_path / "absent.yaml") == {}
        assert read_yaml_mapping(None) == {}
    assert not caplog.records


@pytest.mark.parametrize(
    "text", ["a: [unclosed\n", "- just\n- a list\n", "plain string\n"]
)
def test_read_yaml_mapping_bad_content_is_empty_with_a_warning(tmp_path, caplog, text):
    path = tmp_path / "review.yaml"
    path.write_text(text)
    with caplog.at_level(logging.WARNING, logger="multiplai_core.stage_config"):
        assert read_yaml_mapping(path) == {}
    assert "review.yaml" in caplog.text


def test_read_yaml_mapping_undecodable_file_is_empty_with_a_warning(tmp_path, caplog):
    path = tmp_path / "review.yaml"
    path.write_bytes(b"\xff\xfe\x00bad")
    with caplog.at_level(logging.WARNING, logger="multiplai_core.stage_config"):
        assert read_yaml_mapping(path) == {}
    assert "review.yaml" in caplog.text


def test_read_yaml_mapping_empty_file_is_empty(tmp_path):
    path = tmp_path / "review.yaml"
    path.write_text("")
    assert read_yaml_mapping(path) == {}


def test_coerce_int_non_integer_concurrency_is_none_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="multiplai_core.stage_config"):
        assert (
            coerce_int({"concurrency": "four"}, "concurrency", "run/review.yaml")
            is None
        )
        assert (
            coerce_int({"concurrency": [4]}, "concurrency", "run/review.yaml") is None
        )
    assert "concurrency='four' in run/review.yaml" in caplog.text


def test_coerce_int_clamps_to_the_minimum_and_ignores_absent_keys():
    assert coerce_int({"concurrency": "3"}, "concurrency", "p") == 3
    assert coerce_int({"concurrency": -2}, "concurrency", "p") == 1
    assert coerce_int({"max_turns": 5}, "max_turns", "p", minimum=10) == 10
    assert coerce_int({"concurrency": None}, "concurrency", "p") is None
    assert coerce_int({}, "concurrency", "p") is None


def test_check_effort(caplog):
    assert check_effort("high") == "high"
    assert check_effort("HIGH") == "HIGH"  # unchanged, as review's config kept it
    assert check_effort(None) is None and check_effort("") is None
    with caplog.at_level(logging.WARNING, logger="multiplai_core.stage_config"):
        assert check_effort("turbo") is None
    assert "Ignoring unknown effort 'turbo'" in caplog.text
