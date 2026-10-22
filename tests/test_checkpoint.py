"""save_model / load_model: the atomic checkpoint the review and walkthrough `state.py` files shared."""

import logging
import os

import pytest
from pydantic import BaseModel

from multiplai_core import load_model, save_model


class State(BaseModel):
    stage: str
    done: list[str] = []


def test_round_trip_creates_the_directory(tmp_path):
    target = tmp_path / "out" / "target"
    path = save_model(State(stage="verify", done=["find"]), target, "review-state.json")
    assert path == target / "review-state.json"
    assert load_model(path, State) == State(stage="verify", done=["find"])


def test_save_replaces_the_previous_checkpoint_and_leaves_no_temp_file(tmp_path):
    save_model(State(stage="find"), tmp_path, "walkthrough-state.json")
    save_model(State(stage="write"), tmp_path, "walkthrough-state.json")
    assert sorted(os.listdir(tmp_path)) == ["walkthrough-state.json"]
    assert load_model(tmp_path / "walkthrough-state.json", State).stage == "write"


def test_a_failed_write_keeps_the_previous_checkpoint_and_cleans_up(tmp_path):
    save_model(State(stage="find"), tmp_path, "s.json")

    class Broken:
        def model_dump_json(self, **_):
            raise RuntimeError("killed mid-write")

    with pytest.raises(RuntimeError):
        save_model(Broken(), tmp_path, "s.json")
    assert sorted(os.listdir(tmp_path)) == ["s.json"]
    assert load_model(tmp_path / "s.json", State).stage == "find"


def test_the_temp_file_sits_in_the_target_directory(tmp_path, monkeypatch):
    seen = {}
    real_replace = os.replace

    def spy(src, dst):
        seen["src"] = src
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    save_model(State(stage="x"), tmp_path, "review-state.json")
    assert os.path.dirname(seen["src"]) == str(tmp_path)
    assert os.path.basename(seen["src"]).startswith(".review-state-")


def test_a_missing_file_is_none_and_logged(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="multiplai_core.checkpoint"):
        assert load_model(tmp_path / "nope.json", State) is None
    assert "nope.json" in caplog.text


@pytest.mark.parametrize("content", ["{not json", '{"done": []}', "[]", b"\xff\xfe"])
def test_an_invalid_file_is_none_and_logged_as_an_error(tmp_path, caplog, content):
    path = tmp_path / "state.json"
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    with caplog.at_level(logging.ERROR, logger="multiplai_core.checkpoint"):
        assert load_model(path, State) is None
    assert any(
        r.levelno == logging.ERROR and "state.json" in r.getMessage()
        for r in caplog.records
    )
