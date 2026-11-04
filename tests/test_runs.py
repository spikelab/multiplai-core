"""Tests for multiplai_core.runs — the shared run records."""

import json
import multiprocessing
from typing import Any

import pytest

from multiplai_core import runs


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    monkeypatch.setattr(runs, "_data_dir", lambda: d)
    return d


def _row(**kw):
    base: dict[str, Any] = {
        "producer": "multiplai-dev:buildme 0.155.2",
        "kind": "epic",
        "target": {"repo": "Dolcebot", "branch": "DF-1-x"},
        "outcome": "parked",
    }
    base.update(kw)
    return runs.make_row(**base)


def test_paths(data_dir):
    assert runs.runs_path() == data_dir / "runs.jsonl"
    assert runs.change_folder("Dolcebot", "DF-1-account management") == (
        data_dir / "changes" / "Dolcebot" / "DF-1-account_management"
    )


def test_folder_name_matches_review_slug_rule():
    assert runs.folder_name("a/b c") == "a_b_c"
    assert runs.folder_name("__x__") == "x"
    assert runs.folder_name("///") == "unnamed"


def test_make_row_fills_defaults_and_version():
    row = _row(cause="merge-conflict")
    assert row["row_version"] == runs.ROW_VERSION
    assert row["requested_by"] is None
    assert row["finished_at"].endswith("Z")
    assert row["run"] == {}
    assert row["cause"] == "merge-conflict"


@pytest.mark.parametrize("field,value", [("kind", "deploy"), ("outcome", "ok")])
def test_make_row_refuses_unknown_kind_or_outcome(field, value):
    with pytest.raises(ValueError):
        _row(**{field: value})


def test_append_then_read(data_dir):
    assert runs.append_row(_row())
    assert runs.append_row(_row(outcome="succeeded"))
    rows, skipped = runs.read_rows()
    assert [r["outcome"] for r in rows] == ["parked", "succeeded"]
    assert skipped == 0


def test_read_skips_cut_lines_and_unknown_versions(data_dir):
    data_dir.mkdir(parents=True)
    good = json.dumps(_row())
    future = json.dumps({**_row(), "row_version": 99})
    (data_dir / "runs.jsonl").write_text(
        f"{good}\n{future}\n[1]\n{good[:20]}", encoding="utf-8"
    )
    rows, skipped = runs.read_rows()
    assert len(rows) == 1
    assert skipped == 3


def test_read_missing_file_is_empty(data_dir):
    assert runs.read_rows() == ([], 0)


def test_append_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    # the parent "directory" is a file: the write fails and is reported, not raised
    assert runs.append_row(_row(), blocker / "runs.jsonl") is False


def test_producer_reads_plugin_json(tmp_path):
    (tmp_path / ".claude-plugin").mkdir()
    (tmp_path / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": "multiplai-dev", "version": "0.155.2"})
    )
    assert runs.producer(tmp_path, "buildme") == "multiplai-dev:buildme 0.155.2"
    assert runs.producer(tmp_path / "missing", "x") == "missing:x unknown"


def _append_many(args):
    path, n, tag = args
    for i in range(n):
        runs.append_row(_row(run={"tag": tag, "i": i, "pad": "x" * 5000}), path)


def test_concurrent_appends_keep_whole_lines(tmp_path):
    path = tmp_path / "runs.jsonl"
    with multiprocessing.get_context("fork").Pool(4) as pool:
        pool.map(_append_many, [(path, 50, t) for t in range(4)])
    rows, skipped = runs.read_rows(path)
    assert skipped == 0
    assert len(rows) == 200
