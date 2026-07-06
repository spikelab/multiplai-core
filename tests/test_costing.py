"""Unit tests for multiplai_core.costing — pricing math and ledger I/O."""

import pytest

from multiplai_core import costing
from multiplai_core.costing import TokenCounts


@pytest.fixture(autouse=True)
def _paths_to_tmp(monkeypatch, tmp_path, reset_paths_cache):
    """Anchor the ledger under a temp workspace for every test."""
    monkeypatch.setenv("WORKSPACE", str(tmp_path))


# ----------------------------------------------------------------------
# Pricing math
# ----------------------------------------------------------------------

def test_price_opus_golden():
    # 1M of everything at Opus 4.8 rates (in 5.0 / out 25.0):
    # in 5 + out 25 + cw5m 6.25 + cw1h 10 + cr 0.5 = 46.75
    tokens = TokenCounts(
        input=1_000_000, output=1_000_000,
        cw5m=1_000_000, cw1h=1_000_000, cr=1_000_000,
    )
    cost, fallback = costing.price_tokens("claude-opus-4-8", tokens)
    assert cost == pytest.approx(46.75)
    assert fallback is False


def test_price_realistic_call():
    # Real transcript sample: opus-4-8, in=5628 out=255 cw1h=11782 cr=18348
    tokens = TokenCounts(input=5628, output=255, cw1h=11782, cr=18348)
    cost, _ = costing.price_tokens("claude-opus-4-8", tokens)
    expected = (5628 * 5 + 255 * 25 + 11782 * 2 * 5 + 18348 * 0.1 * 5) / 1e6
    assert cost == pytest.approx(expected)


def test_price_haiku_vs_fable_ratio():
    tokens = TokenCounts(input=1_000_000)
    haiku, _ = costing.price_tokens("claude-haiku-4-5", tokens)
    fable, _ = costing.price_tokens("claude-fable-5", tokens)
    assert haiku == pytest.approx(1.0)
    assert fable == pytest.approx(10.0)


def test_dated_snapshot_resolves():
    rates, fallback = costing.resolve_model_rates("claude-haiku-4-5-20251001")
    assert rates["in"] == pytest.approx(1.0)
    assert fallback is False


def test_prefix_match_resolves():
    rates, fallback = costing.resolve_model_rates("claude-3-5-haiku-20241022")
    assert rates["in"] == pytest.approx(0.8)
    assert fallback is False


def test_unknown_model_falls_back_and_flags():
    tokens = TokenCounts(input=1_000_000)
    cost, fallback = costing.price_tokens("gpt-nonsense", tokens)
    assert fallback is True
    assert cost == pytest.approx(5.0)  # fallback in-rate

    rec = costing.build_record(
        ts="2026-07-06T00:00:00Z", source="transcript", session="s1",
        model="gpt-nonsense", msg_id="m1", tokens=tokens,
    )
    assert rec["pricing_fallback"] is True


def test_build_record_prefers_sdk_cost():
    rec = costing.build_record(
        ts="2026-07-06T00:00:00Z", source="sdk", session="s1",
        model="claude-opus-4-8", msg_id="run1",
        tokens=TokenCounts(input=100), cost_usd=1.23456789,
        component="buildme",
    )
    assert rec["cost_usd"] == pytest.approx(1.234568)
    assert "pricing_fallback" not in rec
    assert rec["component"] == "buildme"


# ----------------------------------------------------------------------
# Ledger I/O
# ----------------------------------------------------------------------

def _rec(ts: str, session: str = "s1", msg_id: str = "m1") -> dict:
    return costing.build_record(
        ts=ts, source="transcript", session=session,
        model="claude-opus-4-8", msg_id=msg_id,
        tokens=TokenCounts(input=10, output=5),
    )


def test_append_splits_by_month():
    n = costing.append_records([
        _rec("2026-06-30T23:59:00Z", msg_id="a"),
        _rec("2026-07-01T00:01:00Z", msg_id="b"),
    ])
    assert n == 2
    assert costing.ledger_file("2026-06").exists()
    assert costing.ledger_file("2026-07").exists()


def test_iter_ledger_roundtrip_and_month_filter():
    costing.append_records([
        _rec("2026-06-01T00:00:00Z", msg_id="a"),
        _rec("2026-07-01T00:00:00Z", msg_id="b"),
    ])
    all_recs = list(costing.iter_ledger())
    assert [r["msg_id"] for r in all_recs] == ["a", "b"]
    july = list(costing.iter_ledger(months=["2026-07"]))
    assert [r["msg_id"] for r in july] == ["b"]


def test_iter_ledger_skips_malformed_lines():
    costing.append_records([_rec("2026-07-01T00:00:00Z")])
    path = costing.ledger_file("2026-07")
    with path.open("a") as fh:
        fh.write("{torn write\n")
    costing.append_records([_rec("2026-07-02T00:00:00Z", msg_id="m2")])
    recs = list(costing.iter_ledger())
    assert [r["msg_id"] for r in recs] == ["m1", "m2"]


def test_session_msg_index():
    costing.append_records([
        _rec("2026-07-01T00:00:00Z", session="s1", msg_id="a"),
        _rec("2026-07-01T00:01:00Z", session="s1", msg_id="b"),
        _rec("2026-07-01T00:02:00Z", session="s2", msg_id="a"),
    ])
    index = costing.session_msg_index()
    assert index == {"s1": {"a", "b"}, "s2": {"a"}}


def test_iter_ledger_empty_dir():
    assert list(costing.iter_ledger()) == []


def test_pricing_json_ships_with_package():
    pricing = costing.load_pricing()
    assert "claude-opus-4-8" in pricing["models"]
    assert pricing["multipliers"] == {"cw5m": 1.25, "cw1h": 2.0, "cr": 0.1}
