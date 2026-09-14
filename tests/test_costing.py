"""Unit tests for multiplai_core.costing — pricing math and ledger I/O."""

import pytest

from multiplai_core import costing
from multiplai_core.costing import TokenCounts


@pytest.fixture(autouse=True)
def _paths_to_tmp(tmp_workspace):
    """Anchor the ledger under a temp workspace for every test."""


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


# ----------------------------------------------------------------------
# Live pricing refresh
# ----------------------------------------------------------------------

_PRICING_MD = """\
# Pricing

## Model pricing

| Model | Base input tokens | 5m cache writes | 1h cache writes | Cache hits and refreshes | Output tokens |
| --- | --- | --- | --- | --- | --- |
| Claude Fable 5.1 | $10 / MTok | $12.50 / MTok | $20 / MTok | $0.25 / MTok1 | $50 / MTok |
| Claude Mythos 5.1 ([limited availability](https://anthropic.com/glasswing)) | $10 / MTok | $12.50 / MTok | $20 / MTok | $0.25 / MTok1 | $50 / MTok |
| Claude Opus 5 | $5 / MTok | $6.25 / MTok | $10 / MTok | $0.50 / MTok | $25 / MTok |
| Claude Opus 4 ([retired, except on Google Cloud](https://x/y)) | $15 / MTok | $18.75 / MTok | $30 / MTok | $1.50 / MTok | $75 / MTok |
| Claude Sonnet 5 | $2 / MTok | $2.50 / MTok | $4 / MTok | $0.20 / MTok | $10 / MTok |
| Claude Haiku 3.5 ([retired](https://x/y)) | $0.80 / MTok | $1 / MTok | $1.60 / MTok | $0.08 / MTok | $4 / MTok |

## Cloud platform pricing

### Fast mode pricing

| Model | Input | Output |
| --- | --- | --- |
| Claude Opus 5 / Claude Opus 4.8 | $10 / MTok | $50 / MTok |
"""


@pytest.fixture(autouse=True)
def _fresh_pricing():
    costing.reset_pricing_cache()
    yield
    costing.reset_pricing_cache()


@pytest.mark.parametrize(
    ("name", "model_id"),
    [
        ("Claude Fable 5.1", "claude-fable-5-1"),
        ("Claude Opus 5", "claude-opus-5"), ("Claude Opus 4", "claude-opus-4-0"), ("Claude Fable 5", "claude-fable-5"),
        ("Claude Opus 4.8", "claude-opus-4-8"),
        ("Claude Haiku 3.5", "claude-3-5-haiku"),
    ],
)
def test_model_id_from_display_name(name, model_id):
    assert costing.model_id_from_display_name(name) == model_id


def test_model_id_rejects_non_claude():
    with pytest.raises(ValueError):
        costing.model_id_from_display_name("GPT 5")


def test_parse_pricing_markdown_reads_only_model_table():
    models = costing.parse_pricing_markdown(_PRICING_MD)
    assert set(models) == {
        "claude-fable-5-1", "claude-mythos-5-1", "claude-opus-5",
        "claude-opus-4-0", "claude-sonnet-5", "claude-3-5-haiku",
    }
    assert models["claude-fable-5-1"] == {"in": 10.0, "out": 50.0, "cw5m": 12.5, "cw1h": 20.0, "cr": 0.25}
    assert models["claude-sonnet-5"] == {"in": 2.0, "out": 10.0, "cw5m": 2.5, "cw1h": 4.0, "cr": 0.2}
    # the fast-mode table has 2 price cells and must not leak in
    assert "claude-opus-5-/-claude-opus-4-8" not in models


def test_parse_pricing_markdown_reads_columns_by_heading():
    reordered = """## Model pricing

| Model | Output tokens | Base input tokens | Cache hits and refreshes | 1h cache writes | 5m cache writes |
| --- | --- | --- | --- | --- | --- |
| Claude Sonnet 5 | $10 / MTok | $2 / MTok | $0.20 / MTok | $4 / MTok | $2.50 / MTok |
"""
    models = costing.parse_pricing_markdown(reordered)
    assert models["claude-sonnet-5"] == {"in": 2.0, "out": 10.0, "cw5m": 2.5, "cw1h": 4.0, "cr": 0.2}


def test_parse_pricing_markdown_skips_inconsistent_rows(caplog):
    page = """## Model pricing

| Model | Base input tokens | 5m cache writes | 1h cache writes | Cache hits and refreshes | Output tokens |
| --- | --- | --- | --- | --- | --- |
| Claude Sonnet 5 | $10 / MTok | $2.50 / MTok | $4 / MTok | $0.20 / MTok | $2 / MTok |
| Claude Opus 5 | $5 / MTok | $6.25 / MTok | $10 / MTok | $0.50 / MTok | $25 / MTok |
"""
    with caplog.at_level("WARNING"):
        models = costing.parse_pricing_markdown(page)
    assert set(models) == {"claude-opus-5"}
    assert "sanity check" in caplog.text


def test_parse_pricing_markdown_ignores_tables_without_price_header():
    page = """## Model pricing

| Model | Fast mode input | Fast mode output |
| --- | --- | --- |
| Claude Opus 5 | $30 / MTok | $150 / MTok |
"""
    assert costing.parse_pricing_markdown(page) == {}


def test_fetch_live_pricing_rejects_a_partial_table(monkeypatch):
    partial = """## Model pricing

| Model | Base input tokens | 5m cache writes | 1h cache writes | Cache hits and refreshes | Output tokens |
| --- | --- | --- | --- | --- | --- |
| Claude Opus 5 | $5 / MTok | $6.25 / MTok | $10 / MTok | $0.50 / MTok | $25 / MTok |
"""

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return partial.encode()

    monkeypatch.setattr(costing.urllib.request, "urlopen", lambda *a, **k: _Resp())
    with pytest.raises(ValueError, match="below the floor"):
        costing.fetch_live_pricing()


def test_parse_pricing_markdown_without_section_is_empty():
    assert costing.parse_pricing_markdown("# nothing here\n| Claude Opus 5 | $5 / MTok |") == {}


def test_bundled_table_has_current_models_and_fable_cache_read():
    pricing = costing.load_pricing()
    assert pricing["models"]["claude-sonnet-5"] == {"in": 2.0, "out": 10.0}
    assert pricing["models"]["claude-opus-5"] == {"in": 5.0, "out": 25.0}
    assert pricing["models"]["claude-fable-5-1"]["cr"] == 0.25


def test_explicit_tier_price_beats_multiplier():
    # 1M cache-read tokens on Fable 5.1: $0.25, not 0.1 × $10 = $1.00
    cost, fallback = costing.price_tokens("claude-fable-5-1", TokenCounts(cr=1_000_000))
    assert fallback is False
    assert cost == pytest.approx(0.25)
    # models without explicit tiers still use the multipliers
    cost, _ = costing.price_tokens("claude-opus-5", TokenCounts(cr=1_000_000))
    assert cost == pytest.approx(0.5)


def test_opus_5_is_priced_not_fallback():
    _, fallback = costing.resolve_model_rates("claude-opus-5")
    assert fallback is False


def test_refresh_pricing_writes_cache_and_load_prefers_it(monkeypatch):
    monkeypatch.setattr(costing, "fetch_live_pricing", lambda url, timeout: {
        "claude-opus-5": {"in": 7.0, "out": 35.0, "cw5m": 8.75, "cw1h": 14.0, "cr": 0.7},
    })
    path = costing.refresh_pricing(force=True)
    assert path == costing.pricing_cache_path() and path.is_file()
    pricing = costing.load_pricing()
    assert pricing["models"]["claude-opus-5"]["in"] == 7.0       # refreshed wins
    assert "claude-opus-4-1" in pricing["models"]                # bundled retired model kept
    assert pricing["updated"] == costing._today().isoformat()
    assert costing.pricing_age_days() == 0


def test_refresh_pricing_skips_fresh_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(costing, "fetch_live_pricing", lambda url, timeout: calls.append(1) or {"claude-opus-5": {"in": 5.0, "out": 25.0}})
    assert costing.refresh_pricing(force=True) is not None
    assert costing.refresh_pricing() is not None
    assert len(calls) == 1


def test_refresh_pricing_failure_is_non_fatal(monkeypatch, caplog):
    def boom(url, timeout):
        raise OSError("no network")
    monkeypatch.setattr(costing, "fetch_live_pricing", boom)
    with caplog.at_level("WARNING"):
        assert costing.refresh_pricing(force=True) is None
    assert "Pricing refresh" in caplog.text
    assert not costing.pricing_cache_path().exists()
    # still prices from the bundled table
    cost, fallback = costing.price_tokens("claude-opus-5", TokenCounts(input=1_000_000))
    assert (cost, fallback) == (pytest.approx(5.0), False)


def test_unreadable_pricing_cache_is_ignored(caplog):
    path = costing.pricing_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    with caplog.at_level("WARNING"):
        pricing = costing.load_pricing()
    assert "claude-opus-5" in pricing["models"]
    assert "unreadable pricing cache" in caplog.text


def test_unknown_model_warns_once(caplog):
    with caplog.at_level("WARNING"):
        costing.resolve_model_rates("gpt-nonsense")
        costing.resolve_model_rates("gpt-nonsense")
    assert caplog.text.count("No list price for model") == 1
