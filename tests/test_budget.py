"""SpendLedger and the ContextVar helpers, ported from the review and walkthrough
suites (test_sdk.py), plus the token ceiling buildme's cap maps onto."""

import asyncio
import contextvars
import logging
from types import SimpleNamespace

import pytest

from multiplai_core import BudgetExceededError, SpendLedger, budget


@pytest.fixture(autouse=True)
def _fresh_context():
    """Each test runs with no ledger in its context, whatever a previous test started."""
    token = budget._current.set(None)
    yield
    budget._current.reset(token)


def _usage(**kw):
    return SimpleNamespace(**kw)


def test_budget_state_round_trip():
    a = SpendLedger(max_usd=10)
    a.record(_usage(input_tokens=1, output_tokens=2, cost_usd=0.25), label="verify")
    b = SpendLedger(max_usd=10)
    b.load_state(a.to_state())
    assert (b.cost_usd, b.calls, b.by_label) == (0.25, 1, {"verify": 0.25})


def test_from_state_takes_the_ceilings_from_its_arguments_not_the_state():
    a = SpendLedger(max_usd=10, max_tokens=1000)
    a.record(_usage(input_tokens=5, cost_usd=1.0), "find")
    b = SpendLedger.from_state(a.to_state(), max_usd=20)
    assert (b.cost_usd, b.input_tokens, b.max_usd, b.max_tokens) == (1.0, 5, 20, None)
    assert a.to_state()["max_tokens"] == 1000


def test_by_stage_holds_calls_tokens_and_cost_per_label_and_sums_to_the_totals():
    a = SpendLedger(max_usd=10)
    a.record(_usage(input_tokens=10, output_tokens=2, cache_read_tokens=100,
                    cache_creation_tokens=5, cost_usd=0.5), label="find:tests")
    a.record(_usage(input_tokens=1, output_tokens=1, cost_usd=0.25), label="verify")
    a.record(_usage(input_tokens=3, output_tokens=4, cost_usd=0.25), label="verify")
    assert a.by_stage["verify"] == {"calls": 2, "input_tokens": 4, "output_tokens": 5, "cache_read_tokens": 0,
                                    "cache_creation_tokens": 0, "cost_usd": 0.5, "no_usage_calls": 0}
    stages = a.by_stage.values()
    assert sum(s["calls"] for s in stages) == a.calls == 3
    assert sum(s["cost_usd"] for s in stages) == pytest.approx(a.cost_usd)
    for f in budget.TOKEN_FIELDS:
        assert sum(s[f] for s in stages) == getattr(a, f)
    b = SpendLedger(max_usd=10)
    b.load_state(a.to_state())
    assert b.by_stage == a.by_stage and b.by_label == a.by_label


def test_a_call_with_no_usage_is_counted_with_zero_cost():
    a = SpendLedger(max_usd=10)
    a.record(_usage(), label="merge")
    assert (a.calls, a.cost_usd, a.no_usage_calls) == (1, 0.0, 1)
    assert a.by_stage["merge"]["no_usage_calls"] == 1


def test_a_call_whose_usage_is_none_is_counted_as_a_no_usage_call():
    a = SpendLedger(max_usd=10)
    a.record(None, label="find")
    assert (a.calls, a.cost_usd, a.no_usage_calls, a.total_tokens) == (1, 0.0, 1, 0)
    assert a.by_stage["find"]["no_usage_calls"] == 1


def test_an_old_ledger_without_by_stage_loads_and_keeps_counting():
    old = {"calls": 9, "input_tokens": 100, "output_tokens": 20, "cache_read_tokens": 0,
           "cache_creation_tokens": 0, "cost_usd": 1.25, "by_label": {"verify": 1.25}, "max_usd": 50.0}
    a = SpendLedger(max_usd=50)
    a.load_state(old)
    assert (a.calls, a.cost_usd, a.by_stage, a.no_usage_calls) == (9, 1.25, {}, 0)
    a.record(_usage(input_tokens=1, cost_usd=0.5), label="verify")
    assert a.calls == 10 and a.by_stage["verify"]["calls"] == 1
    assert a.by_label["verify"] == pytest.approx(1.75)


def test_record_never_raises_on_garbage_usage():
    a = SpendLedger(max_usd=10)
    a.record(_usage(input_tokens="lots", cost_usd=object()), label="x")  # must not raise
    a.record(_usage(cost_usd=0.1))
    assert a.cost_usd == pytest.approx(0.1)


def test_check_without_a_ceiling_never_raises():
    a = SpendLedger()
    a.record(_usage(input_tokens=10**9, cost_usd=10**6))
    a.check("anything")


def test_the_usd_breaker_carries_cost_tokens_and_the_diagnosis():
    a = SpendLedger(max_usd=1.0)
    a.record(_usage(input_tokens=7, cost_usd=1.2), label="find:tests")
    with pytest.raises(BudgetExceededError) as caught:
        a.check("verify")
    e = caught.value
    assert str(e) == "circuit breaker stopped the run at $1.20 during verify (ceiling $1.00)"
    assert e.cost_usd == pytest.approx(1.2) and e.tokens == 7
    assert "find:tests=$1.20" in e.diagnosis and "ceiling: $1.00" in e.diagnosis


def test_the_token_breaker_trips_on_tokens_alone():
    a = SpendLedger(max_tokens=100)
    a.record(_usage(input_tokens=60, output_tokens=40))
    with pytest.raises(BudgetExceededError, match=r"at 100 tokens \(ceiling 100 tokens\)") as caught:
        a.check()
    assert caught.value.tokens == 100 and "token ceiling: 100" in caught.value.diagnosis


def test_either_ceiling_trips_the_breaker():
    a = SpendLedger(max_usd=100.0, max_tokens=50)
    a.record(_usage(input_tokens=50, cost_usd=0.01))
    with pytest.raises(BudgetExceededError):
        a.check()
    b = SpendLedger(max_usd=1.0, max_tokens=10**9)
    b.record(_usage(input_tokens=1, cost_usd=1.0))
    with pytest.raises(BudgetExceededError, match=r"\$1.00"):
        b.check()


def test_it_warns_once_at_the_warn_fraction(caplog):
    a = SpendLedger(max_usd=10.0, warn_fraction=0.5)
    a.record(_usage(cost_usd=4.0))
    with caplog.at_level(logging.WARNING, logger="multiplai_core.budget"):
        a.check()
        assert not caplog.records
        a.record(_usage(cost_usd=1.0))
        a.check()
        a.check()
    warnings = [r for r in caplog.records if "of its budget" in r.getMessage()]
    assert len(warnings) == 1 and warnings[0].getMessage().startswith("Run has used 50% of its budget")


def test_a_subclass_names_the_run_and_sets_its_own_default_ceiling(caplog):
    class ReviewBudget(SpendLedger):
        run_name = "Review"

        def __init__(self, max_usd=50.0, **kw):
            super().__init__(max_usd=max_usd, **kw)

    ledger = budget.start(ledger_cls=ReviewBudget)
    assert isinstance(ledger, ReviewBudget) and ledger.max_usd == 50.0
    resumed = budget.start(state={"cost_usd": 3.0}, ledger_cls=ReviewBudget)
    assert resumed.max_usd == 50.0 and resumed.cost_usd == 3.0
    assert ReviewBudget().max_usd == 50.0
    r = ReviewBudget(max_usd=1.0)
    r.record(_usage(cost_usd=0.9))
    with caplog.at_level(logging.WARNING, logger="multiplai_core.budget"):
        r.check()
    assert caplog.records[-1].getMessage().startswith("Review has used 90%")


def test_start_with_an_explicit_none_ceiling_leaves_the_run_uncapped():
    class ReviewBudget(SpendLedger):
        def __init__(self, max_usd=50.0, max_tokens=1_000, **kw):
            super().__init__(max_usd=max_usd, max_tokens=max_tokens, **kw)

    ledger = budget.start(max_usd=None, max_tokens=None, ledger_cls=ReviewBudget)
    assert ledger.max_usd is None and ledger.max_tokens is None
    ledger.record(_usage(cost_usd=500.0))
    ledger.check()  # uncapped: never raises
    only_usd = budget.start(None, ledger_cls=ReviewBudget)
    assert only_usd.max_usd is None and only_usd.max_tokens == 1_000


def test_start_restores_state_and_record_and_check_go_to_the_current_ledger():
    previous = SpendLedger()
    previous.record(_usage(cost_usd=0.75), "find")
    ledger = budget.start(1.0, previous.to_state())
    assert budget.get() is ledger and ledger.cost_usd == pytest.approx(0.75)
    budget.record(_usage(cost_usd=0.5), label="verify")
    assert ledger.by_label == {"find": 0.75, "verify": 0.5}
    with pytest.raises(BudgetExceededError):
        budget.check("merge")


def test_get_creates_a_ledger_on_first_use():
    first = budget.get()
    assert isinstance(first, SpendLedger) and first.max_usd is None and budget.get() is first


def test_each_task_records_against_its_own_ledger():
    """A batch runs one target per asyncio task; spend must not cross between them."""

    async def target(cost):
        ledger = budget.start(10.0)
        await asyncio.sleep(0)
        budget.record(_usage(cost_usd=cost), "find")
        await asyncio.sleep(0)
        return ledger.cost_usd

    async def main():
        return await asyncio.gather(target(1.0), target(2.0), target(3.0))

    assert contextvars.copy_context().run(asyncio.run, main()) == [1.0, 2.0, 3.0]
