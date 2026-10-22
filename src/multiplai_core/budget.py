"""A per-run token and cost ledger with a circuit breaker.

Started from the review pipeline's ``budget.py`` (multiplai-cc-mktplace), which
the codebase-walkthrough pipeline copied, and given buildme's token ceiling
(``max_tokens``) so buildme's cap can map onto it. The rules all three share:

- ``record()`` never raises. Accounting must not be able to break a run.
- Only ``check()`` stops a run, by raising :class:`BudgetExceededError`. Call
  it at loop boundaries, so one expensive call is never cut off half-way.
- ``check()`` warns once when a ceiling is ``warn_fraction`` spent.
- ``to_state()`` / ``load_state()`` / ``from_state()`` carry the spend through
  a checkpoint, so a resumed run does not get a fresh budget. The ceilings are
  not restored: a resume sets its own.

The ContextVar helpers (``start``, ``get``, ``record``, ``check``) are public,
called as ``multiplai_core.budget.start(...)``. They keep one ledger per
context, so concurrent asyncio tasks (one per review target, say) each record
against their own ledger.

A pipeline that needs its own default ceiling, warning text or state fields
subclasses :class:`SpendLedger` and passes the subclass as ``ledger_cls``.
``start()`` forwards only the ceilings it is given, so leaving one out keeps
the subclass default; passing ``None`` turns that ceiling off.
"""

from __future__ import annotations

import contextvars
import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Self, TypeVar

log = logging.getLogger(__name__)

WARN_FRACTION = 0.8

TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
)


def empty_stage() -> dict:
    """One stage label's record: calls, the four token counts, cost, and calls with no usage."""
    return {
        "calls": 0,
        **dict.fromkeys(TOKEN_FIELDS, 0),
        "cost_usd": 0.0,
        "no_usage_calls": 0,
    }


class BudgetExceededError(Exception):
    """A ceiling is spent. Carries where the money went.

    ``cost_usd`` and ``tokens`` are the totals when the breaker tripped;
    ``diagnosis`` is :meth:`SpendLedger.diagnosis` at that moment.
    """

    def __init__(
        self,
        message: str,
        *,
        diagnosis: str = "",
        cost_usd: float = 0.0,
        tokens: int = 0,
    ) -> None:
        super().__init__(message)
        self.diagnosis = diagnosis
        self.cost_usd = cost_usd
        self.tokens = tokens


L = TypeVar("L", bound="SpendLedger")


@dataclass
class SpendLedger:
    max_usd: float | None = None
    max_tokens: int | None = None
    warn_fraction: float = WARN_FRACTION
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    by_label: dict[str, float] = field(default_factory=dict)  # {stage label: usd}
    # {stage label: empty_stage() record}. A ledger saved without it loads with
    # it empty, so after that resume its sums cover only the calls made since.
    by_stage: dict[str, dict] = field(default_factory=dict)
    # Calls whose run returned no usage at all: counted, with 0 tokens and $0.
    no_usage_calls: int = 0
    _warned: bool = field(default=False, init=False, repr=False, compare=False)

    # Names the run in the one-off warning: "<run_name> has used 80% of its budget".
    run_name = "Run"

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_creation_tokens
        )

    @property
    def fraction_used(self) -> float:
        """The highest fraction spent across the set ceilings (0.0 when none is set)."""
        fractions = []
        if self.max_usd:
            fractions.append(self.cost_usd / self.max_usd)
        if self.max_tokens:
            fractions.append(self.total_tokens / self.max_tokens)
        return max(fractions) if fractions else 0.0

    def record(self, usage: Any, label: str = "") -> None:
        """Add one call's usage (``AgentUsage`` or anything shaped like it). Never raises.

        ``None``, or a usage with no tokens and no cost, counts as a call with
        no usage.
        """
        try:
            tokens = {f: int(getattr(usage, f, 0) or 0) for f in TOKEN_FIELDS}
            cost = float(getattr(usage, "cost_usd", 0.0) or 0.0)
            no_usage = usage is None or (not any(tokens.values()) and not cost)
            self.calls += 1
            for f, n in tokens.items():
                setattr(self, f, getattr(self, f) + n)
            self.cost_usd += cost
            self.no_usage_calls += int(no_usage)
            if label:
                self.by_label[label] = self.by_label.get(label, 0.0) + cost
                stage = self.by_stage.setdefault(label, empty_stage())
                stage["calls"] += 1
                for f, n in tokens.items():
                    stage[f] += n
                stage["cost_usd"] += cost
                stage["no_usage_calls"] += int(no_usage)
        except Exception as e:  # pragma: no cover — accounting must never break a run
            log.warning("Budget accounting failed for a call (ignored): %s", e)

    def diagnosis(self) -> str:
        """Where the spend went: totals, the ceilings, and cost per stage label, highest first."""
        lines = [
            f"Spent ${self.cost_usd:.2f} over {self.calls} agent calls ({self.total_tokens:,} tokens)."
        ]
        if self.max_usd:
            lines.append(f"  ceiling: ${self.max_usd:.2f}")
        if self.max_tokens:
            lines.append(f"  token ceiling: {self.max_tokens:,}")
        if self.by_label:
            top = sorted(self.by_label.items(), key=lambda kv: -kv[1])
            lines.append("  by stage: " + ", ".join(f"{k}=${v:.2f}" for k, v in top))
        return "\n".join(lines)

    def check(self, label: str = "") -> None:
        """Raise ``BudgetExceededError`` when a ceiling is spent; warn once at ``warn_fraction``.

        *label* names where the run is ("during <label>") in the message.
        """
        if not (self.max_usd or self.max_tokens):
            return
        where = f" during {label}" if label else ""
        over_usd = bool(self.max_usd) and self.cost_usd >= self.max_usd  # type: ignore[operator]
        over_tokens = bool(self.max_tokens) and self.total_tokens >= self.max_tokens  # type: ignore[operator]
        if over_usd or over_tokens:
            if over_usd:
                message = (
                    f"circuit breaker stopped the run at ${self.cost_usd:.2f}{where} "
                    f"(ceiling ${self.max_usd:.2f})"
                )
            else:
                message = (
                    f"circuit breaker stopped the run at {self.total_tokens:,} tokens{where} "
                    f"(ceiling {self.max_tokens:,} tokens)"
                )
            raise BudgetExceededError(
                message,
                diagnosis=self.diagnosis(),
                cost_usd=self.cost_usd,
                tokens=self.total_tokens,
            )
        used = self.fraction_used
        if used >= self.warn_fraction and not self._warned:
            self._warned = True
            log.warning(
                "%s has used %.0f%% of its budget — %s",
                self.run_name,
                used * 100,
                self.diagnosis(),
            )

    def to_state(self) -> dict:
        """The spend as a JSON-ready dict, for a checkpoint."""
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "by_label": {k: round(v, 6) for k, v in self.by_label.items()},
            "by_stage": {
                k: {**v, "cost_usd": round(v["cost_usd"], 6)}
                for k, v in self.by_stage.items()
            },
            "no_usage_calls": self.no_usage_calls,
            "max_usd": self.max_usd,
            "max_tokens": self.max_tokens,
        }

    def load_state(self, data: dict | None) -> None:
        """Restore spend from a checkpoint (the ceilings stay as they are)."""
        if not data:
            return
        self.calls = int(data.get("calls", 0) or 0)
        for f in TOKEN_FIELDS:
            setattr(self, f, int(data.get(f, 0) or 0))
        self.cost_usd = float(data.get("cost_usd", 0.0) or 0.0)
        self.by_label = {k: float(v) for k, v in (data.get("by_label") or {}).items()}
        self.by_stage = {}
        for label, rec in (data.get("by_stage") or {}).items():
            stage = empty_stage()
            for key in stage:
                stage[key] = (
                    float(rec.get(key, 0) or 0)
                    if key == "cost_usd"
                    else int(rec.get(key, 0) or 0)
                )
            self.by_stage[label] = stage
        self.no_usage_calls = int(data.get("no_usage_calls", 0) or 0)

    @classmethod
    def from_state(cls, data: dict | None, **kwargs: Any) -> Self:
        """A new ledger built with *kwargs* (the ceilings), holding the spend in *data*."""
        ledger = cls(**kwargs)
        ledger.load_state(data)
        return ledger


_current: contextvars.ContextVar[SpendLedger | None] = contextvars.ContextVar(
    "multiplai_spend_ledger", default=None
)


def get(*, ledger_cls: type[SpendLedger] = SpendLedger) -> SpendLedger:
    """The ledger of the current context; a new ``ledger_cls()`` on first use."""
    ledger = _current.get()
    if ledger is None:
        ledger = ledger_cls()
        _current.set(ledger)
    return ledger


class _Unset(enum.Enum):
    UNSET = enum.auto()


_UNSET = _Unset.UNSET


def start(
    max_usd: float | None | _Unset = _UNSET,
    state: dict | None = None,
    *,
    max_tokens: int | None | _Unset = _UNSET,
    ledger_cls: type[SpendLedger] = SpendLedger,
) -> SpendLedger:
    """Begin a ledger in the current context with these ceilings, restoring *state* if given.

    A ceiling left out takes ``ledger_cls``'s own default, so a subclass that
    sets a default ceiling keeps it. A ceiling passed as ``None`` turns that
    ceiling off: the run is uncapped on it.
    """
    ceilings: dict[str, Any] = {}
    if max_usd is not _UNSET:
        ceilings["max_usd"] = max_usd
    if max_tokens is not _UNSET:
        ceilings["max_tokens"] = max_tokens
    ledger = ledger_cls.from_state(state, **ceilings)
    _current.set(ledger)
    return ledger


def record(usage: Any, label: str = "") -> None:
    """``get().record(usage, label)``. Never raises."""
    get().record(usage, label=label)


def check(label: str = "") -> None:
    """``get().check(label)``."""
    get().check(label)
