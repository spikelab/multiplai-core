"""Cost accounting — pricing table, cost math, and the append-only ledger.

One JSONL record per API call is the only persisted truth; every report
(per-session, per-skill, per-day) is derived at read time. Records come from
two sources: the transcript collector in the multiplai-context plugin
(``source: "transcript"``) and ``agent_runner`` SDK runs (``source: "sdk"``).

Costs are API-equivalent USD — what the call *would* have billed at list
prices — computed as::

    (in·P_in + out·P_out + cw5m·1.25·P_in + cw1h·2·P_in + cr·0.1·P_in) / 1e6

Pricing lives in the package-data file ``pricing.json``. Unknown models are
priced at the fallback rate and flagged ``pricing_fallback: true`` — never
silently dropped.

Ledger files are monthly (``<data_dir>/costs/ledger-YYYY-MM.jsonl``) and
written with O_APPEND single-write lines, so concurrent writers (collector +
SDK tap) interleave safely without locks.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Iterator

from .paths import get_paths

logger = logging.getLogger(__name__)

_pricing_cache: dict | None = None

# Dated snapshot suffix, e.g. claude-haiku-4-5-20251001 -> claude-haiku-4-5.
_DATE_SUFFIX = re.compile(r"-\d{8}$")


def load_pricing() -> dict:
    """Return the parsed pricing table (cached for the process lifetime)."""
    global _pricing_cache
    if _pricing_cache is None:
        raw = resources.files("multiplai_core").joinpath("pricing.json").read_text()
        _pricing_cache = dict(json.loads(raw))
    return _pricing_cache


@dataclass(frozen=True)
class TokenCounts:
    """Token counts for one API call, split by cache tier.

    ``cw5m``/``cw1h`` are cache writes (5-minute / 1-hour TTL); ``cr`` is
    cache reads. When a source reports only an undifferentiated
    ``cache_creation_input_tokens`` total, put it in ``cw5m`` — the cheaper
    tier — so the estimate errs low rather than high.
    """

    input: int = 0
    output: int = 0
    cw5m: int = 0
    cw1h: int = 0
    cr: int = 0


def resolve_model_rates(model: str) -> tuple[dict, bool]:
    """Return ``(rates, used_fallback)`` for *model*.

    Match order: exact id, date-suffix-stripped id, then longest known key
    that prefixes the id (catches regional/variant suffixes). Anything else
    gets the fallback rates.
    """
    pricing = load_pricing()
    models: dict[str, dict] = pricing["models"]
    if model in models:
        return models[model], False
    stripped = _DATE_SUFFIX.sub("", model)
    if stripped in models:
        return models[stripped], False
    prefix_matches = [k for k in models if model.startswith(k)]
    if prefix_matches:
        return models[max(prefix_matches, key=len)], False
    return pricing["fallback"], True


def price_tokens(model: str, tokens: TokenCounts) -> tuple[float, bool]:
    """Return ``(cost_usd, used_fallback)`` for one call's token counts."""
    rates, fallback = resolve_model_rates(model)
    mult = load_pricing()["multipliers"]
    cost = (
        tokens.input * rates["in"]
        + tokens.output * rates["out"]
        + tokens.cw5m * mult["cw5m"] * rates["in"]
        + tokens.cw1h * mult["cw1h"] * rates["in"]
        + tokens.cr * mult["cr"] * rates["in"]
    ) / 1_000_000
    return cost, fallback


def build_record(
    *,
    ts: str,
    source: str,
    session: str,
    model: str,
    msg_id: str,
    tokens: TokenCounts,
    project: str = "",
    sidechain: bool = False,
    span: dict | None = None,
    component: str = "",
    cost_usd: float | None = None,
) -> dict[str, Any]:
    """Build one ledger record; prices from *tokens* unless *cost_usd* given.

    *cost_usd* is for SDK runs where the SDK already reports an authoritative
    ``total_cost_usd`` — pass it and the token-derived price is skipped.
    """
    record: dict[str, Any] = {
        "ts": ts,
        "source": source,
        "session": session,
        "project": project,
        "model": model,
        "msg_id": msg_id,
        "sidechain": sidechain,
        "span": span,
        "component": component,
        "tokens": {
            "in": tokens.input,
            "out": tokens.output,
            "cw5m": tokens.cw5m,
            "cw1h": tokens.cw1h,
            "cr": tokens.cr,
        },
    }
    if cost_usd is not None:
        record["cost_usd"] = round(cost_usd, 6)
    else:
        cost, fallback = price_tokens(model, tokens)
        record["cost_usd"] = round(cost, 6)
        if fallback:
            record["pricing_fallback"] = True
    return record


# ----------------------------------------------------------------------
# Ledger I/O
# ----------------------------------------------------------------------

_LEDGER_NAME = re.compile(r"^ledger-(\d{4}-\d{2})\.jsonl$")


def costs_dir() -> Path:
    """Ledger directory ``<data_dir>/costs``."""
    return get_paths().data_dir / "costs"


def ledger_file(month: str) -> Path:
    """Monthly ledger path for *month* (``YYYY-MM``)."""
    return costs_dir() / f"ledger-{month}.jsonl"


def _month_of(record: dict) -> str:
    return str(record.get("ts", ""))[:7] or "unknown"


def append_records(records: Iterable[dict]) -> int:
    """Append *records* to their monthly ledger files; return count written.

    Lines are written individually to files opened in append mode, so
    concurrent writers interleave whole lines (POSIX O_APPEND semantics).
    Failures are logged and re-raised — callers on non-critical paths (the
    SDK tap) should catch and continue.
    """
    by_month: dict[str, list[dict]] = {}
    for rec in records:
        by_month.setdefault(_month_of(rec), []).append(rec)
    written = 0
    costs_dir().mkdir(parents=True, exist_ok=True)
    for month, recs in sorted(by_month.items()):
        path = ledger_file(month)
        with path.open("a", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
                written += 1
    return written


def iter_ledger(months: Iterable[str] | None = None) -> Iterator[dict]:
    """Yield ledger records, oldest month first.

    *months* limits to specific ``YYYY-MM`` strings; ``None`` reads all.
    Malformed lines are skipped with a warning (a torn write must not brick
    every report).
    """
    directory = costs_dir()
    if not directory.is_dir():
        return
    wanted = set(months) if months is not None else None
    files = sorted(
        p for p in directory.iterdir()
        if (m := _LEDGER_NAME.match(p.name)) and (wanted is None or m.group(1) in wanted)
    )
    for path in files:
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("skipping malformed ledger line %s:%d", path, lineno)


def session_msg_index(months: Iterable[str] | None = None) -> dict[str, set[str]]:
    """Return ``{session: {msg_id, ...}}`` from the ledger, for dedup.

    The collector loads this before a pass so records already priced are
    never appended twice.
    """
    index: dict[str, set[str]] = {}
    for rec in iter_ledger(months):
        index.setdefault(rec.get("session", ""), set()).add(rec.get("msg_id", ""))
    return index
