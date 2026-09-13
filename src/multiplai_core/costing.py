"""Cost accounting — pricing table, cost math, and the append-only ledger.

One JSONL record per API call is the only persisted truth; every report
(per-session, per-skill, per-day) is derived at read time. Records come from
two sources: the transcript collector in the multiplai-context plugin
(``source: "transcript"``) and ``agent_runner`` SDK runs (``source: "sdk"``).

Costs are API-equivalent USD — what the call *would* have billed at list
prices — computed as::

    (in·P_in + out·P_out + cw5m·P_cw5m + cw1h·P_cw1h + cr·P_cr) / 1e6

where the cache-tier prices default to multipliers of ``P_in`` (1.25, 2, 0.1)
unless the model's entry states them explicitly (Claude Fable 5.1 reads cache
at 0.025× input, so it needs the explicit form).

Pricing comes from two places. The package ships ``pricing.json`` as an
offline snapshot. ``refresh_pricing()`` fetches the official pricing page
(``PRICING_URL``, served as markdown), parses the model table and writes
``<data_dir>/costs/pricing.json``; ``load_pricing()`` prefers that file when
it exists, merged over the bundled snapshot so retired models stay priced.
The collector calls ``refresh_pricing()`` at the start of every pass, so a
model launched after the last package release is priced correctly as soon as
the page lists it. Unknown models are priced at the fallback rate and flagged
``pricing_fallback: true`` — never silently dropped — and warned about once
per process.

Ledger files are monthly (``<data_dir>/costs/ledger-YYYY-MM.jsonl``) and
written with O_APPEND single-write lines, so concurrent writers (collector +
SDK tap) interleave safely without locks.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Iterator

from .paths import get_paths

logger = logging.getLogger(__name__)

_pricing_cache: dict | None = None
_fallback_warned: set[str] = set()

# Dated snapshot suffix, e.g. claude-haiku-4-5-20251001 -> claude-haiku-4-5.
_DATE_SUFFIX = re.compile(r"-\d{8}$")

#: The official pricing page. The docs site serves every page as markdown at
#: the ``.md`` path, which is what makes the table parseable without HTML.
PRICING_URL = "https://platform.claude.com/docs/en/about-claude/pricing.md"

# Display names on the pricing page whose API id does not follow the
# ``claude-<family>-<major>-<minor>`` pattern.
_DISPLAY_NAME_IDS = {"claude-haiku-3-5": "claude-3-5-haiku"}

# One table row: "| Claude Opus 5 | $5 / MTok | $6.25 / MTok | ... |". The
# name cell may carry a parenthetical link (limited availability, retired).
_PRICE_ROW = re.compile(r"^\|\s*(Claude [^|(]+?)\s*(?:\([^|]*\))?\s*\|(.*)\|\s*$")
_PRICE_CELL = re.compile(r"\$([0-9]+(?:\.[0-9]+)?)\s*/\s*MTok")


def _bundled_pricing() -> dict:
    raw = resources.files("multiplai_core").joinpath("pricing.json").read_text()
    return dict(json.loads(raw))


def pricing_cache_path() -> Path:
    """Where ``refresh_pricing()`` writes: ``<data_dir>/costs/pricing.json``."""
    return costs_dir() / "pricing.json"


def _merge_pricing(base: dict, over: dict) -> dict:
    """*over* wins per model; models only in *base* (retired) survive."""
    merged = dict(base)
    merged["models"] = {**base.get("models", {}), **over.get("models", {})}
    for key in ("updated", "source", "multipliers", "fallback"):
        if key in over:
            merged[key] = over[key]
    return merged


def load_pricing() -> dict:
    """Return the pricing table (cached for the process lifetime).

    The refreshed file under ``costs_dir()`` wins over the bundled snapshot;
    a missing or unparseable refreshed file falls back to the snapshot
    silently, so an offline machine still prices every call.
    """
    global _pricing_cache
    if _pricing_cache is None:
        pricing = _bundled_pricing()
        try:
            path = pricing_cache_path()
            if path.is_file():
                pricing = _merge_pricing(pricing, json.loads(path.read_text()))
        except Exception as exc:  # noqa: BLE001 — a bad cache must not stop pricing
            logger.warning("Ignoring unreadable pricing cache: %s", exc)
        _pricing_cache = pricing
    return _pricing_cache


def reset_pricing_cache() -> None:
    """Forget the in-process table so the next call re-reads disk."""
    global _pricing_cache
    _pricing_cache = None
    _fallback_warned.clear()


def model_id_from_display_name(name: str) -> str:
    """``"Claude Opus 4.8"`` → ``claude-opus-4-8``; ``"Claude Opus 5"`` → ``claude-opus-5``.

    The API ids for 4.x-and-later models are the display name lower-cased
    with dots as dashes. The 4 generation wrote a bare major as ``-4-0``
    (``claude-opus-4-0``, ``claude-sonnet-4-0``); from 5 on the id is bare
    (``claude-opus-5``, ``claude-fable-5``). Names that do not follow the
    pattern are listed in ``_DISPLAY_NAME_IDS``.
    """
    parts = name.strip().lower().split()
    if not parts or parts[0] != "claude":
        raise ValueError(f"not a Claude model display name: {name!r}")
    family = "-".join(parts[1:-1])
    version = parts[-1].replace(".", "-")
    if "-" not in version and version.isdigit() and int(version) <= 4:
        version += "-0"
    model_id = f"claude-{family}-{version}"
    return _DISPLAY_NAME_IDS.get(model_id, model_id)


def parse_pricing_markdown(text: str) -> dict[str, dict[str, float]]:
    """Extract ``{model_id: {in, out, cw5m, cw1h, cr}}`` from the pricing page.

    Reads the first table under ``## Model pricing`` whose rows have five
    ``$N / MTok`` cells: base input, 5m write, 1h write, cache read, output.
    Rows with a different cell count (fast-mode and batch tables) are ignored.
    Returns an empty dict when the section is missing, so a page redesign
    fails loudly in the caller rather than writing an empty table.
    """
    models: dict[str, dict[str, float]] = {}
    in_section = False
    for line in text.splitlines():
        if line.startswith("## "):
            if in_section:
                break
            in_section = line.strip().lower() == "## model pricing"
            continue
        if not in_section:
            continue
        row = _PRICE_ROW.match(line)
        if not row:
            continue
        prices = [float(m) for m in _PRICE_CELL.findall(row.group(2))]
        if len(prices) != 5:
            continue
        base_in, cw5m, cw1h, cr, out = prices
        try:
            model_id = model_id_from_display_name(row.group(1))
        except ValueError:
            continue
        models[model_id] = {"in": base_in, "out": out, "cw5m": cw5m, "cw1h": cw1h, "cr": cr}
    return models


def fetch_live_pricing(url: str = PRICING_URL, timeout: float = 10.0) -> dict[str, dict[str, float]]:
    """GET the pricing page and parse it. Raises on network or parse failure."""
    req = urllib.request.Request(url, headers={"User-Agent": "multiplai-core costing"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    models = parse_pricing_markdown(text)
    if not models:
        raise ValueError(f"no model pricing table found at {url}")
    return models


def refresh_pricing(
    *,
    url: str = PRICING_URL,
    max_age: timedelta = timedelta(days=1),
    force: bool = False,
    timeout: float = 10.0,
) -> Path | None:
    """Fetch current list prices and write ``pricing_cache_path()``.

    Skips the fetch when the cached file is younger than *max_age* (unless
    *force*). Returns the path written or reused, or ``None`` when the fetch
    failed — the caller keeps pricing from whatever ``load_pricing()`` finds,
    and the failure is logged at WARNING, never raised: a pricing refresh must
    not stop a collect pass.
    """
    path = pricing_cache_path()
    if not force and path.is_file():
        age = datetime.now(UTC) - datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        if age < max_age:
            return path
    try:
        models = fetch_live_pricing(url, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — network/parse failure is non-fatal
        logger.warning("Pricing refresh from %s failed (%s); using cached/bundled table", url, exc)
        return None
    payload = {
        "updated": _today().isoformat(),
        "source": url,
        "models": models,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    reset_pricing_cache()
    logger.info("Pricing refreshed from %s: %d models", url, len(models))
    return path


def pricing_age_days() -> int:
    """Days since the active table's ``updated`` date (bundled or refreshed)."""
    updated = load_pricing().get("updated")
    try:
        return (_today() - date.fromisoformat(str(updated))).days
    except (TypeError, ValueError):
        return 10**6


def _today() -> date:
    return datetime.now(UTC).date()


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
    best_prefix = max(
        (k for k in models if model.startswith(k)), key=len, default=None
    )
    if best_prefix is not None:
        return models[best_prefix], False
    if model not in _fallback_warned:
        _fallback_warned.add(model)
        logger.warning(
            "No list price for model %r; pricing at fallback $%s/$%s per MTok "
            "(records carry pricing_fallback=true). Run refresh_pricing() or "
            "add the model to pricing.json.",
            model, pricing["fallback"]["in"], pricing["fallback"]["out"],
        )
    return pricing["fallback"], True


def tier_prices(rates: dict) -> dict[str, float]:
    """Per-MTok price for every tier: explicit entries win over multipliers."""
    mult = load_pricing()["multipliers"]
    return {
        "in": rates["in"],
        "out": rates["out"],
        "cw5m": rates.get("cw5m", mult["cw5m"] * rates["in"]),
        "cw1h": rates.get("cw1h", mult["cw1h"] * rates["in"]),
        "cr": rates.get("cr", mult["cr"] * rates["in"]),
    }


def price_tokens(model: str, tokens: TokenCounts) -> tuple[float, bool]:
    """Return ``(cost_usd, used_fallback)`` for one call's token counts."""
    rates, fallback = resolve_model_rates(model)
    p = tier_prices(rates)
    cost = (
        tokens.input * p["in"]
        + tokens.output * p["out"]
        + tokens.cw5m * p["cw5m"]
        + tokens.cw1h * p["cw1h"]
        + tokens.cr * p["cr"]
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
