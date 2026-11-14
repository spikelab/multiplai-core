"""Run a Fable step on Opus instead when the Fable or weekly allowance runs out too fast.

:func:`decide` is called by :func:`multiplai_core.agent_runner.run_agent` once
before each step whose model is a ``claude-fable-*`` or ``claude-mythos-*`` ID
(or the ``fable`` family name). It reads the subscription's usage from
``GET https://api.anthropic.com/api/oauth/usage`` (internal and undocumented,
so it may change), with the OAuth token of
``$CLAUDE_CONFIG_DIR/.credentials.json``. The token goes into the one request
header and is never logged, printed or saved.

The **pace** of a weekly limit is ``percent / (100 x elapsed fraction of its
7-day window)``. A pace of 1.0 runs the limit out exactly at its reset. A Fable
step runs on ``FABLE_GUARD_FALLBACK`` when any rule holds:

1. Fable percent >= ``FABLE_GUARD_MAX_PERCENT`` (80).
2. Fable percent >= ``FABLE_GUARD_MIN_PERCENT`` (25) and Fable pace >
   ``FABLE_GUARD_MAX_PACE`` (1.0).
3. Weekly-all percent >= 50 and weekly-all pace > ``FABLE_GUARD_MAX_PACE``.
4. Five-hour utilization >= ``FABLE_GUARD_MAX_FIVE_HOUR`` (85).

Once backed out, Fable comes back only when no rule holds **and** every pace is
below ``FABLE_GUARD_RESUME_PACE`` (0.8). A weekly reset clears the back-out.

``FABLE_GUARD`` in ``multiplai.conf`` is ``off`` (no check, no log),
``monitor`` (the default: decides and logs, never switches) or ``enforce``.
Every key is read again on each decision.

State lives in ``skill_state_dir("fable-guard")``: ``state.json`` (the last
reading, when it was read, and the back-out) and ``decisions.jsonl`` (one line
per decision). ``python -m multiplai_core.fable_guard status`` prints the
current reading and what the next Fable step would run on.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .env import CURRENT_MODEL, expand_family, load_multiplai_conf

logger = logging.getLogger(__name__)

USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
OAUTH_BETA = "oauth-2025-04-20"
FETCH_TIMEOUT_S = 5.0
WEEK_S = 7 * 24 * 3600.0
STRAY_TMP_S = 3600.0
FABLE_PREFIXES = ("claude-fable-", "claude-mythos-")
WEEKLY_MIN_PERCENT = 50.0  # rule 3 applies from here
MODES = ("off", "monitor", "enforce")

DEFAULTS: dict[str, object] = {
    "FABLE_GUARD": "monitor",
    "FABLE_GUARD_TTL_S": 60.0,
    "FABLE_GUARD_FALLBACK": CURRENT_MODEL["opus"],
    "FABLE_GUARD_MAX_PERCENT": 80.0,
    "FABLE_GUARD_MIN_PERCENT": 25.0,
    "FABLE_GUARD_MAX_PACE": 1.0,
    "FABLE_GUARD_MAX_FIVE_HOUR": 85.0,
    "FABLE_GUARD_RESUME_PACE": 0.8,
    "FABLE_GUARD_ON_UNREADABLE": "fallback",
}

# Stable reason strings, so `group_by(.reason)` over the log counts each rule.
R_FABLE_MAX = "fable percent at or over max"
R_FABLE_PACE = "fable pace over max"
R_WEEKLY_PACE = "weekly pace over max"
R_FIVE_HOUR = "five-hour at or over max"
R_HOLD = "backed out, a pace not yet below resume"
R_OK = "within limits"
R_RESET = "window reset"
R_UNREADABLE_FALLBACK = "endpoint unreadable, fallback"
R_UNREADABLE_KEEP = "endpoint unreadable, keep"
R_NOT_FABLE = "not a fable model"
R_OFF = "guard off"


@dataclass(frozen=True)
class Usage:
    """One reading of the usage endpoint. A field is None when the endpoint left it out."""

    fable_percent: float | None
    fable_resets_at: str | None
    weekly_percent: float | None
    weekly_resets_at: str | None
    five_hour: float | None
    five_hour_resets_at: str | None


@dataclass(frozen=True)
class Decision:
    model_used: str
    reason: str
    switched: bool = False  # True only when the step runs on another model


@dataclass(frozen=True)
class Settings:
    mode: str
    ttl_s: float
    fallback: str
    max_percent: float
    min_percent: float
    max_pace: float
    max_five_hour: float
    resume_pace: float
    on_unreadable: str


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def _float(conf: dict, key: str) -> float:
    raw = conf.get(key)
    if raw in (None, ""):
        return float(DEFAULTS[key])  # type: ignore[arg-type]
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        logger.warning("%s=%r is not a number; using %s", key, raw, DEFAULTS[key])
        return float(DEFAULTS[key])  # type: ignore[arg-type]


def load_settings() -> Settings:
    """The guard's keys from ``multiplai.conf``, defaults for the rest."""
    conf = load_multiplai_conf()
    mode = str(conf.get("FABLE_GUARD") or DEFAULTS["FABLE_GUARD"]).strip().lower()
    if mode not in MODES:
        logger.warning("FABLE_GUARD=%r is not one of %s; using monitor", mode, MODES)
        mode = "monitor"
    on_unreadable = (
        str(conf.get("FABLE_GUARD_ON_UNREADABLE") or "fallback").strip().lower()
    )
    if on_unreadable not in ("fallback", "keep"):
        logger.warning(
            "FABLE_GUARD_ON_UNREADABLE=%r is not fallback or keep; using fallback",
            on_unreadable,
        )
        on_unreadable = "fallback"
    fallback = str(conf.get("FABLE_GUARD_FALLBACK") or "").strip()
    return Settings(
        mode=mode,
        ttl_s=_float(conf, "FABLE_GUARD_TTL_S"),
        fallback=expand_family(fallback)
        if fallback
        else str(DEFAULTS["FABLE_GUARD_FALLBACK"]),
        max_percent=_float(conf, "FABLE_GUARD_MAX_PERCENT"),
        min_percent=_float(conf, "FABLE_GUARD_MIN_PERCENT"),
        max_pace=_float(conf, "FABLE_GUARD_MAX_PACE"),
        max_five_hour=_float(conf, "FABLE_GUARD_MAX_FIVE_HOUR"),
        resume_pace=_float(conf, "FABLE_GUARD_RESUME_PACE"),
        on_unreadable=on_unreadable,
    )


def is_fable(model: str | None) -> bool:
    """True for a Fable or Mythos model ID, or the ``fable`` family name."""
    if not model:
        return False
    return expand_family(model).lower().startswith(FABLE_PREFIXES)


# ---------------------------------------------------------------------------
# Reading the endpoint
# ---------------------------------------------------------------------------


def _credentials_path() -> Path:
    root = os.environ.get("CLAUDE_CONFIG_DIR") or str(Path.home() / ".claude")
    return Path(root) / ".credentials.json"


def _token() -> str | None:
    try:
        data = json.loads(_credentials_path().read_text(encoding="utf-8"))
        token = data["claudeAiOauth"]["accessToken"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return token if isinstance(token, str) and token else None


def _fetch(token: str) -> dict:
    """The endpoint's JSON. Raises on anything but a 200 with a JSON object."""
    request = urllib.request.Request(
        USAGE_URL,
        headers={"Authorization": f"Bearer {token}", "anthropic-beta": OAUTH_BETA},
    )
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_S) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise TypeError("the answer is not a JSON object")
    return data


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    return float(value) if isinstance(value, int | float) else None


def parse_usage(data: dict) -> Usage:
    """Pick the weekly-all limit, the Fable-scoped weekly limit and the five-hour window."""
    fable_percent = fable_resets = weekly_percent = weekly_resets = None
    limits = data.get("limits")
    for entry in limits if isinstance(limits, list) else []:
        if not isinstance(entry, dict):
            continue
        kind = entry.get("kind")
        if kind == "weekly_all" and weekly_percent is None:
            weekly_percent = _number(entry.get("percent"))
            weekly_resets = entry.get("resets_at") or None
        elif kind == "weekly_scoped" and fable_percent is None:
            scope = entry.get("scope")
            model = scope.get("model") if isinstance(scope, dict) else None
            name = model.get("display_name") if isinstance(model, dict) else None
            if name == "Fable":
                fable_percent = _number(entry.get("percent"))
                fable_resets = entry.get("resets_at") or None
    five = data.get("five_hour")
    five_hour = _number(five.get("utilization")) if isinstance(five, dict) else None
    five_resets = (five.get("resets_at") or None) if isinstance(five, dict) else None
    return Usage(
        fable_percent=fable_percent,
        fable_resets_at=str(fable_resets) if fable_resets else None,
        weekly_percent=weekly_percent,
        weekly_resets_at=str(weekly_resets) if weekly_resets else None,
        five_hour=five_hour,
        five_hour_resets_at=str(five_resets) if five_resets else None,
    )


def read_usage() -> Usage | None:
    """One GET of the endpoint, or None on a missing token or any HTTP or parse failure."""
    token = _token()
    if token is None:
        logger.info("fable-guard: no OAuth token in %s", _credentials_path())
        return None
    try:
        data = _fetch(token)
    except Exception as e:  # noqa: BLE001 - any failure means "unreadable"
        # The exception's class only: an HTTP error's text could echo a header.
        logger.info("fable-guard: usage endpoint not readable (%s)", type(e).__name__)
        return None
    try:
        return parse_usage(data)
    except Exception as e:  # noqa: BLE001
        logger.info("fable-guard: usage answer not parseable (%s)", type(e).__name__)
        return None


# ---------------------------------------------------------------------------
# Pace and the rules
# ---------------------------------------------------------------------------


def _epoch(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def pace(percent: float | None, resets_at: str | None, now: float) -> float | None:
    """``percent / (100 x elapsed fraction)`` of a 7-day window; None when unknown."""
    reset = _epoch(resets_at)
    if percent is None or reset is None:
        return None
    elapsed = 1.0 - (reset - now) / WEEK_S
    elapsed = min(elapsed, 1.0)
    if elapsed <= 0.0:
        return math.inf if percent > 0 else 0.0
    return percent / (100.0 * elapsed)


def _paces(usage: Usage, now: float) -> tuple[float | None, float | None]:
    return (
        pace(usage.fable_percent, usage.fable_resets_at, now),
        pace(usage.weekly_percent, usage.weekly_resets_at, now),
    )


def fired_rule(usage: Usage, s: Settings, now: float) -> str | None:
    """The first of rules 1-4 that holds, as its reason string, or None."""
    fable_pace, weekly_pace = _paces(usage, now)
    fp = usage.fable_percent
    if fp is not None and fp >= s.max_percent:
        return R_FABLE_MAX
    if (
        fp is not None
        and fp >= s.min_percent
        and fable_pace is not None
        and fable_pace > s.max_pace
    ):
        return R_FABLE_PACE
    wp = usage.weekly_percent
    if (
        wp is not None
        and wp >= WEEKLY_MIN_PERCENT
        and weekly_pace is not None
        and weekly_pace > s.max_pace
    ):
        return R_WEEKLY_PACE
    if usage.five_hour is not None and usage.five_hour >= s.max_five_hour:
        return R_FIVE_HOUR
    return None


def evaluate(
    usage: Usage, s: Settings, backed_out: bool, now: float
) -> tuple[bool, str]:
    """``(back out?, reason)`` for one reading, with the resume threshold applied."""
    rule = fired_rule(usage, s, now)
    if rule is not None:
        return True, rule
    if backed_out:
        known = [p for p in _paces(usage, now) if p is not None]
        if any(p >= s.resume_pace for p in known):
            return True, R_HOLD
    return False, R_OK


# ---------------------------------------------------------------------------
# State on disk
# ---------------------------------------------------------------------------


def _state_dir() -> Path:
    from .paths import get_paths

    return get_paths().skill_state_dir("fable-guard")


def state_path() -> Path:
    return _state_dir() / "state.json"


def log_path() -> Path:
    return _state_dir() / "decisions.jsonl"


def load_state() -> dict:
    """``state.json`` as a dict; an absent or unparseable file is an empty state (S2, S5)."""
    try:
        data = json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _reading_from(state: dict) -> Usage | None:
    raw = state.get("reading")
    if not isinstance(raw, dict):
        return None
    try:
        return Usage(**{k: raw.get(k) for k in Usage.__dataclass_fields__})
    except TypeError:
        return None


def save_state(state: dict) -> None:
    """Write ``state.json`` through a temp file and ``os.replace`` (S1, S9).

    Temp files older than an hour, left by a writer killed before its
    ``os.replace``, are deleted first.
    """
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for stray in path.parent.glob(f"{path.name}.*.tmp"):
        try:
            if now - stray.stat().st_mtime > STRAY_TMP_S:
                stray.unlink()
        except OSError:
            pass
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{os.urandom(4).hex()}.tmp")
    try:
        tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def append_decision(row: dict) -> None:
    """Append one JSON line in one ``write()``; start a new line after a cut one (S3)."""
    path = log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, sort_keys=True) + "\n"
        try:
            with path.open("rb") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    line = "\n" + line
        except (OSError, ValueError):
            pass  # absent or empty file
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
    except OSError:
        logger.warning("fable-guard: could not append to %s", path, exc_info=True)


def read_decisions() -> list[dict]:
    """Every parseable line of ``decisions.jsonl``; cut or corrupt lines are skipped (S3)."""
    try:
        text = log_path().read_text(encoding="utf-8")
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _weekly_reset_passed(usage: Usage, now: float) -> bool:
    for iso in (usage.weekly_resets_at, usage.fable_resets_at):
        t = _epoch(iso)
        if t is not None and t <= now:
            return True
    return False


def _any_reset_passed(usage: Usage, now: float) -> bool:
    if _weekly_reset_passed(usage, now):
        return True
    t = _epoch(usage.five_hour_resets_at)
    return t is not None and t <= now


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _current_reading(state: dict, s: Settings, now: float) -> tuple[Usage | None, bool]:
    """``(reading, weekly window reset since the cached one)``.

    A cached reading younger than the TTL, with no reset time passed, is
    reused; otherwise the endpoint is read once (S6: no retry loop). A failed
    read inside the TTL keeps the cached reading.
    """
    cached = _reading_from(state)
    read_at = state.get("read_at")
    reset = cached is not None and _weekly_reset_passed(cached, now)
    fresh = (
        cached is not None
        and isinstance(read_at, int | float)
        and 0 <= now - read_at < s.ttl_s
        and not _any_reset_passed(cached, now)
    )
    if fresh:
        return cached, False
    usage = read_usage()
    if usage is not None:
        state["reading"] = asdict(usage)
        state["read_at"] = now
    return usage, reset


def decide(model: str, task: str = "") -> Decision:
    """The model a step asking for *model* runs on, and why.

    A model that is not Fable passes through with no I/O. Otherwise the
    decision is logged to ``decisions.jsonl``; in ``monitor`` mode
    ``model_used`` is always *model*.
    """
    if not is_fable(model):
        return Decision(model, R_NOT_FABLE)
    s = load_settings()
    if s.mode == "off":
        return Decision(model, R_OFF)
    now = time.time()
    state = load_state()
    usage, reset = _current_reading(state, s, now)
    backed_out = bool(state.get("backed_out")) and not reset

    if usage is None:
        back = s.on_unreadable == "fallback"
        reason = R_UNREADABLE_FALLBACK if back else R_UNREADABLE_KEEP
        new_backed_out = backed_out  # an unreadable endpoint changes nothing saved
    else:
        back, reason = evaluate(usage, s, backed_out, now)
        if not back and reset:
            reason = R_RESET
        new_backed_out = back

    if not new_backed_out:
        state["since"] = None
    elif not backed_out:
        state["since"] = _iso(now)
    state["backed_out"] = new_backed_out
    try:
        save_state(state)
    except OSError:
        logger.warning("fable-guard: could not write %s", state_path(), exc_info=True)

    enforce = s.mode == "enforce"
    used = s.fallback if (back and enforce) else model
    fable_pace, weekly_pace = _paces(usage, now) if usage else (None, None)
    append_decision(
        {
            "ts": _iso(now),
            "task": task,
            "requested": model,
            "used": used,
            "mode": s.mode,
            "reason": reason,
            "fable_percent": usage.fable_percent if usage else None,
            "fable_pace": _round(fable_pace),
            "weekly_percent": usage.weekly_percent if usage else None,
            "weekly_pace": _round(weekly_pace),
            "five_hour": usage.five_hour if usage else None,
        }
    )
    return Decision(used, reason, switched=used != model)


def _round(value: float | None) -> float | str | None:
    if value is None:
        return None
    if math.isinf(value):
        return "inf"
    return round(value, 3)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def _fmt(value: float | str | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    return f"{value:.1f}{suffix}" if suffix else f"{value:.2f}"


def status(out=None) -> int:
    """Print the reading, the paces, the back-out and the next decision. Writes nothing."""
    out = out or sys.stdout
    s = load_settings()
    now = time.time()
    state = load_state()
    cached = _reading_from(state)
    read_at = state.get("read_at")
    usage = None
    source = "endpoint"
    if (
        cached is not None
        and isinstance(read_at, int | float)
        and 0 <= now - read_at < s.ttl_s
        and not _any_reset_passed(cached, now)
    ):
        usage, source = cached, f"cache ({now - read_at:.0f}s old)"
    else:
        usage = read_usage()
    reset = cached is not None and _weekly_reset_passed(cached, now)
    backed_out = bool(state.get("backed_out")) and not reset
    model = CURRENT_MODEL["fable"]
    print(f"mode:           {s.mode}", file=out)
    if usage is None:
        print("reading:        endpoint not readable", file=out)
        back = s.on_unreadable == "fallback"
        reason = R_UNREADABLE_FALLBACK if back else R_UNREADABLE_KEEP
    else:
        fable_pace, weekly_pace = _paces(usage, now)
        print(f"reading:        {source}", file=out)
        print(
            f"fable:          {_fmt(usage.fable_percent, '%')}  pace {_fmt(_round(fable_pace))}"
            f"  resets {usage.fable_resets_at or 'n/a'}",
            file=out,
        )
        print(
            f"weekly (all):   {_fmt(usage.weekly_percent, '%')}  pace {_fmt(_round(weekly_pace))}"
            f"  resets {usage.weekly_resets_at or 'n/a'}",
            file=out,
        )
        print(f"five-hour:      {_fmt(usage.five_hour, '%')}", file=out)
        back, reason = evaluate(usage, s, backed_out, now)
    print(
        f"backed out:     {'yes since ' + str(state.get('since')) if backed_out else 'no'}",
        file=out,
    )
    target = s.fallback if back else model
    verb = "would run" if s.mode == "monitor" else "runs"
    if s.mode == "off":
        print(f"next {model} step: runs on {model} (guard off)", file=out)
    else:
        print(f"next {model} step: {verb} on {target} ({reason})", file=out)
    rows = [r for r in read_decisions() if (_epoch(r.get("ts")) or 0) >= now - WEEK_S]
    if rows:
        counts = Counter(str(r.get("reason")) for r in rows)
        print(f"decisions, last 7 days: {len(rows)}", file=out)
        for reason_name, n in counts.most_common():
            print(f"  {n:5d}  {reason_name}", file=out)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] != ["status"] or len(args) > 1:
        print("usage: python -m multiplai_core.fable_guard status", file=sys.stderr)
        return 2
    return status()


if __name__ == "__main__":
    raise SystemExit(main())
