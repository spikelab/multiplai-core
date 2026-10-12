"""Pi event-stream parsing.

Ported from DolceEvals ``piharness/events.py`` (DolceTech/DolceEvals@d755fcc)
for :mod:`multiplai_core.pi.backend`. Standard library only. Mentions of
``metrics.py`` below refer to that repo; it was not ported.

Consumes the LF-delimited JSONL that `pi --mode rpc` writes to stdout and
normalises it into typed records.

## Why this module is deliberately tolerant

Pi is early-stage software. `SCHEMA` below was **validated against a live
`pi --mode rpc` process (pi 0.83.0, 2026-08-04)** -- event names, `toolName` /
`toolCallId` / `args` fields, and `message.usage` all confirmed on the wire.
Everything schema-shaped stays isolated in `SCHEMA` regardless, so drift in a
future Pi release is a one-place edit rather than a refactor.

Two invariants hold regardless of schema drift:

1. **Parsing is total.** An unrecognised event becomes `Kind.UNKNOWN` with its
   payload preserved in `raw`. Events are never dropped and never fatal --
   silently discarding events would corrupt every metric downstream while
   looking perfectly healthy.
2. **Absence is not zero.** If a field a metric needs is missing, the metric
   reports `None` ("could not measure"), never `0` ("measured, found none").
   See `metrics.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator


class Kind(str, Enum):
    """Normalised event kinds."""

    TURN_START = "turn_start"
    TURN_END = "turn_end"
    TOOL_START = "tool_start"
    TOOL_UPDATE = "tool_update"
    TOOL_END = "tool_end"
    COMPACTION_START = "compaction_start"
    COMPACTION_END = "compaction_end"
    RETRY_START = "retry_start"
    RETRY_END = "retry_end"
    MESSAGE = "message"
    AGENT_SETTLED = "agent_settled"
    """Pi's run is fully settled: no retry, compaction, or queued continuation
    remains. The RPC process stays alive after this -- it is a long-lived
    server -- so this event, not stream EOF, is what ends a run."""
    USAGE = "usage"
    RESULT = "result"
    ERROR = "error"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------
# The one place that encodes assumptions about Pi's wire format.
# Correct this after the first live run; nothing else should need to change.
# --------------------------------------------------------------------------
SCHEMA: dict[str, Any] = {
    # Where to find the event discriminator, in priority order.
    "type_keys": ("type", "event", "kind"),
    # Raw Pi event name -> normalised Kind.
    "type_map": {
        "turn_start": Kind.TURN_START,
        "turn_end": Kind.TURN_END,
        "tool_execution_start": Kind.TOOL_START,
        "tool_execution_update": Kind.TOOL_UPDATE,
        "tool_execution_end": Kind.TOOL_END,
        "compaction_start": Kind.COMPACTION_START,
        "compaction_end": Kind.COMPACTION_END,
        "auto_retry_start": Kind.RETRY_START,
        "auto_retry_end": Kind.RETRY_END,
        "agent_settled": Kind.AGENT_SETTLED,
        "message_start": Kind.MESSAGE,
        "message_update": Kind.MESSAGE,  # pi 0.83.0 name (docs/rpc.md)
        "message_end": Kind.MESSAGE,
        "usage": Kind.USAGE,
        "result": Kind.RESULT,
        "error": Kind.ERROR,
        "extension_error": Kind.ERROR,
    },
    # Candidate paths for common fields, tried in order. Dotted = nested.
    "tool_name_keys": ("toolName", "tool", "name", "tool_name"),
    "tool_id_keys": ("toolCallId", "toolUseId", "tool_use_id", "id", "callId"),
    "tool_input_keys": ("input", "args", "arguments", "parameters"),
    "tool_output_keys": ("output", "result", "content", "stdout"),
    "error_flag_keys": ("isError", "is_error", "error", "failed"),
    "error_text_keys": ("error", "errorMessage", "message", "stderr"),
    # Within a tool input, where a file path or command tends to live.
    "path_keys": ("file_path", "filePath", "path", "notebook_path"),
    "command_keys": ("command", "cmd", "script"),
    "old_string_keys": ("old_string", "oldString", "old"),
    "new_string_keys": ("new_string", "newString", "new"),
}


def first_value(d: dict[str, Any], keys: Iterable[str]) -> Any:
    """Return the first present, non-None value among `keys`.

    Public because `metrics.py` needs the same candidate-key lookup to read
    tool inputs, and both must agree on `SCHEMA`.
    """
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return None


# Internal shorthand.
_first = first_value


@dataclass(frozen=True)
class Event:
    """One normalised event from the Pi stream."""

    seq: int
    kind: Kind
    raw: dict[str, Any]

    # Tool-shaped fields; None when not applicable or not present.
    tool_name: str | None = None
    tool_id: str | None = None
    target: str | None = None
    """What the tool acted on: a file path, or a command string for shell tools."""
    is_error: bool | None = None
    """None means "the stream did not say", which is not the same as False."""
    error_text: str | None = None
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_output: str | None = None

    @property
    def raw_type(self) -> str:
        return str(_first(self.raw, SCHEMA["type_keys"]) or "")


def _extract_target(tool_input: dict[str, Any]) -> str | None:
    """A stable identifier for *what* a tool acted on.

    File-editing tools key on the path; shell tools key on the command string.
    Used by `metrics` to decide whether two calls addressed the same thing.
    """
    path = _first(tool_input, SCHEMA["path_keys"])
    if isinstance(path, str) and path:
        return path
    cmd = _first(tool_input, SCHEMA["command_keys"])
    if isinstance(cmd, str) and cmd:
        return cmd.strip()
    return None


def _coerce_error(payload: dict[str, Any]) -> tuple[bool | None, str | None]:
    """Decide whether a tool result represents failure, and extract its text.

    Returns `(None, None)` when the payload carries no signal either way -- the
    caller must not read that as success.
    """
    flag = _first(payload, SCHEMA["error_flag_keys"])
    text: str | None = None

    raw_text = _first(payload, SCHEMA["error_text_keys"])
    if isinstance(raw_text, str) and raw_text:
        text = raw_text
    elif isinstance(raw_text, dict):
        inner = _first(raw_text, ("message", "text", "detail"))
        if isinstance(inner, str):
            text = inner

    if isinstance(flag, bool):
        return flag, text
    if isinstance(flag, str) and flag:
        return True, text or flag
    if isinstance(flag, dict) and flag:
        return True, text
    if text:
        # Error text present but no explicit flag: treat as an error.
        return True, text
    return None, None


def parse_event(seq: int, payload: dict[str, Any]) -> Event:
    """Normalise one decoded JSON object into an `Event`. Never raises."""
    raw_type = _first(payload, SCHEMA["type_keys"])
    kind = SCHEMA["type_map"].get(str(raw_type), Kind.UNKNOWN)

    tool_input = _first(payload, SCHEMA["tool_input_keys"])
    if not isinstance(tool_input, dict):
        tool_input = {}

    tool_name = _first(payload, SCHEMA["tool_name_keys"])
    tool_name = tool_name if isinstance(tool_name, str) else None

    tool_id = _first(payload, SCHEMA["tool_id_keys"])
    tool_id = str(tool_id) if tool_id is not None else None

    is_error: bool | None = None
    error_text: str | None = None
    tool_output: str | None = None
    if kind in (Kind.TOOL_END, Kind.TOOL_UPDATE, Kind.ERROR):
        is_error, error_text = _coerce_error(payload)
        out = _first(payload, SCHEMA["tool_output_keys"])
        if isinstance(out, str):
            tool_output = out
        elif out is not None:
            tool_output = json.dumps(out, sort_keys=True)[:4000]
        if kind is Kind.ERROR and is_error is None:
            is_error = True

    return Event(
        seq=seq,
        kind=kind,
        raw=payload,
        tool_name=tool_name,
        tool_id=tool_id,
        target=_extract_target(tool_input),
        is_error=is_error,
        error_text=error_text,
        tool_input=tool_input,
        tool_output=tool_output,
    )


def parse_line(seq: int, line: str) -> Event:
    """Parse one JSONL line into an `Event`. Never raises.

    A line that is not valid JSON, or that decodes to something other than an
    object, becomes `Kind.UNKNOWN` carrying the offending text. Shared with
    `rpc.PiRpc`, which reads the same stream live rather than from a file --
    both paths must be equally unable to die on a malformed line.
    """
    text = line.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return Event(seq=seq, kind=Kind.UNKNOWN, raw={"_unparseable": text[:4000]})
    if not isinstance(payload, dict):
        return Event(seq=seq, kind=Kind.UNKNOWN, raw={"_non_object": payload})
    return parse_event(seq, payload)


def parse_stream(lines: Iterable[str]) -> Iterator[Event]:
    """Parse LF-delimited JSONL into events.

    Blank lines are skipped without consuming a sequence number.
    """
    seq = 0
    for line in lines:
        if not line.strip():
            continue
        yield parse_line(seq, line)
        seq += 1
