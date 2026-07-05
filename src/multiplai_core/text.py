"""Text/JSON extraction helpers shared across the Multiplai plugins.

Previously copied into buildme (build_pipeline) and deep-research
(research_pipeline) with slight drift; this is the single source of truth.
"""

from __future__ import annotations

import json
import re


def extract_json(text: str) -> dict | list:
    """Extract a JSON object or array from a model response.

    Handles:
    - ```json ... ``` fenced code blocks
    - Plain JSON with surrounding prose
    - Multi-line JSON objects (via bracket balancing, string-aware)

    Raises ``ValueError`` on empty input, no JSON found, or unbalanced JSON.
    """
    if not text or not text.strip():
        raise ValueError("Empty response")

    # 1. Fenced code blocks — try explicit ```json fences first, then bare
    #    ``` fences. A non-JSON fence earlier in the text (e.g. a ```python
    #    example before the answer) must not shadow the real JSON, so every
    #    candidate is tried and a non-parsing fence falls through to the
    #    bracket-balancing scan instead of raising.
    for pattern in (r"```json\s*\n(.*?)\n```", r"```\s*\n(.*?)\n```"):
        for fence_match in re.finditer(pattern, text, re.DOTALL):
            try:
                return json.loads(fence_match.group(1).strip())
            except json.JSONDecodeError:
                continue

    # 2. First complete JSON object/array via bracket balancing
    stripped = text.strip()
    start = None
    for i, ch in enumerate(stripped):
        if ch in "{[":
            start = i
            break
    if start is None:
        raise ValueError("No JSON object/array found in response")

    open_ch = stripped[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    escape = False
    end = None

    for i in range(start, len(stripped)):
        ch = stripped[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is None:
        raise ValueError("Unbalanced JSON in response")

    return json.loads(stripped[start : end + 1])
