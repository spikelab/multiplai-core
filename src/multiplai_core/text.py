"""Text/JSON extraction helpers shared across the Multiplai plugins.

Previously copied into buildme (build_pipeline) and deep-research
(research_pipeline) with slight drift; this is the single source of truth.
"""

from __future__ import annotations

import json
import re

# Explicit ```json fences first, then bare ``` fences.
_FENCE_RES = (
    re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL),
    re.compile(r"```\s*\n(.*?)\n```", re.DOTALL),
)


def extract_json(text: str) -> dict | list:
    """Extract a JSON object or array from a model response.

    Handles:
    - ```json ... ``` fenced code blocks
    - Plain JSON with surrounding prose (the first complete object/array,
      via ``json.JSONDecoder.raw_decode``)

    Raises ``ValueError`` (``json.JSONDecodeError`` is a subclass) on empty
    input, no JSON found, or malformed/unbalanced JSON.
    """
    if not text or not text.strip():
        raise ValueError("Empty response")

    # 1. Fenced code blocks. A non-JSON fence earlier in the text (e.g. a
    #    ```python example before the answer) must not shadow the real JSON,
    #    so every candidate is tried and a non-parsing fence falls through to
    #    the raw_decode scan instead of raising.
    for pattern in _FENCE_RES:
        for fence_match in pattern.finditer(text):
            try:
                return json.loads(fence_match.group(1).strip())
            except json.JSONDecodeError:
                continue

    # 2. First complete JSON object/array. raw_decode parses from the first
    #    bracket and stops at the value's end, ignoring trailing prose — the
    #    stdlib's string/escape-correct version of a bracket-balancing scan.
    stripped = text.strip()
    start = min(
        (i for i in (stripped.find("{"), stripped.find("[")) if i != -1),
        default=None,
    )
    if start is None:
        raise ValueError("No JSON object/array found in response")
    value, _ = json.JSONDecoder().raw_decode(stripped, start)
    return value
