"""Parse an agent's final answer into a pydantic schema, and build the re-ask
that turns an answer which did not parse into one that does.

Copied from the review pipeline (multiplai-cc-mktplace,
``review_pipeline/sdk.py``), which the codebase-walkthrough pipeline had copied
in turn. There is deliberately no call loop here: how many times to re-ask, and
whether a re-ask reformats the answer or re-runs the whole prompt, differs
between the plugins and stays in each plugin's ``sdk.py``.

Nothing here imports pydantic. *schema* is any class with pydantic v2's
``model_validate``, ``model_fields`` and ``model_json_schema``; pydantic's
``ValidationError`` is a ``ValueError``, so callers catch ``ValueError``.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from .text import extract_json

T = TypeVar("T")

# A reformat re-ask carries the rejected answer. Past this many characters the
# answer is cut, so one runaway answer cannot make the re-ask oversized.
REFORMAT_MAX_ANSWER_CHARS = 60_000


class StructuredAnswerError(ValueError):
    """The answer holds no JSON object naming any of the schema's fields.

    A ``ValueError``, so a caller that already catches ``ValueError`` (and
    pydantic's ``ValidationError``) to decide on a re-ask keeps doing so.
    """


def parse_answer(text: str, schema: type[T]) -> T:
    """*text* as *schema*. Raises ``StructuredAnswerError``, ``ValueError`` or
    pydantic's ``ValidationError``.

    An answer that names none of the schema's fields holds no JSON answer at
    all (a markdown report, say). ``extract_json`` would still parse the first
    brace in its prose, and the error would describe that fragment, so this
    case gets its own message: "no JSON object with the field(s) ...".
    """
    model: Any = schema
    try:
        return model.model_validate(extract_json(text))
    except ValueError as e:  # pydantic's ValidationError is a ValueError
        if not any(f'"{name}"' in text for name in model.model_fields):
            start = " ".join(text.split())[:120]
            raise StructuredAnswerError(
                f"the answer contains no JSON object with the field(s) "
                f"{', '.join(model.model_fields)}; it starts: {start!r}"
            ) from e
        raise


def reformat_prompt(text: str, schema: type, error: Exception) -> str:
    """A one-turn, no-tools prompt that rewrites *text* as one JSON object of *schema*.

    *error* is why a program rejected *text*; it is quoted (cut to 2,000
    characters) so the reformatting model knows what to fix.
    """
    model: Any = schema
    if len(text) > REFORMAT_MAX_ANSWER_CHARS:
        text = text[:REFORMAT_MAX_ANSWER_CHARS] + "\n[... answer cut ...]"
    return (
        "Another agent was asked for ONE JSON object matching the schema below. A program rejected "
        f"its answer: {str(error)[:2000]}\n\n"
        "Rewrite that answer as ONE JSON object matching the schema. Use only what the answer says: "
        "copy every quote, path and line number exactly, and add nothing it does not state. Leave out "
        "any item the answer cannot fill in; when it holds no item for a list, leave the list empty.\n\n"
        f"Schema:\n{json.dumps(model.model_json_schema(), indent=1)}\n\n"
        f"<answer>\n{text}\n</answer>\n\n"
        "Answer with ONE JSON object and nothing after it."
    )
