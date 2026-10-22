"""parse_answer and reformat_prompt, ported from the review and walkthrough suites (test_sdk.py)."""

import ast
import json

import pytest
from pydantic import BaseModel, ValidationError

from multiplai_core import StructuredAnswerError, parse_answer, reformat_prompt
from multiplai_core.structured import REFORMAT_MAX_ANSWER_CHARS


class Answer(BaseModel):
    value: int


class Topic(BaseModel):
    topic: str
    items: list[str] = []


def test_a_json_answer_parses_even_inside_prose_and_a_fence():
    text = 'Here is my answer:\n```json\n{"value": 3}\n```\nThanks.'
    assert parse_answer(text, Answer).value == 3


def test_a_prose_answer_is_reported_as_having_no_json():
    prose = '## Tests review\n\n| tier | covered |\n`{tier: 1, name: x}` then ["merge"]'
    with pytest.raises(
        StructuredAnswerError,
        match="contains no JSON object with the field.s. value; it starts: '## Tests",
    ):
        parse_answer(prose, Answer)
    with pytest.raises(
        ValueError
    ) as caught:  # JSON with the field keeps the parser's own error
        parse_answer('{"value": "x"}', Answer)
    assert "no JSON object" not in str(caught.value)
    assert not isinstance(caught.value, StructuredAnswerError)
    assert isinstance(caught.value, ValidationError)


def test_the_no_json_error_is_a_value_error_so_existing_retry_catches_still_fire():
    """The plugins re-ask on `except (ValidationError, ValueError)`; that must keep catching it."""
    with pytest.raises(ValueError):
        parse_answer("no json here", Answer)
    assert issubclass(StructuredAnswerError, ValueError)


def test_the_no_json_error_names_every_field_and_cuts_the_start_to_120_chars():
    text = "word " * 100
    with pytest.raises(StructuredAnswerError) as caught:
        parse_answer(text, Topic)
    msg = str(caught.value)
    assert "topic, items" in msg
    start = msg.split("it starts: ", 1)[1]
    assert len(ast.literal_eval(start)) == 120  # repr of the first 120 normalised chars


def test_an_answer_with_no_braces_but_a_field_name_keeps_the_parser_error():
    with pytest.raises(ValueError) as caught:
        parse_answer('"value" is 3', Answer)
    assert not isinstance(caught.value, StructuredAnswerError)


def test_reformat_prompt_quotes_the_error_the_schema_and_the_answer():
    prompt = reformat_prompt("the value is three", Answer, ValueError("bad thing"))
    assert "A program rejected its answer: bad thing" in prompt
    assert json.dumps(Answer.model_json_schema(), indent=1) in prompt
    assert "<answer>\nthe value is three\n</answer>" in prompt
    assert prompt.endswith("Answer with ONE JSON object and nothing after it.")


def test_reformat_prompt_cuts_a_long_answer_and_a_long_error():
    long_answer = "x" * (REFORMAT_MAX_ANSWER_CHARS + 50)
    prompt = reformat_prompt(long_answer, Answer, ValueError("e" * 5000))
    assert "x" * REFORMAT_MAX_ANSWER_CHARS + "\n[... answer cut ...]" in prompt
    assert "x" * (REFORMAT_MAX_ANSWER_CHARS + 1) not in prompt
    assert "e" * 2000 in prompt and "e" * 2001 not in prompt
