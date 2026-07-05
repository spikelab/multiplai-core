"""Tests for multiplai_core.text.extract_json."""

import json

import pytest

from multiplai_core.text import extract_json


def test_plain_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_plain_array():
    assert extract_json("[1, 2, 3]") == [1, 2, 3]


def test_fenced_json():
    text = "```json\n" + json.dumps({"x": [1, 2]}) + "\n```"
    assert extract_json(text) == {"x": [1, 2]}


def test_fenced_no_lang():
    text = "```\n{\"y\": true}\n```"
    assert extract_json(text) == {"y": True}


def test_surrounding_prose():
    text = 'Here is the result:\n{"ok": true}\nThanks!'
    assert extract_json(text) == {"ok": True}


def test_brackets_inside_strings_do_not_confuse_balancer():
    text = '{"note": "a } and a ] inside", "n": 1}'
    assert extract_json(text) == {"note": "a } and a ] inside", "n": 1}


def test_escaped_quote_in_string():
    text = '{"q": "he said \\"hi\\"", "n": 2}'
    assert extract_json(text) == {"q": 'he said "hi"', "n": 2}


def test_empty_raises():
    with pytest.raises(ValueError):
        extract_json("")


def test_no_json_raises():
    with pytest.raises(ValueError):
        extract_json("just prose, no json here")


def test_unbalanced_raises():
    with pytest.raises(ValueError):
        extract_json('{"a": 1')
