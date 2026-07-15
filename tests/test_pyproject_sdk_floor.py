"""Regression guard for the [sdk] extra's claude-agent-sdk version constraint.

This project exists because a minor claude-agent-sdk bump (0.1 -> 0.2) shipped a
breaking result-message parse change. The [sdk] extra must therefore:
  - floor at >=0.2.116 (below that, terminal result messages misparse), and
  - cap below <0.3 (a fresh 0.3.x/1.0 resolve could re-break the same class of
    failure for consumers that don't vendor our lock).

Parsed from pyproject.toml with tomllib — no network, no install required.
"""

import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _sdk_requirement() -> Requirement:
    data = tomllib.loads(_PYPROJECT.read_text())
    sdk_extra = data["project"]["optional-dependencies"]["sdk"]
    for spec in sdk_extra:
        req = Requirement(spec)
        if req.name == "claude-agent-sdk":
            return req
    raise AssertionError("claude-agent-sdk not found in [sdk] optional-dependencies")


def test_sdk_floor_and_cap():
    req = _sdk_requirement()
    spec = req.specifier

    # Floor: pre-0.2.116 misparses the terminal result message.
    assert not spec.contains("0.2.115"), "sdk floor must exclude 0.2.115"
    assert not spec.contains("0.1.0"), "sdk floor must exclude the 0.1.x line"
    assert spec.contains("0.2.116"), "sdk floor must admit 0.2.116"

    # Cap: a future minor could reintroduce the breaking parse change.
    assert not spec.contains("0.3.0"), "sdk must cap below 0.3"
    assert not spec.contains("1.0.0"), "sdk must cap below 0.3 (excludes 1.0)"

    # The resolved version we ship in uv.lock must satisfy the constraint.
    assert spec.contains("0.2.119"), "current resolved version must satisfy the spec"


if sys.version_info < (3, 11):  # pragma: no cover - project requires >=3.11
    raise RuntimeError("tomllib requires Python 3.11+")
