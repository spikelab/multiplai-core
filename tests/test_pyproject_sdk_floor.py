"""Regression guard for the [sdk] extra's claude-agent-sdk version constraint.

This project exists because a minor claude-agent-sdk bump (0.1 -> 0.2) shipped a
breaking result-message parse change. The [sdk] extra must therefore:
  - floor at >=0.2.139 (below that, ClaudeAgentOptions may lack `thinking`,
    which run_agent forwards unconditionally; below 0.2.116, terminal result
    messages also misparse), and
  - cap below <0.3 (a fresh 0.3.x/1.0 resolve could re-break the same class of
    failure for consumers that don't vendor our lock).

Parsed from pyproject.toml with tomllib — no network, no install required.
"""

import tomllib  # requires Python 3.11+, matching requires-python
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

    # Floor: pre-0.2.139 may lack ClaudeAgentOptions.thinking, which run_agent
    # forwards unconditionally — a TypeError on every call that sets it.
    assert not spec.contains("0.2.138"), "sdk floor must exclude 0.2.138"
    assert spec.contains("0.2.139"), "sdk floor must admit 0.2.139"

    # The older reason still holds underneath: pre-0.2.116 misparses the
    # terminal result message, and the whole 0.1.x line is broken.
    assert not spec.contains("0.2.115"), "sdk floor must exclude 0.2.115"
    assert not spec.contains("0.1.0"), "sdk floor must exclude the 0.1.x line"

    # Cap: a future minor could reintroduce the breaking parse change.
    assert not spec.contains("0.3.0"), "sdk must cap below 0.3"
    assert not spec.contains("1.0.0"), "sdk must cap below 0.3 (excludes 1.0)"
