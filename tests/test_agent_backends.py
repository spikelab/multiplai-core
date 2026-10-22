"""Tests for run_agent's backend choice (register_agent_backend and dispatch)."""

import asyncio
import inspect
import subprocess
import sys
from unittest.mock import patch

import pytest
from _fakes import _make_mock_sdk

from multiplai_core import agent_runner
from multiplai_core.agent_runner import (
    AgentRunError,
    AgentRunResult,
    AgentUsage,
    UnknownAgentBackendError,
    register_agent_backend,
    registered_agent_backends,
    run_agent,
)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    monkeypatch.setattr(agent_runner, "_AGENT_BACKENDS", {})


def _result(text="fake"):
    return AgentRunResult(
        text=text, turns=1, usage=AgentUsage(), files_changed=[], stderr_tail=""
    )


class TestAnthropicPath:
    @pytest.mark.parametrize(
        "spec, expected",
        [
            ("anthropic:claude-opus-4-8", "claude-opus-4-8"),
            ("claude:claude-fable-5-1", "claude-fable-5-1"),
            ("claude-opus-4-8", "claude-opus-4-8"),
            ("opus", "opus"),
            # Bedrock IDs and ARNs carry colons but are model IDs, not specs.
            (
                "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "anthropic.claude-3-5-sonnet-20240620-v1:0",
            ),
            (
                "arn:aws:bedrock:us-east-1:123:inference-profile/x",
                "arn:aws:bedrock:us-east-1:123:inference-profile/x",
            ),
        ],
    )
    def test_anthropic_spec_takes_the_sdk_path(self, spec, expected):
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            result = _run(run_agent("hi", model=spec))
        assert result.text == "default text"
        assert mock_sdk.query.call_args.kwargs["options"].model == expected

    def test_model_none_takes_the_sdk_path(self):
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("hi"))
        assert mock_sdk.query.called

    def test_anthropic_cannot_be_replaced(self):
        for name in ("anthropic", "claude", "Claude"):
            with pytest.raises(ValueError):
                register_agent_backend(name, lambda *a, **k: None)


class TestRegisteredBackend:
    def test_fake_backend_gets_every_keyword(self):
        seen = {}

        async def fake(prompt, **kwargs):
            seen["prompt"] = prompt
            seen.update(kwargs)
            return _result("from fake")

        register_agent_backend("fake", fake)
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            result = _run(
                run_agent(
                    "p",
                    system_prompt="s",
                    allowed_tools=["Read"],
                    max_turns=3,
                    model="fake:m@x",
                    effort="low",
                    cwd="/tmp",
                    label="L",
                    component="c",
                )
            )
        assert result.text == "from fake"
        assert not mock_sdk.query.called
        expected = set(inspect.signature(run_agent).parameters) - {"prompt"}
        assert set(seen) - {"prompt"} == expected
        assert seen["prompt"] == "p"
        assert seen["model"] == "fake:m@x"
        assert seen["allowed_tools"] == ["Read"]
        assert seen["effort"] == "low"
        assert seen["component"] == "c"

    def test_backend_gets_the_spec_as_dispatch_read_it(self):
        seen = {}

        async def fake(prompt, **kwargs):
            seen.update(kwargs)
            return _result("from fake")

        register_agent_backend("fake", fake)
        _run(run_agent("p", model="  FAKE: m@x "))
        assert seen["model"] == "fake:m@x"

    def test_registered_list_includes_builtins_and_new(self):
        register_agent_backend("Fake", lambda *a, **k: None)
        names = registered_agent_backends()
        assert {"anthropic", "pi", "fake"} <= set(names)
        assert list(names) == sorted(names)


class TestUnknownBackend:
    def test_unknown_provider_raises(self):
        with pytest.raises(UnknownAgentBackendError) as exc:
            _run(run_agent("p", model="nosuch:model"))
        assert isinstance(exc.value, AgentRunError)
        assert "nosuch" in str(exc.value)
        assert "anthropic" in str(exc.value) and "pi" in str(exc.value)


class TestLazyPi:
    def test_importing_agent_runner_does_not_import_pi(self):
        code = (
            "import multiplai_core.agent_runner, sys; "
            "assert 'multiplai_core.pi' not in sys.modules, "
            "[m for m in sys.modules if m.startswith('multiplai_core.pi')]"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_pi_spec_loads_and_registers_the_pi_backend(self, monkeypatch, tmp_path):
        # Empty registry, so dispatch must import multiplai_core.pi.backend,
        # which registers itself. With no pi binary the run then fails with the
        # install hint -- proof it reached the Pi backend and did not fall
        # back to Claude.
        monkeypatch.delitem(sys.modules, "multiplai_core.pi.backend", raising=False)
        monkeypatch.setenv("MULTIPLAI_PI_BIN", str(tmp_path / "no-such-pi"))
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            with pytest.raises(AgentRunError) as exc:
                _run(
                    run_agent(
                        "p",
                        model="pi:deepseek/deepseek-v4-flash@fireworks",
                        cwd=tmp_path,
                    )
                )
        assert "pi" in agent_runner._AGENT_BACKENDS
        assert "MULTIPLAI_PI_BIN" in str(exc.value)
        assert not mock_sdk.query.called
