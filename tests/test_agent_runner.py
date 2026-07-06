"""Tests for the single SDK agent runner (multiplai_core/agent_runner.py)."""

import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from multiplai_core.agent_runner import (
    MAX_PROMPT_BYTES,
    AgentRunError,
    AgentRunResult,
    AgentRunTimeout,
    AgentUsage,
    run_agent,
)


# ---------------------------------------------------------------------------
# Fake SDK message/block types (real classes so isinstance() works)
# ---------------------------------------------------------------------------


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeToolUseBlock:
    def __init__(self, name: str, input: dict) -> None:
        self.name = name
        self.input = input


class _FakeAssistantMessage:
    def __init__(self, blocks: list) -> None:
        self.content = blocks


class _FakeResultMessage:
    def __init__(self, usage: dict | None = None, cost: float = 0.0) -> None:
        self.usage = usage or {}
        self.total_cost_usd = cost


def _make_mock_sdk(
    messages: list | None = None,
    *,
    fail: Exception | None = None,
    stderr_lines: list[str] | None = None,
    with_extras: bool = True,
) -> MagicMock:
    """Build a mock ``claude_agent_sdk`` module.

    ``with_extras=False`` omits ToolUseBlock/ResultMessage to simulate an old
    SDK — the runner must degrade gracefully.
    """
    if messages is None:
        messages = [_FakeAssistantMessage([_FakeTextBlock("default text")])]

    mock = MagicMock()
    mock.AssistantMessage = _FakeAssistantMessage
    mock.TextBlock = _FakeTextBlock
    if with_extras:
        mock.ToolUseBlock = _FakeToolUseBlock
        mock.ResultMessage = _FakeResultMessage

    def _options_ctor(**kwargs):
        stderr_cb = kwargs.get("stderr")
        if stderr_cb and stderr_lines:
            for line in stderr_lines:
                stderr_cb(line)
        opts = MagicMock()
        for k, v in kwargs.items():
            setattr(opts, k, v)
        return opts

    mock.ClaudeAgentOptions = MagicMock(side_effect=_options_ctor)

    async def _agen(prompt, options):
        if fail is not None:
            raise fail
        for m in messages:
            yield m

    mock.query = MagicMock(side_effect=_agen)
    return mock


def _run(coro):
    return asyncio.run(coro)


class TestBasicRuns:
    def test_no_tools_text_run(self):
        mock_sdk = _make_mock_sdk(
            [_FakeAssistantMessage([_FakeTextBlock("hello "), _FakeTextBlock("world")])]
        )
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            result = _run(run_agent("hi"))
        assert isinstance(result, AgentRunResult)
        assert result.text == "hello world"
        assert result.turns == 1
        assert result.files_changed == []
        assert result.stderr_tail == ""

    def test_turns_counts_assistant_messages(self):
        mock_sdk = _make_mock_sdk(
            [
                _FakeAssistantMessage([_FakeTextBlock("a")]),
                _FakeAssistantMessage([_FakeTextBlock("b")]),
                _FakeResultMessage(),
            ]
        )
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            result = _run(run_agent("hi", max_turns=5))
        assert result.turns == 2
        assert result.text == "ab"

    def test_files_changed_from_write_edit_tool_use(self):
        mock_sdk = _make_mock_sdk(
            [
                _FakeAssistantMessage(
                    [
                        _FakeToolUseBlock("Write", {"file_path": "/a.py"}),
                        _FakeToolUseBlock("Read", {"file_path": "/ignored.py"}),
                        _FakeToolUseBlock("Edit", {"file_path": "/b.py"}),
                        _FakeToolUseBlock("Edit", {"file_path": "/a.py"}),  # dedup
                    ]
                ),
                _FakeAssistantMessage([_FakeTextBlock("done")]),
            ]
        )
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            result = _run(run_agent("go", allowed_tools=["Write", "Edit"], max_turns=10))
        assert result.files_changed == ["/a.py", "/b.py"]
        assert result.text == "done"

    def test_usage_captured_from_result_message(self):
        mock_sdk = _make_mock_sdk(
            [
                _FakeAssistantMessage([_FakeTextBlock("ok")]),
                _FakeResultMessage(
                    usage={
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "cache_creation_input_tokens": 5,
                        "cache_read_input_tokens": 7,
                    },
                    cost=0.03,
                ),
            ]
        )
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            result = _run(run_agent("hi"))
        assert result.usage == AgentUsage(
            input_tokens=100,
            output_tokens=20,
            cache_creation_tokens=5,
            cache_read_tokens=7,
            cost_usd=0.03,
        )

    def test_old_sdk_without_extras_still_works(self):
        """An SDK missing ToolUseBlock/ResultMessage (MagicMock attrs are not
        types either) must not crash isinstance checks."""
        mock_sdk = _make_mock_sdk(with_extras=False)
        # MagicMock auto-attrs: getattr returns a MagicMock instance, not a type
        assert not isinstance(mock_sdk.ToolUseBlock, type)
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            result = _run(run_agent("hi"))
        assert result.text == "default text"
        assert result.usage == AgentUsage()

    def test_sdk_missing_raises_agent_run_error(self):
        with patch.dict(sys.modules, {"claude_agent_sdk": None}):
            with pytest.raises(AgentRunError, match="not available"):
                _run(run_agent("hi"))


class TestOptionsIsolation:
    def test_isolation_bundle_always_applied(self):
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("hi"))
        opts = mock_sdk.query.call_args.kwargs["options"]
        assert opts.permission_mode == "bypassPermissions"
        assert opts.setting_sources == []
        assert opts.extra_args["setting-sources"] == ""
        assert "debug-to-stderr" in opts.extra_args
        assert "strict-mcp-config" in opts.extra_args
        assert opts.env["_HOOK_CHILD_SESSION"] == "1"
        assert opts.cwd.endswith("hook-sessions")

    def test_caller_env_merged_over_baseline(self):
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("hi", env={"FOO": "bar"}))
        opts = mock_sdk.query.call_args.kwargs["options"]
        assert opts.env["FOO"] == "bar"
        assert opts.env["_HOOK_CHILD_SESSION"] == "1"

    def test_cwd_override(self, tmp_path):
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("hi", cwd=tmp_path))
        opts = mock_sdk.query.call_args.kwargs["options"]
        assert opts.cwd == str(tmp_path)

    def test_effort_omitted_when_none(self):
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("hi"))
        assert "effort" not in mock_sdk.ClaudeAgentOptions.call_args.kwargs

    def test_effort_forwarded_when_set(self):
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("hi", effort="low"))
        assert mock_sdk.ClaudeAgentOptions.call_args.kwargs["effort"] == "low"

    def test_disallowed_tools_omitted_by_default_and_forwarded_when_set(self):
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("hi"))
            assert (
                "disallowed_tools"
                not in mock_sdk.ClaudeAgentOptions.call_args.kwargs
            )
            _run(run_agent("hi", disallowed_tools=["Bash"]))
            assert mock_sdk.ClaudeAgentOptions.call_args.kwargs[
                "disallowed_tools"
            ] == ["Bash"]

    def test_system_prompt_and_model_forwarded(self):
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("hi", system_prompt="sys", model="claude-x"))
        opts = mock_sdk.query.call_args.kwargs["options"]
        assert opts.system_prompt == "sys"
        assert opts.model == "claude-x"


class TestUnknownMessageSkip:
    def test_skips_unknown_message_types(self):
        class _UnknownThenGood:
            def __init__(self):
                self._i = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                self._i += 1
                if self._i == 1:
                    return _FakeAssistantMessage([_FakeTextBlock("hello ")])
                if self._i == 2:
                    raise RuntimeError("Unknown message type: rate_limit_event")
                if self._i == 3:
                    return _FakeAssistantMessage([_FakeTextBlock("world")])
                raise StopAsyncIteration

        mock = MagicMock()
        mock.AssistantMessage = _FakeAssistantMessage
        mock.TextBlock = _FakeTextBlock
        mock.ClaudeAgentOptions = MagicMock(side_effect=lambda **kw: MagicMock())
        mock.query = MagicMock(side_effect=lambda prompt, options: _UnknownThenGood())

        with patch.dict(sys.modules, {"claude_agent_sdk": mock}):
            result = _run(run_agent("hi", max_turns=5))
        assert result.text == "hello world"


class TestFailures:
    def test_error_wrapped_with_stderr_tail_and_partial(self):
        class _PartialThenFail:
            def __init__(self):
                self._i = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                self._i += 1
                if self._i == 1:
                    return _FakeAssistantMessage([_FakeTextBlock("partial out")])
                raise RuntimeError("boom")

        mock = MagicMock()
        mock.AssistantMessage = _FakeAssistantMessage
        mock.TextBlock = _FakeTextBlock

        def _options_ctor(**kwargs):
            kwargs["stderr"]("[ERROR] API error: 500")
            return MagicMock(**kwargs)

        mock.ClaudeAgentOptions = MagicMock(side_effect=_options_ctor)
        mock.query = MagicMock(side_effect=lambda prompt, options: _PartialThenFail())

        with patch.dict(sys.modules, {"claude_agent_sdk": mock}):
            with pytest.raises(AgentRunError) as exc:
                _run(run_agent("hi"))
        e = exc.value
        assert not isinstance(e, AgentRunTimeout)
        assert "boom" in str(e)
        assert e.reason == "failed: boom"
        assert e.attempts == 1
        assert "API error: 500" in e.stderr_tail
        assert e.partial is not None
        assert e.partial.text == "partial out"
        assert e.partial.turns == 1

    def test_stderr_tail_prefers_error_lines(self):
        mock_sdk = _make_mock_sdk(
            fail=RuntimeError("Command failed with exit code 1"),
            stderr_lines=[
                "[DEBUG] noise",
                "[ERROR] 429 rate_limit_error",
                "[DEBUG] more noise",
            ],
        )
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            with pytest.raises(AgentRunError) as exc:
                _run(run_agent("hi"))
        assert "429 rate_limit_error" in exc.value.stderr_tail
        assert "[DEBUG]" not in exc.value.stderr_tail

    def test_retry_recovers(self):
        calls = {"n": 0}

        async def _agen(prompt, options):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("Command failed with exit code 1")
            yield _FakeAssistantMessage([_FakeTextBlock("recovered")])

        mock = MagicMock()
        mock.AssistantMessage = _FakeAssistantMessage
        mock.TextBlock = _FakeTextBlock
        mock.ClaudeAgentOptions = MagicMock(side_effect=lambda **kw: MagicMock())
        mock.query = MagicMock(side_effect=_agen)

        with patch.dict(sys.modules, {"claude_agent_sdk": mock}):
            result = _run(run_agent("hi", max_attempts=2, retry_backoff_s=0))
        assert result.text == "recovered"
        assert calls["n"] == 2

    def test_exhausted_attempts_reports_count(self):
        mock_sdk = _make_mock_sdk(fail=RuntimeError("persistent"))
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            with pytest.raises(AgentRunError) as exc:
                _run(run_agent("hi", max_attempts=2, retry_backoff_s=0))
        assert exc.value.attempts == 2
        assert "after 2 attempt(s)" in str(exc.value)


class TestTimeout:
    @staticmethod
    def _hanging_sdk() -> MagicMock:
        mock = MagicMock()
        mock.AssistantMessage = _FakeAssistantMessage
        mock.TextBlock = _FakeTextBlock
        mock.ClaudeAgentOptions = MagicMock(side_effect=lambda **kw: MagicMock())
        closed = {"cancelled": False}

        async def _agen(prompt, options):
            try:
                await asyncio.sleep(10)
                yield _FakeAssistantMessage([_FakeTextBlock("never")])
            except asyncio.CancelledError:
                closed["cancelled"] = True
                raise

        mock.query = MagicMock(side_effect=_agen)
        mock._closed = closed
        return mock

    def test_timeout_raises_agent_run_timeout(self):
        mock_sdk = self._hanging_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            with pytest.raises(AgentRunTimeout) as exc:
                _run(run_agent("hi", timeout_s=0.05))
        assert "timed out after 0s" in str(exc.value) or "timed out" in str(exc.value)
        assert exc.value.reason.startswith("timed out")

    def test_timeout_tears_down_generator(self):
        mock_sdk = self._hanging_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            with pytest.raises(AgentRunTimeout):
                _run(run_agent("hi", timeout_s=0.05))
        assert mock_sdk._closed["cancelled"] is True

    def test_timeout_then_recovery_within_retry_budget(self):
        state = {"calls": 0}
        mock = MagicMock()
        mock.AssistantMessage = _FakeAssistantMessage
        mock.TextBlock = _FakeTextBlock
        mock.ClaudeAgentOptions = MagicMock(side_effect=lambda **kw: MagicMock())

        async def _agen(prompt, options):
            state["calls"] += 1
            if state["calls"] == 1:
                await asyncio.sleep(10)
            yield _FakeAssistantMessage([_FakeTextBlock("recovered")])

        mock.query = MagicMock(side_effect=_agen)

        with patch.dict(sys.modules, {"claude_agent_sdk": mock}):
            result = _run(
                run_agent("hi", timeout_s=0.05, max_attempts=2, retry_backoff_s=0)
            )
        assert result.text == "recovered"
        assert state["calls"] == 2


class TestPromptFileFallback:
    def test_big_prompt_written_to_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        import tempfile as _tf
        _tf.tempdir = None  # force TMPDIR re-resolution
        big_prompt = "x" * (MAX_PROMPT_BYTES + 1)
        seen = {}

        async def _agen(prompt, options):
            seen["prompt"] = prompt
            seen["options"] = options
            # the prompt file must exist while the agent runs
            marker = prompt.split("Read the file ")[1].split(" using")[0]
            seen["file_existed"] = os.path.exists(marker)
            seen["file"] = marker
            yield _FakeAssistantMessage([_FakeTextBlock("done")])

        mock = _make_mock_sdk()
        mock.query = MagicMock(side_effect=_agen)

        with patch.dict(sys.modules, {"claude_agent_sdk": mock}):
            result = _run(run_agent(big_prompt, max_turns=1))

        assert result.text == "done"
        assert "Read the file" in seen["prompt"]
        assert seen["file_existed"] is True
        assert not os.path.exists(seen["file"])  # cleaned up afterwards
        opts_kwargs = mock.ClaudeAgentOptions.call_args.kwargs
        assert "Read" in opts_kwargs["allowed_tools"]
        assert opts_kwargs["max_turns"] >= 3
        assert (
            opts_kwargs["env"]["CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS"] == "100000"
        )
        _tf.tempdir = None

    def test_fallback_disabled_leaves_prompt_inline(self):
        big_prompt = "x" * (MAX_PROMPT_BYTES + 1)
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent(big_prompt, prompt_file_fallback=False))
        assert mock_sdk.query.call_args.kwargs["prompt"] == big_prompt
        opts_kwargs = mock_sdk.ClaudeAgentOptions.call_args.kwargs
        assert "Read" not in opts_kwargs["allowed_tools"]

    def test_small_prompt_untouched(self):
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("small"))
        assert mock_sdk.query.call_args.kwargs["prompt"] == "small"


class TestCostLedgerTap:
    """component= writes the run's usage to the cost ledger; ledger failures
    never break the run."""

    @pytest.fixture(autouse=True)
    def _tmp_workspace(self, monkeypatch, tmp_path, reset_paths_cache):
        monkeypatch.setenv("WORKSPACE", str(tmp_path))

    def _result_message(self):
        return _FakeResultMessage(
            usage={
                "input_tokens": 1000,
                "output_tokens": 500,
                "cache_creation_input_tokens": 200,
                "cache_read_input_tokens": 3000,
            },
            cost=0.123,
        )

    def test_component_writes_ledger_record(self):
        from multiplai_core.costing import iter_ledger

        mock_sdk = _make_mock_sdk([
            _FakeAssistantMessage([_FakeTextBlock("done")]),
            self._result_message(),
        ])
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("p", model="claude-opus-4-8", component="buildme"))
        records = list(iter_ledger())
        assert len(records) == 1
        rec = records[0]
        assert rec["source"] == "sdk"
        assert rec["component"] == "buildme"
        assert rec["cost_usd"] == pytest.approx(0.123)  # SDK cost wins
        assert rec["tokens"] == {"in": 1000, "out": 500, "cw5m": 200, "cw1h": 0, "cr": 3000}

    def test_no_component_writes_nothing(self):
        from multiplai_core.costing import iter_ledger

        mock_sdk = _make_mock_sdk([self._result_message()])
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("p"))
        assert list(iter_ledger()) == []

    def test_zero_usage_writes_nothing(self):
        from multiplai_core.costing import iter_ledger

        mock_sdk = _make_mock_sdk([_FakeAssistantMessage([_FakeTextBlock("x")])])
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("p", component="buildme"))
        assert list(iter_ledger()) == []

    def test_ledger_failure_never_breaks_the_run(self, monkeypatch):
        import multiplai_core.costing as costing

        def _boom(records):
            raise OSError("disk full")

        monkeypatch.setattr(costing, "append_records", _boom)
        mock_sdk = _make_mock_sdk([
            _FakeAssistantMessage([_FakeTextBlock("done")]),
            self._result_message(),
        ])
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            result = _run(run_agent("p", component="buildme"))
        assert result.text == "done"

    def test_failed_run_records_partial_usage(self):
        from multiplai_core.costing import iter_ledger

        # Result message arrives, then the stream dies — partial usage exists.
        async def _agen(prompt, options):
            yield _FakeAssistantMessage([_FakeTextBlock("some")])
            yield self._result_message()
            raise RuntimeError("stream died")

        mock_sdk = _make_mock_sdk()
        mock_sdk.query = MagicMock(side_effect=_agen)
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            with pytest.raises(AgentRunError):
                _run(run_agent("p", component="deep-research"))
        records = list(iter_ledger())
        assert len(records) == 1
        assert records[0]["component"] == "deep-research"
