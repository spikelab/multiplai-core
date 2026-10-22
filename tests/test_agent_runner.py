"""Tests for the single SDK agent runner (multiplai_core/agent_runner.py)."""

import asyncio
import logging
import os
import re
import sys
from unittest.mock import MagicMock, patch

import pytest
from _fakes import (
    _FakeAssistantMessage,
    _FakeResultMessage,
    _FakeTextBlock,
    _FakeToolUseBlock,
    _make_mock_sdk,
)

from multiplai_core.agent_runner import (
    MAX_PROMPT_BYTES,
    TOOL_UNIVERSE,
    AgentRunError,
    AgentRunResult,
    AgentRunTimeout,
    AgentUsage,
    ToolCall,
    UsageLimitError,
    deny_list,
    is_usage_limit,
    path_allowed,
    run_agent,
)
from multiplai_core.env import CURRENT_MODEL


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

    def test_tool_calls_hold_every_tool_use_in_order(self):
        read_input = {"file_path": "/repo/a.py", "offset": 10, "limit": 20}
        grep_input = {"pattern": "def run", "path": "/repo", "glob": "*.py"}
        mock_sdk = _make_mock_sdk(
            [
                _FakeAssistantMessage([_FakeToolUseBlock("Read", read_input)]),
                _FakeAssistantMessage(
                    [
                        _FakeToolUseBlock("Grep", grep_input),
                        _FakeTextBlock("done"),
                    ]
                ),
            ]
        )
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            result = _run(run_agent("go", allowed_tools=["Read", "Grep"], max_turns=10))
        assert result.tool_calls == (
            ToolCall("Read", {"file_path": "/repo/a.py", "offset": 10, "limit": 20}),
            ToolCall("Grep", {"pattern": "def run", "path": "/repo", "glob": "*.py"}),
        )
        # A copy, not a reference to the SDK object's input.
        read_input["offset"] = 99
        assert result.tool_calls[0].input["offset"] == 10
        assert result.files_changed == []

    def test_tool_calls_include_write_and_edit(self):
        mock_sdk = _make_mock_sdk(
            [
                _FakeAssistantMessage(
                    [
                        _FakeToolUseBlock("Write", {"file_path": "/a.py"}),
                        _FakeToolUseBlock("Edit", {"file_path": "/a.py"}),
                    ]
                ),
            ]
        )
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            result = _run(run_agent("go", allowed_tools=["Write", "Edit"], max_turns=10))
        assert [c.name for c in result.tool_calls] == ["Write", "Edit"]
        assert result.files_changed == ["/a.py"]

    def test_result_built_without_tool_calls_still_works(self):
        result = AgentRunResult(
            text="t", turns=1, usage=AgentUsage(), files_changed=[], stderr_tail=""
        )
        assert result.tool_calls == ()

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

    def test_caller_env_cannot_clear_the_child_session_guard(self):
        """The fork-bomb guard is merged last: a caller's env must not be
        able to blank or override ``_HOOK_CHILD_SESSION``."""
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("hi", env={"_HOOK_CHILD_SESSION": ""}))
        opts = mock_sdk.query.call_args.kwargs["options"]
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

    def test_thinking_omitted_when_none(self):
        """An old SDK has no `thinking` option — passing nothing must not add it."""
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("hi"))
        assert "thinking" not in mock_sdk.ClaudeAgentOptions.call_args.kwargs

    def test_thinking_forwarded_verbatim_when_set(self):
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("hi", thinking={"type": "disabled"}))
        kwargs = mock_sdk.ClaudeAgentOptions.call_args.kwargs
        assert kwargs["thinking"] == {"type": "disabled"}

    def test_thinking_and_effort_are_independent(self):
        """Both knobs travel together; neither implies nor suppresses the other."""
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("hi", effort="low", thinking={"type": "disabled"}))
        kwargs = mock_sdk.ClaudeAgentOptions.call_args.kwargs
        assert kwargs["effort"] == "low"
        assert kwargs["thinking"] == {"type": "disabled"}

    def test_default_denies_the_whole_tool_universe(self):
        """No tool args at all must mean no tools — the fail-closed default."""
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("hi"))
        denied = mock_sdk.ClaudeAgentOptions.call_args.kwargs["disallowed_tools"]
        assert denied == list(TOOL_UNIVERSE)
        for tool in ("Bash", "Read", "Write", "WebFetch", "ToolSearch"):
            assert tool in denied

    def test_allow_list_denies_the_complement(self):
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("hi", allowed_tools=["WebFetch"]))
        denied = mock_sdk.ClaudeAgentOptions.call_args.kwargs["disallowed_tools"]
        assert "WebFetch" not in denied
        for tool in ("Bash", "Read", "Write", "WebSearch"):
            assert tool in denied

    def test_explicit_empty_deny_list_is_an_opt_out(self):
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("hi", disallowed_tools=[]))
        assert mock_sdk.ClaudeAgentOptions.call_args.kwargs[
            "disallowed_tools"
        ] == []

    def test_explicit_deny_list_forwarded_verbatim(self):
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("hi", disallowed_tools=["Bash"]))
        assert mock_sdk.ClaudeAgentOptions.call_args.kwargs[
            "disallowed_tools"
        ] == ["Bash"]

    def test_deny_list_helper_complements(self):
        assert deny_list(None) == list(TOOL_UNIVERSE)
        assert deny_list([]) == list(TOOL_UNIVERSE)
        assert "Read" not in deny_list(["Read"])
        assert deny_list(["NotATool"]) == list(TOOL_UNIVERSE)

    def test_base_tool_set_is_the_allow_list(self):
        """`tools` is the real boundary: it sets which tools exist at all.

        Unlike the deny-list it needs no enumeration to be correct, so this is
        what holds when TOOL_UNIVERSE goes stale against a newer CLI.
        """
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("hi", allowed_tools=["WebFetch"]))
        assert mock_sdk.ClaudeAgentOptions.call_args.kwargs["tools"] == [
            "WebFetch"
        ]

    def test_no_tools_call_gets_an_empty_base_set(self):
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("hi"))
        assert mock_sdk.ClaudeAgentOptions.call_args.kwargs["tools"] == []

    def test_explicit_deny_list_opt_out_still_bounds_the_base_set(self):
        """`disallowed_tools=[]` opts out of layer two, never out of layer one.

        Otherwise the documented escape hatch would quietly restore the full
        built-in tool set — the exact fail-open this change exists to remove.
        """
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("hi", disallowed_tools=[]))
        kwargs = mock_sdk.ClaudeAgentOptions.call_args.kwargs
        assert kwargs["disallowed_tools"] == []
        assert kwargs["tools"] == []

    def test_universe_names_the_tools_that_carry_the_risk(self):
        """Pin specific names, not the list against itself.

        A test comparing the deny-list to TOOL_UNIVERSE passes no matter what
        is missing from it. These are the capabilities the fix exists to
        remove — execution, egress, and deferred execution — so removing any
        of them should break a test.
        """
        for tool in (
            "Bash", "REPL",                      # execution
            "WebFetch", "Artifact", "SendMessage",  # egress
            "TaskCreate", "CronCreate", "Workflow",  # deferred execution
            "Read", "Write", "Edit",             # local files
            "ToolSearch", "Skill",               # loading tools back in
        ):
            assert tool in TOOL_UNIVERSE, tool

    def test_unknown_tool_name_is_logged(self, caplog):
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            with caplog.at_level(logging.WARNING):
                _run(run_agent("hi", allowed_tools=["Websearch"]))
        assert "Websearch" in caplog.text

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
    def test_options_signature_mismatch_raises_agent_run_error(self):
        """A TypeError from ClaudeAgentOptions (SDK signature mismatch) must
        surface as AgentRunError per the Raises contract, never escape raw."""
        mock_sdk = _make_mock_sdk()
        mock_sdk.ClaudeAgentOptions = MagicMock(
            side_effect=TypeError("unexpected keyword argument 'tools'")
        )
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            with pytest.raises(
                AgentRunError, match="unexpected keyword argument 'tools'"
            ) as exc:
                _run(run_agent("hi"))
        assert not isinstance(exc.value, AgentRunTimeout)
        assert isinstance(exc.value.__cause__, TypeError)

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


# The CLI's result text for a hit usage limit, as logged by run_agent on
# 2026-10-18 (session) and 2026-10-17 (weekly) in the multiplai runtime logs.
SESSION_LIMIT = (
    "Claude Code returned an error result: You've hit your session limit "
    "\u00b7 resets 9:40am (UTC) (exit code: 1)"
)
WEEKLY_LIMIT = (
    "Claude Code returned an error result: You've hit your weekly limit "
    "\u00b7 resets Oct 19, 4am (UTC) (exit code: 1)"
)


class _ResultError(Exception):
    """Stands in for claude_agent_sdk's ResultError."""


class TestUsageLimit:
    """A hit usage limit resets hours later: run_agent must not retry it."""

    def _counting_sdk(self, message: str):
        calls = {"n": 0}
        mock = _make_mock_sdk()

        async def _agen(prompt, options):
            calls["n"] += 1
            raise _ResultError(message)
            yield  # pragma: no cover — makes this an async generator

        mock.query = MagicMock(side_effect=_agen)
        return mock, calls

    @pytest.mark.parametrize("message,limit,resets", [
        (SESSION_LIMIT, "session", "9:40am (UTC)"),
        (WEEKLY_LIMIT, "weekly", "Oct 19, 4am (UTC)"),
    ])
    def test_limit_raises_usage_limit_error_without_retry(self, message, limit, resets):
        mock_sdk, calls = self._counting_sdk(message)
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            with pytest.raises(UsageLimitError) as exc:
                _run(run_agent("hi", max_attempts=3, retry_backoff_s=0))
        assert calls["n"] == 1
        assert exc.value.attempts == 1
        assert exc.value.limit == limit
        assert exc.value.resets == resets
        assert isinstance(exc.value, AgentRunError)  # old `except AgentRunError` still catches it
        assert "hit your " + limit + " limit" in str(exc.value)
        assert isinstance(exc.value.__cause__, _ResultError)

    @pytest.mark.parametrize("stderr", [
        "[ERROR] API rate_limit after retries: This request would exceed your account's rate limit.",
        "[DEBUG] tool result: You've hit your session limit \u00b7 resets 9:40am (UTC)",
    ])
    def test_stderr_text_never_makes_a_usage_limit(self, stderr):
        """A 429 the CLI retried is transient, and debug lines can echo any text: only the
        result text says a limit was hit."""
        mock_sdk = _make_mock_sdk(fail=RuntimeError("Command failed with exit code 1"), stderr_lines=[stderr])
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            with pytest.raises(AgentRunError) as exc:
                _run(run_agent("hi"))
        assert not isinstance(exc.value, UsageLimitError)
        assert not is_usage_limit(exc.value)

    def test_the_phrase_quoted_inside_a_result_is_not_a_usage_limit(self):
        """The CLI's message starts with the phrase (it classifies it with startsWith)."""
        mock_sdk, calls = self._counting_sdk(
            "Claude Code returned an error result: the finding says \"You've hit your session limit\" (exit code: 1)")
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            with pytest.raises(AgentRunError) as exc:
                _run(run_agent("hi", max_attempts=2, retry_backoff_s=0))
        assert not isinstance(exc.value, UsageLimitError) and calls["n"] == 2

    @pytest.mark.parametrize("message,limit,resets", [
        ("Claude Code returned an error result: You've hit your session limit \u00b7 resets 5pm "
         "\u00b7 progress saved (exit code: 1)", "session", "5pm"),
        ("Claude Code returned an error result: You've hit your Opus limit \u00b7 resets 9:40am (UTC) "
         "\u00b7 progress saved", "opus", "9:40am (UTC)"),
        ("Claude Code returned an error result: You've hit your team's shared budget. /model to switch models.",
         "team's shared", ""),
        ("Claude Code returned an error result: You've hit your session limit \u00b7 resets 5pm (exit code: 1)",
         "session", "5pm"),
        ("Claude Code returned an error result: You've hit your Opus 4.1 limit (exit code: 1)", "opus 4.1", ""),
        ("You've hit your limit (exit code: 1)", "", ""),
    ])
    def test_other_wordings_of_the_bundled_cli(self, message, limit, resets):
        """Shapes the CLI bundled with claude-agent-sdk 0.2.168 builds: `You've hit your ${name}${resets}${saved}`."""
        mock_sdk, _ = self._counting_sdk(message)
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            with pytest.raises(UsageLimitError) as exc:
                _run(run_agent("hi"))
        assert (exc.value.limit, exc.value.resets) == (limit, resets)

    def test_limit_logs_one_fail_line_with_the_stderr_and_no_retry_line(self, caplog):
        mock_sdk, _ = self._counting_sdk(SESSION_LIMIT)
        mock_sdk.ClaudeAgentOptions = _make_mock_sdk(stderr_lines=["[ERROR] cli: limit reached"]).ClaudeAgentOptions
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            with caplog.at_level(logging.WARNING, logger="multiplai_core.agent_runner"):
                with pytest.raises(UsageLimitError):
                    _run(run_agent("hi", max_attempts=3, retry_backoff_s=0, label="verify"))
        fails = [r.getMessage() for r in caplog.records if r.getMessage().startswith("FAIL")]
        assert len(fails) == 1 and "usage limit hit, not retried" in fails[0]
        assert "--- captured CLI stderr (errors) ---" in fails[0] and "cli: limit reached" in fails[0]
        assert "retrying" not in caplog.text

    def test_limit_records_partial_usage(self, tmp_workspace):
        from multiplai_core.costing import iter_ledger

        async def _agen(prompt, options):
            yield _FakeAssistantMessage([_FakeTextBlock("some")])
            yield _FakeResultMessage(usage={"input_tokens": 10, "output_tokens": 2})
            raise _ResultError(SESSION_LIMIT)

        mock_sdk = _make_mock_sdk()
        mock_sdk.query = MagicMock(side_effect=_agen)
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            with pytest.raises(UsageLimitError) as exc:
                _run(run_agent("p", component="review"))
        assert exc.value.partial is not None and exc.value.partial.text == "some"
        assert len(list(iter_ledger())) == 1


class TestIsUsageLimit:
    def test_true_for_a_wrapper_whose_cause_is_one(self):
        class Wrapper(Exception):
            pass

        try:
            try:
                raise UsageLimitError("run_agent [verify] failed", limit="session")
            except UsageLimitError as e:
                raise Wrapper("agent call failed") from e
        except Wrapper as w:
            assert is_usage_limit(w)

    def test_true_for_a_wrapper_raised_without_from(self):
        """Reachable only through __context__, as when a caller wraps without `from`."""
        try:
            try:
                raise UsageLimitError("run_agent [verify] failed", limit="session")
            except UsageLimitError:
                raise RuntimeError("verify step failed")
        except RuntimeError as w:
            assert w.__cause__ is None
            assert is_usage_limit(w)

    def test_false_when_the_context_is_suppressed(self):
        try:
            try:
                raise UsageLimitError("run_agent [verify] failed", limit="session")
            except UsageLimitError:
                raise RuntimeError("unrelated") from None
        except RuntimeError as w:
            assert not is_usage_limit(w)

    def test_stderr_holding_the_sdk_message_does_not_count_through_any_wrapper(self):
        """The phrase, with the SDK's own prefix, in a stderr line: neither the
        AgentRunError nor a wrapper whose str() includes that stderr counts."""
        from multiplai_core.model_client import SDKQueryError

        line = "[DEBUG] Claude Code returned an error result: You've hit your session limit"
        e = AgentRunError("run_agent [x] failed: exit 1", reason="failed: exit 1", stderr_tail=line)
        assert not is_usage_limit(e)
        try:
            try:
                raise e
            except AgentRunError as inner:
                raise SDKQueryError("claude_agent_sdk.query() failed: exit 1 after 2 attempts",
                                    stderr_tail=line) from inner
        except SDKQueryError as w:
            assert line in str(w)
            assert not is_usage_limit(w)

    def test_a_message_quoting_the_sdk_message_does_not_count(self):
        assert not is_usage_limit(
            "Claude Code returned an error result: finding F1: 'Claude Code returned an error result: "
            "You've hit your session limit'")

    def test_true_for_the_raw_message_anywhere_in_the_chain(self):
        try:
            try:
                raise _ResultError(WEEKLY_LIMIT)
            except _ResultError as e:
                raise RuntimeError("query failed") from e
        except RuntimeError as w:
            assert is_usage_limit(w)

    def test_true_for_a_string(self):
        assert is_usage_limit(SESSION_LIMIT)

    @pytest.mark.parametrize("value", [
        RuntimeError("Command failed with exit code 1"),
        AgentRunError("run_agent [x] timed out after 600s"),
        "This request would exceed your account's rate limit. Please try again later.",
        "",
    ])
    def test_false_otherwise(self, value):
        assert not is_usage_limit(value)

    def test_a_cause_cycle_terminates(self):
        a, b = RuntimeError("a"), RuntimeError("b")
        a.__cause__, b.__cause__ = b, a
        assert not is_usage_limit(a)


class TestTimeout:
    @staticmethod
    def _hanging_sdk() -> MagicMock:
        mock = _make_mock_sdk()
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
        mock = _make_mock_sdk()

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

    def test_timeout_partial_reports_tool_calls(self):
        mock = _make_mock_sdk()

        async def _agen(prompt, options):
            yield _FakeAssistantMessage(
                [_FakeToolUseBlock("Read", {"file_path": "/repo/a.py"})]
            )
            await asyncio.sleep(10)

        mock.query = MagicMock(side_effect=_agen)
        with patch.dict(sys.modules, {"claude_agent_sdk": mock}):
            with pytest.raises(AgentRunTimeout) as exc:
                _run(run_agent("hi", timeout_s=0.05))
        assert exc.value.partial.tool_calls == (
            ToolCall("Read", {"file_path": "/repo/a.py"}),
        )


class TestLargePromptsInline:
    """A prompt of any size goes to ``sdk.query`` unchanged.

    The SDK writes every prompt to the CLI's stdin, so there is no argv limit
    to work around. The old fallback wrote large prompts to a temp file and
    opened ``Read`` to reach it, which under bypassPermissions gave a no-tools
    call read access to the whole filesystem.
    """

    @staticmethod
    def _run_big(tmp_path, monkeypatch, **kwargs):
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        import tempfile as _tf
        _tf.tempdir = None  # force TMPDIR re-resolution
        big_prompt = "x" * (MAX_PROMPT_BYTES + 20_001)
        mock_sdk = _make_mock_sdk()
        try:
            with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
                _run(run_agent(big_prompt, **kwargs))
        finally:
            _tf.tempdir = None
        return big_prompt, mock_sdk

    def _assert_inline(self, tmp_path, big_prompt, mock_sdk):
        assert mock_sdk.query.call_args.kwargs["prompt"] == big_prompt
        opts_kwargs = mock_sdk.ClaudeAgentOptions.call_args.kwargs
        assert "Read" not in opts_kwargs["allowed_tools"]
        assert "Read" in opts_kwargs["disallowed_tools"]
        assert opts_kwargs["max_turns"] == 1
        assert "CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS" not in opts_kwargs["env"]
        assert list(tmp_path.glob("agent_prompt_*")) == []

    def test_big_prompt_default_args_stays_inline(self, tmp_path, monkeypatch):
        big_prompt, mock_sdk = self._run_big(tmp_path, monkeypatch)
        self._assert_inline(tmp_path, big_prompt, mock_sdk)

    def test_fallback_true_is_a_no_op_and_warns(self, tmp_path, monkeypatch):
        with pytest.warns(DeprecationWarning, match="prompt_file_fallback") as rec:
            big_prompt, mock_sdk = self._run_big(
                tmp_path, monkeypatch, prompt_file_fallback=True
            )
        assert sum(issubclass(w.category, DeprecationWarning) for w in rec) == 1
        self._assert_inline(tmp_path, big_prompt, mock_sdk)

    @pytest.mark.parametrize("kwargs", [{}, {"prompt_file_fallback": False}])
    def test_fallback_false_or_omitted_does_not_warn(
        self, tmp_path, monkeypatch, kwargs
    ):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            big_prompt, mock_sdk = self._run_big(tmp_path, monkeypatch, **kwargs)
        self._assert_inline(tmp_path, big_prompt, mock_sdk)

    def test_small_prompt_untouched(self):
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("small"))
        assert mock_sdk.query.call_args.kwargs["prompt"] == "small"

    def test_max_prompt_bytes_still_exported(self):
        import multiplai_core

        assert MAX_PROMPT_BYTES == 80_000
        assert multiplai_core.MAX_PROMPT_BYTES == 80_000
        assert "MAX_PROMPT_BYTES" in multiplai_core.__all__


class TestHeartbeat:
    """A run emits nothing between START and DONE, so a multi-minute call is
    indistinguishable from a wedged one. The heartbeat is that signal — and it
    must die with the attempt that spawned it."""

    @staticmethod
    def _alive_lines(caplog) -> list[str]:
        return [
            r.getMessage()
            for r in caplog.records
            if " alive " in r.getMessage()
        ]

    @staticmethod
    def _slow_sdk(delay: float) -> MagicMock:
        """SDK that emits text, then stalls *delay* seconds before finishing."""
        mock = _make_mock_sdk()

        async def _agen(prompt, options):
            yield _FakeAssistantMessage([_FakeTextBlock("hello")])
            await asyncio.sleep(delay)
            yield _FakeAssistantMessage([_FakeTextBlock(" world")])

        mock.query = MagicMock(side_effect=_agen)
        return mock

    def test_heartbeat_logs_progress_while_the_call_runs(self, caplog, monkeypatch):
        monkeypatch.setenv("MULTIPLAI_AGENT_HEARTBEAT_S", "0.05")
        caplog.set_level(logging.INFO, logger="multiplai_core.agent_runner")
        mock_sdk = self._slow_sdk(0.3)

        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            result = _run(run_agent("hi", max_turns=5, label="slow"))

        assert result.text == "hello world"
        lines = self._alive_lines(caplog)
        assert lines, "expected at least one alive heartbeat"
        assert "run_agent [slow] alive" in lines[0]
        assert "attempt=1/1" in lines[0]
        # The byte count is the point: it separates "slow but producing" from
        # "stalled with nothing". "hello" is already consumed by the first tick.
        byte_counts = [
            int(m.group(1))
            for line in lines
            if (m := re.search(r"text=(\d+) bytes", line))
        ]
        assert byte_counts and max(byte_counts) > 0

    def test_no_heartbeat_after_the_call_returns(self, caplog, monkeypatch):
        """The task is cancelled AND awaited in a finally — nothing may keep
        ticking once run_agent has handed back a result."""
        monkeypatch.setenv("MULTIPLAI_AGENT_HEARTBEAT_S", "0.05")
        caplog.set_level(logging.INFO, logger="multiplai_core.agent_runner")
        mock_sdk = self._slow_sdk(0.2)

        async def _test():
            await run_agent("hi", max_turns=5)
            at_return = len(self._alive_lines(caplog))
            await asyncio.sleep(0.3)  # several more intervals
            return at_return, len(self._alive_lines(caplog))

        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            at_return, later = _run(_test())

        assert at_return >= 1
        assert later == at_return

    def test_no_heartbeat_after_a_timed_out_attempt(self, caplog, monkeypatch):
        """On timeout the attempt is cancelled fire-and-forget; the heartbeat
        must not outlive it (the reason _consume() aclose()s its generator)."""
        monkeypatch.setenv("MULTIPLAI_AGENT_HEARTBEAT_S", "0.02")
        caplog.set_level(logging.INFO, logger="multiplai_core.agent_runner")
        mock_sdk = self._slow_sdk(10)

        async def _test():
            with pytest.raises(AgentRunTimeout):
                await run_agent("hi", max_turns=5, timeout_s=0.1)
            at_raise = len(self._alive_lines(caplog))
            await asyncio.sleep(0.2)
            return at_raise, len(self._alive_lines(caplog))

        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            at_raise, later = _run(_test())

        assert at_raise >= 1
        assert later == at_raise

    def test_interval_zero_disables_the_heartbeat(self, caplog, monkeypatch):
        monkeypatch.setenv("MULTIPLAI_AGENT_HEARTBEAT_S", "0")
        caplog.set_level(logging.INFO, logger="multiplai_core.agent_runner")
        mock_sdk = self._slow_sdk(0.3)

        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            result = _run(run_agent("hi", max_turns=5))

        assert result.text == "hello world"
        assert self._alive_lines(caplog) == []

    def test_interval_read_at_call_time_not_import_time(self, monkeypatch):
        """The knob is a call-time read, so a caller can set it per run —
        unlike model_client's _SDK_CALL_TIMEOUT_S, which is read once when that
        module is first imported."""
        from multiplai_core.agent_runner import (
            _HEARTBEAT_DEFAULT_S,
            _HEARTBEAT_ENV,
            _env_float,
        )

        monkeypatch.delenv(_HEARTBEAT_ENV, raising=False)
        assert _env_float(_HEARTBEAT_ENV, _HEARTBEAT_DEFAULT_S) == 60.0
        monkeypatch.setenv(_HEARTBEAT_ENV, "5")
        assert _env_float(_HEARTBEAT_ENV, _HEARTBEAT_DEFAULT_S) == 5.0
        monkeypatch.setenv(_HEARTBEAT_ENV, "not-a-number")
        assert _env_float(_HEARTBEAT_ENV, _HEARTBEAT_DEFAULT_S) == 60.0
        monkeypatch.setenv(_HEARTBEAT_ENV, "")
        assert _env_float(_HEARTBEAT_ENV, _HEARTBEAT_DEFAULT_S) == 60.0


class TestCostLedgerTap:
    """component= writes the run's usage to the cost ledger; ledger failures
    never break the run."""

    @pytest.fixture(autouse=True)
    def _tmp_workspace(self, tmp_workspace):
        pass

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
        from multiplai_core import costing

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


# ---------------------------------------------------------------------------
# CLI debug log on disk
# ---------------------------------------------------------------------------


class _FakeSystemMessage:
    def __init__(self, subtype: str, data: dict) -> None:
        self.subtype = subtype
        self.data = data


@pytest.fixture
def sdk_logs(tmp_path, monkeypatch):
    """Point run_agent's debug logs at tmp_path/sdk and reset the sweep flag."""
    import multiplai_core.agent_runner as ar

    monkeypatch.setattr("multiplai_core.log_utils._get_logs_dir", lambda: tmp_path)
    monkeypatch.setattr(ar, "_debug_logs_swept", False)
    monkeypatch.delenv("MULTIPLAI_SDK_DEBUG_LOG", raising=False)
    monkeypatch.delenv("MULTIPLAI_SDK_DEBUG_KEEP_S", raising=False)
    return tmp_path / "sdk"


class TestDebugLog:
    def test_lines_reach_disk_while_the_call_is_running(self, sdk_logs):
        seen: dict = {}
        mock = _make_mock_sdk(stderr_lines=["[DEBUG] [API REQUEST] /v1/messages x-client-request-id=abc"])

        async def _agen(prompt, options):
            # Read the file mid-call: the line must already be on disk, not
            # held until the attempt ends.
            files = list(sdk_logs.glob("*.log"))
            seen["files"] = files
            seen["text"] = files[0].read_text() if files else ""
            yield _FakeAssistantMessage([_FakeTextBlock("ok")])

        mock.query = MagicMock(side_effect=_agen)
        with patch.dict(sys.modules, {"claude_agent_sdk": mock}):
            _run(run_agent("hi", label="router", component="memory-router"))
        assert len(seen["files"]) == 1
        assert "x-client-request-id=abc" in seen["text"]
        assert "memory-router-router" in seen["files"][0].name

    def test_fast_success_keeps_the_file(self, sdk_logs, monkeypatch):
        # Every attempt's file is kept by default, however fast it succeeded.
        monkeypatch.delenv("MULTIPLAI_SDK_DEBUG_KEEP_S", raising=False)
        mock = _make_mock_sdk(stderr_lines=["[DEBUG] hello"])
        with patch.dict(sys.modules, {"claude_agent_sdk": mock}):
            _run(run_agent("hi"))
        files = list(sdk_logs.glob("*.log"))
        assert len(files) == 1
        assert files[0].read_text() == "[DEBUG] hello\n"

    def test_positive_keep_s_deletes_a_fast_success(self, sdk_logs, monkeypatch):
        # Opt-in only: a positive MULTIPLAI_SDK_DEBUG_KEEP_S still drops fast
        # successes.
        monkeypatch.setenv("MULTIPLAI_SDK_DEBUG_KEEP_S", "10")
        mock = _make_mock_sdk(stderr_lines=["[DEBUG] hello"])
        with patch.dict(sys.modules, {"claude_agent_sdk": mock}):
            _run(run_agent("hi"))
        assert list(sdk_logs.glob("*.log")) == []

    def test_keep_zero_keeps_every_file(self, sdk_logs, monkeypatch):
        monkeypatch.setenv("MULTIPLAI_SDK_DEBUG_KEEP_S", "0")
        mock = _make_mock_sdk(stderr_lines=["[DEBUG] hello"])
        with patch.dict(sys.modules, {"claude_agent_sdk": mock}):
            _run(run_agent("hi"))
        files = list(sdk_logs.glob("*.log"))
        assert len(files) == 1
        assert files[0].read_text() == "[DEBUG] hello\n"

    def test_slow_success_keeps_the_file(self, sdk_logs, monkeypatch):
        monkeypatch.setenv("MULTIPLAI_SDK_DEBUG_KEEP_S", "0.05")
        mock = _make_mock_sdk(stderr_lines=["[DEBUG] hello"])

        async def _agen(prompt, options):
            await asyncio.sleep(0.1)
            yield _FakeAssistantMessage([_FakeTextBlock("ok")])

        mock.query = MagicMock(side_effect=_agen)
        with patch.dict(sys.modules, {"claude_agent_sdk": mock}):
            _run(run_agent("hi"))
        assert len(list(sdk_logs.glob("*.log"))) == 1

    def test_concurrent_same_label_calls_get_their_own_files(self, sdk_logs, monkeypatch):
        # Two calls in one process, same label and component, started in the
        # same second. The fast one deletes its file on success; that must not
        # take the slow one's evidence with it.
        monkeypatch.setenv("MULTIPLAI_SDK_DEBUG_KEEP_S", "0.05")
        mock = _make_mock_sdk()

        async def _agen(prompt, options):
            options.stderr(f"[DEBUG] {prompt}")
            if prompt == "slow":
                await asyncio.sleep(0.15)
            yield _FakeAssistantMessage([_FakeTextBlock("ok")])

        mock.query = MagicMock(side_effect=_agen)

        async def _both():
            await asyncio.gather(run_agent("slow"), run_agent("fast"))

        with patch.dict(sys.modules, {"claude_agent_sdk": mock}):
            _run(_both())
        files = list(sdk_logs.glob("*.log"))
        assert len(files) == 1
        assert files[0].read_text() == "[DEBUG] slow\n"

    def test_failure_keeps_the_file_and_names_it(self, sdk_logs):
        mock = _make_mock_sdk(
            fail=RuntimeError("Command failed with exit code 1"),
            stderr_lines=["[DEBUG] getaddrinfo EAI_AGAIN api.anthropic.com"],
        )
        with patch.dict(sys.modules, {"claude_agent_sdk": mock}):
            with pytest.raises(AgentRunError) as exc:
                _run(run_agent("hi"))
        path = exc.value.debug_log_path
        assert path and os.path.exists(path)
        assert "EAI_AGAIN" in open(path).read()

    def test_timeout_keeps_the_file(self, sdk_logs):
        mock = _make_mock_sdk(stderr_lines=["[DEBUG] [API:auth] OAuth token check starting"])

        async def _agen(prompt, options):
            await asyncio.sleep(10)
            yield _FakeAssistantMessage([])

        mock.query = MagicMock(side_effect=_agen)
        with patch.dict(sys.modules, {"claude_agent_sdk": mock}):
            with pytest.raises(AgentRunTimeout) as exc:
                _run(run_agent("hi", timeout_s=0.05))
        assert "OAuth token check starting" in open(exc.value.debug_log_path).read()

    def test_off_writes_nothing(self, sdk_logs, monkeypatch):
        monkeypatch.setenv("MULTIPLAI_SDK_DEBUG_LOG", "off")
        mock = _make_mock_sdk(fail=RuntimeError("x"), stderr_lines=["[DEBUG] a"])
        with patch.dict(sys.modules, {"claude_agent_sdk": mock}):
            with pytest.raises(AgentRunError) as exc:
                _run(run_agent("hi"))
        assert exc.value.debug_log_path == ""
        assert not sdk_logs.exists() or list(sdk_logs.glob("*.log")) == []

    def test_unwritable_dir_does_not_break_the_run(self, tmp_path, monkeypatch):
        import multiplai_core.agent_runner as ar

        def _boom():
            raise OSError("read-only file system")

        monkeypatch.setattr("multiplai_core.log_utils._get_logs_dir", _boom)
        monkeypatch.setattr(ar, "_debug_logs_swept", False)
        mock = _make_mock_sdk(stderr_lines=["[DEBUG] a"])
        with patch.dict(sys.modules, {"claude_agent_sdk": mock}):
            result = _run(run_agent("hi"))
        assert result.text == "default text"

    def test_files_younger_than_a_year_are_kept_by_default(self, sdk_logs, monkeypatch):
        monkeypatch.delenv("MULTIPLAI_LOG_RETENTION_DAYS", raising=False)
        sdk_logs.mkdir(parents=True)
        old = sdk_logs / "old.log"
        old.write_text("x")
        months = 300 * 86400
        os.utime(old, (old.stat().st_atime - months, old.stat().st_mtime - months))
        with patch.dict(sys.modules, {"claude_agent_sdk": _make_mock_sdk()}):
            _run(run_agent("hi"))
        assert old.exists()

    def test_retention_sweep_deletes_old_files_when_set(self, sdk_logs, monkeypatch):
        monkeypatch.setenv("MULTIPLAI_LOG_RETENTION_DAYS", "7")
        sdk_logs.mkdir(parents=True)
        old = sdk_logs / "old.log"
        old.write_text("x")
        eight_days = 8 * 86400
        os.utime(old, (old.stat().st_atime - eight_days, old.stat().st_mtime - eight_days))
        fresh = sdk_logs / "fresh.log"
        fresh.write_text("y")
        with patch.dict(sys.modules, {"claude_agent_sdk": _make_mock_sdk()}):
            _run(run_agent("hi"))
        assert not old.exists()
        assert fresh.exists()

    def test_file_survives_sigkill_of_the_caller(self, tmp_path):
        """The case that motivated the file: a hook killed at its ceiling."""
        import signal
        import subprocess

        tests_dir = os.path.dirname(os.path.abspath(__file__))
        script = tmp_path / "child.py"
        script.write_text(
            "import asyncio, sys\n"
            f"sys.path.insert(0, {tests_dir!r})\n"
            "from pathlib import Path\n"
            "from unittest.mock import MagicMock\n"
            "import multiplai_core.log_utils as lu\n"
            f"lu._get_logs_dir = lambda: Path({str(tmp_path)!r})\n"
            "from _fakes import _make_mock_sdk\n"
            "mock = _make_mock_sdk(stderr_lines=['[DEBUG] [API REQUEST] /v1/messages x-client-request-id=zzz'])\n"
            "async def _agen(prompt, options):\n"
            "    print('READY', flush=True)\n"
            "    await asyncio.sleep(60)\n"
            "    yield None\n"
            "mock.query = MagicMock(side_effect=_agen)\n"
            "sys.modules['claude_agent_sdk'] = mock\n"
            "from multiplai_core.agent_runner import run_agent\n"
            "asyncio.run(run_agent('hi', label='router'))\n"
        )
        env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
        proc = subprocess.Popen(
            [sys.executable, str(script)], stdout=subprocess.PIPE, text=True, env=env,
        )
        try:
            assert proc.stdout.readline().strip() == "READY"
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()
        files = list((tmp_path / "sdk").glob("*.log"))
        assert len(files) == 1
        assert "x-client-request-id=zzz" in files[0].read_text()


class TestTimingLines:
    def test_cli_ready_and_first_reply_are_logged(self, sdk_logs, caplog):
        mock = _make_mock_sdk([
            _FakeSystemMessage("init", {"session_id": "child-123"}),
            _FakeAssistantMessage([_FakeTextBlock("a")]),
            _FakeAssistantMessage([_FakeTextBlock("b")]),
        ])
        mock.SystemMessage = _FakeSystemMessage
        with caplog.at_level(logging.INFO, logger="multiplai_core.agent_runner"):
            with patch.dict(sys.modules, {"claude_agent_sdk": mock}):
                _run(run_agent("hi", label="router", max_turns=3))
        msgs = [r.getMessage() for r in caplog.records]
        assert any(re.search(r"CLI ready after [\d.]+s child_session=child-123", m) for m in msgs)
        assert sum("first reply after" in m for m in msgs) == 1
        assert any("CLI debug log:" in m for m in msgs)

    def test_no_system_message_logs_unknown_child(self, sdk_logs, caplog):
        mock = _make_mock_sdk()
        with caplog.at_level(logging.INFO, logger="multiplai_core.agent_runner"):
            with patch.dict(sys.modules, {"claude_agent_sdk": mock}):
                _run(run_agent("hi"))
        assert any("child_session=?" in r.getMessage() for r in caplog.records)


class TestDefaultModelAndEffort:
    """With no model or effort passed, run_agent uses the session defaults."""

    def _opts(self, **kwargs):
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("hi", **kwargs))
        return mock_sdk.ClaudeAgentOptions.call_args.kwargs

    def test_explicit_model_and_effort_win_over_multiplai_model(self, monkeypatch):
        monkeypatch.setenv("MULTIPLAI_MODEL", "claude-test-default")
        monkeypatch.setenv("MULTIPLAI_EFFORT", "max")
        monkeypatch.setenv("MULTIPLAI_SESSION_MODEL", "claude-test-session")
        opts = self._opts(model="claude-explicit", effort="low")
        assert opts["model"] == "claude-explicit"
        assert opts["effort"] == "low"

    def test_multiplai_model_wins_over_the_session(self, monkeypatch):
        monkeypatch.setenv("MULTIPLAI_MODEL", "claude-test-default")
        monkeypatch.setenv("MULTIPLAI_EFFORT", "medium")
        monkeypatch.setenv("MULTIPLAI_SESSION_MODEL", "claude-test-session")
        monkeypatch.setenv("MULTIPLAI_SESSION_EFFORT", "low")
        opts = self._opts()
        assert opts["model"] == "claude-test-default"
        assert opts["effort"] == "medium"

    def test_the_session_is_used_when_nothing_is_set(self, monkeypatch):
        monkeypatch.setenv("MULTIPLAI_SESSION_MODEL", "claude-test-session")
        monkeypatch.setenv("MULTIPLAI_SESSION_EFFORT", "low")
        opts = self._opts()
        assert opts["model"] == "claude-test-session"
        assert opts["effort"] == "low"

    def test_multiplai_model_alone_keeps_the_session_effort(self, monkeypatch):
        monkeypatch.setenv("MULTIPLAI_MODEL", "sonnet")
        monkeypatch.setenv("MULTIPLAI_SESSION_EFFORT", "xhigh")
        opts = self._opts()
        assert opts["model"] == CURRENT_MODEL["sonnet"]
        assert opts["effort"] == "xhigh"

    def test_nothing_found_omits_both(self):
        opts = self._opts()
        assert opts.get("model") is None
        assert "effort" not in opts


class TestPathAllowed:
    """``path_allowed`` decides whether a read tool's path is inside a root."""

    def test_absolute_path_outside_is_denied(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "secret.txt"
        outside.write_text("x")
        assert not path_allowed("Read", {"file_path": str(outside)}, [root], root)

    def test_absolute_path_inside_is_allowed(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        assert path_allowed("Read", {"file_path": str(tmp_path / "a.py")},
                            [tmp_path], tmp_path)

    def test_relative_path_resolves_against_cwd(self, tmp_path):
        (tmp_path / "sub").mkdir()
        assert path_allowed("Read", {"file_path": "sub/a.py"}, [tmp_path], tmp_path)

    def test_dotdot_escape_is_denied(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        assert not path_allowed("Read", {"file_path": "../secret.txt"}, [root], root)
        assert not path_allowed(
            "Read", {"file_path": str(root / "sub" / ".." / ".." / "secret.txt")},
            [root], root)

    def test_symlink_inside_root_pointing_outside_is_denied(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("x")
        (root / "link").symlink_to(outside)
        assert not path_allowed("Read", {"file_path": str(root / "link" / "secret.txt")},
                                [root], root)
        assert not path_allowed("Grep", {"path": "link"}, [root], root)

    def test_root_given_through_a_symlink_still_matches(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        (tmp_path / "alias").symlink_to(real)
        assert path_allowed("Read", {"file_path": str(real / "a.py")},
                            [tmp_path / "alias"], real)

    def test_grep_with_no_path_uses_cwd(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        assert path_allowed("Grep", {"pattern": "secret"}, [root], root)
        assert not path_allowed("Grep", {"pattern": "secret"}, [root], tmp_path)

    def test_glob_with_absolute_pattern_outside_is_denied(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        assert not path_allowed("Glob", {"pattern": f"{tmp_path}/**/*.txt"}, [root], root)
        assert not path_allowed("Glob", {"pattern": "/etc/*"}, [root], root)

    def test_glob_relative_pattern_inside_is_allowed(self, tmp_path):
        assert path_allowed("Glob", {"pattern": "**/*.py"}, [tmp_path], tmp_path)
        assert path_allowed("Glob", {"pattern": "src/*.py", "path": "pkg"},
                            [tmp_path], tmp_path)

    def test_glob_relative_pattern_escaping_is_denied(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        assert not path_allowed("Glob", {"pattern": "../*"}, [root], root)
        assert not path_allowed("Glob", {"pattern": "*.py", "path": str(tmp_path)},
                                [root], root)

    def test_notebook_and_ls_paths_are_checked(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        assert not path_allowed("NotebookRead", {"notebook_path": str(tmp_path / "n.ipynb")},
                                [root], root)
        assert not path_allowed("LS", {"path": str(tmp_path)}, [root], root)
        assert path_allowed("LS", {"path": str(root)}, [root], root)

    def test_tilde_expands_to_home(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        root = tmp_path / "root"
        home.mkdir()
        root.mkdir()
        monkeypatch.setenv("HOME", str(home))
        assert not path_allowed("Read", {"file_path": "~/.ssh/id_rsa"}, [root], root)
        assert not path_allowed("Glob", {"pattern": "~/*"}, [root], root)

    def test_non_string_path_is_denied(self, tmp_path):
        assert not path_allowed("Read", {"file_path": ["/etc/passwd"]}, [tmp_path], tmp_path)

    def test_any_of_several_roots(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        assert path_allowed("Read", {"file_path": str(b / "x")}, [a, b], a)

    def test_sibling_with_shared_prefix_is_denied(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        assert not path_allowed("Read", {"file_path": str(tmp_path / "root-other" / "x")},
                                [root], root)


class TestReadRoots:
    """``read_roots`` installs a PreToolUse hook that denies reads outside."""

    def _opts_and_matcher(self, **kwargs):
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("hi", allowed_tools=["Read", "Grep", "Glob"], **kwargs))
        return mock_sdk.ClaudeAgentOptions.call_args.kwargs, mock_sdk.HookMatcher

    def test_no_hooks_without_read_roots(self):
        opts, matcher = self._opts_and_matcher()
        assert "hooks" not in opts
        assert not matcher.called

    def test_hook_installed_for_read_tools(self, tmp_path):
        opts, matcher = self._opts_and_matcher(read_roots=[tmp_path], cwd=tmp_path)
        assert list(opts["hooks"]) == ["PreToolUse"]
        assert opts["hooks"]["PreToolUse"] == [matcher.return_value]
        assert matcher.call_args.kwargs["matcher"] == "Read|Grep|Glob|LS|NotebookRead"
        assert len(matcher.call_args.kwargs["hooks"]) == 1

    def _hook(self, tmp_path, **kwargs):
        _, matcher = self._opts_and_matcher(read_roots=[tmp_path], **kwargs)
        return matcher.call_args.kwargs["hooks"][0]

    def test_hook_denies_outside(self, tmp_path):
        hook = self._hook(tmp_path, cwd=tmp_path)
        out = _run(hook({"hook_event_name": "PreToolUse", "tool_name": "Read",
                         "tool_input": {"file_path": "/etc/passwd"}}, "t1", None))
        spec = out["hookSpecificOutput"]
        assert spec["hookEventName"] == "PreToolUse"
        assert spec["permissionDecision"] == "deny"
        assert spec["permissionDecisionReason"]

    def test_hook_allows_inside(self, tmp_path):
        hook = self._hook(tmp_path, cwd=tmp_path)
        out = _run(hook({"hook_event_name": "PreToolUse", "tool_name": "Read",
                         "tool_input": {"file_path": str(tmp_path / "a.py")}}, "t1", None))
        assert out == {}

    def test_relative_paths_resolve_against_run_cwd(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        hook = self._hook(root, cwd=root)
        out = _run(hook({"hook_event_name": "PreToolUse", "tool_name": "Grep",
                         "tool_input": {"pattern": "x"}}, "t1", None))
        assert out == {}
        out = _run(hook({"hook_event_name": "PreToolUse", "tool_name": "Read",
                         "tool_input": {"file_path": "../x"}}, "t1", None))
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_empty_read_roots_denies_every_read(self, tmp_path):
        mock_sdk = _make_mock_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run(run_agent("hi", allowed_tools=["Read"], read_roots=[], cwd=tmp_path))
        hook = mock_sdk.HookMatcher.call_args.kwargs["hooks"][0]
        out = _run(hook({"hook_event_name": "PreToolUse", "tool_name": "Read",
                         "tool_input": {"file_path": str(tmp_path / "a")}}, "t1", None))
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_old_sdk_without_hook_matcher_refuses(self, tmp_path):
        """Fail closed: an SDK with no HookMatcher cannot enforce the roots."""
        mock_sdk = _make_mock_sdk()
        del mock_sdk.HookMatcher
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            with pytest.raises(AgentRunError, match="read_roots"):
                _run(run_agent("hi", read_roots=[tmp_path]))
        assert not mock_sdk.query.called

    def test_pi_backend_accepts_read_roots(self):
        import inspect

        from multiplai_core.pi.backend import run_pi_agent
        param = inspect.signature(run_pi_agent).parameters["read_roots"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is None
