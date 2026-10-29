"""run_agent retries connection errors and HTTP 5xx for up to 15 minutes.

The error texts below are the ones the bundled CLI (claude-agent-sdk 0.2.168)
writes into its error result, read from the CLI's own strings, plus the EAI_AGAIN
line run_agent logged on 2026-10-28 during a DNS outage.
"""

import asyncio
import sys
from unittest.mock import MagicMock, patch

import pytest
from _fakes import _FakeAssistantMessage, _FakeTextBlock, _make_mock_sdk

from multiplai_core import agent_runner
from multiplai_core.agent_runner import (
    AgentRunError,
    AgentRunTimeout,
    UsageLimitError,
    _is_network_error,
    run_agent,
)

PREFIX = "Claude Code returned an error result: "
# Logged by run_agent on 2026-10-28T18:08:59Z (buildme, owner-sections-source).
EAI_AGAIN = (
    PREFIX + "API Error: Can't reach the API server — check your internet "
    "or DNS (EAI_AGAIN) (exit code: 1)"
)
ECONNRESET = PREFIX + "API Error: Connection dropped (ECONNRESET)"
ENOTFOUND = (
    PREFIX + "API Error: Can't reach the API server — check your internet "
    "or DNS (ENOTFOUND)"
)
CONNECTION_LOST = (
    PREFIX + "API Error: Connection to the API was lost (ECONNRESET). This is "
    "usually temporary — try again."
)
NO_ROUTE = (
    PREFIX + "API Error: No internet route — check your connection or VPN (ENETUNREACH)"
)
HTTP_500 = (
    PREFIX + "API Error: 500 Internal server error. This is a server-side issue, "
    "usually temporary — try again in a moment. If it persists, check "
    "https://status.claude.com."
)
HTTP_503 = (
    PREFIX + "API Error: 503 upstream connect error. This is a server-side "
    "issue, usually temporary — try again in a moment."
)
REPEATED_529 = (
    PREFIX + "API Error: Repeated 529 Overloaded errors. The API is at capacity "
    "— this is usually temporary. Try again in a moment."
)
SESSION_LIMIT = (
    PREFIX + "You've hit your session limit · resets 9:40am (UTC) (exit code: 1)"
)


class _ResultError(Exception):
    """Stands in for claude_agent_sdk's ResultError."""


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setenv("MULTIPLAI_AGENT_HEARTBEAT_S", "0")
    monkeypatch.setenv("MULTIPLAI_SDK_DEBUG_LOG", "off")


def _scripted_sdk(outcomes, clock=None, attempt_s=0.0):
    """An SDK whose Nth query raises ``outcomes[N]`` (a message) or, for
    ``None``, answers "ok". Each query moves *clock* on by *attempt_s*, the way
    the CLI spends minutes on its own retries before it reports an error.
    The last outcome repeats."""
    calls = {"n": 0}
    mock = _make_mock_sdk()

    async def _agen(prompt, options):
        i = min(calls["n"], len(outcomes) - 1)
        calls["n"] += 1
        if clock is not None:
            clock["t"] += attempt_s
        outcome = outcomes[i]
        if outcome is not None:
            raise _ResultError(outcome)
        yield _FakeAssistantMessage([_FakeTextBlock("ok")])

    mock.query = MagicMock(side_effect=_agen)
    return mock, calls


def _run_clocked(coro_fn, clock, sleeps, monkeypatch, *, block_sleep=None):
    """Run ``coro_fn()`` on a loop whose clock is ``clock["t"]`` plus real time.

    ``asyncio.sleep`` is replaced: it records the wait, moves the clock on by
    it, and returns at once. With *block_sleep* set, the wait at that index
    (0-based) never returns, so the test can kill the run during it.
    """
    real_sleep = asyncio.sleep

    async def _fake_sleep(delay, result=None):
        sleeps.append(delay)
        if block_sleep is not None and len(sleeps) - 1 == block_sleep:
            await asyncio.get_running_loop().create_future()  # waits forever
        clock["t"] += delay
        await real_sleep(0)
        return result

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    loop = asyncio.new_event_loop()
    real_time = loop.time
    loop.time = lambda: real_time() + clock["t"]
    try:
        return loop.run_until_complete(coro_fn())
    finally:
        loop.close()


class TestClassification:
    @pytest.mark.parametrize(
        "text",
        [
            EAI_AGAIN,
            ECONNRESET,
            ENOTFOUND,
            CONNECTION_LOST,
            NO_ROUTE,
            HTTP_500,
            HTTP_503,
            REPEATED_529,
            "API Error: 502 Bad Gateway",
        ],
    )
    def test_connection_errors_and_5xx_are_network_errors(self, text):
        assert _is_network_error(text)

    @pytest.mark.parametrize(
        "text",
        [
            SESSION_LIMIT,
            "Command failed with exit code 1",
            PREFIX + "API Error: 400 due to tool use concurrency issues.",
            PREFIX + "API Error: 401 Invalid API key · Please run /login",
            PREFIX + "API Error: 429 rate_limit_error",
            PREFIX + "API Error: Connection refused — a firewall or proxy may "
            "be blocking it (ECONNREFUSED)",
            PREFIX + "API Error: Unable to connect to API: SSL certificate has expired",
            PREFIX + "Request timed out",
            # The phrase quoted inside a result, not the CLI's own error.
            PREFIX + "The agent said: API Error: 500 is what the server sent",
            "Error in hook: ECONNRESET while posting (exit code: 1)",
            "",
        ],
    )
    def test_everything_else_is_not(self, text):
        assert not _is_network_error(text)


class TestRetry:
    def test_eai_again_is_retried_with_the_default_one_attempt(self, monkeypatch):
        clock, sleeps = {"t": 0.0}, []
        mock_sdk, calls = _scripted_sdk([EAI_AGAIN, EAI_AGAIN, None])
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            result = _run_clocked(lambda: run_agent("hi"), clock, sleeps, monkeypatch)
        assert result.text == "ok"
        assert calls["n"] == 3
        assert sleeps == [10.0, 20.0]

    @pytest.mark.parametrize("text", [HTTP_500, REPEATED_529, ECONNRESET])
    def test_5xx_and_resets_are_retried(self, text, monkeypatch):
        clock, sleeps = {"t": 0.0}, []
        mock_sdk, calls = _scripted_sdk([text, None])
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            result = _run_clocked(lambda: run_agent("hi"), clock, sleeps, monkeypatch)
        assert result.text == "ok"
        assert calls["n"] == 2

    def test_other_failures_are_still_not_retried(self, monkeypatch):
        clock, sleeps = {"t": 0.0}, []
        mock_sdk, calls = _scripted_sdk(["Command failed with exit code 1"])
        with (
            patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}),
            pytest.raises(AgentRunError) as exc,
        ):
            _run_clocked(lambda: run_agent("hi"), clock, sleeps, monkeypatch)
        assert calls["n"] == 1
        assert exc.value.attempts == 1
        assert sleeps == []

    def test_usage_limit_is_still_not_retried(self, monkeypatch):
        clock, sleeps = {"t": 0.0}, []
        mock_sdk, calls = _scripted_sdk([EAI_AGAIN, SESSION_LIMIT])
        with (
            patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}),
            pytest.raises(UsageLimitError) as exc,
        ):
            _run_clocked(lambda: run_agent("hi"), clock, sleeps, monkeypatch)
        assert calls["n"] == 2
        assert exc.value.attempts == 2
        assert sleeps == [10.0]

    def test_eai_again_only_in_stderr_is_not_retried(self, monkeypatch):
        """The CLI logs EAI_AGAIN for its telemetry export even on runs that
        work; only the error result says the request itself failed."""
        clock, sleeps = {"t": 0.0}, []
        mock_sdk = _make_mock_sdk(
            fail=RuntimeError("Command failed with exit code 1"),
            stderr_lines=[
                (
                    "[ERROR] 1P event logging: 2 events failed to export "
                    "(code=EAI_AGAIN, getaddrinfo EAI_AGAIN api.anthropic.com)"
                )
            ],
        )
        with (
            patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}),
            pytest.raises(AgentRunError),
        ):
            _run_clocked(lambda: run_agent("hi"), clock, sleeps, monkeypatch)
        assert mock_sdk.query.call_count == 1

    def test_a_timeout_is_not_retried(self):
        async def _agen(prompt, options):
            await asyncio.sleep(10)
            yield  # pragma: no cover

        mock_sdk = _make_mock_sdk()
        mock_sdk.query = MagicMock(side_effect=_agen)
        with (
            patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}),
            pytest.raises(AgentRunTimeout),
        ):
            asyncio.run(run_agent("hi", timeout_s=0.05))
        assert mock_sdk.query.call_count == 1

    def test_network_retries_do_not_use_up_max_attempts(self, monkeypatch):
        clock, sleeps = {"t": 0.0}, []
        mock_sdk, calls = _scripted_sdk(
            [EAI_AGAIN, "Command failed with exit code 1", EAI_AGAIN, None]
        )
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            result = _run_clocked(
                lambda: run_agent("hi", max_attempts=2, retry_backoff_s=1.5),
                clock,
                sleeps,
                monkeypatch,
            )
        assert result.text == "ok"
        assert calls["n"] == 4
        assert sleeps == [10.0, 1.5, 20.0]


class TestWindow:
    def test_gives_up_15_minutes_after_the_first_network_failure(self, monkeypatch):
        """Each attempt spends 300 s (the CLI's own retries) before failing.
        Failures at 300, 610 and 930 s are retried; the one at 1270 s is past
        300 + 900 s, so the run fails there."""
        clock, sleeps = {"t": 0.0}, []
        mock_sdk, calls = _scripted_sdk([EAI_AGAIN], clock, attempt_s=300.0)
        with (
            patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}),
            pytest.raises(AgentRunError) as exc,
        ):
            _run_clocked(lambda: run_agent("hi"), clock, sleeps, monkeypatch)
        assert not isinstance(exc.value, AgentRunTimeout)
        assert calls["n"] == 4
        assert sleeps == [10.0, 20.0, 40.0]
        assert exc.value.attempts == 4
        assert "after 4 attempt(s)" in str(exc.value)
        assert "EAI_AGAIN" in exc.value.reason

    def test_backoff_doubles_up_to_60_s_and_stops_at_the_window(self, monkeypatch):
        clock, sleeps = {"t": 0.0}, []
        mock_sdk, calls = _scripted_sdk([HTTP_500], clock, attempt_s=0.0)
        with (
            patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}),
            pytest.raises(AgentRunError),
        ):
            _run_clocked(lambda: run_agent("hi"), clock, sleeps, monkeypatch)
        assert sleeps[:5] == [10.0, 20.0, 40.0, 60.0, 60.0]
        assert max(sleeps) == 60.0
        # The last wait is cut to what is left of the 900 s window.
        # (The test loop's clock also runs in real time, so allow a second.)
        assert sum(sleeps) == pytest.approx(
            agent_runner._NETWORK_RETRY_WINDOW_S, abs=1.0
        )
        assert calls["n"] == len(sleeps) + 1


class TestStopResume:
    def test_s27_killed_during_a_retry_wait_reruns_with_a_fresh_window(
        self, monkeypatch
    ):
        """S27: a run killed while it waits to retry keeps nothing. The step
        that called it reruns from its own checkpoint, and that rerun gets the
        whole 15-minute window again: it makes the same calls and waits, and
        fails or succeeds the same way, as a run that was never stopped."""

        def _outcomes():
            return [EAI_AGAIN]

        # A run that was never stopped.
        clock, sleeps = {"t": 0.0}, []
        mock_sdk, calls = _scripted_sdk(_outcomes(), clock, attempt_s=300.0)
        with (
            patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}),
            pytest.raises(AgentRunError) as uninterrupted,
        ):
            _run_clocked(lambda: run_agent("hi"), clock, sleeps, monkeypatch)
        expected_calls, expected_sleeps = calls["n"], list(sleeps)

        # The same run, killed during its second retry wait.
        clock, sleeps = {"t": 0.0}, []
        mock_sdk, calls = _scripted_sdk(_outcomes(), clock, attempt_s=300.0)

        async def _kill_during_wait():
            task = asyncio.ensure_future(run_agent("hi"))
            while len(sleeps) < 2:
                await asyncio.get_running_loop().run_in_executor(None, lambda: None)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            _run_clocked(_kill_during_wait, clock, sleeps, monkeypatch, block_sleep=1)
        assert calls["n"] == 2
        assert clock["t"] > 600.0  # the killed run spent time before the kill

        # The rerun, later on the same clock: a fresh window, not the rest of
        # the killed run's.
        rerun_sleeps: list[float] = []
        mock_sdk, calls = _scripted_sdk(_outcomes(), clock, attempt_s=300.0)
        with (
            patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}),
            pytest.raises(AgentRunError) as rerun,
        ):
            _run_clocked(lambda: run_agent("hi"), clock, rerun_sleeps, monkeypatch)
        assert calls["n"] == expected_calls
        assert rerun_sleeps == expected_sleeps
        assert rerun.value.attempts == uninterrupted.value.attempts
        assert rerun.value.reason == uninterrupted.value.reason
