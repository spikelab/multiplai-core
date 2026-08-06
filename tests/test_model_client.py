"""Tests for model client abstraction (multiplai_core/model_client.py)."""

import asyncio
import inspect
import logging
import os
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers for mocking claude_agent_sdk.query() (async generator)
# ---------------------------------------------------------------------------


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeAssistantMessage:
    def __init__(self, blocks: list) -> None:
        self.content = blocks


def _make_mock_sdk(
    texts: list[str] | None = None,
    *,
    fail: Exception | None = None,
    stderr_lines: list[str] | None = None,
) -> MagicMock:
    """Build a mock ``claude_agent_sdk`` module.

    ``self._sdk.query(prompt=..., options=...)`` must be an async generator
    yielding ``AssistantMessage`` objects whose ``.content`` is a list of
    ``TextBlock``. The mock also exposes the types used by ``isinstance()``
    checks and can simulate CLI stderr via the ``stderr`` options callback.
    """
    if texts is None:
        texts = ["default text"]

    mock = MagicMock()
    mock.AssistantMessage = _FakeAssistantMessage
    mock.TextBlock = _FakeTextBlock

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
        yield _FakeAssistantMessage([_FakeTextBlock(t) for t in texts])

    mock.query = MagicMock(side_effect=_agen)
    return mock


class TestModelClientInterface:
    """Verify ModelClient Protocol definition."""

    def test_model_client_is_protocol(self):
        from multiplai_core.model_client import ModelClient
        assert hasattr(ModelClient, "query")

    def test_query_is_async(self):
        """Every caller awaits ``query()``, so the Protocol must declare it as
        a coroutine function.

        Asserted via ``inspect``, not via ``typing`` internals: this test used
        to read ``ModelClient.__protocol_attrs__``, a CPython implementation
        detail that only exists on 3.12+, so it raised AttributeError on the
        3.11 floor ``requires-python`` claims. It also never checked async at
        all — only that the name existed, which is already covered by
        ``test_model_client_is_protocol``.
        """
        from multiplai_core.model_client import ModelClient
        assert inspect.iscoroutinefunction(ModelClient.query)

    def test_unimplemented_subclass_fails(self):
        from multiplai_core.model_client import ModelClient
        class BadClient:
            pass
        assert not isinstance(BadClient(), ModelClient)

    def test_query_signature_accepts_all_specified_params(self):
        """WHEN query() is called with system, messages, model, max_tokens, temperature
        THEN the method accepts all parameters without error."""
        from multiplai_core.model_client import ModelClient
        sig = inspect.signature(ModelClient.query)
        param_names = set(sig.parameters.keys())
        # Must include all specified parameters
        assert "system" in param_names or "self" in param_names
        assert "messages" in param_names
        assert "model" in param_names
        assert "max_tokens" in param_names
        assert "temperature" in param_names


class TestAgentSDKClient:
    """Verify AgentSDKClient implementation."""

    def test_raises_import_error_when_sdk_missing(self):
        from multiplai_core.model_client import AgentSDKClient
        with patch.dict(sys.modules, {"claude_agent_sdk": None}):
            with pytest.raises(ImportError, match="claude_agent_sdk"):
                AgentSDKClient()

    def test_successful_instantiation_with_mock_sdk(self):
        mock_sdk = MagicMock()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            from multiplai_core.model_client import AgentSDKClient
            client = AgentSDKClient()
            assert client._sdk is mock_sdk

    def test_query_delegates_to_sdk(self):
        mock_sdk = _make_mock_sdk(["test response"])
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            from multiplai_core.model_client import AgentSDKClient
            client = AgentSDKClient()

            async def _test():
                result = await client.query("system", [{"role": "user", "content": "hello"}])
                assert result.content == "test response"

            asyncio.run(_test())

    def test_query_propagates_exceptions(self):
        """WHEN the SDK async generator raises
        THEN the error is wrapped in SDKQueryError with the captured stderr tail."""
        mock_sdk = _make_mock_sdk(
            fail=RuntimeError("SDK error"),
            stderr_lines=["auth: token expired"],
        )
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            from multiplai_core.model_client import AgentSDKClient, SDKQueryError
            client = AgentSDKClient()

            async def _test():
                with pytest.raises(SDKQueryError) as exc_info:
                    await client.query("system", [{"role": "user", "content": "hi"}])
                assert "SDK error" in str(exc_info.value)
                assert "auth: token expired" in exc_info.value.stderr_tail

            asyncio.run(_test())

    def test_query_forwards_prompt_and_options(self):
        """WHEN query() is called with system, messages, and model
        THEN ``self._sdk.query(prompt=..., options=...)`` receives the joined
        user messages as prompt and the system + model in ClaudeAgentOptions."""
        mock_sdk = _make_mock_sdk(["ok"])

        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            from multiplai_core.model_client import AgentSDKClient, _NO_TOOLS_SUFFIX
            client = AgentSDKClient()

            messages = [{"role": "user", "content": "test"}]

            async def _test():
                await client.query(
                    "system prompt",
                    messages,
                    model="claude-opus-4-20250514",
                )
                call = mock_sdk.query.call_args
                assert call.kwargs["prompt"] == "test"
                opts = call.kwargs["options"]
                # query() appends the no-tools directive to the caller's system
                # prompt before forwarding it to the SDK.
                assert opts.system_prompt == "system prompt" + _NO_TOOLS_SUFFIX
                assert opts.model == "claude-opus-4-20250514"

            asyncio.run(_test())

    def test_query_sets_strict_mcp_config(self):
        """Regression: the SDK subprocess MUST pass --strict-mcp-config so it
        ignores account-level MCP integrations (claude.ai Gmail/Drive/etc).
        Without it the nested CLI tries non-interactive OAuth and exits 1.
        Verified root cause 2026-05-19 — do not remove."""
        mock_sdk = _make_mock_sdk(["ok"])
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            from multiplai_core.model_client import AgentSDKClient
            client = AgentSDKClient()

            async def _test():
                await client.query("sys", [{"role": "user", "content": "hi"}])
                opts = mock_sdk.query.call_args.kwargs["options"]
                assert "strict-mcp-config" in opts.extra_args

            asyncio.run(_test())

    def test_query_joins_multiple_user_messages(self):
        """WHEN query() is called with multiple user messages
        THEN they are concatenated into the prompt (non-user roles are dropped)."""
        mock_sdk = _make_mock_sdk(["ok"])
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            from multiplai_core.model_client import AgentSDKClient
            client = AgentSDKClient()

            messages = [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ignored"},
                {"role": "user", "content": "second"},
            ]

            async def _test():
                await client.query("sys", messages)
                prompt = mock_sdk.query.call_args.kwargs["prompt"]
                assert "first" in prompt and "second" in prompt
                assert "ignored" not in prompt

            asyncio.run(_test())

    def test_query_accepts_content_block_lists(self):
        """WHEN a user message's content is an Anthropic content-block list
        THEN the text blocks are flattened into the prompt (parity with the
        API backend), not stringified into a TypeError."""
        mock_sdk = _make_mock_sdk(["ok"])
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            from multiplai_core.model_client import AgentSDKClient
            client = AgentSDKClient()

            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": "block one"},
                    {"type": "image", "source": {}},
                    {"type": "text", "text": "block two"},
                ]},
            ]

            async def _test():
                await client.query("sys", messages)
                prompt = mock_sdk.query.call_args.kwargs["prompt"]
                assert "block one" in prompt and "block two" in prompt

            asyncio.run(_test())

    def test_query_extracts_text_from_text_blocks(self):
        """WHEN the SDK yields AssistantMessage with multiple TextBlock entries
        THEN their .text values are concatenated into ModelResponse.content."""
        mock_sdk = _make_mock_sdk(["first ", "second"])
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            from multiplai_core.model_client import AgentSDKClient
            client = AgentSDKClient()

            async def _test():
                result = await client.query(
                    "sys", [{"role": "user", "content": "hi"}],
                )
                assert result.content == "first second"

            asyncio.run(_test())


    def test_safe_query_skips_unknown_message_types(self):
        """WHEN the SDK parser raises 'Unknown message type' mid-stream
        THEN _safe_query skips it and the surrounding good messages still
        produce a complete response (regression: debug-to-stderr emits
        internal message types the bundled parser doesn't recognize)."""

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
            from multiplai_core.model_client import AgentSDKClient
            client = AgentSDKClient()

            async def _test():
                result = await client.query("sys", [{"role": "user", "content": "hi"}])
                assert result.content == "hello world"

            asyncio.run(_test())

    def test_query_retries_then_succeeds(self):
        """WHEN the first SDK call fails with a transient exit-1
        THEN query() retries and returns the second call's result."""
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
            from multiplai_core import model_client
            client = model_client.AgentSDKClient()

            async def _test():
                with patch.object(model_client, "_SDK_RETRY_BACKOFF_S", 0):
                    result = await client.query(
                        "sys", [{"role": "user", "content": "hi"}]
                    )
                assert result.content == "recovered"
                assert calls["n"] == 2

            asyncio.run(_test())

    def test_query_raises_after_max_attempts(self):
        """WHEN every attempt fails THEN SDKQueryError reports the attempt
        count and preserves the captured stderr tail."""
        mock_sdk = _make_mock_sdk(
            fail=RuntimeError("persistent failure"),
            stderr_lines=["cli: fatal"],
        )
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            from multiplai_core import model_client
            client = model_client.AgentSDKClient()

            async def _test():
                with patch.object(model_client, "_SDK_RETRY_BACKOFF_S", 0):
                    with pytest.raises(model_client.SDKQueryError) as exc:
                        await client.query("sys", [{"role": "user", "content": "x"}])
                assert "after 2 attempts" in str(exc.value)
                assert "persistent failure" in str(exc.value)
                assert "cli: fatal" in exc.value.stderr_tail

            asyncio.run(_test())

    def test_failure_tail_prefers_error_lines(self):
        """WHEN the CLI emits [ERROR] lines amid DEBUG noise and the call fails
        THEN the surfaced stderr_tail keeps the [ERROR] lines and drops DEBUG."""
        mock_sdk = _make_mock_sdk(
            fail=RuntimeError("Command failed with exit code 1"),
            stderr_lines=[
                "2026-01-01 [DEBUG] configureGlobalAgents complete",
                "2026-01-01 [DEBUG] MDM settings load completed",
                "2026-01-01 [ERROR] API error: 429 rate_limit_error",
                "2026-01-01 [DEBUG] LSP server manager shut down",
            ],
        )
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            from multiplai_core import model_client
            client = model_client.AgentSDKClient()

            async def _test():
                with patch.object(model_client, "_SDK_RETRY_BACKOFF_S", 0):
                    with pytest.raises(model_client.SDKQueryError) as exc:
                        await client.query("sys", [{"role": "user", "content": "x"}])
                tail = exc.value.stderr_tail
                assert "429 rate_limit_error" in tail
                assert "[DEBUG]" not in tail

            asyncio.run(_test())

    def test_no_stderr_files_written(self):
        """REGRESSION: stderr is captured in memory only — the per-invocation
        file machinery (and its helper) must not come back."""
        from multiplai_core import model_client
        assert not hasattr(model_client, "_stderr_log_dir")
        assert "stderr_log_path" not in inspect.signature(
            model_client.SDKQueryError.__init__
        ).parameters


class TestSummarizeStderr:
    """Unit tests for the in-memory stderr summarizer."""

    def test_keeps_error_lines_and_dedups_consecutive(self):
        from multiplai_core.model_client import _summarize_stderr
        out = _summarize_stderr(
            error_lines=["[ERROR] boom", "[ERROR] boom", "[ERROR] kapow"],
            recent_lines=["[DEBUG] noise"],
        )
        assert out == "[ERROR] boom\n[ERROR] kapow"

    def test_falls_back_to_recent_lines_when_no_errors(self):
        from multiplai_core.model_client import _summarize_stderr
        out = _summarize_stderr(
            error_lines=[],
            recent_lines=["[DEBUG] a", "[WARN] b", "last line"],
        )
        assert out == "[DEBUG] a\n[WARN] b\nlast line"

    def test_empty_when_nothing_captured(self):
        from multiplai_core.model_client import _summarize_stderr
        assert _summarize_stderr([], []) == ""


class TestAnthropicAPIClient:
    """Verify AnthropicAPIClient implementation."""

    def test_raises_value_error_on_empty_key(self):
        from multiplai_core.model_client import AnthropicAPIClient
        with pytest.raises(ValueError, match="API key is required"):
            AnthropicAPIClient("")

    def test_raises_value_error_on_none_key(self):
        from multiplai_core.model_client import AnthropicAPIClient
        with pytest.raises(ValueError, match="API key is required"):
            AnthropicAPIClient(None)

    def test_successful_instantiation(self):
        from multiplai_core.model_client import AnthropicAPIClient
        client = AnthropicAPIClient("sk-test-key")
        assert client._api_key == "sk-test-key"

    def test_default_model(self):
        from multiplai_core.model_client import AnthropicAPIClient, DEFAULT_MODEL
        client = AnthropicAPIClient("sk-test-key")
        sig = inspect.signature(client.query)
        assert sig.parameters["model"].default == DEFAULT_MODEL

    def test_successful_query_via_anthropic_api(self):
        """WHEN AnthropicAPIClient.query() is called with valid key, system, messages
        THEN it calls anthropic.AsyncAnthropic().messages.create() and returns
        a response with .content containing the model's text."""
        from multiplai_core.model_client import AnthropicAPIClient

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "API response text"
        mock_api_response = MagicMock()
        mock_api_response.content = [mock_text_block]

        mock_async_client = MagicMock()
        mock_async_client.messages.create = AsyncMock(return_value=mock_api_response)

        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_async_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            client = AnthropicAPIClient("sk-test-key")
            client._client = None  # ensure lazy init triggers

            async def _test():
                result = await client.query(
                    "You are helpful.",
                    [{"role": "user", "content": "hello"}],
                )
                assert result.content == "API response text"
                mock_async_client.messages.create.assert_called_once()
                call_kwargs = mock_async_client.messages.create.call_args
                assert call_kwargs.kwargs["system"] == "You are helpful."
                assert call_kwargs.kwargs["messages"] == [{"role": "user", "content": "hello"}]

            asyncio.run(_test())

    def test_model_override(self):
        """WHEN query() is called with model='claude-opus-4-20250514'
        THEN the request uses that model instead of the default."""
        from multiplai_core.model_client import AnthropicAPIClient

        mock_text_block = MagicMock()
        mock_text_block.text = "opus response"
        mock_api_response = MagicMock()
        mock_api_response.content = [mock_text_block]

        mock_async_client = MagicMock()
        mock_async_client.messages.create = AsyncMock(return_value=mock_api_response)

        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_async_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            client = AnthropicAPIClient("sk-test-key")
            client._client = None

            async def _test():
                await client.query("sys", [], model="claude-opus-4-20250514")
                call_kwargs = mock_async_client.messages.create.call_args
                assert call_kwargs.kwargs["model"] == "claude-opus-4-20250514"

            asyncio.run(_test())

    def test_default_model_sent_to_api(self):
        """WHEN query() is called without explicit model kwarg
        THEN the request uses DEFAULT_MODEL.

        Asserting the literal ID here would only duplicate what
        ``env.CURRENT_MODEL`` already says — and a stale duplicate is how the
        default drifted a generation behind that table in the first place.
        """
        from multiplai_core.model_client import AnthropicAPIClient, DEFAULT_MODEL

        mock_text_block = MagicMock()
        mock_text_block.text = "response"
        mock_api_response = MagicMock()
        mock_api_response.content = [mock_text_block]

        mock_async_client = MagicMock()
        mock_async_client.messages.create = AsyncMock(return_value=mock_api_response)

        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_async_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            client = AnthropicAPIClient("sk-test-key")
            client._client = None

            async def _test():
                await client.query("sys", [])
                call_kwargs = mock_async_client.messages.create.call_args
                assert call_kwargs.kwargs["model"] == DEFAULT_MODEL

            asyncio.run(_test())

    def test_default_max_tokens_sent_to_api(self):
        """WHEN query() is called without max_tokens kwarg
        THEN the request is sent with max_tokens=4096."""
        from multiplai_core.model_client import AnthropicAPIClient

        mock_text_block = MagicMock()
        mock_text_block.text = "response"
        mock_api_response = MagicMock()
        mock_api_response.content = [mock_text_block]

        mock_async_client = MagicMock()
        mock_async_client.messages.create = AsyncMock(return_value=mock_api_response)

        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_async_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            client = AnthropicAPIClient("sk-test-key")
            client._client = None

            async def _test():
                await client.query("sys", [])
                call_kwargs = mock_async_client.messages.create.call_args
                assert call_kwargs.kwargs["max_tokens"] == 4096

            asyncio.run(_test())

    def test_caller_override_max_tokens(self):
        """WHEN query() is called with max_tokens=16000
        THEN the request uses 16000, not the default."""
        from multiplai_core.model_client import AnthropicAPIClient

        mock_text_block = MagicMock()
        mock_text_block.text = "response"
        mock_api_response = MagicMock()
        mock_api_response.content = [mock_text_block]

        mock_async_client = MagicMock()
        mock_async_client.messages.create = AsyncMock(return_value=mock_api_response)

        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_async_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            client = AnthropicAPIClient("sk-test-key")
            client._client = None

            async def _test():
                await client.query("sys", [], max_tokens=16000)
                call_kwargs = mock_async_client.messages.create.call_args
                assert call_kwargs.kwargs["max_tokens"] == 16000

            asyncio.run(_test())


class TestCreateClientFactory:
    """Verify create_client() factory function."""

    def test_returns_agent_sdk_when_available(self):
        mock_sdk = MagicMock()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            from multiplai_core.model_client import create_client, AgentSDKClient

            async def _test():
                client = await create_client()
                assert isinstance(client, AgentSDKClient)

            asyncio.run(_test())

    def test_falls_back_to_api_client_with_key(self):
        with patch.dict(sys.modules, {"claude_agent_sdk": None}):
            from multiplai_core.model_client import create_client, AnthropicAPIClient

            async def _test():
                client = await create_client(api_key="sk-test")
                assert isinstance(client, AnthropicAPIClient)

            asyncio.run(_test())

    def test_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_ANTHROPIC_API_KEY", "sk-env-key")
        with patch.dict(sys.modules, {"claude_agent_sdk": None}):
            from multiplai_core.model_client import create_client, AnthropicAPIClient

            async def _test():
                client = await create_client()
                assert isinstance(client, AnthropicAPIClient)

            asyncio.run(_test())

    def test_raises_when_no_sdk_no_key(self, monkeypatch):
        monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_ANTHROPIC_API_KEY", raising=False)
        with patch.dict(sys.modules, {"claude_agent_sdk": None}):
            from multiplai_core.model_client import create_client

            async def _test():
                with pytest.raises(RuntimeError, match="Neither"):
                    await create_client()

            asyncio.run(_test())


class TestResponseNormalization:
    """Verify both clients return consistent response objects."""

    def test_agent_sdk_response_has_content(self):
        mock_sdk = _make_mock_sdk(["hello"])
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            from multiplai_core.model_client import AgentSDKClient

            async def _test():
                client = AgentSDKClient()
                result = await client.query(
                    "sys", [{"role": "user", "content": "hi"}],
                )
                assert hasattr(result, "content")
                assert isinstance(result.content, str)

            asyncio.run(_test())

    def test_anthropic_api_response_has_content(self):
        """WHEN AnthropicAPIClient.query() returns successfully
        THEN the return value has a .content attribute that is a string,
        extracted from the Anthropic API's response.content[0].text structure."""
        from multiplai_core.model_client import AnthropicAPIClient, ModelResponse

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "anthropic response text"
        mock_api_response = MagicMock()
        mock_api_response.content = [mock_text_block]

        mock_async_client = MagicMock()
        mock_async_client.messages.create = AsyncMock(return_value=mock_api_response)

        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_async_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            client = AnthropicAPIClient("sk-test-key")
            client._client = None

            async def _test():
                result = await client.query("sys", [{"role": "user", "content": "hi"}])
                assert hasattr(result, "content")
                assert isinstance(result.content, str)
                assert result.content == "anthropic response text"
                # Verify it's a ModelResponse instance for type consistency
                assert isinstance(result, ModelResponse)

            asyncio.run(_test())

    def test_both_clients_return_model_response(self):
        """Both implementations return ModelResponse for interface consistency."""
        from multiplai_core.model_client import AgentSDKClient, AnthropicAPIClient, ModelResponse

        mock_sdk = _make_mock_sdk(["sdk text"])

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "api text"
        mock_api_response = MagicMock()
        mock_api_response.content = [mock_text_block]

        mock_async_client = MagicMock()
        mock_async_client.messages.create = AsyncMock(return_value=mock_api_response)

        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_async_client

        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk, "anthropic": mock_anthropic}):
            sdk_client = AgentSDKClient()
            api_client = AnthropicAPIClient("sk-test")
            api_client._client = None

            async def _test():
                messages = [{"role": "user", "content": "hi"}]
                sdk_result = await sdk_client.query("sys", messages)
                api_result = await api_client.query("sys", messages)
                assert isinstance(sdk_result, ModelResponse)
                assert isinstance(api_result, ModelResponse)
                # Assert the actual extracted text, not just the type — a weak
                # isinstance-only check let the API-vs-SDK extraction asymmetry
                # (first-block-only truncation) go unnoticed.
                assert sdk_result.content == "sdk text"
                assert api_result.content == "api text"

            asyncio.run(_test())

    def test_api_concatenates_multiple_text_blocks(self):
        """A response whose text is split across multiple text blocks (around
        thinking/citation blocks) must be joined, not truncated to the first —
        matching AgentSDKClient. Non-text blocks are skipped."""
        from multiplai_core.model_client import AnthropicAPIClient

        def _blk(btype, text):
            b = MagicMock()
            b.type = btype
            b.text = text
            return b

        mock_api_response = MagicMock()
        mock_api_response.content = [
            _blk("text", "first "),
            _blk("thinking", "IGNORED"),
            _blk("text", "second"),
        ]
        mock_async_client = MagicMock()
        mock_async_client.messages.create = AsyncMock(return_value=mock_api_response)
        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_async_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            client = AnthropicAPIClient("sk-test")
            client._client = None

            async def _test():
                result = await client.query("sys", [{"role": "user", "content": "hi"}])
                assert result.content == "first second"

            asyncio.run(_test())

    def test_api_empty_content_returns_empty_string(self):
        """A tool-only/refusal turn (no text blocks) yields "" not IndexError."""
        from multiplai_core.model_client import AnthropicAPIClient

        mock_api_response = MagicMock()
        mock_api_response.content = []
        mock_async_client = MagicMock()
        mock_async_client.messages.create = AsyncMock(return_value=mock_api_response)
        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_async_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            client = AnthropicAPIClient("sk-test")
            client._client = None

            async def _test():
                result = await client.query("sys", [{"role": "user", "content": "hi"}])
                assert result.content == ""

            asyncio.run(_test())


class TestAsyncInterface:
    """Verify async nature of the interface."""

    def test_query_is_coroutine(self):
        from multiplai_core.model_client import AnthropicAPIClient
        client = AnthropicAPIClient("sk-test")
        result = client.query("sys", [])
        assert asyncio.iscoroutine(result)
        result.close()  # clean up

    def test_create_client_is_coroutine(self):
        from multiplai_core.model_client import create_client
        result = create_client()
        assert asyncio.iscoroutine(result)
        result.close()

    def test_create_client_and_query_work_inside_asyncio_run(self):
        """WHEN create_client() and client.query() are called inside asyncio.run()
        THEN both complete without event loop errors."""
        mock_sdk = _make_mock_sdk(["integration test"])
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            from multiplai_core.model_client import create_client

            async def _test():
                client = await create_client()
                result = await client.query(
                    "system",
                    [{"role": "user", "content": "test"}],
                )
                assert result.content == "integration test"

            asyncio.run(_test())

    def test_agent_sdk_query_is_awaitable(self):
        """WHEN client.query() is called on AgentSDKClient
        THEN it returns a coroutine that must be awaited."""
        mock_sdk = _make_mock_sdk(["ok"])
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            from multiplai_core.model_client import AgentSDKClient
            client = AgentSDKClient()
            result = client.query("sys", [{"role": "user", "content": "hi"}])
            assert asyncio.iscoroutine(result)
            actual = asyncio.run(result)
            assert actual.content == "ok"


class TestNoVendoring:
    """Verify the SDK is depended on, not vendored (source bundled)."""

    def test_sdk_import_deferred(self):
        """Module imports without claude_agent_sdk installed."""
        # Just importing the module should succeed
        from multiplai_core import model_client
        assert hasattr(model_client, "ModelClient")

    def test_pyproject_declares_deps_not_vendored(self):
        """anthropic and claude-agent-sdk are declared as dependencies (the
        opposite of vendoring), with loose ranges rather than exact pins so a
        consumer's resolver can satisfy them. Vendoring = bundling SDK source
        into the repo, which would show up as a committed package dir."""
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        text = pyproject.read_text().lower()
        assert "anthropic" in text
        assert "claude-agent-sdk" in text
        # Deliberately NOT exact-pinned (a library must not pin its deps).
        assert "anthropic==" not in text
        assert "claude-agent-sdk==" not in text
        # Not vendored: no SDK source tree committed under the package.
        pkg = Path(__file__).resolve().parent.parent / "src" / "multiplai_core"
        assert not (pkg / "claude_agent_sdk").exists()


class TestMaxTokensDefaults:
    """Verify default max_tokens behavior."""

    def test_default_max_tokens_value(self):
        from multiplai_core.model_client import DEFAULT_MAX_TOKENS
        assert DEFAULT_MAX_TOKENS == 4096

    def test_agent_sdk_accepts_max_tokens_for_interface_parity(self):
        """WHEN AgentSDKClient.query() is called with max_tokens
        THEN the call succeeds even though the SDK uses session defaults.

        ``claude_agent_sdk.query()`` takes no per-query max_tokens argument —
        the parameter is accepted only so ``AgentSDKClient`` and
        ``AnthropicAPIClient`` share a consistent interface.
        """
        mock_sdk = _make_mock_sdk(["ok"])
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            from multiplai_core.model_client import AgentSDKClient
            client = AgentSDKClient()

            async def _test():
                result = await client.query(
                    "sys",
                    [{"role": "user", "content": "hi"}],
                    max_tokens=16000,
                )
                assert result.content == "ok"

            asyncio.run(_test())


class TestLoggingOnFallback:
    """Verify logging when client is selected."""

    def test_agent_sdk_logs_info(self):
        """Verify Agent SDK selection is logged at INFO level."""
        mock_sdk = MagicMock()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            from multiplai_core.model_client import create_client, logger as mc_logger
            records = []

            class Capture(logging.Handler):
                def emit(self, record):
                    records.append(record)

            cap = Capture()
            mc_logger.addHandler(cap)
            mc_logger.setLevel(logging.DEBUG)
            try:
                asyncio.run(create_client())
                assert any("Agent SDK" in r.getMessage() for r in records)
                # Must be info-level specifically
                sdk_records = [r for r in records if "Agent SDK" in r.getMessage()]
                assert any(r.levelno == logging.INFO for r in sdk_records)
            finally:
                mc_logger.removeHandler(cap)

    def test_fallback_logs_warning(self):
        """Verify fallback to API client is logged as warning."""
        from multiplai_core import model_client
        records = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        actual_logger = model_client.logger
        cap = Capture()
        actual_logger.addHandler(cap)
        actual_logger.setLevel(logging.DEBUG)
        try:
            with patch.dict(sys.modules, {"claude_agent_sdk": None}):
                asyncio.run(model_client.create_client(api_key="sk-test"))
            assert any("Falling back" in r.getMessage() for r in records)
            assert any(r.levelno >= logging.WARNING for r in records)
        finally:
            actual_logger.removeHandler(cap)


class TestSDKCallTimeout:
    """Cover the v0.2 hard-timeout guard around a stalled SDK call."""

    @staticmethod
    def _hanging_sdk() -> MagicMock:
        """Mock SDK whose query() generator never yields (simulates a hang)."""
        mock = MagicMock()
        mock.AssistantMessage = _FakeAssistantMessage
        mock.TextBlock = _FakeTextBlock

        def _options_ctor(**kwargs):
            opts = MagicMock()
            for k, v in kwargs.items():
                setattr(opts, k, v)
            return opts

        mock.ClaudeAgentOptions = MagicMock(side_effect=_options_ctor)

        closed = {"aclosed": False}

        async def _agen(prompt, options):
            try:
                await asyncio.sleep(10)  # longer than the patched timeout
                yield _FakeAssistantMessage([_FakeTextBlock("never")])
            except asyncio.CancelledError:
                closed["aclosed"] = True
                raise

        mock.query = MagicMock(side_effect=_agen)
        mock._closed = closed
        return mock

    def test_timeout_raises_sdk_query_error_with_reason(self):
        """A stalled call times out on every attempt and surfaces as
        SDKQueryError whose message names the timeout (TimeoutError
        stringifies to '')."""
        from multiplai_core import model_client

        mock_sdk = self._hanging_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}), \
                patch.object(model_client, "_SDK_CALL_TIMEOUT_S", 0.05), \
                patch.object(model_client, "_SDK_RETRY_BACKOFF_S", 0.0):
            client = model_client.AgentSDKClient()

            async def _test():
                with pytest.raises(model_client.SDKQueryError) as exc_info:
                    await client.query("sys", [{"role": "user", "content": "hi"}])
                assert "timed out" in str(exc_info.value)

            asyncio.run(_test())

    def test_timeout_closes_generator(self):
        """On timeout the generator is aclose()'d so the CLI subprocess is torn
        down synchronously rather than leaking until GC."""
        from multiplai_core import model_client

        mock_sdk = self._hanging_sdk()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}), \
                patch.object(model_client, "_SDK_CALL_TIMEOUT_S", 0.05), \
                patch.object(model_client, "_SDK_RETRY_BACKOFF_S", 0.0):
            client = model_client.AgentSDKClient()

            async def _test():
                with pytest.raises(model_client.SDKQueryError):
                    await client.query("sys", [{"role": "user", "content": "hi"}])

            asyncio.run(_test())
            assert mock_sdk._closed["aclosed"] is True

    def test_retry_recovers_after_first_timeout(self):
        """A first-attempt hang that then succeeds recovers within the retry
        budget instead of failing."""
        from multiplai_core import model_client

        state = {"calls": 0}
        mock = MagicMock()
        mock.AssistantMessage = _FakeAssistantMessage
        mock.TextBlock = _FakeTextBlock
        mock.ClaudeAgentOptions = MagicMock(side_effect=lambda **kw: MagicMock())

        async def _agen(prompt, options):
            state["calls"] += 1
            if state["calls"] == 1:
                await asyncio.sleep(10)  # first attempt hangs
            yield _FakeAssistantMessage([_FakeTextBlock("recovered")])

        mock.query = MagicMock(side_effect=_agen)

        with patch.dict(sys.modules, {"claude_agent_sdk": mock}), \
                patch.object(model_client, "_SDK_CALL_TIMEOUT_S", 0.05), \
                patch.object(model_client, "_SDK_RETRY_BACKOFF_S", 0.0):
            client = model_client.AgentSDKClient()

            async def _test():
                result = await client.query("sys", [{"role": "user", "content": "hi"}])
                assert result.content == "recovered"
                assert state["calls"] == 2

            asyncio.run(_test())


class TestEnvFloat:
    """The SDK timeout constant is parsed at import — bad values must not
    crash `import multiplai_core` for every consumer."""

    def test_env_float_valid(self):
        from multiplai_core.model_client import _env_float
        with patch.dict(os.environ, {"X_TIMEOUT": "1800"}):
            assert _env_float("X_TIMEOUT", 600.0) == 1800.0

    def test_env_float_garbage_returns_default(self):
        from multiplai_core.model_client import _env_float
        with patch.dict(os.environ, {"X_TIMEOUT": "abc"}):
            assert _env_float("X_TIMEOUT", 600.0) == 600.0

    def test_env_float_empty_returns_default(self):
        from multiplai_core.model_client import _env_float
        with patch.dict(os.environ, {"X_TIMEOUT": ""}):
            assert _env_float("X_TIMEOUT", 600.0) == 600.0

    def test_env_float_unset_returns_default(self):
        from multiplai_core.model_client import _env_float
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("X_TIMEOUT", None)
            assert _env_float("X_TIMEOUT", 600.0) == 600.0


class TestProviderSeam:
    """The registry that lets a non-Anthropic reviewer backend plug in.

    The seam ships without any such backend on purpose — picking a provider
    means picking whose API key and whose bill — so these tests register a
    stub, which is exactly what an out-of-tree backend does.
    """

    @pytest.fixture(autouse=True)
    def _restore_registry(self):
        from multiplai_core import model_client as mc
        before = dict(mc._PROVIDERS)
        yield
        mc._PROVIDERS.clear()
        mc._PROVIDERS.update(before)

    def test_anthropic_registered_by_default(self):
        from multiplai_core.model_client import registered_providers
        assert "anthropic" in registered_providers()

    def test_bare_model_string_routes_to_anthropic(self):
        from multiplai_core.model_client import AgentSDKClient, create_client_for
        mock_sdk = MagicMock()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            client = asyncio.run(create_client_for("claude-opus-4-8"))
        assert isinstance(client, AgentSDKClient)

    def test_unknown_provider_raises_with_registered_names(self):
        from multiplai_core.model_client import UnknownProviderError, create_client_for

        with pytest.raises(UnknownProviderError) as exc:
            asyncio.run(create_client_for("openai:gpt-5"))
        # The usual cause is a config typo, so the message must name what IS
        # registered rather than only what is missing.
        assert "openai" in str(exc.value)
        assert "anthropic" in str(exc.value)

    def test_registered_sync_factory_is_used(self):
        from multiplai_core.model_client import (
            ModelResponse, create_client_for, register_provider,
        )

        class _Stub:
            def __init__(self, spec):
                self.spec = spec

            async def query(self, system, messages, *, model=None, max_tokens=None,
                            temperature=1.0, effort=None):
                return ModelResponse(content="stub")

        register_provider("acme", lambda spec, **kw: _Stub(spec))
        client = asyncio.run(create_client_for("acme:m-1"))
        assert client.spec.model == "m-1"
        assert asyncio.run(client.query("s", [])).content == "stub"

    def test_registered_async_factory_is_awaited(self):
        from multiplai_core.model_client import create_client_for, register_provider

        class _Stub:
            def __init__(self, spec):
                self.spec = spec

        async def _factory(spec, **kw):
            return _Stub(spec)

        register_provider("acme", _factory)
        assert asyncio.run(create_client_for("acme:m-2")).spec.model == "m-2"

    def test_factory_receives_component_and_api_key(self):
        from multiplai_core.model_client import create_client_for, register_provider
        seen = {}

        def _factory(spec, *, api_key=None, component="model_client"):
            seen.update(spec=spec, api_key=api_key, component=component)
            return MagicMock()

        register_provider("acme", _factory)
        asyncio.run(create_client_for("acme:m", api_key="k", component="buildme"))
        assert seen["api_key"] == "k" and seen["component"] == "buildme"
        assert seen["spec"].qualified == "acme:m"

    def test_model_spec_object_accepted(self):
        from multiplai_core.env import ModelSpec
        from multiplai_core.model_client import create_client_for, register_provider
        register_provider("acme", lambda spec, **kw: MagicMock(name=spec.model))
        assert asyncio.run(create_client_for(ModelSpec("acme", "m"))) is not None

    def test_unregister_removes_the_provider(self):
        from multiplai_core.model_client import (
            UnknownProviderError, create_client_for, register_provider,
            registered_providers, unregister_provider,
        )
        register_provider("acme", lambda spec, **kw: MagicMock())
        assert "acme" in registered_providers()
        unregister_provider("acme")
        with pytest.raises(UnknownProviderError):
            asyncio.run(create_client_for("acme:m"))


class TestEffortForwarding:
    """Effort is the second tuning axis; it must reach the SDK when set."""

    def test_effort_forwarded_to_run_agent(self):
        from multiplai_core.model_client import AgentSDKClient
        mock_sdk = MagicMock()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            client = AgentSDKClient()
        with patch("multiplai_core.model_client.run_agent", new_callable=AsyncMock) as ra:
            ra.return_value = types.SimpleNamespace(text="ok")
            asyncio.run(client.query("sys", [{"role": "user", "content": "hi"}], effort="low"))
        assert ra.call_args.kwargs["effort"] == "low"

    def test_effort_none_by_default(self):
        from multiplai_core.model_client import AgentSDKClient
        mock_sdk = MagicMock()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            client = AgentSDKClient()
        with patch("multiplai_core.model_client.run_agent", new_callable=AsyncMock) as ra:
            ra.return_value = types.SimpleNamespace(text="ok")
            asyncio.run(client.query("sys", [{"role": "user", "content": "hi"}]))
        # None means run_agent omits the option entirely — older SDKs keep working.
        assert ra.call_args.kwargs["effort"] is None


class TestPerCallTimeout:
    """`query(timeout_s=…)` escalates the ceiling for ONE call.

    The module default is a global read from the env; patching it to raise the
    timeout for a single oversized request would change the ceiling for every
    other call in flight under ``asyncio.gather``. This keyword is the
    non-racy route.
    """

    @staticmethod
    def _client():
        from multiplai_core.model_client import AgentSDKClient
        mock_sdk = MagicMock()
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
            return AgentSDKClient()

    def test_timeout_s_reaches_run_agent(self):
        client = self._client()
        with patch("multiplai_core.model_client.run_agent", new_callable=AsyncMock) as ra:
            ra.return_value = types.SimpleNamespace(text="ok")
            asyncio.run(client.query(
                "sys", [{"role": "user", "content": "hi"}], timeout_s=1800.0,
            ))
        assert ra.call_args.kwargs["timeout_s"] == 1800.0

    def test_omitting_it_preserves_the_module_default(self):
        from multiplai_core import model_client

        client = self._client()
        with patch("multiplai_core.model_client.run_agent", new_callable=AsyncMock) as ra:
            ra.return_value = types.SimpleNamespace(text="ok")
            asyncio.run(client.query("sys", [{"role": "user", "content": "hi"}]))
        assert ra.call_args.kwargs["timeout_s"] == model_client._SDK_CALL_TIMEOUT_S

    def test_default_still_read_at_call_time(self):
        """Existing callers patch the module global; that must keep working."""
        from multiplai_core import model_client

        client = self._client()
        with patch("multiplai_core.model_client.run_agent", new_callable=AsyncMock) as ra, \
                patch.object(model_client, "_SDK_CALL_TIMEOUT_S", 123.0):
            ra.return_value = types.SimpleNamespace(text="ok")
            asyncio.run(client.query("sys", [{"role": "user", "content": "hi"}]))
        assert ra.call_args.kwargs["timeout_s"] == 123.0

    def test_explicit_value_wins_over_the_module_default(self):
        from multiplai_core import model_client

        client = self._client()
        with patch("multiplai_core.model_client.run_agent", new_callable=AsyncMock) as ra, \
                patch.object(model_client, "_SDK_CALL_TIMEOUT_S", 123.0):
            ra.return_value = types.SimpleNamespace(text="ok")
            asyncio.run(client.query(
                "sys", [{"role": "user", "content": "hi"}], timeout_s=7.0,
            ))
        assert ra.call_args.kwargs["timeout_s"] == 7.0

    def test_keyword_only_and_defaults_to_none(self):
        """Additive keyword-only argument — no positional slot was taken, so
        every existing call site is unaffected."""
        from multiplai_core.model_client import (
            AgentSDKClient, AnthropicAPIClient, ModelClient,
        )
        for meth in (
            ModelClient.query, AgentSDKClient.query, AnthropicAPIClient.query,
        ):
            p = inspect.signature(meth).parameters["timeout_s"]
            assert p.kind is inspect.Parameter.KEYWORD_ONLY
            assert p.default is None

    def test_api_client_accepts_it_and_does_not_forward_it(self):
        """Parity only: the fallback client has its own transport timeouts, so
        a caller that escalates must not get a TypeError or a bogus API param."""
        from multiplai_core.model_client import AnthropicAPIClient

        block = MagicMock()
        block.type = "text"
        block.text = "ok"
        response = MagicMock()
        response.content = [block]
        mock_async_client = MagicMock()
        mock_async_client.messages.create = AsyncMock(return_value=response)
        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_async_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            client = AnthropicAPIClient("sk-test")

            async def _test():
                result = await client.query(
                    "sys", [{"role": "user", "content": "hi"}], timeout_s=1800.0,
                )
                assert result.content == "ok"
                assert "timeout_s" not in (
                    mock_async_client.messages.create.call_args.kwargs
                )

            asyncio.run(_test())


class TestPromptCaching:
    """The direct-API path sent `system=` as a bare string, which is never
    cached — every call re-paid full input price for an identical prompt."""

    def test_short_system_prompt_passes_through_unchanged(self):
        from multiplai_core.model_client import cacheable_system
        assert cacheable_system("You are helpful.") == "You are helpful."

    def test_empty_system_passes_through(self):
        from multiplai_core.model_client import cacheable_system
        assert cacheable_system("") == ""

    def test_long_system_prompt_gets_a_cache_breakpoint(self):
        from multiplai_core.model_client import (
            MIN_CACHEABLE_SYSTEM_BYTES, cacheable_system,
        )
        big = "x" * (MIN_CACHEABLE_SYSTEM_BYTES + 1)
        out = cacheable_system(big)
        assert out == [{"type": "text", "text": big,
                        "cache_control": {"type": "ephemeral"}}]

    def test_threshold_is_measured_in_bytes_not_characters(self):
        """Multi-byte text must not be counted as cacheable on character
        count alone — the API minimum is about tokens, and a conservative
        bytes floor only ever skips prompts that could not be cached."""
        from multiplai_core.model_client import (
            MIN_CACHEABLE_SYSTEM_BYTES, cacheable_system,
        )
        # Under the byte floor despite a large character count would be wrong;
        # 'é' is 2 bytes, so this crosses the floor with half the characters.
        text = "é" * (MIN_CACHEABLE_SYSTEM_BYTES // 2 + 1)
        assert isinstance(cacheable_system(text), list)

    def test_query_sends_the_breakpoint_to_the_api(self):
        from multiplai_core.model_client import (
            MIN_CACHEABLE_SYSTEM_BYTES, AnthropicAPIClient,
        )
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "ok"
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_async_client = MagicMock()
        mock_async_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_async_client

        big = "y" * (MIN_CACHEABLE_SYSTEM_BYTES + 10)
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            client = AnthropicAPIClient("sk-test-key")

            async def _test():
                await client.query(big, [{"role": "user", "content": "hi"}])
                sent = mock_async_client.messages.create.call_args.kwargs["system"]
                assert sent[0]["cache_control"] == {"type": "ephemeral"}

            asyncio.run(_test())
