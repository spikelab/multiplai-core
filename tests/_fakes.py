"""Fake ``claude_agent_sdk`` / ``anthropic`` harnesses shared by the tests.

Kept in its own module rather than in ``conftest.py``: importing names *from*
a conftest only resolves under pytest's legacy ``prepend`` import mode, so
``pytest --import-mode=importlib`` (the mode pytest documents as preferred)
failed collection on every module that did it. ``conftest.py`` puts this
directory on ``sys.path``, which it can do in any mode because pytest imports
conftest files itself.
"""

from unittest.mock import AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# Fake claude_agent_sdk harness (real classes so isinstance() works).
# One definition for both test_agent_runner and test_model_client — the two
# suites exercise the same message-consumption code in run_agent, and two
# hand-maintained doubles drifted against one real SDK surface.
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

    ``sdk.query(prompt=..., options=...)`` is an async generator yielding the
    given *messages*. The mock exposes the types used by ``isinstance()``
    checks and can simulate CLI stderr via the ``stderr`` options callback.
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


# ---------------------------------------------------------------------------
# Mock `anthropic` module (for AnthropicAPIClient tests)
# ---------------------------------------------------------------------------


def _mock_anthropic(*texts: str, blocks: list | None = None):
    """Mock ``anthropic`` module whose ``messages.create`` returns *texts*.

    Each text becomes one ``type="text"`` content block; pass *blocks* to
    supply custom block mocks (e.g. a ``thinking`` block) instead. Returns
    ``(mock_module, mock_async_client)`` — patch ``sys.modules["anthropic"]``
    with the module, assert against the client's ``messages.create``.
    """
    if blocks is None:
        blocks = []
        for text in texts:
            block = MagicMock()
            block.type = "text"
            block.text = text
            blocks.append(block)
    response = MagicMock()
    response.content = list(blocks)
    async_client = MagicMock()
    async_client.messages.create = AsyncMock(return_value=response)
    module = MagicMock()
    module.AsyncAnthropic.return_value = async_client
    return module, async_client

