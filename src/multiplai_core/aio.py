"""Async helpers shared across the Multiplai plugins.

``hard_timeout`` was independently ported into both buildme and deep-research
after a multi-hour ~0-CPU hang that ``asyncio.wait_for`` could not break; this
is the single source of truth.
"""

from __future__ import annotations

import asyncio


def swallow_task_result(task: "asyncio.Task") -> None:
    """Consume a cancelled/failed background task's result so asyncio does not
    log 'Task exception was never retrieved'."""
    try:
        task.result()
    except (asyncio.CancelledError, Exception):
        pass


async def hard_timeout(coro, timeout_s: float):
    """Run ``coro`` with a wall-clock timeout that ALWAYS returns control.

    Drop-in replacement for ``asyncio.wait_for`` with one critical difference:
    on timeout it cancels the task fire-and-forget and returns immediately
    instead of awaiting the cancellation. ``asyncio.wait_for`` awaits the
    cancellation it triggers, so if a subprocess (e.g. the claude-agent-sdk
    CLI) is wedged and its transport teardown never finishes, ``wait_for``
    hangs forever. ``asyncio.wait`` returns (done, pending) at the deadline and
    does NOT await pending tasks, so a wedged subprocess can leak in the
    background but never stalls the caller.

    Raises ``asyncio.TimeoutError`` on timeout (same contract as wait_for).
    """
    task = asyncio.ensure_future(coro)
    done, _ = await asyncio.wait({task}, timeout=timeout_s)
    if task not in done:
        task.cancel()  # best-effort; do NOT await — cancellation may block
        task.add_done_callback(swallow_task_result)
        raise asyncio.TimeoutError()
    return task.result()
