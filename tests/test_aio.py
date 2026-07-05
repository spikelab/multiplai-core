"""Tests for multiplai_core.aio.hard_timeout."""

import asyncio

import pytest

from multiplai_core.aio import hard_timeout


def test_returns_result_when_fast():
    async def _f():
        return 42

    assert asyncio.run(hard_timeout(_f(), 1.0)) == 42


def test_raises_timeout_on_slow():
    async def _slow():
        await asyncio.sleep(10)
        return 1

    async def _run():
        with pytest.raises(asyncio.TimeoutError):
            await hard_timeout(_slow(), 0.05)

    asyncio.run(_run())


def test_returns_promptly_even_if_cleanup_blocks():
    # A coroutine whose cancellation cleanup blocks must not stall the caller.
    async def _wedged():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            # Simulate slow teardown; hard_timeout must not await this.
            await asyncio.sleep(10)
            raise

    async def _run():
        loop = asyncio.get_event_loop()
        start = loop.time()
        with pytest.raises(asyncio.TimeoutError):
            await hard_timeout(_wedged(), 0.05)
        # Returned well before the wedged cleanup's 10s.
        assert loop.time() - start < 2.0

    asyncio.run(_run())
