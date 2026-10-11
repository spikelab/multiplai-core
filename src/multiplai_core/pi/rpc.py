"""Pi's JSONL RPC transport.

Ported from DolceEvals ``piharness/rpc.py`` (DolceTech/DolceEvals@d755fcc) for
:mod:`multiplai_core.pi.backend`. Standard library only. Mentions of
``supervisor.py`` and "the trial" below refer to that repo; here the backend's
own run loop plays the supervisor's part, and nothing steers a run. One change
from the original: :class:`SubprocessTransport` drains the child's stderr into
a bounded ring (``stderr_tail``) so a chatty Pi cannot fill the pipe and stall,
and a failed run can say what Pi printed.

`pi --mode rpc` speaks newline-delimited JSON over stdin/stdout: commands in,
events out. This module is the wire layer -- framing, encoding, and turning
raw lines into `events.Event` objects. Policy (when to steer, when to abort)
lives in `supervisor.py`.

## Two framing rules that are easy to get wrong

**LF only, on the way out.** Pi's reader splits on `\\n`. A `\\r\\n` terminator
leaves a stray `\\r` inside the JSON text of the next frame on some readers, and
the failure is intermittent rather than immediate. `encode_command` emits exactly
one `\\n` and nothing else, and JSON escaping guarantees a multi-line steer
message still occupies exactly one frame.

**Partial reads, on the way in.** A pipe read returns whatever bytes are
available, not whole lines: one JSON object can arrive across three reads, and
three objects can arrive in one. `LineFramer` buffers across reads. Without it
the harness loses events under load -- exactly when the interesting runs happen.

## Interjection

`steer` injects operator text into a running turn (plan §2.3). Every steer must
be recorded on the trial: a steered trial is excluded from headline scores, and
that exclusion is only possible if the interjection was captured. The supervisor
owns that bookkeeping; this module only guarantees delivery.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from typing import Any, AsyncIterator, Protocol

from .events import Event, parse_line

# Pi's steering modes are "all" and "one-at-a-time" (docs/rpc.md, verified
# against pi 0.83.0). We default to "all": deliver every queued steer after the
# current assistant turn, so an operator interjection is never silently held.
DEFAULT_STEERING_MODE = "all"


class Transport(Protocol):
    """The byte-level channel to a Pi process.

    A protocol rather than a concrete class so tests can substitute an
    in-process fake -- there is no Pi binary in the Claude container, and even
    where there is, driving a real subprocess makes bound-enforcement tests
    slow and flaky.
    """

    async def send(self, data: bytes) -> None: ...

    def stream(self) -> AsyncIterator[bytes]: ...

    async def close(self) -> None: ...


# ------------------------------------------------------------------ framing


class LineFramer:
    """Reassembles LF-delimited lines from arbitrary byte chunks."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> list[str]:
        """Return every *complete* line in `chunk`, holding any remainder."""
        self._buf.extend(chunk)
        if b"\n" not in self._buf:
            return []
        *complete, remainder = bytes(self._buf).split(b"\n")
        self._buf = bytearray(remainder)
        return [line.rstrip(b"\r").decode("utf-8", "replace") for line in complete]

    def flush(self) -> list[str]:
        """Return a trailing unterminated line, if any.

        A process killed mid-write leaves one. Returning it rather than dropping
        it is the difference between a truncated event and a missing one.
        """
        if not self._buf:
            return []
        line = bytes(self._buf).rstrip(b"\r").decode("utf-8", "replace")
        self._buf = bytearray()
        return [line] if line else []

    @property
    def pending_bytes(self) -> int:
        return len(self._buf)


def encode_command(command: str, payload: dict[str, Any] | None = None) -> bytes:
    """Serialise one RPC command as a single LF-terminated frame.

    Pi's discriminator field is `type` (verified against pi 0.83.0 docs/rpc.md),
    e.g. `{"type": "prompt", "message": "..."}`.
    """
    obj: dict[str, Any] = {"type": command}
    if payload:
        obj.update(payload)
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    assert "\n" not in line and "\r" not in line, "JSON escaping must keep one frame"
    return line.encode("utf-8") + b"\n"


# --------------------------------------------------------------- transports


class SubprocessTransport:
    """A real `pi --mode rpc` child process."""

    STDERR_LINES = 50

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self.proc = proc
        self._stderr: deque[str] = deque(maxlen=self.STDERR_LINES)
        self._stderr_task: asyncio.Task | None = None
        if proc.stderr is not None:
            self._stderr_task = asyncio.ensure_future(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        # Chunked reads, not readline(): a single stderr line longer than the
        # stream limit makes readline() raise, and the drain must never die.
        assert self.proc.stderr is not None
        framer = LineFramer()
        while True:
            chunk = await self.proc.stderr.read(65536)
            if not chunk:
                break
            self._stderr.extend(line[:2000] for line in framer.feed(chunk))
        self._stderr.extend(line[:2000] for line in framer.flush())

    def stderr_tail(self) -> str:
        """The last lines Pi wrote to stderr, newest last."""
        return "\n".join(self._stderr)

    @classmethod
    async def spawn(
        cls, argv: list[str], *, cwd: str | None = None, env: dict[str, str] | None = None
    ) -> "SubprocessTransport":
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
        return cls(proc)

    async def send(self, data: bytes) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(data)
        await self.proc.stdin.drain()

    async def stream(self) -> AsyncIterator[bytes]:
        assert self.proc.stdout is not None
        while True:
            chunk = await self.proc.stdout.read(65536)
            if not chunk:
                return
            yield chunk

    async def close(self) -> None:
        if self.proc.stdin is not None and not self.proc.stdin.is_closing():
            self.proc.stdin.close()
        if self.proc.returncode is None:
            try:
                self.proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                self.proc.kill()
                await self.proc.wait()
        if self._stderr_task is not None:
            try:
                await asyncio.wait_for(self._stderr_task, timeout=2)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._stderr_task.cancel()


# ---------------------------------------------------------------- the client


class PiRpc:
    """Command/event client over a `Transport`.

    Records every command sent in `sent`, so a test -- or a post-hoc audit of a
    surprising trial -- can see exactly what the harness asked Pi to do.
    """

    def __init__(self, transport: Any) -> None:
        self.transport = transport
        self.framer = LineFramer()
        self.sent: list[dict[str, Any]] = []
        self._seq = 0

    # --- commands --------------------------------------------------------

    async def _command(self, command: str, payload: dict[str, Any] | None = None) -> None:
        self.sent.append({"type": command, **(payload or {})})
        await self.transport.send(encode_command(command, payload))

    async def prompt(self, text: str) -> None:
        await self._command("prompt", {"message": text})

    async def steer(self, text: str) -> None:
        """Inject operator text into the running turn (plan §2.3)."""
        await self._command("steer", {"message": text})

    async def follow_up(self, text: str) -> None:
        await self._command("follow_up", {"message": text})

    async def abort(self) -> None:
        await self._command("abort")

    async def set_steering_mode(self, mode: str = DEFAULT_STEERING_MODE) -> None:
        await self._command("set_steering_mode", {"mode": mode})

    async def set_follow_up_mode(self, mode: str) -> None:
        await self._command("set_follow_up_mode", {"mode": mode})

    # --- events ----------------------------------------------------------

    async def events(self) -> AsyncIterator[Event]:
        """Yield parsed events until the transport closes.

        Never raises on malformed input: `parse_line` is total, so a line Pi
        emits that we cannot interpret becomes an UNKNOWN event with its payload
        intact rather than ending the run.
        """
        async for chunk in self.transport.stream():
            for line in self.framer.feed(chunk):
                if not line.strip():
                    continue
                yield self._parse(line)
        for line in self.framer.flush():
            if line.strip():
                yield self._parse(line)

    def _parse(self, line: str) -> Event:
        ev = parse_line(self._seq, line)
        self._seq += 1
        return ev

    async def close(self) -> None:
        await self.transport.close()
