"""Tests for the pi agent backend (multiplai_core.pi.backend).

No `pi` binary and no network: every run goes through an in-process stub
transport that replays scripted Pi events. Event shapes follow a recorded
`pi --mode rpc` run (DolceEvals evals archive, 2026-08-15).
"""

import asyncio
import json
import subprocess

import pytest

from multiplai_core import agent_runner
from multiplai_core.agent_runner import AgentRunError, AgentRunTimeout, run_agent
from multiplai_core.pi import backend as pb
from multiplai_core.pi.events import Kind, parse_event
from multiplai_core.pi.rpc import LineFramer, encode_command

SPEC = "pi:deepseek/deepseek-v4-flash@fireworks"


def _run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------ event helpers


def _usage(i=0, o=0, cr=0, cw=0, cost=0.0):
    return {"input": i, "output": o, "cacheRead": cr, "cacheWrite": cw,
            "totalTokens": i + o + cr + cw,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                     "total": cost}}


def _assistant_end(text="", *, usage=None, stop="stop", error=None, tool_calls=()):
    content = [{"type": "text", "text": text}] if text else []
    content += [{"type": "toolCall", "id": f"t{n}", "name": name, "arguments": args}
                for n, (name, args) in enumerate(tool_calls)]
    msg = {"role": "assistant", "content": content, "usage": usage or _usage(),
           "stopReason": stop}
    if error:
        msg["errorMessage"] = error
    return {"type": "message_end", "message": msg}


def _tool(name, args):
    return [{"type": "tool_execution_start", "toolCallId": "c", "toolName": name, "args": args},
            {"type": "tool_execution_end", "toolCallId": "c", "toolName": name,
             "result": {"content": []}, "isError": False}]


TURN_END = {"type": "turn_end"}
SETTLED = {"type": "agent_settled"}


class StubTransport:
    """Replays *script* after Pi receives the prompt; records what was sent."""

    def __init__(self, script, *, hang=False, chunk=7, fail_on=None, wedge_on=None):
        self.script = script
        self.hang = hang
        self.chunk = chunk
        self.fail_on = fail_on or {}
        """Command type -> exception raised when it is written (Pi already exited)."""
        self.wedge_on = wedge_on or ()
        """Command types whose write never completes (Pi stopped reading stdin)."""
        self.sent = []
        self.closed = False
        self._prompted = asyncio.Event()

    async def send(self, data):
        frame = json.loads(data.decode())
        if frame["type"] in self.fail_on:
            raise self.fail_on[frame["type"]]
        self.sent.append(frame)
        if frame["type"] in self.wedge_on:
            await asyncio.Event().wait()
        if frame["type"] == "prompt":
            self._prompted.set()

    async def stream(self):
        await self._prompted.wait()
        blob = b"".join(json.dumps(e).encode() + b"\n" for e in self.script)
        # Odd-sized chunks: a frame split across reads must still parse.
        for i in range(0, len(blob), self.chunk):
            yield blob[i:i + self.chunk]
        if self.hang:
            await asyncio.Event().wait()

    async def close(self):
        self.closed = True

    def stderr_tail(self):
        return "stub stderr"


@pytest.fixture
def stub(monkeypatch, tmp_path):
    """Install a stub transport factory; returns a dict that captures the run."""
    seen = {"script": [], "hang": False, "fail_on": None, "wedge_on": None, "opens": 0}

    async def fake_open(argv, *, cwd, env):
        seen["argv"], seen["cwd"], seen["env"] = argv, cwd, env
        seen["opens"] += 1
        t = StubTransport(seen["script"], hang=seen["hang"], fail_on=seen["fail_on"],
                          wedge_on=seen["wedge_on"])
        seen["transport"] = t
        return t

    monkeypatch.setattr(pb, "_open_transport", fake_open)
    monkeypatch.chdir(tmp_path)
    # A run with no cwd defaults to $CLAUDE_CONFIG_DIR/hook-sessions; keep it
    # out of the real config dir.
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
    return seen


# ----------------------------------------------------------------- parsing


class TestParsePiSpec:
    def test_valid(self):
        assert pb.parse_pi_spec("pi:deepseek/deepseek-v4-flash-0731@fireworks") == (
            "deepseek/deepseek-v4-flash-0731", "fireworks")

    def test_last_at_splits(self):
        assert pb.parse_pi_spec("pi:org/m@v2@prov") == ("org/m@v2", "prov")

    @pytest.mark.parametrize("bad", [
        "claude", "pi:model-no-provider", "pi:@fireworks", "pi:model@",
    ])
    def test_malformed(self, bad):
        with pytest.raises(pb.PiSpecError):
            pb.parse_pi_spec(bad)

    def test_malformed_spec_through_run_agent_is_an_agent_run_error(self):
        with pytest.raises(AgentRunError):
            _run(run_agent("p", model="pi:no-provider"))


class TestToolMap:
    def test_translates_dedupes_and_drops_unknown(self):
        assert pb.pi_tools(["Read", "Grep", "Glob", "Bash", "Read", "WebFetch"]) == [
            "read", "grep", "find", "bash"]

    def test_disallowed_removes_from_allow_list(self):
        assert pb.pi_tools(["Read", "Bash"], ["Bash"]) == ["read"]

    def test_none_and_empty_are_no_tools(self):
        assert pb.pi_tools(None) == [] and pb.pi_tools([]) == []


class TestArgv:
    def test_tools_effort_and_model(self):
        argv = pb.build_argv("pi", "m/x", "prov", tools=["read", "write"], effort="high")
        assert argv[:4] == ["pi", "--mode", "rpc", "--no-session"]
        assert argv[argv.index("--provider") + 1] == "prov"
        assert argv[argv.index("--model") + 1] == "m/x"
        assert argv[argv.index("-t") + 1] == "read,write"
        assert argv[argv.index("--thinking") + 1] == "high"
        assert "--no-tools" not in argv

    def test_no_tool_call_has_no_tool_allow_list(self):
        argv = pb.build_argv("pi", "m", "p", tools=[], effort=None)
        assert "--no-tools" in argv and "-t" not in argv

    def test_effort_none_passes_no_thinking(self):
        assert "--thinking" not in pb.build_argv("pi", "m", "p", tools=[], effort=None)

    def test_disabled_thinking_without_effort_is_off(self):
        argv = pb.build_argv("pi", "m", "p", tools=[], effort=None,
                             thinking={"type": "disabled"})
        assert argv[argv.index("--thinking") + 1] == "off"

    def test_gate_extension(self):
        argv = pb.build_argv("pi", "m", "p", tools=["bash"], effort=None, gate_ext="/g.ts")
        assert argv[-3:] == ["--no-extensions", "--extension", "/g.ts"]


class TestFraming:
    def test_frame_split_across_reads(self):
        f = LineFramer()
        assert f.feed(b'{"type":"tu') == []
        assert f.feed(b'rn_end"}\n{"a"') == ['{"type":"turn_end"}']
        assert f.flush() == ['{"a"']

    def test_encode_is_one_lf_frame(self):
        data = encode_command("prompt", {"message": "a\nb"})
        assert data.endswith(b"\n") and data.count(b"\n") == 1


# ---------------------------------------------------------------- results


def _ev(seq, payload):
    return parse_event(seq, payload)


class TestAssistantText:
    def test_last_rendition_per_id_wins_in_first_seen_order(self):
        def m(seq, mid, text):
            return _ev(seq, {"type": "message_update", "id": mid, "message": {
                "role": "assistant", "content": [{"type": "text", "text": text}]}})
        events = [m(0, "m1", "part"), m(1, "m1", "partial answer"),
                  m(2, "m2", "second message"), m(3, "m1", "final answer")]
        assert pb.assistant_text(events) == "final answer\nsecond message"

    def test_ignores_user_and_textless_messages(self):
        events = [
            _ev(0, {"type": "message_end", "id": "u", "message": {
                "role": "user", "content": [{"type": "text", "text": "user text"}]}}),
            _ev(1, {"type": "tool_execution_start"}),
            _ev(2, {"type": "message_end", "id": "a", "message": {
                "role": "assistant", "content": [{"type": "text", "text": "answer"}]}}),
        ]
        assert pb.assistant_text(events) == "answer"


class TestUsage:
    def test_sums_assistant_message_end_only(self):
        events = [_ev(i, e) for i, e in enumerate([
            {"type": "message_start", "message": {"role": "assistant", "usage": _usage(9, 9)}},
            _assistant_end("a", usage=_usage(100, 10, cr=50, cw=5, cost=0.01)),
            {"type": "turn_end", "message": {"role": "assistant",
                                             "usage": _usage(100, 10, cost=0.01)}},
            _assistant_end("b", usage=_usage(20, 2, cost=0.002)),
            {"type": "message_end", "message": {"role": "toolResult", "usage": _usage(7, 7)}},
        ])]
        usage, reported = pb.usage_from_events(events)
        assert (usage.input_tokens, usage.output_tokens) == (120, 12)
        assert (usage.cache_read_tokens, usage.cache_creation_tokens) == (50, 5)
        assert usage.cost_usd == pytest.approx(0.012)
        assert reported

    def test_no_cost_is_zero_and_unreported(self):
        usage, reported = pb.usage_from_events([_ev(0, _assistant_end("a", usage=_usage(5, 1)))])
        assert usage.cost_usd == 0 and not reported


class TestFilesChanged:
    def _events(self):
        return [_ev(0, e) for e in _tool("write", {"path": "src/made_by_event.py"})] + [
            _ev(2, e) for e in _tool("read", {"path": "ignored.py"})]

    def test_write_and_edit_targets_in_order_without_duplicates(self):
        events = self._events() + [_ev(4, e) for e in _tool("edit", {"path": "b.py"})] + [
            _ev(6, e) for e in _tool("write", {"path": "src/made_by_event.py"})]
        assert pb.files_changed(events) == ["src/made_by_event.py", "b.py"]

    def test_bash_writes_are_not_reported(self):
        # Same as the SDK path: only Write/Edit file paths count.
        events = [_ev(0, e) for e in _tool("bash", {"command": "echo x > c.py"})]
        assert pb.files_changed(events) == []


# ------------------------------------------------------------ whole runs


class TestRun:
    def test_result_shape_matches_the_sdk_path(self, stub):
        stub["script"] += [
            _assistant_end(usage=_usage(10, 2, cost=0.001),
                           tool_calls=[("write", {"path": "a.py"})], stop="toolUse"),
            *_tool("write", {"path": "a.py", "content": "x"}),
            TURN_END,
            _assistant_end("done", usage=_usage(5, 1, cost=0.0005)),
            TURN_END, SETTLED,
        ]
        r = _run(run_agent("go", model=SPEC, allowed_tools=["Write", "Read"], max_turns=5,
                           effort="medium"))
        assert isinstance(r, agent_runner.AgentRunResult)
        assert r.text == "done" and r.turns == 2
        assert r.files_changed == ["a.py"]
        assert r.tool_calls == (agent_runner.ToolCall("Write", {"path": "a.py", "content": "x"}),)
        assert (r.usage.input_tokens, r.usage.output_tokens) == (15, 3)
        assert r.usage.cost_usd == pytest.approx(0.0015)
        assert r.stderr_tail == ""
        argv = stub["argv"]
        assert argv[argv.index("-t") + 1] == "write,read"
        assert argv[argv.index("--thinking") + 1] == "medium"
        assert stub["transport"].sent[0] == {"type": "prompt", "message": "go"}
        assert stub["transport"].closed
        assert stub["env"]["_HOOK_CHILD_SESSION"] == "1"

    def test_files_changed_reports_only_the_agents_write_and_edit_targets(
            self, stub, monkeypatch, tmp_path):
        # The run's cwd is a subdirectory of a repo that other processes are
        # also changing: one creates a file elsewhere in the repo while the run
        # is going, and one re-edits a file that was already dirty. Neither is
        # the agent's write/edit, so neither is reported, the same contract as
        # the SDK path (file_path args of Write/Edit only).
        repo = tmp_path / "repo"
        work = repo / "sub"
        work.mkdir(parents=True)
        subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
        (work / "dirty.py").write_text("x = 1\n")
        opened = pb._open_transport

        async def open_and_race(argv, *, cwd, env):
            (repo / "other_process.py").write_text("y = 2\n")
            (work / "dirty.py").write_text("x = 3\n")
            return await opened(argv, cwd=cwd, env=env)

        monkeypatch.setattr(pb, "_open_transport", open_and_race)
        stub["script"] += [
            _assistant_end(tool_calls=[("edit", {"path": "a.py"})], stop="toolUse"),
            *_tool("edit", {"path": "a.py", "oldText": "a", "newText": "b"}),
            TURN_END, _assistant_end("done"), TURN_END, SETTLED,
        ]
        r = _run(run_agent("go", model=SPEC, allowed_tools=["Edit"], max_turns=5, cwd=work))
        assert r.files_changed == ["a.py"]

    def test_default_cwd_is_the_hook_session_dir(self, stub, monkeypatch, tmp_path):
        # Same default as the SDK path: with no cwd the run must not edit or
        # pick up context from whatever project the caller is in.
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
        stub["script"] += [_assistant_end("hi"), TURN_END, SETTLED]
        _run(run_agent("p", model=SPEC))
        assert stub["cwd"] == str(tmp_path / "cfg" / "hook-sessions")
        assert (tmp_path / "cfg" / "hook-sessions").is_dir()

    def test_no_tool_call(self, stub):
        stub["script"] += [_assistant_end("hi"), TURN_END, SETTLED]
        r = _run(run_agent("p", model=SPEC))
        assert r.text == "hi"
        assert "--no-tools" in stub["argv"] and "-t" not in stub["argv"]
        assert "--thinking" not in stub["argv"]

    def test_system_prompt_leads_the_message(self, stub):
        stub["script"] += [_assistant_end("ok"), SETTLED]
        _run(run_agent("task", model=SPEC, system_prompt="be terse"))
        msg = stub["transport"].sent[0]["message"]
        assert msg.startswith("be terse") and msg.endswith("task")

    def test_max_turns_aborts_and_returns(self, stub):
        stub["script"] += [_assistant_end("one"), TURN_END, _assistant_end("two"), TURN_END,
                           _assistant_end("late"), SETTLED]
        r = _run(run_agent("p", model=SPEC, max_turns=1))
        # abort goes out once, at the first turn_end; the turns Pi finishes
        # while the abort is in flight still count.
        assert stub["transport"].sent.count({"type": "abort"}) == 1
        assert r.turns == 2
        assert "late" in r.text  # in-flight events after abort are kept

    def test_timeout_raises_with_partial(self, stub):
        stub["script"] += [_assistant_end("so far"), TURN_END]
        stub["hang"] = True
        with pytest.raises(AgentRunTimeout) as exc:
            _run(run_agent("p", model=SPEC, timeout_s=0.3, max_turns=10))
        assert exc.value.partial.text == "so far"
        assert exc.value.stderr_tail == "stub stderr"
        assert {"type": "abort"} in stub["transport"].sent

    def test_timeout_is_hard_when_abort_wedges(self, stub, monkeypatch):
        # Pi stopped reading stdin: the abort write never completes. The run
        # must still end close to timeout_s, not hang on the abort.
        monkeypatch.setattr(pb, "ABORT_WRITE_TIMEOUT_S", 0.2, raising=False)
        stub["script"] += [_assistant_end("so far"), TURN_END]
        stub["hang"] = True
        stub["wedge_on"] = ("abort",)

        async def bounded():
            return await asyncio.wait_for(
                run_agent("p", model=SPEC, timeout_s=0.3, max_turns=10), timeout=5)

        with pytest.raises(AgentRunTimeout) as exc:
            _run(bounded())
        assert exc.value.partial.text == "so far"
        assert stub["transport"].closed

    @pytest.mark.parametrize("err", [BrokenPipeError(32, "Broken pipe"),
                                     ConnectionResetError("Connection lost")])
    def test_pipe_error_on_prompt_raises_agent_run_error(self, stub, err):
        # e.g. a model missing from models.json: Pi exits before reading the prompt.
        stub["fail_on"] = {"prompt": err}
        with pytest.raises(AgentRunError) as exc:
            _run(run_agent("p", model=SPEC, max_attempts=2, retry_backoff_s=0))
        assert type(exc.value) is AgentRunError
        assert exc.value.stderr_tail == "stub stderr"
        assert exc.value.partial is not None
        assert exc.value.attempts == 2 and stub["opens"] == 2
        assert stub["transport"].closed

    def test_pipe_error_on_max_turns_abort_keeps_partial(self, stub):
        stub["script"] += [_assistant_end("one"), TURN_END]
        stub["fail_on"] = {"abort": BrokenPipeError(32, "Broken pipe")}
        with pytest.raises(AgentRunError) as exc:
            _run(run_agent("p", model=SPEC, max_turns=1))
        assert exc.value.partial.text == "one"
        assert exc.value.stderr_tail == "stub stderr"

    def test_final_model_error_raises(self, stub):
        stub["script"] += [_assistant_end(stop="error", error="429: rate limited"),
                           {"type": "auto_retry_end", "success": False}, SETTLED]
        with pytest.raises(AgentRunError) as exc:
            _run(run_agent("p", model=SPEC))
        assert "429" in str(exc.value)

    def test_exit_before_settled_raises(self, stub):
        stub["script"] += [_assistant_end("partial")]
        with pytest.raises(AgentRunError) as exc:
            _run(run_agent("p", model=SPEC))
        assert exc.value.partial.text == "partial"

    def test_retries_then_succeeds(self, stub, monkeypatch):
        scripts = [[_assistant_end("x")], [_assistant_end("ok"), SETTLED]]

        async def fake_open(argv, *, cwd, env):
            t = StubTransport(scripts.pop(0))
            stub["transport"] = t
            return t

        monkeypatch.setattr(pb, "_open_transport", fake_open)
        r = _run(run_agent("p", model=SPEC, max_attempts=2, retry_backoff_s=0))
        assert r.text == "ok"


class TestCostLedger:
    def test_records_spec_and_pi_cost(self, stub, monkeypatch):
        recs = []
        monkeypatch.setattr("multiplai_core.costing.append_records", recs.extend)
        stub["script"] += [_assistant_end("a", usage=_usage(10, 2, cost=0.25)), SETTLED]
        _run(run_agent("p", model=SPEC, component="buildme"))
        assert recs[0]["model"] == SPEC and recs[0]["component"] == "buildme"
        assert recs[0]["cost_usd"] == 0.25 and "cost_unreported" not in recs[0]

    def test_no_pi_cost_is_zero_and_says_so(self, stub, monkeypatch):
        recs = []
        monkeypatch.setattr("multiplai_core.costing.append_records", recs.extend)
        stub["script"] += [_assistant_end("a", usage=_usage(10, 2)), SETTLED]
        _run(run_agent("p", model=SPEC, component="buildme"))
        assert recs[0]["cost_usd"] == 0 and recs[0]["cost_unreported"] is True
        assert "pricing_fallback" not in recs[0]


class TestBinary:
    def test_missing_binary_raises_with_hint(self, monkeypatch, tmp_path):
        monkeypatch.setenv(pb.PI_BIN_ENV, str(tmp_path / "no-such-pi"))
        with pytest.raises(AgentRunError) as exc:
            _run(run_agent("p", model=SPEC, cwd=tmp_path))
        assert "not found" in str(exc.value) and pb.PI_BIN_ENV in str(exc.value)

    def test_gate_extension_must_exist(self, stub, monkeypatch, tmp_path):
        monkeypatch.setenv(pb.GATE_EXT_ENV, str(tmp_path / "missing.ts"))
        with pytest.raises(AgentRunError):
            _run(run_agent("p", model=SPEC))

    def test_gate_extension_is_loaded(self, stub, monkeypatch, tmp_path):
        gate = tmp_path / "gating.ts"
        gate.write_text("")
        monkeypatch.setenv(pb.GATE_EXT_ENV, str(gate))
        stub["script"] += [_assistant_end("ok"), SETTLED]
        _run(run_agent("p", model=SPEC, allowed_tools=["Bash"], cwd=tmp_path))
        assert stub["argv"][-1] == str(gate)
        assert stub["env"]["PIHARNESS_WORKSPACE"] == str(tmp_path.resolve())


def test_events_parse_tool_kind():
    ev = parse_event(0, _tool("write", {"path": "x"})[0])
    assert ev.kind is Kind.TOOL_START and ev.target == "x"
