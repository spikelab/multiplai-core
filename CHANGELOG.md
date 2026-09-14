# Changelog

All notable changes to `multiplai-core` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with the pre-1.0 reading spelled out in
[README → Versioning and what a bump means](README.md#versioning-and-what-a-bump-means):
**a `0.x.0` minor bump may add and may break the public API; a `0.x.Y` patch bump
is fixes only and never breaks.** Read this file before taking a minor bump.

Entries are written for the **consumer** — a plugin author deciding whether to
move a PEP 723 pin. "Public API" means the names exported from `multiplai_core`
(its `__all__`); anything else is internal.

Sections start at `0.6.0`. Tags `v0.1`–`v0.5.2` predate this changelog and are
not backfilled; their contents are recoverable from `git log`.

## [Unreleased]

### Added

- **`costing.refresh_pricing()` — list prices are fetched, not hand-copied.**
  It GETs the official pricing page (`PRICING_URL`, the docs site's markdown
  rendering), parses the model table with `parse_pricing_markdown()` and
  writes `<data_dir>/costs/pricing.json`; `load_pricing()` now prefers that
  file, merged over the bundled snapshot so retired models stay priced. The
  fetch is skipped while the cached file is under a day old and never raises:
  offline, the bundled table is used and a WARNING is logged. New helpers:
  `fetch_live_pricing()`, `model_id_from_display_name()`, `tier_prices()`,
  `pricing_cache_path()`, `pricing_age_days()`, `reset_pricing_cache()`.
- **Per-tier prices in `pricing.json`.** A model entry may state `cw5m`,
  `cw1h` and `cr` per MTok explicitly; the multipliers stay as the default.
  Needed because Claude Fable 5.1 and Mythos 5.1 read cache at 0.025× input
  ($0.25/MTok), not the 0.1× every other model uses.
- **Unknown models are warned about once per process** in
  `resolve_model_rates()`, on top of the existing `pricing_fallback` flag.

### Fixed

- **Bundled `pricing.json` was stale in three ways** (verified against the
  pricing page on 2026-09-13): `claude-sonnet-5` was listed at $3/$15, the
  Sonnet 4.6 rate — the real price is $2/$10, so every Sonnet 5 record was
  overstated by 50%; `claude-opus-5` was missing and priced at fallback (the
  fallback happens to equal its $5/$25 list price, so the flag was wrong but
  the number was right); `claude-fable-5-1` / `claude-mythos-5-1` were
  missing and their cache reads were priced at $1.00/MTok instead of $0.25.
  Existing ledger records keep the price they were written with.

## [0.14.0] – 2026-08-20

### Added

- **`tests/test_package_api.py` — the public surface now has a test.**
  `__all__`, the `TYPE_CHECKING` import block and the new `_LAZY_ATTRS` map
  each spell out the exported names, and a type checker reads only the first
  two. Dropping one name from `_LAZY_ATTRS` left all 499 tests passing while
  `from multiplai_core import <name>` raised `ImportError` — so every exported
  name is now asserted to resolve both as an attribute and through a real
  import in a fresh interpreter, the three lazy submodules are asserted to
  survive a bare `import multiplai_core`, and a subprocess check pins the
  headline claim that `asyncio` stays out of `sys.modules`.

- **`hook_run()` — two log lines that make a killed hook diagnosable.** New
  `hook_run(name, logger, *, session_id=None)` context manager and the `HookRun`
  it yields (`run.stage("router")`, `run.note(injected=3)`). Wrapping a hook's
  `main()` writes `HOOK_ENTRY hook=… pid=… startup_ms=… session=…` **before** the
  body runs and `HOOK_EXIT hook=… status=… ms=… startup_ms=… pid=… session=…
  stages=a:12,b:4400` after it.

  Why you would move a pin for this: when the harness kills a hook at its
  timeout, the process cannot log its own death — so a hook whose first log line
  comes after its work leaves *no trace at all*. This happened on 2026-08-10: a
  `UserPromptSubmit` hook was killed at 30 s, the prompt lost its injected
  context, and the component log had zero lines for that session. With
  `hook_run`, an `ENTRY` with no matching `EXIT` is the tombstone, and
  `startup_ms` separates interpreter/import cost from the hook body.

  Additive and self-defending: it never raises (a broken logger is swallowed),
  never suppresses (an exception is logged `status=error err=<type>` and
  re-raised), and reads a clean `SystemExit` as success, which is how hooks
  normally end.

  Four properties a consumer's parser can rely on:

  - **Both lines are written regardless of `MULTIPLAI_LOG_LEVEL`.** They go to
    the component log through the same level-independent principle as
    `log_event()` — a level that hides the tombstone would produce the exact
    zero-lines symptom this exists to diagnose. Level still governs the other
    sinks, so an `EXIT` with `status=error` also lands in the shared
    `hook-errors.log`, and a routine one does not.
  - **Pair on `(hook, session, pid)`.** Concurrent runs of the same hook append
    to one log; `pid=` is what tells two interleaved runs apart. `session=` is
    on the `ENTRY` line too — that is the line a killed run leaves behind.
  - **`startup_ms` is measured once per process** from the kernel's process
    start time (`/proc/self/stat`, falling back to module import time where
    that is unavailable). A second `hook_run` in the same process reports the
    same value rather than accumulating uptime.
  - **The record format owns `hook`, `status`, `ms`, `startup_ms`, `session`,
    `stages`, `pid` and `err`.** A `note()` using one of those keys is emitted
    prefixed `note_`, so a line never carries two `status=` tokens.

  Known limit, stated so nobody reads more into a silent log than is there: a
  run killed *before* Python reaches `hook_run()` — during interpreter start,
  `uv` resolution, or imports — writes no line at all. Detecting that needs a
  marker written outside the process, by the launcher.

- **`thinking` pass-through on the model path.** New keyword-only
  `thinking: dict | None = None` on `run_agent`, on the `ModelClient.query`
  protocol, and on both clients. Forwarded verbatim only when set, so passing
  nothing behaves exactly as before and an older `claude-agent-sdk` without the
  option keeps working (same tolerance as `effort`).

  Why you would move a pin for this: `thinking={"type": "disabled"}` takes a
  cold single-turn SDK call from **18.4 s to 2.9 s** (measured 2026-08-09). If
  you call a model from inside a Claude Code hook, that is the difference
  between fitting the budget and being killed mid-call. It buys latency by
  giving up reasoning depth — do not set it on work where the answer's quality
  matters more than its arrival time.

  Note the asymmetry with `effort`, which `AnthropicAPIClient` ignores because
  it is an Agent-SDK session knob: `thinking` *is* a Messages API parameter, so
  that client forwards it too and both backends behave the same way.

- **Memory banks — `multiplai_core.banks`.** `memory_dir` is now the first of an
  ordered list of memory corpora. New exports: `MemoryBank`, `load_banks`,
  `personal_bank`, `bank_ref`, `split_bank_ref`, `parse_bank_ref`,
  `PERSONAL_BANK`, `PERSONAL_MODE`, `BANK_MODES`, `DEFAULT_SHARED_MODE`,
  `BANKS_FILENAME`, `is_bank_name`; new accessors `Paths.memory_banks()` and
  `Paths.memory_banks_file()`.

  `is_bank_name(name)` is exported deliberately: a consumer's write floor has to
  answer "is this ref's first segment a bank name?" with exactly the same answer
  this module gives, and a re-declared copy of the regex is a copy that can
  drift.

  **Nothing changes for a consumer that does not configure a bank.** With no
  `memory-banks.yaml`, `Paths.memory_banks()` returns exactly one bank named
  `personal` at today's `memory_dir`, and `memory_dir` itself is untouched — so
  every existing call site keeps working with no edit. That equivalence is
  asserted by a test that sets no configuration at all.

  What a consumer gains: a declared list of corpora with a trust flag on each.
  `MemoryBank.is_shared` is the single question a rendering path should ask
  before injecting content (shared bank content is authored by other people and
  belongs in an `<untrusted-content>` fence), and `accepts_direct_writes` is
  `True` for the personal bank and nothing else, in any configuration —
  a config asking for `mode: rw` on a shared bank is coerced to `propose`
  (contribute by pull request) and warns. Banks are declared in
  `<workspace>/.multiplai/memory-banks.yaml`, beside `project-map.yaml`;
  a malformed, unreadable, or partly-invalid file yields the banks it could
  parse and always at least `personal`, so a typo can neither add a bank nor
  break a session.

  Three enforcement rules are worth knowing before you configure one, because
  each is a refusal you might otherwise read as a bug:

  - **`name: personal` relocates the corpus; it does not declare one.** The
    entry is honoured only if `path:` names a directory that already exists and
    no shared bank covers, and it can never change the bank's `mode`, `remote`
    or trust flags. Anything else keeps the configured `memory_dir`. The reason
    is that `is_shared` is `False` for this bank *by name alone*, so a config
    line that relocated it freely would point the trusted corpus at somebody
    else's repo — injected unfenced, written directly.
  - **No two banks may overlap**, in either direction, including two banks that
    resolve to the same directory by different routes. The check runs once over
    the fully-resolved list *after* parsing, so its answer does not depend on
    the order entries were written in, and every path is `resolve()`d so a
    symlinked `.multiplai/` cannot hide a nested bank. Where a directory is
    claimed by both a trusted and an untrusted bank, the untrusted bank wins.
  - **`MemoryBank` refuses to exist in a self-contradicting state.**
    `MemoryBank(name="team", mode="rw")` and
    `dataclasses.replace(shared, name="personal")` now raise `ValueError`, and
    `MemoryBank.file()` raises unless given a bare filename (it used to accept
    an absolute path, which `Path.__truediv__` treats as a total override).
    `load_banks` remains the only trusted factory and cannot produce any of
    these; the guard is for hand-built banks in consumer code.

  One behavioural note on refs: `split_bank_ref` lower-cases the bank segment
  (so `Team/dev.md` reaches a bank configured as `team` instead of resolving
  nowhere), and a ref whose bank segment is *empty* — `"/dev.md"`, `"//dev.md"`
  — now names **no** bank rather than `personal`. If you relied on `"/dev.md"`
  meaning `dev.md`, it is now a refusal; that spelling bypassed the
  bare-basename check consumers apply to `"dev.md"` itself.

### Changed

- **The `[sdk]` extra now requires `claude-agent-sdk>=0.2.139` (was
  `>=0.2.116`).** `run_agent` forwards `thinking` into
  `ClaudeAgentOptions(**opts_kwargs)`, and an SDK without that field raises
  `TypeError` on every call that sets it. The floor moves that failure from
  runtime to install time, where a resolver can refuse.

  Why you would move a pin for this: it lets you delete any code that detects
  the gap at runtime. Two plugins had grown a probe that inspects signatures
  before deciding whether to pass `thinking`, plus a warning naming a fix the
  person reading it could not perform. With the floor in place, an install
  either has the field or does not resolve, so the probe has nothing to decide.

  What you must change: nothing, if you resolve fresh — 0.2.139 is the newest
  release (2026-08-14). If you pin `claude-agent-sdk` yourself below 0.2.139,
  `multiplai-core[sdk]` will no longer solve alongside it. The `<0.3` ceiling is
  unchanged.

- **`import multiplai_core` no longer imports `asyncio` (lazy submodules).**
  The asyncio-heavy modules — `agent_runner`, `aio`, `model_client` — are now
  imported lazily via PEP 562. Every exported name still resolves through
  `from multiplai_core import X` exactly as before, and
  `multiplai_core.agent_runner` attribute access still works after a bare
  `import multiplai_core`; the import just happens on first use. Measured on
  the dev tree: package import drops ~35 ms → ~19 ms, which is real money
  inside a hook budget that only needs `get_paths()` / `option()` /
  `log_event()`. **What you must change:** nothing — unless you relied on
  `import multiplai_core` alone having already imported those submodules as a
  side effect (e.g. checking `sys.modules`), which was never documented.

  One consequence worth stating outright: `MULTIPLAI_SDK_CALL_TIMEOUT_S` is
  read once when `model_client` is first imported, and that moment is now the
  first use of the model path rather than `import multiplai_core`. The
  instruction changes from "set it before import" to "set it before your first
  model call" — strictly more forgiving, but a different moment than the old
  docstring named.

- **Internal simplification pass — no public API change.** One shared
  `_env_float` (model_client now imports agent_runner's), one atomic-write
  helper in `config`, one merge-or-rename helper in `log_utils` (rotation now
  streams via `shutil.copyfileobj` instead of slurping the old log into
  memory), the workspace-base cascade in `paths.resolve()` computed once
  instead of walking the marker discovery twice, `extract_json` rebuilt on
  `json.JSONDecoder.raw_decode` (unbalanced-JSON failures now raise
  `json.JSONDecodeError` — still a `ValueError`, message text differs),
  single-pass breaker replacement in `untrusted.defang`, and dead code removed
  (`model_client._DISALLOWED_TOOLS` no-op restatement, `env._EFFORT_TIERS`
  alias, an unused `deny_list` import). A failed `write_session_state` also no
  longer leaves a stale temp file behind — the cleanup `save_yaml` already had
  now applies to both writers, and the temp name carries pid + random suffix
  so two writers of the same file cannot delete each other's in-flight temp.

  Two behavior notes, because "no API change" is not the same as "nothing
  moved": `defang` and `Paths.resolve()` were checked exhaustively against the
  previous implementations (240,000 adversarial strings and 384 environment
  combinations respectively, zero differences), but **when
  `MULTIPLAI_SDK_CALL_TIMEOUT_S` is read has changed** — see the lazy-submodule
  entry above.

- **Workspace discovery now walks up to the nearest `.multiplai/` marker.**
  `Paths` resolution gains a third step between `$WORKSPACE` and the
  `~/.multiplai` standalone fallback: if `$CLAUDE_PROJECT_DIR` is set, its
  nearest ancestor containing a `.multiplai/` directory becomes the workspace
  base. This removes the coupling that made the workspace knowable only because
  a launcher exported `WORKSPACE` — a plugin installed on plain Claude Code
  inside an existing workspace previously wrote to `~/.multiplai` instead.

  **Explicit configuration still wins:** `workspace_dir` and `WORKSPACE` are
  both checked first. Two things are worth stating precisely, because the
  obvious reading of "only the fallback case is affected" is not quite right:

  - **`data_dir` ranks `CLAUDE_PLUGIN_DATA` *above* discovery** — the one place
    in this resolver that is not simply "most explicit first". A managed data
    dir is a fact about the install; a discovered marker is an inference. If
    discovery outranked it, `data_dir` — and with it `venv_dir`,
    `catalogs_dir`, logs and dream state — would move for **every** plugin
    install that has a managed data dir and no `WORKSPACE`, orphaning an
    already-bootstrapped venv and catalog set. Discovery exists to rescue the
    case that fell through to `~/.multiplai`, and that is the *last* step, not
    the `CLAUDE_PLUGIN_DATA` one. `memory_dir`, `diary_dir`, `now_dir` and
    `learnings_dir` do follow a discovered workspace — that is the point of the
    change, and none of them is runtime state.
  - **`$HOME` itself does not satisfy the marker test.** `~/.multiplai` is the
    standalone fallback layout, not a discovered workspace; counting it would
    fire for any session rooted at the home directory with no closer marker.

  The walk is bounded (12 levels, stops before `$HOME`) and **never starts from
  the cwd** — Claude shifts cwd between sub-projects of one workspace, so a
  cwd-rooted walk would make resolution depend on where a script happened to be
  run from. A **relative** `CLAUDE_PROJECT_DIR` (`.`, `..`) is therefore
  ignored rather than resolved, since resolving it re-introduces the cwd. If you
  test against `Paths`, scrub `CLAUDE_PROJECT_DIR` alongside `WORKSPACE`.

- **`model_client.DEFAULT_MODEL` and the default `MULTIPLAI_MODEL` ceiling now
  follow `env.CURRENT_MODEL["sonnet"]`** instead of the literal
  `claude-sonnet-4-6`. `CURRENT_MODEL` had already moved to `claude-sonnet-5`;
  these two defaults had not, so every caller that omitted `model=` — the whole
  extraction/dream/catalogs path — silently ran a generation behind and paid the
  older model's rate. Callers that pass an explicit `model=` are unaffected. If
  you relied on the old default, set `MULTIPLAI_MODEL=claude-sonnet-4-6` or pass
  `model=` at the call site.

  **This changes behaviour for every caller that omits `model=`, so it must ship
  in a `0.x.0` minor — never a patch.** See README → "Versioning and what a bump
  means".
- **`run_agent` is now fail-closed on tools.** `disallowed_tools=None` (the
  default) used to forward *nothing*, so under `permission_mode=
  "bypassPermissions"` every caller that did not pass a deny-list ran with the
  full tool set present and auto-approved — `Bash`, `Read`, `Write`, `WebFetch`
  included. Callers that feed untrusted text (fetched web pages, emails, logs)
  through `run_agent` were one injected instruction away from an auto-approved
  exfiltration chain.

  `allowed_tools` is now forwarded as the SDK's **`tools`** — the *base set* of
  tools that exist at all, which replaces the built-in set rather than adding
  to it — and paired with `disallowed_tools=deny_list(allowed_tools)` as a
  second layer. So a call naming no tools now runs with no tools, and a call
  naming `WebFetch` gets `WebFetch` and nothing else.

  **What you must change:** nothing, if you already pass `allowed_tools` for
  every tool you use — that is the supported way to open a tool and it keeps
  working. If you relied on the old behavior (tools available without naming
  them), name them in `allowed_tools`. `disallowed_tools=[]` opts out of the
  deny-list layer only; the base set still bounds the run to `allowed_tools`.
  There is deliberately no way to ask for the full built-in tool set back.

### Added

- **`multiplai_core.deny_list(allowed_tools)` and `TOOL_UNIVERSE`** — the
  complement helper and the tool list behind the deny-list layer, exported so
  consumers can compute the same list instead of copying it.
  `model_client._DISALLOWED_TOOLS` is now derived from it (`deny_list(None)`),
  so there is exactly one copy of the list in the repo.

  `TOOL_UNIVERSE` is a **tuple** (it is shared mutable state otherwise) and is
  **best-effort, not a safety floor** — `tools` is what makes the boundary a
  guarantee. It was re-derived on 2026-08-06 from the CLI's own generated
  schema list and grew from 18 names to 50, adding the egress tools
  (`Artifact`, `SendMessage`, `PushNotification`, `RemoteTrigger`), the
  deferred-execution tools (`Task*`, `Cron*`, `Workflow`, `ScheduleWakeup`,
  `Monitor`), `REPL`, `MultiEdit`, `NotebookRead`, `TodoWrite` and the MCP
  resource tools. The command to re-derive it is in the source comment.

- **A warning when `allowed_tools` names a tool outside `TOOL_UNIVERSE`** —
  either a typo (previously silent: the tool simply never appeared and the
  model improvised around it) or a tool newer than the list.

### Fixed

- **`plugin_options.option_var` (and every accessor built on it) now raises
  `ValueError` on a key that cannot name an environment variable.** A key
  containing `-`, `.`, a space, or a leading digit builds a
  `CLAUDE_PLUGIN_OPTION_<KEY>` name the harness never exports, so every read
  silently returned the default forever. The raise is deliberate where value
  parsing stays tolerant: a bad key is developer error caught by tests, not
  user config. If a consumer passes such a key today, that call site was
  already dead — rename the key in `plugin.json` to `[A-Za-z_][A-Za-z0-9_]*`
  form.

- **`run_agent` no longer leaks a raw `TypeError` when the installed
  `claude-agent-sdk` rejects an option.** `ClaudeAgentOptions(**kwargs)` sat
  outside the per-attempt error handling, so a signature mismatch (an older
  SDK without `tools`, `effort`, or `thinking`) escaped as `TypeError` —
  violating the documented contract that `run_agent` raises only
  `AgentRunError`/`AgentRunTimeout`, and bypassing every caller's
  `except AgentRunError`. It now raises `AgentRunError` carrying the
  original `TypeError` text.

- **`run_agent`'s `_HOOK_CHILD_SESSION` guard can no longer be cleared by a
  caller's `env`.** The child env was built `{"_HOOK_CHILD_SESSION": "1",
  **env}`, so `env={"_HOOK_CHILD_SESSION": ""}` (or any override) disabled the
  fork-bomb guard that stops hooks from re-firing inside SDK child sessions.
  The guard is now merged last and always wins. No caller passed an override
  in this repo or its consumer; if yours did, it was getting an unguarded
  child, which is the bug.

## [0.13.0] – 2026-08-05

### Added

- **`multiplai_core.plugin_options`** — one place that reads Claude Code plugin
  `userConfig` values under the name the harness actually exports. Exports
  `option`, `option_bool`, `option_int`, `option_float`, `option_present`,
  `option_var` and `OPTION_PREFIX`. Callers pass the **bare option name**
  (`option("enable_skills")`); the module uppercases it, because Claude Code
  exports `CLAUDE_PLUGIN_OPTION_<KEY>` with `<KEY>` **uppercased**
  ([plugins reference](https://code.claude.com/docs/en/plugins-reference.md)).
  Malformed values log a warning and yield the caller's default — these run
  inside hooks and must never raise.

### Fixed

- **Plugin options were read in the wrong case and therefore never read at
  all.** `paths.py` (`workspace_dir`, `data_dir`, `memory_dir`, `diary_dir`,
  `now_dir`, `learnings_dir`) and `model_client.py` (`anthropic_api_key`) all
  looked up `CLAUDE_PLUGIN_OPTION_<lowercase key>`, which the harness never
  sets, so every one of them silently fell through to its fallback.
  **What you gain:** these options take effect for the first time. **What to
  check before moving your pin:** if you were relying on the fallback path
  (`WORKSPACE`, `CLAUDE_PLUGIN_DATA`, `~/.multiplai`) while *also* having one of
  these options configured, the option now wins. Cascade order and defaults are
  otherwise unchanged — only the variable name consulted changed. There is
  deliberately **no lowercase fallback**; a regression test fails the build if a
  lowercase read reappears anywhere in `src/`.

## [0.12.1] – 2026-08-05

### Changed

- **`uv.lock` refreshed; no declared dependency range moved.** Notably
  `anthropic` 0.102.0 → 0.120.2, `claude-agent-sdk` 0.2.119 → 0.2.129 and
  `cryptography` 49.0.0 → 50.0.0 (the last carries advisories). **Nothing to do
  for a pin** — no export, signature or declared constraint changed, so this is
  invisible to anyone resolving fresh; it matters only if you vendor our lock.
- **Dependabot now runs with `versioning-strategy: increase-if-necessary`.** Its
  default strategy had been rewriting the *declared* floors in
  `pyproject.toml` to whatever it had just resolved — it proposed
  `anthropic>=0.40` → `>=0.120.2` and moved the deliberately-chosen
  `claude-agent-sdk>=0.2.116` floor to `>=0.2.128`. For a library those raised
  floors are a real cost to consumers: they narrow what you can resolve
  alongside us for no stated reason. `tests/test_pyproject_sdk_floor.py`
  caught it, which is what that guard is for. Dependabot will now touch a
  declared range only when a new version genuinely falls outside it.

## [0.12.0] – 2026-07-31

### Added

- **`run_agent` now logs an `alive` heartbeat while a call is in flight.**
  Between `START` and `DONE` a run emitted nothing, so a multi-minute call was
  indistinguishable from a wedged one — the only signal was a `DONE`/`FAIL`
  line that might be half an hour away. Every attempt now logs at INFO, every
  60 s by default:
  `run_agent [<label>] alive 120s attempt=1/2 turns=3 text=41252 bytes`.
  The byte count is the useful part: it separates "slow but producing" from
  "stalled with nothing". The task is cancelled and awaited when the attempt
  ends, so nothing keeps ticking past a return, a timeout, or into a retry
  backoff. **Nothing to do for a pin** — no signature or export changed. The
  interval is `MULTIPLAI_AGENT_HEARTBEAT_S`, read at *call* time (not import),
  so you can set it per run; **opt out with `MULTIPLAI_AGENT_HEARTBEAT_S=0`**
  (`0` or negative disables it) if your caller's log must stay quiet.
- **`ModelClient.query()` gains a keyword-only `timeout_s: float | None = None`**
  — a per-call override of the SDK hard timeout, on `AgentSDKClient` and (for
  interface parity, where it is accepted and ignored like `effort`)
  `AnthropicAPIClient`. `None` keeps today's behaviour exactly: the module
  default from `MULTIPLAI_SDK_CALL_TIMEOUT_S`. Until now the ceiling was
  reachable only as a module global read from the env at import, so a caller
  that needed a longer timeout for **one** oversized request had to patch
  `model_client._SDK_CALL_TIMEOUT_S` — private, and racy under
  `asyncio.gather`, where it changes the ceiling for every call in flight.
  Pass the keyword instead. **Action for pin-movers:** none if you only call
  `query()`; if you *implement* `ModelClient` out of tree (a registered
  provider backend), add `timeout_s: float | None = None` to your `query()`
  signature — ignoring it is fine, and `isinstance` against the runtime-checkable
  Protocol was never affected.

## [0.11.0] – 2026-07-27

### Added

- **`EFFORT_TIERS` and `KNOWN_EFFORTS` are now exported** — the effort-name
  table `pick_effort` caps against, previously private as `_EFFORT_TIERS`.
  Validate against these instead of mirroring the table: a drifted copy is
  worse than none, because `pick_effort` normalizes a name it does not
  recognize away and floors to `"high"`, so a caller that believes an unknown
  name is valid silently loses its own "unknown → default" fallback.
  `EFFORT_TIERS` is a read-only mapping name → rank (`low` 1 … `max` 5); treat
  membership as the question and the integers as relative order only, since a
  future release may add a tier. `KNOWN_EFFORTS` is `frozenset(EFFORT_TIERS)`
  for membership tests. **Action for pin-movers:** if you keep a hand-copied
  list of effort names (multiplai-cc-mktplace's buildme `KNOWN_EFFORTS` did),
  delete it and import this one. Purely additive — `_EFFORT_TIERS` still works.
- `SECURITY.md` — how to report a vulnerability (security@spikelab.org), what
  this code can reach, which versions get fixes, and how the immutable-tag
  delivery model (README → Availability guarantee) bounds the blast radius.
  No API change; nothing to do for a pin.

### Changed

- README intro no longer describes the suite as improving itself — the memory
  system learns what the user approves, and the docs now say exactly that.
  Docs only; no API change.

## [0.10.0] – 2026-07-27

### Added

- **`untrusted` module** — consolidated defang/fence primitives for
  externally-authored text, replacing the four diverged copies in
  `multiplai-cc-mktplace` (log-doctor's `defang`/`fence`/`contains_injection`,
  gmail's `defang`, slack's `_defang`, deep-research's `defang_untrusted`).
  New exports: `defang`, `fence`, `contains_injection`, `markdown_notice`,
  `bracket_notice`.
  - `defang(text, limit=None, *, markdown_fences=True, mark_injections=False)`
    always strips control/bidi/zero-width characters (including U+2028/U+2029),
    strips full ANSI sequences, and HTML-escapes the `<untrusted-content>`
    markers. `None`/falsy → `""`; non-str input is `str()`-coerced. The
    escaping path is idempotent, so chained defangs don't double-escape.
  - **`markdown_fences` defaults to `True`** — the safe behaviour is the
    default one, so a caller who never thinks about the flag still gets a
    fence the payload cannot break out of. If your output is *not* markdown
    (plain stdout, a JSON field), pass `markdown_fences=False` to keep
    ` ``` ` in the payload intact: that is the byte-for-byte equivalent of
    the gmail/slack/deep-research copies. `mark_injections` defaults to
    `False` and is an annotation, not a boundary; log-doctor's exact output is
    `defang(text, limit, mark_injections=True)`.
  - `fence(text, source, limit=None) -> list[str]` reproduces log-doctor's
    fenced-block contract: `[]` on empty body, ` ```text ` inner fence,
    defanged `source` attribute, injection spans marked `⟪INJECTION?⟫…⟪/⟫`.
  - `markdown_notice(what, channel, *, injection_marker=False)` and
    `bracket_notice(channel)` rebuild the two existing notice shapes
    byte-exactly (log-doctor's blockquote; gmail/slack's bracketed one-liner).

  For a plugin author: if your script carries a local defang copy, moving your
  pin to the release containing this lets you delete it and import from core.
  Two deliberate improvements on the copies, so output is *not* byte-identical
  in these two cases: `fence()` now escapes `"` in the `source` attribute (a
  label containing a quote could previously close the attribute and append
  attributes to our own tag), and the `ignore …` injection pattern now matches
  "ignore **the** previous instructions", which every copy missed. Both
  strictly widen protection; neither changes a signature.

### Added — documentation and tooling

- This `CHANGELOG.md`, a stated compatibility promise in `README.md`, a gated
  `release.sh`, CI on Python 3.11 and 3.12, and a `CLAUDE.md` for agents
  editing the library.

## [0.9.0] – 2026-07-26

### Added

- **Provider seam** for cross-family model clients. New exports:
  `ModelSpec`, `parse_model_spec`, `create_client_for`, `register_provider`,
  `unregister_provider`, `registered_providers`, `UnknownProviderError`,
  `DEFAULT_PROVIDER`. A bare model ID still resolves to Anthropic, so existing
  config values and call sites behave exactly as before. Non-Anthropic backends
  do **not** ship here — an out-of-tree factory joins the registry, because
  choosing a provider means choosing whose API key and whose bill.
  Parsing splits on the *first* colon only (`ollama:llama3:70b` →
  provider `ollama`, model `llama3:70b`).
- **`pick_effort(default, task=…)`** — effort becomes a first-class second axis
  next to `pick_model`, resolved per `multiplai.conf` task section with the same
  fallback shape and capped by the `MULTIPLAI_EFFORT` ceiling. Effort tiers rank
  `low < medium < high < xhigh < max`.
- **`pick_model_spec(default_tier, task=…)`** — the provider-qualified
  equivalent of `pick_model`, returning a `ModelSpec`.
- **Direct-API prompt caching.** `model_client.cacheable_system()` attaches an
  `ephemeral` cache breakpoint to long, stable system prompts on the
  Anthropic-API fallback path, which previously sent `system=` as a bare string
  and re-paid full input price on every call. Prompts below
  `MIN_CACHEABLE_SYSTEM_BYTES` (4096) pass through unchanged. This changes
  billing, not behaviour — responses are identical. The Agent-SDK path already
  cached on its own.

### Fixed

- `packaging` is declared as a direct `dev` dependency instead of being leaned
  on as a transitive dependency of pytest; a dead Python-version guard in the
  tests was dropped.

### Notes for consumers

- Purely additive at the export level: nothing exported by `0.8.1` was removed
  or renamed, so a pin bumped from `v0.8.1` needs no code changes.

## [0.8.1] – 2026-07-15

### Fixed

- **`[sdk]` extra now resolves a working Agent SDK.** The floor moved to
  `claude-agent-sdk>=0.2.116` (was `>=0.1,<0.2`): the 0.1.x line misparses the
  terminal result message emitted by modern Claude CLIs (>= 2.x) and raises
  `Claude Code returned an error result: success` after a full generation — a
  deterministic failure the retry wrapper cannot help with. The floor matches
  the one pinned by `multiplai-kit` / `multiplai-gui` so all SDK consumers
  resolve compatibly.
- Ceiling added: `claude-agent-sdk<0.3`, with a test guarding the constraint. A
  single minor bump (0.1 → 0.2) already shipped a breaking result-message parse
  change, so an uncapped consumer could silently re-break on a future release.

### Changed

- `log_utils.setup_logging(propagate_loggers=…)` caveats documented; a test
  fixture now restores the package logger level it changes.

### Notes for consumers

- Fixes only — no API change. Consumers on `v0.8.0` using the `[sdk]` extra
  should take this.

## [0.8.0] – 2026-07-15

### Added

- **`setup_logging(propagate_loggers=…)`** — opt in to capturing loggers from
  named third-party packages into the component log file, for scripts that need
  a dependency's output on disk alongside their own.

### Changed

- The README states the **availability guarantee** that the delivery model rests
  on: this repository stays public, release tags are immutable (never moved,
  deleted, or reused), and fixes ship as new tags.

## [0.7.0] – 2026-07-09

### Added

- **Semantic model tiers.** New exports `pick_model(default_tier, task=…)` and
  the `CURRENT_MODEL` registry: a consumer asks for a tier (`opus`, `sonnet`,
  `haiku`) and gets the current model ID for it, resolved per `multiplai.conf`
  task section and capped by the `MULTIPLAI_MODEL` env ceiling. Callers stop
  hard-coding dated model IDs that go stale on every model release.

## [0.6.0] – 2026-07-09

### Added

- **`multiplai_core.costing`** — a pricing table (`pricing.json`), cost math
  (`TokenCounts`, `resolve_model_rates`, `price_tokens`, `build_record`), and a
  monthly JSONL ledger under `<data_dir>/costs/ledger-YYYY-MM.jsonl`
  (`costs_dir`, `ledger_file`, `append_records`). Model IDs match by exact id,
  then date-suffix-stripped id, then longest known prefix, so a newly dated
  model still prices.
- **Cost-ledger tap in `run_agent`** via a new keyword-only
  `component="…"` argument: pass a component tag (e.g. `"buildme"`) and each
  agent run appends a priced record to the ledger. Defaults to `""`, which
  records nothing — existing calls are unaffected.

[Unreleased]: https://github.com/spikelab/multiplai-core/compare/v0.14.0...HEAD
[0.14.0]: https://github.com/spikelab/multiplai-core/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/spikelab/multiplai-core/compare/v0.12.1...v0.13.0
[0.12.1]: https://github.com/spikelab/multiplai-core/compare/v0.12.0...v0.12.1
[0.12.0]: https://github.com/spikelab/multiplai-core/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/spikelab/multiplai-core/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/spikelab/multiplai-core/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/spikelab/multiplai-core/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/spikelab/multiplai-core/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/spikelab/multiplai-core/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/spikelab/multiplai-core/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/spikelab/multiplai-core/compare/v0.5.2...v0.6.0
