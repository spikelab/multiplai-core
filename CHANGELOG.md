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

[Unreleased]: https://github.com/spikelab/multiplai-core/compare/v0.13.0...HEAD
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
