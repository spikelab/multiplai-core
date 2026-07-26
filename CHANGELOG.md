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

- Documentation and tooling only, no library changes: this `CHANGELOG.md`, a
  stated compatibility promise in `README.md`, a gated `release.sh`, CI on
  Python 3.11 and 3.12, and a `CLAUDE.md` for agents editing the library.

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

[Unreleased]: https://github.com/spikelab/multiplai-core/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/spikelab/multiplai-core/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/spikelab/multiplai-core/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/spikelab/multiplai-core/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/spikelab/multiplai-core/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/spikelab/multiplai-core/compare/v0.5.2...v0.6.0
