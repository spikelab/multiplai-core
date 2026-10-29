# multiplai-core

> Part of the **[Multiplai suite](https://github.com/spikelab/multiplai)** — what the suite is, how the five repos fit together, and which part you need.

Shared library for the [Multiplai](https://github.com/spikelab/multiplai) Claude Code plugins.
Multiplai is a plugin suite that turns Claude Code into a persistent working
environment that compounds (it learns what you approve) — a memory/context
engine plus six themed skill packs, shipped
via [multiplai-cc-mktplace](https://github.com/spikelab/multiplai-cc-mktplace). This
library is the plumbing their scripts share.
One source of truth for the things every plugin needs, so no plugin vendors its
own drifting copy:

| Module | Purpose |
|---|---|
| `multiplai_core.paths` | Path-resolver cascade: plugin option (`CLAUDE_PLUGIN_OPTION_*`, else the saved `pluginConfigs` value) → workspace → `CLAUDE_PLUGIN_DATA` → `~/.multiplai`. |
| `multiplai_core.plugin_options` | Reads plugin `userConfig` options under the name the harness exports (`CLAUDE_PLUGIN_OPTION_<KEY>`, key **uppercased**). Pass the bare option name. Outside a hook, where the harness exports nothing, falls back to the plugin's saved options in `$CLAUDE_CONFIG_DIR/settings.json` → `pluginConfigs`. |
| `multiplai_core.config` | YAML/JSON load-save, memory-file reads, atomic session-state I/O. |
| `multiplai_core.env` | `.env` discovery/loading, `multiplai.conf` parsing, model/effort name resolution (family aliases), provider-qualified `ModelSpec`s. |
| `multiplai_core.session` | The model and effort for a step no config names: `MULTIPLAI_MODEL` / `MULTIPLAI_EFFORT`, else the Claude Code session's own (`default_model()`, `default_effort()`, `pin_session_settings()`). |
| `multiplai_core.text` | `extract_json()` — pull a JSON object/array out of a model response. |
| `multiplai_core.aio` | `hard_timeout()` and async task helpers. |
| `multiplai_core.log_utils` | `setup_logging(component)`, `log_event(...)` — ISO-8601 UTC, one file per component per day, kept 365 days unless `MULTIPLAI_LOG_RETENTION_DAYS` (environment, then `multiplai.conf`) says otherwise; `0` keeps them forever. `hook-errors.log` moves to `hook-errors-YYYY-MM-DD.log` each day and is never shortened. `hook_run(name, logger)` / `HookRun` — ENTRY/EXIT timing pair that makes a killed hook diagnosable. |
| `multiplai_core.model_client` | `create_client()` — Agent SDK first, Anthropic API fallback — plus the [provider seam](#provider-seam) (`create_client_for`, `register_provider`). |
| `multiplai_core.agent_runner` | `run_agent()` — the single SDK agent invocation path (isolation flags, hard timeout, stderr capture, big-prompt fallback, retry, usage/files-changed reporting). Connection errors and HTTP 5xx are retried with backoff for up to 15 minutes, whatever `max_attempts` says, and the [agent backend](#agent-backends) registry (`register_agent_backend`). A hit Claude usage limit raises `UsageLimitError` and is never retried; `is_usage_limit(exc)` recognises it through a wrapper. |
| `multiplai_core.pi` | The `pi` agent backend: `run_agent(model="pi:<model-id>@<provider>")` through `pi --mode rpc`. Imported on first use only. |
| `multiplai_core.structured` | `parse_answer()` / `reformat_prompt()` — validate an agent's answer against a pydantic schema, and build the no-tools re-ask that reformats one that did not parse. No call loop: the retry policy stays in each pipeline. |
| `multiplai_core.budget` | `SpendLedger` — per-run token and cost ledger with a circuit breaker (`BudgetExceededError`), checkpointable spend, and ContextVar helpers (`start`/`get`/`record`/`check`) for one ledger per asyncio task. |
| `multiplai_core.checkpoint` | `save_model()` / `load_model()` — atomic JSON checkpoint of a pydantic model. |
| `multiplai_core.progress` | `ProgressWriter` — tailable `[<UTC>Z] STAGE/DONE/FAILED …` progress file. |
| `multiplai_core.stage_config` | `conf_value()`, `read_yaml_mapping()`, `coerce_int()`, `check_effort()` — the shared half of a pipeline's `<name>.yaml` > `multiplai.conf` > defaults config loader. |
| `multiplai_core.specs` | The `specs/` directory format buildme and pm-spec share: requirement-file parsing and validation (`parse_requirements`, `valid_requirement_text`, `expected_capabilities`), the delta merge into `specs/registry/` (`apply_delta`, `merge_delta_requirements`, `archive_change`), and the ready record in `.change.yaml` (`spec_revision`, `mark_ready`, `is_ready`). |

## Install

Consumed as a git-URL dependency — no PyPI. Two shapes are supported; pick by
whether you have a lockfile.

**With a lockfile** (a uv workspace or any `uv.lock`-managed project) — declare
it in `pyproject.toml` and let the lock record the commit:

```toml
[tool.uv.sources]
multiplai-core = { git = "https://github.com/spikelab/multiplai-core", branch = "main" }
```

The lock is what freezes resolution, so nothing moves under you; you take a new
version by running `uv lock --upgrade-package multiplai-core` and committing the
result. Note that Dependabot does **not** bump git-sourced dependencies, so that
step is deliberate rather than automatic. This is what
[`multiplai-cc-mktplace`](https://github.com/spikelab/multiplai-cc-mktplace)
does.

**Without a lockfile** — a genuinely standalone one-file script — pin a tag in a
PEP 723 header:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["multiplai-core @ git+https://github.com/spikelab/multiplai-core@v0.13.0"]
# ///
```

Pin by **git tag** (`@v0.13.0`); cut a new tag rather than moving an existing
one. Be aware that `uv run` re-resolves an inline dependency on **every**
invocation, so this shape is a poor fit for anything latency-sensitive or run
in bulk — that is exactly why the marketplace moved off it.

Optional extras (append to the requirement, e.g. `multiplai-core[sdk] @ git+...@v0.8.1`):

- `sdk` — the Agent SDK backend (`claude-agent-sdk`) when running outside the
  Claude Code runtime, which otherwise injects it.
- `dotenv` — `python-dotenv`, required for `env.load_env()` to auto-load `.env`
  files (without it, `load_env()` is a no-op that warns).

### Availability guarantee

Every installed multiplai plugin resolves this library from GitHub on first
run, so its availability is part of the plugins' contract with their users:

- **This repository stays public.** Taking it private or deleting it would
  break every installed plugin; it will not happen casually.
- **Release tags are immutable.** A `vX.Y.Z` tag is never moved, deleted, or
  reused — what a pin resolved yesterday is what it resolves tomorrow. The same
  holds for a commit recorded in a lockfile.
- Fixes land on `main` and are tagged; consumers take them by bumping a pin or
  re-locking. Nothing upgrades on its own, in either shape.

(PyPI publication — which would also remove the GitHub-availability dependency
and speed up first-run resolution — is under consideration; until then the
guarantee above is the contract.)

### Versioning and what a bump means

The guarantee above covers **reachability** — that your pin keeps resolving. This
covers **API stability** — what you are agreeing to when you move it.

This library is pre-1.0, and the rule is:

- **`0.x.0` (minor) — may add and may break the public API.** A minor bump can
  introduce new names *and* remove, rename, or change the signature of existing
  ones. **Read [`CHANGELOG.md`](CHANGELOG.md) before taking one.**
- **`0.x.Y` (patch) — fixes only, never breaks.** Safe to take without reading
  anything, though the changelog still says what was fixed.

Once the library reaches 1.0 this tightens to ordinary semver (breaking changes
only in a major bump); until then, the minor is the breaking axis.

**Which surface is public:** the names exported from the `multiplai_core`
package — its `__all__`, i.e. what `from multiplai_core import X` resolves.
Everything else is internal: private helpers, module layout, submodule contents
not re-exported at the top level, and the shape of `pricing.json`. Internals may
change in any release, including a patch. If you need something that is not
exported, ask for it to be exported rather than reaching in.

**Which version should I take?** If you pin a tag, pin the **newest** one when
you add the dependency. Either way, move **deliberately, per consumer**, after
reading the changelog entry for the version you are moving to. Neither shape
floats: a tag is immutable and a lockfile records a commit, so nothing upgrades
underneath you and every upgrade is a reviewable diff in the consuming repo.
What a version *range* would give you — silent minor upgrades — is deliberately
not on offer, because the pre-1.0 minor is the breaking axis (above).

## Usage

```python
from multiplai_core import create_client, extract_json

client = await create_client()  # Agent SDK if present, else API key
resp = await client.query(
    system="You output only JSON.",
    messages=[{"role": "user", "content": 'Give me {"ok": true}'}],
)
data = extract_json(resp.content)  # -> {"ok": True}
```

## Model and effort are two axes

`multiplai.conf` resolves both, per task section, with the same fallback shape:

```ini
[buildme]                 # applies to the whole task
MODEL=opus
EFFORT=medium

[buildme.review]          # one step overrides just the effort
EFFORT=high
```

| Helper | Returns |
|---|---|
| `pick_model(default_tier, task=…)` | the model ID for a task section; a family alias expands to its current ID |
| `pick_effort(default, task=…)` | the reasoning effort for a task section |
| `pick_model_spec(default_tier, task=…)` | a provider-qualified [`ModelSpec`](#provider-seam) |

Nothing caps the resolved value. A family alias (`haiku`, `sonnet`, `opus`,
`fable`) expands to the current ID in `CURRENT_MODEL`; every other string,
a dated ID included, is used exactly as written. Valid efforts are
`KNOWN_EFFORTS`: `low`, `medium`, `high`, `xhigh`, `max`.

### The default for a step no config names

`multiplai_core.session` answers "which model and effort when nothing is
configured", and `run_agent` uses it whenever it is called with no `model` or
no `effort`. Model and effort are resolved separately, each in this order:

1. `MULTIPLAI_MODEL` / `MULTIPLAI_EFFORT` from the environment, then from
   `multiplai.conf`. An empty value counts as unset.
2. The Claude Code session's own: `MULTIPLAI_SESSION_MODEL` /
   `MULTIPLAI_SESSION_EFFORT` when a parent pinned them, else the last
   assistant `model` in the session transcript
   (`$CLAUDE_CONFIG_DIR/projects/*/<CLAUDE_CODE_SESSION_ID>.jsonl`) and
   `CLAUDE_EFFORT`.
3. Nothing: `run_agent` omits the option and the CLI uses its own default
   (logged once at WARNING for the model).

A pipeline calls `pin_session_settings()` at start, so every child process it
starts sees the values the run started with. An explicit model or effort —
a step's config, a background job's own default — never goes through these
defaults, so `MULTIPLAI_MODEL` does not change it.

## Provider seam

Reviewer panels want models from *different families*: Claude and GPT reviewers
empirically find largely disjoint error sets, while a same-family panel mostly
re-finds the same things. But the family aliases in `env.py` cover the
Claude family only, so cross-vendor support cannot be another branch inside
`create_client()` — it is a registry an out-of-tree backend joins.

```python
from multiplai_core import (
    ModelSpec,
    parse_model_spec,
    create_client_for,
    register_provider,
)

spec = parse_model_spec("openai:gpt-5")  # -> ModelSpec("openai", "gpt-5")
parse_model_spec("claude-opus-5")  # -> ModelSpec("anthropic", "claude-opus-5")

register_provider("openai", my_factory)  # async (spec, *, api_key=…) -> ModelClient
client = await create_client_for(spec)  # UnknownProviderError if unregistered
```

- A **bare** model ID stays Anthropic (`DEFAULT_PROVIDER`), so every existing
  config value and code path resolves exactly as before the seam existed.
- Parsing splits on the **first** colon only — `ollama:llama3:70b` is provider
  `ollama`, model `llama3:70b`.
- A provider-qualified `MODEL=` in `multiplai.conf` is passed through verbatim:
  Claude family aliases mean nothing for another vendor's model, and coercing
  it to a Claude family would defeat the point of naming a cross-family model.
- **No non-Anthropic model client ships here.** Choosing a provider means
  choosing whose API key and whose bill — a human decision, not a default.
  Registering one from the outside is three lines. The `pi` *agent* backend
  below is the one exception, and it keeps the rule's point: it runs only when
  a caller names a `pi:` spec, and the keys and the bill live in Pi's own
  configuration, not here.

### Picking a model from an ordered list

`resolve_model_spec(chain)` resolves the first non-empty entry
of an ordered list, most specific first. A plugin decides which config keys go
into the list (for buildme: the step, then its stage, then the default); core
only resolves it. An Anthropic spec — `opus`, `fable`, `claude:sonnet`,
`anthropic:claude-opus-4-8` — has its family alias expanded and is otherwise
kept as written, exactly like `pick_model_spec`. Any other spec passes through
unchanged. All entries empty returns `None`, so the caller falls back to
`default_model()`. The `ceiling` keyword is deprecated and ignored.

`claude:` is an alias of `anthropic:`.

## Agent backends

`run_agent(model=...)` takes the same `provider:model` specs. An Anthropic spec
(bare, `anthropic:` or `claude:`) runs on the Claude Agent SDK exactly as
before, with the prefix removed. Any other provider runs on the backend
registered for it:

```python
from multiplai_core import register_agent_backend, registered_agent_backends


async def my_runner(prompt, **kwargs) -> AgentRunResult: ...  # run_agent's keywords


register_agent_backend("mine", my_runner)
await run_agent("…", model="mine:some-model")  # UnknownAgentBackendError if unregistered  # fmt: skip
```

One backend ships: `pi`, for `pi:<model-id>@<provider>` (for example
`pi:deepseek/deepseek-v4-flash@fireworks`). It drives `pi --mode rpc` (binary
`pi` on `PATH`, or `MULTIPLAI_PI_BIN`), maps `allowed_tools` onto Pi's `-t`
allow-list, maps `effort` onto `--thinking`, enforces `max_turns` and
`timeout_s`, and sums usage and cost from Pi's own events. It loads the first
time a `pi:` spec runs, so importing `agent_runner` never imports it. Installing
and configuring Pi is not covered here; a missing binary is an error, never a
silent fall back to Claude.

## Prompt caching

The Agent-SDK path caches on its own (measured ~100% hit ratio in the cost
ledger). The direct-API fallback used to send `system=` as a bare string, which
is never cached — every call re-paid full input price for the same prompt.
`cacheable_system()` now attaches an `ephemeral` cache breakpoint to long, stable
system prompts on that path.

Prompts under `MIN_CACHEABLE_SYSTEM_BYTES` (4096) pass through as a plain string:
Anthropic ignores a breakpoint below its per-model minimum (1024 tokens for
Sonnet/Opus, 2048 for Haiku), and the gate is on **bytes** to avoid a tokenizer
dependency — ~2 bytes/token is a deliberately conservative floor, so the
breakpoint is only skipped on prompts that could not have been cached anyway.
This changes billing, not behaviour: responses are identical.

`agent_prompt_cache_ttl` in `multiplai.conf` (for example `5m` or `1h`) sets
`CLAUDE_CODE_PROMPT_CACHE_TTL` for every `run_agent` child on the Agent-SDK
path. It is unset by default, so the CLI keeps its own default. A
`CLAUDE_CODE_PROMPT_CACHE_TTL` the caller passes in `env=`, or one already in
the process environment, wins over it.

## Develop

```bash
uv run --extra dev pytest        # run the test suite in the project venv
```

## Layout

```
src/multiplai_core/   # the package
tests/                # pytest suite
```
