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
| `multiplai_core.paths` | Path-resolver cascade: `CLAUDE_PLUGIN_OPTION_*` → workspace → `CLAUDE_PLUGIN_DATA` → `~/.multiplai`. |
| `multiplai_core.plugin_options` | Reads plugin `userConfig` options under the name the harness exports (`CLAUDE_PLUGIN_OPTION_<KEY>`, key **uppercased**). Pass the bare option name. |
| `multiplai_core.config` | YAML/JSON load-save, memory-file reads, atomic session-state I/O. |
| `multiplai_core.env` | `.env` discovery/loading, `multiplai.conf` parsing, model/effort ceiling resolution, provider-qualified `ModelSpec`s. |
| `multiplai_core.text` | `extract_json()` — pull a JSON object/array out of a model response. |
| `multiplai_core.aio` | `hard_timeout()` and async task helpers. |
| `multiplai_core.log_utils` | `setup_logging(component)`, `log_event(...)` — ISO-8601 UTC, dated rotation, retention. |
| `multiplai_core.model_client` | `create_client()` — Agent SDK first, Anthropic API fallback — plus the [provider seam](#provider-seam) (`create_client_for`, `register_provider`). |
| `multiplai_core.agent_runner` | `run_agent()` — the single SDK agent invocation path (isolation flags, hard timeout, stderr capture, big-prompt fallback, retry, usage/files-changed reporting). |

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

client = await create_client()            # Agent SDK if present, else API key
resp = await client.query(
    system="You output only JSON.",
    messages=[{"role": "user", "content": "Give me {\"ok\": true}"}],
)
data = extract_json(resp.content)         # -> {"ok": True}
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
| `pick_model(default_tier, task=…)` | the model ID for a task section, tier-ranked and ceiling-capped |
| `pick_effort(default, task=…)` | the reasoning effort for a task section, ceiling-capped |
| `pick_model_spec(default_tier, task=…)` | a provider-qualified [`ModelSpec`](#provider-seam) |

The `MULTIPLAI_MODEL` / `MULTIPLAI_EFFORT` env ceilings cap the resolved value,
so a budget run forces everything down and a conf override cannot escape it.
Effort tiers rank `low < medium < high < xhigh < max` — `xhigh` sits between
`high` and `max`, and omitting it from the table silently caps xhigh to high.

## Provider seam

Reviewer panels want models from *different families*: Claude and GPT reviewers
empirically find largely disjoint error sets, while a same-family panel mostly
re-finds the same things. But the tier/ceiling machinery in `env.py` ranks the
Claude family only, so cross-vendor support cannot be another branch inside
`create_client()` — it is a registry an out-of-tree backend joins.

```python
from multiplai_core import (
    ModelSpec, parse_model_spec, create_client_for, register_provider,
)

spec = parse_model_spec("openai:gpt-5")   # -> ModelSpec("openai", "gpt-5")
parse_model_spec("claude-opus-5")          # -> ModelSpec("anthropic", "claude-opus-5")

register_provider("openai", my_factory)    # async (spec, *, api_key=…) -> ModelClient
client = await create_client_for(spec)     # UnknownProviderError if unregistered
```

- A **bare** model ID stays Anthropic (`DEFAULT_PROVIDER`), so every existing
  config value and code path resolves exactly as before the seam existed.
- Parsing splits on the **first** colon only — `ollama:llama3:70b` is provider
  `ollama`, model `llama3:70b`.
- A provider-qualified `MODEL=` in `multiplai.conf` is passed through verbatim:
  Claude tier ranking and the `MULTIPLAI_MODEL` ceiling are meaningless for
  another vendor's model, and coercing it to a Claude tier would defeat the point
  of naming a cross-family model.
- **No non-Anthropic backend ships here.** Choosing a provider means choosing
  whose API key and whose bill — a human decision, not a default. Registering one
  from the outside is three lines.

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

## Develop

```bash
uv run --extra dev pytest        # run the test suite in the project venv
```

## Layout

```
src/multiplai_core/   # the package
tests/                # pytest suite
```
