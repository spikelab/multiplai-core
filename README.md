# multiplai-core

Shared library for the [Multiplai](https://github.com/spikelab) Claude Code plugins.
One source of truth for the things every plugin needs, so no plugin vendors its
own drifting copy:

| Module | Purpose |
|---|---|
| `multiplai_core.paths` | Path-resolver cascade: `CLAUDE_PLUGIN_OPTION_*` → workspace → `CLAUDE_PLUGIN_DATA` → `~/.multiplai`. |
| `multiplai_core.config` | YAML/JSON load-save, memory-file reads, atomic session-state I/O. |
| `multiplai_core.env` | `.env` discovery/loading, `multiplai.conf` parsing, model/effort ceiling resolution. |
| `multiplai_core.text` | `extract_json()` — pull a JSON object/array out of a model response. |
| `multiplai_core.aio` | `hard_timeout()` and async task helpers. |
| `multiplai_core.log_utils` | `setup_logging(component)`, `log_event(...)` — ISO-8601 UTC, dated rotation, retention. |
| `multiplai_core.model_client` | `create_client()` — Agent SDK first, Anthropic API fallback. |

## Install

Consumed as a git-URL dependency — no PyPI. In a script's PEP 723 header:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["multiplai-core @ git+https://github.com/spikelab/multiplai-core@v0.4.0"]
# ///
```

Pin by **git tag** (`@v0.4.0`); cut a new tag rather than moving an existing one.

Optional extras (append to the requirement, e.g. `multiplai-core[sdk] @ git+...@v0.4.0`):

- `sdk` — the Agent SDK backend (`claude-agent-sdk`) when running outside the
  Claude Code runtime, which otherwise injects it.
- `dotenv` — `python-dotenv`, required for `env.load_env()` to auto-load `.env`
  files (without it, `load_env()` is a no-op that warns).

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

## Develop

```bash
uv run --extra dev pytest        # run the test suite in the project venv
```

## Layout

```
src/multiplai_core/   # the package
tests/                # pytest suite
```
