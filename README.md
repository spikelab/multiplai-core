# multiplai-core

Shared library for the [Multiplai](https://github.com/spikelab) Claude Code plugins.
One source of truth for the things every plugin needs, so no plugin vendors its
own drifting copy:

| Module | Purpose |
|---|---|
| `multiplai_core.paths` | Path-resolver cascade: `CLAUDE_PLUGIN_OPTION_*` → workspace → `CLAUDE_PLUGIN_DATA` → `~/.multiplai`. |
| `multiplai_core.config` | YAML/JSON load-save, memory-file reads, atomic session-state I/O. |
| `multiplai_core.log_utils` | `setup_logging(component)`, `log_event(...)` — ISO-8601 UTC, dated rotation, retention. |
| `multiplai_core.model_client` | `create_client()` — Agent SDK first, Anthropic API fallback. |

## Install

Consumed as a git-URL dependency — no PyPI. In a script's PEP 723 header:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["multiplai-core @ git+https://github.com/spikelab/multiplai-core@v0.3.0"]
# ///
```

Pin by **git tag** (`@v0.3.0`); cut a new tag rather than moving an existing one.
For the Agent SDK backend from plain Python, install the `sdk` extra:
`multiplai-core[sdk] @ git+...@v0.3.0`.

## Develop

```bash
uv run --extra dev pytest        # run the test suite in an ephemeral env
```

## Layout

```
src/multiplai_core/   # the package
tests/                # pytest suite
```
