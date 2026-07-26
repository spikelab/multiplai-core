# multiplai-core — repo guide

The shared library every [Multiplai](https://github.com/spikelab/multiplai)
plugin depends on: path resolution, config/`.env`/`multiplai.conf` loading,
model + effort resolution, logging, cost ledger, the model client and the
provider seam, and the single SDK agent-invocation path.

**It is not on PyPI.** Every consumer resolves it **from GitHub at a pinned git
tag** in a PEP 723 script header:

```python
# dependencies = ["multiplai-core @ git+https://github.com/spikelab/multiplai-core@v0.9.0"]
```

Read that sentence again before you edit `src/`. Those pins are on other
people's machines, already resolved. This is the repo where a careless rename
breaks installed plugins for users you cannot reach.

## The two rules that cannot be walked back

### 1. Tags are immutable

**Never move, delete, or re-cut a tag.** A `vX.Y.Z` tag resolves tomorrow to
exactly what it resolved yesterday — that is a public promise in
`README.md` ("Availability guarantee"), and the whole delivery model rests on
it. A bad release is fixed by cutting a **new** tag, never by re-pointing the
old one. Same for the repo itself: it stays public.

Corollary: **merging to `main` delivers nothing.** `main` is the releasable
line; **tags are the unit of delivery.**

### 2. Removing or renaming an exported symbol is a breaking change

The public surface is the names exported from the `multiplai_core` package —
its `__all__`. Anything else (private helpers, module internals, submodule
contents not re-exported, the shape of `pricing.json`) is internal and may
change in any release.

For anything exported:

- **Prefer adding.** New function, new keyword-only argument with a default,
  new module. Additive changes cost consumers nothing.
- **Never change the meaning of an existing signature** — a new required
  argument, a reordered positional, a different return type — without treating
  it as breaking.
- **When removing is unavoidable: deprecate in one release, remove in a later
  one.** Keep the old name working (warn), and say so in `CHANGELOG.md` under
  `Deprecated` in the deprecating release and `Removed` in the removing one.
- The versioning rule (pre-1.0, stated in `README.md` →
  "Versioning and what a bump means"): a **`0.x.0` minor may add and may
  break**; a **`0.x.Y` patch is fixes only and never breaks.** So a breaking
  change is *permitted* but must ship in a minor bump with a changelog entry
  a consumer can act on. Never in a patch.

## Changelog is part of the change

`CHANGELOG.md` ([Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
hand-written). Add your entry to `## [Unreleased]` in the same PR as the code,
written **for a plugin author deciding whether to move a pin** — what they gain,
what they must change. `release.sh` refuses to release with an empty
`[Unreleased]`, so skipping it blocks the next release rather than shipping a
silent tag.

## How to release — `./release.sh`

```bash
./release.sh minor --dry-run    # ALWAYS this first
./release.sh minor              # 0.9.0 → 0.10.0, tag v0.10.0
./release.sh patch              # 0.9.0 → 0.9.1
./release.sh 1.0.0              # explicit
```

It requires `main`, clean, in sync with origin, then runs two gates —
**neither is optional and there is no flag to skip either**:

1. **Test gate** — `uv run --extra dev pytest` must pass. You cannot tag a
   broken library, because you cannot withdraw the tag.
2. **Changelog gate** — `## [Unreleased]` must be non-empty. It is then
   retitled `## [X.Y.Z] – <date>`, a fresh empty `[Unreleased]` is opened, and
   the compare link is appended.

Then it bumps `__version__` in `src/multiplai_core/__init__.py` (hatch's single
version source — `pyproject.toml` has no hand-written version string), commits
`chore(release): vX.Y.Z`, creates an annotated tag, and pushes `main` and the
tag with `git push --atomic`. Both gates also run in `--dry-run` (they are
read-only); only the branch/clean/sync preflight is advisory there.

## Where consumers pin — and why this script doesn't touch them

`release.sh` deliberately does **not** bump any consumer's pin (unlike
`multiplai-container/release.sh`, which has exactly one consumer). Core has
many, and pins move **deliberately, per consumer**, after someone reads the
changelog. Bumping them is a **separate PR in the consuming repo**:

- **`multiplai-cc-mktplace`** — PEP 723 headers across ~20 plugin scripts
  (`plugins/*/scripts/`, `plugins/*/skills/*/scripts/`). It has a test,
  `plugins/multiplai-context/tests/test_core_pin_consistency.py`, that fails on
  a partial bump — so bump a repo's pins together.
- **`multiplai-cc-mktplace`** — two vendored lockfiles, which pin a commit as
  well as a tag:
  `plugins/multiplai-dev/skills/buildme/scripts/uv.lock` and
  `plugins/multiplai-research/skills/deep-research/scripts/uv.lock`.
- **`multiplai-kit` / `multiplai-gui`** — any script or venv that installs core.

## Layout and tests

```
src/multiplai_core/   # the package (__init__.py holds __all__ and __version__)
tests/                # pytest suite
.github/workflows/    # CI: the same pytest gate on Python 3.11 and 3.12
```

```bash
uv run --extra dev pytest                      # the suite, in the project venv
uv run --python 3.11 --extra dev pytest -q     # the floor CI also tests
```

`requires-python = ">=3.11"`, so 3.11 is a supported floor, not an aspiration —
don't reach for 3.12-only syntax or private stdlib attributes.

## Design constraint worth preserving: the provider seam stays a seam

Cross-family model support is a **registry an out-of-tree backend joins**
(`register_provider`, `create_client_for`, `parse_model_spec`), not another
branch inside `create_client()`. Two reasons, both load-bearing:

- The tier/ceiling machinery in `env.py` ranks the **Claude family only**. A
  provider-qualified `MODEL=` is passed through verbatim, because coercing
  another vendor's model into a Claude tier would defeat the point of naming it.
- **No non-Anthropic backend ships here.** Choosing a provider means choosing
  whose API key and whose bill — a human decision, not a default. Registering
  one from the outside is three lines.

So: do not add an `openai`/`gemini`/`ollama` client to this repo, and keep a
bare model ID resolving to Anthropic (`DEFAULT_PROVIDER`) so every pre-seam
config value and code path behaves exactly as before.

## Dependency policy

Loose ranges, not exact pins: this is a library, and exact pins make it
unsolvable alongside a consumer or sibling dep that needs a different version.
The `claude-agent-sdk` bounds in `pyproject.toml` are the exception and the
comments there explain why (a 0.1→0.2 minor already shipped a breaking
result-message parse change) — read them before widening either bound. Keep the
`[sdk]` floor aligned with `multiplai-kit` / `multiplai-gui` so all SDK
consumers resolve compatibly.
