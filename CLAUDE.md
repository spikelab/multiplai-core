# multiplai-core — repo guide

The shared library every [Multiplai](https://github.com/spikelab/multiplai)
plugin depends on: path resolution, config/`.env`/`multiplai.conf` loading,
model + effort resolution, logging, cost ledger, the model client and the
provider seam, and the single SDK agent-invocation path.

**It is not on PyPI.** Every consumer resolves it **from GitHub**, and the
resolved code lands on other people's machines as part of an installed plugin.
This is the repo where a careless rename breaks installed plugins for users you
cannot reach. Read the section on consumers below **before you edit `src/`** —
it is short, and it is not what it used to be.

## The two rules that cannot be walked back

### 1. Tags are immutable

**Never move, delete, or re-cut a tag.** A `vX.Y.Z` tag resolves tomorrow to
exactly what it resolved yesterday — that is a public promise in
`README.md` ("Availability guarantee"), and the whole delivery model rests on
it. A bad release is fixed by cutting a **new** tag, never by re-pointing the
old one. Same for the repo itself: it stays public.

What a tag is **not**, as of 2026-08-04: the unit of delivery. It used to be —
every consumer named a tag in a PEP 723 header — and this file said so for
months after it stopped being true. Today the only consumer tracks `main` and
freezes resolution in a lockfile. A tag is now a **permanent reference point**
(pinnable by anyone who wants one) and the **anchor a `CHANGELOG` section is
written against**. See "Who consumes core" below for what actually delivers.

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

## Who consumes core, and how — verified 2026-08-05

**One repo installs core: `multiplai-cc-mktplace`.** It declares it **unpinned,
tracked from `main`**, in a single workspace member:

```toml
# plugins/multiplai-context/scripts/pyproject.toml
[tool.uv.sources]
multiplai-core = { git = "https://github.com/spikelab/multiplai-core", branch = "main" }
```

Resolution is frozen by **that repo's single root `uv.lock`, which records a
commit**. So merging here does not reach anyone by itself, and neither does
tagging. The fix travels when somebody runs, from the mktplace root:

```bash
uv lock --upgrade-package multiplai-core   # then commit the lock
```

Dependabot does not bump git-sourced dependencies, so that re-lock is a
deliberate manual act — which is the point: it lands as a reviewable diff in a
PR, with CI running against the new resolution.

**`multiplai-kit`, `multiplai-gui` and `multiplai-container` do not install
core.** They mention it in prose, or run mktplace's scripts through that repo's
member directories. There is nothing in them to bump.

### What this section used to say, and why it was wrong

Until 2026-08-04 every consumer pinned a tag in a PEP 723 header, and this file
described that world for months after it ended. If you have read an older copy —
or `release.sh`'s output before this change — it will have told you to bump
"~20 PEP 723 headers under `plugins/*/scripts`", "the two vendored lockfiles"
(`buildme/scripts/uv.lock`, `deep-research/scripts/uv.lock`), and pointed at
`plugins/multiplai-context/tests/test_core_pin_consistency.py` as the guard
against a partial bump.

**None of those exist.** The mktplace workspace consolidation deleted them: PEP
723 blocks and nested lockfiles are now both *rejected* by that repo's
`scripts/lint_workspace.py`, and the pin-consistency test went with them. Going
looking for those files is a dead end, and it is the specific confusion this
rewrite exists to stop.

`release.sh` still deliberately does **not** touch any consumer.
`multiplai-container/release.sh` bumps the kit's `CONTAINER_REF` because it has
exactly one consumer and one pin; core's consumer chooses its own moment.

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
