# Security

This library is resolved **from GitHub onto users' machines**: every installed
Multiplai plugin pins it by git tag in a PEP 723 script header
(`multiplai-core @ git+https://github.com/spikelab/multiplai-core@vX.Y.Z`), and
`uv` fetches and installs it the first time such a script runs. That makes this
repository a dependency that lands on machines whose owners never chose it
directly, and its integrity part of the plugins' contract with their users.

What keeps a pin meaning what it meant is the README's
[Availability guarantee](README.md#availability-guarantee): the repository
stays public, release tags are immutable — never moved, deleted, or reused —
and fixes ship as new tags. Nothing consumes `main`, and no pin floats, so a
version an installed plugin resolved yesterday is the version it resolves
tomorrow.

## Reporting a vulnerability

Email **security@spikelab.org**. Please include:

- the version (the `vX.Y.Z` tag, or the pin in the consuming script's PEP 723
  header);
- the module or file and, if you have it, the shortest reproduction;
- what an attacker gets — a credential read, an unexpected network egress, a
  command executed on the user's behalf.

Do **not** open a public issue for something exploitable. Anything else — a
confusing failure, a wrong doc, a missing export — is a normal issue and is
welcome as one.

Expect a first reply within a few days. This is a small project maintained by
one person; there is no bounty and no SLA, and saying so is more useful than
implying otherwise.

## What this code can reach

Useful when assessing impact. The library runs with the permissions of
whatever Claude Code session or plugin script imports it — it has no sandbox
of its own. Within that, it: reads configuration and memory files
(`paths`, `config`, `env` — including `.env` files that may carry tokens),
holds API keys in memory and talks to model endpoints (`model_client`,
`agent_runner`), and writes logs and session state. The `untrusted` module is
a prompt-injection *mitigation* used by consuming skills; it raises the cost
of an injection, it does not make one impossible.

## Which versions get fixes

**The latest tag, and only that one.** A fix ships as a new tag with a
[`CHANGELOG.md`](CHANGELOG.md) entry; existing tags are never patched in place
(see the Availability guarantee above). Because every consumer pins an exact
tag, a fix reaches an installed plugin only when that plugin's pin is bumped
and the plugin is updated — there is no auto-remediation path, which is the
price of pins that never move underneath you.

## Scope

In scope: the code in this repository (`src/multiplai_core/`), its release
process (`release.sh`), and the tags it publishes. Out of scope: the plugins
that consume it, Claude Code itself (report to Anthropic), `uv`, and
third-party dependencies. Issues in
[`multiplai-cc-mktplace`](https://github.com/spikelab/multiplai-cc-mktplace),
[`multiplai-kit`](https://github.com/spikelab/multiplai-kit) or
[`multiplai-container`](https://github.com/spikelab/multiplai-container) reach
the same address — say which repo.
