#!/usr/bin/env bash
#
# release.sh — cut a test-gated, changelog-gated, tagged release of
# multiplai-core, keeping the package version, the git tag, and the release
# notes in agreement.
#
# Why this exists: this library is resolved from GitHub, never PyPI, and a
# vX.Y.Z tag can never be moved or re-cut. A tag is a permanent, immutable
# reference point — the thing a consumer pins to when it wants resolution
# frozen, and the anchor the CHANGELOG section is written against. So the tag
# must be right the first time, which means three things must agree at the
# moment it is created:
#
#   src/multiplai_core/__init__.py::__version__   (hatch's single version source)
#   the git tag vX.Y.Z
#   the CHANGELOG.md section consumers read before bumping their pin
#
# Doing that by hand is how a tag ships with no notes (which is exactly what
# v0.6.0–v0.9.0 did). This script does ALL local work first — test gate,
# changelog rewrite, version bump, commit, tag — and pushes LAST, atomically, so
# the only failure window is a single push:
#
#   main clean + in sync  →  pytest MUST pass  →  ## [Unreleased] MUST be
#     non-empty  →  retitle notes + bump version (local commit + tag)
#     →  git push --atomic main + tag
#
# You cannot tag a failing test suite, and you cannot tag an undescribed change.
# There is deliberately NO flag to skip either gate.
#
# What this script does NOT do: touch any consumer. multiplai-container's
# release.sh bumps the kit's CONTAINER_REF because it has exactly one consumer
# and one pin; core's consumers each choose their own resolution strategy, so a
# release here is never a push to anybody. It prints a reminder of who consumes
# core and how, instead.
#
# Usage:
#   ./release.sh <major|minor|patch>     # bump from __version__
#   ./release.sh <X.Y.Z>                 # explicit version
#
# Options:
#   --dry-run   Show what would happen; make no writes, commits, tags or pushes.
#               The test and changelog gates still run FOR REAL (they are
#               read-only), and the branch/clean/sync preflight is downgraded to
#               warnings — so you can preview a release, and validate your notes,
#               from a feature branch before merging.
#   --yes, -y   Don't prompt before pushing.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VERSION_FILE="src/multiplai_core/__init__.py"
CHANGELOG="CHANGELOG.md"
REPO_URL="https://github.com/spikelab/multiplai-core"

# ---- args ------------------------------------------------------------------
BUMP=""
DRY_RUN=false; ASSUME_YES=false
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)   DRY_RUN=true ;;
    --yes|-y)    ASSUME_YES=true ;;
    -h|--help)   sed -n '2,48p' "$0"; exit 0 ;;
    major|minor|patch) BUMP="$1" ;;
    [0-9]*)      BUMP="$1" ;;
    *) echo "release: unknown argument '$1' (see --help)" >&2; exit 2 ;;
  esac
  shift
done
[ -n "$BUMP" ] || { echo "release: need a version or major|minor|patch (see --help)" >&2; exit 2; }

say()  { printf '  %s\n' "$*"; }
step() { printf '\n▸ %s\n' "$*"; }
die()  { printf 'release: %s\n' "$*" >&2; exit 1; }
# In a dry run the preflight reports instead of aborting, so a release can be
# previewed (and its notes validated) from a feature branch. A real run aborts.
softdie() { if $DRY_RUN; then say "!! would fail: $*"; else die "$*"; fi; }
# Execute argv directly — arguments are never re-parsed by the shell, so a
# tag/version containing shell metachars can't inject.
run()  { if $DRY_RUN; then printf '  [dry-run] %s\n' "$*"; else "$@"; fi; }

# ---- preflight: clean, on main, up to date ---------------------------------
step "Preflight"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not a git repo"
[ -f "$VERSION_FILE" ] || die "$VERSION_FILE not found — run this from the repo root"
[ -f "$CHANGELOG" ] || die "$CHANGELOG not found — release notes are not optional"
command -v uv >/dev/null 2>&1 || die "uv not found — the test gate needs it (https://docs.astral.sh/uv/)"

BRANCH="$(git branch --show-current)"
[ "$BRANCH" = "main" ] || softdie "must release from main (on '${BRANCH:-<detached>}')"
[ -z "$(git status --porcelain)" ] || softdie "working tree not clean — commit or stash first"
if git fetch --quiet origin main 2>/dev/null; then
  LOCAL="$(git rev-parse @)"
  if REMOTE="$(git rev-parse origin/main 2>/dev/null)"; then
    [ "$LOCAL" = "$REMOTE" ] || softdie "local HEAD ($(git rev-parse --short @)) not in sync with origin/main ($(git rev-parse --short origin/main)) — pull/push first"
  else
    softdie "no origin/main — set an upstream first"
  fi
else
  softdie "cannot fetch origin/main"
fi
say "branch=${BRANCH:-<detached>} $($DRY_RUN && echo '(dry run: preflight advisory)' || echo '(clean, in sync)')"

# ---- compute next version --------------------------------------------------
step "Version"
CUR="$(sed -n 's/^__version__ = "\([^"]*\)".*/\1/p' "$VERSION_FILE" | head -1)"
[ -n "$CUR" ] || die "could not read __version__ from $VERSION_FILE"
IFS='.' read -r MA MI PA <<<"$CUR"; MA=${MA:-0}; MI=${MI:-0}; PA=${PA:-0}
case "$BUMP" in
  major) NEW="$((MA+1)).0.0" ;;
  minor) NEW="${MA}.$((MI+1)).0" ;;
  patch) NEW="${MA}.${MI}.$((PA+1))" ;;
  *)     [[ "$BUMP" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "bad version '$BUMP' (want X.Y.Z)"; NEW="$BUMP" ;;
esac
TAG="v$NEW"
PREV_TAG="v$CUR"
TODAY="$(date -u +%Y-%m-%d)"
say "current __version__=$CUR  →  new=$NEW  →  tag=$TAG"
git rev-parse -q --verify "refs/tags/$TAG" >/dev/null && die "tag $TAG already exists locally — tags are immutable; pick a new version"
git ls-remote --exit-code --tags origin "$TAG" >/dev/null 2>&1 && die "tag $TAG already exists on origin — tags are immutable; pick a new version"
git rev-parse -q --verify "refs/tags/$PREV_TAG" >/dev/null || say "note: no tag $PREV_TAG to compare from; the changelog link will still be written"

# ---- test gate (runs first, and in dry runs too) ---------------------------
# The analogue of multiplai-container's `docker build` gate: you cannot tag a
# library whose own suite fails, because the tag can never be withdrawn. It is
# read-only, so it runs in a dry run as well — that is the point of a dry run.
step "Test gate"
say "uv run --extra dev pytest — a tag is only cut if this passes"
uv run --extra dev pytest -q || die "tests failed — refusing to tag a broken release"
say "tests passed"

# ---- changelog gate --------------------------------------------------------
# `## [Unreleased]` must exist and carry at least one non-blank content line.
# An empty section means nobody wrote down what a consumer would be adopting.
step "Changelog gate"
grep -q '^## \[Unreleased\]' "$CHANGELOG" || die "no '## [Unreleased]' section in $CHANGELOG — add release notes before releasing"
NOTES="$(awk '
  /^## \[Unreleased\]/ { inside=1; next }
  inside && /^## / { exit }
  inside && /^\[[^]]+\]: / { next }
  inside { print }
' "$CHANGELOG" | grep -c '[^[:space:]]' || true)"
[ "${NOTES:-0}" -gt 0 ] || die "'## [Unreleased]' in $CHANGELOG is empty — describe what a consumer is adopting before cutting $TAG"
say "$NOTES line(s) of unreleased notes found"
printf '\n--- notes that would ship as %s ---\n' "$TAG"
awk '/^## \[Unreleased\]/{inside=1;next} inside && /^## /{exit} inside' "$CHANGELOG"
printf -- '--- end notes ---\n'

# ---- plan ------------------------------------------------------------------
step "Plan"
say "retitle '## [Unreleased]' → '## [$NEW] – $TODAY' in $CHANGELOG, insert a fresh empty [Unreleased]"
say "append compare link  [$NEW]: $REPO_URL/compare/$PREV_TAG...$TAG"
say "set __version__ = \"$NEW\" in $VERSION_FILE"
say "commit 'chore(release): $TAG', annotated tag $TAG"
say "push --atomic origin main $TAG  ($(git remote get-url origin 2>/dev/null || echo 'no origin'))"
if ! $ASSUME_YES && ! $DRY_RUN; then
  printf '\nProceed? [y/N] '; read -r ans; [ "$ans" = "y" ] || { echo "aborted."; exit 1; }
fi

# ---- rewrite changelog + version (local only) ------------------------------
step "Preparing $TAG (local)"
if $DRY_RUN; then
  say "[dry-run] rewrite $CHANGELOG and $VERSION_FILE"
else
  NEW="$NEW" TODAY="$TODAY" TAG="$TAG" PREV_TAG="$PREV_TAG" REPO_URL="$REPO_URL" \
  CHANGELOG="$CHANGELOG" python3 - <<'PY' || die "changelog rewrite failed"
import os, pathlib, re, sys

new, today = os.environ["NEW"], os.environ["TODAY"]
tag, prev, url = os.environ["TAG"], os.environ["PREV_TAG"], os.environ["REPO_URL"]
p = pathlib.Path(os.environ["CHANGELOG"])
text = p.read_text()

# Retitle the section and open a fresh, empty one above it.
text, n = re.subn(
    r"^## \[Unreleased\]\n",
    f"## [Unreleased]\n\n## [{new}] – {today}\n",
    text, count=1, flags=re.M,
)
if n != 1:
    sys.exit("could not retitle [Unreleased]")

# Point the Unreleased compare link at the new tag and add the release's own.
text, n = re.subn(
    r"^\[Unreleased\]: .*$",
    f"[Unreleased]: {url}/compare/{tag}...HEAD\n[{new}]: {url}/compare/{prev}...{tag}",
    text, count=1, flags=re.M,
)
if n != 1:
    # No link footer yet — append both.
    text = text.rstrip("\n") + (
        f"\n\n[Unreleased]: {url}/compare/{tag}...HEAD"
        f"\n[{new}]: {url}/compare/{prev}...{tag}\n"
    )
p.write_text(text)
PY
  tmp="$(mktemp)"
  sed -E "s/^__version__ = \"[^\"]*\"/__version__ = \"$NEW\"/" "$VERSION_FILE" > "$tmp"
  grep -qF "__version__ = \"$NEW\"" "$tmp" || { rm -f "$tmp"; die "version bump failed — __version__ not updated"; }
  # Write content in place so the file keeps its inode and mode.
  cat "$tmp" > "$VERSION_FILE"
  rm -f "$tmp"
  say "changelog retitled, __version__ = $NEW"
fi
run git add "$CHANGELOG" "$VERSION_FILE"
run git commit -q -m "chore(release): $TAG"
run git tag -a "$TAG" -m "Release $TAG"
$DRY_RUN || say "committed + tagged locally"

# ---- publish ---------------------------------------------------------------
# --atomic: main and the tag land together or not at all — a raced rejection of
# main can't leave an orphaned public tag (which could never be withdrawn).
step "Publishing"
run git push --atomic --quiet origin main "$TAG"
$DRY_RUN || say "pushed main + $TAG"

# ---- done ------------------------------------------------------------------
if $DRY_RUN; then
  step "Dry run complete — nothing was written, committed, tagged or pushed"
else
  step "Released $TAG"
fi
say "Nothing upgrades on its own — every consumer decides when to move."
say ""
say "Who consumes core, and how (verified 2026-08-05):"
say "  • multiplai-cc-mktplace — the ONLY repo that installs core. It declares"
say "    it UNPINNED and tracked from main, in one workspace member:"
say "      plugins/multiplai-context/scripts/pyproject.toml  [tool.uv.sources]"
say "    Resolution is frozen by the single root uv.lock, which records a"
say "    COMMIT. So this tag is not what it picks up: it moves when someone"
say "    runs, from that repo's root,"
say "      uv lock --upgrade-package multiplai-core"
say "    and commits the lock. Dependabot does not bump git-sourced deps, so"
say "    that re-lock is a deliberate, manual act — a reviewable diff in a PR."
say "  • multiplai-kit / multiplai-gui / multiplai-container — do NOT install"
say "    core. They reference it in prose, or run mktplace's scripts through"
say "    that repo's member dirs. Nothing to bump."
say ""
say "So what is the tag for? It is the permanent reference a future consumer"
say "can pin to, and the anchor the $NEW CHANGELOG section is written against."
say "It is not, today, how the fix reaches anyone."
