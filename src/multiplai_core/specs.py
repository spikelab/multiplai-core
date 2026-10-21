"""The ``specs/`` directory format that buildme and pm-spec both read and write.

Ported from buildme's ``build_pipeline/change_manager.py`` in
``multiplai-cc-mktplace``, behaviour unchanged, so the skill that writes a
change's requirements and the skill that builds from them parse them with the
same code. Layout::

    specs/
      changes/<name>/              one active change
        .change.yaml               metadata, including the ready record below
        proposal.md                ## Capabilities lists one `name` per capability
        use-cases.md
        requirements/<cap>.md      a delta: ## ADDED/MODIFIED/REMOVED Requirements
      registry/<cap>.md            the merged current behaviour, one file per capability
      archive/YYYY-MM-DD-<name>/   archived changes

A requirement file is markdown: ``### Requirement: <name>`` blocks, each with
one or more ``#### Scenario: <name>`` blocks whose text holds a ``WHEN`` line
and a ``THEN`` line. A requirement under ``## REMOVED Requirements`` needs no
scenario.

The ready record is four keys in ``.change.yaml``: ``spec_status`` (``draft``
or ``ready``), ``spec_revision`` (``spec_revision()`` of the files when they
were marked ready), ``spec_checked_at`` (when) and ``spec_issues`` (the path
of an open ``spec-issues.md``, or null). ``is_ready()`` is the check a builder
runs before it starts.

Standard library plus ``pyyaml`` (already a core dependency). No model call.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

CHANGE_META_FILE = ".change.yaml"
SPEC_STATUS_DRAFT = "draft"
SPEC_STATUS_READY = "ready"


# --- change names and directories -------------------------------------------


def normalize_change_name(name: str) -> str:
    """Normalize a change name to a safe kebab-case slug.

    Strips path separators, dots, and other non-word characters, so a name can
    never traverse outside specs/changes/ (e.g. '../../foo' → 'foo').

    A non-empty name that normalizes to nothing (e.g. '!!!') is refused with
    ValueError — silently returning "" would make change_dir collapse to
    specs/changes and the branch to 'buildme/', dying much later with a raw
    git error. An empty input stays "" (an unset --change is legitimate).
    """
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    if name and not slug:
        raise ValueError(
            f"Change name {name!r} contains no usable characters "
            f"(letters, digits, '-'). Pick a real name, e.g. --change my-feature."
        )
    return slug


def archived_change_dirs(specs_dir: Path, name: str) -> list[Path]:
    """All of this change's archive directories, sorted oldest → newest.

    Matches exactly ``specs/archive/YYYY-MM-DD-<slug>`` (the `archive_change`
    naming scheme). The match is anchored on the whole dirname — a loose
    ``*-<slug>`` glob would also catch every OTHER change whose slug merely
    *ends* with this one (change ``foo`` matching ``2026-07-26-bar-foo``),
    silently crossing two changes' files.
    """
    norm = normalize_change_name(name)
    if not norm:
        return []
    pattern = re.compile(rf"^\d{{4}}-\d{{2}}-\d{{2}}-{re.escape(norm)}$")
    archive_root = specs_dir / "archive"
    if not archive_root.is_dir():
        return []
    return sorted(d for d in archive_root.iterdir() if d.is_dir() and pattern.match(d.name))


# --- proposal capabilities and requirement files ----------------------------


def expected_capabilities(proposal: str) -> list[str]:
    """The capability names a proposal lists, in order, without duplicates.

    A capability is a list item that starts with a backticked kebab-case name
    (``- `user-auth``` or ``- `user-auth`: description``). Only the
    ``## Capabilities`` section is read, because the proposal template wraps
    every dependency name in backticks under ``## Impact``. A proposal with no
    ``## Capabilities`` heading is read whole, minus ``## Impact``.
    """
    section = re.search(r"^##\s+Capabilities\s*$(.*?)(?=^##\s|\Z)", proposal, re.M | re.S)
    scope = section.group(1) if section else re.sub(r"^##\s+Impact\s*$.*?(?=^##\s|\Z)", "", proposal, flags=re.M | re.S)
    return list(dict.fromkeys(re.findall(r"^-\s+`([a-z0-9]+(?:-[a-z0-9]+)*)`", scope, re.M)))


@dataclass(frozen=True)
class ScenarioBlock:
    """One ``#### Scenario:`` block: its name (stripped) and the text after
    the heading line, up to the next scenario or the end of its requirement."""

    name: str
    body: str


@dataclass(frozen=True)
class RequirementBlock:
    """One ``### Requirement:`` block of a requirements file.

    ``name`` is the heading text, stripped (``""`` when the heading is empty).
    ``section`` is the operation of the last ``## <word> Requirements`` heading
    above it (``"ADDED"``, ``"MODIFIED"``, ``"REMOVED"``, ``"RENAMED"``), or
    None when there is none, as in a registry file. ``body`` is the text after
    the heading line, up to the next requirement, the next ``## `` heading or
    the end. ``scenarios`` are the ``#### Scenario:`` blocks in ``body``.
    """

    name: str
    section: str | None
    body: str
    scenarios: tuple[ScenarioBlock, ...]


def parse_requirements(text: str) -> list[RequirementBlock]:
    """The requirement blocks of a requirements or registry file, in order.

    HTML comments (``<!-- ... -->``) are removed first, so a template's
    placeholder comments never count as a name or a scenario. This is the
    parse ``valid_requirement_text`` checks.
    """
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    blocks = []
    for requirement in re.finditer(r"^### Requirement:([^\n]*)(.*?)(?=^### Requirement:|^## |\Z)",
                                   text, re.M | re.S):
        sections = re.findall(r"^## (\w+) Requirements", text[:requirement.start()], re.M)
        scenarios = tuple(
            ScenarioBlock(name=s.group(1).strip(), body=s.group(2))
            for s in re.finditer(r"^#### Scenario:([^\n]*)(.*?)(?=^#### Scenario:|\Z)",
                                 requirement.group(2), re.M | re.S)
        )
        blocks.append(RequirementBlock(
            name=requirement.group(1).strip(),
            section=sections[-1] if sections else None,
            body=requirement.group(2),
            scenarios=scenarios,
        ))
    return blocks


def valid_requirement(path: Path) -> bool:
    """``valid_requirement_text`` of the file at *path*; False when it cannot
    be read or decoded."""
    try:
        text = path.read_text()
    except (OSError, UnicodeError):
        return False
    return valid_requirement_text(text)


def valid_requirement_text(text: str) -> bool:
    """Validate each requirement, preserving the scenario-free REMOVED format.

    True when the text has at least one requirement, every requirement has a
    name, and every requirement outside ``## REMOVED Requirements`` has at
    least one scenario, each with a name, a ``WHEN`` and a ``THEN`` followed
    by text on the same line.
    """
    requirements = parse_requirements(text)
    if not requirements:
        return False
    for requirement in requirements:
        if not requirement.name:
            return False
        if requirement.section == "REMOVED":
            continue
        if not requirement.scenarios:
            return False
        for scenario in requirement.scenarios:
            if not scenario.name or not all(
                re.search(rf"\b{keyword}\b[^\n]*\w", scenario.body)
                for keyword in ("WHEN", "THEN")
            ):
                return False
    return True


# --- merging a change's requirements into the registry ----------------------

_DELTA_OPS = ("ADDED", "MODIFIED", "REMOVED", "RENAMED")


def _delta_section(delta: str, op: str) -> re.Match | None:
    """The ``## <op> Requirements`` section of *delta*: group 1 is its text,
    up to the next ``## <other op> Requirements`` heading or the end."""
    others = "|".join(o for o in _DELTA_OPS if o != op)
    return re.search(rf"## {op} Requirements\s*\n(.*?)(?=\n## (?:{others}) Requirements|\Z)",
                     delta, re.DOTALL)


def apply_delta(main: str, delta: str) -> str:
    """Apply delta spec operations (ADDED/MODIFIED/REMOVED) to main spec.

    In this order: the ADDED section's text is appended to *main*; each
    requirement named under REMOVED is deleted from *main*; each requirement
    under MODIFIED replaces the block of the same name in *main*. A block runs
    from its ``### Requirement:`` heading to the next one or the end, and is
    matched by name as a prefix of the heading text. RENAMED is not applied.
    The result ends with exactly one newline.
    """
    # Extract ADDED requirements — append to main
    added_match = _delta_section(delta, "ADDED")
    if added_match:
        added_block = added_match.group(1).strip()
        if added_block:
            main = main.rstrip() + "\n\n" + added_block + "\n"

    # Extract REMOVED requirements — remove matching blocks from main
    removed_match = _delta_section(delta, "REMOVED")
    if removed_match:
        removed_block = removed_match.group(1)
        req_names = re.findall(r"### Requirement:\s*(.+)", removed_block)
        for name in req_names:
            pattern = re.compile(
                rf"### Requirement:\s*{re.escape(name.strip())}.*?"
                rf"(?=### Requirement:|\Z)",
                re.DOTALL,
            )
            main = pattern.sub("", main)

    # Extract MODIFIED requirements — replace matching blocks
    modified_match = _delta_section(delta, "MODIFIED")
    if modified_match:
        mod_block = modified_match.group(1)
        mod_reqs = re.split(r"(?=### Requirement:)", mod_block)
        for req in mod_reqs:
            name_match = re.match(r"### Requirement:\s*(.+)", req)
            if not name_match:
                continue
            name = name_match.group(1).strip()
            pattern = re.compile(
                rf"### Requirement:\s*{re.escape(name)}.*?"
                rf"(?=### Requirement:|\Z)",
                re.DOTALL,
            )
            # Use a function replacement so backslash sequences in the
            # requirement text (e.g. a regex like `\d+` or `\1` in a BDD
            # scenario) are inserted literally instead of being interpreted
            # as regex backreferences (which would raise re.error or corrupt
            # the output).
            replacement = req.strip() + "\n\n"
            main = pattern.sub(lambda _m: replacement, main)

    return main.strip() + "\n"


def merge_delta_requirements(change_dir: Path, registry_dir: Path) -> None:
    """Merge delta requirements from a change into the main registry.

    Reads flat files from changes/<name>/requirements/<capability>.md and
    merges them into registry/<capability>.md: a capability with no registry
    file gets the delta copied as-is, any other gets ``apply_delta``. Does
    nothing when the change has no ``requirements/`` directory.
    """
    delta_dir = change_dir / "requirements"
    if not delta_dir.exists():
        return

    registry_dir.mkdir(parents=True, exist_ok=True)
    for delta_file in sorted(delta_dir.glob("*.md")):
        capability_name = delta_file.stem
        target_file = registry_dir / f"{capability_name}.md"

        delta_content = delta_file.read_text()
        if not target_file.exists():
            # New capability — copy the delta as the main registry file
            target_file.write_text(delta_content)
            log.info("Created main registry: %s", target_file)
            continue

        # Merge into existing main registry file
        main_content = target_file.read_text()
        merged = apply_delta(main_content, delta_content)
        target_file.write_text(merged)
        log.info("Merged delta into: %s", target_file)


def archive_change(change_dir: Path, specs_dir: Path, *, merge_specs: bool = True,
                   today: date | None = None) -> Path:
    """Archive a completed change.

    Moves change to <specs_dir>/archive/YYYY-MM-DD-<name>/ (*today*, default
    ``date.today()``) and, with *merge_specs*, first merges its delta
    requirements into <specs_dir>/registry/. Returns the archive path.
    """
    archive_dir = specs_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"{(today or date.today()).isoformat()}-{change_dir.name}"
    dest = archive_dir / archive_name

    if merge_specs:
        merge_delta_requirements(change_dir, specs_dir / "registry")

    shutil.move(str(change_dir), str(dest))
    log.info("Archived change '%s' to %s", change_dir.name, dest)
    return dest
