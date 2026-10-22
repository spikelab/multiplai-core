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

import hashlib
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

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
    section = re.search(r"^##\s+Capabilities\s*$(.*?)(?=^##\s|\Z)", proposal, re.MULTILINE | re.DOTALL)
    scope = section.group(1) if section else re.sub(r"^##\s+Impact\s*$.*?(?=^##\s|\Z)", "", proposal, flags=re.MULTILINE | re.DOTALL)
    return list(dict.fromkeys(re.findall(r"^-\s+`([a-z0-9]+(?:-[a-z0-9]+)*)`", scope, re.MULTILINE)))


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
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    blocks = []
    for requirement in re.finditer(r"^### Requirement:([^\n]*)(.*?)(?=^### Requirement:|^## |\Z)",
                                   text, re.MULTILINE | re.DOTALL):
        sections = re.findall(r"^## (\w+) Requirements", text[:requirement.start()], re.MULTILINE)
        scenarios = tuple(
            ScenarioBlock(name=s.group(1).strip(), body=s.group(2))
            for s in re.finditer(r"^#### Scenario:([^\n]*)(.*?)(?=^#### Scenario:|\Z)",
                                 requirement.group(2), re.MULTILINE | re.DOTALL)
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


# --- the ready record in .change.yaml ----------------------------------------


def read_change_meta(change_dir: Path) -> dict[str, Any]:
    """The mapping in ``<change_dir>/.change.yaml``; ``{}`` when there is no
    such file or it is empty.

    Raises ValueError, naming the file, when it is not valid YAML or does not
    hold a mapping, so a damaged record is never read as an empty one.
    """
    path = Path(change_dir) / CHANGE_META_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path} is not valid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a mapping (found {type(data).__name__})")
    return data


def write_change_meta(change_dir: Path, updates: Mapping[str, Any]) -> dict[str, Any]:
    """Set the keys in *updates* in ``<change_dir>/.change.yaml`` and return
    the whole mapping written.

    Every other key already in the file is kept, in its order; new keys go at
    the end. A value of None is written as null, not removed. The file is
    replaced atomically (a temporary file in the same directory, then
    ``os.replace``), so a stop part way leaves the old record or the new one.
    Raises ValueError as ``read_change_meta`` does when the existing file is
    damaged, rather than overwriting it.
    """
    change_dir = Path(change_dir)
    meta = read_change_meta(change_dir)
    meta.update(updates)
    text = yaml.safe_dump(meta, default_flow_style=False, sort_keys=False, allow_unicode=True)
    fd, tmp = tempfile.mkstemp(dir=change_dir, prefix=".change.", suffix=".yaml.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, change_dir / CHANGE_META_FILE)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return meta


def spec_files(change_dir: Path) -> list[Path]:
    """The files ``spec_revision`` hashes, in the order it hashes them:
    ``proposal.md``, ``use-cases.md`` and ``requirements/*.md`` (not
    recursive), those that exist as files, sorted by their path relative to
    *change_dir* in POSIX form."""
    change_dir = Path(change_dir)
    candidates = [change_dir / "proposal.md", change_dir / "use-cases.md"]
    req_dir = change_dir / "requirements"
    if req_dir.is_dir():
        candidates.extend(req_dir.glob("*.md"))
    files = [p for p in candidates if p.is_file()]
    return sorted(files, key=lambda p: p.relative_to(change_dir).as_posix())


def spec_revision(change_dir: Path) -> str:
    r"""A sha256 hex digest of a change's "what" documents.

    The algorithm, exactly:

    1. Take ``spec_files(change_dir)``: ``proposal.md``, ``use-cases.md`` and
       ``requirements/*.md`` (not recursive) that exist as regular files.
    2. Sort them by their path relative to *change_dir*, in POSIX form
       (``proposal.md`` < ``requirements/a.md`` < ``use-cases.md``), comparing
       the strings by code point.
    3. For each file, read its bytes and normalise line endings to LF:
       every ``\r\n`` becomes ``\n``, then every remaining ``\r`` becomes
       ``\n``. Nothing else is changed (no decoding, no trailing-whitespace or
       final-newline changes).
    4. Feed one sha256 with, per file in that order: the relative path encoded
       as UTF-8, a NUL byte, the length of the normalised content as ASCII
       decimal digits, a NUL byte, then the normalised content.

    So the digest changes when a file's content changes (other than CRLF/CR
    versus LF), when a file is added, removed or renamed, and never with
    modification times or git state. A change with none of the files hashes
    the empty input (``e3b0c442...``). Raises OSError when a listed file
    cannot be read.
    """
    change_dir = Path(change_dir)
    digest = hashlib.sha256()
    for path in spec_files(change_dir):
        content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        rel = path.relative_to(change_dir).as_posix().encode("utf-8")
        digest.update(rel + b"\0" + str(len(content)).encode("ascii") + b"\0" + content)
    return digest.hexdigest()


def is_ready(change_dir: Path) -> tuple[bool, str]:
    """``(True, "")`` when the change's specs are marked ready and unchanged
    since; otherwise ``(False, reason)``, *reason* being one line for a person.

    Ready means ``.change.yaml`` has ``spec_status: ready`` and a
    ``spec_revision`` equal to ``spec_revision(change_dir)`` now. A missing or
    damaged ``.change.yaml``, a record written before these keys existed, a
    ``draft`` status and an edit after marking all give False. An open
    ``spec_issues`` does not: callers that must stop on it read it themselves.
    """
    change_dir = Path(change_dir)
    if not (change_dir / CHANGE_META_FILE).is_file():
        return False, f"no {CHANGE_META_FILE} in {change_dir}"
    try:
        meta = read_change_meta(change_dir)
    except (OSError, UnicodeError, ValueError) as exc:
        return False, f"{CHANGE_META_FILE} cannot be read: {exc}".replace("\n", " ")
    status = meta.get("spec_status")
    if status is None:
        return False, "spec_status is not set: the specs were never marked ready"
    if status != SPEC_STATUS_READY:
        return False, f"spec_status is {status!r}, not {SPEC_STATUS_READY!r}"
    saved = meta.get("spec_revision")
    if not saved:
        return False, "spec_status is ready but spec_revision is missing"
    try:
        current = spec_revision(change_dir)
    except OSError as exc:
        return False, f"the spec files cannot be read: {exc}".replace("\n", " ")
    if saved != current:
        return False, (f"the specs changed after they were marked ready "
                       f"(spec_revision {str(saved)[:12]}, files now {current[:12]})")
    return True, ""


def mark_ready(change_dir: Path, now: datetime | None = None) -> str:
    """Record the change's specs as ready, as they are on disk now, and return
    the revision recorded.

    Sets ``spec_status: ready``, ``spec_revision: spec_revision(change_dir)``
    and ``spec_checked_at`` (*now*, default the current time, as ISO 8601 in
    UTC to the second; a naive *now* is taken as UTC). ``spec_issues`` and
    every other key are left as they are. This records the result of a check;
    it does not check anything.
    """
    revision = spec_revision(change_dir)
    when = now or datetime.now(UTC)
    when = when.replace(tzinfo=UTC) if when.tzinfo is None else when.astimezone(UTC)
    write_change_meta(change_dir, {
        "spec_status": SPEC_STATUS_READY,
        "spec_revision": revision,
        "spec_checked_at": when.isoformat(timespec="seconds"),
    })
    return revision


def mark_draft(change_dir: Path) -> None:
    """Set ``spec_status: draft``. The saved revision and check time stay, as
    the record of the last check; ``is_ready`` is False until ``mark_ready``."""
    write_change_meta(change_dir, {"spec_status": SPEC_STATUS_DRAFT})


def set_spec_issues(change_dir: Path, path: Path | str | None) -> None:
    """Set ``spec_issues`` to the open ``spec-issues.md`` at *path*, or to null
    when *path* is None.

    A path inside *change_dir* is saved relative to it (``spec-issues.md``),
    so the record still holds after the change dir moves into the archive;
    any other path is saved as given.
    """
    change_dir = Path(change_dir)
    value: str | None = None
    if path is not None:
        p = Path(path)
        try:
            value = (change_dir / p if not p.is_absolute() else p).resolve().relative_to(
                change_dir.resolve()).as_posix()
        except ValueError:
            value = str(path)
    write_change_meta(change_dir, {"spec_issues": value})


def clear_spec_issues(change_dir: Path) -> None:
    """Set ``spec_issues`` to null: no requirements issue is open."""
    set_spec_issues(change_dir, None)
