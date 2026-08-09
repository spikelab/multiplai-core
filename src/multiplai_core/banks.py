"""Memory banks — the ordered list of corpora that make up "memory".

A **bank** is a directory of memory files. Historically there was exactly
one, at ``paths.memory_dir()``, and every consumer joined filenames onto it.
That single path is now the *first* bank, named ``personal``, and it stays
exactly where it was: **absent configuration, :func:`load_banks` returns one
bank at today's path**, so nothing that reads ``memory_dir`` changes.

Additional banks are **git repositories somebody else may also write to** —
a team's shared notes, a household's shared facts. Three properties follow
from that and are enforced here rather than left to each caller:

* **A shared bank is never directly written.** ``mode`` for anything that is
  not ``personal`` is ``ro`` (read and route only) or ``propose``
  (contributions leave as a pull request). A configuration asking for direct
  writes to a shared bank is *coerced down* to ``propose`` with a warning —
  see :func:`_coerce_mode`. There is no configuration that yields a direct
  write to a bank the local user does not solely own.
* **Bank content is authored by other people.** :attr:`MemoryBank.is_shared`
  is the flag every rendering path keys off to fence the content as
  untrusted (:mod:`multiplai_core.untrusted`). It is a property of the bank,
  not of any individual file, so a new file in a subscribed bank is fenced
  the moment it syncs.
* **A bank may not nest inside the personal corpus.** Otherwise its files
  are discovered twice — once as bank files and once as personal ones — and
  the no-duplication rule the whole design rests on is broken by layout.

Parsing is **fail-open to the narrow state**: a missing, unreadable,
malformed, or partly-invalid config yields the banks it could understand,
always including ``personal``. A typo must not be able to add a bank, and
must not be able to break a session either.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "BANKS_FILENAME",
    "BANK_MODES",
    "DEFAULT_SHARED_MODE",
    "PERSONAL_BANK",
    "PERSONAL_MODE",
    "MemoryBank",
    "bank_ref",
    "load_banks",
    "parse_bank_ref",
    "personal_bank",
    "split_bank_ref",
]

#: The file a workspace declares its banks in, beside ``project-map.yaml``.
BANKS_FILENAME = "memory-banks.yaml"

#: The always-present first bank. Its name is reserved.
PERSONAL_BANK = "personal"

#: The personal bank's mode. Only the personal bank ever has it.
PERSONAL_MODE = "rw"

#: Modes a *shared* bank may have. ``rw`` is deliberately not among them.
BANK_MODES: tuple[str, ...] = ("ro", "propose")

#: What a shared bank gets when it names no mode, or names an unusable one.
DEFAULT_SHARED_MODE = "propose"

#: Bank names are used as a path segment and as a reference prefix, so they
#: are restricted to a shape that can be neither a traversal nor a filename.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

_SYNC_MODES: tuple[str, ...] = ("session-start", "manual")


@dataclasses.dataclass(frozen=True)
class MemoryBank:
    """One corpus of memory files.

    ``path`` is absolute and is the directory memory filenames join onto —
    the same role ``memory_dir`` has always played for the personal bank.
    """

    name: str
    path: Path
    mode: str = PERSONAL_MODE
    remote: str = ""
    sync: str = "session-start"

    @property
    def is_personal(self) -> bool:
        """Is this the local, solely-owned corpus?"""
        return self.name == PERSONAL_BANK

    @property
    def is_shared(self) -> bool:
        """Is this content other people write?

        The single question every trust decision keys off: fencing on
        injection, refusing a direct write, requiring a pull request.
        """
        return not self.is_personal

    @property
    def accepts_direct_writes(self) -> bool:
        """May the local memory-apply path write into this bank?

        True for ``personal`` and nothing else, in any configuration.
        """
        return self.is_personal

    @property
    def accepts_contributions(self) -> bool:
        """May local content be *proposed* to this bank (as a pull request)?"""
        return self.mode == "propose"

    @property
    def syncs_at_session_start(self) -> bool:
        """Should a background pull run for this bank at session start?"""
        return self.is_shared and bool(self.remote) and self.sync == "session-start"

    def ref(self, filename: str) -> str:
        """The catalog/router reference for *filename* in this bank.

        ``"dev.md"`` for the personal bank, ``"team/dev.md"`` for a bank
        named ``team``. The personal form is unprefixed on purpose: every
        existing catalog entry, proposal target, and receipt already spells
        it that way, and rewriting them would be churn with no gain.
        """
        return bank_ref(self.name, filename)

    def file(self, filename: str) -> Path:
        """``path / filename`` — the one place a bank filename is joined."""
        return self.path / filename


def bank_ref(bank_name: str, filename: str) -> str:
    """``"dev.md"`` for the personal bank, ``"<bank>/dev.md"`` otherwise."""
    if not bank_name or bank_name == PERSONAL_BANK:
        return filename
    return f"{bank_name}/{filename}"


def split_bank_ref(ref: str) -> tuple[str, str]:
    """``"team/dev.md"`` → ``("team", "dev.md")``; ``"dev.md"`` → ``("personal", "dev.md")``.

    Purely lexical and total: it never touches the filesystem and never
    raises. A ref with more than one separator keeps everything after the
    first as the filename, which then fails the caller's own filename check
    — refusing here would only move the same rejection earlier.
    """
    text = (ref or "").strip()
    if "/" not in text:
        return PERSONAL_BANK, text
    name, _, filename = text.partition("/")
    return (name.strip() or PERSONAL_BANK), filename.strip()


def parse_bank_ref(
    ref: str, banks: Iterable[MemoryBank]
) -> tuple[Optional[MemoryBank], str]:
    """Resolve *ref* against *banks*: ``(bank, filename)``, bank ``None`` if unknown.

    An unknown bank name is not an error to raise — a ref can outlive an
    unsubscribe — but it *is* a refusal: the caller gets ``None`` and must
    not fall back to the personal bank, which is how a stale ``team/dev.md``
    would otherwise become a write to a personal file of that name.
    """
    name, filename = split_bank_ref(ref)
    for bank in banks:
        if bank.name == name:
            return bank, filename
    return None, filename


def personal_bank(memory_dir: Path) -> MemoryBank:
    """The one bank that always exists, at today's path."""
    return MemoryBank(
        name=PERSONAL_BANK,
        path=Path(memory_dir),
        mode=PERSONAL_MODE,
        remote="",
        sync="manual",
    )


def _coerce_mode(name: str, raw: Any) -> str:
    """The mode a shared bank actually gets.

    ``rw`` is accepted as *input* and coerced to ``propose``: people will
    write it (the July design listed it), and the useful reading of "I want
    to contribute to this bank" is the pull-request path, not a silent
    demotion to read-only that quietly drops contributions on the floor.
    Anything unrecognised gets the same treatment for the same reason —
    ``propose`` still cannot write to the bank, so the coercion cannot widen
    anything.
    """
    mode = str(raw or "").strip().lower()
    if mode in BANK_MODES:
        return mode
    if mode:
        logger.warning(
            "memory bank %r: mode %r is not one of %s — using %r "
            "(a shared bank is never written directly)",
            name, mode, "/".join(BANK_MODES), DEFAULT_SHARED_MODE,
        )
    return DEFAULT_SHARED_MODE


def _coerce_sync(name: str, raw: Any) -> str:
    sync = str(raw or "").strip().lower() or "session-start"
    if sync in _SYNC_MODES:
        return sync
    logger.warning(
        "memory bank %r: sync %r is not one of %s — using 'manual'",
        name, sync, "/".join(_SYNC_MODES),
    )
    return "manual"


def _is_inside(child: Path, parent: Path) -> bool:
    """Is *child* at or below *parent*? Lexical, on already-resolved paths."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_bank_path(raw: Any, *, name: str, workspace_base: Path) -> Path:
    """Absolute directory for a bank entry.

    A relative ``path:`` resolves against the workspace ``.multiplai/`` root
    (the same anchor ``memory/``, ``diary/`` and ``project-map.yaml`` use),
    not against the process cwd — hooks run from wherever Claude happens to
    be. With no ``path:`` at all a bank lives at ``<workspace>/.multiplai/
    banks/<name>``, which is where :mod:`sync` clones it.
    """
    text = str(raw or "").strip()
    if not text:
        return workspace_base / "banks" / name
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = workspace_base / candidate
    return candidate.resolve()


def _bank_from_entry(
    entry: Any, *, workspace_base: Path, personal: MemoryBank
) -> Optional[MemoryBank]:
    """One validated :class:`MemoryBank`, or ``None`` when the entry is unusable."""
    if not isinstance(entry, dict):
        logger.warning("memory bank entry is not a mapping (%r) — ignored", entry)
        return None
    name = str(entry.get("name") or "").strip().lower()
    if not _NAME_RE.match(name):
        logger.warning(
            "memory bank name %r is not a lowercase [a-z0-9._-] token — ignored", name
        )
        return None
    if name == PERSONAL_BANK:
        # The personal bank is not declared into existence; it exists. An
        # entry for it may only relocate it, and only to a real directory.
        path = _resolve_bank_path(entry.get("path"), name=name, workspace_base=workspace_base)
        return dataclasses.replace(personal, path=path)
    path = _resolve_bank_path(entry.get("path"), name=name, workspace_base=workspace_base)
    if _is_inside(path, personal.path) or _is_inside(personal.path, path):
        logger.warning(
            "memory bank %r at %s overlaps the personal memory dir %s — ignored "
            "(its files would be catalogued twice)",
            name, path, personal.path,
        )
        return None
    return MemoryBank(
        name=name,
        path=path,
        mode=_coerce_mode(name, entry.get("mode")),
        remote=str(entry.get("remote") or "").strip(),
        sync=_coerce_sync(name, entry.get("sync")),
    )


def _read_config(config_path: Optional[Path]) -> list[Any]:
    """The raw ``memory_banks:`` list, or ``[]`` on any failure at all."""
    if config_path is None:
        return []
    try:
        from .config import load_yaml

        data = load_yaml(Path(config_path))
    except Exception:  # pragma: no cover - defensive; yaml may be absent
        logger.warning("Could not read memory banks config at %s", config_path)
        return []
    raw = data.get("memory_banks")
    if raw is None:
        return []
    if not isinstance(raw, list):
        logger.warning(
            "memory_banks in %s is not a list (got %s) — no banks configured",
            config_path, type(raw).__name__,
        )
        return []
    return raw


def load_banks(
    *, memory_dir: Path, config_path: Optional[Path] = None
) -> tuple[MemoryBank, ...]:
    """The ordered banks for this workspace. ``personal`` is always first.

    With no config file, an empty one, or one that cannot be parsed, the
    result is exactly ``(personal_bank(memory_dir),)`` — which is the
    behaviour every consumer had before banks existed. That equivalence is
    the back-compat contract and is asserted by a test that sets no
    configuration at all.
    """
    personal = personal_bank(Path(memory_dir))
    entries = _read_config(config_path)

    resolved: list[MemoryBank] = []
    seen: set[str] = set()
    for entry in entries:
        bank = _bank_from_entry(entry, workspace_base=_workspace_base_for(memory_dir), personal=personal)
        if bank is None:
            continue
        if bank.name in seen:
            logger.warning("memory bank %r declared twice — later entry ignored", bank.name)
            continue
        seen.add(bank.name)
        if bank.is_personal:
            personal = bank
            continue
        resolved.append(bank)

    # A second shared bank pointing at the same directory is the same
    # collision as overlapping the personal dir, one level over.
    deduped: list[MemoryBank] = []
    paths_seen: set[Path] = {personal.path}
    for bank in resolved:
        if bank.path in paths_seen:
            logger.warning(
                "memory bank %r shares a directory with an earlier bank (%s) — ignored",
                bank.name, bank.path,
            )
            continue
        paths_seen.add(bank.path)
        deduped.append(bank)

    return (personal, *deduped)


def _workspace_base_for(memory_dir: Path) -> Path:
    """The ``.multiplai/`` root implied by *memory_dir* (its parent)."""
    return Path(memory_dir).parent
