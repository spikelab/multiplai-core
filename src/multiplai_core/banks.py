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
* **No two banks may overlap** — not a bank inside the personal corpus, not
  the personal corpus inside a bank, not two banks sharing or nesting in one
  directory. Otherwise a file is discovered twice, once per bank, and the
  no-duplication rule the whole design rests on is broken by layout. The
  check runs **once, over the fully resolved list, after parsing**, so its
  answer cannot depend on the order the entries were declared in; and where a
  directory is claimed by both a trusted and an untrusted bank, the untrusted
  one wins, because the cost of getting that backwards is unfenced injection.

The one entry that is not a declaration is ``name: personal``: that bank
exists already, so an entry for it may only **relocate** it, and only onto a
directory that already exists and no shared bank covers.

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
    "is_bank_name",
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
_NAME_RE = re.compile(r"\A[a-z0-9][a-z0-9._-]{0,63}\Z")

_SYNC_MODES: tuple[str, ...] = ("session-start", "manual")


def is_bank_name(name: Any) -> bool:
    """Is *name* a usable bank name — a path segment that cannot traverse?

    Exported because it is a **shared predicate, not an implementation
    detail**: a consumer's write floor has to answer "is this ref's first
    segment a bank name?" with exactly the same answer this module gives, and
    re-declaring the regex there makes the two able to drift apart silently.
    Whatever enforces bank naming, enforces it from here.

    Deliberately total: any non-string, ``None``, or empty value is ``False``.
    """
    if not isinstance(name, str):
        return False
    return bool(_NAME_RE.match(name))


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

    def __post_init__(self) -> None:
        """Refuse a bank that contradicts the module's own invariants.

        :func:`load_banks` is the trusted factory and cannot produce any of
        these — ``_coerce_mode`` only ever returns a member of
        :data:`BANK_MODES`. This guard is here for the *other* constructors:
        ``MemoryBank(name="team", mode="rw")`` and
        ``dataclasses.replace(shared, name="personal")`` both used to mint a
        bank whose :attr:`accepts_direct_writes` contradicted its ``mode``,
        and both are reachable from any consumer that builds one by hand.

        A property derived from ``name`` is only as trustworthy as ``name``,
        so the pairing is checked where it is established.
        """
        if self.name == PERSONAL_BANK:
            if self.mode != PERSONAL_MODE:
                raise ValueError(
                    f"the {PERSONAL_BANK!r} bank's mode is {PERSONAL_MODE!r}, "
                    f"not {self.mode!r}"
                )
        elif self.mode not in BANK_MODES:
            raise ValueError(
                f"memory bank {self.name!r}: mode {self.mode!r} is not one of "
                f"{'/'.join(BANK_MODES)} (a shared bank is never written directly)"
            )

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
        """``path / filename`` — the one place a bank filename is joined.

        *filename* must be a bare basename. Enforced here rather than trusted
        from the caller, because ``path / filename`` gives an absolute
        *filename* total override (``bank.file("/etc/passwd")`` used to return
        ``/etc/passwd``) and does not normalise ``..``. Billing this as "the
        one place a filename is joined" while validating in every caller is
        the shape that produces an unguarded second caller.
        """
        text = str(filename or "").strip()
        if not text or text != Path(text).name or text in (".", ".."):
            raise ValueError(
                f"memory bank {self.name!r}: {filename!r} is not a bare filename"
            )
        return self.path / text


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

    Two rules that look like details and are not:

    * **The personal bank is the answer only when there is no separator at
      all.** A ref *with* a separator whose bank segment is empty
      (``"/dev.md"``, ``"//dev.md"``) names no bank, so the bank comes back
      ``""`` and :func:`parse_bank_ref` refuses it. It used to come back
      ``personal``, which made ``"/dev.md"`` a *permitted* personal write of
      ``dev.md`` — a spelling that bypassed the bare-basename check every
      consumer applies to ``"dev.md"`` itself.
    * **The bank segment is lower-cased**, because :func:`load_banks` lowercases
      the names it accepts from config. Without this, a bank configured as
      ``team`` is unreachable through the perfectly reasonable ref
      ``Team/dev.md``, and the failure is a silent resolution to nothing.
      Filenames stay case-sensitive — those are real paths.
    """
    text = str(ref or "").strip()
    if "/" not in text:
        return PERSONAL_BANK, text
    name, _, filename = text.partition("/")
    return name.strip().lower(), filename.strip()


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
    if not name:
        return None, filename
    for bank in banks:
        if bank.name == name:
            return bank, filename
    return None, filename


def personal_bank(memory_dir: Path) -> MemoryBank:
    """The one bank that always exists, at today's path.

    The path is ``.resolve()``d, because :func:`_is_inside` is a *lexical*
    comparison and a lexical comparison between one resolved path and one
    unresolved one is not a containment test. ``.multiplai/`` is a symlink in
    at least one real workspace, which is exactly the case where an unresolved
    personal path lets a bank sit inside the personal corpus undetected.
    """
    return MemoryBank(
        name=PERSONAL_BANK,
        path=_safe_resolve(Path(memory_dir)),
        mode=PERSONAL_MODE,
        remote="",
        sync="manual",
    )


def _safe_resolve(path: Path) -> Path:
    """``path.resolve()``, falling back to the input on a filesystem error.

    ``resolve()`` does not raise for a path that does not exist, but it can
    raise ``OSError`` on a symlink loop or an unreadable parent. Resolution
    failing must not take a session down.
    """
    try:
        return path.expanduser().resolve()
    except (OSError, ValueError, RuntimeError):
        return path


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
    return child.is_relative_to(parent)


def _overlaps(a: Path, b: Path) -> bool:
    """Do *a* and *b* name the same directory, or does either contain the other?

    The mutual-containment test behind every refusal in this module — the
    underlying rule is **a memory file belongs to exactly one bank**.
    """
    return _is_inside(a, b) or _is_inside(b, a)


def _resolve_bank_path(raw: Any, *, name: str, workspace_base: Path) -> Path:
    """Absolute, resolved directory for a bank entry.

    A relative ``path:`` resolves against *workspace_base* — ``memory_dir``'s
    parent — not against the process cwd, because hooks run from wherever
    Claude happens to be. Note that is the same anchor ``diary/`` and
    ``learnings/`` use *only when ``memory_dir`` has not been overridden*; when
    it has, this anchor follows ``memory_dir``. With no ``path:`` at all a bank
    lives at ``<workspace_base>/banks/<name>``, which is where the sync path
    clones it.

    **Every** return value is resolved, including the default. Returning one
    resolved and one unresolved path made the ``paths_seen`` dedup in
    :func:`load_banks` miss two banks that were the same directory reached two
    ways — the precise duplicate-catalogue failure that dedup exists to stop.
    """
    text = str(raw or "").strip()
    if not text:
        return _safe_resolve(workspace_base / "banks" / name)
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = workspace_base / candidate
    return _safe_resolve(candidate)


def _personal_relocation(entry: dict, *, workspace_base: Path) -> Optional[Path]:
    """Where an entry named ``personal`` moves the personal corpus to, or ``None``.

    The personal bank is not declared into existence; it exists. An entry for
    it may only *relocate* it, and — as this module has always claimed and did
    not enforce — **only to a directory that already exists.**

    That check is the load-bearing one, not a tidiness check. Every trust
    decision in every consumer keys off :attr:`MemoryBank.is_shared`, which is
    ``False`` for this bank by name alone. So an unvalidated path here means a
    single config line can point the *trusted* corpus at somebody else's git
    repo: its content is then injected **unfenced** and the memory-apply path
    writes directly into it. Requiring the directory to exist does not make
    that safe by itself, but it removes the case where the path was never a
    memory directory at all, and it makes the relocation something the local
    filesystem has to agree with.
    """
    raw = str(entry.get("path") or "").strip()
    if not raw:
        logger.warning(
            "memory bank %r declares no path: — nothing to relocate, entry ignored",
            PERSONAL_BANK,
        )
        return None
    path = _resolve_bank_path(raw, name=PERSONAL_BANK, workspace_base=workspace_base)
    try:
        exists = path.is_dir()
    except OSError:
        exists = False
    if not exists:
        logger.warning(
            "memory bank %r cannot be relocated to %s — it is not an existing "
            "directory; keeping the configured memory dir",
            PERSONAL_BANK, path,
        )
        return None
    return path


def _bank_from_entry(
    entry: dict, *, name: str, workspace_base: Path
) -> MemoryBank:
    """One validated *shared* :class:`MemoryBank`.

    Overlap is deliberately **not** checked here — see :func:`load_banks`. A
    per-entry overlap check has to compare against whatever ``personal`` was
    at the moment the entry was parsed, which made the outcome depend on
    declaration order.
    """
    path = _resolve_bank_path(entry.get("path"), name=name, workspace_base=workspace_base)
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

    Parsing runs in **two phases**, and the split is the whole point:

    1. Every entry is parsed and resolved, and a ``personal`` entry's
       relocation is applied. Nothing is compared to anything yet.
    2. One overlap pass over the *fully resolved* list decides what survives.

    A single pass cannot do this correctly. The overlap rule used to be
    enforced inside the per-entry parse, against whatever ``personal`` was at
    that moment — so a shared bank declared *before* a personal relocation
    was checked against the old personal path and never re-checked, and the
    personal entry itself was never checked at all. The same two-entry
    configuration was accepted or refused depending on the order the entries
    were written in, which is not a property a containment check may have.
    """
    configured = personal_bank(Path(memory_dir))
    personal = configured
    # The anchor for relative bank paths is the workspace root implied by the
    # *configured* memory dir, taken before any relocation: relocating the
    # personal corpus must not silently move where every other bank resolves.
    workspace_base = _workspace_base_for(configured.path)
    entries = _read_config(config_path)

    candidates: list[MemoryBank] = []
    seen: set[str] = set()
    for entry in entries:
        # Phase 1 is fail-open to the narrow state: one unusable entry costs
        # itself and nothing else. Nothing below is expected to raise —
        # is_dir() can, and MemoryBank refuses a self-contradicting bank — and
        # a config typo must never be able to break a session.
        try:
            if not isinstance(entry, dict):
                logger.warning("memory bank entry is not a mapping (%r) — ignored", entry)
                continue
            name = str(entry.get("name") or "").strip().lower()
            if not is_bank_name(name):
                logger.warning(
                    "memory bank name %r is not a lowercase [a-z0-9._-] token — ignored",
                    name,
                )
                continue
            if name in seen:
                logger.warning("memory bank %r declared twice — later entry ignored", name)
                continue
            seen.add(name)
            if name == PERSONAL_BANK:
                relocated = _personal_relocation(entry, workspace_base=workspace_base)
                if relocated is not None:
                    personal = dataclasses.replace(personal, path=relocated)
                continue
            candidates.append(
                _bank_from_entry(entry, name=name, workspace_base=workspace_base)
            )
        except Exception:  # pragma: no cover - defensive
            logger.warning("memory bank entry %r could not be read — ignored", entry)

    # A relocation that lands on or inside a declared shared bank loses to the
    # shared bank. Both orderings are now refused (that is H2), but "refuse"
    # has to mean *this* direction: if the two were resolved the other way the
    # shared bank would be dropped and its directory would become the personal
    # corpus — read unfenced, written directly. When a path is claimed by both
    # a trusted and an untrusted bank, untrusted has to win.
    if personal.path != configured.path:
        clash = next(
            (c for c in candidates if _overlaps(personal.path, c.path)),
            None,
        )
        if clash is not None:
            logger.warning(
                "memory bank %r cannot be relocated to %s — memory bank %r at %s "
                "already covers it; keeping the configured memory dir %s",
                PERSONAL_BANK, personal.path, clash.name, clash.path, configured.path,
            )
            personal = configured

    return (personal, *_without_overlaps(candidates, personal=personal))


def _without_overlaps(
    candidates: list[MemoryBank], *, personal: MemoryBank
) -> list[MemoryBank]:
    """The candidates that may coexist with *personal* and with each other.

    One pass, over already-resolved paths, with the final personal path — so
    the answer does not depend on declaration order. Three refusals, all the
    same underlying rule (**a memory file belongs to exactly one bank**):
    nesting either way with the personal corpus, sharing a directory with a
    bank already kept, and nesting either way with a bank already kept.

    The personal bank is never dropped. It is the corpus; a configuration
    cannot remove it, only relocate it.
    """
    kept: list[MemoryBank] = []
    for bank in candidates:
        if _overlaps(bank.path, personal.path):
            logger.warning(
                "memory bank %r at %s overlaps the personal memory dir %s — ignored "
                "(its files would be catalogued twice)",
                bank.name, bank.path, personal.path,
            )
            continue
        clash = next(
            (k for k in kept if _overlaps(bank.path, k.path)),
            None,
        )
        if clash is not None:
            logger.warning(
                "memory bank %r at %s overlaps memory bank %r at %s — ignored",
                bank.name, bank.path, clash.name, clash.path,
            )
            continue
        kept.append(bank)
    return kept


def _workspace_base_for(memory_dir: Path) -> Path:
    """The ``.multiplai/`` root implied by *memory_dir* (its parent).

    Callers pass the **resolved** memory dir, so the anchor a relative
    ``path:`` joins onto is resolved too and compares equal to the same
    directory reached any other way.
    """
    return Path(memory_dir).parent
