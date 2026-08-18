"""Tests for memory banks (multiplai_core/banks.py) and their path wiring.

The load-bearing test in this file is
``TestBackCompat::test_no_config_resolves_to_one_personal_bank``: banks are an
additive change to a resolver every installed plugin depends on, and the
promise is that a user who has never heard of them sees no difference at all.
"""

from pathlib import Path

import pytest

from multiplai_core.banks import (
    BANK_MODES,
    DEFAULT_SHARED_MODE,
    PERSONAL_BANK,
    PERSONAL_MODE,
    MemoryBank,
    bank_ref,
    is_bank_name,
    load_banks,
    parse_bank_ref,
    personal_bank,
    split_bank_ref,
)


def _write_banks(tmp_path: Path, body: str) -> Path:
    """Write a ``memory-banks.yaml`` beside a ``memory/`` dir; return its path."""
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "memory-banks.yaml"
    cfg.write_text(body, encoding="utf-8")
    return cfg


# ---------------------------------------------------------------------------
# Back-compat — the contract
# ---------------------------------------------------------------------------


class TestBackCompat:
    """Requirement: absent configuration is exactly the pre-banks world."""

    def test_no_config_resolves_to_one_personal_bank(self, tmp_path):
        """Scenario: no config file at all."""
        memory_dir = tmp_path / "memory"
        banks = load_banks(memory_dir=memory_dir, config_path=None)

        assert len(banks) == 1
        (only,) = banks
        assert only.name == PERSONAL_BANK
        assert only.path == memory_dir
        assert only.mode == PERSONAL_MODE
        assert only.is_personal and not only.is_shared
        assert only.accepts_direct_writes

    def test_missing_config_file_resolves_to_one_personal_bank(self, tmp_path):
        """Scenario: a path is offered but nothing is there."""
        banks = load_banks(
            memory_dir=tmp_path / "memory", config_path=tmp_path / "nope.yaml"
        )
        assert [b.name for b in banks] == [PERSONAL_BANK]

    def test_paths_memory_banks_matches_memory_dir_with_no_config(
        self, tmp_workspace
    ):
        """Scenario: the resolver's own accessor, with nothing configured."""
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        banks = p.memory_banks()
        assert len(banks) == 1
        assert banks[0].path == p.memory_dir()
        assert (
            p.memory_banks_file()
            == tmp_workspace / ".multiplai" / "memory-banks.yaml"
        )

    def test_memory_banks_file_honours_option_override(
        self, monkeypatch, tmp_workspace
    ):
        monkeypatch.setenv(
            "CLAUDE_PLUGIN_OPTION_MEMORY_BANKS_FILE",
            str(tmp_workspace / "custom.yaml"),
        )
        from multiplai_core.paths import Paths

        assert Paths.resolve().memory_banks_file() == (tmp_workspace / "custom.yaml")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestLoadBanks:
    def test_shared_bank_is_added_after_personal(self, tmp_path):
        cfg = _write_banks(
            tmp_path,
            """
memory_banks:
  - name: dolcebot-team
    path: banks/dolcebot-team
    remote: git@github.com:example/memory-bank.git
    mode: propose
""",
        )
        banks = load_banks(memory_dir=tmp_path / "memory", config_path=cfg)

        assert [b.name for b in banks] == [PERSONAL_BANK, "dolcebot-team"]
        team = banks[1]
        assert team.is_shared
        assert not team.accepts_direct_writes
        assert team.accepts_contributions
        assert team.path == (tmp_path / "banks" / "dolcebot-team").resolve()
        assert team.syncs_at_session_start

    def test_default_path_is_under_workspace_banks_dir(self, tmp_path):
        cfg = _write_banks(tmp_path, "memory_banks:\n  - name: team\n")
        banks = load_banks(memory_dir=tmp_path / "memory", config_path=cfg)
        assert banks[1].path == tmp_path / "banks" / "team"

    def test_absolute_path_is_used_verbatim(self, tmp_path):
        elsewhere = tmp_path / "elsewhere"
        cfg = _write_banks(tmp_path, f"memory_banks:\n  - name: team\n    path: {elsewhere}\n")
        banks = load_banks(memory_dir=tmp_path / "memory", config_path=cfg)
        assert banks[1].path == elsewhere.resolve()

    def test_mode_defaults_to_propose(self, tmp_path):
        cfg = _write_banks(tmp_path, "memory_banks:\n  - name: team\n")
        assert load_banks(memory_dir=tmp_path / "memory", config_path=cfg)[1].mode == (
            DEFAULT_SHARED_MODE
        )

    def test_ro_mode_is_kept(self, tmp_path):
        cfg = _write_banks(tmp_path, "memory_banks:\n  - name: team\n    mode: ro\n")
        team = load_banks(memory_dir=tmp_path / "memory", config_path=cfg)[1]
        assert team.mode == "ro"
        assert not team.accepts_contributions
        assert not team.accepts_direct_writes

    @pytest.mark.parametrize("mode", ["rw", "write", "RW", "banana"])
    def test_write_modes_are_coerced_to_propose(self, tmp_path, mode):
        """A shared bank is never directly writable, whatever the config says."""
        cfg = _write_banks(tmp_path, f"memory_banks:\n  - name: team\n    mode: {mode}\n")
        team = load_banks(memory_dir=tmp_path / "memory", config_path=cfg)[1]
        assert team.mode in BANK_MODES
        assert team.mode == DEFAULT_SHARED_MODE
        assert not team.accepts_direct_writes

    def test_personal_entry_only_relocates_and_stays_first(self, tmp_path):
        (tmp_path / "other-memory").mkdir()
        cfg = _write_banks(
            tmp_path,
            """
memory_banks:
  - name: team
  - name: personal
    path: other-memory
    mode: ro
""",
        )
        banks = load_banks(memory_dir=tmp_path / "memory", config_path=cfg)
        assert [b.name for b in banks] == [PERSONAL_BANK, "team"]
        assert banks[0].path == (tmp_path / "other-memory").resolve()
        # The mode of the personal bank is not configurable — it is the one
        # corpus the local user owns outright.
        assert banks[0].mode == PERSONAL_MODE

    def test_bank_nested_inside_personal_memory_is_refused(self, tmp_path):
        cfg = _write_banks(tmp_path, "memory_banks:\n  - name: team\n    path: memory/team\n")
        banks = load_banks(memory_dir=tmp_path / "memory", config_path=cfg)
        assert [b.name for b in banks] == [PERSONAL_BANK]

    def test_bank_containing_personal_memory_is_refused(self, tmp_path):
        cfg = _write_banks(tmp_path, "memory_banks:\n  - name: team\n    path: .\n")
        banks = load_banks(memory_dir=tmp_path / "memory", config_path=cfg)
        assert [b.name for b in banks] == [PERSONAL_BANK]

    def test_two_banks_at_the_same_path_keeps_the_first(self, tmp_path):
        cfg = _write_banks(
            tmp_path,
            """
memory_banks:
  - name: team
    path: shared
  - name: other
    path: shared
""",
        )
        banks = load_banks(memory_dir=tmp_path / "memory", config_path=cfg)
        assert [b.name for b in banks] == [PERSONAL_BANK, "team"]

    def test_duplicate_names_keep_the_first(self, tmp_path):
        cfg = _write_banks(
            tmp_path,
            """
memory_banks:
  - name: team
    path: a
  - name: team
    path: b
""",
        )
        banks = load_banks(memory_dir=tmp_path / "memory", config_path=cfg)
        assert [b.name for b in banks] == [PERSONAL_BANK, "team"]
        assert banks[1].path == (tmp_path / "a").resolve()

    @pytest.mark.parametrize(
        "name", ["../evil", "Team Bank", "", "-leading", "a" * 65, "te/am"]
    )
    def test_unusable_names_are_dropped(self, tmp_path, name):
        cfg = _write_banks(tmp_path, f'memory_banks:\n  - name: "{name}"\n')
        banks = load_banks(memory_dir=tmp_path / "memory", config_path=cfg)
        assert [b.name for b in banks] == [PERSONAL_BANK]

    @pytest.mark.parametrize(
        "body",
        [
            "memory_banks: not-a-list\n",
            "memory_banks:\n  - just-a-string\n",
            "memory_banks:\n",
            ": : :\n",
            "",
        ],
    )
    def test_malformed_config_falls_back_to_personal_only(self, tmp_path, body):
        cfg = _write_banks(tmp_path, body)
        banks = load_banks(memory_dir=tmp_path / "memory", config_path=cfg)
        assert [b.name for b in banks] == [PERSONAL_BANK]

    def test_one_bad_entry_does_not_drop_the_good_ones(self, tmp_path):
        cfg = _write_banks(
            tmp_path,
            """
memory_banks:
  - name: "../escape"
  - name: team
""",
        )
        banks = load_banks(memory_dir=tmp_path / "memory", config_path=cfg)
        assert [b.name for b in banks] == [PERSONAL_BANK, "team"]

    def test_bank_without_remote_does_not_sync(self, tmp_path):
        cfg = _write_banks(tmp_path, "memory_banks:\n  - name: team\n")
        assert not load_banks(memory_dir=tmp_path / "memory", config_path=cfg)[1].syncs_at_session_start

    def test_manual_sync_is_honoured(self, tmp_path):
        cfg = _write_banks(
            tmp_path,
            "memory_banks:\n  - name: team\n    remote: git@x:y.git\n    sync: manual\n",
        )
        assert not load_banks(memory_dir=tmp_path / "memory", config_path=cfg)[1].syncs_at_session_start

    def test_unknown_sync_value_becomes_manual(self, tmp_path):
        cfg = _write_banks(
            tmp_path,
            "memory_banks:\n  - name: team\n    remote: git@x:y.git\n    sync: hourly\n",
        )
        bank = load_banks(memory_dir=tmp_path / "memory", config_path=cfg)[1]
        assert bank.sync == "manual"
        assert not bank.syncs_at_session_start


# ---------------------------------------------------------------------------
# The personal entry — the one that is a relocation, not a declaration
# ---------------------------------------------------------------------------


class TestPersonalRelocation:
    """Requirement: ``name: personal`` may move the corpus, not redefine it.

    Every trust decision in every consumer keys off ``is_shared``, which is
    ``False`` for this bank *by name alone*. So an unvalidated ``path:`` here
    is not a tidiness bug: it points the trusted corpus somewhere, and that
    somewhere is then injected unfenced and written to directly.
    """

    def test_relocation_to_a_nonexistent_directory_is_refused(self, tmp_path):
        cfg = _write_banks(
            tmp_path,
            "memory_banks:\n"
            "  - name: personal\n"
            f"    path: {tmp_path / 'not-created-yet'}\n"
            "    mode: rw\n"
            "    remote: git@github.com:other/people.git\n",
        )
        banks = load_banks(memory_dir=tmp_path / "memory", config_path=cfg)
        assert banks[0].path == (tmp_path / "memory").resolve()
        assert banks[0].remote == ""

    def test_relocation_with_no_path_relocates_nothing(self, tmp_path):
        cfg = _write_banks(tmp_path, "memory_banks:\n  - name: personal\n")
        banks = load_banks(memory_dir=tmp_path / "memory", config_path=cfg)
        assert [b.name for b in banks] == [PERSONAL_BANK]
        assert banks[0].path == (tmp_path / "memory").resolve()

    @pytest.mark.parametrize("spelling", ["personal", "PERSONAL", "  personal  "])
    def test_every_spelling_reaches_the_same_validation(self, tmp_path, spelling):
        cfg = _write_banks(
            tmp_path,
            f'memory_banks:\n  - name: "{spelling}"\n    path: {tmp_path / "gone"}\n',
        )
        banks = load_banks(memory_dir=tmp_path / "memory", config_path=cfg)
        assert banks[0].path == (tmp_path / "memory").resolve()

    def test_relocation_onto_a_declared_shared_bank_loses_to_the_bank(self, tmp_path):
        """A directory claimed by both banks belongs to the untrusted one.

        Refusing the *bank* instead would hand a shared repo to the personal
        bank — read unfenced, written directly — which is strictly worse than
        declining the relocation.
        """
        (tmp_path / "shared" / "mine").mkdir(parents=True)
        cfg = _write_banks(
            tmp_path,
            "memory_banks:\n"
            f"  - name: team\n    path: {tmp_path / 'shared'}\n"
            f"  - name: personal\n    path: {tmp_path / 'shared' / 'mine'}\n",
        )
        banks = load_banks(memory_dir=tmp_path / "memory", config_path=cfg)
        assert banks[0].path == (tmp_path / "memory").resolve()
        assert [b.name for b in banks] == [PERSONAL_BANK, "team"]

    def test_the_personal_bank_can_never_be_removed_by_configuration(self, tmp_path):
        cfg = _write_banks(
            tmp_path, f"memory_banks:\n  - name: team\n    path: {tmp_path / 'memory'}\n"
        )
        banks = load_banks(memory_dir=tmp_path / "memory", config_path=cfg)
        assert banks[0].is_personal and banks[0].accepts_direct_writes


class TestOverlapIsOrderIndependent:
    """Requirement: the same layout gets the same answer in either order.

    The overlap rule used to be enforced during the per-entry parse, against
    whatever ``personal`` was at that moment. An entry declared *before* a
    personal relocation was therefore checked against the old path and never
    re-checked — so this configuration was accepted written one way round and
    refused written the other. A containment check may not have that property.
    """

    LAYOUT = (
        ("team", "shared"),
        ("personal", "shared/mine"),
    )

    @pytest.mark.parametrize("reverse", [False, True])
    def test_a_personal_bank_nested_in_a_shared_bank_is_refused_either_way(
        self, tmp_path, reverse
    ):
        (tmp_path / "shared" / "mine").mkdir(parents=True)
        entries = list(reversed(self.LAYOUT)) if reverse else list(self.LAYOUT)
        body = "memory_banks:\n" + "".join(
            f"  - name: {n}\n    path: {tmp_path / p}\n" for n, p in entries
        )
        banks = load_banks(memory_dir=tmp_path / "memory", config_path=_write_banks(tmp_path, body))
        personal = banks[0]
        shared = [b for b in banks if b.is_shared]
        # Whatever survives, the invariant is the same: no shared bank may
        # contain the personal corpus.
        for bank in shared:
            assert not str(personal.path).startswith(str(bank.path) + "/")
            assert personal.path != bank.path

    def test_a_bank_nested_inside_another_bank_is_refused(self, tmp_path):
        (tmp_path / "outer" / "inner").mkdir(parents=True)
        cfg = _write_banks(
            tmp_path,
            "memory_banks:\n"
            f"  - name: outer\n    path: {tmp_path / 'outer'}\n"
            f"  - name: inner\n    path: {tmp_path / 'outer' / 'inner'}\n",
        )
        banks = load_banks(memory_dir=tmp_path / "memory", config_path=cfg)
        assert [b.name for b in banks] == [PERSONAL_BANK, "outer"]

    def test_a_symlinked_memory_dir_does_not_hide_a_nested_bank(self, tmp_path):
        """``_is_inside`` is lexical, so both sides have to be resolved.

        This workspace's own ``.multiplai/`` is a symlink, which is precisely
        the configuration where an unresolved personal path lets a bank sit
        inside the personal corpus undetected.
        """
        real = tmp_path / "runtime"
        (real / "memory" / "team").mkdir(parents=True)
        link = tmp_path / ".multiplai"
        link.symlink_to(real)
        cfg = tmp_path / "memory-banks.yaml"
        cfg.write_text(
            f"memory_banks:\n  - name: team\n    path: {real / 'memory' / 'team'}\n",
            encoding="utf-8",
        )
        banks = load_banks(memory_dir=link / "memory", config_path=cfg)
        assert [b.name for b in banks] == [PERSONAL_BANK]

    def test_the_same_directory_reached_two_ways_is_deduped(self, tmp_path):
        """A default path and an explicit one must be comparable.

        One resolved and one unresolved return value made the ``paths_seen``
        dedup miss two banks that were the same directory — the duplicate
        catalogue the dedup exists to prevent.
        """
        real = tmp_path / "runtime"
        (real / "memory").mkdir(parents=True)
        (real / "banks" / "team").mkdir(parents=True)
        link = tmp_path / ".multiplai"
        link.symlink_to(real)
        cfg = tmp_path / "memory-banks.yaml"
        cfg.write_text(
            "memory_banks:\n"
            "  - name: team\n"  # default path -> <base>/banks/team
            f"  - name: other\n    path: {link / 'banks' / 'team'}\n",
            encoding="utf-8",
        )
        banks = load_banks(memory_dir=link / "memory", config_path=cfg)
        assert [b.name for b in banks] == [PERSONAL_BANK, "team"]


class TestConstructionIsGuarded:
    """Requirement: a bank cannot exist whose properties contradict its mode.

    ``load_banks`` is the trusted factory and cannot mint any of these. These
    guards are for the other constructors, which any consumer can reach.
    """

    def test_a_shared_bank_cannot_be_built_with_a_write_mode(self, tmp_path):
        with pytest.raises(ValueError):
            MemoryBank(name="team", path=tmp_path, mode=PERSONAL_MODE)

    def test_a_shared_bank_cannot_be_renamed_into_the_personal_one(self, tmp_path):
        import dataclasses

        shared = MemoryBank(name="team", path=tmp_path, mode="ro")
        with pytest.raises(ValueError):
            dataclasses.replace(shared, name=PERSONAL_BANK)

    def test_the_personal_bank_cannot_be_built_with_a_shared_mode(self, tmp_path):
        with pytest.raises(ValueError):
            MemoryBank(name=PERSONAL_BANK, path=tmp_path, mode="ro")

    @pytest.mark.parametrize(
        "filename", ["/etc/passwd", "../x.md", "sub/dir.md", "", "  ", ".", "..", None]
    )
    def test_file_refuses_anything_that_is_not_a_bare_filename(self, tmp_path, filename):
        bank = personal_bank(tmp_path / "memory")
        with pytest.raises(ValueError):
            bank.file(filename)

    def test_file_joins_a_bare_filename(self, tmp_path):
        bank = personal_bank(tmp_path / "memory")
        assert bank.file("dev.md") == (tmp_path / "memory").resolve() / "dev.md"

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("team", True), ("a", True), ("a-b_c.d", True), ("0", True),
            ("Team", False), ("", False), ("-x", False), ("../x", False),
            ("te/am", False), ("a" * 65, False), (None, False), (5, False),
        ],
    )
    def test_is_bank_name_is_exported_and_total(self, name, expected):
        """The consumer's write floor must not re-declare this regex."""
        assert is_bank_name(name) is expected


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


class TestRefs:
    def test_personal_refs_are_unprefixed(self):
        assert bank_ref(PERSONAL_BANK, "dev.md") == "dev.md"
        assert personal_bank(Path("/m")).ref("dev.md") == "dev.md"

    def test_shared_refs_carry_the_bank_name(self):
        bank = MemoryBank(name="team", path=Path("/b/team"), mode="propose")
        assert bank.ref("dev.md") == "team/dev.md"
        assert bank.file("dev.md") == Path("/b/team/dev.md")

    @pytest.mark.parametrize(
        "ref,expected",
        [
            ("dev.md", (PERSONAL_BANK, "dev.md")),
            ("team/dev.md", ("team", "dev.md")),
            ("  team/dev.md  ", ("team", "dev.md")),
            ("a/b/c.md", ("a", "b/c.md")),
            # A separator with an empty bank segment names NO bank. It used to
            # answer `personal`, which made "/dev.md" a permitted personal
            # write of dev.md — a spelling that slips past the bare-basename
            # check every consumer applies to "dev.md" itself.
            ("/dev.md", ("", "dev.md")),
            ("//dev.md", ("", "/dev.md")),
            ("/", ("", "")),
            # The bank segment is lower-cased to match the names load_banks
            # accepts; filenames are real paths and stay case-sensitive.
            ("Team/dev.md", ("team", "dev.md")),
            ("TEAM/Dev.MD", ("team", "Dev.MD")),
            # Total on non-strings.
            (None, (PERSONAL_BANK, "")),
        ],
    )
    def test_split_bank_ref(self, ref, expected):
        assert split_bank_ref(ref) == expected

    def test_an_empty_bank_segment_resolves_to_no_bank(self, tmp_path):
        """Requirement: "/dev.md" is a refusal, not a personal write."""
        banks = (personal_bank(tmp_path / "memory"),)
        for ref in ("/dev.md", "//dev.md", "///etc/passwd", "/"):
            bank, _ = parse_bank_ref(ref, banks)
            assert bank is None, ref

    def test_a_ref_reaches_a_bank_whatever_its_case(self, tmp_path):
        """A hand- or model-written ``Team/dev.md`` must not go silently nowhere."""
        banks = (
            personal_bank(tmp_path / "memory"),
            MemoryBank(name="team", path=tmp_path / "team", mode="propose"),
        )
        bank, filename = parse_bank_ref("Team/dev.md", banks)
        assert bank is not None and bank.name == "team" and filename == "dev.md"

    def test_parse_bank_ref_resolves_against_the_bank_list(self, tmp_path):
        banks = (
            personal_bank(tmp_path / "memory"),
            MemoryBank(name="team", path=tmp_path / "team", mode="propose"),
        )
        bank, filename = parse_bank_ref("team/dev.md", banks)
        assert bank is not None and bank.name == "team" and filename == "dev.md"

    def test_unknown_bank_never_falls_back_to_personal(self, tmp_path):
        """A stale ref must not become a write to a personal file of that name."""
        banks = (personal_bank(tmp_path / "memory"),)
        bank, filename = parse_bank_ref("gone/dev.md", banks)
        assert bank is None
        assert filename == "dev.md"
