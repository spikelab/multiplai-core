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
        self, monkeypatch, tmp_path, reset_paths_cache
    ):
        """Scenario: the resolver's own accessor, with nothing configured."""
        monkeypatch.setenv("WORKSPACE", str(tmp_path))
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        banks = p.memory_banks()
        assert len(banks) == 1
        assert banks[0].path == p.memory_dir()
        assert p.memory_banks_file() == tmp_path / ".multiplai" / "memory-banks.yaml"

    def test_memory_banks_file_honours_option_override(
        self, monkeypatch, tmp_path, reset_paths_cache
    ):
        monkeypatch.setenv("WORKSPACE", str(tmp_path))
        monkeypatch.setenv(
            "CLAUDE_PLUGIN_OPTION_MEMORY_BANKS_FILE", str(tmp_path / "custom.yaml")
        )
        from multiplai_core.paths import Paths

        assert Paths.resolve().memory_banks_file() == (tmp_path / "custom.yaml")


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
            ("/dev.md", (PERSONAL_BANK, "dev.md")),
            ("a/b/c.md", ("a", "b/c.md")),
        ],
    )
    def test_split_bank_ref(self, ref, expected):
        assert split_bank_ref(ref) == expected

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


# ---------------------------------------------------------------------------
# Marker-based workspace discovery
# ---------------------------------------------------------------------------


class TestWorkspaceDiscovery:
    """Requirement: a workspace is findable without the launcher exporting it."""

    def test_marker_directory_is_discovered_from_project_dir(
        self, monkeypatch, tmp_path, reset_paths_cache
    ):
        (tmp_path / ".multiplai").mkdir()
        nested = tmp_path / "PROJECTS" / "thing" / "src"
        nested.mkdir(parents=True)
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(nested))
        from multiplai_core.paths import Paths

        assert Paths.resolve().memory_dir() == tmp_path / ".multiplai" / "memory"

    def test_explicit_workspace_still_wins(
        self, monkeypatch, tmp_path, reset_paths_cache
    ):
        (tmp_path / "discovered" / ".multiplai").mkdir(parents=True)
        explicit = tmp_path / "explicit"
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path / "discovered"))
        monkeypatch.setenv("WORKSPACE", str(explicit))
        from multiplai_core.paths import Paths

        assert Paths.resolve().memory_dir() == explicit / ".multiplai" / "memory"

    def test_no_project_dir_means_no_discovery(
        self, monkeypatch, tmp_path, reset_paths_cache
    ):
        """The cwd is never a start point — resolution must not depend on it."""
        (tmp_path / ".multiplai").mkdir()
        monkeypatch.chdir(tmp_path)
        from multiplai_core.paths import Paths

        assert Paths.resolve().memory_dir() == Path.home() / ".multiplai" / "memory"

    def test_no_marker_anywhere_falls_back_to_standalone(
        self, monkeypatch, tmp_path, reset_paths_cache
    ):
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(nested))
        from multiplai_core.paths import Paths

        assert Paths.resolve().memory_dir() == Path.home() / ".multiplai" / "memory"
