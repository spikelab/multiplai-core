"""Tests for multiplai_core.specs — the specs/ directory format.

The classes marked "Ported" are buildme's tests for the same functions
(`plugins/multiplai-dev/skills/buildme/scripts/tests/test_change_manager.py`
and `test_spec_generator.py::TestExtractCapabilities` in multiplai-cc-mktplace),
with `ChangeManager` replaced by the module-level functions and
`cm.create_change` by a plain directory.
"""

import shutil
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from multiplai_core.specs import (
    RequirementBlock,
    ScenarioBlock,
    apply_delta,
    archive_change,
    archived_change_dirs,
    capabilities,
    change_root,
    clear_spec_issues,
    component_dirs,
    expected_capabilities,
    is_ready,
    mark_draft,
    mark_ready,
    merge_delta_requirements,
    normalize_change_name,
    parse_requirements,
    read_change_meta,
    requirement_texts,
    set_spec_issues,
    spec_files,
    spec_revision,
    split_capabilities,
    valid_requirement,
    valid_requirement_text,
    valid_specs_text,
    write_change_meta,
)

SPEC_EXTRACT_SAMPLE = (
    Path(__file__).parent / "fixtures" / "spec_extract_registry_sample.md"
)


@pytest.fixture
def specs_dir(tmp_path):
    """Create a minimal specs/ directory structure."""
    d = tmp_path / "specs"
    d.mkdir()
    (d / "changes").mkdir()
    (d / "registry").mkdir()
    (d / "archive").mkdir()
    return d


def _change(specs_dir: Path, name: str) -> Path:
    d = specs_dir / "changes" / normalize_change_name(name)
    d.mkdir(parents=True)
    (d / ".change.yaml").write_text("schema: spec-driven\n")
    return d


# --- Ported from test_change_manager.py ---


class TestNormalizeChangeName:
    def test_empty_input_stays_empty(self):
        assert normalize_change_name("") == ""

    def test_traversal_is_stripped(self):
        assert normalize_change_name("../../etc") == "etc"

    def test_refuses_name_that_normalizes_to_nothing(self):
        """'!!!' → '' would collapse change_dir to specs/changes and the
        branch to 'buildme/', dying late with a raw git error."""
        with pytest.raises(ValueError) as exc:
            normalize_change_name("!!!")
        assert "no usable characters" in str(exc.value)

    def test_normalizes_case_spaces_and_underscores(self):
        # TestCreateChange.test_normalizes_name / test_underscores_to_hyphens
        assert normalize_change_name("My Feature") == "my-feature"
        assert normalize_change_name("my_feature") == "my-feature"


class TestArchivedChangeDirs:
    def test_anchored_match_ignores_suffix_collisions(self, specs_dir):
        """`foo` must not match `2026-07-26-bar-foo` — a loose `*-foo` glob
        would cross two changes' archives."""
        (specs_dir / "archive" / "2026-07-26-bar-foo").mkdir(parents=True)
        (specs_dir / "archive" / "2026-07-20-foo").mkdir()
        (specs_dir / "archive" / "2026-07-25-foo").mkdir()
        (specs_dir / "archive" / "not-a-date-foo").mkdir()
        assert archived_change_dirs(specs_dir, "foo") == [
            specs_dir / "archive" / "2026-07-20-foo",
            specs_dir / "archive" / "2026-07-25-foo",
        ]
        assert archived_change_dirs(specs_dir, "bar-foo") == [
            specs_dir / "archive" / "2026-07-26-bar-foo",
        ]

    def test_no_archive_root_or_empty_name(self, tmp_path):
        assert archived_change_dirs(tmp_path / "specs", "foo") == []
        assert archived_change_dirs(tmp_path / "specs", "") == []


class TestArchiveChange:
    def test_moves_to_archive(self, specs_dir):
        change = _change(specs_dir, "done")
        (change / "proposal.md").write_text("## Capabilities\n- `cap1`\n")
        dest = archive_change(change, specs_dir, merge_specs=False)
        assert dest.exists()
        assert not change.exists()
        assert "done" in dest.name

    def test_merges_delta_requirements(self, specs_dir):
        change = _change(specs_dir, "done")
        (change / "proposal.md").write_text("## Capabilities\n- `cap1`\n")
        # Create delta requirement
        (change / "requirements").mkdir(parents=True)
        (change / "requirements" / "auth.md").write_text(
            "## ADDED Requirements\n\n"
            "### Requirement: User can login\nThe system SHALL allow login.\n\n"
            "#### Scenario: Valid credentials\n- **WHEN** valid creds\n- **THEN** login succeeds\n"
        )

        archive_change(change, specs_dir, merge_specs=True)
        main_file = specs_dir / "registry" / "auth.md"
        assert main_file.exists()
        assert "User can login" in main_file.read_text()

    def test_archive_name_is_the_date_then_the_change_name(self, specs_dir):
        change = _change(specs_dir, "done")
        dest = archive_change(change, specs_dir, today=date(2026, 10, 21))
        assert dest == specs_dir / "archive" / "2026-10-21-done"
        assert archived_change_dirs(specs_dir, "done") == [dest]

    def test_creates_archive_and_registry_dirs_when_missing(self, tmp_path):
        specs_dir = tmp_path / "specs"
        change = specs_dir / "changes" / "new"
        (change / "requirements").mkdir(parents=True)
        (change / "requirements" / "cap.md").write_text("## ADDED Requirements\n")
        dest = archive_change(change, specs_dir)
        assert dest.parent == specs_dir / "archive"
        assert (
            specs_dir / "registry" / "cap.md"
        ).read_text() == "## ADDED Requirements\n"


class TestDeltaRequirementMerging:
    def test_added_requirements_append(self, specs_dir):
        # Existing main registry file
        registry = specs_dir / "registry"
        registry.mkdir(parents=True, exist_ok=True)
        (registry / "data.md").write_text(
            "### Requirement: Existing\nThe system SHALL exist.\n\n"
            "#### Scenario: It exists\n- **WHEN** checked\n- **THEN** it exists\n"
        )
        # Delta with ADDED
        change = _change(specs_dir, "add-export")
        (change / "requirements").mkdir(parents=True)
        (change / "requirements" / "data.md").write_text(
            "## ADDED Requirements\n\n"
            "### Requirement: Export data\nThe system SHALL export.\n\n"
            "#### Scenario: CSV export\n- **WHEN** export\n- **THEN** CSV\n"
        )
        merge_delta_requirements(change, registry)
        merged = (registry / "data.md").read_text()
        assert "Existing" in merged
        assert "Export data" in merged

    def test_removed_requirements_delete(self, specs_dir):
        registry = specs_dir / "registry"
        registry.mkdir(parents=True, exist_ok=True)
        (registry / "old.md").write_text(
            "### Requirement: Keep me\nStay.\n\n"
            "#### Scenario: Stays\n- **WHEN** x\n- **THEN** y\n\n"
            "### Requirement: Remove me\nGo away.\n\n"
            "#### Scenario: Goes\n- **WHEN** x\n- **THEN** y\n"
        )
        change = _change(specs_dir, "cleanup")
        (change / "requirements").mkdir(parents=True)
        (change / "requirements" / "old.md").write_text(
            "## REMOVED Requirements\n\n"
            "### Requirement: Remove me\n**Reason**: Obsolete\n"
        )
        merge_delta_requirements(change, registry)
        merged = (registry / "old.md").read_text()
        assert "Keep me" in merged
        assert "Remove me" not in merged

    def test_no_requirements_dir_is_a_no_op(self, specs_dir):
        change = _change(specs_dir, "nothing")
        merge_delta_requirements(change, specs_dir / "registry")
        assert list((specs_dir / "registry").iterdir()) == []


def test_merges_delta_into_spec_extract_registry_file(specs_dir):
    (specs_dir / "registry" / "sample.md").write_text(SPEC_EXTRACT_SAMPLE.read_text())
    change = _change(specs_dir, "calc-change")
    (change / "requirements").mkdir(exist_ok=True)
    (change / "requirements" / "sample.md").write_text(
        "## MODIFIED Requirements\n\n"
        "### Requirement: Adds two numbers\nadd SHALL return the sum as an int.\n\n"
        "#### Scenario: adding ints\n- **WHEN** add(1, 2) is called\n- **THEN** it returns the int 3\n\n"
        "## REMOVED Requirements\n\n### Requirement: Subtracts two numbers\n"
    )
    archive_change(change, specs_dir)
    merged = (specs_dir / "registry" / "sample.md").read_text()
    assert merged.startswith("<!-- spec-extract widgets/tree@")
    assert "add SHALL return the sum as an int." in merged
    assert "add SHALL return the sum of its arguments." not in merged
    assert "Subtracts two numbers" not in merged and "sub(3, 1)" not in merged
    assert merged.count("### Requirement:") == 1


def test_requirements_validate_every_requirement_and_allow_removal_deltas():
    valid = (
        "### Requirement: Works\n#### Scenario: Normal\nWHEN called\nTHEN succeeds\n"
    )
    assert valid_requirement_text(valid)
    assert not valid_requirement_text(valid + "### Requirement: Missing scenarios\n")
    assert not valid_requirement_text(
        valid + "#### Scenario: Incomplete\nWHEN called\n"
    )
    assert valid_requirement_text(
        "## REMOVED Requirements\n### Requirement: Old behavior\n"
    )
    assert not valid_requirement_text(
        "## ADDED Requirements\n### Requirement: <!-- name -->\n"
    )


# --- apply_delta: behaviour the merge tests above do not reach ---


class TestApplyDelta:
    def test_modified_replaces_only_the_named_block(self):
        main = (
            "### Requirement: A\nold a\n\n#### Scenario: s\n- **WHEN** x\n- **THEN** y\n\n"
            "### Requirement: B\nold b\n"
        )
        delta = "## MODIFIED Requirements\n\n### Requirement: A\nnew a\n"
        out = apply_delta(main, delta)
        assert out == "### Requirement: A\nnew a\n\n### Requirement: B\nold b\n"

    def test_backslashes_in_a_modified_block_are_inserted_literally(self):
        main = "### Requirement: Parse\nold\n"
        delta = "## MODIFIED Requirements\n\n### Requirement: Parse\nmatches `\\d+` and `\\1`\n"
        assert (
            apply_delta(main, delta)
            == "### Requirement: Parse\nmatches `\\d+` and `\\1`\n"
        )

    def test_added_section_stops_at_the_next_operation_heading(self):
        main = "### Requirement: Old\nold\n"
        delta = (
            "## ADDED Requirements\n\n### Requirement: New\nnew\n\n"
            "## REMOVED Requirements\n\n### Requirement: Old\n"
        )
        assert apply_delta(main, delta) == "### Requirement: New\nnew\n"

    def test_result_ends_with_one_newline(self):
        assert (
            apply_delta("### Requirement: A\na\n\n\n", "") == "### Requirement: A\na\n"
        )


# --- Ported from test_spec_generator.py::TestExtractCapabilities ---
# buildme's `_extract_capabilities` is `expected_capabilities` under another name.


class TestExtractCapabilities:
    def test_extracts_backtick_names(self):
        text = """\
## Capabilities

### New Capabilities
- `user-auth`: Handle user authentication
- `data-export`: Export data to CSV
"""
        caps = expected_capabilities(text)
        assert caps == ["user-auth", "data-export"]

    def test_empty_proposal(self):
        assert expected_capabilities("") == []

    def test_no_capabilities_section(self):
        text = "## Why\nSome motivation\n## What Changes\nSome changes\n"
        assert expected_capabilities(text) == []

    def test_ignores_non_backtick_items(self):
        text = "- regular item\n- `valid-cap`: desc\n- another item\n"
        caps = expected_capabilities(text)
        assert caps == ["valid-cap"]

    def test_impact_dependencies_are_not_capabilities(self):
        """The proposal template asks for every dependency name in backticks
        under `## Impact`. Harvesting the whole file turned those into
        capabilities and wrote a requirements file for each."""
        text = """\
## Capabilities

### New Capabilities
- `password-reset`
- `rate-limiting`

## Impact

- `django-ses` for email sending
- `redis` for the rate-limit counters
"""
        assert expected_capabilities(text) == ["password-reset", "rate-limiting"]

    def test_modified_capabilities_are_still_harvested(self):
        text = """\
## Capabilities

### New Capabilities
- `password-reset`

### Modified Capabilities
- `user-auth`

## Impact
- `django-ses`
"""
        assert expected_capabilities(text) == ["password-reset", "user-auth"]

    def test_impact_is_skipped_even_without_a_capabilities_heading(self):
        text = "- `password-reset`\n\n## Impact\n- `django-ses`\n"
        assert expected_capabilities(text) == ["password-reset"]

    def test_duplicates_are_listed_once(self):
        assert expected_capabilities("## Capabilities\n- `a`\n- `b`\n- `a`\n") == [
            "a",
            "b",
        ]


# --- requirement-block parsing ---


class TestParseRequirements:
    def test_names_sections_and_scenarios(self):
        text = (
            "## ADDED Requirements\n\n"
            "### Requirement: Login\nThe system SHALL log in.\n\n"
            "#### Scenario: ok\n- **WHEN** valid\n- **THEN** in\n\n"
            "#### Scenario: bad\n- **WHEN** invalid\n- **THEN** refused\n\n"
            "## REMOVED Requirements\n\n"
            "### Requirement: Legacy login\n**Reason**: gone\n"
        )
        reqs = parse_requirements(text)
        assert [(r.name, r.section) for r in reqs] == [
            ("Login", "ADDED"),
            ("Legacy login", "REMOVED"),
        ]
        assert [s.name for s in reqs[0].scenarios] == ["ok", "bad"]
        assert "- **WHEN** invalid" in reqs[0].scenarios[1].body
        assert reqs[1].scenarios == ()
        assert isinstance(reqs[0], RequirementBlock) and isinstance(
            reqs[0].scenarios[0], ScenarioBlock
        )

    def test_registry_file_has_no_section(self):
        reqs = parse_requirements(
            SPEC_EXTRACT_SAMPLE.read_text().replace("## ADDED Requirements\n", "")
        )
        assert [(r.name, r.section) for r in reqs] == [
            ("Adds two numbers", None),
            ("Subtracts two numbers", None),
        ]

    def test_a_requirement_ends_at_the_next_h2(self):
        reqs = parse_requirements("### Requirement: A\nbody\n## Notes\ntrailing\n")
        assert reqs[0].body == "\nbody\n"

    def test_comments_are_removed_first(self):
        reqs = parse_requirements(
            "### Requirement: <!-- name -->\n<!-- ### Requirement: hidden -->\n"
        )
        assert [r.name for r in reqs] == [""]


class TestValidRequirement:
    def test_unreadable_or_missing_file_is_invalid(self, tmp_path):
        assert not valid_requirement(tmp_path / "missing.md")
        (tmp_path / "bad.md").write_bytes(b"\xff\xfe\x00bad")
        assert not valid_requirement(tmp_path / "bad.md")

    def test_valid_file(self, tmp_path):
        p = tmp_path / "ok.md"
        p.write_text(
            "### Requirement: Works\n#### Scenario: Normal\n- **WHEN** called\n- **THEN** succeeds\n"
        )
        assert valid_requirement(p)

    def test_text_with_no_requirement_is_invalid(self):
        assert not valid_requirement_text("## ADDED Requirements\n\nnothing here\n")

    def test_when_and_then_need_text_after_them(self):
        assert not valid_requirement_text(
            "### Requirement: R\n#### Scenario: S\n- **WHEN**\n- **THEN** x\n"
        )

    def test_spec_extract_registry_file_is_valid(self):
        assert valid_requirement_text(SPEC_EXTRACT_SAMPLE.read_text())


# --- the ready record ---


def _what_docs(change: Path) -> Path:
    change.mkdir(parents=True, exist_ok=True)
    (change / "proposal.md").write_text(
        "## Why\nBecause.\n\n## Capabilities\n- `auth`\n"
    )
    (change / "use-cases.md").write_text(
        "## Use cases\n- Ana wants to log in so that she sees her page\n"
    )
    (change / "requirements").mkdir(exist_ok=True)
    (change / "requirements" / "auth.md").write_text(
        "## ADDED Requirements\n\n### Requirement: Login\nLogs in.\n\n"
        "#### Scenario: ok\n- **WHEN** valid\n- **THEN** in\n"
    )
    return change


@pytest.fixture
def change(tmp_path):
    c = _what_docs(tmp_path / "specs" / "changes" / "add-auth")
    (c / ".change.yaml").write_text("schema: spec-driven\ncreated: '2026-10-21'\n")
    return c


class TestChangeMeta:
    def test_missing_or_empty_file_reads_as_empty(self, tmp_path):
        assert read_change_meta(tmp_path) == {}
        (tmp_path / ".change.yaml").write_text("")
        assert read_change_meta(tmp_path) == {}

    def test_damaged_file_raises_naming_it(self, tmp_path):
        (tmp_path / ".change.yaml").write_text("a: [unclosed\n")
        with pytest.raises(ValueError, match=r"\.change\.yaml is not valid YAML"):
            read_change_meta(tmp_path)
        (tmp_path / ".change.yaml").write_text("- a list\n")
        with pytest.raises(ValueError, match="does not hold a mapping"):
            read_change_meta(tmp_path)

    def test_write_keeps_unknown_keys_in_order(self, change):
        (change / ".change.yaml").write_text(
            "schema: spec-driven\ncreated: '2026-10-21'\nplane_refs:\n- SPK-12\nspec_status: draft\n"
        )
        written = write_change_meta(
            change, {"spec_status": "ready", "spec_issues": None}
        )
        assert list(written) == [
            "schema",
            "created",
            "plane_refs",
            "spec_status",
            "spec_issues",
        ]
        assert read_change_meta(change) == {
            "schema": "spec-driven",
            "created": "2026-10-21",
            "plane_refs": ["SPK-12"],
            "spec_status": "ready",
            "spec_issues": None,
        }
        assert "spec_issues: null" in (change / ".change.yaml").read_text()

    def test_write_creates_the_file(self, tmp_path):
        write_change_meta(tmp_path, {"spec_status": "draft"})
        assert read_change_meta(tmp_path) == {"spec_status": "draft"}
        assert [p.name for p in tmp_path.iterdir()] == [".change.yaml"], (
            "no temporary file left"
        )

    def test_write_refuses_to_overwrite_a_damaged_file(self, tmp_path):
        (tmp_path / ".change.yaml").write_text("a: [unclosed\n")
        with pytest.raises(ValueError):
            write_change_meta(tmp_path, {"spec_status": "draft"})
        assert (tmp_path / ".change.yaml").read_text() == "a: [unclosed\n"


class TestSpecRevision:
    def test_is_the_documented_algorithm(self, change):
        import hashlib

        h = hashlib.sha256()
        for rel in ("proposal.md", "requirements/auth.md", "use-cases.md"):
            data = (change / rel).read_bytes()
            h.update(rel.encode() + b"\0" + str(len(data)).encode() + b"\0" + data)
        assert spec_revision(change) == h.hexdigest()
        assert [p.relative_to(change).as_posix() for p in spec_files(change)] == [
            "proposal.md",
            "requirements/auth.md",
            "use-cases.md",
        ]

    def test_only_the_what_documents_count(self, change):
        before = spec_revision(change)
        (change / "design.md").write_text("## Decisions\n")
        (change / "interview.md").write_text("Q&A\n")
        (change / "requirements" / "notes.txt").write_text("not markdown\n")
        (change / "requirements" / "nested").mkdir()
        (change / "requirements" / "nested" / "deep.md").write_text(
            "### Requirement: X\n"
        )
        write_change_meta(change, {"spec_status": "ready"})
        assert spec_revision(change) == before

    def test_lf_normalisation(self, tmp_path):
        lf = _what_docs(tmp_path / "lf")
        crlf = _what_docs(tmp_path / "crlf")
        cr = _what_docs(tmp_path / "cr")
        for p in spec_files(crlf):
            p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
        for p in spec_files(cr):
            p.write_bytes(p.read_bytes().replace(b"\n", b"\r"))
        assert b"\r\n" in (crlf / "proposal.md").read_bytes()
        assert spec_revision(lf) == spec_revision(crlf) == spec_revision(cr)

    def test_content_add_remove_and_rename_change_it(self, change):
        seen = {spec_revision(change)}
        (change / "requirements" / "auth.md").write_text(
            (change / "requirements" / "auth.md").read_text() + "\n"
        )
        seen.add(spec_revision(change))
        (change / "requirements" / "auth.md").rename(
            change / "requirements" / "login.md"
        )
        seen.add(spec_revision(change))
        (change / "requirements" / "extra.md").write_text("")
        seen.add(spec_revision(change))
        (change / "use-cases.md").unlink()
        seen.add(spec_revision(change))
        assert len(seen) == 5

    def test_moving_text_between_files_changes_it(self, tmp_path):
        a = tmp_path / "a"
        (a / "requirements").mkdir(parents=True)
        (a / "requirements" / "x.md").write_text("ab")
        (a / "requirements" / "y.md").write_text("")
        b = tmp_path / "b"
        (b / "requirements").mkdir(parents=True)
        (b / "requirements" / "x.md").write_text("a")
        (b / "requirements" / "y.md").write_text("b")
        assert spec_revision(a) != spec_revision(b)

    def test_empty_change_hashes_the_empty_input(self, tmp_path):
        assert spec_revision(tmp_path) == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )


class TestIsReady:
    def test_ready(self, change):
        revision = mark_ready(change, now=datetime(2026, 10, 21, 9, 30, tzinfo=UTC))
        assert is_ready(change) == (True, "")
        meta = read_change_meta(change)
        assert meta["spec_status"] == "ready"
        assert meta["spec_revision"] == revision == spec_revision(change)
        assert meta["spec_checked_at"] == "2026-10-21T09:30:00+00:00"
        assert meta["schema"] == "spec-driven" and meta["created"] == "2026-10-21"

    def test_draft(self, change):
        mark_ready(change)
        mark_draft(change)
        ok, reason = is_ready(change)
        assert not ok and reason == "spec_status is 'draft', not 'ready'"
        assert read_change_meta(change)["spec_revision"] == spec_revision(change), (
            "last check kept"
        )

    def test_edited_file_after_ready(self, change):
        mark_ready(change)
        (change / "requirements" / "auth.md").write_text(
            (change / "requirements" / "auth.md")
            .read_text()
            .replace("Logs in.", "Logs in fast.")
        )
        ok, reason = is_ready(change)
        assert not ok
        assert reason.startswith("the specs changed after they were marked ready")
        mark_ready(change)
        assert is_ready(change) == (True, "")

    def test_line_ending_change_after_ready_is_still_ready(self, change):
        mark_ready(change)
        p = change / "proposal.md"
        p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
        assert is_ready(change) == (True, "")

    def test_old_change_yaml_without_the_keys(self, change):
        """A `.change.yaml` buildme wrote before the ready record existed."""
        ok, reason = is_ready(change)
        assert (
            not ok
            and reason == "spec_status is not set: the specs were never marked ready"
        )

    def test_ready_without_a_revision(self, change):
        write_change_meta(change, {"spec_status": "ready"})
        assert is_ready(change) == (
            False,
            "spec_status is ready but spec_revision is missing",
        )

    def test_no_change_yaml(self, tmp_path):
        ok, reason = is_ready(tmp_path)
        assert not ok and reason == f"no .change.yaml in {tmp_path}"

    def test_damaged_change_yaml_is_one_line(self, change):
        (change / ".change.yaml").write_text("a: [unclosed\n")
        ok, reason = is_ready(change)
        assert (
            not ok
            and reason.startswith(".change.yaml cannot be read:")
            and "\n" not in reason
        )

    def test_open_spec_issues_do_not_change_readiness(self, change):
        mark_ready(change)
        set_spec_issues(change, change / "spec-issues.md")
        assert is_ready(change) == (True, "")


class TestMarkReady:
    def test_naive_now_is_utc_and_aware_now_is_converted(self, change):
        mark_ready(change, now=datetime(2026, 10, 21, 9, 30))
        assert (
            read_change_meta(change)["spec_checked_at"] == "2026-10-21T09:30:00+00:00"
        )
        mark_ready(
            change,
            now=datetime(2026, 10, 21, 11, 30, tzinfo=timezone(timedelta(hours=2))),
        )
        assert (
            read_change_meta(change)["spec_checked_at"] == "2026-10-21T09:30:00+00:00"
        )

    def test_default_now_is_the_current_utc_time(self, change):
        before = datetime.now(UTC).replace(microsecond=0)
        mark_ready(change)
        checked = datetime.fromisoformat(read_change_meta(change)["spec_checked_at"])
        assert before <= checked <= datetime.now(UTC)

    def test_leaves_spec_issues_alone(self, change):
        set_spec_issues(change, "spec-issues.md")
        mark_ready(change)
        assert read_change_meta(change)["spec_issues"] == "spec-issues.md"


class TestSpecIssues:
    def test_path_inside_the_change_is_saved_relative(self, change):
        set_spec_issues(change, change / "spec-issues.md")
        assert read_change_meta(change)["spec_issues"] == "spec-issues.md"
        set_spec_issues(change, "spec-issues.md")
        assert read_change_meta(change)["spec_issues"] == "spec-issues.md"

    def test_path_outside_is_saved_as_given(self, change, tmp_path):
        outside = tmp_path / "elsewhere" / "spec-issues.md"
        set_spec_issues(change, outside)
        assert read_change_meta(change)["spec_issues"] == str(outside)

    def test_clear_writes_null(self, change):
        set_spec_issues(change, "spec-issues.md")
        clear_spec_issues(change)
        meta = read_change_meta(change)
        assert "spec_issues" in meta and meta["spec_issues"] is None


# --- nested components and specs.md ---
#
# The samples are the auth-session change on the pm-spec workflow samples
# page: two components, token-retry and timeout-setting, whose specs.md files
# touch three capabilities.

SESSION_LENGTH = (
    "## ADDED Requirements\n\n"
    "### Requirement: A session lasts the account's session length\n"
    "A new session expires after the session length the admin set.\n\n"
    "#### Scenario: Admin set 8 hours\n"
    "- **WHEN** a user signs in on an account set to 8 hours\n"
    "- **THEN** the session is still valid 7 hours 59 minutes later\n"
)
ACCOUNT_SETTINGS = (
    "## MODIFIED Requirements\n\n"
    "### Requirement: Only admins change account settings\n"
    "Only an account admin changes the settings, now with the session length.\n\n"
    "#### Scenario: Out-of-range session length\n"
    "- **WHEN** an admin enters 20 hours\n"
    "- **THEN** the value is refused\n"
)
SESSION_REFRESH = (
    "## ADDED Requirements\n\n"
    "### Requirement: Retry the refresh after the device wakes\n"
    "A failed refresh right after a wake is retried, not treated as a logout.\n\n"
    "#### Scenario: Network back after 5 seconds\n"
    "- **WHEN** the refresh fails while the network comes back\n"
    "- **THEN** the session is kept\n"
)
TIMEOUT_SPECS = (
    f"# Capability: session-length\n\n{SESSION_LENGTH}\n"
    f"# Capability: account-settings\n\n{ACCOUNT_SETTINGS}"
)
COMPONENTS_TABLE = (
    "# Components: Stop logging people out in the middle of a task\n\n"
    "| Component        | Depends on   | Ticket  |\n"
    "|------------------|--------------|---------|\n"
    "| token-retry      |              | pending |\n"
    "| timeout-setting  | token-retry  | pending |\n"
)


def _component(change: Path, name: str, specs: str | None = None) -> Path:
    d = change / "components" / name
    d.mkdir(parents=True)
    (d / "plan.md").write_text(f"# Plan: {name}\n")
    (d / ".change.yaml").write_text("depends_on: []\n")
    if specs is not None:
        (d / "specs.md").write_text(specs)
    return d


@pytest.fixture
def nested(specs_dir):
    """auth-session with both components, as at handoff."""
    change = _change(specs_dir, "auth-session")
    (change / "proposal.md").write_text("## Why\nPeople get logged out.\n")
    (change / "use-cases.md").write_text("## Use cases\n- Ana keeps her form\n")
    (change / "components.md").write_text(COMPONENTS_TABLE)
    _component(
        change, "token-retry", f"# Capability: session-refresh\n\n{SESSION_REFRESH}"
    )
    _component(change, "timeout-setting", TIMEOUT_SPECS)
    return change


class TestComponentDirs:
    def test_sorted_dirs_that_hold_a_plan(self, nested):
        (nested / "components" / "no-plan").mkdir()
        (nested / "components" / "stray.md").write_text("x")
        assert component_dirs(nested) == [
            nested / "components" / "timeout-setting",
            nested / "components" / "token-retry",
        ]

    def test_none_without_components(self, change):
        assert component_dirs(change) == []
        assert component_dirs(change / "missing") == []


class TestChangeRoot:
    def test_component_belongs_to_its_change(self, nested):
        assert change_root(nested / "components" / "token-retry") == nested

    def test_a_change_is_its_own_root(self, nested, change):
        assert change_root(nested) == nested
        assert change_root(change) == change, "old layout"

    def test_needs_the_changes_dir_above(self, tmp_path):
        stray = tmp_path / "x" / "components" / "y"
        assert change_root(stray) == stray
        named = tmp_path / "specs" / "changes" / "components"
        assert change_root(named) == named


class TestSplitCapabilities:
    def test_sections_in_file_order_are_todays_delta_files(self):
        sections = split_capabilities(TIMEOUT_SPECS)
        assert list(sections) == ["session-length", "account-settings"]
        assert sections == {
            "session-length": SESSION_LENGTH,
            "account-settings": ACCOUNT_SETTINGS,
        }

    def test_text_before_the_first_heading_is_ignored(self):
        assert split_capabilities("Intro\n## ADDED Requirements\n" + TIMEOUT_SPECS) == (
            split_capabilities(TIMEOUT_SPECS)
        )
        assert split_capabilities("no headings\n") == {}

    def test_heading_in_a_comment_is_not_a_section(self):
        text = "<!--\n# Capability: <name>\n-->\n" + TIMEOUT_SPECS
        assert list(split_capabilities(text)) == ["session-length", "account-settings"]

    def test_empty_section(self):
        assert split_capabilities("# Capability: a\n\n# Capability: b\nx\n") == {
            "a": "",
            "b": "x\n",
        }

    def test_repeated_name_is_an_error(self):
        with pytest.raises(ValueError, match="'session-length' appears twice"):
            split_capabilities(TIMEOUT_SPECS + "\n# Capability: session-length\n")

    def test_name_that_is_not_kebab_case_is_an_error(self):
        with pytest.raises(ValueError, match="'Session length' is not kebab-case"):
            split_capabilities("# Capability: Session length\n" + SESSION_LENGTH)


class TestValidSpecsText:
    def test_every_section_valid(self):
        assert valid_specs_text(TIMEOUT_SPECS)

    def test_one_invalid_section_fails_it(self):
        assert not valid_specs_text(
            TIMEOUT_SPECS + "\n# Capability: other\n\n## ADDED Requirements\n\n"
            "### Requirement: No scenario\nText.\n"
        )
        assert not valid_specs_text(TIMEOUT_SPECS + "\n# Capability: empty\n")

    def test_needs_a_section(self):
        assert not valid_specs_text(SESSION_LENGTH), "a delta with no heading"
        assert not valid_specs_text("")

    def test_split_error_is_false(self):
        assert not valid_specs_text(TIMEOUT_SPECS + TIMEOUT_SPECS)


class TestRequirementTexts:
    def test_reads_specs_md(self, nested):
        assert requirement_texts(nested / "components" / "timeout-setting") == {
            "session-length": SESSION_LENGTH,
            "account-settings": ACCOUNT_SETTINGS,
        }

    def test_old_layout_reads_requirements_sorted(self, specs_dir):
        change = _change(specs_dir, "old")
        (change / "requirements").mkdir()
        (change / "requirements" / "b.md").write_text(SESSION_LENGTH)
        (change / "requirements" / "a.md").write_text(ACCOUNT_SETTINGS)
        (change / "requirements" / "notes.txt").write_text("x")
        texts = requirement_texts(change)
        assert list(texts) == ["a", "b"]
        assert texts == {"a": ACCOUNT_SETTINGS, "b": SESSION_LENGTH}

    def test_specs_md_wins_over_requirements(self, change):
        (change / "specs.md").write_text(TIMEOUT_SPECS)
        assert list(requirement_texts(change)) == ["session-length", "account-settings"]

    def test_neither_is_empty(self, nested):
        assert requirement_texts(nested) == {}


class TestCapabilities:
    def test_from_specs_md(self, nested):
        assert capabilities(nested / "components" / "timeout-setting") == [
            "session-length",
            "account-settings",
        ]

    def test_old_layout_from_requirements(self, change):
        (change / "requirements" / "billing.md").write_text(SESSION_LENGTH)
        assert capabilities(change) == ["auth", "billing"]

    def test_falls_back_to_the_proposal(self, change):
        shutil.rmtree(change / "requirements")
        assert capabilities(change) == ["auth"]

    def test_nothing(self, nested, tmp_path):
        assert capabilities(nested) == []
        assert capabilities(tmp_path) == []


class TestNestedMerge:
    def test_archive_merges_two_components_into_three_registry_files(
        self, specs_dir, nested
    ):
        registry = specs_dir / "registry"
        (registry / "account-settings.md").write_text(
            "### Requirement: Only admins change account settings\nOld text.\n\n"
            "#### Scenario: Non-admin\n- **WHEN** a member opens settings\n"
            "- **THEN** they are refused\n"
        )
        (registry / "session-refresh.md").write_text(
            "### Requirement: Refresh before expiry\nRefreshes.\n\n"
            "#### Scenario: Near expiry\n- **WHEN** 1 minute is left\n"
            "- **THEN** the token is refreshed\n"
        )
        dest = archive_change(nested, specs_dir, today=date(2026, 10, 23))
        assert dest == specs_dir / "archive" / "2026-10-23-auth-session"
        assert sorted(p.name for p in registry.iterdir()) == [
            "account-settings.md",
            "session-length.md",
            "session-refresh.md",
        ]
        account = (registry / "account-settings.md").read_text()
        assert "now with the session length" in account and "Old text." not in account
        assert (registry / "session-length.md").read_text() == SESSION_LENGTH
        refresh = (registry / "session-refresh.md").read_text()
        assert refresh.index("Refresh before expiry") < refresh.index(
            "Retry the refresh after the device wakes"
        )

    def _three_adding(self, specs_dir, table: str | None) -> Path:
        change = _change(specs_dir, "ordered")
        if table is not None:
            (change / "components.md").write_text(table)
        for name in ("alpha", "beta", "gamma"):
            _component(
                change,
                name,
                f"# Capability: shared\n\n## ADDED Requirements\n\n"
                f"### Requirement: From {name}\nText.\n",
            )
        return change

    def _order(self, registry: Path) -> list[str]:
        return [
            line.removeprefix("### Requirement: From ")
            for line in (registry / "shared.md").read_text().splitlines()
            if line.startswith("### Requirement:")
        ]

    def test_components_md_order_then_unlisted_sorted(self, specs_dir):
        change = self._three_adding(
            specs_dir,
            "| Component | Depends on | Ticket |\n|---|---|---|\n"
            "| `gamma` | | pending |\n| alpha | gamma | pending |\n",
        )
        merge_delta_requirements(change, specs_dir / "registry")
        assert self._order(specs_dir / "registry") == ["gamma", "alpha", "beta"]

    def test_sorted_without_components_md(self, specs_dir):
        change = self._three_adding(specs_dir, None)
        merge_delta_requirements(change, specs_dir / "registry")
        assert self._order(specs_dir / "registry") == ["alpha", "beta", "gamma"]

    def test_single_component_change_merges_its_own_specs_md(self, specs_dir):
        change = _change(specs_dir, "single")
        (change / "plan.md").write_text("# Plan\n")
        (change / "specs.md").write_text(TIMEOUT_SPECS)
        merge_delta_requirements(change, specs_dir / "registry")
        assert sorted(p.name for p in (specs_dir / "registry").iterdir()) == [
            "account-settings.md",
            "session-length.md",
        ]

    def test_a_bad_specs_md_writes_nothing(self, specs_dir, nested):
        (nested / "components" / "token-retry" / "specs.md").write_text(
            TIMEOUT_SPECS + TIMEOUT_SPECS
        )
        with pytest.raises(ValueError):
            merge_delta_requirements(nested, specs_dir / "registry")
        assert list((specs_dir / "registry").iterdir()) == []


class TestComponentRevision:
    def test_files_are_the_roots_what_documents_and_the_components(self, nested):
        component = nested / "components" / "token-retry"
        assert [p.relative_to(nested).as_posix() for p in spec_files(component)] == [
            "components/token-retry/plan.md",
            "components/token-retry/specs.md",
            "proposal.md",
            "use-cases.md",
        ]
        import hashlib

        h = hashlib.sha256()
        for p in spec_files(component):
            rel, data = p.relative_to(nested).as_posix(), p.read_bytes()
            h.update(rel.encode() + b"\0" + str(len(data)).encode() + b"\0" + data)
        assert spec_revision(component) == h.hexdigest()

    def test_change_root_hashes_its_plan_and_specs_md(self, change):
        (change / "plan.md").write_text("# Plan\n")
        (change / "specs.md").write_text(TIMEOUT_SPECS)
        assert [p.relative_to(change).as_posix() for p in spec_files(change)] == [
            "plan.md",
            "proposal.md",
            "requirements/auth.md",
            "specs.md",
            "use-cases.md",
        ]

    @pytest.mark.parametrize(
        "rel", ["proposal.md", "use-cases.md", "components/token-retry/plan.md"]
    )
    def test_an_edit_makes_a_ready_component_stale(self, nested, rel):
        component = nested / "components" / "token-retry"
        mark_ready(component)
        assert is_ready(component) == (True, "")
        (nested / rel).write_text((nested / rel).read_text() + "More.\n")
        ok, reason = is_ready(component)
        assert not ok and reason.startswith("the specs changed")

    def test_the_other_components_files_do_not_count(self, nested):
        component = nested / "components" / "token-retry"
        before = spec_revision(component)
        (nested / "components" / "timeout-setting" / "specs.md").write_text("x")
        (nested / "components.md").write_text("changed")
        assert spec_revision(component) == before
