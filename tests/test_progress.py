"""ProgressWriter: the line format the review and walkthrough SKILL.md wait loops grep for."""

import re

from multiplai_core import ProgressWriter

# The wait loops in review's and walkthrough's SKILL.md: grep -E '^\[[^]]*\] (DONE|FAILED)'.
WAIT_LOOP = re.compile(r"^\[[^]]*\] (DONE|FAILED)")
LINE = re.compile(r"^\[\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\] (.*)$")


def _lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def test_each_event_is_one_utc_timestamped_line(tmp_path):
    p = ProgressWriter(tmp_path / "run" / "progress.log")
    p.stage("find", "5 finders")
    p.stage("verify")
    p.line("NOTE free text")
    p.done("3 findings")
    bodies = [LINE.match(line).group(1) for line in _lines(tmp_path / "run" / "progress.log")]
    assert bodies == ["STAGE find: 5 finders", "STAGE verify", "NOTE free text", "DONE 3 findings"]


def test_done_and_failed_match_the_skill_wait_loop_and_stage_does_not(tmp_path):
    path = tmp_path / "progress.log"
    p = ProgressWriter(path)
    p.stage("merge")
    p.failed("budget stop")
    p.done("ok")
    matches = [bool(WAIT_LOOP.match(line)) for line in _lines(path)]
    assert matches == [False, True, True]
    assert _lines(path)[1].endswith("] FAILED budget stop")


def test_lines_append_to_an_existing_file(tmp_path):
    path = tmp_path / "progress.log"
    path.write_text("[2026-01-01T00:00:00Z] STARTED x\n")
    ProgressWriter(path).done("y")
    assert len(_lines(path)) == 2


def test_a_pipeline_adds_its_own_started_line(tmp_path):
    class ReviewProgress(ProgressWriter):
        def started(self, label, base, head, files):
            self.line(f"STARTED {label} {base[:12]}..{head[:12]} ({files} files)")

    path = tmp_path / "progress.log"
    ReviewProgress(path).started("pr-1", "a" * 40, "b" * 40, 3)
    assert LINE.match(_lines(path)[0]).group(1) == f"STARTED pr-1 {'a' * 12}..{'b' * 12} (3 files)"


def test_a_str_path_is_accepted(tmp_path):
    ProgressWriter(str(tmp_path / "p.log")).done("x")
    assert (tmp_path / "p.log").is_file()
