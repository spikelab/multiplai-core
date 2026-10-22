"""The cited-lines check: does ``path:line`` at a commit contain a quote.

Every test runs against a real temporary git repo with two commits, so what is
checked is what ``git show`` returns, not a stub of it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from multiplai_core import citations
from multiplai_core.citations import (
    GIT_ARGV,
    check_quote,
    file_at,
    git_env,
    lines_at,
    normalise_ws,
)

FIRST = "def greet(name):\n    return f'hello {name}'\n\n\nVALUE = 1\n"
# The second commit moves greet() down three lines and drops old.py.
SECOND = "import os\n\n\ndef greet(name):\n    return f'hello {name}'\n\n\nVALUE = 2\n"


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> tuple[str, str, str]:
    """(repo path, first sha, second sha)."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "mod.py").write_text(FIRST)
    (root / "old.py").write_text("gone = True\n")
    _git(root, "add", "mod.py", "old.py")
    _git(root, "commit", "-q", "-m", "first")
    first = _git(root, "rev-parse", "HEAD")
    (root / "mod.py").write_text(SECOND)
    _git(root, "rm", "-q", "old.py")
    _git(root, "add", "mod.py")
    _git(root, "commit", "-q", "-m", "second")
    second = _git(root, "rev-parse", "HEAD")
    citations.file_at.cache_clear()
    return str(root), first, second


def test_a_quote_passes_across_a_whitespace_difference(repo):
    path, first, _ = repo
    quote = "def greet(name):\n\t\treturn   f'hello {name}'"
    assert check_quote(path, first, "mod.py", 1, 2, quote) is None


def test_a_quote_at_the_right_lines_in_one_commit_but_not_the_other(repo):
    path, first, second = repo
    quote = "def greet(name):"
    assert check_quote(path, first, "mod.py", 1, 2, quote) is None
    assert (
        check_quote(path, second, "mod.py", 1, 2, quote) == "quote not at cited lines"
    )
    assert check_quote(path, second, "mod.py", 4, 5, quote) is None


def test_a_path_missing_at_a_commit(repo):
    path, first, second = repo
    assert check_quote(path, first, "old.py", 1, 1, "gone = True") is None
    assert (
        check_quote(path, second, "old.py", 1, 1, "gone = True") == "path not at commit"
    )
    assert file_at(path, second, "old.py") is None
    assert lines_at(path, second, "old.py", 1, 1) is None


@pytest.mark.parametrize("quote", ["", "   ", "\n\t "])
def test_an_empty_quote(repo, quote):
    path, first, _ = repo
    assert check_quote(path, first, "mod.py", 1, 2, quote) == "empty quote"


def test_an_empty_quote_is_reported_before_a_missing_path(repo):
    path, _, second = repo
    assert check_quote(path, second, "old.py", 1, 1, "") == "empty quote"


def test_line_end_past_the_end_of_the_file(repo):
    path, first, _ = repo
    # FIRST has 5 lines and a trailing newline, so git's numbering ends at 5.
    assert lines_at(path, first, "mod.py", 5, 500) == ["VALUE = 1", ""]
    assert check_quote(path, first, "mod.py", 5, 500, "VALUE = 1") is None
    assert (
        check_quote(path, first, "mod.py", 6, 500, "VALUE = 1")
        == "quote not at cited lines"
    )


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1000, 1000), (-1, 5), (0, 3), (0, 0), (3, 2), (5, 1)],
)
def test_a_line_range_that_is_not_valid_is_an_empty_window(repo, start, end):
    path, first, _ = repo
    # A start below 1 must not wrap the slice round to the end of the file,
    # and an end before the start covers no lines at all.
    assert lines_at(path, first, "mod.py", start, end) == []
    for quote in ("def greet(name):", "VALUE = 1"):
        assert (
            check_quote(path, first, "mod.py", start, end, quote)
            == "quote not at cited lines"
        )


def test_a_line_range_that_is_not_valid_still_reports_a_missing_path(repo):
    path, _, second = repo
    assert lines_at(path, second, "old.py", 0, 1) is None
    assert (
        check_quote(path, second, "old.py", 0, 1, "gone = True") == "path not at commit"
    )


def test_lines_at_is_one_based_and_inclusive(repo):
    path, _, second = repo
    assert lines_at(path, second, "mod.py", 4, 5) == [
        "def greet(name):",
        "    return f'hello {name}'",
    ]


def test_file_at_returns_the_text_at_that_commit(repo):
    path, first, second = repo
    assert file_at(path, first, "mod.py") == FIRST
    assert file_at(path, second, "mod.py") == SECOND


def test_file_at_replaces_bytes_that_are_not_utf8(tmp_path):
    root = tmp_path / "bin"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "b.txt").write_bytes(b"ok \xff here\n")
    _git(root, "add", "b.txt")
    _git(root, "commit", "-q", "-m", "bytes")
    sha = _git(root, "rev-parse", "HEAD")
    assert file_at(str(root), sha, "b.txt") == "ok � here\n"


def test_normalise_ws():
    assert normalise_ws("  a\n\tb   c \n") == "a b c"
    assert normalise_ws("") == ""


def test_git_argv_and_env_keep_user_config_out(monkeypatch):
    assert GIT_ARGV == ("git", "-c", "color.ui=never", "-c", "core.quotepath=off")
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", "/bin/false")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "1")
    env = git_env()
    assert "GIT_EXTERNAL_DIFF" not in env
    assert env["GIT_TERMINAL_PROMPT"] == "0"


def test_the_module_imports_nothing_outside_the_standard_library():
    """review-viewer installs core without the [sdk] extra."""
    source = Path(citations.__file__).read_text()
    imports = {
        line.split()[1].split(".")[0]
        for line in source.splitlines()
        if line.startswith(("import ", "from "))
    }
    assert imports <= {"__future__", "functools", "os", "subprocess"}
