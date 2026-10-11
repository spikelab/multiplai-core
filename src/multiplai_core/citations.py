"""Check that a cited ``path:line`` range at a commit contains a quote.

Copied from the review and codebase-walkthrough pipelines' ``gates.py`` in
``multiplai-cc-mktplace``, which each carried the same core: read a file at a
commit with ``git show <sha>:<path>`` (cached), take lines
``line_start..line_end``, and look for the whitespace-normalised quote in them.
A fix to how quotes are matched now lands here once.

This module holds only the check. Which claims a pipeline drops, how it counts
them and what it does with a failure stay in that pipeline.

Standard library only, and no model call: ``review-viewer`` depends on
``multiplai-core`` without the ``[sdk]`` extra.
"""

from __future__ import annotations

import functools
import os
import subprocess

# User or repo config must not change what git prints.
GIT_ARGV: tuple[str, ...] = ("git", "-c", "color.ui=never", "-c", "core.quotepath=off")


def git_env() -> dict[str, str]:
    """The environment for a git call: no external diff tool, never prompt."""
    env = dict(os.environ)
    env.pop("GIT_EXTERNAL_DIFF", None)
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


@functools.lru_cache(maxsize=2048)
def file_at(repo: str, sha: str, path: str) -> str | None:
    """The text of *path* at commit *sha* in *repo*, or None when it is not there.

    Results are cached per ``(repo, sha, path)``, so pass a commit sha, not a
    ref such as ``HEAD`` that can move. Bytes that are not UTF-8 are replaced.
    """
    proc = subprocess.run(
        [*GIT_ARGV, "-C", str(repo), "show", f"{sha}:{path}"],
        capture_output=True,
        stdin=subprocess.DEVNULL,
        env=git_env(),
        shell=False,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", errors="replace")


def lines_at(repo: str, sha: str, path: str, start: int, end: int) -> list[str] | None:
    """Lines *start*..*end* of *path* at *sha* (1-based, inclusive), or None.

    The text is split on ``\\n`` the way git numbers lines. An *end* past the
    end of the file returns the lines that exist.
    """
    text = file_at(repo, sha, path)
    if text is None:
        return None
    return text.split("\n")[start - 1:end]


def normalise_ws(text: str) -> str:
    """*text* with every run of whitespace collapsed to one space, ends trimmed."""
    return " ".join(text.split())


def check_quote(repo: str, sha: str, path: str, start: int, end: int, quote: str) -> str | None:
    """None when *quote* is inside lines *start*..*end* of *path* at *sha*.

    Both sides are whitespace-normalised first, so a quote that differs from
    the file only in indentation or line breaks passes. On a failure it
    returns the reason: ``"empty quote"``, ``"path not at commit"`` or
    ``"quote not at cited lines"``.
    """
    wanted = normalise_ws(quote)
    if not wanted:
        return "empty quote"
    lines = lines_at(repo, sha, path, start, end)
    if lines is None:
        return "path not at commit"
    if wanted not in normalise_ws("\n".join(lines)):
        return "quote not at cited lines"
    return None
