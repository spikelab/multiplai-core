"""Shared fixtures for multiplai-core unit tests.

The fake SDK/``anthropic`` harnesses these suites share live in ``_fakes.py``
next door; see the note at the foot of this file for why.
"""

import os
import sys
from pathlib import Path

import pytest


def _scrub_plugin_env(monkeypatch):
    """Remove ambient CLAUDE_PLUGIN_* / WORKSPACE / CLAUDE_PROJECT_DIR so the
    workspace-anchored path resolver can't pick up the real host environment.

    ``CLAUDE_PROJECT_DIR`` joined this list when marker-based workspace
    discovery did: a leaked value would let the resolver walk up to a real
    ``.multiplai/`` on the developer's machine and point tests at a live
    corpus.
    """
    for key in list(os.environ):
        if (
            key.startswith("CLAUDE_PLUGIN")
            or key == "WORKSPACE"
            or key == "CLAUDE_PROJECT_DIR"
        ):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Scrub ambient CLAUDE_PLUGIN_* / WORKSPACE before every test.

    A leaked host ``WORKSPACE`` (or ``CLAUDE_PLUGIN_OPTION_*``) would point
    resolution at the real environment and break isolation. Tests that need
    these set them explicitly via monkeypatch (applied after this autouse
    fixture).
    """
    _scrub_plugin_env(monkeypatch)


@pytest.fixture
def clean_env(monkeypatch):
    """Explicit alias of the autouse scrub, kept so tests can name the
    dependency where the isolation intent matters to the reader."""
    _scrub_plugin_env(monkeypatch)


@pytest.fixture
def reset_paths_cache():
    from multiplai_core.paths import _reset_cache

    _reset_cache()
    yield
    _reset_cache()


@pytest.fixture
def tmp_workspace(monkeypatch, tmp_path, reset_paths_cache):
    """Anchor the path resolver at a temp workspace; yield its root."""
    monkeypatch.setenv("WORKSPACE", str(tmp_path))
    yield tmp_path


# ---------------------------------------------------------------------------
# The fake SDK/anthropic harnesses live in ``_fakes.py``. pytest imports this
# conftest itself under every import mode, so putting the directory on
# sys.path here is what lets the test modules say ``from _fakes import …``
# without depending on the legacy ``prepend`` mode.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
