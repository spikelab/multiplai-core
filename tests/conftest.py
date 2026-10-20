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


# The variables that choose the default model and effort for a step with none
# configured (multiplai_core.session). Run inside a Claude Code session, the
# real ones would hand every run_agent call the developer's session model.
_SESSION_DEFAULT_VARS = (
    "MULTIPLAI_MODEL",
    "MULTIPLAI_EFFORT",
    "MULTIPLAI_SESSION_MODEL",
    "MULTIPLAI_SESSION_EFFORT",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_EFFORT",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path_factory):
    """Scrub ambient CLAUDE_PLUGIN_* / WORKSPACE before every test.

    A leaked host ``WORKSPACE`` (or ``CLAUDE_PLUGIN_OPTION_*``) would point
    resolution at the real environment and break isolation. Tests that need
    these set them explicitly via monkeypatch (applied after this autouse
    fixture).

    Also clears the session-default variables and points
    ``CLAUDE_MULTIPLAI_HOME`` at an empty directory, so the developer's
    ``multiplai.conf`` and session transcript never set a test's model; and
    points ``CLAUDE_CONFIG_DIR`` at an empty directory, so the developer's
    saved plugin options never set a test's option.
    """
    _scrub_plugin_env(monkeypatch)
    for key in _SESSION_DEFAULT_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path_factory.mktemp("no-conf")))
    # Plugin options fall back to the saved ones in $CLAUDE_CONFIG_DIR's
    # settings.json; an empty config dir keeps the developer's real saved
    # options out of every test, and the per-process cache must not carry a
    # parse from one test into the next.
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path_factory.mktemp("no-claude-config")))
    from multiplai_core.plugin_options import _reset_cache as _reset_options

    _reset_options()


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
