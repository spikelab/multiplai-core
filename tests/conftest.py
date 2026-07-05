"""Shared fixtures for multiplai-core unit tests."""

import os

import pytest


def _scrub_plugin_env(monkeypatch):
    """Remove ambient CLAUDE_PLUGIN_* / WORKSPACE so the workspace-anchored
    path resolver can't pick up the real host environment."""
    for key in list(os.environ):
        if key.startswith("CLAUDE_PLUGIN") or key == "WORKSPACE":
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
