"""Shared fixtures for multiplai-core unit tests."""

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Scrub ambient CLAUDE_PLUGIN_* / WORKSPACE before every test.

    The path resolver is workspace-anchored: a leaked host ``WORKSPACE``
    (or ``CLAUDE_PLUGIN_OPTION_*``) would point resolution at the real
    environment and break isolation. Tests that need these set them
    explicitly via monkeypatch (applied after this autouse fixture).
    """
    for key in list(os.environ):
        if key.startswith("CLAUDE_PLUGIN") or key == "WORKSPACE":
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("CLAUDE_PLUGIN") or key == "WORKSPACE":
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def reset_paths_cache():
    from multiplai_core.paths import _reset_cache

    _reset_cache()
    yield
    _reset_cache()
