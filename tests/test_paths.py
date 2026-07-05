"""Tests for the path resolver module (multiplai_core/paths.py).

Covers:
- Plugin environment variable resolution
- Standalone fallback resolution
- Plugin mode detection
- Derived path accessors
- All accessors return Path objects
- Environment variable override precedence
- Partial plugin environment configuration
- Path expansion and normalization
- Empty environment variable handling
- Thread safety and immutability
"""

import asyncio
import dataclasses
import os
import threading
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Plugin Environment Variable Resolution
# ---------------------------------------------------------------------------


class TestPluginEnvResolution:
    """Requirement: Plugin environment variable resolution.

    The paths module MUST resolve file locations from plugin environment
    variables when running inside a Claude Code plugin context.
    """

    def test_resolve_plugin_root_from_env(self, monkeypatch, reset_paths_cache):
        """Scenario: Resolve plugin root from environment."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "/home/user/.claude/plugins/multiplai")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.plugin_root() == Path("/home/user/.claude/plugins/multiplai")

    def test_resolve_plugin_data_from_env(self, monkeypatch, reset_paths_cache):
        """Scenario: Resolve plugin data directory from environment."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "/plugin")
        monkeypatch.setenv(
            "CLAUDE_PLUGIN_DATA", "/home/user/.claude/plugins/multiplai/data"
        )
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.plugin_data() == Path("/home/user/.claude/plugins/multiplai/data")

    def test_resolve_memory_dir_from_env(self, monkeypatch, reset_paths_cache):
        """Scenario: Resolve user-configured memory directory."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "/plugin")
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_memory_dir", "/home/user/custom-memory")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.memory_dir() == Path("/home/user/custom-memory")

    def test_resolve_diary_dir_from_env(self, monkeypatch, reset_paths_cache):
        """Scenario: Resolve user-configured diary directory."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "/plugin")
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_diary_dir", "/home/user/custom-diary")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.diary_dir() == Path("/home/user/custom-diary")


# ---------------------------------------------------------------------------
# Standalone Fallback Resolution
# ---------------------------------------------------------------------------


class TestStandaloneFallback:
    """Requirement: Standalone fallback resolution.

    When plugin environment variables are absent, the paths module MUST fall
    back to standalone conventions rooted at ~/.multiplai/.
    """

    def test_fallback_memory_dir(self, clean_env, reset_paths_cache):
        """Scenario: Fallback memory directory when no plugin env vars set."""
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.memory_dir() == Path.home() / ".multiplai" / "memory"

    def test_fallback_diary_dir(self, clean_env, reset_paths_cache):
        """Without workspace_dir env, diary falls back to ~/.multiplai/diary."""
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.diary_dir() == Path.home() / ".multiplai" / "diary"

    def test_workspace_dir_env_anchors_workspace_paths(
        self, clean_env, monkeypatch, reset_paths_cache,
    ):
        """CLAUDE_PLUGIN_OPTION_workspace_dir anchors diary/now/learnings."""
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_workspace_dir", "/ws/proj")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.memory_dir() == Path("/ws/proj/.multiplai/memory")
        assert p.diary_dir() == Path("/ws/proj/.multiplai/diary")
        assert p.now_dir() == Path("/ws/proj/.multiplai/now")
        assert p.learnings_dir() == Path("/ws/proj/.multiplai/learnings")

    def test_fallback_plugin_data(self, clean_env, reset_paths_cache):
        """Scenario: Fallback plugin data directory."""
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.plugin_data() == Path.home() / ".multiplai" / "data"

    def test_fallback_plugin_root(self, clean_env, reset_paths_cache):
        """Scenario: Fallback plugin root directory."""
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.plugin_root() == Path.home() / ".multiplai"


# ---------------------------------------------------------------------------
# Plugin Mode Detection
# ---------------------------------------------------------------------------


class TestPluginModeDetection:
    """Requirement: Plugin mode detection.

    The paths module MUST expose is_plugin_mode() that reports whether
    paths were resolved from plugin environment variables.
    """

    def test_plugin_mode_when_root_set(self, monkeypatch, reset_paths_cache):
        """Scenario: Plugin mode detected when CLAUDE_PLUGIN_ROOT is set."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "/some/path")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.is_plugin_mode() is True

    def test_standalone_mode_when_root_absent(self, clean_env, reset_paths_cache):
        """Scenario: Standalone mode detected when CLAUDE_PLUGIN_ROOT is absent."""
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.is_plugin_mode() is False

    def test_is_plugin_mode_returns_bool(self, clean_env, reset_paths_cache):
        """is_plugin_mode() must return a bool, not a truthy/falsy value."""
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert type(p.is_plugin_mode()) is bool


# ---------------------------------------------------------------------------
# Derived Path Accessors
# ---------------------------------------------------------------------------


class TestDerivedPaths:
    """Requirement: Derived path accessors for known file locations.

    The paths module MUST provide accessors for venv, catalogs, logs,
    dream state, learnings, templates, and scripts.
    """

    def test_venv_derived_from_data(self, monkeypatch, reset_paths_cache):
        """Scenario: Venv path derived from plugin data."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "/plugin")
        monkeypatch.setenv("CLAUDE_PLUGIN_DATA", "/data")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.venv_dir() == Path("/data/venv")

    def test_catalogs_derived_from_data(self, monkeypatch, reset_paths_cache):
        """Scenario: Catalogs path derived from plugin data."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "/plugin")
        monkeypatch.setenv("CLAUDE_PLUGIN_DATA", "/data")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.catalogs_dir() == Path("/data/catalogs")

    def test_logs_derived_from_data(self, monkeypatch, reset_paths_cache):
        """Scenario: Logs path derived from plugin data."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "/plugin")
        monkeypatch.setenv("CLAUDE_PLUGIN_DATA", "/data")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.logs_dir() == Path("/data/logs")

    def test_dream_state_derived_from_data(self, monkeypatch, reset_paths_cache):
        """Scenario: Dream state path derived from plugin data."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "/plugin")
        monkeypatch.setenv("CLAUDE_PLUGIN_DATA", "/data")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.dream_state_file() == Path("/data/dream_state.yaml")

    def test_learnings_path_per_day_under_learnings_dir(
        self, monkeypatch, reset_paths_cache,
    ):
        """Scenario: per-day learnings file lives under learnings_dir."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "/plugin")
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_diary_dir", "/ws/.multiplai/captainslog")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        # Default learnings_dir is sibling of diary_dir
        assert p.learnings_dir() == Path("/ws/.multiplai/learnings")
        assert (
            p.learnings_file("2026-01-01")
            == Path("/ws/.multiplai/learnings/2026-01-01.md")
        )

    def test_learnings_dir_env_override(self, monkeypatch, reset_paths_cache):
        """CLAUDE_PLUGIN_OPTION_learnings_dir overrides the default."""
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_learnings_dir", "/custom-learn")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.learnings_dir() == Path("/custom-learn")
        assert p.learnings_file("2026-01-01") == Path("/custom-learn/2026-01-01.md")

    def test_templates_derived_from_root(self, monkeypatch, reset_paths_cache):
        """Scenario: Templates path derived from plugin root."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "/plugin")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.templates_dir() == Path("/plugin/templates")

    def test_scripts_derived_from_root(self, monkeypatch, reset_paths_cache):
        """Scenario: Scripts path derived from plugin root."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "/plugin")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.scripts_dir() == Path("/plugin/scripts")


# ---------------------------------------------------------------------------
# All Path Accessors Return Path Objects
# ---------------------------------------------------------------------------


class TestReturnTypes:
    """Requirement: All path accessors return Path objects.

    Every public function MUST return a pathlib.Path instance, never a raw string.
    """

    def test_all_accessors_return_path_instances(self, clean_env, reset_paths_cache):
        """Scenario: Return types are Path instances."""
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        accessors = [
            ("plugin_root", p.plugin_root),
            ("plugin_data", p.plugin_data),
            ("memory_dir", p.memory_dir),
            ("diary_dir", p.diary_dir),
            ("venv_dir", p.venv_dir),
            ("catalogs_dir", p.catalogs_dir),
            ("logs_dir", p.logs_dir),
            ("templates_dir", p.templates_dir),
            ("scripts_dir", p.scripts_dir),
            ("dream_state_file", p.dream_state_file),
            ("learnings_file", p.learnings_file),
        ]
        for name, accessor in accessors:
            result = accessor()
            assert isinstance(result, Path), (
                f"{name}() returned {type(result).__name__}, expected Path"
            )


# ---------------------------------------------------------------------------
# Environment Variable Override Precedence
# ---------------------------------------------------------------------------


class TestOverridePrecedence:
    """Requirement: Environment variable override takes precedence over defaults.

    When both a plugin env var and standalone defaults could apply,
    the environment variable MUST win.
    """

    def test_option_overrides_default_memory_path(
        self, monkeypatch, reset_paths_cache
    ):
        """Scenario: CLAUDE_PLUGIN_OPTION overrides default memory path."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "/plugin")
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_memory_dir", "/custom/mem")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.memory_dir() == Path("/custom/mem")
        # Must NOT return something derived from CLAUDE_PLUGIN_ROOT or ~/.multiplai
        assert p.memory_dir() != Path("/plugin") / "memory"
        assert p.memory_dir() != Path.home() / ".multiplai" / "memory"

    def test_data_env_overrides_default_data_path(
        self, monkeypatch, reset_paths_cache
    ):
        """Scenario: Plugin data env var overrides default data path."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "/plugin")
        monkeypatch.setenv("CLAUDE_PLUGIN_DATA", "/custom/data")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.plugin_data() == Path("/custom/data")
        assert p.plugin_data() != Path("/plugin") / "data"


# ---------------------------------------------------------------------------
# Partial Plugin Environment Configuration
# ---------------------------------------------------------------------------


class TestPartialPluginConfig:
    """Requirement: Partial plugin environment configuration.

    When CLAUDE_PLUGIN_ROOT is set but optional vars are absent,
    derive sensible defaults.
    """

    def test_memory_defaults_to_home_when_option_unset(
        self, monkeypatch, reset_paths_cache
    ):
        """Scenario: Memory dir defaults to home when option is unset (no workspace)."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "/plugin")
        monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_memory_dir", raising=False)
        monkeypatch.delenv("WORKSPACE", raising=False)
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.memory_dir() == Path.home() / ".multiplai" / "memory"

    def test_data_defaults_when_root_set_but_data_unset(
        self, monkeypatch, reset_paths_cache
    ):
        """Scenario: Plugin data defaults to workspace/.multiplai/data regardless of plugin mode."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "/plugin")
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_workspace_dir", "/ws")
        monkeypatch.delenv("CLAUDE_PLUGIN_DATA", raising=False)
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.plugin_data() == Path("/ws/.multiplai/data")


# ---------------------------------------------------------------------------
# Path Expansion and Normalization
# ---------------------------------------------------------------------------


class TestPathExpansion:
    """Requirement: Path expansion and normalization.

    The paths module MUST expand ~ and resolve paths to absolute form.
    """

    def test_tilde_expansion_in_memory_dir(self, monkeypatch, reset_paths_cache):
        """Scenario: Tilde expansion in environment variable."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "/plugin")
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_memory_dir", "~/my-memory")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        result = p.memory_dir()
        assert result.is_absolute(), f"Expected absolute path, got {result}"
        assert "~" not in str(result), f"Tilde not expanded in {result}"
        assert str(result).endswith("/my-memory")

    def test_relative_path_resolved_to_absolute(self, monkeypatch, reset_paths_cache):
        """Scenario: Relative path in environment variable is resolved to absolute."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "/plugin")
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_diary_dir", "relative/diary")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        result = p.diary_dir()
        assert result.is_absolute(), (
            f"Expected absolute path for relative env var, got {result}"
        )

    def test_tilde_expansion_in_plugin_root(self, monkeypatch, reset_paths_cache):
        """Tilde in CLAUDE_PLUGIN_ROOT should be expanded."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "~/my-plugin")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        result = p.plugin_root()
        assert result.is_absolute()
        assert "~" not in str(result)

    def test_tilde_expansion_in_plugin_data(self, monkeypatch, reset_paths_cache):
        """Tilde in CLAUDE_PLUGIN_DATA should be expanded."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "/plugin")
        monkeypatch.setenv("CLAUDE_PLUGIN_DATA", "~/my-data")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        result = p.plugin_data()
        assert result.is_absolute()
        assert "~" not in str(result)


# ---------------------------------------------------------------------------
# Empty Environment Variable Handling
# ---------------------------------------------------------------------------


class TestEmptyVars:
    """Requirement: Empty environment variable treated as unset.

    If a plugin env var is set to empty string, treat as unset and use fallback.
    """

    def test_empty_root_treated_as_standalone(self, monkeypatch, reset_paths_cache):
        """Scenario: Empty CLAUDE_PLUGIN_ROOT treated as standalone mode."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.is_plugin_mode() is False
        assert p.plugin_root() == Path.home() / ".multiplai"

    def test_empty_memory_dir_uses_default(self, monkeypatch, reset_paths_cache):
        """Scenario: Empty CLAUDE_PLUGIN_OPTION_memory_dir uses default (no workspace)."""
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_memory_dir", "")
        monkeypatch.delenv("WORKSPACE", raising=False)
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.memory_dir() == Path.home() / ".multiplai" / "memory"

    def test_whitespace_only_root_treated_as_standalone(
        self, monkeypatch, reset_paths_cache
    ):
        """Whitespace-only env var should also be treated as unset."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "   ")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.is_plugin_mode() is False
        assert p.plugin_root() == Path.home() / ".multiplai"

    def test_whitespace_only_memory_dir_uses_default(
        self, monkeypatch, reset_paths_cache
    ):
        """Whitespace-only memory dir env var should use default (no workspace)."""
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_memory_dir", "  \t  ")
        monkeypatch.delenv("WORKSPACE", raising=False)
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.memory_dir() == Path.home() / ".multiplai" / "memory"


# ---------------------------------------------------------------------------
# Thread Safety and Immutability
# ---------------------------------------------------------------------------


class TestCachingAndImmutability:
    """Requirement: Thread safety and immutability per process lifetime.

    Path resolution MUST read env vars once and cache results. Subsequent
    calls MUST return the same values even if env vars change.
    """

    def test_cached_resolution_survives_env_mutation(
        self, monkeypatch, reset_paths_cache
    ):
        """Scenario: Cached resolution survives env var mutation."""
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_memory_dir", "/first")
        from multiplai_core.paths import get_paths, _reset_cache

        _reset_cache()
        first = get_paths().memory_dir()
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_memory_dir", "/second")
        second = get_paths().memory_dir()
        assert first == second, (
            f"Cache broken: first={first}, second={second} after env mutation"
        )

    def test_concurrent_access_returns_consistent_values(
        self, clean_env, reset_paths_cache
    ):
        """Scenario: Concurrent access returns consistent values."""
        from multiplai_core.paths import get_paths, _reset_cache

        _reset_cache()

        async def _check():
            results = await asyncio.gather(
                asyncio.to_thread(lambda: get_paths().plugin_data()),
                asyncio.to_thread(lambda: get_paths().plugin_data()),
            )
            assert results[0] == results[1]

        asyncio.run(_check())

    def test_concurrent_threads_get_same_instance(
        self, clean_env, reset_paths_cache
    ):
        """Multiple threads calling get_paths() get the same Paths object."""
        from multiplai_core.paths import get_paths, _reset_cache

        _reset_cache()
        results = []
        errors = []

        def _worker():
            try:
                results.append(id(get_paths()))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert len(set(results)) == 1, (
            f"Expected single Paths instance, got {len(set(results))} distinct ids"
        )

    def test_paths_immutability(self, clean_env, reset_paths_cache):
        """Paths instances must be immutable — setting attributes must raise."""
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        with pytest.raises(AttributeError):
            p.some_new_attr = "should fail"

    def test_paths_frozen_dataclass(self, clean_env, reset_paths_cache):
        """Paths should be implemented as a frozen dataclass."""
        from multiplai_core.paths import Paths

        assert dataclasses.is_dataclass(Paths), (
            "Paths must be a dataclass (@dataclasses.dataclass(frozen=True))"
        )


# ---------------------------------------------------------------------------
# Dataclass Structure (D2 Spec)
# ---------------------------------------------------------------------------


class TestDataclassStructure:
    """Design D2: Paths dataclass with all seven path fields.

    The block spec requires a frozen dataclass with fields:
    plugin_root, data_dir, memory_dir, diary_dir, venv_dir,
    catalogs_dir, templates_dir.
    """

    def test_paths_has_seven_fields(self, clean_env, reset_paths_cache):
        """Paths dataclass must expose all seven fields from D2 table."""
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        # All seven must be accessible as dataclass fields
        fields = {f.name for f in dataclasses.fields(p)}
        expected = {
            "plugin_root",
            "data_dir",
            "memory_dir",
            "diary_dir",
            "venv_dir",
            "catalogs_dir",
            "templates_dir",
        }
        assert expected.issubset(fields), (
            f"Missing fields: {expected - fields}. Found: {fields}"
        )

    def test_frozen_dataclass_rejects_field_assignment(
        self, clean_env, reset_paths_cache
    ):
        """Frozen dataclass must raise FrozenInstanceError on field assignment."""
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.plugin_root = Path("/new/path")


# ---------------------------------------------------------------------------
# Module-Level Singleton
# ---------------------------------------------------------------------------


class TestModuleSingleton:
    """Block spec: Export module-level `paths = Paths.resolve()` singleton."""

    def test_module_level_paths_exists(self, reset_paths_cache):
        """Module must export a `paths` object at module level."""
        from multiplai_core import paths as paths_mod

        assert hasattr(paths_mod, "paths"), (
            "Module must export 'paths' at module level"
        )

    def test_module_level_paths_is_paths_instance(self, reset_paths_cache):
        """The module-level `paths` must be a Paths instance."""
        from multiplai_core.paths import paths, Paths

        assert isinstance(paths, Paths), (
            f"Module-level paths is {type(paths).__name__}, expected Paths"
        )

    def test_module_level_paths_has_accessors(self, reset_paths_cache):
        """Module-level paths must support all accessor methods."""
        from multiplai_core.paths import paths

        # Verify all accessors are callable
        assert callable(paths.plugin_root)
        assert callable(paths.plugin_data)
        assert callable(paths.memory_dir)
        assert callable(paths.diary_dir)
        assert callable(paths.venv_dir)
        assert callable(paths.catalogs_dir)
        assert callable(paths.logs_dir)
        assert callable(paths.templates_dir)
        assert callable(paths.scripts_dir)
        assert callable(paths.dream_state_file)
        assert callable(paths.learnings_file)
        assert callable(paths.is_plugin_mode)


# ---------------------------------------------------------------------------
# Public import surface
# ---------------------------------------------------------------------------


class TestPublicImports:
    """The documented public import paths resolve (including the lazily-
    resolved `paths` singleton via PEP 562 module __getattr__)."""

    def test_lib_paths_importable(self):
        """from multiplai_core.paths import paths must work (lazy resolution)."""
        from multiplai_core.paths import paths

        assert paths is not None

    def test_lib_paths_get_paths_importable(self):
        """from multiplai_core.paths import get_paths must work."""
        from multiplai_core.paths import get_paths

        assert callable(get_paths)

    def test_lib_paths_class_importable(self):
        """from multiplai_core.paths import Paths must work."""
        from multiplai_core.paths import Paths

        assert callable(Paths.resolve)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Additional edge cases for robustness."""

    def test_all_env_vars_set_simultaneously(self, monkeypatch, reset_paths_cache):
        """When ALL four env vars are set, each is respected independently."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "/root")
        monkeypatch.setenv("CLAUDE_PLUGIN_DATA", "/data")
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_memory_dir", "/memory")
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_diary_dir", "/diary")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.plugin_root() == Path("/root")
        assert p.plugin_data() == Path("/data")
        assert p.memory_dir() == Path("/memory")
        assert p.diary_dir() == Path("/diary")
        assert p.is_plugin_mode() is True

    def test_derived_paths_use_resolved_base(self, monkeypatch, reset_paths_cache):
        """Derived paths must use the resolved (not default) base dirs."""
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", "/custom-root")
        monkeypatch.setenv("CLAUDE_PLUGIN_DATA", "/custom-data")
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_memory_dir", "/custom-mem")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        # Derived from custom data
        assert p.venv_dir() == Path("/custom-data/venv")
        assert p.catalogs_dir() == Path("/custom-data/catalogs")
        assert p.logs_dir() == Path("/custom-data/logs")
        assert p.dream_state_file() == Path("/custom-data/dream_state.yaml")
        # Derived from custom root
        assert p.templates_dir() == Path("/custom-root/templates")
        assert p.scripts_dir() == Path("/custom-root/scripts")
        # Per-day learnings under learnings_dir (defaults to diary parent)
        assert p.learnings_file("2026-01-01") == p.learnings_dir() / "2026-01-01.md"

    def test_reset_cache_allows_re_resolution(self, monkeypatch, reset_paths_cache):
        """After _reset_cache(), next get_paths() resolves fresh from env."""
        from multiplai_core.paths import get_paths, _reset_cache

        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_memory_dir", "/first")
        _reset_cache()
        first = get_paths().memory_dir()

        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_memory_dir", "/second")
        _reset_cache()
        second = get_paths().memory_dir()

        assert first != second, (
            "_reset_cache should allow re-resolution from updated env"
        )
        assert first == Path("/first")
        assert second == Path("/second")

    def test_standalone_mode_derived_paths_consistent(
        self, clean_env, reset_paths_cache
    ):
        """Without any env vars, all paths fall back to ~/.multiplai/.

        Workspace data only diverges from $HOME when
        CLAUDE_PLUGIN_OPTION_workspace_dir is set.
        """
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        home_base = Path.home() / ".multiplai"

        # Plugin-private (follows the user)
        assert p.plugin_root() == home_base
        assert p.plugin_data() == home_base / "data"
        assert p.venv_dir() == home_base / "data" / "venv"
        assert p.catalogs_dir() == home_base / "data" / "catalogs"
        assert p.logs_dir() == home_base / "data" / "logs"
        assert p.templates_dir() == home_base / "templates"
        assert p.scripts_dir() == home_base / "scripts"
        assert p.memory_dir() == home_base / "memory"
        assert p.dream_state_file() == home_base / "data" / "dream_state.yaml"

        # Workspace-scoped — same fallback when no workspace_dir set
        assert p.diary_dir() == home_base / "diary"
        assert p.now_dir() == home_base / "now"
        assert p.learnings_dir() == home_base / "learnings"
        assert p.learnings_file("2026-01-01") == home_base / "learnings" / "2026-01-01.md"

    def test_empty_diary_env_var_uses_default(self, monkeypatch, reset_paths_cache):
        """Empty CLAUDE_PLUGIN_OPTION_diary_dir falls back to ~/.multiplai/diary (no workspace)."""
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_diary_dir", "")
        monkeypatch.delenv("WORKSPACE", raising=False)
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.diary_dir() == Path.home() / ".multiplai" / "diary"

    def test_empty_data_env_var_standalone_fallback(
        self, monkeypatch, reset_paths_cache
    ):
        """Empty CLAUDE_PLUGIN_DATA with no root should use workspace fallback."""
        monkeypatch.delenv("CLAUDE_PLUGIN_ROOT", raising=False)
        monkeypatch.delenv("WORKSPACE", raising=False)
        monkeypatch.setenv("CLAUDE_PLUGIN_DATA", "")
        from multiplai_core.paths import Paths

        p = Paths.resolve()
        assert p.plugin_data() == Path.home() / ".multiplai" / "data"
