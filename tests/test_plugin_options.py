"""Tests for the plugin-option accessor.

The contract under test is the harness's: Claude Code exports a plugin's
``userConfig`` values as ``CLAUDE_PLUGIN_OPTION_<KEY>`` with ``<KEY>``
uppercased. Reading the lowercase spelling silently misses — the bug this
module exists to make unrepresentable.
"""

import logging
import re
from pathlib import Path

import pytest

from multiplai_core.plugin_options import (
    OPTION_PREFIX,
    option,
    option_bool,
    option_float,
    option_int,
    option_present,
    option_var,
)


class TestOptionVar:
    def test_uppercases_the_key(self):
        assert option_var("enable_skills") == "CLAUDE_PLUGIN_OPTION_ENABLE_SKILLS"

    def test_already_uppercase_is_idempotent(self):
        assert option_var("ENABLE_SKILLS") == "CLAUDE_PLUGIN_OPTION_ENABLE_SKILLS"

    def test_prefix_constant_matches(self):
        assert option_var("x") == f"{OPTION_PREFIX}X"

    @pytest.mark.parametrize(
        "bad",
        ["enable-skills", "enable.skills", "", "1skills", "en able", "skills!"],
    )
    def test_invalid_key_shape_raises(self, bad):
        """A key that cannot round-trip through an env-var name would silently
        read the default forever — developer error, so it raises."""
        with pytest.raises(ValueError, match="invalid plugin option key"):
            option_var(bad)

    def test_accessors_propagate_the_key_validation(self):
        with pytest.raises(ValueError, match="invalid plugin option key"):
            option("enable-skills")
        with pytest.raises(ValueError, match="invalid plugin option key"):
            option_bool("enable-skills", False)
        with pytest.raises(ValueError, match="invalid plugin option key"):
            option_present("enable-skills")


class TestOption:
    def test_resolves_the_uppercase_variable(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_ENABLE_SKILLS", "true")
        assert option("enable_skills") == "true"

    def test_lowercase_variable_is_not_honoured(self, monkeypatch):
        """The dead spelling must stay dead — no fallback."""
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_enable_skills", "true")
        assert option("enable_skills", "false") == "false"

    def test_unset_returns_default(self):
        assert option("enable_skills", "false") == "false"

    def test_unset_without_default_returns_empty(self):
        assert option("enable_skills") == ""

    def test_blank_is_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_SKILLS_DIR", "   \t ")
        assert option("skills_dir", "~/.claude/skills") == "~/.claude/skills"

    def test_value_is_stripped(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_SKILLS_DIR", "  /a/b  ")
        assert option("skills_dir") == "/a/b"


class TestOptionPresent:
    def test_true_when_delivered(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_MEMORY_ROUTER", "llm")
        assert option_present("memory_router") is True

    def test_false_when_unset(self):
        assert option_present("memory_router") is False

    def test_false_when_blank(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_MEMORY_ROUTER", "  ")
        assert option_present("memory_router") is False

    def test_false_for_lowercase_only(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_memory_router", "llm")
        assert option_present("memory_router") is False


class TestOptionBool:
    @pytest.mark.parametrize("raw", ["true", "TRUE", "1", "yes", "On"])
    def test_truthy(self, monkeypatch, raw):
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_ENABLE_COSTS", raw)
        assert option_bool("enable_costs", False) is True

    @pytest.mark.parametrize("raw", ["false", "FALSE", "0", "no", "Off"])
    def test_falsy(self, monkeypatch, raw):
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_ENABLE_COSTS", raw)
        assert option_bool("enable_costs", True) is False

    def test_unset_returns_default(self):
        assert option_bool("enable_costs", True) is True

    def test_lowercase_variable_is_not_honoured(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_enable_costs", "true")
        assert option_bool("enable_costs", False) is False

    def test_malformed_warns_and_defaults(self, monkeypatch, caplog):
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_ENABLE_COSTS", "maybe")
        with caplog.at_level(logging.WARNING):
            assert option_bool("enable_costs", True) is True
        assert "Malformed plugin option enable_costs" in caplog.text


class TestOptionInt:
    def test_parses(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_CHECKPOINT_TIMEOUT_S", "480")
        assert option_int("checkpoint_timeout_s", 900) == 480

    def test_unset_returns_default(self):
        assert option_int("checkpoint_timeout_s", 900) == 900

    def test_lowercase_variable_is_not_honoured(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_checkpoint_timeout_s", "480")
        assert option_int("checkpoint_timeout_s", 900) == 900

    def test_malformed_warns_and_defaults(self, monkeypatch, caplog):
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_CHECKPOINT_TIMEOUT_S", "soon")
        with caplog.at_level(logging.WARNING):
            assert option_int("checkpoint_timeout_s", 900) == 900
        assert "Malformed plugin option checkpoint_timeout_s" in caplog.text


class TestOptionFloat:
    def test_parses(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_KEEP_RATIO", "0.5")
        assert option_float("keep_ratio", 0.3) == 0.5

    def test_unset_returns_default(self):
        assert option_float("keep_ratio", 0.3) == 0.3

    def test_lowercase_variable_is_not_honoured(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_keep_ratio", "0.5")
        assert option_float("keep_ratio", 0.3) == 0.3

    def test_malformed_warns_and_defaults(self, monkeypatch, caplog):
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_KEEP_RATIO", "half")
        with caplog.at_level(logging.WARNING):
            assert option_float("keep_ratio", 0.3) == 0.3
        assert "Malformed plugin option keep_ratio" in caplog.text


class TestNoLowercaseReadsSurvive:
    """Regression guard for the bug class, not for one call site.

    A read spelled ``CLAUDE_PLUGIN_OPTION_<lowercase>`` cannot ever match what
    the harness exports, and it fails silently in production. Docstrings count:
    they are the documentation, and the previous ones taught the wrong form.
    """

    def test_no_lowercase_option_literal_in_src(self):
        src = Path(__file__).resolve().parent.parent / "src" / "multiplai_core"
        pattern = re.compile(r"CLAUDE_PLUGIN_OPTION_[a-z]")
        offenders = [
            f"{path.relative_to(src)}:{n}: {line.strip()}"
            for path in sorted(src.rglob("*.py"))
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if pattern.search(line)
        ]
        assert not offenders, (
            "Plugin options are exported UPPERCASED; these reads/docs can never "
            "match. Use multiplai_core.plugin_options.option(...):\n"
            + "\n".join(offenders)
        )


# --- Fallback to the plugin's saved options in the user settings file -------

import json
import sys
import types

from multiplai_core import plugin_options

PLUGIN_ID = "myplug@mkt"


class _Install:
    """A fake ``$CLAUDE_CONFIG_DIR`` with one installed plugin."""

    def __init__(self, tmp_path, monkeypatch):
        self.monkeypatch = monkeypatch
        self.config = tmp_path / "cfg"
        self.root = self.config / "plugins" / "cache" / "mkt" / "myplug" / "1.0.0"
        (self.root / ".claude-plugin").mkdir(parents=True)
        (self.root / ".claude-plugin" / "plugin.json").write_text(
            json.dumps({"name": "myplug"})
        )
        (self.root / "scripts").mkdir()
        self.script = self.root / "scripts" / "tool.py"
        self.script.write_text("")
        self.write_installed({PLUGIN_ID: self.root})
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(self.config))
        # Nothing in this process is the plugin until a test says so.
        self.run_as(tmp_path / "elsewhere" / "main.py")

    def write_installed(self, mapping):
        (self.config / "plugins" / "installed_plugins.json").write_text(
            json.dumps(
                {
                    "version": 2,
                    "plugins": {
                        key: [{"scope": "user", "installPath": str(path)}]
                        for key, path in mapping.items()
                    },
                }
            )
        )

    def write_settings(self, plugin_configs=None, *, raw=None):
        path = self.config / "settings.json"
        if raw is not None:
            path.write_text(raw)
        else:
            path.write_text(json.dumps({"pluginConfigs": plugin_configs or {}}))
        plugin_options._reset_cache()

    def options(self, opts, plugin_id=PLUGIN_ID):
        self.write_settings({plugin_id: {"options": opts}})

    def run_as(self, script_path):
        """Make *script_path* this process's entry script."""
        main = types.ModuleType("__main__")
        main.__file__ = str(script_path)
        self.monkeypatch.setitem(sys.modules, "__main__", main)
        self.monkeypatch.setattr(sys, "argv", [str(script_path)])
        plugin_options._reset_cache()


@pytest.fixture
def install(tmp_path, monkeypatch):
    plugin_options._reset_cache()
    inst = _Install(tmp_path, monkeypatch)
    yield inst
    plugin_options._reset_cache()


class TestSettingsFallback:
    def test_skill_script_reads_saved_option(self, install):
        install.options({"memory_router": "llm"})
        install.run_as(install.script)
        assert option("memory_router", "token_overlap") == "llm"

    def test_env_var_wins_over_settings(self, install, monkeypatch):
        install.options({"memory_router": "llm"})
        install.run_as(install.script)
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_MEMORY_ROUTER", "token_overlap")
        assert option("memory_router") == "token_overlap"

    def test_blank_env_var_falls_through_to_settings(self, install, monkeypatch):
        install.options({"memory_router": "llm"})
        install.run_as(install.script)
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_MEMORY_ROUTER", "  ")
        assert option("memory_router") == "llm"

    def test_unsaved_option_uses_default(self, install):
        install.options({"memory_router": "llm"})
        install.run_as(install.script)
        assert option("router_model", "haiku") == "haiku"

    def test_boolean_reads_as_the_harness_would_export_it(self, install):
        """The harness exports String(value): JSON true becomes "true"."""
        install.options({"enable_skills": True, "enable_costs": False})
        install.run_as(install.script)
        assert option("enable_skills") == "true"
        assert option_bool("enable_skills", False) is True
        assert option_bool("enable_costs", True) is False

    def test_numbers_read_as_the_harness_would_export_them(self, install):
        install.options({"timeout_s": 30, "whole": 30.0, "ratio": 0.25})
        install.run_as(install.script)
        assert option_int("timeout_s", 5) == 30
        assert option("whole") == "30"
        assert option_int("whole", 5) == 30
        assert option_float("ratio", 0.5) == 0.25

    def test_numeric_string_value_parses(self, install):
        install.options({"checkpoint_timeout_s": "45"})
        install.run_as(install.script)
        assert option_int("checkpoint_timeout_s", 5) == 45

    def test_string_list_joins_like_javascript_string(self, install):
        install.options({"dirs": ["a", "b"]})
        install.run_as(install.script)
        assert option("dirs") == "a,b"

    def test_null_and_blank_values_count_as_unset(self, install):
        install.options({"a": None, "b": "   ", "c": {"nested": 1}})
        install.run_as(install.script)
        assert option("a", "d") == "d"
        assert option("b", "d") == "d"
        assert option("c", "d") == "d"
        assert option_present("a") is False
        assert option_present("b") is False

    def test_saved_value_is_stripped(self, install):
        install.options({"skills_dir": "  /x/y  "})
        install.run_as(install.script)
        assert option("skills_dir") == "/x/y"

    def test_saved_key_matches_regardless_of_case_like_the_env_var(self, install):
        """The env var's name is uppercased, so a hook reads ``Router_Mode``
        and ``router_mode`` alike; a skill script must resolve the same."""
        install.options({"Router_Mode": "llm"})
        install.run_as(install.script)
        assert option("router_mode") == "llm"
        assert option("ROUTER_MODE") == "llm"

    def test_option_present_sees_saved_option(self, install):
        install.options({"enable_skills": True})
        install.run_as(install.script)
        assert option_present("enable_skills") is True
        assert option_present("skills_dir") is False

    def test_malformed_saved_value_warns_and_defaults(self, install, caplog):
        install.options({"timeout_s": "soon"})
        install.run_as(install.script)
        with caplog.at_level(logging.WARNING):
            assert option_int("timeout_s", 5) == 5
        assert "Malformed plugin option timeout_s" in caplog.text

    def test_another_plugins_options_are_not_read(self, install):
        install.write_settings(
            {
                "other@mkt": {"options": {"memory_router": "llm"}},
                PLUGIN_ID: {"options": {}},
            }
        )
        install.run_as(install.script)
        assert option("memory_router", "token_overlap") == "token_overlap"


class TestSettingsFileProblems:
    def test_missing_settings_file_uses_default(self, install):
        install.run_as(install.script)
        assert not (install.config / "settings.json").exists()
        assert option("memory_router", "token_overlap") == "token_overlap"

    def test_malformed_json_uses_default_and_logs_debug(self, install, caplog):
        install.write_settings(raw="{not json")
        install.run_as(install.script)
        with caplog.at_level(logging.DEBUG, logger="multiplai_core.plugin_options"):
            assert option("memory_router", "token_overlap") == "token_overlap"
        assert "cannot read settings file" in caplog.text

    @pytest.mark.parametrize(
        "raw",
        [
            "[]",
            '{"pluginConfigs": []}',
            '{"pluginConfigs": {"myplug@mkt": "x"}}',
            '{"pluginConfigs": {"myplug@mkt": {"options": ["a"]}}}',
        ],
    )
    def test_unexpected_shapes_use_default(self, install, raw):
        install.write_settings(raw=raw)
        install.run_as(install.script)
        assert option("memory_router", "token_overlap") == "token_overlap"

    def test_malformed_installed_plugins_still_uses_cache_layout(self, install):
        (install.config / "plugins" / "installed_plugins.json").write_text("{oops")
        install.options({"memory_router": "llm"})
        install.run_as(install.script)
        assert option("memory_router") == "llm"

    def test_unknown_home_uses_default_instead_of_raising(self, install, monkeypatch):
        install.options({"memory_router": "llm"})
        install.run_as(install.script)
        monkeypatch.delenv("CLAUDE_CONFIG_DIR")

        def no_home():
            raise RuntimeError("Could not determine home directory.")

        monkeypatch.setattr(plugin_options.Path, "home", staticmethod(no_home))
        assert option("memory_router", "token_overlap") == "token_overlap"

    def test_config_dir_defaults_to_home_dot_claude(
        self, install, monkeypatch, tmp_path
    ):
        home = tmp_path / "home"
        (home / ".claude").parent.mkdir(parents=True)
        install.config.rename(home / ".claude")
        monkeypatch.delenv("CLAUDE_CONFIG_DIR")
        monkeypatch.setenv("HOME", str(home))
        script = (
            home
            / ".claude"
            / "plugins"
            / "cache"
            / "mkt"
            / "myplug"
            / "1.0.0"
            / "scripts"
            / "tool.py"
        )
        install.config = home / ".claude"
        install.write_installed({PLUGIN_ID: script.parent.parent})
        install.options({"memory_router": "llm"})
        install.run_as(script)
        assert option("memory_router") == "llm"


class TestPluginIdentification:
    def test_no_plugin_in_sight_uses_default(self, install):
        install.options({"memory_router": "llm"})
        # install's entry script lives outside any installed plugin.
        assert option("memory_router", "token_overlap") == "token_overlap"

    def test_plugin_root_env_identifies_the_plugin(self, install, monkeypatch):
        install.options({"memory_router": "llm"})
        monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(install.root))
        plugin_options._reset_cache()
        assert option("memory_router") == "llm"

    def test_install_record_identifies_a_plugin_outside_the_cache(
        self, install, tmp_path
    ):
        """installed_plugins.json is the exact source: it names the key
        pluginConfigs is filed under, wherever the plugin is installed."""
        root = tmp_path / "elsewhere-installed" / "myplug"
        (root / "scripts").mkdir(parents=True)
        install.write_installed({PLUGIN_ID: root})
        install.options({"memory_router": "llm"})
        install.run_as(root / "scripts" / "tool.py")
        assert option("memory_router") == "llm"

    def test_install_record_wins_over_cache_layout(self, install):
        install.write_installed({"renamed@othermkt": install.root})
        install.write_settings(
            {
                "renamed@othermkt": {"options": {"memory_router": "llm"}},
                PLUGIN_ID: {"options": {"memory_router": "token_overlap"}},
            }
        )
        install.run_as(install.script)
        assert option("memory_router") == "llm"

    def test_cache_layout_identifies_a_version_no_longer_recorded(self, install):
        """A session still running a superseded version dir is still that plugin."""
        old = install.root.parent / "0.9.0"
        (old / "scripts").mkdir(parents=True)
        install.options({"memory_router": "llm"})
        install.run_as(old / "scripts" / "tool.py")
        assert option("memory_router") == "llm"

    def test_dev_checkout_is_not_matched_by_name(self, install, tmp_path):
        """A checkout outside the config dir must never pick up the user's
        saved options: that would leak real settings into test runs."""
        dev = tmp_path / "repo" / "plugins" / "myplug"
        (dev / ".claude-plugin").mkdir(parents=True)
        (dev / ".claude-plugin" / "plugin.json").write_text('{"name": "myplug"}')
        (dev / "scripts").mkdir()
        install.options({"memory_router": "llm"})
        install.run_as(dev / "scripts" / "tool.py")
        assert option("memory_router", "token_overlap") == "token_overlap"

    def test_module_run_from_site_packages_is_not_identified(self, install):
        venv_main = (
            install.root
            / "scripts"
            / ".venv"
            / "lib"
            / "site-packages"
            / "pytest"
            / "__main__.py"
        )
        venv_main.parent.mkdir(parents=True)
        install.options({"memory_router": "llm"})
        install.run_as(venv_main)
        assert option("memory_router", "token_overlap") == "token_overlap"

    @pytest.mark.parametrize("argv0", ["-c", "-", ""])
    def test_no_entry_script_identifies_nothing(self, install, monkeypatch, argv0):
        """``python -c`` / stdin have no script; the cwd must not decide."""
        install.options({"memory_router": "llm"})
        monkeypatch.chdir(install.root / "scripts")
        monkeypatch.setitem(sys.modules, "__main__", types.ModuleType("__main__"))
        monkeypatch.setattr(sys, "argv", [argv0])
        plugin_options._reset_cache()
        assert option("memory_router", "token_overlap") == "token_overlap"

    def test_unreadable_install_record_is_named_in_the_log(self, install, caplog):
        (install.config / "plugins" / "installed_plugins.json").write_text("{oops")
        install.options({"memory_router": "llm"})
        install.run_as(install.script)
        with caplog.at_level(logging.DEBUG, logger="multiplai_core.plugin_options"):
            option("memory_router")
        assert "cannot read install record" in caplog.text

    def test_falls_back_to_argv_when_main_has_no_file(self, install, monkeypatch):
        install.options({"memory_router": "llm"})
        install.run_as(install.script)
        monkeypatch.setitem(sys.modules, "__main__", types.ModuleType("__main__"))
        plugin_options._reset_cache()
        assert option("memory_router") == "llm"


class TestCaching:
    def test_settings_are_read_once_per_process(self, install):
        install.options({"memory_router": "llm"})
        install.run_as(install.script)
        assert option("memory_router") == "llm"
        (install.config / "settings.json").write_text(
            json.dumps(
                {"pluginConfigs": {PLUGIN_ID: {"options": {"memory_router": "x"}}}}
            )
        )
        assert option("memory_router") == "llm"
        plugin_options._reset_cache()
        assert option("memory_router") == "x"

    def test_env_var_is_not_cached(self, install, monkeypatch):
        install.options({"memory_router": "llm"})
        install.run_as(install.script)
        assert option("memory_router") == "llm"
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_MEMORY_ROUTER", "other")
        assert option("memory_router") == "other"
