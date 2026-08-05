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
