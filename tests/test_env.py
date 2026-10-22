"""Tests for multiplai_core.env."""


import pytest

from multiplai_core import env as env_mod
from multiplai_core.env import (
    CURRENT_MODEL,
    EFFORT_TIERS,
    KNOWN_EFFORTS,
    ModelSpec,
    env_candidates,
    find_project_root,
    load_multiplai_conf,
    parse_model_spec,
    pick_effort,
    pick_model,
    pick_model_spec,
    resolve_effort,
    resolve_model,
    resolve_model_spec,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("MULTIPLAI_ENV_FILE", "CLAUDE_MULTIPLAI_HOME", "CLAUDE_CONFIG_DIR",
              "MULTIPLAI_MODEL", "MULTIPLAI_EFFORT"):
        monkeypatch.delenv(k, raising=False)


class TestDeprecatedResolvers:
    """resolve_model / resolve_effort no longer cap; they return the input."""

    def test_resolve_model_returns_the_request_unchanged(self, monkeypatch):
        monkeypatch.setenv("MULTIPLAI_MODEL", "claude-haiku-4-5")
        with pytest.warns(DeprecationWarning):
            assert resolve_model("claude-opus-4", "claude-sonnet-4-6") == "claude-opus-4"
        with pytest.warns(DeprecationWarning):
            assert resolve_model("claude-fable-5-1") == "claude-fable-5-1"

    def test_resolve_effort_returns_the_request_unchanged(self, monkeypatch):
        monkeypatch.setenv("MULTIPLAI_EFFORT", "low")
        with pytest.warns(DeprecationWarning):
            assert resolve_effort("max", "medium") == "max"
        with pytest.warns(DeprecationWarning):
            assert resolve_effort("xhigh") == "xhigh"


class TestEnvCandidates:
    def test_explicit_first(self, monkeypatch, tmp_path):
        f = tmp_path / "custom.env"
        monkeypatch.setenv("MULTIPLAI_ENV_FILE", str(f))
        assert env_candidates()[0] == f

    def test_home_included(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        assert (tmp_path / ".env") in env_candidates()

    def test_cwd_included(self):
        assert (env_mod.Path.cwd() / ".env") in env_candidates()


class TestFindProjectRoot:
    def test_kit_marker(self, tmp_path):
        (tmp_path / ".env.example").write_text("")
        (tmp_path / "dotfiles").mkdir()
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        assert find_project_root(sub) == tmp_path

    def test_env_fallback(self, tmp_path):
        (tmp_path / ".env").write_text("")
        sub = tmp_path / "a"
        sub.mkdir()
        assert find_project_root(sub) == tmp_path

    def test_none_when_absent(self, tmp_path):
        assert find_project_root(tmp_path) is None


class TestLoadMultiplaiConf:
    def test_sections_and_globals(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        (tmp_path / "multiplai.conf").write_text(
            'MULTIPLAI_MODEL="claude-opus-4-6"\n'
            "[deep-research]\n"
            "MODEL=sonnet\n"
        )
        conf = load_multiplai_conf()
        assert conf["MULTIPLAI_MODEL"] == "claude-opus-4-6"
        assert conf["_sections"]["deep-research"]["MODEL"] == "sonnet"

    def test_missing_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        assert load_multiplai_conf() == {"_sections": {}}

    def test_dotted_section_name(self, monkeypatch, tmp_path):
        # Dotted task keys must parse as section names — pick_model's per-task
        # override channel depends on this.
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        (tmp_path / "multiplai.conf").write_text(
            "[deep-research.parse]\nMODEL=sonnet\n"
        )
        conf = load_multiplai_conf()
        assert conf["_sections"]["deep-research.parse"]["MODEL"] == "sonnet"


def _write_conf(tmp_path, text):
    (tmp_path / "multiplai.conf").write_text(text)


class TestPickModel:
    def test_default_tier_opus(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, "")
        assert pick_model() == CURRENT_MODEL["opus"]

    def test_explicit_sonnet_tier(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, "")
        assert pick_model("sonnet") == CURRENT_MODEL["sonnet"]

    def test_task_override_wins_over_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, "[deep-research.parse]\nMODEL=sonnet\n")
        assert pick_model("opus", task="deep-research.parse") == CURRENT_MODEL["sonnet"]

    def test_missing_task_section_falls_back_to_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, "")
        assert pick_model("opus", task="unconfigured.task") == CURRENT_MODEL["opus"]

    def test_multiplai_model_does_not_change_an_explicit_choice(self, monkeypatch, tmp_path):
        # MULTIPLAI_MODEL is a default for unconfigured steps, never a cap.
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, 'MULTIPLAI_MODEL="haiku"\nMULTIPLAI_EFFORT="low"\n')
        monkeypatch.setenv("MULTIPLAI_MODEL", "haiku")
        monkeypatch.setenv("MULTIPLAI_EFFORT", "low")
        assert pick_model("opus") == CURRENT_MODEL["opus"]
        assert pick_model("fable") == CURRENT_MODEL["fable"]
        assert pick_effort("high") == "high"
        assert pick_effort("max") == "max"

    def test_override_keeps_a_dated_id_as_written(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, "[buildme.implementer]\nMODEL=claude-opus-4-6\n")
        assert pick_model("sonnet", task="buildme.implementer") == "claude-opus-4-6"

    def test_default_tier_may_be_a_full_id(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, "")
        assert pick_model("claude-haiku-4-5-20251001") == "claude-haiku-4-5-20251001"


class TestParseModelSpec:
    def test_bare_id_defaults_to_anthropic(self):
        assert parse_model_spec("claude-opus-4-8") == ModelSpec("anthropic", "claude-opus-4-8")

    def test_provider_prefix(self):
        assert parse_model_spec("openai:gpt-5") == ModelSpec("openai", "gpt-5")

    def test_splits_on_first_colon_only(self):
        # Some vendors put colons inside the model name (ollama tags).
        assert parse_model_spec("ollama:llama3:70b") == ModelSpec("ollama", "llama3:70b")

    def test_provider_case_normalized(self):
        assert parse_model_spec("OpenAI:gpt-5").provider == "openai"

    def test_qualified_round_trips(self):
        spec = parse_model_spec("openai:gpt-5")
        assert spec.qualified == "openai:gpt-5"
        assert parse_model_spec(spec.qualified) == spec

    @pytest.mark.parametrize("bad", ["", "  ", ":gpt-5", "openai:"])
    def test_malformed_rejected(self, bad):
        with pytest.raises(ValueError):
            parse_model_spec(bad)


class TestPickModelSpec:
    def test_default_is_anthropic(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, "")
        spec = pick_model_spec("opus")
        assert spec.is_anthropic and spec.model == CURRENT_MODEL["opus"]

    def test_cross_vendor_override_passes_through(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, "[buildme.reviewer2]\nMODEL=openai:gpt-5\n")
        spec = pick_model_spec("opus", task="buildme.reviewer2")
        assert spec == ModelSpec("openai", "gpt-5")

    def test_anthropic_qualified_override_is_kept(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, "[t]\nMODEL=anthropic:claude-opus-4-8\n")
        monkeypatch.setenv("MULTIPLAI_MODEL", "claude-sonnet-4-6")
        assert pick_model_spec("haiku", task="t") == ModelSpec("anthropic", "claude-opus-4-8")

    def test_anthropic_qualified_family_name_normalizes(self, monkeypatch, tmp_path):
        # `anthropic:opus` must take the same path as a bare `opus`: the
        # literal string "opus" is not a model ID.
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, "[t]\nMODEL=anthropic:opus\n")
        assert pick_model_spec("haiku", task="t") == ModelSpec(
            "anthropic", CURRENT_MODEL["opus"])

    def test_anthropic_qualified_dated_id_passes_through(self, monkeypatch, tmp_path):
        """A dated ID is used exactly as written, qualified or not."""
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, "[t]\nMODEL=anthropic:claude-opus-4-2\n")
        assert pick_model_spec("haiku", task="t").model == "claude-opus-4-2"

    def test_pick_model_rejects_a_cross_vendor_override(self, monkeypatch, tmp_path):
        # Returning the bare "gpt-5" would reach an Anthropic client and 404 at
        # request time, far from the config line responsible.
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, "[t]\nMODEL=openai:gpt-5\n")
        with pytest.raises(ValueError) as exc:
            pick_model("opus", task="t")
        assert "pick_model_spec" in str(exc.value)
        assert "openai:gpt-5" in str(exc.value)

    def test_pick_model_unaffected_without_a_provider_prefix(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, "[t]\nMODEL=haiku\n")
        assert pick_model("opus", task="t") == CURRENT_MODEL["haiku"]


class TestPickEffort:
    def test_default_high(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, "")
        assert pick_effort() == "high"

    def test_task_override_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, "[buildme.implementer]\nEFFORT=low\n")
        assert pick_effort("high", task="buildme.implementer") == "low"

    def test_multiplai_effort_does_not_cap_the_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, 'MULTIPLAI_EFFORT="medium"\n[t]\nEFFORT=max\n')
        assert pick_effort("low", task="t") == "max"

    def test_model_and_effort_are_independent_axes(self, monkeypatch, tmp_path):
        # The whole point: a cheap model at high effort is a reachable point.
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, "[t]\nMODEL=haiku\nEFFORT=xhigh\n")
        assert pick_model("opus", task="t") == CURRENT_MODEL["haiku"]
        assert pick_effort("low", task="t") == "xhigh"

    def test_unknown_effort_value_falls_back(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, "[t]\nEFFORT=turbo\n")
        assert pick_effort("low", task="t") == "low"


class TestEffortTiers:
    """The exported tier table — consumers validate against it instead of copying it."""

    def test_exported_from_the_package_root(self):
        import multiplai_core

        assert multiplai_core.EFFORT_TIERS is EFFORT_TIERS
        assert multiplai_core.KNOWN_EFFORTS is KNOWN_EFFORTS
        assert {"EFFORT_TIERS", "KNOWN_EFFORTS"} <= set(multiplai_core.__all__)

    def test_xhigh_sits_between_high_and_max(self):
        # The regression the old copies kept reintroducing.
        assert EFFORT_TIERS["high"] < EFFORT_TIERS["xhigh"] < EFFORT_TIERS["max"]

    def test_ordering_is_strictly_ascending_low_to_max(self):
        ranks = [EFFORT_TIERS[name] for name in ("low", "medium", "high", "xhigh", "max")]
        assert ranks == sorted(ranks)
        assert len(set(ranks)) == len(ranks)

    def test_known_efforts_matches_the_table(self):
        assert KNOWN_EFFORTS == frozenset(EFFORT_TIERS)

    def test_table_is_read_only(self):
        with pytest.raises(TypeError):
            EFFORT_TIERS["turbo"] = 6  # type: ignore[index]

    def test_every_known_effort_survives_pick_effort(self, monkeypatch, tmp_path):
        # The contract that makes the table safe to validate against: a name in
        # KNOWN_EFFORTS is recognized and returned verbatim, never normalized
        # away to "high" the way an unknown name is.
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, "")
        for name in KNOWN_EFFORTS:
            assert pick_effort(name) == name


class TestClaudeAlias:
    def test_claude_prefix_means_anthropic(self):
        assert parse_model_spec("claude:fable") == ModelSpec("anthropic", "fable")
        assert parse_model_spec("CLAUDE:opus").is_anthropic

    @pytest.mark.parametrize("raw", ["claude:opus", "anthropic:opus", "opus"])
    def test_every_spelling_resolves_to_the_same_id(self, monkeypatch, tmp_path, raw):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, f"[t]\nMODEL={raw}\n")
        assert pick_model_spec("haiku", task="t") == ModelSpec(
            "anthropic", CURRENT_MODEL["opus"])


class TestFable:
    def test_fable_is_a_known_family(self):
        assert CURRENT_MODEL["fable"] == "claude-fable-5-1"
        assert parse_model_spec("fable") == parse_model_spec("claude:fable")

    def test_pick_model_fable_task_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, "[t]\nMODEL=fable\n")
        assert pick_model("opus", task="t") == "claude-fable-5-1"


class TestResolveModelSpec:
    def test_first_non_empty_wins(self):
        spec = resolve_model_spec([None, "", "  ", "sonnet", "opus"])
        assert spec == ModelSpec("anthropic", CURRENT_MODEL["sonnet"])

    def test_all_empty_gives_none(self):
        assert resolve_model_spec([None, "", None]) is None
        assert resolve_model_spec([]) is None

    def test_multiplai_model_does_not_cap_a_spec(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MULTIPLAI_MODEL", CURRENT_MODEL["haiku"])
        monkeypatch.setenv("CLAUDE_MULTIPLAI_HOME", str(tmp_path))
        _write_conf(tmp_path, 'MULTIPLAI_MODEL="claude-sonnet-5"\n')
        assert resolve_model_spec(["opus"]).model == CURRENT_MODEL["opus"]
        assert resolve_model_spec(["claude:fable"]).model == CURRENT_MODEL["fable"]

    def test_ceiling_argument_is_ignored_and_deprecated(self):
        with pytest.warns(DeprecationWarning):
            spec = resolve_model_spec(["claude:fable"], ceiling=CURRENT_MODEL["haiku"])
        assert spec == ModelSpec("anthropic", CURRENT_MODEL["fable"])

    def test_family_names_expand_and_dated_ids_pass_through(self):
        assert resolve_model_spec(["anthropic:opus"]).model == CURRENT_MODEL["opus"]
        assert resolve_model_spec(["claude-opus-4-2"]).model == "claude-opus-4-2"
        assert resolve_model_spec(["claude:claude-opus-4-6"]).model == "claude-opus-4-6"

    def test_pi_spec_passes_through(self, monkeypatch):
        monkeypatch.setenv("MULTIPLAI_MODEL", CURRENT_MODEL["haiku"])
        spec = resolve_model_spec([None, "pi:deepseek/deepseek-v4-flash@fireworks", "opus"])
        assert spec == ModelSpec("pi", "deepseek/deepseek-v4-flash@fireworks")

    def test_malformed_first_entry_raises(self):
        with pytest.raises(ValueError):
            resolve_model_spec(["pi:", "opus"])
