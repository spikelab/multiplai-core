"""Environment + config loading shared across the Multiplai plugins.

`.env` discovery, `multiplai.conf` parsing, and the model/effort ceiling
resolver were copied (and drifted) across buildme and deep-research; this is
the single source of truth.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

log = logging.getLogger(__name__)


def find_project_root(start: Path | None = None) -> Path | None:
    """Walk upward from *start* looking for the multiplai-kit root.

    A directory qualifies if it contains both a ``.env.example`` AND a
    ``dotfiles/`` directory. Falls back to the first ancestor with a ``.env``.
    """
    current = (start or Path.cwd()).resolve()
    for ancestor in [current, *current.parents]:
        if (ancestor / ".env.example").exists() and (ancestor / "dotfiles").is_dir():
            return ancestor
    for ancestor in [current, *current.parents]:
        if (ancestor / ".env").exists():
            return ancestor
    return None


def env_candidates(start: Path | None = None) -> list[Path]:
    """Ordered ``.env`` locations, most explicit first.

    Covers a plain plugin install with no kit tree: an explicit override
    (``MULTIPLAI_ENV_FILE``), the kit home (``CLAUDE_MULTIPLAI_HOME``), the
    current working directory, and finally the marker/walk-up.
    """
    candidates: list[Path] = []
    explicit = os.environ.get("MULTIPLAI_ENV_FILE")
    if explicit:
        candidates.append(Path(explicit))
    home = os.environ.get("CLAUDE_MULTIPLAI_HOME")
    if home:
        candidates.append(Path(home) / ".env")
    candidates.append(Path.cwd() / ".env")
    root = find_project_root(start)
    if root is not None:
        candidates.append(root / ".env")
    return candidates


def load_env(start: Path | None = None) -> bool:
    """Load ``.env`` into ``os.environ`` from the first candidate that exists.

    Existing environment variables are NOT overridden — explicit env wins.
    Returns True if a file was found and loaded.
    """
    env_file = next((p for p in env_candidates(start) if p.exists()), None)
    if env_file is None:
        log.debug("No .env found in any candidate location — skipping")
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        log.warning(
            "python-dotenv not installed; cannot auto-load %s "
            "(pip install python-dotenv)", env_file,
        )
        return False
    loaded = load_dotenv(env_file, override=False)
    if loaded:
        log.info("Loaded .env from %s", env_file)
    return loaded


def load_multiplai_conf() -> dict:
    """Load ``multiplai.conf`` with optional INI-style section support.

    Returns a dict with global keys at the top level plus a ``_sections`` dict
    for per-skill overrides.
    """
    multiplai_home = os.environ.get("CLAUDE_MULTIPLAI_HOME")
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if multiplai_home:
        conf_path = Path(multiplai_home) / "multiplai.conf"
    elif config_dir:
        conf_path = Path(config_dir).parent / "multiplai.conf"
    else:
        root = find_project_root()
        conf_path = (root / "multiplai.conf") if root else None
    if conf_path is None or not conf_path.exists():
        return {"_sections": {}}

    try:
        text = conf_path.read_text()
    except OSError as e:
        # An unreadable conf (permissions, race with deletion) degrades to
        # defaults like every other loader here, instead of crashing callers.
        log.warning("Could not read %s (%s); using defaults", conf_path, e)
        return {"_sections": {}}

    result: dict[str, str] = {}
    sections: dict[str, dict[str, str]] = {}
    current_section: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Dots allowed so dotted task keys work as section names, e.g.
        # ``[deep-research.parse]`` — see pick_model()'s per-task overrides.
        section_match = re.match(r"^\[([a-zA-Z0-9_.-]+)\]\s*$", line)
        if section_match:
            current_section = section_match.group(1)
            sections.setdefault(current_section, {})  # type: ignore[arg-type]
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if current_section:
                sections[current_section][key] = value
            else:
                result[key] = value
    result["_sections"] = sections  # type: ignore[assignment]
    return result


_TIERS = {"haiku": 1, "sonnet": 2, "opus": 3}
# `xhigh` sits between high and max. An earlier copy of this table (the now-dead
# sync_skill_config.py) omitted it, which silently ranked xhigh as "unknown" and
# capped it to high — which is why this table is now exported rather than copied.
EFFORT_TIERS: Final[Mapping[str, int]] = MappingProxyType(
    {"low": 1, "medium": 2, "high": 3, "xhigh": 4, "max": 5}
)
"""Effort names ranked low → high, the ordering :func:`pick_effort` caps against.

Exported so consumers validate against *this* table instead of mirroring it. A
copy that drifts is worse than no copy: :func:`pick_effort` normalizes a name it
does not recognize away and floors to ``"high"``, so a caller that believes an
unknown name is valid loses its own "unknown → default" fallback silently.

Read-only, and additive by policy — a future release may add a tier, so treat
membership as the question and the integers as relative order only.
"""

# Private alias kept so the module's own call sites read as before.
_EFFORT_TIERS = EFFORT_TIERS

KNOWN_EFFORTS: Final[frozenset[str]] = frozenset(EFFORT_TIERS)
"""The valid effort names — ``frozenset(EFFORT_TIERS)``, for membership tests."""

# The provider a bare model string belongs to. Everything in this repo predates
# multi-vendor support, so an unqualified ID is Anthropic by definition.
DEFAULT_PROVIDER = "anthropic"


def _tier(model: str) -> int:
    model_lower = model.lower()
    for name, rank in _TIERS.items():
        if name in model_lower:
            return rank
    return 2  # default to sonnet


def resolve_model(requested: str, ceiling: str | None = None) -> str:
    """Return *requested*, or the ceiling model if requested is above it.

    Ceiling comes from *ceiling* or ``MULTIPLAI_MODEL`` (default sonnet).
    """
    if ceiling is None:
        ceiling = os.environ.get("MULTIPLAI_MODEL", "claude-sonnet-4-6")
    if _tier(requested) > _tier(ceiling):
        log.info("Model ceiling: %s → %s", requested, ceiling)
        return ceiling
    return requested


# The ONE place a dated model ID lives. Bump these 3 lines when a model ships;
# everything downstream references families (opus/sonnet/haiku), never a dated ID.
# IDs verified against the claude-api catalog (2026-07-09): opus/sonnet are bare
# aliases, haiku's alias resolves to claude-haiku-4-5-20251001.
CURRENT_MODEL = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-4-8",
}


def _normalize_tier(value: str | None) -> str | None:
    """Map a MODEL override value to a known family key, or None.

    Accepts a family name (``opus``) or a full/dated ID (``claude-opus-4-8``);
    returns the matching key in ``CURRENT_MODEL``. Unknown values → None so the
    caller can fall back to its default tier.
    """
    if not value:
        return None
    v = value.strip().lower()
    if v in CURRENT_MODEL:
        return v
    for family in CURRENT_MODEL:
        if family in v:
            return family
    return None


@dataclass(frozen=True)
class ModelSpec:
    """A model identifier plus the provider that serves it.

    The tier/ceiling machinery above only understands the Claude family, so a
    provider-qualified spec is the seam that lets a *non*-Anthropic model be
    named in config without pretending it has a haiku/sonnet/opus rank.
    """

    provider: str
    model: str

    @property
    def qualified(self) -> str:
        """``provider:model`` — the round-trippable form used in config."""
        return f"{self.provider}:{self.model}"

    @property
    def is_anthropic(self) -> bool:
        return self.provider == DEFAULT_PROVIDER


def parse_model_spec(value: str, *, default_provider: str = DEFAULT_PROVIDER) -> ModelSpec:
    """Parse ``provider:model`` (or a bare model ID) into a :class:`ModelSpec`.

    A bare ID keeps *default_provider*, so every existing config value and code
    path resolves exactly as it did before this seam existed. Splitting on the
    FIRST colon only, because some vendors put colons inside the model name
    (e.g. ``ollama:llama3:70b`` → provider ``ollama``, model ``llama3:70b``).
    """
    raw = (value or "").strip()
    if not raw:
        raise ValueError("model spec is empty")
    provider, sep, model = raw.partition(":")
    if not sep:
        return ModelSpec(default_provider, raw)
    provider, model = provider.strip().lower(), model.strip()
    if not provider or not model:
        raise ValueError(f"malformed model spec {value!r} (expected 'provider:model')")
    return ModelSpec(provider, model)


def pick_model_spec(default_tier: str = "opus", task: str | None = None) -> ModelSpec:
    """Provider-aware :func:`pick_model`.

    A ``[task] MODEL=`` value carrying a provider prefix (``openai:gpt-5``) is
    passed through verbatim: the Claude-family tier ranking and the
    ``MULTIPLAI_MODEL`` ceiling are meaningless for another vendor's model, and
    silently coercing it to a Claude tier would defeat the whole point of naming
    a cross-family model (reviewer panels want *disjoint* error sets).

    An ``anthropic:``-qualified value takes the SAME path as a bare one — family
    normalization (``anthropic:opus`` → the current opus ID) and then the
    ceiling. Resolving `spec.model` directly instead would hand back the literal
    string ``"opus"``, which is not a model ID, and would let a dated legacy ID
    (``anthropic:claude-opus-4-2``) survive un-pinned where the bare path pins
    it to CURRENT_MODEL.
    """
    conf = load_multiplai_conf()
    raw = ((conf.get("_sections", {}) or {}).get(task) or {}).get("MODEL") if task else None
    ceiling = conf.get("MULTIPLAI_MODEL")  # None → resolve_model uses env/default
    if raw and ":" in raw:
        spec = parse_model_spec(raw)
        if not spec.is_anthropic:
            log.info("Task %s uses non-Anthropic model %s", task, spec.qualified)
            return spec
        raw = spec.model  # fall through to the shared Anthropic path below
    tier = _normalize_tier(raw) or _normalize_tier(default_tier) or "opus"
    return ModelSpec(DEFAULT_PROVIDER, resolve_model(CURRENT_MODEL[tier], ceiling=ceiling))


def pick_model(default_tier: str = "opus", task: str | None = None) -> str:
    """Resolve a semantic tier to a concrete, ceiling-capped model ID.

    *default_tier* is the call site's choice (``opus`` for hard work, ``sonnet``
    for cheap bulk work). A ``[task] MODEL=...`` section in ``multiplai.conf``
    overrides it per task without a code edit. The result is then capped by the
    ``MULTIPLAI_MODEL`` ceiling, so a laptop/budget run can force all-sonnet.

    Returns the bare model ID.

    Raises:
        ValueError: the task is configured for a non-Anthropic provider. This
            call site cannot honor it, and returning the bare ID would send
            e.g. ``gpt-5`` to an Anthropic client — a 404 at request time, far
            from the config line that caused it. Failing here names the config
            key and the function to use instead.
    """
    spec = pick_model_spec(default_tier, task)
    if not spec.is_anthropic:
        raise ValueError(
            f"Task {task!r} is configured for {spec.qualified!r}, but this call "
            f"site is provider-unaware and would send {spec.model!r} to an "
            f"Anthropic client. Use pick_model_spec() + create_client_for() "
            f"here, or drop the provider prefix from [{task}] MODEL= in "
            f"multiplai.conf."
        )
    return spec.model


def _effort_tier(effort: str) -> int:
    return _EFFORT_TIERS.get(effort.lower(), 3)


def resolve_effort(requested: str, ceiling: str | None = None) -> str:
    """Return *requested*, or the ceiling effort if requested is above it."""
    if ceiling is None:
        ceiling = os.environ.get("MULTIPLAI_EFFORT", "high")
    if _effort_tier(requested) > _effort_tier(ceiling):
        log.info("Effort ceiling: %s → %s", requested, ceiling)
        return ceiling
    return requested


def _normalize_effort(value: str | None) -> str | None:
    """Map an EFFORT override to a known tier name, or None if unrecognized."""
    if not value:
        return None
    v = value.strip().lower()
    return v if v in _EFFORT_TIERS else None


def pick_effort(default_effort: str = "high", task: str | None = None) -> str:
    """Resolve a reasoning-effort tier, the second axis alongside :func:`pick_model`.

    Model and effort form a 2-axis grid — a cheap model at high effort and an
    expensive one at low effort are different points, and tuning only the model
    leaves half the grid unexplored. A ``[task] EFFORT=`` section in
    ``multiplai.conf`` overrides the call site's default, then the
    ``MULTIPLAI_EFFORT`` ceiling (env or conf) caps it, mirroring
    :func:`pick_model` exactly.
    """
    conf = load_multiplai_conf()
    section = (conf.get("_sections", {}) or {}).get(task) if task else None
    override = _normalize_effort((section or {}).get("EFFORT"))
    effort = override or _normalize_effort(default_effort) or "high"
    ceiling = conf.get("MULTIPLAI_EFFORT")  # None → resolve_effort uses env/default
    return resolve_effort(effort, ceiling=ceiling)
