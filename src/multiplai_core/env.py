"""Environment + config loading shared across the Multiplai plugins.

`.env` discovery, `multiplai.conf` parsing, and model/effort name resolution
were copied (and drifted) across buildme and deep-research; this is the single
source of truth.

Nothing here caps a model or an effort. The default for a step no config
names comes from :mod:`multiplai_core.session`.
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

log = logging.getLogger(__name__)

# ``multiplai.conf`` section header. Dots allowed so dotted task keys work as
# section names, e.g. ``[deep-research.parse]`` — see pick_model()'s per-task
# overrides.
_SECTION_RE = re.compile(r"^\[([a-zA-Z0-9_.-]+)\]\s*$")


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
            "(pip install python-dotenv)",
            env_file,
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

    result: dict[str, object] = {}
    sections: dict[str, dict[str, str]] = {}
    current_section: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        section_match = _SECTION_RE.match(line)
        if section_match:
            current_section = section_match.group(1)
            sections.setdefault(current_section, {})
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if current_section:
                sections[current_section][key] = value
            else:
                result[key] = value
    result["_sections"] = sections
    return result


def _task_setting(conf: dict, task: str | None, key: str) -> str | None:
    """The per-task ``[task] KEY=`` override from *conf*, or ``None``."""
    if not task:
        return None
    return ((conf.get("_sections", {}) or {}).get(task) or {}).get(key)


EFFORT_TIERS: Final[Mapping[str, int]] = MappingProxyType(
    {"low": 1, "medium": 2, "high": 3, "xhigh": 4, "max": 5}
)
"""Effort names ranked low → high.

Exported so consumers validate against *this* table instead of mirroring it.
Nothing caps an effort against this order any more; prefer
:data:`KNOWN_EFFORTS` for the membership test.

Read-only, and additive by policy — a future release may add a tier, so treat
membership as the question and the integers as relative order only.
"""

KNOWN_EFFORTS: Final[frozenset[str]] = frozenset(EFFORT_TIERS)
"""The valid effort names — ``frozenset(EFFORT_TIERS)``, for membership tests."""

# The provider a bare model string belongs to. Everything in this repo predates
# multi-vendor support, so an unqualified ID is Anthropic by definition.
DEFAULT_PROVIDER = "anthropic"

# Provider names that mean the same thing as DEFAULT_PROVIDER. ``claude:opus``
# reads naturally next to ``pi:<model>@<provider>`` in a config file, so
# parse_model_spec folds it into ``anthropic``.
_PROVIDER_ALIASES = {"claude": DEFAULT_PROVIDER}


def resolve_model(requested: str, ceiling: str | None = None) -> str:
    """Return *requested* unchanged.

    Deprecated: this used to cap *requested* at a model ceiling. Nothing caps
    a model any more, so *ceiling* is ignored. Kept importable until a later
    release removes it.
    """
    warnings.warn(
        "resolve_model() no longer caps anything and will be removed; "
        "use the model as is",
        DeprecationWarning,
        stacklevel=2,
    )
    return requested


# The ONE place a dated model ID lives. Bump these lines when a model ships;
# everything downstream references families (opus/sonnet/haiku/fable), never a dated ID.
# IDs verified against the claude-api catalog (2026-07-09): opus/sonnet are bare
# aliases, haiku's alias resolves to claude-haiku-4-5-20251001. fable verified
# against the same catalog (cached 2026-10-06) on 2026-10-11: Claude Fable 5.1
# is `claude-fable-5-1`. haiku/sonnet/opus moved to the 5.5 generation on
# 2026-10-31, IDs from the same catalog (cached 2026-10-06).
CURRENT_MODEL = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
    "fable": "claude-fable-5-1",
}


def expand_family(value: str) -> str:
    """Turn a family alias (``opus``) into its current ID; keep anything else.

    Only the bare names in :data:`CURRENT_MODEL` expand. Every other string,
    a dated ID included, passes through exactly as written: naming
    ``claude-opus-4-6`` gets ``claude-opus-4-6``.
    """
    v = value.strip()
    return CURRENT_MODEL.get(v.lower(), v)


@dataclass(frozen=True)
class ModelSpec:
    """A model identifier plus the provider that serves it.

    The family aliases above only cover Claude models, so a provider-qualified
    spec is the seam that lets a *non*-Anthropic model be named in config
    without pretending it is a haiku/sonnet/opus.
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


def parse_model_spec(
    value: str, *, default_provider: str = DEFAULT_PROVIDER
) -> ModelSpec:
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
    return ModelSpec(_PROVIDER_ALIASES.get(provider, provider), model)


def pick_model_spec(default_tier: str = "opus", task: str | None = None) -> ModelSpec:
    """Provider-aware :func:`pick_model`.

    A ``[task] MODEL=`` value carrying a provider prefix (``openai:gpt-5``) is
    passed through verbatim: the Claude family aliases mean nothing for
    another vendor's model.

    An ``anthropic:``-qualified value takes the SAME path as a bare one: a
    family name expands to the current ID (``anthropic:opus`` → the current
    opus ID) and any other ID is kept as written.
    """
    conf = load_multiplai_conf()
    raw = _task_setting(conf, task, "MODEL")
    if raw and ":" in raw:
        spec = parse_model_spec(raw)
        if not spec.is_anthropic:
            log.info("Task %s uses non-Anthropic model %s", task, spec.qualified)
            return spec
        raw = spec.model  # fall through to the shared Anthropic path below
    chosen = (raw or "").strip() or (default_tier or "").strip() or "opus"
    return ModelSpec(DEFAULT_PROVIDER, expand_family(chosen))


def resolve_model_spec(
    chain: Sequence[str | None], *, ceiling: str | None = None
) -> ModelSpec | None:
    """Resolve the first non-empty spec in *chain* to a :class:`ModelSpec`.

    *chain* is an ordered list of candidate values, most specific first: a
    plugin decides which config keys go into it (``models.<step>``, then
    ``models.<stage>``, …) and core only resolves the list. Returns ``None``
    when every entry is empty or ``None``, so the caller keeps its own default
    (typically :func:`multiplai_core.session.default_model`).

    An Anthropic spec (``opus``, ``claude:fable``, ``anthropic:claude-opus-4-8``)
    has its family alias expanded; any other ID is kept as written. A
    non-Anthropic spec (``pi:deepseek/deepseek-v4-flash@fireworks``) passes
    through unchanged.

    *ceiling* is deprecated and ignored: nothing caps a model any more.

    Raises:
        ValueError: the first non-empty entry is a malformed spec
            (``"pi:"``). The chain does not skip past it, because the user
            wrote that value and a silent fallback would hide the typo.
    """
    if ceiling is not None:
        warnings.warn(
            "resolve_model_spec(ceiling=...) is ignored and will be removed",
            DeprecationWarning,
            stacklevel=2,
        )
    raw = next((c.strip() for c in chain if c and c.strip()), None)
    if raw is None:
        return None
    spec = parse_model_spec(raw)
    if not spec.is_anthropic:
        return spec
    return ModelSpec(DEFAULT_PROVIDER, expand_family(spec.model))


def pick_model(default_tier: str = "opus", task: str | None = None) -> str:
    """Resolve a family alias or model ID to a concrete model ID.

    *default_tier* is the call site's choice (``opus`` for hard work, ``sonnet``
    for cheap bulk work). A ``[task] MODEL=...`` section in ``multiplai.conf``
    overrides it per task without a code edit. Nothing caps the result, and
    the session-wide default (:func:`multiplai_core.session.default_model`)
    does not change an explicit choice made here.

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


def resolve_effort(requested: str, ceiling: str | None = None) -> str:
    """Return *requested* unchanged.

    Deprecated: this used to cap *requested* at an effort ceiling. Nothing
    caps an effort any more, so *ceiling* is ignored. Kept importable until a
    later release removes it.
    """
    warnings.warn(
        "resolve_effort() no longer caps anything and will be removed; "
        "use the effort as is",
        DeprecationWarning,
        stacklevel=2,
    )
    return requested


def _normalize_effort(value: str | None) -> str | None:
    """Map an EFFORT override to a known effort name, or None if unrecognized."""
    if not value:
        return None
    v = value.strip().lower()
    return v if v in EFFORT_TIERS else None


def pick_effort(default_effort: str = "high", task: str | None = None) -> str:
    """Resolve a reasoning-effort name, the second axis alongside :func:`pick_model`.

    Model and effort form a 2-axis grid — a cheap model at high effort and an
    expensive one at low effort are different points, and tuning only the model
    leaves half the grid unexplored. A ``[task] EFFORT=`` section in
    ``multiplai.conf`` overrides the call site's default. Nothing caps the
    result, and the session-wide default
    (:func:`multiplai_core.session.default_effort`) does not change an
    explicit choice made here.
    """
    conf = load_multiplai_conf()
    override = _normalize_effort(_task_setting(conf, task, "EFFORT"))
    return override or _normalize_effort(default_effort) or "high"
