"""Path resolver for multiplai plugin.

Resolves file locations from plugin environment variables with standalone
fallbacks.  Paths are cached at first access and immutable for the process
lifetime (frozen dataclass + module-level singleton).

Resolution order per D2 spec:
    1. Plugin env var (``CLAUDE_PLUGIN_ROOT``, ``CLAUDE_PLUGIN_DATA``,
       ``CLAUDE_PLUGIN_OPTION_*``) — expanded and resolved to absolute.
    2. Workspace-scoped fallback rooted at ``$WORKSPACE/.multiplai/`` (or
       ``CLAUDE_PLUGIN_OPTION_workspace_dir/.multiplai/``).
    3. Hardcoded standalone fallback rooted at ``~/.multiplai/``.
"""

import dataclasses
import os
import threading
from pathlib import Path


_lock = threading.Lock()
_cached_paths: "Paths | None" = None

# Standalone base — used only when no workspace or plugin env vars are set.
_STANDALONE_BASE = Path.home() / ".multiplai"


def _explicit_workspace_base() -> Path | None:
    """Workspace ``.multiplai/`` root *if explicitly configured*, else None.

    Resolution:
      1. ``CLAUDE_PLUGIN_OPTION_workspace_dir`` if set.
      2. ``WORKSPACE`` env var (set by the container launcher) — lets
         scripts invoked outside the plugin hook mechanism resolve
         workspace paths correctly.

    Returns ``None`` when neither is set, so callers can distinguish a
    configured workspace from the pure-standalone fallback.
    """
    env = _env("CLAUDE_PLUGIN_OPTION_workspace_dir")
    if env:
        return Path(env).expanduser().resolve() / ".multiplai"
    workspace = _env("WORKSPACE")
    if workspace:
        return Path(workspace).expanduser().resolve() / ".multiplai"
    return None


def _workspace_base() -> Path:
    """Workspace-scoped ``.multiplai/`` root.

    Diary, learnings, and per-project ``now`` files are workspace
    data: there should be one ``.multiplai/`` per workspace, not one
    per ``cwd`` (Claude routinely shifts ``cwd`` into sub-projects
    that all belong to the same workspace).

    Falls back to ``~/.multiplai/`` when no workspace is configured so
    a fresh install still writes somewhere sensible. We deliberately do
    NOT use ``cwd`` as a fallback — it would pollute every sub-project
    with its own data tree.

    Override any individual directory via the matching
    ``CLAUDE_PLUGIN_OPTION_{diary,now,learnings}_dir`` env var.
    """
    return _explicit_workspace_base() or _STANDALONE_BASE


class _CallablePath(type(Path())):
    """A ``Path`` subclass whose instances are callable (returning *self*).

    Dataclass fields store ``_CallablePath`` instances so that both attribute
    access (``p.plugin_root``) and method-call syntax (``p.plugin_root()``)
    work identically.  This keeps the public API uniform — callers can always
    use ``()`` regardless of whether the accessor is a dataclass field or a
    derived-path method.
    """

    def __call__(self) -> Path:
        return self


def _callable(p: Path) -> _CallablePath:
    """Wrap *p* as a ``_CallablePath`` so it can be called with ``()``."""
    return _CallablePath(p)


def _env(name: str) -> str:
    """Read an environment variable, treating empty/whitespace as unset."""
    return os.environ.get(name, "").strip()


def _resolve_env_path(value: str, fallback: Path) -> Path:
    """Return an absolute ``Path`` from *value*, or *fallback* if empty.

    Non-empty values are tilde-expanded and resolved to absolute form.
    """
    if value:
        return Path(value).expanduser().resolve()
    return fallback


@dataclasses.dataclass(frozen=True)
class Paths:
    """Immutable container of resolved plugin paths.

    Use :meth:`resolve` to create an instance from the current environment.
    Fields are ``_CallablePath`` instances — they behave as regular ``Path``
    objects but are also callable (returning themselves) so that the
    ``paths.field()`` accessor pattern works uniformly.
    """

    plugin_root: Path
    data_dir: Path
    memory_dir: Path
    diary_dir: Path
    now_dir: Path
    learnings_dir: Path
    venv_dir: Path
    catalogs_dir: Path
    templates_dir: Path
    _is_plugin_mode: bool = dataclasses.field(default=False, repr=False)

    @classmethod
    def resolve(cls) -> "Paths":
        """Resolve all paths from environment variables, with fallbacks.

        Each path category follows an env-var-first, fallback-second cascade
        per the D2 design table.
        """
        env_root = _env("CLAUDE_PLUGIN_ROOT")
        env_data = _env("CLAUDE_PLUGIN_DATA")

        is_plugin = bool(env_root)

        plugin_root = _resolve_env_path(env_root, _STANDALONE_BASE)
        workspace_base = _workspace_base()

        # Data dir holds runtime state: logs, catalogs, venv, dream state.
        # Whenever a workspace is explicitly configured it stays inside
        # <workspace>/.multiplai/data — beside memory/diary/learnings —
        # NOT in the per-install dir Claude Code points CLAUDE_PLUGIN_DATA
        # at (anchoring there split runtime state away from the workspace).
        # CLAUDE_PLUGIN_DATA is kept only as a managed fallback for installs
        # with no configured workspace. Resolution:
        #   1. CLAUDE_PLUGIN_OPTION_data_dir   (explicit override)
        #   2. <explicit workspace>/.multiplai/data
        #   3. CLAUDE_PLUGIN_DATA              (managed dir; no workspace)
        #   4. ~/.multiplai/data              (pure standalone)
        explicit_ws = _explicit_workspace_base()
        opt_data = _env("CLAUDE_PLUGIN_OPTION_data_dir")
        if opt_data:
            data_dir = Path(opt_data).expanduser().resolve()
        elif explicit_ws is not None:
            data_dir = explicit_ws / "data"
        elif env_data:
            data_dir = Path(env_data).expanduser().resolve()
        else:
            data_dir = _STANDALONE_BASE / "data"

        # User-data dirs share the same workspace fallback hierarchy:
        #   1. CLAUDE_PLUGIN_OPTION_<name>_dir (specific override)
        #   2. CLAUDE_PLUGIN_OPTION_workspace_dir/.multiplai/<name>
        #   3. $WORKSPACE/.multiplai/<name>
        #   4. ~/.multiplai/<name> (pure standalone)
        memory_dir = _resolve_env_path(
            _env("CLAUDE_PLUGIN_OPTION_memory_dir"),
            workspace_base / "memory",
        )
        diary_dir = _resolve_env_path(
            _env("CLAUDE_PLUGIN_OPTION_diary_dir"),
            workspace_base / "diary",
        )
        now_dir = _resolve_env_path(
            _env("CLAUDE_PLUGIN_OPTION_now_dir"),
            diary_dir.parent / "now",
        )
        learnings_dir = _resolve_env_path(
            _env("CLAUDE_PLUGIN_OPTION_learnings_dir"),
            diary_dir.parent / "learnings",
        )

        return cls(
            plugin_root=_callable(plugin_root),
            data_dir=_callable(data_dir),
            memory_dir=_callable(memory_dir),
            diary_dir=_callable(diary_dir),
            now_dir=_callable(now_dir),
            learnings_dir=_callable(learnings_dir),
            venv_dir=_callable(data_dir / "venv"),
            catalogs_dir=_callable(data_dir / "catalogs"),
            templates_dir=_callable(plugin_root / "templates"),
            _is_plugin_mode=is_plugin,
        )

    # ------------------------------------------------------------------
    # Method-style accessors (backward compatibility)
    # ------------------------------------------------------------------

    def plugin_data(self) -> Path:
        """Runtime data directory for venv, logs, catalogs, and state files."""
        return self.data_dir

    def is_plugin_mode(self) -> bool:
        """Whether paths were resolved from plugin environment variables."""
        return self._is_plugin_mode

    # ------------------------------------------------------------------
    # Derived path accessors
    # ------------------------------------------------------------------

    def logs_dir(self) -> Path:
        """Plugin log files directory."""
        return self.data_dir / "logs"

    def dream_state_file(self) -> Path:
        """Dream state tracking file (YAML)."""
        return self.data_dir / "dream_state.yaml"

    def project_map_file(self) -> Path:
        """Project-identity config (YAML) at the workspace ``.multiplai/`` root.

        Sits beside ``memory/``, ``diary/``, ``now/`` (``diary_dir.parent`` is
        the workspace base). Read by ``lib.project_identity`` to map a session
        ``cwd`` onto a stable project name. Optional — absent means defaults.
        """
        return self.diary_dir.parent / "project-map.yaml"

    def learnings_file(self, date_str: str | None = None) -> Path:
        """Per-day structured learnings file ``learnings_dir/{YYYY-MM-DD}.md``.

        When *date_str* is omitted, returns today's file (UTC). Per-day
        naming lets downstream tooling (``/multiplai-context:dream-remember``)
        read the full learnings backlog without changes.
        """
        from datetime import datetime, timezone
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.learnings_dir / f"{date_str}.md"

    def dreams_dir(self) -> Path:
        """Dreams directory for pending Dream proposals awaiting review."""
        return self.data_dir.parent / "dreams"

    def scripts_dir(self) -> Path:
        """Hook and utility scripts directory."""
        return self.plugin_root / "scripts"


def get_paths() -> Paths:
    """Return the cached Paths singleton. Thread-safe, resolved once."""
    global _cached_paths
    if _cached_paths is not None:
        return _cached_paths
    with _lock:
        if _cached_paths is not None:
            return _cached_paths
        _cached_paths = Paths.resolve()
        return _cached_paths


def _reset_cache() -> None:
    """Reset the cached paths. For testing only."""
    global _cached_paths
    with _lock:
        _cached_paths = None


# Module-level convenience accessors
paths = get_paths()
