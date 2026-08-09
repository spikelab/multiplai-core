"""Path resolver for multiplai plugin.

Resolves file locations from plugin environment variables with standalone
fallbacks.  Paths are cached at first access and immutable for the process
lifetime (frozen dataclass + module-level singleton).

Resolution order:
    1. Plugin env var (``CLAUDE_PLUGIN_ROOT``, ``CLAUDE_PLUGIN_DATA``,
       ``CLAUDE_PLUGIN_OPTION_*``) — expanded and resolved to absolute.
    2. Workspace-scoped fallback rooted at ``$WORKSPACE/.multiplai/`` (or
       the ``workspace_dir`` plugin option's ``.multiplai/``).
    3. The nearest ancestor ``.multiplai/`` marker directory, walking up
       from an absolute ``$CLAUDE_PROJECT_DIR``. **Never from the cwd**, and
       never ``$HOME`` itself.
    4. Hardcoded standalone fallback rooted at ``~/.multiplai/``.

``data_dir`` ranks ``CLAUDE_PLUGIN_DATA`` *above* step 3 — see
:meth:`Paths.resolve` for why that one exception exists.

``memory_dir`` is additionally the first of an ordered list of **memory
banks** — see :mod:`multiplai_core.banks` and :meth:`Paths.memory_banks`.
With no bank configuration the list is exactly that one directory, so every
consumer of ``memory_dir`` is unaffected.
"""

import dataclasses
import os
import threading
from pathlib import Path

from .banks import BANKS_FILENAME, MemoryBank, load_banks
from .plugin_options import option


_lock = threading.Lock()
_cached_paths: "Paths | None" = None

# Standalone base — used only when no workspace or plugin env vars are set.
_STANDALONE_BASE = Path.home() / ".multiplai"


# How far up the tree the marker search walks before giving up. Bounded so a
# hook started deep inside a monorepo cannot spend its budget stat-ing its way
# to ``/`` on a cold network filesystem.
_MARKER_MAX_DEPTH = 12


def _discovered_workspace_base() -> Path | None:
    """Nearest ancestor ``.multiplai/`` of ``$CLAUDE_PROJECT_DIR``, or None.

    Walks up from ``$CLAUDE_PROJECT_DIR`` — the harness's own notion of where
    the session is rooted — and returns the first ``.multiplai/`` directory
    found.

    This exists to break the coupling that made the workspace knowable only
    because the container launcher exported ``WORKSPACE``: a plugin installed
    on a plain Claude Code with no launcher had no way to find the workspace
    it was plainly sitting inside, and silently wrote to ``~/.multiplai``
    instead. The marker is the same directory the data already lives in, so
    discovery cannot point somewhere that is not already a workspace.

    **The start point is never the cwd.** Claude routinely shifts cwd into
    sub-projects that all belong to one workspace (the reason
    :func:`_workspace_base` refuses cwd as a fallback), and a cwd-rooted walk
    would additionally make resolution depend on where a test or a script
    happened to be run from. No ``CLAUDE_PROJECT_DIR`` means no discovery —
    and a *relative* ``CLAUDE_PROJECT_DIR`` means no discovery either, because
    ``Path(".").resolve()`` re-introduces the cwd through the back door and
    would resolve, during development, against whatever directory a test
    happened to run in.

    ``$HOME`` itself does not satisfy the marker test. ``~/.multiplai`` is the
    standalone fallback layout, not a discovered workspace: treating it as one
    would silently relocate ``data_dir`` for every plain install whose session
    happens to be rooted under the home directory, which is the common case
    rather than an edge case.

    Deliberately ranked *below* both explicit signals and above the
    standalone fallback: an explicit ``workspace_dir`` or ``WORKSPACE`` still
    wins. It is also ranked *below* ``CLAUDE_PLUGIN_DATA`` for ``data_dir``
    specifically — see :meth:`Paths.resolve` — so an install with a managed
    data dir keeps it.
    """
    start = _env("CLAUDE_PROJECT_DIR")
    if not start:
        return None
    if not Path(start).expanduser().is_absolute():
        return None
    try:
        current = Path(start).expanduser().resolve()
    except (OSError, ValueError):
        return None
    home = Path.home()
    for _ in range(_MARKER_MAX_DEPTH):
        if current == home:
            break
        marker = current / ".multiplai"
        try:
            if marker.is_dir():
                return marker
        except OSError:
            return None
        if current == current.parent:
            break
        current = current.parent
    return None


def _configured_workspace_base() -> Path | None:
    """Workspace ``.multiplai/`` root from an **explicit** setting, else None.

    Resolution:
      1. The ``workspace_dir`` plugin option if set.
      2. ``WORKSPACE`` env var (set by the container launcher) — lets
         scripts invoked outside the plugin hook mechanism resolve
         workspace paths correctly.

    Kept separate from :func:`_discovered_workspace_base` because the two rank
    differently against ``CLAUDE_PLUGIN_DATA``: somebody *said* where the
    workspace is, versus we *found* something that looks like one.
    """
    env = option("workspace_dir")
    if env:
        return Path(env).expanduser().resolve() / ".multiplai"
    workspace = _env("WORKSPACE")
    if workspace:
        return Path(workspace).expanduser().resolve() / ".multiplai"
    return None


def _explicit_workspace_base() -> Path | None:
    """Workspace ``.multiplai/`` root *if locatable at all*, else None.

    :func:`_configured_workspace_base` first, then the nearest ancestor
    ``.multiplai/`` marker directory (:func:`_discovered_workspace_base`).

    Returns ``None`` when none of the three answer, so callers can
    distinguish a located workspace from the pure-standalone fallback.
    """
    return _configured_workspace_base() or _discovered_workspace_base()


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
    ``{diary,now,learnings}_dir`` plugin option.
    """
    return _explicit_workspace_base() or _STANDALONE_BASE


class CallablePath(type(Path())):
    """A ``Path`` subclass whose instances are callable (returning *self*).

    :class:`Paths` fields hold ``CallablePath`` instances so that both
    attribute access (``p.plugin_root``) and method-call syntax
    (``p.plugin_root()``) work identically.  This keeps the public API
    uniform — callers can always use ``()`` regardless of whether the
    accessor is a dataclass field or a derived-path method — and the fields
    are annotated with this type so both spellings pass type checking.
    """

    def __call__(self) -> Path:
        return self


def _callable(p: Path) -> CallablePath:
    """Wrap *p* as a ``CallablePath`` so it can be called with ``()``."""
    return CallablePath(p)


def _env(name: str) -> str:
    """Read an environment variable, treating empty/whitespace as unset."""
    return os.environ.get(name, "").strip()


def _ensure_data_gitignore(data_dir: Path) -> None:
    """Drop a ``.gitignore`` of ``*`` at *data_dir* if none exists.

    Makes the whole runtime data bucket git-ignored by mechanism, so a
    skill writing state under a workspace repo can never stage it — no
    reliance on a workspace-level ignore rule. Best-effort; never raises.
    """
    gi = data_dir / ".gitignore"
    try:
        if not gi.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            gi.write_text("*\n", encoding="utf-8")
    except OSError:
        pass


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
    Fields are :class:`CallablePath` instances — they behave as regular
    ``Path`` objects but are also callable (returning themselves) so that the
    ``paths.field()`` accessor pattern works uniformly.
    """

    plugin_root: CallablePath
    data_dir: CallablePath
    memory_dir: CallablePath
    diary_dir: CallablePath
    now_dir: CallablePath
    learnings_dir: CallablePath
    venv_dir: CallablePath
    catalogs_dir: CallablePath
    templates_dir: CallablePath
    _is_plugin_mode: bool = dataclasses.field(default=False, repr=False)

    @classmethod
    def resolve(cls) -> "Paths":
        """Resolve all paths from environment variables, with fallbacks.

        Each path category follows an env-var-first, fallback-second cascade.
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
        #   1. the `data_dir` option           (explicit override)
        #   2. <configured workspace>/.multiplai/data
        #   3. CLAUDE_PLUGIN_DATA              (managed dir; no said workspace)
        #   4. <discovered workspace>/.multiplai/data
        #   5. ~/.multiplai/data               (pure standalone)
        #
        # Steps 3 and 4 are in that order deliberately, and it is the one
        # ranking in this function that is not simply "most explicit first".
        # A *discovered* workspace is an inference from a marker directory; a
        # managed data dir is a fact about the install. Ranking discovery
        # above it would move data_dir — and with it venv_dir, catalogs_dir,
        # logs and dream state — for every plugin install that has a managed
        # data dir and no WORKSPACE, orphaning an already-bootstrapped venv
        # and catalog set. Discovery exists to rescue the case that fell
        # through to ~/.multiplai, and that case is step 5, not step 3.
        configured_ws = _configured_workspace_base()
        discovered_ws = _discovered_workspace_base()
        opt_data = option("data_dir")
        if opt_data:
            data_dir = Path(opt_data).expanduser().resolve()
        elif configured_ws is not None:
            data_dir = configured_ws / "data"
        elif env_data:
            data_dir = Path(env_data).expanduser().resolve()
        elif discovered_ws is not None:
            data_dir = discovered_ws / "data"
        else:
            data_dir = _STANDALONE_BASE / "data"

        # User-data dirs share the same workspace fallback hierarchy:
        #   1. the `<name>_dir` option        (specific override)
        #   2. the `workspace_dir` option's .multiplai/<name>
        #   3. $WORKSPACE/.multiplai/<name>
        #   4. ~/.multiplai/<name> (pure standalone)
        memory_dir = _resolve_env_path(
            option("memory_dir"),
            workspace_base / "memory",
        )
        diary_dir = _resolve_env_path(
            option("diary_dir"),
            workspace_base / "diary",
        )
        now_dir = _resolve_env_path(
            option("now_dir"),
            diary_dir.parent / "now",
        )
        learnings_dir = _resolve_env_path(
            option("learnings_dir"),
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

    def skill_state_dir(self, name: str) -> Path:
        """Git-ignored per-skill state bucket at ``data_dir/skills/<name>``.

        One home for a skill's regenerable runtime state (SQLite caches,
        downloaded assets, token files) — replacing the hand-rolled
        WORKSPACE→``~/.multiplai`` resolvers each skill used to copy. The
        directory is created on first access and, at the same time, a
        ``.gitignore`` containing ``*`` is dropped at the ``data_dir`` root
        so *everything* under the data bucket is git-ignored by mechanism,
        not by a claim in docs or a workspace-level rule that a standalone
        checkout might lack.

        Best-effort: directory creation and the ignore-file write never
        raise (a read-only or racing filesystem must not break the skill);
        the resolved path is always returned so callers can proceed.
        """
        d = self.data_dir / "skills" / name
        try:
            d.mkdir(parents=True, exist_ok=True)
            _ensure_data_gitignore(self.data_dir)
        except OSError:
            pass
        return d

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

    def memory_banks_file(self) -> Path:
        """Bank declarations (YAML) at the workspace ``.multiplai/`` root.

        Sits beside ``project-map.yaml`` for the same reason: it describes the
        *workspace*, not the runtime, so it belongs in the tracked tree rather
        than the git-ignored data bucket. Optional — absent means one bank.
        Override with the ``memory_banks_file`` plugin option.
        """
        override = option("memory_banks_file")
        if override:
            return Path(override).expanduser().resolve()
        return self.diary_dir.parent / BANKS_FILENAME

    def memory_banks(self) -> tuple[MemoryBank, ...]:
        """The ordered memory banks, ``personal`` first and always present.

        With no ``memory-banks.yaml`` this is exactly one bank at
        :attr:`memory_dir` — the pre-banks world, unchanged. Resolved on each
        call rather than cached on the frozen instance so that subscribing to
        a bank takes effect on the next hook run rather than the next
        process; the file is a few lines and the read is not on a hot path.
        """
        return load_banks(
            memory_dir=self.memory_dir, config_path=self.memory_banks_file()
        )

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


# Module-level convenience accessor. Resolved lazily via PEP 562 so that
# merely importing the package does not read env vars and permanently cache
# the result — a consumer that sets CLAUDE_PLUGIN_OPTION_* / WORKSPACE after
# `import multiplai_core` but before first path use still gets correct paths.
# `from multiplai_core.paths import paths` continues to work unchanged.
def __getattr__(name: str) -> "Paths":
    if name == "paths":
        return get_paths()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
