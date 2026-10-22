"""The model and effort a step uses when no config names one.

A skill started from a Claude Code session runs its steps through
:func:`multiplai_core.run_agent`, which starts a fresh ``claude`` CLI child.
That child does not inherit the session's model: it reads no settings
(``setting_sources=[]``), so without ``--model`` it runs on the CLI's own
default. This module finds the values a step should use instead, in order:

1. ``MULTIPLAI_MODEL`` / ``MULTIPLAI_EFFORT`` from the environment, then from
   ``multiplai.conf``. A family alias (``opus``) expands to its current ID;
   an empty value counts as unset.
2. The session's own values: ``MULTIPLAI_SESSION_MODEL`` /
   ``MULTIPLAI_SESSION_EFFORT`` when a parent process pinned them
   (:func:`pin_session_settings`), else the last model the session transcript
   recorded and Claude Code's ``CLAUDE_EFFORT``.

Model and effort are resolved separately: ``MULTIPLAI_MODEL`` set with
``MULTIPLAI_EFFORT`` unset gives that model at the session's effort.

Explicit choices (a step's config, a background job's own default) never come
through here, so nothing in this module caps or changes them.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path

from .env import KNOWN_EFFORTS, expand_family, load_multiplai_conf

log = logging.getLogger(__name__)

SESSION_MODEL_ENV = "MULTIPLAI_SESSION_MODEL"
SESSION_EFFORT_ENV = "MULTIPLAI_SESSION_EFFORT"

# Bytes read per step when scanning the transcript backwards. The last
# assistant turn is almost always in the final chunk; a long tool result after
# it can push it further back, and the loop keeps reading until it is found.
_CHUNK = 64 * 1024

# (what, value, source) triples already logged in this process, so each result
# is logged once however many steps ask for it.
_logged: set[tuple[str, str | None, str]] = set()
_warned_no_model = False


def _log_once(what: str, value: str | None, source: str) -> None:
    key = (what, value, source)
    if key in _logged:
        return
    _logged.add(key)
    log.info(
        "Default %s for steps with none configured: %s (from %s)", what, value, source
    )


def _transcript_paths(session_id: str) -> list[Path]:
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR") or str(Path.home() / ".claude")
    # Glob over the project directory: Claude Code encodes the cwd into that
    # directory name, and matching by session id avoids re-deriving it.
    pattern = os.path.join(
        glob.escape(config_dir), "projects", "*", f"{glob.escape(session_id)}.jsonl"
    )
    return [Path(p) for p in glob.glob(pattern)]


def _model_in_line(line: bytes) -> str | None:
    """The model of an assistant turn on this transcript line, or None."""
    if b'"model"' not in line:
        return None
    try:
        entry = json.loads(line)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(entry, dict) or entry.get("type") != "assistant":
        return None
    if entry.get("isSidechain"):
        return None
    message = entry.get("message")
    model = message.get("model") if isinstance(message, dict) else None
    if not isinstance(model, str) or not model or model == "<synthetic>":
        return None
    return model


def _last_model_in(path: Path) -> str | None:
    """The last assistant model recorded in *path*, reading it backwards."""
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            pos = fh.tell()
            tail = b""
            while pos > 0:
                step = min(_CHUNK, pos)
                pos -= step
                fh.seek(pos)
                buf = fh.read(step) + tail
                lines = buf.split(b"\n")
                # The first piece may be a partial line unless we reached the
                # start of the file; keep it for the next round.
                tail = lines[0] if pos > 0 else b""
                complete = lines[1:] if pos > 0 else lines
                for line in reversed(complete):
                    model = _model_in_line(line.strip())
                    if model:
                        return model
            return None
    except OSError:
        return None


def session_model() -> str | None:
    """The model of the Claude Code session that started this process.

    ``MULTIPLAI_SESSION_MODEL`` when set (a parent pinned it). Otherwise the
    last assistant ``model`` that is not ``<synthetic>`` in the session
    transcript ``$CLAUDE_CONFIG_DIR/projects/*/<CLAUDE_CODE_SESSION_ID>.jsonl``.
    None when neither exists.
    """
    pinned = (os.environ.get(SESSION_MODEL_ENV) or "").strip()
    if pinned:
        return pinned
    session_id = (os.environ.get("CLAUDE_CODE_SESSION_ID") or "").strip()
    if not session_id:
        return None
    for path in _transcript_paths(session_id):
        model = _last_model_in(path)
        if model:
            return model
    return None


def session_effort() -> str | None:
    """The effort of the Claude Code session that started this process.

    ``MULTIPLAI_SESSION_EFFORT`` when set, else ``CLAUDE_EFFORT`` (which Claude
    Code exports to the commands it runs) when it names a known effort, else
    None.
    """
    pinned = (os.environ.get(SESSION_EFFORT_ENV) or "").strip().lower()
    if pinned in KNOWN_EFFORTS:
        return pinned
    effort = (os.environ.get("CLAUDE_EFFORT") or "").strip().lower()
    return effort if effort in KNOWN_EFFORTS else None


def pin_session_settings() -> tuple[str | None, str | None]:
    """Resolve the session's model and effort once and put them in the environment.

    Writes ``MULTIPLAI_SESSION_MODEL`` / ``MULTIPLAI_SESSION_EFFORT`` into
    ``os.environ``, so every child process this run starts sees the values the
    run started with, even if the session switches model meanwhile. A value
    that cannot be found is left unset. Returns ``(model, effort)``.
    """
    model = session_model()
    effort = session_effort()
    if model:
        os.environ[SESSION_MODEL_ENV] = model
    if effort:
        os.environ[SESSION_EFFORT_ENV] = effort
    return model, effort


def _configured(name: str) -> tuple[str | None, str]:
    """``(value, source)`` for *name* from the environment, then ``multiplai.conf``."""
    value = (os.environ.get(name) or "").strip()
    if value:
        return value, name
    conf_value = str(load_multiplai_conf().get(name) or "").strip()
    if conf_value:
        return conf_value, "multiplai.conf"
    return None, ""


def default_model_source() -> tuple[str | None, str]:
    """``(model, source)`` for :func:`default_model`.

    *source* names where the model came from: ``MULTIPLAI_MODEL``,
    ``multiplai.conf``, ``MULTIPLAI_SESSION_MODEL``, ``session transcript``,
    or ``""`` when no model was found.
    """
    value, source = _configured("MULTIPLAI_MODEL")
    if value:
        return expand_family(value), source
    if (os.environ.get(SESSION_MODEL_ENV) or "").strip():
        return session_model(), SESSION_MODEL_ENV
    model = session_model()
    return model, ("session transcript" if model else "")


def default_effort_source() -> tuple[str | None, str]:
    """``(effort, source)`` for :func:`default_effort`.

    *source* is ``MULTIPLAI_EFFORT``, ``multiplai.conf``,
    ``MULTIPLAI_SESSION_EFFORT``, ``CLAUDE_EFFORT``, or ``""``.
    """
    value, source = _configured("MULTIPLAI_EFFORT")
    if value and value.lower() in KNOWN_EFFORTS:
        return value.lower(), source
    if value:
        log.warning(
            "Ignoring %s=%r: not one of %s",
            source,
            value,
            ", ".join(sorted(KNOWN_EFFORTS)),
        )
    pinned = (os.environ.get(SESSION_EFFORT_ENV) or "").strip().lower()
    if pinned in KNOWN_EFFORTS:
        return pinned, SESSION_EFFORT_ENV
    effort = session_effort()
    return effort, ("CLAUDE_EFFORT" if effort else "")


def default_model() -> str | None:
    """The model for a step no config names.

    ``MULTIPLAI_MODEL`` (environment, then ``multiplai.conf``), else the
    session's model (:func:`session_model`), else None — in which case the
    caller omits ``--model`` and the CLI uses its own default.
    """
    global _warned_no_model
    model, source = default_model_source()
    if model:
        _log_once("model", model, source)
    elif not _warned_no_model:
        _warned_no_model = True
        log.warning(
            "No default model found (MULTIPLAI_MODEL unset and no Claude Code "
            "session transcript); steps without a configured model will use "
            "the CLI's default."
        )
    return model


def default_effort() -> str | None:
    """The effort for a step no config names.

    ``MULTIPLAI_EFFORT`` (environment, then ``multiplai.conf``) when it names
    a known effort, else the session's effort (:func:`session_effort`), else
    None.
    """
    effort, source = default_effort_source()
    if effort:
        _log_once("effort", effort, source)
    return effort
