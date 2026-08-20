"""The package's public surface, and the three lists that have to agree on it.

`__all__`, the `TYPE_CHECKING` import block and `_LAZY_ATTRS` each spell out
the exported names. A type checker reads the first two; nothing read the third,
so an export added to `__all__` and forgotten in `_LAZY_ATTRS` raised
`ImportError` on `from multiplai_core import <name>` while every test in the
suite still passed. These tests are what reads it.
"""

import subprocess
import sys

import pytest

import multiplai_core
from multiplai_core import _LAZY_ATTRS
from multiplai_core import __all__ as PUBLIC_NAMES


def _run(code: str) -> subprocess.CompletedProcess:
    """Run *code* in a fresh interpreter — import side effects need one."""
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


@pytest.mark.parametrize("name", sorted(PUBLIC_NAMES))
def test_every_exported_name_resolves(name):
    """Each name in `__all__` is reachable as an attribute.

    This is the assertion that fails when `_LAZY_ATTRS` loses an entry: the
    eager path binds nothing for a deferred name, so `__getattr__` is the only
    thing that can serve it, and it serves only what the map knows.
    """
    assert getattr(multiplai_core, name) is not None


@pytest.mark.parametrize("name", sorted(PUBLIC_NAMES))
def test_every_exported_name_is_importable(name):
    """...and reachable through `from multiplai_core import <name>`, which is
    how consumers spell it and the form that raises ImportError."""
    result = _run(f"from multiplai_core import {name}")
    assert result.returncode == 0, result.stderr


def test_lazy_map_holds_no_names_the_package_does_not_export():
    """Every key is either an exported name or one of the submodules the eager
    imports used to bind. A key that is neither silently widens the package's
    attribute surface."""
    submodules = set(_LAZY_ATTRS.values())
    unexplained = {
        name
        for name in _LAZY_ATTRS
        if name not in PUBLIC_NAMES and name not in submodules
    }
    assert not unexplained


@pytest.mark.parametrize("submodule", sorted(set(_LAZY_ATTRS.values())))
def test_submodule_attribute_survives_a_bare_import(submodule):
    """`import multiplai_core` then `multiplai_core.agent_runner` worked before
    the submodules went lazy, because the eager `from .agent_runner import ...`
    bound it. Deferring the import must not take that away."""
    result = _run(
        f"import multiplai_core; assert multiplai_core.{submodule}.__name__ "
        f"== 'multiplai_core.{submodule}'"
    )
    assert result.returncode == 0, result.stderr


def test_bare_import_does_not_pull_in_asyncio():
    """The point of the lazy submodules. A hook that only wants `get_paths()`
    or `option()` should not pay for the asyncio-heavy import chain -- and the
    only way that stays true is if something checks."""
    result = _run(
        "import sys, multiplai_core; "
        "assert 'asyncio' not in sys.modules, sorted(sys.modules)"
    )
    assert result.returncode == 0, result.stderr


def test_unknown_attribute_still_raises_attribute_error():
    """`__getattr__` must not turn a typo into an import attempt."""
    with pytest.raises(AttributeError):
        multiplai_core.no_such_name


def test_dir_lists_the_public_names():
    listed = set(dir(multiplai_core))
    assert set(PUBLIC_NAMES) <= listed
    assert set(_LAZY_ATTRS) <= listed
