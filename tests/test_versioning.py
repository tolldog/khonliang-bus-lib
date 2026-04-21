"""Tests for ``khonliang_bus.versioning`` — resolution chain + cache."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from khonliang_bus import versioning


# ---------------------------------------------------------------------------
# Metadata fallback path (behavior carried over from PR #11 + #12 resolver)
# ---------------------------------------------------------------------------


def test_missing_distribution_returns_none(monkeypatch):
    """Helper returns None for modules with no installed distribution."""
    import importlib.metadata as md

    monkeypatch.setattr(md, "packages_distributions", lambda: {})
    assert versioning.resolve_version("nope.mod") is None
    assert versioning.resolve_version("") is None


def test_dash_m_recovery_via_main_spec(monkeypatch, tmp_path):
    """``module_name == '__main__'`` consults ``sys.modules['__main__'].__spec__.name``."""
    import importlib.metadata as md

    fake_spec = types.SimpleNamespace(name="reviewer.agent")
    # fake main module needs no __file__ so pyproject walk falls through to metadata
    fake_main = types.SimpleNamespace(__spec__=fake_spec)
    monkeypatch.setitem(sys.modules, "__main__", fake_main)
    # Same for "reviewer.agent" — no __file__ forces metadata path
    monkeypatch.setitem(
        sys.modules, "reviewer.agent", types.SimpleNamespace(__file__=None)
    )

    monkeypatch.setattr(
        md, "packages_distributions", lambda: {"reviewer": ["khonliang-reviewer"]}
    )
    monkeypatch.setattr(md, "version", lambda name: "0.5.0")

    assert versioning.resolve_version("__main__") == "0.5.0"


def test_main_without_spec_returns_none(monkeypatch):
    """REPL / script launch leaves ``__main__`` without a useful ``__spec__``."""
    monkeypatch.setitem(
        sys.modules, "__main__", types.SimpleNamespace(__spec__=None)
    )
    assert versioning.resolve_version("__main__") is None


# ---------------------------------------------------------------------------
# pyproject.toml walk-up (new in FR-A1)
# ---------------------------------------------------------------------------


def _make_fake_pkg(tmp_path: Path, *, package_name: str, pyproject_version: str | None, add_git: bool = True) -> Path:
    """Scaffold a fake repo tree with optional pyproject.toml + .git/ sentinel.

    Returns the path to ``<package>/agent.py`` inside the repo.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    if add_git:
        (repo / ".git").mkdir()
    if pyproject_version is not None:
        (repo / "pyproject.toml").write_text(
            f'[project]\nname = "{package_name}"\nversion = "{pyproject_version}"\n'
        )
    pkg = repo / package_name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    agent = pkg / "agent.py"
    agent.write_text("")
    return agent


def test_pyproject_walk_reads_current_version(monkeypatch, tmp_path):
    """Reader walks up from the module's source file and reads `project.version`."""
    agent_path = _make_fake_pkg(tmp_path, package_name="fakeagent", pyproject_version="2.7.1")
    fake_module = types.SimpleNamespace(__file__=str(agent_path))
    monkeypatch.setitem(sys.modules, "fakeagent.agent", fake_module)

    assert versioning.resolve_version("fakeagent.agent") == "2.7.1"


def test_pyproject_walk_stops_at_git_sentinel(monkeypatch, tmp_path):
    """Walk terminates at ``.git/`` and does not escape into a parent repo.

    Construct nested fake repos: outer has pyproject v9.9.9, inner has
    its own ``.git/`` but no pyproject. The module lives in the inner
    repo — the resolver must return ``None`` (inner has no pyproject;
    ``.git/`` sentinel prevents falling through to outer) rather than
    silently reporting the outer's version.
    """
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / "pyproject.toml").write_text('[project]\nname = "outer"\nversion = "9.9.9"\n')
    inner = outer / "inner"
    inner.mkdir()
    (inner / ".git").mkdir()  # sentinel — walk must stop here
    pkg = inner / "fakepkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    agent = pkg / "agent.py"
    agent.write_text("")

    fake_module = types.SimpleNamespace(__file__=str(agent))
    monkeypatch.setitem(sys.modules, "fakepkg.agent", fake_module)

    # Prevent importlib fallback from picking up a real distribution named fakepkg
    import importlib.metadata as md

    monkeypatch.setattr(md, "packages_distributions", lambda: {})

    assert versioning.resolve_version("fakepkg.agent") is None


def test_pyproject_walk_preferred_over_metadata(monkeypatch, tmp_path):
    """When pyproject walk succeeds, metadata lookup is never consulted."""
    agent_path = _make_fake_pkg(tmp_path, package_name="mixedpkg", pyproject_version="3.3.3")
    fake_module = types.SimpleNamespace(__file__=str(agent_path))
    monkeypatch.setitem(sys.modules, "mixedpkg.agent", fake_module)

    import importlib.metadata as md

    # Metadata would return a different value if consulted — assert the
    # walk is preferred and the metadata is not read.
    called = {"yes": False}

    def spy_packages_distributions():
        called["yes"] = True
        return {"mixedpkg": ["khonliang-mixed"]}

    monkeypatch.setattr(md, "packages_distributions", spy_packages_distributions)
    monkeypatch.setattr(md, "version", lambda name: "7.7.7")

    assert versioning.resolve_version("mixedpkg.agent") == "3.3.3"
    assert called["yes"] is False, "metadata consulted despite successful pyproject walk"


def test_pyproject_walk_falls_back_to_metadata_when_absent(monkeypatch, tmp_path):
    """Module outside any repo → walk returns None → metadata fallback runs."""
    # Place the module file somewhere with no pyproject / .git anywhere up.
    stray = tmp_path / "stray"
    stray.mkdir()
    agent = stray / "strayagent.py"
    agent.write_text("")

    fake_module = types.SimpleNamespace(__file__=str(agent))
    monkeypatch.setitem(sys.modules, "straypkg.agent", fake_module)

    import importlib.metadata as md

    monkeypatch.setattr(
        md, "packages_distributions", lambda: {"straypkg": ["khonliang-stray"]}
    )
    monkeypatch.setattr(md, "version", lambda name: "8.8.8")

    assert versioning.resolve_version("straypkg.agent") == "8.8.8"


def test_pyproject_above_non_git_rooted_module_is_ignored(monkeypatch, tmp_path):
    """Module outside a git root must not match an ancestor pyproject.

    Without a ``.git/`` anchor above the module, the walk must refuse
    to pick up whatever pyproject happens to exist further up the tree
    — e.g., a stray ``$HOME/pyproject.toml`` or a parent-repo config.
    The caller falls through to ``importlib.metadata`` instead.
    """
    # Ancestor has a pyproject but NO .git/ anywhere on the path to the module.
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "ancestor"\nversion = "6.6.6"\n'
    )
    nested = tmp_path / "sub" / "deeper"
    nested.mkdir(parents=True)
    agent = nested / "agent.py"
    agent.write_text("")

    fake_module = types.SimpleNamespace(__file__=str(agent))
    monkeypatch.setitem(sys.modules, "uncontrolled.agent", fake_module)

    import importlib.metadata as md

    monkeypatch.setattr(
        md, "packages_distributions", lambda: {"uncontrolled": ["khonliang-unctl"]}
    )
    monkeypatch.setattr(md, "version", lambda name: "1.2.3")

    # Must fall through to metadata, NOT report "6.6.6" from the ancestor.
    assert versioning.resolve_version("uncontrolled.agent") == "1.2.3"


def test_malformed_pyproject_returns_none(monkeypatch, tmp_path):
    """Syntax-broken pyproject doesn't crash — resolver returns None."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("this is :: not toml == at all\n")
    pkg = repo / "brokenpkg"
    pkg.mkdir()
    agent = pkg / "agent.py"
    agent.write_text("")

    fake_module = types.SimpleNamespace(__file__=str(agent))
    monkeypatch.setitem(sys.modules, "brokenpkg.agent", fake_module)

    import importlib.metadata as md

    monkeypatch.setattr(md, "packages_distributions", lambda: {})

    assert versioning.resolve_version("brokenpkg.agent") is None


def test_pyproject_without_project_version_returns_none(monkeypatch, tmp_path):
    """Valid TOML but no ``project.version`` key → None without crash."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text('[tool.black]\nline-length = 100\n')
    pkg = repo / "noverpkg"
    pkg.mkdir()
    agent = pkg / "agent.py"
    agent.write_text("")

    fake_module = types.SimpleNamespace(__file__=str(agent))
    monkeypatch.setitem(sys.modules, "noverpkg.agent", fake_module)

    import importlib.metadata as md

    monkeypatch.setattr(md, "packages_distributions", lambda: {})

    assert versioning.resolve_version("noverpkg.agent") is None


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


def test_resolve_version_is_cached(monkeypatch, tmp_path):
    """Second call returns the cached value without re-reading the file."""
    agent_path = _make_fake_pkg(
        tmp_path, package_name="cachedpkg", pyproject_version="1.0.0"
    )
    fake_module = types.SimpleNamespace(__file__=str(agent_path))
    monkeypatch.setitem(sys.modules, "cachedpkg.agent", fake_module)

    assert versioning.resolve_version("cachedpkg.agent") == "1.0.0"

    # Mutate the pyproject on disk; cached call must still return 1.0.0.
    (agent_path.parent.parent / "pyproject.toml").write_text(
        '[project]\nname = "cachedpkg"\nversion = "2.0.0"\n'
    )
    assert versioning.resolve_version("cachedpkg.agent") == "1.0.0"

    # Explicit cache reset recomputes.
    versioning._reset_cache()
    assert versioning.resolve_version("cachedpkg.agent") == "2.0.0"
