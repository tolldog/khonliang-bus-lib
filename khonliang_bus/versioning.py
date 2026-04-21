"""Public version-resolution surface for khonliang agents.

Answers "what version of this package am I running?" at agent construction
time without each subclass re-implementing the ``importlib.metadata`` dance.

Resolution chain (first non-None wins):

1. **`-m` launch recovery** — when ``module_name == "__main__"``, consult
   ``sys.modules["__main__"].__spec__.name``. Python stashes the original
   dotted argument to ``python -m`` there. Without this, every
   bus-supervised agent (launched via ``python -m pkg.agent``) falls
   through to the metadata layer and reads stale install-time versions.
2. **`pyproject.toml` walk-up** — starting from the module's source file,
   walk parent directories until one contains ``.git/``. Try
   ``pyproject.toml`` at each level; on hit, parse ``project.version``
   via :mod:`tomllib`. The ``.git/`` sentinel prevents the walk from
   escaping the repo root into sibling-repo or home-directory
   pyprojects. This branch reads the **current on-disk** version, so
   bumping ``pyproject.toml`` produces an immediate change at the next
   agent restart — no ``pip install -e .`` rerun required.
3. **`importlib.metadata` fallback** — for non-editable installs, use
   ``packages_distributions()`` to map the top-level package to its
   distribution name, then read :func:`~importlib.metadata.version`.
4. **None** when nothing above succeeds.

Caching is per resolved ``module_name`` key for the process lifetime.
Once ``resolve_version`` answers for a given key (with explicit
``None`` normalized to ``"__main__"``), subsequent calls return the
same value; module names and pyproject locations do not change at
runtime, so no invalidation path is needed.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


_resolution_cache: dict[str, str | None] = {}


def _reset_cache() -> None:
    """Clear the resolution cache. Intended for tests only — production
    callers never need to invalidate because module names and pyproject
    locations do not change at runtime.
    """
    _resolution_cache.clear()


def resolve_version(module_name: str | None = None) -> str | None:
    """Return the version string that owns ``module_name``.

    Callers should usually pass an explicit module name such as
    ``type(self).__module__`` at construction time. When
    ``module_name`` is omitted or ``None``, the lookup resolves from
    the current ``__main__`` spec (the ``python -m`` recovery path).
    There is no implicit call-stack introspection.

    Returns ``None`` when nothing in the resolution chain can identify a
    version — callers decide whether to surface the miss or fall back to
    a default.
    """
    key = module_name if module_name is not None else "__main__"
    if key in _resolution_cache:
        return _resolution_cache[key]
    resolved = _resolve_uncached(key)
    _resolution_cache[key] = resolved
    return resolved


def _resolve_uncached(module_name: str) -> str | None:
    if not module_name:
        return None
    effective = module_name
    if effective == "__main__":
        effective = _dash_m_module_name() or effective
        if effective == "__main__":
            return None
    top = effective.split(".", 1)[0]
    if top == "__main__":
        return None

    from_pyproject = _resolve_from_pyproject(effective)
    if from_pyproject is not None:
        return from_pyproject
    return _resolve_from_metadata(top)


def _resolve_from_pyproject(module_name: str) -> str | None:
    """Walk up from the module's source file looking for ``pyproject.toml``.

    The search is anchored to a git root: we first locate the nearest
    ancestor directory containing ``.git/``, then only accept a
    ``pyproject.toml`` found *within that subtree* (between the module
    file and the git root, inclusive). If no ``.git/`` is found, the
    search returns ``None`` immediately — this prevents site-packages
    code from matching unrelated parent pyprojects (e.g. a stray
    ``$HOME/pyproject.toml``) and silently reporting the wrong version,
    letting the caller fall through to ``importlib.metadata`` instead.

    Returns ``None`` when the module has no resolvable source file, is
    not inside a git-tracked directory, finds no ``pyproject.toml``
    inside that git subtree, or when the found file is unreadable /
    lacks a ``project.version`` key.
    """
    try:
        import tomllib
    except ImportError:
        return None
    module = sys.modules.get(module_name)
    if module is None:
        try:
            import importlib
            module = importlib.import_module(module_name)
        except Exception:
            return None
    source_file = getattr(module, "__file__", None)
    if not source_file:
        return None
    start = Path(source_file).resolve().parent
    repo_root: Path | None = None
    for directory in [start, *start.parents]:
        if (directory / ".git").exists():
            repo_root = directory
            break
    if repo_root is None:
        return None
    for directory in [start, *start.parents]:
        candidate = directory / "pyproject.toml"
        if candidate.is_file():
            try:
                with candidate.open("rb") as f:
                    data = tomllib.load(f)
            except (OSError, tomllib.TOMLDecodeError) as exc:
                logger.debug("pyproject.toml at %s unreadable: %s", candidate, exc)
                return None
            version = (data.get("project") or {}).get("version")
            if isinstance(version, str) and version:
                return version
            return None
        if directory == repo_root:
            break
    return None


def _resolve_from_metadata(top_package: str) -> str | None:
    """Fallback: look up via ``importlib.metadata.packages_distributions``.

    This is the legacy behavior preserved from the earlier auto-derive
    iteration (PR #11). Used when the pyproject walk finds nothing —
    e.g., a non-editable installed distribution whose source files live
    in ``site-packages/`` with no enclosing ``.git/`` or
    ``pyproject.toml``.
    """
    try:
        from importlib.metadata import (
            PackageNotFoundError,
            packages_distributions,
            version,
        )
    except ImportError:
        return None
    try:
        dists = packages_distributions().get(top_package) or []
    except Exception:
        return None
    for dist_name in dists:
        try:
            return version(dist_name)
        except PackageNotFoundError:
            continue
        except Exception:
            return None
    return None


def _dash_m_module_name() -> str | None:
    """Recover the dotted name passed to ``python -m <name>``.

    When Python runs a module via ``-m``, the module is registered in
    ``sys.modules`` under both its original dotted name *and*
    ``"__main__"``. The original name lives on
    ``sys.modules["__main__"].__spec__.name``. Returns ``None`` when
    the ``__main__`` module has no spec (e.g. script launch, REPL).
    """
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return None
    spec = getattr(main_mod, "__spec__", None)
    if spec is None:
        return None
    name = getattr(spec, "name", None)
    return name if isinstance(name, str) and name else None
