"""Shared pytest fixtures for the khonliang-bus-lib test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_versioning_cache():
    """Clear the ``resolve_version`` cache before every test.

    The cache is keyed by module name and holds for the process
    lifetime in production. In tests we monkeypatch metadata sources
    and module attributes per-test, so a lingering cache entry from a
    previous test would leak into the next one and produce
    order-dependent flakes.
    """
    from khonliang_bus import versioning

    versioning._reset_cache()
    yield
    versioning._reset_cache()
