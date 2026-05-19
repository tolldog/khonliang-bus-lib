"""Unit tests for ``khonliang_bus.launch`` — the launch-metadata capture
helpers consumed by ``BaseAgent.start()`` to extend the register handshake
with ``launch_spec`` + ``launch_info``.

fr_khonliang-bus-lib_2cfc0de6 (launch_spec) +
fr_khonliang-bus-lib_cccaa6a9 (launch_info).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from khonliang_bus.launch import (
    _git_info,
    _parse_config_arg,
    capture_launch_info,
    capture_launch_spec,
)


# ---------------------------------------------------------------------------
# _parse_config_arg
# ---------------------------------------------------------------------------


def test_parse_config_arg_space_form():
    assert _parse_config_arg(["-m", "agent", "--config", "/etc/agent/config.yaml"]) == "/etc/agent/config.yaml"


def test_parse_config_arg_equals_form():
    assert _parse_config_arg(["-m", "agent", "--config=/etc/agent/config.yaml"]) == "/etc/agent/config.yaml"


def test_parse_config_arg_absent():
    assert _parse_config_arg(["-m", "agent", "--bus", "http://localhost:8788"]) is None


def test_parse_config_arg_trailing_flag_with_no_value():
    # Defensive: --config at end with no value must not IndexError.
    assert _parse_config_arg(["-m", "agent", "--config"]) is None


def test_parse_config_arg_picks_first_occurrence():
    argv = ["-m", "agent", "--config", "first.yaml", "--config", "second.yaml"]
    assert _parse_config_arg(argv) == "first.yaml"


# ---------------------------------------------------------------------------
# capture_launch_spec
# ---------------------------------------------------------------------------


def test_capture_launch_spec_returns_required_keys():
    spec = capture_launch_spec()
    assert set(spec.keys()) == {"executable", "argv", "cwd", "config"}


def test_capture_launch_spec_executable_matches_sys():
    assert capture_launch_spec()["executable"] == sys.executable


def test_capture_launch_spec_cwd_matches():
    assert capture_launch_spec()["cwd"] == os.getcwd()


def test_capture_launch_spec_argv_is_list_of_str():
    argv = capture_launch_spec()["argv"]
    assert isinstance(argv, list)
    assert all(isinstance(a, str) for a in argv)


def test_capture_launch_spec_prefers_orig_argv(monkeypatch):
    """sys.orig_argv is the canonical source on Python 3.10+ — preserves -m module form."""
    monkeypatch.setattr(sys, "orig_argv", ["python3", "-m", "test_agent", "--config", "test.yaml"], raising=False)
    spec = capture_launch_spec()
    assert spec["argv"] == ["python3", "-m", "test_agent", "--config", "test.yaml"]
    assert spec["config"] == "test.yaml"


def test_capture_launch_spec_falls_back_to_argv_without_orig_argv(monkeypatch):
    # Older Pythons (or unusual harnesses) may not have sys.orig_argv.
    if hasattr(sys, "orig_argv"):
        monkeypatch.delattr(sys, "orig_argv", raising=False)
    monkeypatch.setattr(sys, "argv", ["script.py", "--config", "x.yaml"])
    spec = capture_launch_spec()
    assert spec["argv"] == ["script.py", "--config", "x.yaml"]
    assert spec["config"] == "x.yaml"


def test_capture_launch_spec_excludes_runtime_only_fields():
    """launch_spec must NOT include pid/started_at — those belong in launch_info."""
    spec = capture_launch_spec()
    assert "pid" not in spec
    assert "started_at" not in spec


# ---------------------------------------------------------------------------
# capture_launch_info
# ---------------------------------------------------------------------------


def test_capture_launch_info_returns_stable_key_set():
    info = capture_launch_info()
    # Stable wire shape: these five keys are always present (None when unknown).
    assert {"pid", "started_at", "commit_sha", "branch", "dirty"}.issubset(info.keys())


def test_capture_launch_info_pid_matches_process():
    assert capture_launch_info()["pid"] == os.getpid()


def test_capture_launch_info_started_at_is_recent():
    import time

    before = time.time()
    info = capture_launch_info()
    after = time.time()
    assert before <= info["started_at"] <= after


def test_capture_launch_info_excludes_spec_fields():
    """launch_info should not duplicate launch_spec's declarative fields."""
    info = capture_launch_info()
    assert "executable" not in info
    assert "argv" not in info
    assert "cwd" not in info
    assert "config" not in info


# ---------------------------------------------------------------------------
# _git_info — best-effort, never raises
# ---------------------------------------------------------------------------


def test_git_info_non_git_cwd_returns_empty(tmp_path: Path):
    """A directory that's not a git working tree returns {}."""
    assert _git_info(str(tmp_path)) == {}


def test_git_info_in_git_repo_returns_metadata(tmp_path: Path):
    """A real git repo yields commit_sha + branch + dirty."""
    # Skip if git isn't installed — _git_info handles this gracefully too.
    import shutil

    if not shutil.which("git"):
        pytest.skip("git not installed")

    # Initialize a minimal repo.
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "t@example.com"], check=True,
    )
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "f.txt").write_text("hello")
    subprocess.run(["git", "-C", str(tmp_path), "add", "f.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m", "init"], check=True,
    )

    info = _git_info(str(tmp_path))
    assert info["commit_sha"] is not None
    assert len(info["commit_sha"]) == 40
    assert info["branch"] == "main"
    assert info["dirty"] is False

    # Make the working tree dirty and re-check.
    (tmp_path / "f.txt").write_text("hello modified")
    info_dirty = _git_info(str(tmp_path))
    assert info_dirty["dirty"] is True


def test_git_info_never_raises(monkeypatch):
    """Even with a busted PATH or unreachable cwd, _git_info returns {} not raises."""
    monkeypatch.setenv("PATH", "")  # No git on PATH.
    assert _git_info("/") == {}


# ---------------------------------------------------------------------------
# Integration: the captured dicts compose correctly into a register payload
# ---------------------------------------------------------------------------


def test_launch_spec_and_info_together_have_no_field_overlap():
    """The two artifacts cover disjoint concerns; no key appears in both.

    This invariant lets the bus store them in separate tables (canonical
    vs runtime) without ambiguity about which is the source of truth.
    """
    spec = capture_launch_spec()
    info = capture_launch_info()
    overlap = set(spec.keys()) & set(info.keys())
    assert overlap == set(), f"unexpected overlap: {overlap}"
