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
    _strip_interpreter,
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
# _strip_interpreter (Copilot R1: argv shape consistency)
# ---------------------------------------------------------------------------


def test_strip_interpreter_drops_exact_executable():
    """orig_argv[0] equals sys.executable absolute path → stripped."""
    assert _strip_interpreter(
        ["/opt/foo/.venv/bin/python3", "-m", "foo.agent", "--config", "x.yaml"],
        "/opt/foo/.venv/bin/python3",
    ) == ["-m", "foo.agent", "--config", "x.yaml"]


def test_strip_interpreter_drops_python_basename():
    """argv[0] basename ``python3`` (not the absolute path) → stripped."""
    assert _strip_interpreter(
        ["python3", "-m", "foo.agent"], "/usr/bin/python3.12",
    ) == ["-m", "foo.agent"]


def test_strip_interpreter_drops_versioned_python():
    """``python3.12``, ``python3.11``, etc. — basename starts with python + digits."""
    for interp in ("python", "python3", "python3.12", "python3.11"):
        out = _strip_interpreter([interp, "-m", "x"], "/usr/bin/python3")
        assert out == ["-m", "x"], f"failed for interp={interp!r}"


def test_strip_interpreter_keeps_script_argv():
    """sys.argv fallback: argv[0] is the script — must NOT be stripped."""
    assert _strip_interpreter(
        ["script.py", "--config", "x.yaml"], "/usr/bin/python3",
    ) == ["script.py", "--config", "x.yaml"]


def test_strip_interpreter_keeps_non_python_executable():
    """argv[0] like ``my_agent`` or ``./run.sh`` — leave alone."""
    assert _strip_interpreter(["my_agent", "-c"], "/usr/bin/python3") == ["my_agent", "-c"]
    assert _strip_interpreter(["./run.sh"], "/usr/bin/python3") == ["./run.sh"]


def test_strip_interpreter_empty_argv():
    """Defensive: empty argv stays empty."""
    assert _strip_interpreter([], "/usr/bin/python3") == []


# ---------------------------------------------------------------------------
# capture_launch_spec
# ---------------------------------------------------------------------------


def test_capture_launch_spec_returns_required_keys():
    spec = capture_launch_spec()
    assert set(spec.keys()) == {"executable", "args", "cwd", "config"}


def test_capture_launch_spec_executable_matches_sys():
    assert capture_launch_spec()["executable"] == sys.executable


def test_capture_launch_spec_cwd_matches():
    assert capture_launch_spec()["cwd"] == os.getcwd()


def test_capture_launch_spec_args_is_list_of_str():
    args = capture_launch_spec()["args"]
    assert isinstance(args, list)
    assert all(isinstance(a, str) for a in args)


def test_capture_launch_spec_strips_interpreter_from_orig_argv(monkeypatch):
    """Copilot R1: ensure ``args`` never contains the interpreter as its head,
    regardless of whether orig_argv (Py 3.10+) or sys.argv was the source.
    """
    monkeypatch.setattr(sys, "orig_argv", ["python3", "-m", "test_agent", "--config", "test.yaml"], raising=False)
    spec = capture_launch_spec()
    # Interpreter stripped — args is the respawn arg list.
    assert spec["args"] == ["-m", "test_agent", "--config", "test.yaml"]
    assert spec["config"] == "test.yaml"


def test_capture_launch_spec_preserves_script_argv_when_no_orig_argv(monkeypatch):
    # Older Pythons (or unusual harnesses) may not have sys.orig_argv.
    if hasattr(sys, "orig_argv"):
        monkeypatch.delattr(sys, "orig_argv", raising=False)
    monkeypatch.setattr(sys, "argv", ["script.py", "--config", "x.yaml"])
    spec = capture_launch_spec()
    # Script-mode argv has no interpreter prefix; preserve as-is.
    assert spec["args"] == ["script.py", "--config", "x.yaml"]
    assert spec["config"] == "x.yaml"


def test_capture_launch_spec_args_shape_consistent_across_sources(monkeypatch):
    """Both code paths produce ``args`` shaped as the respawn arg list."""
    # Path 1: orig_argv with interpreter — interpreter must be absent in args.
    monkeypatch.setattr(sys, "orig_argv", [sys.executable, "-m", "x.agent"], raising=False)
    spec1 = capture_launch_spec()
    assert spec1["args"][0] != sys.executable
    assert "python" not in os.path.basename(spec1["args"][0]).lower()

    # Path 2: orig_argv with script (rare but possible) — preserved.
    monkeypatch.setattr(sys, "orig_argv", ["script.py", "--flag"], raising=False)
    spec2 = capture_launch_spec()
    assert spec2["args"] == ["script.py", "--flag"]


def test_capture_launch_spec_excludes_runtime_only_fields():
    """launch_spec must NOT include pid/started_at — those belong in launch_info / top-level."""
    spec = capture_launch_spec()
    assert "pid" not in spec
    assert "started_at" not in spec


# ---------------------------------------------------------------------------
# capture_launch_info
# ---------------------------------------------------------------------------


def test_capture_launch_info_returns_stable_key_set():
    info = capture_launch_info()
    # Stable wire shape: these four keys are always present (None when unknown).
    # Note: ``pid`` is NOT here — see Copilot R1 / single-source-of-truth
    # commentary. The outer registration payload carries it at the top level.
    assert {"started_at", "commit_sha", "branch", "dirty"}.issubset(info.keys())


def test_capture_launch_info_does_not_carry_pid():
    """Copilot R1: pid is already in the outer registration payload — must
    not be duplicated here, to avoid ambiguity when the two diverge.
    """
    assert "pid" not in capture_launch_info()


def test_capture_launch_info_started_at_is_recent():
    import time

    before = time.time()
    info = capture_launch_info()
    after = time.time()
    assert before <= info["started_at"] <= after


def test_capture_launch_info_started_at_is_wall_clock_epoch():
    """Documented contract: started_at is wall-clock epoch seconds, not
    monotonic. Verify it's within a sensible range of ``time.time()``.
    """
    import time

    info = capture_launch_info()
    # If time.time() drifts catastrophically vs this value, the contract is broken.
    # Allow a generous window (10s) for test scheduling latency.
    assert abs(time.time() - info["started_at"]) < 10.0


def test_capture_launch_info_excludes_spec_fields():
    """launch_info should not duplicate launch_spec's declarative fields."""
    info = capture_launch_info()
    assert "executable" not in info
    assert "args" not in info
    assert "argv" not in info  # old name; verify it really is gone
    assert "cwd" not in info
    assert "config" not in info


# ---------------------------------------------------------------------------
# _git_info — best-effort, never raises
# ---------------------------------------------------------------------------


def test_git_info_non_git_cwd_returns_empty(tmp_path: Path):
    """A directory that's not a git working tree returns {}."""
    assert _git_info(str(tmp_path)) == {}


def _init_minimal_repo(path: Path) -> None:
    """Helper: initialize a minimal git repo at ``path`` with one commit."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"], check=True)
    (path / "f.txt").write_text("hello")
    subprocess.run(["git", "-C", str(path), "add", "f.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "init"], check=True)


def test_git_info_in_git_repo_returns_metadata(tmp_path: Path):
    """A real git repo where cwd IS the toplevel yields commit_sha + branch + dirty."""
    # Skip if git isn't installed — _git_info handles this gracefully too.
    import shutil

    if not shutil.which("git"):
        pytest.skip("git not installed")

    _init_minimal_repo(tmp_path)

    info = _git_info(str(tmp_path))
    assert info["commit_sha"] is not None
    assert len(info["commit_sha"]) == 40
    assert info["branch"] == "main"
    assert info["dirty"] is False

    # Make the working tree dirty and re-check.
    (tmp_path / "f.txt").write_text("hello modified")
    info_dirty = _git_info(str(tmp_path))
    assert info_dirty["dirty"] is True


def test_git_info_returns_empty_when_cwd_is_subdir_of_repo(tmp_path: Path):
    """Copilot R1: cwd inside a git repo but not at the toplevel must NOT
    return that repo's metadata — would attach unrelated provenance to the
    agent's runtime info.
    """
    import shutil

    if not shutil.which("git"):
        pytest.skip("git not installed")

    _init_minimal_repo(tmp_path)
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    # cwd is a child of the repo root; toplevel-equality check fails.
    assert _git_info(str(subdir)) == {}


def test_git_info_returns_empty_when_cwd_outside_repo_root_via_traversal(tmp_path: Path):
    """Concrete dotfiles-checkout scenario from the Copilot finding:
    cwd is /opt/foo (agent install) but /opt is inside someone's dotfiles
    checkout. ``_git_info`` walking up would return the dotfiles repo's
    info. Toplevel-equality stops that.
    """
    import shutil

    if not shutil.which("git"):
        pytest.skip("git not installed")

    # tmp_path = the "dotfiles" repo.
    _init_minimal_repo(tmp_path)
    # opt_foo = "/opt/foo" — the agent install dir, a sibling-child of the repo.
    opt_foo = tmp_path / "opt" / "foo"
    opt_foo.mkdir(parents=True)
    assert _git_info(str(opt_foo)) == {}


def test_git_info_handles_symlinked_cwd(tmp_path: Path):
    """If cwd reaches the repo toplevel via a symlink, realpath comparison
    keeps the toplevel-equality check working correctly.
    """
    import shutil

    if not shutil.which("git"):
        pytest.skip("git not installed")

    _init_minimal_repo(tmp_path)
    link = tmp_path.parent / (tmp_path.name + "_link")
    link.symlink_to(tmp_path)
    try:
        info = _git_info(str(link))
        assert info.get("commit_sha") is not None
        assert info.get("branch") == "main"
    finally:
        link.unlink()


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
    Per Copilot R1, ``pid`` is also intentionally absent from both
    artifacts — the outer registration payload's top-level ``pid``
    field is the single authoritative source.
    """
    spec = capture_launch_spec()
    info = capture_launch_info()
    overlap = set(spec.keys()) & set(info.keys())
    assert overlap == set(), f"unexpected overlap: {overlap}"
    # Additionally: pid lives only at the top-level register payload.
    assert "pid" not in spec
    assert "pid" not in info
