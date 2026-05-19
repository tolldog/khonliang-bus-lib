"""Agent-side launch metadata capture for the registration handshake.

Two distinct artifacts captured at agent startup:

- ``launch_spec`` — declarative how-to-launch metadata. What the bus needs
  to spawn another process matching this agent_id. Used to populate
  ``installed_agents`` (canonical install row). Excludes runtime-only
  fields like ``pid``. Closes ``fr_khonliang-bus-lib_2cfc0de6``.

- ``launch_info`` — observational runtime snapshot. What process is
  *currently* serving this agent_id, where from, what commit. Used to
  populate ``runtime_agents`` (or the runtime-overlay columns on
  ``installed_agents``). Bus diffs against ``launch_spec`` to surface
  the canonical-vs-ad-hoc distinction (``bus_agent_provenance``).
  Closes ``fr_khonliang-bus-lib_cccaa6a9``.

Wire format is additive: a bus that doesn't know these fields ignores
them; an agent that doesn't send them keeps the prior register-payload
shape working.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from typing import Any


def _parse_config_arg(argv: list[str]) -> str | None:
    """Extract ``--config <path>`` from argv. Returns None if absent.

    Supports both ``--config path`` and ``--config=path`` forms.
    """
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


def _git_info(cwd: str) -> dict[str, Any]:
    """Best-effort git metadata for ``cwd``. Empty dict if cwd isn't a git repo.

    Never raises — git introspection is informational, not load-bearing.
    """
    if not shutil.which("git"):
        return {}
    try:
        # Confirm we're in a working tree first; cheap and authoritative.
        check = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=cwd, capture_output=True, text=True, timeout=2,
        )
        if check.returncode != 0 or check.stdout.strip() != "true":
            return {}
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=2,
        )
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=2,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd, capture_output=True, text=True, timeout=2,
        )
    except (subprocess.TimeoutExpired, OSError):
        return {}
    return {
        "commit_sha": commit.stdout.strip() or None,
        "branch": branch.stdout.strip() or None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def capture_launch_spec() -> dict[str, Any]:
    """Declarative how-to-launch metadata.

    Captured at agent startup from the running process's invocation.
    Excludes pid/started_at — those belong in ``launch_info``.

    Returns:
        ``{executable, argv, cwd, config}``.
        ``argv`` uses ``sys.orig_argv`` (Python 3.10+) when available so
        ``-m module`` invocations are preserved exactly; falls back to
        ``sys.argv`` otherwise.
    """
    argv = list(getattr(sys, "orig_argv", sys.argv))
    return {
        "executable": sys.executable,
        "argv": argv,
        "cwd": os.getcwd(),
        "config": _parse_config_arg(argv),
    }


def capture_launch_info() -> dict[str, Any]:
    """Observational runtime snapshot.

    Captured at agent startup. Includes pid, start time, and best-effort
    git metadata from cwd. Overlapping fields with ``launch_spec``
    (executable, argv, cwd, config) are deliberately not duplicated —
    callers join the two artifacts by ``agent_id`` server-side.

    Returns:
        ``{pid, started_at, commit_sha, branch, dirty}``.
        Git fields are None if cwd isn't a git working tree.
    """
    info: dict[str, Any] = {
        "pid": os.getpid(),
        "started_at": time.time(),
    }
    info.update(_git_info(os.getcwd()))
    # Ensure all expected keys exist for stable wire shape, even when git
    # introspection found nothing.
    info.setdefault("commit_sha", None)
    info.setdefault("branch", None)
    info.setdefault("dirty", None)
    return info
