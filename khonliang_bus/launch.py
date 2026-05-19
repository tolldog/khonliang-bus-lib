"""Agent-side launch metadata capture for the registration handshake.

Two distinct artifacts captured at agent startup, plus the outer
registration payload's existing ``pid`` field — three fields with
disjoint roles:

- ``launch_spec`` — declarative how-to-launch metadata. What the bus
  needs to spawn another process matching this agent_id. Used to
  populate ``installed_agents`` (canonical install row). Carries
  ``executable``, ``args``, ``cwd``, ``config`` — the trailing argument
  list a consumer passes to ``[executable] + args`` to respawn the
  agent. Closes ``fr_khonliang-bus-lib_2cfc0de6``.

- ``launch_info`` — observational runtime snapshot. What process is
  *currently* serving this agent_id, where from, what commit. Used to
  populate ``runtime_agents`` (or the runtime-overlay columns on
  ``installed_agents``). Bus diffs against ``launch_spec`` to surface
  the canonical-vs-ad-hoc distinction (``bus_agent_provenance``).
  Carries ``started_at`` (wall-clock epoch — see :func:`capture_launch_info`
  for monotonic-vs-wall-clock caveats) + best-effort git fields.
  Closes ``fr_khonliang-bus-lib_cccaa6a9``.

- ``pid`` — already a top-level field on the register payload; the
  authoritative process identity. Not duplicated in either artifact
  above. Bus joins by ``agent_id`` server-side.

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


def _strip_interpreter(argv: list[str], executable: str) -> list[str]:
    """Drop ``argv[0]`` if it's the Python interpreter, so ``args`` is
    unambiguously the trailing argument list a respawn would pass to
    ``[executable] + args``.

    ``sys.orig_argv`` (Python 3.10+) puts the interpreter as argv[0]
    (e.g. ``["python3", "-m", "foo.agent", ...]``). ``sys.argv`` does
    not (it starts with the script path or module). Normalize to a
    single shape so consumers can construct a respawn command as
    ``[executable] + args`` without case-splitting.

    Heuristic: strip when argv[0] equals ``sys.executable`` exactly OR
    its basename matches ``python``/``python3``/``pythonX.Y``. Leaves
    untouched otherwise (sys.argv fallback case, where argv[0] is the
    script and must be preserved).
    """
    if not argv:
        return argv
    head = argv[0]
    if head == executable:
        return argv[1:]
    base = os.path.basename(head)
    if base == "python" or (base.startswith("python") and base[6:].replace(".", "").isdigit()):
        return argv[1:]
    return list(argv)


def _git_info(cwd: str) -> dict[str, Any]:
    """Best-effort git metadata for ``cwd``, scoped to the case where ``cwd``
    is the working-tree root.

    Returns ``{}`` (empty) unless:
    - ``cwd`` contains a ``.git`` entry (fast filesystem check), AND
    - ``git`` is on PATH, AND
    - ``cwd`` is inside a git working tree, AND
    - ``git rev-parse --show-toplevel`` resolves to ``cwd`` itself.

    The toplevel-equality check is deliberate: if the agent's cwd happens
    to sit inside an unrelated repo (operator's dotfiles checkout, an
    ``/opt`` parent that someone cloned, etc.), walking up would attach
    that repo's commit/branch/dirty status to the agent's provenance —
    a false positive ``bus_agent_provenance`` consumers cannot
    distinguish from real signal. Scoping to ``cwd == toplevel`` means
    git provenance fires only when the agent IS the repo (typical dev
    clone case); prod ``pip install`` deployments correctly report no
    git info.

    **Startup-cost profile** (Copilot R2):
    - Common case (prod pip install, no ``.git`` at cwd): single
      filesystem ``stat`` call — sub-millisecond. No git subprocess.
    - Dev clone where the agent IS the repo: up to 4 ``git`` subprocess
      calls, each with a 1-second timeout. Worst case latency ~4s
      *only* if git is hung; healthy git completes in <100ms total.
    - The per-call timeout dropped from 2s to 1s post-R2 to bound the
      worst case more tightly.

    Never raises — git introspection is informational, not load-bearing.
    """
    # Fast path: if there's no .git entry at cwd, this isn't a working-tree
    # root we care about. Avoids the git subprocess entirely for prod
    # pip-install deployments (the common case).
    if not os.path.exists(os.path.join(cwd, ".git")):
        return {}
    if not shutil.which("git"):
        return {}
    try:
        # Confirm we're in a working tree first; cheap and authoritative.
        check = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=cwd, capture_output=True, text=True, timeout=1,
        )
        if check.returncode != 0 or check.stdout.strip() != "true":
            return {}
        # Scope check: only attach git info when cwd is the working-tree root.
        toplevel = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd, capture_output=True, text=True, timeout=1,
        )
        if toplevel.returncode != 0:
            return {}
        # Compare resolved (realpath) forms to handle symlinks the same way.
        top = os.path.realpath(toplevel.stdout.strip())
        if top != os.path.realpath(cwd):
            return {}
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=1,
        )
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=1,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd, capture_output=True, text=True, timeout=1,
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
        ``{executable, args, cwd, config}`` where ``args`` is the
        respawn argument list — what a consumer would pass as the
        trailing arguments to ``[executable] + args`` to construct
        a process matching this agent. ``args`` is normalized to
        exclude the interpreter (see :func:`_strip_interpreter`),
        so the shape is consistent regardless of whether the source
        was ``sys.orig_argv`` or ``sys.argv``.
    """
    executable = sys.executable
    raw = list(getattr(sys, "orig_argv", sys.argv))
    args = _strip_interpreter(raw, executable)
    return {
        "executable": executable,
        "args": args,
        "cwd": os.getcwd(),
        "config": _parse_config_arg(args),
    }


def capture_launch_info() -> dict[str, Any]:
    """Observational runtime snapshot.

    Captured at agent startup. Reports the wall-clock start time plus
    best-effort git metadata from cwd. Overlapping fields with
    ``launch_spec`` (executable, args, cwd, config) and the outer
    registration payload (``pid``, already present at top level) are
    deliberately not duplicated.

    The bus joins ``launch_info`` with ``launch_spec`` (declarative
    install) and the top-level ``pid`` (process identity) by ``agent_id``
    server-side. Single source of truth for each field.

    Returns:
        ``{started_at, commit_sha, branch, dirty}``.

        ``started_at`` is **wall-clock epoch seconds** (``time.time()``).
        This is an absolute timestamp suitable for ordering and audit
        display; it is NOT a monotonic value and MUST NOT be subtracted
        from a later ``time.monotonic()`` reading. For elapsed-time
        comparisons against a later observation, fetch a fresh
        ``time.time()`` on the consumer side and diff against this.
        The wall clock can shift backwards under NTP/DST adjustments;
        consumers that need stable ordering across such shifts should
        rely on bus-side receive timestamps as a secondary signal.

        Git fields are None if ``cwd`` is not the root of a git working
        tree (see :func:`_git_info` for the scoping rule).
    """
    info: dict[str, Any] = {
        "started_at": time.time(),
    }
    info.update(_git_info(os.getcwd()))
    # Ensure all expected keys exist for stable wire shape, even when git
    # introspection found nothing.
    info.setdefault("commit_sha", None)
    info.setdefault("branch", None)
    info.setdefault("dirty", None)
    return info
