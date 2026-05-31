"""Git metadata helpers for MLflow run tagging.

Isolated here so the MLflow logger stays pure (no subprocess calls inside
the logger itself). Both functions return defensive fallbacks when not
run inside a git repo — important for test environments and CI.
"""
from __future__ import annotations

import subprocess


def git_sha(short: bool = True) -> str:
    """Return current HEAD SHA (short by default).

    Returns empty string if not a git repo (caller can tag with '' to
    indicate 'unknown state' without crashing the run).
    """
    try:
        cmd = ["git", "rev-parse"]
        if short:
            cmd.append("--short")
        cmd.append("HEAD")
        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def git_dirty() -> bool:
    """True if working tree has uncommitted changes.

    Returns False outside a git repo. `git status --porcelain` is empty
    on a clean tree, non-empty on any dirty state.
    """
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
