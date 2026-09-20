"""
git_head
========

Read-only helper returning the current repository git HEAD sha.

Used by the Indicator Viewer page to stamp the source SHA that produced a
ViewerTrack (display-only metadata). No mutation, no network.
"""

from __future__ import annotations

import subprocess


def _repo_root() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()


_ROOT = _repo_root()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        text=True,
    ).strip()
