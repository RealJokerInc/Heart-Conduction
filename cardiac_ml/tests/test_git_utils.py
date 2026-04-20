"""Tests for cardiac_ml/utils/git.py."""
from __future__ import annotations

import re

from cardiac_ml.utils.git import git_sha, git_dirty


def test_git_sha_short():
    """Short SHA is a 7-12 char hex string inside this repo."""
    sha = git_sha(short=True)
    assert sha, "git_sha() returned empty — is this not a git repo?"
    assert re.fullmatch(r"[0-9a-f]{7,12}", sha), f"unexpected SHA format: {sha!r}"


def test_git_sha_full():
    """Full SHA is a 40-char hex string."""
    sha = git_sha(short=False)
    assert re.fullmatch(r"[0-9a-f]{40}", sha), f"unexpected full SHA format: {sha!r}"


def test_git_dirty_returns_bool():
    """git_dirty() returns a bool regardless of tree state."""
    assert isinstance(git_dirty(), bool)
