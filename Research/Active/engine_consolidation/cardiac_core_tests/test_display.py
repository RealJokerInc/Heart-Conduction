"""Tests for cardiac_core._display — the environment probes and materialise-then-open sequence
behind ``.show()``.

This is the leaf module both the video and image layers delegate to; testing it once here is what
lets the per-class ``.show()`` tests stay thin. Every test that drives the TERMINAL branch points
``XDG_CACHE_HOME`` at a per-test tmp dir (autouse fixture below), so the suite never materialises
into the developer's real ``~/.cache/cardiac_core/show``.
"""

import os
import subprocess
import sys
import time

import pytest

from cardiac_core import _display as d


@pytest.fixture(autouse=True)
def _xdg_to_tmp(tmp_path, monkeypatch):
    """Isolate the materialise-cache. Individual tests may override with their own setenv."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))


# --------------------------------------------------------------------- environment probes

def test_in_notebook_false_in_a_plain_interpreter(monkeypatch):
    import IPython
    monkeypatch.setattr(IPython, "get_ipython", lambda: None)
    assert d.in_notebook() is False


def test_in_notebook_true_with_a_kernel(monkeypatch):
    """Without this, a regression to an unconditional ``return False`` passes every other test."""
    import IPython

    class FakeShell:
        kernel = object()          # a ZMQ kernel shell has this; a terminal shell does not

    monkeypatch.setattr(IPython, "get_ipython", lambda: FakeShell())
    assert d.in_notebook() is True


def test_open_externally_declines_without_a_display(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert d.open_externally("/tmp/whatever.mp4") is False


# --------------------------------------------------------------------- show_payload

def test_show_payload_prints_the_path_when_nothing_opens(monkeypatch, capsys):
    monkeypatch.setattr(d, "in_notebook", lambda: False)
    monkeypatch.setattr(d, "open_externally", lambda p: False)
    d.show_payload(read=lambda: b"payload", path=None, suffix="mp4",
                   noun="video", label="video player", rich=None)
    out = capsys.readouterr().out
    assert "No video player" in out and ".mp4" in out


@pytest.mark.parametrize("exc", [OSError("gone"), ValueError("no bytes")])
def test_show_payload_survives_an_unreadable_source(monkeypatch, capsys, tmp_path, exc):
    """A read() that raises (saved-then-deleted file, or unsaved-and-byteless) must degrade to a
    printed message and leave NO zero-byte file behind — never raise (the headless contract)."""
    monkeypatch.setattr(d, "in_notebook", lambda: False)
    monkeypatch.setattr(d, "open_externally", lambda p: True)   # must not be reached

    def bad_read():
        raise exc

    d.show_payload(read=bad_read, path=None, suffix="mp4",
                   noun="video", label="video player", rich=None)
    assert "video unavailable" in capsys.readouterr().out
    cache = tmp_path / "cardiac_core" / "show"
    leftover = list(cache.glob("*")) if cache.exists() else []
    assert leftover == [], f"a read() failure orphaned a file: {leftover}"


def test_cache_dir_is_absolute_without_home(monkeypatch):
    """Guards Success Criterion 5. A relative XDG_CACHE_HOME (or a literal '~' from a passwd-less
    container) must fall back to an absolute temp dir, never './relative/...' inside the project."""
    monkeypatch.setenv("XDG_CACHE_HOME", "relative_not_absolute")
    got = d._cache_dir()
    assert os.path.isabs(got), f"cache dir not absolute: {got!r}"
    assert not os.path.abspath(got).startswith(os.path.abspath(os.getcwd())), \
        f"cache dir is inside the project: {got!r}"


def test_cache_is_pruned_by_count(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    cache = tmp_path / "cardiac_core" / "show"
    cache.mkdir(parents=True)
    n = d._MAX_ENTRIES + 5
    now = time.time()
    for i in range(n):
        p = cache / f"video_{i:03d}.mp4"
        p.write_bytes(b"x")
        # recent (well inside _MAX_AGE_S so age-pruning does NOT fire) but staggered:
        # i=0 is oldest, i=n-1 newest.
        os.utime(p, (now - (n - i), now - (n - i)))

    d._cache_dir()                     # triggers _prune on entry

    remaining = {p.name for p in cache.glob("*")}
    assert len(remaining) <= d._MAX_ENTRIES, f"count cap not enforced: {len(remaining)}"
    assert "video_000.mp4" not in remaining, "oldest was not pruned first"
    assert f"video_{n - 1:03d}.mp4" in remaining, "newest was wrongly pruned"


def test_module_imports_no_heavy_deps():
    """Importing the leaf module must not pull matplotlib/numpy/torch — the whole reason it exists."""
    code = ("import sys, cardiac_core._display; "
            "print([m for m in ('matplotlib', 'numpy', 'torch') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr            # an ImportError also yields empty stdout
    assert out.stdout.strip() == "[]", f"heavy modules pulled in: {out.stdout!r}"
