"""Environment probes and the materialise-then-open sequence behind ``.show()``.

A neutral leaf module that both the video and image layers import. It exists to break an import
chain: ``image/info.py`` must NOT import ``video.encoders`` at module scope, because importing
``cardiac_core.video`` eagerly pulls in ``render`` and therefore ``matplotlib.use("Agg")``,
forcing that backend process-wide — the very thing the package's lazy ``__getattr__`` avoids.
Keeping the shared display logic here, with stdlib-only imports (plus a guarded lazy ``IPython``),
lets both layers call it without either dragging in the other.

Scope (v1): Jupyter/Colab (inline) and a GUI terminal (external player). A headless/SSH box
degrades to printing the file path — it never raises. An IDE console that lacks a Jupyter kernel
falls through to the external player, which still shows the media.

Windows caveat: ``XDG_CACHE_HOME`` is unset there, so materialised files land under
``~/.cache/cardiac_core/show`` rather than ``%LOCALAPPDATA%``. Harmless — the directory is
created, bounded, and pruned regardless of platform.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Optional

__all__ = ["in_notebook", "open_externally", "show_payload"]

# The terminal branch writes a real file for the OS player and CANNOT delete it on exit: the viewer
# is spawned DETACHED and outlives us, so an atexit cleanup would race it to "file not found".
# Instead files accumulate in a stable per-user cache, pruned on entry by BOTH age and count so it
# can never grow without bound. INLINE_MAX_BYTES does not gate this path — full payloads land here.
_MAX_AGE_S = 24 * 3600
_MAX_ENTRIES = 32


def _cache_dir() -> str:
    """A stable, absolute, bounded directory for materialised ``.show()`` files."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    # expanduser returns a literal "~" when HOME is unset AND there is no passwd entry (containers,
    # some CI); a relative XDG_CACHE_HOME is likewise invalid per the XDG spec. Either would create
    # "./~/.cache/..." INSIDE THE USER'S PROJECT — the one thing .show() promises never to do.
    if not os.path.isabs(base):
        base = tempfile.gettempdir()
    d = os.path.join(base, "cardiac_core", "show")
    os.makedirs(d, exist_ok=True)
    _prune(d)
    return d


def _prune(d: str) -> None:
    """Best-effort: drop entries older than ``_MAX_AGE_S``, and the oldest beyond ``_MAX_ENTRIES``."""
    try:
        entries = [(os.path.getmtime(os.path.join(d, n)), os.path.join(d, n))
                   for n in os.listdir(d)]
    except OSError:
        return                                       # never fatal — pruning is housekeeping
    now = time.time()
    stale = [p for m, p in entries if now - m > _MAX_AGE_S]
    if len(entries) > _MAX_ENTRIES:                  # oldest-first beyond the cap
        stale += [p for _, p in sorted(entries)[:len(entries) - _MAX_ENTRIES]]
    for p in set(stale):
        try:
            os.remove(p)
        except OSError:
            pass


def in_notebook() -> bool:
    """True inside a Jupyter/Colab kernel — somewhere that can render HTML inline.

    Terminal IPython has no ``kernel`` attribute, so it correctly falls through to the external
    player: it can echo a repr but cannot show a video.
    """
    try:
        from IPython import get_ipython                # lazy + guarded: IPython is optional
    except Exception:
        return False
    ip = get_ipython()
    return ip is not None and hasattr(ip, "kernel")


def open_externally(path: str) -> bool:
    """Hand a file to the OS's default application. True if something was launched.

    Returns False rather than raising on a headless box (no DISPLAY, an SSH session, a container),
    so the caller can report the path instead of failing.
    """
    if sys.platform == "win32":                       # pragma: no cover - platform
        try:
            os.startfile(path)                        # type: ignore[attr-defined]
            return True
        except OSError:
            return False

    opener = "open" if sys.platform == "darwin" else "xdg-open"
    if shutil.which(opener) is None:
        return False
    if sys.platform != "darwin" and not (os.environ.get("DISPLAY")
                                         or os.environ.get("WAYLAND_DISPLAY")):
        return False                                  # xdg-open with no GUI just fails
    try:
        subprocess.Popen([opener, path],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        return True
    except OSError:
        return False


def show_payload(*, read: Callable[[], bytes], path: Optional[str],
                 suffix: str, noun: str, label: str, rich: Any) -> None:
    """Display a media payload: inline in a notebook, else in an external viewer.

    Parameters
    ----------
    read : callable
        Zero-arg callable returning the encoded bytes.
    path : str | None
        An existing file to reuse, or ``None`` to materialise one in the cache dir.
    suffix : str
        File extension WITHOUT the leading dot (``"mp4"``, ``"png"`` ...).
    noun : str
        ``"video"`` / ``"figure"`` / ``"image"`` — used in the "The … is at:" line.
    label : str
        The viewer phrase, kept verbatim from the callers' historical strings
        (``"video player"`` / ``"image viewer"``) so their assertions still hold.
    rich : Any
        The object handed to ``IPython.display.display`` in the notebook branch — its
        ``_repr_html_`` carries the size cap, MIME-by-codec, and the deleted-file degrade.

    Returns ``None`` (like ``plt.show()``). Never raises: an unreadable source or a headless
    session degrades to a printed message.
    """
    if in_notebook():
        from IPython.display import display
        display(rich)
        return
    try:
        if path and os.path.exists(path):
            target = path
        else:
            payload = read()                          # read FIRST: creating the file first would
            fd, target = tempfile.mkstemp(            # orphan an empty one if read() raises
                dir=_cache_dir(), prefix=f"{noun}_", suffix=f".{suffix}")
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
    except (OSError, ValueError) as exc:              # read() raises ValueError when byte-less
        print(f"{noun} unavailable: {exc}")           # never raises — the headless contract
        return
    if not open_externally(target):
        print(f"No {label} could be opened (headless or remote session?).\n"
              f"The {noun} is at: {target}")
