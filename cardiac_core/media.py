"""media_path — canonical save-path helper for project images & videos.

Enforces the repo-wide convention (see CLAUDE.md "Saving images & videos"):

    media/{question}/{images|videos}/{YYYY-MM-DD}/{slug}_NN.ext

Self-contained (stdlib only) — import directly: ``from cardiac_core.media import media_path``.
Does NOT trigger the lazy ``cardiac_core`` API import.

Example
-------
    from cardiac_core.media import media_path
    import matplotlib.pyplot as plt
    plt.savefig(media_path("boundary_conduction_speedup", "images", "wavefront snapshot t10"))
    # -> media/boundary_conduction_speedup/images/2026-05-31/wavefront-snapshot-t10_01.png
"""
from __future__ import annotations

import os
import re
from datetime import date as _date

# repo root = parent of the cardiac_core/ package directory
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_IMAGE_EXT = {"png", "jpg", "jpeg", "svg", "gif"}
_VIDEO_EXT = {"mp4", "webm", "mov", "avi"}


def slugify(text: str) -> str:
    """lowercase, non-alphanumeric -> '-', collapsed and trimmed."""
    s = re.sub(r"[^a-z0-9]+", "-", str(text).lower())
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "file"


def media_path(
    question: str,
    kind: str,
    slug: str,
    ext: str = "png",
    *,
    date: str | None = None,
    bulk: bool = False,
    root: str | None = None,
) -> str:
    """Return a convention-compliant path for a NEW image/video, creating dirs.

    media/{question}/{kind}/{date}/{slug}_NN.ext   (``NN`` auto-increments per slug/day)

    Parameters
    ----------
    question : research-question slug (the ``Research/Active/{question}/`` folder).
               Use ``"_unmapped"`` if the asset has no clear owner.
    kind     : ``"images"`` or ``"videos"``.
    slug     : short description (slugified automatically).
    ext      : file extension (with or without leading dot). Default ``"png"``.
    date     : ``YYYY-MM-DD``; defaults to today.
    bulk     : if True, place under ``{question}/_sim_outputs/{kind}/...`` (gitignored)
               for regenerable bulk output instead of the committed tree.
    root     : repo root override (defaults to the repo containing cardiac_core/).

    Returns the full path; the containing directory is created.

    Contract: ``NN`` is the next slot whose file does not yet exist on disk. Save the
    returned path (``plt.savefig``/``cv2.VideoWriter``/...) before requesting another
    path for the SAME slug+day, otherwise both calls return ``_01``. The normal
    get-path-then-save loop increments correctly.
    """
    if kind not in ("images", "videos"):
        raise ValueError(f"kind must be 'images' or 'videos', got {kind!r}")
    ext = ext.lstrip(".").lower()
    expected = _IMAGE_EXT if kind == "images" else _VIDEO_EXT
    if ext not in expected:
        raise ValueError(f"extension {ext!r} is not a {kind[:-1]} type {sorted(expected)}")

    root = root or _REPO_ROOT
    day = date or _date.today().isoformat()
    slug = slugify(slug)

    parts = [root, "media", str(question)]
    if bulk:
        parts.append("_sim_outputs")
    parts += [kind, day]
    directory = os.path.join(*parts)
    os.makedirs(directory, exist_ok=True)

    n = 1
    while os.path.exists(os.path.join(directory, f"{slug}_{n:02d}.{ext}")):
        n += 1
    return os.path.join(directory, f"{slug}_{n:02d}.{ext}")
