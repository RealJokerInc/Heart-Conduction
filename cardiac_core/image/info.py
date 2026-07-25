"""``ImageInfo`` — what :func:`cardiac_core.image.draw` produced.

The still analogue of :class:`cardiac_core.video.VideoInfo`, and it carries the same contract:
**drawing displays; naming a destination saves**. ``path`` is ``None`` unless the caller said where
the figure should go, in which case the encoded bytes are the sole copy and live on ``data``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

__all__ = ["ImageInfo"]

# Above this the inline payload is reported rather than embedded — a notebook that swallows a
# 40 MB data URI is worse than one that tells you the figure is large.
_MAX_INLINE_BYTES = 16 * 1024 * 1024

_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "svg": "image/svg+xml",
}


@dataclass
class ImageInfo:
    """A rendered figure.

    Displays itself: in Jupyter/Colab the bare expression shows the image inline, with the bytes
    embedded as a data URI. That needs neither a file server nor a persistent disk, which is what
    makes it work on an ephemeral runtime.

    Attributes
    ----------
    path : str | None
        Where it was written, or ``None`` when nothing was saved (the default).
    data : bytes | None
        The encoded figure. Retained only when nothing was written to disk — then it is the sole copy.
    format : str
        ``"png"`` | ``"svg"`` | ``"pdf"`` | ``"jpg"`` | ``"jpeg"`` | ``"webp"``.
    width, height : int | None
        Pixel size, read back from the written file. ``None`` for vector formats — not fabricated.
    n_panels : int
        1 for a single spec, ``len(specs)`` for a multi-panel layout.
    vmin, vmax : float | None
        The resolved colour range. ``None`` when no map panel set one (e.g. a trace-only figure).
    size_bytes : int
        Size of the encoded figure.
    """

    path: Optional[str]
    data: Optional[bytes]
    format: str
    width: Optional[int]
    height: Optional[int]
    n_panels: int
    vmin: Optional[float]
    vmax: Optional[float]
    size_bytes: int

    @property
    def saved(self) -> bool:
        """True when the figure was written to a file the caller asked for."""
        return self.path is not None

    def read(self) -> bytes:
        """The encoded bytes, from memory or from the saved file."""
        if self.data is not None:
            return self.data
        with open(self.path, "rb") as fh:
            return fh.read()

    def save(self, path: str) -> str:
        """Write to ``path`` after the fact. Returns the path."""
        path = os.path.abspath(os.path.expanduser(str(path)))
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(self.read())
        return path

    def __fspath__(self) -> str:
        if self.path is None:
            raise TypeError(
                "this figure was not written to a file, so it has no path — pass `path=...` to "
                "save it, or call `.save('fig.png')` on the result."
            )
        return self.path

    def _repr_html_(self) -> str:
        import base64

        # PDF has no inline browser representation. Say so rather than emit a dead <img>.
        if self.format == "pdf":
            return (f"<p>PDF figure, {self.size_bytes:,} bytes — "
                    f"call <code>.save('fig.pdf')</code> to keep it.</p>")
        try:
            raw = self.read()
        except OSError as exc:                                   # pragma: no cover - defensive
            return f"<p>figure unavailable: {exc}</p>"
        if len(raw) > _MAX_INLINE_BYTES:
            return (f"<p>figure too large to display inline ({len(raw):,} bytes) — "
                    f"call <code>.save('fig.{self.format}')</code>.</p>")
        mime = _MIME.get(self.format, "image/png")
        b64 = base64.b64encode(raw).decode("ascii")
        return f'<img src="data:{mime};base64,{b64}" style="max-width:100%" alt="figure">'

    def __repr__(self) -> str:
        where = f"path={self.path!r}" if self.saved else "unsaved"
        size = f"{self.width}x{self.height}" if self.width else self.format
        rng = "" if self.vmin is None else f", range=({self.vmin:.1f}, {self.vmax:.1f})"
        return (f"ImageInfo({where}, {size}, panels={self.n_panels}{rng}, "
                f"{self.size_bytes:,} bytes)")
