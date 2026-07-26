"""Encoder backends, canvas geometry, and the :class:`VideoInfo` return type.

Backend selection happens BEFORE the output path is built: a fallback changes both the extension
and the ``kind`` directory, and :func:`cardiac_core.media.media_path` validates ext against kind,
so a writer that discovers its own fallback after the path exists cannot fix it.

No fallback is silent — every downgrade emits a ``UserWarning`` naming the backend actually used,
and the choice is reported on ``VideoInfo.backend``.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "VideoInfo", "ImagePath", "RESOLUTIONS", "CODECS",
    "select_backend", "resolve_canvas", "fit_frame", "burn_timestamp", "open_writer",
]

# File extension per encoder codec, for naming a temp file a player can open.
_EXT_FOR_CODEC = {"libx264": "mp4", "mp4v": "mp4", "libvpx-vp9": "webm", "gif": "gif"}


def _in_notebook() -> bool:
    """True inside a Jupyter/Colab kernel — i.e. somewhere that can render HTML.

    Terminal IPython has no ``kernel`` attribute, so it correctly falls through to the external
    player: it can echo a repr but cannot show a video.
    """
    try:
        from IPython import get_ipython
    except Exception:
        return False
    ip = get_ipython()
    return ip is not None and hasattr(ip, "kernel")


def _open_externally(path: str) -> bool:
    """Hand a file to the OS's default application. True if something was launched.

    Returns False rather than raising on a headless box (no DISPLAY, an SSH session, a container),
    so the caller can report the path instead of failing.
    """
    import shutil
    import subprocess
    import sys

    if sys.platform == "win32":                              # pragma: no cover - platform
        try:
            os.startfile(path)                               # type: ignore[attr-defined]
            return True
        except OSError:
            return False

    opener = "open" if sys.platform == "darwin" else "xdg-open"
    if shutil.which(opener) is None:
        return False
    if sys.platform != "darwin" and not (os.environ.get("DISPLAY")
                                         or os.environ.get("WAYLAND_DISPLAY")):
        return False                                         # xdg-open with no GUI just fails
    try:
        subprocess.Popen([opener, path],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        return True
    except OSError:
        return False


class ImagePath(str):
    """A rendered still. Behaves exactly like the path string it always was, and displays inline.

    Subclassing ``str`` keeps every existing caller working (``p.endswith('.png')``,
    ``open(p)``, path joins) while adding notebook display. When nothing was saved the string
    value is a short summary instead of a path, and the encoded bytes live in ``.data``.
    ``format`` records what those bytes actually are — a preview follows ``path=``'s extension,
    so it is not always PNG.
    """

    data: Optional[bytes]
    saved: bool
    format: str

    UNSAVED_TEXT = "<image — not saved (pass path= to write a file)>"

    # Browser MIME per still format. A preview follows path='s extension, so this cannot assume PNG.
    _MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
             "webp": "image/webp", "svg": "image/svg+xml"}

    def __new__(cls, path: Optional[str], data: Optional[bytes] = None,
                format: str = "png") -> "ImagePath":
        text = path if path is not None else cls.UNSAVED_TEXT
        obj = super().__new__(cls, text)
        obj.data = data
        obj.saved = path is not None
        obj.format = str(format).lower().lstrip(".")
        return obj

    def __reduce__(self):
        # str supplies __getnewargs__, which would rebuild this from its TEXT alone — turning an
        # unsaved instance into a "saved" one whose path is the human summary. Carry every field.
        return (self.__class__, (str(self) if self.saved else None, self.data, self.format))

    def read(self) -> bytes:
        """The encoded bytes, from memory or from the saved file."""
        if self.data is not None:
            return self.data
        if not self.saved:                                       # pragma: no cover - defensive
            raise ValueError("this image was not saved and carries no bytes")
        with open(str(self), "rb") as fh:
            return fh.read()

    def save(self, path: str) -> str:
        """Write to ``path`` after the fact. Returns the path.

        Note: a ``str`` cannot change its own value, so THIS object stays unsaved — unlike
        :meth:`VideoInfo.save` / :meth:`ImageInfo.save`, which mark themselves saved. Use the
        returned path.
        """
        path = os.path.abspath(os.path.expanduser(str(path)))
        # Read FIRST: open(..., "wb") truncates, so saving onto our own path would zero the file
        # and then read back nothing. Same reason ImageInfo.save() reads before it opens.
        payload = self.read()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(payload)
        return path

    @property
    def size_bytes(self) -> int:
        """Encoded size, without reading the payload when it is on disk."""
        if self.data is not None:
            return len(self.data)
        try:
            return os.path.getsize(str(self))
        except OSError:
            return 0

    def _repr_html_(self) -> str:
        import base64
        if self.format == "pdf":            # no inline browser representation
            return "<p>PDF figure — call <code>.save('fig.pdf')</code> to keep it.</p>"
        # Same cap as VideoInfo/ImageInfo, decided BEFORE reading: an annotated preview at a
        # high dpi is easily multi-megabyte, and it would be base64'd into the .ipynb.
        if self.size_bytes > INLINE_MAX_BYTES:
            return (f"<p>image too large to display inline ({self.size_bytes:,} bytes) — "
                    f"call <code>.save('frame.{self.format}')</code>.</p>")
        try:
            raw = self.read()
        except OSError as exc:      # the saved file was moved/deleted since the render
            return f"<p>image unavailable: {exc}</p>"
        mime = self._MIME.get(self.format, "image/png")
        b64 = base64.b64encode(raw).decode("ascii")
        return f'<img src="data:{mime};base64,{b64}" style="max-width:100%" alt="preview frame">'


RESOLUTIONS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4k": (3840, 2160),
}

# Per-container codec + pixel format. Verified available in the bundled ffmpeg.
# NOTE: the webm ``None`` is documentation, not suppression — imageio's writer DEFAULTS to
# pixelformat="yuv420p", so omitting the kwarg re-supplies it. Harmless (webm was verified to
# write correctly either way); do not expect ``None`` to disable it.
CODECS = {
    "mp4": ("libx264", "yuv420p"),
    "webm": ("libvpx-vp9", None),
}

SUPPORTED_FORMATS = ("mp4", "webm", "gif")

# GIF must hold every frame for the palette, so that backend alone accumulates. Cap it.
GIF_MAX_FRAMES = 200

# Ceiling on inline notebook display. The bytes are base64'd into the .ipynb itself (+33%), so an
# uncapped embed would commit a multi-megabyte payload to git every time a cell is re-run.
INLINE_MAX_BYTES = 16 * 1024 * 1024

# Browser MIME by codec. mp4v (OpenCV's MPEG-4 Part 2) is deliberately absent: it writes a valid
# .mp4 that no browser will decode, which is reported rather than embedded.
_MIME = {"libx264": "video/mp4", "libvpx-vp9": "video/webm", "gif": "image/gif"}


@dataclass
class VideoInfo:
    """What :func:`cardiac_core.video.render` produced.

    Displays itself: in Jupyter/Colab the bare expression plays the video inline, with the bytes
    embedded as a data URI. That is the same mechanism matplotlib uses for a figure, so it needs
    no file server and survives an ephemeral runtime.

    ``path`` is ``None`` unless a destination was named (``path=`` or the ``media/`` convention
    keywords). Following matplotlib: displaying is not saving. When a path IS set, ``str(info)``
    and ``os.fspath(info)`` give it, so it can be passed to anything taking a filename.
    """

    path: Optional[str]
    n_frames: int
    fps: float
    backend: str            # "imageio-ffmpeg" | "opencv" | "pillow-gif"
    codec: str              # "libx264" | "libvpx-vp9" | "mp4v" | "gif" — never None
    width: int
    height: int
    duration_s: float
    vmin: float
    vmax: float
    stride: int
    size_bytes: int
    # The RESOLVED rate handed to the encoder ("2M" for webm/VP9). Surfaced because it is
    # otherwise untestable: the VP9 "Neither bitrate nor constrained quality" message comes from
    # the ffmpeg SUBPROCESS and imageio sets ffmpeg_log_level="quiet", so neither pytest.warns
    # nor capfd can observe it.
    bitrate: Optional[str] = None
    # Encoded bytes, retained only when nothing was written to disk — they are the sole copy.
    # repr=False: the dataclass repr would otherwise dump the whole video as an escaped bytes
    # literal into any REPL echo, log line, or failing-assertion message.
    data: Optional[bytes] = field(default=None, repr=False)

    @property
    def saved(self) -> bool:
        """True when the render was written to a file the caller asked for."""
        return self.path is not None

    def read(self) -> bytes:
        """The encoded bytes, from memory or from the saved file."""
        if self.data is not None:
            return self.data
        if self.path is None:                               # pragma: no cover - defensive
            raise ValueError("this render was not saved and carries no bytes")
        with open(self.path, "rb") as fh:
            return fh.read()

    def save(self, path: str) -> str:
        """Write to ``path`` after the fact, and mark this result saved. Returns the path."""
        path = os.path.abspath(os.path.expanduser(str(path)))
        # Read FIRST. `open(..., "wb")` truncates on open, so when `path` is the file we are
        # already saved to, reading afterwards returns b"" and the only copy is destroyed.
        payload = self.read()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(payload)
        # Becoming saved is the point of the call: `.saved`, `os.fspath()` and the over-cap
        # display must all agree with the file that now exists. `data` is released so the
        # documented invariant (bytes held ONLY while unsaved) stays true and a later read cannot
        # return a stale copy of a file that has since changed.
        self.path = path
        self.data = None
        return path

    def show(self) -> "VideoInfo":
        """Show the video — inline in a notebook, or in a player from a terminal.

        The matplotlib contract: a figure appears inline in a notebook, and ``plt.show()`` opens
        a window from a script. Same idea, so ``r.video().show()`` works everywhere.

        A terminal needs a real file, so an unsaved render is written to a temp file first (the
        path is reported). Returns self, so it chains.
        """
        if _in_notebook():
            from IPython.display import display
            display(self)
            return self

        path = self.path
        if path is None:                       # a player needs something on disk
            fd, path = tempfile.mkstemp(prefix="cardiac_", suffix=f".{_EXT_FOR_CODEC.get(self.codec, 'mp4')}")
            os.close(fd)
            with open(path, "wb") as fh:
                fh.write(self.read())

        if not _open_externally(path):
            print(f"No video player could be opened (headless or remote session?).\n"
                  f"The video is at: {path}")
        return self

    def __fspath__(self) -> str:
        if self.path is None:
            raise TypeError(
                "this render was not written to a file, so it has no path — pass `path=...` to "
                "save it, or call `.save('out.mp4')` on the result."
            )
        return self.path

    def __str__(self) -> str:
        if self.path is None:
            return (f"<video {self.width}x{self.height}, {self.n_frames} frames, "
                    f"{self.duration_s:.1f}s — not saved (pass path= to write a file)>")
        return self.path

    def _repr_html_(self) -> str:
        mime = _MIME.get(self.codec)
        if mime is None:                       # mp4v — a real file no browser will decode
            return (
                f"<pre>{self.width}x{self.height}, {self.n_frames} frames — encoded with "
                f"<b>{self.codec}</b> ({self.backend}), which browsers cannot play inline.\n"
                f"Install an H.264 encoder for inline playback:  "
                f"<code>pip install imageio-ffmpeg</code></pre>"
            )
        # Gate on the recorded size FIRST — reading a 400 MB file into RAM only to decline to
        # embed it is the opposite of what the cap is for.
        if self.size_bytes > INLINE_MAX_BYTES:
            where = self.path or "(not saved — call .save('out.mp4') to keep it)"
            return (
                f"<pre>{self.width}x{self.height}, {self.n_frames} frames, "
                f"{self.size_bytes / 1e6:.1f} MB — too large to embed in a notebook "
                f"(limit {INLINE_MAX_BYTES / 1e6:.0f} MB).\nSaved at: {where}\n"
                f"Reduce with max_frames= or resolution=.</pre>"
            )
        try:
            payload = self.read()
        except OSError as exc:      # the saved file was moved/deleted since the render
            return f"<pre>video unavailable: {exc}</pre>"
        import base64
        b64 = base64.b64encode(payload).decode("ascii")
        if mime == "image/gif":
            return f'<img src="data:{mime};base64,{b64}" alt="simulation animation">'
        return (
            f'<video controls loop muted playsinline style="max-width:100%">'
            f'<source src="data:{mime};base64,{b64}" type="{mime}">'
            f'</video>'
        )


def _importable(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def select_backend(fmt: str) -> Tuple[str, str, str]:
    """Choose the encoder BEFORE the path is built. Returns ``(backend, ext, kind)``.

    ``kind`` is the ``media_path`` directory — a GIF is an *image* in the repo convention.
    """
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"format must be one of {SUPPORTED_FORMATS}, got {fmt!r}. "
            f"(media_path also permits 'mov'/'avi', but they are deliberately not exposed here — "
            f"they would reach the writer with an incompatible codec.)"
        )

    if fmt == "gif":
        return "pillow-gif", "gif", "images"

    if _importable("imageio") and _importable("imageio_ffmpeg"):
        return "imageio-ffmpeg", fmt, "videos"

    if fmt == "webm":
        warnings.warn(
            "no ffmpeg backend available; webm cannot be produced by OpenCV — "
            "DOWNGRADING to animated GIF.", UserWarning, stacklevel=2,
        )
        return "pillow-gif", "gif", "images"

    if _importable("cv2"):
        warnings.warn(
            "imageio-ffmpeg unavailable; falling back to the OpenCV 'mp4v' encoder. The .mp4 is "
            "valid but uses MPEG-4 Part 2, which browsers CANNOT play — it will not display "
            "inline in Jupyter/Colab and may not open in a web player. Install an H.264 encoder "
            "with `pip install imageio-ffmpeg` (or `pip install cardiac-core[viz]`).",
            UserWarning, stacklevel=2,
        )
        return "opencv", "mp4", "videos"

    warnings.warn(
        "no video encoder available (imageio-ffmpeg and OpenCV both missing) — "
        "DOWNGRADING to animated GIF.", UserWarning, stacklevel=2,
    )
    return "pillow-gif", "gif", "images"


def resolve_canvas(resolution) -> Optional[Tuple[int, int]]:
    """``"1080p"`` / ``(w, h)`` -> an even ``(w, h)``; ``None`` -> ``None`` (skip fitting)."""
    if resolution is None:
        return None
    if isinstance(resolution, str):
        key = resolution.lower()
        if key not in RESOLUTIONS:
            raise ValueError(
                f"resolution must be one of {sorted(RESOLUTIONS)} or an explicit (w, h), "
                f"got {resolution!r}"
            )
        w, h = RESOLUTIONS[key]
    else:
        w, h = resolution
    return (int(w) + int(w) % 2, int(h) + int(h) % 2)


def fit_frame(rgb: np.ndarray, canvas, fit: str = "contain",
              interpolation: str = "nearest", pad=(0, 0, 0)) -> np.ndarray:
    """Scale ``rgb`` onto ``canvas`` without ever distorting aspect (unless ``fit="stretch"``)."""
    from PIL import Image

    if canvas is None:
        return rgb
    W, H = canvas
    h, w = rgb.shape[:2]

    if fit == "stretch":
        sx, sy = W / w, H / h
    elif fit == "cover":
        s = max(W / w, H / h)
        sx = sy = s
    elif fit == "contain":
        s = min(W / w, H / h)
        sx = sy = s
    else:
        raise ValueError(f"fit must be 'contain', 'stretch' or 'cover', got {fit!r}")

    # A degenerate axis (Grid(N,1)) must still be visible.
    if round(h * sy) < 2:
        sy = 2.0 / h
    if round(w * sx) < 2:
        sx = 2.0 / w

    smin = min(sx, sy)
    if interpolation == "nearest" and smin >= 1:
        resample = Image.NEAREST      # crisp + honest about grid resolution
    elif smin < 1:
        resample = Image.BOX          # area-average; nearest DOWNscaling aliases wavefronts
    else:
        resample = Image.BILINEAR

    new_w, new_h = max(1, round(w * sx)), max(1, round(h * sy))
    out = np.asarray(Image.fromarray(rgb).resize((new_w, new_h), resample))

    if fit == "cover":
        y0 = max(0, (out.shape[0] - H) // 2)
        x0 = max(0, (out.shape[1] - W) // 2)
        return np.ascontiguousarray(out[y0:y0 + H, x0:x0 + W])

    # Rounding can overshoot the canvas by a pixel — crop, then pad symmetrically.
    out = out[:H, :W]
    ph, pw = H - out.shape[0], W - out.shape[1]
    if ph or pw:
        top, left = ph // 2, pw // 2
        canvas_arr = np.empty((H, W, 3), dtype=np.uint8)
        canvas_arr[:, :] = np.asarray(pad, dtype=np.uint8)
        canvas_arr[top:top + out.shape[0], left:left + out.shape[1]] = out
        out = canvas_arr
    return np.ascontiguousarray(out)


def burn_timestamp(rgb: np.ndarray, text: str) -> np.ndarray:
    """Draw ``text`` into the top-left. Call AFTER the canvas fit, or it is drawn at grid scale."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import matplotlib
    except Exception:                                    # pragma: no cover - defensive
        return rgb

    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    size = max(8, img.height // 40)                      # clamp: a grid-sized frame would give 0
    try:
        font_path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
        font = ImageFont.truetype(font_path, size)       # PIL's default bitmap font is ~8 px
    except Exception:                                    # pragma: no cover - defensive
        font = ImageFont.load_default()

    pad = max(3, size // 3)
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def _even(frame: np.ndarray) -> np.ndarray:
    """H.264/yuv420p needs even width & height — pad one edge pixel when odd."""
    h, w = frame.shape[:2]
    ph, pw = h % 2, w % 2
    if ph or pw:
        frame = np.pad(frame, ((0, ph), (0, pw), (0, 0)), mode="edge")
    return frame


class _Writer:
    """Uniform ``append(rgb)`` / ``close()`` over the three backends."""

    def __init__(self, path: str, fps: float, backend: str, fmt: str,
                 quality: int = 8, bitrate: Optional[str] = None):
        self.path = path
        self.fps = float(fps)
        self.backend = backend
        self.bitrate = bitrate
        self.width = 0
        self.height = 0
        self._impl = None
        self._frames = None
        self._closed = False
        # True once bytes have actually reached `path`. Streaming backends create the
        # file on the FIRST append; pillow-gif not until close(). Constructing a
        # writer touches nothing, so `writer is not None` is not this.
        self.touched = False

        codec, pixfmt = CODECS.get(fmt, (None, None))

        if backend == "imageio-ffmpeg":
            import imageio.v2 as iio
            kwargs = dict(format="FFMPEG", mode="I", fps=self.fps, codec=codec,
                          quality=quality, macro_block_size=1)
            if pixfmt:
                kwargs["pixelformat"] = pixfmt
            if bitrate:
                kwargs["bitrate"] = bitrate
            self._impl = iio.get_writer(path, **kwargs)
            self.codec = codec
        elif backend == "opencv":
            import cv2
            self._cv2 = cv2
            self.codec = "mp4v"
        elif backend == "pillow-gif":
            # PIL ONLY — never imageio. This backend is selected precisely WHEN imageio is
            # unimportable, so an imageio-based implementation would crash in the only
            # environment that chooses it.
            self._frames = []
            self.codec = "gif"
        else:                                            # pragma: no cover - defensive
            raise ValueError(f"unknown backend {backend!r}")

    def append(self, rgb: np.ndarray) -> None:
        rgb = _even(np.ascontiguousarray(rgb, dtype=np.uint8))
        self.height, self.width = rgb.shape[0], rgb.shape[1]   # PADDED dims are the real ones
        if self.backend == "imageio-ffmpeg":
            self._impl.append_data(rgb)
            self.touched = True
        elif self.backend == "opencv":
            if self._impl is None:
                fourcc = self._cv2.VideoWriter_fourcc(*"mp4v")
                self._impl = self._cv2.VideoWriter(
                    self.path, fourcc, self.fps, (self.width, self.height))
                if not self._impl.isOpened():
                    raise RuntimeError(f"OpenCV could not open a writer for {self.path}")
            self._impl.write(rgb[:, :, ::-1])              # OpenCV wants BGR
            self.touched = True
        else:
            from PIL import Image
            self._frames.append(Image.fromarray(rgb))

    def abort(self) -> None:
        """Release the encoder WITHOUT finalizing the output — the ERROR path.

        ``close()`` is a WRITE for the buffered backend: pillow-gif holds every frame and emits
        the whole file here, so calling it on the error path creates the very file we are trying
        not to touch (it can overwrite a caller's pre-existing ``path=``). Buffered frames are
        dropped instead. Streaming backends (the ffmpeg subprocess, OpenCV) have already put
        bytes on disk and cannot un-write them; they are only released.

        Never raises: it runs inside an ``except`` block and must not replace the exception
        being propagated.
        """
        if self._closed:
            return
        self._closed = True
        self._frames = None            # discarded, NOT written
        try:
            if self.backend == "imageio-ffmpeg" and self._impl is not None:
                self._impl.close()
            elif self.backend == "opencv" and self._impl is not None:
                self._impl.release()
        except BaseException:
            pass
        finally:
            self._impl = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.backend == "imageio-ffmpeg" and self._impl is not None:
                self._impl.close()
            elif self.backend == "opencv" and self._impl is not None:
                self._impl.release()
            elif self.backend == "pillow-gif" and self._frames:
                self.touched = True          # the whole file is written by the call below
                duration = max(1, round(1000.0 / self.fps))
                self._frames[0].save(self.path, save_all=True, append_images=self._frames[1:],
                                     duration=duration, loop=0, optimize=False)
        except BaseException:
            # `_closed` is already set, so the caller's recovery close() short-circuits and would
            # never retry this. Drop the handles so the ffmpeg subprocess / OpenCV writer is
            # released by finalization instead of leaking for the life of the process.
            self._impl = None
            self._frames = None
            raise


def open_writer(path: str, fps: float, backend: str, fmt: str, *,
                quality: int = 8, bitrate: Optional[str] = None) -> _Writer:
    """Open a writer for an already-selected backend. ``fmt`` drives codec + pixel format."""
    return _Writer(path, fps, backend, fmt, quality=quality, bitrate=bitrate)
