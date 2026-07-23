"""Encoder backends, canvas geometry, and the :class:`VideoInfo` return type.

Backend selection happens BEFORE the output path is built: a fallback changes both the extension
and the ``kind`` directory, and :func:`cardiac_core.media.media_path` validates ext against kind,
so a writer that discovers its own fallback after the path exists cannot fix it.

No fallback is silent — every downgrade emits a ``UserWarning`` naming the backend actually used,
and the choice is reported on ``VideoInfo.backend``.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "VideoInfo", "RESOLUTIONS", "CODECS",
    "select_backend", "resolve_canvas", "fit_frame", "burn_timestamp", "open_writer",
]

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


@dataclass
class VideoInfo:
    """What :func:`cardiac_core.video.render` produced.

    Str-like: ``str(info)`` and ``os.fspath(info)`` give the path, so it can be passed straight to
    anything taking a filename while still carrying the metadata.
    """

    path: str
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

    def __fspath__(self) -> str:
        return self.path

    def __str__(self) -> str:
        return self.path


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
            "imageio-ffmpeg unavailable; falling back to the OpenCV 'mp4v' encoder.",
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
        elif self.backend == "opencv":
            if self._impl is None:
                fourcc = self._cv2.VideoWriter_fourcc(*"mp4v")
                self._impl = self._cv2.VideoWriter(
                    self.path, fourcc, self.fps, (self.width, self.height))
                if not self._impl.isOpened():
                    raise RuntimeError(f"OpenCV could not open a writer for {self.path}")
            self._impl.write(rgb[:, :, ::-1])              # OpenCV wants BGR
        else:
            from PIL import Image
            self._frames.append(Image.fromarray(rgb))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.backend == "imageio-ffmpeg" and self._impl is not None:
            self._impl.close()
        elif self.backend == "opencv" and self._impl is not None:
            self._impl.release()
        elif self.backend == "pillow-gif" and self._frames:
            duration = max(1, round(1000.0 / self.fps))
            self._frames[0].save(self.path, save_all=True, append_images=self._frames[1:],
                                 duration=duration, loop=0, optimize=False)


def open_writer(path: str, fps: float, backend: str, fmt: str, *,
                quality: int = 8, bitrate: Optional[str] = None) -> _Writer:
    """Open a writer for an already-selected backend. ``fmt`` drives codec + pixel format."""
    return _Writer(path, fps, backend, fmt, quality=quality, bitrate=bitrate)
