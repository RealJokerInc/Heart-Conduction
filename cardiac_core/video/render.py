"""``render()`` — turn a :class:`Video` spec into a file.

The ordered sequence in :func:`render` is load-bearing and must not be reordered::

    enforce -> select_backend -> stride (+ GIF cap) -> resolve -> path -> writer -> loop

Backend selection precedes stride because the GIF frame cap depends on it; the path follows the
backend because ext/kind come from it.

Two producers: ``bare`` colormaps the array directly with no matplotlib figure (~0.1 ms/frame),
``figure`` builds the axes once and swaps data per frame (~8 ms/frame). The common case is the
fast one.
"""

from __future__ import annotations

import math
import os
import tempfile
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")     # headless — video rendering is for scripts/CI, never a GUI

import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402

from ..media import media_path, slugify   # noqa: E402
from .clip import Video, _to_numpy        # noqa: E402
from .encoders import (             # noqa: E402
    GIF_MAX_FRAMES, ImagePath, VideoInfo, fit_frame, burn_timestamp, open_writer,
    resolve_canvas, select_backend,
)

__all__ = ["render", "render_video", "preview_frame"]

_PAD_BLACK = (0, 0, 0)    # not the masked grey — padding must not read as inactive tissue
_LEGAL_FIT = ("contain", "stretch", "cover")


class _Unset:
    """Sentinel: 'the caller passed nothing'.

    A literal default cannot express this. `resolution` defaults to "1080p", so keying the
    figsize/dpi conflict on `resolution is not None` made every plain
    ``render(v, figsize=(6, 3))`` raise. Mirrors ``image/_draw.py``'s sentinel.
    """

    def __repr__(self) -> str:                       # pragma: no cover - cosmetic
        return "<unset>"


_UNSET = _Unset()
_DEFAULT_RESOLUTION = "1080p"
_DEFAULT_FIT = "contain"

# Extension -> render format, for deriving `format` from an explicit `path=`.
_EXT_FORMAT = {".mp4": "mp4", ".webm": "webm", ".gif": "gif"}


def _resolve_format(fmt: Optional[str], path: Optional[str]) -> str:
    """An explicit ``format=`` wins; otherwise take it from ``path=``'s extension."""
    if fmt is not None:
        return fmt
    if path is not None:
        ext = os.path.splitext(str(path))[1].lower()
        if ext in _EXT_FORMAT:
            return _EXT_FORMAT[ext]
        if ext:
            raise ValueError(
                f"cannot infer a video format from {ext!r} (path={path!r}); "
                f"use .mp4/.webm/.gif or pass format= explicitly"
            )
    return "mp4"


def _named_destination(question, bulk, date, root) -> bool:
    """Did the caller name a ``media/`` convention destination?

    Saving follows matplotlib: rendering displays, ``path=`` (or these convention keywords, which
    name a destination just as explicitly) writes a file.
    """
    return any(x is not None for x in (question, bulk, date, root))


# `bulk` defaults to True when only the OTHER convention keywords are given: the overwhelmingly
# common case is regenerable output, and the gitignored `_sim_outputs/` subtree is where that
# belongs. Pass bulk=False for a curated figure meant to be committed.
_BULK_DEFAULT = True


def _resolve_destination(slug: str, kind: str, ext: str, *, path, question, bulk,
                         date, root) -> Tuple[str, bool, bool]:
    """Return ``(filesystem_path, is_temporary, owned)``.

    Every backend writes through a real filename (ffmpeg is a subprocess, OpenCV a C writer), so
    an unsaved render still needs a file — it just goes to a temp path that is deleted once the
    bytes have been read back.

    ``owned`` is False when ``path=`` names a file that ALREADY EXISTS. The error path must never
    delete such a file: a render can raise before writing a single byte (a validation error, or a
    KeyboardInterrupt on a backend that only writes at close), and removing the caller's existing
    file would destroy data this call never touched.
    """
    if path is not None:
        if _named_destination(question, bulk, date, root):
            warnings.warn(
                "path= and the media/ convention keywords (question=/bulk=/root=/date=) both "
                "name a destination; path= wins and the others are ignored. Pass only one.",
                UserWarning, stacklevel=3)
        p = os.path.abspath(os.path.expanduser(str(path)))
        if os.path.isdir(p):
            raise IsADirectoryError(
                f"path={path!r} is a directory — pass a FILE path, e.g. "
                f"{os.path.join(str(path), 'output.' + ext)!r}")
        stem, given = os.path.splitext(p)
        # The backend can DOWNGRADE (no ffmpeg -> GIF), so the encoder's extension is the only
        # one that describes the bytes. Writing GIF data into a .webm is the silent-format-
        # downgrade defect this subsystem exists to prevent; PIL also refuses the write outright.
        if given and given.lower() != f".{ext}":
            warnings.warn(
                f"writing '{os.path.basename(stem)}.{ext}' rather than the requested '{given}', "
                f"so the file describes its own contents. This follows a backend downgrade (see "
                f"the preceding warning) or a format= that disagrees with the path.",
                UserWarning, stacklevel=3)
        if given.lower() != f".{ext}":
            p = f"{stem}.{ext}"
        parent = os.path.dirname(p)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return p, False, not os.path.lexists(p)

    if _named_destination(question, bulk, date, root):
        # media_path returns the next NN that does not exist, so this is normally ours. The
        # re-check only catches a file that appeared between the two calls; it does NOT close
        # media_path's documented get-path-then-save race, where two concurrent renders of the
        # same slug/day are both handed the SAME name. That race is inherent to the convention.
        mp = media_path(question if question is not None else "lab", kind, slug, ext=ext,
                        bulk=_BULK_DEFAULT if bulk is None else bulk, date=date, root=root)
        return mp, False, not os.path.lexists(mp)

    fd, p = tempfile.mkstemp(prefix=f"{slugify(slug)}_", suffix=f".{ext}")
    os.close(fd)
    return p, True, True


def discard_partial(out_path: str, owned: bool, opened: bool = False) -> None:
    """Remove a half-written output on the error path — but ONLY if this render created it.

    When ``path=`` named a file that already existed, deleting it would destroy data the render
    may never even have opened. ``opened`` says whether a writer was actually handed the path;
    without it the warning would tell a scientist their intact file might be corrupt every time
    a render failed during validation.
    """
    try:
        if not os.path.lexists(out_path):
            return
        if owned:
            os.remove(out_path)
            return
        if opened:
            warnings.warn(
                f"render failed after opening {out_path!r}, which already existed. It was NOT "
                f"deleted, but a streaming encoder may have truncated it — check the file.",
                UserWarning, stacklevel=3)
        else:
            warnings.warn(
                f"render failed before writing anything; {out_path!r} already existed and is "
                f"untouched.", UserWarning, stacklevel=3)
    except BaseException:
        # This runs inside `except BaseException:` before a bare `raise`. Under `-W error` the
        # warn itself raises, and a failing os.remove raises too; either would REPLACE the real
        # exception (a KeyboardInterrupt would surface as a UserWarning). Never mask it.
        pass


def _finalize(out_path: str, is_temp: bool) -> Tuple[Optional[str], Optional[bytes]]:
    """Return ``(path, data)`` — a temp render is read into memory and its file removed."""
    if not is_temp:
        return out_path, None
    try:
        with open(out_path, "rb") as fh:
            data = fh.read()
    finally:
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except OSError:
            pass        # cleaning up must never replace the error that is propagating
    return None, data


@dataclass
class _FigState:
    """Everything ``produce_figure`` mutates per frame. The multi-panel path holds one per panel."""
    fig: Any
    ax: Any
    im: Any
    Xc: Any
    Yc: Any
    contour: Any = None
    suptitle: Any = None


def enforce_capabilities(clip: Video, *, colorbar, show_time, figsize, dpi, title) -> None:
    """Reject figure-only features on a BARE clip, loudly. Never called on the multi-panel path."""
    if clip.style != "bare":
        return
    hint = "Use Video.annotated(...) — the bare producer draws no axes, so it cannot render this."
    if colorbar is True:
        raise ValueError(f"colorbar=True is not available on a bare clip. {hint}")
    if title is not None:
        raise ValueError(f"title= is not available on a bare clip. {hint}")
    if figsize is not None or dpi is not None:
        raise ValueError(
            f"figsize=/dpi= apply to the matplotlib figure only, not a bare clip. {hint}")
    if clip.label is not None:
        raise ValueError(f"label= is a panel title and needs a figure. {hint}")
    if clip.front is not None:
        raise ValueError(f"front= draws a contour and needs a figure. {hint}")
    if clip.isochrones:
        raise ValueError(f"isochrones=True draws contours and needs a figure. {hint}")


def isochrone_lat(clip: Video, idx) -> np.ndarray:
    """Static activation-time map for the isochrone overlay, honouring the mask."""
    from .. import analysis

    idx = list(idx)          # once: the guard below must not consume a generator

    if len(clip.frames) < 2:
        warnings.warn("isochrones need >= 2 frames; skipping the overlay",
                      UserWarning, stacklevel=2)
        return np.full(clip.frames.shape[1:], np.nan)

    is_vm = isinstance(clip.field, str) and clip.field in ("Vm", "V")
    if clip.result is not None and is_vm:
        # torch path is only valid for Vm — result.Vm is NOT the displayed field for
        # field="phi_e" or an explicit array.
        # It reads the FULL history off the result and is independent of `idx`, so a
        # single-index draw (a preview, or max_frames=1) still gets a correct overlay. The
        # `len(idx)` guard below MUST NOT be hoisted above this branch: doing so silently
        # dropped a computable overlay and warned that it was uncomputable.
        lat = _to_numpy(analysis.activation_time(clip.result.Vm, clip.result.times))
        lat = np.asarray(lat, dtype=np.float64)
        if clip.active_mask is not None:
            lat = np.where(clip.active_mask, lat, np.nan)
        return lat

    if len(idx) < 2:
        # Only the numpy branch depends on `idx`: it stacks the DRAWN frames, so a single
        # index yields a constant LAT — an invisible, wrong overlay rather than an absent one.
        warnings.warn("isochrones need >= 2 drawn frames; skipping the overlay",
                      UserWarning, stacklevel=2)
        return np.full(clip.frames.shape[1:], np.nan)

    # Built ONLY here and STRIDED — a full unstrided history stack would blow the memory rule.
    masked = np.stack([clip.display_values(t) for t in idx])
    return np.asarray(
        analysis.activation_time_interp(masked, clip.times[idx], threshold=-40.0),
        dtype=np.float64)


def _extent_and_labels(clip: Video, units_resolved: str):
    Nx, Ny = clip.frames.shape[1], clip.frames.shape[2]
    if units_resolved == "cm" and clip.dx and clip.dy:
        return ([0.0, (Nx - 1) * clip.dx, 0.0, (Ny - 1) * clip.dy], "x (cm)", "y (cm)")
    return ([0, Nx - 1, 0, Ny - 1], "x (nodes)", "y (nodes)")


def _build_figure(clip: Video, cmap, norm, *, colorbar_on: bool, title: Optional[str],
                  figsize, dpi, units, idx,
                  lat=None, contour_levels: int = 12, filled: bool = False) -> _FigState:
    """Build the annotated figure once; the caller swaps data per frame.

    ``lat``/``contour_levels``/``filled`` are additive and default to the historical behaviour:

    * ``lat`` — a precomputed ``(Nx, Ny)`` activation map. When given it is used directly and
      ``isochrone_lat`` is NOT called, which is what lets a one-frame clip draw isochrones at all
      (``isochrone_lat`` refuses a clip with fewer than 2 frames, and refuses fewer than 2
      DRAWN indices on its numpy branch).
    * ``contour_levels`` — replaces the previously hard-coded ``levels=12``.
    * ``filled`` — draw ``contourf`` bands instead of an image. The ``QuadContourSet`` is stored on
      ``_FigState.im`` because it IS the colorbar mappable; ``fig.colorbar(None, ...)`` does not
      raise, it silently fabricates a meaningless 0..1 scale.
    """
    units_resolved = units or clip.units
    if units_resolved == "auto":
        units_resolved = "cm" if (clip.dx and clip.dy) else "nodes"
    extent, xlab, ylab = _extent_and_labels(clip, units_resolved)

    Nx, Ny = clip.frames.shape[1], clip.frames.shape[2]
    if figsize is None:
        span_x = extent[1] - extent[0] or 1.0
        span_y = extent[3] - extent[2] or 1.0
        h = min(max(6.0 * float(span_y) / float(span_x), 1.8), 7.0)
        figsize = (6.0 + 1.6, h + 1.2)
    fig, ax = plt.subplots(figsize=tuple(figsize), dpi=(dpi or 100))
    try:
        return _populate_figure(fig, ax, clip, cmap, norm, colorbar_on=colorbar_on,
                                title=title, extent=extent, xlab=xlab,
                                ylab=ylab, Nx=Nx, Ny=Ny, idx=idx, lat=lat,
                                contour_levels=contour_levels, filled=filled)
    except BaseException:
        # Everything below can fail (contourf on <2 levels, colorbar, isochrone_lat, tight_layout).
        # The caller only closes the figure it receives, so a raise here would leave this one
        # registered in pyplot's Gcf for the life of the process.
        plt.close(fig)
        raise


def _populate_figure(fig, ax, clip: Video, cmap, norm, *, colorbar_on, title,
                     extent, xlab, ylab, Nx, Ny, idx, lat, contour_levels, filled) -> _FigState:
    """Fill an already-created figure. Split out so :func:`_build_figure` can close it on error."""
    # Contour coordinates MUST come from the SAME extent, or a cm-space contour lands on a
    # node-index axis (every .npz/array clip defaults to "nodes").
    x = np.linspace(extent[0], extent[1], Nx)
    y = np.linspace(extent[2], extent[3], Ny)
    Xc, Yc = np.meshgrid(x, y, indexing="ij")     # pairs with the UNtransposed array

    if filled:
        # Filled bands ARE the map: no image beneath, and the contour set is the mappable.
        vals = clip.display_values(idx[0])
        im = None
        if np.isfinite(vals).any():               # mirrors viz.activation_isochrones' guard
            im = ax.contourf(Xc, Yc, np.ma.masked_invalid(vals),
                             levels=contour_levels, cmap=cmap, norm=norm)
        ax.set_aspect(clip.aspect)                # `aspect` otherwise only reaches mpl via imshow
    else:
        im = ax.imshow(
            np.ma.masked_invalid(clip.display_values(idx[0]).T),
            origin="lower",          # MUST be explicit: the bare producer's flipud(.T) is pinned
                                     # by a test, and a mismatch here would flip one producer only
            extent=extent, aspect=clip.aspect, cmap=cmap, norm=norm,
            interpolation=clip.gradient.interpolation,
        )
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if colorbar_on and im is not None:
        fig.colorbar(im, ax=ax, label=clip.value_label)
    if clip.label:
        ax.set_title(clip.label)

    if lat is not None or clip.isochrones:
        lat_arr = lat if lat is not None else isochrone_lat(clip, idx)
        if np.isfinite(lat_arr).any():            # mirrors viz.activation_isochrones
            ax.contour(Xc, Yc, np.ma.masked_invalid(lat_arr), levels=contour_levels,
                       colors="white", linewidths=0.6, alpha=0.55)

    sup = fig.suptitle(title or "")
    fig.tight_layout()
    return _FigState(fig=fig, ax=ax, im=im, Xc=Xc, Yc=Yc, contour=None, suptitle=sup)


def _produce_bare(clip: Video, t: int, cmap, norm) -> np.ndarray:
    # flipud(V.T) is pixel-identical to imshow(V.T, origin="lower") — verified.
    a = np.flipud(clip.display_values(t).T)
    rgba = cmap(norm(np.ma.masked_invalid(a)))
    return (np.asarray(rgba)[..., :3] * 255.0).astype(np.uint8)


def _produce_figure(st: _FigState, clip: Video, t: int, *, show_time: bool,
                    title: Optional[str]) -> np.ndarray:
    # `hasattr`, not `is not None`: in filled mode `st.im` holds a QuadContourSet, which is the
    # colorbar mappable but has no set_data. Video always sets a real AxesImage.
    if hasattr(st.im, "set_data"):
        st.im.set_data(np.ma.masked_invalid(clip.display_values(t).T))

    if clip.front is not None:
        if st.contour is not None:
            try:
                st.contour.remove()                        # mpl >= 3.7
            except AttributeError:                         # pragma: no cover - older mpl
                for coll in getattr(st.contour, "collections", []):
                    coll.remove()
        st.contour = st.ax.contour(
            st.Xc, st.Yc, clip.display_values(t), levels=[clip.front],
            colors="white", linewidths=1.4)

    stamp = f"t = {clip.times[t]:.1f} ms"
    if show_time:
        st.suptitle.set_text(f"{title} — {stamp}" if title else stamp)
    elif title:
        st.suptitle.set_text(title)

    st.fig.canvas.draw()
    return np.asarray(st.fig.canvas.buffer_rgba())[..., :3].copy()


def render(video: Union[Video, Sequence[Video]], slug: str = "video", *,
           path: Optional[str] = None,
           question: Optional[str] = None, bulk: Optional[bool] = None,
           resolution: Any = _UNSET, fit: Any = _UNSET,
           fps: float = 20.0, speed: Optional[float] = None,
           max_frames: Optional[int] = 300, format: Optional[str] = None,
           bitrate: Optional[str] = None,
           show_time: Optional[bool] = None, colorbar: Optional[bool] = None,
           title: Optional[str] = None,
           figsize: Optional[Sequence[float]] = None, dpi: Optional[int] = None,
           units: Optional[str] = None, progress: bool = False,
           labels: Optional[Sequence[str]] = None,
           rows: Optional[int] = None, cols: Optional[int] = None,
           date: Optional[str] = None, root: Optional[str] = None) -> VideoInfo:
    """Render a :class:`Video` (or a LIST of them) and return a displayable :class:`VideoInfo`.

    **Rendering displays; naming a destination saves** — the matplotlib contract. With no
    destination the result plays inline in a notebook and no file is left behind. A file is
    written when you say where it goes::

        render(Video(r))                            # displays; nothing on disk
        render(Video(r), path="out.mp4")            # writes ./out.mp4
        render(Video(r), "wave", question="lab")    # media/lab/_sim_outputs/videos/{date}/wave_01.mp4
        render(Video(r), "wave", bulk=False)        # media/lab/videos/{date}/wave_01.mp4

    ``bulk`` defaults to **True** whenever any convention keyword is used — regenerable output
    belongs in the gitignored ``_sim_outputs/`` subtree. Pass ``bulk=False`` for a curated figure
    meant to be committed. ``path=`` ignores the convention entirely and warns if both are given.

    ``format`` follows ``path``'s extension when not given explicitly.

    ``speed`` is in SIMULATION MILLISECONDS PER REAL SECOND and overrides ``fps``.

    Passing a list renders N panels sharing ONE colorbar and ONE time stamp; ``labels``,
    ``rows`` and ``cols`` apply to that path only. Panels must share a grid and a field kind;
    the frame count truncates to the shortest clip. Bare clips are promoted to the figure
    producer (with a warning) because a shared colorbar needs axes.
    """
    fmt = _resolve_format(format, path)    # `format` shadows the builtin; bind locally
    # Only an EXPLICIT resolution=/fit= can conflict with figsize=/dpi=; the defaults must not.
    resolution = _DEFAULT_RESOLUTION if resolution is _UNSET else resolution
    fit = _DEFAULT_FIT if fit is _UNSET else fit
    if fit not in _LEGAL_FIT:
        raise ValueError(f"fit must be one of {_LEGAL_FIT}, got {fit!r}")

    # 1. validate + capability gate ------------------------------------------------
    # NOTE: the figsize/dpi conflict above is checked BEFORE this dispatch, so the panel path
    # inherits it — resolution/fit reach _render_panels already resolved to concrete values.
    if isinstance(video, (list, tuple)):
        return _render_panels(
            list(video), slug, path=path, question=question, bulk=bulk,
            resolution=resolution, fit=fit,
            fps=fps, speed=speed, max_frames=max_frames, fmt=fmt, bitrate=bitrate,
            show_time=show_time, colorbar=colorbar, title=title, figsize=figsize, dpi=dpi,
            units=units, progress=progress, date=date, root=root,
            labels=labels, rows=rows, cols=cols)
    clip = video
    enforce_capabilities(clip, colorbar=colorbar, show_time=show_time,
                         figsize=figsize, dpi=dpi, title=title)
    show_time_resolved = show_time if show_time is not None else (clip.style == "annotated")
    colorbar_resolved = colorbar if colorbar is not None else (clip.style == "annotated")

    # 2. backend FIRST (the GIF cap and the path both depend on it) ----------------
    backend, ext, kind = select_backend(fmt)

    # 3. stride, with the backend-dependent GIF cap -------------------------------
    T = len(clip.frames)
    eff_max = max_frames
    if backend == "pillow-gif":
        eff_max = GIF_MAX_FRAMES if eff_max is None else min(eff_max, GIF_MAX_FRAMES)
    stride = math.ceil(T / eff_max) if (eff_max and T > eff_max) else 1
    idx = list(range(0, T, stride))

    # 4. colour range, over MASKED values, AFTER striding -------------------------
    cmap, norm, lo, hi = clip.gradient.resolve(clip.masked_iter(idx), field=clip.field)

    # 5. playback rate -------------------------------------------------------------
    if speed is not None:
        d = np.diff(clip.times[idx])
        dt = float(np.median(d)) if d.size else 1.0
        raw = float(speed) / max(dt, 1e-12)
        fps = min(max(raw, 1.0), 240.0)
        if abs(fps - raw) > 1e-9:
            warnings.warn(
                f"requested speed implies {raw:.1f} fps; clamped to {fps:.1f} — playback rate "
                f"will not match `speed`", UserWarning, stacklevel=2)
    fps = float(max(1.0, fps))

    # 6. canvas + rate BEFORE the path — resolve_canvas() raises on a bad resolution=, and
    #    _resolve_destination has a side effect (it creates a temp file). Anything that can fail
    #    must fail before we own a file to clean up.
    canvas = resolve_canvas(resolution) if (figsize is None and dpi is None) else None
    if fmt == "webm" and bitrate is None:
        bitrate = "2M"     # VP9 has no `quality` mapping; without a rate ffmpeg silently uses CRF 32
    use_figure = clip.requires_figure()

    out_path, is_temp, owned = _resolve_destination(slug, kind, ext, path=path,
                                                    question=question, bulk=bulk,
                                                    date=date, root=root)

    # 7. stream — writer AND figure construction go INSIDE the guarded region -------
    n = 0
    writer = None
    st = None
    try:
        writer = open_writer(out_path, fps, backend, fmt, quality=8, bitrate=bitrate)
        st = (_build_figure(clip, cmap, norm, colorbar_on=colorbar_resolved, title=title,
                            figsize=figsize, dpi=dpi, units=units, idx=idx)
              if use_figure else None)
        for k, t in enumerate(idx):
            if use_figure:
                rgb = _produce_figure(st, clip, t, show_time=show_time_resolved, title=title)
            else:
                rgb = _produce_bare(clip, t, cmap, norm)
            if canvas is not None:
                rgb = fit_frame(rgb, canvas, fit, clip.gradient.interpolation, _PAD_BLACK)
            # The time stamp is drawn EXACTLY ONCE, by whichever producer owns it:
            #   figure -> fig.suptitle (vector text);  bare -> burned here, AFTER the fit.
            if show_time_resolved and not use_figure:
                rgb = burn_timestamp(rgb, f"t = {clip.times[t]:.1f} ms")
            writer.append(rgb)
            n += 1
            if progress and k % 50 == 0:
                print(f"  ... {k}/{len(idx)} frames", flush=True)
        # INSIDE the guard: the pillow-gif backend writes the ENTIRE file here, so a failure at
        # close would otherwise leave a partial file behind and consume the media_path NN slot.
        writer.close()
        # Inside the guard (matching the image layer) but non-fatal: a getsize() failure on an
        # otherwise SUCCESSFUL render must not send a good output to discard_partial.
        try:
            size = os.path.getsize(writer.path) if os.path.exists(writer.path) else 0
        except OSError:
            size = 0
    except BaseException:
        if writer is not None:
            writer.abort()      # release WITHOUT flushing — close() would write the file
        discard_partial(out_path, owned, opened=bool(writer is not None and writer.touched))
        raise
    finally:
        if st is not None:
            plt.close(st.fig)   # else the suite leaks a figure per render()

    final_path, data = _finalize(writer.path, is_temp)
    return VideoInfo(path=final_path, n_frames=n, fps=fps, backend=writer.backend,
                     codec=writer.codec, width=writer.width, height=writer.height,
                     duration_s=(n / fps if fps else 0.0), vmin=lo, vmax=hi, stride=stride,
                     size_bytes=size, bitrate=writer.bitrate, data=data)


render_video = render      # alias so the _LAZY export name resolves


def preview_frame(video: Video, t_ms: Optional[float] = None, *, frame: Optional[int] = None,
                  slug: str = "preview", path: Optional[str] = None,
                  question: Optional[str] = None, bulk: Optional[bool] = None,
                  units: Optional[str] = None, title: Optional[str] = None,
                  figsize=None, dpi=None, date=None, root=None) -> ImagePath:
    """Render ONE frame through the clip's OWN producer (PNG unless ``path=`` says otherwise).

    Displays inline; writes a file only when a destination is named (``path=`` or the ``media/``
    convention keywords). The return value is still the path string when one was written.

    Delegates to :func:`cardiac_core.image.draw`. Two arguments are load-bearing for keeping the
    output pixel-identical: ``resolution=None`` (the image layer's default upscales a bare still)
    and ``dpi`` passed RAW so ``enforce_capabilities`` still sees the caller's ``None`` on a bare
    clip and ``draw`` applies the historical ``dpi or 100`` afterwards. The format is PNG unless
    ``path=`` names another one — forcing ``"png"`` there would make ``preview(path='f.jpg')`` a
    dead end, since the format/path disagreement raises.
    """
    if t_ms is not None and frame is not None:
        raise ValueError("pass t_ms= or frame=, not both")

    if frame is not None:
        t = int(frame)
    elif t_ms is not None:
        t = int(np.argmin(np.abs(video.times - float(t_ms))))
    else:
        t = len(video.frames) // 2
    if not (0 <= t < len(video.frames)):
        raise IndexError(f"frame {t} out of range for {len(video.frames)} frames")

    from ..image._draw import draw          # function-local: image/ imports video/ at module scope
    info = draw(video, slug, frame=t, show_time=True, resolution=None,
                format=("png" if path is None else None),
                path=path, question=question, bulk=bulk, date=date, root=root,
                units=units, title=title, figsize=figsize, dpi=dpi)
    return ImagePath(info.path, info.data, info.format)


def _default_layout(n: int):
    """2 -> side-by-side; 3 -> stacked; 4 -> 2x2; 5+ -> stacked.

    The 4-panel prior art (``video_boundary_modes.py``) is ``subplots(1, 4)`` with per-panel
    colorbars; we default to 2x2 instead because four panels across a 1920-wide canvas give each
    ~480 px and the wavefront becomes unreadable. ``cols=4`` restores the original arrangement.
    """
    if n == 1:
        return 1, 1
    if n == 2:
        return 1, 2
    if n == 4:
        return 2, 2
    return n, 1


def _setup_panel(clip: Video, ax, cmap, norm, *, units, idx, label=None,
                 lat=None, contour_levels: int = 12, filled: bool = False) -> _FigState:
    """Configure ONE axes for a panel and return its per-frame state carrier.

    ``lat``/``contour_levels``/``filled`` mirror :func:`_build_figure` exactly — the multi-panel
    path draws through here, so stopping the seam at ``_build_figure`` would leave a panelled
    activation map contour-free with no warning.
    """
    units_resolved = units or clip.units
    if units_resolved == "auto":
        units_resolved = "cm" if (clip.dx and clip.dy) else "nodes"
    extent, xlab, ylab = _extent_and_labels(clip, units_resolved)
    Nx, Ny = clip.frames.shape[1], clip.frames.shape[2]

    x = np.linspace(extent[0], extent[1], Nx)
    y = np.linspace(extent[2], extent[3], Ny)
    Xc, Yc = np.meshgrid(x, y, indexing="ij")

    if filled:
        vals = clip.display_values(idx[0])
        im = None
        if np.isfinite(vals).any():
            im = ax.contourf(Xc, Yc, np.ma.masked_invalid(vals),
                             levels=contour_levels, cmap=cmap, norm=norm)
        ax.set_aspect(clip.aspect)
    else:
        im = ax.imshow(
            np.ma.masked_invalid(clip.display_values(idx[0]).T),
            origin="lower", extent=extent, aspect=clip.aspect, cmap=cmap, norm=norm,
            interpolation=clip.gradient.interpolation,
        )
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if label:
        ax.set_title(label, fontsize=10)

    if lat is not None or clip.isochrones:
        lat_arr = lat if lat is not None else isochrone_lat(clip, idx)
        if np.isfinite(lat_arr).any():
            ax.contour(Xc, Yc, np.ma.masked_invalid(lat_arr), levels=contour_levels,
                       colors="white", linewidths=0.6, alpha=0.55)

    return _FigState(fig=None, ax=ax, im=im, Xc=Xc, Yc=Yc, contour=None, suptitle=None)


def _render_panels(clips: List[Video], slug: str, *, path, question, bulk, resolution, fit, fps,
                   speed, max_frames, fmt, bitrate, show_time, colorbar, title, figsize, dpi,
                   units, progress, date, root, labels=None, rows=None, cols=None) -> VideoInfo:
    """N panels sharing ONE colorbar and ONE time stamp."""
    if not clips:
        raise ValueError("render() needs at least one Video")

    # --- compatibility -----------------------------------------------------------
    shapes = {c.frames.shape[1:] for c in clips}
    if len(shapes) > 1:
        raise ValueError(
            f"all panels must share a grid; got shapes {sorted(shapes)}. "
            f"Panels are paired frame-by-frame, so differing grids cannot be compared.")
    kinds = {(c.field if isinstance(c.field, str) else "<array>") for c in clips}
    if len(kinds) > 1:
        raise ValueError(
            f"all panels must show the same field kind; got {sorted(kinds)}. "
            f"A shared colorbar across different fields would be meaningless.")

    # Do NOT mutate the caller's Video objects: `labels=` is a render-time override, and a
    # persisted label would leak into later renders (and would make a bare single-clip render
    # of the same object raise, since label= is figure-only).
    panel_labels = [(labels[i] if (labels and i < len(labels)) else c.label)
                    for i, c in enumerate(clips)]

    # enforce_capabilities is deliberately NOT called here: bare clips are PROMOTED to the
    # figure producer rather than rejected (calling it would raise on colorbar=True first).
    if any(c.style == "bare" for c in clips):
        warnings.warn(
            "multi-panel rendering always uses the figure producer; bare clips were promoted "
            "to annotated for layout (they gain axes and a shared colorbar).",
            UserWarning, stacklevel=3)
    show_time_resolved = True if show_time is None else bool(show_time)
    colorbar_resolved = True if colorbar is None else bool(colorbar)

    # --- backend, stride, one idx for ALL panels ---------------------------------
    backend, ext, kind = select_backend(fmt)
    T = min(len(c.frames) for c in clips)
    eff_max = max_frames
    if backend == "pillow-gif":
        eff_max = GIF_MAX_FRAMES if eff_max is None else min(eff_max, GIF_MAX_FRAMES)
    stride = math.ceil(T / eff_max) if (eff_max and T > eff_max) else 1
    idx = list(range(0, T, stride))

    dts = []
    for c in clips:
        d = np.diff(c.times[:T])
        dts.append(float(np.median(d)) if d.size else 1.0)
    if max(dts) - min(dts) > 1e-9:
        warnings.warn(
            f"panels have different save intervals {dts}; frames are paired by INDEX, not by "
            f"time — the shared time stamp follows panel 0.", UserWarning, stacklevel=3)

    # --- colour: shared when every panel carries the same gradient ---------------
    shared = all(c.gradient.key() == clips[0].gradient.key() for c in clips)
    if shared:
        def _pooled():
            for c in clips:
                for f in c.masked_iter(idx):
                    yield f
        cmap, norm, lo, hi = clips[0].gradient.resolve(_pooled(), field=clips[0].field)
        per_panel = [(cmap, norm)] * len(clips)
    else:
        warnings.warn(
            "panels use different gradients and are NOT directly comparable; drawing a colorbar "
            "per panel instead of one shared scale.", UserWarning, stacklevel=3)
        per_panel = [c.gradient.resolve(c.masked_iter(idx), field=c.field)[:2] for c in clips]
        lo, hi = clips[0].gradient.resolve(clips[0].masked_iter(idx), field=clips[0].field)[2:]

    # --- rate + path --------------------------------------------------------------
    if speed is not None:
        d = np.diff(clips[0].times[idx])
        dt = float(np.median(d)) if d.size else 1.0
        raw = float(speed) / max(dt, 1e-12)
        fps = min(max(raw, 1.0), 240.0)
    fps = float(max(1.0, fps))
    # Fallible work BEFORE the destination — resolve_canvas() raises on a bad resolution=, and
    # acquiring the destination creates a temp file we would then orphan. (Same rule as render().)
    canvas = resolve_canvas(resolution) if (figsize is None and dpi is None) else None
    if fmt == "webm" and bitrate is None:
        bitrate = "2M"
    out_path, is_temp, owned = _resolve_destination(slug, kind, ext, path=path,
                                                    question=question, bulk=bulk,
                                                    date=date, root=root)

    nrows, ncols = (rows, cols) if (rows and cols) else _default_layout(len(clips))
    if rows and not cols:
        nrows, ncols = rows, math.ceil(len(clips) / rows)
    elif cols and not rows:
        ncols, nrows = cols, math.ceil(len(clips) / cols)
    if figsize is None:
        figsize = (min(6.5 * ncols, 19.0), min(3.6 * nrows, 10.0))

    n = 0
    writer = None
    fig = None
    try:
        writer = open_writer(out_path, fps, backend, fmt, quality=8, bitrate=bitrate)
        fig, axes = plt.subplots(nrows, ncols, figsize=tuple(figsize), dpi=(dpi or 100),
                                 constrained_layout=True, squeeze=False)
        flat = [ax for row in axes for ax in row]
        for ax in flat[len(clips):]:
            ax.set_axis_off()

        states = []
        for c, ax, (cm, nm), lab in zip(clips, flat, per_panel, panel_labels):
            st = _setup_panel(c, ax, cm, nm, units=units, idx=idx, label=lab)
            st.fig = fig
            states.append(st)

        if colorbar_resolved:
            if shared:
                fig.colorbar(states[0].im, ax=flat[:len(clips)],
                             label=clips[0].value_label, shrink=0.75)
            else:
                for c, st in zip(clips, states):
                    fig.colorbar(st.im, ax=st.ax, label=c.value_label, shrink=0.75)

        sup = fig.suptitle(title or "")
        for st in states:
            st.suptitle = sup

        for k, t in enumerate(idx):
            for c, st in zip(clips, states):
                if hasattr(st.im, "set_data"):
                    st.im.set_data(np.ma.masked_invalid(c.display_values(t).T))
                if c.front is not None:
                    if st.contour is not None:
                        try:
                            st.contour.remove()
                        except AttributeError:          # pragma: no cover - older mpl
                            for coll in getattr(st.contour, "collections", []):
                                coll.remove()
                    st.contour = st.ax.contour(
                        st.Xc, st.Yc, c.display_values(t), levels=[c.front],
                        colors="white", linewidths=1.4)
            if show_time_resolved:                       # ONE suptitle for the whole figure
                stamp = f"t = {clips[0].times[t]:.1f} ms"
                sup.set_text(f"{title} — {stamp}" if title else stamp)
            elif title:
                sup.set_text(title)
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            if canvas is not None:
                rgb = fit_frame(rgb, canvas, fit, clips[0].gradient.interpolation, _PAD_BLACK)
            writer.append(rgb)
            n += 1
            if progress and k % 50 == 0:
                print(f"  ... {k}/{len(idx)} frames", flush=True)
        writer.close()          # inside the guard — see the single-panel path
        try:
            size = os.path.getsize(writer.path) if os.path.exists(writer.path) else 0
        except OSError:
            size = 0
    except BaseException:
        if writer is not None:
            writer.abort()      # release WITHOUT flushing — close() would write the file
        discard_partial(out_path, owned, opened=bool(writer is not None and writer.touched))
        raise
    finally:
        if fig is not None:
            plt.close(fig)

    final_path, data = _finalize(writer.path, is_temp)
    return VideoInfo(path=final_path, n_frames=n, fps=fps, backend=writer.backend,
                     codec=writer.codec, width=writer.width, height=writer.height,
                     duration_s=(n / fps if fps else 0.0), vmin=lo, vmax=hi, stride=stride,
                     size_bytes=size, bitrate=writer.bitrate, data=data)
