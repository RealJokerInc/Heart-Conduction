"""``render()`` — turn a :class:`Video` spec into a file.

The ordered sequence in :func:`render` is load-bearing and must not be reordered::

    enforce -> select_backend -> stride (+ GIF cap) -> resolve -> path -> writer -> loop

Backend selection precedes stride because the GIF frame cap depends on it; the path follows the
backend because ext/kind come from it (and ``media_path`` consumes its ``NN`` slot on call).

Two producers: ``bare`` colormaps the array directly with no matplotlib figure (~0.1 ms/frame),
``figure`` builds the axes once and swaps data per frame (~8 ms/frame). The common case is the
fast one.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")     # headless — video rendering is for scripts/CI, never a GUI

import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402

from ..media import media_path      # noqa: E402
from .clip import Video, _to_numpy  # noqa: E402
from .encoders import (             # noqa: E402
    GIF_MAX_FRAMES, VideoInfo, fit_frame, burn_timestamp, open_writer,
    resolve_canvas, select_backend,
)

__all__ = ["render", "render_video", "preview_frame"]

_PAD_BLACK = (0, 0, 0)    # not the masked grey — padding must not read as inactive tissue
_LEGAL_FIT = ("contain", "stretch", "cover")


@dataclass
class _FigState:
    """Everything ``produce_figure`` mutates per frame. Phase 2 holds one per panel."""
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

    if len(clip.frames) < 2:
        warnings.warn("isochrones need >= 2 frames; skipping the overlay",
                      UserWarning, stacklevel=2)
        return np.full(clip.frames.shape[1:], np.nan)

    is_vm = isinstance(clip.field, str) and clip.field in ("Vm", "V")
    if clip.result is not None and is_vm:
        # torch path is only valid for Vm — result.Vm is NOT the displayed field for
        # field="phi_e" or an explicit array.
        lat = _to_numpy(analysis.activation_time(clip.result.Vm, clip.result.times))
        lat = np.asarray(lat, dtype=np.float64)
        if clip.active_mask is not None:
            lat = np.where(clip.active_mask, lat, np.nan)
        return lat

    # Built ONLY here and STRIDED — a full unstrided history stack would blow the memory rule.
    masked = np.stack([clip.display_values(t) for t in idx])
    return np.asarray(
        analysis.activation_time_interp(masked, clip.times[list(idx)], threshold=-40.0),
        dtype=np.float64)


def _extent_and_labels(clip: Video, units_resolved: str):
    Nx, Ny = clip.frames.shape[1], clip.frames.shape[2]
    if units_resolved == "cm" and clip.dx and clip.dy:
        return ([0.0, (Nx - 1) * clip.dx, 0.0, (Ny - 1) * clip.dy], "x (cm)", "y (cm)")
    return ([0, Nx - 1, 0, Ny - 1], "x (nodes)", "y (nodes)")


def _build_figure(clip: Video, cmap, norm, *, colorbar_on: bool, title: Optional[str],
                  figsize, dpi, units, idx) -> _FigState:
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

    im = ax.imshow(
        np.ma.masked_invalid(clip.display_values(idx[0]).T),
        origin="lower",              # MUST be explicit: the bare producer's flipud(.T) is pinned
                                     # by a test, and a mismatch here would flip one producer only
        extent=extent, aspect=clip.aspect, cmap=cmap, norm=norm,
        interpolation=clip.gradient.interpolation,
    )
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if colorbar_on:
        fig.colorbar(im, ax=ax, label=clip.value_label)
    if clip.label:
        ax.set_title(clip.label)

    # Contour coordinates MUST come from the SAME extent, or a cm-space contour lands on a
    # node-index axis (every .npz/array clip defaults to "nodes").
    x = np.linspace(extent[0], extent[1], Nx)
    y = np.linspace(extent[2], extent[3], Ny)
    Xc, Yc = np.meshgrid(x, y, indexing="ij")     # pairs with the UNtransposed array

    if clip.isochrones:
        lat = isochrone_lat(clip, idx)
        if np.isfinite(lat).any():                # mirrors viz.activation_isochrones
            ax.contour(Xc, Yc, np.ma.masked_invalid(lat), levels=12,
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


def render(video: Union[Video, Sequence[Video]], slug: str, *,
           question: str = "lab", bulk: bool = True,
           resolution: Any = "1080p", fit: str = "contain",
           fps: float = 20.0, speed: Optional[float] = None,
           max_frames: Optional[int] = 300, format: str = "mp4",
           bitrate: Optional[str] = None,
           show_time: Optional[bool] = None, colorbar: Optional[bool] = None,
           title: Optional[str] = None,
           figsize: Optional[Sequence[float]] = None, dpi: Optional[int] = None,
           units: Optional[str] = None, progress: bool = False,
           labels: Optional[Sequence[str]] = None,
           rows: Optional[int] = None, cols: Optional[int] = None,
           date: Optional[str] = None, root: Optional[str] = None) -> VideoInfo:
    """Render a :class:`Video` (or a LIST of them) to a file at a convention ``media/`` path.

    ``speed`` is in SIMULATION MILLISECONDS PER REAL SECOND and overrides ``fps``.

    Passing a list renders N panels sharing ONE colorbar and ONE time stamp; ``labels``,
    ``rows`` and ``cols`` apply to that path only. Panels must share a grid and a field kind;
    the frame count truncates to the shortest clip. Bare clips are promoted to the figure
    producer (with a warning) because a shared colorbar needs axes.
    """
    fmt = format                       # `format` shadows the builtin; bind locally
    if fit not in _LEGAL_FIT:
        raise ValueError(f"fit must be one of {_LEGAL_FIT}, got {fit!r}")

    # 1. validate + capability gate ------------------------------------------------
    if isinstance(video, (list, tuple)):
        return _render_panels(
            list(video), slug, question=question, bulk=bulk, resolution=resolution, fit=fit,
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

    # 6. path (consumes the NN slot) + canvas + rate --------------------------------
    path = media_path(question, kind, slug, ext=ext, bulk=bulk, date=date, root=root)
    canvas = resolve_canvas(resolution) if (figsize is None and dpi is None) else None
    if fmt == "webm" and bitrate is None:
        bitrate = "2M"     # VP9 has no `quality` mapping; without a rate ffmpeg silently uses CRF 32
    use_figure = clip.requires_figure()

    # 7. stream — writer AND figure construction go INSIDE the guarded region -------
    n = 0
    writer = None
    st = None
    try:
        writer = open_writer(path, fps, backend, fmt, quality=8, bitrate=bitrate)
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
    except BaseException:
        if writer is not None:
            writer.close()
        if os.path.exists(path):
            os.remove(path)     # a truncated file would otherwise keep the media_path NN slot
        raise
    finally:
        if st is not None:
            plt.close(st.fig)   # else the suite leaks a figure per render()
    writer.close()

    size = os.path.getsize(writer.path) if os.path.exists(writer.path) else 0
    return VideoInfo(path=writer.path, n_frames=n, fps=fps, backend=writer.backend,
                     codec=writer.codec, width=writer.width, height=writer.height,
                     duration_s=(n / fps if fps else 0.0), vmin=lo, vmax=hi, stride=stride,
                     size_bytes=size, bitrate=writer.bitrate)


render_video = render      # alias so the _LAZY export name resolves


def preview_frame(video: Video, t_ms: Optional[float] = None, *, frame: Optional[int] = None,
                  slug: str = "preview", question: str = "lab", bulk: bool = True,
                  units: Optional[str] = None, title: Optional[str] = None,
                  figsize=None, dpi=None, date=None, root=None) -> str:
    """Render ONE frame to PNG through the clip's OWN producer. Returns the path."""
    if t_ms is not None and frame is not None:
        raise ValueError("pass t_ms= or frame=, not both")
    # Same gate as render(), or preview would accept clips render() rejects.
    enforce_capabilities(video, colorbar=None, show_time=None,
                         figsize=figsize, dpi=dpi, title=title)

    if frame is not None:
        t = int(frame)
    elif t_ms is not None:
        t = int(np.argmin(np.abs(video.times - float(t_ms))))
    else:
        t = len(video.frames) // 2
    if not (0 <= t < len(video.frames)):
        raise IndexError(f"frame {t} out of range for {len(video.frames)} frames")

    cmap, norm, _lo, _hi = video.gradient.resolve(video.masked_iter([t]), field=video.field)
    path = media_path(question, "images", slug, ext="png", bulk=bulk, date=date, root=root)

    if video.requires_figure():
        st = _build_figure(video, cmap, norm, colorbar_on=(video.style == "annotated"),
                           title=title, figsize=figsize, dpi=dpi, units=units, idx=[t])
        try:
            _produce_figure(st, video, t, show_time=True, title=title)
            st.fig.savefig(path, dpi=(dpi or 100), bbox_inches="tight")
        finally:
            plt.close(st.fig)
    else:
        from PIL import Image
        rgb = _produce_bare(video, t, cmap, norm)
        rgb = burn_timestamp(rgb, f"t = {video.times[t]:.1f} ms")
        Image.fromarray(rgb).save(path)
    return path


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


def _setup_panel(clip: Video, ax, cmap, norm, *, units, idx, label=None) -> _FigState:
    """Configure ONE axes for a panel and return its per-frame state carrier."""
    units_resolved = units or clip.units
    if units_resolved == "auto":
        units_resolved = "cm" if (clip.dx and clip.dy) else "nodes"
    extent, xlab, ylab = _extent_and_labels(clip, units_resolved)
    Nx, Ny = clip.frames.shape[1], clip.frames.shape[2]

    im = ax.imshow(
        np.ma.masked_invalid(clip.display_values(idx[0]).T),
        origin="lower", extent=extent, aspect=clip.aspect, cmap=cmap, norm=norm,
        interpolation=clip.gradient.interpolation,
    )
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if label:
        ax.set_title(label, fontsize=10)

    x = np.linspace(extent[0], extent[1], Nx)
    y = np.linspace(extent[2], extent[3], Ny)
    Xc, Yc = np.meshgrid(x, y, indexing="ij")

    if clip.isochrones:
        lat = isochrone_lat(clip, idx)
        if np.isfinite(lat).any():
            ax.contour(Xc, Yc, np.ma.masked_invalid(lat), levels=12,
                       colors="white", linewidths=0.6, alpha=0.55)

    return _FigState(fig=None, ax=ax, im=im, Xc=Xc, Yc=Yc, contour=None, suptitle=None)


def _render_panels(clips: List[Video], slug: str, *, question, bulk, resolution, fit, fps, speed,
                   max_frames, fmt, bitrate, show_time, colorbar, title, figsize, dpi, units,
                   progress, date, root, labels=None, rows=None, cols=None) -> VideoInfo:
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
    path = media_path(question, kind, slug, ext=ext, bulk=bulk, date=date, root=root)
    canvas = resolve_canvas(resolution) if (figsize is None and dpi is None) else None
    if fmt == "webm" and bitrate is None:
        bitrate = "2M"

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
        writer = open_writer(path, fps, backend, fmt, quality=8, bitrate=bitrate)
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
    except BaseException:
        if writer is not None:
            writer.close()
        if os.path.exists(path):
            os.remove(path)
        raise
    finally:
        if fig is not None:
            plt.close(fig)
    writer.close()

    size = os.path.getsize(writer.path) if os.path.exists(writer.path) else 0
    return VideoInfo(path=writer.path, n_frames=n, fps=fps, backend=writer.backend,
                     codec=writer.codec, width=writer.width, height=writer.height,
                     duration_s=(n / fps if fps else 0.0), vmin=lo, vmax=hi, stride=stride,
                     size_bytes=size, bitrate=writer.bitrate)
