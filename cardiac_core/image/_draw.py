"""``draw()`` — turn an :class:`Image` (or a :class:`~cardiac_core.video.Video`) into a figure.

The ordered sequence is load-bearing and must not be reordered::

    dispatch -> frame_resolved -> format vs path -> media guard -> enforce_capabilities (RAW
    figsize/dpi) -> spec-type defaults -> resolve gradient over MASKED values -> destination
    -> produce + save -> read back (w,h) + getsize -> _finalize -> ImageInfo

``enforce_capabilities`` must see exactly what the caller passed: it rejects a non-``None``
``figsize``/``dpi`` on a bare clip, so substituting a default before the gate would make every bare
draw raise. The pixel size and byte count must be read BEFORE ``_finalize``, which deletes the temp
file of an unsaved render.

The module is ``_draw`` rather than ``draw`` because a submodule whose name matches a public export
shadows it under PEP 562 — ``cardiac_core.draw`` would become a module instead of the function.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use("Agg")     # headless — figure rendering is for scripts/CI, never a GUI

import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402

from ..video.clip import Video, _to_numpy                        # noqa: E402
from ..video.encoders import burn_timestamp, fit_frame, resolve_canvas  # noqa: E402
from ..video.render import (                                     # noqa: E402
    _LEGAL_FIT, _PAD_BLACK, _build_figure, _default_layout, _finalize, _named_destination,
    _produce_bare, _produce_figure, _resolve_destination, _setup_panel, discard_partial,
    enforce_capabilities,
)
from .info import ImageInfo                                      # noqa: E402
from .panel import Image, Trace                                  # noqa: E402

__all__ = ["draw", "_draw_trace_on"]


class _Unset:
    """Sentinel: 'the caller passed nothing'.

    A literal default cannot express this, because ``resolution``/``fit`` have DIFFERENT defaults
    per spec type. Without the sentinel, either a plain ``draw(Video.annotated(v))`` raises, or an
    explicit ``resolution="auto"`` becomes a silent no-op.
    """

    def __repr__(self) -> str:                                   # pragma: no cover - cosmetic
        return "<unset>"


_UNSET = _Unset()

_LEGAL_EXT = ("png", "svg", "pdf", "jpg", "jpeg", "webp")
_VECTOR = ("svg", "pdf")
_MEDIA_ONLY_EXT = ("png", "jpg", "jpeg", "svg", "gif")   # what media_path accepts
_MIN_BARE_EDGE = 512                                     # a bare still must not be a postage stamp


def _resolve_format(fmt: Optional[str], path: Optional[str]) -> str:
    """An explicit ``format=`` wins; otherwise take it from ``path``'s extension.

    A disagreement RAISES rather than silently rewriting the path, which is what the video layer's
    ``_resolve_destination`` would otherwise do (with an explanation about encoder downgrades that
    is meaningless here).
    """
    path_ext = None
    if path is not None:
        ext = os.path.splitext(str(path))[1].lower().lstrip(".")
        if ext:
            if ext not in _LEGAL_EXT:
                raise ValueError(
                    f"cannot infer a figure format from '.{ext}' (path={path!r}); "
                    f"use one of {_LEGAL_EXT} or pass format= explicitly")
            path_ext = ext
    if fmt is not None:
        fmt = str(fmt).lower().lstrip(".")
        if fmt not in _LEGAL_EXT:
            raise ValueError(f"format must be one of {_LEGAL_EXT}, got {fmt!r}")
        if path_ext is not None and path_ext != fmt:
            raise ValueError(
                f"format={fmt!r} disagrees with path={path!r} — pass one or the other.")
        return fmt
    return path_ext if path_ext is not None else "png"


def _resolve_show_time(show_time: Optional[bool], t0: float) -> bool:
    """Whether to stamp the frame time — ONE rule for the single- and multi-panel paths.

    An explicit ``show_time=True`` on a map with no real time (activation/apd/frequency, whose frame
    time is NaN) RAISES rather than stamping "t = nan ms". ``None`` means auto: stamp iff finite.
    """
    if show_time is True and not np.isfinite(t0):
        raise ValueError(
            "show_time=True needs a real time; this map has none (use what='snapshot').")
    return show_time if show_time is not None else bool(np.isfinite(t0))


def _reject_vector_on_media_path(fmt: str, path: Optional[str],
                                 question, bulk, date, root) -> None:
    """pdf/webp cannot go on a media/ convention path (media_path accepts png/jpg/jpeg/svg/gif)."""
    if fmt in ("pdf", "webp") and path is None and _named_destination(question, bulk, date, root):
        raise ValueError(
            f"format={fmt!r} cannot be written to a media/ path — media_path accepts "
            f"{'/'.join(_MEDIA_ONLY_EXT)}. Pass path='fig.{fmt}' instead.")


def _reraise_for_image(exc: ValueError) -> ValueError:
    """Restate a ``Video``-flavoured capability error in ``Image`` vocabulary.

    The premise of this layer is that the caller does not know the video/clip API; naming it in an
    error message is the one place that premise leaks.
    """
    msg = (str(exc)
           .replace("Use Video.annotated(...) — the bare producer",
                    'Use style="annotated" — the bare producer')
           .replace("a bare clip", 'style="bare"'))
    return ValueError(msg)


def _measure(out_path: str, fmt: str):
    """``(size, width, height)`` — NEVER raises.

    This runs inside the cleanup guard, so a failure here would send a SUCCESSFULLY written
    figure to ``discard_partial`` and delete it. A probe that cannot read the file costs us the
    dimensions, not the output.
    """
    try:
        size = os.path.getsize(out_path)
    except OSError:
        size = 0
    width = height = None
    if fmt not in _VECTOR:
        try:
            from PIL import Image as PILImage
            with PILImage.open(out_path) as probe:
                width, height = probe.size
        except BaseException as exc:
            # Dimensions are informational; the bytes are what matter, so the file SURVIVES.
            # Say so — silence would report a zero-byte or undecodable raster as a clean
            # success — but the warn MUST NOT escape: this runs inside the caller's cleanup
            # guard, and under `-W error` a raised UserWarning would reach discard_partial()
            # and delete the figure we just wrote successfully. Same hazard discard_partial
            # documents and defends against.
            try:
                warnings.warn(
                    f"wrote {out_path!r} but could not read its dimensions ({exc}); the file "
                    f"is kept and width/height are None.", UserWarning, stacklevel=3)
            except BaseException:
                pass
    return size, width, height


def _bare_canvas(rgb: np.ndarray, resolution) -> Optional[tuple]:
    """Target canvas for the bare producer, or None for 'leave the pixels alone'.

    ``"auto"`` upscales by an integer factor to a long edge of at least 512 px with NO padding: an
    exact multiple preserves the aspect exactly, so ``fit_frame`` adds no bars. A fixed
    ``resolution`` letterboxes instead, which is right for video and wrong for a still that someone
    will crop into a slide.
    """
    if resolution is None:
        return None
    h, w = rgb.shape[:2]
    if resolution == "auto":
        k = max(1, math.ceil(_MIN_BARE_EDGE / max(w, h)))
        return (w * k, h * k)
    return resolve_canvas(resolution)


def draw(spec, slug: str = "figure", *, path: Optional[str] = None,
         question: Optional[str] = None, bulk: Optional[bool] = None,
         date: Optional[str] = None, root: Optional[str] = None,
         format: Optional[str] = None, frame: Optional[int] = None,
         figsize: Optional[Sequence[float]] = None, dpi: Optional[int] = None,
         tight: Optional[bool] = None, title: Optional[str] = None,
         colorbar: Optional[bool] = None, show_time: Optional[bool] = None,
         units: Optional[str] = None, transparent: bool = False,
         resolution: Any = _UNSET, fit: Any = _UNSET,
         labels: Optional[Sequence[str]] = None,
         rows: Optional[int] = None, cols: Optional[int] = None) -> ImageInfo:
    """Draw a spec and return a displayable :class:`ImageInfo`.

    **Drawing displays; naming a destination saves** — the matplotlib contract. With no destination
    the figure comes back in memory and shows inline; a file is written when you say where::

        draw(Image(r))                              # displays; nothing on disk
        draw(Image(r), path="fig.png")              # writes ./fig.png
        draw(Image(r), "wave", question="lab")      # media/lab/images/{date}/wave_01.png

    ``format`` follows ``path``'s extension when not given explicitly, and a disagreement raises.
    """
    if isinstance(spec, (list, tuple)):
        if frame is not None:
            raise ValueError(
                "frame= selects one map's frame; a multi-panel figure resolves each panel from its "
                "own spec. Set the frame on the Image(...), or draw a single map.")
        if resolution is not _UNSET or fit is not _UNSET:
            raise ValueError(
                "resolution=/fit= scale the bare-map producer; a multi-panel figure always uses the "
                "annotated producer. Drop them, or draw a single bare map.")
        return _draw_panels(list(spec), slug, path=path, question=question, bulk=bulk,
                            date=date, root=root, fmt=_resolve_format(format, path),
                            figsize=figsize, dpi=dpi, tight=tight, title=title,
                            colorbar=colorbar, show_time=show_time, units=units,
                            transparent=transparent, labels=labels, rows=rows, cols=cols)
    is_image = isinstance(spec, Image)
    is_trace = isinstance(spec, Trace)
    if not is_image and not is_trace and not isinstance(spec, Video):
        raise TypeError(
            f"draw() takes an Image, a Trace or a Video spec; got {type(spec).__name__}.")

    for name, val in (("labels", labels), ("rows", rows), ("cols", cols)):
        if val is not None:
            raise ValueError(
                f"{name}= applies to multi-panel rendering; pass a list of specs.")

    if is_trace:
        for name, val in (("frame", frame), ("colorbar", colorbar), ("show_time", show_time),
                          ("units", units)):
            if val is not None:
                raise ValueError(
                    f"{name}= applies to a spatial map; a Trace has no image. Use r.image(...).")
        for name, val in (("resolution", resolution), ("fit", fit)):
            if val is not _UNSET:
                raise ValueError(
                    f"{name}= scales the bare map producer; a Trace is always a figure.")
        return _draw_trace(spec, slug, path=path, question=question, bulk=bulk, date=date,
                           root=root, fmt=_resolve_format(format, path), figsize=figsize,
                           dpi=dpi if dpi is not None else 150,
                           tight=True if tight is None else bool(tight),
                           title=title, transparent=transparent)

    clip = spec._clip if is_image else spec

    # --- frame_resolved: ONE name for "which frame", bound first ---------------------
    if is_image:
        if frame is not None:
            raise ValueError(
                "frame= selects a frame of a Video; an Image selects with at= (a TIME in ms).")
        frame_resolved = 0                       # an Image clip always holds exactly one frame
    else:
        frame_resolved = int(frame) if frame is not None else len(clip.frames) // 2
        if not (0 <= frame_resolved < len(clip.frames)):
            raise IndexError(
                f"frame {frame_resolved} out of range for {len(clip.frames)} frames")

    # --- format, and the media-convention guard --------------------------------------
    fmt = _resolve_format(format, path)
    _reject_vector_on_media_path(fmt, path, question, bulk, date, root)

    # --- capability gate: the RAW figsize/dpi, before any default is applied ----------
    try:
        enforce_capabilities(clip, colorbar=colorbar, show_time=show_time,
                             figsize=figsize, dpi=dpi, title=title)
    except ValueError as exc:
        raise (_reraise_for_image(exc) if is_image else exc) from None

    bare = clip.style == "bare"
    if bare:
        for name, val in (("tight", tight), ("transparent", transparent or None)):
            if val is not None and val is not False:
                raise ValueError(
                    f"{name}= applies to the matplotlib figure only, not a bare spec. "
                    f'Use style="annotated".')
        if is_image and units is not None:
            raise ValueError(
                'units= draws axis labels and needs a figure. Use style="annotated".')
        if colorbar is False:
            raise ValueError(
                'colorbar= applies to the matplotlib figure only, not a bare spec. '
                'Use style="annotated".')

    # --- spec-type defaults, applied AFTER the gate ----------------------------------
    # Compare against the SENTINEL, never against "the default": an annotated Image's default
    # reads as "auto", so a `!= default` test would let `resolution="auto"` through as a silent
    # no-op. A Video keeps its default exempt so the preview delegation's explicit None passes.
    default_resolution = ("auto" if is_image else None)
    if resolution is _UNSET:
        resolution = default_resolution
    elif not bare and (is_image or resolution != default_resolution):
        raise ValueError(
            "resolution= scales the bare producer's pixels; an annotated figure is sized by "
            "figsize=/dpi=.")
    if fit is _UNSET:
        fit = "contain"
    elif not bare and (is_image or fit != "contain"):
        raise ValueError(
            "fit= scales the bare producer's pixels; an annotated figure is sized by figsize=/dpi=.")
    if fit not in _LEGAL_FIT:
        raise ValueError(f"fit must be one of {_LEGAL_FIT}, got {fit!r}")

    dpi_resolved = dpi if dpi is not None else (150 if is_image else 100)
    tight_resolved = True if tight is None else bool(tight)
    colorbar_resolved = colorbar if colorbar is not None else (clip.style == "annotated")

    # --- show_time: ONE formula, keyed on the TIME, not the selector -----------------
    t0 = float(clip.times[frame_resolved])
    show_time_resolved = _resolve_show_time(show_time, t0)

    # --- colour, over MASKED values, at the frame being drawn ------------------------
    cmap, norm, lo, hi = clip.gradient.resolve(
        clip.masked_iter([frame_resolved]), field=clip.field)

    # --- the isochrone overlay: gated SOLELY by the resolved `isochrones` -------------
    lat = None
    if is_image and spec.isochrones:
        lat = spec._lat
        if lat is None:
            lat = _lat_from_result(spec)
        if lat is not None and clip.active_mask is not None:
            lat = np.where(clip.active_mask, lat, np.nan)

    # --- everything that can RAISE, before we own a file ------------------------------
    # Acquiring the destination has side effects (a temp file; a caller's existing file becomes
    # ours to protect). A raise afterwards means the cleanup guard runs for a render that never
    # wrote a byte, which orphans temp files and warns about untouched files.
    rgb = None
    if bare:
        if fmt == "svg":
            raise ValueError(
                'a bare spec is written with PIL, which cannot produce SVG. '
                'Use style="annotated".')
        rgb = _produce_bare(clip, frame_resolved, cmap, norm)
        canvas = _bare_canvas(rgb, resolution)          # raises on a bad resolution=
        if canvas is not None:
            rgb = fit_frame(rgb, canvas, fit, clip.gradient.interpolation, _PAD_BLACK)

    # --- destination ------------------------------------------------------------------
    out_path, is_temp, owned = _resolve_destination(slug, "images", fmt, path=path,
                                                    question=question, bulk=bulk,
                                                    date=date, root=root)

    st = None
    opened = False
    try:
        if bare:
            from PIL import Image as PILImage
            if show_time_resolved:
                # AFTER the resize, or the stamp is drawn at grid scale and blown up with it.
                rgb = burn_timestamp(rgb, f"t = {t0:.1f} ms")
            opened = True
            PILImage.fromarray(rgb).save(out_path)
        else:
            st = _build_figure(
                clip, cmap, norm, colorbar_on=colorbar_resolved, title=title,
                figsize=figsize, dpi=dpi_resolved, units=units, idx=[frame_resolved],
                lat=lat,
                contour_levels=getattr(spec, "contour_levels", 12),
                filled=getattr(spec, "filled", False),
            )
            _produce_figure(st, clip, frame_resolved, show_time=show_time_resolved, title=title)
            opened = True
            st.fig.savefig(out_path, dpi=dpi_resolved,
                           bbox_inches=("tight" if tight_resolved else None),
                           transparent=transparent)
        # Non-raising: a probe failure must not destroy a successfully written figure.
        size, width, height = _measure(out_path, fmt)
    except BaseException:
        discard_partial(out_path, owned, opened=opened)
        raise
    finally:
        if st is not None:
            plt.close(st.fig)     # else the suite leaks a figure per draw()

    final_path, data = _finalize(out_path, is_temp)
    return ImageInfo(path=final_path, data=data, format=fmt, width=width, height=height,
                     n_panels=1, vmin=lo, vmax=hi, size_bytes=size)


def _lat_from_result(spec: Image):
    """Activation map for the overlay, computed from the SOURCE result (torch), or None.

    No ``what_kwargs``: those belong to the selector's own analysis function, and forwarding them
    here is either a ``TypeError`` or a silently different LAT drawn over the map.
    """
    import warnings

    from .. import analysis

    result = spec._clip.result
    if result is None:
        warnings.warn(
            "isochrones need a SimulationResult to compute activation times; skipping the overlay",
            UserWarning, stacklevel=3)
        return None
    return np.asarray(
        _to_numpy(analysis.activation_time(result.Vm, result.times)), dtype=np.float64)


def _draw_trace_on(spec: Trace, ax) -> None:
    """Draw a ``Trace`` onto an EXISTING axes. Owns no figure — the layout path calls this too."""
    for i, (label, x, y) in enumerate(spec.series):
        color = spec.colors[i] if spec.colors and i < len(spec.colors) else None
        ax.plot(x, y, label=label, marker=spec.marker,
                linestyle=("none" if spec.linestyle == "none" else spec.linestyle),
                color=color)
    for value, label in spec.hlines:
        ax.axhline(value, color="0.4", lw=1.0, ls="--", label=label)
    for value, label in spec.vlines:
        ax.axvline(value, color="0.4", lw=1.0, ls=":", label=label)
    if spec.xlabel:
        ax.set_xlabel(spec.xlabel)
    if spec.ylabel:
        ax.set_ylabel(spec.ylabel)
    if spec.label:
        ax.set_title(spec.label)
    if spec.logx:
        ax.set_xscale("log")
    if spec.logy:
        ax.set_yscale("log")
    if spec.xlim:
        ax.set_xlim(*spec.xlim)
    if spec.ylim:
        ax.set_ylim(*spec.ylim)
    if spec.legend and any(h.get_label() and not h.get_label().startswith("_")
                           for h in ax.get_lines()):
        ax.legend()


def _draw_trace(spec: Trace, slug: str, *, path, question, bulk, date, root, fmt,
                figsize, dpi, tight, title, transparent) -> ImageInfo:
    """Single-panel wrapper: owns the figure, delegates the drawing, shares the delivery path."""
    _reject_vector_on_media_path(fmt, path, question, bulk, date, root)
    # Fallible setup BEFORE the destination — acquiring it creates a temp file and makes a
    # caller's existing file ours to protect. (Same rule as draw()/render().)
    fig, ax = plt.subplots(figsize=tuple(figsize) if figsize else (6.4, 3.6), dpi=dpi)
    try:
        out_path, is_temp, owned = _resolve_destination(slug, "images", fmt, path=path,
                                                        question=question, bulk=bulk,
                                                        date=date, root=root)
    except BaseException:
        plt.close(fig)                 # the figure exists already; do not leak it
        raise
    opened = False
    try:
        _draw_trace_on(spec, ax)
        if title:
            fig.suptitle(title)
        opened = True
        fig.savefig(out_path, dpi=dpi, bbox_inches=("tight" if tight else None),
                    transparent=transparent)
        # Non-raising: a probe failure must not destroy a successfully written figure.
        size, width, height = _measure(out_path, fmt)
    except BaseException:
        discard_partial(out_path, owned, opened=opened)
        raise
    finally:
        if fig is not None:
            plt.close(fig)

    final_path, data = _finalize(out_path, is_temp)
    # A trace has no colour range — vmin/vmax stay None rather than being invented.
    return ImageInfo(path=final_path, data=data, format=fmt, width=width, height=height,
                     n_panels=1, vmin=None, vmax=None, size_bytes=size)


def _draw_panels(specs, slug, *, path, question, bulk, date, root, fmt, figsize, dpi, tight,
                 title, colorbar, show_time, units, transparent, labels, rows, cols) -> ImageInfo:
    """N panels in a grid. Map panels sharing a gradient AND a label share ONE colorbar."""
    import math as _math
    import warnings

    if not specs:
        raise ValueError("draw() needs at least one spec")
    for sp in specs:
        if isinstance(sp, Video):
            raise ValueError(
                "a Video belongs to the video layer's multi-panel path — use render([...]).")
        if not isinstance(sp, (Image, Trace)):
            raise TypeError(f"draw() takes Image/Trace specs; got {type(sp).__name__}.")

    maps = [sp for sp in specs if isinstance(sp, Image)]
    shapes = {sp._clip.frames.shape[1:] for sp in maps}
    if len(shapes) > 1:
        raise ValueError(
            f"all map panels must share a grid; got {sorted(shapes)}. Panels are compared "
            f"side by side, so differing grids cannot be.")
    if any(sp.style == "bare" for sp in maps):
        warnings.warn(
            "multi-panel rendering always uses the figure producer; bare panels were promoted to "
            "annotated for layout (they gain axes and a shared colorbar).",
            UserWarning, stacklevel=3)

    # Never mutate the caller's specs: `labels=` is a draw-time override.
    panel_labels = [(labels[i] if (labels and i < len(labels)) else getattr(sp, "label", None))
                    for i, sp in enumerate(specs)]

    _reject_vector_on_media_path(fmt, path, question, bulk, date, root)

    # show_time parity, decided up front — before plt.subplots creates the figure (so a raise orphans
    # nothing) and AFTER the media guard above (so it keeps precedence, matching the single-panel
    # path). Compute the stamp decision + time ONCE and reuse them at the stamp block below.
    stamp_on, stamp_t0 = False, None
    if maps:
        stamp_t0 = float(maps[0]._clip.times[0])
        stamp_on = _resolve_show_time(show_time, stamp_t0)      # raises on show_time=True + NaN time
    elif show_time is not None:
        raise ValueError(
            "show_time= stamps a map's time; this multi-panel figure has only traces. "
            "Drop show_time=, or include a map panel.")

    # Colour: pooled only when every map panel agrees on BOTH the gradient and the quantity.
    # `_clip.field` is 'Vm' for every Image, so it cannot distinguish an APD map from a voltage one.
    shared = bool(maps) and all(
        sp.gradient.key() == maps[0].gradient.key()
        and sp._clip.value_label == maps[0]._clip.value_label for sp in maps)
    lo = hi = None
    per_panel = {}
    if maps:
        if shared:
            def _pooled():
                for sp in maps:
                    yield from sp._clip.masked_iter([0])
            cmap, norm, lo, hi = maps[0].gradient.resolve(_pooled(), field=maps[0]._clip.field)
            per_panel = {id(sp): (cmap, norm) for sp in maps}
        else:
            warnings.warn(
                "panels use different gradients or show different quantities and are NOT directly "
                "comparable; drawing a colorbar per panel instead of one shared scale.",
                UserWarning, stacklevel=3)
            for sp in maps:
                c, n, l0, h0 = sp.gradient.resolve(sp._clip.masked_iter([0]),
                                                   field=sp._clip.field)
                per_panel[id(sp)] = (c, n)
                if lo is None:
                    lo, hi = l0, h0

    nrows, ncols = (rows, cols) if (rows and cols) else _default_layout(len(specs))
    if rows and not cols:
        nrows, ncols = rows, _math.ceil(len(specs) / rows)
    elif cols and not rows:
        ncols, nrows = cols, _math.ceil(len(specs) / cols)
    if figsize is None:
        figsize = (min(6.5 * ncols, 19.0), min(3.6 * nrows, 10.0))
    dpi_resolved = dpi if dpi is not None else 150
    tight_resolved = True if tight is None else bool(tight)
    colorbar_resolved = True if colorbar is None else bool(colorbar)

    fig, axes = plt.subplots(nrows, ncols, figsize=tuple(figsize), dpi=dpi_resolved,
                             constrained_layout=True, squeeze=False)
    try:
        out_path, is_temp, owned = _resolve_destination(slug, "images", fmt, path=path,
                                                        question=question, bulk=bulk,
                                                        date=date, root=root)
    except BaseException:
        plt.close(fig)
        raise
    opened = False
    try:
        flat = [ax for row in axes for ax in row]
        for ax in flat[len(specs):]:
            ax.set_axis_off()

        states = []
        for sp, ax, lab in zip(specs, flat, panel_labels):
            if isinstance(sp, Trace):
                _draw_trace_on(sp, ax)
                if lab:
                    ax.set_title(lab, fontsize=10)
                states.append(None)
                continue
            cmap, norm = per_panel[id(sp)]
            lat = sp._lat if sp.isochrones else None
            if lat is None and sp.isochrones:
                lat = _lat_from_result(sp)
            if lat is not None and sp._clip.active_mask is not None:
                lat = np.where(sp._clip.active_mask, lat, np.nan)
            st = _setup_panel(sp._clip, ax, cmap, norm, units=units, idx=[0], label=lab,
                              lat=lat, contour_levels=sp.contour_levels, filled=sp.filled)
            st.fig = fig
            states.append(st)

        if colorbar_resolved and maps:
            mappables = [(sp, st) for sp, st in zip(specs, states)
                         if isinstance(sp, Image) and st is not None and st.im is not None]
            if mappables and shared:
                fig.colorbar(mappables[0][1].im,
                             ax=[st.ax for _, st in mappables],
                             label=maps[0]._clip.value_label, shrink=0.75)
            else:
                for sp, st in mappables:
                    fig.colorbar(st.im, ax=st.ax, label=sp._clip.value_label, shrink=0.75)

        # ONE suptitle for the whole figure; `_setup_panel` leaves suptitle=None, so calling
        # _produce_figure here would raise on it (and would overwrite the shared stamp per panel).
        sup = fig.suptitle(title or "")
        for st in states:
            if st is not None:
                st.suptitle = sup

        # `front` is the one thing _setup_panel omits — draw it per map panel.
        for sp, st in zip(specs, states):
            if isinstance(sp, Image) and sp.front is not None and st is not None:
                st.contour = st.ax.contour(st.Xc, st.Yc, sp._clip.display_values(0),
                                           levels=[sp.front], colors="white", linewidths=1.4)

        # Stamp decided up front (stamp_on/stamp_t0). A Trace has no clip and no time.
        if stamp_on:
            text = f"t = {stamp_t0:.1f} ms"
            sup.set_text(f"{title} — {text}" if title else text)

        opened = True
        fig.savefig(out_path, dpi=dpi_resolved,
                    bbox_inches=("tight" if tight_resolved else None), transparent=transparent)
        # Non-raising: a probe failure must not destroy a successfully written figure.
        size, width, height = _measure(out_path, fmt)
    except BaseException:
        discard_partial(out_path, owned, opened=opened)
        raise
    finally:
        if fig is not None:
            plt.close(fig)

    final_path, data = _finalize(out_path, is_temp)
    return ImageInfo(path=final_path, data=data, format=fmt, width=width, height=height,
                     n_panels=len(specs), vmin=lo, vmax=hi, size_bytes=size)
