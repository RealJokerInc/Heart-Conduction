"""
cardiac_core.viz — standardized result visuals.

One place to turn a ``SimulationResult`` into the figures and videos most experiments want, saved
to convention-compliant ``media/`` paths (``cardiac_core.media.media_path``). Headless (Agg) +
float64, so analysis scripts call these instead of hand-rolling matplotlib.

    from cardiac_core import propagation_video, apd_map_figure, activation_isochrones
    propagation_video(result, "my-experiment")      # -> media/lab/videos/{date}/my-experiment_NN.mp4
"""

import matplotlib
matplotlib.use("Agg")  # headless — viz is for scripts/CI, never a GUI
import numpy as np
import matplotlib.pyplot as plt

from .media import media_path  # noqa: F401  (kept: part of this module's public surface)


def _vm_numpy(result):
    """(T, Nx, Ny) float64 numpy from a SimulationResult."""
    return result.Vm.detach().cpu().numpy()


def propagation_video(result, slug, *, question="lab", fps=20, vmin=-90.0, vmax=40.0,
                      cmap="inferno", bulk=False, root=None) -> str:
    """Animate the voltage propagation. Returns the path (str).

    Thin wrapper over :func:`cardiac_core.video.render`, kept for backwards compatibility. It
    preserves this function's historical look: annotated (axes + colorbar + time), stretched
    ``aspect="auto"``, node-index labels, no masking, 600x300.

    For anything more — physical cm axes, ``phi_e``, playback ``speed=``, overlays, multi-panel,
    the fast full-frame style, gradient presets — use ``result.video(slug, ...)`` or
    ``cardiac_core.render`` directly.

    .. note::
       Previously this saved through matplotlib's ``ffmpeg`` writer inside a bare ``except``
       that silently rewrote the output to a GIF at a different path and extension whenever
       ffmpeg was not on ``PATH`` (and swallowed codec/disk errors identically). It now goes
       through a PATH-independent bundled-ffmpeg backend, and any fallback warns loudly.

    Two deliberate, documented behaviour changes: the per-frame time text moves from
    ``ax.set_title`` to ``fig.suptitle``; and a GIF fallback is now named ``{slug}.gif`` rather
    than ``{slug}-propagation.gif`` (the fallback is announced by a warning instead).
    """
    from .video import render, Video, Gradient
    info = render(
        Video(result,
              gradient=Gradient(cmap=cmap, value_range=(vmin, vmax), bad="0.55"),
              style="annotated",
              aspect="auto",       # a Video field, NOT a render() kwarg
              units="nodes",       # legacy drew node indices
              mask=False),         # legacy did NOT mask — keep it that way
        slug, question=question, bulk=bulk, root=root, fps=fps,
        figsize=(6.0, 3.0), dpi=100, resolution=None,   # explicit figsize wins -> 600x300
        max_frames=None, colorbar=True, show_time=True,
    )
    return info.path


def apd_map_figure(result, slug, *, question="lab", cmap="viridis", bulk=False, root=None,
                   **apd_kw) -> str:
    """APD90 map (ms) as a heatmap PNG. Returns the path.

    Delegates to :func:`cardiac_core.image.draw`, preserving this function's historical look:
    ``figsize=(6, 3)``, ``aspect="auto"``, node-index axes, no masking, dpi 120, the ``"APD map"``
    title and the ``str`` return. Two documented differences: the axes extent shifts by half a node
    (``_extent_and_labels`` returns ``[0, Nx-1]`` where a bare ``imshow`` uses ``[-0.5, Nx-0.5]``),
    and an all-NaN map now emits the colour-range warning that ``Gradient`` raises for empty data —
    a silently blank APD map is worth saying out loud.
    """
    from .image import Image, draw
    from .video import Gradient
    info = draw(Image(result, what="apd", what_kwargs=apd_kw or None,
                      gradient=Gradient(cmap=cmap, value_range="auto"),
                      aspect="auto", units="nodes", mask=False, label="APD map"),
                f"{slug}-apd", question=question, bulk=bulk, root=root,
                figsize=(6.0, 3.0), dpi=120)
    return info.path


def activation_isochrones(result, slug, *, question="lab", levels=15, cmap="plasma",
                          bulk=False, root=None, **lat_kw) -> str:
    """Local-activation-time isochrones (filled contours) PNG. Returns the path.

    Delegates to :func:`cardiac_core.image.draw` with ``filled=True``, which is what preserves this
    function's composition: filled ``contourf`` bands with the colorbar sourced from the contour
    set, and NO line contours on top. ``isochrones=False`` is explicit for the same reason — filled
    bands ARE the isochrones, so letting the automatic overlay fire would double-draw.
    """
    from .image import Image, draw
    from .video import Gradient
    info = draw(Image(result, what="activation", what_kwargs=lat_kw or None,
                      gradient=Gradient(cmap=cmap, value_range="auto"),
                      aspect="auto", units="nodes", mask=False,
                      filled=True, contour_levels=levels, isochrones=False,
                      label="Activation isochrones"),
                f"{slug}-isochrones", question=question, bulk=bulk, root=root,
                figsize=(6.0, 3.0), dpi=120)
    return info.path
