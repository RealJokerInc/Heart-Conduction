"""
cardiac_core.viz — standardized result visuals (engine_consolidation Goal-2 /sim-media).

One place to turn a ``SimulationResult`` into the canonical figures/videos lab scientists want, saved to
convention-compliant ``media/`` paths (``cardiac_core.media.media_path``). Headless (Agg) + float64.
Generated experiment scripts call these instead of hand-rolling matplotlib.

    from cardiac_core import propagation_video, apd_map_figure, activation_isochrones
    propagation_video(result, "my-experiment")      # -> media/lab/videos/{date}/my-experiment_NN.mp4
"""

import matplotlib
matplotlib.use("Agg")  # headless — viz is for scripts/CI, never a GUI
import numpy as np
import matplotlib.pyplot as plt

from .media import media_path


def _vm_numpy(result):
    """(T, Nx, Ny) float64 numpy from a SimulationResult."""
    return result.Vm.detach().cpu().numpy()


def propagation_video(result, slug, *, question="lab", fps=20, vmin=-90.0, vmax=40.0,
                      cmap="inferno", bulk=False) -> str:
    """Animate the voltage propagation. Returns the path (str).

    Thin wrapper over :func:`cardiac_core.video.render`, kept for the ``/sim-media`` skill and
    existing Lab scripts. It preserves this function's historical look: annotated (axes +
    colorbar + time), stretched ``aspect="auto"``, node-index labels, no masking, 600x300.

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
        slug, question=question, bulk=bulk, fps=fps,
        figsize=(6.0, 3.0), dpi=100, resolution=None,   # explicit figsize wins -> 600x300
        max_frames=None, colorbar=True, show_time=True,
    )
    return info.path


def apd_map_figure(result, slug, *, question="lab", cmap="viridis", bulk=False, **apd_kw) -> str:
    """APD90 map (ms) as a heatmap PNG. Returns the path."""
    from . import analysis
    apd = analysis.apd_map(result.Vm, result.times, **apd_kw).detach().cpu().numpy()
    apd = np.where(np.isfinite(apd), apd, np.nan)

    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(apd.T, origin="lower", aspect="auto", cmap=cmap)
    fig.colorbar(im, ax=ax, label="APD90 (ms)")
    ax.set_title("APD map")
    ax.set_xlabel("x (nodes)")
    ax.set_ylabel("y (nodes)")
    path = media_path(question, "images", f"{slug}-apd", ext="png", bulk=bulk)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def activation_isochrones(result, slug, *, question="lab", levels=15, cmap="plasma",
                          bulk=False, **lat_kw) -> str:
    """Local-activation-time isochrones (filled contours) PNG. Returns the path."""
    from . import analysis
    lat = analysis.activation_time(result.Vm, result.times, **lat_kw).detach().cpu().numpy()
    lat = np.where(np.isfinite(lat), lat, np.nan)

    fig, ax = plt.subplots(figsize=(6, 3))
    if np.isfinite(lat).any():
        cs = ax.contourf(lat.T, levels=levels, cmap=cmap)
        fig.colorbar(cs, ax=ax, label="activation time (ms)")
    ax.set_title("Activation isochrones")
    ax.set_xlabel("x (nodes)")
    ax.set_ylabel("y (nodes)")
    path = media_path(question, "images", f"{slug}-isochrones", ext="png", bulk=bulk)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
