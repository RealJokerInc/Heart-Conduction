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
from matplotlib import animation

from .media import media_path


def _vm_numpy(result):
    """(T, Nx, Ny) float64 numpy from a SimulationResult."""
    return result.Vm.detach().cpu().numpy()


def propagation_video(result, slug, *, question="lab", fps=20, vmin=-90.0, vmax=40.0,
                      cmap="inferno", bulk=False) -> str:
    """Animate the voltage propagation. Saves mp4 (ffmpeg); falls back to gif. Returns the path."""
    Vm = _vm_numpy(result)
    T = Vm.shape[0]
    times = result.times.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(Vm[0].T, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(im, ax=ax, label="Vm (mV)")
    ax.set_xlabel("x (nodes)")
    ax.set_ylabel("y (nodes)")

    def _update(t):
        im.set_data(Vm[t].T)
        ax.set_title(f"t = {times[t]:.1f} ms")
        return (im,)

    anim = animation.FuncAnimation(fig, _update, frames=T, blit=False)
    try:
        path = media_path(question, "videos", slug, ext="mp4", bulk=bulk)
        anim.save(path, writer="ffmpeg", fps=fps)
    except Exception:
        # gif is an IMAGE type in the media convention → kind="images"
        path = media_path(question, "images", f"{slug}-propagation", ext="gif", bulk=bulk)
        anim.save(path, writer="pillow", fps=fps)
    plt.close(fig)
    return path


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
