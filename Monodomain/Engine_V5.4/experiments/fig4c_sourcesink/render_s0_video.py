"""Render the S0 expanding-circle propagation to MP4 (from the cached npz).

Phase-1 visual: the isotropic expanding wave whose curvature kappa=1/r sweeps the
eikonal relation CV_n = CV0 - D*kappa. No re-simulation — loads _sim_outputs cache.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path("/home/norepinephrine/Documents/Heart-Conduction")
sys.path.insert(0, str(REPO))
from cardiac_core.media import media_path
from cardiac_core.analysis import activation_time_interp

LX = LY = 1.0
DX = 0.005
CACHE = REPO / "media/source_sink_mismatch_investigation/_sim_outputs/s0_eikonal.npz"


def main():
    z = np.load(CACHE)
    times, V = z["times"], z["V"]
    nN, NX, NY = V.shape
    print(f"loaded {nN} frames, {NX}x{NY}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import imageio_ffmpeg
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    ext = [0, LX, 0, LY]
    lat = activation_time_interp(V, times, -40.0)

    fig, (axv, axl) = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    # left: live voltage
    imv = axv.imshow(V[0].T, origin="lower", extent=ext, cmap="inferno",
                     vmin=float(V.min()), vmax=40, aspect="equal", interpolation="bilinear")
    axv.set_title("S0 expanding wave — t=0 ms"); axv.set_xlabel("x (cm)"); axv.set_ylabel("y (cm)")
    fig.colorbar(imv, ax=axv, shrink=0.8, label="V (mV)")
    # right: LAT map + isochrones (static reference) with a moving isochrone highlight
    Lm = lat.T
    axl.imshow(Lm, origin="lower", extent=ext, cmap="viridis", aspect="equal")
    xs = np.linspace(0, LX, NX); ys = np.linspace(0, LY, NY)
    axl.contour(xs, ys, Lm, levels=np.arange(0, np.nanmax(lat), 1.0),
                colors="white", linewidths=0.5, alpha=0.7)
    axl.set_title("LAT + isochrones (κ=1/r → CV=CV0−D/r)"); axl.set_xlabel("x (cm)"); axl.set_ylabel("y (cm)")
    moving = axl.contour(xs, ys, Lm, levels=[0.0], colors="red", linewidths=2.0)

    def upd(fr):
        nonlocal moving
        imv.set_array(V[fr].T)
        axv.set_title(f"S0 expanding wave — t={times[fr]:.1f} ms")
        moving.remove()
        moving = axl.contour(xs, ys, Lm, levels=[max(times[fr], 1e-3)], colors="red", linewidths=2.0)
        return [imv]

    anim = FuncAnimation(fig, upd, frames=range(0, nN), blit=False)
    pv = media_path("source_sink_mismatch_investigation", "videos", "s0-eikonal-expanding-circle", ext="mp4")
    anim.save(pv, writer=FFMpegWriter(fps=12, bitrate=3500), dpi=110)
    plt.close(fig)
    print("wrote", pv)


if __name__ == "__main__":
    main()
