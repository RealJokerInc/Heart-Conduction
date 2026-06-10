"""Side-by-side hourglass propagation: original params vs fixed params (from cache).

Left: orig (cardinal4, dx=250um). Right: fixed (moore8_iso, dx=50um). Same physical
domain (2.0x1.4cm) and frame times -> the dilation convex-slowing is visible in BOTH
(the original was never a failure to recreate curvature; only the absolute CV scale and
interpretation differed).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path("/home/norepinephrine/Documents/Heart-Conduction")
sys.path.insert(0, str(REPO / "Monodomain/Engine_V5.4"))
sys.path.insert(0, str(REPO))
from cardiac_core.media import media_path
from run_s0d_hourglass_confirm import hourglass_fluid, LX, LY, X_NECK
import torch

ARMS = [
    ("orig: cardinal4, dx=250um", 0.025, "cardinal4"),
    ("fixed: moore8_iso, dx=50um", 0.005, "moore8_iso"),
]


def load(dx, stencil):
    z = np.load(REPO / f"media/source_sink_mismatch_investigation/_sim_outputs/s0d_dx{int(dx*1e4)}_{stencil}.npz")
    times, V = z["times"], z["V"]
    nx, ny = V.shape[1], V.shape[2]
    solid = ~hourglass_fluid(dx, nx, ny, torch.device("cpu")).numpy()
    return times, V, solid


def main():
    data = [(lab, *load(dx, st)) for lab, dx, st in ARMS]
    nfr = min(d[2].shape[0] for d in [(l, t, V, s) for (l, t, V, s) in data])
    nfr = min(V.shape[0] for (_, _, V, _) in data)
    times = data[0][1]

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import imageio_ffmpeg
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    ext = [0, LX, 0, LY]
    cmap = plt.cm.inferno.copy(); cmap.set_bad("0.55")
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)
    ims = []
    for ax, (lab, t, V, solid) in zip(axes, data):
        def disp(a, s=solid): return np.ma.array(a.T, mask=s.T)
        im = ax.imshow(disp(V[0]), origin="lower", extent=ext, cmap=cmap,
                       vmin=-85, vmax=40, aspect="equal", interpolation="bilinear")
        ax.axvline(X_NECK, color="cyan", ls=":", lw=1)
        ax.set_title(f"{lab}"); ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
        ims.append((im, V, solid))
    sup = fig.suptitle(f"hourglass — t=0.0 ms")

    def upd(k):
        for im, V, solid in ims:
            im.set_array(np.ma.array(V[k].T, mask=solid.T))
        sup.set_text(f"hourglass — t={times[k]:.1f} ms  (dilation right of cyan neck line)")
        return [im for im, _, _ in ims]

    step = max(1, nfr // 220)
    anim = FuncAnimation(fig, upd, frames=range(0, nfr, step), blit=False)
    pv = media_path("source_sink_mismatch_investigation", "videos", "s0d-hourglass-orig-vs-fixed", ext="mp4")
    anim.save(pv, writer=FFMpegWriter(fps=18, bitrate=3500), dpi=110)
    plt.close(fig); print("wrote", pv)


if __name__ == "__main__":
    main()
