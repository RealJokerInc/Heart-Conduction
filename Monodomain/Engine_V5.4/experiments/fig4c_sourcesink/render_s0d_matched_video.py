"""Matched-dx video: cardinal4 vs moore8_iso, BOTH at dx=50um — isolates the stencil
(diagonal connectivity) from resolution. The original orig-vs-fixed video confounded
the two (250um vs 50um). If the inverse crescent is stencil-driven it appears only on
the right; if it was resolution, both look identical.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path("/home/norepinephrine/Documents/Heart-Conduction")
sys.path.insert(0, str(REPO / "Monodomain/Engine_V5.4"))
sys.path.insert(0, str(REPO))
from cardiac_core.media import media_path
from run_s0d_hourglass_confirm import hourglass_fluid, LX, LY, X_NECK

ARMS = [
    ("cardinal4 (dx=50um)", 0.005, "cardinal4"),
    ("moore8_iso (dx=50um)", 0.005, "moore8_iso"),
]


def load(dx, stencil):
    z = np.load(REPO / f"media/source_sink_mismatch_investigation/_sim_outputs/s0d_dx{int(dx*1e4)}_{stencil}.npz")
    times, V = z["times"], z["V"]
    solid = ~hourglass_fluid(dx, V.shape[1], V.shape[2], torch.device("cpu")).numpy()
    return times, V, solid


def main():
    data = [(lab, *load(dx, st)) for lab, dx, st in ARMS]
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
        im = ax.imshow(np.ma.array(V[0].T, mask=solid.T), origin="lower", extent=ext, cmap=cmap,
                       vmin=-85, vmax=40, aspect="equal", interpolation="nearest")
        ax.axvline(X_NECK, color="cyan", ls=":", lw=1)
        ax.set_title(lab); ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
        ims.append((im, V, solid))
    sup = fig.suptitle("hourglass MATCHED dx=50um — t=0.0 ms")

    def upd(k):
        for im, V, solid in ims:
            im.set_array(np.ma.array(V[k].T, mask=solid.T))
        sup.set_text(f"hourglass MATCHED dx=50um  (cardinal4 vs moore8_iso)  t={times[k]:.1f} ms")
        return [im for im, _, _ in ims]

    anim = FuncAnimation(fig, upd, frames=range(0, nfr, max(1, nfr // 220)), blit=False)
    pv = media_path("source_sink_mismatch_investigation", "videos", "s0d-hourglass-matched-dx-cardinal-vs-moore8", ext="mp4")
    anim.save(pv, writer=FFMpegWriter(fps=18, bitrate=3500), dpi=110)
    plt.close(fig); print("wrote", pv)


if __name__ == "__main__":
    main()
