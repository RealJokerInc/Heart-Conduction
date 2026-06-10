"""dx-sweep video grid on the hourglass (moore8_iso) — visual: does the dilation-wall
inverse crescent strengthen-and-converge or peak-then-die as dx refines?
All panels: same physical extent, time-synced. Reads s0d_ caches."""
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

DXS = [0.025, 0.0167, 0.0125, 0.0083, 0.005]   # 250..50 um


def load(dx):
    z = np.load(REPO / f"media/source_sink_mismatch_investigation/_sim_outputs/s0d_dx{int(dx*1e4)}_moore8_iso.npz")
    return z["times"], z["V"], ~hourglass_fluid(dx, z["V"].shape[1], z["V"].shape[2], torch.device("cpu")).numpy()


def main():
    data = [(dx, *load(dx)) for dx in DXS]
    nfr = min(V.shape[0] for (_, _, V, _) in data)
    times = data[0][1]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import imageio_ffmpeg
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    ext = [0, LX, 0, LY]; cmap = plt.cm.inferno.copy(); cmap.set_bad("0.55")
    nrow = len(DXS)
    fig, axes = plt.subplots(nrow, 1, figsize=(8, 2.0 * nrow), constrained_layout=True)
    ims = []
    for ax, (dx, t, V, solid) in zip(axes, data):
        im = ax.imshow(np.ma.array(V[0].T, mask=solid.T), origin="lower", extent=ext, cmap=cmap,
                       vmin=-85, vmax=40, aspect="equal", interpolation="nearest")
        ax.axvline(X_NECK, color="cyan", ls=":", lw=0.8)
        ax.set_xlim(X_NECK - 0.05, LX); ax.set_title(f"dx={dx*1e4:.0f}um  (r*/dx={134/(dx*1e4):.1f})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ims.append((im, V, solid))
    sup = fig.suptitle("hourglass dilation, dx sweep — t=0.0 ms")

    def upd(k):
        for im, V, solid in ims:
            im.set_array(np.ma.array(V[k].T, mask=solid.T))
        sup.set_text(f"hourglass dilation, dx sweep (moore8_iso)  t={times[k]:.1f} ms")
        return [im for im, _, _ in ims]

    anim = FuncAnimation(fig, upd, frames=range(0, nfr, max(1, nfr // 200)), blit=False)
    pv = media_path("source_sink_mismatch_investigation", "videos", "s0e-hourglass-dx-sweep", ext="mp4")
    anim.save(pv, writer=FFMpegWriter(fps=16, bitrate=4000), dpi=120)
    plt.close(fig); print("wrote", pv)


if __name__ == "__main__":
    main()
