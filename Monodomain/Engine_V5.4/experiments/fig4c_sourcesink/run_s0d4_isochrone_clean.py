"""S0d4 — clean isochrones. Threshold V -> LAT, draw the front contours, equal aspect.
4 arms (rows=dx 250/50um, cols=cardinal4|moore8_iso). Dilation only. Look at whether
the front bows toward +x at the wall (inverse crescent) — no derived metric.
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
from cardiac_core.analysis import activation_time_interp
from run_s0d_hourglass_confirm import hourglass_fluid, LX, LY, X_NECK, X_END, THR

ARMS = [("cardinal4", 0.025), ("moore8_iso", 0.025),
        ("cardinal4", 0.005), ("moore8_iso", 0.005)]


def load(stencil, dx):
    z = np.load(REPO / f"media/source_sink_mismatch_investigation/_sim_outputs/s0d_dx{int(dx*1e4)}_{stencil}.npz")
    return z["times"], z["V"], dx, hourglass_fluid(dx, z["V"].shape[1], z["V"].shape[2], torch.device("cpu")).numpy()


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for (stencil, dx), ax in zip(ARMS, axes.ravel()):
        times, V, dx, fluid = load(stencil, dx)
        nx, ny = V.shape[1], V.shape[2]
        lat = activation_time_interp(V, times, THR)
        xs = np.arange(nx) * dx; ys = np.arange(ny) * dx
        Lm = np.ma.array(lat.T, mask=(~fluid).T)
        ax.contour(xs, ys, Lm, levels=np.arange(0, np.nanmax(lat), 0.5),
                   colors="k", linewidths=0.8)
        ax.imshow(np.ma.array((~fluid).astype(float).T, mask=fluid.T),
                  origin="lower", extent=[0, LX, 0, LY], cmap="Greys", alpha=0.3, aspect="equal")
        ax.set_xlim(X_NECK, X_END); ax.set_ylim(LY / 2 - 0.02, LY)
        ax.set_aspect("equal")
        ax.set_title(f"{stencil}, dx={dx*1e4:.0f}um", fontsize=11)
        ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
    fig.suptitle("Hourglass dilation isochrones (0.5ms), equal aspect — front bowing +x at wall = inverse crescent", fontsize=11)
    p = media_path("source_sink_mismatch_investigation", "images", "s0d4-isochrones-clean")
    fig.savefig(p, dpi=140); plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    main()
