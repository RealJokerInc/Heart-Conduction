"""S0f — the REAL test: the CONVERGING half of the hourglass across dx.

The diverging half makes a forward crescent trivially (geometric fan). The converging
half has no fan tendency, so any inverse crescent (front bowing toward +x at the slanted
walls = edges leading) is the boundary effect/artifact. Visual isochrones, dx sweep,
from existing full-hourglass caches (no re-sim). Does it strengthen-and-converge or
peak-then-die as dx refines?
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
from run_s0d_hourglass_confirm import hourglass_fluid, LX, LY, X_START, X_NECK, THR

CAND = [0.025, 0.0167, 0.0125, 0.0083, 0.005, 0.0037, 0.0025]


def available():
    out = []
    for dx in CAND:
        p = REPO / f"media/source_sink_mismatch_investigation/_sim_outputs/s0d_dx{int(dx*1e4)}_moore8_iso.npz"
        if p.exists():
            out.append((dx, p))
    return out


def main():
    arms = available()
    print("dx available (um):", [int(dx*1e4) for dx, _ in arms])
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(arms); ncol = 3; nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.4 * nrow), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for (dx, p), ax in zip(arms, axes):
        z = np.load(p); times, V = z["times"], z["V"]
        nx, ny = V.shape[1], V.shape[2]
        fluid = hourglass_fluid(dx, nx, ny, torch.device("cpu")).numpy()
        lat = activation_time_interp(V, times, THR)
        xs = np.arange(nx) * dx; ys = np.arange(ny) * dx
        Lm = np.ma.array(lat.T, mask=(~fluid).T)
        ax.contour(xs, ys, Lm, levels=np.arange(0, np.nanmax(lat), 0.5), colors="k", linewidths=0.6)
        ax.imshow(np.ma.array((~fluid).astype(float).T, mask=fluid.T), origin="lower",
                  extent=[0, LX, 0, LY], cmap="Greys", alpha=0.35, aspect="equal")
        ax.set_xlim(X_START + 0.15, X_NECK - 0.02)          # CONVERGING region only
        ax.set_ylim(LY / 2 - 0.02, LY / 2 + 0.45)           # upper half (center -> top wall)
        ax.set_aspect("equal")
        ax.set_title(f"dx={dx*1e4:.0f}um  (r*/dx={134/(dx*1e4):.1f})", fontsize=10)
        ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("CONVERGING half — isochrones (0.5ms) vs dx. Inverse crescent = front bows toward +x at the slanted wall (edges lead)", fontsize=11)
    p = media_path("source_sink_mismatch_investigation", "images", "s0f-converging-dx-sweep-isochrones")
    fig.savefig(p, dpi=140); plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    main()
