"""S0e — DX SWEEP on the hourglass (moore8_iso), isochrones for visual confirmation.

Question: as dx refines, does the dilation inverse crescent CONVERGE (physical, was
under-resolved) or PEAK-then-DIE (numerical artifact, a la the boundary-speedup BC that
vanishes with refinement unless dx ~ AP wavelength)? Visual front (isochrones/video) is
ground truth here — derived crescent scalars proved unreliable.

dx sweep at FIXED geometry + FIXED wavelength (same ionic model). Step 2 (separate) then
varies the AP wavelength at fixed dx.
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
from run_s0d_hourglass_confirm import run_arm, hourglass_fluid, LX, LY, X_NECK, X_END, THR

# dx (cm) -> dt obeying explicit CFL D*dt/dx^2 <= 0.25 (D=0.001)
SWEEP = [
    (0.025,  0.02),
    (0.0167, 0.02),
    (0.0125, 0.02),
    (0.0083, 0.015),
    (0.005,  0.004),
]
FINE = [(0.0037, 0.003), (0.0025, 0.0014)]   # background tail (convergence test)


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--fine", action="store_true")
    args = ap.parse_args()
    sweep = SWEEP + (FINE if args.fine else [])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(sweep)
    ncol = 2; nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.0 * nrow), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for (dx, dt), ax in zip(sweep, axes):
        times, V, dx, nx, ny, fluid = run_arm(dx, "moore8_iso", "face_mirror_iso", dt, dev)
        lat = activation_time_interp(V, times, THR)
        xs = np.arange(nx) * dx; ys = np.arange(ny) * dx
        Lm = np.ma.array(lat.T, mask=(~fluid).T)
        ax.contour(xs, ys, Lm, levels=np.arange(0, np.nanmax(lat), 0.5), colors="k", linewidths=0.6)
        ax.imshow(np.ma.array((~fluid).astype(float).T, mask=fluid.T), origin="lower",
                  extent=[0, LX, 0, LY], cmap="Greys", alpha=0.35, aspect="equal")
        ax.axvline(X_NECK, color="red", ls=":", lw=0.8)
        ax.set_xlim(X_NECK - 0.05, X_END + 0.05); ax.set_ylim(LY / 2 - 0.02, LY)
        ax.set_aspect("equal")
        ax.set_title(f"dx={dx*1e4:.0f}um  (r*/dx={134/(dx*1e4):.1f})", fontsize=10)
        ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Hourglass dilation isochrones (0.5ms) vs dx — does the wall crescent converge or die?", fontsize=11)
    tag = "s0e-dx-sweep-isochrones" + ("-fine" if args.fine else "")
    p = media_path("source_sink_mismatch_investigation", "images", tag)
    fig.savefig(p, dpi=140); plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    main()
