"""S0d2 — isochrones + wall-vs-center crescent on the hourglass, with dx CONTROLLED.

Correction to S0d: the centerline-CV/median-kappa metric was blind to the
WALL-ADJACENT inverse crescent (edges leading center) — the connectivity-dependent
boundary-speedup signal. Here we (a) plot isochrones (LAT contours) so the front
shape is visible, and (b) compare cardinal4 vs moore8_iso at the SAME dx to separate
stencil (diagonal connectivity) from resolution.

Crescent metric (dilation only): mean(LAT at the two channel walls) - LAT at centerline,
per x-column. NEGATIVE = walls LEAD center = INVERSE crescent (boundary speedup).
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

# rows = dx (coarse, fine); cols = stencil (cardinal4, moore8_iso)
GRID = [
    ("cardinal4",  "face_mirror",     0.025, 0.02),
    ("moore8_iso", "face_mirror_iso", 0.025, 0.02),
    ("cardinal4",  "face_mirror",     0.005, 0.004),
    ("moore8_iso", "face_mirror_iso", 0.005, 0.004),
]


def wall_crescent(lat, dx, nx, ny, fluid):
    """per dilation x-column: mean(wall LAT) - center LAT. negative = walls lead."""
    xs = np.arange(nx) * dx
    xc, cres = [], []
    for i in range(nx):
        if not (X_NECK + 0.05 <= xs[i] <= X_END):
            continue
        js = np.where(fluid[i])[0]
        if js.size < 7:
            continue
        jt, jb, jcen = js[-1], js[0], js[len(js) // 2]
        lw = 0.5 * (lat[i, jt] + lat[i, jb]); lc = lat[i, jcen]
        if np.isfinite(lw) and np.isfinite(lc):
            xc.append(xs[i]); cres.append(lw - lc)
    return np.array(xc), np.array(cres)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    figc, axc = plt.subplots(figsize=(9, 5), constrained_layout=True)
    order = [(0, 0), (0, 1), (1, 0), (1, 1)]
    summ = []
    for (stencil, bmode, dx, dt), (r, c) in zip(GRID, order):
        times, V, dx, nx, ny, fluid = run_arm(dx, stencil, bmode, dt, dev)
        lat = activation_time_interp(V, times, THR)
        xc, cres = wall_crescent(lat, dx, nx, ny, fluid)
        mc = float(np.nanmean(cres)) if cres.size else np.nan
        summ.append((stencil, dx, mc))
        sign = "INVERSE (walls lead)" if mc < -0.05 else ("forward (walls lag)" if mc > 0.05 else "~flat")
        print(f"  {stencil:11s} dx={dx*1e4:3.0f}um  mean wall-center = {mc*1000:+6.0f} us  -> {sign}")
        # isochrone panel (zoom on dilation)
        ax = axes[r, c]
        ext = [0, LX, 0, LY]
        Lm = np.ma.array(lat.T, mask=(~fluid).T)
        ax.imshow(Lm, origin="lower", extent=ext, cmap="viridis", aspect="equal")
        xs = np.linspace(0, LX, nx); ys = np.linspace(0, LY, ny)
        mx = np.nanmax(lat)
        ax.contour(xs, ys, Lm, levels=np.arange(0, mx, 0.5), colors="white", linewidths=0.6, alpha=0.85)
        ax.axvline(X_NECK, color="red", ls=":", lw=1)
        ax.set_xlim(X_NECK - 0.1, X_END + 0.1)
        ax.set_title(f"{stencil}, dx={dx*1e4:.0f}um   wall-center={mc*1000:+.0f}us", fontsize=10)
        ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
        axc.plot(xc, np.array(cres) * 1000.0, lw=1.8, label=f"{stencil} dx={dx*1e4:.0f}um")
    p1 = media_path("source_sink_mismatch_investigation", "images", "s0d2-hourglass-isochrones")
    fig.suptitle("Hourglass dilation isochrones — rows: dx(250,50um); cols: cardinal4 | moore8_iso", fontsize=11)
    fig.savefig(p1, dpi=120); plt.close(fig); print("wrote", p1)
    axc.axhline(0, color="k", lw=0.6); axc.set_xlabel("x (cm)")
    axc.set_ylabel("wall - center LAT (us)  (<0 = inverse crescent)")
    axc.set_title("Wall-vs-center crescent in the dilation (dx-controlled)")
    axc.legend(fontsize=8); axc.grid(alpha=0.3)
    p2 = media_path("source_sink_mismatch_investigation", "images", "s0d2-wall-crescent")
    figc.savefig(p2, dpi=120); plt.close(figc); print("wrote", p2)
    print("\n  SUMMARY (controlled): does the INVERSE crescent depend on stencil at fixed dx?")
    for stencil, dx, mc in summ:
        print(f"   {stencil:11s} dx={dx*1e4:3.0f}um : wall-center {mc*1000:+.0f} us")


if __name__ == "__main__":
    main()
