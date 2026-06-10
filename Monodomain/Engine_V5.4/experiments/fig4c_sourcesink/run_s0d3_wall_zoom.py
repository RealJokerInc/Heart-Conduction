"""S0d3 — zoom on the dilation WALL: is the inverse crescent stencil-dependent at
fixed dx? Local crescent = LAT[wall] - LAT[wall - K cells] (K~r*). NEGATIVE = the
edge LEADS its inner neighbour = inverse crescent (boundary speedup). The global
convex fan is subtracted out by comparing locally, not wall-vs-far-center.

Uses cached S0d/S0d2 runs (no re-sim). Key controlled pair: both at dx=50um.
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

ARMS = [
    ("cardinal4",  0.025), ("moore8_iso", 0.025),
    ("cardinal4",  0.005), ("moore8_iso", 0.005),
]


def load(stencil, dx):
    z = np.load(REPO / f"media/source_sink_mismatch_investigation/_sim_outputs/s0d_dx{int(dx*1e4)}_{stencil}.npz")
    times, V = z["times"], z["V"]
    nx, ny = V.shape[1], V.shape[2]
    fluid = hourglass_fluid(dx, nx, ny, torch.device("cpu")).numpy()
    return times, V, nx, ny, fluid


def local_crescent(lat, dx, nx, ny, fluid):
    """LAT[topwall] - LAT[topwall - K], K ~ r*(134um). <0 = edge leads inner."""
    K = max(2, int(round(0.0134 / dx)))      # ~ r*
    xs = np.arange(nx) * dx; xc, cr = [], []
    for i in range(nx):
        if not (X_NECK + 0.05 <= xs[i] <= X_END):
            continue
        js = np.where(fluid[i])[0]
        if js.size < 2 * K + 3:
            continue
        jt = js[-1]; jb = js[0]
        ct = lat[i, jt] - lat[i, jt - K]
        cb = lat[i, jb] - lat[i, jb + K]
        if np.isfinite(ct) and np.isfinite(cb):
            xc.append(xs[i]); cr.append(0.5 * (ct + cb))
    return np.array(xc), np.array(cr), K


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    figc, axc = plt.subplots(figsize=(9, 5), constrained_layout=True)
    order = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (stencil, dx), (r, c) in zip(ARMS, order):
        times, V, nx, ny, fluid = load(stencil, dx)
        lat = activation_time_interp(V, times, THR)
        xc, cr, K = local_crescent(lat, dx, nx, ny, fluid)
        mc = float(np.nanmean(cr)) if cr.size else np.nan
        sign = "INVERSE (edge leads)" if mc < -0.02 else ("forward (edge lags)" if mc > 0.02 else "~flat")
        print(f"  {stencil:11s} dx={dx*1e4:3.0f}um  K={K}cells  local edge-inner = {mc*1000:+6.0f} us -> {sign}")
        ax = axes[r, c]
        ext = [0, LX, 0, LY]
        Lm = np.ma.array(lat.T, mask=(~fluid).T)
        ax.imshow(Lm, origin="lower", extent=ext, cmap="viridis", aspect="auto")
        xs = np.linspace(0, LX, nx); ys = np.linspace(0, LY, ny)
        ax.contour(xs, ys, Lm, levels=np.arange(0, np.nanmax(lat), 0.25), colors="white", linewidths=0.7, alpha=0.9)
        # zoom on the TOP wall of the mid-dilation
        ax.set_xlim(X_NECK + 0.1, X_END - 0.05)
        # top wall y at mid-dilation
        i_mid = int((X_NECK + 0.45) / dx); js = np.where(fluid[min(i_mid, nx-1)])[0]
        ywall = (js[-1] * dx) if js.size else LY * 0.8
        ax.set_ylim(LY / 2 + 0.05, ywall + 0.06)
        ax.set_title(f"{stencil}, dx={dx*1e4:.0f}um   local edge-inner={mc*1000:+.0f}us", fontsize=10)
        ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
        axc.plot(xc, np.array(cr) * 1000.0, lw=1.8, label=f"{stencil} dx={dx*1e4:.0f}um")
    fig.suptitle("Dilation TOP-WALL zoom — dense isochrones (0.25ms). Inverse crescent = isochrones bow toward +x at the wall", fontsize=10)
    p1 = media_path("source_sink_mismatch_investigation", "images", "s0d3-wall-zoom-isochrones")
    fig.savefig(p1, dpi=130); plt.close(fig); print("wrote", p1)
    axc.axhline(0, color="k", lw=0.6); axc.set_xlabel("x (cm)")
    axc.set_ylabel("local edge-inner LAT (us)  (<0 = inverse crescent)")
    axc.set_title("Local boundary crescent in the dilation (edge vs ~r* inward), dx-controlled")
    axc.legend(fontsize=8); axc.grid(alpha=0.3)
    p2 = media_path("source_sink_mismatch_investigation", "images", "s0d3-local-crescent")
    figc.savefig(p2, dpi=120); plt.close(figc); print("wrote", p2)


if __name__ == "__main__":
    main()
