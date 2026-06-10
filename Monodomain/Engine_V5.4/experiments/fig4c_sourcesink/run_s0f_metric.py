"""S0f metric — wall-minus-center crescent in the CONVERGING region vs dx.

In convergence there is no geometric fan (a planar wave entering a converging channel
stays planar absent a boundary effect), so wall-center LAT is a clean boundary-crescent
measure: NEGATIVE = walls LEAD center = inverse crescent (boundary speedup). Tracks vs dx:
does it converge (physical) or peak-then-die (artifact)?
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
from run_s0d_hourglass_confirm import hourglass_fluid, LX, LY, X_START, X_NECK, X_END, THR

CAND = [0.025, 0.0167, 0.0125, 0.0083, 0.005, 0.0037, 0.0025]


def crescent_region(lat, dx, nx, ny, fluid, x0, x1):
    xs = np.arange(nx) * dx; vals = []
    for i in range(nx):
        if not (x0 <= xs[i] <= x1):
            continue
        js = np.where(fluid[i])[0]
        if js.size < 7:
            continue
        jt, jb, jc = js[-1], js[0], js[len(js) // 2]
        lw = 0.5 * (lat[i, jt] + lat[i, jb]); lc = lat[i, jc]
        if np.isfinite(lw) and np.isfinite(lc):
            vals.append(lw - lc)
    return float(np.nanmean(vals)) if vals else np.nan


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = []
    for dx in CAND:
        p = REPO / f"media/source_sink_mismatch_investigation/_sim_outputs/s0d_dx{int(dx*1e4)}_moore8_iso.npz"
        if not p.exists():
            continue
        z = np.load(p); times, V = z["times"], z["V"]
        nx, ny = V.shape[1], V.shape[2]
        fluid = hourglass_fluid(dx, nx, ny, torch.device("cpu")).numpy()
        lat = activation_time_interp(V, times, THR)
        conv = crescent_region(lat, dx, nx, ny, fluid, X_START + 0.15, X_NECK - 0.05)
        dil = crescent_region(lat, dx, nx, ny, fluid, X_NECK + 0.05, X_END - 0.05)
        rows.append((dx, conv, dil))
        print(f"  dx={dx*1e4:4.0f}um (r*/dx={134/(dx*1e4):.1f})  CONVERGING wall-center={conv*1000:+7.0f}us   "
              f"diverging={dil*1000:+7.0f}us")
    dxs = np.array([r[0] for r in rows]) * 1e4
    conv = np.array([r[1] for r in rows]) * 1000
    dil = np.array([r[2] for r in rows]) * 1000
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(dxs, conv, "o-", label="CONVERGING (real test)")
    ax.plot(dxs, dil, "s--", color="0.6", label="diverging (geometric fan)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("dx (um)"); ax.set_ylabel("wall - center LAT (us)  (<0 = inverse crescent)")
    ax.invert_xaxis(); ax.set_title("Hourglass wall-center crescent vs dx (converge or die?)")
    ax.legend(); ax.grid(alpha=0.3)
    p = media_path("source_sink_mismatch_investigation", "images", "s0f-crescent-vs-dx")
    fig.savefig(p, dpi=130); plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    main()
