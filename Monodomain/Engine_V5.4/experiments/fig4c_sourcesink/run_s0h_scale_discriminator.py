"""S0h — dx/constriction vs dx/r*? Scale the WHOLE geometry AND dx by sigma together
(cell count fixed, dx/constriction fixed), while r*=D/CV0~134um is physical and does
NOT scale. Converging crescent vs sigma:
  invariant  -> dx/constriction governs (geometry-resolution ratio)
  grows as sigma->small (finer dx/r*) -> dx/r* governs (electrotonic-front ratio)
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path("/home/norepinephrine/Documents/Heart-Conduction")
sys.path.insert(0, str(REPO / "Monodomain/Engine_V5.4"))
sys.path.insert(0, str(REPO))
from cardiac_core.media import media_path
from cardiac_core.analysis import activation_time_interp

D = 0.001; THR = -40.0
# base (sigma=1) geometry, cm
B = dict(LX=2.0, LY=1.4, W_BASE=0.5, W_NECK=0.03, X_START=0.2, X_NECK=1.0, X_END=1.8)
DX1 = 0.01           # dx at sigma=1 -> dx/neck(full=0.06) = 0.167, constant across sigma
# (sigma, dt obeying CFL D*dt/dx^2<=0.25 with dx=DX1*sigma)
RUNS = [(2.0, 0.05), (1.0, 0.02), (0.5, 0.004)]


def geom(sigma):
    g = {k: v * sigma for k, v in B.items()}
    g["dx"] = DX1 * sigma
    return g


def hw_of_x(xc, g):
    tL = (xc - g["X_START"]) / (g["X_NECK"] - g["X_START"])
    tR = (g["X_END"] - xc) / (g["X_END"] - g["X_NECK"])
    t = np.clip(np.where(xc < g["X_NECK"], tL, tR), 0.0, 1.0)
    return g["W_BASE"] + (g["W_NECK"] - g["W_BASE"]) * t


def fluid(g, dev):
    dx = g["dx"]; nx = int(round(g["LX"] / dx)) + 1; ny = int(round(g["LY"] / dx)) + 1
    jc = ny // 2; x = np.arange(nx) * dx
    hwc = np.round(hw_of_x(x, g) / dx).astype(int)
    m = np.zeros((nx, ny), bool)
    for i in range(nx):
        lo = max(0, jc - hwc[i]); hi = min(ny, jc + hwc[i] + 1); m[i, lo:hi] = True
    return torch.tensor(m, device=dev), nx, ny


def run(sigma, dt, dev):
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol
    from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
    from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation
    g = geom(sigma); dx = g["dx"]
    fl, nx, ny = fluid(g, dev)
    cache = REPO / f"media/source_sink_mismatch_investigation/_sim_outputs/s0h_sig{int(sigma*100)}.npz"
    if cache.exists():
        z = np.load(cache); return z["times"], z["V"], g, nx, ny, fl.cpu().numpy()
    grid = StructuredGrid.from_mask(fl, dx, dx, device=str(dev))
    fdm = FDMDiscretization(grid, D=D, chi=1.0, Cm=1.0, stencil="moore8_iso", boundary_mode="face_mirror_iso")
    proto = StimulusProtocol()
    proto.add_stimulus(region=lambda x, y: x < 0.05 * sigma, start_time=0.0, duration=2.0, amplitude=-52.0)
    sim = MonodomainSimulation(spatial=fdm, ionic_model="ttp06", stimulus=proto, dt=dt,
                               splitting="strang", ionic_solver="rush_larsen",
                               diffusion_solver="forward_euler", cell_type="EPI")
    t0 = time.time(); times, Vflat = sim.run_to_array(t_end=40.0 * sigma, save_every=0.25 * sigma)
    fn = fl.cpu().numpy(); V = np.full((len(times), nx, ny), -85.23, dtype=np.float32)
    V[:, fn] = np.asarray(Vflat, dtype=np.float32)
    print(f"    sigma={sigma} {nx}x{ny} dx={dx*1e4:.0f}um dt={dt} {time.time()-t0:.0f}s")
    cache.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(cache, times=times, V=V)
    return np.asarray(times, float), V, g, nx, ny, fn


def crescent(times, V, g, nx, ny, fl):
    dx = g["dx"]; lat = activation_time_interp(V, times, THR)
    xs = np.arange(nx) * dx; vals = []
    for i in range(nx):
        if not (g["X_START"] + 0.15 * (g["LX"] / B["LX"]) <= xs[i] <= g["X_NECK"] - 0.05 * (g["LX"] / B["LX"])):
            continue
        js = np.where(fl[i])[0]
        if js.size < 7:
            continue
        jt, jb, jc = js[-1], js[0], js[len(js) // 2]
        if np.isfinite(lat[i, jt]) and np.isfinite(lat[i, jc]):
            vals.append(0.5 * (lat[i, jt] + lat[i, jb]) - lat[i, jc])
    return float(np.nanmean(vals)) if vals else np.nan


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for sigma, dt in RUNS:
        times, V, g, nx, ny, fl = run(sigma, dt, dev)
        c = crescent(times, V, g, nx, ny, fl)
        dxum = g["dx"] * 1e4; neck_full_um = 2 * g["W_NECK"] * 1e4
        rows.append((sigma, dxum, neck_full_um, c))
        print(f"  sigma={sigma}: dx={dxum:.0f}um  neck={neck_full_um:.0f}um  dx/neck={dxum/neck_full_um:.2f}  "
              f"r*/dx={134/dxum:.2f}  crescent={c*1000:+.0f}us")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sig = [r[0] for r in rows]; cr = [r[3]*1000 for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    ax.plot([r[1] for r in rows], cr, "o-")
    for s, dxum, nk, c in rows:
        ax.annotate(f"sig={s}\nr*/dx={134/dxum:.1f}", (dxum, c*1000), fontsize=8)
    ax.set_xlabel("dx (um)  [geometry scaled WITH dx -> dx/constriction FIXED]")
    ax.set_ylabel("converging crescent (us)")
    ax.set_title("S0h: crescent at FIXED dx/constriction.\nflat=dx/constriction governs; varies=dx/r* governs (r* unscaled)")
    ax.grid(alpha=0.3)
    p = media_path("source_sink_mismatch_investigation", "images", "s0h-scale-discriminator")
    fig.savefig(p, dpi=130); plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    main()
