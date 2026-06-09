"""S0c — diagonal connectivity is necessary but NOT sufficient: the leading-edge
source-sink crescent emerges only when r*=D/CV0 is RESOLVED (tune D or dx).

Setup: planar wave past a circular non-conducting obstacle (the infarct-boundary
geometry of diag_eikonal_circle / sim_semicircle). Measure the boundary-adjacent
LAT minus bulk-at-same-x:  negative = LEADS (inverse crescent / source-sink
speedup, fewer downstream sinks at the obstacle edge); positive = LAGS (crescent /
path-length shadow).

Cases vary the control knob r*/dx (r* = D/CV0 ~ 134um at D=0.001):
  1. coarse cardinal4   (no diagonal connectivity)        — historical
  2. coarse moore8_iso  (diagonal connectivity, coarse)   — diagonal alone
  3. coarse moore8_iso, D x4 (tune the diffusion param)   — raise r* at fixed dx
  4. fine   moore8_iso  (resolved, r*/dx~2.7)             — target
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

LX, LY = 2.0, 1.5
CXC, CYC, RC = 0.9, 0.75, 0.30      # circle obstacle (cm)
THR = -40.0
T_END = 30.0
SAVE_EVERY = 0.25

# (label, dx, D, stencil, bmode, dt)
CASES = [
    ("coarse cardinal4 (no diag)",      0.020, 0.001, "cardinal4",   "face_mirror",     0.02),
    ("coarse moore8_iso (diag only)",   0.020, 0.001, "moore8_iso",  "face_mirror_iso", 0.02),
    ("coarse moore8_iso, D x4 (tune D)",0.020, 0.004, "moore8_iso",  "face_mirror_iso", 0.02),
    ("fine moore8_iso (resolved)",      0.005, 0.001, "moore8_iso",  "face_mirror_iso", 0.004),
]


def run_case(dx, Dval, stencil, bmode, dt, dev):
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol
    from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
    from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation
    nx = int(round(LX / dx)) + 1; ny = int(round(LY / dx)) + 1
    tag = f"s0c_dx{int(dx*1e4)}_{stencil}_D{int(Dval*1e4)}"
    cache = REPO / f"media/source_sink_mismatch_investigation/_sim_outputs/{tag}.npz"
    ix = torch.arange(nx, device=dev).view(nx, 1).double() * dx
    iy = torch.arange(ny, device=dev).view(1, ny).double() * dx
    fluid = torch.sqrt((ix - CXC) ** 2 + (iy - CYC) ** 2) > RC      # True = tissue
    if cache.exists():
        z = np.load(cache); return z["times"], z["V"], dx, nx, ny, fluid.cpu().numpy()
    grid = StructuredGrid.from_mask(fluid, dx, dx, device=str(dev))
    fdm = FDMDiscretization(grid, D=Dval, chi=1.0, Cm=1.0, stencil=stencil, boundary_mode=bmode)
    proto = StimulusProtocol()
    proto.add_stimulus(region=lambda x, y: x < 0.05, start_time=0.0, duration=2.0, amplitude=-52.0)
    sim = MonodomainSimulation(spatial=fdm, ionic_model='ttp06', stimulus=proto, dt=dt,
                               splitting='strang', ionic_solver='rush_larsen',
                               diffusion_solver='forward_euler', cell_type='EPI')
    t0 = time.time(); times, Vflat = sim.run_to_array(t_end=T_END, save_every=SAVE_EVERY)
    fn = fluid.cpu().numpy()
    V = np.full((len(times), nx, ny), -85.23, dtype=np.float32)
    V[:, fn] = np.asarray(Vflat, dtype=np.float32)
    print(f"    ran {nx}x{ny} dt={dt} in {time.time()-t0:.0f}s")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, times=times, V=V)
    return np.asarray(times, float), V, dx, nx, ny, fn


def crescent(times, V, dx, nx, ny, fluid):
    """boundary-adjacent LAT minus bulk-at-same-x; split leading (x<cx) / trailing."""
    lat = activation_time_interp(V, times, THR)
    solid = ~fluid
    nbr = np.zeros_like(solid)
    nbr[1:, :] |= solid[:-1, :]; nbr[:-1, :] |= solid[1:, :]
    nbr[:, 1:] |= solid[:, :-1]; nbr[:, :-1] |= solid[:, 1:]
    bdry = nbr & fluid & np.isfinite(lat)
    ii, jj = np.where(bdry)
    xb = ii * dx
    ref_row = lat[:, max(2, int(0.05 / dx))]          # bulk reference row near bottom edge
    lead = np.array([lat[i, j] - ref_row[i] for i, j in zip(ii, jj)])
    front = xb < CXC; back = xb > CXC
    return (float(np.nanmean(lead[front])) if front.any() else np.nan,
            float(np.nanmean(lead[back])) if back.any() else np.nan, lat)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"S0c obstacle tuning: circle R={RC} at ({CXC},{CYC}) in {LX}x{LY}cm  dev={dev}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    rows = []
    for ax, (label, dx, Dval, stencil, bmode, dt) in zip(axes.ravel(), CASES):
        print(f"[{label}]")
        times, V, dx, nx, ny, fluid = run_case(dx, Dval, stencil, bmode, dt, dev)
        rstar = Dval / 0.0624 * np.sqrt(0.001 / Dval)   # r*=D_eik/CV0; CV0∝√D so r*∝√D
        # simpler: r* ~ D/CV0 with CV0 measured; approximate CV0(D)=0.0624*sqrt(D/0.001)
        cv0 = 0.0624 * np.sqrt(Dval / 0.001); rstar = Dval / cv0
        lead, back, lat = crescent(times, V, dx, nx, ny, fluid)
        rows.append((label, dx, Dval, rstar, rstar / dx, lead, back))
        print(f"    r*/dx={rstar/dx:.2f}  LEADING={lead*1000 if np.isfinite(lead) else float('nan'):+.0f}us  "
              f"TRAILING={back*1000 if np.isfinite(back) else float('nan'):+.0f}us")
        ext = [0, LX, 0, LY]
        Lm = np.ma.array(lat.T, mask=(~fluid).T)
        im = ax.imshow(Lm, origin="lower", extent=ext, cmap="viridis", aspect="equal")
        xs = np.linspace(0, LX, nx); ys = np.linspace(0, LY, ny)
        ax.contour(xs, ys, Lm, levels=np.arange(0, np.nanmax(lat), 2.0), colors="w", linewidths=0.5, alpha=0.8)
        th = np.linspace(0, 2*np.pi, 80); ax.plot(CXC+RC*np.cos(th), CYC+RC*np.sin(th), 'r-', lw=1.2)
        ax.set_title(f"{label}\nr*/dx={rstar/dx:.2f}  lead={lead*1000:+.0f}us  trail={back*1000:+.0f}us", fontsize=9)
        ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
    p = media_path("source_sink_mismatch_investigation", "images", "s0c-obstacle-rstar-tuning")
    fig.savefig(p, dpi=110); plt.close(fig); print("wrote", p)
    print("\n  SUMMARY (LEADING <0 = inverse crescent = source-sink speedup resolved):")
    print("   r*/dx   lead(us)  trail(us) | case")
    for label, dx, Dval, rstar, ratio, lead, back in rows:
        print(f"   {ratio:5.2f}   {lead*1000:+7.0f}  {back*1000:+8.0f} | {label}")


if __name__ == "__main__":
    main()
