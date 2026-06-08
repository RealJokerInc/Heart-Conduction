"""S0b — Was it the PARAMETERS? Re-run the (working) S0 eikonal-circle test at the
hourglass's parameters and watch the eikonal signal degrade.

Controlled isolation: identical geometry (expanding circle), vary ONLY
(dx, stencil). If the clean CV_n=CV0-D*kappa recovery (S0: dx=50um, moore8_iso)
collapses at the hourglass params (dx=250um, cardinal4), then resolution+stencil
— not the model — caused the hourglass's inability to show source-sink curvature.

(This isolates resolution/stencil only; the hourglass also had a reservoir-fed
over-driven source and healthy excitability — separate factors, tested in S2/S3.)
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

LX = LY = 1.0
D = 0.001
T_END = 18.0
SAVE_EVERY = 0.25
CX = CY = LX / 2.0
R_STIM = 0.06
THR = -40.0
R_IN, R_OUT = 0.12, 0.40

# (label, dx_cm, stencil, boundary_mode, dt)  -- dt obeys explicit CFL D*dt/dx^2<=0.25
CASES = [
    ("S0 baseline (50um, moore8_iso)", 0.005, "moore8_iso", "face_mirror_iso", 0.004),
    ("hourglass dx, iso stencil (250um, moore8_iso)", 0.025, "moore8_iso", "face_mirror_iso", 0.02),
    ("HOURGLASS params (250um, cardinal4)", 0.025, "cardinal4", "face_mirror", 0.02),
]


def run_case(dx, stencil, bmode, dt, dev):
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol
    from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
    from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation
    nx = int(round(LX / dx)) + 1; ny = int(round(LY / dx)) + 1
    cache = REPO / f"media/source_sink_mismatch_investigation/_sim_outputs/s0b_dx{int(dx*1e4)}_{stencil}.npz"
    if cache.exists():
        z = np.load(cache); return z["times"], z["V"], dx, nx, ny
    mask = torch.ones((nx, ny), dtype=torch.bool, device=dev)
    grid = StructuredGrid.from_mask(mask, dx, dx, device=str(dev))
    fdm = FDMDiscretization(grid, D=D, chi=1.0, Cm=1.0, stencil=stencil, boundary_mode=bmode)
    proto = StimulusProtocol()
    proto.add_stimulus(region=lambda x, y: (x - CX) ** 2 + (y - CY) ** 2 < R_STIM ** 2,
                       start_time=0.0, duration=2.0, amplitude=-52.0)
    sim = MonodomainSimulation(spatial=fdm, ionic_model='ttp06', stimulus=proto, dt=dt,
                               splitting='strang', ionic_solver='rush_larsen',
                               diffusion_solver='forward_euler', cell_type='EPI')
    t0 = time.time(); times, Vflat = sim.run_to_array(t_end=T_END, save_every=SAVE_EVERY)
    V = np.full((len(times), nx, ny), -85.23, dtype=np.float32)
    V[:, mask.cpu().numpy()] = np.asarray(Vflat, dtype=np.float32)
    print(f"    ran {nx}x{ny} dt={dt} in {time.time()-t0:.0f}s")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, times=times, V=V)
    return np.asarray(times, float), V, dx, nx, ny


def integrated_fit(times, V, dx, nx, ny):
    """LAT(r)=r/CV0+(D/CV0^2)ln r+c -> CV0,D_eik,r*,R2. Returns (rc,lc,fit)."""
    lat = activation_time_interp(V, times, THR)
    i = np.arange(nx) * dx; j = np.arange(ny) * dx
    X, Y = np.meshgrid(i, j, indexing="ij"); R = np.hypot(X - CX, Y - CY)
    fin = np.isfinite(lat); r_cell = R[fin]; lat_cell = lat[fin]
    nbin = max(6, int((R_OUT - R_IN) / (2 * dx)))
    edges = np.linspace(R_IN, R_OUT, nbin + 1); rc, lc = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (r_cell >= a) & (r_cell < b)
        if sel.sum() < 8:
            continue
        rc.append(0.5 * (a + b)); lc.append(np.median(lat_cell[sel]))
    rc = np.array(rc); lc = np.array(lc)
    if len(rc) < 4:
        return rc, lc, dict(CV0=np.nan, D_eik=np.nan, r2=np.nan, r_star=np.nan, bins=len(rc))
    A = np.vstack([rc, np.log(rc), np.ones_like(rc)]).T
    (a_, b_, c_), *_ = np.linalg.lstsq(A, lc, rcond=None)
    CV0 = 1 / a_; D_eik = b_ * CV0 ** 2
    pred = A @ np.array([a_, b_, c_])
    ss_res = float(np.sum((lc - pred) ** 2)); ss_tot = float(np.sum((lc - lc.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return rc, lc, dict(CV0=CV0, D_eik=D_eik, r2=r2, r_star=D_eik / CV0, bins=len(rc))


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    rows = []
    for label, dx, stencil, bmode, dt in CASES:
        print(f"[{label}]")
        times, V, dx, nx, ny = run_case(dx, stencil, bmode, dt, dev)
        rc, lc, fit = integrated_fit(times, V, dx, nx, ny)
        rows.append((label, dx, fit))
        print(f"    CV0={fit['CV0']*1000:.1f}cm/s  D_eik={fit['D_eik']:.5f}  "
              f"D_eik/D={fit['D_eik']/D:.2f}  r*={fit['r_star']*1e4:.0f}um  "
              f"R2={fit['r2']:.4f}  r*/dx={fit['r_star']/dx:.2f}  bins={fit['bins']}")
        if len(rc):
            cvr = 1.0 / np.gradient(lc, rc)
            ax.plot(1.0 / rc, cvr * 1000.0, "o-", ms=4, alpha=0.85,
                    label=f"{label}\n  D_eik/D={fit['D_eik']/D:.2f}, R2={fit['r2']:.3f}, r*/dx={fit['r_star']/dx:.1f}")
    ax.set_xlabel("curvature kappa = 1/r (1/cm)"); ax.set_ylabel("CV_n (cm/s)")
    ax.set_title("S0b — same eikonal circle, hourglass params degrade the signal")
    ax.legend(fontsize=8, loc="best")
    p = media_path("source_sink_mismatch_investigation", "images", "s0b-param-degradation")
    fig.savefig(p, dpi=120); plt.close(fig); print("wrote", p)

    print("\n  SUMMARY (D_eik/D near 1.0 = curvature resolved):")
    for label, dx, fit in rows:
        print(f"   dx={dx*1e4:.0f}um  D_eik/D={fit['D_eik']/D:5.2f}  R2={fit['r2']:.4f}  r*/dx={fit['r_star']/dx:.2f}  | {label}")


if __name__ == "__main__":
    main()
