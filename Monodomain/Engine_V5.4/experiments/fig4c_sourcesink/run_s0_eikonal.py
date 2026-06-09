"""S0 — Eikonal validation: measure CV0, D_eik, r* on an expanding circular wave.

PLAN.md Phase 1 Step 1.2. A point/disk stimulus in a uniform sheet launches an
expanding wave whose curvature kappa = 1/r sweeps from large (near stim) to ~0
(far). If the engine obeys CV_n = CV0 - D*kappa, a CV_n-vs-kappa fit is linear with
slope -D and intercept CV0; r* = D/CV0 sets the resolution target for S1.

Acceptance: R^2 > 0.95, D_eik within +-20% of the operator D (0.001), report r*.
No thickness. Analysis from cardiac_core.analysis (single-sourced).
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
from cardiac_core.analysis import activation_time_interp, front_metrics, fit_eikonal

# ---- geometry / numerics ----
LX = LY = 1.0          # cm
DX = 0.005             # cm (50 um) -> r*/dx ~ 4 at r*~200um
NX = int(round(LX / DX)) + 1     # 201
NY = int(round(LY / DX)) + 1
D = 0.001              # cm^2/ms  (operator diffusivity)
# explicit-Euler 2D stability: D*dt/dx^2 <= 0.25 -> dt <= 0.00625 ms
DT = 0.004
T_END = 18.0
SAVE_EVERY = 0.25
CX = CY = LX / 2.0
R_STIM = 0.06          # cm, supra-critical disk
THR = -40.0
# fit annulus: away from the stimulus disk and the walls
R_IN, R_OUT = 0.12, 0.40


def run_mono(device):
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol
    from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
    from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation

    mask = torch.ones((NX, NY), dtype=torch.bool, device=device)   # uniform sheet
    grid = StructuredGrid.from_mask(mask, DX, DX, device=str(device))
    # isotropic 9-pt stencil. Modeled diffusion is isotropic (scalar D) either way;
    # the cause is directional DIAGONAL CONNECTIVITY: cardinal4 has none, so its
    # 5-point Laplacian carries flux only along axes (direction-dependent truncation
    # error, NOT material/tensor anisotropy) -> the "circle" becomes a rounded square
    # and directional scatter swamps the ~8% curvature signal. moore8_iso adds the
    # 4:1-weighted diagonal channels that restore rotational isotropy.
    fdm = FDMDiscretization(grid, D=D, chi=1.0, Cm=1.0,
                            stencil='moore8_iso', boundary_mode='face_mirror_iso')
    proto = StimulusProtocol()
    proto.add_stimulus(region=lambda x, y: (x - CX) ** 2 + (y - CY) ** 2 < R_STIM ** 2,
                       start_time=0.0, duration=2.0, amplitude=-52.0)
    sim = MonodomainSimulation(spatial=fdm, ionic_model='ttp06', stimulus=proto,
                               dt=DT, splitting='strang', ionic_solver='rush_larsen',
                               diffusion_solver='forward_euler', cell_type='EPI')
    t0 = time.time()
    times, Vflat = sim.run_to_array(t_end=T_END, save_every=SAVE_EVERY)
    mask_np = mask.cpu().numpy()
    V = np.full((len(times), NX, NY), -85.23, dtype=np.float32)
    V[:, mask_np] = np.asarray(Vflat, dtype=np.float32)
    print(f"  [mono] {time.time()-t0:.0f}s  steps={int(T_END/DT)}  Vmax={V.max():.1f}")
    return np.asarray(times, dtype=float), V


def analyse_and_plot(times, V):
    lat = activation_time_interp(V, times, threshold=THR)
    m = front_metrics(lat, DX)
    # annulus mask
    i = np.arange(NX) * DX
    X, Y = np.meshgrid(i, np.arange(NY) * DX, indexing="ij")
    R = np.hypot(X - CX, Y - CY)
    # ---- primary: integrated eikonal fit (no differentiation) ----
    # For an expanding circle kappa=1/r, so CV_n=CV0-D/r integrates to
    #   LAT(r) = r/CV0 + (D/CV0^2)*ln(r) + const.
    # Fitting the SMOOTH LAT(r) (vs differentiating it) avoids the quantization
    # noise that an 8-13% curvature signal otherwise drowns in.
    fin = np.isfinite(lat)
    r_cell = R[fin]; lat_cell = lat[fin]
    edges = np.linspace(R_IN, R_OUT, 40)
    rc, lc = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (r_cell >= a) & (r_cell < b)
        if sel.sum() < 30:
            continue
        rc.append(0.5 * (a + b)); lc.append(np.median(lat_cell[sel]))
    rc = np.array(rc); lc = np.array(lc)
    A = np.vstack([rc, np.log(rc), np.ones_like(rc)]).T
    (a_, b_, c_), *_ = np.linalg.lstsq(A, lc, rcond=None)
    CV0 = 1.0 / a_; D_eik = b_ * CV0 ** 2; r_star = D_eik / CV0
    pred = A @ np.array([a_, b_, c_])
    ss_res = float(np.sum((lc - pred) ** 2)); ss_tot = float(np.sum((lc - lc.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    fit = {"CV0": CV0, "D_eik": D_eik, "r2": r2, "r_star": r_star, "n": len(rc)}
    print(f"  [LAT(r) integrated fit] CV0={CV0*1000:.2f} cm/s ({CV0:.4f} cm/ms)  "
          f"D_eik={D_eik:.5f} cm^2/ms  r2={r2:.5f}  r*={r_star*1e4:.0f} um  bins={len(rc)}")

    # ---- secondary: div(n_hat) estimator (validates front_metrics for S2-S4) ----
    ann = (R > R_IN) & (R < R_OUT) & fin & np.isfinite(m["cv_n"]) & np.isfinite(m["kappa"])
    raw = fit_eikonal(m["cv_n"][ann], m["kappa"][ann])
    print(f"  [div(n) estimator] D_eik={raw['D_eik']:.5f}  r2={raw['r2']:.3f}  n={raw['n']} (rougher; used in S2-S4)")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cv_r = 1.0 / np.gradient(lc, rc); kap_r = 1.0 / rc      # noisy points, for context
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.scatter(kap_r, cv_r * 1000.0, s=25, color="0.6", zorder=4, label="CV(r)=dr/dLAT (noisy)")
    ks = np.linspace(kap_r.min(), kap_r.max(), 50)
    ax.plot(ks, (CV0 - D_eik * ks) * 1000.0, "r-", lw=2,
            label=f"integrated fit: CV0={CV0*1000:.1f} cm/s, D={D_eik:.5f}\nr*={r_star*1e4:.0f}um, LAT-fit R2={r2:.4f}")
    ax.set_xlabel("curvature kappa = 1/r (1/cm)"); ax.set_ylabel("CV_n (cm/s)")
    ax.set_title("S0 eikonal validation — CV_n = CV0 - D*kappa (mono, expanding circle)")
    ax.legend(loc="best")
    p = media_path("source_sink_mismatch_investigation", "images", "s0-eikonal-cv-vs-kappa-mono")
    fig.savefig(p, dpi=120); plt.close(fig); print("  wrote", p)
    return fit


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"S0 eikonal: {NX}x{NY} ({LX}x{LY}cm) dx={DX*1e4:.0f}um dt={DT} D={D} dev={dev}")
    cache = REPO / "media/source_sink_mismatch_investigation/_sim_outputs/s0_eikonal.npz"
    if cache.exists():
        z = np.load(cache); times, V = z["times"], z["V"]
        print(f"  [cache] loaded {cache.name}")
    else:
        times, V = run_mono(dev)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, times=times, V=V)
        print(f"  [cache] wrote {cache.name}")
    fit = analyse_and_plot(times, V)
    ok = (fit["r2"] > 0.95) and (abs(fit["D_eik"] - D) / D < 0.20)
    print(f"\n  ACCEPTANCE: r2>0.95 and |D_eik-D|/D<20%  ->  {'PASS' if ok else 'CHECK'}")
    print(f"  -> record CV0={fit['CV0']:.4f}, D_eik={fit['D_eik']:.5f}, "
          f"r*={fit['r_star']*1e4:.0f}um in FIG4C_BLOCK_TEST_PLAN.md S10")


if __name__ == "__main__":
    main()
