"""S0d — CONFIRM on the hourglass: original params miss the dilation curvature;
fixed params (moore8_iso + resolved dx) recover it. Constriction stays flat in
both (convergence = correct physics, not a failure).

Geometry = diag_hourglass (wide base -> thin neck -> wide dilation), scaled 1/3
so fine dx is affordable (shape preserved; r*=D/CV0~134um is physical, unchanged).
Planar wave enters the wide base, through the neck, out the dilation.

Metric: centerline CV(x) (constriction left of neck, dilation right) + front
curvature/CV in the dilation via cardiac_core front_metrics.
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
from cardiac_core.analysis import activation_time_interp, front_metrics

# scaled hourglass (1/3 of the original 6x4)
LX, LY = 2.0, 1.4
W_BASE, W_NECK = 0.5, 0.03           # half-widths (cm); neck full = 0.6mm
X_START, X_NECK, X_END = 0.2, 1.0, 1.8
THR = -40.0
T_END = 60.0
SAVE_EVERY = 0.25
D = 0.001

# (label, dx, stencil, bmode, dt)
ARMS = [
    ("orig (cardinal4, dx=250um)",  0.025, "cardinal4",  "face_mirror",     0.02),
    ("fixed (moore8_iso, dx=50um)", 0.005, "moore8_iso", "face_mirror_iso", 0.004),
]


def hw_of_x(xc):
    tL = (xc - X_START) / (X_NECK - X_START)
    tR = (X_END - xc) / (X_END - X_NECK)
    t = np.clip(np.where(xc < X_NECK, tL, tR), 0.0, 1.0)
    return W_BASE + (W_NECK - W_BASE) * t


def hourglass_fluid(dx, nx, ny, dev):
    jc = ny // 2
    x = np.arange(nx) * dx
    hw_cells = np.round(hw_of_x(x) / dx).astype(int)
    mask = np.zeros((nx, ny), dtype=bool)
    for i in range(nx):
        lo = max(0, jc - hw_cells[i]); hi = min(ny, jc + hw_cells[i] + 1)
        mask[i, lo:hi] = True
    return torch.tensor(mask, device=dev)


def run_arm(dx, stencil, bmode, dt, dev):
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol
    from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
    from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation
    nx = int(round(LX / dx)) + 1; ny = int(round(LY / dx)) + 1
    cache = REPO / f"media/source_sink_mismatch_investigation/_sim_outputs/s0d_dx{int(dx*1e4)}_{stencil}.npz"
    fluid = hourglass_fluid(dx, nx, ny, dev)
    if cache.exists():
        z = np.load(cache); return z["times"], z["V"], dx, nx, ny, fluid.cpu().numpy()
    grid = StructuredGrid.from_mask(fluid, dx, dx, device=str(dev))
    fdm = FDMDiscretization(grid, D=D, chi=1.0, Cm=1.0, stencil=stencil, boundary_mode=bmode)
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


def centerline_cv(lat, dx, nx, ny):
    jc = ny // 2
    latc = lat[:, jc]; cv = np.full(nx, np.nan); x = np.arange(nx) * dx
    span = max(3, int(round(0.06 / dx)))            # ~0.6mm finite-diff span
    for i in range(span, nx - span):
        d = latc[i + span] - latc[i - span]
        if np.isfinite(d) and d > 0:
            cv[i] = (2 * span * dx) / d * 1000.0
    return x, cv


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"S0d hourglass confirm: base hw={W_BASE} -> neck hw={W_NECK} @x={X_NECK}  dev={dev}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    summ = []
    for label, dx, stencil, bmode, dt in ARMS:
        print(f"[{label}]")
        times, V, dx, nx, ny, fluid = run_arm(dx, stencil, bmode, dt, dev)
        lat = activation_time_interp(V, times, THR)
        x, cv = centerline_cv(lat, dx, nx, ny)
        # dilation curvature: front_metrics kappa/CV_n averaged over the dilation centerline band
        m = front_metrics(lat, dx)
        jc = ny // 2; band = slice(max(0, jc - 2), jc + 3)
        dil = (x >= X_NECK + 0.1) & (x <= X_END)
        ix = np.where(dil)[0]
        kap = np.nanmedian(m["kappa"][ix][:, band])
        cvn = np.nanmedian(m["cv_n"][ix][:, band]) * 1000.0
        # CV at constriction vs dilation (centerline)
        cv_neck = np.nanmedian(cv[(x > X_NECK - 0.2) & (x < X_NECK)])         # just before neck (constriction)
        cv_dil = np.nanmin(cv[(x > X_NECK) & (x < X_END)])                    # min in dilation
        cv_far = np.nanmedian(cv[(x > X_END) & (x < LX - 0.1)])               # recovered downstream
        summ.append((label, dx, cv_neck, cv_dil, cv_far, kap, cvn))
        print(f"    constriction CV~{cv_neck:.1f}  dilation CVmin~{cv_dil:.1f}  far CV~{cv_far:.1f} cm/s "
              f"| dilation kappa~{kap:.2f}/cm CV_n~{cvn:.1f}")
        ax.plot(x, cv, lw=1.8, label=f"{label}")
    ax.axvline(X_NECK, color="red", ls=":", lw=1, label="neck")
    ax.set_xlabel("x (cm)"); ax.set_ylabel("centerline CV (cm/s)")
    ax.set_title("S0d hourglass — dilation CV slowing: original (flat/muted) vs fixed (resolved)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(0, LX)
    p = media_path("source_sink_mismatch_investigation", "images", "s0d-hourglass-confirm")
    fig.savefig(p, dpi=120); plt.close(fig); print("wrote", p)
    print("\n  SUMMARY (dilation CVmin << constriction/far = convex slowing resolved):")
    for label, dx, cvn_, cvd, cvf, kap, cvnn in summ:
        drop = (1 - cvd / cvf) * 100 if np.isfinite(cvd) and np.isfinite(cvf) and cvf > 0 else float('nan')
        print(f"   {label}: constr={cvn_:.1f} dilMIN={cvd:.1f} far={cvf:.1f} cm/s  dilation-dip={drop:+.0f}%  kappa={kap:.2f}/cm")


if __name__ == "__main__":
    main()
