"""S0g (Step 2) — WAVELENGTH test. Hold dx fixed; vary AP wavelength lambda = CV*APD
by scaling K-conductances (GKr,GKs) at fixed D (so CV and r* stay fixed). Does the
converging inverse crescent track lambda (-> dx/lambda artifact, user's hypothesis) or
stay invariant (-> governed by dx/r*, physical leading-front effect)?
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
from run_s0d_hourglass_confirm import hourglass_fluid, LX, LY, X_START, X_NECK, THR

DX = 0.0083; DT = 0.015; T2D = 40.0; SAVE = 0.25; D = 0.001
SCALES = [0.5, 1.0, 2.0, 4.0, 8.0]      # multiplies GKr, GKs -> APD long..short


def make_model(scale, dev):
    from cardiac_sim.ionic.ttp06.model import TTP06Model
    from cardiac_sim.ionic.base import CellType
    m = TTP06Model(cell_type=CellType.EPI, device=dev)
    m.params.GKr = m.params.GKr * scale
    m.params.GKs = m.params.GKs * scale
    return m


def measure_apd(scale, dev):
    """0D AP (uniformly-stimulated small patch) -> APD90 (ms)."""
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol
    from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
    from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation
    mask = torch.ones((3, 3), dtype=torch.bool, device=dev)
    grid = StructuredGrid.from_mask(mask, DX, DX, device=str(dev))
    # D=0 -> pure ionic 0D AP (no diffusion, no CFL limit); APD is a 0D/ionic property
    fdm = FDMDiscretization(grid, D=0.0, chi=1.0, Cm=1.0, stencil="moore8_iso", boundary_mode="face_mirror_iso")
    proto = StimulusProtocol()
    proto.add_stimulus(region=lambda x, y: x < 1e9, start_time=0.0, duration=2.0, amplitude=-52.0)  # all cells (0D)
    sim = MonodomainSimulation(spatial=fdm, ionic_model=make_model(scale, dev), stimulus=proto, dt=0.1,
                               splitting="strang", ionic_solver="rush_larsen",
                               diffusion_solver="forward_euler", cell_type="EPI")
    times, Vflat = sim.run_to_array(t_end=450.0, save_every=1.0)
    tr = np.asarray(Vflat)[:, Vflat.shape[1] // 2]; t = np.asarray(times)
    Vr = float(np.nanmin(tr)); Vp = float(np.nanmax(tr)); V90 = Vp - 0.9 * (Vp - Vr)
    up = np.argmax(tr >= -40.0); pk = np.argmax(tr)
    below = np.where(tr[pk:] <= V90)[0]
    if below.size == 0:
        return np.nan
    return float(t[pk + below[0]] - t[up])


def run_hourglass(scale, dev):
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol
    from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
    from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation
    nx = int(round(LX / DX)) + 1; ny = int(round(LY / DX)) + 1
    cache = REPO / f"media/source_sink_mismatch_investigation/_sim_outputs/s0g_gk{int(scale*100)}.npz"
    fluid = hourglass_fluid(DX, nx, ny, dev)
    if cache.exists():
        z = np.load(cache); return z["times"], z["V"], nx, ny, fluid.cpu().numpy()
    grid = StructuredGrid.from_mask(fluid, DX, DX, device=str(dev))
    fdm = FDMDiscretization(grid, D=D, chi=1.0, Cm=1.0, stencil="moore8_iso", boundary_mode="face_mirror_iso")
    proto = StimulusProtocol()
    proto.add_stimulus(region=lambda x, y: x < 0.05, start_time=0.0, duration=2.0, amplitude=-52.0)
    sim = MonodomainSimulation(spatial=fdm, ionic_model=make_model(scale, dev), stimulus=proto, dt=DT,
                               splitting="strang", ionic_solver="rush_larsen",
                               diffusion_solver="forward_euler", cell_type="EPI")
    t0 = time.time(); times, Vflat = sim.run_to_array(t_end=T2D, save_every=SAVE)
    fn = fluid.cpu().numpy(); V = np.full((len(times), nx, ny), -85.23, dtype=np.float32)
    V[:, fn] = np.asarray(Vflat, dtype=np.float32)
    print(f"      hourglass {time.time()-t0:.0f}s")
    cache.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(cache, times=times, V=V)
    return np.asarray(times, float), V, nx, ny, fn


def conv_crescent_and_cv(times, V, nx, ny, fluid):
    lat = activation_time_interp(V, times, THR)
    xs = np.arange(nx) * DX; vals = []
    for i in range(nx):
        if not (X_START + 0.15 <= xs[i] <= X_NECK - 0.05):
            continue
        js = np.where(fluid[i])[0]
        if js.size < 7:
            continue
        jt, jb, jc = js[-1], js[0], js[len(js) // 2]
        if np.isfinite(lat[i, jt]) and np.isfinite(lat[i, jc]):
            vals.append(0.5 * (lat[i, jt] + lat[i, jb]) - lat[i, jc])
    cres = float(np.nanmean(vals)) if vals else np.nan
    # base CV (planar, before convergence), centerline
    jc = ny // 2; latc = lat[:, jc]
    i0 = int(0.10 / DX); i1 = int(0.18 / DX)
    cv = (i1 - i0) * DX / (latc[i1] - latc[i0]) * 1000.0 if np.isfinite(latc[i1] - latc[i0]) else np.nan
    return cres, cv


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for s in SCALES:
        apd = measure_apd(s, dev)
        times, V, nx, ny, fluid = run_hourglass(s, dev)
        cres, cv = conv_crescent_and_cv(times, V, nx, ny, fluid)
        lam = cv / 1000.0 * apd          # cm  (cv cm/s -> cm/ms *apd ms)
        rows.append((s, apd, cv, lam, cres))
        print(f"  GKr/GKs x{s:.1f}: APD90={apd:6.1f}ms  CV={cv:5.1f}cm/s  lambda={lam:5.2f}cm "
              f"(lambda/dx={lam/DX:5.0f})  conv crescent={cres*1000:+6.0f}us")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    lam = np.array([r[3] for r in rows]); cres = np.array([r[4] for r in rows]) * 1000
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(lam, cres, "o-")
    for (s, apd, cv, l, c) in rows:
        ax.annotate(f"x{s}", (l, c * 1000), fontsize=8)
    ax.set_xlabel("AP wavelength lambda = CV*APD (cm)")
    ax.set_ylabel("converging crescent (us)  (<0 = inverse crescent)")
    ax.set_title(f"Step 2: crescent vs wavelength at FIXED dx={DX*1e4:.0f}um\n(flat = governed by dx/r*; sloped = dx/lambda matters)")
    ax.grid(alpha=0.3)
    p = media_path("source_sink_mismatch_investigation", "images", "s0g-crescent-vs-wavelength")
    fig.savefig(p, dpi=130); plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    main()
