"""S0i — the CV channel. r* = D/CV, so the crescent should depend on CV (not APD).
Hold dx and D fixed; scale GNa to move CV (hence r*); measure the converging crescent.
Prediction: lower CV -> larger r* -> larger r*/dx -> BIGGER crescent (more negative).
This is why Step 2 (lambda via APD, CV fixed) saw nothing: APD is absent from r*=D/CV.
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

DX = 0.0083; DT = 0.015; T2D = 50.0; SAVE = 0.25; D = 0.001
GNA_SCALES = [0.5, 0.75, 1.0, 1.5, 2.25]


def make_model(scale, dev):
    from cardiac_sim.ionic.ttp06.model import TTP06Model
    from cardiac_sim.ionic.base import CellType
    m = TTP06Model(cell_type=CellType.EPI, device=dev)
    m.params.GNa = m.params.GNa * scale
    return m


def run(scale, dev):
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol
    from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
    from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation
    nx = int(round(LX / DX)) + 1; ny = int(round(LY / DX)) + 1
    cache = REPO / f"media/source_sink_mismatch_investigation/_sim_outputs/s0i_gna{int(scale*100)}.npz"
    fl = hourglass_fluid(DX, nx, ny, dev)
    if cache.exists():
        z = np.load(cache); return z["times"], z["V"], nx, ny, fl.cpu().numpy()
    grid = StructuredGrid.from_mask(fl, DX, DX, device=str(dev))
    fdm = FDMDiscretization(grid, D=D, chi=1.0, Cm=1.0, stencil="moore8_iso", boundary_mode="face_mirror_iso")
    proto = StimulusProtocol()
    proto.add_stimulus(region=lambda x, y: x < 0.05, start_time=0.0, duration=2.0, amplitude=-52.0)
    sim = MonodomainSimulation(spatial=fdm, ionic_model=make_model(scale, dev), stimulus=proto, dt=DT,
                               splitting="strang", ionic_solver="rush_larsen",
                               diffusion_solver="forward_euler", cell_type="EPI")
    t0 = time.time(); times, Vflat = sim.run_to_array(t_end=T2D, save_every=SAVE)
    fn = fl.cpu().numpy(); V = np.full((len(times), nx, ny), -85.23, dtype=np.float32)
    V[:, fn] = np.asarray(Vflat, dtype=np.float32)
    print(f"    GNa x{scale} {time.time()-t0:.0f}s")
    cache.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(cache, times=times, V=V)
    return np.asarray(times, float), V, nx, ny, fn


def measure(times, V, nx, ny, fl):
    lat = activation_time_interp(V, times, THR)
    jc = ny // 2; latc = lat[:, jc]
    i0 = int(0.06 / DX); i1 = int(0.16 / DX)
    cv = (i1 - i0) * DX / (latc[i1] - latc[i0]) * 1000.0     # base planar CV cm/s
    xs = np.arange(nx) * DX; vals = []
    for i in range(nx):
        if not (X_START + 0.15 <= xs[i] <= X_NECK - 0.05):
            continue
        js = np.where(fl[i])[0]
        if js.size < 7:
            continue
        jt, jb, jcn = js[-1], js[0], js[len(js) // 2]
        if np.isfinite(lat[i, jt]) and np.isfinite(lat[i, jcn]):
            vals.append(0.5 * (lat[i, jt] + lat[i, jb]) - lat[i, jcn])
    return cv, (float(np.nanmean(vals)) if vals else np.nan)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for s in GNA_SCALES:
        times, V, nx, ny, fl = run(s, dev)
        cv, cres = measure(times, V, nx, ny, fl)
        rstar = D / (cv / 1000.0)          # cm
        rows.append((s, cv, rstar, cres))
        print(f"  GNa x{s:.2f}: CV={cv:5.1f}cm/s  r*=D/CV={rstar*1e4:4.0f}um  r*/dx={rstar/DX:.2f}  "
              f"crescent={cres*1000:+.0f}us")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    cv = [r[1] for r in rows]; cr = [r[3]*1000 for r in rows]; rd = [r[2]/DX for r in rows]
    a1.plot(cv, cr, "o-"); a1.set_xlabel("CV (cm/s)"); a1.set_ylabel("converging crescent (us)")
    a1.set_title("crescent vs CV (via GNa) at fixed dx,D"); a1.grid(alpha=0.3)
    a2.plot(rd, cr, "s-", color="C1"); a2.set_xlabel("r*/dx = D/(CV*dx)"); a2.set_ylabel("crescent (us)")
    a2.set_title("crescent collapses on r*/dx (CV is the operative knob)"); a2.grid(alpha=0.3)
    p = media_path("source_sink_mismatch_investigation", "images", "s0i-cv-channel")
    fig.savefig(p, dpi=130); plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    main()
