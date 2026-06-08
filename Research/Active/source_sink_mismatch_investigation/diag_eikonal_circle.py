"""
Eikonal curvature test: planar wave past a CIRCULAR obstacle, plain no-flux
(HBB / zero-flux) everywhere. No speedup BC. Run in LBM and Monodomain.

Expected eikonal / source-sink signature:
  - LEADING hemisphere: boundary-adjacent tissue activates EARLIER than bulk
    (inverse crescent) — fewer downstream sinks -> less electrotonic load -> faster.
  - TRAILING hemisphere: lee shadow, boundary lags (crescent) as the two wrapped
    fronts converge behind the obstacle.

Outputs (per engine): LAT map + isochrones, propagation video, front/back summary.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path("/home/norepinephrine/Documents/Heart-Conduction")
sys.path.insert(0, str(REPO / "LBM/Engine_V1"))
sys.path.insert(0, str(REPO / "Monodomain/Engine_V5.4"))
sys.path.insert(0, str(REPO))
from cardiac_core.media import media_path

LX, LY, DX = 6.0, 4.0, 0.025
NX = int(round(LX / DX)) + 1      # 241
NY = int(round(LY / DX)) + 1      # 161
DT = 0.02
T_END = 120.0
SAVE_EVERY = 0.5
CXC, CYC, RC = 3.0, 2.0, 1.0      # circle center (cm) and radius
THR = -40.0


def circle_fluid(device):
    ix = torch.arange(NX, device=device).view(NX, 1).double() * DX
    iy = torch.arange(NY, device=device).view(1, NY).double() * DX
    dist = torch.sqrt((ix - CXC) ** 2 + (iy - CYC) ** 2)
    return dist > RC          # True = fluid, False = circular obstacle


def lat_field(V, times, thr=THR):
    above = V >= thr
    ever = above.any(0)
    idx = np.argmax(above, axis=0)
    idxc = np.clip(idx, 1, len(times) - 1)
    v1 = np.take_along_axis(V, idxc[None], 0)[0]
    v0 = np.take_along_axis(V, (idxc - 1)[None], 0)[0]
    t1 = np.asarray(times)[idxc]; t0 = np.asarray(times)[idxc - 1]
    denom = np.where(v1 == v0, 1.0, v1 - v0)
    lat = t0 + (thr - v0) * (t1 - t0) / denom
    lat[idx == 0] = times[0]
    lat[~ever] = np.nan
    return lat


# ---------------- LBM ----------------
def run_lbm():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.collision.bgk import bgk_collide
    from src.streaming.d2q9 import stream_d2q9
    from src.boundary.masks import precompute_bounce_masks
    from src.boundary.neumann import apply_neumann_d2q9
    from src.state import recover_voltage
    from src.solver.rush_larsen import compute_source_term, ionic_step
    from src.diffusion import tau_from_D
    from src.lattice import D2Q9
    from ionic.ttp06.model import TTP06Model
    from ionic.base import CellType

    lat = D2Q9()
    domain = circle_fluid(dev)
    solid = ~domain
    obm = precompute_bounce_masks(domain, lat)
    bm = {}
    for a in range(1, 9):
        m = torch.zeros(NX, NY, dtype=torch.bool, device=dev)
        ex, ey = lat.e[a]
        if ex == 1:  m[-1, :] = True
        if ex == -1: m[0, :] = True
        if ey == 1:  m[:, -1] = True
        if ey == -1: m[:, 0] = True
        bm[a] = obm[a] | m

    ionic = TTP06Model(cell_type=CellType.EPI, device=dev)
    V_rest = float(ionic.V_rest)
    w = torch.tensor(lat.w, dtype=torch.float64, device=dev)
    omega = 1.0 / tau_from_D(0.001, DX, DT, cs2=lat.cs2)
    V = torch.full((NX, NY), V_rest, dtype=torch.float64, device=dev)
    V[:2, :] = 20.0
    V[solid] = V_rest
    f = w[:, None, None] * V[None, :, :]
    states = ionic.get_initial_state(n_cells=NX * NY)
    Iz = torch.zeros(NX * NY, dtype=torch.float64, device=dev)

    n = int(round(T_END / DT)); siv = int(round(SAVE_EVERY / DT))
    frames = [V.cpu().numpy().astype(np.float32)]; tt = [0.0]
    t0 = time.time()
    for k in range(1, n + 1):
        I_ion = ionic.compute_Iion(V.reshape(-1), states)
        R = compute_source_term(I_ion, Iz, 1.0).reshape(NX, NY)
        f = bgk_collide(f, V, R, DT, omega, w); fs = f.clone()
        f = stream_d2q9(f); f = apply_neumann_d2q9(f, fs, bm)
        f[:, solid] = w[:, None] * V_rest
        V = recover_voltage(f)
        states = ionic_step(ionic, V.reshape(-1), states, DT)
        if k % siv == 0:
            frames.append(V.cpu().numpy().astype(np.float32)); tt.append(k * DT)
    print(f"  [LBM] {n} steps {time.time()-t0:.0f}s  Vmax={max(fr.max() for fr in frames):.1f}")
    return np.array(tt), np.array(frames), solid.cpu().numpy()


# ---------------- Monodomain ----------------
def run_mono():
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol
    from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
    from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation

    mask = circle_fluid(torch.device("cpu"))
    grid = StructuredGrid.from_mask(mask, DX, DX, device='cpu')
    fdm = FDMDiscretization(grid, D=0.001, chi=1.0, Cm=1.0,
                            stencil='cardinal4', boundary_mode='face_mirror')
    proto = StimulusProtocol()
    proto.add_stimulus(region=lambda x, y: x < 0.05, start_time=0.0,
                       duration=2.0, amplitude=-52.0)
    sim = MonodomainSimulation(spatial=fdm, ionic_model='ttp06', stimulus=proto,
                               dt=DT, splitting='strang', ionic_solver='rush_larsen',
                               diffusion_solver='forward_euler', cell_type='EPI')
    t0 = time.time()
    times, Vflat = sim.run_to_array(t_end=T_END, save_every=SAVE_EVERY)
    mask_np = mask.cpu().numpy()
    V_rest = -85.23
    V = np.full((len(times), NX, NY), V_rest, dtype=np.float32)
    V[:, mask_np] = np.asarray(Vflat, dtype=np.float32)
    print(f"  [mono] {time.time()-t0:.0f}s  Vmax={V.max():.1f}")
    return np.asarray(times), V, (~mask_np)


# ---------------- analysis + figures ----------------
def front_back_summary(lat, solid, name):
    # boundary-adjacent fluid cells (4-neighbour of obstacle)
    s = solid
    nbr = np.zeros_like(s)
    nbr[1:, :] |= s[:-1, :]; nbr[:-1, :] |= s[1:, :]
    nbr[:, 1:] |= s[:, :-1]; nbr[:, :-1] |= s[:, 1:]
    bdry = nbr & ~s & np.isfinite(lat)
    ii, jj = np.where(bdry)
    xb = ii * DX
    # bulk reference LAT at same x (far row near top edge, away from obstacle)
    ref_row = lat[:, 5]                       # y ~ 0.125 cm, far from circle (center y=2)
    lead = np.array([lat[i, j] - ref_row[i] for i, j in zip(ii, jj)])  # + = boundary later (crescent)
    front = xb < CXC; back = xb > CXC
    print(f"[{name}] boundary-adjacent LAT minus bulk-at-same-x (+=lag/crescent, -=lead/inverse):")
    print(f"   LEADING half  (x<{CXC}): mean {np.nanmean(lead[front]):+.2f} ms  "
          f"(expect NEGATIVE = inverse crescent)")
    print(f"   TRAILING half (x>{CXC}): mean {np.nanmean(lead[back]):+.2f} ms  "
          f"(expect POSITIVE = crescent/shadow)")


def figures(tag, times, V, solid):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import imageio_ffmpeg
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    lat = lat_field(V, times)
    front_back_summary(lat, solid, tag)
    ext = [0, LX, 0, LY]
    obs = solid
    # LAT + isochrones
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    Lm = np.ma.array(lat.T, mask=obs.T)
    im = ax.imshow(Lm, origin="lower", extent=ext, cmap="viridis", aspect="equal")
    xs = np.linspace(0, LX, NX); ys = np.linspace(0, LY, NY)
    ax.contour(xs, ys, Lm, levels=np.arange(0, np.nanmax(lat), 4.0),
               colors="white", linewidths=0.6, alpha=0.8)
    th = np.linspace(0, 2*np.pi, 100)
    ax.plot(CXC + RC*np.cos(th), CYC + RC*np.sin(th), 'r-', lw=1.5)
    fig.colorbar(im, ax=ax, shrink=0.8, label="LAT (ms)")
    ax.set_title(f"Eikonal curvature — {tag}: LAT + isochrones (no-flux circle)\n"
                 f"front=inverse crescent (leads), back=crescent (shadow)")
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
    p = media_path("source_sink_mismatch_investigation", "images", f"eikonal-circle-{tag}-lat")
    fig.savefig(p, dpi=120); plt.close(fig); print("  wrote", p)
    # video
    cmap = plt.cm.inferno.copy(); cmap.set_bad("0.55")
    def disp(a): return np.ma.array(a.T, mask=obs.T)
    figv, axv = plt.subplots(figsize=(9, 6), constrained_layout=True)
    imv = axv.imshow(disp(V[0]), origin="lower", extent=ext, cmap=cmap,
                     vmin=float(V.min()), vmax=40, aspect="equal", interpolation="bilinear")
    axv.plot(CXC + RC*np.cos(th), CYC + RC*np.sin(th), 'c-', lw=1.0)
    axv.set_xlabel("x (cm)"); axv.set_ylabel("y (cm)")
    figv.colorbar(imv, ax=axv, shrink=0.8, label="V (mV)")
    ttl = axv.set_title(f"Eikonal circle — {tag}  t=0 ms")
    step = max(1, len(V)//200)
    def upd(fr):
        imv.set_array(disp(V[fr])); ttl.set_text(f"Eikonal circle — {tag}  t={times[fr]:.0f} ms")
        return imv, ttl
    anim = FuncAnimation(figv, upd, frames=range(0, len(V), step), blit=False)
    pv = media_path("source_sink_mismatch_investigation", "videos", f"eikonal-circle-{tag}", ext="mp4")
    anim.save(pv, writer=FFMpegWriter(fps=18, bitrate=3500), dpi=120)
    plt.close(figv); print("  wrote", pv)


def main():
    print(f"Eikonal circle test: {NX}x{NY} ({LX}x{LY}cm) dx={DX}, circle R={RC} at ({CXC},{CYC})")
    print("== LBM ==");  tL, VL, sL = run_lbm();  figures("lbm", tL, VL, sL)
    print("== MONO =="); tM, VM, sM = run_mono(); figures("mono", tM, VM, sM)


if __name__ == "__main__":
    main()
