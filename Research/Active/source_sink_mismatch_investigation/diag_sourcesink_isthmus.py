"""
Source-sink mismatch test: an ISTHMUS (wide -> thin neck -> wide) channel, plain
no-flux everywhere, planar wave from the left. Run in LBM and Monodomain.

If gradient-driven RD captures source-sink mismatch (it should), we expect:
  - CONTRACTION (wide->neck): less lateral sink -> wave focuses / speeds up.
  - EXPANSION  (neck->wide): thin source charging a large sink -> CV drops
    sharply, and below a critical neck width the wave BLOCKS.

Outputs per engine: CV-vs-x along centerline, LAT map, propagation video.
"""
from __future__ import annotations
import sys, os, time
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
T_END = 130.0
SAVE_EVERY = 0.5
THR = -40.0
YC = LY / 2
W_WIDE = 1.5                      # half-width of wide channel (cm) -> 3 cm tall
W_NECK = float(os.environ.get("W_NECK", "0.20"))   # half-width of neck (cm)
NECK_X0, NECK_X1 = 2.7, 3.3       # neck extent in x (cm)
ENGINE = os.environ.get("ENGINE", "both")          # lbm | mono | both
NOVIZ = os.environ.get("NOVIZ", "0") == "1"        # skip figs/videos for fast sweep
NCELLS = int(round(2 * W_NECK / DX))               # neck height in cells


GEOM = os.environ.get("GEOM", "isthmus")   # "isthmus" | "strand_expansion"
EXPAND_X = 3.0                              # x of the abrupt expansion (strand_expansion)


def isthmus_fluid(device):
    ix = torch.arange(NX, device=device).view(NX, 1).double() * DX
    iy = torch.arange(NY, device=device).view(1, NY).double() * DX
    if GEOM == "strand_expansion":
        # thin strand (the ONLY source, stim inside it) -> abrupt wide region.
        hw = torch.where(ix < EXPAND_X,
                         torch.tensor(W_NECK, device=device, dtype=torch.float64),
                         torch.tensor(W_WIDE, device=device, dtype=torch.float64))
    else:
        hw = torch.where((ix >= NECK_X0) & (ix <= NECK_X1),
                         torch.tensor(W_NECK, device=device, dtype=torch.float64),
                         torch.tensor(W_WIDE, device=device, dtype=torch.float64))
    return (torch.abs(iy - YC) <= hw)      # True = fluid (channel)


def lat_field(V, times, thr=THR):
    above = V >= thr
    ever = above.any(0)
    idx = np.argmax(above, axis=0)
    idxc = np.clip(idx, 1, len(times) - 1)
    v1 = np.take_along_axis(V, idxc[None], 0)[0]
    v0 = np.take_along_axis(V, (idxc - 1)[None], 0)[0]
    t = np.asarray(times); t1 = t[idxc]; t0 = t[idxc - 1]
    denom = np.where(v1 == v0, 1.0, v1 - v0)
    lat = t0 + (thr - v0) * (t1 - t0) / denom
    lat[idx == 0] = times[0]
    lat[~ever] = np.nan
    return lat


def run_lbm(domain):
    dev = domain.device
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
    lat = D2Q9(); solid = ~domain
    obm = precompute_bounce_masks(domain, lat); bm = {}
    for a in range(1, 9):
        m = torch.zeros(NX, NY, dtype=torch.bool, device=dev)
        ex, ey = lat.e[a]
        if ex == 1: m[-1, :] = True
        if ex == -1: m[0, :] = True
        if ey == 1: m[:, -1] = True
        if ey == -1: m[:, 0] = True
        bm[a] = obm[a] | m
    ionic = TTP06Model(cell_type=CellType.EPI, device=dev); V_rest = float(ionic.V_rest)
    w = torch.tensor(lat.w, dtype=torch.float64, device=dev)
    omega = 1.0 / tau_from_D(0.001, DX, DT, cs2=lat.cs2)
    V = torch.full((NX, NY), V_rest, dtype=torch.float64, device=dev)
    V[:2, :] = 20.0; V[solid] = V_rest
    f = w[:, None, None] * V[None, :, :]
    states = ionic.get_initial_state(n_cells=NX * NY)
    Iz = torch.zeros(NX * NY, dtype=torch.float64, device=dev)
    n = int(round(T_END / DT)); siv = int(round(SAVE_EVERY / DT))
    fr = [V.cpu().numpy().astype(np.float32)]; tt = [0.0]; t0 = time.time()
    for k in range(1, n + 1):
        I_ion = ionic.compute_Iion(V.reshape(-1), states)
        R = compute_source_term(I_ion, Iz, 1.0).reshape(NX, NY)
        f = bgk_collide(f, V, R, DT, omega, w); fs = f.clone()
        f = stream_d2q9(f); f = apply_neumann_d2q9(f, fs, bm)
        f[:, solid] = w[:, None] * V_rest; V = recover_voltage(f)
        states = ionic_step(ionic, V.reshape(-1), states, DT)
        if k % siv == 0:
            fr.append(V.cpu().numpy().astype(np.float32)); tt.append(k * DT)
    print(f"  [LBM] {time.time()-t0:.0f}s Vmax={max(x.max() for x in fr):.1f}")
    return np.array(tt), np.array(fr), solid.cpu().numpy()


def run_mono(domain):
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol
    from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
    from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation
    mask = domain.cpu()
    grid = StructuredGrid.from_mask(mask, DX, DX, device='cpu')
    fdm = FDMDiscretization(grid, D=0.001, chi=1.0, Cm=1.0,
                            stencil='cardinal4', boundary_mode='face_mirror')
    proto = StimulusProtocol()
    proto.add_stimulus(region=lambda x, y: x < 0.05, start_time=0.0,
                       duration=2.0, amplitude=-52.0)
    sim = MonodomainSimulation(spatial=fdm, ionic_model='ttp06', stimulus=proto,
                               dt=DT, splitting='strang', ionic_solver='rush_larsen',
                               diffusion_solver='forward_euler', cell_type='EPI')
    t0 = time.time(); times, Vflat = sim.run_to_array(t_end=T_END, save_every=SAVE_EVERY)
    mnp = mask.numpy(); V = np.full((len(times), NX, NY), -85.23, dtype=np.float32)
    V[:, mnp] = np.asarray(Vflat, dtype=np.float32)
    print(f"  [mono] {time.time()-t0:.0f}s Vmax={V.max():.1f}")
    return np.asarray(times), V, (~mnp)


def analyze(tag, times, V, solid):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import imageio_ffmpeg
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    lat = lat_field(V, times)
    jc = int(round(YC / DX))
    x = np.arange(NX) * DX
    latc = lat[:, jc]
    # local CV = dx/dLAT (centered), cm/ms -> cm/s
    cv = np.full(NX, np.nan)
    for i in range(2, NX - 2):
        d = latc[i + 2] - latc[i - 2]
        if np.isfinite(d) and d > 0:
            cv[i] = (4 * DX) / d * 1000.0
    reached = np.isfinite(latc)
    xmax = x[reached].max() if reached.any() else 0.0
    blocked = xmax < LX - 0.3
    print(f"[{tag}] wave reached x={xmax:.2f}cm  {'(BLOCKED at expansion)' if blocked else '(propagated through)'}")
    for xc in (1.5, 2.6, 3.0, 3.5, 4.5):
        i = int(xc / DX)
        print(f"   x={xc:.1f}cm  CV={cv[i]:.1f} cm/s" if np.isfinite(cv[i]) else f"   x={xc:.1f}cm  CV=--- (not reached)")
    if NOVIZ:
        return

    ext = [0, LX, 0, LY]; obs = solid
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True,
                                 gridspec_kw={'height_ratios': [2, 1]})
    Lm = np.ma.array(lat.T, mask=obs.T)
    im = a1.imshow(Lm, origin="lower", extent=ext, cmap="viridis", aspect="equal")
    xs = np.linspace(0, LX, NX); ys = np.linspace(0, LY, NY)
    a1.contour(xs, ys, Lm, levels=np.arange(0, np.nanmax(lat) if np.isfinite(np.nanmax(lat)) else 1, 3.0),
               colors="white", linewidths=0.5, alpha=0.7)
    a1.axvspan(NECK_X0, NECK_X1, color='red', alpha=0.12)
    fig.colorbar(im, ax=a1, shrink=0.8, label="LAT (ms)")
    a1.set_title(f"Source-sink isthmus — {tag}: LAT + isochrones"); a1.set_xlabel("x (cm)"); a1.set_ylabel("y (cm)")
    a2.plot(x, cv, lw=1.5)
    a2.axvspan(NECK_X0, NECK_X1, color='red', alpha=0.15, label='neck')
    a2.set_xlabel("x (cm)"); a2.set_ylabel("CV (cm/s)"); a2.set_title("centerline CV: up into contraction, down at expansion")
    a2.grid(alpha=0.3); a2.legend(); a2.set_xlim(0, LX)
    p = media_path("source_sink_mismatch_investigation", "images", f"isthmus-{tag}-n{NCELLS}")
    fig.savefig(p, dpi=120); plt.close(fig); print("  wrote", p)

    cmap = plt.cm.inferno.copy(); cmap.set_bad("0.55")
    def disp(arr): return np.ma.array(arr.T, mask=obs.T)
    figv, axv = plt.subplots(figsize=(9, 6), constrained_layout=True)
    imv = axv.imshow(disp(V[0]), origin="lower", extent=ext, cmap=cmap,
                     vmin=float(V.min()), vmax=40, aspect="equal", interpolation="bilinear")
    figv.colorbar(imv, ax=axv, shrink=0.8, label="V (mV)")
    ttl = axv.set_title(f"isthmus — {tag}  t=0 ms"); axv.set_xlabel("x (cm)"); axv.set_ylabel("y (cm)")
    step = max(1, len(V) // 200)
    def upd(k):
        imv.set_array(disp(V[k])); ttl.set_text(f"isthmus — {tag}  t={times[k]:.0f} ms"); return imv, ttl
    anim = FuncAnimation(figv, upd, frames=range(0, len(V), step), blit=False)
    pv = media_path("source_sink_mismatch_investigation", "videos", f"isthmus-{tag}-n{NCELLS}", ext="mp4")
    anim.save(pv, writer=FFMpegWriter(fps=18, bitrate=3500), dpi=120); plt.close(figv); print("  wrote", pv)


def main():
    print(f"Isthmus: wide half-w={W_WIDE}cm, neck half-w={W_NECK}cm "
          f"({NCELLS} cells tall) at x∈[{NECK_X0},{NECK_X1}]  | ENGINE={ENGINE}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ENGINE in ("lbm", "both"):
        print("== LBM ==");  tL, VL, sL = run_lbm(isthmus_fluid(dev)); analyze("lbm", tL, VL, sL)
    if ENGINE in ("mono", "both"):
        print("== MONO =="); tM, VM, sM = run_mono(isthmus_fluid(torch.device("cpu"))); analyze("mono", tM, VM, sM)


if __name__ == "__main__":
    main()
