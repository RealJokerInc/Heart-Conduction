"""
Hourglass channel: triangular CONSTRICTION (wide->neck) followed by the inverse
triangular DILATION (neck->wide), symmetric about the neck. Plain no-flux.
Planar wave enters wide base -> through the neck -> out the dilation.

Expectation: planar/high-SF through the constriction (no visible effect), then
the DILATION is the source-sink expansion -> radial collapse / lateral-fill
delay as the thin neck charges the widening sink.

Run in LBM and Monodomain. Verifies geometry symmetry (top-bottom and
left-right about the neck) before running. Two videos.
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
NX = int(round(LX / DX)) + 1
NY = int(round(LY / DX)) + 1
DT = 0.02
T_END = 140.0
SAVE_EVERY = 0.5
D = 0.001
YC = LY / 2
W_BASE = 1.5
W_NECK = 0.05
X_START, X_NECK, X_END = 0.5, 3.0, 5.5     # symmetric: 2.5 cm each side
VTHR = -40.0


def hw_of_x(xc):
    tL = (xc - X_START) / (X_NECK - X_START)
    tR = (X_END - xc) / (X_END - X_NECK)
    t = np.where(xc < X_NECK, tL, tR)
    t = np.clip(t, 0.0, 1.0)
    return W_BASE + (W_NECK - W_BASE) * t


def hourglass_fluid(device):
    # Build from INTEGER cell half-widths centered on the middle row -> exactly
    # symmetric top-bottom (and left-right, since hw_of_x is symmetric about X_NECK).
    jc = NY // 2
    x = np.arange(NX) * DX
    hw_cells = np.round(hw_of_x(x) / DX).astype(int)
    mask = np.zeros((NX, NY), dtype=bool)
    for i in range(NX):
        lo = max(0, jc - hw_cells[i]); hi = min(NY, jc + hw_cells[i] + 1)
        mask[i, lo:hi] = True
    return torch.tensor(mask, device=device)


def check_symmetry(domain):
    m = domain.cpu().numpy()
    # top-bottom (about y = YC, the horizontal x-axis through center)
    tb = bool(np.array_equal(m, m[:, ::-1]))
    # left-right about the neck column
    ink = int(round(X_NECK / DX))
    half = min(ink, NX - 1 - ink)
    lr = bool(np.array_equal(m[ink - half:ink], m[ink + 1:ink + 1 + half][::-1]))
    print(f"[symmetry] top-bottom about y={YC}: {'PASS' if tb else 'FAIL'} ; "
          f"left-right about neck x={X_NECK}: {'PASS' if lr else 'FAIL'}")
    return tb and lr


def lat_field(V, times, thr=VTHR):
    above = V >= thr; ever = above.any(0); idx = np.argmax(above, axis=0)
    idxc = np.clip(idx, 1, len(times) - 1)
    v1 = np.take_along_axis(V, idxc[None], 0)[0]; v0 = np.take_along_axis(V, (idxc - 1)[None], 0)[0]
    t = np.asarray(times); denom = np.where(v1 == v0, 1.0, v1 - v0)
    lat = t[idxc - 1] + (thr - v0) * (t[idxc] - t[idxc - 1]) / denom
    lat[idx == 0] = times[0]; lat[~ever] = np.nan
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
        m = torch.zeros(NX, NY, dtype=torch.bool, device=dev); ex, ey = lat.e[a]
        if ex == 1: m[-1, :] = True
        if ex == -1: m[0, :] = True
        if ey == 1: m[:, -1] = True
        if ey == -1: m[:, 0] = True
        bm[a] = obm[a] | m
    ionic = TTP06Model(cell_type=CellType.EPI, device=dev); V_rest = float(ionic.V_rest)
    w = torch.tensor(lat.w, dtype=torch.float64, device=dev)
    omega = 1.0 / tau_from_D(D, DX, DT, cs2=lat.cs2)
    V = torch.full((NX, NY), V_rest, dtype=torch.float64, device=dev); V[:2, :] = 20.0; V[solid] = V_rest
    f = w[:, None, None] * V[None, :, :]; states = ionic.get_initial_state(n_cells=NX * NY)
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
        if k % siv == 0: fr.append(V.cpu().numpy().astype(np.float32)); tt.append(k * DT)
    print(f"  [LBM] {time.time()-t0:.0f}s Vmax={max(x.max() for x in fr):.1f}")
    return np.array(tt), np.array(fr), solid.cpu().numpy()


def run_mono(domain):
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol
    from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
    from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation
    mask = domain.cpu()
    grid = StructuredGrid.from_mask(mask, DX, DX, device='cpu')
    fdm = FDMDiscretization(grid, D=D, chi=1.0, Cm=1.0, stencil='cardinal4', boundary_mode='face_mirror')
    proto = StimulusProtocol()
    proto.add_stimulus(region=lambda x, y: x < 0.05, start_time=0.0, duration=2.0, amplitude=-52.0)
    sim = MonodomainSimulation(spatial=fdm, ionic_model='ttp06', stimulus=proto, dt=DT,
                               splitting='strang', ionic_solver='rush_larsen',
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
    lat = lat_field(V, times); jc = int(round(YC / DX)); x = np.arange(NX) * DX
    latc = lat[:, jc]; cv = np.full(NX, np.nan)
    for i in range(3, NX - 3):
        d = latc[i + 3] - latc[i - 3]
        if np.isfinite(d) and d > 0: cv[i] = (6 * DX) / d * 1000.0
    print(f"[{tag}] centerline CV (cm/s) along the hourglass:")
    for xc in (1.0, 2.5, 3.0, 3.5, 5.0):
        i = int(xc / DX); s = f"{cv[i]:.1f}" if np.isfinite(cv[i]) else "--"
        print(f"   x={xc:.1f}cm  half-w={hw_of_x(xc)*10:.1f}mm  CV={s} cm/s")

    ext = [0, LX, 0, LY]; obs = solid
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True,
                                 gridspec_kw={'height_ratios': [2, 1]})
    Lm = np.ma.array(lat.T, mask=obs.T)
    im = a1.imshow(Lm, origin="lower", extent=ext, cmap="viridis", aspect="equal")
    xs = np.linspace(0, LX, NX); ys = np.linspace(0, LY, NY)
    mx = np.nanmax(lat) if np.isfinite(np.nanmax(lat)) else 1
    a1.contour(xs, ys, Lm, levels=np.arange(0, mx, 2.0), colors="white", linewidths=0.5, alpha=0.7)
    a1.axvline(X_NECK, color='red', ls=':', lw=1)
    fig.colorbar(im, ax=a1, shrink=0.8, label="LAT (ms)")
    a1.set_title(f"Hourglass (constriction+dilation) — {tag}: LAT + isochrones")
    a1.set_xlabel("x (cm)"); a1.set_ylabel("y (cm)")
    a2.plot(x, cv, lw=1.5); a2.axvline(X_NECK, color='red', ls=':', lw=1, label='neck')
    a2.set_xlabel("x (cm)"); a2.set_ylabel("centerline CV (cm/s)")
    a2.set_title("constriction (left of neck) then dilation (right)"); a2.grid(alpha=0.3); a2.legend(); a2.set_xlim(0, LX)
    p = media_path("source_sink_mismatch_investigation", "images", f"hourglass-{tag}")
    fig.savefig(p, dpi=120); plt.close(fig); print("  wrote", p)

    cmap = plt.cm.inferno.copy(); cmap.set_bad("0.55")
    def disp(a): return np.ma.array(a.T, mask=obs.T)
    figv, axv = plt.subplots(figsize=(9, 6), constrained_layout=True)
    imv = axv.imshow(disp(V[0]), origin="lower", extent=ext, cmap=cmap,
                     vmin=float(V.min()), vmax=40, aspect="equal", interpolation="bilinear")
    figv.colorbar(imv, ax=axv, shrink=0.8, label="V (mV)"); axv.set_xlabel("x (cm)"); axv.set_ylabel("y (cm)")
    ttl = axv.set_title(f"hourglass — {tag}  t=0 ms"); step = max(1, len(V)//220)
    def upd(k):
        imv.set_array(disp(V[k])); ttl.set_text(f"hourglass — {tag}  t={times[k]:.0f} ms"); return imv, ttl
    anim = FuncAnimation(figv, upd, frames=range(0, len(V), step), blit=False)
    pv = media_path("source_sink_mismatch_investigation", "videos", f"hourglass-{tag}", ext="mp4")
    anim.save(pv, writer=FFMpegWriter(fps=18, bitrate=3500), dpi=120); plt.close(figv); print("  wrote", pv)


def main():
    print(f"Hourglass: half-w {W_BASE}cm -> neck {W_NECK}cm @x={X_NECK} -> {W_BASE}cm; "
          f"taper {X_START}->{X_NECK}->{X_END}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dom = hourglass_fluid(dev)
    if not check_symmetry(dom):
        print("WARNING: geometry not symmetric — aborting"); return
    print("== LBM ==");  tL, VL, sL = run_lbm(dom); analyze("lbm", tL, VL, sL)
    print("== MONO =="); tM, VM, sM = run_mono(hourglass_fluid(torch.device("cpu"))); analyze("mono", tM, VM, sM)


if __name__ == "__main__":
    main()
