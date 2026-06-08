"""
DIFFUSION-ONLY on the hourglass mask (no TTP06, no reaction). Sustained source
clamped at the left edge; pure heat equation dV/dt = D grad^2 V in both engines.

Point: with no regenerative ionics there is NO accumulation/propagation. Charge
spreads diffusively as ~sqrt(t) -- glacially -- and just dissipates through the
geometry. Contrast with the active wave (76 cm/s, crossed 6 cm in ~80 ms): here
diffusion creeps ~0.5 cm in the same 140 ms. Two videos (LBM, monodomain).
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
T_END = 600.0
SAVE_EVERY = 4.0
D = 0.001
YC = LY / 2
W_BASE, W_NECK = 1.5, 0.05
X_START, X_NECK, X_END = 0.5, 3.0, 5.5
V_REST = -85.23
V_CLAMP = 0.0           # sustained depolarized source at the left edge
CLAMP_COLS = 2


def hw_of_x(xc):
    tL = (xc - X_START) / (X_NECK - X_START); tR = (X_END - xc) / (X_END - X_NECK)
    t = np.clip(np.where(xc < X_NECK, tL, tR), 0.0, 1.0)
    return W_BASE + (W_NECK - W_BASE) * t


def hourglass_mask():
    jc = NY // 2
    hw_cells = np.round(hw_of_x(np.arange(NX) * DX) / DX).astype(int)
    m = np.zeros((NX, NY), dtype=bool)
    for i in range(NX):
        lo = max(0, jc - hw_cells[i]); hi = min(NY, jc + hw_cells[i] + 1)
        m[i, lo:hi] = True
    return m


def run_lbm_diffusion(mask_np):
    from src.collision.bgk import bgk_collide
    from src.streaming.d2q9 import stream_d2q9
    from src.boundary.masks import precompute_bounce_masks
    from src.boundary.neumann import apply_neumann_d2q9
    from src.state import recover_voltage
    from src.diffusion import tau_from_D
    from src.lattice import D2Q9
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lat = D2Q9()
    domain = torch.tensor(mask_np, device=dev); solid = ~domain
    obm = precompute_bounce_masks(domain, lat); bm = {}
    for a in range(1, 9):
        m = torch.zeros(NX, NY, dtype=torch.bool, device=dev); ex, ey = lat.e[a]
        if ex == 1: m[-1, :] = True
        if ex == -1: m[0, :] = True
        if ey == 1: m[:, -1] = True
        if ey == -1: m[:, 0] = True
        bm[a] = obm[a] | m
    w = torch.tensor(lat.w, dtype=torch.float64, device=dev)
    omega = 1.0 / tau_from_D(D, DX, DT, cs2=lat.cs2)
    left = torch.zeros(NX, NY, dtype=torch.bool, device=dev); left[:CLAMP_COLS, :] = True
    left &= domain
    V = torch.full((NX, NY), V_REST, dtype=torch.float64, device=dev)
    V[left] = V_CLAMP
    f = w[:, None, None] * V[None, :, :]
    Rz = torch.zeros(NX, NY, dtype=torch.float64, device=dev)
    n = int(round(T_END / DT)); siv = int(round(SAVE_EVERY / DT))
    fr = [V.cpu().numpy().astype(np.float32)]; tt = [0.0]; t0 = time.time()
    for k in range(1, n + 1):
        f = bgk_collide(f, V, Rz, DT, omega, w); fs = f.clone()
        f = stream_d2q9(f); f = apply_neumann_d2q9(f, fs, bm)
        f[:, solid] = w[:, None] * V_REST
        f[:, left] = w[:, None] * V_CLAMP        # sustained Dirichlet source
        V = recover_voltage(f)
        if k % siv == 0: fr.append(V.cpu().numpy().astype(np.float32)); tt.append(k * DT)
    print(f"  [LBM diff] {time.time()-t0:.0f}s")
    return np.array(tt), np.array(fr), (~mask_np)


def run_mono_diffusion(mask_np):
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
    mask = torch.tensor(mask_np)
    grid = StructuredGrid.from_mask(mask, DX, DX, device='cpu')
    fdm = FDMDiscretization(grid, D=D, chi=1.0, Cm=1.0, stencil='cardinal4', boundary_mode='face_mirror')
    is_left = np.zeros((NX, NY), dtype=bool); is_left[:CLAMP_COLS, :] = True
    left_flat = torch.tensor(is_left[mask_np])
    Vg = np.full((NX, NY), V_REST); Vg[:CLAMP_COLS, :] = V_CLAMP
    V = torch.tensor(Vg[mask_np], dtype=torch.float64)
    n = int(round(T_END / DT)); siv = int(round(SAVE_EVERY / DT))
    def to_grid(vflat):
        g = np.full((NX, NY), V_REST, dtype=np.float32); g[mask_np] = vflat.numpy().astype(np.float32); return g
    fr = [to_grid(V)]; tt = [0.0]; t0 = time.time()
    for k in range(1, n + 1):
        V = V + DT * fdm.apply_diffusion(V)
        V[left_flat] = V_CLAMP
        if k % siv == 0: fr.append(to_grid(V)); tt.append(k * DT)
    print(f"  [mono diff] {time.time()-t0:.0f}s")
    return np.array(tt), np.array(fr), (~mask_np)


def reach(times, V, obs):
    x = np.arange(NX) * DX
    for tt in (140.0, 300.0, 600.0):
        k = int(np.argmin(np.abs(times - tt)))
        front = V[k]
        cols = x[(front > V_REST + 5).any(axis=1)]
        xr = cols.max() if cols.size else 0.0
        print(f"     t={tt:.0f}ms: charge (V>rest+5) reached x={xr:.2f} cm")


def render(tag, times, V, obs):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import imageio_ffmpeg
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"  [{tag}] diffusive reach:"); reach(times, V, obs)
    ext = [0, LX, 0, LY]; cmap = plt.cm.inferno.copy(); cmap.set_bad("0.55")
    def disp(a): return np.ma.array(a.T, mask=obs.T)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    im = ax.imshow(disp(V[0]), origin="lower", extent=ext, cmap=cmap, vmin=V_REST, vmax=40,
                   aspect="equal", interpolation="bilinear")
    fig.colorbar(im, ax=ax, shrink=0.8, label="V (mV)"); ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
    ttl = ax.set_title(f"DIFFUSION ONLY (no ionics) — {tag}  t=0 ms"); step = max(1, len(V)//220)
    def upd(k):
        im.set_array(disp(V[k])); ttl.set_text(f"DIFFUSION ONLY (no ionics) — {tag}  t={times[k]:.0f} ms"); return im, ttl
    anim = FuncAnimation(fig, upd, frames=range(0, len(V), step), blit=False)
    pv = media_path("source_sink_mismatch_investigation", "videos", f"hourglass-diffusion-{tag}", ext="mp4")
    anim.save(pv, writer=FFMpegWriter(fps=18, bitrate=3500), dpi=120); plt.close(fig); print("  wrote", pv)


def front_curvature(times, V, obs, level, x_target=1.8):
    """Iso-V front shape in the CONSTRICTION (x<X_NECK). Fit x_front(y) =
    c0 + c2*(y-YC)^2.  c2>0 => edges lead = INVERSE crescent; c2<0 => center leads."""
    x = np.arange(NX) * DX; jc = NY // 2
    fc = []
    for fr in V:
        row = fr[:, jc]; cr = np.where(row > level)[0]
        fc.append(x[cr.max()] if cr.size else 0.0)
    k = int(np.argmin(np.abs(np.array(fc) - x_target))); snap = V[k]
    ys, xf = [], []
    for j in range(NY):
        row = snap[:, j]; cr = np.where(row > level)[0]
        if cr.size == 0: continue
        i2 = cr.max()
        if i2 < 1 or i2 >= NX - 1 or obs[i2, j]: continue
        v2, v3 = row[i2], row[i2 + 1]
        xfr = x[i2] + (level - v2) / (v3 - v2) * DX if v3 != v2 else x[i2]
        if xfr >= X_NECK or xfr <= 0.1: continue
        ys.append(j * DX); xf.append(xfr)
    ys = np.array(ys); xf = np.array(xf)
    if ys.size < 8:
        return None
    c2, c1, c0 = np.polyfit(ys - YC, xf, 2)
    dy = 0.6
    edge_lead = c2 * dy ** 2 * 10.0  # mm, edges relative to center at |y-YC|=dy
    return dict(t=times[k], ys=ys, xf=xf, c2=c2, edge_lead=edge_lead,
                xc=c0, kind=("INVERSE crescent (edges lead)" if c2 > 0 else "crescent (center leads)"))


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    m = hourglass_mask()
    print(f"Hourglass DIFFUSION-ONLY, clamp left {CLAMP_COLS} cols @ {V_CLAMP} mV, "
          f"D={D}, T_END={T_END} ms")
    print("== LBM (diffusion) =="); tL, VL, oL = run_lbm_diffusion(m); reach(tL, VL, oL)
    print("== MONO (diffusion) =="); tM, VM, oM = run_mono_diffusion(m); reach(tM, VM, oM)

    print("\n=== front curvature in constriction (pure diffusion) ===")
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for tag, (t, V, o), col in [("lbm", (tL, VL, oL), "tab:blue"),
                                ("mono", (tM, VM, oM), "tab:orange")]:
        for lvl, ls in [(V_REST + 5, "-"), (V_REST + 30, "--")]:
            r = front_curvature(t, V, o, lvl)
            if r is None:
                print(f"  [{tag}] level={lvl-V_REST:+.0f}: too few points"); continue
            print(f"  [{tag}] level=rest{lvl-V_REST:+.0f}mV  t={r['t']:.0f}ms  "
                  f"c2={r['c2']:+.4f} cm/cm^2  edge-lead@0.6cm={r['edge_lead']:+.3f} mm  -> {r['kind']}")
            ax.plot(r['ys'] - YC, (r['xf'] - r['xc']) * 10, ls, color=col, lw=1.3,
                    label=f"{tag} rest{lvl-V_REST:+.0f}mV (c2={r['c2']:+.3f})")
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel("y - center (cm)"); ax.set_ylabel("front x - center-x (mm)\n(>0 = leads)")
    ax.set_title("Pure-diffusion front shape in the constriction\n(>0 at edges = inverse crescent)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    p = media_path("source_sink_mismatch_investigation", "images", "hourglass-diffusion-front-curvature")
    fig.savefig(p, dpi=120); plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    main()
