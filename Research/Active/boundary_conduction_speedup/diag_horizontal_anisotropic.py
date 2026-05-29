"""Test how horizontal anisotropy (D_xx > D_yy) affects the wall-row
pre-charge equilibrium under horizontal redirect BC.

Hypothesis: the wall pre-charge level is fed by the bulk wavefront column.
Increasing D_xx (horizontal diffusion):
  - propagates the wavefront eastward faster (eikonal v ∝ √D_xx)
  - changes the gradient profile at the wavefront
  → so pre-charge level may scale with D_xx.

Setup:
  - D2Q9 + MRT (BGK can't represent anisotropic D)
  - Horizontal redirect at top/bottom walls
  - HBB at east/west walls
  - TTP06 EPI, vertical line stim at col 0, T_END = 25 ms
  - Ratios tested: D_xx/D_yy = 1, 2, 4, 8 (D_yy fixed at D0 = 0.001)

Outputs:
  figures/video_aniso_horizontal_{1to1,2to1,4to1,8to1}.mp4
  figures/horizontal_anisotropic_wallcharge.png — wall pre-charge vs ratio,
                                                    plus V(t) trajectories at
                                                    col 20 wall for each ratio
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['animation.ffmpeg_path'] = (
    '/home/norepinephrine/.conda/envs/heart-conduction/bin/ffmpeg'
)
import matplotlib.pyplot as plt
import matplotlib.animation as animation

LBM_ROOT = Path("/home/norepinephrine/Documents/Heart-Conduction/LBM/Engine_V1")
sys.path.insert(0, str(LBM_ROOT))

from src.simulation import LBMSimulation
from src.collision.mrt.d2q9 import mrt_collide_d2q9
from src.streaming.d2q9 import stream_d2q9
from src.boundary.neumann import apply_neumann_d2q9
from src.state import recover_voltage
from src.solver.rush_larsen import compute_source_term, ionic_step

from ionic.ttp06.model import TTP06Model
from ionic.base import CellType

sys.path.insert(0, str(Path(__file__).parent))
from diag_lbm_specular import apply_horizontal_redirect_top_bottom_d2q9


# ---------- config ----------
NX, NY = 41, 21
DX = 0.025
DT = 0.02
D0 = 0.001
V_STIM = 0.0
T_END = 25.0
SAVE_EVERY_MS = 0.25
CS2 = 1.0 / 3.0
FPS = 20

OUT_DIR = Path("/home/norepinephrine/Documents/Heart-Conduction/Research/Active/"
               "boundary_conduction_speedup/figures")

RATIOS = [1, 2, 4, 8]   # D_xx / D_yy


def run(ratio: int):
    D_xx = ratio * D0
    D_yy = D0
    s_jx = 1.0 / (0.5 + D_xx * DT / (CS2 * DX * DX))
    s_jy = 1.0 / (0.5 + D_yy * DT / (CS2 * DX * DX))
    s_e = s_eps = s_q = s_pxx = s_pxy = s_jx

    print(f"  ratio {ratio}:1  D_xx={D_xx:.4g}  D_yy={D_yy:.4g}  "
          f"s_jx={s_jx:.4f}  s_jy={s_jy:.4f}", flush=True)

    device = torch.device("cpu")
    ionic = TTP06Model(cell_type=CellType.EPI, device=device)
    V_rest = float(ionic.V_rest)

    sim = LBMSimulation(
        Nx=NX, Ny=NY, dx=DX, dt=DT, D=D0,
        ionic_model=ionic, Cm=1.0,
        lattice="d2q9", weights_mode="canonical",
    )
    V_init = torch.full((NX, NY), V_rest, dtype=sim.dtype, device=device)
    V_init[0, :] = V_STIM
    sim.V = V_init
    sim.f = sim.w[:, None, None] * sim.V[None, :, :]

    bounce_masks_full = sim.bounce_masks
    n_steps = int(round(T_END / DT))
    save_stride = max(1, int(round(SAVE_EVERY_MS / DT)))
    n_frames = (n_steps // save_stride) + 1

    t_arr = np.empty(n_frames, dtype=np.float64)
    V_hist = np.empty((n_frames, NX, NY), dtype=np.float64)
    t_arr[0] = 0.0
    V_hist[0] = sim.V.cpu().numpy()
    save_idx = 1

    f = sim.f
    V = sim.V
    t0 = time.time()
    for k in range(1, n_steps + 1):
        I_ion = sim.ionic_model.compute_Iion(V.reshape(-1), sim.ionic_states)
        R = compute_source_term(I_ion,
                                 torch.zeros(NX*NY, device=device, dtype=sim.dtype),
                                 sim.Cm).reshape(NX, NY)
        f = mrt_collide_d2q9(f, V, R, sim.dt,
                              s_e, s_eps, s_jx, s_q, s_pxx, s_pxy,
                              sim.w, s_jy=s_jy)
        f_star = f.clone()
        f = stream_d2q9(f)
        f = apply_neumann_d2q9(f, f_star, bounce_masks_full)
        f = apply_horizontal_redirect_top_bottom_d2q9(f, f_star, NX, NY)
        V = recover_voltage(f)
        sim.ionic_states = ionic_step(sim.ionic_model, V.reshape(-1),
                                      sim.ionic_states, sim.dt)

        if k % save_stride == 0 and save_idx < n_frames:
            t_arr[save_idx] = k * DT
            V_hist[save_idx] = V.cpu().numpy()
            save_idx += 1

    elapsed = time.time() - t0
    print(f"           elapsed {elapsed:5.1f}s   "
          f"V ∈ [{V_hist[:save_idx].min():+6.2f}, {V_hist[:save_idx].max():+6.2f}] mV  "
          f"frames={save_idx}", flush=True)
    return t_arr[:save_idx], V_hist[:save_idx]


def render(label, times, V_field):
    out_path = OUT_DIR / f"video_aniso_horizontal_{label}.mp4"
    V_disp = np.transpose(V_field, (0, 2, 1))
    fig = plt.figure(figsize=(8, 4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    im = ax.imshow(V_disp[0], vmin=-90, vmax=40, cmap="viridis",
                    aspect="auto", origin="lower", interpolation="nearest")

    def update(k):
        im.set_data(V_disp[k])
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=len(times),
                                    interval=1000 / FPS, blit=True)
    writer = animation.FFMpegWriter(
        fps=FPS, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
    )
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"           saved  {out_path.name}  "
          f"({out_path.stat().st_size / 1024:.0f} KB)", flush=True)


def main():
    print(f"=== MRT + horizontal redirect, D_xx/D_yy ratios {RATIOS} ===\n")
    runs = {}
    for ratio in RATIOS:
        t, V = run(ratio)
        runs[ratio] = (t, V)
        render(f"{ratio}to1", t, V)

    # ─── Analysis figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    # Left: V(t) at col 20 wall (j=0) for each ratio
    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(RATIOS)))
    for ratio, color in zip(RATIOS, cmap):
        t, V = runs[ratio]
        ax.plot(t, V[:, 20, 0], color=color, lw=2,
                 label=f"D_xx:D_yy = {ratio}:1")
    ax.axhline(-85.23, ls='--', c='black', alpha=0.4, lw=0.8)
    ax.axhline(-40,    ls=':',  c='red',   alpha=0.4, lw=0.8, label='_LAT_thresh')
    ax.set_title("V(t) at col 20, j=0 (wall)  —  horizontal redirect + MRT",
                  fontsize=11)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("V (mV)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)

    # Right: V along wall row (j=0) at t=1 ms — visualize pre-charge profile
    ax = axes[1]
    for ratio, color in zip(RATIOS, cmap):
        t, V = runs[ratio]
        k = int(np.argmin(np.abs(t - 1.0)))
        ax.plot(np.arange(NX), V[k, :, 0], color=color, lw=2,
                 label=f"D_xx:D_yy = {ratio}:1", marker='.', ms=4)
    ax.axhline(-85.23, ls='--', c='black', alpha=0.4, lw=0.8)
    ax.set_title("V along wall row (j=0) at t = 1 ms  —  pre-charge profile",
                  fontsize=11)
    ax.set_xlabel("x col")
    ax.set_ylabel("V (mV)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    fig.suptitle("Does horizontal anisotropy (D_xx > D_yy) change the wall "
                 "pre-charge equilibrium?",
                 fontsize=12)
    OUT = OUT_DIR / "horizontal_anisotropic_wallcharge.png"
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT.name}")

    # ─── Console: wall pre-charge equilibrium ─────────────────────────
    print(f"\n=== Wall pre-charge at col 20, j=0 ===")
    print(f"{'ratio':>8}  " + "  ".join(f"{'t='+str(tt)+'ms':>10}"
                                          for tt in [0.5, 1, 2, 5, 10, 25]))
    for ratio in RATIOS:
        t, V = runs[ratio]
        vals = []
        for tt in [0.5, 1, 2, 5, 10, 25]:
            k = int(np.argmin(np.abs(t - tt)))
            vals.append(f"{V[k, 20, 0]:+10.3f}")
        print(f"  {ratio}:1   " + "  ".join(vals))

    # Mean pre-charge across cols 10-30 at t=1ms (before bulk wave arrives at col 20)
    print(f"\n=== Mean wall pre-charge across cols 10-30 ===")
    print(f"{'ratio':>8}  " + "  ".join(f"{'t='+str(tt)+'ms':>10}"
                                          for tt in [0.5, 1, 2, 5]))
    for ratio in RATIOS:
        t, V = runs[ratio]
        vals = []
        for tt in [0.5, 1, 2, 5]:
            k = int(np.argmin(np.abs(t - tt)))
            mean_v = V[k, 10:31, 0].mean()
            vals.append(f"{mean_v:+10.3f}")
        print(f"  {ratio}:1   " + "  ".join(vals))


if __name__ == "__main__":
    main()
