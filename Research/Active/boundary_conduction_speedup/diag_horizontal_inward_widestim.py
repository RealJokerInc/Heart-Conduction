"""Two questions answered in one script:

(a) Inward diffusion check (TTP06): does the wall row's pre-depolarization
    propagate inward to j=1, j=2, j=3, ...? Plot V(t) at (col 20, j=k) for
    k = 0..NY-1, horizontal vs HBB.

(b) Wider stim region: does stimulating cols 0..4 (5-col-wide line stim)
    instead of just col 0 worsen the artifact? Run horizontal + HBB with
    both narrow and wide stim, render videos and compare V_wall(x, t).

Outputs:
  figures/horizontal_inward_diffusion.png — V(t) at col 20 across j rows
  figures/horizontal_widestim_compare.png — wavefront snapshots,
                                             narrow vs wide stim
  figures/video_widestim_horizontal_5col.mp4 — 5-col-stim TTP06 horizontal
  figures/video_widestim_hbb_5col.mp4         — 5-col-stim TTP06 HBB (control)
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
from src.collision.bgk import bgk_collide
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
D = 0.001
V_STIM = 0.0
T_END = 200.0
SAVE_EVERY_MS = 1.0
FPS = 20

OUT_DIR = Path("/home/norepinephrine/Documents/Heart-Conduction/Research/Active/"
               "boundary_conduction_speedup/figures")


def run(bc: str, n_stim_cols: int, t_end: float = T_END):
    """Returns (t_arr, V_field) with V_field shape (n_frames, NX, NY).

    n_stim_cols: how many leftmost cols to set to V_STIM in IC.
    """
    device = torch.device("cpu")
    ionic = TTP06Model(cell_type=CellType.EPI, device=device)
    V_rest = float(ionic.V_rest)

    sim = LBMSimulation(
        Nx=NX, Ny=NY, dx=DX, dt=DT, D=D,
        ionic_model=ionic, Cm=1.0,
        lattice="d2q9", weights_mode="canonical",
    )
    V_init = torch.full((NX, NY), V_rest, dtype=sim.dtype, device=device)
    V_init[:n_stim_cols, :] = V_STIM
    sim.V = V_init
    sim.f = sim.w[:, None, None] * sim.V[None, :, :]

    bounce_masks_full = sim.bounce_masks
    n_steps = int(round(t_end / DT))
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
        f = bgk_collide(f, V, R, sim.dt, sim.omega, sim.w)
        f_star = f.clone()
        f = stream_d2q9(f)
        f = apply_neumann_d2q9(f, f_star, bounce_masks_full)
        if bc == "horizontal":
            f = apply_horizontal_redirect_top_bottom_d2q9(f, f_star, NX, NY)
        V = recover_voltage(f)
        sim.ionic_states = ionic_step(sim.ionic_model, V.reshape(-1),
                                      sim.ionic_states, sim.dt)

        if k % save_stride == 0 and save_idx < n_frames:
            t_arr[save_idx] = k * DT
            V_hist[save_idx] = V.cpu().numpy()
            save_idx += 1

    elapsed = time.time() - t0
    print(f"  {bc:<11} n_stim={n_stim_cols}  elapsed {elapsed:5.1f}s   "
          f"V ∈ [{V_hist[:save_idx].min():+6.2f}, {V_hist[:save_idx].max():+6.2f}] mV  "
          f"frames={save_idx}", flush=True)
    return t_arr[:save_idx], V_hist[:save_idx]


def render_video(label, times, V_field, v_min=-90.0, v_max=40.0):
    out_path = OUT_DIR / f"video_widestim_{label}.mp4"
    V_disp = np.transpose(V_field, (0, 2, 1))
    fig = plt.figure(figsize=(8, 4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    im = ax.imshow(V_disp[0], vmin=v_min, vmax=v_max, cmap="viridis",
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
    print("=== Running sims ===")
    # Run all 4 cases
    runs = {}
    for bc in ("hbb", "horizontal"):
        for n_stim in (1, 5):
            runs[(bc, n_stim)] = run(bc, n_stim)

    # ─── Figure 1: Inward diffusion check (1-col stim) ────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    cols_to_check = 20
    rows_to_plot = [0, 1, 2, 3, 5, 8, 10]
    cmap_j = plt.cm.viridis(np.linspace(0.05, 0.95, len(rows_to_plot)))

    for ax, (bc, n_stim) in zip(axes, [("hbb", 1), ("horizontal", 1)]):
        t, V = runs[(bc, n_stim)]
        for j, color in zip(rows_to_plot, cmap_j):
            ax.plot(t, V[:, cols_to_check, j], color=color, lw=1.8,
                     label=f"j={j}")
        ax.axhline(-85.23, ls='--', c='black', alpha=0.4, lw=0.8)
        ax.axhline(-40,    ls=':',  c='red',   alpha=0.4, lw=0.8, label='_LAT_thresh')
        ax.set_title(f"V(t) at col 20, multiple j  —  {bc}", fontsize=11)
        ax.set_xlabel("t (ms)")
        ax.set_ylabel("V (mV)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9, ncol=2)

    fig.suptitle("Inward diffusion check: does j=0's pre-depolarization "
                 "propagate to j=1, 2, 3, ...?\n"
                 "If yes: lines for different j should be ordered top-to-bottom by j. "
                 "If no: lines clump.",
                 fontsize=12)
    OUT1 = OUT_DIR / "horizontal_inward_diffusion.png"
    fig.savefig(OUT1, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT1.name}")

    # ─── Figure 2: Narrow vs wide stim comparison ──────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    snapshots_ms = [1, 5, 25, 100]
    for row, n_stim in enumerate([1, 5]):
        t, V = runs[("horizontal", n_stim)]
        for col, t_target in enumerate(snapshots_ms):
            ax = axes[row, col]
            k = int(np.argmin(np.abs(t - t_target)))
            im = ax.imshow(V[k].T, origin="lower", aspect="auto",
                            vmin=-90, vmax=40, cmap="viridis",
                            interpolation="nearest")
            ax.set_title(f"horiz {n_stim}-col stim, t = {t_target} ms",
                         fontsize=10)
            ax.set_xlabel("x col")
            ax.set_ylabel("y row")
    fig.colorbar(im, ax=axes, shrink=0.8, label="V (mV)")
    fig.suptitle("Horizontal redirect — narrow (1-col) vs wider (5-col) stim",
                 fontsize=12)
    OUT2 = OUT_DIR / "horizontal_widestim_compare.png"
    fig.savefig(OUT2, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT2.name}")

    # ─── Videos: 5-col stim, both HBB and horizontal ──────────────────
    print("\n=== Rendering 5-col-stim videos ===")
    for bc in ("hbb", "horizontal"):
        t, V = runs[(bc, 5)]
        render_video(f"{bc}_5col", t, V)

    # ─── Console: pre-charge level at col 20 wall, narrow vs wide ─────
    print("\n=== Pre-charge level at col 20 wall (j=0) ===")
    print(f"{'t (ms)':>8}  {'horiz-1col':>12}  {'horiz-5col':>12}  {'Δ':>10}")
    for tt in [0.5, 1, 2, 5, 10, 25, 50, 100, 200]:
        t1, V1 = runs[("horizontal", 1)]
        t5, V5 = runs[("horizontal", 5)]
        k1 = int(np.argmin(np.abs(t1 - tt)))
        k5 = int(np.argmin(np.abs(t5 - tt)))
        v1 = V1[k1, 20, 0]
        v5 = V5[k5, 20, 0]
        print(f"{tt:>8.1f}  {v1:>+12.3f}  {v5:>+12.3f}  {v5-v1:>+10.3f}")

    # Also sub-edge dip
    print("\n=== Sub-edge V at col 20 (j=1) ===")
    print(f"{'t (ms)':>8}  {'horiz-1col':>12}  {'horiz-5col':>12}  {'Δ':>10}")
    for tt in [0.5, 1, 2, 5, 10, 25, 50, 100, 200]:
        t1, V1 = runs[("horizontal", 1)]
        t5, V5 = runs[("horizontal", 5)]
        k1 = int(np.argmin(np.abs(t1 - tt)))
        k5 = int(np.argmin(np.abs(t5 - tt)))
        v1 = V1[k1, 20, 1]
        v5 = V5[k5, 20, 1]
        print(f"{tt:>8.1f}  {v1:>+12.3f}  {v5:>+12.3f}  {v5-v1:>+10.3f}")


if __name__ == "__main__":
    main()
