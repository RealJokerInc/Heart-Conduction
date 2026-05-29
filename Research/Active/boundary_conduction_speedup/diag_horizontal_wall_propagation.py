"""High-resolution trace of how the wall pre-charge spreads in x.

Run LBM-horizontal diffusion-only for 5 ms with very fine save (every 2
LBM steps = 0.04 ms). At each saved time, plot V(x, j=0) — the entire
wall row.

Question: is the wall uniformly depolarized everywhere, or is there a
sharp propagating front separating "depolarized" cols from "rest" cols?
And what happens just behind the stim/bulk wavefront?

Output:
  figures/horizontal_wall_propagation.png — V(x, j=0) at multiple early times
                                              + zoom on cols 0-10 (post-stim region)
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
T_END = 5.0
SAVE_EVERY_STEPS = 2          # save every 2 LBM steps = 0.04 ms
N_STIM_COLS = 1


def run(bc: str, physics: str):
    device = torch.device("cpu")
    ionic = TTP06Model(cell_type=CellType.EPI, device=device)
    V_rest = float(ionic.V_rest)

    sim = LBMSimulation(
        Nx=NX, Ny=NY, dx=DX, dt=DT, D=D,
        ionic_model=ionic, Cm=1.0,
        lattice="d2q9", weights_mode="canonical",
    )
    V_init = torch.full((NX, NY), V_rest, dtype=sim.dtype, device=device)
    V_init[:N_STIM_COLS, :] = V_STIM
    sim.V = V_init
    sim.f = sim.w[:, None, None] * sim.V[None, :, :]
    bounce_masks_full = sim.bounce_masks

    n_steps = int(round(T_END / DT))
    n_save = (n_steps // SAVE_EVERY_STEPS) + 1

    t_arr = np.empty(n_save, dtype=np.float64)
    V_wall = np.empty((n_save, NX), dtype=np.float64)   # j=0 row
    V_sub  = np.empty((n_save, NX), dtype=np.float64)   # j=1 row
    V_full = np.empty((n_save, NX, NY), dtype=np.float64)
    t_arr[0]  = 0.0
    V_wall[0] = sim.V[:, 0].cpu().numpy()
    V_sub[0]  = sim.V[:, 1].cpu().numpy()
    V_full[0] = sim.V.cpu().numpy()
    save_idx = 1

    f = sim.f
    V = sim.V
    t0 = time.time()
    for k in range(1, n_steps + 1):
        if physics == "ttp06":
            I_ion = sim.ionic_model.compute_Iion(V.reshape(-1), sim.ionic_states)
            R = compute_source_term(I_ion,
                                     torch.zeros(NX*NY, device=device, dtype=sim.dtype),
                                     sim.Cm).reshape(NX, NY)
        else:
            R = torch.zeros(NX, NY, device=device, dtype=sim.dtype)

        f = bgk_collide(f, V, R, sim.dt, sim.omega, sim.w)
        f_star = f.clone()
        f = stream_d2q9(f)
        f = apply_neumann_d2q9(f, f_star, bounce_masks_full)
        if bc == "horizontal":
            f = apply_horizontal_redirect_top_bottom_d2q9(f, f_star, NX, NY)
        V = recover_voltage(f)
        if physics == "ttp06":
            sim.ionic_states = ionic_step(sim.ionic_model, V.reshape(-1),
                                          sim.ionic_states, sim.dt)

        if k % SAVE_EVERY_STEPS == 0 and save_idx < n_save:
            t_arr[save_idx]  = k * DT
            V_wall[save_idx] = V[:, 0].cpu().numpy()
            V_sub[save_idx]  = V[:, 1].cpu().numpy()
            V_full[save_idx] = V.cpu().numpy()
            save_idx += 1

    elapsed = time.time() - t0
    print(f"  {bc:<11} {physics:<10}  elapsed {elapsed:5.1f}s  frames={save_idx}",
          flush=True)
    return (t_arr[:save_idx], V_wall[:save_idx], V_sub[:save_idx],
            V_full[:save_idx])


def main():
    print("=== high-resolution wall-row trace (diffusion-only) ===")
    runs = {}
    for bc in ("hbb", "horizontal"):
        runs[bc] = run(bc, "diffusion")

    # ─── 3-panel figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    times_to_plot = [0.04, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.95, len(times_to_plot)))

    # Panel 0,0: V(x, j=0) full row, HBB
    ax = axes[0, 0]
    t_arr, V_wall, _, _ = runs["hbb"]
    for t_target, color in zip(times_to_plot, cmap):
        k = int(np.argmin(np.abs(t_arr - t_target)))
        ax.plot(np.arange(NX), V_wall[k], color=color, lw=1.6,
                 label=f"t={t_target} ms", marker='.', ms=4)
    ax.axhline(-85.23, ls='--', c='black', alpha=0.4, lw=0.8)
    ax.set_title("HBB:  V along wall (j=0) at multiple times", fontsize=11)
    ax.set_xlabel("x col")
    ax.set_ylabel("V (mV)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8, ncol=2)

    # Panel 0,1: V(x, j=0) full row, horizontal
    ax = axes[0, 1]
    t_arr, V_wall, _, _ = runs["horizontal"]
    for t_target, color in zip(times_to_plot, cmap):
        k = int(np.argmin(np.abs(t_arr - t_target)))
        ax.plot(np.arange(NX), V_wall[k], color=color, lw=1.6,
                 label=f"t={t_target} ms", marker='.', ms=4)
    ax.axhline(-85.23, ls='--', c='black', alpha=0.4, lw=0.8)
    ax.set_title("HORIZONTAL:  V along wall (j=0) at multiple times",
                  fontsize=11)
    ax.set_xlabel("x col")
    ax.set_ylabel("V (mV)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8, ncol=2)

    # Panel 1,0: Zoom on cols 0-10 — what happens immediately after the stim?
    ax = axes[1, 0]
    t_arr, V_wall, _, _ = runs["horizontal"]
    zoom_times = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56]
    cmap2 = plt.cm.plasma(np.linspace(0.05, 0.95, len(zoom_times)))
    for t_target, color in zip(zoom_times, cmap2):
        k = int(np.argmin(np.abs(t_arr - t_target)))
        ax.plot(np.arange(11), V_wall[k, :11], color=color, lw=1.8,
                 label=f"t={t_target:.2f} ms", marker='o', ms=5)
    ax.axhline(-85.23, ls='--', c='black', alpha=0.4, lw=0.8)
    ax.set_title("HORIZONTAL zoom: cols 0-10 wall  —  what does\n"
                 "the wall look like RIGHT AFTER the stim cell?", fontsize=11)
    ax.set_xlabel("x col (0-10)")
    ax.set_ylabel("V (mV)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8, ncol=2)

    # Panel 1,1: ratio of wall vs sub-edge (j=1) at multiple times, horizontal
    ax = axes[1, 1]
    t_arr, V_wall, V_sub, _ = runs["horizontal"]
    times_for_diff = [0.1, 0.5, 1.0, 2.0, 5.0]
    cmap3 = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(times_for_diff)))
    for t_target, color in zip(times_for_diff, cmap3):
        k = int(np.argmin(np.abs(t_arr - t_target)))
        diff = V_wall[k] - V_sub[k]
        ax.plot(np.arange(NX), diff, color=color, lw=1.8,
                 label=f"t={t_target} ms", marker='.', ms=4)
    ax.axhline(0, ls='--', c='black', alpha=0.4, lw=0.8)
    ax.set_title("HORIZONTAL:  V(j=0) − V(j=1)  across x  —  the wall-vs-sub gap",
                  fontsize=11)
    ax.set_xlabel("x col")
    ax.set_ylabel("ΔV (mV)   positive = wall hotter than sub-edge")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    fig.suptitle("Wall pre-charge propagation pattern  —  is it a "
                 "front sweeping east, or a uniform plateau, or both?",
                 fontsize=12)
    OUT = Path(__file__).parent / "figures" / "horizontal_wall_propagation.png"
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT.name}")

    # ─── Console: spatial profile snapshots ────────────────────────────
    t_arr, V_wall, _, _ = runs["horizontal"]
    print(f"\n=== horizontal: V(x, j=0) snapshots ===")
    print(f"  cols shown: 0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 38, 40")
    shown_cols = [0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 38, 40]
    print(f"  {'t (ms)':>7}  " + "  ".join(f"c{c:>2}" for c in shown_cols))
    for tt in [0.04, 0.08, 0.16, 0.32, 0.5, 1.0, 2.0, 5.0]:
        k = int(np.argmin(np.abs(t_arr - tt)))
        vals = "  ".join(f"{V_wall[k, c]:+6.1f}" for c in shown_cols)
        print(f"  {tt:>7.3f}  {vals}")


if __name__ == "__main__":
    main()
