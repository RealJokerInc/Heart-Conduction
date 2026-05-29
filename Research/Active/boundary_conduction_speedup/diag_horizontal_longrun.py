"""Long-run wall-row trajectory under horizontal redirect.

Question: does the uniform wall pre-depolarization keep growing, or
saturate, with longer simulation time? Run T_END = 200 ms (10× the
standard run) for both diffusion-only and TTP06.

Stores only the wall row V(x, j=0) over time, plus the j=1 sub-edge
row, to keep file size small.

Output:
  figures/horizontal_longrun.png — V(t) at wall (j=0) and sub-edge (j=1)
                                    at multiple cols, HBB vs horizontal,
                                    diffusion + TTP06.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

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

# Import the BC routines from the diag script
sys.path.insert(0, str(Path(__file__).parent))
from diag_lbm_specular import apply_horizontal_redirect_top_bottom_d2q9

# ---------- config ----------
NX, NY = 41, 21
DX = 0.025
DT = 0.02
D = 0.001
V_STIM = 0.0
T_END = 200.0                       # 10× standard
SAVE_EVERY_MS = 0.5                 # 400 frames
N_FRAMES = int(round(T_END / SAVE_EVERY_MS)) + 1


def run(bc: str, physics: str):
    """Returns (t_arr, V_wall_arr, V_sub_arr) where each *_arr is shape (n_frames, NX)."""
    device = torch.device("cpu")
    ionic = TTP06Model(cell_type=CellType.EPI, device=device)
    V_rest = float(ionic.V_rest)

    sim = LBMSimulation(
        Nx=NX, Ny=NY, dx=DX, dt=DT, D=D,
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

    t_arr     = np.empty(N_FRAMES, dtype=np.float64)
    V_wall    = np.empty((N_FRAMES, NX), dtype=np.float64)
    V_sub     = np.empty((N_FRAMES, NX), dtype=np.float64)
    t_arr[0]  = 0.0
    V_wall[0] = sim.V[:, 0].cpu().numpy()
    V_sub[0]  = sim.V[:, 1].cpu().numpy()
    save_idx  = 1

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

        if k % save_stride == 0 and save_idx < N_FRAMES:
            t_arr[save_idx]  = k * DT
            V_wall[save_idx] = V[:, 0].cpu().numpy()
            V_sub[save_idx]  = V[:, 1].cpu().numpy()
            save_idx += 1

    elapsed = time.time() - t0
    print(f"  {bc:<11} {physics:<10}  elapsed {elapsed:5.1f}s   "
          f"V_wall final: max={V_wall[-1].max():+6.2f}  "
          f"mean(cols10-30)={V_wall[-1, 10:31].mean():+6.2f}")
    return t_arr[:save_idx], V_wall[:save_idx], V_sub[:save_idx]


def main():
    print(f"Long-run wall-row trajectory: T_END = {T_END} ms ({int(T_END/DT)} steps)")
    print()

    runs = {}
    for physics in ("diffusion", "ttp06"):
        for bc in ("hbb", "horizontal"):
            t, vw, vs = run(bc, physics)
            runs[(bc, physics)] = (t, vw, vs)

    # ─── 2x2 figure: top=diffusion, bottom=TTP06; left=wall traces, right=sub-edge traces
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    cols_to_plot = [5, 10, 20, 30]
    color_cols = {5: 'C0', 10: 'C1', 20: 'C2', 30: 'C3'}

    for row, physics in enumerate(("diffusion", "ttp06")):
        # Left: V(t) at wall (j=0)
        ax = axes[row, 0]
        for bc, ls in [("hbb", "--"), ("horizontal", "-")]:
            t, vw, _ = runs[(bc, physics)]
            for c in cols_to_plot:
                ax.plot(t, vw[:, c], ls=ls, color=color_cols[c],
                         label=f"{bc} c{c}" if row == 0 else None, lw=1.5)
        ax.axhline(-85.23, ls=':', c='black', alpha=0.5, lw=0.8)
        ax.set_title(f"V at wall row (j=0)  —  {physics}", fontsize=11)
        ax.set_xlabel("t (ms)")
        ax.set_ylabel("V (mV)")
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.legend(loc='lower right', fontsize=8, ncol=2)

        # Right: V(t) at sub-edge (j=1)
        ax = axes[row, 1]
        for bc, ls in [("hbb", "--"), ("horizontal", "-")]:
            t, _, vs = runs[(bc, physics)]
            for c in cols_to_plot:
                ax.plot(t, vs[:, c], ls=ls, color=color_cols[c], lw=1.5)
        ax.axhline(-85.23, ls=':', c='black', alpha=0.5, lw=0.8)
        ax.set_title(f"V at sub-edge row (j=1)  —  {physics}", fontsize=11)
        ax.set_xlabel("t (ms)")
        ax.set_ylabel("V (mV)")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Long-run (T = {T_END:.0f} ms) wall-row evolution under horizontal redirect vs HBB.\n"
        "Question: does the wall pre-depolarization keep growing?",
        fontsize=12,
    )
    OUT = Path(__file__).parent / "figures" / "horizontal_longrun.png"
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT.name}")

    # ─── Console: trajectory of V at (col 20, j=0) over time ───────────
    print(f"\n=== V(col=20, j=0) trajectory ===")
    print(f"{'t (ms)':>8}  " + "  ".join(f"{label:>22}" for label in
                                          ["hbb-diff", "horiz-diff", "hbb-ttp06", "horiz-ttp06"]))
    sample_ts = [0, 1, 5, 10, 25, 50, 100, 150, 200]
    for tt in sample_ts:
        row_vals = []
        for physics in ("diffusion", "ttp06"):
            for bc in ("hbb", "horizontal"):
                t, vw, _ = runs[(bc, physics)]
                k = int(np.argmin(np.abs(t - tt)))
                row_vals.append(f"{vw[k, 20]:+22.4f}")
        print(f"{tt:>8.1f}  " + "  ".join(row_vals))


if __name__ == "__main__":
    main()
