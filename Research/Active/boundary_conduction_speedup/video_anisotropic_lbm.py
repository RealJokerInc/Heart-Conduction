"""Anisotropic LBM videos — D_xx : D_yy = 2:1 and 1:2.

Same grid + colormap as the previous BC-family videos. D2Q9 + MRT collision
(BGK can't represent D_xx != D_yy), HBB at all walls. V poke at col 0 (left
wall) sets initial condition. TTP06 EPI, 25 ms.

Two configurations:
  2:1  → D_xx = 2·D0, D_yy = D0
  1:2  → D_xx = D0,   D_yy = 2·D0
"""
from __future__ import annotations
import sys
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


# ---------- config ----------
LX, LY = 1.0, 0.5
DX = 0.025
NX = int(round(LX / DX)) + 1   # 41
NY = int(round(LY / DX)) + 1   # 21
DT = 0.02
T_END = 25.0
SAVE_EVERY_MS = 0.25
D0 = 0.001
V_STIM = 0.0                     # mV (sub-threshold poke at col 0)
V_MIN, V_MAX = -90.0, 40.0
CMAP = "viridis"
FPS = 20

CS2 = 1.0 / 3.0                  # canonical D2Q9 lattice sound-speed²

CONFIGS = [
    # (label, D_xx_scale, D_yy_scale)
    ("aniso_lbm_2to1", 2.0, 1.0),
    ("aniso_lbm_1to2", 1.0, 2.0),
]

OUT_DIR = Path("/home/norepinephrine/Documents/Heart-Conduction/Research/Active/"
               "boundary_conduction_speedup/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_one(label, sx, sy):
    D_xx = sx * D0
    D_yy = sy * D0
    # MRT relaxation rates from anisotropic diffusion tensor.
    # From test_phase5.py:175 and src/collision/mrt/d2q9.py docstring:
    #   D_xx = cs2 · (1/s_jx − 0.5) · dt
    #   → s_jx = 1 / (0.5 + D_xx · dt / (cs2 · dx²))
    s_jx = 1.0 / (0.5 + D_xx * DT / (CS2 * DX * DX))
    s_jy = 1.0 / (0.5 + D_yy * DT / (CS2 * DX * DX))
    # Free moments — fixed at s_jx (Chapman-Enskog accuracy not sensitive)
    s_e = s_eps = s_q = s_pxx = s_pxy = s_jx

    print(f"[sim {label}]  D_xx={D_xx:.4g}  D_yy={D_yy:.4g}  "
          f"s_jx={s_jx:.4f}  s_jy={s_jy:.4f}", flush=True)

    device = torch.device("cpu")
    ionic = TTP06Model(cell_type=CellType.EPI, device=device)
    V_rest = float(ionic.V_rest)

    # We use LBMSimulation only for bookkeeping (Ionic state, masks, weights);
    # the actual step loop calls MRT directly since LBMSimulation hard-codes BGK.
    sim = LBMSimulation(
        Nx=NX, Ny=NY, dx=DX, dt=DT, D=D0,    # D here is unused (we override via MRT)
        ionic_model=ionic, Cm=1.0,
        lattice="d2q9", weights_mode="canonical",
    )

    V_init = torch.full((NX, NY), V_rest, dtype=sim.dtype, device=device)
    V_init[0, :] = V_STIM
    sim.V = V_init
    sim.f = sim.w[:, None, None] * sim.V[None, :, :]

    n_steps = int(round(T_END / DT))
    save_stride = max(1, int(round(SAVE_EVERY_MS / DT)))
    n_save = (n_steps // save_stride) + 1
    V_hist = np.empty((n_save, NX, NY), dtype=np.float64)
    t_hist = np.empty(n_save, dtype=np.float64)
    V_hist[0] = sim.V.cpu().numpy()
    t_hist[0] = 0.0
    save_idx = 1

    f = sim.f
    V = sim.V
    for k in range(1, n_steps + 1):
        I_stim = torch.zeros(NX, NY, device=device, dtype=sim.dtype)
        I_ion = sim.ionic_model.compute_Iion(V.reshape(-1), sim.ionic_states)
        R = compute_source_term(I_ion, I_stim.reshape(-1), sim.Cm).reshape(NX, NY)

        f = mrt_collide_d2q9(f, V, R, sim.dt,
                              s_e, s_eps, s_jx, s_q, s_pxx, s_pxy,
                              sim.w, s_jy=s_jy)
        f_star = f.clone()
        f = stream_d2q9(f)
        f = apply_neumann_d2q9(f, f_star, sim.bounce_masks)
        V = recover_voltage(f)
        sim.ionic_states = ionic_step(sim.ionic_model, V.reshape(-1),
                                      sim.ionic_states, sim.dt)

        if k % save_stride == 0 and save_idx < n_save:
            V_hist[save_idx] = V.cpu().numpy()
            t_hist[save_idx] = k * DT
            save_idx += 1

    V_field = V_hist[:save_idx]
    t_field = t_hist[:save_idx]
    print(f"           done. V ∈ [{V_field.min():.2f}, {V_field.max():.2f}] mV  "
          f"frames={len(t_field)}", flush=True)
    return t_field, V_field


def render(label, times, V_field):
    out_path = OUT_DIR / f"video_{label}.mp4"
    V_disp = np.transpose(V_field, (0, 2, 1))

    fig = plt.figure(figsize=(8, 4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    im = ax.imshow(
        V_disp[0], vmin=V_MIN, vmax=V_MAX, cmap=CMAP,
        aspect="auto", origin="lower", interpolation="nearest",
    )

    def update(k):
        im.set_data(V_disp[k])
        return [im]

    anim = animation.FuncAnimation(
        fig, update, frames=len(times), interval=1000 / FPS, blit=True,
    )
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
    for label, sx, sy in CONFIGS:
        times, V_field = run_one(label, sx, sy)
        render(label, times, V_field)
    print("\n[done] anisotropic LBM videos written.")


if __name__ == "__main__":
    main()
