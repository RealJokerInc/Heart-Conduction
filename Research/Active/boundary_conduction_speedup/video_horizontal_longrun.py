"""Long-run (200 ms) videos for visual confirmation of wall behavior under
horizontal redirect vs HBB. Diffusion-only AND TTP06.

Output: figures/video_longrun_{diff,ttp06}_{hbb,horiz}.mp4

What to look for:
  - diff/horiz: wall row pre-charges uniformly to ~−64 mV within ~25 ms,
                stays there; sub-edge dips below rest. HBB: gentle relaxation
                to ~−83 mV everywhere.
  - ttp06/horiz: wall AP triggers ~40 ms LATER than HBB because pre-charging
                  inactivates Na channels. Bulk AP runs normally.
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
CMAP = "viridis"

OUT_DIR = Path("/home/norepinephrine/Documents/Heart-Conduction/Research/Active/"
               "boundary_conduction_speedup/figures")


def run(bc: str, physics: str):
    """Returns (t_arr, V_field) with V_field shape (n_frames, NX, NY)."""
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

        if k % save_stride == 0 and save_idx < n_frames:
            t_arr[save_idx] = k * DT
            V_hist[save_idx] = V.cpu().numpy()
            save_idx += 1

    elapsed = time.time() - t0
    print(f"  {bc:<11} {physics:<10}  elapsed {elapsed:5.1f}s   "
          f"V ∈ [{V_hist[:save_idx].min():+6.2f}, {V_hist[:save_idx].max():+6.2f}] mV  "
          f"frames={save_idx}", flush=True)
    return t_arr[:save_idx], V_hist[:save_idx]


def render(label, times, V_field, v_min, v_max):
    out_path = OUT_DIR / f"video_longrun_{label}.mp4"
    V_disp = np.transpose(V_field, (0, 2, 1))   # (T, NY, NX)

    fig = plt.figure(figsize=(8, 4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    im = ax.imshow(
        V_disp[0], vmin=v_min, vmax=v_max, cmap=CMAP,
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
    print(f"Long-run videos: T = {T_END:.0f} ms, save_every {SAVE_EVERY_MS} ms, "
          f"{FPS} fps, T_video = {T_END/SAVE_EVERY_MS/FPS:.1f} s\n")

    # Diffusion-only: V range [-95, 0] to show the dip and pre-charge clearly
    for bc in ("hbb", "horizontal"):
        t, V = run(bc, "diffusion")
        render(f"diff_{bc}", t, V, v_min=-95.0, v_max=0.0)

    # TTP06: V range [-90, 40] matches the existing BC-family videos
    for bc in ("hbb", "horizontal"):
        t, V = run(bc, "ttp06")
        render(f"ttp06_{bc}", t, V, v_min=-90.0, v_max=40.0)

    print("\n[done] 4 long-run videos written.")


if __name__ == "__main__":
    main()
