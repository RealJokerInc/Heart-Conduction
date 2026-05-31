"""Render the combined HBB<->same-cell-specular axis as a 5-panel video.

alpha = HBB weight: 1.0 (forward) ... 0.0 (inverse, same-cell specular).
All panels share the colormap and time clock. Shows the crescent morphing
from forward (boundary lags) through flat to inverse (boundary leads), with
NO wall pre-charge artifact at any alpha.

Output: figures/video_combined_axis.mp4
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
sys.path.insert(0, str(Path(__file__).parent))

from src.simulation import LBMSimulation
from src.collision.bgk import bgk_collide
from src.streaming.d2q9 import stream_d2q9
from src.boundary.neumann import apply_neumann_d2q9
from src.state import recover_voltage
from src.solver.rush_larsen import compute_source_term, ionic_step
from ionic.ttp06.model import TTP06Model
from ionic.base import CellType
from diag_lbm_specular import apply_combined_top_bottom_d2q9

NX, NY = 41, 21
DX, DT, D = 0.025, 0.02, 0.001
T_END = 25.0
SAVE_EVERY_MS = 0.25
V_MIN, V_MAX = -90.0, 40.0
CMAP = "viridis"
FPS = 20
LX, LY = (NX - 1) * DX, (NY - 1) * DX

ALPHAS = [1.0, 0.75, 0.5, 0.25, 0.0]
LABELS = [
    "α=1.0  pure HBB (forward)",
    "α=0.75",
    "α=0.5",
    "α=0.25",
    "α=0.0  same-cell specular (inverse)",
]

OUT = Path(__file__).parent / "figures" / "video_combined_axis.mp4"


def run(alpha):
    ionic = TTP06Model(cell_type=CellType.EPI, device=torch.device("cpu"))
    V_rest = float(ionic.V_rest)
    sim = LBMSimulation(Nx=NX, Ny=NY, dx=DX, dt=DT, D=D, ionic_model=ionic,
                        Cm=1.0, lattice="d2q9", weights_mode="canonical")
    V = torch.full((NX, NY), V_rest, dtype=sim.dtype)
    V[0, :] = 0.0
    f = sim.w[:, None, None] * V[None, :, :]
    n_steps = int(round(T_END / DT))
    save_stride = max(1, int(round(SAVE_EVERY_MS / DT)))
    n_frames = (n_steps // save_stride) + 1
    t_arr = np.empty(n_frames)
    V_hist = np.empty((n_frames, NX, NY))
    t_arr[0] = 0.0
    V_hist[0] = V.cpu().numpy()
    s = 1
    for k in range(1, n_steps + 1):
        I = sim.ionic_model.compute_Iion(V.reshape(-1), sim.ionic_states)
        R = compute_source_term(I, torch.zeros(NX * NY, dtype=sim.dtype), sim.Cm).reshape(NX, NY)
        f = bgk_collide(f, V, R, sim.dt, sim.omega, sim.w)
        fs = f.clone()
        f = stream_d2q9(f)
        f = apply_neumann_d2q9(f, fs, sim.bounce_masks)
        f = apply_combined_top_bottom_d2q9(f, fs, NX, NY, alpha)
        V = recover_voltage(f)
        sim.ionic_states = ionic_step(sim.ionic_model, V.reshape(-1), sim.ionic_states, sim.dt)
        if k % save_stride == 0 and s < n_frames:
            t_arr[s] = k * DT
            V_hist[s] = V.cpu().numpy()
            s += 1
    print(f"  alpha={alpha:.2f} done  V in [{V_hist[:s].min():.2f},{V_hist[:s].max():.2f}]",
          flush=True)
    return t_arr[:s], V_hist[:s]


def main():
    print("Running 5 combined-axis sims...")
    results = [run(a) for a in ALPHAS]
    n_frames = min(len(t) for t, _ in results)
    times = results[0][0][:n_frames]

    fig, axes = plt.subplots(5, 1, figsize=(7, 11), constrained_layout=True)
    ims = []
    for ax, (label, (t, Vf)) in zip(axes, zip(LABELS, results)):
        im = ax.imshow(Vf[0].T, origin="lower", extent=[0, LX, 0, LY],
                       aspect="equal", cmap=CMAP, vmin=V_MIN, vmax=V_MAX,
                       interpolation="nearest")
        ax.set_title(label, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ims.append((im, Vf))
    suptitle = fig.suptitle(f"Combined BC axis (HBB↔same-cell specular)   t = {times[0]:5.2f} ms",
                            fontsize=12)

    def update(k):
        for im, Vf in ims:
            im.set_data(Vf[k].T)
        suptitle.set_text(f"Combined BC axis (HBB↔same-cell specular)   t = {times[k]:5.2f} ms")
        return [im for im, _ in ims] + [suptitle]

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=1000 / FPS, blit=False)
    writer = animation.FFMpegWriter(
        fps=FPS, codec="libx264", bitrate=3000,
        extra_args=["-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
    )
    anim.save(str(OUT), writer=writer, dpi=120)
    plt.close(fig)
    print(f"Saved {OUT.name}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
