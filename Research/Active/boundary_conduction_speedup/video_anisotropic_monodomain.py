"""Anisotropic monodomain videos — D_x : D_y = 2:1 and 1:2.

Same grid + colormap as the previous BC-family videos. Cardinal-4 stencil
(only stencil that supports anisotropic D in V5.4 FDM). face_mirror BC.
Vertical line stim at col 0 (leftmost wall), TTP06 EPI, 25 ms.

Two configurations:
  2:1  → D_x = 2·D0, D_y = D0   (fiber along x; "fast along propagation")
  1:2  → D_x = D0,   D_y = 2·D0 (fiber along y; "fast along the wall")
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

ENGINE = Path("/home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.4")
sys.path.insert(0, str(ENGINE))

from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol
from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation


# ---------- config ----------
LX, LY = 1.0, 0.5
DX = 0.025
NX = int(round(LX / DX)) + 1   # 41
NY = int(round(LY / DX)) + 1   # 21
DT = 0.02
T_END = 25.0
SAVE_EVERY = 0.25
D0 = 0.001                       # base diffusion coefficient (cm² / ms)
V_MIN, V_MAX = -90.0, 40.0       # match existing BC-family videos
CMAP = "viridis"
FPS = 20

CONFIGS = [
    # (label, D_xx_scale, D_yy_scale, title_suffix)
    ("aniso_monodomain_2to1", 2.0, 1.0, "D_x : D_y = 2 : 1"),
    ("aniso_monodomain_1to2", 1.0, 2.0, "D_x : D_y = 1 : 2"),
]

OUT_DIR = Path("/home/norepinephrine/Documents/Heart-Conduction/Research/Active/"
               "boundary_conduction_speedup/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_one(label, sx, sy):
    print(f"[sim {label}]  D_xx = {sx*D0:.4g}  D_yy = {sy*D0:.4g}", flush=True)
    grid = StructuredGrid.create_rectangle(LX, LY, NX, NY)

    # Anisotropic D_field: full-grid tensors with constant Dxx, Dyy, Dxy=0.
    # cardinal4 is the only stencil that supports Dxy ≠ 0 OR anisotropic Dxx≠Dyy
    # under all BC modes (moore8_* paths reject anisotropy).
    device = grid.device
    dtype = grid.dtype
    Dxx = torch.full((NX, NY), sx * D0, device=device, dtype=dtype)
    Dyy = torch.full((NX, NY), sy * D0, device=device, dtype=dtype)
    Dxy = torch.zeros(NX, NY, device=device, dtype=dtype)

    fdm = FDMDiscretization(
        grid, chi=1.0, Cm=1.0,
        D_field=(Dxx, Dxy, Dyy),
        stencil='cardinal4', boundary_mode='face_mirror',
    )

    proto = StimulusProtocol()
    # Vertical line stim at the most leftward column (col 0).
    proto.add_stimulus(
        region=lambda x, y: x < DX / 2,
        start_time=0.0, duration=2.0, amplitude=-52.0,
    )

    sim = MonodomainSimulation(
        spatial=fdm, ionic_model='ttp06', stimulus=proto,
        dt=DT, splitting='strang', ionic_solver='rush_larsen',
        diffusion_solver='forward_euler', cell_type='EPI',
    )
    times, V_hist = sim.run_to_array(t_end=T_END, save_every=SAVE_EVERY)
    V_field = V_hist.reshape(len(times), NX, NY)
    print(f"           done. V ∈ [{V_field.min():.2f}, {V_field.max():.2f}] mV  "
          f"frames={len(times)}", flush=True)
    return times, V_field


def render(label, times, V_field):
    out_path = OUT_DIR / f"video_{label}.mp4"
    # Display orientation: x horizontal, y vertical, walls top/bottom.
    # Match render_bc_videos.py: full-figure axes, no titles/colorbar in frame.
    V_disp = np.transpose(V_field, (0, 2, 1))   # (T, NY, NX)

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
    for label, sx, sy, suffix in CONFIGS:
        times, V_field = run_one(label, sx, sy)
        render(label, times, V_field)
    print("\n[done] anisotropic monodomain videos written.")


if __name__ == "__main__":
    main()
