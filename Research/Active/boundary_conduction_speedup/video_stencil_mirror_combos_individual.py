"""Per-combo videos for the same 5 (stencil × boundary_mode) cases as
video_stencil_mirror_combos.py, but each saved as its own standalone mp4 so
panels can be embedded / discussed individually.

Same physics: 1.0 × 0.5 cm tissue, dx=0.025, line stim at col 0 (leftmost wall,
full y), TTP06 EPI, forward-Euler diffusion, T_END=25 ms, dt=0.02 ms.

Outputs (one per combo):
  figures/video_cardinal4_face_mirror.mp4
  figures/video_moore8_uniform_face_mirror.mp4
  figures/video_moore8_uniform_face_iso.mp4
  figures/video_moore8_iso_face_mirror.mp4
  figures/video_moore8_iso_face_iso.mp4
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


# ---------- config (matches video_stencil_mirror_combos.py) ----------
LX, LY = 1.0, 0.5
DX = 0.025
NX = int(round(LX / DX)) + 1   # 41
NY = int(round(LY / DX)) + 1   # 21
DT = 0.02
T_END = 25.0
SAVE_EVERY = 0.25
V_MIN, V_MAX = -90.0, 50.0
FPS = 20

COMBOS = [
    ("cardinal4_face_mirror",      "cardinal4",      "face_mirror",
     "cardinal-4 + face_mirror\n(baseline; no diag)"),
    ("moore8_uniform_face_mirror", "moore8_uniform", "face_mirror",
     "moore8 uniform + face_mirror\n(DEFICIT — John artifact)"),
    ("moore8_uniform_face_iso",    "moore8_uniform", "face_mirror_iso",
     "moore8 uniform + face_iso\n(FIX — bounce-back)"),
    ("moore8_iso_face_mirror",     "moore8_iso",     "face_mirror",
     "moore8 iso 4:1 + face_mirror\n(smaller deficit, 5/6)"),
    ("moore8_iso_face_iso",        "moore8_iso",     "face_mirror_iso",
     "moore8 iso 4:1 + face_iso\n(LBM analog — zero deficit)"),
]

OUT_DIR = Path("/home/norepinephrine/Documents/Heart-Conduction/Research/Active/"
               "boundary_conduction_speedup/figures")


def run_one(label, stencil, boundary_mode):
    print(f"[sim {label}] stencil={stencil}  bc={boundary_mode}", flush=True)
    grid = StructuredGrid.create_rectangle(LX, LY, NX, NY)
    fdm = FDMDiscretization(
        grid, D=0.001, chi=1.0, Cm=1.0,
        stencil=stencil, boundary_mode=boundary_mode,
    )
    proto = StimulusProtocol()
    # Stim at the most leftward column only (col 0).
    # DX/2 threshold catches x=0.0 but excludes x=DX (col 1).
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


# ---------- run all 5 ----------
results = {}
for label, stencil, bm, _ in COMBOS:
    results[label] = run_one(label, stencil, bm)
n_frames = len(results[COMBOS[0][0]][0])

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- one mp4 per combo ----------
for label, _, _, title in COMBOS:
    times, V_field = results[label]
    out_path = OUT_DIR / f"video_{label}.mp4"

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    im = ax.imshow(
        V_field[0].T, origin='lower',
        extent=[0, LX, 0, LY], aspect='equal',
        cmap='RdBu_r', vmin=V_MIN, vmax=V_MAX,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    fig.colorbar(im, ax=ax, label="V (mV)", shrink=0.85)
    suptitle = fig.suptitle(f"t = {times[0]:6.2f} ms", fontsize=12)

    def update(frame, _im=im, _suptitle=suptitle, _V=V_field, _t=times):
        _im.set_data(_V[frame].T)
        _suptitle.set_text(f"t = {_t[frame]:6.2f} ms")
        return [_im, _suptitle]

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / FPS, blit=False,
    )

    print(f"[video {label}] writing {out_path.name}  "
          f"({n_frames} frames @ {FPS} fps ~ {n_frames/FPS:.1f} s)",
          flush=True)
    # `pad` filter rounds odd dimensions up to even; libx264+yuv420p needs even.
    writer = animation.FFMpegWriter(
        fps=FPS, codec='libx264', bitrate=2500,
        extra_args=[
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
            '-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '20',
        ],
    )
    anim.save(str(out_path), writer=writer, dpi=130)
    plt.close(fig)
    print(f"           saved  {out_path}  "
          f"({out_path.stat().st_size / 1024:.0f} KB)", flush=True)

print("\n[done] all 5 individual videos written.")
