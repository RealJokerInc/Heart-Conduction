"""Side-by-side video of monodomain V(x,y,t) for all 5 (stencil × boundary_mode)
combinations from PLAN Phase 3:

  cardinal4    + face_mirror       (baseline — no diagonals to lose)
  moore8_uniform + face_mirror     (DEFICIT — John's artifact in monodomain)
  moore8_uniform + face_mirror_iso (FIX — bounce-back eliminates deficit)
  moore8_iso   + face_mirror       (smaller deficit — iso 4:1 reduces ratio to 5/6)
  moore8_iso   + face_mirror_iso   (LBM analog — full elimination via 9-pt + bounce-back)

Same setup as video_boundary_modes.py: 1.0 × 0.5 cm tissue, dx=0.025, line stim
at left edge (uniform y), TTP06 EPI, forward-Euler diffusion.

Output: figures/video_stencil_mirror_combos.mp4
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
V_MIN, V_MAX = -90.0, 50.0
FPS = 20

# (label, stencil, boundary_mode, short_title)
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

OUT = Path("/home/norepinephrine/Documents/Heart-Conduction/Research/Active/"
           "boundary_conduction_speedup/figures/video_stencil_mirror_combos.mp4")


def run_one(label, stencil, boundary_mode):
    print(f"[{label}] stencil={stencil}  bc={boundary_mode}", flush=True)
    grid = StructuredGrid.create_rectangle(LX, LY, NX, NY)
    fdm = FDMDiscretization(
        grid, D=0.001, chi=1.0, Cm=1.0,
        stencil=stencil, boundary_mode=boundary_mode,
    )
    proto = StimulusProtocol()
    proto.add_stimulus(
        region=lambda x, y: x < 0.05,
        start_time=0.0, duration=2.0, amplitude=-52.0,
    )
    sim = MonodomainSimulation(
        spatial=fdm, ionic_model='ttp06', stimulus=proto,
        dt=DT, splitting='strang', ionic_solver='rush_larsen',
        diffusion_solver='forward_euler', cell_type='EPI',
    )
    times, V_hist = sim.run_to_array(t_end=T_END, save_every=SAVE_EVERY)
    V_field = V_hist.reshape(len(times), NX, NY)
    print(f"        done. V ∈ [{V_field.min():.2f}, {V_field.max():.2f}] mV  "
          f"frames={len(times)}", flush=True)
    return times, V_field


# ---------- run all 5 ----------
results = {}
for label, stencil, bm, _ in COMBOS:
    results[label] = run_one(label, stencil, bm)
times = results[COMBOS[0][0]][0]
n_frames = len(times)


# ---------- animate ----------
fig, axes = plt.subplots(1, 5, figsize=(25, 5), constrained_layout=True)
ims = []
for c, (label, _, _, title) in enumerate(COMBOS):
    V_field = results[label][1]
    im = axes[c].imshow(
        V_field[0].T, origin='lower',
        extent=[0, LX, 0, LY], aspect='equal',
        cmap='RdBu_r', vmin=V_MIN, vmax=V_MAX,
    )
    axes[c].set_title(title, fontsize=10)
    axes[c].set_xlabel("x (cm)")
    if c == 0:
        axes[c].set_ylabel("y (cm)")
    plt.colorbar(im, ax=axes[c], label="V (mV)", shrink=0.7)
    ims.append(im)

suptitle = fig.suptitle(f"t = {times[0]:6.2f} ms", fontsize=14)


def update(frame):
    for c, (label, _, _, _) in enumerate(COMBOS):
        ims[c].set_data(results[label][1][frame].T)
    suptitle.set_text(f"t = {times[frame]:6.2f} ms")
    return ims + [suptitle]


anim = animation.FuncAnimation(
    fig, update, frames=n_frames, interval=1000 / FPS, blit=False,
)

OUT.parent.mkdir(parents=True, exist_ok=True)
print(f"\n[video] writing {OUT}", flush=True)
print(f"        {n_frames} frames @ {FPS} fps  ~ {n_frames/FPS:.1f} s", flush=True)

writer = animation.FFMpegWriter(
    fps=FPS, codec='libx264', bitrate=4000,
    extra_args=['-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '20'],
)
anim.save(str(OUT), writer=writer, dpi=120)
print(f"[video] saved {OUT}")
print(f"        size: {OUT.stat().st_size / 1024:.0f} KB")


# ---------- numerical summary ----------
print("\n=== peak V (max over field, max over time) per case ===")
for label, _, _, _ in COMBOS:
    V_field = results[label][1]
    print(f"  {label:<30}  V_max = {V_field.max():+6.2f} mV   "
          f"V_min = {V_field.min():+6.2f} mV")

# Mid-column boundary-vs-center comparison at peak frame
i_mid = NX // 2
print(f"\n=== mid-column (x = {i_mid*DX:.2f} cm) max|V[top] - V[ctr]| over time ===")
for label, _, _, _ in COMBOS:
    V_field = results[label][1]
    V_top = V_field[:, i_mid, 0]
    V_ctr = V_field[:, i_mid, NY // 2]
    max_dev = float(np.abs(V_top - V_ctr).max())
    print(f"  {label:<30}  max|top - ctr| = {max_dev:.4e} mV")
