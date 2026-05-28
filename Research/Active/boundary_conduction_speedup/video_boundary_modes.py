"""
Side-by-side video of V(x,y,t) for FOUR FDM boundary modes.
All four use the same uniform-y line stimulus, TTP06 EPI, grid, and
forward-Euler diffusion solver (so rest_pad — which currently lives only
in apply_diffusion — is on equal footing with the others).

Output: figures/boundary_modes_video.gif

Modes
-----
  face_mirror           ghost = V[i,0]                Neumann; energy-conserving wall.
  node_mirror_existing  ghost = V[i,1]                Neumann legacy V5.4; non-symmetric matrix.
  zero_pad              ghost = 0                     Dirichlet-to-zero; boundary swamps in.
  rest_pad              ghost = V_rest (-85.23 mV)    Dirichlet-to-rest; silent at rest.
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
NX = int(round(LX / DX)) + 1
NY = int(round(LY / DX)) + 1
DT = 0.02
T_END = 25.0
SAVE_EVERY = 0.25
V_REST_TTP06 = -85.23
MODES = (
    ('face_mirror',          0.0),
    ('node_mirror_existing', 0.0),
    ('zero_pad',             0.0),
    ('rest_pad',             V_REST_TTP06),
)
V_MIN, V_MAX = -90.0, 50.0
FPS = 20

OUT = Path("/home/norepinephrine/Documents/Heart-Conduction/Research/Active/"
           "boundary_conduction_speedup/figures/boundary_modes_video.mp4")


def run_one(mode: str, pad_value: float):
    label = f"{mode}" + (f" (pad={pad_value:g})" if pad_value != 0.0 else "")
    print(f"[{label}] running...", flush=True)
    grid = StructuredGrid.create_rectangle(LX, LY, NX, NY)
    fdm = FDMDiscretization(
        grid, D=0.001, chi=1.0, Cm=1.0,
        boundary_mode=mode, pad_value=pad_value,
    )

    proto = StimulusProtocol()
    # Stim at the most leftward column only (col 0 = leftmost wall).
    # DX/2 threshold catches x=0.0 but excludes x=DX (col 1).
    proto.add_stimulus(
        region=lambda x, y: x < DX / 2,
        start_time=0.0,
        duration=2.0,
        amplitude=-52.0,
    )

    sim = MonodomainSimulation(
        spatial=fdm,
        ionic_model='ttp06',
        stimulus=proto,
        dt=DT,
        splitting='strang',
        ionic_solver='rush_larsen',
        diffusion_solver='forward_euler',  # uses apply_diffusion (rest_pad-aware)
        cell_type='EPI',
    )

    times, V_hist = sim.run_to_array(t_end=T_END, save_every=SAVE_EVERY)
    V_field = V_hist.reshape(len(times), NX, NY)
    print(f"[{label}] done   V in [{V_field.min():.2f}, {V_field.max():.2f}] mV   "
          f"frames={len(times)}", flush=True)
    return times, V_field


# ---------- run all four ----------
results = {mode: run_one(mode, pad) for mode, pad in MODES}
times = results[MODES[0][0]][0]
n_frames = len(times)


# ---------- animate ----------
fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
ims = []
titles = {
    'face_mirror':          'face_mirror\n(ghost = V[i,0])',
    'node_mirror_existing': 'node_mirror_existing\n(ghost = V[i,1])',
    'zero_pad':             'zero_pad\n(ghost = 0)',
    'rest_pad':             f'rest_pad\n(ghost = {V_REST_TTP06} mV)',
}
for c, (mode, _) in enumerate(MODES):
    V_field = results[mode][1]
    im = axes[c].imshow(
        V_field[0].T, origin='lower',
        extent=[0, LX, 0, LY], aspect='equal',
        cmap='RdBu_r', vmin=V_MIN, vmax=V_MAX,
    )
    axes[c].set_title(titles[mode], fontsize=11)
    axes[c].set_xlabel("x (cm)")
    if c == 0:
        axes[c].set_ylabel("y (cm)")
    plt.colorbar(im, ax=axes[c], label="V (mV)", shrink=0.8)
    ims.append(im)

suptitle = fig.suptitle(f"t = {times[0]:6.2f} ms", fontsize=14)


def update(frame):
    for c, (mode, _) in enumerate(MODES):
        V_field = results[mode][1]
        ims[c].set_data(V_field[frame].T)
    suptitle.set_text(f"t = {times[frame]:6.2f} ms")
    return ims + [suptitle]


anim = animation.FuncAnimation(
    fig, update, frames=n_frames, interval=1000 / FPS, blit=False,
)

OUT.parent.mkdir(parents=True, exist_ok=True)
print(f"\n[video] writing {OUT}  ({n_frames} frames @ {FPS} fps "
      f"~ {n_frames/FPS:.1f} s)...", flush=True)
writer = animation.FFMpegWriter(
    fps=FPS,
    codec='libx264',
    bitrate=4000,
    extra_args=['-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '20'],
)
anim.save(str(OUT), writer=writer, dpi=120)
print(f"[video] saved {OUT}")
print(f"        size: {OUT.stat().st_size / 1024:.0f} KB")


# ---------- numerical summary ----------
print("\n=== peak V (max over field, max over time) ===")
for mode, _ in MODES:
    V_field = results[mode][1]
    print(f"  {mode:25s}  V_max = {V_field.max():+6.2f} mV   "
          f"V_min = {V_field.min():+6.2f} mV")
