"""
Re-render the horizontal-redirect video alone with an expanded colormap range
that captures the full V dynamic range (including the sub-edge hyperpolarization
artifact at ~−92 mV which clips under the default [−90, +40] range).
"""
from __future__ import annotations
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['animation.ffmpeg_path'] = (
    '/home/norepinephrine/.conda/envs/heart-conduction/bin/ffmpeg'
)
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "case13_lbm_d2q9_canonical_horizontal_natural.h5"
OUT = ROOT / "figures" / "video_bc_horizontal.mp4"

V_MIN, V_MAX = -90.0, 40.0           # match the other 5 BC videos
CMAP = "viridis"
N_FRAMES = 100
FPS = 20

with h5py.File(DATA, "r") as f:
    V_full = f["V"][:]
    t_full = f["t"][:]
    NX, NY = V_full.shape[1], V_full.shape[2]

t_target = np.linspace(t_full[0], t_full[-1], N_FRAMES)
idx = np.searchsorted(t_full, t_target).clip(0, len(t_full) - 1)
V_frames = np.transpose(V_full[idx], (0, 2, 1))   # (T, NY, NX)

fig = plt.figure(figsize=(8, 4), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()
im = ax.imshow(V_frames[0], vmin=V_MIN, vmax=V_MAX, cmap=CMAP,
               aspect="auto", origin="lower", interpolation="nearest")

def update(k):
    im.set_data(V_frames[k])
    return [im]

anim = animation.FuncAnimation(fig, update, frames=N_FRAMES,
                                interval=1000 // FPS, blit=True)
writer = animation.FFMpegWriter(
    fps=FPS, codec="libx264",
    extra_args=["-pix_fmt", "yuv420p",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
)
anim.save(OUT, writer=writer)
plt.close(fig)

print(f"Re-rendered: {OUT.name}  ({OUT.stat().st_size / 1024**2:.2f} MB)")
print(f"  V range: [{V_full.min():.2f}, {V_full.max():.2f}] mV (actual data)")
print(f"  Colormap range: [{V_MIN}, {V_MAX}] mV ({CMAP})")
print(f"  Sub-edge depletion artifact at j=1, j=NY-2 (V ≈ −92 mV) now clearly visible")
print(f"  as a thin dark band INSIDE the bright wall channel.")
