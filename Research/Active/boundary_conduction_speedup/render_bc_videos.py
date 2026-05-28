"""
Render individual propagation videos for 6 boundary conditions.

Same grid (NX=41, NY=21), same V color range, no titles, no colorbar.
Just the raw colormap evolution per BC.

Cases:
  fm         — case3_fm_ttp06.h5                                  (PDE monodomain, face_mirror)
  fmi        — case4_fmi_ttp06.h5                                 (PDE monodomain, face_mirror_iso)
  hbb        — case10_lbm_d2q9_canonical_hbb_natural.h5           (LBM, halfway bounce-back)
  specular   — case9_lbm_d2q9_canonical_specular_natural.h5       (LBM, specular reflection)
  horizontal — case13_lbm_d2q9_canonical_horizontal_natural.h5    (LBM, horizontal redirect)
  equalmix   — case_weighted_a0.33_b0.33_g0.33_canonical.h5       (LBM, α=β=γ=1/3)

Output: 6 MP4 files in figures/, one per BC.
"""
from __future__ import annotations
from pathlib import Path
import subprocess
import shutil

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
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared color range across all BCs (original viridis [−90, +40] mV)
V_MIN, V_MAX = -90.0, 40.0
CMAP = "viridis"
N_FRAMES = 100   # 100 frames spanning [0, t_end]
FPS = 20         # 5 seconds total per video

CASES = [
    ("fm",         "case3_fm_ttp06.h5"),
    ("fmi",        "case4_fmi_ttp06.h5"),
    ("hbb",        "case10_lbm_d2q9_canonical_hbb_natural.h5"),
    ("specular",   "case9_lbm_d2q9_canonical_specular_natural.h5"),
    ("horizontal", "case13_lbm_d2q9_canonical_horizontal_natural.h5"),
    ("equalmix",   "case_weighted_a0.33_b0.33_g0.33_canonical.h5"),
]


def lat_crossing(V_t, t_arr, thresh=-40.0):
    above = V_t >= thresh
    if not above.any():
        return float("nan")
    idx = int(np.argmax(above))
    if idx == 0:
        return t_arr[0]
    v0, v1 = V_t[idx - 1], V_t[idx]
    t0, t1 = t_arr[idx - 1], t_arr[idx]
    if v1 == v0:
        return t1
    return t0 + (thresh - v0) * (t1 - t0) / (v1 - v0)


def render_case(name: str, fname: str) -> dict:
    with h5py.File(DATA_DIR / fname, "r") as f:
        V_full = f["V"][:]
        t_full = f["t"][:]
        NX, NY = V_full.shape[1], V_full.shape[2]

    # Resample to N_FRAMES evenly across [0, t_end]
    t_target = np.linspace(t_full[0], t_full[-1], N_FRAMES)
    # Index into V_full for nearest-time
    idx = np.searchsorted(t_full, t_target).clip(0, len(t_full) - 1)
    V_frames = V_full[idx]  # (N_FRAMES, NX, NY)

    # Display orientation: x horizontal (propagation), y vertical (walls top/bottom)
    # imshow expects (rows, cols) — we want rows=y, cols=x
    # so transpose each frame to (NY, NX)
    V_frames_disp = np.transpose(V_frames, (0, 2, 1))  # (N_FRAMES, NY, NX)

    fig = plt.figure(figsize=(8, 4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])  # fill entire figure
    ax.set_axis_off()

    im = ax.imshow(
        V_frames_disp[0],
        vmin=V_MIN, vmax=V_MAX,
        cmap=CMAP,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    def update(frame_idx):
        im.set_data(V_frames_disp[frame_idx])
        return [im]

    anim = animation.FuncAnimation(
        fig, update, frames=N_FRAMES, interval=1000 // FPS, blit=True
    )

    out_path = OUT_DIR / f"video_bc_{name}.mp4"
    # Save via ffmpeg; pad to even dimensions for yuv420p compatibility
    writer = animation.FFMpegWriter(
        fps=FPS,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
    )
    anim.save(out_path, writer=writer)
    plt.close(fig)

    # Compute LAT profile at col 38 (or near end) for verification
    c_check = min(38, NX - 3)
    j_bdry, j_mid = 0, NY // 2
    lat_b = lat_crossing(V_full[:, c_check, j_bdry], t_full)
    lat_c = lat_crossing(V_full[:, c_check, j_mid],  t_full)
    diff_us = (lat_b - lat_c) * 1000.0
    v_max = float(V_full.max())
    v_min = float(V_full.min())

    return dict(
        name=name, out=str(out_path.name), shape=V_full.shape,
        lat_diff_us=diff_us, v_min=v_min, v_max=v_max,
        size_mb=out_path.stat().st_size / 1024**2,
    )


def main():
    print(f"Rendering {len(CASES)} BC propagation videos to {OUT_DIR}/")
    print(f"Grid: 41×21, V range [{V_MIN}, {V_MAX}] mV, cmap={CMAP}, {N_FRAMES} frames @ {FPS} fps")
    print()
    results = []
    for name, fname in CASES:
        print(f"  rendering {name}...", end=" ", flush=True)
        r = render_case(name, fname)
        results.append(r)
        print(f"done ({r['size_mb']:.2f} MB)")

    print()
    print("Expectation check — bdry−ctr LAT at col 38 (positive = forward crescent, negative = inverse):")
    print(f'  {"BC":<12} {"file":<48} {"LAT diff (µs)":>14} {"verdict":>30}')
    print("  " + "-" * 110)
    for r in results:
        # Expected pattern
        name = r["name"]
        d = r["lat_diff_us"]
        expected = {
            "fm":         ("forward crescent (boundary slows)",      d > 100),
            "fmi":        ("flat / zero deficit",                     abs(d) < 10),
            "hbb":        ("mild forward crescent",                   0 < d < 200),
            "specular":   ("near zero bias",                          abs(d) < 50),
            "horizontal": ("strong inverse crescent",                 d < -500),
            "equalmix":   ("moderate inverse",                        -500 < d < 0),
        }[name]
        verdict = "✓" if expected[1] else "✗"
        print(f"  {name:<12} {r['out']:<48} {d:+14.1f} {verdict} {expected[0]}")


if __name__ == "__main__":
    main()
