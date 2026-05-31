"""Clean single-panel videos of the connectivity ablation runs, matching
render_camel_to_crescent_clean.py exactly: full-frame V(x,y,t) heatmap,
INFERNO colormap, 1920x1080, 30 fps, no labels.

Renders three variants for direct visual comparison of the boundary shape
(use side by side in a player to see them aligned in time):

  baseline_uniform_clean.mp4  : moore8 uniform + threshold       (R1, full crescent)
  iso_thresh_clean.mp4         : moore8 isotropic 4:1 + threshold (R5, V-shape / inverse crescent)
  iso_no_thresh_clean.mp4      : moore8 isotropic 4:1, no thresh  (R6)

All three share gradient mode + one_way + zero_pad + line geometry, so the
only differences are the connectivity weighting and the threshold gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import configs
import tanks_vec


VIDEO_W, VIDEO_H = 1920, 1080
FPS = 30
SNAP_EVERY = 5
STEPS = 4000

# Each entry: (output_name, connectivity, threshold_gate, optional_steps)
# moore8_iso wave is ~10x faster, so use fewer steps so the video isn't all
# steady-state. Tweak per-run.
RUNS = [
    ("baseline_uniform_clean", "moore8",     True,  4000),
    ("iso_thresh_clean",        "moore8_iso", True,  4000),
    ("iso_no_thresh_clean",     "moore8_iso", False, 4000),
]


def render_one(name: str, connectivity: str, threshold_gate: bool, steps: int):
    base = configs.GRADIENT  # gradient mode + one_way + zero_pad + line
    geom, rule, pipes, bc = (
        base["geometry"], base["rule"], base["pipes"], base["boundary"]
    )
    inlet, outlet = configs.resolve_geometry(geom)
    print(f"\n[sim] {name}: connectivity={connectivity} threshold_gate={threshold_gate} "
          f"steps={steps}", flush=True)
    out = tanks_vec.run(
        Nx=geom["Nx"], Ny=geom["Ny"],
        mode=rule["type"], steps=steps,
        inlet_cells=inlet, outlet_cells=outlet,
        threshold=rule["threshold"], max_volume=rule["max_volume"],
        max_pump=rule["max_pump"], gradient_k=rule["gradient_k"],
        directionality=pipes["directionality"], boundary=bc["type"],
        damping_cap=rule["damping_cap"],
        record_history=True, snap_every=SNAP_EVERY,
        connectivity=connectivity,
        threshold_gate=threshold_gate,
    )
    snaps = out["snaps"]
    v_max = rule["max_volume"]
    print(f"[sim] {len(snaps)} snapshots, V_max_observed={snaps.max():.2f}", flush=True)

    out_dir = Path(__file__).parent / "outputs" / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{name}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, FPS, (VIDEO_W, VIDEO_H))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {video_path}")

    print(f"[render] {len(snaps)} frames @ {VIDEO_W}x{VIDEO_H} {FPS}fps -> {video_path}",
          flush=True)
    for i, V_t in enumerate(snaps):
        if i % 100 == 0 or i == len(snaps) - 1:
            print(f"[render] {i + 1}/{len(snaps)}", flush=True)
        norm = np.clip(V_t / v_max * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        frame = cv2.resize(colored, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_CUBIC)
        writer.write(frame)
    writer.release()
    size_mb = video_path.stat().st_size / (1024 * 1024)
    print(f"[render] done. {video_path}  ({len(snaps) / FPS:.1f}s, {size_mb:.1f} MB)",
          flush=True)


def main():
    for name, conn, gate, steps in RUNS:
        render_one(name, conn, gate, steps)


if __name__ == "__main__":
    main()
