"""Clean simulation-only video — no labels, no annotations, no panels.
Single full-frame V(x,y,t) heatmap of any named storage-tank config.

Usage:  python render_camel_to_crescent_clean.py [config_name]
        config_name defaults to "baseline".

Output: simulation/outputs/video/<config_name>_clean.mp4
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


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    if name not in configs.REGISTRY:
        raise SystemExit(f"unknown config {name!r}. available: {sorted(configs.REGISTRY)}")
    base = configs.REGISTRY[name]
    cfg = configs.make(
        {
            "name": f"video_clean_{name}",
            "sim": {"steps": 4000, "record_history": True, "snap_every": SNAP_EVERY},
        },
        base=base,
    )
    inlet, outlet = configs.resolve_geometry(cfg["geometry"])
    rule, pipes, bc, sim, geom = (
        cfg["rule"], cfg["pipes"], cfg["boundary"], cfg["sim"], cfg["geometry"]
    )

    print(f"[sim] running {name} for {sim['steps']} steps "
          f"(rule={rule['type']}, dir={pipes['directionality']}, "
          f"bc={bc['type']}, max_pump={rule['max_pump']}, threshold={rule['threshold']})",
          flush=True)
    out = tanks_vec.run(
        Nx=geom["Nx"], Ny=geom["Ny"],
        mode=rule["type"], steps=sim["steps"],
        inlet_cells=inlet, outlet_cells=outlet,
        threshold=rule["threshold"], max_volume=rule["max_volume"],
        max_pump=rule["max_pump"], gradient_k=rule["gradient_k"],
        directionality=pipes["directionality"], boundary=bc["type"],
        damping_cap=rule["damping_cap"],
        record_history=True, snap_every=sim["snap_every"],
    )
    snaps = out["snaps"]
    v_max = rule["max_volume"]
    print(f"[sim] {len(snaps)} snapshots", flush=True)

    out_dir = Path(__file__).parent / "outputs" / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{name}_clean.mp4"

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
    print(f"[render] done. {video_path}  ({len(snaps) / FPS:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
