"""Stacked comparison video: GRADIENT rule (Fickian k·(V_src−V_dst), one-way + zero-pad)
at six gradient_k values, simulations stacked vertically and synchronized in time.
Top row = highest k (fastest diffusion), bottom = lowest k.

Each strip freezes at its own completion (when the leading edge stops advancing —
either reaching the penultimate column for fast k, or hitting the threshold-gated
stall point further inside the domain for slow k).

Output: simulation/outputs/video/gradient_k_stack.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import configs
import tanks_vec


KS = (0.16, 0.12, 0.08, 0.04, 0.02, 0.01)  # top → bottom (descending k)
STEPS = 20000
SNAP_EVERY = 10
NX = 320
GAP_NATIVE = 4
SCALE = 6
FPS = 30


def main():
    base = configs.make({"geometry": {"Nx": NX}}, base=configs.GRADIENT)
    geom = base["geometry"]
    rule = base["rule"]
    pipes = base["pipes"]
    bc = base["boundary"]
    inlet, outlet = configs.resolve_geometry(geom)
    Ny, Nx = geom["Ny"], geom["Nx"]
    v_max = rule["max_volume"]

    print(f"[sim] running {len(KS)} sims, up to {STEPS} steps each "
          f"on Nx={Nx} tissue, gradient rule (k = {list(KS)})", flush=True)
    sims = []
    end_indices = []
    for k in KS:
        out = tanks_vec.run(
            Nx=Nx, Ny=Ny,
            mode="gradient", steps=STEPS,
            inlet_cells=inlet, outlet_cells=outlet,
            threshold=rule["threshold"], max_volume=rule["max_volume"],
            max_pump=rule["max_pump"], gradient_k=k,
            directionality=pipes["directionality"], boundary=bc["type"],
            damping_cap=rule["damping_cap"],
            record_history=True, snap_every=SNAP_EVERY,
        )
        snaps = out["snaps"]
        snap_steps = out["snap_steps"]
        iso = out["iso"]
        fired_cols = (iso >= 0).any(axis=0)
        if not fired_cols.any():
            end_step = -1
            end_idx = len(snaps) - 1
        else:
            rightmost_col = int(np.where(fired_cols)[0].max())
            col_iso = iso[:, rightmost_col]
            end_step = int(col_iso[col_iso >= 0].max())
            end_idx = int(np.searchsorted(snap_steps, end_step, side="left"))
            end_idx = min(end_idx, len(snaps) - 1)
        sims.append(snaps)
        end_indices.append(end_idx)
        rmost = int(np.where(fired_cols)[0].max()) if fired_cols.any() else -1
        print(f"[sim]   k={k:>5.3f}  rightmost_col={rmost:>3d}  "
              f"end_step={end_step:>6d}  end_idx={end_idx:>4d}", flush=True)

    n_frames = max(end_indices) + 1
    print(f"[sim] all done. video length = {n_frames} frames "
          f"({n_frames / FPS:.1f}s, set by slowest sim's completion)", flush=True)

    native_h = Ny * len(KS) + GAP_NATIVE * (len(KS) - 1)
    native_w = Nx
    out_h = native_h * SCALE
    out_w = native_w * SCALE

    gap_mask = np.zeros((native_h, native_w), dtype=bool)
    offset = Ny
    for _ in range(len(KS) - 1):
        gap_mask[offset:offset + GAP_NATIVE, :] = True
        offset += Ny + GAP_NATIVE

    out_dir = Path(__file__).parent / "outputs" / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "gradient_k_stack.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, FPS, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {video_path}")

    print(f"[render] {n_frames} frames @ {out_w}x{out_h} {FPS}fps -> {video_path}",
          flush=True)
    for i in range(n_frames):
        if i % 100 == 0 or i == n_frames - 1:
            print(f"[render] {i + 1}/{n_frames}", flush=True)
        composite = np.zeros((native_h, native_w), dtype=np.float64)
        offset = 0
        for sim_snaps, end_idx in zip(sims, end_indices):
            eff_i = min(i, end_idx)
            composite[offset:offset + Ny, :] = sim_snaps[eff_i]
            offset += Ny + GAP_NATIVE
        norm = np.clip(composite / v_max * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        colored[gap_mask] = 0
        frame = cv2.resize(colored, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        writer.write(frame)

    writer.release()
    print(f"[render] done. {video_path}  ({n_frames / FPS:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
