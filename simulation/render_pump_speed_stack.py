"""Stacked comparison video: BASELINE (constant + one-way + zero-pad + damping)
at six max_pump values, simulations stacked vertically and synchronized in time.
Top row = highest pump speed, bottom = lowest.

Output: simulation/outputs/video/pump_speed_stack.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import configs
import tanks_vec


PUMPS = (30.0, 20.0, 15.0, 10.0, 5.0, 2.0)  # top → bottom (descending)
STEPS = 20000      # large upper bound; each strip freezes individually at its completion
SNAP_EVERY = 10    # 2001 snaps max per sim; final video length = slowest's end_idx
NX = 320           # 4x longer tissue (was 80) so the slow-pump wave has room to develop
GAP_NATIVE = 4     # native rows of black between panels
SCALE = 6          # final pixels per native cell
FPS = 30


def main():
    base = configs.make({"geometry": {"Nx": NX}}, base=configs.BASELINE)
    geom = base["geometry"]
    rule = base["rule"]
    pipes = base["pipes"]
    bc = base["boundary"]
    inlet, outlet = configs.resolve_geometry(geom)
    Ny, Nx = geom["Ny"], geom["Nx"]
    v_max = rule["max_volume"]

    # Run all sims; track each strip's individual completion frame so we can
    # freeze fast strips while slow strips keep playing.
    print(f"[sim] running {len(PUMPS)} sims, up to {STEPS} steps each "
          f"on Nx={Nx} tissue (max_pump = {list(PUMPS)})", flush=True)
    sims = []
    end_indices = []
    for mp in PUMPS:
        out = tanks_vec.run(
            Nx=Nx, Ny=Ny,
            mode=rule["type"], steps=STEPS,
            inlet_cells=inlet, outlet_cells=outlet,
            threshold=rule["threshold"], max_volume=rule["max_volume"],
            max_pump=mp, gradient_k=rule["gradient_k"],
            directionality=pipes["directionality"], boundary=bc["type"],
            damping_cap=rule["damping_cap"],
            record_history=True, snap_every=SNAP_EVERY,
        )
        snaps = out["snaps"]
        snap_steps = out["snap_steps"]
        iso = out["iso"]
        # Detect when the wave's leading edge stops advancing: find the rightmost
        # column the wave ever reaches, then take the latest first-firing in that
        # column as the "end" step. Works for both fast pumps (reach the
        # penultimate column = 318) and slow pumps (stall out earlier when V
        # drops below θ before traversal completes).
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
        print(f"[sim]   max_pump={mp:>5.1f}  rightmost_col={rmost:>3d}  "
              f"end_step={end_step:>6d}  end_idx={end_idx:>4d}", flush=True)

    n_frames = max(end_indices) + 1
    print(f"[sim] all done. video length = {n_frames} frames "
          f"({n_frames / FPS:.1f}s, sets by slowest sim's completion)", flush=True)

    # Composite native dimensions (height = sims stacked + gaps)
    native_h = Ny * len(PUMPS) + GAP_NATIVE * (len(PUMPS) - 1)
    native_w = Nx
    out_h = native_h * SCALE
    out_w = native_w * SCALE

    # Pre-build a gap mask (True for gap rows in native coords)
    gap_mask = np.zeros((native_h, native_w), dtype=bool)
    offset = Ny
    for _ in range(len(PUMPS) - 1):
        gap_mask[offset:offset + GAP_NATIVE, :] = True
        offset += Ny + GAP_NATIVE

    # Set up writer
    out_dir = Path(__file__).parent / "outputs" / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "pump_speed_stack.mp4"
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
            eff_i = min(i, end_idx)  # freeze at this strip's completion
            composite[offset:offset + Ny, :] = sim_snaps[eff_i]
            offset += Ny + GAP_NATIVE
        norm = np.clip(composite / v_max * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        colored[gap_mask] = 0  # pure black for gap rows
        frame = cv2.resize(colored, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        writer.write(frame)

    writer.release()
    print(f"[render] done. {video_path}  ({n_frames / FPS:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
