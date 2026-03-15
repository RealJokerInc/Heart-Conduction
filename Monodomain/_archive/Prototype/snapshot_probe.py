"""
CLI helper that records voltage snapshots at evenly spaced times so we can
verify that the depolarization front actually progresses across the mesh.
This avoids plotting entirely and prints representative samples per column.
"""

from __future__ import annotations

import numpy as np

from mesh_setup import create_left_to_right_mesh
from plot_simulation import run_simulation


def probe_snapshots(
    nx: int = 80,
    ny: int = 40,
    dt: float = 0.02,
    t_stop: float = 20.0,
    output_stride: int = 10,
    sample_columns: tuple[int, ...] = (0, 5, 10, 20, 40, 60, 79),
):
    mesh = create_left_to_right_mesh(nx=nx, ny=ny)
    stim_fn = mesh.periodic_left_edge_stimulus()
    params = dict(
        C_m=1.0,
        k=8.0,
        a=0.1,
        epsilon0=0.01,
        mu1=0.2,
        mu2=0.3,
    )
    frames, times = run_simulation(
        mesh,
        params,
        dt=dt,
        t_stop=t_stop,
        stim_fn=stim_fn,
        output_stride=output_stride,
    )
    mid = mesh.ny // 2
    print(f"Collected {len(frames)} frames; sampling row {mid}")
    for t, frame in zip(times, frames):
        values = [frame[mid, col] for col in sample_columns]
        formatted = ", ".join(f"{val:6.3f}" for val in values)
        print(f"t={t:5.2f} ms -> {formatted}")


if __name__ == "__main__":
    probe_snapshots()
