"""
CLI probe to sample the circular infarct simulation at fixed positions so we
can verify the propagation direction numerically.
"""

from __future__ import annotations

import numpy as np

from plot_simulation_circular import run_simulation
from mesh_setup import create_circular_infarct_mesh
from version2 import default_params


def main():
    mesh = create_circular_infarct_mesh(nx=120, ny=120, infarct_radius=4.0)
    stim = mesh.pulse_train_left_edge_stimulus(amplitude=20.0, pulse_ms=1.0, interval_ms=8.0)
    frames, times = run_simulation(
        mesh,
        default_params(),
        dt=0.02,
        t_stop=20.0,
        stim_fn=stim,
        output_stride=5,
    )

    mid_y = mesh.ny // 2
    sample_cols = [5, 20, 40, 60, 90, 110]

    for idx, (t, frame) in enumerate(zip(times, frames)):
        vals = [frame[mid_y, col] for col in sample_cols]
        print(f"t={t:6.2f} ms: " + ", ".join(f"{v:6.3f}" for v in vals))
        if idx >= 12:
            break


if __name__ == "__main__":
    main()
