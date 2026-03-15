"""
Lightweight diagnostic runner to verify that the monodomain solver produces a
traveling depolarization when only the left-most column receives periodic
stimulation. This avoids any matplotlib dependencies so it can be invoked
directly from the CLI to inspect propagation timings.
"""

from __future__ import annotations

import numpy as np

from mesh_setup import create_left_to_right_mesh
from version1 import step_monodomain


def run_diagnostic(
    nx: int = 80,
    ny: int = 20,
    dt: float = 0.02,
    t_stop: float = 20.0,
    threshold: float = 0.2,
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

    V, w = mesh.empty_state()
    columns_to_track = np.linspace(0, nx - 1, 8, dtype=int)
    crossing_times = {col: None for col in columns_to_track}

    n_steps = int(np.ceil(t_stop / dt))
    for n in range(n_steps):
        t = n * dt
        I_stim = stim_fn(t)
        V, w = step_monodomain(
            V, w, mesh.Dxx, mesh.Dxy, mesh.Dyy, mesh.dx, mesh.dy, dt, params, I_stim
        )

        for col in columns_to_track:
            if crossing_times[col] is None and np.max(V[:, col]) >= threshold:
                crossing_times[col] = t

    return crossing_times, V


def main():
    threshold = 0.2
    crossings, final_V = run_diagnostic(threshold=threshold)
    print("Voltage propagation diagnostic")
    print(
        "Threshold crossings (ms) for select columns "
        f"(threshold={threshold} mV):"
    )
    for col, t_cross in crossings.items():
        print(f"  column {col:02d}: {t_cross if t_cross is not None else 'never'}")
    print("\nFinal voltage profile along midline:")
    mid = final_V.shape[0] // 2
    print(final_V[mid, :])


if __name__ == "__main__":
    main()
