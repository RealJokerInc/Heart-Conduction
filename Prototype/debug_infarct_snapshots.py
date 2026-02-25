"""
Diagnostic script to visualise infarct simulation snapshots every 10 ms.
"""

from __future__ import annotations

import numpy as np

from mesh_setup import create_flow_around_infarct_mesh
from version3 import step_relaxed_monodomain, default_params


def run_simulation(mesh, params, dt=0.02, t_stop=60.0, output_stride=5):
    stim_fn = mesh.periodic_left_edge_stimulus(period_ms=8.0, pulse_ms=1.0)
    V_rest = params.get("V_rest", 0.0)
    V, w = mesh.empty_state(V_rest, V_rest)

    frames = []
    times = []
    n_steps = int(t_stop / dt)
    for step in range(n_steps + 1):
        t = step * dt
        if step % output_stride == 0:
            frames.append(V.copy())
            times.append(t)
        I_stim = stim_fn(t)
        V, w = step_relaxed_monodomain(V, w, mesh, params, I_stim, dt)
    return np.stack(frames), np.asarray(times)


def print_snapshots(frames, times, mesh, interval_ms=10.0):
    target_times = np.arange(0.0, times[-1] + 1e-6, interval_ms)
    for target in target_times:
        idx = np.argmin(np.abs(times - target))
        frame = frames[idx]
        print(f"\nt = {times[idx]:6.2f} ms")
        np.set_printoptions(precision=3, suppress=True, linewidth=200)
        print(frame)


def main():
    mesh = create_flow_around_infarct_mesh()
    params = default_params()
    frames, times = run_simulation(mesh, params, dt=0.02, t_stop=60.0, output_stride=5)
    print_snapshots(frames, times, mesh, interval_ms=10.0)


if __name__ == "__main__":
    main()
