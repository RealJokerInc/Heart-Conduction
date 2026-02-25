"""
Visualization utilities for the 2-D monodomain conduction simulation.

This module wires together the solver in ``version1.py`` with the left-to-right
mesh defined in ``mesh_setup.py`` and offers both animation and static plotting
helpers. Run this file directly to generate plots without editing notebooks.
"""

from __future__ import annotations

import argparse
import numpy as np

from version1 import step_monodomain
from mesh_setup import (
    TissueMesh,
    create_left_to_right_mesh,
    create_circular_infarct_mesh,
)


def run_simulation(
    mesh: TissueMesh,
    params: dict,
    dt: float = 0.02,
    t_stop: float = 60.0,
    stim_fn=None,
    output_stride: int = 5,
):
    """
    Execute the explicit monodomain scheme and collect snapshots for plotting.
    """
    if stim_fn is None:
        stim_fn = mesh.periodic_left_edge_stimulus()

    V, w = mesh.empty_state()
    frames = []
    times = []

    n_steps = int(np.ceil(t_stop / dt))
    for n in range(n_steps):
        t = n * dt
        I_stim = stim_fn(t)
        V, w = step_monodomain(
            V, w, mesh.Dxx, mesh.Dxy, mesh.Dyy, mesh.dx, mesh.dy, dt, params, I_stim
        )

        if n % output_stride == 0:
            frames.append(V.copy())
            times.append(t)

    return np.array(frames), np.array(times)


def _color_limits(frames: np.ndarray, percentile_clip: tuple[float, float] | None):
    if percentile_clip is None:
        return float(np.min(frames)), float(np.max(frames))
    low, high = percentile_clip
    return (
        float(np.percentile(frames, low)),
        float(np.percentile(frames, high)),
    )


def animate_voltage(
    frames: np.ndarray,
    times: np.ndarray,
    mesh: TissueMesh,
    percentile_clip: tuple[float, float] | None = (1.0, 99.0),
    interval_ms: int = 40,
    repeat: bool = True,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Create an animation of the voltage field evolution using imshow().
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if len(frames) == 0:
        raise ValueError("No frames were recorded; run the simulation first.")

    extent = mesh.extent()
    vmin, vmax = _color_limits(frames, percentile_clip)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        frames[0],
        origin="lower",
        extent=extent,
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.colorbar(im, ax=ax, label="Voltage (mV)")

    time_text = ax.text(
        0.02,
        0.95,
        f"t = {times[0]:.2f} ms",
        transform=ax.transAxes,
        color="white",
        fontsize="medium",
        bbox=dict(facecolor="black", alpha=0.3, boxstyle="round,pad=0.2"),
    )

    def _update(idx):
        im.set_data(frames[idx])
        time_text.set_text(f"t = {times[idx]:.2f} ms")
        return im, time_text

    anim = FuncAnimation(
        fig,
        _update,
        frames=len(frames),
        interval=interval_ms,
        blit=True,
        repeat=repeat,
    )

    plt.tight_layout()

    if save_path:
        anim.save(save_path)

    if show:
        plt.show()

    return fig, anim


def plot_snapshots(
    frames: np.ndarray,
    times: np.ndarray,
    mesh: TissueMesh,
    frame_indices: list[int] | None = None,
    percentile_clip: tuple[float, float] | None = (1.0, 99.0),
):
    """
    Plot a static grid of selected snapshots for quick debugging.
    """
    import matplotlib.pyplot as plt

    if len(frames) == 0:
        raise ValueError("No frames were recorded; run the simulation first.")

    if frame_indices is None:
        frame_indices = np.linspace(0, len(frames) - 1, 6, dtype=int).tolist()

    vmin, vmax = _color_limits(frames, percentile_clip)
    extent = mesh.extent()

    ncols = min(3, len(frame_indices))
    nrows = int(np.ceil(len(frame_indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes, frame_indices):
        im = ax.imshow(
            frames[idx],
            origin="lower",
            extent=extent,
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"t = {times[idx]:.2f} ms")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

    for ax in axes[len(frame_indices) :]:
        ax.axis("off")

    fig.colorbar(im, ax=axes.tolist(), label="Voltage (mV)")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize the conduction model.")
    parser.add_argument("--dt", type=float, default=0.02, help="Time step in ms.")
    parser.add_argument(
        "--t-stop", type=float, default=80.0, help="Total simulated time in ms."
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Store every Nth step for visualization.",
    )
    parser.add_argument(
        "--mode",
        choices=("animate", "static", "both"),
        default="animate",
        help="Choose whether to animate, plot static panels, or both.",
    )
    parser.add_argument(
        "--percentile",
        nargs=2,
        type=float,
        default=(1.0, 99.0),
        metavar=("LOW", "HIGH"),
        help="Percentile clip for color limits (0 100 disables clipping).",
    )
    parser.add_argument(
        "--save-animation",
        type=str,
        help="Optional path to save the animation (requires ffmpeg or pillow).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build the plots without calling plt.show(). Useful for testing.",
    )
    parser.add_argument("--stim-amplitude", type=float, default=20.0, help="Pulse amplitude (mV).")
    parser.add_argument(
        "--stim-pulse-ms",
        type=float,
        default=1.0,
        help="Duration of each stimulus pulse (ms).",
    )
    parser.add_argument(
        "--stim-interval-ms",
        type=float,
        default=8.0,
        help="Spacing between pulses (ms).",
    )
    parser.add_argument(
        "--stim-pulses",
        type=int,
        default=5,
        help="Number of pulses in the train (<=0 for continuous periodic stimulation).",
    )
    parser.add_argument(
        "--mesh",
        choices=("left-right", "circular-infarct"),
        default="left-right",
        help="Choose between the baseline mesh or the circular infarct mesh.",
    )
    args = parser.parse_args()

    if args.mesh == "left-right":
        mesh = create_left_to_right_mesh()
    else:
        mesh = create_circular_infarct_mesh()
    pulse_count = args.stim_pulses
    if pulse_count is not None and pulse_count <= 0:
        pulse_count = None
    stim_fn = mesh.pulse_train_left_edge_stimulus(
        amplitude=args.stim_amplitude,
        pulse_ms=args.stim_pulse_ms,
        interval_ms=args.stim_interval_ms,
        n_pulses=pulse_count,
    )

    params = dict(
        C_m=1.0,
        k=8.0,
        a=0.1,
        epsilon0=0.01,
        mu1=0.2,
        mu2=0.3,
    )

    percentile_clip = None
    if args.percentile is not None:
        low, high = args.percentile
        if not (low == 0.0 and high == 100.0):
            percentile_clip = (low, high)

    frames, times = run_simulation(
        mesh,
        params,
        dt=args.dt,
        t_stop=args.t_stop,
        stim_fn=stim_fn,
        output_stride=args.stride,
    )

    if args.mode in ("static", "both"):
        plot_snapshots(frames, times, mesh, percentile_clip=percentile_clip)

    if args.mode in ("animate", "both"):
        animate_voltage(
            frames,
            times,
            mesh,
            percentile_clip=percentile_clip,
            save_path=args.save_animation,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
