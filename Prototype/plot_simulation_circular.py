
"""
Visualization helper dedicated to the circular-infarct mesh.
"""

from __future__ import annotations

import argparse
import numpy as np

from version2 import step_relaxed_monodomain, default_params
from mesh_setup import TissueMesh, create_spiral_mesh


def run_simulation(
    mesh: TissueMesh,
    params: dict,
    dt: float = 0.02,
    t_stop: float = 60.0,
    stim_fn=None,
    output_stride: int = 5,
):
    if stim_fn is None:
        stim_fn = mesh.periodic_left_edge_stimulus()

    V, w = mesh.empty_state()
    frames = []
    times = []

    n_steps = int(np.ceil(t_stop / dt))
    for n in range(n_steps):
        t = n * dt
        I_stim = stim_fn(t)
        V, w = step_relaxed_monodomain(V, w, mesh, params, I_stim, dt)

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


def plot_geometry(mesh: TissueMesh, ax):
    extent = mesh.extent()
    geom = ax.imshow(
        mesh.epsilon_tissue,
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title("Geometry / fiber field")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig = ax.figure
    fig.colorbar(geom, ax=ax, label="epsilon_tissue")

    skip_x = max(1, mesh.nx // 20)
    skip_y = max(1, mesh.ny // 20)
    y_coords = np.linspace(0.0, mesh.Ly, mesh.ny)[::skip_y]
    x_coords = np.linspace(0.0, mesh.Lx, mesh.nx)[::skip_x]
    X, Y = np.meshgrid(x_coords, y_coords)
    theta_sample = mesh.theta[::skip_y, ::skip_x]
    U = np.cos(theta_sample)
    V = np.sin(theta_sample)
    ax.quiver(X, Y, U, V, color="red", scale=20, width=0.003)


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
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if len(frames) == 0:
        raise ValueError("No frames were recorded; run the simulation first.")

    extent = mesh.extent()
    vmin, vmax = _color_limits(frames, percentile_clip)

    fig, (ax_field, ax_geom) = plt.subplots(1, 2, figsize=(12, 5))
    im = ax_field.imshow(
        frames[0],
        origin="lower",
        extent=extent,
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
    )
    ax_field.set_xlabel("x (mm)")
    ax_field.set_ylabel("y (mm)")
    ax_field.set_title("Voltage field")
    fig.colorbar(im, ax=ax_field, label="Voltage (mV)")

    time_text = ax_field.text(
        0.02,
        0.95,
        f"t = {times[0]:.2f} ms",
        transform=ax_field.transAxes,
        color="white",
        fontsize="medium",
        bbox=dict(facecolor="black", alpha=0.3, boxstyle="round,pad=0.2"),
    )

    plot_geometry(mesh, ax_geom)

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
    import matplotlib.pyplot as plt

    if len(frames) == 0:
        raise ValueError("No frames were recorded; run the simulation first.")

    if frame_indices is None:
        frame_indices = np.linspace(0, len(frames) - 1, 6, dtype=int).tolist()

    vmin, vmax = _color_limits(frames, percentile_clip)
    extent = mesh.extent()

    ncols = min(3, len(frame_indices))
    nrows = int(np.ceil(len(frame_indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
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
        ax.set_aspect("equal")

    for ax in axes[len(frame_indices) :]:
        ax.axis("off")

    fig.colorbar(im, ax=axes.tolist(), label="Voltage (mV)")
    plt.tight_layout()

    fig_geom, ax_geom = plt.subplots(figsize=(5, 5))
    plot_geometry(mesh, ax_geom)
    fig_geom.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Circular infarct visualization.")
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
        "--infarct-radius",
        type=float,
        default=3.0,
        help="Radius of the circular infarct (mm).",
    )
    parser.add_argument(
        "--domain-size",
        type=float,
        nargs=2,
        default=(20.0, 20.0),
        metavar=("LX", "LY"),
        help="Domain size (mm) along x and y.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        nargs=2,
        default=(180, 180),
        metavar=("NX", "NY"),
        help="Grid resolution along x/y.",
    )
    args = parser.parse_args()

    Lx, Ly = args.domain_size
    nx, ny = args.grid
    mesh = create_spiral_mesh(
        nx=nx,
        ny=ny,
        Lx=Lx,
        Ly=Ly,
        spiral_pitch=0.1,
    )

    pulse_count = args.stim_pulses
    if pulse_count is not None and pulse_count <= 0:
        pulse_count = None
    stim_fn = mesh.pulse_train_left_edge_stimulus(
        amplitude=args.stim_amplitude,
        pulse_ms=args.stim_pulse_ms,
        interval_ms=args.stim_interval_ms,
        n_pulses=pulse_count,
    )

    params = default_params()

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
