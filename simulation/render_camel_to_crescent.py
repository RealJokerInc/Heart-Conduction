"""Render a high-quality video of John's storage-tank baseline (BASELINE config:
constant rule + one-way pipes + zero-pad BC + max_pump=10) showing the wavefront
propagating from inlet (left) to outlet (right). Per-column LAT shape transitions
from camel-toe (boundary leads, mid columns) to crescent (interior leads, far
columns).

Output: simulation/outputs/video/camel_to_crescent.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).parent))
import configs
import tanks_vec


VIDEO_W, VIDEO_H = 1920, 1080
FPS = 30
SNAP_EVERY = 5  # 4000/5 = 800 frames -> ~27s @ 30fps


def main():
    cfg = configs.make(
        {
            "name": "video_camel_to_crescent",
            "sim": {"steps": 4000, "record_history": True, "snap_every": SNAP_EVERY},
        },
        base=configs.BASELINE,
    )

    inlet_cells, outlet_cells = configs.resolve_geometry(cfg["geometry"])
    rule, pipes, bc, sim, geom = (
        cfg["rule"], cfg["pipes"], cfg["boundary"], cfg["sim"], cfg["geometry"]
    )

    print(f"[sim] running BASELINE for {sim['steps']} steps "
          f"(max_pump={rule['max_pump']}, threshold={rule['threshold']})", flush=True)
    out = tanks_vec.run(
        Nx=geom["Nx"], Ny=geom["Ny"],
        mode=rule["type"], steps=sim["steps"],
        inlet_cells=inlet_cells, outlet_cells=outlet_cells,
        threshold=rule["threshold"], max_volume=rule["max_volume"],
        max_pump=rule["max_pump"], gradient_k=rule["gradient_k"],
        directionality=pipes["directionality"], boundary=bc["type"],
        damping_cap=rule["damping_cap"],
        record_history=True, snap_every=sim["snap_every"],
    )
    snaps = out["snaps"]
    snap_steps = out["snap_steps"]
    iso = out["iso"]

    Ny, Nx = iso.shape
    lat = iso.astype(float)
    lat[lat < 0] = np.nan
    delta = 0.5 * (lat[0, :] + lat[-1, :]) - lat[Ny // 2, :]

    print(f"[sim] {len(snaps)} snapshots, iso_max = {int(iso.max())}, "
          f"Δ range [{np.nanmin(delta):+.1f}, {np.nanmax(delta):+.1f}]", flush=True)

    out_dir = Path(__file__).parent / "outputs" / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "camel_to_crescent.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, FPS, (VIDEO_W, VIDEO_H))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {video_path}")

    print(f"[render] {len(snaps)} frames @ {VIDEO_W}x{VIDEO_H} {FPS}fps -> {video_path}",
          flush=True)
    n = len(snaps)
    for i, (V_t, t_i) in enumerate(zip(snaps, snap_steps)):
        if i % 50 == 0 or i == n - 1:
            print(f"[render] frame {i+1}/{n}", flush=True)
        img_bgr = _render_frame(
            V_t, int(t_i), delta, total_steps=sim["steps"],
            threshold=rule["threshold"], v_max=rule["max_volume"],
        )
        writer.write(img_bgr)

    writer.release()
    print(f"[render] done. {video_path}  ({n / FPS:.1f}s)", flush=True)


def _render_frame(V, t, delta, total_steps, threshold, v_max):
    fig = Figure(figsize=(VIDEO_W / 100.0, VIDEO_H / 100.0), dpi=100, facecolor="white")
    gs = fig.add_gridspec(
        2, 1, height_ratios=[2.4, 1], hspace=0.32,
        left=0.06, right=0.96, top=0.93, bottom=0.08,
    )
    ax_V = fig.add_subplot(gs[0])
    ax_d = fig.add_subplot(gs[1])

    Ny, Nx = V.shape

    im = ax_V.imshow(
        V, origin="upper", cmap="inferno",
        vmin=0, vmax=v_max, aspect="equal", interpolation="bilinear",
    )
    if (V > threshold).any():
        ax_V.contour(V, levels=[threshold], colors="cyan", linewidths=2.0, alpha=0.95)
    ax_V.set_title(
        f"John's storage tanks  —  baseline (constant rule, one-way, zero-pad)  "
        f"—  step {t} / {total_steps}",
        fontsize=17, pad=12,
    )
    ax_V.set_xlabel("x  (column)", fontsize=13)
    ax_V.set_ylabel("y  (row)", fontsize=13)
    ax_V.tick_params(labelsize=11)
    cbar = fig.colorbar(im, ax=ax_V, shrink=0.85, pad=0.015)
    cbar.set_label(f"Volume V    (θ = {threshold:.0f})", fontsize=12)
    cbar.ax.axhline(threshold, color="cyan", lw=1.6)
    cbar.ax.tick_params(labelsize=10)

    xs = np.arange(Nx)
    valid = ~np.isnan(delta)
    fired_now = V > threshold
    leading_x = np.where(fired_now.any(axis=0))[0]
    x_now = int(leading_x.max()) if len(leading_x) else -1

    seen = valid & (xs <= x_now)
    future = valid & (xs > x_now)

    if seen.any():
        ax_d.fill_between(
            xs[seen], delta[seen], 0,
            where=delta[seen] < 0, color="#3066be", alpha=0.6,
            label="camel toe (boundary leads)",
        )
        ax_d.fill_between(
            xs[seen], delta[seen], 0,
            where=delta[seen] > 0, color="#d23a3a", alpha=0.6,
            label="crescent (interior leads)",
        )
    if future.any():
        ax_d.plot(
            xs[future], delta[future],
            color="gray", lw=1.0, alpha=0.45, ls="--", label="not yet reached",
        )
    if seen.any():
        ax_d.plot(xs[seen], delta[seen], color="black", lw=2.0, zorder=10)
    ax_d.axhline(0, color="black", lw=0.7)
    if x_now >= 0:
        ax_d.axvline(x_now, color="cyan", lw=2.0, alpha=0.9)

    ax_d.set_xlim(0, Nx - 1)
    ymin = float(np.nanmin(delta)) if valid.any() else -1.0
    ymax = float(np.nanmax(delta)) if valid.any() else 1.0
    pad = max(2.0, 0.08 * (ymax - ymin))
    ax_d.set_ylim(ymin - pad, ymax + pad)
    ax_d.set_xlabel("x  (column)", fontsize=13)
    ax_d.set_ylabel("Δ  =  ½(top+bot) − mid   LAT", fontsize=12)
    ax_d.set_title(
        "Per-column wavefront shape   (negative = camel toe / boundary speedup)",
        fontsize=13, pad=6,
    )
    ax_d.grid(alpha=0.3)
    ax_d.tick_params(labelsize=11)
    ax_d.legend(loc="lower right", fontsize=10, ncol=3, framealpha=0.88)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    if bgr.shape[1] != VIDEO_W or bgr.shape[0] != VIDEO_H:
        bgr = cv2.resize(bgr, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_AREA)
    return bgr


if __name__ == "__main__":
    main()
