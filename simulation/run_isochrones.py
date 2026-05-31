# -*- coding: utf-8 -*-
"""Vectorised version of run_isochrones.py.

Uses tanks_vec instead of the OOP tanks_channel_states. Mathematically
equivalent (verified by test_vec_matches.py: max|ΔV| ~ 5e-14, max|Δiso| = 0).
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import tanks_vec


def run_config(geom, mode, Nx, Ny, steps, threshold, gradient_k):
    if geom == "john":
        inlet_cells = [(14, 3), (14, 5), (14, 6), (14, 7)]
        outlet_cells = [(Nx - 1, y) for y in range(Ny)]
    elif geom == "line":
        inlet_cells = [(0, y) for y in range(Ny)]
        outlet_cells = [(Nx - 1, y) for y in range(Ny)]
    else:
        raise ValueError(geom)

    t0 = time.perf_counter()
    out = tanks_vec.run(
        Nx, Ny, mode, steps,
        inlet_cells=inlet_cells,
        outlet_cells=outlet_cells,
        threshold=threshold,
        gradient_k=gradient_k,
    )
    iso = out["iso"]
    elapsed = time.perf_counter() - t0
    print(f"  [{geom}/{mode}] {elapsed:.2f}s  filled={int((iso >= 0).sum())}/{Nx * Ny}  max_step={int(iso.max())}")
    return iso, inlet_cells, elapsed


def compute_metric(iso):
    Ny, Nx = iso.shape
    x_front = np.full(Ny, -1, dtype=np.int32)
    for y in range(Ny):
        reached = np.where(iso[y] >= 0)[0]
        if len(reached) > 0:
            x_front[y] = reached.max()
    edge = 0.5 * (float(x_front[0]) + float(x_front[-1]))
    mid = float(x_front[Ny // 2])
    return edge, mid, edge - mid


def plot_isochrones(isos, inlets, out_path, total_steps):
    geoms = ("john", "line")
    modes = ("constant", "gradient")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for row, geom in enumerate(geoms):
        row_max = max(
            np.where(isos[(geom, m)] >= 0, isos[(geom, m)], 0).max() for m in modes
        )
        row_max = max(int(row_max), 1)
        for col, mode in enumerate(modes):
            ax = axes[row, col]
            iso = isos[(geom, mode)].astype(float)
            iso_plot = np.where(iso >= 0, iso, np.nan)
            im = ax.imshow(iso_plot, origin="upper", cmap="plasma",
                           aspect="auto", vmin=0, vmax=row_max)
            levels = np.arange(1, row_max, max(row_max // 12, 1))
            ax.contour(iso_plot, levels=levels, colors="white",
                       linewidths=0.6, alpha=0.7)
            if inlets.get((geom, mode)):
                ixs = [c[0] for c in inlets[(geom, mode)]]
                iys = [c[1] for c in inlets[(geom, mode)]]
                ax.scatter(ixs, iys, s=20, marker="*", c="cyan",
                           edgecolors="black", linewidths=0.5, zorder=5)
            ax.set_title(f"{geom} geometry / {mode} mode", fontsize=10)
            ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel("y")
            cbar = fig.colorbar(im, ax=ax, shrink=0.85)
            cbar.set_label("step of first crossing")

    fig.suptitle(
        f"Isochrones after {total_steps} steps\n"
        "If boundary speedup is present, contours bow forward at y=0 and y=49",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def main():
    Nx, Ny = 80, 50
    steps = 2000
    threshold = 45.0
    gradient_k = 0.08

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    isos, inlets, metrics = {}, {}, []
    t_total = time.perf_counter()
    for geom in ("john", "line"):
        for mode in ("constant", "gradient"):
            print(f"=== {geom} / {mode} ===")
            iso, inl, elapsed = run_config(
                geom, mode, Nx, Ny, steps, threshold, gradient_k
            )
            isos[(geom, mode)] = iso
            inlets[(geom, mode)] = inl
            edge, mid, delta = compute_metric(iso)
            metrics.append((geom, mode, edge, mid, delta, elapsed))
    total_elapsed = time.perf_counter() - t_total

    np.savez(
        out_dir / "isochrones_vec.npz",
        **{f"{g}_{m}": isos[(g, m)] for g in ("john", "line") for m in ("constant", "gradient")},
    )
    plot_isochrones(isos, inlets, out_dir / "isochrones_vec.png", total_steps=steps)

    lines = [f"Total wall time: {total_elapsed:.2f}s ({steps} steps x {Nx}x{Ny} x 4 configs)"]
    for geom, mode, edge, mid, delta, elapsed in metrics:
        label = "edge LEADS" if delta > 0 else ("edge lags" if delta < 0 else "equal")
        lines.append(
            f"  {geom:5} / {mode:8}: edge_x={edge:5.1f}  mid_x={mid:5.1f}  "
            f"delta={delta:+5.1f}  [{label}]  ({elapsed:.2f}s)"
        )
    summary = "\n".join(lines) + "\n"
    (out_dir / "isochrone_summary_vec.txt").write_text(summary)
    print("\n=== Summary ===")
    print(summary)


if __name__ == "__main__":
    main()
