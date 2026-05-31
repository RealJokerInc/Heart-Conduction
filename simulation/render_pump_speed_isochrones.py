"""Quantitative wavefront-evolution figure: 2x3 grid of subplots, each showing
threshold-contour isochrones at evenly-spaced time levels (in absolute step
number, capped by the slowest sim) for one max_pump value of John's BASELINE
rule (constant + one-way + zero-pad + damping).

Top row:    max_pump = 30, 20, 15
Bottom row: max_pump = 10,  5,  2

Output: simulation/outputs/images/pump_speed_isochrones.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).parent))
import configs
import tanks_vec


PUMPS = (30.0, 20.0, 15.0, 10.0, 5.0, 2.0)  # 2x3 layout, top→bottom descending
STEPS = 20000
NX = 320
NUM_LEVELS = 10  # evenly-spaced contour times


def main():
    base = configs.make({"geometry": {"Nx": NX}}, base=configs.BASELINE)
    geom = base["geometry"]
    rule = base["rule"]
    pipes = base["pipes"]
    bc = base["boundary"]
    inlet, outlet = configs.resolve_geometry(geom)
    Ny, Nx = geom["Ny"], geom["Nx"]

    print(f"[sim] running 6 sims (max_pump = {list(PUMPS)}), Nx={Nx}, {STEPS} steps",
          flush=True)
    isos = []
    rightmost_cols = []
    for mp in PUMPS:
        print(f"[sim]   max_pump={mp}", flush=True)
        out = tanks_vec.run(
            Nx=Nx, Ny=Ny,
            mode=rule["type"], steps=STEPS,
            inlet_cells=inlet, outlet_cells=outlet,
            threshold=rule["threshold"], max_volume=rule["max_volume"],
            max_pump=mp, gradient_k=rule["gradient_k"],
            directionality=pipes["directionality"], boundary=bc["type"],
            damping_cap=rule["damping_cap"],
            record_history=False,
        )
        iso = out["iso"]
        fired_cols = (iso >= 0).any(axis=0)
        rmost = int(np.where(fired_cols)[0].max()) if fired_cols.any() else -1
        isos.append(iso)
        rightmost_cols.append(rmost)
        print(f"[sim]    rightmost_col={rmost}", flush=True)
    print("[sim] all done", flush=True)

    levels = np.linspace(STEPS / NUM_LEVELS, STEPS, NUM_LEVELS)
    print(f"[plot] contour levels (step number): {levels.astype(int).tolist()}",
          flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(20, 5.5), constrained_layout=True)
    cmap = plt.cm.viridis
    norm = Normalize(vmin=levels.min(), vmax=levels.max())

    for i, (mp, iso, rmost) in enumerate(zip(PUMPS, isos, rightmost_cols)):
        ax = axes[i // 3, i % 3]
        iso_plot = iso.astype(float)
        iso_plot[iso_plot < 0] = np.nan
        # Light-gray shading for the never-fired region
        unfired = (iso < 0).astype(float)
        ax.contourf(unfired, levels=[0.5, 1.5], colors=["#e8e8e8"], zorder=0)
        # Wavefront isochrone contours at evenly-spaced absolute times
        ax.contour(iso_plot, levels=levels, cmap=cmap, norm=norm,
                   linewidths=1.3, zorder=2)
        ax.set_xlim(0, Nx - 1)
        ax.set_ylim(Ny - 1, 0)  # y=0 at top to match imshow origin='upper'
        ax.set_aspect("equal")
        ax.set_title(f"max_pump = {mp:.0f}    "
                     f"(reached col {rmost})", fontsize=11)
        if i // 3 == 1:
            ax.set_xlabel("x  (column)", fontsize=10)
        if i % 3 == 0:
            ax.set_ylabel("y  (row)", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.18, zorder=1)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation="vertical",
                        shrink=0.85, pad=0.012, label="time step (LAT, evenly spaced)")
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Wavefront isochrones at evenly-spaced time levels   —   "
        "John's BASELINE rule (constant √(V−θ) + one-way + zero-pad + damping)   —   "
        f"Nx={Nx},  Ny={Ny},  {STEPS} steps",
        fontsize=12,
    )

    out_dir = Path(__file__).parent / "outputs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pump_speed_isochrones.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[plot] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
