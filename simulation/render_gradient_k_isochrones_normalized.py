"""Same 2x3 gradient_k sweep as gradient_k_per_column_lat_nx80.png but
rendered as ISOCHRONES (matching pump_speed_isochrones format) with
NORMALIZED contour levels — chosen at evenly-spaced x-positions of the
wavefront's mean. Factors out wave-slowing.

GRADIENT mode (Fickian k·(V_src−V_dst), one-way + zero-pad).

Output: simulation/outputs/images/gradient_k_isochrones_normalized.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import configs
import tanks_vec
from render_norm_helpers import x_evenly_spaced_levels


KS = (0.16, 0.12, 0.08, 0.04, 0.02, 0.01)
STEPS = 20000
NX = 320
NUM_LEVELS = 10


def main():
    base = configs.make({"geometry": {"Nx": NX}}, base=configs.GRADIENT)
    geom, rule, pipes, bc = (
        base["geometry"], base["rule"], base["pipes"], base["boundary"]
    )
    inlet, outlet = configs.resolve_geometry(geom)
    Ny, Nx = geom["Ny"], geom["Nx"]

    print(f"[sim] running {len(KS)} gradient_k sims, Nx={Nx}, {STEPS} steps",
          flush=True)
    isos, rmosts = [], []
    for k in KS:
        print(f"[sim]   gradient_k={k}", flush=True)
        out = tanks_vec.run(
            Nx=Nx, Ny=Ny, mode=rule["type"], steps=STEPS,
            inlet_cells=inlet, outlet_cells=outlet,
            threshold=rule["threshold"], max_volume=rule["max_volume"],
            max_pump=rule["max_pump"], gradient_k=k,
            directionality=pipes["directionality"], boundary=bc["type"],
            damping_cap=rule["damping_cap"],
            record_history=False,
        )
        iso = out["iso"]
        fired_cols = (iso >= 0).any(axis=0)
        rmost = int(np.where(fired_cols)[0].max()) if fired_cols.any() else -1
        isos.append(iso)
        rmosts.append(rmost)

    fig, axes = plt.subplots(2, 3, figsize=(20, 5.5), constrained_layout=True)
    cmap = plt.cm.viridis
    contour_norm = Normalize(vmin=1, vmax=NUM_LEVELS)
    sm = plt.cm.ScalarMappable(norm=contour_norm, cmap=cmap)
    sm.set_array([])

    for i, (k, iso, rmost) in enumerate(zip(KS, isos, rmosts)):
        ax = axes[i // 3, i % 3]
        levels = x_evenly_spaced_levels(iso, NUM_LEVELS)
        iso_plot = iso.astype(float)
        iso_plot[iso_plot < 0] = np.nan
        unfired = (iso < 0).astype(float)
        ax.contourf(unfired, levels=[0.5, 1.5], colors=["#e8e8e8"], zorder=0)
        if len(levels) > 0:
            level_colors = [cmap(contour_norm(j + 1)) for j in range(len(levels))]
            for lvl, col in zip(levels, level_colors):
                ax.contour(iso_plot, levels=[lvl], colors=[col],
                           linewidths=1.3, zorder=2)
        ax.set_xlim(0, Nx - 1)
        ax.set_ylim(Ny - 1, 0)
        ax.set_aspect("equal")
        ax.set_title(f"k = {k:.2f}    (reached col {rmost})", fontsize=11)
        if i // 3 == 1:
            ax.set_xlabel("x (column)", fontsize=10)
        if i % 3 == 0:
            ax.set_ylabel("y (row)", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.18, zorder=1)

    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(),
                        orientation="vertical", shrink=0.85, pad=0.012,
                        label=f"x-reach level (1=earliest, {NUM_LEVELS}=latest)")
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        "NORMALIZED gradient_k isochrones — GRADIENT (Fickian) rule\n"
        "(k·(V_src − V_dst) + one-way + zero-pad)\n"
        f"Nx={Nx}, Ny={Ny}, {STEPS} steps.    Contours at evenly-spaced "
        "x-reach (NOT step times) — factors out wave-slowing.",
        fontsize=12,
    )

    out_dir = ROOT / "outputs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gradient_k_isochrones_normalized.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\n[plot] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
