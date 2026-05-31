"""Per-column LAT-deviation figure for the gradient_k sweep, NORMALIZED:
divide each column's dev = (mean_iso − iso) by that column's per-column
traversal time Δmean_x. y-axis becomes "fractional lag per step", so curves
at different x should overlay if the per-step deficit is constant.

Same 2x3 layout as gradient_k_per_column_lat_nx80.png.
Top row: k = 0.16, 0.12, 0.08. Bottom: 0.04, 0.02, 0.01.

Output: simulation/outputs/images/gradient_k_per_column_lat_normalized.png
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
from render_norm_helpers import per_column_dev_normalized


KS = (0.16, 0.12, 0.08, 0.04, 0.02, 0.01)
STEPS = 8000
NX = 80
SAMPLE_COLS = (3, 8, 18, 30, 45, 60, 70)


def main():
    base = configs.make({"geometry": {"Nx": NX}}, base=configs.GRADIENT)
    geom, rule, pipes, bc = (
        base["geometry"], base["rule"], base["pipes"], base["boundary"]
    )
    inlet, outlet = configs.resolve_geometry(geom)
    Ny, Nx = geom["Ny"], geom["Nx"]
    ys = np.arange(Ny)

    print(f"[sim] running {len(KS)} sims, Nx={Nx}, {STEPS} steps", flush=True)
    isos = []
    for k in KS:
        out = tanks_vec.run(
            Nx=Nx, Ny=Ny, mode=rule["type"], steps=STEPS,
            inlet_cells=inlet, outlet_cells=outlet,
            threshold=rule["threshold"], max_volume=rule["max_volume"],
            max_pump=rule["max_pump"], gradient_k=k,
            directionality=pipes["directionality"], boundary=bc["type"],
            damping_cap=rule["damping_cap"],
            record_history=False,
        )
        isos.append(out["iso"])
        print(f"[sim]   k={k}: iso_max={int(out['iso'].max())}", flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True,
                             sharey=False, sharex=True)
    cmap = plt.cm.viridis
    norm_cols = Normalize(vmin=min(SAMPLE_COLS), vmax=max(SAMPLE_COLS))

    for i, (k, iso) in enumerate(zip(KS, isos)):
        ax = axes[i // 3, i % 3]
        for x in SAMPLE_COLS:
            color = cmap(norm_cols(x))
            dev, dt = per_column_dev_normalized(iso, x)
            if dev is None or dt is None:
                ax.plot([], [], color=color, lw=1.4, label=f"x={x} (not full)")
                continue
            ax.plot(ys, dev / dt, color=color, lw=1.4, label=f"x={x}")
        ax.axhline(0, color="gray", lw=0.5)
        ax.grid(alpha=0.3)
        ax.set_title(f"k = {k:.2f}", fontsize=11)
        ax.legend(fontsize=8, loc="best", ncol=2, framealpha=0.85)
        ax.tick_params(labelsize=9)

    fig.supxlabel("y (wavefront axis)", fontsize=12)
    fig.supylabel("dev / Δmean_x  (fractional lag, per-step deficit)", fontsize=12)
    fig.suptitle(
        "NORMALIZED per-column LAT — gradient_k sweep (Fickian rule)\n"
        "k·(V_src − V_dst) + one-way + zero-pad,    "
        f"Nx={Nx}, Ny={Ny}, {STEPS} steps.\n"
        "y-axis is dev/Δmean_x, so curves at different x should overlay if "
        "the per-step deficit is constant.",
        fontsize=12,
    )

    out_dir = ROOT / "outputs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gradient_k_per_column_lat_normalized.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[plot] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
