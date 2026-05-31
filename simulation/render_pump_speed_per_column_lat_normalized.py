"""Per-column LAT-deviation figure for the pump-speed sweep, NORMALIZED:
divide each column's dev = (mean_iso − iso) by that column's per-column
traversal time Δmean_x. This converts the y-axis from raw step-count
(dilated by wave-slowing) into "fraction of per-column traversal time"
(true per-step deficit).

Same 2x3 layout as pump_speed_per_column_lat_nx80.png.
Layout: top row max_pump = 30, 20, 15. bottom = 10, 5, 2.

Output: simulation/outputs/images/pump_speed_per_column_lat_normalized.png
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


PUMPS = (30.0, 20.0, 15.0, 10.0, 5.0, 2.0)
STEPS = 8000
NX = 80
SAMPLE_COLS = (3, 8, 18, 30, 45, 60, 70)


def main():
    base = configs.make({"geometry": {"Nx": NX}}, base=configs.BASELINE)
    geom, rule, pipes, bc = (
        base["geometry"], base["rule"], base["pipes"], base["boundary"]
    )
    inlet, outlet = configs.resolve_geometry(geom)
    Ny, Nx = geom["Ny"], geom["Nx"]
    ys = np.arange(Ny)

    print(f"[sim] running {len(PUMPS)} sims, Nx={Nx}, {STEPS} steps", flush=True)
    isos = []
    for mp in PUMPS:
        out = tanks_vec.run(
            Nx=Nx, Ny=Ny, mode=rule["type"], steps=STEPS,
            inlet_cells=inlet, outlet_cells=outlet,
            threshold=rule["threshold"], max_volume=rule["max_volume"],
            max_pump=mp, gradient_k=rule["gradient_k"],
            directionality=pipes["directionality"], boundary=bc["type"],
            damping_cap=rule["damping_cap"],
            record_history=False,
        )
        isos.append(out["iso"])
        print(f"[sim]   max_pump={mp}: iso_max={int(out['iso'].max())}", flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True,
                             sharey=False, sharex=True)
    cmap = plt.cm.viridis
    norm_cols = Normalize(vmin=min(SAMPLE_COLS), vmax=max(SAMPLE_COLS))

    for i, (mp, iso) in enumerate(zip(PUMPS, isos)):
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
        ax.set_title(f"max_pump = {mp:.0f}", fontsize=11)
        ax.legend(fontsize=8, loc="best", ncol=2, framealpha=0.85)
        ax.tick_params(labelsize=9)

    fig.supxlabel("y (wavefront axis)", fontsize=12)
    fig.supylabel("dev / Δmean_x  (fractional lag, per-step deficit)", fontsize=12)
    fig.suptitle(
        "NORMALIZED per-column LAT — pump-speed sweep (John's BASELINE rule)\n"
        "constant √(V−θ) + one-way + zero-pad + damping,    "
        f"Nx={Nx}, Ny={Ny}, {STEPS} steps.\n"
        "y-axis is dev/Δmean_x, so curves at different x should overlay if "
        "the per-step deficit is constant.",
        fontsize=12,
    )

    out_dir = ROOT / "outputs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pump_speed_per_column_lat_normalized.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[plot] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
