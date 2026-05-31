"""Per-column LAT-deviation figure for the pump-speed sweep on John's BASELINE rule.

For each pump rate (max_pump = 30, 20, 15, 10, 5, 2), and each sampled column x,
plot iso(y, x) − column-mean iso vs y. Negative deviation at the boundary (y=0 or
y=49) means the boundary fires earlier than the column average — U-shape = camel
toe. Positive at boundary = inverted-U = crescent.

Columns sampled: scaled 4× from the reference figure (Nx=80→320), so
(3,8,18,30,45,60,70) → (12, 32, 72, 120, 180, 240, 280).

Layout: 2×3 grid. Top row: max_pump = 30, 20, 15. Bottom: 10, 5, 2.
Output: simulation/outputs/images/pump_speed_per_column_lat.png
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


PUMPS = (30.0, 20.0, 15.0, 10.0, 5.0, 2.0)
STEPS = 20000
NX = 320
SAMPLE_COLS = (12, 32, 72, 120, 180, 240, 280)


def main():
    base = configs.make({"geometry": {"Nx": NX}}, base=configs.BASELINE)
    geom = base["geometry"]
    rule = base["rule"]
    pipes = base["pipes"]
    bc = base["boundary"]
    inlet, outlet = configs.resolve_geometry(geom)
    Ny, Nx = geom["Ny"], geom["Nx"]

    print(f"[sim] running {len(PUMPS)} sims (max_pump = {list(PUMPS)}), "
          f"Nx={Nx}, {STEPS} steps", flush=True)
    isos = []
    for mp in PUMPS:
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
        isos.append(out["iso"])
        print(f"[sim]   max_pump={mp}: iso_max={int(out['iso'].max())}", flush=True)

    # Differential y-scaling: each subplot autoscales to its own deviation
    # range so the shape is legible at every pump rate.
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True,
                             sharey=False, sharex=True)
    cmap = plt.cm.viridis
    norm = Normalize(vmin=min(SAMPLE_COLS), vmax=max(SAMPLE_COLS))
    ys = np.arange(Ny)

    for i, (mp, iso) in enumerate(zip(PUMPS, isos)):
        ax = axes[i // 3, i % 3]
        for x in SAMPLE_COLS:
            col_iso = iso[:, x].astype(float)
            mask = col_iso >= 0
            color = cmap(norm(x))
            if not mask.any():
                ax.plot([], [], color=color, lw=1.4, label=f"x={x} (not reached)")
                continue
            if not mask.all():
                col_iso[~mask] = np.nan
                mean_iso = float(np.nanmean(col_iso))
                dev = mean_iso - col_iso  # positive = ahead, negative = behind
                ax.plot(ys, dev, color=color, lw=1.4, label=f"x={x} (partial)")
            else:
                mean_iso = float(col_iso.mean())
                dev = mean_iso - col_iso  # positive = ahead, negative = behind
                ax.plot(ys, dev, color=color, lw=1.4, label=f"x={x}")
        ax.axhline(0, color="gray", lw=0.5)
        ax.grid(alpha=0.3)
        ax.set_title(f"max_pump = {mp:.0f}", fontsize=11)
        ax.legend(fontsize=8, loc="best", ncol=2, framealpha=0.85)
        ax.tick_params(labelsize=9)

    fig.supxlabel("wavefront axis", fontsize=12)
    fig.supylabel("axis of propagation", fontsize=12)

    out_dir = Path(__file__).parent / "outputs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pump_speed_per_column_lat.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[plot] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
