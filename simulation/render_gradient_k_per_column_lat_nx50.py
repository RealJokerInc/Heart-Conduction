"""Per-column LAT-deviation figure for the gradient_k sweep on the GRADIENT
rule (Fickian k·(V_src−V_dst), one-way + zero-pad), short Nx=50 tissue so the
slow-k waves can fully traverse before stalling.

Layout: 2×3. Top row: k = 0.16, 0.12, 0.08. Bottom: 0.04, 0.02, 0.01.
Sample columns: (5, 15, 25, 35, 45) — 5 increments of 10 across the domain.

Convention: dev = column-mean iso − iso(y, x).
  positive  →  AHEAD of wavefront mean   negative  →  BEHIND

Output: simulation/outputs/images/gradient_k_per_column_lat_nx50.png
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


KS = (0.16, 0.12, 0.08, 0.04, 0.02, 0.01)
STEPS = 20000
NX = 50
SAMPLE_COLS = (5, 15, 25, 35, 45)


def main():
    base = configs.make({"geometry": {"Nx": NX}}, base=configs.GRADIENT)
    geom = base["geometry"]
    rule = base["rule"]
    pipes = base["pipes"]
    bc = base["boundary"]
    inlet, outlet = configs.resolve_geometry(geom)
    Ny, Nx = geom["Ny"], geom["Nx"]

    print(f"[sim] running {len(KS)} sims (k = {list(KS)}), "
          f"Nx={Nx}, {STEPS} steps, gradient rule", flush=True)
    isos = []
    for k in KS:
        out = tanks_vec.run(
            Nx=Nx, Ny=Ny,
            mode="gradient", steps=STEPS,
            inlet_cells=inlet, outlet_cells=outlet,
            threshold=rule["threshold"], max_volume=rule["max_volume"],
            max_pump=rule["max_pump"], gradient_k=k,
            directionality=pipes["directionality"], boundary=bc["type"],
            damping_cap=rule["damping_cap"],
            record_history=False,
        )
        isos.append(out["iso"])
        fired = (out["iso"] >= 0)
        rmost = int(np.where(fired.any(axis=0))[0].max()) if fired.any() else -1
        print(f"[sim]   k={k}: iso_max={int(out['iso'].max())}  rightmost_col={rmost}",
              flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True,
                             sharey=False, sharex=True)
    cmap = plt.cm.viridis
    norm = Normalize(vmin=min(SAMPLE_COLS), vmax=max(SAMPLE_COLS))
    ys = np.arange(Ny)

    for i, (k, iso) in enumerate(zip(KS, isos)):
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
                dev = mean_iso - col_iso
                ax.plot(ys, dev, color=color, lw=1.4, label=f"x={x} (partial)")
            else:
                mean_iso = float(col_iso.mean())
                dev = mean_iso - col_iso
                ax.plot(ys, dev, color=color, lw=1.4, label=f"x={x}")
        ax.axhline(0, color="gray", lw=0.5)
        ax.grid(alpha=0.3)
        ax.set_title(f"k = {k:g}", fontsize=11)
        ax.legend(fontsize=9, loc="best", ncol=1, framealpha=0.85)
        ax.tick_params(labelsize=9)

    fig.supxlabel("wavefront axis", fontsize=12)
    fig.supylabel("axis of propagation", fontsize=12)

    out_dir = Path(__file__).parent / "outputs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gradient_k_per_column_lat_nx50.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[plot] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
