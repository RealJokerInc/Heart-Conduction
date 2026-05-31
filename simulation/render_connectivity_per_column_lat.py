"""Per-column LAT-deviation comparison: 8-pt uniform vs 4-pt cardinal vs
ISO 4:1 — same per-column LAT format as pump_speed_per_column_lat_nx80.png,
but with NORMALIZED y-axis (dev / Δmean_x) so the wave-slowing artifact is
factored out.

Three runs (gradient mode + one_way + zero_pad + line + threshold):
  R1  moore8       — 8-pt uniform
  R2  cardinal4    — 4-pt cardinal
  R5  moore8_iso   — 4:1 isotropic

Output: simulation/outputs/images/connectivity_per_column_lat.png
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


NX = 80
STEPS = 8000
SAMPLE_COLS = (3, 8, 18, 30, 45, 60, 70)

CASES = [
    ("moore8 (8-pt uniform)", "moore8",     True),
    ("cardinal4 (4-pt)",      "cardinal4",  True),
    ("moore8_iso (4:1)",      "moore8_iso", True),
]


def main():
    base = configs.make({"geometry": {"Nx": NX}}, base=configs.GRADIENT)
    geom, rule, pipes, bc = (
        base["geometry"], base["rule"], base["pipes"], base["boundary"]
    )
    inlet, outlet = configs.resolve_geometry(geom)
    Ny, Nx = geom["Ny"], geom["Nx"]
    ys = np.arange(Ny)

    print(f"[sim] running {len(CASES)} sims, Nx={Nx}, {STEPS} steps", flush=True)
    runs = []
    for label, conn, gate in CASES:
        print(f"[sim]   {label}", flush=True)
        out = tanks_vec.run(
            Nx=Nx, Ny=Ny, mode=rule["type"], steps=STEPS,
            inlet_cells=inlet, outlet_cells=outlet,
            threshold=rule["threshold"],
            max_volume=rule["max_volume"],
            max_pump=rule["max_pump"],
            gradient_k=rule["gradient_k"],
            directionality=pipes["directionality"],
            boundary=bc["type"],
            damping_cap=rule["damping_cap"],
            connectivity=conn,
            threshold_gate=gate,
            record_history=False,
        )
        iso = out["iso"]
        runs.append((label, conn, iso))

    # 2 rows × 3 cols: top = absolute dev, bottom = normalized dev (dev/Δmean)
    fig, axes = plt.subplots(2, len(CASES), figsize=(6 * len(CASES), 9),
                             constrained_layout=True, sharex=True)
    cmap = plt.cm.viridis
    norm_cols = Normalize(vmin=min(SAMPLE_COLS), vmax=max(SAMPLE_COLS))

    for c, (label, conn, iso) in enumerate(runs):
        ax_abs = axes[0, c]
        ax_norm = axes[1, c]

        for x in SAMPLE_COLS:
            color = cmap(norm_cols(x))
            dev, dt = per_column_dev_normalized(iso, x)
            if dev is None or dt is None:
                ax_abs.plot([], [], color=color, lw=1.4, label=f"x={x} (not full)")
                ax_norm.plot([], [], color=color, lw=1.4, label=f"x={x} (not full)")
                continue
            ax_abs.plot(ys, dev, color=color, lw=1.4, label=f"x={x}")
            ax_norm.plot(ys, dev / dt, color=color, lw=1.4, label=f"x={x}")

        for ax in (ax_abs, ax_norm):
            ax.axhline(0, color="gray", lw=0.5)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=9)

        ax_abs.set_title(f"{label}\nabsolute dev = mean − iso  (steps)", fontsize=11)
        ax_norm.set_title(f"NORMALIZED  dev / Δmean_x  (fraction of "
                          f"per-col traversal time)", fontsize=11)
        ax_abs.legend(fontsize=8, loc="best", ncol=2, framealpha=0.85)
        if c == 0:
            ax_abs.set_ylabel("dev (steps)", fontsize=10)
            ax_norm.set_ylabel("dev / Δmean_x", fontsize=10)
        ax_norm.set_xlabel("y (row)", fontsize=10)

    fig.suptitle(
        "Per-column LAT deviation vs y, three connectivities\n"
        "GRADIENT + one_way + zero_pad + line + threshold,    "
        f"Nx={Nx}, Ny={Ny}, {STEPS} steps.\n"
        "Top row: absolute (step units, dilated by wave-slowing).  "
        "Bottom row: normalized by per-col traversal time (true per-step deficit).",
        fontsize=12,
    )

    out_dir = ROOT / "outputs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "connectivity_per_column_lat.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\n[plot] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
