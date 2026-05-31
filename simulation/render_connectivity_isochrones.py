"""Connectivity-comparison isochrone figure: 8-pt uniform Moore vs 4-pt
cardinal vs ISO 4:1 normalized 9-pt — same isochrone format as
pump_speed_isochrones.png, but with x-evenly-spaced contour levels (factors
out wave-slowing) so the per-column boundary asymmetry is comparable
across connectivities.

Three runs (gradient mode + one_way + zero_pad + line + threshold gate):
  R1  moore8       — 8-pt uniform (John's Fickian-modified default)
  R2  cardinal4    — 4-pt cardinal (matches monodomain face_mirror)
  R5  moore8_iso   — 4:1 isotropic (Patra-Kałuża, normalized 9-pt)

Output: simulation/outputs/images/connectivity_isochrones.png
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


NX = 320
STEPS = 20000
NUM_LEVELS = 10

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

    print(f"[sim] running {len(CASES)} sims, Nx={Nx}, {STEPS} steps", flush=True)
    runs = []
    for label, conn, gate in CASES:
        print(f"[sim]   {label}  (connectivity={conn})", flush=True)
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
        rmost = int((iso >= 0).any(axis=0).cumsum().argmax()) if (iso >= 0).any() else -1
        print(f"[sim]    rightmost reached col: {rmost}", flush=True)
        runs.append((label, conn, iso, rmost))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    cmap = plt.cm.viridis

    # Build a per-panel set of normalized levels (each connectivity has its
    # own propagation rate, so its contour levels live in its own time scale).
    per_panel_levels = []
    for _, _, iso, _ in runs:
        levels = x_evenly_spaced_levels(iso, NUM_LEVELS)
        per_panel_levels.append(levels)

    # For colorbar, normalize by FRACTIONAL x-reach (1..NUM_LEVELS), so
    # contours mean the same thing across panels.
    contour_norm = Normalize(vmin=1, vmax=NUM_LEVELS)
    sm_for_cbar = plt.cm.ScalarMappable(norm=contour_norm, cmap=cmap)
    sm_for_cbar.set_array([])

    for i, ((label, conn, iso, rmost), levels) in enumerate(zip(runs, per_panel_levels)):
        ax = axes[i]
        iso_plot = iso.astype(float)
        iso_plot[iso_plot < 0] = np.nan
        unfired = (iso < 0).astype(float)
        ax.contourf(unfired, levels=[0.5, 1.5], colors=["#e8e8e8"], zorder=0)
        if len(levels) > 0:
            # color levels by their index (fraction of x-reach) so all panels
            # use the same colormap meaning regardless of absolute step values
            level_colors = [cmap(contour_norm(j + 1)) for j in range(len(levels))]
            for lvl, col in zip(levels, level_colors):
                ax.contour(iso_plot, levels=[lvl], colors=[col],
                           linewidths=1.5, zorder=2)
        ax.set_xlim(0, Nx - 1)
        ax.set_ylim(Ny - 1, 0)
        ax.set_aspect("equal")
        ax.set_title(f"{label}\n(reached col {rmost})", fontsize=11)
        ax.set_xlabel("x  (column)", fontsize=10)
        if i == 0:
            ax.set_ylabel("y  (row)", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.18, zorder=1)

    cbar = fig.colorbar(sm_for_cbar, ax=axes.tolist(),
                        orientation="vertical", shrink=0.85, pad=0.012,
                        label=f"x-reach level (1 = early, {NUM_LEVELS} = late;\n"
                              f"levels chosen at evenly-spaced x-fractions, NOT step times)")
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Connectivity-comparison isochrones (NORMALIZED)\n"
        "GRADIENT (Fickian) + one_way + zero_pad + line + threshold,    "
        f"Nx={Nx}, Ny={Ny}, {STEPS} steps.\n"
        "Contour levels at x-evenly-spaced wave-mean positions (factors out wave-slowing).",
        fontsize=12,
    )

    out_dir = ROOT / "outputs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "connectivity_isochrones.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\n[plot] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
