"""
Diagnostic: is the per-column LAT spread (the "crescent curvature") really
GROWING as the wavefront moves outward, or is the apparent growth an artifact
of the wave slowing down (so each column takes more steps to traverse,
inflating the absolute step-count lag while the *fractional* lag stays
constant)?

For each fully-activated column x, compute three quantities from the
isochrone:
  - mean_x       =  mean_y( iso[y, x] )                 ← arrival time
  - spread_x     =  max_y( iso[y, x] ) − min_y( iso[y, x] )    absolute lag (steps)
  - rel_spread_x =  spread_x / (mean_x − mean_{x−1})    lag as fraction of THIS column's
                                                        traversal time
  - rel_spread_2 =  spread_x / mean_x                   lag as fraction of total time-from-inlet

If user's assessment is right:
  spread_x grows roughly linearly with x         (apparent curvature growth)
  rel_spread_x is approximately CONSTANT          (real per-step deficit is fixed)
  mean_{x+1} − mean_x grows with x                (wave slowing down)

If user is wrong (deficit truly compounds):
  spread_x grows non-linearly (super-linear)
  rel_spread_x also grows
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from configs import GRADIENT, resolve_geometry
import tanks_vec


# Match R1 baseline (the standard crescent producer)
NX = GRADIENT["geometry"]["Nx"]
NY = GRADIENT["geometry"]["Ny"]
STEPS = GRADIENT["sim"]["steps"]
THRESHOLD = GRADIENT["rule"]["threshold"]
GRADIENT_K = GRADIENT["rule"]["gradient_k"]
inlet, outlet = resolve_geometry(GRADIENT["geometry"])

CASES = [
    ("R1_baseline",  "moore8",     True),
    ("R5_iso",       "moore8_iso", True),
    ("R3_no_thresh", "moore8",     False),
]

print(f"NX={NX} NY={NY} STEPS={STEPS}\n")

results = {}
for name, conn, gate in CASES:
    print(f"[run] {name}: connectivity={conn}, threshold_gate={gate}", flush=True)
    out = tanks_vec.run(
        Nx=NX, Ny=NY, mode="gradient", steps=STEPS,
        inlet_cells=inlet, outlet_cells=outlet,
        threshold=THRESHOLD,
        max_volume=GRADIENT["rule"]["max_volume"],
        max_pump=GRADIENT["rule"]["max_pump"],
        gradient_k=GRADIENT_K,
        directionality=GRADIENT["pipes"]["directionality"],
        boundary=GRADIENT["boundary"]["type"],
        damping_cap=GRADIENT["rule"]["damping_cap"],
        record_isochrone=True,
        record_history=False,
        connectivity=conn,
        threshold_gate=gate,
    )
    results[name] = out["iso"]


def per_column_metrics(iso: np.ndarray):
    """For each fully-activated column, return arrays:
       x_idx, mean_x, spread_x, traversal_dt, rel_spread_dt, rel_spread_total."""
    invalid = iso < 0
    ok = ~invalid.any(axis=0)
    iso_f = iso.astype(np.float64)
    iso_f[invalid] = np.nan

    x_idx = np.where(ok)[0]
    mean_x = np.nanmean(iso_f[:, x_idx], axis=0)
    spread_x = np.nanmax(iso_f[:, x_idx], axis=0) - np.nanmin(iso_f[:, x_idx], axis=0)
    # Column-to-column traversal time = mean_{x+1} − mean_x
    if len(x_idx) >= 2:
        traversal_dt = np.diff(mean_x, prepend=mean_x[0])
        traversal_dt[0] = traversal_dt[1] if len(traversal_dt) > 1 else 1.0
    else:
        traversal_dt = np.array([1.0])
    rel_spread_dt = spread_x / np.maximum(traversal_dt, 1e-9)
    rel_spread_total = spread_x / np.maximum(mean_x, 1e-9)
    return x_idx, mean_x, spread_x, traversal_dt, rel_spread_dt, rel_spread_total


fig, axes = plt.subplots(4, len(CASES), figsize=(5 * len(CASES), 14), constrained_layout=True)

for c, (name, conn, gate) in enumerate(CASES):
    iso = results[name]
    x_idx, mean_x, spread_x, traversal_dt, rel_dt, rel_total = per_column_metrics(iso)

    # Row 0: arrival time (mean_y of LAT)  vs x
    ax = axes[0, c]
    ax.plot(x_idx, mean_x, lw=1.5)
    ax.set_xlabel("x (column)")
    ax.set_ylabel("mean_y( LAT )  (steps)")
    ax.set_title(f"{name}\nWave arrival time")
    ax.grid(alpha=0.3)

    # Row 1: traversal time per column = mean_{x+1} − mean_x
    ax = axes[1, c]
    ax.plot(x_idx, traversal_dt, lw=1.5, color='tab:orange')
    ax.set_xlabel("x (column)")
    ax.set_ylabel("Δ mean_y(LAT) / Δ x  (steps per col)")
    ax.set_title("Per-column traversal time\n(higher = slower wave)")
    ax.grid(alpha=0.3)

    # Row 2: absolute spread = max(LAT) − min(LAT) per column
    ax = axes[2, c]
    ax.plot(x_idx, spread_x, lw=1.5, color='tab:red')
    ax.set_xlabel("x (column)")
    ax.set_ylabel("spread_x = max_y(LAT) − min_y(LAT)  (steps)")
    ax.set_title("Absolute crescent spread")
    ax.grid(alpha=0.3)

    # Row 3: relative spread = spread / per-column traversal time
    ax = axes[3, c]
    ax.plot(x_idx, rel_dt, lw=1.5, color='tab:green', label='spread / per-col Δt')
    ax.plot(x_idx, rel_total, lw=1.5, color='tab:purple', ls='--', label='spread / total mean_x')
    ax.set_xlabel("x (column)")
    ax.set_ylabel("relative spread")
    ax.set_title("Relative crescent\n(per-col fraction & total fraction)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle(
    "Per-column crescent diagnostic — does the curvature really grow,\n"
    "or is the apparent growth from the wave slowing down?",
    fontsize=13,
)

OUT = ROOT / "outputs" / "connectivity_threshold" / "per_column_curvature.png"
fig.savefig(OUT, dpi=140, bbox_inches='tight')
print(f"\n[plot] saved {OUT}")

# Numeric summary
print("\nNumeric summary (sampled at 5 columns):")
print(f"{'name':<14} {'x':>4} {'mean_x':>10} {'Δmean':>8} {'spread':>8} {'spread/Δmean':>14} {'spread/mean':>13}")
for name, _, _ in CASES:
    iso = results[name]
    x_idx, mean_x, spread_x, traversal_dt, rel_dt, rel_total = per_column_metrics(iso)
    if len(x_idx) < 5:
        sample = list(range(len(x_idx)))
    else:
        sample = [int(len(x_idx) * f) for f in (0.10, 0.30, 0.50, 0.70, 0.90)]
    for i in sample:
        if i < len(x_idx):
            print(f"{name:<14} {x_idx[i]:>4d} {mean_x[i]:>10.1f} "
                  f"{traversal_dt[i]:>8.2f} {spread_x[i]:>8.2f} "
                  f"{rel_dt[i]:>14.4f} {rel_total[i]:>13.4f}")
    print()
