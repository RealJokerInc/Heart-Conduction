"""
Ablation: which ingredient creates the crescent in John's Fickian-modified setup?

Four runs, all using GRADIENT mode (Fickian rate) + one_way + zero_pad +
line geometry — i.e., the user's "Fickian variation":

  R1  moore8    + threshold_gate=True       (baseline; user's setup)
  R2  cardinal4 + threshold_gate=True       (kill diagonals only)
  R3  moore8    + threshold_gate=False      (drop fired_p only)
  R4  cardinal4 + threshold_gate=False      (drop both)

Predicted ranking (largest → smallest crescent):
  R1 > R3 > R2 ≈ R4 ≈ flat

Bridge claim: Moore-8 connectivity is necessary for the crescent in y-uniform
line stim. The threshold gate amplifies but is not required.

Output:
  outputs/connectivity_threshold/
    iso_R1.png  iso_R2.png  iso_R3.png  iso_R4.png    isochrone heatmaps
    crescent_summary.png                              4-panel comparison + LAT(y) curves
    crescent_summary.txt                              numerical max|LAT_dev| per run
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


# ---------- config ----------
BASE = GRADIENT  # gradient mode + one_way + zero_pad + line geometry
NX = BASE["geometry"]["Nx"]   # 80
NY = BASE["geometry"]["Ny"]   # 50
STEPS = BASE["sim"]["steps"]  # 4000
THRESHOLD = BASE["rule"]["threshold"]  # 45
GRADIENT_K = BASE["rule"]["gradient_k"]  # 0.08

inlet_cells, outlet_cells = resolve_geometry(BASE["geometry"])

RUNS = [
    ("R1_moore8_thresh",         "moore8",     True),
    ("R2_cardinal4_thresh",      "cardinal4",  True),
    ("R3_moore8_no_thresh",      "moore8",     False),
    ("R4_cardinal4_no_thresh",   "cardinal4",  False),
    ("R5_moore8iso_thresh",      "moore8_iso", True),
    ("R6_moore8iso_no_thresh",   "moore8_iso", False),
]

OUT = ROOT / "outputs" / "connectivity_threshold"
OUT.mkdir(parents=True, exist_ok=True)


def run_one(connectivity: str, threshold_gate: bool) -> dict:
    return tanks_vec.run(
        Nx=NX, Ny=NY, mode="gradient", steps=STEPS,
        inlet_cells=inlet_cells, outlet_cells=outlet_cells,
        threshold=THRESHOLD,
        max_volume=BASE["rule"]["max_volume"],
        max_pump=BASE["rule"]["max_pump"],
        gradient_k=GRADIENT_K,
        directionality=BASE["pipes"]["directionality"],   # one_way
        boundary=BASE["boundary"]["type"],                # zero_pad
        damping_cap=BASE["rule"]["damping_cap"],
        record_isochrone=True,
        record_history=False,
        connectivity=connectivity,
        threshold_gate=threshold_gate,
    )


# ---------- run all 4 ----------
results = {}
for name, conn, gate in RUNS:
    print(f"[run] {name}  connectivity={conn}  threshold_gate={gate} ...", flush=True)
    res = run_one(conn, gate)
    results[name] = res
    iso = res["iso"]
    activated = (iso >= 0).sum()
    print(f"       activated cells: {activated}/{NX*NY}    iso_max: {iso.max()}", flush=True)


# ---------- crescent metric ----------
def crescent_metric(iso: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """For each x-column, compute mean LAT and the deviation field
    LAT(x, y) - mean_y[LAT(x, y)]. Return:
      max |LAT_dev|  (only over fully-activated columns)
      ok_mask[x]    True where the column is fully activated
      dev[y, x]
    iso has shape (Ny, Nx), -1 where not activated.
    """
    invalid = iso < 0
    # Only count columns where every y is activated:
    ok = ~invalid.any(axis=0)   # shape (Nx,)
    iso_f = iso.astype(np.float64)
    iso_f[invalid] = np.nan
    col_mean = np.nanmean(iso_f, axis=0, keepdims=True)
    dev = iso_f - col_mean
    if ok.any():
        m = np.abs(dev[:, ok]).max()
    else:
        m = float('nan')
    return m, ok, dev


# ---------- per-run isochrone plots ----------
print("\n[render] isochrones...", flush=True)
for name, _, _ in RUNS:
    iso = results[name]["iso"]
    fig, ax = plt.subplots(figsize=(8, 5))
    iso_plot = np.where(iso >= 0, iso, np.nan)
    im = ax.imshow(iso_plot, origin='lower', cmap='viridis', aspect='auto')
    ax.set_title(f"{name}\nfirst-activation step (V crosses {THRESHOLD})")
    ax.set_xlabel("x (column)")
    ax.set_ylabel("y (row)")
    plt.colorbar(im, ax=ax, label="step")
    fig.savefig(OUT / f"iso_{name}.png", dpi=140, bbox_inches='tight')
    plt.close(fig)


# ---------- 4-panel summary ----------
fig, axes = plt.subplots(2, len(RUNS), figsize=(5 * len(RUNS), 9), constrained_layout=True)

for c, (name, conn, gate) in enumerate(RUNS):
    iso = results[name]["iso"]
    m, ok, dev = crescent_metric(iso)

    # row 0: dev[y, x] heatmap
    ax = axes[0, c]
    dev_show = np.where(np.isnan(dev), 0.0, dev)
    vmax = max(np.abs(dev_show).max(), 1.0)
    im = ax.imshow(dev_show, origin='lower', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_title(f"{name}\n{conn}, thresh_gate={gate}\nmax|LAT-y_mean|={m:.1f} steps")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="LAT - mean_y(LAT) (steps)")

    # row 1: LAT(y) at sampled columns
    ax = axes[1, c]
    iso_f = iso.astype(np.float64)
    iso_f[iso < 0] = np.nan
    sample_xs = [int(NX * f) for f in (0.20, 0.40, 0.60, 0.80)]
    cmap = plt.cm.plasma
    for k, xi in enumerate(sample_xs):
        if not ok[xi]:
            continue
        col = iso_f[:, xi]
        ax.plot(np.arange(NY), col, color=cmap(k / 3.0),
                lw=1.3, label=f"x={xi}")
    ax.set_title(f"{name}: LAT vs y at sampled x")
    ax.set_xlabel("y")
    ax.set_ylabel("LAT (step)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle(
    f"Connectivity × threshold-gate ablation on user's Fickian variant\n"
    f"(GRADIENT mode + one_way + zero_pad, line geometry, {NX}x{NY}, {STEPS} steps)",
    fontsize=13,
)
fig.savefig(OUT / "crescent_summary.png", dpi=140, bbox_inches='tight')
plt.close(fig)


# ---------- numerical summary ----------
lines = ["Connectivity × threshold-gate ablation\n",
         "user's Fickian variant: GRADIENT + one_way + zero_pad + line\n",
         f"NX={NX} NY={NY} STEPS={STEPS} threshold={THRESHOLD} gradient_k={GRADIENT_K}\n",
         "\n",
         f"{'name':<28} {'connectivity':<12} {'thresh_gate':<12} "
         f"{'cols_full':<10} {'max|LAT-meanY|':<18} {'iso_max':<10}\n",
         "-"*100 + "\n"]
for name, conn, gate in RUNS:
    iso = results[name]["iso"]
    m, ok, _ = crescent_metric(iso)
    lines.append(
        f"{name:<28} {conn:<12} {str(gate):<12} "
        f"{int(ok.sum()):<10} {m:<18.4f} {int(iso.max()):<10}\n"
    )
summary_txt = "".join(lines)
print("\n" + summary_txt)
(OUT / "crescent_summary.txt").write_text(summary_txt)

print(f"\n[done] outputs in {OUT}", flush=True)
