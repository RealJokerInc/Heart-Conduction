"""Step 1 of PLAN.md — mass-conservation audit for horizontal-redirect BC.

Compares V_sum trajectories for HBB (case10) vs horizontal (case13),
decomposed by region:
  - Corners (4 cells)
  - Wall non-corner (top + bottom rows excluding corners)
  - Interior (rows j ∈ [1, NY-2])

Output: figures/horizontal_mass_audit.png + console summary.
"""
from __future__ import annotations
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "figures" / "horizontal_mass_audit.png"

CASES = {
    "HBB":        "case10_lbm_d2q9_canonical_hbb_natural.h5",
    "horizontal": "case13_lbm_d2q9_canonical_horizontal_natural.h5",
}


def decompose(V):
    """V shape (T, NX, NY). Returns four time-series, each shape (T,)."""
    T, NX, NY = V.shape
    total = V.sum(axis=(1, 2))
    corners = (V[:, 0, 0] + V[:, 0, -1]
               + V[:, -1, 0] + V[:, -1, -1])
    # Wall non-corner = top + bottom rows excluding the four corner cells
    top_nc = V[:, 1:NX - 1, NY - 1].sum(axis=1)
    bot_nc = V[:, 1:NX - 1, 0].sum(axis=1)
    wall_nc = top_nc + bot_nc
    # Interior = rows j in [1, NY-2], all cols
    interior = V[:, :, 1:NY - 1].sum(axis=(1, 2))
    return total, corners, wall_nc, interior


def main():
    data = {}
    for name, fname in CASES.items():
        with h5py.File(DATA / fname, "r") as f:
            V = f["V"][:]
            t = f["t"][:]
        data[name] = (t, V, *decompose(V))

    # 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [
        ("V_sum (total)",            1),
        ("V_sum (corners only)",     2),
        ("V_sum (wall non-corner)",  3),
        ("V_sum (interior)",         4),
    ]
    for (title, idx), ax in zip(panels, axes.flat):
        for name, color in zip(CASES, ("tab:blue", "tab:red")):
            t, V, total, corners, wall_nc, interior = data[name]
            series = (total, corners, wall_nc, interior)[idx - 1]
            ax.plot(t, series, label=name, color=color, lw=1.6)
        ax.set_title(title)
        ax.set_xlabel("t (ms)")
        ax.set_ylabel("Σ V (mV)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Mass-conservation audit: HBB vs horizontal redirect", fontsize=13)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT.name}")

    # ─── Console summary ───────────────────────────────────────────────────
    print("\n=== mass-balance summary (t = final) ===")
    print(f"  {'BC':<12}  {'total':>12}  {'corners':>10}  {'wall_nc':>10}  {'interior':>10}")
    for name in CASES:
        t, V, total, corners, wall_nc, interior = data[name]
        print(f"  {name:<12}  {total[-1]:>12.2f}  {corners[-1]:>10.2f}  "
              f"{wall_nc[-1]:>10.2f}  {interior[-1]:>10.2f}")

    # Excess attribution
    t_hbb, V_hbb, tot_hbb, cor_hbb, wnc_hbb, int_hbb = data["HBB"]
    t_h,   V_h,   tot_h,   cor_h,   wnc_h,   int_h   = data["horizontal"]
    excess_total = tot_h[-1] - tot_hbb[-1]
    excess_corn  = cor_h[-1] - cor_hbb[-1]
    excess_wnc   = wnc_h[-1] - wnc_hbb[-1]
    excess_int   = int_h[-1] - int_hbb[-1]
    print(f"\n  excess  total={excess_total:+.2f}  "
          f"corners={excess_corn:+.2f}  "
          f"wall_nc={excess_wnc:+.2f}  "
          f"interior={excess_int:+.2f}")
    if abs(excess_total) > 1e-6:
        pct_corn = 100 * excess_corn / excess_total
        pct_wnc  = 100 * excess_wnc  / excess_total
        pct_int  = 100 * excess_int  / excess_total
        print(f"  excess fractions:  corners={pct_corn:+.1f}%  "
              f"wall_nc={pct_wnc:+.1f}%  interior={pct_int:+.1f}%")

    # Peak leak rate
    leak = (tot_h - tot_hbb)
    leak_rate = np.diff(leak) / np.diff(t_h)
    peak_k = int(np.argmax(np.abs(leak_rate)))
    print(f"\n  peak leak rate: {leak_rate[peak_k]:+.2f} mV/ms "
          f"at t = {t_h[peak_k]:.2f} ms")


if __name__ == "__main__":
    main()
