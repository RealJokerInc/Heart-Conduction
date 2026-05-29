"""Investigate the uniform pre-depolarization of the wall row under
horizontal redirect.

The observation: at t = 0.5 ms (25 LBM steps), the bottom wall row
(j=0) is depolarized to V ≈ -72 mV UNIFORMLY across cols 10-30 (10 mV
above V_rest), even though the bulk wavefront has barely left col 0.
Under HBB, the same cells stay at V_rest.

This is NOT just "the wall channel propagates faster" — it's
*instantaneous* uniform pre-charging of the entire wall row.

Output:
  figures/horizontal_wallrow_evolution.png — V(x, j=0) at several times,
    HBB vs horizontal (TTP06 and diffusion-only).
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
OUT = ROOT / "figures" / "horizontal_wallrow_evolution.png"

CASES_TTP06 = [
    ("HBB",        "case10_lbm_d2q9_canonical_hbb_natural.h5",   "tab:blue"),
    ("horizontal", "case13_lbm_d2q9_canonical_horizontal_natural.h5", "tab:red"),
]
CASES_DIFF = [
    ("HBB-diff",        "case10_lbm_d2q9_canonical_hbb_natural_diffusion.h5",         "tab:blue"),
    ("horizontal-diff", "case13_lbm_d2q9_canonical_horizontal_natural_diffusion.h5", "tab:red"),
]
TIMES_MS = [0.5, 1.0, 2.0, 5.0]


def load(fname):
    with h5py.File(DATA / fname, "r") as f:
        return f["t"][:], f["V"][:]


def main():
    fig, axes = plt.subplots(2, len(TIMES_MS), figsize=(16, 8),
                              sharex=True, sharey='row',
                              constrained_layout=True)

    # ─── Row 1: TTP06 ─────────────────────────────────────────────────
    for c, t_target in enumerate(TIMES_MS):
        ax = axes[0, c]
        for name, fname, color in CASES_TTP06:
            t, V = load(fname)
            k = int(np.argmin(np.abs(t - t_target)))
            wall_row = V[k, :, 0]   # V along x at j=0
            ax.plot(np.arange(len(wall_row)), wall_row,
                     label=name, color=color, lw=1.8, marker='.', ms=5)
        ax.axhline(-85.23, ls='--', c='black', alpha=0.4, lw=0.8, label='_V_rest')
        ax.set_title(f"TTP06   t = {t_target:.1f} ms", fontsize=11)
        ax.set_xlabel("x col")
        ax.set_ylabel("V at j=0 (mV)")
        ax.grid(True, alpha=0.3)
        if c == 0:
            ax.legend(loc='upper right', fontsize=9)

    # ─── Row 2: diffusion-only ─────────────────────────────────────────
    for c, t_target in enumerate(TIMES_MS):
        ax = axes[1, c]
        for name, fname, color in CASES_DIFF:
            t, V = load(fname)
            k = int(np.argmin(np.abs(t - t_target)))
            wall_row = V[k, :, 0]
            ax.plot(np.arange(len(wall_row)), wall_row,
                     label=name, color=color, lw=1.8, marker='.', ms=5)
        ax.axhline(-85.23, ls='--', c='black', alpha=0.4, lw=0.8)
        ax.set_title(f"diffusion only   t = {t_target:.1f} ms", fontsize=11)
        ax.set_xlabel("x col")
        ax.set_ylabel("V at j=0 (mV)")
        ax.grid(True, alpha=0.3)
        if c == 0:
            ax.legend(loc='upper right', fontsize=9)

    fig.suptitle(
        "Wall-row depolarization under horizontal redirect — UNIFORM across x "
        "within ~25 LBM steps,\nwell ahead of any propagating wavefront. "
        "HBB stays at V_rest at the same x positions.",
        fontsize=12,
    )
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT.name}")

    # Console: how uniform is the wall row under horizontal at t=0.5ms?
    print("\n=== uniformity test at t=0.5 ms (diffusion-only) ===")
    for name, fname, _ in CASES_DIFF:
        t, V = load(fname)
        k = int(np.argmin(np.abs(t - 0.5)))
        wall = V[k, :, 0]
        # Stats over cols 10-35 (away from stim and east corner)
        mid = wall[10:36]
        print(f"  {name:<18}  cols 10-35:  mean={mid.mean():+.3f}  "
              f"std={mid.std():.5f}  max-min={mid.max()-mid.min():.5f}  "
              f"V_rest diff={mid.mean() - (-85.23):+.3f}")


if __name__ == "__main__":
    main()
