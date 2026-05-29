"""Step 5 of PLAN.md — synthesis figure for horizontal-redirect diagnosis.

3×2 grid:
  Row 1: V(y) at col 20, t=10 ms  —  HBB, buggy horizontal, fixed horizontal
  Row 2: V_sum(t) trajectories     —  TTP06 vs diffusion-only

Shows three things at once:
 (a) wall channel + sub-edge dip pattern is identical buggy vs fixed
 (b) diffusion-only V_sum stays flat (no leak)
 (c) TTP06 V_sum grows in both — wall channel advances faster ⇒ more plateau cells
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
OUT = ROOT / "figures" / "horizontal_synthesis.png"

CASES_TTP06 = [
    ("HBB",              "case10_lbm_d2q9_canonical_hbb_natural.h5",                                "tab:blue"),
    ("horiz (buggy)",    "case13_lbm_d2q9_canonical_horizontal_natural.h5",                        "tab:red"),
    ("horiz (fixed)",    "case_horiz_fixed_lbm_d2q9_canonical_horizontal_fixed_natural.h5",        "tab:purple"),
]
CASES_DIFF = [
    ("HBB",              "case10_lbm_d2q9_canonical_hbb_natural_diffusion.h5",                     "tab:blue"),
    ("horiz (buggy)",    "case13_lbm_d2q9_canonical_horizontal_natural_diffusion.h5",             "tab:red"),
    ("horiz (fixed)",    "case_horiz_fixed_lbm_d2q9_canonical_horizontal_fixed_natural_diffusion.h5", "tab:purple"),
]


def load(fname):
    with h5py.File(DATA / fname, "r") as f:
        return f["t"][:], f["V"][:]


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    # ─── Row 1: V(y) at col 20, t=10 ms (just past wavefront) ──────────
    for col_idx, t_target in enumerate([5.0, 10.0, 25.0]):
        ax = axes[0, col_idx]
        for name, fname, color in CASES_TTP06:
            t, V = load(fname)
            k = int(np.argmin(np.abs(t - t_target)))
            ax.plot(np.arange(V.shape[2]), V[k, 20, :], label=name,
                     color=color, lw=1.8, marker='.', ms=5)
        ax.axhline(-85.23, ls='--', c='black', alpha=0.4, lw=0.8)
        ax.axvline(0,      ls=':', c='grey', alpha=0.5)
        ax.axvline(V.shape[2]-1, ls=':', c='grey', alpha=0.5)
        ax.axvline(1,      ls=':', c='orange', alpha=0.5)
        ax.axvline(V.shape[2]-2, ls=':', c='orange', alpha=0.5)
        ax.set_title(f"V(y) at col 20, t = {t_target:.0f} ms  (TTP06)", fontsize=11)
        ax.set_xlabel("y index")
        ax.set_ylabel("V (mV)")
        ax.grid(True, alpha=0.3)
        if col_idx == 0:
            ax.legend(loc='lower right', fontsize=9)

    # ─── Row 2: V_sum trajectories (TTP06 and diffusion-only) ──────────
    # Panel 0: TTP06 V_sum vs t
    ax = axes[1, 0]
    for name, fname, color in CASES_TTP06:
        t, V = load(fname)
        ax.plot(t, V.sum(axis=(1, 2)) / V[0].size, label=name, color=color, lw=1.8)
    ax.set_title("V_sum / N_cells over time  (TTP06)", fontsize=11)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("⟨V⟩ (mV)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)

    # Panel 1: diffusion-only V_sum vs t (should be flat)
    ax = axes[1, 1]
    for name, fname, color in CASES_DIFF:
        t, V = load(fname)
        vs = V.sum(axis=(1, 2)) / V[0].size
        # Plot deviation from initial value to make the conservation visible
        ax.plot(t, vs - vs[0], label=name, color=color, lw=1.8)
    ax.axhline(0, ls='--', c='black', alpha=0.4)
    ax.set_title("⟨V⟩(t) − ⟨V⟩(0)  (diffusion only — should be 0)", fontsize=11)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("Δ⟨V⟩ (mV)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    # Panel 2: V(y) at col 20, diffusion-only, t = 25 ms (sub-edge dip persistence)
    ax = axes[1, 2]
    for name, fname, color in CASES_DIFF:
        t, V = load(fname)
        k = int(np.argmin(np.abs(t - 25.0)))
        ax.plot(np.arange(V.shape[2]), V[k, 20, :], label=name,
                 color=color, lw=1.8, marker='.', ms=5)
    ax.axhline(-85.23, ls='--', c='black', alpha=0.4, lw=0.8)
    ax.axvline(0, ls=':', c='grey', alpha=0.5)
    ax.axvline(V.shape[2]-1, ls=':', c='grey', alpha=0.5)
    ax.axvline(1, ls=':', c='orange', alpha=0.5)
    ax.axvline(V.shape[2]-2, ls=':', c='orange', alpha=0.5)
    ax.set_title("V(y) at col 20, t = 25 ms  (diffusion only)", fontsize=11)
    ax.set_xlabel("y index")
    ax.set_ylabel("V (mV)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)

    fig.suptitle(
        "Horizontal-redirect BC diagnosis — wall channel is REAL designed behavior; "
        "no mass leak;\nbuggy vs fixed are nearly identical; sub-edge dip is BC-mechanical "
        "(persists under pure diffusion)",
        fontsize=12,
    )
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT.name}")


if __name__ == "__main__":
    main()
