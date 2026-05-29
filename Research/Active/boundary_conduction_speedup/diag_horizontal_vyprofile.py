"""Step 2 of PLAN.md — V(y) profile traces.

Plots V vs y for HBB, specular, and horizontal at columns 3, 10, 20, 38
and times t = 5, 10, 15, 25 ms. Reveals the structure of the wall-channel
depolarization (j=0, j=NY-1) and the sub-edge dip (j=1, j=NY-2).

Output: figures/horizontal_vy_profiles.png
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
OUT = ROOT / "figures" / "horizontal_vy_profiles.png"

CASES = {
    "HBB":        ("case10_lbm_d2q9_canonical_hbb_natural.h5",   "tab:blue"),
    "specular":   ("case9_lbm_d2q9_canonical_specular_natural.h5", "tab:green"),
    "horizontal": ("case13_lbm_d2q9_canonical_horizontal_natural.h5", "tab:red"),
}

COLS = [3, 10, 20, 38]
TIMES_MS = [5.0, 10.0, 15.0, 25.0]


def main():
    data = {}
    for name, (fname, color) in CASES.items():
        with h5py.File(DATA / fname, "r") as f:
            V = f["V"][:]
            t = f["t"][:]
        data[name] = (t, V, color)

    NY = data["HBB"][1].shape[2]
    y_idx = np.arange(NY)

    fig, axes = plt.subplots(len(COLS), len(TIMES_MS),
                              figsize=(15, 11),
                              sharey=True, sharex=True,
                              constrained_layout=True)

    for r, col in enumerate(COLS):
        for c, t_target in enumerate(TIMES_MS):
            ax = axes[r, c]
            for name, (t, V, color) in data.items():
                k = int(np.argmin(np.abs(t - t_target)))
                profile = V[k, col, :]
                ax.plot(y_idx, profile, label=name, color=color, lw=1.6, marker='.', ms=4)
            # Mark wall and sub-edge rows
            ax.axvline(0,    ls=":", c="grey", alpha=0.5)
            ax.axvline(NY-1, ls=":", c="grey", alpha=0.5)
            ax.axvline(1,    ls=":", c="orange", alpha=0.4)
            ax.axvline(NY-2, ls=":", c="orange", alpha=0.4)
            ax.axhline(-85.23, ls="--", c="black", alpha=0.4, lw=0.8, label="_V_rest" if (r, c) == (0, 0) else None)
            ax.set_title(f"col {col}, t={t_target:.0f} ms", fontsize=10)
            ax.grid(True, alpha=0.3)
            if r == len(COLS) - 1:
                ax.set_xlabel("y index")
            if c == 0:
                ax.set_ylabel("V (mV)")
            if (r, c) == (0, 0):
                ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("V(y) profiles across BCs — walls at y=0 (grey), sub-edge at y=1 (orange). "
                 "Dashed black = V_rest. ",
                 fontsize=12)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT.name}")

    # ─── Console summary: numerical V at wall and sub-edge ───────────
    print("\n=== V at wall (j=0) and sub-edge (j=1), col 20, across BCs ===")
    print(f"  {'BC':<12}  " + "  ".join(f"{'t='+str(int(t))+'ms':>12}" for t in TIMES_MS))
    for name, (t, V, _) in data.items():
        wall_vals = []
        for t_target in TIMES_MS:
            k = int(np.argmin(np.abs(t - t_target)))
            wall = V[k, 20, 0]
            sub  = V[k, 20, 1]
            wall_vals.append(f"j0={wall:+.1f} j1={sub:+.1f}")
        print(f"  {name:<12}  " + "  ".join(f"{v:>12}" for v in wall_vals))


if __name__ == "__main__":
    main()
