"""Per-column firing-time pattern: does each column fill 'edges first'?

For a planar wavefront moving rightward, each column N gets crossed at some
mean time. The QUESTION is: within column N, does iso[y=0, N] fire
*before* iso[y=25, N]? If yes -> 'camel toe' filling pattern.

Re-runs line / constant with John's effective params (max_pump=10).
"""

import numpy as np
import matplotlib.pyplot as plt

import tanks_vec


def main():
    Nx, Ny, steps = 80, 50, 2000
    threshold = 45.0
    inlet_cells = [(0, y) for y in range(Ny)]
    outlet_cells = [(Nx - 1, y) for y in range(Ny)]

    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "constant"
    out = tanks_vec.run(
        Nx, Ny, mode, steps,
        inlet_cells=inlet_cells,
        outlet_cells=outlet_cells,
        threshold=threshold,
        max_pump=10.0,
        gradient_k=0.08,
    )
    iso = out["iso"]
    print(f"\n=== mode={mode} ===")

    # For each column N, get iso[y, N] for every y. The wavefront 'arrives'
    # over many steps; we want to see the firing-order across rows within the
    # column.
    sample_cols = [1, 3, 5, 8, 12, 18, 25, 35, 45]
    print(f"{'col':>4}  {'mean_t':>7}  {'iso vs y (rounded to nearest 5 rows)':<50}  pattern")
    print("-" * 110)
    for c in sample_cols:
        col = iso[:, c].astype(float)
        col = np.where(col >= 0, col, np.nan)
        if np.all(np.isnan(col)):
            print(f"{c:>4}  not-reached")
            continue
        # Subtract the row-mean to highlight intra-column shape
        mean_t = float(np.nanmean(col))
        dev = col - mean_t
        sample_ys = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49)
        sample_str = "  ".join(f"y={y}:{int(col[y]) if not np.isnan(col[y]) else '-':>4}" for y in sample_ys)
        pattern = ("EDGES first (camel toe)"
                   if (col[0] < col[Ny // 2] and col[-1] < col[Ny // 2])
                   else "MIDDLE first (interior bulge)"
                   if (col[0] > col[Ny // 2] and col[-1] > col[Ny // 2])
                   else "mixed")
        print(f"{c:>4}  {mean_t:>7.1f}  {sample_str}    {pattern}")

    # Plot iso[y, c] vs y for several columns: does the curve dip at the edges?
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.viridis(np.linspace(0, 0.95, len(sample_cols)))
    for k, c in enumerate(sample_cols):
        col = iso[:, c].astype(float)
        col = np.where(col >= 0, col, np.nan)
        if np.all(np.isnan(col)):
            continue
        ax.plot(np.arange(Ny), col, label=f"x={c}", color=cmap[k], lw=1.5)
    ax.set_xlabel("y (row within column)")
    ax.set_ylabel("first-crossing step (lower = fires earlier)")
    ax.set_title("Per-column firing-time vs row\n"
                 "U-shape (low at y=0, y=49; high at y=25) = CAMEL TOE filling")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("outputs/per_column_firing_shape.png", dpi=140)
    print("\nwrote outputs/per_column_firing_shape.png")


if __name__ == "__main__":
    main()
