"""Test the operator-level argument: replace zero-pad BC with reflection BC.

Three treatments per rule (constant, gradient):
    baseline    : zero-pad — corner has 3 nbrs, edge 5, interior 8 (current)
    refl-y      : reflect across y boundaries only — edge cells gain ghost
                  inflow from mirror cells; corners get 5 effective inflows.
    refl-all    : reflect across y AND x boundaries — corners get 8 effective
                  inflows; operator becomes translation-invariant in the
                  bulk of the grid (modulo inlet/outlet forcing at x=0, Nx-1).

Mass-conserving: when a source pumps into a ghost destination, the flux
is routed back to the mirror real cell (so the ghost's gain becomes the
mirror real cell's gain).

Outputs:
    outputs/ghost_corner_camel_toe.png   per-column LAT profile, all 6 configs
    outputs/ghost_corner_summary.txt     filled fraction + late-time activity
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from tanks_vec import _shift, _valid_mask, MOORE_8


def fold_ghosts_to_real(flux_p, Ny: int, Nx: int, pad_y: int, pad_x: int) -> np.ndarray:
    """Take a padded flux array, fold ghost-cell contributions back into the
    mirror real cells, and return the (Ny, Nx) real flux."""
    real = flux_p[pad_y:pad_y + Ny, pad_x:pad_x + Nx].copy()
    if pad_y > 0:
        # Top ghost row mirrors real row 1 (in real coordinates)
        # In padded coords, ghost row 0 mirrors padded row 2 -> real row 1
        real[1, :] += flux_p[0, pad_x:pad_x + Nx]
        real[Ny - 2, :] += flux_p[Ny + pad_y, pad_x:pad_x + Nx]
    if pad_x > 0:
        real[:, 1] += flux_p[pad_y:pad_y + Ny, 0]
        real[:, Nx - 2] += flux_p[pad_y:pad_y + Ny, Nx + pad_x]
    if pad_y > 0 and pad_x > 0:
        real[1, 1] += flux_p[0, 0]
        real[1, Nx - 2] += flux_p[0, Nx + pad_x]
        real[Ny - 2, 1] += flux_p[Ny + pad_y, 0]
        real[Ny - 2, Nx - 2] += flux_p[Ny + pad_y, Nx + pad_x]
    return real


def run_with_reflect(
    mode: str,
    steps: int,
    Nx: int,
    Ny: int,
    inlet_cells,
    outlet_cells,
    reflect_axes=(),
    threshold: float = 45.0,
    max_volume: float = 100.0,
    max_pump: float = 10.0,
    gradient_k: float = 0.08,
):
    pad_y = 1 if "y" in reflect_axes else 0
    pad_x = 1 if "x" in reflect_axes else 0
    pumpfactor = np.sqrt(max_volume - threshold)

    inlet_mask = np.zeros((Ny, Nx), dtype=bool)
    outlet_mask = np.zeros((Ny, Nx), dtype=bool)
    for x, y in inlet_cells:
        inlet_mask[y, x] = True
    for x, y in outlet_cells:
        outlet_mask[y, x] = True

    V = np.zeros((Ny, Nx), dtype=np.float64)
    iso = np.full((Ny, Nx), -1, dtype=np.int32)

    for step in range(steps):
        # Build padded V via reflection on requested axes
        if pad_y > 0 or pad_x > 0:
            Vp = np.pad(V, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
        else:
            Vp = V

        Ny_p, Nx_p = Vp.shape
        flux_in_p = np.zeros_like(Vp)
        flux_out_p = np.zeros_like(Vp)
        fired_p = Vp > threshold

        for dy, dx in MOORE_8:
            valid = _valid_mask(Ny_p, Nx_p, dy, dx)
            V_dst = _shift(Vp, -dy, -dx)
            gap = Vp - V_dst
            gate = fired_p & (gap > 0) & valid

            if mode == "constant":
                base = max_pump * np.sqrt(np.clip(Vp - threshold, 0.0, None)) / pumpfactor
                base = np.minimum(base, max_pump)
                over = base > np.abs(gap)
                amt = np.where(over, gap / 4.0, base)
                amt = np.where(gate, amt, 0.0)
                amt = np.maximum(amt, 0.0)
            elif mode == "gradient":
                amt = np.where(gate, gradient_k * gap, 0.0)
            else:
                raise ValueError(mode)

            flux_out_p += amt
            flux_in_p += _shift(amt, dy, dx)

        # Update real cells: take net flux at real cells AND fold ghost cell
        # flux_in into mirror real cells (mass-conserving reflection BC)
        if pad_y > 0 or pad_x > 0:
            real_in = fold_ghosts_to_real(flux_in_p, Ny, Nx, pad_y, pad_x)
            real_out = flux_out_p[pad_y:pad_y + Ny, pad_x:pad_x + Nx]
        else:
            real_in = flux_in_p
            real_out = flux_out_p

        V = V - real_out + real_in
        np.clip(V, 0.0, max_volume, out=V)
        V[inlet_mask] = max_volume
        V[outlet_mask] = 0.0

        newly = (V > threshold) & (iso < 0)
        if newly.any():
            iso[newly] = step

    return V, iso


def main():
    Nx, Ny, steps = 80, 50, 4000
    inlet_cells = [(0, y) for y in range(Ny)]
    outlet_cells = [(Nx - 1, y) for y in range(Ny)]

    treatments = [
        ("baseline (3-5-8)", ()),
        ("refl-y (5-8-8)", ("y",)),
        ("refl-all (8-8-8)", ("y", "x")),
    ]
    modes = ("constant", "gradient")

    isos = {}
    for tname, axes in treatments:
        for mode in modes:
            print(f"  [{mode} | {tname}] running...", flush=True)
            _, iso = run_with_reflect(
                mode, steps, Nx, Ny, inlet_cells, outlet_cells,
                reflect_axes=axes,
            )
            isos[(mode, tname)] = iso
            filled = int((iso >= 0).sum())
            print(f"      filled={filled}/{Nx*Ny}, max_step={int(iso.max())}")

    # Plot per-column LAT profile, 2 rows (mode) x 3 cols (treatment)
    sample_cols = (3, 8, 18, 30, 45)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=False,
                             constrained_layout=True)
    for i, mode in enumerate(modes):
        for j, (tname, _) in enumerate(treatments):
            ax = axes[i, j]
            iso = isos[(mode, tname)]
            cmap = plt.cm.viridis(np.linspace(0, 0.9, len(sample_cols)))
            for k, c in enumerate(sample_cols):
                col = iso[:, c].astype(float)
                col = np.where(col >= 0, col, np.nan)
                if np.all(np.isnan(col)):
                    ax.plot([], [], color=cmap[k], label=f"x={c} (nr)")
                    continue
                mean = float(np.nanmean(col))
                ax.plot(np.arange(Ny), col - mean, color=cmap[k], lw=1.4,
                        label=f"x={c}")
            ax.axhline(0, color="gray", lw=0.5)
            ax.grid(alpha=0.3)
            ax.set_title(f"{mode} | {tname}", fontsize=10)
            ax.legend(fontsize=7, ncol=2, loc="best")
            if j == 0:
                ax.set_ylabel("iso[y, x] − col-mean")
            if i == 1:
                ax.set_xlabel("y (row)")
    fig.suptitle(
        "Per-column LAT profile under different boundary treatments\n"
        "U-shape = camel toe (boundary speedup),  inverted-U = crescent (boundary slowdown)",
        fontsize=12,
    )
    fig.savefig("outputs/ghost_corner_camel_toe.png", dpi=180,
                bbox_inches="tight")
    print("wrote outputs/ghost_corner_camel_toe.png")

    # Summary
    lines = ["Treatment      | mode      | filled    | max_step | edge−mid @ x=18"]
    lines.append("-" * 80)
    for tname, _ in treatments:
        for mode in modes:
            iso = isos[(mode, tname)]
            filled = int((iso >= 0).sum())
            max_step = int(iso.max())
            col = iso[:, 18].astype(float)
            col = np.where(col >= 0, col, np.nan)
            edge = 0.5 * (col[0] + col[-1])
            mid = col[Ny // 2]
            delta = edge - mid if not (np.isnan(edge) or np.isnan(mid)) else float("nan")
            sign = ("LEADS" if delta < 0 else "lags" if delta > 0 else "equal") if not np.isnan(delta) else "—"
            lines.append(f"{tname:18}| {mode:10}| {filled:4d}/{Nx*Ny}| {max_step:8d} | "
                         f"Δ = {delta:+6.1f} ({sign})")
    summary = "\n".join(lines) + "\n"
    open("outputs/ghost_corner_summary.txt", "w").write(summary)
    print()
    print(summary)


if __name__ == "__main__":
    main()
