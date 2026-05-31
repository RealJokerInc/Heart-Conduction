"""Test John's pump rule with bidirectional pipes (gate `V_src > V_dst` removed).

A→B fires whenever V_A > threshold (regardless of V_B), with amount f(V_A).
The symmetric pipe B→A fires whenever V_B > threshold, with amount f(V_B).
When both endpoints are above threshold, both directions fire simultaneously and
the *net* flow is f(V_A) − f(V_B) — Fickian-like self-limiting behaviour.

Compare:
    one-way constant rule (John's original)
    bidirectional constant rule (this test)
    gradient rule (the diffusion analog)
on the same line-source geometry.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from tanks_vec import _shift, _valid_mask, MOORE_8


def run_bidir(mode: str, steps: int, Nx: int, Ny: int,
              inlet_cells, outlet_cells,
              threshold: float = 45.0, max_volume: float = 100.0,
              max_pump: float = 10.0, gradient_k: float = 0.08,
              bidirectional: bool = False):
    """Run sim. mode is 'constant' or 'gradient'. If bidirectional=True for constant,
    drop the V_src > V_dst gate (both directions of each pipe fire when their source
    is above threshold). bidirectional has no effect for gradient (it's already symmetric)."""
    inlet_mask = np.zeros((Ny, Nx), dtype=bool)
    outlet_mask = np.zeros((Ny, Nx), dtype=bool)
    for x, y in inlet_cells:
        inlet_mask[y, x] = True
    for x, y in outlet_cells:
        outlet_mask[y, x] = True

    valid = {d: _valid_mask(Ny, Nx, *d) for d in MOORE_8}
    pumpfactor = np.sqrt(max_volume - threshold)

    V = np.zeros((Ny, Nx), dtype=np.float64)
    iso = np.full((Ny, Nx), -1, dtype=np.int32)

    for step in range(steps):
        flux_in = np.zeros_like(V)
        flux_out = np.zeros_like(V)
        fired = V > threshold

        for dy, dx in MOORE_8:
            v_dst = _shift(V, -dy, -dx)
            gap = V - v_dst
            if mode == "constant":
                if bidirectional:
                    # Drop the V_src > V_dst gate. Pipe fires whenever source above threshold.
                    gate = fired & valid[(dy, dx)]
                else:
                    gate = fired & (gap > 0) & valid[(dy, dx)]
                base = max_pump * np.sqrt(np.clip(V - threshold, 0.0, None)) / pumpfactor
                base = np.minimum(base, max_pump)
                # Damping cap: only meaningful when gap > 0
                # When bidirectional and gap < 0, base > |gap| can be true with
                # positive base and negative gap. We cap at 0 in that case (the
                # back-pipe fires symmetrically, which we account for in the
                # opposite direction).
                if bidirectional:
                    over = base > np.abs(gap)
                    amt = np.where(over, np.maximum(gap / 4.0, 0.0), base)
                else:
                    over = base > np.abs(gap)
                    amt = np.where(over, gap / 4.0, base)
                amt = np.where(gate, amt, 0.0)
                amt = np.maximum(amt, 0.0)
            elif mode == "gradient":
                gate = fired & (gap > 0) & valid[(dy, dx)]
                amt = np.where(gate, gradient_k * gap, 0.0)
            flux_out += amt
            flux_in += _shift(amt, dy, dx)

        V = V - flux_out + flux_in
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

    runs = [
        ("constant one-way (John)", "constant", False),
        ("constant bidirectional",  "constant", True),
        ("gradient",                "gradient", False),
    ]

    isos = {}
    for label, mode, bidir in runs:
        print(f"  [{label}] running...", flush=True)
        _, iso = run_bidir(mode, steps, Nx, Ny, inlet_cells, outlet_cells,
                           bidirectional=bidir)
        isos[label] = iso
        filled = int((iso >= 0).sum())
        max_step = int(iso.max())
        print(f"    filled={filled}/{Nx*Ny}, max_step={max_step}")

    sample_cols = (3, 8, 18, 30, 45, 60)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True,
                             constrained_layout=True)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(sample_cols)))
    for ax, (label, _, _) in zip(axes, runs):
        iso = isos[label]
        for k, c in enumerate(sample_cols):
            col = iso[:, c].astype(float)
            col = np.where(col >= 0, col, np.nan)
            if np.all(np.isnan(col)):
                ax.plot([], [], color=cmap[k], label=f"x={c} (nr)")
                continue
            ax.plot(np.arange(Ny), col - float(np.nanmean(col)),
                    color=cmap[k], lw=1.5, label=f"x={c}")
        ax.axhline(0, color="gray", lw=0.5)
        ax.grid(alpha=0.3)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8, ncol=2)
        ax.set_xlabel("y (row)")
    axes[0].set_ylabel("iso[y, x] − col-mean")
    fig.suptitle("Per-column LAT — one-way vs bidirectional vs gradient (line source, 4000 steps)",
                 fontsize=12)
    fig.savefig("outputs/bidirectional_camel_toe.png", dpi=180,
                bbox_inches="tight")
    print("wrote outputs/bidirectional_camel_toe.png")

    # Numeric summary at a few columns
    print()
    print(f"{'rule':28} {'col':>4}  {'edge_top':>9} {'mid':>5} {'edge_bot':>9}  {'edge−mid':>9}  shape")
    print("-" * 95)
    for label, _, _ in runs:
        iso = isos[label]
        for c in (8, 18, 30, 45):
            col = iso[:, c].astype(float)
            col = np.where(col >= 0, col, np.nan)
            if np.all(np.isnan(col)):
                continue
            top = col[0]; mid = col[Ny // 2]; bot = col[-1]
            edge = 0.5 * (top + bot)
            delta = edge - mid
            shape = ("CAMEL" if delta < -1 else "crescent" if delta > 1 else "flat")
            print(f"{label:28} {c:>4}  {top:>9.0f} {mid:>5.0f} {bot:>9.0f}  {delta:>+9.1f}  {shape}")


if __name__ == "__main__":
    main()
