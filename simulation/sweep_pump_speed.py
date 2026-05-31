# -*- coding: utf-8 -*-
"""Test whether pumping speed modulates the camel-toe vs crescent balance.

Hypothesis:
    Slower per-channel pump in CONSTANT mode → drainage effect dominates more
    strongly → larger / more persistent camel toe.
    GRADIENT mode is self-limiting at any k → crescent always.

Sweep:
    constant rule:  max_pump ∈ {2, 5, 10, 15, 20, 30}
    gradient rule:  gradient_k ∈ {0.02, 0.04, 0.08, 0.12, 0.16}

After each individual run is logged into outputs/experiments/, this script
also produces a single comparison plot side-by-side at outputs/sweep_pump_speed.png
showing the per-column LAT for every sweep value.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from configs import BASELINE, GRADIENT, make
from experiment import run_experiment


def main():
    constant_pumps = (2.0, 5.0, 10.0, 15.0, 20.0, 30.0)
    gradient_ks = (0.02, 0.04, 0.08, 0.12, 0.16)

    constant_isos = {}
    for mp in constant_pumps:
        cfg = make({"name": f"sweep_const_maxpump_{mp:04.1f}",
                    "description": f"Constant rule, max_pump = {mp}",
                    "tags": ["sweep", "max_pump", "constant"],
                    "rule": {"max_pump": mp}}, base=BASELINE)
        run_dir = run_experiment(cfg)
        iso = np.load(run_dir / "iso.npz")["iso"]
        constant_isos[mp] = iso

    gradient_isos = {}
    for k in gradient_ks:
        cfg = make({"name": f"sweep_grad_k_{k:.3f}",
                    "description": f"Gradient rule, gradient_k = {k}",
                    "tags": ["sweep", "gradient_k", "gradient"],
                    "rule": {"gradient_k": k}}, base=GRADIENT)
        run_dir = run_experiment(cfg)
        iso = np.load(run_dir / "iso.npz")["iso"]
        gradient_isos[k] = iso

    # Side-by-side figure: LAT profile at column 18 for every sweep value
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    Ny = 50
    sample_col = 18

    cmap_c = plt.cm.plasma(np.linspace(0.05, 0.85, len(constant_pumps)))
    for k, mp in enumerate(constant_pumps):
        iso = constant_isos[mp]
        col = iso[:, sample_col].astype(float)
        col = np.where(col >= 0, col, np.nan)
        if np.all(np.isnan(col)):
            continue
        axes[0].plot(np.arange(Ny), col - float(np.nanmean(col)),
                     color=cmap_c[k], lw=1.6, label=f"max_pump={mp:.0f}")
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].grid(alpha=0.3)
    axes[0].set_xlabel("y (row)")
    axes[0].set_ylabel("LAT − col-mean   (negative = fires earlier)")
    axes[0].set_title(f"constant rule, x={sample_col}", fontsize=11)
    axes[0].legend(fontsize=9)

    cmap_g = plt.cm.viridis(np.linspace(0.05, 0.85, len(gradient_ks)))
    for k, gk in enumerate(gradient_ks):
        iso = gradient_isos[gk]
        col = iso[:, sample_col].astype(float)
        col = np.where(col >= 0, col, np.nan)
        if np.all(np.isnan(col)):
            continue
        axes[1].plot(np.arange(Ny), col - float(np.nanmean(col)),
                     color=cmap_g[k], lw=1.6, label=f"k={gk:.3f}")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].grid(alpha=0.3)
    axes[1].set_xlabel("y (row)")
    axes[1].set_title(f"gradient rule, x={sample_col}", fontsize=11)
    axes[1].legend(fontsize=9)

    fig.suptitle("Pumping-speed sweep — does drainage effect strengthen as pump slows?",
                 fontsize=12)
    fig.savefig("outputs/sweep_pump_speed.png", dpi=160, bbox_inches="tight")
    print("\nwrote outputs/sweep_pump_speed.png")

    # Numeric summary
    print()
    print(f"{'rule':10} {'param':>8}  {'top':>5} {'mid':>5} {'bot':>5}  {'edge−mid':>9}  shape")
    print("-" * 70)
    for mp in constant_pumps:
        iso = constant_isos[mp]
        col = iso[:, sample_col].astype(float)
        col = np.where(col >= 0, col, np.nan)
        if np.all(np.isnan(col)):
            print(f"{'constant':10} {mp:>8.1f}  not reached")
            continue
        top, mid, bot = col[0], col[Ny // 2], col[-1]
        delta = 0.5 * (top + bot) - mid
        shape = "CAMEL" if delta < -1 else "crescent" if delta > 1 else "flat"
        print(f"{'constant':10} {mp:>8.1f}  {top:>5.0f} {mid:>5.0f} {bot:>5.0f}  {delta:>+9.1f}  {shape}")
    for gk in gradient_ks:
        iso = gradient_isos[gk]
        col = iso[:, sample_col].astype(float)
        col = np.where(col >= 0, col, np.nan)
        if np.all(np.isnan(col)):
            print(f"{'gradient':10} {gk:>8.3f}  not reached")
            continue
        top, mid, bot = col[0], col[Ny // 2], col[-1]
        delta = 0.5 * (top + bot) - mid
        shape = "CAMEL" if delta < -1 else "crescent" if delta > 1 else "flat"
        print(f"{'gradient':10} {gk:>8.3f}  {top:>5.0f} {mid:>5.0f} {bot:>5.0f}  {delta:>+9.1f}  {shape}")


if __name__ == "__main__":
    main()
