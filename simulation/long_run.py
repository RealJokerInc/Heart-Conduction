"""Long-time simulation: run both rules for many steps, look for steady state.

Outputs:
    outputs/long_isochrones.png        end-state isochrone (one panel per rule)
    outputs/long_front_shape.png       x_front(y) vs y at multiple times
    outputs/long_centerline.png        V(x, y=Ny/2) vs x at multiple times
    outputs/long_activity.png          per-step activity ||dV||_2 vs time
    outputs/long_summary.txt           steady-state diagnostics
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import tanks_vec
from tanks_vec import _shift, _valid_mask, MOORE_8


def run_long(mode: str, steps: int, snap_every: int,
             Nx: int, Ny: int, threshold: float,
             max_pump: float, gradient_k: float):
    inlet_cells = [(0, y) for y in range(Ny)]
    outlet_cells = [(Nx - 1, y) for y in range(Ny)]
    inlet_mask = np.zeros((Ny, Nx), dtype=bool)
    outlet_mask = np.zeros((Ny, Nx), dtype=bool)
    for x, y in inlet_cells:
        inlet_mask[y, x] = True
    for x, y in outlet_cells:
        outlet_mask[y, x] = True

    valid = {d: _valid_mask(Ny, Nx, *d) for d in MOORE_8}
    pumpfactor = np.sqrt(100.0 - threshold)

    V = np.zeros((Ny, Nx), dtype=np.float64)
    iso = np.full((Ny, Nx), -1, dtype=np.int32)
    snaps = []
    snap_steps = []
    activity = np.zeros(steps, dtype=np.float64)

    V_prev = V.copy()
    for step in range(steps):
        flux_in = np.zeros_like(V)
        flux_out = np.zeros_like(V)
        fired = V > threshold
        for dy, dx in MOORE_8:
            v_dst = _shift(V, -dy, -dx)
            gap = V - v_dst
            gate = fired & (gap > 0) & valid[(dy, dx)]
            if mode == "constant":
                base = max_pump * np.sqrt(np.clip(V - threshold, 0.0, None)) / pumpfactor
                base = np.minimum(base, max_pump)
                over = base > np.abs(gap)
                amt = np.where(over, gap / 4.0, base)
                amt = np.where(gate, amt, 0.0)
                amt = np.maximum(amt, 0.0)
            elif mode == "gradient":
                amt = np.where(gate, gradient_k * gap, 0.0)
            flux_out += amt
            flux_in += _shift(amt, dy, dx)

        V_new = V - flux_out + flux_in
        np.clip(V_new, 0.0, 100.0, out=V_new)
        V_new[inlet_mask] = 100.0
        V_new[outlet_mask] = 0.0

        # activity = L2 norm of state change
        activity[step] = float(np.linalg.norm(V_new - V_prev))
        V_prev = V.copy()
        V = V_new

        newly = (V > threshold) & (iso < 0)
        if newly.any():
            iso[newly] = step

        if step % snap_every == 0 or step == steps - 1:
            snaps.append(V.copy())
            snap_steps.append(step)

    return iso, np.array(snaps), np.array(snap_steps), activity


def main():
    Nx, Ny = 80, 50
    steps = 8000
    snap_every = 100
    threshold = 45.0
    max_pump = 10.0
    gradient_k = 0.08

    print(f"Running {steps} steps for both modes...")
    results = {}
    for mode in ("constant", "gradient"):
        print(f"  [{mode}] ...", flush=True)
        results[mode] = run_long(mode, steps, snap_every,
                                 Nx, Ny, threshold, max_pump, gradient_k)
        iso, snaps, snap_steps, activity = results[mode]
        filled = int((iso >= 0).sum())
        max_step = int(iso.max())
        print(f"    filled {filled}/{Nx*Ny}, max iso step = {max_step}, "
              f"final activity = {activity[-1]:.3e}")

    np.savez("outputs/long_run.npz",
             constant_iso=results["constant"][0],
             constant_snaps=results["constant"][1],
             constant_snap_steps=results["constant"][2],
             constant_activity=results["constant"][3],
             gradient_iso=results["gradient"][0],
             gradient_snaps=results["gradient"][1],
             gradient_snap_steps=results["gradient"][2],
             gradient_activity=results["gradient"][3])

    # ---------- plot 1: end-state isochrones ----------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    shared_max = max(np.where(results[m][0] >= 0, results[m][0], 0).max()
                     for m in results)
    shared_max = int(shared_max)
    for ax, mode in zip(axes, ("constant", "gradient")):
        iso = results[mode][0].astype(float)
        iso_plot = np.where(iso >= 0, iso, np.nan)
        im = ax.imshow(iso_plot, origin="upper", cmap="plasma",
                       aspect="equal", vmin=0, vmax=shared_max)
        levels = np.linspace(shared_max * 0.05, shared_max * 0.95, 14)
        ax.contour(iso_plot, levels=levels, colors="white",
                   linewidths=0.7, alpha=0.7)
        ax.axvline(0, color="cyan", lw=2, alpha=0.7)
        ax.set_title(f"{mode} rule  ({steps} steps)", fontsize=12)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.85, label="step of first crossing")
    fig.suptitle(f"Line-source isochrones, {steps} steps", fontsize=13)
    fig.savefig("outputs/long_isochrones.png", dpi=180, bbox_inches="tight")
    print("wrote outputs/long_isochrones.png")

    # ---------- plot 2: per-row x_front(y) at multiple times ----------
    times = (200, 500, 1000, 2000, 3500, 5000, 6500, 8000)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True, constrained_layout=True)
    cmap = plt.cm.viridis(np.linspace(0, 0.95, len(times)))
    for ax, mode in zip(axes, ("constant", "gradient")):
        iso = results[mode][0]
        for k, T in enumerate(times):
            x_front = np.full(Ny, -1, dtype=np.int32)
            for y in range(Ny):
                mask = (iso[y] >= 0) & (iso[y] <= T)
                xs = np.where(mask)[0]
                if xs.size:
                    x_front[y] = xs.max()
            ax.plot(x_front, np.arange(Ny), label=f"T={T}",
                    color=cmap[k], lw=1.5)
        ax.invert_yaxis()
        ax.set_xlabel("x (front position)")
        ax.set_title(f"{mode} rule")
        ax.legend(ncol=2, fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("y")
    fig.suptitle("Wavefront shape vs time per row", fontsize=12)
    fig.savefig("outputs/long_front_shape.png", dpi=160, bbox_inches="tight")
    print("wrote outputs/long_front_shape.png")

    # ---------- plot 3: V(x, y=Ny/2) at multiple times ----------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True, constrained_layout=True)
    for ax, mode in zip(axes, ("constant", "gradient")):
        snaps = results[mode][1]
        snap_steps = results[mode][2]
        # Pick ~7 evenly-spaced snapshot indices
        idx = np.linspace(0, len(snaps) - 1, 7, dtype=int)
        cmap = plt.cm.viridis(np.linspace(0, 0.95, len(idx)))
        for k, i in enumerate(idx):
            ax.plot(snaps[i][Ny // 2, :],
                    label=f"t={snap_steps[i]}", color=cmap[k], lw=1.5)
        ax.axhline(threshold, color="red", ls="--", lw=0.7, alpha=0.5,
                   label=f"threshold = {threshold}")
        ax.set_xlabel("x"); ax.set_title(f"{mode} rule")
        ax.set_ylabel("V")
        ax.legend(ncol=2, fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Volume profile along centerline y={Ny//2}", fontsize=12)
    fig.savefig("outputs/long_centerline.png", dpi=160, bbox_inches="tight")
    print("wrote outputs/long_centerline.png")

    # ---------- plot 4: activity vs time ----------
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for mode, color in (("constant", "C0"), ("gradient", "C1")):
        activity = results[mode][3]
        ax.semilogy(np.arange(steps), activity, label=mode, color=color, lw=0.9)
    ax.set_xlabel("step")
    ax.set_ylabel("‖V(t) − V(t−1)‖₂   (log)")
    ax.set_title("Per-step state change. If this decays to 0 → static steady state")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.savefig("outputs/long_activity.png", dpi=160, bbox_inches="tight")
    print("wrote outputs/long_activity.png")

    # ---------- summary ----------
    lines = [f"Run: line source, {steps} steps, max_pump={max_pump}, "
             f"threshold={threshold}, gradient_k={gradient_k}", ""]
    for mode in ("constant", "gradient"):
        iso, snaps, snap_steps, activity = results[mode]
        # Late-time activity statistics
        tail = activity[max(0, steps - 1000):]
        v_late = snaps[-1]
        v_first_late = snaps[max(0, len(snaps) - 11)]
        max_drift = float(np.max(np.abs(v_late - v_first_late)))
        lines += [
            f"=== {mode} ===",
            f"  filled tanks       : {int((iso >= 0).sum())}/{Nx*Ny}",
            f"  max isochrone step : {int(iso.max())}",
            f"  activity initial   : {activity[0]:.3e}",
            f"  activity at step 1k: {activity[min(1000, steps-1)]:.3e}",
            f"  activity at step 4k: {activity[min(4000, steps-1)]:.3e}",
            f"  activity final     : {activity[-1]:.3e}",
            f"  activity tail-mean : {tail.mean():.3e}",
            f"  activity tail-std  : {tail.std():.3e}",
            f"  V drift (last 1k)  : max|V(end) - V(end-1k)| = {max_drift:.3e}",
            "",
        ]
    summary = "\n".join(lines)
    open("outputs/long_summary.txt", "w").write(summary)
    print()
    print(summary)


if __name__ == "__main__":
    main()
