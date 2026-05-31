"""
Benchmark: does uniform-y initialization develop a non-uniform wavefront
under each FDM boundary mode?

Setup
-----
- Rectangular tissue 1.0 cm x 0.5 cm at dx=dy=0.025 cm  (Nx=41, Ny=21)
- TTP06 EPI, V_rest = -85.23 mV everywhere (genuinely uniform IC)
- Line stimulus on left edge:  region = (x < 0.05),  uniform in y
- 25 ms run, save every 0.5 ms, Strang + Rush-Larsen + Crank-Nicolson + PCG
- chi=1.0, Cm=1.0, D=0.001 cm^2/ms  (operator convention: D contains chi*Cm)

Math-card prediction (Research/Active/boundary_conduction_speedup/bc_discretization_math.tex)
  For strict y-uniform fields (eps = V[i,1]-V[i,0] = 0):
    face_mirror          : L_y = 0           -> y-uniformity preserved
    node_mirror_existing : L_y = 2*0 = 0     -> y-uniformity preserved
    zero_pad             : L_y = 0 - V[i,0]  -> top/bottom rows feel extra
                                                forcing -V_C at every step,
                                                breaks y-uniformity instantly

Output
------
Research/Active/boundary_conduction_speedup/figures/uniform_init_lat_dev.png
  Row 1: V(x,y) snapshot at t=12 ms for each of 3 modes
  Row 2: LAT(x,y) - mean_y(LAT(x,y))    deviation map per mode
  Row 3: dev curves at sampled x-columns for each mode

Single number per mode: max |LAT_dev| across the field.
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Put V5.4 on the path
ENGINE = Path("/home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.4")
sys.path.insert(0, str(ENGINE))

from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol
from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation


# ---------- config ----------
LX, LY = 1.0, 0.5             # cm
DX = 0.025                    # cm
NX = int(round(LX / DX)) + 1  # 41
NY = int(round(LY / DX)) + 1  # 21
DT = 0.02                     # ms
T_END = 25.0                  # ms
SAVE_EVERY = 0.5              # ms
LAT_THRESH = -40.0            # mV
SNAPSHOT_T = 12.0             # ms (mid-run snapshot for visualization)
SAMPLE_X_FRACS = (0.25, 0.50, 0.75, 0.95)
MODES = ('face_mirror', 'node_mirror_existing', 'zero_pad')

# ---------- run one simulation ----------
def run_one(mode: str):
    print(f"\n=== mode = {mode} ===", flush=True)
    grid = StructuredGrid.create_rectangle(LX, LY, NX, NY)
    fdm = FDMDiscretization(grid, D=0.001, chi=1.0, Cm=1.0, boundary_mode=mode)

    proto = StimulusProtocol()
    proto.add_stimulus(
        region=lambda x, y: x < 0.05,   # uniform in y
        start_time=0.0,
        duration=2.0,
        amplitude=-52.0,
    )

    sim = MonodomainSimulation(
        spatial=fdm,
        ionic_model='ttp06',
        stimulus=proto,
        dt=DT,
        splitting='strang',
        ionic_solver='rush_larsen',
        diffusion_solver='crank_nicolson',
        linear_solver='pcg',
        cell_type='EPI',
    )

    times, V_hist = sim.run_to_array(t_end=T_END, save_every=SAVE_EVERY)
    print(f"  V_min={V_hist.min():.2f} mV, V_max={V_hist.max():.2f} mV", flush=True)

    V_field = V_hist.reshape(len(times), NX, NY)  # state stores i*ny+j

    # LAT: first time at each (i,j) where V crosses LAT_THRESH (rising)
    LAT = np.full((NX, NY), np.nan, dtype=np.float64)
    for it, t in enumerate(times):
        crossed = (V_field[it] >= LAT_THRESH) & np.isnan(LAT)
        LAT[crossed] = t
    activated = ~np.isnan(LAT)
    print(f"  activation: {activated.sum()}/{NX*NY} cells "
          f"({100*activated.mean():.1f}%)", flush=True)

    return times, V_field, LAT


# ---------- run all three ----------
results = {mode: run_one(mode) for mode in MODES}


# ---------- analysis ----------
def lat_dev_y(LAT):
    """Return LAT - mean_over_y(LAT) at each (i,j); NaN if any cell in column unactivated."""
    valid = ~np.isnan(LAT)
    col_ok = valid.all(axis=1)                 # (NX,) cols where all y activated
    dev = np.full_like(LAT, np.nan)
    if col_ok.any():
        col_mean = LAT[col_ok, :].mean(axis=1, keepdims=True)
        dev[col_ok, :] = LAT[col_ok, :] - col_mean
    return dev, col_ok


print("\n=== max |LAT - mean_y(LAT)| per mode (over fully-activated columns) ===")
for mode in MODES:
    _, _, LAT = results[mode]
    dev, ok = lat_dev_y(LAT)
    if not ok.any():
        print(f"  {mode:25s}  no columns reached y-everywhere — wave too slow")
        continue
    dev_clean = dev[ok, :]
    print(f"  {mode:25s}  max|dev|={np.abs(dev_clean).max():.4e} ms   "
          f"std={dev_clean.std():.4e} ms   "
          f"cols_ok={ok.sum()}/{NX}")


# ---------- figure ----------
out_dir = Path("/home/norepinephrine/Documents/Heart-Conduction/Research/Active/"
               "boundary_conduction_speedup/figures")
out_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
xs = np.linspace(0, LX, NX)
ys = np.linspace(0, LY, NY)

# Row 1: V snapshot at SNAPSHOT_T
snap_idx = int(round(SNAPSHOT_T / SAVE_EVERY))
v_min, v_max = -90, 35
for c, mode in enumerate(MODES):
    times, V_field, _ = results[mode]
    snap = V_field[min(snap_idx, len(times) - 1)]
    ax = axes[0, c]
    im = ax.imshow(
        snap.T, origin='lower',
        extent=[0, LX, 0, LY], aspect='auto',
        cmap='RdBu_r', vmin=v_min, vmax=v_max,
    )
    ax.set_title(f"{mode}\nV(x,y) at t={SNAPSHOT_T} ms", fontsize=11)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    plt.colorbar(im, ax=ax, label="V (mV)")

# Row 2: LAT deviation map  (LAT - col_mean)
dev_max = 0.0
dev_maps = {}
for mode in MODES:
    _, _, LAT = results[mode]
    dev, ok = lat_dev_y(LAT)
    dev_maps[mode] = (dev, ok)
    dev_clean = dev[ok, :] if ok.any() else np.array([0.0])
    dev_max = max(dev_max, np.abs(dev_clean).max() if dev_clean.size else 0.0)
dev_max = max(dev_max, 1e-9)

for c, mode in enumerate(MODES):
    dev, ok = dev_maps[mode]
    ax = axes[1, c]
    im = ax.imshow(
        dev.T, origin='lower',
        extent=[0, LX, 0, LY], aspect='auto',
        cmap='RdBu_r', vmin=-dev_max, vmax=+dev_max,
    )
    ax.set_title(f"{mode}\nLAT - mean_y(LAT)   max|.|={np.nanmax(np.abs(dev)):.3e} ms", fontsize=10)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    plt.colorbar(im, ax=ax, label="dev (ms)")

# Row 3: dev profile vs y at sampled x-columns
sample_x_idx = [int(round(f * (NX - 1))) for f in SAMPLE_X_FRACS]
for c, mode in enumerate(MODES):
    dev, ok = dev_maps[mode]
    ax = axes[2, c]
    cmap = plt.cm.viridis
    for k, xi in enumerate(sample_x_idx):
        if not ok[xi]:
            ax.plot([], [], label=f"x={xs[xi]:.2f} cm (not full)")
            continue
        color = cmap(k / max(len(sample_x_idx) - 1, 1))
        ax.plot(ys, dev[xi, :], color=color, lw=1.4,
                label=f"x={xs[xi]:.2f} cm")
    ax.axhline(0, color="gray", lw=0.5)
    ax.grid(alpha=0.3)
    ax.set_xlabel("y (cm)")
    ax.set_ylabel("LAT - mean_y(LAT)  (ms)")
    ax.set_title(f"{mode}\ndev profile", fontsize=10)
    ax.legend(fontsize=8, loc="best")

fig.suptitle(
    f"Uniform-y init -> wavefront y-deviation by FDM boundary mode\n"
    f"TTP06 EPI, {LX}x{LY} cm, dx={DX} cm, t_end={T_END} ms, line-stim at x<0.05",
    fontsize=12,
)

out_path = out_dir / "uniform_init_lat_dev.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"\n[plot] saved {out_path}")
