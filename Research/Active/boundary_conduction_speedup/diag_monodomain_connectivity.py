"""
Monodomain V5.4 column diagnostic across (stencil, boundary_mode) combinations.

Tests the bridge claim from the storage-tank ablation: connectivity creates
boundary deficit, except where the BC handles it (cardinal-4 OR face_mirror_iso
for moore8 stencils). Empirically establishes that monodomain ALSO has the
artifact under the John-equivalent setup (moore8_uniform + face_mirror).

Save_every = 0.025 ms (40× finer than typical) to capture sub-µs LAT shifts.

Five configurations:
  cardinal4_face_mirror       — baseline (existing default since 2026-04-29)
  moore8_uniform_face_mirror  — John-equivalent: deficit appears
  moore8_uniform_face_iso     — bounce-back: deficit eliminated
  moore8_iso_face_mirror      — iso 4:1 alone: deficit reduced (5/6) but present
  moore8_iso_face_iso         — full LBM analog: zero deficit

Output:
  figures/diag_monodomain_connectivity.png
  stdout: numerical summary table
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENGINE = Path("/home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.4")
sys.path.insert(0, str(ENGINE))

from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol
from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation


# ---------- config ----------
LX, LY = 1.0, 0.5             # cm
DX = 0.025                    # cm  (square grid required for moore8)
NX = int(round(LX / DX)) + 1  # 41
NY = int(round(LY / DX)) + 1  # 21
DT = 0.02                     # ms
T_END = 25.0                  # ms
SAVE_EVERY = 0.025            # ms (40x finer than default 1.0)
LAT_THRESH = -40.0            # mV

CASES = [
    ("cardinal4_face_mirror",      "cardinal4",      "face_mirror"),
    ("moore8_uniform_face_mirror", "moore8_uniform", "face_mirror"),
    ("moore8_uniform_face_iso",    "moore8_uniform", "face_mirror_iso"),
    ("moore8_iso_face_mirror",     "moore8_iso",     "face_mirror"),
    ("moore8_iso_face_iso",        "moore8_iso",     "face_mirror_iso"),
]


def run_one(stencil: str, boundary_mode: str):
    grid = StructuredGrid.create_rectangle(LX, LY, NX, NY)
    fdm = FDMDiscretization(
        grid, D=0.001, chi=1.0, Cm=1.0,
        stencil=stencil, boundary_mode=boundary_mode,
    )
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
        diffusion_solver='forward_euler',
        cell_type='EPI',
    )
    times, V_hist = sim.run_to_array(t_end=T_END, save_every=SAVE_EVERY)
    V_field = V_hist.reshape(len(times), NX, NY)
    return times, V_field


def lat_crossing(V_t, times, threshold=LAT_THRESH):
    """Linear-interp threshold crossing time. Returns NaN if never crossed."""
    above = V_t >= threshold
    if not above.any():
        return float('nan')
    idx = int(np.argmax(above))
    if idx == 0:
        return times[0]
    v0, v1 = V_t[idx - 1], V_t[idx]
    t0, t1 = times[idx - 1], times[idx]
    if v1 == v0:
        return t1
    return t0 + (threshold - v0) * (t1 - t0) / (v1 - v0)


print(f"Grid: {NX} × {NY} = {NX*NY} cells, dx={DX} cm, dt={DT} ms, "
      f"t_end={T_END} ms, save_every={SAVE_EVERY} ms\n")

results = {}
for name, stencil, boundary_mode in CASES:
    print(f"[run] {name:<35}  stencil={stencil:<16}  bc={boundary_mode}",
          flush=True)
    times, V_field = run_one(stencil, boundary_mode)
    print(f"        V_min={V_field.min():.3f} mV, V_max={V_field.max():.3f} mV  "
          f"({len(times)} frames)")
    results[name] = (times, V_field)


# Numerical summary table — middle column (i=NX//2), j=0/NY//2/NY-1
print(f"\n{'case':<35} {'V_max_top':>10} {'V_max_ctr':>10} "
      f"{'max|top-ctr|':>14} {'LAT_top':>10} {'LAT_ctr':>10} {'ΔLAT (µs)':>12}")
print("-" * 120)
i_mid = NX // 2
summary_rows = []
for name, _, _ in CASES:
    times, V_field = results[name]
    V_top = V_field[:, i_mid, 0]           # top boundary
    V_ctr = V_field[:, i_mid, NY // 2]      # center
    V_bot = V_field[:, i_mid, NY - 1]       # bottom boundary

    max_dev = float(np.abs(V_top - V_ctr).max())
    lat_top = lat_crossing(V_top, times)
    lat_ctr = lat_crossing(V_ctr, times)
    delta_lat_us = (lat_top - lat_ctr) * 1000.0  # ms -> µs

    print(f"{name:<35} {float(V_top.max()):+10.3f} {float(V_ctr.max()):+10.3f} "
          f"{max_dev:14.3e} {lat_top:10.4f} {lat_ctr:10.4f} {delta_lat_us:+12.4f}")
    summary_rows.append((name, max_dev, delta_lat_us))


# Plot
out_dir = Path("/home/norepinephrine/Documents/Heart-Conduction/Research/Active/"
               "boundary_conduction_speedup/figures")
out_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(len(CASES), 2, figsize=(14, 3.0 * len(CASES)),
                         constrained_layout=True)

for row, (name, _, _) in enumerate(CASES):
    times, V_field = results[name]
    V_top = V_field[:, i_mid, 0]
    V_ctr = V_field[:, i_mid, NY // 2]
    V_bot = V_field[:, i_mid, NY - 1]

    # Left: V(t) traces
    ax_v = axes[row, 0]
    ax_v.plot(times, V_top, label=f'j=0 (top)', lw=1.2)
    ax_v.plot(times, V_ctr, label=f'j={NY//2} (ctr)', lw=1.2, ls='--')
    ax_v.plot(times, V_bot, label=f'j={NY-1} (bot)', lw=1.2, ls=':')
    ax_v.axhline(LAT_THRESH, color='gray', lw=0.5)
    ax_v.set_ylabel('V (mV)')
    ax_v.set_title(name, fontsize=10)
    ax_v.legend(fontsize=8, loc='lower right')
    ax_v.grid(alpha=0.3)
    if row == len(CASES) - 1:
        ax_v.set_xlabel('t (ms)')

    # Right: deviation V_top - V_ctr
    ax_d = axes[row, 1]
    ax_d.plot(times, V_top - V_ctr, label='top - ctr', lw=1.2)
    ax_d.plot(times, V_bot - V_ctr, label='bot - ctr', lw=1.2, ls='--')
    ax_d.axhline(0, color='gray', lw=0.5)
    ax_d.set_ylabel('ΔV (mV)')
    ax_d.set_title(f"{name}: deviation from center", fontsize=10)
    ax_d.legend(fontsize=8, loc='best')
    ax_d.grid(alpha=0.3)
    if row == len(CASES) - 1:
        ax_d.set_xlabel('t (ms)')

fig.suptitle(
    f"Monodomain V5.4 column diagnostic — connectivity × boundary_mode\n"
    f"TTP06 EPI, line stim x<0.05 (y-uniform), dx={DX} cm, save={SAVE_EVERY} ms.\n"
    f"Mid-column (i={i_mid}, x={i_mid*DX:.2f} cm).",
    fontsize=12,
)

out_path = out_dir / "diag_monodomain_connectivity.png"
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n[plot] saved {out_path}")

# Final classification — does the empirical pattern match predictions?
print("\n=== prediction check ===")
for name, max_dev, delta_lat in summary_rows:
    expected_zero = name in ('cardinal4_face_mirror',
                              'moore8_uniform_face_iso',
                              'moore8_iso_face_iso')
    floor = 1e-9  # noise floor for "zero deficit" classification
    if expected_zero:
        verdict = ("✓ AS PREDICTED (no deficit)" if max_dev < floor
                   else f"✗ UNEXPECTED DEVIATION {max_dev:.2e}")
    else:
        verdict = ("✓ AS PREDICTED (deficit present)" if max_dev > floor
                   else f"✗ EXPECTED DEFICIT, GOT {max_dev:.2e}")
    print(f"  {name:<35} {verdict}")
