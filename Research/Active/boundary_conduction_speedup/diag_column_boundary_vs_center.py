"""
Diagnostic: for a single column at x = LX/2 in the face_mirror benchmark,
record V(t) at j = 0 (top boundary), j = Ny//2 (center), j = Ny-1 (bottom
boundary). Print and plot:
  - V_max at each j
  - Time of threshold crossing (V > -40 mV) at each j
  - Max |V_top - V_center| over time

Analytical prediction (this code is the empirical check):
With strict y-uniform line stim and y-uniform initial V, the top/bottom/center
cells in any column SHOULD have identical V(t) under face_mirror. Both BC
modes give L*V identical for y-uniform fields. Any nonzero difference is
floating-point round-off, not a real boundary effect.

If we DO see a difference larger than ~1e-12 mV, that's interesting and
contradicts the analysis.
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


# ---------- config (matches video_boundary_modes.py) ----------
LX, LY = 1.0, 0.5
DX = 0.025
NX = int(round(LX / DX)) + 1   # 41
NY = int(round(LY / DX)) + 1   # 21
DT = 0.02
T_END = 25.0
SAVE_EVERY = 0.05  # 20x finer than the 4-mode video, to catch sub-ms shifts
LAT_THRESH = -40.0


def run_face_mirror():
    grid = StructuredGrid.create_rectangle(LX, LY, NX, NY)
    fdm = FDMDiscretization(grid, D=0.001, chi=1.0, Cm=1.0, boundary_mode='face_mirror')
    proto = StimulusProtocol()
    proto.add_stimulus(
        region=lambda x, y: x < 0.05,
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


def lat(V_t, times):
    """Threshold crossing time for a single 1D trace V(t). NaN if never crossed."""
    above = V_t >= LAT_THRESH
    if not above.any():
        return float('nan')
    idx = np.argmax(above)
    if idx == 0:
        return times[0]
    # Linear interpolation between samples
    v0, v1 = V_t[idx-1], V_t[idx]
    t0, t1 = times[idx-1], times[idx]
    return t0 + (LAT_THRESH - v0) * (t1 - t0) / (v1 - v0)


print("Running face_mirror simulation...", flush=True)
times, V_field = run_face_mirror()
print(f"  V_min={V_field.min():.4f} mV, V_max={V_field.max():.4f} mV")

# Pick the middle column (x = LX/2) — far from x-wall effects
i_mid = NX // 2
xs = np.linspace(0, LX, NX)
ys = np.linspace(0, LY, NY)
print(f"\nColumn at i={i_mid}, x={xs[i_mid]:.3f} cm")

j_top = 0          # top boundary
j_ctr = NY // 2    # center (j=10 for NY=21)
j_bot = NY - 1     # bottom boundary

V_top = V_field[:, i_mid, j_top]   # (n_frames,)
V_ctr = V_field[:, i_mid, j_ctr]
V_bot = V_field[:, i_mid, j_bot]

# Numerical comparison
diff_top_ctr = V_top - V_ctr
diff_bot_ctr = V_bot - V_ctr
diff_top_bot = V_top - V_bot

print(f"\n=== V_max at each j (column i={i_mid}) ===")
print(f"  j=0      (top boundary):    V_max = {V_top.max():+.6f} mV  at t = {times[V_top.argmax()]:.3f} ms")
print(f"  j={j_ctr:<2d}      (center):           V_max = {V_ctr.max():+.6f} mV  at t = {times[V_ctr.argmax()]:.3f} ms")
print(f"  j={j_bot}      (bottom boundary): V_max = {V_bot.max():+.6f} mV  at t = {times[V_bot.argmax()]:.3f} ms")

print(f"\n=== max |V[j_a] - V[j_b]| across the trace ===")
print(f"  max|V_top - V_ctr| = {np.abs(diff_top_ctr).max():.6e} mV")
print(f"  max|V_bot - V_ctr| = {np.abs(diff_bot_ctr).max():.6e} mV")
print(f"  max|V_top - V_bot| = {np.abs(diff_top_bot).max():.6e} mV")

print(f"\n=== LAT (V crosses {LAT_THRESH} mV, linear-interp between samples) ===")
lat_top = lat(V_top, times)
lat_ctr = lat(V_ctr, times)
lat_bot = lat(V_bot, times)
print(f"  LAT(j=0)  = {lat_top:.6f} ms")
print(f"  LAT(j={j_ctr}) = {lat_ctr:.6f} ms")
print(f"  LAT(j={j_bot}) = {lat_bot:.6f} ms")
print(f"  LAT(top) - LAT(ctr) = {(lat_top - lat_ctr)*1000:+.3f} µs")
print(f"  LAT(bot) - LAT(ctr) = {(lat_bot - lat_ctr)*1000:+.3f} µs")
print(f"  LAT(top) - LAT(bot) = {(lat_top - lat_bot)*1000:+.3f} µs")

# Plot
out = Path("/home/norepinephrine/Documents/Heart-Conduction/Research/Active/"
           "boundary_conduction_speedup/figures/diag_column_boundary_vs_center.png")
out.parent.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

ax0 = axes[0]
ax0.plot(times, V_top, label=f'j=0 (top boundary, y={ys[j_top]:.3f})', lw=1.5)
ax0.plot(times, V_ctr, label=f'j={j_ctr} (center, y={ys[j_ctr]:.3f})', lw=1.5, ls='--')
ax0.plot(times, V_bot, label=f'j={j_bot} (bottom boundary, y={ys[j_bot]:.3f})', lw=1.5, ls=':')
ax0.axhline(LAT_THRESH, color='gray', lw=0.5, label=f'LAT thresh = {LAT_THRESH} mV')
ax0.set_xlabel('t (ms)')
ax0.set_ylabel('V (mV)')
ax0.set_title(f'Column at x = {xs[i_mid]:.3f} cm — face_mirror, line stim x<0.05')
ax0.legend()
ax0.grid(alpha=0.3)

ax1 = axes[1]
ax1.plot(times, diff_top_ctr, label='V[j=0] - V[j=center]', lw=1.5)
ax1.plot(times, diff_bot_ctr, label='V[j=bot] - V[j=center]', lw=1.5, ls='--')
ax1.plot(times, diff_top_bot, label='V[j=0] - V[j=bot]', lw=1.5, ls=':')
ax1.axhline(0, color='gray', lw=0.5)
ax1.set_xlabel('t (ms)')
ax1.set_ylabel('ΔV (mV)')
ax1.set_title('Boundary minus center — should be ~0 for y-uniform line stim')
ax1.legend()
ax1.grid(alpha=0.3)

fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"\n[plot] saved {out}")
