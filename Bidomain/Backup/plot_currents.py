"""
Plot LRd94 Action Potential with Ionic Currents
================================================

Generates a comprehensive plot showing:
- Membrane voltage (V)
- Sodium current (I_Na)
- L-type calcium current (I_Ca_L)
- Potassium currents (I_K, I_K1)
- Calcium concentration ([Ca]_i)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sys
import os

# Add the directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from luo_rudy_1994 import LuoRudy1994

print("=" * 60)
print("LRd94 Action Potential and Ionic Currents")
print("=" * 60)

# Create model
model = LuoRudy1994(dt=0.005)
print(f"Model created with dt = {model.dt} ms")

# Run simulation
print("Running 500 ms simulation...")
results = model.run_simulation(
    t_end=500.0,
    stim_times=[10.0],
    stim_duration=1.0,
    stim_amplitude=-80.0
)
print("Simulation complete!")

# Analysis
t = results['t']
V = results['V']
V_rest = V[0]
V_peak = np.max(V)

# APD90
V_90 = V_rest + 0.1 * (V_peak - V_rest)
above_90 = V > V_90
cross_up = np.where(np.diff(above_90.astype(int)) == 1)[0]
cross_down = np.where(np.diff(above_90.astype(int)) == -1)[0]
if len(cross_up) > 0 and len(cross_down) > 0:
    APD90 = t[cross_down[0]] - t[cross_up[0]]
else:
    APD90 = np.nan

print(f"\nResults:")
print(f"  V_rest = {V_rest:.1f} mV")
print(f"  V_peak = {V_peak:.1f} mV")
print(f"  APD90 = {APD90:.1f} ms")
print(f"  [Ca]_i peak = {np.max(results['Ca_i'])*1e6:.1f} nM")

# Create figure
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Panel 1: Voltage
ax1 = axes[0]
ax1.plot(t, V, 'b-', linewidth=1.5)
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_ylabel('V [mV]', fontsize=12)
ax1.set_title(f'LRd94 Action Potential (V_rest={V_rest:.1f} mV, V_peak={V_peak:.1f} mV, APD90={APD90:.1f} ms)', fontsize=14)
ax1.set_ylim(-100, 80)
ax1.grid(True, alpha=0.3)
ax1.legend(['V_m'], loc='upper right')

# Panel 2: Inward currents (I_Na, I_Ca_L)
ax2 = axes[1]
ax2.plot(t, results['I_Na'], 'b-', label='I_Na', linewidth=1.5)
ax2.plot(t, results['I_Ca_L'], 'r-', label='I_Ca,L', linewidth=1.5)
ax2.set_ylabel('Current [uA/cm²]', fontsize=12)
ax2.set_title('Inward Currents', fontsize=12)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# Add inset for I_Na peak (it's very fast)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
ax_inset = inset_axes(ax2, width="30%", height="40%", loc='right', borderpad=2)
stim_start_idx = int(10.0 / model.dt)
window = int(5.0 / model.dt)  # 5 ms window
ax_inset.plot(t[stim_start_idx:stim_start_idx+window],
              results['I_Na'][stim_start_idx:stim_start_idx+window], 'b-', linewidth=1)
ax_inset.set_xlabel('t [ms]', fontsize=8)
ax_inset.set_ylabel('I_Na', fontsize=8)
ax_inset.set_title('I_Na zoom (5ms)', fontsize=8)
ax_inset.grid(True, alpha=0.3)

# Panel 3: Outward currents (I_K, I_K1)
ax3 = axes[2]
ax3.plot(t, results['I_K'], 'g-', label='I_K', linewidth=1.5)
ax3.plot(t, results['I_K1'], 'm-', label='I_K1', linewidth=1.5)
ax3.plot(t, results['I_Kp'], 'c-', label='I_Kp', linewidth=1, alpha=0.7)
ax3.set_ylabel('Current [uA/cm²]', fontsize=12)
ax3.set_title('Outward Currents (K+)', fontsize=12)
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# Panel 4: Calcium concentration
ax4 = axes[3]
ax4_ca = ax4
ax4_ca.plot(t, results['Ca_i'] * 1e6, 'r-', linewidth=1.5, label='[Ca]_i')
ax4_ca.set_ylabel('[Ca]_i [nM]', fontsize=12, color='r')
ax4_ca.tick_params(axis='y', labelcolor='r')
ax4_ca.set_xlabel('Time [ms]', fontsize=12)
ax4_ca.set_title('Intracellular Calcium', fontsize=12)
ax4_ca.grid(True, alpha=0.3)
ax4_ca.legend(loc='upper right')

plt.tight_layout()

# Save figure
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lrd94_currents.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {output_path}")

# Also create a zoomed view of the upstroke
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))

# Time window for upstroke (8-20 ms)
t_start, t_end_zoom = 8.0, 25.0
idx_start = int(t_start / model.dt)
idx_end = int(t_end_zoom / model.dt)
t_zoom = t[idx_start:idx_end]

# Upstroke voltage
ax = axes2[0, 0]
ax.plot(t_zoom, V[idx_start:idx_end], 'b-', linewidth=2)
ax.set_xlabel('Time [ms]')
ax.set_ylabel('V [mV]')
ax.set_title('Upstroke Phase')
ax.grid(True, alpha=0.3)

# I_Na during upstroke
ax = axes2[0, 1]
ax.plot(t_zoom, results['I_Na'][idx_start:idx_end], 'b-', linewidth=2)
ax.set_xlabel('Time [ms]')
ax.set_ylabel('I_Na [uA/cm²]')
ax.set_title(f'Sodium Current (peak: {np.min(results["I_Na"]):.1f} uA/cm²)')
ax.grid(True, alpha=0.3)

# Plateau phase (50-200 ms)
t_start_p, t_end_p = 50.0, 250.0
idx_start_p = int(t_start_p / model.dt)
idx_end_p = int(t_end_p / model.dt)
t_plateau = t[idx_start_p:idx_end_p]

# I_Ca_L during plateau
ax = axes2[1, 0]
ax.plot(t_plateau, results['I_Ca_L'][idx_start_p:idx_end_p], 'r-', linewidth=2)
ax.set_xlabel('Time [ms]')
ax.set_ylabel('I_Ca,L [uA/cm²]')
ax.set_title('L-type Ca Current during Plateau')
ax.grid(True, alpha=0.3)

# K currents during repolarization
ax = axes2[1, 1]
ax.plot(t_plateau, results['I_K'][idx_start_p:idx_end_p], 'g-', linewidth=2, label='I_K')
ax.plot(t_plateau, results['I_K1'][idx_start_p:idx_end_p], 'm-', linewidth=2, label='I_K1')
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Current [uA/cm²]')
ax.set_title('K Currents during Repolarization')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lrd94_phases.png')
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"Phase detail plot saved: {output_path2}")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
