"""
Plot Single Action Potential with LRd94
========================================

Displays a single AP showing:
- V (membrane voltage)
- E_Na (sodium reversal potential)
- E_K (potassium reversal potential)
- E_Ca (calcium reversal potential)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from luo_rudy_1994 import LuoRudy1994

print("=" * 60)
print("LRd94 Single Action Potential")
print("=" * 60)

# Create model
model = LuoRudy1994(dt=0.005)
print(f"Model created with dt = {model.dt} ms")

# Run simulation - single stimulus
print("Running 500 ms simulation...")
results = model.run_simulation(
    t_end=500.0,
    stim_times=[10.0],
    stim_duration=1.0,
    stim_amplitude=-80.0
)
print("Simulation complete!")

# Extract data
t = results['t']
V = results['V']
E_Na = results['E_Na']
E_K = results['E_K']
E_Ca = results['E_Ca']

# Calculate APD90
V_rest = V[0]
V_peak = np.max(V)
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
print(f"  E_Na (rest) = {E_Na[0]:.1f} mV")
print(f"  E_K (rest) = {E_K[0]:.1f} mV")
print(f"  E_Ca (rest) = {E_Ca[0]:.1f} mV")

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Plot V and reversal potentials
ax.plot(t, V, 'b-', linewidth=2.5, label='V (membrane)')
ax.plot(t, E_Na, 'r--', linewidth=1.5, label=f'E_Na ({E_Na[0]:.0f} mV)', alpha=0.8)
ax.plot(t, E_K, 'g--', linewidth=1.5, label=f'E_K ({E_K[0]:.0f} mV)', alpha=0.8)
ax.plot(t, E_Ca, 'm--', linewidth=1.5, label=f'E_Ca ({E_Ca[0]:.0f} mV)', alpha=0.8)

# Zero line
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

# Labels and formatting
ax.set_xlabel('Time [ms]', fontsize=14)
ax.set_ylabel('Voltage [mV]', fontsize=14)
ax.set_title(f'LRd94 Action Potential with Reversal Potentials\n'
             f'V_rest={V_rest:.1f} mV, V_peak={V_peak:.1f} mV, APD90={APD90:.1f} ms',
             fontsize=14)
ax.legend(loc='upper right', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 500)
ax.set_ylim(-120, 180)

# Add annotations
# Mark E_Na
ax.annotate('Na+ equilibrium', xy=(300, E_Na[0]), xytext=(350, E_Na[0]+20),
            fontsize=10, color='red',
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))

# Mark E_Ca
ax.annotate('Ca2+ equilibrium', xy=(300, E_Ca[0]), xytext=(350, E_Ca[0]-30),
            fontsize=10, color='purple',
            arrowprops=dict(arrowstyle='->', color='purple', alpha=0.5))

# Mark E_K
ax.annotate('K+ equilibrium', xy=(300, E_K[0]), xytext=(350, E_K[0]-20),
            fontsize=10, color='green',
            arrowprops=dict(arrowstyle='->', color='green', alpha=0.5))

plt.tight_layout()

# Save figure
output_path = os.path.join(os.path.dirname(__file__), 'lrd94_single_ap.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {output_path}")

# Show
plt.show()

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
