"""
Phase 2c: Gating Curve Comparison

Generate steady-state gating curves for comparison with literature (ORd 2011 Figure 4).
Output CSV files for plotting.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

def run_gating_curve_analysis():
    print("=" * 70)
    print("PHASE 2c: Gating Curve Analysis")
    print("=" * 70)
    print()

    # Voltage range
    V = np.linspace(-120, 60, 181)  # -120 to +60 mV, 1 mV steps

    def safe_exp(x, limit=80.0):
        return np.exp(np.clip(x, -limit, limit))

    # =========================================================================
    # INa Gating Curves
    # =========================================================================
    print("Computing INa gating curves...")

    # Activation (m)
    m_inf = 1.0 / (1.0 + safe_exp(-(V + 39.57) / 9.871))
    m_tau = 1.0 / (6.765 * safe_exp((V + 11.64) / 34.77) +
                   8.552 * safe_exp(-(V + 77.42) / 5.955))

    # Inactivation (h)
    h_inf = 1.0 / (1.0 + safe_exp((V + 82.90) / 6.086))
    hf_tau = 1.0 / (1.432e-5 * safe_exp(-(V + 1.196) / 6.285) +
                    6.149 * safe_exp((V + 0.5096) / 20.27))
    hs_tau = 1.0 / (0.009794 * safe_exp(-(V + 17.95) / 28.05) +
                    0.3343 * safe_exp((V + 5.730) / 56.66))

    # Recovery (j)
    j_inf = h_inf  # Same as h_inf
    j_tau = 2.038 + 1.0 / (0.02136 * safe_exp(-(V + 100.6) / 8.281) +
                           0.3052 * safe_exp((V + 0.9941) / 38.45))

    # =========================================================================
    # ICaL Gating Curves
    # =========================================================================
    print("Computing ICaL gating curves...")

    # Activation (d)
    d_inf = 1.0 / (1.0 + safe_exp(-(V + 3.940) / 4.230))
    d_tau = 0.6 + 1.0 / (safe_exp(-0.05 * (V + 6.0)) +
                         safe_exp(0.09 * (V + 14.0)))

    # Inactivation (f)
    f_inf = 1.0 / (1.0 + safe_exp((V + 19.58) / 3.696))
    ff_tau = 7.0 + 1.0 / (0.0045 * safe_exp(-(V + 20.0) / 10.0) +
                          0.0045 * safe_exp((V + 20.0) / 10.0))
    fs_tau = 1000.0 + 1.0 / (0.000035 * safe_exp(-(V + 5.0) / 4.0) +
                             0.000035 * safe_exp((V + 5.0) / 6.0))

    # =========================================================================
    # Output directory
    # =========================================================================
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(output_dir, "phase2_data")
    os.makedirs(data_dir, exist_ok=True)

    # =========================================================================
    # Save INa curves
    # =========================================================================
    ina_file = os.path.join(data_dir, "phase2c_ina_curves.csv")
    np.savetxt(ina_file,
               np.column_stack([V, m_inf, m_tau, h_inf, hf_tau, hs_tau, j_tau]),
               delimiter=',',
               header='V_mV,m_inf,m_tau_ms,h_inf,hf_tau_ms,hs_tau_ms,j_tau_ms',
               comments='')
    print(f"Saved: {ina_file}")

    # Save ICaL curves
    ical_file = os.path.join(data_dir, "phase2c_ical_curves.csv")
    np.savetxt(ical_file,
               np.column_stack([V, d_inf, d_tau, f_inf, ff_tau, fs_tau]),
               delimiter=',',
               header='V_mV,d_inf,d_tau_ms,f_inf,ff_tau_ms,fs_tau_ms',
               comments='')
    print(f"Saved: {ical_file}")

    print()
    print("=" * 70)
    print("KEY GATING PARAMETERS")
    print("=" * 70)
    print()

    # Find V_half for each curve
    def find_v_half(V, curve):
        """Find voltage at which curve = 0.5"""
        idx = np.argmin(np.abs(curve - 0.5))
        return V[idx]

    print("STEADY-STATE ACTIVATION/INACTIVATION:")
    print()
    print(f"{'Gate':<10} {'V_half (mV)':<15} {'Slope (mV)':<15} {'Type':<15}")
    print("-" * 55)

    # INa m (activation)
    v_half_m = find_v_half(V, m_inf)
    print(f"{'INa m':<10} {v_half_m:<15.1f} {'9.87':<15} {'activation':<15}")

    # INa h (inactivation)
    v_half_h = find_v_half(V, h_inf)
    print(f"{'INa h':<10} {v_half_h:<15.1f} {'6.09':<15} {'inactivation':<15}")

    # ICaL d (activation)
    v_half_d = find_v_half(V, d_inf)
    print(f"{'ICaL d':<10} {v_half_d:<15.1f} {'4.23':<15} {'activation':<15}")

    # ICaL f (inactivation)
    v_half_f = find_v_half(V, f_inf)
    print(f"{'ICaL f':<10} {v_half_f:<15.1f} {'3.70':<15} {'inactivation':<15}")

    print()
    print("=" * 70)
    print("COMPARISON WITH LITERATURE (ORd 2011)")
    print("=" * 70)
    print()
    print("INa V_half for inactivation:")
    print(f"  Our model: {v_half_h:.1f} mV")
    print(f"  ORd paper: -82.9 mV (specified in formula)")
    print(f"  Experimental (Sakakibara): -76.9 mV at 17°C, adjusted to -82.9 mV at 37°C")
    print()

    print("KEY VALUES AT V_rest = -87.5 mV:")
    v_rest = -87.5
    idx = np.argmin(np.abs(V - v_rest))
    print(f"  m_inf  = {m_inf[idx]:.6f}")
    print(f"  h_inf  = {h_inf[idx]:.6f}")
    print(f"  m_tau  = {m_tau[idx]:.4f} ms")
    print(f"  hf_tau = {hf_tau[idx]:.4f} ms")
    print(f"  hs_tau = {hs_tau[idx]:.4f} ms")
    print(f"  j_tau  = {j_tau[idx]:.4f} ms")
    print()

    print("=" * 70)
    print("WINDOW CURRENT ANALYSIS")
    print("=" * 70)
    print()
    print("INa 'window current' occurs where m_inf and h_inf overlap:")
    print()
    window = m_inf * h_inf
    max_window = np.max(window)
    v_max_window = V[np.argmax(window)]
    print(f"  Maximum window: {max_window:.4f} at V = {v_max_window:.1f} mV")
    print()
    print("  At V = -87.5 mV: m_inf * h_inf = {:.6f}".format(m_inf[idx] * h_inf[idx]))
    print("  At V = -70 mV:   m_inf * h_inf = {:.6f}".format(
        m_inf[np.argmin(np.abs(V + 70))] * h_inf[np.argmin(np.abs(V + 70))]))
    print()

    print("=" * 70)
    print("Phase 2c complete!")
    print("=" * 70)


if __name__ == '__main__':
    run_gating_curve_analysis()
