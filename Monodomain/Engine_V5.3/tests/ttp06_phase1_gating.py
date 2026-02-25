#!/usr/bin/env python3
"""
TTP06 Phase 1: Gating Curves Validation

Validates steady-state (x_inf) and time constant (tau_x) curves against
literature reference values.

Reference values from:
- TNNP model page: https://www.ibiblio.org/e-notes/html5/tnnp.html
- CellML repository: https://models.cellml.org/exposure/de5058f16f829f91a1e4e5990a10ed71

CSV Output:
- phase1_steadystate.csv: Steady-state curves for all gates
- phase1_timeconstants.csv: Time constant curves for all gates
- phase1_validation.csv: Key validation points with pass/fail
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from ionic.ttp06 import gating


def print_progress(current, total, prefix='', width=40):
    """Print progress bar."""
    pct = current / total
    filled = int(width * pct)
    bar = '█' * filled + ' ' * (width - filled)
    print(f'\r  {prefix} [{bar}] {pct*100:5.1f}%', end='', flush=True)


def compute_steadystate_curves(V_range):
    """Compute all steady-state curves over voltage range."""
    n = len(V_range)

    # Storage
    m_inf = np.zeros(n)
    h_inf = np.zeros(n)
    j_inf = np.zeros(n)

    d_inf = np.zeros(n)
    f_inf = np.zeros(n)
    f2_inf = np.zeros(n)

    r_inf = np.zeros(n)
    s_inf_endo = np.zeros(n)
    s_inf_epi = np.zeros(n)

    Xr1_inf = np.zeros(n)
    Xr2_inf = np.zeros(n)
    Xs_inf = np.zeros(n)

    print("Computing steady-state curves...")
    for i, V in enumerate(V_range):
        V_t = torch.tensor([V], dtype=torch.float64)

        # INa gates
        m_inf[i] = gating.INa_m_inf(V_t).item()
        h_inf[i] = gating.INa_h_inf(V_t).item()
        j_inf[i] = gating.INa_j_inf(V_t).item()

        # ICaL gates
        d_inf[i] = gating.ICaL_d_inf(V_t).item()
        f_inf[i] = gating.ICaL_f_inf(V_t).item()
        f2_inf[i] = gating.ICaL_f2_inf(V_t).item()

        # Ito gates
        r_inf[i] = gating.Ito_r_inf(V_t).item()
        s_inf_endo[i] = gating.Ito_s_inf_endo(V_t).item()
        s_inf_epi[i] = gating.Ito_s_inf_epi(V_t).item()

        # IKr gates
        Xr1_inf[i] = gating.IKr_Xr1_inf(V_t).item()
        Xr2_inf[i] = gating.IKr_Xr2_inf(V_t).item()

        # IKs gate
        Xs_inf[i] = gating.IKs_Xs_inf(V_t).item()

        if i % 10 == 0 or i == n - 1:
            print_progress(i + 1, n, f'V = {V:+6.1f} mV')

    print()  # newline after progress

    return {
        'V': V_range,
        'm_inf': m_inf, 'h_inf': h_inf, 'j_inf': j_inf,
        'd_inf': d_inf, 'f_inf': f_inf, 'f2_inf': f2_inf,
        'r_inf': r_inf, 's_inf_endo': s_inf_endo, 's_inf_epi': s_inf_epi,
        'Xr1_inf': Xr1_inf, 'Xr2_inf': Xr2_inf, 'Xs_inf': Xs_inf,
    }


def compute_timeconstant_curves(V_range):
    """Compute all time constant curves over voltage range."""
    n = len(V_range)

    # Storage
    tau_m = np.zeros(n)
    tau_h = np.zeros(n)
    tau_j = np.zeros(n)

    tau_d = np.zeros(n)
    tau_f = np.zeros(n)
    tau_f2 = np.zeros(n)

    tau_r = np.zeros(n)
    tau_s_endo = np.zeros(n)
    tau_s_epi = np.zeros(n)

    tau_Xr1 = np.zeros(n)
    tau_Xr2 = np.zeros(n)
    tau_Xs = np.zeros(n)

    print("Computing time constant curves...")
    for i, V in enumerate(V_range):
        V_t = torch.tensor([V], dtype=torch.float64)

        # INa gates
        tau_m[i] = gating.INa_m_tau(V_t).item()
        tau_h[i] = gating.INa_h_tau(V_t).item()
        tau_j[i] = gating.INa_j_tau(V_t).item()

        # ICaL gates
        tau_d[i] = gating.ICaL_d_tau(V_t).item()
        tau_f[i] = gating.ICaL_f_tau(V_t).item()
        tau_f2[i] = gating.ICaL_f2_tau(V_t).item()

        # Ito gates
        tau_r[i] = gating.Ito_r_tau(V_t).item()
        tau_s_endo[i] = gating.Ito_s_tau_endo(V_t).item()
        tau_s_epi[i] = gating.Ito_s_tau_epi(V_t).item()

        # IKr gates
        tau_Xr1[i] = gating.IKr_Xr1_tau(V_t).item()
        tau_Xr2[i] = gating.IKr_Xr2_tau(V_t).item()

        # IKs gate
        tau_Xs[i] = gating.IKs_Xs_tau(V_t).item()

        if i % 10 == 0 or i == n - 1:
            print_progress(i + 1, n, f'V = {V:+6.1f} mV')

    print()  # newline after progress

    return {
        'V': V_range,
        'tau_m': tau_m, 'tau_h': tau_h, 'tau_j': tau_j,
        'tau_d': tau_d, 'tau_f': tau_f, 'tau_f2': tau_f2,
        'tau_r': tau_r, 'tau_s_endo': tau_s_endo, 'tau_s_epi': tau_s_epi,
        'tau_Xr1': tau_Xr1, 'tau_Xr2': tau_Xr2, 'tau_Xs': tau_Xs,
    }


def validate_key_points(ss_data, tau_data):
    """
    Validate key gating values against literature reference.

    Reference values from:
    - TNNP model page: tau_m(-85)=0.0011ms, tau_h(20)=0.18ms, tau_j(20)=0.54ms
    - CellML generated code
    """
    validations = []

    # Find indices for key voltages
    V = ss_data['V']
    idx_m85 = np.argmin(np.abs(V - (-85)))
    idx_m40 = np.argmin(np.abs(V - (-40)))
    idx_0 = np.argmin(np.abs(V - 0))
    idx_20 = np.argmin(np.abs(V - 20))

    # INa m gate
    # At V=-85mV, m_inf should be very small (~0.0017)
    val = ss_data['m_inf'][idx_m85]
    exp = 0.00165  # from CellML initial state
    tol = 0.001
    passed = abs(val - exp) < tol
    validations.append({
        'gate': 'm_inf', 'V_mV': -85, 'computed': val, 'expected': exp,
        'tolerance': tol, 'passed': passed
    })

    # tau_m at -85mV should be ~0.0011 ms (very fast)
    val = tau_data['tau_m'][idx_m85]
    exp = 0.0011
    tol = 0.001
    passed = abs(val - exp) < tol
    validations.append({
        'gate': 'tau_m', 'V_mV': -85, 'computed': val, 'expected': exp,
        'tolerance': tol, 'passed': passed
    })

    # INa h gate (fast inactivation)
    # At V=-85mV, h_inf should be near 1 (~0.75-0.99)
    val = ss_data['h_inf'][idx_m85]
    exp = 0.75  # approximate
    tol = 0.25
    passed = val > 0.5  # should be > 0.5 at rest
    validations.append({
        'gate': 'h_inf', 'V_mV': -85, 'computed': val, 'expected': exp,
        'tolerance': tol, 'passed': passed
    })

    # tau_h at 20mV should be ~0.18 ms
    val = tau_data['tau_h'][idx_20]
    exp = 0.18
    tol = 0.1
    passed = abs(val - exp) < tol
    validations.append({
        'gate': 'tau_h', 'V_mV': 20, 'computed': val, 'expected': exp,
        'tolerance': tol, 'passed': passed
    })

    # INa j gate (slow inactivation)
    # tau_j at 20mV should be ~0.54 ms
    val = tau_data['tau_j'][idx_20]
    exp = 0.54
    tol = 0.2
    passed = abs(val - exp) < tol
    validations.append({
        'gate': 'tau_j', 'V_mV': 20, 'computed': val, 'expected': exp,
        'tolerance': tol, 'passed': passed
    })

    # ICaL d gate
    # At V=-85mV, d_inf should be very small
    val = ss_data['d_inf'][idx_m85]
    exp = 0.0
    tol = 0.01
    passed = val < 0.01
    validations.append({
        'gate': 'd_inf', 'V_mV': -85, 'computed': val, 'expected': exp,
        'tolerance': tol, 'passed': passed
    })

    # At V=0mV, d_inf should be ~0.5 (half-activation)
    val = ss_data['d_inf'][idx_0]
    exp = 0.5
    tol = 0.3
    passed = 0.2 < val < 0.8
    validations.append({
        'gate': 'd_inf', 'V_mV': 0, 'computed': val, 'expected': exp,
        'tolerance': tol, 'passed': passed
    })

    # ICaL f gate
    # At V=-85mV, f_inf should be near 1
    val = ss_data['f_inf'][idx_m85]
    exp = 1.0
    tol = 0.1
    passed = val > 0.9
    validations.append({
        'gate': 'f_inf', 'V_mV': -85, 'computed': val, 'expected': exp,
        'tolerance': tol, 'passed': passed
    })

    # IKr Xr1 gate
    # At V=-85mV, Xr1_inf should be small
    val = ss_data['Xr1_inf'][idx_m85]
    exp = 0.0
    tol = 0.05
    passed = val < 0.05
    validations.append({
        'gate': 'Xr1_inf', 'V_mV': -85, 'computed': val, 'expected': exp,
        'tolerance': tol, 'passed': passed
    })

    # IKr Xr2 gate
    # At V=-85mV: xr2_inf = 1/(1+exp((V+88)/24)) = 1/(1+exp(3/24)) ≈ 0.469
    # V1/2 = -88 mV, so at -85 mV we're only 3 mV away from half-inactivation
    val = ss_data['Xr2_inf'][idx_m85]
    exp = 0.469  # 1/(1+exp(3/24))
    tol = 0.05
    passed = abs(val - exp) < tol
    validations.append({
        'gate': 'Xr2_inf', 'V_mV': -85, 'computed': val, 'expected': exp,
        'tolerance': tol, 'passed': passed
    })

    return validations


def main():
    print("=" * 70)
    print("TTP06 Phase 1: Gating Curves Validation")
    print("=" * 70)
    print()

    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'ttp06_data')
    os.makedirs(output_dir, exist_ok=True)

    # Voltage range for curves
    V_range = np.linspace(-120, 60, 181)  # 1 mV resolution

    # Compute curves
    ss_data = compute_steadystate_curves(V_range)
    tau_data = compute_timeconstant_curves(V_range)

    # Save steady-state CSV
    ss_file = os.path.join(output_dir, 'phase1_steadystate.csv')
    ss_array = np.column_stack([
        ss_data['V'],
        ss_data['m_inf'], ss_data['h_inf'], ss_data['j_inf'],
        ss_data['d_inf'], ss_data['f_inf'], ss_data['f2_inf'],
        ss_data['r_inf'], ss_data['s_inf_endo'], ss_data['s_inf_epi'],
        ss_data['Xr1_inf'], ss_data['Xr2_inf'], ss_data['Xs_inf'],
    ])
    np.savetxt(ss_file, ss_array, delimiter=',',
               header='V_mV,m_inf,h_inf,j_inf,d_inf,f_inf,f2_inf,r_inf,s_inf_endo,s_inf_epi,Xr1_inf,Xr2_inf,Xs_inf',
               comments='')
    print(f"Saved: {ss_file}")

    # Save time constant CSV
    tau_file = os.path.join(output_dir, 'phase1_timeconstants.csv')
    tau_array = np.column_stack([
        tau_data['V'],
        tau_data['tau_m'], tau_data['tau_h'], tau_data['tau_j'],
        tau_data['tau_d'], tau_data['tau_f'], tau_data['tau_f2'],
        tau_data['tau_r'], tau_data['tau_s_endo'], tau_data['tau_s_epi'],
        tau_data['tau_Xr1'], tau_data['tau_Xr2'], tau_data['tau_Xs'],
    ])
    np.savetxt(tau_file, tau_array, delimiter=',',
               header='V_mV,tau_m,tau_h,tau_j,tau_d,tau_f,tau_f2,tau_r,tau_s_endo,tau_s_epi,tau_Xr1,tau_Xr2,tau_Xs',
               comments='')
    print(f"Saved: {tau_file}")

    # Validate key points
    print()
    print("=" * 70)
    print("Validation Against Literature Reference")
    print("=" * 70)
    print()

    validations = validate_key_points(ss_data, tau_data)

    # Print validation results
    print(f"{'Gate':<12} {'V (mV)':<8} {'Computed':<12} {'Expected':<12} {'Status':<8}")
    print("-" * 60)

    n_passed = 0
    n_total = len(validations)

    for v in validations:
        status = "PASS" if v['passed'] else "FAIL"
        status_symbol = "✓" if v['passed'] else "✗"
        print(f"{v['gate']:<12} {v['V_mV']:<8.0f} {v['computed']:<12.6f} {v['expected']:<12.6f} {status_symbol} {status}")
        if v['passed']:
            n_passed += 1

    print("-" * 60)
    print(f"Passed: {n_passed}/{n_total}")
    print()

    # Save validation CSV
    val_file = os.path.join(output_dir, 'phase1_validation.csv')
    with open(val_file, 'w') as f:
        f.write('gate,V_mV,computed,expected,tolerance,passed\n')
        for v in validations:
            f.write(f"{v['gate']},{v['V_mV']},{v['computed']:.8f},{v['expected']:.8f},{v['tolerance']},{v['passed']}\n")
    print(f"Saved: {val_file}")

    # Summary statistics
    print()
    print("=" * 70)
    print("Summary: Key Time Constants")
    print("=" * 70)

    idx_m85 = np.argmin(np.abs(V_range - (-85)))
    idx_m40 = np.argmin(np.abs(V_range - (-40)))
    idx_0 = np.argmin(np.abs(V_range - 0))
    idx_20 = np.argmin(np.abs(V_range - 20))

    print(f"\nAt V = -85 mV (resting):")
    print(f"  tau_m = {tau_data['tau_m'][idx_m85]:.6f} ms (ref: 0.0011 ms)")
    print(f"  tau_h = {tau_data['tau_h'][idx_m85]:.4f} ms")
    print(f"  tau_j = {tau_data['tau_j'][idx_m85]:.4f} ms")

    print(f"\nAt V = -40 mV (threshold region):")
    print(f"  tau_h = {tau_data['tau_h'][idx_m40]:.4f} ms")
    print(f"  tau_j = {tau_data['tau_j'][idx_m40]:.4f} ms")

    print(f"\nAt V = +20 mV (plateau):")
    print(f"  tau_h = {tau_data['tau_h'][idx_20]:.4f} ms (ref: 0.18 ms)")
    print(f"  tau_j = {tau_data['tau_j'][idx_20]:.4f} ms (ref: 0.54 ms)")

    print()
    print("=" * 70)
    print("Phase 1 Complete!")
    print("=" * 70)

    return 0 if n_passed == n_total else 1


if __name__ == '__main__':
    sys.exit(main())
