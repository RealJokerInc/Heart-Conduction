#!/usr/bin/env python3
"""
TTP06 Phase 2: dt Sensitivity and ERP Validation

Tests:
1. dt sensitivity: Compare AP with dt=0.005ms (reference) vs dt=0.01ms
2. Gate recovery during single AP
3. ERP measurement using S1-S2 protocol

CSV Output:
- phase2_dt_comparison.csv: AP traces at different dt values
- phase2_gate_recovery.csv: Gate values during AP and recovery
- phase2_erp.csv: S1-S2 protocol results

Reference dt=0.005ms (5µs) from literature (PMC12246384)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from ionic import TTP06Model, CellType


def print_progress(current, total, prefix='', width=40):
    """Print progress bar."""
    pct = current / total
    filled = int(width * pct)
    bar = '█' * filled + ' ' * (width - filled)
    print(f'\r  {prefix} [{bar}] {pct*100:5.1f}%', end='', flush=True)


def run_ap_simulation(model, dt, t_end, stim_start=10.0, stim_duration=1.0,
                      stim_amplitude=-52.0, save_every=None):
    """
    Run single AP simulation and return voltage trace.

    Parameters
    ----------
    model : TTP06Model
        Ionic model instance
    dt : float
        Time step (ms)
    t_end : float
        End time (ms)
    stim_start : float
        Stimulus start time (ms)
    stim_duration : float
        Stimulus duration (ms)
    stim_amplitude : float
        Stimulus amplitude (pA/pF)
    save_every : int or None
        Save every N steps (None = save all)

    Returns
    -------
    dict
        Contains t, V arrays and AP metrics
    """
    n_steps = int(t_end / dt)
    save_every = save_every or 1
    n_save = n_steps // save_every

    state = model.get_initial_state()

    t_data = np.zeros(n_save)
    V_data = np.zeros(n_save)

    save_idx = 0
    I_stim_on = torch.tensor(stim_amplitude, dtype=model.dtype, device=model.device)
    I_stim_off = torch.tensor(0.0, dtype=model.dtype, device=model.device)

    for i in range(n_steps):
        t = i * dt

        # Stimulus
        if stim_start <= t < stim_start + stim_duration:
            I_stim = I_stim_on
        else:
            I_stim = I_stim_off

        state = model.step(state, dt, I_stim)

        if i % save_every == 0 and save_idx < n_save:
            t_data[save_idx] = t
            V_data[save_idx] = state[model.V_index].item()
            save_idx += 1

    # Compute metrics
    V_rest = V_data[0]
    V_peak = V_data.max()
    peak_idx = np.argmax(V_data)
    t_peak = t_data[peak_idx]

    # APD90
    V90 = V_rest + 0.1 * (V_peak - V_rest)
    APD90 = np.nan
    for i in range(peak_idx, len(V_data)):
        if V_data[i] < V90:
            APD90 = t_data[i] - t_data[peak_idx]
            break

    return {
        't': t_data,
        'V': V_data,
        'V_rest': V_rest,
        'V_peak': V_peak,
        't_peak': t_peak,
        'APD90': APD90,
    }


def run_gate_recovery(model, dt, t_end, stim_start=10.0, stim_duration=1.0,
                      stim_amplitude=-52.0, save_every=10):
    """
    Run AP and track gate recovery.

    Returns gate values over time for INa (m, h, j) and ICaL (d, f, f2).
    """
    n_steps = int(t_end / dt)
    n_save = n_steps // save_every

    state = model.get_initial_state()

    # Storage
    t_data = np.zeros(n_save)
    V_data = np.zeros(n_save)
    m_data = np.zeros(n_save)
    h_data = np.zeros(n_save)
    j_data = np.zeros(n_save)
    d_data = np.zeros(n_save)
    f_data = np.zeros(n_save)
    f2_data = np.zeros(n_save)
    Xr1_data = np.zeros(n_save)
    Xr2_data = np.zeros(n_save)
    Xs_data = np.zeros(n_save)

    save_idx = 0
    I_stim_on = torch.tensor(stim_amplitude, dtype=model.dtype, device=model.device)
    I_stim_off = torch.tensor(0.0, dtype=model.dtype, device=model.device)

    # Gate indices (from TTP06Model StateIndex)
    # State: [V, Ki, Nai, Cai, CaSR, CaSS, m, h, j, r, s, d, f, f2, fCass, Xr1, Xr2, Xs, RR]
    m_idx = 6
    h_idx = 7
    j_idx = 8
    d_idx = 11
    f_idx = 12
    f2_idx = 13
    Xr1_idx = 15
    Xr2_idx = 16
    Xs_idx = 17

    for i in range(n_steps):
        t = i * dt

        if stim_start <= t < stim_start + stim_duration:
            I_stim = I_stim_on
        else:
            I_stim = I_stim_off

        state = model.step(state, dt, I_stim)

        if i % save_every == 0 and save_idx < n_save:
            t_data[save_idx] = t
            V_data[save_idx] = state[model.V_index].item()
            m_data[save_idx] = state[m_idx].item()
            h_data[save_idx] = state[h_idx].item()
            j_data[save_idx] = state[j_idx].item()
            d_data[save_idx] = state[d_idx].item()
            f_data[save_idx] = state[f_idx].item()
            f2_data[save_idx] = state[f2_idx].item()
            Xr1_data[save_idx] = state[Xr1_idx].item()
            Xr2_data[save_idx] = state[Xr2_idx].item()
            Xs_data[save_idx] = state[Xs_idx].item()
            save_idx += 1

        if i % (n_steps // 20) == 0:
            print_progress(i + 1, n_steps, f't = {t:6.1f} ms')

    print()

    return {
        't': t_data,
        'V': V_data,
        'm': m_data, 'h': h_data, 'j': j_data,
        'd': d_data, 'f': f_data, 'f2': f2_data,
        'Xr1': Xr1_data, 'Xr2': Xr2_data, 'Xs': Xs_data,
    }


def run_s1s2_protocol(model, dt, S1_time=10.0, S2_CIs=None,
                      stim_duration=1.0, stim_amplitude=-52.0):
    """
    Run S1-S2 protocol to measure ERP.

    Parameters
    ----------
    model : TTP06Model
        Model instance
    dt : float
        Time step (ms)
    S1_time : float
        S1 stimulus time (ms)
    S2_CIs : list
        Coupling intervals to test (ms)
    stim_duration : float
        Stimulus duration (ms)
    stim_amplitude : float
        Stimulus amplitude (pA/pF)

    Returns
    -------
    list
        Results for each CI
    """
    if S2_CIs is None:
        S2_CIs = list(range(200, 520, 20))

    results = []

    I_stim_on = torch.tensor(stim_amplitude, dtype=model.dtype, device=model.device)
    I_stim_off = torch.tensor(0.0, dtype=model.dtype, device=model.device)

    for ci_idx, CI in enumerate(S2_CIs):
        print_progress(ci_idx + 1, len(S2_CIs), f'CI = {CI:3.0f} ms')

        S2_time = S1_time + CI
        t_end = S2_time + 400.0  # Run 400ms after S2
        n_steps = int(t_end / dt)

        state = model.get_initial_state()

        V_max_after_S2 = -100.0
        V_at_S2 = None

        for i in range(n_steps):
            t = i * dt

            # S1 or S2 stimulus
            in_S1 = S1_time <= t < S1_time + stim_duration
            in_S2 = S2_time <= t < S2_time + stim_duration

            if in_S1 or in_S2:
                I_stim = I_stim_on
            else:
                I_stim = I_stim_off

            state = model.step(state, dt, I_stim)

            V = state[model.V_index].item()

            # Record voltage at S2 start
            if abs(t - S2_time) < dt:
                V_at_S2 = V

            # Track max voltage after S2
            if t > S2_time + stim_duration:
                V_max_after_S2 = max(V_max_after_S2, V)

        # Determine if AP was triggered (V > 0 mV is strong criterion)
        AP_triggered = V_max_after_S2 > 0.0

        results.append({
            'CI': CI,
            'V_at_S2': V_at_S2,
            'V_max_after_S2': V_max_after_S2,
            'AP_triggered': AP_triggered,
        })

    print()
    return results


def main():
    print("=" * 70)
    print("TTP06 Phase 2: dt Sensitivity and ERP Validation")
    print("=" * 70)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    output_dir = os.path.join(os.path.dirname(__file__), 'ttp06_data')
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # Test 1: dt Sensitivity
    # =========================================================================
    print()
    print("=" * 70)
    print("Test 1: dt Sensitivity")
    print("=" * 70)
    print()

    dt_values = [0.005, 0.01, 0.02]  # ms
    t_end = 500.0

    dt_results = {}

    for dt in dt_values:
        print(f"Running with dt = {dt} ms...")
        model = TTP06Model(celltype=CellType.ENDO, device=device)

        # Save every N steps to get ~0.1 ms resolution
        save_every = max(1, int(0.1 / dt))

        result = run_ap_simulation(model, dt, t_end, save_every=save_every)
        dt_results[dt] = result

        print(f"  V_rest = {result['V_rest']:.2f} mV")
        print(f"  V_peak = {result['V_peak']:.2f} mV")
        print(f"  APD90  = {result['APD90']:.1f} ms")
        print()

    # Save dt comparison
    dt_file = os.path.join(output_dir, 'phase2_dt_comparison.csv')

    # Interpolate to common time base for CSV
    t_common = np.arange(0, t_end, 0.1)
    dt_array = np.column_stack([t_common])
    header = 'time_ms'

    for dt in dt_values:
        V_interp = np.interp(t_common, dt_results[dt]['t'], dt_results[dt]['V'])
        dt_array = np.column_stack([dt_array, V_interp])
        header += f',V_dt{dt}'

    np.savetxt(dt_file, dt_array, delimiter=',', header=header, comments='')
    print(f"Saved: {dt_file}")

    # dt sensitivity analysis
    print()
    print("dt Sensitivity Analysis:")
    print("-" * 60)
    print(f"{'dt (ms)':<12} {'V_peak (mV)':<14} {'APD90 (ms)':<14} {'Diff from ref':<14}")
    print("-" * 60)

    ref_dt = 0.005
    ref_apd = dt_results[ref_dt]['APD90']

    for dt in dt_values:
        r = dt_results[dt]
        diff = r['APD90'] - ref_apd
        diff_str = f"{diff:+.1f} ms" if dt != ref_dt else "(reference)"
        print(f"{dt:<12.3f} {r['V_peak']:<14.2f} {r['APD90']:<14.1f} {diff_str:<14}")

    # =========================================================================
    # Test 2: Gate Recovery During AP
    # =========================================================================
    print()
    print("=" * 70)
    print("Test 2: Gate Recovery During AP")
    print("=" * 70)
    print()

    dt = 0.01  # Use standard dt
    t_end = 600.0  # Run 600ms for full recovery

    model = TTP06Model(celltype=CellType.ENDO, device=device)
    print(f"Running gate recovery simulation (dt={dt}ms, t_end={t_end}ms)...")

    gate_data = run_gate_recovery(model, dt, t_end, save_every=10)

    # Save gate recovery CSV
    gate_file = os.path.join(output_dir, 'phase2_gate_recovery.csv')
    gate_array = np.column_stack([
        gate_data['t'], gate_data['V'],
        gate_data['m'], gate_data['h'], gate_data['j'],
        gate_data['d'], gate_data['f'], gate_data['f2'],
        gate_data['Xr1'], gate_data['Xr2'], gate_data['Xs'],
    ])
    np.savetxt(gate_file, gate_array, delimiter=',',
               header='time_ms,V_mV,m,h,j,d,f,f2,Xr1,Xr2,Xs',
               comments='')
    print(f"Saved: {gate_file}")

    # Analyze gate recovery
    print()
    print("Gate Recovery Analysis (at t=600ms):")
    print("-" * 60)

    gates = ['m', 'h', 'j', 'd', 'f', 'f2', 'Xr1', 'Xr2', 'Xs']

    print(f"{'Gate':<8} {'Initial':<12} {'Min':<12} {'Final':<12} {'Recovery %':<12}")
    print("-" * 60)

    for gate in gates:
        data = gate_data[gate]
        initial = data[0]
        minimum = data.min()
        final = data[-1]

        if initial > 0.001:
            recovery_pct = (final / initial) * 100
        else:
            recovery_pct = np.nan

        print(f"{gate:<8} {initial:<12.6f} {minimum:<12.6f} {final:<12.6f} {recovery_pct:<12.1f}")

    # =========================================================================
    # Test 3: ERP Measurement (S1-S2 Protocol)
    # =========================================================================
    print()
    print("=" * 70)
    print("Test 3: ERP Measurement (S1-S2 Protocol)")
    print("=" * 70)
    print()

    dt = 0.01
    model = TTP06Model(celltype=CellType.ENDO, device=device)

    # Test coupling intervals from 200ms to 500ms
    S2_CIs = list(range(180, 360, 10))

    print(f"Testing {len(S2_CIs)} coupling intervals...")
    erp_results = run_s1s2_protocol(model, dt, S2_CIs=S2_CIs)

    # Save ERP CSV
    erp_file = os.path.join(output_dir, 'phase2_erp.csv')
    with open(erp_file, 'w') as f:
        f.write('CI_ms,V_at_S2_mV,V_max_after_S2_mV,AP_triggered\n')
        for r in erp_results:
            f.write(f"{r['CI']},{r['V_at_S2']:.2f},{r['V_max_after_S2']:.2f},{r['AP_triggered']}\n")
    print(f"Saved: {erp_file}")

    # Find ERP
    print()
    print("S1-S2 Protocol Results:")
    print("-" * 60)
    print(f"{'CI (ms)':<10} {'V_at_S2 (mV)':<14} {'V_max (mV)':<14} {'AP?':<8}")
    print("-" * 60)

    erp = None
    for r in erp_results:
        ap_str = "YES" if r['AP_triggered'] else "NO"
        print(f"{r['CI']:<10.0f} {r['V_at_S2']:<14.2f} {r['V_max_after_S2']:<14.2f} {ap_str:<8}")

        if r['AP_triggered'] and erp is None:
            erp = r['CI']

    print("-" * 60)
    if erp:
        print(f"ERP (single cell) ≤ {erp} ms")
    else:
        print("ERP > 500 ms (no AP triggered)")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Phase 2 Summary")
    print("=" * 70)
    print()

    print("dt Sensitivity:")
    for dt in dt_values:
        r = dt_results[dt]
        print(f"  dt={dt}ms: APD90={r['APD90']:.1f}ms, V_peak={r['V_peak']:.2f}mV")

    # Check dt sensitivity threshold (should be < 5% difference)
    max_diff = max(abs(dt_results[dt]['APD90'] - ref_apd) for dt in dt_values)
    dt_ok = max_diff < 0.05 * ref_apd
    print(f"  Max APD90 difference: {max_diff:.1f} ms ({max_diff/ref_apd*100:.1f}%)")
    print(f"  dt sensitivity: {'OK (<5%)' if dt_ok else 'WARNING (>5%)'}")

    print()
    print("Gate Recovery (at t=600ms):")
    h_final = gate_data['h'][-1]
    j_final = gate_data['j'][-1]
    h_initial = gate_data['h'][0]
    j_initial = gate_data['j'][0]
    print(f"  h: {h_final/h_initial*100:.1f}% of initial")
    print(f"  j: {j_final/j_initial*100:.1f}% of initial")

    print()
    print(f"ERP (single cell): {'≤ ' + str(erp) + ' ms' if erp else '> 500 ms'}")

    print()
    print("=" * 70)
    print("Phase 2 Complete!")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
