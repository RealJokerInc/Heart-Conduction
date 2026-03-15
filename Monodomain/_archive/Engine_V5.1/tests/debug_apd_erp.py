"""
Debug Script: APD90 and ERP Measurement

Phase 0 of ERP/APD Discrepancy Investigation

This script measures:
1. Single-cell APD90 (time from upstroke to 90% repolarization)
2. Single-cell ERP (minimum S1-S2 coupling interval that triggers AP)
3. Gating variable recovery kinetics
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt

from ionic import ORdModel, CellType, StateIndex


def test_single_cell_apd(celltype=CellType.ENDO, plot=True):
    """
    Test 0.A: Measure single-cell APD90.

    Returns:
        dict with APD90, APD50, V_rest, V_peak, and time traces
    """
    print("=" * 70)
    print("TEST 0.A: Single-Cell APD90 Measurement")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ORdModel(celltype, device=device)

    dt = 0.01  # 10µs steps for accuracy
    t_end = 600.0  # 600ms total

    # Stimulus parameters
    stim_start = 10.0
    stim_duration = 1.0
    stim_amplitude = -80.0  # µA/µF

    # Storage
    n_steps = int(t_end / dt)
    V_trace = np.zeros(n_steps)
    t_trace = np.zeros(n_steps)

    print(f"Cell type: {celltype.name}")
    print(f"dt = {dt} ms, t_end = {t_end} ms")
    print(f"Stimulus: {stim_amplitude} µA/µF for {stim_duration} ms at t={stim_start} ms")
    print()

    # Get initial state
    state = model.get_initial_state()

    # Run simulation
    print("Running simulation...")
    for i in range(n_steps):
        t = i * dt

        # Apply stimulus
        if stim_start <= t < stim_start + stim_duration:
            I_stim = torch.tensor(stim_amplitude, dtype=model.dtype, device=model.device)
        else:
            I_stim = torch.tensor(0.0, dtype=model.dtype, device=model.device)

        state = model.step(state, dt, I_stim)

        V_trace[i] = state[StateIndex.V].item()
        t_trace[i] = t

    # Analyze APD
    V_rest = V_trace[0]
    V_peak = V_trace.max()

    # Find upstroke (max dV/dt)
    dVdt = np.diff(V_trace) / dt
    upstroke_idx = np.argmax(dVdt)
    t_upstroke = t_trace[upstroke_idx]

    # APD90: 90% repolarization
    V_90 = V_rest + 0.1 * (V_peak - V_rest)

    # APD50: 50% repolarization
    V_50 = V_rest + 0.5 * (V_peak - V_rest)

    # Find when V crosses these thresholds (after peak)
    peak_idx = np.argmax(V_trace)

    apd90_idx = None
    apd50_idx = None

    for i in range(peak_idx, len(V_trace)):
        if apd50_idx is None and V_trace[i] < V_50:
            apd50_idx = i
        if apd90_idx is None and V_trace[i] < V_90:
            apd90_idx = i
            break

    if apd90_idx is not None:
        t_apd90 = t_trace[apd90_idx]
        APD90 = t_apd90 - t_upstroke
    else:
        APD90 = None
        print("WARNING: APD90 not reached within simulation time!")

    if apd50_idx is not None:
        t_apd50 = t_trace[apd50_idx]
        APD50 = t_apd50 - t_upstroke
    else:
        APD50 = None

    # Print results
    print()
    print("RESULTS:")
    print("-" * 40)
    print(f"V_rest = {V_rest:.1f} mV")
    print(f"V_peak = {V_peak:.1f} mV")
    print(f"V_90 threshold = {V_90:.1f} mV")
    print(f"Upstroke time = {t_upstroke:.1f} ms")
    if APD50:
        print(f"APD50 = {APD50:.1f} ms")
    if APD90:
        print(f"APD90 = {APD90:.1f} ms")
    else:
        print("APD90 = NOT REACHED")
    print()

    # Plot if requested
    if plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(t_trace, V_trace, 'b-', linewidth=1)
        ax.axhline(V_90, color='r', linestyle='--', label=f'V_90 = {V_90:.1f} mV')
        ax.axhline(V_50, color='orange', linestyle='--', label=f'V_50 = {V_50:.1f} mV')
        ax.axvline(t_upstroke, color='g', linestyle=':', label=f'Upstroke = {t_upstroke:.1f} ms')
        if APD90:
            ax.axvline(t_upstroke + APD90, color='r', linestyle=':', label=f'APD90 = {APD90:.1f} ms')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('V (mV)')
        ax.set_title(f'Single Cell Action Potential - {celltype.name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, t_end)
        plt.tight_layout()
        plt.savefig('debug_apd90.png', dpi=150)
        print("Saved: debug_apd90.png")

    return {
        'APD90': APD90,
        'APD50': APD50,
        'V_rest': V_rest,
        'V_peak': V_peak,
        't_trace': t_trace,
        'V_trace': V_trace,
    }


def test_single_cell_erp(celltype=CellType.ENDO, plot=True, apd90_for_di=None):
    """
    Test 0.B: Measure single-cell ERP using S1-S2 protocol.

    Also computes Diastolic Interval (DI) for each coupling interval:
        DI = CI - APD90

    Args:
        celltype: CellType enum (ENDO, EPI, M_CELL)
        plot: Whether to generate plots
        apd90_for_di: APD90 value for DI calculation (if None, uses 270ms default)

    Returns:
        dict with ERP, DI at ERP, and detailed S1-S2 results
    """
    print("=" * 70)
    print("TEST 0.B: Single-Cell ERP Measurement (S1-S2 Protocol)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dt = 0.01  # 10µs steps
    t_end = 800.0  # 800ms total

    # Stimulus parameters
    s1_time = 10.0
    stim_duration = 1.0
    stim_amplitude = -80.0

    # Coupling intervals to test - extended range to find ERP
    coupling_intervals = [200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500]

    # APD90 for DI calculation
    apd90 = apd90_for_di if apd90_for_di is not None else 270.0  # Default ORd ENDO APD90

    print(f"Cell type: {celltype.name}")
    print(f"S1 at t = {s1_time} ms")
    print(f"APD90 for DI calculation = {apd90:.1f} ms")
    print(f"Testing coupling intervals: {coupling_intervals}")
    print()
    print(f"{'CI (ms)':>8} | {'DI (ms)':>8} | {'Max V':>8} | {'Status':>8}")
    print("-" * 45)

    results = []
    V_traces = {}

    for ci in coupling_intervals:
        model = ORdModel(celltype, device=device)
        state = model.get_initial_state()

        s2_time = s1_time + ci
        n_steps = int(t_end / dt)

        V_trace = np.zeros(n_steps)
        t_trace = np.zeros(n_steps)

        max_V_after_s2 = -100

        for i in range(n_steps):
            t = i * dt

            # S1 stimulus
            if s1_time <= t < s1_time + stim_duration:
                I_stim = torch.tensor(stim_amplitude, dtype=model.dtype, device=model.device)
            # S2 stimulus
            elif s2_time <= t < s2_time + stim_duration:
                I_stim = torch.tensor(stim_amplitude, dtype=model.dtype, device=model.device)
            else:
                I_stim = torch.tensor(0.0, dtype=model.dtype, device=model.device)

            state = model.step(state, dt, I_stim)

            V = state[StateIndex.V].item()
            V_trace[i] = V
            t_trace[i] = t

            # Track max V after S2
            if t > s2_time + stim_duration + 5:  # 5ms after S2 ends
                max_V_after_s2 = max(max_V_after_s2, V)

        # AP triggered if V exceeds 0mV after S2
        ap_triggered = max_V_after_s2 > 0

        # Calculate DI (Diastolic Interval)
        di = ci - apd90

        results.append({
            'ci': ci,
            'di': di,
            'max_V_after_s2': max_V_after_s2,
            'ap_triggered': ap_triggered
        })

        V_traces[ci] = (t_trace.copy(), V_trace.copy())

        status = "AP" if ap_triggered else "No AP"
        print(f"{ci:8d} | {di:8.1f} | {max_V_after_s2:+8.1f} | {status:>8}")

    # Find ERP (minimum CI that triggers AP)
    erp = None
    di_at_erp = None
    for r in results:
        if r['ap_triggered']:
            erp = r['ci']
            di_at_erp = r['di']
            break

    print()
    print("RESULTS:")
    print("-" * 40)
    print(f"APD90 (used for DI calculation) = {apd90:.1f} ms")
    if erp:
        print(f"ERP (Effective Refractory Period) = {erp} ms")
        print(f"DI at ERP = {di_at_erp:.1f} ms")
        print()
        print("Interpretation:")
        if di_at_erp > 20:
            print(f"  - Tissue requires {di_at_erp:.0f}ms of diastole to recover")
            print(f"  - This is LONGER than expected (~0-20ms for normal recovery)")
            print(f"  - Suggests slow recovery of INa or ICaL gating variables")
        elif di_at_erp < 0:
            print(f"  - ERP occurs BEFORE full repolarization")
            print(f"  - This is normal: tissue becomes excitable during late repolarization")
        else:
            print(f"  - Normal: DI at ERP is ~{di_at_erp:.0f}ms (expected 0-20ms)")
    else:
        print(f"ERP > {coupling_intervals[-1]} ms (no AP triggered in tested range)")
        print(f"DI at max CI tested = {coupling_intervals[-1] - apd90:.1f} ms")
        print("  - Tissue still refractory even with long diastolic interval!")
    print()

    # Plot if requested
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: All traces
        ax1 = axes[0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(coupling_intervals)))

        for (ci, color) in zip(coupling_intervals, colors):
            t, V = V_traces[ci]
            di = ci - apd90
            label = f'CI={ci}ms (DI={di:.0f}ms)'
            ax1.plot(t, V, color=color, linewidth=0.8, label=label, alpha=0.8)

        ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('V (mV)')
        ax1.set_title(f'S1-S2 Protocol - {celltype.name} (DI = CI - APD90)')
        ax1.legend(loc='upper right', fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 600)

        # Plot 2: Max V after S2 vs CI and DI (dual x-axis)
        ax2 = axes[1]
        cis = [r['ci'] for r in results]
        dis = [r['di'] for r in results]
        max_Vs = [r['max_V_after_s2'] for r in results]

        ax2.plot(cis, max_Vs, 'bo-', markersize=8)
        ax2.axhline(0, color='r', linestyle='--', label='AP threshold (0 mV)')
        if erp:
            ax2.axvline(erp, color='g', linestyle='--', label=f'ERP = {erp} ms (DI={di_at_erp:.0f}ms)')
        ax2.set_xlabel('Coupling Interval CI (ms)')
        ax2.set_ylabel('Max V after S2 (mV)')
        ax2.set_title('S2 Response vs Coupling Interval / Diastolic Interval')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add secondary x-axis for DI
        ax2_top = ax2.twiny()
        ax2_top.set_xlim(ax2.get_xlim()[0] - apd90, ax2.get_xlim()[1] - apd90)
        ax2_top.set_xlabel('Diastolic Interval DI (ms)')

        plt.tight_layout()
        plt.savefig('debug_erp.png', dpi=150)
        print("Saved: debug_erp.png")

    return {
        'ERP': erp,
        'DI_at_ERP': di_at_erp,
        'APD90_used': apd90,
        'results': results,
        'V_traces': V_traces
    }


def test_gating_recovery(celltype=CellType.ENDO, plot=True):
    """
    Test 0.C: Analyze gating variable recovery during and after AP.

    Tracks key gates that affect refractoriness:
    - hf, hs: INa fast/slow inactivation
    - j: INa recovery from inactivation
    - ff, fcaf: ICaL voltage/Ca-dependent inactivation

    Returns:
        dict with gating variable traces and recovery times
    """
    print("=" * 70)
    print("TEST 0.C: Gating Variable Recovery Analysis")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ORdModel(celltype, device=device)
    state = model.get_initial_state()

    dt = 0.01
    t_end = 600.0

    stim_start = 10.0
    stim_duration = 1.0
    stim_amplitude = -80.0

    n_steps = int(t_end / dt)

    # Storage for voltage and key gating variables
    t_trace = np.zeros(n_steps)
    V_trace = np.zeros(n_steps)

    # Gating variables to track - key gates for refractoriness
    # Using correct StateIndex mapping from ORd model
    gate_traces = {
        'hf': np.zeros(n_steps),     # INa fast inactivation (StateIndex.hf = 10)
        'hs': np.zeros(n_steps),     # INa slow inactivation (StateIndex.hs = 11)
        'j': np.zeros(n_steps),      # INa recovery (StateIndex.j = 12)
        'hL': np.zeros(n_steps),     # INaL inactivation (StateIndex.hL = 16)
        'ff': np.zeros(n_steps),     # ICaL fast voltage inactivation (StateIndex.ff = 25)
        'fcaf': np.zeros(n_steps),   # ICaL Ca-dependent inactivation (StateIndex.fcaf = 27)
    }

    print(f"Cell type: {celltype.name}")
    print(f"Tracking gating variables: {list(gate_traces.keys())}")
    print()

    # Run simulation
    print("Running simulation...")
    for i in range(n_steps):
        t = i * dt

        if stim_start <= t < stim_start + stim_duration:
            I_stim = torch.tensor(stim_amplitude, dtype=model.dtype, device=model.device)
        else:
            I_stim = torch.tensor(0.0, dtype=model.dtype, device=model.device)

        state = model.step(state, dt, I_stim)

        t_trace[i] = t
        V_trace[i] = state[StateIndex.V].item()

        # Extract gating variables using correct StateIndex
        gate_traces['hf'][i] = state[StateIndex.hf].item()
        gate_traces['hs'][i] = state[StateIndex.hs].item()
        gate_traces['j'][i] = state[StateIndex.j].item()
        gate_traces['hL'][i] = state[StateIndex.hL].item()
        gate_traces['ff'][i] = state[StateIndex.ff].item()
        gate_traces['fcaf'][i] = state[StateIndex.fcaf].item()

    # Analyze recovery times
    print()
    print("Analyzing recovery times...")
    print("-" * 50)

    # Find upstroke time
    dVdt = np.diff(V_trace) / dt
    upstroke_idx = np.argmax(dVdt)
    t_upstroke = t_trace[upstroke_idx]

    # For each gate, find when it recovers to 90% of initial value
    recovery_times = {}
    recovery_threshold = 0.9

    for gate_name, trace in gate_traces.items():
        initial_value = trace[0]
        min_value = trace.min()

        # Skip if gate doesn't change much or if it's all zeros
        if initial_value < 0.01 or (initial_value - min_value) < 0.01:
            recovery_times[gate_name] = None
            print(f"  {gate_name}: No significant change (init={initial_value:.3f}, min={min_value:.3f})")
            continue

        # Find when gate recovers to 90% of initial value
        target = initial_value * recovery_threshold
        recovery_idx = None

        # Start searching after minimum
        min_idx = np.argmin(trace)
        for i in range(min_idx, len(trace)):
            if trace[i] >= target:
                recovery_idx = i
                break

        if recovery_idx is not None:
            t_recovery = t_trace[recovery_idx]
            recovery_time = t_recovery - t_upstroke
            recovery_times[gate_name] = recovery_time
            print(f"  {gate_name}: recovers to {recovery_threshold*100:.0f}% at t = {t_recovery:.1f} ms "
                  f"({recovery_time:.1f} ms after upstroke)")
        else:
            recovery_times[gate_name] = None
            print(f"  {gate_name}: does NOT recover to {recovery_threshold*100:.0f}% within simulation")

    # Plot if requested
    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # Plot 1: Voltage
        ax1 = axes[0]
        ax1.plot(t_trace, V_trace, 'b-', linewidth=1)
        ax1.set_ylabel('V (mV)')
        ax1.set_title(f'Action Potential and Gating Variable Recovery - {celltype.name}')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, t_end)

        # Plot 2: INa gates (hf, hs, j for INa; hL for INaL)
        ax2 = axes[1]
        ax2.plot(t_trace, gate_traces['hf'], 'r-', label='hf (INa fast inact)', linewidth=1)
        ax2.plot(t_trace, gate_traces['hs'], 'b-', label='hs (INa slow inact)', linewidth=1)
        ax2.plot(t_trace, gate_traces['j'], 'g-', label='j (INa recovery)', linewidth=1)
        ax2.plot(t_trace, gate_traces['hL'], 'm--', label='hL (INaL inact)', linewidth=1)
        ax2.axhline(0.9, color='gray', linestyle=':', alpha=0.5, label='90% recovery')
        ax2.set_ylabel('Gate Value')
        ax2.set_title('INa/INaL Inactivation Gates (must recover for excitability)')
        ax2.legend(loc='right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, t_end)
        ax2.set_ylim(-0.05, 1.05)

        # Plot 3: ICaL gates (ff, fcaf)
        ax3 = axes[2]
        ax3.plot(t_trace, gate_traces['ff'], 'r-', label='ff (ICaL fast volt-inact)', linewidth=1)
        ax3.plot(t_trace, gate_traces['fcaf'], 'b-', label='fcaf (ICaL Ca-dep inact)', linewidth=1)
        ax3.axhline(0.9, color='gray', linestyle=':', alpha=0.5, label='90% recovery')
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Gate Value')
        ax3.set_title('ICaL Inactivation Gates')
        ax3.legend(loc='right')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, t_end)
        ax3.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        plt.savefig('debug_gating.png', dpi=150)
        print()
        print("Saved: debug_gating.png")

    return {
        'recovery_times': recovery_times,
        't_trace': t_trace,
        'V_trace': V_trace,
        'gate_traces': gate_traces
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Debug APD90 and ERP measurements')
    parser.add_argument('--test', type=str, choices=['apd', 'erp', 'gates', 'all'],
                       default='all', help='Which test to run')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    args = parser.parse_args()

    plot = not args.no_plot

    print()
    print("=" * 70)
    print("ERP/APD DEBUG INVESTIGATION - PHASE 0")
    print("=" * 70)
    print()

    results = {}

    if args.test in ['apd', 'all']:
        results['apd'] = test_single_cell_apd(plot=plot)
        print()

    if args.test in ['erp', 'all']:
        # Use measured APD90 for DI calculation if available
        apd90_for_di = None
        if 'apd' in results and results['apd']['APD90'] is not None:
            apd90_for_di = results['apd']['APD90']
        results['erp'] = test_single_cell_erp(plot=plot, apd90_for_di=apd90_for_di)
        print()

    if args.test in ['gates', 'all']:
        results['gates'] = test_gating_recovery(plot=plot)
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    if 'apd' in results:
        apd90 = results['apd']['APD90']
        apd50 = results['apd']['APD50']
        print(f"APD90 = {apd90:.1f} ms" if apd90 else "APD90 = NOT MEASURED")
        print(f"APD50 = {apd50:.1f} ms" if apd50 else "APD50 = NOT MEASURED")

    if 'erp' in results:
        erp = results['erp']['ERP']
        di_at_erp = results['erp']['DI_at_ERP']
        print(f"ERP = {erp} ms" if erp else f"ERP > tested range")
        if di_at_erp is not None:
            print(f"DI at ERP = {di_at_erp:.1f} ms")

    if 'apd' in results and 'erp' in results:
        apd90 = results['apd']['APD90']
        erp = results['erp']['ERP']
        di_at_erp = results['erp']['DI_at_ERP']
        if apd90 and erp:
            gap = erp - apd90
            print()
            print("-" * 50)
            print("ANALYSIS")
            print("-" * 50)
            print(f"GAP between ERP and APD90 = {gap:.1f} ms")
            print(f"DI (Diastolic Interval) at ERP = {di_at_erp:.1f} ms")
            print()
            if gap > 30:
                print("❌ WARNING: ERP >> APD90")
                print("   This indicates POST-REPOLARIZATION REFRACTORINESS!")
                print("   The tissue remains refractory for {:.0f}ms after APD90.".format(gap))
                print()
                print("   Likely causes:")
                print("   1. Slow recovery of INa h-gates (hf, hs)")
                print("   2. Slow recovery of j-gate")
                print("   3. Slow recovery of ICaL gates (ff, fcaf)")
                print()
                print("   → Check gating recovery test for which gate is slowest")
            elif gap < -30:
                print("✓ NOTE: ERP << APD90")
                print("   Tissue is excitable before full repolarization.")
                print("   This is the normal expected behavior.")
            else:
                print("✓ OK: ERP ≈ APD90 (normal relationship)")
                print("   Gap of {:.0f}ms is within expected range (-30 to +30ms)".format(gap))

    # Gate recovery summary
    if 'gates' in results:
        recovery_times = results['gates']['recovery_times']
        print()
        print("-" * 50)
        print("GATING VARIABLE RECOVERY SUMMARY")
        print("-" * 50)
        for gate, time in sorted(recovery_times.items(), key=lambda x: (x[1] is None, x[1] or 0), reverse=True):
            if time is not None:
                print(f"  {gate:5}: {time:6.1f} ms after upstroke")
            else:
                print(f"  {gate:5}: Does not fully recover")

        # Find the slowest-recovering gate
        valid_times = {k: v for k, v in recovery_times.items() if v is not None}
        if valid_times:
            slowest = max(valid_times.items(), key=lambda x: x[1])
            print()
            print(f"  → Slowest recovery: {slowest[0]} at {slowest[1]:.1f} ms")
            print(f"  → This gate likely determines ERP")

    print()


if __name__ == '__main__':
    main()
