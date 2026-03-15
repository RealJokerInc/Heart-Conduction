"""
Phase 2b: Multi-AP Comparison Test

Compare first AP vs subsequent APs to verify:
1. First AP has elevated V_peak due to initial conditions (h=1.0)
2. Subsequent APs have lower V_peak after gates equilibrate to h_inf ≈ 0.68

This test confirms whether initial conditions are causing the elevated V_peak issue.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from ionic import ORdModel, CellType, StateIndex

def run_multi_ap_test():
    print("=" * 70)
    print("PHASE 2b: Multi-AP Comparison Test")
    print("=" * 70)
    print()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = ORdModel(CellType.ENDO, device=device)

    dt = 0.01  # ms
    bcl = 1000.0  # Basic cycle length (ms)
    n_beats = 5
    t_end = n_beats * bcl + 100  # Extra time after last beat

    stim_duration = 1.0  # ms
    stim_amplitude = -80.0  # uA/uF

    # Stimulus times
    stim_times = [10.0 + i * bcl for i in range(n_beats)]

    print(f"BCL = {bcl} ms")
    print(f"Number of beats: {n_beats}")
    print(f"Stimulus times: {stim_times}")
    print()

    n_steps = int(t_end / dt)

    # Storage for each AP metrics
    ap_metrics = []

    # Full resolution data for dV/dt calculation
    V_data = np.zeros(n_steps)
    t_data = np.zeros(n_steps)

    # Gate values at each stimulus time
    gate_at_stim = []

    state = model.get_initial_state()

    print("Running simulation...")

    current_beat = 0
    for i in range(n_steps):
        t = i * dt
        t_data[i] = t

        # Check if we're about to apply stimulus
        if current_beat < len(stim_times):
            stim_t = stim_times[current_beat]
            if abs(t - stim_t) < dt/2:
                # Record gate values just before stimulus
                gate_at_stim.append({
                    'beat': current_beat + 1,
                    't': t,
                    'hf': state[StateIndex.hf].item(),
                    'hs': state[StateIndex.hs].item(),
                    'j': state[StateIndex.j].item(),
                    'm': state[StateIndex.m].item(),
                })

        # Stimulus logic
        I_stim = torch.tensor(0.0, dtype=model.dtype, device=model.device)
        for stim_t in stim_times:
            if stim_t <= t < stim_t + stim_duration:
                I_stim = torch.tensor(stim_amplitude, dtype=model.dtype, device=model.device)
                break

        state = model.step(state, dt, I_stim)
        V_data[i] = state[StateIndex.V].item()

        # Check if we just finished a beat (detect repolarization)
        if current_beat < len(stim_times):
            stim_t = stim_times[current_beat]
            if t > stim_t + 50 and t < stim_t + bcl - 100:
                # Look for peak in this window
                pass

    print("Simulation complete.")
    print()

    # Analyze each AP
    print("=" * 70)
    print("AP ANALYSIS BY BEAT")
    print("=" * 70)
    print()

    # Compute dV/dt
    dVdt = np.diff(V_data) / dt

    for beat_idx, stim_t in enumerate(stim_times):
        # Window for this AP
        start_idx = int(stim_t / dt)
        end_idx = int((stim_t + bcl - 50) / dt)
        if end_idx > len(V_data):
            end_idx = len(V_data)

        V_window = V_data[start_idx:end_idx]
        dVdt_window = dVdt[start_idx:min(end_idx, len(dVdt))]

        V_rest = V_window[0]
        V_peak = V_window.max()
        peak_idx = np.argmax(V_window)

        # dV/dt max
        dVdt_max = dVdt_window.max() if len(dVdt_window) > 0 else 0

        # APD90
        V_90 = V_rest + 0.1 * (V_peak - V_rest)
        apd90 = None
        for j in range(peak_idx, len(V_window)):
            if V_window[j] < V_90:
                apd90 = j * dt
                break

        ap_metrics.append({
            'beat': beat_idx + 1,
            'V_rest': V_rest,
            'V_peak': V_peak,
            'dVdt_max': dVdt_max,
            'APD90': apd90,
        })

    # Print results table
    print(f"{'Beat':<6} {'V_rest (mV)':<12} {'V_peak (mV)':<12} {'dV/dt_max':<12} {'APD90 (ms)':<12}")
    print("-" * 60)
    for m in ap_metrics:
        apd_str = f"{m['APD90']:.1f}" if m['APD90'] else "N/A"
        print(f"{m['beat']:<6} {m['V_rest']:<12.2f} {m['V_peak']:<12.2f} {m['dVdt_max']:<12.1f} {apd_str:<12}")

    print()
    print("=" * 70)
    print("GATE VALUES AT EACH STIMULUS")
    print("=" * 70)
    print()
    print(f"{'Beat':<6} {'t (ms)':<10} {'hf':<10} {'hs':<10} {'j':<10} {'m':<10}")
    print("-" * 60)
    for g in gate_at_stim:
        print(f"{g['beat']:<6} {g['t']:<10.1f} {g['hf']:<10.4f} {g['hs']:<10.4f} {g['j']:<10.4f} {g['m']:<10.6f}")

    print()
    print("=" * 70)
    print("COMPARISON: BEAT 1 vs BEAT 5")
    print("=" * 70)
    print()

    if len(ap_metrics) >= 5:
        b1 = ap_metrics[0]
        b5 = ap_metrics[4]

        print(f"V_peak change: {b1['V_peak']:.2f} mV -> {b5['V_peak']:.2f} mV (Δ = {b5['V_peak'] - b1['V_peak']:.2f} mV)")
        print(f"dV/dt_max change: {b1['dVdt_max']:.1f} -> {b5['dVdt_max']:.1f} mV/ms (Δ = {b5['dVdt_max'] - b1['dVdt_max']:.1f})")

        if len(gate_at_stim) >= 5:
            g1 = gate_at_stim[0]
            g5 = gate_at_stim[4]
            print()
            print("Gate values before stimulus:")
            print(f"  hf: {g1['hf']:.4f} -> {g5['hf']:.4f} (Δ = {g5['hf'] - g1['hf']:.4f})")
            print(f"  hs: {g1['hs']:.4f} -> {g5['hs']:.4f} (Δ = {g5['hs'] - g1['hs']:.4f})")
            print(f"  j:  {g1['j']:.4f} -> {g5['j']:.4f} (Δ = {g5['j'] - g1['j']:.4f})")

    print()

    # Theoretical h_inf at V_rest
    V_rest = -87.5
    h_inf = 1.0 / (1.0 + np.exp((V_rest + 82.90) / 6.086))
    print(f"Theoretical h_inf at V_rest = {V_rest} mV: {h_inf:.4f}")
    print()

    # Save data
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(output_dir, "phase2_data")
    os.makedirs(data_dir, exist_ok=True)

    # Save multi-AP voltage trace
    voltage_file = os.path.join(data_dir, "phase2b_multi_ap_voltage.csv")
    np.savetxt(voltage_file,
               np.column_stack([t_data, V_data]),
               delimiter=',',
               header='time_ms,voltage_mV',
               comments='')
    print(f"Saved: {voltage_file}")

    # Save AP metrics
    metrics_file = os.path.join(data_dir, "phase2b_ap_metrics.csv")
    with open(metrics_file, 'w') as f:
        f.write("beat,V_rest_mV,V_peak_mV,dVdt_max,APD90_ms\n")
        for m in ap_metrics:
            apd_str = f"{m['APD90']:.1f}" if m['APD90'] else ""
            f.write(f"{m['beat']},{m['V_rest']:.2f},{m['V_peak']:.2f},{m['dVdt_max']:.1f},{apd_str}\n")
    print(f"Saved: {metrics_file}")

    # Save gate values at stimulus
    gates_file = os.path.join(data_dir, "phase2b_gates_at_stim.csv")
    with open(gates_file, 'w') as f:
        f.write("beat,t_ms,hf,hs,j,m\n")
        for g in gate_at_stim:
            f.write(f"{g['beat']},{g['t']:.1f},{g['hf']:.6f},{g['hs']:.6f},{g['j']:.6f},{g['m']:.6f}\n")
    print(f"Saved: {gates_file}")

    print()
    print("=" * 70)
    print("Phase 2b complete!")
    print("=" * 70)


if __name__ == '__main__':
    run_multi_ap_test()
