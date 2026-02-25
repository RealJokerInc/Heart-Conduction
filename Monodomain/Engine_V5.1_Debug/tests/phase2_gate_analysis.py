"""
Phase 2: Detailed Gate Trace Analysis

Outputs CSV files with time-series data for plotting:
- phase2_voltage.csv: Time and voltage trace
- phase2_gates_ina.csv: INa gating variables (m, hf, hs, j)
- phase2_gates_ical.csv: ICaL gating variables (d, ff, fs, fcaf, fcas)
- phase2_availability.csv: INa and ICaL availability over time
- phase2_summary.txt: Summary statistics
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from ionic import ORdModel, CellType, StateIndex

def run_phase2_analysis():
    print("=" * 70)
    print("PHASE 2: Detailed Gate Trace Analysis")
    print("=" * 70)
    print()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = ORdModel(CellType.ENDO, device=device)

    dt = 0.01  # ms
    t_end = 600.0  # ms
    stim_start, stim_duration, stim_amplitude = 10.0, 1.0, -80.0

    n_steps = int(t_end / dt)
    save_every = 10  # Save every 10 steps (0.1 ms resolution for CSV)
    n_save = n_steps // save_every

    print(f"dt = {dt} ms, t_end = {t_end} ms")
    print(f"Saving every {save_every} steps ({save_every * dt} ms resolution)")
    print(f"Total data points: {n_save}")
    print()

    # Storage arrays
    t_data = np.zeros(n_save)
    V_data = np.zeros(n_save)

    # INa gates
    m_data = np.zeros(n_save)
    hf_data = np.zeros(n_save)
    hs_data = np.zeros(n_save)
    j_data = np.zeros(n_save)
    jp_data = np.zeros(n_save)

    # ICaL gates
    d_data = np.zeros(n_save)
    ff_data = np.zeros(n_save)
    fs_data = np.zeros(n_save)
    fcaf_data = np.zeros(n_save)
    fcas_data = np.zeros(n_save)

    # Full resolution for dV/dt calculation
    V_full = np.zeros(n_steps)

    state = model.get_initial_state()

    print("Running simulation...")
    save_idx = 0

    for i in range(n_steps):
        t = i * dt

        # Stimulus
        if stim_start <= t < stim_start + stim_duration:
            I_stim = torch.tensor(stim_amplitude, dtype=model.dtype, device=model.device)
        else:
            I_stim = torch.tensor(0.0, dtype=model.dtype, device=model.device)

        state = model.step(state, dt, I_stim)

        V_full[i] = state[StateIndex.V].item()

        # Save at reduced resolution
        if i % save_every == 0 and save_idx < n_save:
            t_data[save_idx] = t
            V_data[save_idx] = state[StateIndex.V].item()

            # INa gates
            m_data[save_idx] = state[StateIndex.m].item()
            hf_data[save_idx] = state[StateIndex.hf].item()
            hs_data[save_idx] = state[StateIndex.hs].item()
            j_data[save_idx] = state[StateIndex.j].item()
            jp_data[save_idx] = state[StateIndex.jp].item()

            # ICaL gates
            d_data[save_idx] = state[StateIndex.d].item()
            ff_data[save_idx] = state[StateIndex.ff].item()
            fs_data[save_idx] = state[StateIndex.fs].item()
            fcaf_data[save_idx] = state[StateIndex.fcaf].item()
            fcas_data[save_idx] = state[StateIndex.fcas].item()

            save_idx += 1

    print("Simulation complete.")
    print()

    # Compute derived quantities
    # INa availability: m^3 * h * j where h = 0.99*hf + 0.01*hs
    h_data = 0.99 * hf_data + 0.01 * hs_data
    INa_avail = (m_data ** 3) * h_data * j_data

    # ICaL availability: d * f * fca where f = 0.4*ff + 0.6*fs, fca = 0.3*fcaf + 0.7*fcas
    f_data = 0.4 * ff_data + 0.6 * fs_data
    fca_data = 0.3 * fcaf_data + 0.7 * fcas_data
    ICaL_avail = d_data * f_data * fca_data

    # Compute dV/dt
    dVdt_full = np.diff(V_full) / dt
    dVdt_max = np.max(dVdt_full)
    upstroke_idx = np.argmax(dVdt_full)
    t_upstroke = upstroke_idx * dt

    # Output directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(output_dir, "phase2_data")
    os.makedirs(data_dir, exist_ok=True)

    print(f"Saving data to: {data_dir}")
    print()

    # Save voltage trace
    voltage_file = os.path.join(data_dir, "phase2_voltage.csv")
    np.savetxt(voltage_file,
               np.column_stack([t_data, V_data]),
               delimiter=',',
               header='time_ms,voltage_mV',
               comments='')
    print(f"  Saved: phase2_voltage.csv")

    # Save INa gates
    ina_file = os.path.join(data_dir, "phase2_gates_ina.csv")
    np.savetxt(ina_file,
               np.column_stack([t_data, m_data, hf_data, hs_data, j_data, jp_data, h_data, INa_avail]),
               delimiter=',',
               header='time_ms,m,hf,hs,j,jp,h_combined,INa_availability',
               comments='')
    print(f"  Saved: phase2_gates_ina.csv")

    # Save ICaL gates
    ical_file = os.path.join(data_dir, "phase2_gates_ical.csv")
    np.savetxt(ical_file,
               np.column_stack([t_data, d_data, ff_data, fs_data, fcaf_data, fcas_data, f_data, fca_data, ICaL_avail]),
               delimiter=',',
               header='time_ms,d,ff,fs,fcaf,fcas,f_combined,fca_combined,ICaL_availability',
               comments='')
    print(f"  Saved: phase2_gates_ical.csv")

    # Save availability summary
    avail_file = os.path.join(data_dir, "phase2_availability.csv")
    np.savetxt(avail_file,
               np.column_stack([t_data, INa_avail, ICaL_avail]),
               delimiter=',',
               header='time_ms,INa_availability,ICaL_availability',
               comments='')
    print(f"  Saved: phase2_availability.csv")

    print()

    # Compute summary statistics
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print()

    # Voltage metrics
    V_rest = V_data[0]
    V_peak = V_data.max()
    V_90 = V_rest + 0.1 * (V_peak - V_rest)

    # Find APD90
    peak_idx = np.argmax(V_data)
    apd90_idx = None
    for i in range(peak_idx, len(V_data)):
        if V_data[i] < V_90:
            apd90_idx = i
            break

    APD90 = t_data[apd90_idx] - t_upstroke if apd90_idx else None

    print("VOLTAGE METRICS:")
    print(f"  V_rest      = {V_rest:.2f} mV")
    print(f"  V_peak      = {V_peak:.2f} mV")
    print(f"  dV/dt_max   = {dVdt_max:.1f} mV/ms")
    print(f"  t_upstroke  = {t_upstroke:.2f} ms")
    print(f"  APD90       = {APD90:.1f} ms" if APD90 else "  APD90       = NOT REACHED")
    print()

    # Gate statistics
    print("GATE VALUES (Initial -> Minimum -> Final @ 600ms -> Recovery %):")
    print("-" * 70)

    gates_info = [
        ("m (INa act)", m_data),
        ("hf (INa fast)", hf_data),
        ("hs (INa slow)", hs_data),
        ("j (INa recov)", j_data),
        ("d (ICaL act)", d_data),
        ("ff (ICaL fast)", ff_data),
        ("fcaf (ICaL Ca)", fcaf_data),
    ]

    gate_summary = []
    for name, data in gates_info:
        initial = data[0]
        minimum = data.min()
        final = data[-1]
        recovery_pct = (final / initial * 100) if initial > 0.001 else 0

        # Find time to 50% and 90% recovery (after minimum)
        min_idx = np.argmin(data)
        t50, t90 = None, None

        if initial > 0.001:
            target_50 = initial * 0.5
            target_90 = initial * 0.9

            for i in range(min_idx, len(data)):
                if t50 is None and data[i] >= target_50:
                    t50 = t_data[i]
                if t90 is None and data[i] >= target_90:
                    t90 = t_data[i]

        gate_summary.append({
            'name': name,
            'initial': initial,
            'minimum': minimum,
            'final': final,
            'recovery_pct': recovery_pct,
            't50': t50,
            't90': t90
        })

        print(f"  {name:<15}: {initial:.4f} -> {minimum:.4f} -> {final:.4f} ({recovery_pct:.1f}%)")
        if t50:
            print(f"                   50% recovery at t={t50:.1f}ms, 90% at t={t90:.1f}ms" if t90 else f"                   50% recovery at t={t50:.1f}ms, 90% NOT REACHED")
        elif initial > 0.001:
            print(f"                   50% recovery NOT REACHED")

    print()

    # INa availability analysis
    print("INa AVAILABILITY (m³·h·j):")
    print(f"  Initial     = {INa_avail[0]:.6f}")
    print(f"  Minimum     = {INa_avail.min():.6f} (at t={t_data[np.argmin(INa_avail)]:.1f} ms)")
    print(f"  Final       = {INa_avail[-1]:.6f}")
    print(f"  Recovery    = {INa_avail[-1]/INa_avail[0]*100:.1f}%")
    print()

    # ICaL availability analysis
    print("ICaL AVAILABILITY (d·f·fca):")
    print(f"  Initial     = {ICaL_avail[0]:.6f}")
    print(f"  Minimum     = {ICaL_avail.min():.6f} (at t={t_data[np.argmin(ICaL_avail)]:.1f} ms)")
    print(f"  Final       = {ICaL_avail[-1]:.6f}")
    print(f"  Recovery    = {ICaL_avail[-1]/ICaL_avail[0]*100:.1f}%")
    print()

    # Save summary to text file
    summary_file = os.path.join(data_dir, "phase2_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("PHASE 2: Detailed Gate Trace Analysis - Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: Generated by phase2_gate_analysis.py\n")
        f.write(f"Cell Type: ORd ENDO\n")
        f.write(f"Device: {device}\n\n")

        f.write("SIMULATION PARAMETERS:\n")
        f.write(f"  dt = {dt} ms\n")
        f.write(f"  t_end = {t_end} ms\n")
        f.write(f"  Stimulus: {stim_amplitude} uA/uF for {stim_duration} ms at t={stim_start} ms\n\n")

        f.write("VOLTAGE METRICS:\n")
        f.write(f"  V_rest = {V_rest:.2f} mV\n")
        f.write(f"  V_peak = {V_peak:.2f} mV\n")
        f.write(f"  dV/dt_max = {dVdt_max:.1f} mV/ms\n")
        f.write(f"  t_upstroke = {t_upstroke:.2f} ms\n")
        f.write(f"  APD90 = {APD90:.1f} ms\n\n" if APD90 else "  APD90 = NOT REACHED\n\n")

        f.write("GATE RECOVERY SUMMARY:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Gate':<15} {'Initial':>10} {'Minimum':>10} {'Final':>10} {'Recovery%':>10} {'t_90%':>10}\n")
        f.write("-" * 70 + "\n")
        for g in gate_summary:
            t90_str = f"{g['t90']:.1f}" if g['t90'] else "N/A"
            f.write(f"{g['name']:<15} {g['initial']:>10.4f} {g['minimum']:>10.4f} {g['final']:>10.4f} {g['recovery_pct']:>9.1f}% {t90_str:>10}\n")
        f.write("\n")

        f.write("INa AVAILABILITY (m³·h·j):\n")
        f.write(f"  Initial = {INa_avail[0]:.6f}\n")
        f.write(f"  Minimum = {INa_avail.min():.6f}\n")
        f.write(f"  Final = {INa_avail[-1]:.6f}\n")
        f.write(f"  Recovery = {INa_avail[-1]/INa_avail[0]*100:.1f}%\n\n")

        f.write("ICaL AVAILABILITY (d·f·fca):\n")
        f.write(f"  Initial = {ICaL_avail[0]:.6f}\n")
        f.write(f"  Minimum = {ICaL_avail.min():.6f}\n")
        f.write(f"  Final = {ICaL_avail[-1]:.6f}\n")
        f.write(f"  Recovery = {ICaL_avail[-1]/ICaL_avail[0]*100:.1f}%\n")

    print(f"  Saved: phase2_summary.txt")
    print()
    print("=" * 70)
    print("Phase 2 analysis complete!")
    print("=" * 70)


if __name__ == '__main__':
    run_phase2_analysis()
