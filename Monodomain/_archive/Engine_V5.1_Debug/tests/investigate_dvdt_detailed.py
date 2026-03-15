"""
Detailed Investigation: Why dV/dt_max differs from estimated INa

Our model: 348 mV/ms measured
Estimated from INa: 269 mV/ms
Missing: 79 mV/ms - likely from stimulus current!
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from ionic import ORdModel, CellType, StateIndex


def main():
    print("="*70)
    print("DETAILED dV/dt INVESTIGATION - STIMULUS TIMING")
    print("="*70)

    model = ORdModel(celltype=CellType.ENDO, device='cpu')
    state = model.get_initial_state()

    dt = 0.002
    stim_time = 5.0
    stim_duration = 1.0
    stim_amp = -80.0
    t_end = 15.0

    GNa = 75.0
    nao = 140.0

    data = {'t': [], 'V': [], 'I_stim': [], 'm': [], 'hf': [], 'j': [],
            'nai': [], 'INa_est': []}

    n_steps = int(t_end / dt)

    for i in range(n_steps):
        t = i * dt
        V = state[StateIndex.V].item()
        m = state[StateIndex.m].item()
        hf = state[StateIndex.hf].item()
        hs = state[StateIndex.hs].item()
        j = state[StateIndex.j].item()
        nai = state[StateIndex.nai].item()

        h = 0.99 * hf + 0.01 * hs
        m3hj = (m**3) * h * j
        ENa = 26.71 * np.log(nao / nai)
        INa_est = GNa * m3hj * (V - ENa)

        # Stimulus
        if stim_time <= t < stim_time + stim_duration:
            I_stim_val = stim_amp
        else:
            I_stim_val = 0.0

        data['t'].append(t)
        data['V'].append(V)
        data['I_stim'].append(I_stim_val)
        data['m'].append(m)
        data['hf'].append(hf)
        data['j'].append(j)
        data['nai'].append(nai)
        data['INa_est'].append(INa_est)

        I_stim = torch.tensor(I_stim_val, dtype=state.dtype, device=state.device)
        state = model.step(state, dt, I_stim)

    # Analysis
    t = np.array(data['t'])
    V = np.array(data['V'])
    I_stim = np.array(data['I_stim'])
    INa_est = np.array(data['INa_est'])

    dV = np.diff(V)
    dt_arr = np.diff(t)
    dV_dt = dV / dt_arr

    peak_idx = np.argmax(dV_dt)
    dV_dt_max = np.max(dV_dt)

    print(f"\n--- TIMING ANALYSIS ---")
    print(f"Stimulus: {stim_time:.1f} to {stim_time + stim_duration:.1f} ms")
    print(f"Peak dV/dt at: {t[peak_idx]:.3f} ms")
    print(f"I_stim at peak: {I_stim[peak_idx]:.1f} µA/µF")
    print()
    print(f"dV/dt_max (measured): {dV_dt_max:.1f} mV/ms")
    print(f"INa at peak: {INa_est[peak_idx]:.1f} µA/µF")
    print(f"Contribution from INa: {-INa_est[peak_idx]:.1f} mV/ms")
    print(f"Contribution from I_stim: {-I_stim[peak_idx]:.1f} mV/ms")
    print(f"Sum: {-INa_est[peak_idx] - I_stim[peak_idx]:.1f} mV/ms")
    print()

    # Find peak dV/dt AFTER stimulus ends
    stim_end_idx = int((stim_time + stim_duration) / dt)
    dV_dt_after_stim = dV_dt[stim_end_idx:]
    dV_dt_max_after = np.max(dV_dt_after_stim)
    peak_after_idx = stim_end_idx + np.argmax(dV_dt_after_stim)

    print(f"--- AFTER STIMULUS ENDS ---")
    print(f"Peak dV/dt (after stim): {dV_dt_max_after:.1f} mV/ms")
    print(f"At t = {t[peak_after_idx]:.3f} ms")
    print(f"V = {V[peak_after_idx]:.1f} mV")
    print(f"INa at peak: {INa_est[peak_after_idx]:.1f} µA/µF")

    # Find the actual pure INa peak
    print(f"\n--- FINDING PURE INa PEAK ---")
    mask_after = t >= stim_time + stim_duration
    idx_after = np.where(mask_after)[0]

    if len(idx_after) > 0:
        # Look at where INa is most negative (peak current)
        INa_after = INa_est[mask_after]
        t_after = t[mask_after]
        peak_INa_idx = np.argmin(INa_after)

        print(f"Peak INa at t = {t_after[peak_INa_idx]:.3f} ms")
        print(f"Peak INa = {INa_after[peak_INa_idx]:.1f} µA/µF")
        print(f"This would give dV/dt = {-INa_after[peak_INa_idx]:.1f} mV/ms")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print("The discrepancy is explained by stimulus timing:")
    print()
    print(f"  dV/dt_max during stimulus: {dV_dt_max:.1f} mV/ms")
    print(f"    - INa contribution: {-INa_est[peak_idx]:.1f} mV/ms")
    print(f"    - I_stim contribution: {-I_stim[peak_idx]:.1f} mV/ms")
    print(f"    - Sum: {-INa_est[peak_idx] - I_stim[peak_idx]:.1f} mV/ms")
    print()
    print(f"  dV/dt_max after stimulus: {dV_dt_max_after:.1f} mV/ms")
    print(f"    - This is the \"true\" intrinsic dV/dt_max")
    print()
    print("ORd paper likely reports dV/dt_max AFTER stimulus ends")
    print("since they measure intrinsic upstroke velocity.")


if __name__ == '__main__':
    main()
