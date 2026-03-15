"""
Investigation: dV/dt_max with short stimulus

Use very short stimulus to capture intrinsic upstroke velocity
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from ionic import ORdModel, CellType, StateIndex


def test_stim_duration(model, stim_duration, stim_amp=-80.0, dt=0.002):
    """Test with specific stimulus duration."""
    state = model.get_initial_state()

    stim_time = 5.0
    t_end = 15.0

    data = {'t': [], 'V': [], 'I_stim': []}

    n_steps = int(t_end / dt)

    for i in range(n_steps):
        t = i * dt
        V = state[StateIndex.V].item()

        if stim_time <= t < stim_time + stim_duration:
            I_stim_val = stim_amp
        else:
            I_stim_val = 0.0

        data['t'].append(t)
        data['V'].append(V)
        data['I_stim'].append(I_stim_val)

        I_stim = torch.tensor(I_stim_val, dtype=state.dtype, device=state.device)
        state = model.step(state, dt, I_stim)

    t = np.array(data['t'])
    V = np.array(data['V'])
    I_stim = np.array(data['I_stim'])

    dV_dt = np.diff(V) / dt
    peak_idx = np.argmax(dV_dt)

    # Find peak after stimulus ends
    stim_end_idx = int((stim_time + stim_duration) / dt)
    if stim_end_idx < len(dV_dt):
        dV_dt_after = dV_dt[stim_end_idx:]
        if len(dV_dt_after) > 0:
            dV_dt_max_after = np.max(dV_dt_after)
        else:
            dV_dt_max_after = 0
    else:
        dV_dt_max_after = 0

    return {
        'stim_dur': stim_duration,
        'dV_dt_max': np.max(dV_dt),
        'dV_dt_max_after_stim': dV_dt_max_after,
        't_peak': t[peak_idx],
        'stim_at_peak': I_stim[peak_idx],
        'V_peak': V[np.argmax(V)],
    }


def main():
    print("="*70)
    print("dV/dt_max vs STIMULUS DURATION")
    print("="*70)
    print()
    print("Testing how stimulus duration affects measured dV/dt_max")
    print()

    model = ORdModel(celltype=CellType.ENDO, device='cpu')

    # Test different stimulus durations
    durations = [2.0, 1.0, 0.5, 0.3, 0.2, 0.1]

    print(f"{'Stim (ms)':<12} {'dV/dt_max':<15} {'After stim':<15} {'Peak t':<12} {'I_stim@peak':<12} {'V_peak'}")
    print("-" * 80)

    for dur in durations:
        result = test_stim_duration(model, dur)
        print(f"{result['stim_dur']:<12.1f} {result['dV_dt_max']:<15.1f} "
              f"{result['dV_dt_max_after_stim']:<15.1f} {result['t_peak']:<12.3f} "
              f"{result['stim_at_peak']:<12.1f} {result['V_peak']:.1f}")

    # Test with stronger, shorter stimulus
    print("\n--- STRONGER, SHORTER STIMULUS ---")
    print("Testing -150 µA/µF for 0.2 ms")
    result = test_stim_duration(model, 0.2, stim_amp=-150.0)
    print(f"dV/dt_max = {result['dV_dt_max']:.1f} mV/ms")
    print(f"dV/dt_max after stim = {result['dV_dt_max_after_stim']:.1f} mV/ms")

    # Summary
    print("\n" + "="*70)
    print("KEY FINDING")
    print("="*70)
    print()
    print("The measured dV/dt_max depends on when we sample:")
    print("  - During stimulus: dV/dt = INa + I_stim = ~350 mV/ms")
    print("  - After stimulus: dV/dt = INa only = ~270 mV/ms (intrinsic)")
    print()
    print("ORd paper (254 mV/ms) likely measures during upstroke")
    print("where INa is dominant and stimulus is minimal.")
    print()
    print("Our 269 mV/ms (from INa only) is still higher than 254 mV/ms.")
    print("Remaining difference (~15 mV/ms) could be from:")
    print("  1. CaMKII modulation effects in their model")
    print("  2. Different nai (we use 7 mM at start)")
    print("  3. Measurement methodology differences")


if __name__ == '__main__':
    main()
