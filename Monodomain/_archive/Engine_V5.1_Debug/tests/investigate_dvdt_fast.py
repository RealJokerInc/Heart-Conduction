"""
Fast Investigation: High dV/dt_max in ORd Model

Our model:     347 mV/ms
ORd paper:     254 mV/ms
Discrepancy:   +37%

This script focuses on first beat analysis only (fast).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from ionic import ORdModel, CellType, StateIndex


def main():
    print("="*70)
    print("FAST dV/dt_max INVESTIGATION")
    print("="*70)

    model = ORdModel(celltype=CellType.ENDO, device='cpu')
    state = model.get_initial_state()

    # Parameters
    dt = 0.002  # Fine dt for accurate measurement
    stim_time = 5.0
    t_end = 15.0  # Just capture upstroke

    GNa = 75.0
    nao = 140.0

    # Storage for upstroke data
    data = {'t': [], 'V': [], 'm': [], 'hf': [], 'hs': [], 'j': [],
            'nai': [], 'm3hj': [], 'INa_est': []}

    n_steps = int(t_end / dt)
    print(f"\nRunning {n_steps} steps at dt={dt} ms...")

    for i in range(n_steps):
        t = i * dt
        V = state[StateIndex.V].item()
        m = state[StateIndex.m].item()
        hf = state[StateIndex.hf].item()
        hs = state[StateIndex.hs].item()
        j = state[StateIndex.j].item()
        nai = state[StateIndex.nai].item()

        # h = 0.99*hf + 0.01*hs (from currents.py)
        h = 0.99 * hf + 0.01 * hs
        m3hj = (m**3) * h * j

        # ENa and estimated INa
        ENa = 26.71 * np.log(nao / nai)
        INa_est = GNa * m3hj * (V - ENa)

        data['t'].append(t)
        data['V'].append(V)
        data['m'].append(m)
        data['hf'].append(hf)
        data['hs'].append(hs)
        data['j'].append(j)
        data['nai'].append(nai)
        data['m3hj'].append(m3hj)
        data['INa_est'].append(INa_est)

        # Stimulus
        if stim_time <= t < stim_time + 1.0:
            I_stim = torch.tensor(-80.0, dtype=state.dtype, device=state.device)
        else:
            I_stim = torch.tensor(0.0, dtype=state.dtype, device=state.device)

        state = model.step(state, dt, I_stim)

    # Analyze upstroke
    t = np.array(data['t'])
    V = np.array(data['V'])

    dV = np.diff(V)
    dt_arr = np.diff(t)
    dV_dt = dV / dt_arr

    peak_idx = np.argmax(dV_dt)
    dV_dt_max = np.max(dV_dt)

    print("\n--- FIRST BEAT ANALYSIS ---")
    print(f"\ndV/dt_max = {dV_dt_max:.1f} mV/ms")
    print(f"At t = {t[peak_idx]:.3f} ms, V = {V[peak_idx]:.1f} mV")
    print()
    print("Gate values at peak dV/dt:")
    print(f"  m = {data['m'][peak_idx]:.4f}")
    print(f"  hf = {data['hf'][peak_idx]:.4f}")
    print(f"  hs = {data['hs'][peak_idx]:.4f}")
    print(f"  j = {data['j'][peak_idx]:.4f}")
    h = 0.99 * data['hf'][peak_idx] + 0.01 * data['hs'][peak_idx]
    print(f"  h = 0.99*hf + 0.01*hs = {h:.4f}")
    print(f"  m^3*h*j = {data['m3hj'][peak_idx]:.4f}")
    print(f"  nai = {data['nai'][peak_idx]:.2f} mM")

    # ENa calculation
    ENa = 26.71 * np.log(nao / data['nai'][peak_idx])
    print(f"  ENa = {ENa:.1f} mV")
    print(f"  V - ENa = {V[peak_idx] - ENa:.1f} mV (driving force)")

    print()
    print(f"Estimated INa = {data['INa_est'][peak_idx]:.1f} uA/uF")
    print(f"dV/dt = -INa/Cm = {-data['INa_est'][peak_idx]:.1f} mV/ms (Cm=1)")

    # Calculate what GNa would give 254 mV/ms
    target_dvdt = 254.0
    ratio = target_dvdt / dV_dt_max
    print(f"\nTo achieve dV/dt_max = {target_dvdt} mV/ms:")
    print(f"  Scale factor needed: {ratio:.3f}")
    print(f"  GNa_new = 75.0 x {ratio:.3f} = {75.0 * ratio:.1f} mS/uF")

    # Check initial conditions
    print("\n--- INITIAL CONDITIONS CHECK ---")
    state_init = model.get_initial_state()
    print(f"Initial hf = {state_init[StateIndex.hf].item():.4f}")
    print(f"Initial hs = {state_init[StateIndex.hs].item():.4f}")
    print(f"Initial j = {state_init[StateIndex.j].item():.4f}")
    print(f"Initial m = {state_init[StateIndex.m].item():.4f}")
    print(f"Initial V = {state_init[StateIndex.V].item():.1f} mV")

    # Expected h_inf at V=-87.5
    V_rest = -87.5
    h_inf_expected = 1.0 / (1.0 + np.exp((V_rest + 82.9) / 6.086))
    print(f"\nExpected h_inf at V={V_rest}: {h_inf_expected:.4f}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print(f"Our dV/dt_max:    {dV_dt_max:.1f} mV/ms")
    print(f"ORd paper:        254 mV/ms")
    print(f"Discrepancy:      +{100*(dV_dt_max/254-1):.0f}%")
    print()
    print("KEY OBSERVATION:")
    print(f"At peak dV/dt, m^3*h*j = {data['m3hj'][peak_idx]:.4f}")
    print(f"This directly controls INa magnitude")
    print()
    print("POSSIBLE CAUSES:")
    print("1. ORd paper may report experimental value, not model value")
    print("2. Measurement methodology differences (derivative calculation)")
    print("3. Known model limitation (ORd paper notes upstroke velocity issues)")
    print()
    print("RECOMMENDATION:")
    print("Since APD90 (272 ms) matches literature perfectly,")
    print("the elevated dV/dt_max may not significantly affect tissue behavior.")


if __name__ == '__main__':
    main()
