#!/usr/bin/env python3
"""
Stage 1 Validation Script

Validates that both ionic models (ORd and TTP06) produce physiologically
correct action potentials with appropriate cell-type differentiation.

Tests:
1. ORd model - 3 cell types (ENDO, EPI, M-cell)
2. TTP06 model - 3 cell types (ENDO, EPI, M-cell)

Expected characteristics:
- Resting potential: -85 to -90 mV
- Peak potential: 20-50 mV
- APD90: 200-350 ms depending on model and cell type
- M-cells should have longest APD (reduced IKs)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from ionic import ORdModel, TTP06Model, CellType


def run_single_ap(model, dt=0.01, stim_duration=1.0, stim_amplitude=-52.0,
                  total_duration=500.0):
    """
    Run a single action potential simulation.

    Parameters
    ----------
    model : IonicModel
        Ionic model instance
    dt : float
        Time step (ms)
    stim_duration : float
        Stimulus duration (ms)
    stim_amplitude : float
        Stimulus amplitude (pA/pF, negative = depolarizing)
    total_duration : float
        Total simulation duration (ms)

    Returns
    -------
    dict
        Results including V_rest, V_peak, APD90, time trace, voltage trace
    """
    state = model.get_initial_state()
    states = state.clone()

    n_stim = int(stim_duration / dt)
    n_total = int(total_duration / dt)

    V_trace = [states[model.V_index].item()]
    t_trace = [0.0]

    # Stimulation phase
    I_stim = torch.tensor(stim_amplitude)
    for i in range(n_stim):
        states = model.step(states, dt, I_stim)
        V_trace.append(states[model.V_index].item())
        t_trace.append((i + 1) * dt)

    # Post-stimulus phase
    for i in range(n_total - n_stim):
        states = model.step(states, dt, None)
        V_trace.append(states[model.V_index].item())
        t_trace.append((n_stim + i + 1) * dt)

    V_trace = torch.tensor(V_trace)
    t_trace = torch.tensor(t_trace)

    # Analyze
    V_rest = V_trace[0].item()
    V_peak = V_trace.max().item()
    t_peak = t_trace[V_trace.argmax()].item()
    V_final = V_trace[-1].item()

    # APD90 calculation
    V90 = V_final + 0.1 * (V_peak - V_final)
    APD90 = float('nan')
    for i in range(V_trace.argmax().item(), len(V_trace)):
        if V_trace[i] < V90:
            APD90 = t_trace[i].item()
            break

    return {
        'V_rest': V_rest,
        'V_peak': V_peak,
        't_peak': t_peak,
        'V_final': V_final,
        'APD90': APD90,
        't': t_trace.numpy(),
        'V': V_trace.numpy(),
    }


def validate_ap(result, model_name, celltype,
                V_rest_range=(-92, -80), V_peak_range=(15, 55), APD90_range=(150, 450)):
    """
    Validate action potential characteristics.
    """
    errors = []

    if not (V_rest_range[0] <= result['V_rest'] <= V_rest_range[1]):
        errors.append(f"V_rest={result['V_rest']:.1f} mV outside range {V_rest_range}")

    if not (V_peak_range[0] <= result['V_peak'] <= V_peak_range[1]):
        errors.append(f"V_peak={result['V_peak']:.1f} mV outside range {V_peak_range}")

    if not (APD90_range[0] <= result['APD90'] <= APD90_range[1]):
        errors.append(f"APD90={result['APD90']:.0f} ms outside range {APD90_range}")

    return errors


def main():
    device = torch.device('cpu')
    print("=" * 70)
    print("Stage 1 Validation: Ionic Models")
    print("=" * 70)

    results = {}
    all_passed = True

    # Test ORd model
    print("\n[ORd Model - O'Hara-Rudy 2011]")
    print("-" * 40)
    for ctype in [CellType.ENDO, CellType.EPI, CellType.M_CELL]:
        model = ORdModel(celltype=ctype, device=device)
        result = run_single_ap(model, dt=0.01, total_duration=500.0)
        results[f'ORd_{ctype.name}'] = result

        errors = validate_ap(result, 'ORd', ctype)
        status = "PASS" if not errors else "FAIL"
        print(f"  {ctype.name:8s}: V_rest={result['V_rest']:6.1f} mV, "
              f"V_peak={result['V_peak']:5.1f} mV, APD90={result['APD90']:5.0f} ms [{status}]")

        if errors:
            all_passed = False
            for e in errors:
                print(f"           ERROR: {e}")

    # Verify M-cell has longest APD
    ord_apds = [results[f'ORd_{ct.name}']['APD90'] for ct in CellType]
    if results['ORd_M_CELL']['APD90'] < max(ord_apds) * 0.95:
        print("  WARNING: M-cell APD is not longest as expected")

    # Test TTP06 model
    print("\n[TTP06 Model - ten Tusscher-Panfilov 2006]")
    print("-" * 40)
    for ctype in [CellType.ENDO, CellType.EPI, CellType.M_CELL]:
        model = TTP06Model(celltype=ctype, device=device)
        result = run_single_ap(model, dt=0.01, total_duration=500.0)
        results[f'TTP06_{ctype.name}'] = result

        errors = validate_ap(result, 'TTP06', ctype)
        status = "PASS" if not errors else "FAIL"
        print(f"  {ctype.name:8s}: V_rest={result['V_rest']:6.1f} mV, "
              f"V_peak={result['V_peak']:5.1f} mV, APD90={result['APD90']:5.0f} ms [{status}]")

        if errors:
            all_passed = False
            for e in errors:
                print(f"           ERROR: {e}")

    # Verify M-cell has longest APD
    ttp06_apds = [results[f'TTP06_{ct.name}']['APD90'] for ct in CellType]
    if results['TTP06_M_CELL']['APD90'] < max(ttp06_apds) * 0.95:
        print("  WARNING: M-cell APD is not longest as expected")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"ORd   - ENDO: {results['ORd_ENDO']['APD90']:5.0f} ms, "
          f"EPI: {results['ORd_EPI']['APD90']:5.0f} ms, "
          f"M-cell: {results['ORd_M_CELL']['APD90']:5.0f} ms")
    print(f"TTP06 - ENDO: {results['TTP06_ENDO']['APD90']:5.0f} ms, "
          f"EPI: {results['TTP06_EPI']['APD90']:5.0f} ms, "
          f"M-cell: {results['TTP06_M_CELL']['APD90']:5.0f} ms")
    print()

    if all_passed:
        print("All validations PASSED")
        return 0
    else:
        print("Some validations FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
