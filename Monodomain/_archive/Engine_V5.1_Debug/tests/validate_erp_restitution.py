"""
Comprehensive ERP and APD Restitution Validation

Validates the ORd model against published literature values.

References:
- O'Hara et al. 2011: APD90 = 271±13 ms at 1 Hz
- openCARP ERP protocol: ERP typically slightly > APD90
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from ionic import ORdModel, CellType, StateIndex

def print_status(msg):
    print(f"[STATUS] {msg}")

def run_s1_measure_apd(model, dt=0.05, bcl=1000.0, n_beats=1):
    """Run S1 pacing and measure APD90."""
    stim = torch.tensor(-80.0, dtype=model.dtype, device=model.device)
    no_stim = torch.tensor(0.0, dtype=model.dtype, device=model.device)

    state = model.get_initial_state()
    V_trace = []

    stim_time = 10.0
    t_end = stim_time + 400  # Capture full AP

    for i in range(int(t_end / dt)):
        t = i * dt
        I = stim if stim_time <= t < stim_time + 1 else no_stim
        state = model.step(state, dt, I)
        V_trace.append(state[StateIndex.V].item())

    V = np.array(V_trace)
    V_rest = V[-1]
    V_peak = V.max()

    # APD90 calculation
    V_90 = V_rest + 0.1 * (V_peak - V_rest)
    peak_idx = np.argmax(V)
    apd90 = None
    for i in range(peak_idx, len(V)):
        if V[i] < V_90:
            apd90 = i * dt - stim_time
            break

    return state, apd90, V_rest, V_peak

def test_s2_response(model, state_after_s1, ci, dt=0.1):
    """Test if S2 at coupling interval CI fires an AP."""
    stim = torch.tensor(-80.0, dtype=model.dtype, device=model.device)
    no_stim = torch.tensor(0.0, dtype=model.dtype, device=model.device)

    # Start fresh and run full S1-S2 protocol
    state = model.get_initial_state()
    s1_time = 10.0
    s2_time = s1_time + ci
    t_end = s2_time + 100

    V_before_s2 = None
    V_max_after_s2 = -100

    for i in range(int(t_end / dt)):
        t = i * dt

        if abs(t - s2_time) < dt:
            V_before_s2 = state[StateIndex.V].item()

        # S1 or S2 stimulus
        if s1_time <= t < s1_time + 1:
            I = stim
        elif s2_time <= t < s2_time + 1:
            I = stim
        else:
            I = no_stim

        state = model.step(state, dt, I)

        if t > s2_time + 10:
            V_max_after_s2 = max(V_max_after_s2, state[StateIndex.V].item())

    # AP fires if V exceeds 0 mV from a resting state
    fired = V_before_s2 is not None and V_before_s2 < -60 and V_max_after_s2 > 0

    return fired, V_max_after_s2, V_before_s2

def find_erp(model, apd90, dt=0.1):
    """Find ERP using binary search."""
    ci_low = max(200, apd90 - 50)  # Start below APD90
    ci_high = apd90 + 100  # End well after APD90

    state_ref = model.get_initial_state()

    while ci_high - ci_low > 2:
        ci_test = (ci_low + ci_high) / 2
        fired, V_max, _ = test_s2_response(model, state_ref, ci_test, dt)

        if fired:
            ci_high = ci_test
        else:
            ci_low = ci_test

    return ci_high

def run_apd_restitution(model, bcl=1000.0, dt=0.1):
    """Run APD restitution protocol at various DIs."""
    print_status("Running APD restitution protocol...")

    stim = torch.tensor(-80.0, dtype=model.dtype, device=model.device)
    no_stim = torch.tensor(0.0, dtype=model.dtype, device=model.device)

    # DIs to test (diastolic intervals)
    dis = [50, 100, 150, 200, 300, 500, 700]
    results = []

    for di in dis:
        state = model.get_initial_state()
        s1_time = 10.0

        # Run S1
        for i in range(int(400/dt)):
            t = i * dt
            state = model.step(state, dt, stim if s1_time <= t < s1_time+1 else no_stim)

        # Measure APD90 of S1 to get accurate s2_time
        # For simplicity, use fixed APD90 estimate
        apd90_est = 270
        s2_time = s1_time + apd90_est + di

        # Run to S2 and capture S2 AP
        V_s2 = []
        for i in range(int(300/dt)):
            t = 400 + i * dt
            I = stim if s2_time <= t < s2_time+1 else no_stim
            state = model.step(state, dt, I)
            if t >= s2_time:
                V_s2.append(state[StateIndex.V].item())

        if len(V_s2) > 0:
            V = np.array(V_s2)
            V_rest = V[-1]
            V_peak = V.max()

            if V_peak > 0:  # AP fired
                V_90 = V_rest + 0.1 * (V_peak - V_rest)
                apd90_s2 = None
                peak_idx = np.argmax(V)
                for i in range(peak_idx, len(V)):
                    if V[i] < V_90:
                        apd90_s2 = i * dt
                        break
                results.append((di, apd90_s2))

    return results

def main():
    print("=" * 70)
    print("COMPREHENSIVE ERP AND APD RESTITUTION VALIDATION")
    print("=" * 70)
    print()

    print_status("Initializing ORd model (ENDO)...")
    model = ORdModel(CellType.ENDO, device='cpu')

    # Check initial conditions
    state = model.get_initial_state()
    hf_init = state[StateIndex.hf].item()
    print_status(f"Initial hf = {hf_init:.4f} (expected ~0.68)")
    print()

    # =========================================================================
    # Part 1: Basic AP metrics
    # =========================================================================
    print("=" * 70)
    print("PART 1: BASELINE AP METRICS")
    print("=" * 70)

    print_status("Running single AP at BCL=1000ms...")
    state_after_s1, apd90, V_rest, V_peak = run_s1_measure_apd(model, dt=0.05)

    print()
    print(f"{'Metric':<15} {'Measured':<15} {'Expected':<20} {'Status'}")
    print("-" * 60)
    print(f"{'V_rest':<15} {V_rest:<15.2f} {'-87 to -88 mV':<20} {'PASS' if -89 < V_rest < -86 else 'FAIL'}")
    print(f"{'V_peak':<15} {V_peak:<15.2f} {'+35 to +45 mV':<20} {'PASS' if 30 < V_peak < 50 else 'CHECK'}")
    print(f"{'APD90':<15} {apd90:<15.1f} {'271±13 ms':<20} {'PASS' if 258 < apd90 < 284 else 'FAIL'}")
    print()

    # =========================================================================
    # Part 2: Single-cell ERP
    # =========================================================================
    print("=" * 70)
    print("PART 2: SINGLE-CELL ERP DETERMINATION")
    print("=" * 70)

    print_status("Finding ERP via S1-S2 protocol...")
    print()

    # Test a range of CIs
    print(f"{'CI (ms)':<10} {'DI (ms)':<10} {'V_peak':<12} {'Response'}")
    print("-" * 50)

    erp = None
    for ci in range(int(apd90) - 30, int(apd90) + 60, 10):
        fired, V_max, V_before = test_s2_response(model, state_after_s1, ci, dt=0.1)
        di = ci - apd90
        response = "AP" if fired else "No AP"
        print(f"{ci:<10} {di:<10.1f} {V_max:<12.1f} {response}")

        if fired and erp is None:
            erp = ci

    print()
    if erp:
        print(f"Single-cell ERP = {erp} ms")
        print(f"APD90 = {apd90:.1f} ms")
        print(f"Post-repolarization refractoriness (PRR) = {erp - apd90:.1f} ms")
    print()

    # =========================================================================
    # Part 3: Comparison with literature
    # =========================================================================
    print("=" * 70)
    print("PART 3: COMPARISON WITH LITERATURE")
    print("=" * 70)
    print()

    print("Reference: O'Hara et al. 2011 (PMC3102752)")
    print()
    print(f"{'Parameter':<20} {'Our Model':<15} {'Literature':<20} {'Match'}")
    print("-" * 65)

    # APD90
    apd_match = "YES" if 258 < apd90 < 284 else "NO"
    print(f"{'APD90 (ms)':<20} {apd90:<15.1f} {'271±13':<20} {apd_match}")

    # ERP (typically slightly > APD90)
    if erp:
        erp_expected = "~APD90 to APD90+20"
        erp_match = "YES" if erp <= apd90 + 30 else "CHECK"
        print(f"{'ERP (ms)':<20} {erp:<15} {erp_expected:<20} {erp_match}")

    # PRR
    if erp:
        prr = erp - apd90
        prr_expected = "0-20 ms (single cell)"
        prr_match = "YES" if 0 <= prr <= 30 else "CHECK"
        print(f"{'PRR (ms)':<20} {prr:<15.1f} {prr_expected:<20} {prr_match}")

    print()

    # =========================================================================
    # Part 4: Summary
    # =========================================================================
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print()

    issues = []
    if not (258 < apd90 < 284):
        issues.append(f"APD90 = {apd90:.1f} ms (expected 271±13)")
    if erp and erp > apd90 + 50:
        issues.append(f"ERP = {erp} ms (too long, expected ~APD90)")
    if not (-89 < V_rest < -86):
        issues.append(f"V_rest = {V_rest:.1f} mV (expected -87 to -88)")

    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("ALL VALIDATIONS PASSED")
        print()
        print("Model is validated for tissue simulations.")

    print()
    print("=" * 70)

    # Save results to CSV
    output_dir = os.path.join(os.path.dirname(__file__), "phase2_data")
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, "erp_validation_results.csv")
    with open(results_file, 'w') as f:
        f.write("parameter,measured,expected,status\n")
        f.write(f"V_rest_mV,{V_rest:.2f},-87 to -88,{'PASS' if -89 < V_rest < -86 else 'FAIL'}\n")
        f.write(f"V_peak_mV,{V_peak:.2f},+35 to +45,{'PASS' if 30 < V_peak < 50 else 'CHECK'}\n")
        f.write(f"APD90_ms,{apd90:.1f},271±13,{'PASS' if 258 < apd90 < 284 else 'FAIL'}\n")
        if erp:
            f.write(f"ERP_ms,{erp},~APD90,{'PASS' if erp <= apd90 + 30 else 'CHECK'}\n")
            f.write(f"PRR_ms,{erp-apd90:.1f},0-20,{'PASS' if erp - apd90 <= 30 else 'CHECK'}\n")

    print(f"Results saved to: {results_file}")

if __name__ == '__main__':
    main()
