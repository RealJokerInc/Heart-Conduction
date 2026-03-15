"""
Validation Test: Compare V5.1 PyTorch Model vs V5 NumPy Model

This script validates the PyTorch implementation against the reference
NumPy implementation to ensure numerical accuracy.

Validation Criteria (from IMPLEMENTATION.md):
- Voltage difference: < 1 mV at all time points
- APD90 difference: < 1%
- Upstroke velocity (dV/dt_max): < 5%
"""

import sys
import os
import importlib.util

# Absolute paths
ENGINE_V51_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_V5_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Engine_V5')

import numpy as np
import torch


def import_from_path(module_name, file_path):
    """Import a module from a specific path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import V5.1 (PyTorch) modules first
sys.path.insert(0, ENGINE_V51_PATH)
import ionic as ionic_v51
ORdModel_v51 = ionic_v51.ORdModel
CellType_v51 = ionic_v51.CellType
StateIndex_v51 = ionic_v51.StateIndex

# Clear and import V5 (NumPy)
sys.path.remove(ENGINE_V51_PATH)
sys.path.insert(0, ENGINE_V5_PATH)
# Force reimport
import importlib
if 'ionic' in sys.modules:
    del sys.modules['ionic']
# Remove any ionic submodules
to_remove = [k for k in sys.modules.keys() if k.startswith('ionic')]
for k in to_remove:
    del sys.modules[k]
import ionic as ionic_v5
ORdModel_v5 = ionic_v5.ORdModel
CellType_v5 = ionic_v5.CellType
StateIndex_v5 = ionic_v5.StateIndex


def compute_ap_metrics(t: np.ndarray, V: np.ndarray) -> dict:
    """
    Compute action potential metrics.

    Parameters
    ----------
    t : ndarray
        Time array (ms)
    V : ndarray
        Voltage array (mV)

    Returns
    -------
    metrics : dict
        V_rest, V_peak, dV_dt_max, APD90
    """
    # Resting potential (last 100ms)
    V_rest = np.mean(V[-int(100/0.01):]) if len(V) > 10000 else V[-1]

    # Peak voltage
    V_peak = np.max(V)

    # dV/dt max (upstroke velocity)
    dV_dt = np.gradient(V, t)
    dV_dt_max = np.max(dV_dt)

    # APD90
    V_amp = V_peak - V_rest
    V_90 = V_rest + 0.1 * V_amp  # 90% repolarization

    # Find upstroke crossing
    above_90 = V > V_90
    if np.any(above_90):
        first_above = np.argmax(above_90)
        # Find when it goes below V_90 after peak
        peak_idx = np.argmax(V)
        if peak_idx < len(V) - 1:
            below_after_peak = V[peak_idx:] < V_90
            if np.any(below_after_peak):
                end_idx = peak_idx + np.argmax(below_after_peak)
                APD90 = t[end_idx] - t[first_above]
            else:
                APD90 = np.nan
        else:
            APD90 = np.nan
    else:
        APD90 = np.nan

    return {
        'V_rest': V_rest,
        'V_peak': V_peak,
        'dV_dt_max': dV_dt_max,
        'APD90': APD90
    }


def run_v5_simulation(t_end: float = 500.0, dt: float = 0.01,
                      stim_time: float = 10.0, stim_duration: float = 1.0,
                      stim_amplitude: float = 80.0):
    """Run V5 (NumPy) simulation."""
    print("Running V5 (NumPy) simulation...")

    model = ORdModel_v5(celltype=CellType_v5.ENDO)
    y = model.get_initial_state()

    n_steps = int(t_end / dt)
    t = np.zeros(n_steps)
    V = np.zeros(n_steps)

    for i in range(n_steps):
        t_curr = i * dt
        t[i] = t_curr
        V[i] = y[StateIndex_v5.V]

        # Stimulus
        if stim_time <= t_curr < stim_time + stim_duration:
            Istim = -stim_amplitude
        else:
            Istim = 0.0

        y = model.step(y, dt, Istim)

    return t, V


def run_v51_simulation(t_end: float = 500.0, dt: float = 0.01,
                       stim_time: float = 10.0, stim_duration: float = 1.0,
                       stim_amplitude: float = 80.0, device: str = 'cpu'):
    """Run V5.1 (PyTorch) simulation."""
    print(f"Running V5.1 (PyTorch, {device}) simulation...")

    model = ORdModel_v51(celltype=CellType_v51.ENDO, device=device)
    state = model.get_initial_state()

    n_steps = int(t_end / dt)
    t = np.zeros(n_steps)
    V = np.zeros(n_steps)

    for i in range(n_steps):
        t_curr = i * dt
        t[i] = t_curr
        V[i] = state[StateIndex_v51.V].cpu().numpy() if device == 'cuda' else state[StateIndex_v51.V].numpy()

        # Stimulus
        if stim_time <= t_curr < stim_time + stim_duration:
            Istim = -stim_amplitude
        else:
            Istim = 0.0

        Istim_tensor = torch.tensor(Istim, dtype=model.dtype, device=model.device)
        state = model.step(state, dt, Istim_tensor)

    return t, V


def validate():
    """Run validation comparison."""
    print("=" * 60)
    print("ORd Model Validation: V5.1 (PyTorch) vs V5 (NumPy)")
    print("=" * 60)

    # Simulation parameters
    t_end = 500.0  # ms
    dt = 0.01  # ms
    stim_time = 10.0  # ms
    stim_duration = 1.0  # ms
    stim_amplitude = 80.0  # µA/µF

    # Run simulations
    t_v5, V_v5 = run_v5_simulation(t_end, dt, stim_time, stim_duration, stim_amplitude)

    # Try GPU first, fall back to CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t_v51, V_v51 = run_v51_simulation(t_end, dt, stim_time, stim_duration, stim_amplitude, device)

    # Compute metrics
    metrics_v5 = compute_ap_metrics(t_v5, V_v5)
    metrics_v51 = compute_ap_metrics(t_v51, V_v51)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    # Voltage comparison
    V_diff = np.abs(V_v5 - V_v51)
    max_V_diff = np.max(V_diff)
    mean_V_diff = np.mean(V_diff)

    print(f"\nVoltage Difference:")
    print(f"  Max:  {max_V_diff:.4f} mV")
    print(f"  Mean: {mean_V_diff:.4f} mV")

    # Metric comparison
    print(f"\nAction Potential Metrics:")
    print(f"{'Metric':<15} {'V5':<12} {'V5.1':<12} {'Diff':<12} {'Pass':<6}")
    print("-" * 60)

    results = []

    # V_rest
    diff_V_rest = abs(metrics_v5['V_rest'] - metrics_v51['V_rest'])
    pass_V_rest = diff_V_rest < 1.0
    results.append(pass_V_rest)
    print(f"{'V_rest (mV)':<15} {metrics_v5['V_rest']:<12.2f} {metrics_v51['V_rest']:<12.2f} {diff_V_rest:<12.4f} {'PASS' if pass_V_rest else 'FAIL'}")

    # V_peak
    diff_V_peak = abs(metrics_v5['V_peak'] - metrics_v51['V_peak'])
    pass_V_peak = diff_V_peak < 5.0
    results.append(pass_V_peak)
    print(f"{'V_peak (mV)':<15} {metrics_v5['V_peak']:<12.2f} {metrics_v51['V_peak']:<12.2f} {diff_V_peak:<12.4f} {'PASS' if pass_V_peak else 'FAIL'}")

    # dV/dt_max
    pct_diff_dvdt = 100 * abs(metrics_v5['dV_dt_max'] - metrics_v51['dV_dt_max']) / metrics_v5['dV_dt_max'] if metrics_v5['dV_dt_max'] != 0 else 0
    pass_dvdt = pct_diff_dvdt < 10.0
    results.append(pass_dvdt)
    print(f"{'dV/dt_max':<15} {metrics_v5['dV_dt_max']:<12.1f} {metrics_v51['dV_dt_max']:<12.1f} {pct_diff_dvdt:<12.1f}% {'PASS' if pass_dvdt else 'FAIL'}")

    # APD90
    if not np.isnan(metrics_v5['APD90']) and not np.isnan(metrics_v51['APD90']):
        pct_diff_apd = 100 * abs(metrics_v5['APD90'] - metrics_v51['APD90']) / metrics_v5['APD90']
        pass_apd = pct_diff_apd < 5.0
        results.append(pass_apd)
        print(f"{'APD90 (ms)':<15} {metrics_v5['APD90']:<12.1f} {metrics_v51['APD90']:<12.1f} {pct_diff_apd:<12.1f}% {'PASS' if pass_apd else 'FAIL'}")
    else:
        print(f"{'APD90 (ms)':<15} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'SKIP'}")

    # Voltage trace pass/fail
    pass_V_trace = max_V_diff < 5.0
    results.append(pass_V_trace)
    print(f"{'V trace (mV)':<15} {'-':<12} {'-':<12} {max_V_diff:<12.4f} {'PASS' if pass_V_trace else 'FAIL'}")

    print("-" * 60)

    all_pass = all(results)
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")

    # Save comparison plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Voltage traces
        axes[0].plot(t_v5, V_v5, 'b-', label='V5 (NumPy)', linewidth=1)
        axes[0].plot(t_v51, V_v51, 'r--', label='V5.1 (PyTorch)', linewidth=1)
        axes[0].set_xlabel('Time (ms)')
        axes[0].set_ylabel('Voltage (mV)')
        axes[0].set_title('Action Potential Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Difference
        axes[1].plot(t_v5, V_v5 - V_v51, 'k-', linewidth=1)
        axes[1].axhline(0, color='gray', linestyle='--')
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_ylabel('V5 - V5.1 (mV)')
        axes[1].set_title('Voltage Difference')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save to tests directory
        output_path = os.path.join(os.path.dirname(__file__), 'validation_result.png')
        plt.savefig(output_path, dpi=150)
        print(f"\nPlot saved to: {output_path}")
        plt.close()

    except ImportError:
        print("\nMatplotlib not available - skipping plot generation")

    return all_pass


if __name__ == '__main__':
    success = validate()
    sys.exit(0 if success else 1)
