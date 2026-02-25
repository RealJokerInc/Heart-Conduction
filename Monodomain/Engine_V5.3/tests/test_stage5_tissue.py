#!/usr/bin/env python3
"""
Stage 5 Validation: Full Integration (Tissue Simulation)

Tests:
5.1: Single cell TTP06 matches Stage 1 results
5.2: Wave propagation in 1D cable
5.3: Conduction velocity measurement
5.4: Stimulus localization
5.5: CN vs BDF2 cross-validation at tissue level
5.6: 2D wavefront shape
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


def print_test_header(test_id: str, description: str):
    """Print test header."""
    print(f"\n[Test {test_id}] {description}")
    print("-" * 60)


def print_result(passed: bool, message: str = ""):
    """Print test result."""
    status = "PASS" if passed else "FAIL"
    symbol = "✓" if passed else "✗"
    if message:
        print(f"  {symbol} {status}: {message}")
    else:
        print(f"  {symbol} {status}")
    return passed


def test_single_cell_ttp06():
    """
    Test 5.1: Single cell TTP06 simulation matches Stage 1 results.

    Uses minimal mesh (single element) to test ionic model integration.
    """
    print_test_header("5.1", "Single cell TTP06")

    from fem import TriangularMesh
    from ionic import TTP06Model, CellType
    from tissue import MonodomainSimulation
    from tissue.simulation import SimulationConfig

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Minimal mesh - effectively single cell (no spatial gradients)
    mesh = TriangularMesh.create_rectangle(0.01, 0.01, 3, 3, device=device)

    # TTP06 model
    model = TTP06Model(celltype=CellType.EPI, device=device)

    # Very small D to minimize diffusion effects
    config = SimulationConfig(D=1e-10, dt=0.02, save_interval=1.0)
    sim = MonodomainSimulation(mesh, model, config)

    # Stimulus at t=10 ms
    sim.add_stimulus(lambda x, y: torch.ones_like(x, dtype=torch.bool),
                     start_time=10.0, duration=1.0, amplitude=-52.0)

    # Run for 500 ms
    times, voltages = sim.run(500.0)

    # Get center node voltage trace
    center_idx = mesh.n_nodes // 2
    V = voltages[:, center_idx]

    # Check AP characteristics
    V_rest = V[0]
    V_peak = V.max()
    V_min = V.min()

    # Find APD90
    V_90 = V_rest + 0.1 * (V_peak - V_rest)
    above_90 = np.where(V > V_90)[0]
    if len(above_90) > 1:
        apd90 = times[above_90[-1]] - times[above_90[0]]
    else:
        apd90 = 0

    passed = (-90 < V_rest < -80) and (V_peak > 20) and (200 < apd90 < 400)

    print(f"  V_rest = {V_rest:.1f} mV (expected -86 ± 5)")
    print(f"  V_peak = {V_peak:.1f} mV (expected > 20)")
    print(f"  APD90 ≈ {apd90:.0f} ms (expected 200-400)")

    return print_result(passed, f"Vrest={V_rest:.1f}, Vpeak={V_peak:.1f}, APD90≈{apd90:.0f}")


def test_wave_propagation():
    """
    Test 5.2: Wave propagates through 1D cable.

    D needs to be large enough to overcome chi*Cm scaling.
    For CV ~60 cm/s with chi=1400, Cm=1: D ≈ 1-3 cm²/ms
    """
    print_test_header("5.2", "Wave propagation in 1D cable")

    from fem import TriangularMesh
    from ionic import TTP06Model, CellType
    from tissue import MonodomainSimulation
    from tissue.simulation import SimulationConfig

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1D-like cable: 2 cm x 0.1 cm, finer mesh
    mesh = TriangularMesh.create_rectangle(2.0, 0.1, 201, 6, device=device)

    model = TTP06Model(celltype=CellType.EPI, device=device)

    # D=1.5 cm²/ms gives reasonable CV with chi=1400
    config = SimulationConfig(D=1.5, chi=1400.0, dt=0.01, save_interval=0.5)
    sim = MonodomainSimulation(mesh, model, config)

    # Stimulus at left edge
    sim.add_stimulus(lambda x, y: x < 0.1, start_time=1.0, duration=1.0)

    # Run for 40 ms
    times, voltages = sim.run(40.0)

    # Check that wave has propagated
    x = mesh.nodes[:, 0].cpu().numpy()

    # Find nodes near left (x=0.3) and middle (x=1.0)
    left_idx = np.argmin(np.abs(x - 0.3))
    mid_idx = np.argmin(np.abs(x - 1.0))

    V_left = voltages[:, left_idx]
    V_mid = voltages[:, mid_idx]

    # Both should have activated
    left_activated = V_left.max() > 0
    mid_activated = V_mid.max() > -40

    passed = left_activated and mid_activated

    print(f"  Left (x=0.3) V_max = {V_left.max():.1f} mV")
    print(f"  Mid (x=1.0) V_max = {V_mid.max():.1f} mV")
    print(f"  Wave propagated: {passed}")

    return print_result(passed, f"Left Vmax={V_left.max():.1f}, Mid Vmax={V_mid.max():.1f}")


def test_cv_measurement():
    """
    Test 5.3: Conduction velocity is in physiological range.

    With D=1.5 cm²/ms and chi=1400: expect CV ~0.5-1.0 m/s
    """
    print_test_header("5.3", "Conduction velocity measurement")

    from fem import TriangularMesh
    from ionic import TTP06Model, CellType
    from tissue import MonodomainSimulation
    from tissue.simulation import SimulationConfig

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Longer cable for accurate CV
    mesh = TriangularMesh.create_rectangle(3.0, 0.1, 151, 6, device=device)

    model = TTP06Model(celltype=CellType.EPI, device=device)

    # D=1.5 cm²/ms with chi=1400 gives CV ~0.5-1.0 m/s
    config = SimulationConfig(D=1.5, chi=1400.0, dt=0.01, save_interval=0.5)
    sim = MonodomainSimulation(mesh, model, config)

    # Stimulus at left edge
    sim.add_stimulus(lambda x, y: x < 0.1, start_time=1.0, duration=1.0)

    # Run until wave reaches far end
    times, voltages = sim.run(50.0)

    # Compute CV between x=0.5 and x=2.5 (middle of cable)
    cv = sim.compute_cv(voltages, times, 0.5, 2.5, threshold=-20.0)

    # CV in cm/ms, convert to m/s (* 10)
    cv_m_s = cv * 10 if not np.isnan(cv) else 0

    # Expected ~0.4-0.8 m/s
    passed = 0.3 < cv_m_s < 1.0

    print(f"  CV = {cv:.4f} cm/ms = {cv_m_s:.2f} m/s")
    print(f"  Expected range: 0.3-1.0 m/s")

    return print_result(passed, f"CV = {cv_m_s:.2f} m/s")


def test_stimulus_localization():
    """
    Test 5.4: Stimulus is correctly localized.
    """
    print_test_header("5.4", "Stimulus localization")

    from fem import TriangularMesh
    from ionic import TTP06Model, CellType
    from tissue import MonodomainSimulation
    from tissue.simulation import SimulationConfig
    from tissue.stimulus import rectangular_region

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2D domain
    mesh = TriangularMesh.create_rectangle(1.0, 1.0, 51, 51, device=device)

    model = TTP06Model(celltype=CellType.EPI, device=device)
    config = SimulationConfig(D=0.001, dt=0.02, save_interval=0.5)
    sim = MonodomainSimulation(mesh, model, config)

    # Stimulus only in center square (0.4-0.6 x 0.4-0.6)
    sim.add_stimulus(rectangular_region(0.4, 0.4, 0.6, 0.6),
                     start_time=1.0, duration=1.0)

    # Run for just a few ms to see stimulus effect
    times, voltages = sim.run(5.0)

    x = mesh.nodes[:, 0].cpu().numpy()
    y = mesh.nodes[:, 1].cpu().numpy()

    # Check voltage at t=2.5 ms (during/after stimulus)
    t_idx = len(times) // 2
    V = voltages[t_idx]

    # Center should be activated
    center_mask = (x > 0.45) & (x < 0.55) & (y > 0.45) & (y < 0.55)
    V_center = V[center_mask].mean()

    # Corners should not be activated yet
    corner_mask = (x < 0.1) | (x > 0.9) | (y < 0.1) | (y > 0.9)
    V_corner = V[corner_mask].mean()

    # Center should be depolarized, corners still near rest
    passed = V_center > V_corner + 20

    print(f"  Center V (stimulated) = {V_center:.1f} mV")
    print(f"  Corner V (unstimulated) = {V_corner:.1f} mV")

    return print_result(passed, f"Center={V_center:.1f}, Corner={V_corner:.1f}")


def test_cn_vs_bdf2_tissue():
    """
    Test 5.5: CN and BDF2 give similar qualitative results at tissue level.

    With operator splitting and stiff ionic models, we compare:
    - Both produce action potentials (Vpeak > 0)
    - Activation times are similar
    - Both have wave propagation
    """
    print_test_header("5.5", "CN vs BDF2 cross-validation (tissue)")

    from fem import TriangularMesh
    from ionic import TTP06Model, CellType
    from tissue import MonodomainSimulation
    from tissue.simulation import SimulationConfig

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1D cable
    mesh = TriangularMesh.create_rectangle(1.0, 0.1, 51, 6, device=device)

    # Run with CN
    model_cn = TTP06Model(celltype=CellType.EPI, device=device)
    config_cn = SimulationConfig(D=1.5, chi=1400.0, dt=0.01, time_scheme='CN', save_interval=1.0)
    sim_cn = MonodomainSimulation(mesh, model_cn, config_cn)
    sim_cn.add_stimulus(lambda x, y: x < 0.1, start_time=1.0, duration=1.0)
    times_cn, V_cn = sim_cn.run(20.0)

    # Run with BDF2
    model_bdf = TTP06Model(celltype=CellType.EPI, device=device)
    config_bdf = SimulationConfig(D=1.5, chi=1400.0, dt=0.01, time_scheme='BDF2', save_interval=1.0)
    sim_bdf = MonodomainSimulation(mesh, model_bdf, config_bdf)
    sim_bdf.add_stimulus(lambda x, y: x < 0.1, start_time=1.0, duration=1.0)
    times_bdf, V_bdf = sim_bdf.run(20.0)

    # Compare peak voltages (both should produce APs)
    V_peak_cn = V_cn.max()
    V_peak_bdf = V_bdf.max()

    # Check that both activated (peak > 0)
    cn_activated = V_peak_cn > 0
    bdf_activated = V_peak_bdf > 0

    # Compute activation times at stimulated region
    x = mesh.nodes[:, 0].cpu().numpy()
    stim_idx = np.argmin(np.abs(x - 0.05))  # Near left edge

    lat_cn = sim_cn.compute_activation_time(V_cn, times_cn, threshold=-20.0)
    lat_bdf = sim_bdf.compute_activation_time(V_bdf, times_bdf, threshold=-20.0)

    lat_diff = abs(lat_cn[stim_idx] - lat_bdf[stim_idx]) if not np.isnan(lat_cn[stim_idx]) and not np.isnan(lat_bdf[stim_idx]) else 999

    passed = cn_activated and bdf_activated and lat_diff < 2.0  # Within 2 ms

    print(f"  CN: Vpeak = {V_peak_cn:.1f} mV, activated = {cn_activated}")
    print(f"  BDF2: Vpeak = {V_peak_bdf:.1f} mV, activated = {bdf_activated}")
    print(f"  LAT difference at stim site: {lat_diff:.2f} ms")

    return print_result(passed, f"CN Vpeak={V_peak_cn:.1f}, BDF2 Vpeak={V_peak_bdf:.1f}, LAT diff={lat_diff:.2f}ms")


def test_2d_wavefront():
    """
    Test 5.6: 2D wavefront shape (qualitative).

    Point stimulus should create roughly circular wavefront.
    """
    print_test_header("5.6", "2D wavefront shape")

    from fem import TriangularMesh
    from ionic import TTP06Model, CellType
    from tissue import MonodomainSimulation
    from tissue.simulation import SimulationConfig
    from tissue.stimulus import circular_region

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2D domain
    mesh = TriangularMesh.create_rectangle(2.0, 2.0, 81, 81, device=device)

    model = TTP06Model(celltype=CellType.EPI, device=device)
    config = SimulationConfig(D=1.5, chi=1400.0, dt=0.01, save_interval=2.0)
    sim = MonodomainSimulation(mesh, model, config)

    # Point stimulus at center
    sim.add_stimulus(circular_region(1.0, 1.0, 0.1),
                     start_time=1.0, duration=1.0)

    # Run for longer to allow wave to propagate
    times, voltages = sim.run(30.0)

    # Compute activation times
    lat = sim.compute_activation_time(voltages, times, threshold=-20.0)

    x = mesh.nodes[:, 0].cpu().numpy()
    y = mesh.nodes[:, 1].cpu().numpy()

    # Check isotropy: activation time should depend only on distance from center
    r = np.sqrt((x - 1.0)**2 + (y - 1.0)**2)

    # For activated nodes, check correlation between r and LAT
    activated = ~np.isnan(lat)
    if np.sum(activated) > 10:
        r_act = r[activated]
        lat_act = lat[activated]

        # Fit linear: LAT = a * r + b
        # Good isotropy means high R²
        corr = np.corrcoef(r_act, lat_act)[0, 1]
        r_squared = corr ** 2

        passed = r_squared > 0.9  # High correlation indicates isotropic propagation
    else:
        r_squared = 0
        passed = False

    print(f"  Activated nodes: {np.sum(activated)}")
    print(f"  LAT vs distance R² = {r_squared:.3f} (expected > 0.9)")

    return print_result(passed, f"R² = {r_squared:.3f}")


def main():
    print("=" * 70)
    print("Stage 5 Validation: Full Integration (Tissue Simulation)")
    print("=" * 70)

    results = []

    results.append(("5.1", "Single cell TTP06", test_single_cell_ttp06()))
    results.append(("5.2", "Wave propagation", test_wave_propagation()))
    results.append(("5.3", "CV measurement", test_cv_measurement()))
    results.append(("5.4", "Stimulus localization", test_stimulus_localization()))
    results.append(("5.5", "CN vs BDF2 tissue", test_cn_vs_bdf2_tissue()))
    results.append(("5.6", "2D wavefront", test_2d_wavefront()))

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    n_passed = sum(1 for _, _, p in results if p)
    n_total = len(results)

    for test_id, name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {test_id}: {name} - {status}")

    print("-" * 70)
    print(f"Passed: {n_passed}/{n_total}")
    print("=" * 70)

    return 0 if n_passed == n_total else 1


if __name__ == '__main__':
    sys.exit(main())
