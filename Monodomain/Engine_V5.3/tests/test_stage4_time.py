#!/usr/bin/env python3
"""
Stage 4 Validation: Time Integration

Tests:
4.1: Heat equation decay (analytic Gaussian)
4.2: CN order of accuracy (O(dt²))
4.3: BDF1 order of accuracy (O(dt))
4.4: BDF2 order of accuracy (O(dt²))
4.5: Stability at large dt
4.6: BDF2 startup (uses BDF1 for first step)
4.7: CN vs BDF2 cross-validation
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


def get_gaussian_ic(mesh, sigma2=0.01, x0=0.5, y0=0.5):
    """Get Gaussian initial condition."""
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    return torch.exp(-((x - x0)**2 + (y - y0)**2) / (4 * sigma2))


def get_gaussian_analytic(mesh, sigma2, D, t, x0=0.5, y0=0.5):
    """Get analytic Gaussian solution at time t."""
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    denom = sigma2 + D * t
    return (sigma2 / denom) * torch.exp(-((x - x0)**2 + (y - y0)**2) / (4 * denom))


def test_heat_equation_cn():
    """
    Test 4.1: Heat equation ∂u/∂t = D∇²u with Gaussian IC.

    Tests qualitative behavior: solution should decay and smooth out.
    Note: Exact Gaussian solution is for infinite domain, so we test
    that the solution is decaying as expected with correct physics.
    """
    print_test_header("4.1", "Heat equation decay (CN)")

    from fem import TriangularMesh, assemble_mass_matrix, assemble_stiffness_matrix
    from solver import CrankNicolsonStepper

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup - small diffusion and short time to minimize boundary effects
    D = 0.01
    sigma2 = 0.005
    t_end = 0.1
    dt = 0.001

    # Larger domain to minimize boundary effects
    mesh = TriangularMesh.create_rectangle(2.0, 2.0, 81, 81, device=str(device))
    M = assemble_mass_matrix(mesh, chi=1.0, Cm=1.0)
    K = assemble_stiffness_matrix(mesh, D=D)

    # Initial condition (centered at domain center)
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    u0 = torch.exp(-((x - 1.0)**2 + (y - 1.0)**2) / (4 * sigma2))

    u = u0.clone()

    # Time stepper
    stepper = CrankNicolsonStepper(M, K, theta=0.5)

    # Evolve
    n_steps = int(t_end / dt)
    for _ in range(n_steps):
        u = stepper.step(u, torch.zeros_like(u), dt)

    # Test physics: solution should have decayed (max decreased, spread out)
    max_initial = u0.max().item()
    max_final = u.max().item()

    # Peak should decrease due to diffusion
    decay_ratio = max_final / max_initial

    # Mass should be conserved (integral should be same)
    # For FEM: mass = 1^T M u (lumped approximation)
    mass_initial = (torch.sparse.mm(M, u0.unsqueeze(1))).sum().item()
    mass_final = (torch.sparse.mm(M, u.unsqueeze(1))).sum().item()
    mass_conservation = abs(mass_final - mass_initial) / abs(mass_initial)

    passed = (decay_ratio < 1.0) and (mass_conservation < 0.01)

    print(f"  t_end = {t_end} ms, dt = {dt} ms")
    print(f"  Peak decay: {max_initial:.4f} -> {max_final:.4f} (ratio={decay_ratio:.4f})")
    print(f"  Mass conservation error: {mass_conservation*100:.4f}%")

    return print_result(passed, f"Decay={decay_ratio:.4f}, Mass err={mass_conservation*100:.2f}%")


def test_cn_order():
    """
    Test 4.2: Crank-Nicolson is O(dt²).

    Uses Method of Manufactured Solutions (MMS) to isolate temporal error:
    - Use very fine spatial mesh so spatial error << temporal error
    - Use simple exponential decay ODE: du/dt = -λu, u(0) = 1
    - Exact: u(t) = exp(-λt)

    For pure ODE (no spatial terms), this tests the time stepper directly.
    """
    print_test_header("4.2", "CN order of accuracy (O(dt²))")

    from fem import TriangularMesh, assemble_mass_matrix, assemble_stiffness_matrix
    from solver import CrankNicolsonStepper

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Very fine mesh to minimize spatial error
    mesh = TriangularMesh.create_rectangle(1.0, 1.0, 101, 101, device=str(device))
    M = assemble_mass_matrix(mesh, chi=1.0, Cm=1.0)

    # Use K = λM (uniform decay, no spatial gradients)
    # This gives M·du/dt = -λM·u, or du/dt = -λu
    lam = 1.0
    K = lam * M

    # Initial condition: uniform
    u0 = torch.ones(mesh.n_nodes, dtype=torch.float64, device=device)

    t_end = 1.0

    errors = {}
    dt_values = [0.1, 0.05, 0.025]

    for dt in dt_values:
        u = u0.clone()
        stepper = CrankNicolsonStepper(M, K, theta=0.5)

        n_steps = int(t_end / dt)
        for _ in range(n_steps):
            u = stepper.step(u, torch.zeros_like(u), dt)

        # Exact solution: u(t) = exp(-λt)
        u_exact = np.exp(-lam * t_end) * u0
        errors[dt] = torch.norm(u - u_exact).item()
        print(f"  dt={dt:.3f}: error={errors[dt]:.6e}")

    # Check order: error(dt/2) / error(dt) ≈ 1/4 for O(dt²)
    ratio1 = errors[0.05] / errors[0.1]
    ratio2 = errors[0.025] / errors[0.05]

    # For O(dt²), ratio should be around 0.25
    passed = (0.15 < ratio1 < 0.40) and (0.15 < ratio2 < 0.40)

    print(f"  Error ratio (dt/2)/dt: {ratio1:.3f}, {ratio2:.3f} (expected ~0.25)")

    return print_result(passed, f"Ratios = {ratio1:.3f}, {ratio2:.3f}")


def test_bdf1_order():
    """
    Test 4.3: BDF1 (Backward Euler) is O(dt).

    Uses same MMS approach as CN test.
    """
    print_test_header("4.3", "BDF1 order of accuracy (O(dt))")

    from fem import TriangularMesh, assemble_mass_matrix, assemble_stiffness_matrix
    from solver import BDFStepper

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mesh = TriangularMesh.create_rectangle(1.0, 1.0, 101, 101, device=str(device))
    M = assemble_mass_matrix(mesh, chi=1.0, Cm=1.0)

    lam = 1.0
    K = lam * M

    u0 = torch.ones(mesh.n_nodes, dtype=torch.float64, device=device)
    t_end = 1.0

    errors = {}
    dt_values = [0.1, 0.05, 0.025]

    for dt in dt_values:
        u = u0.clone()
        stepper = BDFStepper(M, K, order=1)

        n_steps = int(t_end / dt)
        for _ in range(n_steps):
            u = stepper.step(u, torch.zeros_like(u), dt)

        u_exact = np.exp(-lam * t_end) * u0
        errors[dt] = torch.norm(u - u_exact).item()
        print(f"  dt={dt:.3f}: error={errors[dt]:.6e}")

    # Check order: error(dt/2) / error(dt) ≈ 1/2 for O(dt)
    ratio1 = errors[0.05] / errors[0.1]
    ratio2 = errors[0.025] / errors[0.05]

    # For O(dt), ratio should be around 0.5
    passed = (0.40 < ratio1 < 0.60) and (0.40 < ratio2 < 0.60)

    print(f"  Error ratio (dt/2)/dt: {ratio1:.3f}, {ratio2:.3f} (expected ~0.5)")

    return print_result(passed, f"Ratios = {ratio1:.3f}, {ratio2:.3f}")


def test_bdf2_order():
    """
    Test 4.4: BDF2 is O(dt²).

    Uses same MMS approach. Note: BDF2 first step uses BDF1, which affects
    the overall order slightly for small number of steps.
    """
    print_test_header("4.4", "BDF2 order of accuracy (O(dt²))")

    from fem import TriangularMesh, assemble_mass_matrix, assemble_stiffness_matrix
    from solver import BDFStepper

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mesh = TriangularMesh.create_rectangle(1.0, 1.0, 101, 101, device=str(device))
    M = assemble_mass_matrix(mesh, chi=1.0, Cm=1.0)

    lam = 1.0
    K = lam * M

    u0 = torch.ones(mesh.n_nodes, dtype=torch.float64, device=device)
    t_end = 2.0  # Longer to amortize first-step BDF1 error

    errors = {}
    dt_values = [0.1, 0.05, 0.025]

    for dt in dt_values:
        u = u0.clone()
        stepper = BDFStepper(M, K, order=2)

        n_steps = int(t_end / dt)
        for _ in range(n_steps):
            u = stepper.step(u, torch.zeros_like(u), dt)

        u_exact = np.exp(-lam * t_end) * u0
        errors[dt] = torch.norm(u - u_exact).item()
        print(f"  dt={dt:.3f}: error={errors[dt]:.6e}")

    # Check order: error(dt/2) / error(dt) ≈ 1/4 for O(dt²)
    ratio1 = errors[0.05] / errors[0.1]
    ratio2 = errors[0.025] / errors[0.05]

    # For O(dt²), ratio should be around 0.25
    # Allow wider tolerance due to BDF1 startup
    passed = (0.15 < ratio1 < 0.45) and (0.15 < ratio2 < 0.45)

    print(f"  Error ratio (dt/2)/dt: {ratio1:.3f}, {ratio2:.3f} (expected ~0.25)")

    return print_result(passed, f"Ratios = {ratio1:.3f}, {ratio2:.3f}")


def test_stability_large_dt():
    """
    Test 4.5: Implicit schemes are stable at large dt.
    """
    print_test_header("4.5", "Stability at large dt")

    from fem import TriangularMesh, assemble_mass_matrix, assemble_stiffness_matrix
    from solver import CrankNicolsonStepper, BDFStepper

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    D = 0.1
    sigma2 = 0.01

    mesh = TriangularMesh.create_rectangle(1.0, 1.0, 21, 21, device=str(device))
    M = assemble_mass_matrix(mesh, chi=1.0, Cm=1.0)
    K = assemble_stiffness_matrix(mesh, D=D)

    u0 = get_gaussian_ic(mesh, sigma2)

    # Large dt = 0.5 ms (would be unstable for explicit methods)
    dt = 0.5
    n_steps = 10

    results = {}

    for name, stepper in [
        ('CN', CrankNicolsonStepper(M, K, theta=0.5)),
        ('BDF1', BDFStepper(M, K, order=1)),
        ('BDF2', BDFStepper(M, K, order=2)),
    ]:
        u = u0.clone()
        for _ in range(n_steps):
            u = stepper.step(u, torch.zeros_like(u), dt)

        # Check for blow-up
        is_stable = torch.isfinite(u).all().item()
        max_val = u.abs().max().item()
        results[name] = (is_stable, max_val)

        status = "stable" if is_stable else "UNSTABLE"
        print(f"  {name}: {status}, max|u| = {max_val:.4f}")

    all_stable = all(r[0] for r in results.values())

    return print_result(all_stable)


def test_bdf2_startup():
    """
    Test 4.6: BDF2 uses BDF1 for first step when history unavailable.
    """
    print_test_header("4.6", "BDF2 startup (BDF1 fallback)")

    from fem import TriangularMesh, assemble_mass_matrix, assemble_stiffness_matrix
    from solver import BDFStepper

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    D = 0.1
    sigma2 = 0.01

    mesh = TriangularMesh.create_rectangle(1.0, 1.0, 21, 21, device=str(device))
    M = assemble_mass_matrix(mesh, chi=1.0, Cm=1.0)
    K = assemble_stiffness_matrix(mesh, D=D)

    u0 = get_gaussian_ic(mesh, sigma2)
    dt = 0.05

    # Fresh BDF2 stepper
    stepper = BDFStepper(M, K, order=2)

    # First step: should use BDF1 (history empty)
    u1 = stepper.step(u0.clone(), torch.zeros_like(u0), dt)

    # Pure BDF1 for comparison
    stepper_bdf1 = BDFStepper(M, K, order=1)
    u1_bdf1 = stepper_bdf1.step(u0.clone(), torch.zeros_like(u0), dt)

    # First step of BDF2 should be identical to BDF1
    diff = torch.norm(u1 - u1_bdf1) / torch.norm(u1_bdf1)

    passed = diff < 1e-10

    print(f"  BDF2 first step vs BDF1: diff = {diff:.2e}")
    print(f"  History length after 1 step: {len(stepper.history)}")

    # Second step should use full BDF2
    u2 = stepper.step(u1, torch.zeros_like(u1), dt)
    print(f"  History length after 2 steps: {len(stepper.history)}")

    return print_result(passed, f"First step diff = {diff:.2e}")


def test_cn_vs_bdf2_crossval():
    """
    Test 4.7: CN and BDF2 give similar results (cross-validation).

    Both are O(dt²) schemes, so they should converge to the same solution
    as dt → 0. We test with a small dt and compare.
    """
    print_test_header("4.7", "CN vs BDF2 cross-validation")

    from fem import TriangularMesh, assemble_mass_matrix, assemble_stiffness_matrix
    from solver import CrankNicolsonStepper, BDFStepper

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mesh = TriangularMesh.create_rectangle(1.0, 1.0, 51, 51, device=str(device))
    M = assemble_mass_matrix(mesh, chi=1.0, Cm=1.0)

    # Use the MMS test case
    lam = 1.0
    K = lam * M

    u0 = torch.ones(mesh.n_nodes, dtype=torch.float64, device=device)
    t_end = 1.0
    dt = 0.01  # Small dt for accurate comparison

    n_steps = int(t_end / dt)

    # Run CN
    u_cn = u0.clone()
    stepper_cn = CrankNicolsonStepper(M, K, theta=0.5)
    for _ in range(n_steps):
        u_cn = stepper_cn.step(u_cn, torch.zeros_like(u_cn), dt)

    # Run BDF2
    u_bdf2 = u0.clone()
    stepper_bdf2 = BDFStepper(M, K, order=2)
    for _ in range(n_steps):
        u_bdf2 = stepper_bdf2.step(u_bdf2, torch.zeros_like(u_bdf2), dt)

    # Compare
    max_diff = torch.max(torch.abs(u_cn - u_bdf2)).item()
    rel_diff = max_diff / torch.max(torch.abs(u_cn)).item()

    # Both are O(dt²), difference should be O(dt²) as well
    passed = rel_diff < 0.01  # 1% tolerance

    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Relative difference: {rel_diff*100:.4f}%")

    return print_result(passed, f"Relative diff = {rel_diff*100:.4f}%")


def test_source_term():
    """
    Test 4.8: Time stepper handles source terms correctly.
    """
    print_test_header("4.8", "Source term handling")

    from fem import TriangularMesh, assemble_mass_matrix, assemble_stiffness_matrix
    from solver import CrankNicolsonStepper

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mesh = TriangularMesh.create_rectangle(1.0, 1.0, 21, 21, device=str(device))
    M = assemble_mass_matrix(mesh, chi=1.0, Cm=1.0)
    K = assemble_stiffness_matrix(mesh, D=0.1)

    # Start from zero
    u = torch.zeros(mesh.n_nodes, dtype=torch.float64, device=device)
    dt = 0.1

    stepper = CrankNicolsonStepper(M, K, theta=0.5)

    # Apply constant source
    f = torch.ones(mesh.n_nodes, dtype=torch.float64, device=device)

    # Evolve with source
    for _ in range(10):
        u = stepper.step(u, f, dt)

    # With source, solution should grow (not stay at zero)
    max_val = u.abs().max().item()
    passed = max_val > 0.1

    print(f"  After 10 steps with f=1: max|u| = {max_val:.4f}")

    return print_result(passed, f"max|u| = {max_val:.4f}")


def main():
    print("=" * 70)
    print("Stage 4 Validation: Time Integration")
    print("=" * 70)

    results = []

    results.append(("4.1", "Heat equation (CN)", test_heat_equation_cn()))
    results.append(("4.2", "CN order O(dt²)", test_cn_order()))
    results.append(("4.3", "BDF1 order O(dt)", test_bdf1_order()))
    results.append(("4.4", "BDF2 order O(dt²)", test_bdf2_order()))
    results.append(("4.5", "Stability large dt", test_stability_large_dt()))
    results.append(("4.6", "BDF2 startup", test_bdf2_startup()))
    results.append(("4.7", "CN vs BDF2", test_cn_vs_bdf2_crossval()))
    results.append(("4.8", "Source term", test_source_term()))

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
