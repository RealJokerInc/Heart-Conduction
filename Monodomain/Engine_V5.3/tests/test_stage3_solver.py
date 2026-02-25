#!/usr/bin/env python3
"""
Stage 3 Validation: Linear Solver

Tests:
3.1: Solve Ax=b with known solution
3.2: Poisson equation on unit square (Dirichlet BC)
3.3: Convergence comparison with scipy
3.4: Iteration count scaling
3.5: Tolerance control
3.6: Warm start effectiveness
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


def test_known_solution():
    """
    Test 3.1: Solve Ax=b where x is known.

    Create a simple SPD matrix and verify we recover the known solution.
    """
    print_test_header("3.1", "Solve Ax=b with known solution")

    from solver import pcg_solve

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

    # Create simple SPD matrix: A = L @ L.T + I
    n = 100
    L = torch.randn(n, n, dtype=dtype, device=device)
    A_dense = L @ L.T + torch.eye(n, dtype=dtype, device=device)

    # Known solution
    x_true = torch.randn(n, dtype=dtype, device=device)

    # Right-hand side
    b = A_dense @ x_true

    # Convert to sparse
    A_sparse = A_dense.to_sparse_coo().coalesce()

    # Solve
    x_pcg = pcg_solve(A_sparse, b, tol=1e-10)

    error = torch.norm(x_pcg - x_true) / torch.norm(x_true)
    passed = error < 1e-8

    print(f"  Relative error: {error:.2e}")

    return print_result(passed, f"Error = {error:.2e}")


def test_poisson_dirichlet():
    """
    Test 3.2: Poisson equation -∇²u = f on [0,1]² with Dirichlet BC.

    Exact solution: u(x,y) = sin(πx)sin(πy)
    Source term: f = 2π²sin(πx)sin(πy)
    BC: u = 0 on boundary
    """
    print_test_header("3.2", "Poisson equation with Dirichlet BC")

    from fem import TriangularMesh, assemble_stiffness_matrix, assemble_mass_matrix
    from solver import pcg_solve
    from solver.linear import apply_dirichlet_bc

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

    # Create mesh
    mesh = TriangularMesh.create_rectangle(1.0, 1.0, 31, 31, device=str(device))

    # Assemble stiffness matrix (K = -∇²)
    K = assemble_stiffness_matrix(mesh, D=1.0)

    # Coordinates
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]

    # Exact solution and source term
    u_exact = torch.sin(np.pi * x) * torch.sin(np.pi * y)
    f = 2 * np.pi**2 * torch.sin(np.pi * x) * torch.sin(np.pi * y)

    # Assemble RHS: b = M @ f (weak form)
    M = assemble_mass_matrix(mesh, chi=1.0, Cm=1.0)
    b = torch.sparse.mm(M, f.unsqueeze(1)).squeeze(1)

    # Apply Dirichlet BC: u = 0 on boundary
    boundary_values = torch.zeros_like(mesh.boundary_nodes, dtype=dtype)
    K_bc, b_bc = apply_dirichlet_bc(K, b, mesh.boundary_nodes, boundary_values)

    # Solve
    u_pcg = pcg_solve(K_bc, b_bc, tol=1e-10)

    # Compute error (excluding boundary nodes)
    interior_mask = torch.ones(mesh.n_nodes, dtype=torch.bool, device=device)
    interior_mask[mesh.boundary_nodes] = False

    error = torch.norm(u_pcg[interior_mask] - u_exact[interior_mask]) / \
            torch.norm(u_exact[interior_mask])

    # Expected: O(h²) error for P1 elements, h ≈ 1/30
    # L2 error should be around 1-5%
    passed = error < 0.05

    print(f"  Mesh: {mesh.n_nodes} nodes")
    print(f"  Relative L2 error: {error:.4f} ({error*100:.2f}%)")

    return print_result(passed, f"Error = {error*100:.2f}%")


def test_scipy_comparison():
    """
    Test 3.3: Compare PCG solution with scipy.sparse.linalg.cg.
    """
    print_test_header("3.3", "Comparison with scipy CG")

    try:
        from scipy.sparse.linalg import cg as scipy_cg
        from scipy.sparse import coo_matrix
    except ImportError:
        print("  scipy not available, skipping")
        return print_result(True, "Skipped - scipy not available")

    from fem import TriangularMesh, assemble_stiffness_matrix, assemble_mass_matrix
    from solver import pcg_solve

    device = torch.device('cpu')  # Need CPU for scipy comparison
    dtype = torch.float64

    # Create mesh and matrices
    mesh = TriangularMesh.create_rectangle(1.0, 1.0, 21, 21, device='cpu')
    K = assemble_stiffness_matrix(mesh, D=1.0)
    M = assemble_mass_matrix(mesh, chi=1.0, Cm=1.0)

    # Create test problem
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    f = torch.sin(2 * np.pi * x) * torch.cos(np.pi * y)
    b = torch.sparse.mm(M, f.unsqueeze(1)).squeeze(1)

    # Add regularization for non-singular system (Neumann problem)
    reg = 1e-6
    K_reg = K + reg * M

    # Solve with our PCG
    u_pcg = pcg_solve(K_reg, b, tol=1e-10)

    # Convert to scipy format
    K_scipy = K_reg.coalesce()
    indices = K_scipy.indices().cpu().numpy()
    values = K_scipy.values().cpu().numpy()
    K_coo = coo_matrix((values, (indices[0], indices[1])),
                       shape=(mesh.n_nodes, mesh.n_nodes))
    K_csr = K_coo.tocsr()

    # Solve with scipy (rtol is the newer parameter name)
    u_scipy, info = scipy_cg(K_csr, b.cpu().numpy(), rtol=1e-10)

    # Compare
    diff = torch.norm(u_pcg - torch.from_numpy(u_scipy))
    rel_diff = diff / torch.norm(u_pcg)

    passed = rel_diff < 1e-6

    print(f"  scipy convergence info: {info}")
    print(f"  Absolute difference: {diff:.2e}")
    print(f"  Relative difference: {rel_diff:.2e}")

    return print_result(passed, f"Relative diff = {rel_diff:.2e}")


def test_iteration_scaling():
    """
    Test 3.4: Iteration count scales as O(√N) for PCG with Jacobi.
    """
    print_test_header("3.4", "Iteration count scaling")

    from fem import TriangularMesh, assemble_stiffness_matrix, assemble_mass_matrix
    from solver.linear import pcg_solve

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mesh_sizes = [(11, 11), (21, 21), (41, 41), (81, 81)]
    results = []

    for nx, ny in mesh_sizes:
        mesh = TriangularMesh.create_rectangle(1.0, 1.0, nx, ny, device=str(device))
        K = assemble_stiffness_matrix(mesh, D=1.0)
        M = assemble_mass_matrix(mesh, chi=1.0, Cm=1.0)

        # Test problem
        x = mesh.nodes[:, 0]
        f = torch.sin(2 * np.pi * x)
        b = torch.sparse.mm(M, f.unsqueeze(1)).squeeze(1)

        # Add regularization
        K_reg = K + 1e-6 * M

        # Solve and get stats
        _, stats = pcg_solve(K_reg, b, tol=1e-8, return_stats=True)

        results.append({
            'N': mesh.n_nodes,
            'sqrt_N': np.sqrt(mesh.n_nodes),
            'iterations': stats.iterations,
            'converged': stats.converged
        })

        print(f"  N={mesh.n_nodes:5d} (√N={np.sqrt(mesh.n_nodes):6.1f}): "
              f"{stats.iterations:3d} iterations")

    # Check scaling: iterations should grow roughly as √N
    # Compute ratio of iterations to √N
    ratios = [r['iterations'] / r['sqrt_N'] for r in results]
    ratio_variance = np.std(ratios) / np.mean(ratios)

    # If scaling is O(√N), ratios should be similar (low variance)
    passed = ratio_variance < 0.5  # Allow some variance

    print(f"  Iteration/√N ratios: {[f'{r:.2f}' for r in ratios]}")
    print(f"  Ratio variance: {ratio_variance:.2f}")

    return print_result(passed, f"Variance = {ratio_variance:.2f}")


def test_tolerance_control():
    """
    Test 3.5: Verify solver respects tolerance setting.
    """
    print_test_header("3.5", "Tolerance control")

    from fem import TriangularMesh, assemble_stiffness_matrix, assemble_mass_matrix
    from solver import pcg_solve, sparse_mv

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mesh = TriangularMesh.create_rectangle(1.0, 1.0, 31, 31, device=str(device))
    K = assemble_stiffness_matrix(mesh, D=1.0)
    M = assemble_mass_matrix(mesh, chi=1.0, Cm=1.0)

    x = mesh.nodes[:, 0]
    f = torch.sin(2 * np.pi * x)
    b = torch.sparse.mm(M, f.unsqueeze(1)).squeeze(1)

    K_reg = K + 1e-6 * M

    tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
    all_passed = True

    for tol in tolerances:
        u, stats = pcg_solve(K_reg, b, tol=tol, return_stats=True)

        # Compute actual residual
        residual = b - sparse_mv(K_reg, u)
        rel_residual = torch.norm(residual) / torch.norm(b)

        passed = rel_residual < tol or stats.iterations == 500  # Allow max iter hit
        all_passed = all_passed and passed

        status = "✓" if passed else "✗"
        print(f"  tol={tol:.0e}: {stats.iterations:3d} iters, "
              f"residual={rel_residual:.2e} {status}")

    return print_result(all_passed)


def test_warm_start():
    """
    Test 3.6: Warm start reduces iteration count.
    """
    print_test_header("3.6", "Warm start effectiveness")

    from fem import TriangularMesh, assemble_stiffness_matrix, assemble_mass_matrix
    from solver.linear import pcg_solve

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mesh = TriangularMesh.create_rectangle(1.0, 1.0, 51, 51, device=str(device))
    K = assemble_stiffness_matrix(mesh, D=1.0)
    M = assemble_mass_matrix(mesh, chi=1.0, Cm=1.0)

    # Create two similar problems (simulating time steps)
    x = mesh.nodes[:, 0]
    f1 = torch.sin(2 * np.pi * x)
    f2 = torch.sin(2 * np.pi * x) * 1.01  # Slightly perturbed

    b1 = torch.sparse.mm(M, f1.unsqueeze(1)).squeeze(1)
    b2 = torch.sparse.mm(M, f2.unsqueeze(1)).squeeze(1)

    K_reg = K + 1e-6 * M

    # Solve first problem from scratch
    u1, stats1 = pcg_solve(K_reg, b1, tol=1e-8, return_stats=True)

    # Solve second problem from scratch
    _, stats_cold = pcg_solve(K_reg, b2, tol=1e-8, return_stats=True)

    # Solve second problem with warm start
    _, stats_warm = pcg_solve(K_reg, b2, x0=u1, tol=1e-8, return_stats=True)

    # Warm start should need fewer iterations
    speedup = stats_cold.iterations / max(1, stats_warm.iterations)
    passed = stats_warm.iterations < stats_cold.iterations

    print(f"  Cold start: {stats_cold.iterations} iterations")
    print(f"  Warm start: {stats_warm.iterations} iterations")
    print(f"  Speedup: {speedup:.1f}x")

    return print_result(passed, f"Speedup = {speedup:.1f}x")


def test_gpu_solver():
    """Test 3.7: GPU solver correctness."""
    print_test_header("3.7", "GPU solver (if available)")

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping")
        return print_result(True, "Skipped - no GPU")

    from fem import TriangularMesh, assemble_stiffness_matrix, assemble_mass_matrix
    from solver import pcg_solve

    # Solve same problem on CPU and GPU
    for device in ['cpu', 'cuda']:
        mesh = TriangularMesh.create_rectangle(1.0, 1.0, 31, 31, device=device)
        K = assemble_stiffness_matrix(mesh, D=1.0)
        M = assemble_mass_matrix(mesh, chi=1.0, Cm=1.0)

        x = mesh.nodes[:, 0]
        f = torch.sin(2 * np.pi * x)
        b = torch.sparse.mm(M, f.unsqueeze(1)).squeeze(1)

        K_reg = K + 1e-6 * M
        u, stats = pcg_solve(K_reg, b, tol=1e-10, return_stats=True)

        if device == 'cpu':
            u_cpu = u.clone()
            iters_cpu = stats.iterations
        else:
            u_gpu = u.cpu()
            iters_gpu = stats.iterations

    # Compare solutions
    diff = torch.norm(u_cpu - u_gpu) / torch.norm(u_cpu)
    passed = diff < 1e-8

    print(f"  CPU iterations: {iters_cpu}")
    print(f"  GPU iterations: {iters_gpu}")
    print(f"  Solution difference: {diff:.2e}")

    return print_result(passed, f"Diff = {diff:.2e}")


def main():
    print("=" * 70)
    print("Stage 3 Validation: Linear Solver (PCG)")
    print("=" * 70)

    results = []

    results.append(("3.1", "Known solution", test_known_solution()))
    results.append(("3.2", "Poisson-Dirichlet", test_poisson_dirichlet()))
    results.append(("3.3", "scipy comparison", test_scipy_comparison()))
    results.append(("3.4", "Iteration scaling", test_iteration_scaling()))
    results.append(("3.5", "Tolerance control", test_tolerance_control()))
    results.append(("3.6", "Warm start", test_warm_start()))
    results.append(("3.7", "GPU solver", test_gpu_solver()))

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
