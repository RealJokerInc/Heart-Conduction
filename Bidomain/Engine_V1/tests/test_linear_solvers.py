"""
Validation tests for linear solver bug fixes.

Tests:
  LS-T1: Chebyshev preconditioned Gershgorin bounds (CH-1 fix)
  LS-T2: Chebyshev solves a known SPD system correctly
  LS-T3: Chebyshev warm start reduces iterations (CH-4 fix)
  LS-T4: PCG solves same system, cross-check against Chebyshev
  LS-T5: PCG relative pAp threshold (PCG-1 fix)
  LS-T6: All solvers agree on a bidomain-like system

All tests run on CPU with small systems, <10s total on Mac.

Run: ./venv/bin/python Bidomain/Engine_V1/tests/test_linear_solvers.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
torch.set_default_dtype(torch.float64)


def _make_spd_system(n=50):
    """Create a simple SPD system (1D Laplacian + identity)."""
    # Tridiagonal: A = I + alpha * L, where L is 1D Laplacian
    alpha = 0.1
    rows, cols, vals = [], [], []
    for i in range(n):
        rows.append(i); cols.append(i); vals.append(1.0 + 2*alpha)
        if i > 0:
            rows.append(i); cols.append(i-1); vals.append(-alpha)
        if i < n-1:
            rows.append(i); cols.append(i+1); vals.append(-alpha)

    A = torch.sparse_coo_tensor(
        torch.tensor([rows, cols]), torch.tensor(vals), (n, n)).coalesce()

    # Known solution: x = 1, 2, ..., n
    x_true = torch.arange(1, n+1, dtype=torch.float64)
    b = torch.sparse.mm(A, x_true.unsqueeze(1)).squeeze(1)

    return A, b, x_true


def _make_elliptic_system(nx=15, ny=15):
    """Create a bidomain-like elliptic system (no identity term)."""
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec
    from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
    from cardiac_sim.simulation.classical.discretization.fdm import BidomainFDMDiscretization

    dx = 0.025
    Lx = dx * (nx - 1)
    grid = StructuredGrid(Nx=nx, Ny=ny, Lx=Lx, Ly=Lx,
                          boundary_spec=BoundarySpec.insulated())
    cond = BidomainConductivity(D_i=0.00124, D_e=0.00446)
    spatial = BidomainFDMDiscretization(grid, cond, Cm=1.0)

    A_ellip = spatial.get_elliptic_operator()

    # Pin node 0 for null space
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.base import BidomainDiffusionSolver
    A_ellip = BidomainDiffusionSolver.apply_elliptic_pinning(A_ellip, 0)

    # Create a test RHS
    x, y = spatial.coordinates
    Vm_test = torch.sin(3.14159 * x / Lx) * torch.sin(3.14159 * y / Lx)
    b = spatial.apply_L_i(Vm_test)
    b[0] = 0.0  # pinned node

    return A_ellip, b


# ============================================================
# LS-T1: Chebyshev preconditioned Gershgorin bounds
# ============================================================
def test_ls_t1_chebyshev_preconditioned_bounds():
    """Verify preconditioned Gershgorin bounds center around 1."""
    from cardiac_sim.simulation.classical.solver.linear_solver.chebyshev import (
        _gershgorin_bounds, _gershgorin_bounds_preconditioned, ChebyshevSolver)

    A, b, _ = _make_spd_system(50)

    # Unpreconditioned bounds
    lam_min_raw, lam_max_raw = _gershgorin_bounds(A, safety_margin=0.0)

    # Preconditioned bounds (for D^{-1}A)
    cs = ChebyshevSolver()
    diag_inv = cs._extract_diag_inv(A)
    lam_min_pre, lam_max_pre = _gershgorin_bounds_preconditioned(
        A, diag_inv, safety_margin=0.0)

    print(f"    Unpreconditioned bounds: [{lam_min_raw:.4f}, {lam_max_raw:.4f}]")
    print(f"    Preconditioned bounds:   [{lam_min_pre:.4f}, {lam_max_pre:.4f}]")

    # Preconditioned should be tighter, centered around 1
    assert lam_min_pre > 0, "Preconditioned lam_min should be positive"
    assert lam_max_pre < lam_max_raw * 1.1, "Preconditioned should be tighter"
    assert abs(1.0 - (lam_min_pre + lam_max_pre) / 2) < 0.5, \
        "Preconditioned center should be near 1"
    print("LS-T1 PASS: Preconditioned Gershgorin bounds correct")


# ============================================================
# LS-T2: Chebyshev solves a known system
# ============================================================
def test_ls_t2_chebyshev_solve():
    """Verify Chebyshev solver converges to correct answer."""
    from cardiac_sim.simulation.classical.solver.linear_solver.chebyshev import ChebyshevSolver

    A, b, x_true = _make_spd_system(50)

    solver = ChebyshevSolver(max_iters=100, use_jacobi_precond=True)
    x = solver.solve(A, b)

    rel_err = (x - x_true).norm().item() / x_true.norm().item()
    print(f"    Chebyshev relative error: {rel_err:.6e}")
    print(f"    Eigenvalue bounds: [{solver._lam_min:.4f}, {solver._lam_max:.4f}]")

    assert rel_err < 1e-4, f"Chebyshev error too large: {rel_err}"
    print("LS-T2 PASS: Chebyshev solves correctly")


# ============================================================
# LS-T3: Chebyshev warm start
# ============================================================
def test_ls_t3_chebyshev_warm_start():
    """Verify warm start gives better initial residual."""
    from cardiac_sim.simulation.classical.solver.linear_solver.chebyshev import ChebyshevSolver

    A, b, x_true = _make_spd_system(50)

    # Use the harder elliptic system where convergence is slower
    A_e, b_e = _make_elliptic_system(15, 15)

    # Solve once from zero with few iterations
    solver = ChebyshevSolver(max_iters=10, use_jacobi_precond=True)
    x_cold = solver.solve(A_e, b_e)
    res_cold = (torch.sparse.mm(A_e, x_cold.unsqueeze(1)).squeeze(1) - b_e).norm().item()

    # Solve again with warm start from the previous result
    x_warm = solver.solve(A_e, b_e, x0=x_cold)
    res_warm = (torch.sparse.mm(A_e, x_warm.unsqueeze(1)).squeeze(1) - b_e).norm().item()

    print(f"    Cold start residual (10 iters): {res_cold:.6e}")
    print(f"    Warm start residual (10 more):  {res_warm:.6e}")

    assert res_warm < res_cold, "Warm start should improve solution"
    print("LS-T3 PASS: Warm start improves convergence")


# ============================================================
# LS-T4: PCG cross-check
# ============================================================
def test_ls_t4_pcg_cross_check():
    """PCG and Chebyshev should agree on the same system."""
    from cardiac_sim.simulation.classical.solver.linear_solver.chebyshev import ChebyshevSolver
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import PCGSolver

    A, b, x_true = _make_spd_system(50)

    cheby = ChebyshevSolver(max_iters=100, use_jacobi_precond=True)
    pcg = PCGSolver(max_iters=100, tol=1e-10)

    x_cheby = cheby.solve(A, b)
    x_pcg = pcg.solve(A, b)

    err_cheby = (x_cheby - x_true).norm().item() / x_true.norm().item()
    err_pcg = (x_pcg - x_true).norm().item() / x_true.norm().item()

    print(f"    Chebyshev rel error: {err_cheby:.6e}")
    print(f"    PCG rel error:       {err_pcg:.6e}")

    # Both should be accurate
    assert err_pcg < 1e-6, f"PCG error too large: {err_pcg}"
    assert err_cheby < 1e-3, f"Chebyshev error too large: {err_cheby}"
    print("LS-T4 PASS: PCG and Chebyshev both solve correctly")


# ============================================================
# LS-T5: PCG relative pAp threshold
# ============================================================
def test_ls_t5_pcg_relative_threshold():
    """PCG should handle scaled systems correctly (PCG-1 fix)."""
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import PCGSolver

    A, b, x_true = _make_spd_system(50)

    # Solve normal system
    pcg = PCGSolver(max_iters=200, tol=1e-10)
    x_normal = pcg.solve(A, b)
    err_normal = (x_normal - x_true).norm().item() / x_true.norm().item()

    # Scale everything by 1e-6 — should still work with relative threshold
    A_scaled = A * 1e-6
    b_scaled = b * 1e-6
    pcg2 = PCGSolver(max_iters=200, tol=1e-10)
    x_scaled = pcg2.solve(A_scaled, b_scaled)
    err_scaled = (x_scaled - x_true).norm().item() / x_true.norm().item()

    print(f"    Normal system error:   {err_normal:.6e}")
    print(f"    Scaled system error:   {err_scaled:.6e}")

    assert err_scaled < 1e-4, f"Scaled system error too large: {err_scaled}"
    print("LS-T5 PASS: PCG handles scaled systems (relative threshold)")


# ============================================================
# LS-T6: Elliptic system cross-check
# ============================================================
def test_ls_t6_elliptic_cross_check():
    """All solvers agree on a bidomain elliptic system."""
    from cardiac_sim.simulation.classical.solver.linear_solver.chebyshev import ChebyshevSolver
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import PCGSolver

    A_ellip, b = _make_elliptic_system(15, 15)

    pcg = PCGSolver(max_iters=500, tol=1e-10)

    x_pcg = pcg.solve(A_ellip, b)

    # Verify PCG converged
    residual_pcg = (torch.sparse.mm(A_ellip, x_pcg.unsqueeze(1)).squeeze(1) - b).norm()
    b_norm = b.norm()

    # Chebyshev with warm start from PCG (verify it doesn't degrade)
    cheby = ChebyshevSolver(max_iters=50, use_jacobi_precond=True)
    x_cheby = cheby.solve(A_ellip, b, x0=x_pcg)
    residual_cheby = (torch.sparse.mm(A_ellip, x_cheby.unsqueeze(1)).squeeze(1) - b).norm()

    print(f"    PCG residual / b_norm:      {residual_pcg.item() / b_norm.item():.6e}")
    print(f"    Cheby (warm) res / b_norm:  {residual_cheby.item() / b_norm.item():.6e}")

    # PCG should converge well
    assert residual_pcg.item() / b_norm.item() < 1e-5, \
        f"PCG didn't converge: {residual_pcg.item()}"
    # Chebyshev warm start should be at least as good
    assert residual_cheby.item() <= residual_pcg.item() * 1.1, \
        "Chebyshev warm start degraded solution"
    print("LS-T6 PASS: PCG converges on elliptic, Chebyshev warm start preserves")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Linear Solver Bug Fix Validation\n")

    test_ls_t1_chebyshev_preconditioned_bounds()
    print()
    test_ls_t2_chebyshev_solve()
    print()
    test_ls_t3_chebyshev_warm_start()
    print()
    test_ls_t4_pcg_cross_check()
    print()
    test_ls_t5_pcg_relative_threshold()
    print()
    test_ls_t6_elliptic_cross_check()

    print("\nLinear Solvers: ALL 6 TESTS PASS")
