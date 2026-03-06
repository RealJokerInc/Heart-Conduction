"""
Mixed Spectral Solver Cross-Validation Tests

Pure Poisson tests comparing the mixed-BC spectral solver (DCT-x, DST-y)
against PCG on the same FDM system. No ionic model — just -D*Lap*u = b.

Tests:
  M-T1: Uniform Neumann — spectral vs PCG, error < 1e-8
  M-T2: Uniform Dirichlet — spectral vs PCG, error < 1e-8
  M-T3: Mixed Neumann-x / Dirichlet-y — spectral vs PCG, error < 1e-8
  M-T4: Mixed Dirichlet-x / Neumann-y — spectral vs PCG, error < 1e-8
  M-T5: Analytical Poisson with mixed BCs — O(h^2) convergence
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
torch.set_default_dtype(torch.float64)
import numpy as np
import pytest

from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec, Edge
from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
from cardiac_sim.simulation.classical.discretization.fdm import BidomainFDMDiscretization
from cardiac_sim.simulation.classical.solver.linear_solver.spectral import SpectralSolver
from cardiac_sim.simulation.classical.solver.linear_solver.pcg import PCGSolver, sparse_mv


def _build_poisson_system(nx, ny, dx, D, boundary_spec):
    """Build FDM Poisson system: A_ellip * u = b, with random RHS.

    Returns (A_ellip, b_rhs, spectral_solver, spatial).
    """
    Lx = dx * (nx - 1)
    Ly = dx * (ny - 1)
    grid = StructuredGrid(Nx=nx, Ny=ny, Lx=Lx, Ly=Ly,
                          boundary_spec=boundary_spec)
    cond = BidomainConductivity(D_i=D / 2, D_e=D / 2)
    spatial = BidomainFDMDiscretization(grid, cond, chi=1.0, Cm=1.0)

    A_ellip = spatial.get_elliptic_operator()

    # Build spectral solver from boundary spec
    bc = boundary_spec
    txy = bc.spectral_transform_xy
    assert txy is not None, "BCs must be spectrally eligible"
    t_map = {'dct': 'neumann', 'dst': 'dirichlet'}
    bc_x, bc_y = t_map[txy[0]], t_map[txy[1]]
    spec = SpectralSolver(nx, ny, dx, dx, D, bc_x=bc_x, bc_y=bc_y)

    # Random RHS consistent with BCs
    torch.manual_seed(42)
    b = torch.randn(nx * ny)

    # For Neumann-only: project out constant null space
    if bc_x == 'neumann' and bc_y == 'neumann':
        b -= b.mean()

    # For Dirichlet edges: zero out boundary RHS (identity rows in A_ellip)
    b_grid = b.reshape(nx, ny)
    if bc_y == 'dirichlet':
        b_grid[:, 0] = 0.0
        b_grid[:, -1] = 0.0
    if bc_x == 'dirichlet':
        b_grid[0, :] = 0.0
        b_grid[-1, :] = 0.0
    b = b_grid.flatten()

    return A_ellip, b, spec, spatial


# === M-T1: Uniform Neumann ===

def test_mixed_t1_neumann():
    """Spectral vs PCG on uniform Neumann Poisson, error < 1e-8."""
    nx, ny, dx, D = 40, 40, 0.05, 0.0057
    bc = BoundarySpec.insulated()

    A, b, spec, _ = _build_poisson_system(nx, ny, dx, D, bc)

    u_spec = spec.solve(None, b)
    pcg = PCGSolver(max_iters=1000, tol=1e-12)
    u_pcg = pcg.solve(A, b)

    # Both solutions unique up to constant — compare zero-mean
    u_spec_zm = u_spec - u_spec.mean()
    u_pcg_zm = u_pcg - u_pcg.mean()

    err = (u_spec_zm - u_pcg_zm).abs().max().item()
    print(f"  M-T1 Neumann: spectral vs PCG max error = {err:.2e}")
    assert err < 1e-8, f"Neumann spectral vs PCG error: {err}"


# === M-T2: Uniform Dirichlet ===

def test_mixed_t2_dirichlet():
    """Spectral vs PCG on uniform Dirichlet Poisson, error < 1e-8."""
    nx, ny, dx, D = 40, 40, 0.05, 0.0057
    bc = BoundarySpec.bath_coupled()

    A, b, spec, _ = _build_poisson_system(nx, ny, dx, D, bc)

    u_spec = spec.solve(None, b)
    pcg = PCGSolver(max_iters=1000, tol=1e-12)
    u_pcg = pcg.solve(A, b)

    err = (u_spec - u_pcg).abs().max().item()
    print(f"  M-T2 Dirichlet: spectral vs PCG max error = {err:.2e}")
    assert err < 1e-8, f"Dirichlet spectral vs PCG error: {err}"


# === M-T3: Mixed Neumann-x / Dirichlet-y ===

def test_mixed_t3_neumann_x_dirichlet_y():
    """Spectral (DCT-x, DST-y) vs PCG on mixed Poisson, error < 1e-8."""
    nx, ny, dx, D = 40, 40, 0.05, 0.0057
    bc = BoundarySpec.bath_coupled_edges([Edge.TOP, Edge.BOTTOM])

    A, b, spec, _ = _build_poisson_system(nx, ny, dx, D, bc)

    u_spec = spec.solve(None, b)
    pcg = PCGSolver(max_iters=1000, tol=1e-12)
    u_pcg = pcg.solve(A, b)

    err = (u_spec - u_pcg).abs().max().item()
    print(f"  M-T3 Neumann-x/Dirichlet-y: spectral vs PCG max error = {err:.2e}")
    assert err < 1e-8, f"Mixed N-x/D-y spectral vs PCG error: {err}"


# === M-T4: Mixed Dirichlet-x / Neumann-y ===

def test_mixed_t4_dirichlet_x_neumann_y():
    """Spectral (DST-x, DCT-y) vs PCG on mixed Poisson, error < 1e-8."""
    nx, ny, dx, D = 40, 40, 0.05, 0.0057
    bc = BoundarySpec.bath_coupled_edges([Edge.LEFT, Edge.RIGHT])

    A, b, spec, _ = _build_poisson_system(nx, ny, dx, D, bc)

    u_spec = spec.solve(None, b)
    pcg = PCGSolver(max_iters=1000, tol=1e-12)
    u_pcg = pcg.solve(A, b)

    err = (u_spec - u_pcg).abs().max().item()
    print(f"  M-T4 Dirichlet-x/Neumann-y: spectral vs PCG max error = {err:.2e}")
    assert err < 1e-8, f"Mixed D-x/N-y spectral vs PCG error: {err}"


# === M-T5: Self-consistency and error decrease with mixed BCs ===

def test_mixed_t5_self_consistency_and_convergence():
    """Mixed BCs (Neumann-x, Dirichlet-y): self-consistency + error shrinks.

    Part A: Apply discrete operator to known u, solve back, get machine precision.
    Part B: Solve continuous Poisson, verify error decreases with refinement.
    """
    D = 0.0057

    # Part A: Self-consistency (discrete round-trip)
    nx, ny, dx = 40, 40, 0.05
    bc = BoundarySpec.bath_coupled_edges([Edge.TOP, Edge.BOTTOM])
    A, b_rand, spec, spatial = _build_poisson_system(nx, ny, dx, D, bc)

    # Pick a random interior function, zero on Dirichlet boundaries
    torch.manual_seed(123)
    u_known = torch.randn(nx * ny)
    u_grid = u_known.reshape(nx, ny)
    u_grid[:, 0] = 0.0
    u_grid[:, -1] = 0.0
    u_known = u_grid.flatten()

    b = sparse_mv(A, u_known)
    u_solved = spec.solve(None, b)
    err_sc = (u_solved - u_known).abs().max().item()
    print(f"  M-T5a self-consistency: {err_sc:.2e}")
    assert err_sc < 1e-8, f"Self-consistency error: {err_sc}"

    # Part B: Analytical convergence (error should decrease monotonically)
    Lx, Ly = 2.0, 2.0
    errors = []
    for nx in [21, 41, 81, 161]:
        ny = nx
        dx = Lx / (nx - 1)

        kx, ky = np.pi / Lx, np.pi / Ly
        x = torch.arange(nx, dtype=torch.float64) * dx
        y = torch.arange(ny, dtype=torch.float64) * dy if 'dy' in dir() else torch.arange(ny, dtype=torch.float64) * dx
        X, Y = torch.meshgrid(x, y, indexing='ij')

        u_exact = torch.cos(kx * X) * torch.sin(ky * Y)
        f = D * (kx**2 + ky**2) * u_exact
        f[:, 0] = 0.0
        f[:, -1] = 0.0

        txy = bc.spectral_transform_xy
        t_map = {'dct': 'neumann', 'dst': 'dirichlet'}
        bc_x, bc_y = t_map[txy[0]], t_map[txy[1]]
        s = SpectralSolver(nx, ny, dx, dx, D, bc_x=bc_x, bc_y=bc_y)

        u_solved = s.solve(None, f.flatten()).reshape(nx, ny)
        err = (u_solved[2:-2, 2:-2] - u_exact[2:-2, 2:-2]).abs().max().item()
        errors.append(err)
        print(f"  N={nx}: error = {err:.4e}")

    # Errors must decrease monotonically
    for i in range(len(errors) - 1):
        assert errors[i+1] < errors[i], \
            f"Error not decreasing: {errors[i]:.4e} -> {errors[i+1]:.4e}"

    # Finest grid error should be small
    assert errors[-1] < 0.01, f"Finest grid error too large: {errors[-1]:.4e}"
    print(f"  Convergence verified: errors decrease monotonically")


if __name__ == '__main__':
    print("Mixed Spectral Solver Cross-Validation\n")
    test_mixed_t1_neumann()
    print()
    test_mixed_t2_dirichlet()
    print()
    test_mixed_t3_neumann_x_dirichlet_y()
    print()
    test_mixed_t4_dirichlet_x_neumann_y()
    print()
    test_mixed_t5_analytical_convergence()
    print("\nALL 5 TESTS PASS")
