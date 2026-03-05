"""
Phase 2 Validation Tests — FDM Discretization

Tests 2-T1 through 2-T6 from PROGRESS.md.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
import numpy as np


def _make_fdm(nx=10, ny=10, lx=1.0, ly=1.0, D_i=0.00124, D_e=0.00446, bc='insulated'):
    """Helper: create BidomainFDMDiscretization with given params."""
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec
    from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
    from cardiac_sim.simulation.classical.discretization.fdm import BidomainFDMDiscretization

    grid = StructuredGrid.create_rectangle(Lx=lx, Ly=ly, Nx=nx, Ny=ny)
    if bc == 'insulated':
        grid.boundary_spec = BoundarySpec.insulated()
    elif bc == 'bath':
        grid.boundary_spec = BoundarySpec.bath_coupled()
    cond = BidomainConductivity(D_i=D_i, D_e=D_e)
    return BidomainFDMDiscretization(grid, cond, chi=1400.0, Cm=1.0)


def _to_dense(A):
    """Convert sparse tensor to dense."""
    return A.to_dense() if A.is_sparse else A


# === 2-T1: L_i symmetry ===

def test_L_i_symmetry():
    """2-T1: L_i == L_i^T for isotropic D_i."""
    fdm = _make_fdm(nx=15, ny=15)
    L_i = _to_dense(fdm.L_i)
    diff = torch.abs(L_i - L_i.T).max().item()
    assert diff < 1e-14, f"L_i asymmetry: {diff}"


# === 2-T2: L_e symmetry ===

def test_L_e_symmetry_neumann():
    """2-T2a: L_e == L_e^T for Neumann BCs (insulated)."""
    fdm = _make_fdm(nx=15, ny=15, bc='insulated')
    L_e = _to_dense(fdm.L_e)
    diff = torch.abs(L_e - L_e.T).max().item()
    assert diff < 1e-14, f"L_e asymmetry (Neumann): {diff}"


def test_L_e_symmetry_dirichlet():
    """2-T2b: L_e is symmetric even with bath-coupled BCs.
    Dirichlet enforcement is only in get_elliptic_operator(), not in L_e itself."""
    fdm = _make_fdm(nx=10, ny=10, bc='bath')
    L_e = _to_dense(fdm.L_e)
    diff = torch.abs(L_e - L_e.T).max().item()
    assert diff < 1e-14, f"L_e asymmetry (Dirichlet grid): {diff}"


# === 2-T3: Neumann stencil convergence ===

def test_neumann_stencil_convergence():
    """2-T3: apply_L_i to cos(pi*x/L) on Neumann grid, check O(h^2) convergence."""
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec
    from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
    from cardiac_sim.simulation.classical.discretization.fdm import BidomainFDMDiscretization

    D_i = 0.00124
    Lx, Ly = 1.0, 1.0
    errors = []

    for N in [20, 40, 80]:
        grid = StructuredGrid.create_rectangle(Lx=Lx, Ly=Ly, Nx=N, Ny=N)
        grid.boundary_spec = BoundarySpec.insulated()
        cond = BidomainConductivity(D_i=D_i, D_e=0.00446)
        fdm = BidomainFDMDiscretization(grid, cond)

        # u(x,y) = cos(pi*x/L)*cos(pi*y/L) satisfies Neumann BCs
        x, y = fdm.coordinates
        kx = np.pi / Lx
        ky = np.pi / Ly
        u = torch.cos(kx * x) * torch.cos(ky * y)

        # Exact: L_i * u = D_i * (-(kx^2 + ky^2)) * u
        exact = D_i * (-(kx**2 + ky**2)) * u

        # Numerical
        Lu = fdm.apply_L_i(u)

        # Check interior nodes only — boundary nodes use face-based stiffness
        # form (no ghost-node doubling) so they give half-weight at edges.
        boundary_mask = grid.boundary_mask.flatten()
        interior = ~boundary_mask
        err = torch.abs(Lu[interior] - exact[interior]).max().item()
        errors.append(err)

    # Check O(h^2) convergence: error ratio ~ 4 when h halves
    ratio1 = errors[0] / errors[1]
    ratio2 = errors[1] / errors[2]

    assert ratio1 > 3.0, f"L_i Neumann convergence ratio 1: {ratio1:.2f} (expected ~4)"
    assert ratio2 > 3.0, f"L_i Neumann convergence ratio 2: {ratio2:.2f} (expected ~4)"


# === 2-T4: Dirichlet stencil convergence ===

def test_dirichlet_stencil_convergence():
    """2-T4: apply_L_e to sin(pi*x/L) on Dirichlet grid (bath BC), check O(h^2)."""
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec
    from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
    from cardiac_sim.simulation.classical.discretization.fdm import BidomainFDMDiscretization

    D_e = 0.00446
    Lx, Ly = 1.0, 1.0
    errors = []

    for N in [20, 40, 80]:
        grid = StructuredGrid.create_rectangle(Lx=Lx, Ly=Ly, Nx=N, Ny=N)
        grid.boundary_spec = BoundarySpec.bath_coupled()
        cond = BidomainConductivity(D_i=0.00124, D_e=D_e)
        fdm = BidomainFDMDiscretization(grid, cond)

        # u(x,y) = sin(pi*x/L)*sin(pi*y/L) satisfies homogeneous Dirichlet
        x, y = fdm.coordinates
        kx = np.pi / Lx
        ky = np.pi / Ly
        u = torch.sin(kx * x) * torch.sin(ky * y)

        # Exact: L_e * u = D_e * (-(kx^2 + ky^2)) * u
        exact = D_e * (-(kx**2 + ky**2)) * u

        # Numerical (only check interior nodes — boundary rows are identity/zero)
        Le_u = fdm.apply_L_e(u)

        # Build interior mask (non-boundary nodes)
        boundary_mask = grid.boundary_mask.flatten()
        interior = ~boundary_mask

        err = torch.abs(Le_u[interior] - exact[interior]).max().item()
        errors.append(err)

    ratio1 = errors[0] / errors[1]
    ratio2 = errors[1] / errors[2]

    assert ratio1 > 3.0, f"L_e Dirichlet convergence ratio 1: {ratio1:.2f} (expected ~4)"
    assert ratio2 > 3.0, f"L_e Dirichlet convergence ratio 2: {ratio2:.2f} (expected ~4)"


# === 2-T5: A_para SPD ===

def test_A_para_spd():
    """2-T5: get_parabolic_operators(), check A_para has all eigenvalues > 0."""
    fdm = _make_fdm(nx=10, ny=10)
    A_para, B_para = fdm.get_parabolic_operators(dt=0.01, theta=0.5)

    A_dense = _to_dense(A_para)
    eigvals = torch.linalg.eigvalsh(A_dense)
    min_eig = eigvals.min().item()

    assert min_eig > 0, f"A_para not positive definite: min eigenvalue = {min_eig}"

    # Also check A_para is symmetric
    diff = torch.abs(A_dense - A_dense.T).max().item()
    assert diff < 1e-12, f"A_para asymmetry: {diff}"


# === 2-T6: A_ellip SPD ===

def test_A_ellip_spd_neumann():
    """2-T6a: A_ellip with Neumann BCs — should be positive semi-definite (null space)."""
    fdm = _make_fdm(nx=10, ny=10, bc='insulated')
    A_ellip = _to_dense(fdm.get_elliptic_operator())

    eigvals = torch.linalg.eigvalsh(A_ellip)
    # Neumann: one eigenvalue should be ~0 (null space), rest > 0
    sorted_eigs = eigvals.sort().values
    assert sorted_eigs[0].item() < 1e-10, f"Expected null space, min eig = {sorted_eigs[0].item()}"
    assert sorted_eigs[1].item() > 1e-10, f"Second eigenvalue should be positive: {sorted_eigs[1].item()}"

    # Symmetric
    diff = torch.abs(A_ellip - A_ellip.T).max().item()
    assert diff < 1e-12, f"A_ellip asymmetry: {diff}"


def test_A_ellip_spd_dirichlet():
    """2-T6b: A_ellip with Dirichlet BCs — should be strictly positive definite."""
    fdm = _make_fdm(nx=10, ny=10, bc='bath')
    A_ellip = _to_dense(fdm.get_elliptic_operator())

    eigvals = torch.linalg.eigvalsh(A_ellip)
    min_eig = eigvals.min().item()
    assert min_eig > 0, f"A_ellip not positive definite with Dirichlet: min eig = {min_eig}"

    # Symmetric
    diff = torch.abs(A_ellip - A_ellip.T).max().item()
    assert diff < 1e-12, f"A_ellip asymmetry (Dirichlet): {diff}"


# === Additional: Basic functionality checks ===

def test_apply_L_ie():
    """Verify apply_L_ie = apply_L_i + apply_L_e."""
    fdm = _make_fdm(nx=10, ny=10)
    V = torch.randn(fdm.n_dof, dtype=torch.float64)

    L_i_V = fdm.apply_L_i(V)
    L_e_V = fdm.apply_L_e(V)
    L_ie_V = fdm.apply_L_ie(V)

    diff = torch.abs(L_ie_V - (L_i_V + L_e_V)).max().item()
    assert diff < 1e-14, f"L_ie != L_i + L_e: {diff}"


def test_repr():
    """Verify repr works."""
    fdm = _make_fdm(nx=10, ny=10)
    r = repr(fdm)
    assert "10x10" in r
    assert "insulated" in r


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
