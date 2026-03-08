"""
Phase 4 Validation Tests — Decoupled Bidomain Diffusion Solver

Tests 4-T1 through 4-T7 from PROGRESS.md.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
import numpy as np


def _make_system(nx=15, ny=15, lx=1.0, ly=1.0, D_i=0.00124, D_e=0.00446,
                 bc='insulated', dt=0.01, theta=0.5):
    """Helper: create FDM + DecoupledBidomainDiffusionSolver + BidomainState."""
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec
    from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
    from cardiac_sim.simulation.classical.discretization.fdm import BidomainFDMDiscretization
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.decoupled_gs import DecoupledBidomainDiffusionSolver
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import PCGSolver
    from cardiac_sim.simulation.classical.state import BidomainState

    grid = StructuredGrid.create_rectangle(Lx=lx, Ly=ly, Nx=nx, Ny=ny)
    if bc == 'insulated':
        grid.boundary_spec = BoundarySpec.insulated()
    elif bc == 'bath':
        grid.boundary_spec = BoundarySpec.bath_coupled()
    cond = BidomainConductivity(D_i=D_i, D_e=D_e)
    fdm = BidomainFDMDiscretization(grid, cond, Cm=1.0)

    para_solver = PCGSolver(max_iters=500, tol=1e-10)
    ellip_solver = PCGSolver(max_iters=500, tol=1e-10)

    diffusion = DecoupledBidomainDiffusionSolver(
        fdm, dt, para_solver, ellip_solver, theta=theta)

    # Create state
    x, y = fdm.coordinates
    n_dof = fdm.n_dof
    Vm = torch.zeros(n_dof, dtype=torch.float64)
    phi_e = torch.zeros(n_dof, dtype=torch.float64)
    ionic_states = torch.zeros(n_dof, 0, dtype=torch.float64)

    state = BidomainState(
        spatial=fdm, n_dof=n_dof, x=x, y=y,
        Vm=Vm, phi_e=phi_e, ionic_states=ionic_states,
        gate_indices=[], concentration_indices=[]
    )

    return fdm, diffusion, state


# === 4-T1: Parabolic only (no coupling) ===

def test_parabolic_only():
    """4-T1: With phi_e=0, bidomain parabolic step reduces to monodomain."""
    fdm, diffusion, state = _make_system(nx=15, ny=15, dt=0.01, bc='insulated')

    # Initialize Vm to a Gaussian pulse
    x, y = fdm.coordinates
    cx, cy = 0.5, 0.5
    state.Vm.copy_(torch.exp(-((x - cx)**2 + (y - cy)**2) / 0.01))
    state.phi_e.zero_()

    Vm_before = state.Vm.clone()

    # With phi_e=0, the parabolic step is:
    # A_para * Vm_new = B_para * Vm_old + theta * L_i * 0 = B_para * Vm_old
    # This is identical to monodomain diffusion with D_i
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import sparse_mv
    rhs_expected = sparse_mv(diffusion.B_para, Vm_before)
    Vm_expected = diffusion.parabolic_solver.solve(diffusion.A_para, rhs_expected)

    # Run one step
    diffusion.step(state, 0.01)

    # After step, phi_e should be nonzero (elliptic solve with L_i * Vm_new)
    # But Vm should match the monodomain-equivalent step
    err = torch.abs(state.Vm - Vm_expected).max().item()
    assert err < 1e-8, f"Parabolic-only Vm error: {err}"


# === 4-T2: Elliptic only (static) ===

def test_elliptic_static():
    """4-T2: For uniform Vm, phi_e should be proportional to -sigma_i/(sigma_i+sigma_e)*Vm."""
    fdm, diffusion, state = _make_system(nx=15, ny=15, dt=0.01, bc='insulated')

    D_i, D_e = 0.00124, 0.00446

    # Set Vm to a Neumann-compatible function
    x, y = fdm.coordinates
    Lx, Ly = 1.0, 1.0
    kx, ky = np.pi / Lx, np.pi / Ly
    state.Vm.copy_(torch.cos(kx * x) * torch.cos(ky * y))

    # Solve elliptic: -(L_i + L_e) * phi_e = L_i * Vm
    # For isotropic: L_i = D_i * Lap, L_e = D_e * Lap
    # -(D_i + D_e) * Lap * phi_e = D_i * Lap * Vm
    # phi_e = -D_i / (D_i + D_e) * Vm (up to constant, for Neumann)
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import sparse_mv

    rhs_ellip = fdm.apply_L_i(state.Vm)
    rhs_ellip[0] = 0.0  # pinning
    phi_e = diffusion.elliptic_solver.solve(diffusion.A_ellip, rhs_ellip)
    phi_e = phi_e - phi_e[0]  # post-subtract pinning

    expected_ratio = -D_i / (D_i + D_e)
    # phi_e should be proportional to Vm (same spatial pattern)
    # Check ratio at interior nodes (boundary may differ)
    boundary_mask = fdm.grid.boundary_mask.flatten()
    interior = ~boundary_mask
    Vm_int = state.Vm[interior]
    phi_e_int = phi_e[interior]

    # Where Vm is large enough, check ratio
    large_mask = Vm_int.abs() > 0.1
    ratios = phi_e_int[large_mask] / Vm_int[large_mask]
    mean_ratio = ratios.mean().item()
    rel_err = abs(mean_ratio - expected_ratio) / abs(expected_ratio)

    assert rel_err < 0.05, f"Elliptic ratio error: mean={mean_ratio:.6f}, expected={expected_ratio:.6f}, rel_err={rel_err:.4f}"


# === 4-T3: Coupled step (Neumann) — energy check ===

def test_coupled_neumann_energy():
    """4-T3: Full step with insulated BCs, check energy doesn't blow up."""
    fdm, diffusion, state = _make_system(nx=15, ny=15, dt=0.01, bc='insulated')

    # Initialize with Gaussian pulse
    x, y = fdm.coordinates
    state.Vm.copy_(torch.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.01))

    # Run 10 steps
    energy_history = []
    for _ in range(10):
        energy = (state.Vm**2).sum().item()
        energy_history.append(energy)
        diffusion.step(state, 0.01)

    # Energy should be bounded (diffusion dissipates)
    for i in range(1, len(energy_history)):
        assert energy_history[i] <= energy_history[i - 1] * 1.01, \
            f"Energy increased at step {i}: {energy_history[i]:.6f} > {energy_history[i-1]:.6f}"


# === 4-T4: Coupled step (Dirichlet) — boundary check ===

def test_coupled_dirichlet_boundary():
    """4-T4: Full step with bath BCs, phi_e=0 at boundary nodes."""
    fdm, diffusion, state = _make_system(nx=15, ny=15, dt=0.01, bc='bath')

    # Initialize with Gaussian pulse
    x, y = fdm.coordinates
    state.Vm.copy_(torch.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.01))

    # Run a few steps
    for _ in range(5):
        diffusion.step(state, 0.01)

    # phi_e should be zero at boundary nodes
    boundary_mask = fdm.grid.boundary_mask.flatten()
    boundary_phi_e = state.phi_e[boundary_mask]
    max_boundary = boundary_phi_e.abs().max().item()
    assert max_boundary < 1e-8, f"phi_e at boundary not zero: {max_boundary}"


# === 4-T5: Null space pinning (Neumann) ===

def test_null_space_pinning():
    """4-T5: Solve with all-Neumann, verify phi_e has zero at pinned node."""
    fdm, diffusion, state = _make_system(nx=15, ny=15, dt=0.01, bc='insulated')

    # The solver should have _needs_pinning = True
    assert diffusion._needs_pinning, "Expected pinning for Neumann BCs"

    # Initialize and run
    x, y = fdm.coordinates
    state.Vm.copy_(torch.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.01))
    diffusion.step(state, 0.01)

    # phi_e should have zero at pinned node (node 0)
    assert abs(state.phi_e[0].item()) < 1e-10, \
        f"phi_e at pinned node: {state.phi_e[0].item()}"


# === 4-T6: No pinning (Dirichlet) ===

def test_no_pinning_dirichlet():
    """4-T6: Solve with bath BCs, verify no pinning was applied."""
    fdm, diffusion, state = _make_system(nx=15, ny=15, dt=0.01, bc='bath')

    assert not diffusion._needs_pinning, "Should not need pinning for Dirichlet BCs"

    # Initialize and run
    x, y = fdm.coordinates
    state.Vm.copy_(torch.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.01))
    diffusion.step(state, 0.01)

    # phi_e should be uniquely determined (not all zeros if Vm is nonzero)
    assert state.phi_e.abs().max().item() > 1e-10, \
        "phi_e should be nonzero for nonzero Vm"


# === 4-T7: Operator rebuild ===

def test_operator_rebuild():
    """4-T7: rebuild_operators(new_dt) changes A_para."""
    fdm, diffusion, state = _make_system(nx=10, ny=10, dt=0.01, bc='insulated')

    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import extract_diagonal

    diag_old = extract_diagonal(diffusion.A_para).clone()

    diffusion.rebuild_operators(fdm, dt=0.005)

    diag_new = extract_diagonal(diffusion.A_para)

    # A_para = 1/dt * I - theta*L_i
    # Changing dt changes 1/dt, so diagonal should change
    diff = torch.abs(diag_old - diag_new).max().item()
    assert diff > 1.0, f"A_para diagonal didn't change after rebuild: diff={diff}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
