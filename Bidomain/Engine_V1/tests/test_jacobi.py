"""
Validation tests for DecoupledJacobiSolver.

Tests:
  JA-T1: Gaussian diffusion rate matches D_eff
  JA-T2: Cross-check against Gauss-Seidel (both should converge to same solution)
  JA-T3: Factory integration ('jacobi' string in bidomain.py)

All tests run on CPU with small grids (20x20), <15s total on Mac.

Run: ./venv/bin/python Bidomain/Engine_V1/tests/test_jacobi.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
torch.set_default_dtype(torch.float64)

from cv_shared import D_I, D_E, D_EFF, DX, DT


def _make_gaussian_state(nx, ny, spatial):
    """Create a Gaussian initial state for diffusion tests."""
    from cardiac_sim.simulation.classical.state import BidomainState

    dx = DX
    Lx = dx * (nx - 1)
    x, y = spatial.coordinates
    x_center, y_center = Lx / 2, Lx / 2
    sigma_0 = 5 * dx

    Vm = torch.exp(-((x - x_center)**2 + (y - y_center)**2) / (2 * sigma_0**2))
    return BidomainState(
        spatial=spatial, n_dof=spatial.n_dof, x=x, y=y,
        Vm=Vm.clone(), phi_e=torch.zeros_like(Vm),
        ionic_states=torch.zeros(spatial.n_dof, 1),
        gate_indices=[], concentration_indices=[],
    )


def _make_spatial(nx, ny):
    """Create FDM spatial discretization."""
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec
    from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
    from cardiac_sim.simulation.classical.discretization.fdm import BidomainFDMDiscretization

    Lx = DX * (nx - 1)
    grid = StructuredGrid(Nx=nx, Ny=ny, Lx=Lx, Ly=Lx,
                          boundary_spec=BoundarySpec.insulated())
    cond = BidomainConductivity(D_i=D_I, D_e=D_E)
    return BidomainFDMDiscretization(grid, cond, Cm=1.0)


# ============================================================
# JA-T1: Gaussian diffusion rate
# ============================================================
def test_ja_t1_gaussian_diffusion():
    """Verify Gaussian spreads at D_eff rate under Jacobi solver."""
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.decoupled_jacobi import DecoupledJacobiSolver
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import PCGSolver

    nx, ny = 30, 30
    dt = DT
    Lx = DX * (nx - 1)
    spatial = _make_spatial(nx, ny)
    state = _make_gaussian_state(nx, ny, spatial)

    pcg = PCGSolver(max_iters=500, tol=1e-10)
    solver = DecoupledJacobiSolver(spatial, dt, pcg, pcg, theta=0.5)

    def measure_variance_x(Vm_flat):
        Vm_grid = spatial.grid.flat_to_grid(Vm_flat)
        marginal = Vm_grid.sum(dim=1)
        marginal = marginal / marginal.sum()
        x_1d = torch.linspace(0, Lx, nx)
        mean_x = (x_1d * marginal).sum()
        return ((x_1d - mean_x)**2 * marginal).sum().item()

    var_0 = measure_variance_x(state.Vm)

    n_steps = 200
    for _ in range(n_steps):
        solver.step(state, dt)

    var_final = measure_variance_x(state.Vm)
    t_total = n_steps * dt

    expected_growth = 2 * D_EFF * t_total
    actual_growth = var_final - var_0
    rel_err = abs(actual_growth - expected_growth) / expected_growth

    print(f"    Actual growth:   {actual_growth:.6f} cm^2")
    print(f"    Expected growth: {expected_growth:.6f} cm^2")
    print(f"    Relative error:  {rel_err:.4f}")

    assert rel_err < 0.15, f"Gaussian variance growth error: {rel_err:.4f}"
    print("JA-T1 PASS: Jacobi Gaussian diffuses at D_eff rate")


# ============================================================
# JA-T2: Cross-check against Gauss-Seidel
# ============================================================
def test_ja_t2_cross_check_gs():
    """Jacobi and GS should give similar results (Jacobi slightly more splitting error)."""
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.decoupled_jacobi import DecoupledJacobiSolver
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.decoupled_gs import DecoupledBidomainDiffusionSolver
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import PCGSolver

    nx, ny = 20, 20
    dt = DT
    spatial = _make_spatial(nx, ny)

    state_ja = _make_gaussian_state(nx, ny, spatial)
    state_gs = _make_gaussian_state(nx, ny, spatial)

    pcg = PCGSolver(max_iters=500, tol=1e-10)
    ja = DecoupledJacobiSolver(spatial, dt, pcg, pcg, theta=0.5)
    gs = DecoupledBidomainDiffusionSolver(spatial, dt, pcg, pcg, theta=0.5)

    n_steps = 100
    for _ in range(n_steps):
        ja.step(state_ja, dt)
        gs.step(state_gs, dt)

    Vm_diff = (state_ja.Vm - state_gs.Vm).abs()
    max_diff = Vm_diff.max().item()
    rel_diff = max_diff / state_gs.Vm.abs().max().item()

    print(f"    After {n_steps} steps:")
    print(f"    Max |Vm_JA - Vm_GS|:  {max_diff:.6e}")
    print(f"    Relative difference:   {rel_diff:.6e}")

    # Jacobi uses lagged Vm in elliptic, so slightly more splitting error
    assert rel_diff < 0.10, f"Jacobi vs GS Vm diverged: rel_diff = {rel_diff:.4f}"
    print("JA-T2 PASS: Jacobi agrees with Gauss-Seidel within 10%")


# ============================================================
# JA-T3: Factory integration
# ============================================================
def test_ja_t3_factory():
    """Verify 'jacobi' string works in BidomainSimulation factory."""
    from cardiac_sim.simulation.classical.bidomain import BidomainSimulation
    from cardiac_sim.tissue_builder.stimulus import StimulusProtocol

    nx, ny = 10, 10
    spatial = _make_spatial(nx, ny)

    sim = BidomainSimulation(
        spatial=spatial, ionic_model='ttp06',
        stimulus=StimulusProtocol(), dt=DT,
        splitting='strang',
        diffusion_solver='jacobi',
        elliptic_solver='auto', theta=0.5)

    assert sim.state is not None
    print(f"    Created BidomainSimulation with diffusion_solver='jacobi'")
    print("JA-T3 PASS: Factory integration works")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Jacobi Solver Validation\n")

    test_ja_t1_gaussian_diffusion()
    print()
    test_ja_t2_cross_check_gs()
    print()
    test_ja_t3_factory()

    print("\nJacobi: ALL 3 TESTS PASS")
