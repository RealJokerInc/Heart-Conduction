"""
Validation tests for SemiImplicitSolver.

Tests:
  SI-T1: CFL enforcement (rejects dt > dt_max)
  SI-T2: Gaussian diffusion rate matches D_eff (pure diffusion, no ionic)
  SI-T3: Cross-check against Gauss-Seidel solver (same initial conditions)
  SI-T4: Factory integration ('semi_implicit' string in bidomain.py)

All tests run on CPU with small grids (20x20-30x30), <30s total on Mac.

Run: ./venv/bin/python Bidomain/Engine_V1/tests/test_semi_implicit.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
torch.set_default_dtype(torch.float64)

from cv_shared import D_I, D_E, D_EFF, DX, DT


# ============================================================
# SI-T1: CFL enforcement
# ============================================================
def test_si_t1_cfl_enforcement():
    """Verify SemiImplicitSolver rejects dt above CFL limit."""
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec
    from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
    from cardiac_sim.simulation.classical.discretization.fdm import BidomainFDMDiscretization
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.semi_implicit import SemiImplicitSolver
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import PCGSolver

    nx, ny = 10, 10
    Lx = DX * (nx - 1)
    grid = StructuredGrid(Nx=nx, Ny=ny, Lx=Lx, Ly=Lx,
                          boundary_spec=BoundarySpec.insulated())
    cond = BidomainConductivity(D_i=D_I, D_e=D_E)
    spatial = BidomainFDMDiscretization(grid, cond, Cm=1.0)
    solver = PCGSolver(max_iters=100, tol=1e-8)

    # dt = 0.01 should be fine (CFL limit ~ 0.126 for dx=0.025, D_i=0.00124)
    dt_ok = 0.01
    si = SemiImplicitSolver(spatial, dt_ok, solver)
    print(f"    dt={dt_ok}: CFL ratio = {si._cfl_ratio:.3f} (OK)")

    # dt = 0.2 should violate CFL
    dt_bad = 0.2
    try:
        SemiImplicitSolver(spatial, dt_bad, solver)
        assert False, "Should have raised ValueError for CFL violation"
    except ValueError as e:
        print(f"    dt={dt_bad}: correctly rejected — {e}")

    print("SI-T1 PASS: CFL enforcement works")


# ============================================================
# SI-T2: Gaussian diffusion rate
# ============================================================
def test_si_t2_gaussian_diffusion():
    """Verify Gaussian spreads at D_eff rate under semi-implicit solver."""
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec
    from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
    from cardiac_sim.simulation.classical.discretization.fdm import BidomainFDMDiscretization
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.semi_implicit import SemiImplicitSolver
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import PCGSolver
    from cardiac_sim.simulation.classical.state import BidomainState

    nx, ny = 30, 30
    dx = DX
    dt = DT  # 0.01 ms
    Lx = dx * (nx - 1)

    grid = StructuredGrid(Nx=nx, Ny=ny, Lx=Lx, Ly=Lx,
                          boundary_spec=BoundarySpec.insulated())
    cond = BidomainConductivity(D_i=D_I, D_e=D_E)
    spatial = BidomainFDMDiscretization(grid, cond, Cm=1.0)

    # Gaussian initial condition
    x, y = spatial.coordinates
    x_center, y_center = Lx / 2, Lx / 2
    sigma_0 = 5 * dx

    Vm = torch.exp(-((x - x_center)**2 + (y - y_center)**2) / (2 * sigma_0**2))
    phi_e = torch.zeros_like(Vm)

    state = BidomainState(
        spatial=spatial, n_dof=spatial.n_dof, x=x, y=y,
        Vm=Vm.clone(), phi_e=phi_e,
        ionic_states=torch.zeros(spatial.n_dof, 1),
        gate_indices=[], concentration_indices=[],
    )

    # Build solver
    ellip_solver = PCGSolver(max_iters=500, tol=1e-10)
    si = SemiImplicitSolver(spatial, dt, ellip_solver)

    # Measure initial x-variance
    def measure_variance_x(Vm_flat):
        Vm_grid = grid.flat_to_grid(Vm_flat)
        marginal = Vm_grid.sum(dim=1)
        marginal = marginal / marginal.sum()
        x_1d = torch.linspace(0, Lx, nx)
        mean_x = (x_1d * marginal).sum()
        var_x = ((x_1d - mean_x)**2 * marginal).sum()
        return var_x.item()

    var_0 = measure_variance_x(state.Vm)

    # Run 200 steps (t = 2.0 ms)
    n_steps = 200
    for _ in range(n_steps):
        si.step(state, dt)

    var_final = measure_variance_x(state.Vm)
    t_total = n_steps * dt

    expected_growth = 2 * D_EFF * t_total
    actual_growth = var_final - var_0
    rel_err = abs(actual_growth - expected_growth) / expected_growth

    print(f"    Initial variance: {var_0:.6f} cm^2")
    print(f"    Final variance:   {var_final:.6f} cm^2")
    print(f"    Actual growth:    {actual_growth:.6f} cm^2")
    print(f"    Expected growth:  {expected_growth:.6f} cm^2")
    print(f"    Relative error:   {rel_err:.4f}")

    assert rel_err < 0.15, f"Gaussian variance growth error: {rel_err:.4f}"
    print("SI-T2 PASS: Semi-implicit Gaussian diffuses at D_eff rate")


# ============================================================
# SI-T3: Cross-check against Gauss-Seidel
# ============================================================
def test_si_t3_cross_check_gs():
    """Semi-implicit and GS should give similar results for small dt."""
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec
    from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
    from cardiac_sim.simulation.classical.discretization.fdm import BidomainFDMDiscretization
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.semi_implicit import SemiImplicitSolver
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.decoupled_gs import DecoupledBidomainDiffusionSolver
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import PCGSolver
    from cardiac_sim.simulation.classical.state import BidomainState

    nx, ny = 20, 20
    dx = DX
    dt = DT
    Lx = dx * (nx - 1)

    grid = StructuredGrid(Nx=nx, Ny=ny, Lx=Lx, Ly=Lx,
                          boundary_spec=BoundarySpec.insulated())
    cond = BidomainConductivity(D_i=D_I, D_e=D_E)
    spatial = BidomainFDMDiscretization(grid, cond, Cm=1.0)

    x, y = spatial.coordinates
    x_center, y_center = Lx / 2, Lx / 2
    sigma_0 = 5 * dx
    Vm_init = torch.exp(-((x - x_center)**2 + (y - y_center)**2) / (2 * sigma_0**2))

    # State for semi-implicit
    state_si = BidomainState(
        spatial=spatial, n_dof=spatial.n_dof, x=x, y=y,
        Vm=Vm_init.clone(), phi_e=torch.zeros_like(Vm_init),
        ionic_states=torch.zeros(spatial.n_dof, 1),
        gate_indices=[], concentration_indices=[],
    )

    # State for GS
    state_gs = BidomainState(
        spatial=spatial, n_dof=spatial.n_dof, x=x, y=y,
        Vm=Vm_init.clone(), phi_e=torch.zeros_like(Vm_init),
        ionic_states=torch.zeros(spatial.n_dof, 1),
        gate_indices=[], concentration_indices=[],
    )

    # Build both solvers
    pcg = PCGSolver(max_iters=500, tol=1e-10)
    pcg2 = PCGSolver(max_iters=500, tol=1e-10)
    si = SemiImplicitSolver(spatial, dt, pcg)
    gs = DecoupledBidomainDiffusionSolver(spatial, dt, pcg2, pcg2, theta=0.5)

    # Run 100 steps
    n_steps = 100
    for _ in range(n_steps):
        si.step(state_si, dt)
        gs.step(state_gs, dt)

    # Compare Vm
    Vm_diff = (state_si.Vm - state_gs.Vm).abs()
    max_diff = Vm_diff.max().item()
    rel_diff = max_diff / state_gs.Vm.abs().max().item()

    print(f"    After {n_steps} steps:")
    print(f"    Max |Vm_SI - Vm_GS|:   {max_diff:.6e}")
    print(f"    Relative difference:    {rel_diff:.6e}")

    # Compare phi_e
    phi_diff = (state_si.phi_e - state_gs.phi_e).abs()
    max_phi_diff = phi_diff.max().item()
    print(f"    Max |phi_SI - phi_GS|:  {max_phi_diff:.6e}")

    # They won't be identical (different methods), but should agree within ~5%
    # for small dt well within CFL
    assert rel_diff < 0.10, f"SI vs GS Vm diverged: rel_diff = {rel_diff:.4f}"
    print("SI-T3 PASS: Semi-implicit agrees with Gauss-Seidel within 10%")


# ============================================================
# SI-T4: Factory integration
# ============================================================
def test_si_t4_factory():
    """Verify 'semi_implicit' string works in BidomainSimulation factory."""
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec
    from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
    from cardiac_sim.simulation.classical.discretization.fdm import BidomainFDMDiscretization
    from cardiac_sim.simulation.classical.bidomain import BidomainSimulation
    from cardiac_sim.tissue_builder.stimulus import StimulusProtocol

    nx, ny = 10, 10
    Lx = DX * (nx - 1)
    grid = StructuredGrid(Nx=nx, Ny=ny, Lx=Lx, Ly=Lx,
                          boundary_spec=BoundarySpec.insulated())
    cond = BidomainConductivity(D_i=D_I, D_e=D_E)
    spatial = BidomainFDMDiscretization(grid, cond, Cm=1.0)

    sim = BidomainSimulation(
        spatial=spatial, ionic_model='ttp06',
        stimulus=StimulusProtocol(), dt=DT,
        splitting='strang',
        diffusion_solver='semi_implicit',
        elliptic_solver='auto', theta=0.5)

    assert sim.state is not None
    print(f"    Created BidomainSimulation with diffusion_solver='semi_implicit'")
    print(f"    Elliptic solver: {sim._elliptic_solver_name}")
    print("SI-T4 PASS: Factory integration works")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Semi-Implicit Solver Validation\n")

    test_si_t1_cfl_enforcement()
    print()
    test_si_t2_gaussian_diffusion()
    print()
    test_si_t3_cross_check_gs()
    print()
    test_si_t4_factory()

    print("\nSemi-Implicit: ALL 4 TESTS PASS")
