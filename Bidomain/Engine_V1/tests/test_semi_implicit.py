"""
Validation tests for SemiImplicitSolver.

Tests:
  SI-T1: CFL enforcement (rejects dt > dt_max)
  SI-T2: Cosine mode decay validates D_eff (pure diffusion, no ionic)
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
from test_helpers import _make_spatial, _make_cosine_state, validate_deff_cosine


# ============================================================
# SI-T1: CFL enforcement
# ============================================================
def test_si_t1_cfl_enforcement():
    """Verify SemiImplicitSolver rejects dt above CFL limit."""
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.semi_implicit import SemiImplicitSolver
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import PCGSolver

    spatial = _make_spatial(10, 10)
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
# SI-T2: Cosine mode D_eff validation
# ============================================================
def test_si_t2_cosine_deff():
    """Verify cosine mode decays at D_eff rate under semi-implicit solver."""
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.semi_implicit import SemiImplicitSolver
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import PCGSolver

    nx, ny = 30, 30
    dt = DT
    spatial = _make_spatial(nx, ny)

    ellip_solver = PCGSolver(max_iters=500, tol=1e-10)
    si = SemiImplicitSolver(spatial, dt, ellip_solver)

    validate_deff_cosine(si, spatial, dt, nx, ny, n_steps=200, tol=0.05)
    print("SI-T2 PASS: Semi-implicit cosine mode decays at D_eff rate")


# ============================================================
# SI-T3: Cross-check against Gauss-Seidel
# ============================================================
def test_si_t3_cross_check_gs():
    """Semi-implicit and GS should give similar results for small dt."""
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.semi_implicit import SemiImplicitSolver
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.decoupled_gs import DecoupledBidomainDiffusionSolver
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import PCGSolver

    nx, ny = 20, 20
    dt = DT
    spatial = _make_spatial(nx, ny)

    state_si = _make_cosine_state(nx, ny, spatial)
    state_gs = _make_cosine_state(nx, ny, spatial)

    # Separate PCG instances for each solver to avoid warm-start contamination
    si = SemiImplicitSolver(spatial, dt, PCGSolver(max_iters=500, tol=1e-10))
    gs = DecoupledBidomainDiffusionSolver(
        spatial, dt,
        PCGSolver(max_iters=500, tol=1e-10),
        PCGSolver(max_iters=500, tol=1e-10),
        theta=0.5)

    n_steps = 100
    for _ in range(n_steps):
        si.step(state_si, dt)
        gs.step(state_gs, dt)

    Vm_diff = (state_si.Vm - state_gs.Vm).abs()
    max_diff = Vm_diff.max().item()
    rel_diff = max_diff / state_gs.Vm.abs().max().item()

    print(f"    After {n_steps} steps:")
    print(f"    Max |Vm_SI - Vm_GS|:   {max_diff:.6e}")
    print(f"    Relative difference:    {rel_diff:.6e}")

    assert rel_diff < 0.10, f"SI vs GS Vm diverged: rel_diff = {rel_diff:.4f}"
    print("SI-T3 PASS: Semi-implicit agrees with Gauss-Seidel within 10%")


# ============================================================
# SI-T4: Factory integration
# ============================================================
def test_si_t4_factory():
    """Verify 'semi_implicit' string works in BidomainSimulation factory."""
    from cardiac_sim.simulation.classical.bidomain import BidomainSimulation
    from cardiac_sim.tissue_builder.stimulus import StimulusProtocol

    spatial = _make_spatial(10, 10)

    sim = BidomainSimulation(
        spatial=spatial, ionic_model='ttp06',
        stimulus=StimulusProtocol(), dt=DT,
        splitting='strang',
        diffusion_solver='semi_implicit',
        elliptic_solver='auto', theta=0.5)

    assert sim.state is not None
    # Run a short simulation to verify the solver actually works end-to-end
    for _ in sim.run(t_end=DT*2, save_every=DT):
        break
    print(f"    Created and stepped BidomainSimulation with diffusion_solver='semi_implicit'")
    print("SI-T4 PASS: Factory integration works")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Semi-Implicit Solver Validation\n")

    test_si_t1_cfl_enforcement()
    print()
    test_si_t2_cosine_deff()
    print()
    test_si_t3_cross_check_gs()
    print()
    test_si_t4_factory()

    print("\nSemi-Implicit: ALL 4 TESTS PASS")
