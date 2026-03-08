"""
Validation tests for DecoupledJacobiSolver.

Tests:
  JA-T1: Cosine mode decay validates D_eff
  JA-T2: Cross-check against Gauss-Seidel solver
  JA-T3: Factory integration ('jacobi' string in bidomain.py)

All tests run on CPU with small grids (20x20-30x30), <30s total on Mac.

Run: ./venv/bin/python Bidomain/Engine_V1/tests/test_jacobi.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
torch.set_default_dtype(torch.float64)

from cv_shared import D_I, D_E, D_EFF, DX, DT
from test_helpers import _make_spatial, _make_cosine_state, validate_deff_cosine


# ============================================================
# JA-T1: Cosine mode D_eff validation
# ============================================================
def test_ja_t1_cosine_deff():
    """Verify cosine mode decays at D_eff rate under Jacobi solver."""
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.decoupled_jacobi import DecoupledJacobiSolver
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import PCGSolver

    nx, ny = 30, 30
    dt = DT
    spatial = _make_spatial(nx, ny)

    ja = DecoupledJacobiSolver(
        spatial, dt,
        PCGSolver(max_iters=500, tol=1e-10),
        PCGSolver(max_iters=500, tol=1e-10),
        theta=0.5)

    validate_deff_cosine(ja, spatial, dt, nx, ny, n_steps=200, tol=0.05)
    print("JA-T1 PASS: Jacobi cosine mode decays at D_eff rate")


# ============================================================
# JA-T2: Cross-check against Gauss-Seidel
# ============================================================
def test_ja_t2_cross_check_gs():
    """Jacobi and GS should give similar results."""
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.decoupled_jacobi import DecoupledJacobiSolver
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.decoupled_gs import DecoupledBidomainDiffusionSolver
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import PCGSolver

    nx, ny = 20, 20
    dt = DT
    spatial = _make_spatial(nx, ny)

    state_ja = _make_cosine_state(nx, ny, spatial)
    state_gs = _make_cosine_state(nx, ny, spatial)

    # Separate PCG instances for each solver to avoid warm-start contamination
    ja = DecoupledJacobiSolver(
        spatial, dt,
        PCGSolver(max_iters=500, tol=1e-10),
        PCGSolver(max_iters=500, tol=1e-10),
        theta=0.5)
    gs = DecoupledBidomainDiffusionSolver(
        spatial, dt,
        PCGSolver(max_iters=500, tol=1e-10),
        PCGSolver(max_iters=500, tol=1e-10),
        theta=0.5)

    n_steps = 100
    for _ in range(n_steps):
        ja.step(state_ja, dt)
        gs.step(state_gs, dt)

    Vm_diff = (state_ja.Vm - state_gs.Vm).abs()
    max_diff = Vm_diff.max().item()
    rel_diff = max_diff / state_gs.Vm.abs().max().item()

    print(f"    After {n_steps} steps:")
    print(f"    Max |Vm_JA - Vm_GS|:   {max_diff:.6e}")
    print(f"    Relative difference:    {rel_diff:.6e}")

    assert rel_diff < 0.10, f"Jacobi vs GS Vm diverged: rel_diff = {rel_diff:.4f}"
    print("JA-T2 PASS: Jacobi agrees with Gauss-Seidel within 10%")


# ============================================================
# JA-T3: Factory integration
# ============================================================
def test_ja_t3_factory():
    """Verify 'jacobi' string works in BidomainSimulation factory."""
    from cardiac_sim.simulation.classical.bidomain import BidomainSimulation
    from cardiac_sim.tissue_builder.stimulus import StimulusProtocol

    spatial = _make_spatial(10, 10)

    sim = BidomainSimulation(
        spatial=spatial, ionic_model='ttp06',
        stimulus=StimulusProtocol(), dt=DT,
        splitting='strang',
        diffusion_solver='jacobi',
        elliptic_solver='auto', theta=0.5)

    assert sim.state is not None
    for _ in sim.run(t_end=DT*2, save_every=DT):
        break
    print(f"    Created and stepped BidomainSimulation with diffusion_solver='jacobi'")
    print("JA-T3 PASS: Factory integration works")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Jacobi Solver Validation\n")

    test_ja_t1_cosine_deff()
    print()
    test_ja_t2_cross_check_gs()
    print()
    test_ja_t3_factory()

    print("\nJacobi: ALL 3 TESTS PASS")
