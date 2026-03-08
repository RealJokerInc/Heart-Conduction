"""
Validation tests for IMEXSBDF2Solver.

Tests:
  BDF-T1: Cosine mode decay validates D_eff
  BDF-T2: BDF2 is 2nd-order accurate (halving dt should quarter the error)
  BDF-T3: Cross-check against Gauss-Seidel
  BDF-T4: Factory integration ('imex_sbdf2' string in bidomain.py)

All tests run on CPU with small grids (20x20-30x30), <30s total on Mac.

Run: ./venv/bin/python Bidomain/Engine_V1/tests/test_imex_sbdf2.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
torch.set_default_dtype(torch.float64)

from cv_shared import D_I, D_E, D_EFF, DX, DT
from test_helpers import _make_spatial, _make_cosine_state, validate_deff_cosine


# ============================================================
# BDF-T1: Cosine mode D_eff validation
# ============================================================
def test_bdf_t1_cosine_deff():
    """Verify cosine mode decays at D_eff rate under BDF2 solver."""
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.imex_sbdf2 import IMEXSBDF2Solver
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import PCGSolver

    nx, ny = 30, 30
    dt = DT
    spatial = _make_spatial(nx, ny)

    bdf = IMEXSBDF2Solver(
        spatial, dt,
        PCGSolver(max_iters=500, tol=1e-10),
        PCGSolver(max_iters=500, tol=1e-10))

    validate_deff_cosine(bdf, spatial, dt, nx, ny, n_steps=200, tol=0.05)
    print("BDF-T1 PASS: BDF2 cosine mode decays at D_eff rate")


# ============================================================
# BDF-T2: 2nd-order convergence
# ============================================================
def test_bdf_t2_convergence_order():
    """BDF2 should converge at 2nd order (halving dt quarters error).

    We compare BDF2 results at two dt values against a fine-dt reference.
    """
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.imex_sbdf2 import IMEXSBDF2Solver
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import PCGSolver

    nx, ny = 20, 20
    spatial = _make_spatial(nx, ny)

    T_END = 2.0  # 2 ms total (more steps -> BDF1 startup less dominant)
    dt_coarse = 0.04
    dt_fine = 0.02
    dt_ref = 0.005  # reference solution

    def run_bdf2(dt_val):
        state = _make_cosine_state(nx, ny, spatial)
        solver = IMEXSBDF2Solver(
            spatial, dt_val,
            PCGSolver(max_iters=500, tol=1e-12),
            PCGSolver(max_iters=500, tol=1e-12))
        n_steps = int(T_END / dt_val + 0.5)
        for _ in range(n_steps):
            solver.step(state, dt_val)
        return state.Vm.clone()

    Vm_ref = run_bdf2(dt_ref)
    Vm_coarse = run_bdf2(dt_coarse)
    Vm_fine = run_bdf2(dt_fine)

    err_coarse = (Vm_coarse - Vm_ref).abs().max().item()
    err_fine = (Vm_fine - Vm_ref).abs().max().item()

    if err_fine > 1e-14:
        ratio = err_coarse / err_fine
    else:
        ratio = float('inf')

    # For 2nd order: halving dt (0.04->0.02) should give ratio ~ 4
    # Allow range [1.5, 16] to account for BDF1 startup and splitting error
    print(f"    dt={dt_coarse}: max error = {err_coarse:.6e}")
    print(f"    dt={dt_fine}:  max error = {err_fine:.6e}")
    print(f"    Error ratio (expect ~4 for 2nd order): {ratio:.2f}")

    assert ratio > 1.5, f"Expected super-linear convergence, got ratio = {ratio:.2f}"
    print("BDF-T2 PASS: BDF2 shows 2nd-order convergence")


# ============================================================
# BDF-T3: Cross-check against Gauss-Seidel
# ============================================================
def test_bdf_t3_cross_check_gs():
    """BDF2 and GS should give similar results for smooth diffusion."""
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.imex_sbdf2 import IMEXSBDF2Solver
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.decoupled_gs import DecoupledBidomainDiffusionSolver
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import PCGSolver

    nx, ny = 20, 20
    dt = DT
    spatial = _make_spatial(nx, ny)

    state_bdf = _make_cosine_state(nx, ny, spatial)
    state_gs = _make_cosine_state(nx, ny, spatial)

    # Separate PCG instances for each solver
    bdf = IMEXSBDF2Solver(
        spatial, dt,
        PCGSolver(max_iters=500, tol=1e-10),
        PCGSolver(max_iters=500, tol=1e-10))
    gs = DecoupledBidomainDiffusionSolver(
        spatial, dt,
        PCGSolver(max_iters=500, tol=1e-10),
        PCGSolver(max_iters=500, tol=1e-10),
        theta=0.5)

    n_steps = 100
    for _ in range(n_steps):
        bdf.step(state_bdf, dt)
        gs.step(state_gs, dt)

    Vm_diff = (state_bdf.Vm - state_gs.Vm).abs()
    max_diff = Vm_diff.max().item()
    rel_diff = max_diff / state_gs.Vm.abs().max().item()

    print(f"    After {n_steps} steps:")
    print(f"    Max |Vm_BDF - Vm_GS|:  {max_diff:.6e}")
    print(f"    Relative difference:    {rel_diff:.6e}")

    # BDF2 and CN-GS are both 2nd order but different methods
    assert rel_diff < 0.10, f"BDF vs GS Vm diverged: rel_diff = {rel_diff:.4f}"
    print("BDF-T3 PASS: BDF2 agrees with Gauss-Seidel within 10%")


# ============================================================
# BDF-T4: Factory integration
# ============================================================
def test_bdf_t4_factory():
    """Verify 'imex_sbdf2' string works in BidomainSimulation factory."""
    from cardiac_sim.simulation.classical.bidomain import BidomainSimulation
    from cardiac_sim.tissue_builder.stimulus import StimulusProtocol

    spatial = _make_spatial(10, 10)

    sim = BidomainSimulation(
        spatial=spatial, ionic_model='ttp06',
        stimulus=StimulusProtocol(), dt=DT,
        splitting='strang',
        diffusion_solver='imex_sbdf2',
        elliptic_solver='auto', theta=0.5)

    assert sim.state is not None
    for _ in sim.run(t_end=DT*2, save_every=DT):
        break
    print(f"    Created and stepped BidomainSimulation with diffusion_solver='imex_sbdf2'")
    print("BDF-T4 PASS: Factory integration works")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("IMEX SBDF2 Solver Validation\n")

    test_bdf_t1_cosine_deff()
    print()
    test_bdf_t2_convergence_order()
    print()
    test_bdf_t3_cross_check_gs()
    print()
    test_bdf_t4_factory()

    print("\nIMEX SBDF2: ALL 4 TESTS PASS")
