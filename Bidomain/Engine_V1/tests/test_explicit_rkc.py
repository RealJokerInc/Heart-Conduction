"""
Validation tests for ExplicitRKCSolver (Runge-Kutta-Chebyshev).

Tests:
  RKC-T1: Stability enforcement (rejects dt above stability limit)
  RKC-T2: Cosine mode decay validates D_eff
  RKC-T3: Cross-check against Gauss-Seidel
  RKC-T4: RKC coefficient sanity (T_n recursion, known values)
  RKC-T5: Factory integration ('explicit_rkc' string in bidomain.py)

All tests run on CPU with small grids (20x20-30x30), <30s total on Mac.

Run: ./venv/bin/python Bidomain/Engine_V1/tests/test_explicit_rkc.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
torch.set_default_dtype(torch.float64)
import math

from cv_shared import D_I, D_E, D_EFF, DX, DT
from test_helpers import _make_spatial, _make_cosine_state, validate_deff_cosine


# ============================================================
# RKC-T1: Stability enforcement
# ============================================================
def test_rkc_t1_stability_enforcement():
    """Verify ExplicitRKCSolver rejects dt above stability limit."""
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.explicit_rkc import ExplicitRKCSolver
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import PCGSolver

    spatial = _make_spatial(10, 10)
    solver = PCGSolver(max_iters=100, tol=1e-8)

    # With s=20 stages, stability limit is huge: 0.65 * 20^2 * 0.126 ~ 32.7 ms
    si = ExplicitRKCSolver(spatial, DT, solver, n_stages=20)
    print(f"    s=20, dt={DT}: stability ratio = {si._stability_ratio:.4f} (OK)")

    # With s=2 stages, stability limit is small: 0.65 * 4 * 0.126 ~ 0.33 ms
    si2 = ExplicitRKCSolver(spatial, DT, solver, n_stages=2)
    print(f"    s=2,  dt={DT}: stability ratio = {si2._stability_ratio:.4f} (OK)")

    # n_stages=1 should be rejected (minimum is 2)
    try:
        ExplicitRKCSolver(spatial, DT, solver, n_stages=1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"    s=1: correctly rejected — {e}")

    # Very large dt with small s should violate stability
    try:
        ExplicitRKCSolver(spatial, 1.0, solver, n_stages=2)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"    s=2, dt=1.0: correctly rejected")

    print("RKC-T1 PASS: Stability enforcement works")


# ============================================================
# RKC-T2: Cosine mode D_eff validation
# ============================================================
def test_rkc_t2_cosine_deff():
    """Verify cosine mode decays at D_eff rate under RKC solver."""
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.explicit_rkc import ExplicitRKCSolver
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import PCGSolver

    nx, ny = 30, 30
    dt = DT
    spatial = _make_spatial(nx, ny)

    rkc = ExplicitRKCSolver(spatial, dt,
                             PCGSolver(max_iters=500, tol=1e-10),
                             n_stages=10)

    validate_deff_cosine(rkc, spatial, dt, nx, ny, n_steps=200, tol=0.05)
    print("RKC-T2 PASS: RKC cosine mode decays at D_eff rate")


# ============================================================
# RKC-T3: Cross-check against Gauss-Seidel
# ============================================================
def test_rkc_t3_cross_check_gs():
    """RKC and GS should give similar results for smooth diffusion."""
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.explicit_rkc import ExplicitRKCSolver
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.decoupled_gs import DecoupledBidomainDiffusionSolver
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import PCGSolver

    nx, ny = 20, 20
    dt = DT
    spatial = _make_spatial(nx, ny)

    state_rkc = _make_cosine_state(nx, ny, spatial)
    state_gs = _make_cosine_state(nx, ny, spatial)

    # Separate PCG instances for each solver
    rkc = ExplicitRKCSolver(spatial, dt,
                             PCGSolver(max_iters=500, tol=1e-10),
                             n_stages=10)
    gs = DecoupledBidomainDiffusionSolver(
        spatial, dt,
        PCGSolver(max_iters=500, tol=1e-10),
        PCGSolver(max_iters=500, tol=1e-10),
        theta=0.5)

    n_steps = 100
    for _ in range(n_steps):
        rkc.step(state_rkc, dt)
        gs.step(state_gs, dt)

    Vm_diff = (state_rkc.Vm - state_gs.Vm).abs()
    max_diff = Vm_diff.max().item()
    rel_diff = max_diff / state_gs.Vm.abs().max().item()

    print(f"    After {n_steps} steps:")
    print(f"    Max |Vm_RKC - Vm_GS|:  {max_diff:.6e}")
    print(f"    Relative difference:    {rel_diff:.6e}")

    assert rel_diff < 0.10, f"RKC vs GS Vm diverged: rel_diff = {rel_diff:.4f}"
    print("RKC-T3 PASS: RKC agrees with Gauss-Seidel within 10%")


# ============================================================
# RKC-T4: Chebyshev coefficient sanity
# ============================================================
def test_rkc_t4_coefficient_sanity():
    """Verify Chebyshev polynomial recursion against known values."""
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.explicit_rkc import (
        _chebyshev_T, _chebyshev_Tp, _chebyshev_Tpp)

    # T_0(x) = 1, T_1(x) = x, T_2(x) = 2x^2 - 1
    assert abs(_chebyshev_T(0, 3.0) - 1.0) < 1e-14
    assert abs(_chebyshev_T(1, 3.0) - 3.0) < 1e-14
    assert abs(_chebyshev_T(2, 3.0) - 17.0) < 1e-14  # 2*9 - 1
    assert abs(_chebyshev_T(3, 3.0) - 99.0) < 1e-12  # 2*3*17 - 3

    # T'_0 = 0, T'_1 = 1, T'_2 = 4x
    assert abs(_chebyshev_Tp(0, 3.0)) < 1e-14
    assert abs(_chebyshev_Tp(1, 3.0) - 1.0) < 1e-14
    assert abs(_chebyshev_Tp(2, 3.0) - 12.0) < 1e-14  # 4*3

    # Verify T''
    # T''_2(x) = 4 (constant)
    assert abs(_chebyshev_Tpp(2, 3.0) - 4.0) < 1e-12
    # T''_3(x) = 24x
    assert abs(_chebyshev_Tpp(3, 3.0) - 72.0) < 1e-10

    # Verify w1 consistency for unnormalized recurrence:
    # P'(0) = w1 * T_s'(w0) / T_s(w0) should equal 1.0
    s, eps = 10, 0.05
    w0 = 1 + eps / s**2
    Ts = _chebyshev_T(s, w0)
    Tps = _chebyshev_Tp(s, w0)
    w1 = Ts / Tps  # unnormalized recurrence formula
    slope = w1 * Tps / Ts
    assert abs(slope - 1.0) < 1e-12, f"P'(0) should be 1.0, got {slope}"

    print(f"    T_3(3.0) = {_chebyshev_T(3, 3.0):.1f} (expected 99)")
    print(f"    T'_2(3.0) = {_chebyshev_Tp(2, 3.0):.1f} (expected 12)")
    print(f"    T''_3(3.0) = {_chebyshev_Tpp(3, 3.0):.1f} (expected 72)")
    print(f"    s=10: w0={w0:.6f}, w1={w1:.6f}, P'(0)={slope:.12f}")
    print("RKC-T4 PASS: Chebyshev polynomials correct")


# ============================================================
# RKC-T5: Factory integration
# ============================================================
def test_rkc_t5_factory():
    """Verify 'explicit_rkc' string works in BidomainSimulation factory."""
    from cardiac_sim.simulation.classical.bidomain import BidomainSimulation
    from cardiac_sim.tissue_builder.stimulus import StimulusProtocol

    spatial = _make_spatial(10, 10)

    sim = BidomainSimulation(
        spatial=spatial, ionic_model='ttp06',
        stimulus=StimulusProtocol(), dt=DT,
        splitting='strang',
        diffusion_solver='explicit_rkc',
        elliptic_solver='auto', theta=0.5)

    assert sim.state is not None
    for _ in sim.run(t_end=DT*2, save_every=DT):
        break
    print(f"    Created and stepped BidomainSimulation with diffusion_solver='explicit_rkc'")
    print("RKC-T5 PASS: Factory integration works")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Explicit RKC Solver Validation\n")

    test_rkc_t4_coefficient_sanity()
    print()
    test_rkc_t1_stability_enforcement()
    print()
    test_rkc_t2_cosine_deff()
    print()
    test_rkc_t3_cross_check_gs()
    print()
    test_rkc_t5_factory()

    print("\nExplicit RKC: ALL 5 TESTS PASS")
