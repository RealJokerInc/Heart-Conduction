"""Regression tests for the audit-driven solver fixes (solver-hardening Steps 1 & opt-in).

- Step 1: SolverConvergenceWarning fires on non-convergence; mono Chebyshev preconditioned
  bounds (07-02 M1) — accurate at high diffusion-number.
- Opt-in: bidomain pcg_spectral falls back to plain PCG on a mixed per-axis BC (Lane C2 HIGH);
  IMEX-SBDF2 runs with the 2nd-order coupling extrapolation.
"""

import warnings

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float64)

from cardiac_core.mesh.structured import StructuredGrid
from cardiac_core.mesh.boundary import BoundarySpec, Edge
from cardiac_core._bidomain.tissue.conductivity import BidomainConductivity
from cardiac_core._bidomain.simulation.classical.discretization.fdm import BidomainFDMDiscretization
from cardiac_core._bidomain.simulation.classical.bidomain import (
    BidomainSimulation, _build_linear_solver, _build_diffusion_solver)
from cardiac_core._monodomain.simulation.classical.solver.diffusion_time_stepping.linear_solver.pcg import PCGSolver
from cardiac_core._monodomain.simulation.classical.solver.diffusion_time_stepping.linear_solver.chebyshev import ChebyshevSolver
from cardiac_core._monodomain.simulation.classical.solver.diffusion_time_stepping.linear_solver.base import SolverConvergenceWarning


def _spd(n, cond_scale=1.0, seed=0):
    torch.manual_seed(seed)
    M = torch.randn(n, n)
    return (M @ M.T + cond_scale * n * torch.eye(n)).to_sparse()


# --- Step 1: non-convergence signal ----------------------------------------

def test_pcg_warns_on_nonconvergence():
    A, b = _spd(60), torch.randn(60)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        PCGSolver(max_iters=2, tol=1e-12).solve(A, b)
    assert any(issubclass(x.category, SolverConvergenceWarning) for x in w)


def test_pcg_silent_when_converged():
    A, b = _spd(60), torch.randn(60)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        PCGSolver(max_iters=500, tol=1e-8).solve(A, b)
    assert not any(issubclass(x.category, SolverConvergenceWarning) for x in w)


def test_nonconvergence_warns_at_most_once_per_run():
    # Regression (audit R1, Lane A): a chronically under-converging elliptic tier (default
    # declarative bidomain hits pcg_spectral breakdown at ~1e-4) must warn ONCE per run, not
    # once per step (437-warning flood otherwise). Also checks the 'breakdown' reason label.
    import cardiac_core as cc
    g = cc.Grid(41, 41, 0.02)
    cond = cc.ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
    stim = {"region": lambda x, y: x < 0.06, "start_time": 1.0, "duration": 2.0, "amplitude": -52.0}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cc.bidomain(g, "ttp06", cond, stim, dt=0.05).run(t_end=2.0, save_every=0.5)  # ~40 steps
    # NOTE: match by category NAME — the mono and bidomain trees each define their own
    # SolverConvergenceWarning class (copy-vendoring), so a bidomain run emits the bidomain
    # class; both subclass UserWarning. (A single shared class awaits the Phase-4 dedup.)
    hits = [x for x in w if x.category.__name__ == "SolverConvergenceWarning"]
    assert len(hits) == 1, f"expected 1 non-convergence warning per run, got {len(hits)}"
    assert "breakdown" in str(hits[0].message)   # correct reason, not the old 'max_iters' mislabel


def test_nonconvergence_warning_rearms_each_run():
    # Regression (audit R2): warn-once must be per-RUN, not per-solver-lifetime. A reused sim
    # (restitution / S1-S2 pattern) must re-warn on every run; the flag is reset at run start
    # (_reset_solver_diagnostics). Without the reset, run 2+ would silently under-solve.
    import cardiac_core as cc
    g = cc.Grid(41, 41, 0.02)
    cond = cc.ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
    stim = {"region": lambda x, y: x < 0.06, "start_time": 1.0, "duration": 2.0, "amplitude": -52.0}
    sim = cc.bidomain(g, "ttp06", cond, stim, dt=0.05)
    counts = []
    for _ in range(3):
        sim.reset()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sim.run(t_end=2.0, save_every=0.5)
        counts.append(sum(1 for x in w if x.category.__name__ == "SolverConvergenceWarning"))
    assert counts == [1, 1, 1], f"warn-once must re-arm per run, got {counts}"


def test_convergence_warning_escalates_to_error():
    A, b = _spd(60), torch.randn(60)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=SolverConvergenceWarning)
        with pytest.raises(SolverConvergenceWarning):
            PCGSolver(max_iters=2, tol=1e-12).solve(A, b)


# --- Step 1: mono Chebyshev preconditioned bounds (07-02 M1) ----------------

def test_chebyshev_jacobi_accurate_high_diffusion_number():
    # A REAL FDM diffusion operator at a high diffusion number (large dt / fine dx), the regime
    # where the OLD raw-A Gershgorin bounds gave up to ~46% silent error. The preconditioned
    # bounds (07-02 M1 fix) must recover machine precision, matching plain PCG.
    import cardiac_core as cc
    g = cc.Grid(41, 41, 0.01)
    cond = cc.ConductivityConfig.bidomain(1.74, 6.25, chi=1.0)   # chi=1 -> large diffusion number
    stim = {"region": lambda x, y: x < 0.02, "start_time": 1.0, "duration": 2.0, "amplitude": -52.0}
    sim = cc.monodomain(g, "ttp06", cond, stim, dt=1.0, diffusion_solver="bdf1")
    A = sim._engine.splitting.diffusion_solver.ops.A_lhs
    n = A.shape[0]
    torch.manual_seed(1)
    b = torch.randn(n)
    x_true = torch.linalg.solve(A.to_dense(), b)
    x = ChebyshevSolver(max_iters=2000, tol=1e-8, use_jacobi_precond=True).solve(A, b)
    rel = float((x - x_true).norm() / x_true.norm())
    # The old raw-A bounds gave ~46% error in this regime; the preconditioned bounds converge.
    # (Threshold 1e-3 cleanly separates fixed ~2e-5 from the broken ~0.46; Gershgorin bounds
    # are conservative so it doesn't reach machine precision here.)
    assert rel < 1e-3, f"Chebyshev(Jacobi) rel error {rel:.2e} — preconditioned bounds regressed (M1)"


# --- Opt-in: bidomain pcg_spectral mixed-BC fallback (Lane C2 HIGH) ---------

def _mixed_bc_spatial():
    grid = StructuredGrid(Nx=24, Ny=20, Lx=1.0, Ly=1.0,
                          boundary_spec=BoundarySpec.bath_coupled_edges([Edge.TOP, Edge.BOTTOM]))
    theta = torch.full((24, 20), 0.6)
    cond = BidomainConductivity(D_i_fiber=0.0020, D_i_cross=0.0005,
                                D_e_fiber=0.0060, D_e_cross=0.0025, theta=theta)
    return BidomainFDMDiscretization(grid, cond)


def test_pcg_spectral_mixed_bc_falls_back_and_converges():
    spatial = _mixed_bc_spatial()
    # auto-selects pcg_spectral; the fix must warn + fall back to plain PCG and converge.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        solver = _build_linear_solver("pcg_spectral", spatial)
    assert any("MIXED per-axis" in str(x.message) for x in w)
    A = spatial.get_elliptic_operator()
    n = A.shape[0]
    torch.manual_seed(2)
    b = spatial.apply_L_i(torch.randn(n))   # a compatible RHS
    x = solver.solve(A, b)
    rel_resid = float((A.to_dense() @ x - b).norm() / b.norm())
    assert rel_resid < 1e-5, f"mixed-BC elliptic solve did not converge (resid {rel_resid:.2e})"


# --- Opt-in: IMEX-SBDF2 runs with the coupling extrapolation ----------------

def test_imex_sbdf2_runs_and_is_stable():
    # Build the (non-public) imex solver directly on a factory-made bidomain and step it
    # (pure diffusion, no ionic). Result must stay finite and track the default decoupled
    # solver reasonably (both integrate the same diffusion problem).
    import math
    import cardiac_core as cc
    g = cc.Grid(25, 25, 0.04)
    condcfg = cc.ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
    stim = {"region": lambda x, y: x < 0.04, "start_time": 1e9, "duration": 1.0, "amplitude": 0.0}

    def run(which):
        sim = cc.bidomain(g, "ttp06", condcfg, stim, dt=0.02)
        eng = sim._engine
        spatial = eng.splitting.diffusion_solver._spatial
        para = _build_linear_solver("pcg", spatial)
        ellip = _build_linear_solver(eng._elliptic_solver_name, spatial)
        ds = _build_diffusion_solver(which, spatial, 0.02, para, ellip, 0.5)
        st = eng.state
        xx = spatial.grid._xx
        st.Vm.copy_(spatial.grid.grid_to_flat(30.0 * torch.cos(math.pi * xx / float(xx.max())) - 20.0))
        st.phi_e.zero_()
        for _ in range(50):
            ds.step(st, 0.02)
        assert torch.isfinite(st.Vm).all()
        return st.Vm.clone()

    a, b = run("decoupled"), run("imex_sbdf2")
    rel = float((b - a).norm() / a.norm())
    assert rel < 0.05, f"imex vs decoupled diverged: {rel:.3f}"
