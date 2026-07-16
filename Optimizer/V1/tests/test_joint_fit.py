"""
Tests for the constrained-scalarization joint fit (PLAN Step 3.3).

Synthetic oracle (no sims): CV = scale·√(D·g_Na), APD = 350·g_CaL, dV/dt = 130·g_Na,
blocked (→NaN) where r*/dx < 3. Two contracts:
  - a KNOWN-FEASIBLE target is hit within tol (returns a JointFitResult);
  - an INFEASIBLE setup (coarse dx → nothing resolves) returns an InfeasReport naming
    the binding lock, NOT a fake fit (architecture §7).
"""

import math

from tests.test_emulator import synth_evaluator     # reuse the oracle


def _min_axes():
    """Minimal noise-free decision space (g_Na, g_CaL, D_long, D_trans) — exercises the
    full fit machinery without the CV-irrelevant conductances that make a small-sample
    GP noisy (the real high-dim fit uses a reduced CV feature set; open-Q8)."""
    from tuner.decision_space import Axis
    return [
        Axis('g_Na', 'cond', (0.5, 2.0)),
        Axis('g_CaL', 'cond', (0.3, 2.0)),
        Axis('D_long', 'diffusion', (5e-5, 1e-3)),
        Axis('D_trans', 'diffusion', (2.5e-5, 5e-4)),
    ]


def _targets(cv_l, cv_t, apd=350.0):
    from tuner.config import TuningTargets
    return TuningTargets(cv_longitudinal=cv_l, cv_transverse=cv_t, apd_90=apd,
                         dvdt_max=110.0, dvdt_max_upper=130.0)


def _cfg(dx_mm):
    from tuner.config import TuningConfig
    return TuningConfig(device='cpu', ionic_model='mhas13', tier=1, dx_mm=dx_mm)


def test_feasible_converges():
    """A known-feasible synthetic target (CV_L=8, CV_T=4, APD=350) with a fat interior
    feasible region (g_Na∈~0.5–1.0) is hit within tol."""
    from tuner.joint_fit import refine_joint_cc, JointFitResult
    axes = _min_axes()
    ev = synth_evaluator(axes, dx_cm=0.002)           # fine dx → resolvable
    res = refine_joint_cc(axes, ev, _targets(8.0, 4.0), config=_cfg(0.02),
                          n_training=64, n_candidates=4000, cv_tol=0.15,
                          dvdt_band=(20.0, 130.0), seed=7)

    assert isinstance(res, JointFitResult), getattr(res, 'detail', res)
    assert abs(res.achieved['cv_l'] - 8.0) / 8.0 < 0.15
    assert abs(res.achieved['cv_t'] - 4.0) / 4.0 < 0.15
    assert res.achieved['rstar_over_dx_t'] >= 3.0     # resolved, not grid-fudged


def test_reports_infeasible():
    """Coarse dx (0.5 mm) → nothing reaches r*/dx≥3 → InfeasReport naming a lock,
    not a fabricated fit."""
    from tuner.joint_fit import refine_joint_cc, InfeasReport
    axes = _min_axes()
    ev = synth_evaluator(axes, dx_cm=0.05)            # coarse dx → all blocked
    res = refine_joint_cc(axes, ev, _targets(5.2, 2.6), config=_cfg(0.5),
                          n_training=32, seed=3)

    assert isinstance(res, InfeasReport)
    assert isinstance(res.binding_lock, str) and res.binding_lock
