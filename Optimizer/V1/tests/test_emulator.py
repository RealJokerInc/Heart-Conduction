"""
Tests for the joint_fit GP emulator (PLAN Step 3.2).

Uses a SYNTHETIC oracle (no sims): CV = scale·√(D·g_Na), blocked (→NaN) where
r*/dx < 3. Verifies the block region is MASKED (NaN never enters a CV GP target — the
architecture §7 fix for the joint_refiner 50.0-penalty bug) and CV GPs train on the
feasible subset only.
"""

import math

import torch


def synth_evaluator(axes, *, dx_cm=0.008, k=3.0, scale=600.0):
    # Default dx=0.008 cm straddles the r*/dx=3 block edge over the box → a real MIX
    # of feasible/blocked points (test_joint_fit overrides dx_cm per case).
    from tuner.joint_fit import PointEval

    def ev(vector):
        d = {ax.name: float(v) for ax, v in zip(axes, vector)}
        gNa = d.get('g_Na', 1.0)
        gCaL = d.get('g_CaL', 1.0)
        Dl = d.get('D_long', 1e-4)
        Dt = d.get('D_trans', 5e-5)
        tau_m = d.get('tau_m_scale', 1.0)

        def cv(D):
            return scale * math.sqrt(max(D, 0.0) * max(gNa, 1e-6))

        cvl, cvt = cv(Dl), cv(Dt)
        rl = (Dl / (cvl / 1000.0)) / dx_cm if cvl > 0 else float('nan')
        rt = (Dt / (cvt / 1000.0)) / dx_cm if cvt > 0 else float('nan')
        if not (math.isfinite(rl) and rl >= k):
            cvl = float('nan')                       # blocked → NaN (the cliff)
        if not (math.isfinite(rt) and rt >= k):
            cvt = float('nan')
        feasible = math.isfinite(cvl) and math.isfinite(cvt)
        return PointEval(list(vector), 350.0 * gCaL, 130.0 * gNa / tau_m,
                         cvl, cvt, rl, rt, feasible)

    return ev


def test_nan_guard():
    """Blocked (NaN) points must not poison the CV GPs (no NaN in GP targets)."""
    from tuner.decision_space import build_axes
    from tuner.joint_fit import build_training_set, build_emulator
    axes = build_axes(tier=1, include_kinetics=False)
    X, evals = build_training_set(axes, synth_evaluator(axes), n=32, seed=1)
    assert any(not e.feasible_sim for e in evals)     # some blocked points exist

    emu = build_emulator(X, evals)                    # must not raise (isfinite guard)
    for key in ('cvl', 'cvt'):
        if emu[key] is not None:
            assert torch.isfinite(emu[key].train_targets).all()


def test_feasible_only_training():
    """CV GPs train on the FEASIBLE subset; masked points are excluded."""
    from tuner.decision_space import build_axes
    from tuner.joint_fit import build_training_set, build_emulator
    axes = build_axes(tier=1, include_kinetics=False)
    X, evals = build_training_set(axes, synth_evaluator(axes), n=32, seed=2)
    n_feas = sum(e.feasible_sim for e in evals)

    emu = build_emulator(X, evals)
    assert emu['n_feasible'] == n_feas
    assert 0 < n_feas < len(evals)                    # a real mask (some in, some out)
    if emu['cvl'] is not None:
        assert emu['cvl'].train_inputs[0].shape[0] == n_feas
