"""
Tests for the Na-kinetic axes on MHAS13 (PLAN Step 2.2 / P1.5).

The multipliers (tau_m/h/j_scale, v_half_shift) live on the MHAS13 INSTANCE and are
applied in the gate HOOKS (compute_gate_*), NOT step() — because the tissue
Rush-Larsen solver drives the model via the hooks. The guard test is
test_tau_m_MOVES_cv: if scaling were placed where CV can't see it (e.g. step()), CV
would not respond and that test would FAIL.
"""

import math

import pytest
import torch


def _chip_cfg(dx_cm=0.01):
    from tuner.config import TuningConfig
    return TuningConfig(device='cpu', ionic_model='mhas13', tier=1,
                        dx_cm=dx_cm, cable_length_cm=0.5, dt=0.02,
                        stim_amplitude=-40.0, stim_start=1.0, engine='monodomain',
                        n_beats=4, pacing_cl=1000.0, dt_cell=0.2)


def test_identity_parity():
    """scales=1, shift=0 → the Na entries of compute_gate_* are BITWISE the raw INa_*
    functions (nothing changes unless a tuner sets the knobs)."""
    from cardiac_core.ionic import MHAS13Model
    from cardiac_core.ionic.phas13.gating import (
        INa_m_inf, INa_h_inf, INa_j_inf, INa_m_tau, INa_h_tau, INa_j_tau,
    )
    m = MHAS13Model(device='cpu')
    V = torch.linspace(-90.0, 40.0, 60, dtype=torch.float64)
    S = m.get_initial_state().unsqueeze(0).expand(60, -1).clone()

    inf = m.compute_gate_steady_states(V, S)
    tau = m.compute_gate_time_constants(V, S)

    assert torch.equal(inf[:, 0], INa_m_inf(V))
    assert torch.equal(inf[:, 1], INa_h_inf(V))
    assert torch.equal(inf[:, 2], INa_j_inf(V))
    assert torch.equal(tau[:, 0], INa_m_tau(V))
    assert torch.equal(tau[:, 1], INa_h_tau(V))
    assert torch.equal(tau[:, 2], INa_j_tau(V))


def test_phas13_untouched():
    """The MHAS13 kinetic knobs do NOT exist on / affect PHAS13 (shared-module safety)."""
    from cardiac_core.ionic import PHAS13Model, MHAS13Model
    p = PHAS13Model(device='cpu')
    V = torch.linspace(-90.0, 40.0, 40, dtype=torch.float64)
    S = p.get_initial_state().unsqueeze(0).expand(40, -1).clone()

    tau_before = p.compute_gate_time_constants(V, S).clone()
    m = MHAS13Model(device='cpu')
    m.tau_m_scale = 3.0
    m.v_half_shift = 8.0
    tau_after = p.compute_gate_time_constants(V, S)

    assert torch.equal(tau_before, tau_after)
    assert not hasattr(p, 'tau_m_scale')


@pytest.mark.slow
def test_tau_m_MOVES_cv():
    """τ_m×2 changes tissue CV by a nonzero, measurable amount (guards against the
    'invisible to CV' failure — a CV delta of 0 FAILS)."""
    from cardiac_core.ionic import MHAS13Model
    from tuner.cc_runner import run_1d_cable
    cfg = _chip_cfg(dx_cm=0.01)
    D = 1e-4                                  # in the baseline propagating window

    m0 = MHAS13Model(device='cpu')
    cv0 = run_1d_cable(None, D, cfg, model=m0)
    m2 = MHAS13Model(device='cpu')
    m2.tau_m_scale = 2.0
    cv2 = run_1d_cable(None, D, cfg, model=m2)

    assert math.isfinite(cv0) and math.isfinite(cv2)
    assert abs(cv2 - cv0) > 0.1              # CV genuinely responds to τ_m


@pytest.mark.slow
def test_tau_m_decouples():
    """τ_m×3 shifts the dV/dt : CV ratio (both measured on the same model) — the
    decoupling the architecture predicts (conductance scaling can't do this): dV/dt
    (peak I_Na) is MORE sensitive to slower Na activation than CV (charge-to-sink), so
    the ratio moves. Finer CV sampling (save_every) removes 1 ms quantization noise."""
    from cardiac_core.ionic import MHAS13Model
    from tuner.cc_runner import run_1d_cable
    from tuner.cell_runner_cc import run_single_cell_cc
    cfg = _chip_cfg(dx_cm=0.01)
    D = 1e-4

    m0 = MHAS13Model(device='cpu')
    m3 = MHAS13Model(device='cpu')
    m3.tau_m_scale = 3.0

    dvdt0 = run_single_cell_cc(None, cfg, model=m0, n_beats=4).dvdt_max
    dvdt3 = run_single_cell_cc(None, cfg, model=m3, n_beats=4).dvdt_max
    cv0 = run_1d_cable(None, D, cfg, model=m0, save_every=0.25)
    cv3 = run_1d_cable(None, D, cfg, model=m3, save_every=0.25)

    assert all(x is not None and math.isfinite(x) for x in (dvdt0, dvdt3, cv0, cv3))
    # dV/dt drops with slower activation, and NOT in lockstep with CV → ratio shifts.
    assert dvdt3 < dvdt0
    rel_dvdt = (dvdt0 - dvdt3) / dvdt0
    rel_cv = (cv0 - cv3) / cv0
    assert abs(rel_dvdt - rel_cv) > 0.005            # non-proportional → decoupled
