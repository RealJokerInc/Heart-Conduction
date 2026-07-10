"""
Tests for cell_runner_cc.py — cardiac_core-backed single-cell AP eval (P-1).

The runner drives a small uniform strip via cardiac_core's run_monodomain (the
hook-based Rush-Larsen path the tissue-CV path uses), paced to steady state, so a
future Na-kinetics axis is identifiable from BOTH dV/dt (cell) and CV (tissue).

Parity finding (Step 0.1, 2026-07-10): the V5.4-vs-cardiac_core delta is PACING
HISTORY, not a formulation delta — APD Δ 9.35% @6 beats → 1.80% @12 → 0.67% @20
(dV/dt 8.33%→1.24%→0.76%). V_rest matches to 0.06% throughout. So parity ≤1% holds
once paced to steady state (n_beats≈20). All sim-driven tests are marked `slow`.
"""

import pytest
import torch

_STEADY_BEATS = 20     # beats to reach APD/dV/dt parity ≤1% vs V5.4 (measured)


def _cfg(**kw):
    from tuner.config import TuningConfig
    base = dict(device='cpu', ionic_model='mhas13', tier=1,
                n_beats=6, pacing_cl=1000.0, dt_cell=0.2)
    base.update(kw)
    return TuningConfig(**base)


@pytest.mark.slow
class TestCellRunnerCC:

    def test_baseline_runs(self):
        """Identity theta → converged AP with sane biomarkers."""
        from tuner.config import get_param_names
        from tuner.cell_runner_cc import run_single_cell_cc
        cfg = _cfg()
        theta = torch.ones(len(get_param_names(1)), dtype=torch.float64)
        r = run_single_cell_cc(theta, cfg)
        assert r.converged
        assert r.apd90 is not None
        assert -95.0 < r.v_rest < -70.0        # MHAS13 RMP ≈ -83.7 mV
        assert r.v_peak > 0.0

    def test_parity_vs_v54(self):
        """APD90/dVdt/Vrest/Vpeak within 1% of the V5.4 cell_runner (identity θ),
        paced to steady state. Guards that the cardiac_core hook path reproduces
        the V5.4 model (faithful P-1 port)."""
        from tuner.config import get_param_names
        from tuner.cell_runner import run_single_cell
        from tuner.cell_runner_cc import run_single_cell_cc
        cfg = _cfg(n_beats=_STEADY_BEATS)
        theta = torch.ones(len(get_param_names(1)), dtype=torch.float64)
        r54 = run_single_cell(theta, cfg)
        rcc = run_single_cell_cc(theta, cfg)

        assert abs(rcc.v_rest - r54.v_rest) / abs(r54.v_rest) < 0.01
        assert abs(rcc.apd90 - r54.apd90) / r54.apd90 < 0.01
        assert abs(rcc.dvdt_max - r54.dvdt_max) / r54.dvdt_max < 0.01
        assert abs(rcc.v_peak - r54.v_peak) / abs(r54.v_peak) < 0.01

    def test_scaling_moves_biomarkers(self):
        """g_CaL×1.5 raises APD; g_Na×0.5 lowers dV/dt (monotone, sane)."""
        from tuner.config import get_param_names
        from tuner.cell_runner_cc import run_single_cell_cc
        cfg = _cfg()
        names = get_param_names(1)
        base = torch.ones(len(names), dtype=torch.float64)
        r_base = run_single_cell_cc(base, cfg)

        th_cal = base.clone()
        th_cal[names.index('g_CaL')] = 1.5
        r_cal = run_single_cell_cc(th_cal, cfg)
        assert r_cal.apd90 > r_base.apd90            # more Ca influx → longer plateau

        th_na = base.clone()
        th_na[names.index('g_Na')] = 0.5
        r_na = run_single_cell_cc(th_na, cfg)
        assert r_na.dvdt_max < r_base.dvdt_max       # less Na → slower upstroke
