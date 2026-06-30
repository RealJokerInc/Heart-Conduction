"""Phase 1 — cc_runner (cardiac_core-backed CV measurement).

Verifies the functional-API runner: finite CV, CV scales with D (CV ∝ √D),
LBM path runs, and per-axis anisotropy (CV_long > CV_trans). Uses TTP06 + a
fast-propagating D so the test is quick; chi=1.0 + dx=0.01 per the Phase 0
convention. Covers PLAN Phase 1 Step 1.1.
"""
import torch

from tuner.config import TuningConfig
from tuner.cc_runner import run_1d_cable, run_2d_tissue

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _cfg(engine="monodomain"):
    return TuningConfig(ionic_model="ttp06", device=DEV, tier=2,
                        dx_cm=0.01, cable_length_cm=0.8, dt=0.02, dt_lbm=0.01,
                        stim_amplitude=-52.0, stim_duration=2.0, stim_start=1.0,
                        engine=engine)


def test_cv_finite_and_scales_with_D():
    """Monodomain CV is physiological and increases with D (CV ∝ √D)."""
    cfg = _cfg("monodomain")
    cv_hi = run_1d_cable(None, 0.001, cfg, t_end=45.0, save_every=0.5)
    cv_lo = run_1d_cable(None, 0.0005, cfg, t_end=70.0, save_every=0.5)
    assert 10.0 < cv_hi < 120.0, cv_hi
    assert cv_lo < cv_hi, (cv_lo, cv_hi)            # lower D -> slower
    assert 1.2 < cv_hi / cv_lo < 1.7, cv_hi / cv_lo  # ~sqrt(2)=1.41


def test_lbm_path_runs():
    """LBM CV path returns a finite, positive CV (may differ ~35% from FDM)."""
    cfg = _cfg("lbm")
    cv = run_1d_cable(None, 0.001, cfg, t_end=45.0, save_every=0.5)
    assert cv == cv and cv > 0.0, cv                # finite, positive


def test_2d_tissue_anisotropic():
    """run_2d_tissue: CV_long (D_long) > CV_trans (D_trans)."""
    cfg = _cfg("monodomain")
    res = run_2d_tissue(None, D_long=0.001, D_trans=0.00025, config=cfg,
                        t_end=90.0, save_every=0.5)
    assert res["cv_long"] > res["cv_trans"], res
    ratio = res["cv_long"] / res["cv_trans"]
    assert 1.5 < ratio < 2.6, ratio                 # ~sqrt(4)=2
