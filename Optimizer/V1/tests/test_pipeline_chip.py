"""Phase 3 — chip-fit orchestration smoke test.

Exercises targets -> cc_runner CV -> Tier-1 record end-to-end with a tiny budget
(smoke=True skips the BayesOpt cell fit; 2-point secant; TTP06 + near-natural CV
targets for speed/robustness). The FULL fit (run_chip_fit.main) is gated.
"""
import math

import torch

from tuner.config import TuningConfig, TuningTargets
from run_chip_fit import fit_chip_baseline

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_chip_fit_smoke(tmp_path):
    cfg = TuningConfig(ionic_model="ttp06", device=DEV, tier=2,
                       dx_cm=0.01, cable_length_cm=0.8, dt=0.02,
                       stim_amplitude=-52.0, stim_start=1.0, engine="monodomain",
                       domain_mm=16.0, dx_mm=0.1)
    # near-natural TTP06 CV so the warm-start D stays moderate (fast + robust)
    tgt = TuningTargets(cv_longitudinal=40.0, cv_transverse=25.0,
                        apd_90=300.0, dvdt_max=110.0, dvdt_max_upper=120.0)
    rec = fit_chip_baseline("nrvm", cfg, targets=tgt, smoke=True, n_secant=2,
                            presets_dir=str(tmp_path), t_end=40.0)

    assert rec["baseline"] == "nrvm"
    assert rec["ionic_model"] == "ttp06"
    mono = rec["tissue"]["monodomain"]
    assert math.isfinite(mono["D_long"]) and mono["D_long"] > 0
    assert math.isfinite(mono["D_trans"]) and mono["D_trans"] > 0
    assert (tmp_path / "chip_nrvm.json").exists()
