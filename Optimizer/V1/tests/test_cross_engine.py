"""Phase 4 — cross-engine validation + LBM recalibration (smoke).

Runs a tiny record across mono/bidomain/lbm and recalibrates LBM. TTP06 +
near-natural CV targets + small cable for speed. Heavy convergence is gated.
"""
import math

import torch

from tuner.config import TuningConfig
from tuner.presets import make_record
from tuner.cross_engine import validate, recalibrate_lbm, enrich_record

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _cfg():
    return TuningConfig(ionic_model="ttp06", device=DEV, dx_cm=0.01,
                        cable_length_cm=0.6, dt=0.02, dt_lbm=0.01,
                        stim_amplitude=-52.0, stim_start=1.0,
                        domain_mm=16.0, dx_mm=0.1)


def _record():
    return make_record(
        name="chip_smoke", baseline="nrvm", ionic_model="ttp06",
        theta_ionic={},                       # identity (baseline TTP06)
        tissue={"monodomain": {"D_long": 0.001, "D_trans": 0.0004, "dt_ms": 0.02}},
        targets={"cv_longitudinal": 40.0, "cv_transverse": 25.0,
                 "apd_90": 300.0, "dvdt_max": 110.0},
        dx_mm=0.1,                            # coarse smoke grid (not the resolved default)
    )


def test_validate_three_engines():
    val = validate(_record(), _cfg(), t_end=30.0)
    per = val["per_engine"]
    for eng in ("monodomain", "bidomain", "lbm"):
        assert eng in per
        assert per[eng]["cv_long"] > 0 and math.isfinite(per[eng]["cv_long"]), (eng, per[eng])
    # deltas vs monodomain present + finite
    assert math.isfinite(val["deltas"]["mono_vs_bidomain_long_pct"])
    assert math.isfinite(val["deltas"]["mono_vs_lbm_long_pct"])


def test_recalibrate_lbm_and_enrich():
    rec = _record()
    lbm_t = recalibrate_lbm(rec, _cfg(), n=2, t_end=30.0)
    assert lbm_t["collision"] == "mrt"
    assert math.isfinite(lbm_t["D_long"]) and lbm_t["D_long"] > 0
    assert math.isfinite(lbm_t["s_jx"]) and math.isfinite(lbm_t["s_jy"])
    # enrich writes lbm tissue + cross_engine deltas into the record
    enrich_record(rec, _cfg(), engines=("monodomain", "lbm"), n=2, t_end=30.0)
    assert "lbm" in rec["tissue"]
    assert "cross_engine" in rec["validation"]
