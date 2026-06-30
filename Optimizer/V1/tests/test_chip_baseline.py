"""Phase 5 — LBM planar-wave chip baseline (smoke).

Runs baseline_lbm on a tiny anisotropic chip (exercises the full MRT chip path
end-to-end: planar stim -> D2Q9-MRT run -> CV/λ -> Lab-preset export). Full 161²
chip run is gated (run_chip_baseline_lbm.main).
"""
import math

import torch

from tuner.presets import make_record
from run_chip_baseline_lbm import baseline_lbm

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_baseline_lbm_smoke(tmp_path):
    rec = make_record(
        "chip_smoke", "nrvm",
        {},                                      # identity theta (baseline TTP06)
        {"lbm": {"D_long": 0.001, "D_trans": 0.00025, "collision": "mrt",
                 "s_jx": 1.8, "s_jy": 1.95, "dx_mm": 0.1, "dt_ms": 0.01},
         "monodomain": {"D_long": 0.001, "D_trans": 0.00025, "dt_ms": 0.02}},
        {"cv_longitudinal": 40.0, "cv_transverse": 25.0, "apd_90": 300.0, "dvdt_max": 110.0},
        ionic_model="ttp06", domain_mm=2.0, dx_mm=0.1,
    )
    out = baseline_lbm(rec, domain_mm=2.0, dx_mm=0.1, t_end=20.0, save_every=0.5,
                       device=DEV, export=True, lab_dir=str(tmp_path))
    assert out["n_saves"] > 0
    assert isinstance(out["cv"], float)          # finite or NaN both acceptable for smoke
    assert math.isnan(out["wavelength_mm"]) or out["wavelength_mm"] > 0
    assert out["preset_path"] and (tmp_path / "chip_smoke.yaml").exists()
