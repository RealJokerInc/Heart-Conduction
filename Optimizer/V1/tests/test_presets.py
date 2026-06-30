"""Phase 3/5 — tuned-parameter record storage (round-trip). Fast, no sim."""
import os
import tempfile

import pytest

from tuner.presets import make_record, save_record, load_record, list_records, to_sim_kwargs


def _record():
    return make_record(
        name="chip_nrvm", baseline="nrvm",
        theta_ionic={"g_Na": 0.83, "g_CaL": 1.12, "kNaCa": 0.9},
        tissue={
            "monodomain": {"D_long": 4.2e-4, "D_trans": 1.05e-4, "dt_ms": 0.02},
            "lbm": {"D_long": 4.2e-4, "D_trans": 1.05e-4, "collision": "mrt",
                    "s_jx": 1.8, "s_jy": 1.95, "dx_mm": 0.1, "dt_ms": 0.05},
        },
        targets={"cv_longitudinal": 9.33, "cv_transverse": 4.665, "apd_90": 350},
        validation={"cv_long": 9.30, "tissue_apd": 348},
        provenance={"date": "2026-06-30", "tuner_version": "V1"},
    )


def test_record_roundtrip():
    rec = _record()
    with tempfile.TemporaryDirectory() as d:
        path = save_record(rec, presets_dir=d)
        assert os.path.isfile(path)
        loaded = load_record("chip_nrvm", presets_dir=d)
        assert loaded == rec
        assert list_records(presets_dir=d) == ["chip_nrvm"]


def test_to_sim_kwargs_per_engine():
    rec = _record()
    mono = to_sim_kwargs(rec, "monodomain")
    assert mono["D_long"] == 4.2e-4 and mono["D_trans"] == 1.05e-4
    assert mono["ionic_model"] == "mhas13"
    assert mono["theta_ionic"]["g_Na"] == 0.83
    lbm = to_sim_kwargs(rec, "lbm")
    assert lbm["dt"] == 0.05 and lbm["dx_mm"] == 0.1
    with pytest.raises(KeyError):
        to_sim_kwargs(rec, "bidomain")     # not fitted in this record


def test_make_record_rejects_bad_baseline():
    with pytest.raises(ValueError):
        make_record("x", "adult", {}, {}, {})
