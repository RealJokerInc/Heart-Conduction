"""
Optimizer V1 — cross-engine validation + LBM recalibration (Phase 4).

validate(): run a record's (theta_ionic, D_long/D_trans) on monodomain, bidomain,
and LBM via cc_runner; report per-engine CV + deltas vs monodomain. Expect
mono↔bidomain small (~6%) and mono↔LBM ~35% raw (numerical dispersion).

recalibrate_lbm(): per-axis secant on D so LBM reproduces the record's physical
CV targets (≤2%), returning LBM tissue params incl. the MRT relaxation rates
s = 1/tau (from tau_tensor_from_D; the BGK 35% offset is removed by re-fitting D).

enrich_record(): write tissue['lbm'] + validation['cross_engine'] into a record.
"""
import math
from dataclasses import replace

from cardiac_core._lbm.diffusion import tau_tensor_from_D

from .cc_runner import run_1d_cable, fit_D_for_cv


def _pct(ref, other):
    if not (math.isfinite(ref) and math.isfinite(other) and ref != 0):
        return float("nan")
    return abs(ref - other) / abs(ref) * 100.0


def validate(record, base_config, *, engines=("monodomain", "bidomain", "lbm"),
             t_end=None):
    """Run the record's params on each engine; report CV + deltas vs monodomain."""
    theta = record["theta_ionic"] or None
    D_long = record["tissue"]["monodomain"]["D_long"]
    D_trans = record["tissue"]["monodomain"]["D_trans"]

    per = {}
    for eng in engines:
        cfg = replace(base_config, engine=eng, ionic_model=record["ionic_model"])
        per[eng] = {
            "cv_long": run_1d_cable(theta, D_long, cfg, t_end=t_end),
            "cv_trans": run_1d_cable(theta, D_trans, cfg, t_end=t_end),
        }

    deltas = {}
    if "monodomain" in per:
        m = per["monodomain"]
        for eng, cv in per.items():
            if eng == "monodomain":
                continue
            deltas[f"mono_vs_{eng}_long_pct"] = _pct(m["cv_long"], cv["cv_long"])
            deltas[f"mono_vs_{eng}_trans_pct"] = _pct(m["cv_trans"], cv["cv_trans"])
    return {"per_engine": per, "deltas": deltas}


def recalibrate_lbm(record, base_config, *, n=8, t_end=None, cs2=1.0 / 3.0):
    """Per-axis secant on D so LBM CV matches the record's targets; LBM tissue params."""
    cfg = replace(base_config, engine="lbm", ionic_model=record["ionic_model"])
    theta = record["theta_ionic"] or None
    tgt = record["targets"]

    D_long, cvL = fit_D_for_cv(theta, tgt["cv_longitudinal"], cfg, n=n, t_end=t_end)
    D_trans, cvT = fit_D_for_cv(theta, tgt["cv_transverse"], cfg, n=n, t_end=t_end)

    txx, tyy, _ = tau_tensor_from_D(D_long, D_trans, 0.0, cfg.dx_cm, cfg.dt_lbm, cs2)
    return {
        "D_long": D_long, "D_trans": D_trans, "collision": "mrt",
        "s_jx": 1.0 / txx, "s_jy": 1.0 / tyy,
        "dx_mm": cfg.dx_mm, "dt_ms": cfg.dt_lbm,
        "cv_long": cvL, "cv_trans": cvT,
    }


def enrich_record(record, base_config, *,
                  engines=("monodomain", "bidomain", "lbm"), n=8, t_end=None):
    """Add tissue['lbm'] (recalibrated) + validation['cross_engine'] to a record (in place)."""
    val = validate(record, base_config, engines=engines, t_end=t_end)
    record["tissue"]["lbm"] = recalibrate_lbm(record, base_config, n=n, t_end=t_end)
    record.setdefault("validation", {})["cross_engine"] = val["deltas"]
    return record
