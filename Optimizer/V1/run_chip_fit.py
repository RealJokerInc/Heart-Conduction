"""
Optimizer V1 — chip-EP fit orchestration (Phase 3).

For each baseline (NRVM, hiPSC): (1) cell fit -> theta_ionic [gated, full BayesOpt],
(2) dual-axis tissue fit -> D_long, D_trans via cc_runner secant on cardiac_core
monodomain, (3) save a Tier-1 record (presets.py).

Smoke mode (`smoke=True`) skips the BayesOpt cell fit (identity theta) and does a
2-point secant — it exercises the targets -> cc_runner CV -> record wire cheaply.
The FULL fit (`main()`, smoke=False) is a multi-hour GPU run — gated.

Run (GATED): conda run -n heart-conduction python Optimizer/V1/run_chip_fit.py
"""
import torch

from tuner.config import TuningConfig, TuningTargets, theta_to_dict
from tuner.cc_runner import fit_D_for_cv
from tuner.chip import PARKER, boundary_number
from tuner.presets import make_record, save_record, PRESETS_DIR


def select_best(cellres) -> torch.Tensor:
    """Pick the Pareto-front theta with the best (max) summed objective."""
    scores = cellres.pareto_Y.sum(dim=1)
    return cellres.pareto_X[int(scores.argmax())]


# NOTE: the secant `_fit_D_for_cv` that used to live here was DE-DUPLICATED into the
# single `cc_runner.fit_D_for_cv` (imported above), which now brackets DOWN on a
# non-propagating start (the old ×4-up-bump was a Known Failure) and returns
# (NaN, NaN) — never a fake D — when nothing propagates. This retired the block-
# masking garbage-D fallback that produced the `D=0.004, CV=nan` records.


def _cell_fit(config, targets, smoke):
    """Returns (theta_ionic_tensor_or_None, theta_dict)."""
    if smoke:
        return None, {}                           # identity (baseline model)
    from tuner.cell_fitter import fit_cell          # lazy: heavy (botorch)
    cellres = fit_cell(config, targets)
    theta_t = select_best(cellres)
    return theta_t, theta_to_dict(theta_t, config.tier)


def fit_chip_baseline(baseline, config, *, targets=None, smoke=False,
                      n_secant=8, presets_dir=None, t_end=None):
    """Fit one baseline; save + return a Tier-1 record."""
    targets = targets or PARKER[baseline]
    theta_ionic, theta_dict = _cell_fit(config, targets, smoke)

    n = 2 if smoke else n_secant
    D_long, cvL = fit_D_for_cv(theta_ionic, targets.cv_longitudinal, config, n=n, t_end=t_end)
    D_trans, cvT = fit_D_for_cv(theta_ionic, targets.cv_transverse, config, n=n, t_end=t_end)

    rec = make_record(
        name=f"chip_{baseline}", baseline=baseline, theta_ionic=theta_dict,
        tissue={"monodomain": {"D_long": D_long, "D_trans": D_trans, "dt_ms": config.dt}},
        targets={"cv_longitudinal": targets.cv_longitudinal,
                 "cv_transverse": targets.cv_transverse,
                 "apd_90": targets.apd_90, "dvdt_max": targets.dvdt_max},
        validation={"cv_long": cvL, "cv_trans": cvT},
        provenance={"tuner_version": "V1", "engine": config.engine, "smoke": smoke},
        ionic_model=config.ionic_model,
        domain_mm=config.domain_mm, dx_mm=config.dx_mm,
    )
    # Lateral boundary-speedup GUIDE (β=D·dt/dx², τ=0.5+3β) for the free dt knob —
    # not fit (no curvature metric yet); just records which wall-crescent regime
    # this (D, dt, dx) lands in, per axis. See chip.boundary_number.
    rec["boundary"] = {
        "dt_ms": config.dt, "dx_mm": config.dx_mm,
        "long": boundary_number(D_long, config.dt, config.dx_mm),
        "trans": boundary_number(D_trans, config.dt, config.dx_mm),
    }
    save_record(rec, presets_dir=presets_dir or PRESETS_DIR)
    return rec


def main():  # pragma: no cover — gated full run
    cfg = TuningConfig(
        ionic_model="mhas13", tier=2, engine="monodomain",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dx_cm=0.01, cable_length_cm=1.6, dt=0.02,
        stim_amplitude=-40.0, stim_start=1.0,
        domain_mm=16.0, dx_mm=0.1,
    )
    for baseline in ("nrvm", "hipsc"):
        rec = fit_chip_baseline(baseline, cfg, smoke=False, n_secant=8)
        print(baseline, rec["tissue"]["monodomain"], rec["validation"])
        b = rec["boundary"]   # lateral-wall speedup guide for the free dt knob
        print(f"  boundary @dt={b['dt_ms']}ms: "
              f"long τ={b['long']['tau']:.3f} [{b['long']['regime']}] | "
              f"trans τ={b['trans']['tau']:.3f} [{b['trans']['regime']}]")


if __name__ == "__main__":  # pragma: no cover
    main()
