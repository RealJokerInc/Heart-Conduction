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
import math

import torch

from tuner.config import TuningConfig, TuningTargets, theta_to_dict
from tuner.cc_runner import run_1d_cable
from tuner.chip import PARKER
from tuner.presets import make_record, save_record, PRESETS_DIR


def select_best(cellres) -> torch.Tensor:
    """Pick the Pareto-front theta with the best (max) summed objective."""
    scores = cellres.pareto_Y.sum(dim=1)
    return cellres.pareto_X[int(scores.argmax())]


def _fit_D_for_cv(theta_ionic, target_cv, config, *, D0=0.001, n=6, tol=0.02,
                  D_lo=1e-6, D_hi=1e-2, t_end=None):
    """Secant on diffusion D to hit `target_cv` (cm/s) via cc_runner.

    Warm-started by CV ∝ √D, then two-point secant (NOT Newton — Known Failure).
    Returns (D, cv_achieved). Robust to a non-propagating warm-start guess.
    """
    def cv(D):
        return run_1d_cable(theta_ionic, D, config, t_end=t_end)

    cv0 = cv(D0)
    if not (math.isfinite(cv0) and cv0 > 0):
        D0 = min(D_hi, D0 * 4.0)                 # bump D for propagation
        cv0 = cv(D0)
        if not (math.isfinite(cv0) and cv0 > 0):
            return D0, float("nan")

    # analytic warm start
    D1 = min(D_hi, max(D_lo, D0 * (target_cv / cv0) ** 2))
    cv1 = cv(D1)
    if not (math.isfinite(cv1) and cv1 > 0):
        return D0, cv0                            # keep the propagating point

    it = 2
    while it < n and abs(cv1 - target_cv) / target_cv > tol:
        if cv1 == cv0:
            break
        D2 = D1 + (target_cv - cv1) * (D1 - D0) / (cv1 - cv0)
        D2 = min(D_hi, max(D_lo, D2))
        cv2 = cv(D2)
        if not (math.isfinite(cv2) and cv2 > 0):
            break
        D0, cv0, D1, cv1 = D1, cv1, D2, cv2
        it += 1
    return D1, cv1


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
    D_long, cvL = _fit_D_for_cv(theta_ionic, targets.cv_longitudinal, config, n=n, t_end=t_end)
    D_trans, cvT = _fit_D_for_cv(theta_ionic, targets.cv_transverse, config, n=n, t_end=t_end)

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


if __name__ == "__main__":  # pragma: no cover
    main()
