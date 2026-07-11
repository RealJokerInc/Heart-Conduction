"""
Optimizer V2 — PRODUCTION joint fit (gated heavy run).

Runs the constrained-scalarization joint fit on REAL cardiac_core sims: a GP emulator
over {tier-2 conductances (g_Na floor widened), Na kinetics, D_long, D_trans(free)} is
trained on the resolved 0.02 mm grid, then constrained scalarization finds a θ* that
hits CV_L and CV_T at r*/dx≥3 with a physiological dV/dt band — OR reports which lock
binds. Warm-started from the saved cell fit (presets/chip_{baseline}.json).

Cost: ~n_training × (1 cell AP + 2 tissue CV at 0.02 mm) — multi-hour on CPU. Smoke mode
(--smoke) does a 6-point tiny run to prove the wiring end-to-end.

Run: conda run -n heart-conduction python Optimizer/V1/run_joint_fit.py --baseline hipsc
"""
import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

from tuner.config import TuningConfig                                   # noqa: E402
from tuner.chip import PARKER, RESOLVED_DX_MM                           # noqa: E402
from tuner.decision_space import build_axes                            # noqa: E402
from tuner.joint_fit import (make_sim_evaluator, refine_joint_cc,       # noqa: E402
                             JointFitResult, InfeasReport)
from tuner.presets import make_record, save_record, load_record, PRESETS_DIR  # noqa: E402


def _seed_vectors(axes, base_theta, *, tunable_dx=True):
    """Known-propagating warm-start vectors: the saved cell-fit conductances × a couple
    τ_m values × D pairs × (if tunable) coarse dx values. These anchor the CV GP with
    propagating points (pure Sobol rarely hits the slow-CV manifold). With dx tunable, the
    seeds span coarse dx (0.2–0.3 mm) where slow CV lives at a moderate, non-blocking D."""
    dx_seeds_cm = [0.02, 0.03] if tunable_dx else [None]     # 0.2, 0.3 mm
    D_pairs = [(1.25e-4, 6.25e-5), (2.0e-4, 1.0e-4), (3.0e-4, 1.5e-4)]
    seeds = []
    for tau_m in (1.0, 1.5, 2.0):
        for D_long, D_trans in D_pairs:
            for dx_cm in dx_seeds_cm:
                v = []
                for a in axes:
                    if a.subsystem == "cond":
                        v.append(float(base_theta.get(a.name, 1.0)))
                    elif a.name == "tau_m_scale":
                        v.append(tau_m)
                    elif a.subsystem == "kinetic":
                        v.append(0.0 if a.name == "v_half_shift" else 1.0)
                    elif a.name == "D_long":
                        v.append(D_long)
                    elif a.name == "D_trans":
                        v.append(D_trans)
                    elif a.name == "dx_cm":
                        v.append(dx_cm if dx_cm is not None else 0.02)
                    else:
                        v.append(1.0)
                seeds.append(v)
    return seeds


def run(baseline="hipsc", dx_mm=RESOLVED_DX_MM, n_training=40, n_validate=12,
        n_candidates=4000, n_beats_cell=6, gNa_floor=0.15, kinetics=True,
        cable_length_cm=0.4, seed=42, smoke=False, tunable_dx=True,
        dx_bounds_mm=(0.02, 0.5)):
    dx_cm = dx_mm / 10.0
    config = TuningConfig(
        device="cpu", ionic_model="mhas13", tier=2,
        dx_cm=dx_cm, cable_length_cm=cable_length_cm, dt=0.02,
        stim_amplitude=-40.0, stim_start=1.0, engine="monodomain",
        n_beats=n_beats_cell, pacing_cl=1000.0, dt_cell=0.2,
        dx_mm=dx_mm, domain_mm=16.0,
    )
    dx_bounds_cm = (dx_bounds_mm[0] / 10.0, dx_bounds_mm[1] / 10.0)
    axes = build_axes(tier=2, gNa_floor=gNa_floor, include_kinetics=kinetics,
                      include_dx=tunable_dx, dx_bounds_cm=dx_bounds_cm)
    base_theta = load_record(f"chip_{baseline}")["theta_ionic"]
    targets = PARKER[baseline]

    if smoke:
        n_training, n_validate, n_candidates = 6, 3, 500

    ev = make_sim_evaluator(config, axes, base_theta, resolved_dx_cm=dx_cm,
                            n_beats_cell=n_beats_cell, require_resolved=not tunable_dx)

    print(f"[joint fit] baseline={baseline} axes={len(axes)} tunable_dx={tunable_dx} "
          f"dx_bounds={dx_bounds_mm}mm n_training={n_training} n_validate={n_validate} "
          f"targets(CV_L={targets.cv_longitudinal},CV_T={targets.cv_transverse})",
          flush=True)

    seeds = _seed_vectors(axes, base_theta, tunable_dx=tunable_dx)
    res = refine_joint_cc(
        axes, ev, targets, config=config, base_theta=base_theta,
        n_training=n_training, n_candidates=n_candidates, n_validate=n_validate,
        cv_tol=0.12, dvdt_band=(20.0, 130.0), seed=seed, seed_points=seeds,
        require_resolved=not tunable_dx, verbose=True,
    )

    if isinstance(res, JointFitResult):
        rec = make_record(
            name=f"chip_{baseline}_joint", baseline=baseline,
            theta_ionic=res.theta, kinetics=res.kinetics,
            tissue={"monodomain": {"D_long": res.D_long, "D_trans": res.D_trans,
                                   "dt_ms": config.dt}},
            targets={"cv_longitudinal": targets.cv_longitudinal,
                     "cv_transverse": targets.cv_transverse,
                     "apd_90": targets.apd_90, "dvdt_max": targets.dvdt_max},
            validation={"achieved": res.achieved,
                        "achieved_rstar_over_dx": {
                            "long": res.achieved["rstar_over_dx_l"],
                            "trans": res.achieved["rstar_over_dx_t"]}},
            provenance={"tuner_version": "V2",
                        "fit": "joint_constrained_scalarization",
                        "dx_tunable": tunable_dx},
            dx_mm=(res.dx_cm * 10.0 if res.dx_cm else config.dx_mm),
        )
        path = save_record(rec, name=f"chip_{baseline}_joint")
        dxmm = (res.dx_cm * 10.0) if res.dx_cm else config.dx_mm
        print(f"\n=== JOINT FIT SUCCESS ===\nθ*={res.theta}\nkinetics={res.kinetics}\n"
              f"D_long={res.D_long:.3e} D_trans={res.D_trans:.3e} dx={dxmm:.3f}mm\n"
              f"achieved={res.achieved}\nsaved: {path}", flush=True)
    else:
        print(f"\n=== INFEASIBLE (binding lock: {res.binding_lock}) ===\n{res.detail}",
              flush=True)
        path = os.path.join(PRESETS_DIR, f"chip_{baseline}_joint_INFEASIBLE.json")
        os.makedirs(PRESETS_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"binding_lock": res.binding_lock, "detail": res.detail,
                       "baseline": baseline, "dx_mm": config.dx_mm}, f, indent=2)
        print(f"saved: {path}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="hipsc", choices=("nrvm", "hipsc"))
    ap.add_argument("--n-training", type=int, default=40)
    ap.add_argument("--n-validate", type=int, default=12)
    ap.add_argument("--n-beats-cell", type=int, default=6)
    ap.add_argument("--no-kinetics", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run(baseline=args.baseline, n_training=args.n_training, n_validate=args.n_validate,
        n_beats_cell=args.n_beats_cell, kinetics=not args.no_kinetics, smoke=args.smoke)


if __name__ == "__main__":
    main()
