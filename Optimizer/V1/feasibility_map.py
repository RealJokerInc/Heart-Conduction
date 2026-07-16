"""
Optimizer V1 — Step 2.1 (P1a): conductance-only feasibility map.

The GATE for the whole joint fit (architecture §9): BEFORE building the optimizer,
answer empirically "which lock is required?" — is CV_T reachable at a RESOLVED grid
(r*/dx≥3) with conductance scaling ALONE (no kinetics), and at what dV/dt cost?

Design (efficient; a MAP not a fit): for each (baseline, g_Na, dx) find the D that
hits the CV_T target with the fixed secant (`fit_D_for_cv` at that dx), then check
whether that D RESOLVES the source-sink physics (r* = D/CV, r*/dx ≥ k=3). A point is
  feasible ⇔ CV hits CV_T (within tol)  AND  r*/dx ≥ 3.
dV/dt (the contested lock-1) is recorded per g_Na (cell eval, dx-independent) and
reported for feasible points — rather than pre-picking a dV/dt target, we let the map
show what dV/dt the feasible g_Na implies (physiological hiPSC ~20–50, README ~110).

  GATE: conductance-only feasible at some dx ⇒ skip P1.5 kinetics, go Phase 3.
        infeasible even at dx=0.02 mm with g_Na≥0.15 ⇒ kinetics required (→ P1.5).

Note (vs plan pseudocode): uses the fixed-dx secant + r*/dx≥3 filter instead of the
ladder estimator per point — feasible points ARE resolved (r*/dx≥3 ⇒ CV trustworthy),
so this is equivalent and ~Nx cheaper (the ladder is reserved for the Phase-3 emulator).

Run: conda run -n heart-conduction python Optimizer/V1/feasibility_map.py --baseline hipsc
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from tuner.config import TuningConfig                     # noqa: E402
from tuner.cc_runner import fit_D_for_cv, _build_model    # noqa: E402
from tuner.cell_runner_cc import run_single_cell_cc       # noqa: E402
from tuner.cv_estimator import rstar_cm                   # noqa: E402
from tuner.chip import PARKER                             # noqa: E402
from tuner.presets import load_record                     # noqa: E402

RSTAR_OVER_DX_FLOOR = 3.0
# Widened g_Na floor for the sweep (architecture lock-2: 0.17× ≈ 0.62 absolute needs
# a floor ≤0.17, below the production 0.5 bound).
DEFAULT_GNA_GRID = (0.15, 0.3, 0.5, 1.0)
DEFAULT_DX_MM_GRID = (0.1, 0.05, 0.03, 0.02)
# Physiological hiPSC-CM dV/dt band (Paci-family matured); README target is ~110.
DVDT_PHYS = (20.0, 60.0)


def _theta_with_gNa(base_theta, gNa):
    t = dict(base_theta)
    t["g_Na"] = gNa
    return t


def _cable_cfg(dx_cm, *, cable_length_cm=0.5, dt=0.02):
    return TuningConfig(device="cpu", ionic_model="mhas13", tier=2,
                        dx_cm=dx_cm, cable_length_cm=cable_length_cm, dt=dt,
                        stim_amplitude=-40.0, stim_start=1.0, engine="monodomain")


def _kinetic_model(base_theta, gNa, tau_m, config):
    """Conductance-scaled MHAS13 with an optional Na τ_m multiplier (P1b)."""
    model = _build_model(_theta_with_gNa(base_theta, gNa), config)
    model.tau_m_scale = tau_m
    return model


def feasibility_map(baseline, gNa_grid=DEFAULT_GNA_GRID, dx_mm_grid=DEFAULT_DX_MM_GRID,
                    *, tau_m_grid=(1.0,), cv_tol=0.10, k=RSTAR_OVER_DX_FLOOR,
                    n_beats_cell=3, save_media=True, verbose=True):
    """Return the feasibility map + the (conductance-only, or kinetics) gate.

    `tau_m_grid=(1.0,)` → conductance-only (P1a). A wider grid (e.g. (1.0,1.5,2.0,3.0))
    adds the Na τ_m kinetics axis (P1b) — does slowing Na activation open feasibility
    for CV_T at r*/dx≥k that conductance scaling alone cannot reach?"""
    targets = PARKER[baseline]
    CV_T = targets.cv_transverse
    base_theta = load_record(f"chip_{baseline}")["theta_ionic"]
    kinetics = tuple(tau_m_grid) != (1.0,)

    # dV/dt per (g_Na, τ_m) (cell eval — dx-independent; cache).
    dvdt = {}
    for gNa in gNa_grid:
        for tau_m in tau_m_grid:
            cfg_cell = _cable_cfg(0.01)      # dx irrelevant for the 0-D strip cell eval
            model = _kinetic_model(base_theta, gNa, tau_m, cfg_cell)
            res = run_single_cell_cc(None, cfg_cell, model=model, n_beats=n_beats_cell)
            dvdt[(gNa, tau_m)] = res.dvdt_max if res.dvdt_max is not None else float("nan")
            if verbose:
                print(f"  cell g_Na={gNa:.2f} τ_m={tau_m:.1f} -> dV/dt={dvdt[(gNa, tau_m)]}",
                      flush=True)

    rows = []
    for dx_mm in dx_mm_grid:
        dx_cm = dx_mm / 10.0
        cfg = _cable_cfg(dx_cm)
        for gNa in gNa_grid:
            for tau_m in tau_m_grid:
                model = _kinetic_model(base_theta, gNa, tau_m, cfg)
                D, cv = fit_D_for_cv(None, CV_T, cfg, n=8, tol=0.02, model=model)
                rox = (rstar_cm(D, cv) / dx_cm) if (math.isfinite(cv) and cv > 0) else float("nan")
                cv_ok = math.isfinite(cv) and abs(cv - CV_T) / CV_T < cv_tol
                resolved = math.isfinite(rox) and rox >= k
                feasible = bool(cv_ok and resolved)
                rows.append({"dx_mm": dx_mm, "gNa": gNa, "tau_m": tau_m, "D": D, "cv": cv,
                             "rstar_over_dx": rox, "dvdt": dvdt[(gNa, tau_m)],
                             "cv_ok": cv_ok, "resolved": resolved, "feasible": feasible})
                if verbose:
                    cvs = f"{cv:.2f}" if math.isfinite(cv) else "nan"
                    roxs = f"{rox:.2f}" if math.isfinite(rox) else "nan"
                    print(f"  dx={dx_mm:.02f}mm g_Na={gNa:.2f} τ_m={tau_m:.1f}: D={D:.2e} "
                          f"CV={cvs} r*/dx={roxs} feasible={feasible}", flush=True)

    any_feasible = {dx: any(r["feasible"] for r in rows if r["dx_mm"] == dx)
                    for dx in dx_mm_grid}
    result = {
        "baseline": baseline, "CV_T": CV_T, "rows": rows,
        "any_feasible_by_dx": any_feasible,
        "conductance_only_feasible": (not kinetics) and any(any_feasible.values()),
        "feasible": any(any_feasible.values()),
        "kinetics": kinetics,
        "dvdt_by_gNa": {g: dvdt[(g, tau_m_grid[0])] for g in gNa_grid},
    }
    if save_media:
        result["figure"] = _plot(result, gNa_grid, dx_mm_grid)
    return result


def _plot(result, gNa_grid, dx_mm_grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from cardiac_core.media import media_path

    # status grid: 2=feasible, 1=CV hit but unresolved (r*/dx<3), 0=CV missed/blocked
    status = np.zeros((len(gNa_grid), len(dx_mm_grid)))
    rox = np.full_like(status, np.nan)

    def _score(x):        # prefer feasible, then higher r*/dx (best τ_m per cell)
        return (1 if x["feasible"] else 0,
                x["rstar_over_dx"] if math.isfinite(x["rstar_over_dx"]) else -1.0)
    lut = {}
    for r in result["rows"]:
        key = (r["gNa"], r["dx_mm"])
        if key not in lut or _score(r) > _score(lut[key]):
            lut[key] = r
    for i, gNa in enumerate(gNa_grid):
        for j, dx in enumerate(dx_mm_grid):
            r = lut[(gNa, dx)]
            status[i, j] = 2 if r["feasible"] else (1 if r["cv_ok"] else 0)
            rox[i, j] = r["rstar_over_dx"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#c0392b", "#f1c40f", "#27ae60"])   # blocked / unresolved / feasible
    ax.imshow(status, cmap=cmap, vmin=0, vmax=2, aspect="auto", origin="lower")
    ax.set_xticks(range(len(dx_mm_grid)))
    ax.set_xticklabels([f"{d:.02f}" for d in dx_mm_grid])
    ax.set_yticks(range(len(gNa_grid)))
    ax.set_yticklabels([f"{g:.2f}" for g in gNa_grid])
    ax.set_xlabel("dx (mm)")
    ax.set_ylabel("g_Na scale")
    for i, gNa in enumerate(gNa_grid):
        for j, dx in enumerate(dx_mm_grid):
            v = rox[i, j]
            ax.text(j, i, "nan" if not math.isfinite(v) else f"{v:.1f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if status[i, j] != 1 else "black")
    ax.set_title(f"{result['baseline']} CV_T={result['CV_T']} conductance-only "
                 f"(green=feasible r*/dx≥3 · yellow=CV hit,unresolved · red=blocked)\n"
                 f"cells show r*/dx")
    fig.tight_layout()
    path = media_path("ionic_model_optimization", "images",
                      f"feasibility-{result['baseline']}")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    import warnings
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="hipsc", choices=("nrvm", "hipsc"))
    ap.add_argument("--kinetics", action="store_true",
                    help="(P1b) add the Na τ_m axis (sweep 1.0/1.5/2.0/3.0)")
    ap.add_argument("--fast", action="store_true",
                    help="small grid probe (1 g_Na × 2 dx) for a quick answer")
    args = ap.parse_args()

    kw = {}
    if args.kinetics:
        kw["tau_m_grid"] = (1.0, 1.5, 2.0, 3.0)
    if args.fast:
        kw["gNa_grid"] = (0.5,)
        kw["dx_mm_grid"] = (0.05, 0.03)
    res = feasibility_map(args.baseline, **kw)
    print("\n=== FEASIBILITY GATE ===")
    print(f"baseline={res['baseline']}  CV_T={res['CV_T']} cm/s")
    print(f"dV/dt by g_Na: " + ", ".join(f"{g}->{d:.0f}" for g, d in res['dvdt_by_gNa'].items()))
    for dx, feas in res["any_feasible_by_dx"].items():
        print(f"  dx={dx:.02f} mm: any feasible = {feas}")
    print(f"CONDUCTANCE-ONLY FEASIBLE (some dx): {res['conductance_only_feasible']}")
    print(f"figure: {res.get('figure')}")


if __name__ == "__main__":
    main()
