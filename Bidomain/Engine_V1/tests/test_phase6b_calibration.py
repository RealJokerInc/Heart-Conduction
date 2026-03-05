"""Phase 6B: CV Calibration Across Engines.

Runs a full TTP06 wave propagation in each engine and measures absolute
conduction velocity at the domain center. The monodomain FDM control
(V5.3-style explicit Euler) is the primary reference since it shares the
same FDM spatial operator and ionic model as the bidomain engine.

Tests:
  6B-T1: Monodomain FDM center CV in physiological range
  6B-T2: Bidomain FDM insulated center CV matches monodomain within 5%
  6B-T3: LBM D2Q5 center CV in physiological range
  6B-T4: Calibration summary table

Run: conda run -n heart-conduction python Bidomain/Engine_V1/tests/test_phase6b_calibration.py
     conda run -n heart-conduction python Bidomain/Engine_V1/tests/test_phase6b_calibration.py mono bidomain
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'Monodomain', 'LBM_V1'))

import torch
torch.set_default_dtype(torch.float64)

from cv_shared import (
    NX, NY, DX, DT, D_I, D_E, D_EFF, T_END, SAVE_EVERY,
    Y_CENTER, measure_cv_from_history, is_nan, format_cv,
    run_lbm, run_bidomain, run_monodomain_fdm, timed_run,
)


# ============================================================
# Individual Engine Runs
# ============================================================
def run_mono():
    """Monodomain FDM with D=D_eff (primary control)."""
    return run_monodomain_fdm(D=D_EFF)


def run_d2q5():
    """LBM D2Q5 wave propagation."""
    return run_lbm(lattice='d2q5')


def run_bidomain_insulated():
    """Bidomain FDM with insulated BCs."""
    return run_bidomain(bc_type='insulated')


# ============================================================
# Tests
# ============================================================
def test_6b(results):
    """Run all calibration tests on collected results."""
    print("\n" + "=" * 68)
    print("Phase 6B: CV Calibration Results")
    print("=" * 68)
    print(f"Domain: {NX}x{NY}, dx={DX}cm, dt={DT}ms, D_eff={D_EFF:.6f}")
    print(f"CV measured at y_center={Y_CENTER}")
    print()

    header = f"  {'Engine':<30} {'CV_center (cm/s)':>18} {'Status':>10}"
    print(header)
    print("  " + "-" * 60)

    for name, cv in results.items():
        cv_str = format_cv(cv)
        if is_nan(cv):
            status = "FAIL"
        elif 0.03 < cv < 0.15:
            status = "OK"
        else:
            status = "OUT_RANGE"
        print(f"  {name:<30} {cv_str:>16}   {status:>10}")

    print()

    # 6B-T1: Monodomain in physiological range
    cv_mono = results.get('Monodomain FDM', float('nan'))
    if not is_nan(cv_mono):
        ok = 0.03 < cv_mono < 0.15
        print(f"6B-T1: Monodomain FDM CV = {cv_mono*1000:.1f} cm/s — "
              f"{'PASS' if ok else 'FAIL'} (range: 30-150 cm/s)")
        assert ok, f"Monodomain CV {cv_mono*1000:.1f} outside physiological range"
    else:
        print("6B-T1: SKIP (Monodomain not run)")
        return

    # 6B-T2: Bidomain within 5% of monodomain (same FDM, same ionic model)
    cv_bi = results.get('Bidomain insulated', float('nan'))
    if not is_nan(cv_bi):
        rel_diff = abs(cv_bi - cv_mono) / cv_mono
        ok = rel_diff < 0.05
        print(f"6B-T2: Bidomain CV = {cv_bi*1000:.1f} cm/s, "
              f"diff from mono = {rel_diff*100:.1f}% — "
              f"{'PASS' if ok else 'FAIL'} (threshold: 5%)")
        assert ok, f"Bidomain differs from Monodomain by {rel_diff*100:.1f}%"
    else:
        print("6B-T2: SKIP (Bidomain not run)")

    # 6B-T3: LBM D2Q5 in physiological range (informational)
    cv_d2q5 = results.get('LBM D2Q5', float('nan'))
    if not is_nan(cv_d2q5):
        ok = 0.03 < cv_d2q5 < 0.15
        rel_diff = abs(cv_d2q5 - cv_mono) / cv_mono
        print(f"6B-T3: LBM D2Q5 CV = {cv_d2q5*1000:.1f} cm/s, "
              f"diff from mono = {rel_diff*100:.1f}% — "
              f"{'PASS' if ok else 'FAIL'} (range: 30-150 cm/s)")
    else:
        print("6B-T3: SKIP (LBM not run)")

    # 6B-T4: Summary
    print()
    cv_list = [(k, v) for k, v in results.items() if not is_nan(v)]
    if len(cv_list) >= 2:
        cvs = [v for _, v in cv_list]
        avg = sum(cvs) / len(cvs)
        spread = (max(cvs) - min(cvs)) / avg * 100
        print(f"6B-T4: Calibration spread = {spread:.1f}% "
              f"(mean CV = {avg*1000:.1f} cm/s)")
        if spread < 10:
            print("    FDM engines well-calibrated for cross-validation.")
        else:
            print("    Note: LBM uses different numerics; FDM-FDM match is key.")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    engines = sys.argv[1:] if len(sys.argv) > 1 else ['mono', 'bidomain', 'd2q5']

    runners = {
        'mono': ('Monodomain FDM', run_mono),
        'd2q5': ('LBM D2Q5', run_d2q5),
        'bidomain': ('Bidomain insulated', run_bidomain_insulated),
    }

    results = {}
    for key in engines:
        key = key.lower()
        if key in runners:
            name, fn = runners[key]
            data = timed_run(name, fn)
            if data is not None:
                times, V_hist = data
                cv = measure_cv_from_history(V_hist, times, Y_CENTER)
                results[name] = cv
                print(f"    CV at center: {format_cv(cv)} cm/s")
            else:
                results[name] = float('nan')
        else:
            print(f"Unknown engine: {key} (use mono / bidomain / d2q5)")

    if results:
        test_6b(results)
