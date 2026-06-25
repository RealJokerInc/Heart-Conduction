"""Subprocess driver for the ConductivityConfig live-CV firewall gate (Phase 1, Step 1.2).

Run in a FRESH process (NOT inside the cardiac_core pytest session) so the V5.5 ``cardiac_sim``
namespace is isolated from the wrapper's ``_prepare_engine`` flushing. Drives:

    raw sigma_i/sigma_e/chi  --ConductivityConfig.for_monodomain()-->  (D, chi=1, Cm)
        --> live V5.5 monodomain cable (run_cable_v55)  -->  CV  ==?  bidomain reference CV

Prints a single JSON line to stdout: {"ok": bool, "results": [{Cm, cv, ref, rel}, ...]}.
Exit 0 on overall pass, 1 on fail, 2 on setup error (missing V5.5 dir / ref JSON / imports).
"""

import os
import sys
import json

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_V55 = os.path.join(_REPO, "Monodomain", "Engine_V5.5")
_REF = os.path.join(_V55, "_regression", "bidomain_cm_ref.json")


def main() -> int:
    if not os.path.isdir(_V55) or not os.path.exists(_REF):
        print(json.dumps({"ok": False, "error": f"missing V5.5 dir or ref JSON ({_V55})"}))
        return 2

    # V5.5 first on the path so `import cardiac_sim` (via test_phase10) resolves to V5.5.
    sys.path.insert(0, _V55)
    try:
        from test_phase10_cm_scaling import run_cable_v55, _T_END_BY_CM
        from cardiac_core import ConductivityConfig  # lazy — does NOT touch cardiac_sim
    except Exception as exc:  # pragma: no cover - reported as setup error
        print(json.dumps({"ok": False, "error": f"import failed: {exc!r}"}))
        return 2

    ref = json.load(open(_REF))
    cv_ref = {1.0: ref["cases"]["1.0"]["cv_cm_per_s"],
              2.0: ref["cases"]["2.0"]["cv_cm_per_s"]}

    sigma_i, sigma_e, chi = 1.74, 6.25, 1400.0
    results, ok = [], True
    for Cm in (1.0, 2.0):
        mono = ConductivityConfig.bidomain(sigma_i, sigma_e, chi=chi, Cm=Cm).for_monodomain()
        cv = run_cable_v55(Cm=mono["Cm"], t_end=_T_END_BY_CM[Cm], d_eff=mono["D"])
        rel = abs(float(cv) - cv_ref[Cm]) / cv_ref[Cm]
        ok = ok and (rel <= 0.05)
        results.append({"Cm": Cm, "cv": float(cv), "ref": cv_ref[Cm], "rel": float(rel)})

    # bool()/float() casts — run_cable_v55 returns numpy scalars, which json can't serialize.
    print(json.dumps({"ok": bool(ok), "results": results}))
    return 0 if bool(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
