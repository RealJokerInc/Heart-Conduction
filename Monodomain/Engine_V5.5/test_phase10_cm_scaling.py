"""
Phase 10 — Cm-correctness validation for Monodomain V5.5.

What the fix does: the operator-split reaction now divides by the tissue capacitance,
    dV/dt = -(Iion + Istim) / Cm          (Formulation B; Cm=1 reproduces V5.3/V5.4)

IMPORTANT — there is NO global time-dilation invariant. The tissue Cm divides only the
VOLTAGE update; the gate kinetics (tau from compute_gate_time_constants(V,S)) and the
concentration rates carry NO Cm. So scaling Cm -> k*Cm does NOT rescale the system in
time (V slows, gates keep their kinetics) — the AP morphology changes and APD(Cm=k) is
NOT k*APD(Cm=1). (An earlier plan assumed dilation; empirically false — see IDEALOG.)

So we validate the fix two rigorous ways that do NOT depend on morphology:

  test_0d_exact_cm_scaling   — from an IDENTICAL state, one reaction step: dV scales
                               EXACTLY as 1/Cm (Iion+Istim identical => dV ∝ 1/Cm).
                               Machine-precision proof the code change is correct.
  test_cm_matters_direction  — full 0D AP: Cm changes the AP (slower upstroke at larger
                               Cm), and Cm=1 reproduces the resting/baseline. Sanity that
                               Cm is actually wired into the dynamics, in the right sense.

The Cm=1 no-regression guarantee is covered by _regression/check_golden.py (max|dV|=0).
Cross-engine agreement vs Bidomain V1 at Cm!=1 is handled in Step 2.3 (separate process).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json

import numpy as np
import torch

from cardiac_sim.ionic import TTP06Model, CellType
from cardiac_sim.simulation.classical.state import SimulationState
from cardiac_sim.simulation.classical.solver.ionic_time_stepping.rush_larsen import RushLarsenSolver
from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation
from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol

DEVICE = 'cpu'
DTYPE = torch.float64


def apd90(times, v_trace):
    """APD at 90% repolarization (definition-consistent landmark for ratio comparisons)."""
    v = np.asarray(v_trace); t = np.asarray(times)
    vmin, vmax = float(v.min()), float(v.max())
    thr = vmin + 0.10 * (vmax - vmin)
    above = np.where(v >= thr)[0]
    return float('nan') if len(above) < 2 else float(t[above[-1]] - t[above[0]])


def _make_state(Cm, V0, S0, t=0.0, stim_start=1.0, stim_dur=2.0, stim_amp=-52.0):
    model = TTP06Model(cell_type=CellType.EPI, device=torch.device(DEVICE))
    return model, SimulationState(
        spatial=None, n_dof=1,
        x=torch.zeros(1, device=DEVICE, dtype=DTYPE),
        y=torch.zeros(1, device=DEVICE, dtype=DTYPE),
        V=V0.clone(), ionic_states=S0.clone(),
        gate_indices=list(model.gate_indices),
        concentration_indices=list(model.concentration_indices),
        t=t, Cm=Cm,
        stim_masks=torch.ones(1, 1, device=DEVICE, dtype=DTYPE),
        stim_starts=[stim_start], stim_durations=[stim_dur], stim_amplitudes=[stim_amp],
    )


def _resting_VS():
    model = TTP06Model(cell_type=CellType.EPI, device=torch.device(DEVICE))
    S = model.get_initial_state(n_cells=1).reshape(1, -1).to(DEVICE, DTYPE)
    V = torch.full((1,), float(model.V_rest), device=DEVICE, dtype=DTYPE)
    return V, S


def run_0d(Cm, dt, t_end, **stim):
    """Drive RushLarsen on a 1-dof state with state.Cm set. Returns (times, V_trace)."""
    V0, S0 = _resting_VS()
    model, state = _make_state(Cm, V0, S0, t=0.0, **stim)
    solver = RushLarsenSolver(model)
    times, trace, t = [], [], 0.0
    for _ in range(int(round(t_end / dt))):
        state.t = t
        solver.step(state, dt)
        t += dt
        times.append(t); trace.append(float(state.V[0]))
    return np.array(times), np.array(trace)


def test_0d_exact_cm_scaling():
    """One reaction step from an IDENTICAL non-trivial state: dV must scale as 1/Cm exactly."""
    dt = 0.02
    # Build a non-trivial mid-AP snapshot (plateau, stimulus OFF so dV is pure ionic/Cm).
    _t, _V = run_0d(Cm=1.0, dt=dt, t_end=6.0)  # warm a cell to ~6 ms (post-stim, depolarized)
    # Re-run to grab the full state object at t=6 ms:
    V0, S0 = _resting_VS()
    model, st = _make_state(1.0, V0, S0, t=0.0)
    solver = RushLarsenSolver(model)
    t = 0.0
    while t < 6.0 - 1e-9:
        st.t = t; solver.step(st, dt); t += dt
    V_snap, S_snap, t_snap = st.V.clone(), st.ionic_states.clone(), st.t
    assert t_snap >= 3.0  # stimulus (start=1,dur=2) is OFF by now -> dV is pure -Iion/Cm

    dvs = {}
    for Cm in (0.5, 1.0, 2.0, 4.0):
        _m, s = _make_state(Cm, V_snap, S_snap, t=t_snap)
        RushLarsenSolver(_m).step(s, dt)
        dvs[Cm] = float(s.V[0] - V_snap[0])

    base = dvs[1.0]                       # = -dt * Iion(V_snap,S_snap)
    # dV(Cm) * Cm should be invariant (== base) to machine precision
    inv = {Cm: dvs[Cm] * Cm for Cm in dvs}
    max_dev = max(abs(inv[Cm] - base) for Cm in inv)
    ok = max_dev <= 1e-12 * max(1.0, abs(base))
    print(f"  [exact] dV at snapshot (dt={dt}, t={t_snap:.2f} ms):")
    for Cm in (0.5, 1.0, 2.0, 4.0):
        print(f"      Cm={Cm:<4} dV={dvs[Cm]:+.6e}  dV*Cm={inv[Cm]:+.6e}")
    print(f"      max |dV*Cm - base| = {max_dev:.2e}  (expect ~0)")
    print(f"  {'PASS' if ok else 'FAIL'} test_0d_exact_cm_scaling")
    return ok


def test_cm_matters_direction():
    """Cm is wired into the dynamics: larger Cm => slower upstroke (smaller peak dV/dt)."""
    dt = 0.02
    t1, V1 = run_0d(Cm=1.0, dt=dt, t_end=450.0)
    t2, V2 = run_0d(Cm=2.0, dt=dt, t_end=450.0)
    dvdt1 = float(np.max(np.diff(V1) / dt))
    dvdt2 = float(np.max(np.diff(V2) / dt))
    apd1, apd2 = apd90(t1, V1), apd90(t2, V2)
    # Larger Cm slows the upstroke; the two APs are genuinely different (Cm is active).
    slower = dvdt2 < dvdt1
    differ = abs(apd2 - apd1) > 1.0
    ok = slower and differ
    print(f"  [direction] peak dV/dt: Cm=1 {dvdt1:.1f} mV/ms, Cm=2 {dvdt2:.1f} mV/ms (ratio {dvdt2/dvdt1:.3f})")
    print(f"              APD90:      Cm=1 {apd1:.1f} ms,    Cm=2 {apd2:.1f} ms (NOT 2x — no dilation, as expected)")
    print(f"  {'PASS' if ok else 'FAIL'} test_cm_matters_direction")
    return ok


# ----------------------------------------------------------------------------
# Step 2.3 — Bidomain V1 cross-validation (independent Formulation-B engine)
#
# This is an ABSOLUTE cross-engine agreement test, NOT a dilation test. Bidomain V1
# (Cm-correct, Formulation B) with isotropic D and insulated BC reduces to monodomain
# with D_eff in the bulk; reference CVs are precomputed by _regression/bidomain_cm_ref.py.
# We run V5.5 monodomain at MATCHED physical diffusivity (input D = D_EFF held fixed,
# chi=1, so D_phys = D_EFF/Cm — same as bidomain's D_eff after its D_i,D_e -> /Cm rescale)
# and require V5.5's CV to match bidomain's at BOTH Cm=1 and Cm=2.
#
# Discriminating power: V5.4 (broken reaction) would match at Cm=1 but NOT at Cm=2.
# ----------------------------------------------------------------------------
# Parameters mirror Bidomain/Engine_V1/tests/cv_shared.py (and bidomain_cm_ref.py).
# D_EFF is read from the reference JSON (D_EFF_input) so it always matches the engine the
# reference was generated with — no risk of a hardcoded-sigma drift.
_NX, _NY, _DX = 150, 40, 0.025
_X1, _X2, _YC = 30, 80, 20                  # cv_shared CV gates / center row (indices)
_THRESH = -30.0
_T_END_BY_CM = {1.0: 60.0, 2.0: 160.0}


def _cv_centerline(times, V, xcoord, ycoord, x1_cm, x2_cm, yc_cm, threshold):
    """CV (cm/s) between two x-positions on the center row, from V (n_saves, n_dof)."""
    def node_at(xc, yc):
        return int(np.argmin((xcoord - xc) ** 2 + (ycoord - yc) ** 2))
    n1, n2 = node_at(x1_cm, yc_cm), node_at(x2_cm, yc_cm)

    def act_time(n):
        idx = np.where(V[:, n] > threshold)[0]
        return float(times[idx[0]]) if len(idx) else None
    t1, t2 = act_time(n1), act_time(n2)
    if t1 is None or t2 is None or t2 <= t1:
        return float('nan')
    return abs(xcoord[n2] - xcoord[n1]) / (t2 - t1) * 1000.0  # cm/ms -> cm/s


def run_cable_v55(Cm, t_end, d_eff):
    """V5.5 monodomain matched to the bidomain reference setup. Returns (cv_cm_per_s)."""
    grid = StructuredGrid.create_rectangle(
        Lx=_DX * (_NX - 1), Ly=_DX * (_NY - 1), Nx=_NX, Ny=_NY, device=DEVICE, dtype=DTYPE)
    # Form. A: input D plays the role of conductivity; D_phys = D/(chi*Cm). With chi=1 and
    # D held = d_eff, D_phys = d_eff/Cm — matches bidomain's rescaled D_eff at each Cm.
    spatial = FDMDiscretization(grid, D=d_eff, chi=1.0, Cm=Cm)
    stim = StimulusProtocol()
    x, _y = grid.coordinates
    stim.add_stimulus(region=(x < 5 * _DX), start_time=1.0, duration=2.0, amplitude=-80.0)
    sim = MonodomainSimulation(
        spatial=spatial, ionic_model='ttp06', stimulus=stim, dt=0.01,
        splitting='strang', ionic_solver='rush_larsen',
        diffusion_solver='forward_euler', linear_solver='none')  # explicit: CFL-safe, fast
    times, V = sim.run_to_array(t_end=t_end, save_every=0.5)
    xc = grid.coordinates[0].cpu().numpy()
    yc = grid.coordinates[1].cpu().numpy()
    return _cv_centerline(times, V, xc, yc, _X1 * _DX, _X2 * _DX, _YC * _DX, _THRESH)


def test_bidomain_cross_validation():
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '_regression', 'bidomain_cm_ref.json')
    if not os.path.exists(ref_path):
        print("  SKIP test_bidomain_cross_validation (no bidomain_cm_ref.json — "
              "run _regression/bidomain_cm_ref.py in the Bidomain engine first)")
        return None
    ref_full = json.load(open(ref_path))
    d_eff = ref_full['D_EFF_input']     # match the exact diffusivity the reference used
    ref = ref_full['cases']

    rows, ok = [], True
    for Cm in (1.0, 2.0):
        cv_ref = ref[str(Cm)]['cv_cm_per_s']
        cv_v55 = run_cable_v55(Cm, _T_END_BY_CM[Cm], d_eff)
        rel = abs(cv_v55 - cv_ref) / cv_ref
        rows.append((Cm, cv_ref, cv_v55, rel))
        ok = ok and (rel <= 0.05)
    # Fractional-change agreement (baseline-offset-free corroboration).
    frac_ref = ref['2.0']['cv_cm_per_s'] / ref['1.0']['cv_cm_per_s']
    frac_v55 = rows[1][2] / rows[0][2]

    for Cm, cvr, cvv, rel in rows:
        print(f"  [xeng] Cm={Cm}: bidomain {cvr:.2f} cm/s vs V5.5 {cvv:.2f} cm/s  (rel {rel*100:.1f}%)")
    print(f"  [xeng] CV(2)/CV(1): bidomain {frac_ref:.3f} vs V5.5 {frac_v55:.3f}")
    print(f"  {'PASS' if ok else 'FAIL'} test_bidomain_cross_validation")
    return ok


def main():
    print("=== Phase 10: Cm-correctness validation (V5.5) ===")
    results = {
        'test_0d_exact_cm_scaling': test_0d_exact_cm_scaling(),
        'test_cm_matters_direction': test_cm_matters_direction(),
        'test_bidomain_cross_validation': test_bidomain_cross_validation(),
    }
    results = {k: v for k, v in results.items() if v is not None}  # drop SKIPs
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print("=" * 50)
    print(f"Results: {passed} passed, {total - passed} failed out of {total}")
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
