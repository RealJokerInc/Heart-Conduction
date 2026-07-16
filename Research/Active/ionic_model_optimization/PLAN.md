# PLAN: Joint Ionic + Conduction Tuning (Engine Tuner V2)

Created: 2026-07-10
Engine(s): Optimizer (build) + cardiac_core (integration)
Research question: [ionic_model_optimization](README.md)
Source: [IDEALOG.md](IDEALOG.md) §2026-07-10 (JOINT_TUNING_ARCHITECTURE audit-CONVERGED) +
[Optimizer/V1/JOINT_TUNING_ARCHITECTURE.md](../../../Optimizer/V1/JOINT_TUNING_ARCHITECTURE.md) (the settled, audit-converged design — READ IT FIRST)

> Supersedes the completed cross-plan (archived `plans/2026-06-29_engine-tuner-cardiac-core-cross-plan_DONE.md`).
> **Status: AUDIT-CONVERGED** (3 iters: 1c/1h/6m/4l → 0c/1h/1m/2l → 0c/0h/1m/1l → CONVERGED). Ready for execution — awaiting explicit "go" (hard gate: no implementation before approval).

## Objective
Replace the broken sequential cell→tissue fit with a **joint** ionic+conduction fit that (a) measures
both AP and conduction on **one** ionic backend, (b) treats grid resolution as a *resolved-not-fit*
convergence problem, and (c) optimizes {θ, D, (kinetics)} together via constrained scalarization on a
GP emulator. Goal: reach the anisotropic Kit-Parker chip EP targets on a grid that actually resolves
the wavefront (r*/dx ≳ 3), or *explicitly* report which of the three locks (dV/dt target, g_Na range,
dx) makes a target infeasible.

## Success Criteria
- [ ] **Backend unified** — dV/dt/APD and CV measured on the same cardiac_core model (parity vs current cell fit ≤1%).
- [ ] **Joint refinement** (README criterion) — one optimizer over {θ, D_long, D_trans(free)} on a GP emulator; constrained scalarization surfaces infeasibility explicitly.
- [ ] A tuned θ* reproduces CV_L **and** CV_T within tol at **r*/dx ≥ 3** on a resolved grid — OR the feasibility map documents which lock blocks it.
- [ ] **dVdt target revised/reconciled** for MHAS13 (README criterion).
- [ ] Records/presets carry (θ, D_long, D_trans, achieved r*/dx, dx-ladder); `export_lab_preset` no longer KeyErrors on monodomain-only records.
- [ ] All existing Optimizer V1 + cardiac_core tests pass (no regressions).

## Architecture Changes
- NEW: `Optimizer/V1/tuner/cv_estimator.py` — convergence-aware CV estimator (dx-ladder → resolved CV + achieved r*/dx). Used by P0 discriminators, P1 feasibility map, P2 emulator training.
- NEW: `Optimizer/V1/tuner/decision_space.py` — unified decision-space registry `{subsystem, bounds, apply_fn}` + constraint graph (hard: r*/dx≥k; soft: √D, 2:1 warm-starts). Single `apply(vector) → (scaled_model, mesh)`.
- NEW: `Optimizer/V1/tuner/cell_runner_cc.py` — cardiac_core-backed single-cell AP eval (P-1).
- MOD: `Optimizer/V1/run_chip_fit.py:30-64` + `tuner/cc_runner.py:115-148` — fix the ×4-up-bump; bracket DOWN into the propagating window.
- MOD: `Optimizer/V1/tuner/joint_refiner.py` — rewire off `tissue_runner.run_cv_measurement` to `cc_runner`; add r*/dx constraint; resolved-CV training data; block-region masking (None→NaN guard); constrained-scalarization objective.
- MOD (P1.5, gated): `cardiac_core/ionic/mhas13/model.py` — per-instance Na-kinetic multipliers around the imported `phas13.gating.INa_*` functions (NOT hard-editing shared `phas13/gating.py`).
- MOD: `Optimizer/V1/tuner/chip.py:35-52` — chip mesh dx → ~0.02 mm (P3); `tuner/presets.py:119-128` — guard `export_lab_preset` for absent engine tissue block.

## Known Failures (from IDEALOG — MUST NOT retry)
- **Sequential cell→tissue tuning** — CV=f(θ,D), G_Na shared between dV/dt and CV/r*; frozen-θ secant drives slow targets into the r*/dx<1 block.
- **Secant D fallback bumping D UP ×4** — for slow chip targets the window is BELOW D0=0.001; bumping up (→0.004) goes the wrong way into nan.
- **k=1 resolvability** — only marginally propagates on a grid that doesn't resolve source-sink (need k≈3); CV there is grid-fudgable → fitting to a numerical artifact.
- **4-objective qNEHVI / naive tissue-in-the-loop** — sim cost dominates; keep tissue sims (+dx-ladder) in emulator TRAINING, not the inner loop. Use the existing `joint_refiner` GP-emulator pattern.
- **Hard-tying `D_trans = D_long/ratio²`** — it derives from soft `CV∝√D`; hard-encoding fences out the solution. D_trans is a FREE variable.
- **dt as a CFL hard wall** — solver is implicit Crank-Nicolson (unconditionally stable); dt is accuracy-bounded.
- **dV/dt = 25 V/s target** — unreachable for MHAS13 (native ~110–132) by conductance scaling.

---

## Phase 0 (P-1): Backend Unification — one ionic model for AP + conduction

**Goal**: dV/dt/APD (cell) and CV (tissue) measured on the SAME cardiac_core ionic model, so a kinetics axis is identifiable and θ warm-starts cleanly. Hard prerequisite for everything downstream.
**Tier**: large
**Estimated scope**: reroute cell evaluation off `cardiac_sim`/V5.4 onto `cardiac_core`; parity-validate.

### Phase Context
Today `cell_runner.py`/`batch_ionic.py` import `cardiac_sim.ionic.phas13` (V5.4, via `sys.path.insert` to `Monodomain/Engine_V5.4`); `cc_runner.py` imports `cardiac_core`. The two backends compute the same nominal model differently → dV/dt (V5.4) and CV (cardiac_core) are not from one model. cardiac_core ionic models carry **scalar** `self.params` (`cardiac_core/ionic/mhas13/model.py`), applied via `tuner.config.apply_scaling`. `batch_ionic.build_conductance_tensor` builds `(M,14)` per-candidate conductances for a batched step — cardiac_core cannot batch distinct-conductance candidates, so the port serializes candidates initially (acceptable: emulator training + feasibility map need ~50–200 evals, not thousands). Do NOT modify `Monodomain/Engine_V5.3/` or `Engine_V5.4/`. float64 throughout.

### Step 0.1: Cardiac_core-backed cell runner
**Model**: opus

#### Read First
- `Optimizer/V1/tuner/cell_runner.py:1-60` + `:33-44` — current AP-eval interface; the return dataclass is **`CellResult`** (not "Biomarkers"); `extract_biomarkers_batch` lives here.
- `Optimizer/V1/tuner/metrics.py` — the real biomarker fns: `measure_apd`, `measure_dvdt_max`, `measure_v_rest`, `measure_peak` (NOT `extract_biomarkers`).
- `cardiac_core/ionic/mhas13/model.py:64` (`self.params` init), `:108` (`step()` sig), `:283-319` (`compute_gate_*` hooks — the tissue path).
- ⚑ `cardiac_core/_monodomain/.../solver/ionic_time_stepping/rush_larsen.py:70-93` — tissue drives via the **hooks**, not `step()` (so `step()`'s Cai-dependent ICaL `constf1/constfCa` are OFF the tissue path).
- `Optimizer/V1/tuner/cc_runner.py:37-46` — `_build_model`/`apply_scaling`; `:88-90` — `run_monodomain` call.

#### Why
The kinetics decoupling (P1.5) is only *observable* if dV/dt and CV come from one model **via the same path**. Because tissue CV uses the gate HOOKS (not `step()`), the cell AP must ALSO use the hook path — otherwise cell (`step()`, with ICaL Cai-modifiers) and tissue (hooks, without) disagree on APD, and the P1.5 kinetics edit (in the hooks) would be inert in the cell. **Drive the cell AP as a 0-D `run_monodomain`** (single-cell / D=0 mesh) so it uses the identical hook-based Rush-Larsen path.

#### Implementation Spec
**Files to create:** `Optimizer/V1/tuner/cell_runner_cc.py` — cardiac_core-backed single-cell AP eval returning a **`CellResult`** (same fields as `cell_runner`).
**Files to modify:** `cell_fitter.py` / `_evaluate_batch` AP-eval call site — behind a `config.ionic_backend='cardiac_core'` flag (default), route to the new runner; keep the V5.4 path for parity testing only. **NB the full BO cell fit (`fit_cell`, `n_iterations=200`) is RETIRED** in the new architecture — the emulator (Phase 3) replaces the inner loop, so serial cell eval only ever runs the feasibility map + emulator training (~O(100)).
**Interfaces:** `run_single_cell_cc(theta_ionic: dict, config) -> CellResult(apd90, dvdt_max, v_peak, v_rest, converged)`; use `cc_runner._build_model` for the scaled model; drive a small **uniform strip** via `run_monodomain` (a **few-cell** strip, NOT a literal 1×1/D=0 grid — avoids degenerate CN/PCG operators; diffusion is inert under a flat state), **paced to steady state**, extract from the **last** AP at a single node.
**Pacing parity (load-bearing):** the V5.4 baseline (`cell_runner`) paces `n_beats=10` at `pacing_cl=1000` and measures the *last* AP. `create_cardiac_mesh` defaults to a **single** stimulus (`bcl=0, num_pulses=1`) → one cold-start beat, which differs from the 10th paced beat by ≫1%. So build a **multi-pulse** stimulus (`num_pulses=n_beats, bcl=pacing_cl`; `api.py:1013-1022` honors it) and measure the last beat, or parity fails.

#### Pseudocode
```
model = cc_runner._build_model(theta_ionic, config)                 # scaled cardiac_core model
mesh  = build_uniform_strip(n_cells≈8, dx, num_pulses=n_beats, bcl=pacing_cl)  # multi-pulse to steady state
t, V  = run_monodomain(mesh, ionic_model=model, dt=config.dt_cell,
                       save_every=config.dt_cell, ...)              # HOOK path; fine save_every to resolve the upstroke (dV/dt parity)
Vn    = V[:, ix, iy]                                                 # reduce (n_saves,Nx,Ny) -> node trace (N,)
# arg order: measure_apd(V, t), measure_dvdt_max(V, t) (voltage FIRST); measure_peak(V) is ONE arg
return CellResult(apd90=measure_apd(Vn, t), dvdt_max=measure_dvdt_max(Vn, t),
                  v_peak=measure_peak(Vn), v_rest=measure_v_rest(Vn, t), converged=...)
# batch: loop candidates (serial); throughput note — full BO path retired, so ~O(100) evals only
```

#### Test Spec
- `tests/test_cell_runner_cc.py::test_parity_vs_v54` — Setup: baseline MHAS13 (identity θ). Expected: APD90, dV/dt, V_rest, V_peak within **1%** of the current `cell_runner` (V5.4) values.
- `::test_scaling_moves_biomarkers` — g_CaL×1.5 raises APD; g_Na×0.5 lowers dV/dt (monotone, sane).

#### Checklist
- [ ] New runner returns the biomarker dataclass unchanged in shape.
- [ ] `config.ionic_backend` flag added (default `'cardiac_core'`).
- [ ] Serial batch path + throughput note in docstring.

#### Verify
```bash
conda run -n heart-conduction python -m pytest Optimizer/V1/tests/test_cell_runner_cc.py -v
```

#### Exit Criteria
- [ ] Parity ≤1% vs V5.4 on baseline; scaling monotonic.
- [ ] No `cardiac_sim` import remains on the default AP path.

#### Risk
Parity delta has TWO distinct causes — do not conflate: (1) **pacing history** (beat-count / steady-state loading) — the correctable one; fix the multi-pulse pacing FIRST before blaming the model. (2) **LUT/integrator** differences between V5.4 and cardiac_core — genuine model delta; if >1% after pacing is matched, document it and re-anchor targets to the cardiac_core baseline (the tissue side already uses it → cardiac_core is the reference). (3) **Output sampling** — use `save_every≈dt_cell`; the default 1.0 ms under-samples the upstroke and inflates dV/dt_max error.

### Phase 0 Verification
```bash
conda run -n heart-conduction python -m pytest Optimizer/V1/tests/ -q
conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q -k "ionic or mhas13"
```
### Phase 0 Exit Criteria
- [ ] AP + CV both on cardiac_core; parity documented.
- [ ] All existing Optimizer/cardiac_core tests green.
### Phase 0 Cleanup
- [ ] float64 (no float32 leak); V5.3/V5.4 untouched; no duplicated ionic code (reuse `_build_model`).
**-> Commit point.**

---

## Phase 1 (P0): Discriminators + the CV estimator

**Goal**: cheap CPU experiments that turn the architecture's open questions into data BEFORE building the optimizer: fix the secant bug, find the true hiPSC propagating window, settle the high-D nan, and build the convergence-aware CV estimator everything downstream needs.
**Tier**: medium
**Estimated scope**: 1 bugfix + the CV estimator + 2 diagnostic sweeps (report to `media/`).

### Phase Context
The chip runner is `cc_runner.run_1d_cable` (monodomain, implicit CN — no CFL wall). Blocked propagation returns `NaN` from `analysis.conduction_velocity`. Save all figures under `media/ionic_model_optimization/images/{date}/`. Use the saved warm-start θ in `Optimizer/V1/presets/chip_{nrvm,hipsc}.json`.

### Step 1.1: Fix the secant ×4-up-bump (bracket down)
**Model**: sonnet
#### Read First
- `Optimizer/V1/run_chip_fit.py:30-64` (`_fit_D_for_cv`) and `Optimizer/V1/tuner/cc_runner.py:115-148` (`fit_D_for_cv`) — the duplicated secant with the `D0*4.0` fallback.
#### Why
For slow chip targets the propagating window (~5e-5–1e-4) is BELOW D0=0.001; bumping D *up* goes into nan. Bracketing DOWN may rescue CV_L outright and narrows what the joint fit must own.
#### Implementation Spec
Replace the up-bump: on non-finite `cv(D0)`, **halve** D toward the window (geometric bracket down to `D_lo`), not ×4 up. Add a bracketing search that finds a propagating D before the secant.
#### Pseudocode
```
if not finite(cv(D0)): for D in [D0/2, D0/4, ... >= D_lo]: if finite(cv(D)): D0=D; break
                        else: return (nan, nan)  # honestly infeasible, do NOT return a fake D
```
#### Test Spec
- `tests/test_secant_bracket.py::test_brackets_down` — Setup: target CV=6 cm/s, baseline θ. Expected: returns D in [3e-5,2e-4], finite CV within tol; never returns 0.004.
#### Checklist
- [ ] Both copies fixed (or de-duplicated to one).
- [ ] Non-propagating → (nan,nan), not a fallback D.
#### Verify
```bash
conda run -n heart-conduction python -m pytest Optimizer/V1/tests/test_secant_bracket.py -v
```
#### Exit Criteria
- [ ] Bracket-down verified; no fake-D fallback.
#### Risk
Both baselines may still be infeasible at dx=0.1 — that's a *finding* (feeds P1.3), not a bug.

### Step 1.2: Convergence-aware CV estimator (dx-ladder)
**Model**: opus
#### Read First
- `cardiac_core/analysis.py` — `conduction_velocity` signature/NaN behavior.
- JOINT_TUNING_ARCHITECTURE.md §4 — the estimator spec, the r*/dx≥3 ladder floor, "extrapolate ε out, don't fit dx".
#### Why
Every downstream step needs a *resolved* CV (not a grid-artifact) and the achieved r*/dx. The estimator runs a dx-ladder (fixed θ,D,dt; vary dx only) with all rungs at r*/dx≳3, and reports resolved CV + r*/dx, or "still blocked at finest rung".
#### Implementation Spec
**File:** `Optimizer/V1/tuner/cv_estimator.py`. `resolved_cv(theta, D, config, dx_ladder=(...)) -> {cv_resolved, rstar, rstar_over_dx, converged: bool, rungs}`. r* = D/cv. Reject/flag rungs with r*/dx < 3 (never extrapolate through the corrupted band).
#### Pseudocode
```
cvs = [run_1d_cable(theta,D,cfg@dx) for dx in ladder]
rstar = D / cv_finest
if any rung r*/dx < 3 or any nan: return {converged:False, ...}   # cannot extrapolate through block/corrupt band
cv_resolved = richardson_or_plateau(cvs)                          # smooth trend only
```
#### Test Spec
- `tests/test_cv_estimator.py::test_reports_blocked` — slow D where finest rung has r*/dx<3 → `converged=False`.
- `::test_resolved_stable` — fast D, ladder converges → `cv_resolved` stable across finest 2 rungs (≤3%).
#### Checklist
- [ ] Ladder rejects r*/dx<3 rungs; never extrapolates through nan.
- [ ] Returns achieved r*/dx.
- [ ] **dt adequacy**: at the finest dx rung, confirm CV is dt-converged (halve dt once; ΔCV ≤ tol) — the estimator resolves *dx*; verify dt=0.02 ms is still accuracy-adequate at ~0.02 mm, else add a dt rung. (Solver is implicit CN — no CFL wall — so this is accuracy, not stability. Also flag the stale `config.py:44` CFL comment.)
#### Verify
```bash
conda run -n heart-conduction python -m pytest Optimizer/V1/tests/test_cv_estimator.py -v
```
#### Exit Criteria
- [ ] Estimator distinguishes resolved vs blocked; reports r*/dx.
#### Risk
Ladder cost — keep rungs to 3; this feeds emulator training, not an inner loop.

### Step 1.3: Diagnostics — hiPSC-θ window + high-D nan
**Model**: opus
#### Read First
- `Optimizer/V1/presets/chip_hipsc.json` — saved hiPSC θ.
- `Optimizer/V1/tuner/cc_runner.py:14-17` (chi low-D wave-death comment); `config.py:63-65` (over-depol comment).
#### Why
The architecture's window (5e-5–1e-4) was measured at *NRVM* θ; the unreachable target is hiPSC. And the high-D nan (D=1e-3) is unexplained (CN over-depol vs CV-measurement artifact) — it sets the true window width.
#### Implementation Spec
Script `Optimizer/V1/diag_hipsc_window.py`: sweep D at hiPSC θ across {4e-3…1e-5}, report resolved CV + r*/dx + Vmax + whether the nan is a sim blow-up (Vmax huge) or CV-measurement failure (finite V, no threshold crossing). Save table + a figure to `media/`.
#### Pseudocode
```
for D in grid: cv=resolved_cv(theta_hipsc,D); vmax=max|V|; classify(nan_cause)
plot CV & r*/dx vs D; annotate window + block edge; save media path
```
#### Test Spec
- Smoke: `tests/test_diag_smoke.py::test_hipsc_window_runs` — runs on a tiny grid, produces a dict with `window` and `nan_cause` keys.
#### Checklist
- [ ] hiPSC window identified; high-D nan classified.
- [ ] Figure saved to `media/ionic_model_optimization/images/{date}/`.
#### Verify
```bash
conda run -n heart-conduction python Optimizer/V1/diag_hipsc_window.py
```
#### Exit Criteria
- [ ] Documented hiPSC window + nan cause → update IDEALOG.
#### Risk
If the high-D nan is a CV-measurement artifact, the window is wider than the architecture's "5e-5–1e-4" — update §1/§4 of the architecture doc accordingly.

### Phase 1 Verification / Exit / Cleanup
```bash
conda run -n heart-conduction python -m pytest Optimizer/V1/tests/test_secant_bracket.py Optimizer/V1/tests/test_cv_estimator.py -q
```
- [ ] Secant fixed; estimator works; diagnostics logged. float64; figures in `media/`.
**-> Commit point.**

---

## Phase 2 (P1a/P1.5/P1b): Feasibility Map + gated Kinetics

**Goal**: BEFORE building the optimizer, empirically answer "which lock is required" — plot the feasible region for the chip targets across dx and the g_Na/dV/dt locks. Add kinetics ONLY if conductance-only is infeasible.
**Tier**: large
**Estimated scope**: a feasibility-map sweep (P1a) with a data gate → optional kinetics model change (P1.5) → re-map (P1b).

### Phase Context
"Feasible" = ∃(θ,D) with CV_L, CV_T within tol, dV/dt within the (revised) target band, AND every ladder rung r*/dx≥3, at a given dx. This is a MAP/plot, not a fit. dV/dt target is CONTESTED (README wants ~100 for matured MHAS13; slow-CV wants lower) — sweep it as a parameter, don't hard-pick. Kinetics (P1.5) is a per-instance model change around `phas13.gating.INa_*`, applied in `mhas13/model.py`, NOT a hard edit to shared `phas13/gating.py`; V5.3 untouched.

### Step 2.1: Conductance-only feasibility map (P1a)
**Model**: opus
#### Read First
- JOINT_TUNING_ARCHITECTURE.md §9 P1a + locks 1-3; `tuner/config.py:88-115` (registry + bounds, widen g_Na floor to ≤0.17 for the sweep).
#### Why
Answers open-Q1 for the cheap branch: does A+B (sane dV/dt target + adequate dx, no kinetics) reach CV_T=2.6 at r*/dx≥3? Gates whether P1.5 is even needed.
#### Implementation Spec
`Optimizer/V1/feasibility_map.py`: grid over (g_Na∈[0.15,2], D) × dx∈{0.1,0.05,0.03,0.02 mm} × dV/dt_target∈{110,80,50,30}; for each, `resolved_cv` → check (CV_T tol, dV/dt band, r*/dx≥3). Output feasible-region plots per dx + a boolean "conductance-only feasible?" per (baseline, dx).
#### Pseudocode
```
for baseline in [nrvm,hipsc]: for dx: for gNa:
   dvdt = run_single_cell_cc(theta(gNa),cfg).dvdt_max     # cell eval (cache per gNa — dx-independent)
   for D:
     est = resolved_cv(theta(gNa),D,cfg@dx)               # tissue CV + r*/dx
     feasible = est.converged and cvtol(est.cv_resolved) and dvdt_band(dvdt) and est.rstar_over_dx>=3
plot feasible region; decide gate per (baseline,dx)
```
#### Test Spec
- Smoke `tests/test_feasibility_smoke.py::test_map_runs` — tiny grid returns a feasibility bool array + a saved figure path.
#### Checklist
- [ ] Feasible-region plots saved to `media/`; per-(baseline,dx) gate decision recorded.
#### Verify
```bash
conda run -n heart-conduction python Optimizer/V1/feasibility_map.py --baseline hipsc
```
#### Exit Criteria
- [ ] **GATE**: conductance-only feasible ⇒ skip 2.2/2.3, go Phase 3. Infeasible ⇒ do 2.2.
#### Risk
Sweep cost — coarse grid first (5×5×4), refine only near the feasible boundary.

### Step 2.2: Kinetics model change (P1.5 — GATED on 2.1 infeasible)
**Model**: opus
#### Read First
- ⚑ `cardiac_core/_monodomain/.../solver/ionic_time_stepping/rush_larsen.py:70-93` — the tissue solver drives the model via `compute_Iion` + `compute_gate_steady_states` + `compute_gate_time_constants`. **It NEVER calls `MHAS13Model.step()`.** This dictates where scaling must go.
- `cardiac_core/ionic/mhas13/model.py:283-319` — `compute_gate_steady_states` (:283) and `compute_gate_time_constants` (:303) — **the hooks the tissue path uses**. NB `step()` (:178-201) applies Cai-dependent `constf1/constfCa` ICaL modifiers the hooks don't — do NOT scale in `step()`.
- `cardiac_core/ionic/phas13/gating.py:28-88` — `INa_m_tau` etc. (SHARED — do not hard-edit).
#### Why
Only kinetics decouple dV/dt from CV. **The tissue CV path uses the gate HOOKS, not `step()`** — so scaling must live in `compute_gate_time_constants`/`compute_gate_steady_states`, or the axis is invisible to CV (re-creating the two-backend bug P-1 fixed). Per-instance so PHAS13 is unaffected.
#### Implementation Spec
Add `tau_m_scale`, `tau_h_scale`, `tau_j_scale`, `v_half_shift` instance attributes (default 1.0/0.0 = identity). Apply them **inside the hooks**: multiply `INa_m_tau(V)`/`INa_h_tau`/`INa_j_tau` by their scales in `compute_gate_time_constants` (`model.py:311`), and shift the Na steady-state V in `compute_gate_steady_states` (`model.py:293`). Do NOT edit `step()` or the shared `phas13/gating.py`. Register the new axes in **`config.py` PHAS13_REGISTRY-analog (concrete, Phase 2)** — `decision_space.py` (Phase 3) imports from there, not vice-versa. Add CV-restitution as an identifying observable (multi-rate CV; also breaks IKr/IKs).
#### Pseudocode
```
# mhas13.compute_gate_time_constants(V,S):  tau[:,0] = self.tau_m_scale * INa_m_tau(V); tau[:,1]=self.tau_h_scale*INa_h_tau(V); ...
# mhas13.compute_gate_steady_states(V,S):   m_inf = INa_m_inf(V + self.v_half_shift); ...
# identity (scales=1, shift=0) => hooks bitwise-unchanged (parity test); step() left alone
```
#### Test Spec
- `tests/test_kinetics_axes.py::test_identity_parity` — scales=1,shift=0 ⇒ `compute_gate_*` outputs AND a 0-D-hook-driven AP identical to pre-change (≤1e-9).
- `::test_tau_m_MOVES_cv` — τ_m×2 changes **tissue CV by a nonzero, measurable amount** (via `cc_runner.run_1d_cable`) — guards against the "invisible to CV" failure (a CV delta of 0 FAILS this test).
- `::test_tau_m_decouples` — τ_m×2 changes the **dV/dt : CV ratio** (both move, ratio shifts) — the decoupling the architecture predicts.
- `::test_phas13_untouched` — PHAS13 model `compute_gate_*` unchanged by mhas13 kinetic scales.
#### Checklist
- [ ] Scaling in the HOOKS (not `step()`); identity default; PHAS13 unaffected; V5.3 untouched; axes registered in `config.py`.
#### Verify
```bash
conda run -n heart-conduction python -m pytest Optimizer/V1/tests/test_kinetics_axes.py cardiac_core/tests/ -q -k "phas13 or mhas13"
```
#### Exit Criteria
- [ ] `test_tau_m_MOVES_cv` passes (CV genuinely responds to τ_m); identity-safe; PHAS13/V5.3 unaffected.
#### Risk
Scaling in the wrong place (`step()`) → CV unaffected → silent corruption of Phase 3. `test_tau_m_MOVES_cv` is the specific guard; do not weaken it to "CV changes less".

### Step 2.3: Re-map with kinetics (P1b)
**Model**: opus
#### Read First — 2.1 map + 2.2 axes.
#### Why — answers open-Q1 for the kinetics branch (P1a cannot).
#### Implementation Spec — extend `feasibility_map.py` with the τ_m axis; re-plot feasible region.
#### Pseudocode — `for (gNa, tau_m, D): dvdt=cell(theta(gNa,tau_m)).dvdt_max; est=resolved_cv(...); feasible=...; plot region over (tau_m, D)`
#### Test Spec — smoke: map runs with the extra axis.
#### Checklist — [ ] kinetics feasibility recorded per (baseline,dx).
#### Verify — `conda run -n heart-conduction python Optimizer/V1/feasibility_map.py --baseline hipsc --kinetics`
#### Exit Criteria — [ ] Feasible regime identified (or documented infeasible even with kinetics → escalate to "change base model", architecture §strategy).
#### Risk — if infeasible even with kinetics + fine dx, that's a model-level finding (MHAS13-matured wrong for slow hiPSC).

### Phase 2 Verification / Exit / Cleanup
```bash
conda run -n heart-conduction python -m pytest Optimizer/V1/tests/ -q -k "feasibility or kinetics"
```
- [ ] Feasibility gate decided; kinetics (if built) identity-safe. float64; PHAS13/V5.3 untouched; figures in `media/`.
**-> Commit point.**

---

## Phase 3 (P2): Constrained-scalarization joint fit on the GP emulator

**Goal**: extend `joint_refiner.py` into the real joint fit — GP emulator on resolved CV, constrained scalarization, r*/dx constraint, block masking — over the feasible regime P2 identified. **If P2 (incl. P1b kinetics) concluded NO feasible regime exists**, Phase 3 degrades to a *documented infeasibility report* (Step 3.3's infeasibility path names the binding lock; escalate to the "change base model" strategy) rather than a fit — do not force a fit into an empty feasible set.
**Tier**: large
**Estimated scope**: rewire runner, add constraints + masking, swap NSGA-II→constrained scalarization, wire decision-space registry.

### Phase Context
`joint_refiner.py` already builds a GP emulator (`_build_training_data` ~3 sims/pt, `n_training=50`, `n_validate=5`) but rides legacy `tissue_runner.run_cv_measurement` (returns `None` when blocked) and runs bare NSGA-II. Rewire to `cc_runner`/`resolved_cv` (returns **NaN** when blocked → guard with `isfinite`, not `if cv else 50.0`). Train on resolved CV at r*/dx≥3. Mask (classify) the infeasible/block region rather than penalty-smoothing it.

### Step 3.1: Unified decision-space registry + apply()
**Model**: opus
#### Read First — `config.py:78-170` (registry, apply_scaling, bounds); `chip.py:35-52` (mesh D).
#### Why — the joint vector spans conductances/kinetics/D; one `apply(vector)→(scaled_model,mesh)` behind a registry is what makes it "joint" not three bolted stages.
#### Implementation Spec — `tuner/decision_space.py`: `AXES = [{name, subsystem∈{cond,kinetic,diffusion}, bounds, apply_fn}]` (imports the kinetic axes registered in `config.py` at Phase 2); `apply(vector, config) -> (model, mesh)`; constraint graph `{hard:[r*/dx≥k], soft:[√D, 2:1 warm-start]}`.
#### Pseudocode — `def apply(vec,cfg): model=_build_model(cond_subset(vec)); for ax in kinetic_axes: setattr(model, ax, vec[ax]); mesh=chip_mesh(D_long=vec.D_long, D_trans=vec.D_trans); return model, mesh`
#### Test Spec — `tests/test_decision_space.py::test_apply_roundtrip` — vector → model+mesh with expected scaled params & per-axis D; `::test_dtrans_free` — D_trans independent of D_long. **Caveat (architecture §6/open-Q7):** free D_trans only has *teeth* once CV_T is an *independent* target — today `PARKER` derives `CV_T=CV_L/ratio` and `chip.py:25,29,48` hard-code `D_trans=D_long/ratio²`, so a free D_trans fit to a *derived* CV_T relands near D_long/4. Either wire an independent CV_T target or note this in the record.
#### Checklist — [ ] registry routes each axis; D_trans free; soft warm-starts not hard ties.
#### Verify — `conda run -n heart-conduction python -m pytest Optimizer/V1/tests/test_decision_space.py -v`
#### Exit Criteria — [ ] flat vector ↔ (model, mesh) works.
#### Risk — Cm axis entangled with effective-D convention — leave Cm out unless architecture open-Q6 resolved.

### Step 3.2: Rewire emulator to cc_runner + resolved CV + block masking
**Model**: opus
#### Read First — `joint_refiner.py:74-246` (`_build_training_data`, `_build_emulator`, `refine_joint` INCLUDING the top-k validation loop at `:244-245` which also calls `run_cv_measurement`/`run_single_cell` — rewire those to `resolved_cv`/`cc_runner` too); `cv_estimator.py`.
#### Why — sims dominate cost; emulator keeps them out of the inner loop, but must train on RESOLVED CV and represent the block as a masked/infeasible region, not a `50.0` penalty the GP interpolates through.
#### Implementation Spec — replace `run_cv_measurement` with `resolved_cv`; `NaN`/`converged=False` → feasibility-classifier label (not a numeric target); GP trained only on feasible points; add an r*/dx≥3 constraint evaluated from `D/cv_resolved`. Budget note: ~150 sims × dx-ladder → warm-start from saved θ, active-learning refill near the feasible boundary.
#### Pseudocode — `for pt in design: est=resolved_cv(*pt); if not isfinite(est.cv_resolved) or not est.converged: label(pt, INFEASIBLE); else: Xtr+=pt; Ytr+=est.cv_resolved; gp=fit(Xtr,Ytr); feas_clf=fit_classifier(all_pts, labels)`
#### Test Spec — `tests/test_emulator.py::test_nan_guard` — blocked point does not poison the GP (no NaN in GP targets); `::test_feasible_only_training` — masked points excluded.
#### Checklist — [ ] isfinite guard; masking; r*/dx constraint; resolved-CV training.
#### Verify — `conda run -n heart-conduction python -m pytest Optimizer/V1/tests/test_emulator.py -v`
#### Exit Criteria — [ ] emulator trains on resolved+feasible CV; block masked.
#### Risk — out-of-support queries (thin training manifold vs full box) — cover the searched region in the training design (architecture open-Q8).

### Step 3.3: Constrained scalarization + surface infeasibility
**Model**: opus
#### Read First — `cell_fitter.py:49-59` (`_check_constraints`, reuse for AP bounds only); architecture §7.
#### Why — constrained scalarization surfaces "no feasible (θ,D) → refine dx / revise dV/dt / add kinetics" instead of silent dominated compromises.
#### Implementation Spec — objective = aggregate AP-morphology error; hard constraints CV_L/CV_T tol (new code), r*/dx≥k (new), + AP bounds (reuse `_check_constraints`); optimize on emulator; validate top-k on real `resolved_cv`; if feasible set empty → return an explicit infeasibility report (which lock binds).
#### Pseudocode — `cand = minimize(ap_error, s.t. feas_clf(x)==FEASIBLE and cv_tol(gp(x)) and rstar_dx(x)>=k and ap_bounds(x)); if empty(feasible): return InfeasReport(binding_lock); else: return validate_topk(cand, resolved_cv)`
#### Test Spec — `tests/test_joint_fit.py::test_reports_infeasible` — impossible target → infeasibility report naming the binding lock, not a fake fit; `::test_feasible_converges` — a known-feasible synthetic target is hit within tol.
#### Checklist — [ ] CV/r*/dx constraints new-coded; infeasibility explicit; top-k real-validated.
#### Verify — `conda run -n heart-conduction python -m pytest Optimizer/V1/tests/test_joint_fit.py -v`
#### Exit Criteria — [ ] joint fit returns either a validated θ* (CV_L,CV_T,APD,dVdt in tol, r*/dx≥3) or a lock-named infeasibility report.
#### Risk — emulator-vs-real drift — refit on validation drift (emulator-in-the-loop).

### Phase 3 Verification / Exit / Cleanup
```bash
conda run -n heart-conduction python -m pytest Optimizer/V1/tests/ -q
```
- [ ] Joint fit works or reports infeasibility honestly; no regressions. float64; V5.3/V5.4 untouched; shared logic in cardiac_core.
**-> Commit point.**

---

## Phase 4 (P3): Resolved chip mesh + reentry hand-off

**Goal**: refine the chip mesh dx to the resolved value, persist joint records/presets (with achieved r*/dx + dx-ladder), fix the preset export bug, hand off to the reentry campaign.
**Tier**: medium

### Phase Context
`chip.py:35-52` hard-codes dx=0.1mm; P1/P3 require ~0.02mm at the slow corner. `presets.py:119-128` KeyErrors when `export_lab_preset(engine="lbm")` reads `record["tissue"]["lbm"]` on a monodomain-only record.

### Step 4.1: Refined chip mesh + record/preset schema
**Model**: opus
#### Read First — `chip.py:35-62`; `presets.py:30-157` (record + export).
#### Why — the reentry campaign inherits this grid; it must resolve source-sink (r*/dx≥3).
#### Implementation Spec — `chip_mesh(dx_mm=0.02 default per feasibility)`; record schema gains `{achieved_rstar_over_dx, dx_ladder, kinetics, D_trans}`; `export_lab_preset` guards absent engine tissue (write the engine that exists, or skip with a warning).
#### Pseudocode — `m=chip_mesh(dx_mm=dx*); rec=make_record(theta*, D_long, D_trans, rstar_over_dx=..., dx_ladder=..., kinetics=...); export_lab_preset(rec, engine) => t=rec["tissue"].get(engine); if t is None: warn+skip else write`
#### Test Spec — `tests/test_chip_resolved.py::test_dx_resolves` — chip mesh at chosen dx gives r*/dx≥3 for the fitted θ*; `tests/test_presets.py::test_export_no_keyerror` — monodomain-only record exports without KeyError.
#### Checklist — [ ] dx set from feasibility; export guarded; schema extended.
#### Verify — `conda run -n heart-conduction python -m pytest Optimizer/V1/tests/test_chip_resolved.py Optimizer/V1/tests/test_presets.py -v`
#### Exit Criteria — [ ] resolved chip mesh + clean preset export; record carries r*/dx.
#### Risk — 0.02mm ≈25× cells → heavier reentry sweeps; document the cost in the hand-off note.

### Phase 4 Verification / Exit / Cleanup
```bash
conda run -n heart-conduction python -m pytest Optimizer/V1/tests/ -q
```
- [ ] Resolved mesh + presets; export bug fixed; reentry hand-off note written. float64; media convention for any figures.
**-> Commit point.**

---

## Final Cleanup
- [ ] float64 across all new tensors; no float32 leak.
- [ ] `Monodomain/Engine_V5.3/` and `Engine_V5.4/` read-only (backend port lives in `cardiac_core`/`Optimizer`).
- [ ] No duplicated ionic/CV code across engines — shared logic in `cardiac_core/`.
- [ ] Any new experiment dir has an `EXPERIMENT.md` backlinking this question + MASTER.md.
- [ ] Update README completion criteria checkboxes; update KNOWLEDGE with results.
- [ ] Archive the plan:
```bash
mkdir -p Research/Active/ionic_model_optimization/plans
cp Research/Active/ionic_model_optimization/PLAN.md "Research/Active/ionic_model_optimization/plans/$(date +%Y-%m-%d)_joint-ionic-conduction-tuning.md"
```

## Mutation Log

**EXECUTED 2026-07-10 — all 5 phases / 11 steps done on branch `engine-tuner-v2-joint`.**
Commits: `be21bfe` (P-1), `0c4e349` (substrate), `a95ac8a` (P0), `09a7044` (P1a/P1.5),
`f08950c` (P2 joint fit), `5497864` (P3), `bf3ac96` (docs). Optimizer non-slow suite
59 passed / 0 failed; cardiac_core ionic 11 passed.

- **MUTATED 2026-07-10**: Step 0.1 parity — the ≤1% target was met by matching PACING
  (n_beats=20 to steady state), not by re-anchoring: the V5.4↔cardiac_core delta was
  pacing history (9.35%@6→0.67%@20 beats), not the step()-vs-hooks formulation delta. No
  target re-anchor needed.
- **MUTATED 2026-07-10**: Steps 3.2/3.3 IMPLEMENTED as a NEW `tuner/joint_fit.py` (not an
  in-place mutation of `joint_refiner.py`) — the legacy joint_refiner + `pipeline.py`
  import it and are part of the retired sequential path; a parallel module avoids breaking
  the passing legacy suite. The GP-emulator PATTERN is reused per the plan.
- **INSERTED 2026-07-10**: three joint-fit fixes found via TDD, not in the plan —
  (a) candidates SOLVE D on the emulator (batched bisection) to hit each CV target (random
  D never lands on the thin "hits both CV" manifold); (b) GP inputs NORMALIZED (D~1e-4 vs
  g_Na~1 span 4 orders → RBF cannot fit CV otherwise — this was the true cause of an early
  all-infeasible result); (c) emulator_margin so candidate CV is well inside tol (drift
  buffer).
- **MUTATED 2026-07-10**: P1a/P1b feasibility map uses the fixed-dx secant + r*/dx≥3 filter
  (feasible ⇒ resolved ⇒ CV trustworthy) instead of a per-point resolved_cv LADDER —
  equivalent, ~Nx cheaper; the ladder is reserved for final top-k validation.
- **NOTE 2026-07-10**: P1.5 kinetics built before the P1a gate formally printed (the
  hiPSC diagnostic + first map rows already made infeasibility overwhelming) — the gate then
  confirmed conductance-only INFEASIBLE, so the kinetics work was warranted.
- **Step 1.3**: refined the NaN classifier from 2-way (over-depol/block) to 3-way
  (adds `no_capture`) — data showed the high-D NaN is sink overload, not over-depolarization
  (corrects architecture §4).
- **DEFERRED (gated heavy run, not code)**: the production joint fit → a real θ* on the
  resolved grid with kinetics (multi-hour). P1b showed τ_m alone lowers the CV floor but
  doesn't reach CV_T=2.6 at r*/dx≥3 at g_Na=0.5 — the full joint fit is the definitive test.
