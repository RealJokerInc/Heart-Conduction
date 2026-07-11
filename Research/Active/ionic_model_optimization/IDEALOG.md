# Ionic Model Optimization — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan this in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
**Engine Tuner V2 joint fit is BUILT + committed (branch `engine-tuner-v2-joint`), but the last
run's verdict was WRONG.** ⚑ THE MISTAKE: the fit enforced **`r*/dx ≥ 3`** as a hard feasibility
constraint. That rule is **SCS-specific** (LBM specular-same-cell wall / curvature,
[[boundary_conduction_speedup]]) — NOT a general resolution gate for a monodomain/HBB CV fit.
Applying it outside SCS produced the false "INFEASIBLE" (37/4000 candidates actually hit CV_T=2.6,
discarded by the filter). All the "conductance-only infeasible / kinetics required / CV_T is the
wall" conclusions are **withdrawn** as artifacts. Nuance: r*/dx is real *for SCS* — not a permanent
rejection; the next run just uses HBB, where it doesn't apply. The MACHINERY is sound (P-1 backend
unification, bracket-down secant, kinetics axes, GP-emulator joint fit with block-masking + input
normalization + D-solve + constrained scalarization, chip/preset/export fixes).

## Next Step
**Re-run the joint fit on LBM + HBB, with r*/dx dropped** (`require_resolved=False`, already wired),
targeting ONLY the real EP numbers: CV_L=5.2, CV_T=2.6, APD=350, dV/dt, 2:1 anisotropy — take
whatever (θ, kinetics, D) reaches them. (For SCS work later, use LBM SCS as originally intended;
r*/dx belongs there.) Keep the honest open question: does a plain low-D conduction *block* exist in
LBM+HBB? If yes, lower excitability (kinetics / lower g_Na) is the lever — but do NOT assert
infeasibility until run on the right engine without the filter. Secondary: the warm-start θ
(`presets/chip_hipsc.json`) is a V5.4-era fit and gives APD≈276 (not 350) on cardiac_core → the
ionic AP still needs a cardiac_core re-tune (dV/dt≈113 is fine; APD short).

## Thread

> ⚑ **DECONTAMINATION NOTE (2026-07-11).** The 2026-07-10 execution entries below reach scientific
> conclusions — "conductance-only INFEASIBLE", "kinetics is required", "CV_T=2.6 is the wall /
> excitability floor", the P1a "gate = infeasible", and the production "INFEASIBLE" — that are all
> **WITHDRAWN**. They are artifacts of a wrongly-applied **`r*/dx ≥ 3`** constraint (SCS-specific,
> not a general fit gate; see Current Direction → THE MISTAKE). The **engineering** in those entries
> stands (backend unification, secant fix, kinetics axes, joint-fit machinery, fixes); the
> **scientific verdicts do not**. Read them as "how the mistake happened", not as findings.

### 2026-07-10: EXECUTION STARTED — PLAN Phase 0 (P-1 backend unification) DONE + committed
Greenlit ("plan.md is finished, begin implementation"). Branch `engine-tuner-v2-joint`
(off `textbook-website-refresh`, itself off the implemented `engine-tuner-cardiac-core`
substrate). **Phase 0 (P-1) complete, committed `be21bfe`.**
- `tuner/cell_runner_cc.py::run_single_cell_cc` — cell AP on cardiac_core via a 0-D
  `run_monodomain` on a uniform strip (all cells stimulated → flat field → diffusion
  inert), driven through the SAME hook-based Rush-Larsen path the tissue-CV runner
  (`cc_runner`) uses. Multi-pulse pacing patched into the mesh stimulus
  (`num_pulses`/`bcl`; create_cardiac_mesh hard-codes 1 pulse). So dV/dt (cell) and CV
  (tissue) now come from ONE model/path → the P1.5 kinetics axis will be identifiable.
- `tuner/cell_result.py` — CellResult extracted (backend-neutral) so the default AP
  path imports without pulling `cardiac_sim`; `cell_runner` re-exports it. `config.
  ionic_backend` flag (default `cardiac_core`); `cell_fitter._evaluate_batch` routes on
  it, lazy cardiac_sim imports (V5.4 kept for parity only).
- **KEY FINDING — the V5.4↔cardiac_core parity delta is PACING HISTORY, not the
  step()-vs-hooks (constf1/constfCa) formulation delta I first suspected.** APD Δ
  9.35%@6 beats → 1.80%@12 → **0.67%@20**; dV/dt 8.33% → 1.24% → **0.76%**; V_rest
  0.06% throughout; V_peak 0.56%@20. Both backends converge to a common steady state →
  the port is FAITHFUL and parity ≤1% holds once paced to steady state (~20 beats). The
  plan's Risk ordering ("fix multi-pulse pacing FIRST before blaming the model") was
  right. No re-anchor needed; cardiac_core is the single reference going forward.
  Downstream cost note: 20-beat cell eval ≈100k CN steps (~min); the feasibility map
  can use fewer beats where absolute APD precision isn't the gate.
- Also fixed a pre-existing broken assertion in `test_cell_fitter` (slow, never run in
  the non-slow baseline): BO's descending `q_batch=min(4, n_iter−i)` gives 4+3+2+1=10
  candidates, not n_initial+n_iterations. Baseline non-slow suite: 45 passed.
### 2026-07-10: PLAN Phase 1 (P0 discriminators) DONE + committed (`0c4e349`, `a95ac8a`)
- **1.1 secant bracket-DOWN** (de-duplicated to the single `cc_runner.fit_D_for_cv`;
  `run_chip_fit` delegates). On a non-propagating start it now brackets D DOWN into the
  window (chip window is BELOW D0=1e-3) not the old ×4-up-bump; returns (NaN,NaN) on a
  genuine block, never a fake D. Calib: target CV=6 → D=5.1e-5 (was returning 0.004,NaN).
- **1.2 `cv_estimator.resolved_cv`** — dx-ladder (fix θ,D,dt; vary dx), every rung must
  sit at **r*/dx≥3** (below 3 the CV is grid-corrupted / sign-inverts, arch §4), else
  converged=False. Verified: resolvable (D=2e-4 fine ladder → converged, CV plateau 12.0,
  r*/dx 4→17) vs blocked (D=5e-5 coarse ladder → converged=False, r*/dx<3). **Cost note:
  the fine dx=0.001 ladder is heavy (~8 min for 2 tests) → Phase 2/3 must call it
  economically** (emulator training shell, not an inner loop).
- **1.3 hiPSC-window diagnostic — KEY FINDINGS:**
  - At the saved hiPSC θ (`g_Na=0.5`), **dx=0.1 mm the propagating window is a RAZOR'S
    EDGE: only D=1e-4 propagates** (CV=5.31 ≈ CV_L target 5.2), and even that at
    **r*/dx=1.88 (<3, grid-corrupted)**. CV_T=2.6 needs ~4× lower D → deep in the block.
    ⇒ conductance-only feasibility at dx=0.1 mm for hiPSC is ~NIL; **strongly anticipates
    lock-3 (finer dx REQUIRED)** — Phase 2's map will test dx∈{0.1,0.05,0.03,0.02}.
  - **The HIGH-D NaN is SINK OVERLOAD ('no_capture': Vmax stays sub-threshold −41…−67 mV,
    the sink drains the stimulus before it fires), NOT over-depolarization** — this
    **CORRECTS architecture §4's hypothesis**. Low-D NaN is the classic source-sink block
    (Vmax fires +15…+25 but the wave dies). Figure: `media/ionic_model_optimization/
    images/2026-07-10/hipsc-window_02.png`. (`run_1d_cable` gained `return_vmax`.)
**Next: P1a (Phase 2) — conductance-only feasibility map over (g_Na,D)×dx×dV/dt_target;
GATE decides whether P1.5 kinetics is needed. Expect infeasible at dx=0.1 mm per 1.3.**

### 2026-07-10: PLAN Phases 2–4 IMPLEMENTED + committed (`09a7044`,`f08950c`,`bf22fbd`)
All remaining code written + tested (synthetic where sims are prohibitive); P-1/P0 plus
these = the full 5-phase plan. Commits on branch `engine-tuner-v2-joint`.

**P1a feasibility map (`feasibility_map.py`) — GATE = conductance-only INFEASIBLE.**
For each (g_Na, dx) the fixed secant finds the D hitting CV_T, then checks r*/dx≥3.
Result (hiPSC, CV_T=2.6): infeasible at EVERY dx {0.1,0.05,0.03,0.02 mm}. g_Na=0.5
floors at **CV≈5.86** even when fully resolved (r*/dx=10.7 @0.02 mm); g_Na=0.15/0.30
**BLOCK** (nan) at all dx. dV/dt by g_Na = 60/83/109/159 — the widened floor g_Na=0.15
still can't both slow CV and stay resolved. **The source-sink floor (~5.8 cm/s at
g_Na=0.5) is the wall: reducing g_Na to slow CV just blocks → conductance scaling
cannot reach CV_T=2.6.** ⇒ kinetics required. Fig `media/.../feasibility-hipsc_01.png`.
Design note (vs plan): the map uses the fixed-dx secant + r*/dx≥3 filter (feasible ⇒
resolved ⇒ CV trustworthy), not a per-point ladder — equivalent, ~Nx cheaper.

**P1.5 kinetics (`cardiac_core/ionic/mhas13/model.py`) — built + validated.** Na knobs
`tau_m/h/j_scale`, `v_half_shift` on the MHAS13 INSTANCE, applied in the gate HOOKS
(compute_gate_*), NOT step() — so the tissue solver sees them. Identity default → hooks
bitwise-unchanged (cardiac_core 11 ionic tests green; PHAS13 untouched). Tests: identity
parity, PHAS13-safe, **τ_m MOVES cv** (the guard — CV genuinely responds), τ_m
decouples dV/dt:CV (dV/dt more sensitive than CV — the predicted decoupling; needed
τ_m×3 + fine CV sampling to clear quantization noise). `cc_runner`/`cell_runner_cc` gain
a `model=` passthrough so AP + CV use ONE kinetic-scaled model.

**P2 joint fit (`decision_space.py` + `joint_fit.py`) — built + validated (synthetic).**
- `decision_space`: one `apply(vector)→(kinetic-scaled model, per-axis mesh)`;
  D_trans FREE; hard r*/dx≥k, soft √D/2:1 warm-starts.
- `joint_fit`: GP-emulator pattern on the P-1 backend. Three load-bearing fixes found
  by TDD: (1) **block region MASKED** (NaN never enters a CV GP target — the isfinite
  guard, fixing joint_refiner's 50.0-penalty cliff); (2) **candidates SOLVE D on the
  emulator** (batched bisection) to hit each CV target — random D never lands on the
  thin 'hits both CV' manifold; (3) **GP inputs NORMALIZED** — D~1e-4 vs g_Na~1 span 4
  orders, so the RBF cannot fit CV without it (this was the real cause of an early
  all-infeasible result). Constrained scalarization (min AP err s.t. CV tol + r*/dx≥k +
  dV/dt band) SURFACES infeasibility (names the binding lock). Legacy joint_refiner +
  pipeline left intact (retired with the sequential path). Note for the REAL high-dim
  fit: train the CV GP on a REDUCED CV-relevant feature set (g_Na + kinetics + D), not
  the full 16-dim vector — CV-irrelevant conductances add GP noise (arch open-Q8).
- Tests (synthetic oracle, no sims): NaN-mask, feasible-only training, known-feasible
  target converges (CV in tol at r*/dx≥3), infeasible→InfeasReport naming the lock.

**P3 chip/presets (`chip.py`,`presets.py`) — built + tested.** chip_mesh default dx
0.1→**0.02 mm** (RESOLVED_DX_MM; lock-3, ≈25× cells, reentry inherits). `make_record`
gains `kinetics` + resolved-grid provenance (achieved_rstar_over_dx, dx_ladder) in
`validation`. Fixed the pre-existing `export_lab_preset(engine="lbm")` KeyError on
monodomain-only records (falls back to the present engine + warns).

**P1b re-map with kinetics (DONE) — τ_m is a real lever but INSUFFICIENT ALONE.** Sweeping
τ_m at g_Na=0.5 (`feasibility-hipsc_02.png`): τ_m LOWERS the CV floor at the tissue level
(dx=0.05 mm: 5.81→**4.31** @τ_m=1.5; dx=0.03 mm: 5.86→5.60→**5.48** @τ_m=1.0/1.5/2.0),
confirming the decoupling propagates to CV — but it does NOT reach CV_T=2.6 at r*/dx≥3, and
high τ_m (≥2.0 @dx=0.05; ≥3.0 @dx=0.03) BLOCKS. This is a 1-axis slice at one g_Na, so it
does NOT prove infeasibility — the definitive test is the **full joint fit** (τ_m+τ_h+τ_j+
v_half + g_Na + D jointly, which `joint_fit.refine_joint_cc` does). Escalation if the joint
fit still can't reach CV_T=2.6 at r*/dx≥3: combine kinetic axes / revisit whether
CV_T=2.6-at-resolved-grid is physical for MHAS13-matured (the architecture's "change the
base model" path).

**STATUS: all 11 plan steps implemented + tested; full Optimizer non-slow suite 59 passed /
0 failed (no regressions); cardiac_core ionic 11 passed. Committed on `engine-tuner-v2-joint`
(be21bfe, 0c4e349, a95ac8a, 09a7044, f08950c, bf22fbd + docs).**

### 2026-07-10: PRODUCTION joint fit RAN → INFEASIBLE (the definitive answer)
`run_joint_fit.py --baseline hipsc` at dx=0.02 mm, 16 axes (tier-2 + 4 kinetics + free
D_long/D_trans), 49 training points (9 propagating warm-start seeds + 40 Sobol), GP
emulator with the **reduced CV feature set** + **warm-start seeding** (both added so the
result isn't a sparse-sampling/GP-noise artifact). Verdict: **INFEASIBLE**, and the
constraint counts (of 4000 D-solved candidates) name the wall precisely:
`feas=2072, cvL=538, cvT=37, dvdt=2361, r*/dx=4000`.
- **The binding constraint is CV_T=2.6**: only **37/4000 (0.9%)** of (θ, kinetics) combos
  can be driven that slow without blocking, and NONE of those also hit CV_L=5.2 + the dV/dt
  band. r*/dx passed 4000/4000 (D is solved to the target CV → all resolved), so resolution
  is NOT the wall — the auto-label "dx/resolution" was a heuristic mislabel (fixed
  `_binding_lock` to report from the constraint counts + an anisotropy check).
- **Conclusion: kinetics is NECESSARY but NOT SUFFICIENT** — it lowers the CV floor (P1b:
  5.8→4.3) but the joint fit still cannot reach CV_T=2.6-at-r*/dx≥3 simultaneously with
  CV_L=5.2 for MHAS13-matured. This is exactly the sharpest tension the audits flagged
  (architecture §9 lock-1/3), now confirmed on a resolved-grid joint fit.
- **Escalation (for the user / reentry campaign):** (1) reconsider whether CV_T=2.6-at-
  resolved-grid is physical for MHAS13-matured (the "change the base model" path — a
  slower-upstroke hiPSC base may be needed); (2) relax a lock (wider dV/dt band, CV_T
  slightly >2.6, or revisit the 2:1 anisotropy — cross-construct 2.1±0.8); (3) a larger
  training budget + active-learning refill near the CV_T boundary to tighten the verdict
  (this first run kept n_training modest). Record: `presets/chip_hipsc_joint_INFEASIBLE.json`.
- **The MACHINE WORKS**: the joint fit surfaced infeasibility honestly with a named,
  quantified binding lock instead of a fake θ* — which is precisely what the V2
  architecture was built to do (vs V1's silent garbage `D=0.004, CV=nan`).

### 2026-03-15: Literature survey reveals single-AP fitting is fundamentally non-unique
Reviewed 6 key papers on ionic model parameter optimization. Groenendaal 2015 is the critical finding: fitting 9 parameters to a single AP gives near-perfect waveform match (SSE < 0.01 mV^2) but parameters converge to *wrong values*. Adding stochastic stimulation, voltage-clamp data, and iterative refinement drops prediction error by 3 orders of magnitude. This established our requirement for multi-objective optimization with tissue-level targets, not just AP shape.

### 2026-03-15: Method selection — BayesOpt for V1, HMC deferred
Compared NSGA-II (Pouranbarani), Bayesian HMC (Nieto Ramos), ABC-SMC, GP emulator (Coveney), GA (Groenendaal), and gradient-based (Zhang). Chose BayesOpt with qLogNEHVI via BoTorch for V1: sample-efficient, GPU-native, handles multi-objective without scalarization. HMC deferred to V2 — it gives full posteriors but requires differentiable model and was only tested up to 13 params.

### 2026-03-15: Optimizer V1 pipeline designed and first implementation
Four-phase architecture: (1) Cell fitter — batched single-cell BayesOpt, (2) Tissue fitter — CV via monodomain simulation, (3) Joint refinement — GP emulator, (4) Validation suite. Renamed Paci2013 to PHAS13 for clarity. Achieved 10x speedup via batching ionic steps, subcycling, and analytical CV warm-start.

### 2026-03-16: MHAS13 created — quiescent hiPSC-CM via TTP06 IK1 injection
All 8 published hiPSC-CM ionic models beat spontaneously. The maturation pathway (Paci 2013 -> PHAS13 -> MHAS13) achieves quiescence by injecting TTP06 IK1 at the Verkerk 2019 critical GK1 value plus If suppression (g_f=0). This gave a physiologically mature model: V_rest=-83.7 mV, APD=349ms. The model became our primary optimization target.

### 2026-03-16: First optimizer run — APD matches, but dVdt reveals a tension
MHAS13 through optimizer: APD=347ms (target 350ms, 0.9% error). But baseline dVdt=132 V/s, far from the original 25 V/s target derived from mature CM literature. This was the first sign that dVdt targets need to match the model's physiology, not generic literature values.

### 2026-03-16: Bidomain pipeline validates monodomain results
Ran MHAS13 through bidomain pipeline: APD=349ms, CV=15.8 cm/s. Created tissue_runner_bidomain.py with D_eff decomposition (D_e/D_i=3.597 ratio from cv_shared.py). Monodomain vs bidomain D_eff values agree within 6% (0.000447 mono vs 0.000422 bidomain), confirming the D_eff approximation works for insulated tissue.

### 2026-03-17: Iteration 2 — constraints, tier 2, and the dVdt lesson
Three breakthroughs in one session. (1) dVdt constraint at 60 V/s made 72/74 candidates infeasible; relaxing to 120 V/s gave 41/74 feasible. The constraint must match the model's inherent physiological range. (2) Tier 2 (10 params) works: kNaCa, PNaK, g_pCa, VmaxUp all participate in the solution, with PNaK=0.5 as a new downscaled knob. (3) Secant CV refinement crushes Newton: 50.6% initial overshoot converges to 2.4% error in 3 steps using actual simulation data for the derivative estimate.

### 2026-03-17: APD-dVdt tradeoff is a fundamental model property
MHAS13 cannot simultaneously achieve APD=350ms and dVdt=25 V/s with conductance scaling alone. The hiPSC-CM model has fast sodium kinetics relative to its AP morphology. Achieving 25 V/s would require modifying Na channel gating time constants, not just conductances. This is not a failure of the optimizer — it is a property of the model. The dVdt target should be revised to ~100 V/s for MHAS13.

### 2026-03-17: IKr/IKs compensation identified as the core degeneracy
The key parameter degeneracy: IKr and IKs overlap temporally during the AP plateau and can trade off against each other without changing AP shape. Single-rate pacing cannot separate them, but their ratio matters for rate-dependent behavior and drug response. Breaking this requires multi-rate pacing (restitution curves) or paired waveforms with selective channel block.

### 2026-06-29: Cross-plan with `geometry_induced_reentry` — Engine Tuner → cardiac_core (incl. LBM)
The reentry question needs the tuner to fit a **Kit Parker tissue-chip EP set** (CV: NRVM 9.33 / hiPSC-CM 5.2 cm/s; λ = CV·APD, ≈ 1.5–4 cm — *not* MacQueen's 5 mm geometry artifact) and run it on the **LBM** engine (its primary), cross-validated on monodomain/bidomain. This makes the long-open completion criterion "Cross-engine validation (V5.4 vs Bidomain vs LBM)" concrete and puts the **LBM adapter** + cross-engine validator (already designed in `Optimizer/improvement.md` §4–5) on the critical path. **End goal restated: a finished Engine Tuner adapted to `cardiac_core` (monodomain + bidomain + LBM)**, with the reentry chip-EP fit as the first real application. Ionic target = **MHAS13** (already mature/quiescent — no TTP06/Paci detour). Shared **PLAN.md** to be generated here (build owner); plan-only for now — no coding (user gate). The reentry side owns the *application* (mesh fitted to chip: L=16 mm, dx=0.1 mm; obstacle sweeps); this question owns the *build*.

**Cross-plan `PLAN.md` created + audit-converged (2026-06-29).** 6 phases (Phase 0 cardiac_core tuning seam → 1 cc_runner → 2 chip mesh+targets → 3 fit both baselines → 4 cross-engine → 5 reentry hand-off). Four adversarial-audit iterations (issues 18→13→9→3), CONVERGED at 0 critical / 0 high. Load-bearing findings the audits forced: (1) cardiac_core's OO `CardiacSimulation` methods are stubbed → route through the **functional** `run.py` API; (2) cardiac_core can't yet inject θ_ionic/D → **Phase 0 seam** = pass a pre-built (tuner-scaled) `IonicModel` instance + bake D into the mesh; the **LBM api factory** (`.lower()` rebuild, api.py:1338/1353) is the one that must be patched; (3) fit is **anisotropic ~2:1** (user domain-correction — aligned hiPSC/NRVM tissue is anisotropic; Bursac & Parker 2002 ratio ≈2.1) using the tuner's *existing* `cv_longitudinal/cv_transverse` + `D_long/D_trans` fields; (4) `improvement.md`'s `tau_from_D` prose is wrong — correct is `tau = 0.5 + D·dt/(cs²·dx²)`.

**Anisotropy domain-correction + 2nd convergence (2026-06-29).** Initially simplified to isotropic (from an early audit) — **wrong physics**; reverted to 2:1 anisotropic. The real cost this exposed: cardiac_core's `lbm()` is **BGK-scalar-only** and rejects non-isotropic D, and the LBM engine has **no wired per-axis MRT** (`mrt_collide_d2q5` exists but unwired; only `lbm_step_d2q9_mrt`). So **Phase 0 Step 0.2 = genuine engine-level MRT work** (LBMSimulation collision selector + per-axis rates from `tau_tensor_from_D`, rates `s=1/τ`; D2Q9-MRT recommended) — now the single largest item in the plan. Two more audit iters (5–6) re-converged the per-axis/MRT mechanism: 0 critical / 0 high, verdict CONVERGED. λ-vs-chip tension (λ≈18–33 mm > 16 mm chip) remains flagged as the reentry application's problem, not this fit's.

**IMPLEMENTED 2026-06-30 (branch `engine-tuner-cardiac-core`, 8 commits, ~30 tests).** All 6 phases executed + tested:
- **P0**: `create_cardiac_mesh(D_yy=)` + `lbm()` instance seam; **D2Q9-MRT anisotropy** in `LBMSimulation` (vendored + upstream) + `lbm()` routing; dx≠dt diffusion benchmark proves the `s=1/τ` mapping (≤8%).
- **P1**: `cc_runner.py` — CV via cardiac_core functional API + scaled-instance seam (mono/bidomain/lbm); CV∝√D verified. *(tissue_fitter rewire deliberately deferred — would perturb the passing legacy suite; cc_runner is the parallel new path.)*
- **P2**: `chip.py` — 161² Parker chip mesh + anisotropic PARKER_NRVM/HIPSC targets (CV_T=CV_L/2, dvdt_max_upper=120).
- **P3**: `presets.py` Tier-1 records + `run_chip_fit.py` (dual-axis secant via cc_runner); smoke-tested end-to-end. Full BayesOpt fit GATED (`main()`).
- **P4**: `cross_engine.py` validate + recalibrate_lbm; observed mono↔bidomain CV_T ~12%, mono↔lbm ~29% (≈ expected LBM offset).
- **P5**: `export_lab_preset` (Tier-2 YAML) + `_SCHEMA.md` extension + `run_chip_baseline_lbm.py` (LBM planar-wave chip baseline → reentry hand-off).
- **Key engineering finding**: effective-D meshes need **chi=1.0** (the FDM operator divides by chi; chi=1400 silently kills propagation, Vmax→123 mV). No regressions (32 upstream LBM + cardiac_core suites green). Remaining: full gated fit run + (optional) tissue_fitter rewire.

### 2026-06-30: dt as a 4th tunable, guided by the lateral boundary-speedup number
Mid-session redirect (before running the gated fits): add **one more fitting
parameter** to the chip fit. Disambiguated two dimensionless numbers from
`boundary_conduction_speedup` — the user wants the **lateral boundary speedup**
(β = D·dt/dx² ⟺ τ = 0.5 + 3β, the side-wall crescent; CV-inert, BC+dt-driven), NOT the
source-sink `dx/r* = dx·CV/D` (geometry/CV-driven, deferred to the reentry app). The
knob is **dt** (free; D pinned by CV, dx by chip → dt is the only β dial). Key reality
checks the user supplied: (1) "dt is free"; (2) no meaningful curvature metric yet, so
**don't fit to curvature — use the number as a guide**; (3) dt is shared (it also moves
ionic tuning / APD / CV), so it can't be a clean post-hoc boundary knob. Resolved by
implementing a **guide, not an optimizer**: `chip.boundary_number(D,dt,dx,bc)→{beta,tau,
bc,regime}` (matches the engine `tau_from_D`), surfaced in every Tier-1 record (`boundary`,
per-axis) and the LBM baseline output at the actual run dt. β∝D so dt is per-baseline.
Avoided over-engineering (the user explicitly stopped an outer-dt/inner-refit design as
premature without a metric).

**FINAL CHECK (2026-06-30, same session) — the regime is BC-specific, and the chip's BC
is HBB.** User: "final check. boundaries, ionic engine." Traced cardiac_core: the chip
LBM wall is `apply_neumann_d2q9` = **full halfway bounce-back (HBB)**, and HBB is
**forward at ALL τ — no speed-up ever** (the inverse crescent is same-cell-specular-
specific). So the first version of the guide (labelling τ≲0.67 as "wall speed-up") was
**wrong for the chip's actual BC**. Fixed: `boundary_number` is now **BC-aware**
(default `hbb` → truthful "forward, no speed-up"; `specular` → the flip thresholds;
`zero` → flat). **Root-cause diagnosis (answering "are we miswired?"): NO — the tuner
uses `run_lbm` correctly; it's an ENGINE GAP.** `run_lbm`/`lbm()` have no LBM BC param,
`LBMSimulation` is HBB-only, and the specular/α-blend op (`apply_combined`) lives ONLY
in `boundary_conduction_speedup` research scripts — never ported into any engine src.
→ The wall speed-up is **unreachable until specular/α-blend is wired into the LBM engine**
(3-layer port + selector + `lbm()`/`run_lbm` plumbing, mirroring the Phase-0 MRT pattern).
**Ionic-engine coupling confirmed:** the LBM step integrates Rush-Larsen with the same
`self.dt`, so the dt knob perturbs APD/upstroke (bounds usable dt). 8 chip tests green
(BC-aware regimes + HBB-never-speeds-up + engine-τ consistency), no regressions.

### 2026-07-02 (cont.): sequential fit is architecturally BROKEN → JOINT tuning decision + audit
The GPU overnight run was killed (200 iters + degenerate spontaneous-CL objective → >10 h, no
records; also the shared GPU got a colleague's 29 GB Jupyter kernel — left untouched, ran on CPU
instead). Relaunched lean on CPU (n_iter=40, dropped the CL objective, dt-visible): NRVM cell fit
converged clean (APD err 3.1 ms, dV/dt 0.4 V/s) — **but the tissue leg produced garbage**
(`D_long=D_trans=0.004`, `CV=nan`). Cable D-sweep diagnosed a **source-sink `r*/dx` block**: at
dx=0.1 mm the cable only propagates D∈~[5e-5,1e-4] (CV 5–7); below that `r*=D/CV<dx` → block, so
hiPSC **CV_T=2.6 is unreachable** at chip dx. **User's diagnosis (correct):** we're tuning the
ionic engine and conduction **independently/sequentially** — cell fit θ→(APD,dV/dt), then frozen-θ
secant D→CV. Since `CV~sqrt(D·excitability(θ))` and **G_Na is shared** between dV/dt (cell) and
CV/r* (tissue), the frozen-θ secant can only push D down into the block; it cannot trade G_Na↓+D↑
to hit slow CV on-grid. **Decision: ionic + conduction must be tuned JOINTLY (never sequential)** —
one optimizer over θ+D, tissue-in-the-loop, with an `r*≥k·dx` resolvability constraint. This is
the deferred "Joint refinement" criterion, now on the critical path. Documented:
`Optimizer/V1/JOINT_TUNING_ARCHITECTURE.md` (proposal + 8 open questions). **Running `/audit` on it
next.** hiPSC cell fit left running for its θ warm-start; both tissue records superseded.

### 2026-07-02: /audit on the joint-tuning proposal — necessary but NOT sufficient (12 issues, 2 crit / 4 high)
Adversarial Opus audit of `JOINT_TUNING_ARCHITECTURE.md` (cross-referenced the tuner code + source-sink
findings). Verdict: joint tuning is the right direction but is **one of four coupled fixes**, and the
slow hiPSC **CV_T=2.6 cm/s at dx=0.1 mm is unreachable** under current constraints. Key findings:
(CRIT-1) **dV/dt=110 target structurally forbids the G_Na↓ trade** — at dV/dt≈110 g_Na≈0.83× → CV_T=2.6
lands at D≈1.6e-5 → r*/dx≈0.62 → block even when joint. Must revise dV/dt to physiological hiPSC ~20–50
V/s (the open README criterion, now critical-path). (CRIT-2) **dx refinement unavoidable** — the reentry
campaign needs r*/dx≳3 to *resolve* source-sink; CV_T=2.6 at r*/dx≥3 ⇒ g_Na≈0.17× (< the 0.5 bound →
infeasible). k=1 only marginally propagates on a grid that doesn't resolve the physics → **k=1 tuning is
fitting CV to a numerical artifact.** (HIGH) my failure diagnosis was slightly off — the garbage record is
the secant's **×4 up-bump init failure** (D0=0.001 above window→nan→0.004→nan→fallback), NOT a down-secant
reaching the block; anisotropy makes **CV_T block first** (r*_trans = r*_long/2); the diagnostic sweep was
at **NRVM θ not hiPSC**; adding CV creates a **G_Na–D degeneracy** that only dV/dt breaks (catch-22), and
conductance scaling alone can't decouple dV/dt from CV — needs **Na *kinetics* (gating τ), absent from
tier-2**. (MED) D=1e-3 nan unexplained (maybe CV-measurement artifact → window may be wider); dead
`TISSUE_PARAMS` D_trans lower bound (2.5e-5) sits in the block. **Method recommendation: constrained
scalarization** (minimize AP-morphology error s.t. hard CV_L/CV_T/r*/dx≥k + V_rest/V_peak/dV/dt), NOT
4-obj qNEHVI — cheaper, reuses `_check_constraints`, and surfaces infeasibility explicitly.

**Revised plan (post-audit): three locks must open together** — (1) revise dV/dt target; (2) widen g_Na /
add Na kinetics; (3) refine dx (r*/dx as a first-class precondition, ~0.1→~0.03 mm). Method → constrained
scalarization. Sequenced: **P0** cheap discriminators (fix secant ×4-up-bump→bracket-down [may rescue CV_L];
re-sweep cable at hiPSC θ; settle D=1e-3 nan) → **P1** feasibility map (sweep g_Na×D at dx∈{0.1,0.05,0.03}
vs CV_T=2.6/dV/dt/r*/dx≥3, *plot* feasible region) → **P2** build the constrained joint fit only where P1
proves feasible → **P3** refine chip mesh dx in `chip.py` for the reentry hand-off.

**CPU fit COMPLETED** (both baselines, ~35 min each): cell fits converged (APD err 3.1 ms, dV/dt 0.4 V/s),
θ saved as warm-starts (`presets/chip_{nrvm,hipsc}.json`); **both tissue fits garbage** (D=0.004, CV=nan —
block confirmed on BOTH). LBM baseline crashed on a **separate pre-existing bug**: `export_lab_preset(
engine="lbm")` reads `record["tissue"]["lbm"]` but the monodomain fit only writes `"monodomain"` → KeyError.

### 2026-07-10: JOINT_TUNING_ARCHITECTURE audit-CONVERGED (3 iters) → blueprint next
Ran the revise→audit→converge loop on `Optimizer/V1/JOINT_TUNING_ARCHITECTURE.md` (Opus auditor,
read-only, cross-referencing code+research each pass): **iter1** 0c/2h/7m/4l (wrong cost argument;
ignored existing `joint_refiner.py` GP-emulator; wrong CFL story — solver is implicit CN; constraint
table self-contradiction) → **iter2** 1c/3h/5m/4l (⚑ CRITICAL two-backend blocker: cell fit runs
`cardiac_sim`/V5.4, tissue runs `cardiac_core` → kinetics unidentifiable; kinetics file path didn't
exist; emulator overclaim; wrong g_Na/dx numbers) → **iter3 CONVERGED 0c/0h** (all fixes verified
true against code, 1m/3l folded). Settled architecture (details in the doc): **three-leg** =
resolution shell (dx/dt resolved-not-fit via convergence-extrapolating CV estimator) ⊃ constraint
graph (known-true pair relations; anisotropy is SOFT) ⊃ physical joint fit ({θ, kinetics, D_long,
D_trans free, Cm}) via **constrained scalarization on a GP emulator** (extend `joint_refiner.py`).
**Plan skeleton P-1→P3**: P-1 backend unification (blocker) → P0 discriminators (secant bump, hiPSC-θ
sweep, high-D nan) → P1a conductance-only feasibility map → P1.5 kinetics model change (gated) → P1b
→ P2 emulator joint fit → P3 refine chip dx (~0.02 mm). **Three coupled locks** (dV/dt target
CONTESTED vs README, g_Na floor ≤0.17, dx ~0.02 mm) + necessary-not-sufficient.

**PLAN.md generated + audit-CONVERGED (3 iters, 2026-07-10).** `/blueprint` → PLAN.md (5 phases, 11
steps, cold-start format) superseding the completed cross-plan (archived). Audit-converge loop:
**iter1** 1c/1h/6m/4l — ⚑ CRITICAL: kinetics scaling was placed in `MHAS13.step()`, but the tissue
Rush-Larsen solver drives the model via the `compute_gate_*` HOOKS and never calls `step()` → the
axis would be invisible to CV (re-creating the two-backend bug). Fixed: scale in
`compute_gate_time_constants`/`compute_gate_steady_states`; cell AP driven via a 0-D `run_monodomain`
(same hook path). **iter2** 0c/1h/1m/2l — Step 0.1 needed multi-pulse pacing-to-steady-state (parity)
+ correct metric arg order / V-node reduction. **iter3 CONVERGED 0c/0h** (all fixes code-verified; 2
residual minors folded: save_every, measure_peak arg). **Both docs now audit-converged: architecture
(3 iters) + plan (3 iters).** Plan is execution-ready — **hard gate: no implementation before explicit
"go".** Next physical step when greenlit: P-1 (backend unification).

### 2026-07-02: engine gap CLOSED (user fixed cardiac_core) + overnight fit FIRED
User implemented the specular BC in cardiac_core over 2026-07-01/02 (commits `40cd2ca`
P1 run_lbm/simulate forward boundary+alpha; `1dda8f6` C1 MRT/per-axis wall modes; Phase
3–5 hardening). I re-checked both gaps: (1) `run_lbm(boundary=,alpha=)` forwards ✓;
(2) collision-gate removed + new `lbm_step_d2q9_mrt_wall` → **specular works on the
anisotropic MRT chip** ✓. Smoke via `run_lbm` on an anisotropic chip: specular changes
the field 1.38 mV, wall-localized (1.38 wall vs 0.68 interior), combined(α=0.5)
intermediate (0.67) = correct α-blend. 67 cc LBM tests green. Aligned the guide's BC
vocab to the engine's WALL_MODES names (9 chip tests green). **Green light → FIRED the
overnight run** (`run_chip_fit.py` full BayesOpt EP fit → Tier-1 records for nrvm+hipsc,
then `run_chip_baseline_lbm.py`), GPU, logged to `Optimizer/V1/chip_fit_overnight_2026-07-02.log`.
LBM baseline runs HBB (neutral) by default; specular is now an opt-in wall-curvature study.
**Next: read results in the morning** — CV_L/CV_T achieved, APD, λ, cross-engine offsets,
preset paths → log to KNOWLEDGE + tick completion criteria.

## Failed Approaches

- **⚑ Applying `r*/dx ≥ 3` as a hard feasibility constraint in the joint fit** (2026-07-10, THE
  MISTAKE) — WRONG because: r*/dx≥k is **SCS-specific** (LBM specular-same-cell wall / wavefront
  curvature, [[boundary_conduction_speedup]]), NOT a general resolution gate for a monodomain/HBB
  CV fit. It produced the false "INFEASIBLE" (37/4000 candidates *did* hit CV_T=2.6; all discarded
  by the filter) and every downstream "kinetics required / CV_T is the wall / dx rabbit hole"
  conclusion. Fix: drop r*/dx for HBB fits (`require_resolved=False`); re-run on LBM+HBB. **This
  also voids the two older r*/dx entries below** ("k=1 insufficient, use k≈3"; the "r*/dx<1
  source-sink block" reading of the sequential-fit garbage) — those framed a real secant bug and a
  real frozen-θ problem through the wrong r*/dx lens. r*/dx is real *only for SCS*.

- **Sequential cell→tissue tuning** (2026-07-02) — failed because: CV depends on (θ,D) jointly and
  G_Na is shared between dV/dt (cell stage) and CV/r* (tissue stage); freezing θ leaves only D to
  hit CV, driving slow targets into the r*/dx<1 source-sink block. Ionic + conduction must be tuned
  JOINTLY. Both baselines reproduced garbage tissue (D=0.004, CV=nan).
- **Secant D fallback that bumps D UP ×4 on failure** (`_fit_D_for_cv`, 2026-07-02) — failed because:
  for slow chip targets the propagating window is BELOW D0=0.001, so bumping up (→0.004) goes the
  wrong way into the nan zone → returns (0.004, nan). Should bracket-DOWN into the window.
- **k=1 resolvability constraint (r*/dx≥1)** (2026-07-02, audit) — insufficient because: r*/dx≥1 only
  marginally propagates on a grid that does NOT resolve source-sink curvature (needs r*/dx≳3); CV
  there is grid-dependent/"fudgable" → fitting CV to a numerical artifact. Use k≈3.
- **4-objective qNEHVI for the joint fit** (2026-07-02, audit) — rejected in favor of constrained
  scalarization: 4-obj hypervolume under tissue-in-the-loop is expensive and returns silent dominated
  compromises instead of surfacing infeasibility ("no (θ,D) hits CV_T at r*/dx≥3 within g_Na bounds").


- **dVdt target of 25 V/s** (2026-03-16) — failed because: this target was derived from mature ventricular CM literature, but MHAS13 with TTP06 IK1 injection inherently has fast sodium kinetics. Its physiological range is 80-130 V/s. The target and the model are fundamentally mismatched; fixing it requires either revising the target or modifying Na channel gating (not conductance scaling).

- **dVdt constraint at 60 V/s** (2026-03-17) — failed because: too tight for MHAS13 baseline of 132 V/s. Only 2/74 BayesOpt candidates were feasible. Relaxing to 120 V/s gave 41/74 feasible while still constraining unrealistic upstrokes.

- **Tier 1 only (6 parameters)** (2026-03-17) — failed because: 4 of 6 parameters hit their bounds, indicating the parameter space was too constrained. The optimizer was trying to compensate through extreme values rather than finding a balanced solution. Adding 4 more parameters (kNaCa, PNaK, g_pCa, VmaxUp) in tier 2 resolved this.

- **Newton-based CV refinement** (2026-03-17) — failed because: analytical sqrt(D) approximation for the derivative gave 50.6% overshoot on the first correction step. The two-point secant method using actual simulation data converged to 2.4% error in 3 steps — dramatically better because it uses the real (nonlinear) CV-vs-D relationship.

- **Analytical CV warm-start alone** (2026-03-17) — failed because: CV proportional to sqrt(D_eff) gives a reasonable starting point but is not accurate enough for final convergence. The relationship between diffusion coefficient and CV is nonlinear in practice (discretization, ionic model coupling). Secant refinement on top of the warm-start was necessary.

## Session Log

### 2026-06-30 Session
**Worked on**: Finalizing + executing the shared cross-plan "Engine Tuner → cardiac_core multi-engine" (with `geometry_induced_reentry`) — PLAN.md to convergence, then full implementation.
**Accomplished**:
- **PLAN.md audit-converged** over 6 iterations (issues 18→13→9→3 [structure], then user's anisotropy domain-correction → 7→3 [mechanism]; 0 critical/high at convergence). Caught a wrong LBM-MRT mechanism (engine is BGK-scalar; anisotropy needs MRT + `tau_tensor_from_D`, rates s=1/τ — NOT `tau_from_D` twice), and the dx≠dt benchmark requirement.
- **Implemented all 6 phases** on branch `engine-tuner-cardiac-core` (11 commits, ~30 new tests, no regressions — 32 upstream LBM + cardiac_core suites green): P0 cardiac_core seam (`create_cardiac_mesh(D_yy=)` + `lbm()` instance pass-through) + **LBM D2Q9-MRT anisotropy** (vendored `cardiac_core/_lbm` + upstream `LBM/Engine_V1`) + dx≠dt diffusion benchmark proving the s→D mapping; P1 `cc_runner.py` (CV via functional API, mono/bidomain/lbm, CV∝√D); P2 `chip.py` (161² Parker mesh + anisotropic targets); P3 `presets.py` Tier-1 records + `run_chip_fit.py` (smoke; full fit GATED); P4 `cross_engine.py` (mono↔bidomain CV_T ~12%, mono↔lbm ~29%); P5 `export_lab_preset` + `_SCHEMA.md` ext + `run_chip_baseline_lbm.py`.
- **Engine finding (chi convention)**: effective-D meshes require **chi=1.0** — the monodomain FDM operator solves χ·Cm·∂V/∂t=∇·(D·∇V), so membrane-effective D = D/(χ·Cm); the default chi=1400 silently under-diffuses ~1400× → discretization conduction block (Vmax pools 80–123 mV). Empirically confirmed: degeneracy (D=1e-3,χ=1400 ≡ D=7.14e-7,χ=1, bit-identical block) + block-is-discretization (propagates at finer dx) + **the real chip regime (effD=2.5e-5) propagates cleanly at chip dx, CV≈6 cm/s — no artificial block**. Logged as API-debt to `engine_consolidation` IDEALOG (firewall bypass; recommend a `mode=` flag or routing through `ConductivityConfig`).
**Next**: Run the gated full fits (`run_chip_fit.main`, then `run_chip_baseline_lbm.main`); watch hiPSC-5.2 dx adequacy. Optional: tissue_fitter rewire; `create_cardiac_mesh` chi fix.

### 2026-07-02 → 07-10 Session (long, multi-thread)
**Worked on**: (1) the lateral-boundary-speedup dt guide; (2) the cardiac_core specular-BC engine gap + fix verification; (3) executing the gated chip fits — which exposed a fundamental tissue-fit failure; (4) diagnosing it to a sequential-architecture flaw; (5) the JOINT tuning architecture redesign, audit-converged; (6) the PLAN.md blueprint, audit-converged.
**Accomplished**:
- **dt boundary-speedup guide** (`chip.boundary_number`, β=D·dt/dx²=τ) shipped BC-aware (default HBB → truthful "no speed-up"; specular flip only under specular). Final-check found the chip runs HBB, not specular → user fixed cardiac_core (commits `40cd2ca`/`1dda8f6`: run_lbm forwards boundary/alpha; MRT+specular co-exist). Verified: anisotropic-MRT chip + specular runs, wall-localized field change (1.38 mV), α-blend monotone.
- **Gated chip fits run** (GPU run killed at 10.5 h — n_iter=200 + degenerate spontaneous-CL 5 s-sims; also a colleague's 29 GB Jupyter kernel on the shared GPU, left untouched → moved to CPU). Lean CPU rerun (n_iter=40, dropped CL objective): **cell fits converged** (APD err 3.1 ms, dV/dt 0.4) but **tissue fits garbage** (D=0.004, CV=nan) on BOTH baselines.
- **Root cause = sequential architecture** (user's diagnosis, correct): θ frozen after cell fit, D-only secant → slow targets driven into the **r*/dx<1 source-sink block**; G_Na shared between dV/dt (cell) and CV/r* (tissue) → can't trade. **Decision: joint (never sequential).**
- **Architecture `JOINT_TUNING_ARCHITECTURE.md` audit-CONVERGED (3 iters)** — three-leg (resolution shell / constraint graph / physical joint fit); FIT physical / RESOLVE numerical; convergence-aware CV estimator; attack-all high-dim; known-true pair constraints. Audits caught: ignored `joint_refiner.py`, wrong CFL story (implicit CN), **two ionic backends** (cell=cardiac_sim/V5.4 vs tissue=cardiac_core → P-1 unification blocker), kinetics-in-hooks.
- **PLAN.md audit-CONVERGED (3 iters)** — 5 phases P-1→P3; audit caught the kinetics-scaling-in-`step()`-vs-hooks bug (tissue solver uses `compute_gate_*` hooks, never `step()`) + pacing-to-steady-state parity.
**Next**: GATED — execute PLAN.md starting at **P-1 (backend unification)**. Awaiting explicit "go". Session-end commit pending (user commits after /clear).

### 2026-07-11 Session
**Worked on**: Executed the full PLAN (P-1→P3) + ran the production joint fit — then, on the user's
push, found and corrected **THE MISTAKE**.
**Accomplished**:
- Built + committed the entire Engine Tuner V2 machinery on `engine-tuner-v2-joint` (P-1 backend
  unification; bracket-down secant; `cv_estimator`; feasibility map; MHAS13 Na-kinetics axes;
  `decision_space` + `joint_fit` GP-emulator constrained-scalarization fit with block-masking,
  input normalization, D-solve, reduced CV feature set; chip/preset/export fixes; dx-as-tunable-axis).
  Optimizer non-slow suite 59 passed / 0 failed; cardiac_core ionic green.
- Ran the production joint fit → returned "INFEASIBLE". **This verdict is WRONG.** The user
  identified that the `r*/dx ≥ 3` hard constraint driving it is **SCS-specific** (from the LBM
  specular boundary/curvature work), not a general fit gate — 37/4000 candidates actually hit
  CV_T=2.6 but were filtered out. All "conductance-only infeasible / kinetics necessary / CV_T is
  the wall / coarse-dx / excitability-floor" conclusions are **artifacts** and are withdrawn.
- Side finding (valid): the warm-start `chip_hipsc.json` θ is a V5.4-era fit; on cardiac_core it
  gives **APD≈276** (target 350), dV/dt≈113 — so the ionic AP still needs a cardiac_core re-tune
  (dV/dt fine, APD short). The joint fit never emitted a tuned θ (it errored out on the bogus filter).
- Ran `/save-session` to **decontaminate** KNOWLEDGE + IDEALOG: withdrew the wrong conclusions,
  documented THE MISTAKE, preserved the valid engineering.
**Next**: re-run the joint fit on **LBM + HBB** with **no r*/dx** (`require_resolved=False`),
targeting only CV_L=5.2 / CV_T=2.6 / APD=350 / dV/dt / 2:1 — take whatever (θ, kinetics, D) reaches
them; don't assert infeasibility until run on the right engine. (SCS work later uses LBM SCS, where
r*/dx belongs.)
