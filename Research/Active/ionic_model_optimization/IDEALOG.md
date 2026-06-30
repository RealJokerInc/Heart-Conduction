# Ionic Model Optimization — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan this in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
BayesOpt pipeline (qNEHVI + BoTorch) tuning MHAS13 ionic model parameters against dual objectives (APD + CV). Cell fitter and tissue fitter both implemented, working across monodomain and bidomain engines. Tier 2 (10 params) with hard dVdt/Vpeak/Vrest constraints. Best result: APD=352ms (0.6% err), CV_L=14.6 cm/s (2.4% err), dVdt=106 V/s.

## Next Step
Joint refinement phase: GP emulator + NSGA-II for Pareto exploration across remaining parameter degeneracies. Also: revise dVdt target for MHAS13 to ~100 V/s (not 25), and implement multi-rate pacing (CL=500/1000/1500) to break IKr/IKs compensation.

## Thread

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

**Anisotropy domain-correction + 2nd convergence (2026-06-29).** Initially simplified to isotropic (from an early audit) — **wrong physics**; reverted to 2:1 anisotropic. The real cost this exposed: cardiac_core's `lbm()` is **BGK-scalar-only** and rejects non-isotropic D, and the LBM engine has **no wired per-axis MRT** (`mrt_collide_d2q5` exists but unwired; only `lbm_step_d2q9_mrt`). So **Phase 0 Step 0.2 = genuine engine-level MRT work** (LBMSimulation collision selector + per-axis rates from `tau_tensor_from_D`, rates `s=1/τ`; D2Q9-MRT recommended) — now the single largest item in the plan. Two more audit iters (5–6) re-converged the per-axis/MRT mechanism: 0 critical / 0 high, verdict CONVERGED. PLAN-ONLY gate still in force (no execution). λ-vs-chip tension (λ≈18–33 mm > 16 mm chip) remains flagged as the reentry application's problem, not this fit's.

## Failed Approaches

- **dVdt target of 25 V/s** (2026-03-16) — failed because: this target was derived from mature ventricular CM literature, but MHAS13 with TTP06 IK1 injection inherently has fast sodium kinetics. Its physiological range is 80-130 V/s. The target and the model are fundamentally mismatched; fixing it requires either revising the target or modifying Na channel gating (not conductance scaling).

- **dVdt constraint at 60 V/s** (2026-03-17) — failed because: too tight for MHAS13 baseline of 132 V/s. Only 2/74 BayesOpt candidates were feasible. Relaxing to 120 V/s gave 41/74 feasible while still constraining unrealistic upstrokes.

- **Tier 1 only (6 parameters)** (2026-03-17) — failed because: 4 of 6 parameters hit their bounds, indicating the parameter space was too constrained. The optimizer was trying to compensate through extreme values rather than finding a balanced solution. Adding 4 more parameters (kNaCa, PNaK, g_pCa, VmaxUp) in tier 2 resolved this.

- **Newton-based CV refinement** (2026-03-17) — failed because: analytical sqrt(D) approximation for the derivative gave 50.6% overshoot on the first correction step. The two-point secant method using actual simulation data converged to 2.4% error in 3 steps — dramatically better because it uses the real (nonlinear) CV-vs-D relationship.

- **Analytical CV warm-start alone** (2026-03-17) — failed because: CV proportional to sqrt(D_eff) gives a reasonable starting point but is not accurate enough for final convergence. The relationship between diffusion coefficient and CV is nonlinear in practice (discretization, ionic model coupling). Secant refinement on top of the warm-start was necessary.

## Session Log
