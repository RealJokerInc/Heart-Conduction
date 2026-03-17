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

## Failed Approaches

- **dVdt target of 25 V/s** (2026-03-16) — failed because: this target was derived from mature ventricular CM literature, but MHAS13 with TTP06 IK1 injection inherently has fast sodium kinetics. Its physiological range is 80-130 V/s. The target and the model are fundamentally mismatched; fixing it requires either revising the target or modifying Na channel gating (not conductance scaling).

- **dVdt constraint at 60 V/s** (2026-03-17) — failed because: too tight for MHAS13 baseline of 132 V/s. Only 2/74 BayesOpt candidates were feasible. Relaxing to 120 V/s gave 41/74 feasible while still constraining unrealistic upstrokes.

- **Tier 1 only (6 parameters)** (2026-03-17) — failed because: 4 of 6 parameters hit their bounds, indicating the parameter space was too constrained. The optimizer was trying to compensate through extreme values rather than finding a balanced solution. Adding 4 more parameters (kNaCa, PNaK, g_pCa, VmaxUp) in tier 2 resolved this.

- **Newton-based CV refinement** (2026-03-17) — failed because: analytical sqrt(D) approximation for the derivative gave 50.6% overshoot on the first correction step. The two-point secant method using actual simulation data converged to 2.4% error in 3 steps — dramatically better because it uses the real (nonlinear) CV-vs-D relationship.

- **Analytical CV warm-start alone** (2026-03-17) — failed because: CV proportional to sqrt(D_eff) gives a reasonable starting point but is not accurate enough for final convergence. The relationship between diffusion coefficient and CV is nonlinear in practice (discretization, ionic model coupling). Secant refinement on top of the warm-start was necessary.

## Session Log
