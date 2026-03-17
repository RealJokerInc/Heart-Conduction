# Ionic Model Parameter Optimization — Knowledge File

> This file is a running synthesis. Updated as findings accumulate.
> When the question is complete, a copy is promoted to `Research/Knowledge/`.

## Current Understanding

Tuning ionic model parameters (MHAS13/TTP06/ORd) to match target CV and APD requires **multi-objective optimization** because fitting to a single action potential waveform is fundamentally non-unique. Many parameter sets produce identical APs but yield different tissue-level behavior (CV, membrane resistance, restitution).

### Pipeline status (updated 2026-03-17)

The Optimizer V1 pipeline is implemented end-to-end across both engines:
- **Cell fitter**: batched BayesOpt (qLogNEHVI), 10-param tier 2, hard constraints on dVdt/Vpeak/Vrest
- **Tissue fitter**: analytical CV∝√D_eff warm-start + secant refinement, engine-agnostic (mono or bidomain)
- **Bidomain support**: `tissue_runner_bidomain.py` wraps BidomainSimulation with D_eff→(D_i,D_e) decomposition at fixed ratio D_e/D_i=3.597
- **Best result** (iteration 2): APD=352ms (0.6% err), CV_L=14.6 (2.4%), CV_T=7.6 (1.7%), dVdt=106V/s (constrained <120)
- **Monodomain vs bidomain**: D_eff values agree within 6% (0.000447 mono vs 0.000422 bidomain), confirming D_eff≈D_i·D_e/(D_i+D_e) is a good approximation for insulated tissue

### The core degeneracy problem

The key degeneracy is **IKr/IKs compensation** during the AP plateau: these currents overlap temporally and can trade off against each other without changing the AP shape. This means single-AP fitting leaves IKr and IKs individually unconstrained, even though their ratio matters for rate-dependent behavior and drug response.

Breaking the degeneracy requires richer target data:
- **Multi-rate pacing** (restitution curves) -- probes rate-dependent kinetics
- **Paired waveforms** (control + selective channel block) -- isolates individual currents
- **Stochastic stimulation** (randomly timed pulses) -- probes nonlinear dynamics
- **Tissue-level CV** -- constrains GNa and the diffusion coefficient

### Method landscape

| Method | Strengths | Weaknesses | Best for |
|--------|-----------|------------|----------|
| **NSGA-II** (Pouranbarani 2019) | Pareto front shows explicit tradeoffs; handles 16 params; no gradients needed | No uncertainty quantification; 10,000 evaluations | Exploring AP vs CV tradeoffs |
| **Bayesian HMC** (Nieto Ramos 2023) | Full posterior distributions; identifies compensatory mechanisms; noise diagnostic | Needs differentiable model; tested on 5-13 params only | Parameter uncertainty quantification |
| **ABC-SMC** (Nieto Ramos 2023) | Likelihood-free; handles model inadequacy | Wider posteriors than HMC; computationally expensive | When likelihood is intractable |
| **GP Emulator** (Coveney 2021) | 10^5x speedup after training; tissue-level calibration | 500 simulations for training; 5-param model only | Restitution-based calibration |
| **GA** (Groenendaal 2015) | Simple, robust, handles discontinuities | No uncertainty; no gradients | Baseline rough search |
| **Gradient PO** (Zhang 2024) | Fast convergence for small param sets; two-waveform protocol | Local optima; needs smooth objective | Fine-tuning 6 conductances |

### Parameter subsets

Typically 6-16 maximal conductance scaling factors are tuned, not all model parameters:
- **CV-dominant**: GNa, D (diffusion coefficient)
- **APD-dominant**: GCaL, GKr, GKs, GK1
- **Plateau shape**: GCaL, GKr, GKs (the compensatory trio)
- **Resting potential**: GK1, GNaK

Pouranbarani 2019 tuned 16 conductances in TTP06 (our exact model) and found that adding membrane resistance (Rm) as an optimization objective improved intercellular accuracy by 89-96% at a cost of 20-30% worse AP RMSE.

### Key insight from Groenendaal 2015

Single-AP fitting with 9 parameters: parameters converge to wrong values despite near-perfect AP match (SSE < 0.01 mV^2). Adding stochastic stimulation drops prediction error by ~1 order of magnitude. Adding voltage-clamp data drops it by another order. Iterative GA refinement gains yet another order. Total: ~3 orders of magnitude improvement over naive single-AP fitting.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary method | BayesOpt with qNEHVI (Optimizer V1) | Multi-objective, sample-efficient, GPU-native via BoTorch |
| Objectives | APD + CV (dual) | Captures both cellular and tissue-level behavior |
| Parameter space | Maximal conductance scaling factors (6-16) | Standard approach across all reviewed literature |
| Target data | hiPSC-CM experimental measurements | MHAS13 model provides baseline; Optimizer V1 tunes to targets |
| Differentiability | PyTorch-based ionic models enable gradient computation | Opens path to HMC if BayesOpt proves insufficient |
| dVdt handling | Hard constraint (dvdt<120 V/s), NOT objective | Objective caused infeasibility at 60 V/s; 120 matches MHAS13 physiology |
| Tier level | Tier 2 (10 params) for production | Tier 1 (6) had 4/6 at bounds; tier 2 adds kNaCa, PNaK, g_pCa, VmaxUp |
| CV warm-start | Two-point secant (not analytical √D) | Secant uses real sim data; converges to <3% in 2-3 steps |
| Bidomain tissue | D_eff optimization with fixed D_e/D_i=3.597 | Reduces 2-unknown problem to 1-unknown; ratio from cv_shared.py |
| Reproducibility | seed=42 in config, passed to Sobol + torch | Ensures identical runs |

### Iteration 2 findings (2026-03-17)

**dV/dt constraint is essential but must be tuned carefully.** Setting dvdt_max<60 V/s with MHAS13 (baseline 132 V/s) made 72/74 candidates infeasible — the constraint was too tight. Relaxing to dvdt_max<120 V/s (physiological for TTP06 IK1 on Paci) gave 41/74 feasible, with final dV/dt=106 V/s.

**Tier 2 (10 params) helps but several still hit bounds.** g_Kr=3.0, kNaCa=2.5, g_pCa=2.5, VmaxUp=2.0, g_K1=0.3, g_to=0.3 all at their limits. The extra tier 2 params (kNaCa, PNaK, g_pCa, VmaxUp) participate in the solution, confirming they add value. PNaK=0.5 (downscaled) is a new knob not available in tier 1.

**Secant CV refinement is dramatically better than Newton.** CV_long: 50.6% initial overshoot → 2.4% in 3 secant steps. CV_trans: 20.5% → 1.7% in 2 steps. The two-point secant uses actual simulation data for the derivative estimate rather than the √D analytical approximation.

**The APD-dVdt tradeoff is a fundamental model property.** MHAS13 cannot simultaneously achieve APD=350ms and dV/dt=25V/s with conductance scaling alone. The hiPSC-CM model inherently has fast sodium kinetics relative to its AP morphology. Achieving dV/dt=25 V/s would require modifying the Na channel kinetics (gating time constants), not just the conductance.

## Open Questions

- Should dV/dt target be revised to match MHAS13's physiological range (~80-130 V/s) rather than the original 25 V/s target?
- Can multi-rate pacing (CL=500, 1000, 1500) break the remaining parameter degeneracies?
- Is membrane resistance (Rm) a better tissue-level proxy than direct CV measurement?
- Would a restitution-curve emulator (Coveney approach) provide sufficient accuracy?
- Can HMC scale to 10+ conductances given that Nieto Ramos only tested up to 13?

## Connections
- **Engines**: Monodomain V5.4 (simulation backend for optimization), Bidomain V1 (CV validation)
- **Related research**: hipsc_cm_ionic_models (MHAS13 is the tuning target), boundary_conduction_speedup (boundary effects must be accounted for in tissue-level CV fitting)
- **Pipelines**: Optimizer V1 (BayesOpt pipeline using V5.4)
