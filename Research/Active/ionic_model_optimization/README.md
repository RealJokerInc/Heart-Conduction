# Ionic Model Parameter Optimization

## Question
How do we tune ionic model parameters (TTP06/ORd/MHAS13) to match target CV, APD, and restitution curves?

## Status: Active

## Why It Matters
Ionic model parameters are not universal — they must be tuned to match specific cell types, species, or experimental preparations. Fitting to a single AP waveform is non-unique (many parameter sets produce identical APs but different tissue behavior). Multi-objective optimization with tissue-level targets is required.

## Engines
- **Monodomain V5.4**: Simulation backend for the optimization pipeline
- **Bidomain V1**: Cross-validation of optimized parameters
- **Optimizer V1**: BayesOpt pipeline (qNEHVI + BoTorch)

## Completion Criteria
- [x] Literature review complete (8 papers, method comparison)
- [x] Optimizer V1 architecture designed (4-phase pipeline)
- [x] MHAS13 test run through optimizer (APD=347ms, target 350ms)
- [x] MHAS13 test run through bidomain pipeline (APD=349ms, CV=15.8 cm/s)
- [x] 10x speedup via batching, subcycling, analytical CV
- [x] Optimizer V1 Cell + Tissue fit implemented (tier 2, constrained, both engines)
- [x] Bidomain pipeline support (tissue_runner_bidomain.py, D_eff decomposition)
- [x] Optimizer V2 **Joint refinement IMPLEMENTED** — constrained scalarization on a GP
      emulator (`tuner/joint_fit.py`), NOT sequential; NaN-masked block, D-solved
      candidates, normalized GP inputs, surfaces infeasibility. (Production fit = gated
      multi-hour run; machinery validated on a synthetic oracle.)
- [x] P-1 backend unification — dV/dt/APD and CV on ONE cardiac_core model (parity ≤1%)
- [x] P0 secant bracket-down fix + diagnostics (the r*/dx "resolution shell" is SCS-specific
      — NOT a general fit gate; see KNOWLEDGE → THE MISTAKE)
- [~] P1a feasibility map — BUILT, but its "conductance-only infeasible" verdict is **WITHDRAWN**
      (it was defined by the wrongly-applied r*/dx≥3 constraint; artifact, not a finding)
- [x] P1.5 Na-kinetics axes on MHAS13 (τ_m/h/j, V_half; in the hooks; identity-safe) — a working
      decoupling knob (NOT a claim that kinetics is *required*)
- [x] Chip mesh dx knob + joint-fit record schema (kinetics + per-axis D) + preset export fix
- [ ] Optimizer Validation suite (novel CL, stimulus robustness, stability)
- [ ] Cross-engine validation (V5.4 vs Bidomain vs LBM)
- [ ] Multi-rate pacing (break IKr/IKs degeneracy)
- [~] Revise dVdt target for MHAS13 — DATA in hand (dV/dt by g_Na = 60/83/109/159); the
      joint fit treats dV/dt as a *band* constraint, not a point target

## Sub-Questions

| Sub-Question | Status | Key Finding |
|-------------|--------|-------------|
| Parameter degeneracy | Complete | IKr/IKs compensation — need multi-rate pacing to separate |
| Method selection | Complete | BayesOpt qNEHVI for V1; HMC deferred to V2 |
| Pipeline speedup | Complete | 10x via batching + subcycling + analytical CV |
| dVdt constraint tuning | Complete | dvdt<120 V/s is correct for MHAS13; 60 too tight (2/74 feasible) |
| Tier 2 validation | Complete | 10 params, 41/74 feasible, APD=352ms (0.6% error) |
| CV secant refinement | Complete | 2-point secant: CV_L=14.6 (2.4%), CV_T=7.6 (1.7%) |

## Experiments

| Experiment | Engine | Result | Location |
|-----------|--------|--------|----------|
| MHAS13 optimizer run | V5.4 / Optimizer | APD=347ms (target 350) | `Optimizer/experiments/` |
| MHAS13 bidomain run | Bidomain V1 | APD=349ms, CV=15.8 cm/s | `Bidomain/Engine_V1/experiments/` |
| MHAS13 iter 2 (constrained, tier 2) | V5.4 / Optimizer | APD=352ms, dVdt=106, CV_L=14.6 | `Optimizer/V1/run_mhas13.py` |

## Literature
See `literature/` for paper summaries. Key references:
- Pouranbarani 2019 (NSGA-II, AP + tissue CV via membrane resistance)
- Coveney 2021 (GP emulator + Bayesian MCMC, restitution curves)
- Nieto Ramos 2023 (HMC + ABC-SMC, full posteriors)
- Groenendaal 2015 (GA, proves single-AP fitting is non-unique)
- Zhang 2024 (gradient-based, two-waveform fitting)

## Engine References

Files to read when resuming work on this question:

| File | What it tells you |
|------|-------------------|
| `Optimizer/V1/README.md` | Optimizer V1 overview, pipeline phases |
| `Optimizer/V1/ARCHITECTURE.md` | Input spec, constraints, timeline |
| `Optimizer/V1/IMPLEMENTATION.md` | Phase-by-phase plan with validation |
| `Optimizer/V1/TARGET_VALUES.md` | Target CV, APD, restitution specs |
| `Optimizer/V1/run_mhas13.py` | Latest MHAS13 monodomain optimization script |
| `Optimizer/V1/run_mhas13_bidomain.py` | MHAS13 bidomain optimization script |
| `Optimizer/V1/BIDOMAIN_PLAN.md` | Bidomain pipeline design (D_eff, ratio constraint) |
| `Optimizer/V1/tuner/batch_ionic.py` | Batched ionic step (PHAS13/MHAS13 IK1 switching) |
| `Optimizer/V1/tuner/tissue_runner_bidomain.py` | BidomainSimulation wrapper for CV measurement |
| `Optimizer/V1/tuner/cell_fitter.py` | Constrained multi-objective BayesOpt with feasibility filtering |
| `Optimizer/improvement.md` | Multi-engine adapter design (V2) |
| `Monodomain/Engine_V5.4/cardiac_sim/ionic/mhas13/model.py` | MHAS13 ionic model implementation |
| `Monodomain/Engine_V5.4/PROGRESS.md` | V5.4 engine status |
| `Research/Active/hipsc_cm_ionic_models/KNOWLEDGE.md` | MHAS13 maturation pathway details |

## Future Work
{No deferred items yet.}

## Connected Research
- **hipsc_cm_ionic_models** — MHAS13 is the primary tuning target
- **boundary_conduction_speedup** — Boundary effects must be accounted for in tissue-level CV fitting
- **engine_consolidation** — Cross-engine validation needs unified API
- **geometry_induced_reentry** — *driving application* of the cross-engine tuner: fits a Kit Parker tissue-chip EP set, then runs it on LBM (its primary engine) for reentry sweeps. **Shared cross-plan** (`PLAN.md`); end goal = Engine Tuner adapted to cardiac_core (monodomain + bidomain + LBM).
