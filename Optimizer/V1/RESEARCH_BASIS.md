# Engine Tuner — Research Basis

How each design decision in the pipeline maps to the literature reviewed in `Research/Q8_ionic_model_optimization/`.

## The Three Truths (All 8 Papers Agree)

1. **Single-AP fitting is non-unique** — many parameter sets produce identical APs but different tissue behavior
2. **Tissue-level data is required** — CV, membrane resistance, or restitution curves must be optimization targets
3. **Rate-dependent dynamics break degeneracies** — multi-CL pacing separates compensating parameters (IKr/IKs)

## Paper → Pipeline Mapping

### Pouranbarani 2019 — Multi-objective optimization with Rm

**Key insight**: Adding membrane resistance as an optimization objective improves tissue-level accuracy by 89-96%.

**Pipeline impact**:
- Phase 3 uses multi-objective optimization (AP + CV + tissue APD + restitution)
- The Pareto front concept comes directly from this paper
- We use CV as a direct tissue objective instead of Rm (since we can measure CV in our tissue sims)
- Their 16-parameter NSGA-II search space informed our 8-parameter ionic + 2-4 tissue decomposition

**Citation**: [DOI: 10.1371/journal.pone.0225245](https://doi.org/10.1371/journal.pone.0225245)

### Coveney 2021 — GP emulator for restitution curves

**Key insight**: 500 training simulations → PCA + GP surrogate → 10⁵ predictions/second. Combined CV + APD + ERP restitution is critical for identifiability.

**Pipeline impact**:
- Phase 3 Step B (GP emulator) is directly from this paper
- PCA decomposition of restitution curves (3 components capture >99%)
- Latin Hypercube sampling for training data
- The "500 training sims" number comes from their result that ~100× per parameter is sufficient

**Citation**: [DOI: 10.3389/fphys.2021.693015](https://doi.org/10.3389/fphys.2021.693015)

### Nieto Ramos 2023 — HMC Bayesian inference

**Key insight**: Full posterior distributions reveal which parameters are identifiable. Multi-CL data including near-alternans dynamics is essential. Estimated noise σ diagnoses model adequacy.

**Pipeline impact**:
- Phase 1 protocol uses 3 cycle lengths (1000, 500, 350 ms) including near-alternans
- The noise σ diagnostic is used in validation (if estimated σ >> simulation noise, model structure is inadequate)
- We chose BayesOpt over HMC for the raw simulator (gradient through Rush-Larsen is fragile) but HMC could be used on the Phase 3 GP surrogate

**Citation**: [DOI: 10.1007/s11517-022-02685-y](https://doi.org/10.1007/s11517-022-02685-y)

### Groenendaal 2015 — Cell-specific models via GA

**Key insight**: Single-AP fitting recovers wrong parameters despite perfect AP match. Stochastic stimulation + voltage clamp + iterative refinement drops error by 3 orders of magnitude.

**Pipeline impact**:
- The fundamental motivation for multi-objective / multi-protocol fitting
- Phase 1 uses multi-CL pacing as an alternative to their stochastic stimulation (same principle: probe rate-dependent dynamics)
- The IKr/IKs compensation degeneracy they identified is the core problem our pipeline solves

**Citation**: [DOI: 10.1371/journal.pcbi.1004242](https://doi.org/10.1371/journal.pcbi.1004242)

### Zhang 2024 — Gradient-based two-waveform fitting

**Key insight**: Fitting control AP + IKr-blocked AP simultaneously breaks parameter correlations. Gradient-based optimization is efficient for ≤6 parameters.

**Pipeline impact**:
- The two-waveform strategy could optionally extend Phase 1 (fit baseline + simulated IKr block)
- Gradient-based methods are viable for our PyTorch models (autograd available)
- We chose BayesOpt over gradient descent because it's global (no local optima) and gives uncertainty

**Citation**: [DOI: 10.1038/s41598-024-63413-0](https://doi.org/10.1038/s41598-024-63413-0)

### Chang 2017 — UQ for CiPA ORd model

**Key insight**: ORd conductances were explicitly rescaled for CiPA. >60% channel block needed for reliable IC50. UQ is essential for confidence in predictions.

**Pipeline impact**:
- CiPA ORdv1.0 scaling factors serve as a validated starting point for ORd tuning
- The >60% block rule informs optional Phase 1 drug-block fitting protocol
- UQ is built into the pipeline: BayesOpt GP provides posterior uncertainty; Phase 3 emulator provides prediction bounds

**Citation**: [DOI: 10.3389/fphys.2017.00917](https://doi.org/10.3389/fphys.2017.00917)

### Cairns 2017 — GA parameterization

**Key insight**: GA baseline for cardiac AP model parameterization.

**Pipeline impact**:
- GA (NSGA-II) is used in Phase 3 on the GP surrogate where evals are free
- CMA-ES replaces basic GA as the black-box fallback (consensus best for 5-50 dim continuous optimization)

**Citation**: [DOI: 10.1063/1.5000354](https://doi.org/10.1063/1.5000354)

### Nieto Ramos 2022 — HMC proof-of-concept (CinC)

**Key insight**: HMC scales to 13-parameter cardiac models. Estimated noise σ serves as model adequacy diagnostic.

**Pipeline impact**:
- Validates that Bayesian methods are feasible at our parameter count (8-12)
- Conference precursor to the 2023 paper; same conclusions applied

**Citation**: [DOI: 10.23919/cinc53138.2021.9662836](https://doi.org/10.23919/cinc53138.2021.9662836)

## Method Selection Rationale

### Why BayesOpt (BoTorch) over NSGA-II?

| Criterion | BayesOpt | NSGA-II |
|-----------|----------|---------|
| Evaluations to converge | 200-500 | 10,000-50,000 |
| Reuses past evaluations? | Yes (GP surrogate) | No |
| Multi-objective? | Yes (qNEHVI) | Yes |
| Uncertainty? | Yes (GP posterior) | No |
| PyTorch-native? | Yes (BoTorch) | No (pymoo is numpy) |

BayesOpt is 10-50× more sample-efficient. The only downside is GP scaling in high dimensions (>15 params), which is why CMA-ES is the fallback.

### Why decompose into Cell + Tissue phases?

CV ≈ f(GNa, D) and APD ≈ g(GCaL, GKr, GKs, GK1) are approximately separable. This lets Phase 1 run on cheap single-cell sims (seconds) and Phase 2 run on expensive tissue sims (minutes) with fewer parameters. Phase 3 then handles the coupling (tissue APD ≠ cell APD) with a GP surrogate.

Without decomposition: 10-12 params × tissue sims = need 50K+ tissue-level evaluations. With decomposition: 8 params × cell sims (cheap) + 2-4 params × tissue sims + 500 coupled sims for the emulator.

### Why not pure gradient methods (L-BFGS)?

The objective landscape has ridges (IKr/IKs compensation) and multiple local optima. Gradient methods find the nearest valley, not the deepest one. BayesOpt explores globally.

Exception: Phase 2 (2-4 params, nearly convex) could use L-BFGS, but BayesOpt is equally fast and gives uncertainty bounds.

## Our Competitive Advantage

| Capability | Most Labs | Us |
|------------|-----------|-----|
| Simulator speed | CPU, days for 500 sims | GPU PyTorch, hours for 500 sims |
| Model differentiability | Black-box | Autograd available (enables gradient methods) |
| Tissue-level evaluation | Often single-cell only | Full tissue sim with CV measurement |
| Multiple engines | One solver | Bidomain + Monodomain + LBM (cross-validate) |
| Surrogate modeling | Custom, ad-hoc | BoTorch + GPyTorch (production-grade) |
