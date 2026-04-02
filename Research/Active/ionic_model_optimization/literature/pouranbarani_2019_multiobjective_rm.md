---
paper: pouranbarani_2019_multiobjective_rm
title: "A robust multi-objective optimization framework to capture both cellular and intercellular properties in cardiac cellular model tuning"
authors: "Pouranbarani E, Bhatt SM, Bhatt SM, et al."
year: 2019
journal: "PLOS ONE"
doi: "10.1371/journal.pone.0225245"
pmid: "31730631"
pdf: ../papers/multiobjective_optimization_membrane_resistance_2019_pouranbarani.pdf
questions: [Q8]
---

## Key Findings
- Fitting to AP waveform alone is **non-unique for tissue behavior**: different parameter sets produce identical APs but very different membrane resistance (Rm) profiles and therefore different conduction velocities
- Multi-objective optimization (NSGA-II) with AP + Rm + resting Rm as objectives captures both cellular and intercellular properties
- Adding Rm fitting improves intercellular accuracy by 89-96% compared to AP-only fitting, at the cost of 20-30% worse AP RMSE
- Three optimization scenarios tested: AP-only (GA), AP + Rm curve (NSGA-II), AP + Rm curve + resting Rm (NSGA-II)
- Pareto fronts reveal explicit tradeoffs between AP fidelity and tissue-level accuracy

## Method
- **Optimizer**: NSGA-II (Non-dominated Sorting Genetic Algorithm II) for multi-objective; standard GA for single-objective baseline
- **Models**: Ten Tusscher (TNNP) as fitted model, Iyer-Mazhari-Winslow (IMW) as target — model-to-model fitting
- **Parameters**: 16 maximal conductances/fluxes (GNa, GCaL, GKr, GKs, GK1, GNaL, etc.)
- **Objectives**: (1) AP waveform RMSE, (2) Rm curve RMSE in allowed regions, (3) Resting Rm absolute error
- **Key innovation**: Careful selection of Rm measurement regions to avoid singularities — "allowed" vs "disallowed" voltage regions defined by examining Rm profiles across models
- **Population**: 100, evaluations: 10,000, crossover index: 20, mutation index: 20
- 5 configurations tested: baseline, shortened APD, prolonged APD, hyperkalemia, hypokalemia

## Key Equations / Results
- AP RMSE (Scenario 1): 1.05-1.88 mV across configurations
- AP RMSE (Scenario 3, with Rm): 2.37-2.79 mV — acceptable tradeoff
- Rm improvement: 89-96% reduction in Rm fitting error when included as objective
- Resting Rm improvement: up to 98.6% in Scenario 3
- Rm measured via voltage-clamp perturbation protocol: ΔV/ΔI at each voltage during repolarization

## Connections to Our Models

### Relevant Engine Components
- **Bidomain V1**: The membrane resistance concept directly connects to our parabolic equation — Rm determines the source-sink relationship that governs CV
- **Monodomain V5.4**: TNNP (Ten Tusscher) is exactly our TTP06 model. The 16 parameters optimized here are the same conductance scaling factors we would tune
- Ionic models: `Bidomain/Engine_V1/cardiac_sim/ionic/ttp06/` and `Monodomain/Engine_V5.4/cardiac_sim/ionic/ttp06/`

### Agreements
- Confirms our observation that CV depends on more than just GNa — the full Rm profile during repolarization matters
- Uses TNNP model (= our TTP06) with the same ODE structure

### Disagreements or Gaps
- They don't directly optimize for CV as an objective — they use Rm as a proxy. We could instead measure CV directly in a tissue simulation as an objective
- 10,000 evaluations with NSGA-II is computationally expensive if each evaluation requires a tissue simulation for CV measurement. Their approach avoids this by using single-cell Rm as a surrogate for tissue CV

### Actionable Insights
- **HIGH**: Implement NSGA-II with AP RMSE + CV as dual objectives for TTP06/ORd tuning. Use our existing engine to measure CV from tissue simulations.
- **HIGH**: The 16 parameters they tuned (Table A1 in supplement) are a good starting point for our parameter search space
- **MEDIUM**: The "allowed regions" concept for Rm could be adapted to avoid fitting artifacts in regions where the AP has steep gradients
- **LOW**: Consider Rm as an additional diagnostic even if not used as a direct optimization objective

## Limitations / Caveats
- Model-to-model fitting only (TNNP → IMW), not validated against experimental data
- NSGA-II with 16 parameters and 10,000 evaluations may not fully explore the search space
- The Rm measurement protocol (voltage-clamp perturbation) is computationally expensive
- Requires defining "allowed/disallowed regions" per model — somewhat subjective
