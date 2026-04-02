---
paper: groenendaal_2015_cell_specific
title: "Cell-Specific Cardiac Electrophysiology Models"
authors: "Groenendaal W, Ortega FA, Krogh-Madsen T, Christini DJ"
year: 2015
journal: "PLOS Computational Biology"
doi: "10.1371/journal.pcbi.1004242"
pmid: "25928268"
pdf: ../papers/cell_specific_electrophysiology_models_2015_groenendaal.pdf
questions: [Q8]
---

## Key Findings
- **Single-AP fitting is fundamentally non-unique**: 9 conductance parameters can be set to wildly different values while producing nearly identical AP waveforms
- **Stochastic stimulation** (randomly-timed current pulses) dramatically improves parameter recovery — prediction error drops by ~1 order of magnitude vs single AP
- **Adding voltage-clamp data** drops error by another order of magnitude
- **Iterative GA refinement** (narrow bounds from previous run) gives another order of magnitude — total ~3 orders of magnitude improvement
- **IKr/IKs compensation** is the key degeneracy: these overlap temporally during the plateau and cannot be separated from a single AP
- Cell-specific models of 4 guinea pig myocytes significantly outperform generic published models

## Method
- **Optimizer**: Genetic Algorithm (GA) with iterative refinement
- **Model**: Faber-Rudy guinea pig ventricular model
- **Parameters**: 9 maximal conductances/fluxes (INa, ICaL, ICaT, IK1, IKr, IKs, IKp, IpCa, Jup)
- **Fitting protocols** (tested separately and combined):
  1. Single AP at 1 Hz
  2. Stochastic current-clamp (11 randomly timed stimuli over 5 s)
  3. Multi-step voltage-clamp protocol targeting individual currents
- **GA settings**: Population 500, range 0.01-299% of published values, 100 generations
- **Iterative refinement**: Run GA → narrow parameter bounds around best solution → re-run GA

## Key Equations / Results
- Error metric: Sum of squared errors (SSE) between model and target voltage traces
- Single AP recovery: parameters at wrong values despite perfect AP match (SSE < 0.01 mV²)
- Stochastic + voltage clamp + iterative: all 9 conductances center near true values
- IKr vs IKs: compensatory during plateau, but stochastic stimulation probes different rate-dependent kinetics and separates them
- 4 cell-specific models: prediction error 40-70% lower than generic model on novel stimulation

## Connections to Our Models

### Relevant Engine Components
- **TTP06/ORd ionic models**: The 9 parameters optimized here (maximal conductances) are a subset of what we'd tune in TTP06 (which has ~17 conductances)
- Our Rush-Larsen solver in `cardiac_sim/simulation/classical/solver/ionic_stepping/rush_larsen.py` handles the same ODE structure
- The stochastic stimulation protocol could be implemented via our `Stimulus` API with randomized timing

### Agreements
- Confirms the fundamental non-uniqueness we'd expect when tuning TTP06 to match a target AP
- The IKr/IKs compensation is directly relevant to our TTP06 model (which has both IKr and IKs)

### Disagreements or Gaps
- Guinea pig model (Faber-Rudy), not human ventricular (TTP06/ORd) — but the principles transfer
- Single-cell only — no tissue-level CV fitting (see Pouranbarani 2019 for that)
- GA is less efficient than Bayesian methods for quantifying uncertainty

### Actionable Insights
- **HIGH**: Never fit to a single AP alone — use multi-rate pacing or stochastic stimulation to break conductance degeneracies
- **HIGH**: Use iterative refinement: start with wide bounds GA, narrow bounds around best solutions, re-run. This is computationally cheap and dramatically improves convergence
- **MEDIUM**: Implement stochastic stimulation protocol for our TTP06/ORd fitting — randomly-timed stimuli probe rate-dependent dynamics that reveal hidden parameter sensitivities
- **MEDIUM**: GA with population 500 is a reasonable baseline for 9-16 parameters before moving to Bayesian methods

## Limitations / Caveats
- Guinea pig model only — human ventricular models have different current compositions
- Model-to-model fitting (synthetic ground truth) — experimental validation was on guinea pig myocytes, not human
- GA provides no uncertainty quantification — you get point estimates, not distributions
- Voltage-clamp data requires experimental capability that may not be available for tissue-level calibration
