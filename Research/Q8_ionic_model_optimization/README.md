# Q8: How do I tune ionic model parameters (TTP06/ORd) to match target CV and APD?

## Short Answer

**Multi-objective optimization** is required because fitting to a single AP waveform is non-unique — many parameter sets produce identical APs but different tissue-level behavior (CV, membrane resistance). The best approaches combine:

1. **Rich target data**: not just one AP, but multi-rate pacing (restitution curves), paired waveforms (control + drug block), and tissue-level CV measurements
2. **Multi-objective or Bayesian methods**: NSGA-II for Pareto front exploration (AP fit vs CV fit tradeoffs), or Bayesian inference (HMC, ABC-SMC) for full posterior distributions over parameters
3. **Small parameter subsets**: tune 6-16 maximal conductance scaling factors, not all model parameters

The key degeneracy is **IKr/IKs compensation** during the plateau — these can only be separated with rate-dependent protocols or selective channel block data. CV is primarily controlled by GNa and the diffusion coefficient D, while APD is controlled by GCaL, GKr, GKs, and GK1.

## Key Files in This Folder

### Tier 1 — Core papers

| File | Citation | Method | Key contribution |
|------|----------|--------|-----------------|
| `pouranbarani_2019_multiobjective_rm.md` | Pouranbarani 2019 | NSGA-II multi-objective | Only paper optimizing for BOTH AP shape AND tissue-level CV via membrane resistance |
| `coveney_2021_bayesian_restitution.md` | Coveney 2021 | GP emulator + Bayesian MCMC | Restitution curve emulators for fast calibration; PCA+GP surrogate |
| `nietoramos_2023_bayesian_hmc.md` | Nieto Ramos 2023 | HMC + ABC-SMC | Full posterior distributions over ionic parameters; parameter identifiability |
| `groenendaal_2015_cell_specific.md` | Groenendaal 2015 | Genetic Algorithm (iterative) | Proves single-AP fitting is non-unique; stochastic stimulation resolves degeneracies |

### hiPSC-CM Model Selection & Maturation

| File | Citation | Key contribution |
|------|----------|-----------------|
| `hipsc_cm_maturation_models.md` | Multi-paper survey (2013–2026) | All published hiPSC-CM models beat spontaneously; IK1 upscaling + If suppression creates quiescent "matured" variant (Verkerk 2019); experimental maturation targets compiled |

### Tier 2 — Supporting methodology

| File | Citation | Method | Key contribution |
|------|----------|--------|-----------------|
| `zhang_2024_gradient_two_waveform.md` | Zhang 2024 | Gradient-based PO | Two-waveform fitting (control + IKr block) breaks parameter correlations |
| `chang_2017_uq_cipa.md` | Chang 2017 | Bootstrap + MCMC (DRAM) | UQ for ORd CiPA model; >60% block needed for reliable IC50 |
| `nietoramos_2022_hmc_cinc.md` | Nieto Ramos 2022 | HMC (NUTS/Stan) | Conference precursor proving HMC scales to 13-parameter cardiac models |
| `cairns_2017_ga_parameterization.md` | Cairns 2017 | Genetic Algorithm | GA baseline for AP model parameterization (paywalled — abstract only) |

## Relevant Papers in `../papers/`

*Papers to be downloaded by user — DOIs listed in individual summary files.*

## Method Comparison

| Method | Pros | Cons | Best for |
|--------|------|------|----------|
| **NSGA-II** (Pouranbarani) | Pareto front shows tradeoffs; handles 16 params | No uncertainty quantification; 10,000 evaluations needed | Exploring AP vs CV tradeoffs |
| **Bayesian HMC** (Nieto Ramos) | Full posterior; identifies degeneracies | Needs differentiable model (JAX/PyTorch); tested only on 5-13 params | Quantifying parameter uncertainty |
| **GP Emulator** (Coveney) | 10^5x speedup after training; works at tissue level | 500 simulations for training; only tested on 5-param model | Restitution-based calibration |
| **GA** (Groenendaal, Cairns) | Simple, robust, handles discontinuities | No uncertainty; can be slow; no gradient information | Baseline; rough parameter search |
| **Gradient PO** (Zhang) | Fast convergence for small param sets | Local optima; needs smooth objective | Fine-tuning 6 conductances |

## Connected Questions

- **Q1** — Spatial discretization affects CV measurement accuracy
- **Q3** — Time stepping affects AP duration measurement (splitting errors)
- **Q5** — Boundary effects on CV must be accounted for in tissue-level fitting
