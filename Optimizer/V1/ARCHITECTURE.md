# Engine Tuner — Detailed Architecture

## Input Specification

```python
TuningTargets(
    # Single cell (no diffusion, fast to evaluate)
    apd_90=280.0,              # ms, at CL=1000
    apd_50=210.0,              # ms, optional
    v_rest=-86.0,              # mV
    v_peak=35.0,               # mV
    dvdt_max=300.0,            # V/s (upstroke velocity)
    ap_morphology=None,        # Optional V(t) trace at CL=1000

    # Tissue level (requires simulation)
    cv_longitudinal=65.0,      # cm/s (fiber direction)
    cv_transverse=25.0,        # cm/s (cross-fiber)
    tissue_apd_90=260.0,       # ms (shorter than cell APD due to electrotonic load)

    # Dynamic / rate-dependent (most constraining)
    restitution=[               # (DI_ms, APD_ms) pairs
        (50, 180), (100, 220), (200, 260), (500, 278)
    ],
    erp=240.0,                 # ms, optional
    alternans_cl=280.0,        # ms, optional
)

TuningConfig(
    ionic_model='ttp06',       # or 'ord'
    cell_type='epi',           # or 'endo', 'M'
    param_tier=2,              # 1=core, 2=extended (default), 3=full
    method='bayesopt',         # or 'cmaes' (fallback)
    device='cuda',
    # param_bounds auto-populated from tier; override here if needed
)

# Tier summary:
#   TTP06: Tier 1 = 6 params, Tier 2 = 11, Tier 3 = 17
#   ORd:   Tier 1 = 8 params, Tier 2 = 15, Tier 3 = 25
# See IMPLEMENTATION.md § Tunable Parameter Tiers for full listing.
```

---

## Phase 1: Single-Cell Fit

**Goal**: Find θ_ionic that matches AP shape + restitution curve.

**Engine**: `IonicModel.step()` only — no spatial solver, no diffusion. Each evaluation runs a single-cell ODE for ~5 seconds of simulated time at 3 cycle lengths.

### Simulation Protocol

```
For each θ_ionic candidate:
  1. Pace at CL = 1000 ms for 20 beats (reach steady state)
     → Measure: APD_90, APD_50, V_rest, V_peak, dvdt_max
     → Record: V(t) of last beat (AP morphology)

  2. Pace at CL = 500 ms for 20 beats
     → Measure: APD_90 (rate-dependent)

  3. Pace at CL = 350 ms for 20 beats (near alternans)
     → Measure: APD_90, check for alternans

  4. S1S2 protocol at CL = 1000 ms
     → S2 at DI = 50, 100, 150, 200, 300, 500 ms
     → Measure: APD_90(DI) → restitution curve
```

### Objective Functions

```
f₁ = RMSE(V_model(t), V_target(t))              AP morphology
f₂ = |APD_90_model - APD_90_target| / APD_target  APD accuracy
f₃ = RMSE(restit_model, restit_target) / max(APD)  Restitution match
```

### Optimizer

**Primary**: BoTorch multi-objective BayesOpt (qNEHVI)
- Surrogate: `SingleTaskGP` per objective (3 GPs)
- Acquisition: `qNoisyExpectedHypervolumeImprovement`
- Batch size: 10-20 candidates per iteration
- Budget: 200-500 evaluations total
- Output: Pareto front of ~50 non-dominated θ_ionic solutions

**Fallback**: CMA-ES (`pycma`) with scalarized objective
- Weight: 0.4·f₁ + 0.3·f₂ + 0.3·f₃
- Population: 20, budget: 1000-2000 evaluations

### Key Constraint

Use 3+ cycle lengths to break IKr/IKs degeneracy. At CL=1000, IKr and IKs compensate freely. At CL=350, IKr can't fully recover between beats (τ_recovery ~150 ms) while IKs can (τ_recovery ~500 ms). This differential rate-dependence makes them separable.

---

## Phase 2: Tissue Fit

**Goal**: Find θ_tissue (diffusion parameters) that matches target CV.

**Engine**: Monodomain V5.4 (FDM, Godunov splitting, Rush-Larsen). θ_ionic frozen at best candidate from Phase 1.

### Simulation Geometry

```
Cable 1 (longitudinal CV):
  ══════════════════════════▶
  20 mm × 1 node, dx = 0.1 mm
  Stimulus at x < 1 mm
  Measure activation time at x = 5 mm and x = 15 mm
  CV_L = 10 mm / (t_15 - t_5)

Cable 2 (transverse CV):
  Same geometry, diffusion coefficient = D_transverse
  CV_T = 10 mm / (t_15 - t_5)

Slab (2D, tissue APD):
  ┌────────────────┐
  │                │  10 mm × 10 mm, dx = 0.1 mm
  │   ●───▶        │  D = [[D_L, 0], [0, D_T]]
  │  stim   probe  │  Stimulus at left edge
  │                │  Measure APD at center (5, 5)
  └────────────────┘
```

### Engine (V1: Monodomain only)

```
params: D_long, D_trans
solver: MonodomainSimulation (V5.4, FDM discretization)
config: Godunov splitting, Rush-Larsen ionic, ForwardEuler diffusion
```

Bidomain and LBM engine adapters are deferred to V2.

### Analytical Warm-Start

Before running BayesOpt, compute an initial guess:
```
D_L_init = (CV_L_target / CV_L_baseline)² × D_baseline
D_T_init = (CV_T_target / CV_T_baseline)² × D_baseline
```
CV scales as √D for the monodomain equation, so this gets within ~10%.

### Optimizer

**Primary**: BoTorch single-objective BayesOpt (qEI)
- Objective: `(CV_L - target)² + (CV_T - target)²`
- Initial points: analytical warm-start + 5 random perturbations
- Budget: 20-50 evaluations
- Convergence: stop when CV error < 1%

---

## Phase 3: Joint Refinement

**Goal**: Fine-tune θ_ionic + θ_tissue together, accounting for the tissue APD ≠ single-cell APD coupling.

**Why needed**: Single-cell APD = 280 ms, but in tissue the electrotonic loading from coupled neighbors shortens it to ~260 ms. Phase 1 doesn't know about this. Phase 2 doesn't tune ionic parameters. The coupling is only visible when both are varied together.

### Step A: Generate Training Data

```
1. Take Phase 1 Pareto front (N ≈ 50 θ_ionic candidates)
2. Perturb Phase 2 D values (M ≈ 10 perturbations)
3. Total: ~500 (θ_ionic, θ_tissue) combinations
4. For each: run full 2D tissue simulation
5. Measure: CV_L, CV_T, tissue_APD_90, tissue restitution (S1S2)
```

### Step B: Build GP Emulator

```
1. Stack all restitution curves (N_samples × N_DI_points)
2. PCA: 3 components capture >99% variance
3. GP regression: (θ_ionic, θ_tissue) → PCA coordinates
   Also: (θ_ionic, θ_tissue) → CV_L, CV_T, tissue_APD
4. Result: emulator that predicts all targets in microseconds
```

Using `sklearn.decomposition.PCA` + `sklearn.gaussian_process.GaussianProcessRegressor` or `gpytorch` for GPU acceleration.

### Step C: Optimize on Emulator

**Primary**: NSGA-II (`pymoo`) on the GP emulator
- Evals are microseconds — NSGA-II's inefficiency doesn't matter
- Population: 200, generations: 500 (100K total evals, <1 minute)
- 4 objectives (all normalized to [0, 1]):
  - f₁ = |CV_L - target| / target
  - f₂ = |CV_T - target| / target
  - f₃ = |tissue_APD - target| / target
  - f₄ = RMSE(restitution) / max(APD)

### Step D: Validate on Real Simulator

- Take top 10 candidates from Pareto front
- Run full tissue simulation for each
- If real sim disagrees with emulator by >5%: add point to training data, re-fit GP, re-optimize (active learning)
- Select final θ* as best real-simulation result

---

## Phase 4: Validation

**Goal**: Confirm tuned parameters generalize beyond the fitting conditions.

All tests use the final frozen θ* = θ_ionic* + θ_tissue*.

### Test 1: Novel Pacing Rates

```
CL = 2000, 800, 600, 400, 300 ms (none used in fitting)
Check: APD(CL) follows expected restitution
PASS if: APD error < 5% at all CLs
```

### Test 2: CV in Different Geometries

```
• 1D cable (should match Phase 2 exactly)
• 2D plane wave (bulk CV, no boundary effects)
• 2D with scar boundary (Kleber speedup — see Q5)
PASS if: CV within ±2% of target in bulk
```

### Test 3: Stability

```
• Pace at 2× threshold stimulus (robustness to stimulus strength)
• Pace at 0.5× threshold (should fail to capture — confirms threshold)
• Run for 20 beats at CL=1000 — no APD drift (steady state reached)
• ERP measurement — within physiological range (200-300 ms typically)
```

### Output

```python
TuningResult(
    theta_ionic={
        'GNa': 1.05, 'GNaL': 0.88, 'GCaL': 0.92,
        'GKr': 1.15, 'GKs': 0.78, 'GK1': 1.02,
        'Gto': 0.95, 'GpCa': 1.10,
    },
    theta_tissue={
        'D_longitudinal': 0.00120,  # cm²/ms
        'D_transverse':   0.00032,  # cm²/ms
    },
    validation=ValidationReport(
        cv_long=64.8,       # target: 65, error: 0.3%
        cv_trans=25.2,       # target: 25, error: 0.8%
        tissue_apd=259,      # target: 260, error: 0.4%
        restitution_rmse=2.1,# ms
        status='PASS',
    ),
    pareto_front=[...],      # all non-dominated solutions
    gp_emulator=emulator,    # reusable for future queries
)
```

---

## Class Structure

```
Optimizer/
├── README.md
├── ARCHITECTURE.md          (this file)
├── RESEARCH_BASIS.md        (literature connections)
│
├── tuner/
│   ├── __init__.py
│   ├── config.py            TuningTargets, TuningConfig
│   ├── pipeline.py          EngineTuner.tune() orchestrator
│   │
│   ├── cell_fitter.py       Phase 1: single-cell BayesOpt
│   ├── tissue_fitter.py     Phase 2: CV calibration
│   ├── joint_refiner.py     Phase 3: GP emulator + NSGA-II
│   ├── validator.py         Phase 4: validation tests
│   │
│   ├── objectives/
│   │   ├── ap_metrics.py    APD, dvdt_max, V_rest, morphology RMSE
│   │   ├── cv_metrics.py    CV measurement from activation times
│   │   └── restitution.py   S1S2 protocol, restitution curve fitting
│   │
│   └── surrogate/
│       ├── emulator.py      PCA + GP emulator (Coveney approach)
│       └── active_learn.py  Re-fit GP when predictions diverge
│
└── tests/
    ├── test_cell_fitter.py
    ├── test_tissue_fitter.py
    └── test_pipeline.py
```

---

## Cost Estimate

| Phase | Simulations | Time/sim | GPU Total |
|-------|-------------|----------|-----------|
| 1. Cell fit | ~300 (BayesOpt) | ~0.5 s | ~3 min |
| 2. Tissue CV | ~30 | ~2 min | ~1 hr |
| 3. Joint (training) | 500 | ~5 min | ~4 hr |
| 3. Joint (optimize) | 100K | ~1 μs (surrogate) | ~1 sec |
| 3. Joint (validate) | 10 | ~5 min | ~50 min |
| 4. Validation | ~20 | ~5 min | ~1.5 hr |
| **Total** | **~860 real sims** | | **~7 hours** |

Compare: NSGA-II without surrogate would need ~20K tissue sims = ~70 days on GPU.

---

## Dependencies

```
Already installed: torch, scipy, numpy, matplotlib
Need to install:   botorch, gpytorch, cma, pymoo, SALib

pip install botorch gpytorch cma pymoo SALib
```
