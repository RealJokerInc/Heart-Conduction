# Engine Tuner V1

Automated parameter optimization for cardiac ionic models (TTP06, O'Hara-Rudy) targeting specific CV, APD, and restitution properties.

**V1 scope**: Monodomain Engine V5.4 only. Single-engine pass to validate the optimization strategy works end-to-end on GPU, before extending to Bidomain and LBM in V2.

## The Problem

Fitting a cardiac ionic model to a single AP waveform is **non-unique**: many parameter combinations produce identical AP shapes but different tissue-level behavior (CV, membrane resistance). IKr/IKs compensation during the plateau is the primary culprit. The literature (8 papers in `Research/Q8_ionic_model_optimization/`) converges on: single-AP fitting is broken, tissue-level data is required, and rate-dependent dynamics break degeneracies.

## Pipeline

```
  INPUTS                                               OUTPUTS
  ┌─────────────────────┐                              ┌──────────────────────┐
  │ ionic_model: TTP06  │                              │ Tuned θ_ionic        │
  │ cv_long: 65 cm/s    │         ┌────────┐           │ Tuned D_long, D_trans│
  │ cv_trans: 25 cm/s   │────────▶│ TUNER  │──────────▶│ Pareto front         │
  │ apd_90: 280 ms      │         └────────┘           │ Validation report    │
  │ tissue_apd: 260 ms  │                              │                      │
  │ restitution: [...]  │                              └──────────────────────┘
  └─────────────────────┘
```

| Phase | What | Params | Method | Cost (GPU) |
|-------|------|--------|--------|------------|
| **1. Cell Fit** | AP shape + restitution | 8 ionic conductances | BayesOpt qNEHVI | ~3 min |
| **2. Tissue Fit** | CV longitudinal + transverse | D_long, D_trans | BayesOpt qEI | ~1 hr |
| **3. Joint Refine** | Couple ionic + tissue | 10 combined | GP emulator + NSGA-II | ~4 hr |
| **4. Validate** | Novel rates, geometries | frozen θ | One-shot tests | ~1.5 hr |

## V1 Constraints

- **Engine**: Monodomain V5.4 only (FDM, Godunov splitting, Rush-Larsen)
- **Tissue params**: D_long, D_trans (2 params, isotropic per-axis)
- **Geometry**: 1D cables for CV, 2D slab for tissue APD
- **No cross-engine validation** — that's V2

## Optimizer Backbone: BoTorch

**Primary**: BoTorch (Bayesian optimization, PyTorch-native).

| Phase | Acquisition | Why |
|-------|------------|-----|
| 1. Cell | qNEHVI (multi-objective) | Pareto front in ~300 evals vs NSGA-II's ~20K |
| 2. Tissue | qEI (single-objective) | Designed for expensive evals; converges in ~20 |
| 3. Joint | NSGA-II on GP surrogate | Evals are free on surrogate |

**Fallback**: CMA-ES (`pycma`) if GP struggles in 8+ dimensions.

## Integration with Monodomain V5.4

```python
# Phase 1: single-cell (no spatial solver)
from cardiac_sim.ionic.ttp06 import TTP06Model
model = TTP06Model(cell_type='epi', device='cuda')

# Phase 2-4: tissue simulation
from cardiac_sim.simulation.classical import MonodomainSimulation
from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
```

## Key Files

| File | Purpose |
|------|---------|
| `README.md` | This file |
| `ARCHITECTURE.md` | Phase-by-phase design, class structure, objectives |
| `RESEARCH_BASIS.md` | Literature basis (8 papers from Research/Q8) |
| `IMPLEMENTATION.md` | Phased implementation plan with validation criteria |
| `tuner/` | Source code |
| `tests/` | Validation tests |

## Dependencies

```
Already installed: torch, scipy, numpy
Install: pip install botorch gpytorch cma pymoo SALib
```
