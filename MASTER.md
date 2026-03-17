# Heart-Conduction — Master Progress

Last updated: 2026-03-17

## Active Research

| Question | Engines | Status | Next Step |
|----------|---------|--------|-----------|
| [Boundary conduction speedup](Research/Active/boundary_conduction_speedup/) | Bidomain, LBM | CV ratio 1.071→1.131 confirmed | Anisotropic boundary study |
| [Ionic model optimization](Research/Active/ionic_model_optimization/) | V5.4, Optimizer | Docs done, MHAS13 test run | Optimizer V1 Phase 1 |
| [Mature hiPSC-CM models](Research/Active/mature_hipsc_cm_models/) | V5.4, Bidomain | MHAS13 complete (quiescent, V_rest=-83.7mV) | Tissue validation, ORd-based variant |
| [Engine consolidation](Research/Active/engine_consolidation/) | All | **Phase 0 DONE**: unified file format + `monodomain()`/`bidomain()`/`lbm()` API (34 tests) | Phase 1: extract shared ionic/mesh |
| [Geometry-induced pacemaking](Research/Active/geometry_induced_pacemaking/) | V5.4, LBM, Bidomain | Just started | Literature review + PHAS13 baseline |
| [Research environment optimization](Research/Active/research_environment_optimization/) | All (meta) | 3-doc architecture finalized (KNOWLEDGE + IDEALOG + PLAN) | Implement: extend 3 skills, add 7 new (reason, blueprint, save-session, audit, verify, build-fix, strategic-compact) |

## Complete Research

| Question | Key Answer | Knowledge |
|----------|-----------|-----------|
| [Bidomain simulation](Research/Complete/bidomain_simulation/) | FDM/FEM/FVM + 3-tier solver + Strang/RL/CN | [Knowledge file](Research/Knowledge/bidomain_simulation.md) |
| [LBM for cardiac EP](Research/Complete/lbm_cardiac/) | D2Q5 iso, D2Q9 aniso, MRT > BGK, 34 tests | [Knowledge file](Research/Knowledge/lbm_cardiac.md) |
| [Scar BC validity](Research/Complete/scar_bc_validity/) | Neumann correct; Dirichlet invalid (voltage source artifact) | [Knowledge file](Research/Knowledge/scar_bc_validity.md) |

## Backlog

| Question | Trigger to Activate |
|----------|-------------------|
| [Fetal heart development](Research/Backlog/fetal_heart_development/) | When boundary speedup validated in 3D |

## Engines

| Engine | Tests | Status | Location | Experiments |
|--------|-------|--------|----------|-------------|
| Monodomain V5.4 | 77 | All phases done | [Engines/monodomain_v5.4](Engines/monodomain_v5.4/) | [experiments/](Monodomain/Engine_V5.4/experiments/) |
| Bidomain V1 | 38+ | All phases done | [Engines/bidomain_v1](Engines/bidomain_v1/) | [experiments/](Bidomain/Engine_V1/experiments/) |
| LBM V1 | 34 | All phases done | [Engines/lbm_v1](Engines/lbm_v1/) | [experiments/](Monodomain/LBM_V1/experiments/) |
| Cross-engine | — | — | [Engines/cross_engine](Engines/cross_engine/) | Per research question |
| cardiac_core | — | Not started | — | — |

## Pipelines

| Pipeline | Status | Depends On | Location | Experiments |
|----------|--------|------------|----------|-------------|
| Optimizer V1 | Docs done, implementing | V5.4 | [Pipelines/optimizer](Pipelines/optimizer/) | [experiments/](Optimizer/experiments/) |
| Surrogate | Docs done, not started | Bidomain V1 | [Pipelines/surrogate](Pipelines/surrogate/) | [experiments/](Surrogate/experiments/) |
| Builder | Designed | V5.4 | [Pipelines/builder](Pipelines/builder/) | — |

## Structure

```
Research/   = writing (literature, knowledge, figures, paper summaries)
Engines/    = code (engine source, tests, experiment scripts, outputs)
Pipelines/  = code (optimizer, surrogate, builder + their experiments)
```

Research questions reference experiments in Engines/ by path.
Engine EXPERIMENT.md files link back to their research question.
