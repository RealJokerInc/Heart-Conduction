# Heart-Conduction — Master Progress

Last updated: 2026-04-28

## Active Research

| Question | Engines | Status | Next Step |
|----------|---------|--------|-----------|
| [Boundary conduction speedup](Research/Active/boundary_conduction_speedup/) | Bidomain, LBM | CV ratio 1.071→1.131; clean inverse-crescent BC found (same-cell specular, 2026-05-28) | Anisotropic study; (α,β,γ) simplex on HBB↔same-cell-specular |
| [Source-sink mismatch investigation](Research/Active/source_sink_mismatch_investigation/) | V5.4, LBM | **Spun off 2026-06-06** from boundary speedup. Target = Ciaccio 2018 Fig 4 thickness-driven source-sink wavefront curvature; deep-research confirmed augmented/thickness-weighted monodomain `(1/T)nabla.(T D nablaV)` as the fix (Biktasheva PRL 2019) | Implement thickness-weighted operator in V5.4; reproduce Fig-4 A-D + block |
| [Ionic model optimization](Research/Active/ionic_model_optimization/) | V5.4, Optimizer | Docs done, MHAS13 test run | Optimizer V1 Phase 1 |
| [Mature hiPSC-CM models](Research/Active/mature_hipsc_cm_models/) | V5.4, Bidomain | MHAS13 complete (quiescent, V_rest=-83.7mV) | Tissue validation, ORd-based variant |
| [Engine consolidation](Research/Active/engine_consolidation/) | All | **Both north-star goals SHIPPED.** `cardiac_core` is one self-contained package AND a public pip-installable library (`github.com/RealJokerInc/cardiac-core`). Media layer complete: **video 2026-07-23 + image/trace 2026-07-25** (580 tests) | Optional: `annotations=` on Image/Trace; auto-generated Object Atlas + drift canary; settle monorepo↔standalone sync |
| [Geometry-induced pacemaking](Research/Active/geometry_induced_pacemaking/) | V5.4, LBM, Bidomain | Just started | Literature review + PHAS13 baseline |
| [Geometry-induced reentry](Research/Active/geometry_induced_reentry/) | LBM (primary), cardiac_core, V5.4, Bidomain | **Just started 2026-06-24** | Planar wave hitting a simple 2-D inexcitable infarct (no-flux/bounce-back) -> wavebreak/reentry. Lit search (Agladze-Panfilov 1994, Cabo 1996) + LBM planar-wave-meets-circle baseline |
| [Research environment optimization](Research/Active/research_environment_optimization/) | All (meta) | **Implementation complete**: 16 skills, 3-doc architecture, PreCompact hook | Real-world testing of new workflow |
| [Surrogate pipeline](Research/Active/surrogate_pipeline/) | Bidomain V1, V5.4 | Docs done, no code yet | Phase 1A: single-cell data generation |
| [Cardiac ML harness](Research/Active/cardiac_ml_harness/) | None (project-wide) | IMPLEMENTED. NODE parity met. Reusability proved via diffusion stub. | Consumer migrations (diffusion ResNet, BayesOpt wrapper) |
| [Bidomain parabolic-parabolic](Research/Active/bidomain_parabolic_parabolic/) | Bidomain V1 | Just started | Literature review of PP formulation |
| [LBM-EP](Research/Active/lbm_ep/) | LBM V1, Bidomain V1 | Reopened 2026-04-19 (was lbm_cardiac, complete 2026-03-16) | Audit current LBM V1 — find highest-leverage gap (anisotropy/boundary/tuning) |
| [Mesh builder](Research/Active/mesh_builder/) | All (Builder-first) | Just started | Survey existing `Builder/` package — extend or replace |
| [Monthly report pipeline](Research/Active/monthly_report_pipeline/) | None (meta) | Spec extracted (Zimmerman format V1, 4/20/2026) | Decide: rush manual April report (due 4/30) or target May as first pipeline run |
| [Textbook](Research/Active/textbook/) | All (documents each) | **2026-07-02: full audit-remediation COMPLETE** (5-phase PLAN, commits 7590635→5457c99). Canonical source = `website/chapters/*.html` (monolithic archived after two-copy fork). Fixed all correctness bugs + book-wide cross-refs + figure integrity; added worked examples + 9 figures (40 total); PDF pipeline restored → `Cardiac_Computational_Modeling.pdf` (195 pp). | Deferred: Ch 8 full split, Reader-B non-code path, Ch 4 literature images, §18.1 trim |

## Complete Research

| Question | Key Answer | Knowledge |
|----------|-----------|-----------|
| [Bidomain simulation](Research/Complete/bidomain_simulation/) | FDM/FEM/FVM + 3-tier solver + Strang/RL/CN | [Knowledge file](Research/Knowledge/bidomain_simulation.md) |
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
| LBM V1 | 34 | All phases done | [LBM/Engine_V1](LBM/Engine_V1/) | [experiments/](LBM/Engine_V1/experiments/) |
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
