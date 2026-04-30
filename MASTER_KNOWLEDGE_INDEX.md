# Master Knowledge Index

> Table of contents: where knowledge lives, how questions connect.
> Follow the links for detail. Updated by /save-session.

## Research Statement

Grant materials in `ResearchStatement/` (PDF/DOCX). Machine-readable summary to be extracted.

## Knowledge Files

| Topic | Knowledge | Status |
|-------|-----------|--------|
| Boundary Conduction Speedup | [KNOWLEDGE](Research/Active/boundary_conduction_speedup/KNOWLEDGE.md) | Active — **2026-04-30 BRIDGE CONFIRMED**: Moore-8 stencil + face_mirror BC reproduces John's crescent in monodomain (+486 µs LAT shift, TTP06 EPI). Same mechanism across storage tank / monodomain V5.4 / LBM V1. Cardinal-only OR face_mirror_iso (LBM bounce-back analog) eliminates in all three. V5.4 FDM gained `stencil` + `face_mirror_iso` API; LBM V1 gained `weights_mode` + critical cs2 plumbing fix. |
| Ionic Model Optimization | [KNOWLEDGE](Research/Active/ionic_model_optimization/KNOWLEDGE.md) | Active |
| Engine Consolidation | [KNOWLEDGE](Research/Active/engine_consolidation/KNOWLEDGE.md) | Active |
| Geometry-Induced Pacemaking | [KNOWLEDGE](Research/Active/geometry_induced_pacemaking/KNOWLEDGE.md) | Active |
| Mature hiPSC-CM Models | [KNOWLEDGE](Research/Active/mature_hipsc_cm_models/KNOWLEDGE.md) | Active |
| Research Environment Optimization | [KNOWLEDGE](Research/Active/research_environment_optimization/KNOWLEDGE.md) | Active |
| Surrogate Pipeline | [KNOWLEDGE](Research/Active/surrogate_pipeline/KNOWLEDGE.md) | Active — Neural ODE implemented + first training runs. Ionic MSE 0.047 (500 epochs). KAN for concentrations. dopri5 + backprop-through-solver. 1,988 inference params. |
| Bidomain Parabolic-Parabolic | [KNOWLEDGE](Research/Active/bidomain_parabolic_parabolic/KNOWLEDGE.md) | Active |
| Cardiac ML Harness | [KNOWLEDGE](Research/Active/cardiac_ml_harness/KNOWLEDGE.md) | Implemented 2026-04-20 — NODE parity met (0.00835 ≤ 0.0088), diffusion-stub reusability proven, cutover complete. 80 tests. |
| Bidomain Simulation | [KNOWLEDGE](Research/Knowledge/bidomain_simulation.md) | Promoted |
| LBM Cardiac | [KNOWLEDGE](Research/Knowledge/lbm_cardiac.md) | Promoted |
| Scar BC Validity | [KNOWLEDGE](Research/Knowledge/scar_bc_validity.md) | Promoted |

## Cross-References

- **Boundary Speedup ↔ Scar BC**: Same physics (electrotonic loading at boundaries), opposite BC symmetry. [Boundary](Research/Active/boundary_conduction_speedup/KNOWLEDGE.md) | [Scar](Research/Knowledge/scar_bc_validity.md)
- **Boundary Speedup ↔ Pacemaking**: Source-sink impedance — one affects CV, the other determines pacemaker origin. [Boundary](Research/Active/boundary_conduction_speedup/KNOWLEDGE.md) | [Pacemaking](Research/Active/geometry_induced_pacemaking/KNOWLEDGE.md)
- **Boundary Speedup ↔ LBM**: D2Q9 reproduces Kleber speedup despite ~35% CV baseline offset. [Boundary](Research/Active/boundary_conduction_speedup/KNOWLEDGE.md) | [LBM](Research/Knowledge/lbm_cardiac.md)
- **Mature hiPSC-CM ↔ Optimization**: MHAS13 is the tuning target; maturation provides the quiescent baseline. [Mature](Research/Active/mature_hipsc_cm_models/KNOWLEDGE.md) | [Optimization](Research/Active/ionic_model_optimization/KNOWLEDGE.md)
- **Mature hiPSC-CM ↔ Pacemaking**: PHAS13 (immature) drives pacemaking; MHAS13 (matured) is the negative control. [Mature](Research/Active/mature_hipsc_cm_models/KNOWLEDGE.md) | [Pacemaking](Research/Active/geometry_induced_pacemaking/KNOWLEDGE.md)
- **Optimization ↔ Bidomain**: Tissue-level CV fitting requires bidomain runs; D_eff agrees within 6%. [Optimization](Research/Active/ionic_model_optimization/KNOWLEDGE.md) | [Bidomain](Research/Knowledge/bidomain_simulation.md)
- **Engine Consolidation ↔ All Engines**: Must reconcile two chi/Cm formulations across 149+ tests. [Consolidation](Research/Active/engine_consolidation/KNOWLEDGE.md) | [Bidomain](Research/Knowledge/bidomain_simulation.md) | [LBM](Research/Knowledge/lbm_cardiac.md)
- **LBM ↔ Scar BC**: Bounce-back = Neumann (scar), anti-bounce-back = Dirichlet (bath). [LBM](Research/Knowledge/lbm_cardiac.md) | [Scar](Research/Knowledge/scar_bc_validity.md)
- **Pacemaking ↔ Optimization**: PHAS13 may need optimizer tuning to match experimental beating rates. [Pacemaking](Research/Active/geometry_induced_pacemaking/KNOWLEDGE.md) | [Optimization](Research/Active/ionic_model_optimization/KNOWLEDGE.md)
- **Environment ↔ Consolidation**: Engineering workflow gaps most felt during multi-session engine work. [Environment](Research/Active/research_environment_optimization/KNOWLEDGE.md) | [Consolidation](Research/Active/engine_consolidation/KNOWLEDGE.md)
- **Surrogate ↔ Bidomain**: Surrogate trained on Bidomain V1 output; must reproduce Kleber boundary speedup. [Surrogate](Research/Active/surrogate_pipeline/KNOWLEDGE.md) | [Bidomain](Research/Knowledge/bidomain_simulation.md)
- **Surrogate ↔ Boundary Speedup**: Surrogate validation target — CV_ratio within 10% of simulator. [Surrogate](Research/Active/surrogate_pipeline/KNOWLEDGE.md) | [Boundary](Research/Active/boundary_conduction_speedup/KNOWLEDGE.md)
- **Surrogate ↔ Optimization**: Trained surrogate replaces simulator in optimization loop for fast parameter sweeps. [Surrogate](Research/Active/surrogate_pipeline/KNOWLEDGE.md) | [Optimization](Research/Active/ionic_model_optimization/KNOWLEDGE.md)
- **Surrogate ↔ Consolidation**: Unified engine API simplifies training data generation across engines. [Surrogate](Research/Active/surrogate_pipeline/KNOWLEDGE.md) | [Consolidation](Research/Active/engine_consolidation/KNOWLEDGE.md)
- **Surrogate ↔ Cardiac ML Harness**: Surrogate's ionic NODE is the harness's pilot. Harness's `Surrogate/surrogate/training/node_step.py` adapter reproduces Session-25 parity. [Harness](Research/Active/cardiac_ml_harness/KNOWLEDGE.md) | [Surrogate](Research/Active/surrogate_pipeline/KNOWLEDGE.md)
- **Cardiac ML Harness ↔ Optimization**: Future BayesOpt objective-evaluation consumer — via `Trainer.evaluate()` entry point (OPEN-2, deferred). [Harness](Research/Active/cardiac_ml_harness/KNOWLEDGE.md) | [Optimization](Research/Active/ionic_model_optimization/KNOWLEDGE.md)
