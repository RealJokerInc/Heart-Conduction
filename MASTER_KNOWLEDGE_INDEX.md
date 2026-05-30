# Master Knowledge Index

> Table of contents: where knowledge lives, how questions connect.
> Follow the links for detail. Updated by /save-session.

## Research Statement

Grant materials in `ResearchStatement/` (PDF/DOCX). Machine-readable summary to be extracted.

## Knowledge Files

| Topic | Knowledge | Status |
|-------|-----------|--------|
| Boundary Conduction Speedup | [KNOWLEDGE](Research/Active/boundary_conduction_speedup/KNOWLEDGE.md) | Active — **2026-04-30 BRIDGE CONFIRMED**: Moore-8 + face_mirror reproduces crescent in monodomain V5.4 (+486 µs, TTP06 EPI). **2026-05-02**: equilibrium argument graduated — Fickian flux is structurally sign-locked to crescent. **2026-05-14**: corrected cross-engine mapping — HBB (LBM) ≡ face_mirror (PDE), specular (LBM) ≡ face_mirror_iso (PDE). Verified via 14-case sweep (`diag_dvdt_*.py` + HDF5 in `data/`). Discovered **horizontal redirect** — novel BC that biases toward inverse crescent (sustained boundary speedup, growing to −3.1 ms at col 38 under D2Q9 uniform). Mass-leak caveat at corners flagged for next session. |
| Ionic Model Optimization | [KNOWLEDGE](Research/Active/ionic_model_optimization/KNOWLEDGE.md) | Active |
| Engine Consolidation | [KNOWLEDGE](Research/Active/engine_consolidation/KNOWLEDGE.md) | Active — **2026-05-30**: Monodomain **V5.5** forked from V5.4 (Cm-correct reaction `/Cm`; dead internal LBM path dropped). Validated: Cm=1 bit-identical to V5.4, exact 1/Cm scaling (3.55e-15), Bidomain V1 cross-check (0.0%/1.1%). Physics note: the Cm time-dilation invariant is FALSE (gates carry no Cm). `cardiac_core` extraction (Phases 1–5) still pending; build against V5.5. |
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
- **Engine Consolidation ↔ Boundary Conduction Speedup**: V5.5 dropped V5.4's dead internal LBM path after confirming the boundary work runs solely on the separate `LBM/Engine_V1`. [Consolidation](Research/Active/engine_consolidation/KNOWLEDGE.md) | [Boundary](Research/Active/boundary_conduction_speedup/KNOWLEDGE.md)
- **LBM ↔ Scar BC**: Bounce-back = Neumann (scar), anti-bounce-back = Dirichlet (bath). [LBM](Research/Knowledge/lbm_cardiac.md) | [Scar](Research/Knowledge/scar_bc_validity.md)
- **Pacemaking ↔ Optimization**: PHAS13 may need optimizer tuning to match experimental beating rates. [Pacemaking](Research/Active/geometry_induced_pacemaking/KNOWLEDGE.md) | [Optimization](Research/Active/ionic_model_optimization/KNOWLEDGE.md)
- **Environment ↔ Consolidation**: Engineering workflow gaps most felt during multi-session engine work. [Environment](Research/Active/research_environment_optimization/KNOWLEDGE.md) | [Consolidation](Research/Active/engine_consolidation/KNOWLEDGE.md)
- **Surrogate ↔ Bidomain**: Surrogate trained on Bidomain V1 output; must reproduce Kleber boundary speedup. [Surrogate](Research/Active/surrogate_pipeline/KNOWLEDGE.md) | [Bidomain](Research/Knowledge/bidomain_simulation.md)
- **Surrogate ↔ Boundary Speedup**: Surrogate validation target — CV_ratio within 10% of simulator. [Surrogate](Research/Active/surrogate_pipeline/KNOWLEDGE.md) | [Boundary](Research/Active/boundary_conduction_speedup/KNOWLEDGE.md)
- **Surrogate ↔ Optimization**: Trained surrogate replaces simulator in optimization loop for fast parameter sweeps. [Surrogate](Research/Active/surrogate_pipeline/KNOWLEDGE.md) | [Optimization](Research/Active/ionic_model_optimization/KNOWLEDGE.md)
- **Surrogate ↔ Consolidation**: Unified engine API simplifies training data generation across engines. [Surrogate](Research/Active/surrogate_pipeline/KNOWLEDGE.md) | [Consolidation](Research/Active/engine_consolidation/KNOWLEDGE.md)
- **Surrogate ↔ Cardiac ML Harness**: Surrogate's ionic NODE is the harness's pilot. Harness's `Surrogate/surrogate/training/node_step.py` adapter reproduces Session-25 parity. [Harness](Research/Active/cardiac_ml_harness/KNOWLEDGE.md) | [Surrogate](Research/Active/surrogate_pipeline/KNOWLEDGE.md)
- **Cardiac ML Harness ↔ Optimization**: Future BayesOpt objective-evaluation consumer — via `Trainer.evaluate()` entry point (OPEN-2, deferred). [Harness](Research/Active/cardiac_ml_harness/KNOWLEDGE.md) | [Optimization](Research/Active/ionic_model_optimization/KNOWLEDGE.md)
