# Master Knowledge Index

> Table of contents: where knowledge lives, how questions connect.
> Follow the links for detail. Updated by /save-session.

## Research Statement

Grant materials in `ResearchStatement/` (PDF/DOCX). Machine-readable summary to be extracted.

## Knowledge Files

| Topic | Knowledge |
|-------|-----------|
| Boundary Conduction Speedup | [KNOWLEDGE](Research/Active/boundary_conduction_speedup/KNOWLEDGE.md) |
| Ionic Model Optimization | [KNOWLEDGE](Research/Active/ionic_model_optimization/KNOWLEDGE.md) |
| Engine Consolidation | [KNOWLEDGE](Research/Active/engine_consolidation/KNOWLEDGE.md) |
| Geometry-Induced Pacemaking | [KNOWLEDGE](Research/Active/geometry_induced_pacemaking/KNOWLEDGE.md) |
| Mature hiPSC-CM Models | [KNOWLEDGE](Research/Active/mature_hipsc_cm_models/KNOWLEDGE.md) |
| Research Environment Optimization | [KNOWLEDGE](Research/Active/research_environment_optimization/KNOWLEDGE.md) |
| Bidomain Simulation | [KNOWLEDGE](Research/Knowledge/bidomain_simulation.md) |
| LBM Cardiac | [KNOWLEDGE](Research/Knowledge/lbm_cardiac.md) |
| Scar BC Validity | [KNOWLEDGE](Research/Knowledge/scar_bc_validity.md) |

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
