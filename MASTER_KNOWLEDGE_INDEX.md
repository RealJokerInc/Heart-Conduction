# Master Knowledge Index

> Index book: where knowledge lives, how questions connect.
> NOT a copy of findings -- follow the links for detail.
> Updated by /save-session after each research session.

## Research Statement

Grant materials exist in `ResearchStatement/` (PDF and DOCX versions of "Research Grant Support"), but no machine-readable research statement has been extracted. This section should be populated with the researcher's goals and thesis direction once summarized from the grant documents.

## Knowledge Index

| Question | Status | One-Liner | Knowledge |
|----------|--------|-----------|-----------|
| Boundary Conduction Speedup | Active | Bath-coupled tissue boundaries speed conduction by ~7-13% (Kleber effect) due to asymmetric bidomain BCs that short-circuit the extracellular return path. | [KNOWLEDGE](Research/Active/boundary_conduction_speedup/KNOWLEDGE.md) |
| Ionic Model Optimization | Active | Multi-objective BayesOpt tunes 10 ionic conductances to match CV and APD simultaneously, overcoming the fundamental IKr/IKs degeneracy of single-AP fitting. | [KNOWLEDGE](Research/Active/ionic_model_optimization/KNOWLEDGE.md) |
| Engine Consolidation | Active | Three engines share 15+ duplicated files across two chi/Cm formulations; `cardiac_core/` is the target unified codebase with Phase 0 (API wrapper) complete. | [KNOWLEDGE](Research/Active/engine_consolidation/KNOWLEDGE.md) |
| Geometry-Induced Pacemaking | Active | Sharp tissue tips and narrow exits reduce electrotonic load below the suppression threshold, enabling spontaneously-beating hiPSC-CMs to organize into geometry-determined pacemaker sites. | [KNOWLEDGE](Research/Active/geometry_induced_pacemaking/KNOWLEDGE.md) |
| Mature hiPSC-CM Models | Active | Injecting TTP06 IK1 at the Verkerk 2019 critical conductance and suppressing If converts spontaneously-beating Paci 2013 into quiescent MHAS13 suitable for tissue simulation. | [KNOWLEDGE](Research/Active/mature_hipsc_cm_models/KNOWLEDGE.md) |
| Research Environment Optimization | Active | Context is the scarcest resource in AI-assisted research; our skills cover the research lifecycle but have zero coverage for engineering workflow, session persistence, and compaction management. | [KNOWLEDGE](Research/Active/research_environment_optimization/KNOWLEDGE.md) |
| Bidomain Simulation | Complete | Three-tier spectral/PCG/GMG solver auto-selected from boundary type, with face-based FDM for SPD Laplacians and Strang splitting for second-order time accuracy. | [KNOWLEDGE](Research/Knowledge/bidomain_simulation.md) |
| LBM Cardiac | Complete | LBM replaces PDE discretization with collide-stream kinetics (D2Q5 isotropic, D2Q9 anisotropic, BGK/MRT); CV runs ~35% above FDM at same resolution but converges with refinement. | [KNOWLEDGE](Research/Knowledge/lbm_cardiac.md) |
| Scar BC Validity | Complete | Scar boundaries require Neumann (no-flux) on both domains because dead tissue has no ion channels, no gap junctions, and no voltage source -- Dirichlet at scar is unphysical and produces a fictitious Kleber speedup. | [KNOWLEDGE](Research/Knowledge/scar_bc_validity.md) |

## Cross-References

- **Boundary Conduction Speedup <-> Scar BC Validity**: The Kleber speedup requires asymmetric BCs (Neumann intracellular, Dirichlet extracellular); scar has symmetric Neumann on both domains, so no speedup occurs -- misapplying Dirichlet at scar produces a computational artifact. [Boundary](Research/Active/boundary_conduction_speedup/KNOWLEDGE.md) | [Scar BC](Research/Knowledge/scar_bc_validity.md)

- **Boundary Conduction Speedup <-> Geometry-Induced Pacemaking**: Both phenomena arise from the same source-sink impedance physics -- reduced electrotonic loading at geometric features -- but one affects conduction velocity while the other determines pacemaker origin. [Boundary](Research/Active/boundary_conduction_speedup/KNOWLEDGE.md) | [Pacemaking](Research/Active/geometry_induced_pacemaking/KNOWLEDGE.md)

- **Boundary Conduction Speedup <-> LBM Cardiac**: D2Q9 with Dirichlet BC captures the Kleber boundary speedup, confirming LBM can reproduce bidomain-like boundary effects despite its ~35% CV baseline offset. [Boundary](Research/Active/boundary_conduction_speedup/KNOWLEDGE.md) | [LBM](Research/Knowledge/lbm_cardiac.md)

- **Mature hiPSC-CM Models <-> Ionic Model Optimization**: MHAS13 is the tuning target for Optimizer V1; the maturation pathway (IK1 injection + If suppression) provides the quiescent baseline that the optimizer then refines to match experimental CV and APD. [Mature](Research/Active/mature_hipsc_cm_models/KNOWLEDGE.md) | [Optimization](Research/Active/ionic_model_optimization/KNOWLEDGE.md)

- **Mature hiPSC-CM Models <-> Geometry-Induced Pacemaking**: PHAS13 (immature, spontaneous) drives geometry-induced pacemaking experiments; MHAS13 (matured, quiescent) serves as the negative control that should show no geometry-dependent spontaneous activity. [Mature](Research/Active/mature_hipsc_cm_models/KNOWLEDGE.md) | [Pacemaking](Research/Active/geometry_induced_pacemaking/KNOWLEDGE.md)

- **Ionic Model Optimization <-> Bidomain Simulation**: Tissue-level CV fitting in the optimizer requires running bidomain simulations, and D_eff values must agree between monodomain and bidomain engines (confirmed within 6%). [Optimization](Research/Active/ionic_model_optimization/KNOWLEDGE.md) | [Bidomain](Research/Knowledge/bidomain_simulation.md)

- **Engine Consolidation <-> All Engines**: Consolidation directly depends on all three engine architectures (Bidomain V1, Monodomain V5.4, LBM V1) and must reconcile their two chi/Cm formulations without breaking 149+ existing tests. [Consolidation](Research/Active/engine_consolidation/KNOWLEDGE.md) | [Bidomain](Research/Knowledge/bidomain_simulation.md) | [LBM](Research/Knowledge/lbm_cardiac.md)

- **LBM Cardiac <-> Scar BC Validity**: LBM bounce-back BC implements no-flux (Neumann) at scar boundaries; anti-bounce-back implements Dirichlet at bath interfaces -- the BC type directly determines which conduction effect (if any) appears at each boundary. [LBM](Research/Knowledge/lbm_cardiac.md) | [Scar BC](Research/Knowledge/scar_bc_validity.md)

- **Geometry-Induced Pacemaking <-> Ionic Model Optimization**: The pAP/PHAS13 models used for pacemaking experiments may need parameter tuning via the optimizer to match experimental hiPSC-CM beating rates before geometry sweeps are meaningful. [Pacemaking](Research/Active/geometry_induced_pacemaking/KNOWLEDGE.md) | [Optimization](Research/Active/ionic_model_optimization/KNOWLEDGE.md)

- **Research Environment Optimization <-> Engine Consolidation**: The engineering workflow gaps (no planning, session persistence, or verification skills) are most acutely felt during multi-session engine consolidation work that spans compaction boundaries. [Environment](Research/Active/research_environment_optimization/KNOWLEDGE.md) | [Consolidation](Research/Active/engine_consolidation/KNOWLEDGE.md)
