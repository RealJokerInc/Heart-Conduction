# Engine Consolidation — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan this in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
Extract shared code (ionic, mesh, stimulus, conductivity) from three engines into `cardiac_core/`, provide a unified simulation API, eliminate 15+ duplicated files and the `sys.modules` hack. Phase 0 (API wrapper layer, 34 tests) is done. Next: move ionic models into `cardiac_core/ionic/` as the single copy (Phase 1).

## Next Step
Phase 1: Move ionic models (TTP06, ORd) into `cardiac_core/ionic/`. All three engines import from there. Delete the engine-local copies. Verify all 149+ tests pass.

## Thread
### 2026-03-16: The core tension is engine-centric vs. research-centric layout
Engines serve multiple research questions, but the directory structure forces navigation by engine. A single experiment may touch Bidomain V1 and Monodomain V5.4, yet there is no natural place for it. Proposed restructuring: `Engines/` top-level with `cardiac_core/` as the shared package, `Pipelines/` for optimizer/surrogate/builder, `Research/` for writing only.

### 2026-03-16: Three proposed structures, iterated to final form
First proposal grouped everything under `Engines/` and `Pipelines/`. User pushed back on groupings. Second revision separated concerns more cleanly. Third iteration established the principle: Research = writing (no .py files), Engines = code, cross-linked via MASTER.md and EXPERIMENT.md backlinks.

### 2026-03-16: Identified the experiment gap
Traced the research cycle (Hypothesis -> Script -> Run -> Outputs -> Analysis -> Finding -> Knowledge) and found each step lived in a different location with no links between them. Scripts lived in engine test directories, outputs were ephemeral, and analysis was manual. Solved by adding `experiments/` directories inside engines with EXPERIMENT.md backlinks to research questions.

### 2026-03-16: Chi/Cm audit revealed two valid formulations
Audited chi/Cm handling across all three engines. Found Formulation A (V5.4: chi*Cm in mass term, ionic solver does NOT divide by Cm) and Formulation B (Bidomain V1, LBM V1: D pre-scaled, ionic solver divides by Cm). Both produce identical results when Cm=1.0. Decision: keep both, unify at the API level. Converting V5.4 would risk 77 tests for no practical benefit since Cm is always 1.0 (ionic models output pA/pF).

### 2026-03-16: Diffusion tensor encoding differs by method but is reconcilable
Mapped what each discretization method receives (FDM 5pt gets Dxx/Dyy, FDM 9pt gets full tensor, LBM D2Q5 gets scalar, D2Q9 MRT gets full tensor). ConductivityConfig can be the single entry point: user provides sigma, it converts to D with chi/Cm in one place, then each engine extracts what it needs.

### 2026-03-17: Phase 0 completed — API wrapper with 34 tests
Built `cardiac_core/` as an API wrapper: `monodomain()`, `bidomain()`, `lbm()` functions return `CardiacSimulation` with `.run()` generator. File format: `CardiacMeshData` dataclass with `.npz` save/load. Verified wrapper output matches direct engine construction exactly (atol=1e-10). The `_prepare_engine()` hack flushes `sys.modules` because both engines use `cardiac_sim` namespace. This is temporary; goes away in Phase 1.

### 2026-03-17: Code duplication inventory — 15+ files across 3 engines
IonicModel ABC + lut.py (3 copies, identical), TTP06 5 files (3 copies, identical), ORd 6 files (3 copies, identical), PCG solver (2 copies, minor divergence), splitting strategies (2 copies, identical logic), stimulus protocol (2 copies, semantic difference: += vs =), StructuredGrid (2 copies, bidomain adds boundary_spec).

### 2026-03-17: Decided LBM V1 is canonical over V5.4's LBM
LBM V1 has more features: MRT collision, D2Q9 lattice, 3 boundary condition types, torch.compile kernel fusion. V5.4's LBM is simpler but less capable. Decision: LBM V1 is the canonical implementation.

### 2026-03-17: Full project document/folder map established
Mapped every document and folder in the project, clarifying what belongs where. Established conventions: Research/Active for open questions, Research/Complete for answered questions (read-only KNOWLEDGE.md), Research/Knowledge for promoted findings, experiments inside engine directories.

## Failed Approaches
- **Flat engine-centric structure** (2026-03-16) — failed because: engines serve multiple research questions, making it impossible to find all work related to a single question. No natural place for cross-engine experiments.
- **First proposed restructure** (2026-03-16) — failed because: user wanted different groupings; initial Pipelines/Research separation didn't match actual workflow.
- **Converting V5.4 to Formulation B** (2026-03-16) — rejected because: would risk 77 passing tests for zero practical benefit since Cm is always 1.0. Both formulations are mathematically correct.
- **Merging solver internals into cardiac_core** (2026-03-16) — rejected because: solvers are engine-specific (decoupled GS for bidomain, CN/BDF for monodomain, BGK/MRT for LBM). Only shared code (ionic, mesh, stimulus) should be unified.
- **sys.modules hack as permanent solution** (2026-03-17) — recognized as temporary: `_prepare_engine()` flushes modules because both engines use `cardiac_sim` namespace. Acceptable for Phase 0 wrapper but must be eliminated when shared code moves into `cardiac_core/`.

## Session Log
