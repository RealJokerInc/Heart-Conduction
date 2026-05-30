# Engine Consolidation — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan this in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
**V5.5 DETOUR (2026-05-29)** — before Phase 1, fork V5.4 → independent **Monodomain Engine_V5.5** that fixes the Formulation-A reaction bug (ionic step missing `/Cm`). V5.4 stays frozen as the validated baseline (77 tests, backup). V5.5 becomes the Cm-correct monodomain we develop on going forward. After V5.5 is validated, resume consolidation: extract shared code (ionic, mesh, stimulus, conductivity) into `cardiac_core/`, eliminate the 15+ duplicates and the `sys.modules` hack. Phase 0 (API wrapper, 34 tests) done.

## Next Step
Blueprint + build **Engine_V5.5**: full copy of V5.4, then divide the reaction voltage update by tissue Cm in the two operator-split ionic steppers. Validate via (a) Cm=1 regression bit-identical to V5.4, (b) Cm≠1 time-dilation invariant. THEN return to consolidation Phase 1 (move ionic models into `cardiac_core/ionic/`).

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

### 2026-05-29: V5.5 detour decided — fix the Formulation-A reaction Cm bug in an independent fork
Revisited the chi/Cm audit. Derived *why* chi is safe but Cm is not: normalizing the parent PDE by chi·Cm cancels chi out of the reaction term entirely (chi·I_ion / (chi·Cm) = I_ion/Cm), so chi lives only in D (one half → can't break splitting). Cm appears in BOTH halves (D = sigma/(chi·Cm) AND the reaction /Cm), so Formulation A — which handles Cm in the diffusion mass term but drops it in the reaction — is silently wrong for any Cm != 1.0. NOT a "change-Cm-midway" hazard; it's wrong at t=0 for any Cm != 1. Safe today only because the project pins Cm=1.0 (ionic models output pA/pF).

Decision: create **Monodomain Engine_V5.5** as a full independent copy of V5.4 (rationale: backup — V5.4 stays the frozen validated baseline; don't risk its 77 tests). The ONLY change is making the reaction Cm-correct. Diffusion is NOT touched (its chi·Cm mass term already handles arbitrary Cm — verified: dividing the implicit theta-solve by chi·Cm yields effective D = sigma/(chi·Cm) for any Cm). This is a reaction-only minimal fix, NOT a full Formulation-B structural conversion.

Code facts located (V5.4):
- Bug sites (operator-split ionic steppers, both miss /Cm):
  - `cardiac_sim/simulation/classical/solver/ionic_time_stepping/rush_larsen.py:83` → `state.V = V + dt * (-(Iion + Istim))`
  - `cardiac_sim/simulation/classical/solver/ionic_time_stepping/forward_euler.py:64` → same
  - Fix: `... / Cm`, dividing by the TISSUE Cm.
- Plumbing gap: `SimulationState` (`state.py`) has no Cm field; `_build_ionic_solver(name, ionic_model)` (`monodomain.py:91`) doesn't pass Cm. Need to wire tissue Cm → ionic step. Preferred: add `Cm` to SimulationState and read it in the stepper, mirroring Bidomain V1's `Cm = getattr(state, 'Cm', 1.0)`.
- TWO-Cm hazard: tissue/cable Cm (`tissue/isotropic.py:23`, the `chi·Cm` in `fft.py` denominators) is the one to divide by. The ionic models' internal `Cm` (`ttp06/model.py:548 p.Cm=0.185`, paci/ord calcium `inv_VcF`) is a fixed per-cell constant for Ca/Na concentration flux — DO NOT touch or conflate.
- Aside: ORd standalone `model.py:802` already does `/p.Cm`, and TTP06 `model.py:297 dV=-I_ion` does not — but neither standalone path is used by the classical operator-split solver, so the steppers are the real fix sites.

Test protocol (settled intent, blueprint to formalize):
- **Regression (Cm=1):** V5.5 reproduces V5.4 bit-identically across the existing 77-test suite (the copy must not change Cm=1 behavior). atol ~1e-12.
- **Cm-scaling invariant (Cm=k):** scaling tissue Cm by k is equivalent to time-dilation by k — solution V(x, t; Cm=k) == V(x, t/k; Cm=1); observable as CV → CV/k and APD → k·APD, identical spatial structure. Requires BOTH halves to scale with Cm, so it fails on V5.4 (broken reaction) and passes on V5.5. This is the discriminating test.
- **0D single-cell version:** no diffusion; verify dV/dt = -(I_ion)/Cm directly (trajectory at Cm=k is the Cm=1 trajectory slowed by k).

**Bidomain V1 as independent oracle (added 2026-05-29):** Bidomain V1 `rush_larsen.py:81-84` is ALREADY the exact target form — `Cm=getattr(state,'Cm',1.0); state.V = V + dt*(-(Iion+Istim)/Cm)` (Formulation B). Use it two ways: (1) code-parity anchor for V5.5's fixed line; (2) cross-engine dilation oracle — run a matched cable in bidomain (reuse `cv_shared.py` `run_bidomain`, `measure_cv_from_history`) at Cm=1 and Cm=k, confirm CV ratio→1/k, assert V5.5's ratio matches. Bidomain shares NO solver code with monodomain → strong independent check. Process-isolated (bidomain also uses `cardiac_sim`): generate a `bidomain_cm_ref.json` in a separate process, load it in the V5.5 test.

**V5.4 internal LBM is dead code — DROP it in V5.5 (decided 2026-05-29):** Verified across the repo: `cardiac_sim/simulation/lbm/` has ZERO importers outside itself, ZERO standing tests (PROGRESS "Phase 5 LBM DONE" was historical; no `test_phase5.py` ships; the live suite test_phase7/8 + test_boundary_modes + tissue tests never touch it), ZERO experiments. `simulation/__init__.py` only names it in a docstring. `step_with_V` (ionic base) is the LBM path's only hook and is called solely from `lbm/monodomain.py:237`. The boundary-conduction research — the only active LBM work — runs on the SEPARATE `LBM/Engine_V1` engine (`diag_lbm_specular.py` → `sys.path.insert(LBM/Engine_V1)`, `from src.simulation import LBMSimulation`); all new BCs (same-cell specular, HBB, bounce-back, 27-rule enumeration) live in LBM V1's `src/`. So instead of guarding V5.5's LBM at Cm!=1, just DELETE the `lbm/` package + dead `step_with_V`. V5.5 becomes clean classical-only + Cm-correct; the chi/Cm source-term entanglement (the reason a guard was considered) vanishes with it. V5.4 keeps its LBM as the faithful backup. (Earlier plan had a fail-loud guard phase; superseded.) NOTE: `test_boundary_modes.py` is an FDM boundary-mode test, NOT an LBM test — don't use it to "verify LBM".

**Formulation A vs B D-input asymmetry (CRITICAL for the comparison):** Monodomain V5.4/V5.5 (Form. A) takes input `D` = sigma; engine forms physical diffusivity = D/(chi·Cm) internally. So "scale Cm" = hold D fixed, change Cm (diffusion dilates automatically). Bidomain V1 (Form. B) takes input D_i/D_e = already-scaled sigma/(chi·Cm); so the SAME experiment requires rescaling D_i,D_e→/k when Cm→k·Cm. Compare dimensionless ratios (CV→1/k), not absolute CV, so engines needn't match in absolute units.

Open: exact plumbing (state.Cm field vs constructor arg) — recommend state.Cm. Whether V5.5 keeps the `cardiac_sim` package name (collides with V5.4 if both imported, like the existing Bidomain/V5.4 collision) — for standalone test runs it's fine; note for the eventual cardiac_core consolidation.

## Failed Approaches
- **Flat engine-centric structure** (2026-03-16) — failed because: engines serve multiple research questions, making it impossible to find all work related to a single question. No natural place for cross-engine experiments.
- **First proposed restructure** (2026-03-16) — failed because: user wanted different groupings; initial Pipelines/Research separation didn't match actual workflow.
- **Converting V5.4 to Formulation B** (2026-03-16) — rejected because: would risk 77 passing tests for zero practical benefit since Cm is always 1.0. Both formulations are mathematically correct.
- **Merging solver internals into cardiac_core** (2026-03-16) — rejected because: solvers are engine-specific (decoupled GS for bidomain, CN/BDF for monodomain, BGK/MRT for LBM). Only shared code (ionic, mesh, stimulus) should be unified.
- **sys.modules hack as permanent solution** (2026-03-17) — recognized as temporary: `_prepare_engine()` flushes modules because both engines use `cardiac_sim` namespace. Acceptable for Phase 0 wrapper but must be eliminated when shared code moves into `cardiac_core/`.

## Session Log
