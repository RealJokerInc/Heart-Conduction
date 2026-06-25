# Engine Consolidation

## Question
How do we unify the engines (Bidomain V1, Monodomain V5.5, LBM V1) under one `cardiac_core/` package, expose a **standardized, easy-to-construct** simulation API, and wrap it in a **self-contained LLM layer** (skills + reference docs, strict protocol) so that someone with no coding background can build cardiac simulations conversationally and learn how conduction works?

## North-Star Goals
1. **Unified construction API** — one standardized, engine-agnostic, easy-to-construct way to declare + run a simulation (geometry + stimulus + ionic + solver → run → results); sensible defaults, validation, one obvious way.
2. **Self-contained LLM wrapper** — a bundle of Claude skills + reference documents driving the API under a **strict protocol** (interpret → manifest → confirm → generate → run → verify), with guardrails. (REFRAMED 2026-06-25, see KNOWLEDGE/IDEALOG.)

**End goal (reframed):** a **script-generating skill suite for wet-lab scientists** (cell-culture / tissue-chip, no computational-sim background) — a *transition* tool that lowers the coding barrier by generating runnable `cardiac_core` scripts behind a **manifest + double-check accountability gate**, NOT a hand-holding conversational wizard for non-coders. The unified construction API (Goal 1) is the foundation it generates against (via `cardiac_core/API_CHEATSHEET.md`).

## Status: Active

> **Goal-1 unified construction API SHIPPED (2026-06-24)** — `ConductivityConfig`, `Grid`, `Simulation` Protocol, declarative factories, eager/batch `run()` → `SimulationResult`. See KNOWLEDGE "Goal-1 Construction API — SHIPPED".
>
> **Consolidation SHIPPED (2026-06-25)** — `cardiac_core` is now a **single self-contained package**: the 3 engines are vendored under `_monodomain`/`_bidomain`/`_lbm` + shared `ionic`/`mesh`/`stimulus`, the `_prepare_engine()` hack is deleted, no cross-folder imports. 137 tests green; bit-identical integrity goldens; engine originals untouched (frozen; `cardiac_core` is the centralized home). See KNOWLEDGE "cardiac_core unified ground-up package — SHIPPED" + `plans/2026-06-25_*`. This satisfies the spirit of the consolidation-track criteria below (single home, no duplicated-yet-referenced shared code) by **copy-vendoring** rather than rewire-and-delete.

## Why It Matters
Three engines share ionic models (TTP06, ORd) as file-level copies. Any bug fix or new model must be propagated manually to all three. The Optimizer and Surrogate need to call different engines with the same interface. Two different chi/Cm formulations coexist.

## Engines
- **All three engines** are the subject of this research
- **Monodomain V5.5** (2026-05-30) — Cm-correct fork of V5.4: the canonical monodomain to consolidate going forward. V5.4 stays frozen as the validated baseline.
- Target: `Engines/cardiac_core/` shared package

## Completion Criteria
- [x] Chi/Cm audit across all engines (March 2026)
- [x] Diffusion tensor encoding comparison (FDM 5pt/9pt/Mehrstellen, LBM D2Q5/D2Q9)
- [x] Unified API design (ConductivityConfig, Simulation protocol, create_simulation factory)
- [x] Phase 0: API layer + file format (wrapper over engines, 34 tests)
- [x] **Prerequisite — Monodomain V5.5 fork (2026-05-30):** fixes the Formulation-A reaction Cm bug so `Cm != 1.0` is physically correct; drops the dead internal LBM path. Validated: Cm=1 bit-identical to V5.4 (golden, max|dV|=0), exact 1/Cm reaction scaling (3.55e-15), cross-validated vs Bidomain V1 (CV agree 0.0% / 1.1% at Cm=1/2). 4 commits on `main` (`ac30af55`→`5171bbce`).
- [~] Phase 1: ionic models in `cardiac_core/ionic/` — **canonical copy DONE (2026-05-31, copy-only)**; cardiac_core editable-installed + lazy `__init__`. Engine rewire/delete + downstream-consumer (Surrogate/Optimizer) migration DEFERRED (audit found deletion breaks repo-wide consumers). Duplication knowingly retained for now.
- [ ] Phase 2: mesh + stimulus in `cardiac_core/mesh/`, `cardiac_core/stimulus/`
- [ ] Phase 3: ConductivityConfig (sigma → D in one place)
- [ ] Phase 4: engines rewired, their copies deleted
- [ ] Phase 5: `_prepare_engine()` hack removed, clean namespace
- [ ] All 149+ tests pass, zero duplicated shared code

## Experiments

| Experiment | Engine | Result | Location |
|-----------|--------|--------|----------|

## Literature
| Paper | Summary | Key Insight |
|-------|---------|-------------|

## Engine References

Files to read when resuming work on this question:

| File | What it tells you |
|------|-------------------|
| `GLOSSARY.md` | **Ubiquitous language** (Goal 1 vocab): one canonical name per concept across the 3 engines + `cardiac_core`; decision table (resolved/open), naming principles P1/P2 |
| `API_DESIGN.md` | **Unified `Simulation` interface** (Goal 1): 4 idioms, `Simulation` Protocol, factories, `ConductivityConfig` (chi/Form-A/B firewall + verified gate), `SimulationResult`, `SimulationSpec`/`create_simulation` (Goal-2 bridge), FEM-ditch (confirmed), Form-B convergence |
| `API_REFERENCE.md` | **Library-style API reference**: every public class + function (Grid, ConductivityConfig, IonicModel, Stimulus, factories, Simulation, SimulationResult, SimulationSpec) with signatures, params, returns, examples + impl-status legend |
| `Monodomain/Engine_V5.5/_probe_conductivity_firewall.py` | Firewall build-time gate: raw sigma → `for_monodomain()` → live V5.5 cable; arithmetic to 1.1e-19, CV 54.35/28.09 (matches bidomain ref). Keep-or-toss; permanent test lands in `cardiac_core/tests` at Phase 3 |
| `cardiac_core/__init__.py` | Package exports: api (monodomain/bidomain/lbm), file_format, run, analysis, geometry, io |
| `cardiac_core/file_format.py` | CardiacMeshData dataclass, save/load/create functions, .npz format v1 |
| `cardiac_core/api.py` | Simplified API: monodomain(), bidomain(), lbm() + CardiacSimulation wrapper |
| `cardiac_core/run.py` | One-shot `run_monodomain/run_bidomain/run_lbm`, `simulate`, `SimulationResult` (added post-Phase-0) |
| `cardiac_core/analysis.py` | Pure tensor analysis: activation_time, conduction_velocity, apd_map, phase_singularities, restitution_curve |
| `cardiac_core/geometry.py` | Mask/region/distance/fiber helpers |
| `cardiac_core/io.py` | Result .npz save/load |
| `cardiac_core/tests/` | 77 tests: file format, per-engine, integration, run/analysis/geometry/io, direct-match verification |
| `Bidomain/Engine_V1/REVIEW.md` §6 | Full merger proposal: chi/Cm audit, diffusion tensor encoding, unified API design |
| `Optimizer/improvement.md` | Engine adapter design for Optimizer V2 |
| `Bidomain/Engine_V1/cardiac_sim/ionic/base.py` | IonicModel ABC (identical across engines) |
| `Monodomain/Engine_V5.4/cardiac_sim/simulation/classical/discretization_scheme/fdm.py` | V5.4 FDM (Formulation A, chi·Cm in mass term) |
| `Monodomain/Engine_V5.5/` | Cm-correct monodomain fork of V5.4 (reaction `/Cm`; LBM path removed). Frozen-baseline V5.4 unchanged. |
| `Monodomain/Engine_V5.5/test_phase10_cm_scaling.py` | Cm-correctness validation: exact 1/Cm scaling, direction, Bidomain V1 cross-check |
| `Monodomain/Engine_V5.5/_regression/` | Cm=1 golden (`make_golden`/`check_golden`) + `bidomain_cm_ref.py` cross-engine reference |
| `Bidomain/Engine_V1/cardiac_sim/simulation/classical/discretization/fdm.py` | Bidomain FDM (Formulation B, face-based symmetric) |
| `LBM/Engine_V1/src/diffusion.py` | LBM sigma_to_D, tau_from_D (Formulation B) |
| `Monodomain/Engine_V5.4/cardiac_sim/simulation/lbm/monodomain.py` | V5.4 LBM source term (Formulation A, /(chi·Cm)) |
| `Monodomain/Engine_V5.4/cardiac_sim/tissue_builder/tissue/isotropic.py` | IsotropicTissue (D, chi, Cm separate) |
| `Bidomain/Engine_V1/cardiac_sim/tissue_builder/tissue/conductivity.py` | BidomainConductivity (D pre-scaled) |
| `Research/Knowledge/bidomain_simulation.md` | Solver and discretization knowledge |
| `Research/Knowledge/lbm_cardiac.md` | LBM knowledge |

## Future Work
{No deferred items yet.}
