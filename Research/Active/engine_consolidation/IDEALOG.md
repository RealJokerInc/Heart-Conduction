# Engine Consolidation — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan this in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
**Designing the unified vocabulary + API toward the north-star** (conversational simulation builder for non-coders; see README North-Star Goals). Build order: **(1) ubiquitous language** — one canonical name per concept across the 3 engines (the ionic ABC proves the pattern) — **then (2) the unified API** (a `Simulation` interface/Protocol + idioms, written in that vocabulary), which the declarative `SimulationSpec` and the LLM wrapper then sit on. The 3-engine capability census is done (KNOWLEDGE "Cross-engine capability census"); the glossary is the immediate next artifact.

DONE this session (committed to `main`): V5.5 Cm-correct fork (Phases 0–2); cardiac_core drift reconciled; consolidation Phase 1 COPY-ONLY (canonical `cardiac_core/ionic/` + editable install + lazy `__init__`). The original code-dedup consolidation (engines import from cardiac_core, delete copies) is the FOUNDATION but is DEFERRED — big-bang deletion breaks Surrogate/Optimizer (audit); do it per-consumer later. V5.5 is the canonical monodomain; V5.4 frozen.

## Next Step
**cardiac_core drift RECONCILED (2026-05-30):** the post-Phase-0 additions (`run.py`/`analysis.py`/`geometry.py`/`io.py`) are a benign wrapper-level convenience layer (77 tests now, not 34); no shared-code packages yet, so Phase 1 is unblocked. `Engines/` symlink index fixed (cardiac_core un-broken; lbm_v1 → real `LBM/Engine_V1`; monodomain_v5.5 added). See KNOWLEDGE "cardiac_core drift reconciled".

**Phase 1 (copy) DONE (2026-05-31):** `cardiac_core/ionic/` is the canonical superset copy (from V5.5; latent LUT keyword `cell_type_is_endo`→`celltype_is_endo` fixed); `cardiac_core/__init__` made lazy (PEP 562 — `import cardiac_core.ionic` is engine-free, no `_prepare_engine`); `pyproject.toml` + `pip install -e .` make cardiac_core a real importable package (cwd-independent, scoped to `cardiac_core*` — does NOT expose Builder/cardiac_ml/engines). 77 cardiac_core tests green; V5.5 golden still exact (engines untouched).

**Scope pivot (post-audit, 2026-05-30):** the engine rewire+delete was DROPPED to copy-only after the audit found big-bang deletion breaks engine tests/examples AND active cross-project consumers (`Surrogate/surrogate/data/*_generator.py`, `Optimizer/V1/tuner/tissue_runner_bidomain.py` import `cardiac_sim.ionic` via the Bidomain path). User: "don't delete the originals — just copy them over."

**Next Step:** the DEFERRED migration (PLAN.md "Deferred" section) — when resumed, migrate consumers REPO-WIDE (engines' tests/examples + Surrogate datagen + Optimizer + `cv_shared` bare `from ionic`) to `cardiac_core.ionic`, per-consumer with test gates, never deleting out from under a live consumer; exclude V5.3/V5.4/_archive/torchcor from any survivor check. cardiac_core is now editable-installed (engines/consumers gain `import cardiac_core` for free once rewired).

## Thread

### 2026-05-31: Session vision — unified API + LLM wrapper → conversational simulation builder
North star (now the question's main goal, see README). A non-coder converses with Claude to build cardiac sims and learn how conduction works. Two goals:
1. **Unified construction API (Goal 1)** — one standardized, engine-agnostic, easy-to-construct way to declare + run: a declarative, validated, serializable **SimulationSpec** → run → **SimulationResult** → analysis. Consolidates today's split config (`CardiacMeshData` fields + `simulate()` call-args) into ONE object. Three field tiers: **required** (LLM asks) / **defaulted** (silent good values) / **derived** (computed).
2. **Self-contained LLM wrapper (Goal 2)** — Claude skills + reference docs driving Goal 1 under a strict protocol (gather → validate → construct → run → verify → present).

Key design insights (settled-ish; revisit when building Goal 2):
- **Spec schema = the intake questionnaire.** Make spec fields self-describing (`{required?, prompt, options, default}`); the LLM "gather" step = ask the prompt of each unfilled required field. The questionnaire can't drift from what engines need (same schema). THE cross-goal leverage point.
- **Pacing abstraction.** High-level protocol (`single` / `s1s2` / `regular(bcl, n_beats)`) that EXPANDS into the low-level stimulus list engines consume. Non-coder speaks in beats, not timestamps.
- **Outputs drive the run.** What the user wants to MEASURE (CV / APD / LAT / reentry) feeds back into numerics/run (`save_every`, `t_end`), not just post-hoc analysis.
- **Engine = explicit in spec, but LLM-inferred from the scientific question** + records rationale (auditable, overridable). e.g. bath/boundary effects → bidomain; fast/simple → monodomain.
- **Defaults philosophy.** A minimal spec ("pace this sheet, measure CV") must RUN via physiological defaults (TTP06/EPI, dt=0.02, strang, CN/pcg, chi=1400, Cm=1) — "one obvious way".

Deferred: user geometry input (Fiji drawing → Builder image→mesh; a designated drawings inbox; the Fiji-export→mask format contract). Assume geometry is provided for now.

**Decided focus (2026-05-31):** build the foundational **API** FIRST — everything (the spec questionnaire, the LLM wrapper) is contingent on a clean, standardized construct + run + results surface in `cardiac_core`.

### 2026-05-30: Phase 1 scoped — ionic is ~unified across engines; full direct migration decided
Verified the plan's "ionic identical across engines" assumption before executing. Findings:
- **Classical engines (V5.5 ↔ Bidomain V1): shared model files byte-identical** — `base.py`, `lut.py`, `ttp06`, `ord`, `mhas13`, `phas13`. V5.5 only adds `paci` + `__init__` exports.
- **LBM V1 ionic is NOT a fork** (initial diff misled): same `IonicModel` ABC (base.py byte-identical), same `ttp06/` structure (calcium/celltypes/currents/gating/model/parameters), `ord/model.py` byte-identical. Differs only by: (1) one keyword rename `cell_type_is_endo` (V5.5) vs `celltype_is_endo` (LBM) in a lut call; (2) top-level `ionic/` namespace vs `cardiac_sim.ionic`; (3) model subset (ttp06+ord only); (4) a dead stray `LBM/Engine_V1/ionic/ionic/` (imported by nothing).
- **Rewire surface:** ~23 consumer sites (14 V5.5, 9 Bidomain V1) + LBM's, all relative imports of varying depth (`from ...ionic`, `from .....ionic.base`).

DECISIONS (2026-05-30):
- **Scope: all three engines** (divergence is trivial, not a fork). cardiac_core/ionic/ = canonical SUPERSET (include paci/mhas13/phas13; reconcile the one keyword rename to a single canonical name).
- **Strategy: Option B — direct rewire, EXACT migration. NOT shims, NOT a sys.path trick.** User directive: "exact migration of all engines, not as path." Engine-local ionic copies DELETED; all consumers import `cardiac_core.ionic.*` absolutely; cardiac_core made a properly importable package (editable install / real package), not per-engine `sys.path.insert`.
- **Sequencing: 1a** build canonical `cardiac_core/ionic/` superset → **1b** rewire+delete classical engines (V5.5 + Bidomain V1), run their suites → **1c** rewire+delete LBM V1 (handle top-level namespace, keyword, drop stray ionic/ionic), run its suite. Verify cardiac_core 77 + every engine suite green.
- Rejected: re-export shim (A) and A-then-B staging — user wants the clean end-state directly.

Baseline before Phase 1: cardiac_core 77/77 pass; classical/LBM suites green (this session).

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

### 2026-05-30: Phase 2 physics correction — the Cm time-dilation invariant is FALSE
While executing Phase 2, the 0D test empirically failed: APD90(Cm=2)/APD90(Cm=1) = 1.34, not the predicted 2.0. Root cause: the tissue Cm divides ONLY the voltage update (`dV = -(Iion+Istim)/Cm`). The gate kinetics (`tau` from `compute_gate_time_constants(V,S)`) and the concentration rates carry NO Cm — they are intrinsic membrane kinetics. So scaling Cm→k·Cm does NOT rescale the whole system in time: V slows but gates keep their kinetics, so the AP MORPHOLOGY changes; it is not a `t→t/k` stretch. Substitution proof: for `W(t)=V(t/k)` to satisfy the Cm=k system you'd need `k·tau = tau` ⇒ only k=1. The invariant (and `CV→CV/k`, `APD→k·APD`) is wrong. Both `/reason` and BOTH audit passes missed this (the audit even "verified the physics sound"); the empirical run caught it. The FIX itself is correct — `dV=-(Iion+Istim)/Cm` is the right cable equation; only the *validation strategy* was flawed.

Corrected, rigorous validation (passing): (1) **exact one-step scaling** — from an identical state, `dV·Cm` is invariant across Cm∈{0.5,1,2,4} to 3.55e-15 (machine precision); this directly proves the reaction divides by Cm exactly, independent of morphology. (2) **direction** — larger Cm slows the upstroke (peak dV/dt 368→211 mV/ms) and changes APD (218→292 ms, NOT 2×). Together with the Cm=1 golden (max|dV|=0) and the full existing suite, the fix is rigorously validated. Test file: `Engine_V5.5/test_phase10_cm_scaling.py`.

Step 2.3 RESOLVED (user chose the proper cross-engine check): implemented absolute CV agreement vs Bidomain V1 (independent Formulation-B engine; isotropic + insulated BC reduces to monodomain with D_eff in the bulk). Reference generated by `Engine_V5.5/_regression/bidomain_cm_ref.py` (runs in the Bidomain engine, separate process — both use `cardiac_sim`). Matched physical diffusivity: bidomain D_i,D_e -> /Cm; V5.5 holds input D=D_EFF fixed with chi=1 (so D_phys=D_EFF/Cm) — both give D_eff=D_EFF/Cm. RESULT: Cm=1 V5.5 54.35 vs bidomain 54.35 cm/s (0.0%, exact threshold-grid match; also reproduces the historical 54.3 benchmark); Cm=2 V5.5 28.09 vs bidomain 27.77 cm/s (1.1%). Both << 5% tol. Phase 2 PASSES.

Refinement on CV vs APD scaling: empirically CV(Cm=2)/CV(Cm=1) ≈ 0.51 in BOTH engines — i.e. CV ~ 1/Cm. This is eikonal scaling, NOT dilation: CV ∝ sqrt(D_phys · upstroke_rate), and both D_phys ∝ 1/Cm and the upstroke rate (dV/dt = -Iion/Cm) ∝ 1/Cm, so CV ∝ 1/Cm. APD does NOT scale (set by repolarization gate kinetics, no Cm — measured 218→292 ms, not 2x). So the original plan's "CV→CV/k" was approximately right for the wrong reason; "APD→k·APD" was simply wrong. The cross-engine test does not depend on either — it compares two correct engines' absolute CVs.

NOTE: `cv_shared.run_monodomain_fdm` is NOT Cm-aware (line 303 has no /Cm, takes no Cm arg) — it cannot serve as a Cm!=1 reference. Only Bidomain V1 (run_bidomain) is a confirmed Cm-correct independent engine. cv_shared SIGMA_I=1.74, SIGMA_E=6.25, chi=1400 -> D_EFF=0.000972 (the test reads D_EFF_input from the ref JSON to avoid hardcoding drift).

## Failed Approaches
- **Flat engine-centric structure** (2026-03-16) — failed because: engines serve multiple research questions, making it impossible to find all work related to a single question. No natural place for cross-engine experiments.
- **First proposed restructure** (2026-03-16) — failed because: user wanted different groupings; initial Pipelines/Research separation didn't match actual workflow.
- **Converting V5.4 to Formulation B IN PLACE** (2026-03-16) — rejected: would risk V5.4's 77 passing tests. RESOLUTION (2026-05-30): instead FORKED V5.5 with the Formulation-B reaction; V5.4 stays frozen. (So Formulation B was the right target — just not destructively on V5.4.)
- **Cm time-dilation invariant for validation** (2026-05-30) — FALSE. Assumed `V(x,t;Cm=k)==V(x,t/k;Cm=1)` (⇒ CV→CV/k, APD→k·APD). Tissue Cm divides only the voltage update; gate kinetics/concentration rates carry no Cm, so Cm changes AP morphology, not timescale (APD 218→292 ms at k=2, not 2×). Asserted by the plan AND both audit passes; caught empirically (0D APD ratio 1.34). Replaced with exact 1/Cm one-step scaling (machine precision) + Bidomain V1 absolute-CV cross-check. The fix was always correct; only this validation premise was wrong.
- **`cv_shared.run_monodomain_fdm` as a Cm≠1 reference** (2026-05-30) — won't work: it has no `/Cm` and takes no Cm arg (hardcoded Cm=1). Used Bidomain V1 (`run_bidomain`) instead.
- **Merging solver internals into cardiac_core** (2026-03-16) — rejected because: solvers are engine-specific (decoupled GS for bidomain, CN/BDF for monodomain, BGK/MRT for LBM). Only shared code (ionic, mesh, stimulus) should be unified.
- **sys.modules hack as permanent solution** (2026-03-17) — recognized as temporary: `_prepare_engine()` flushes modules because both engines use `cardiac_sim` namespace. Acceptable for Phase 0 wrapper but must be eliminated when shared code moves into `cardiac_core/`.

## Session Log

### 2026-06-01 Session (handoff)
**Worked on**: Finished the V5.5 detour + consolidation Phase 1, then pivoted to the north-star (conversational simulation builder) and began designing the unified vocabulary/API — including a full 3-engine capability census.
**Accomplished**:
- **V5.5 Cm-correct fork** — Phases 0–2 done + committed (`ac30af55`→`5171bbce`); exact 1/Cm scaling (3.55e-15), Bidomain cross-check (0.0%/1.1%), Cm=1 golden exact. (Earlier this session.)
- **cardiac_core drift reconciled** + committed (`8f032687`); Engines/ symlink index fixed.
- **Consolidation Phase 1 — COPY-ONLY** + committed (`1f6c72e`): canonical `cardiac_core/ionic/` superset (keyword fix `cell_type_is_endo`→`celltype_is_endo`), lazy `__init__` (engine-free `import cardiac_core.ionic`), `pyproject.toml` + `pip install -e .`. Engine rewire + downstream (Surrogate/Optimizer) migration DEFERRED — audit found big-bang deletion breaks repo-wide consumers. README Phase-1 marked PARTIAL.
- **North-star set** (now the question's main goal in README): Goal 1 unified construction API + Goal 2 self-contained LLM wrapper (skills+docs, strict protocol) → conversational builder for non-coders. Key insight: **spec schema = the intake questionnaire**. Build order REFRAMED: **vocabulary first** (a ubiquitous language across the 3 engines), **then** the unified API (interface/Protocol + idioms).
- **3-engine capability census** run (read-only Explore agents) + synthesized into the cross-engine comparison (see KNOWLEDGE "Cross-engine capability census"). Found: ionic ABC + physical conventions + stimulus `+=` already aligned; divergence concentrated in construction, voltage naming (V/Vm), state, and the run/result contract; LBM is the consistent outlier.
**Next (resume cold)**: build the **glossary** off the census — settle the highest-leverage divergences first: (1) voltage `V` vs `Vm` [lean `Vm`], (2) the `State` concept (dataclass vs LBM on-object), (3) the `run()`/result contract (generator vs `(times,V_history)` + flat-vs-grid output). Then the rest of the universal-tier vocabulary, then Goal 1's interface/idioms. Geometry input (Fiji→Builder) and the Optimizer downstream migration both remain DEFERRED.

### 2026-05-30 Session
**Worked on**: Reasoned through the chi/Cm audit (why chi is safe but Cm is the troublemaker — Cm couples to both operator-split halves, chi to only diffusion); decided the V5.5 detour; blueprinted it; ran two adversarial audit passes (11 + 4 findings, all applied); executed Phases 0–2.
**Accomplished**:
- **Engine_V5.5** forked from V5.4 (Phase 0): faithful clone, dead internal LBM path removed (zero importers; boundary work uses LBM/Engine_V1), Cm=1 regression golden captured (`_regression/`, max|dV|=0).
- **Cm fix** (Phase 1): `SimulationState.Cm` plumbed from `spatial.Cm` (fail-loud, no getattr fallback); reaction divides by Cm in rush_larsen + forward_euler; FEM `_Cm`/`_chi` storage added (audit-CRITICAL — FEM only baked them into `self.M`). Cm=1 stays bit-identical; FDM/FEM/FVM all expose `.Cm`.
- **Validation** (Phase 2): `test_phase10_cm_scaling.py` 3/3 — exact 1/Cm reaction scaling to 3.55e-15; Cm-direction; Bidomain V1 cross-check (CV 54.35 vs 54.35 cm/s @Cm=1, 28.09 vs 27.77 @Cm=2).
- **Physics correction**: the Cm time-dilation invariant (assumed by the plan AND both audits) is FALSE — gate kinetics/concentrations carry no Cm, so Cm changes AP morphology, not timescale (APD 218→292 ms, not 2×). CV~1/Cm holds by eikonal scaling, not dilation. Caught empirically by the 0D test. The fix was correct throughout; only the validation strategy was wrong.
- 4 commits on `main` (`ac30af55`→`5171bbce`) + plan archived; README/KNOWLEDGE updated.
**Next**: Consolidation Phase 1 — move ionic models into `cardiac_core/ionic/` (build against V5.5). First reconcile the live `cardiac_core/` drift (added geometry/io/analysis/run; `Engines/lbm_v1` symlink deleted).
