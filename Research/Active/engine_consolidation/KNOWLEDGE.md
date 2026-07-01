# Engine Consolidation — Knowledge File

> This file is a running synthesis. Updated as findings accumulate.
> When the question is complete, a copy is promoted to `Research/Knowledge/`.

> **SHIPPED 2026-07-01 — foundation cleanup + boundary modes.** A cardiac_core+mcp adversarial audit
> (46 findings → [CARDIAC_CORE_AUDIT.md](./CARDIAC_CORE_AUDIT.md)) drove a 3-phase cleanup ([PLAN.md](./PLAN.md)):
> **P1** fixed the monodomain FDM anisotropic cross-derivative bug + unified the chi/D convention
> (`D_xx` is RAW everywhere; effective = `D/(χ·Cm)` in every engine; default `D=1.4`; blocked-default
> fixed) + ionic-override replay + MCP path-traversal; **P2** removed FEM/dead code + API footguns;
> **P3** productized the LBM flat-wall boundary modes as `cardiac_core.lbm(boundary=, alpha=)`
> (hbb / specular_neighbour / specular_samecell / combined-α). 195 cardiac_core+mcp tests green.

> **SHIPPED 2026-07-01 — API-consistency hardening + contract-matrix stress harness.** A 4-lens
> adversarial audit ([API_CONSISTENCY_AUDIT.md](./API_CONSISTENCY_AUDIT.md): 7 HIGH · 8 MED · 6 LOW)
> found the boundary gaps were a *class* — capability unexposed, one kwarg meaning different things per
> engine, a few silent-wrong-result bugs. The fix PLAN was **audited to convergence over 4 rounds**
> (R1 5blk/10maj → R2 1blk/5maj → R3 1blk/1maj → R4 1blk[mechanical] → CONVERGED), then executed in
> 6 phases (commits `1a65d3d`→`9702bb7` on `engine-tuner-cardiac-core`). The keystone is
> `tests/test_api_contract.py` — the contract matrix **written FIRST** (Phase 0) as 22 `{entry × engine
> × param × physics}` cells; unfixed cells were `xfail(strict=True)` so each landing fix XPASS-*forced*
> its in-phase flip to a live assert (the matrix can't rot into "guard-as-feature"). **217 passed / 2
> xfailed** (C2 oblique-LBM capability + C7 boundary-Dxy truncation, both documented-deferred / Audit
> #46); goldens bit-identical every phase. See "API-consistency hardening" section below.

## Current Understanding

The question has TWO layers (see "North-Star" below): the **foundation** is code-consolidation (unify the engines' shared code in `cardiac_core/`); the **goal on top** is a conversational simulation builder for non-coders (unified API + LLM wrapper). Build order is now vocabulary-first → unified API.

Three engines solve cardiac electrophysiology with 15+ duplicated files (ionic models, mesh, stimulus, solvers). They use two chi/Cm formulations, both correct but internally incompatible.

**Target architecture**: `cardiac_core/` is the unified codebase — it owns shared code (ionic models, mesh, stimulus, conductivity) and provides the public API. Engines import from `cardiac_core/`, not the other way around. Engine directories shrink to solver-specific code only.

```
cardiac_core/                  ← THE source of truth
  ionic/                       ← TTP06, ORd, PHAS13, MHAS13 (one copy)
  mesh/                        ← StructuredGrid, BoundarySpec (one copy)
  stimulus/                    ← StimulusProtocol (one copy)
  conductivity.py              ← ConductivityConfig (sigma → D, chi/Cm in one place)
  file_format.py               ← CardiacMeshData, save/load/create
  api.py                       ← monodomain(), bidomain(), lbm()

Bidomain/Engine_V1/            ← imports cardiac_core.ionic, cardiac_core.mesh
  solver/                      ← decoupled GS, 3-tier elliptic (unique to bidomain)
Monodomain/Engine_V5.4/        ← imports cardiac_core.ionic, cardiac_core.mesh
  solver/                      ← CN/BDF/RK diffusion solvers (unique to monodomain)
LBM/Engine_V1/                 ← imports cardiac_core.ionic
  collision/, streaming/       ← BGK/MRT, D2Q5/D2Q9 (unique to LBM)
```

**Current state**: Phase 0 complete (API wrapper imports FROM engines; `_prepare_engine()` flushes `sys.modules` for the shared `cardiac_sim` namespace — eliminated in Phase 1). **Prerequisite done (2026-05-30): Monodomain V5.5** — a Cm-correct fork of V5.4 (reaction `/Cm`, dead LBM path dropped) is now the canonical monodomain to consolidate; V5.4 frozen. See "V5.5 Cm-correct fork". The consolidation (Phases 1–5) has not started.

### Chi/Cm Formulations (audited March 2026)

**Formulation A** (Monodomain V5.4): chi·Cm appears in mass/time-derivative term. Ionic solver does NOT divide by Cm.
```
Operator: A = chi·Cm · I ± θ·dt · L
Ionic:    V += dt · (-(I_ion + I_stim))          # no Cm division
LBM src:  S = -(I_ion + I_stim) / (chi·Cm)
```

**Formulation B** (Bidomain V1, LBM V1): D already contains chi·Cm. Mass term is 1/dt. Ionic solver divides by Cm.
```
Operator: A = 1/dt · I ± θ · L
Ionic:    V += dt · (-(I_ion + I_stim) / Cm)     # divides by Cm
LBM src:  R = -(I_ion + I_stim) / Cm
```

Both produce identical results when Cm = 1.0. V5.4's formulation is fragile: changing Cm from 1.0 would silently break the ionic step because the chi·Cm factor is absorbed by the diffusion mass term, coupling the two halves of the operator split.

### Engine-by-engine chi/Cm audit

**Bidomain V1** (Formulation B):
- `BidomainConductivity`: D_i, D_e pre-scaled (chi/Cm absorbed)
- FDM constructor: warns if chi ≠ 1.0, mass term = 1/dt
- RushLarsen: `V += dt·(-(I_ion+I_stim)/Cm)` with `Cm = getattr(state, 'Cm', 1.0)`
- `fft.py`: DEPRECATED, still uses Formulation A (not called by any solver)

**Monodomain V5.4** (Formulation A):
- `IsotropicTissue`: stores D, chi=1400, Cm=1.0 as separate fields
- FDM/FEM/FVM: chi·Cm in mass terms (`chi·Cm·I`, `chi·Cm·∫φ_iφ_j`, `chi·Cm·Vol`)
- DCT/FFT solvers: chi·Cm in spectral denominators
- RushLarsen: `V += dt·(-(I_ion + I_stim))` — NO Cm division
- V5.4 LBM: source = `-(I_ion+I_stim) / (chi·Cm)`

**LBM V1** (Formulation B):
- `sigma_to_D()`: only place chi appears in entire engine
- `tau_from_D()`: D has chi/Cm baked in, tau = 0.5 + D·dt/(cs²·dx²)
- `compute_source_term()`: `R = -(I_ion + I_stim) / Cm` — only Cm, NOT chi

### Diffusion tensor encoding across methods

| Method | What it receives | Dxx ≠ Dyy | Dxy ≠ 0 |
|--------|-----------------|:---------:|:-------:|
| FDM 5-point | Dxx, Dyy per node (harmonic mean) | Yes | No |
| FDM 9-point | Dxx, Dxy, Dyy per node | Yes | Yes |
| FDM Mehrstellen | Scalar D only (dx=dy required) | No | No |
| FEM P1 | D tensor per element | Yes | Yes |
| FVM (TPFA) | D per face | Yes | No |
| LBM D2Q5 BGK | Scalar D → single tau | No | No |
| LBM D2Q5 MRT | Dxx, Dyy → separate tau_x, tau_y | Yes | No |
| LBM D2Q9 MRT | Dxx, Dyy, Dxy → tau_x, tau_y, tau_xy (shear moment) | Yes | Yes |

D2Q5 cannot encode Dxy because it has no diagonal velocities → no p_xy shear stress moment. D2Q9 adds 4 diagonal velocities enabling the p_xy moment.

### Code duplication inventory

| Module | Copies | Notes |
|--------|--------|-------|
| IonicModel ABC + lut.py | 3 | Identical across all engines |
| TTP06 (5 files) | 3 | Identical |
| ORd (6 files) | 3 | Identical |
| PCG solver | 2 | Minor divergence (breakdown check, warm start) |
| Splitting strategies | 2 | Identical logic, different state type hints |
| Stimulus protocol | 2 | Bidomain uses += (accumulate), V5.4 uses = (overwrite) |
| StructuredGrid | 2 | Bidomain adds boundary_spec |

## API-consistency hardening + contract-matrix harness — SHIPPED (2026-07-01)

Post the boundary-mode ship, the user caught two gaps the tests missed (`run_lbm` dropped `boundary`;
wall modes BGK-only). A 4-lens adversarial audit showed a *class* of API-surface fragility, cataloged
in `API_CONSISTENCY_AUDIT.md` (7 HIGH · 8 MED · 6 LOW). The **numerics were sound** (Cm≠1, build_kwargs
replay, mesh round-trip all correct); every fix was pure surface work.

**The keystone — `tests/test_api_contract.py` (written FIRST, Phase 0).** A `CONTRACT` list of 22
`Cell(entry, engine, param, physics, expected, run, match, status, exc)` rows covering the cross-product,
driven by one parametrized `test_contract`. `status` gates the marker: `to_fix` → `xfail(strict=True)`
(so the XPASS the instant a fix lands FAILS the suite → *forces* an in-phase flip to `landed`, no deferred
cleanup — this is what stops the matrix rotting into the "guard-as-feature" anti-pattern the post-mortem
indicted); `deferred` → `xfail(strict=False)`; `landed` → no marker (live assert). `exc` is the tuple's
last field (namedtuple defaults right-align). Final state: 20 `landed` + 2 `deferred`.

**Fixes (Phases 1–5, one commit each):**
| Finding | Fix | Commit |
|---|---|---|
| P1 | `run_lbm`/`simulate` forward `boundary`/`alpha` | `40cd2ca` |
| C1 | `lbm_step_d2q9_mrt_wall` + drop bgk-only guard (overlay is post-stream → collision-agnostic) | `1dda8f6` |
| S1,I1,I2,S2,S3,C4,I3 | bidomain boundary validation; masked-grid **union**(hole rim, outer rect edges) bounce wired to BOTH LBM branches; `_resolve_mesh` deepcopy (one choke point → factory+with_+reset immutability); alpha/sigma_ratio/lattice **warn**; dtype round-trip | `35327f5` |
| C3,C5,C6,S4 | shared `ionic/registry.py::build_ionic_model` (branches on ctor capability, ENDO default → goldens-safe; phas13/mhas13/paci now on all 3 engines); `weights_mode`/`stencil`/`boundary_mode`/`splitting` exposed; cross-engine knob misroute → validated ValueError (add-and-reject) | `9702bb7` |

**Deferred (documented xfail, not silent gaps):**
- **C2 oblique LBM** — a REAL numerics limitation, not a wiring gap: `mrt_collide_d2q9` discards `D_xy`
  (`p_xy_eq=0`; docstring cites Audit #46 — needs moment-space rotation of `s_jx/s_jy`). The audit's
  "MRT is oblique-CAPABLE" was half-true (the tau helpers compute `tau_xy`; the collision kernel drops
  it). The *raise* is shipped as a documented-limitation message; the *capability* cell is a permanent
  xfail. Two independent audit lenses caught this — it would otherwise have shipped a silent-wrong result.
- **C7** mono/bidomain boundary-Dxy truncation — dispositioned *paired to C2* (all three engines decline
  full oblique → no silent per-engine divergence).

**Trap avoided (an execution catch the audit predicted):** an `EPI` default on the shared ionic builder
would have flipped the bidomain+LBM goldens (whole codebase defaults ENDO); the round-2 audit flagged it
pre-execution → shipped with `ENDO`.

## North-Star: lab-facing simulation platform (BOTH goals SHIPPED — see the two "SHIPPED" sections above)

> **Status (2026-06-25): both north-star goals delivered.** Goal 1 = the unified construction API (shipped + the engines consolidated into one self-contained `cardiac_core`). Goal 2 = the LLM layer, **REFRAMED** from "non-coder conversational builder" to a **script-generating skill suite for wet-lab scientists** (cell-culture / tissue-chip) — it GENERATES runnable `cardiac_core` scripts behind a manifest + double-check gate, not a teaching wizard. The original two-layer vision below is retained for the design rationale it still informs (the deferred Layer-A `SimulationSpec`).

**Original vision (informs the deferred Layer-A `SimulationSpec`):** two layers —
1. **Unified construction API (Goal 1)** — one standardized, engine-agnostic, easy-to-construct way to declare + run: a declarative, validated, serializable **SimulationSpec** → run → **SimulationResult** → analysis. Three field tiers: required (LLM asks) / defaulted (silent good values) / derived (computed).
2. **LLM wrapper (Goal 2)** — Claude skills + reference docs driving Goal 1 under a strict protocol. *Shipped form:* the `/sim-*` skill suite (interpret → manifest → confirm → generate → run → verify), driving the factories directly; `SimulationSpec` deferred.

**Design insights (settled-ish):**
- **Spec schema = the intake questionnaire.** Make spec fields self-describing (`{required?, prompt, options, default}`); the LLM "gather" step = ask each unfilled required field. Questionnaire can't drift from engine needs. The cross-goal leverage point.
- **Pacing abstraction**: high-level `single`/`s1s2`/`regular(bcl,n_beats)` EXPANDS to the low-level stimulus list (the engines already have `add_s1s2_protocol`/`add_regular_pacing` — see census).
- **Outputs drive the run**: what to MEASURE (CV/APD/LAT/reentry) feeds back into `save_every`/`t_end`, not just post-processing.
- **Engine = explicit in spec, LLM-inferred from the scientific question** + recorded rationale.
- **Defaults**: a minimal spec ("pace this sheet, measure CV") must run (TTP06/EPI, dt=0.02, strang, CN/pcg, chi=1400, Cm=1).

**Build order (reframed 2026-05-31/06-01):** (1) **ubiquitous language** — one canonical name per concept across the 3 engines (the IonicModel ABC proves the pattern); (2) **unified API** — a `Simulation` interface/Protocol + idioms (declare/run/change/stimulate), written in that vocabulary; then the `SimulationSpec` and LLM wrapper sit on top. Vocabulary is the immediate next artifact (the glossary).

**Deferred:** user geometry input (Fiji drawing → Builder image→mesh; a designated drawings inbox; the export→mask contract). Assume geometry provided for now.

## Unified API (Goal 1) — `API_DESIGN.md` (2026-06-24)

The glossary's resolved vocabulary is now realized as a concrete interface in **`API_DESIGN.md`**. Summary:

- **Four idioms:** DECLARE (`monodomain()/bidomain()/lbm()` factories + `create_simulation(spec)`) · STIMULATE (`StimulusProtocol` / `sim.stimulate`) · RUN (`sim.run` → `SimulationResult`) · CHANGE (`sim.with_(**overrides)` → new Simulation).
- **`Simulation` Protocol:** engine-agnostic runtime interface (introspection + `Vm` grid `(Nx,Ny)` f64 + `run/step/reset/with_/stimulate`). Optimizer/Surrogate/LLM program against this; the engine is hidden behind the factory.
- **`ConductivityConfig` is the chi/Formulation-A/B firewall** (design source-verified 2026-06-24 against `fdm.py:195–238`, `BidomainConductivity`, LBM `sigma_to_D`). Stores physics (`sigma_i/sigma_e/sigma_eff, chi, Cm, fiber_angle`); emits **per-engine** inputs via `for_monodomain()/for_bidomain()/for_lbm()`. **Confirmed:** monodomain physical diffusivity = `D_input/(chi·Cm)` (Form A); bidomain/LBM take `D=σ/(chi·Cm)` pre-scaled (Form B); reaction divides by tissue `Cm` everywhere (V5.5). **Corrected mechanic (a Cm≠1 trap I first got wrong):** the real `Cm` must reach EVERY engine (reaction needs it); only the *diffusion input's* Cm-scaling differs — Form-A monodomain scales diffusion by Cm internally, so feed it Cm-**un**scaled `D=sigma_eff/chi` with engine `chi=1` + real `Cm`; Form-B gets fully-scaled `D` + real `Cm`. The earlier "feed `D_eff` with `chi=1, Cm=1` no-op" was wrong for Cm≠1 (breaks the reaction — same family as the false time-dilation invariant). At the pinned `Cm=1` all collapse to `D_eff=sigma_eff/chi`. **#13 (chi only here) is what makes #12 work.** Classmethods `.isotropic/.bidomain/.anisotropic`; `sigma` is raw conductivity (mS/cm), `D_eff` derived.
- **`SimulationSpec` + `create_simulation` = the Goal-2 bridge.** Self-describing fields in 3 tiers — **required** (LLM asks: engine, geometry, pacing, measure) / **defaulted** (silent: ionic, dt, Cm, solvers, chi) / **derived** (computed: save_every, t_end, D_eff). "Spec schema = the intake questionnaire" → can't drift from engine needs.

**New decisions (user, 2026-06-24):** stim amplitude **−52** (#9 resolved); CHANGE = **functional `.with_()`** (immutable, sweep-safe); construction = **factories + spec, layered**.

**CONFIRMED (user, 2026-06-24): ditch FEM → structured-grid is the ONLY standard** (P2→P2′). Drops the unstructured/flat-`(n_dof,)` geometry path, `TriangularMesh`, monodomain's `FEMDiscretization`. **FDM primary; FVM survives** (structured-grid-native — TPFA on the grid); a possible FVM→FDM collapse is a separate later question. Composes with the Form-A→B convergence in the Phase-4 rewire. See `API_DESIGN.md` §9.

**Firewall gate CLOSED (2026-06-24):** `Monodomain/Engine_V5.5/_probe_conductivity_firewall.py` drives raw `sigma_i=1.74, sigma_e=6.25, chi=1400` → `for_monodomain()` → live V5.5 cable. Arithmetic `D=0.0009721973895941` = reference `D_EFF` to 1.1e-19 (Cm-independent); CV(Cm=1)=54.35 (0.00% vs bidomain ref), CV(Cm=2)=28.09 vs 27.77 (1.15% < 5%). Cm≠1 firewall path numerically correct in the live engine.

## Goal-1 Construction API — SHIPPED in code (2026-06-24)

`API_DESIGN.md`/`API_REFERENCE.md` design symbols are now REAL `cardiac_core` code (PLAN.md "API-track" Phases 0–5, all green). **121 cardiac_core tests pass** (was 80; +41 new). The engines are unchanged — this is a construction/wrapper layer. Delivered:

- **Phase 0 — V5.4→V5.5 repoint.** `cardiac_core/api.py` adds `_V55_PATH`; the `monodomain()` factory + LBM ionic import now `_prepare_engine(_V55_PATH)` (was V5.4). So the shipped factory runs the **Cm-correct** reaction the firewall was measured against. Gated by a **behavioral** test (`test_monodomain.py::TestEngineIsV55::test_reaction_divides_by_cm`): one Rush-Larsen reaction step at Cm=2 gives exactly `dV(Cm=1)/2` (V5.4 would give equal dV). V5.4 untouched on disk.
- **Phase 1 — `cardiac_core/conductivity.py::ConductivityConfig`** (the χ/Cm + Form-A/B firewall). Frozen dataclass; `.isotropic/.bidomain/.anisotropic`; props `sigma_eff` (harmonic i/e collapse) + `D_eff`; emitters `for_monodomain` (Form A: `D=sigma_eff/chi, chi=1, real Cm`), `for_bidomain`/`for_lbm` (Form B: `σ/(χ·Cm)`). **Public property is `sigma_eff`** (per API_REFERENCE), isotropic value stored in field `sigma_iso` (avoids the field/property clash). Gate test `test_conductivity.py`: arithmetic to 1e-12 (Cm-independent `D`) + **live V5.5 cable CV** in a subprocess (`_live_cv_gate_driver.py`, isolates the `cardiac_sim` namespace) — CV rel 3.7e-15 @Cm=1, 1.15% @Cm=2.
- **Phase 2 — `cardiac_core/grid.py::Grid`** (structured-only). `Nx,Ny,dx,dy,mask`; `Lx/Ly/coordinates/n_dof`; lazy `_structured_grid()`. `coordinates` = `meshgrid(linspace,'ij')` matching the engine (`x[-1,0]==Lx`), verified by a flat-coords roundtrip.
- **Phase 3 — `Vm` canonical, `.V` read-only alias** on `SimulationSnapshot` + `SimulationResult` (kept all 80 prior tests green). `SimulationResult` gained `dx/dy/ionic_states`. **`io.save_result(path, times, Vm=None, phi_e=None, *, V=None, **md)`** — audit-HIGH fix: `phi_e` stays positional-or-keyword so the legacy `save_result(path, times, V, phi_e)` positional call doesn't break; legacy `V=` keyword warns.
- **Phase 4 — declarative factories.** `monodomain/bidomain/lbm(geometry, ionic_model, conductivity, stimulus, *, mesh=…, …)` with a positional-`CardiacMeshData`/`str` type-sniff preserving the legacy `mesh=` path. `_build_mesh_data` maps `ConductivityConfig`→`CardiacMeshData` per engine (mono: `for_monodomain`→`D_xx,chi=1,Cm`; bidomain: **RAW σ tuples as (Nx,Ny) fields** `(σ,σ,0)` — the FDM indexes `[i,j]`; lbm: `for_lbm` D_eff). `CardiacSimulation` now has `dt/Cm/ionic_model` introspection, **functional `with_(**overrides)`** (immutable), `reset()`, and `stimulate(region,…)` — all replay via a stored construction record (`_data` + `_build_kwargs`), unified across declarative + mesh paths (both route stimuli through `data.stimuli`, sidestepping LBM `start`/`start_time`). `cardiac_core/simulation.py::Simulation` is a `runtime_checkable` Protocol; `isinstance(sim, Simulation)` is True.
- **Phase 5 — `run()` is now EAGER.** `sim.run(t_end, save_every, *, batch=None, record=("Vm",), callback=None) -> SimulationResult | Iterator[SimulationResult]`. `batch=k` streams ≤k-frame chunks; the old generator is `sim.snapshots(...)` (back-compat). **~34 iterator call-sites migrated** `*.run(`→`*.snapshots(`, incl. the production `run.py::_collect`. `record=("ionic_states",)` records real `(T,n_states,Nx,Ny)` for classical engines, raises `NotImplementedError` for LBM (no silent None). `SimulationResult.cv()/.apd()/.lat()/.restitution()` delegate to `cardiac_core.analysis`.

**Still design-only (separate plans):** `SimulationSpec`/`create_simulation` (Goal-2 LLM intake), the consolidation-track engine rewire (Form-A→B convergence, delete `for_monodomain()`, FEM removal). The PLAN's "Phase 4" is API factory-wiring, **not** the consolidation-track Phase 4 in the migration table below.

## cardiac_core unified ground-up package — SHIPPED (2026-06-25)

`cardiac_core` is now a **single self-contained package**: the three engines are vendored IN (copied), the `_prepare_engine()` sys.modules hack is **deleted**, and no `cardiac_core/**` file imports `Monodomain/`/`Bidomain/`/`LBM/`. **137 tests green** (was 121; +16). Decision context: user chose **A2 (unified, flat, ground-up)** over relocate-keep-hack/rename-only, and **`cardiac_core` is the centralized home** (future engine dev happens here; the original engine folders freeze → no copy-drift). Backup before vendoring: tag `pre-consolidation-vendoring` + bundle `~/heart-conduction-PRE-CONSOLIDATION-2026-06-25.bundle`.

**Final layout:**
```
cardiac_core/
  api.py run.py conductivity.py grid.py simulation.py analysis.py geometry.py io.py file_format.py media.py
  ionic/  mesh/  stimulus/        ← shared, one copy each
  _monodomain/  _bidomain/  _lbm/  ← vendored solvers (underscore = private; don't shadow the factories)
  tests/  (incl. _integrity/ goldens, test_self_contained.py guard)
```

**Key implementation facts (for future maintenance):**
- **Underscore naming is load-bearing.** The solver packages MUST be `_monodomain`/`_bidomain`/`_lbm` — a non-underscore `cardiac_core/monodomain/` package SHADOWS the public `monodomain()` factory (`from cardiac_core import monodomain` returns the package, `'module' object is not callable`). Public surface unchanged: `monodomain()/bidomain()/lbm()` factories + `simulate(engine=...)`; users never import `_*`.
- **Engines are pure-relative-import internally** (verified: 0 absolute `cardiac_sim`/`src`, 0 sys.path/importlib/__file__). So vendoring = copy the subtree + rewrite ONLY the solver→shared cross-imports (mono 8, bidomain 9, LBM 0 — it receives the ionic model as an object). The cross-ref rewrite regex MUST be `\b`-anchored (`from \.+(ionic|tissue_builder)\b`) or it corrupts `ionic_time_stepping`/`ionic_stepping` internal imports (hit this bug; caught + fixed).
- **Shared reconciliations:** `mesh/structured.py` = bidomain superset (adds `boundary_spec` + `edge_masks`/`dirichlet_mask_phi_e`/`neumann_mask_phi_e`; shared methods byte-identical to mono). `stimulus/protocol.py` = bidomain's canonical `+=` accumulate (vs mono's `=`; differs only for OVERLAPPING stimuli — goldens use single stims so bit-identical). `_bidomain/tissue/` keeps `BidomainConductivity` (per-engine, not shared).
- **Integrity gate (`tests/test_integrity.py`):** per-engine pre-vendor GOLDEN (atol=0, captured Phase 0 against the originals) + source-tree hash. Proved every vendored engine is BEHAVIOR-IDENTICAL and the originals byte-unchanged. `tests/test_self_contained.py` is the durable guard (matches real import lines + `_prepare_engine(` calls, NOT prose).
- **One intentional exception:** `tests/_live_cv_gate_driver.py` still subprocess-drives the original V5.5 cable (firewall gate); excluded from the guard, documented in `cardiac_core/engines_SOURCE.md` (provenance + re-vendor recipe).
- **Per-phase commits** on `main`: Phase 0 `935160b` → Phase 5 `37dc381`.

## Goal-2 LLM layer — script-generating skill suite — SHIPPED (2026-06-25)

The "LLM wrapper" (north-star Goal 2), built for **wet-lab scientists** (cell-culture / tissue-chip, no computational-sim background) — a **transition tool that GENERATES runnable `cardiac_core` scripts**, NOT a conversational non-coder wizard (audience reframed; the README "non-coder conversational builder" wording was corrected). Drives the shipped `cardiac_core` API directly (Layer-A `SimulationSpec` deferred; programmatic claude-api later). **140 tests green** (+4 viz). Phases 1→5 committed `126ff25`→… on `main`.

**The suite (`.claude/skills/`):**
- **`/sim-experiment`** (keystone) — free-form description → INTERPRET (recipe + engine inference) → **MANIFEST** (plain text) → ⛔ **double-check gate** (never runs before the scientist confirms — the accountability principle, "no vibe-coding runoff") → generate `Lab/{date}_{slug}/{MANIFEST.md, run.py}` + append `Lab/NOTEBOOK.md` → offer run + verify. FIRST *bundled* skill in the repo (`SKILL.md` + `reference/{run-template.py, recipes.md, manifest-template.md}`).
- **`/sim-preset`** — save/list/load named YAML parameter sets (`Lab/presets/{name}.yaml`); applied at GENERATION time (inline into `run.py`, self-contained); a loaded preset still passes the gate.
- **`/sim-media`** — standardized visuals via `cardiac_core/viz.py` (`propagation_video` mp4, `apd_map_figure`, `activation_isochrones`); `bulk=True` → gitignored `media/lab/_sim_outputs/` (regenerable).
- **`/sim-notebook`** — `index|summary|compare` over `Lab/`; manifests are source of truth, `NOTEBOOK.md` generated.

**Key asset:** `cardiac_core/API_CHEATSHEET.md` — the maintained, canonical API reference the skills generate against (prevents hallucinated-API failures, the #1 LLM-sim-code failure mode). Canary: `Lab/_validate/smoke.py` (re-run after any API change). Co-located with the code so it can't drift.

**New code:** `cardiac_core/viz.py` (headless Agg, float64, lazy-exported, tested) + `cardiac_core/API_CHEATSHEET.md`. Everything else is markdown skills + `Lab/` scaffolding — additive.

**Demo seed:** `Lab/2026-06-25_cv-strip-{control,knockdown}` — control σ → CV 59.3, half-σ → CV 41.0 (eikonal √D); a real control/knockdown series the notebook compares.

## Goal-2 MCP server — `cardiac-core` — SHIPPED local (2026-06-26)

An **MCP (Model Context Protocol) server** that exposes the shipped `cardiac_core` API to *any* MCP host (Claude Desktop, Claude Code, IDEs) — the portability step the Claude-Code-only `/sim-*` skills could not provide (skills run only inside the Claude Code terminal; the terminal is itself the barrier for the wet-lab audience). The server is a **thin adapter over `cardiac_core`** — the same relationship the PubMed/Drive MCP connectors have to their APIs.

**Layout (`cardiac_mcp/` at repo root, sibling to `cardiac_core`/`cardiac_ml`):**
```
cardiac_mcp/
  core.py        ← ALL logic, transport-agnostic + unit-testable (imports cardiac_core LAZILY)
  server.py      ← FastMCP wrapper: registers core.* as tools/resources (no logic)
  __main__.py    ← `python -m cardiac_mcp` → stdio; HTTP later = one-line transport swap
  tests/test_core.py  (10 tests, gate logic fast + 1 slow end-to-end simulate)
.mcp.json        ← registers "cardiac-core" with Claude Code (env python, stdio, PYTHONPATH=repo)
```

**Two design decisions (user, 2026-06-26):**
- **Local stdio now, remote-HTTP later.** Achieved by keeping every behaviour in `core.py` pure functions; `server.py` only binds them to a transport; `__main__` chooses it. Promote to remote = `mcp.run()` → `mcp.run(transport="streamable-http")`, zero tool changes.
- **Both tracks, as separate tools** (the user's "quick look" vs "recorded" split):
  - **DIRECT** — `simulate(...)` → ephemeral CV measurement, **no `Lab/` record**, defaults to coarse `dx=0.02` (~8s; fine `dx=0.01` ~38s). Returns `{cv_cm_per_s, activated, grid, conductivity, note}` with a sanity `note` (mirrors the skill's "verify before presenting").
  - **GATED** — `build_manifest(...)` → `commit_experiment(token, confirmed=True)` → `run_experiment(dir)`. Ports the `/sim-experiment` **double-check gate STRUCTURALLY**: `build_manifest` returns a plain-text manifest + a **self-signed `experiment_token`** (base64 of `{manifest_text, params, sig=sha256}`); `commit_experiment` **refuses unless `confirmed=True` AND the token verifies intact**, so the written `Lab/{date}_{slug}/{MANIFEST.md, run.py}` is *provably* the manifest the scientist reviewed (the model cannot commit a script differing from what it showed). Reuses the skill conventions: slug-overwrite guard (never clobber a prior `MANIFEST.md`), `NOTEBOOK.md` row (built→done|failed), `run.py` generated from `API_CHEATSHEET.md` only. `run_experiment` runs the script in-env from repo root, parses CV, records the outcome both ways.

**Resources:** `cardiac://cheatsheet` (the canonical `API_CHEATSHEET.md` — the anti-hallucination asset, now available to *any* host, not just Claude Code) and `cardiac://notebook` (`Lab/NOTEBOOK.md`).

**Maintenance facts:**
- `mcp` SDK **1.28.0** installed in the `heart-conduction` env (`pip install "mcp>=1.2.0"`; bundles `FastMCP`). The server imports `cardiac_core` (torch) **lazily** inside `simulate`/`run_experiment` only → fast MCP handshake (no torch load on boot).
- FastMCP derives each tool's JSON schema from the `core` function's **signature + docstring** → the cheatsheet-accurate signatures are the contract. Bare-`dict` returns arrive as JSON **text content** (`structuredContent` only populated for typed-model returns); the host model reads the text — fine.
- **Validation (2026-06-26):** 10 `cardiac_mcp` tests pass (gate refusal w/o confirm, tampered-token refusal, folder+notebook write, no-overwrite, status update, end-to-end `simulate` CV in range); server boots with 5 tools + 2 resources; real stdio client↔server roundtrip OK (`build_manifest` + cheatsheet read over the wire).
- **Activation in Claude Code:** project-scoped `.mcp.json` → Claude Code prompts to approve the `cardiac-core` server on next start; once approved its tools/resources appear. (Server logs to stderr; stdout is the protocol channel.)

**Deferred (next MCP increments):** more resources/prompts (presets, `GLOSSARY.md`, a control-vs-knockdown prompt template); a media tool wrapping `cardiac_core.viz`; reentry/restitution recipes in the gated path (v1 is CV-strip only); the **streamable-HTTP transport + hosting/auth** for remote wet-lab scientists (the "end product").

## Goal-2 MCP server — standardization audit (2026-06-28)

Audited `cardiac_mcp` against the **official MCP spec, revision 2025-11-25** (4 parallel spec-research agents, verified against modelcontextprotocol.io + `schema.ts`, not memory). **Core insight:** a *running* server only needs code; a *standardized* one adds (1) a per-primitive **metadata layer** and (2) **distribution documents** — the latter is the "list of supporting materials" intuition. All audited features (annotations, outputSchema, mimeType) exist in the installed SDK `mcp` 1.28.0 (they predate 2025-11-25).

**Authoritative requirements that matter for our server (MUST/SHOULD):**
- **Tools** — MUST: `name` + valid non-null `inputSchema`; declare `tools` capability; validate inputs; recoverable failures returned as `isError: true` tool-results (NOT JSON-RPC errors — those are for protocol faults). SHOULD: `description`; `outputSchema` + conforming `structuredContent` for structured returns; tool **annotations** set intentionally. **Annotation defaults are a trap:** unset ⇒ `readOnlyHint=false, destructiveHint=true, idempotentHint=false, openWorldHint=true` (annotations are untrusted hints, but a good server sets them so careful hosts gate correctly).
- **Resources** — MUST: `uri` + `name`; SHOULD: `mimeType`. Templates use RFC 6570 `uriTemplate`. Declare `subscribe`/`listChanged` only if you actually emit them.
- **Prompts** — MUST: `name`; each argument MUST have `name`. Optional but high-fit for our recipes.
- **Lifecycle/serverInfo** — MUST: `serverInfo.name` + `version`. (FastMCP leaves `version=None` ⇒ falls back to the SDK version `1.28.0`; set it.) `instructions` optional/recommended (we have it).
- **Transports** — stdio (local, current) is correct; auth SHOULD NOT be used for stdio (env creds). Streamable HTTP (remote) carries the heavy obligations below. stdio MUST keep `stdout` clean (FastMCP logs to stderr — OK; `run_experiment` captures the child's stdout — OK).
- **Security (server)** — MUST validate all tool inputs; host MUST get user consent before invoking a tool; a code-executing tool SHOULD be sandboxed with filesystem restricted (to `Lab/`) and least privilege; MUST NOT accept tokens not issued to this server (no passthrough).
- **Remote (HTTP) delta** — auth flips from "SHOULD NOT" to a MUST stack: OAuth 2.1 + PKCE(S256), RFC 9728 Protected Resource Metadata, RFC 8707 Resource Indicators (audience binding), per-request `Authorization`, secure non-deterministic session IDs (MUST NOT authenticate via session), `Origin` validation → 403 (DNS-rebinding; FastMCP auto-enables for localhost binds), SSRF defenses.
- **Distribution (REQUIRED only to publish to the registry)** — `server.json` manifest (reverse-DNS `name` e.g. `io.github.<user>/cardiac-core`, immutable semver `version`, `packages[]`/`remotes[]`), README + ownership marker (`<!-- mcp-name: … -->`), `pyproject` console-script, LICENSE, Dockerfile, committed `uv.lock`; validate with the MCP Inspector (`uv run mcp dev`).

**`cardiac_mcp` gaps found (→ the 4-tier PLAN, 2026-06-28):**
- **Tier 1 (correctness, now):** tool annotations unset (all 5 advertised destructive/open-world); `serverInfo.version` defaults to SDK's; resource `mimeType` defaults to text/plain (cheatsheet/notebook are markdown); **two path-traversal input-validation bugs** — `run_experiment` runs any `run.py` (absolute/`..` `experiment_dir` escapes via `REPO_ROOT/experiment_dir`), `commit_experiment` folder uses the unsanitized model-supplied `date`.
- **Tier 2 (completeness):** bare-`dict` returns ⇒ no `outputSchema`/`structuredContent` (type the returns); no `cardiac_mcp/README.md`; 0 prompts (recipes are a natural fit); relies on a `PYTHONPATH` hack in `.mcp.json` instead of an installed console-script.
- **Tier 3 (remote-readiness):** `run_experiment` unsandboxed (local-OK, remote = RCE); the full OAuth/Origin/session/SSRF stack unbuilt — a project unto itself, do when a real HTTP deployment target exists.
- **Tier 4 (publishing, optional):** no `server.json`/reverse-DNS name/LICENSE/Dockerfile/Inspector validation — only needed for public registry discoverability.

**FastMCP (1.28.0) mechanics pinned for the plan:** ctor has `website_url` but **no `version`/`title`** ⇒ set `mcp._mcp_server.version = "0.1.0"` (verified flows to `serverInfo.version`); `mcp.add_tool(fn, annotations=ToolAnnotations(...))` (fields: title, readOnlyHint, destructiveHint, idempotentHint, openWorldHint); `@mcp.resource(uri, mime_type=…)`; structured output via typed returns or `add_tool(..., structured_output=True)`.

### Standardization — SHIPPED Tiers 1–3 (2026-06-28)

Executed the 4-tier PLAN on branch `mcp-standardization` (5 commits: docs `66263a8` → T1 `38d3178` → T2 `8458046` → T3 `2c4bc3a` → record `64c333d`). **Phase 4 (registry publishing) SKIPPED** by user choice (server fully usable locally + documented; publishing only adds public discoverability). **16 `cardiac_mcp` + 140 `cardiac_core` tests green; HTTP mode live-verified.**

- **T1 (metadata + security):** per-tool `ToolAnnotations` (read-only `simulate`/`build_manifest`/`list_experiments`; `commit_experiment` additive; `run_experiment` destructive); `serverInfo.version=0.1.0` via `mcp._mcp_server.version`; resources `text/markdown`; **two path-traversal guards** — `run_experiment` must resolve inside `Lab/` (`is_relative_to`), `build_manifest` date must match `^\d{4}-\d{2}-\d{2}$`.
- **T2 (completeness):** `TypedDict` returns → `outputSchema`+`structuredContent` on all 5 tools (FastMCP auto-emits from typed returns — no `structured_output=True` needed); `list_experiments` always returns `count`; 2 prompts (`measure_cv`, `control_vs_knockdown`); `cardiac_mcp/README.md`; **Option B packaging** — extended the ROOT `pyproject.toml` (`include=["cardiac_core*","cardiac_mcp*"]`, `dependencies=["mcp>=1.2.0"]`, `[project.scripts] cardiac-mcp`); single `pip install -e .`; `.mcp.json` now launches the `cardiac-mcp` console script (no `PYTHONPATH`).
- **T3 (remote-readiness):** `run_experiment` provenance-marker check (only runs cardiac-core-generated scripts) + `RLIMIT_CPU`(loose, `timeout_s*ncpu`)/`RLIMIT_FSIZE` preexec — **NO `RLIMIT_AS`** (caps virtual AS → aborts torch); `CARDIAC_MCP_TRANSPORT=http` → localhost-bound Streamable HTTP (uvicorn 127.0.0.1, DNS-rebinding protection, unauthenticated-warning to stderr); `cardiac_mcp/REMOTE_DEPLOY.md` (spec-cited OAuth/Origin/SSRF/sandbox checklist + deploy gate).
- **Execution note:** `RLIMIT_CPU` loosened from the plan's `≈timeout_s` to `timeout_s*ncpu` (it sums CPU-time across torch threads → would false-kill a real run); verified by `test_run_experiment_under_limits` (the one test exercising the real subprocess+limits path).

The plan (PLAN.md + Mutation Log) was hardened through 3 audit rounds to CONVERGENCE: CLEAR before execution; archived in `plans/2026-06-28_*`.

## Cross-Engine Capability Census (2026-06-01)

Read-only census of all three engines' construct/run/state/stimulus/geometry surfaces, to ground the vocabulary + API. The **ionic layer and physical conventions are already a shared language; divergence is concentrated in construction, voltage naming, state, and the run/result contract.** LBM is the consistent outlier.

| Concept | Monodomain V5.5 | Bidomain V1 | LBM V1 |
|---|---|---|---|
| **Construct** | `MonodomainSimulation(spatial, ionic_model, stimulus, dt, splitting, ionic_solver, diffusion_solver, linear_solver, cell_type, pcg_tol, pcg_max_iter)` | `BidomainSimulation(spatial, ionic_model, stimulus, dt, splitting, ionic_solver, diffusion_solver='decoupled', parabolic_solver, elliptic_solver='auto', theta, device)` | `LBMSimulation(Nx, Ny, dx, dt, D, ionic_model, Cm, lattice, weights_mode, bounce_masks)` |
| **Spatial obj** | separate: `FDM/FEM/FVMDiscretization(grid, D, chi=1400, Cm=1, …)` | separate: `BidomainFDMDiscretization(grid, BidomainConductivity(D_i=0.00124,D_e=0.00446), Cm, stencil='5pt')` | **none** — Nx/Ny/dx/D inline |
| **Voltage field** | `state.V` | `state.Vm` (+ `.V` alias) | `self.V` (= Σfᵢ) |
| **State** | `SimulationState` dataclass | `BidomainState` dataclass (+phi_e) | on the sim object (`LBMState` dataclass exists, unused) |
| **run()** | yields `SimulationState`; also `run_to_array(t_end,save_every)→(times,V) np` | yields `BidomainState`; no run_to_array | **returns** `(times list, V_history list of (Nx,Ny))` |
| **step()** | `step(dt)` | via splitting | `step()` (no arg) |
| **Voltage out** | flat `(n_dof,)` | flat `(n_dof,)` | grid `(Nx,Ny)` |
| **Stimulus** | `StimulusProtocol` + region callables + `add_s1s2_protocol`/`add_regular_pacing` | `StimulusProtocol` + region helpers (`left_edge_region`…) | `add_stimulus(mask,…)` — raw mask, no protocol/pacing |
| **Stim default amp** | −52 | −52 | −80 |
| **Geometry** | `StructuredGrid.create_rectangle(Lx,Ly,Nx,Ny,device,dtype)` (+ TriangularMesh for FEM) | same `StructuredGrid` + `BoundarySpec` (insulated/bath_coupled/bath_coupled_edges) | Nx,Ny,dx inline + `bounce_masks` dict |
| **Engine-specific knobs** | diffusion_solver (cn/bdf1/bdf2/fe/rk2/rk4), linear_solver (pcg/chebyshev/dct/fft/none), boundary_mode (face_mirror…), stencil (cardinal4/moore8_*) | parabolic_solver (pcg/chebyshev/spectral), elliptic_solver (auto/spectral/pcg_spectral/pcg_gmg), theta (0.5=CN/1=BDF1), stencil (5pt/mehrstellen), BoundarySpec | lattice (d2q5/d2q9), weights_mode (canonical/uniform_8), ω/τ derived from D (`tau_from_D`) |

**Already aligned (free wins for the vocabulary):** IonicModel ABC (identical across all 3); `dt`, `Cm`, `device`, float64, `(Nx,Ny)` ij convention; stimulus **accumulate (`+=`) in all three** (resolves the old `=` vs `+=` open question); negative amplitude = depolarizing; reaction `/Cm` (all, post-V5.5-fix); `splitting` (strang/godunov) + `ionic_solver` (rush_larsen/forward_euler) names (mono+bidomain); `run(t_end, save_every)` exists on all three.

**Divergent → decisions for the glossary/API:**
1. Voltage name: `V` (mono/LBM) vs `Vm` (bidomain). [lean `Vm`]
2. State: dataclass (mono/bidomain) vs on-object (LBM).
3. `run()` contract: generator-of-state vs `(times, V_history)` tuple; + mono's `run_to_array`.
4. Voltage output: flat `(n_dof,)` vs grid `(Nx,Ny)`; np vs tensor; list vs array.
5. Construction shape: spatial-object (mono/bidomain) vs inline params (LBM); `StimulusProtocol` vs raw mask; conductivity (`D/chi/Cm` vs `BidomainConductivity(D_i,D_e)` vs inline `D`).
6. Stimulus declaration: region-callable + pacing helpers vs raw mask; default amp −52 vs −80.
7. Geometry input: `StructuredGrid(Lx,Ly,Nx,Ny)` vs raw `Nx,Ny,dx`.
8. `chi` handling: exposed (mono) vs deprecated/absorbed-into-D (bidomain/LBM).
9. Internal naming: `ionic_time_stepping` vs `ionic_stepping`.

**Two-tier lens for the glossary:** UNIVERSAL concepts (every engine has them → one enforced name: transmembrane potential, ionic model, stimulus, dt, grid, orchestrator, step/run, ionic+diffusion stepper) vs ENGINE-SPECIFIC (canonical name, only where applicable: phi_e + elliptic solve = bidomain; f/distributions/lattice/collision = LBM).

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Canonical formulation = B** (decided 2026-06-24) | **Form B is the target; converge monodomain in Phase 4** | Both forms are now physically correct (V5.5 fixed A's reaction), so the choice is pure software engineering — and B wins on every axis: it confines all χ/Cm scaling to `ConductivityConfig` (decision #13; A's engine is a *second* scaling authority via its χ·Cm mass term), it's the non-fragile single-site convention (A scatters χ·Cm across FDM mass + FEM M + FVM Vol + DCT/FFT denominators — exactly the scattering that caused the V5.4 reaction bug), it's already 2/3 engines, and it's the clean textbook operator `(I − θ·dt·L)`. A's only edge (operator transparency) is a docs value, neutralized at the API since the user passes σ/χ/Cm to `ConductivityConfig` either way. **Two-phase:** keep both now (firewall's `for_monodomain()` absorbs the asymmetry); convert monodomain's Form-A diffusion → B *as part of* the Phase-4 rewire into cardiac_core (drop χ·Cm from mass term + spectral denominators, consume pre-scaled D) — no new fork needed. At that point the `for_monodomain()` special-case is **deleted** and `ConductivityConfig` collapses to one emitter (physical D + Cm). |
| Formulation unification (reaction) | Keep V5.4 frozen; **fork V5.5** (Formulation-B reaction) | Converting V5.4 in place risked its 77 tests; instead forked V5.5 (2026-05-30) which divides the reaction by Cm. V5.5 is the Cm-correct monodomain to consolidate. See "V5.5 Cm-correct fork" below. |
| Shared code | `cardiac_core/` package | Extract ionic/mesh/stimulus; don't merge solver internals |
| ConductivityConfig | sigma → D in one place | chi/Cm appear only in ConductivityConfig |
| Simulation protocol | `Simulation` protocol + `create_simulation()` factory | Optimizer/Surrogate call any engine interchangeably |
| Target architecture | **Option B — unified core** | `cardiac_core/` owns shared code, engines import from it. Not a wrapper; the actual codebase. Eliminates `sys.modules` hack and code duplication. |
| Implementation order | API first, then extract | Phase 0 builds API as wrapper (low risk). Later phases move shared code in and rewire engine imports. |
| Canonical LBM | LBM V1 | More features: MRT, D2Q9, 3 BC types, compiled steps |
| FDM stencil (bidomain) | Face-based symmetric | SPD required for PCG on elliptic |
| Cm convention | Always 1.0 | All ionic models output pA/pF |
| Project structure | Research = writing, Engines = code | Experiments in engine `experiments/`, EXPERIMENT.md backlinks to research |

### Phase 0: API Layer (completed 2026-03-17)

`cardiac_core/` API currently wraps unchanged engines — engines still own their code, `cardiac_core` imports from them. The `_prepare_engine()` function flushes `sys.modules` to work around both engines using the `cardiac_sim` namespace. Eliminated in Phase 1 when shared code moves into `cardiac_core/`.

**File format** (`file_format.py`): `CardiacMeshData` dataclass + `save_cardiac_mesh()` / `load_cardiac_mesh()` / `create_cardiac_mesh()`. Format version 1 `.npz` stores grid, conductivity, stimulus regions, and optional bidomain sigma fields.

**API** (`api.py`): `monodomain()`, `bidomain()`, `lbm()` each accept `str | CardiacMeshData`, construct the engine, and return `CardiacSimulation` with a generator `.run()` interface yielding `SimulationSnapshot(t, V, phi_e, Nx, Ny, dx, dy)`. V is always (Nx, Ny) grid shape.

**Bidomain D_i/D_e derivation**: When file lacks sigma_i/sigma_e, derives from D_eff and sigma_ratio: `D_i = D_eff * (1+r)/r`, `D_e = D_eff * (1+r)`.

**34 tests** (10 file format, 6 monodomain, 6 LBM, 7 bidomain, 5 integration). Wrapper output matches direct engine construction exactly (verified with `torch.allclose` at atol=1e-10). All 93 V5.4 tests and 10 bidomain spot-check tests still pass.

### cardiac_core drift reconciled (2026-05-30)

Since Phase 0, `cardiac_core/` grew a **convenience/analysis layer** (all still wrapper-level — imports FROM engines, no shared-code packages yet). Current package = `file_format.py`, `api.py` **plus**:
- `run.py` — one-shot `run_monodomain/run_bidomain/run_lbm`, `simulate`, `SimulationResult` (call once → `(times, V)`, no generator).
- `analysis.py` — pure tensor analysis: `activation_time`, `conduction_velocity`, `apd_at`/`apd_map`, `dominant_frequency`, `wavefront_mask`, `phase_map`, `phase_singularities`, `restitution_curve`.
- `geometry.py` — mask/region/distance/fiber helpers (`circle_mask`, `annulus_mask`, `left_edge_mask`, `boundary_distance`, `fiber_field_transmural`, …).
- `io.py` — result `.npz` save/load (`save_result`/`load_result`).
- **Test count is now 77** (not 34) — the convenience layer is tested (`test_run`, `test_analysis`, `test_geometry`, `test_io`).

**Assessment:** the drift is benign and additive — it does NOT block consolidation Phase 1 (ionic extraction is orthogonal to these wrappers). The target architecture (ionic/mesh/stimulus/conductivity packages, engines importing from them) is still unstarted. Build Phase 1 against **V5.5**. Also fixed the stale `Engines/` symlink index: `Engines/cardiac_core` (was broken `../../cardiac_core` → `../cardiac_core`), restored `lbm_v1 → ../LBM/Engine_V1` (old target `../Monodomain/LBM_V1` was stale), added `monodomain_v5.5 → ../Monodomain/Engine_V5.5`.

### V5.5 Cm-correct fork (completed 2026-05-30)

Rather than convert V5.4 in place (risking its 77 tests), forked `Monodomain/Engine_V5.5` — a full copy of V5.4 with ONE functional change: the operator-split reaction divides by the tissue Cm, `dV = -(Iion + Istim)/state.Cm` (Formulation B, matching Bidomain V1 / LBM V1). V5.4 stays the frozen baseline.

- **Plumbing:** `SimulationState.Cm` (default 1.0) populated from `spatial.Cm` at construction (direct read, no `getattr` fallback → fails loud). All three schemes expose a `Cm` property; **FEM had to add `self._Cm`/`self._chi`** (it only baked them into `self.M`) — an audit-CRITICAL catch.
- **Diffusion untouched:** the `chi·Cm` mass term already handled arbitrary Cm; only the reaction was broken.
- **Dead LBM path removed:** V5.5's internal `cardiac_sim/simulation/lbm/` had zero importers/tests (boundary work uses the separate `LBM/Engine_V1`); deleted along with the now-dead `step_with_V`.
- **Validation** (`test_phase10_cm_scaling.py`): Cm=1 bit-identical to V5.4 (golden, max|dV|=0); exact 1/Cm reaction scaling to 3.55e-15; Bidomain V1 cross-check CV 54.35 vs 54.35 cm/s (Cm=1), 28.09 vs 27.77 cm/s (Cm=2, 1.1%).

**Physics correction (important):** there is NO Cm time-dilation invariant. Tissue Cm divides only the voltage update; gate time-constants and concentration rates carry no Cm. So scaling Cm changes AP **morphology**, not timescale: APD does NOT scale (218→292 ms at k=2, not 2×). CV *does* scale ~1/Cm, but by eikonal scaling (CV ∝ √(D_phys·upstroke_rate), both ∝1/Cm), not dilation. The original PLAN and both audit passes asserted the (false) dilation invariant; the empirical 0D test caught it. The fix was correct throughout — only the validation strategy was wrong.

### Migration plan (toward unified core)

> **SUPERSEDED (2026-06-25).** This original "rewire engines to import from cardiac_core, delete their copies" plan was NOT executed as written. The unified core was instead achieved by **A2 copy-vendoring** — the engines were COPIED into `cardiac_core/_monodomain/_bidomain/_lbm` (originals frozen, not deleted/rewired), the `_prepare_engine` hack deleted, shared `ionic/mesh/stimulus` extracted. See "cardiac_core unified ground-up package — SHIPPED". The deferred cleanups below that DO remain: Phase-4 Form-A→B convergence + delete `for_monodomain()`, and the Phase-1 downstream consumer (Surrogate/Optimizer) migration off the engine-local `cardiac_sim.ionic`. Table kept for historical context.

| Phase | What | Status |
|-------|------|--------|
| Prereq | **Monodomain V5.5 fork** — Cm-correct reaction; drop dead LBM | **DONE** (2026-05-30) |
| Phase 0 | API layer + file format (wrapper, imports from engines) | **DONE** (2026-03-17) |
| Phase 1 (copy) | Canonical `cardiac_core/ionic/` superset copy + editable install + lazy `__init__` | **DONE** (2026-05-31, copy-only) |
| Phase 1 (migrate) | Rewire engines to import `cardiac_core.ionic` + delete local copies + migrate Surrogate/Optimizer consumers | **DEFERRED** — audit found big-bang deletion breaks engine tests/examples + active Surrogate datagen + Optimizer. Must be done per-consumer, repo-wide discovery, never delete out from under a live consumer. |
| Phase 2 | Move mesh + stimulus into `cardiac_core/mesh/`, `cardiac_core/stimulus/` | Not started |
| Phase 3 | Add `ConductivityConfig` — sigma → D conversion in one place | Not started |
| Phase 4 | Rewire engines to import from `cardiac_core` (delete their copies) **+ convert monodomain diffusion Form A→B** (drop χ·Cm from mass term/spectral denominators, consume pre-scaled D) so all engines share Form B; then delete `ConductivityConfig.for_monodomain()` asymmetry | Not started |
| Phase 5 | Remove `_prepare_engine()` hack, unify `cardiac_sim` namespace | Not started |
| Validation | All 149+ tests pass, no duplicated ionic/mesh/stimulus code | Not started |

## Open Questions

- ~~Should V5.4 eventually convert to Formulation B?~~ **RESOLVED (2026-05-30):** done in the V5.5 fork (V5.4 stays frozen). See "V5.5 Cm-correct fork" above.
- ~~V5.4 LBM source term `/(chi·Cm)` should switch to `/Cm`~~ **MOOT for V5.5:** the dead internal LBM path was removed from V5.5. The canonical LBM (LBM V1) is already Formulation B. The `/(chi·Cm)` reconciliation only matters if cardiac_core ever revives a monodomain-LBM path under ConductivityConfig.
- When consolidating, build `cardiac_core` against **V5.5** (Cm-correct), not V5.4.
- ~~Stimulus overlap: `=` vs `+=`?~~ **RESOLVED (2026-06-01 census):** ALL THREE engines accumulate (`+=`) — V5.5 (`_evaluate_Istim`, `Istim = Istim + …`), Bidomain (`Istim[mask] += …`), LBM (overlapping stimuli accumulate). Canonical = accumulate. (The earlier "V5.4 uses `=`" note was wrong or pre-fix.)
- **API-debt — `create_cardiac_mesh(D=…)` bypasses the Form-A/B firewall (2026-06-30, from `ionic_model_optimization` chip-fit):** passing an *effective* diffusivity `D≈1e-3` with the **default `chi=1400`** silently mis-scales — the FDM operator (`_monodomain/.../fdm.py:37,159`) builds the Laplacian from `D` alone with `χ·Cm` only in the mass term, so membrane-effective diffusivity = `D/(χ·Cm)` ≈ 1400× too low → CV ∝ √D collapses ~37× → discrete source–sink conduction block (Vmax pools ~80–123 mV, CV=NaN). `(D=1e-3, χ=1400)` is exactly degenerate with `(D=7.14e-7, χ=1)` — faithful physics of the wrong number, not a solver bug. **Workaround:** pass `chi=1.0` when feeding an effective D. **Fix (open):** route `create_cardiac_mesh` through `ConductivityConfig`, or warn when an effective-D is supplied with `chi≠1`. See IDEALOG 2026-06-30 thread + Next Step.

## Connections
- **Engines**: All three + cardiac_core (target)
- **Related research**: All active questions depend on stable engines
- **Pipelines**: Optimizer V1 needs unified API for cross-engine validation (V2)
- **Key document**: [REVIEW.md §6](../../Bidomain/Engine_V1/REVIEW.md) — full technical proposal
