# Engine Consolidation — Knowledge File

> This file is a running synthesis. Updated as findings accumulate.
> When the question is complete, a copy is promoted to `Research/Knowledge/`.

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

## North-Star: Conversational Simulation Builder (vocabulary-first → unified API)

**Goal (now the question's main goal).** A non-coder converses with Claude, which both *builds* cardiac simulations and *teaches* how conduction works. Two layers:
1. **Unified construction API (Goal 1)** — one standardized, engine-agnostic, easy-to-construct way to declare + run: a declarative, validated, serializable **SimulationSpec** → run → **SimulationResult** → analysis. Three field tiers: required (LLM asks) / defaulted (silent good values) / derived (computed).
2. **Self-contained LLM wrapper (Goal 2)** — Claude skills + reference docs driving Goal 1 under a strict protocol (gather → validate → construct → run → verify → present).

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

## Connections
- **Engines**: All three + cardiac_core (target)
- **Related research**: All active questions depend on stable engines
- **Pipelines**: Optimizer V1 needs unified API for cross-engine validation (V2)
- **Key document**: [REVIEW.md §6](../../Bidomain/Engine_V1/REVIEW.md) — full technical proposal
