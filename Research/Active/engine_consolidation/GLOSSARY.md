# Engine Consolidation — Unified Glossary (DRAFT)

**Status:** DRAFT — harvested 2026-06-02 from source via read-only census of all four surfaces.
**Purpose:** the *ubiquitous language* for the conversational simulation builder (North-Star Goal 1).
One canonical name per concept across the three engines + the existing `cardiac_core` wrapper, so
the unified API, the SimulationSpec, and the LLM intake questionnaire all speak one vocabulary.

**How to read this doc.** Each row = one concept. Columns show the *actual identifier each
surface uses today* (with `file:line` so claims are checkable), then a **Proposed canonical** name
and a **Status**:

| Status tag | Meaning |
|---|---|
| ✅ ALIGNED | all surfaces already agree (free win — adopt as-is) |
| 🟡 MINOR | trivial divergence (rename / alias), low risk |
| 🔴 DECISION | genuine, high-leverage choice — **needs sign-off before the API freezes it** |
| ⚙️ ENGINE-SPECIFIC | not universal; canonical name only where the concept applies |

Source surfaces: **M** = Monodomain V5.5, **B** = Bidomain V1, **L** = LBM V1, **CC** = `cardiac_core` (existing wrapper, an *already-made* unification attempt — a strong prior, not gospel).

---

## Two-Tier Lens

- **UNIVERSAL** — every engine has the concept → exactly one enforced name (transmembrane potential, ionic model, stimulus, dt, grid, orchestrator, state, step/run, ionic+diffusion stepping).
- **ENGINE-SPECIFIC** — concept exists only in some engines → canonical name *where applicable* (φ_e + elliptic solve = bidomain only; distributions/lattice/collision/streaming = LBM only; FEM/FVM/multi-solver knobs = monodomain only).

---

## Naming Principles (user, 2026-06-02)

Two rules govern every decision below. They turn "pick a name" into "apply the rule".

### P1 — Mixed (subscripted) notation for any intracellular / extracellular / membrane quantity

For anything that *could* carry an intra/extra/membrane distinction, keep the **explicit domain
subscript** (`Vm`, `phi_e`, `phi_i`, `D_i`, `D_e`, `D_eff`, `sigma_i`, `sigma_e`). Only collapse to a
bare symbol when the concept is **absolutely, identically true across all engines** — otherwise the
bare symbol silently hides *which* domain it is.

- ✅ `Vm` — transmembrane potential is *the same physical thing* in monodomain and bidomain → the `m`
  subscript is universally true, so `Vm` is the one clean name (this is why bare `V` was rejected).
- ❌ bare `D` — **not** universally true: monodomain/LBM carry an **effective** diffusivity `D_eff`
  (the i/e collapse `D_i·D_e/(D_i+D_e)`), while bidomain carries the **components** `D_i`, `D_e`.
  A bare `D` would pretend these are the same quantity; they aren't. So conductivity stays subscripted:
  **`D_eff` (mono/LBM), `D_i`/`D_e` (bidomain)**, with `sigma_*` as the corresponding inputs.

> Rule of thumb: a name is allowed to drop its subscript only if removing it changes nothing about
> *what domain the quantity belongs to*. `Vm`'s "m" is invariant across engines → safe to treat as the
> primary name. `D`'s domain is not invariant → must stay explicit.

### P2 — Structured grid is the primary standard; complex mesh is the explicit secondary path

Nearly every simulation we run is on a **structured grid**, and the LBM-style grid notation is the
simplest, most direct way to express that. So the **structured grid is the canonical/default
representation**:

- voltage & fields are **grid-shaped `(Nx, Ny)`** (time series `(T, Nx, Ny)`) — *not* flat `(n_dof,)`;
- geometry is the lightweight grid descriptor (`Nx, Ny, dx[, dy]`), LBM-style;
- the **unstructured / complex-mesh path (FEM, `TriangularMesh`, flat `(n_dof,)`) is a separate,
  explicitly-named secondary standard** — supported, but not the default the API/LLM reaches for first.

This makes LBM's representation the template for the common case and quarantines the flat-`n_dof`
generality (which only FEM truly needs) behind an opt-in.

---

## TIER 1 — Universal Concepts

### 1. Top-level orchestrator

| Surface | Identifier | Location |
|---|---|---|
| M | `MonodomainSimulation` | `Monodomain/Engine_V5.5/cardiac_sim/simulation/classical/monodomain.py:231` |
| B | `BidomainSimulation` | `Bidomain/Engine_V1/cardiac_sim/simulation/classical/bidomain.py:18` |
| L | `LBMSimulation` | `LBM/Engine_V1/src/simulation.py:34` |
| CC | `CardiacSimulation` (wrapper) + factory fns `monodomain()`/`bidomain()`/`lbm()` | `cardiac_core/api.py:150, 802, 901, 1047` |

**Proposed canonical:** `Simulation` (the unified construct), built by an engine-agnostic factory
(`create_simulation(spec)` / per-model `monodomain()/bidomain()/lbm()`). **Status: 🟡 MINOR** —
all three already use the `<Model>Simulation` pattern; CC already provides the factory layer.

---

### 2. Transmembrane potential  ✅ **RESOLVED → `Vm`** (user, 2026-06-02)

| Surface | Identifier | Shape | Location |
|---|---|---|---|
| M | `state.V` | flat `(n_dof,)` | `…/classical/state.py:72` |
| B | `state.Vm` (with `.V` alias) | flat `(n_dof,)` | `Bidomain/…/state.py:53, 93` |
| L | `self.V` (= Σ fᵢ) | grid `(Nx,Ny)` | `LBM/Engine_V1/src/simulation.py:115` |
| CC | `V` (snapshot + result) | grid `(Nx,Ny)`, tensor f64 | `cardiac_core/api.py:127`, `run.py:33` |

**Canonical: `Vm`** — physically precise (transmembrane potential); disambiguates cleanly from
`phi_e` / `phi_i` in bidomain. Bidomain is already native `Vm`.

**Migration cost (accepted):** rename `V`→`Vm` in M + L; **revert cardiac_core's shipped `V`** end-to-end
(`SimulationSnapshot.V`, `SimulationResult.V`, analysis-fn arg names, the 77 tests). Keep a read-only
`.V` *alias* (bidomain already has one) for back-compat during migration, then deprecate.
**Status: ✅ RESOLVED.**

> ⚠️ The ionic `IonicModel` ABC methods take a positional `V` argument (`compute_Iion(V, …)` etc.),
> byte-identical across engines and consumed by Surrogate/Optimizer. **Lean: leave the ABC param as
> `V`; canonicalize `Vm` only at the State/API layer.** Renaming the ABC is a separate, wide-blast follow-up.

---

### 3. Ionic model interface  ✅ **the proof that unification works**

The `IonicModel` ABC is **byte-identical** across M/B/L (and is the canonical copy in `cardiac_core/ionic/`).

| Concept | Canonical (already universal) | Location (M) |
|---|---|---|
| ABC | `IonicModel` | `…/ionic/base.py:25` |
| cell-type enum | `CellType` (`ENDO=0, EPI=1, M_CELL=2`) | `…/ionic/base.py:18` |
| total ionic current | `compute_Iion(V, ionic_states)` | `base.py:127` |
| gate steady states | `compute_gate_steady_states(V, ionic_states)` | `base.py:146` |
| gate time constants | `compute_gate_time_constants(V, ionic_states)` | `base.py:166` |
| concentration rates | `compute_concentration_rates(V, ionic_states)` | `base.py:185` |
| one-step advance | `step(V, ionic_states, dt, Istim=None) → (V_new, ionic_states_new)` | `base.py:101` |
| initial state | `get_initial_state(n_cells=1)` | `base.py:84` |
| props | `name`, `n_states`, `V_rest`, `state_names`, `gate_indices`, `concentration_indices` | `base.py:49–79` |

**Concrete models:** `TTP06Model`, `ORdModel` (all 3); `PHAS13Model`, `MHAS13Model`, `PaciModel` alias (M + CC superset). **Status: ✅ ALIGNED.**

> ⚠️ One latent rename, already fixed in the canonical copy: LUT keyword `celltype_is_endo` (L) vs `cell_type_is_endo` (old M) → canonical **`celltype_is_endo`** (`cardiac_core/ionic/lut.py:106`).

---

### 4. Ionic state vector  ✅

| Surface | Identifier | Shape |
|---|---|---|
| M / B | `ionic_states` (on State) | `(n_dof, n_states)` |
| L | `self.ionic_states` / `LBMState.ionic_states` | `(n_cells, n_states)` = `(Nx*Ny, n_states)` |

**Proposed canonical:** `ionic_states`, `(n_nodes, n_states)`, with companion `gate_indices` /
`concentration_indices` (all identical across surfaces). **Status: ✅ ALIGNED.**

---

### 5. State container  ✅ **public RESOLVED** (only `SimulationResult`) / 🟡 **internal live-State**

| Surface | Identifier | Lives where | Notable fields |
|---|---|---|---|
| M | `SimulationState` (dataclass) | passed through solvers | `spatial, n_dof, x, y, V, ionic_states, gate_indices, concentration_indices, t, Cm, stim_*` — `state.py:25` |
| B | `BidomainState` (dataclass) | passed through solvers | same **+ `Vm`/`phi_e`/`stim_amplitudes_e`**, `.V` alias, `.clone()` — `state.py:22` |
| L | `LBMState` dataclass **exists but UNUSED**; real state on the sim object (`self.f/V/ionic_states/t`) | on orchestrator | `f, V, ionic_states, mask, t` — `src/state.py:8` |

**Two distinct objects (the #6 `batch` model collapsed the old 3-object view to 2):**

1. **Public output — `SimulationResult` ONLY** (✅ RESOLVED via #6). The user never sees a live state
   object; eager and streamed both hand back `SimulationResult`. The old per-engine yield of the live
   mutable State (a footgun — it mutates under any retained reference) is **gone**. No `Snapshot` type.

2. **Internal live `State`** (🟡 — internal refactor, not user-facing). Recommend **one unified dataclass**
   `State(t, Vm, ionic_states, gate_indices, concentration_indices, Cm, n_dof, coords)`; `phi_e` an
   optional bidomain field (P1 subscript); `f` an LBM-only extension. LBM should *use* it instead of
   free attributes — kills the last structural outlier. **Deferrable**: it doesn't change the public
   contract, so it can land in a later code phase. **Status: ✅ public / 🟡 internal (recommend unify, defer-able).**

---

### 6. `run()` / result contract  ✅ **RESOLVED → eager `run()`, streaming via `batch=`** (user, 2026-06-02)

| Surface (today) | `run()` signature | Returns / yields | Output shape |
|---|---|---|---|
| M | `run(t_end, save_every=1.0, callback=None)` | **yields** `SimulationState` | flat `(n_dof,)` |
| M | `run_to_array(t_end, save_every=1.0, …)` | **returns** `(times, V)` numpy | `(n_saves,)`, `(n_saves, n_dof)` |
| B | `run(t_end, save_every=1.0, callback=None)` | **yields** `BidomainState` | flat `(n_dof,)` |
| L | `run(t_end, save_every=1.0)` | **returns** `(times, V_history)` lists | list of `(Nx,Ny)` tensors |
| CC | `CardiacSimulation.run(t_end, save_every)` | **yields** `SimulationSnapshot` | grid `(Nx,Ny)` |
| CC | `simulate(…)` / `run_<engine>(…)` | **returns** `SimulationResult(times, V, phi_e)` | `(n_saves, Nx, Ny)` tensor f64 |

**Canonical contract — ONE method, eager by default, one knob for streaming:**

```
run(t_end, save_every=1.0, *, batch=None, record=("Vm",), callback=None)
```

| `batch` | behavior | returns |
|---|---|---|
| `None` (default) | **eager** — integrate to `t_end`, collect everything | one `SimulationResult` |
| `k` (int ≥ 1) | **streaming** — yield in chunks of ≤ k save-points (`k=1` = frame-by-frame) | `Iterator[SimulationResult]` |

- **No separate `stream()` method and no `Snapshot` type.** A streamed chunk is just a
  `SimulationResult` with `T ≤ k` (a frame is `T=1`). So the *only public output type is
  `SimulationResult`* — eager or streamed.
- **`SimulationResult`** (per #2/#7/P1/P2): `times (T,)`, **`Vm (T, Nx, Ny)`** torch f64,
  `phi_e (T,Nx,Ny)|None` (auto for bidomain), optional `ionic_states` (opt-in via `record`),
  `dx, dy`; thin `.cv()/.apd()/…` hooks into `analysis.py`.
- **`record=(...)`** decides which fields are kept ("outputs drive the run"): `Vm` always; `phi_e`
  auto-added for bidomain; `ionic_states` opt-in (Surrogate ground-truth gen). Bounds memory.
- **`callback`** retained for eager early-stop / progress; in streaming mode the caller's loop can `break`.

> ⚠️ Accepted wart: `run()`'s return type depends on `batch` (`SimulationResult` vs
> `Iterator[SimulationResult]`). Deliberate — one verb instead of two; the eager default is unaffected.

**Memory rationale:** 2D f64 grids are small (200×200 over 300 ms ≈ 100 MB eager) → eager is the right
default; `batch=k` is the pressure-release valve for fine `save_every` / future 3D / live viz.
**Status: ✅ RESOLVED.**

---

### 7. Voltage output shape & type  ✅ **RESOLVED → grid `(Nx,Ny)` (structured primary)** (P2, 2026-06-02)

| Surface | Shape | Type |
|---|---|---|
| M / B | flat `(n_dof,)` | torch tensor (numpy via `run_to_array`) |
| L | grid `(Nx,Ny)` | torch tensor (python list of frames) |
| CC | grid `(Nx,Ny)` / `(n_saves,Nx,Ny)` | torch.Tensor float64 |

**Canonical (per P2 — structured is primary):** the API returns **grid-shaped `Vm`** —
`(Nx, Ny)` per snapshot, `(T, Nx, Ny)` for histories — as **`torch.Tensor` float64**. This is the
LBM/CC representation and matches the analysis helpers (which already assume `(n_saves, Nx, Ny)`).
Flat `(n_dof,)` is **not** the default — it is reserved for the *complex-mesh (FEM) secondary path*,
where there is no grid. On the structured path, flat is an internal engine detail reshaped at the API
boundary (CC already does this). **Status: ✅ RESOLVED.**

---

### 8. `step()` — single time advance  🟡

| Surface | Signature | Effect |
|---|---|---|
| M | `step(dt=None)` (uses `self.dt`) | mutates state in place |
| B | via `SplittingStrategy.step(state, dt)` (no public `step(dt)` on sim) | mutates state |
| L | `step()` (no arg; uses `self.dt`) | mutates `self.f/V/…` |
| CC | `CardiacSimulation.step()` | delegates |

**Proposed canonical:** `step()` (no required arg; uses configured `dt`) on the orchestrator.
**Status: 🟡 MINOR** (add a public `step()` to bidomain sim).

---

### 9. Stimulus  🟡 / ✅ (default amplitude RESOLVED → −52, user 2026-06-24)

| Concept | M | B | L |
|---|---|---|---|
| event class | `Stimulus(region, start_time, duration, amplitude=-52.0)` | `Stimulus(region, start_time, duration, amplitude=-52.0)` | `Stimulus(mask, start, duration, amplitude)` |
| collection | `StimulusProtocol` (`stimuli: list`) | `StimulusProtocol` | **none** (list on sim) |
| add one | `add_stimulus(region, start_time, duration=1.0, amplitude=-52.0)` | same | `add_stimulus(mask, start, duration, amplitude=-80.0)` |
| S1–S2 | `add_s1s2_protocol(region, n_s1, bcl, s2_ci, …)` | same | — |
| regular pacing | `add_regular_pacing(region, bcl, n_beats, start_time, …)` | same | — |
| region spec | callable `(x,y)->bool` **or** mask tensor | callable/mask + helpers (`left_edge_region`, `circular_region`, …) | **raw `(Nx,Ny)` bool mask only** |
| overlap semantics | accumulate (`+=`) | accumulate (`+=`) | accumulate (`+=`) |
| **default amplitude** | **−52.0** | **−52.0** | **−80.0** ⚠️ |
| field name nit | `start_time` | `start_time` | `start` ⚠️ |

Locations: M `…/tissue_builder/stimulus/protocol.py:15,71,86,121`; B `…/stimulus/protocol.py:15,71,86,121` + `regions.py`; L `src/simulation.py:24,136`.

**Proposed canonical:** `Stimulus(region, start_time, duration, amplitude)` + `StimulusProtocol`
with `add_stimulus` / `add_s1s2_protocol` / `add_regular_pacing`; `region` accepts callable **or**
mask (LBM gains the protocol + pacing helpers + region callables; rename `start`→`start_time`);
accumulate is canonical (✅ all three). **Default amplitude ✅ RESOLVED → −52** (user, 2026-06-24): classical-engine (M/B) majority; LBM's −80 retired. Depolarization is identical because the ionic model is byte-identical across engines and the stimulus enters the same `R=-(Iion+Istim)/Cm` term in the same units — so "verify it works under L" is automatically satisfied.

---

### 10. Pacing abstraction  ✅ (mono/bidomain) → extend to LBM

`add_s1s2_protocol` and `add_regular_pacing` already identical in M + B. North-Star wants a
high-level `single` / `s1s2` / `regular(bcl, n_beats)` that **expands** to the stimulus list.
**Proposed canonical:** keep these names as the expansion target; LBM gains them. **Status: 🟡 MINOR.**

---

### 11. Physical parameters  🟡

| Param | M | B | L | CC | Canonical |
|---|---|---|---|---|---|
| time step | `dt=0.02` | `dt=0.02` | `dt` (required) | `dt=0.02` | **`dt`** ✅ (default 0.02) |
| capacitance | `Cm=1.0` | `Cm=1.0` | `Cm=1.0` | `Cm=1.0` | **`Cm`** ✅ |
| surface-to-vol | `chi=1400` (on discretization) | `chi` *(deprecated, absorbed into D)* | only inside `sigma_to_D` | `chi=1400` (mesh meta) | **`chi`** 🟡 (see #13) |
| diffusivity | `D=0.001` (scalar/field) | `BidomainConductivity(D_i,D_e)` | `D` (required scalar) | `D_xx/D_yy/D_xy` fields | see #12 |

**Status: 🟡 MINOR** except conductivity (#12) and chi handling (#13).

---

### 12. Conductivity / diffusivity  ✅ **naming RESOLVED → mixed subscripts** (P1, 2026-06-02); interface 🟡

| Surface | How expressed | Location |
|---|---|---|
| M | scalar `D` (+ optional `D_field=(Dxx,Dxy,Dyy)`); chi·Cm divided **internally** (Formulation A) | `…/discretization_scheme/fdm.py:115,118` |
| B | `BidomainConductivity(D_i=0.00124, D_e=0.00446, …fiber fields)`; D **pre-scaled** = σ/(χ·Cm) (Formulation B) | `…/tissue/conductivity.py:13` |
| L | scalar `D` inline; `sigma_to_D(σ_l,σ_t,angle,chi,Cm)→(Dxx,Dyy,Dxy)` helper | `src/diffusion.py:9` |
| CC | mesh stores `D_xx/D_yy/D_xy` (effective monodomain) + optional `sigma_i/sigma_e` (bidomain) | `file_format.py:53,71` |

**Canonical names (per P1 — keep the subscript; bare `D` is banned):**
- **monodomain / LBM** → **`D_eff`** (effective diffusivity = the i/e collapse `D_i·D_e/(D_i+D_e)`),
  with input **`sigma_eff`** (or just `sigma` for isotropic single-domain). *Not* `D`, *not* `D_i`.
- **bidomain** → **`D_i` / `D_e`** components, with inputs **`sigma_i` / `sigma_e`**.
- anisotropy keeps tensor subscripts: `D_eff_xx/xy/yy`, `D_i_xx/…`, etc.

> Why subscripted (the exemplar of P1): monodomain's diffusivity is genuinely the *effective* one,
> not an intra/extra component — calling it `D_eff` says so; a bare `D` would falsely equate it with
> bidomain's `D_i`. User: "`D_i` is not true for monodomain; `D_eff` is more correct."

**Interface (still 🟡):** one **`ConductivityConfig`** takes `sigma_*` (+ chi, Cm, fiber angle) and
emits `D_eff` (scalar/tensor) for mono/LBM or `(D_i, D_e)` for bidomain — converting **in one place**
so the Formulation-A/B asymmetry never leaks to the user. Direction settled (Phase 3); only the exact
class shape remains. **Status: ✅ naming RESOLVED / 🟡 interface.**

---

### 13. `chi` (surface-to-volume) handling  ✅ **RESOLVED → only inside `ConductivityConfig`** (2026-06-02)

Exposed + live in M (in the χ·Cm mass term); **deprecated/absorbed-into-D** in B; conversion-only
in L; stored as mesh metadata in CC. **Canonical:** χ lives **only inside `ConductivityConfig`**
(the `sigma_* → D_*` conversion), **never a free solver knob** — matches B/L already; monodomain's
exposure is the outlier and gets folded in. **Status: ✅ RESOLVED** (tied to #12).

---

### 14. Grid / geometry  🟡 / ⚙️

| Surface | Construct | Location |
|---|---|---|
| M | `StructuredGrid.create_rectangle(Lx, Ly, Nx, Ny, device, dtype)` (+ `from_mask`; `TriangularMesh` for FEM) | `…/mesh/structured.py:113` |
| B | same `StructuredGrid.create_rectangle(...)` **+ `BoundarySpec`** | `Bidomain/…/mesh/structured.py:119` |
| L | **no grid object** — inline `Nx, Ny, dx` (+ `bounce_masks`) | `src/simulation.py:55` |
| CC | `CardiacMeshData(dx, dy, mask, D_*, …)` file format; helpers in `geometry.py` | `file_format.py:15` |

**Canonical (per P2 — structured primary):** the default geometry is a **structured grid**, expressed
the LBM-simple way — `Nx, Ny, dx[, dy]` (+ optional domain `mask`); `(Nx, Ny)` ij convention is ✅
already universal. `StructuredGrid.create_rectangle(Lx, Ly, Nx, Ny, …)` is the lightweight constructor
for it (LBM gains it; today it builds inline). The **unstructured `TriangularMesh` (FEM) is the explicit
secondary path**, not the default. `BoundarySpec` is ⚙️ engine-specific (bidomain). **Status: 🟡 MINOR.**

---

## TIER 2 — Engine-Specific Concepts (canonical name where applicable)

### Bidomain only ⚙️

| Concept | Canonical | Location |
|---|---|---|
| extracellular potential | **`phi_e`** (✅ B + CC agree) | `Bidomain/…/state.py:54`, `cardiac_core/api.py:128` |
| paired conductivity | `BidomainConductivity(D_i, D_e, …)` | `…/tissue/conductivity.py:13` |
| boundary conditions | `BoundarySpec` — `insulated()` / `bath_coupled()` / `bath_coupled_edges()`; `BCType{NEUMANN,DIRICHLET}`, `Edge{LEFT,RIGHT,TOP,BOTTOM}`, `EdgeBC` | `…/mesh/boundary.py:37,120,125,132` |
| diffusion solver | `diffusion_solver='decoupled'` (+ gauss_seidel/semi_implicit/jacobi/imex_sbdf2/explicit_rkc) | `bidomain.py:36` |
| parabolic linear solver | `parabolic_solver` ∈ {pcg, chebyshev, spectral} | `bidomain.py:38` |
| elliptic linear solver | `elliptic_solver` ∈ {auto, spectral, pcg_spectral, pcg, pcg_gmg} | `bidomain.py:40` |
| implicitness | `theta` (0.5=CN, 1.0=BDF1) | `bidomain.py:42` |
| stencil | `stencil` ∈ {5pt, mehrstellen} | `fdm.py:148` |
| operators | `L_i`, `L_e`, parabolic `A_para`, elliptic `A_ellip` | `fdm.py:222–294` |

### LBM only ⚙️

| Concept | Canonical | Location |
|---|---|---|
| distributions / populations | **`f`** `(Q, Nx, Ny)`; component `f[a]` / `f_i`; post-collision `f_star` | `src/simulation.py:114`, `src/step.py:20` |
| lattice | `lattice` ∈ {`d2q5`, `d2q9`, `d2q9_uniform`}; `Lattice` ABC (`Q, cs2, e, w, opposite`) | `src/lattice/…`, `base.py:8` |
| weights mode | `weights_mode` ∈ {`canonical`, `uniform_8`} | `src/simulation.py:58` |
| collision | `BGK` (`bgk_collide`, single-τ) / `MRT` (`mrt_collide_d2q5/d2q9`, multi-relax) | `src/collision/bgk.py`, `…/mrt/` |
| streaming | `stream_d2q5` / `stream_d2q9` (pull-convention, `torch.roll`) | `src/streaming/` |
| equilibrium | `f_eq = w·Vm` | `src/collision/bgk.py:30` |
| relaxation | `tau` (`tau_from_D`), `omega = 1/tau` | `src/diffusion.py:29`, `simulation.py:102` |
| boundary conditions | Neumann bounce-back / Dirichlet anti-bounce-back / absorbing; `bounce_masks` dict | `src/boundary/{neumann,dirichlet,absorbing}.py` |
| unit conversion | `sigma_to_D`, `tau_from_D`, `tau_tensor_from_D`, `check_stability` | `src/diffusion.py` |

> ⚠️ LBM Dirichlet anti-bounce-back is a current **sink** (slows conduction) — NOT the Kleber boundary speedup. Don't conflate in user-facing docs.

### Monodomain only ⚙️

| Concept | Canonical | Location |
|---|---|---|
| discretization scheme | `FDMDiscretization` / `FEMDiscretization` / `FVMDiscretization` (`SpatialDiscretization` ABC) | `…/discretization_scheme/` |
| diffusion solver | `diffusion_solver` ∈ {crank_nicolson, bdf1, bdf2, forward_euler, rk2, rk4} | `monodomain.py:153` |
| linear solver | `linear_solver` ∈ {pcg, chebyshev, dct, fft, none} | `monodomain.py:116` |
| FDM boundary | `boundary_mode` ∈ {face_mirror, face_mirror_iso, node_mirror_existing, zero_pad, rest_pad} | `fdm.py:100` |
| FDM stencil | `stencil` ∈ {cardinal4, moore8_uniform, moore8_iso} | `fdm.py:110` |
| FEM mesh | `TriangularMesh` | `…/mesh/triangular.py:18` |

### Shared solver vocabulary (mono + bidomain) ✅

| Concept | Canonical | Note |
|---|---|---|
| operator splitting | `splitting` ∈ {`strang`, `godunov`} | M aliases `lie→godunov` |
| ionic stepper | `ionic_solver` ∈ {`rush_larsen`, `forward_euler`} | M aliases `rl/fe`; LBM uses rush_larsen internally (`src/solver/rush_larsen.py`) |

---

## TIER 3 — Internal subsystem naming (rename targets, not user-facing)

| Concept | M | B | L | Proposed canonical |
|---|---|---|---|---|
| ionic stepping module | `ionic_time_stepping` | `ionic_stepping` | `src/solver/` | **`ionic_stepping`** 🟡 |
| diffusion stepping module | `diffusion_time_stepping` | `diffusion_stepping` | n/a | **`diffusion_stepping`** 🟡 |
| spatial module | `discretization_scheme` | `discretization` | n/a | **`discretization`** 🟡 |
| builder package | `tissue_builder` | `tissue_builder` | `src/` | **`tissue_builder`** 🟡 |
| top namespace | `cardiac_sim` | `cardiac_sim` | `src` + `ionic` | **`cardiac_core`** (target) |

> The `cardiac_sim` namespace collision between M and B is the root of CC's `_prepare_engine()`
> `sys.modules` hack (`cardiac_core/api.py:25`) — removed when engines move under `cardiac_core`.

---

## Decisions

### Naming principles (govern all rows)
- **P1** — mixed/subscripted notation for any intra/extra/membrane quantity; bare symbol only if the concept is identical across engines (`Vm` ✅, bare `D` ❌).
- **P2** — structured grid is the primary standard (grid-shaped `(Nx,Ny)`, LBM-simple); complex mesh (FEM, flat `(n_dof,)`) is the explicit secondary path.

### Resolved
| # | Decision | Choice | Migration note |
|---|---|---|---|
| 2 | Transmembrane potential name | ✅ **`Vm`** (P1) | rename M/L `V`→`Vm`; revert CC's shipped `V` + tests; keep `.V` alias then deprecate. Ionic-ABC `V` param left as-is (follow-up). |
| 7 | Voltage output shape/type | ✅ **grid `(Nx,Ny)` torch f64** (P2) | structured default; flat `(n_dof,)` reserved for FEM secondary path; reshape at API boundary (CC already does) |
| 12 | Conductivity **naming** | ✅ **`D_eff` (mono/LBM), `D_i`/`D_e` (bidomain)**; inputs `sigma_*` (P1) | ban bare `D`; mono's is the *effective* diffusivity |
| 13 | `chi` handling | ✅ **only inside `ConductivityConfig`** | never a free solver knob; fold M's exposed χ in |
| 6 | `run()` / result contract | ✅ **eager `run()`; streaming via `batch=k`; only `SimulationResult`** (no `stream()`, no `Snapshot`) | unify M/B/L `run()` → eager; add `batch`/`record` knobs; accepted return-type-varies-with-`batch` wart |
| 5 | State container (**public**) | ✅ **only `SimulationResult` is user-facing** (live State never exposed) | per #6; remove per-engine yield of mutable live state |
| 9 | Default stimulus amplitude | ✅ **−52** (user 2026-06-24) | rename LBM `-80`→`-52`; identical depolarization (byte-identical ionic model) |
| — | CHANGE idiom (re-parameterize for sweeps) | ✅ **functional `sim.with_(**overrides)` → new Simulation** (user 2026-06-24) | immutable, sweep-safe; no mutable setters in the public API |
| — | Construction layering | ✅ **per-model factories + `SimulationSpec`/`create_simulation` layered** (user 2026-06-24) | spec→factory; spec is the LLM-intake bridge (Goal 2) |
| 11/12 | **Canonical formulation** | ✅ **Form B is the target; converge monodomain in Phase 4** (user 2026-06-24) | both forms now physically correct → engineering choice; B confines χ/Cm to `ConductivityConfig` (#13), is non-fragile + already 2/3 engines. `ConductivityConfig.for_monodomain()` asymmetry is temporary, deleted at Phase-4 rewire. See KNOWLEDGE "Canonical formulation = B". |

### Open (🔴 needing sign-off)
| # | Decision | Recommendation | Cost of recommendation |
|---|---|---|---|
| 5 | State container (**internal**) | one unified live `State` dataclass; LBM adopts it | internal refactor; **deferrable** (no public-contract impact). FEM-ditch (below) simplifies it further. |
| 12 | Conductivity **interface** (naming done) | `ConductivityConfig` stores σ-physics, emits `D_eff`/`D_i`/`D_e`; chi=1 fed to mono internally neutralizes Form-A. Classmethods `.isotropic/.bidomain/.anisotropic`. See `API_DESIGN.md` §4. | Phase 3 work; direction + shape now drafted |
| — | **Ditch FEM → structured-grid only** | ✅ **CONFIRMED** (user 2026-06-24) | P2→P2′ "structured is the *only* standard"; drop unstructured/flat path, `TriangularMesh`, monodomain `FEMDiscretization`. **FDM primary; FVM survives** (structured); FVM→FDM collapse is a separate later question. See `API_DESIGN.md` §9 |

---

## Provenance

Harvested 2026-06-02 by four parallel read-only census agents (one per surface). Every identifier
above carries a `file:line`; re-run the census if engines change. Builds on the **Cross-Engine
Capability Census (2026-06-01)** in `KNOWLEDGE.md` — this doc is its line-referenced, decision-tagged expansion.
