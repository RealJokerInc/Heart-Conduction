# cardiac_core API — Task-Based Usability Audit (2026-07-15)

**Method.** 24 realistic scientist tasks (7 categories), each executed by an independent agent that
WROTE and RAN the minimal `cardiac_core` script a scientist would write from `API_CHEATSHEET.md`, then
rated **Possible?** (Yes / Partial / No) and **Ease** (1–5) with the real errors/results as evidence.
6 parallel agents, empirical (not from-memory). Scope: public API ergonomics, not physics accuracy.

**Verdict.** The API is broadly **"possible but painful"** — mean ease ≈ **2.7/5**. It is strongest at
*expressing* a simulation (geometry, stimulus, running) and weakest at *parameterizing* it (conductances,
heterogeneity) and *measuring* it (aggregate/multi-beat/off-axis analysis).

- **2 tasks are effectively IMPOSSIBLE** via the public API: a transmural endo→mid→epi cell-type gradient
  (T18) and a clean single-cell automaticity run (T3).
- **~12 tasks are Partial** — they work only via undocumented routes, private internals, or manual assembly.
- **~10 are cleanly Yes.**
- Ease histogram: **1/5** ×3 (T1, T3, T18) · **2/5** ×7 (T2, T4, T5, T13, T14, T16, T17) · **3/5** ×8
  (T6, T7, T8, T10, T11, T12, T23, T24) · **4/5** ×5 (T9, T19, T20, T21, T22) · **5/5** ×1 (T15).

The **motivating category — ionic tuning & pharmacology (T1–T4) — is the worst-served** (mean 1.5/5),
which is notable because it is the stated audience workflow.

---

## Ranked themes (severity × breadth)

### T1 · CRITICAL — The parameter/heterogeneity layer is documented-but-unimplemented (`NotImplementedError` stubs)
`CardiacSimulation.scale_conductance`, `set_parameter`, `set_conductivity`, `scale_conductivity`,
`clamp_voltage` all `raise NotImplementedError` **while shipping full, inviting docstrings with worked
scar/drug examples** (e.g. `sim.set_conductivity(scar_mask, D=0.0)  # scar`). A scientist — or an LLM —
reading source or IDE autocomplete confidently writes code that dies at runtime.
**Breadth:** blocks the "proper" version of **7 tasks** — T1/T2/T4 (drug & conductance tuning), T6/T8
(scar / graded isthmus), T17 (fibrosis). This is the single biggest hallucination trap in the API.
**Fix (highest ROI):** implement the two highest-value stubs — `scale_conductance` (the engine already
accepts a scaled `IonicModel` instance) and `set_conductivity` (the monodomain factory already has a
non-uniform `D_field` branch). If not implementing now, **delete the misleading docstrings and make the
methods advertise their real status**, and document the working routes.

### T2 · HIGH — No conductance knob on the documented surface; the only working route is undocumented and inconsistent
There is no way to scale a conductance from the cheatsheet's string API (`monodomain(g,"ord",...)`).
The real path is to pass a **pre-built model instance** where the API asks for a string
(`monodomain(g, ORdModel(params_override={"GNa":...}), ...)`), discoverable only by reading `api.py`
(the "pre-built (e.g. tuner-scaled) IonicModel" comment) + `ionic/*/model.py`. Worse, the interface is
**inconsistent across models**: `ORdModel(params_override={...})` works, but `TTP06Model(params_override=...)`
raises `TypeError` — for TTP06 you must mutate `model.params.GKr` after construction.
**Breadth:** all of category A. **Fix:** ship `scale_conductance`; unify the override interface across
models; document the instance route; add a `GCaL`→`PCa` alias (see T3).

### T3 · HIGH — Cheatsheet errors & omissions cause wrong-engine, wrong-knob, and under-discovery
Concrete, verified defects in `API_CHEATSHEET.md`:
- **ORd is listed as a monodomain model but DIES on monodomain** (`NotImplementedError` mid-run) — it runs
  on **LBM only**. The cheatsheet's "monodomain by default" advice picks the one engine ORd can't use. (T1)
- **`paci`/`phas13`/`mhas13` (the hiPSC models) are omitted** from §4 despite being in the registry. (T3)
- **`save_result`/`load_result`, `dominant_frequency`, `phase_map`, `create_cardiac_mesh`/`CardiacMeshData`
  per-node-D** all work but are **absent from the cheatsheet** → scientists hand-roll `np.savez`,
  per-node loops, etc. (T23, T16, T14, T17)
- **`phase_singularities` is mis-primed:** shown as `cc.phase_singularities(...)` alongside `(Vm,times)`
  functions, but its real signature is `phase_singularities(phase)` and needs `phase_map(V,times,t_idx)`
  first → the natural guess `phase_singularities(Vm, times)` raises `TypeError`. (T12, T14)
- **`restitution` is shown scalar** but returns a `(DI, APD)` tuple of tensors. (T10)
- Undocumented: `fiber_angle` units (radians), the CV∝√σ rule, ffmpeg-optional gif-fallback, **which
  conductivity config each engine accepts**, and the **known LBM CV +~30–35% offset**. (T5, T15, T20/T24, T24)
**Fix:** a documentation pass — cheapest high-leverage lift after T1.

### T4 · HIGH — Analysis is single-point / x-axis-centric, with no aggregate or multi-beat forms
The measurement surface is the inverse of the well-rounded `apd_map`/`activation_time`/`lat()` family:
- **`r.cv` measures along an x-row only** (`cv(x1,x2,y)`) — no axis/direction arg, no radial, no curvature,
  no edge helper. Transverse CV (T5), radial CV (T11), and edge-vs-interior CV (T21) are all hand-rolled
  from `r.lat()`, incl. a cm/ms→cm/s ×1000 units trap.
- **`dominant_frequency` is single-node** — no map version; T16 is a manual `Nx×Ny` double-loop.
- **`phase_singularities` is a per-frame charge map** — no tip coordinates, no cross-frame tracking (T14).
- **CV has no per-beat/restitution form** (restitution is APD-only); per-beat CV needs fragile manual
  windowing that silently returns `nan` (T13).
**Fix:** add the aggregate/axis forms — `dominant_frequency_map`, `cv(..., along=)` / `radial_cv`,
`track_singularities`, `cv_restitution` (mirror the existing APD `restitution_curve`).

### T5 · MEDIUM — Silent failures instead of signals
The API frequently returns a benign-looking value where it should warn or raise:
- **`r.cv` returns `nan`** indistinguishably for block, a mid-repolarization window, or no-activation. (T13, T19)
- **The σ/D trap is silent:** `ConductivityConfig.isotropic(0.001)` → effective D=7e-7 → conduction block
  → `nan` CV, **no warning** — even though `create_cardiac_mesh` emits an out-of-band warning on the same
  arithmetic. (T19)
- **A stimulus region that covers no tissue is a silent no-op** — the wave never starts, no warning. Easy
  to hit on any inset/irregular mask with a `region: x<small` reflex. (T7)
- **Masked-out nodes are returned at 0.0 mV** (above the −20 mV activation threshold) in the full-grid
  `Vm`, so `lat()`/`cv()`/`apd()` silently count dead filler as "activated" on every holed/irregular
  geometry. (T6, T7, T8) **NaN-filling masked-out `Vm` fixes all three at once.**
- **Single-cell stim overdrive:** the cheatsheet's tissue stimulus (−80) over-depolarizes an isolated cell
  (no diffusive sink) → `Vmax=109 mV, APD90=8 ms` garbage that `r.apd()` reports without complaint. (T2)
- **Pacing:** `BCL < APD` degenerates silently (fewer restitution points), and the last beat's APD is
  truncated unless `t_end` includes a ~500 ms repolarization tail. (T10)
**Fix:** warn/raise on zero-node stimulus; NaN-fill masked nodes; emit the band-warning on the `.isotropic`
path too; distinguish "block/no-activation" from `nan`.

### T6 · MEDIUM — No single-cell (0-D) mode
Every factory requires a `Grid`; "one cell" is only reachable as a tiny-grid space-clamp hack, which is
**slow** (ORd on 3×3, 400 ms ≈ 160 s — cost is per-timestep, not per-node) and **unphysical** without a
hand-tuned stimulus. Blocks/degrades T2, T3, T4 (all of pharmacology, which is intrinsically single-cell).
**Fix:** a first-class `cc.single_cell(model, ...)` 0-D path — removes the grid, the space-clamp, the
stim-overdrive trap, and the runtime cost in one move.

### T7 · MEDIUM — No batch/sweep/fitting/comparison helpers
Every sweep (T4, T13, T19, T23) and the entire tuning workflow (T1) is a manual loop + manual plot; the
actual fitting machinery lives in the **private `Optimizer/V1/tuner/`** and is not importable as a
documented helper. No `sweep()`, no scalar-vs-parameter curve helper, no cross-engine comparison-figure
helper (T24 hand-rolls matplotlib).
**Fix:** `cc.sweep(param, values, metric=...)` returning arrays; expose or document a tuning entry point;
a cross-engine comparison-figure helper.

### T8 · MEDIUM — Uniform-only conductivity & global-only cell type
`ConductivityConfig` is **uniform-only** (every ctor does `float(sigma)`; an `(Nx,Ny)` array raises). The
only spatially-varying route is an undocumented per-node `CardiacMeshData.D_xx`/`D_yy` array. Cell type is
**global-only** (the factory uses `group_cell_types[0]`; there is no per-node cell-type field) and **`MID`
crashes** (`AttributeError`) — so a transmural gradient (T18) is structurally impossible, and graded
scar/fibrosis (T6, T8, T17) needs the unimplemented `set_conductivity`.
**Fix:** per-node conductivity field and per-node cell-type field in the public API; register the MID variant.

---

## Bright spots (keep these; they are the model for the rest)
- **Viz (`activation_isochrones`, `propagation_video`, `apd_map_figure`) — 5/5.** Take a `SimulationResult`
  directly, return the output path, save to convention-compliant `media/`. This is how the analysis surface
  *should* feel. (Only nit: silent gif-fallback when ffmpeg is absent, routed to `images/` not `videos/`.)
- **LBM d2q9 boundary error messages — "the strongest part of the current surface."** Requesting `scs`/
  `combined` on d2q5 raises a clear, actionable error; the lattice-aware default is documented. (Validates
  the F2 change landed the day of this audit.)
- **Stimulus expression** — the list-of-dicts + `region` callable model handles every protocol tested
  (S1–S2, cross-field, point, multi-beat pacing) without reaching into internals.
- **Masks** (`circle_mask`/`annulus_mask`/`rectangle_mask`) — one-call geometry, incl. disc-with-hole.
- **Bidomain `boundary="bath"`** — works and is documented; reproduces the Kleber edge speedup.
- **Real validation** — an unknown ORd parameter raises loudly (`ValueError: Unknown parameter: GCaL`).

---

## Prioritized fix list (biggest ease-lift per effort)
1. **Kill the stub trap (T1).** Implement `scale_conductance` + `set_conductivity` (both have engine
   support already), OR remove the misleading docstrings and document the real routes. Unblocks 7 tasks.
2. **Cheatsheet correctness pass (T3).** Fix the ORd-on-monodomain error; add hiPSC models; document
   save/load (+ return-tuple order), `dominant_frequency`, `phase_map` (+ the two-step phase call),
   the per-node-D recipe, the `restitution` tuple, `fiber_angle` radians, per-engine config acceptance,
   and the LBM CV offset. Cheap, high discoverability payoff.
3. **NaN-fill masked-out `Vm` (T5).** One change removes silent analysis pollution on all irregular geometry.
4. **Analysis aggregates/axes (T4).** `dominant_frequency_map`, `cv(along=)`/`radial_cv`,
   `track_singularities`, `cv_restitution`.
5. **Single-cell 0-D mode + `cc.sweep` (T6, T7).** Directly serves the pharmacology audience.
6. **Per-node conductivity + cell-type fields; register MID (T8).**
7. **Warn on zero-node stimulus; band-warn on `.isotropic` out-of-band D (T5).**

---

## Appendix — all 24 tasks

| # | Task | Possible | Ease | Headline friction |
|---|------|----------|------|-------------------|
| 1 | Tune ORd G_Na/G_CaL/G_Kr → CV & APD targets | Partial | 1 | No conductance knob (`scale_conductance` stub); real route = inject model instance (undocumented); **ORd dies on monodomain (cheatsheet wrong) — LBM only**; `GCaL` phantom (=`PCa`); no fitting surface |
| 2 | 50% I_Kr block, APD90 diff | Partial | 2 | Instance injection; **TTP06 lacks `params_override` (mutate `.params`) while ORd has it**; single-cell APD is silent garbage unless stim hand-lowered |
| 3 | Single-cell automaticity (I_f/I_K1) | **No** | 1 | **No 0-D API**; tiny-grid proxy 160 s & unphysical; ORd has no I_f & won't run on mono; `paci` undocumented; no automaticity detector |
| 4 | G_CaL sweep (6 pts) + plot | Partial | 2 | No sweep/curve helper; **`GCaL` phantom (=`PCa`)**; inherits single-cell-stim trap |
| 5 | Anisotropic strip, both CVs | Partial | 2 | `r.cv` x-only → transverse hand-rolled from `lat()` incl. cm/ms→cm/s ×1000; `fiber_angle` units undocumented; CV∝√σ undocumented |
| 6 | Disc + central scar | Yes (hole) / No (non-hole) | 3 | `annulus_mask` easy; real inexcitable scar → `set_conductivity` `NotImplementedError`; masked nodes 0.0 mV pollute analysis |
| 7 | Arbitrary microscopy mask | Yes | 3 | **Stimulus covering no tissue = silent no-op**; masked-out 0.0 mV counts as "activated" |
| 8 | Isthmus / conduction block | Yes | 3 | No block helper (roll from `lat()`); masked PCG slow; graded block needs unimplemented `set_conductivity` |
| 9 | S1–S2 focal | Yes | 4 | Not first-class (hand-built list works); `lat()/apd()` see first activation only → can't verify S2 capture without raw Vm |
| 10 | BCL pace + restitution | Yes | 3 | `bcl`/`num_pulses` work; `restitution` returns undocumented `(DI,APD)` tuple; `t_end` needs a repol tail; `BCL<APD` degenerates silently |
| 11 | Point stim, radial CV | Yes / Partial | 3 | Point stim trivial; `r.cv` x-row only — no radial/off-axis/curvature helper |
| 12 | Cross-field rotor | Yes / Partial | 3 | Cross-field natural; `phase_singularities(Vm,times)` **TypeError** — needs `phase_map` first; cheatsheet mis-primes |
| 13 | CV restitution | Partial | 2 | No CV-restitution helper; `r.cv` first-beat-only → per-beat CV needs fragile windowing (returns `nan`) |
| 14 | Spiral-tip tracking | Partial | 2 | Per-frame only; charge map not coords; no tracking; spurious tips on non-spiral data; `phase_map` undocumented |
| 15 | Isochrones + video | **Yes** | **5** | Clean — both take `SimulationResult`; only nit is silent gif fallback (no ffmpeg) into `images/` |
| 16 | Dominant-frequency map | Partial | 2 | `dominant_frequency` single-node, no map (manual double-loop); not in cheatsheet |
| 17 | Fibrotic patch (spatial D) | Partial | 2 | `ConductivityConfig` uniform-only; **4 documented per-region methods are `NotImplementedError`**; working path = undocumented per-node `CardiacMeshData.D_xx` |
| 18 | Transmural cell-type gradient | **No** | 1 | No per-node cell type (global `group_cell_types[0]`); **`MID` crashes**; no `cell_type` kwarg; ENDO≈EPI APD |
| 19 | σ sweep → CV vs D | Yes | 4 | σ-vs-D trap reproduces (σ=0.001 → block → **silent `nan`**); `.isotropic` doesn't emit the band-warning `create_cardiac_mesh` does |
| 20 | Mono vs bidomain | Yes | 4 | Must use `.bidomain()` config (`.isotropic` raises on bidomain); bidomain ~30–60× slower; bath loading only ~5% |
| 21 | Bidomain bath edge vs interior | Yes | 4 | Works & documented; no edge-CV helper (hand-pick y=0); coarse-grid CV quantizes |
| 22 | LBM d2q9 specular (β) | Yes | 4 | d2q9 requirement + error excellent; `alpha` blend unresolvable on a tiny grid; no curvature helper |
| 23 | 2D sweep + save/reload | Yes | 3 | **`save_result`/`load_result` absent from cheatsheet**; no `r.save()`; undocumented return-tuple order; no sweep helper |
| 24 | Cross-engine CV + figure | Yes* | 3 | Bidomain rejects `.isotropic` config (trap); no comparison-figure helper; **LBM +~12 cm/s offset unwarned** |

*Backlinks:* research question [engine_consolidation](./README.md) · project [MASTER](../../../MASTER.md). Method
scripts in the session scratchpad (`usab_{A..F}_*.py`), not committed. No repo source was modified by the audit.
