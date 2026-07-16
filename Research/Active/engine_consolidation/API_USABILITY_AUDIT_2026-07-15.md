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

---
---

# ROUND 2 — full-solve-and-run, +30 tasks (2026-07-16)

**Method.** Same premise, harder bar: agents had to **actually solve and RUN each task to completion**
(produce the real number / detection / figure at a scale that achieves the goal), not smoke-test. 30 NEW
tasks (25–54: pharmacology & dose-response, reentry, alternans, fiber architecture, fibrosis, electrograms
/defibrillation, cell-level state, calibration) + a full-scale RE-RUN of the prior 24. 10 parallel agents.
**Caveat:** the box was heavily oversubscribed during the run (load ~34 on 8 cores + a GPU job), so absolute
wall-times below are inflated ~2–4×; the *per-step* cost, the bugs, and the verdicts are real, the exact
seconds are not.

**Headline.** Running to completion **lowered the grade** — it exposed a class of defects that only appear at
real scale: **a shipped GPU bug that crashes all analysis, a fast-solver path that is broken (forcing slow
PCG), a fixed per-step runtime wall, and several silent-wrong analysis bugs.** Two "impossible" verdicts
also *improved* (automaticity, non-hole scar — both via undocumented routes). Net: the API can *express* far
more than round 1 implied (reentry, figure-8, alternans, Wenckebach, electrograms all achieved), but the
**path to a correct, timely result is booby-trapped** — wrong defaults, broken fast paths, and silent
corruption dominate.

## New concrete BUGS (the round-2 core — each is a blueprint target)

| # | Bug | Where | Impact | Found by |
|---|-----|-------|--------|----------|
| B1 | **GPU `device="cuda"` crashes all analysis/viz** — `_result_from` builds `times` on CPU, `Vm` on CUDA; every `analysis.*` dies at `times[first_idx]` (`RuntimeError: indices ... on cpu`) | `api.py:993` (`_result_from`); mirror-check `run.py::_collect` | `.lat/.apd/.cv/isochrones/apd_map_figure/dominant_frequency/phase_map` all crash on the documented GPU path | R3 (root-caused), N4 |
| B2 | **Fast spectral solvers broken via factory** — `linear_solver='fft'`/`'dct'` → `TypeError: FFTSolver.__init__() missing 6 required positional args` | `api.py` monodomain factory ↔ `FFTSolver.__init__` | The fast path is unusable → **everything is stuck on slow PCG** (root cause of the runtime wall) | N3 |
| B3 | **`apd_at`/`restitution_curve` peak-over-remaining bug** — AP peak = `trace[beat_start:].max()` (max over the ENTIRE rest of the trace), so a later taller beat corrupts earlier beats' APD (got APD90=1341 ms) | `analysis.py` (`apd_at`) | Silently wrong multi-beat APD → corrupts every restitution/alternans measurement | N3 |
| B4 | **`apd_at` notch artifact** — first-crossing-after-peak catches the TTP06 spike-and-dome NOTCH for low repol fractions → APD30=7.4 ms | `analysis.py` (`apd_at`) | Silently wrong low-repol APD (APD30/APD50) | N5 |
| B5 | **`cc.Grid(N, 1, dx)` crashes** — `StructuredGrid.__post_init__` does `dy = Ly/(Ny-1)` → `ZeroDivisionError` at Ny=1 | `mesh/structured.py` `__post_init__` (+ `from_mask`) | A true 1-D cable can't be built the documented way (Ny=2 workaround) | N6 |
| B6 | **`forward_euler` silently blows up** when `dt > dx²/4D` — produces oscillating garbage (threshold crossed 3693×), not an error | `_monodomain` explicit stepper | Silent numerical garbage; no stability guard/warning | N3 |
| B7 | **`record=` silently ignores unknown keys** — `record=("Vm","I_Kr")` runs, produces no attribute, no error (`want_ionic = "ionic_states" in record`) | `api.py` run record handling | Scientist thinks they recorded a current and got nothing | N6 |
| B8 | **Masked-out nodes returned at 0.0 mV** (> −20 mV threshold) → `lat()`/`apd()`/`cv()` count dead tissue as "activated at t=0" | `_monodomain` `flat_to_grid` fill + `analysis.*` | **Quantified silent-wrong: 23% CV error; 100% of masked nodes falsely activated.** Corrupts every scar/fibrosis/irregular-geometry study | N4, R1, R2, N2 (universal) |
| B9 | **Dead `stim_amplitudes_e`** — bidomain extracellular-stim amplitude is hardcoded 0.0 and never read (elliptic RHS has no `I_e` term); forcing it → bit-identical output | `bidomain.py:263`, `decoupled_jacobi.py:95` | No defibrillation / applied-field / virtual-electrode studies possible | N5 |
| B10 | **`dominant_frequency` silent FFT-bin quantization** — DF snaps to `1/T`-spaced bins with no warning about frequency resolution | `analysis.py` (`dominant_frequency`) | Silently coarse DF unless the recording is long | R3 |
| B11 | **`save_every` quantizes derived metrics** — coarse save → identical CV for nearby params → **secant div-by-zero** in a fit loop; also collapses dLAT/crescent metrics | `analysis` CV/LAT + `run` cadence | Closed-loop fitting and fine front-shape metrics silently break | N6, R4, R1 |
| B12 | **`TTP06Model()` defaults to CUDA** → device mismatch against the CPU engine unless `device='cpu'` passed | `ionic/ttp06/model.py` ctor default | Undocumented device trap on the model-instance route | N1 |
| B13 | **`restitution` raw slope → spurious `inf`** — near-duplicate DIs give a divide-by-zero max-slope; a fit is required | `analysis` restitution | Instability metric unreliable without smoothing | N3 |

## Performance findings (the runtime wall)
- **Fixed per-step overhead, ~grid-independent:** ~1.5–3 ms/step CPU and **~13 ms/step GPU (kernel-launch-bound
  by TTP06's 18-state kernels)** — so **wall-clock ∝ simulated time, not grid size**. Long-horizon protocols
  (restitution, reentry, alternans, ATP) run **5–11 min each**; the default `crank_nicolson`+`pcg` makes even
  trivial runs time out. GPU gives **no speedup** on small grids (often slower) and is broken for analysis anyway (B1).
- **Root cause = B2** (the fast spectral path is broken) plus **no LUT/fused ionic kernel**. The only escape is
  undocumented: `diffusion_solver='forward_euler', linear_solver='none'` on GPU, and raising `dt` to 0.04–0.15
  (CN is unconditionally stable; APD90 converged within ~1–3 ms). **Nothing in the cheatsheet says any of this.**
- **Bidomain `bath` perf cliff:** ~35 s/ms vs ~6.7 s/ms insulated (bath isn't spectral-eligible → PCG). The
  cheatsheet's own 201-node strip example would take **hours** in bidomain+bath, yet it only quotes monodomain timing.

## Verdict flips from full running
- **T3 single-cell automaticity: No → Yes** — the undocumented `paci` (hiPSC, funny current I_f) beats spontaneously
  with `stimulus=None` (spontaneous AP ~600 ms, DD slope +0.0136 mV/ms); runs on both monodomain and LBM.
- **T6 non-hole scar: No → Possible** — zeroing per-node `D` in a scar disc (`mesh=CardiacMeshData`) gives a true
  inexcitable scar (nodes hold −85 mV, wave routes around) and is *cleaner* than a mask hole (no 0.0-mV pollution).
- **T8 isthmus block: Yes → No via geometry** — every isthmus width down to 2 nodes conducts; a real block needs
  finer dx or the unimplemented `set_conductivity`.
- **T10 BCL restitution / T15 isochrones-video** degrade at scale — T10 to a runtime-wall/`dt`-workaround task, T15
  from 5/5 to broken-on-GPU (B1).
- **Reentry (30–32,34) ACHIEVED** with the solver workarounds: anchored rotor CL=296 ms (~2.4 rotations), figure-8
  CL≈344 ms, ring min-circumference ≈ wavelength 2.82 cm. **T35 spiral breakup = No** (no rotor-seeding/initial-state
  API + wavelength > domain). **T33 ATP = Partial** (open-loop only — no mid-run stimulus / no rotor-phase readout;
  ~11 min/run).

## Capability gaps confirmed by full running
- **No mid-run state control** — `set_voltage`/`set_state`/`get_state`/`state_names`/`scale_conductance`/
  `set_parameter`/`set_conductivity`/`scale_conductivity`/`clamp_voltage`/`add_clamp_protocol`/`add_pacing`/
  `inject_current` **all `raise NotImplementedError`** with polished docstrings. Blocks: drug/conductance knobs,
  voltage clamp, scar, rotor phase-IC seeding, closed-loop ATP. **The dominant hazard: the object advertises a
  large capability surface in `dir()` that is pure stubs.**
- **No per-beat analysis** — `r.cv/apd/lat` are first-beat/first-activation only → every multi-beat task
  (restitution, alternans, Wenckebach, use-dependence, CV-restitution) is manual upstroke detection.
- **`phase_singularities`** returns a per-frame topological-charge field (not tip coords), with **no boundary/obstacle
  rejection** (spurious edge/obstacle tips) and **no tip tracking**; `phase_map` recomputes the full-trace Hilbert
  on every per-frame call (**O(T²·N²)**). No rotor-seeding helper.
- **`anisotropic()` is scalar-only** (no per-node fiber-angle field; the exported `fiber_field_transmural` is
  orphaned — nothing consumes it); **cell type is global-only** and `MID` is the enum `M_CELL` (crashes on `'MID'`);
  **`ConductivityConfig` is uniform-only** (spatial D only via undocumented `CardiacMeshData.D_xx`).
- **No electrogram/pseudo-ECG helper; no applied field; no sweep/fit helper; no disk checkpoint/resume** (in-process
  continue works bitwise; state-serialize impossible). `record=("Vm","ionic_states")` works but is undocumented and
  returns an **unlabeled** `(T,18,Nx,Ny)` tensor (`state_names` is a stub → source-dive for channel names).

## New-tasks appendix (25–54)

| # | Task | Possible | Ease | Headline (achieved result / blocker) |
|---|------|----------|------|--------------------------------------|
| 25 | I_Kr dose-response | Yes | 2 | APD90 224→274 ms across 0–90% block; drug knob = undocumented `.params.GKr` on a model instance (stub `scale_conductance`); slow default solver timed out at 180 s |
| 26 | Multi-channel drug | Yes | 2 | APD 224→201, CV −12.5%; `PCa` not "GCaL" |
| 27 | Use-dependence | Partial | 2 | Static block only; rate-dependent CV/failure observable; no beat-k CV helper |
| 28 | Rheobase | Yes | 4 | ≈ −26.5 µA/µF; capture all-or-none (~110 mV gap) — cleanest task |
| 29 | Voltage clamp | **No** | 1 | `clamp_voltage`/`add_clamp_protocol` are stubs; no I–V path |
| 30 | Cross-field spiral (period+tip) | Yes (anchored) | 2 | Rotor T=296 ms, ~2.4 rotations; free spiral drifts out; no phase-IC seeding |
| 31 | Anchored spiral CL | Yes | 3 | CL=296 ms, sustained; masked-node trap |
| 32 | Figure-of-8 | Yes | 3 | Dual-loop CL≈344 ms; `phase_singularities` under-counts anchored tips |
| 33 | ATP termination | Partial | 2 | Open-loop only (no mid-run stim / no rotor phase); ~11 min/run |
| 34 | Ring min circumference | Partial | 2 | min circ ≈ λ = 2.82 cm; period=path/CV verified |
| 35 | Spiral breakup | **No** | 2 | 6 seeding attempts → 0 sustained tips; no rotor-induction API; λ>domain (compute NOT blocker) |
| 36 | Alternans onset | Yes (weak) | 2 | Long-short ~8.6 ms at BCL≈275; `apd_at` peak bug (B3) forced manual APD |
| 37 | Discordant alternans | Partial | 2 | Got concordant ~10 ms; discordant needs steep restitution (slope 0.59) |
| 38 | Restitution slope | Yes | 3 | max slope 0.59 <1; raw slope → spurious `inf` (B13) |
| 39 | Wenckebach / conduction failure | Yes | 3 | Clean 2:1 block, dropped beat, delay 57→77.5 ms; no per-beat capture helper |
| 40 | Transmural fiber-angle field | Yes (undoc.) | 2 | LAT range 4 ms; `anisotropic()` scalar-only; per-node D hand-rolled; 51×51 full-tensor times out |
| 41 | Along vs cross APD | Yes | 4 | 228 vs 226 ms (APD dir-independent; anisotropy is in LAT) |
| 42 | CV ellipse | Yes | 3 | CV 48.4→16.8, ratio 2.88; `r.cv` x-row only → hand-roll + ×1000 |
| 43 | Diffuse random fibrosis | Yes | 2 | CV 50→28.2 (44% slow, not blocked); **B8 quantified: naive CV 34.6 vs 28.2 (23% error)** |
| 44 | Percolation threshold | Yes | 3 | Critical density ≈0.35; many slow serial runs; GPU broken |
| 45 | Pseudo-ECG / electrogram | Yes (manual) | 3 | Biphasic ptp=17.8 mV; no helper; bath grounds boundary node (trap) |
| 46 | Defibrillation / applied field | **No** | 1 | Transmembrane-only; `stim_amplitudes_e` dead (B9) |
| 47 | phi_e map at wavefront | Yes | 4 | −7.7…+11.3 mV source-sink dipole |
| 48 | Bath vs insulated phi_e | Yes | 4 | Insulated ptp 21.3 vs bath ≈0 (~17×); must demean insulated (floating ref) |
| 49 | AP morphology | Yes | 3 | dV/dt_max=358.7 V/s, APD30/50/90=7.4/155/214 (APD30 = notch artifact B4) |
| 50 | Ca transient / ionic state | Yes | 2 | Cai 125.8→165.7 nM; `record=` undocumented; state channels unlabeled (`state_names` stub) |
| 51 | Checkpoint & resume | Partial | 2 | In-process continue bitwise-exact; no disk state save (stubs) |
| 52 | Single ionic current trace | **No** | 1 | No per-current output; `record=` silently ignores unknown keys (B7) |
| 53 | Fit D to target CV | Yes | 3 | σ=0.816→D=5.83e-4, 3 iters; save_every→secant div-by-zero (B11); no fitter |
| 54 | 1-D cable | Partial | 2 | Ny=2 CV=59.26; `Grid(N,1)` crashes (B5) |

## Re-run deltas (1–24): what changed at full scale
Verdicts mostly held; the **new content is runtime + silent-wrong**, not new features. Flips: **T3 No→Yes**,
**T6 No→Possible**, **T8 Yes→No-via-geometry**, **T15 5/5→broken-on-GPU**, **T10 → runtime wall**. Deepened:
single-cell APD is clean (not "garbage") once the stim is hand-lowered (T2); the LBM CV offset is **+47%**
(bigger, grid-dependent) not +12 (T24); `GCaL` is a phantom in **both** ionic models (T4); masked-node pollution
is **total** (T6/T7). New universal difficulty: the per-step runtime wall + the CN-`dt` escape hatch.

---

# MERGED prioritized fix list (rounds 1+2) — authoritative for the blueprint

Ordered by (silent-wrong or crash) × (breadth) ÷ effort. **P0 = correctness/crash, mostly cheap; P4 = larger.**

**P0 — bugs that crash or silently corrupt (do first):**
1. **B1 GPU device-mismatch** — set `times` device from `Vm` in `_result_from` (`api.py:993`) and audit `_collect`. One-liner; unbreaks the whole GPU analysis/viz surface.
2. **B8 masked-node pollution** — NaN-fill out-of-domain nodes in the returned `Vm` (or expose `r.mask` and auto-apply in `lat/apd/cv`). Fixes every scar/fibrosis/irregular study (23% silent error today).
3. **B3+B4 `apd_at` peak/notch bug** — window the peak to the current beat and offer a last-crossing / dome-aware repol option. Fixes all multi-beat APD/restitution.
4. **B5 `Grid(N,1)`** — guard `__post_init__`: `dy = Ly/(Ny-1) if Ny>1 else dx`.
5. **B7 `record=` validation** — raise on unknown keys.
6. **B6 explicit-solver stability guard** — warn/raise when `dt > dx²/4D` for `forward_euler`.
7. **B2 fast spectral solver wiring** — fix the `fft`/`dct` factory construction (missing ctor args) so the fast path works (also the biggest perf lever).

**P1 — kill the stub trap + the cheatsheet/`dt` documentation gap (highest ergonomics ROI):**
8. Implement the highest-value stubs (`scale_conductance`, `set_conductivity`, `set_voltage`/`get_state`) — the engine already supports scaled instances + a `D_field`; OR remove the misleading docstrings and mark them unavailable. Wire or delete dead `stim_amplitudes_e` (B9) and the phantom `state_names`.
9. Cheatsheet correctness + **solver/`dt` guidance** pass: document `diffusion_solver`/`linear_solver`/`dt` choices and the CN-stability escape (the #1 usability lever); the drug-block model-instance route + `PCa`/`GKr` name map; `paci`/hiPSC models; ORd-is-LBM-only; `record=`, `save_result`/`load_result` (+ tuple order), `dominant_frequency`, `phase_map` (+ two-step call); per-node-D recipe; `fiber_angle`=radians; bidomain-bath cost; LBM CV offset; `TTP06Model` CPU device (B12).

**P2 — analysis aggregates / per-beat / axes:**
10. `dominant_frequency_map`; `cv(along=)`/`radial_cv`/`cv_between`; per-beat family (`cv_beat`/`apd_per_beat`/`capture`/`cv_restitution`); `restitution_slope` (fit + DI\*); distinguish block/no-activation from `nan`; warn on out-of-band `.isotropic` D and on zero-node stimulus; DF resolution warning (B10); save_every→CV quantization guard (B11).

**P3 — capability the tasks needed:**
11. Per-node conductivity field + per-node cell-type field + per-node fiber-angle (wire `fiber_field_transmural`); register `MID`. Single-cell 0-D path (`cc.single_cell`). `cc.sweep`/`cc.fit_conductivity`. Rotor tooling: batched `phase_map`, obstacle/boundary-aware `track_singularities`, mid-run `add_stimulus`/phase-IC seeding. `cc.electrogram`/`pseudo_ecg`. Disk checkpoint (`sim.save`/`cc.load`).

**P4 — performance (larger):**
12. LUT/fused ionic kernel to cut the ~13 ms/step launch overhead (the root reentry blocker); make bidomain-bath elliptic solve faster or warn. (B2 in P0 already removes the biggest single perf wall.)

*Round-2 method scripts: scratchpad `usab2_{N1..N6,R1..R4}_*.py` (not committed); figures under `media/_unmapped/images/2026-07-16/`. No repo source modified by the audit.*
