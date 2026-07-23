# PLAN.md — "Cardiac Core by Experiment" Jupyter tutorial series

> Design doc for an 11-lesson notebook series, organized as **experiments a lab member would
> actually run**, that teaches a wet-lab scientist (cell-culture / tissue-chip, no computational
> background) how to operate `cardiac_core`. Two tiers: a finishable **Core (01–06)** and an
> **Advanced (07–11)** set. Ships in `cardiac_core/tutorials/` so it travels with the library and
> doubles as the documentation the package still lacks.
>
> **Status: DESIGN CONVERGED 2026-07-22. Lesson 01 SHIPPED 2026-07-23. Remaining lessons GATED.**
>
> **SHIPPED — [`01_build_a_simulation.ipynb`](./01_build_a_simulation.ipynb)** (34 cells, 12 code,
> ~90 s runtime). A single self-contained walkthrough covering the whole loop: build a **grid**, set
> its **conductivity**, place a **`Stim`** (with the mask drawn, full view + zoom), run **monodomain**,
> then re-run the identical setup on **bidomain** (adds `phi_e`) and **LBM**, and measure
> **conduction velocity** on all three. Verified end to end in the project environment:
> every code cell executes (exit 0) and the three CVs come out **58.8 / 59.6 / 64.6 cm/s**. Figures
> were rendered and inspected, not merely "raised no exception" — the wavefront snapshots track the
> measured CV (fronts at 0.17 → 1.62 cm between t=5 and t=30 ms ⇒ ~58 cm/s, matching `r.cv()`), and
> the `phi_e` sign flip at the front was confirmed visually before the prose claimed it.
> Source of truth: [`_build/build_01_build_a_simulation.py`](./_build/build_01_build_a_simulation.py)
> (regenerates the notebook; `--script PATH` emits a flat `.py` of the code cells for headless
> regression). Index: [`README.md`](./README.md).
>
> **⚠ The Core tier must be re-cut before authoring continues.** Lesson 01 was scoped by the user
> (2026-07-23) as "one simple interval" and deliberately front-loads the *mechanics* — grid, stim,
> run, all three engines. That now overlaps designed lessons **03** (cell → monolayer CV), **10**
> (two engines) and part of **11** (bidomain/`phi_e`). Do NOT author 03 or 10 as specced; re-cut the
> Core arc against what 01 already delivers. The § 3 tables below are the pre-01 design and are kept
> for the lesson *content*, not the ordering.
>
> The remaining lessons stay gated until the two in-flight branches land (the `Stim` pipeline and the
> `Video`/`Gradient` pipeline). See § Authoring gate.
>
> **Tooling note (blocks the anti-rot gate):** `nbformat` and `nbconvert` are **not installed** in the
> project environment (`ipykernel`/`jupyter_client`/`jupyter_core`/`matplotlib` are). Lesson 01 was
> therefore emitted as plain JSON and ships with **empty outputs**. Installing them
> (`pip install nbformat nbconvert`) unlocks both executing-and-embedding outputs and the § 8
> execute-all gate. Not done during authoring: two other agents were running in the shared env and a
> dependency upgrade mid-flight was not worth the risk.

---

## 1. Converged decisions (2026-07-22, user)

| # | Decision | Chosen |
|---|----------|--------|
| D1 | Organizing spine | **Lab-experiment ladder** — each lesson is a bench experiment, simulated. API is taught as a side effect of doing recognizable science. (Rejected: API-concept ladder, physics-first ladder.) |
| D2 | Scope | **Two-tier: 6 core + 5 advanced.** Core is completable in an afternoon and delivers a real result; advanced is opt-in. |
| D3 | Numerical caveats | **Minimal — operational only.** Teach the calls and the science. Do NOT decorate lessons with numerics disclaimers. One structural exception, § 5. |

### Supersedes the 2026-07-21 plan
That plan (8 lessons, API-concept order) is replaced. Specifically:
- **P0.2 is DONE, not a prerequisite.** `cc.single_cell()` shipped 2026-07-22 (commit `63f6982`)
  along with `cc.safety_factor` / `cc.threshold_charge`. Real signature:
  `cc.single_cell('ttp06', celltype='EPI', pre_pace=5)` → `sc.V`, `sc.apd(0.9)`, `sc.final_state`
  (NOT the `stim=/t_end=/dt=/cell_type=` signature the old plan guessed).
- **The dict stimulus is dead as teaching material.** `{"region": …, "start_time": …}` now emits a
  `DeprecationWarning` (Stim Phase 2, commit `743e6d4`). Canonical is `cc.Stim.boundary(g, "left")`
  / `.point` / `.center` / `.from_region` / `cc.Stim(mask)`. The old plan's L1→L3 continuity story
  ("Lesson 1 teaches the stim keys that carry into Lesson 3+") is void and replaced by § 4.
- **Voltage clamp is promoted from "v2 bonus" to a lesson** (`Stim(clamp=-20)`, all three engines).
  It is the one technique this audience already knows in their hands.
- **The `r.fields.*` layer earns its own lesson.** It did not exist when the old plan was written.

---

## 2. Audience & teaching contract

- **Reader**: a wet-lab scientist. Can run a notebook cell and edit a number. Does **not** know
  numpy, torch, or OOP. Knows action potentials, conduction, drugs, and scars from the bench.
  Same persona the `/sim-*` skill suite targets — the notebooks are the *learn it* track, the
  skills are the *do it* track. Core lesson 06 ends by pointing at `/sim-experiment`.
- **Every lesson is standalone.** Re-imports, rebuilds its own grid, runs top-to-bottom in a fresh
  kernel. A reader can open lesson 08 cold. Shared concepts get a one-sentence re-cue, never an
  assumption.
- **Rhythm per notebook**: `What you'll learn` → `What you need` → short prose beat → one small code
  cell → figure → "what just happened" → **Try it yourself** (1–2 edit-a-number exercises) →
  **Recap** (3 bullets) → `Where next`.
- **Code style**: one idea per cell; full words, no abbreviations; every non-obvious argument
  commented; physics in prose beside the code.
- **Framing**: lesson titles and prose name the *experiment*, not the API. "Measure conduction
  velocity across your monolayer", not "the `r.cv()` hook".

---

## 3. The series

### Core (01–06) — the afternoon path

| # | Notebook | The experiment | Primary API |
|---|----------|----------------|-------------|
| 01 | `01_one_cell_one_beat.ipynb` | Record a single action potential from one cell; name its phases; read APD90. | `cc.single_cell`, `sc.V`, `sc.apd(0.9)` |
| 02 | `02_drug_on_one_cell.ipynb` | Apply a channel blocker to that cell and watch the AP change. | `sim.scale_conductance("GNa"/"GKr", f)` |
| 03 | `03_cell_to_monolayer.ipynb` | Stimulate one edge of a strip, watch the wave cross it, measure conduction velocity. | `cc.Grid`, `cc.Stim.boundary`, `cc.monodomain`, `r.cv` |
| 04 | `04_watch_it_propagate.ipynb` | Turn that run into a movie and an activation map — the two pictures every talk needs. | Video pipeline (§ 6), `r.lat()`, isochrones |
| 05 | `05_pace_it.ipynb` | Pace at decreasing cycle lengths; watch APD shorten (restitution). | `Stim(bcl=, num_pulses=)`, `r.apd_per_beat`, `r.restitution_slope` |
| 06 | `06_scar_and_block.ipynb` | Carve an inexcitable scar, route a wave around it, narrow the isthmus until it blocks. | `cc.rectangle_mask`/`circle_mask`, `sim.set_conductivity(mask, D=0)` |

**Core exit**: the reader has run a cell, a tissue, a drug, a pacing protocol and a scar, and has a
movie. Lesson 06 closes by handing them `/sim-experiment` for their own work.

### Advanced (07–11) — opt-in

| # | Notebook | The experiment | Primary API |
|---|----------|----------------|-------------|
| 07 | `07_fibers_and_direction.ipynb` | Point-stimulate the centre of aligned tissue; get an elliptical wavefront; measure fast vs slow axis. | `ConductivityConfig.anisotropic(sigma_l, sigma_t, fiber_angle)`, `cc.Stim.center`, `r.radial_cv` |
| 08 | `08_voltage_clamp.ipynb` | Hold a region at a command voltage — the patch clamp they already know, in tissue. | `cc.Stim(mask, clamp=-20, duration=…)` |
| 09 | `09_seeing_the_physics.ipynb` | Make the invisible visible: where the wave sources and sinks current, where it curves, where it is about to fail. | `r.fields.source_sink`, `.curvature`, `.velocity`, `cc.safety_factor` |
| 10 | `10_two_engines.ipynb` | Run the same tissue on the FDM monodomain and on LBM; learn which to reach for. | `cc.monodomain` vs `cc.lbm` |
| 11 | `11_capstone_infarct.ipynb` | Bidomain infarct in a bath: the extracellular signal around a scar — what an electrode outside the tissue would see. | `cc.bidomain(..., boundary="bath")`, scar mask, `r.phi_e` |

Explicitly **out of v1**: reentry/rotor formation (belongs to `geometry_induced_reentry`), ERP
protocols (`cc.erp` runs sims — slow), the `.npz` save/load workflow, GPU.

### ⚠ Verified API corrections (2026-07-23) — the old plan named two signatures that don't exist

1. **Anisotropy takes conductivities, not diffusivities.** The real call is
   `ConductivityConfig.anisotropic(sigma_l, sigma_t, fiber_angle, chi=1400.0, Cm=1.0)` — raw σ in
   **mS/cm**, and `fiber_angle` in **RADIANS** (0 = fibers along +x). The old plan's `anisotropic(D_l,
   D_t, …)` would have taught a units error of exactly χ·Cm. Also: `anisotropic` uses **one global
   angle** — per-node fiber fields are not exposed by the factories, so "curved fibers" is not a
   lesson-07 exercise. Note `cond.sigma_eff`/`cond.D_eff` return a **3-tuple (xx, yy, xy)** in the
   anisotropic case, not a scalar — don't `float()` them in a lesson cell.
2. **Mixed boundary conditions are NOT reachable from the public API — capstone re-scoped.** The
   declarative factory validates `boundary` against exactly `('bath', 'insulated')` (`api.py:1751`)
   and maps `'bath'` to `BoundarySpec.bath_coupled()` — **all four edges** (`api.py:1771`). Per-edge
   coupling (`BoundarySpec.bath_coupled_edges([Edge.TOP, Edge.BOTTOM])`) exists only by importing
   `cardiac_core.mesh.boundary` and hand-assembling a `StructuredGrid`, which is precisely the
   drop-to-internals move the § 2 audience contract forbids. So lesson 11 is now **"infarct in a
   bath"** on `boundary="bath"`, and mixed-BC is **out of v1**.
   **→ This is an API-surface gap, not a tutorial problem**: `bath_coupled_edges` is real,
   test-covered functionality with no public route. Logged to the engine_consolidation IDEALOG as a
   candidate `boundary=` extension (e.g. accepting a list of edges). If it ships, the capstone can
   regain mixed BCs.
3. **Lesson 02 ("drug on one cell") is BLOCKED on a small API addition.** Verified signature:
   `single_cell(model, *, celltype='ENDO', dt, bcl, n_beats, pre_pace, stim_amplitude,
   stim_duration, t0, Cm, save_every, device)` — there is **no conductance knob**.
   `scale_conductance` is a `CardiacSimulation` (tissue) method. The only 0-D route is to build an
   ionic-model instance, mutate a conductance attribute, and pass the instance as `model=` — exactly
   the OOP move § 2 forbids.
   **→ Candidate fix (small, additive): `single_cell(..., conductances={'GKr': 0.5})`**, lowering to
   the same name-validation `scale_conductance` already does. Logged to the IDEALOG.
   **Fallback if it doesn't land**: move the drug lesson after tissue is introduced (it becomes
   lesson 04, using `sim.scale_conductance("GKr", 0.5)`), and Core reads 01 cell → 02 CV → 03 video
   → 04 drug → 05 pacing → 06 scar. Decide at authoring time; do NOT teach the instance hatch.
   Also verified for lesson 01: `celltype` is `'ENDO'` (default) / `'EPI'` / `'MID'` — **`'MID'`, not
   `'M'`** — and 0-D pacing already exists via `bcl=` + `n_beats=` + `pre_pace=`.
4. **The 0-D stimulus keywords do NOT match `Stim`'s.** `single_cell` takes `stim_amplitude`,
   `stim_duration`, `t0`; `Stim` takes `amplitude`, `duration`, `start_time`. Harmless for the § 4
   through-line (lessons 01–02 introduce no stimulus object at all), but worth a consistency pass —
   a reader who graduates from lesson 01 to lesson 03 meets three renamed keywords for one idea.

---

## 4. The stimulus through-line (replaces the old dict spine)

The old plan's bridge from lesson 1 to lesson 3 was "the same stimulus keys". That bridge is gone:
`single_cell()` takes `pre_pace=`, not a spatial stimulus, and the dict form is deprecated. The
new through-line is **the `Stim` object as a named place**:

- **01–02** — no `Stim` at all. One cell, `pre_pace=`. The reader never meets geometry.
- **03** — first `Stim`, introduced as *"where you put the electrode"*: `cc.Stim.boundary(g, "left")`
  reads as one English sentence, which a `lambda x, y: x < 0.05` never did. This is the single
  most important teaching moment in the series and it now costs one line.
- **05** — the *same* `Stim`, plus `bcl=` / `num_pulses=`: a pacing train is the electrode firing
  repeatedly, not a new concept.
- **07** — `cc.Stim.center(g)`: same object, different place.
- **08** — the same object with `clamp=` instead of `amplitude=`: same electrode, holding voltage
  instead of injecting current.

One object, learned once in lesson 03, re-used four ways. Author lessons so this is visible.

---

## 5. Caveat policy (D3 — minimal, operational only)

Lessons state what to type and what it means physiologically. They do **not** carry numerics
disclaimers: no "effective D = D/(χ·Cm)" asides, no under-resolution warnings, no CV ∝ √D
digressions. Those live in `API_CHEATSHEET.md` for whoever goes looking.

**Two things are operational, not caveats, and stay in:**
1. **Lesson 10 is the comparison lesson.** Its entire content is that the two engines return
   different numbers for the same tissue. Stating that difference is the lesson, not a disclaimer.
   Without it the notebook teaches something false.
2. **Masked nodes read back `NaN`** (lesson 06). The reader will see holes in their own plot and
   ask. One sentence, at the point they see it.

Anything else that feels like it needs a warning is a signal to pick different lesson parameters
instead — choose values that behave, rather than explaining why the chosen ones don't.

---

## 6. Dependencies on the two in-flight branches

| Lesson | Depends on | Status |
|--------|-----------|--------|
| 03, 05, 07, 08 | `Stim` canonical + `clamp=` mode | **READY** — Stim Phases 1–2 committed (`c087b8c`, `743e6d4`) |
| 04 | The `Video`/`Gradient` pipeline | **BLOCKED** — blueprint only (`cardiac_core/VIDEO_OBJECT_PLAN.md`), not implemented |
| all | `cc.single_cell`, `r.fields.*`, `cc.safety_factor` | **READY** — shipped `63f6982` |

**`tutorial_helpers.py` — default to not writing one.** The old plan specified a helper module to
hide matplotlib. With `cc.viz` plus the incoming `Video`/`Gradient` API, most of that is now library
surface, and a helper would teach a vocabulary that only exists inside the tutorials. Decide at
authoring time, after the video pipeline lands; the bar is "the library genuinely cannot do this",
and the likely residue is one `plot_ap`-style convenience for lesson 01.

---

## 7. Authoring gate & conventions

**Do not author notebooks until**: (a) the video pipeline has landed (unblocks 04, and settles the
helper question), and (b) the Stim work is merged to `main`. Lesson 04 is the long pole; consider
authoring 01–03 first if the video branch stretches, but **do not** author 04 against
`cc.propagation_video` and then rewrite it.

- **Location**: `cardiac_core/tutorials/NN_slug.ipynb`; index at `cardiac_core/tutorials/README.md`
  with the Core/Advanced split marked, one-line objective and runtime per lesson.
- **Kernel**: the project environment's kernel. First code cell of every notebook is an import sanity check that
  fails loudly with a one-line fix hint.
- **Authoring method**: build `.ipynb` programmatically from a Python source-of-truth per lesson via
  `nbformat` (reviewable, diffable), with a small `_build_notebook.py` scaffold. Confirm `nbformat`
  and `nbconvert` are importable in the env before starting.
- **Figures**: inline; outputs committed (they *are* the shipped teaching artifact). A lesson that
  exports a video follows the project media-path convention →
  `media/cardiac_core_tutorials/videos/{date}/{slug}_NN.ext`; bulk output under `_sim_outputs/`
  stays gitignored.
- **Runtime**: each notebook executes end-to-end in **< ~90 s on CPU**. Small grids, short `t_end`.
  Flag any single cell over 20 s. Lessons 10 and 11 are the risk; keep them small.
- **Determinism**: fixed grids and parameters so the committed outputs are stable.

## 8. Anti-rot gate (not optional)

These notebooks execute real `cardiac_core`, so they rot silently as the library moves — which is
exactly what happened to the plan they replace. Ship an **execute-all gate**: a pytest (or
`run_all_tutorials.sh`) that runs every notebook headless via
`jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=<project-kernel>`
and fails on any error, wired so `/verify` picks it up.

## 9. Execution order (when un-gated)

P0 prep (env check + nbformat scaffold + verify each lesson's exact calls against the live API) →
01 → 02 → … → 11, one commit per lesson (notebook + README row) → wrap: README index, execute-all
gate, difficulty read-through of 01 as a near-beginner, cross-link from `API_CHEATSHEET.md`
("new here? start with `tutorials/`").

## 10. Open questions for authoring time

1. Does lesson 02's drug story land better with `GNa` (excitability loss, visible in the upstroke)
   or `GKr` (APD prolongation — the classic hERG/torsade story the audience knows)? Both are one
   call; pick when writing, possibly both.
2. Lesson 09's payoff figure: `source_sink` at the moment of block is the most striking, but it
   needs lesson 06's geometry. Consider making 09 reuse 06's scar so the reader recognizes it.
3. Whether the Core tier gets its own top-level README entry (a "start here" pointer from the repo
   root / `API_CHEATSHEET.md`), or lives only under `tutorials/`.
