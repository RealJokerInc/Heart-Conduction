# PLAN.md — "Intro to Cardiac Core" Jupyter tutorial series

> Cold-start execution plan for an 8-lesson notebook series that teaches a reader with
> **minimal Python** but **some cardiac/neuro exposure** how to operate `cardiac_core`.
> Group = one independent, self-contained lesson per notebook. Start simple (import +
> single cell) → end at the capstone (bidomain infarct with mixed boundary conditions).
> Deliverable lives in `cardiac_core/tutorials/` so it ships *with* the library (doubles as
> the "proper documentation" the library is missing).

## Audience & teaching contract
- **Reader**: can run a notebook cell and edit a number; does NOT know numpy/torch/OOP.
  Knows roughly what an action potential, conduction, and a scar are.
- **Every lesson is standalone**: re-imports, re-defines its mesh, runs top-to-bottom in a
  fresh kernel. No lesson depends on a variable from another. (A reader can open Lesson 5
  cold.) Shared *concepts* are re-cued in one sentence, not assumed.
- **Rhythm per notebook**: `What you'll learn` → `What you need first` (prereqs/install) →
  short prose beat → tiny code cell → figure → "what just happened" → **Try it yourself**
  (1–2 edit-a-number exercises) → **Recap** (3 bullets) → `Where next`.
- **Code style for beginners**: one idea per cell; full words not abbreviations; every
  non-obvious argument commented; hide plotting boilerplate behind `tutorials/tutorial_helpers.py`
  (a small, *documented* plotting/util module — reader calls `plot_ap(...)`, not raw matplotlib).
  Physics explained in prose next to the code, not only in the science.
- **Truthful science**: state the model (TTP06), the units (mV, ms, cm, cm²/ms), and the
  known caveats inline (e.g. LBM CV runs ~30–47% higher; `D` is RAW, effective = `D/(χ·Cm)`).

## Conventions (decided up front)
- **Location**: notebooks `cardiac_core/tutorials/NN_slug.ipynb` (e.g. `01_hello_action_potential.ipynb`).
  Helper: `cardiac_core/tutorials/tutorial_helpers.py`. Index: `cardiac_core/tutorials/README.md`.
- **Kernel**: `heart-conduction` conda env. Every notebook's first code cell is an
  install/import sanity check that fails LOUD with a one-line fix hint.
- **Figures**: notebooks render inline (outputs embed in the `.ipynb`); no scattered PNGs.
  If a lesson exports a **video/GIF** (e.g. a wavefront animation), it follows the CLAUDE.md
  media rule → `media/cardiac_core_tutorials/videos/{date}/{slug}_NN.ext` (gitignore bulk sim
  output under `_sim_outputs/`). Default: keep everything inline.
- **Runtime budget**: each notebook executes end-to-end in **< ~90 s** on CPU (small grids,
  short `t_end`). Tutorials teach, they don't benchmark. Flag any cell > 20 s with a note.
- **Determinism**: fixed seeds / fixed meshes so outputs are stable for the shipped copy.

---

## Phase P0 — PREP  (do ALL of this before writing any lesson)

### P0.1 — Environment & tooling
- [ ] Confirm `nbformat` + `nbconvert` importable in the env; `pip install` into
      `heart-conduction` if missing (`ipykernel` 6.31 already present).
- [ ] Confirm headless execute-test works: `jupyter nbconvert --to notebook --execute
      --ExecutePreprocessor.kernel_name=heart-conduction <nb>` on a throwaway notebook.
- [ ] Decide authoring method: build `.ipynb` programmatically via `nbformat` from a Python
      source-of-truth per lesson (recommended — keeps cells reviewable/diffable), OR author
      JSON directly. Pick nbformat; write a tiny `_build_notebook.py` scaffold.

### P0.2 — Single-cell: a DEDICATED `cardiac_core.single_cell()` feature (DECIDED — build it)
This is a real library feature, not a tutorial crutch. **Decision (2026-07-21, user): tap the
ionic engine directly (true 0-D), consistent API.** Rationale: single-cell was ALWAYS done by
integrating the ionic model directly — no grid, no diffusion (see `Surrogate/surrogate/data/
single_cell_generator.py` + the Optimizer). cardiac_core's ionic layer already exposes everything
needed — `TTP06Model.get_initial_state(n_cells=1)` + `model.step(V, states, dt, I_stim)` — it just
isn't wrapped. The "small uniform grid" trick is REJECTED (dishonest: it's secretly 3×3 tissue).
- [ ] **Implement `cardiac_core.single_cell(...)`** wrapping the ionic-direct loop: build resting
      state via `get_initial_state`, run the monolithic `model.step(V, states, dt, I_stim)` loop
      under a stimulus schedule, package `(times, V(t), states)` into a result. Closes the
      usability-audit gap ("no 0-D single-cell mode / clean single-cell automaticity IMPOSSIBLE via
      the public API"). Export at top level (`cc.single_cell`).
- [ ] **CONSISTENT API (user requirement):** reuse the SAME stimulus vocabulary as the tissue
      factories — the `{start_time, duration, amplitude}` dict / `Stimulus` (the spatial `region`
      is simply omitted → the whole cell). So Lesson 1 teaches the exact stimulus keys used in
      Lessons 2-8; the only thing that drops away is geometry. Signature target:
      `single_cell(model="ttp06", stim=..., t_end=..., dt=..., cell_type=...) -> result` with
      `result.times`, `result.V`, `result.states` (+ `apd_at`/analysis reusable on `result.V`).
- [ ] **Two properties to VERIFY (expected to hold):** (a) it sidesteps the #14 mono-ionic
      ordering bug — `model.step` is the monolithic V5.3 path (currents from OLD gates), so
      single-cell is correct by construction; (b) `single_cell("ord")` likely WORKS even though
      `monodomain("ord")` raises (the ORd single-cell generator already uses the monolithic step) —
      confirm, and if so it makes ORd reachable for single-cell.
- [ ] Tests: resting cell stays at rest (no drift); a supra-threshold stim fires one clean AP
      (rest→upstroke→plateau→repol); sub-threshold does not; APD90 in the physiological band;
      determinism (fixed dt → bit-reproducible). Land in `cardiac_core/tests/`.

### P0.3 — API-surface audit (lock the exact calls each lesson uses)
- [ ] Run the `API_CHEATSHEET.md` `# runnable-canary` block; confirm it passes on this branch.
- [ ] For EACH lesson, list the exact public calls + verify each runs and returns what the
      lesson claims. Capture gotchas inline so lessons don't teach footguns:
      `ConductivityConfig` (`isotropic`/`anisotropic`/`bidomain`, chi/Cm, D RAW),
      `scale_conductance('GNa', f)` (case-sensitivity; `PCa`≠`GCaL`), `set_conductivity(mask, D=0)`,
      `rectangle_mask`/`circle_mask`, `r.cv()/lat()/apd()`, `apd_map`/`radial_cv`/`restitution_slope`/
      `dominant_frequency_map`, `bidomain(...)` + `BoundarySpec.bath_coupled_edges`,
      `clamp_voltage` (new). Note: **ORd is LBM-only**; masked nodes read back **NaN** (explain, don't hide).
- [ ] Confirm the anisotropy knobs: which arg is longitudinal (`D_l`, along fiber) vs transverse
      (`D_t`, cross-fiber) and how `fiber_angle` rotates them. Record the exact signature used.

### P0.4 — Parameter fact-sheet (verified numbers, not memory) → `tutorial_helpers.py` docstring
- [ ] **GNa** = fast Na⁺ conductance (I_Na). Drives phase-0 upstroke velocity/excitability →
      CV. Measure: max `dV/dt`, CV vs GNa scale (e.g. 0.5×, 1×, 1.5×); find the block threshold.
- [ ] **D_l / D_t** = longitudinal / transverse diffusivity (effective = D/(χ·Cm)). CV ∝ √D.
      Anisotropy ratio D_l/D_t sets wavefront ellipse aspect (√ratio). Record CV_L, CV_T for the
      lesson's chosen values.
- [ ] **monodomain vs LBM**: same physical D, LBM CV ~+30–47% (numerics). Record both engines'
      CV on the L5 mesh so the lesson states real numbers.
- [ ] **Single AP landmarks** (TTP06 EPI/ENDO): resting V, peak, APD90 range — so Lesson 1's
      annotations are correct.

### P0.5 — Notebook template + helper module
- [ ] `tutorial_helpers.py`: PLOTTING/util only — `plot_ap(times, V)`, `plot_field(grid2d, title)`,
      `plot_isochrones(lat)`, `plot_two(a, b, labels)`, `measure_cv(result, ...)`. Fully
      docstring'd; import-light (matplotlib only). NOTE: `single_cell()` is NOT here — it's a
      first-class `cardiac_core` API (P0.2), so Lesson 1 imports it from `cc`, same as every other
      simulation call. The helpers are strictly for figures, never for simulation.
- [ ] A `TEMPLATE.ipynb` (or the nbformat scaffold) encoding the Rhythm above, so lessons are
      structurally uniform.

### P0.6 — Series design doc (one page, top of README)
- [ ] Finalize the 8-lesson arc + one-line objective each (below). Note bonus/future lessons
      (voltage-clamp APD protocol using the new `clamp_voltage`; reentry/rotor) as **not** in v1.

**P0 exit criteria**: `cc.single_cell()` implemented + tested (correct AP, ORd checked); helper
module runs; template executes clean; every lesson's API calls verified; parameter numbers
recorded. THEN author lessons. (P0.2 `single_cell` is the one code change to the library — commit
it independently, goldens bit-identical since it's additive.)

---

## Lessons (author one notebook per lesson; each = its own "phase")

Each lesson section, when authored, must produce: (1) the `.ipynb`, (2) a `--execute` pass with
no errors, (3) inline figures, (4) 1–2 exercises with expected outcomes, (5) a README row.

### L1 — `01_hello_action_potential.ipynb` — Import & your first single-cell recording
- **Learn**: install-check + `import cardiac_core as cc`; run ONE cell with `cc.single_cell("ttp06",
  stim=...)` — TRUE 0-D, one cell, no tissue; plot V(t); name the 4 AP phases; read APD90. The
  `stim` uses the SAME `{start_time, duration, amplitude}` keys the tissue lessons use (just no
  spatial region) — so the stimulus vocabulary carries straight into Lesson 3+. **API**:
  `cc.single_cell`, `plot_ap`, `apd_at`.
- **Exercise**: change stimulus amplitude; find the threshold below which nothing fires.
- **Recap**: what an AP is, how to run one cell, how to read APD; a cell has no CV (that's Lesson 3).

### L2 — `02_sodium_and_excitability.ipynb` — GNa: the sodium channel & the upstroke
- **Learn**: `scale_conductance('GNa', f)`; overlay APs at 0.5×/1×/1.5×; measure max dV/dt;
  see excitability loss. **API**: `scale_conductance`, dV/dt from `r.Vm`.
- **Exercise**: lower GNa until the cell won't fire; relate to Na-channel blockers (class-I drugs).
- **Recap**: GNa ↔ upstroke ↔ excitability.

### L3 — `03_cell_to_cable_conduction_velocity.ipynb` — From one cell to a propagating wave
- **Learn**: build a 1-D strip `Grid`; stimulate one end; watch propagation; measure CV.
  **API**: `Grid`, stimulus region, `r.lat()` activation map, `r.cv(x1,x2,y)`.
- **Exercise**: change D (or dx); watch CV change (CV ∝ √D). Note the effective-D caveat.
- **Recap**: CV, activation maps, CV ∝ √D.

### L4 — `04_fibers_and_anisotropy.ipynb` — Direction matters: D_l vs D_t
- **Learn**: `ConductivityConfig.anisotropic(D_l, D_t, fiber_angle)`; point-stimulate center;
  elliptical wavefront; CV_longitudinal vs CV_transverse. **API**: anisotropic config,
  `radial_cv`, isochrone plot.
- **Exercise**: rotate `fiber_angle`; watch the ellipse rotate. Set D_l=D_t → circle.
- **Recap**: fibers, anisotropy ratio → wavefront ellipse (√ratio).

### L5 — `05_monodomain_vs_lbm.ipynb` — Two engines, one heart
- **Learn**: run the SAME 2-D problem with `cc.monodomain` and `cc.lbm`; compare wavefronts/CV;
  when to use which. **API**: both factories, side-by-side `plot_two`.
- **Exercise**: match D and read both CVs; observe the ~30–47% LBM offset (numerics, not error).
- **Recap**: engine choice, tradeoffs, LBM CV caveat.

### L6 — `06_scars_source_sink_and_block.ipynb` — Injured tissue
- **Learn**: carve an inexcitable scar (`set_conductivity(mask, D=0)`); wavefront curvature
  around it; conduction block through a narrow isthmus (source–sink mismatch). **API**:
  `rectangle_mask`/`circle_mask`, `set_conductivity`, curvature/block viz. Explain masked-node NaN.
- **Exercise**: shrink the isthmus width until propagation blocks.
- **Recap**: scars, source–sink, geometric block.

### L7 — `07_the_ep_toolkit.ipynb` — Measuring like an electrophysiologist
- **Learn**: `apd_map` across tissue, `apd_per_beat`, `restitution_slope`, `dominant_frequency_map`.
  Optional: use the new `clamp_voltage` to hold a region for a clean APD protocol. **API**: the
  analysis aggregates + (optional) voltage clamp.
- **Exercise**: pace faster (shorter BCL); watch APD shorten (restitution).
- **Recap**: the standard EP measurements + where each comes from.

### L8 — `08_bidomain_infarct_mixed_bc.ipynb` — Capstone: bidomain infarct with mixed BCs
- **Learn**: `cc.bidomain` (Vm AND phi_e); place an infarct scar; a bath with **mixed boundary
  conditions** (`BoundarySpec.bath_coupled_edges`); visualize extracellular potential φ_e around
  the infarct. Ties every prior lesson together. **API**: `bidomain`, `BoundarySpec`, scar mask,
  φ_e viz. Note: mixed-BC anisotropic elliptic now falls back to PCG (the recent fix) — mention
  the non-convergence warning is informational.
- **Exercise**: move the infarct / change the bath edges; watch φ_e redistribute.
- **Recap**: bidomain vs monodomain, φ_e, boundary conditions, the whole arc.

---

## Phase W — WRAP / polish (after all 8 authored)
- [ ] `README.md` index: series blurb, prerequisites, install/kernel setup, a table (lesson →
      one-line objective → est. runtime), "how to use these" note.
- [ ] **Execute-all gate**: a script `run_all_tutorials.sh` (or a pytest) that `--execute`s every
      notebook headless and fails on any error. Wire it so CI/`/verify` can catch tutorial rot.
- [ ] **Difficulty pass**: read L1 as a near-beginner — is any jump too big? Is every new term
      defined on first use? Trim jargon.
- [ ] **Ship-with-package check**: ensure `tutorials/` is included by the packaging filter (or
      intentionally excluded from the wheel but present in the repo) — decide + document.
- [ ] Cross-link from `API_CHEATSHEET.md` ("new here? start with tutorials/").

## Risks / open decisions (resolve during P0)
1. **Single-cell** — DECIDED (2026-07-21, user): a dedicated `cc.single_cell()` cardiac_core
   feature (ionic-direct, true 0-D), NOT the uniform-grid trick, with a stimulus API consistent
   with the tissue factories. This is a real library addition that closes the audit's "no 0-D
   mode" gap. Build + test in P0.2 before any lesson. Open sub-question: exact result object /
   whether to also expose a bare pacing shortcut (defer — start with the consistent-stim signature).
2. **`nbformat`/`nbconvert` install** — confirm early; the whole authoring/testing loop needs them.
3. **Runtime creep** — L5 (two engines) and L8 (bidomain) are the slowest; keep grids small and
   `t_end` short; pre-verify each < ~90 s.
4. **API drift** — these notebooks execute real `cardiac_core`; the execute-all gate (Phase W) is
   what keeps them from silently rotting as the library evolves. Not optional.
5. **Scope** — v1 = the 8 lessons above. Voltage-clamp-APD and reentry/rotor are bonus/v2.

## Execution order
P0 (all) → L1 → L2 → … → L8 (each: author → `--execute` → figures → exercises → README row) → W.
Commit per lesson (notebook + helper deltas + README row). Keep each notebook's outputs committed
(they ARE the shipped teaching artifact), but gitignore any `_sim_outputs/` bulk media.
