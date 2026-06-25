# PLAN: Goal-2 LLM layer — script-generating skill suite for wet-lab scientists

Created: 2026-06-25
Engine(s): None (skills + reference docs + a small `cardiac_core/viz.py`); drives the shipped `cardiac_core` API
Research question: [engine_consolidation](README.md)
Source: [IDEALOG.md](IDEALOG.md) — 2026-06-25 "Goal 2 design" thread entry

## Objective
Build the Goal-2 LLM wrapper as a **skill suite** that lets wet-lab scientists (cell-culture / tissue-chip,
no computational-sim background) run cardiac simulations by **describing the experiment** — the skill
**generates a runnable `cardiac_core` script** with correct API usage, behind a mandatory **manifest +
double-check accountability gate** (no vibe-coding runoff). Suite: a maintained api-cheatsheet → the
keystone `/sim-experiment` → `/sim-preset` → `/sim-media` → `/sim-notebook`. Drives the shipped API
directly (Layer-A `SimulationSpec` deferred; programmatic claude-api later).

## Success Criteria
- [ ] `cardiac_core/API_CHEATSHEET.md` — accurate, current; a smoke script using ONLY its patterns runs and prints a physiological CV.
- [ ] `/sim-experiment` skill bundle (`SKILL.md` + `reference/`) implements RECEIVE→INTERPRET→MANIFEST→**double-check gate**→generate `Lab/{date}_{slug}/` (`MANIFEST.md`+`run.py`) + append `Lab/NOTEBOOK.md`→offer run. The generated `run.py` RUNS against `cardiac_core`.
- [ ] `/sim-preset` — save / list / load named parameter sets (`Lab/presets/{name}.yaml`); a preset round-trips into a runnable script.
- [ ] `/sim-media` + `cardiac_core/viz.py` — standardized propagation video + CV/APD maps to canonical `media/` paths; tested.
- [ ] `/sim-notebook` — organizes `Lab/` (master log index, per-experiment manifests, cross-experiment summary).
- [ ] All 137 existing cardiac_core tests still pass (the suite is additive; only `viz.py` adds code).

## Architecture Changes
- NEW: `cardiac_core/API_CHEATSHEET.md` — the canonical, maintained API reference (co-located with the code so it can't drift); all skills reference it.
- NEW: `.claude/skills/sim-experiment/{SKILL.md, reference/{recipes.md, manifest-template.md, run-template.py}}` — the keystone bundle (FIRST bundled skill in the repo).
- NEW: `.claude/skills/sim-preset/SKILL.md`, `.claude/skills/sim-media/SKILL.md`, `.claude/skills/sim-notebook/SKILL.md`.
- NEW: `Lab/` — `Lab/README.md`, `Lab/NOTEBOOK.md` (master log), `Lab/presets/`, per-experiment `Lab/{date}_{slug}/`.
- NEW: `cardiac_core/viz.py` (+ `cardiac_core/tests/test_viz.py`) — standardized `propagation_video()`, `cv_map()`/`apd_map_figure()`, `activation_isochrones()`; export via `cardiac_core/__init__.py` lazy map.
- MOD: `cardiac_core/__init__.py` — add `viz` to the lazy export map.

## Known Failures (from IDEALOG — do NOT retry / reintroduce)
- **Conversational non-coder wizard / auto-teaching UX** — REJECTED twice. Audience is wet-lab scientists; the deliverable is a GENERATED SCRIPT, not a hand-holding chat. No engine-choice interrogation, no auto physics lectures.
- **Hallucinated `cardiac_core` API** — the #1 failure mode. Generated scripts MUST use only what's in `API_CHEATSHEET.md`; the cheatsheet is verified by a runnable smoke script. Never invent signatures.
- **Running without confirmation** — the double-check gate is mandatory. Never execute a generated sim before the scientist confirms the manifest.
- **Building `SimulationSpec`/`create_simulation` (Layer A)** — DEFERRED this pass; the skills map free-form input → the existing factories directly.
- **Touching the engines / V5.3 / V5.4 / `_archive`** — out of scope; this is skills + one viz module on top of the shipped API.

---

## Phase 1: `cardiac_core/API_CHEATSHEET.md` (the shared, maintained asset)

**Goal**: One accurate, current reference of every `cardiac_core` call a generated script needs — the asset
that prevents hallucinated-API failures. Validated by a smoke script that uses ONLY the cheatsheet.
**Tier**: medium
**Estimated scope**: one reference doc + one runnable smoke script.

### Phase Context
- Co-locate the cheatsheet WITH the code (`cardiac_core/API_CHEATSHEET.md`) so it's updated when the API changes; the skills reference it by repo-relative path. (The older `Research/Active/engine_consolidation/API_REFERENCE.md` predates the consolidation and is design-oriented — do NOT reuse it as the cheatsheet; distill from the SHIPPED code.)
- The public surface is the `_LAZY` map in `cardiac_core/__init__.py`. The script-relevant subset: `Grid`, `ConductivityConfig`, `monodomain`/`bidomain`/`lbm`, `simulate`, `SimulationResult` (`.Vm/.times/.cv()/.apd()/.lat()/.restitution()`), `analysis.*`, `geometry.*` masks, `media_path`, `create_cardiac_mesh`.
- conda env `heart-conduction`; CPU + float64; deterministic.

### Step 1.1: Write the cheatsheet + a runnable smoke script
**Model**: opus

#### Read First
- `cardiac_core/__init__.py` (`_LAZY` map — the exact public names).
- `cardiac_core/api.py:1025` `monodomain(geometry, ionic_model, conductivity, stimulus, *, mesh, dt, splitting, diffusion_solver, linear_solver, device)`; `:1137` bidomain; `:1292` lbm; `:161` `CardiacSimulation.run(t_end, save_every=1.0, *, batch=None, record=("Vm",), callback=None)`.
- `cardiac_core/grid.py:37` `Grid(Nx, Ny, dx, dy=None, *, mask=None, device='cpu', dtype=float64)`.
- `cardiac_core/conductivity.py:61` `.isotropic(sigma, chi=1400, Cm=1)`, `:66` `.bidomain(sigma_i, sigma_e, ...)`, `:73` `.anisotropic(...)`.
- `cardiac_core/run.py:49-67` `SimulationResult.cv(x1,x2,y)/.apd()/.lat()/.restitution(ix,iy)`; `:231` `simulate(mesh, t_end, save_every, *, engine, device)`.
- `cardiac_core/geometry.py:109` `left_edge_mask(Nx,Ny,dx,width)`, `:23` `circle_mask`, `:51` `rectangle_mask`; `cardiac_core/media.py:37` `media_path(question, kind, slug)`.
- `cardiac_core/tests/test_construction_api.py` + `test_run_contract.py` — copy the VERIFIED call patterns (Grid + ConductivityConfig.bidomain + stimulus dict + run().cv()).

#### Why
Every downstream skill generates code against this doc. If it's wrong, every generated script breaks — so
it must be distilled from the shipped source and *proven runnable*, not written from memory.

#### Implementation Spec
**Files to create:**
- `cardiac_core/API_CHEATSHEET.md` — sections: (1) one-import (`import cardiac_core as cc`); (2) geometry (`cc.Grid`, masks); (3) conductivity (`cc.ConductivityConfig.isotropic/.bidomain` — UNITS: σ raw mS/cm, `D_eff=σ/(χCm)` derived; do NOT pass a pre-divided D); (4) stimulus (the dict form `{'region': callable|mask, 'start_time','duration','amplitude'}`); (5) construct (`cc.monodomain/bidomain/lbm(grid, 'ttp06', cond, stim)`); (6) run (`sim.run(t_end, save_every)` → `SimulationResult`); (7) measure (`r.cv(x1,x2,y)`, `r.apd()`, `r.lat()`, `analysis.*`); (8) a full end-to-end example (the CV strip). Each entry: exact signature + a 1-line example. Mark engine-choice guidance (bidomain only for bath/boundary).
- `Lab/_validate/smoke.py` — a minimal script using ONLY cheatsheet patterns: build a 2×0.5 cm strip, σ via `.bidomain(1.74,6.25)`, left-edge stim, `run(t_end=40)`, print `cv`. Assert `10 < cv < 100`.

#### Pseudocode
```
# smoke.py
import cardiac_core as cc
g = cc.Grid(200, 50, 0.01)
cond = cc.ConductivityConfig.bidomain(1.74, 6.25, chi=1400)
sim = cc.monodomain(g, 'ttp06', cond, {'region': lambda x,y: x < 0.05, 'start_time':1.0,'duration':2.0,'amplitude':-80.0})
r = sim.run(t_end=40.0, save_every=0.5)
cv = r.cv(x1=20, x2=180, y=25); assert 10 < cv < 100; print(cv)
```

#### Test Spec
- `Lab/_validate/smoke.py` — run it; exits 0 and prints a CV in ~tens of cm/s. This IS the cheatsheet's correctness gate.

#### Checklist
- [ ] `API_CHEATSHEET.md` covers every script-relevant call with exact signatures + examples + the σ-units trap.
- [ ] `smoke.py` uses ONLY cheatsheet patterns and runs green.

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
conda run -n heart-conduction python Lab/_validate/smoke.py
```

#### Exit Criteria
- [ ] smoke script prints a physiological CV; cheatsheet entries all match shipped signatures (spot-check 3 against source).

#### Risk
Cheatsheet drifts from code later. Mitigation: co-located with `cardiac_core/`; `smoke.py` is the canary (re-run after any API change). Note this in the cheatsheet header.

### Phase 1 Verification / Exit / Cleanup
```bash
conda run -n heart-conduction python Lab/_validate/smoke.py && conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q --deselect cardiac_core/tests/test_conductivity.py::test_live_cv_gate
```
- [ ] smoke green; 127 existing tests still green (additive). No engine edits.

**-> Commit point: git commit after Phase 1** (`docs(cardiac_core): API_CHEATSHEET + Lab smoke validator`)

---

## Phase 2: `/sim-experiment` keystone skill + `Lab/` scaffolding

**Goal**: The keystone skill — free-form description → parameter manifest → **double-check gate** → generate a
runnable `Lab/{date}_{slug}/run.py` + `MANIFEST.md` + a `Lab/NOTEBOOK.md` line. The generated script runs.
**Tier**: large
**Estimated scope**: one bundled skill (SKILL.md + 3 reference files) + the `Lab/` scaffold + a worked-example validation.

### Phase Context
- Skill bundle layout (FIRST bundled skill here): `.claude/skills/sim-experiment/SKILL.md` + `reference/{recipes.md, manifest-template.md, run-template.py}`. The `SKILL.md` frontmatter: `name: sim-experiment`, `description: ...`, `argument-hint: "[free-form experiment description]"`.
- THE PROTOCOL (non-negotiable order): RECEIVE free-form → INTERPRET (map to `cardiac_core` params; infer engine — bidomain only if bath/boundary; ask ONLY genuine gaps) → present MANIFEST (plain text; core fields required, scientist/hypothesis/runtime OPTIONAL) → ⛔ DOUBLE-CHECK GATE (explicit confirm; NEVER generate/run before it) → on "go": create `Lab/{YYYY-MM-DD}_{slug}/`, write `MANIFEST.md` (the confirmed text verbatim) + `run.py` (from the template, filled via the cheatsheet) + append one line to `Lab/NOTEBOOK.md` → OFFER to run; on run, verify (CV physiological / not NaN / activated), then offer `/sim-media`.
- `run.py` template: a `# === PARAMETERS (edit these) ===` block (plain-language comments) then the `cardiac_core` construction/run/measure — scientists edit params, not API calls. Generation fills it strictly from `cardiac_core/API_CHEATSHEET.md`.
- `recipes.md`: intent→recipe templates (CV strip · reentry sheet (S1-S2) · APD restitution (regular pacing) · edge/bath CV (bidomain)) with default geometry/pacing/measure each maps to.
- Reference the cheatsheet by repo path; do NOT inline the API (single source of truth).

### Step 2.1: `Lab/` scaffold + the `run.py` template + recipes + manifest template
**Model**: opus

#### Read First
- `cardiac_core/API_CHEATSHEET.md` (Phase 1) — the only API source the template may use.
- `Lab/_validate/smoke.py` — the proven call pattern the template generalizes.
- CLAUDE.md "Saving images & videos" — the `media/` convention the template's media step follows.

#### Why
The template + recipes are what make generation deterministic and runnable; the manifest template fixes the
accountability record's shape.

#### Implementation Spec
**Files to create:** `Lab/README.md` (explains the structure + the accountability gate), `Lab/NOTEBOOK.md` (master-log header + table columns: date · slug · goal · engine · status · result), `.claude/skills/sim-experiment/reference/run-template.py` (PARAMETERS block + construction/run/measure/media placeholders), `reference/recipes.md`, `reference/manifest-template.md`.
**Interfaces:** `run-template.py` must, when filled, match `smoke.py`'s working pattern + add a standardized media call (Phase 4 wires real media; here a placeholder/`print`).

#### Pseudocode
```
run-template.py:
  # === PARAMETERS (edit these) ===  LENGTH_CM, WIDTH_CM, DX, SIGMA, T_END_MS, ...
  g = cc.Grid(...); cond = cc.ConductivityConfig.<mode>(...); stim = {...}
  sim = cc.<engine>(g, '<ionic>', cond, stim); r = sim.run(t_end=T_END_MS, save_every=...)
  print measurement(s); (media placeholder)
```

#### Test Spec
- Fill `run-template.py` by hand for the CV-strip recipe → save as a temp script → it runs and prints a CV (same gate as smoke).

#### Checklist
- [ ] `Lab/` scaffold (README, NOTEBOOK header, `presets/` dir created empty with `.gitkeep`).
- [ ] `run-template.py` fills to a runnable script; recipes.md has ≥4 recipes; manifest-template.md matches the agreed field list (core required + optional).

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
# fill the template for CV strip into /tmp and run it:
conda run -n heart-conduction python - <<'PY'
# (agent: render run-template.py with CV-strip params, exec, assert a CV prints)
PY
```

#### Exit Criteria
- [ ] template renders to a runnable script; recipes + manifest template complete.

#### Risk
Template embeds a stale API call. Mitigation: it must mirror `smoke.py` exactly; re-run after filling.

### Step 2.2: `/sim-experiment` SKILL.md (the protocol + the gate)
**Model**: opus

#### Read First
- `.claude/skills/research-new/SKILL.md` and `.claude/skills/audit/SKILL.md` — the repo's SKILL.md conventions (YAML frontmatter, numbered Steps, Rules section).
- the Phase-2 reference files (run-template, recipes, manifest-template) + `cardiac_core/API_CHEATSHEET.md`.

#### Why
This is the keystone deliverable — the protocol that turns free-form input into an accountable, runnable
experiment. The double-check gate is the whole point ("no vibe-coding runoff").

#### Implementation Spec
**Files to create:** `.claude/skills/sim-experiment/SKILL.md` — frontmatter + Steps: 1 RECEIVE, 2 INTERPRET (→ recipe + cheatsheet; engine-inference rule; ask only gaps), 3 MANIFEST (render `manifest-template.md`), 4 **DOUBLE-CHECK GATE** (hard stop; the Rules section forbids proceeding without confirm), 5 GENERATE (`Lab/{date}_{slug}/` + MANIFEST.md verbatim + run.py from template + NOTEBOOK.md append), 6 RUN (offer; verify sane; hand to `/sim-media`). Plus a **Rules** block: never run before confirm; only cheatsheet API; manifest = the record.

#### Pseudocode
```
slug = kebab(goal); dir = Lab/{today}_{slug}
on confirm: write dir/MANIFEST.md (verbatim), dir/run.py (filled template), append NOTEBOOK row
offer: "run it now?" -> python dir/run.py -> verify cv/apd sane -> "make figures? /sim-media"
```

#### Test Spec
- DRY-RUN the protocol on a sample ("how fast does the wave travel in a 2cm strip?"): produces a manifest, and after a simulated "confirm" the generated `Lab/{...}/run.py` runs and prints a CV; `NOTEBOOK.md` gains a row. (Validate by following the SKILL.md steps manually end-to-end.)

#### Checklist
- [ ] SKILL.md has the 6 steps + the hard double-check Rule + cheatsheet-only Rule.
- [ ] worked example produces folder + MANIFEST.md + run.py (runs) + NOTEBOOK row.

#### Verify
```bash
# manual: follow SKILL.md on the sample input; then:
conda run -n heart-conduction python Lab/2026-06-25_*/run.py   # the generated script runs
grep -c "|" Lab/NOTEBOOK.md   # a row was appended
```

#### Exit Criteria
- [ ] end-to-end: description → manifest → (confirm) → runnable script + logged. Gate enforced in the Rules.

#### Risk
Skill generates+runs without the gate. Mitigation: the gate is a top Rule + Step 4 is a hard stop; the worked example must show the pause.

### Phase 2 Verification / Exit / Cleanup
```bash
conda run -n heart-conduction python Lab/2026-06-25_*/run.py
conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q --deselect cardiac_core/tests/test_conductivity.py::test_live_cv_gate
```
- [ ] keystone skill produces a runnable, logged, accountable experiment; existing tests green. **(audit-worthy — see Final.)**

**-> Commit point: git commit after Phase 2** (`feat(skills): /sim-experiment keystone + Lab/ scaffold`)

---

## Phase 3: `/sim-preset` — save / store / reuse parameter sets

**Goal**: Scientists save named parameter sets and reuse them across experiments.
**Tier**: medium
**Estimated scope**: one SKILL.md + a preset format + a loader pattern the template understands.

### Phase Context
- Preset = a YAML at `Lab/presets/{name}.yaml` holding the manifest's editable params (geometry, conductivity, ionic, pacing, sim length, measure). `/sim-preset` save/list/load. `/sim-experiment` can start from a preset (`/sim-experiment using preset healthy_strip`).
- Keep it plain YAML (human-readable, hand-editable); no new runtime dep beyond `pyyaml` (already available via conda env — verify).

### Step 3.1: `/sim-preset` skill + preset format + loader
**Model**: opus

#### Read First
- `Lab/` scaffold + `reference/run-template.py` (the params a preset must carry).
- `reference/manifest-template.md` (same field set).

#### Why
Presets turn one accountable experiment into a reusable, comparable series (control vs knockdown vs …) — the
wet-lab workflow.

#### Implementation Spec
**Files to create:** `.claude/skills/sim-preset/SKILL.md` (save/list/load), `Lab/presets/_SCHEMA.md` (the YAML keys), an example `Lab/presets/healthy_ventricle_strip.yaml`. **Files to modify:** `reference/run-template.py` — accept an optional `PRESET = "name"` that loads `Lab/presets/{name}.yaml` over the defaults.

#### Pseudocode
```
save: write Lab/presets/{name}.yaml from the confirmed manifest params
load: yaml.safe_load -> dict -> overrides the PARAMETERS block (script stays editable)
```

#### Test Spec
- Save a preset from a manifest, load it into the template, run → same CV as the inline-param version (round-trip fidelity).

#### Checklist
- [ ] save/list/load documented; `_SCHEMA.md` matches the manifest fields; example preset present; template preset-aware.

#### Verify
```bash
conda run -n heart-conduction python -c "import yaml; print(yaml.safe_load(open('Lab/presets/healthy_ventricle_strip.yaml')))"
# then run a preset-driven script and confirm CV matches
```

#### Exit Criteria
- [ ] a preset round-trips into a runnable script with identical results.

#### Risk
Preset/manifest field drift. Mitigation: `_SCHEMA.md` is the single field list both reference.

### Phase 3 Verification / Exit / Cleanup
- [ ] preset round-trip green; existing tests unaffected.

**-> Commit point: git commit after Phase 3** (`feat(skills): /sim-preset parameter sets`)

---

## Phase 4: `/sim-media` + `cardiac_core/viz.py` (standardized output)

**Goal**: One standardized way to turn a result into a propagation video + CV/APD/isochrone figures, saved to
canonical `media/` paths — so every experiment's output looks the same and lands in the right place.
**Tier**: large
**Estimated scope**: a tested `viz.py` module + a thin skill that calls it.

### Phase Context
- The reusable code lives in `cardiac_core/viz.py` (so generated scripts call ONE helper, not bespoke matplotlib). Functions consume a `SimulationResult` (`.Vm (T,Nx,Ny)`, `.times`, `.dx`). Save via `cardiac_core.media.media_path(question, kind, slug)` (default `question='lab'`).
- float64 in; matplotlib for figures; `matplotlib.animation`/`cv2` for the video (check which is available; prefer matplotlib FuncAnimation → mp4 via ffmpeg, fallback gif).
- `/sim-media` is thin: given a result (or a saved `.npz`/a `Lab/{exp}/`), call the viz functions, report the saved paths.

### Step 4.1: `cardiac_core/viz.py` + tests
**Model**: opus

#### Read First
- `cardiac_core/run.py:19-67` (`SimulationResult` fields + hooks), `cardiac_core/media.py:37` (`media_path`), `cardiac_core/analysis.py` (`activation_time`, `apd_map`).
- `cardiac_core/__init__.py` `_LAZY` (add `viz`).

#### Why
Standardized, tested visuals are the scientist-facing payoff; centralizing them stops every generated script
from reinventing (and mis-coloring) plots.

#### Implementation Spec
**Files to create:** `cardiac_core/viz.py` — `propagation_video(result, slug, question='lab', fps=20) -> path`; `cv_map`/`apd_map_figure(result, slug, ...) -> path`; `activation_isochrones(result, slug, ...) -> path`. `cardiac_core/tests/test_viz.py`. **Files to modify:** `cardiac_core/__init__.py` (`'propagation_video': 'viz'`, etc.).
**Interfaces:** all take a `SimulationResult`, return the saved file path (under `media/lab/...`), float64-safe, headless (`matplotlib.use('Agg')`).

#### Pseudocode
```
propagation_video: Agg backend; FuncAnimation over result.Vm[t]; imshow(vmin=-90,vmax=40); save mp4 (ffmpeg) else gif; return media_path('lab','videos',slug)
apd_map_figure: analysis.apd_map(Vm,times) -> imshow -> savefig -> media_path('lab','images',slug)
```

#### Test Spec
- `test_viz.py::test_propagation_video` — small result (run a 30×10 monodomain 10 ms) → returns an existing non-empty file under `media/`. `::test_apd_map_figure` — returns an existing PNG. Use `tmp`/`media/lab/_sim_outputs` (gitignored) for test artifacts.

#### Checklist
- [ ] 3 viz functions, Agg backend, float64, return real paths under `media/`; exported lazily; tests green.

#### Verify
```bash
conda run -n heart-conduction python -m pytest cardiac_core/tests/test_viz.py -v
```

#### Exit Criteria
- [ ] viz functions produce real media files; tests pass; no regressions.

#### Risk
No ffmpeg → mp4 fails. Mitigation: try mp4, fall back to gif; test asserts "a file exists", not the format.

### Step 4.2: `/sim-media` skill + wire into the run-template
**Model**: opus

#### Read First
- `cardiac_core/viz.py` (Step 4.1), `reference/run-template.py` (the media placeholder to replace).

#### Implementation Spec
**Files to create:** `.claude/skills/sim-media/SKILL.md` (given a result or a `Lab/{exp}/`, produce standardized media; report paths). **Files to modify:** `reference/run-template.py` — replace the media placeholder with `cc.propagation_video(r, slug)` + a measurement figure.

#### Pseudocode
```
SKILL: locate result (run the exp's run.py if no result) -> cc.propagation_video + cc.<map> -> print media/ paths -> log to MANIFEST/NOTEBOOK
```

#### Test Spec
- Run a `Lab/{exp}/run.py` whose template now calls viz → a video + figure exist under `media/lab/`.

#### Checklist
- [ ] skill produces standardized media for an experiment; template wired to viz; paths reported.

#### Verify
```bash
conda run -n heart-conduction python Lab/2026-06-25_*/run.py && ls media/lab/**/ 2>/dev/null | head
```

#### Exit Criteria
- [ ] an experiment run yields standardized video+figure in canonical paths.

#### Risk
Media bloats git. Mitigation: bulk outputs → `media/**/_sim_outputs/` (gitignored per CLAUDE.md); only curated figures committed.

### Phase 4 Verification / Exit / Cleanup
```bash
conda run -n heart-conduction python -m pytest cardiac_core/tests/test_viz.py cardiac_core/tests/ -q --deselect cardiac_core/tests/test_conductivity.py::test_live_cv_gate
```
- [ ] viz tested + wired; standardized media produced; float64; existing tests green.

**-> Commit point: git commit after Phase 4** (`feat(cardiac_core+skills): viz module + /sim-media`)

---

## Phase 5: `/sim-notebook` — lab-notebook organization

**Goal**: Keep `Lab/` organized — a master index of experiments, per-experiment manifests, and a cross-experiment
summary (compare a series at a glance).
**Tier**: medium
**Estimated scope**: one SKILL.md operating on the `Lab/` structure Phases 2–4 produce.

### Phase Context
- The master log `Lab/NOTEBOOK.md` is appended by `/sim-experiment`; `/sim-notebook` curates it: rebuild/sort the index from the per-experiment `MANIFEST.md` files (source of truth), flag stale/failed runs, and render a comparison view for a preset-series (e.g. σ-sweep → CV table).
- Read-only over `cardiac_core` (no engine/code changes); pure organization over `Lab/`.

### Step 5.1: `/sim-notebook` skill
**Model**: opus

#### Read First
- `Lab/NOTEBOOK.md` + a couple of `Lab/{exp}/MANIFEST.md` (the data it organizes).
- `.claude/skills/research-status/SKILL.md` — the repo's "status/index" skill pattern to mirror.

#### Why
Without curation the notebook rots; scientists need to find/compare past runs — the lab-notebook half of the goal.

#### Implementation Spec
**Files to create:** `.claude/skills/sim-notebook/SKILL.md` — modes: `index` (rebuild `NOTEBOOK.md` from manifests), `summary` (one experiment), `compare` (a list/preset-series → a parameter+result table).

#### Pseudocode
```
index: scan Lab/*/MANIFEST.md -> rows(date,slug,goal,engine,status,result) -> rewrite NOTEBOOK.md table (sorted)
compare: given slugs -> table of differing params + each result side by side
```

#### Test Spec
- With ≥2 `Lab/{exp}/` present, `index` rebuilds `NOTEBOOK.md` listing both; `compare` emits a table with the differing param + results. (Manual: follow SKILL.md; check the rendered markdown.)

#### Checklist
- [ ] index/summary/compare documented; manifests are the source of truth; NOTEBOOK rebuild is idempotent.

#### Verify
```bash
ls Lab/*/MANIFEST.md && wc -l Lab/NOTEBOOK.md
```

#### Exit Criteria
- [ ] notebook index rebuilds from manifests; compare view works on a 2-experiment series.

#### Risk
Index overwrites hand notes. Mitigation: manifests are source-of-truth; NOTEBOOK.md is generated (header warns "regenerated by /sim-notebook").

### Phase 5 Verification / Exit / Cleanup
- [ ] `/sim-notebook` organizes `Lab/`; all 4 skills + cheatsheet coherent; 137 tests green.

**-> Commit point: git commit after Phase 5** (`feat(skills): /sim-notebook lab-notebook organization`)

---

## Final Cleanup (cross-phase)
- [ ] float64 in `cardiac_core/viz.py` (no float32 leaks); headless Agg backend; no GUI calls.
- [ ] V5.3/V5.4/`_archive`/engines untouched; the suite is additive (skills + `viz.py` + `Lab/` + cheatsheet).
- [ ] All 137 cardiac_core tests pass (+ `test_viz.py`); `Lab/_validate/smoke.py` green.
- [ ] Generated scripts use ONLY `cardiac_core/API_CHEATSHEET.md` — no hallucinated API anywhere in templates/skills.
- [ ] `.gitignore`: `media/**/_sim_outputs/`, `Lab/**/outputs/` bulk artifacts (keep MANIFEST.md/run.py/NOTEBOOK.md).
- [ ] KNOWLEDGE.md + IDEALOG updated with the shipped suite; README north-star Goal-2 wording corrected (wet-lab scientists, not "non-coder").
- [ ] **`/audit` the keystone Phase 2** before relying on it (the double-check gate must be un-bypassable).
- [ ] Archive this plan:
```bash
mkdir -p Research/Active/engine_consolidation/plans
cp Research/Active/engine_consolidation/PLAN.md "Research/Active/engine_consolidation/plans/$(date +%Y-%m-%d)_goal2-skill-suite.md"
```
- [ ] Revert the bottom tmux pane from PLAN.md back to WHITEBOARD.md:
```bash
tmux send-keys -t 2 C-c
sleep 0.3
tmux send-keys -t 2 'W=$(tput cols); H=""; while true; do N=$(md5sum Research/Active/engine_consolidation/WHITEBOARD.md 2>/dev/null | cut -d" " -f1); if [ "$N" != "$H" ]; then clear; glow -s .glow-style.json -w $W Research/Active/engine_consolidation/WHITEBOARD.md 2>/dev/null; H=$N; fi; sleep 1; done' Enter
```

## Out of Scope (follow-up plans)
- Layer A: `SimulationSpec`/`create_simulation` declarative bridge.
- Programmatic claude-api wrapper (NL → spec → run, embeddable).
- Notebook (`.ipynb`) output format (scripts only for now).
- Optimizer/Surrogate-facing variants of the skills.

## Mutation Log
(empty — populated during execution: `**MUTATED {date}**: Step X.Y {SKIPPED|SPLIT|INSERTED} — {reason}`)
