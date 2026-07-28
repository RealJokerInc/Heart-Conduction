# PLAN.md — "Cardiac Core by Experiment" — CHAPTER RESTRUCTURE (forward authoring plan)

> **Continues** [`2026-07-23_tutorials_PLAN.md`](./2026-07-23_tutorials_PLAN.md) (the converged
> audience contract, Stim through-line, caveat policy, and anti-rot gate all carry forward). This doc
> supersedes its lesson *ordering*: the user restructured the series (2026-07-28) from a flat 10-lesson
> arc into **5 chapters** built on the project's real workflow — intro → cell electrophysiology →
> tissue → pacing → **engine tuning as the capstone where everything converges**. It is the
> machine-targeted plan an agent authors against.
>
> **Status: SHIPPED 2026-07-28 — all 5 chapters authored (12 notebooks) on branch
> `cardiac-core-tutorials` (commits `967c44a`…`a9af4e5`, one per chapter), each verified headless
> (flat `--script` exit 0) with its key figures rendered and inspected, real measured numbers in the
> prose. Notebooks regenerate deterministically from `_build/*.py`. The rotor (4.2) genuinely forms
> (|charge|=1.00), the anisotropy ellipse (5.2) is 2.06:1, and the dx under-resolution trap (5.4) is
> intact (coarse dx reads ~21% low). REMAINING: the `nbconvert --execute` anti-rot gate (Phase W),
> still blocked on the `nbformat`/`nbconvert` install; branch not yet merged/pushed (user's call).**

---

## 1. Where this stands

**Three status facts (from the 2026-07-28 re-cut, still current):**
1. **✅ Authoring gate CLEARED.** `Video`/`Gradient`/`render` + `image`/`trace` + portable-media +
   `.show()` are on `main` (`c3c5a10`); Stim is merged. The old long pole (video) is unblocked.
2. **✅ `single_cell(conductances={...})` is SHIPPED** (parallel agent, committed **`37f8d57`**, via the
   shared `ionic.scaling` helper; cheatsheet updated). This is Chapter 2's drug knob, and its verified
   behaviour is teachable content, not just a satisfied dependency:
   - **Multiplicative** factors (`<1` block, `>1` upregulation), keyword `conductances={NAME: factor}`.
   - **Applied BEFORE `pre_pace`** (`_single_cell.py:107-118`) → pre-pacing settles the *drugged* cell to
     its own steady state. The load-bearing design point — worth stating in Ch2.
   - **Deep-copies the model** (the caller's model is untouched) and **validates names**: a mis-cased or
     unknown name **raises `ValueError`** listing the model's available conductances — *no silent no-op*.
     This safety is itself a Ch2 teaching beat.
   - Verified: `single_cell('ttp06', celltype='EPI', conductances={'GKr': 0.5})` prolongs APD90 217→245 ms.
3. **⚠️ `nbformat`/`nbconvert` NOT installed** — blocks the execute-all anti-rot gate + output
   embedding, **not authoring** (the plain-JSON builder needs neither). A Phase-W item.

**The restructure (user, 2026-07-28), superseding the flat arc:**
- The old "drug on one cell" lesson is **renamed and re-themed** into a systematic *conductance → AP
  morphology* chapter (Chapter 2).
- Tissue construction (grid/conductivity/stim/mask/engines) is its own **in-depth** chapter (3), one
  example per engine.
- Pacing gets its own chapter (4): single-cell rate protocols **and** tissue reentry induction.
- The capstone is **tuning by hand** (5) — the tissue-level counterpart to Chapter 2. Where Ch2 asks
  "what does each conductance do to the AP *shape*?", Ch5 asks "what does each raw knob do to the
  propagating *wave*?": conductivity σ, the diffusion constant D, isotropy vs anisotropy, and
  conductances → CV, propagation, wavefront shape. **Purely by hand — NOT the Optimizer/BayesOpt.**
- **Long chapters split into sub-chapters** (3.1/3.2/3.3 by engine; 4.1/4.2 by scale); short ones
  combine. Rule of thumb: no single notebook much over ~90 s runtime or a scroll the reader loses.

---

## 2. The series — 5 chapters

| # | Chapter | What it teaches | Split if long |
|---|---------|-----------------|---------------|
| **1** | **Cardiac Core Intro** | Install & import; the object landscape (what you need to build a simulation); a single-cell AP; a monodomain tissue run rendered with the image/video pipeline. | — (keep it short) |
| **2** | **Conductances & AP morphology** | TTP06's major conductance channels and what each does to the action-potential shape. **Advanced:** self-pacing models + the funny current. | 2 (main) / 2A (advanced, self-pacing) |
| **3** | **Tissue simulation, in depth** | Re-introduce grid, conductivity, stim, mask, and the engines properly; one worked example per engine. | 3.1 monodomain / 3.2 bidomain / 3.3 LBM |
| **4** | **Pacing** | Single-cell rate protocols (shorten BCL, effective BCL, restitution) and a tissue reentry-induction example. | 4.1 single-cell / 4.2 tissue (reentry) |
| **5** | **Tuning by hand (capstone)** | The full catalog of raw knobs → tissue effect: σ/D/χ/Cm (CV), isotropy·anisotropy·scar·stim-site (wavefront & block), conductances (CV/wavelength), and dx/dt/engine (numerical hygiene). Hand-tuning only; no optimizer. | 5.1 coupling / 5.2 geometry / 5.3 cellular / 5.4 numerical |

**Reading paths:** Chapters 1–2 are the quick start (one cell, an afternoon). 3–4 go to tissue. 5 is
the payoff. Sub-chapters are independent notebooks; combine adjacent ones at authoring time if either
comes out thin.

**Homes for the old "Advanced" topics:** **fibers/anisotropy** is now on the spine — introduced
mechanically in Ch3 (conductivity) and explored for its *effect* (elliptical wavefront, CV_L vs CV_T)
in Ch5's isotropy-vs-anisotropy beat. **Voltage clamp** → a protocol in Ch2 or Ch4 (redline §6). The
`r.fields.*` maps (source–sink, curvature, `safety_factor`) → Ch5, as the "why" behind a conductance
change slowing/blocking the wave (source–sink is the mechanism), or Ch3. The bidomain-infarct/`phi_e`
capstone is retired — bidomain becomes Ch3.2's engine example.

### Chapter 1 in detail — `01_intro.ipynb` (the rewrite, authored first)

Four beats, in the reader's order of first contact. Light on science, heavy on orientation.

1. **Install & import the package.** The Colab-first setup cell (reuse shipped-01's
   `importlib.util.find_spec` guard + `pip install "cardiac-core[viz] @ git+…"`), then
   `import cardiac_core as cc`. One or two plain sentences on what a *package* is and why alias it `cc`.
   `[viz]` is load-bearing (H.264 for beat 4's video) — keep it.
2. **The object landscape — what you need to build a simulation.** Framed as a checklist of the pieces,
   sourced from `cardiac_core/API_OBJECTS.md` "The object map" so it can't drift:
   - a **geometry** — the `Grid`, and **masks** to carve regions/scars;
   - a **conductivity** — `ConductivityConfig` (how well it conducts);
   - an **ionic model** — the cell machinery (`"ttp06"`, …), also runnable alone via `single_cell`;
   - a **stimulus / pacing** — `Stim` (the electrode; trains via `bcl`/`num_pulses`);
   - an **engine** — `monodomain` / `bidomain` / `lbm` (the numerical method), and **the simulation
     object that runs it** (`.run()` → a `SimulationResult` with `r.cv`/`r.apd`/`r.image`/`r.video`).
   Present it as a table + one sentence each: "you'll meet each by doing." No heavy code.
3. **Example — a single cell → the AP curve.** `sc = cc.single_cell('ttp06', celltype='EPI',
   pre_pace=5)`; read `sc.v_rest`, `sc.v_peak`, `sc.apd(0.9)`; plot with **`sc.trace()`** (returns an
   `ImageInfo` from the media layer — the image pipeline, not raw matplotlib). Name the four AP phases.
4. **Example — a monodomain tissue run, rendered with the image/video pipeline.** Build a small strip
   (grid → conductivity → `Stim.boundary`), `cc.monodomain`, `.run()`. **Replace shipped-01's raw
   `plt.imshow` snapshots with `r.video()` (the inline wave movie) and `r.image()` (a snapshot / the
   activation map).** This is the beat that shows "how you get pictures and movies out." Keep the tissue
   details black-box — "Chapter 3 explains every line; here, watch what one command draws." Defer CV to
   Ch3 (a one-liner mention is fine).

Rename `_build/build_01_build_a_simulation.py` → `_build/build_01_intro.py`; update `README.md`; retire
`01_build_a_simulation.ipynb` (delete, or keep as a redirect stub — redline §6).

### Chapter 2 in detail — conductances & AP morphology (TTP06)

**Theme (user):** the reader should learn *which conductance changes what about the AP shape.* Not a
drug lesson — a channel-by-channel tour on a single TTP06 cell.

- Baseline AP (from Ch1's `single_cell`), then scale one conductance at a time and overlay the AP —
  all via the **shipped** `cc.single_cell(conductances={NAME: factor})` (`37f8d57`). The six channels
  that shape the AP (verified names — do NOT invent; TTP06's full scalable set is `GK1/GKr/GKs/GNa/
  GbCa/GbNa/GpCa/GpK/Gto/PCa/PNaK`):
  - **`GNa`** (I_Na, phase-0 upstroke / excitability), **`PCa`** (I_CaL — a permeability, NOT "GCaL";
    the plateau), **`Gto`** (I_to, the phase-1 notch), **`GKr`** (I_Kr, phase-3 repol → *block PROLONGS
    APD*, the hERG story), **`GKs`** (I_Ks, phase-3, rate-dependent), **`GK1`** (I_K1, resting potential
    + late phase-3).
  - Each: one sentence on the current, one figure of the AP at e.g. 0.5×/1×/1.5×, one sentence on the
    morphology change. **Verify each effect by running it** before writing the claim.
- **Teach the knob itself, not just the channels** (uses the shipped behaviour as content):
  - **Drug-before-steady-state** — `conductances=` is applied *before* `pre_pace`, so with `pre_pace>0`
    you record the drugged cell at *its own* steady state, not a control cell nudged once. Worth one
    sentence; it is why the AP shifts are honest.
  - **Names are validated** — a mis-typed/mis-cased name (`'gKr'`, `'GCaL'`) **raises** a `ValueError`
    that *lists the real conductances*, rather than silently doing nothing. Show this once as a feature:
    the reader can trust that a drug that "did nothing" really did nothing, not that they fat-fingered a
    name. (This is the whole reason the knob exists over mutating a model by hand.)
- **Chapter 2A (advanced) — self-pacing & the funny current.** Switch to a spontaneously-active hiPSC
  model — **`"paci"`** (note: `"phas13"` is an **alias** for the same `PHAS13Model`; only `"mhas13"` is
  a distinct variant — present *one* self-pacing model, not three) — run with `stim_amplitude=0.0`
  (verified: paci self-paces, ~0.5 Hz); show diastolic depolarization and automaticity. Then scale the
  **funny current I_f** with the same shipped knob: `conductances={'g_f': 0.5}` slows the spontaneous
  rate. ⚠️ Paci conductances use **lowercase `g_*`** names (`g_Na`, `g_Kr`, `g_f`, …), NOT TTP06's `G*` —
  so the validation error here lists *Paci's* names; check them at authoring.

### Chapter 5 in detail — tuning by hand (the capstone)

**Theme (user):** purely hand-tuning the *raw* parameters and seeing what each does at the **tissue**
level. No optimizer, no BayesOpt, no target-fitting loop — the reader turns one knob at a time on a
tissue sim and watches CV / propagation / wavefront respond. It is Chapter 2's method (sweep, overlay,
one-sentence effect) lifted from the AP to the wave, and it ties Ch2 (conductances) and Ch3 (tissue
params) together into intuition.

**A full catalog of knobs, grouped.** Each entry = sweep it, measure the tissue-level observable,
state the rule — and **verify by running before writing the claim** (state directions here; the exact
exponent/number comes from the run). Grouped so the chapter can split into **5.1–5.4** if it overflows
~90 s / a long scroll; combine adjacent groups if any is thin.

**5.1 — Coupling knobs: how fast the wave goes.** All feed the membrane-effective diffusivity
`D_eff = D/(χ·Cm)` and set CV together.
- **Conductivity σ** — `ConductivityConfig.bidomain(sigma_i, sigma_e)`; `r.cv`; **CV ∝ √σ** (halve σ →
  ~30% slower, not 50%). The headline knob.
- **Diffusion constant D** — same √ law via the D representation (`set_conductivity`/isotropic-D
  config). This is where **`D_eff = D/(χ·Cm)`** is taught — the one numerics fact that earns its place,
  because the reader is turning D directly.
- **χ (surface-to-volume) and Cm (capacitance)** — the *denominators*: raising **χ** or **Cm** lowers
  `D_eff`, so both **slow CV**; Cm also stretches the AP timescale. Sweep each on `ConductivityConfig(…,
  chi=…, Cm=…)` and show the degeneracy — since `ConductivityConfig` is σ-valued, the config-level
  version is `(σ, χ)` and `(σ/k, χ/k)` give the same `D_eff = σ/(χ·Cm)` hence the same wave (the raw-`D`
  form `(D,χ)↔(D/k,χ/k)` is the `set_conductivity` representation). Payoff: the reader sees *why* σ, D,
  χ, Cm are one knob wearing four hats.

**5.2 — Direction & geometry: what shape the wave takes.**
- **Isotropy vs anisotropy** — isotropic (σ_l = σ_t) → a **circular** front from a point stim;
  anisotropic (`anisotropic(sigma_l, sigma_t, fiber_angle)`) → an **ellipse**, CV_L vs CV_T, axis ratio
  ≈ √(σ_l/σ_t). `cc.Stim.center` + `r.radial_cv` + isochrone `r.image(what="activation")`. Rotate
  `fiber_angle`; the ellipse rotates. (Ch3's anisotropy mechanics pay off here.)
- **Scar / heterogeneity geometry** — carve a mask (`rectangle_mask`/`circle_mask` +
  `set_conductivity(mask, D=0)`): the front **curves** around it, and a **narrow enough isthmus
  blocks** (source–sink mismatch). Sweep the isthmus width to the block threshold; show
  `r.fields.source_sink` / `cc.safety_factor` (<1 = block) as the mechanism.
- **Stimulus strength & site size** — the *launch* question: too weak (`amplitude`) or too small a site
  fails to excite (a point source can't drive enough neighbours — source–sink again). Sweep amplitude to
  the capture threshold, and site size to the minimum that launches a wave.

**5.3 — Cellular knobs at tissue scale** (reconnecting Ch2 to propagation).
- **Conductances** — **`GNa`↓ slows CV and, pushed far, blocks**; **`GKr`↓ prolongs APD → longer
  refractory → longer wavelength** (`cc.wavelength(cv, erp)`). The bridge sentence of the series: a
  channel you met at the *cell* in Ch2 changes what the *wave* does here.
- **Cell type** (ENDO/EPI/**`M_CELL`**) → different APD → different refractoriness/wavelength — **IF**
  the tissue factory threads cell type (verified: LBM forces ENDO; the monodomain factory derives cell
  type from mesh `group_cell_types`, so the clean way to set it in tissue is passing a pre-built model
  instance). Drop this beat if it's not cleanly settable — it's optional.

**5.4 — Numerical hygiene: trust your tuned number.** Not physiology, but a hand-tuner *must* know it.
- **Grid spacing dx** — the trap. Too coarse relative to the ~0.5–1 mm upstroke (≈ a handful of nodes)
  → **grid-dominated, WRONG CV that can look like block** — a numerical artifact, not physiology.
  Refine dx until CV stops moving (convergence); tune only in the converged regime. (This is the real
  ionic-tuner "phantom conduction block" failure — worth one honest paragraph.)
- **Time step dt** — accuracy, not stability (CN is unconditionally stable); a too-large dt shifts CV a
  few %. Show a small dt sweep so the reader knows how much their number can wander.
- **Engine choice** — mono / bidomain / LBM give *different absolute CV* for the same σ (numerics), so
  **compare like-to-like** — an engine against itself across conditions, never one engine's number to
  another's. (Closes the loop with Ch1's three-engine mention and Ch3's per-engine examples.)

Keep every sweep on one small tissue and a handful of values. Pure `cardiac_core` — no external package.

---

## 3. Verified API surface (use these exact calls; do not generate against stubs)

Confirmed live on `main` this session.

- **Geometry / conductivity.** `cc.Grid(Nx, Ny, dx)` — `dx` in **cm**, `Lx = dx·(Nx−1)`.
  `cc.ConductivityConfig.bidomain(sigma_i, sigma_e, chi=1400.0)`, `.isotropic(...)`; anisotropic
  (Ch3 fibers) is `anisotropic(sigma_l, sigma_t, fiber_angle, chi=1400.0, Cm=1.0)` — raw σ mS/cm,
  angle **radians**, ONE global angle; `sigma_eff`/`D_eff` return a **3-tuple (xx,yy,xy)** — don't
  `float()`. Masks: `cc.rectangle_mask`, `cc.circle_mask`.
- **Ionic models.** `"ttp06"` (all engines; Ch1–2), `"paci"` (hiPSC self-pacing; Ch2A — `"phas13"` is
  the **same** `PHAS13Model`, `"mhas13"` a distinct variant), `"ord"` (**LBM only**).
- **Single cell (Ch1, 2, 4.1).** `cc.single_cell('ttp06', celltype='EPI', pre_pace=5, bcl=…,
  n_beats=…)` → `sc.V`, `sc.times`, `sc.apd(0.9)`, `sc.v_peak`, `sc.v_rest`, `sc.final_state`,
  `sc.trace()` (→ `ImageInfo`). `celltype` ∈ ENDO/EPI/**`M_CELL`** — ⚠️ the string is **`'M_CELL'`**,
  NOT `'MID'`/`'M'` (verified: `celltype='MID'` raises `AttributeError`; the `single_cell` *docstring*
  carries the same `MID` typo — flag for the single_cell owner). 0-D pacing via `bcl`+`n_beats`+`pre_pace`.
  **`conductances={NAME: factor}` is shipped** (`37f8d57`; Ch2's drug knob) — multiplicative, applied
  *before* `pre_pace`, deep-copies the model, and **raises `ValueError` on an unknown/mis-cased name**.
- **Stim (Ch1, 3, 4).** `cc.Stim.boundary(g, "left"/"right"/"top"/"bottom", amplitude=-52.0,
  start_time=…, duration=…, bcl=…, num_pulses=…)`, `.point(g,(x,y))`, `.center(g)`, `.from_region`,
  `cc.Stim(mask)`. Clamp mode: `cc.Stim(mask, clamp=-20, duration=…)`. A list of Stims for
  multi-site (reentry S1–S2 in Ch4.2).
- **Engines / run (Ch1, 3, 4).** `cc.monodomain/bidomain/lbm(g, model, cond, stim, dt=…)`;
  `sim.run(t_end, save_every=…, batch=…)`. Heterogeneity: `sim.scale_conductance("GKr", 0.5)`,
  `sim.set_conductivity(mask, D=0.0)` (scar), `sim.scale_conductivity(mask, f)`.
- **Measure (Ch3, 4, 5).** `r.cv(x1,x2,y)`, `r.apd()`, `r.lat()`, `r.restitution(ix,iy)`,
  `r.apd_per_beat`, `r.restitution_slope` (`{max_slope, DI_star, n}`), `r.radial_cv`, `r.cv_between`,
  `r.df_map`; `cc.wavelength`, `cc.di`; fields `r.fields.source_sink/.velocity/.curvature`,
  `cc.safety_factor(r)`. Reentry (Ch4.2): `cc.phase_map(Vm,times,t_idx)` + `cc.phase_singularities`.
- **Media (Ch1 + throughout).** `r.video()` displays inline / `r.video("slug", bulk=True)` saves;
  `r.image(what="activation"/"apd"/…)` for a static map OR `r.image(at=<ms>)` for a voltage snapshot
  (they are **separate** calls — `at=` raises on a `what=` map); `r.trace(at=…, what="restitution")`; `.show()`,
  `.save()`; `Gradient.physiological()/…`; multi-panel `render([Video.annotated(...), …])`. Colab needs
  `cardiac-core[viz]`.
- **Tuning by hand (Ch5).** Pure `cardiac_core`, **no external package / no optimizer**. Knobs:
  `ConductivityConfig.bidomain/isotropic/anisotropic(…, chi=…, Cm=…)` (σ, D, χ, Cm, fiber_angle);
  `cc.Grid(Nx, Ny, dx)` (spatial resolution); `dt=` on the engine factory; `Stim(amplitude=…)` +
  site size; `set_conductivity`/`scale_conductivity` (scar); `scale_conductance` (channels). Measures:
  `r.cv`/`r.radial_cv`/`r.image(what="activation")`/`cc.wavelength`/`cc.di`/`cc.safety_factor`/
  `r.fields.source_sink`.

---

## 4. Phases (each chapter/sub-chapter = one deliverable + one commit)

### Phase P0 — prep (once)
- [ ] Confirm the gate: `Video`, `Gradient`, `render`, `single_cell`, `safety_factor`,
      `rectangle_mask`, `circle_mask` import from `cc` (verified 2026-07-28).
- [ ] Reuse the shipped builder pattern — do NOT introduce `nbformat`. Copy
      `tutorials/_build/build_01_build_a_simulation.py` (`CELLS` list → JSON emit + `--script`) per notebook.
- [ ] Coordinate the `nbconvert` install with the parallel agent (shared env) before Phase W.
- [ ] For every chapter, run its exact calls against the live API and record real numbers (APD90,
      per-channel morphology deltas, CV per engine, restitution slope) so prose states measured values.

### Authoring order
**Ch1 (rewrite) → Ch2 → Ch3 → Ch4 → Ch5.** Per notebook, produce:
1. `_build/build_NN_slug.py` (reviewable source — the `.py` diff is what gets read).
2. Headless verify: `python _build/build_NN_slug.py --script /tmp/x.py` then
   `conda run -n heart-conduction python /tmp/x.py` → exit 0.
3. **Render and LOOK AT every figure** (not just "no exception").
4. The `.ipynb` (empty outputs until `nbconvert`).
5. 1–2 "Try it yourself" edit-a-number exercises with expected outcomes.
6. A `README.md` row (chapter/sub-chapter, one-line objective, runtime).
7. Commit `docs(tutorials): chapter NN — <slug>`.

Per-chapter notes beyond §2/§3:
- **Ch1** — see §2 detail. Prefer `sc.trace()`/`r.image()`/`r.video()` over raw matplotlib.
- **Ch2** — one figure per channel; verify each morphology claim by running it. The `conductances=`
  knob is shipped (`37f8d57`), so Ch2 is fully unblocked. Also teach the knob's own behaviour (drug
  applied before `pre_pace`; name-validation raises) — see §2 detail. Split 2A (self-pacing) out.
- **Ch3** — the deep tissue chapter. Author 3.1 (monodomain) first; add 3.2 (bidomain, incl. `phi_e`)
  and 3.3 (LBM) as separate notebooks **only if 3.1+all-engines overflows** ~90 s / a long scroll;
  otherwise one notebook with three engine sections. Introduce `mask`/scar here.
- **Ch4** — 4.1 single-cell: a `bcl` sweep of `single_cell` runs → effective BCL, restitution, alternans.
  ⚠️ `restitution_slope`/`DI_star` are **tissue-`r` hooks**, NOT on `SingleCellResult` (which has only
  `V/times/apd/v_peak/v_rest/final_state/trace`) — assemble the single-cell restitution curve from the
  per-BCL `sc.apd()` values (or `cc.restitution_curve`/`cc.di`), not `r.restitution_slope`. 4.2 tissue
  reentry: an S1–S2 cross-field protocol (list of Stims) → a rotor; verify with `phase_singularities`.
  Keep the reentry grid small; this is the runtime risk. (Tissue pacing 4.2 CAN use `r.restitution_slope`.)
- **Ch5** — see §2 detail (now a full catalog in 4 groups: 5.1 coupling σ/D/χ/Cm · 5.2 geometry
  anisotropy/scar/stim-site · 5.3 cellular conductances/cell-type · 5.4 numerical dx/dt/engine). Each a
  small sweep → its tissue-level effect; verify by running. Author 5.1 first; split 5.2–5.4 into separate
  notebooks only if it overflows. Pure by-hand, no optimizer. The payoff chapter — reuse Ch2's channel
  names and Ch3's configs. **The `dx` under-resolution beat (5.4) is the load-bearing one** — it's the
  real "phantom conduction block" gotcha; don't cut it.

### Phase W — wrap
- [ ] `README.md` full index (chapters + sub-chapters, objectives, runtimes).
- [ ] Execute-all anti-rot gate (needs `nbconvert`): pytest / `run_all_tutorials.sh` running
      `nbconvert --to notebook --execute` on every notebook, wired into `/verify`.
- [ ] Re-emit shipped notebooks WITH executed outputs once `nbconvert` lands.
- [ ] Near-beginner difficulty read of Ch1–2; cross-link from `API_CHEATSHEET.md`.

---

## 5. Dependencies & gate

| Chapter | Depends on | Status |
|---------|-----------|--------|
| 1 | `single_cell`, `sc.trace()`, `r.image`/`r.video`, `monodomain` | **✅ READY** — on `main` |
| 2 | `single_cell(conductances={...})` | **✅ SHIPPED** — committed `37f8d57`; verified functional (validates names, applied before pre_pace) |
| 2A | `"paci"` self-pacing (verified: paces at `stim_amplitude=0`); `conductances={'g_f': …}` | **✅ SHIPPED** — same `conductances=` knob (`37f8d57`); Paci names are lowercase `g_*` |
| 3 | `Grid`/`ConductivityConfig`/`Stim`/masks/all engines | **✅ READY** |
| 4 | `Stim(bcl,num_pulses)`, `restitution_slope`, `phase_singularities` | **✅ READY** |
| 5 | `ConductivityConfig` (iso/aniso), `set/scale_conductivity`, `scale_conductance`, `r.cv`/`radial_cv`/`wavelength`/`safety_factor` | **✅ READY** — pure `cardiac_core`, no optimizer |
| W gate | `nbformat` + `nbconvert` | **⚠️ NOT INSTALLED** |

`tutorial_helpers.py` — default to NOT writing one; `sc.trace()`/`r.image()`/`r.video()` are the plotting surface.

---

## 6. Open questions for the redline
*(Resolved 2026-07-28: Ch5 = pure by-hand parameter→tissue-effect, no Optimizer; fibers/anisotropy
lands on the spine — mechanics in Ch3, effect in Ch5. Ch2 drug-knob timing is MOOT — `conductances=`
has landed, so Ch2 is unblocked and can be authored right after Ch1.)*
1. **Voltage clamp** — still homeless. A protocol beat in Ch2 (a clamped single-cell measurement) or in
   Ch4 (a clamp-based pacing/refractoriness protocol), or drop from v1? (My lean: a short Ch4 beat.)
2. **Ch1 tissue example** — monodomain only, kept black-box (my plan), or a hair more (mention CV)?
3. **Ch1 title/filename** — `01_intro.ipynb` "Cardiac Core Intro" vs "Meet Cardiac Core"; delete the
   old `01_build_a_simulation.ipynb` or keep a redirect stub.
