# PLAN: Figure-4C/D 2-D Reproduction — Four-Condition Campaign

Created: 2026-06-08
Engine(s): Monodomain V5.4 (primary), LBM V1 (cross-check), cardiac_core (shared analysis)
Research question: [source_sink_mismatch_investigation](README.md)
Source: [FIG4C_BLOCK_TEST_PLAN.md](FIG4C_BLOCK_TEST_PLAN.md) + [IDEALOG.md](IDEALOG.md) (2026-06-08 corrected error-trace)

## Objective
Determine whether our 2-D engine can reproduce Ciaccio 2018 Fig-4C (convex slowing) and
4D (functional block) from **in-plane** source-sink mismatch — no thickness. Test the four
necessary conditions our hourglass runs broke (resolve `r*`, current-limited source,
excitability, measure `CV_n` vs `kappa`), each tuning a parameter or returning a
prove/disprove verdict, ending in a global CAN / CAN-WITH / CANNOT verdict.

## Success Criteria
- [ ] Eikonal coupling validated: `CV_n = CV0 - D*kappa` recovered (R^2>0.95, slope≈D±20%); `r*` measured
- [ ] `dx_resolved` found where `r_crit` is grid-converged (`r*/dx >= 3`), OR documented unreachable
- [ ] Regime isolated: current-limited strand→bulk slows where reservoir hourglass does not
- [ ] Functional block (Fig-4D) produced by lowering excitability, with parameter recipe — OR disproved in physiological range
- [ ] Fig-4 A–D panels + `theta = theta0 - D*(dW/W)` reproduced
- [ ] Global verdict recorded in KNOWLEDGE.md; all existing tests pass (no regressions)

## Architecture Changes
- NEW: `cardiac_core/analysis/eikonal.py` — front normal/`CV_n`/`kappa` extraction + eikonal fitter + synthetic self-test (cross-engine, reused by every stage)
- NEW: `Monodomain/Engine_V5.4/experiments/fig4c_sourcesink/` — `run_s0..s4.py`, `EXPERIMENT.md` (backlinks to this question + MASTER)
- REUSE: question-folder `diag_eikonal_circle.py` (expanding-circle/LAT harness, both engines) and `diag_hourglass.py` as templates — copy patterns, do not import from Research/

## Known Failures (from IDEALOG — do NOT retry)
- **Thickness-weighted / augmented monodomain pivot** — based on misreading Fig 5 (thickness = IBZ measurable proxy) as the Fig-4 mechanism (in-plane cross-section curvature). Thickness is NOT a variable in this plan.
- **2-D in-plane width as a "dead end"** — false; it IS the faithful realization. The failure was conditions, not dimensionality.
- **Expecting an inverse crescent at a converging funnel** — convergence is correctly flat (walls reflect current in); a funnel clips, not collapses, a planar front. Don't treat its absence as model failure.
- **Reservoir-fed hourglass to get block** — over-drives a clamped source (Fig-4A/B regime); cannot starve it. Use current-limited strand→bulk.
- **Reading CV from axial LAT only** — conflates kinematic Huygens fan with dynamic `-D*kappa`. Always use the §Phase-1 normal/kappa pipeline.

---

## Phase 1: Measurement pipeline + eikonal validation (Conditions 4, and measure r*)

**Goal**: A validated `CV_n`-vs-`kappa` analysis pipeline, plus measured `CV0, D_eik, r*` from a clean expanding circle. Foundation for every later stage.
**Tier**: medium
**Estimated scope**: one shared analysis module (+self-test) and one S0 run script per engine.

### Phase Context
- Run harness pattern is in `diag_eikonal_circle.py` (read it). Monodomain: `StructuredGrid.from_mask`, `FDMDiscretization(grid, D=0.001, chi=1, Cm=1, stencil='cardinal4', boundary_mode='face_mirror')`, `MonodomainSimulation(..., ionic_model='ttp06', dt=0.02, splitting='strang', ionic_solver='rush_larsen', diffusion_solver='forward_euler', cell_type='EPI')`, `sim.run_to_array(t_end, save_every)`. THR=-40, V_rest≈-85.23. LBM: hand loop with `bgk_collide`/`stream_d2q9`, D via `omega=1/tau_from_D(D,DX,DT,cs2)`.
- `lat_field(V, times, thr)` (in diag_eikonal_circle.py) computes interpolated LAT — lift it into the shared module.
- Figures/videos via `from cardiac_core.media import media_path` → `media/source_sink_mismatch_investigation/...`. Bulk arrays → `media/.../_sim_outputs/` (gitignored).
- All tensors float64. Do NOT modify V5.3.

### Step 1.1: Shared eikonal-metrics module + self-test
**Model**: opus

#### Read First
- `Research/Active/source_sink_mismatch_investigation/diag_eikonal_circle.py:42-54` — `lat_field` LAT extraction
- `cardiac_core/media.py` — `media_path` signature

#### Why
Every stage's verdict depends on cleanly separating dynamic slowing (`-D*kappa`) from kinematic fanning. A wrong pipeline silently invalidates the whole campaign — so it is gated on a synthetic test with a known answer before any physics run.

#### Implementation Spec
**Files to create:** `cardiac_core/analysis/eikonal.py`
**Interfaces:**
- `lat_field(V, times, thr=-40.0) -> np.ndarray`  (lifted from diag script)
- `front_metrics(lat, dx) -> dict(cv_n, kappa, n_hat)` — `n_hat = grad(LAT)/|grad(LAT)|`; `cv_n = 1/|grad(LAT)|`; `kappa = div(n_hat)` (sign + = convex)
- `fit_eikonal(cv_n, kappa, mask=None) -> dict(CV0, D_eik, r2, r_star)` — linear regress `cv_n ~ kappa`; `r_star = D_eik/CV0`

#### Pseudocode
```
gx,gy = np.gradient(lat, dx)            # ms/cm
speed = 1/hypot(gx,gy)                  # cm/ms = CV_n
nx,ny = -gx/|g|, -gy/|g|               # propagation dir
kappa = div([nx,ny]) via np.gradient
fit: CV0,D = polyfit(kappa, cv_n, 1)[::-1]; r_star = D/CV0
```

#### Test Spec
- `tests/test_eikonal_metrics.py::test_radial_synthetic` — Setup: `lat = r/CV0` for CV0=0.06 on a 50µm grid, point at center. Expected: `kappa ≈ 1/r` and `cv_n ≈ 0.06` (flat) to <2%; `fit_eikonal` returns CV0≈0.06, D_eik≈0.
- `::test_planar_synthetic` — Setup: `lat = x/CV0`. Expected: `kappa≈0`, `cv_n≈CV0`.

#### Checklist
- [x] module + functions, float64 — added to `cardiac_core/analysis.py` (`activation_time_interp`, `front_metrics`, `fit_eikonal`)
- [x] self-tests pass — `cardiac_core/tests/test_eikonal_metrics.py` (3 pass)
- [x] no duplication: scripts import these (not a new subpackage; extended existing `analysis.py`)

#### Verify
`/opt/miniforge3/bin/conda run -n heart-conduction python -m pytest cardiac_core/tests/test_eikonal_metrics.py -v`
(NB: `conda` not on PATH in non-interactive shells — use full path `/opt/miniforge3/bin/conda`)

#### Exit Criteria
- [x] Synthetic radial + planar tests pass (3/3); existing analysis tests still pass (16/16)
**MUTATED 2026-06-08**: Step 1.1 — eikonal fns appended to existing `cardiac_core/analysis.py` instead of new `analysis/eikonal.py` subpackage (analysis is a single module; avoids duplication of existing `activation_time`/`conduction_velocity`).

#### Risk
`np.gradient` noise near the wavefront edge — mitigation: mask to cells with finite LAT and `|grad|>eps`; smooth `n_hat` with a 3x3 mean before `div`.

### Step 1.2: S0 eikonal validation run (measure CV0, D_eik, r*)
**Model**: opus

#### Read First
- `diag_eikonal_circle.py:114-137` (mono run), `:58-110` (LBM run)

#### Why
Confirms the engine has the curvature–CV coupling and gives the empirical `r*` that sets the S1 resolution target. The −0.14 ms inverse crescent says the coupling exists; this quantifies it.

#### Implementation Spec
**Files to create:** `Monodomain/Engine_V5.4/experiments/fig4c_sourcesink/run_s0_eikonal.py`, `EXPERIMENT.md`
Geometry: uniform sheet ≥8mm, small-disk stimulus at center → expanding circular wave. dx=50µm (fine; not resolution-limited here). Run mono (and LBM cross-check).

#### Pseudocode
```
build uniform grid (no obstacle mask)
stim: disk region |x-cx,y-cy|<r_stim, amplitude=-52, dur=2
run_to_array -> times,V ; lat=lat_field ; m=front_metrics ; fit=fit_eikonal
save CV vs kappa scatter+fit line -> media images ; print CV0,D_eik,r2,r_star
```

#### Test Spec
- Acceptance (not pytest): R^2>0.95, `D_eik` within ±20% of 0.001, report `r_star`.

#### Checklist
- [x] run_s0 + EXPERIMENT.md backlinks present
- [x] CV-vs-kappa figure to media (`s0-eikonal-cv-vs-kappa-mono_06.png`)
- [x] recorded CV0=62.4cm/s, D_eik=0.00084, r*≈134µm into FIG4C_BLOCK_TEST_PLAN.md §10

#### Verify
`/opt/miniforge3/bin/conda run -n heart-conduction python Monodomain/Engine_V5.4/experiments/fig4c_sourcesink/run_s0_eikonal.py`

#### Exit Criteria
- [x] Integrated-form fit passes (R²=0.99997, D_eik within 16% of D); `r*≈134µm` measured & recorded

#### Risk
If fit fails (FAIL branch): numerics/measurement broken → STOP, debug pipeline before S1.
**MUTATED 2026-06-08**: Step 1.2 — (a) used `moore8_iso` not `cardinal4` (anisotropy swamped the curvature signal); (b) acceptance via integrated `LAT(r)=r/CV0+(D/CV0²)ln r+c` fit, not dr/dLAT differentiation (low-SNR); (c) added npz cache under `_sim_outputs/` (gitignored). **Finding for S1:** r*≈134µm → dx ≤ ~45µm; S0's 50µm is borderline, sweep to ~35µm.

### Phase 1 Verification
`/opt/miniforge3/bin/conda run -n heart-conduction python -m pytest cardiac_core/tests/test_eikonal_metrics.py -v` (3 pass) and S0 acceptance met (PASS).
### Phase 1 Exit Criteria
- [x] pipeline self-tests pass (3/3); existing analysis tests pass (16/16); eikonal coupling confirmed; `r*≈134µm` locked
### Phase 1 Cleanup
- float64 everywhere; V5.3 untouched; EXPERIMENT.md backlinks present; `lat_field`/metrics single-sourced in cardiac_core (no copy in run scripts)
**→ Commit point: git commit after Phase 1**

---

## Phase 2: Resolution — can block-scale curvature be represented? (Condition 1)

**Goal**: Find `dx_resolved` (grid-converged `r_crit`, `r*/dx>=3`) or prove block-curvature is unreachable at feasible dx.
**Tier**: medium

### Phase Context
Critical-nucleus probe: stimulate a disk of radius `r_stim`; below `r_crit` the wave collapses (curvature block), above it expands. Theory `r_crit ~ r*`. Coarse dx cannot represent collapse → spurious propagation of sub-critical nuclei is the failure signature.

### Step 2.1: Critical-nucleus dx sweep
**Model**: opus
#### Read First
- `cardiac_core/analysis/eikonal.py` (Phase 1), `run_s0_eikonal.py`
#### Why
This is the make-or-break numerical test: if `r*` (~200µm) is sub-grid at every feasible dx, block cannot form regardless of geometry/excitability — a real finding that redirects the whole question.
#### Implementation Spec
**Files to create:** `.../fig4c_sourcesink/run_s1_nucleus.py`
2-D matrix: `r_stim ∈ [0.5,3]*r*` (~8), `dx ∈ {200,100,70,50,35} µm`. Per cell: propagate? measure `r_crit(dx)`.
#### Pseudocode
```
for dx: for r_stim: run short sim; classify propagate/collapse via final activated area
  r_crit(dx) = smallest r_stim that propagates
converged dx = coarsest dx where |r_crit(dx)-r_crit(dx/2)|/r_crit < 10%
```
#### Test Spec
- Acceptance: `r_crit(dx)` table; pick `dx_resolved` with `r*/dx_resolved>=3`.
#### Checklist
- [ ] sweep script; r_crit-vs-dx plot to media; record `dx_resolved` in §10
#### Verify
`conda run -n heart-conduction python .../run_s1_nucleus.py`
#### Exit Criteria
- [ ] `dx_resolved` locked, OR documented "no convergence at feasible dx" (CANNOT branch)
#### Risk
Fine dx is costly — mitigation: small domains (4–6 mm), short t_end, GPU; only the expansion neighborhood needs fine dx (consider local patch later).

### Phase 2 Verification / Exit / Cleanup
- [ ] verdict recorded; no regressions; EXPERIMENT.md updated
**→ Commit point after Phase 2**

---

## Phase 3: Source regime — current-limited vs reservoir (Condition 2)

**Goal**: Show convex slowing appears with a current-limited strand→bulk but not with the reservoir hourglass at matched expansion ratio.
**Tier**: medium

### Phase Context
At `dx_resolved`, healthy excitability. G-hourglass (wide→1-cell neck→wide) clamps/over-drives the neck (expected: no slowing). G-strand (long thin strand, no wide backing → abrupt bulk) starves the source. Reuse `diag_hourglass.py` geometry construction.

### Step 3.1: Hourglass vs strand head-to-head
**Model**: opus
#### Read First
- `diag_hourglass.py` (geometry+run), Phase-1 metrics
#### Why
Isolates "regime" as the causal variable, directly correcting the hourglass mistake.
#### Implementation Spec
**Files to create:** `.../run_s2_regime.py`. Build both masks; sweep strand width `w∈{1,2,4,8}` cells and bulk opening angle. Measure junction `CV_n` dip + safety factor.
#### Pseudocode
```
for geom in [hourglass, strand]: for w,angle: run; lat; metrics
  record CV_n at junction, min CV_n downstream, SF
compare: strand shows CV_n dip where hourglass ~CV0
```
#### Test Spec
- Acceptance: measurable convex slowing in strand absent/weaker in hourglass at same ratio.
#### Checklist
- [ ] both geometries; CV_n maps to media; record `w*`, exp-ratio* of slowing onset
#### Verify
`conda run -n heart-conduction python .../run_s2_regime.py`
#### Exit Criteria
- [ ] regime dependence demonstrated (or both flat → defer to Phase 4 excitability)
#### Risk
Healthy SF may keep even strand from slowing visibly — acceptable; carry geometry to Phase 4.

### Phase 3 Verification / Exit / Cleanup → Commit point

---

## Phase 4: Excitability → block + full Fig-4 A–D (Conditions 3 + synthesis)

**Goal**: Drive convex slowing into functional block by lowering safety factor; reproduce panels A–D and recover the eikonal law + cross-section form. Emit global verdict.
**Tier**: large

### Phase Context
`dx_resolved`, G-strand from Phase 3. Lower excitability via gNa scale, [K]o, IK1, and gap-junction (D) uncoupling. Block = downstream bulk never activates (double-line LAT). Then assemble A (concave-forcing geometry — notch/converging-to-focus, NOT a funnel), B (straight), C (mild expansion sub-threshold), D (sharp expansion / low SF).

### Step 4.1: Excitability sweep to block
**Model**: opus
#### Read First
- `diag_hourglass.py`, ionic model knobs in `ionic/ttp06/` (gNa, IK1, [K]o); confirm how to scale in V5.4 path
#### Why
Excitability is the dominant remaining knob; the IBZ is diseased, not healthy.
#### Implementation Spec
**Files to create:** `.../run_s3_excitability.py`. Sweep `gNa∈{1,.75,.5,.35,.25}`, `[K]o∈{5.4,6.5,8,10}`, `D∈{1,.5,.25}×`. Map (expansion-ratio × excitability) block boundary.
#### Pseudocode
```
for knob in sweep: run strand→bulk; detect block (downstream activated area≈0)
  record SF, CV_n(junction); build phase boundary block/no-block
```
#### Test Spec
- Acceptance: at least one physiological-range setting yields block; record SF*, gNa*, [K]o*.
#### Checklist
- [ ] block detector; phase-boundary plot to media; thresholds → §10
#### Verify
`conda run -n heart-conduction python .../run_s3_excitability.py`
#### Exit Criteria
- [ ] block achieved (CAN) or disproved in physiological range (CAN-WITH/CANNOT) — recorded
#### Risk
Non-physiological extremes only → document as CAN-WITH; revisit dx (Phase 2).

### Step 4.2: Fig-4 A–D panels + eikonal/cross-section close-out
**Model**: opus
#### Read First
- `cardiac_core/analysis/eikonal.py`; §8 of FIG4C_BLOCK_TEST_PLAN.md
#### Why
The publishable artifact: the four panels + quantitative `CV_n=CV0-D*kappa` and `theta=theta0-D*(dW)/(c*W)`.
#### Implementation Spec
**Files to create:** `.../run_s4_figure4.py`. Four geometries/knobs → A,B,C,D. Plot CV vs kappa (slope=D from S0) and CV vs relative width gradient `dW/W`; locate block threshold; compare to Ciaccio ~2/mm.
#### Test Spec
- Acceptance: A `kappa<0,CV>CV0`; B flat; C `kappa>0,CV<CV0`; D block. Eikonal slope matches S0 `D_eik`.
#### Checklist
- [ ] A–D composite figure + both quantitative plots to media
- [ ] global verdict (CAN/CAN-WITH/CANNOT) written to KNOWLEDGE.md with §10 registry
#### Verify
`conda run -n heart-conduction python .../run_s4_figure4.py`
#### Exit Criteria
- [ ] all four panels reproduced (subject to Step 4.1 verdict); eikonal law recovered
#### Risk
Panel A (concave speedup) needs a concave-front-forcing geometry — a plain constriction will read flat (expected); use a notch/focus.

### Phase 4 Verification / Exit Criteria
- [ ] verdict + parameter recipe in KNOWLEDGE.md; A–D + quantitative plots in media; existing tests pass
### Phase 4 Cleanup
- float64; V5.3 untouched; EXPERIMENT.md backlinks; shared analysis in cardiac_core; gitignore bulk `_sim_outputs/`
**→ Commit point after Phase 4**

---

## Final Cleanup
1. Archive plan:
```bash
mkdir -p Research/Active/source_sink_mismatch_investigation/plans
cp Research/Active/source_sink_mismatch_investigation/PLAN.md "Research/Active/source_sink_mismatch_investigation/plans/$(date +%Y-%m-%d)_fig4c-block-four-condition.md"
```
2. Update README completion criteria + MASTER.md row to corrected (in-plane) framing.
3. Revert tmux pane to WHITEBOARD.md (see skill).
- float64 consistency; V5.3 read-only; EXPERIMENT.md backlinks; no cross-engine duplication (analysis in cardiac_core).

## Mutation Log
(empty — populate during execution: `**MUTATED {date}**: Step X.Y {SKIPPED|SPLIT|INSERTED} — {reason}`)
