# Figure-4C/D Reproduction — Systematic Test Plan

> Can our 2-D engine reproduce Ciaccio 2018 **Figure 4C (convex slowing)** and
> **4D (functional block)** from in-plane source-sink mismatch alone — no thickness?
> This plan tests the **four necessary conditions** that our earlier hourglass runs
> all violated, isolating each so it either **TUNES a parameter** for the next stage
> or returns a **PROVE / DISPROVE** verdict on engine capability.
>
> Backlinks: [README](./README.md) · [KNOWLEDGE](./KNOWLEDGE.md) ·
> [project MASTER](../../../MASTER.md)
> Origin: corrected error-trace 2026-06-08 (see KNOWLEDGE "Error trace"; memory
> `feedback_ciaccio_fig4_mechanism_not_thickness`). Thickness is NOT a variable here.

---

## 0. The claim under test

Figure 4C/D are governed by the eikonal relation

```
   CV_n = CV0 - D * kappa          CV_n = front speed along its normal
                                   kappa = front curvature (>0 convex)
```

C = regime where `D*kappa` is a sizable fraction of `CV0`; D = pushed to `CV_n -> 0`.
The engine has the right PDE (the `diag_eikonal_circle.py` leading inverse crescent of
-0.14 ms proves the curvature-CV coupling is alive). The question is whether our
**runs** can be put in C/D's regime, at adequate resolution, and measured. Four
conditions; our hourglass broke all four:

| # | Condition | How we broke it | Stage |
|---|-----------|-----------------|-------|
| 1 | resolve the curvature length `r* = D/CV0 ~ 200 um` | dx = 250-500 um -> r*/dx < 1, block-curvature sub-grid | S1 |
| 2 | source must be current-LIMITED (source<sink) | hourglass = wide reservoir clamps neck -> over-driven (4A/B regime) | S2 |
| 3 | excitability low enough (diseased IBZ) | used healthy TTP06 EPI, SF ~1.5-2 | S3 |
| 4 | measure `CV_n` vs `kappa`, not axial LAT | only saw the kinematic Huygens fan | S0 (instrument) |

**Global verdict produced by this plan:** `CAN` / `CAN-WITH-conditions(dx,SF,geometry)`
/ `CANNOT-at-feasible-dx` (a real, publishable numerical-limit finding either way).

---

## 1. Engines, geometries, params

- **Primary:** Monodomain V5.4 FDM, explicit Euler + Rush-Larsen, TTP06 EPI. Use the
  validated `run_monodomain_fdm()` path (see `cv_shared.py`; confirm API before running).
- **Cross-check:** LBM V1 (D2Q5 BGK). Note CV is ~35% higher (different numerics) — use
  for qualitative corroboration, not quantitative D.
- **Grid/stim API** (verify, don't trust from memory): `StructuredGrid.create_rectangle(
  Lx,Ly,Nx,Ny,device='cuda')`; `Stimulus(region=lambda x,y:..., start_time, duration,
  amplitude)`. float64, GPU.
- **Reference D, CV0 (estimates to be MEASURED in S0):** D ~ 0.0012 cm^2/ms,
  CV0 ~ 0.056 cm/ms (mono) -> `r* ~ 210 um`. LBM CV0 ~ 0.076 -> r* ~ 160 um.
- Reuse existing semicircle/obstacle assets where possible (`sim_semicircle_obstacle.py`,
  `data/sim_semicircle_*`).

**Figures/videos** -> `media/source_sink_mismatch_investigation/{images|videos}/{date}/`
via `cardiac_core.media.media_path`. Bulk sim output -> `media/.../_sim_outputs/` (gitignored).

---

## 2. Measurement pipeline  (Condition 4 — built FIRST; every stage depends on it)

Without this, no result is interpretable (the fan conflates kinematic spreading with
dynamic `-D*kappa` slowing). Build and unit-test before any physics run.

**Inputs:** activation-time field `LAT(x,y)` (time V crosses -40 mV) + V snapshots.
**Compute:**
1. front normal `n_hat = grad(LAT)/|grad(LAT)|`
2. local conduction velocity `CV_n = 1/|grad(LAT)|`
3. front curvature `kappa = div(n_hat)` (sign: + for convex/expanding front)
4. eikonal fit: regress `CV_n` on `kappa` -> slope `= D_eik`, intercept `= CV0`.

**Self-test (must pass before S0 physics):** synthetic radial LAT `LAT=r/CV0` on the grid
-> pipeline must return `kappa=1/r` and flat `CV_n=CV0` to <2%. Gate on this.

**Deliverable:** `analysis/eikonal_metrics.py` (front normal/CV/kappa + fitter + self-test).

---

## 3. Controls (run once, both engines)

| Control | Geometry | Expect | Catches |
|---|---|---|---|
| C0 planar | uniform sheet, line stim | flat front, `kappa=0`, `CV_n=CV0` everywhere | baseline / measurement bias |
| C1 converging | symmetric funnel, planar in | flat (`kappa~0`), `CV_n~CV0` — NO inverse crescent | confirms convergence asymmetry (correct physics, not a bug) |

C1 documents that "constriction flat" is expected, closing the prior misinterpretation.

---

## 4. STAGE 0 — Eikonal validation: measure D, CV0, r*  (Condition 4)

**Hypothesis:** the engine obeys `CV_n = CV0 - D*kappa`; we can measure `D, CV0` and
hence `r* = D/CV0` empirically (not just estimate).

**Setup:** uniform sheet, point/small-disk stimulus -> expanding circular wave. `kappa=1/r`
sweeps continuously from large (near stim) to ~0 (far). Resolution dx = 50 um (fine, so
this stage is not resolution-limited). Domain >= 8 mm so r spans ~0.1-3 mm.

**Sweep:** none (single run per engine); extract `CV_n(kappa)` across radii.

**Observable / Pass:** linear `CV_n` vs `kappa`, R^2 > 0.95; slope `D_eik` within +-20%
of the operator `D`; report measured `r* = D_eik/CV0`.

**Outcome:**
- PASS -> curvature-CV coupling confirmed; **lock measured `r*`** -> sets dx target for S1.
- FAIL (no slope / nonlinear) -> numerics/measurement broken; STOP and debug before S1.
  (Disproves the premise at the most basic level — a finding in itself.)

---

## 5. STAGE 1 — Resolution: can block-scale curvature be represented?  (Condition 1)

**Hypothesis:** `r*` must be resolved by >= 3-4 cells; at dx >= r* the engine spuriously
conducts because it cannot represent the block-causing high-curvature region.

**Setup — critical-nucleus test (cleanest κ-block probe):** stimulate a disk of radius
`r_stim` in a uniform sheet; below a critical `r_crit` the wave collapses (fails to
expand = curvature block), above it propagates. Theory: `r_crit ~ r*`.

**Sweep (2-D matrix):**
- `r_stim`: bracket `r_crit` (e.g. 0.5*r* .. 3*r*, ~8 values)
- `dx`: 200, 100, 70, 50, 35 um

**Observable:** measured `r_crit(dx)`; does it converge as dx shrinks? At coarse dx, is a
sub-critical nucleus spuriously propagated (engine can't represent collapse)?

**Pass / tuning:**
- Define `dx_resolved` = coarsest dx where `r_crit` changes < 10% vs dx/2 (grid-converged).
- Require `r*/dx_resolved >= 3`. **Lock `dx_resolved`** for S2-S4 (use local refinement at
  the expansion if uniform fine grid is too costly).

**Outcome:**
- Converges at feasible dx -> Condition 1 satisfiable; proceed.
- `r_crit` keeps shrinking / always spurious conduction down to smallest feasible dx ->
  **DISPROVE at feasible resolution** for this scheme: report the numerical-limit finding;
  consider (a) higher-order / adaptive operator, (b) accept block is unreachable at our dx.

---

## 6. STAGE 2 — Source regime: current-limited vs reservoir-fed  (Condition 2)

**Hypothesis:** C/D need an intrinsically current-limited source. Hourglass
(reservoir->neck) clamps the neck near plateau -> over-driven (4A/B); strand->bulk
(no wide backing) starves the source -> 4C/D.

**Setup:** at `dx_resolved`, healthy excitability. Two geometries, matched expansion ratio:
- **G-hourglass:** wide -> 1-cell neck -> wide (the old setup; expected: no slowing).
- **G-strand:** long thin strand (width w) -> abrupt wide 2-D bulk (Fast-Kleber).

**Sweep:** strand width `w` (1,2,4,8 cells); expansion ratio / bulk opening angle.

**Observable:** junction `CV_n` dip; safety factor SF at junction; does the convex front
SLOW (not just fan)? compare G-hourglass vs G-strand head-to-head.

**Pass:** demonstrate **measurable convex slowing in G-strand that is absent/weaker in
G-hourglass at the same expansion ratio** -> isolates regime as the variable. Tune `w` and
expansion ratio to the onset of slowing (entering 4C).

**Outcome:**
- Slowing appears in G-strand -> Condition 2 confirmed; carry the slowing geometry to S3.
- No slowing in either at healthy excitability -> expected (SF too high); defer to S3
  (excitability is the dominant remaining knob). Not yet a disproof.

---

## 7. STAGE 3 — Excitability: drive C -> D (block)  (Condition 3)

**Hypothesis:** lowering safety factor raises `r*` and liminal area until convex slowing
becomes functional block (4D).

**Setup:** `dx_resolved`, G-strand (or the S2 slowing geometry), fixed expansion ratio.

**Sweep (one knob at a time, then combine):**
- `gNa` scale: 1.0, 0.75, 0.5, 0.35, 0.25
- `[K]o`: 5.4 (ctrl), 6.5, 8.0, 10 mM
- partial depolarization / reduced IK1
- gap-junction uncoupling: scale `D` down 1.0, 0.5, 0.25

**Observable:** SF vs knob; `CV_n(junction)`; **block yes/no** (downstream bulk never
activates; double-line LAT discontinuity = Fig-4D); the (expansion-ratio x excitability)
block phase boundary.

**Pass:** produce **functional block** (`CV_n -> 0`, downstream silent) and map the
A->B->C->D progression by sweeping one knob. Record the threshold (SF*, gNa*, etc.).

**Outcome:**
- Block achieved -> **PROVE the engine handles 4C/D in 2-D**; report parameter recipe.
- No block down to extreme (non-physiological) excitability at `dx_resolved` ->
  **DISPROVE** in the physiological regime; document the gap (revisit S1 dx, or geometry).

---

## 8. STAGE 4 — Reproduce Figure 4 A-D + the eikonal law

With `dx_resolved`, G-strand, tuned excitability, produce all four panels and the equation:

| Panel | Geometry / knob | Expect |
|---|---|---|
| A concave speedup | concave-forcing geometry (wave into a notch / converging-to-focus — NOT a plain funnel) | `kappa<0`, `CV_n>CV0` |
| B rectilinear | straight channel | `kappa=0`, `CV_n=CV0` |
| C convex slow | mild expansion, sub-threshold excitability | `kappa>0`, `CV_n<CV0` |
| D block | sharp expansion and/or SF below threshold | `CV_n->0`, double line |

**Quantitative close-out:** recover `CV_n = CV0 - D*kappa` (slope = measured D from S0) AND
the cross-section form `theta = theta0 - D*(dW)/(c*W)` (width W = the 2-D analog of
Ciaccio's thickness T). Plot CV vs relative cross-section gradient; locate the block
threshold and compare to Ciaccio's `~2 per mm`.

---

## 9. Decision tree (global verdict)

```
S0 eikonal fit?
 ├─ FAIL ............................. numerics broken -> STOP, debug (deep problem)
 └─ PASS -> r* measured
     S1 r_crit converges at feasible dx?
      ├─ NO ........................... CANNOT at feasible dx (scheme-limited) -> report,
      │                                 consider higher-order/adaptive operator
      └─ YES -> dx_resolved locked
          S2 G-strand slows where G-hourglass doesn't?
           ├─ NO (and S3 also fails) ... regime not the issue -> excitability-limited
           └─ YES -> regime confirmed
               S3 block achievable in physiological excitability range?
                ├─ YES .................. CAN  -> recipe (dx*, geometry, SF*) -> S4 full A-D
                └─ NO ................... CAN-WITH only non-physio excitability -> document
```

Each stage's output is either a **tuned parameter** passed downstream (`r*`,
`dx_resolved`, `w`, expansion ratio, SF*) or a **prove/disprove** branch above.

---

## 10. Parameter registry (fill as measured)

| Symbol | Meaning | Source | Value |
|---|---|---|---|
| CV0 | radial CV (expanding circle) | S0 | **62.4 cm/s** (0.0624 cm/ms) |
| D_eik | eikonal coefficient (LAT(r) integrated fit) | S0 | **0.00084 cm²/ms** (16% < operator D=0.001; expected leading-order eikonal correction) |
| r* = D_eik/CV0 | critical radius | S0 | **~134 µm** (160 µm if using operator D) |
| dx_resolved | grid for C/D | S1 | ___ (target dx ≤ r*/3 ≈ **45 µm**; S0's 50 µm gives r*/dx=2.7, borderline → sweep to ~35 µm) |
| r_crit | nucleus block radius | S1 | ___ |
| w*, exp-ratio* | onset of slowing | S2 | ___ |
| SF*, gNa*, [K]o* | block thresholds | S3 | ___ |

**S0 method note:** the eikonal coefficient is robustly recovered by fitting the SMOOTH
integrated form `LAT(r) = r/CV0 + (D/CV0²)·ln r + c` (R²=0.99997), NOT by differentiating
binned LAT (an 8–13% curvature signal drowns in differentiation noise → R²≈0.13) nor by
per-cell `div(n̂)` (R²≈0.04). `cardinal4` is anisotropic → use `moore8_iso` for circular
waves. The `div(n̂)` estimator (`front_metrics`) is still needed for arbitrary geometries
in S2–S4; bin/smooth it there.

## 11. Deliverables

- `analysis/eikonal_metrics.py` (+ self-test) — the §2 pipeline
- per-stage `run_*.py` in the engine's `experiments/` dir (Research/ stays writing-only)
- one EXPERIMENT.md per stage with backlinks (this plan + README + MASTER)
- panels A-D + CV-vs-kappa and CV-vs-(dW/W) plots -> media path above
- KNOWLEDGE.md update with the global verdict + parameter registry
