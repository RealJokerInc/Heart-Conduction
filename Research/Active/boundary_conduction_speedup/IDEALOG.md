# Boundary Conduction Speedup — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan this in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
CV ratio confirmed at 1.0714 (dx=0.025cm) converging toward theoretical 1.131. Isotropic case fully characterized: mechanism (8-link argument chain), mesh convergence, triangle merger wavefront, stencil comparison, and conductivity sweep all complete. Anisotropic boundary study is active — initial 2:1 anisotropy test shows sharper triangle (2.27cm edge lead vs 1.62cm isotropic), confirmed by eikonal prediction. Still need fiber-parallel vs perpendicular systematic study, 3D validation, and tissue thickness study.

## Next Step
**Immediate (storage-tank thread)**: investigate the role of the activation
threshold (II.3) in the pipe-firing rule. Sweep θ values across both the constant
and gradient rules at fixed pump rate / k. Predict: lowering θ widens the active
band → bigger camel toe in constant rule (more drainage-advantage time);
raising θ narrows the band → both rules eventually fail to propagate. Test
whether threshold removal on the gradient rule lets ohmic Fickian show ANY
camel toe (it shouldn't — Effect 2 is intrinsic to gradient-driven coupling).

**Background (Tier-I minimal model)**: still pending. Build "Tier-I-only" model
(linear ohmic + smooth threshold + bidirectional + anisotropic) and check
whether camel toe survives. Predict it flips sign.

**Background (cardiac thread)**: anisotropic boundary study — systematic fiber-parallel vs
fiber-perpendicular analysis at the bath-coupled boundary. The initial 2:1 anisotropy experiment
ran successfully but needs a full parameter sweep across fiber orientations.

## Thread

### 2026-03-13: Research restructured from domain-based to question-driven
The Research folder was reorganized from domain-based directories (Bidomain/, LBM/, FHD/) to question-driven folders (Q1 through Q7). Boundary conduction speedup became Q5. This was motivated by the workflow being question-driven: "does CV increase at boundaries?" rather than "what does the bidomain engine do?". Papers were re-filed by research question, INDEX.md created as the master question map.

### 2026-03-15: Mehrstellen 9-point stencil implementation for curved wavefront accuracy
Implemented the Mehrstellen (isotropic 9-point, O(h^4)) stencil to resolve whether the 5-point stencil's truncation error was contaminating the Kleber wavefront shape. This was needed because the boundary speedup produces curved wavefronts where stencil isotropy matters. Eight implementation steps, 16 tests passing. The stencil turned out to affect absolute CV (~4% lower, 47.1 vs 49.1 cm/s) but not the relative Kleber effect -- both stencils produce identical wavefront shapes within 0.05cm.

### 2026-03-15: Triangle merger experiment — the "merger" does not happen
Ran the full triangle merger pipeline (3 configs: monodomain Mehrstellen, bidomain 5pt, bidomain Mehrstellen) on a 50x8cm domain for 800ms. Key surprise: the triangular wavefront is the steady state, not a transient that merges. The edge-center lead saturates at 1.65-1.70cm by t~300-450ms and remains constant. The "triangle merger" terminology was misleading. The Kleber ratio of 1.131 describes the transient speedup during wavefront shape establishment; once the chevron forms, edge and center propagate at equal velocity. The effect is encoded in the accumulated lead distance, not in a persistent CV difference.

### 2026-03-15: Monodomain control produces perfectly flat wavefront
The monodomain Mehrstellen config on the same grid produced exactly 0.000cm deviation from flat. This definitively isolates the Kleber effect as a bidomain boundary coupling phenomenon (asymmetric BCs: Neumann intracellular + Dirichlet extracellular), not a numerical artifact of the stencil or grid.

### 2026-03-15: GPU 4x speedup confirmed for bidomain simulations
Bidomain triangle merger on GPU (RTX PRO 4500 Blackwell): 6.0ms per step vs 23.4ms on CPU, 4.0x speedup. Total pipeline dropped from ~75min to ~26min. This makes parameter sweeps practical.

### 2026-03-15 (approx): Conductivity sweep — edge lead scales with sqrt(D_eff)
Five configurations tested: 0.5x iso, 1x iso (baseline), 2x iso, 4x iso, and 4x sigma_i only. Scaling both sigma_i and sigma_e uniformly preserves the Kleber ratio (all at 1.131) with edge lead growing as sqrt(D_eff). The 4x-sigma_i-only config is the key result: increasing sigma_i to 6.96 while holding sigma_e at 6.25 raises the theoretical Kleber ratio from 1.131 to 1.454, and the edge lead jumps to 4.48cm (vs 1.45cm baseline). This confirms the boundary speedup is governed by the sigma_i/sigma_e ratio, not just absolute conductivity.

### 2026-03-15 (approx): Anisotropic test — 2:1 ratio produces sharper triangles
First anisotropic test with 2:1 conductivity ratio (longitudinal:transverse). Edge lead increased from 1.62cm (isotropic) to 2.27cm (anisotropic), a 40% increase. The sharper triangle is consistent with the eikonal prediction: lower transverse conductivity means the wavefront curves less easily, so the boundary speedup accumulates a larger geometric distortion. This is the precursor to the full anisotropic study.

### 2026-03-16: Research reorganized from Q-numbers to Active/Complete/Backlog
The Q-number naming (Q5_boundary_conduction_speedup) was replaced with status-based paths (Active/boundary_conduction_speedup). MASTER.md became the project dashboard. Experiments got their own directories inside engine folders with backlinks to the research question. The gap identified: experiments (scripts, parameters, outputs) had no standardized home -- they lived in engine test files with no connection to the research question that motivated them.

### 2026-03-16: Realized experiments need a home between "hypothesis" and "knowledge"
The research workflow goes Hypothesis -> Script -> Run -> Outputs -> Analysis -> Finding -> Knowledge. Each step was living in a different place with no links. Experiments directory structure was created inside engine folders, cross-linked to research questions via EXPERIMENT.md files. This solved the problem of the conductivity sweep and triangle merger results being detached from the boundary speedup question.

## Failed Approaches
- **"Triangle merger" framing** (2026-03-15) — failed because: the triangular wavefront does not merge. It IS the steady state. Edge-center lead saturates at 1.65-1.70cm and remains constant through 800ms. The experiment was designed expecting two triangular deformations from opposite edges to interact, but instead the chevron shape is simply the equilibrium between boundary speedup and wavefront curvature. The terminology was corrected but the experiment name was kept for historical continuity.
- **Late-time CV ratio as Kleber measurement** (2026-03-15) — failed because: at steady state, all y-rows advance at the same velocity (ratio ~1.0). The curvature-induced diffusive correction exactly compensates the boundary speedup. The Kleber ratio must be measured during the transient growth phase (t=25-200ms) before wavefront curvature develops, or from the accumulated edge lead distance. The existing phase 6C tests on the smaller 150x40 grid at dx=0.025cm are the correct measurement approach.
- **5-point stencil sufficiency for boundary effect** (2026-03-15) — not a hard failure but a resolved concern: the 5pt stencil overestimates absolute CV by ~4% compared to Mehrstellen at dx=0.05cm, but both produce identical Kleber wavefront shapes. The stencil choice affects absolute accuracy, not the relative boundary physics.

### 2026-03-18: Literature survey — parabolic modeling of extracellular domain
Compiled a comprehensive bibliography of papers treating the extracellular potential equation as parabolic rather than elliptic. Three main approaches identified: (1) mathematical regularization (add ε·∂φ_e/∂t, prove existence, take ε→0) — Bourgault et al. 2009, Colli Franzone & Savaré 2002, Bendahmane & Karlsen 2006; (2) LBM dual-lattice where both Vm and φ_e diffuse parabolically — Corre & Belmiloudi 2016, Belmiloudi 2019; (3) augmented monodomain collapsing bidomain to single parabolic PDE — Bishop & Plank 2011 (already in corpus). Key insight: Potse et al. 2006 quantifies where monodomain fails and identifies boundary effects as the main failure mode — directly validates our Kleber effect work. Bourgault 2009 is the foundational paper; user downloading it for full summary.

### 2026-04-24: PI toy model (storage tanks) and unified single-field PDE formulation
John shared a 2D storage-tank sim (Colab export, filed at repo-root `simulation/`) that exhibits a
boundary speedup: Moore-neighbourhood pumping, source-state-dependent flux, threshold-gated.
Interior tanks have 8 outflow channels, edge 5, corner 3 — fewer sinks at the edge means a
fired tank retains potential longer and sustains drive along the boundary. The flux rule is
non-Fickian (flux per link ∝ f(u_i), gated by H(u_i−u_j), no dependence on gradient).

Worked out the PDE framing. Standard heat + Neumann cannot produce tangential speedup
(this matches the monodomain control). Three candidate modifications:
(A) heat + distance-dependent loss γ(x)u — pedagogical
(B) heat with spatially-varying D(x), D_bulk = σᵢσₑ/(σᵢ+σₑ), D_bdry = σᵢ — canonical, exact
(C) non-local / peridynamic with kernel truncated at ∂Ω — most faithful to tank rule

Settled: (B) is the single-field PDE that captures Kleber with the right numerical ratio
(1.131 for human ventricle longitudinal). It is the monodomain reduction of the bidomain
with the boundary correction baked into D(x). The anisotropic extension is immediate —
promote scalar D(x) to tensor D(x), eikonal-limit speed becomes 2√(n̂·D(x)·n̂·f'(0)).
Full derivation filed in KNOWLEDGE.md § "PDE Formulations of the Effect".

Open: (B) vs (C) as the bridge from John's discrete rule to the bidomain PDE — (C) is
the honest continuum limit of the tank rule, but collapses to (B) under a gradient expansion.
Could be worth a short note on when that approximation breaks (sharp wavefronts of width < R).

### 2026-04-25: Storage-tank deep dive — vec rewrite, per-column camel toe, two-effect framing

Rebuilt John's tank model in numpy (`simulation/tanks_vec.py`), bit-equivalent to the
OOP version (`test_vec_matches.py`: max|ΔV| = 5e-14, max|Δiso| = 0). ~200× faster, makes
parameter sweeps practical. Found a real bug in John's original Colab: he sets
`maxpump = 5.0` in main but never threads it to the tanks, which run with the constructor
default of 10.0. All my analysis from here on uses max_pump=10 (his actual value).

**Per-column camel toe finding.** The leading-edge max-x metric I'd been using was the
wrong test. The correct one is intra-column firing order: for column N, when does
iso[y, N] cross threshold for each y. Side-by-side line-source tests:
  - constant rule: edges fire 15-25 steps BEFORE middle in cols 5-30 (camel toe / inverted
    crescent), reverts to interior bulge at cols 1-3 (geometric) and >45 (chaotic regime).
  - gradient rule: pure crescent (interior bulge) at every column, monotonically growing
    with distance. NO camel toe ever.
Filed `simulation/per_column_camel_toe.py` and `outputs/per_column_camel_toe.png`.

**Long-time / steady-state analysis (8000 steps).** `simulation/long_run.py` runs both
rules and tracks ‖V(t)−V(t−1)‖₂ to detect stationarity:
  - Constant rule does NOT reach steady state. Activity asymptotes at ~1100 indefinitely
    — system enters a chaotic limit cycle of fire / drain / refill. No fixed point exists
    given the inlet/outlet BCs because the rule is bistable per tank without recovery.
  - Gradient rule approaches steady state. Activity decays ~100× by step 1k, slow plateau
    after, would settle near zero if run to ~16k steps. Steady-state target is the
    Laplace solution V(x) = 100·max(0, 1 - x/x*) up to threshold-gate stall point.

**Two-effect framing for the boundary asymmetry.** Distilled the mechanism:
  - Effect A (inflow deficit): edge tank has 2 fired upstream nbrs vs 3 interior. Pure
    geometry; appears in *every* nearest-nbr-coupled lattice with no-flux walls. Always
    pushes toward crescent (boundary slowdown).
  - Effect B (outflow dividend / sustained source): edge tank has 5 outflow channels vs
    8, drains slower, stays fired longer, integrates more drive into next column's edge.
    Only manifests under non-self-limiting flux (constant rule). Pushes toward inverted
    crescent (camel toe).
  - Gradient rule: B is killed by self-limiting flux. A unopposed → permanent crescent.
  - Constant rule: B beats A in mid columns → camel toe. Chaotic regime kills both.

**Operator-level argument (user's brain wave).** A perfectly uniform initial wavefront
should not produce A or B by symmetry — but it does. Reason: the discrete update operator
U is *not translation-invariant in y* near a no-flux wall. Edge rows of U have fewer
non-zero entries than interior rows. So uniform input → non-uniform output on the very
first step. Effect A is baked into the *operator*, not the *state*. This is the same
reason monodomain control comes out flat-to-slow rather than flat — even with perfect
ICs, the discrete Laplacian at the wall is asymmetric.

**Open / next**: corner-ghost-cell test. Augment corner tanks (currently 3 nbrs) to act
as if they have 5 (i.e., reflection BC at corners only) and see if Effect A vanishes
locally. If yes → confirms operator-level argument experimentally. If no → look elsewhere.

### 2026-04-25 (cont): Reflection-BC test — boundary OPERATOR dominates pump rule

Ran `simulation/ghost_corner_test.py`: same line geometry, both rules, three boundary
treatments — baseline (zero-pad), refl-y (corners=5, edges=8), refl-all (corners=8,
edges=8). Mass-conserving routing: when a source pumps to a ghost destination, the
flux is folded back into the mirror real cell.

Result was dramatic and reverses one of yesterday's conclusions:

| treatment       | constant Δ@x=18 | gradient Δ@x=18 |
|-----------------|-----------------|-----------------|
| baseline 3-5-8  | -12.5  (camel)  | +49   (crescent) |
| refl-y 5-8-8    | -231  (HUGE)    | -270  (HUGE camel — flipped!) |
| refl-all 8-8-8  | -257  (HUGE)    | -255  (HUGE camel) |

Both rules show massive camel toe under reflection, ~20× the magnitude of the original
constant-rule camel toe. Constant mode also runs ~4× faster under reflection (1177 steps
to fill vs 3959 baseline). Why: reflection padding *duplicates* upstream values — a
boundary cell at (0, x) sees 8 channels but only 5 unique upstream cells, with the y=1
row contributing twice. That double drive overwhelms Effect A and creates a strong
boundary speedup. Call this **Effect B′** — mirror-duplication enhancement.

**Sharpened conclusion (replaces "gradient never produces camel toe").** The
*boundary operator* (how the wall handles missing neighbours) determines the sign of
the boundary effect *more strongly than the pump rule does*:

  - Zero-pad BC ≈ Neumann ≈ monodomain → both rules give crescent baseline; constant
    rule layers a transient camel toe on top via Effect B.
  - Reflection BC ≈ enhanced inflow → both rules give massive camel toe via Effect B′,
    pump-rule choice barely matters.

This maps directly onto the cardiac dichotomy:
  - Monodomain (single field, Neumann at wall) ↔ zero-pad here ↔ boundary slowdown.
  - Bidomain with bath-coupled extracellular Dirichlet ↔ enhanced-inflow BC ↔ Kleber.

**User's framing settled.** "Starting condition influences downstream A↔B interaction"
is correct, with the refinement that "starting condition" = *boundary operator*, not
*initial state*. The operator-level argument from yesterday's note rules out IC as the
source of asymmetry; the reflection experiment proves that the BC is the actual lever.
The pump rule (constant vs gradient) modulates magnitude but the BC choice fixes the
sign of the per-column LAT shape.

**Open question.** Is there a discrete operator that's *truly translation-invariant* at
the boundary — no Effect A, no Effect B′? Periodic BC trivially gives this but changes
the geometry to a torus. A "scaled reflection" (ghost contribution × 0.5) might cancel
the duplication, but it's ad-hoc. The honest answer is probably that on a finite lattice
with any local BC, the boundary operator is necessarily different from the bulk operator,
and the only choice is what *kind* of difference you allow.

### 2026-04-25 (cont): Bidirectional pipes kill the camel toe — directionality is the *third* axis

`simulation/bidirectional_test.py`: dropped the `V_src > V_dst` gate from the constant
rule (kept the threshold gate). When both endpoints are above threshold, both pipes
A→B and B→A fire, with rates f(V_A) and f(V_B) — net flow becomes f(V_A) − f(V_B),
self-limiting just like Fickian.

Same line geometry, 4000 steps, edge−mid at column 18:

| rule                         | Δ(edge−mid)@x=18 |
|------------------------------|------------------|
| constant one-way (John's)    | −12.5  (camel)   |
| constant **bidirectional**   | +25.5  (crescent) |
| gradient                     | +49    (crescent) |

Bidirectional + sqrt rate produces a clean crescent at every column, just like the
gradient rule but with smaller magnitude. Camel toe is gone entirely. So:

> **One-way pipes are the *only* feature in John's model that produces camel toe.**
> Threshold gating, Moore connectivity, source-state-dependent rate — none alone do
> it. Drop the one-way assumption and Effect B dies, leaving only Effect A.

Clean axis decomposition for the per-column LAT shape:

| axis                    | options                          | controls            |
|-------------------------|----------------------------------|---------------------|
| boundary operator       | zero-pad / reflect-y / reflect-all | sign (Effect A vs B′) |
| pipe directionality     | one-way / bidirectional          | Effect B existence  |
| pump rule               | constant / gradient              | rate magnitude      |

Camel toe requires (zero-pad BC) AND (one-way pipes). Either modification kills it.

Bidirectional also presumably reaches a true steady state (no chaotic limit cycle)
because reverse pumping prevents the perpetual fire-drain-refill loop — but I didn't
run long enough to confirm. Worth a long-run check if needed.

### 2026-04-26: Experiment harness + pump-speed Goldilocks zone

**Harness built.** `simulation/configs.py` + `simulation/experiment.py`:
  - `tanks_vec.run()` extended with `directionality`, `boundary`, `damping_cap`,
    `record_history` flags. Returns dict instead of tuple. Parity test still passes
    (max|ΔV| = 5e-14, max|Δiso| = 0).
  - `configs.DEFAULT` documents the full schema; `configs.REGISTRY` has 8 named
    configs (baseline, gradient, bidirectional, reflect_y, reflect_all,
    long_run_constant/gradient, john_radial).
  - `configs.make(overrides, *, base=DEFAULT)` deep-merges; *must* use `base=` kwarg
    for variants — `make({**BASELINE, "rule": ...})` is broken because the spread
    is shallow and clobbers nested rule fields. Caught this bug when first
    gradient-sweep silently ran constant rule.
  - `experiment.run_experiment(cfg)` writes self-contained run dir at
    `outputs/experiments/{date}_{name}/` with config.json, iso.npz, isochrone.png,
    per_column_lat.png, summary.txt, metadata.json. Appends one row to
    `outputs/experiments/INDEX.md`.
  - CLI: `python experiment.py {name}` or `python experiment.py all`.

**Re-ran the prior list of experiments through the harness**: all 8 named configs.
Numbers match earlier ad-hoc runs to the tank (baseline Δ=−12.5, bidirectional
Δ=+25.5, gradient Δ=+49, etc.).

**Pump-speed Goldilocks zone (constant rule).** `simulation/sweep_pump_speed.py`
sweeps `max_pump ∈ {2, 5, 10, 15, 20, 30}` for constant rule, line geometry:

| max_pump | Δ@x=18 | shape       |
|----------|--------|-------------|
| 2        | +53    | crescent    |
| 5        | −8     | camel       |
| 10       | −12.5  | camel (peak) |
| 15       | −1.5   | camel (weak) |
| 20       | 0      | flat        |
| 30       | +3     | crescent    |

Non-monotonic. The drainage advantage only wins when *drainage timescale ≈ inflow
timescale* — a resonance condition. Too slow and the wavefront passes before the
edge's slower drain matters; too fast and the column fires nearly simultaneously
across rows so drainage delta is negligible. **John's effective max_pump=10 is
right at the camel-toe peak** — possibly tuned by him until the effect appeared.

**Gradient sweep**: `gradient_k ∈ {0.02, 0.04, 0.08, 0.12, 0.16}`. Always crescent;
magnitude scales with 1/k (k just rescales the time axis). Confirms structural
prediction: gradient rule's self-limiting flux makes Effect B impossible at any
k; only Effect A acts.

**Effect labelling settled.**
  - "drainage effect" / "Effect B" (my old A) = corner has fewer outflow channels,
    drains slower, retains V longer → camel toe / inverted crescent.
  - "inflow effect" / "Effect A" (my old B) = edge has fewer above-threshold
    upstream nbrs when wavefront arrives → fires later → crescent.
  - User's preferred labels: A = drainage, B = inflow. Going forward: drainage
    effect / inflow effect (avoid A/B numerals to escape the relabeling cycle).

**Open / next**: 
  - Threshold sweep: predict that lowering threshold widens the excitable band and
    makes drainage advantage integrate over more steps → bigger camel toe.
  - Damping-cap toggle: turning off the (V_src − V_dst)/4 clamp should de-Fickianise
    the rule when source/dest values are close → bigger camel toe.
  - Both should be testable in <30s with the harness.

### 2026-04-28: Hydrostatic derivation, single-cell LBM mechanism, axiom restructure

**John's per-cell law derived from first principles.** For a single tank with outlet
hole at height θ draining to atmosphere, Bernoulli gives v = √(2g·(h − θ)) — Torricelli.
John's `max_pump · √((V−θ)/(V_max−θ))` is exactly Torricelli, normalized so V=V_max
gives max_pump. The √ and threshold are not arbitrary modeling choices; they fall out
of energy conservation (potential head → outlet kinetic energy) and outlet-hole geometry.
**Per-cell physics is hydrostatically correct, no modification needed.**

For a *submerged pipe* between two tanks (h_C > θ AND h_i > θ), Bernoulli on free-surfaces
gives v = √(2g·(h_C − h_i)) — gap-driven, NOT source-driven. John applies his single-cell
law to this regime too, which is incorrect: he over-estimates the rate by
√((h_C−θ)/(h_C−h_i)). His quarter-gap damping clamp is a CRUDE LINEAR approximation to
the missing √-gap Bernoulli physics. Unified hydrostatic-faithful form:

```
rate(C → i) = max_pump · √( max(V_C − max(θ, V_i), 0) / (V_max − θ) )
```

**Continuous physics, numerically discretized.** John's per-cell ODE has analytic
solution V(t) = θ + (√(V₀−θ) − k·t/2)² with k = max_pump/√(V_max−θ). Empties V₀=V_max
to θ in t* = 2·(V_max−θ)/max_pump = 11 steps. His code is forward Euler with dt=1 —
the damping clamp is a numerical-stability hack for the multi-cell coupled regime, not
a physics feature.

**Single-cell LBM-style mechanism.** For one cell C with 8 Moore neighbors, each link
carries two populations φ_i⁺ (C→i outflow) and φ_i⁻ (i→C inflow). All variants share
the same Jacobi-buffered streaming; they differ only in firing conditions and rate.
Filed full population-table comparison covering John, const+1way+nodamp, const+bidir,
gradient+1way, gradient+bidir.

**Tank-level role as the LBM differentiator.** Standard LBM diffusion has equilibrium
f_i^eq = w_i · ρ — *linear* in density. John has rate ∝ √(V_C − θ) — *concave* in
source state. The concavity is the necessary ingredient for Effect 1 (drainage advantage):
high-V cells punch above their weight only when the rate-vs-source curve is concave.
Linear rate laws (LBM, gradient rule) cannot produce camel toe at the boundary by this
mechanism. To give the LBM setup John-like boundary speedup, the recipe is to make
f_i^eq concave in ρ — not standard cardiac LBM but a clean test target.

**Threshold step function as asymmetry amplifier.** The hard step at θ creates two
regimes per cell: accumulation (inflow only) and pumping (full √-rate). The switch is
binary. This separation lets each effect express cleanly — accumulation phase amplifies
inflow-channel deficit (Effect 2), pumping phase amplifies outflow-channel deficit
(Effect 1). Smooth-onset variants weaken both because cells continuously self-leak
while accumulating. **Predicted ordering: step > smooth-ramp > no-threshold** for
camel-toe magnitude.

**Tiered axiom restructure (cardiac claims vs model implementation).** Distinction
flagged by user: not every feature of John's Colab represents what he'd defend in a
cardiac journal club.

  TIER I — Genuine cardiac axioms (defendable in heart literature):
    I.1  Discreteness matters at the cell scale
    I.2  Sub-threshold accumulation in cells (form open: step or smooth)

  TIER II — Model implementation features (NOT cardiac claims):
    II.1  Torricelli √-law for source-state driving
    II.2  Source-state-only coupling
    II.3  Hard step function threshold
    II.4  Hard one-way valve at gap-junction level
    II.5  Moore-8 dense connectivity
    II.6  Square lattice
    II.7  Synchronous Jacobi update
    II.8  Quarter-gap damping clamp
    II.9  Memoryless cells
    II.10 No-flux Neumann boundary

User explicitly rejected adding I.3 (threshold-gating), I.4 (refractoriness), I.5
(extracellular bath/Kleber loading) to Tier I. Tier I stays minimal at I.1 + I.2.

**Three-question evaluation program.** Replaces the old two-phase plan.

  Q1 — SENSITIVITY: does each axiom produce a boundary artifact in the toy model?
       (Already partly characterized for II.4 bidirectional, II.10 reflect-y BC,
       and the rate-law axes.)

  Q2 — ROBUSTNESS: does Tier I alone (cardiac-realistic Tier II) produce camel toe?
         IF YES → boundary speedup is a Tier-I consequence; cardiac defense reduces
                  to defending I.1 + I.2.
         IF NO  → speedup depends on Tier-II artifacts John doesn't defend as cardiac.
                  The boundary effect is a model artifact, not a cardiac prediction.

  Q3 — CARDIAC TRUTH OF I.1 + I.2: defend or reject from biology (gap-junction density,
       optical mapping at edges, Spach/Kleber literature).

**Prior on Q2 outcome.** The biophysically suspect axioms (II.1, II.2) are exactly
what produces Effect 1. The defensible Tier-I axioms can at most support Effect 2
(inflow deficit → crescent / slowdown). Predict: under cardiac-realistic Tier-II,
the boundary effect *flips sign* relative to John's setup — slowdown, not speedup.
This is the testable Q2 prediction.

### 2026-04-28 (cont): Visualization sweep + Fickian intrinsically crescent

Built a visualization stack for the pump-speed sweep on John's BASELINE rule
and the equivalent k-sweep on the GRADIENT rule. Key infrastructure:
  - `simulation/render_pump_speed_stack.py` — vertically stacked time-evolution
    video with per-strip freeze-on-completion (each strip freezes when its
    wavefront's leading edge stops advancing). Robust to both fast pumps that
    fully traverse and slow pumps that stall mid-domain.
  - `simulation/render_camel_to_crescent_clean.py` — config-parameterized
    single-config clean video; works for any name in `configs.REGISTRY`.
  - Per-column LAT-deviation figures at Nx=80, Nx=320, Nx=50 for both rules,
    in 2x3 grids. Final convention: `dev = column-mean iso − iso(y, x)` so
    *positive = AHEAD of wavefront mean*, *negative = BEHIND*. Per-subplot
    y-scaling so fast/slow regimes are both legible.

**Finding 1 — propagation stalls under threshold gating.** With finite max_pump
(constant rule) or finite k (gradient rule), the wave only propagates a finite
distance before the upstream V drops below θ and the next column can never
fire. For Nx=320:
  - constant rule: stalls at col ~120 for max_pump ≤ 10
  - gradient rule: stalls at col ~33 for k=0.01, ~132 for k=0.16
The stall point is set by the local rule, not by upstream conditions. Boosting
inlet doesn't help — confirmed analytically. Real cardiac analog: this is the
same phenomenon as conduction block / safety-factor failure when sodium current
is too low.

**Finding 2 — Fickian (gradient) rule intrinsically crescents.** Across the
full k-sweep at three tissue lengths (Nx=80, 320, 50), the GRADIENT rule
produces a CLEAN INVERTED-U at every column, every k, every domain length.
No camel-toe regime appears anywhere. The middle of every column always leads
its boundaries. This confirms the structural prediction from earlier
(KNOWLEDGE.md "Single-cell mechanism"): linear gap-driven flux is
self-limiting, so Effect 1 (drainage advantage) is impossible; only Effect 2
(inflow channel deficit at boundary) can express, and it always favors
crescent.

**Finding 3 — Effect 2 amplitude scales with 1/k and with x.** Lower k →
deeper crescent. Larger column index → deeper crescent (cumulative integration
of inflow deficit along propagation). At k=0.01 on Nx=50, x=25 shows
~+150-step lead at the middle relative to boundaries.

**Finding 4 — at slow pump rates on the constant rule, Effect 1 also dies.**
On Nx=80 at max_pump=2, the per-column LAT shape is *crescent at every column*
(deepest at x=70 with ~+280 step lead). The Goldilocks camel-toe at max_pump=10
mid-columns IS the only camel-toe regime — slow pumps lose drainage advantage
entirely (drainage timescale becomes too long relative to inflow timescale,
no resonance), fast pumps blow through too fast for either effect to express.

This sharpens the program: the camel toe in John's rule is a narrow Goldilocks
phenomenon dependent on (concave √ rate-law) AND (mid pump rate) AND
(threshold step function). Any one of these gone → crescent everywhere.

### 2026-04-29: BC discretization is the missing piece — face-centered vs node-centered mirror

User asked the sharp question: storage-tank with gradient (Fickian) rule shows
persistent crescent on uniform input, but monodomain and LBM with linear ohmic
coupling never show this. Why?

**Diagnosis.** It's not the rate law (gradient is linear in both), not the
diagonal coupling (D2Q9 has it too), not threshold position (gradient rule still
shows crescent without it). It comes down to **HOW the no-flux Neumann BC is
discretized**.

**Two inequivalent discretizations of ∂V/∂n = 0:**

  Face-centered mirror (standard FDM/LBM):
    wall at y = −0.5,   V_ghost = V_boundary[y=0]
    Ghost cell mirrors the BOUNDARY itself.
  
  Node-centered mirror (John's reflect_y via np.pad):
    wall at y = 0,      V_ghost = V_subedge[y=1]
    Ghost cell mirrors the SUB-EDGE one row in.
    Plus mass-conserving fold of ghost-row inflow back to sub-edge.

**Why face-centered cancels everything.** For 5-point Laplacian at boundary with
V_ghost = V_C:
  ∇²V|_boundary = (V_C + V[1,x] + V[0,x−1] + V[0,x+1] − 4·V_C)/h²
                = (V[1,x] + V[0,x−1] + V[0,x+1] − 3·V_C)/h²
For uniform y (V[1,x] = V_C), reduces to (V[0,x−1] + V[0,x+1] − 2·V_C)/h²,
identical to the interior 1D Laplacian. Boundary and interior fire simultaneously.
This is exactly the user's `0.000 cm deviation from flat` monodomain control.

**Why node-centered creates a cascade.** The np.pad reflect mode mirrors boundary
about its own NODE (y=0), so V_ghost at (-1, x) = V[1, x] = sub-edge. Plus the
fold-back: flux that pumps a ghost destination is rerouted to the real cell that
the ghost mirrors. Step-by-step:
  - Column 1 from line inlet: boundary and interior both get 3 channels of inflow
    (W, NW, SW from inlet column). SUB-EDGE gets 3 channels PLUS ghost-folded
    contribution from upstream boundary's ghost neighbor → 5 effective channels.
  - Sub-edge fires faster than boundary or interior at column 1.
  - Column 2 onwards: sub-edge pumps boundary from BOTH the real S direction
    AND the ghost mirror N direction (ghost reflects sub-edge across y=0). Boundary
    gets a doubled y-pump from sub-edge. Interior gets no y-pump (uniform-y symmetry
    cancels its flanking cells).
  - → Boundary advances faster than interior over many columns. Camel toe.

This explains the Δ@x=18 = −270 we measured for `gradient + reflect_y`. It's the
sub-edge-mediated cascade, not concavity, not source-state coupling.

**For zero_pad** the asymmetry is even simpler — boundary genuinely has 2
upstream channels vs interior's 3. → crescent. The standard FDM equivalent
(implementing a one-sided "amputated" stencil with V_outside = 0 instead of
V_outside = V_boundary) would reproduce this directly.

**Classification table:**

| System                        | BC discretization              | Boundary asymmetry?         |
|-------------------------------|--------------------------------|-----------------------------|
| storage-tank zero_pad         | amputated stencil              | YES (Effect 2 → crescent)   |
| storage-tank reflect_y        | node-centered + fold-back      | YES (sub-edge → camel)      |
| monodomain (5pt/9pt) Neumann  | face-centered mirror           | NO                          |
| LBM bounce-back               | equivalent to face-centered    | NO                          |
| bidomain bath-coupled         | Dirichlet on V_e at wall       | YES (Kleber, real continuum)|

**Implication for John's argument.** John is right that discrete connectivity
asymmetry produces persistent boundary artifacts — but ONLY for discretizations
that don't compensate the missing channels. Standard cardiac FDM/LBM use exactly
the discretization that does compensate. So the storage-tank artifact is a
*choice of discretization*, not a property of the underlying continuum equation.
The Kleber bath-coupled speedup in bidomain is a different mechanism altogether
(asymmetric Dirichlet/Neumann pair) and is genuinely a continuum effect.

**Caveat (added 2026-04-29 after audit):** the step-1 walk-through of the
cascade in this entry is too neat. With the gradient + one_way rule's strict
`V_src > V_dst` gate, two cells at exactly equal V do NOT fire — the cancellation
of y-direction flux at uniform-y interior is "no flux fires" rather than
"fluxes cancel". So the cascade can't develop in step 1 from a strictly
uniform line inlet via y-direction pumps alone. The asymmetry must seed
somewhere else (corner ghost stencils, x-direction NW/NE diagonals when their
ghost destinations get folded). The empirical Δ=−270 we measured is a many-step
accumulation, not a step-1 phenomenon. PLAN.md Step C.3 was added to verify
the cascade by simulation rather than by handwaving. Updated KNOWLEDGE.md
with a "Caveat — the cascade requires SEEDED asymmetry" note.

**Concrete program built into PLAN.md:**
  Phase A — Add `boundary_mode` switch to monodomain (mirror | zero_pad | node_mirror).
  Phase B — Add absorbing BC variant to LBM.
  Phase C — Mesh-refinement scan for each variant (h, h/2, h/4) to confirm artifact
            scaling.
  Phase D — Final summary figure: per-column LAT deviation across {bath-bidomain,
            mirror-monodomain, zero-pad-monodomain, node-mirror-monodomain}.
  Phase E — Knowledge promotion + cross-reference to MASTER.

Predicted outcome: zero-pad and node-mirror monodomain produce the storage-tank's
boundary shapes; mirror monodomain stays flat; bath-bidomain produces Kleber.
Magnitude of zero-pad / node-mirror artifacts scales as h (vanishes in continuum
limit), magnitude of Kleber stays constant in h. This cleanly classifies the
artifacts as numerical vs physical.

### 2026-04-29 (cont): V5.4 FDM default flipped from node_mirror_existing → face_mirror

After running the 4-mode video benchmark (face_mirror / node_mirror_existing /
zero_pad / rest_pad), confirmed the column-wise mechanism by which
node_mirror_existing produces boundary artifacts and decided to make
face_mirror the V5.4 default.

**Why face_mirror is correct.** For ghost = V[i,0]:
  flux = D · (V_ghost - V[i,0]) / h = D · (V[i,0] - V[i,0]) / h ≡ 0
The ghost dynamically tracks the boundary cell, so the wall is opaque to any
field. By contrast, node_mirror sets ghost = V[i,1] (sub-edge cell), so for
a column-wise wave climbing toward the wall:
  L_y[boundary, node_mirror] = 2 · (V[i,1] - V[i,0])
  L_y[boundary, face_mirror] =     (V[i,1] - V[i,0])
node_mirror amplifies any in-column gradient by exactly 2x at the wall —
this is the root cause of the storage-tank "camel-toe" artifact.

**Why we'd missed it.** node_mirror is the textbook FDM Neumann (Taylor
expansion centered at the boundary node), and for plane waves with uniform-y
stim the difference vanishes. Most cardiac validation tests are 1D cables or
y-uniform 2D plane waves, so the 2x amplification was invisible. The matrix
asymmetry (off-diagonals 2w-vs-w at the boundary, which is genuinely
asymmetric, NOT the symmetric mirror entry the math card incorrectly claimed)
also doesn't break PCG because the operator is self-adjoint in a weighted
inner product. Nothing crashes; you just get the wrong answer at the boundary,
small enough to dismiss as "boundary noise" until the storage-tank study
made the shape artifact visible.

**Empirical evidence.** Peak voltage in our 4-mode TTP06 video, uniform-y line
stim, forward-Euler diffusion, V_rest = -85.23 mV:
  face_mirror              V_max = +53.66 mV   no boundary drag
  node_mirror_existing     V_max = +46.89 mV   2x in-column amplification
  zero_pad                 V_max = +40.54 mV   Dirichlet drag toward 0
  rest_pad (ghost=-85)     V_max = +26.48 mV   strongest Dirichlet drag
The "trail filling up the sides" the user noticed in rest_pad is the
Dirichlet clamp pulling boundary cells back toward V_rest during the AP —
which we confirmed is mathematically identical to applying ghost=V_rest
on-the-fly in the diffusion solver (no encoding changes the math). Only
ghost = V[i,0] gives genuinely zero flux.

**Change.**
- `Monodomain/Engine_V5.4/cardiac_sim/.../fdm.py`:
    `boundary_mode='node_mirror_existing'` → `boundary_mode='face_mirror'`
    BOUNDARY_MODES tuple reordered to (face_mirror, node_mirror_existing,
    zero_pad, rest_pad). Docstring rewritten.
- `test_boundary_modes.py`: `test_a5_default_unchanged` → renamed to
    `test_a5_default_is_face_mirror`. Asserts default == face_mirror AND
    default ≠ legacy.
- `test_phase8.py::test_8v7_backward_compatibility`: pinned the convergence
    sub-test to `boundary_mode='node_mirror_existing'` because its
    manufactured solution V = cos(πx)cos(πy) has Neumann at y=0 (node-centered
    wall), not at y=-h/2 (face-centered wall). face_mirror gives the right
    answer for a different boundary placement; the test is checking O(h²)
    convergence to a specific manufactured solution, so it stays on the
    legacy mode.

**Verification.**
- 5/5 boundary-mode tests pass.
- 7/7 phase 7 tests pass (mesh builder integration).
- 7/7 phase 8 tests pass (per-node conductivity); patched 8-V7 now shows
  ratios 3.99 / 4.00 (textbook O(h²)) on legacy mode and row-sums = 0 on
  the new face_mirror default — both BC modes preserve mass conservation.

**Failed approach (logged):** rest_pad as the "smart fix." The user's
intuition was that ghost = -85 might isolate the boundary while keeping
it physically sensible. It does the opposite: when V[i,0] depolarizes to
+50 mV, the drag toward -85 is -135/h² mV/ms — STRONGER than zero_pad's
-50/h². rest_pad is silent at rest but clamps hardest during the AP. Any
constant-value ghost is Dirichlet in disguise; only ghost = V[i,0]
(face_mirror) gives true Neumann.

**Open follow-up.** A genuine Robin BC `D ∂V/∂n + α(V - V_bath) = 0` would
let the boundary exchange charge with a finite-conductance reservoir
(α → 0 = face_mirror; α → ∞ = rest_pad). That's the actual Kleber
bath-coupled story and deferred to the LBM Phase B work or a possible
mode-5 implementation later.

### 2026-04-29 (cont): The boundary effect is shared — only the amplification differs

After running the 4-mode video and noticing face_mirror and node_mirror look
visually identical, sat down and pushed on whether face_mirror "correctly
models John's storage-tank." The honest answer reframed the entire research
question.

**The realization.** John's storage-tank artifact (camel-toe under reflect_y,
crescent under one_way pumps) and monodomain's near-flat boundary share the
SAME mechanism. Boundary cells have one fewer neighbor → less diffusive
drain → V accumulates / fills faster. This is real in both models. The
difference is amplification:

```
                                  John's tanks            Monodomain face_mirror
─────────────────────────────────────────────────────────────────────────────────
Boundary topology                 3 pipes vs 4            3 neighbors vs 4
Boundary accumulation             YES                     YES
Fill → fire conversion            One-way + threshold     Continuous HH kinetics
                                  binary, irreversible    smooth, bidirectional
Time-difference amplification     ~150x (cascade)         ~1x (Fickian smoothing)
Visible LAT shift                 ms-scale (camel-toe)    µs-scale (~50 µs)
```

**Quantitative scale check:**
```
Quantity                          Value         Notes
─────────────────────────────────────────────────────────────────────
Total AP swing (rest → peak)      ~135 mV       TTP06 EPI
Boundary V overshoot              ~7 mV         face_mirror video data
Overshoot / swing                 ~5%
Upstroke duration                 ~1 ms         sharp Na activation
Upstroke slope                    ~135 mV/ms    near-vertical
Implied LAT shift at boundary     ~50 µs        7/135 ≈ 0.05 ms
Our LAT save_every                500 µs        10x undersampled
```

So the boundary IS firing earlier in monodomain — by about 50 microseconds.
Our LAT measurement floor is 500 microseconds. The artifact isn't absent;
it's beneath our sampling rate. The earlier benchmark
(`benchmark_uniform_init.py`) measured face/node deviation = 1e-15 ms, but
that benchmark used a strictly y-uniform line stim, so ε = 0 at every wall
node and BOTH Neumann modes give L_y = 0 at the boundary. In that setup
neither mode CAN show camel-toe (the math card already noted this).

**Why this matters.** The original research-question hypothesis was:
"camel-toe is a discretization artifact — face-centered mirror eliminates
it." The truth is more nuanced:
1. node_mirror's 2x amplification IS a discretization artifact. Fixed by
   the default flip earlier today.
2. The fewer-neighbor accumulation effect is REAL physics, present in
   both John's tanks and monodomain face_mirror.
3. John's tanks AMPLIFY this real effect into a visible LAT distortion via
   one_way + threshold kinetics. Monodomain's continuous kinetics dilute
   it back to invisibility at typical resolution.

So the camel-toe in John's model is NOT purely a numerical artifact. It's
a real boundary effect that monodomain also has — just below the noise
floor.

**Two follow-up experiment groups designed (PLAN extension).**

Group A — make the boundary effect visible in monodomain by dialing toward
John's regime (smaller domain, sub-ms LAT sampling, sharper Na, sweep
domain size at fixed sharpness). Discriminator: face_mirror vs node_mirror
should give same shift if the effect is purely fewer-neighbor topology;
node_mirror should give 2x larger shift if 2x amplification dominates.

Group B — make the camel-toe disappear in John's tanks by dialing toward
monodomain regime (face_mirror BC swap, soft sigmoid threshold,
bidirectional pumps, larger cell count). Discriminator: cammel-toe should
shrink monotonically as we go from sharp/one-way → smooth/bidirectional.

Together these two groups test the bridge claim "same mechanism, different
amplification" by interpolating between the two regimes and watching the
artifact morph continuously.

**Parallel literature search fired.** Looking for prior work on
sharp-Na-driven boundary acceleration, lattice-automaton-to-PDE bridge in
cardiac modeling, and explicit numerical-vs-physical attribution of the
edge effect. Results below.

### 2026-04-29 (cont): Literature search — scale-amplification framing appears novel

Background search agent ran 9 PubMed/Web searches on boundary acceleration,
no-flux Neumann artifacts, and CA-vs-PDE bridges in cardiac modeling.
Adjacent prior work exists; the specific bridge claim does not.

**Established adjacent work.**

| Paper | Mechanism | Sign of effect | Distinct from us? |
|-------|-----------|----------------|-------------------|
| Sperelakis & Kalloor 2005 (PMID 16144554) | discrete PSpice circuit edge effect | SLOWS transverse CV | discrete-circuit, opposite sign |
| Ramasamy & Sperelakis 2006 (PMID 16875501) | end-effects in single chains | slowing | discrete-circuit lineage |
| Roth 1991 (PMID 1984858) | bidomain bath-coupling | ACCELERATES surface AP | physical, bath-shunt mechanism |
| Henriquez/Trayanova/Plonsey 1990 (PMID 2221506) | foundational bath-coupled bidomain | accelerates at boundary | bath-coupling, not no-flux |
| Bishop & Plank 2011 (PMC3244060, PMC3075562) | bath-loading shunt → curved front | accelerates | bath-coupling, augmented monodomain |
| Cherry/Ehrlich/Nattel/Fenton (PMC3163047) | "Effects of boundaries and geometry on APD" | spatial APD distribution | closest hit; verify if they note V_max overshoot |
| Fenton & Cherry 2005 (PMID 15836267) | phase-field for irregular boundaries | numerical conservation | diffuse-interface, not visibility framing |
| Bueno-Orovio/Pérez-García/Fenton (arXiv 1003.1983) | CA ↔ PDE continuum limit chapter | "PDE from CA is non-unique" | generic, not boundary-specific |
| Kléber Compr Physiol review | conduction overview | bath-coupling | textbook, not ours |

**Five direct quotes captured:**
- Sperelakis 2005: *"the larger the model size, the smaller the relative
  edge area, we conclude that the edge effects slow the transverse velocity."*
  — note: SLOW, not accelerate. Opposite sign of John's camel-toe. Suggests
  in discrete-circuit / RC-network land the boundary effect can go EITHER
  direction depending on coupling rule (one_way vs symmetric).
- Roth 1991: *"the action potential at the surface of the strand leading
  that at the center... rate of rise... varies with depth."*
  — boundary acceleration, attributed to bath shunting (not fewer neighbors).
- Bath-loading review: *"shunting effect increases conduction velocity at
  the tissue edges, causing a curvature in the transmural profile."*
- Fenton/Cherry phase-field: *"ensure current conservation when dealing with
  irregular boundaries"* — purely numerical conservation, not perceptual
  visibility.
- CA-PDE chapter: *"It is relatively straightforward to derive updating
  rules for cellular automata from corresponding partial differential
  equations, however, the reverse is usually very difficult... nonunique."*
  — directly relevant: there is no canonical PDE that John's tanks reduce to.

**Assessment.** The specific scale-amplification framing — *the same
fewer-neighbor boundary mechanism produces a visible camel-toe under
threshold + one-way kinetics but is washed out by bidirectional Fickian +
smooth HH + coarse save-resolution* — appears NOVEL. None of the surveyed
papers articulate the bridge. What IS established:

- Bidomain bath-coupling acceleration is well-documented (Roth, Plonsey,
  Bishop/Plank), but it's a DIFFERENT mechanism (extracellular shunt, not
  intracellular no-flux topology). Sign happens to match.
- Discrete-circuit edge effects (Sperelakis) go the OPPOSITE sign in their
  setup, suggesting the John's camel-toe direction is rule-dependent.
- CA→PDE continuum-limit math exists generically, but no paper specifically
  derives John's storage-tank → monodomain limit or quantifies the
  damping factor (~150x) we estimated.

**Three papers worth acquiring full PDFs of:**
1. Cherry/Ehrlich/Nattel/Fenton PMC3163047 — closest prior work on no-flux
   boundary APD distribution.
2. Roth 1991 PMID 1984858 — to compare our ~7 mV V_max overshoot scaling
   against the bidomain bath result.
3. Fenton & Cherry 2005 PMID 15836267 — to position our discrete↔continuous
   framing relative to phase-field formalism.

**Next steps (informed by lit).** When writing this up, frame as:
"We extend the boundary-effect literature beyond bath-coupled bidomain to
the no-flux fewer-neighbor case in monodomain, and quantify the
amplification gap between discrete-threshold and continuous-Fickian
realizations of the same mechanism."

### 2026-04-29 (cont): Connectivity × threshold-gate ablation — Moore-8 is the smoking gun

After confirming face_mirror in 4-neighbor monodomain shows ZERO boundary
deviation in a single column (V[boundary] = V[center] to 1e-13 mV), the
user pushed back with: "but my Fickian-modified version of John's code
STILL gives a crescent." That contradicted the simple "4-neighbor face-
mirror eliminates boundary effect" narrative.

Read tanks_vec.py carefully and ran a 4-way ablation on John's user-modified
Fickian setup (mode='gradient' + one_way + zero_pad + line geometry):

```
Run                       Connectivity  Threshold gate  max|LAT-mean_y(LAT)|  Verdict
─────────────────────────────────────────────────────────────────────────────────────
R1  baseline              moore8        True            91.8 steps            Full crescent
R2  cardinal-4 only       cardinal4     True             0.0 steps            FLAT
R3  no threshold only     moore8        False           11.5 steps            8x smaller
R4  both off              cardinal4     False            0.0 steps            FLAT
```

**Two clean conclusions:**

1. **Cardinal-4 connectivity gives EXACTLY ZERO deviation** (R2, R4) regardless
   of whether the threshold gate is on. Every column that fully activates has
   `LAT(boundary) == LAT(center)` to floating-point precision. This matches our
   monodomain face_mirror result (1e-13 mV deviation). Cardinal-4 in y-uniform
   line stim simply cannot produce a crescent.

2. **Moore-8 connectivity always produces a crescent** (R1, R3), with the
   threshold gate amplifying by ~8× (R1: 91.8 vs R3: 11.5). The threshold is
   the amplifier; Moore-8 is the source.

**Why Moore-8 is the source:** in y-uniform field with wavefront at column k,
each interior cell has 3 firing inflow pipes (NW + W + SW from upstream) and
3 firing outflow pipes (NE + E + SE). Boundary cells lose NW and NE (off-grid),
giving 2 inflow + 2 outflow = 2/3 of the interior charging rate.

In contrast, the cardinal-4 "missing N pipe" at the boundary already has
gap = 0 in y-uniform (V[i,1] = V[i,0]), so losing it costs nothing.

The diagonal pipes carry x-direction flux even in y-uniform fields because
they span both axes simultaneously — that is the mechanism the cardinal-only
4-neighbor scheme structurally cannot capture, and it is exactly what makes
Moore-8 different.

**This finally bridges the storage-tank ↔ monodomain divergence:**

```
Setup                                    Gives boundary effect?  Reason
─────────────────────────────────────────────────────────────────────────────
John's model (Moore-8 + threshold)       YES (large)             Both ingredients
John's Fickian (Moore-8, no threshold)   YES (small)             Just Moore-8 deficit
Monodomain (4-neighbor + face_mirror)    NO                      Cardinal-only cancels
                                                                  in y-uniform
Hypothetical: Moore-8 monodomain         predicted YES           Moore-8 is necessary
Hypothetical: cardinal-4 John's          predicted NO            CONFIRMED (R2, R4)
```

Ablation code: `simulation/connectivity_threshold_ablation.py`. Outputs
in `simulation/outputs/connectivity_threshold/`. Patches added to
`tanks_vec.py:run()`: new params `connectivity` ('moore8' | 'cardinal4')
and `threshold_gate` (bool). Backward compatible (defaults match prior
behavior).

### 2026-04-29 (cont): LBM-style isotropic weighting (4:1) — smooths but doesn't eliminate

The user noted from LBM theory that proper isotropic weighting (Patra-Kałuża
9-point stencil ratio 4:1 cardinals:diagonals, identical to LBM D2Q9
quadrature weights modulo a normalisation) should mitigate boundary
asymmetry. Tested by adding a 'moore8_iso' connectivity option.

Extended the ablation with R5 (iso + threshold) and R6 (iso, no threshold):

```
Run                      Connectivity   max|LAT-meanY|    iso_max    Notes
─────────────────────────────────────────────────────────────────────────────────
R1   moore8 + thresh     equal-weight   91.8 steps        3996       baseline crescent
R5   iso 4:1 + thresh    weighted 9pt   18.9 steps         412       5x smaller crescent
R6   iso 4:1 no thresh   weighted 9pt   24.2 steps         217       still non-zero
R2   cardinal-4          4-pt           0.0 steps         3792       FLAT
```

**Analytic deficit ratio (boundary charging rate / interior charging rate)
in y-uniform line stim:**
  (w_c + w_d) / (w_c + 2*w_d)

  equal-weight (1, 1) → 2/3   (33% deficit, big crescent)
  isotropic    (4, 1) → 5/6   (17% deficit, half-as-big crescent)
  cardinal     (1, 0) → 1     (0% deficit, flat)

So weighting REDUCES the deficit but does not eliminate it. The 5x reduction
in crescent magnitude is the empirical confirmation. The deficit remains
because at the boundary, off-grid diagonal pipes simply don't fire (zero_pad
valid mask) — proper isotropic 9-point would require **face-centered diagonal
reflection**: ghost(-1, k±1) = V[0, k±1], so the off-grid diagonal contributes
(V_self - V[0, k±1]) just as if the neighbour were real. This is the LBM
"bounce-back" trick generalised to 9-point. None of the BC modes in
tanks_vec.py implement diagonal reflection (the existing reflect_y folds
ghost flux to row[1] but doesn't compute the right ghost VALUE for diagonals).

**Implication for the bridge:** the family of stencils between 4-point cardinal
(no boundary effect, our monodomain default) and 9-point uniform Moore (full
boundary effect, John's default) is *continuously connected* via the
cardinal:diagonal weight ratio. Pushing toward isotropic weighting smooths the
boundary effect but doesn't structurally fix it. Cardinal-only or proper
9-point with reflection are the two ways to actually zero it.

**Adjacent observation about wave speed.** Iso 4:1 propagates ~10x faster
than uniform Moore-8 (iso_max 412 vs 3996 to traverse 80 columns), because
cardinals carry 4x weight in the east direction and dominate the throughput.
This means LBM-style weighting changes the propagation regime, not just the
boundary handling. Comparisons of crescent magnitude across connectivities
need to be careful about normalisation (max|LAT-meanY| in step units is not
the same as a fractional crescent).

**Sharp follow-up from the user (relative vs absolute crescent).** Looking
at the same data, computing relative crescent (max|LAT-meanY| / iso_max):

```
Run                      Connectivity     Absolute     Relative crescent
                                          step lag     (% of total time)
─────────────────────────────────────────────────────────────────────────
R1  uniform + thresh     moore8           91.8 steps   2.30%
R3  uniform, no thresh   moore8           11.5 steps   0.30%
R5  iso + thresh         moore8_iso       18.9 steps   4.59%   ← 2x larger relative than R1
R6  iso, no thresh       moore8_iso       24.2 steps  11.15%   ← 37x larger relative than R3
```

Two surprising patterns:
1. **Threshold gate REDUCES absolute crescent in iso mode** (R5 < R6: 18.9 < 24.2),
   opposite of uniform mode (R1 > R3: 91.8 > 11.5). With cardinals dominating
   transport, threshold delays apply less to the deficit-bearing diagonals.
2. **Iso mode has LARGER relative crescent** in both threshold variants. The
   17% deficit per step compounds over fewer total steps, producing a bigger
   fraction-of-propagation-time lag than the 33% deficit over more steps.

**Conceptual implication.** LBM's "isotropic stencil smooths boundary
asymmetry" claim holds only when paired with proper boundary handling
(bounce-back, ghost reflection). The weighting alone is a necessary but not
sufficient condition. Without it, iso weighting just relocates the boundary
effect from "ms-scale lag in a slow wave" to "shorter-step lag in a fast
wave" — same physical asymmetry, different time scale. The shape distortion
(visible to the eye) is actually larger in iso mode because the wavefront's
transit time is compressed.

To genuinely zero the crescent with 8-neighbour, need both:
  (a) Isotropic 4:1 weights AND
  (b) Diagonal-aware face_mirror reflection at the wall:
      ghost(-1, k±1) = V[0, k±1]   for the off-grid NW, NE pipes
      so they contribute (V_self - V[0, k±1]) just as in-grid neighbours would.

This is the LBM bounce-back trick generalised. Not implemented yet; would
require new BC mode in tanks_vec.py. **Group B5 experiment.**

### 2026-04-29 (cont): Bug fix — iso weights need 1/6 normalisation prefactor

The iso 4:1 weighting was first implemented as raw multiplicative weights
(cardinal × 4, diagonal × 1) without the canonical 1/6 prefactor of the
Patra-Kałuża 9-point Laplacian:

  ∇²V ≈ (1/6h²) · [4·(cardinals) + 1·(diagonals) − 20·V_self]

The 1/6 IS the operator's normalisation — without it, the discrete operator
computes 6× the true Laplacian magnitude. This made D_eff = 6·k = 0.48 at
k=0.08, well above the explicit-2D-diffusion CFL limit of 0.25. The result
was grid-scale mosaic instability (checkerboard pattern), which the user
flagged on visual inspection of the rendered video.

**Fix.** Multiply both weights by 1/6: cardinal → 4/6, diagonal → 1/6.
D_eff = (4/6 + 2·1/6) · k = 1·k = 0.08 (same as cardinal-4 / 5-point
Laplacian, fully stable). Deficit ratio (4/6 + 1/6) / 1 = 5/6 unchanged —
so the boundary asymmetry is the same.

**Re-running ablation with fixed weights gives a substantively different
empirical conclusion:**

```
Run                      Connectivity    Threshold    max|LAT-meanY|     iso_max
─────────────────────────────────────────────────────────────────────────────────
Pre-fix (buggy):
  R5  iso + threshold    moore8_iso 6×   True          18.9 steps        412     ← mosaic
Post-fix:
  R5  iso + threshold    moore8_iso 1×   True          81.0 steps       3898     ← clean
  
                                         crescent reduction vs R1 (91.8):
  Pre-fix:                                ~5x smaller (artifact of fast wave)
  Post-fix:                                ~12% smaller (matches 5/6 deficit prediction)
```

**Substantive update to the LBM-isotropic claim.** The pre-fix conclusion
("iso weighting reduces crescent by 5x but doesn't eliminate") was
artifactual. The corrected conclusion: iso 4:1 weighting reduces the
crescent by only ~12%, exactly matching the deficit-ratio prediction
(1/3 → 1/6 = halving the deficit, not halving the crescent). The threshold
gate's nonlinear amplification means deficit reduction translates very
weakly into visible crescent reduction.

This actually STRENGTHENS the "Moore-8 connectivity is the smoking gun"
finding — iso weighting only shifts the deficit ratio from 1/3 to 1/6,
which barely moves the threshold-amplified crescent. Cardinal-only is the
only structural fix.

**Lesson for our practice.** When implementing higher-order discrete
operators with weighting schemes, the normalisation prefactor IS part of
the operator definition, not an afterthought. CFL stability is a sharp
diagnostic for forgotten normalisation. Worth checking D_eff·Δt/Δx² ≤ 1/4
against any weighted-stencil implementation before reading off scientific
conclusions.

### 2026-04-30: Wave-slowing dilation as the dominant apparent-curvature artifact

User's pushback: "the apparent crescent-curvature growth as wavefront moves
outward in John's isochrone images is fake — it's wave-slowing dilation,
not real per-step deficit growth." Investigated by computing per-column
metrics (mean_x = arrival, Δmean = traversal time, spread = max-min LAT,
spread/Δmean = fractional lag).

**Empirical breakdown for R1 baseline (uniform Moore-8 + threshold):**

```
column       Δarrival       spread       spread/Δmean
─────────────────────────────────────────────────────
   4            14            6              0.42
  12            50           28              0.56
  21            91           59              0.65
  29           128           88              0.69
  37           165          117              0.71
```

- Δarrival grows 12× across the propagation (wave decelerating)
- Spread grows 20× (apparent crescent growth)
- spread/Δmean grows ~1.7× (REAL per-step deficit growth, on top of slowing)

So user's claim is ~70% correct: ~12 of the ~20 crescent-magnitude growth
is from wave deceleration. The remaining ~1.7× is genuine compounding from
threshold-amplified discrete activation (each column's boundary lag is
inherited by next column with no back-flow to cancel it).

**Confirming experiment (no threshold, R3): spread plateaus at 17 steps and
spread/Δmean DECREASES from 0.22 → 0.08 as wave moves outward.** Without
threshold gating, the per-step deficit really is constant (modulo
diffusive smoothing), and apparent growth is purely the dilation artifact.
The compounding requires the threshold to lock in inherited lag.

**Classification:**
| Regime | Wave slows? | Spread grows? | Per-step deficit growth? |
|---|---|---|---|
| R1 uniform + thresh | YES (12×) | YES (20×) | YES (1.7×) — compounding |
| R5 iso + thresh     | YES (16×) | YES (23×) | YES (1.5×) — compounding |
| R3 uniform, no thresh | YES (12×) | grows then plateaus | NO — constant per step |

### 2026-04-30: Normalized isochrone / per-column LAT diagnostics

Built `render_norm_helpers.py` with two normalization tools:
1. `x_evenly_spaced_levels(iso, N)` — picks contour LEVELS at evenly-spaced
   x-positions of wavefront mean (NOT step times). Removes wave-slowing
   dilation in isochrone plots.
2. `per_column_dev_normalized(iso, x)` — returns dev/Δmean (fractional lag
   relative to per-column traversal time) instead of raw step counts.
   Removes dilation in per-column LAT plots.

Generated 6 normalised figures in `simulation/outputs/images/`:
- `pump_speed_isochrones_normalized.png`     — re-run of John's BASELINE sweep
- `gradient_k_isochrones_normalized.png`     — new for Fickian k sweep
- `pump_speed_per_column_lat_normalized.png` — pump-speed per-column LAT
- `gradient_k_per_column_lat_normalized.png` — gradient-k per-column LAT
- `connectivity_isochrones.png`              — uniform / cardinal / iso comparison
- `connectivity_per_column_lat.png`          — same comparison, per-column LAT
                                                (top: absolute, bottom: normalised)

Render scripts: `render_pump_speed_isochrones_normalized.py`,
`render_gradient_k_isochrones_normalized.py`,
`render_connectivity_isochrones.py`, `render_connectivity_per_column_lat.py`,
`render_pump_speed_per_column_lat_normalized.py`,
`render_gradient_k_per_column_lat_normalized.py`.

After normalisation, curves at different x within each panel should overlay
if per-step deficit is constant. They don't quite overlay (compounding
residual) but the spread between curves is much smaller than in the absolute
plots — confirming the dilation hypothesis for ~70% of the visible effect.

## Session Log

### 2026-04-30 Session
**Worked on**: V5.4 monodomain face_mirror default flip (post-audit), four-mode
boundary video (face/node/zero_pad/rest_pad), John's storage-tank ablation
to localise the crescent mechanism, isotropic 9-pt Patra-Kałuża implementation
(buggy then fixed), normalised isochrone/per-column LAT diagnostics to
factor out wave-slowing.

**Accomplished**:
- V5.4 FDM default `boundary_mode` flipped from `node_mirror_existing` →
  `face_mirror`. Eliminates the 2× column-gradient amplification at the
  wall. Five boundary-mode tests pass; Phase 7 (7/7) and Phase 8 (7/7 after
  patching test_8v7 to pin its O(h²) sub-test to legacy mode since its
  manufactured solution V=cos(πx)cos(πy) has node-centered Neumann).
- Monodomain column diagnostic confirms zero boundary effect with
  face_mirror in y-uniform line stim: V[boundary] = V[center] to 1e-13 mV
  precision, LAT identical to picosecond resolution.
- John's storage-tank: traced his "Fickian-modified" code and confirmed
  the gradient mode STILL has the `fired_p & gap > 0` threshold gate.
  Added two new params to `tanks_vec.run()`: `connectivity` ('moore8' /
  'cardinal4' / 'moore8_iso') and `threshold_gate` (bool).
- 6-way ablation R1-R6 establishes: (a) cardinal-4 connectivity → 0.0 step
  crescent (R2/R4 in y-uniform stim — flat to floating point); (b) Moore-8
  always crescents (R1=91.8, R3=11.5, R5=81, R6=16.5); (c) threshold gate
  amplifies by ~8× under uniform Moore-8 weights.
- Iso 4:1 (Patra-Kałuża) implementation: first attempt missed the 1/6
  normalisation prefactor → D_eff = 6k = 0.48, well above 2D explicit-CFL
  limit (0.25), produced grid-scale mosaic instability (visible in user's
  spot-check of `iso_thresh_clean.mp4`). Fixed by absorbing 1/6 into the
  weights (cardinal=4/6, diagonal=1/6). Re-rendered videos clean. Crescent
  reduced ~12% (not the ~5× the unstable run faked).
- Wave-slowing-dilation finding: ~70% of the apparent per-column crescent
  growth in John's images is wave deceleration (Δarrival grows 12× across
  the run); the remaining ~1.7× is real threshold-amplified compounding.
  Without threshold (R3), spread/Δmean decreases — pure deceleration story.
- Built `render_norm_helpers.py` and 6 normalised figures to factor out
  the dilation artifact.
- Literature search via background research agent found: scale-amplification
  framing (same fewer-neighbour mechanism, dilution factor set by
  threshold-sharpness + bidirectional smoothing + save resolution) is
  apparently NOVEL. Adjacent: Roth 1991, Henriquez/Plonsey 1990, Bishop &
  Plank 2011 (bath-coupling), Sperelakis 2005 (discrete-circuit edge
  effects), Cherry/Ehrlich/Nattel/Fenton (PMC3163047, closest hit on
  no-flux boundary APD), Fenton/Cherry phase-field (PMID 15836267).

**Next**: Implement Moore-8 (uniform) and Moore-8-iso (4:1 Patra-Kałuża)
stencils in monodomain V5.4 FDMDiscretization, plus a custom-weight
9-point option in LBM V1 D2Q9 (currently uses canonical 1/9, 1/36 LBM
weights, which IS the iso ratio). Test whether the boundary effect
appears in monodomain when 8-neighbour connectivity is enabled — predicted
to match John's via the same fewer-diagonals-at-wall mechanism. Will run
through `/blueprint` → `/audit` before implementing.

### 2026-04-30 (overnight execution): PLAN.md fully delivered, bridge claim CONFIRMED

5-phase PLAN executed end-to-end across 4 commits:

- **Phases 1+2** (V5.4 FDM): added `stencil` parameter (cardinal4 / moore8_uniform /
  moore8_iso) + `face_mirror_iso` boundary mode to FDMDiscretization. 21/21 boundary
  tests + 14/14 phase 7/8 regressions. Verified deficit ratios numerically:
  moore8_uniform→2/3, moore8_iso→5/6, cardinal4→1, all match prediction.
- **Phase 3** (column diagnostic): ported John's storage-tank ablation to monodomain
  V5.4 across 5 (stencil, BC) pairs. The headline finding:

```
case                                   max|top-ctr|  ΔLAT (µs)  Verdict
─────────────────────────────────────────────────────────────────────
cardinal4 + face_mirror               1.1e-13 mV    0          baseline ✓
moore8_uniform + face_mirror          70.5  mV    +486          DEFICIT ✓ (John-equivalent in monodomain)
moore8_uniform + face_mirror_iso      2.8e-14 mV    0          fix ✓
moore8_iso + face_mirror              48.4  mV    +230          smaller deficit ✓ (5/6 vs 2/3)
moore8_iso + face_mirror_iso          7.5e-14 mV    0          LBM analog ✓ (full fix)
```
The ΔLAT = +486 µs at moore8_uniform + face_mirror is the smoking gun: monodomain
PDE reproduces John's crescent-shaped boundary lag when the stencil + BC are
faithful to John's discrete topology. Boundary fires LATER than bulk (consistent
with fewer-neighbours → less inflow → slower charging).

- **Phase 4** (LBM V1): added `D2Q9_uniform` lattice (1/8 weights, cs2=0.75) +
  `weights_mode` parameter to `LBMSimulation`. CRITICAL latent bug fixed in
  `simulation.py:73`: was calling `tau_from_D(D, dx, dt)` with default cs2=1/3
  regardless of lattice; now passes `cs2=self.lattice.cs2`. Bit-correct for
  canonical D2Q9 (cs2=1/3=default), required for D2Q9_uniform (cs2=0.75).
  32/32 LBM tests pass (8 new + 24 existing).

- **Phase 5** (cross-engine validation figure): 3×3 panel showing the bridge
  claim across storage-tank / monodomain / LBM. Numerical summary:

```
                                Deficit (max|top-ctr|)  Baseline  Fix
Storage tank (LAT spread):       130 steps              0         103 (iso reduces)
Monodomain (mV at x=mid):        1.76 mV                0         0   (full fix)
LBM (mV at x=mid):               0.077                  0         0.044 (sub-mV; LBM
                                                                  small effect overall)
```

All success criteria met. The connectivity-mediated boundary effect is
confirmed as a unified mechanism across discrete (storage tank), continuum
PDE (monodomain), and lattice Boltzmann frameworks. Cardinal-only or
diagonal-aware bounce-back eliminates it in all three.

**Commits** (main):
- `a660f8be` V5.4 FDM Phases 1+2
- `6c21db14` Phase 3 monodomain column diagnostic
- `060b9dce` Phase 4 LBM D2Q9_uniform + cs2 plumbing fix
- `b45f39f3` Phase 5 cross-engine validation figure

**Files created**:
- `Monodomain/Engine_V5.4/cardiac_sim/.../fdm.py` (modified, +385 lines)
- `Monodomain/Engine_V5.4/test_boundary_modes.py` (rewrite, 569 lines, 21 tests)
- `Research/Active/boundary_conduction_speedup/diag_monodomain_connectivity.py`
- `Research/Active/boundary_conduction_speedup/figures/diag_monodomain_connectivity.png`
- `LBM/Engine_V1/src/lattice/d2q9_uniform.py`
- `LBM/Engine_V1/src/lattice/__init__.py` (export D2Q9_uniform)
- `LBM/Engine_V1/src/simulation.py` (modified: weights_mode param + cs2 plumbing fix)
- `LBM/Engine_V1/tests/test_d2q9_uniform_weights.py` (8 tests)
- `Research/Active/boundary_conduction_speedup/figures/connectivity_cross_engine.{py,png,pdf}`

## Session Log
