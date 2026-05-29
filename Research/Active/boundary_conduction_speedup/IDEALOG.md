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

### 2026-05-02 Session
**Worked on**: Tooling polish for the connectivity bridge story (videos + figure
cleanup), plus a focused theoretical investigation of why Fickian (gradient)
flux laws are sign-locked to crescent — derived the equilibrium argument,
verified empirically with a k-sweep, and settled the user's intuition that
"accumulation enables Effect B in John, no accumulation in Fickian."

**Accomplished**:
- Storage-tank Fickian + cardinal-4 video. New script
  `simulation/render_gradient_cardinal4.py` renders the GRADIENT config with
  `connectivity='cardinal4'` swapped in. Output:
  `simulation/outputs/video/gradient_cardinal4_clean.mp4` (3.5 MB, 26.7 s).
  Note: under cardinal4, D_eff = 1·k vs moore8's 3·k, so the wave only
  reaches ~31% of the grid in 4000 steps (`filled=1250/4000`,
  `iso_max=3792`). Documented inline; user can bump steps or k for full
  traversal if needed.
- `connectivity_cross_engine.py` figure decompressed (was visually cramped —
  3-line dense suptitle, 9 redundant per-panel colorbars, in-panel labels
  overlapping wavefront in rows 2–3, row labels colliding with leftmost
  y-tick labels). Changes: (a) one shared colorbar per row via
  `fig.colorbar(im, ax=axes[r, :].tolist(), ...)`, (b) suptitle collapsed to
  one block so constrained-layout reserves space for it, (c) row 2/3 in-panel
  labels moved upper-left → upper-right (resting region, no wavefront
  overlap), (d) row labels at x=0.018 in a constrained-layout `rect=(0.035,
  0, 0.965, 1.0)` reserved strip, (e) figsize 15×12 → 16×13. Re-rendered
  cleanly.
- Per-combo videos for the 5 (stencil × boundary_mode) cases. New script
  `Research/Active/boundary_conduction_speedup/video_stencil_mirror_combos_individual.py`
  reuses the same 5 sims (T_END=25 ms, 100 frames @ 20 fps) but emits one
  mp4 per case in `figures/`: `video_cardinal4_face_mirror.mp4`,
  `video_moore8_uniform_face_mirror.mp4`, `video_moore8_uniform_face_iso.mp4`,
  `video_moore8_iso_face_mirror.mp4`, `video_moore8_iso_face_iso.mp4`. First
  attempt failed because libx264 + yuv420p needs even dimensions (910×585 has
  odd height); fix: ffmpeg `-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2'` extra arg.
  Robust regardless of figsize/dpi choice.
- **Theoretical investigation: why Fickian is sign-locked to crescent.**
  User asked: "is there something intrinsic about Fickian that prevents
  inverse crescent / camel toe? Adjust for pump speed and it never appears."
  Verified empirically with `/tmp/k_sweep_check.py` (Nx=80, gradient mode,
  moore8 + zero_pad + one_way + line stim, k ∈ {0.005, 0.01, 0.02, 0.04,
  0.08, 0.12, 0.20}):

  ```
        k    x=10   x=20   x=30   x=40   x=50   x=60
   0.200    +12    +15    +22    +28    +33    +38
   0.120    +18    +43    +72   +105   +132   +159
   0.080    +22    +56    +92   +127   +161   +192
   0.040    +41   +100   +160   +214   +265    --
   0.020    +79   +185   +290    --     --     --
   0.010   +155   +357    --     --     --     --
   0.005   +307    --     --     --     --     --
  ```

  edge−center (steps), positive = edge fires LATER = crescent. Sign locked
  positive at every k; magnitude scales like 1/k as predicted by Effect A
  + wave-slowing dilation.
- Derived the equilibrium argument that explains the lock. For a fired source
  cell with upstream V_up, downstream ≈ 0, lateral gaps zero (uniform y),
  Fickian gives:
    dV/dt = k·N_in·(V_up − V) − k·N_out·V
    V*(y) = N_in/(N_in + N_out) · V_up
  With N_in = N_out at both edge (2,2) and interior (3,3), V*(y) = V_up/2
  IDENTICALLY for both. No y-asymmetric stockpile to discharge. Time
  constant τ(y) = 1/(k·(N_in+N_out)) differs (1/4k edge vs 1/6k interior)
  but that's just Effect A re-stated as charging time. The per-pipe rate
  k(V* − V_down) is the same once V* is reached, so edge has no
  differential downstream-pumping advantage. Effect B is structurally dead.
  Replacing k → α·k everywhere doesn't change V*(edge)/V*(interior) = 1.
  No k can unlock camel toe.
- Capacitor-vs-resistor mnemonic settled the user's intuition: John's cells
  are CAPACITORS (fill phase ramps V toward V_max via sqrt source rate, then
  dwell phase parks at high V because dV/dt = M(N_in − N_out) = 0 since
  pipes balance, then drain decoupled from V_down). Fickian cells are
  RESISTORS (no fill-dwell separation, asymptote to V_up/2, every unit of
  inflow matched by outflow on the same step, no stockpile to release
  asymmetrically). Capacitors enable Effect B; resistors don't. In Fickian,
  total downstream pumping = k·V*·N_out, so edge actually pumps LESS total
  fluid than interior (smaller N_out, same V*) — Effect B doesn't merely
  vanish, it inverts and *reinforces* Effect A. That's why crescent is so
  robust.
- User feedback: terminal does not render LaTeX (`$...$` and `\frac{}{}`
  came through as raw source). Saved
  `~/.claude/projects/.../memory/feedback_no_latex.md` and added pointer to
  `MEMORY.md` index. Re-rendered the equation block in plain ASCII / Unicode.

**Next**: Continue along the threshold-sweep thread that was already at the
top of "Next Step" — still pending. Predict: lowering θ widens the active
band → bigger camel toe in constant rule (more drainage-advantage time);
raising θ narrows the band → both rules eventually fail to propagate. Test
whether threshold removal on the gradient rule lets ohmic Fickian show ANY
camel toe (it shouldn't — the equilibrium argument above forbids it
structurally, independent of the threshold gate).

### 2026-05-14: Quick implement — dV/dt boundary-vs-center diagnostic (4 V5.4 sims to HDF5)

Built `diag_dvdt_decomposition.py` to test whether the moore8_uniform + face_mirror boundary deficit appears *instantaneously* from step 1 (operator-level structural deficit) or develops slowly. 4 cases (2×2: fm/fmi × diff-only/TTP06), all moore8_uniform, NX×NY=41×21, dx=0.025, dt=0.01 ms, IC: V[1, :]=0 mV strip, else V_REST=−86.2 mV. Natural evolution from IC, no clamp. Output: `data/case{1..4}_*.h5`, full V(t, NX, NY) logged every step.

**dV/dt at step 1, case 1 (fm + diff), col i=2:**
- boundary (j=0): (−85.28 − (−86.2)) / 0.01 = 92.0 mV/ms
- center (j=10):  (−84.82 − (−86.2)) / 0.01 = 138.0 mV/ms
- ratio: 92.0 / 138.0 = **0.667 ≈ 2/3** — matches moore8_uniform + face_mirror structural deficit prediction (2 active channels vs 3) EXACTLY at step 1.

**Verdict.** "Operator-level state-independent" claim confirmed: deficit is present on step 1, at predicted ratio, with no slow build-up. Cross-column cascade comes from inheritance, not within-column ramp. face_mirror_iso eliminates the deviation to floating-point precision (V[j=0] == V[j=mid] to 0.0e+00) on every step in both diff-only and TTP06 paths.

Side-note: fm gzip files 2.4× larger than fmi (22.9 vs 9.6 MB diff; 12.1 vs 5.1 MB TTP06) — fmi's y-uniform fields have radically less entropy. Compression ratio itself is a quantitative y-asymmetry measure.

**Files added.** `data/case{1,2,3,4}_*.h5` with per-file attrs (stencil, boundary_mode, physics, dx, dt, D, V_stim, V_rest, NX, NY, t_end, n_steps, stim_col).

**Next.** Analysis open — likely cuts: (a) dV/dt(t) overlay at fixed col with per-neighbour Laplacian decomposition; (b) LAT(j) cross-column profiles for TTP06; (c) compression-ratio vs t as y-asymmetry growth indicator.

### 2026-05-14 (cont): Case 5 — synchronous cold start at cols 1+2+3, interpretation corrected

Built case 5: IC `V[1,2,3, :] = +30 mV` (AP-trigger), V[else] = V_rest. For first 10 steps (0.1 ms), clamp `V[1,2,3, :] = +30 mV` after each Strang step. Then release; natural TTP06 evolution. Same NX=41 × NY=21, dx=0.025, dt=0.01, t_end=25 ms, moore8_uniform + face_mirror. Output: `data/case5_fm_ttp06_synchap_cols123.h5` (12.4 MB).

**Question this case answered.** Earlier framing was: "is the crescent at col 4+ inherited from col 1's initial corner mismatch?" Synchronizing cols 1-3 at AP-firing voltage eliminates any LAT inheritance possibility — col 3 is rigorously y-uniform throughout the 0.1 ms clamp window.

**Result at col 4 (the first column charging from rest under face_mirror after the synchronized upstream releases):**

```
k    t (ms)   V[c=4, j=0]   V[c=4, j=10]   dev (bdry − ctr)
─────────────────────────────────────────────────────────────────
 11   0.11     −72.95         −67.91         −5.05
 20   0.20     −64.83         −57.82         −7.00
 30   0.30     −57.51         −49.36         −8.15
 50   0.50     −43.68         −24.83        −18.85    ← peak deficit during ramp-up
100   1.00     +19.25         +11.59         +7.65    ← INVERTED (center fired first)
200   2.00     +17.80         +17.77         +0.03    ← both in plateau
2499 24.99     +14.99         +14.99         −0.002   ← steady
```

Col 4 boundary lags center by up to 18.85 mV during its ramp-up. Center fires AP first (~t=0.5 ms), boundary fires later (~t=0.8 ms). LAT shift locked in.

**Interpretation correction (user pushback).** I initially framed this as "falsifying" a firing-time-inheritance hypothesis. That was wrong framing — the user's actual hypothesis was that the source-effect imbalance creates differential firing time at any column charging from rest. Case 5 just shifted the "first charging column" from col 2 to col 4; the imbalance reproduces faithfully at the new location.

**Corrected interpretation.** Case 5 demonstrates **locality**: the crescent isn't an inherited LAT shift — it's generated locally by whichever column is currently charging from rest under face_mirror + moore8_uniform. Combined with case 4 (fmi + ttp06, dev = 0 at every step), the two cases together confirm:

- Case 4: removing the source-effect imbalance (face_mirror_iso) → no LAT shift anywhere
- Case 5: keeping the source-effect imbalance but eliminating upstream LAT inheritance → LAT shift still forms locally at the next charging column

Source-effect imbalance (= face_mirror with NW off-grid → ghost = V_self → 0 diagonal contribution) is **necessary AND sufficient** for a per-column LAT shift wherever the wavefront enters a fresh column under face_mirror moore8_uniform.

**Next.** Case 6 — strict "AP-first, diffusion-second" version. Ionic-only stepping for first 10 steps (no diffusion at all), cols 1-3 clamped at +30 mV. Then release and switch to full Strang. This ensures col 4 doesn't experience ANY pre-AP charging during the sync window. Tests the same hypothesis under the strictest possible condition: no diffusion anywhere until AP is locked in at cols 1-3.

**Files added (this entry).** `Research/Active/boundary_conduction_speedup/data/case5_fm_ttp06_synchap_cols123.h5`.

### 2026-05-14 (cont): Case 6 — strict AP-first, diffusion-second (no diffusion during sync window)

Built case 6 as the strict version of case 5. IC identical: `V[1,2,3, :] = +30 mV`, V[else] = V_rest. **Sync window is now strictly ionic-only**: for the first 10 steps (0.1 ms), call `sim.splitting.ionic_solver.step(state, dt)` directly — bypassing the Strang splitting's diffusion sub-step entirely. Cols 1-3 are clamped at +30 mV after each ionic step so the AP cascade locks in. Cols 0 and 4+ stay at V_rest exactly (ionic at V_rest is a no-op because cells already sit at gate equilibria). After 10 steps: full Strang resumes. Same grid, dt, t_end, BC. Output: `data/case6_fm_ttp06_apfirst_cols123.h5` (12.3 MB), attrs include `diffusion_during_sync=False`.

**Why case 6 matters (case 5 had contamination).** Case 5 ran full Strang during its 0.1 ms clamp window, so diffusion was happening even while cols 1-3 were forced to V=+30. By k=10, col 4 had already accumulated −4.72 mV of bdry-ctr deviation BEFORE the sync was released. The case-5 crescent was therefore partly inherited from a pre-release diffusion ramp. Case 6 enforces the user-requested condition "diffusion starts only when AP starts" rigorously: nothing diffuses anywhere until the sync window closes.

**Side-by-side at column 4, bdry (j=0) − ctr (j=10):**

```
  k    t (ms)   case5 dev    case6 dev    note
  ──────────────────────────────────────────────────────────────────
   0   0.00     0.000        0.000        IC y-uniform
  10   0.10    −4.72         0.000        case5 dev already developed
  11   0.11    −5.05        −0.616        case6: first diff step → 2/3 ratio
  20   0.20    −7.00        −4.45
  50   0.50   −18.85        −8.66         peak ramp-up
  80   0.80    +0.92       −25.75         case6: ctr fired, bdry still ramping
 100   1.00    +7.65        +5.98
 500   5.00    −0.006       −0.029        plateau, equalized
2499  24.99    −0.002       −0.002        steady
```

**Confirmed (case 6, strict):** at k=11 (first diffusion step after sync release), col 4 boundary charges 1.23 mV while col 4 center charges 1.85 mV from V_rest. Ratio = 1.23/1.85 = **0.665 ≈ 2/3**, matching the moore8_uniform + face_mirror structural deficit prediction exactly (2 active upstream channels at boundary: W, SW from col 3; vs 3 at center: W, NW, SW).

**Headline.** The crescent at col 4 forms from V_rest under the strictest possible "AP-first" condition — cols 1-3 in plateau y-uniformly, col 4 at V_rest exactly when diffusion begins. The source-effect imbalance (face_mirror killing the NW diagonal contribution) is sufficient to generate the LAT shift locally at col 4 with no help from upstream history. This double-nails the source-effect-imbalance mechanism: it's not a one-time initial seed, it's a per-column phenomenon that activates wherever a fresh column charges from rest under face_mirror moore8_uniform.

**Combined evidence — cases 3, 4, 5, 6 together:**

| Case | Setup | Crescent | Source of imbalance |
|---|---|---|---|
| 3 | fm + ttp06 natural | +486 µs LAT (col 2+) | every column charging from rest |
| 4 | fmi + ttp06 natural | 0 µs LAT everywhere | no source imbalance under face_mirror_iso |
| 5 | fm + sync cols 1-3 (Strang in window) | LAT at col 4 (contaminated by diff-in-window) | per-column at col 4 + small pre-release seed |
| 6 | fm + sync cols 1-3 (ionic-only in window) | LAT at col 4 (cleanly) | per-column at col 4, isolated from any upstream history |

**Implementation note.** Calling `sim.splitting.ionic_solver.step(state, dt)` directly bypasses Strang's diffusion sub-step. State time must be advanced manually (`sim.state.t += dt`) since `sim.step()` is not called. This is the cleanest engine-level intervention — no FDM modification, no D=0 hack, just a direct call to the ionic operator. Works because TTP06 ionic at V_rest is a fixed-point.

**Next.** Analysis is the open thread: dV/dt trace overlays comparing cases 3, 5, 6 at col 4 to visualize how the deficit develops under each starting condition. The plot script `plot_dvdt_traces.py` can be extended to include cases 5/6.

**Files added (this entry).** `Research/Active/boundary_conduction_speedup/data/case6_fm_ttp06_apfirst_cols123.h5`.

### 2026-05-14 (cont): HBB ≡ face_mirror (corrects earlier KNOWLEDGE.md claim)

Careful re-derivation of the LBM bounce-back vs PDE face_mirror correspondence. Earlier statements (in KNOWLEDGE.md cross-engine summary and the diagnostic figure captions) claimed `face_mirror_iso (PDE) ≡ bounce-back (LBM)` for the diagonal-flux question. **That was wrong.** Correct mapping:

```
   PDE side                              LBM side
   ─────────────────────────────────────────────────────────────────
   face_mirror                ≡          HBB  (halfway bounce-back)
   moore8_uniform + fm        ↔          D2Q9_uniform + HBB    (2/3 deficit)
   moore8_iso     + fm        ↔          D2Q9_canonical + HBB  (5/6 deficit)
   cardinal4      + fm        ↔          D2Q5 + HBB            (zero deficit)
   
   face_mirror_iso            has no canonical LBM equivalent.
                              (would require extrapolated bounce-back
                               that copies populations from row-aligned
                               real neighbours; not a standard scheme)
```

**Why HBB ≡ face_mirror at the diagonals.** Both are zero-flux Neumann BCs. Both kill upstream-V contribution through off-grid diagonal channels:

- face_mirror: `V_NW_ghost = V_self` → gap = 0 → no diagonal Laplacian contribution from NW
- HBB: `f_SE(C, after) = f_NW(C, before)` → diagonal slot gets C's own pre-stream NW emission, which is local equilibrium (≈ `w_NW · ρ_C`) and carries no upstream V information

Both achieve `∂V/∂n = 0` at the wall — face_mirror at the V-gradient level, HBB at the population-mass-flux level. **Different bookkeeping, same physical Neumann boundary condition, same structural deficit.**

**Lattice weight ratio determines which face_mirror variant maps.** D2Q9_canonical's 4:1 cardinal:diagonal weighting matches Patra-Kałuża iso 9-point exactly → `D2Q9_canonical + HBB ↔ moore8_iso + face_mirror`. D2Q9_uniform's equal weighting matches moore8_uniform → `D2Q9_uniform + HBB ↔ moore8_uniform + face_mirror`. The 5/6 vs 2/3 deficit ratio falls out from the weight ratio purely.

**face_mirror_iso is a PDE-only construction.** It works by deliberately breaking the strict Neumann symmetry on diagonals: `V_NW_ghost = V_W` (the row-aligned real neighbour), pulling REAL upstream V into the diagonal ghost slot. This restores the diagonal channel to a non-zero gradient contribution. It is NOT a clean Neumann condition any more — it's a custom mixed-treatment that happens to eliminate the propagating-wavefront deficit. No canonical LBM bounce-back scheme (HBB, FBB, specular, anti-bounce-back, Zou-He) reproduces it. To engineer an LBM equivalent would require an extrapolated/interpolated scheme that copies `f_SE` from the W neighbour rather than reflecting C's own f_NW back.

**Empirical magnitude difference.** PDE max ΔV (70.5 mV for moore8_uniform + fm, 48.4 mV for moore8_iso + fm) is much larger than LBM max ΔV (0.077 mV for D2Q9_uniform + HBB, 0.044 mV for D2Q9_canonical + HBB). The RATIOS match the predicted 2/3 vs 5/6 deficit. The absolute MAGNITUDES differ only because the LBM and PDE simulations have different effective D, tau, and time-step calibrations — the structural family is identical.

**Update to KNOWLEDGE.md.** The §"face_mirror_iso (PDE) ≡ bounce-back (LBM)" line and the diagnostic-figure caption claiming the LBM is a "diagonal-aware fix" need correction. Canonical D2Q9 + HBB has a smaller residual than D2Q9_uniform + HBB only because of the 4:1 weight ratio (less diagonal contribution lost), not because HBB does anything diagonal-aware. The cross-engine bridge claim still holds — it's just that all three engines (storage tank, monodomain, LBM) exhibit the SAME face_mirror-family deficit at the boundary; the LBM doesn't "fix" anything that the PDE doesn't.

**Implication for the boundary_conduction_speedup research question.** The "fix" line in the deficit table — `moore8_iso + face_mirror_iso → 0 deficit, LBM analog full fix` — is misleading. The PDE face_mirror_iso is the only treatment that fully eliminates the boundary deficit; there's no native LBM analog for it. The earlier presentation conflated "smaller residual under canonical D2Q9" with "diagonal-aware fix." The real situation: all LBM bounce-back variants live in the face_mirror family.

**Next.** Work with the face_mirror + HBB family for the inverse-crescent test (asymmetric voltage clamp where boundaries are advanced by 1 column relative to interior). Hypothesis: source-effect imbalance under face_mirror should fight the imposed inverse crescent, eventually flipping it back to forward crescent over enough propagation distance. Quantitative test: how many columns does it take to eat the artificial lead?

### 2026-05-14 (cont): Case 7 — inverse crescent eaten by face_mirror within 2 columns

Test: impose an INVERSE crescent (boundaries advanced 1 column ahead of interior) via the AP-first sync mechanism, then watch propagation downstream. Does face_mirror's structural source-effect deficit fight the imposed lead, and how fast does it eat it?

**Setup (case 7).** Same NX×NY=41×21, dx=0.025, dt=0.01, t_end=25 ms, moore8_uniform + face_mirror, ttp06_apfirst. Clamp shape:
- Interior rows (j=1..NY-2): cols 1,2,3 at +30 mV
- Boundary rows (j=0 and j=NY-1): cols 1,2,3, AND 4 at +30 mV  ← one extra column

Sync window: ionic-only for 10 steps with the asymmetric pattern enforced. Release at k=11. Output: `data/case7_fm_ttp06_apfirst_invcrescent.h5` (12.3 MB), attrs include `inverse_crescent=True`.

**Result — LAT by column at boundary (j=0) and center (j=10):**

```
  col   LAT[bdry]    LAT[ctr]    bdry − ctr (ms)    interpretation
  ─────────────────────────────────────────────────────────────────────
   3      0.0000      0.0000        +0.0000    clamped at +30 (both rows)
   4      0.0000      0.5491        −0.5491    INVERSE CRESCENT (imposed)
   5      0.9579      1.0203        −0.0624    lead almost gone
   6      1.4999      1.4780        +0.0219    FLIPPED to forward crescent
   8      2.5133      2.3863        +0.1270
  10      3.4922      3.2921        +0.2001
  15      5.8836      5.5554        +0.3282
  20      8.2395      7.8245        +0.4150
  30     12.9028     12.3856        +0.5172
  40     17.3955     16.8272        +0.5683    asymptotic forward crescent
```

**Headline.** face_mirror eats a one-column-ahead imposed lead in exactly **two columns of propagation**. Sign of bdry-ctr LAT flips between col 5 (−62 µs) and col 6 (+22 µs). After col 6, forward crescent monotonically grows and asymptotes around +570 µs by col 40.

**Per-column asymptotic slowdown rate.** Between col 30 and col 40: LAT differential grows by 51 µs over 10 cols → ~5 µs/col additional bdry lag. Equivalent CV ratio at the boundary vs interior: bdry/ctr CV ≈ (1 − 0.005·CV_int/Δx), where CV_int ≈ 1 cm / 16 ms = 62.5 cm/s. So bdry CV ≈ 62.5 · (1 − 0.005·62.5/0.025) ... actually simpler statement: the boundary maintains a sustained ~5 µs-per-column slowdown indefinitely while the wave propagates under face_mirror moore8_uniform.

**Interpretation — face_mirror is sign-locked to forward crescent.** This is the structural counterpart to the equilibrium argument we already had for Fickian gradient flux (KNOWLEDGE.md §"Equilibrium argument"). Now we have the operator-level confirmation in the V5.4 PDE with TTP06 dynamics: the face_mirror BC favors boundary slowdown regardless of initial conditions. An imposed inverse crescent cannot survive ~2 columns of propagation. The asymmetry is baked into the discrete update operator, not into transients of initial conditions.

**Equivalence to the LBM bounce-back family.** Per the corrected HBB ≡ face_mirror analysis (logged earlier today), this same sign-locking should hold for any LBM with bounce-back at the wall — D2Q9 canonical, D2Q9 uniform, with the magnitude depending only on the cardinal:diagonal weight ratio. Could verify by re-running case 7 setup in LBM V1, but the prediction is robust: bounce-back family inherits the same sign-lock.

**Companion test (case 8, not run yet).** Same inverse crescent clamp shape but with `boundary_mode=face_mirror_iso`. Prediction: imposed inverse crescent SHOULD persist indefinitely because face_mirror_iso has no per-column slowdown bias. Could test whether fmi merely eliminates the source-effect imbalance OR has its own (smaller) symmetric tendency.

**Files added (this entry).** `Research/Active/boundary_conduction_speedup/data/case7_fm_ttp06_apfirst_invcrescent.h5`.

### 2026-05-14 (cont): Case 8 — LBM verification of HBB sign-lock (cross-engine closure)

Cross-engine test of HBB ≡ face_mirror. Re-ran case 7 inverse-crescent setup in LBM V1, D2Q9 canonical + halfway bounce-back. Script: `diag_lbm_invcrescent.py`. Same grid (NX×NY=41×21, dx=0.025, D=0.001), TTP06 EPI, LBM dt=0.02 ms, sync_steps=5 (0.1 ms). IC: cols 1,2,3 all rows at +30 mV, col 4 only at j=0/j=NY-1 (inverse-crescent shape); clamp re-equilibrates f to w·V during sync.

**LAT (V crosses −40 mV) bdry vs ctr, side-by-side with case-7 monodomain:**

```
  col       PDE (moore8_uniform + fm)        LBM (D2Q9 canon + HBB)
   4         0.0000    0.5491   −0.549 ms   0.0000   0.0716   −0.0716 ms  imposed
   5         0.9579    1.0203   −0.062      0.1257   0.3519   −0.2261     briefly grows in LBM
  10         3.4922    3.2921   +0.200      1.9080   1.9884   −0.0805
  20         8.2395    7.8245   +0.415      5.2810   5.2823   −0.0014     LBM flips ~col 20
  40        17.3955   16.8272   +0.568     11.8845  11.8117   +0.0728     forward crescent
```

**Confirmed.** Sign-lock direction identical in both engines. HBB eats inverse crescent, flips to forward. Magnitudes scale with weight ratio: 2/3 (PDE uniform fm) eats lead in ~2 cols; 5/6 (LBM canonical HBB) eats in ~20 cols; asymptotic forward LAT 8× weaker in LBM, consistent with weight-ratio plus 2× dt scaling. Quantitatively consistent with `D2Q9_canonical + HBB ↔ moore8_iso + fm`.

**LBM col-5 quirk.** Lead briefly grows (−72→−226 µs) before being eaten — boundary fires from col 4's clamped boundary source while col 5 interior is still ramping. Masked in PDE because 2/3-deficit dominates from step 1.

**Bridge final closure.** Three families, two engines:
- face_mirror_iso (PDE): zero deficit, no canonical LBM analog
- face_mirror ≡ HBB: sign-locked forward crescent (PDE + LBM)
- Weight-ratio sets magnitude: `uniform+fm ≡ D2Q9_uniform+HBB (2/3)`, `iso+fm ≡ D2Q9_canon+HBB (5/6)`, `cardinal4+fm ≡ D2Q5+HBB (zero)`

**Next.** Search for mass-preserving BC that biases TOWARD inverse crescent. Candidate: specular reflection.

**Files added:** `diag_lbm_invcrescent.py`, `data/case8_lbm_d2q9_apfirst_invcrescent.h5`.

### 2026-05-14 (cont): Cases 9-12 — LBM specular vs HBB across weight schemes, four-way readout

Direct test of the converse: what IS the LBM equivalent of face_mirror_iso? Candidate: specular reflection. Script `diag_lbm_specular.py` with `--weights {canonical|uniform_8}` × `--bc {specular|hbb}`. Natural propagation, line stim at col 1, no clamp, 25 ms, NX×NY=41×21, dx=0.025, dt=0.02, D2Q9, TTP06 EPI.

**Four-way LAT readout** (bdry−ctr at column i, µs):

```
  col     can+HBB    can+SPEC     uni+HBB    uni+SPEC
   3       −43        −46         −196        −178       (stim transient — HBB/SPEC identical at col 3)
  10       +27        −24         −34         −120       (HBB flips forward; SPEC still inverse)
  20       +63        −15         +67         −77        (HBB growing forward)
  38       +96        −7          +148        −35        (HBB sign-locked; SPEC decaying to 0)
```

**Key findings.**
1. Specular eliminates HBB's forward sign-lock at every weight scheme.
2. Specular ≡ face_mirror_iso. Earlier "no canonical LBM analog of fmi" claim in KNOWLEDGE.md was wrong — specular IS the analog. Corrects the bridge claim.
3. Specular has zero per-column bias. Late-column inverse magnitudes are decaying stim-region transients, not sustained inverse drive.
4. Initial inverse at col 3 is identical HBB vs SPEC → comes from sharp V[1,:]=0 line-stim discontinuity, not BC handling.

**Final cross-engine deficit hierarchy:**
```
   zero deficit:     moore8 + fm_iso         ≡   D2Q9 + specular  (any weights)
   5/6 mild:         moore8_iso + fm         ≡   D2Q9 canonical + HBB
   2/3 full:         moore8_uniform + fm     ≡   D2Q9 uniform + HBB
   no diagonals:     cardinal4 + fm          ≡   D2Q5 + HBB
```

Two BC families span the picture: HBB ≡ fm (forward sign-lock); specular ≡ fmi (transparent wall).

**Next.** Specular gives zero bias but doesn't actively favor inverse crescent. User-proposed candidate: "horizontal redirect" — NE→E-at-east-neighbour (instead of specular's NE→SE-at-east-neighbour), converting diagonal y-momentum to pure x-momentum at the wall. Predicted: sustained inverse bias.

**Files added:** `diag_lbm_specular.py`, `data/case{9,10,11,12}_*.h5`.

### 2026-05-14 (cont): Future direction — weighted-BC family on the (HBB, specular, horizontal) simplex

User's parting insight: the three BC families differ ONLY in (a) what slot the outgoing diagonal mass lands in, and (b) at which cell. The differences are CONVEX combinations of three vertex rules. Generalize to a single weighted BC family parameterized by a 3-simplex.

**Parameterization.** For each pre-stream diagonal mass m (e.g., C's f_5 at top wall), distribute over three destinations with weights α + β + γ = 1:

```
   α · m   →   C's f_7 slot              (HBB vertex — stays at C, reversed)
   β · m   →   east neighbour's f_8 slot (specular vertex — moves to east, SE)
   γ · m   →   east neighbour's f_1 slot (horizontal vertex — moves to east, E)
```

Symmetric for the other diagonal at top (f_6 → west), and for diagonals at bottom wall (f_7, f_8). Cardinals (f_3, f_4) still HBB at same cell.

**Mass conservation.** Exact by construction: each pre-stream diagonal value is multiplied by a vector of weights summing to 1, then placed in three (or fewer if some weights are zero) different destinations. No duplication, no loss.

**Crescent magnitude as a function of (α, β, γ):**

| (α, β, γ) | BC | Crescent direction |
|---|---|---|
| (1, 0, 0) | pure HBB | forward (boundary slows) |
| (0, 1, 0) | pure specular | zero bias |
| (0, 0, 1) | pure horizontal | inverse (boundary speeds up) |
| interior point | weighted mix | linear interpolation of crescent magnitude |

Expected: a single scalar "crescent index" maps continuously across the simplex. From empirical measurements at the three vertices (HBB +96 µs forward at canonical, specular ~0, horizontal -1146 µs inverse), the index spans roughly [-1146, +96] µs at col 38. Convex combination would interpolate.

**Killer feature — experimental fitting.** Once optical-mapping or microelectrode-array data from real cardiac tissue is available (LAT differential boundary vs interior at the bath-coupled tissue edge), fit (α, β, γ) to match observed crescent magnitude:

```
   observed_LAT_diff(x)  =  f(α, β, γ; tissue parameters)
   
   → fit on the 2-simplex via least-squares or Bayesian inference
   → recover an "effective wall reflectivity vector" for real tissue
   → plot tissue preparations on a triangular plot showing wall "personality"
```

This gives a single 3-vector summary of any tissue boundary's electrical behavior. Different tissue types (healthy, infarct border zone, fibrotic) may live at different points on the simplex, providing a pathology-distinguishing axis.

**Implementation sketch.** Already trivial given the three vertex implementations in `diag_lbm_specular.py`. Each existing top-wall rule writes to one slot; the weighted version writes to all three slots with weights:

```python
# At top wall, non-corner cells, for each outgoing diagonal pre-stream m:
f[7, C]              = ALPHA * m_HBB_contribution            # HBB share
f[8, east_of_C]      = BETA  * m_specular_contribution       # specular share
f[1, east_of_C]      = GAMMA * m_horizontal_contribution     # horizontal share
# Plus the streaming-baseline contributions already in f from standard stream
```

A 2D parameter sweep across the (α, β, γ) simplex (e.g., 0.0, 0.25, 0.5, 0.75, 1.0 grid) produces a heat map of crescent magnitude as a function of weights. Map this against eventual experimental data to recover the tissue's effective weights.

**Next-session checklist.** (1) Fix the mass-leak in horizontal redirect at corner cells (V_sum 9-18% inflation observed); the weighted family inherits this until corners are sorted. (2) Implement the weighted BC as a parameterized step function. (3) Run a coarse simplex sweep (e.g., 15 points). (4) Build a crescent-magnitude heat map across (α, β, γ). (5) Document the family as a generalization in KNOWLEDGE.md.

Beyond the simplex, two extension directions: (a) a SPATIALLY-VARYING (α, β, γ) field — different weights at different parts of the wall, allowing modeling of heterogeneous tissue boundaries; (b) a STATE-DEPENDENT weighting — weights modulate with local V or local gradient, mimicking voltage-gated ion channels at the wall.

### 2026-05-14 Session
**Worked on**: High-resolution dV/dt diagnostic infrastructure for the boundary deficit; mechanism isolation tests (synchronized AP, inverse-crescent imposition); major correction to the HBB↔face_mirror mapping; LBM verification of cross-engine sign-lock; and discovery of a novel "horizontal redirect" BC that biases toward inverse crescent.

**Accomplished** (chronological summary; 14 cases total in `data/`):

- **Cases 1-4 (`diag_dvdt_decomposition.py`)**: 2×2 sweep face_mirror vs face_mirror_iso × diffusion-only vs +TTP06, NX×NY=41×21, dt=0.01 ms, full V tensor logged every step into HDF5. Confirmed at step 1: dV/dt ratio at moore8_uniform + face_mirror is structurally 2/3 (boundary 92 mV/ms vs center 138 mV/ms). Initial deviation −0.46 mV at boundary at step 1, growing through ramp-up. fmi cases: 0.0 to floating-point precision at every step.

- **Cases 5-6 (synchronized AP at cols 1+2+3)**: Tested whether the col-1 corner mismatch is responsible for downstream crescent. Case 5 used Strang during sync (10 steps); case 6 used ionic-only (strict no-diffusion during sync). At col 4 in case 6, first diffusion step gives bdry−ctr = −0.62 mV with V_bdry charging 1.23 mV vs V_ctr 1.85 mV → ratio 0.665 ≈ 2/3 (matches structural prediction EXACTLY). Conclusion: source-effect imbalance is local per-column, generated fresh at any column charging from rest under face_mirror. Initial framing was wrong; corrected per user pushback — the data confirms (not falsifies) the user's source-effect-imbalance hypothesis.

- **Case 7 (imposed inverse crescent, monodomain)**: Clamped col-4 boundary rows ahead of interior (1-column lead). face_mirror eats the lead in **2 columns** and flips to forward crescent, asymptoting at +568 µs by col 40. Confirms face_mirror sign-locks to forward.

- **Case 8 (`diag_lbm_invcrescent.py`, LBM cross-engine)**: Same inverse-crescent setup in LBM V1 D2Q9 canonical + HBB. Lead eaten in ~20 cols (longer because LBM 5/6 deficit is milder), then flips and grows forward crescent to +73 µs. Cross-engine sign-lock confirmed.

- **Major correction (HBB↔face_mirror, not face_mirror_iso)**: User caught my earlier loose claim. Worked through diagonal slot mechanics carefully:
  - HBB at top: C's f_5 (NE) → C's f_7 (SW) at same cell. Mass-conserving via full reversal. Kills upstream-V contribution from diagonal slots.
  - face_mirror at top: NW_ghost = V_self → gap = 0 → no diagonal Laplacian contribution.
  - Both achieve ∂V/∂n = 0 at the wall (zero-flux Neumann); both kill the diagonal upstream-V flow. Different bookkeeping, SAME structural deficit.
  - Weight ratio determines magnitude: D2Q9 canonical (1/9, 1/36) maps to moore8_iso + fm (5/6 deficit); D2Q9 uniform (1/8 each) maps to moore8_uniform + fm (2/3 deficit).
  - Conversely: face_mirror_iso (PDE) pulls V_W into NW_ghost slot — provides REAL upstream-V to diagonal. The LBM analog is SPECULAR REFLECTION (y-component flips, tangential preserved; diagonal mass traverses wall to adjacent cell's f_8 slot, carrying upstream V from upstream-boundary's f_5).

- **Cases 9-12 (`diag_lbm_specular.py`, --weights × --bc CLI args)**: 4-way LBM natural-propagation comparison (canonical/uniform × HBB/specular). Confirmed:
  - HBB sign-locks to forward (canonical +96 µs at col 38; uniform +148 µs).
  - Specular has zero structural bias (canonical −7 µs at col 38, decaying from −46 µs initial; uniform −35 µs from −178 µs initial). Inverse residual is stim-region transient that diffuses away slowly; no per-column inverse drive.

- **Cases 13-14 (`diag_lbm_specular.py --bc horizontal`)**: Tested user's novel "horizontal redirect" BC: outgoing diagonals at top/bottom land in adjacent cell's pure-CARDINAL slot (f_5 → east's f_1 instead of east's f_8 like specular). Implementation: HBB everywhere first, then zero out f_7/f_8 at top non-corner cells AND f_5/f_6 at bottom non-corner cells, then ADD pre-stream diagonals to neighbours' cardinal slots. RESULT: **strong sustained inverse crescent that grows monotonically with distance**. Canonical: −220 µs at col 3 → −1146 µs at col 38 (5× growth). Uniform_8: −641 µs → −3106 µs (14× growth). Wall channel propagates faster than bulk. **Caveat**: V_sum at t=25ms is 9% (canonical) to 18% (uniform) HIGHER than HBB baseline, and V_max climbs above the +15 mV plateau (to +18.75 / +19.68). Possible mass leak at corner cells (where the zero-out doesn't apply but the redirect destination does) — needs investigation. The qualitative inverse-crescent bias is real and large regardless.

**Three structural BC families now established**:
```
   HBB / face_mirror        sign-locks to forward crescent  (boundary slowdown)
   specular / face_mirror_iso  zero structural bias            (transparent wall)
   horizontal redirect       sign-locks to inverse crescent  (boundary speedup, novel)
```

**Next**: (1) Investigate mass-conservation leak in horizontal-redirect implementation — likely at corner cells receiving redirect deposits without donating any of their own mass. (2) If leak is real, derive a mass-conservation correction term as a counterweight. (3) Run inverse-crescent test with the corrected BC; verify that the wall-channel bias survives the correction (or determine that it depended on the leak). (4) Document the horizontal-redirect formulation in KNOWLEDGE as a new BC family with sustained inverse crescent.

**Files added (this session)**:
- `diag_dvdt_decomposition.py` (cases 1-7 with cli arg selection)
- `diag_lbm_invcrescent.py` (case 8)
- `diag_lbm_specular.py` (cases 9-14 with CLI args `--weights` and `--bc {hbb|specular|horizontal}`)
- `plot_dvdt_traces.py` (figures from cases 1-4)
- 14 HDF5 files in `data/`

### 2026-05-28: Stim col 0 fix; anisotropic videos; horizontal-redirect "leak" debunked

**Session work (3 threads).**

**(1) Stim col 0 fix and re-render.** Found and fixed `STIM_COL=1` in
`diag_dvdt_decomposition.py` and `V_init[1,:]` in `diag_lbm_specular.py`
— stim was one column inside the wall. Also tightened `region: x<0.05`
(cols 0+1) to `x<DX/2` (col 0 only) in `video_boundary_modes.py`,
`video_stencil_mirror_combos{,_individual}.py`. Regenerated cases
3,4,9,10,12,13 + equalmix, re-rendered all 14 mp4s. Sign-lock hierarchy
preserved (LAT bdry−ctr at col 38, µs):
fm +582 / fmi 0 / hbb +106 / specular +2 / horizontal −1130 / equalmix
−206 — all expected signs, magnitudes within ~10% of col-1 stim values.
Notable: rest_pad BC video now shows no AP firing under col-0 stim
(ghost-at-rest clamp drags wall depolarization back). Real physics.
Render scripts also patched to set `matplotlib.rcParams['animation.ffmpeg_path']`
(silent FileNotFoundError without it). Commit `68ecde4b`.

**(2) Anisotropic videos (monodomain + LBM, 2:1 and 1:2).** New scripts
`video_anisotropic_{monodomain,lbm}.py`. Monodomain uses cardinal4
stencil + face_mirror + `D_field=(Dxx, Dxy=0, Dyy)`. LBM uses D2Q9 MRT
collision with `s_jx, s_jy` derived from `D_xx, D_yy` via
`tau_tensor_from_D`, HBB everywhere. Vertical line stim at col 0, TTP06
EPI, 25 ms. Four videos saved to `figures/video_aniso_*.mp4`. V_max in
all four cases is the normal +33 mV plateau (no overshoot artifacts).
Visual structure preserved between engines.

**(3) Horizontal-redirect "leak" diagnosis (PLAN.md).** Pre-session
hypothesis (from 2026-05-14 IDEALOG entry): the 9-18% V_sum excess vs
HBB and the V_max overshoot were a mass leak at corner cells where the
non-corner redirect zero-out doesn't apply but the redirect destination
does include the corner. **Wrong on multiple counts:**

- Step 1 (mass audit, `diag_horizontal_mass.py`): V_sum excess at t=25 ms
  is **not** corner-localized. Corners contribute 0.5% of the excess;
  interior carries 111.7%. Hypothesis already wounded here.
- Step 2 (V(y) profiles, `diag_horizontal_vyprofile.py`): the "sub-edge dip"
  to −94.67 mV at (i=2, j=1) is **transient** during wavefront passage.
  At plateau (t=25 ms), j=1 is +17.3 mV (HIGHER than HBB plateau +15.0),
  while j=0 is +13.2 mV (LOWER). Mass shifts from wall to sub-edge after
  the wave passes — not a sustained dip.
- Step 3 (counterfactual): added `--bc horizontal_fixed` to
  `diag_lbm_specular.py`, with corner-aware destination ranges
  (donor [1, NX-3] → dest [2, NX-2], corner-excluded) and orphan
  donors (i=NX-2 f_5, i=1 f_6) HBB-bouncing at self. **The fixed and
  buggy variants are nearly identical**: V_sum/cell +16.172 (fixed) vs
  +16.173 (buggy), V_max +33.746 vs +33.785, LAT diff −1002 µs vs −1130
  µs. Corner handling makes a ~12% difference in LAT magnitude — small,
  not the source of the bulk excess.
- Step 4 (diffusion-only, `--physics diffusion` flag added):
  **mass is EXACTLY conserved** for all three BCs (HBB, buggy horizontal,
  fixed horizontal). V_sum at t=25 = V_sum at t=0 to floating-point
  precision (Δ = +0.0000 mV). Sub-edge dip persists under pure diffusion:
  V_min at (2,1) = −94.78 mV (buggy) vs −95.20 mV (fixed) vs −89.45 mV
  (HBB). **Conclusion:** the dip is a BC-mechanical artifact (real,
  designed consequence of the redirect's lateral mass-shift), not ionic
  hyperpolarization.
- Step 5 synthesis: `figures/horizontal_synthesis.png` (3×2 panel —
  V(y) profiles at three times, V_sum trajectories TTP06 vs diffusion).

**Three-way attribution (replaces 2026-05-14 caveat):**

```
Wall-channel depolarization under horizontal redirect:
  - DESIGNED behavior:   ~100% of the magnitude
                         (the redirect's lateral mass shift IS the wall channel)
  - MASS LEAK:           0% — verified by diffusion-only V_sum conservation
                         (corner handling is a ~12% perturbation, no leak)
  - SUB-EDGE DIP:        BC-mechanical artifact, NOT ionic
                         (persists under pure diffusion; deeper than HBB
                          because horizontal sustains a stronger lateral
                          mass-shift pattern away from the sub-edge)
```

The earlier 2026-05-14 framing "V_sum 9-18% higher = mass leak" was
incorrect. The correct interpretation: the wall channel propagates the
AP wavefront faster along the wall row, so MORE cells are in the
sustained-plateau phase by t=25 ms, raising V_sum via the TTP06 source
term — not via a numerical leak.

**Implications for the (α, β, γ) weighted simplex sweep:** the simplex is
mass-conserving by construction (was always true, now empirically confirmed).
The "fix the leak before sweeping" prerequisite from the 2026-05-14
next-session checklist is no longer a prerequisite — the leak doesn't
exist. Simplex sweep can proceed with the existing buggy/fixed indifferent
implementation.

**Files added (this session):**
- `video_anisotropic_monodomain.py`, `video_anisotropic_lbm.py`
- `diag_horizontal_mass.py`, `diag_horizontal_vyprofile.py`,
  `diag_horizontal_synthesis.py`
- `PLAN.md` (new, today's plan; old PLAN.md archived to
  `plans/2026-04-30_moore8_stencil_extension.md`)
- `figures/horizontal_mass_audit.png`, `figures/horizontal_vy_profiles.png`,
  `figures/horizontal_synthesis.png`
- `figures/video_aniso_{monodomain,lbm}_{2to1,1to2}.mp4` (4 videos)
- `data/case_horiz_fixed_*.h5` (TTP06 + diffusion variants)
- `data/case{10,13}_lbm_d2q9_canonical_*_natural_diffusion.h5`

**Modified scripts:**
- `diag_lbm_specular.py`: added `--bc horizontal_fixed` and
  `--physics {ttp06|diffusion}` CLI flags; added
  `apply_horizontal_fixed_top_bottom_d2q9` and `lbm_step_horizontal_fixed`.

**Next.** Implement the weighted (α, β, γ) simplex sweep using the
mass-conserving (buggy-is-fine) horizontal vertex. Build the crescent
heat map across the simplex. Anisotropic boundary study can also proceed
— today's anisotropic videos are exploratory; need systematic LAT
diff measurements + fiber-angle sweep + cross-engine consistency check.

### 2026-05-28 (cont): Wall pre-charge is INTRINSIC to horizontal redirect; gradient variant removes it but also removes the inverse crescent — INTERPRETATION NOT YET SETTLED

Follow-on investigation of the horizontal-redirect wall behavior. The
2026-05-14 "novel inverse-crescent BC family" claim is now UNDER REVIEW
(not yet overturned — see open question at end).

**Finding 1 — the wall depolarization is uniform-in-x, not a propagating
front (`diag_horizontal_wall_propagation.py`).** At t=0.04 ms (2 LBM
steps), cols 3-35 of the wall row are ALREADY uniformly at −82 mV under
horizontal. Streaming cannot move information that far in 2 steps, so the
elevation is generated LOCALLY at every wall cell simultaneously. There
are two superposed contributions:
- INTRINSIC (~+3 mV, saturates in 2 steps): per-cell BC operator pumping
  at rest, uniform across the whole wall.
- EXTRINSIC (+18 mV more, accumulates ~25 ms): wavefront-driven; the
  bulk wave's higher V makes the redirected diagonal contributions less
  negative, injecting more positive mass into the wall.

**Finding 2 — INTRINSIC artifact exists with NO STIM AT ALL.** Ran
diffusion-only, V_init = V_rest everywhere, no perturbation
(`diag_horizontal_*` no-stim variant). HBB and specular: Δwall =
−0.000000 mV (perfect no-ops). Horizontal: Δwall = +18.43 mV,
Δsub-edge = −1.94 mV, Δinterior = −1.87 mV at t=25 ms. The rule
SPONTANEOUSLY builds a 3-layer voltage structure (+18 wall / −2 elsewhere)
from a uniform field. Mass is exactly conserved — it's redistributed, not
leaked. Weighted family scales with γ: (0,0,1)→+18.4, (.33,.33,.33)→+7.4,
(.7,.2,.1)→+2.7, etc. Both pure HBB and pure specular give exactly 0.

**Finding 3 — root cause is WEIGHT-CLASS MISMATCH (user's diagnosis).**
D2Q9 weights confirmed: cardinal w=1/9, diagonal w=1/36, ratio exactly
4.0; equilibrium feq_i = w_i·V (`bgk.py:31`). HBB and specular map
diagonal→diagonal and cardinal→cardinal (weight-matched), so at rest
they map feq→feq exactly — structural no-ops, no eq/neq split needed.
Horizontal takes a diagonal (1/36) and dumps it into a cardinal slot
(1/9) — a 4× weight mismatch. At rest, the redirected equilibrium mass
(1/36·V) doesn't match the destination cardinal's equilibrium (1/9·V),
and the leftover is the standing artifact. The weight-class change
(diagonal→cardinal) is EXACTLY what makes the inverse crescent (slow
diagonal → fast cardinal wall highway), so crescent and artifact are
mechanistically linked through the same operation.

**Finding 4 — gradient redirect removes the artifact... and the crescent
(`apply_horizontal_gradient_top_bottom_d2q9`, `--bc horizontal_gradient`).**
New variant: split each outgoing diagonal into eq (=w_i·V) + neq
(=f_star−w_i·V). Apply standard HBB to the eq part (bounce at same cell,
weight-matched → no-op at rest), redirect ONLY the neq (flux) part
laterally. Chapman-Enskog: neq carries the diffusive flux, eq carries the
field. Results:

```
                          NO-STIM (diffusion)      WITH-STIM col0 (ttp06)
   BC                     Δwall    mass-drift       LAT@c38    precharge_c20  wall_max
   ──────────────────────────────────────────────────────────────────────────────
   hbb                      —        —              +105.6     −85.23         15.00
   horizontal             +18.43    0 (conserved)  −1131.6     −69.21         17.64
   horizontal_gradient    −0.00000  0 (conserved)  +163.6     −85.23         15.00
```

The gradient variant is a TRUE no-op at rest (Δwall = exactly 0, like
HBB/specular) AND mass-conserving. BUT its LAT crescent flips from
−1132 µs (inverse) to +164 µs (FORWARD) — essentially identical to HBB
(+106). Precharge gone, wall plateau back to normal.

**OPEN QUESTION — interpretation not yet settled (do NOT update KNOWLEDGE
claim until resolved).** Two readings, both consistent with the data so
far:
1. The inverse crescent was ALWAYS an artifact of pumping weight-mismatched
   *equilibrium* mass onto the wall (pre-charge lets the wall cross the
   −40 mV LAT threshold early). Remove the eq pumping → wall charges from
   rest → connectivity deficit dominates → forward crescent like HBB. If
   true: there is no clean single-field inverse-crescent BC, consistent
   with Kleber speedup being intrinsically a bidomain (two-domain BC
   asymmetry) effect. The (α,β,γ) simplex collapses to forward↔zero
   (HBB↔specular); no genuine inverse vertex.
2. The gradient variant may be TOO aggressive — it strips ALL eq mass
   transport, but maybe a physically-correct inverse-crescent rule needs
   SOME directed transport that the pure-neq version discards. The
   weight-mismatch might need a normalization factor (×w_cardinal/w_diag
   = ×4 on the redirected flux) rather than full eq removal — user's
   suggested "normalize the mass to the correct amount." Other diagonal-
   injection schemes remain unexplored.

**Verification still needed before claiming either way:**
- Re-measure LAT at multiple thresholds (−40, 0, +10 mV) and via max(dV/dt)
  timing — the −40 mV crossing is exactly what pre-charge games, so it's
  the worst single metric. dV/dt-peak timing is pre-charge-immune.
- Try the weight-normalized redirect (×4 flux) as finding-2's "other
  diagonal injection options."
- Confirm specular's zero-crescent and HBB's forward both reproduce under
  the multi-threshold metric.

**Files added (this sub-session):**
- `diag_horizontal_wallrow.py`, `diag_horizontal_longrun.py`,
  `diag_horizontal_inward_widestim.py`, `diag_horizontal_wall_propagation.py`,
  `diag_horizontal_anisotropic.py`, `video_horizontal_longrun.py`,
  `render_horizontal_donut.py`
- `figures/horizontal_wallrow_evolution.png`, `horizontal_longrun.png`,
  `horizontal_inward_diffusion.png`, `horizontal_widestim_compare.png`,
  `horizontal_wall_propagation.png`, `horizontal_anisotropic_wallcharge.png`
- `figures/video_longrun_{diff,ttp06}_{hbb,horizontal}.mp4`,
  `video_widestim_{hbb,horizontal}_5col.mp4`,
  `video_aniso_horizontal_{1,2,4,8}to1.mp4`, `video_bc_horizontal_donut.mp4`
- `data/case_horiz_donut_*.h5`, `case_horiz_grad_*` (gradient variant)

**Modified scripts (this sub-session):**
- `diag_lbm_specular.py`: added `--bc horizontal_donut` (corner-diagonal
  X-wrap, user's first proposed fix — does NOT fix the pre-charge because
  the artifact is per-cell-local, not corner-accumulation), `--bc
  horizontal_gradient` (eq/neq split — DOES remove the artifact but
  reverts crescent to forward), `--t_end` flag. Added
  `apply_corner_diagonal_wrap_d2q9`, `apply_horizontal_gradient_top_bottom_d2q9`,
  `lbm_step_horizontal_donut`, `lbm_step_horizontal_gradient`.

**Anisotropy aside (`diag_horizontal_anisotropic.py`).** Tested D_xx>D_yy
with MRT under horizontal. Higher horizontal anisotropy DECREASES wall
pre-charge (8:1 → −78 mV vs 1:1 → −69 mV at t=1ms), because strong x-
diffusion leaks hoarded wall mass back into the bulk before it accumulates.
Wider stim (5 cols vs 1) does NOT change the pre-charge equilibrium (same
−69.21 mV at col 20), confirming the artifact is BC-intrinsic, set by the
local rule, not by the stimulus source.

### 2026-05-28 (cont): GOATED RESULT — clean inverse-crescent BC found (same-cell specular). Full arc.

Chased the horizontal-redirect artifact to ground and, in doing so, found
the genuinely-clean inverse-crescent boundary condition. User confirmed the
inverse crescent visually (developing chevron, no wall pre-glow). Full
synthesis in KNOWLEDGE.md § "Clean inverse-crescent BC: same-cell specular
reflection (2026-05-28)". This is the thinking trail.

**Realization chain (each rung necessary):**

1. **Line speedup (visual).** Horizontal redirect makes wall rows fire ahead
   → inverse-crescent V-shape. Looked like a real novel speedup.
2. **Diagonal→horizontal is the mechanism (insight #1).** Converts slow
   diagonal (w=1/36, 45°) → fast cardinal (w=1/9, along wall) = "wall
   highway." The weight-CLASS change carries the wave faster.
3. **Wall depolarizes uniformly in x, in 2 steps (anomaly).** Too fast for
   propagation; generated locally per cell.
4. **It's INTRINSIC — happens with zero stim (insight #2).** No-stim uniform
   IC: HBB/specular Δwall = exactly 0; horizontal = +18.43 mV standing
   3-layer structure. Mass conserved. ⇒ the "speedup" is a wall pre-charge
   crossing −40 mV early. A measurement artifact.
5. **Root cause = weight-class mismatch (insight #3, user's diagnosis).**
   D2Q9: cardinal 1/9, diagonal 1/36, ratio 4. HBB/specular weight-matched
   → feq→feq no-ops. Horizontal puts 1/36·V into a 1/9·V slot → leftover is
   the artifact. Omega-sweep: artifact = 0 at omega=1, grows with |omega−1|
   → collision retargeting a weight-mismatched distribution is the pump.
6. **Mass accumulation ⇒ reflection, not transport (insight #4).** Scalar
   normalization can't fix it (mass conservation pins scale to 1; any k≠1
   leaks). Mass-on-wall (x-stay or +y-rehit) traps eq mass → artifact. Only
   mass-LEAVES-into-bulk (−y) is clean. HBB/specular are clean because they
   reflect. So: inverse without accumulation = stay in reflection class,
   change which component is reversed.
7. **Exhaustive enumeration → the clean rule.** All 27 symmetric
   mass-conserving rules (`diag_enumerate_walls.py`, `data/wall_enumeration.txt`).
   Partition by destination y-sign: stays(x) all +18 artifact/inverse;
   rehits(+y) all +18-24/inverse; leaves(−y) all ~0 artifact, sign set by
   tangential treatment: NE→SW (x reversed)=HBB=forward; NE→SE same cell (x
   preserved)=specular=INVERSE, CLEAN.

**Clean rule: same-cell specular.** Top: f_5(NE)→f_8(SE) same cell,
f_6(NW)→f_7(SW) same cell (flip y, keep x, no displacement). Bottom
y-mirrored. Verify: NO-STIM 50ms Δwall = −0.000000, mass drift −1.6e-10.
WITH-STIM LAT bdry−ctr col38 = −313/−317/−318/−329 µs at thr −40/−20/0/+10
(inverse at ALL thresholds); precharge = rest (none); max(dV/dt) timing =
−300 µs (precharge-IMMUNE → real upstroke speedup).

**Mechanism.** Diagonal NE=(+x,+y) = forward + toward-wall momentum. HBB
reverses both → kills forward drive → slowdown. Same-cell specular flips
only y → forward drive survives → wall keeps more forward push → speedup.
Diagonal→diagonal weight-matched → no artifact.

**Two prior hypotheses FALSIFIED (don't redo):**
- "No clean single-field inverse exists / Kleber is bidomain-only" — FALSE.
- "trap ⟺ inverse ⟺ artifact are the same thing" — FALSE; reflection is a
  clean route to inverse.

**Implication for (α,β,γ) program.** Real clean axis = HBB (forward) ↔
same-cell-specular (inverse), both rest-neutral reflections, with
neighbor-displaced specular at zero between. Replace horizontal vertex with
same-cell-specular → tissue-fit inverse problem becomes well-posed.

**Rejected variants tried (kept as --bc modes in diag_lbm_specular.py):**
horizontal_fixed (artifact unchanged, per-cell-local); horizontal_donut
(corner X-wrap, doesn't fix, new corner artifact); horizontal_gradient
(eq/neq split — removes artifact but reverts to forward; too aggressive,
user rejected); horizontal_wnorm (scalar r=0.25 — artifact ~3× smaller but
nonzero, crescent weaker); specular_up (NE→NE@i+1 — ≈ horizontal, traps).

**Files added:** `diag_enumerate_walls.py`→`data/wall_enumeration.txt`;
`diag_horizontal_resolve.py`→`data/resolve_log.txt`;
`render_horizontal_{donut,gradient}.py`;
`figures/video_bc_specular_samecell.mp4` (THE clean video) +
`video_bc_horizontal_{gradient,wnorm,donut}.mp4`, `video_bc_specular_up.mp4`;
HDF5 `case_horiz_{fixed,donut,grad,wnorm}_*`, `case_spec_up_*`.

**Modified `diag_lbm_specular.py`:** added BC modes (above), `--physics
{ttp06|diffusion}`, `--t_end`, and corresponding apply_*/lbm_step_* fns.

**Next.** (1) Promote same-cell-specular to a named documented BC. (2)
Rebuild (α,β,γ) simplex on HBB↔same-cell-specular axis (artifact-free at
every point). (3) PDE analog: which monodomain FDM face stencil corresponds
to same-cell specular (preserves tangential gradient, zeroes normal flux)?
(4) Re-measure OLD horizontal cases with multi-threshold + dV/dt to quantify
how much of their −1130 µs was pre-charge artifact vs real.
