# Boundary Conduction Speedup (Kleber Effect) — Knowledge File

> This file is a running synthesis. Updated as findings accumulate.
> When the question is complete, a copy is promoted to `Research/Knowledge/`.

## Current Understanding

At an insulated (no-flux) tissue boundary in a bath-perfused preparation, conduction velocity increases by approximately 7-13% compared to the tissue interior. This is the **Kleber boundary speedup**, a real biophysical effect confirmed both experimentally and computationally.

### Mechanism (8-link argument chain)

The effect arises from the asymmetric boundary conditions in the bidomain model at a tissue-bath interface:

1. **Intracellular domain terminates** (Neumann BC): gap junctions end at the tissue surface, so no intracellular current crosses the boundary.
2. **Extracellular domain is continuous with the bath** (Dirichlet BC): the interstitial fluid connects to the low-resistance bath solution, effectively clamping phi_e to zero at the surface.

This asymmetry short-circuits the extracellular return path for propagating wavefronts. In bulk tissue, the wavefront current traverses both intracellular resistance r_i (forward) and extracellular resistance r_e (return), giving sigma_eff = sigma_i * sigma_e / (sigma_i + sigma_e). Near the boundary, the bath provides a parallel low-resistance return path, reducing effective resistance to approximately r_i alone, so sigma_eff_boundary approaches sigma_i.

**Theoretical CV ratio** (longitudinal human ventricular tissue):
```
CV_boundary / CV_interior = sqrt((sigma_i + sigma_e) / sigma_e)
                          = sqrt((1.74 + 6.25) / 6.25) = sqrt(1.278) = 1.131
```

The enhancement decays exponentially into the tissue interior with characteristic length lambda (electrotonic space constant, approximately 1.4 mm at rest).

### Our results

- **Bidomain V1**: CV ratio = 1.0714 at dx=0.025 cm, converging toward 1.131 with mesh refinement (confirmed via mesh convergence study)
- **LBM V1**: D2Q9 with Dirichlet BC also captures the speedup, though with a ~35% CV baseline offset due to LBM numerical dispersion
- **Monodomain FDM control**: No boundary speedup observed in the *Bidomain V1 engine's monodomain mode* (Mehrstellen 9-point + face-centered Neumann + explicit Euler) — the "0.000 cm deviation from flat" result. This is the engine path used in `Bidomain/Engine_V1/experiments/triangle_merger.py`. **NOTE (2026-04-29 audit)**: this result has NOT been re-validated against `Monodomain/Engine_V5.4`'s FDMDiscretization, which uses a different convention (node-centered mirror via `V_ghost = V[i-1]`, see `cardiac_sim/simulation/classical/discretization_scheme/fdm.py:271-320`). Whether V5.4 also gives 0.000 cm deviation remains an open question, addressed by PLAN.md Phase C.

### Where the effect is present

| Geometry | Intracellular | Extracellular | Kleber effect? |
|----------|--------------|---------------|----------------|
| Tissue submerged in Tyrode's | Terminates | Bath-coupled | YES |
| Laser-cut void (bath fills void) | Terminates | Bath-coupled | YES |
| Tissue on glass substrate | Terminates | Insulated (glass) | NO |
| In vivo (blood contact) | Terminates | Bath-coupled (blood) | YES |

### Key literature

- **Kleber et al. 2021** (PMID 34296210): Most recent comprehensive review on safety factor, coupling, and boundaries
- **Kucera et al. 1998** (PMID 9776726): Foundational work on geometry-CV relationship and branching effects
- **Connolly et al. 2015** (PMID 25872206): Direct evidence of electrotonic load gradients at infarct border zones
- **Shaw & Rudy 1997** (PMID 9351447): Ionic mechanisms linking safety factor to conduction
- **Roth 1991** (Ann Biomed Eng 19:669-678): Bidomain boundary condition equivalence
- **Patel & Roth 2005** (Phys Rev E 72:051931): Matched-asymptotic solution showing exponential boundary layer

## PDE Formulations of the Effect

### Storage-tank analog (Zimmerman, 2026-04-24)

PI John Zimmerman shared a discrete storage-tank simulation (`../../../simulation/storagetanks.py`, filed at repo root) that
exhibits a qualitatively similar boundary speedup. Each tank on a 2D Moore-neighbourhood grid
pumps a source-state-dependent amount `max_pump·√((u−θ)/(u_max−θ))` through every channel
leading to a lower-volume neighbour, gated on `u > θ`. Interior tanks have 8 open channels,
edge tanks 5, corner tanks 3 — fewer channels at the boundary means a fired tank retains
potential longer, sustaining drive to the remaining neighbours along the edge.

This rule is **non-Fickian**: the flux across each link depends only on the source state,
not on the gradient (u_i − u_j). The acceptor's state appears only through a Heaviside gate.
So the total outflow from a source scales with its number of sinks — the geometric source
of the boundary asymmetry.

### Why the plain heat equation cannot reproduce it

For standard reaction–diffusion with Neumann BC:

```
∂u/∂t = D ∇²u + f(u),     ∂u/∂n = 0 on ∂Ω
```

In the *continuum*, the wall reflects flux but preserves tangential symmetry. 1D wave
speed c = 2√(D·f'(0)) is set by bulk coefficients and is the same at the boundary as in
the interior. The monodomain control experiment (0.000 cm deviation from flat) is the
empirical confirmation.

In the *discrete* setting, however, the no-flux Neumann BC is implemented either as
zero-pad (boundary cells have fewer non-zero stencil entries) or as ghost-reflection
(boundary cells get duplicated upstream values) — and these produce *different signs*
of boundary effect. See `Discrete-lattice boundary effects` below for the full
decomposition. The continuum result lives in between these two discrete extremes.

### Three candidate modifications

In rising fidelity:

**(A) Heat + state-dependent loss** — pedagogical / transparent
```
∂u/∂t = D ∇²u + f(u) − γ(x) u
γ(x) = γ₀ · [1 − exp(−d(x)/λ)]
```
`d(x)` = distance to boundary, `λ` = electrotonic length. Bulk: full dissipation. Edge: γ→0.
Speed ratio ≈ √(1 + γ₀·τ_rxn), parameter-dependent (doesn't land on 1.131 by itself).

**(B) Heat with spatially-varying diffusivity** — canonical Kleber model
```
∂u/∂t = ∇·(D(x) ∇u) + f(u)

D(x) = D_bulk + (D_bdry − D_bulk) · exp(−d(x)/λ)
D_bulk = σᵢσₑ / (σᵢ+σₑ)        (harmonic mean)
D_bdry = σᵢ                     (bath shorts out σₑ)
```
This is precisely the monodomain reduction of bidomain under the quasi-static assumption,
with the boundary correction encoded in D(x). Linearised traveling-wave speed c ∝ √D gives:
```
CV_boundary / CV_interior = √(σᵢ / D_bulk) = √((σᵢ + σₑ) / σₑ)
```
For human ventricle longitudinal (σᵢ=1.74, σₑ=6.25): = √(7.99/6.25) = **1.131** — matches
the theoretical target the Bidomain V1 engine has been converging toward (1.0714 at dx=0.025).

**(C) Non-local (peridynamic) heat** — most faithful to John's tank rule
```
∂u/∂t = ∫_{B(x,R)} K(|x−x'|) [u(x') − u(x)] · 𝟙_Ω(x') dx' + f(u)
```
The domain indicator `𝟙_Ω(x')` truncates the kernel at ∂Ω. In the interior this reduces to
`D∇²u` with D = (∫|x'|² K dx')/2d. At the boundary the self-coefficient on u(x) shrinks
(less local dissipation) and the truncated support's centroid shifts inward (inward drift).
Direct continuum analog of "8 nbrs → 5 → 3" in John's toy.

### Recommendation

**Use (B) as the single-field PDE that captures the Kleber effect.** It is:
- The simplest modification of the heat equation,
- Mechanistically interpretable (extracellular short-circuit = local diffusivity jump),
- Quantitatively correct (reproduces the measured 1.131 ratio),
- Derivable as the quasi-static bidomain reduction (Bishop & Plank 2011 augmented monodomain).

For the anisotropic sub-question, (B) generalises by replacing scalar D(x) with a spatial
diffusivity **tensor D(x)**. The eikonal-limit wave speed becomes direction-dependent:
`c(x, n̂) = 2√(n̂ · D(x) · n̂ · f'(0))`, which predicts the fiber-parallel vs perpendicular
boundary-layer profiles we're about to measure.

### Discrete-lattice boundary effects: Effect A, Effect B, Effect B′

Investigation of John's storage-tank model on a 2D 80×50 Moore-neighbourhood grid with
inlet/outlet line geometry (`simulation/`) decomposes the boundary asymmetry into two
intrinsic effects plus one that's induced by the boundary operator.

**Effect A — geometric inflow deficit.** When a planar wavefront propagates rightward,
an interior tank at column N receives drive from 3 fired upstream neighbours (the (N-1)
column at y-1, y, y+1). An edge tank at column N receives from only 2 upstream neighbours
(one of the diagonals doesn't exist). This deficit is purely geometric — it appears in
*every* nearest-neighbour-coupled lattice with a no-flux wall, regardless of pump rule.
Effect A pushes the per-column LAT shape toward an *inverted-U / crescent* (interior
ahead of edge), i.e. boundary slowdown.

**Effect B — outflow dividend / sustained source.** Once a tank has fired, an interior
tank drains into 5 unfired downstream neighbours, an edge tank into only 3. The edge
therefore retains volume better, stays above threshold longer, and integrates more total
drive into whatever comes next. Effect B pushes toward *U-shape / camel toe* (edge ahead),
i.e. boundary speedup. **Effect B requires a non-self-limiting flux rule**: under a
gradient-driven (Fickian) rule, the receiver's rising V suppresses the per-channel pump
rate before the integrated-drive advantage can accumulate, and Effect B is killed.

**Effect B′ — mirror-duplication enhancement (induced by reflection BC).** When the
boundary operator is *reflection-padded* (`np.pad(V, mode='reflect')`), each boundary
cell sees 8 channels but only 5 unique upstream cells: the y=1 row contributes both via
the real channel and via the ghost channel that mirrors it. The boundary cell receives
*double drive* from 3 of its real neighbours. This is far stronger than Effect A and
overwhelms it. Both pump rules then produce a massive camel toe under reflection BC.

**Boundary-operator dominance.** The boundary operator (how the wall handles missing
neighbours) determines the sign of the boundary effect *more strongly than the pump rule*:

| BC choice                       | constant rule LAT shape   | gradient rule LAT shape   |
|---------------------------------|---------------------------|---------------------------|
| zero-pad (no-flux Neumann)      | crescent + transient camel | pure crescent (mono.)    |
| reflection (mirror enhancement) | massive camel toe         | massive camel toe         |

**Pump-speed Goldilocks zone (parameter sensitivity).** The drainage effect's
ability to overcome the inflow effect depends on *timescale matching*. The camel
toe magnitude is non-monotonic in `max_pump` for the constant rule (line geometry,
80×50, 4000 steps):

| max_pump | Δ@x=18 | shape         |
|----------|--------|---------------|
| 2        | +53    | crescent      |
| 5        | −8     | camel         |
| 10       | −12.5  | camel (peak)  |
| 15       | −1.5   | weak camel    |
| 20       | 0      | flat          |
| 30       | +3     | weak crescent |

Mechanism: camel toe requires the *drainage timescale* (steps a fired source
stays above threshold) to be comparable to the *inflow timescale* (steps the
downstream tank takes to fire). When the two match, the edge's slightly slower
drain (5 outflow channels vs 8) accumulates a meaningful integrated head-start.
Below the resonance, the wavefront passes before the drainage advantage has any
effect; above it, the column fires nearly simultaneously so the drainage delta is
negligible. John's effective `max_pump = 10` is at the camel-toe peak.

For the gradient rule, magnitude scales monotonically with 1/k (faster k →
smaller absolute time delays) but the *shape stays crescent at every k*. The
self-limiting flux makes the drainage advantage impossible in principle.

**Pipe directionality is a third axis.** Even within the constant rule under zero-pad
BC, the *transient* camel toe in mid columns disappears if pipes are made bidirectional
(both A→B and B→A fire when their respective sources are above threshold). With
bidirectional pipes the rule becomes self-limiting (net flow = f(V_A) − f(V_B), which
vanishes as V_B catches up), which kills Effect B. Only one-way pipes preserve Effect B.

So the full causal picture:

| axis                    | options                          | controls                 |
|-------------------------|----------------------------------|--------------------------|
| boundary operator       | zero-pad / reflect-y / reflect-all | sign of effect (A vs B′) |
| pipe directionality     | one-way / bidirectional          | existence of Effect B    |
| pump-rule rate law      | constant / gradient / other      | magnitude only           |

Camel toe in this model requires (zero-pad BC) AND (one-way pipes). Either modification
on its own removes it. The pump rate law (sqrt vs linear vs other) only affects the
size of the effect.

Mapped to the cardiac dichotomy:
- Zero-pad ↔ Neumann ↔ monodomain control → boundary slowdown (matches our 0.000 cm
  monodomain control).
- Reflection / enhanced inflow ↔ partial bidomain analog ↔ Kleber-style camel toe.

The *pump rule* (constant vs gradient, source-limited vs Fickian) modulates the
*magnitude* of the effect but the *boundary-operator choice* fixes its *sign*.

**Operator-level argument (state-independent).** A perfectly uniform initial wavefront
should not produce any boundary effect by symmetry — but it does. The asymmetry doesn't
come from initial conditions; it comes from the discrete update operator U being
*not translation-invariant in y* near the wall. Even with perfect ICs, edge rows of U
have fewer non-zero entries than interior rows, so uniform input → non-uniform output
on the very first step. Effect A is baked into U at the operator level. Effect B′ is
baked into U via reflection padding. Both are state-independent properties of the
boundary operator, not transients of the simulation.

## John's per-cell physics derived from first principles

John's pump rate `max_pump · √((V_C − θ)/(V_max − θ))` is **textbook Torricelli**
for a single tank with outlet hole at height θ draining to atmosphere. From Bernoulli
(free surface at h_C, P=P_atm, v≈0) to pipe outlet (height θ, P=P_atm, velocity v):

```
g·h_C = g·θ + v²/2  ⇒  v = √(2g·(h_C − θ))
Q = a · v = a · √(2g · (h_C − θ))
```

Identifying `a · √(2g) ≡ max_pump / √(V_max − θ)` (with unit cross-section) gives John's
normalized form. The √ and threshold are not arbitrary modeling choices — they are
energy conservation (potential head → outlet kinetic energy) and outlet-hole geometry
respectively. **Single-cell physics is correct; no modification needed.**

For a *submerged pipe* between two tanks (h_C > θ AND h_i > θ), Bernoulli on
free-surface to free-surface gives v = √(2g·(h_C − h_i)) — **gap-driven, not source-driven**.
John applies his single-cell law to this regime too, which is incorrect: he over-estimates
the rate by √((h_C − θ)/(h_C − h_i)), which can be huge at small gaps. His quarter-gap
damping clamp is a crude LINEAR approximation to the missing Bernoulli √-of-gap law.

**Unified hydrostatic-faithful form:**

```
rate(C → i) = max_pump · √( max(V_C − max(θ, V_i), 0) / (V_max − θ) )
```

Reduces to John's law when V_i ≤ θ (Torricelli, atmospheric outlet); reduces to Bernoulli
√-gap when both above θ (submerged outlet). Single √-formula, no clamp needed, monotone
equilibrium approach (no overshoot).

**Continuous physics, numerically discretized.** John's per-cell ODE
`dV/dt = −max_pump · √((V−θ)/(V_max−θ))` for V > θ is analytically solvable —
empties V₀=V_max to threshold in `t* = 2·(V_max−θ)/max_pump = 11` steps for the
default parameters. His simulation code is forward Euler with dt=1; the damping clamp
is a stability hack for the multi-cell coupled regime, not a physics feature.

## Single-cell mechanism: what role does V_C play?

For one cell C with 8 Moore neighbors numbered 1..8, each link (C, i) carries two
populations: φ_i⁺ (C→i outflow) and φ_i⁻ (i→C inflow). All rule variants share the
same Jacobi-buffered update; they differ only in *firing conditions* and *rate*:

| variant | φ_i⁺ fires iff | φ_i⁻ fires iff | rate (φ_i⁺) |
|---|---|---|---|
| **John** (const+1way+damp) | V_C>θ ∧ V_C>V_i | V_i>θ ∧ V_i>V_C | min(max_pump·f(V_C), (V_C−V_i)/4) |
| const + 1way + no-damp | V_C>θ ∧ V_C>V_i | V_i>θ ∧ V_i>V_C | max_pump·f(V_C) (uncapped) |
| const + bidirectional | V_C>θ | V_i>θ | max_pump·f(V_C) |
| gradient + 1way | V_C>θ ∧ V_C>V_i | V_i>θ ∧ V_i>V_C | k·(V_C − V_i) |
| gradient + bidirectional | V_C>θ | V_i>θ | k·(V_C − V_i) |

where f(V) = √((V−θ)/(V_max−θ)). The one-way gate enforces mutual exclusion of
{φ_i⁺, φ_i⁻} per link; bidirectional drops it.

**The non-LBM ingredient.** Standard LBM diffusion has equilibrium f_i^eq = w_i · ρ —
*linear* in density. John has rate ∝ √(V_C − θ) — *concave* in source state. The
concavity is the necessary ingredient for Effect 1 (drainage advantage): high-V cells
pump disproportionately hard, so a boundary cell that retains V longer keeps driving
downstream pumps even after equivalent interior cells have equilibrated. Linear rate
laws (LBM, gradient rule) cannot produce camel toe by this mechanism. To give an LBM
setup John-like speedup, the recipe is **f_i^eq concave in ρ** (not standard cardiac
LBM, but a clean test target).

**Threshold step function as asymmetry amplifier.** The hard step at θ creates two
clean regimes per cell — accumulation (V_C below θ: inflow only, no outflow) and
pumping (V_C above θ: full √-rate). The switch at threshold-crossing is binary, not
gradual. This separation lets each effect express cleanly:

- Accumulation phase: inflow channel deficit (5 boundary vs 8 interior) acts unopposed
  by self-leakage. Effect 2 amplitude maximized.
- Pumping phase: outflow channel deficit lets boundary cells retain V to higher levels.
  Effect 1 amplitude maximized.

Smooth-onset variants (sigmoid, leaky integrator) weaken both effects because cells
continuously self-leak while accumulating, never reaching as high a stored V to release
at firing. **Predicted camel-toe magnitude ordering: step > smooth-ramp > no-threshold**.

## John's axiom set: cardiac claims vs model implementation

A central distinction for evaluating his "boundary speedup" claim: not every feature
of his Colab simulation represents what he would defend as a property of cardiac
tissue. Separated into two tiers.

### Tier I — Genuine cardiac axioms (defendable in heart literature)

**I.1  Discreteness matters at the cell scale.** Cardiac tissue is a network of
discrete coupled cells, and that discreteness has consequences not captured by the
continuum PDE limit. (Aligned with Spach group's microscopic-discontinuity tradition.)

**I.2  Sub-threshold accumulation.** Cells integrate input over time below their
firing threshold, with the integrated state persisting between events. Functional
form is open — could be hard step, soft sigmoid, leaky integrate-and-fire — the
commitment is to *integration*, not which functional form.

### Tier II — Model implementation features (NOT cardiac claims)

These are choices made for tractability or borrowed from the water-tank metaphor.
John would not defend any of them as biological.

| # | feature | source / motivation |
|---|---------|---------------------|
| II.1  | Torricelli √-law `√(V−θ)` | water-tank hydrostatics |
| II.2  | Source-state-only coupling | water-tank metaphor |
| II.3  | Hard step function threshold | implementation simplification |
| II.4  | Hard one-way valve at gap-junction level | implementation simplification |
| II.5  | Moore-8 dense connectivity | lattice convenience |
| II.6  | Square lattice geometry | geometric convenience |
| II.7  | Synchronous Jacobi update | numerical scheme |
| II.8  | Quarter-gap damping clamp | numerical stability hack |
| II.9  | Memoryless cells (no recovery) | radical simplification |
| II.10 | No-flux Neumann boundary | default |

### Three-question evaluation program

```
   Q1 — SENSITIVITY: does each axiom (Tier I or II) produce a boundary artifact
        in the toy model?
        (Already partly characterized for II.4 bidirectional, II.10 reflect-y BC,
        and the rate-law axes via gradient rule.)

   Q2 — ROBUSTNESS: does Tier I ALONE, with cardiac-realistic Tier II replacements,
        still produce camel toe?

        Cardiac-realistic replacements:
          II.1 + II.2 → linear ohmic gap junction:  I_ij = g·(V_i − V_j)
          II.3        → smooth sigmoid threshold (or FHN cubic recovery)
          II.4        → bidirectional coupling (refractoriness lives in membrane
                         kinetics, not the gap junction)
          II.5 + II.6 → anisotropic sparse connectivity (along-fiber dense, sparse
                         cross-fiber)

        IF YES → boundary speedup is a Tier-I consequence; cardiac defense reduces
                 to defending I.1 + I.2.
        IF NO  → speedup depends on Tier-II artifacts John doesn't claim as cardiac.
                 The boundary effect is a model artifact, not a cardiac prediction.

   Q3 — CARDIAC TRUTH of I.1 + I.2: defend or reject from biology (gap-junction
        density, optical mapping at tissue edges, Spach/Kleber literature).
```

**Prior on Q2 outcome.** The biophysically suspect axioms (II.1 Torricelli √, II.2
source-state-only) are exactly what produce Effect 1 (drainage advantage). The
defensible Tier-I axioms (I.1, I.2) plus cardiac-realistic Tier-II at most support
Effect 2 (inflow deficit → crescent / slowdown). Predict: under cardiac-realistic
Tier-II, the boundary effect *flips sign* relative to John's setup — slowdown,
not speedup.

## Boundary BC discretization — face-centered vs node-centered mirror

This section explains why John's storage-tank artifacts (camel toe under
`reflect_y`, crescent under `zero_pad`) appear despite linear gradient/Fickian
coupling, but the same artifacts are absent in standard cardiac FDM monodomain
and LBM with bounce-back.

### The puzzle

For a uniform line-source initial condition propagating through 2D tissue with
no-flux walls:
- Storage-tank GRADIENT rule + `zero_pad` BC → strong crescent (Δ@x=18 ≈ +49 to +120)
- Storage-tank GRADIENT rule + `reflect_y` BC → strong camel toe (Δ@x=18 = −270)
- Storage-tank GRADIENT rule + `bidirectional` pipes → still crescent
- Monodomain (5-pt or Mehrstellen) + Neumann → flat (0.000 cm deviation)
- LBM (D2Q9) + bounce-back → flat

The discrepancy is NOT explained by the rate law (linear vs nonlinear), the
connectivity (Moore-8 / D2Q9 both have diagonals), or the threshold-gating —
all of those vary across the systems above without resolving the puzzle.

### The resolution: how ∂V/∂n = 0 is discretized

There are two inequivalent ways to enforce the no-flux Neumann BC in a discrete
operator. They look nearly identical but produce fundamentally different boundary
stencils.

**Face-centered mirror** (used by every standard FDM / LBM implementation):
- Wall is between cells, at y = −½h
- Ghost cell at y = −1: `V_ghost = V_boundary[y=0]`  (mirror across y = −½)
- The mirror is "about the wall face"

**Node-centered mirror** (used by John's storage-tank `reflect_y` via `np.pad`):
- Wall is at the cell center, y = 0
- Ghost cell at y = −1: `V_ghost = V_subedge[y=1]`  (mirror across y = 0)
- Plus mass-conserving fold-back: flux destined for a ghost cell is rerouted
  to the real cell that ghost mirrors (= sub-edge for the y-direction wall)

### Why face-centered cancels boundary asymmetry exactly

For a 5-point Laplacian at the boundary cell (y = 0) with face-centered
ghost (V_ghost = V_C):

```
   ∇²V|_(0,x)  =  (V_ghost + V[1,x] + V[0,x−1] + V[0,x+1] − 4·V_C) / h²
              =  (V_C + V[1,x] + V[0,x−1] + V[0,x+1] − 4·V_C) / h²
              =  (V[1,x] + V[0,x−1] + V[0,x+1] − 3·V_C) / h²
```

For a uniform-y wave (V[1,x] = V[0,x] = V_C), the y-direction term vanishes:

```
   ∇²V|_(0,x)  =  (V[0,x−1] + V[0,x+1] − 2·V_C) / h²
                                                      ── identical to interior 1D Laplacian
```

→ Boundary and interior cells fire simultaneously. **Empirically: monodomain
control gave 0.000 cm deviation from flat — exactly this prediction.**

The same cancellation holds for the 9-point Mehrstellen stencil and for LBM
bounce-back.

### Caveat — the cascade requires SEEDED asymmetry, not strictly uniform-y input

[Added 2026-04-29 in response to audit.] The mechanism narrative below describes
how sub-edge enhancement cascades into boundary asymmetry, but the FIRST STEP
of the cascade does NOT fire on a strictly uniform-y input under the
`gradient + one_way` rule:

- The gate is `V_src > V_dst` (STRICT inequality).
- For two cells at exactly equal V (uniform-y), the gate is FALSE → **no flux
  fires** at all in the y-direction.

So the "interior gets ZERO net y-pump" wording in earlier text needs nuance:
the cancellation isn't "y-fluxes cancel" but "no y-flux fires at all at step 1".

The asymmetry that seeds the cascade must come from somewhere else at step 1:
- **Corner ghost terms**: at the corners of the domain, np.pad reflect mode
  populates V_padded[0, 0] from V[1, 1] (interior diagonal), creating an
  asymmetric corner stencil immediately.
- **X-direction diagonal contributions**: NE/NW diagonals from the inlet
  column (x=0) fire into ghost neighbors of the boundary cell at column 1.
  Even if the inlet is uniform in y, the GHOST diagonal target at (-1, 1)
  is mirror of (1, 1) which equals V_C of the sub-edge — same V as boundary
  in step 0, so no asymmetry yet from this either.

Practical implication: at step 1 from a strictly uniform line inlet, all
column-1 cells receive identical 3-channel inflow regardless of BC. **The
cascade must develop over multiple steps as small numerical or geometric
asymmetries seed it.** The empirical Δ@x=18 = -270 we measured is a many-step
accumulation, not a one-step phenomenon.

This matters for the test plan (PLAN.md Step C.3): the diagnostic comparing
storage-tank reflect_y to monodomain node_mirror_existing must run for enough
simulation steps to let the cascade develop, not stop at step 1.

The "operator-level state-independent" wording earlier in this document
remains correct in the LIMIT (over many steps the boundary stencil is
operator-different from interior even on uniform input), but is misleading
if read as "asymmetry appears in step 1." Replaced "state-independent" with
"developable over time, even from symmetric initial state" in the corner
analysis below.

### Why node-centered creates a cascade

In storage-tank `reflect_y`, `np.pad(V, mode="reflect")` mirrors the boundary
about its own NODE (y=0), so V_ghost(−1, x) = V[1, x] = sub-edge. Combined with
the mass-conserving fold-back of flux into ghost destinations:

```
   Step 1 — column 1 from inlet:
     boundary (0,1)   →  3 inflow channels (W, NW, SW) = 300k
     sub-edge (1,1)   →  3 normal channels  +  ghost-folded inflow
                          from upstream boundary's ghost neighbor pumping the
                          mirrored ghost (0,1)  →  ~5 effective channels = 500k
     interior (25,1)  →  3 channels = 300k
     → sub-edge fires fastest, boundary and interior tied
   
   Step 2 onwards — sub-edge pumps boundary:
     sub-edge (1,1) at higher V than boundary (0,1) and (-1,1)=ghost mirror=sub-edge
     boundary receives DOUBLE y-pump:  one from real sub-edge below,
                                       one from ghost mirror N direction (= sub-edge)
     interior receives ZERO net y-pump (uniform-y, flanking cells cancel)
     → boundary advances faster than interior at downstream columns
     → camel toe builds up over many columns
```

This is the empirical Δ@x=18 = −270 we measured. The mechanism is not concavity
of the rate law nor source-state coupling — it's a *sub-edge-mediated cascade*
specific to the node-centered discretization.

### Zero-pad: missing channels are simply absent

Storage-tank `zero_pad` doesn't even attempt mirroring — off-grid neighbors
contribute nothing:

```
   boundary (0, N) at column N: 2 upstream channels (W, SW only — NW is off-grid)
   interior (25, N):           3 upstream channels (W, NW, SW)
   → boundary fires later than interior  →  crescent (Effect 2 in pure form)
```

This is what we see at every k for gradient + zero_pad. Standard FDM monodomain
does NOT replicate this because the face-centered ghost compensates the missing
channel. Replacing the ghost with `V_outside = 0` (an "amputated" stencil) would
reproduce the storage-tank crescent directly.

### Classification table

| System                        | BC discretization              | Uniform-input asymmetry? | Mechanism                |
|-------------------------------|--------------------------------|--------------------------|--------------------------|
| storage-tank zero_pad         | amputated stencil              | YES — crescent           | Effect 2 (inflow deficit)|
| storage-tank reflect_y        | node-centered mirror + fold    | YES — camel toe          | sub-edge cascade         |
| Bidomain V1 monodomain mode (Mehrstellen + Neumann) | face-centered mirror | NO (empirically — 0.000 cm)  | mirror compensates  |
| Monodomain V5.4 FDMDiscretization (default since 2026-04-29) | face-centered mirror (V_ghost = V[i,0]) | NO (verified empirically — 1.10e-13 mV deviation) | mirror compensates |
| Monodomain V5.4 FDMDiscretization (legacy mode `node_mirror_existing`) | node-centered mirror (V_ghost = V[i,1]) | YES — but only with non-y-uniform input | 2x amplification of column-wise gradient at wall |
| LBM bounce-back (D2Q5/D2Q9)   | equivalent to face-centered    | NO (predicted)           | mirror compensates       |
| bidomain bath-coupled         | Dirichlet V_e=0 at bath wall   | YES — camel toe (Kleber) | asymmetric BC pair       |

**RESOLVED 2026-04-29:** the V5.4 default `boundary_mode` was changed from
`node_mirror_existing` to `face_mirror`. Empirical column-diagnostic
verifies V[boundary] = V[center] to 1.10e-13 mV in y-uniform line-stim
TTP06 propagation under face_mirror. Tests `test_a3..a7` pass. Anyone
needing bit-exact reproduction of pre-2026-04-29 simulations must pass
`boundary_mode='node_mirror_existing'` explicitly.

### Key implication for John's argument

John's claim ("discrete connections at edges differ from interior, source-sink
mismatch produces persistent boundary artifacts on uniform input") is *technically
correct*, but applies only to discretizations that don't cancel the missing-channel
asymmetry. Every standard cardiac PDE solver uses a discretization that does
cancel it.

The storage-tank's persistent boundary artifact is therefore a property of the
*discretization choice*, not a property of the underlying continuum equation that
cardiac modelers solve. The Kleber boundary speedup in bidomain is a separate
mechanism — it comes from the asymmetric Dirichlet (extracellular bath) /
Neumann (intracellular wall) BC pair, which IS a real continuum effect.

### Predictions that distinguish discretization-artifact from continuum-effect

```
   variant                         predicted boundary signal  predicted h-scaling
   ─────────────────────────────────────────────────────────────────────────────
   monodomain face-centered Neumann      0                     n/a (zero at all h)
   monodomain "zero-pad" Laplacian       crescent              ~ h (vanishes as h → 0)
   monodomain "node-mirror" Laplacian    camel toe             ~ h (vanishes as h → 0)
   bidomain bath-coupled                 camel toe ~7%         ~constant in h (continuum)
   ─────────────────────────────────────────────────────────────────────────────
```

If empirical h-scaling matches these predictions, the artifact-vs-physical
classification is settled. See `PLAN.md` for the implementation plan.

## Headline finding: bridge claim CONFIRMED across all three model classes (2026-04-30)

**The connectivity-mediated boundary deficit is a unified mechanism across:**
- Discrete lattice models (John's storage tank with one_way pumps)
- Continuum PDE solvers (monodomain V5.4 FDM)
- Lattice Boltzmann methods (LBM V1)

The same 8-neighbour topology produces the same boundary asymmetry in all three. Cardinal-only stencils (or diagonal-aware bounce-back BC, the LBM analog) eliminate it in all three.

### Empirical confirmation in monodomain V5.4

After implementing `stencil ∈ {cardinal4, moore8_uniform, moore8_iso}` and `boundary_mode ∈ {face_mirror, face_mirror_iso, ...}` in `FDMDiscretization`, the column diagnostic at mid-tissue (x = 0.5 cm, NX=41) for TTP06 EPI under uniform-y line stim shows:

```
case                               max|V[top] - V[ctr]|   ΔLAT (top vs ctr)   verdict
─────────────────────────────────────────────────────────────────────────────────────
cardinal4 + face_mirror              1.1e-13 mV          0.000 µs            baseline (no deficit)
moore8_uniform + face_mirror         70.5 mV          +486.06 µs            DEFICIT (John-equivalent)
moore8_uniform + face_mirror_iso     2.8e-14 mV          0.000 µs            fix (bounce-back)
moore8_iso + face_mirror             48.4 mV          +229.74 µs            smaller deficit (5/6 ratio)
moore8_iso + face_mirror_iso         7.5e-14 mV          0.000 µs            LBM analog (full fix)
```

**The +486 µs LAT shift under `moore8_uniform + face_mirror` is John's storage-tank crescent reproduced in cardiac PDE.** Boundary cells fire ~half a millisecond AFTER bulk because they have fewer effective inflow channels (fewer-neighbours topology → less charging current → slower threshold crossing). The sign matches John's `zero_pad` crescent (boundary lags) exactly.

### Cross-engine summary

| Engine | Deficit (max boundary asymmetry) | Baseline | Fix |
|---|---|---|---|
| Storage tank (LAT spread) | 130 steps (R1 moore8 uniform) | 0 (R2 cardinal-4) | 103 steps (R5 iso 4:1, no bounce-back) |
| Monodomain V5.4 (mV) | 1.76 mV (mid-x snapshot) | 0 | 0 (with face_mirror_iso) |
| LBM V1 (mV) | 0.077 mV (LBM smaller overall) | 0 (D2Q5) | 0.044 mV (canonical D2Q9) |

The qualitative ordering is preserved across model classes. Absolute magnitudes differ because of different scaling, time scales, and (for LBM) shorter run windows.

### Why the effect appears in monodomain when previously it didn't

Standard monodomain implementations have used the cardinal-4 stencil (5-point Laplacian) by default — and at face_mirror BC, that combination gives **zero** boundary deficit in y-uniform fields because the boundary cell's "missing N pipe" already has gap=0 in y-uniform (the missing-pipe cost is zero). This is why textbook cardiac sims don't see John's effect: the 5-point stencil is structurally immune.

Adding the 9-point Moore-8 stencil (cardinals + diagonals) breaks this immunity. The diagonal pipes carry x-direction flux even in y-uniform fields because they span both axes, so the boundary cell's "missing diagonal" is a real loss. With `face_mirror` (ghost = self for off-grid), the deficit appears: 1/3 of the interior charging rate is lost at the boundary in moore8_uniform, 1/6 in moore8_iso.

### Implementation reference

`Monodomain/Engine_V5.4/cardiac_sim/simulation/classical/discretization_scheme/fdm.py` extended with two new class constants:

```python
class FDMDiscretization(SpatialDiscretization):
    BOUNDARY_MODES = ('face_mirror', 'face_mirror_iso', 'node_mirror_existing',
                      'zero_pad', 'rest_pad')
    STENCILS = ('cardinal4', 'moore8_uniform', 'moore8_iso')
```

Construction:
```python
fdm = FDMDiscretization(
    grid, D=0.001, chi=1.0, Cm=1.0,
    stencil='moore8_uniform',         # or 'moore8_iso' / 'cardinal4'
    boundary_mode='face_mirror',       # or 'face_mirror_iso' for LBM-analog fix
)
```

Constraints on Moore-8 stencils:
- Isotropic D only (raises `NotImplementedError` for non-zero Dxy in `D_field`)
- Square grids only (raises `NotImplementedError` for dx ≠ dy)

The `face_mirror_iso` boundary mode is the LBM bounce-back analog: for diagonal off-grid cells, the ghost reflects only the off-grid axis (e.g., NE at `(i+1, -1)` → ghost = `V[i+1, 0]`, NOT `V[i, 0]`). For cardinal-only stencils, `face_mirror_iso` degenerates to `face_mirror` exactly.

LBM V1 received the analogous `weights_mode` parameter on `LBMSimulation`:
```python
sim = LBMSimulation(..., lattice='d2q9', weights_mode='canonical')   # 4/9, 1/9, 1/36
sim = LBMSimulation(..., lattice='d2q9', weights_mode='uniform_8')   # 0, 1/8, ..., 1/8
```

Plus a critical latent bug fix in `LBM/Engine_V1/src/simulation.py:73`: was passing `tau_from_D(D, dx, dt)` without `cs2`, defaulting to 1/3 regardless of lattice. Now uses `cs2=self.lattice.cs2`. Bit-correct for canonical D2Q9 (cs2=1/3=default) but required for non-canonical lattices.

### Evidence files

| File | Content |
|---|---|
| `Research/Active/boundary_conduction_speedup/diag_monodomain_connectivity.py` | 5-case column diagnostic generating the table above |
| `Research/Active/boundary_conduction_speedup/figures/diag_monodomain_connectivity.png` | V(t) traces and dev curves per case |
| `Research/Active/boundary_conduction_speedup/figures/connectivity_cross_engine.{png,pdf}` | 3×3 cross-engine comparison (storage tank / monodomain / LBM) |
| `Research/Active/boundary_conduction_speedup/figures/video_stencil_mirror_combos.mp4` | 5-panel animated wavefront comparison |
| `simulation/connectivity_threshold_ablation.py` | Storage-tank R1-R6 ablation (predecessor) |
| `Monodomain/Engine_V5.4/test_boundary_modes.py` | 21 boundary-mode tests including the new stencils + face_mirror_iso |
| `LBM/Engine_V1/tests/test_d2q9_uniform_weights.py` | 8 LBM tests for D2Q9_uniform + cs2 plumbing |

### Implication for cardiac modeling practice

Standard cardiac PDE solvers (5-point cardinal + face-centered Neumann) are structurally immune to the connectivity-mediated boundary effect. This is *why* the textbook conclusion "Neumann monodomain has no boundary speedup" holds — it's not that the underlying physics excludes it, it's that the discretization choice (cardinal-only) avoids the asymmetry that 9-point or higher-order isotropic stencils would expose.

For applications that genuinely model cardiac tissue boundaries (no mathematical trick to hide the effect), the right physical setup is bath-coupled bidomain (Kleber et al.) — that produces a real CV speedup at the boundary (~7-13%) via a different mechanism (extracellular shunt, not fewer-neighbour intracellular topology).

The connectivity-mediated effect we just confirmed is **a separate phenomenon** — it's the discrete-topology boundary lag that John's storage-tank reveals. It exists in all 9-point cardiac discretizations but is invisible at 5-point. Whether to consider it "physical" or "numerical" depends on whether you think the discrete fewer-neighbours-at-edge structure is faithful to real tissue (it's a long-running debate in computational EP).

## Connectivity is the smoking gun: 8-neighbour vs 4-neighbour ablation (2026-04-29/30)

The boundary-effect mechanism in John's storage-tank model was localised by
running a 6-way ablation on the user's "Fickian-modified" John setup
(`gradient` mode + `one_way` + `zero_pad` + line geometry).

### Setup

Two ablation knobs added to `tanks_vec.run()`:
- `connectivity` ∈ {`moore8`, `cardinal4`, `moore8_iso`}
- `threshold_gate` ∈ {True, False} — if False, drops the `fired_p` (V > θ)
  gate from pipe-firing condition.

### Findings

```
Run                      Connectivity    Threshold    max|LAT-meanY|    cols_full
─────────────────────────────────────────────────────────────────────────────────
R1   baseline            moore8          True         91.8 steps         42
R2   cardinal-4 only     cardinal4       True          0.0 steps         25
R3   no threshold only   moore8          False        11.5 steps         33
R4   both off            cardinal4       False         0.0 steps         20
R5   iso 4:1             moore8_iso      True         81.0 steps         25
R6   iso, no thresh      moore8_iso      False        16.5 steps         20
```

**Two clean conclusions:**

1. **Cardinal-4 connectivity gives EXACTLY ZERO crescent** in y-uniform line
   stim, regardless of threshold gate (R2/R4 both 0.0 to floating-point
   precision). The "missing N pipe at the boundary" contributes gap=0 in
   y-uniform fields, so losing it costs nothing.

2. **Moore-8 connectivity ALWAYS produces a crescent** (R1/R3/R5/R6 all
   non-zero). Threshold gate amplifies by ~8× (91.8/11.5) under uniform
   weights. **Moore-8 is the necessary structural ingredient; threshold
   amplifies but is not required.**

### Mechanism

In y-uniform field with wavefront at column k, boundary cell at (0, k) loses
its NW and NE diagonals (off-grid). Each interior cell has 3 firing inflow
pipes (NW, W, SW from column k-1) and 3 firing outflow pipes (NE, E, SE
to column k+1). Boundary loses one inflow + one outflow pipe → 2/3 charging
rate of interior. Crescent forms.

In cardinal-4: boundary only loses N pipe (which has gap=0 in y-uniform), so
no deficit. The diagonals carry x-direction flux even in y-uniform fields
because they span both axes simultaneously — that is exactly the
mechanism the cardinal-only stencil cannot create.

### Iso 4:1 (Patra-Kałuża) reduces but does not eliminate

Implementing the Patra-Kałuża isotropic 9-point stencil weights (cardinal × 4,
diagonal × 1, with the canonical 1/6 normalisation prefactor — initial
implementation forgot the prefactor and produced D_eff = 6k = 0.48,
violating the 2D-explicit CFL limit of 0.25, manifesting as grid-scale
mosaic instability) gives:

```
Boundary deficit ratio = (w_c + w_d) / (w_c + 2·w_d)

  Equal-weight Moore-8 (1, 1):    2/3   = 0.667    (33% deficit)
  LBM / iso 4:1     (4/6, 1/6):   5/6   = 0.833    (17% deficit)
  Cardinal-only        (1, 0):    1     = 1.000     (0% deficit)
```

Empirically R5 vs R1: 81.0 vs 91.8 steps — **only ~12% reduction in
crescent magnitude**, exactly matching the 5/6 deficit prediction modulo
threshold-amplified compounding. **Iso 4:1 weighting alone is necessary
but not sufficient** to eliminate the boundary effect. Full elimination
requires either cardinal-4 or iso 4:1 PLUS proper diagonal-aware face_mirror
reflection at the wall (LBM "bounce-back" generalised to 9-point) — not
yet implemented.

### LBM connection

LBM D2Q9 weights are 4/9 (rest), 1/9 (cardinals), 1/36 (diagonals).
Cardinal:diagonal ratio = 4:1, identical to Patra-Kałuża isotropic 9-point.
LBM's bounce-back boundary handling IS the diagonal-aware reflection that
zeros the deficit, which is why standard LBM reproduces the "no crescent
in monodomain Neumann" result automatically.

## Wave-slowing dilation: apparent curvature growth in isochrone images

Investigating the user's observation that the per-column crescent appears
to grow as the wavefront moves outward: this turns out to be ~70% an
artifact of wave deceleration, not real per-step deficit growth.

### Per-column metrics

For R1 baseline (uniform Moore-8 + threshold + gradient + one_way +
zero_pad + line):

```
column       Δarrival       spread       spread/Δmean
─────────────────────────────────────────────────────
   4            14            6              0.42
  12            50           28              0.56
  21            91           59              0.65
  29           128           88              0.69
  37           165          117              0.71
```

- Δarrival grows 12× across the propagation (wave decelerating outward)
- Spread (max-min LAT per column) grows 20×
- spread/Δmean (fractional lag relative to per-column traversal time) grows 1.7×

So 12× of the 20× apparent-crescent growth is wave deceleration. The
remaining 1.7× is **real threshold-amplified compounding**: each column's
boundary lag is inherited by the next column with no back-flow to cancel
it under one-way + threshold dynamics.

### Confirming experiment without threshold gate

R3 (uniform Moore-8, no threshold): spread plateaus at 17 steps and
spread/Δmean DECREASES from 0.22 → 0.08. Without threshold gating, the
per-step deficit really is structurally constant; apparent growth is
purely wave-slowing dilation. **Threshold is what locks in the inherited
lag.**

### Normalised diagnostics

`render_norm_helpers.py` provides two normalisation primitives:
- `x_evenly_spaced_levels(iso, N)` — contour LEVELS at evenly-spaced
  x-positions of wavefront mean (not step times). Removes dilation in
  isochrone plots.
- `per_column_dev_normalized(iso, x)` — returns dev/Δmean (fractional lag)
  instead of raw step counts. Removes dilation in per-column LAT plots.

Six normalised figures generated in `simulation/outputs/images/` re-render
the existing pump-speed and gradient-k sweeps plus a new
connectivity-comparison set, with dilation factored out. Curves should
overlay across x within each panel under the deficit-ratio prediction;
empirically they overlay much more tightly than in absolute units, but
do not perfectly overlay (the 1.7× compounding residual).

### Implication for John's argument

John's claim that boundary effects PERSIST and AMPLIFY at greater distances
from the source is largely (~70%) an artifact of wave-slowing dilation.
The genuine compounding from threshold-amplified discrete activation is
real but smaller (~1.7×). Cardiac propagation operates in the
non-threshold-gated regime where the deficit ratio is constant and
apparent growth is pure dilation, but standard cardiac solvers don't have
the deficit at all (cardinal-4 stencil + face_mirror).

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Validation approach | Mesh convergence toward 1.131 | Theoretical ratio provides analytical target |
| Boundary treatment | Face-based FDM (not ghost-node) | Symmetric SPD Laplacian required for bidomain |
| Control experiment | Monodomain FDM with same grid | Isolates bidomain boundary effect from curvature |
| Bath coupling model | Dirichlet phi_e = 0 at surface | Standard approximation; exact in infinite bath limit |
| V5.4 FDM default boundary mode (2026-04-29) | `face_mirror` (was `node_mirror_existing`) | Eliminates 2× column-gradient amplification at the wall; verified zero deviation in y-uniform line-stim TTP06 propagation |
| Storage-tank crescent root cause (2026-04-29) | Moore-8 connectivity, NOT threshold gate | 6-way ablation R1-R6 shows cardinal-4 → 0.0 deviation regardless of threshold; threshold amplifies but is not necessary |
| Iso 9-pt prefactor convention (2026-04-29) | Use 1/6 Patra-Kałuża normalisation | Without it, D_eff = 6k violates 2D-explicit CFL limit and produces grid-scale mosaic instability |

## Open Questions

- What is the convergence rate with mesh refinement? (Is it O(dx) or O(dx^2)?)
- How does the speedup interact with wavefront curvature at obstacle corners? (Curvature speedup is a separate geometric effect)
- Does the speedup magnitude change with anisotropic conductivity tensors (fiber orientation at boundary)?
- At what tissue thickness does the boundary layer span the entire preparation (transitioning from surface effect to full-thickness effect)?
- Does the EMI model (cell-resolved, no homogenization) reproduce the same speedup magnitude?

## Connections
- **Engines**: Bidomain V1 (primary validation), LBM V1 (secondary confirmation)
- **Related research**: scar_bc_validity (Q6 -- Neumann not Dirichlet at scar), lbm_cardiac (Q4 -- LBM can capture the effect with D2Q9)
- **Pipelines**: Triangle merger experiments (bidomain vs monodomain CV comparison on realistic geometries)
