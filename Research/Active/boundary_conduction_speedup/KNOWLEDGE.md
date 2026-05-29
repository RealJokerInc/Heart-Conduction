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
See "Equilibrium argument" below for the structural proof and empirical
k-sweep confirmation.

### Equilibrium argument: why Fickian is sign-locked to crescent at every k (2026-05-02)

Take a fired source cell mid-wavefront. Upstream column at V_up, downstream
column ≈ 0, lateral N/S gaps ≈ 0 in the y-uniform regime. Under Fickian:

```
dV/dt = k·N_in·(V_up − V) − k·N_out·V

V*(y) = [N_in / (N_in + N_out)] · V_up   (steady state)
τ(y)  = 1 / [k·(N_in + N_out)]            (time constant)
```

In the moore8 + zero_pad lattice with one_way pipes and uniform y, both edge
and interior cells have N_in = N_out (3,3 interior; 2,2 edge), so:

```
V*(edge)     = (2/4)·V_up = V_up/2
V*(interior) = (3/6)·V_up = V_up/2
ratio        = 1            ←  IDENTICALLY, independent of k
```

**Edge plateau equals interior plateau.** No y-asymmetric stockpile exists
to discharge. The time constant τ does differ — τ_edge = 1/(4k) takes 50%
longer to reach V* than τ_interior = 1/(6k) — but that is just Effect A
re-stated as a charging time, not a differential downstream-pumping
advantage. Once V*(y) is reached, the per-pipe outflow rate k·(V* − V_down)
is the same at edge and interior; total downstream pumping
= k·V*·N_out scales with N_out, so the edge actually pumps LESS total
fluid into the next column than interior does. Effect B doesn't merely
vanish under Fickian — it inverts and *reinforces* Effect A.

Replacing k → α·k everywhere rescales τ uniformly (isochrones stretch by
1/α) but leaves the V*(y) ratio at 1. **No k can produce a non-unity
ratio.** This is structural, not parametric.

### Capacitor vs resistor mnemonic

```
                    John's constant rule        Fickian gradient rule
─────────────────────────────────────────────────────────────────────────
Phase structure     Fill → dwell → drain        No phases — asymptotic
Stockpile high V?   YES (parks near V_max)      NO (asymptotes to V_up/2)
Drain depends on V_down?  No (until cap kicks)  Yes (linear in gap)
Edge advantage      Slower drain → more         No advantage — V* is
                    integrated downstream       y-independent and total
                    pumping → camel toe         pumping ∝ N_out
                    possible                    favours interior
Behaves like…       Capacitor discharging       Resistor in steady state
─────────────────────────────────────────────────────────────────────────
```

Capacitors hold a stockpile that asymmetric drainage can release on its
own schedule. Resistors carry no stockpile to release asymmetrically.
Effect B is the capacitor's discharge bonus; it cannot exist in a
resistor network.

### Empirical confirmation (2026-05-02 k-sweep)

`Nx=80`, `Ny=50`, `gradient` mode, `moore8` connectivity, `zero_pad`,
`one_way`, line stim, `threshold=45`, sweeping k. Edge−center LAT (steps,
positive = edge fires LATER = crescent):

```
   k     x=10   x=20   x=30   x=40   x=50   x=60
 0.200    +12    +15    +22    +28    +33    +38
 0.120    +18    +43    +72   +105   +132   +159
 0.080    +22    +56    +92   +127   +161   +192
 0.040    +41   +100   +160   +214   +265    --
 0.020    +79   +185   +290    --     --     --
 0.010   +155   +357    --     --     --     --
 0.005   +307    --     --     --     --     --
```

Sign locked positive at every k from 0.005 to 0.20 (40× range).
Magnitude scales as 1/k as predicted by Effect A combined with
wave-slowing dilation. (`--` columns indicate the wavefront stalled
before reaching that x — see Finding 1 in IDEALOG re finite-distance
propagation under threshold gating.)

### Generalisation

The equilibrium argument generalises to **any** flux law of the form
`q-dot = f(V_src − V_dst)` with f(0) = 0 and f monotonic. Solving
dV/dt = 0 in the y-uniform regime gives V*(y) = N_in/(N_in+N_out)·V_up,
which is y-independent whenever N_in = N_out at every cell. Any such law
can only manifest Effect A (geometric inflow deficit) and is sign-locked
to crescent. The bias is baked into the diffusion operator's structure,
not the rate-law parameters.

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
- Monodomain (cardinal-4 stencil) + Neumann → flat (0.000 cm deviation)
- LBM (D2Q5) + bounce-back → flat
- LBM (D2Q9) + bounce-back → forward crescent (small magnitude; see §"Three BC families" below — HBB is NOT a "fix", it has the same sign as face_mirror)

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
| LBM halfway bounce-back (D2Q5) | NO diagonals → no deficit channel | NO                | no diagonal mass to lose at wall |
| LBM halfway bounce-back (D2Q9, canonical 4:1) | structurally ≡ face_mirror + moore8_iso | YES — forward crescent (mild, 5/6) | diagonal upstream-V info lost at wall (see §"Three BC families") |
| LBM halfway bounce-back (D2Q9, uniform_8 1:1) | structurally ≡ face_mirror + moore8_uniform | YES — forward crescent (full, 2/3) | same mechanism, more diagonal weight = bigger deficit |
| LBM specular reflection (D2Q9) | structurally ≡ face_mirror_iso | NO — zero structural bias | diagonal mass crosses to neighbour cell, preserves upstream-V (see §"Three BC families") |
| LBM horizontal redirect (D2Q9, novel) | NEW family — no PDE analog yet | YES — sustained INVERSE crescent | diagonal mass redirected to cardinal eastward slot at wall (see §"Three BC families") |
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
   variant                                       predicted boundary signal   predicted h-scaling
   ────────────────────────────────────────────────────────────────────────────────────────────
   monodomain face_mirror + cardinal4                  0                     n/a (zero at all h)
   monodomain face_mirror + moore8_uniform             crescent (2/3)        ~ constant
   monodomain face_mirror + moore8_iso                 crescent (5/6, mild)  ~ constant
   monodomain face_mirror_iso + any moore8 stencil     0                     n/a (zero at all h)
   monodomain "zero-pad" Laplacian                     crescent              ~ h (vanishes as h → 0)
   monodomain "node-mirror" Laplacian                  camel toe             ~ h (vanishes as h → 0)
   bidomain bath-coupled                               camel toe ~7%         ~constant in h (continuum)
   ────────────────────────────────────────────────────────────────────────────────────────────
```

If empirical h-scaling matches these predictions, the artifact-vs-physical
classification is settled. See `PLAN.md` for the implementation plan.

## Headline finding: bridge claim CONFIRMED across all three model classes (2026-04-30, refined 2026-05-14)

**The connectivity-mediated boundary deficit is a unified mechanism across:**
- Discrete lattice models (John's storage tank with one_way pumps)
- Continuum PDE solvers (monodomain V5.4 FDM)
- Lattice Boltzmann methods (LBM V1)

The same 8-neighbour topology produces the same boundary asymmetry in all three. Cardinal-only stencils eliminate the deficit in all three. The LBM bounce-back family is **structurally equivalent to face_mirror** (NOT to face_mirror_iso — see §"Three BC families: structural bias hierarchy" below for the corrected mapping established 2026-05-14). The LBM analog of face_mirror_iso is specular reflection.

### Empirical confirmation in monodomain V5.4

After implementing `stencil ∈ {cardinal4, moore8_uniform, moore8_iso}` and `boundary_mode ∈ {face_mirror, face_mirror_iso, ...}` in `FDMDiscretization`, the column diagnostic at mid-tissue (x = 0.5 cm, NX=41) for TTP06 EPI under uniform-y line stim shows:

```
case                               max|V[top] - V[ctr]|   ΔLAT (top vs ctr)   verdict
─────────────────────────────────────────────────────────────────────────────────────
cardinal4 + face_mirror              1.1e-13 mV          0.000 µs            no deficit (no diagonals to lose)
moore8_uniform + face_mirror         70.5 mV          +486.06 µs            DEFICIT 2/3 (full John-equivalent)
moore8_uniform + face_mirror_iso     2.8e-14 mV          0.000 µs            FIX — face_mirror_iso eliminates
moore8_iso + face_mirror             48.4 mV          +229.74 µs            DEFICIT 5/6 (mild, canonical D2Q9 weights)
moore8_iso + face_mirror_iso         7.5e-14 mV          0.000 µs            FIX — face_mirror_iso eliminates
```

**Note (corrected 2026-05-14):** face_mirror_iso is a PDE-only construction. It is NOT the LBM bounce-back analog. The LBM analog of face_mirror_iso is **specular reflection**. LBM bounce-back (HBB) is structurally equivalent to face_mirror (forward crescent, same sign-lock). See §"Three BC families" below for the full corrected mapping.

**The +486 µs LAT shift under `moore8_uniform + face_mirror` is John's storage-tank crescent reproduced in cardiac PDE.** Boundary cells fire ~half a millisecond AFTER bulk because they have fewer effective inflow channels (fewer-neighbours topology → less charging current → slower threshold crossing). The sign matches John's `zero_pad` crescent (boundary lags) exactly.

### Cross-engine summary

| Engine | Full deficit (2/3 family) | Mild deficit (5/6 family) | Zero-deficit BC | Notes |
|---|---|---|---|---|
| Storage tank (LAT spread) | 130 steps (R1 moore8 uniform) | 103 steps (R5 iso 4:1) | 0 (R2 cardinal-4) | both moore8 variants have forward crescent |
| Monodomain V5.4 (max ΔV at mid-x) | 70.5 mV (moore8_uniform + fm) | 48.4 mV (moore8_iso + fm) | ~0 (face_mirror_iso OR cardinal4) | face_mirror_iso fully eliminates |
| LBM V1 (max ΔV at mid-x) | 0.077 mV (D2Q9 uniform_8 + HBB) | 0.044 mV (D2Q9 canonical + HBB) | ~0 (D2Q5 + HBB, or D2Q9 + specular) | specular is LBM analog of fm_iso |

The qualitative ordering is preserved across model classes. Absolute magnitudes differ because of different scaling, time scales, and time-step calibrations between LBM and PDE — but the family structure (full / mild / zero deficit) maps cleanly. **The previously-stated claim that "canonical D2Q9 + HBB fixes the deficit" was wrong**: canonical D2Q9 + HBB has a smaller residual than D2Q9 uniform + HBB only because of the 4:1 cardinal:diagonal weight ratio (less diagonal upstream-V mass to lose at the wall), NOT because HBB does anything diagonal-aware. All HBB variants lie in the face_mirror sign-locked family.

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
    boundary_mode='face_mirror',       # standard Neumann; sign-locks to forward crescent at the wall
                                       # or 'face_mirror_iso' to eliminate the diagonal deficit
)
```

Constraints on Moore-8 stencils:
- Isotropic D only (raises `NotImplementedError` for non-zero Dxy in `D_field`)
- Square grids only (raises `NotImplementedError` for dx ≠ dy)

The `face_mirror_iso` boundary mode pulls the row-aligned REAL neighbour into the diagonal ghost slot (e.g., NE at `(i+1, -1)` → ghost = `V[i+1, 0]`, NOT `V[i, 0]`). This injects real upstream-V information into the diagonal channel and eliminates the boundary deficit. For cardinal-only stencils, `face_mirror_iso` degenerates to `face_mirror` exactly. **The LBM structural analog of face_mirror_iso is specular reflection**, NOT bounce-back. (Both achieve zero per-column bias; bounce-back is in the face_mirror family with the forward sign-lock — see §"Three BC families" below.)

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
| `Research/Active/boundary_conduction_speedup/diag_dvdt_decomposition.py` (2026-05-14) | Cases 1-7: dV/dt diagnostic at step 1, AP-first protocols; full V tensor logged to HDF5 |
| `Research/Active/boundary_conduction_speedup/diag_lbm_invcrescent.py` (2026-05-14) | Case 8: LBM cross-engine inverse-crescent test (HBB sign-lock confirmation) |
| `Research/Active/boundary_conduction_speedup/diag_lbm_specular.py` (2026-05-14) | Cases 9-14: LBM 4-way comparison HBB/specular × canonical/uniform_8; cases 13-14 horizontal redirect |
| `Research/Active/boundary_conduction_speedup/plot_dvdt_traces.py` (2026-05-14) | Figures from cases 1-4 |
| `Research/Active/boundary_conduction_speedup/data/case{1..14}_*.h5` (2026-05-14) | HDF5 dumps of full V(t, NX, NY) tensors for each case, per-file attrs |
| `simulation/connectivity_threshold_ablation.py` | Storage-tank R1-R6 ablation (predecessor) |
| `Monodomain/Engine_V5.4/test_boundary_modes.py` | 21 boundary-mode tests including the new stencils + face_mirror_iso |
| `LBM/Engine_V1/tests/test_d2q9_uniform_weights.py` | 8 LBM tests for D2Q9_uniform + cs2 plumbing |

### Implication for cardiac modeling practice

Standard cardiac PDE solvers (5-point cardinal + face-centered Neumann) are structurally immune to the connectivity-mediated boundary effect. This is *why* the textbook conclusion "Neumann monodomain has no boundary speedup" holds — it's not that the underlying physics excludes it, it's that the discretization choice (cardinal-only) avoids the asymmetry that 9-point or higher-order isotropic stencils would expose.

For applications that genuinely model cardiac tissue boundaries (no mathematical trick to hide the effect), the right physical setup is bath-coupled bidomain (Kleber et al.) — that produces a real CV speedup at the boundary (~7-13%) via a different mechanism (extracellular shunt, not fewer-neighbour intracellular topology).

The connectivity-mediated effect we just confirmed is **a separate phenomenon** — it's the discrete-topology boundary lag that John's storage-tank reveals. It exists in all 9-point cardiac discretizations but is invisible at 5-point. Whether to consider it "physical" or "numerical" depends on whether you think the discrete fewer-neighbours-at-edge structure is faithful to real tissue (it's a long-running debate in computational EP).

## Three BC families: structural bias hierarchy (2026-05-14)

The 2026-05-14 session corrected an earlier mis-mapping in this document and established a clean three-family taxonomy of boundary treatments based on their structural bias direction. Each family is realized in BOTH the PDE (V5.4 FDM) and LBM (V1) engines via a different bookkeeping but the SAME physical effect.

### The taxonomy

```
   Family                         Sign-lock direction       PDE realization        LBM realization
   ─────────────────────────────────────────────────────────────────────────────────────────────
   Forward crescent (slowdown)    boundary lags interior    face_mirror            HBB (halfway bounce-back)
   Zero bias (transparent wall)   no sign-lock              face_mirror_iso        specular-at-neighbor
   Inverse crescent (speedup)     boundary leads interior   [no PDE analog yet]    same-cell specular ← CLEAN (2026-05-28)
```

> **Updated 2026-05-28**: the inverse-crescent LBM realization is
> **same-cell specular reflection**, NOT horizontal redirect. Horizontal
> redirect (the original 2026-05-14 candidate) gives inverse crescent only
> via a wall pre-charge artifact. See "Clean inverse-crescent BC" section
> below for the corrected rule and full derivation.

### Forward crescent: HBB ≡ face_mirror

**Both are zero-flux Neumann BCs**, just expressed differently:
- face_mirror sets `V_NW_ghost = V_self` → gap = 0 → no diagonal Laplacian contribution at the wall
- HBB reverses populations: `f_SE(C, after) = f_NW(C, before)` → diagonal slot receives C's own pre-stream NW emission, which is local-equilibrium-like (≈ `w_NW · ρ_C`) and carries no upstream V

Different bookkeeping, same Neumann condition at the operator level, same structural deficit: diagonal channels at the wall are starved of upstream-V information, so the boundary cell charges more slowly than interior. This is sign-locked: imposed inverse crescents are eaten within a few columns of propagation (case 7 in monodomain: 2 cols; case 8 in LBM: ~20 cols).

**Weight ratio determines deficit magnitude.** The cardinal:diagonal weight ratio sets how much diagonal mass is being lost, which sets the deficit ratio (boundary charging rate / interior charging rate):

```
   Mapping                                                 deficit ratio    boundary/center
   ──────────────────────────────────────────────────────────────────────────────────────
   moore8_uniform + fm  ↔  D2Q9 uniform_8 (1:1) + HBB      2/3 (full)       0.667
   moore8_iso     + fm  ↔  D2Q9 canonical (4:1) + HBB      5/6 (mild)       0.833
   cardinal4      + fm  ↔  D2Q5 + HBB                      no diagonals     1.000 (no deficit)
```

The 2/3 ratio was verified at the very first diffusion step in case 6 (strict AP-first protocol): boundary cell charges 1.23 mV vs center cell 1.85 mV from V_rest, ratio = 1.23/1.85 = 0.665. **The deficit is local and per-column** — it activates at whichever column is currently charging from rest under face_mirror, NOT inherited LAT-shift from upstream.

### Operator-level confirmation: dV/dt at step 1 (cases 1-6, 2026-05-14)

The "operator-level state-independent" claim from the earlier discrete-lattice analysis was verified directly by measuring dV/dt at step 1 from a clean V_rest initial condition. Earlier (case 1, natural propagation): boundary dV/dt = 92.0 mV/ms, center dV/dt = 138.0 mV/ms → ratio 0.667 ≈ 2/3 EXACTLY at step 1, with no accumulation needed.

Case 6 (strict AP-first protocol) eliminates upstream-LAT-inheritance as an alternative explanation: cols 1-3 are clamped to +30 mV using ionic-only updates for the first 10 steps (no diffusion anywhere during the sync window). Then diffusion is released, and the very first diffusion step at col 4 (which sits at V_rest exactly because no diffusion has happened yet) shows the same 0.665 boundary/center charging-rate ratio. **The deficit is generated fresh at whichever column is currently charging from rest under face_mirror; it is NOT an inherited LAT shift from upstream history.**

Cases 1-4 also confirmed: face_mirror_iso eliminates the deviation to floating-point precision (V[j=0] == V[j=mid] to 0.0e+00) on every step in both diffusion-only and TTP06 paths. The fix is also instantaneous and operator-level.

| Case | Setup | Crescent | Source of imbalance |
|---|---|---|---|
| 3 | fm + ttp06 natural | +486 µs LAT (col 2+) | every column charging from rest |
| 4 | fmi + ttp06 natural | 0 µs LAT everywhere | no source imbalance under face_mirror_iso |
| 5 | fm + sync cols 1-3 (Strang in window) | LAT at col 4 (contaminated by diff-in-window) | per-column at col 4 + small pre-release seed |
| 6 | fm + sync cols 1-3 (ionic-only in window) | LAT at col 4 (cleanly) | per-column at col 4, isolated from any upstream history |

### Sign-lock confirmation: imposed inverse crescent (cases 7-8, 2026-05-14)

The forward sign-lock claim was tested operationally by imposing an INVERSE crescent (boundaries clamped one column ahead of interior) and watching what face_mirror / HBB does to it.

**Case 7 (monodomain, moore8_uniform + face_mirror).** Sync window clamps cols 1-3 at +30 mV on all rows AND col 4 at +30 mV only on boundary rows (j=0, j=NY-1). Ionic-only stepping for 10 steps to lock in the asymmetric AP plateau, then release.

| col | LAT[bdry] | LAT[ctr] | bdry − ctr | interpretation |
|---|---|---|---|---|
|  4 | 0.0000 | 0.5491 | −549 µs | imposed inverse crescent |
|  5 | 0.9579 | 1.0203 | −62 µs  | lead almost gone |
|  6 | 1.4999 | 1.4780 | +22 µs  | FLIPPED to forward |
| 10 | 3.4922 | 3.2921 | +200 µs | growing forward |
| 40 | 17.3955 | 16.8272 | +568 µs | asymptotic forward |

face_mirror eats a 1-column lead in 2 columns and grows its own forward crescent to +568 µs by col 40.

**Case 8 (LBM V1, D2Q9 canonical + HBB).** Same inverse-crescent setup, cross-engine. Lead eaten in ~20 cols (longer because the 5/6 mild deficit acts more gradually), then flips and grows forward to +73 µs by col 40. The 8× weaker per-column slope is exactly what the 5/6 (LBM canonical) vs 2/3 (PDE uniform) weight ratio predicts.

**Conclusion.** The face_mirror / HBB family is sign-locked to forward crescent in BOTH engines. No initial condition (including a fully imposed inverse crescent) can survive long-range propagation. The asymmetry is baked into the operator, not into transients.

### Zero bias: specular reflection ≡ face_mirror_iso

**Both inject real upstream-V into the diagonal channel at the wall**:
- face_mirror_iso: `V_NW_ghost = V_W` (the row-aligned real neighbour) → diagonal channel sees real upstream gradient
- Specular reflection: diagonal populations cross the wall to the *adjacent cell's row-aligned slot* — e.g., NE at top redirects to SE at east-neighbour cell, carrying upstream-V from upstream-boundary's emission

Empirical confirmation (cases 9-12, natural propagation in LBM D2Q9):

```
  col       can+HBB    can+SPEC     uni+HBB    uni+SPEC
  ──────────────────────────────────────────────────────
   3       −43 µs     −46 µs       −196 µs    −178 µs   (stim transient — identical between HBB/SPEC at col 3)
  10       +27        −24          −34        −120      (HBB flips forward; SPEC still inverse)
  20       +63        −15          +67        −77       (HBB grows forward)
  38       +96        −7           +148       −35       (HBB sign-locked forward; SPEC decaying to 0)
```

Specular has zero per-column structural bias. Initial inverse residuals from the stim transient decay slowly toward zero; the bias does NOT asymptote to a fixed inverse value. Specular eliminates HBB's forward sign-lock without introducing a counter-bias of its own — it's a transparent wall.

**face_mirror_iso is a PDE-only construction** in the same family (zero deficit). It works by deliberately breaking strict Neumann symmetry on the diagonals (the ghost is NOT a mirror of the boundary cell — it's pulled from the row-aligned real neighbour). Specular reflection achieves the same effect on the LBM side via a different population-bookkeeping route.

### Inverse crescent (novel): horizontal redirect — SUPERSEDED 2026-05-28

> **WARNING (2026-05-28): this "inverse crescent" is an ARTIFACT.**
> Horizontal redirect produces inverse crescent only by spontaneously
> pre-depolarizing the wall row (a +18 mV standing offset that appears with
> NO stimulus — see "Edge-row depolarization diagnosis" above and "Clean
> inverse-crescent BC" below). The −1146 µs "speedup" is an early LAT
> threshold crossing on the pre-charged ramp; it vanishes under the
> precharge-immune dV/dt metric. The genuine clean inverse-crescent BC is
> same-cell specular reflection. Treat this section as historical.

A NEW BC family introduced 2026-05-14 (cases 13-14, user's invention). At the top/bottom walls, diagonal populations are redirected into the adjacent neighbour's **pure-cardinal eastward (or westward) slot** instead of specular's row-aligned diagonal slot:

```
   Standard HBB:        f_5 (NE) at top cell → f_7 (SW) at SAME cell (reversed in place)
   Specular:            f_5 (NE) at top cell → f_8 (SE) at east-neighbour cell (y-flip, x-preserved)
   Horizontal redirect: f_5 (NE) at top cell → f_1 (E)  at east-neighbour cell (y-momentum → x-momentum)
```

Implementation: HBB everywhere first, then zero out f_7/f_8 at top non-corner cells and f_5/f_6 at bottom non-corner cells, then ADD the pre-stream diagonal masses to neighbours' cardinal slots.

**Result: sustained inverse crescent that grows monotonically with distance** (cases 13-14, natural propagation, NX=41):

| col | canonical (4:1) | uniform_8 (1:1) |
|---|---|---|
| 3   | −220 µs  | −641 µs  |
| 10  | −517 µs  | −1496 µs |
| 20  | −839 µs  | −2289 µs |
| 38  | −1146 µs | −3106 µs |

Wall channel propagates faster than bulk by sustained ~30-80 µs per column. This is the OPPOSITE sign of HBB and gives a real boundary speedup — the long-sought BC analog of Kleber-style boundary acceleration via a purely numerical / lattice-side mechanism.

**Caveat (open issue for next session).** Mass-conservation in the horizontal-redirect implementation is imperfect:
- V_sum at t_end is 9% (canonical) to 18% (uniform_8) HIGHER than HBB baseline
- V_max climbs above the expected +15 mV plateau (to +18.75 / +19.68 mV)

~~Likely mass leak at corner cells~~ — **disproven 2026-05-28** (see below).

#### Edge-row depolarization diagnosis (2026-05-28)

Ran a 4-step diagnostic (`PLAN.md`) on the horizontal-redirect BC:

- **Mass IS conserved.** Diffusion-only runs of HBB, buggy horizontal,
  and a corner-aware variant (`--bc horizontal_fixed`) all show
  `V_sum(t=25 ms) = V_sum(0)` to floating-point precision. No leak.
- **The 9% V_sum excess vs HBB in TTP06 runs is wave-propagation, not
  leak.** The wall channel propagates the AP wave faster along the wall
  row → more cells reach the TTP06 plateau by t=25 ms → higher Σ V via
  the source term. Mass is conserved per LBM step; "more depolarization"
  is the BC working as intended.
- **The sub-edge dip at j=1 (V ≈ −94 mV) is a BC-MECHANICAL artifact,
  not ionic.** It persists under pure diffusion (R=0): V_min at (2, 1)
  is −94.8 mV (buggy horizontal) / −95.2 mV (fixed) / −89.4 mV (HBB).
  The horizontal redirect's lateral mass shift starves the sub-edge row
  during wavefront passage; ionic hyperpolarization is not required.
- **Corner-aware variant barely changes anything.** Donor-excludes-corner
  + orphan-HBB-self-bounce reduces LAT magnitude by ~12 % (−1130 → −1002
  µs at col 38) and shifts V_sum by 0.001 mV/cell. The wall channel and
  sub-edge dip patterns are visually indistinguishable from the original.

**Updated mass-conserving column in the summary table** ↓: yes for all
three families.

Diagnostic figure: `figures/horizontal_synthesis.png`. Scripts:
`diag_horizontal_{mass,vyprofile,synthesis}.py`. Code:
`diag_lbm_specular.py::apply_horizontal_fixed_top_bottom_d2q9`.

### Summary table (CORRECTED 2026-05-28)

> The horizontal-redirect "inverse crescent" is an ARTIFACT (wall pre-charge,
> see above + the dedicated clean-BC section below). The genuine clean
> inverse-crescent BC is **same-cell specular reflection**.

| BC family | LBM form | crescent | mass-conserving? | rest no-op? | Notes |
|---|---|---|---|---|---|
| Forward (slowdown) | HBB: NE→SW same cell | forward | yes | YES | Standard no-flux wall; reverses both x and y. |
| Zero bias | specular-at-neighbor: NE→SE @ i+1 | none | yes | YES | Transparent wall; y-flip displaced one cell cancels bias. |
| **Inverse (CLEAN)** | **same-cell specular: NE→SE @ same cell** | **inverse** | **yes** | **YES** | **Genuine novel result; flip wall-normal, keep tangential, same cell. Zero standing artifact, inverse at all thresholds + dV/dt.** |
| Inverse (ARTIFACT) | horizontal redirect: diagonal→cardinal | inverse | yes (mass) | **NO (+18 mV)** | Pre-charges wall with no stim. Discard. |

### Implications

1. **The earlier KNOWLEDGE claim that "face_mirror_iso (PDE) ≡ bounce-back (LBM)" was wrong.** The correct mapping has bounce-back on the face_mirror side (NOT the face_mirror_iso side). Sections of this document that previously stated "LBM bounce-back fixes the deficit" have been corrected: it doesn't — it inherits the same forward sign-lock as face_mirror, just with a smaller magnitude due to D2Q9's 4:1 cardinal:diagonal weight ratio.

2. **All three engines (storage tank, monodomain, LBM) live in the face_mirror family by default.** Cardinal-4 / D2Q5 escape the deficit by having no diagonals at all. Moore-8 + face_mirror and D2Q9 + HBB express the same operator-level pathology.

3. **A genuine clean inverse-crescent BC exists** — same-cell specular reflection (2026-05-28). The horizontal-redirect route was an artifact, but the *goal* (single-field boundary speedup) is achievable cleanly. See the dedicated section below.

4. **The bridge-claim narrative still holds, refined.** Three engines, same connectivity-mediated boundary deficit when using the face_mirror-family Neumann condition. The structure is the operator (face_mirror), not the model class.

## Clean inverse-crescent BC: same-cell specular reflection (2026-05-28)

> **Headline.** There IS a discrete LBM wall boundary condition that produces
> a genuine inverse crescent (boundary conduction FASTER than interior) while
> being exactly mass-conserving AND a perfect no-op on a uniform field. It is
> **same-cell specular reflection**: at the wall, flip the wall-normal (y)
> velocity component, keep the tangential (x) component, deposit at the SAME
> cell. In D2Q9 slots, top wall: f_5 (NE) → f_8 (SE) same cell, f_6 (NW) →
> f_7 (SW) same cell (bottom y-mirrored). User-confirmed visually (developing
> inverse chevron, no wall pre-glow). Video:
> `figures/video_bc_specular_samecell.mp4`.

### The journey — part-by-part realization

Each sub-finding was a necessary rung from "line speedup looks real" to the
clean rule:

**1. Line speedup is real (visual).** Under horizontal redirect, the wall
rows fire ahead of the interior → inverse-crescent (V-shaped) wavefront.
Looked like a genuine novel boundary speedup.

**2. Diagonal→horizontal is the inverse-crescent mechanism (big insight #1).**
Horizontal redirect converts a slow diagonal population (weight 1/36, 45°)
into a fast pure-cardinal population (weight 1/9, straight along the wall) —
a "wall highway." The weight-CLASS change (diagonal → cardinal) is what
carries the wave faster at the boundary.

**3. The wall depolarizes uniformly in x, in 2 steps (anomaly).** Hi-res
traces (`diag_horizontal_wall_propagation.py`): cols 3-35 all at −82 mV at
t=0.04 ms — far faster than any wave. Generated LOCALLY per cell, not
transported.

**4. The depolarization is INTRINSIC — exists with zero stimulus (big
insight #2).** No-stim test (uniform V_rest IC, diffusion only): HBB and
specular give Δwall = exactly 0; horizontal gives +18.43 mV wall / −1.9 sub
/ −1.9 interior — a standing 3-layer structure from a uniform field. Mass
exactly conserved (redistributed). ⇒ the "speedup" is a wall pre-charge that
crosses the −40 mV LAT threshold early. A MEASUREMENT artifact, not faster
physics.

**5. Root cause = weight-class mismatch (big insight #3).** D2Q9 weights:
cardinal 1/9, diagonal 1/36, ratio exactly 4. feq_i = w_i·V. HBB and
specular map diagonal→diagonal and cardinal→cardinal (weight-matched), so at
rest they map feq→feq exactly — structural no-ops. Horizontal puts a 1/36·V
diagonal into a 1/9·V cardinal slot — 4× mismatch; the leftover is the
artifact. Single-step instrumentation: the redirect conserves V at its own
step, but leaves a non-equilibrium distribution (cardinals over-full,
diagonals empty) that the NEXT collision retargets, pumping the offset. An
omega-sweep nailed it: artifact = 0 at omega = 1 (instant equilibration),
growing with |omega − 1|.

**6. Mass-accumulation ⇒ REFLECTION, not transport (big insight #4).** A
scalar weight-normalization can't fix it: mass conservation pins the transfer
scale to exactly 1 (any k≠1 leaks/blows up — tested). Any rule that keeps
mass ON the wall (stays in x, or re-hits in +y) traps equilibrium mass →
standing artifact. The only artifact-free option is a rule where the mass
LEAVES the wall into the interior (−y) — a genuine REFLECTION. HBB and
specular are clean precisely because they reflect. So: to get inverse
crescent without accumulation, stay in the reflection class but change which
velocity component is reversed.

**7. Final — exhaustive enumeration finds the clean rule.** Enumerated all
27 symmetric mass-conserving rules (f_5 → every slot × {west, same, east};
f_6 x-mirrored; bottom y-mirrored). `diag_enumerate_walls.py` →
`data/wall_enumeration.txt`. They partition CLEANLY by destination y-sign:

```
   destination y-sign    artifact (Δwall, no stim)   crescent
   ────────────────────────────────────────────────────────────────
   stays on wall (x)        +18 mV  (ALL 9 rules)     inverse  (trap)
   re-hits wall (+y)        +18 to +24 (ALL 9)        inverse  (trap)
   leaves into bulk (−y)    ~0      (ALL 9)           forward OR inverse
                                                       ├ x-REVERSED  (NE→SW) = HBB       → forward
                                                       ├ x-PRESERVED (NE→SE) = specular  → INVERSE ✓
                                                       └ NE→S (pure down)               → mild inverse
```

The reflection class (mass leaves, −y) is the only artifact-free class; the
crescent SIGN within it is set by whether the tangential (x) component is
reversed (forward, HBB) or preserved (inverse, same-cell specular).

### The clean rule (D2Q9, top wall, non-corner cells)

```
   f_5 (NE, +x+y)  →  f_8 (SE, +x−y)  at the SAME cell   (flip y, keep x)
   f_6 (NW, −x+y)  →  f_7 (SW, −x−y)  at the SAME cell
   f_3 (N)         →  f_4 (S)          HBB cardinal (unchanged)
   bottom wall: y-mirror (f_8→f_5, f_7→f_6 at same cell)
```

Contrast with the two known reflection rules:
- **HBB**: f_5 → f_7 (NE→SW) — reverses BOTH x and y. Kills forward drive → forward crescent (slowdown).
- **specular-at-neighbor** (old "specular"): f_5 → f_8 at cell i+1 — flips y but displaces one cell east → cancels → zero bias.
- **specular-same-cell** (NEW): f_5 → f_8 at the same cell — flips y, keeps x, no displacement → inverse crescent (speedup).

### Verification

```
   NO-STIM, diffusion only, 50 ms, V_init = V_rest everywhere:
     Δwall = −0.000000   Δsub = −0.000000   Δinterior = −0.000000
     mass drift = −1.6e-10   (machine precision; exact no-op + exact conservation)
     top wall = bottom wall  (symmetric)

   WITH-STIM (col-0 line stim, TTP06 EPI, 25 ms), LAT bdry−ctr at col 38:
     threshold −40 mV:  −313 µs   ┐
     threshold −20 mV:  −317 µs   │ inverse at EVERY threshold
     threshold   0 mV:  −318 µs   │ (not a threshold-crossing artifact)
     threshold +10 mV:  −329 µs   ┘
     wall precharge @ t=1 ms:  −85.232 mV  =  V_rest  (NO pre-charge)
     max(dV/dt) timing:        −300 µs   (precharge-IMMUNE → real upstroke speedup)
```

The precharge-immune dV/dt-peak metric is the clincher: −300 µs inverse
means the regenerative upstroke initiates earlier at the wall — a real
boundary speedup, not an early threshold crossing on a charged ramp. (The
horizontal-redirect "speedup" vanishes under the dV/dt metric — that is the
qualitative difference.)

### Why same-cell specular gives speedup (mechanism)

A diagonal NE = (+x, +y) carries momentum forward (+x) and toward-wall (+y).
At the wall it can't go +y; the three reflection choices differ in the +x part:
- HBB reverses it (→ −x): forward drive destroyed → wall receives less push
  than interior → slower → forward crescent.
- Same-cell specular keeps it (+x preserved, y flipped to −y): forward drive
  survives the bounce and stays at the wall cell → wall keeps MORE forward
  push than interior → faster → inverse crescent.

Diagonal → diagonal is weight-matched (both 1/36), so the map sends feq → feq
exactly at rest: no standing offset, exact mass conservation.

### Implications

1. **The (forward ↔ inverse) BC axis is real and clean.** Both endpoints are
   rest-neutral reflections: HBB (forward) and same-cell specular (inverse),
   with neighbor-displaced specular at zero in between. A weighted blend of
   HBB and same-cell specular gives a tunable, artifact-free crescent
   magnitude — the correct basis for the (α, β, γ) simplex / tissue-fitting
   program, replacing the artifact-laden horizontal vertex.

2. **Single-field boundary speedup IS achievable.** Earlier reasoning held
   that Kleber speedup is intrinsically bidomain and a single-field wall can
   only slow down. False at the discrete-operator level: a
   tangential-momentum-preserving reflection produces speedup in a single
   field. (Whether it maps to a *physical* tissue boundary is a separate,
   open question.)

3. **The "trap ⟺ inverse ⟺ artifact" coupling hypothesized mid-session was
   WRONG.** Inverse crescent does NOT require trapping mass on the wall.
   Trapping (horizontal/specular_up) is one route and it drags the artifact
   along; reflection (same-cell specular) is a clean route to the same sign.

### Reproduce

- Enumeration: `diag_enumerate_walls.py` → `data/wall_enumeration.txt`
  (winning rule is `f5->SE@0`).
- BC modes added to `diag_lbm_specular.py` this session: `horizontal_fixed`,
  `horizontal_donut`, `horizontal_gradient`, `horizontal_wnorm`,
  `specular_up` (all rejected; kept for reference).
- Diagnostics: `diag_horizontal_{wallrow,longrun,inward_widestim,wall_propagation,anisotropic,resolve}.py`.
- Video: `figures/video_bc_specular_samecell.mp4`.

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
requires either cardinal-4 (no diagonals at all) OR a custom boundary
treatment (face_mirror_iso on the PDE side, specular reflection on the
LBM side) that restores real upstream-V information to the diagonal
ghost / population slots at the wall.

### LBM connection (corrected 2026-05-14)

LBM D2Q9 weights are 4/9 (rest), 1/9 (cardinals), 1/36 (diagonals).
Cardinal:diagonal ratio = 4:1, identical to Patra-Kałuża isotropic 9-point.
**LBM halfway bounce-back (HBB) is structurally equivalent to face_mirror**
(not face_mirror_iso). Both kill upstream-V contribution from diagonal
channels at the wall:
- face_mirror: `V_NW_ghost = V_self` → gap = 0 → no diagonal Laplacian contribution
- HBB: `f_SE(C, after) = f_NW(C, before)` → diagonal slot gets C's own pre-stream NW emission, which is local equilibrium (≈ `w_NW · ρ_C`) and carries no upstream V information

Different bookkeeping, same physical Neumann boundary condition, same structural deficit. The LBM bounce-back family inherits the forward sign-lock from face_mirror, and the magnitude of the deficit follows the cardinal:diagonal weight ratio (canonical 4:1 → 5/6 mild; uniform_8 1:1 → 2/3 full). The LBM analog of face_mirror_iso (zero deficit, no sign-lock) is **specular reflection**, in which diagonal mass crosses the wall to the adjacent cell's row-aligned slot and carries real upstream-V information from the upstream-boundary's diagonal emission. See §"Three BC families" for the full mapping.

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
| HBB ↔ face_mirror equivalence (2026-05-14) | HBB is in the **face_mirror family**, NOT the face_mirror_iso family | Both kill diagonal upstream-V at the wall; weight ratio determines magnitude. Corrects earlier mis-statement that "canonical D2Q9 + HBB fixes the deficit". |
| LBM analog of face_mirror_iso (2026-05-14) | Specular reflection | Diagonal mass crosses to neighbour cell's row-aligned slot, carrying real upstream-V. Zero structural per-column bias. Confirmed cases 9-12. |
| Inverse-crescent BC (2026-05-14) | Horizontal redirect (novel, LBM-only so far) | Diagonal y-momentum → cardinal x-momentum at wall; sustained inverse crescent. Has mass leak at corners; needs correction. |
| Operator-level deficit locality (2026-05-14) | LOCAL per-column, not inherited from upstream | Case 6 (strict AP-first): first diffusion step at col 4 from V_rest gives boundary/center ratio 0.665 ≈ 2/3 exactly, with cols 1-3 frozen y-uniform. Deficit activates fresh wherever a column charges from rest. |

## Open Questions

- What is the convergence rate with mesh refinement? (Is it O(dx) or O(dx^2)?)
- How does the speedup interact with wavefront curvature at obstacle corners? (Curvature speedup is a separate geometric effect)
- Does the speedup magnitude change with anisotropic conductivity tensors (fiber orientation at boundary)?
- At what tissue thickness does the boundary layer span the entire preparation (transitioning from surface effect to full-thickness effect)?
- Does the EMI model (cell-resolved, no homogenization) reproduce the same speedup magnitude?
- Is there a mass-conserving PDE analog of the horizontal-redirect BC? (Currently LBM-only with a corner mass leak.)
- Why does V_sum climb 9-18% above HBB baseline under horizontal redirect — is the leak at corners, or is the mechanism more deeply non-mass-conserving?

## Connections
- **Engines**: Bidomain V1 (primary validation), LBM V1 (secondary confirmation)
- **Related research**: scar_bc_validity (Q6 -- Neumann not Dirichlet at scar), lbm_cardiac (Q4 -- LBM can capture the effect with D2Q9)
- **Pipelines**: Triangle merger experiments (bidomain vs monodomain CV comparison on realistic geometries)
