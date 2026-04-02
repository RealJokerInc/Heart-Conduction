# Bidomain Parabolic-Parabolic Formulation — Knowledge File

> This file is a running synthesis. Updated as findings accumulate.
> When the question is complete, a copy is promoted to `Research/Knowledge/`.

## Current Understanding

### The problem: instantaneous extracellular propagation

The standard bidomain has two kinds of "instantaneous" behavior:

1. **Global spatial adjustment** (the actual problem): When Vm changes at point A, the elliptic equation forces φ_e to adjust at ALL points simultaneously. The Green's function of the Laplacian has global support. At tissue-bath boundaries, this creates triangular wavefront artifacts — sharp, linear transitions instead of the physically expected smooth shapes.

2. **Local inter-domain coupling** (physically correct): At each point x, current leaving intracellular = current entering extracellular, same instant. This is conservation of charge. The membrane capacitance Cm·∂Vm/∂t already provides the physical time delay. No continuum model can or should "fix" this.

The triangular artifact observed in boundary speedup research is caused by problem #1.

### Terminology trap: "PP bidomain" ≠ what we initially assumed

The literature uses "parabolic-parabolic bidomain" to mean the (φ_i, φ_e) variable form of the standard bidomain — two coupled reaction-diffusion equations sharing the SAME time derivative dVm/dt:

```
∇·(σ_i ∇φ_i) = β·(Cm·∂Vm/∂t + Iion)
∇·(σ_e ∇φ_e) = -β·(Cm·∂Vm/∂t + Iion)
```

This is **degenerate parabolic** — add the two equations and dVm/dt cancels:

```
∇·(σ_i ∇φ_i) + ∇·(σ_e ∇φ_e) = 0    ← elliptic constraint, hidden inside
```

The PP and PE formulations are mathematically equivalent. Neither φ_i nor φ_e has its own independent time derivative — only their difference Vm = φ_i - φ_e evolves in time. The degeneracy IS the physics (charge conservation).

**There is no standard "ε·∂φ_e/∂t" regularization used as a computational method.** Bendahmane & Karlsen 2006 use it as a mathematical proof technique only (ε→0 recovers the true bidomain).

### Two approaches to finite extracellular propagation

#### Approach A: ε-Regularization (ad hoc, not in literature as computational method)

Add ε·∂φ_e/∂t to the elliptic equation, making it parabolic:

```
ε·∂φ_e/∂t = ∇·((σ_i+σ_e)∇φ_e) + ∇·(σ_i∇Vm)
```

- No physical justification for ε
- Trivial to implement: spectral denominator changes from `eigenvals` to `ε/dt + eigenvals`
- Useful as a **numerical experiment** to test whether the artifact is caused by the elliptic character
- No null-space pinning needed for ε > 0 (shift regularizes the singularity)

#### Approach B: Cattaneo Hyperbolic Bidomain (Rossi & Griffith 2017)

The ONLY published formulation with finite propagation speed. Does not add a term to the PDE — instead modifies the **constitutive law** (Ohm's law → Cattaneo relation).

## The Cattaneo Derivation (from first principles)

The standard bidomain derivation has three steps. Cattaneo modifies only step 2.

### Step 1 — Conservation of current (universal, never changes)

Current leaving one domain = current entering the other, mediated by the membrane:

```
∇·J_i = -β·Im        (intracellular current divergence = membrane current out)
∇·J_e = +β·Im        (extracellular current divergence = membrane current in)

where Im = Cm·∂Vm/∂t + Iion    (membrane current = capacitive + ionic)
```

These are conservation laws. They hold regardless of what constitutive relation we choose.

### Step 2 — Constitutive law (THIS is the only thing that changes)

**Standard (Ohm's law):** Current is instantaneously proportional to voltage gradient.

```
J_i = -σ_i ∇V_i
J_e = -σ_e ∇V_e
```

This is an algebraic relation — J has no dynamics of its own. Substituting into Step 1 **eliminates J entirely**, producing the standard bidomain where current density never appears in the final equations.

**Cattaneo:** Current *relaxes toward* the Ohm's law value with time constant τ.

```
τ_i·∂J_i/∂t + J_i = -σ_i ∇V_i
τ_e·∂J_e/∂t + J_e = -σ_e ∇V_e
```

Now J has its own time derivative — it cannot be simply substituted away. Current no longer responds instantaneously to voltage gradients; it has inertia.

### Step 3 — Eliminate J to get equations in V only

With Ohm's law, substitution is direct (J = -σ∇V into ∇·J = ±β·Im). With Cattaneo, J has dynamics, so eliminating it requires differentiating the conservation equations in time and substituting. The chain rule produces second-order time derivatives and ∂Iion/∂t terms.

**The resulting hyperbolic bidomain** (Rossi & Griffith eqs 13-14):

```
Vm equation:
  τ_i·Cm·∂²Vm/∂t² + Cm·∂Vm/∂t - ∇·(D_i∇Vm) - ∇·(D_i∇φ_e) = -Iion - τ_i·∂Iion/∂t

φ_e equation:
  (τ_e-τ_i)·Cm·∂²Vm/∂t² + ∇·(D_i∇Vm) + ∇·((D_i+D_e)∇φ_e) = (τ_i-τ_e)·∂Iion/∂t
```

Everything unfamiliar (∂²V/∂t², ∂Iion/∂t) is the fingerprint of eliminating J from a system where J has its own dynamics.

### Critical subtlety: φ_e equation depends on (τ_e - τ_i)

The coefficient of ∂²Vm/∂t² in the φ_e equation is **(τ_e - τ_i)**.

- If **τ_i = τ_e**: the coefficient is zero → φ_e equation is **STILL ELLIPTIC**. Only the Vm equation gets the wave term.
- If **τ_i ≠ τ_e**: the φ_e equation becomes hyperbolic with finite propagation.

For our boundary artifact problem, we specifically need τ_e ≠ τ_i to make φ_e propagate finitely. The monodomain reduction (which assumes τ_i = τ_e = τ) hides this issue entirely.

### The monodomain reduction (equal anisotropy ratios, τ_i = τ_e = τ)

```
τ·Cm·∂²V/∂t² + Cm·∂V/∂t - ∇·(D∇V) = -Iion - τ·∂Iion/∂t
```

This is the **telegraph equation** — a wave equation with damping. Solved as a first-order system with Q = ∂V/∂t:

```
∂V/∂t = Q
τ·Cm·∂Q/∂t + Cm·Q - ∇·(D∇V) = -Iion - τ·∂Iion/∂t
```

### Physical origin of τ

```
τ = (L_i + L_e) / (R_i + R_e)
```

where L = axial inductance, R = axial resistance in the cable model. The inductance arises from the 3D structure of cells — self-inductance of cell membranes. Experimentally, embryonic heart cell membranes exhibit impedance resonance at ~1 Hz, suggesting real inductance.

**No experimental τ values exist for cardiac tissue.** Rossi & Griffith used τ = 0.4 ms (calibrated to match parabolic CV for Fenton-Karma ionic model). Linear analysis says inductances are negligibly small, but nonlinear dynamics show they can matter.

## Which approach fixes what

| Aspect | PE (current) | ε-Regularization | Cattaneo (τ_e ≠ τ_i) |
|--------|-------------|------------------|----------------------|
| φ_e spatial propagation | Infinite (elliptic) | Finite (parabolic) | Finite (hyperbolic) |
| Local inter-domain coupling | Instant (correct) | Instant (correct) | Instant (correct) |
| Triangular boundary artifact | YES (the problem) | Should fix | Should fix |
| Physical justification | Quasi-static (valid) | None (numerical experiment) | Tissue inductance (weak) |
| Implementation effort | Current engine | ~5 lines (spectral shift) | ~100+ lines (2nd-order time) |
| Null-space pinning | Needed (Neumann) | Not needed (ε > 0) | Not needed (τ > 0) |

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| "PP bidomain" is a misnomer | Rename to "finite extracellular propagation" | Literature PP = same physics as PE, just different variables |
| Three-phase strategy | ε experiment → Cattaneo → comparison | Cheapest test first, then physically justified implementation |
| Why the artifact might be correct | Quasi-static assumption valid by 10 orders of magnitude | Need to consider that the triangle IS the correct physics |

## Open Questions

- Does the triangular artifact persist with ε-regularization? (Phase 1 experiment)
- What ε / τ values produce observable changes in wavefront shape?
- Does the Kleber CV *ratio* change, or just the wavefront geometry?
- For Cattaneo: what (τ_i, τ_e) combination is needed to make φ_e non-elliptic? τ_e ≠ τ_i is required.
- Is the quasi-static assumption actually wrong at boundaries? Or is the triangle the correct physics?
- Rossi & Griffith found small τ INCREASES CV — how does this interact with the Kleber boundary speedup?

## Rossi & Griffith 2017 Key Findings

- **CV increases** with small-to-moderate τ (contradicts linear analysis)
- Spiral waves form after 2 stimuli (hyperbolic) vs 3 (parabolic) — qualitatively different arrhythmia inducibility
- τ ≈ 0.4 ms produces effects on the same timescale as ionic dynamics (dt = 0.01–0.05 ms)
- **NOT tested at tissue-bath boundaries** — our niche
- Used IMEX-RK time discretization, first-order system (V, Q) where Q = ∂V/∂t

## Literature

| Paper | Year | Key Insight |
|-------|------|-------------|
| Rossi & Griffith, Chaos 27:093926 | 2017 | Cattaneo hyperbolic bidomain — the ONLY published finite-propagation formulation. τ = L/R. |
| Bourgault, Coudiere & Pierre | 2009 | PP well-posedness via bidomain operator. PP = (φ_i, φ_e) form, same physics as PE. |
| Pavarino & Scacchi, SIAM J Sci Comp | 2011 | PP vs PE preconditioners. PP block system is harder (not scalable); PE is preferred. |
| Bendahmane & Karlsen | 2006 | ε-regularization as PROOF TECHNIQUE only (ε→0). Not a computational method. |
| Colli Franzone & Savaré | 2002 | First well-posedness proof for bidomain with FitzHugh-Nagumo. |
| Potse et al. | 2006 | Where monodomain fails — boundary effects are the main failure mode. |
| Jaeger & Tveito | 2022 | EMI model — cell-resolved, finite coupling via membrane RC, 300-3000x cost. |
| Jaeger & Tveito, npj Sys Bio | 2023 | KNM — cell-based, 1% of bidomain cost, discrete gap junctions. |
| Corre & Belmiloudi | 2016 | LBM dual-lattice bidomain. Without time-delay extension, same elliptic φ_e. |

## LBM–Cattaneo Correspondence: Why Hyperbolic Bidomain Makes LBM Natural

### The problem with standard bidomain + LBM

LBM is a time-marching method: one collision-streaming cycle = one physical timestep. This is natural for parabolic equations (∂u/∂t = D∇²u + source). The elliptic φ_e equation has no ∂φ_e/∂t, so solving it with LBM requires **pseudo-time iteration** — many LBM sub-cycles per physical timestep to converge to steady state. This destroys LBM's core advantage (one step per timestep, massively parallel, no linear solve).

### The deep connection: LBM already implements Cattaneo

The Chapman-Enskog expansion of the BGK collision-streaming process does NOT recover Ohm's law (J = -σ∇V) at leading order. At the Navier-Stokes level (second-order Chapman-Enskog), it recovers:

```
τ_LBM · ∂J/∂t + J = -D∇u
```

This IS the Cattaneo relation. LBM **naturally implements Cattaneo dynamics as its mesoscopic physics**. The parabolic (Ohm's law) behavior emerges only in the long-time/large-scale limit where τ_LBM · ∂J/∂t becomes negligible. The Cattaneo correction is usually treated as a "numerical artifact" that practitioners try to minimize by choosing τ_LBM close to 0.5.

**For the hyperbolic bidomain, this "artifact" becomes the desired physics.**

### The natural LBM hyperbolic bidomain

Work in (φ_i, φ_e) variables. Two coupled lattices, each with its own distribution functions and relaxation parameter:

```
φ_i lattice:  collision-streaming with τ_LBM_i,  source = -β·Im
φ_e lattice:  collision-streaming with τ_LBM_e,  source = +β·Im

where Im = Cm·∂Vm/∂t + Iion    (membrane current = coupling between lattices)
```

The LBM relaxation parameter maps directly to the physical Cattaneo relaxation time:

```
τ_Cattaneo ≈ (τ_LBM - 0.5) · dt
```

Both lattices march forward **one physical timestep per collision-streaming cycle**. No pseudo-time iteration. No elliptic solve. No linear algebra. Just two parallel LBM evolutions coupled through membrane current as an explicit source term.

### Why this works: the constitutive law match

| Level | Standard bidomain | Hyperbolic bidomain |
|-------|------------------|---------------------|
| Constitutive law | Ohm: J = -σ∇V (algebraic) | Cattaneo: τ·∂J/∂t + J = -σ∇V (dynamic) |
| LBM mesoscopic physics | Cattaneo (mismatch! must minimize) | Cattaneo (**exact match**) |
| φ_e equation character | Elliptic (no ∂/∂t) | Hyperbolic (has ∂/∂t via Cattaneo) |
| LBM suitability for φ_e | Unnatural (pseudo-time iteration) | Natural (one step per timestep) |

The mesoscopic physics of LBM (distribution functions relaxing toward equilibrium with time constant τ_LBM) maps directly onto the mesoscopic physics of Cattaneo (current relaxing toward Ohm's law value with time constant τ). This is not a workaround — it is a genuine mathematical correspondence between the LBM kinetic equation and the Cattaneo constitutive law.

### τ_i ≠ τ_e comes for free

Each lattice has its own relaxation parameter. Different τ_LBM_i and τ_LBM_e automatically give different Cattaneo relaxation times. This is exactly the condition needed to make the φ_e equation non-elliptic (the critical subtlety from Rossi & Griffith's formulation where the φ_e equation depends on τ_e - τ_i). In LBM, you get τ_i ≠ τ_e by default simply by having two lattices with different relaxation parameters — which you would naturally have anyway since σ_i ≠ σ_e implies different τ_LBM values.

### Comparison: standard vs hyperbolic bidomain in LBM

```
Standard bidomain + LBM:
  Vm lattice  → natural (parabolic, one step per timestep)
  φ_e         → UNNATURAL (elliptic, needs pseudo-time iteration)
  Result: LBM loses its main advantage; might as well use FDM

Hyperbolic bidomain + LBM:
  φ_i lattice → natural (Cattaneo IS what LBM does)
  φ_e lattice → natural (Cattaneo IS what LBM does)
  Result: two parallel lattices, one step per timestep, no linear solve
  Bonus: fully GPU-parallelizable, no global communication needed
```

### Subtleties and risks

1. **Hidden constraint when τ_i = τ_e**: Adding the two conservation equations gives ∇·(J_i + J_e) = 0 at all times. If τ_i = τ_e, this is still an instantaneous constraint (elliptic ghost). When τ_i ≠ τ_e, the constraint becomes a dynamic equation — the ghost is exorcised. Since LBM naturally gives τ_i ≠ τ_e (different lattice relaxation parameters), this is automatically handled.

2. **Constraint drift**: In continuous equations, ∇·(J_i + J_e) = 0 is exact. In LBM, this is approximate. Error may accumulate over many timesteps. May need periodic projection/correction.

3. **Coupling stability**: Membrane current Im is large during the action potential upstroke. Explicit coupling (source terms) could cause instability. May need operator splitting (half-step ionic → full-step LBM → half-step ionic) or implicit treatment of Im.

4. **Accuracy of τ mapping**: The correspondence τ_Cattaneo ≈ (τ_LBM - 0.5)·dt is approximate (from truncated Chapman-Enskog). Higher-order corrections exist but complicate the mapping.

### Implications for the project

This connection suggests a third implementation path beyond ε-regularization and FDM-Cattaneo:

- **LBM Cattaneo bidomain**: Extend the existing LBM V1 engine (D2Q5/D2Q9, BGK/MRT, 34 tests) to a dual-lattice bidomain. The engine already has the collision, streaming, and boundary infrastructure. Adding a second lattice for φ_e and coupling via membrane current would give a fully parallel, GPU-native hyperbolic bidomain solver with no linear algebra.
- This would be the first LBM implementation of the Cattaneo bidomain in the literature (Corre & Belmiloudi's LBM bidomain uses the standard parabolic formulation, not Cattaneo).

## Rossi & Griffith 2017 Computational Details

- **Spatial discretization**: FEM, P1 (linear) elements, BeatIt/libMesh/PETSc (C++)
- **Ionic models tested**: McKean (piecewise-linear, has analytical solution), Aliev-Panfilov, Fenton-Karma (4 parameter sets), **TTP06** (20-variable), Grandi '11 atrial (57-variable)
- **TTP06 result**: Peak CV enhancement at τ ≈ 0.1–0.2 ms
- **Time stepping**: IMEX-RK (ARS(1,1,1) first-order, H-CN(2,2,2) second-order). dt = 0.0025–0.125 ms
- **Geometries**: 1D cables, 2D sheets (12x12 cm), 3D patient-specific left atrium (~3.5M elements)
- **Cost** (hyperbolic vs parabolic, 2D with TTP06): -3% overall (7% cheaper reaction, 15% more expensive diffusion, but fewer CG iterations)
- **Mesh convergence**: Needs ~25 μm (one cardiomyocyte) for fiber-aligned convergence
- **Software**: Open source — github.com/rossisimone/beatit

## Connections

- **Engines**: Bidomain V1 — add ε-regularization (Phase 1) and Cattaneo (Phase 2) as alternative FDM solver pathways
- **Engines**: LBM V1 — extend to dual-lattice Cattaneo bidomain (Phase 3 candidate, exploiting LBM–Cattaneo correspondence)
- **Related research**: [Boundary conduction speedup](../boundary_conduction_speedup/) — triangular artifact origin, cross-validation target
- **Related research**: [Geometry-induced pacemaking](../geometry_induced_pacemaking/) — boundary accuracy matters here too
- **Related research**: [LBM for cardiac EP](../../Complete/lbm_cardiac/) — LBM V1 foundation (D2Q5/D2Q9, BGK/MRT)
- **Pipelines**: Surrogate pipeline may need to account for modified formulation if it becomes preferred
