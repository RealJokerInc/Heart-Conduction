# Two-DOF Dual-Evolving Bidomain — Mathematical Reformulation

**Goal.** Reformulate the bidomain equations so that both V_i and V_e have their own
first-order time derivatives (heat-equation style), suitable for direct mapping onto two
independent LBM lattices. Identify exactly which mathematical operation in Rossi &
Griffith's derivation collapses the genuinely 2-DOF Cattaneo system into the 1-DOF-with-
memory bidomain they present, and what we have to change to preserve 2-DOF.

---

## 1. What Rossi & Griffith actually start from (honest answer: it was 2-DOF)

The starting point in the paper is **Cattaneo flux dynamics** (eqs 1, 2, p. 3):

```
(R1)   τ_i ∂J_i/∂t + J_i = -σ_i ∇V_i
(R2)   τ_e ∂J_e/∂t + J_e = -σ_e ∇V_e
```

These are **two genuinely independent first-order time evolutions of vector flux
fields**. Each compartment's current has its own relaxation dynamics with its own
relaxation time. No coupling at this stage — this is manifestly 2-DOF.

**This IS the dual-evolving formulation**, sitting in plain sight as eqs 1-2.

Rossi & Griffith do NOT present them as such because they proceed to impose
constraints that collapse the structure. The collapse happens in three substantive
steps, only one of which is a "physical" choice. The other two are mathematical
rearrangements.

---

## 2. The three moves that collapse 2-DOF to 1-DOF-with-memory

### Move #1 (physical). Impose quasi-static charge conservation — **this is eq 5**

Rossi's eq 5 (p. 3): `∇·(J_i + J_e) = 0`.

Stated in the paper as "a quasistatic form of charge conservation." Physically this
asserts that net charge density in the tissue stays at zero — any charge accumulating
in one compartment must be draining from the other. Mathematically it is a **hard
constraint coupling J_i and J_e**: it forces `∇·J_i = -∇·J_e`.

**This is the step that couples the two Cattaneo equations.** Without eq 5, eqs R1
and R2 are genuinely independent dual-evolving equations (coupled only through a
membrane source we introduce later). With eq 5, the two flux divergences are locked
equal-and-opposite, forcing one equation to be slaved to the other.

**The physical justification for eq 5:** charge neutrality is a universal result of
electrostatics at timescales much longer than the dielectric relaxation time
`τ_Debye = ε/σ`. For tissue, ε ≈ 80·ε_0 and σ ≈ 0.5 S/m, giving τ_Debye ≈ 10⁻⁹ s (1 ns).
Since cardiac APs evolve on 10⁻³ s (1 ms) timescales, we're six orders of magnitude
above the neutrality relaxation, so eq 5 is a very good approximation.

**But it is still an approximation.** The fast charge-redistribution modes it projects
out are real; they just happen too fast to matter for standard cardiac EP. If we want
2-DOF dual evolution, this is the approximation we have to relax.

### Move #2 (mathematical). Eliminate J via substitution

After imposing eq 5, Rossi uses `I_t = χ(C_m ∂V/∂t + I_ion) = -∇·J_i = ∇·J_e` (eq 6)
to eliminate J from the system entirely. The result is eqs 7-8 (in V_i, V_e). This is
a straightforward algebraic elimination — reversible, no information loss.

### Move #3 (variable choice). Change variables (V_i, V_e) → (V, V_e) where V = V_i − V_e

This is the step that makes V_e **appear** to be non-dynamical in the final equations.

In (V_i, V_e) variables, eqs 7-8 expand to:

```
(V_i, V_e) form:
τ_e [χ C_m ∂²(V_i − V_e)/∂t² + χ ∂I_ion/∂t] + χ C_m ∂(V_i − V_e)/∂t + χ I_ion = -∇·σ_e∇V_e
τ_i [χ C_m ∂²(V_i − V_e)/∂t² + χ ∂I_ion/∂t] + χ C_m ∂(V_i − V_e)/∂t + χ I_ion = ∇·σ_i∇V_i
```

**Both equations manifestly contain ∂²V_i/∂t² AND ∂²V_e/∂t².** If you kept (V_i, V_e)
as your primitive variables, the system would *look* 2-DOF all the way through.

Rossi's change of variables to (V, V_e) absorbs V_i's time derivatives into V's, making
V_e's derivatives vanish from the explicit form. His final eqs 13-14 have ∂²V/∂t² in
both equations but no ∂²V_e/∂t² anywhere — V_e looks "static." This is an artifact of
the variable choice, not a physical statement.

**So what's the honest story?** Rossi's formulation has 1 physical dynamical DOF (due
to Move #1) disguised as a 2-unknown system (due to Move #3). The 2-DOF character of
eqs 1-2 was lost at Move #1, not at Move #3.

---

## 3. To get genuine 2-DOF: relax Move #1

Replacing eq 5 with **independent compartmental charge conservation with finite storage**.

### 3.1 The reformulated equations

Keep Cattaneo fluxes (unchanged from Rossi):

```
(A1)   τ_i ∂J_i/∂t + J_i = -σ_i ∇V_i
(A2)   τ_e ∂J_e/∂t + J_e = -σ_e ∇V_e
```

Replace Rossi's eq 5 (quasi-static `∇·(J_i + J_e) = 0`) with **non-quasi-static
conservation** for each compartment:

```
(A3)   C_i ∂V_i/∂t + ∇·J_i = -I_m
(A4)   C_e ∂V_e/∂t + ∇·J_e = +I_m
```

where `I_m = χ[C_m ∂V/∂t + I_ion(V)]` with V = V_i − V_e is the membrane current per
unit tissue volume (identical to Rossi's I_t in eq 6).

The new parameters `C_i, C_e` are **volumetric storage capacitances** (units: F/m³)
that allow each compartment to accumulate net charge locally. They are zero in the
strict quasi-static limit (recovering Rossi).

### 3.2 What each equation now looks like

Eliminate J via Cattaneo: in the limit τ_i, τ_e → 0 (Ohm's law), J_c = -σ_c ∇V_c, so
`∇·J_c = -∇·σ_c ∇V_c`. Substituting into (A3, A4):

```
(A3')  C_i ∂V_i/∂t = ∇·σ_i ∇V_i − I_m
(A4')  C_e ∂V_e/∂t = ∇·σ_e ∇V_e + I_m
```

Each equation is **a heat equation with a source term**. V_i has its own time
evolution, V_e has its own time evolution, and they couple only through I_m.

For nonzero τ_i, τ_e, the equations become telegraph-style (second-order in time after
eliminating J), but the fundamental structure "each compartment evolves independently"
is preserved. In first-order-system form (keeping J as a dynamical variable):

```
Intracellular lattice:                Extracellular lattice:
  τ_i ∂J_i/∂t + J_i = -σ_i ∇V_i         τ_e ∂J_e/∂t + J_e = -σ_e ∇V_e
  C_i ∂V_i/∂t + ∇·J_i = -I_m             C_e ∂V_e/∂t + ∇·J_e = +I_m
```

Each "lattice" is a standard (V, J) telegraph system — the textbook LBM BGK recovery
at the Navier-Stokes level of Chapman-Enskog. Two independent LBM evolutions coupled
through the membrane source I_m.

### 3.3 Recovery of the standard bidomain

Take C_i, C_e → 0 in (A3, A4):

```
∇·J_i = -I_m    and    ∇·J_e = +I_m
```

which is exactly Rossi's eq 6, and adding them gives Rossi's eq 5:

```
∇·(J_i + J_e) = 0
```

So (A1-A4) reduce to Rossi's formulation in the quasi-static limit. No inconsistency,
no new physics — just a generalization with two extra parameters that tune how far we
stray from quasi-staticity.

### 3.4 Recovery in the Ohm limit (τ_i = τ_e = 0)

Take τ_i = τ_e = 0, C_i, C_e > 0:

```
C_i ∂V_i/∂t = ∇·σ_i ∇V_i − I_m
C_e ∂V_e/∂t = ∇·σ_e ∇V_e + I_m
```

Two coupled reaction-diffusion equations in (V_i, V_e). Each one first-order in time.
This is the **parabolic-parabolic bidomain** in the honest sense — two heat equations
with a membrane-current coupling. Not equivalent to the standard bidomain (which is
PE), but reduces to it as C_i, C_e → 0.

### 3.5 Recovery in the standard bidomain limit (τ = 0, C_i = C_e = 0)

Adding (A3', A4') with C_i = C_e = 0:

```
∇·σ_i ∇V_i + ∇·σ_e ∇V_e = 0
```

which (using V_i = V + V_e) is the standard elliptic constraint

```
∇·σ_i ∇V + ∇·(σ_i + σ_e) ∇V_e = 0
```

And subtracting gives the standard parabolic V equation. Full reduction to the
familiar PE bidomain.

---

## 4. Mapping to two independent LBM lattices (heat-equation style)

The (A1-A4) system in the **Ohm limit** (τ = 0) is two coupled heat equations — the
cleanest LBM target. Each LBM lattice solves one heat equation with a source term.

### 4.1 One lattice per compartment

```
Intracellular LBM: evolves V_i
   f_i^α on velocity set {e_α}  (D2Q5 isotropic or D2Q9 anisotropic)
   collision: f_i^α ← f_i^α − (f_i^α − f_i^α,eq) / τ_LBM,i + w_α · (source_i) · Δt
   equilibrium: f_i^α,eq = w_α · V_i  (standard diffusion LBM)
   source_i = -I_m / C_i   (coupling to extracellular via membrane)
   V_i = Σ_α f_i^α
   LBM-Ohm correspondence: σ_i / C_i = cs² (τ_LBM,i − 1/2) · Δt  (isotropic D2Q5)

Extracellular LBM: evolves V_e
   f_e^α on the same velocity set
   collision: f_e^α ← f_e^α − (f_e^α − f_e^α,eq) / τ_LBM,e + w_α · (source_e) · Δt
   equilibrium: f_e^α,eq = w_α · V_e
   source_e = +I_m / C_e
   V_e = Σ_α f_e^α
   LBM-Ohm correspondence: σ_e / C_e = cs² (τ_LBM,e − 1/2) · Δt
```

Coupling computed once per timestep: I_m = χ · [C_m · (V_i − V_e − V_i^old + V_e^old)/Δt
+ I_ion(V_i − V_e)]. Explicit, pointwise, no linear solve.

### 4.2 How this differs from Belmiloudi's dual-lattice LBM bidomain

Belmiloudi's dual-lattice uses two LBM lattices but enforces quasi-staticity — one
lattice time-marches Vm (parabolic) while the other pseudo-time-iterates φ_e (elliptic).
That recovers standard bidomain physics at the cost of LBM's main advantage (one step
per timestep).

The (A1-A4) formulation above lets **both** lattices time-march in real time. The
extracellular lattice is no longer a pseudo-time iterator — it evolves with its own
physical relaxation time. This is the novel algorithmic contribution.

### 4.3 Consistency check: does Chapman-Enskog recover the right macroscopic equation?

BGK's Chapman-Enskog for a scalar field u with source S gives, at the Navier-Stokes
level:

```
∂u/∂t = D ∇²u + S + O(Kn²)
```

where `D = cs² (τ_LBM − 1/2) Δt` and Kn is the Knudsen number (lattice spacing /
macroscopic gradient length). The second-order term (Kn²) contains the Cattaneo-type
`(τ_LBM − 1/2) Δt · ∂²u/∂t² · ...` correction — the "numerical artifact" we discussed
in KNOWLEDGE.md.

Mapping `u = V_i`, `D = σ_i / C_i`, `S = −I_m/C_i` gives exactly equation (A3') in the
Ohm limit. Two lattices, one per compartment, recover (A3', A4'). ✓

---

## 5. Parameter budget and physical interpretation

### 5.1 Parameters introduced by (A1-A4)

| Parameter | Units | Physical interpretation | Size for cardiac tissue |
|-----------|-------|--------------------------|--------------------------|
| τ_i, τ_e | s | Axial inductance relaxation time (Cattaneo) | 0.1–0.5 ms (Rossi range) |
| C_i, C_e | F/m³ | Volumetric compartmental capacitance (non-quasi-static) | **Not measured** — see below |
| C_m (existing) | F/m² | Membrane capacitance per unit area | 1.0 μF/cm² |
| σ_i, σ_e (existing) | S/m | Compartmental conductivities | 2–3 mS/cm |

### 5.2 What determines C_i and C_e?

Three interpretations, from most physical to most numerical:

**(a) Dielectric interpretation.** C = ε/L² where L is a characteristic length.
For tissue ε ≈ 80 ε_0 ≈ 7 × 10⁻¹⁰ F/m, and L ≈ 10 μm (cell size) gives
C ≈ 7 × 10⁻² F/m³. The corresponding time constant `τ_dielec = C/σ ≈ 10⁻⁹ s` (1 ns).
This is the Debye relaxation timescale — physically correct but irrelevant for cardiac
dynamics.

**(b) Effective compartmental capacitance from microstructure.** If the intracellular
and extracellular spaces contain capacitive substructure (connexins, tortuous bath
paths, membranous organelles), the effective volumetric capacitance could be orders of
magnitude larger than dielectric. No direct measurement in the literature.

**(c) Lattice-scale numerical parameter.** In LBM, C_c naturally emerges from the
lattice timestep and relaxation time. The LBM field's "capacitance" is effectively
`C_LBM = Δt / (cs² (τ_LBM − 1/2))` — set by the discretization, not by physics. This
gives us the freedom to choose C_c in the range where the resulting macroscopic
dynamics are interesting (i.e., τ_c and C_c such that the associated relaxation time is
comparable to the AP timescale, not the Debye timescale).

**Default choice:** use C_i, C_e as **free parameters** characterizing the
non-quasi-static regime. Set them based on the numerical experiment, with the
understanding that C_c = 0 recovers standard bidomain and C_c large recovers a
genuinely dual-evolving system. The physically "correct" value is not known, but the
sensitivity to C_c is the question we'd be answering with the LBM simulations.

---

## 6. Boundary conditions

At the tissue-bath interface in (A1-A4):

- V_i: Neumann (no-flux) on J_i (the intracellular compartment ends at the membrane)
- V_e: continuity with bath (V_e = V_bath at interface)
- Bath: use standard quasi-static elliptic (the bath is truly quasi-static — no cells,
  no membranes, nothing to store charge at sub-ms timescale)

So the bath stays 1-DOF elliptic; the tissue is 2-DOF dual-evolving. This is
consistent with the physical picture that the non-quasi-static charge storage
originates in the cellular microstructure, which doesn't exist in the bath.

---

## 7. What's gained, what's lost, what's unknown

### Gained
- **Two independent LBM lattices, genuinely time-marching.** Each step of each lattice
  is one collision-streaming cycle, no pseudo-time iteration. Fully GPU-parallelizable,
  no global linear solve.
- **A well-defined extension of standard bidomain** with two parameters (C_i, C_e)
  that recover the standard bidomain in the limit C → 0.
- **A cleaner PDE structure** — two heat-equation-style equations are easier to reason
  about than one elliptic constraint coupled to a parabolic equation.
- **A potentially novel physical regime.** If the macroscopic dynamics at finite C_c
  differ from the C_c = 0 limit, we've discovered non-trivial non-quasi-static effects.
  If they don't, we've numerically confirmed quasi-staticity is a good approximation
  at cardiac timescales, which is itself a meaningful check.

### Lost
- **Strict equivalence to standard bidomain.** The (A1-A4) system is *not* the same
  PDE as Rossi's — only in the limit C_i, C_e → 0. Runs with finite C_c solve a
  different equation. This is either a feature (new physics) or a bug (unknown
  systematic error), and which one depends on whether the results make physical sense.
- **A clear theoretical baseline from the literature.** The parameters C_i, C_e have no
  published "correct" values. Nobody has reported CV, APD, or arrhythmia dynamics in
  the (A1-A4) system. We'd be in uncharted territory.

### Unknown
- **Does the dual-evolving solution converge to the standard bidomain solution** as
  C_i, C_e → 0? Should, but convergence rate and any stiff regimes need numerical
  confirmation.
- **What values of C_i, C_e produce observable differences** from quasi-static? Might
  be so small that differences are hidden below discretization error; might be large
  enough that the system is interestingly different.
- **Stability.** Hyperbolic systems can have strict CFL-type conditions. The coupling
  through I_m is an explicit source; its stability at large membrane currents (during
  the upstroke) needs analysis.

---

## 8. Concrete proposal for the implementation path

1. **Derive and write down (A1-A4) in the specific variables we'll simulate.** This
   document is the derivation; the next step is to write `src/bidomain_two_dof.py`-level
   interface specs.

2. **Numerical sanity check: C_i = C_e → 0 recovery of standard PE bidomain.** Before
   any LBM, run an FDM or spectral version of (A1-A4) with small C_c and verify it
   converges to our existing Bidomain V1 PE results. This retires the "did we write
   down the right equation" risk cheaply.

3. **C_i = C_e > 0 single-lattice-equivalent experiment.** Run (A1-A4) with a range of
   C_c values (e.g., 10⁻⁶, 10⁻⁵, ..., 10⁻¹ in normalized units) and look for the
   threshold at which dynamics diverge from the PE baseline. This characterizes the
   physical regime where non-quasi-staticity matters.

4. **Dual-lattice LBM implementation.** Build on the existing LBM V1 engine, adding a
   second lattice and coupling via membrane source. This is where the implementation
   novelty lives.

5. **Kleber boundary experiment in the (A1-A4) system.** Compare wavefront curvature at
   tissue-bath interface between standard PE, Rossi's hyperbolic-hyperbolic (FEM), and
   our LBM (A1-A4) dual lattice. If they agree in the appropriate limits, we've
   validated the new formulation. If they disagree at the boundary, that's the novel
   physical finding.

---

## 9. One-line summary

Rossi's Cattaneo flux equations (eqs 1-2) are **already 2-DOF**; it is his **eq 5**
(quasi-static charge conservation) that mathematically collapses them to 1-DOF-with-
memory. Replacing eq 5 with independent compartmental conservation with storage
parameters C_i, C_e yields a system where both V_i and V_e have genuine first-order
time derivatives — a direct fit for dual-lattice LBM.

---

## 10. CORRECTION (2026-04-23): the phenomenological C_c was not rigorous

**§3 and §5 above are retained as an archive of the first attempt but should not be
used as the project's formulation.** The proposal "replace eq 5 with C_c ∂V_c/∂t +
∇·J_c = ±I_m" implicitly assumes a linear constitutive law ρ_c = C_c V_c (capacitor-
like). That relation is not wrong, but it is **phenomenological** — defending the
values of C_c requires a separate physical argument that wasn't made. Selecting C_c as
a free parameter ("lattice-scale freedom") amounts to letting a numerical knob
determine the physics.

### What I conflated

Two distinct "charge conservation" statements:

- **Continuity** (universal, unbreakable): `∂ρ/∂t + ∇·J = 0`
- **Quasi-static neutrality** (an approximation): `∂ρ/∂t ≈ 0`, from which eq 5 follows

Rossi's eq 5 follows from *both*. I claimed to "replace eq 5" but the phenomenological
ansatz `ρ_c = C_c V_c` is equivalent to a specific assumption about the continuity
equation's structure, not a first-principles derivation.

### The honest 2-DOF formulation: Poisson-Nernst-Planck (PNP)

Don't absorb ρ_c into V_c via an ansatz. Track ρ_c as an explicit dynamical variable.
Per compartment c ∈ {i, e}:

```
(P)   -∇·(ε_c ∇V_c) = ρ_c                         (Poisson: V_c follows ρ_c)
(C)    ∂ρ_c/∂t + ∇·J_c = S_c                      (continuity: ρ_c evolves)
(J)    J_c = -σ_c ∇V_c - D_c ∇ρ_c + drift terms   (constitutive: Ohm + ion diffusion)
(M)    S_i = -I_m,  S_e = +I_m,  I_m = χ(C_m ∂V/∂t + I_ion(V))
```

- **ρ_i, ρ_e are the two dynamical DOFs** (one per compartment).
- **V_i, V_e are instantaneous images of ρ_i, ρ_e through Poisson** (each compartment).
- Quasi-static bidomain = project onto the ρ_c = 0 manifold. Relaxing it = keep ρ_c
  dynamics explicitly.
- Conservation is preserved everywhere: eq (C) is the full continuity law; eq (M) is
  the membrane-transfer source/sink; adding S_i + S_e = 0 confirms no charge is
  created at the membrane.

### Why the timescale actually matters for cardiac EP

The dielectric Debye time `τ_Debye = ε/σ ≈ 1 ns` is not the relevant timescale for
PNP-vs-bidomain divergence. The dominant physical mechanism is **ion redistribution
via diffusion over cellular scales**:

```
τ_diff = L² / D_ion ≈ (10 μm)² / (10⁻⁹ m²/s) ≈ 10⁻⁴ s = 0.1 ms
```

That is the same order as the AP upstroke. **The quasi-static reduction projects out
dynamics at exactly the timescale we care about.** This is the scientifically honest
motivation for the 2-DOF direction — not "finite volumetric capacitance" (weakly
justified) but "ion redistribution physics that bidomain averages over" (physically
real, cardiac-relevant).

### Two routes from PNP to something computable

- **(PNP-lite)** Track `ρ_c` as a lumped scalar per compartment (not individual ion
  species). Constitutive: `J_c = -σ_c ∇V_c - D_eff ∇ρ_c`. Two fields per compartment
  (ρ_c, V_c) linked by Poisson. Minimal 2-DOF extension of bidomain. Two LBM lattices
  sufficient.

- **(Full multi-species PNP)** Track each ion species: `∂n_{α,c}/∂t =
  ∇·(D_α ∇n_{α,c} + μ_α n_{α,c} ∇V_c) + reactions`. `ρ_c = F Σ_α z_α n_{α,c}`.
  Maximally rigorous. One LBM lattice per (species × compartment) — many lattices, but
  each is standard reaction-drift-diffusion.

### What's still true from §§1-2

The identification of **Rossi's eq 5 as the collapse step** is correct and stands
unchanged. What's wrong is my specific replacement for it in §§3-5. The PNP framework
above is the physically rigorous replacement; the C_c ansatz should be read as an LBM
numerical-regime interpretation, not a physical model.

### Open question the correction raises

Does there exist a derivation from PNP that yields a closed reduced system in
(V_i, V_e) alone (without needing to track ion species or ρ_c)? This would be a
"PNP-bidomain" — intermediate between quasi-static bidomain and full PNP. If it
exists, it would be the ideal target for the dual-lattice LBM approach. Literature
search pending (Mori, Peskin, Mori-Eisenberg 2011 etc.).
