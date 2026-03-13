# ML-DO Infarct Geometry: Problem Statement & Critical Analysis

## Overview

This folder investigates the central simulation artifact underpinning the proposed
Oxford research: **Machine Learning-Directed Optimisation of Arrhythmogenic Inert
Scar Geometries (ML-DO)**. The proposal's computational engine relies on D2Q9 LBM
with Dirichlet boundaries at scar borders to capture boundary conduction velocity
effects. Work already completed in this project reveals that this mechanism is
physically incorrect and must be addressed before the ML-DO framework can produce
trustworthy results.

---

## 1. The Oxford Proposal (Summary)

**Title:** ML-DO of Arrhythmogenic Inert Scar Geometries

**Core idea:** Use machine learning to optimise infarct scar shapes that maximise
arrhythmogenicity (reentry inducibility), then validate against engineered 2D hiPSC
tissues with laser-cut inert regions.

**The ML-DO loop:**
1. Encode scar geometries via Bezier curves or Fourier descriptors
2. Evaluate each geometry in a D2Q9 LBM-EP simulation (S1-S2 reentry protocol)
3. Train a neural network to map geometry -> arrhythmogenicity score
4. Use the NN to select high-risk candidates for further simulation
5. Repeat until convergence

**Ionic models:** ORd or TTP06, tuned to hiPSC APD values. Optionally a
transformer-based surrogate ionic model trained on patch clamp recordings.

**Objective functions:** Spiral wave anchoring stability, wavelength-to-obstacle-size
ratio (CV x APD), reentry inducibility via S1-S2.

**In vitro validation:** hiPSC-CM monolayers with laser-ablated scar geometries,
recorded via Arclight (voltage) + Rhod-2 AM (calcium).

**Collaborators:** Molly Stevens (tissue models), David Paterson (Burdon Sanderson,
MEA), Cathy Ye (iPSC fabrication), Jean-Baptiste Lugagne (closed-loop ML control).

---

## 2. Two Distinct Boundary CV Phenomena

The existing work in this project has rigorously established that there are **two
fundamentally different** mechanisms that alter conduction velocity near obstacle
boundaries. These must not be conflated.

### 2A. Curvature Effect (Monodomain, Geometric)

**Source:** Eikonal-curvature relation: v(kappa) = v_0 - D * kappa

- At obstacle **corners**, the wavefront curves -> concave front (kappa < 0) -> speedup
- At obstacle **flat edges** parallel to propagation: kappa = 0 -> NO effect
- Captured by ANY consistent discretization (FDM, FEM, LBM D2Q5, LBM D2Q9)
- This is a geometric effect of the Laplacian operator, not a boundary condition effect
- No-flux BC is passive (zero power contribution) -- proved via energy balance

**Evidence:** `Research/Infarct_Boundary_Speedup/proofs/No_Flux_BC_Proof.md`
(Theorems 1.1, 2.1, 3.1, 4.1 -- formal proofs with matched asymptotics)

### 2B. Kleber Boundary Speedup (Bidomain, Physical BCs)

**Source:** Intracellular/extracellular boundary condition asymmetry at tissue-bath
or tissue-scar interfaces where the scar retains extracellular conductivity.

- Intracellular: Neumann (cell network terminates at scar)
- Extracellular: Dirichlet (continuous with bath/scar extracellular space, phi_e -> 0)
- Bath "shorts out" extracellular resistance near boundary
- Effective D transitions from D_eff to D_i over electrotonic space constant lambda
- **CV ratio = sqrt((sigma_i + sigma_e) / sigma_e) ~ 1.13 (13% speedup)**
- Occurs along **flat edges**, not just corners
- Requires bidomain equations or monodomain with spatially varying D(x,y)
- Cannot be captured by monodomain with any uniform BC

**Evidence:** `Bidomain/Engine_V1/research/BOUNDARY_SPEEDUP_ANALYSIS.md` (theory)
and Phase 6 cross-validation (16/16 tests PASSED):
- Insulated bidomain: ratio = 1.0000 (no effect, as expected)
- Bath-coupled bidomain: ratio = 1.0714 (7.1% speedup at dx=0.025, converging to 1.131)
- Mesh convergence confirmed: 1.039 (coarse) -> 1.071 (medium) -> 1.131 (theory)

---

## 3. The Critical Problem

### What the Oxford Proposal Claims

> "in a D2Q9 LBM-EP model with a Dirichlet boundary, the diagonal streaming
> directions connecting to the inert regions will bounce back, which lowers the
> effective coupling, thus mimicking the artefacts mentioned above."

The proposal asserts that D2Q9 + Dirichlet BC at scar borders will reproduce the
boundary CV effects seen in real tissue. This is the computational justification
for choosing LBM over FDM/FEM.

### What Our Analysis Shows

**The D2Q9 mechanism does NOT produce the correct physics.** Three independent lines
of evidence:

#### Evidence 1: D2Q9 Bounce-Back Produces SLOWDOWN, Not Speedup

Phase 6 cross-validation (Config B: D2Q9 Neumann):
- D2Q9 diagonal bounce-back at flat boundaries produces ~3% CV **slowdown**
- This is an O(dx^2) numerical artifact that vanishes with mesh refinement
- It is in the **opposite direction** to the Kleber speedup

The diagonal distributions (f_5, f_6, f_7, f_8) carry x-momentum even at y-boundaries.
Bounce-back introduces a small error in the non-equilibrium part, causing:
```
V(x, boundary) - V(x, interior) ~ -w * dx^2 * d^2V/dx^2
```
At the rising edge: d^2V/dx^2 > 0, so V_boundary < V_interior -> later activation
-> SLOWDOWN.

**Reference:** `BOUNDARY_SPEEDUP_ANALYSIS.md` Section 8.

#### Evidence 2: Dirichlet BC on V Creates a Current Sink

Dirichlet V = V_rest at scar boundary:
- During depolarization (V > V_rest), the boundary **drains** current from adjacent cells
- In LBM anti-bounce-back: f_opp = -f_i* + 2*w_i*V_rest (returning distributions
  carry less energy than sent out)
- This INCREASES electrotonic loading on boundary cells -> SLOWS conduction

The Kleber effect operates on phi_e (extracellular potential), not V (transmembrane
voltage). Clamping phi_e = 0 reduces extracellular resistance. Clamping V = V_rest
creates an artificial current sink. These are fundamentally different operations on
different variables.

**Reference:** `BOUNDARY_SPEEDUP_ANALYSIS.md` Section 4.

#### Evidence 3: Monodomain Cannot Capture the Kleber Effect

The Kleber speedup requires asymmetric BCs on two different domains:
- Intracellular Neumann + Extracellular Dirichlet

The monodomain equation lumps both domains into a single effective diffusion. With
any single BC (Neumann, Dirichlet, or Robin) on V, the monodomain PDE admits a 1D
solution for planar waves. Any consistent discretization must converge to this 1D
solution. Deviations are numerical artifacts, not physics.

The **only** way to capture Kleber in monodomain is via spatially varying D(x,y)
that encodes the bidomain boundary layer analytically:
```
D(x,y) = D_eff + (D_i - D_eff) * exp(-d(x,y) / lambda)
```
where d is distance to nearest scar boundary and lambda is the electrotonic space
constant (~1.4 mm for human ventricular tissue).

---

## 4. Implications for the ML-DO Framework

### What IS Still Valid

1. **The ML-DO optimization loop** -- geometry encoding, NN surrogate, iterative
   refinement. This is independent of the simulation engine details.

2. **Curvature effects at scar corners** -- these ARE captured by D2Q9 LBM
   (or any method). The v(kappa) relation works correctly in LBM.

3. **Reentry inducibility via S1-S2** -- the protocol itself is standard and
   well-suited to LBM's parallel nature.

4. **In vitro validation** -- hiPSC monolayers with laser-cut scars will show
   real physics regardless of what the simulation predicts.

5. **Surrogate ionic model** -- transformer on patch clamp data is independent
   of the boundary physics question.

### What Needs To Be Fixed

The simulation engine must correctly capture boundary CV effects at scar borders.
Three options, in order of physical fidelity:

#### Option A: Full Bidomain LBM (Best Physics)

Two LBM lattices -- one for phi_i (or V), one for phi_e:
- Intracellular lattice: Neumann BC at scar border (bounce-back)
- Extracellular lattice: Dirichlet BC at scar border (anti-bounce-back, phi_e = 0)
- Coupled through transmembrane current: R = chi * (Cm * dV/dt + I_ion)

**Advantage:** No global elliptic solve (unlike FDM bidomain). Each lattice streams
independently; coupling is through the local source term. This preserves LBM's
parallelism and GPU-friendliness.

**Cost:** 2x memory (two sets of distributions), ~2x compute per step. But
eliminating the elliptic solve makes bidomain LBM potentially FASTER than
bidomain FDM for the same problem.

**Reference:** `BOUNDARY_SPEEDUP_ANALYSIS.md` Section 7.

#### Option B: Monodomain LBM with Spatially Varying D (Good Approximation)

Use the analytically derived D(x,y) profile that encodes the bidomain boundary layer:
```
D(x,y) = D_eff + (D_i - D_eff) * exp(-d(x,y) / lambda)
```

In MRT-LBM, each node gets its own relaxation rate from local D:
```
tau(x,y) = 0.5 + D(x,y) * dt / (cs2 * dx^2)
```
No stencil modification needed. Standard Neumann BC (bounce-back) everywhere.

**Advantage:** Minimal code changes from existing monodomain LBM. Same memory
footprint. Captures ~90% of the Kleber effect (validated in Phase 6).

**Cost:** Requires computing distance-to-scar for every node (one-time cost per
geometry). The exponential profile is an approximation -- accuracy depends on
scar geometry complexity.

**Reference:** `BOUNDARY_SPEEDUP_ANALYSIS.md` Section 3.

#### Option C: Monodomain LBM with Neumann BC (Curvature Only)

Standard monodomain D2Q9 with bounce-back at scar borders.

**What it captures:** Curvature effects at scar corners (v(kappa) relation).
**What it misses:** Kleber boundary speedup along flat scar edges (~13% CV change).

**When this is acceptable:** If the ML-DO objective function is dominated by
corner curvature effects rather than flat-edge speedup. For highly irregular scar
shapes (many corners, few straight edges), the curvature effect may dominate.

---

## 5. Quantitative Impact Assessment

How much does the boundary physics matter for ML-DO results?

### Curvature Effect Magnitude

For a wavefront wrapping around a scar corner with radius of curvature R:
```
Delta_v / v_0 = D / (v_0 * R)
```
For typical values (D = 0.001 cm^2/ms, v_0 = 0.05 cm/ms, R = 0.1 cm):
```
Delta_v / v_0 = 0.001 / (0.05 * 0.1) = 0.2 = 20%
```
At sharp corners (small R), curvature effects are LARGE and dominate.

### Kleber Effect Magnitude

Along flat scar edges:
```
Delta_v / v_0 = sqrt((sigma_i + sigma_e) / sigma_e) - 1 ~ 0.13 = 13%
```
This is a constant 13% independent of geometry.

### Relative Importance for ML-DO

For the S1-S2 reentry inducibility protocol, what matters most is:
1. **Wavelength = CV x APD** -- determines whether reentry is possible around the scar
2. **CV heterogeneity** -- creates vulnerability windows for S2 capture
3. **Wavefront curvature at corners** -- determines whether the wave can navigate
   around the scar geometry

The Kleber effect modifies wavelength by ~13% near scar borders. This shifts the
critical scar size for reentry and affects the vulnerability window. For the ML-DO
objective function, this is a **systematic bias** -- all geometries are affected
similarly, so the RANKING of geometries may be preserved even without Kleber
correction. However, the absolute thresholds (critical scar size, minimum
coupling interval) would be wrong.

**Bottom line:** The ML-DO ranking may be approximately correct with Option C,
but Option B or A is needed for quantitative accuracy and meaningful comparison
with in vitro experiments.

---

## 6. Open Questions

1. **Does the hiPSC monolayer have a bath-like extracellular environment?**
   If hiPSC-CM monolayers are perfused (submerged in Tyrode's), the extracellular
   space IS bath-coupled, and the Kleber effect is present. If they are grown on
   a solid substrate with minimal extracellular volume, the effect may be reduced.
   This determines whether the in vitro experiments will show Kleber speedup.

2. **What is the effective lambda in hiPSC tissue?**
   The space constant depends on D_eff and resting membrane conductance. hiPSC-CMs
   have different conductivities than adult human ventricular tissue. Need to
   characterise sigma_i, sigma_e, G_m_rest for the specific cell line.

3. **Can we validate Kleber speedup in the hiPSC system directly?**
   This would be a valuable standalone result -- first direct measurement of
   boundary CV speedup in engineered cardiac tissue.

4. **Does the D2Q9 diagonal artifact matter for ranking?**
   The D2Q9 artifact produces a small SLOWDOWN (~3%) at flat edges. Since this is
   O(dx^2) and vanishes with refinement, it introduces a mesh-dependent bias. If
   ML-DO training uses a fixed mesh, this artifact could systematically alter which
   geometries appear most arrhythmogenic. Need to verify ranking stability across
   mesh resolutions.

5. **Does the surrogate ionic model need to capture APD restitution?**
   Reentry depends critically on APD restitution (APD vs diastolic interval). If the
   transformer surrogate doesn't capture this, the S1-S2 protocol results will be
   unreliable regardless of boundary physics.

---

## 7. Recommended Path Forward

### Phase 1: Validate Curvature Effects in LBM (Weeks 1-2)

Demonstrate that the existing monodomain LBM correctly captures v(kappa) around
circular and rectangular obstacles. Compare against eikonal theory. This establishes
the baseline that curvature effects ARE correct.

### Phase 2: Implement Spatially Varying D (Option B) (Weeks 3-4)

Add distance-to-scar computation and D(x,y) profile to the LBM engine. Validate
against the bidomain Engine V1 Phase 6 results (Kleber ratio convergence).

### Phase 3: Quantify Impact on ML-DO Ranking (Weeks 5-6)

Run a small ML-DO experiment (10-20 scar geometries) with and without the Kleber
correction. Compare rankings. If rankings are preserved (high correlation), Option B
is sufficient. If rankings change significantly, Option A (full bidomain LBM) is
needed.

### Phase 4: Full Bidomain LBM (If Needed) (Weeks 7-10)

Implement two-lattice bidomain LBM. Validate against bidomain FDM Engine V1.
This becomes the production engine for ML-DO if Phase 3 shows ranking sensitivity.

---

## 8. Document Cross-References

| Document | Location | Relevance |
|----------|----------|-----------|
| Oxford Research Statement | `~/Downloads/Oxford Research Statement.pdf` | Original proposal |
| Boundary Speedup Analysis | `Bidomain/Engine_V1/research/BOUNDARY_SPEEDUP_ANALYSIS.md` | Kleber theory + LBM analysis |
| No-Flux BC Proof | `Research/Infarct_Boundary_Speedup/proofs/No_Flux_BC_Proof.md` | Rigorous curvature-only proof |
| Experimental Validation | `Research/Infarct_Boundary_Speedup/Experimental_Validation.md` | Literature gap analysis |
| Cross-Validation Plan | `Bidomain/Engine_V1/CROSS_VALIDATION_PLAN.md` | Phase 6 experiment design |
| Bidomain Engine Progress | `Bidomain/Engine_V1/PROGRESS.md` | Phase 6 results (16/16 pass) |
| LBM V1 Progress | `Monodomain/LBM_V1/PROGRESS.md` | D2Q9 artifact documented |
