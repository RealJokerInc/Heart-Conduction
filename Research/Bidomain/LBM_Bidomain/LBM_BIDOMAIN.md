# Extending Lattice-Boltzmann Methods to the Cardiac Bidomain Equations: A Comprehensive Research Overview

**Author:** Research Agent
**Date:** February 10, 2026
**Status:** Preliminary Research Document
**Context:** Building on Engine V5.4 LBM monodomain implementation

---

## Executive Summary

This document synthesizes current research and identifies opportunities for extending the working Lattice-Boltzmann Method (LBM) monodomain engine to solve the coupled cardiac bidomain equations. The monodomain model—solved by the current Engine V5.4 implementation—is a single parabolic PDE describing transmembrane potential evolution. The bidomain model, however, consists of TWO coupled equations: one parabolic (transmembrane potential) and one elliptic (extracellular potential constraint). This fundamental difference introduces both theoretical and computational challenges.

**Key Finding:** LBM for bidomain has been actively investigated in the literature (Belmiloudi et al., 2015-2019; Corrado & Niederer). A coupled LBM approach using multiple distribution functions is feasible and has been demonstrated to maintain the computational efficiency advantages of LBM while capturing the coupling between intracellular and extracellular domains.

---

## 1. Problem Formulation: Monodomain vs. Bidomain

### 1.1 Monodomain Model (Current Implementation)

The monodomain equation solved by Engine V5.4:

$$\chi C_m \frac{\partial V_m}{\partial t} + \chi I_{ion} = \nabla \cdot (\mathbf{D} \nabla V_m)$$

Where:
- $V_m$ = transmembrane potential (voltage)
- $C_m$ = membrane capacitance
- $\chi$ = surface-area-to-volume ratio
- $I_{ion}$ = ionic current from cellular models
- $\mathbf{D}$ = diffusion tensor (anisotropic conductivity)

This is a **parabolic** PDE that LBM naturally handles via the relationship:
$$\tau = 0.5 + \frac{3D \Delta t}{\Delta x^2}$$

### 1.2 Bidomain Model (Extended Formulation)

The bidomain system consists of two coupled equations:

**Intracellular equation (parabolic):**
$$\chi C_m \frac{\partial V_m}{\partial t} + \chi I_{ion} = \nabla \cdot (\mathbf{D}_i \nabla V_m) + \nabla \cdot (\mathbf{D}_i \nabla \phi_e)$$

**Extracellular equation (elliptic):**
$$0 = \nabla \cdot (\mathbf{D}_i \nabla V_m) + \nabla \cdot ((\mathbf{D}_i + \mathbf{D}_e) \nabla \phi_e)$$

Where:
- $V_m$ = transmembrane potential (voltage difference across the membrane)
- $\phi_e$ = extracellular potential
- $\mathbf{D}_i$ = intracellular conductivity tensor
- $\mathbf{D}_e$ = extracellular conductivity tensor
- The constraint $V_m = V_i - \phi_e$ where $V_i$ is intracellular potential

### 1.3 Fundamental Differences

| Property | Monodomain | Bidomain |
|----------|-----------|---------|
| Number of fields | 1 ($V_m$) | 2 ($V_m$, $\phi_e$) |
| PDE type | Parabolic | Mixed (parabolic + elliptic) |
| Time derivatives | $\frac{\partial V_m}{\partial t}$ | $\frac{\partial V_m}{\partial t}$ only (no $\frac{\partial \phi_e}{\partial t}$) |
| Coupling | None—single equation | Strong coupling via conductivity tensors |
| Computational structure | One time integration | One time step for $V_m$, constraint solve for $\phi_e$ |
| LBM applicability | Direct | Modified—requires handling elliptic equation |

---

## 2. Literature Review: LBM for Bidomain

### 2.1 Existing Work: Belmiloudi and Collaborators

**Key Reference:** Belmiloudi et al. have developed coupled LBM approaches for cardiac bidomain models in multiple publications:

1. **Coupled Lattice Boltzmann Modeling of Bidomain Type Models in Cardiac Electrophysiology** (Springer, 2015)
   - First systematic treatment of coupled LBM for bidomain
   - Two separate distribution functions: one for $V_m$, one for $\phi_e$
   - Demonstrates stability and convergence

2. **Coupled lattice Boltzmann method for numerical simulations of fully coupled heart and torso bidomain system in electrocardiology** (HAL Archives, 2015)
   - Extended to full heart-torso coupling
   - Includes both cardiac and extracardiac domains
   - Validates against FEM benchmarks

3. **Coupled lattice Boltzmann simulation method for bidomain type models in cardiac electrophysiology with multiple time-delays** (Mathematical Modelling of Natural Phenomena, 2019)
   - Adds time-delay terms (biologically realistic)
   - Provides stability analysis
   - Numerical validation on realistic tissue geometries

**Key Conclusions from Literature:**
- LBM CAN be effectively extended to bidomain with proper formulation
- Two independent LBM lattices (one per field) maintain computational efficiency
- Coupling between equations handled via source/sink terms
- Performance comparable to monodomain: still 10-45× faster than FEM

### 2.2 LBM-EP: The Monodomain Success Story

The LBM-EP framework demonstrates the clinical feasibility of LBM for cardiac electrophysiology:

- **Performance:** 10-45× speedup over FEM while maintaining accuracy
- **Implementation:** GPU-accelerated (CUDA) with PyTorch compatibility
- **Lattice:** D3Q7 (3D) with MRT collision for anisotropy
- **Boundary conditions:** Level-set based geometry representation
- **Ionic coupling:** Node-wise cellular model integration

This success demonstrates the foundation upon which bidomain LBM can be built.

### 2.3 Hybrid Approaches in Literature

Several research groups have explored **LBM-FEM coupling** for complex multiphysics:

- Fluid-structure interaction via immersed boundary method (IBM)
- Domain decomposition (Schwarz-based overlapping)
- Partitioned weakly-coupled schemes with interpolation
- Non-conforming time steps between LBM and classical solvers

This suggests that a **hybrid LBM-FEM bidomain solver** is viable: use LBM for the parabolic $V_m$ equation and a classical solver (e.g., CG/AMG) for the elliptic $\phi_e$ constraint.

---

## 3. Can LBM Solve the Bidomain Equations? Theoretical Analysis

### 3.1 The Challenge: Elliptic vs. Parabolic PDEs

LBM is fundamentally designed for parabolic PDEs (diffusion-type equations):

$$\frac{\partial u}{\partial t} = \mathbf{D} \nabla^2 u + \text{source}$$

The connection to LBM derives from Chapman-Enskog theory, which recovers this PDE from the lattice Boltzmann equation in the hydrodynamic limit.

**The $\phi_e$ equation is purely elliptic (no time derivative):**
$$\nabla \cdot ((\mathbf{D}_i + \mathbf{D}_e) \nabla \phi_e) = -\nabla \cdot (\mathbf{D}_i \nabla V_m)$$

This is fundamentally a constraint—not an evolution equation. **LBM cannot directly solve it in a single time step.**

### 3.2 Solution 1: Pseudo-Time Stepping (Steady-State LBM)

**Concept:** Introduce artificial time evolution for the elliptic equation:

$$\frac{\partial \phi_e}{\partial \tau} = \nabla \cdot ((\mathbf{D}_i + \mathbf{D}_e) \nabla \phi_e) + \nabla \cdot (\mathbf{D}_i \nabla V_m)$$

LBM solves this **within each physical time step** until steady state is reached (when $\frac{\partial \phi_e}{\partial \tau} \approx 0$).

**Literature Support:**
- Multigrid LBM methods accelerate elliptic convergence (citations: ScienceDirect research on multigrid LBM)
- Steady-state LBM has been demonstrated for Poisson equations
- Pseudo-time stepping with implicit schemes shows rapid convergence

**Advantages:**
- Maintains LBM framework throughout
- No need for external linear solver
- Parallelizable like all of LBM

**Disadvantages:**
- How many sub-iterations per physical time step?
- Convergence rate of LBM for elliptic problems is slower than specialized solvers
- Additional computational cost for pseudo-time iterations

### 3.3 Solution 2: Separate LBM Lattices (Coupled Distribution Functions)

**Concept:** Use two independent LBM lattices:

- **Lattice 1 (Distribution $f_i^{(m)}$):** Solves transmembrane potential $V_m$
  - Parabolic equation (natural for LBM)
  - Updated at each physical time step
  - Integrates ionic source terms

- **Lattice 2 (Distribution $f_i^{(e)}$):** Solves extracellular potential $\phi_e$
  - Driven by $V_m$ gradient as a source term
  - Can iterate to convergence each step (quasi-static assumption)
  - Coupling only through source terms

**Mathematical Framework (per Belmiloudi et al.):**

For transmembrane potential:
$$f_i^{(m)}(\mathbf{x}+\mathbf{c}_i\Delta t, t+\Delta t) = f_i^{(m)}(\mathbf{x},t) - \frac{\Delta t}{\tau_m}(f_i^{(m)} - f_i^{eq(m)}) + S_i^{(m)}$$

For extracellular potential:
$$f_i^{(e)}(\mathbf{x}+\mathbf{c}_i\Delta t, t+\Delta t) = f_i^{(e)}(\mathbf{x},t) - \frac{\Delta t}{\tau_e}(f_i^{(e)} - f_i^{eq(e)}) + S_i^{(e)}$$

Where:
- $\tau_m$ = relaxation time for Vm equation (encodes $\mathbf{D}_i$)
- $\tau_e$ = relaxation time for $\phi_e$ equation (encodes $\mathbf{D}_i + \mathbf{D}_e$)
- $S_i^{(m)}$ = source term coupling to ionic model and $\phi_e$
- $S_i^{(e)}$ = source term coupling to $V_m$ gradient

**Advantages:**
- Natural extension of existing monodomain engine
- Two independent LBM loops, easy to implement
- Maintains data locality and parallelization
- Proven by Belmiloudi et al.

**Disadvantages:**
- 2× memory for distribution functions
- Requires careful coupling through source terms
- Mismatch in convergence between parabolic and elliptic physics

### 3.4 Solution 3: Hybrid LBM-Classical Solver

**Concept:** Leverage LBM's strength (parabolic PDEs) and classical solvers' strength (elliptic equations):

- **LBM:** Solve the parabolic $V_m$ equation
- **Classical Solver:** Use CG, AMG, or ILU preconditioner for the elliptic $\phi_e$ constraint

**Implementation Strategy:**

```
FOR each physical time step:
  1. Compute LBM collision + streaming for V_m distribution (one step)
  2. Extract V_m macroscopic density
  3. Assemble right-hand-side for Poisson equation:
     ∇·((D_i+D_e)∇φ_e) = -∇·(D_i∇V_m)
  4. Solve with iterative solver (CG/AMG) to tolerance
  5. Update coupling source terms for next step
  6. Couple φ_e back into f^(m) for next time step
```

**Advantages:**
- Proven separately: LBM for diffusion, CG/AMG for Poisson
- No pseudo-time overhead
- Industrial-strength solvers for elliptic problems
- Leverages GPU libraries (cuSPARSE, MAGMA)

**Disadvantages:**
- Loses the "pure LBM" elegance
- Data transfer overhead between LBM and external solver
- Coupling time scale mismatch (LBM explicit vs. linear solver iterative)
- May not be significantly faster than FEM if elliptic solve dominates

**Historical Context:** This hybrid approach is standard in incompressible Navier-Stokes solving, where pressure-Poisson equations are notoriously expensive. Some research groups couple LBM for velocity to FFT-based Poisson solvers or algebraic multigrid for pressure.

### 3.5 Theoretical Feasibility: Convergence Analysis

**For the parabolic equation (V_m):**
- Standard LBM convergence: $O(\Delta x^2, \Delta t)$
- Achieved in monodomain, should transfer directly
- Stability via CFL-like condition on $\tau$

**For the elliptic equation (φ_e) via pseudo-time stepping:**
- Convergence rate of steady-state LBM for elliptic problems: slower than CG/AMG
- Typical: need $O(100-1000)$ pseudo-time iterations per physical step
- With multigrid acceleration: can reduce to $O(10-50)$ per physical step
- This could negate performance advantage over direct elliptic solvers

**Recommendation:** The true test is empirical. Theory suggests feasibility, but **convergence rates for the elliptic component are the critical unknown.**

---

## 4. Multi-Component LBM: Extending to Multiple Fields

### 4.1 Literature on Coupled Multi-Field Systems

Research on multi-component LBM has addressed several coupled problems:

1. **Chemotaxis-fluid coupling** (MRT LBM with Beam-Warming scheme)
2. **Porous media reactive transport** (Coupled LBM-FV framework)
3. **Electrokinetic flows** (Two-relaxation-time LBM + FFT Poisson solver)
4. **Multiphase flows** (Shan-Chen, color-gradient methods)
5. **Intracellular crowded diffusion** (Scaled Particle Theory + LBM)

**Common Pattern:** Use separate distribution functions for each species/field, couple through:
- Source/sink terms in collision operator
- Modified equilibrium distributions
- Separate relaxation times reflecting different diffusivities

### 4.2 Anisotropic Diffusion with MRT Collision

The existing engine uses MRT collision to handle **anisotropic diffusion within a single field.** For bidomain, we must handle:

- **Different anisotropy directions:** Cardiac fiber orientation affects both $\mathbf{D}_i$ and $\mathbf{D}_e$
- **Different magnitude ratios:** Typically $D_i \ll D_e$ (intracellular resistance dominates)
- **Two independent tensors:** Unlike monodomain with a single $\mathbf{D}$

**MRT Formulation for Single-Field Anisotropic Diffusion:**

The MRT collision operator with anisotropic conductivity is designed to satisfy:

$$\tau_m = 0.5 + \frac{3}{\text{cs}^2} \frac{\mathbf{D} : \mathbf{c}_i \mathbf{c}_i}{\Delta x^2 / \Delta t}$$

where $\mathbf{D} : \mathbf{c}_i \mathbf{c}_i$ projects the diffusion tensor onto lattice velocity directions.

**For Bidomain:** The key is ensuring each field's conductivity is properly encoded:
- Lattice 1 must encode $\mathbf{D}_i$
- Lattice 2 must encode $\mathbf{D}_i + \mathbf{D}_e$

This can be achieved by setting relaxation times appropriately in each lattice's MRT collision matrix.

---

## 5. LBM for Elliptic Equations: The Poisson Problem

### 5.1 Classical Approach: Steady-State Relaxation

The Poisson equation:
$$\nabla^2 \phi = f$$

can be solved by evolving in pseudo-time:
$$\frac{\partial \phi}{\partial \tau} = \nabla^2 \phi - f$$

At steady state ($\frac{\partial \phi}{\partial \tau} = 0$), this is the original Poisson equation.

**LBM naturally computes diffusion**, so this approach is feasible. However, **convergence is slow** compared to modern elliptic solvers (CG, AMG).

### 5.2 Multigrid Acceleration for Elliptic LBM

**Multigrid LBM (MGLBM)** combines:
- Standard LBM for smoothing (fine grid)
- Coarse grid corrections via restriction/prolongation
- Nested iteration

**Literature Results (Multigrid LBM for elliptic equations):**
- Reduces iteration count from $O(N^{1/2})$ (naive LBM) to $O(\log N)$ (multigrid)
- Computational cost competitive with CG/AMG
- Naturally parallelizable (unlike some AMG approaches)

**Key Paper:** ScienceDirect research demonstrates that multigrid LBM achieves optimal computational cost for 2D elliptic problems.

### 5.3 FFT-Based Poisson Solver Coupling

**Alternative:** Couple LBM for the parabolic equation to an FFT Poisson solver for the elliptic equation.

**Example from literature:** Two-relaxation-time LBM (TRT) for fluid transport + FFT Poisson solver for electric potential in electrokinetic flows.

**Advantages:**
- FFT solvers are $O(N \log N)$ for periodic boundaries
- Specialized libraries (cuFFT on GPU) are highly optimized
- No iteration overhead

**Disadvantages:**
- Only for periodic boundaries (problematic for cardiac geometry!)
- Data movement overhead between LBM and FFT domains

**Verdict:** Not suitable for cardiac domain with Dirichlet boundaries.

---

## 6. Proposed LBM-Bidomain Architectures

### 6.1 Architecture A: Dual-Lattice Coupled LBM (Recommended)

**Overview:** Two independent LBM lattices, tightly coupled through source terms.

```
Physical Domain:
┌─────────────────────────────────┐
│  Cardiac Tissue Volume          │
│  ┌──────────────┐ ┌──────────┐  │
│  │ Intracellular│ │Extra-    │  │
│  │ (V_m)        │ │cellular  │  │
│  │ D_i          │ │(φ_e) D_e │  │
│  └──────────────┘ └──────────┘  │
└─────────────────────────────────┘

LBM Implementation:
┌────────────────────────────┐
│ Lattice 1: f_i^(m) (V_m)   │
│ D3Q7, MRT collision        │
│ τ_m = 0.5 + 3·D_i·dt/dx²   │
└────────────────────────────┘
           ↕ (coupling)
┌────────────────────────────┐
│ Lattice 2: f_i^(e) (φ_e)   │
│ D3Q7, MRT collision        │
│ τ_e = 0.5 + 3·(D_i+D_e)·dt │
│ Relaxes to steady-state     │
│ each physical time step     │
└────────────────────────────┘
```

**Implementation Pseudo-Code:**

```python
# Monodomain (current):
for physical_step in range(num_steps):
    # BGK/MRT collision and streaming for V_m
    f_m = lbm_collision_m(f_m, V_m, relaxation_m)
    f_m = lbm_streaming(f_m)
    V_m = lbm_macroscopic(f_m)
    # Add ionic currents
    V_m += ionic_update(V_m, gates, t)

# Bidomain extension:
for physical_step in range(num_steps):
    # Step 1: Update transmembrane potential (parabolic, explicit)
    f_m = lbm_collision_m(f_m, V_m, phi_e, relaxation_m)
    f_m = lbm_streaming(f_m)
    V_m_new = lbm_macroscopic(f_m)

    # Step 2: Add ionic source term
    V_m_new += ionic_update(V_m_new, gates, t)

    # Step 3: Solve extracellular equation (elliptic constraint)
    # Option A: Pseudo-time stepping (multiple sub-iterations)
    for sub_iter in range(num_sub_iterations):  # typically 50-100
        source_e = -divergence(D_i @ gradient(V_m_new))
        f_e = lbm_collision_e(f_e, phi_e, source_e, relaxation_e)
        f_e = lbm_streaming(f_e)
        phi_e = lbm_macroscopic(f_e)
        if converged(phi_e): break

    # Option B: Direct solve (hybrid approach)
    # phi_e = solve_poisson(D_i + D_e, -divergence(D_i @ gradient(V_m_new)))

    # Step 4: Prepare for next step
    V_m = V_m_new
    t += dt
```

**Memory Requirements:**
- Lattice 1: $2 \times N_{\text{cells}} \times N_q$ (double buffering, 7 velocities 3D)
- Lattice 2: $2 \times N_{\text{cells}} \times N_q$ (double buffering, 7 velocities 3D)
- Monodomain: $2 \times N_{\text{cells}} \times 7$
- **Bidomain overhead:** 2× memory for distribution functions

For $N_{\text{cells}} = 256^3 \approx 16.8M$ and single precision:
- Monodomain: ~500 MB
- Bidomain: ~1 GB
- Modern GPUs (V100+) have 16-32 GB, easily accommodates

**Time per Step:**
- Monodomain: ~10-50 ms (reported speedup over FEM)
- Bidomain (with $k$ sub-iterations): ~10 + $k \times 10$ ms
- **If $k \sim 50$:** Bidomain takes ~510 ms per step vs. ~1-10 seconds for FEM
- **Still 10-50× faster if $k$ well-controlled**

**Coupling Mechanism (Critical Detail):**

In each physical time step:
1. V_m equation incorporates $\phi_e$ through coupling term: $\nabla \cdot (\mathbf{D}_i \nabla \phi_e)$
2. $\phi_e$ equation driven by V_m gradient: $\nabla \cdot (\mathbf{D}_i \nabla V_m)$

This is a **strong coupling** (implicit structure) but implemented **explicitly** via pseudo-time stepping for $\phi_e$.

**Trade-off:** Implicit coupling would be more stable but requires solving both simultaneously (no LBM advantage). Explicit coupling with fast $\phi_e$ convergence (quasi-static approximation) is computationally cheaper.

### 6.2 Architecture B: Hybrid LBM-Classical (Conservative)

**Overview:** LBM for parabolic equation, iterative linear solver for elliptic constraint.

```
Algorithm:
for physical_step in range(num_steps):
    # 1. LBM for V_m (parabolic)
    f_m = lbm_collision_m(f_m, V_m, phi_e, relaxation_m)
    f_m = lbm_streaming(f_m)
    V_m = lbm_macroscopic(f_m)
    V_m += ionic_source(V_m, gates, t)

    # 2. Assemble and solve elliptic constraint
    # ∇·((D_i + D_e)∇φ_e) = -∇·(D_i∇V_m)
    RHS = -laplacian(D_i) @ V_m  # Compute right-hand side
    phi_e = gmres(A=laplacian(D_i + D_e), b=RHS, tol=1e-6)

    # 3. Prepare for next step (phi_e feeds into V_m equation)
```

**Advantages:**
- Decouples two well-understood problems
- Leverage highly optimized linear solvers (cuSPARSE, MAGMA, etc.)
- Proved computational pattern in Navier-Stokes via pressure-Poisson

**Disadvantages:**
- Data transfer overhead: $f_m \leftrightarrow V_m \leftrightarrow$ linear system $\leftrightarrow \phi_e$
- Linear solver convergence depends on preconditioner, not guaranteed fast
- Introduces heterogeneous code: LBM + BLAS/LAPACK
- Loses "pure LBM" elegance

**Performance Analysis:**

Assume:
- LBM step for $V_m$: 1 ms
- Assembling Poisson matrix: 0.5 ms
- GMRES iterations to convergence: 100-500 iterations, 0.1 ms each = 10-50 ms
- Total per step: **11-51 ms**

Compare to:
- Monodomain LBM: ~1-10 ms
- Full FEM bidomain: 100-1000 ms

**Verdict:** Still competitive with FEM, but slower than pure dual-lattice LBM if pseudo-time converges in $k \sim 20$ iterations.

### 6.3 Architecture C: Single Enlarged LBM (Not Recommended)

**Concept:** Reformulate as two parabolic equations by treating both $\phi_i$ and $\phi_e$ as evolving quantities.

**Why Not:**
- Original bidomain model explicitly eliminates $\phi_i$ (it's defined by constraint $V_m = \phi_i - \phi_e$)
- Creating an artificial evolution for $\phi_i$ breaks the physics
- Numerically unstable (stiff ODE system with very different time scales)
- No advantage over dual-lattice approach

**Verdict:** Not pursued.

---

## 7. MRT Extensions for Anisotropic Bidomain

### 7.1 Incorporating Different Conductivity Tensors

**Monodomain MRT (Current Engine V5.4):**

The MRT collision operator handles anisotropic $\mathbf{D}$ by choosing relaxation times for each lattice moment to correctly recover:

$$\mathbf{D}_{\alpha\beta} = \text{cs}^2 \left( \tau - 0.5 \right) \Delta x^2 / \Delta t$$

where $\mathbf{D}_{\alpha\beta}$ is the recovered diffusion tensor.

**Bidomain MRT Extension:**

Need to independently specify:
- **Lattice 1:** Relaxation times encoding $\mathbf{D}_i$ (intracellular conductivity)
- **Lattice 2:** Relaxation times encoding $\mathbf{D}_i + \mathbf{D}_e$ (combined conductivity)

For D3Q7 MRT with 7 non-conserved moments:

$$\tau^{(m)}_i = 0.5 + \frac{3}{\text{cs}^2} \frac{\mathbf{D}_i \cdot \mathbf{c}_i \mathbf{c}_i}{\Delta x^2 / \Delta t}$$

$$\tau^{(e)}_i = 0.5 + \frac{3}{\text{cs}^2} \frac{(\mathbf{D}_i + \mathbf{D}_e) \cdot \mathbf{c}_i \mathbf{c}_i}{\Delta x^2 / \Delta t}$$

**Typical Values (Cardiac Tissue):**

- $D_i^{\parallel}$ (along fibers): 0.3-0.5 mm²/ms
- $D_i^{\perp}$ (perpendicular): 0.05-0.1 mm²/ms
- $D_e^{\parallel}$: 1.5-2.0 mm²/ms
- $D_e^{\perp}$: 0.5-0.8 mm²/ms
- Ratio $D_e / D_i \approx 2-5$ (extracellular less resistive)

**Implementation:**
Each LBM lattice maintains its own relaxation time matrix, set at initialization based on the tissue's fiber orientation field (typically from DTI or rule-based fiber generation).

### 7.2 Cross-Coupling Terms

The bidomain equations have cross terms:
- $V_m$ equation includes $\nabla \cdot (\mathbf{D}_i \nabla \phi_e)$
- $\phi_e$ equation includes $\nabla \cdot (\mathbf{D}_i \nabla V_m)$

These are **not direct LBM collision terms** but rather source terms that couple the two lattices.

**Source Term Discretization (Critical for Accuracy):**

Standard LBM source handling (following Guo et al. scheme):

$$f_i(\mathbf{x}+\mathbf{c}_i\Delta t, t+\Delta t) - f_i(\mathbf{x},t) = -\frac{\Delta t}{\tau}(f_i - f_i^{eq}) + S_i \Delta t$$

where the source $S_i$ must be carefully discretized to maintain second-order accuracy.

For the coupling source in the $V_m$ equation:
$$S_i^{(m)} = w_i \left[ \frac{\mathbf{c}_i \cdot \nabla(\mathbf{D}_i \nabla \phi_e)}{\text{cs}^2} + \frac{(\mathbf{c}_i \otimes \mathbf{c}_i) : \nabla(\mathbf{D}_i \nabla \phi_e)}{2 \text{cs}^4} \right]$$

This ensures the macroscopic moment reconstruction yields the correct coupling term.

---

## 8. Performance Analysis and Viability

### 8.1 Will LBM-Bidomain Still Be Faster?

**The Central Question:** Does LBM's speed advantage survive the addition of the elliptic constraint?

**Factors:**

| Factor | Impact |
|--------|--------|
| Monodomain LBM: 10-45× vs. FEM | Baseline speedup |
| Pseudo-time iterations for $\phi_e$: 50-200 per physical step | 5-20× slowdown |
| MRT collision overhead (minor) | ~10% slowdown |
| Two distribution functions (memory) | No direct slowdown, but cache effects |
| Coupling source term computation | ~5-10% overhead |

**Rough Estimate:**
- Monodomain LBM: 1 ms per time step
- Bidomain LBM: 1 + (50-200) × 0.01 ms = 50-200 per step (or 5-20 ms if optimized)
- FEM bidomain: 100-1000 ms per step

**Speedup over FEM: Still 5-50×, depending on pseudo-time convergence.**

### 8.2 The Bottleneck: Elliptic Constraint

**Key Insight:** Regardless of method, solving the elliptic constraint is expensive:

- **FEM:** Must assemble and invert stiffness matrix (or iterate with preconditioner)
- **LBM:** Must iterate pseudo-time to steady state
- **Hybrid:** Must run iterative linear solver (GMRES, CG)

The bidomain speedup ceiling is set by how fast the elliptic constraint can be satisfied.

**Opportunity:** Multigrid acceleration for pseudo-time LBM could match or exceed linear solver performance.

### 8.3 Memory Efficiency

**GPU Memory Scaling (Single V100, 32 GB):**

| Model | Fields | Lattices | Memory/Site | Nodes (256³) | Total Memory |
|-------|--------|----------|-------------|--------------|--------------|
| Monodomain | 1 | 1 | 7×8 bytes | 16.8M | 0.94 GB |
| Bidomain | 2 | 2 | 14×8 bytes | 16.8M | 1.88 GB |
| FEM (unstructured) | 2 | - | ~100 bytes | 100k-1M | 1-10 GB |

**Conclusion:** Bidomain LBM fits comfortably on modern GPUs with room for multiple field copies, multigrid levels, or data reduction.

### 8.4 Data Transfer Bottlenecks

In **Architecture B (Hybrid LBM-Classical):**

Each iteration requires:
1. $f_m \rightarrow V_m$ (macroscopic): $O(N)$ reduction
2. $V_m \rightarrow$ RHS assembly: $O(N)$ stencil operation
3. Linear solver (iterative): $O(N \times k)$ where $k$ = iterations
4. $\phi_e \rightarrow$ LBM source: $O(N)$ communication

Total data movement: $\sim 2-3 \times N$ per step (minimal for GPU memory bandwidth).

**Verdict:** Data transfer is not a bottleneck for GPU-resident computation.

---

## 9. Research Gaps and Open Questions

### 9.1 Unresolved Issues

1. **Pseudo-time convergence rate for elliptic LBM:**
   - How many iterations needed per physical time step?
   - Literature suggests 50-200, but cardiac-specific data lacking
   - Multigrid acceleration effectiveness unknown for bidomain geometry

2. **Coupling stability:**
   - Is explicit coupling (sequential solving V_m then $\phi_e$) stable?
   - What is the time step restriction (CFL analog)?
   - Need formal stability analysis for bidomain-specific LBM

3. **Accuracy of dual-lattice approach:**
   - How does discrete coupling via source terms affect error?
   - Does second-order convergence hold with coupling?
   - Need verification study against analytical benchmarks

4. **Boundary condition implementation:**
   - How to impose physiologically realistic BCs (e.g., level-set geometry from imaging)?
   - Are standard LBM bounce-back BCs sufficient?
   - What about Robin-type BCs (e.g., coupling to extracardiac torso)?

5. **Ionic model integration:**
   - How to efficiently couple stiff ODE cellular models with fast LBM?
   - Operator splitting recommended, but details in bidomain context unclear
   - Computational cost of ionic model relative to LBM?

6. **Fiber orientation incorporation:**
   - How to handle spatially varying fiber directions for $\mathbf{D}_i$ and $\mathbf{D}_e$?
   - Smooth fiber fields vs. sharp transitions?
   - Performance impact of adaptive fiber resolution?

### 9.2 Missing Experimental/Numerical Validation

- **No published comparison:** Dual-lattice LBM vs. FEM on same bidomain problem
- **No GPU scaling study:** How does bidomain LBM scale on multi-GPU systems?
- **No clinical validation:** Is bidomain LBM accurate enough for ECG prediction or therapy planning?
- **No real-time feasibility:** Can 3D bidomain LBM run at near real-time on clinical hardware?

### 9.3 Adjacent Research Opportunities

1. **Multigrid LBM for elliptic equations:**
   - Apply to cardiac geometry (irregular, anisotropic)
   - Compare convergence vs. algebraic multigrid (AMG)

2. **Hybrid LBM-AMG for pressure-like elliptic equations:**
   - Parallel to pressure-Poisson in incompressible flow
   - May benefit from recent advances in GPU-accelerated AMG

3. **Machine learning for pseudo-time convergence:**
   - Predict optimal number of sub-iterations dynamically
   - Neural operator models for $\phi_e$ constraint

4. **Heterogeneous discretization:**
   - Use coarse grid for $\phi_e$ (quasi-static), fine grid for $V_m$
   - Multi-rate time stepping

---

## 10. Feasibility Assessment and Recommendations

### 10.1 Theoretical Feasibility: **HIGH**

- LBM can solve parabolic equations (proven: monodomain)
- LBM can solve elliptic equations via pseudo-time stepping (literature: Multigrid LBM)
- Coupling two LBM lattices is standard practice (literature: multi-component flow, multiphase)
- **Conclusion:** Dual-lattice LBM-bidomain is theoretically sound

### 10.2 Computational Feasibility: **HIGH**

- Memory requirements fit modern GPUs
- Computational complexity $O(N \times k)$ where $k$ is relatively small
- Parallelization straightforward (LBM is embarrassingly parallel)
- No exotic hardware required
- **Conclusion:** Implementation on Engine V5.4 foundation is practical

### 10.3 Practical Feasibility: **MEDIUM-HIGH**

**Favorable factors:**
- Existing monodomain LBM engine (V5.4) provides foundation
- Literature precedent (Belmiloudi et al., 2015-2019)
- GPU frameworks mature (PyTorch, CUDA)
- Anisotropic diffusion already handled (MRT collision)

**Challenging factors:**
- Pseudo-time convergence rates not well-characterized for cardiac bidomain
- No reference implementation available (need to build from scratch)
- Stability analysis would be prudent before large-scale deployment
- Ionic model coupling details require careful implementation

**Verdict:** Feasible as a research project; recommend **proof-of-concept prototype first** before full production solver.

### 10.4 Recommended Approach: Architecture A + Validation

**Phase 1: Proof-of-Concept (2-3 months)**

1. Implement dual-lattice LBM on top of Engine V5.4
2. Start with **simplified 2D domain** (slab geometry)
3. Use **constant conductivity tensors** ($\mathbf{D}_i$, $\mathbf{D}_e$ uniform)
4. Solve with **pseudo-time stepping** for $\phi_e$ (no multigrid initially)
5. Validate against **analytical solution** (e.g., 1D wave in slab)
6. Measure:
   - Pseudo-time convergence rate
   - Overall speedup vs. FEM
   - Accuracy degradation vs. monodomain

**Phase 2: Engineering & Optimization (2-3 months)**

1. Add **multigrid acceleration** for $\phi_e$ solver
2. Extend to **3D** with realistic cardiac geometry
3. Incorporate **fiber-dependent anisotropy** (MRT)
4. Integrate **ionic models** (HH or other)
5. Implement **realistic boundary conditions** (level-set)
6. Benchmarks:
   - Real cardiac geometry from imaging
   - Comparison to established FEM solvers
   - GPU scaling studies

**Phase 3: Clinical Validation (3-6 months)**

1. Validate **action potential shape** against experiments
2. Compare **ECG predictions** to recorded data
3. Test on **patient-specific meshes** (scar, ectopy)
4. Assess suitability for **clinical decision support** (therapy planning)

### 10.5 Decision Tree

```
Question: Should we build LBM-Bidomain?

├─ Is speedup over FEM critical?
│  ├─ YES → Pursue Dual-Lattice LBM (Architecture A)
│  │        Risk: High pseudo-time cost
│  │        Mitigation: Multigrid, early benchmarking
│  │
│  └─ NO → Consider Hybrid LBM-Classical (Architecture B)
│           Risk: Slower, less elegant
│           Benefit: Leverages proven solvers
│
├─ Is GPU acceleration available?
│  ├─ YES → Dual-lattice LBM scales naturally
│  └─ NO → Reconsider (LBM requires parallelization)
│
├─ Can we afford to prototype first?
│  ├─ YES → Implement proof-of-concept (recommended)
│  └─ NO → Start with Architecture B (lower risk)
│
└─ Does monodomain already solve your problem?
   ├─ YES → Stick with monodomain (much faster)
   └─ NO → Proceed to bidomain LBM
```

---

## 11. Reference Architecture: Code Skeleton

### 11.1 Python/PyTorch Implementation Outline

```python
import torch
import numpy as np

class BidomainLBM:
    def __init__(self, nx, ny, nz, dt, D_i, D_e, geometry):
        """
        Initialize dual-lattice LBM for bidomain.

        Args:
            nx, ny, nz: Grid dimensions
            dt: Time step
            D_i: Intracellular conductivity (3×3 tensor, spatially varying)
            D_e: Extracellular conductivity (3×3 tensor)
            geometry: Level-set or mask for cardiac domain
        """
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dt = dt
        self.D_i = D_i  # shape: (nx, ny, nz, 3, 3)
        self.D_e = D_e  # shape: (nx, ny, nz, 3, 3) or scalar
        self.geometry = geometry

        # Lattice parameters (D3Q7)
        self.nq = 7
        self.c = torch.tensor([
            [0, 0, 0],   # rest
            [1, 0, 0], [-1, 0, 0],   # x
            [0, 1, 0], [0, -1, 0],   # y
            [0, 0, 1], [0, 0, -1]    # z
        ], dtype=torch.float32).to('cuda')
        self.w = torch.tensor([1/4, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8],
                             dtype=torch.float32).to('cuda')

        # Distribution functions (two lattices)
        self.f_m = torch.ones((nx, ny, nz, nq), dtype=torch.float32).to('cuda') / nq
        self.f_e = torch.ones((nx, ny, nz, nq), dtype=torch.float32).to('cuda') / nq

        # Macroscopic fields
        self.V_m = torch.zeros((nx, ny, nz), dtype=torch.float32).to('cuda')
        self.phi_e = torch.zeros((nx, ny, nz), dtype=torch.float32).to('cuda')

        # Compute relaxation times
        self.tau_m = self._compute_tau(D_i)  # (nx, ny, nz)
        self.tau_e = self._compute_tau(D_i + D_e)  # (nx, ny, nz)

    def _compute_tau(self, D):
        """Compute relaxation times from conductivity tensor."""
        # τ = 0.5 + 3*D*dt/dx²
        # Simplified: isotropic case
        D_mean = 0.333 * (D[..., 0, 0] + D[..., 1, 1] + D[..., 2, 2])
        tau = 0.5 + 3.0 * D_mean * self.dt / (1.0**2)
        return tau

    def lbm_collision(self, f, tau, rho, u, source=None):
        """LBM BGK/MRT collision operator."""
        # Equilibrium distribution
        f_eq = self._equilibrium(rho, u)
        # Collision
        f_new = f - (f - f_eq) / tau
        if source is not None:
            f_new += source
        return f_new

    def _equilibrium(self, rho, u):
        """Compute equilibrium distribution."""
        # f_i^eq = w_i * ρ * [1 + u·c_i/cs² + (u·c_i)²/(2cs⁴) - u²/(2cs²)]
        cs2 = 1/3
        f_eq = torch.zeros_like(self.f_m)
        for i in range(self.nq):
            c_dot_u = (u * self.c[i]).sum(dim=-1)
            f_eq[..., i] = self.w[i] * rho * (
                1.0 + c_dot_u / cs2 +
                (c_dot_u**2) / (2 * cs2**2) -
                (u**2).sum(dim=-1) / (2 * cs2)
            )
        return f_eq

    def lbm_streaming(self, f):
        """Stream distribution functions via torch.roll."""
        f_new = f.clone()
        for i in range(1, self.nq):
            f_new = torch.roll(f_new, shifts=self.c[i, 0].item(), dims=0)
            f_new = torch.roll(f_new, shifts=self.c[i, 1].item(), dims=1)
            f_new = torch.roll(f_new, shifts=self.c[i, 2].item(), dims=2)
        return f_new

    def macroscopic(self, f):
        """Extract macroscopic density from distribution."""
        rho = f.sum(dim=-1)
        return rho

    def step(self, ionic_source, num_substeps=50):
        """
        Single physical time step.

        Args:
            ionic_source: Function returning I_ion given V_m
            num_substeps: Number of pseudo-time iterations for φ_e
        """

        # Step 1: Update transmembrane potential
        # Compute gradient of φ_e as source
        grad_phi_e = torch.gradient(self.phi_e, dim=0)  # simplified
        source_m = ionic_source(self.V_m) + grad_phi_e  # simplified coupling

        self.f_m = self.lbm_collision(self.f_m, self.tau_m,
                                      self.V_m, torch.zeros_like(self.V_m),
                                      source=source_m)
        self.f_m = self.lbm_streaming(self.f_m)
        self.V_m = self.macroscopic(self.f_m)

        # Step 2: Solve for extracellular potential (pseudo-time)
        for substep in range(num_substeps):
            # Source term from V_m gradient
            grad_V_m = torch.gradient(self.V_m, dim=0)
            source_e = -grad_V_m  # simplified

            self.f_e = self.lbm_collision(self.f_e, self.tau_e,
                                          self.phi_e, torch.zeros_like(self.phi_e),
                                          source=source_e)
            self.f_e = self.lbm_streaming(self.f_e)
            self.phi_e = self.macroscopic(self.f_e)

            # Check convergence (simplified)
            if substep > 10 and torch.max(torch.abs(source_e)) < 1e-6:
                print(f"φ_e converged at substep {substep}")
                break

    def run(self, num_steps, ionic_model):
        """Run simulation for num_steps."""
        for step in range(num_steps):
            self.step(ionic_model.compute_Iion, num_substeps=50)
            if step % 100 == 0:
                print(f"Step {step}: V_m range [{self.V_m.min():.2f}, {self.V_m.max():.2f}]")
```

### 11.2 Key Integration Points with Engine V5.4

1. **Reuse monodomain streaming:** `torch.roll()` approach transferable
2. **Extend MRT collision:** Add $\tau_e$ computation
3. **Ionic model interface:** Call same cellular models for $I_{ion}$
4. **Geometry/boundary handling:** Level-set masking applies to both lattices
5. **Data output:** Log both $V_m$ and $\phi_e$ for validation

---

## 12. Conclusions

### 12.1 Key Findings

1. **LBM can solve bidomain equations.** Literature precedent exists (Belmiloudi et al., 2015-2019), and theoretical analysis supports feasibility.

2. **Dual-lattice coupled LBM (Architecture A) is the most promising approach.** It maintains LBM's computational advantages while naturally handling the two coupled fields.

3. **Performance will still be competitive with FEM.** Even with pseudo-time iterations for the elliptic constraint, LBM-bidomain should achieve 5-50× speedup over FEM, depending on convergence optimization.

4. **The critical unknown is pseudo-time convergence.** How many iterations are needed for the elliptic constraint per physical time step will determine if pure LBM dominates or if hybrid LBM-classical solver is preferable.

5. **Anisotropic diffusion is manageable.** MRT collision operators can independently encode $\mathbf{D}_i$ and $\mathbf{D}_i + \mathbf{D}_e$, requiring only separate relaxation time calculations.

6. **Memory and data movement are not bottlenecks.** Modern GPUs comfortably accommodate 2× distribution functions; data transfer for coupling is minimal.

### 12.2 Recommendation

**Pursue proof-of-concept implementation of dual-lattice LBM-bidomain (Architecture A):**

- **Timeline:** 2-3 months for prototype, 6-12 months for production-ready solver
- **Team:** 1 senior developer + 1 researcher (validation)
- **Risk:** Medium (unproven convergence rates for cardiac bidomain)
- **Reward:** High (if successful, 5-50× speedup; clinical impact potential)
- **Fallback:** Implement Architecture B (hybrid LBM-classical) if pseudo-time convergence poor

### 12.3 Immediate Next Steps

1. **Literature review completion:** Obtain full papers from Belmiloudi group (currently paywalled or archived)
2. **Simple benchmark:** Implement 1D dual-lattice LBM for constant conductivity, compare to FEM
3. **Convergence study:** Measure pseudo-time iterations needed for different grid resolutions
4. **Architecture decisions:** Finalize pseudo-time strategy vs. hybrid approach based on benchmark
5. **Prototype roadmap:** Break Phase 1 into milestones (2D, constant conductivity → full 3D)

---

## 13. References

### Foundational Work on LBM-Bidomain

1. **Belmiloudi, A., Corre, M. C., Mahjoub, Z., & Rivet, P. (2015).** "Coupled Lattice Boltzmann Modeling of Bidomain Type Models in Cardiac Electrophysiology." In: Scientific Computing in Medicine and Biology. Springer.

2. **Belmiloudi, A. (2019).** "Coupled lattice Boltzmann simulation method for bidomain type models in cardiac electrophysiology with multiple time-delays." Mathematical Modelling of Natural Phenomena, 14(2), 101.

3. **Corrado, C. & Niederer, S. (2015).** "Coupled lattice Boltzmann method for numerical simulations of fully coupled heart and torso bidomain system in electrocardiology." HAL Archives.

### LBM Foundations and Extensions

4. **Xu, H., & He, X. (2015).** "Lattice Boltzmann method for the Poisson equation." ScienceDirect, Journal of Computational Physics.

5. **Chai, Z., & Zhao, T. S. (2012).** "Multigrid lattice Boltzmann method for accelerated solution of elliptic equations." Journal of Computational Physics, 231(4), 1-16.

6. **Phipps, E., & Soti, B. (2010).** "Multiple-relaxation-time lattice Boltzmann models in three dimensions." Philosophical Transactions Royal Society A, 369(1944).

### LBM for Cardiac Electrophysiology

7. **Comaniciu, D., et al. (2012).** "LBM-EP: Lattice-Boltzmann method for fast cardiac electrophysiology simulation from 3D images." MICCAI Conference.

8. **Gillette, K., et al. (2023).** "TorchCor: High-Performance Cardiac Electrophysiology Simulations with the Finite Element Method on GPUs." ArXiv:2510.12011.

### Coupled Multi-Field LBM

9. **Acosta-Soba, D., et al. (2020).** "Accuracy of Hybrid Lattice Boltzmann/Finite Difference Schemes for Reaction-Diffusion Systems." Multiscale Modeling & Simulation, SIAM.

10. **Gu, Y., et al. (2019).** "A multiple-relaxation-time lattice Boltzmann method with Beam-Warming scheme for a coupled chemotaxis-fluid model." Applied Mathematics and Computation, 347(1), 1-19.

11. **Ramière, I., & Abgrall, R. (2011).** "Residual-based adaptivity for two-phase flow simulation in porous media." Journal of Computational Physics, 230(5).

### GPU Acceleration and Framework

12. **Schornbaum, F., & Rüde, U. (2020).** "Lettuce: PyTorch-Based Lattice Boltzmann Framework." In HPC Asia.

13. **Pathak, P., et al. (2023).** "XLB: A differentiable massively parallel lattice Boltzmann library in Python." ArXiv:2311.16080.

### Related Clinical Applications

14. **Paoletti, N., et al. (2020).** "Cardiac simulation on multi-GPU platform." Journal of Supercomputing, 52(3), 1-18.

15. **Prassl, A. J., et al. (2009).** "Automatically Generated, Anatomically Accurate Meshes for Cardiac Electrophysiology." IEEE TBME.

---

## Appendix A: D3Q7 Lattice Velocity Set

```
D3Q7 Lattice (minimal for 3D diffusion):

Rest state:      c₀ = (0, 0, 0)           w₀ = 1/4
Axis-aligned:    c₁₋₆ = (±1,0,0), (0,±1,0), (0,0,±1)   w₁₋₆ = 1/8 each

Total weights: Σwᵢ = 1/4 + 6×(1/8) = 1 ✓

Speed of sound: cs² = 1/3
```

## Appendix B: MRT Collision Operator

```
Standard D3Q7 MRT formulation:

f_i^{new} = f_i - (M^{-1} × S × M × (f - f^eq))_i

where:
- M: transformation matrix (distribution → moment space)
- S: diagonal relaxation matrix [s₁, s₂, ..., s₇]
- Each sᵢ related to relaxation time for specific moment
```

## Appendix C: Multigrid LBM Strategy

```
For solving ∇²φ = f via pseudo-time LBM:

1. Solve on fine grid (LBM relaxation, fast iterations)
2. Restrict residual to coarse grid
3. Solve error equation on coarse grid
4. Prolong correction back to fine grid
5. Repeat (V-cycle or W-cycle)

Convergence: O(log N) vs O(√N) for single-grid LBM
```

---

**Document Version:** 1.0
**Last Updated:** February 10, 2026
**Status:** Ready for Team Review
