# Comprehensive Research Document: Solver Methods for Cardiac Bidomain Equations

**Author:** Computational Cardiac Electrophysiology Research
**Date:** February 2025
**Version:** 1.0
**Focus:** Time-Stepping and Operator Splitting Methods

---

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Formulation](#mathematical-formulation)
3. [Operator Splitting for Bidomain](#operator-splitting-for-bidomain)
4. [Time Integration Schemes](#time-integration-schemes)
5. [Linear System Structure](#linear-system-structure)
6. [Decoupled vs Coupled Approaches](#decoupled-vs-coupled-approaches)
7. [The Extracellular Equation](#the-extracellular-equation)
8. [Comparison with Monodomain Solver Architecture](#comparison-with-monodomain-solver-architecture)
9. [Performance Considerations](#performance-considerations)
10. [Implementation Architecture](#implementation-architecture)
11. [References](#references)

---

## Introduction

The cardiac bidomain model is one of the most complete mathematical descriptions of electrical activity in cardiac tissue, accounting for both intracellular and extracellular domains. Unlike the monodomain simplification, which reduces the problem to a single parabolic PDE with one unknown (membrane potential V), the bidomain formulation introduces coupled unknowns: the transmembrane potential (Vm) and the extracellular potential (φe).

This fundamental difference creates significant computational challenges:

- **Complexity increase:** 2 coupled unknowns instead of 1
- **Different PDE types:** 1 parabolic equation (Vm) + 1 elliptic equation (φe)
- **Linear system size:** 2N × 2N instead of N × N (where N = number of nodes)
- **Computational cost:** Bidomain is 10-20x more expensive than monodomain for the same mesh with optimal solvers
- **Linear solver bottleneck:** The elliptic equation for φe requires an expensive linear solve at each time step

This document provides a comprehensive review of numerical methods for solving the bidomain equations, with emphasis on the solver architecture that could extend our existing monodomain Engine (V5.4) to handle bidomain problems.

---

## Mathematical Formulation

### The Bidomain System

The bidomain model is typically written as a system of coupled parabolic and elliptic PDEs:

**Transmembrane Potential (Parabolic Equation):**
$$\chi C_m \frac{\partial V_m}{\partial t} + \chi I_{ion}(V_m, \mathbf{w}) = \nabla \cdot (\mathbf{D}_i \nabla V_m) + \nabla \cdot (\mathbf{D}_i \nabla \varphi_e)$$

**Extracellular Potential (Elliptic Equation):**
$$0 = \nabla \cdot ((\mathbf{D}_i + \mathbf{D}_e) \nabla \varphi_e) + \nabla \cdot (\mathbf{D}_i \nabla V_m)$$

**Ionic Gate Equations (ODE):**
$$\frac{d\mathbf{w}}{dt} = \mathbf{f}(V_m, \mathbf{w})$$

### Notation and Parameters

| Symbol | Meaning |
|--------|---------|
| $V_m$ | Transmembrane potential (V) |
| $\varphi_e$ | Extracellular potential (V) |
| $\mathbf{w}$ | Vector of ionic gate variables (gating variables) |
| $\chi$ | Surface-to-volume ratio (cm⁻¹) |
| $C_m$ | Membrane capacitance (μF/cm²) |
| $I_{ion}$ | Ionic current density (μA/cm²) |
| $\mathbf{D}_i, \mathbf{D}_e$ | Intracellular and extracellular conductivity tensors (mS/cm) |
| $t$ | Time (ms) |

### Key Structural Properties

1. **The first equation is parabolic in time:** It has a time derivative for Vm and requires time integration
2. **The second equation is elliptic:** No time derivative; it acts as an algebraic constraint that must be satisfied at every instant
3. **Strong coupling:** Changes in Vm directly affect φe through the diffusion terms, and vice versa
4. **Stiffness from ionic gates:** The ODEs for ionic gates often have widely different time scales (milliseconds to seconds)

---

## Operator Splitting for Bidomain

Operator splitting is one of the most practical approaches for solving the coupled bidomain system. It decomposes the problem into simpler subproblems that can be solved independently or with simpler algorithms.

### Historical Background

The first- and second-order operator splitting approaches are standard in cardiac electrophysiology:

- **Godunov splitting:** First-order temporal accuracy, also known as Lie-Trotter splitting
- **Strang splitting:** Second-order temporal accuracy, also known as Strang-Marchuk splitting

These methods have been successfully extended from monodomain to bidomain equations, as documented in the literature.

### Extension to Bidomain: Conceptual Framework

In the monodomain model, operator splitting typically separates the reaction (ionic) step from the diffusion step. For bidomain, the principle extends naturally:

**Reaction Step:**
- Solve only the ionic ODEs: $\frac{d\mathbf{w}}{dt} = \mathbf{f}(V_m, \mathbf{w})$
- Update Vm explicitly using a simple integrator (e.g., Forward Euler, Rush-Larsen)
- φe is decoupled (not solved in this step)

**Diffusion Step:**
- Solve the coupled elliptic-parabolic system for (Vm, φe)
- This requires solving a 2×2 block linear system
- The ionic current Iion is held constant (explicit treatment from the previous reaction step)

### Godunov Splitting (1st Order)

The Godunov operator splitting for bidomain follows the classic fractional-step approach:

**Algorithm (from time n to time n+1):**

1. **Reaction Sub-step** (from tn to tn + Δt):
   - Advance only ionic dynamics:
     $$V_m^*, \mathbf{w}^* = \text{ExplicitIntegrator}(V_m^n, \mathbf{w}^n, \Delta t)$$
   - The transmembrane potential is updated based on ionic currents only
   - φe is NOT solved in this step

2. **Diffusion Sub-step** (from tn + Δt to tn+1 = tn + 2Δt):
   - Solve the coupled system with constant ionic current:
     $$\chi C_m \frac{V_m^{n+1} - V_m^*}{\Delta t} = \nabla \cdot (\mathbf{D}_i \nabla V_m^{n+1}) + \nabla \cdot (\mathbf{D}_i \nabla \varphi_e^{n+1}) + \chi I_{ion}(V_m^*, \mathbf{w}^*)$$
     $$0 = \nabla \cdot ((\mathbf{D}_i + \mathbf{D}_e) \nabla \varphi_e^{n+1}) + \nabla \cdot (\mathbf{D}_i \nabla V_m^{n+1})$$
   - This is a coupled 2×2 system for (Vm^{n+1}, φe^{n+1})

**Temporal Accuracy:** O(Δt) — first order

**Advantage:** Simplicity; each step is conceptually straightforward

**Disadvantage:** Lower accuracy requires smaller time steps for adequate precision

### Strang Splitting (2nd Order)

The Strang operator splitting improves accuracy by using a symmetric decomposition:

**Algorithm (from time n to time n+1):**

1. **Reaction Sub-step** (from tn to tn + Δt/2):
   - Advance ionic dynamics for half the time step:
     $$V_m^*, \mathbf{w}^* = \text{ExplicitIntegrator}(V_m^n, \mathbf{w}^n, \Delta t/2)$$

2. **Diffusion Sub-step** (from tn + Δt/2 to tn + Δt):
   - Solve the coupled diffusion system for the full time step:
     $$\chi C_m \frac{V_m^{**} - V_m^*}{\Delta t} = \nabla \cdot (\mathbf{D}_i \nabla V_m^{**}) + \nabla \cdot (\mathbf{D}_i \nabla \varphi_e^{**}) + \chi I_{ion}(V_m^*, \mathbf{w}^*)$$
     $$0 = \nabla \cdot ((\mathbf{D}_i + \mathbf{D}_e) \nabla \varphi_e^{**}) + \nabla \cdot (\mathbf{D}_i \nabla V_m^{**})$$

3. **Reaction Sub-step** (from tn + Δt to tn+1):
   - Advance ionic dynamics for another half time step:
     $$V_m^{n+1}, \mathbf{w}^{n+1} = \text{ExplicitIntegrator}(V_m^{**}, \mathbf{w}^*, \Delta t/2)$$

**Temporal Accuracy:** O(Δt²) — second order

**Advantage:** Better accuracy; allows larger time steps for the same precision

**Disadvantage:** Slightly more expensive than Godunov (3 reaction sub-steps per full step vs 1); still sequential

### Why Splitting Works for Bidomain

The bidomain equations have a natural decomposition structure:

- **Reaction (ionic) step:** Decoupled from diffusion; depends only on local values of Vm and w
- **Diffusion step:** Linear in the unknowns (after treating Iion as source term); requires solving an elliptic problem

This structure makes operator splitting particularly attractive because:
1. The ionic solver can use specialized, efficient methods (e.g., Rush-Larsen exponential integrator)
2. The diffusion step, though coupled, becomes linear and solvable with sparse linear algebra
3. The method is naturally parallelizable on unstructured meshes

### Stability Considerations

Research has shown that **Strang splitting has better stability properties than Godunov splitting for the bidomain equations.** This is because the symmetric arrangement of reaction-diffusion-reaction reduces the splitting error that would otherwise propagate.

---

## Time Integration Schemes

Beyond operator splitting, various time integration approaches are available for the bidomain equations, with different trade-offs between accuracy, stability, and computational cost.

### IMEX (Implicit-Explicit) Methods

IMEX schemes treat different terms with different implicit/explicit discretizations, balancing stability and computational cost.

#### General Framework

For a system of the form:
$$\frac{d\mathbf{u}}{dt} = \mathbf{f}_{explicit}(\mathbf{u}) + \mathbf{f}_{implicit}(\mathbf{u})$$

IMEX methods integrate one part explicitly and the other implicitly:
- **Explicit part:** Usually the nonlinear ionic current Iion (less expensive)
- **Implicit part:** Usually the diffusion operator (requires linear solve; more stable)

#### Advantages for Bidomain

1. **Stability:** Implicit treatment of stiff diffusion allows larger time steps
2. **Computational efficiency:** Avoid nonlinear solves in diffusion step; only linear systems required
3. **Flexibility:** Can adjust explicit/implicit balance based on problem stiffness

#### Example: Semi-Implicit IMEX

A common semi-implicit approach for bidomain:

**Transmembrane potential equation (Crank-Nicolson in space, implicit diffusion, explicit ionic):**
$$\chi C_m \frac{V_m^{n+1} - V_m^n}{\Delta t} + \chi I_{ion}(V_m^n, \mathbf{w}^n) = \frac{1}{2}\nabla \cdot (\mathbf{D}_i \nabla (V_m^{n+1} + V_m^n)) + \frac{1}{2}\nabla \cdot (\mathbf{D}_i \nabla (\varphi_e^{n+1} + \varphi_e^n))$$

**Extracellular potential equation (unchanged):**
$$0 = \nabla \cdot ((\mathbf{D}_i + \mathbf{D}_e) \nabla \varphi_e^{n+1}) + \nabla \cdot (\mathbf{D}_i \nabla V_m^{n+1})$$

**Ionic gates (explicit or Rush-Larsen):**
$$\mathbf{w}^{n+1} = \text{RushLarsen}(\mathbf{w}^n, V_m^n, \Delta t)$$

**Characteristics:**
- Time step can be 100+ times larger than purely explicit methods
- Coupling between Vm and φe remains nonlinear in the implicit part
- Typically solved with Newton's method or fixed-point iteration

### Fully Implicit Methods

Treat all terms implicitly; solve a fully coupled nonlinear system at each time step.

#### Advantages
- Maximum stability for stiff systems
- Unconditionally stable schemes (e.g., Backward Euler, Crank-Nicolson)
- Insensitive to the diffusion coefficient magnitude

#### Disadvantages
- Expensive Newton solver with Jacobian computation
- Large coupled nonlinear systems (dimension = 2N where N = number of nodes)
- Requires robust preconditioner

#### When to Use
- Very stiff ionic models with millisecond and second time scales
- Large diffusion coefficients creating stiffness
- When high accuracy is essential and computational cost is not limiting

### Crank-Nicolson for the Parabolic Part

The Crank-Nicolson method is a second-order, unconditionally stable scheme that averages forward and backward Euler:

For the parabolic Vm equation:
$$\chi C_m \frac{V_m^{n+1} - V_m^n}{\Delta t} = \frac{1}{2}[\mathcal{L}(V_m^{n+1}) + \mathcal{L}(V_m^n)]$$

where $\mathcal{L}$ represents the spatial differential operators and source terms.

**Advantages:**
- Second-order accuracy in time
- Unconditionally stable
- Good damping of high-frequency oscillations in implicit schemes

**Disadvantages:**
- Can produce spurious oscillations for certain parameter values (mitigated by SDIRK methods)
- Requires implicit solve at each time step

**Comparison with Backward Differentiation Formulas (BDF):**
- **Crank-Nicolson (CN):** Optimal for smooth solutions; may oscillate for steep fronts
- **BDF1 (Backward Euler):** Very stable but only first-order accurate
- **BDF2:** Second-order, slightly less smooth damping than CN, but more robust for discontinuities

Research shows that **SDIRK methods (Singly Diagonally Implicit Runge-Kutta) eliminate spurious oscillations while maintaining second-order accuracy** and low computational cost.

### SDIRK Methods (Singly Diagonally Implicit Runge-Kutta)

SDIRK methods are particularly suited for stiff systems with reaction-diffusion coupling.

#### Two-Stage L-Stable SDIRK2

**Butcher tableau:**
```
c | A
--|---
  | b^T

Typical SDIRK2:
λ  | λ    0
1  | 1-λ  λ
---|--------
   | 1-λ  λ
```

where λ ≈ 0.2929 for L-stability (all eigenvalues inside unit circle).

**Advantages:**
- **Elimination of spurious oscillations:** Unlike Crank-Nicolson, SDIRK2 avoids unphysical oscillations in transmembrane potential
- **L-stability:** Damps high-frequency modes completely, important for ionic currents with fast dynamics
- **Second-order accuracy:** Competitive with Crank-Nicolson in accuracy
- **Minimal overhead:** Only slightly more expensive than Crank-Nicolson

**Disadvantage:**
- Two implicit stages per time step (vs one for CN)

**Practical Outcome:** For bidomain simulations, SDIRK2 is often the "sweet spot" balancing accuracy, stability, and computational cost.

### Rosenbrock Methods

Rosenbrock methods use Jacobian information to achieve high-order accuracy with reduced computational cost.

**Structure:**
$$k_i = f(t_n + c_i \Delta t, y_n + \sum_{j=1}^i a_{ij} k_j) + \gamma_i \Delta t J(t_n, y_n) k_i$$

where J is the Jacobian matrix.

**Advantages:**
- Can achieve 3rd-4th order accuracy
- Particularly efficient for problems with accurate/cheap Jacobians
- Natural handling of stiff terms

**Disadvantages:**
- Expensive Jacobian computation for bidomain (2N × 2N matrix)
- May not be cost-effective compared to SDIRK for typical cardiac simulations
- Requires careful linearization of nonlinear ionic terms

---

## Linear System Structure

### The Discrete Bidomain System

After spatial discretization (typically finite elements) and temporal discretization, the bidomain system at time step n+1 reduces to solving a **coupled 2×2 block linear system**:

$$\begin{bmatrix}
A_{11} & A_{12} \\
A_{21} & A_{22}
\end{bmatrix}
\begin{bmatrix}
V_m^{n+1} \\
\varphi_e^{n+1}
\end{bmatrix}
=
\begin{bmatrix}
b_1 \\
b_2
\end{bmatrix}$$

where:
- $V_m^{n+1}, \varphi_e^{n+1} \in \mathbb{R}^N$ (N = number of mesh nodes)
- The full system is of size 2N × 2N
- This is the fundamental bottleneck in bidomain computations

### Block Structure for Semi-Implicit Crank-Nicolson

For the semi-implicit IMEX approach with Crank-Nicolson in space:

**A11 block** (from the Vm equation):
$$A_{11} = \frac{\chi C_m}{\Delta t} M + \frac{1}{2} K_i$$

where:
- M is the mass matrix (from finite element assembly)
- $K_i$ is the stiffness matrix from the $\nabla \cdot (\mathbf{D}_i \nabla V_m)$ term
- $\frac{\chi C_m}{\Delta t}$ is the temporal scaling

**A12 block** (coupling from φe to Vm equation):
$$A_{12} = \frac{1}{2} K_i$$

This comes from the $\nabla \cdot (\mathbf{D}_i \nabla \varphi_e)$ term in the Vm equation.

**A21 block** (coupling from Vm to φe equation):
$$A_{21} = \nabla \cdot (\mathbf{D}_i \nabla)$$

This is the discrete diffusion from Vm in the φe equation.

**A22 block** (from the φe equation):
$$A_{22} = \nabla \cdot ((\mathbf{D}_i + \mathbf{D}_e) \nabla)$$

This is the discrete Poisson operator with combined conductivity.

**Right-hand side b1:**
$$b_1 = \frac{\chi C_m}{\Delta t} M V_m^n - \frac{1}{2} K_i V_m^n - \frac{1}{2} K_i \varphi_e^n + \chi \Delta t I_{ion}(V_m^n, \mathbf{w}^n) \mathbf{1}$$

where $\mathbf{1}$ is a vector of ones (source term).

**Right-hand side b2:**
$$b_2 = -\nabla \cdot (\mathbf{D}_i \nabla V_m^n)$$

### Properties of the Block System

1. **Size:** 2N × 2N (twice the monodomain problem size)

2. **Symmetry:**
   - If diffusion tensors are symmetric, the system is **not symmetric** due to the coupling structure
   - The system is better described as "indefinite" in the mathematical sense

3. **Diagonal dominance:** Depends on the temporal discretization and diffusion coefficient ratios
   - For very large diffusion coefficients, the problem is more ill-conditioned
   - This is why preconditioners are essential

4. **Sparsity:** The system is **sparse** but with a more complex sparsity pattern than monodomain
   - Each node couples to its neighbors (diffusion)
   - Each node couples to all other nodes through the φe equation (global coupling if φe is not localized)

5. **Coupling strength:** Depends on the conductivity ratios σi/σe
   - When σi ≈ σe (isotropic conductivity), the system is **weakly coupled**
   - This allows semi-decoupled solution strategies

### Definiteness and Conditioning

The conditioning number κ(A) = σmax / σmin determines how difficult the linear system is to solve:

- **Well-conditioned:** κ ~ 10²–10³ → iterative solvers converge quickly
- **Ill-conditioned:** κ ~ 10⁴–10⁶ → requires preconditioner
- **Very ill-conditioned:** κ > 10⁶ → requires advanced preconditioner + adaptive refinement

The bidomain system is **typically ill-conditioned**, especially for:
- Fine spatial meshes (conditioning worsens as mesh is refined)
- Large diffusion coefficients
- Anisotropic conductivity tensors

**Preconditioners are mandatory** for practical bidomain simulations.

### BDF1 (Backward Euler) Formulation

For comparison, the block system under **BDF1** (backward Euler) is:

$$A_{11} = \frac{\chi C_m}{\Delta t} M + K_i$$

$$A_{12} = K_i$$

(These are identical to Crank-Nicolson except without the 1/2 factors, making the system slightly more stable but only first-order accurate.)

---

## Decoupled vs Coupled Approaches

The 2×2 block system must be solved at each time step. There are fundamentally different strategies for solving it, with different computational costs and accuracy properties.

### Fully Coupled (Monolithic) Approach

**Strategy:** Solve the entire 2×2 block system simultaneously

$$\begin{bmatrix}
A_{11} & A_{12} \\
A_{21} & A_{22}
\end{bmatrix}
\begin{bmatrix}
V_m^{n+1} \\
\varphi_e^{n+1}
\end{bmatrix}
=
\begin{bmatrix}
b_1 \\
b_2
\end{bmatrix}$$

**Solver approach:**
1. Direct solve with sparse LU factorization (if 2N is small enough)
2. Iterative solve with block preconditioners (for large problems)
3. Krylov space methods (GMRES, BiCGSTAB) with multigrid preconditioners

**Advantages:**
- Highest accuracy; no splitting error in the diffusion step
- Unconditional stability properties are preserved
- Convergence is guaranteed (if properly preconditioned)

**Disadvantages:**
- Most computationally expensive per time step
- Requires robustly preconditioned iterative solver
- Difficult to parallelize efficiently due to global coupling
- Memory bandwidth intensive

**When to use:**
- When maximum accuracy is required and computational resources are available
- For validation and verification of other approaches
- When the bidomain effects (extracellular potential) are critical to the physics

**Preconditioner options:**
- Block LU preconditioner with AMG for A22 block
- Constraint preconditioners
- Inexact Schur complement preconditioners

### Semi-Decoupled Approach (Operator Splitting)

**Strategy:** Alternately solve for Vm and φe using operator splitting

This is the approach discussed in detail in the **Operator Splitting for Bidomain** section.

**Algorithm (reaction-diffusion-reaction for Strang):**

1. **Reaction sub-step:** Solve ionic ODEs independently
   - Cost: Local ODE solver per node
   - Can be embarrassingly parallel

2. **Diffusion sub-step:** Solve coupled elliptic-parabolic system
   - Cost: One 2×2 block solve
   - Still coupled but reduced compared to monolithic approach

**Advantages:**
- Better parallelization of ionic step
- Separates the different physics (reaction vs diffusion)
- Simpler to implement incrementally from monodomain code
- Lower memory requirements

**Disadvantages:**
- Introduces splitting error (O(Δt) for Godunov, O(Δt²) for Strang)
- Requires smaller time steps to achieve high accuracy
- Not suitable for very stiff systems where splitting error dominates

**Practical observation:** Despite the splitting error, operator splitting often performs better in practice because the time steps can be adjusted independently for reaction vs diffusion, and the simpler structure avoids preconditioner issues.

### Semi-Decoupled Approach (Sequential Solve)

**Strategy:** Solve for Vm and φe sequentially within each time step

Rather than splitting in time, split in space:

1. **Solve for φe first** (given Vm^n):
   $$A_{22} \varphi_e^{n+1} = b_2(V_m^n)$$

   Then use φe^{n+1} to update:

2. **Solve for Vm** (given φe^{n+1}):
   $$A_{11} V_m^{n+1} = b_1(V_m^n, \varphi_e^{n+1})$$

**Or the reverse order:**

1. Solve for Vm first
2. Solve for φe given updated Vm

**Advantages:**
- Single decoupling pass per time step; no iteration needed
- Each solve is still large (N × N) but simpler than coupled system
- Better conditioning than coupled approach if diffusion dominates

**Disadvantages:**
- Moderate accuracy loss; essentially a fixed-point iteration (one pass)
- Convergence not guaranteed; may diverge for stiff problems
- Subtle: the "coupling" error depends on how strongly φe and Vm are coupled

**Practical use:** This approach is viable only when:
- The conductivity ratio σi/σe leads to weak coupling
- Time steps are small enough
- Ionic nonlinearity is moderate

**Iteration to convergence:** To improve accuracy, the sequential solve can be iterated multiple times:

```
for k = 1 to max_iter
    Solve for φe^{n+1,k}
    Solve for Vm^{n+1,k}
    if ||Vm^{n+1,k} - Vm^{n+1,k-1}|| < tol then break
```

This is equivalent to a fixed-point iteration on the coupled system.

### The Keener-Bogar Decoupling Strategy

Keener and Bogar (1998) developed an influential **operator splitting scheme** that cleanly decouples the bidomain equations:

**Key insight:** The bidomain can be split into:
1. A reaction step (ionic ODE only)
2. A diffusion step (coupled elliptic-parabolic for Vm and φe)

**This is precisely the Godunov/Strang splitting described earlier**, but Keener-Bogar formalized the approach and proved its convergence properties.

**Practical impact:** Their work demonstrated that operator splitting is a viable, efficient approach for bidomain, and it has become the de facto standard in many cardiac electrophysiology codes (e.g., CHASTE, CARP).

**Why it works:**
- The two equations have very different mathematical character (parabolic vs elliptic)
- Splitting respects this distinction
- The ionic ODE solver can be chosen independently of the diffusion solver
- Each component can be optimized separately

### Convergence and Accuracy Implications

**Fully coupled:**
- Temporal error: O(Δt²) for Crank-Nicolson, O(Δt) for Backward Euler
- No additional splitting error

**Godunov operator splitting:**
- Temporal error: O(Δt) + O(Δt) = O(Δt) overall
- Splitting error of O(Δt) dominates and cannot be eliminated with smaller spatial mesh

**Strang operator splitting:**
- Temporal error: O(Δt²) + O(Δt²) = O(Δt²) overall
- Splitting error is second-order; comparable to Crank-Nicolson without coupling

**Sequential solve (one pass):**
- Temporal error: O(Δt²) + O(Δt) decoupling error
- Decoupling error is typically O(Δt) in practice
- Overall: O(Δt) + O(Δt²) = O(Δt)

---

## The Extracellular Equation

### Why This Is the Bottleneck

The extracellular potential φe is determined by the elliptic equation:

$$0 = \nabla \cdot ((\mathbf{D}_i + \mathbf{D}_e) \nabla \varphi_e) + \nabla \cdot (\mathbf{D}_i \nabla V_m)$$

**Key observations:**

1. **No time derivative:** This is an algebraic constraint, not an evolutionary PDE
2. **Purely spatial:** φe must satisfy the constraint at every instant
3. **Linear in φe:** The equation is linear in the unknown φe (though nonlinear overall due to Vm coupling)
4. **Global coupling:** φe at one location depends on Vm everywhere through the diffusion operator

### The Uniqueness Problem: Pinning Constraints

**Mathematical issue:** The system has a null space consisting of constant functions.

If φe is a solution, so is φe + C for any constant C. This is because:
$$\nabla \cdot (\nabla C) = 0$$

**Consequence:** The linear system for φe is **singular**; the matrix A22 is not invertible.

**Solution:** Impose a constraint to pin the potential to a reference value.

#### Common Pinning Strategies

**1. Average Potential Constraint:**
$$\int_\Omega \varphi_e \, dx = 0 \quad \text{or} \quad \frac{1}{|\Omega|} \sum_i \varphi_e(x_i) = 0$$

This fixes the mean value of φe.

**Implementation:**
- Add a Lagrange multiplier to the linear system
- Or enforce through post-processing: $\varphi_e^{corrected} = \varphi_e - \text{mean}(\varphi_e)$

**2. Point Pinning (Dirichlet Condition):**
$$\varphi_e(x_0) = 0 \quad \text{at some reference node } x_0$$

**Implementation:**
- Set one diagonal entry of A22 to a large value (penalty method)
- Or eliminate one degree of freedom and solve a (N-1) × (N-1) system
- Modify the row and column corresponding to the pinned node

**3. Pointwise Norm Bound:**
$$\min_i |\varphi_e(x_i)| = 0 \quad \text{or} \quad \max_i |\varphi_e(x_i)| = 1$$

Less commonly used; more complex to enforce in iterative solvers.

### Computational Cost of the Elliptic Solve

The φe solve is the **dominant computational cost** in bidomain simulations because:

1. **Size:** Solving a coupled 2×2 block system is more expensive than monodomain
2. **Frequency:** Must solve at every time step
3. **Conditioning:** The system is ill-conditioned; requires good preconditioner
4. **Sparsity pattern:** The diffusion operator creates dense coupling in unstructured meshes

**Cost comparison (relative to monodomain):**
- Monodomain: 1 linear solve per time step (N × N system)
- Bidomain (fully coupled): 1 linear solve per time step (2N × 2N system)
- Bidomain (operator splitting): Still 1 linear solve per diffusion step

The 2N × 2N system is roughly **4 times more expensive** to solve than N × N (due to increased bandwidth and coupling).

With properly optimized solvers (algebraic multigrid preconditioners), the ratio can be reduced to **2-3 times**.

### Solvers for the Elliptic Equation

#### Direct Sparse LU Factorization
- **Pros:** Robust; no iteration; deterministic cost
- **Cons:** High memory overhead; not parallelizable for serial LU; factorization is O(N^1.5) for 3D problems
- **Usage:** Small problems (N < 100K); as reference solution

#### Preconditioned Iterative Solvers with AMG

The **Algebraic Multigrid (AMG) preconditioner** is the standard approach for bidomain:

**Algorithm:**
```
Preconditioner Setup:
  1. Build AMG hierarchy from system matrix (coarsening, interpolation)
  2. Create prolongation P and restriction R matrices

Solve with GMRES or BiCGSTAB:
  For each GMRES iteration:
    1. Compute residual r = b - Ax
    2. Apply AMG preconditioner:
       - Smooth on fine grid
       - Transfer to coarse grid
       - Solve on coarse grid
       - Interpolate back to fine grid
       - Post-smooth on fine grid
    3. Update solution using Krylov recurrence
```

**Advantages of AMG for bidomain:**
- **Optimal complexity:** O(N) operations for solver with O(N) memory
- **Scalable:** Works well on unstructured grids
- **Robustness:** Effective for ill-conditioned systems

**Performance gains:**
- Speedup of **5.9–7.7x** compared to ILU preconditioning
- BiCGSTAB with AMG preconditioner: typical convergence in 5–15 iterations for well-discretized problems

**Implementation tools:**
- **hypre/BoomerAMG:** Standard open-source implementation; widely used in CHASTE, CARP
- **ML (Trilinos):** Alternative AMG framework
- **GAMG (PETSc):** Built into PETSc parallel framework

#### Krylov Space Methods

The choice of Krylov method depends on system properties:

- **CG (Conjugate Gradient):** For symmetric positive definite systems; optimal for SPD
  - Not directly applicable to the coupled 2×2 bidomain system (not SPD)
  - Could apply to the extracted A22 block alone if φe is solved first

- **MINRES:** For symmetric indefinite systems; stationary residual norm guarantee
  - Could apply if the system is recast symmetrically

- **GMRES:** For general nonsymmetric systems; flexible preconditioner choice
  - Most robust choice for full bidomain system
  - Stores full Krylov basis (memory intensive)
  - Typical restart: 30–50 (GMRES(m))

- **BiCGSTAB:** Cheaper than GMRES; better memory efficiency
  - Uses 3-term recurrence instead of full basis
  - Converges faster than GMRES in some cases
  - Can be irregular; not guaranteed convergence

**Practical choice:** BiCGSTAB with AMG preconditioner is the **standard in cardiac electrophysiology** codes because it balances robustness, memory efficiency, and speed.

---

## Comparison with Monodomain Solver Architecture

### Monodomain Equations (Current Engine V5.4)

The monodomain model simplifies bidomain by assuming **proportional intracellular and extracellular conductivities:**

$$\chi C_m \frac{\partial V}{\partial t} + \chi I_{ion}(V, \mathbf{w}) = \nabla \cdot (\mathbf{D} \nabla V)$$

where the conductivity tensor $\mathbf{D}$ is a combination of intra/extracellular conductivities.

**Advantages:**
- Single unknown V; linear in spatial discretization
- N × N linear system per time step
- No elliptic constraint
- 10–20x faster than bidomain (with optimal solvers)

**Disadvantages:**
- Cannot model extracellular stimulation accurately
- Cannot account for separate intra/extracellular conductivities
- No extracellular potential information

### Existing Monodomain Architecture

Your Engine V5.4 has:

1. **Operator splitting:** Godunov (1st) + Strang (2nd)
2. **Ionic solvers:** Rush-Larsen, Forward Euler
3. **Diffusion solvers:** FE, RK2, RK4 (explicit); CN, BDF1, BDF2 (implicit)
4. **Linear solvers:** (Not specified in problem statement; presumably direct sparse LU or basic iterative)

**Conceptual structure:**
```
MonodomainSolver
├── IonicSolver (ODE integration)
│   ├── RushLarsenSolver
│   └── ForwardEulerSolver
├── DiffusionSolver (spatial discretization + time stepping)
│   ├── ExplicitDiffusion (RK family)
│   └── ImplicitDiffusion (CN, BDF family)
└── LinearSolver (for implicit diffusion)
    └── SparseDirectSolver (likely)
```

### Required Changes for Bidomain

To extend Engine V5.4 to bidomain, the architecture must change significantly:

#### New Classes/Abstractions Needed

**1. BidomainSplitting (new):**
```cpp
class BidomainSplitting
{
    // Manages overall reaction-diffusion-reaction decomposition
    // Handles two coupled unknowns: Vm, φe

    IonicSolver ionic;              // Unchanged from monodomain
    BidomainDiffusionSolver diffusion;  // NEW: couples Vm and φe

    void reactionStep(double dt);
    void diffusionStep(double dt);   // Solves 2x2 block system
};
```

**2. BidomainDiffusionSolver (new):**
```cpp
class BidomainDiffusionSolver
{
    // Solves the coupled diffusion step:
    // χ Cm (Vm^{n+1} - Vm^*) / Δt = ∇·(Di ∇Vm) + ∇·(Di ∇φe) + ...
    // 0 = ∇·((Di + De) ∇φe) + ∇·(Di ∇Vm)

    BlockLinearSystem system;  // 2N x 2N matrix
    LinearSolver linSolver;    // Must handle 2x2 block structure

    void assemble(const VectorXd& Vm_old, const VectorXd& currentIonic);
    void solve(VectorXd& Vm_new, VectorXd& phi_e);
};
```

**3. BlockLinearSystem (new):**
```cpp
class BlockLinearSystem
{
    // Represents the 2x2 block structure:
    // [A11  A12] [u1]   [b1]
    // [A21  A22] [u2] = [b2]

    SparseMatrix A11, A12, A21, A22;  // Individual blocks
    VectorXd b1, b2;                   // RHS blocks

    // Assembly, matrix-vector product, preconditioning
};
```

**4. BidomainLinearSolver (new):**
```cpp
class BidomainLinearSolver
{
    // Specialized solver for the 2x2 block system
    // Options:
    //   - Fully coupled (GMRES + AMG preconditioner)
    //   - Sequential solve (φe first, then Vm)
    //   - Iterative decoupling

    enum SolveStrategy { FullyCoupled, Sequential, Iterative };

    BlockPreconditioner precond;  // Block LU or constraint preconditioner
    void solve(BlockLinearSystem& system, VectorXd& u1, VectorXd& u2);
};
```

**5. BlockPreconditioner (new):**
```cpp
class BlockPreconditioner
{
    // Provides preconditioning for iterative solvers
    // Strategies:
    //   - Block LU with AMG on A22
    //   - Constraint preconditioner
    //   - Schur complement preconditioner

    void setup(const BlockLinearSystem& system);
    void apply(const VectorXd& rhs, VectorXd& preconditioned);
};
```

**6. PinningConstraint (new):**
```cpp
class PinningConstraint
{
    // Enforces uniqueness of φe (null space issue)
    // Strategies:
    //   - Average potential (Lagrange multiplier)
    //   - Point pinning (Dirichlet)

    void applyToProblem(BlockLinearSystem& system);
};
```

#### Reusable Components

**Can be reused from monodomain:**
- `IonicSolver` and its implementations (Rush-Larsen, FE) — **unchanged**
- `Mesh` and `FunctionSpace` classes — **unchanged**
- `FiniteElementAssembler` for spatial discretization — mostly **unchanged**
- `TimeIntegrationScheme` interface — **extended** but not fundamentally changed

**Must be enhanced:**
- `LinearSolver`: Must handle 2x2 block systems; add GMRES, BiCGSTAB with AMG
- `FiniteElementAssembler`: Must assemble cross-coupling terms (A12, A21)
- Matrix/Vector abstractions: Must support block operations

### Solver Hierarchy Diagram

**Monodomain (Current V5.4):**
```
Monodomain Solver
├── Time stepping loop
│   ├── Reaction step (Ionic ODEs)
│   └── Diffusion step (Linear solve for V)
└── Linear solver (for diffusion)
```

**Bidomain (Proposed):**
```
Bidomain Solver
├── Time stepping loop
│   ├── Reaction step (Ionic ODEs) ← REUSE
│   └── Diffusion step
│       ├── Assemble 2x2 block system ← NEW
│       └── Block linear solver ← NEW
│           ├── Setup preconditioner ← NEW
│           └── Iterative solver (BiCGSTAB/GMRES) ← NEW
└── Linear solvers
    ├── Monodomain (existing)
    └── Bidomain (new)
        ├── GMRES with AMG
        └── BiCGSTAB with AMG
```

### Configuration Examples

**Configuration for semi-implicit Strang splitting with Crank-Nicolson:**
```yaml
TimeIntegration:
  Scheme: StrangSplitting
  TimeStep: 0.1  # ms

Reaction:
  Solver: RushLarsen

Diffusion:
  Temporal: CrankNicolson
  Coupling: SemiImplicit  # Explicit ionic, implicit diffusion

LinearSolver:
  Type: BiCGSTAB
  Preconditioner: AMG
  MaxIterations: 100
  Tolerance: 1e-6

ExtracelluarPotential:
  PinningType: AveragePotential  # or PointPinning
```

---

## Performance Considerations

### Computational Cost Ratios

Extensive research has established the relative computational costs of bidomain vs monodomain simulations.

#### Earlier Claims vs. Reality

**Older claims:** Bidomain is ~100 times more expensive than monodomain

**Modern assessment (with order-optimal solvers):**
- **Simple ionic models:** Bidomain is ~10x more expensive
- **Complex ionic models:** Bidomain is ~2-4x more expensive
- **With specialized optimizations:** Approaches 2x for very complex models

**Why the reduction:** Advanced preconditioners (AMG) and parallel algorithms reduce the linear solver bottleneck from O(N^2) to O(N).

#### Cost Breakdown for Bidomain Time Step

Assuming operator splitting (Strang) with semi-implicit Crank-Nicolson diffusion:

| Task | % of Time |
|------|-----------|
| Ionic ODE solve (Rush-Larsen) | 10–15% |
| Diffusion assembly (2x2 block) | 5–10% |
| Preconditioner setup (AMG) | 2–3% per setup cycle |
| Linear solve (BiCGSTAB iterations) | 70–85% |

**Key insight:** The linear solver dominates. Optimization of the linear solver (preconditioner, Krylov method) has the highest impact.

### GPU Parallelism Opportunities

Bidomain simulations are ideal candidates for GPU acceleration:

#### Embarrassingly Parallel Parts (GPU-native)

1. **Ionic ODE solve:** Each node is independent
   - Speedup: **50–200x** on modern GPUs (Tesla V100/H100)
   - One thread per node; minimal synchronization
   - Excellent memory access patterns

2. **Diffusion assembly:** Element-wise operations
   - Speedup: **5–30x**
   - Fine-grained parallelism
   - Memory-bandwidth limited

#### Challenging Parts

1. **Linear system solve (BiCGSTAB/GMRES + AMG):**
   - Speedup: **5–15x** (more modest)
   - Iterative solver with global synchronization barriers
   - Preconditioner application requires sparse matrix kernels
   - Communication-bound on distributed systems

2. **Preconditioner (AMG) setup:**
   - Speedup: **3–7x** (limited by coarsening algorithm)
   - Mostly sequential coarsening step
   - Setup cost amortized over multiple solves (not every step)

#### Realistic GPU Performance

**Hardware:** NVIDIA Tesla V100 or newer (H100)

**Speedup relative to high-end CPU (e.g., 16-core Xeon):**
- Full 3D bidomain simulation: **8–20x faster**
- Ionic computation-heavy: **30–50x faster**
- Solver-limited: **5–10x faster**

**Bottleneck:** Memory bandwidth to/from GPU; bidomain requires moving 2x data compared to monodomain.

**Practical recommendation:** GPU acceleration is most beneficial when:
1. Complex ionic models (many gating variables) — maximize computation/communication ratio
2. Large spatial meshes (> 1M nodes) — hide communication latency
3. Multiple samples or ensemble simulations — amortize GPU setup costs

### Memory Bandwidth Considerations

#### Memory Footprint

**Per node (single precision float, 32-bit):**
- Monodomain:
  - Vm: 1 × 4 B = 4 B
  - w (ionic gates, ~10 vars): 10 × 4 B = 40 B
  - Total: ~50 B/node

- Bidomain:
  - Vm: 1 × 4 B = 4 B
  - φe: 1 × 4 B = 4 B
  - w: ~40 B
  - Total: ~50 B/node (same!)

**Surprising result:** Bidomain requires only ~2–4% more memory for state variables compared to monodomain.

**However, the linear system is larger:**
- Monodomain: N × N matrix (sparse, ~10 entries/row) → ~10 N floats
- Bidomain: 2N × 2N matrix (sparse, ~20 entries/row) → ~40 N floats

**Memory ratio:** Bidomain linear system requires ~4x more memory than monodomain.

#### Memory Bandwidth Utilization

Modern CPUs have peak bandwidth ~100–200 GB/s; GPUs have peak bandwidth ~500–2000 GB/s.

**BiCGSTAB iteration (matrix-vector product bottleneck):**
- Operations: 1 sparse matrix-vector product ≈ 10 floats accessed per result
- Arithmetic: 1 multiply-add per nonzero = 2 FLOPs per nonzero
- Ratio: 2 FLOPs / 10 bytes = 0.2 FLOPs/byte

**Result:** This is **compute-bound** (requires high bandwidth) but achieves only ~30% of peak arithmetic throughput due to memory latency and sparsity overhead.

**Mitigation:**
- Optimize sparse matrix format (CSR, ELL, COO depending on kernel)
- Overlap computation with communication (important on distributed systems)
- Use mixed precision (float32 for compute, float64 for critical calculations)

### Scaling Analysis

#### Strong Scaling (Fixed Problem Size)

**Monodomain on 128 cores:**
- Parallel efficiency: ~60–70% (communication overhead small)
- Speedup: ~80–90x

**Bidomain on 128 cores:**
- Parallel efficiency: ~50–65% (more communication in coupled solve)
- Speedup: ~64–83x

**Conclusion:** Bidomain scales slightly worse than monodomain due to:
1. More complex linear system requiring more communication per iteration
2. Preconditioner setup is harder to parallelize
3. Coarser grids in AMG hierarchy become more of a bottleneck

#### Weak Scaling (Problem Size Increases with Cores)

Both monodomain and bidomain show good weak scaling up to ~1000s of cores on distributed systems.

**Important for cardiac simulations:**
- Typical 3D problem: 100K–10M mesh nodes
- Typical cluster: 16–512 cores
- Parallel efficiency achievable: 80–90%

---

## Implementation Architecture

### Proposed Software Design

Based on best practices in cardiac electrophysiology software (CHASTE, CARP) and the existing Engine V5.4 structure:

#### Core Components

**1. Bidomain Problem Definition**
```cpp
class BidomainProblem
{
    // Problem parameters
    double chi;                    // Surface-to-volume ratio
    double Cm;                     // Membrane capacitance

    // Tissue properties
    TensorField Di, De;            // Intra/extracellular conductivity

    // Boundary conditions
    BoundaryConditions bcs;        // Neumann/Dirichlet for Vm and φe

    // Geometry
    Mesh mesh;
    FunctionSpace functionSpace;
};
```

**2. Bidomain Solver Main Loop**
```cpp
class BidomainSolver
{
    // Configuration
    TimeSteppingScheme timeScheme;  // Godunov, Strang, etc.
    double timeStep, totalTime;

    // Components
    IonicSolver ionicSolver;           // Reused from monodomain
    BidomainDiffusionSolver diffSolver; // NEW

    void solve()
    {
        double t = 0;
        while (t < totalTime)
        {
            // Operator splitting (Strang example)
            reactionHalfStep(0.5 * timeStep);  // Vm, w
            diffusionFullStep(timeStep);       // Vm, φe coupled solve
            reactionHalfStep(0.5 * timeStep);  // Vm, w

            t += timeStep;
        }
    }

private:
    void reactionHalfStep(double dt);
    void diffusionFullStep(double dt);
};
```

**3. Diffusion Solver for Bidomain**
```cpp
class BidomainDiffusionSolver
{
    // Assembles and solves:
    // χ Cm (Vm^{n+1} - Vm^*) / Δt = ∇·(Di ∇Vm^{n+1}) + ...
    // 0 = ∇·((Di + De) ∇φe^{n+1}) + ...

    BidomainAssembler assembler;
    BlockLinearSystem system;
    BidomainLinearSolver linSolver;

    void solve(VectorXd& Vm, VectorXd& phi_e,
               const VectorXd& Vm_old, const VectorXd& ionicCurrent)
    {
        assembler.assemble(system, Vm_old, ionicCurrent);
        linSolver.solve(system, Vm, phi_e);
        applyPinning(phi_e);  // Enforce uniqueness
    }
};
```

**4. Block Linear System**
```cpp
class BlockLinearSystem
{
    SparseMatrix A11, A12, A21, A22;
    VectorXd b1, b2;

    // Useful operations
    void matvec(const VectorXd& u, VectorXd& Au);  // [A11 A12; A21 A22] * u
    void extractBlock(int blockIdx, SparseMatrix& block);
    void symmetrize();  // For analysis
};
```

**5. Linear Solver with AMG**
```cpp
class BidomainLinearSolver
{
    KrylovSolver krylov;      // BiCGSTAB or GMRES
    AMGPreconditioner amg;     // Algebraic multigrid

    void solve(const BlockLinearSystem& system,
               VectorXd& u1, VectorXd& u2)
    {
        // Solve [A11 A12; A21 A22] * [u1; u2] = [b1; b2]

        // Option 1: Fully coupled (pack into single vector)
        VectorXd u_packed = [u1; u2];
        krylov.solve(system, u_packed, amg);  // BiCGSTAB + AMG
        [u1; u2] = u_packed;

        // Option 2: Sequential solve
        // Solve A22 * u2 = b2 - A21 * u1_old
        // Then A11 * u1 = b1 - A12 * u2
    }
};
```

**6. AMG Preconditioner Setup**
```cpp
class AMGPreconditioner
{
    vector<SparseMatrix> prolongation;  // Coarse-to-fine transfer
    vector<SparseMatrix> restriction;   // Fine-to-coarse transfer
    vector<SparseMatrix> coarseMatrices; // System on each level

    void setup(const SparseMatrix& A)
    {
        // 1. Coarsen: select coarse nodes using heavy edge matching
        // 2. Interpolation: construct P (fine-to-coarse)
        // 3. Restriction: R = P^T
        // 4. Recursive: Create A_coarse = R * A * P
        // 5. Repeat until coarsest level
    }

    void apply(const VectorXd& rhs, VectorXd& solution)
    {
        // V-cycle: smooth, coarsen, solve, interpolate, post-smooth
    }
};
```

#### Testing and Validation Strategy

**Unit tests:**
- Bidomain assembly correctness (compare with hand-calculated small problems)
- Block linear system operations
- Pinning constraint enforcement
- Comparison with monodomain (when σi ≈ σe, should match monodomain solution)

**Integration tests:**
- 1D cable equation with exact solution
- 2D square domain with periodic BCs (validate convergence)
- Manufactured solutions (Taylor series method)

**Regression tests:**
- Standard cardiac action potential propagation
- Extracellular potential in uniform conducting medium
- Comparison with published results (e.g., from CHASTE, CARP)

**Verification against literature:**
- Conduction velocity propagation
- Extracellular potential fields
- Activation time maps

---

## References

This research document synthesizes findings from the following sources:

### Review Articles and Books

1. Solvers for the Cardiac Bidomain Equations - PMC
   https://pmc.ncbi.nlm.nih.gov/articles/PMC2881536/

2. A numerical guide to the solution of the bidomain equations of cardiac electrophysiology - ResearchGate
   https://www.researchgate.net/publication/44678641_A_numerical_guide_to_the_solution_of_the_bidomain_equations_of_cardiac_electrophysiology

3. Deriving the Bidomain Model of Cardiac Electrophysiology - Frontiers in Physiology (2021)
   https://public-pages-files-2025.frontiersin.org/journals/physiology/articles/10.3389/fphys.2021.811029/pdf

### Operator Splitting Methods

4. Operator splitting for the bidomain model revisited - ScienceDirect
   https://www.sciencedirect.com/science/article/pii/S0377042715004677

5. High-Order Operator Splitting for the Bidomain and Monodomain Models - SIAM Journal on Scientific Computing
   https://epubs.siam.org/doi/10.1137/17M1137061

6. Chapter 7: Operator Splitting - Springer
   https://link.springer.com/content/pdf/10.1007/978-3-031-30852-9_7

7. A simple and efficient adaptive time stepping technique for low-order operator splitting schemes applied to cardiac electrophysiology (2023)
   https://onlinelibrary.wiley.com/doi/10.1002/cnm.3670

### IMEX and Semi-Implicit Methods

8. GEMS: A Fully Integrated PETSc-Based Solver for Coupled Cardiac Electromechanics and Bidomain Simulations - PMC
   https://pmc.ncbi.nlm.nih.gov/articles/PMC6198176/

9. A fully implicit finite element method for bidomain models of cardiac electromechanics - PMC
   https://pmc.ncbi.nlm.nih.gov/articles/PMC3501134/

10. Semi-Implicit Time-Discretization Schemes for the Bidomain Model - ResearchGate
    https://www.researchgate.net/publication/220179333_Semi-Implicit_Time-Discretization_Schemes_for_the_Bidomain_Model

### Stability and Oscillation Control

11. Stable time integration suppresses unphysical oscillations in the bidomain model - Frontiers in Physics
    https://www.frontiersin.org/articles/10.3389/fphy.2014.00040/full

12. Composite Backward Differentiation Formula for the Bidomain Equations - Frontiers
    https://www.frontiersin.org/articles/10.3389/fphys.2020.591159/full

### High-Order and Spectral Deferred Correction Methods

13. Novel bidomain partitioned strategies for the simulation of ventricular fibrillation dynamics - arXiv (2025)
    https://arxiv.org/html/2510.27447v1
    https://arxiv.org/pdf/2510.27447

### Computational Complexity Analysis

14. On the computational complexity of the bidomain and the monodomain models of electrophysiology - Springer
    https://link.springer.com/article/10.1007/s10439-006-9082-z

### Preconditioners and Algebraic Multigrid

15. Algebraic Multigrid Preconditioner for the Cardiac Bidomain Model - PMC
    https://pmc.ncbi.nlm.nih.gov/articles/PMC5428748/

16. Non-Symmetric Algebraic Multigrid Preconditioners for the Bidomain Reaction–Diffusion system - Springer
    https://link.springer.com/chapter/10.1007/978-3-642-11795-4_78

17. Performance Comparison of Parallel Geometric and Algebraic Multigrid Preconditioners for the Bidomain Equations - Springer
    https://link.springer.com/chapter/10.1007/11758501_15

18. A comparison of Algebraic Multigrid Bidomain solvers on hybrid CPU–GPU architectures - ScienceDirect
    https://www.sciencedirect.com/science/article/pii/S0045782524001312

### GPU Acceleration

19. Solving the cardiac bidomain equations using graphics processing units - ScienceDirect
    https://www.sciencedirect.com/science/article/pii/S1877750312000701

20. Accelerating Cardiac Bidomain Simulations Using Graphics Processing Units - PMC
    https://pmc.ncbi.nlm.nih.gov/articles/PMC3696513/

21. Parallel Optimization of 3D Cardiac Electrophysiological Model Using GPU - PMC
    https://pmc.ncbi.nlm.nih.gov/articles/PMC4637086/

22. Cardiac simulation on multi-GPU platform - Springer
    https://link.springer.com/article/10.1007/s11227-010-0540-x

23. GPU accelerated solver for nonlinear reaction–diffusion systems - ScienceDirect
    https://www.sciencedirect.com/science/article/abs/pii/S0010465515002635

### Software Implementations

24. CHASTE: An Open Source C++ Library for Computational Physiology and Biology - PLOS Computational Biology
    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002970

25. Chaste - Miguel O Bernabeu et al. (2014) - International Journal of High Performance Computing Applications
    https://journals.sagepub.com/doi/10.1177/1094342012474997

26. Cardiac Chaste Documentation
    https://chaste.github.io/components/cardiac/

27. Cellular cardiac electrophysiology modeling with Chaste and CellML - PMC
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4285015/

### Extracellular Potential and Uniqueness

28. Existence and uniqueness of the solution for the bidomain model used in cardiac electrophysiology - HAL Science
    https://hal.science/hal-00101458/document

29. Bidomain model - Wikipedia
    https://en.wikipedia.org/wiki/Bidomain_model

30. Optimal control approach to termination of re-entry waves in cardiac electrophysiology - PMC
    https://pmc.ncbi.nlm.nih.gov/articles/PMC3978702/

### Adaptive Time Stepping

31. Automatic Control and Adaptive Time-Stepping - ResearchGate
    https://www.researchgate.net/publication/226490811_Automatic_Control_and_Adaptive_Time-Stepping

32. On time integration error estimation and adaptive time stepping in structural dynamics - ResearchGate
    https://www.researchgate.net/publication/36450933_On_Time_Integration_Error_Estimation_and_Adaptive_Time_Stepping_in_Structural_Dynamics

---

## Summary and Recommendations

### Key Takeaways

1. **Operator splitting (Godunov/Strang) is the standard approach** for bidomain equations in practice. It cleanly separates ionic and diffusion physics and is naturally parallelizable.

2. **The linear system is the bottleneck.** Each diffusion step requires solving a 2N × 2N coupled system, which is 4–10x more expensive than a monodomain solve without careful preconditioner selection.

3. **Algebraic multigrid (AMG) preconditioners are essential.** They reduce the effective cost of the bidomain linear system by 6–8x compared to basic methods, making the overall bidomain cost ~2–10x that of monodomain (depending on ionic complexity).

4. **Crank-Nicolson or SDIRK2 are recommended for temporal discretization.** Both achieve second-order accuracy; SDIRK2 avoids spurious oscillations without significant overhead.

5. **GPU acceleration is effective for the ionic ODE solve (50–200x speedup)** but moderate for the linear solver (~5–10x). Overall 8–20x speedup on large 3D problems is achievable.

6. **The extracellular potential φe requires a pinning constraint** to ensure uniqueness. Average potential pinning is simple and effective.

### Path Forward for Engine V5.4 Extension

**Phase 1 (Foundational):**
- Implement `BlockLinearSystem` abstraction
- Add BiCGSTAB + basic ILU preconditioner for 2×2 block solve
- Implement pinning constraint for φe uniqueness
- Develop 1D validation test suite

**Phase 2 (Core Bidomain Solver):**
- Implement `BidomainDiffusionSolver` with Strang splitting
- Reuse existing `IonicSolver` components
- Test on 2D academic problems (convergence, comparison with literature)

**Phase 3 (Performance Optimization):**
- Integrate AMG preconditioner (via hypre/PETSc or custom implementation)
- Parallel assembly of block system
- GPU kernels for ionic solve
- Benchmark against CHASTE/CARP reference solutions

**Phase 4 (Advanced Features):**
- Adaptive time stepping based on splitting error
- Spectral deferred correction for higher-order accuracy
- Support for bidomain-with-bath (external potential)
- Extended bidomain (includes secondary variables)

---

**Document Completed:** February 2025
**Classification:** Research and Development
**Access:** Internal Engineering Team
