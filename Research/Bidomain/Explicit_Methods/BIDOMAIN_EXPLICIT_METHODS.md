# Explicit, Semi-Explicit, and IMEX Methods for the Cardiac Bidomain Equations

## Table of Contents
1. [Why Fully Explicit Methods Are Problematic for Bidomain](#1-why-fully-explicit-methods-are-problematic-for-bidomain)
2. [The CFL Stability Bound for Forward Euler](#2-the-cfl-stability-bound-for-forward-euler)
3. [Semi-Implicit (IMEX) Methods — The Practical Workhorse](#3-semi-implicit-imex-methods--the-practical-workhorse)
4. [IMEX Multistep Schemes: CN-FE, CN-AB, SBDF2](#4-imex-multistep-schemes-cn-fe-cn-ab-sbdf2)
5. [IMEX Runge-Kutta Methods](#5-imex-runge-kutta-methods)
6. [Decoupled (Partitioned) Methods](#6-decoupled-partitioned-methods)
7. [Explicit Stabilized Methods: RKC and Multirate RKC](#7-explicit-stabilized-methods-rkc-and-multirate-rkc)
8. [Exponential Integrators and Higher-Order Rush-Larsen](#8-exponential-integrators-and-higher-order-rush-larsen)
9. [Spectral Deferred Correction (SDC)](#9-spectral-deferred-correction-sdc)
10. [Comparison Table](#10-comparison-table)
11. [Recommendations for Engine V5.4](#11-recommendations-for-engine-v54)
12. [References](#12-references)

---

## 1. Why Fully Explicit Methods Are Problematic for Bidomain

The bidomain model has a structural feature that fundamentally limits the applicability of fully explicit time integration: **the elliptic equation for the extracellular potential φ_e has no time derivative**.

### The Two-Equation Structure

Recall the bidomain system after operator splitting (diffusion step):

**Parabolic equation** (has ∂V_m/∂t):
$$\chi C_m \frac{\partial V_m}{\partial t} = \nabla \cdot (\mathbf{D}_i \nabla V_m) + \nabla \cdot (\mathbf{D}_i \nabla \phi_e)$$

**Elliptic equation** (no time derivative):
$$0 = \nabla \cdot ((\mathbf{D}_i + \mathbf{D}_e) \nabla \phi_e) + \nabla \cdot (\mathbf{D}_i \nabla V_m)$$

### The Fundamental Constraint

The elliptic equation is a **constraint** that must be satisfied at every instant in time. It is not an evolution equation — there is no "marching forward" possible. Regardless of how the parabolic equation is discretized in time, the elliptic equation always requires solving a linear system:

$$(\mathbf{K}_i + \mathbf{K}_e) \boldsymbol{\phi}_e = -\mathbf{K}_i \mathbf{V}_m$$

This means **at least one implicit linear solve per time step is unavoidable** in the bidomain model. This is the single most important distinction between the bidomain and the monodomain models. In the monodomain model, there is only a single parabolic PDE, and fully explicit methods (including LBM) can handle the entire system.

### Three Levels of "Explicitness"

Given this constraint, we can classify bidomain time-stepping methods by what is treated explicitly:

| Level | What is Explicit | What is Implicit | Methods |
|-------|-----------------|------------------|---------|
| **Fully Implicit** | Nothing | Everything (ionic + diffusion + elliptic) | Backward Euler, CBDF2, Newton-Krylov |
| **Semi-Implicit (IMEX)** | Ionic currents I_ion | Diffusion + elliptic coupling | IMEX, CN-FE, SBDF2, operator splitting |
| **Maximally Explicit** | Ionic currents + parabolic diffusion | Only the elliptic constraint | Forward Euler + elliptic solve, RKC + elliptic solve |

The "maximally explicit" approach treats the parabolic PDE with an explicit method (forward Euler, RK4, or RKC) and only solves the elliptic equation implicitly. This is possible but incurs a severe CFL restriction on the time step.

### Contrast with Monodomain

In the monodomain model, the elliptic equation is absent (or equivalently, it reduces to a known relationship between V_m and φ_e under the equal anisotropy ratio assumption). This means:

- **Monodomain**: Fully explicit methods (forward Euler, RK4, LBM) are viable because there is only a parabolic PDE
- **Bidomain**: At minimum, the elliptic constraint forces one linear solve per step; the question is whether the parabolic part is also treated implicitly

This is exactly why Engine V5.4's LBM solver works beautifully for monodomain but cannot directly extend to bidomain without modification (see Chapter 6 of the textbook on pseudo-time relaxation and hybrid approaches).

---

## 2. The CFL Stability Bound for Forward Euler

### The Derivation (Puwal & Roth, 2007)

Puwal and Roth derived the explicit stability condition for the forward Euler method applied to the bidomain equations. The key result is a relationship between the time step Δt and the spatial discretization Δx that ensures numerical stability.

### General CFL Form for Parabolic PDEs

For a standard diffusion equation ∂u/∂t = D ∇²u discretized with forward Euler in time and central differences in space, the stability condition in d dimensions is:

$$\Delta t \leq \frac{\Delta x^2}{2\,d\,D}$$

For the **bidomain model**, the effective diffusion coefficient is determined by the conductivity tensors σ_i and σ_e, the membrane capacitance C_m, and the surface-to-volume ratio χ.

### Bidomain-Specific CFL Condition

Under the assumption of equal anisotropy ratios (σ_i = α σ_e) and aligned straight fibers, an approximate CFL condition for the bidomain parabolic equation is:

$$\Delta t \leq \frac{C_m \, \chi \, \Delta x^2}{2\,d\,D_{\max}}$$

where D_max is the maximum effective diffusion coefficient:

$$D_{\max} = \max\left(\frac{\sigma_{iL}\,\sigma_{eL}}{\sigma_{iL} + \sigma_{eL}},\; \frac{\sigma_{iT}\,\sigma_{eT}}{\sigma_{iT} + \sigma_{eT}}\right) \cdot \frac{1}{\chi\,C_m}$$

Here σ_{iL}, σ_{iT} are the intracellular longitudinal and transverse conductivities, and σ_{eL}, σ_{eT} are the extracellular counterparts.

### Practical Numbers

Using typical bidomain conductivity values (Roth, 1997):

| Parameter | Value |
|-----------|-------|
| σ_{iL} | 0.17 S/m |
| σ_{iT} | 0.019 S/m |
| σ_{eL} | 0.62 S/m |
| σ_{eT} | 0.24 S/m |
| C_m | 1.0 μF/cm² |
| χ | 1400 cm⁻¹ |

For a fine mesh with **Δx = 0.1 mm** (100 μm) in 3D:

$$\Delta t_{\text{explicit}} \lesssim \frac{(1.0 \times 10^{-6})(1400)(0.01)^2}{2 \times 3 \times D_{\max}} \approx 0.5\;\mu\text{s}$$

For the same mesh, semi-implicit methods (CN or SDIRK2) can use:

$$\Delta t_{\text{semi-implicit}} \approx 50\;\mu\text{s} \quad \text{(100× larger)}$$

### The Weber dos Santos Verification (2004)

Weber dos Santos et al. numerically verified that for Δx = 3.3 μm, the forward Euler scheme required Δt ≤ 0.07 μs for stability. A semi-implicit scheme allowed Δt > 100× larger — confirming the theoretical CFL bound.

### Scaling Behavior

The CFL condition scales as Δt ∝ Δx². This means:

| Δx (mm) | Δt_explicit (μs) | Δt_semi-implicit (μs) | Ratio |
|---------|-------------------|------------------------|-------|
| 0.5 | ~12.5 | ~50 | 4× |
| 0.2 | ~2.0 | ~50 | 25× |
| 0.1 | ~0.5 | ~50 | 100× |
| 0.05 | ~0.125 | ~50 | 400× |

For production-quality cardiac simulations with Δx = 0.1-0.2 mm, the explicit CFL restriction is 25-100× more restrictive than what accuracy alone requires. This is the fundamental reason why semi-implicit methods dominate the bidomain literature.

---

## 3. Semi-Implicit (IMEX) Methods — The Practical Workhorse

### General Framework

IMEX (Implicit-Explicit) methods split the right-hand side of the ODE system into stiff and non-stiff parts:

$$\frac{d\mathbf{u}}{dt} = \mathbf{f}_E(\mathbf{u}) + \mathbf{f}_I(\mathbf{u})$$

where:
- **f_E** (explicit part): the nonlinear ionic current I_ion — cheap to evaluate, local (node-by-node), but potentially stiff
- **f_I** (implicit part): the linear diffusion operator + elliptic coupling — global, requires linear solve, but linear

### Why IMEX Works Well for Bidomain

1. **The diffusion operator is linear**: The implicit part only requires solving a *linear* system, not a nonlinear one. This avoids Newton iteration entirely.
2. **Ionic currents are local**: Evaluating I_ion(V_m, w) is embarrassingly parallel — each node is independent.
3. **The time step is limited by accuracy, not stability**: Semi-implicit methods remove the parabolic CFL restriction while keeping the implementation simple.

### The Standard Semi-Implicit Bidomain Step

Given the state (V_m^n, φ_e^n, w^n) at time t^n:

**Step 1 — Evaluate ionic currents explicitly:**
$$I_{\text{ion}}^n = I_{\text{ion}}(V_m^n, \mathbf{w}^n)$$

**Step 2 — Advance gating variables explicitly (or with Rush-Larsen):**
$$\mathbf{w}^{n+1} = \text{RushLarsen}(\mathbf{w}^n, V_m^n, \Delta t)$$

**Step 3 — Solve the coupled diffusion/elliptic system implicitly for V_m^{n+1}, φ_e^{n+1}:**

$$\begin{pmatrix} \mathbf{A}_{11} & \mathbf{A}_{12} \\ \mathbf{A}_{21} & \mathbf{A}_{22} \end{pmatrix} \begin{pmatrix} \mathbf{V}_m^{n+1} \\ \boldsymbol{\phi}_e^{n+1} \end{pmatrix} = \begin{pmatrix} \mathbf{b}_1(V_m^n, I_{\text{ion}}^n) \\ \mathbf{b}_2 \end{pmatrix}$$

The exact form of the A blocks and right-hand side depends on which time-stepping scheme is used for the implicit part (backward Euler, Crank-Nicolson, SDIRK2, BDF2).

### Stability Properties

Ethier and Bourgault (2008) proved that semi-implicit methods for the bidomain model have stability conditions that depend only on the *explicit treatment of the ionic term*, not on the diffusion. The stability condition takes the form:

$$\Delta t \leq C \cdot \frac{1}{\max_k |\partial I_{\text{ion}}/\partial V_m|}$$

This depends on the ionic model's stiffness, not on the mesh size. For typical ionic models (ten Tusscher, O'Hara-Rudy), this allows Δt ≈ 10-50 μs — adequate for accurate resolution of the action potential.

---

## 4. IMEX Multistep Schemes: CN-FE, CN-AB, SBDF2

Several specific IMEX multistep schemes have been analyzed for the bidomain model. All share the same structure: explicit treatment of I_ion, implicit treatment of diffusion.

### First-Order: IMEX Euler (SBDF1)

The simplest IMEX scheme — backward Euler for diffusion, forward Euler for reaction:

$$\frac{\chi C_m}{\Delta t} \mathbf{M} (\mathbf{V}_m^{n+1} - \mathbf{V}_m^n) + \mathbf{K}_i \mathbf{V}_m^{n+1} + \mathbf{K}_i \boldsymbol{\phi}_e^{n+1} = -\chi \mathbf{M}\, I_{\text{ion}}^n$$

$$\mathbf{K}_i \mathbf{V}_m^{n+1} + (\mathbf{K}_i + \mathbf{K}_e) \boldsymbol{\phi}_e^{n+1} = 0$$

**Properties:**
- First-order accurate: O(Δt)
- L-stable (strongly damps high-frequency errors)
- Unconditionally stable for the diffusion part
- Stability restricted only by the explicit ionic treatment
- Simple to implement — natural starting point

### Second-Order: CN-FE (Crank-Nicolson / Forward Euler)

Crank-Nicolson for diffusion, forward Euler for ionic currents:

$$\frac{\chi C_m}{\Delta t} \mathbf{M} (\mathbf{V}_m^{n+1} - \mathbf{V}_m^n) + \frac{1}{2}\mathbf{K}_i (\mathbf{V}_m^{n+1} + \mathbf{V}_m^n) + \frac{1}{2}\mathbf{K}_i (\boldsymbol{\phi}_e^{n+1} + \boldsymbol{\phi}_e^n) = -\chi \mathbf{M}\, I_{\text{ion}}^n$$

**Properties:**
- Second-order accurate: O(Δt²)
- A-stable but NOT L-stable
- **Can produce spurious oscillations** near sharp wave fronts (the action potential upstroke)
- The oscillations arise because CN does not damp high-frequency error modes

### Second-Order: CN-AB (Crank-Nicolson / Adams-Bashforth)

Crank-Nicolson for diffusion, second-order Adams-Bashforth extrapolation for ionic currents:

$$\text{Ionic RHS:}\quad \frac{3}{2} I_{\text{ion}}^n - \frac{1}{2} I_{\text{ion}}^{n-1}$$

**Properties:**
- Second-order accurate: O(Δt²)
- Small error constant (often most accurate in practice)
- Weak damping of high-frequency error modes
- Requires storing I_ion from the previous time step (two-step method)
- The weak damping can cause extra iterations in multigrid solvers

### Second-Order: SBDF2 (Semi-Implicit BDF2)

BDF2 for diffusion, second-order explicit extrapolation for ionic currents:

$$\frac{\chi C_m}{\Delta t} \mathbf{M} \left(\frac{3}{2}\mathbf{V}_m^{n+1} - 2\mathbf{V}_m^n + \frac{1}{2}\mathbf{V}_m^{n-1}\right) + \mathbf{K}_i \mathbf{V}_m^{n+1} + \mathbf{K}_i \boldsymbol{\phi}_e^{n+1} = -\chi \mathbf{M} (2 I_{\text{ion}}^n - I_{\text{ion}}^{n-1})$$

**Properties:**
- Second-order accurate: O(Δt²)
- A-stable AND strongly damping (though not technically L-stable)
- **Best damping of high-frequency error** among second-order IMEX multistep schemes
- Requires two previous time levels (two-step method; needs a one-step method for startup)
- Generally preferred over CN-AB for cardiac simulations

### Head-to-Head Comparison (Ethier & Bourgault, 2008)

| Scheme | Order | Damping | Oscillations | Memory | Startup |
|--------|-------|---------|-------------|--------|---------|
| IMEX Euler | 1 | Strong (L-stable) | None | Minimal | Self-starting |
| CN-FE | 2 | None | **Yes, near upstroke** | Minimal | Self-starting |
| CN-AB | 2 | Weak | Mild | +1 history vector | Needs startup |
| SBDF2 | 2 | Strong | None | +1 history vector | Needs startup |
| IMEX Gear | 2 | Moderate | Rare | +1 history vector | Needs startup |

**Recommendation:** SBDF2 is generally the best second-order IMEX multistep scheme for bidomain due to its strong damping. CN-AB is a viable alternative when accuracy is prioritized over damping. Avoid CN-FE for bidomain unless oscillation control measures are in place.

---

## 5. IMEX Runge-Kutta Methods

### Motivation

IMEX multistep methods (SBDF2, CN-AB) require storing previous time levels and need a startup procedure. IMEX Runge-Kutta methods are **single-step** (self-starting) and can achieve higher order without history storage.

### General IMEX-RK Structure

An IMEX Runge-Kutta method with s stages uses two Butcher tableaux — one explicit (Â) and one implicit (A):

$$\mathbf{k}_i^E = \mathbf{f}_E\left(\mathbf{u}^n + \Delta t \sum_{j=1}^{i-1} \hat{a}_{ij} \mathbf{k}_j\right)$$
$$\mathbf{k}_i^I = \mathbf{f}_I\left(\mathbf{u}^n + \Delta t \sum_{j=1}^{i} a_{ij} \mathbf{k}_j\right)$$

The implicit part is restricted to **DIRK** (Diagonally Implicit Runge-Kutta) form so that each stage requires only one linear solve.

### Advantages for Bidomain

1. **Self-starting**: No need for a separate startup procedure
2. **Better stability regions**: IMEX-RK methods generally have larger stability regions than IMEX multistep methods for a given order
3. **L-stability achievable**: SDIRK-based implicit parts can provide L-stability
4. **Natural coupling with operator splitting**: Each RK stage can embed a splitting step

### Disadvantages

1. **More linear solves per step**: An s-stage method requires s implicit solves per time step (vs. 1 for multistep)
2. **Higher memory for stages**: Must store intermediate stage values
3. **Order barriers**: Achieving order > 2 with L-stability requires more stages

### Popular IMEX-RK Schemes for Cardiac Applications

- **IMEX-SSP2(2,2,2)**: 2-stage, 2nd-order, SSP (strong stability preserving) explicit part with L-stable SDIRK implicit part
- **ARK2(3,3,2)** (Kennedy & Carpenter): 3-stage, 2nd-order, good balance of accuracy and stability
- **ARS(2,2,2)** (Ascher, Ruuth, Spiteri): 2-stage, 2nd-order, widely used

### Cost Comparison with Multistep

For the bidomain model, the dominant cost is the linear solve. An s-stage IMEX-RK method costs s× more per step than an IMEX multistep method, but may allow larger Δt due to better stability. In practice:

- **SBDF2**: 1 linear solve per step, Δt limited by ionic model stiffness
- **IMEX-RK (2 stages)**: 2 linear solves per step, potentially 1.5-2× larger Δt

The net efficiency depends on the specific ionic model and mesh. For production bidomain codes, IMEX multistep methods remain more common.

---

## 6. Decoupled (Partitioned) Methods

### Motivation

In the standard approach, V_m and φ_e are solved simultaneously in the coupled 2×2 block system. Decoupled methods solve them separately, reducing the linear system size by half at each sub-step.

### Gauss-Seidel Splitting

Solve the two equations sequentially, using the most recent values:

**Sub-step 1**: Solve the parabolic equation for V_m^{n+1} using φ_e^n:
$$\left(\frac{\chi C_m}{\Delta t} \mathbf{M} + \mathbf{K}_i\right) \mathbf{V}_m^{n+1} = \frac{\chi C_m}{\Delta t} \mathbf{M}\, \mathbf{V}_m^n - \mathbf{K}_i \boldsymbol{\phi}_e^n - \chi \mathbf{M}\, I_{\text{ion}}^n$$

**Sub-step 2**: Solve the elliptic equation for φ_e^{n+1} using the just-computed V_m^{n+1}:
$$(\mathbf{K}_i + \mathbf{K}_e) \boldsymbol{\phi}_e^{n+1} = -\mathbf{K}_i \mathbf{V}_m^{n+1}$$

### Jacobi Splitting

Solve both equations using values from the previous time step only:

**Sub-step 1**: Solve the parabolic equation for V_m^{n+1} using φ_e^n
**Sub-step 2**: Solve the elliptic equation for φ_e^{n+1} using V_m^n (not V_m^{n+1})

### Stability of Decoupled Methods

A key result by Fernández & Zemzemi (2010) shows that **both Gauss-Seidel and Jacobi splittings preserve energy stability** — they simply alter the energy norm. The time-step restrictions are uniquely dictated by the semi-implicit treatment of the ionic term, not by the splitting itself.

This is an important result: decoupling does not compromise stability.

### Advantages of Decoupling

1. **Smaller linear systems**: Each sub-step solves an N×N system instead of 2N×2N
2. **Different solvers per sub-step**: Can use CG+AMG for the SPD elliptic system and a different solver for the parabolic system
3. **Parallelism**: Sub-steps can be pipelined or overlapped

### Disadvantage: Accuracy

Decoupling introduces a splitting error that is O(Δt). To recover second-order accuracy with decoupled methods, techniques like iterative coupling or spectral deferred correction (SDC) are needed.

### The Coupled vs. Decoupled Debate

Interestingly, Cervi & Bhatt (2017) showed that in some cases, the **coupled method can be up to 80% faster** than the decoupled method. The reason: the coupled 2N×2N system, when solved with a good block preconditioner, requires fewer total Krylov iterations than the two separate N×N solves combined. The larger coupled system also exhibits better parallel scaling.

**Bottom line:** Decoupling is simpler to implement but not necessarily faster. The best choice depends on the preconditioner quality and parallel architecture.

---

## 7. Explicit Stabilized Methods: RKC and Multirate RKC

### Background: Runge-Kutta-Chebyshev (RKC) Methods

RKC methods are **fully explicit** s-stage Runge-Kutta methods designed for parabolic PDEs. Their key property: the stability domain grows **quadratically** with the number of stages:

$$\beta_s \approx 0.65\,s^2$$

where β_s is the length of the real stability interval for an s-stage method. This means:

- Standard forward Euler: β₁ = 2 (stability interval [-2, 0])
- RKC with s = 10 stages: β₁₀ ≈ 65
- RKC with s = 50 stages: β₅₀ ≈ 1625

### Why RKC Is Attractive for Cardiac Simulations

1. **Fully explicit**: No linear system solves needed for the parabolic part
2. **Simple to parallelize**: Each stage is a matrix-vector product (like forward Euler)
3. **Adaptive number of stages**: s adjusts automatically based on the spectral radius of the diffusion operator
4. **Memory-efficient**: Only a few vectors needed (independent of s)

### RKC for the Bidomain Parabolic Equation

RKC can be applied to the parabolic equation of the bidomain model:

$$\chi C_m \frac{\partial \mathbf{V}_m}{\partial t} = -\mathbf{K}_i \mathbf{V}_m - \mathbf{K}_i \boldsymbol{\phi}_e - \chi I_{\text{ion}}$$

At each physical time step:
1. **Evaluate φ_e** by solving the elliptic equation (implicit, unavoidable)
2. **Advance V_m** using RKC with s stages chosen to satisfy the stability condition
3. **Advance ionic variables** with Rush-Larsen

The elliptic solve is still needed, but the parabolic part avoids any implicit linear solve. The effective time step is Δt (the physical step), with s internal RKC stages providing stability.

### The Spectral Radius Problem

The number of RKC stages required is:

$$s \geq \sqrt{\frac{\rho(\mathbf{M}^{-1}\mathbf{K}_i)\,\Delta t}{0.65}}$$

where ρ(M⁻¹K_i) is the spectral radius of the discretized diffusion operator. For fine meshes:

| Δx (mm) | ρ (approx) | s needed for Δt=50μs |
|---------|-----------|---------------------|
| 0.5 | ~400 | ~6 |
| 0.2 | ~2,500 | ~14 |
| 0.1 | ~10,000 | ~28 |
| 0.05 | ~40,000 | ~55 |

Each stage requires one matrix-vector product, so the cost scales as O(s) per time step. For fine meshes, this is still cheaper than a preconditioned Krylov solve, but the advantage diminishes.

### Multirate RKC (mRKC)

The stiffness of the cardiac monodomain/bidomain system comes from two sources:
1. **Diffusion operator**: Moderately stiff, ρ ∝ 1/Δx²
2. **Ionic model**: Potentially very stiff (gating variables with τ ~ 0.1 ms)

Standard RKC must match the *most stiff* component, which wastes effort on the less stiff components. **Multirate RKC** (Abdulle et al., 2022) addresses this by:

- Using **fewer stages** for the diffusion (moderate stiffness)
- Using **more stages** (or a different integrator) for the stiff ionic components
- The number of stages is adapted **locally** for each component

### Explicit Exponential Multirate Stabilized (emRKC)

Rosilho de Souza et al. (2024) proposed the emRKC method specifically for cardiac electrophysiology:

1. **Diffusion**: Treated with multirate RKC (explicit, quadratic stability scaling)
2. **Gating variables**: Treated with Rush-Larsen exponential integration (exploits diagonal structure)
3. **Remaining ionic terms**: Treated with forward Euler

**Key result**: emRKC **outperforms the standard IMEX baseline** for the monodomain model. It is faster, inherently more parallel, and achieves comparable accuracy.

### Applicability to Bidomain

**Important caveat**: RKC and emRKC have been demonstrated primarily for the **monodomain** model. For the bidomain:

- The elliptic solve is still required and dominates the cost (82-90% of runtime)
- RKC would only replace the implicit treatment of the parabolic part, which is the cheaper portion
- The benefit of RKC for bidomain is marginal unless the elliptic solve can be made very cheap

**However**, in a decoupled scheme where the elliptic equation is solved separately, RKC could replace the implicit parabolic solve entirely. The algorithmic loop would be:

```
for each time step:
    1. Evaluate I_ion explicitly (Rush-Larsen for gating)
    2. Solve elliptic equation for φ_e (implicit, CG+AMG)
    3. Advance V_m with RKC (explicit, using φ_e from step 2)
```

This hybrid approach combines the strengths of RKC (no implicit parabolic solve, easy parallelism) with the necessary elliptic solve.

---

## 8. Exponential Integrators and Higher-Order Rush-Larsen

### The Rush-Larsen Method (RL1)

The original Rush-Larsen method (1978) exploits the structure of ionic gating variables:

$$\frac{dw}{dt} = \frac{w_\infty(V_m) - w}{\tau_w(V_m)} = a(V_m) - b(V_m)\,w$$

This is linear in w (for fixed V_m), with exact solution:

$$w(t + \Delta t) = w_\infty + (w(t) - w_\infty)\,e^{-\Delta t/\tau_w}$$

Rush-Larsen evaluates w_∞ and τ_w at V_m^n, then applies the exponential update. This is:
- **First-order accurate** (because V_m is frozen at t^n)
- **Unconditionally stable** for the gating variables (the exponential never blows up)
- **Explicit** (no system solve needed)

### Higher-Order Rush-Larsen (RL_k)

Coudière, Douanla Lontsi, and Pierre (2017) generalized Rush-Larsen to higher orders:

**RL2** (second-order, Perego & Veneziani 2009):
Uses a predictor-corrector approach — first predict V_m at t^{n+1/2} or t^{n+1}, then use the corrected ionic parameters in the exponential update.

**RL3 and RL4** (third and fourth order):
These are explicit exponential multistep integrators that achieve order k convergence while maintaining large stability domains. They are shown to be 0-stable (stable under perturbation) with very large stability domains provided the exponential "stabilizer" captures the stiff modes well enough.

### Combining Exponential Integrators with RKC

The emRKC method (Section 7) combines Rush-Larsen for gating variables with RKC for diffusion. This is the state-of-the-art in fully explicit cardiac time stepping.

### Applicability to Bidomain

Exponential integrators apply to the **ionic ODE** part of the bidomain system. They are used in the reaction sub-step of operator splitting or as the explicit part of IMEX schemes. They do not directly address the diffusion or elliptic parts, but they improve the stability of the explicit ionic treatment, allowing larger Δt.

---

## 9. Spectral Deferred Correction (SDC)

### Motivation

Operator splitting introduces a splitting error (O(Δt) for Godunov, O(Δt²) for Strang). Higher-order splitting is possible but becomes unwieldy. Spectral deferred correction (SDC) offers an alternative path to high accuracy.

### The SDC Approach

SDC iteratively corrects a low-order provisional solution to achieve higher-order accuracy:

1. **Provisional solve**: Use a simple first-order method (e.g., IMEX Euler)
2. **Correction sweeps**: Each sweep increases the order by 1 (up to the number of collocation nodes)
3. **After K sweeps**: The method is O(Δt^{K+1}) accurate

### SDC for Bidomain (Gopika et al., 2025)

A recent paper proposes a **partitioned strategy with SDC** for the bidomain model:

1. **Partitioned solve**: The parabolic equation and ionic ODEs are solved together, while the elliptic equation is solved separately
2. **SDC correction**: Applied to the coupled parabolic-ODE system to recover high-order accuracy
3. **Result**: Accuracy comparable to the fully coupled method at significantly lower computational cost

### Key Finding

The CN-RK2 method (Crank-Nicolson for diffusion + RK2 for reaction, within the SDC framework) yields **significantly smaller error** compared to a plain IMEX scheme at the same time step. The SDC correction effectively "repairs" the splitting error without requiring higher-order splitting.

---

## 10. Comparison Table

| Method | Order | Elliptic Solve | Parabolic Treatment | Linear Solves/Step | CFL Restricted? | Oscillations? | Best For |
|--------|-------|---------------|--------------------|--------------------|----------------|--------------|----------|
| **Forward Euler + Elliptic** | 1 | Implicit | Explicit | 1 (elliptic only) | **Yes, severe** | No | Research/testing |
| **RKC + Elliptic** | 2 | Implicit | Explicit (s stages) | 1 (elliptic only) | Relaxed (s² scaling) | No | Moderate meshes |
| **IMEX Euler (SBDF1)** | 1 | Implicit | Implicit (BE) | 1 (coupled) | No | No | Startup step |
| **CN-FE** | 2 | Implicit | Implicit (CN) | 1 (coupled) | No | **Yes** | Avoid for bidomain |
| **SBDF2** | 2 | Implicit | Implicit (BDF2) | 1 (coupled) | No | No | **Production codes** |
| **CN-AB** | 2 | Implicit | Implicit (CN) | 1 (coupled) | No | Mild | Alternative to SBDF2 |
| **IMEX-RK (2-stage)** | 2 | Implicit | Implicit (SDIRK) | 2 (coupled) | No | No | Self-starting needs |
| **Operator Split + CN** | 2 | Implicit | Implicit (CN) | 1 (coupled) | No | **Yes** | Use SDIRK2 instead |
| **Operator Split + SDIRK2** | 2 | Implicit | Implicit (SDIRK) | 2 (coupled) | No | No | **Production codes** |
| **Gauss-Seidel Decoupled** | 1 | Implicit (N×N) | Implicit (N×N) | 2 (separate) | No | No | Simple implementation |
| **SDC Partitioned** | High | Implicit (N×N) | Mixed | K+1 (separate) | No | No | High accuracy needs |
| **emRKC (monodomain only)** | 2 | N/A | Explicit (mRKC) | 0 | Relaxed | No | Monodomain only |

---

## 11. Recommendations for Engine V5.4

### Current Engine V5.4 Capabilities

Engine V5.4 has:
- Operator splitting: Godunov (1st) + Strang (2nd)
- Ionic solvers: Rush-Larsen, Forward Euler
- Diffusion solvers: FE, RK2, RK4 (explicit); CN, BDF1, BDF2 (implicit)
- LBM solver for monodomain (GPU-accelerated, 10-45× faster than FEM)

### Recommended Bidomain Extension Path

**Phase 1 — Baseline Implementation (Operator Splitting + IMEX):**
- Use Strang splitting with SBDF1 (first-order IMEX Euler) for initial correctness testing
- Upgrade to Strang splitting + SDIRK2 for production (second-order, L-stable, no oscillations)
- This leverages the existing operator splitting and implicit solver infrastructure

**Phase 2 — Optimized IMEX (If Needed):**
- Implement SBDF2 as a monolithic IMEX method (no operator splitting)
- This eliminates splitting error entirely and provides the best accuracy/cost ratio
- Requires a startup step with SBDF1

**Phase 3 — Explicit Parabolic (For GPU Optimization):**
- Consider RKC for the parabolic equation combined with CG+AMG for the elliptic equation
- This would allow the parabolic step to run entirely on GPU (like the existing LBM solver)
- Only the elliptic solve needs CPU-side preconditioning

### What NOT to Do

1. **Do not use forward Euler for parabolic diffusion**: The CFL restriction is 25-100× too severe for production meshes
2. **Do not use Crank-Nicolson without oscillation control**: CN produces spurious oscillations near the action potential upstroke. Use SDIRK2 or SBDF2 instead.
3. **Do not expect fully explicit bidomain**: The elliptic constraint always requires an implicit solve. Accept this and optimize the elliptic solver instead.

---

## 12. References

### CFL Condition and Forward Euler Stability
1. Puwal S, Roth BJ. "Forward Euler Stability of the Bidomain Model of Cardiac Tissue." *IEEE Trans Biomed Eng.* 2007;54(5):951-953. DOI: 10.1109/TBME.2006.889204
2. Weber dos Santos R, Plank G, Bauer S, Vigmond EJ. "Parallel multigrid preconditioner for the cardiac bidomain equations." *IEEE Trans Biomed Eng.* 2004;51(11):1960-8.

### Semi-Implicit and IMEX Methods
3. Ethier M, Bourgault Y. "Semi-Implicit Time-Discretization Schemes for the Bidomain Model." *SIAM J Numer Anal.* 2008;46(5):2443-2468. DOI: 10.1137/070680503
4. Whiteley JP. "An efficient numerical technique for the solution of the monodomain and bidomain equations." *IEEE Trans Biomed Eng.* 2006;53(11):2139-47.

### Decoupled Methods
5. Fernández MA, Zemzemi N. "Decoupled time-marching schemes in computational cardiac electrophysiology and ECG numerical simulation." *Math Biosci.* 2010;226(1):58-75. DOI: 10.1016/j.mbs.2010.04.003
6. Cervi J, Bhatt R. "Solving the Coupled System Improves Computational Efficiency of the Bidomain Equations." *PLoS One.* 2017.

### Explicit Stabilized Methods
7. Rosilho de Souza G, Krause R, Bhatt A. "Explicit stabilized multirate methods for the monodomain model in cardiac electrophysiology." *ESAIM: M2AN.* 2024;58(6). DOI: 10.1051/m2an/2024030
8. Abdulle A. "Explicit Stabilized Runge-Kutta Methods." In: *Numerical Time Integration of PDEs*. 2011.

### Exponential Integrators
9. Rush S, Larsen H. "A practical algorithm for solving dynamic membrane equations." *IEEE Trans Biomed Eng.* 1978;25:389-392.
10. Coudière Y, Douanla Lontsi C, Pierre C. "Rush-Larsen time stepping methods of high order for stiff problems in cardiac electrophysiology." *HAL-INRIA.* 2017.
11. Perego M, Veneziani A. "An efficient generalization of the Rush-Larsen method for solving electro-physiology membrane equations." *ETNA.* 2009;35:234-256.

### Spectral Deferred Correction
12. Gopika PB, Peter B, Nagaiah C. "Novel bidomain partitioned strategies for the simulation of ventricular fibrillation dynamics." arXiv:2510.27447. 2025.

### Fully Implicit Methods
13. Torabi Ziaratgahi S, Marsh ME, Bhatt A, Bhatt R. "Composite Backward Differentiation Formula for the Bidomain Equations." *Front Physiol.* 2020. DOI: 10.3389/fphy.2020.00261

### General Reviews
14. Pathmanathan P, et al. "A numerical guide to the solution of the bidomain equations of cardiac electrophysiology." *Prog Biophys Mol Biol.* 2010;102(2-3):136-155.
15. Vigmond EJ, Weber dos Santos R, Prassl AJ, Deo M, Plank G. "Solvers for the Cardiac Bidomain Equations." *Prog Biophys Mol Biol.* 2008;96(1-3):3-18.
16. Krishnamoorthi S, et al. "Stable time integration suppresses unphysical oscillations in the bidomain model." *Front Phys.* 2014;2:40.

### Stability and Oscillation Control
17. Marsh ME, Bhatt R. "Stable time integration suppresses unphysical oscillations in the bidomain model." *Front Phys.* 2014.
