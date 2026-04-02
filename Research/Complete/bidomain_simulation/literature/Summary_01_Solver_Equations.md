# Solver Equations & Linear Systems for Cardiac Electrophysiology

## 1. The Monodomain Equation

### Mathematical Formulation

The monodomain equation is a simplified model describing transmembrane voltage propagation in cardiac tissue:

$$\chi \cdot C_m \cdot \frac{dV}{dt} = -\chi \cdot I_{ion}(V,u) + \nabla \cdot (\mathbf{D} \cdot \nabla V) + I_{stim}$$

**Standard reaction-diffusion rearrangement:**

$$\frac{dV}{dt} = \frac{1}{\chi \cdot C_m} \nabla \cdot (\mathbf{D} \cdot \nabla V) - \frac{I_{ion}}{C_m} + \frac{I_{stim}}{\chi \cdot C_m}$$

### Physical Interpretation

- **χ**: Surface-to-volume ratio (cm⁻¹) — controls coupling between voltage and current
- **Cm**: Membrane capacitance (μF/cm²) — capacitive response of cell membrane
- **D**: Anisotropic diffusion tensor — fiber and transverse conductivity
- **Iion(V,u)**: Ionic current (pA/pF) — non-linear state-dependent term
- **Istim**: Stimulus current — imposed external forcing

### Typical Parameters

| Parameter | Value | Unit | Notes |
|-----------|-------|------|-------|
| Cm | 1.0 | μF/cm² | Standard mammalian tissue |
| χ | 1400 | cm⁻¹ | Typical ventricular tissue |
| D_fiber | 0.001 | cm²/ms | Longitudinal conductivity |
| D_cross | 0.00025 | cm²/ms | Transverse conductivity |
| Anisotropy ratio | ~4:1 | — | Fiber direction preferred |

### Derivation from Bidomain

The monodomain equation can be derived from bidomain when the **equal anisotropy ratio assumption** holds:

$$\frac{\sigma_{i,l}}{\sigma_{i,t}} = \frac{\sigma_{e,l}}{\sigma_{e,t}}$$

This simplification reduces two coupled PDEs to a single parabolic equation, significantly reducing computational cost while maintaining physiological accuracy in many scenarios.

---

## 2. The Bidomain Equations

### Mathematical Formulation

The bidomain model consists of two coupled PDEs describing intracellular and extracellular electric potentials:

**Parabolic equation (temporal dynamics):**

$$\chi \cdot C_m \cdot \frac{dV_m}{dt} = \nabla \cdot (\sigma_i \cdot \nabla V_m) + \nabla \cdot (\sigma_i \cdot \nabla \varphi_e) - \chi \cdot I_{ion} + I_{stim}$$

**Elliptic equation (quasi-steady extracellular):**

$$\nabla \cdot ((\sigma_i + \sigma_e) \cdot \nabla \varphi_e) = -\nabla \cdot (\sigma_i \cdot \nabla V_m)$$

### Key Variables

- **Vm**: Transmembrane potential (intracellular minus extracellular) — physiological variable
- **φe**: Extracellular potential (absolute reference) — auxiliary variable
- **σi**: Intracellular conductivity tensor (mS/cm) — anisotropic
- **σe**: Extracellular conductivity tensor (mS/cm) — anisotropic

### Operator Decomposition

Two primary differential operators required:

- **Ai**: Represents $\nabla \cdot (\sigma_i \cdot \nabla \cdot)$ (diffusion with intracellular conductivity)
- **Asum**: Represents $\nabla \cdot ((\sigma_i + \sigma_e) \cdot \nabla \cdot)$ (elliptic operator)

### Typical Conductivity Values (Clerc 1976)

| Conductivity | Value | Unit |
|--------------|-------|------|
| σi_longitudinal | 1.74 | mS/cm |
| σi_transverse | 0.19 | mS/cm |
| σe_longitudinal | 6.25 | mS/cm |
| σe_transverse | 2.36 | mS/cm |

**Anisotropy ratios:**
- Intracellular: 1.74/0.19 ≈ 9.2:1
- Extracellular: 6.25/2.36 ≈ 2.6:1

---

## 3. Operator Splitting Strategies

Operator splitting decomposes the combined reaction-diffusion system into sequential reaction (ionic) and diffusion (electroporation) steps, enabling specialized solvers for each physics.

### Godunov Splitting (1st Order)

**Order of operations:**

1. Ionic step: $\frac{dV}{dt} = -I_{ion}(V,u)$ over $[t^n, t^{n+1}]$
2. Diffusion step: $\frac{dV}{dt} = \frac{1}{\chi C_m} \nabla \cdot (\mathbf{D} \nabla V)$ over $[t^n, t^{n+1}]$

**Local truncation error:** O(dt²)

**Stability:** Depends on individual solvers; generally stable for moderate dt

### Strang Splitting (2nd Order)

**Order of operations:**

1. Half-ionic step: Ionic from $t^n$ to $t^{n+1/2}$ (dt/2)
2. Full-diffusion step: Diffusion from $t^n$ to $t^{n+1}$ (full dt)
3. Half-ionic step: Ionic from $t^{n+1/2}$ to $t^{n+1}$ (dt/2)

**Local truncation error:** O(dt³)

**Stability:** Better for stiff systems; improves convergence in time

### Vigmond 3-Step Bidomain Splitting

Specialized for bidomain equations:

1. **Parabolic CN step**: Implicit Crank-Nicolson on parabolic equation with current approximation
2. **Reaction step**: Integrate gating variables and concentrations
3. **Elliptic solve**: Solve extracellular equation for φe (static in time step)

**Advantages:** Handles diffusion-dominated and reaction-dominated regimes efficiently

### Gating Variable Update (Rush-Larsen)

Analytical solution for linear gating kinetics:

$$x^{n+1} = x_\infty - (x_\infty - x^n) \exp(-\Delta t / \tau)$$

Where:
- **x**: Gating variable (dimensionless, 0-1 range)
- **x∞**: Asymptotic/steady-state value at current voltage
- **τ**: Time constant (ms) — voltage-dependent
- **Δt**: Time step

**Advantages:** Accurate for stiff gating variables; unconditionally stable

**Implementation note:** Derived from $\frac{dx}{dt} = (x_\infty - x)/\tau$ with constant coefficients

### Concentration and Voltage Reaction (Forward Euler)

Forward Euler for non-gating variables:

$$y^{n+1} = y^n + \Delta t \cdot f(V^n, y^n)$$

Where f represents ion concentration fluxes or metabolite changes.

**Trade-off:** Simpler, lower accuracy than Rush-Larsen but sufficient for slowly-varying concentrations

---

## 4. Time Stepping Methods

### Forward Euler (Explicit, 1st Order)

**Discrete scheme:**

$$V^{n+1} = V^n + \frac{\Delta t}{C_m} \mathbf{L} V^n$$

Where **L** is the discrete Laplacian operator.

**CFL Stability Condition:**

$$\Delta t \leq \frac{C_m h^2}{4 D_{max}}$$

**Practical guidance:**
- h = grid spacing (cm)
- D_max = maximum conductivity component
- Restrictive for fine meshes — quadratic time complexity O(1/h²)

**Advantages:** Simple to implement, no linear solve

**Disadvantages:** Accuracy limited by stability; expensive for fine spatial resolution

### Crank-Nicolson (Implicit, 2nd Order)

**Discrete scheme:**

$$\left(I - \frac{\Delta t}{2 C_m} \mathbf{L}\right) V^{n+1} = \left(I + \frac{\Delta t}{2 C_m} \mathbf{L}\right) V^n$$

**Rewritten as linear system:**

$$\mathbf{A} V^{n+1} = b$$

Where $\mathbf{A} = I - \frac{\Delta t}{2 C_m} \mathbf{L}$ and $b = (I + \frac{\Delta t}{2 C_m} \mathbf{L}) V^n + \text{sources}$

**Stability:** Unconditionally stable (no CFL restriction)

**Accuracy:** Second-order in time O(Δt²), second-order in space O(h²)

**Advantages:**
- Unconditional stability
- Second-order convergence
- No CFL time step restriction

**Disadvantages:**
- Requires linear solve each time step
- Slightly more dissipative than RK4

### Backward Differentiation Formula (BDF)

**BDF1 (same as implicit Euler):**

$$\frac{3V^{n+1} - 4V^n + V^{n-1}}{2\Delta t} = \mathbf{L} V^{n+1}$$

**Order:** 1st, Local truncation error O(Δt²)

**Stability:** A-stable

**BDF2 (2nd order):**

$$\frac{3V^{n+1} - 4V^n + V^{n-1}}{2\Delta t} = \mathbf{L} V^{n+1}$$

**Order:** 2nd, Local truncation error O(Δt³)

**Stability:** A-stable up to 3rd order

### Runge-Kutta 2 (RK2, Heun's Method, 2nd Order Explicit)

**Stage 1:** Predict

$$V^* = V^n + \Delta t \, \mathbf{L} V^n$$

**Stage 2:** Correct

$$V^{n+1} = V^n + \frac{\Delta t}{2}(\mathbf{L} V^n + \mathbf{L} V^*)$$

**Laplacian evaluations:** 2 per time step

**Order:** 2nd, Local truncation error O(Δt³)

**Stability:** Conditional, CFL ≈ 1.0

### Runge-Kutta 4 (RK4, 4th Order Explicit)

**Standard four-stage method:**

$$k_1 = \Delta t \, \mathbf{L} V^n$$
$$k_2 = \Delta t \, \mathbf{L} (V^n + k_1/2)$$
$$k_3 = \Delta t \, \mathbf{L} (V^n + k_2/2)$$
$$k_4 = \Delta t \, \mathbf{L} (V^n + k_3)$$
$$V^{n+1} = V^n + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

**Laplacian evaluations:** 4 per time step

**Order:** 4th, Local truncation error O(Δt⁵)

**Stability:** Conditional, CFL ≈ 2.8

**Reference solution:** RK4 provides highly accurate reference solutions for method validation and error analysis

**Advantages:**
- High order of accuracy
- Stable for moderate time steps
- Provides excellent reference solutions

**Disadvantages:**
- Multiple Laplacian evaluations per step
- CFL-limited

---

## 5. Linear Systems from Implicit Methods

### General System Form

Implicit time stepping discretization yields:

$$\mathbf{A} \cdot \mathbf{x} = \mathbf{b}$$

**System matrix:**

$$\mathbf{A} = \mathbf{M} + \theta \, \Delta t \, \mathbf{K}$$

**Right-hand side:**

$$\mathbf{b} = (\mathbf{M} - (1-\theta) \, \Delta t \, \mathbf{K}) \, V^n + \text{source terms}$$

**Time integration parameter:**
- θ = 0: Forward Euler (explicit)
- θ = 0.5: Crank-Nicolson (implicit midpoint)
- θ = 1.0: Backward Euler (fully implicit)

### Finite Difference Method (FDM) with Identity Mass

**Simplified form (no explicit mass matrix):**

$$\mathbf{A} = I - \theta \frac{\Delta t}{\chi C_m} \mathbf{L}$$

Where **L** is the discrete Laplacian operator matrix.

**Advantages:**
- No matrix inversion for mass matrix
- Smaller memory footprint
- Faster matrix-vector products

**Disadvantages:**
- Requires uniform grid structure
- Limited geometric flexibility

### Finite Element Method (FEM) with Consistent Mass

**System matrix with explicit mass matrix:**

$$\mathbf{A} = \frac{\mathbf{M}}{\Delta t} + \theta \, \mathbf{K}$$

**Right-hand side:**

$$\mathbf{b} = \frac{\mathbf{M}}{\Delta t} V^n + \text{sources}$$

Where:
- **M**: Consistent mass matrix (N×N, SPD)
- **K**: Stiffness matrix from diffusion (N×N, SPSD)

**Advantages:**
- Handles unstructured meshes
- Better geometric representation
- Higher-order elements possible

**Disadvantages:**
- Larger matrices (more memory)
- Requires mass matrix assembly and factorization

### Matrix Properties

**Symmetry:** Both **A** and **K** are symmetric under standard discretization

**Positive Definiteness:**
- **A is SPD** when **M is SPD** and **K is SPSD** (true for diffusion operators)
- Guarantees unique solution and enables CG, PCG solvers

**Condition Number:**
- **Parabolic problems:** κ(A) ∈ O(1) to O(10²)
- **Elliptic problems:** κ(A) ∈ O(1/h²) (worsens with finer mesh)
- Preconditioning essential for elliptic problems

### Scaling with Discretization

**Degrees of freedom:** N = (1/h³) for 3D problems

**Matrix storage:** O(N) for sparse stencils (7-point FDM)

**Time to solve:** Depends on solver:
- Direct: O(N^3) → prohibitive for large N
- Iterative (CG): O(κ(A) · N) iterations × O(N) per matvec = O(κ(A) · N²)

---

## 6. Elliptic System Singularity (Bidomain)

### Mathematical Singularity

The extracellular potential equation:

$$\nabla \cdot ((\sigma_i + \sigma_e) \cdot \nabla \varphi_e) = -\nabla \cdot (\sigma_i \cdot \nabla V_m)$$

exhibits singularity because:

**Operator deficiency:** $\mathbf{K}_{ie} = \mathbf{A}_{ie}$ (symmetric Laplacian with conducting boundary) has a **rank deficiency of 1**

**Null space:** Contains constant vectors (1-vectors)
- Any solution φe is determined only up to an additive constant
- Physically: Extracellular potential has arbitrary reference zero

**Rank:** rank(K_ie) = N - 1 (for N nodes)

### Resolution Approaches

#### 1. Node Pinning (Dirichlet Condition)

**Method:** Fix potential at one node to φe(node_ref) = 0

**Implementation:**
- Modify matrix row for reference node: K_ie[ref, :] = 0, K_ie[ref, ref] = 1
- Modify RHS: b[ref] = 0

**Advantages:** Simplest, minimal overhead

**Disadvantages:** May cause ill-conditioning at pinned node

#### 2. Nullspace Deflation in CG

**Principle:** Project residuals to orthogonal complement of nullspace

**Algorithm:**

After standard CG matvec, compute residual r = b - A·x, then:

$$r_{deflated} = r - \frac{r \cdot \mathbf{1}}{\mathbf{1} \cdot \mathbf{1}} \mathbf{1}$$

Where **1** is the all-ones vector (basis for nullspace).

**Advantage:** Elegant, maintains symmetry, no matrix modification

**Disadvantage:** Extra dot products per iteration

#### 3. Regularization

**Method:** Add regularization term: A_reg = A + ε·M where ε ≈ 10⁻⁶

**Trade-off:** Slightly perturbs solution but eliminates singularity

### Compatibility Condition

**Constraint:** For consistent system, compatibility condition must hold:

$$\int_{\Omega} (-\nabla \cdot (\sigma_i \cdot \nabla V_m)) \, d\Omega = 0$$

By divergence theorem, this requires:

$$\int_{\partial \Omega} \sigma_i \nabla V_m \cdot \mathbf{n} \, dS = 0$$

**Physical meaning:** No net current escapes domain (isolated tissue preparation)

**Automatic satisfaction:** Proper operator splitting ensures compatibility is maintained

---

## 7. Connection to V5.4 Architecture

### Spatial Discretization Abstraction

**SpatialDiscretization** abstract base class provides:

- **Operator M**: Mass matrix (for FEM) or identity (for FDM)
- **Operator K**: Stiffness matrix from diffusion operator
- **Operator L**: Discrete Laplacian (for monodomain)

**Responsibilities:**
- Mesh topology management
- Basis function definition
- Matrix assembly and sparsity pattern
- Boundary condition handling

### Diffusion Solver Interface

**DiffusionSolver** abstract base class:

- **Owns:** LinearSolver instance (CG, GMRES, direct, etc.)
- **Consumes:** Operators (M, K, L) from SpatialDiscretization
- **Produces:** Voltage update V^{n+1}

**Methods:**
- `setup(spatialDisc)`: Initialize solvers, preconditioners, factorizations
- `solve(V_n, dt, forcing_terms) → V_n+1`: Execute single time step

### Splitting Strategy Orchestration

**SplittingStrategy** abstract base class manages integration sequence:

```
For each time step:
  1. Compute Iion(V^n, gates^n) — IonicModel
  2. Update gating: x^{n+1} ← IonicModel::update_gates()
  3. Update concentration: c^{n+1} ← IonicModel::update_concentrations()
  4. Diffusion step: V^{n+1} ← DiffusionSolver::solve(...)
```

**Variants:**
- GodunenovSplitting: Ionic → Diffusion
- StrangSplitting: Ionic_half → Diffusion → Ionic_half
- Vigmond3StepBidomain: Parabolic → Reaction → Elliptic

### Ionic Model Interface

**IonicModel** abstract base class provides:

- `compute_Iion(V, gates, concentrations) → Iion`: Current calculation
- `get_gate_state() → vector`: Current gating variable values
- `update_gates(V, gates, dt) → gates_new`: Gating kinetics solver
- `update_concentrations(current, dt) → conc_new`: Ion balance

**Responsibility partition:**
- **IonicModel decides:** Which gating variables are stiff (Rush-Larsen vs. Forward Euler)
- **SplittingStrategy decides:** How to interleave ionic/diffusion steps
- **DiffusionSolver decides:** Implicit vs. explicit diffusion stepping

### SimulationState Management

**SimulationState** holds:

- `V`: Transmembrane voltage (monodomain) or Vm (bidomain)
- `phi_e`: Extracellular potential (bidomain only)
- `gates`: All gating variables {h, j, m, d, f, ...}
- `concentrations`: Ion pools {Na_i, K_i, Ca_sr, ...}

**Design pattern:**
- **Zero allocation per step:** Pre-allocated in constructor
- **In-place updates:** Operators modify state in-place
- **No temporary copies:** Memory-efficient for long simulations

**Update flow:**
```
V_current ← SimulationState
gates_new, concentrations_new ← IonicModel::update(...)
V_new ← DiffusionSolver::solve(V_current, ...)
SimulationState ← {V_new, gates_new, concentrations_new, ...}
```

### Integration with Solver Pipeline

**Full solve sequence (Strang splitting example):**

```
Initialize: Create SpatialDiscretization, DiffusionSolver, IonicModel, SplittingStrategy
Setup: spatialDisc→assemble(), diffusionSolver→setup(spatialDisc)

For each time step t^n → t^{n+1}:
  1. Ionic half-step:
     gates_half ← ionicModel→update_gates(V^n, gates^n, dt/2)
     conc_half ← ionicModel→update_concentrations(..., dt/2)

  2. Diffusion full-step:
     Iion ← ionicModel→compute_Iion(V^n, gates_half, conc_half)
     V^{n+1} ← diffusionSolver→solve(V^n, Iion, dt)

  3. Ionic half-step:
     gates^{n+1} ← ionicModel→update_gates(V^{n+1}, gates_half, dt/2)
     conc^{n+1} ← ionicModel→update_concentrations(..., dt/2)

  4. Update state:
     simulationState ← {V^{n+1}, gates^{n+1}, conc^{n+1}, ...}
```

---

## Summary Table: Method Properties

| Aspect | Monodomain | Bidomain | Comments |
|--------|-----------|----------|----------|
| **PDEs** | 1 parabolic | 1 parabolic + 1 elliptic | Bidomain more complex |
| **Parameters** | χ, Cm, D_l, D_t | χ, Cm, σ_i, σ_e | Bidomain requires 4 tensors |
| **Computational** | ~2x faster | Baseline | Extra elliptic solve |
| **Accuracy** | Good if anisotropy ratio equal | Highest fidelity | Captures extracellular effects |
| **Linear systems** | Parabolic (κ ~ 10²) | Mixed (κ ~ 10²-1/h²) | Elliptic requires preconditioning |

---

## References and Further Reading

- **Classical Bidomain:** Plonsey & Barr (1986), Henriquez (1993)
- **Operator Splitting:** Godunov (1959), Strang (1968), Vigmond et al. (2003)
- **Gating Variables:** Rush & Larsen (1978)
- **Finite Element Methods:** Fitzhugh-Nagumo models, FEM discretization of diffusion
- **Iterative Solvers:** Saad (2003), Templates for solving linear systems

---

**Document Version:** 1.0
**Last Updated:** 2025-02
**Status:** Research Summary for openCARP FDM/FVM Project
