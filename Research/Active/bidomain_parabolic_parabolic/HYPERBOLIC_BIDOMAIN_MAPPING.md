# Hyperbolic Bidomain: BeatIt C++ → Our Bidomain V1 Python Mapping

This document provides a precise, line-number-referenced mapping from the BeatIt
hyperbolic bidomain implementation to our Bidomain Engine V1 Python codebase. It
is intended for direct use during implementation.

---

## 1. BeatIt Architecture Summary

### 1.1 Source Files

| File | Lines | Role |
|------|-------|------|
| `ElectroSolver.hpp` | 252 | Base class: `EquationType` enum (L59-63), `TimeIntegrator` enum (L65-66) |
| `Bidomain.hpp` | 93 | Bidomain subclass declaration |
| `Bidomain.cpp` | 1034 | Full implementation: setup, assembly, RHS, solve |
| `Monowave.cpp` | 1764 | Hyperbolic monodomain reference (same pattern, simpler) |

### 1.2 Equation Type Selection (ElectroSolver.hpp L59-66)

```cpp
enum class EquationType {
    ReactionDiffusion,                  // standard monodomain
    Wave,                               // hyperbolic monodomain (tau > 0)
    ParabolicEllipticBidomain,          // standard PE bidomain (tau_i = tau_e = 0)
    ParabolicEllipticHyperbolic,        // tau_i = tau_e > 0
    ParabolicParabolicHyperbolic        // tau_i != tau_e (most general)
};
```

Selection logic (Bidomain.cpp L248-252):
```
if tau_i == tau_e == 0  →  ParabolicEllipticBidomain     (standard PE)
if tau_i == tau_e > 0   →  ParabolicEllipticHyperbolic   (elliptic unchanged, hyperbolic Vm)
if tau_i != tau_e       →  ParabolicParabolicHyperbolic   (both equations modified)
```

**Key insight**: When `tau_i == tau_e`, the `(tau_i - tau_e)` coupling term vanishes
and the phi_e equation reduces to the standard elliptic form. This is
why that case is called "ParabolicEllipticHyperbolic" -- the phi_e equation
stays elliptic.

### 1.3 The "Wave" Variable Q = dV/dt

The core idea of the hyperbolic formulation is to introduce an auxiliary variable:

    Q(x, t) = dV/dt

This converts the standard bidomain PDE (a parabolic-elliptic system) into a
system where the V equation becomes hyperbolic (second-order in time) through the
relaxation time constants tau_i, tau_e.

**State storage in BeatIt** (Bidomain.cpp L140-173):
- `bidomain_system` stores two coupled unknowns: **Q** (= dV/dt) and **Ve** (= phi_e)
- `wave_system` stores one unknown: **V** (= Vm)
- The linear solve produces (Q^{n+1}, Ve^{n+1}), then V is updated explicitly

### 1.4 The tau_i, tau_e Parameters

Physical meaning: intracellular/extracellular relaxation times (ms). They
parameterize inductive ("hyperbolic") behavior in the respective domains.

Read from input (Bidomain.cpp L225-226):
```cpp
double tau_i = M_datafile(section + "/tau_i", 0.0);
double tau_e = M_datafile(section + "/tau_e", 0.0);
```

When both are zero, the formulation reduces identically to the standard
parabolic-elliptic bidomain.

### 1.5 Matrix Assembly (Bidomain.cpp L343-784)

The assembled system matrix is a 2x2 block system in (Q, Ve):

```
| A_QQ    A_QVe  |   | Q^{n+1}  |     | rhs_Q  |
|                 | * |          |  =  |        |
| A_VeQ   A_VeVe |   | Ve^{n+1} |     | rhs_Ve |
```

**Block QQ** (L674): `(1 + tau_i/cdt) * Cm * M_lumped + cdt * K_i`
- `M_lumped` = lumped mass matrix (diagonal)
- `K_i` = intracellular stiffness matrix (standard diffusion)
- `cdt` = effective dt (= dt for SBDF1, = 2/3*dt for SBDF2)

**Block QVe** (L678): `K_i`
- Just the intracellular stiffness (coupling Q to Ve)

**Block VeQ** (L682): `(tau_i - tau_e)/cdt * Cm * M_lumped + cdt * K_i`
- Zero when tau_i == tau_e (reduces to standard elliptic)

**Block VeVe** (L687): `K_ie = K_i + K_e`
- Combined intra+extra stiffness (standard elliptic operator)

### 1.6 RHS Assembly — SBDF1 (Bidomain.cpp L904-919)

For the first time step (Backward-Forward Euler, BFE):

```
RHS_Q  = tau_i/cdt * Cm * M * Q^n  -  M*(Iion + tau_i*dIion + Istim)  -  Ki*V^n
RHS_Ve = (tau_i-tau_e)/cdt * Cm * M * Q^n  +  (tau_e-tau_i)*M*dIion  -  Ki*V^n
```

Breaking this down:
- `tau_i/cdt * Cm * Q^n`: history from the Q time derivative (uses lumped mass)
- `-Iion - tau_i*dIion - Istim`: ionic current + its time derivative + stimulus (uses consistent mass)
- `-Ki*V^n`: diffusion of current V (applied via matrix-vector product)
- The Ve row has `(tau_i-tau_e)` coupling terms that vanish when tau_i==tau_e

### 1.7 RHS Assembly — SBDF2 (Bidomain.cpp L879-901)

For steps 1+ with second-order time integration:

```
cdt = 2/3 * dt

RHS_Q  = tau_i/cdt * Cm * M * (4/3*Q^n - 1/3*Q^{n-1})
       - M*(2*Iion^n - Iion^{n-1} + 2*tau_i*dIion^n - tau_i*dIion^{n-1} + Istim)
       - Ki*(4/3*V^n - 1/3*V^{n-1})

RHS_Ve = (tau_i-tau_e)/cdt * Cm * M * (4/3*Q^n - 1/3*Q^{n-1})
       + (tau_e-tau_i)*(2*dIion^n - dIion^{n-1})
       - Ki*(4/3*V^n - 1/3*V^{n-1})
```

The SBDF2 extrapolation coefficients are: 4/3 for n, -1/3 for n-1, and the
ionic terms use 2*f^n - f^{n-1} (second-order Adams-Bashforth extrapolation).

### 1.8 dIion/dt Computation

BeatIt stores the time derivative of the ionic current `dIion` in a vector
`iion_system.get_vector("diion")`. This is **computed during the reaction step**
(in ElectroSolver's `solve_reaction_step()`), not in the diffusion step.

The computation is: `dIion^n = (Iion^n - Iion^{n-1}) / dt` (first-order finite difference).

For SBDF2, two history levels are needed:
- `diion`: current dIion^n
- `diion_old`: previous dIion^{n-1}

Sign convention (Bidomain.cpp L820): "we store in iion -I^n and in diion -dI^n"
-- i.e., Iion and dIion have the sign as used in the RHS (negated from the
standard convention where Iion is positive outward).

### 1.9 V Update After Solve (Bidomain.cpp L990-1031)

After solving for (Q^{n+1}, Ve^{n+1}):

**SBDF1** (L1018-1021):
```
V^{n+1} = V^n + dt * Q^{n+1}
```

**SBDF2** (L1024-1029):
```
V^{n+1} = 4/3*V^n - 1/3*V^{n-1} + 2/3*dt * Q^{n+1}
```

This is an **explicit reconstruction** — no linear solve needed for V.

### 1.10 Time Integrator Switching

- Step 0: always SBDF1 (Backward-Forward Euler) regardless of setting
- Step 1+: SBDF2 if configured, otherwise stays SBDF1
- At step 1, the matrix is reassembled with the SBDF2 coefficient
  `cdt = 2/3*dt` (Bidomain.cpp L970-973)

---

## 2. Our Architecture Summary

### 2.1 Current PE Bidomain Solver (decoupled_gs.py)

File: `Bidomain/Engine_V1/cardiac_sim/simulation/classical/solver/diffusion_stepping/decoupled_gs.py`

**Constructor** (L48-60): Takes `spatial`, `dt`, `parabolic_solver`, `elliptic_solver`, `theta`, `pin_node`.

**`_build_operators()`** (L62-70):
- `A_para, B_para = spatial.get_parabolic_operators(dt, theta)` — from fdm.py L248-266:
  - `A_para = 1/dt * I - theta * L_i`
  - `B_para = 1/dt * I + (1-theta) * L_i`
- `A_ellip = spatial.get_elliptic_operator()` — `-(L_i + L_e)`

**`step()`** (L72-111):
```python
# Step 1: Parabolic
rhs_para = B_para @ Vm + L_i @ phi_e      # Full coupling (NOT theta*L_i*phi_e)
Vm_new = parabolic_solver.solve(A_para, rhs_para)

# Step 2: Elliptic
rhs_ellip = L_i @ Vm_new
phi_e_new = elliptic_solver.solve(A_ellip, rhs_ellip)
```

### 2.2 Spectral Solver (spectral.py)

File: `Bidomain/Engine_V1/cardiac_sim/simulation/classical/solver/linear_solver/spectral.py`

Solves `-D * Lap(u) = b` via spectral transforms. Eigenvalues are precomputed:
- Neumann: `lam_k = 2/dx^2 * (1 - cos(pi*k/N))`
- Dirichlet: `lam_k = 2/dx^2 * (1 - cos(pi*(k+1)/(m+1)))`

The solve is: `u_hat = b_hat / eigenvalues` (L228).

The matrix A passed to `solve(A, b)` is **ignored** — eigenvalues encode the operator.

### 2.3 BidomainState (state.py)

File: `Bidomain/Engine_V1/cardiac_sim/simulation/classical/state.py`

Current fields (L22-78):
- `Vm: torch.Tensor` — transmembrane potential (n_dof,)
- `phi_e: torch.Tensor` — extracellular potential (n_dof,)
- `ionic_states: torch.Tensor` — gates + concentrations (n_dof, n_states)
- `Cm: float` — membrane capacitance
- `t: float` — current time

**Missing for hyperbolic**: Q (= dV/dt), Iion_prev, dIion, dIion_prev, Vm_prev, Q_prev.

### 2.4 Orchestrator (bidomain.py)

File: `Bidomain/Engine_V1/cardiac_sim/simulation/classical/bidomain.py`

Factory `_build_diffusion_solver()` (L203-223): dispatches by name string.
Already has `'imex_sbdf2'` entry pointing to `IMEXSBDF2Solver`.

### 2.5 Existing IMEX SBDF2 (imex_sbdf2.py)

File: `Bidomain/Engine_V1/cardiac_sim/simulation/classical/solver/diffusion_stepping/imex_sbdf2.py`

This is the standard PE bidomain with BDF2 time integration. It already:
- Self-starts with BDF1
- Stores `_Vm_prev` for 2nd-order history
- Builds `A_bdf1` and `A_bdf2` operators with different `1/dt` coefficients

This is the **closest existing code** to what we need.

### 2.6 Ionic Stepping (rush_larsen.py)

File: `Bidomain/Engine_V1/cardiac_sim/simulation/classical/solver/ionic_stepping/rush_larsen.py`

`step()` (L45-98):
1. Computes `Iion = model.compute_Iion(V, S)` (L70)
2. Updates V: `V_new = V + dt * (-(Iion + Istim) / Cm)` (L84)
3. Updates gates via Rush-Larsen
4. Updates concentrations via Forward Euler

**Missing for hyperbolic**: does not store Iion, does not compute dIion/dt.

### 2.7 Splitting (strang.py / godunov.py)

Strang (L31-45): `ionic(dt/2) → diffusion(dt) → ionic(dt/2)`
Godunov (L30-42): `ionic(dt) → diffusion(dt)`

Both pass the full `state` to the diffusion solver, which reads Vm, phi_e and
writes them back in-place.

---

## 3. The Mapping: BeatIt → Our Code

### 3.1 tau_i, tau_e Parameters

**BeatIt**: Read in `setup_systems()` (L225-229), stored in
`M_equationSystems.parameters`.

**Our code**: Not present anywhere.

**Where to add**:
- `BidomainConductivity` (tissue_builder): add `tau_i: float = 0.0`, `tau_e: float = 0.0`
- OR simpler: pass `tau_i`, `tau_e` directly to the new diffusion solver constructor
  (like `theta` is passed to `DecoupledBidomainDiffusionSolver`).

**Recommendation**: Add to the diffusion solver constructor. These are time-stepping
parameters, not spatial discretization parameters. Keep it minimal:

```python
class HyperbolicBidomainSolver(BidomainDiffusionSolver):
    def __init__(self, spatial, dt, parabolic_solver, elliptic_solver,
                 tau_i=0.0, tau_e=0.0, pin_node=0):
```

### 3.2 EquationType Selection

**BeatIt**: `EquationType` enum (ElectroSolver.hpp L59-63), selected at setup
based on tau_i/tau_e values (Bidomain.cpp L248-252).

**Our code**: No enum. Solver type chosen by string in `_build_diffusion_solver()`.

**Mapping**:
- `tau_i == tau_e == 0` → existing `DecoupledBidomainDiffusionSolver` (unchanged)
- `tau_i > 0 or tau_e > 0` → new `HyperbolicBidomainSolver`

Internally the new solver can branch on `tau_i == tau_e` to skip the `(tau_i - tau_e)`
coupling terms, but this is an optimization, not a structural change.

### 3.3 The Q (= dV/dt) Auxiliary Variable

**BeatIt**: Stored in `bidomain_system` as the first of two coupled unknowns
(Bidomain.cpp L142: `bidomain_system.add_variable("Q")`). After solving, Q is
used to update V (L990-1031).

**Our code**: No equivalent. Vm is updated directly by the diffusion solver.

**What to add**:
1. New field `Q: torch.Tensor` in `BidomainState` (n_dof,)
2. New field `Q_prev: Optional[torch.Tensor]` for SBDF2 history
3. New field `Vm_prev: Optional[torch.Tensor]` for SBDF2 V history
4. Initialize Q to zero (dV/dt = 0 at rest)
5. Initialize Q_prev, Vm_prev to None

### 3.4 dIion/dt Computation

**BeatIt**: Computed as `dIion = (Iion^n - Iion^{n-1}) / dt`. Stored in
`iion_system.get_vector("diion")`. For SBDF2, also stores `diion_old`.

**Our code**: Iion is computed inside `RushLarsenSolver.step()` (L70) but
immediately consumed and discarded. No history is kept.

**What to add**:
1. New fields in `BidomainState`:
   - `Iion_prev: Optional[torch.Tensor]` — Iion from previous step
   - `dIion: Optional[torch.Tensor]` — current dIion/dt
   - `dIion_prev: Optional[torch.Tensor]` — previous dIion/dt (for SBDF2)
2. The ionic solver must be modified (or wrapped) to:
   a. Compute Iion BEFORE the voltage update
   b. Store it in state so the diffusion solver can access it
   c. Compute `dIion = (Iion_current - Iion_prev) / dt`

**Critical detail**: The splitting order matters. In BeatIt, the reaction step
comes first (Godunov-like), and Iion is computed there. Then the diffusion step
uses Iion and dIion. Our Strang splitting does ionic(dt/2) → diffusion → ionic(dt/2).
For the hyperbolic formulation, the diffusion step needs Iion, so the first
half-ionic step must compute and store it.

### 3.5 Mass Matrix (M) vs Our FDM Identity

**BeatIt** uses FEM with mass matrices (lumped and consistent). In their
formulation, terms like `M * Iion` appear because the weak form introduces mass
matrices.

**Our code** uses FDM (finite differences), where the mass matrix is the identity
(all nodes equally weighted on a uniform grid, with `dx*dy` volume absorbed into
the operator scaling).

**Mapping**:
- Their `M * Iion` → our `Iion` (identity mass)
- Their `M_lumped * Q^n` → our `Q^n` (identity mass)
- Their `K_i * V^n` → our `L_i @ V^n` (stiffness = our Laplacian, sign convention: our L_i is negative semi-definite)

**CRITICAL SIGN CONVENTION**: Their stiffness K is assembled as `int(grad(phi_i) . D . grad(phi_j))`,
which is the **negative** of our Laplacian L. So:
- Their `K_i` = `-L_i` in our notation
- Their `A_ellip = K_ie` in block VeVe → our `A_ellip = -(L_i + L_e)` (same thing)

### 3.6 Block QQ Operator → Spectral Eigenvalue Modification

**BeatIt** (L674):
```
A_QQ = (1 + tau_i/cdt) * Cm * M_lumped + cdt * K_i
```

**Our equivalent** (FDM, identity mass):
```
A_QQ = (1 + tau_i/cdt) * Cm * I + cdt * (-L_i)
     = (1 + tau_i/cdt) * Cm / cdt * cdt * I - cdt * L_i     [rearranging]
```

Wait -- let's be more precise. In our notation where L_i is neg-semi-def:
```
A_QQ = (1 + tau_i/cdt) * Cm * I - cdt * L_i
```

Compare with our standard parabolic operator:
```
A_para = 1/dt * I - theta * L_i
```

The hyperbolic A_QQ has the same structure but with:
- Mass coefficient: `(1 + tau_i/cdt) * Cm` instead of `1/dt`
- Stiffness coefficient: `cdt` instead of `theta`

For the **spectral solver**, we need eigenvalues of A_QQ. Since L_i has
eigenvalues `-D_i * lambda_k` (where lambda_k are the discrete Laplacian
eigenvalues), the A_QQ eigenvalues are:

```
mu_k = (1 + tau_i/cdt) * Cm + cdt * D_i * lambda_k
```

This can be implemented by creating a `SpectralSolver` with modified eigenvalues.

**Alternatively**, since we only need to solve `A_QQ * Q = rhs`, and A_QQ is SPD
with the same sparsity as A_para, we can reuse the existing spectral solver
infrastructure by building a custom eigenvalue array:

```python
# In the hyperbolic solver's operator construction:
lam = spectral_solver._eigenvalues  # D_ie * lambda_k (from elliptic)

# For QQ block: need D_i * lambda_k
lam_i = (D_i / (D_i + D_e)) * lam   # scale to get D_i * lambda_k

A_QQ_eigenvalues = (1 + tau_i/cdt) * Cm + cdt * lam_i
```

BUT: the spectral solver currently only solves `-D*Lap(u) = b` (pure Laplacian).
For the hyperbolic system, the QQ block is `(mass + stiffness)`, which is not a
pure Laplacian. The spectral approach still works because the eigenvectors are
the same (discrete cosines/sines), but the eigenvalue formula changes.

**Implementation approach**: Create a new spectral-compatible solver mode that
accepts custom eigenvalues, or build the eigenvalue array directly in the
hyperbolic solver and do the forward/inverse transforms manually.

### 3.7 Coupled 2x2 System vs Decoupled GS

**BeatIt** solves the full 2N x 2N coupled system in one shot (Bidomain.cpp L987):
```cpp
M_linearSolver->solve(*bidomain_system.matrix, *bidomain_system.solution, *bidomain_system.rhs, tol, max_iter);
```

**Our code** uses Gauss-Seidel decoupling: solve Vm first, then phi_e.

**For the hyperbolic case**, the 2x2 block system has off-diagonal blocks:
- A_QVe = K_i (intracellular stiffness)
- A_VeQ = `(tau_i-tau_e)/cdt * Cm * M + cdt * K_i`

When `tau_i == tau_e`, A_VeQ simplifies to `cdt * K_i`, and the system becomes:

```
| (1+tau/cdt)*Cm*I - cdt*L_i    -L_i       |   | Q   |   | rhs_Q  |
|                                            | * |     | = |        |
| cdt*(-L_i)                    -(L_i+L_e)  |   | Ve  |   | rhs_Ve |
```

This has the same qualitative structure as the standard PE bidomain (upper-left is
"parabolic-like", lower-right is "elliptic-like", off-diagonals couple them).
**Gauss-Seidel decoupling still works**:

1. Solve for Q: `A_QQ * Q^{n+1} = rhs_Q - A_QVe * Ve^n` (lag Ve)
2. Solve for Ve: `A_VeVe * Ve^{n+1} = rhs_Ve - A_VeQ * Q^{n+1}` (use new Q)
3. Update V: `V^{n+1} = V^n + dt * Q^{n+1}` (explicit)

When `tau_i != tau_e`, the VeQ block has an extra mass term, but the decoupled
approach still applies with the VeQ contribution moved to the RHS.

### 3.8 RHS Assembly Mapping (SBDF1 case)

**BeatIt** (L904-919) → **Our code** mapping:

| BeatIt term | Our equivalent | Notes |
|-------------|---------------|-------|
| `tau_i/cdt * Cm * M * Q^n` | `tau_i/cdt * Cm * state.Q` | Lumped mass = identity for FDM |
| `-(Iion + tau_i*dIion + Istim)` | `-(state.Iion + tau_i*state.dIion + Istim)` | Sign: Iion stored with physical sign |
| `M * ionic_terms` | `ionic_terms` | Consistent mass → identity for FDM |
| `-Ki * V^n` | `L_i @ state.Vm` | Ki = -L_i, so -Ki = +L_i |
| `(tau_i-tau_e)/cdt * Cm * M * Q^n` | `(tau_i-tau_e)/cdt * Cm * state.Q` | Ve row coupling |
| `(tau_e-tau_i) * dIion` | `(tau_e-tau_i) * state.dIion` | Ve row ionic |

So the SBDF1 RHS in our notation:

```python
# Q equation
rhs_Q = tau_i/cdt * Cm * Q_n \
      - (Iion + tau_i * dIion + Istim) \
      + L_i @ Vm_n

# Ve equation
rhs_Ve = (tau_i - tau_e)/cdt * Cm * Q_n \
       + (tau_e - tau_i) * dIion \
       + L_i @ Vm_n
```

### 3.9 RHS Assembly Mapping (SBDF2 case)

```python
cdt = 2.0/3.0 * dt

# BDF2 history terms
Q_extrap = (4.0/3.0) * Q_n - (1.0/3.0) * Q_nm1
V_extrap = (4.0/3.0) * Vm_n - (1.0/3.0) * Vm_nm1
Iion_extrap = 2.0 * Iion_n - Iion_nm1
dIion_extrap = 2.0 * dIion_n - dIion_nm1

# Q equation
rhs_Q = tau_i/cdt * Cm * Q_extrap \
      - (Iion_extrap + tau_i * dIion_extrap + Istim) \
      + L_i @ V_extrap

# Ve equation
rhs_Ve = (tau_i - tau_e)/cdt * Cm * Q_extrap \
       + (tau_e - tau_i) * dIion_extrap \
       + L_i @ V_extrap
```

### 3.10 V Update Mapping

**SBDF1** (BeatIt L1018-1021):
```python
Vm_new = Vm_n + dt * Q_new
```

**SBDF2** (BeatIt L1024-1029):
```python
Vm_new = 4.0/3.0 * Vm_n - 1.0/3.0 * Vm_nm1 + 2.0/3.0 * dt * Q_new
```

### 3.11 Reduction to Standard PE Bidomain

When tau_i = tau_e = 0, the formulation must reduce to our existing PE solver.
Let's verify:

With tau_i = tau_e = 0, cdt = dt:
- A_QQ = `Cm * I - dt * L_i` = `1/dt * I - L_i` (when Cm = 1, dividing by dt... hmm)

Actually wait. Let me re-examine the BeatIt operator more carefully.

BeatIt's A_QQ (L674): `(1 + tau_i/cdt) * Cm * M_lumped + cdt * K_i`

With tau_i = 0, cdt = dt:
`A_QQ = Cm * M + dt * K_i = Cm * I - dt * L_i`

Our A_para = `1/dt * I - theta * L_i` (with theta=1 for backward Euler, Cm=1).

These are NOT the same! BeatIt has `Cm*I - dt*L_i` while ours has `1/dt*I - L_i`.

The difference is that BeatIt's system solves for Q (not Vm), and the RHS is
already scaled differently. Let me trace through the full BeatIt system when
tau=0 to understand the equivalence.

**BeatIt with tau=0, SBDF1 (monodomain Monowave for clarity)**:

System matrix (Monowave L1528): `Cm * (1 + 0) * M + dt * K = Cm*M + dt*K`
RHS (Monowave L1626-1661): `-(Iion + Istim)*M - K*V^n`
Solve for Q^{n+1}, then: `V^{n+1} = V^n + dt * Q^{n+1}`

Substituting Q = (V^{n+1} - V^n)/dt into the system:
```
Cm*M*(V^{n+1}-V^n)/dt + dt*K*(V^{n+1}-V^n)/dt = -(Iion+Istim)*M - K*V^n
Cm*M*(V^{n+1}-V^n)/dt + K*(V^{n+1}-V^n) = -(Iion+Istim)*M - K*V^n
```
Hmm, this is not quite standard. But with tau=0, we can see that BeatIt's wave
formulation is still slightly different from our direct V formulation because
they solve for Q and then integrate. When tau=0, the "wave" auxiliary variable
is just a reformulation -- mathematically equivalent in the limit but
numerically using Q as the primary unknown.

**Key realization**: The hyperbolic solver is NOT a modification of the existing
PE solver. It is a **new solver** with different primary unknowns (Q, Ve instead
of Vm, phi_e). When tau → 0, it converges to the PE solution but through a
different numerical path.

This means we should implement it as a **new diffusion stepping class** rather
than modifying `DecoupledBidomainDiffusionSolver`.

---

## 4. Implementation Plan

### 4.0 Prerequisites

Verify understanding: the hyperbolic solver is a NEW solver class that lives
alongside the existing PE solvers. It shares the spatial discretization and
linear solver infrastructure but has different time stepping logic.

### 4.1 File: `state.py` — Add Hyperbolic State Fields

**File**: `/home/norepinephrine/Documents/Heart-Conduction/Bidomain/Engine_V1/cardiac_sim/simulation/classical/state.py`

Add after `phi_e` field (L54):

```python
# Hyperbolic bidomain auxiliary fields (None when using standard PE)
Q: Optional[torch.Tensor] = None          # dV/dt, (n_dof,)
Q_prev: Optional[torch.Tensor] = None     # Q^{n-1} for SBDF2
Vm_prev: Optional[torch.Tensor] = None    # V^{n-1} for SBDF2

# Ionic current history for hyperbolic formulation
Iion_current: Optional[torch.Tensor] = None   # Iion at current step
Iion_prev: Optional[torch.Tensor] = None      # Iion at previous step
dIion: Optional[torch.Tensor] = None           # dIion/dt at current step
dIion_prev: Optional[torch.Tensor] = None      # dIion/dt at previous step
```

Update `clone()` (L112-131) to copy these fields.

### 4.2 File: `ionic_stepping/rush_larsen.py` — Store Iion

**File**: `/home/norepinephrine/Documents/Heart-Conduction/Bidomain/Engine_V1/cardiac_sim/simulation/classical/solver/ionic_stepping/rush_larsen.py`

After computing Iion (L70), store it in state:

```python
# Store Iion for hyperbolic bidomain diffusion step
if hasattr(state, 'Iion_current') and state.Iion_current is not None:
    # Shift history
    if state.Iion_prev is None:
        state.Iion_prev = state.Iion_current.clone()
    else:
        state.Iion_prev.copy_(state.Iion_current)
    # Compute dIion
    if state.dIion is not None:
        if state.dIion_prev is None:
            state.dIion_prev = state.dIion.clone()
        else:
            state.dIion_prev.copy_(state.dIion)
    state.dIion = (Iion - state.Iion_current) / dt
    state.Iion_current.copy_(Iion)
elif state.Q is not None:
    # First call -- initialize
    state.Iion_current = Iion.clone()
    state.Iion_prev = Iion.clone()
    state.dIion = torch.zeros_like(Iion)
    state.dIion_prev = torch.zeros_like(Iion)
```

**Important**: Also store Istim in state so the diffusion solver can access it.
Add to ionic base class `_evaluate_Istim` or store after evaluation.

Actually, a cleaner approach: add a method to the ionic solver (or a small
wrapper) that computes and stores Iion + dIion WITHOUT doing the voltage
update. The voltage update should be handled differently for the hyperbolic
case (V is updated from Q, not from Iion).

**Alternative approach (recommended)**: Instead of modifying the existing ionic
solver, have the hyperbolic diffusion solver compute Iion itself before the
diffusion step. This keeps the ionic solver unchanged but requires the diffusion
solver to have access to the ionic model. This is messier architecturally.

**Best approach**: The ionic solver step should still compute Iion and update
gates/concentrations, but for the hyperbolic case, the voltage update
`V += dt * (-Iion/Cm)` should be SKIPPED in the ionic step (because V is
updated from Q after the diffusion step). Instead, store Iion in state.

This means we need a **variant ionic step mode** for the hyperbolic case.

### 4.3 New File: `diffusion_stepping/hyperbolic_bidomain.py`

**New file**: `Bidomain/Engine_V1/cardiac_sim/simulation/classical/solver/diffusion_stepping/hyperbolic_bidomain.py`

This is the main implementation. Structure:

```python
class HyperbolicBidomainSolver(BidomainDiffusionSolver):
    """
    Hyperbolic bidomain solver with SBDF1/SBDF2 time integration.

    Solves the coupled (Q, Ve) system where Q = dV/dt, then reconstructs V.

    Reduces to standard PE bidomain when tau_i = tau_e = 0.

    Parameters
    ----------
    spatial : BidomainSpatialDiscretization
    dt : float
    parabolic_solver : LinearSolver  (for Q sub-problem)
    elliptic_solver : LinearSolver   (for Ve sub-problem)
    tau_i : float  — intracellular relaxation time (ms)
    tau_e : float  — extracellular relaxation time (ms)
    Cm : float     — membrane capacitance (uF/cm^2)
    pin_node : int
    """

    def __init__(self, spatial, dt, parabolic_solver, elliptic_solver,
                 tau_i=0.0, tau_e=0.0, Cm=1.0, pin_node=0):
        super().__init__(spatial, dt)
        self.tau_i = tau_i
        self.tau_e = tau_e
        self.Cm = Cm
        self.parabolic_solver = parabolic_solver
        self.elliptic_solver = elliptic_solver
        self._needs_pinning = spatial.grid.boundary_spec.phi_e_has_null_space
        self._pin_node = pin_node
        self._step_count = 0
        self._build_operators(spatial, dt)

    def _build_operators(self, spatial, dt):
        """Build operators for SBDF1 and SBDF2."""
        # SBDF1: cdt = dt
        # SBDF2: cdt = 2/3 * dt
        self._build_operators_for_cdt(spatial, dt, suffix='_bdf1')
        self._build_operators_for_cdt(spatial, 2.0/3.0 * dt, suffix='_bdf2')

        # Elliptic operator (unchanged)
        self.A_ellip = spatial.get_elliptic_operator()
        if self._needs_pinning:
            self.A_ellip = self.apply_elliptic_pinning(self.A_ellip, self._pin_node)

    def _build_operators_for_cdt(self, spatial, cdt, suffix):
        """Build A_QQ for a given cdt (effective time step).

        A_QQ = (1 + tau_i/cdt) * Cm * I - cdt * L_i

        This is the same structure as a parabolic operator but with
        modified mass and stiffness coefficients.
        """
        n = spatial.n_dof
        I = _speye(n, ...)
        A_QQ = (1.0 + self.tau_i / cdt) * self.Cm * I - cdt * spatial.L_i
        # Store with suffix
        setattr(self, f'A_QQ{suffix}', A_QQ.coalesce())

    def step(self, state, dt):
        if self._step_count == 0:
            self._step_sbdf1(state, dt)
        else:
            self._step_sbdf2(state, dt)
        self._step_count += 1

    def _step_sbdf1(self, state, dt):
        cdt = dt
        Cm = self.Cm
        tau_i = self.tau_i
        tau_e = self.tau_e

        Q_n = state.Q
        Vm_n = state.Vm
        Iion = state.Iion_current    # stored by ionic step
        dIion = state.dIion
        Istim = ...                  # evaluate from state

        # Save history for SBDF2
        state.Q_prev = Q_n.clone()
        state.Vm_prev = Vm_n.clone()

        # --- RHS_Q ---
        rhs_Q = tau_i / cdt * Cm * Q_n \
              - (Iion + tau_i * dIion + Istim) \
              + self._spatial.apply_L_i(Vm_n)

        # --- Solve for Q^{n+1} (Gauss-Seidel: lag Ve) ---
        # A_QQ * Q^{n+1} = rhs_Q + L_i @ Ve^n   (coupling from A_QVe block)
        rhs_Q = rhs_Q + self._spatial.apply_L_i(state.phi_e)
        Q_new = self.parabolic_solver.solve(self.A_QQ_bdf1, rhs_Q)

        # --- RHS_Ve ---
        rhs_Ve = (tau_i - tau_e) / cdt * Cm * Q_n \
               + (tau_e - tau_i) * dIion \
               + self._spatial.apply_L_i(Vm_n)

        # A_VeVe * Ve^{n+1} = rhs_Ve + coupling from A_VeQ * Q^{n+1}
        # A_VeQ = (tau_i-tau_e)/cdt * Cm * I + cdt * (-L_i)
        # coupling = A_VeQ @ Q_new ... but in GS, we fold VeQ into RHS
        # Actually: the VeQ block contributes to the RHS.
        # After GS decoupling, the Ve equation RHS absorbs the VeQ*Q^{n+1} term.
        #
        # Full Ve RHS = rhs_Ve_standalone + A_VeQ @ Q_new
        # But A_VeQ contains stiffness + mass terms...
        #
        # Let's reconsider. In BeatIt they solve the coupled system directly.
        # For our decoupled approach, we can use the fact that the Ve equation
        # has the form:
        #   A_VeVe * Ve = rhs_Ve_assembled
        # where rhs_Ve_assembled includes all Q-dependent terms.
        #
        # The Ve equation in BeatIt's assembled form is:
        #   [A_VeQ | A_VeVe] * [Q; Ve] = rhs_Ve_vec
        # → A_VeVe * Ve = rhs_Ve_vec - A_VeQ * Q^{n+1}
        #
        # A_VeQ from assembly (L682-683):
        #   (tau_i-tau_e)/cdt * Cm * M_lumped (only Q-mass part)
        #   + cdt * K_i (stiffness part)
        #
        # But wait: the stiffness Ki in block VeQ (L683) uses the SAME Ki
        # as in block QQ. And Ki*Q is not the same as Ki*V.
        # Actually looking more carefully at the assembly, line 683:
        #   Ke(i + n_Q_dofs, j) += cdt * JxW * DigradV * dphi[j][qp];
        # This is cdt * K_i operating on Q (the first variable).
        #
        # So A_VeQ = (tau_i-tau_e)/cdt * Cm * M_lumped + cdt * K_i
        # In our FDM notation: (tau_i-tau_e)/cdt * Cm * I - cdt * L_i
        #
        # The RHS_Ve from the node loop (L910-915) is the "source" part only.
        # The stiffness part (-Ki*V^n) is added separately via KiV (L954-955).
        # And the matrix blocks VeQ*Q and VeVe*Ve are on the LHS.
        #
        # For our GS decoupling:
        # Solve Ve from: A_VeVe * Ve^{n+1} = rhs_Ve_source - A_VeQ * Q^{n+1}

        # But this requires constructing A_VeQ explicitly. Alternatively,
        # we can compute the Ve RHS directly as BeatIt does.

        # SIMPLER APPROACH: Build the full Ve RHS mimicking BeatIt's assembly.
        # After solving for Q^{n+1}, compute the full Ve RHS:
        rhs_Ve_full = (tau_i - tau_e) / cdt * Cm * Q_new \
                    + (tau_e - tau_i) * dIion \
                    + self._spatial.apply_L_i(Vm_n)  # note: this is -Ki*V^n in BeatIt's notation

        # Wait, I need to re-examine. In BeatIt, the Ve RHS includes -Ki*V^n
        # but the LHS includes Ki*Q (block VeQ) and Kie*Ve (block VeVe).
        # The "-Ki*V^n" term is the explicit part from the time stepping.
        # The "A_VeQ * Q" is part of the implicit LHS.
        #
        # In the GS approach, after solving Q, we move everything to the RHS:
        #
        # A_VeVe * Ve = full_rhs_Ve
        # where full_rhs_Ve = rhs_Ve_nodal + lumped_mass*rhs_Ve_old - Ki*(kv)
        #                     - A_VeQ_mass_part * Q^{n+1} - A_VeQ_stiff_part * Q^{n+1}
        #
        # Hmm, this is getting convoluted because of BeatIt's FEM mass matrices.
        # Let me re-derive from the PDE for FDM.

        # === CLEAN DERIVATION FOR FDM ===
        # (See section 3.8 for the RHS in our notation)
        #
        # The full system (FDM, identity mass) for SBDF1:
        #
        # QQ block: [(1 + tau_i/dt)*Cm*I - dt*L_i] * Q^{n+1}
        # QVe block: [-L_i] * Ve^{n+1}
        #   = tau_i/dt*Cm*Q^n - (Iion + tau_i*dIion + Istim) + L_i*Vm^n
        #
        # VeQ block: [(tau_i-tau_e)/dt*Cm*I - dt*L_i] * Q^{n+1}
        # VeVe block: [-(L_i+L_e)] * Ve^{n+1}
        #   = (tau_i-tau_e)/dt*Cm*Q^n + (tau_e-tau_i)*dIion + L_i*Vm^n
        #
        # GS Step 1: Solve for Q with Ve^n lagged
        # A_QQ * Q^{n+1} = rhs_Q + L_i * Ve^n     (move QVe block to RHS with old Ve)
        #
        # Wait, the QVe block is -L_i acting on Ve, so:
        # A_QQ * Q + (-L_i) * Ve = rhs_Q
        # → A_QQ * Q = rhs_Q + L_i * Ve^n
        #
        # GS Step 2: Solve for Ve with Q^{n+1} known
        # A_VeVe * Ve^{n+1} = rhs_Ve - A_VeQ * Q^{n+1}
        # where A_VeQ = (tau_i-tau_e)/dt*Cm*I - dt*L_i
        # → A_VeVe * Ve = rhs_Ve - (tau_i-tau_e)/dt*Cm*Q^{n+1} + dt*L_i*Q^{n+1}
        #
        # But L_i*Vm^n already appears in rhs_Ve. And the VeQ terms introduce
        # additional L_i*Q^{n+1} and mass*Q^{n+1} terms.
        #
        # Full Ve RHS after GS:
        # = (tau_i-tau_e)/dt*Cm*Q^n + (tau_e-tau_i)*dIion + L_i*Vm^n
        #   - (tau_i-tau_e)/dt*Cm*Q^{n+1} + dt*L_i*Q^{n+1}
        #
        # = (tau_i-tau_e)/dt*Cm*(Q^n - Q^{n+1}) + (tau_e-tau_i)*dIion
        #   + L_i*(Vm^n + dt*Q^{n+1})
        #
        # Note: Vm^n + dt*Q^{n+1} = Vm^{n+1} (the updated V!)
        #
        # So: rhs_Ve_full = (tau_i-tau_e)/dt*Cm*(Q^n - Q^{n+1})
        #                 + (tau_e-tau_i)*dIion
        #                 + L_i*Vm^{n+1}
        #
        # When tau_i == tau_e: rhs_Ve_full = L_i*Vm^{n+1}
        # This is EXACTLY our standard elliptic equation!

        # Compute Vm^{n+1} first
        Vm_new = Vm_n + dt * Q_new

        # Full Ve RHS
        rhs_Ve_final = self._spatial.apply_L_i(Vm_new)
        if tau_i != tau_e:
            rhs_Ve_final += (tau_i - tau_e) / dt * Cm * (Q_n - Q_new)
            rhs_Ve_final += (tau_e - tau_i) * dIion

        # Elliptic solve for Ve
        self._zero_dirichlet_rhs(rhs_Ve_final)
        if self._needs_pinning:
            rhs_Ve_final[self._pin_node] = 0.0
        phi_e_new = self.elliptic_solver.solve(self.A_ellip, rhs_Ve_final)
        if self._needs_pinning:
            phi_e_new -= phi_e_new[self._pin_node]

        # Update state
        state.Q.copy_(Q_new)
        state.Vm.copy_(Vm_new)
        state.phi_e.copy_(phi_e_new)
```

**KEY INSIGHT FROM THE DERIVATION ABOVE**: After the Gauss-Seidel decoupling
and substituting Vm^{n+1} = Vm^n + dt*Q^{n+1}:

1. When `tau_i == tau_e`: the Ve equation reduces EXACTLY to the standard
   elliptic equation `A_ellip * Ve = L_i * Vm^{n+1}`. The elliptic solver
   and its spectral implementation are completely unchanged.

2. When `tau_i != tau_e`: the Ve equation has additional source terms from the
   `(tau_i - tau_e)` coupling, but the operator `A_ellip = -(L_i + L_e)` is
   still the same. Only the RHS changes. The spectral solver still works.

3. The Q equation has a MODIFIED operator `A_QQ = (1+tau_i/cdt)*Cm*I - cdt*L_i`.
   This is spectrally compatible (same eigenvectors as the Laplacian) but with
   different eigenvalues. We need a spectral solver variant for this.

### 4.4 Spectral Solver for the Q Sub-Problem

The Q operator `A_QQ = alpha*I - beta*L_i` has eigenvalues:
```
mu_k = alpha + beta * D_i * lambda_k
```
where `lambda_k` are the discrete Laplacian eigenvalues (same ones the spectral
solver already computes), `alpha = (1 + tau_i/cdt) * Cm`, `beta = cdt`.

Two approaches:

**Approach A**: Modify `SpectralSolver` to accept a `mass_shift` parameter:
```python
class SpectralSolver:
    def __init__(self, ..., mass_shift=0.0):
        self._mass_shift = mass_shift
    # In solve():
    eigenvalues = self._eigenvalues + mass_shift  # shifted eigenvalues
```

**Approach B**: Create a wrapper that replaces eigenvalues:
```python
class ShiftedSpectralSolver:
    """Solves (alpha*I + beta*(-Lap_D))*u = b via spectral transform."""
    def __init__(self, base_spectral, alpha, beta_over_D):
        self._base = base_spectral
        # eigenvalues of A = alpha + beta * (D_i * lam_k)
        # base._eigenvalues = D * lam_k for the elliptic case
        # We need D_i * lam_k = (D_i / D) * base._eigenvalues
        self._eigenvalues = alpha + beta_over_D * base._eigenvalues
```

**Recommendation**: Approach A is simpler. Add an optional `mass_coeff` parameter
to `SpectralSolver.solve()` or create a lightweight `SpectralHelmholtzSolver`
subclass.

Actually, the simplest approach: build the eigenvalue array directly in the
hyperbolic solver and reuse the spectral transforms from the existing solver:

```python
# In HyperbolicBidomainSolver._build_operators:
# Reuse the elliptic spectral solver's infrastructure
# but compute Q-specific eigenvalues

# Get Laplacian eigenvalues per axis (same as spectral solver)
lam_x = SpectralSolver._axis_eigenvalues(bc_x, nx, dx, device, dtype)
lam_y = SpectralSolver._axis_eigenvalues(bc_y, ny, dy, device, dtype)
LAM_X, LAM_Y = torch.meshgrid(lam_x, lam_y, indexing='ij')
laplacian_eigs = D_i * (LAM_X + LAM_Y)

# Q operator eigenvalues for SBDF1:
self._Q_eigs_bdf1 = (1 + tau_i/dt) * Cm + dt * laplacian_eigs
# Q operator eigenvalues for SBDF2:
cdt2 = 2.0/3.0 * dt
self._Q_eigs_bdf2 = (1 + tau_i/cdt2) * Cm + cdt2 * laplacian_eigs
```

Then implement the Q solve using the spectral transforms directly:
```python
def _solve_Q(self, rhs, step_count):
    eigs = self._Q_eigs_bdf2 if step_count > 0 else self._Q_eigs_bdf1
    rhs_2d = rhs.reshape(self._nx, self._ny)
    rhs_hat = _fwd(_fwd(rhs_2d, dim=0, bc=self._bc_x), dim=1, bc=self._bc_y)
    u_hat = rhs_hat / eigs
    u_work = _inv(_inv(u_hat, dim=0, bc=self._bc_x), dim=1, bc=self._bc_y)
    if torch.is_complex(u_work):
        u_work = u_work.real
    return u_work.flatten()
```

This reuses the spectral transform functions from `spectral.py` but with custom
eigenvalues. No modification to the existing `SpectralSolver` needed.

### 4.5 Splitting Modification for Hyperbolic

The current splitting assumes the ionic step updates Vm directly:
`Vm_new = Vm + dt * (-Iion/Cm)`. For the hyperbolic case, Vm is updated from Q
in the diffusion step, not in the ionic step.

**Option A (clean)**: Create a new splitting strategy `HyperbolicSplitting` that:
1. Calls the ionic solver to update gates + concentrations + store Iion
2. Does NOT update Vm in the ionic step
3. Calls the diffusion solver which produces (Q, Ve, Vm)

**Option B (minimal)**: Add a flag to the ionic solver to skip the Vm update.
The diffusion solver then handles Vm = Vm_old + dt * Q_new.

**Recommendation**: Option B. Add `skip_voltage_update: bool = False` to the
ionic solver step. When True, the ionic step computes Iion and updates gates
and concentrations but leaves Vm unchanged. The hyperbolic diffusion solver
updates Vm via the Q integration.

### 4.6 File: `bidomain.py` — Orchestrator Registration

**File**: `Bidomain/Engine_V1/cardiac_sim/simulation/classical/bidomain.py`

Add to `_build_diffusion_solver()` (after L218):
```python
elif name == 'hyperbolic':
    from .solver.diffusion_stepping.hyperbolic_bidomain import HyperbolicBidomainSolver
    return HyperbolicBidomainSolver(
        spatial, dt, para_ls, ellip_ls,
        tau_i=kwargs.get('tau_i', 0.0),
        tau_e=kwargs.get('tau_e', 0.0))
```

Add `tau_i`, `tau_e` parameters to `BidomainSimulation.__init__()`.

### 4.7 File: `__init__.py` — Export

**File**: `Bidomain/Engine_V1/cardiac_sim/simulation/classical/solver/diffusion_stepping/__init__.py`

Add import and export of `HyperbolicBidomainSolver`.

---

## 5. Summary: Files to Modify/Create

| # | Action | File | Changes |
|---|--------|------|---------|
| 1 | **MODIFY** | `state.py` | Add Q, Q_prev, Vm_prev, Iion_current, Iion_prev, dIion, dIion_prev fields |
| 2 | **MODIFY** | `ionic_stepping/rush_larsen.py` | Add Iion storage + dIion computation; add `skip_voltage_update` flag |
| 3 | **MODIFY** | `ionic_stepping/base.py` | Add `skip_voltage_update` parameter to `step()` |
| 4 | **CREATE** | `diffusion_stepping/hyperbolic_bidomain.py` | New solver: Q/Ve/Vm update with SBDF1/SBDF2, spectral Q-solve |
| 5 | **MODIFY** | `diffusion_stepping/__init__.py` | Add import/export |
| 6 | **MODIFY** | `bidomain.py` | Add `tau_i`, `tau_e` params; register `'hyperbolic'` solver |
| 7 | **MODIFY** | `splitting/strang.py` + `godunov.py` | Pass `skip_voltage_update` to ionic step when diffusion is hyperbolic |

**Estimated total diff**: ~250 lines new code (hyperbolic_bidomain.py), ~60 lines
modifications across existing files.

### Implementation Order

1. `state.py` — add fields (non-breaking, all Optional with None default)
2. `ionic_stepping/base.py` + `rush_larsen.py` — Iion storage + skip flag
3. `hyperbolic_bidomain.py` — new solver (the core)
4. `__init__.py` + `bidomain.py` — wiring
5. `splitting/` — pass skip flag (or create wrapper)
6. Test: run with tau_i=tau_e=0 and verify it matches standard PE
7. Test: run with tau_i=tau_e=0.01 and verify wave-like behavior

---

## Appendix A: Complete PDE Reference

### Standard Parabolic-Elliptic Bidomain (tau_i = tau_e = 0)

```
Cm * dVm/dt = div(D_i * grad(Vm + phi_e)) - Iion
0 = div((D_i + D_e) * grad(phi_e)) + div(D_i * grad(Vm))
```

### Hyperbolic Bidomain (tau_i, tau_e > 0)

```
Cm * (dVm/dt + tau_i * d²Vm/dt²) = div(D_i * grad(Vm + phi_e)) - (Iion + tau_i * dIion/dt)
(tau_i - tau_e) * Cm * d²Vm/dt² = div((D_i + D_e) * grad(phi_e)) + div(D_i * grad(Vm))
                                  + (tau_i - tau_e) * dIion/dt
```

With Q = dVm/dt, this becomes:

```
Cm * (Q + tau_i * dQ/dt) = div(D_i * grad(Vm + phi_e)) - (Iion + tau_i * dIion/dt)
(tau_i - tau_e) * Cm * dQ/dt = div((D_i + D_e) * grad(phi_e)) + div(D_i * grad(Vm))
                               + (tau_i - tau_e) * dIion/dt
```

### Reduction Cases

- `tau_i = tau_e = tau > 0`: Second equation becomes `0 = div((D_i+D_e)*grad(phi_e)) + div(D_i*grad(Vm))` (standard elliptic, unchanged)
- `tau_i = tau_e = 0`: Standard PE bidomain

## Appendix B: Sign Convention Cross-Reference

| Quantity | BeatIt convention | Our convention |
|----------|-------------------|---------------|
| Stiffness K_i | `int(grad(phi).D_i.grad(phi))` (positive) | L_i (negative semi-definite) |
| K_i * V | positive diffusion | L_i @ V (negative of diffusion) |
| Iion | stored negated in some places | positive outward (standard cardiac) |
| Istim | stored as-is | negative = depolarizing |
| dIion | stored negated | to be determined (follow Iion sign) |
| A_ellip | K_ie (positive) | -(L_i + L_e) (positive definite) |

The critical sign relationship: **Their `-Ki*V` = our `+L_i @ V`** (because Ki = -L_i).

## Appendix C: BeatIt Line Number Quick Reference

| Function | Lines | Purpose |
|----------|-------|---------|
| `setup_systems` | 133-256 | Create systems, read params, select equation type |
| `init_systems` | 258-341 | Set ICs, fibers, conductivity |
| `assemble_matrices` | 343-784 | Build 2x2 block matrix + mass matrices + Ki |
| Block QQ assembly | 674 | `(1+tau_i/cdt)*Cm*M + cdt*Ki` |
| Block QVe assembly | 678 | `Ki` (intracellular stiffness) |
| Block VeQ assembly | 682-683 | `(tau_i-tau_e)/cdt*Cm*M + cdt*Ki` |
| Block VeVe assembly | 687 | `Kie = Ki + Ke` |
| `form_system_rhs` | 786-965 | Build RHS vectors for (Q, Ve) |
| SBDF2 RHS | 879-903 | Second-order extrapolation |
| SBDF1 RHS | 904-920 | First-order (backward-forward Euler) |
| Ki*V^n applied | 939-956 | Diffusion of current V added to both rows |
| `solve_diffusion_step` | 967-1032 | Solve system, update V from Q |
| SBDF1 V update | 1018-1021 | `V = V^n + dt * Q^{n+1}` |
| SBDF2 V update | 1024-1029 | `V = 4/3*V^n - 1/3*V^{n-1} + 2/3*dt*Q^{n+1}` |
| SBDF2 matrix rebuild | 970-973 | Reassemble at step 1 with cdt=2/3*dt |
