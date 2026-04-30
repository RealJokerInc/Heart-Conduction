# LBM_V1 — Naming Conventions and Symbol Reference

## Mathematical Symbols → Code Names

We follow the conventions established by Rapaka et al. (2012) and Campos et al. (2016),
using the standard LBM notation from the lattice-Boltzmann community.

### Primary Variables

| Symbol | Meaning | Code name | Type / Shape |
|--------|---------|-----------|-------------|
| V | Transmembrane potential | `V` | `(Nx, Ny)` tensor |
| f_i | Distribution function for direction i | `f` | `(Q, Nx, Ny)` tensor |
| f*_i | Post-collision distribution | `f_star` | `(Q, Nx, Ny)` tensor |
| f_i^eq | Equilibrium distribution | `f_eq` | `(Q, Nx, Ny)` tensor |
| v | Recovered macroscopic voltage = sum(f_i) | `V` (same — they are the same thing) | |

### Lattice Constants

| Symbol | Meaning | Code name | Notes |
|--------|---------|-----------|-------|
| Q | Number of discrete velocities | `Q` | 5 (D2Q5), 9 (D2Q9) |
| e_i | Discrete velocity vectors | `e` | Tuple of (dx, dy) per direction |
| w_i | Lattice weights | `w` | Tuple of floats, sum = 1.0 |
| c_s^2 | Speed of sound squared | `cs2` | Always 1/3 for D2Q5, D2Q9 |
| opposite(i) | Index of reverse direction | `opposite` | Tuple of ints |

### Relaxation Parameters

| Symbol | Meaning | Code name | Notes |
|--------|---------|-----------|-------|
| tau | BGK relaxation time (scalar) | `tau` | Must be > 0.5 |
| omega | BGK collision frequency = 1/tau | `omega` | |
| tau_ij | MRT relaxation time tensor (flux moments) | `tau_ij` | 2x2 (D2Q5) or 2x2 (D2Q9) |
| S | Relaxation rate matrix in moment space | `S_diag` or `S_matrix` | Diagonal or block-structured |
| M | Transformation matrix (distribution -> moment space) | `M` | (Q x Q) |
| M^{-1} | Inverse transformation matrix | `M_inv` | (Q x Q) |

### Collision Operators

| Symbol | Meaning | Code name | Notes |
|--------|---------|-----------|-------|
| Omega | Total collision operator | (not used directly — split into parts) | |
| Omega^NR | Non-reactive collision (diffusion) | `collide_nr` or `Omega_NR` | Relaxation toward equilibrium |
| Omega^R | Reactive collision (source/ionic) | `Omega_R` | = w_i * R |
| R | Reaction source term | `R` | = -(I_ion + I_stim) / (chi * Cm) |
| A | Collision matrix = M^{-1} S M | (hand-coded, not stored) | |

Following Campos et al. Eq. (7):
```
f_i(x + e_i*dx, t + dt) - f_i(x, t) = Omega(x, t)
```

Where:
```
Omega = Omega_NR + Omega_R
Omega_NR = -M^{-1} S M (f - f_eq)     (MRT, Campos Eq. 15)
         = -(1/tau)(f_i - f_i_eq)      (BGK, Campos Eq. 8)
Omega_R  = w_i * R                      (Campos Eq. 11)
```

### Diffusion / Conductivity

| Symbol | Meaning | Code name | Notes |
|--------|---------|-----------|-------|
| sigma | Conductivity tensor | `sigma` | Physical units: mS/cm |
| sigma_l | Longitudinal (fiber) conductivity | `sigma_l` | |
| sigma_t | Transverse (cross-fiber) conductivity | `sigma_t` | |
| D | Diffusion tensor = sigma / (chi * Cm) | `D` | Units: cm^2/ms |
| D_xx, D_yy, D_xy | Tensor components | `D_xx`, `D_yy`, `D_xy` | |
| a_l | Fiber direction unit vector | `fiber_direction` | |
| chi | Surface-to-volume ratio | `chi` | cm^{-1} |
| Cm | Membrane capacitance | `Cm` | uF/cm^2 |

### Ionic Model

| Symbol | Meaning | Code name | Notes |
|--------|---------|-----------|-------|
| I_ion | Total ionic current | `I_ion` | uA/uF (from IonicModel.compute_Iion) |
| I_stim | Stimulus current | `I_stim` | uA/uF (negative depolarizes) |
| eta | Ionic state variables vector | `ionic_states` | (n_dof, n_states) |

### Grid / Spatial

| Symbol | Meaning | Code name | Notes |
|--------|---------|-----------|-------|
| dx | Grid spacing | `dx` | cm |
| dt | Time step | `dt` | ms |
| Nx, Ny | Grid dimensions | `Nx`, `Ny` | Axis 0 = x, axis 1 = y |
| x | Spatial position | (implicit via grid index) | |

---

## The tau-D Mapping Function

Central to the LBM: converting a physical diffusion tensor D into relaxation parameters.

### Scalar (BGK, isotropic)
```python
def tau_from_D(D: float, dx: float, dt: float, cs2: float = 1/3) -> float:
    """tau = 0.5 + D * dt / (cs2 * dx^2)"""
    return 0.5 + D * dt / (cs2 * dx * dx)
```

### Tensor (MRT, anisotropic) — D2Q9
```python
def tau_tensor_from_D(D_xx, D_yy, D_xy, dx, dt, cs2=1/3):
    """
    Compute the 2x2 relaxation time tensor for flux moments.

    tau_ij = delta_ij/2 + D_ij * dt / (cs2 * dx^2)

    Returns the S^{-1} sub-block for rows/columns corresponding
    to the flux moments in moment space.
    """
    scale = dt / (cs2 * dx * dx)
    tau_xx = 0.5 + D_xx * scale
    tau_yy = 0.5 + D_yy * scale
    tau_xy = D_xy * scale           # No delta_ij term for off-diagonal
    return tau_xx, tau_yy, tau_xy
```

### From Conductivity Tensor to D
```python
def sigma_to_D(sigma_l, sigma_t, fiber_angle, chi, Cm):
    """
    Convert conductivity tensor to diffusion tensor.

    D = sigma / (chi * Cm)

    sigma = sigma_t * I + (sigma_l - sigma_t) * a * a^T
    where a = (cos(theta), sin(theta))

    Returns D_xx, D_yy, D_xy
    """
    cos_a = cos(fiber_angle)
    sin_a = sin(fiber_angle)

    sigma_xx = sigma_t + (sigma_l - sigma_t) * cos_a**2
    sigma_yy = sigma_t + (sigma_l - sigma_t) * sin_a**2
    sigma_xy = (sigma_l - sigma_t) * cos_a * sin_a

    scale = 1.0 / (chi * Cm)
    return sigma_xx * scale, sigma_yy * scale, sigma_xy * scale
```

### Full Pipeline
```
User provides:  sigma_l, sigma_t, fiber_angle, chi, Cm, dx, dt
         |
         v
   sigma_to_D() → D_xx, D_yy, D_xy
         |
         v
   tau_tensor_from_D() → tau_xx, tau_yy, tau_xy
         |
         v
   MRTCollision builds S from tau values
         |
         v
   collide() uses hand-coded M^{-1} S M per moment
```

---

## Moment Space Definitions

### D2Q5 Moment Space (5 moments)

| Row | Symbol | Physical meaning | Equilibrium | Relaxation |
|-----|--------|-----------------|-------------|------------|
| 0 | rho | Conserved: V = sum(f_i) | V | s_0 = 0 (no relax) |
| 1 | j_x | x-flux: f_+x - f_-x | 0 | s_x = 1/tau_xx |
| 2 | j_y | y-flux: f_+y - f_-y | 0 | s_y = 1/tau_yy |
| 3 | e | Energy-like | -4/3 * V (lattice-dependent) | s_e (free, stability) |
| 4 | eps | Energy-square | 4/3 * V (lattice-dependent) | s_eps (free, stability) |

**Limitation:** No p_xx or p_xy moments → cannot encode D_xy.

### D2Q9 Moment Space (9 moments, Lallemand & Luo 2000)

| Row | Symbol | Physical meaning | Equilibrium | Relaxation |
|-----|--------|-----------------|-------------|------------|
| 0 | rho | Conserved: V = sum(f_i) | V | s_0 = 0 |
| 1 | e | Energy | e_eq(V) | s_e (free) |
| 2 | eps | Energy-square | eps_eq(V) | s_eps (free) |
| 3 | j_x | x-momentum (flux) | 0 | s_jx (from D) |
| 4 | q_x | Energy-flux x | 0 | s_qx (free) |
| 5 | j_y | y-momentum (flux) | 0 | s_jy (from D) |
| 6 | q_y | Energy-flux y | 0 | s_qy (free) |
| 7 | p_xx | Stress tensor (xx - yy) | 0 | s_pxx (from D) |
| 8 | p_xy | Stress tensor (xy) | 0 | s_pxy (from D) |

**Key: rows 3, 5 encode isotropic diffusion; rows 7, 8 encode anisotropy including D_xy.**

Relaxation mapping for diffusion (not fluid):
- s_jx, s_jy → related to (D_xx + D_yy)/2 (trace)
- s_pxx → related to (D_xx - D_yy) (anisotropy)
- s_pxy → related to D_xy (off-diagonal / fiber rotation)
- s_0 = 0, s_e, s_eps, s_qx, s_qy = free (stability tuning, typically 1.0-1.5)

---

## Collision Step: Pseudocode

### BGK (isotropic, either lattice)
```
For each node (x, y):
    f_eq_i = w_i * V                              # Equilibrium
    Omega_NR_i = -omega * (f_i - f_eq_i)          # Non-reactive (Eq. 8)
    Omega_R_i  = w_i * R                           # Reactive (Eq. 11)
    f_star_i   = f_i + Omega_NR_i + Omega_R_i      # Post-collision
```

### MRT (anisotropic, D2Q9)
```
For each node (x, y):
    m = M @ f              # Transform to moment space
    m_eq = equilibrium_moments(V)
    delta_m = S * (m - m_eq)     # Relax in moment space
    m_star = m - delta_m
    m_star[0] += R * dt          # Add source to conserved moment
    f_star = M_inv @ m_star      # Transform back
```

Where the S * (m - m_eq) for flux/stress moments uses the tau values from the D tensor.

---

## Streaming Step: Direction Names

### D2Q5

Pull convention: `f_post[a](x) = f_pre[a](x - e_a)`.
`torch.roll(shifts=+s)` gives `output[i] = input[i-s]`, so shift = `+e_component`.

```
Index  Name    e_i        Roll operation
0      rest    (0, 0)     none
1      east    (+1, 0)    roll(dim=0, +1)    [pull from x-1]
2      west    (-1, 0)    roll(dim=0, -1)    [pull from x+1]
3      north   (0, +1)    roll(dim=1, +1)    [pull from y-1]
4      south   (0, -1)    roll(dim=1, -1)    [pull from y+1]
```

### D2Q9
```
Index  Name    e_i        Roll operation
0      rest    (0, 0)     none
1      east    (+1, 0)    roll(dim=0, +1)
2      west    (-1, 0)    roll(dim=0, -1)
3      north   (0, +1)    roll(dim=1, +1)
4      south   (0, -1)    roll(dim=1, -1)
5      NE      (+1, +1)   roll(dim=0, +1), roll(dim=1, +1)
6      NW      (-1, +1)   roll(dim=0, -1), roll(dim=1, +1)
7      SW      (-1, -1)   roll(dim=0, -1), roll(dim=1, -1)
8      SE      (+1, -1)   roll(dim=0, +1), roll(dim=1, -1)
```

---

## Ionic Model Integration

We copy the entire `ionic/` folder from Engine V5.4 into LBM_V1 as a dependency.
The ionic model interface we consume:

```python
# From IonicModel ABC (base.py):
model.V_rest                          # float, resting potential (mV)
model.n_states                        # int, number of ionic states
model.compute_Iion(V, ionic_states)   # (n_dof,) → I_ion in uA/uF
model.get_initial_state(n_cells)      # → (n_cells, n_states) tensor
model.gate_indices                    # list of int
model.concentration_indices           # list of int
model.compute_gate_steady_states(V, ionic_states)    # → (n_dof, n_gates)
model.compute_gate_time_constants(V, ionic_states)   # → (n_dof, n_gates)
model.compute_concentration_rates(V, ionic_states)   # → (n_dof, n_conc)
```

The Rush-Larsen logic from V5.4 is reimplemented locally in LBM_V1 as a standalone
function (not wrapped in the IonicSolver class which depends on SimulationState):

```python
def rush_larsen_step(model, V_flat, ionic_states, I_stim, dt):
    """
    One Rush-Larsen step. Modifies V_flat and ionic_states in-place.

    1. I_ion = model.compute_Iion(V_flat, ionic_states)
    2. gate_inf, gate_tau from current V (BEFORE voltage update)
    3. V += dt * -(I_ion + I_stim)
    4. gates: x = x_inf - (x_inf - x) * exp(-dt/tau)
    5. concentrations: x += dt * rate
    """
```

This keeps our LBM code decoupled from V5.4's solver hierarchy while reusing the
ionic model data (TTP06, ORd) directly.
