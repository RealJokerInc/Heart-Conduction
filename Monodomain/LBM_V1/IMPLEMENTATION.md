# LBM_V1 — Implementation Plan

## Framework

- **Language:** Python 3.10+
- **Tensor library:** PyTorch (matches V5.4 ionic models, MPS support on macOS,
  torch.compile for kernel fusion, eager-mode debugging)
- **Ionic models:** Copied from Engine V5.4 `cardiac_sim/ionic/` (TTP06, ORd, LUT)
- **Naming conventions:** See CONVENTIONS.md for symbol -> code name mapping

## Code Structure

```
LBM_V1/
├── README.md
├── IMPLEMENTATION.md          <- This file
├── PROGRESS.md                <- Living checkpoint
├── CONVENTIONS.md             <- Symbol/naming reference, tau-D mapping
├── research/                  <- Paper extracts, comparison docs
│   ├── PAPER_COMPARISON.md
│   ├── rapaka_2012.txt
│   └── campos_2016.txt
│
├── ionic/                     <- COPIED from Engine V5.4 (read-only reference)
│   ├── __init__.py            │  IonicModel ABC, CellType enum
│   ├── base.py                │  TTP06Model, ORdModel exports
│   ├── lut.py                 │  Lookup table acceleration
│   ├── ttp06/                 │  ten Tusscher-Panfilov 2006
│   └── ord/                   │  O'Hara-Rudy 2011
│
├── src/
│   ├── __init__.py
│   │
│   ├── lattice/               <- Lattice definitions (constants only)
│   │   ├── __init__.py        │  Exports D2Q5, D2Q9
│   │   ├── base.py            │  Lattice ABC (Q, e, w, cs2, opposite)
│   │   ├── d2q5.py            │  D2Q5 frozen dataclass singleton
│   │   └── d2q9.py            │  D2Q9 frozen dataclass singleton
│   │
│   ├── collision/             <- Collision operators
│   │   ├── __init__.py        │  Exports BGK, MRT_D2Q5, MRT_D2Q9
│   │   ├── base.py            │  CollisionOperator ABC
│   │   ├── bgk.py             │  BGKCollision (isotropic, either lattice)
│   │   └── mrt/               │  Multiple relaxation time
│   │       ├── __init__.py
│   │       ├── d2q5.py        │  MRT_D2Q5 (axis-aligned anisotropy only)
│   │       └── d2q9.py        │  MRT_D2Q9 (full tensor including D_xy)
│   │
│   ├── streaming/             <- Streaming (advection step)
│   │   ├── __init__.py        │  Exports stream_d2q5, stream_d2q9
│   │   ├── d2q5.py            │  stream_d2q5(f) -> f_streamed
│   │   └── d2q9.py            │  stream_d2q9(f) -> f_streamed
│   │
│   ├── boundary/              <- Boundary conditions
│   │   ├── __init__.py        │  Exports apply_neumann, apply_dirichlet, apply_absorbing
│   │   ├── neumann.py         │  Bounce-back (no-flux)
│   │   ├── dirichlet.py       │  Anti-bounce-back (fixed voltage)
│   │   └── absorbing.py       │  Equilibrium incoming (open boundary)
│   │
│   ├── solver/                <- Ionic time-stepping
│   │   ├── __init__.py
│   │   └── rush_larsen.py     │  Standalone RL step function
│   │
│   ├── diffusion.py           <- sigma_to_D(), tau_from_D(), tau_tensor_from_D()
│   ├── state.py               <- LBMState dataclass
│   ├── step.py                <- @torch.compile'd fused step functions (Layer 2)
│   └── simulation.py          <- Master coordinator (Layer 1, user-facing)
│
├── tests/
│   ├── __init__.py
│   ├── test_lattice.py        # Lattice constants, weight sums, isotropy
│   ├── test_collision.py      # BGK + MRT collision correctness
│   ├── test_streaming.py      # Streaming + state + voltage recovery
│   ├── test_boundary.py       # Neumann, Dirichlet, absorbing BC tests
│   ├── test_diffusion.py      # Pure diffusion convergence (both lattices)
│   ├── test_ionic.py          # RL step, single-cell AP
│   └── test_simulation.py     # Full simulation integration tests
│
└── experiments/
    ├── exp01_planar_wave.py
    ├── exp02_boundary_cv.py
    └── exp03_d2q5_vs_d2q9.py
```

### Two-Layer Architecture (torch.compile Optimization)

The code follows a two-layer pattern inspired by V5.4:

- **Layer 1 (OOP Configuration):** Classes that hold parameters, validate inputs, and
  provide a user-friendly API. These live in the module files (e.g., `collision/mrt/d2q9.py`).
  They are NOT compiled.

- **Layer 2 (Pure Functions):** Stateless functions that do the actual math on tensors.
  These are what `torch.compile` can fuse into efficient GPU kernels. They live alongside
  their class wrappers but are importable independently.

**Example pattern** (`collision/mrt/d2q9.py`):

```python
# --- Layer 2: Pure function (compilable) ---
def mrt_collide_d2q9(f, V, R, dt, s_e, s_eps, s_j, s_q, s_pxx, s_pxy, w):
    """Hand-coded MRT collision for D2Q9. No classes, no state, no side effects."""
    # Transform to moment space (hand-coded, no matmul)
    m0 = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8]
    m1 = -4*f[0] - f[1] - f[2] - f[3] - f[4] + 2*f[5] + 2*f[6] + 2*f[7] + 2*f[8]
    # ... (all 9 moments)
    # Relax each moment independently
    # Transform back (hand-coded M_inv)
    # Add source: f_star += w * R * dt
    return f_star

# --- Layer 1: Class wrapper (user-facing) ---
class MRT_D2Q9:
    def __init__(self, D_xx, D_yy, D_xy, dx, dt, device, dtype):
        # Compute relaxation rates from D tensor
        self.s_e = ...
        self.s_pxx = ...
        # Pre-compute w tensor
        self.w = D2Q9.get_w_tensor(device, dtype)

    def collide(self, f, V, R, dt):
        return mrt_collide_d2q9(f, V, R, dt,
            self.s_e, self.s_eps, self.s_j, self.s_q,
            self.s_pxx, self.s_pxy, self.w)
```

**The `step.py` bridge:**

`step.py` imports all Layer 2 pure functions and composes them into fused step functions:

```python
from src.collision.mrt.d2q9 import mrt_collide_d2q9
from src.streaming.d2q9 import stream_d2q9
from src.boundary.neumann import apply_neumann_d2q9
from src.boundary.dirichlet import apply_dirichlet_d2q9

@torch.compile
def lbm_step_d2q9_mrt(f, V, R, dt, s_e, s_eps, s_j, s_q, s_pxx, s_pxy,
                        w, bounce_masks_neu, bounce_masks_dir, V_dir):
    f = mrt_collide_d2q9(f, V, R, dt, s_e, s_eps, s_j, s_q, s_pxx, s_pxy, w)
    f_star = f.clone()   # Save pre-streaming state for BC (Campos Eq. 17)
    f = stream_d2q9(f)
    f = apply_neumann_d2q9(f, f_star, bounce_masks_neu)
    f = apply_dirichlet_d2q9(f, f_star, bounce_masks_dir, V_dir, w)
    V = f.sum(dim=0)
    return f, V
```

This fuses collide+stream+BC+recover into a single GPU kernel. The `simulation.py`
coordinator selects the right compiled step at init time based on lattice/collision choice.

### The diffusion.py Helper

Provides the sigma -> D -> tau pipeline (CONVENTIONS.md):

```python
def sigma_to_D(sigma_l, sigma_t, fiber_angle, chi, Cm) -> (D_xx, D_yy, D_xy)
def tau_from_D(D, dx, dt, cs2=1/3) -> float
def tau_tensor_from_D(D_xx, D_yy, D_xy, dx, dt, cs2=1/3) -> (tau_xx, tau_yy, tau_xy)
def check_stability_tensor(D_xx, D_yy, D_xy, dx, dt, cs2=1/3) -> (bool, tau_min)
```

### The rush_larsen.py Module

Standalone function, no class hierarchy dependency:

```python
def rush_larsen_step(model: IonicModel, V_flat, ionic_states, I_stim, dt):
    """
    Modifies V_flat and ionic_states in-place.
    Uses model.compute_Iion, gate_indices, concentration_indices, etc.
    """
```

This reuses V5.4's IonicModel ABC (via the copied ionic/ folder) without
depending on V5.4's SimulationState or IonicSolver class hierarchy.

## Phase Overview

| Phase | Goal | Files | Tests |
|-------|------|-------|-------|
| 0 | Copy ionic models from V5.4 | ionic/ (copy) | 0 |
| 1 | Lattice definitions (D2Q5, D2Q9) | lattice/base.py, lattice/d2q5.py, lattice/d2q9.py, diffusion.py | 4 |
| 2 | Collision operators (BGK, MRT) | collision/base.py, collision/bgk.py, collision/mrt/d2q5.py, collision/mrt/d2q9.py | 6 |
| 3 | State + streaming | state.py, streaming/d2q5.py, streaming/d2q9.py | 5 |
| 4 | Boundary conditions | boundary/masks.py, boundary/neumann.py, boundary/dirichlet.py, boundary/absorbing.py | 4 |
| 5 | Pure diffusion validation | (integration tests, no new files) | 5 |
| 6 | Ionic coupling | solver/rush_larsen.py | 3 |
| 7 | Simulation orchestrator | step.py, simulation.py | 4 |
| 8 | Boundary speedup experiment | experiments/ | 3 |

**Total: ~34 validation tests across 9 phases (Phase 0 has no tests)**

---

## Phase 0: Copy Ionic Models

**Goal:** Copy V5.4 ionic models into LBM_V1 for standalone use.

**Action:** Copy `Engine_V5.4/cardiac_sim/ionic/` -> `LBM_V1/ionic/`
**Fix imports:** Convert V5.4 relative imports to work standalone.

---

## Phase 1: Lattice Definitions

**Goal:** Define D2Q5 and D2Q9 lattice constants as frozen dataclasses.

### D2Q5 (reference, for comparison)
```
Velocities: (0,0), (1,0), (-1,0), (0,1), (0,-1)
Weights:    1/3,   1/6,   1/6,    1/6,   1/6
c_s^2 = 1/3
tau(D) = 0.5 + 3*D*dt/dx^2
```

### D2Q9 (primary lattice)
```
Velocities: (0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (-1,-1), (1,-1)
Weights:    4/9,   1/9,   1/9,    1/9,   1/9,    1/36,  1/36,   1/36,    1/36
c_s^2 = 1/3
tau(D) = 0.5 + 3*D*dt/dx^2  (same formula, different weights)
```

### Files
- `src/lattice/base.py`: Lattice ABC
  - `Q`, `D`, `cs2`, `e` (velocity tuples), `w` (weight tuples), `opposite` (index map)
  - `get_e_tensor(device, dtype)`, `get_w_tensor(device, dtype)`
- `src/lattice/d2q5.py`: D2Q5 frozen dataclass singleton
- `src/lattice/d2q9.py`: D2Q9 frozen dataclass singleton
- `src/diffusion.py`: `sigma_to_D()`, `tau_from_D()`, `tau_tensor_from_D()`, `check_stability()`

### Validation
| Test | Criterion |
|------|-----------|
| 1-V1 | D2Q5 weights sum to 1.0 |
| 1-V2 | D2Q9 weights sum to 1.0 |
| 1-V3 | D2Q9 isotropy: sum(w_i*e_ix*e_iy) = 0, sum(w_i*e_ix^2) = c_s^2 |
| 1-V4 | tau<->D roundtrip for both lattices |

---

## Phase 2: Collision Operators

**Goal:** Implement BGK and MRT collision. MRT must support full 2x2 D tensor including D_xy.

### BGK Collision
```
f*_i = f_i - (1/tau)(f_i - w_i*V) + dt*w_i*R
```
Single tau -> isotropic only. Used as reference.

### MRT Collision -- D2Q9

**Transformation matrix M (9x9):**

Following Lallemand & Luo (2000) / d'Humieres (2002):
```
Row 0: rho = sum(f_i)                    (conserved: voltage)
Row 1: e   = -4f_0 + sum_card(-f_i) + sum_diag(2f_i)  (energy)
Row 2: eps = energy square                (higher moment)
Row 3: j_x = sum(f_i*e_ix)               (x-momentum/flux)
Row 4: q_x = related to j_x              (energy flux x)
Row 5: j_y = sum(f_i*e_iy)               (y-momentum/flux)
Row 6: q_y = related to j_y              (energy flux y)
Row 7: p_xx = sum(f_i*(e_ix^2 - e_iy^2)) (stress tensor xx-yy)
Row 8: p_xy = sum(f_i*e_ix*e_iy)         (stress tensor xy)
```

**Standard D2Q9 M matrix (Lallemand & Luo):**
```
M = [ 1,  1,  1,  1,  1,  1,  1,  1,  1]   <- rho (conserved)
    [-4, -1, -1, -1, -1,  2,  2,  2,  2]   <- e (energy)
    [ 4, -2, -2, -2, -2,  1,  1,  1,  1]   <- eps (energy^2)
    [ 0,  1, -1,  0,  0,  1, -1, -1,  1]   <- j_x (x-flux)
    [ 0, -2,  2,  0,  0,  1, -1, -1,  1]   <- q_x (energy flux x)
    [ 0,  0,  0,  1, -1,  1,  1, -1, -1]   <- j_y (y-flux)
    [ 0,  0,  0, -2,  2,  1,  1, -1, -1]   <- q_y (energy flux y)
    [ 0,  1,  1, -1, -1,  0,  0,  0,  0]   <- p_xx (normal stress diff)
    [ 0,  0,  0,  0,  0,  1, -1,  1, -1]   <- p_xy (shear stress)
```

**Relaxation rates in moment space:**
```
S = diag(s_0, s_e, s_eps, s_j, s_q, s_j, s_q, s_pxx, s_pxy)
```

For diffusion (not fluid flow), the mapping is:
- s_0 = 0 (conserved, no relaxation)
- s_j = 1/tau where tau = 0.5 + 3*D*dt/dx^2 (flux moments -> diffusion)
- s_pxx, s_pxy: encode the diffusion tensor anisotropy
- s_e, s_eps, s_q: free parameters for stability (typically 1.0-1.5)

**Anisotropic diffusion encoding (Yoshida & Nagaoka approach for D2Q9):**

For a 2x2 diffusion tensor D = [[D_xx, D_xy], [D_xy, D_yy]], the relaxation
rates for the stress moments (p_xx, p_xy) encode the tensor:

```
s_pxx  -> encodes (D_xx - D_yy)  via p_xx moment
s_pxy  -> encodes D_xy           via p_xy moment
s_j    -> encodes (D_xx + D_yy)/2 via flux moments
```

Specifically:
```
tau_+ = 0.5 + 3*(D_xx + D_yy)/2 * dt/dx^2    (isotropic part)
tau_- = related to (D_xx - D_yy)               (anisotropic part)
tau_xy = related to D_xy                        (off-diagonal part)
```

**IMPORTANT -- Why D2Q9 and not D2Q5 for this:**

D2Q5 has only 5 moments: rho, j_x, j_y, and 2 higher moments.
The flux moments j_x, j_y can encode D_xx and D_yy independently (via separate tau_x, tau_y).
But D2Q5 has **no stress-tensor moments** (p_xx, p_xy) because it has no diagonal velocities.
Therefore D2Q5 **cannot encode D_xy** -- it can only do axis-aligned anisotropy.

D2Q9 adds moments 7 (p_xx) and 8 (p_xy) which directly encode the off-diagonal diffusion.
This is essential for:
1. Fiber rotation at arbitrary angles
2. Proper gradient resolution at boundaries (diagonal directions "see" the boundary)
3. The boundary speedup effect (asymmetric loading requires diagonal interactions)

### Files
- `src/collision/base.py`: `CollisionOperator` ABC with `collide(f, V, R, dt) -> f_star`
- `src/collision/bgk.py`:
  - `bgk_collide(f, V, R, dt, omega, w) -> f_star` (Layer 2, pure function)
  - `BGKCollision(tau, lattice, device, dtype)` (Layer 1, class wrapper)
- `src/collision/mrt/d2q5.py`:
  - `mrt_collide_d2q5(f, V, R, dt, s_x, s_y, s_e, s_eps, w) -> f_star` (Layer 2)
  - `MRT_D2Q5(D_xx, D_yy, dx, dt, device, dtype)` (Layer 1)
- `src/collision/mrt/d2q9.py`:
  - `mrt_collide_d2q9(f, V, R, dt, s_e, s_eps, s_j, s_q, s_pxx, s_pxy, w) -> f_star` (Layer 2)
  - `MRT_D2Q9(D_xx, D_yy, D_xy, dx, dt, device, dtype)` (Layer 1)

### Validation
| Test | Criterion |
|------|-----------|
| 2-V1 | BGK: equilibrium f unchanged by collision |
| 2-V2 | MRT D2Q9: equilibrium f unchanged by collision |
| 2-V3 | MRT D2Q9: isotropic D gives same result as BGK |
| 2-V4 | MRT D2Q9: anisotropic D_xx != D_yy gives directional diffusion |
| 2-V5 | MRT D2Q9: D_xy != 0 gives rotated diffusion |
| 2-V6 | Source term conservation: sum(f*_i) = V + dt*R |

---

## Phase 3: State + Streaming

**Goal:** State container and streaming step.

### State
```python
@dataclass
class LBMState:
    f: Tensor          # (Q, Nx, Ny) distribution functions
    V: Tensor          # (Nx, Ny) macroscopic voltage
    ionic_states: Tensor  # (Nx*Ny, n_states)
    mask: Tensor       # (Nx, Ny) bool domain mask
    t: float
    lattice: Lattice   # D2Q5 or D2Q9
```

### Streaming
D2Q9 streaming requires 8 shifts (rest stays). Pull convention:
`f_post[a](x) = f_pre[a](x - e_a)`. Roll shift = `+e_component`.
```
Direction  Velocity   Roll
0 (rest)   (0,0)     none
1 (+x)     (1,0)     roll(dim=0, shifts=+1)
2 (-x)     (-1,0)    roll(dim=0, shifts=-1)
3 (+y)     (0,1)     roll(dim=1, shifts=+1)
4 (-y)     (0,-1)    roll(dim=1, shifts=-1)
5 (+x+y)   (1,1)     roll(dim=0, shifts=+1) then roll(dim=1, shifts=+1)
6 (-x+y)   (-1,1)    roll(dim=0, shifts=-1) then roll(dim=1, shifts=+1)
7 (-x-y)   (-1,-1)   roll(dim=0, shifts=-1) then roll(dim=1, shifts=-1)
8 (+x-y)   (1,-1)    roll(dim=0, shifts=+1) then roll(dim=1, shifts=-1)
```

Grid convention: (Nx, Ny) with axis 0 = x, axis 1 = y (matching V5.4).

### Files
- `src/state.py`: LBMState dataclass, `create_lbm_state()`, `recover_voltage()`
- `src/streaming/d2q5.py`:
  - `stream_d2q5(f) -> f_streamed` (Layer 2, pure function)
- `src/streaming/d2q9.py`:
  - `stream_d2q9(f) -> f_streamed` (Layer 2, pure function)

### Validation
| Test | Criterion |
|------|-----------|
| 3-V1 | Stream: uniform f unchanged (periodic BC via roll) |
| 3-V2 | Stream: delta pulse moves correctly for each direction |
| 3-V3 | Bounce-back: total sum(f_i) conserved over 1000 steps |
| 3-V4 | recover_voltage: V = sum(f_i) to machine precision |
| 3-V5 | D2Q9 diagonal streaming: pulse at (5,5) reaches (6,6) after stream |

---

## Phase 4: Boundary Conditions

**Goal:** Implement Neumann (no-flux), Dirichlet (fixed value), and absorbing BCs
using the f_star clone approach (Campos Eq. 17).

### 4.1 Direction and Wall Reference Tables

All BC logic indexes into these tables. Values must match CONVENTIONS.md exactly.

#### D2Q5 Complete Reference

| Index | Name  | e_i     | opp | w_i  | Roll operation       |
|-------|-------|---------|-----|------|----------------------|
| 0     | rest  | (0, 0)  | 0   | 1/3  | none                           |
| 1     | east  | (+1, 0) | 2   | 1/6  | roll(dim=0, +1) [pull from x-1] |
| 2     | west  | (-1, 0) | 1   | 1/6  | roll(dim=0, -1) [pull from x+1] |
| 3     | north | (0, +1) | 4   | 1/6  | roll(dim=1, +1) [pull from y-1] |
| 4     | south | (0, -1) | 3   | 1/6  | roll(dim=1, -1) [pull from y+1] |

Opposite tuple: `(0, 2, 1, 4, 3)`
Weights sum: 1/3 + 4*(1/6) = 1.0

#### D2Q9 Complete Reference

| Index | Name | e_i       | opp | w_i   | Roll operation                        |
|-------|------|-----------|-----|-------|---------------------------------------|
| 0     | rest | (0, 0)   | 0   | 4/9   | none                                  |
| 1     | E    | (+1, 0)  | 2   | 1/9   | roll(dim=0, +1)                       |
| 2     | W    | (-1, 0)  | 1   | 1/9   | roll(dim=0, -1)                       |
| 3     | N    | (0, +1)  | 4   | 1/9   | roll(dim=1, +1)                       |
| 4     | S    | (0, -1)  | 3   | 1/9   | roll(dim=1, -1)                       |
| 5     | NE   | (+1, +1) | 7   | 1/36  | roll(dim=0, +1), roll(dim=1, +1)      |
| 6     | NW   | (-1, +1) | 8   | 1/36  | roll(dim=0, -1), roll(dim=1, +1)      |
| 7     | SW   | (-1, -1) | 5   | 1/36  | roll(dim=0, -1), roll(dim=1, -1)      |
| 8     | SE   | (+1, -1) | 6   | 1/36  | roll(dim=0, +1), roll(dim=1, -1)      |

Opposite tuple: `(0, 2, 1, 4, 3, 7, 8, 5, 6)`
Weights sum: 4/9 + 4*(1/9) + 4*(1/36) = 1.0
Verify: `opp[opp[i]] == i` for all i.

**Note:** This is our project ordering (CONVENTIONS.md), NOT the lettuce library ordering
(which swaps N/W indices). All code must use this ordering.

#### Wall-to-Outgoing-Direction Mapping

"Outgoing" = the direction that would stream a particle OUT of the domain at this wall.
These are the directions whose distributions need bounce-back correction.

**D2Q5** (1 outgoing direction per wall):

| Wall  | Outgoing dirs | Explanation            |
|-------|---------------|------------------------|
| South (y=0)    | 4 (S)         | particles heading -y leave |
| North (y=Ny-1) | 3 (N)         | particles heading +y leave |
| West  (x=0)    | 2 (W)         | particles heading -x leave |
| East  (x=Nx-1) | 1 (E)         | particles heading +x leave |

**D2Q9** (3 outgoing directions per wall, including diagonals):

| Wall  | Outgoing dirs      | Cardinal + diagonals       |
|-------|--------------------|----------------------------|
| South | 4 (S), 7 (SW), 8 (SE) | all have e_y = -1          |
| North | 3 (N), 5 (NE), 6 (NW) | all have e_y = +1          |
| West  | 2 (W), 6 (NW), 7 (SW) | all have e_x = -1          |
| East  | 1 (E), 5 (NE), 8 (SE) | all have e_x = +1          |

**Corners (D2Q9):** A corner node (e.g., SW corner at (0,0)) is on two walls simultaneously.
The mask-based approach handles this automatically: the union of outgoing directions for
both walls gives the correct set. For SW corner: South{4,7,8} + West{2,6,7} = {2,4,6,7,8}.

### 4.2 Neumann BC (Bounce-Back) — f_star Approach

**Campos et al. (2016) Eq. 17:**
```
f_a(x, t+dt) = f_b*(x, t)     where b = opp[a]
```
At a boundary node x where direction a is outgoing (would leave domain),
the post-collision distribution f_star[a] is reflected back into direction opp[a].

#### Why f_star (Pre-Streaming) Is Required

With roll-based streaming on a periodic domain:
1. Collision produces f_star[a] at boundary node x_b
2. Streaming via `roll` moves f_star[a](x_b) to the OPPOSITE side of the domain (periodic wrap)
3. After streaming, f[a](x_b) contains whatever rolled IN — NOT the original outgoing value

The original f_star[a](x_b) is **gone** from position x_b after streaming. We must save it
before streaming to apply bounce-back correctly. Hence: `f_star = f.clone()` before stream.

#### Pseudocode: `apply_neumann_d2q5(f, f_star, bounce_masks)`

```python
# bounce_masks: dict of bool tensors, one per direction
# bounce_masks[a] is True at nodes where direction a is outgoing
#
# D2Q5 opposite: opp = (0, 2, 1, 4, 3)

def apply_neumann_d2q5(f, f_star, bounce_masks):
    # Direction 1 (E) bounces to 2 (W)
    f[2] = torch.where(bounce_masks[1], f_star[1], f[2])
    # Direction 2 (W) bounces to 1 (E)
    f[1] = torch.where(bounce_masks[2], f_star[2], f[1])
    # Direction 3 (N) bounces to 4 (S)
    f[4] = torch.where(bounce_masks[3], f_star[3], f[4])
    # Direction 4 (S) bounces to 3 (N)
    f[3] = torch.where(bounce_masks[4], f_star[4], f[3])
    return f
```

For D2Q9, the same pattern extends to all 8 non-rest directions (indices 1-8).

**Implementation note:** Use `torch.where(mask, value, original)` instead of boolean
indexing (`f[i][mask] = ...`) for torch.compile kernel fusion compatibility.

#### D2Q9 Bounce-Back Artifact

For D2Q9 with Neumann BC, diagonal distributions (SW, SE) at the south wall have
nonzero non-equilibrium components even for planar waves (because they carry x-velocity).
Bounce-back of these diagonals introduces an O(dx^2) artifact at boundaries — a tiny
numerical slowdown (~0.1-0.5%) that vanishes with mesh refinement.

This is NOT the Kleber boundary speedup. It is a lattice discretization artifact.
See BOUNDARY_SPEEDUP_ANALYSIS.md S8 for the full derivation.

D2Q5 has zero artifact (only cardinal directions, no x-velocity component at y-walls).

### 4.3 Dirichlet BC (Anti-Bounce-Back)

For a fixed voltage V_D at the boundary wall midpoint:
```
f[opp[a]](x) = -f_star[a](x) + 2 * w[a] * V_D
```

**Derivation:** The wall sits at the midpoint between node x and its (virtual) neighbor
outside the domain. Anti-bounce-back enforces the equilibrium average:
`(f_star[a] + f[opp[a]]) / 2 = w[a] * V_D`, which gives the formula above.

#### Pseudocode: `apply_dirichlet_d2q5(f, f_star, bounce_masks, V_bc, w)`

```python
# V_bc: scalar or (Nx, Ny) tensor of boundary voltage values
# w: weight tuple (1/3, 1/6, 1/6, 1/6, 1/6)

def apply_dirichlet_d2q5(f, f_star, bounce_masks, V_bc, w):
    # Direction 1 (E, w=1/6) bounces to 2 (W)
    f[2] = torch.where(bounce_masks[1], -f_star[1] + 2 * w[1] * V_bc, f[2])
    # Direction 2 (W, w=1/6) bounces to 1 (E)
    f[1] = torch.where(bounce_masks[2], -f_star[2] + 2 * w[2] * V_bc, f[1])
    # Direction 3 (N, w=1/6) bounces to 4 (S)
    f[4] = torch.where(bounce_masks[3], -f_star[3] + 2 * w[3] * V_bc, f[4])
    # Direction 4 (S, w=1/6) bounces to 3 (N)
    f[3] = torch.where(bounce_masks[4], -f_star[4] + 2 * w[4] * V_bc, f[3])
    return f
```

For D2Q9, extend to all 8 directions. Note that diagonal weights are 1/36 (not 1/9).

#### Critical Caveat: Dirichlet BC on V != Kleber Boundary Speedup

**WARNING:** Applying Dirichlet V = V_rest at tissue boundaries does NOT reproduce the
Kleber boundary speedup effect. Analysis (BOUNDARY_SPEEDUP_ANALYSIS.md S4) shows:

1. Dirichlet on V acts as a **current sink** — the anti-bounce-back formula drains energy
   from boundary-adjacent cells
2. This INCREASES the electrical loading on boundary cells, SLOWING conduction
3. The Kleber effect arises from reduced intracellular loading via the extracellular bath,
   which requires bidomain physics (separate intra/extracellular potentials)

**When to use Dirichlet BC:**
- Manufactured solution tests (prescribe known V at boundaries)
- Stimulus injection (fix V at electrode locations)
- Coupling to external circuits or bath potentials

**When NOT to use:**
- Modeling tissue boundaries (use Neumann instead — correct for monodomain)
- Attempting to reproduce boundary speedup (physically incorrect in monodomain)

### 4.4 Absorbing BC (Equilibrium Incoming)

```
f[opp[a]](x) = w[opp[a]] * V(x, t)
```

Sets incoming distributions to their equilibrium value based on the current local voltage.
This minimizes reflections — the incoming distribution carries no non-equilibrium information.

**Key details:**
- V(x, t) is the PREVIOUS timestep's voltage (before this step's collision/streaming)
- No f_star needed — this BC uses macroscopic V, not distribution-level data
- **Non-conservative:** total sum(f) is not preserved. Absorbing BCs act as open boundaries
  where energy can leave the domain. This is intentional.

#### Pseudocode: `apply_absorbing_d2q5(f, bounce_masks, V, w)`

```python
def apply_absorbing_d2q5(f, bounce_masks, V, w):
    # At east wall, incoming direction is W (2), set to equilibrium
    f[2] = torch.where(bounce_masks[1], w[2] * V, f[2])
    # At west wall, incoming direction is E (1)
    f[1] = torch.where(bounce_masks[2], w[1] * V, f[1])
    # At north wall, incoming direction is S (4)
    f[4] = torch.where(bounce_masks[3], w[4] * V, f[4])
    # At south wall, incoming direction is N (3)
    f[3] = torch.where(bounce_masks[4], w[3] * V, f[3])
    return f
```

### 4.5 Mask Pre-computation

Boundary masks are computed once at simulation init and reused every timestep.
This avoids redundant shifted-mask computation inside the @torch.compile'd step function.

#### `precompute_bounce_masks(domain_mask, lattice)`

```python
def precompute_bounce_masks(domain_mask, lattice):
    """
    Args:
        domain_mask: (Nx, Ny) bool tensor, True = inside domain
        lattice: object with .e (velocity vectors) and .Q (number of directions)

    Returns:
        bounce_masks: dict[int, Tensor], one (Nx, Ny) bool mask per direction
            bounce_masks[a] is True where direction a is outgoing (neighbor is outside)
    """
    bounce_masks = {}
    for a in range(1, lattice.Q):  # skip rest (index 0)
        ex, ey = lattice.e[a]
        # Shift domain mask in direction a: where does this particle land?
        neighbor_mask = torch.roll(
            torch.roll(domain_mask, shifts=-ex, dims=0),
            shifts=-ey, dims=1
        )
        # Outgoing = inside domain AND neighbor is outside
        bounce_masks[a] = domain_mask & ~neighbor_mask
    return bounce_masks
```

**Storage:** These masks are stored in the simulation coordinator (simulation.py) and
passed as arguments to the compiled step function. They are bool tensors — 1 bit per
node per direction, negligible memory.

**Rectangular domains:** For a simple rectangular Nx x Ny domain, the masks reduce to
single-row/column slices (e.g., bounce_masks[4] is True only at y=0). But the general
formulation handles arbitrary shapes (circular, irregular tissue masks from Builder).

### 4.6 Integration with step.py

The f_star clone happens inside the compiled step function:

```python
@torch.compile
def lbm_step_d2q5_bgk(f, V, R, dt, tau, w, bounce_masks_neu):
    f = bgk_collide_d2q5(f, V, R, dt, tau, w)
    f_star = f.clone()          # <-- save pre-streaming state
    f = stream_d2q5(f)
    f = apply_neumann_d2q5(f, f_star, bounce_masks_neu)
    V = f.sum(dim=0)
    return f, V
```

The `f.clone()` is a single contiguous memory copy. For D2Q5 on a 200x50 grid, this is
200*50*5*4 = 200 KB (float32) — negligible compared to the ionic model computation.

For mixed BCs (Neumann on some walls, Dirichlet on others), apply both in sequence:
```python
f = apply_neumann_d2q5(f, f_star, bounce_masks_neu)
f = apply_dirichlet_d2q5(f, f_star, bounce_masks_dir, V_dir, w)
```
The masks are disjoint (a node is either Neumann or Dirichlet, not both), so order
does not matter.

### 4.7 Literature References

| Reference | Relevance |
|-----------|-----------|
| Campos et al. (2016) Eq. 17 | Neumann bounce-back formula for LBM reaction-diffusion |
| Dawson, Chen & Doolen (1993) | Original LBM bounce-back for reaction-diffusion equations |
| Inamuro et al. (1995) | Anti-bounce-back (Dirichlet) BC formulation |
| Zou & He (1997) | Pressure/velocity BC — context for LBM BC taxonomy |
| BOUNDARY_SPEEDUP_ANALYSIS.md S4 | Proof: Dirichlet on V is current sink, not Kleber effect |
| BOUNDARY_SPEEDUP_ANALYSIS.md S8 | D2Q9 bounce-back O(dx^2) artifact derivation |

### Files

- `src/boundary/masks.py`:
  - `precompute_bounce_masks(domain_mask, lattice) -> dict[int, Tensor]` (Layer 1, called once at init)
- `src/boundary/neumann.py`:
  - `apply_neumann_d2q5(f, f_star, bounce_masks) -> f` (Layer 2)
  - `apply_neumann_d2q9(f, f_star, bounce_masks) -> f` (Layer 2)
- `src/boundary/dirichlet.py`:
  - `apply_dirichlet_d2q5(f, f_star, bounce_masks, V_bc, w) -> f` (Layer 2)
  - `apply_dirichlet_d2q9(f, f_star, bounce_masks, V_bc, w) -> f` (Layer 2)
- `src/boundary/absorbing.py`:
  - `apply_absorbing_d2q5(f, bounce_masks, V, w) -> f` (Layer 2)
  - `apply_absorbing_d2q9(f, bounce_masks, V, w) -> f` (Layer 2)

### Validation

| Test | Criterion |
|------|-----------|
| 4-V1 | Neumann conservation: Gaussian diffusion on both D2Q5 and D2Q9, rectangular + circular mask. Total V drift < 1e-10 relative over 1000 steps. |
| 4-V2 | Dirichlet steady state: 1D domain with V=0 left, V=1 right. Converge to linear profile. L_inf error < 1e-4. |
| 4-V3 | Absorbing: Gaussian pulse near boundary exits without reflection. Reflected energy < 1% (D2Q5), < 2% (D2Q9, diagonal artifact). |
| 4-V4 | Mixed BC: Dirichlet top/bottom (V=0, V=1) + Neumann left/right. Steady state = horizontal linear gradient. |

---

## Phase 5: Pure Diffusion Validation

**Goal:** Verify that LBM reproduces the diffusion equation correctly for both lattices.

### Tests
| Test | Criterion |
|------|-----------|
| 5-V1 | Isotropic BGK D2Q5: Gaussian variance grows as 2Dt (error < 1%) |
| 5-V2 | Isotropic BGK D2Q9: same test (error < 1%) |
| 5-V3 | Isotropic MRT D2Q9: matches BGK result |
| 5-V4 | Anisotropic MRT D2Q9 (D_xx != D_yy): elliptical spreading, correct ratio |
| 5-V5 | Anisotropic MRT D2Q9 (D_xy != 0, 45 deg fiber): rotated ellipse at correct angle |

---

## Phase 6: Ionic Model Coupling

**Goal:** Couple ionic models with LBM via Rush-Larsen integration.

### Rush-Larsen Standalone Function

```python
def rush_larsen_step(model: IonicModel, V_flat, ionic_states, I_stim, dt):
    """
    One Rush-Larsen step. Modifies V_flat and ionic_states in-place.

    1. I_ion = model.compute_Iion(V_flat, ionic_states)
    2. gate_inf, gate_tau from current V (BEFORE voltage update)
    3. V += dt * -(I_ion + I_stim)
    4. gates: x = x_inf - (x_inf - x) * exp(-dt/tau)
    5. concentrations: x += dt * rate
    """
```

### Source Term Integration
Following Rapaka (source in collision):
```
R = -(I_ion + I_stim) / (chi * Cm)
f*_i = f_i - A_ij(f_j - w_j*V) + dt*w_i*R
```

Note: In the LBM framework, the ionic source R is added during collision (source-in-collision).
The Rush-Larsen step handles only the ionic state update (gates + concentrations).
V is recovered from the distributions after streaming, NOT updated by Rush-Larsen.

### Files
- `src/solver/rush_larsen.py`:
  - `rush_larsen_step(model, V_flat, ionic_states, I_stim, dt)` (Layer 2)

### Validation
| Test | Criterion |
|------|-----------|
| 6-V1 | Single-cell AP with TTP06: correct shape, APD matches literature |
| 6-V2 | Rush-Larsen vs Forward Euler: RL allows 10x larger dt |
| 6-V3 | Source conservation: V increase matches integral(R*dt) |

---

## Phase 7: Simulation Orchestrator

**Goal:** Complete simulation loop with stimulus, output, and CV measurement.

### Algorithm (per timestep)
```
1. Compute ionic source: R = -(I_ion + I_stim) / (chi * Cm)
2. Collision with source: f* = collide(f, V, R, dt)
3. Save pre-streaming state: f_star = f.clone()
4. Stream: f_i(x+e_i) = f*_i(x)
5. Apply boundary conditions (using f_star for bounce-back, Campos Eq. 17)
6. Recover voltage: V = sum(f_i)
7. Update ionic states: rush_larsen_step(model, V, ionic_states, I_stim, dt)
8. Advance time: t += dt
```

### Files
- `src/step.py`:
  - `lbm_step_d2q5_bgk(...)` (Layer 2, @torch.compile'd)
  - `lbm_step_d2q9_bgk(...)` (Layer 2, @torch.compile'd)
  - `lbm_step_d2q9_mrt(...)` (Layer 2, @torch.compile'd)
  - Each fuses: collide -> clone(f_star) -> stream -> BC(f_star) -> recover_V into one compiled kernel
- `src/simulation.py`:
  - `LBMSimulation(Nx, Ny, dx, dt, lattice, collision, bc_config, ionic_model, chi, Cm)`
  - `add_stimulus(mask, start, duration, amplitude)`
  - `step()` -- delegates to the appropriate compiled step function
  - `run(t_end, save_every)`, `run_to_array()`
  - `measure_cv(activation_times, x1, x2)` utility

### Validation
| Test | Criterion |
|------|-----------|
| 7-V1 | Planar wave propagation: CV matches analytical estimate |
| 7-V2 | D2Q5 vs D2Q9: same CV for isotropic case |
| 7-V3 | Anisotropic propagation: CV ratio matches sqrt(D_l/D_t) |
| 7-V4 | Stimulus protocol: correct activation time |

---

## Phase 8: Boundary Speedup Experiment

**Goal:** The main experiment. Test boundary CV behavior for different BC types and lattices.

**Critical caveat:** BOUNDARY_SPEEDUP_ANALYSIS.md (S4) proved that Dirichlet BC on V
is NOT the Kleber boundary speedup — it acts as a current sink and SLOWS conduction.
The monodomain equation with uniform D admits no boundary speedup mechanism.
Any CV variation near boundaries is a numerical artifact (see S8 for D2Q9 O(dx^2) analysis).

### Experimental Setup
```
Domain: 200 x 50 nodes (long strip)
dx = 0.025 cm, dt = 0.01 ms
D = 0.001 cm^2/ms (isotropic for simplicity)

Stimulus: left edge (x < 5 nodes), v = 1.0, duration 1 ms

BC configurations:
  A. All Neumann (bounce-back) -- baseline
  B. Top/bottom Dirichlet (V = V_rest) -- current sink, expect SLOWDOWN not speedup
  C. All Neumann with D2Q5 -- control (no diagonal artifact)
  D. Top/bottom Dirichlet with D2Q5 -- control
```

### Measurements
1. **Activation time map**: time when V crosses threshold at each node
2. **CV profile**: compute CV as function of distance from boundary
   - Interior CV (y = Ny/2): baseline reference
   - Boundary CV (y = 1 or y = Ny-2): expect slowdown in config B, uniform in A
3. **CV ratio**: boundary_CV / interior_CV
   - Expected: < 1.0 for Dirichlet configs (current sink slows boundary)
   - Expected: ~= 1.0 for D2Q5 Neumann (no artifact)
   - Expected: ~= 1.0 for D2Q9 Neumann (O(dx^2) artifact, vanishes with refinement)

### Validation
| Test | Criterion |
|------|-----------|
| 8-V1 | Config A (all Neumann): uniform CV across y (no speedup) |
| 8-V2 | Config B (Dirichlet top/bottom): CV DECREASES near boundary (current sink effect) |
| 8-V3 | Config C/D (D2Q5 controls): uniform CV regardless of BC |

---

## Cross-References

| What | Where |
|------|-------|
| Rapaka's MRT formulation | research/PAPER_COMPARISON.md S 1 |
| Campos's MRT formulation | research/PAPER_COMPARISON.md S 1 |
| Bounce-back BC (Neumann) | Campos Eq. 17; Dawson, Chen & Doolen 1993 |
| Anti-bounce-back BC (Dirichlet) | Inamuro et al. 1995 |
| Dirichlet BC caveat (not Kleber) | research/BOUNDARY_SPEEDUP_ANALYSIS.md S 4 |
| D2Q9 bounce-back O(dx^2) artifact | research/BOUNDARY_SPEEDUP_ANALYSIS.md S 8 |
| D2Q9 moment matrix derivation | Lallemand & Luo, Phys Rev E 61(6), 2000 |
| Anisotropic LBM diffusion | Yoshida & Nagaoka, J Comp Phys 229, 2010 |
| Boundary speedup physiology | Kleber & Rudy, Physiol Rev 84(2), 2004 |
| MRT optimization tricks | research/PAPER_COMPARISON.md S 5 |
| D2Q9 vs D2Q5 for D_xy | This file, Phase 2 discussion |
| Symbol naming conventions | CONVENTIONS.md |
| tau-D mapping pipeline | CONVENTIONS.md S The tau-D Mapping Function |
| Boundary speedup theory | research/BOUNDARY_SPEEDUP_ANALYSIS.md |
| Why Dirichlet on V fails | research/BOUNDARY_SPEEDUP_ANALYSIS.md S 4 |
| Spatially varying D approach | research/BOUNDARY_SPEEDUP_ANALYSIS.md S 3 |
| Bidomain FDM approach | research/BOUNDARY_SPEEDUP_ANALYSIS.md S 2 |
