---

# Lattice-Boltzmann Method for Electrophysiology (LBM-EP): Comprehensive Research Report

## Table of Contents
1. [Paper Summary: Rapaka et al. (MICCAI 2012)](#1-paper-summary)
2. [D2Q7 / D3Q7 Lattice Structures for Diffusion](#2-lattice-structures)
3. [BGK vs MRT Collision Operators for the Monodomain Equation](#3-bgk-vs-mrt)
4. [Coupling LBM Diffusion with Rush-Larsen Ionic Stepping](#4-coupling-lbm-with-rush-larsen)
5. [Boundary Conditions: Bounce-Back for No-Flux](#5-boundary-conditions)
6. [Stability Analysis and Parameter Selection](#6-stability-analysis)
7. [GPU Implementation Strategies](#7-gpu-implementation)
8. [Open-Source LBM Cardiac Simulation Code](#8-open-source-code)
9. [The 10-45x Speedup Claim: When and Why](#9-speedup-analysis)
10. [Complete PyTorch/Python Implementation Blueprint](#10-implementation)

---

## 1. Paper Summary: Rapaka et al. (MICCAI 2012)

**Original Paper:** Rapaka, Mansi, Georgescu et al. (Siemens Research, MICCAI 2012)

The paper by Rapaka, Mansi, Georgescu et al. from Siemens Research introduced LBM-EP -- the first application of the Lattice-Boltzmann method to cardiac electrophysiology. Key elements:

**Problem solved:** The monodomain reaction-diffusion equation:

```
dv/dt = J_in + J_out + J_stim + c * div(D * grad(v))
```

where `v(t)` is the normalized transmembrane action potential in `[0,1]`, `D` is the anisotropic diffusion tensor along fiber direction `a` with anisotropy ratio `rho`, and `c` is the diffusion coefficient.

**Cell model:** Mitchell-Schaeffer (phenomenological, 2-variable):
- Inward current: `J_in = h * v^2 * (1 - v) / tau_in`
- Outward current: `J_out = -v / tau_out`
- Gating variable: `dh/dt = (1-h)/tau_open` if `v < v_gate`, else `dh/dt = -h/tau_close`

**LBM formulation (D3Q7):**
- 7-connectivity topology (6 cardinal neighbors + center) on a Cartesian grid
- Distribution functions `f_i(x)` for `i = 1..7`
- Weights: `omega_i = 1/8` for the 6 neighbor edges, `omega_i = 1/4` for center
- Two-step algorithm per time step:
  - **Collision** (Eq. 2): `f*_i = f_i - A_ij(f_j - omega_j * v) + dt * omega_i * (J_in + J_out + J_stim)`
  - **Streaming** (Eq. 3): `f_i(x + e_i, t + dt) = f*_i(x, t)`
- Potential recovered: `v(x, t) = sum_i f_i(x, t)`

**Isotropic diffusion coefficient:** `c = (2*tau - 1) / 8` for the D3Q7 lattice with BGK collision.

**Anisotropic diffusion via MRT:** The collision matrix becomes `A = M^{-1} S M` where `M` is a 7x7 transformation matrix and `S^{-1}` contains relaxation times `tau_ij = delta_ij/2 + 4*D_ij*dt/dx^2` related to the diffusion tensor components.

**Results:**
- Synthetic slab (401x401x3, dt=0.1ms): 80ms/iteration for LBM vs 700ms for FEM = **8.75x speedup**
- CESC'10 porcine heart (0.5mm grid, dt=0.1ms): 0.35s/iteration vs 16s for FEM = **45x speedup**
- Patient with scar (1mm grid): 0.2s/iteration
- All on single Xeon desktop, unoptimized Fortran, no GPU

---

## 2. D2Q7 / D3Q7 Lattice Structures for Diffusion

### D3Q7 (3D -- used in the paper)

The D3Q7 lattice is the primary lattice used in LBM-EP. It has 7 discrete velocities: 6 pointing to cardinal directions (plus/minus x, y, z) and 1 rest velocity.

**Velocity vectors:**
```
e_0 = ( 0,  0,  0)   # rest
e_1 = (+1,  0,  0)   # +x
e_2 = (-1,  0,  0)   # -x
e_3 = ( 0, +1,  0)   # +y
e_4 = ( 0, -1,  0)   # -y
e_5 = ( 0,  0, +1)   # +z
e_6 = ( 0,  0, -1)   # -z
```

**Weights (from the paper):**
- `w_0 = 1/4` (center/rest)
- `w_1..w_6 = 1/8` (each neighbor direction)
- Normalization: `1/4 + 6*(1/8) = 1/4 + 3/4 = 1`

**Diffusion coefficient relationship (BGK):**
```
D = c_s^2 * (tau - 0.5) * dt

For D3Q7 with the paper's weights:
c = (2*tau - 1) / 8
```

This means `c_s^2 = 1/4` for the D3Q7 lattice with these weights.

### D2Q7 (2D hexagonal -- for 2D simulations)

The D2Q7 lattice is hexagonal/triangular with 7 velocities (1 rest + 6 at 60-degree intervals).

**Velocity vectors:**
```
e_0 = (0, 0)                          # rest
e_i = (cos((i-1)*pi/3), sin((i-1)*pi/3))  for i = 1..6
```

**Standard weights:** `w_0 = 1/2`, `w_{1..6} = 1/12`

**Speed of sound:** `c_s = 1/sqrt(3)`

### D2Q5 (2D Cartesian -- simpler alternative for 2D)

For a 2D Cartesian implementation, D2Q5 is actually recommended over D2Q9 for pure diffusion problems (Li et al., 2016). It uses 5 velocities: 4 cardinal + 1 rest.

**Velocity vectors:**
```
e_0 = (0, 0)    # rest
e_1 = (1, 0)    # +x
e_2 = (-1, 0)   # -x
e_3 = (0, 1)    # +y
e_4 = (0, -1)   # -y
```

**Weights:** `w_0 = 1/3`, `w_{1..4} = 1/6`

**Diffusion coefficient (BGK):**
```
D = (1/3) * (tau - 0.5) * dx^2 / dt
```

For 2D cardiac simulation on a Cartesian grid, D2Q5 is the most natural analogue of the paper's D3Q7, and it avoids the complexity of hexagonal indexing that D2Q7 requires.

---

## 3. BGK vs MRT Collision Operators for the Monodomain Equation

### BGK (Bhatnagar-Gross-Krook) / Single Relaxation Time (SRT)

The simplest collision operator. All distributions relax toward equilibrium at the same rate:

```
f*_i = f_i - (1/tau) * (f_i - f_i^eq) + dt * w_i * S
```

where `f_i^eq = w_i * v` (for diffusion) and `S` is the source term (ionic currents).

**Advantages:**
- Extremely simple implementation
- Only one free parameter (tau)
- Sufficient for isotropic diffusion

**Limitations:**
- Cannot handle anisotropic diffusion natively
- Numerical instability when tau is close to 0.5 (small diffusion coefficients)
- Less accurate boundary conditions

### MRT (Multiple Relaxation Time)

The collision happens in moment space with different relaxation rates for different moments:

```
f* = f - M^{-1} S (Mf - Mf^eq) + dt * w * source
```

The key matrices from the paper (D3Q7):

**Transformation matrix M (7x7):**
```
M = [[1,  1,  1,  1,  1,  1,  1],    # conserved: v = sum(f_i)
     [1, -1,  0,  0,  0,  0,  0],    # gradient x
     [0,  0,  1, -1,  0,  0,  0],    # gradient y
     [0,  0,  0,  0,  1, -1,  0],    # gradient z
     [1,  1,  1,  1,  1,  1, -6],    # higher moment
     [1,  1, -1, -1,  0,  0,  0],    # higher moment
     [1,  1,  1,  1, -2, -2,  0]]    # higher moment
```

**Inverse relaxation matrix S^{-1}:**
- Rows 1-3 (gradient/flux components): `tau_ij = delta_ij/2 + 4*D_ij*dt/dx^2`
- Row 0 (conserved quantity): `tau_1 = 1` (fixed)
- Rows 4-6 (higher moments): `tau_5 = tau_6 = tau_7 = 1.33` (tunable for stability)

**Why MRT is essential for cardiac simulation:**
The cardiac tissue has strong anisotropy -- conduction velocity along fibers is roughly 3x faster than across fibers. The diffusion tensor is:

```
D = rho * I + (1 - rho) * a * a^T
```

where `a` is the fiber direction and `rho` is the anisotropy ratio (typically 0.25). The MRT formulation allows encoding the full 3x3 (or 2x2) diffusion tensor in the off-diagonal relaxation times, which is impossible with BGK.

**Practical recommendation:** Always use MRT for cardiac simulation. The computational overhead vs BGK is minimal (one matrix-vector multiply), but the ability to handle anisotropy and improved stability are essential.

---

## 4. Coupling LBM Diffusion with Rush-Larsen Ionic Stepping

### The Operator-Splitting Approach

The monodomain equation is naturally split into two operators:

```
dv/dt = R(v, h) + D(v)
```

where `R(v,h) = J_in + J_out + J_stim` (reaction/ionic) and `D(v) = c * div(D * grad(v))` (diffusion).

The standard approach uses **Strang splitting** (second-order) or **Godunov splitting** (first-order):

**Godunov (first-order, simpler):**
```
Step 1: Solve dv/dt = R(v, h) for dt  (pointwise ODE, Rush-Larsen)
Step 2: Solve dv/dt = D(v) for dt      (LBM collision + streaming)
```

**Strang (second-order):**
```
Step 1: Solve dv/dt = R(v, h) for dt/2  (half-step Rush-Larsen)
Step 2: Solve dv/dt = D(v) for dt       (full LBM step)
Step 3: Solve dv/dt = R(v, h) for dt/2  (half-step Rush-Larsen)
```

### The Rush-Larsen Method

The Rush-Larsen (RL) method is an exponential integrator specifically designed for Hodgkin-Huxley-type gating variables. For a gating variable `h` with the form:

```
dh/dt = (h_inf(V) - h) / tau_h(V)
```

The RL update (exact for frozen V) is:

```
h(t+dt) = h_inf + (h(t) - h_inf) * exp(-dt / tau_h)
```

For Mitchell-Schaeffer, the gating variable has piecewise dynamics:
- If `v < v_gate`: `h_inf = 1`, `tau_h = tau_open`
- If `v >= v_gate`: `h_inf = 0`, `tau_h = tau_close`

So the Rush-Larsen update is:

```python
if v < v_gate:
    h_new = 1.0 - (1.0 - h) * exp(-dt / tau_open)
else:
    h_new = h * exp(-dt / tau_close)
```

For the non-gating voltage variable, after the RL gating update, forward Euler is used for the reaction part:

```python
v_new = v + dt * (J_in + J_out + J_stim)
```

### Combined LBM + Rush-Larsen Algorithm

The paper uses a slightly different approach -- incorporating the source term directly into the collision step (Eq. 2). The combined algorithm is:

```
For each time step:
    1. At each node x:
       a. Compute ionic currents: J_in, J_out, J_stim from current v, h
       b. Collision with source:
          f*_i = f_i - A_ij(f_j - w_j*v) + dt*w_i*(J_in + J_out + J_stim)
       c. Update gating: h_new via Rush-Larsen (or forward Euler)
    2. At each node x:
       a. Stream: f_i(x + e_i, t+dt) = f*_i(x, t)
       b. Apply boundary conditions
    3. Recover potential: v = sum_i f_i
```

An alternative (and arguably cleaner) approach uses explicit operator splitting:

```
For each time step:
    1. REACTION half-step (dt/2) at each node:
       v += (dt/2) * (J_in(v,h) + J_out(v) + J_stim)
       h = rush_larsen_update(h, v, dt/2)
    2. UPDATE distributions from new v:
       f_i = w_i * v   (re-initialize to equilibrium)
    3. DIFFUSION full step via LBM:
       Collision: f*_i = f_i - A_ij(f_j - w_j*v)
       Streaming: f_i(x+e_i) = f*_i(x)
       v = sum_i f_i
    4. REACTION half-step (dt/2):
       v += (dt/2) * (J_in(v,h) + J_out(v) + J_stim)
       h = rush_larsen_update(h, v, dt/2)
```

However, re-initializing to equilibrium in step 2 destroys the non-equilibrium information. The paper's approach of embedding the source in the collision step preserves this information and is preferred.

---

## 5. Boundary Conditions: Bounce-Back for No-Flux

### The Physical Requirement

Cardiac tissue boundaries (epicardium, endocardium, around scars) require **zero-flux (homogeneous Neumann)** boundary conditions:

```
n . (D * grad(v)) = 0
```

where `n` is the outward normal to the boundary.

### Bounce-Back Implementation

In LBM, the potential gradient at a node is related to the distributions through:

```
c * grad(v) = (1 - 1/(2*tau)) * sum_i f_i * e_i
```

The Neumann condition `sum_i f_i * e_i . n = 0` is satisfied automatically when the incoming distribution equals the outgoing one. This is the **bounce-back rule**:

```
For a boundary node with wall normal to direction e_k:
    f_k_incoming(x, t+dt) = f_k_outgoing(x, t)
```

In practice, for a grid-aligned boundary at the +x face:
```
f_{-x}(x_boundary) = f_{+x}(x_boundary)   # incoming = outgoing
```

### Level-Set Boundary Handling

The paper uses a level-set `phi(x)` to define complex geometry:
- `phi(x) < 0`: inside the myocardium
- `phi(x) > 0`: outside
- `phi(x) = 0`: boundary

For nodes near the boundary, the incoming distribution is calculated from the distance to the wall (provided by the level-set), using interpolated bounce-back:

```python
# For boundary node x with neighbor x+e_i outside domain:
# q = distance_to_wall / dx (fractional distance, 0 < q <= 1)
# Simple bounce-back (first order):
f_opposite(x, t+dt) = f_i_star(x, t)

# Interpolated bounce-back (second order, for curved boundaries):
if q < 0.5:
    f_opposite(x, t+dt) = 2*q*f_i_star(x,t) + (1-2*q)*f_i_star(x-e_i,t)
else:
    f_opposite(x, t+dt) = (1/(2*q))*f_i_star(x,t) + (1-1/(2*q))*f_opposite_star(x,t)
```

### Scar Handling

Scars are treated as internal boundaries with the same bounce-back condition, which naturally blocks conduction:
- Scar nodes identified from level-set (e.g., from late gadolinium enhancement MRI)
- At scar boundaries, apply bounce-back to enforce zero flux
- The wavefront naturally bends around the scar

---

## 6. Stability Analysis and Parameter Selection

### The Fundamental Relationship

For the D3Q7 lattice (as in the paper):

```
c = (2*tau - 1) / 8

Therefore:
tau = (4*c*dx^2/dt + 1) / 2 = 0.5 + 4*c*dt/dx^2
```

Wait -- let me correct this using the paper's notation carefully. The paper states `c = (2*tau - 1)/8` where `c` is the diffusion coefficient. Solving for `tau`:

```
tau = (8c + 1) / 2 = 0.5 + 4c
```

But this assumes `dx = dt = 1` in lattice units. In physical units:

```
tau = 0.5 + 4 * D_physical * dt / dx^2
```

where `D_physical` is the physical diffusion coefficient.

### Stability Constraints

**Hard constraint:** `tau > 0.5` (required for positive diffusion)

**Practical constraints:**
- `tau` close to 0.5: numerical instability (small diffusion, Gibbs oscillations)
- `tau >> 1`: excessive numerical diffusion, accuracy loss
- **Recommended range:** `0.5 < tau < 2.0`, ideally `0.6 < tau < 1.5`
- **Optimal value** (from literature): `tau ~ 0.8` provides good balance of stability and accuracy

### Parameter Selection for Cardiac Simulation

Typical cardiac parameters:
- Diffusion coefficient along fibers: `D = 0.001 cm^2/ms` (range: 0.0003 to 0.0035 cm^2/ms)
- Grid spacing: `dx = 0.025 cm` (0.25 mm) to `dx = 0.1 cm` (1 mm)
- Time step: `dt = 0.01 ms` to `dt = 0.5 ms`

**Example calculation (from the paper's CESC'10 experiment):**
```
D = 0.0035 cm^2/ms, dx = 0.05 cm, dt = 0.1 ms

tau_fiber = 0.5 + 4 * D * dt / dx^2
         = 0.5 + 4 * 0.0035 * 0.1 / 0.05^2
         = 0.5 + 4 * 0.00035 / 0.0025
         = 0.5 + 0.56
         = 1.06    (good -- in the stable range)
```

**Cross-fiber with rho=0.25:**
```
D_cross = rho * D = 0.25 * 0.0035 = 0.000875 cm^2/ms

tau_cross = 0.5 + 4 * 0.000875 * 0.1 / 0.0025
          = 0.5 + 0.14
          = 0.64    (still stable, but closer to 0.5)
```

### Von Neumann Stability Analysis Summary

From the literature, the key stability results for the BGK/SRT advection-diffusion LBM:

1. The scheme is linearly stable when `tau > 0.5` and the lattice Peclet number `Pe = u*dx/D` is below a threshold
2. Non-negativity of equilibrium distribution functions is a sufficient condition for stability
3. Second-order equilibrium distributions have larger stability domains than linear ones
4. The TRT (two-relaxation-time) model with the "magic parameter" `Lambda = (tau_+ - 0.5)(tau_- - 0.5)` set to `1/4` provides optimal stability independent of Peclet number

### MRT Higher-Moment Relaxation Times

For the non-physical relaxation times (tau_5, tau_6, tau_7 in the paper):
- Setting them to 1.0 is simplest
- The paper uses 1.33 for improved stability
- They should satisfy `tau_k > 0.5` for stability
- They do not affect the diffusion solution, only stability

---

## 7. GPU Implementation Strategies

### Why LBM is Ideal for GPU

The LBM algorithm is embarrassingly parallel:
1. **Collision** is purely local (each node independent)
2. **Streaming** only accesses immediate neighbors (regular memory access pattern)
3. No global linear system solve (unlike implicit FEM)
4. Uniform Cartesian grid (no irregular memory access)

### Architecture: Fused Collision-Streaming Kernel

The most efficient GPU approach fuses collision and streaming into a single kernel:

```python
# Pseudocode for fused kernel
@cuda_kernel
def lbm_step(f_in, f_out, h, v, domain_mask, ...):
    # Each thread handles one lattice node
    idx = blockIdx.x * blockDim.x + threadIdx.x
    
    if not domain_mask[idx]:
        return
    
    # 1. Load distributions from current node
    f_local = [f_in[i][idx] for i in range(7)]
    
    # 2. Compute macroscopic variable
    v_local = sum(f_local)
    
    # 3. Compute ionic currents (reaction)
    h_local = h[idx]
    J_in = h_local * v_local**2 * (1 - v_local) / tau_in
    J_out = -v_local / tau_out
    
    # 4. Collision (with source term)
    # MRT: transform to moment space, relax, transform back
    m = M @ f_local
    m_eq = compute_equilibrium_moments(v_local)
    m_star = m - S @ (m - m_eq)
    m_star[0] += dt * (J_in + J_out + J_stim)  # source to conserved moment
    f_star = M_inv @ m_star
    
    # 5. Stream: write to neighbor locations in output array
    for i in range(7):
        neighbor_idx = get_neighbor(idx, i)
        if is_boundary(neighbor_idx):
            # Bounce-back
            f_out[opposite(i)][idx] = f_star[i]
        else:
            f_out[i][neighbor_idx] = f_star[i]
    
    # 6. Update gating variable (Rush-Larsen)
    h[idx] = rush_larsen(h_local, v_local, dt)
```

### Memory Layout: Structure of Arrays (SoA)

For coalesced GPU memory access, use SoA layout:

```python
# BAD: Array of Structures (AoS) -- non-coalesced access
# f[node_idx][direction] -- adjacent threads access non-adjacent memory

# GOOD: Structure of Arrays (SoA) -- coalesced access
# f[direction][node_idx] -- adjacent threads access adjacent memory
```

In PyTorch, this means:
```python
# f shape: (Q, Nx, Ny, Nz) or (Q, Nx, Ny) for 2D
f = torch.zeros(7, Nx, Ny, Nz, device='cuda', dtype=torch.float32)
```

### A-B Pattern (Double Buffering)

To avoid race conditions during streaming, use two grids:

```python
f_src = torch.zeros(7, Nx, Ny, device='cuda')
f_dst = torch.zeros(7, Nx, Ny, device='cuda')

for step in range(n_steps):
    # Collision on f_src, stream to f_dst
    collide_and_stream(f_src, f_dst, ...)
    # Swap
    f_src, f_dst = f_dst, f_src
```

### Performance Benchmarks

From the literature:
- **Campos et al. (2016):** 419 MLUPS on GPU with D3Q7 MRT for cardiac, 97s for 128^3 mesh full simulation
- **FluidX3D:** Up to 8.4 GLUPS on A100 with D3Q19 (fluid dynamics)
- **Lettuce (PyTorch):** Consumer GPU-level performance, competitive with CUDA for prototyping

### PyTorch-Specific Optimizations

1. Use `torch.roll()` for streaming (simple but not the most efficient)
2. Use `torch.nn.functional.pad()` for boundary handling
3. Avoid Python loops -- vectorize everything as tensor operations
4. Use `torch.compile()` (PyTorch 2.0+) or custom CUDA extensions for critical loops
5. Use `float32` (not float64) -- sufficient for LBM with proper implementation

---

## 8. Open-Source LBM Cardiac Simulation Code

### Dedicated LBM Cardiac Code

There is **no widely-available open-source repository** that specifically implements LBM-EP for cardiac electrophysiology. The original LBM-EP paper (Rapaka et al.) was from Siemens Research and the code was not publicly released. The Campos et al. GPU implementation was published as a paper but the code is not on GitHub.

### General LBM Frameworks That Could Be Adapted

| Framework | Language | GPU | License | Link |
|-----------|----------|-----|---------|------|
| **Lettuce** | Python/PyTorch | CUDA via PyTorch | MIT | [github.com/lettucecfd/lettuce](https://github.com/lettucecfd/lettuce) |
| **TorchLBM** | Python/PyTorch | CUDA + Modulus | Research | [TUM MEP](https://www.mep.tum.de/mep/scicohub/torchlbm/) |
| **XLB** | Python/JAX | CUDA/TPU/Multi-GPU | Open | [arxiv.org/html/2311.16080v3](https://arxiv.org/html/2311.16080v3) |
| **OpenLB** | C++ | CPU (parallel) | GPLv2 | [openlb.net](https://www.openlb.net/) |
| **FluidX3D** | C++/OpenCL | All GPUs | Non-commercial | [github.com/ProjectPhysX/FluidX3D](https://github.com/ProjectPhysX/FluidX3D) |
| **jviquerat/lbm** | Python | CPU | Open | [github.com/jviquerat/lbm](https://github.com/jviquerat/lbm) |

### Cardiac Simulators (Not LBM, but FEM/FDM)

| Tool | Method | GPU | Link |
|------|--------|-----|------|
| **openCARP** | FEM (PETSc) | CPU (GPU via MLIR) | [opencarp.org](https://opencarp.org/) |
| **SimVascular** | FEM | CPU | [simvascular.github.io](https://simvascular.github.io/) |
| **TorchCor** | FEM (PyTorch) | CUDA | [arxiv.org/html/2510.12011](https://arxiv.org/html/2510.12011) |
| **lifex-ep** | FEM (deal.II) | CPU | [arxiv.org/abs/2308.01651](https://arxiv.org/abs/2308.01651) |

### Curated List of LBM Codes

The repository at [github.com/sthavishtha/list-lattice-Boltzmann-codes](https://github.com/sthavishtha/list-lattice-Boltzmann-codes) maintains an extensive list of open-source LBM implementations across multiple languages and platforms.

**Practical recommendation:** The most viable path for building an LBM cardiac simulator in Python/PyTorch is to start from **Lettuce** (which already has BGK, MRT, streaming, and GPU support via PyTorch) and add the cardiac-specific components: ionic models, Rush-Larsen integration, level-set boundaries, and fiber architecture.

---

## 9. The 10-45x Speedup Claim: When and Why

### Where the Numbers Come From

The paper reports three scenarios with different speedups:

| Scenario | LBM time/iter | FEM time/iter | Speedup | Grid |
|----------|---------------|---------------|---------|------|
| Synthetic slab | 80 ms | 700 ms | **8.75x** | 401x401x3, dx=0.25mm |
| CESC'10 heart | 0.35 s | 16 s | **45x** | 0.5mm, ~500K nodes |
| Patient scar | 0.2 s | N/A | N/A | 1mm |

### Why Such a Wide Range (10-45x)?

The speedup depends critically on several factors:

**1. Problem size and mesh complexity:**
- FEM requires mesh generation (tetrahedral meshing of complex heart geometry) and a global sparse linear system solve
- As mesh size increases, FEM cost grows super-linearly (due to linear solver), while LBM cost grows linearly
- Larger, more complex geometries favor LBM more

**2. FEM time step restriction:**
- The paper used `dt_FEM = 0.5 ms` vs `dt_LBM = 0.1 ms` for CESC'10
- FEM with implicit time stepping allows larger dt, but each step is much more expensive
- For the same total simulation time, LBM takes 5x more steps but each step is ~225x cheaper

**3. Parallelism:**
- LBM: trivially parallel, pure nearest-neighbor communication
- FEM: requires a linear system solve (CG/GMRES with preconditioner), which has communication bottlenecks
- The paper's FEM used OpenMP (shared memory), while LBM was single-threaded unoptimized Fortran -- so the 45x speedup is comparing parallelized FEM vs. serial LBM

**4. No meshing overhead:**
- LBM operates on a Cartesian grid with level-set boundaries
- FEM requires high-quality tetrahedral meshes, which can take minutes to hours to generate

### Conditions That Maximize LBM Advantage

- **Large 3D domains** with complex geometry (heart anatomy)
- **GPU execution**: LBM maps perfectly to GPU; Campos et al. showed 419 MLUPS = another 100-500x over CPU LBM
- **Explicit time stepping**: when the stiffness is handled by the ionic solver (Rush-Larsen), the PDE diffusion can be explicit
- **Cartesian-compatible geometries**: level-set representations work well for hearts segmented from MRI/CT

### Conditions That Reduce LBM Advantage

- **Small domains** (slab test: only 8.75x speedup vs 45x for the full heart)
- **When FEM uses very optimized solvers** (AMG preconditioners, GPU-accelerated sparse linear algebra)
- **When high-order accuracy is needed** (LBM is second-order; FEM can use higher-order elements)
- **Bidomain model** (not just monodomain) -- requires solving an additional elliptic PDE where FEM's implicit solvers have an advantage

### Post-Paper Developments

- **Campos et al. (2016)** achieved 419 MLUPS on GPU with LBM-EP, with overall simulation speedup of ~500x compared to CPU FEM
- **GPU FEM** (TorchCor, openCARP with MLIR) has also dramatically improved, narrowing the gap
- The fundamental architectural advantage of LBM (locality, no global solve) persists on GPU

---

## 10. Complete PyTorch/Python Implementation Blueprint

Below is a practical implementation of a 2D LBM-EP cardiac simulator in Python/PyTorch, using the D2Q5 lattice (Cartesian 2D analogue of D3Q7) with MRT collision and Mitchell-Schaeffer ionic model with Rush-Larsen stepping.

### Core Data Structures

```python
import torch
import torch.nn.functional as F
import numpy as np
import math

class LBM_EP_2D:
    """
    2D Lattice-Boltzmann Cardiac Electrophysiology Simulator
    Uses D2Q5 lattice with MRT collision for anisotropic diffusion
    and Mitchell-Schaeffer ionic model with Rush-Larsen stepping.
    """
    
    # D2Q5 lattice vectors: rest, +x, -x, +y, -y
    #   e_i = [(0,0), (1,0), (-1,0), (0,1), (0,-1)]
    # Weights: w_0 = 1/3, w_{1..4} = 1/6
    # Speed of sound squared: c_s^2 = 1/3
    # Diffusion relation (BGK): D = c_s^2 * (tau - 0.5) * dx^2 / dt
    #                             = (1/3) * (tau - 0.5) * dx^2 / dt
    
    Q = 5  # number of discrete velocities
    
    def __init__(self, Nx, Ny, dx, dt, D_fiber, D_cross,
                 fiber_angle_field=None, domain_mask=None,
                 device='cuda'):
        """
        Args:
            Nx, Ny: grid dimensions
            dx: spatial resolution (cm)
            dt: time step (ms)
            D_fiber: diffusion coefficient along fibers (cm^2/ms)
            D_cross: diffusion coefficient across fibers (cm^2/ms)
            fiber_angle_field: (Ny, Nx) tensor of fiber angles (radians)
            domain_mask: (Ny, Nx) boolean tensor (True = myocardium)
        """
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dt = dx, dt
        self.device = device
        
        # D2Q5 velocity vectors
        self.e = torch.tensor([
            [0, 0],   # rest
            [1, 0],   # +x
            [-1, 0],  # -x
            [0, 1],   # +y
            [0, -1],  # -y
        ], dtype=torch.float32, device=device)
        
        # Weights
        self.w = torch.tensor([1/3, 1/6, 1/6, 1/6, 1/6],
                              dtype=torch.float32, device=device)
        
        # Opposite direction indices (for bounce-back)
        self.opposite = [0, 2, 1, 4, 3]  # 0<->0, 1<->2, 3<->4
        
        # Domain mask
        if domain_mask is None:
            self.mask = torch.ones(Ny, Nx, dtype=torch.bool, device=device)
        else:
            self.mask = domain_mask.to(device)
        
        # Fiber angles
        if fiber_angle_field is None:
            self.theta = torch.zeros(Ny, Nx, dtype=torch.float32, device=device)
        else:
            self.theta = fiber_angle_field.to(device)
        
        # Precompute MRT relaxation matrices per node
        self._setup_mrt(D_fiber, D_cross)
        
        # Distribution functions: (Q, Ny, Nx)
        self.f = torch.zeros(self.Q, Ny, Nx, dtype=torch.float32, device=device)
        
        # State variables
        self.v = torch.zeros(Ny, Nx, dtype=torch.float32, device=device)  # voltage
        self.h = torch.ones(Ny, Nx, dtype=torch.float32, device=device)   # gate
        
        # Initialize distributions to equilibrium
        self._init_equilibrium()
    
    def _setup_mrt(self, D_fiber, D_cross):
        """
        Set up the MRT collision operator for anisotropic diffusion.
        
        For D2Q5, the transformation matrix M (5x5) maps distributions
        to moments: [density, jx, jy, energy, energy_square]
        
        The relaxation in moment space has rates tied to the diffusion tensor.
        """
        dx, dt = self.dx, self.dt
        
        # Transformation matrix for D2Q5
        # Rows: conserved density, x-flux, y-flux, energy, energy-square
        self.M = torch.tensor([
            [1,  1,  1,  1,  1],     # m0 = sum(f_i) = v (conserved)
            [0,  1, -1,  0,  0],     # m1 = f1 - f2 (x-flux ~ dv/dx)
            [0,  0,  0,  1, -1],     # m2 = f3 - f4 (y-flux ~ dv/dy)
            [-4, 1,  1,  1,  1],     # m3 = energy moment
            [4, -2, -2, -2, -2],     # m4 = energy-square moment
        ], dtype=torch.float32, device=self.device)
        
        self.M_inv = torch.linalg.inv(self.M)
        
        # Build per-node relaxation matrix in moment space
        # For anisotropic diffusion, the flux relaxation times depend on
        # the local diffusion tensor D_ij
        
        cos_t = torch.cos(self.theta)
        sin_t = torch.sin(self.theta)
        
        # Diffusion tensor in 2D: D = R * diag(D_fiber, D_cross) * R^T
        # D_xx = D_fiber * cos^2(theta) + D_cross * sin^2(theta)
        # D_yy = D_fiber * sin^2(theta) + D_cross * cos^2(theta)
        # D_xy = (D_fiber - D_cross) * cos(theta) * sin(theta)
        D_xx = D_fiber * cos_t**2 + D_cross * sin_t**2
        D_yy = D_fiber * sin_t**2 + D_cross * cos_t**2
        D_xy = (D_fiber - D_cross) * cos_t * sin_t
        
        # Relaxation times for flux moments:
        # For D2Q5 with c_s^2 = 1/3:
        #   tau_flux = 0.5 + D * dt / (c_s^2 * dx^2)
        #            = 0.5 + 3 * D * dt / dx^2
        c_s2 = 1.0 / 3.0
        
        self.s_xx = 1.0 / (0.5 + D_xx * dt / (c_s2 * dx**2))  # (Ny, Nx)
        self.s_yy = 1.0 / (0.5 + D_yy * dt / (c_s2 * dx**2))
        self.s_xy = D_xy * dt / (c_s2 * dx**2)  # off-diagonal coupling
        
        # Higher-moment relaxation rates (free parameters for stability)
        self.s_energy = 1.0 / 1.0       # tau_3 = 1.0
        self.s_energy_sq = 1.0 / 1.33   # tau_4 = 1.33
    
    def _init_equilibrium(self):
        """Initialize distribution functions to equilibrium: f_i = w_i * v"""
        for i in range(self.Q):
            self.f[i] = self.w[i] * self.v
    
    def set_stimulus(self, region_mask, v_value=1.0, duration_ms=1.0):
        """Apply electrical stimulus to a region."""
        self.v[region_mask] = v_value
        self._init_equilibrium()  # re-equilibrate after setting v
    
    def _collide_mrt(self):
        """
        MRT collision step with source term.
        
        For the general anisotropic case with D2Q5, we transform to
        moment space, relax each moment, and transform back.
        
        This handles the diagonal anisotropy through separate relaxation
        rates for x-flux and y-flux moments.
        """
        # Compute moments: m = M @ f (for each spatial point)
        # f shape: (5, Ny, Nx), M shape: (5, 5)
        f_flat = self.f.reshape(self.Q, -1)  # (5, N)
        m = self.M @ f_flat                   # (5, N)
        m = m.reshape(self.Q, self.Ny, self.Nx)
        
        # Current voltage
        v = m[0]  # conserved moment = voltage
        
        # Equilibrium moments
        m_eq = torch.zeros_like(m)
        m_eq[0] = v      # density: v
        m_eq[1] = 0.0    # x-flux: 0 at equilibrium (no advection)
        m_eq[2] = 0.0    # y-flux: 0
        m_eq[3] = -4.0/3.0 * v  # energy equilibrium for D2Q5
        m_eq[4] = 4.0/3.0 * v   # energy-square equilibrium
        
        # Compute ionic currents (Mitchell-Schaeffer)
        J_in = self.h * v * v * (1.0 - v) / self.tau_in
        J_out = -v / self.tau_out
        J_stim = self.J_stim if hasattr(self, 'J_stim') else 0.0
        source = J_in + J_out + J_stim
        
        # Relax moments
        # m0 (conserved): no relaxation, but add source
        m_star = m.clone()
        m_star[0] = m[0] + self.dt * source
        
        # m1, m2 (fluxes): relax with diffusion-related rates
        # For diagonal anisotropy (D_xy = 0):
        m_star[1] = m[1] - self.s_xx * (m[1] - m_eq[1])
        m_star[2] = m[2] - self.s_yy * (m[2] - m_eq[2])
        
        # Handle off-diagonal diffusion (fiber rotation)
        # Cross-coupling between x and y flux moments
        if torch.any(self.s_xy != 0):
            # Coupled relaxation for off-diagonal diffusion
            dm1 = -self.s_xy * (m[2] - m_eq[2])
            dm2 = -self.s_xy * (m[1] - m_eq[1])
            m_star[1] = m_star[1] + dm1
            m_star[2] = m_star[2] + dm2
        
        # m3, m4 (higher moments): relax for stability
        m_star[3] = m[3] - self.s_energy * (m[3] - m_eq[3])
        m_star[4] = m[4] - self.s_energy_sq * (m[4] - m_eq[4])
        
        # Transform back to distribution space
        m_star_flat = m_star.reshape(self.Q, -1)
        f_star = self.M_inv @ m_star_flat
        self.f = f_star.reshape(self.Q, self.Ny, self.Nx)
        
        # Update gating variable (Rush-Larsen)
        self._rush_larsen_update(v)
        
        # Update voltage
        self.v = self.f.sum(dim=0)
    
    def _rush_larsen_update(self, v):
        """
        Rush-Larsen exponential integration for Mitchell-Schaeffer gate.
        
        dh/dt = (1-h)/tau_open   if v < v_gate
        dh/dt = -h/tau_close     if v >= v_gate
        
        Analytical solution (frozen v):
        h(t+dt) = h_inf + (h - h_inf) * exp(-dt/tau_h)
        
        Below threshold: h_inf = 1, tau_h = tau_open
        Above threshold: h_inf = 0, tau_h = tau_close
        """
        below = v < self.v_gate
        above = ~below
        
        # Below v_gate: h -> 1 with time constant tau_open
        exp_open = torch.exp(-self.dt / self.tau_open)
        h_below = 1.0 - (1.0 - self.h) * exp_open
        
        # Above v_gate: h -> 0 with time constant tau_close
        exp_close = torch.exp(-self.dt / self.tau_close)
        h_above = self.h * exp_close
        
        self.h = torch.where(below, h_below, h_above)
    
    def _stream(self):
        """
        Streaming step: f_i(x + e_i, t+dt) = f*_i(x, t)
        
        Uses torch.roll for simplicity. Each distribution function
        is shifted by its velocity vector.
        """
        f_new = torch.zeros_like(self.f)
        
        # Direction 0: rest (no streaming)
        f_new[0] = self.f[0]
        
        # Direction 1: +x -> roll by -1 in x (dim=2)
        f_new[1] = torch.roll(self.f[1], shifts=-1, dims=1)
        
        # Direction 2: -x -> roll by +1 in x
        f_new[2] = torch.roll(self.f[2], shifts=1, dims=1)
        
        # Direction 3: +y -> roll by -1 in y (dim=1)
        f_new[3] = torch.roll(self.f[3], shifts=-1, dims=0)
        
        # Direction 4: -y -> roll by +1 in y
        f_new[4] = torch.roll(self.f[4], shifts=1, dims=0)
        
        self.f = f_new
    
    def _apply_boundary_conditions(self):
        """
        Apply bounce-back boundary conditions at domain boundaries.
        
        For no-flux (Neumann) BC: incoming distribution = outgoing distribution.
        At nodes outside the domain, reflect distributions back.
        """
        # For each direction, if the neighbor is outside the domain,
        # apply bounce-back: f_opposite(x) = f_i(x)
        
        # Create shifted masks to identify boundary neighbors
        mask = self.mask.float()
        
        # +x boundary: if node at (y, x+1) is outside, bounce f1 -> f2
        mask_px = torch.roll(mask, shifts=-1, dims=1)
        bounce_px = (mask > 0) & (mask_px == 0)
        self.f[2][bounce_px] = self.f[1][bounce_px]
        
        # -x boundary
        mask_mx = torch.roll(mask, shifts=1, dims=1)
        bounce_mx = (mask > 0) & (mask_mx == 0)
        self.f[1][bounce_mx] = self.f[2][bounce_mx]
        
        # +y boundary
        mask_py = torch.roll(mask, shifts=-1, dims=0)
        bounce_py = (mask > 0) & (mask_py == 0)
        self.f[4][bounce_py] = self.f[3][bounce_py]
        
        # -y boundary
        mask_my = torch.roll(mask, shifts=1, dims=0)
        bounce_my = (mask > 0) & (mask_my == 0)
        self.f[3][bounce_my] = self.f[4][bounce_my]
        
        # Zero out distributions outside domain
        for i in range(self.Q):
            self.f[i] = self.f[i] * self.mask.float()
    
    def step(self):
        """Execute one LBM-EP time step."""
        self._collide_mrt()
        self._stream()
        self._apply_boundary_conditions()
        self.v = self.f.sum(dim=0)
    
    def run(self, n_steps, save_every=None):
        """Run simulation for n_steps."""
        history = []
        for t in range(n_steps):
            self.step()
            if save_every and t % save_every == 0:
                history.append(self.v.cpu().clone())
        return history
```

### Mitchell-Schaeffer Parameters Setup

```python
def setup_mitchell_schaeffer(sim):
    """Configure Mitchell-Schaeffer ionic model parameters."""
    sim.tau_in = 0.3      # ms (fast inward current time constant)
    sim.tau_out = 6.0      # ms (outward current time constant)  
    sim.tau_open = 120.0   # ms (gate opening time constant)
    sim.tau_close = 150.0  # ms (gate closing time constant)
    sim.v_gate = 0.13      # dimensionless (gate threshold voltage)
    sim.J_stim = 0.0       # stimulus current (set per-node)
```

### Example Usage

```python
# --- Simulation setup ---
Nx, Ny = 400, 400
dx = 0.025  # cm (0.25 mm)
dt = 0.01   # ms

# Diffusion coefficients
D_fiber = 0.001   # cm^2/ms (along fibers)
D_cross = 0.00025 # cm^2/ms (across fibers, rho=0.25)

# Verify stability
c_s2 = 1.0 / 3.0
tau_fiber = 0.5 + D_fiber * dt / (c_s2 * dx**2)
tau_cross = 0.5 + D_cross * dt / (c_s2 * dx**2)
print(f"tau_fiber = {tau_fiber:.4f}")  # Should be > 0.5
print(f"tau_cross = {tau_cross:.4f}")  # Should be > 0.5

# Create simulator
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sim = LBM_EP_2D(Nx, Ny, dx, dt, D_fiber, D_cross, device=device)
setup_mitchell_schaeffer(sim)

# --- Apply stimulus ---
# Stimulate left edge for 1 ms
stim_region = torch.zeros(Ny, Nx, dtype=torch.bool)
stim_region[:, :5] = True  # leftmost 5 columns
sim.v[stim_region] = 1.0
sim.h[stim_region] = 1.0
sim._init_equilibrium()

# --- Run simulation ---
n_steps = int(500 / dt)  # 500 ms of simulation
history = sim.run(n_steps, save_every=int(10 / dt))

# --- Visualize ---
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    if i < len(history):
        im = ax.imshow(history[i].numpy(), cmap='hot', vmin=0, vmax=1)
        ax.set_title(f't = {i * 10} ms')
        ax.axis('off')
plt.tight_layout()
plt.savefig('lbm_ep_propagation.png', dpi=150)
plt.show()
```

### Streaming via `torch.roll` -- Performance Note

The `torch.roll` approach is clean but involves memory copies. For maximum performance, an index-based streaming approach is faster:

```python
def _stream_indexed(self):
    """
    Index-based streaming using pre-computed neighbor indices.
    More memory-efficient than torch.roll for large domains.
    """
    # Pre-compute flat indices for each direction's neighbor
    # (do this once in __init__)
    if not hasattr(self, '_stream_indices'):
        self._precompute_stream_indices()
    
    f_flat = self.f.reshape(self.Q, -1)
    f_new = torch.zeros_like(f_flat)
    
    for i in range(self.Q):
        f_new[i].scatter_(0, self._stream_dst[i], f_flat[i])
    
    self.f = f_new.reshape(self.Q, self.Ny, self.Nx)
```

### Extension to 3D (D3Q7)

The extension to 3D is straightforward -- add two more velocity directions (`+z`, `-z`) and adjust weights:

```python
# D3Q7 velocity vectors
e = [(0,0,0), (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

# D3Q7 weights (as in the paper)
w = [1/4, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8]

# Diffusion relation: D = (2*tau - 1) / 8  (in lattice units)
# Or equivalently: tau = 0.5 + 4 * D * dt / dx^2

# Distribution tensor shape: (7, Nz, Ny, Nx)
```

### Key Formulas Summary

| Quantity | D2Q5 Formula | D3Q7 Formula |
|----------|-------------|-------------|
| Weights (center) | `w_0 = 1/3` | `w_0 = 1/4` |
| Weights (neighbors) | `w_{1-4} = 1/6` | `w_{1-6} = 1/8` |
| `c_s^2` | `1/3` | `1/4` |
| `D(tau)` | `(1/3)(tau - 1/2) dx^2/dt` | `(1/4)(tau - 1/2) dx^2/dt` |
| `tau(D)` | `0.5 + 3D*dt/dx^2` | `0.5 + 4D*dt/dx^2` |
| Stability | `tau > 0.5` | `tau > 0.5` |
| MRT: `tau_ij` | `delta_ij/2 + 3*D_ij*dt/dx^2` | `delta_ij/2 + 4*D_ij*dt/dx^2` |

---

## Summary and Recommendations

**For building an LBM cardiac simulator in Python/PyTorch:**

1. **Start 2D with D2Q5** on a Cartesian grid. This is the simplest lattice that captures diffusion correctly and maps directly to PyTorch tensor operations. The D2Q5 is explicitly recommended over D2Q9 for diffusion-only problems by the literature.

2. **Use MRT collision** from the beginning. The overhead is minimal (one 5x5 matrix multiply per node), but it handles anisotropic diffusion and provides better stability. The paper's M and S matrices for D3Q7 map directly to D2Q5 analogues.

3. **Couple with Rush-Larsen** for the ionic model. Embed the source term in the collision step (as in the paper's Eq. 2) rather than using operator splitting, to preserve non-equilibrium information. Use Rush-Larsen for the gating variable update, which allows larger time steps than forward Euler.

4. **Bounce-back boundaries** for no-flux conditions. Simple bounce-back is first-order accurate but mass-conserving and trivial to implement. For curved boundaries, interpolated bounce-back provides second-order accuracy.

5. **Use Lettuce** ([github.com/lettucecfd/lettuce](https://github.com/lettucecfd/lettuce)) as a reference for PyTorch LBM patterns (SoA layout, roll-based streaming, collision abstractions).

6. **GPU performance**: PyTorch gives you GPU execution "for free" via tensor operations. For production performance, use `torch.compile()` and consider custom CUDA kernels. The fundamental LBM advantage is that it requires only nearest-neighbor communication and no global linear system solve.

7. **Verify with the paper's slab test**: 10x10x0.5 cm slab, dx=0.25mm, dt=0.1ms, Mitchell-Schaeffer parameters as listed. Compare wavefront speed and action potential shape against known analytical/FEM solutions.

8. **The 10-45x speedup** is achievable vs. standard CPU FEM and increases with problem size and geometric complexity. On GPU, Campos et al. showed an additional ~100x over CPU LBM (419 MLUPS), making near-real-time cardiac simulation feasible.

### Sources

- [Rapaka et al. - LBM-EP (MICCAI 2012)](https://comaniciu.net/Papers/LBM_EP_MICCAI12.pdf)
- [Campos et al. - LBM GPU Cardiac Electrophysiology (ScienceDirect 2016)](https://www.sciencedirect.com/science/article/pii/S0377042715000692)
- [Yoshida & Nagaoka - MRT LBM for Anisotropic Diffusion (J. Comp. Phys. 2010)](https://www.sciencedirect.com/science/article/abs/pii/S0021999110003134)
- [Dawson et al. - LBM for Reaction-Diffusion (J. Chem. Phys. 1993)](https://pubs.aip.org/aip/jcp/article-abstract/98/2/1514/461768)
- [Marsh et al. - Secrets of Rush-Larsen Method (IEEE TBME 2012)](https://pubmed.ncbi.nlm.nih.gov/22736685/)
- [Lettuce PyTorch LBM Framework](https://github.com/lettucecfd/lettuce)
- [XLB: JAX-based LBM Library](https://arxiv.org/html/2311.16080v3)
- [TorchLBM (TU Munich)](https://www.mep.tum.de/mep/scicohub/torchlbm/)
- [FluidX3D - GPU LBM Code](https://github.com/ProjectPhysX/FluidX3D)
- [OpenLB - Open Source LBM](https://www.openlb.net/)
- [openCARP - Cardiac Electrophysiology Simulator](https://opencarp.org/)
- [Li et al. - D2Q5 vs D2Q9 for Convection-Diffusion (ScienceDirect 2016)](https://www.sciencedirect.com/science/article/abs/pii/S0017931016326047)
- [Ginzburg - Optimal Stability of TRT LBM (J. Stat. Phys. 2010)](https://link.springer.com/article/10.1007/s10955-010-9969-9)
- [Coupled LBM for Bidomain Models (MMNP 2019)](https://www.mmnp-journal.org/articles/mmnp/abs/2019/02/mmnp190101/mmnp190101.html)
- [Curated List of LBM Codes](https://github.com/sthavishtha/list-lattice-Boltzmann-codes)
- [lifex-ep FEM Cardiac Simulator](https://arxiv.org/abs/2308.01651)
- [TorchCor PyTorch FEM Cardiac Simulator](https://arxiv.org/html/2510.12011)
- [LBM Wikipedia](https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods)
- [Lattice Boltzmann D2Q9 Python Tutorial (Mocz)](https://medium.com/swlh/create-your-own-lattice-boltzmann-simulation-with-python-8759e8b53b1c)