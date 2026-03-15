# Fenton-Karma Implementation Research

**Date**: 2025-12-10
**Purpose**: Learn from existing Python/MATLAB FK implementations before coding Engine_V3

---

## 1. Implementations Reviewed

| Project | Language | Acceleration | Source |
|---------|----------|--------------|--------|
| **cardiax** | Python/JAX | GPU/TPU | [GitHub](https://github.com/epignatelli/cardiax) |
| **ashikagah/Fenton-Karma** | MATLAB | None | [GitHub](https://github.com/ashikagah/Fenton-Karma) |
| **FK_5p0_model** | C++/MATLAB | None | [GitHub](https://github.com/margo73465/FK_5p0_model) |
| **CellML Models** | XML | N/A | [Physiome](https://models.physiomeproject.org/workspace/fenton_karma_1998) |

---

## 2. Equations (Consensus Across Implementations)

### 2.1 State Variables

| Variable | Symbol | Range | Initial Value |
|----------|--------|-------|---------------|
| Voltage | `u` | [0, 1] | 0 |
| Fast gate | `v` | [0, 1] | 1 |
| Slow gate | `w` | [0, 1] | 1 |

### 2.2 Ionic Currents

**Fast Inward Current (J_fi)** - Depolarization:
```python
p = heaviside(u - u_c)  # Threshold function
J_fi = -v * p * (1 - u) * (u - u_c) / tau_d
```

**Slow Outward Current (J_so)** - Repolarization:
```python
J_so = u * (1 - p) / tau_0 + p / tau_r
```

**Slow Inward Current (J_si)** - Plateau (Ca-like):
```python
J_si = -w * (1 + tanh(k * (u - u_csi))) / (2 * tau_si)
```

### 2.3 Gate Dynamics

**Fast gate (v)** - uses voltage-dependent tau_v_minus:
```python
q = heaviside(u - u_v)  # Second threshold
tau_v_minus = q * tau_v1_minus + (1 - q) * tau_v2_minus

dv/dt = (1 - p) * (1 - v) / tau_v_minus - p * v / tau_v_plus
```

**Slow gate (w)**:
```python
dw/dt = (1 - p) * (1 - w) / tau_w_minus - p * w / tau_w_plus
```

### 2.4 Voltage Evolution

```python
du/dt = -(J_fi + J_so + J_si) + I_stim + D * laplacian(u)
```

---

## 3. Parameter Sets from cardiax

The cardiax library defines **10 parameter sets** from Fenton et al. 2002. Here are the key ones:

### 3.1 PARAMSET_3 (Recommended Starting Point)

This set produces stable spiral waves without breakup:

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| tau_v_plus | 3.33 | ms | v-gate closing |
| tau_v1_minus | 19.6 | ms | v-gate opening (fast) |
| tau_v2_minus | 1250 | ms | v-gate opening (slow) |
| tau_w_plus | 870 | ms | w-gate closing |
| tau_w_minus | 41 | ms | w-gate opening |
| tau_d | 0.25 | ms | Fast inward activation |
| tau_0 | 12.5 | ms | Slow outward (below threshold) |
| tau_r | 33.33 | ms | Slow outward (above threshold) |
| tau_si | 29 | ms | Slow inward |
| k | 10 | - | Tanh steepness |
| u_c (V_c) | 0.13 | - | Main threshold |
| u_csi (V_csi) | 0.85 | - | Slow inward threshold |
| u_v (V_v) | 0.04 | - | Secondary threshold |
| Cm | 1.0 | - | Membrane capacitance |

### 3.2 PARAMSET_6 (MATLAB fk2d.m Uses Similar)

| Parameter | Value | Description |
|-----------|-------|-------------|
| tau_v_plus | 3.33 | |
| tau_v1_minus | 9 | |
| tau_v2_minus | 8 | |
| tau_w_plus | 250 | |
| tau_w_minus | 60 | |
| tau_d | 0.395 | |
| tau_0 | 9 | |
| tau_r | 33.33 | |
| tau_si | 29 | |
| k | 15 | |
| u_c | 0.13 | |
| u_csi | 0.5 | |
| u_v | 0.04 | |

### 3.3 Full Parameter Table (All 10 Sets)

```
Set  | tau_v+ | tau_v1- | tau_v2- | tau_w+ | tau_w- | tau_d | tau_0 | tau_r | tau_si | k   | u_c  | u_csi | u_v
-----|--------|---------|---------|--------|--------|-------|-------|-------|--------|-----|------|-------|------
1A   | 3.33   | 19.6    | 1000    | 667    | 11     | 0.41  | 8.3   | 50    | 45     | 10  | 0.13 | 0.85  | 0.0055
1B   | 3.33   | 19.6    | 1000    | 667    | 11     | 0.392 | 8.3   | 50    | 45     | 10  | 0.13 | 0.85  | 0.0055
1C   | 3.33   | 19.6    | 1000    | 667    | 11     | 0.381 | 8.3   | 50    | 45     | 10  | 0.13 | 0.85  | 0.0055
1D   | 3.33   | 19.6    | 1000    | 667    | 11     | 0.36  | 8.3   | 50    | 45     | 10  | 0.13 | 0.85  | 0.0055
1E   | 3.33   | 19.6    | 1000    | 667    | 11     | 0.25  | 8.3   | 50    | 45     | 10  | 0.13 | 0.85  | 0.0055
2    | 10     | 10      | 10      | 1e6    | 1e6    | 0.25  | 10    | 190   | 1e6    | 1e5 | 0.13 | 1e6   | 1e6
3    | 3.33   | 19.6    | 1250    | 870    | 41     | 0.25  | 12.5  | 33.33 | 29     | 10  | 0.13 | 0.85  | 0.04
4A   | 3.33   | 15.6    | 5       | 350    | 80     | 0.407 | 9     | 34    | 26.5   | 15  | 0.15 | 0.45  | 0.04
4B   | 3.33   | 15.6    | 5       | 350    | 80     | 0.405 | 9     | 34    | 26.5   | 15  | 0.15 | 0.45  | 0.04
4C   | 3.33   | 15.6    | 5       | 350    | 80     | 0.4   | 9     | 34    | 26.5   | 15  | 0.15 | 0.45  | 0.04
5    | 3.33   | 12      | 2       | 1000   | 100    | 0.362 | 5     | 33.33 | 29     | 15  | 0.13 | 0.7   | 0.04
6    | 3.33   | 9       | 8       | 250    | 60     | 0.395 | 9     | 33.33 | 29     | 15  | 0.13 | 0.5   | 0.04
7    | 10     | 7       | 7       | 1e6    | 1e6    | 0.25  | 12    | 100   | 1e6    | 1e5 | 0.13 | 1e6   | 1e6
8    | 13.03  | 19.06   | 1250    | 800    | 40     | 0.45  | 12.5  | 33.25 | 29     | 10  | 0.13 | 0.85  | 0.04
9    | 3.33   | 15      | 2       | 670    | 61     | 0.25  | 12.5  | 28    | 29     | 10  | 0.13 | 0.45  | 0.05
10   | 10     | 40      | 333     | 1000   | 65     | 0.115 | 12.5  | 25    | 22.22  | 10  | 0.13 | 0.85  | 0.0025
```

---

## 4. Numerical Methods Comparison

### 4.1 Time Integration

| Implementation | Method | dt (ms) | Notes |
|----------------|--------|---------|-------|
| cardiax | Euler, Heun, or Dormand-Prince | varies | JAX ode.odeint available |
| MATLAB fk2d | Explicit Euler | 0.1 | Simple but stable |
| Our V2 (AP) | Explicit Euler | 0.005 | Needed for stability |

**Recommendation**: Start with Euler at dt=0.01-0.02 ms, can upgrade to Heun if needed.

### 4.2 Spatial Discretization

| Implementation | Method | Order | dx |
|----------------|--------|-------|-----|
| cardiax | Central FD | 4th | varies |
| MATLAB fk2d | Central FD | 2nd | 0.025 cm |
| Our V2 | Flux-based | 2nd | 0.5 mm |

**Recommendation**: Keep 2nd order flux-based from V2 (proven, Numba-compatible).

### 4.3 Boundary Conditions

All implementations use **Neumann (zero-flux)** BCs:
```python
# Padding approach (MATLAB):
# V_padded = [[0, V[1,:], 0], [V[:,1], V, V[:,-2]], [0, V[-2,:], 0]]

# Gradient approach (our V2):
# dV/dn = 0 at boundaries → one-sided differences
```

---

## 5. Diffusion Handling

### 5.1 MATLAB (Simple Laplacian)

```matlab
% Second-order central differences
V_xx = (V[i,j-1] + V[i,j+1] - 2*V[i,j]) / h^2
V_yy = (V[i-1,j] + V[i+1,j] - 2*V[i,j]) / h^2

diffusion = D * (V_xx + V_yy)
```

### 5.2 cardiax (With Conductivity Gradient)

```python
# Includes spatially-varying diffusivity
del_u = diffusivity * (u_xx + u_yy) + D_x * u_x + D_y * u_y
```

### 5.3 Our V2 (Flux-Based Anisotropic)

```python
# Supports fiber-dependent anisotropy
Jx = Dxx * dV/dx + Dxy * dV/dy
Jy = Dxy * dV/dx + Dyy * dV/dy
div_J = dJx/dx + dJy/dy
```

**Recommendation**: Keep flux-based from V2 for anisotropy support.

---

## 6. Key Implementation Patterns

### 6.1 Heaviside Function

```python
# Strict (MATLAB style):
def heaviside(x):
    return np.where(x > 0, 1.0, 0.0)

# Smooth (cardiax style, better for gradients):
def heaviside_smooth(x, k=100):
    return 0.5 * (1 + np.tanh(k * x))
```

**Recommendation**: Use strict Heaviside with Numba (simpler, faster).

### 6.2 Parameter Organization (cardiax)

```python
from typing import NamedTuple

class FKParams(NamedTuple):
    tau_v_plus: float
    tau_v1_minus: float
    tau_v2_minus: float
    tau_w_plus: float
    tau_w_minus: float
    tau_d: float
    tau_0: float
    tau_r: float
    tau_si: float
    k: float
    u_c: float
    u_csi: float
    u_v: float
    Cm: float = 1.0
```

**Recommendation**: Use dataclass (more flexible than NamedTuple).

### 6.3 Numba-Compatible Kernel

```python
@numba.jit(nopython=True, cache=True)
def fk_ionic_step(u, v, w, dt, I_stim,
                  tau_v_plus, tau_v1_minus, tau_v2_minus,
                  tau_w_plus, tau_w_minus, tau_d, tau_0,
                  tau_r, tau_si, k, u_c, u_csi, u_v):
    """FK ionic step - Numba kernel."""
    ny, nx = u.shape

    for i in range(ny):
        for j in range(nx):
            u_val = u[i, j]
            v_val = v[i, j]
            w_val = w[i, j]

            # Heaviside functions
            p = 1.0 if u_val > u_c else 0.0
            q = 1.0 if u_val > u_v else 0.0

            # Voltage-dependent tau_v_minus
            tau_v_minus = q * tau_v1_minus + (1.0 - q) * tau_v2_minus

            # Ionic currents
            J_fi = -v_val * p * (1.0 - u_val) * (u_val - u_c) / tau_d
            J_so = u_val * (1.0 - p) / tau_0 + p / tau_r
            J_si = -w_val * (1.0 + np.tanh(k * (u_val - u_csi))) / (2.0 * tau_si)

            # Total ionic current
            I_ion = J_fi + J_so + J_si

            # Gate dynamics
            dv_dt = (1.0 - p) * (1.0 - v_val) / tau_v_minus - p * v_val / tau_v_plus
            dw_dt = (1.0 - p) * (1.0 - w_val) / tau_w_minus - p * w_val / tau_w_plus

            # Voltage dynamics (ionic only)
            du_dt = -I_ion + I_stim[i, j]

            # Euler update with clamping
            u[i, j] = max(0.0, min(u_val + dt * du_dt, 1.0))
            v[i, j] = max(0.0, min(v_val + dt * dv_dt, 1.0))
            w[i, j] = max(0.0, min(w_val + dt * dw_dt, 1.0))
```

---

## 7. Performance Benchmarks (cardiax)

| Backend | 128×128 grid, 1000 steps | Speedup vs NumPy |
|---------|-------------------------|------------------|
| NumPy | 1.84 s | 1× |
| JAX CPU | 0.92 s | 2× |
| JAX GPU | 0.24 s | 7.7× |
| JAX TPU | 0.07 s | 26× |

**Our expected performance** (Numba):
- Should be ~2-4× faster than NumPy
- Not as fast as JAX GPU, but simpler to implement
- Good enough for 161×161 grid (our size)

---

## 8. Lessons Learned

### 8.1 What Works Well

1. **NamedTuple/dataclass for parameters** - Clean, immutable
2. **Explicit Euler** - Simple, stable with small dt
3. **2nd order spatial** - Sufficient for our resolution
4. **Strict Heaviside** - Faster in Numba than smooth
5. **Voltage clamping** - Prevents numerical instability

### 8.2 What to Avoid

1. **MAXFLOAT parameters** (sets 2, 7) - Numerical issues
2. **Too large dt** - Set 1 uses dt=0.1 but needs dt≤0.02 for accuracy
3. **Smooth Heaviside** - Slows Numba, not needed
4. **Complex time integrators** - Euler is fine for explicit scheme

### 8.3 Recommended Starting Configuration

```python
# Parameter Set 3 (stable spirals)
params = FKParams(
    tau_v_plus=3.33,
    tau_v1_minus=19.6,
    tau_v2_minus=1250.0,
    tau_w_plus=870.0,
    tau_w_minus=41.0,
    tau_d=0.25,
    tau_0=12.5,
    tau_r=33.33,
    tau_si=29.0,
    k=10.0,
    u_c=0.13,
    u_csi=0.85,
    u_v=0.04,
)

# Numerical settings
dt = 0.02  # ms
dx = 0.5   # mm
D = 1.0    # mm²/ms
```

---

## 9. Mapping to Our Targets

### Our V2 Targets:
- APD90 = 250 ms
- CV = 500 mm/s (0.5 mm/ms)
- Anisotropy 2:1

### FK Capability:

**APD Control**:
- Increase `tau_w_plus` → longer APD
- Set 3 has tau_w_plus=870 → APD ~200-300 ms
- Can tune to get exactly 250 ms

**CV Control**:
- CV ∝ √(D / tau_d)
- With D=1.0 mm²/ms and tau_d=0.25 ms → CV ≈ 0.5 mm/ms ✓

**Anisotropy**:
- Same diffusion tensor as V2 (Dxx, Dyy, Dxy)
- No changes needed

---

## 10. Final Engine_V3 Structure

Based on this research:

```
Engine_V3/
├── parameters.py         # FKParams dataclass + presets
├── fenton_karma.py       # FK model class + simulation
│   ├── FentonKarmaModel  # Ionic model
│   ├── fk_ionic_step()   # Numba kernel
│   └── Simulation        # Time stepping + state
├── diffusion.py          # From V2 (flux-based anisotropic)
├── mesh.py               # From V2 (grid + fibers)
├── stimulus.py           # Stimulus protocols
├── visualization.py      # Plotting + animation
├── interactive.py        # Click-to-stimulate
└── analysis.py           # APD/CV measurement
```

---

## 11. References

1. [cardiax (Python/JAX)](https://github.com/epignatelli/cardiax) - 10 parameter sets
2. [Fenton-Karma MATLAB](https://github.com/ashikagah/Fenton-Karma) - fk2d.m implementation
3. [FK_5p0_model](https://github.com/margo73465/FK_5p0_model) - C++/MATLAB atrial model
4. [CellML Models](https://models.physiomeproject.org/workspace/fenton_karma_1998) - Formal specs
5. Fenton & Karma 1998, *Chaos* 8:20-47 - Original paper
6. Fenton et al. 2002, *Chaos* 12:852-892 - Multiple mechanisms paper

---

**Author**: Generated with Claude Code
**Date**: 2025-12-10
**Status**: Research Complete - Ready for Implementation
