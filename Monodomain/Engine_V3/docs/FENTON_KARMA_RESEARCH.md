# Fenton-Karma Model Research for Engine_V3

**Date**: 2025-12-10
**Purpose**: Evaluate Fenton-Karma (FK) model for upgrading from Aliev-Panfilov in cardiac simulation
**Status**: Research Phase

---

## 1. Model Overview

The **Fenton-Karma (FK) model** is a 3-variable simplified ionic model of cardiac action potential, introduced in:

> Fenton F, Karma A. "Vortex dynamics in three-dimensional continuous myocardium with fiber rotation: Filament instability and fibrillation." *Chaos* 8(1), 20-47 (1998).

### Why FK over Aliev-Panfilov?

| Feature | Aliev-Panfilov | Fenton-Karma |
|---------|----------------|--------------|
| Variables | 2 (V, w) | 3 (u, v, w) |
| Currents | 1 phenomenological | 3 physiological-like |
| APD Restitution | Poor | Good |
| CV Restitution | Poor | Good |
| Spike-and-dome | No | Partial |
| Calcium-like current | No | Yes (Isi) |
| Computational cost | Very low | Low-moderate |

**Key advantage**: FK reproduces **restitution properties** (APD and CV dependence on diastolic interval), which is critical for:
- Accurate spiral wave dynamics
- Alternans behavior
- Realistic wave breakup

---

## 2. State Variables

The FK model uses **dimensionless** variables:

| Variable | Physical Interpretation | Range |
|----------|------------------------|-------|
| `u` | Transmembrane voltage (normalized) | [0, 1] |
| `v` | Fast inward current inactivation gate | [0, 1] |
| `w` | Slow inward current inactivation gate | [0, 1] |

### Voltage Normalization

Same as Aliev-Panfilov:
```
u = (V_physical - V_rest) / (V_peak - V_rest)

Where:
  V_rest = -85 mV (resting potential)
  V_peak = +40 mV (peak depolarization)
  V_range = 125 mV
```

---

## 3. Model Equations

### 3.1 Membrane Voltage Equation

```
du/dt = -J_fi - J_so - J_si + I_stim + D∇²u
```

Where:
- `J_fi` = Fast inward current (Na-like depolarization)
- `J_so` = Slow outward current (K-like repolarization)
- `J_si` = Slow inward current (Ca-like plateau)
- `I_stim` = External stimulus current
- `D∇²u` = Diffusion term (spatial coupling)

### 3.2 Ionic Currents

**Fast Inward Current (J_fi)** - Rapid depolarization:
```
J_fi = -v * Θ(u - u_c) * (1 - u) * (u - u_c) / τ_d
```

**Slow Outward Current (J_so)** - Repolarization:
```
J_so = u * Θ(u_c - u) / τ_o + Θ(u - u_c) / τ_r
```

**Slow Inward Current (J_si)** - Plateau phase (calcium-like):
```
J_si = -w * (1 + tanh[k(u - u_csi)]) / (2 * τ_si)
```

### 3.3 Gate Variables

**Fast gate (v)** - Controls J_fi:
```
dv/dt = Θ(u_c - u) * (1 - v) / τ_v^- - Θ(u - u_c) * v / τ_v^+
```

**Slow gate (w)** - Controls J_si:
```
dw/dt = Θ(u_c - u) * (1 - w) / τ_w^- - Θ(u - u_c) * w / τ_w^+
```

### 3.4 Heaviside Step Function

```
Θ(x) = { 1  if x > 0
       { 0  if x ≤ 0
```

For numerical implementation, can use smooth approximation:
```python
def heaviside(x):
    return 0.5 * (1.0 + np.tanh(100.0 * x))  # Smooth approximation
    # Or strict: return np.where(x > 0, 1.0, 0.0)
```

---

## 4. Parameter Sets

### 4.1 Original Parameters (from ibiblio.org)

| Parameter | Symbol | Value | Units | Description |
|-----------|--------|-------|-------|-------------|
| τ_d | tau_d | 1/5.8 ≈ 0.172 | ms | Fast inward time constant |
| τ_r | tau_r | 130 | ms | Repolarization time constant |
| τ_si | tau_si | 127 | ms | Slow inward time constant |
| τ_o | tau_o | 12.5 | ms | Outward current time constant |
| τ_v^+ | tau_v_plus | 10 | ms | v-gate closing time |
| τ_v^- | tau_v_minus | 18.2 | ms | v-gate opening time |
| τ_w^+ | tau_w_plus | 1020 | ms | w-gate closing time |
| τ_w^- | tau_w_minus | 80 | ms | w-gate opening time |
| u_c | u_c | 0.13 | - | Voltage threshold |
| u_csi | u_csi | 0.85 | - | Slow inward threshold |
| k | k_tanh | 10 | - | Tanh steepness |

### 4.2 BR (Beeler-Reuter) Variant Parameters

From MATLAB implementation (GitHub: ashikagah/Fenton-Karma):

| Parameter | Value | Notes |
|-----------|-------|-------|
| tau_d | 0.395 ms | Faster upstroke |
| tau_si | 29 ms | Shorter plateau |
| Vc (u_c) | 0.13 | Threshold |
| Vv | 0.04 | Additional threshold |
| D | 0.001 cm²/ms | Diffusion coefficient |
| dx | 0.025 cm | Grid spacing |
| dt | 0.1 ms | Time step |

### 4.3 Parameter Effects on Model Behavior

| Property | Primary Parameters |
|----------|-------------------|
| Maximum upstroke velocity | τ_d, τ_v^+, Θ_v |
| AP amplitude | u_u, u_o |
| Maximum APD | τ_w^+, τ_si |
| Minimum APD | τ_so |
| Minimum DI | τ_v^- |
| CV restitution | τ_d, τ_v^- |

---

## 5. Diffusion and Conduction Velocity

### 5.1 Monodomain Equation

In 2D with anisotropic diffusion:
```
∂u/∂t = I_ion(u, v, w) + ∇·(D∇u)

Where D is the diffusion tensor:
D = R @ diag(D_parallel, D_perp) @ R^T

R = rotation matrix based on fiber angle θ
```

### 5.2 CV-Diffusion Relationship

For reaction-diffusion systems:
```
CV ∝ √D

Specifically for cardiac tissue:
CV ≈ C * √(D / τ_rise)

Where:
  C ≈ constant depending on model
  D = diffusion coefficient
  τ_rise = upstroke time constant
```

### 5.3 Converting Units

```
Diffusion: 1 mm²/ms = 0.01 cm²/ms = 10 cm²/s
Velocity: 1 mm/ms = 1 m/s = 100 cm/s
```

### 5.4 Typical Values

| Tissue | D (cm²/ms) | D (mm²/ms) | CV (cm/s) | CV (mm/ms) |
|--------|-----------|-----------|----------|-----------|
| Longitudinal | 0.001 | 0.1 | 50-70 | 0.5-0.7 |
| Transverse | 0.0002-0.0005 | 0.02-0.05 | 17-30 | 0.17-0.30 |

---

## 6. Comparison to Our Target Parameters

### Our Current Targets (from parameters.py):

| Parameter | Target Value | Units |
|-----------|-------------|-------|
| V_rest | -85 | mV |
| V_peak | +40 | mV |
| APD90 | 250 | ms |
| CV_longitudinal | 500 | mm/s |
| CV_transverse | 250 | mm/s |
| D_parallel | 1.0 | mm²/ms |
| D_perp | 0.5 | mm²/ms |
| Anisotropy ratio | 2:1 | - |

### FK Model Capability Assessment

| Target | FK Capable? | Notes |
|--------|-------------|-------|
| APD90 = 250ms | **YES** | Tunable via τ_w^+, τ_si, τ_so |
| CV = 500 mm/s | **YES** | Via diffusion coefficient D |
| Anisotropy 2:1 | **YES** | Via D tensor |
| Restitution | **YES** | Core FK strength |
| Alternans | **YES** | Can emerge naturally |
| Calcium dynamics | **PARTIAL** | J_si mimics ICa |
| Real ion channels | **NO** | Phenomenological currents |

### Parameter Tuning Strategy

To achieve APD90 ≈ 250ms:
```python
# Plateau duration dominated by τ_w^+ and τ_si
# Increase these for longer APD:
tau_w_plus = 800-1200 ms  # Main APD control
tau_si = 100-150 ms       # Plateau shape
tau_o = 10-15 ms          # Early repolarization
```

To achieve CV = 500 mm/s = 0.5 mm/ms:
```python
# CV ≈ sqrt(D / tau_d) * constant
# For D = 1.0 mm²/ms and target CV = 0.5 mm/ms:
# Need to calibrate with actual simulation

D_parallel = 1.0  # mm²/ms (keep)
tau_d = 0.17-0.4  # ms (tune)
```

---

## 7. Numerical Implementation

### 7.1 Discretization

**Spatial** (2D):
```python
# Central differences for diffusion (interior points)
d2u_dx2 = (u[i+1,j] - 2*u[i,j] + u[i-1,j]) / dx²
d2u_dy2 = (u[i,j+1] - 2*u[i,j] + u[i,j-1]) / dy²

# Anisotropic with cross-derivatives
d2u_dxdy = (u[i+1,j+1] - u[i+1,j-1] - u[i-1,j+1] + u[i-1,j-1]) / (4*dx*dy)
```

**Temporal**:
- Explicit Euler (simplest, needs small dt)
- Rush-Larsen (better for gates)
- RK2/RK4 (more accurate)

### 7.2 Stability Constraints

For explicit Euler with 2D diffusion:
```
dt < dx² / (4 * D_max)  # CFL condition

Example:
dx = 0.5 mm, D_max = 1.0 mm²/ms
dt < 0.25² / 4 = 0.0156 ms

Use dt = 0.01 ms for safety
```

### 7.3 Operator Splitting (Godunov)

Same as current V2 engine:
```
1. Diffusion half-step: u* = u + (dt/2) * D∇²u
2. Full reaction step: u**, v', w' = reaction(u*, v, w, dt)
3. Diffusion half-step: u_new = u** + (dt/2) * D∇²u**
```

Or simpler first-order splitting:
```
1. Diffusion: u* = u + dt * D∇²u
2. Reaction: u_new, v_new, w_new = reaction(u*, v, w, dt)
```

### 7.4 Numba Implementation Skeleton

```python
@numba.jit(nopython=True, parallel=False, cache=True)
def fk_ionic_step(u, v, w, dt, I_stim, params):
    """Fenton-Karma ionic model step."""
    ny, nx = u.shape

    # Unpack parameters
    tau_d = params['tau_d']
    tau_r = params['tau_r']
    tau_si = params['tau_si']
    tau_o = params['tau_o']
    tau_v_plus = params['tau_v_plus']
    tau_v_minus = params['tau_v_minus']
    tau_w_plus = params['tau_w_plus']
    tau_w_minus = params['tau_w_minus']
    u_c = params['u_c']
    u_csi = params['u_csi']
    k = params['k']

    for i in range(ny):
        for j in range(nx):
            u_val = u[i, j]
            v_val = v[i, j]
            w_val = w[i, j]
            stim = I_stim[i, j]

            # Heaviside functions
            H_uc = 1.0 if u_val > u_c else 0.0
            H_uc_inv = 1.0 - H_uc

            # Ionic currents
            J_fi = -v_val * H_uc * (1.0 - u_val) * (u_val - u_c) / tau_d
            J_so = u_val * H_uc_inv / tau_o + H_uc / tau_r
            J_si = -w_val * (1.0 + np.tanh(k * (u_val - u_csi))) / (2.0 * tau_si)

            # Total ionic current
            I_ion = J_fi + J_so + J_si

            # Gate dynamics
            dv_dt = H_uc_inv * (1.0 - v_val) / tau_v_minus - H_uc * v_val / tau_v_plus
            dw_dt = H_uc_inv * (1.0 - w_val) / tau_w_minus - H_uc * w_val / tau_w_plus

            # Voltage update (ionic only, diffusion separate)
            du_dt = -I_ion + stim

            # Euler update
            u[i, j] = max(0.0, min(u_val + dt * du_dt, 1.0))
            v[i, j] = max(0.0, min(v_val + dt * dv_dt, 1.0))
            w[i, j] = max(0.0, min(w_val + dt * dw_dt, 1.0))
```

---

## 8. Engine_V3 Architecture Plan

### 8.1 File Structure

```
Engine_V3/
├── fenton_karma.py          # FK ionic model class
├── parameters_v3.py         # Parameter management (FK params)
├── simulate_fk.py           # Main simulation with diffusion
├── interactive_fk.py        # Click-to-stimulate interface
├── mesh_2d.py              # 2D mesh utilities (from V2)
├── diffusion.py            # Anisotropic diffusion (from V2)
├── visualization.py        # Plotting and animation
└── tests/
    ├── test_single_cell.py  # Single-cell AP validation
    ├── test_cv.py           # Conduction velocity test
    └── test_restitution.py  # APD restitution curve
```

### 8.2 Key Changes from V2

1. **Ionic Model**: 2 variables → 3 variables
2. **Currents**: 1 phenomenological → 3 semi-physiological
3. **Gates**: Heaviside-based switching
4. **Parameters**: Different set (tau values)
5. **Validation**: Restitution curves required

### 8.3 Migration Path

| V2 Component | V3 Equivalent | Changes |
|--------------|---------------|---------|
| aliev_panfilov_fixed.py | fenton_karma.py | Complete rewrite |
| parameters.py | parameters_v3.py | New FK parameters |
| simulate_infarct_v2.py | simulate_fk.py | Update ionic step |
| interactive_simulation.py | interactive_fk.py | Update ionic step |
| diffusion (compute_diffusion_flux_based) | Keep | No change |
| mesh_builder.py | Keep | No change |

---

## 9. Validation Plan

### 9.1 Single Cell Tests

1. **AP Shape**: Compare to BR or LR1 model
2. **APD90**: Should be ~250ms for target parameters
3. **Upstroke velocity**: dV/dt_max ~ 100-400 V/s

### 9.2 1D Cable Tests

1. **CV measurement**: Should match target (500 mm/s longitudinal)
2. **CV restitution**: CV vs DI curve

### 9.3 2D Tissue Tests

1. **Planar wave**: Measure CV in both directions
2. **Spiral wave**: Initiation and stability
3. **Reentry**: Around obstacle

---

## 10. References

### Primary Sources

1. [Fenton-Karma 1998 Original Paper (PubMed)](https://pubmed.ncbi.nlm.nih.gov/12779708/)
2. [CellML Model Repository](https://models.cellml.org/exposure/cfe1ed59d236f5e57c2ccd7a375a6268)
3. [ibiblio FK Model](https://www.ibiblio.org/e-notes/html5/fk.html)

### Implementations

4. [GitHub: ashikagah/Fenton-Karma (MATLAB)](https://github.com/ashikagah/Fenton-Karma)
5. [GitHub: epignatelli/cardiax (Python/JAX)](https://github.com/epignatelli/cardiax)
6. [GitHub: annien094/2D-3D-EP-PINNs](https://github.com/annien094/2D-3D-EP-PINNs)

### Parameter Tuning

7. [Reproducing Cardiac Restitution Properties (Springer)](https://link.springer.com/article/10.1007/s10439-005-3948-3)
8. [ResearchGate: FK Parameter Table](https://www.researchgate.net/figure/Parameters-for-the-RA-model-with-the-Fenton-Karma-model-of-atrial-tissue_tbl1_235690931)

---

## 11. Conclusion

### Can FK Model Meet Our Targets?

| Target | Assessment |
|--------|------------|
| APD90 = 250ms | **YES** - Tunable |
| CV = 500 mm/s | **YES** - Via D coefficient |
| Anisotropy 2:1 | **YES** - Via D tensor |
| Voltage range [-85, +40] mV | **YES** - Same normalization |
| Stable numerical behavior | **YES** - With proper dt |
| Restitution (bonus) | **YES** - FK strength |
| Computational speed | **YES** - Still fast with Numba |

### Recommendation

**Proceed with Engine_V3 implementation** using the Fenton-Karma model.

Priority order:
1. Implement FK ionic model class
2. Create single-cell validation
3. Integrate with existing diffusion code
4. Tune parameters for targets
5. Create interactive simulation
6. Document and validate

---

**Author**: Generated with Claude Code
**Date**: 2025-12-10
**Status**: Research Complete - Ready for Implementation
