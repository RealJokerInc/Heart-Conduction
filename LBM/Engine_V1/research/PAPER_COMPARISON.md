# LBM-EP Paper Comparison: Rapaka et al. (2012) vs Campos et al. (2016)

## Overview

| Attribute | Rapaka et al. (MICCAI 2012) | Campos et al. (JCAM 2016) |
|-----------|---------------------------|--------------------------|
| Affiliation | Siemens Research, Princeton | UFJF, Brazil |
| Lattice | D3Q7 | D3Q7 |
| Collision | BGK (isotropic) + MRT (anisotropic) | SRT → MRT (anisotropic) |
| Cell models | Mitchell-Schaeffer (2 vars) | LRI (8 vars), TT2 (19 vars), MSH (26 vars) |
| ODE solver | Forward Euler (implied) | Rush-Larsen (explicit) |
| Platform | CPU (Fortran, unoptimized) | CPU (C++) + GPU (CUDA) |
| Best speedup | 45x vs FEM (CPU serial) | 500x vs CPU (GPU), 419 MLUPS |
| Source code | Not released | Not released (MonoAlg3D_C uses FVM, not LBM) |
| Boundary | Level-set interpolated bounce-back | Simple bounce-back (Eq. 17) |
| Geometry | Patient-specific (MRI segmentation) | Regular + rabbit biventricular |

---

## 1. How Each Author Encodes Anisotropy

### Rapaka et al. — MRT via Full Diffusion Tensor

Rapaka uses the MRT collision operator `A = M^{-1} S M` where:

**Transformation matrix M (7x7 for D3Q7):**
```
M = [1   1   1   1   1   1   1]   ← row 0: conserved (V = Σf_i)
    [1  -1   0   0   0   0   0]   ← row 1: x-gradient
    [0   0   1  -1   0   0   0]   ← row 2: y-gradient
    [0   0   0   0   1  -1   0]   ← row 3: z-gradient
    [1   1   1   1   1   1  -6]   ← row 4: higher moment
    [-1 -1   0   0   0   1   1]   ← row 5: higher moment
    [1   1  -2  -2   0   1   1]   ← row 6: higher moment (typo-corrected)
```

**Note:** The paper's M matrix has a known formatting issue due to the LNCS template. The rows above are reconstructed from Yoshida & Nagaoka (2010), which is the cited source.

**Relaxation matrix S^{-1}:**
```
S^{-1} = diag(τ_0, [τ_ij]_{3x3}, τ_4, τ_5, τ_6)
```

Where `[τ_ij]` is a **full 3x3 sub-matrix** in rows/columns 1-3:
```
τ_ij = δ_ij/2 + 4·D_ij·dt/dx²
```

This is the key: the relaxation sub-matrix for flux moments 1-3 is **not diagonal** — it encodes the full anisotropic diffusion tensor D including off-diagonal terms (D_xy, D_xz, D_yz). This means fiber rotation at arbitrary angles is handled directly.

**Dummy moments (rows 4-6):** τ_0 = 1, τ_4 = τ_5 = τ_6 = 1.33. These do not affect the diffusion solution, only stability.

### Campos et al. — MRT via Gram-Schmidt Moments

Campos uses the same `Ω_NR = -M^{-1} S M (f - f^eq)` framework but constructs M differently:

**Transformation matrix M (7x7 for D3Q7):**
```
M = [1   1   1   1   1   1   1]   ← row 0: conserved (V)
    [0   1  -1   0   0   0   0]   ← row 1: x-flux
    [0   0   0   1  -1   0   0]   ← row 2: y-flux
    [0   0   0   0   0   1  -1]   ← row 3: z-flux
    [6  -1  -1  -1  -1  -1  -1]   ← row 4: Gram-Schmidt moment
    [0   2   2  -1  -1  -1  -1]   ← row 5: Gram-Schmidt moment
    [0   0   0   1   1  -1  -1]   ← row 6: Gram-Schmidt moment
```

**Key difference:** Campos's M has different higher-moment rows (4-6) obtained via Gram-Schmidt orthogonalization. Rows 0-3 serve the same physical role (conserved quantity + 3 flux components).

**Relaxation matrix S^{-1}:**
```
S^{-1} = [τ_0    0      0      0     0    0    0]
         [0    τ̄_xx  τ̄_xy  τ̄_xz   0    0    0]
         [0    τ̄_yx  τ̄_yy  τ̄_yz   0    0    0]
         [0    τ̄_zx  τ̄_zy  τ̄_zz   0    0    0]
         [0      0      0      0   τ_4    0    0]
         [0      0      0      0     0  τ_5    0]
         [0      0      0      0     0    0  τ_6]
```

With:
```
τ̄_ij = δ_ij/2 + 4·σ_ij·dt/dx²
```

This is **identical physics** to Rapaka's encoding. The conductivity tensor σ_ij (from the fiber model σ = σ_t·I + (σ_l - σ_t)·a·a^T) maps directly into the 3x3 sub-block of S^{-1}.

---

## 2. What is the Moment Space?

The moment space is a transformed representation of the distribution functions where each component has a clear physical interpretation:

| Moment | Physical meaning | Relaxation |
|--------|-----------------|------------|
| m_0 = Σf_i | Conserved quantity (voltage V) | τ_0 (no relaxation, or τ_0 = 1) |
| m_1 = f_+x - f_-x | x-component of flux (∝ ∂V/∂x) | τ_xx (from D_xx) |
| m_2 = f_+y - f_-y | y-component of flux (∝ ∂V/∂y) | τ_yy (from D_yy) |
| m_3 = f_+z - f_-z | z-component of flux (∝ ∂V/∂z) | τ_zz (from D_zz) |
| m_4, m_5, m_6 | Higher-order (non-physical) | Free parameters for stability |

**Why moment space?** In distribution space (f_i), all directions relax at the same rate (BGK). In moment space, we can relax each physical quantity independently — crucially, we can relax the x-flux and y-flux at different rates, giving us anisotropic diffusion.

The collision in moment space is:
```
m* = m - S · (m - m^eq)
```
where S is diagonal (or block-diagonal for off-diagonal D) in moment space. Then transform back: `f* = M^{-1} · m*`.

**Off-diagonal anisotropy:** When the diffusion tensor has off-diagonal terms (rotated fibers), the flux moments become coupled: the relaxation of m_1 depends on m_2 and vice versa. This is encoded by the off-diagonal entries τ̄_xy in S^{-1}.

---

## 3. The Diffusion Tensor Encoding via τ

Both papers use the same fundamental relationship (from Yoshida & Nagaoka 2010):

```
τ̄_ij = δ_ij/2 + 4·D_ij·dt/dx²     (D3Q7)
```

For D2Q5, this becomes:
```
τ̄_ij = δ_ij/2 + 3·D_ij·dt/dx²     (D2Q5, since c_s² = 1/3)
```

**Example — fiber at 45 degrees:**
```
D_fiber = 0.001, D_cross = 0.00025
θ = 45°

D_xx = D_fiber·cos²θ + D_cross·sin²θ = 0.000625
D_yy = D_fiber·sin²θ + D_cross·cos²θ = 0.000625
D_xy = (D_fiber - D_cross)·cosθ·sinθ  = 0.000375

With dx=0.025, dt=0.01:
τ̄_xx = 0.5 + 3·0.000625·0.01/0.000625 = 0.5 + 0.03 = 0.53
τ̄_yy = 0.53
τ̄_xy = 3·0.000375·0.01/0.000625 = 0.018
```

The S^{-1} matrix (rows 1-2) becomes:
```
[0.53  0.018]
[0.018 0.53 ]
```

Then S (the relaxation rate matrix) = inv(S^{-1}).

**Key insight for D2Q5 vs D2Q9:**
- D2Q5 has only 2 flux moments → can encode a 2x2 diffusion tensor
- But D2Q5 cannot properly encode D_xy off-diagonal terms because its velocity vectors don't span the diagonals
- D2Q9 adds 4 diagonal velocities → additional moments that can naturally encode D_xy
- This is why **D2Q9 is needed for proper anisotropic diffusion with rotated fibers**

---

## 4. What They Did with Dummy τ (Higher Moments)

### Rapaka: τ_0 = 1.0, τ_4 = τ_5 = τ_6 = 1.33

These are explicitly called out as not affecting diffusion, only stability. The choice of 1.33 is empirical — closer to 1.0 gives slightly more damping of higher-order oscillations.

### Campos: τ_0, τ_4, τ_5, τ_6 are free parameters

Campos does not specify exact values in the paper text but follows the same Yoshida & Nagaoka (2010) guidance. The implementation likely uses similar values (1.0 or slightly above).

**Practical guidance:**
- τ_0 (conserved moment): Usually set to 1.0. Some implementations set s_0 = 0 (no relaxation of the conserved quantity), which is equivalent since m_0 = m_0^eq = V always.
- τ_4, τ_5, τ_6: Set to 1.0 for simplicity, or 1.33 for slightly better stability. Must be > 0.5 for stability.
- **TRT magic parameter:** For two-relaxation-time models, setting Λ = (τ+ - 0.5)(τ- - 0.5) = 1/4 gives optimal stability. This can guide the higher-moment τ choices.

---

## 5. Optimization Tricks and Speedup Methods

### Rapaka: Minimal Optimization
- Unoptimized Fortran, single-threaded
- Level-set for complex geometry (avoids meshing overhead)
- Source term embedded in collision (no separate ODE-then-diffusion split)
- Still achieved 8.75-45x vs optimized parallel FEM due to LBM's inherent locality

### Campos: Systematic GPU Optimization

**Trick 1: Swap Algorithm (zero-copy streaming)**
Traditional LBM needs two arrays (A-B pattern) for streaming. Campos uses the "swap" algorithm:
- Reorganize direction indices so opposite(i) = i + (N-1)/2
- Process nodes in strict order, swapping f_i with f_opposite from neighbor
- Eliminates temporary array → **halves memory usage**
- Allows collision + streaming in a single grid traversal

**Trick 2: Structure of Arrays (SoA) memory layout**
```
f[direction * Nx*Ny*Nz + z*Ny*Nx + y*Nx + x]
```
Adjacent threads access adjacent memory addresses → **coalesced GPU memory access** → max bandwidth utilization.

**Trick 3: Sparse/indirect addressing for irregular geometry**
- Adjacency array: `adj[node_k][direction]` → neighbor index
- Only allocate for active (tissue) nodes
- For rabbit heart: uses only **36% of memory** vs full matrix
- Slight overhead from indirection, but massive memory savings

**Trick 4: Fused collision-streaming kernel**
- Single CUDA kernel does: load f_i → compute Iion → collision → stream to neighbor → apply BC
- One grid traversal per timestep
- Minimizes kernel launch overhead and global memory transactions

**Trick 5: Zero CPU-GPU transfer during simulation**
- All computation stays on GPU
- V values stored in GPU buffer at save points
- Single bulk transfer at end of simulation

**Trick 6: Hand-coded MRT (no actual matrix multiply)**
- The M^{-1} S M operation is analytically expanded
- Each m_i and the inverse transform are hand-coded as arithmetic operations
- Two versions: diagonal-only (faster) and full tensor (general)
- Diagonal implementation: ~400 MLUPS; Full tensor: ~320 MLUPS

### Performance Results

| Implementation | MLUPS | Notes |
|---------------|-------|-------|
| Campos CPU (C++) | ~1 | Baseline |
| Campos GPU diagonal (SP) | 400 | Aligned fibers |
| Campos GPU full tensor (SP) | 320 | Arbitrary fibers |
| Campos GPU full tensor (DP) | ~160 | Half of SP |
| Rapaka CPU (Fortran) | ~12 | 80ms/iter for 480K nodes |

---

## 6. What We Can Learn for Our Implementation

### Architecture decisions:
1. **Use D2Q9 for 2D** — D2Q5 cannot encode off-diagonal D_xy, which we need for boundary speedup with rotated fibers and proper Dirichlet BCs
2. **MRT is mandatory** — BGK cannot do anisotropy
3. **Hand-code the MRT** — Don't do actual matrix multiply. Pre-expand M^{-1}·S·M into explicit per-moment operations
4. **Embed source in collision** — Rapaka's approach (Eq. 2) preserves non-equilibrium info vs operator-splitting re-equilibration

### Performance decisions:
1. **SoA layout** — f shape (Q, Nx, Ny) already correct in our V5.4 code
2. **torch.compile** — PyTorch equivalent of hand-coded kernels
3. **Single-buffer streaming** — Swap algorithm if we want to minimize memory
4. **float32 sufficient** — Campos showed SP gives same physics, 2x faster

### Physics decisions for boundary speedup:
1. **D2Q9 lattice** — 9 velocities including diagonals are needed to properly resolve gradients at boundaries
2. **Dirichlet BC** — Anti-bounce-back or fixed-value BC (different from Neumann bounce-back)
3. **Planar wavefront** — Start with uniform stimulus on one edge
4. **Measure CV near boundary vs interior** — The speedup effect is ~10-20% increase in CV within 1-2 space constants of the boundary
