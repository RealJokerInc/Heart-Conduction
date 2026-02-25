# Bidomain Engine V1

## Overview

The bidomain model is the gold standard for cardiac electrophysiology simulation, representing both intracellular and extracellular electrical domains. Unlike the monodomain model (which assumes infinite extracellular conductivity), bidomain captures:

- **Extracellular potential (φ_e)**: Tracks voltage in the extracellular space
- **Transmembrane potential (V_m = φ_i - φ_e)**: Voltage across the cell membrane
- **Ion accumulation effects**: Critical for modeling ischemia and infarct borders

## Mathematical Formulation

### Governing Equations (Vm-φe formulation)

The bidomain equations in their standard form:

**Parabolic PDE (transmembrane dynamics):**
```
χ * C_m * ∂V_m/∂t = ∇·(σ_i ∇V_m) + ∇·(σ_i ∇φ_e) - χ * I_ion(V_m, w, c)
```

**Elliptic PDE (extracellular potential):**
```
∇·((σ_i + σ_e) ∇φ_e) = -∇·(σ_i ∇V_m) - I_e
```

Where:
- `V_m`: Transmembrane potential [mV]
- `φ_e`: Extracellular potential [mV]
- `σ_i`, `σ_e`: Intracellular/extracellular conductivity tensors [mS/cm]
- `χ`: Surface-to-volume ratio [1/cm], typically ~1400 cm⁻¹
- `C_m`: Membrane capacitance [μF/cm²], typically 1.0 μF/cm²
- `I_ion`: Ionic current density [μA/cm²]
- `I_e`: External stimulus current [μA/cm³]

### Conductivity Values (ventricular tissue)

| Parameter | Longitudinal | Transverse | Units |
|-----------|-------------|------------|-------|
| σ_i (intracellular) | 3.0 | 0.31525 | mS/cm |
| σ_e (extracellular) | 2.0 | 1.3514 | mS/cm |

Anisotropy ratio: ~10:1 intracellular, ~1.5:1 extracellular

## Numerical Implementation Strategy

### Operator Splitting (Godunov/Strang)

Split the problem into three sub-problems per timestep:

1. **Reaction step**: Solve ionic model ODEs
   ```
   ∂V_m/∂t = -I_ion(V_m, w, c) / C_m
   ∂w/∂t = f(V_m, w)
   ∂c/∂t = g(V_m, w, c)  # concentration dynamics
   ```

2. **Diffusion step**: Solve parabolic PDE for V_m
   ```
   χ * C_m * ∂V_m/∂t = ∇·(σ_i ∇V_m) + ∇·(σ_i ∇φ_e)
   ```

3. **Elliptic solve**: Solve for φ_e given V_m
   ```
   ∇·((σ_i + σ_e) ∇φ_e) = -∇·(σ_i ∇V_m)
   ```

### Time Discretization

- **Reaction terms**: Forward Euler or Rush-Larsen (for gating variables)
- **Parabolic PDE**: Crank-Nicolson (implicit, unconditionally stable)
- **Elliptic PDE**: Direct solve each timestep

### Spatial Discretization

For 2D finite differences on uniform grid:
- Standard 5-point stencil for isotropic Laplacian
- 9-point stencil for anisotropic diffusion (if fiber rotation present)

### Linear Solver Options

1. **Direct (LU)**: Fast for small problems (<10⁵ nodes)
2. **Iterative (CG+ILU)**: Memory efficient, good for medium problems
3. **Multigrid (AMG)**: Best for large problems, 7-14x faster than ILU

## Ionic Model Selection

### For Ion Accumulation Effects

The bidomain needs an ionic model that tracks concentration dynamics:

1. **Luo-Rudy II (LRd)** - Recommended
   - Tracks [K]o, [Na]i, [Ca]i
   - Models Na/K pump, Na/Ca exchanger
   - Captures ischemic effects (elevated [K]o)
   - ~15 state variables

2. **Ten Tusscher-Panfilov (TTP06)**
   - Human ventricular model
   - Full concentration dynamics
   - ~19 state variables

3. **O'Hara-Rudy (ORd)**
   - State-of-art human ventricular model
   - Detailed Ca handling
   - ~41 state variables

## Implementation Phases

### Phase 1: Basic Bidomain Solver
- [ ] Finite difference discretization (2D)
- [ ] Elliptic solver (scipy.sparse.linalg)
- [ ] Operator splitting framework
- [ ] Fenton-Karma ionic model (as baseline)

### Phase 2: Luo-Rudy Integration
- [ ] Implement LRd ionic model with [K]o dynamics
- [ ] Concentration diffusion in extracellular space
- [ ] Validate single-cell AP against literature

### Phase 3: Ischemia/Infarct Modeling
- [ ] Regional [K]o elevation
- [ ] Border zone effects
- [ ] Verify speedup phenomenon

### Phase 4: Performance Optimization
- [ ] Numba JIT compilation
- [ ] Sparse matrix assembly
- [ ] Preconditioned iterative solver

## File Structure

```
Bidomain/Engine_V1/
├── README.md           # This file
├── parameters.py       # Physical constants and parameters
├── bidomain_solver.py  # Core bidomain PDE solver
├── elliptic_solver.py  # Elliptic equation solver
├── luo_rudy.py        # Luo-Rudy ionic model
├── concentration.py    # Ion concentration dynamics
├── mesh_2d.py         # Mesh utilities
├── visualizer.py      # Plotting/animation
└── examples/
    ├── basic_propagation.py
    ├── ischemia_test.py
    └── border_speedup.py
```

## Key References

- Sundnes et al., "Computing the Electrical Activity in the Heart" (2006)
- Keener & Sneyd, "Mathematical Physiology" Chapter 11
- Luo & Rudy, Circ Res 1994 (LRd model)
- PMC2881536 - "Solvers for the Cardiac Bidomain Equations"

## Why Bidomain Over Monodomain?

The monodomain equation:
```
χ * C_m * ∂V_m/∂t = ∇·(σ_eff ∇V_m) - χ * I_ion
```

assumes σ_e → ∞ (infinite extracellular bath), so φ_e = 0 everywhere.

Bidomain is necessary when:
1. **Extracellular potential matters** (ECG simulation)
2. **Unequal anisotropy ratios** (σ_i and σ_e have different fiber/cross-fiber ratios)
3. **External stimulation** (defibrillation, pacing)
4. **Ion accumulation effects** ([K]o buildup in confined extracellular space)

For the infarct border speedup question: bidomain with LRd ionic model will capture the effect of elevated [K]o in confined spaces near infarct borders.
