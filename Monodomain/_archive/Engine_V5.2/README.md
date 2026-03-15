# Engine V5.2: Cardiac Electrophysiology with FEM Diffusion and Calibration Pipeline

High-performance cardiac electrophysiology simulation featuring:
- **O'Hara-Rudy 2011 (ORd)** ventricular myocyte ionic model (validated from V5.1)
- **Finite Element Method (FEM)** for tissue diffusion (new in V5.2)
- **Optimization-based calibration pipeline** for parameter tuning
- **GPU-accelerated** using PyTorch CUDA

## What's New in V5.2

### 1. FEM Diffusion Operator (Replaces FVM)

V5.1 used Finite Volume Method which exhibited:
- Stadium-shaped wavefronts (grid discretization artifact)
- ~12% deviation from elliptical shape at typical resolutions
- Mesh-dependent D_min stability limits

V5.2 implements FEM following OpenCarp's formulation:
- Improved wavefront shape accuracy
- Better handling of anisotropic diffusion tensors
- More consistent behavior across mesh resolutions

### 2. Optimization-Based Calibration Pipeline

V5.1 used ad-hoc empirical formulas for D computation:
```python
# V5.1 approach (limited)
D = (CV / k)^2  where k ≈ 1.514
```

V5.2 introduces a proper calibration pipeline that:
- Takes target electrophysiological parameters as input
- Runs 1D cable simulations for CV/APD measurement
- Uses **2D tissue simulation** for tissue ERP measurement
- Uses **tiered optimization** (fixed dt first, variable dt if needed)
- Validates against mesh-dependent stability criteria

**2D Tissue ERP Measurement:**
- Square mesh sized proportional to wavelength (CV × ERP) with 1.5× margin
- Central stimulus (S1, S2)
- Probes at x-terminus (right edge) and y-terminus (top edge)
- Single ERP = minimum S1-S2 interval where BOTH probes activate
- Both D_x and D_y affect the result

**Tiered Optimization:**
- Tier 1: Fixed dt - faster, simpler
- Tier 2: Variable dt - if Tier 1 fails, allows dt to vary
- dt regularization ensures D accuracy is prioritized (w=0.01)

### 3. Preserved Ionic Model

The ORd ionic model from V5.1 is unchanged and validated:
- 41 state variables
- 15 ionic currents
- CaMKII signaling
- GPU-accelerated Rush-Larsen integration

## Architecture

```
Engine_V5.2/
├── ionic/              # Cellular ionic model (from V5.1)
│   ├── model.py        # ORdModel class
│   ├── gating.py       # Voltage-dependent gating
│   ├── currents.py     # Ion current calculations
│   ├── calcium.py      # Ca2+ handling
│   ├── camkii.py       # CaMKII signaling
│   └── parameters.py   # Model parameters
├── tissue/             # Tissue-level simulation (NEW in V5.2)
│   ├── fem_diffusion.py    # FEM diffusion operator
│   ├── mass_matrix.py      # Mass matrix assembly
│   ├── stiffness_matrix.py # Stiffness matrix assembly
│   └── simulation.py       # MonodomainSimulation
├── calibration/        # NEW: Calibration pipeline
│   ├── cable_1d.py         # 1D cable simulation
│   ├── optimizer.py        # Parameter optimization
│   ├── targets.py          # Target specifications
│   └── validator.py        # Result validation
├── utils/              # Utilities
│   └── device.py       # GPU device management
├── examples/           # Example scripts
└── tests/              # Validation tests
```

## Quick Start

### Single Cell Simulation

```python
from ionic import ORdModel, CellType

# Create model
model = ORdModel(celltype=CellType.ENDO)
state = model.get_initial_state()

# Run simulation
dt = 0.01  # ms
for t in range(50000):  # 500 ms
    Istim = -80.0 if 10.0 <= t*dt < 11.0 else 0.0
    state = model.step(state, dt, Istim)
```

### Tissue Simulation with Calibration

```python
from calibration import CableCalibrator
from tissue import MonodomainSimulation

# Define targets
targets = {
    'cv_longitudinal': 0.06,   # 0.6 m/s
    'cv_transverse': 0.02,     # 0.2 m/s
    'apd90': 280,              # ms
    'dx': 0.02,                # 0.2 mm
    'dt': 0.02,                # ms
    'anisotropy_ratio': 3.0    # D_L / D_T
}

# Calibrate diffusion parameters
calibrator = CableCalibrator()
params = calibrator.calibrate(targets)

# Create simulation with calibrated parameters
sim = MonodomainSimulation(
    ny=300, nx=300,
    dx=targets['dx'],
    D_L=params['D_L'],
    D_T=params['D_T']
)
```

## Key Improvements Over V5.1

### Problem 1: D is Mesh-Dependent (Identified in DIFFUSION_BUG.md)

**V5.1 Finding:** The minimum stable diffusion coefficient D_min depends on:
- Mesh spacing dx: `D_min ∝ dx²`
- APD: `D_min ∝ (280/APD)^0.25`

**V5.1 Formula:**
```
D_min(dx, APD) = 0.92 × dx² × (280/APD)^0.25
```

**V5.2 Solution:** The calibration pipeline:
1. Takes dx, APD as inputs
2. Computes mesh-appropriate D values
3. Validates against stability criteria before simulation

### Problem 2: Stadium-Shaped Wavefronts

**V5.1 Finding:** FVM on square grids produces ~12% deviation from elliptical wavefronts due to different numerical properties along diagonals vs axes.

**V5.2 Solution:** FEM formulation with:
- Consistent weak form discretization
- Mass lumping for stability
- Proper handling of anisotropic tensor rotation

### Problem 3: Tissue ERP ≠ Single-Cell APD

**V5.1 Finding:** S2 capture requires 1.17× APD due to source-sink mismatch.

**V5.2 Solution:** Calibration pipeline includes ERP/FRP validation:
- Runs S1-S2 protocol on calibrated parameters
- Measures actual tissue ERP
- Reports ERP/APD ratio for spiral wave timing

## Calibration Pipeline Details

### Input Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `cv_longitudinal` | Target longitudinal CV | 0.05-0.07 cm/ms |
| `anisotropy_ratio` | D_L / D_T ratio (CV_T derived) | 2-4 |
| `erp_tissue_target` | Target 2D tissue ERP | 300-350 ms |
| `apd_min` | Minimum APD constraint | 250-300 ms |
| `dx` | Mesh spacing | 0.01-0.03 cm |
| `dt_default` | Default time step | 0.01-0.05 ms |

### Calibration Algorithm

Tiered optimization with 2D tissue ERP measurement:

1. **Initialize:** Estimate D from CV ∝ √D relationship
2. **Tier 1 (Fixed dt):**
   - Optimize [D_L, D_T] with dt = dt_default
   - Run 1D cable simulations for CV_L, CV_T
   - Run 2D tissue simulation for tissue ERP
   - If loss < threshold (0.1), accept result
3. **Tier 2 (Variable dt):** If Tier 1 fails:
   - Optimize [D_L, D_T, dt] with dt regularization
   - dt can vary within stability bounds
   - Very small regularization (w=0.01) prioritizes D accuracy

### 2D Tissue ERP Protocol

```
┌─────────────────────────────────┐
│            probe_y              │  (ny-1, nx//2)
│               ↓                 │
│      [center]━━━━━━→ probe_x    │  (ny//2, nx-1)
│      S1, S2                     │
└─────────────────────────────────┘

1. Apply S1 at center → wave propagates to edges
2. Wait for repolarization
3. Apply S2 at decreasing intervals
4. ERP = minimum interval where BOTH probes activate
```

### Loss Function

```
L = w_cv_L × (CV_L - target)²/target²
  + w_cv_T × (CV_T - target)²/target²
  + w_apd × max(0, APD_min - APD)²/APD_min²
  + w_erp × (ERP_tissue - target)²/target²
  + w_dt × (dt - dt_default)²/dt_default²  [Tier 2 only]
```

Weights: w_cv_L=1.0, w_cv_T=1.0, w_apd=0.5, w_erp=5.0, w_dt=0.01

### Output Parameters

```python
{
    'D_longitudinal': 0.00151,   # cm²/ms
    'D_transverse': 0.00050,     # cm²/ms
    'dt_used': 0.02,             # ms
    'cv_longitudinal': 0.059,    # cm/ms
    'cv_transverse': 0.020,      # cm/ms
    'erp_tissue': 318.0,         # ms (single 2D ERP)
    'wavelength_long': 18.76,    # cm
    'wavelength_trans': 6.36,    # cm
    'tier_used': 1,              # Which tier succeeded
    'convergence': True
}
```

## FEM Implementation

### Weak Form

Starting from the monodomain equation:
```
χ·Cm·∂V/∂t = ∇·(D·∇V) - χ·Iion
```

Multiply by test function φ and integrate:
```
∫∫ χ·Cm·(∂V/∂t)·φ dΩ = -∫∫ (D·∇V)·∇φ dΩ + ∫∂Ω (D·∇V·n)·φ dS - ∫∫ χ·Iion·φ dΩ
```

With no-flux BC, boundary term vanishes:
```
M·dV/dt = -K·V - M·(Iion/Cm)
```

Where:
- `M`: Mass matrix (lumped for stability)
- `K`: Stiffness matrix (includes anisotropic D tensor)

### Bilinear Element

Using Q1 (bilinear) quadrilateral elements:
- 4 nodes per element
- Shape functions: `N_i(ξ,η) = (1±ξ)(1±η)/4`
- 2×2 Gaussian quadrature for K
- Nodal quadrature (lumped) for M

### Stiffness Matrix Assembly

For element with anisotropic diffusion:
```
K_e = ∫∫ B^T · D · B · det(J) dξdη
```

Where:
- `B = [∂N/∂x, ∂N/∂y]`: Shape function derivatives
- `D = [[D_xx, D_xy], [D_xy, D_yy]]`: Diffusion tensor
- `J`: Jacobian of coordinate transformation

## Requirements

- Python 3.9+
- PyTorch 2.0+ with CUDA support
- NumPy, SciPy
- NVIDIA GPU with CUDA capability

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy scipy matplotlib
```

## References

1. O'Hara T, et al. (2011). "Simulation of the Undiseased Human Cardiac Ventricular Action Potential." PLoS Comput Biol.

2. OpenCarp Documentation - Tissue Simulations and Calibration
   https://opencarp.org/documentation

3. Niederer SA, et al. (2011). "Verification of cardiac tissue electrophysiology simulators using an N-version benchmark." Phil Trans R Soc A.

4. Pezzuto S, et al. (2016). "Space-discretization error analysis and stabilization schemes for conduction velocity in cardiac electrophysiology." Int J Numer Meth Biomed Eng.

## License

MIT License
