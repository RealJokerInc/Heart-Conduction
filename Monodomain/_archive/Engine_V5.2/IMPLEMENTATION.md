# Engine V5.2: Implementation Plan

## Overview

V5.2 introduces two major architectural changes:
1. **Calibration Pipeline**: Optimization-based parameter tuning for diffusion coefficients
2. **FEM Diffusion**: Finite Element Method replacing Finite Volume Method

The ionic model (ORd) is preserved from V5.1.

---

## Part 1: Calibration Pipeline

### Motivation

V5.1 identified that diffusion coefficient D depends on multiple factors:
- Target conduction velocity (CV)
- Mesh spacing (dx)
- Action potential duration (APD)
- Ionic model excitability

The ad-hoc formula `D = (CV/k)²` is insufficient because:
1. k is not constant across mesh sizes
2. APD affects minimum stable D
3. Source-sink dynamics create non-linear coupling

### OpenCarp Approach (Reference)

OpenCarp uses a "bench" calibration:
1. Run 1D cable simulations along X and Y axes
2. Measure CV from activation times
3. Iterate D values until CV matches target
4. Validate on 2D tissue before full simulation

### V5.2 Approach: Tiered Optimization with 2D Tissue ERP

Instead of iterative bisection, use Differential Evolution optimization with tiered approach:

**Tier 1: Fixed dt**
```
minimize: L(D_L, D_T) = w₁|CV_L - CV_target_L|²/CV_L²
                       + w₂|CV_T - CV_target_T|²/CV_T²
                       + w₃·max(0, APD_min - APD)²/APD_min²
                       + w₄|ERP_tissue - ERP_target|²/ERP_target²
subject to: D_L, D_T ∈ [D_min, D_max]
```

**Tier 2: Variable dt** (if Tier 1 fails)
```
minimize: L(D_L, D_T, dt) = [same as Tier 1]
                           + w₅|dt - dt_default|²/dt_default²
subject to: D_L, D_T ∈ [D_min, D_max]
            dt ∈ [dt_min, dt_max]  where dt_max = 0.5 × dx²/D_max
```

**Weights:** w₁=1.0, w₂=1.0, w₃=0.5, w₄=5.0, w₅=0.01 (very small to prioritize D accuracy)

**2D Tissue ERP Measurement:**
- Square mesh sized by wavelength: L = 1.5 × CV × ERP (with margin)
- Central stimulus (S1, S2)
- Probes at x-terminus and y-terminus
- Single ERP = min interval where BOTH probes activate

---

### Stage C1: 1D Cable Simulation

**Objective**: Create a 1D cable simulation for CV measurement.

#### Mathematical Model

1D monodomain equation:
```
χ·Cm·∂V/∂t = D·∂²V/∂x² - χ·Iion
```

#### Implementation

```python
class Cable1D:
    """1D cable simulation for calibration."""

    def __init__(
        self,
        length_cm: float,     # Cable length
        dx: float,            # Mesh spacing
        D: float,             # Diffusion coefficient
        celltype: CellType = CellType.ENDO,
        params_override: dict = None
    ):
        self.n_cells = int(length_cm / dx) + 1
        self.dx = dx
        self.D = D

        # Create ionic model
        self.model = ORdModel(celltype=celltype, params_override=params_override)

        # State array (n_cells, n_states)
        self.states = self._init_states()

    def _diffusion_1d(self, V: torch.Tensor) -> torch.Tensor:
        """Compute D·∂²V/∂x² with no-flux BC."""
        n = len(V)
        diff = torch.zeros_like(V)

        # Interior: central difference
        diff[1:-1] = self.D * (V[2:] - 2*V[1:-1] + V[:-2]) / (self.dx**2)

        # Boundaries: no-flux (ghost cell approach)
        diff[0] = self.D * (V[1] - V[0]) / (self.dx**2)
        diff[-1] = self.D * (V[-2] - V[-1]) / (self.dx**2)

        return diff

    def measure_cv(self, threshold: float = -40.0) -> float:
        """
        Stimulate left end, measure CV from activation times.

        Returns CV in cm/ms.
        """
        # Run simulation with left-end stimulus
        # Record activation times at each node
        # Fit linear regression: x = CV * t + offset
        # Return CV
        pass
```

#### Validation C1

- [ ] 1D cable propagates correctly
- [ ] Activation times increase linearly with distance
- [ ] CV measurement matches V5.1 for same D
- [ ] Boundary effects don't contaminate interior CV

---

### Stage C2: APD Measurement

**Objective**: Measure APD from 1D cable or isolated cell.

#### Implementation

```python
def measure_apd90(V_trace: np.ndarray, t_array: np.ndarray) -> float:
    """
    Compute APD at 90% repolarization.

    Parameters
    ----------
    V_trace : array
        Voltage time series
    t_array : array
        Time points

    Returns
    -------
    apd90 : float
        APD90 in ms (or NaN if no AP detected)
    """
    # Find upstroke (dV/dt max)
    dVdt = np.gradient(V_trace, t_array)
    upstroke_idx = np.argmax(dVdt)
    V_peak = np.max(V_trace)
    V_rest = V_trace[0]

    # 90% repolarization level
    V_90 = V_rest + 0.1 * (V_peak - V_rest)

    # Find crossing after upstroke
    for i in range(upstroke_idx, len(V_trace)-1):
        if V_trace[i] > V_90 and V_trace[i+1] <= V_90:
            # Linear interpolation
            frac = (V_90 - V_trace[i]) / (V_trace[i+1] - V_trace[i])
            t_90 = t_array[i] + frac * (t_array[i+1] - t_array[i])
            return t_90 - t_array[upstroke_idx]

    return np.nan
```

#### Validation C2

- [ ] APD90 matches published ORd values (~280ms for ENDO)
- [ ] APD varies correctly with params_override (GKr, PCa)
- [ ] APD measured in cable matches isolated cell

---

### Stage C3: Optimization Core

**Objective**: Implement the optimization loop.

#### Cost Function

```python
def calibration_cost(
    params: np.ndarray,  # [D_L, D_T]
    targets: dict,
    cable_x: Cable1D,
    cable_y: Cable1D
) -> float:
    """
    Cost function for calibration optimization.

    Minimizes:
    - CV error (longitudinal and transverse)
    - APD error
    - Stability violations
    """
    D_L, D_T = params

    # Update cables with new D values
    cable_x.D = D_L
    cable_y.D = D_T

    # Measure CV
    cv_long = cable_x.measure_cv()
    cv_trans = cable_y.measure_cv()

    # Measure APD (from X cable center)
    apd = cable_x.measure_apd_at_center()

    # Compute cost
    cv_cost = ((cv_long - targets['cv_longitudinal'])**2 +
               (cv_trans - targets['cv_transverse'])**2)

    apd_cost = (apd - targets['apd90'])**2

    # Stability penalty
    D_min = compute_D_min(targets['dx'], apd)
    stability_penalty = 0.0
    if D_T < D_min:
        stability_penalty = 1000 * (D_min - D_T)**2

    # Anisotropy penalty (soft constraint)
    target_ratio = targets.get('anisotropy_ratio', D_L / D_T)
    ratio_penalty = (D_L / D_T - target_ratio)**2

    return cv_cost + 0.001 * apd_cost + stability_penalty + 0.1 * ratio_penalty
```

#### Optimizer

```python
from scipy.optimize import minimize

def calibrate_diffusion(targets: dict) -> dict:
    """
    Calibrate D_L and D_T to match target CV and APD.

    Parameters
    ----------
    targets : dict
        Target parameters (cv_longitudinal, cv_transverse, apd90, dx, dt)

    Returns
    -------
    result : dict
        Calibrated D values and validation metrics
    """
    # Initial estimate from empirical formula
    k = 1.514
    D_L_init = (targets['cv_longitudinal'] / k)**2
    D_T_init = (targets['cv_transverse'] / k)**2

    # Create 1D cables
    cable_length = 2.0  # 2 cm
    cable_x = Cable1D(cable_length, targets['dx'], D_L_init)
    cable_y = Cable1D(cable_length, targets['dx'], D_T_init)

    # Bounds
    D_min = compute_D_min(targets['dx'], targets['apd90'])
    bounds = [(D_min, 0.01), (D_min, 0.01)]

    # Optimize
    result = minimize(
        calibration_cost,
        x0=[D_L_init, D_T_init],
        args=(targets, cable_x, cable_y),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 50}
    )

    D_L_opt, D_T_opt = result.x

    # Validate
    cable_x.D = D_L_opt
    cable_y.D = D_T_opt
    cv_final_long = cable_x.measure_cv()
    cv_final_trans = cable_y.measure_cv()

    return {
        'D_L': D_L_opt,
        'D_T': D_T_opt,
        'cv_actual_long': cv_final_long,
        'cv_actual_trans': cv_final_trans,
        'cv_error_long': abs(cv_final_long - targets['cv_longitudinal']),
        'cv_error_trans': abs(cv_final_trans - targets['cv_transverse']),
        'stability_margin': D_T_opt / D_min,
        'converged': result.success
    }
```

#### Validation C3

- [x] Optimizer converges for typical parameter ranges
- [x] Final CV within 5% of targets
- [x] Stability constraints satisfied
- [x] Anisotropy ratio preserved

---

### Stage C3.5: 2D Tissue ERP Measurement ✓

**Objective**: Measure tissue ERP using 2D simulation with central stimulus and edge probes.

**File**: `calibration/tissue_erp_2d.py`

#### Geometry

```
┌─────────────────────────────────┐
│                                 │
│            probe_y              │  (ny-1, nx//2)
│               ↓                 │
│                                 │
│      [center]━━━━━━→ probe_x    │  (ny//2, nx-1)
│      S1, S2                     │
│                                 │
└─────────────────────────────────┘
```

#### Implementation

```python
@dataclass
class Tissue2DConfig:
    """Configuration for 2D tissue simulation."""
    nx: Optional[int] = None    # Cells in x direction
    ny: Optional[int] = None    # Cells in y direction
    dx: float = 0.02            # Mesh spacing (cm)
    dt: float = 0.02            # Time step (ms)
    stim_radius: int = 3        # Radius of central stimulus (cells)

    def compute_mesh_size(self, cv_x: float, cv_y: float, erp_est: float, margin: float = 1.5):
        """Compute mesh size based on wavelength with margin."""
        wavelength_x = cv_x * erp_est
        wavelength_y = cv_y * erp_est
        self.nx = max(int(margin * wavelength_x / self.dx) + 1, 50)
        self.ny = max(int(margin * wavelength_y / self.dx) + 1, 50)

class ERPResult2D(NamedTuple):
    erp: float          # Single ERP where BOTH probes activate
    erp_x: float        # ERP at x-terminus only
    erp_y: float        # ERP at y-terminus only
    apd_center: float
    success: bool

def measure_tissue_erp_2d(D_x, D_y, dx, dt, cv_x_est, cv_y_est, erp_est, ...):
    """
    Protocol:
    1. Apply S1 at center → wave propagates to edges
    2. Wait for repolarization
    3. Apply S2 at decreasing intervals
    4. ERP = minimum interval where BOTH probes activate
    """
```

#### Diffusion

Uses explicit 2D diffusion with Neumann boundary conditions:

```python
def _diffusion_step(self, V):
    V_new[1:-1, 1:-1] = V[1:-1, 1:-1] + (
        alpha_x * (V[1:-1, 2:] - 2*V[1:-1, 1:-1] + V[1:-1, :-2]) +
        alpha_y * (V[2:, 1:-1] - 2*V[1:-1, 1:-1] + V[:-2, 1:-1])
    )
    # No-flux boundaries
    V_new[0, :] = V_new[1, :]
    V_new[-1, :] = V_new[-2, :]
    V_new[:, 0] = V_new[:, 1]
    V_new[:, -1] = V_new[:, -2]
```

#### Validation C3.5

- [x] Central stimulus propagates correctly
- [x] Both probes detect activation
- [x] ERP decreases with S2 interval scan
- [x] Single ERP reflects combined D_x/D_y effect

---

### Stage C4: 2D Validation

**Objective**: Validate calibrated parameters on 2D tissue.

#### Implementation

```python
def validate_2d(D_L: float, D_T: float, targets: dict) -> dict:
    """
    Run quick 2D simulation to validate calibration.

    Tests:
    1. Planar wave CV in X direction
    2. Planar wave CV in Y direction
    3. Point stimulus wavefront shape
    """
    # Create small 2D simulation
    sim = MonodomainSimulation(
        ny=100, nx=200,
        dx=targets['dx'],
        D_L=D_L, D_T=D_T
    )

    # Test 1: X-direction plane wave
    sim.add_stimulus(region=(slice(None), slice(0, 3)), start_time=1.0)
    t, V = sim.run(t_end=50.0, dt=0.02)
    cv_x = sim.compute_cv_from_activation(V, direction='x')

    # Test 2: Y-direction plane wave
    sim.reset()
    sim.add_stimulus(region=(slice(0, 3), slice(None)), start_time=1.0)
    t, V = sim.run(t_end=50.0, dt=0.02)
    cv_y = sim.compute_cv_from_activation(V, direction='y')

    # Test 3: Point stimulus
    sim.reset()
    sim.add_stimulus(region=(slice(48, 52), slice(98, 102)), start_time=1.0)
    t, V = sim.run(t_end=30.0, dt=0.02)
    ellipse_ratio = compute_wavefront_ellipse_ratio(V[-1])
    expected_ratio = np.sqrt(D_T / D_L)

    return {
        'cv_x_2d': cv_x,
        'cv_y_2d': cv_y,
        'cv_x_error': abs(cv_x - targets['cv_longitudinal']) / targets['cv_longitudinal'],
        'cv_y_error': abs(cv_y - targets['cv_transverse']) / targets['cv_transverse'],
        'ellipse_ratio': ellipse_ratio,
        'expected_ratio': expected_ratio,
        'shape_error': abs(ellipse_ratio - expected_ratio) / expected_ratio
    }
```

#### Validation C4

- [ ] 2D CV matches 1D CV within 10%
- [ ] Wavefront shape error < 15%
- [ ] No propagation failures

---

## Part 2: FEM Diffusion Operator

### Motivation

FVM on structured grids produces:
- Stadium-shaped wavefronts (12% deviation at dx=0.2mm)
- Grid-aligned preferential conduction
- Mesh-dependent artifacts

FEM provides:
- Consistent weak form discretization
- Natural handling of boundary conditions
- Better anisotropic tensor interpolation

### OpenCarp FEM Reference

OpenCarp uses:
- Bilinear (Q1) quadrilateral elements
- Mass lumping for explicit time integration
- Operator splitting (ionic + diffusion)

### Mathematical Formulation

#### Strong Form

```
χ·Cm·∂V/∂t = ∇·(D·∇V) - χ·Iion + Istim
```

#### Weak Form

Multiply by test function φ and integrate by parts:

```
∫Ω χ·Cm·(∂V/∂t)·φ dΩ = -∫Ω (D·∇V)·∇φ dΩ + ∫∂Ω (D·∇V·n)·φ dS - ∫Ω χ·Iion·φ dΩ
```

With no-flux BC (D·∇V·n = 0 on ∂Ω):

```
∫Ω χ·Cm·(∂V/∂t)·φ dΩ = -∫Ω (D·∇V)·∇φ dΩ - ∫Ω χ·Iion·φ dΩ
```

#### Semi-Discrete Form

Expand V in basis functions: `V(x,t) = Σⱼ Vⱼ(t)·Nⱼ(x)`

Choosing φ = Nᵢ:

```
M·dV/dt = -K·V - M·(Iion/Cm)
```

Where:
- `Mᵢⱼ = χ·Cm · ∫Ω Nᵢ·Nⱼ dΩ` (mass matrix)
- `Kᵢⱼ = ∫Ω (D·∇Nᵢ)·∇Nⱼ dΩ` (stiffness matrix)

---

### Stage F1: Element Basis Functions

**Objective**: Implement bilinear (Q1) shape functions.

#### Reference Element

Mapping from reference square [-1,1]² to physical element:
```
x(ξ,η) = Σᵢ xᵢ·Nᵢ(ξ,η)
y(ξ,η) = Σᵢ yᵢ·Nᵢ(ξ,η)
```

#### Shape Functions

For bilinear quadrilateral:
```
N₁(ξ,η) = (1-ξ)(1-η)/4  # Node at (-1,-1)
N₂(ξ,η) = (1+ξ)(1-η)/4  # Node at (+1,-1)
N₃(ξ,η) = (1+ξ)(1+η)/4  # Node at (+1,+1)
N₄(ξ,η) = (1-ξ)(1+η)/4  # Node at (-1,+1)
```

#### Derivatives

```
∂N₁/∂ξ = -(1-η)/4,  ∂N₁/∂η = -(1-ξ)/4
∂N₂/∂ξ = +(1-η)/4,  ∂N₂/∂η = -(1+ξ)/4
∂N₃/∂ξ = +(1+η)/4,  ∂N₃/∂η = +(1+ξ)/4
∂N₄/∂ξ = -(1+η)/4,  ∂N₄/∂η = +(1-ξ)/4
```

#### Implementation

```python
def shape_functions(xi: float, eta: float) -> np.ndarray:
    """Bilinear shape functions at (xi, eta)."""
    return np.array([
        (1 - xi) * (1 - eta) / 4,
        (1 + xi) * (1 - eta) / 4,
        (1 + xi) * (1 + eta) / 4,
        (1 - xi) * (1 + eta) / 4
    ])

def shape_derivatives(xi: float, eta: float) -> np.ndarray:
    """
    Shape function derivatives.

    Returns (2, 4) array: [[dN/dxi], [dN/deta]]
    """
    return np.array([
        [-(1 - eta) / 4, +(1 - eta) / 4, +(1 + eta) / 4, -(1 + eta) / 4],  # dN/dxi
        [-(1 - xi) / 4, -(1 + xi) / 4, +(1 + xi) / 4, +(1 - xi) / 4]       # dN/deta
    ])
```

#### Validation F1

- [ ] Shape functions sum to 1 at any point
- [ ] Shape function is 1 at corresponding node, 0 at others
- [ ] Partition of unity: Σ ∂Nᵢ/∂ξ = 0

---

### Stage F2: Jacobian and Coordinate Transform

**Objective**: Map between reference and physical coordinates.

#### Jacobian Matrix

```
J = [[∂x/∂ξ, ∂y/∂ξ],
     [∂x/∂η, ∂y/∂η]]
```

For rectangular elements with spacing (dx, dy):
```
J = [[dx/2,    0 ],
     [  0,  dy/2]]
```

#### Physical Derivatives

```
[∂N/∂x]     -1   [∂N/∂ξ]
[∂N/∂y] = J   · [∂N/∂η]
```

For rectangular elements:
```
∂N/∂x = (2/dx) · ∂N/∂ξ
∂N/∂y = (2/dy) · ∂N/∂η
```

#### Implementation

```python
def compute_jacobian(dx: float, dy: float) -> Tuple[np.ndarray, float]:
    """
    Compute Jacobian for rectangular element.

    Returns (J, det_J)
    """
    J = np.array([[dx / 2, 0],
                  [0, dy / 2]])
    det_J = (dx * dy) / 4
    return J, det_J

def physical_derivatives(dN_dxi: np.ndarray, J_inv: np.ndarray) -> np.ndarray:
    """
    Transform derivatives to physical coordinates.

    Parameters
    ----------
    dN_dxi : (2, 4) array
        Derivatives w.r.t. (xi, eta)
    J_inv : (2, 2) array
        Inverse Jacobian

    Returns
    -------
    dN_dx : (2, 4) array
        Derivatives w.r.t. (x, y)
    """
    return J_inv @ dN_dxi
```

#### Validation F2

- [ ] Jacobian determinant = dx·dy/4 for rectangular elements
- [ ] Physical derivatives scale correctly with mesh size
- [ ] Transformation is exact for rectangular elements

---

### Stage F3: Stiffness Matrix Assembly

**Objective**: Compute element and global stiffness matrices.

#### Element Stiffness

```
Kₑ = ∫∫ Bᵀ · D · B · det(J) dξdη
```

Where B is the gradient matrix:
```
B = [∂N₁/∂x, ∂N₂/∂x, ∂N₃/∂x, ∂N₄/∂x]
    [∂N₁/∂y, ∂N₂/∂y, ∂N₃/∂y, ∂N₄/∂y]
```

#### Numerical Integration (2×2 Gauss)

Gauss points: ξ,η ∈ {-1/√3, +1/√3}
Weights: wᵢ = wⱼ = 1

```
Kₑ ≈ Σᵢ Σⱼ wᵢ·wⱼ · Bᵀ(ξᵢ,ηⱼ) · D · B(ξᵢ,ηⱼ) · det(J)
```

#### Implementation

```python
def compute_element_stiffness(
    dx: float, dy: float,
    D_xx: float, D_yy: float, D_xy: float
) -> np.ndarray:
    """
    Compute 4×4 element stiffness matrix.

    Uses 2×2 Gauss quadrature.
    """
    # Gauss points and weights
    gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
    gw = [1.0, 1.0]

    # Jacobian (constant for rectangular element)
    J, det_J = compute_jacobian(dx, dy)
    J_inv = np.linalg.inv(J)

    # Diffusion tensor
    D = np.array([[D_xx, D_xy],
                  [D_xy, D_yy]])

    # Initialize element matrix
    Ke = np.zeros((4, 4))

    # Gauss quadrature
    for i, xi in enumerate(gp):
        for j, eta in enumerate(gp):
            # Shape function derivatives in reference coords
            dN_dxi = shape_derivatives(xi, eta)

            # Transform to physical derivatives
            B = J_inv @ dN_dxi  # (2, 4)

            # Integrate
            Ke += gw[i] * gw[j] * (B.T @ D @ B) * det_J

    return Ke
```

#### Global Assembly

For structured grid (ny × nx nodes), element (i,j) connects nodes:
```
[i·nx + j,   i·nx + j+1,   (i+1)·nx + j+1,   (i+1)·nx + j]
```

```python
def assemble_stiffness_matrix(
    ny: int, nx: int,
    dx: float, dy: float,
    D_xx: float, D_yy: float, D_xy: float
) -> torch.sparse.Tensor:
    """
    Assemble global stiffness matrix in sparse format.
    """
    n_nodes = ny * nx
    n_elements = (ny - 1) * (nx - 1)

    # Compute reference element stiffness (same for all elements)
    Ke = compute_element_stiffness(dx, dy, D_xx, D_yy, D_xy)

    # Build sparse matrix
    rows = []
    cols = []
    vals = []

    for ei in range(ny - 1):
        for ej in range(nx - 1):
            # Global node indices
            nodes = [
                ei * nx + ej,
                ei * nx + ej + 1,
                (ei + 1) * nx + ej + 1,
                (ei + 1) * nx + ej
            ]

            # Add element contributions
            for a in range(4):
                for b in range(4):
                    rows.append(nodes[a])
                    cols.append(nodes[b])
                    vals.append(Ke[a, b])

    # Create sparse tensor
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float64)
    K = torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).coalesce()

    return K
```

#### Validation F3

- [ ] Element stiffness is symmetric
- [ ] Null space: K·1 = 0 (constant voltage has zero diffusion)
- [ ] Eigenvalues are non-negative
- [ ] Matches FVM Laplacian for isotropic case

---

### Stage F4: Mass Matrix (Lumped)

**Objective**: Compute lumped mass matrix for explicit integration.

#### Consistent Mass Matrix

```
Mₑ = χ·Cm · ∫∫ Nᵀ·N · det(J) dξdη
```

For Q1 element with 2×2 quadrature:
```
Mₑ = χ·Cm·det(J)·(4/9) · [[4, 2, 1, 2],
                          [2, 4, 2, 1],
                          [1, 2, 4, 2],
                          [2, 1, 2, 4]]
```

#### Lumped Mass Matrix

Row-sum lumping (diagonal):
```
Mₗ[i,i] = Σⱼ M[i,j]
```

For uniform rectangular elements:
```
Mₗ[i,i] = χ·Cm·dx·dy  (for interior nodes, scaled at boundaries)
```

#### Implementation

```python
def compute_lumped_mass(ny: int, nx: int, dx: float, dy: float,
                        chi: float = 1400.0, Cm: float = 1.0) -> torch.Tensor:
    """
    Compute lumped mass vector.

    Returns diagonal of lumped mass matrix as (ny*nx,) tensor.
    """
    # Each interior node is surrounded by 4 elements, each contributing dx·dy/4
    # So total = dx·dy

    # Corner nodes: 1 element → dx·dy/4
    # Edge nodes: 2 elements → dx·dy/2
    # Interior nodes: 4 elements → dx·dy

    M = torch.zeros(ny * nx, dtype=torch.float64)

    for i in range(ny):
        for j in range(nx):
            node = i * nx + j

            # Count contributing elements
            n_elements = 0
            if i > 0 and j > 0: n_elements += 1
            if i > 0 and j < nx - 1: n_elements += 1
            if i < ny - 1 and j > 0: n_elements += 1
            if i < ny - 1 and j < nx - 1: n_elements += 1

            M[node] = chi * Cm * dx * dy * n_elements / 4

    return M
```

#### Validation F4

- [ ] Mass matrix conserves total mass: Σ M[i] = χ·Cm·(domain area)
- [ ] Corner nodes have 1/4 of interior mass
- [ ] Edge nodes have 1/2 of interior mass

---

### Stage F5: Time Integration

**Objective**: Combine FEM matrices with operator splitting.

#### Semi-Discrete System

```
M·dV/dt = -K·V - M·(Iion/Cm)
```

Rearranging:
```
dV/dt = M⁻¹·(-K·V) - Iion/Cm
```

With lumped mass (diagonal), M⁻¹ is trivial.

#### Operator Splitting

Godunov (first-order):
1. **Ionic step**: `V* = V + dt·(-Iion/Cm)` (Rush-Larsen for gates)
2. **Diffusion step**: `V^(n+1) = V* + dt·M⁻¹·(-K·V*)`

Strang (second-order):
1. **Half diffusion**: `V* = V + (dt/2)·M⁻¹·(-K·V)`
2. **Full ionic**: `V** = V* + dt·(-Iion/Cm)`
3. **Half diffusion**: `V^(n+1) = V** + (dt/2)·M⁻¹·(-K·V**)`

#### Implementation

```python
class FEMDiffusionOperator:
    """FEM-based diffusion operator."""

    def __init__(self, ny: int, nx: int, dx: float, dy: float,
                 D_L: float, D_T: float, fiber_angle: float = 0.0):
        self.ny = ny
        self.nx = nx
        self.n_nodes = ny * nx

        # Compute diffusion tensor
        D_xx, D_yy, D_xy = compute_diffusion_tensor(fiber_angle, D_L, D_T)

        # Assemble matrices
        self.K = assemble_stiffness_matrix(ny, nx, dx, dy, D_xx, D_yy, D_xy)
        self.M_inv = 1.0 / compute_lumped_mass(ny, nx, dx, dy)

    def apply(self, V: torch.Tensor) -> torch.Tensor:
        """
        Compute M⁻¹·(-K·V).

        Parameters
        ----------
        V : (ny, nx) tensor
            Voltage field

        Returns
        -------
        dV : (ny, nx) tensor
            Diffusion contribution to dV/dt
        """
        # Flatten V for sparse matrix multiply
        V_flat = V.flatten()

        # Compute -K·V (sparse matrix-vector product)
        KV = torch.sparse.mm(self.K, V_flat.unsqueeze(1)).squeeze()

        # Apply M⁻¹
        dV = -self.M_inv * KV

        # Reshape to (ny, nx)
        return dV.view(self.ny, self.nx)
```

#### Validation F5

- [ ] Explicit Euler is stable below CFL limit
- [ ] Energy decreases monotonically (diffusion only)
- [ ] Matches FVM for isotropic case within numerical tolerance

---

### Stage F6: Stability Analysis

**Objective**: Compute CFL condition for explicit integration.

#### Theoretical Limit

For explicit Euler with FEM:
```
dt_max < 2 / λ_max(M⁻¹K)
```

For rectangular elements, approximately:
```
dt_max ≈ min(dx², dy²) / (4 · max(D_xx, D_yy))
```

#### Implementation

```python
def get_stability_limit(self, safety: float = 0.9) -> float:
    """
    Estimate maximum stable time step.

    Uses conservative approximation.
    """
    D_max = max(self.D_L, self.D_T)
    dx_min = min(self.dx, self.dy)

    # CFL condition for explicit diffusion
    dt_max = 0.25 * dx_min**2 / D_max

    return safety * dt_max
```

#### Validation F6

- [ ] dt at 90% limit is stable for 10000 steps
- [ ] dt at 110% limit shows oscillations/instability
- [ ] Stability limit matches theoretical prediction

---

## Part 3: Integration

### Stage I1: MonodomainSimulation Class

**Objective**: Integrate calibration, FEM diffusion, and ionic model.

```python
class MonodomainSimulation:
    """
    2D Monodomain simulation with FEM diffusion.

    Usage:
    1. Calibration mode: Provide CV targets, auto-calibrate D
    2. Direct mode: Provide D values directly
    """

    def __init__(
        self,
        ny: int, nx: int,
        dx: float = 0.02,
        dy: float = None,
        # Calibration mode
        cv_long: float = None,
        cv_trans: float = None,
        apd_target: float = None,
        # Direct mode
        D_L: float = None,
        D_T: float = None,
        # Common
        fiber_angle: float = 0.0,
        celltype: CellType = CellType.ENDO,
        auto_calibrate: bool = True
    ):
        dy = dy or dx

        # Determine D values
        if D_L is not None and D_T is not None:
            # Direct mode
            self.D_L = D_L
            self.D_T = D_T
            self.calibration_result = None
        elif auto_calibrate and cv_long is not None:
            # Calibration mode
            targets = {
                'cv_longitudinal': cv_long,
                'cv_transverse': cv_trans or cv_long / 3,
                'apd90': apd_target or 280.0,
                'dx': dx
            }
            self.calibration_result = calibrate_diffusion(targets)
            self.D_L = self.calibration_result['D_L']
            self.D_T = self.calibration_result['D_T']
        else:
            raise ValueError("Must provide either D values or CV targets")

        # Create ionic model
        self.ionic = ORdModel(celltype=celltype)

        # Create FEM diffusion operator
        self.diffusion = FEMDiffusionOperator(
            ny, nx, dx, dy, self.D_L, self.D_T, fiber_angle
        )

        # Initialize state
        self.states = self._init_states()
        self.time = 0.0
```

#### Validation I1

- [ ] Both initialization modes work
- [ ] Calibration results stored and accessible
- [ ] Simulation runs without errors

---

### Stage I2: Validation Suite

**Objective**: Comprehensive validation tests.

```python
def run_validation_suite():
    """
    Complete validation of V5.2 implementation.
    """
    results = {}

    # Test 1: Single cell AP
    print("Test 1: Single cell AP...")
    model = ORdModel()
    apd = run_single_cell_and_measure_apd(model)
    results['single_cell_apd'] = apd
    assert 270 < apd < 290, f"APD out of range: {apd}"

    # Test 2: 1D cable CV
    print("Test 2: 1D cable CV...")
    cable = Cable1D(length=2.0, dx=0.02, D=0.00151)
    cv = cable.measure_cv()
    results['cable_cv'] = cv
    assert 0.05 < cv < 0.07, f"CV out of range: {cv}"

    # Test 3: Calibration convergence
    print("Test 3: Calibration...")
    cal_result = calibrate_diffusion({
        'cv_longitudinal': 0.06,
        'cv_transverse': 0.02,
        'apd90': 280,
        'dx': 0.02
    })
    results['calibration'] = cal_result
    assert cal_result['converged'], "Calibration failed to converge"
    assert cal_result['cv_error_long'] < 0.003, "CV error too large"

    # Test 4: FEM vs FVM comparison
    print("Test 4: FEM vs FVM...")
    fem_cv, fvm_cv = compare_fem_fvm(D=0.00151, dx=0.02)
    results['fem_cv'] = fem_cv
    results['fvm_cv'] = fvm_cv
    assert abs(fem_cv - fvm_cv) < 0.01, "FEM/FVM mismatch too large"

    # Test 5: Wavefront shape
    print("Test 5: Wavefront shape...")
    shape_error = measure_wavefront_ellipticity(D_L=0.00151, D_T=0.0005, dx=0.02)
    results['shape_error'] = shape_error
    # FEM should have better shape than V5.1's 12%
    assert shape_error < 0.10, f"Shape error too large: {shape_error}"

    print("\n=== VALIDATION PASSED ===")
    return results
```

---

## Implementation Phases

### Phase 1: Calibration Pipeline (Priority: HIGH)

| Stage | Description | Status |
|-------|-------------|--------|
| C1 | 1D Cable Simulation | ✓ Complete |
| C2 | APD Measurement | ✓ Complete |
| C3 | Optimization Core (Tiered) | ✓ Complete |
| C3.5 | 2D Tissue ERP Measurement | ✓ Complete |
| C4 | 2D Validation | Pending |

**Key Implementation:**
- Tiered optimization: Tier 1 (fixed dt) → Tier 2 (variable dt)
- 2D tissue ERP: central stim, x/y probes, single ERP where both activate
- dt regularization with very small weight (0.01) to prioritize D accuracy

**Dependencies**: Ionic model (done), diffusion operator (can use V5.1 FVM initially)

### Phase 2: FEM Diffusion (Priority: HIGH)

| Stage | Description | Estimated Effort |
|-------|-------------|------------------|
| F1 | Shape Functions | Low |
| F2 | Jacobian Transform | Low |
| F3 | Stiffness Matrix | High |
| F4 | Mass Matrix | Medium |
| F5 | Time Integration | Medium |
| F6 | Stability Analysis | Low |

**Dependencies**: Can be developed in parallel with calibration

### Phase 3: Integration (Priority: MEDIUM)

| Stage | Description | Estimated Effort |
|-------|-------------|------------------|
| I1 | MonodomainSimulation | Medium |
| I2 | Validation Suite | Medium |

**Dependencies**: Phases 1 and 2 complete

---

## File Structure

```
Engine_V5.2/
├── README.md
├── IMPLEMENTATION.md
├── ionic/                    # From V5.1 (unchanged)
│   ├── __init__.py
│   ├── model.py
│   ├── gating.py
│   ├── currents.py
│   ├── calcium.py
│   ├── camkii.py
│   └── parameters.py
├── tissue/                   # NEW
│   ├── __init__.py
│   ├── fem_diffusion.py      # FEMDiffusionOperator
│   ├── shape_functions.py    # Q1 basis functions
│   ├── assembly.py           # Matrix assembly
│   └── simulation.py         # MonodomainSimulation
├── calibration/              # NEW
│   ├── __init__.py
│   ├── cable_1d.py           # 1D cable simulation for CV/APD
│   ├── erp_measurement.py    # Single-cell and 1D tissue ERP
│   ├── tissue_erp_2d.py      # 2D tissue ERP measurement ✓
│   └── optimizer.py          # Tiered calibration optimizer ✓
├── utils/                    # From V5.1
│   ├── __init__.py
│   └── device.py
├── examples/
│   ├── calibrate_and_run.py
│   ├── compare_fem_fvm.py
│   └── spiral_wave.py
└── tests/
    ├── test_ionic.py
    ├── test_calibration.py
    ├── test_fem.py
    └── test_integration.py
```

---

## References

1. **OpenCarp**: https://opencarp.org/documentation
   - Bench calibration procedure
   - FEM tissue implementation

2. **O'Hara et al. (2011)**: ORd ionic model
   - PLoS Comput Biol 7(5): e1002061

3. **Niederer et al. (2011)**: Verification benchmarks
   - Phil Trans R Soc A 369: 4331-4351

4. **Pezzuto et al. (2016)**: Space-discretization error in cardiac EP
   - Int J Numer Meth Biomed Eng 32(10): e02762

5. **V5.1 DIFFUSION_BUG.md**: Lessons learned on D_min and mesh dependence
