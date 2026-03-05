# Bidomain Engine V1: FDM-First Coding Structure Plan

**Date:** 2026-03-04
**Priority:** FDM only. FEM/FVM deferred to later phases.
**Design Philosophy:** Follows Monodomain Engine V5.4 patterns.

---

## Motivation: Why FDM-First

1. **Structured grids enable matrix-free GPU computation** -- stencil application via `conv2d` or direct indexing, no sparse matrix assembly needed
2. **Geometric multigrid is natural** on structured grids -- no AMG setup cost, simple restriction/prolongation
3. **The boundary speedup (Kleber effect)** we want to validate requires a clean FDM implementation with clear boundary condition handling
4. **V5.4 already has a proven FDM** -- we extend it to dual conductivity, not rewrite

## Key Architectural Decision: Decoupled over Coupled

Based on literature review (see `research/GPU_BIDOMAIN_LITERATURE.md`):

**Every production GPU bidomain solver decouples the system.** The fully coupled 2N x 2N MINRES approach is correct but:
- Requires indefinite linear solver (MINRES/GMRES, not PCG)
- Block preconditioners are complex
- Matrix-free application is harder with block matvec
- No GPU implementation uses it in practice

**Our approach: Decoupled (Gauss-Seidel splitting)**
```
Given Vm^n, phi_e^n:

Step 1: Ionic (local, GPU-trivial)
   Vm* = Vm^n + dt/2 * (-Iion + Istim)
   Update gates, concentrations (Rush-Larsen)

Step 2: Parabolic solve for Vm^{n+1} (N x N, SPD)
   (chi*Cm/dt * I - theta*L_i) * Vm^{n+1} =
      (chi*Cm/dt * I + (1-theta)*L_i) * Vm* + theta*L_i*phi_e^n
   i.e.  A_para * Vm^{n+1} = B_para * Vm* + theta * L_i * phi_e^n
   Solve with: PCG + Jacobi (or geometric multigrid)

Step 3: Elliptic solve for phi_e^{n+1} (N x N, SPD)
   -(L_i + L_e) * phi_e^{n+1} = L_i * Vm^{n+1}
   Solve with: PCG + geometric multigrid
   Pinning: phi_e(corner) = 0 (only if all-Neumann BCs; skip for Dirichlet)

Step 4: Ionic (local, GPU-trivial)
   Vm^{n+1} += dt/2 * (-Iion + Istim)
   Update gates, concentrations
```

This is a Strang-split, semi-implicit scheme where:
- Ionic: explicit (Rush-Larsen), split into half-steps
- Parabolic Vm: implicit (CN or BDF1), standard PCG
- Elliptic phi_e: direct solve (no time derivative), PCG + multigrid

**Advantage:** Each solve is N x N and SPD -- we can reuse V5.4's PCG solver directly.

---

## Directory Structure

```
Bidomain/Engine_V1/
|
+-- cardiac_sim/
|   |
|   +-- __init__.py                              # v1.0.0
|   |
|   +-- ionic/                                   # DIRECT COPY from V5.4
|   |   +-- base.py                              # IonicModel ABC
|   |   +-- lut.py                               # Lookup tables
|   |   +-- ttp06/                               # TTP06 model
|   |   +-- ord/                                 # ORd model
|   |
|   +-- tissue_builder/                          # MOSTLY COPIED from V5.4
|   |   +-- mesh/
|   |   |   +-- base.py                          # Mesh ABC
|   |   |   +-- structured.py                    # StructuredGrid (extended with BoundarySpec)
|   |   |   +-- boundary.py                      # NEW: BoundarySpec, BCType, EdgeBC
|   |   +-- tissue/
|   |   |   +-- isotropic.py                     # V5.4 compat
|   |   |   +-- conductivity.py                  # NEW: BidomainConductivity
|   |   +-- stimulus/
|   |       +-- protocol.py                      # StimulusProtocol
|   |       +-- regions.py                       # Stimulus regions
|   |
|   +-- simulation/
|   |   +-- classical/
|   |   |   +-- state.py                         # BidomainState
|   |   |   +-- bidomain.py                      # BidomainSimulation orchestrator
|   |   |   |
|   |   |   +-- discretization/                  # Spatial operators
|   |   |   |   +-- base.py                      # BidomainDiscretization ABC
|   |   |   |   +-- fdm.py                       # FDM: L_i, L_e stencils
|   |   |   |
|   |   |   +-- solver/
|   |   |       +-- splitting/
|   |   |       |   +-- base.py                  # SplittingStrategy ABC
|   |   |       |   +-- strang.py                # Strang: half-ionic -> diff -> half-ionic
|   |   |       |   +-- godunov.py               # Godunov: ionic -> diff
|   |   |       |
|   |   |       +-- ionic_stepping/              # COPIED from V5.4
|   |   |       |   +-- base.py                  # IonicSolver ABC
|   |   |       |   +-- rush_larsen.py           # Rush-Larsen
|   |   |       |   +-- forward_euler.py         # Forward Euler
|   |   |       |
|   |   |       +-- diffusion_stepping/          # NEW: bidomain-specific
|   |   |       |   +-- base.py                  # BidomainDiffusionSolver ABC
|   |   |       |   +-- decoupled.py             # Decoupled: parabolic + elliptic
|   |   |       |
|   |   |       +-- linear_solver/               # Extended from V5.4
|   |   |           +-- base.py                  # LinearSolver ABC
|   |   |           +-- pcg.py                   # PCG + Jacobi (from V5.4)
|   |   |           +-- chebyshev.py             # Chebyshev (from V5.4)
|   |   |           +-- spectral.py              # NEW: SpectralSolver (DCT/DST/FFT unified)
|   |   |           +-- pcg_spectral.py          # NEW: PCG + spectral preconditioner
|   |   |           +-- multigrid.py             # NEW: GeometricMultigridPreconditioner
|   |   |           +-- pcg_gmg.py              # NEW: Tier 3: PCG + geometric multigrid
|   |   |
|   |   +-- lbm/                                 # FUTURE
|   |       +-- __init__.py
|   |
|   +-- utils/
|       +-- backend.py                           # Device abstraction (from V5.4)
|
+-- research/
|   +-- BOUNDARY_SPEEDUP_ANALYSIS.md             # Kleber effect derivation
|   +-- GPU_BIDOMAIN_LITERATURE.md               # Literature review
|
+-- tests/
|   +-- test_phase1_foundation.py
|   +-- test_phase2_fdm.py
|   +-- test_phase3_solver.py
|   +-- test_phase4_diffusion.py
|   +-- test_phase5_orchestration.py
|   +-- test_phase6_boundary.py
|
+-- examples/
|   +-- planar_wave.py                           # Basic wave propagation
|   +-- boundary_speedup.py                      # Kleber effect validation
|
+-- README.md
+-- improvement.md
+-- IMPLEMENTATION.md
+-- PROGRESS.md
+-- FDM_CODING_PLAN.md                           # THIS FILE
```

---

## File-by-File Specification

### Phase 1: Foundation (Copy + Create)

#### `cardiac_sim/__init__.py`
```python
__version__ = "1.0.0"
```

#### `cardiac_sim/ionic/` -- DIRECT COPY from V5.4
No changes. All ionic model code is model-agnostic.

#### `cardiac_sim/tissue_builder/mesh/boundary.py` -- NEW
```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional

class BCType(Enum):
    NEUMANN = "neumann"
    DIRICHLET = "dirichlet"

class Edge(Enum):
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"

@dataclass
class EdgeBC:
    bc_type: BCType = BCType.NEUMANN
    value: float = 0.0

@dataclass
class BoundarySpec:
    """Per-edge, per-variable boundary condition specification.

    Stored in StructuredGrid. Consumed by:
    - Discretization (stencil construction at boundary nodes)
    - Solver (spectral transform selection: Neumann->DCT, Dirichlet->DST)
    - Diffusion solver (null space detection: Dirichlet -> no pinning needed)
    """
    Vm: Dict[Edge, EdgeBC] = field(default_factory=lambda: {e: EdgeBC() for e in Edge})
    phi_e: Dict[Edge, EdgeBC] = field(default_factory=lambda: {e: EdgeBC() for e in Edge})

    @property
    def phi_e_has_null_space(self) -> bool:
        """True if all phi_e BCs are Neumann (needs pinning)."""
        return all(bc.bc_type == BCType.NEUMANN for bc in self.phi_e.values())

    @property
    def phi_e_spectral_eligible(self) -> bool:
        """True if all phi_e edges have the SAME BC type (can use spectral)."""
        types = {bc.bc_type for bc in self.phi_e.values()}
        return len(types) == 1

    @property
    def spectral_transform(self) -> Optional[str]:
        """Which spectral transform to use, or None if mixed BCs."""
        if not self.phi_e_spectral_eligible:
            return None
        bc_type = next(iter(self.phi_e.values())).bc_type
        return {"neumann": "dct", "dirichlet": "dst"}[bc_type.value]

    @classmethod
    def insulated(cls) -> 'BoundarySpec':
        """All Neumann on both Vm and phi_e (standard insulated tissue)."""
        return cls()

    @classmethod
    def bath_coupled(cls, bath_value=0.0) -> 'BoundarySpec':
        """Vm=Neumann, phi_e=Dirichlet on ALL edges (tissue in bath)."""
        return cls(
            Vm={e: EdgeBC() for e in Edge},
            phi_e={e: EdgeBC(BCType.DIRICHLET, bath_value) for e in Edge},
        )

    @classmethod
    def bath_coupled_edges(cls, bath_edges, bath_value=0.0) -> 'BoundarySpec':
        """phi_e=Dirichlet only on specified edges, Neumann elsewhere."""
        phi_e = {}
        for e in Edge:
            if e in bath_edges:
                phi_e[e] = EdgeBC(BCType.DIRICHLET, bath_value)
            else:
                phi_e[e] = EdgeBC()
        return cls(Vm={e: EdgeBC() for e in Edge}, phi_e=phi_e)
```

#### `cardiac_sim/tissue_builder/tissue/conductivity.py` -- NEW
```python
@dataclass
class BidomainConductivity:
    """Paired intracellular and extracellular conductivity."""

    # Scalar isotropic (simplest)
    D_i: float = 0.00124       # cm^2/ms (sigma_i / chi*Cm)
    D_e: float = 0.00446       # cm^2/ms (sigma_e / chi*Cm)

    # Per-node fields (for heterogeneous tissue)
    D_i_field: Optional[Tuple[Tensor, Tensor, Tensor]] = None  # (Dxx_i, Dyy_i, Dxy_i)
    D_e_field: Optional[Tuple[Tensor, Tensor, Tensor]] = None

    # Fiber-based (for anisotropic tissue)
    D_i_fiber: Optional[float] = None   # Along fibers
    D_i_cross: Optional[float] = None   # Across fibers
    D_e_fiber: Optional[float] = None
    D_e_cross: Optional[float] = None
    theta: Optional[Tensor] = None      # Fiber angle field

    def get_effective_monodomain_D(self) -> float:
        """D_eff = D_i * D_e / (D_i + D_e) for validation."""
        return self.D_i * self.D_e / (self.D_i + self.D_e)

    def get_boundary_enhanced_D(self) -> float:
        """D at tissue-bath boundary = D_i (Kleber effect)."""
        return self.D_i
```

#### `cardiac_sim/simulation/classical/state.py` -- NEW
```python
@dataclass
class BidomainState:
    """Runtime data for bidomain simulation."""

    # Geometry
    spatial: 'BidomainDiscretization'
    n_dof: int
    x: torch.Tensor              # (n_dof,)
    y: torch.Tensor              # (n_dof,)

    # Potentials
    Vm: torch.Tensor             # (n_dof,) transmembrane
    phi_e: torch.Tensor          # (n_dof,) extracellular

    # Ionic
    ionic_states: torch.Tensor   # (n_dof, n_ionic_states)
    gate_indices: List[int]
    concentration_indices: List[int]

    # Time
    t: float

    # Stimulus (may have separate intracellular/extracellular)
    stim_masks: torch.Tensor
    stim_starts: torch.Tensor
    stim_durations: torch.Tensor
    stim_amplitudes_i: torch.Tensor   # Intracellular stimulus
    stim_amplitudes_e: torch.Tensor   # Extracellular stimulus

    # Output
    output_buffer_Vm: torch.Tensor
    output_buffer_phi_e: torch.Tensor
    buffer_idx: int

    @property
    def Vm_flat(self) -> torch.Tensor:
        return self.Vm
```

### Phase 2: FDM Bidomain Discretization

#### `cardiac_sim/simulation/classical/discretization/base.py` -- NEW
```python
class BidomainSpatialDiscretization(ABC):
    """ABC for bidomain spatial discretization.

    Provides TWO Laplacians (L_i, L_e) and operators for decoupled
    parabolic + elliptic solves. The grid and its BoundarySpec are
    accessible via self.grid.

    Matches improvement.md § BidomainSpatialDiscretization ABC.
    """

    @property
    @abstractmethod
    def n_dof(self) -> int: ...

    @property
    @abstractmethod
    def grid(self) -> 'StructuredGrid':
        """The mesh, including boundary_spec."""

    @property
    @abstractmethod
    def coordinates(self) -> Tuple[Tensor, Tensor]: ...

    # --- Operator application (matrix-free compatible) ---

    @abstractmethod
    def apply_L_i(self, V: Tensor) -> Tensor:
        """Apply intracellular Laplacian: L_i * V = div(D_i * grad(V))"""

    @abstractmethod
    def apply_L_e(self, V: Tensor) -> Tensor:
        """Apply extracellular Laplacian: L_e * V = div(D_e * grad(V))"""

    @abstractmethod
    def apply_L_ie(self, V: Tensor) -> Tensor:
        """Apply combined Laplacian: (L_i + L_e) * V"""

    # --- Decoupled solver operators ---

    @abstractmethod
    def get_parabolic_operators(self, dt: float, theta: float) -> Tuple:
        """Build (A_para, B_para) for: A_para * Vm^{n+1} = B_para * Vm^n + coupling."""

    @abstractmethod
    def get_elliptic_operator(self) -> Tensor:
        """Build A_ellip = -(L_i + L_e) for: A_ellip * phi_e = L_i * Vm."""

    # --- Raw matrices (for preconditioner setup) ---

    @property
    @abstractmethod
    def L_i(self) -> Tensor:
        """Intracellular Laplacian matrix (sparse)."""

    @property
    @abstractmethod
    def L_e(self) -> Tensor:
        """Extracellular Laplacian matrix (sparse)."""
```

#### `cardiac_sim/simulation/classical/discretization/fdm.py` -- NEW
```python
class BidomainFDMDiscretization(BidomainSpatialDiscretization):
    """FDM bidomain discretization on structured grid.

    Builds two 9-pt stencil Laplacians L_i and L_e from D_i and D_e.
    Supports scalar D, per-node D_field, and fiber-based anisotropy.

    Matrix-free option: apply_L_i/apply_L_e use conv2d stencil
    application without building sparse matrices.

    Sparse matrix option: builds sparse L_i, L_e for implicit solvers
    that need matrix-vector products.
    """

    def __init__(self, grid: StructuredGrid, conductivity: BidomainConductivity,
                 chi: float = 1400.0, Cm: float = 1.0):
        self._grid = grid
        self._chi_Cm = chi * Cm

        # Build two separate Laplacians using V5.4's proven stencil construction
        self._L_i = self._build_laplacian(conductivity.D_i, conductivity.D_i_field, 'Vm')
        self._L_e = self._build_laplacian(conductivity.D_e, conductivity.D_e_field, 'phi_e')
        self._L_ie = None  # Lazy: built on first use

    def apply_L_i(self, V):
        """Matrix-free: stencil application via sparse_mv or conv2d."""
        return sparse_mv(self._L_i, V)

    def apply_L_e(self, V):
        return sparse_mv(self._L_e, V)

    def apply_L_ie(self, V):
        if self._L_ie is None:
            self._L_ie = (self._L_i + self._L_e).coalesce()
        return sparse_mv(self._L_ie, V)

    def get_parabolic_operators(self, dt, theta=0.5):
        """Build A_lhs, B_rhs for: (chi*Cm/dt * I - theta*L_i) * Vm = rhs"""
        I = speye(self.n_dof)
        A_lhs = (self._chi_Cm / dt * I - theta * self._L_i).coalesce()
        B_rhs = (self._chi_Cm / dt * I + (1-theta) * self._L_i).coalesce()
        return A_lhs, B_rhs

    def get_elliptic_operator(self):
        """Build -(L_i + L_e) for: -(L_i+L_e) * phi_e = L_i * Vm"""
        if self._L_ie is None:
            self._L_ie = (self._L_i + self._L_e).coalesce()
        return (-self._L_ie).coalesce()

    def _build_laplacian(self, D_scalar, D_field, variable: str):
        """Build 9-pt stencil Laplacian. BC-aware: reads grid.boundary_spec per variable.
        See improvement.md § FDM Stencil at Boundary Nodes for pseudocode."""
        # Same core logic as V5.4 fdm.py, but stencil at boundary nodes
        # depends on BC type: Neumann → ghost-node mirror, Dirichlet → eliminate
        pass
```

### Phase 3: Linear Solvers (Three Tiers)

#### `cardiac_sim/simulation/classical/solver/linear_solver/pcg.py` -- COPY from V5.4
Direct copy. Used for both parabolic and elliptic sub-solves.

#### `cardiac_sim/simulation/classical/solver/linear_solver/spectral.py` -- NEW
```python
class SpectralSolver(LinearSolver):
    """Tier 1: Direct spectral solve for constant-coefficient Laplacian.

    Selects transform based on bc_type (physics name):
    - 'neumann'   -> DCT-II/III (Type 2 forward, Type 3 inverse)
    - 'dirichlet' -> DST via FFT odd-extension
    - 'periodic'  -> FFT (real-to-complex)

    O(N log N). No iterations. Only valid for isotropic, uniform-grid problems
    with homogeneous BCs of a single type on all edges.

    NOTE: bc_type uses physics names ('neumann', 'dirichlet', 'periodic'),
    NOT transform names ('dct', 'dst', 'fft'). This matches improvement.md.
    """

    def __init__(self, Nx, Ny, dx, dy, D, bc_type='neumann', device='cpu'):
        self._eigenvalues = self._compute_eigenvalues(Nx, Ny, dx, dy, D, bc_type)
        self._bc_type = bc_type

    def solve(self, A_unused, b):
        b_2d = b.reshape(self._Nx, self._Ny)
        if self._bc_type == 'neumann':
            b_hat = torch.fft.dctn(b_2d, type=2, norm='ortho')
            x_hat = b_hat / self._eigenvalues
            return torch.fft.dctn(x_hat, type=3, norm='ortho').reshape(-1)
        elif self._bc_type == 'dirichlet':
            # DST via FFT odd-extension
            b_ext = self._odd_extend(b_2d)
            b_hat = torch.fft.rfft2(b_ext)
            x_hat = b_hat / self._eigenvalues_ext
            x_ext = torch.fft.irfft2(x_hat)
            return x_ext[1:self._Nx+1, 1:self._Ny+1].reshape(-1)
        else:  # periodic
            b_hat = torch.fft.rfft2(b_2d)
            x_hat = b_hat / self._eigenvalues
            return torch.fft.irfft2(x_hat, s=(self._Nx, self._Ny)).reshape(-1)
```

#### `cardiac_sim/simulation/classical/solver/linear_solver/pcg_spectral.py` -- NEW
```python
class PCGSpectralSolver(LinearSolver):
    """Tier 2: PCG with spectral preconditioner.

    For anisotropic conductivity on structured grids. The spectral solver
    (DCT/DST/FFT of the isotropic approximation) serves as preconditioner.
    Typically converges in 1-3 iterations.
    """

    def __init__(self, Nx, Ny, dx, dy, D_avg, bc_type='neumann', tol=1e-6, device='cpu'):
        self._precond = SpectralSolver(Nx, Ny, dx, dy, D_avg, bc_type, device)
        self._tol = tol
```

#### `cardiac_sim/simulation/classical/solver/linear_solver/multigrid.py` -- NEW
```python
class GeometricMultigridPreconditioner:
    """Geometric multigrid V-cycle preconditioner for FDM structured grids.

    NOT a standalone solver — used inside PCG (EllipticPCGMGSolver).
    V-cycle with:
    - Smoother: weighted Jacobi (omega=2/3)
    - Restriction: full-weighting (2D averaging)
    - Prolongation: bilinear interpolation
    - Coarse solve: direct (dense, small grid)

    O(N) per V-cycle. Works for any anisotropy, any coefficient field.
    """

    def __init__(self, nx, ny, n_levels=4, n_smooth=3, smoother='jacobi'):
        self.n_levels = min(n_levels, self._max_levels(nx, ny))
        self.n_smooth = n_smooth
        self._grids = []  # [(nx_l, ny_l) for each level]
        self._A_levels = [None] * self.n_levels

    def setup(self, A_fine):
        """Build coarse-level operators via Galerkin: A_c = R * A_f * P"""
        self._A_levels[0] = A_fine
        for level in range(self.n_levels - 1):
            R = self._build_restriction(level)
            P = self._build_prolongation(level)
            self._A_levels[level + 1] = (R @ self._A_levels[level] @ P).coalesce()

    def apply(self, r):
        """Apply one V-cycle: z ≈ A^{-1} r"""
        return self._v_cycle(r, level=0)

    def _v_cycle(self, b, level):
        if level == self.n_levels - 1:
            return self._coarse_solve(b, level)
        A = self._A_levels[level]
        x = torch.zeros_like(b)
        x = self._smooth(A, b, x, self.n_smooth)     # Pre-smooth
        r = b - sparse_mv(A, x)
        r_coarse = self._restrict(r, level)
        e_coarse = self._v_cycle(r_coarse, level + 1) # Recurse
        x = x + self._prolongate(e_coarse, level)      # Correct
        x = self._smooth(A, b, x, self.n_smooth)      # Post-smooth
        return x
```

#### `cardiac_sim/simulation/classical/solver/linear_solver/pcg_gmg.py` -- NEW
```python
class EllipticPCGMGSolver(LinearSolver):
    """Tier 3: PCG with geometric multigrid preconditioner.

    General-purpose solver for any anisotropy, any coefficient variation,
    any BC combination. O(N) per iteration via GMG V-cycle.
    Convergence: 10-25 PCG iterations.
    """

    def __init__(self, nx, ny, n_levels=4, max_iters=50, tol=1e-8):
        self.mg = GeometricMultigridPreconditioner(nx, ny, n_levels)
        self.max_iters = max_iters
        self.tol = tol
        self._setup_done = False

    def solve(self, A, b):
        if not self._setup_done:
            self.mg.setup(A)
            self._setup_done = True
        # Standard PCG with self.mg.apply(r) as preconditioner
        ...
```

### Phase 4: Bidomain Diffusion Solver

#### `cardiac_sim/simulation/classical/solver/diffusion_stepping/decoupled.py` -- NEW
```python
class DecoupledBidomainDiffusionSolver:
    """Decoupled (partitioned) bidomain diffusion solver.

    Splits the coupled parabolic-elliptic system into:
    1. Parabolic solve for Vm (N x N, SPD)
    2. Elliptic solve for phi_e (N x N, SPD)

    Each sub-solve uses PCG (or multigrid) independently.
    This is the standard GPU-friendly approach used by all
    production bidomain solvers.

    Coupling is handled via operator splitting:
    - phi_e from previous step used in parabolic RHS
    - Updated Vm used in elliptic RHS
    """

    def __init__(self, spatial: BidomainFDMDiscretization, dt: float,
                 parabolic_solver: LinearSolver,
                 elliptic_solver: LinearSolver,
                 theta: float = 0.5,
                 pin_node: int = 0):
        self.spatial = spatial
        self.dt = dt
        self.theta = theta

        # Read BCs from mesh — no string parameters
        bc = spatial.grid.boundary_spec
        self._needs_pinning = bc.phi_e_has_null_space

        # Build parabolic operators (A_lhs * Vm = B_rhs * Vm_old + coupling)
        self.A_para, self.B_para = spatial.get_parabolic_operators(dt, theta)

        # Build elliptic operator (A_ellip * phi_e = rhs)
        self.A_ellip = spatial.get_elliptic_operator()
        if self._needs_pinning:
            self.pin_node = pin_node
            self._apply_pinning(self.A_ellip, pin_node)

        self.parabolic_solver = parabolic_solver
        self.elliptic_solver = elliptic_solver

    def step(self, state: BidomainState, dt: float) -> None:
        """Advance Vm and phi_e by one diffusion time step."""

        # --- Step 1: Parabolic solve for Vm ---
        # RHS = B_para * Vm + theta * L_i * phi_e_old
        rhs_para = sparse_mv(self.B_para, state.Vm) \
                   + self.theta * self.spatial.apply_L_i(state.phi_e)

        Vm_new = self.parabolic_solver.solve(self.A_para, rhs_para)

        # --- Step 2: Elliptic solve for phi_e ---
        # -(L_i + L_e) * phi_e = L_i * Vm_new
        rhs_ellip = self.spatial.apply_L_i(Vm_new)
        if self._needs_pinning:
            rhs_ellip[self.pin_node] = 0.0

        phi_e_new = self.elliptic_solver.solve(self.A_ellip, rhs_ellip)

        # --- Update state ---
        state.Vm.copy_(Vm_new)
        state.phi_e.copy_(phi_e_new)
```

### Phase 5: Orchestration

#### `cardiac_sim/simulation/classical/bidomain.py` -- NEW
```python
class BidomainSimulation:
    """Top-level orchestrator for bidomain simulations.

    Follows V5.4's MonodomainSimulation pattern:
    - String-based config for solver selection
    - Factory functions for construction
    - Generator-based run loop
    """

    def __init__(self, spatial, ionic_model, stimulus, dt,
                 splitting="strang",
                 ionic_solver="rush_larsen",
                 diffusion_solver="decoupled",
                 parabolic_solver="pcg",
                 elliptic_solver="auto",
                 theta=0.5):

        # Build state
        x, y = spatial.coordinates
        Vm_init = torch.full((spatial.n_dof,), ionic_model.V_rest, ...)
        phi_e_init = torch.zeros(spatial.n_dof, ...)
        ionic_states = ionic_model.get_initial_state(spatial.n_dof)

        self.state = BidomainState(...)

        # Auto-select elliptic solver from BoundarySpec if "auto"
        if elliptic_solver == "auto":
            elliptic_solver = self._auto_select_elliptic_solver(spatial)

        # Build solver chain
        para_solver = _build_linear_solver(parabolic_solver)
        ellip_solver = _build_linear_solver(elliptic_solver, spatial=spatial)
        diffusion = DecoupledBidomainDiffusionSolver(
            spatial, dt, para_solver, ellip_solver, theta
        )
        ionic = _build_ionic_solver(ionic_solver, ionic_model)
        self.splitting = _build_splitting(splitting, ionic, diffusion)

    @staticmethod
    def _auto_select_elliptic_solver(spatial):
        """Read BoundarySpec from mesh and pick best solver tier."""
        bc = spatial.grid.boundary_spec
        is_isotropic = (spatial.conductivity.D_i_field is None
                        and spatial.conductivity.theta is None)

        if bc.phi_e_spectral_eligible and is_isotropic:
            return "spectral"       # Tier 1: O(N log N), no iterations
        elif bc.phi_e_spectral_eligible:
            return "pcg_spectral"   # Tier 2: PCG + spectral precond, 1-3 iters
        else:
            return "pcg_gmg"        # Tier 3: PCG + GMG, 10-25 iters

    def run(self, t_end, save_every=1.0, callback=None):
        """Run simulation. Yields state at save points."""
        # Same pattern as V5.4
        ...
```

### Phase 6: Boundary Speedup Validation

#### `examples/boundary_speedup.py`
```python
"""Validate Kleber boundary speedup effect.

Four configurations (from BOUNDARY_SPEEDUP_ANALYSIS.md):
A: Monodomain, uniform D_eff, Neumann BC -> CV ratio = 1.00
B: Monodomain, enhanced D near boundary, Neumann BC -> CV ratio ~ 1.13
C: Monodomain, uniform D_eff, Dirichlet BC -> CV ratio < 1.00 (slowdown)
D: Bidomain FDM, Intra-Neumann + Extra-Dirichlet -> CV ratio ~ 1.13

Config D is the gold standard. Config B is the monodomain approximation.
Both should show ~13% CV increase at tissue-bath boundary.
"""
```

---

## Implementation Order (FDM-Focused)

| Phase | What | Files | Tests | Depends On |
|-------|------|-------|-------|------------|
| 1 | Foundation: copy ionic, mesh, stim, create BoundarySpec + state | ~13 | 6 | Nothing |
| 2 | FDM discretization: L_i, L_e stencils, BC-aware stencils | 2 | 6 | Phase 1 |
| 3 | Linear solvers: spectral (DCT/DST/FFT), PCG+spectral, PCG+GMG | 5 | 7 | Phase 2 |
| 4 | Diffusion solver: decoupled parabolic + elliptic, BC-driven null space | 2 | 7 | Phase 2, 3 |
| 5 | Orchestration: BidomainSimulation, splitting, auto-solver selection | 4 | 8 | Phase 1-4 |
| 6 | Boundary speedup validation (Kleber effect, DST for bath BCs) | 2 | 4 | Phase 5 |
| **Total** | | **~28 files** | **~38 tests** | |

### What's Deferred

- FEM discretization (needs unstructured mesh support, less GPU-friendly)
- FVM discretization (similar to FDM, can add later)
- Coupled MINRES solver (overkill for most applications)
- Block preconditioners (not needed with decoupled approach)
- emRKC fully explicit solver (no linear solves, ~30-50 stages/step)
- LBM bidomain (future exploratory work)
- Builder integration (after core solver works)
- GPU-specific optimizations (torch.compile, temporal blocking)

---

## Key Design Decisions

### 1. Decoupled Over Coupled (Literature-Driven)
Every GPU bidomain implementation in production decouples the system. The coupled MINRES approach is academically correct but unnecessary overhead for FDM on GPU. We implement decoupled first, add coupled as option later.

### 2. Matrix-Free Stencil Application Where Possible
For explicit operations (apply_L_i, apply_L_e), use `sparse_mv` initially but design for future `conv2d`-based matrix-free application. The FDM stencil is regular and maps directly to convolution.

### 3. Separate Parabolic and Elliptic Solvers
The parabolic and elliptic sub-problems have different character:
- **Parabolic (Vm):** Well-conditioned (chi*Cm/dt dominates), PCG converges in 10-15 iters
- **Elliptic (phi_e):** Ill-conditioned (pure Laplacian), needs multigrid for O(N) solve
Allow different solver choices for each.

### 4. Reuse V5.4 IonicSolver Without Changes
The ionic solver operates on `state.Vm` which has the same semantics in both monodomain and bidomain. No changes needed.

### 5. Phi_e Pinning is BC-Dependent
Null space exists only when ALL phi_e edges are Neumann. The `DecoupledBidomainDiffusionSolver`
reads `spatial.grid.boundary_spec.phi_e_has_null_space` to decide whether to pin. When any
edge has Dirichlet (bath-coupled), no null space exists and no pinning is needed.

### 6. BoundarySpec Protocol — BCs Encoded in Mesh
Boundary conditions are stored in `StructuredGrid.boundary_spec` (a `BoundarySpec` dataclass),
not passed as string parameters through the solver chain. Each downstream component reads BCs
from the mesh:
- **Discretization:** Reads BCs to construct Neumann (ghost-node mirror) or Dirichlet (identity row) stencils
- **SpectralSolver:** Reads `boundary_spec.spectral_transform` to select DCT (Neumann) or DST (Dirichlet)
- **DiffusionSolver:** Reads `boundary_spec.phi_e_has_null_space` for pinning decision
- **BidomainSimulation:** Reads `boundary_spec` for auto-solver selection

Factory methods: `BoundarySpec.insulated()`, `BoundarySpec.bath_coupled()`,
`BoundarySpec.bath_coupled_edges(bath_edges)`.

### 7. Three-Tier Elliptic Solver with Auto-Selection
| Tier | Solver | When | Iterations |
|------|--------|------|------------|
| 1 | SpectralSolver (DCT/DST/FFT) | Isotropic, uniform BCs | 0 (direct) |
| 2 | PCG + Spectral preconditioner | Anisotropic, uniform BCs | 1-3 |
| 3 | PCG + Geometric Multigrid | Mixed BCs or general | 10-25 |

Auto-selection reads `BoundarySpec.phi_e_spectral_eligible` and conductivity isotropy.
For Kleber validation (Dirichlet on all edges), Tier 1 with DST is selected automatically.

### 8. Boundary Conditions for Kleber Effect
Support two BC configurations at tissue edges:
- **Insulated (standard):** Neumann on both Vm and phi_e → DCT → no boundary speedup
- **Bath-coupled (Kleber):** Vm=Neumann, phi_e=Dirichlet → DST → ~13% CV increase
The bath-coupled BC is what produces the boundary speedup. DST (not DCT) is required
because phi_e=0 is a Dirichlet condition.
