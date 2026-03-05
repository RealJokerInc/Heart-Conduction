# Monodomain-to-Bidomain Architecture Extension

## Overview

This document specifies the architecture for extending the Monodomain Engine V5.4 to a full Bidomain solver. The design preserves V5.4's modular separation of concerns while introducing:

- **Dual-potential state** (Vm + phi_e instead of single V)
- **Decoupled solver** — two sequential N×N SPD solves (parabolic Vm + elliptic phi_e), not coupled 2N×2N
- **Three-tier elliptic strategy** — Spectral direct (DCT/DST/FFT), PCG+Spectral, PCG+GMG
- **BoundarySpec protocol** — per-edge, per-variable BC encoding stored in the mesh
- **Extracellular potential pinning** (null space treatment for phi_e, only when all-Neumann)
- **Dual-conductivity spatial discretization** (L_i and L_e from D_i and D_e)

---

## Current Architecture (V5.4 Monodomain)

```
cardiac_sim/
|
+-- ionic/                   # Cellular ionic models
|   +-- base.py              # IonicModel ABC (data provider only)
|   +-- ord/                 # O'Hara-Rudy 2011 (40 ionic states)
|   +-- ttp06/               # ten Tusscher-Panfilov 2006 (18 ionic states)
|
+-- tissue_builder/          # Builders (create data at init)
|   +-- mesh/                # Geometry structures
|   +-- tissue/              # Material properties
|   +-- stimulus/            # Pacing protocols
|
+-- simulation/
|   +-- classical/
|   |   +-- state.py                    # SimulationState (V, ionic_states)
|   |   +-- monodomain.py              # Orchestrator
|   |   +-- discretization_scheme/     # FEM/FDM/FVM spatial operators
|   |   +-- solver/
|   |       +-- splitting/             # Godunov, Strang
|   |       +-- ionic_time_stepping/   # RushLarsen, ForwardEuler
|   |       +-- diffusion_time_stepping/
|   |           +-- explicit/          # FE, RK2, RK4
|   |           +-- implicit/          # CN, BDF1, BDF2
|   |           +-- linear_solver/     # PCG, Chebyshev, FFT/DCT
|   +-- lbm/                           # Lattice-Boltzmann alternative
|
+-- utils/
    +-- backend.py                     # CPU/GPU device abstraction
```

### Monodomain Architecture Issues (for Bidomain Extension)

1. **Single-potential state** -- SimulationState stores only V; bidomain needs Vm and phi_e
2. **Single conductivity** -- SpatialDiscretization takes one D tensor; bidomain needs D_i and D_e
3. **Single PDE** -- DiffusionSolver solves one parabolic equation; bidomain has parabolic + elliptic
4. **No boundary condition protocol** -- V5.4 hardcodes Neumann; bidomain needs per-edge, per-variable BCs
5. **No null space handling** -- phi_e is determined only up to a constant (pure Neumann)
6. **No elliptic solver** -- V5.4 has no dedicated elliptic solver tier (spectral/GMG)

---

## Proposed Architecture (V1 Bidomain — FDM-First, Decoupled)

```
cardiac_sim/
|
+-- ionic/                                   # DIRECT COPY from V5.4
|   +-- base.py                              # IonicModel ABC (data provider only)
|   +-- lut.py                               # Lookup table acceleration
|   +-- ord/                                 # O'Hara-Rudy 2011 (40 ionic states)
|   +-- ttp06/                               # ten Tusscher-Panfilov 2006 (18 ionic states)
|
+-- tissue_builder/                          # MOSTLY COPIED from V5.4
|   +-- mesh/
|   |   +-- base.py                          # Mesh ABC
|   |   +-- structured.py                    # StructuredGrid (extended with BoundarySpec)
|   |   +-- boundary.py                      # NEW: BoundarySpec, BCType, EdgeBC
|   +-- tissue/
|   |   +-- isotropic.py                     # V5.4 compat
|   |   +-- conductivity.py                  # NEW: BidomainConductivity (D_i, D_e)
|   +-- stimulus/
|       +-- protocol.py                      # StimulusProtocol
|       +-- regions.py                       # Stimulus regions
|
+-- simulation/
|   +-- classical/
|   |   +-- state.py                         # BidomainState (Vm, phi_e, ionic_states)
|   |   +-- bidomain.py                      # BidomainSimulation (orchestrator)
|   |   |
|   |   +-- discretization/                  # Spatial operators
|   |   |   +-- base.py                      # BidomainDiscretization ABC
|   |   |   +-- fdm.py                       # FDM: L_i, L_e 9-pt stencils
|   |   |
|   |   +-- solver/
|   |       +-- splitting/
|   |       |   +-- base.py                  # SplittingStrategy ABC
|   |       |   +-- strang.py                # Strang: half-ionic -> diff -> half-ionic
|   |       |   +-- godunov.py               # Godunov: ionic -> diff
|   |       |
|   |       +-- ionic_stepping/              # COPIED from V5.4
|   |       |   +-- base.py                  # IonicSolver ABC
|   |       |   +-- rush_larsen.py           # Rush-Larsen
|   |       |   +-- forward_euler.py         # Forward Euler
|   |       |
|   |       +-- diffusion_stepping/          # NEW: bidomain-specific
|   |       |   +-- base.py                  # BidomainDiffusionSolver ABC
|   |       |   +-- decoupled.py             # Decoupled: parabolic + elliptic
|   |       |
|   |       +-- linear_solver/               # Extended from V5.4
|   |           +-- base.py                  # LinearSolver ABC (from V5.4)
|   |           +-- pcg.py                   # PCG + Jacobi (from V5.4)
|   |           +-- chebyshev.py             # Chebyshev (from V5.4)
|   |           +-- spectral.py              # Tier 1: SpectralSolver (FFT/DCT/DST unified)
|   |           +-- pcg_spectral.py          # Tier 2: PCG + spectral preconditioner
|   |           +-- multigrid.py             # GeometricMultigridPreconditioner (NEW)
|   |           +-- pcg_gmg.py               # Tier 3: PCG + geometric multigrid
|   |
|   +-- lbm/                                 # FUTURE
|       +-- __init__.py
|
+-- utils/
|   +-- backend.py                           # Device abstraction (from V5.4)
|
+-- tests/
```

---

## Top-Level Structure

```
cardiac_sim/
|
+-- ionic/                  # WHAT the cell computes (physics) — UNCHANGED
|
+-- tissue_builder/         # WHAT to build (geometry, materials, protocols)
|
+-- simulation/
|   +-- classical/
|   |   +-- discretization/          # HOW space is discretized (returns L_i, L_e)
|   |   +-- solver/                  # HOW time is advanced
|   |   |   +-- splitting/           # WHEN to call ionic vs diffusion
|   |   |   +-- ionic_stepping/      # HOW to advance cell ODEs
|   |   |   +-- diffusion_stepping/  # HOW to advance bidomain PDE (decoupled)
|   |   |   +-- linear_solver/       # HOW to solve N x N SPD sub-problems
|   |   +-- state.py                 # WHAT runtime data we store
|   |   +-- bidomain.py              # Orchestrator
|   +-- lbm/                         # Alternative paradigm (future)
|
+-- utils/                  # WHERE to compute (CPU/GPU)
```

Each folder has one job:

| Folder | Responsibility | When Used |
|--------|----------------|-----------|
| `ionic/` | Cell model equations | Shared by both paradigms |
| `tissue_builder/` | Create geometry, stimulus | Init time only |
| `discretization/` | Build spatial operators (L_i, L_e) | Init time |
| `solver/splitting/` | Decides ionic/diffusion call order | Runtime |
| `solver/ionic_stepping/` | Advances ionic ODEs | Runtime |
| `solver/diffusion_stepping/` | Advances bidomain PDE (decoupled) | Runtime |
| `solver/linear_solver/` | Solves N × N SPD sub-problems (PCG/DCT/GMG) | Runtime |
| `state.py` | Store all runtime data | Runtime |
| `lbm/` | Self-contained dual-lattice LBM alternative (future) | Runtime |
| `utils/` | Device management | Throughout |

---

## Key Design: Monodomain vs Bidomain Differences

### What Changes

| Aspect | Monodomain (V5.4) | Bidomain (V1) |
|--------|-------------------|---------------|
| State fields | V (n_dof,) | Vm (n_dof,), phi_e (n_dof,) |
| Spatial operators | M, L (one Laplacian) | M, L_i, L_e (two Laplacians) |
| Diffusion step | 1 parabolic solve (N×N, SPD) | 2 decoupled solves: parabolic Vm + elliptic phi_e (each N×N, SPD) |
| Elliptic solver | None | Three tiers: Spectral / PCG+Spectral / PCG+GMG |
| Boundary conditions | Hardcoded Neumann | BoundarySpec: per-edge, per-variable (Neumann/Dirichlet) |
| Null space | None | phi_e up to constant (only when all-Neumann) |
| Conductivity input | Single D | (D_i, D_e) pair via BidomainConductivity |
| Stimulus | I_stim (single) | I_stim_i, I_stim_e (intracellular, extracellular) |

### What Stays the Same

| Component | Why Unchanged |
|-----------|---------------|
| IonicModel ABC | Cell physics identical; Iion(Vm, w) same equation |
| IonicSolver (rush_larsen, forward_euler) | ODE integration unchanged; operates on Vm only |
| Mesh classes | Geometry is model-independent |
| StimulusProtocol | Stimulus regions unchanged (amplitude may split) |
| Backend/device | Hardware abstraction unchanged |

---

## Physical Equations

### Bidomain PDE System

**Parabolic (transmembrane potential evolution):**
```
chi * Cm * dVm/dt = -chi * Iion(Vm, w) + div(D_i * grad(Vm)) + div(D_i * grad(phi_e)) + Istim_i
```

**Elliptic (extracellular potential constraint):**
```
0 = div((D_i + D_e) * grad(phi_e)) + div(D_i * grad(Vm)) + Istim_e
```

**Ionic dynamics (ODEs):**
```
dw/dt = f(Vm, w)
```

### Operator Splitting Decomposition

With operator splitting, the system is decomposed into:

**Reaction step** (local, ionic only):
```
chi * Cm * dVm/dt = -chi * Iion(Vm, w) + Istim_i
dw/dt = f(Vm, w)
```

**Diffusion step** (global, coupled parabolic-elliptic):
```
chi * Cm * dVm/dt = div(D_i * grad(Vm)) + div(D_i * grad(phi_e))
0 = div((D_i + D_e) * grad(phi_e)) + div(D_i * grad(Vm))
```

The reaction step is identical to monodomain. The diffusion step is the new bidomain-specific computation.

---

## Conductivity Input

### BidomainConductivity

```python
@dataclass
class BidomainConductivity:
    """Paired intracellular and extracellular conductivity tensors."""

    # Scalar isotropic (sigma / (chi * Cm), with chi=1400, Cm=1.0)
    D_i: float = 0.00124     # Intracellular (sigma_i=1.74 mS/cm -> 1.74/1400 cm^2/ms)
    D_e: float = 0.00446     # Extracellular (sigma_e=6.25 mS/cm -> 6.25/1400 cm^2/ms)

    # OR anisotropic (per-node)
    D_i_field: Optional[Tuple[Tensor, Tensor, Tensor]] = None  # (Dxx_i, Dyy_i, Dxy_i)
    D_e_field: Optional[Tuple[Tensor, Tensor, Tensor]] = None  # (Dxx_e, Dyy_e, Dxy_e)

    # OR fiber-based
    D_i_fiber: float = 0.003     # Along fibers
    D_i_cross: float = 0.0003   # Across fibers
    D_e_fiber: float = 0.002     # Along fibers
    D_e_cross: float = 0.001     # Across fibers
    theta: Optional[Tensor] = None  # Fiber angle field (Nx, Ny)
```

### Conductivity to Tensor Conversion

For fiber-based input, the tensor components are:
```
D_xx = D_fiber * cos^2(theta) + D_cross * sin^2(theta)
D_yy = D_fiber * sin^2(theta) + D_cross * cos^2(theta)
D_xy = (D_fiber - D_cross) * cos(theta) * sin(theta)
```

This applies independently for D_i and D_e, producing K_i and K_e matrices.

---

## Boundary Condition Protocol

### The Problem

V5.4 hardcodes Neumann BCs everywhere — there's nothing to configure because monodomain
always uses zero-flux at tissue boundaries. Bidomain breaks this:

| Variable | Interior (insulated edge) | Bath-coupled edge |
|----------|--------------------------|-------------------|
| Vm | Neumann (membrane insulated) | **Modified Neumann**: dV/dn = -d(phi_e)/dn |
| phi_e | Neumann (extracellular insulated) | **Dirichlet**: phi_e = 0 |

Furthermore, **different edges can have different BCs** — e.g., a tissue strip with bath
on top/bottom but insulated on left/right. The BC type also determines:
- Stencil construction at boundary nodes
- Spectral transform type (DCT/DST)
- Null space existence
- Parabolic Vm coupling at the boundary

A global string like `phi_e_bc='dirichlet'` can't handle mixed cases.

### BoundarySpec — Per-Edge, Per-Variable BC Description

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch


class BCType(Enum):
    """Boundary condition types for bidomain variables."""
    NEUMANN = "neumann"          # Zero-flux: n·D·grad(u) = 0
    DIRICHLET = "dirichlet"      # Fixed value: u = value
    # Robin BCs (future): alpha*u + beta*du/dn = gamma


class Edge(Enum):
    """Named edges of a rectangular domain."""
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


@dataclass
class EdgeBC:
    """BC specification for one variable on one edge."""
    bc_type: BCType = BCType.NEUMANN
    value: float = 0.0           # Dirichlet value (ignored for Neumann)


@dataclass
class BoundarySpec:
    """
    Complete boundary condition specification for a bidomain simulation.

    Encodes per-edge, per-variable BC types. Stored in the mesh at init time,
    read by discretization and solver layers. No BC logic leaks into strings.

    Design principles:
    - Init-time data, not runtime (BCs don't change during simulation)
    - Lives in tissue_builder/ (alongside mesh and conductivity)
    - The mesh owns it, the discretization reads it, the solver obeys it
    - Precomputes boundary masks and flags for downstream consumers

    Attributes
    ----------
    Vm : Dict[Edge, EdgeBC]
        BC for transmembrane potential on each edge.
        Default: Neumann everywhere (membrane is always insulated).
    phi_e : Dict[Edge, EdgeBC]
        BC for extracellular potential on each edge.
        Default: Neumann everywhere (fully insulated tissue).
    """
    Vm: Dict[Edge, EdgeBC] = field(default_factory=lambda: {
        Edge.LEFT: EdgeBC(), Edge.RIGHT: EdgeBC(),
        Edge.TOP: EdgeBC(), Edge.BOTTOM: EdgeBC(),
    })
    phi_e: Dict[Edge, EdgeBC] = field(default_factory=lambda: {
        Edge.LEFT: EdgeBC(), Edge.RIGHT: EdgeBC(),
        Edge.TOP: EdgeBC(), Edge.BOTTOM: EdgeBC(),
    })

    # --- Derived properties (computed once, queried many times) ---

    @property
    def phi_e_has_null_space(self) -> bool:
        """True if phi_e has a constant null space (all edges Neumann)."""
        return all(bc.bc_type == BCType.NEUMANN for bc in self.phi_e.values())

    @property
    def phi_e_uniform_bc(self) -> Optional[BCType]:
        """If all edges have the same BC type, return it. Else None."""
        types = {bc.bc_type for bc in self.phi_e.values()}
        return types.pop() if len(types) == 1 else None

    @property
    def phi_e_spectral_eligible(self) -> bool:
        """True if spectral solver (Tier 1/2) can be used for phi_e.
        Requires all edges to have the same BC type."""
        return self.phi_e_uniform_bc is not None

    @property
    def spectral_transform(self) -> Optional[str]:
        """Which spectral transform to use, or None if mixed BCs."""
        bc = self.phi_e_uniform_bc
        if bc == BCType.NEUMANN:
            return 'dct'
        elif bc == BCType.DIRICHLET:
            return 'dst'
        return None

    # --- Factory methods for common configurations ---

    @classmethod
    def insulated(cls) -> 'BoundarySpec':
        """Standard insulated tissue — Neumann everywhere on all variables.
        Use for: tissue in free space, no bath."""
        return cls()  # All defaults are Neumann

    @classmethod
    def bath_coupled(cls, bath_value: float = 0.0) -> 'BoundarySpec':
        """Bath-coupled on ALL edges — phi_e Dirichlet everywhere.
        Use for: small tissue immersed in bath (Kleber validation)."""
        return cls(
            phi_e={edge: EdgeBC(BCType.DIRICHLET, bath_value) for edge in Edge},
        )

    @classmethod
    def bath_coupled_edges(cls, bath_edges: list, bath_value: float = 0.0) -> 'BoundarySpec':
        """Partial bath — specified edges are bath-coupled, rest insulated.
        Use for: tissue strip with bath on top/bottom only."""
        phi_e = {}
        for edge in Edge:
            if edge in bath_edges or edge.value in bath_edges:
                phi_e[edge] = EdgeBC(BCType.DIRICHLET, bath_value)
            else:
                phi_e[edge] = EdgeBC(BCType.NEUMANN)
        return cls(phi_e=phi_e)
```

### StructuredGrid — Stores Boundary Masks

The mesh owns per-edge boundary node masks. These are precomputed at init time and
queried by the discretization when building stencils.

```python
@dataclass
class StructuredGrid(Mesh):
    # ... existing fields from V5.4 ...
    boundary_spec: BoundarySpec = field(default_factory=BoundarySpec.insulated)

    @property
    def edge_masks(self) -> Dict[Edge, torch.Tensor]:
        """Per-edge boolean masks (Nx, Ny). Precomputed."""
        masks = {}
        masks[Edge.LEFT] = torch.zeros(self.Nx, self.Ny, dtype=torch.bool, device=self._device)
        masks[Edge.LEFT][0, :] = True
        masks[Edge.RIGHT] = torch.zeros_like(masks[Edge.LEFT])
        masks[Edge.RIGHT][-1, :] = True
        masks[Edge.BOTTOM] = torch.zeros_like(masks[Edge.LEFT])
        masks[Edge.BOTTOM][:, 0] = True
        masks[Edge.TOP] = torch.zeros_like(masks[Edge.LEFT])
        masks[Edge.TOP][:, -1] = True
        return masks

    @property
    def dirichlet_mask_phi_e(self) -> torch.Tensor:
        """Combined mask of all Dirichlet-BC nodes for phi_e. (Nx, Ny) bool."""
        mask = torch.zeros(self.Nx, self.Ny, dtype=torch.bool, device=self._device)
        edge_masks = self.edge_masks
        for edge, bc in self.boundary_spec.phi_e.items():
            if bc.bc_type == BCType.DIRICHLET:
                mask |= edge_masks[edge]
        return mask

    @property
    def neumann_mask_phi_e(self) -> torch.Tensor:
        """Combined mask of all Neumann-BC nodes for phi_e."""
        return self.boundary_mask & ~self.dirichlet_mask_phi_e
```

### How Each Layer Consumes BoundarySpec

```
Layer                  What it reads                What it does
─────────────────────  ───────────────────────────  ────────────────────────────────
StructuredGrid         boundary_spec                Precomputes edge_masks,
                                                    dirichlet_mask_phi_e

BidomainFDMDiscret.    grid.boundary_spec           Builds L_i stencil (always Neumann)
                       grid.dirichlet_mask_phi_e    Builds L_e stencil:
                                                      Neumann edges → ghost-node mirror
                                                      Dirichlet edges → eliminate node
                                                    Builds elliptic operator:
                                                      Dirichlet rows → identity (A[k,k]=1)

SpectralSolver         boundary_spec.spectral_      Selects DCT/DST/FFT transform
                       transform

DecoupledDiffusion     boundary_spec.phi_e_has_     Enables/disables null space pinning
                       null_space

BidomainSimulation     boundary_spec.phi_e_         Auto-selects solver tier:
                       spectral_eligible              uniform BC → spectral eligible
                                                      mixed BC → must use PCG+GMG
```

### FDM Stencil at Boundary Nodes

**IMPLEMENTED: Symmetric face-based stencil (differs from original spec)**

The original spec described V5.4's ghost-node mirror for Neumann BCs and ghost-node
Dirichlet elimination. During implementation, this was found to produce **asymmetric
Laplacian matrices** at boundary nodes:

```
Ghost-node mirror at i=N-1:
  L[N-1, N-2] = 2D/dx²  (doubled by mirror)
  L[N-2, N-1] =  D/dx²  (normal interior stencil)
  → L is asymmetric
```

For the monodomain parabolic solve (A = chi_Cm/dt * I - theta*L), the identity term
dominates by ~10^6, masking the asymmetry. But the bidomain elliptic operator
A_ellip = -(L_i + L_e) has **no identity term** — the asymmetry makes the matrix
non-SPD, causing PCG to fail with negative eigenvalues.

**Solution: face-based stencil.** Each interior face contributes symmetrically to
both adjacent nodes. Out-of-domain faces are skipped (zero flux = Neumann).

```python
class BidomainFDMDiscretization:
    def _build_laplacian(self, Dxx, Dxy, Dyy):
        """Build symmetric Laplacian — face-based, no ghost nodes."""
        for i, j in all_active_nodes:
            for neighbor in cardinal_neighbors(i, j):
                if neighbor_is_active(ni, nj):
                    D_face = harmonic_mean(D[i,j], D[ni,nj])
                    add_to_stencil(k, neighbor_idx, D_face / dx²)
                    center -= D_face / dx²
                # else: out of domain → skip (no flux = Neumann)
```

Properties of the resulting matrix:
- **Symmetric**: L[i,j] = L[j,i] (same face contributes to both nodes)
- **Zero row sum**: conservation guaranteed
- **Negative semi-definite**: all eigenvalues ≤ 0
- **Interior accuracy**: O(h²) truncation error
- **Boundary accuracy**: stiffness form gives d²u/dx² / 2 at boundary nodes
  (half control volume), but this cancels in the bidomain LHS vs RHS

**Dirichlet enforcement** is NOT baked into L_i or L_e. Both are built identically
(face-based, Neumann everywhere). Dirichlet is applied only in
`get_elliptic_operator()` via symmetric row+column elimination:

```python
def get_elliptic_operator(self):
    A = -(L_i + L_e)
    # For each Dirichlet boundary node of phi_e:
    #   zero row AND column, set diagonal = 1
    A = self._enforce_dirichlet(A)
    return A  # symmetric, SPD for Dirichlet; symmetric, PSD for Neumann
```

### User API with BoundarySpec

```python
from cardiac_sim.tissue_builder.mesh import StructuredGrid, BoundarySpec, Edge

# Kleber validation: bath on all edges
grid = StructuredGrid.create_rectangle(Lx=2.0, Ly=0.5, Nx=800, Ny=200)
grid.boundary_spec = BoundarySpec.bath_coupled()

# Strip with bath on top/bottom only, insulated on left/right
grid.boundary_spec = BoundarySpec.bath_coupled_edges([Edge.TOP, Edge.BOTTOM])

# Fully insulated (standard, default)
grid.boundary_spec = BoundarySpec.insulated()

# Custom: Dirichlet on right edge only (e.g., defibrillation electrode)
from cardiac_sim.tissue_builder.mesh import EdgeBC, BCType
grid.boundary_spec = BoundarySpec()
grid.boundary_spec.phi_e[Edge.RIGHT] = EdgeBC(BCType.DIRICHLET, value=-10.0)

# The simulation reads BCs from the grid — no BC strings in solver config
sim = BidomainSimulation(
    spatial=BidomainFDMDiscretization(grid, conductivity, chi=1400.0, Cm=1.0),
    ionic_model="ttp06",
    stimulus=stimulus,
    dt=0.02,
    # No phi_e_bc parameter needed — it's in the grid
)
```

### Mixed BC Implications for Solver Selection

When BCs are mixed (some edges Dirichlet, some Neumann), the spectral solver cannot
be used directly because DCT/DST assume uniform BCs on all edges.

```
boundary_spec.phi_e_spectral_eligible?
  YES (all edges same BC)
    → phi_e_uniform_bc == NEUMANN  → Tier 1: SpectralSolver(bc_type='neumann')  [DCT]
    → phi_e_uniform_bc == DIRICHLET → Tier 1: SpectralSolver(bc_type='dirichlet') [DST]
  NO (mixed BCs)
    → Must use Tier 3: PCG + GMG (handles any BC combination)
    → OR: Tier 2 with spectral preconditioner (approximate, use isotropic part)
```

The BidomainSimulation factory reads `boundary_spec` and auto-selects the best solver:

```python
def _auto_select_elliptic_solver(spatial, grid):
    """Auto-select elliptic solver based on BCs and conductivity."""
    bs = grid.boundary_spec

    if bs.phi_e_spectral_eligible and conductivity_is_isotropic(spatial):
        # Best case: direct spectral solve
        bc = 'dirichlet' if bs.phi_e_uniform_bc == BCType.DIRICHLET else 'neumann'
        return SpectralSolver(grid.Nx, grid.Ny, grid.dx, grid.dy,
                              D=spatial.D_sum, bc_type=bc)

    elif bs.phi_e_spectral_eligible and conductivity_is_moderate_anisotropy(spatial):
        # Spectral preconditioner
        bc = 'dirichlet' if bs.phi_e_uniform_bc == BCType.DIRICHLET else 'neumann'
        return PCGSpectralSolver(grid.Nx, grid.Ny, grid.dx, grid.dy,
                                 D_iso=spatial.D_sum_iso, bc_type=bc)

    else:
        # General: GMG handles any BC mix
        return EllipticPCGMGSolver(grid.Nx, grid.Ny)
```

---

## BidomainState — Runtime Data Container

```python
@dataclass
class BidomainState:
    """
    Runtime simulation data for bidomain.
    Extends monodomain SimulationState with phi_e.
    """
    # === Discretization Reference ===
    spatial: BidomainSpatialDiscretization

    # === Abstract Geometry ===
    n_dof: int                       # Number of degrees of freedom
    x: torch.Tensor                  # (n_dof,) x-coordinates
    y: torch.Tensor                  # (n_dof,) y-coordinates

    # === Potentials ===
    Vm: torch.Tensor                 # (n_dof,) transmembrane potential
    phi_e: torch.Tensor              # (n_dof,) extracellular potential

    # === Ionic States ===
    ionic_states: torch.Tensor       # (n_dof, n_ionic_states)
    gate_indices: List[int]
    concentration_indices: List[int]

    # === Time ===
    t: float

    # === Stimulus ===
    stim_masks: torch.Tensor         # (n_stimuli, n_dof)
    stim_starts: torch.Tensor
    stim_durations: torch.Tensor
    stim_amplitudes_i: torch.Tensor  # Intracellular stimulus amplitudes
    stim_amplitudes_e: torch.Tensor  # Extracellular stimulus amplitudes

    # === Output ===
    output_buffer_Vm: torch.Tensor   # Vm snapshots
    output_buffer_phi_e: torch.Tensor  # phi_e snapshots
    buffer_idx: int

    @property
    def Vm_flat(self) -> torch.Tensor:
        return self.Vm  # Already 1D for classical path

    @property
    def phi_e_flat(self) -> torch.Tensor:
        return self.phi_e
```

**Key design decisions:**

1. **Vm and phi_e are separate 1D fields** -- parallel to V5.4's V separation from ionic_states
2. **Separate stimulus amplitudes** -- I_stim_i and I_stim_e can differ (e.g., extracellular defibrillation)
3. **Dual output buffers** -- save both Vm and phi_e for analysis/ECG
4. **State is scheme-agnostic** -- works with FEM, FDM, or FVM

---

## BidomainSpatialDiscretization ABC

```python
class BidomainSpatialDiscretization(ABC):
    """
    Bidomain spatial discretization provides two Laplacians (L_i, L_e)
    and operators for the decoupled parabolic + elliptic solves.

    The grid and its BoundarySpec are accessible via self.grid.
    """

    @property
    @abstractmethod
    def n_dof(self) -> int:
        """Number of degrees of freedom."""
        pass

    @property
    @abstractmethod
    def grid(self) -> 'StructuredGrid':
        """The mesh, including boundary_spec."""
        pass

    @property
    @abstractmethod
    def coordinates(self) -> Tuple[Tensor, Tensor]:
        """(x, y) coordinates for stimulus evaluation."""
        pass

    @property
    @abstractmethod
    def mass_type(self) -> MassType:
        """IDENTITY (FDM), DIAGONAL (FVM), or SPARSE (FEM)."""
        pass

    # --- Operator application (matrix-free compatible) ---

    @abstractmethod
    def apply_L_i(self, V: Tensor) -> Tensor:
        """Apply intracellular Laplacian: L_i * V = div(D_i * grad(V))."""
        pass

    @abstractmethod
    def apply_L_e(self, V: Tensor) -> Tensor:
        """Apply extracellular Laplacian: L_e * V = div(D_e * grad(V))."""
        pass

    @abstractmethod
    def apply_L_ie(self, V: Tensor) -> Tensor:
        """Apply combined Laplacian: (L_i + L_e) * V."""
        pass

    # --- Decoupled solver operators ---

    @abstractmethod
    def get_parabolic_operators(self, dt: float, theta: float) -> Tuple[Tensor, Tensor]:
        """Build (A_para, B_para) for: A_para * Vm^{n+1} = B_para * Vm^n + coupling."""
        pass

    @abstractmethod
    def get_elliptic_operator(self) -> Tensor:
        """Build A_ellip = -(L_i + L_e) for: A_ellip * phi_e = L_i * Vm."""
        pass

    # --- Raw matrices (for preconditioner setup, etc.) ---

    @property
    @abstractmethod
    def L_i(self) -> Tensor:
        """Intracellular Laplacian matrix (sparse)."""
        pass

    @property
    @abstractmethod
    def L_e(self) -> Tensor:
        """Extracellular Laplacian matrix (sparse)."""
        pass
```

### Concrete Implementations

**FDM Bidomain (Primary):**

```python
class BidomainFDMDiscretization(BidomainSpatialDiscretization):
    """FDM bidomain on structured grid. BC-aware stencil construction."""

    def __init__(self, grid: StructuredGrid, conductivity: BidomainConductivity,
                 chi: float = 1400.0, Cm: float = 1.0):
        self._grid = grid
        self._chi_Cm = chi * Cm
        # Build two Laplacians — stencils respect grid.boundary_spec per variable
        self._L_i = self._build_laplacian(conductivity.D_i, conductivity.D_i_field, 'Vm')
        self._L_e = self._build_laplacian(conductivity.D_e, conductivity.D_e_field, 'phi_e')
        self._L_ie = None  # Lazy: L_i + L_e (built on first use)

    @property
    def grid(self): return self._grid

    def apply_L_i(self, V): return sparse_mv(self._L_i, V)
    def apply_L_e(self, V): return sparse_mv(self._L_e, V)

    def apply_L_ie(self, V):
        if self._L_ie is None:
            self._L_ie = (self._L_i + self._L_e).coalesce()
        return sparse_mv(self._L_ie, V)

    def get_parabolic_operators(self, dt, theta=0.5):
        """Build (A_para, B_para) for decoupled parabolic Vm solve.
        A_para * Vm^{n+1} = B_para * Vm^n + coupling
        """
        I = speye(self.n_dof)
        A_para = (self._chi_Cm / dt * I - theta * self._L_i).coalesce()
        B_para = (self._chi_Cm / dt * I + (1 - theta) * self._L_i).coalesce()
        return A_para, B_para

    def get_elliptic_operator(self):
        """Build A_ellip = -(L_i + L_e) for: A_ellip * phi_e = L_i * Vm."""
        if self._L_ie is None:
            self._L_ie = (self._L_i + self._L_e).coalesce()
        return (-self._L_ie).coalesce()

    def _build_laplacian(self, D_scalar, D_field, variable):
        """Build 9-pt stencil Laplacian. See FDM Stencil at Boundary Nodes section."""
        # Uses grid.boundary_spec to select Neumann/Dirichlet stencil per edge
        pass
```

**FEM Bidomain (Future):**

```python
class BidomainFEMDiscretization(BidomainSpatialDiscretization):
    """FEM bidomain on unstructured mesh. Deferred to future phase."""

    def __init__(self, mesh: TriangularMesh, D_i: float, D_e: float,
                 chi: float = 1400.0, Cm: float = 1.0):
        self._mesh = mesh
        self._chi_Cm = chi * Cm
        self._M = assemble_mass_matrix(mesh, chi, Cm)
        self._K_i = assemble_stiffness_matrix(mesh, D_i)
        self._K_e = assemble_stiffness_matrix(mesh, D_e)

    def apply_L_i(self, V): return sparse_mv(self._K_i, V)
    def apply_L_e(self, V): return sparse_mv(self._K_e, V)

    def get_parabolic_operators(self, dt, theta=0.5):
        A_para = (self._M / dt + theta * self._K_i).coalesce()
        B_para = (self._M / dt - (1 - theta) * self._K_i).coalesce()
        return A_para, B_para

    def get_elliptic_operator(self):
        return (self._K_i + self._K_e).coalesce()
```

---

## Decoupled Operators — What the Solver Receives

The decoupled approach does NOT build a coupled 2N×2N block system.
Instead, the discretization provides:

```
A_para, B_para = spatial.get_parabolic_operators(dt, theta)
    A_para: (chi*Cm/dt * I - theta*L_i)    — LHS of Vm solve (N×N, SPD)
    B_para: (chi*Cm/dt * I + (1-theta)*L_i) — RHS multiplier for Vm^n

A_ellip = spatial.get_elliptic_operator()
    A_ellip: -(L_i + L_e)                  — LHS of phi_e solve (N×N, SPD)
```

Each is N×N and SPD — solved directly by V5.4's LinearSolver (PCG/Spectral/GMG).

---

## Solver ABCs — Runtime Interfaces

### Ownership Chain (Decoupled Architecture)

```
BidomainSimulation (bidomain.py)
  |
  |  builds state, constructs solvers from config strings
  |
  +-- SplittingStrategy
        |  owns sub-solvers, decides call order
        |
        +-- IonicSolver                          (UNCHANGED from V5.4)
        |     +-- IonicModel                     (stateless physics)
        |
        +-- BidomainDiffusionSolver              (NEW)
              |  owns BidomainDiscretization + two LinearSolvers
              |
              +-- BidomainDiscretization          (L_i, L_e stencils)
              +-- LinearSolver (parabolic)        (PCG / Chebyshev / DCT)
              +-- LinearSolver (elliptic)         (DCT / PCG+DCT / PCG+GMG)
```

**Key departure from original improvement.md:** No BlockLinearSolver, no BlockPreconditioner,
no SubSolver hierarchy. The decoupled approach reduces the bidomain to TWO independent N x N
SPD solves, which reuse V5.4's LinearSolver directly. This eliminates the entire block solver
layer and is the approach used by every production GPU bidomain solver.

### SplittingStrategy ABC (Extended)

```python
class SplittingStrategy(ABC):
    """
    Owns ionic and diffusion solvers.
    Determines the order they are called.
    Works with BidomainState (Vm, phi_e).
    """
    def __init__(self, ionic_solver: IonicSolver, diffusion_solver: BidomainDiffusionSolver):
        self.ionic_solver = ionic_solver
        self.diffusion_solver = diffusion_solver

    @abstractmethod
    def step(self, state: BidomainState, dt: float) -> None:
        """Advance state by dt using operator splitting."""
        pass


class GodunovSplitting(SplittingStrategy):
    def step(self, state, dt):
        self.ionic_solver.step(state, dt)        # Updates Vm, ionic_states
        self.diffusion_solver.step(state, dt)    # Updates Vm AND phi_e


class StrangSplitting(SplittingStrategy):
    def step(self, state, dt):
        self.ionic_solver.step(state, dt / 2)
        self.diffusion_solver.step(state, dt)
        self.ionic_solver.step(state, dt / 2)
```

**Key difference from V5.4:** The diffusion solver now updates BOTH Vm and phi_e, not just V.

### IonicSolver ABC (UNCHANGED)

```python
class IonicSolver(ABC):
    """
    UNCHANGED from V5.4.
    Owns an IonicModel. Advances ionic ODEs in-place on state.

    For bidomain: operates on state.Vm (not state.V).
    The IonicSolver doesn't know about phi_e.
    """
    def __init__(self, ionic_model: IonicModel):
        self.ionic_model = ionic_model

    @abstractmethod
    def step(self, state, dt: float) -> None:
        """Advance ionic variables by dt. Modifies state.Vm and state.ionic_states."""
        pass
```

**Compatibility note:** IonicSolver accesses `state.Vm` which is the same field name in both monodomain (where `Vm = V`) and bidomain. The ionic solver has no knowledge of phi_e.

### BidomainDiffusionSolver ABC (NEW — Decoupled)

```python
class BidomainDiffusionSolver(ABC):
    """
    Solves the bidomain diffusion step.
    Updates both Vm and phi_e in-place.

    The decoupled approach splits the coupled system into two sequential
    N x N SPD solves, each handled by a standard LinearSolver (from V5.4).

    Owns:
    - BidomainDiscretization (provides L_i, L_e stencils)
    - LinearSolver for parabolic sub-problem (Vm)
    - LinearSolver for elliptic sub-problem (phi_e)
    """
    def __init__(self, spatial: BidomainSpatialDiscretization, dt: float):
        self._spatial = spatial
        self._dt = dt

    @abstractmethod
    def step(self, state: BidomainState, dt: float) -> None:
        """
        Advance diffusion by dt.
        Modifies state.Vm AND state.phi_e in-place.
        """
        pass

    def rebuild_operators(self, spatial: BidomainSpatialDiscretization, dt: float) -> None:
        """Rebuild operators when dt changes (adaptive time stepping)."""
        self._dt = dt
        self._build_operators(spatial, dt)
```

### DecoupledBidomainDiffusionSolver (Primary Implementation)

```python
class DecoupledBidomainDiffusionSolver(BidomainDiffusionSolver):
    """
    Decoupled (Gauss-Seidel splitting) bidomain diffusion solver.

    Splits the coupled parabolic-elliptic system into:
    1. Parabolic solve for Vm  (N x N, SPD) — uses PCG or spectral
    2. Elliptic solve for phi_e (N x N, SPD) — uses spectral or PCG+GMG

    This is the standard GPU-friendly approach used by ALL production
    bidomain solvers. O(dt) splitting error, acceptable for typical dt.

    Boundary conditions are read from `spatial.grid.boundary_spec` — NOT passed
    as string parameters. The BoundarySpec determines:
    - Stencil construction at boundary nodes (Neumann vs Dirichlet)
    - Whether null space pinning is needed (only for all-Neumann phi_e)
    - Which spectral transform to use (DCT/DST) if Tier 1/2 solver selected

    Parameters
    ----------
    spatial : BidomainDiscretization
        Provides apply_L_i, apply_L_e, apply_L_ie, parabolic/elliptic operators.
        Also provides grid.boundary_spec for BC information.
    dt : float
        Time step (ms)
    parabolic_solver : LinearSolver
        Solver for Vm sub-problem (PCG, Chebyshev, or spectral)
    elliptic_solver : LinearSolver
        Solver for phi_e sub-problem (spectral, PCG+spectral, or PCG+GMG)
    theta : float
        Implicitness parameter (0.5 = CN, 1.0 = BDF1)
    pin_node : int
        Node index for phi_e null space pinning (only used when needed)
    """

    def __init__(self, spatial, dt, parabolic_solver, elliptic_solver,
                 theta=0.5, pin_node=0):
        super().__init__(spatial, dt)
        self.theta = theta
        self.parabolic_solver = parabolic_solver
        self.elliptic_solver = elliptic_solver

        # Read BCs from mesh — no string parameters
        self._needs_pinning = spatial.grid.boundary_spec.phi_e_has_null_space
        self._pin_node = pin_node

        # Build parabolic operators: (chi*Cm/dt * I - theta*L_i)
        self.A_para, self.B_para = spatial.get_parabolic_operators(dt, theta)

        # Build elliptic operator: -(L_i + L_e)
        # BCs are already encoded in the stencil by the discretization
        self.A_ellip = spatial.get_elliptic_operator()

        # Null space pinning only needed when ALL phi_e edges are Neumann
        # Dirichlet BCs (bath-coupled) have no null space — phi_e is unique
        if self._needs_pinning:
            self._apply_pinning(self.A_ellip, pin_node)

    def step(self, state, dt):
        """
        Advance Vm and phi_e by one diffusion time step.

        Step 1 — Parabolic solve for Vm^{n+1}:
            (chi*Cm/dt * I - theta*L_i) * Vm^{n+1} =
                B_para * Vm^n + theta * L_i * phi_e^n

        Step 2 — Elliptic solve for phi_e^{n+1}:
            -(L_i + L_e) * phi_e^{n+1} = L_i * Vm^{n+1}
        """
        # --- Step 1: Parabolic (Vm) ---
        rhs_para = sparse_mv(self.B_para, state.Vm) \
                   + self.theta * self._spatial.apply_L_i(state.phi_e)
        Vm_new = self.parabolic_solver.solve(self.A_para, rhs_para)

        # --- Step 2: Elliptic (phi_e) ---
        rhs_ellip = self._spatial.apply_L_i(Vm_new)
        if self._needs_pinning:
            rhs_ellip[self._pin_node] = 0.0
        phi_e_new = self.elliptic_solver.solve(self.A_ellip, rhs_ellip)

        # --- Update state in-place ---
        state.Vm.copy_(Vm_new)
        state.phi_e.copy_(phi_e_new)

    def _apply_pinning(self, A, pin_node):
        """Enforce phi_e(pin_node) = 0 by modifying the elliptic matrix."""
        # Zero out row and column for pin_node, set diagonal to 1
        # This removes the null space (constant vector) from the operator
        pass
```

---

## Three-Tier Elliptic Solver Strategy

The elliptic solve for phi_e dominates bidomain compute time (60-80%).
We provide three solver tiers, ordered by increasing generality and decreasing speed.

### Tier 1: Spectral Direct Solve (Isotropic, O(N log N))

```python
class SpectralSolver(LinearSolver):
    """
    Direct spectral solve for the elliptic equation:
        -(D_i + D_e) * Laplacian(phi_e) = rhs

    Uses spectral transforms to diagonalize the constant-coefficient Laplacian.
    O(N log N) — no iterations, no preconditioner.

    ONLY valid when:
    - Conductivity is isotropic and spatially uniform
    - Grid is structured Cartesian with uniform spacing

    Supports three BC types via different transforms:
    - 'neumann'   → DCT (Discrete Cosine Transform)  — insulated boundaries
    - 'dirichlet' → DST (Discrete Sine Transform)    — bath-coupled φ_e = 0
    - 'periodic'  → FFT (Fast Fourier Transform)     — periodic domain

    The critical distinction: our Kleber boundary speedup validation uses
    DIRICHLET BCs on phi_e (bath-coupled), which requires DST, not DCT.
    Standard insulated tissue uses Neumann → DCT.

    Extends V5.4's DCTSolver + FFTSolver into a unified interface with
    DST support for bath-coupled boundaries.

    Parameters
    ----------
    nx, ny : int       Grid dimensions
    dx, dy : float     Grid spacing (cm)
    D : float          Diffusion coefficient (D_i+D_e for elliptic, D_i for parabolic)
    bc_type : str      'neumann' (DCT), 'dirichlet' (DST), or 'periodic' (FFT)
    """

    def __init__(self, nx, ny, dx, dy, D, bc_type='neumann'):
        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, dy
        self.D = D
        self.bc_type = bc_type
        self._eigenvalues = None

    def _compute_eigenvalues(self, device, dtype):
        """
        Eigenvalues of -D*Laplacian depend on BC type:

        Neumann (DCT):
            λ_{i,j} = D * [(2/dx²)(1 - cos(πi/N)) + (2/dy²)(1 - cos(πj/N))]
            Null space at (0,0): λ_{0,0} = 0

        Dirichlet (DST):
            λ_{i,j} = D * [(2/dx²)(1 - cos(π(i+1)/(N+1))) + (2/dy²)(1 - cos(π(j+1)/(N+1)))]
            All eigenvalues > 0 — NO null space (Dirichlet removes it)

        Periodic (FFT):
            λ_{i,j} = D * [4sin²(πi/N)/dx² + 4sin²(πj/N)/dy²]
            Null space at (0,0)
        """
        i_idx = torch.arange(self.nx, device=device, dtype=dtype)
        j_idx = torch.arange(self.ny, device=device, dtype=dtype)

        if self.bc_type == 'neumann':
            # DCT eigenvalues
            lam_x = (2.0 / self.dx**2) * (1.0 - torch.cos(torch.pi * i_idx / self.nx))
            lam_y = (2.0 / self.dy**2) * (1.0 - torch.cos(torch.pi * j_idx / self.ny))

        elif self.bc_type == 'dirichlet':
            # DST eigenvalues — note (i+1)/(N+1) indexing
            lam_x = (2.0 / self.dx**2) * (1.0 - torch.cos(torch.pi * (i_idx + 1) / (self.nx + 1)))
            lam_y = (2.0 / self.dy**2) * (1.0 - torch.cos(torch.pi * (j_idx + 1) / (self.ny + 1)))

        elif self.bc_type == 'periodic':
            # FFT eigenvalues
            kx = torch.fft.fftfreq(self.nx, d=self.dx, device=device, dtype=dtype) * 2 * torch.pi
            ky = torch.fft.fftfreq(self.ny, d=self.dy, device=device, dtype=dtype) * 2 * torch.pi
            KX, KY = torch.meshgrid(kx, ky, indexing='ij')
            self._eigenvalues = self.D * (KX**2 + KY**2)
            # Handle zero frequency
            self._eigenvalues[0, 0] = 1.0
            return

        LAM_X, LAM_Y = torch.meshgrid(lam_x, lam_y, indexing='ij')
        self._eigenvalues = self.D * (LAM_X + LAM_Y)

        # Null space handling depends on BC type
        if self.bc_type == 'neumann':
            # (0,0) eigenvalue is 0 — set to 1, result will be zeroed
            self._eigenvalues[0, 0] = 1.0
        # Dirichlet: all eigenvalues > 0, no null space treatment needed

    def solve(self, A, b):
        """Solve via spectral transform. Matrix A is ignored."""
        if self._eigenvalues is None:
            self._compute_eigenvalues(b.device, b.dtype)

        rhs_grid = b.reshape(self.nx, self.ny)

        # Forward transform
        rhs_hat = self._forward(rhs_grid)

        # Spectral division
        u_hat = rhs_hat / self._eigenvalues

        # Null space pinning (Neumann and periodic only)
        if self.bc_type in ('neumann', 'periodic'):
            u_hat[0, 0] = 0.0

        # Inverse transform
        u_grid = self._inverse(u_hat)

        return u_grid.flatten()

    def _forward(self, x):
        """Forward spectral transform."""
        if self.bc_type == 'neumann':
            return self._dct2(x)
        elif self.bc_type == 'dirichlet':
            return self._dst2(x)
        elif self.bc_type == 'periodic':
            return torch.fft.fft2(x)

    def _inverse(self, x):
        """Inverse spectral transform."""
        if self.bc_type == 'neumann':
            return self._idct2(x)
        elif self.bc_type == 'dirichlet':
            return self._idst2(x)
        elif self.bc_type == 'periodic':
            return torch.fft.ifft2(x).real

    def _dct2(self, x):
        """2D DCT-II via FFT (Neumann BCs)."""
        # Same implementation as V5.4's DCTSolver._dct2_via_fft
        pass

    def _idct2(self, x):
        """2D IDCT (DCT-III) via FFT."""
        # Same implementation as V5.4's DCTSolver._idct2_via_fft
        pass

    def _dst2(self, x):
        """
        2D DST-II via FFT (Dirichlet BCs).

        DST-II of length N is equivalent to:
        1. Extend x to length 2(N+1) by odd reflection
        2. Take FFT
        3. Extract imaginary parts of first N coefficients

        More efficient: use the relation DST = imag(DFT of zero-padded odd extension)
        """
        nx, ny = x.shape

        # DST along first dimension
        # Odd extension: [0, x_0, x_1, ..., x_{N-1}, 0, -x_{N-1}, ..., -x_0]
        x_ext = torch.zeros(2 * (nx + 1), ny, device=x.device, dtype=x.dtype)
        x_ext[1:nx+1, :] = x
        x_ext[nx+2:, :] = -x.flip(0)
        fft_x = torch.fft.rfft(x_ext, dim=0)
        # DST coefficients = -imag(FFT) * normalization
        dst_x = -fft_x[1:nx+1, :].imag * (2.0 / (nx + 1)) ** 0.5

        # DST along second dimension (same logic, transposed)
        dst_x_ext = torch.zeros(nx, 2 * (ny + 1), device=x.device, dtype=x.dtype)
        dst_x_ext[:, 1:ny+1] = dst_x
        dst_x_ext[:, ny+2:] = -dst_x.flip(1)
        fft_y = torch.fft.rfft(dst_x_ext, dim=1)
        dst_xy = -fft_y[:, 1:ny+1].imag * (2.0 / (ny + 1)) ** 0.5

        return dst_xy

    def _idst2(self, x):
        """
        2D IDST (DST-III) via FFT (inverse of DST-II).

        The inverse DST-II is proportional to DST-III.
        DST-III can also be computed via FFT with odd extension.
        """
        # DST-III is (up to scaling) the same as DST-II
        # For orthonormal DST: IDST2 = DST3 = scaled DST2
        # Implementation mirrors _dst2 with adjusted phase
        pass
```

**When to use:**
- `bc_type='neumann'` — Standard insulated tissue, no bath
- `bc_type='dirichlet'` — **Bath-coupled boundaries (Kleber validation)**
- `bc_type='periodic'` — Periodic test domains

**Critical for Kleber effect:** The boundary speedup requires Dirichlet BCs on phi_e
at the tissue-bath interface. Using DCT (Neumann) here would be physically wrong — it
would model insulated boundaries and miss the speedup entirely.

### Tier 2: PCG + Spectral Preconditioner (Anisotropic, 1-3 iterations)

```python
class PCGSpectralSolver(LinearSolver):
    """
    PCG with spectral preconditioner for anisotropic elliptic solve.

    The isotropic spectral solve (DCT/DST/FFT) approximates the full
    anisotropic operator. Using it as a preconditioner inside PCG yields
    convergence in 1-3 iterations for moderate anisotropy ratios.

    The spectral preconditioner automatically uses the correct transform
    (DCT/DST/FFT) based on the bc_type, matching the physical BCs.

    This is the sweet spot for production anisotropic cases:
    - Near-optimal convergence rate
    - O(N log N) per PCG iteration (spectral application)
    - Only 1-3 iterations needed

    Parameters
    ----------
    nx, ny : int       Grid dimensions
    dx, dy : float     Grid spacing
    D_iso : float      Isotropic approximation of D (e.g., mean of D_i+D_e)
    bc_type : str      'neumann', 'dirichlet', or 'periodic'
    max_iters : int    Max PCG iterations (typically 5 is more than enough)
    tol : float        Convergence tolerance
    """

    def __init__(self, nx, ny, dx, dy, D_iso, bc_type='neumann',
                 max_iters=10, tol=1e-8):
        self.spectral_precond = SpectralSolver(nx, ny, dx, dy, D_iso, bc_type)
        self.max_iters = max_iters
        self.tol = tol
        # PCG workspace (lazy)
        self._r = self._z = self._p = self._Ap = self._x = None

    def solve(self, A, b):
        """
        PCG iteration:
            1. r = b - A*x
            2. z = spectral_solve(r)  <-- O(N log N) preconditioner application
            3. Standard CG update
        """
        n = b.shape[0]
        self._ensure_workspace(n, b.device, b.dtype)

        x = self._x; x.zero_()
        r = self._r; r.copy_(b)
        z = self._z; z.copy_(self.spectral_precond.solve(None, r))
        p = self._p; p.copy_(z)
        rz = torch.dot(r, z)

        for k in range(self.max_iters):
            Ap = self._Ap; Ap.copy_(sparse_mv(A, p))
            alpha = rz / torch.dot(p, Ap)
            x.add_(p, alpha=alpha)
            r.sub_(Ap, alpha=alpha)

            if torch.norm(r) / torch.norm(b) < self.tol:
                break

            z.copy_(self.spectral_precond.solve(None, r))
            rz_new = torch.dot(r, z)
            p.mul_(rz_new / rz).add_(z)
            rz = rz_new

        return x.clone()
```

**When to use:** Anisotropic fiber-based conductivity, production cardiac simulations.
Automatically selects DCT/DST/FFT based on bc_type.

### Tier 3: PCG + Geometric Multigrid (General, 10-25 iterations)

```python
class GeometricMultigridPreconditioner:
    """
    Geometric multigrid V-cycle for FDM structured grids.

    Natural preconditioner for FDM Laplacians on regular grids:
    - No setup phase (unlike AMG)
    - No coarsening heuristics
    - Simple restriction (full-weighting) / prolongation (bilinear)
    - O(N) per V-cycle
    - Works for any anisotropy, any coefficient variation

    Used inside PCG as a preconditioner. NOT a standalone solver.

    Parameters
    ----------
    nx, ny : int           Fine grid dimensions
    n_levels : int         Number of multigrid levels
    n_smooth : int         Smoothing sweeps per level
    smoother : str         'jacobi' or 'chebyshev'
    """

    def __init__(self, nx, ny, n_levels=4, n_smooth=3, smoother='jacobi'):
        self.n_levels = min(n_levels, self._max_levels(nx, ny))
        self.n_smooth = n_smooth
        self.smoother = smoother

        # Build grid hierarchy
        self._grids = []  # [(nx_l, ny_l) for each level]
        nx_l, ny_l = nx, ny
        for _ in range(self.n_levels):
            self._grids.append((nx_l, ny_l))
            nx_l = max(nx_l // 2, 2)
            ny_l = max(ny_l // 2, 2)

        # Operator at each level (built lazily when matrix is set)
        self._A_levels = [None] * self.n_levels
        # Workspace at each level
        self._workspace = [None] * self.n_levels

    def setup(self, A_fine):
        """Build coarse-level operators via Galerkin: A_c = R * A_f * P"""
        self._A_levels[0] = A_fine
        for level in range(self.n_levels - 1):
            R = self._build_restriction(level)
            P = self._build_prolongation(level)
            A_c = R @ self._A_levels[level] @ P
            self._A_levels[level + 1] = A_c.coalesce()

    def apply(self, r):
        """Apply one V-cycle: z ≈ A^{-1} r"""
        return self._v_cycle(r, level=0)

    def _v_cycle(self, b, level):
        if level == self.n_levels - 1:
            return self._coarse_solve(b, level)

        A = self._A_levels[level]
        x = torch.zeros_like(b)

        # Pre-smooth
        x = self._smooth(A, b, x, self.n_smooth)

        # Residual → restrict
        r = b - sparse_mv(A, x)
        r_coarse = self._restrict(r, level)

        # Recurse
        e_coarse = self._v_cycle(r_coarse, level + 1)

        # Prolongate → correct
        e = self._prolongate(e_coarse, level)
        x = x + e

        # Post-smooth
        x = self._smooth(A, b, x, self.n_smooth)
        return x

    def _restrict(self, v, level):
        """Full-weighting restriction (2D averaging)."""
        nx, ny = self._grids[level]
        v_grid = v.reshape(nx, ny)
        # Full-weighting stencil: [1,2,1; 2,4,2; 1,2,1] / 16
        # Downsample to (nx//2, ny//2)
        pass

    def _prolongate(self, v, level):
        """Bilinear interpolation (2D)."""
        nx_c, ny_c = self._grids[level + 1]
        nx_f, ny_f = self._grids[level]
        # Bilinear interpolation from coarse to fine
        pass

    def _smooth(self, A, b, x, n_sweeps):
        """Weighted Jacobi smoothing."""
        diag_inv = self._get_diag_inv(A)
        omega = 2.0 / 3.0  # Optimal for Laplacian
        for _ in range(n_sweeps):
            r = b - sparse_mv(A, x)
            x = x + omega * diag_inv * r
        return x

    def _coarse_solve(self, b, level):
        """Direct solve at coarsest level."""
        A = self._A_levels[level]
        # Coarse grid is small enough for dense solve
        A_dense = A.to_dense()
        return torch.linalg.solve(A_dense, b)


class EllipticPCGMGSolver(LinearSolver):
    """
    PCG with geometric multigrid preconditioner.

    General-purpose solver for any anisotropy, any coefficient variation,
    any grid structure. O(N) per iteration via GMG V-cycle.

    Convergence: 10-25 PCG iterations with GMG preconditioner.

    Parameters
    ----------
    nx, ny : int         Grid dimensions
    n_levels : int       Multigrid levels
    max_iters : int      Max PCG iterations
    tol : float          Convergence tolerance
    """

    def __init__(self, nx, ny, n_levels=4, max_iters=50, tol=1e-8):
        self.mg = GeometricMultigridPreconditioner(nx, ny, n_levels)
        self.max_iters = max_iters
        self.tol = tol
        self._setup_done = False
        # PCG workspace (lazy)
        self._r = self._z = self._p = self._Ap = self._x = None

    def solve(self, A, b):
        """PCG with GMG preconditioner."""
        if not self._setup_done:
            self.mg.setup(A)
            self._setup_done = True

        n = b.shape[0]
        self._ensure_workspace(n, b.device, b.dtype)

        x = self._x; x.zero_()
        r = self._r; r.copy_(b)
        z = self._z; z.copy_(self.mg.apply(r))    # GMG V-cycle precondition
        p = self._p; p.copy_(z)
        rz = torch.dot(r, z)

        for k in range(self.max_iters):
            Ap = self._Ap; Ap.copy_(sparse_mv(A, p))
            alpha = rz / torch.dot(p, Ap)
            x.add_(p, alpha=alpha)
            r.sub_(Ap, alpha=alpha)

            if torch.norm(r) / torch.norm(b) < self.tol:
                break

            z.copy_(self.mg.apply(r))
            rz_new = torch.dot(r, z)
            p.mul_(rz_new / rz).add_(z)
            rz = rz_new

        return x.clone()
```

**When to use:** Highly heterogeneous tissue, strong anisotropy, non-uniform coefficients.

### Tier Comparison

| Tier | Solver | Transform | Iters | Cost/Iter | Total Cost | Valid For |
|------|--------|-----------|-------|-----------|------------|-----------|
| 1 | Spectral Direct | DCT/DST/FFT | 0 | O(N log N) | O(N log N) | Isotropic, uniform grid |
| 2 | PCG + Spectral | DCT/DST/FFT | 1-3 | O(N log N) | O(N log N) | Moderate anisotropy |
| 3 | PCG + GMG | N/A | 10-25 | O(N) | O(N) | Any coefficient field |

**Decision flow:**
```
Is conductivity isotropic and spatially uniform?
  YES → Tier 1 (spectral direct — DCT/DST/FFT based on BCs)
  NO  → Is anisotropy ratio < 10:1?
          YES → Tier 2 (PCG + spectral preconditioner)
          NO  → Tier 3 (PCG + geometric multigrid)

Which spectral transform?
  phi_e insulated (Neumann)       → DCT
  phi_e bath-coupled (Dirichlet)  → DST   ← Kleber validation
  periodic domain                 → FFT
```

---

## Alternative: emRKC Fully Explicit Solver (No Linear Solves)

```python
class EMRKCBidomainDiffusionSolver(BidomainDiffusionSolver):
    """
    Explicit Multirate Runge-Kutta-Chebyshev (emRKC) solver.

    Eliminates ALL linear solves by using extended Chebyshev stability
    polynomials to make explicit time stepping stable for parabolic PDEs.

    Each time step requires s stages (s ~ 30-50), but each stage is
    just a stencil application — embarrassingly parallel, zero sync
    points, no preconditioner, no inner iteration.

    Trade-off:
    - MORE FLOPs per step (s stages vs 10-25 PCG iterations)
    - BETTER GPU utilization (no sync points, no convergence checks)
    - SIMPLER implementation (no linear solver infrastructure)

    Ref: Abdulle et al. (2024), Journal of Computational Physics

    Parameters
    ----------
    spatial : BidomainDiscretization
        Provides apply_L_i, apply_L_e, apply_L_ie
    dt : float
        Macro time step
    spectral_radius : float
        Estimated spectral radius of L_ie (for stage count)
    damping : float
        Damping parameter eta (default 0.05)
    """

    def __init__(self, spatial, dt, spectral_radius=None, damping=0.05):
        super().__init__(spatial, dt)
        self.damping = damping

        # Estimate spectral radius if not provided
        if spectral_radius is None:
            spectral_radius = self._estimate_spectral_radius(spatial)

        # Compute number of stages needed for stability
        # s ~ sqrt(spectral_radius * dt / 2)
        self.n_stages = max(int(torch.ceil(torch.sqrt(
            torch.tensor(spectral_radius * dt / 2.0)
        ))), 3)

        # Chebyshev coefficients
        self._mu, self._nu, self._kappa = self._compute_coefficients()

    def step(self, state, dt):
        """
        Advance Vm and phi_e by one macro time step using s RKC stages.

        Each stage k:
            Y_k = (1 - mu_k - nu_k) * Y_0 + mu_k * Y_{k-1} + nu_k * Y_{k-2}
                  + dt * kappa_k * F(Y_{k-1})  + dt * kappa_{k-1} * F(Y_{k-2})

        where F is the RHS of the bidomain diffusion (stencil application).
        """
        Vm = state.Vm.clone()
        phi_e = state.phi_e.clone()

        # Stage 0
        Vm_0 = Vm.clone()
        phi_e_0 = phi_e.clone()

        # Stage 1
        F_Vm = self._spatial.apply_L_i(Vm) + self._spatial.apply_L_i(phi_e)
        F_phi_e = -(self._spatial.apply_L_ie(phi_e) + self._spatial.apply_L_i(Vm))
        Vm_prev = Vm.clone()
        phi_e_prev = phi_e.clone()
        Vm = Vm + dt * self._mu[1] * F_Vm
        phi_e = phi_e + dt * self._mu[1] * F_phi_e

        # Stages 2..s
        for k in range(2, self.n_stages + 1):
            F_Vm = self._spatial.apply_L_i(Vm) + self._spatial.apply_L_i(phi_e)
            F_phi_e = -(self._spatial.apply_L_ie(phi_e) + self._spatial.apply_L_i(Vm))

            Vm_new = (1 - self._mu[k] - self._nu[k]) * Vm_0 \
                     + self._mu[k] * Vm + self._nu[k] * Vm_prev \
                     + dt * self._kappa[k] * F_Vm
            phi_e_new = (1 - self._mu[k] - self._nu[k]) * phi_e_0 \
                        + self._mu[k] * phi_e + self._nu[k] * phi_e_prev \
                        + dt * self._kappa[k] * F_phi_e

            Vm_prev = Vm
            phi_e_prev = phi_e
            Vm = Vm_new
            phi_e = phi_e_new

        # Pin phi_e (only needed for Neumann/periodic BCs)
        if self._spatial.grid.boundary_spec.phi_e_has_null_space:
            phi_e = phi_e - phi_e[0]

        state.Vm.copy_(Vm)
        state.phi_e.copy_(phi_e)
```

**When to use:** GPU-throughput-limited scenarios where sync points dominate, or as a simple
first implementation before building the full implicit solver stack.

---

## Solver Decision Matrix

| Scenario | Parabolic (Vm) | Elliptic (phi_e) | φ_e BC | Transform | Why |
|----------|---------------|-------------------|--------|-----------|-----|
| Kleber validation | DCT (Neumann) | DST Direct (Tier 1) | Dirichlet | DST | Bath-coupled, exact |
| Insulated tissue | DCT (Neumann) | DCT Direct (Tier 1) | Neumann | DCT | Standard, exact |
| Periodic test | FFT | FFT Direct (Tier 1) | Periodic | FFT | Periodic domain |
| Anisotropic + bath | PCG + Jacobi | PCG + DST (Tier 2) | Dirichlet | DST | 1-3 iters |
| Anisotropic insulated | PCG + Jacobi | PCG + DCT (Tier 2) | Neumann | DCT | 1-3 iters |
| Heterogeneous | PCG + Jacobi | PCG + GMG (Tier 3) | Any | N/A | General purpose |
| GPU throughput | emRKC | emRKC | Any | N/A | No linear solves |
| Accuracy-critical | Coupled MINRES | (not decoupled) | — | — | Future |

---

## Null Space Handling

The null space depends on boundary conditions:

| φ_e BC | Null Space | Treatment |
|--------|------------|-----------|
| **Neumann** (insulated) | Yes — constant vector | Pinning required |
| **Dirichlet** (bath-coupled) | **No** — all eigenvalues > 0 | No treatment needed |
| **Periodic** | Yes — constant vector | Pinning required |

**Key insight:** The Kleber boundary speedup validation uses Dirichlet BCs on phi_e,
which means **no null space exists** and no pinning is needed. This is physically correct:
the bath provides an absolute reference for extracellular potential.

### Point Pinning (Neumann/Periodic only)

```python
def apply_phi_e_pinning(A_ellip, rhs, pin_node: int = 0):
    """
    Enforce phi_e(pin_node) = 0 by modifying the linear system.

    Only needed for Neumann or periodic BCs where phi_e is determined
    up to a constant. NOT needed for Dirichlet (bath-coupled) BCs.

    For sparse A:
    - Zero out row and column for pin_node
    - Set A[pin_node, pin_node] = 1
    - Set rhs[pin_node] = 0

    For spectral solver: subtract phi_e[0] after solve (post-correction).
    For emRKC: subtract phi_e[0] after each macro step.
    """
    pass
```

**Pinning strategy per solver tier and BC type:**
| Tier | Neumann BC | Dirichlet BC |
|------|-----------|--------------|
| Spectral Direct | Post-subtract: `phi_e -= phi_e[0]` | Not needed (DST, all λ > 0) |
| PCG + Spectral | Post-subtract: `phi_e -= phi_e[0]` | Not needed |
| PCG + GMG | Matrix modification: zero row/col, diag=1 | Not needed |
| emRKC | Post-subtract after each macro step | Not needed |

---

## Runtime Step Specification

### BidomainSimulation.run()

```python
class BidomainSimulation:
    """
    Top-level orchestrator for bidomain simulations.
    Builds state and solvers from config strings.
    Follows V5.4's MonodomainSimulation pattern exactly.
    """

    def __init__(self, spatial, ionic_model, stimulus, dt,
                 splitting="strang",
                 ionic_solver="rush_larsen",
                 diffusion_solver="decoupled",
                 parabolic_solver="pcg",
                 elliptic_solver="auto",
                 theta=0.5):
        # 1. Build state
        x, y = spatial.coordinates
        Vm_init = torch.full((spatial.n_dof,), ionic_model.V_rest, ...)
        phi_e_init = torch.zeros(spatial.n_dof, ...)
        ionic_states = ionic_model.get_initial_state(spatial.n_dof)

        self.state = BidomainState(
            spatial=spatial,
            n_dof=spatial.n_dof,
            x=x, y=y,
            Vm=Vm_init,
            phi_e=phi_e_init,
            ionic_states=ionic_states,
            ...
        )
        self.dt = dt

        # 2. Auto-select elliptic solver from BoundarySpec if "auto"
        if elliptic_solver == "auto":
            elliptic_solver = self._auto_select_elliptic_solver(spatial)

        # 3. Construct solver chain from strings
        # BCs are read from spatial.grid.boundary_spec — not passed as strings
        para_ls = _build_linear_solver(parabolic_solver, spatial)
        ellip_ls = _build_linear_solver(elliptic_solver, spatial)
        diffusion = _build_diffusion_solver(diffusion_solver, spatial, dt,
                                            para_ls, ellip_ls, theta)
        ionic = _build_ionic_solver(ionic_solver, ionic_model)
        self.splitting = _build_splitting(splitting, ionic, diffusion)

    @staticmethod
    def _auto_select_elliptic_solver(spatial):
        """Read BoundarySpec from mesh and pick best solver tier."""
        bc = spatial.grid.boundary_spec
        is_isotropic = not hasattr(spatial, '_conductivity') or \
                       (spatial._conductivity.D_i_field is None and
                        spatial._conductivity.theta is None)

        if bc.phi_e_spectral_eligible and is_isotropic:
            return "spectral"       # Tier 1: O(N log N), no iterations
        elif bc.phi_e_spectral_eligible:
            return "pcg_spectral"   # Tier 2: PCG + spectral precond, 1-3 iters
        else:
            return "pcg_gmg"        # Tier 3: PCG + GMG, 10-25 iters

    def run(self, t_end, save_every=1.0, callback=None):
        """Run simulation. Yields state at save points."""
        state = self.state
        dt = self.dt
        next_save = save_every

        while state.t < t_end:
            self.splitting.step(state, dt)
            state.t += dt

            if state.t >= next_save:
                self._buffer_output(state)
                next_save += save_every
                if callback and callback(state):
                    break
                yield state
```

### Splitting Step (Strang example, Decoupled)

```
splitting.step(state, dt):
    |
    +-- ionic_solver.step(state, dt/2)         <-- half-step ionic
    |     |
    |     +-- Vm = state.Vm
    |     +-- Iion = model.compute_Iion(Vm, state.ionic_states)
    |     +-- Istim_i = self._evaluate_Istim(state, 'i')
    |     +-- state.Vm += (dt/2) * (-Iion + Istim_i)
    |     +-- gates: Rush-Larsen exponential update
    |     +-- concentrations: Forward Euler update
    |
    +-- diffusion_solver.step(state, dt)       <-- full-step diffusion (DECOUPLED)
    |     |
    |     +-- Step 1: Parabolic solve for Vm
    |     |   +-- rhs = B_para * Vm + theta * L_i * phi_e
    |     |   +-- Vm_new = parabolic_solver.solve(A_para, rhs)
    |     |
    |     +-- Step 2: Elliptic solve for phi_e
    |     |   +-- rhs = L_i * Vm_new
    |     |   +-- phi_e_new = elliptic_solver.solve(A_ellip, rhs)
    |     |
    |     +-- state.Vm = Vm_new
    |     +-- state.phi_e = phi_e_new
    |
    +-- ionic_solver.step(state, dt/2)         <-- half-step ionic
```

---

## Two Simulation Paradigms

### Classical (FDM-First)

Traditional PDE discretization with operator splitting and decoupled linear solves:

```
tissue_builder -> discretization -> state -> solver
   (mesh)         (L_i, L_e)       (data)   (compute)
   [init]          [init]          [init]    [runtime]
```

Decoupled solves at each diffusion step:
```
Step 1 (Parabolic):  A_para * Vm^{n+1} = rhs_para(Vm^n, phi_e^n)
Step 2 (Elliptic):   A_ellip * phi_e^{n+1} = L_i * Vm^{n+1}
```

Each is N x N SPD — solved by PCG, DCT, or GMG (no block system needed).

### Lattice-Boltzmann (Dual-Lattice LBM) -- Future

Two independent D2Q5 lattices for Vm and phi_e:

```
Vm lattice:  collision -> streaming -> bounce-back -> recover Vm
phi_e lattice: collision -> streaming -> bounce-back -> recover phi_e (pseudo-time)
```

- Vm lattice: standard LBM with ionic source term
- phi_e lattice: iterates to steady state (elliptic solve via pseudo-time relaxation)
- Coupling: Vm drives phi_e source; phi_e feeds back to Vm collision

Potential 5-50x speedup over FEM, but pseudo-time convergence rate is uncertain.

---

## Spatial Discretization Methods

| Method | Mass Matrix M | Intra K_i | Extra K_e | K_i + K_e | Mesh |
|--------|-------------|-----------|-----------|-----------|------|
| FEM | Sparse (integral) | Sparse stiffness | Sparse stiffness | Sum | Unstructured |
| FDM | Identity | 9-pt stencil L_i | 9-pt stencil L_e | Sum | Structured |
| FVM | Diagonal (volumes) | TPFA flux F_i | TPFA flux F_e | Sum | Structured |

### How Bidomain Affects Each Method

**FEM:** Assemble K_i and K_e independently using D_i and D_e conductivities. Same element loop, different D tensors. Two `assemble_stiffness_matrix()` calls.

**FDM:** Build two separate Laplacian matrices L_i and L_e using D_i and D_e fields. Same stencil construction, different conductivity inputs. Two `_build_laplacian()` calls.

**FVM:** Build two separate TPFA flux matrices F_i and F_e. Same harmonic mean interface treatment, different D values.

---

## Linear Solver Comparison (Decoupled Approach)

### Elliptic Solver Tiers (phi_e — dominates 60-80% of runtime)

| Tier | Solver | Transform | Iters | Cost/Iter | Total | Valid For | GPU Sync |
|------|--------|-----------|-------|-----------|-------|-----------|----------|
| 1 | Spectral Direct | DCT/DST/FFT | 0 | O(N log N) | O(N log N) | Isotropic, uniform | 0 |
| 2 | PCG + Spectral | DCT/DST/FFT | 1-3 | O(N log N) | O(N log N) | Moderate anisotropy | 2-3/iter |
| 3 | PCG + GMG | N/A | 10-25 | O(N) | O(N) | Any coefficients | 2-3/iter |
| alt | emRKC (explicit) | N/A | 0 | O(s*N) | O(s*N) | GPU throughput | 0 |

### Parabolic Solver Options (Vm — well-conditioned, 10-15 iters)

| Solver | Iters | Best For | Notes |
|--------|-------|----------|-------|
| PCG + Jacobi | 10-15 | General | Reused from V5.4, always works |
| DCT Direct | 0 | Isotropic FDM | O(N log N), exact for constant D |
| Chebyshev | Fixed | GPU throughput | Zero sync, needs eigenvalue bounds |

### Why Decoupled Beats Coupled for GPU FDM

| Aspect | Coupled (MINRES 2N) | Decoupled (2 × PCG N) |
|--------|---------------------|------------------------|
| Matrix size | 2N × 2N indefinite | 2 × (N × N) SPD |
| Krylov method | MINRES (indefinite) | PCG (SPD) or DCT (direct) |
| Preconditioner | Block diagonal/triangular (complex) | Jacobi, DCT, or GMG (simple) |
| GPU sync points | 2-3 per MINRES iter × 15-30 iters | 0 (DCT) or 2-3 per PCG iter × 1-25 |
| Matrix-free | Harder (block matvec) | Natural (separate stencils) |
| Memory | 4× matrix storage | 2× matrix storage |
| Implementation | Complex (block solver + precond + subsolver) | Simple (reuse V5.4 LinearSolver) |
| Accuracy | Exact coupling | O(dt) splitting error (acceptable) |

---

## Component Dependency Diagram

```
BidomainSimulation (bidomain.py)
  |
  |  [init] builds from config strings:
  |
  +---> tissue_builder/mesh/             Create geometry (StructuredGrid)
  +---> tissue_builder/tissue/           Create BidomainConductivity (D_i, D_e)
  +---> tissue_builder/stimulus/         Precompute masks, timing
  +---> discretization/                  Build L_i, L_e stencils
  |
  |  [init] assembles state + solver chain:
  |
  +---> state.py                         Vm, phi_e, ionic_states, stim data
  |
  +---> SplittingStrategy                Owns sub-solvers
        |
        +---> IonicSolver                 UNCHANGED from V5.4
        |       +---> ionic/              compute_Iion(), gate_inf(), gate_tau()
        |
        +---> DecoupledBidomainDiffusionSolver
                |
                +---> BidomainDiscretization   L_i, L_e, A_para, A_ellip
                +---> LinearSolver (parabolic) PCG / DCT / Chebyshev
                +---> LinearSolver (elliptic)  DCT / PCG+DCT / PCG+GMG
```

---

## File Responsibility Matrix

| File | Owns | Knows / Decides |
|------|------|-----------------|
| `ionic/` | Physics functions | Cell biology (UNCHANGED) |
| `tissue_builder/mesh/` | Geometry | Domain shapes (UNCHANGED) |
| `tissue_builder/tissue/conductivity.py` | D_i, D_e pair | Conductivity tensor construction |
| `tissue_builder/stimulus/` | Protocol | Regions, timing (UNCHANGED) |
| `discretization/base.py` | BidomainDiscretization ABC | What spatial operators to provide |
| `discretization/fdm.py` | L_i, L_e stencils | How to build FDM Laplacians |
| `state.py` | Vm, phi_e, ionic_states | Nothing (pure data) |
| `bidomain.py` | State construction, time loop | Solver construction from strings |
| `solver/splitting/*` | IonicSolver + DiffusionSolver refs | Call ordering |
| `solver/ionic_stepping/*` | IonicModel ref | ODE method (UNCHANGED) |
| `solver/diffusion_stepping/decoupled.py` | Parabolic + elliptic solvers | Decoupled bidomain PDE |
| `solver/linear_solver/pcg.py` | PCG workspace | SPD iterative solve (from V5.4) |
| `solver/linear_solver/spectral.py` | FFT/DCT/DST eigenvalues | Spectral direct solve (BC-aware) |
| `solver/linear_solver/pcg_spectral.py` | PCG + spectral precond | Anisotropic spectral-preconditioned solve |
| `solver/linear_solver/multigrid.py` | GMG hierarchy | Geometric multigrid V-cycle |
| `solver/linear_solver/pcg_gmg.py` | PCG + GMG precond | General-purpose multigrid-preconditioned solve |
| `solver/linear_solver/chebyshev.py` | Chebyshev coefficients | Sync-free polynomial solve (from V5.4) |

---

## User API

### Classical Approach (BoundarySpec + Auto-Selection)

```python
from cardiac_sim.tissue_builder.mesh import StructuredGrid
from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec, Edge
from cardiac_sim.tissue_builder.stimulus import StimulusProtocol, EdgeRegion
from cardiac_sim.tissue_builder.tissue import BidomainConductivity
from cardiac_sim.simulation.classical.discretization import BidomainFDMDiscretization
from cardiac_sim.simulation.classical import BidomainSimulation

# Build geometry
grid = StructuredGrid(Lx=2.0, Ly=0.5, nx=800, ny=200)

# Define bidomain conductivities
conductivity = BidomainConductivity(D_i=0.00124, D_e=0.00446)

# Define stimulus
stimulus = StimulusProtocol()
stimulus.add_stimulus(EdgeRegion('left', width=0.1), start=0, duration=1, amplitude_i=-52)

# --- Example 1: Insulated tissue (default BCs, auto-selects DCT) ---
grid.boundary_spec = BoundarySpec.insulated()  # Neumann everywhere (default)
spatial = BidomainFDMDiscretization(grid, conductivity, chi=1400.0, Cm=1.0)

sim_insulated = BidomainSimulation(
    spatial=spatial,
    ionic_model="ttp06",
    stimulus=stimulus,
    dt=0.02,
    # elliptic_solver="auto" → auto-selects Tier 1 SpectralSolver(DCT)
    # No phi_e_bc parameter — BCs are in grid.boundary_spec
)

# --- Example 2: Kleber boundary speedup (bath-coupled, auto-selects DST) ---
grid.boundary_spec = BoundarySpec.bath_coupled()  # phi_e = 0 on all edges
spatial_kleber = BidomainFDMDiscretization(grid, conductivity, chi=1400.0, Cm=1.0)

sim_kleber = BidomainSimulation(
    spatial=spatial_kleber,
    ionic_model="ttp06",
    stimulus=stimulus,
    dt=0.02,
    # elliptic_solver="auto" → auto-selects Tier 1 SpectralSolver(DST)
    # No null space pinning needed — Dirichlet BCs eliminate null space
)

# --- Example 3: Anisotropic + bath (auto-selects PCG+Spectral) ---
conductivity_aniso = BidomainConductivity(
    D_i_fiber=0.003, D_i_cross=0.0003,
    D_e_fiber=0.002, D_e_cross=0.001,
    theta=fiber_angle_field,
)
grid.boundary_spec = BoundarySpec.bath_coupled()
spatial_aniso = BidomainFDMDiscretization(grid, conductivity_aniso, chi=1400.0, Cm=1.0)

sim_aniso = BidomainSimulation(
    spatial=spatial_aniso,
    ionic_model="ttp06",
    stimulus=stimulus,
    dt=0.02,
    # elliptic_solver="auto" → auto-selects Tier 2 PCGSpectralSolver(DST)
    # OR override: elliptic_solver="pcg_gmg"
)

# --- Example 4: Mixed BCs (auto-selects PCG+GMG) ---
grid.boundary_spec = BoundarySpec.bath_coupled_edges([Edge.TOP, Edge.BOTTOM])
spatial_mixed = BidomainFDMDiscretization(grid, conductivity, chi=1400.0, Cm=1.0)

sim_mixed = BidomainSimulation(
    spatial=spatial_mixed,
    ionic_model="ttp06",
    stimulus=stimulus,
    dt=0.02,
    # elliptic_solver="auto" → mixed BCs, spectral ineligible → Tier 3 PCG+GMG
)

# Run and get results
for state in sim_kleber.run(t_end=500.0, save_every=1.0):
    plot_vm(state.Vm)
    plot_phi_e(state.phi_e)
```

---

## GPU Optimization Strategy

### Transfer Points (Decoupled Approach)

| # | Location | Transfer | Frequency | Severity |
|---|----------|----------|-----------|----------|
| 1 | Matrix assembly | L_i, L_e to GPU | Once | OK |
| 2 | PCG convergence | Dot products (residual norm) | 2-3 per iter | Medium |
| 3 | Output saving | Vm, phi_e arrays | Per save | Major |

### Mitigation

| Fix | Strategy |
|-----|----------|
| DCT elliptic solver (Tier 1) | **Zero sync** — FFT is fully GPU-resident |
| Chebyshev parabolic solver | Zero sync — fixed iteration count |
| torch.compile on ionic model | 2-5x speedup via kernel fusion, zero code changes |
| emRKC explicit solver | **Zero sync** — every stage is stencil application |
| GPU output buffer | Batch transfers (same as V5.4) |

### GPU Sync Analysis by Solver Configuration

| Configuration | Sync Points/Step | Best For |
|---------------|-----------------|----------|
| DCT + DCT | 0 | Isotropic validation |
| PCG + DCT | 2-3 per parabolic iter | Isotropic with implicit Vm |
| emRKC | 0 | Maximum GPU throughput |
| PCG + PCG+GMG | 2-3 per iter × (15+25) | General production |

---

## Research Reference Guide

All research documentation lives in `Research/Bidomain/`.

### Research Documents

| # | File | Covers |
|---|------|--------|
| 00 | `00_MASTER_INDEX.md` | Overview, roadmap, decisions |
| 01 | `Discretization/BIDOMAIN_DISCRETIZATION.md` | FEM/FDM/FVM for bidomain, block matrix structure |
| 02 | `Solver_Methods/BIDOMAIN_SOLVER_METHODS.md` | Operator splitting, IMEX, time stepping |
| 03 | `Linear_Solvers/BIDOMAIN_LINEAR_SOLVERS.md` | Block preconditioners, AMG, MINRES/GMRES |
| 04 | `LBM_Bidomain/LBM_BIDOMAIN.md` | Dual-lattice LBM approach |
| 05 | `Explicit_Methods/BIDOMAIN_EXPLICIT_METHODS.md` | IMEX, RKC, explicit methods |
| 06 | `Code_Examples/` | Reference implementations (~3600 lines) |

### Module -> Research Mapping

#### `discretization/fdm.py` (Bidomain FDM)
- **Read:** Discretization doc Sections 3.1-3.4 (FDM stencils)
- **Read:** Monodomain Research/01_FDM (9-pt stencil details, L49-76)
- **Read:** `research/GPU_BIDOMAIN_LITERATURE.md` Paper 5 (matrix-free stencil)
- **Key:** Two Laplacians L_i and L_e, same stencil construction, different D

#### `solver/diffusion_stepping/decoupled.py`
- **Read:** Solver_Methods doc Sections 4.1-4.3 (operator splitting)
- **Read:** `research/GPU_BIDOMAIN_LITERATURE.md` Papers 4, 9 (partitioned approach)
- **Key:** Gauss-Seidel splitting — parabolic then elliptic, O(dt) error

#### `solver/linear_solver/spectral.py` (Unified FFT/DCT/DST)
- **Read:** V5.4 `linear_solver/fft.py` (DCT/FFT implementation — adapt for DST)
- **Read:** `research/GPU_BIDOMAIN_LITERATURE.md` Paper 10 (FFT/DCT insight)
- **Key:** Three BC types → three transforms. Eigenvalue formulas differ per BC.
  Neumann→DCT, Dirichlet→DST (Kleber validation), Periodic→FFT.
  DST has NO null space (all eigenvalues > 0) — no pinning needed.

#### `solver/linear_solver/pcg_spectral.py` (Tier 2)
- **Read:** V5.4 `linear_solver/pcg.py` (PCG implementation to wrap)
- **Key:** Spectral solve as preconditioner inside PCG. bc_type flows through.

#### `solver/linear_solver/multigrid.py` (Geometric Multigrid)
- **Read:** `research/GPU_BIDOMAIN_LITERATURE.md` Papers 5, 3 (GMG for FDM)
- **Read:** Monodomain Research/03_GPU_Linear (Chebyshev smoother L39-65)
- **Key:** V-cycle, full-weighting restriction, bilinear prolongation, Jacobi smoother

#### `solver/diffusion_stepping/emrkc.py` (Explicit Alternative)
- **Read:** `research/GPU_BIDOMAIN_LITERATURE.md` Paper 8 (emRKC)
- **Read:** Explicit_Methods/BIDOMAIN_EXPLICIT_METHODS.md
- **Key:** Extended Chebyshev stability, s stages per step, zero sync

#### `research/BOUNDARY_SPEEDUP_ANALYSIS.md` (Validation Target)
- **Read:** Kleber boundary speedup derivation
- **Key:** CV ratio = sqrt((sigma_i+sigma_e)/sigma_e) ≈ 1.13

---

## Reference Parameters (2D Cardiac Bidomain)

```
D_i_fiber  = 0.003     cm^2/ms    (intracellular along fibers)
D_i_cross  = 0.0003    cm^2/ms    (intracellular across fibers, 10:1 ratio)
D_e_fiber  = 0.002     cm^2/ms    (extracellular along fibers)
D_e_cross  = 0.001     cm^2/ms    (extracellular across fibers, 2:1 ratio)
C_m        = 1.0       uF/cm^2    (membrane capacitance)
chi        = 1400      cm^-1      (surface-to-volume ratio)
h          = 0.025     cm         (250 um grid spacing)
dt         = 0.01      ms         (time step)
```

### Performance Expectations (Decoupled FDM)

| Metric | Monodomain | Bidomain (DCT) | Bidomain (PCG+GMG) |
|--------|-----------|----------------|---------------------|
| State memory | N | 2N | 2N |
| Matrix memory | 10N nnz | 20N nnz | 20N nnz + GMG levels |
| Linear solver iters | 10-30 | 0 (DCT) + 10-15 (para) | 10-25 + 10-15 |
| Time per step | 1x | 2-3x | 5-10x |

---

## Migration Path (FDM-First, Decoupled)

### Phase 1: Foundation
1. Create directory structure
2. Copy ionic/ from V5.4 (unchanged)
3. Copy mesh/, stimulus/ from V5.4 (unchanged)
4. Create BoundarySpec protocol (boundary.py)
5. Create BidomainConductivity
6. Create BidomainState
7. Copy PCG, Chebyshev linear solvers from V5.4

### Phase 2: FDM Discretization
8. Create BidomainSpatialDiscretization ABC (decoupled interface)
9. Implement BidomainFDMDiscretization (L_i, L_e, BC-aware stencils)
10. Implement get_parabolic_operators / get_elliptic_operator
11. Validate stencil accuracy (convergence test)

### Phase 3: Linear Solvers
12. Implement SpectralSolver (Tier 1 — unified DCT/DST/FFT)
13. Implement PCGSpectralSolver (Tier 2 — PCG + spectral preconditioner)
14. Implement GeometricMultigridPreconditioner + EllipticPCGMGSolver (Tier 3)
15. Validate each tier against manufactured solutions

### Phase 4: Diffusion Solver
15. Implement DecoupledBidomainDiffusionSolver
16. Implement null space pinning (phi_e)
17. Validate parabolic + elliptic solve against known solutions

### Phase 5: Orchestration
18. Implement BidomainSimulation orchestrator
19. Implement SplittingStrategy (Strang, Godunov) for bidomain
20. Wire factory functions for string-based config
21. End-to-end validation (planar wave, TTP06)

### Phase 6: Boundary Speedup Validation
22. Implement bath-coupled boundary conditions
23. Run 4-config experiment (BOUNDARY_SPEEDUP_ANALYSIS.md)
24. Validate ~13% CV increase at tissue-bath boundary

### Future Phases (Deferred)
- FEM/FVM discretization
- Coupled MINRES solver (if accuracy-critical applications arise)
- emRKC explicit solver
- LBM bidomain
- Builder integration

---

## TODO Summary

### Core Bidomain Infrastructure
- [ ] BidomainConductivity (D_i, D_e pair)
- [ ] BidomainState (Vm, phi_e, ionic_states)
- [ ] BidomainDiscretization ABC
- [ ] Copy ionic/, mesh/, stimulus/, backend from V5.4

### FDM Discretization
- [ ] BidomainFDMDiscretization (L_i, L_e from D_i, D_e)
- [ ] Parabolic operator construction (A_para, B_para)
- [ ] Elliptic operator construction (A_ellip)
- [ ] Stencil convergence validation

### Linear Solvers (Three Tiers)
- [ ] SpectralSolver (Tier 1 — unified DCT/DST/FFT direct, O(N log N))
- [ ] PCGSpectralSolver (Tier 2 — PCG + spectral preconditioner, 1-3 iters)
- [ ] GeometricMultigridPreconditioner (V-cycle for structured grids)
- [ ] EllipticPCGMGSolver (Tier 3 — PCG + GMG preconditioner, 10-25 iters)
- [ ] Copy PCG, Chebyshev from V5.4 (parabolic solver)

### Diffusion Solver
- [ ] DecoupledBidomainDiffusionSolver (parabolic + elliptic)
- [ ] Null space pinning for phi_e
- [ ] rebuild_operators() for adaptive dt

### Orchestration
- [ ] BidomainSimulation (factory functions, run loop)
- [ ] SplittingStrategy for bidomain (Strang, Godunov)
- [ ] String-based config: parabolic_solver, elliptic_solver

### Boundary Speedup Validation
- [ ] Bath-coupled boundary conditions (intracellular Neumann, extracellular Dirichlet)
- [ ] 4-config experiment (A: mono/Neumann, B: mono/enhanced, C: mono/Dirichlet, D: bidomain)
- [ ] CV measurement and comparison (~13% boundary speedup)

### Future (Deferred)
- [ ] emRKC explicit bidomain solver (zero linear solves)
- [ ] FEM/FVM discretization
- [ ] Coupled MINRES solver (accuracy-critical)
- [ ] LBM bidomain (dual-lattice)
- [ ] Builder integration

---

## Summary

The bidomain extension preserves V5.4's core design principles:

- **Separating spatial from temporal** -- discretization provides L_i, L_e; solvers consume them
- **Separating physics from numerics** -- ionic models unchanged; bidomain is purely a spatial/solver concern
- **One file per method** -- adding a new solver tier means adding one file
- **Solvers own their artifacts** -- each LinearSolver owns workspace buffers
- **Zero allocation per step** -- all workspace pre-allocated
- **Reuse over rewrite** -- PCG, DCT, Chebyshev copied directly from V5.4

New principles specific to bidomain:

- **Decoupled over coupled** -- two N×N SPD solves, not one 2N×2N indefinite system
- **Three-tier elliptic strategy** -- DCT (isotropic) → PCG+DCT (anisotropic) → PCG+GMG (general)
- **Null space handled per solver tier** -- post-subtract for DCT, matrix modification for PCG
- **Ionic solver is model-agnostic** -- doesn't know if it's monodomain or bidomain
- **Literature-driven architecture** -- every design choice backed by GPU bidomain literature
