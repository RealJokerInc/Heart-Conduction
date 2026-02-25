# Engine V5.3 Architecture Improvement Proposal

## Overview

This document outlines a proposed restructuring of the monodomain cardiac simulation codebase to improve modularity, extensibility, and maintainability. The design separates:

- **Physics** (what the cell computes) — `ionic/`
- **Geometry & Materials** (what to build) — `tissue_builder/`
- **Spatial Discretization** (how space is represented) — `simulation/classical/discretization_scheme/`
- **Temporal Discretization** (how time is advanced) — `simulation/classical/solver/`
- **Hardware** (where we compute) — `utils/`

---

## Current Architecture

```
Engine_V5.3/
├── ionic/                   # Cellular ionic models
│   ├── base.py              # Abstract IonicModel interface
│   ├── lut.py               # Lookup table acceleration
│   ├── ord/                 # O'Hara-Rudy 2011 (41 states)
│   │   ├── model.py
│   │   ├── parameters.py
│   │   ├── gating.py
│   │   ├── currents.py
│   │   ├── calcium.py
│   │   └── camkii.py
│   └── ttp06/               # ten Tusscher-Panfilov 2006 (19 states)
│       ├── model.py
│       ├── parameters.py
│       ├── gating.py
│       ├── currents.py
│       ├── calcium.py
│       └── celltypes/
│
├── fem/                     # Finite Element Infrastructure
│   ├── mesh.py              # TriangularMesh class
│   └── assembly.py          # Matrix assembly
│
├── solver/                  # Time integration
│   ├── linear.py            # PCG solver
│   └── time_stepping.py     # CN, BDF1, BDF2
│
├── tissue/                  # Tissue-level simulation
│   ├── simulation.py        # MonodomainSimulation
│   └── stimulus.py          # Stimulus protocols
│
├── utils/
│   └── backend.py           # CPU/GPU abstraction
│
├── examples/
└── tests/
```

### Current Architecture Issues

1. **FEM is isolated** - Mesh and assembly are separate from tissue, but mesh is inherently tissue-specific
2. **Single discretization method** - Only FEM supported, no path for FDM/FVM
3. **LUT at wrong level** - Lookup tables serve ionic models but sit at ionic root
4. **Simulation buried in tissue** - Entry point not clearly separated from domain logic
5. **step() inside ionic models** - Models mix physics knowledge with numerical stepping strategy
6. **Solver lives outside simulation** - `solver/linear.py` only serves diffusion but sits at top level
7. **No explicit splitting strategy** - Godunov splitting is hardcoded, no path to Strang
8. **No alternative paradigms** - No path for Lattice-Boltzmann or other methods
9. **Discretization coupled to time stepping** - No clean separation between spatial and temporal discretization

---

## Final Proposed Architecture

```
cardiac_sim/
│
├── ionic/                                   # Shared cell models (physics)
│   ├── base.py                              # IonicModel ABC (data provider only)
│   └── models/
│       ├── ord/                             # O'Hara-Rudy 2011 (41 states)
│       │   ├── model.py                     # ORdModel - compute functions
│       │   ├── parameters.py                # State indices, cell types, initial conditions
│       │   ├── gating.py                    # compute_gate_inf(), compute_gate_tau()
│       │   ├── currents.py                  # compute_INa(), compute_ICaL(), ...
│       │   ├── calcium.py                   # compute_calcium_fluxes()
│       │   └── camkii.py                    # compute_camkii()
│       ├── ttp06/                           # ten Tusscher-Panfilov 2006 (19 states)
│       │   ├── model.py                     # TTP06Model - compute functions
│       │   ├── parameters.py                # State indices, cell types
│       │   ├── gating.py                    # Gate kinetics functions
│       │   ├── currents.py                  # Ion channel current functions
│       │   ├── calcium.py                   # Calcium dynamics functions
│       │   └── celltypes/                   # Custom cell type configs
│       └── lut.py                           # Lookup table acceleration
│
├── tissue_builder/                          # Builders (create data at init)
│   ├── __init__.py
│   │
│   ├── mesh/                                # Geometry structures
│   │   ├── __init__.py
│   │   ├── base.py                          # Mesh ABC
│   │   ├── triangular.py                    # TriangularMesh (2D FEM)
│   │   ├── tetrahedral.py                   # [TODO] TetrahedralMesh (3D FEM)
│   │   └── structured.py                    # [TODO] StructuredGrid (FDM/FVM)
│   │
│   ├── tissue/                              # Geometry + material properties
│   │   ├── __init__.py
│   │   ├── base.py                          # TissueDomain ABC
│   │   ├── isotropic.py                     # IsotropicTissue
│   │   ├── anisotropic.py                   # [TODO] AnisotropicTissue
│   │   └── heterogeneous.py                 # [TODO] HeterogeneousTissue (scar)
│   │
│   └── stimulus/                            # Pacing protocols
│       ├── __init__.py
│       ├── protocol.py                      # StimulusProtocol
│       └── regions.py                       # StimulusRegion, RectangularRegion, etc.
│
├── simulation/                              # Runtime simulation
│   │
│   ├── classical/                           # FEM/FDM/FVM + traditional solvers
│   │   ├── __init__.py
│   │   ├── state.py                         # SimulationState (data container)
│   │   ├── monodomain.py                    # MonodomainSimulation (orchestrator)
│   │   │
│   │   ├── discretization_scheme/           # Spatial discretization (HOW space is represented)
│   │   │   ├── __init__.py
│   │   │   ├── base.py                      # SpatialDiscretization ABC
│   │   │   ├── fem.py                       # Finite Element Method
│   │   │   ├── fdm.py                       # [TODO] Finite Difference Method
│   │   │   └── fvm.py                       # [TODO] Finite Volume Method
│   │   │
│   │   └── solver/                          # Time stepping algorithms
│   │       ├── __init__.py                  # Package exports
│   │       │
│   │       ├── splitting/                   # Operator splitting strategies
│   │       │   ├── __init__.py              # SplittingStrategy ABC
│   │       │   ├── godunov.py               # GodunovSplitting (1st order)
│   │       │   ├── strang.py                # StrangSplitting (2nd order)
│   │       │   └── symmetric_strang.py      # SymmetricStrangSplitting
│   │       │
│   │       ├── ionic_time_stepping/         # Ionic ODE solvers
│   │       │   ├── __init__.py              # IonicSolver ABC
│   │       │   ├── rush_larsen.py           # RushLarsenSolver
│   │       │   ├── forward_euler.py         # ForwardEulerSolver
│   │       │   └── adaptive.py              # AdaptiveDTSolver
│   │       │
│   │       └── diffusion_time_stepping/     # Diffusion PDE solvers
│   │           ├── __init__.py              # DiffusionSolver ABC
│   │           │
│   │           ├── explicit/                # Explicit time discretization
│   │           │   ├── __init__.py          # ExplicitDiffusionSolver ABC
│   │           │   ├── forward_euler.py     # [TODO] ForwardEulerDiffusion
│   │           │   ├── rk2.py               # [TODO] RK2Diffusion (Heun's)
│   │           │   └── rk4.py               # [TODO] RK4Diffusion
│   │           │
│   │           ├── implicit/                # Implicit time discretization
│   │           │   ├── __init__.py          # ImplicitDiffusionSolver ABC
│   │           │   ├── crank_nicolson.py    # CrankNicolsonSolver
│   │           │   ├── bdf1.py              # BDF1Solver (Backward Euler)
│   │           │   └── bdf2.py              # BDF2Solver
│   │           │
│   │           └── linear_solver/           # Linear system solvers (Ax = b)
│   │               ├── __init__.py          # LinearSolver ABC
│   │               ├── pcg.py               # PCGSolver (Krylov method)
│   │               ├── chebyshev.py         # [TODO] ChebyshevSolver (polynomial)
│   │               ├── multigrid.py         # [TODO] Algebraic Multigrid
│   │               └── fft.py               # [TODO] FFT solver (structured grids)
│   │
│   └── lbm/                                 # Lattice-Boltzmann (independent)
│       ├── __init__.py
│       ├── state.py                         # [TODO] LBM state
│       ├── monodomain.py                    # [TODO] LBM simulation entry
│       ├── stimulus.py                      # [TODO] LBM stimulus handling
│       ├── lattice.py                       # [TODO] Lattice structures
│       ├── d2q7.py                          # [TODO] 2D, 7 velocities
│       ├── d3q7.py                          # [TODO] 3D, 7 velocities
│       ├── d3q19.py                         # [TODO] 3D, 19 velocities
│       └── collision.py                     # [TODO] BGK, MRT operators
│
├── utils/                                   # Shared utilities
│   ├── backend.py                           # Device abstraction (CPU/CUDA/MPS)
│   └── platform.py                          # Platform optimization profiles
│
└── tests/
    ├── test_ionic/
    ├── test_tissue_builder/
    ├── test_simulation/
    │   ├── test_classical/
    │   │   ├── test_discretization_scheme/
    │   │   ├── test_splitting/
    │   │   ├── test_ionic_time_stepping/
    │   │   └── test_diffusion_time_stepping/
    │   └── test_lbm/
    └── test_utils/
```

---

## Top-Level Structure

```
cardiac_sim/
│
├── ionic/                  # WHAT the cell computes (physics)
│
├── tissue_builder/         # WHAT to build (geometry, materials, protocols)
│
├── simulation/
│   ├── classical/
│   │   ├── discretization_scheme/   # HOW space is discretized (FEM/FDM/FVM)
│   │   └── solver/                  # HOW time is advanced
│   └── lbm/                         # Alternative paradigm (self-contained)
│
└── utils/                  # WHERE to compute (CPU/GPU)
```

Each folder has one job:

| Folder | Responsibility | When Used |
|--------|----------------|-----------|
| `ionic/` | Cell model equations | Shared by both paradigms |
| `tissue_builder/` | Create geometry, stimulus | Init time only |
| `simulation/classical/discretization_scheme/` | Build spatial operators | Init time |
| `simulation/classical/solver/` | Advance time | Runtime |
| `simulation/classical/state.py` | Store all runtime data | Runtime |
| `simulation/lbm/` | Self-contained LBM alternative | Runtime |
| `utils/` | Device management | Throughout |

---

## Key Design: Spatial vs Temporal Discretization

The architecture cleanly separates two orthogonal concerns:

### Spatial Discretization (`discretization_scheme/`)

Determines HOW space is represented and HOW spatial derivatives are computed.

| Method | Semi-Discrete Form | Mass Matrix M | Stiffness K |
|--------|-------------------|---------------|-------------|
| **FEM** | `M·dV/dt = -K·V + M·f` | Sparse (∫φᵢφⱼ) | Sparse (∫∇φᵢ·D·∇φⱼ) |
| **FDM** | `dV/dt = L·V + f` | Identity (implicit) | Stencil matrix L |
| **FVM** | `Vol·dV/dt = F·V + Vol·f` | Diagonal (cell volumes) | Flux operator F |

### Temporal Discretization (`solver/diffusion_time_stepping/`)

Determines HOW time is advanced using the spatial operators.

| Method | Category | Formula | Linear Solve? |
|--------|----------|---------|---------------|
| **Forward Euler** | Explicit | `V^{n+1} = V^n + dt·f(V^n)` | No |
| **RK2/RK4** | Explicit | Multi-stage | No |
| **Crank-Nicolson** | Implicit | `(M + θdt·K)V^{n+1} = (M - (1-θ)dt·K)V^n + dt·M·f` | Yes |
| **BDF1/BDF2** | Implicit | Multi-step backward difference | Yes |

### How Discretization Affects Time Stepping

The spatial discretization method influences the time stepping formulation:

**Crank-Nicolson with FEM:**
```
(M + 0.5·dt·K)·V^{n+1} = (M - 0.5·dt·K)·V^n + dt·M·f
```

**Crank-Nicolson with FDM:**
```
(I + 0.5·dt·L)·V^{n+1} = (I - 0.5·dt·L)·V^n + dt·f
```

**Crank-Nicolson with FVM:**
```
(Vol + 0.5·dt·F)·V^{n+1} = (Vol - 0.5·dt·F)·V^n + dt·Vol·f
```

The time stepper remains generic; the discretization scheme provides pre-built operators:

```python
# discretization_scheme/base.py
class SpatialDiscretization(ABC):
    @property
    @abstractmethod
    def n_dof(self) -> int:
        """Number of degrees of freedom (nodes for FEM, cells for FDM/FVM)."""
        pass

    @property
    @abstractmethod
    def coordinates(self) -> Tuple[Tensor, Tensor]:
        """(x, y) coordinates for stimulus evaluation."""
        pass

    @abstractmethod
    def get_diffusion_operators(self, dt: float, scheme: str) -> DiffusionOperators:
        """Get (A_lhs, B_rhs, apply_mass_fn) for time stepping."""
        pass

    @abstractmethod
    def apply_diffusion(self, V: Tensor) -> Tensor:
        """Compute ∇·(D∇V) directly (for explicit methods)."""
        pass
```

---

## Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INPUT                                      │
│                                                                              │
│   • Geometry (image or dimensions)                                          │
│   • Material properties (D, chi, Cm)                                        │
│   • Ionic model choice (TTP06, ORd)                                         │
│   • Discretization choice (FEM, FDM, FVM)                                   │
│   • Time stepping choice (CN, BDF, RK)                                      │
│   • Stimulus protocol                                                        │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                    ══════════════╧══════════════
                         INITIALIZATION PHASE
                    ═════════════════════════════
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ tissue_builder/ │    │ tissue_builder/ │    │     ionic/      │
│     mesh/       │    │    stimulus/    │    │    models/      │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ TriangularMesh  │    │ StimulusProtocol│    │ get_initial_    │
│ StructuredGrid  │    │ StimRegions     │    │ state(n_dof)    │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         │ geometry             │ masks, timing        │ states tensor
         │                      │                      │
         ▼                      │                      │
┌─────────────────────────┐     │                      │
│ discretization_scheme/  │     │                      │
├─────────────────────────┤     │                      │
│ User selects: "fem"     │     │                      │
│                         │     │                      │
│  ┌─────┬─────┬─────┐   │     │                      │
│  │ fem │ fdm │ fvm │   │     │                      │
│  └──┬──┴─────┴─────┘   │     │                      │
│     │                   │     │                      │
│     ▼                   │     │                      │
│ • Assemble M, K         │     │                      │
│ • Build operators for   │     │                      │
│   time stepping         │     │                      │
│ • get_diffusion_ops()   │     │                      │
└────────┬────────────────┘     │                      │
         │                      │                      │
         │ M, K, operators      │                      │
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                                ▼
              ┌─────────────────────────────────────────┐
              │     simulation/classical/state.py       │
              ├─────────────────────────────────────────┤
              │ Stores ALL simulation data:             │
              │ • n_dof, x, y (abstract geometry)       │
              │ • A_lhs, B_rhs, apply_mass (operators)  │
              │ • states (ionic)                        │
              │ • stim_masks, stim_times (stimulus)     │
              │ • spatial: SpatialDiscretization ref    │
              │ • t (current time)                      │
              └─────────────────┬───────────────────────┘
                                │
                    ════════════╧════════════
                          RUNTIME PHASE
                    ═════════════════════════
                                │
                                ▼
              ┌─────────────────────────────────────────┐
              │  simulation/classical/monodomain.py     │
              │         (Orchestrator)                  │
              └─────────────────┬───────────────────────┘
                                │
                                ▼
              ┌─────────────────────────────────────────┐
              │         solver/splitting/               │
              ├─────────────────────────────────────────┤
              │ User selects: "strang"                  │
              │                                         │
              │  ┌─────────┬────────┬───────────────┐  │
              │  │ godunov │ strang │ sym_strang    │  │
              │  └────┬────┴────────┴───────────────┘  │
              │       │                                 │
              │       ▼                                 │
              │  Determines order of ionic/diffusion   │
              └───────┬─────────────────┬───────────────┘
                      │                 │
        ┌─────────────┘                 └─────────────┐
        │                                             │
        ▼                                             ▼
┌───────────────────────────┐           ┌───────────────────────────┐
│ solver/ionic_time_stepping│           │solver/diffusion_time_step │
├───────────────────────────┤           ├───────────────────────────┤
│ User selects: "rush_larsen"│          │ User selects: "CN"        │
│                           │           │                           │
│ ┌──────────┬────────────┐ │           │ ┌──────────┬────────────┐ │
│ │rush_lars │fwd_euler   │ │           │ │ explicit │  implicit  │ │
│ └────┬─────┴────────────┘ │           │ └──────────┴─────┬──────┘ │
│      │                    │           │                  │        │
│      ▼                    │           │    ┌─────────────┘        │
│ Calls ionic/models/*:     │           │    ▼                      │
│ • compute_Iion()          │           │ ┌──────┬──────┬──────┐   │
│ • compute_gate_inf()      │           │ │  CN  │ BDF1 │ BDF2 │   │
│ • compute_gate_tau()      │           │ └──┬───┴──────┴──────┘   │
│                           │           │    │                      │
│ Returns: updated states   │           │    ▼                      │
└───────────────────────────┘           │ Needs linear solver       │
                                        │    │                      │
                                        │    ▼                      │
                                        │ ┌────────────────────┐   │
                                        │ │   linear_solver/   │   │
                                        │ ├────────────────────┤   │
                                        │ │ ┌─────┬──────────┐ │   │
                                        │ │ │ pcg │chebyshev │ │   │
                                        │ │ └──┬──┴──────────┘ │   │
                                        │ │    │               │   │
                                        │ │    ▼               │   │
                                        │ │ Solves Ax = b      │   │
                                        │ └────────────────────┘   │
                                        │                          │
                                        │ Returns: V_new           │
                                        └──────────────────────────┘
                                                    │
                                                    ▼
                                        ┌───────────────────────┐
                                        │   Update state.py     │
                                        │   • states            │
                                        │   • t += dt           │
                                        └───────────────────────┘
                                                    │
                                                    ▼
                                        ┌───────────────────────┐
                                        │   Output (periodic)   │
                                        │   • Save V to buffer  │
                                        │   • Progress callback │
                                        └───────────────────────┘
```

---

## Two Simulation Paradigms

### Classical (FEM/FDM/FVM)

Traditional PDE-based approach:
1. Spatial discretization creates operators (M, K or equivalent)
2. Time stepping advances solution
3. Implicit methods require linear solvers

```
tissue_builder/  →  discretization_scheme/  →  state.py  →  solver/
    (mesh)              (operators)            (store)      (compute)

    [init]               [init]                [init]       [runtime]
```

### Lattice-Boltzmann (LBM)

Alternative paradigm:
1. Works on Cartesian grids
2. Collision + streaming steps (no linear solve)
3. Self-contained, highly parallel
4. 10-45× faster than FEM (see LBM-EP paper)

```
LBM is completely independent — has its own:
  - state.py
  - monodomain.py
  - stimulus.py
  - lattice structures
  - collision operators
```

---

## Builder vs Storage Pattern

### `tissue_builder/` — Builders (knowledge/methods)

Creates data structures at initialization. Classes know HOW to build things.

```python
# tissue_builder/mesh/triangular.py
class TriangularMesh:
    @classmethod
    def create_rectangle(cls, Lx, Ly, nx, ny): ...

# tissue_builder/stimulus/protocol.py
class StimulusProtocol:
    def add_stimulus(region, start, duration, amplitude): ...
    def precompute(nodes, device) -> dict: ...
```

### `discretization_scheme/` — Spatial Operators

Builds matrices/operators from geometry. Provides interface for time steppers.

```python
# discretization_scheme/fem.py
class FEMDiscretization(SpatialDiscretization):
    def __init__(self, mesh, D, chi, Cm):
        self.M = assemble_mass_matrix(mesh, chi, Cm)
        self.K = assemble_stiffness_matrix(mesh, D)

    def get_diffusion_operators(self, dt, scheme):
        if scheme == "CN":
            A = (self.M + 0.5 * dt * self.K).coalesce()
            B = (self.M - 0.5 * dt * self.K).coalesce()
            return A, B, lambda f: sparse_mv(self.M, f)
        # ... other schemes

    def apply_diffusion(self, V):
        # For explicit methods
        return solve_mass(self.M, sparse_mv(self.K, V))
```

### `simulation/classical/state.py` — Storage (actual data)

Holds runtime simulation data. No algorithms — pure data. Operators and workspace buffers live in their respective solvers (see Solver ABCs section).

**Key design:** State is **scheme-agnostic**. It stores flat arrays indexed by `n_dof` (degrees of freedom), not mesh/grid-specific structures. This allows the same state.py to work with FEM, FDM, or FVM.

```python
@dataclass
class SimulationState:
    # === Discretization Reference ===
    spatial: SpatialDiscretization   # Reference to FEM/FDM/FVM discretization

    # === Abstract Geometry (from spatial.coordinates) ===
    n_dof: int                       # Number of degrees of freedom (nodes or cells)
    x: torch.Tensor                  # (n_dof,) x-coordinates (flat)
    y: torch.Tensor                  # (n_dof,) y-coordinates (flat)

    # === Cell States (from ionic/) ===
    states: torch.Tensor             # (n_dof, n_ionic_states)

    # === State Layout (from ionic model, set once at init) ===
    V_index: int                     # Column index for voltage in states
    gate_indices: List[int]          # Column indices for gating variables
    concentration_indices: List[int] # Column indices for concentrations

    # === Time ===
    t: float

    # === Stimulus (from tissue_builder/stimulus/) ===
    stim_masks: torch.Tensor         # (n_stimuli, n_dof)
    stim_starts: torch.Tensor        # (n_stimuli,)
    stim_durations: torch.Tensor     # (n_stimuli,)
    stim_amplitudes: torch.Tensor    # (n_stimuli,)

    # === Output ===
    output_buffer: torch.Tensor
    buffer_idx: int
```

**What moved OUT of state (into solvers):**

| Field | Now lives in | Why |
|-------|-------------|-----|
| `A_lhs, B_rhs, apply_mass` | `DiffusionSolver` (via `DiffusionOperators`) | Operators depend on dt; solver rebuilds if dt changes |
| `pcg_r, pcg_p, pcg_Ap` | `LinearSolver` (e.g., `PCGSolver`) | Workspace belongs to the algorithm that uses it |
| `Istim_buffer` | `IonicSolver` (local during step) | Intermediate value, doesn't need to persist |

**All solvers access state layout uniformly:**

```python
# Both ionic and diffusion solvers use the same pattern:
V = state.states[:, state.V_index]
```

**Why this works for all discretizations:**

| Field | FEM | FDM | FVM |
|-------|-----|-----|-----|
| `n_dof` | `mesh.n_nodes` | `nx * ny` | `n_cells` |
| `x, y` | `mesh.nodes[:, 0/1]` | `meshgrid().flatten()` | `cell_centers` |
| `states` | `(n_dof, s)` | `(n_dof, s)` | `(n_dof, s)` |

**Geometry-specific queries** (for visualization) go through `state.spatial`:

```python
# Visualization example
if isinstance(state.spatial, FEMDiscretization):
    mesh = state.spatial.mesh
    plot_triangles(mesh.nodes, mesh.elements, V)
elif isinstance(state.spatial, FDMDiscretization):
    V_grid = V.reshape(state.spatial.nx, state.spatial.ny)
    plot_heatmap(V_grid)
```

---

## Solver Folder Structure (Classical)

Three parallel subfolders, each one concept:

```
solver/
├── __init__.py              → Package exports (no dispatcher)
├── splitting/               → WHEN to call ionic vs diffusion (owns sub-solvers)
├── ionic_time_stepping/     → HOW to advance cell ODEs (owns IonicModel)
└── diffusion_time_stepping/ → HOW to advance spatial PDE (owns operators)
    ├── explicit/            → No linear solver needed
    ├── implicit/            → Owns a LinearSolver
    └── linear_solver/       → Solves Ax = b (owns workspace)
```

### Diffusion Time Stepping Hierarchy

```
diffusion_time_stepping/
│
├── explicit/                    # No linear solver needed
│   ├── forward_euler.py         # [TODO] 1st order, dt ≤ dx²/2D
│   ├── rk2.py                   # [TODO] 2nd order (Heun's)
│   └── rk4.py                   # [TODO] 4th order
│
├── implicit/                    # Requires linear solver
│   ├── crank_nicolson.py        # 2nd order, unconditionally stable
│   ├── bdf1.py                  # 1st order, L-stable
│   └── bdf2.py                  # 2nd order, A-stable
│
└── linear_solver/               # Solves Ax = b (used by implicit/)
    ├── pcg.py                   # Iterative, Krylov subspace
    ├── chebyshev.py             # [TODO] Iterative, polynomial (no sync)
    ├── multigrid.py             # [TODO] O(n) hierarchical
    └── fft.py                   # [TODO] O(n log n), structured grids
```

### Linear Solver Comparison

| Method | Sync Points | Best For |
|--------|-------------|----------|
| PCG | Every iteration (convergence check) | General SPD systems |
| Chebyshev | None (predetermined coefficients) | GPU, known spectrum |
| Multigrid | None (fixed V-cycles) | Large problems, O(n) |
| FFT | None | Structured grids only |

GPU-friendly options avoid inner products (dot products) which require sync.

---

## Design Principles

### 1. Physics vs Spatial vs Temporal vs Hardware vs Data

```
                        Biology   Spatial   Temporal   Hardware   Data
                        ───────   ───────   ────────   ────────   ────
ionic/models/             ✓
tissue_builder/                                                   (builds)
discretization_scheme/              ✓
solver/splitting/                            ✓
solver/ionic_time_stepping/                  ✓
solver/diffusion_time_stepping/              ✓
simulation/classical/state.py                                      ✓
simulation/lbm/                     ✓        ✓                     ✓
utils/backend.py                                        ✓
utils/platform.py                                       ✓
```

### 2. Ionic Models Are Data Providers Only

Models provide **computation functions**. They do not control stepping strategy.

```
ionic/base.py defines:              solver/ decides:
─────────────────────               ──────────────────
compute_Iion()                      When to call Iion
compute_gate_steady_states()        Rush-Larsen or Euler for gates
compute_gate_time_constants()       dt size and substep count
compute_concentration_rates()       Splitting order
get_initial_state()                 Godunov or Strang
```

### 3. Discretization Provides Operators to Solvers

DiffusionSolver builds its operators at init by calling the spatial discretization. The discretization provides raw matrices; the solver combines them for its specific time stepping scheme.

```
discretization_scheme/ provides:    DiffusionSolver uses at init:
────────────────────────────────    ────────────────────────────
get_diffusion_operators(dt, "CN")   → DiffusionOperators(A_lhs, B_rhs, apply_mass)
apply_diffusion(V)                  → ∇·(D∇V) for explicit methods
n_dof                               → state allocation
coordinates                         → stimulus evaluation
```

### 4. Solvers Own Their Computational Artifacts

Each solver holds what it needs. State is pure runtime data.

```
IonicSolver        owns: IonicModel (stateless physics functions)
DiffusionSolver    owns: DiffusionOperators (A_lhs, B_rhs, apply_mass)
LinearSolver       owns: workspace buffers (r, p, Ap)
SplittingStrategy  owns: IonicSolver + DiffusionSolver references
```

### 5. One File Per Method

Each solver/strategy is self-contained in a single file. Adding a new one means adding one file — no existing files change.

```
Adding a new discretization:      Adding a new linear solver:
1. Create fvm.py                  1. Create chebyshev.py
2. Implement SpatialDiscretization ABC   2. Implement LinearSolver ABC
3. Done.                          3. Done.
```

### 6. Builders vs Storage vs Solvers

```
tissue_builder/     →     discretization_scheme/     →     state.py
   (geometry)                (spatial operators)            (runtime data)
   (stimulus)                (coordinates)                  (ionic states)

                      DiffusionSolver     IonicSolver
                      (A_lhs, B_rhs)      (IonicModel)
                      (LinearSolver)      (Istim eval)
```

---

## IonicModel ABC — Data Provider Interface

```python
class IonicModel(ABC):
    """
    Models provide computation FUNCTIONS.
    Models do NOT control stepping order or numerical method.
    """

    # Identity
    name: str
    n_states: int
    V_index: int
    gate_indices: List[int]
    concentration_indices: List[int]

    # Initialization
    def get_initial_state(n_cells) -> Tensor: ...

    # Computation functions (called by IonicSolver)
    def compute_Iion(states) -> Tensor: ...
    def compute_gate_steady_states(V) -> Tensor: ...
    def compute_gate_time_constants(V) -> Tensor: ...
    def compute_concentration_rates(states) -> Tensor: ...

    # Convenience
    def get_voltage(states) -> Tensor: ...
    def set_voltage(states, V): ...
```

---

## SpatialDiscretization ABC — Operator Provider Interface

```python
class SpatialDiscretization(ABC):
    """
    Discretization schemes provide spatial OPERATORS.
    They do NOT control time stepping method.
    """

    # Identity
    @property
    @abstractmethod
    def n_dof(self) -> int:
        """Number of degrees of freedom (nodes for FEM, cells for FDM/FVM)."""
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

    # Operator provision
    @abstractmethod
    def get_diffusion_operators(self, dt: float, scheme: str) -> DiffusionOperators:
        """
        Get operators for time stepping.

        Returns: (A_lhs, B_rhs, apply_mass_fn) tuple
        - A_lhs: LHS matrix for implicit solve
        - B_rhs: RHS matrix for explicit part
        - apply_mass_fn: efficient M·f operation
        """
        pass

    @abstractmethod
    def apply_diffusion(self, V: Tensor) -> Tensor:
        """Compute ∇·(D∇V) directly for explicit methods."""
        pass
```

### Concrete Implementations

**FEM (unstructured mesh):**

```python
# discretization_scheme/fem.py
class FEMDiscretization(SpatialDiscretization):
    def __init__(self, mesh: TriangularMesh, D: float, chi: float, Cm: float):
        self._mesh = mesh
        self._n_dof = mesh.n_nodes
        self._x = mesh.nodes[:, 0]
        self._y = mesh.nodes[:, 1]
        self.M = assemble_mass_matrix(mesh, chi, Cm)
        self.K = assemble_stiffness_matrix(mesh, D)

    @property
    def n_dof(self) -> int:
        return self._n_dof

    @property
    def coordinates(self) -> Tuple[Tensor, Tensor]:
        return self._x, self._y

    @property
    def mass_type(self) -> MassType:
        return MassType.SPARSE

    def get_diffusion_operators(self, dt: float, scheme: str):
        if scheme == "CN":
            A = (self.M + 0.5 * dt * self.K).coalesce()
            B = (self.M - 0.5 * dt * self.K).coalesce()
            return A, B, lambda f: sparse_mv(self.M, f)
        # ... BDF1, BDF2

    # Optional: expose mesh for visualization
    @property
    def mesh(self) -> TriangularMesh:
        return self._mesh
```

**FDM (structured grid):**

```python
# discretization_scheme/fdm.py
class FDMDiscretization(SpatialDiscretization):
    def __init__(self, nx: int, ny: int, Lx: float, Ly: float, D: float):
        self._nx, self._ny = nx, ny
        self._dx, self._dy = Lx / (nx - 1), Ly / (ny - 1)
        self._n_dof = nx * ny

        # Build flat coordinate arrays
        x_1d = torch.linspace(0, Lx, nx)
        y_1d = torch.linspace(0, Ly, ny)
        xx, yy = torch.meshgrid(x_1d, y_1d, indexing='ij')
        self._x = xx.flatten()
        self._y = yy.flatten()

        # Build stencil-based Laplacian
        self.L = self._build_laplacian_matrix(D)

    @property
    def n_dof(self) -> int:
        return self._n_dof

    @property
    def coordinates(self) -> Tuple[Tensor, Tensor]:
        return self._x, self._y

    @property
    def mass_type(self) -> MassType:
        return MassType.IDENTITY  # No mass matrix in FDM

    def get_diffusion_operators(self, dt: float, scheme: str):
        if scheme == "CN":
            I = speye(self._n_dof)
            A = (I + 0.5 * dt * self.L).coalesce()
            B = (I - 0.5 * dt * self.L).coalesce()
            return A, B, lambda f: f  # Identity mass operation
        # ... BDF1, BDF2

    # Optional: grid-specific helpers
    @property
    def nx(self) -> int:
        return self._nx

    @property
    def ny(self) -> int:
        return self._ny

    def flat_to_ij(self, idx: int) -> Tuple[int, int]:
        return idx // self._ny, idx % self._ny
```

**FVM (cell-centered):**

```python
# discretization_scheme/fvm.py
class FVMDiscretization(SpatialDiscretization):
    def __init__(self, nx: int, ny: int, Lx: float, Ly: float, D: float):
        self._nx, self._ny = nx, ny
        self._n_dof = nx * ny
        dx, dy = Lx / nx, Ly / ny

        # Cell centers (offset by half cell)
        x_1d = torch.linspace(dx/2, Lx - dx/2, nx)
        y_1d = torch.linspace(dy/2, Ly - dy/2, ny)
        xx, yy = torch.meshgrid(x_1d, y_1d, indexing='ij')
        self._x = xx.flatten()
        self._y = yy.flatten()

        # Cell volumes (uniform grid)
        self._volumes = torch.full((self._n_dof,), dx * dy)

        # Build flux operator
        self.F = self._build_flux_matrix(D, dx, dy)

    @property
    def mass_type(self) -> MassType:
        return MassType.DIAGONAL  # Diagonal volume matrix

    def get_diffusion_operators(self, dt: float, scheme: str):
        if scheme == "CN":
            Vol = torch.diag(self._volumes).to_sparse()
            A = (Vol + 0.5 * dt * self.F).coalesce()
            B = (Vol - 0.5 * dt * self.F).coalesce()
            return A, B, lambda f: f * self._volumes  # Diagonal mass
        # ... BDF1, BDF2
```

---

## Solver ABCs — Runtime Interfaces

### Ownership Chain

```
MonodomainSimulation (monodomain.py)
  │
  │  builds state, constructs solvers from config strings
  │
  └── SplittingStrategy
        │  owns sub-solvers, decides call order
        │
        ├── IonicSolver
        │     │  owns IonicModel (computation provider)
        │     └── IonicModel (stateless physics functions)
        │
        └── DiffusionSolver
              │  owns LinearSolver + DiffusionOperators
              ├── DiffusionOperators (A_lhs, B_rhs, apply_mass)
              └── LinearSolver
                    └── workspace buffers (pcg_r, pcg_p, pcg_Ap)
```

All solvers operate on `state` **in-place**. No tensor allocation per step.

### DiffusionOperators

```python
@dataclass
class DiffusionOperators:
    """Pre-built matrices for implicit time stepping."""
    A_lhs: torch.Tensor      # LHS matrix (e.g., M + 0.5·dt·K for CN)
    B_rhs: torch.Tensor      # RHS matrix (e.g., M - 0.5·dt·K for CN)
    apply_mass: Callable      # M·f operation (identity/diagonal/sparse)
```

### SplittingStrategy ABC

```python
class SplittingStrategy(ABC):
    """
    Owns ionic and diffusion solvers.
    Determines the order they are called.
    """
    def __init__(self, ionic_solver: IonicSolver, diffusion_solver: DiffusionSolver):
        self.ionic_solver = ionic_solver
        self.diffusion_solver = diffusion_solver

    @abstractmethod
    def step(self, state: SimulationState, dt: float) -> None:
        """Advance state by dt using operator splitting."""
        pass


class GodunovSplitting(SplittingStrategy):
    def step(self, state, dt):
        self.ionic_solver.step(state, dt)
        self.diffusion_solver.step(state, dt)


class StrangSplitting(SplittingStrategy):
    def step(self, state, dt):
        self.ionic_solver.step(state, dt / 2)
        self.diffusion_solver.step(state, dt)
        self.ionic_solver.step(state, dt / 2)
```

### IonicSolver ABC

```python
class IonicSolver(ABC):
    """
    Owns an IonicModel. Advances ionic ODEs in-place on state.
    Evaluates Istim internally (reads stimulus data from state).
    """
    def __init__(self, ionic_model: IonicModel):
        self.ionic_model = ionic_model

    @abstractmethod
    def step(self, state: SimulationState, dt: float) -> None:
        """Advance ionic variables by dt. Modifies state.states in-place."""
        pass

    def _evaluate_Istim(self, state: SimulationState) -> torch.Tensor:
        """Compute stimulus current at current time from state's stimulus data."""
        Istim = torch.zeros(state.n_dof, device=state.states.device)
        for i in range(state.stim_masks.shape[0]):
            active = (state.t >= state.stim_starts[i] and
                      state.t < state.stim_starts[i] + state.stim_durations[i])
            if active:
                Istim += state.stim_amplitudes[i] * state.stim_masks[i]
        return Istim
```

### DiffusionSolver ABC

```python
class DiffusionSolver(ABC):
    """
    Owns DiffusionOperators (built at init from spatial discretization).
    Implicit solvers also own a LinearSolver.
    Advances the diffusion PDE in-place on state.
    """
    def __init__(self, spatial: SpatialDiscretization, dt: float):
        self.ops = self._build_operators(spatial, dt)

    @abstractmethod
    def _build_operators(self, spatial, dt) -> DiffusionOperators:
        """Build scheme-specific operators from spatial discretization."""
        pass

    @abstractmethod
    def step(self, state: SimulationState, dt: float) -> None:
        """Advance diffusion by dt. Modifies state.states[:, state.V_index] in-place."""
        pass

    def rebuild_operators(self, spatial: SpatialDiscretization, dt: float) -> None:
        """Rebuild operators when dt changes (adaptive time stepping)."""
        self.ops = self._build_operators(spatial, dt)


class CrankNicolsonSolver(DiffusionSolver):
    def __init__(self, spatial, dt, linear_solver: LinearSolver):
        self.linear_solver = linear_solver
        super().__init__(spatial, dt)

    def _build_operators(self, spatial, dt):
        A, B, apply_mass = spatial.get_diffusion_operators(dt, "CN")
        return DiffusionOperators(A_lhs=A, B_rhs=B, apply_mass=apply_mass)

    def step(self, state, dt):
        V = state.states[:, state.V_index]
        rhs = torch.sparse.mm(self.ops.B_rhs, V.unsqueeze(1)).squeeze(1)
        V_new = self.linear_solver.solve(self.ops.A_lhs, rhs)
        state.states[:, state.V_index] = V_new
```

### LinearSolver ABC

```python
class LinearSolver(ABC):
    """
    Solves Ax = b. Owns its workspace buffers.
    """
    @abstractmethod
    def solve(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Solve Ax = b, return x."""
        pass


class PCGSolver(LinearSolver):
    def __init__(self, max_iters: int = 200, tol: float = 1e-6):
        self.max_iters = max_iters
        self.tol = tol
        # Workspace allocated on first call
        self._r = None
        self._p = None
        self._Ap = None

    def solve(self, A, b):
        n = b.shape[0]
        # Lazy-init workspace on correct device
        if self._r is None or self._r.shape[0] != n:
            self._r = torch.zeros(n, device=b.device)
            self._p = torch.zeros(n, device=b.device)
            self._Ap = torch.zeros(n, device=b.device)

        # PCG iteration using self._r, self._p, self._Ap as workspace
        # ... (implementation details)
        return x
```

---

## Runtime Step Specification

### MonodomainSimulation.run()

```python
class MonodomainSimulation:
    """
    Top-level orchestrator. Builds state and solvers from config strings.
    Owns the time loop, output buffering, and progress reporting.
    """

    def __init__(self, spatial, ionic_model, stimulus, dt,
                 splitting="strang", ionic_solver="rush_larsen",
                 diffusion_solver="crank_nicolson", linear_solver="pcg"):
        # 1. Build state (MonodomainSimulation is responsible)
        x, y = spatial.coordinates
        initial_states = ionic_model.get_initial_state(spatial.n_dof)
        stim_data = stimulus.precompute(x, y, device)

        self.state = SimulationState(
            spatial=spatial,
            n_dof=spatial.n_dof,
            x=x, y=y,
            states=initial_states,
            V_index=ionic_model.V_index,
            gate_indices=ionic_model.gate_indices,
            concentration_indices=ionic_model.concentration_indices,
            t=0.0,
            stim_masks=stim_data['masks'],
            stim_starts=stim_data['starts'],
            stim_durations=stim_data['durations'],
            stim_amplitudes=stim_data['amplitudes'],
            output_buffer=...,
            buffer_idx=0,
        )

        # 2. Construct solver chain from strings
        linear = _build_linear_solver(linear_solver)
        diffusion = _build_diffusion_solver(diffusion_solver, spatial, dt, linear)
        ionic = _build_ionic_solver(ionic_solver, ionic_model)
        self.splitting = _build_splitting(splitting, ionic, diffusion)

    def run(self, t_end, save_every=1.0, callback=None):
        """
        Run simulation. Supports yield and callback.

        Args:
            t_end: End time (ms)
            save_every: Save interval (ms)
            callback: Optional fn(state) called at save points.
                      Return True to stop early.

        Yields:
            state at each save point (generator mode)
        """
        state = self.state
        dt = self.dt
        next_save = save_every

        while state.t < t_end:
            # === Core step ===
            self.splitting.step(state, dt)
            state.t += dt

            # === Output ===
            if state.t >= next_save:
                self._buffer_output(state)
                next_save += save_every

                if callback is not None:
                    if callback(state):
                        break

                yield state
```

### Splitting Step (Strang example)

```
splitting.step(state, dt):
    │
    ├── ionic_solver.step(state, dt/2)     ← half-step ionic
    │     │
    │     ├── V = state.states[:, state.V_index]
    │     ├── Iion = self.ionic_model.compute_Iion(state.states)
    │     ├── Istim = self._evaluate_Istim(state)
    │     ├── state.states[:, state.V_index] += (dt/2) * (-Iion + Istim)
    │     ├── gate_inf, gate_tau = model.compute_gate_inf/tau(V_new)
    │     ├── gates = gate_inf + (gates - gate_inf) * exp(-(dt/2) / gate_tau)
    │     └── concentrations += (dt/2) * concentration_rates
    │
    ├── diffusion_solver.step(state, dt)   ← full-step diffusion
    │     │
    │     ├── V = state.states[:, state.V_index]
    │     ├── rhs = self.ops.B_rhs @ V
    │     ├── V_new = self.linear_solver.solve(self.ops.A_lhs, rhs)
    │     └── state.states[:, state.V_index] = V_new
    │
    └── ionic_solver.step(state, dt/2)     ← half-step ionic (same as above)
```

### Rush-Larsen IonicSolver Step Detail

```python
class RushLarsenSolver(IonicSolver):
    def step(self, state, dt):
        model = self.ionic_model
        S = state.states                          # (n_dof, n_states) — reference

        # 1. Read current voltage
        V = S[:, state.V_index]                   # (n_dof,) — view

        # 2. Compute ionic current from full state
        Iion = model.compute_Iion(S)              # (n_dof,)

        # 3. Evaluate stimulus at current time
        Istim = self._evaluate_Istim(state)       # (n_dof,)

        # 4. Forward Euler on voltage
        S[:, state.V_index] += dt * (-Iion + Istim)

        # 5. Rush-Larsen exponential integration on gates
        V_new = S[:, state.V_index]
        gate_inf = model.compute_gate_steady_states(V_new)
        gate_tau = model.compute_gate_time_constants(V_new)

        for i, idx in enumerate(state.gate_indices):
            S[:, idx] = gate_inf[:, i] + (S[:, idx] - gate_inf[:, i]) * torch.exp(-dt / gate_tau[:, i])

        # 6. Forward Euler on concentrations
        conc_rates = model.compute_concentration_rates(S)
        for i, idx in enumerate(state.concentration_indices):
            S[:, idx] += dt * conc_rates[:, i]
```

---

## Component Dependency Diagram

### Classical Path

```
MonodomainSimulation (monodomain.py)
  │
  │  [init] builds from config strings:
  │
  ├──► tissue_builder/mesh/           Create geometry
  ├──► tissue_builder/stimulus/       Precompute masks, timing
  ├──► discretization_scheme/         Build spatial discretization
  │
  │  [init] assembles state + solver chain:
  │
  ├──► state.py                       Pure data (geometry, states, stim, output)
  │
  └──► SplittingStrategy              Owns sub-solvers, decides call order
        │
        ├──► IonicSolver               Owns IonicModel
        │       │
        │       └──► ionic/models/     compute_Iion(), gate_inf(), gate_tau()
        │
        └──► DiffusionSolver           Owns operators + LinearSolver
                │
                ├──► DiffusionOperators   A_lhs, B_rhs, apply_mass
                └──► LinearSolver         Solves Ax = b, owns workspace
```

### LBM Path

```
User
  │
  └──► simulation/lbm/monodomain.py
                │
                ├──► simulation/lbm/state.py
                ├──► simulation/lbm/lattice.py
                ├──► simulation/lbm/collision.py
                └──► ionic/models/{model}/
```

---

## Integration with Builder Package

The external `Builder/` package (MeshBuilder, StimBuilder) integrates at the `tissue_builder/` level:

```
Builder/                              cardiac_sim/
├── MeshBuilder/                      ├── tissue_builder/
│   └── session.py ─────────────────────► mesh/from_image.py
│       • image_array                      • Converts pixel masks to mesh
│       • conductivity_tensor              • Maps conductivity per region
│       • tissue_dimensions                • Sets physical dimensions
│
├── StimBuilder/                      ├── tissue_builder/
│   └── session.py ─────────────────────► stimulus/from_image.py
│       • get_all_masks()                  • Converts masks to StimRegions
│       • get_stim_config()                • Maps timing protocols
│
└── (discretization choice) ──────────► discretization_scheme/
                                           • User selects FEM/FDM/FVM
                                           • Routes to correct builder
```

### Integration Gaps to Address

| Gap | Location | Status |
|-----|----------|--------|
| Image → FEM mesh | `tissue_builder/mesh/from_image.py` | [TODO] |
| Image → Structured grid | `tissue_builder/mesh/from_image.py` | [TODO] |
| Conductivity tensor support | `discretization_scheme/fem.py` | [TODO] Only scalar D |
| Mask resampling | `tissue_builder/stimulus/from_image.py` | [TODO] |
| Cell type → Ionic model | Mapping needed | [TODO] |
| Voltage clamp stimulus | `tissue_builder/stimulus/` | [TODO] Not supported |

---

## User API

### Classical Approach (Option C — string-based config)

```python
from cardiac_sim.tissue_builder.mesh import TriangularMesh
from cardiac_sim.tissue_builder.stimulus import StimulusProtocol, EdgeRegion
from cardiac_sim.simulation.classical.discretization_scheme import FEMDiscretization
from cardiac_sim.simulation.classical import MonodomainSimulation

# Build geometry
mesh = TriangularMesh.create_rectangle(Lx=2.0, Ly=2.0, nx=100, ny=100)

# Build spatial discretization
spatial = FEMDiscretization(mesh, D=0.001, chi=1400.0, Cm=1.0)

# Define stimulus
stimulus = StimulusProtocol()
stimulus.add_stimulus(EdgeRegion('left', width=0.1), start=0, duration=1, amplitude=-52)

# Create simulation (strings → internal construction)
sim = MonodomainSimulation(
    spatial=spatial,
    ionic_model="ttp06",
    stimulus=stimulus,
    dt=0.02,
    splitting="strang",
    ionic_solver="rush_larsen",
    diffusion_solver="crank_nicolson",
    linear_solver="pcg",
)

# Option 1: Simple run (returns results)
times, voltages = sim.run(t_end=500.0)

# Option 2: Generator with yield
for state in sim.run(t_end=500.0, save_every=1.0):
    plot(state.states[:, state.V_index])

# Option 3: Callback with early stopping
def on_save(state):
    plot(state.states[:, state.V_index])
    return state.t > 200.0  # return True to stop

sim.run(t_end=500.0, save_every=1.0, callback=on_save)
```

### LBM Approach

```python
from cardiac_sim.simulation.lbm import LBMSimulation

# LBM is self-contained
sim = LBMSimulation(
    grid_shape=(100, 100),
    spacing=0.02,
    ionic_model="ttp06",
    dt=0.1,
)
sim.add_stimulus(region, start=0, duration=1, amplitude=-52)
times, voltages = sim.run(t_end=500.0)
```

---

## GPU Optimization Strategy

### Transfer Points in Architecture

| # | Location | Transfer | Direction | Frequency | Severity |
|---|----------|----------|-----------|-----------|----------|
| 1 | `tissue_builder/mesh/` | Node coordinates | CPU→GPU | Once | OK |
| 2 | `ionic/models/lut.py` | Lookup tables | CPU→GPU | Once | OK |
| 3 | `discretization_scheme/` | M, K matrices | CPU→GPU | Once | OK |
| 4 | `tissue_builder/stimulus/` | Region masks | CPU→GPU | Once | OK |
| 5 | `solver/ionic_time_stepping/` | Kernel overhead | N/A | Per step | Medium |
| 6 | `solver/.../linear_solver/pcg.py` | Residual norm | GPU→CPU | 30-50x/step | Bad |
| 7 | `simulation/classical/monodomain.py` | Voltage array | GPU→CPU | Per save | Major |

### Mitigation

| Fix | Where | Strategy |
|-----|-------|----------|
| Precompute stimulus masks | `tissue_builder/stimulus/` | GPU tensors at init |
| Fixed-iter or Chebyshev | `linear_solver/` | Avoid convergence checks |
| GPU output ring buffer | `state.py` | Batch transfers |
| `@torch.compile` | `ionic_time_stepping/` | Fuse GPU kernels |

### Platform-Specific Notes

```
                     NVIDIA (CUDA)              APPLE SILICON (MPS)
                     ════════════════           ════════════════════

Primary bottleneck   PCIe transfer              Kernel sync
.cpu() cost          ~1-10ms (copy)             ~10-100μs (sync only)
PCG convergence      Fixed iters (no check)     Check every 5 iters (cheap)
Output saving        GPU buffer → bulk end      Clone directly (unified mem)
Linear solver        Chebyshev (no sync)        PCG OK (cheap sync)
```

---

## Physical Equation Reference

The monodomain equation being solved:

```
χ·Cm·∂V/∂t = -χ·Iion(V, u) + ∇·(D·∇V) + Istim
```

| Term | Physics Source | Spatial | Temporal | Linear Solver |
|------|---------------|---------|----------|---------------|
| `Iion(V, u)` | `ionic/models/` | N/A | `ionic_time_stepping/` | N/A |
| `∇·(D·∇V)` | N/A | `discretization_scheme/` | `diffusion_time_stepping/` | `linear_solver/` |
| `Istim` | `tissue_builder/stimulus/` | N/A | Evaluated in `solver/` | N/A |

---

## Lattice-Boltzmann Method

LBM is a fundamentally different paradigm — it doesn't fit into the "spatial discretization + time integration + linear solver" hierarchy.

### How LBM Works

Instead of discretizing the PDE and solving linear systems, LBM works with distribution functions:

```
1. COLLISION (local relaxation):
   f*_i = f_i - A_ij(f_j - ω_j·v) + δt·ω_i·(J_in + J_out + J_stim)

2. STREAMING (shift to neighbors):
   f_i(x + e_i, t + δt) = f*_i(x, t)

3. RECOVER POTENTIAL:
   v = Σ f_i
```

### LBM Characteristics

| Property | LBM | Classical (FEM) |
|----------|-----|-----------------|
| Linear solver needed | **No** | Yes (implicit) |
| Mesh required | Cartesian grid only | Various |
| Parallelism | Embarrassingly parallel | Depends on solver |
| Time stepping | Built-in | Separate choice |
| Speed | **10-45× faster** | Baseline |

### LBM is Independent

LBM doesn't use `tissue_builder/` mesh builders or `discretization_scheme/`. It has its own complete implementation in `simulation/lbm/`.

---

## File Responsibility Matrix

| File | Owns | Knows / Decides |
|------|------|-----------------|
| `ionic/models/*` | Physics functions | Cell biology (data provider) |
| `tissue_builder/mesh/*` | Geometry classes | Domain shapes |
| `tissue_builder/stimulus/*` | Protocol classes | Regions, timing |
| `classical/discretization_scheme/*` | Spatial operators (M, K, L, F) | How to discretize space (FEM/FDM/FVM) |
| `classical/state.py` | Runtime data + layout indices | Nothing (pure data) |
| `classical/monodomain.py` | State construction, time loop, output | Solver construction from strings, user API |
| `classical/solver/splitting/*` | IonicSolver + DiffusionSolver refs | Ionic/diffusion call ordering |
| `classical/solver/ionic_time_stepping/*` | IonicModel ref | ODE numerical method, Istim evaluation |
| `classical/solver/diffusion_time_stepping/*` | DiffusionOperators (A_lhs, B_rhs) | PDE numerical method |
| `classical/solver/.../linear_solver/*` | Workspace buffers (r, p, Ap) | Linear algebra (Ax=b) |
| `simulation/lbm/*` | Everything | Self-contained paradigm |
| `utils/backend.py` | Device | Hardware selection |
| `utils/platform.py` | Profile | Platform-specific strategy |

Each file has one reason to change and one thing it owns.

---

## Research Reference Guide

All research documentation lives in `Research/openCARP_FDM_FVM/` with reference implementations in `Research/code_examples/`. The research summary index is `Research/openCARP_FDM_FVM/00_Research_Summary.md`.

### Research Documents

| # | File | Covers |
|---|------|--------|
| 01 | `01_FDM_Stencils_and_Implementation.md` | 5-pt/9-pt stencils, anisotropic diffusion, Neumann BC, sparse assembly, heterogeneous conductivity |
| 02 | `02_openCARP_FDM_FVM_Architecture.md` | Monodomain equation, conductivity tensor from fibers, FDM vs FVM, operator splitting, Rush-Larsen, tool comparison |
| 03 | `03_GPU_Linear_Solvers.md` | Chebyshev iteration, FFT/DCT solvers, AMG (AmgX/PyAMG/AMGCL), fixed-iter PCG, pipelined CG, CUDA Graphs |
| 04 | `04_LBM_EP_Implementation.md` | LBM-EP paper analysis, D2Q5/D3Q7 lattices, BGK/MRT collision, Rush-Larsen coupling, bounce-back BC, full PyTorch blueprint |

### Code Examples (`Research/code_examples/`)

| Repo | What It Is | Use For |
|------|-----------|---------|
| `torchcor/` | PyTorch FEM cardiac EP (PCG+Jacobi, CN) | `pcg.py`, `crank_nicolson.py`, sparse CSR patterns |
| `MonoAlg3D_C/` | C/CUDA cell-centered FVM cardiac EP | `fvm.py`, harmonic mean interfaces, GPU CG |
| `lettuce/` | PyTorch LBM framework (BGK/MRT, D2Q9/D3Q19) | `lbm/` module — collision, streaming, GPU patterns |
| `lbm/` | Simple Python LBM reference | LBM algorithm understanding |
| `pyamg/` | CPU algebraic multigrid | `multigrid.py` prototyping |
| `pyamgx/` | NVIDIA AmgX Python bindings | `multigrid.py` GPU deployment |
| `amgcl/` | C++ header-only AMG with CUDA backend | AMG architecture reference |
| `torch-dct/` | DCT for PyTorch (Neumann BC) | `fft.py` — DCT-based diffusion solve |
| `shape_as_points/` | Differentiable spectral Poisson (PyTorch) | `fft.py` — spectral solver patterns |
| `poisson-dirichlet-neumann/` | NumPy FFT Poisson reference | `fft.py` — algorithm validation |

### Module → Research Mapping

#### `discretization_scheme/fdm.py`
- **Read**: Report 01 (§2–§5), Report 02 (§4, §9)
- **Reference code**: `torchcor/` for sparse construction patterns
- **Key formula**: 9-point anisotropic stencil (Report 01 §2.2)
  ```
  kernel[NW] = -Dxy/(4·dx·dy)    kernel[N] = Dyy/dy²    kernel[NE] = +Dxy/(4·dx·dy)
  kernel[W]  =  Dxx/dx²           kernel[C] = -2(Dxx/dx² + Dyy/dy²)   kernel[E] = Dxx/dx²
  kernel[SW] = +Dxy/(4·dx·dy)    kernel[S] = Dyy/dy²    kernel[SE] = -Dxy/(4·dx·dy)
  ```
- **Implementation**: `F.conv2d` with `mode='replicate'` padding for Neumann BC
- **Spatially varying fibers**: Decompose into 3 convolutions for Dxx, Dyy, Dxy components, or use FVM instead
- **Stability (Forward Euler)**: `dt ≤ 1 / (2·(Dxx/dx² + Dyy/dy²) + |Dxy|/(dx·dy))`

#### `discretization_scheme/fvm.py`
- **Read**: Report 02 (§5, §6, §9)
- **Reference code**: `MonoAlg3D_C/` for cell-centered flux patterns
- **Existing code**: Engine_V5.1 `tissue/diffusion.py` `_apply_uniform_anisotropic()` is already a correct FVM
- **Key pattern**: Face flux = `Dxx_face · (V_right - V_left)/dx + Dxy_face · (cross-gradient average)`
- **Interface conductivity**: **Harmonic mean** `D_face = 2·D_left·D_right / (D_left + D_right)` — gives zero flux at scar (D=0), arithmetic mean gives D/2 (unphysical)
- **Neumann BC**: Set boundary fluxes to zero explicitly

#### `tissue_builder/mesh/structured.py`
- **Read**: Report 01 (§5), Report 02 (§4)
- **Stores**: `(Nx, Ny, dx, dy)` + domain mask + fiber angle field `theta(x,y)`
- **From Builder**: Pixel groups → domain mask + tissue labels

#### `tissue_builder/tissue/anisotropic.py`
- **Read**: Report 01 (§2.3), Report 02 (§3)
- **Key formula**: Fiber angle → tensor components
  ```python
  Dxx = D_fiber·cos²(θ) + D_cross·sin²(θ)
  Dyy = D_fiber·sin²(θ) + D_cross·cos²(θ)
  Dxy = (D_fiber - D_cross)·cos(θ)·sin(θ)
  ```
- **M-matrix condition**: `|Dxy| ≤ min(Dxx·dy/(2·dx), Dyy·dx/(2·dy))`

#### `tissue_builder/tissue/heterogeneous.py`
- **Read**: Report 01 (§6), Report 02 (§5)
- **Key rule**: Harmonic mean for scar interfaces, arithmetic mean for smooth fiber fields
- **Scar handling**: D=0 at scar → harmonic mean gives zero face flux naturally

#### `solver/diffusion_time_stepping/linear_solver/chebyshev.py`
- **Read**: Report 03 (§1) — full algorithm with PyTorch code provided
- **Key property**: **Zero global reductions** per iteration (no dot products)
- **Needs**: Eigenvalue bounds `[λ_min, λ_max]` of preconditioned operator
- **Estimation strategies** (Report 03 §1):
  1. Gershgorin circle theorem (cheapest, GPU-friendly)
  2. Power iteration (~15 SpMVs, moderate)
  3. CG-based estimation (PETSc's default, ~10 CG steps)
- **3-term recurrence**: Only SpMV + vector updates per iteration

#### `solver/diffusion_time_stepping/linear_solver/multigrid.py`
- **Read**: Report 03 (§3)
- **Reference code**: `pyamg/` (CPU prototyping), `pyamgx/` (GPU deployment)
- **GPU path**: NVIDIA AmgX via `pyamgx` — setup once, reuse hierarchy across timesteps
- **Smoother**: Chebyshev preferred over Gauss-Seidel on GPU (fully parallel)
- **Anisotropy tuning**: `strength='evolution'` in PyAMG, tune `eps_strong` in AMGCL

#### `solver/diffusion_time_stepping/linear_solver/fft.py`
- **Read**: Report 03 (§2) — full PyTorch implementations provided
- **Reference code**: `torch-dct/`, `shape_as_points/`, `poisson-dirichlet-neumann/`
- **Neumann BC**: DCT (Type II/III) via `torch-dct` package
- **Periodic BC**: `torch.fft.fftn` (native PyTorch)
- **Eigenvalues**: `λ_{i,j} = (2/dx²)(cos(πi/Nx) - 1) + (2/dy²)(cos(πj/Ny) - 1)`
- **Singularity**: Set zero-frequency component to zero for pure Neumann
- **Limitation**: Only works for isotropic diffusion on structured grids; use as preconditioner for anisotropic case

#### `solver/diffusion_time_stepping/explicit/forward_euler.py`
- **Read**: Report 01 (§7), Report 02 (§7)
- **Formula**: `V^{n+1} = V^n + dt/Cm · L·V^n`
- **CFL**: `dt ≤ Cm·h²/(4·D_max)` — typical cardiac: dt=0.01ms, h=0.025cm → CFL=0.064 (stable)
- **Note**: CFL constraint compatible with ionic model dt requirements (0.01–0.1 ms)

#### `simulation/lbm/` (all files)
- **Read**: Report 04 (full document — comprehensive blueprint)
- **Reference code**: `lettuce/` (PyTorch LBM patterns), `lbm/` (simple reference)
- **Lattice**: D2Q5 for 2D (not D2Q9), D3Q7 for 3D
- **Collision**: **MRT required** for anisotropy (BGK = isotropic only)
- **Source term**: Embed in collision step (paper's Eq. 2), not operator splitting
- **Boundary**: Bounce-back for no-flux (simple first-order, mass-conserving)
- **Memory layout**: SoA — `f: (Q, Ny, Nx)` for coalesced GPU access
- **Streaming**: `torch.roll()` for simplicity; index-based for performance
- **Stability**: `τ > 0.5` required; practical range `0.6 < τ < 1.5`
- **Key relationship**: `τ = 0.5 + D·dt/(c_s²·dx²)` where `c_s² = 1/3` (D2Q5) or `1/4` (D3Q7)
- **Full working code**: Report 04 §10 has ~300 lines of PyTorch LBM-EP class

### Reference Parameters (2D Cardiac)

```
D_fiber  = 0.001   cm²/ms    (along fibers)
D_cross  = 0.00025 cm²/ms    (across fibers, ratio 4:1)
C_m      = 1.0     μF/cm²    (membrane capacitance)
χ        = 1400    cm⁻¹      (surface-to-volume ratio)
h        = 0.025   cm        (250 μm grid spacing)
dt       = 0.01    ms        (time step)
CFL      = 0.064             (well within stability limit)
```

LBM parameters (D2Q5, same cardiac tissue):
```
τ_fiber = 0.5 + 3·D_fiber·dt/dx² = 0.548   (> 0.5, stable)
τ_cross = 0.5 + 3·D_cross·dt/dx² = 0.512   (> 0.5, stable)
```

### Implementation Priority (from Research)

| Phase | Modules | Docs | Notes |
|-------|---------|------|-------|
| 1 | `structured.py`, `fdm.py`, `forward_euler.py` | 01, 02 | Builds on Engine_V5.1; use `F.conv2d` |
| 2 | `chebyshev.py`, `fft.py` | 03 | Full PyTorch code provided in research |
| 3 | `fvm.py`, `heterogeneous.py`, `anisotropic.py` | 01, 02 | Engine_V5.1 has working FVM to refactor |
| 4 | `lbm/` (full module) | 04 | ~300 lines of working blueprint in Report 04 |

### Research PDFs

- `Research/LBM-EP.pdf` — Original LBM-EP paper (Rapaka et al.)
- `Research/12859_2023_Article_5513.pdf` — Related cardiac modeling research

---

## Migration Path

### Phase 1: Foundation
1. Create `tissue_builder/` directory structure
2. Move mesh, stimulus into `tissue_builder/`
3. Create `simulation/classical/` structure

### Phase 2: Discretization Separation
4. Create `simulation/classical/discretization_scheme/`
5. Move FEM assembly logic into `discretization_scheme/fem.py`
6. Define `SpatialDiscretization` ABC
7. Add [TODO] stubs for `fdm.py`, `fvm.py`

### Phase 3: Solver Restructure
8. Create `simulation/classical/solver/` structure
9. Extract `step()` from ionic models into `ionic_time_stepping/`
10. Refactor ionic models to data-provider interface
11. Split diffusion into `explicit/`, `implicit/`, `linear_solver/`
12. Create `solver/splitting/` with Godunov and Strang

### Phase 4: State & Orchestration
13. Create `simulation/classical/state.py`
14. Create `simulation/classical/monodomain.py` orchestrator
15. Wire up all components

### Phase 5: LBM & Extensions
16. Create `simulation/lbm/` skeleton
17. Create `utils/platform.py` with PlatformProfile
18. Update imports, tests, and examples

### Phase 6: Optimizations
19. Add GPU optimizations (Chebyshev, output buffer, etc.)
20. Implement explicit methods (RK2, RK4, Forward Euler)
21. Implement LBM (d2q7, d3q7, collision operators)

### Phase 7: Builder Integration
22. Create `tissue_builder/mesh/from_image.py`
23. Create `tissue_builder/stimulus/from_image.py`
24. Add conductivity tensor support

---

## TODO Summary

### Spatial Discretization
- [ ] `discretization_scheme/fdm.py` - Finite Difference Method
- [ ] `discretization_scheme/fvm.py` - Finite Volume Method
- [ ] Conductivity tensor support (anisotropic D)

### Temporal Discretization
- [ ] `explicit/forward_euler.py` - Forward Euler diffusion
- [ ] `explicit/rk2.py` - Heun's method
- [ ] `explicit/rk4.py` - Classical RK4

### Linear Solvers
- [ ] `linear_solver/chebyshev.py` - GPU-friendly polynomial solver
- [ ] `linear_solver/multigrid.py` - Algebraic multigrid
- [ ] `linear_solver/fft.py` - FFT solver for structured grids

### Mesh Types
- [ ] `mesh/tetrahedral.py` - 3D FEM
- [ ] `mesh/structured.py` - Structured grids for FDM/FVM

### Tissue Types
- [ ] `tissue/anisotropic.py` - Fiber orientation
- [ ] `tissue/heterogeneous.py` - Scar tissue

### LBM
- [ ] `lbm/state.py` - LBM state container
- [ ] `lbm/monodomain.py` - LBM simulation
- [ ] `lbm/lattice.py` - Lattice structures
- [ ] `lbm/d2q7.py`, `d3q7.py`, `d3q19.py` - Velocity sets
- [ ] `lbm/collision.py` - BGK, MRT operators

### Builder Integration
- [ ] `mesh/from_image.py` - MeshBuilder → mesh
- [ ] `stimulus/from_image.py` - StimBuilder → protocol
- [ ] Root Builder coordinator class
- [ ] Voltage clamp stimulus support

---

## Summary

The proposed architecture improves the codebase by:

- **Separating spatial from temporal discretization** — `discretization_scheme/` vs `solver/`
- **Separating builders from storage** — `tissue_builder/` creates data, `state.py` stores it
- **Two simulation paradigms** — `classical/` (FEM/FDM/FVM) and `lbm/` (Lattice-Boltzmann)
- **Separating physics from numerics** — ionic models provide computation functions, solvers decide how to use them
- **Explicit/implicit split** — `diffusion_time_stepping/explicit/` vs `implicit/` with `linear_solver/`
- **One file per method** — adding a new solver or discretization is adding one file
- **GPU-friendly options** — Chebyshev solver, fixed-iteration PCG, output buffering
- **Clear responsibility boundaries** — each file has one reason to change
- **Extensible to new methods** — FDM, FVM, RK2/4, multigrid, FFT all have clear homes
