# Engine V5.4: Modular Cardiac Electrophysiology Simulation Engine

High-performance cardiac electrophysiology simulation with **modular architecture**, **multiple spatial discretization methods**, **pluggable solvers**, and **GPU-accelerated computation** via PyTorch.

## What's New in V5.4

V5.4 is a full architectural restructure of the simulation engine, separating concerns that were previously coupled:

| Concern | V5.3 | V5.4 |
|---------|------|------|
| Spatial discretization | FDM only (hardcoded) | FEM, FDM, FVM (pluggable) |
| Time stepping | Godunov only | Godunov + Strang splitting |
| Ionic stepping | `step()` inside model | Solver owns strategy (Rush-Larsen, Forward Euler) |
| Diffusion stepping | Explicit only | Explicit (FE, RK2, RK4) + Implicit (CN, BDF1, BDF2) |
| Linear solver | None (explicit) | PCG, Chebyshev, FFT/DCT |
| Alternative paradigms | None | Lattice-Boltzmann (LBM) |
| Solver construction | Manual wiring | String-based config |

---

## Architecture

```
cardiac_sim/
│
├── ionic/                                    # Cell models (shared physics)
│   ├── base.py                               # IonicModel ABC, CellType enum
│   ├── lut.py                                # Lookup table acceleration
│   ├── ord/                                  # O'Hara-Rudy 2011 (40 ionic states)
│   └── ttp06/                                # ten Tusscher-Panfilov 2006 (18 ionic states)
│       └── celltypes/                        # Standard + custom cell configs
│
├── tissue_builder/                           # Input construction (init-time only)
│   ├── mesh/                                 # Geometry
│   │   ├── base.py                           # Mesh ABC
│   │   ├── triangular.py                     # Unstructured triangles (FEM)
│   │   ├── structured.py                     # Cartesian grid (FDM/FVM/LBM)
│   │   └── loader.py                         # Load mesh from .npz (Builder)
│   ├── tissue/                               # Material properties
│   │   └── isotropic.py                      # Uniform conductivity
│   └── stimulus/                             # Pacing protocols
│       ├── protocol.py                       # StimulusProtocol
│       ├── regions.py                        # Spatial stimulus regions
│       └── loader.py                         # Load stimulus from .npz (Builder)
│
├── simulation/
│   ├── classical/                            # FEM/FDM/FVM path
│   │   ├── state.py                          # SimulationState (runtime data)
│   │   ├── monodomain.py                     # MonodomainSimulation (orchestrator)
│   │   │
│   │   ├── discretization_scheme/            # Spatial operators
│   │   │   ├── base.py                       # SpatialDiscretization ABC
│   │   │   ├── fem.py                        # Finite Element Method
│   │   │   ├── fdm.py                        # Finite Difference (9-pt, per-node D)
│   │   │   └── fvm.py                        # Finite Volume (TPFA, per-node D)
│   │   │
│   │   └── solver/                           # Time stepping
│   │       ├── splitting/                    # Operator splitting strategies
│   │       │   ├── godunov.py                # 1st order (ionic → diffusion)
│   │       │   └── strang.py                 # 2nd order (half → full → half)
│   │       ├── ionic_time_stepping/          # Ionic ODE solvers
│   │       │   ├── rush_larsen.py            # Exponential integrator
│   │       │   └── forward_euler.py          # Simple explicit
│   │       └── diffusion_time_stepping/      # Diffusion PDE solvers
│   │           ├── explicit/                 # No linear solve needed
│   │           │   ├── forward_euler.py      # dt limited by CFL
│   │           │   ├── rk2.py                # Heun's method O(dt²)
│   │           │   └── rk4.py                # Classical RK4 O(dt⁴)
│   │           ├── implicit/                 # Requires linear solver
│   │           │   ├── crank_nicolson.py     # 2nd order, unconditionally stable
│   │           │   ├── bdf1.py               # Backward Euler
│   │           │   └── bdf2.py               # 2nd-order BDF
│   │           └── linear_solver/            # Ax = b solvers
│   │               ├── pcg.py               # Preconditioned CG + Jacobi
│   │               ├── chebyshev.py          # Sync-free polynomial (GPU)
│   │               └── fft.py               # DCT (Neumann) / FFT (periodic)
│   │
│   └── lbm/                                  # Lattice-Boltzmann path
│       ├── state.py                          # LBM state (distribution functions)
│       ├── monodomain.py                     # LBM simulation entry
│       ├── collision.py                      # BGK (isotropic) / MRT (anisotropic)
│       ├── d2q5.py                           # 2D lattice velocity set
│       └── d3q7.py                           # 3D lattice velocity set
│
├── mesh_builder/                             # SVG → mesh.npz (from Builder)
├── stim_builder/                             # SVG → stim.npz (from Builder)
├── ui/                                       # Flask server with /api/export
│
├── utils/
│   └── backend.py                            # CPU/GPU device abstraction
│
└── tests/
```

---

## Design Principles

**Each folder has one job:**

| Folder | Responsibility | When |
|--------|---------------|------|
| `ionic/` | Cell model equations (physics) | Shared |
| `tissue_builder/` | Create geometry, materials, stimulus | Init |
| `discretization_scheme/` | Build spatial operators (M, K, L) | Init |
| `solver/` | Advance solution in time | Runtime |
| `state.py` | Store all runtime data | Runtime |
| `lbm/` | Self-contained LBM alternative | Runtime |
| `utils/` | Device management | Throughout |

**Key separations:**
- **Physics vs numerics** -- ionic models provide computation functions; solvers decide how to use them
- **Spatial vs temporal** -- discretization schemes provide operators; time steppers consume them
- **Builders vs storage** -- tissue_builder creates data at init; state.py holds it at runtime
- **Solvers own their artifacts** -- operators, workspace buffers, and sub-solvers live in the solver that uses them, not in state

---

## Data Flow

```
                          ┌──────────────────────┐
                          │     USER CONFIG       │
                          │                       │
                          │  geometry, materials,  │
                          │  ionic model, dt,      │
                          │  solver choices        │
                          └──────────┬─────────────┘
                                     │
                       ══════════════╧═══════════════
                            INITIALIZATION PHASE
                       ══════════════════════════════
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
     ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
     │ tissue_builder │    │ tissue_builder │    │    ionic/      │
     │    mesh/       │    │   stimulus/    │    │               │
     │                │    │                │    │                │
     │ geometry,      │    │ masks,         │    │ initial state, │
     │ coordinates    │    │ timing         │    │ layout indices │
     └───────┬────────┘    └───────┬────────┘    └───────┬────────┘
             │                     │                     │
             ▼                     │                     │
     ┌────────────────┐            │                     │
     │ discretization │            │                     │
     │    scheme/     │            │                     │
     │                │            │                     │
     │ M, K or L      │            │                     │
     │ (spatial ops)  │            │                     │
     └───────┬────────┘            │                     │
             │                     │                     │
             └─────────────────────┼─────────────────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   SimulationState    │
                        │                      │
                        │ V, ionic_states,     │
                        │ stimulus data,       │
                        │ output buffer        │
                        └──────────┬───────────┘
                                   │
                       ════════════╧═══════════
                             RUNTIME PHASE
                       ════════════════════════
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │ MonodomainSimulation │
                        │   (orchestrator)     │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │ SplittingStrategy    │
                        │ (Godunov / Strang)   │
                        └─────┬──────────┬─────┘
                              │          │
                ┌─────────────┘          └─────────────┐
                │                                      │
                ▼                                      ▼
     ┌─────────────────────┐            ┌─────────────────────┐
     │    IonicSolver       │            │   DiffusionSolver    │
     │                      │            │                      │
     │ owns: IonicModel     │            │ owns: operators      │
     │ does: Iion, gates,   │            │       (A_lhs, B_rhs) │
     │       concentrations,│            │ owns: LinearSolver   │
     │       Istim eval     │            │       (workspace)    │
     └─────────────────────┘            └─────────────────────┘
                │                                      │
                └──────────────┬───────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  state.V,            │
                    │  state.ionic_states  │
                    │  updated in-place    │
                    │  state.t += dt       │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  yield / callback    │
                    │  at save points      │
                    └──────────────────────┘
```

---

## Solver Ownership Chain

```
MonodomainSimulation
  │
  └── SplittingStrategy          ← decides ionic/diffusion call order
        │
        ├── IonicSolver           ← owns IonicModel (stateless physics)
        │     └── IonicModel      ← compute_Iion(), gate_inf(), gate_tau()
        │
        └── DiffusionSolver       ← owns operators + linear solver
              ├── DiffusionOperators  ← A_lhs, B_rhs, apply_mass
              └── LinearSolver        ← solves Ax=b, owns workspace (r, p, Ap)
```

All solvers modify `state.V` and `state.ionic_states` **in-place**. Zero tensor allocation per time step.

---

## Two Simulation Paradigms

### Classical (FEM / FDM / FVM)

Traditional PDE discretization with operator splitting:

```
tissue_builder → discretization_scheme → state → solver
   (mesh)            (operators)         (data)   (compute)

   [init]              [init]            [init]    [runtime]
```

Spatial and temporal discretization are orthogonal — any spatial method can pair with any time stepper.

### Lattice-Boltzmann (LBM)

Alternative paradigm — no linear system solve:

```
collision → streaming → bounce-back → recover voltage
  (local)    (neighbor)   (boundary)    (sum distributions)
```

- Embarrassingly parallel (perfect GPU fit)
- 10-45x faster than FEM on CPU (Rapaka et al., MICCAI 2012)
- Self-contained in `simulation/lbm/` — shares only `ionic/` with classical path

---

## Spatial vs Temporal Discretization

These two concerns are cleanly separated. The spatial method provides operators; the time stepper consumes them.

### Spatial Methods

| Method | Mass Matrix | Stiffness/Operator | Mesh Type |
|--------|------------|-------------------|-----------|
| FEM | Sparse (integral-based) | Sparse stiffness K | Unstructured triangles |
| FDM | Identity (implicit) | 9-pt stencil Laplacian L (per-node D, harmonic mean) | Structured grid |
| FVM | Diagonal (cell volumes) | TPFA flux operator F (per-node D, harmonic mean) | Structured grid |

### Time Steppers

| Method | Category | Linear Solve? | Stability |
|--------|----------|--------------|-----------|
| Forward Euler | Explicit | No | CFL-limited |
| RK2 / RK4 | Explicit | No | CFL-limited |
| Crank-Nicolson | Implicit | Yes | Unconditional |
| BDF1 / BDF2 | Implicit | Yes | Unconditional |

### Linear Solvers (for implicit methods)

| Solver | GPU Sync Points/Iter | Best For |
|--------|---------------------|----------|
| PCG | 2-3 (dot products) | General SPD systems |
| Chebyshev | 0 (sync-free) | Max GPU throughput |
| FFT/DCT | 0 (direct) | Structured grids, isotropic |

---

## Dependencies

### Required

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.9+ | Runtime |
| PyTorch | 2.0+ | Tensor computation, GPU acceleration |
| NumPy | 1.24+ | Array utilities |
| SciPy | 1.10+ | Sparse matrix construction |

### Optional

| Package | Purpose |
|---------|---------|
| `cairosvg` | SVG image loading (Builder integration) |
| `Pillow` | Image processing (Builder integration) |
| `torch-dct` | DCT solver for Neumann BC (FFT linear solver) |
| `matplotlib` | Plotting and visualization |
| `opencv-python` | Real-time animation |

### Installation

```bash
cd "Monodomain/Engine_V5.4"
source ../../venv/bin/activate

# Core dependencies
pip install torch numpy scipy

# Optional (Builder integration)
pip install cairosvg pillow

# Optional (advanced solvers)
pip install torch-dct
```

---

## Ionic Models

Physics-only data providers. Models supply computation functions; solvers decide stepping strategy.

| Model | States | Source |
|-------|--------|--------|
| O'Hara-Rudy 2011 | 40 | `ionic/ord/` |
| ten Tusscher-Panfilov 2006 | 18 | `ionic/ttp06/` |

State counts are ionic states only (gates + concentrations). Voltage (V) is stored separately.

Each model exposes:
- `compute_Iion(V, ionic_states)` -- total ionic current
- `compute_gate_steady_states(V, ionic_states)` -- gate infinity values
- `compute_gate_time_constants(V, ionic_states)` -- gate tau values
- `compute_concentration_rates(V, ionic_states)` -- concentration derivatives
- `get_initial_state(n_cells)` -- initial conditions (without V)
- Layout info: `V_rest`, `gate_indices`, `concentration_indices`

---

## Builder Integration

The Builder pipeline converts draw.io SVGs into simulation-ready `.npz` files:

```
SVG → mesh_builder/export.py → mesh.npz → tissue_builder/mesh/loader.py → StructuredGrid + D_field
SVG → stim_builder/export.py → stim.npz → tissue_builder/stimulus/loader.py → StimulusProtocol
```

- `mesh_builder/` and `stim_builder/` are self-contained copies of the external `Builder/` tools
- `ui/` provides a Flask server with `/api/export` for interactive SVG editing
- `.npz` files contain per-node conductivity arrays (D_xx, D_yy, D_xy) in grid convention (Nx, Ny)
- Loaders flatten arrays to active-node ordering matching `StructuredGrid.from_mask()`

---

## Physical Equation

The monodomain equation governing cardiac electrical propagation:

```
χ · Cm · dV/dt = -χ · Iion(V, u) + div(D · grad(V)) + Istim
```

| Symbol | Meaning | Typical Value |
|--------|---------|---------------|
| V | Transmembrane potential | -87 to +40 mV |
| Cm | Membrane capacitance | 1.0 uF/cm^2 |
| chi | Surface-to-volume ratio | 1400 cm^-1 |
| D | Conductivity tensor | 0.001 cm^2/ms (fiber) |
| Iion | Ionic current (from cell model) | -- |
| Istim | Stimulus current | 52 uA/cm^2 |

---

## References

1. O'Hara T, et al. (2011). "Simulation of the Undiseased Human Cardiac Ventricular Action Potential." *PLoS Comput Biol*.
2. ten Tusscher KHWJ, Panfilov AV (2006). "Alternans and spiral breakup in a human ventricular tissue model." *Am J Physiol Heart Circ Physiol*.
3. Rush S, Larsen H (1978). "A practical algorithm for solving dynamic membrane equations." *IEEE Trans Biomed Eng*.
4. Rapaka S, et al. (2012). "LBM-EP: Lattice-Boltzmann Method for Fast Cardiac Electrophysiology Simulation." *MICCAI*.
5. Plank G, et al. (2021). "The openCARP simulation environment for cardiac electrophysiology." *Comput Methods Programs Biomed*.

---

## Documentation

| Document | Contents |
|----------|----------|
| `README.md` | This file -- high-level overview |
| `improvement.md` | Full architecture spec with ABCs, solver details, research references |
| `IMPLEMENTATION.md` | 8-phase implementation plan with 70+ validation tests |
| `PROGRESS.md` | Living progress tracker -- what's done, what's next |

## License

MIT License
