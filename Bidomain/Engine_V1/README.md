# Engine V1: Bidomain Cardiac Electrophysiology Simulation Engine

High-performance cardiac electrophysiology simulation solving the **full bidomain equations**, with **decoupled solver architecture**, **three-tier elliptic solver strategy**, and **GPU-accelerated computation** via PyTorch.

## What's New (Bidomain vs Monodomain)

Engine V1 extends the proven Monodomain Engine V5.4 architecture to solve the full bidomain system.

| Concern | Monodomain (V5.4) | Bidomain (V1) |
|---------|-------------------|---------------|
| Unknowns per node | 1 (V) | 2 (Vm, phi_e) |
| Conductivity | Single D | Two tensors: D_i, D_e |
| PDE type | 1 parabolic | 1 parabolic + 1 elliptic |
| Diffusion solve | 1 N×N SPD | 2 decoupled N×N SPD solves |
| Elliptic solver | None | Spectral / PCG+Spectral / PCG+GMG |
| Boundary conditions | Hardcoded Neumann | BoundarySpec: per-edge, per-variable |
| Extracellular potential | Not available | Full phi_e field |
| Defibrillation | Cannot model | Virtual electrode polarization |
| ECG generation | Cannot model | Body surface potentials from phi_e |

---

## Architecture

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
|   |   +-- boundary.py                      # BoundarySpec, BCType, EdgeBC
|   +-- tissue/
|   |   +-- isotropic.py                     # V5.4 compat
|   |   +-- conductivity.py                  # BidomainConductivity (D_i, D_e pair)
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
|   |   |   +-- base.py                      # BidomainSpatialDiscretization ABC
|   |   |   +-- fdm.py                       # FDM: L_i, L_e 9-pt stencils (BC-aware)
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
|   |           +-- spectral.py              # Tier 1: SpectralSolver (DCT/DST/FFT unified)
|   |           +-- pcg_spectral.py          # Tier 2: PCG + spectral preconditioner
|   |           +-- multigrid.py             # GeometricMultigridPreconditioner
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

## Design Principles

**Each folder has one job:**

| Folder | Responsibility | When |
|--------|---------------|------|
| `ionic/` | Cell model equations | Shared (UNCHANGED) |
| `tissue_builder/` | Create geometry, BCs, stimulus | Init |
| `discretization/` | Build spatial operators (L_i, L_e) | Init |
| `solver/splitting/` | Decides ionic/diffusion call order | Runtime |
| `solver/ionic_stepping/` | Advances ionic ODEs | Runtime |
| `solver/diffusion_stepping/` | Decoupled parabolic + elliptic solves | Runtime |
| `solver/linear_solver/` | Solves N×N SPD sub-problems | Runtime |
| `state.py` | Store all runtime data | Runtime |
| `utils/` | Device management | Throughout |

**Key separations:**
- **Physics vs numerics** -- ionic models provide computation functions; solvers decide how to use them
- **Spatial vs temporal** -- discretization provides operators; time steppers consume them
- **BCs in mesh, not solver** -- BoundarySpec lives in StructuredGrid; solvers read it
- **Decoupled over coupled** -- two N×N SPD solves instead of one 2N×2N indefinite system

---

## Data Flow

```
                          +----------------------+
                          |     USER CONFIG       |
                          |                       |
                          |  geometry, D_i, D_e,  |
                          |  BoundarySpec,        |
                          |  ionic model, dt      |
                          +----------+-------------+
                                     |
                       ==============+===============
                            INITIALIZATION PHASE
                       ==============================
                                     |
              +----------------------+----------------------+
              |                      |                      |
              v                      v                      v
     +----------------+    +----------------+    +----------------+
     | tissue_builder |    | tissue_builder |    |    ionic/      |
     |   mesh/ +      |    |   stimulus/    |    |               |
     |  boundary.py   |    |                |    |                |
     | geometry,      |    | masks,         |    | initial state, |
     | BoundarySpec   |    | timing         |    | layout indices |
     +-------+--------+    +-------+--------+    +-------+--------+
             |                     |                     |
             v                     |                     |
     +------------------------+   |                     |
     | discretization/fdm.py  |   |                     |
     |                        |   |                     |
     | L_i, L_e (BC-aware)    |   |                     |
     | A_para, B_para, A_ellip|   |                     |
     +-------+----------------+   |                     |
             |                    |                     |
             +--------------------+---------------------+
                                  |
                                  v
                       +---------------------+
                       |   BidomainState     |
                       |                     |
                       | Vm, phi_e,          |
                       | ionic_states,       |
                       | stimulus data,      |
                       | output buffer       |
                       +----------+----------+
                                  |
                       ===========+===========
                            RUNTIME PHASE
                       =======================
                                  |
                                  v
                       +---------------------+
                       | BidomainSimulation   |
                       |   (orchestrator)     |
                       +----------+----------+
                                  |
                                  v
                       +---------------------+
                       | SplittingStrategy    |
                       | (Godunov / Strang)   |
                       +-----+----------+----+
                             |          |
               +-------------+          +-------------+
               |                                      |
               v                                      v
    +----------------------+        +-------------------------------+
    |    IonicSolver        |        | DecoupledBidomainDiffusion    |
    |                      |        |                               |
    | owns: IonicModel     |        | owns: BidomainDiscretization  |
    | does: Iion, gates,   |        |       LinearSolver (parabolic)|
    |       concentrations |        |       LinearSolver (elliptic) |
    +----------------------+        | reads: grid.boundary_spec     |
               |                    +-------------------------------+
               +------------------+-------------------+
                                  |
                                  v
                    +-----------------------------+
                    |  state.Vm, state.phi_e,     |
                    |  state.ionic_states         |
                    |  updated in-place           |
                    |  state.t += dt              |
                    +-----------------------------+
```

---

## Solver Ownership Chain

```
BidomainSimulation
  |
  +-- SplittingStrategy          <-- decides ionic/diffusion call order
        |
        +-- IonicSolver           <-- owns IonicModel (REUSED from V5.4)
        |     +-- IonicModel      <-- compute_Iion(), gate_inf(), gate_tau()
        |
        +-- DecoupledBidomainDiffusionSolver
              |
              +-- BidomainDiscretization   <-- L_i, L_e, A_para, A_ellip
              +-- LinearSolver (parabolic) <-- PCG / DCT / Chebyshev
              +-- LinearSolver (elliptic)  <-- Spectral / PCG+Spectral / PCG+GMG
```

All solvers modify `state.Vm`, `state.phi_e`, and `state.ionic_states` **in-place**. Zero tensor allocation per time step.

---

## Three-Tier Elliptic Solver Strategy

The phi_e elliptic solve dominates 60-80% of bidomain compute time:

| Tier | Solver | Transform | Iters | Valid For |
|------|--------|-----------|-------|-----------|
| 1 | Spectral Direct | DCT/DST/FFT | 0 | Isotropic, uniform BCs |
| 2 | PCG + Spectral | DCT/DST/FFT | 1-3 | Moderate anisotropy |
| 3 | PCG + GMG | N/A | 10-25 | Any coefficient field |

Auto-selected from `BoundarySpec`:
- Neumann (insulated) -> DCT
- Dirichlet (bath-coupled) -> DST
- Mixed BCs -> PCG+GMG (spectral ineligible)

---

## The Bidomain Equations

**Parabolic (transmembrane potential):**
```
chi * Cm * dVm/dt + chi * Iion(Vm, w) = div(D_i * grad(Vm)) + div(D_i * grad(phi_e)) + Istim_i
```

**Elliptic (extracellular potential):**
```
0 = div((D_i + D_e) * grad(phi_e)) + div(D_i * grad(Vm)) + Istim_e
```

With operator splitting and the decoupled approach, each diffusion time step solves:
```
Step 1 (Parabolic):  A_para * Vm^{n+1} = rhs(Vm^n, phi_e^n)    -- N×N, SPD
Step 2 (Elliptic):   A_ellip * phi_e^{n+1} = L_i * Vm^{n+1}     -- N×N, SPD
```

---

## What's Reused from Monodomain V5.4

| Component | Reuse Level | Notes |
|-----------|-------------|-------|
| `ionic/` (all models) | Direct copy | IonicModel ABC unchanged |
| `tissue_builder/mesh/` | Extended | + BoundarySpec protocol |
| `tissue_builder/stimulus/` | Direct copy | Protocol unchanged |
| `utils/backend.py` | Direct copy | Device abstraction unchanged |
| IonicSolver (rush_larsen, forward_euler) | Direct copy | ODE stepping unchanged |
| SplittingStrategy (godunov, strang) | Minor extension | Handles (Vm, phi_e) pair |
| PCGSolver, ChebyshevSolver | Direct copy | Reused for parabolic sub-solve |
| SpatialDiscretization ABC | **Rewritten** | Returns L_i, L_e, decoupled operators |
| DiffusionSolver | **Rewritten** | Decoupled parabolic + elliptic |
| SimulationState | **Extended** | Adds phi_e field |

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
| `matplotlib` | Plotting and visualization |
| `opencv-python` | Real-time animation |

---

## Documentation

| Document | Contents |
|----------|----------|
| `README.md` | This file -- high-level overview |
| `improvement.md` | Full architecture spec with ABCs, solver details, BoundarySpec |
| `FDM_CODING_PLAN.md` | FDM-focused 6-phase implementation plan with file specs |
| `IMPLEMENTATION.md` | Original 10-phase plan (superseded by FDM_CODING_PLAN.md) |
| `PROGRESS.md` | Living progress tracker -- what's done, what's next |
| `research/GPU_BIDOMAIN_LITERATURE.md` | 12-paper GPU solver literature review |
| `research/BOUNDARY_SPEEDUP_ANALYSIS.md` | Kleber boundary speedup derivation |

## References

1. Sundnes J, et al. (2006). "Computing the Electrical Activity in the Heart." Springer.
2. Plank G, et al. (2021). "The openCARP simulation environment for cardiac electrophysiology."
3. Vigmond EJ, et al. (2008). "Solvers for the cardiac bidomain equations."
4. Abdulle et al. (2024). "emRKC: Explicit Multirate for Bidomain." J. Comp. Phys.
5. ten Tusscher KHWJ, Panfilov AV (2006). "Alternans and spiral breakup in a human ventricular tissue model."
6. O'Hara T, et al. (2011). "Simulation of the Undiseased Human Cardiac Ventricular Action Potential."
7. Zhou et al. (2025). "TorchCor: FEM on GPUs via PyTorch." arXiv:2510.12011.

## License

MIT License
