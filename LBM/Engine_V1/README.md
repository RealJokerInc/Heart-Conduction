# LBM_V1 — Lattice-Boltzmann Monodomain for Boundary Conduction Effects

## Purpose

Test the hypothesis that a D2Q9 Lattice-Boltzmann simulation with Dirichlet boundary
conditions reproduces the **boundary speedup effect** documented by Andre Bhatt/Kleber:
decreased electrical loading at tissue boundaries increases conduction velocity and
safety factor. This effect cannot be captured by:

- Standard monodomain FDM (no concept of loading)
- LBM with D2Q5 (insufficient velocity directions to resolve boundary gradients)
- Any method with only Neumann (no-flux) BCs

The key insight: D2Q9 has diagonal velocity directions that interact with boundaries
differently than cardinal-only D2Q5. When a Dirichlet condition (or absorbing boundary)
is applied, the distribution functions at boundary-adjacent nodes experience **asymmetric
loading** — fewer contributing neighbors on the boundary side. This mimics the reduced
electrotonic load that real cardiac tissue experiences at free edges.

## Architecture

Self-contained implementation. Ionic models (TTP06, ORd) copied from Engine V5.4.
Dependencies: PyTorch, numpy, matplotlib.

```
LBM_V1/
├── README.md                    <- This file
├── IMPLEMENTATION.md            <- Phase-by-phase build plan, code structure
├── PROGRESS.md                  <- Living checkpoint
├── CONVENTIONS.md               <- Symbol naming, tau-D mapping, moment spaces
│
├── research/                    <- Paper extracts, comparison docs
│   ├── PAPER_COMPARISON.md      <- Rapaka vs Campos: MRT, tau, optimizations
│   ├── rapaka_2012.txt
│   └── campos_2016.txt
│
├── ionic/                       <- COPIED from V5.4 (TTP06, ORd, LUT, base ABC)
│   ├── __init__.py
│   ├── base.py                  <- IonicModel ABC, CellType enum
│   ├── lut.py
│   ├── ttp06/                   <- ten Tusscher-Panfilov 2006 (18 ionic states)
│   └── ord/                     <- O'Hara-Rudy 2011 (40 ionic states)
│
├── src/
│   ├── __init__.py
│   │
│   ├── lattice/                 <- Lattice definitions (constants only)
│   │   ├── base.py              <- Lattice ABC
│   │   ├── d2q5.py              <- D2Q5 frozen dataclass singleton
│   │   └── d2q9.py              <- D2Q9 frozen dataclass singleton
│   │
│   ├── collision/               <- Collision operators (Layer 1 + Layer 2)
│   │   ├── base.py              <- CollisionOperator ABC
│   │   ├── bgk.py               <- BGKCollision (isotropic, either lattice)
│   │   └── mrt/                 <- Multiple relaxation time
│   │       ├── d2q5.py          <- MRT_D2Q5 (no D_xy)
│   │       └── d2q9.py          <- MRT_D2Q9 (full tensor)
│   │
│   ├── streaming/               <- Streaming (advection step)
│   │   ├── d2q5.py              <- stream_d2q5() pure function
│   │   └── d2q9.py              <- stream_d2q9() pure function
│   │
│   ├── boundary/                <- Boundary conditions
│   │   ├── neumann.py           <- Bounce-back (no-flux)
│   │   ├── dirichlet.py         <- Anti-bounce-back (fixed voltage)
│   │   └── absorbing.py         <- Equilibrium incoming (open boundary)
│   │
│   ├── solver/                  <- Ionic time-stepping
│   │   └── rush_larsen.py       <- Standalone RL step function
│   │
│   ├── diffusion.py             <- sigma_to_D(), tau_from_D(), tau_tensor_from_D()
│   ├── state.py                 <- LBMState dataclass
│   ├── step.py                  <- @torch.compile'd fused step functions
│   └── simulation.py            <- Master coordinator (lattice-agnostic)
│
├── tests/
│   ├── test_lattice.py          <- Lattice constants, weight sums, isotropy
│   ├── test_collision.py        <- BGK + MRT collision correctness
│   ├── test_streaming.py        <- Streaming + state + voltage recovery
│   ├── test_boundary.py         <- Neumann, Dirichlet, absorbing BC tests
│   ├── test_diffusion.py        <- Pure diffusion convergence (both lattices)
│   ├── test_ionic.py            <- RL step, single-cell AP with TTP06
│   └── test_simulation.py       <- Full simulation integration tests
│
└── experiments/
    ├── exp01_planar_wave.py      <- Basic planar propagation
    ├── exp02_boundary_cv.py      <- CV measurement near boundary
    └── exp03_d2q5_vs_d2q9.py    <- D2Q5 vs D2Q9 boundary speedup comparison
```

### Design Principles

The code follows a **two-layer architecture** for torch.compile optimization:
- **Layer 1 (OOP):** Classes that hold parameters and provide user-facing APIs.
- **Layer 2 (Pure functions):** Stateless tensor functions that torch.compile can fuse
  into efficient GPU kernels. `step.py` composes these into compiled step functions.

Each concern is isolated into its own folder (`lattice/`, `collision/`, `streaming/`,
`boundary/`, `solver/`). The **master coordinator** (`simulation.py`) selects the right
compiled step function at init time based on lattice/collision choice.

See CONVENTIONS.md for the full symbol -> code name mapping and the sigma -> D -> tau pipeline.

## Key Design Decisions

1. **PyTorch** — Matches V5.4 ionic models (direct reuse of TTP06/ORd), MPS on macOS,
   torch.compile for kernel fusion, eager-mode debugging during validation.

2. **D2Q9 as primary lattice** — 9 velocities (4 cardinal + 4 diagonal + rest).
   Moments p_xx and p_xy encode the full 2x2 diffusion tensor including D_xy.
   D2Q5 cannot encode off-diagonal diffusion (no diagonal velocities).

3. **MRT collision with hand-coded moments** — No matrix multiply at runtime.
   Pre-expand M^{-1} S M into explicit per-moment arithmetic (follows Campos optimization).

4. **Source-in-collision** — Following Rapaka Eq. 2, the reactive collision operator
   Omega_R = w_i * R is added directly in the collision step.

5. **Multiple BC types** — Neumann (bounce-back), Dirichlet (anti-bounce-back),
   absorbing (equilibrium incoming). Boundary speedup requires Dirichlet.

6. **Ionic models from V5.4** — Full TTP06 and ORd copied as-is. Rush-Larsen
   reimplemented as standalone function (no SimulationState dependency).

7. **float64 for validation, float32 for production** — Start with f64, switch
   after physics is validated.
