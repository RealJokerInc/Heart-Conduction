# Bidomain Simulation

## Status: Complete (2026-03-16)

Merged from three completed research areas: spatial discretization, linear solvers, and time integration.

## Question
How do you discretize, solve, and time-step the cardiac bidomain (and monodomain) equations?

## Key Answer
**Spatial**: FDM (9-point stencil, face-based BCs), FEM (P1 triangular), FVM (TPFA). Face-based BCs produce symmetric SPD Laplacians required for bidomain PCG.

**Linear solvers**: Three-tier auto-selection — spectral direct (DCT/DST/FFT) for isotropic uniform grids, PCG + spectral preconditioner for moderate anisotropy, PCG + GMG for arbitrary coefficients.

**Time integration**: Operator splitting essential. Strang (2nd order) or Godunov (1st order). Ionic: Rush-Larsen exponential integrator. Diffusion: Crank-Nicolson or BDF2 (implicit, no CFL limit). Fully explicit methods fail for bidomain (elliptic equation has no time derivative).

## Engines
- **Monodomain V5.4**: FDM/FEM/FVM pluggable, 6 diffusion solvers, 77 tests
- **Bidomain V1**: Decoupled GS splitting, 3-tier elliptic solver, 38 tests

## Literature
All literature files are in `literature/`. Key documents:
- `BIDOMAIN_DISCRETIZATION.md` — FEM/FDM/FVM comprehensive guide
- `BIDOMAIN_LINEAR_SOLVERS.md` — block preconditioners, AMG, GPU
- `BIDOMAIN_SOLVER_METHODS.md` — time stepping, operator splitting
- `Summary_01_Solver_Equations.md` — monodomain/bidomain equation formulations
- `Summary_02_Discretization_Methods.md` — comparison table
