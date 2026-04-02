# Q3: What time-stepping methods work for cardiac electrophysiology?

## Short Answer

**Operator splitting** is essential: separate the stiff ionic ODEs from the diffusion PDE. Godunov splitting (1st order: ionic → diffusion) is simplest. Strang splitting (2nd order: half-ionic → diffusion → half-ionic) is more accurate for the same dt.

For ionic stepping, **Rush-Larsen** (exponential integrator exploiting gate variable structure) is standard. For diffusion stepping, Crank-Nicolson or BDF2 (implicit, no CFL limit) are preferred. Explicit methods (ForwardEuler, RK2, RK4) work but are CFL-limited. IMEX-SBDF2 and RKC are stabilized explicit alternatives that avoid linear solves.

Fully explicit methods **fail** for bidomain (the elliptic equation has no time derivative — it must be solved implicitly or via operator splitting).

## Key Files in This Folder

| File | Contents |
|------|----------|
| `BIDOMAIN_EXPLICIT_METHODS.md` | IMEX, RKC, stabilized methods, CFL analysis (12 sections) |
| `BIDOMAIN_SOLVER_METHODS.md` | Time-stepping & operator splitting, decoupled vs coupled (12 sections) |
| `Summary_01_Solver_Equations.md` | Monodomain/bidomain equation formulations |

## Connected Questions

- **Q1** — Spatial discretization determines the stiffness matrix used in implicit steps
- **Q2** — Implicit time steppers require a linear solve at each step
