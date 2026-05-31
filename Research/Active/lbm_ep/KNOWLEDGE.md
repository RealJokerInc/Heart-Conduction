# LBM-EP — Knowledge File

> This file is a running synthesis. Updated as findings accumulate.
> When the question is complete, a copy is promoted to `Research/Knowledge/`.
>
> **Reopened 2026-04-19** as `lbm_ep` (was `lbm_cardiac`, completed 2026-03-16). Content below is the snapshot from completion — preserved as foundation. New findings appended in dated sections at the bottom.

## Foundation (from lbm_cardiac, 2026-03-16)

### Current Understanding

Lattice Boltzmann Methods (LBM) can solve the monodomain cardiac electrophysiology equations and are a feasible (but complex) path for the bidomain. LBM replaces traditional PDE discretization (FDM/FEM/FVM) with a kinetic scheme: distribute, collide, stream. The method is naturally parallel and GPU-friendly because it requires no linear system assembly or solve.

### Monodomain LBM (implemented in LBM V1)

**Lattice types:**
- **D2Q5** (5 velocities): Handles isotropic diffusion. Simpler, fewer operations per node.
- **D2Q9** (9 velocities): Required for anisotropic diffusion tensors with cross-derivative terms (D_xy != 0). More memory and compute, but handles full 2x2 conductivity tensor.

**Collision operators:**
- **BGK** (single relaxation time tau): Simplest. tau = 0.5 + D * dt / (cs^2 * dx^2), where cs^2 = 1/3 for D2Q9. Works well for moderate diffusion.
- **MRT** (multi-relaxation time): Independent relaxation rates for each moment. Better stability and accuracy, especially near tau = 0.5. Required for anisotropic diffusion with D2Q9.

**Key numerical characteristic:** CV is approximately 35% higher than FDM at the same spatial resolution due to numerical dispersion inherent to the LBM discretization. This is not a bug but a property of the scheme; it converges to the correct value with mesh refinement.

**Performance:** 10-45x speedup over FEM reported in literature (Rapaka 2012, Campos 2016). All operations are local (no global communication beyond nearest neighbors), making LBM ideal for GPU parallelism.

### Bidomain LBM (research phase)

Three architectures were evaluated:

| Architecture | Approach | Verdict |
|--------------|----------|---------|
| **A: Dual-Lattice LBM** | Two independent LBM lattices (Vm, phi_e), coupled via source terms | RECOMMENDED -- pure LBM, literature precedent (Belmiloudi 2015-2019) |
| **B: Hybrid LBM-Classical** | LBM for parabolic Vm, iterative solver for elliptic phi_e | Conservative fallback -- proven separately |
| **C: Single Enlarged LBM** | Both fields treated as parabolic in one lattice | NOT RECOMMENDED -- breaks elliptic physics |

The central challenge is the elliptic equation for phi_e, which LBM handles via pseudo-time stepping to steady state. The critical unknown is convergence rate: literature suggests 50-200 pseudo-time iterations per physical step, potentially reducible to 10-50 with multigrid acceleration.

**Feasibility verdict:** HIGH (theoretically sound, literature precedent), but no production implementation exists for cardiac bidomain LBM.

### Key references

- **Rapaka et al. 2012**: LBM-EP framework from Siemens, 3D cardiac geometry, Mitchell-Schaeffer ionic model
- **Campos et al. 2016**: GPU-native LBM monodomain implementation
- **Belmiloudi et al. 2015-2019**: Coupled LBM approaches for cardiac bidomain (dual-lattice and hybrid LBM-FV)
- **Chai & Zhao 2012**: Multigrid LBM for elliptic equations

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Lattice for isotropic | D2Q5 | Fewer ops, sufficient for isotropic diffusion |
| Lattice for anisotropic | D2Q9 | Required for cross-derivative terms |
| Collision default | BGK | Simpler; MRT added for stability when needed |
| Unit conversion | sigma_to_D() / tau_from_D() | Centralized in src/diffusion.py |
| Grid convention | (Nx, Ny) | Matches V5.4 convention |
| Bidomain architecture | Dual-Lattice (Architecture A) | Pure LBM, literature validated |
| Bidomain status | Research only, no implementation | Pseudo-time convergence is unproven for cardiac bidomain |

## Open Questions

- What is the actual pseudo-time convergence rate for the cardiac bidomain elliptic equation? (50-200 iterations is a wide range)
- Can multigrid acceleration reduce pseudo-time iterations to under 20? (This determines whether bidomain LBM is competitive with FEM)
- How does MRT stability compare to BGK near tau = 0.5 for realistic cardiac conductivities?
- Is the 35% CV offset from FDM a concern for validation, or does it converge away at practical resolutions?

## Connections
- **Engines**: LBM V1 (monodomain implementation, D2Q5/D2Q9, BGK/MRT), Monodomain V5.4 (LBM path integrated)
- **Related research**: boundary_conduction_speedup (D2Q9 with Dirichlet BC captures Kleber effect), scar_bc_validity (bounce-back BC for no-flux at scar)
- **Pipelines**: None currently; bidomain LBM would be a separate engine if pursued
