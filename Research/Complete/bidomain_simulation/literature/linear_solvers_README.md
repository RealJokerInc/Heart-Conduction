# Q2: How do I solve the linear systems efficiently for cardiac problems?

## Short Answer

Three tiers, selected by problem structure:
1. **Spectral direct solve** (DCT/DST/FFT) — zero iterations, only works for isotropic uniform grids with pure Neumann or pure Dirichlet BCs
2. **PCG + spectral preconditioner** — 1-3 iterations, handles moderate anisotropy and mixed BCs
3. **PCG + geometric multigrid (GMG)** — 10-25 iterations, arbitrary coefficient fields

The elliptic solve (for phi_e in bidomain) is the bottleneck. AMG preconditioners (AMGX, pyamg, amgcl) are the state of the art for production codes. For GPU, spectral methods dominate when applicable.

## Key Files in This Folder

| File | Contents |
|------|----------|
| `BIDOMAIN_LINEAR_SOLVERS.md` | Comprehensive solver research (2320 lines, 35+ refs). Block preconditioners, AMG, Krylov methods, GPU optimization. |
| `QUICK_START.txt` | Decision trees for solver and library selection |
| `03_GPU_Linear_Solvers.md` | cuSPARSE, AMGX, NVIDIA optimization strategies |

## Connected Questions

- **Q1** — Discretization determines the matrix structure (SPD, block, etc.)
- **Q3** — Time stepping determines how often you solve (implicit → every step)
