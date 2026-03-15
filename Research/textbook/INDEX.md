# Cardiac Computational Modeling — Document Index

**Source**: `bidomain_textbook.html` (~12,300 lines)
**Output**: `Cardiac_Computational_Modeling.pdf`
**Standalone LBM**: `LBM_Textbook_Part_IV.pdf`
**Last updated**: 2026-03-08 (session 14, four-appendix restructure: A trimmed, B LA new, C NumAn new, D PyTorch renumbered)

## Part I — Single Cell Dynamics

| Ch | Title | Sections | Style Status |
|----|-------|----------|-------------|
| 1 | The Hodgkin–Huxley Revolution | 1.1–1.4 | Good — Feynman style, analogies, figures; engine box verified |
| 2 | From Neurons to Heart Cells | 2.1–2.3 | Good — camera flash/light switch analogy |
| 3 | Anatomy of the Cardiac Action Potential | 3.1–3.5 | Good — phase-by-phase narrative |
| 4 | Intracellular Calcium: The Master Regulator | 4.1–4.8 | Good — SVGs pending literature image replacement |
| 5 | The ten Tusscher–Panfilov 2006 Model | 5.1–5.8 | **Verified** — equations checked against Engine V5.4 code |
| 6 | The O'Hara–Rudy 2011 Model | 6.1–6.10 | **Verified** — equations checked against Engine V5.4 code |

## Part II — Tissue-Level Monodomain Modeling (Ch 7–11)

| Ch | Title | Sections | Style Status |
|----|-------|----------|-------------|
| 7 | The Monodomain Equation | 7.1–7.4 | Good — ELI5 + derivation. **7.4 NEW**: BC intro (swimming pool analogy, Neumann/Dirichlet/Robin) |
| 8 | Spatial Discretization | 8.1–8.7 | Good — FDM/FEM/FVM worked examples, comparison table. 8.7 BCs (subsections 8.7.1–8.7.4) |
| 9 | Operator Splitting: Divide and Conquer | 9.1–9.4 | Good — engine box rewritten to match actual splitting code |
| 10 | Explicit Diffusion Solvers | 10.1–10.7 | **Verified** — RK2/RK4 with discrete matrix forms matching code |
| 11 | Implicit Diffusion Solvers | 11.1–11.6 | **Verified** — BDF1/CN/BDF2 with boxed A_lhs/B_rhs matrix forms matching code |

## Part III — Bidomain Modeling (Ch 12–15)

| Ch | Title | Sections | Style Status |
|----|-------|----------|-------------|
| 12 | The Bidomain Model | 12.1–12.7 | **Rewritten session 13**: 12.1–12.3 (physical picture), 12.4 (monodomain review), 12.5 (bidomain derivation with explicit algebra, physical dictionary, worked example), 12.6 (parabolic–elliptic couple with strategy preview), 12.7 (BCs). Convention translation box (σ vs D). Old 12.7–12.8 → sidebars. |
| 13 | From Equations to Matrices | 13.1–13.5 | **Rewritten session 13b**: 13.1 (two stiffness matrices, road network analogy), 13.2 (block system with D-form definitions), 13.3 (**NEW** face-based stencil), 13.4 (FEM weak form brief note), 13.5 (**NEW** 5-node cable worked example with numerical values) |
| 14 | Solving the Coupled System | 14.1–14.6 | **NEW chapter session 13b**: 14.1 (why explicit fails, puppet analogy), 14.2 (**Algorithm 14.1** — boxed GS procedure), 14.3 (4 more strategies: semi-implicit, Jacobi, SBDF2, RKC), 14.4 (comparison table), 14.5 (worked example), 14.6 (operator splitting). Monolithic approach as sidebar. |
| 15 | Linear Solvers and Implementation | 15.1–15.4 | **NEW chapter session 13b** (merged old 15+16+17): 15.1 (parabolic = monodomain, cross-ref), 15.2 (elliptic + null-space handling), 15.3 (**NEW** three-tier auto-selection), 15.4 (**NEW** Engine V1 architecture with real classes/paths) |

## Part IV — Lattice-Boltzmann Methods (Ch 18–20)

| Ch | Title | Sections | Style Status |
|----|-------|----------|-------------|
| 18 | The Lattice-Boltzmann Method: From Kinetic Theory to Computation | 18.1–18.5 | **Rewritten v4.0** — "Quadrature First" quadrature, eqs 18.1–18.41 + 18.7a |
| 19 | Lattice-Boltzmann Methods for Monodomain | 19.1–19.8 | **Rewritten v2.1** — Ω^NR/Ω^R notation, expanded BC section (19.5.1–19.5.5); eqs (19.1)–(19.11) + (19.10b) |
| 20 | Lattice-Boltzmann Methods for Bidomain | 20.1–20.5 | Dual-lattice, pseudo-time, hybrid approaches; eqs (20.1)–(20.4) |

## Appendices

| Ch | Title | Sections | Style Status |
|----|-------|----------|-------------|
| Refs | Key References | 4 groups | Fine |
| A | A Visual Guide to Differential Equations | A.1–A.7 | Good — 3B1B style. Trimmed: A.8–A.10 moved to Appendix C. Cross-ref fix: Ch 17→Part IV. |
| B | Linear Algebra | B.1–B.12 | **Revised session 15**: Added 3 SVGs (B.1 transformation, B.2 SPD bowl, B.3 condition number contours), 2×2 eigenvalue worked example (B.7), numerical Schur example (B.11), softened B.1 vector definition to align with 3B1B. Eqs (B.1)–(B.8). |
| C | Numerical Analysis: The Bridge | C.1–C.14 | **Rewritten session 15**: Method-by-method on 2D grid. C.1 (3×3 grid, build 9×9 Laplacian), C.2 (truncation error), C.3 (Forward Euler + 2D worked example), C.4 (stability/CFL/blowup demo), C.5 (Backward Euler + 2D worked example), C.6 (Crank-Nicolson + comparison table), C.7 (Lax), C.8 (operator splitting with Godunov/Strang), C.9 (3 solver families table), C.10 (spectral/DCT/DST + 1D worked example), C.11 (iterative intro), C.12 (CG: algorithm + 4-point worked example), C.13 (PCG: algorithm + preconditioning), C.14 (Chebyshev: algorithm + GPU workflow). Eqs (C.1)–(C.12). |
| D | Introduction to PyTorch | D.1–D.13 | Renumbered from Appendix B. Content unchanged. |

## Equation Number Registry

| Chapter | Range | Key Equations |
|---------|-------|---------------|
| 1 (HH) | (1.1)–(1.3) | 1.1 = membrane current, 1.2 = gate ODE, 1.3 = relaxation form |
| 2 (Minimal) | (2.1)–(2.2) | 2.1 = FitzHugh-Nagumo, 2.2 = Fenton-Karma |
| 5 (TTP06) | (5.1)–(5.12) | 5.1 = I_ion total, 5.3 = ICaL GHK (with V-15 shift) |
| 6 (ORd) | (6.1)–(6.20) | 6.1 = I_ion total, 6.8 = IKs (with KsCa factor) |
| 7 (Monodomain) | (7.1)–(7.3) | 7.1 = PDE, 7.2 = reaction-diffusion, 7.3 = conductivity tensor |
| 8 (Spatial Disc) | (8.1)–(8.6) | 8.1 = semi-discrete M·dV/dt = -K·V + F, 8.2–8.6 = BC formulations |
| 9 (Splitting) | (9.1) | 9.1 = Rush–Larsen |
| 10 (Explicit) | (10.1)–(10.6) + (10.3b, 10.4b) | 10.1 = FE, 10.2 = CFL, 10.3 = RK2, 10.4 = RK4 |
| 11 (Implicit) | (11.1)–(11.6) | 11.1 = BDF1, 11.2 = BDF1 matrix, 11.3 = CN, 11.4 = CN matrix, 11.5 = BDF2, 11.6 = BDF2 matrix |
| 12 (Bidomain) | (12.1)–(12.6) | 12.1–12.2 = conservation laws, 12.3 = parabolic, 12.4 = elliptic, 12.5 = gating ODE, 12.6 = bidomain block system |
| 13 (Matrices) | (13.1) | 13.1 = 2×2 block system (A11/A12/A21/A22 definitions) |
| 14 (Solving) | (14.1)–(14.3) | 14.1 = elliptic constraint, 14.2 = CFL condition, 14.3 = SBDF2 formula |
| 18 (LBM) | (18.1)–(18.41) + (18.7a) | Phase space, Boltzmann, LBE, quadrature, moment space |
| 19 (LBM Mono) | (19.1)–(19.11) + (19.10b) | 19.1 = diffusion tensor, 19.3 = BGK with source, 19.8 = MRT, 19.11 = 6-stage step |
| 20 (LBM Bi) | (20.1)–(20.4) | 20.1 = pseudo-time, 20.2 = phi_e relaxation, 20.3 = Vm lattice, 20.4 = phi_e lattice |
| A (Appendix) | (A.1)–(A.7) | A.1 = heat equation, A.7 = bidomain PDE classification |
| B (LA) | (B.1)–(B.8) | B.1 = vector def, B.2 = mat-vec, B.3 = eigenvalue def, B.4 = SPD, B.5 = condition number, B.6 = null space, B.7 = block system, B.8 = projection |
| C (Bridge) | (C.1)–(C.12) | C.1 = 3-pt stencil, C.2 = convergence rate, C.3 = Forward Euler, C.4 = CFL, C.5 = 2D Neumann eigenvalues, C.6 = Backward Euler, C.7 = Crank-Nicolson, C.8 = DST eigenvalues, C.9 = 2D eigenvalue sum, C.10 = energy functional, C.11 = CG convergence, C.12 = Chebyshev |

## Engine V5.4 Verification Status

| Chapter | Engine Box | Equations | Status |
|---------|-----------|-----------|--------|
| 1 (HH) | ✅ Verified | — | Method names match IonicModel ABC |
| 5 (TTP06) | ✅ Verified | ✅ Verified | All 12 currents checked against ttp06/currents.py |
| 6 (ORd) | ✅ Verified | ✅ Verified | 15 currents checked against ord/currents.py |
| 7 (Monodomain) | ✅ Verified | — | SpatialDiscretization ABC matches |
| 8 (Spatial Disc) | ✅ Verified | — | FDM/FEM/FVM discretization, BC implementation |
| 9 (Splitting) | ✅ Verified | — | SplittingStrategy, GodunovSplitting, StrangSplitting, RushLarsenSolver verified |
| 10 (Explicit) | ✅ Verified | ✅ Verified | Matrix forms match RK2Solver, RK4Solver bare-k convention |
| 11 (Implicit) | ✅ Verified | ✅ Verified | A_lhs/B_rhs forms match CrankNicolsonSolver, BDF1Solver, BDF2Solver |
| 19 (LBM Mono) | ✅ Verified | ✅ Verified | BGK/MRT collision, streaming, bounce-back, τ-D, 5-stage loop match lbm/ package |

## Known Issues / Future Work

- Ch 4: SVG diagrams should be replaced with literature images
- Ch 6: Chloride currents (IClCa, IClb) described but not implemented in Engine V5.4
- Ch 18: MRT for D3Q7 (7×7 M matrix) described but not yet in Engine V5.4
- Ch 20: Bidomain LBM not yet implemented in Engine V5.4 (architectural research only)
- Cross-references: Verified after major restructuring (session 12), spot-check periodically

## Supporting Files

| File | Purpose |
|------|---------|
| `STYLE_GUIDE.md` | Writing style rules ("Feynman style") + layered complexity principle |
| `CHANGELOG.md` | All major edits with dates |
| `INDEX.md` | This file |
| `LBM_Textbook_Part_IV.pdf` | Standalone Part IV extraction |
| `LBM_Monodomain_D2Q9.pptx` | D2Q9 presentation (slides 18.1–18.6 + summary) |
