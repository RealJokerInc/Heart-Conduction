# Cardiac Computational Modeling — Document Index

**Source**: `bidomain_textbook.html` (~12,262 lines)
**Output**: `Cardiac_Computational_Modeling.pdf`
**Standalone LBM**: `LBM_Textbook_Part_IV.pdf`
**Last updated**: 2026-03-07 (session 12, major restructuring: split Ch 7, merge Ch 11+12, split solvers, add BC sections)

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

## Part III — Bidomain Modeling (Ch 12–17)

| Ch | Title | Sections | Style Status |
|----|-------|----------|-------------|
| 12 | The Bidomain Model | 12.1–12.9 | **Merged** from old Ch 11+12. 12.1–12.3 (physical picture), 12.4–12.8 (equations). **12.9 NEW**: Bidomain BC intro (three canonical scenarios, null-space insight) |
| 13 | Spatial Discretization (Bidomain) | 13.1–13.4 | Good — block matrix diagram. **13.4 NEW**: BC formulation detail (subsections 13.4.1–13.4.4) |
| 14 | Time Integration and Operator Splitting | 14.1–14.5 | Good — splitting error source clarification + monodomain vs bidomain splitting insight box |
| 15 | Parabolic Solvers: The A₁₁ Block | 15.1–15.3 | **NEW chapter** from old 15.3–15.5. PCG (15.1), Chebyshev (15.2), Spectral (15.3 with subsections 15.3.1–15.3.4) |
| 16 | Elliptic Solvers: The Schur Complement | 16.1–16.5 | **NEW chapter** from old 15.1–15.2, 15.6–15.8. Saddle-point (16.1), Block precon (16.2), AMG (16.3), Krylov (16.4), Null-space (16.5) |
| 17 | Implementation Roadmap for Engine V5.4 | 17.1–17.3 | Good — cross-ref fixed |

## Part IV — Lattice-Boltzmann Methods (Ch 18–20)

| Ch | Title | Sections | Style Status |
|----|-------|----------|-------------|
| 18 | The Lattice-Boltzmann Method: From Kinetic Theory to Computation | 18.1–18.5 | **Rewritten v4.0** — "Quadrature First" quadrature, eqs 18.1–18.41 + 18.7a |
| 19 | Lattice-Boltzmann Methods for Monodomain | 19.1–19.8 | **Rewritten v2.1** — Ω^NR/Ω^R notation, expanded BC section (19.5.1–19.5.5); eqs (19.1)–(19.11) + (19.10b) |
| 20 | Lattice-Boltzmann Methods for Bidomain | 20.1–20.5 | Dual-lattice, pseudo-time, hybrid approaches; eqs (20.1)–(20.4) |

## Appendices

| Ch | Title | Sections | Style Status |
|----|-------|----------|-------------|
| Refs | Key References | 3 groups | Fine |
| A | A Visual Guide to Differential Equations | A.1–A.10 | Good — 3B1B style. **A.8–A.10 NEW**: Transform space (A.8), DCT/DST with eigenvalues (A.9), CG/iterative methods with PCG pseudocode (A.10), eqs (A.7)–(A.12) |
| B | Introduction to PyTorch | B.1–B.13 | **Rewritten v2.1** — v2 + B.7 rewritten (cleaner sparse ops catalogue), new B.9 "Advanced Operations for Scientific Computing" (torch.where, roll, einsum, scatter_add_, in-place workspaces, NumPy interop, meshgrid, FFT); old B.9–B.12 renumbered → B.10–B.13 (1,165 lines) |

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
| 13 (Bidomain FEM) | (13.1)–(13.5) | 13.1 = semi-discrete block, 13.2 = 2×2 block system, 13.3–13.5 = BC formulations |
| 15 (Parabolic) | (15.1) | 15.1 = spectral denominator (CN/BDF1/BDF2) |
| 18 (LBM) | (18.1)–(18.41) + (18.7a) | Phase space, Boltzmann, LBE, quadrature, moment space |
| 19 (LBM Mono) | (19.1)–(19.11) + (19.10b) | 19.1 = diffusion tensor, 19.3 = BGK with source, 19.8 = MRT, 19.11 = 6-stage step |
| 20 (LBM Bi) | (20.1)–(20.4) | 20.1 = pseudo-time, 20.2 = phi_e relaxation, 20.3 = Vm lattice, 20.4 = phi_e lattice |
| A (Appendix) | (A.1)–(A.12) | A.1 = heat equation, A.7–A.10 = transforms, A.11–A.12 = CG/PCG |

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
