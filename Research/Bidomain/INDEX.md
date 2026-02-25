# Cardiac Computational Modeling — Document Index

**Source**: `bidomain_textbook.html` (~8455 lines)
**Output**: `Cardiac_Computational_Modeling.pdf` (2289 MathJax containers, 5.3 MB)
**Standalone LBM**: `LBM_Textbook_Part_IV.pdf` (540 MathJax containers, 1.4 MB)
**Last updated**: 2026-02-19 (session 6, Ch 17 v3.1 — Gaussian quadrature rewrite)

## Part I — Single Cell Dynamics

| Ch | Title | Sections | Style Status |
|----|-------|----------|-------------|
| 1 | The Hodgkin–Huxley Revolution | 1.1–1.4 | Good — Feynman style, analogies, figures; engine box verified |
| 2 | From Neurons to Heart Cells | 2.1–2.3 | Good — camera flash/light switch analogy |
| 3 | Anatomy of the Cardiac Action Potential | 3.1–3.5 | Good — phase-by-phase narrative |
| 4 | Intracellular Calcium: The Master Regulator | 4.1–4.8 | Good — SVGs pending literature image replacement |
| 5 | The ten Tusscher–Panfilov 2006 Model | 5.1–5.8 | **Verified** — equations checked against Engine V5.4 code |
| 6 | The O'Hara–Rudy 2011 Model | 6.1–6.10 | **Verified** — equations checked against Engine V5.4 code |

## Part II — Tissue-Level Monodomain Modeling

| Ch | Title | Sections | Style Status |
|----|-------|----------|-------------|
| 7 | The Monodomain Equation | 7.1–7.9 | Good — ELI5 + worked examples + derivation; engine box verified |
| 8 | Operator Splitting: Divide and Conquer | 8.1–8.4 | Good — engine box rewritten to match actual splitting code |
| 9 | Explicit Diffusion Solvers | 9.1–9.7 | **Verified** — RK2/RK4 with discrete matrix forms matching code |
| 10 | Implicit Diffusion Solvers | 10.1–10.6 | **Verified** — BDF1/CN/BDF2 with boxed A_lhs/B_rhs matrix forms matching code |

## Part III — Bidomain Modeling

| Ch | Title | Sections | Style Status |
|----|-------|----------|-------------|
| 11 | Why Bidomain? — The Physical Picture | 11.1–11.3 | Good — balloon analogy, figure fixed |
| 12 | The Bidomain Equations | 12.1–12.5 | Good — step-by-step derivation |
| 13 | Spatial Discretization (Bidomain) | 13.1–13.3 | Good — block matrix diagram |
| 14 | Time Integration and Operator Splitting | 14.1–14.5 | Good — CSS and tag numbers fixed |
| 15 | Linear Solvers for the Block System | 15.1–15.5 | Good — practical recommendations |
| 16 | Implementation Roadmap for Engine V5.4 | 16.1–16.3 | Good — cross-ref fixed |

## Part IV — Lattice-Boltzmann Methods

| Ch | Title | Sections | Style Status |
|----|-------|----------|-------------|
| 17 | The Lattice-Boltzmann Method: From Kinetic Theory to Computation | 17.1–17.3 | **Rewritten v3.1** — Boltzmann eq, LBM-SRT/BGK, Gaussian quadrature framework for weights/velocities, 26 equations, 1 SVG, 9 insight boxes |
| 18 | Lattice-Boltzmann Methods for Monodomain | 18.1–18.7 | **Refactored** — redundancy removed, cross-refs Ch 17, focused on cardiac coupling + 5-stage LBM-EP algorithm |
| 19 | Lattice-Boltzmann Methods for Bidomain | 19.1–19.5 | **New** — Dual-lattice, pseudo-time, hybrid approaches |

## Appendices

| Ch | Title | Sections | Style Status |
|----|-------|----------|-------------|
| Refs | Key References | 3 groups | Fine |
| A | A Visual Guide to Differential Equations | A.1–A.7 | Good — 3B1B style |

## Equation Number Registry

| Chapter | Range | Key Equations |
|---------|-------|---------------|
| 1 (HH) | (1.1)–(1.3) | 1.1 = membrane current, 1.2 = gate ODE, 1.3 = relaxation form |
| 2 (Minimal) | (2.1)–(2.2) | 2.1 = FitzHugh-Nagumo, 2.2 = Fenton-Karma |
| 5 (TTP06) | (5.1)–(5.12) | 5.1 = I_ion total, 5.3 = ICaL GHK (with V-15 shift) |
| 6 (ORd) | (6.1)–(6.20) | 6.1 = I_ion total, 6.8 = IKs (with KsCa factor) |
| 7 (Monodomain) | (7.1)–(7.5) | 7.1 = PDE, 7.5 = semi-discrete system M·dV/dt = -K·V + F |
| 8 (Splitting) | (8.1) | 8.1 = Rush–Larsen |
| 9 (Explicit) | (9.1)–(9.6) | 9.1 = FE, 9.2 = CFL, 9.3 = RK2, 9.3b = RK2 matrix, 9.4 = RK4, 9.4b = RK4 matrix |
| 10 (Implicit) | (10.1)–(10.6) | 10.1 = BDF1, 10.2 = BDF1 matrix, 10.3 = CN, 10.4 = CN matrix, 10.5 = BDF2, 10.6 = BDF2 matrix |
| 11 (Why Bidomain) | (11.1)–(11.2) | Conservation laws |
| 12 (Bidomain Eqs) | (12.1)–(12.4) | 12.2 = parabolic, 12.3 = elliptic, 12.4 = gating ODE |
| 13 (Bidomain FEM) | (13.1)–(13.2) | 13.1 = semi-discrete block, 13.2 = 2×2 block system |
| 14 (Time Integ) | (14.1)–(14.2) | 14.1 = elliptic constraint, 14.2 = bidomain CFL |
| 17 (LBM) | (17.1)–(17.25) | 17.1 = Boltzmann eq, 17.4 = BGK approx, 17.7 = LBM-BGK unified, 17.8 = collision, 17.9 = streaming, 17.12 = Maxwell-Boltzmann, 17.12a = Gauss-Hermite quadrature, 17.13–17.15 = exactness conditions, 17.23 = τ-D, 17.24 = diffusion f_eq, 17.25 = NS f_eq |
| 18 (LBM Mono) | (18.1)–(18.4) | 18.1 = source term S, 18.2 = BGK with source, 18.3 = MRT with source (boxed), 18.4 = LBM-EP 5-stage step (boxed) |
| 19 (LBM Bi) | (19.1)–(19.4) | 19.1 = pseudo-time eq, 19.2 = phi_e relaxation, 19.3 = Vm lattice, 19.4 = phi_e lattice |
| A (Appendix) | (A.1) | Heat equation |

## Engine V5.4 Verification Status

| Chapter | Engine Box | Equations | Status |
|---------|-----------|-----------|--------|
| 1 (HH) | ✅ Verified | — | Method names match IonicModel ABC |
| 5 (TTP06) | ✅ Verified | ✅ Verified | All 12 currents checked against ttp06/currents.py |
| 6 (ORd) | ✅ Verified | ✅ Verified | 15 currents checked against ord/currents.py |
| 7 (Monodomain) | ✅ Verified | — | SpatialDiscretization ABC matches |
| 8 (Splitting) | ✅ Verified | — | SplittingStrategy, GodunovSplitting, StrangSplitting, RushLarsenSolver verified |
| 9 (Explicit) | ✅ Verified | ✅ Verified | Matrix forms match RK2Solver, RK4Solver bare-k convention |
| 10 (Implicit) | ✅ Verified | ✅ Verified | A_lhs/B_rhs forms match CrankNicolsonSolver, BDF1Solver, BDF2Solver |
| 18 (LBM Mono) | ✅ Verified | ✅ Verified | BGK/MRT collision, streaming, bounce-back, τ-D, 5-stage loop match lbm/ package |

## Known Issues / Future Work

- Ch 4: SVG diagrams should be replaced with literature images
- Ch 6: Chloride currents (IClCa, IClb) described but not implemented in Engine V5.4
- Ch 18: MRT for D3Q7 (7×7 M matrix) described in Ch 17 but not yet in Engine V5.4
- Ch 19: Bidomain LBM not yet implemented in Engine V5.4 (architectural research only)
- Cross-references: Some inline references may still use old chapter numbers in prose

## Supporting Files

| File | Purpose |
|------|---------|
| `STYLE_GUIDE.md` | Writing style rules ("Feynman style") + layered complexity principle |
| `CHANGELOG.md` | All major edits with dates |
| `INDEX.md` | This file |
| `LBM_Textbook_Part_IV.pdf` | Standalone Part IV extraction |
