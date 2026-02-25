# Bidomain Discretization Research Package

## Overview

This directory contains comprehensive research documentation on discretization methods for cardiac bidomain equations, prepared as an implementation guide for extending the monodomain Engine V5.4 to support bidomain modeling.

## Files

### BIDOMAIN_DISCRETIZATION.md
**Main research document** (~45 KB, 1230 lines)

A complete technical reference covering:

#### Sections

1. **Introduction** - Motivation, computational scope
2. **Mathematical Foundation** - Strong form, weak form, relationship to monodomain
3. **FEM Discretization** - Basis functions, mass/stiffness matrices, assembly procedures
4. **FDM Discretization** - Stencil development, anisotropic conductivity, boundary conditions
5. **FVM Discretization** - Cell-centered formulation, harmonic mean, conservation properties
6. **Semi-Discrete System Structure** - ODE form, block matrix structure, DOF organization
7. **Practical Implementation Considerations** - Time integration, solvers, preconditioners, memory analysis
8. **References** - 28 peer-reviewed sources (2006-2025)
9. **Appendix** - Implementation checklist for Engine V5.4

#### Key Content Areas

- **Mathematical equations:** Full LaTeX formatting, both strong and weak forms
- **Conductivity tensors:** Orthotropic representation, physiological values (σ_iL/σ_iT ≈ 10:1, σ_eL/σ_eT ≈ 2.5:1)
- **Boundary conditions:** Insulated tissue, bath coupling, Neumann/Robin/Dirichlet types
- **Discretization methods:** Complete derivations for FEM (P1, P2), FDM (7-point, 27-point), FVM (cell-centered)
- **Block system structure:** Explicit 2×2 matrix form, Schur complement reduction
- **Time stepping:** Fully implicit, semi-implicit, operator splitting (Godunov, Strang)
- **Solvers:** Direct (LU), iterative (CG, BiCGSTAB), preconditioners (AMG, ILU)
- **Computational analysis:** Memory requirements (4-10× monodomain), cost comparison (2-4× per step)
- **Practical guidance:** When to use bidomain vs. monodomain, mesh requirements (25-200 μm), validation procedures

#### Tables and Examples

- Physiological conductivity values
- DOF comparison (monodomain vs. bidomain)
- Memory analysis for realistic mesh sizes
- Computational cost breakdown
- Grid convergence requirements
- Time integration scheme comparison
- Linear solver iteration counts

#### Example Code Pseudocode

- FEM assembly loop structure
- Stiffness matrix computation (1D, 2D examples)
- Stencil coefficient calculation
- Operator splitting algorithm

## Research Methodology

### Sources Reviewed (2020-2026)

- **Latest preprints:** Novel bidomain partitioned strategies (2025)
- **Core references:** Sundnes et al. (2006), Colli Franzone et al. (2014), Plank et al. (2021)
- **Mathematical foundations:** Existence/uniqueness proofs, variational formulations
- **Implementation approaches:** openCARP, FEM/FDM/FVM comparisons
- **Computational optimization:** AMG preconditioners, multigrid methods, operator splitting

### Total References: 28

Includes:
- Foundational monographs (Springer)
- Recent preprints (arXiv 2025)
- Peer-reviewed journals (SIAM, PMC, Frontiers, Annals of Biomedical Engineering)
- Software documentation (openCARP v7.0)

## Key Findings

### Bidomain Necessity

- **Monodomain insufficient for:** Defibrillation, extracellular stimulation, virtual electrode polarization, ECG generation
- **Physiological accuracy:** Bidomain activation times 1-2 ms earlier, conduction 2% faster
- **When acceptable:** Activation sequence/timing (3-5% error tolerance), no external fields

### Discretization Comparison

| Method | Geometry | Accuracy | Anisotropy | Cost |
|--------|----------|----------|-----------|------|
| **FEM** | Complex | O(h²) | ✓ handles easily | Higher |
| **FDM** | Regular | O(h²) | 27-pt stencil needed | Lower |
| **FVM** | Unstructured | O(h) | ✓ natural | Medium |

### Critical Implementation Issues

1. **Block system structure:** 2N DOF (V_m + φ_e at each node)
2. **Differential-algebraic (DAE):** Singular mass matrix for φ_e equation
3. **Coupling terms:** K_i appears in both equations
4. **Solver demand:** AMG preconditioner essential (5-10× speedup)
5. **Memory**: 4× monodomain baseline (200-400 MB realistic)
6. **Computation:** 2-4× slower per timestep (but fewer total timesteps possible)

### Recommended Implementation Path

**Phase 1 (Priority):** FEM discretization
- Highest value for complex geometries
- Leverage existing FEM assembly code
- Block matrix structure manageable

**Phase 2:** Time integration and solvers
- Semi-implicit schemes
- AMG preconditioner integration

**Phase 3:** FDM (optional, for structured grids)
- Useful for simplified models
- Faster assembly, lower memory

**Phase 4:** FVM (optional, for conservation focus)
- Better for ischemia/damage modeling
- Requires face-based assembly

## Usage Notes

- Document is self-contained with full mathematical derivations
- Suitable as reference for developers and researchers
- Implementation checklist (Appendix) provides step-by-step guidance
- All equations in LaTeX format compatible with markdown and scientific typesetting

## Next Steps

1. **Review document** for mathematical correctness and completeness
2. **Identify solver infrastructure** in Engine V5.4 (matrix format, preconditioners)
3. **Prototype FEM assembly** on small test problems (1D cable, 2D sheet)
4. **Validate against published benchmarks** (monodomain/bidomain comparison papers)
5. **Optimize memory layout** based on actual matrix sparsity patterns
6. **Benchmark against openCARP** for performance validation

## Contact/Questions

For implementation questions:
- Refer to specific sections and equations in BIDOMAIN_DISCRETIZATION.md
- Cross-reference with cited literature (full URLs provided)
- Validation procedures detailed in "Practical Implementation Considerations" section

---

**Document version:** 1.0
**Last updated:** February 2025
**Status:** Complete research compilation ready for implementation
