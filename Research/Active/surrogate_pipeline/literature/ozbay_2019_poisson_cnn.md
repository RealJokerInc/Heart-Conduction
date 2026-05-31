---
paper: ozbay_2019_poisson_cnn
title: "Poisson CNN: Convolutional neural networks for the solution of the Poisson equation on a Cartesian mesh"
authors: "Ozbay AG, Hamzehloo A, Laizet S, Tzirakis P, Rizos G, Schuller B"
year: 2021
journal: "Data-Centric Engineering 2 e6 (2021)"
doi: "arxiv:1910.08613"
pmid: ""
pdf: ../papers/ozbay_2019_poisson_cnn.pdf
questions: [surrogate_pipeline]
---

## Key Findings
- **Vanilla fully-convolutional CNN** (not multigrid, not V-cycle) solves the 2D Poisson equation on Cartesian grids. Simplest possible architecture class in the learned-Poisson family.
- **Handles arbitrary BCs** via a decomposition trick: split the original Poisson problem into one homogeneous-BC Poisson problem + four inhomogeneous-BC Laplace sub-problems, each handled by its own sub-network, then recombined by linearity. Not a black-box boundary handler.
- **Mean percentage error below 10%** on analytical Poisson test cases.
- **Works as a multigrid preconditioner**: a single CNN forward pass reduces RMS error by **>90%** compared to a zero initial guess when used to seed an iterative solver. Speedup compounds when used as the initial guess for classical multigrid.
- **Resolution generalization**: "encouraging" capacity to produce correct solutions on denser grids than those seen in training, though explicit accuracy drop not quantified.

## Method
- **Architecture**: fully convolutional encoder-decoder (not U-Net, not multigrid V-cycle). 2D Cartesian grid in, 2D solution field out. Separate sub-networks for the homogeneous-Poisson and the four boundary Laplace problems; outputs summed by superposition.
- **BC handling**: **via superposition, not via masking.** The trick exploits linearity: `u = u_homog(f, u|∂Ω=0) + u_BC_east + u_BC_west + u_BC_north + u_BC_south`. Each BC sub-problem has zero RHS and a non-zero boundary patch on one side only. Avoids the masked-iterator complexity of UGrid but requires 5 forward passes per solve.
- **Training**: supervised on ground-truth solutions from classical Poisson solvers on randomly-generated (f, BC) samples.
- **Loss**: direct MSE on solution fields. Not residual-based.
- **Grid**: 2D Cartesian, multiple resolutions in training.

## Connections to Our Models

### Relevant Engine Components
Simplest prototype architecture for the **Bidomain V1 elliptic step**. Good first thing to build before jumping to UGrid's V-cycle complexity — establishes a baseline and lets us understand where a vanilla CNN fails.

### Agreements
- **Structured 2D Cartesian grid**: matches Bidomain V1's `StructuredGrid` exactly.
- **Linear PDE, superposition valid**: our elliptic problem is linear in φ_e, so the five-subproblem decomposition is applicable in principle.
- **CNN + preconditioning pattern**: their finding that a CNN forward pass gives a 90% RMS error reduction when used as an initial guess for multigrid is a *direct template* for our preconditioner-first v1 (per NPO and the 2026-04-21 gap-scan recommendation).
- **Fully-convolutional**: parameter-efficient, translation-equivariant. Good inductive bias for periodic-in-coefficient fiber fields.

### Disagreements or Gaps
- **Isotropic Laplacian only**: no tensor D. Same gap as UGrid — adaptation required for our anisotropic `∇·((D_i+D_e)∇φ_e)`.
- **Dirichlet BCs implicitly assumed** for the BC sub-problems. Neumann not demonstrated. The decomposition trick in principle extends to Neumann by replacing the boundary-value sub-problems with boundary-flux sub-problems, but this requires rewriting the architecture.
- **Supervised training with ground-truth solutions**: requires running a classical Poisson solver to generate training data. Inferior to UGrid/NPO's residual-loss unsupervised training. For us this means extra data-generation cost: we'd need thousands of Bidomain V1 elliptic solves as labels.
- **Five sub-networks per solve**: 5× inference cost vs a single end-to-end model. For a preconditioner this is fine (still cheaper than PCG); for a standalone solver it's a real penalty.
- **Not multigrid-structured**: no formal guarantee on convergence or scaling with problem size. UGrid's V-cycle proofs don't apply here.
- **Solution-space loss (MSE), not residual loss**: the model can look accurate by MSE but still have high residual. Residual-based losses (UGrid, NPO) are more robust for preconditioner use.

### Actionable Insights
- **MEDIUM — Start here for the prototype baseline.** If we want to get a working CNN elliptic surrogate on a structured-grid test problem in days rather than weeks, this is the right architecture class. Simpler than UGrid and easier to debug.
- **MEDIUM — CNN-as-initial-guess pattern**: their 90% RMS-reduction claim suggests a hybrid strategy: CNN produces initial guess → classical PCG/multigrid polishes to tolerance. Three-line integration into Bidomain V1's Tier 2/3 solvers. Low risk, immediate speedup.
- **LOW — Superposition for BCs**: elegant but probably not worth 5× inference cost once we get to production. UGrid's masked iterator is cleaner long-term.
- **LOW — Training-data cost**: if we go supervised (this paper's approach), budget for substantial Bidomain V1 runs to generate labels. Prefer unsupervised residual loss (Greenfeld/UGrid) instead.

## Limitations / Caveats
- **No convergence guarantees**: single-shot CNN inference with no iterative structure. No fallback if the network is wrong for a specific problem.
- **10% error bound** is too loose for the elliptic step inside a bidomain rollout: errors feed back into V_m every step, compound across 30K time steps. We'd need to stack PCG iterations on top.
- **Ground-truth training data requirement** is a large practical cost for our setting.
- **Not architecturally multigrid**: scaling to large grids (>512²) likely requires progressively more parameters or deeper networks; no built-in hierarchy.
- **Five-subproblem decomposition is 2D-specific**: 3D has 6 face sub-problems + more, making the decomposition less clean. For our current 2D Bidomain V1 this is fine; future 3D not so much.
- **Published version (2021 Data-Centric Engineering) may differ from the arXiv v1 (2019)** — architecture or numerical results may have been refined.
