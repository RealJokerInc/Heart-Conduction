---
paper: xie_2023_mgcnn
title: "MGCNN: a learnable multigrid solver for sparse linear systems from PDEs on structured grids"
authors: "Xie Y, Lv M, Zhang C"
year: 2023
journal: "arXiv preprint (revised 2024-05)"
doi: "arxiv:2312.11093"
pmid: ""
pdf: ../papers/mgcnn_2023.pdf
questions: [surrogate_pipeline]
---

## Key Findings
- **Learnable multigrid solver for sparse linear systems from PDEs on structured grids** — directly matches the structural class of our bidomain elliptic problem (structured-grid, sparse stencil from discretized elliptic PDE).
- **3-8× speedup over classical Geometric Multigrid (GMG)**. Concrete, published number we can target. GMG is what Bidomain V1 Tier 3 currently uses — direct drop-in comparison.
- **Generalizes across RHS, PDE coefficients, and grid dimensions** with offline training only. One training covers a range of problems — matches our need to generalize across tissue configurations without retraining.
- **Tested on grids from 31×31 to 4095×4095** — extreme resolution range. Our Bidomain V1 grids are typically 128-512 on each side, squarely in the validated regime.
- **Convection-diffusion with heterogeneous diffusion coefficients** — the closest non-cardiac analog to bidomain's spatially-varying D(x). Generalizes to unseen coefficient distributions, which is what we'd need for variable fiber orientations.
- **Three architectural pillars**: multilevel hierarchy (V-cycle structure), linearity preservation (operator respects linearity of the underlying PDE), weight sharing (parameter efficiency, regularization).

## Method
- **Architecture**: CNN with multigrid-inspired hierarchy. Multilevel structure: coarsening via downsampling, solution via CNN at each level, prolongation via upsampling — conceptually similar to UGrid's V-cycle but the abstract doesn't specify which exact multigrid components (smoother, prolongation, restriction, coarse-grid operator) are learned vs fixed. Need PDF read to determine this precisely.
- **Linearity preservation**: the network structure respects the linearity of the target PDE — i.e., the learned operator acts linearly on the RHS `f`. Important because elliptic PDEs are linear; nonlinear networks would violate this structural property.
- **Weight sharing**: across levels of the hierarchy and across grid sizes. Enables generalization to unseen resolutions.
- **Training**: purely offline (supervised or residual-based, abstract doesn't specify). Generalizes to unseen coefficients after training on a coefficient distribution.
- **Evaluation**: 3-8× wall-clock speedup vs GMG on convection-diffusion with heterogeneous coefficients.
- **BC handling**: not specified in abstract. Must read PDF to verify.

## Connections to Our Models

### Relevant Engine Components
**Direct alternative to UGrid for our Bidomain V1 Tier 3 elliptic solver.** Both MGCNN and UGrid are learnable multigrid solvers for structured-grid linear PDEs. They differ in:
- UGrid: U-Net-structured V-cycle with masked iterator, residual-loss training, Dirichlet focus.
- MGCNN: multilevel hierarchy with explicit linearity preservation, offline training, convection-diffusion focus.
Natural A/B comparison on our bidomain elliptic problem.

### Agreements
- **Multigrid structure is the right inductive bias for elliptic on structured grids.** Same conclusion as UGrid, Greenfeld, and the broader neural-multigrid literature.
- **Heterogeneous coefficient handling** — our D_i + D_e varies with fiber orientation. MGCNN specifically tests heterogeneous diffusion. Probably more directly validated for our case than UGrid's (mostly isotropic) benchmarks.
- **Linearity preservation** — elegantly matches the linear-elliptic structure of the bidomain φ_e equation. UGrid mentions linearity preservation too; both papers converge on this.
- **3-8× speedup is a reasonable, credible target.** Not the wildest claim in the neural-PDE space (some papers claim 100×+) — more believable.
- **Offline training** — no online adaptation needed; useful for deployment.

### Disagreements or Gaps
- **BC handling not specified in abstract.** Critical unknown for our Neumann-dominant bidomain case. Must read the PDF to verify.
- **Abstract doesn't specify which multigrid components are learned** (smoother, prolongation, restriction, coarse-grid operator). Without this, architectural comparison to UGrid / Greenfeld is fuzzy.
- **No cardiac demonstration** — tested on convection-diffusion, a well-understood numerical benchmark. Transfer to bidomain is plausible but not proven.
- **No code repository mentioned** in abstract — adoption friction higher than UGrid or CNO which have open-source code.
- **Anisotropic tensor handling** — paper tests heterogeneous *scalar* diffusion coefficients, not tensor-anisotropic. Bidomain's D(x) is generally anisotropic SPD tensor. Extension plausible but needs verification.
- **Single-field, not coupled.** Like UGrid, MGCNN solves a single Poisson-like system, not coupled V_m + φ_e.

### Actionable Insights
- **HIGH — Benchmark against UGrid as the primary elliptic-solver candidate.** Both papers target the same problem class. Whichever gives better speedup + accuracy on our bidomain test case wins.
- **MEDIUM — Read PDF for BC specifics.** If MGCNN natively handles Neumann BCs (unlikely from abstract), it may be preferable to UGrid which needs BC adaptation.
- **MEDIUM — Extract the "linearity preservation" design pattern.** Explicit linearity constraint on the learned operator is likely valuable for bidomain elliptic (which is linear) — ensures the net respects RHS linearity without specialized training.
- **LOW — Grid-size generalization claim**: 31×31 to 4095×4095 is extreme. For Bidomain V1 this is overkill, but useful if we ever scale to larger tissue domains.
- **LOW — No code** — if we adopt, we'd have to reimplement from the paper. Friction higher than UGrid/CNO.

## Limitations / Caveats
- **BC specifics unknown from abstract** — architectural transferability to Neumann-dominant bidomain untested.
- **No code released** in the abstract's linked resources — may exist in a later arXiv version or in the authors' own repos, but not advertised.
- **Scalar coefficients only** — anisotropic tensor D extension is plausible but not demonstrated.
- **Convection-diffusion benchmark, not elliptic-pure** — results may tilt toward parabolic-advective mixes rather than the pure elliptic case we'd need.
- **Abstract underspecifies the architecture** — "multilevel hierarchy, linearity preservation, weight sharing" is thematic rather than concrete. Need PDF for implementation specifics.
- **Single-field** — coupled-field extension untested.
