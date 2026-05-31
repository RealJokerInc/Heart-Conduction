---
paper: greenfeld_2019_multigrid
title: "Learning to Optimize Multigrid PDE Solvers"
authors: "Greenfeld D, Galun M, Kimmel R, Yavneh I, Basri R"
year: 2019
journal: "ICML 2019 (PMLR 97)"
doi: "arxiv:1902.10248"
pmid: ""
pdf: ../papers/greenfeld_2019_multigrid.pdf
questions: [surrogate_pipeline]
---

## Key Findings
- **First paper to learn the multigrid prolongation operator with a neural network.** Single network generalizes across a *family* of parameterized PDEs — one training, many deployments.
- Trained **unsupervised** (no ground-truth solutions needed) — the loss is the two-grid spectral radius, which directly corresponds to multigrid convergence rate.
- Beats **Black Box Multigrid (BBMG)** convergence rates on 2D diffusion problems. BBMG is the strongest classical baseline for non-uniform coefficient Poisson, so this matters.
- Establishes the **pattern later extended by UGrid (Li 2024) and NPO (Cai 2025)**: treat classical solver components as trainable subroutines while preserving the solver's structural convergence guarantees.

## Method
- **What is learned**: only the **prolongation matrix P** (the restriction is implicitly the transpose P^T). Smoother is a fixed Jacobi-style relaxation with a learned weight. This keeps the method within the classical algebraic multigrid framework — the NN replaces a single heuristic (Ruge-Stüben or BBMG prolongation construction) with a learned one.
- **Architecture**: small MLP/CNN that takes a local stencil patch of the operator A around a fine-grid point and outputs the row of P connecting that point to its coarse-grid neighbors. Operates locally — no global attention.
- **Loss**: unsupervised; minimizes the spectral radius ρ of the two-grid operator E₂ = (I − P(P^TAP)^{−1}P^TA)(I − ωD^{−1}A). This is *the* classical convergence metric.
- **Training regime**: PDEs drawn from a parametric family (varying coefficients in 2D diffusion `−∇·(a(x)∇u) = f`). One network, many operators.
- **PDE classes tested**: 2D diffusion with varying coefficients a(x) on uniform grids.

## Connections to Our Models

### Relevant Engine Components
Foundational to our **Bidomain V1 Tier 3 (GMG elliptic solver)**. Greenfeld's learned-prolongation pattern is what UGrid generalizes to a full end-to-end V-cycle. Reading this paper is the *prerequisite* for understanding why UGrid's architecture works the way it does.

### Agreements
- **Unsupervised training via residual/spectral loss**: same principle UGrid uses (residual loss) and NPO uses (condition-number loss). We'll adopt this rather than requiring ground-truth φ_e from Bidomain V1.
- **Learn a solver component, not the full inverse**: exactly the "safer" design philosophy we want for the hybrid bidomain pivot (preserve convergence guarantees).
- **Parametric PDE family**: matches our need to generalize across tissue configurations (fiber orientations, D_i/D_e ratios).

### Disagreements or Gaps
- **Only prolongation is learned** — the rest of the multigrid hierarchy (smoother, coarse-grid operator, V-cycle schedule) is fixed classical. UGrid learns the *entire* V-cycle end-to-end. For our purposes the UGrid formulation is probably what we want, but Greenfeld's "learn only one piece" approach is the safer minimum-viable variant if UGrid overfits or fails.
- **2D diffusion only**: paper tests scalar elliptic. Our bidomain elliptic is tensor-anisotropic. Nothing in principle prevents extending the method; just untested.
- **Local stencil input**: the NN sees only a small patch of A. This limits receptive field. UGrid's full-V-cycle CNN has a much larger effective receptive field.
- **Old** (2019, ICML): five years of follow-up work; don't copy the implementation, cite as foundational.

### Actionable Insights
- **HIGH — Read BEFORE UGrid.** The paper is short (~8 pages) and establishes the key insight (learn a multigrid subroutine, train on the spectral/residual loss) that UGrid builds on. Skipping Greenfeld makes UGrid harder to parse.
- **MEDIUM — Fallback design**: if UGrid's full learned V-cycle proves unstable, fall back to Greenfeld-style "learn only the prolongation" as a safer intermediate. Keep everything else classical.
- **MEDIUM — Unsupervised loss pattern**: confirms we don't need ground-truth φ_e to train. Operator-residual is enough. This removes one entire class of data-generation pain.
- **LOW — Code**: unofficial reimplementations exist on GitHub; the original paper had no public code.

## Limitations / Caveats
- **Prolongation is only one of many multigrid design choices.** Smoother, coarsening strategy, and V-cycle depth are all fixed. A learned prolongation in a bad V-cycle gives bad results.
- **Spectral-radius loss requires computing eigenvalues / solving small eigenproblems during training** — expensive per step, but pays off in unsupervised regime.
- **No boundary-condition generalization demonstrated** — Dirichlet BCs implicit throughout. Same gap as UGrid for our Neumann-dominant bidomain case.
- **Prolongation alone doesn't help if the coarse-grid operator is ill-conditioned**; Galerkin coarsening (P^TAP) is used but not analyzed for pathological cases relevant to bidomain (singular Neumann null space).
