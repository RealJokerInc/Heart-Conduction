---
paper: lan_2023_neural_preconditioner_mixed_bc
title: "A Neural-preconditioned Poisson Solver for Mixed Dirichlet and Neumann Boundary Conditions"
authors: "Lan KW, Gueidon E, Kaneda A, Panetta J, Teran J"
year: 2023
journal: "arXiv (revised 2024-06-14); targeting numerical analysis journal"
doi: "arxiv:2310.00177"
pmid: ""
pdf: ../papers/lan_2023_neural_preconditioner_mixed_bc.pdf
questions: [surrogate_pipeline]
---

## Key Findings
- **Directly targets the BC gap that kills FNO for bidomain.** This paper is the single closest prior work to what our hybrid bidomain elliptic surrogate needs: a neural preconditioner for the discrete Laplacian with **mixed Dirichlet + Neumann BCs** that change between solves.
- **Preconditioner inside a Krylov solver**, not a full inverse — preserves PCG convergence guarantees. Graceful degradation under distribution shift (worst case: more iterations, still correct answer).
- **Beats both algebraic multigrid (AMG) and prior neural preconditioners** on the tested incompressible-fluid Poisson problems. AMG is the strongest classical baseline; beating it is the bar to clear.
- **Spatially varying convolution kernels** — the key architectural novelty. Standard CNNs have translation-invariant kernels, which can't encode BC-dependent behavior (a Neumann edge looks different from a Dirichlet edge). Spatial-varying kernels let the network specialize per-location.
- **Lightweight network** — explicitly designed to be cheap per iteration so that the preconditioned PCG wall-time actually wins. Expensive-but-accurate preconditioners are a common failure mode; they cost more than the iterations they save.

## Method
- **Architecture**: CNN with **spatially varying convolution kernels**. At each grid point, the conv kernel is computed from local geometry/BC indicator maps rather than shared globally. Concretely, the kernel at position `(i, j)` depends on whether that point is near a Dirichlet boundary, a Neumann boundary, or interior.
- **BC handling**: the BC type is encoded as an input indicator field fed alongside the RHS. The network sees where the boundary is and what kind of BC applies, and the spatially-varying kernels adapt. **This is the mechanism FNO structurally cannot provide** — Fourier-space convolutions are globally periodic and cannot represent location-dependent stencils.
- **Training**: supervised on classical Poisson solves under varied BC configurations (different Dirichlet regions, different Neumann normals). Learns the inverse-Laplacian approximation.
- **Deployment**: wraps around standard PCG. `M⁻¹` = neural preconditioner; `A` = classical Laplacian. PCG convergence theorem applies: the method converges to the true solution regardless of preconditioner quality. Network quality only affects how fast.
- **PDE class**: Poisson `∇²u = f` with mixed BCs. Tested on incompressible fluid simulations (CFD pressure projection — same 80%-wall-time bottleneck pattern as bidomain elliptic).

## Connections to Our Models

### Relevant Engine Components
**This is the highest-priority adoption target for Bidomain V1's Tier 3 (elliptic solver).** Bidomain's elliptic step `∇·((D_i+D_e)∇φ_e) = −∇·(D_i ∇V_m)` has exactly the ingredient this paper solves for: a discrete Poisson-like system with per-simulation-varying BCs (Neumann at tissue-bath, Dirichlet at grounding points, sometimes mixed at interfaces).

### Agreements
- **Preconditioner, not inverse.** The 2026-04-21 gap scan recommendation was explicitly "preconditioner-first" for safety under rollout compounding. This paper validates the choice.
- **Structured Cartesian grid.** Matches Bidomain V1's `StructuredGrid` exactly.
- **BCs change across solves** — our tissue configurations differ per run (infarct masks, stimulus electrode locations). A preconditioner that generalizes across BCs is exactly what we need, not a retrain-per-config solver.
- **CFD pressure-projection analog**: the 80%-wall-time bottleneck in incompressible CFD that this paper attacks is structurally identical to our 94%-wall-time elliptic bottleneck in bidomain. Lessons transfer.

### Disagreements or Gaps
- **Isotropic Laplacian only.** Paper handles `∇²u`, not the tensor-anisotropic `∇·(D(x)∇u)` with fiber-dependent D. Bidomain's D_i + D_e is generally a 2×2 SPD tensor varying spatially. **Adaptation required**: the spatially-varying kernel architecture should extend to encoding D(x) as an input channel, but this is not demonstrated in the paper.
- **No explicit coupled-field story.** Their preconditioner solves a single scalar Poisson; our bidomain has V_m feeding φ_e feeding V_m. Not a deal-breaker — preconditioner can still be applied to each elliptic solve inside the Gauss-Seidel outer loop — but the cross-field information isn't exploited.
- **Robin / Cauchy BCs not tested.** Paper covers Dirichlet + Neumann. Some tissue-bath interface formulations use Robin-like coupling (impedance matching). Unknown whether their kernel-varying structure extends.
- **No long-horizon autoregressive rollout analysis.** The paper is single-shot-preconditioning. Our bidomain use is inside a 30K-step rollout where each elliptic solve feeds the next parabolic step. Preconditioner errors aren't supposed to compound (PCG converges to tolerance regardless), but this should be validated for our use.

### Actionable Insights
- **HIGH — Prototype as Bidomain V1 Tier 2 preconditioner**. Replace or augment our current PCG+Spectral path with a `PCG + Lan2023NeuralPrecond` variant. Minimal risk (PCG convergence guaranteed), immediate upside (fewer iterations to tolerance).
- **HIGH — Adopt the spatially-varying-kernel architecture**. This is the key structural pattern for BC-aware CNN Poisson solvers. Our elliptic operator varies with (a) tissue conductivity D(x), (b) infarct mask, (c) stimulus geometry. Spatially-varying kernels can encode all three.
- **HIGH — Use BC indicator maps as input channels**. Simple, effective, directly generalizes to varied tissue geometries without retraining per case.
- **MEDIUM — Extend to anisotropic D** — the paper's single omission that matters most for us. Prototype with isotropic first (easy win), then condition kernels on D tensor components.
- **MEDIUM — Validate rollout stability**. Even with PCG's convergence guarantee, benchmark whether the preconditioned solver's tolerance-level errors are benign in the 30K-step rollout context. Run a 1000-step rollout test comparing `PCG-classical` vs `PCG-LanNeural` and measure drift in V_m, CV, APD.
- **LOW — Code availability**: check whether the authors released training code. If yes, it's a starting point.

## Limitations / Caveats
- **Isotropic Laplacian only** — explicit extension to tensor-anisotropic Poisson is open work. Adaptation is plausible but untested.
- **Demonstrated on CFD, not cardiac.** The problem class (discrete Poisson with mixed BCs) is the same, but cardiac-specific artifacts (sharp wavefronts driving the RHS, fiber-aligned anisotropy) may expose failure modes not present in CFD benchmarks.
- **Training data requires classical solver.** Like most supervised neural PDE work. Residual-loss training (à la UGrid) is arguably preferable. Worth testing whether unsupervised training extends here.
- **2D demonstration only in abstract** — 3D extension plausible but not shown.
- **Per-iteration cost not fully characterized** — the paper emphasizes iteration count reduction. Need to verify per-step wall-clock on cardiac-sized grids (typically 256²–1024²) before claiming deployment-ready speedup.
