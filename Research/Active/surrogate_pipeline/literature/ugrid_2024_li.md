---
paper: ugrid_2024_li
title: "UGrid: An Efficient-And-Rigorous Neural Multigrid Solver for Linear PDEs"
authors: "Han X, Hou F, Qin H"
year: 2024
journal: "ICML 2024 (PMLR 235)"
doi: "arxiv:2408.04846"
pmid: ""
pdf: ../papers/ugrid_2024_li.pdf
questions: [surrogate_pipeline]
---

## Key Findings
- Hybrid **U-Net + multigrid V-cycle** CNN with mathematically rigorous proof of convergence and correctness for linear PDEs (Theorem 4.1) — a rare guarantee among neural PDE solvers.
- **10–20x faster** than AMGCL, **5–10x faster** than NVIDIA AmgX on large-scale Poisson; 7–20x and 5–10x respectively on Helmholtz; 2–12x and ~2–6x on steady convection-diffusion-reaction — all at residual error ≤ 1e-4.
- Self-supervised **residual loss metric** `L = E‖(1−M)(f − Ax)‖₂` enables training without ground-truth solutions and eliminates the residual-oscillation failure mode of legacy MSE losses.
- Strong **generalization to unseen boundary geometries/topology** (irregular shapes: L-shape, Star, Cat, musical-note, etc. — none in training set, which only saw "donut-like" shapes).
- **Scales without retraining**: trained at 1,050,625 DOF (large), evaluated up to 16,785,409 DOF (XXL). Baselines (AmgX) diverged on several XXL cases; UGrid converged on all.

## Method
- **Architecture**: Fully-convolutional recursive V-cycle. Each iteration = pre-smooth (ν₁ fixed Jacobi-like conv layers) → compute residual (conv) → recursive UGrid submodule (U-Net-style 2x down/up with learnable convolution layers at each level, 6 recursive levels) → correction → post-smooth (ν₂). No nonlinearities, no normalization — pure linear convolutions to preserve the linearity of the underlying multigrid iteration.
- **Masked iterator trick**: Boundary mask `M` (binary diagonal) separates interior/boundary points so the iteration `u_{k+1} = (I−M)[(I−P⁻¹A)u_k + P⁻¹f] + Mb` handles **arbitrary boundary geometry inside a regular grid** without modifying the network structure. Boundary values are re-injected after every conv layer by mask multiplication.
- **Training**: Self-supervised on residual loss only, Adam, LR=1e-3 decaying 0.1× every 50 epochs, 300 epochs, 16,000 synthesized (M, b, f) triples, single RTX 2080 Ti.
- **Input/Output**: Input = (initial guess u₀, boundary values b, boundary mask M, optional Laplacian/RHS field f). Output = numerical solution u on the same structured 2D grid. One model per PDE family (Poisson, Helmholtz, steady convection-diffusion-reaction) — **coefficients are baked into the conv stencils and retraining is required to change PDE class**.
- **Grid**: Structured 2D Cartesian, power-of-2 friendly for the 2x down/upsampling.

## Connections to Our Models

### Relevant Engine Components
Directly targets our **Bidomain V1 elliptic step**: `∇·((D_i+D_e)∇φ_e) = −∇·(D_i ∇V_m)`. UGrid's PDE family (Poisson / Helmholtz / convection-diffusion-reaction) and 5-point stencil discretization are the **same structural class** as our elliptic operator on a `StructuredGrid`. The masked-boundary trick maps cleanly onto our `BoundarySpec` pattern.

### Agreements
- **Structured grid fit**: perfect — Bidomain V1 is `StructuredGrid.create_rectangle(Nx, Ny)`, exactly the 2D regular grid UGrid assumes.
- **Linear PDE + fixed operator per simulation**: our D_i, D_e are fixed per run, so UGrid's "retrain per PDE family" limitation is less painful than for general-purpose solvers — we'd train once per tissue configuration and reuse.
- **Multigrid-style architecture matches Bidomain V1 Tier 3** (GMG); UGrid essentially replaces our hand-rolled GMG with a learned V-cycle, preserving the iterative convergence interpretation.
- **Residual loss**: elegantly avoids needing ground-truth φ_e during training — we can train directly against the elliptic residual from operator-assembled A, b.

### Disagreements or Gaps
- **Anisotropic D tensor: NOT addressed.** Paper demonstrates isotropic Laplacian (Poisson, Helmholtz) and convection-diffusion-reaction with **constant** scalar coefficients α, β plus a divergence-free-ish convection field v(x,y). Our bidomain has a full 2×2 SPD tensor D(x) that varies with fiber orientation, producing a 9-point (or larger) anisotropic stencil — not the 3×3 Laplacian kernel `[[0,1,0],[1,-4,1],[0,1,0]]` UGrid hardcodes into Eq. 16. **Adaptation required**: generalize the smoother kernel from the fixed 5-point Laplacian to a learnable anisotropic stencil (or precompute the exact finite-difference stencil for our D tensor and substitute it into J/L).
- **Boundary conditions: only Dirichlet demonstrated.** Equations 9, 12, 14 all assume `u|∂Ω = b(x,y)`. Our bidomain elliptic problem typically uses **Neumann (zero-flux)** at tissue-bath interfaces or **mixed** conditions. The masked-iterator formulation `Mu = Mb` directly encodes *values*, not *fluxes*. **Adaptation required**: either (a) add a ghost-point Neumann layer before the masked update, or (b) reformulate the mask to enforce a normal-derivative stencil. Non-trivial but feasible since it's still linear.
- **Compatibility condition / null space**: pure-Neumann elliptic has a constant null space (φ_e defined up to a constant). UGrid's theorems assume `P` is full-rank diagonal (Theorem 4.1) which is broken for the pure-Neumann case. We'd need the standard pinning or mean-zero projection — paper does not discuss this.
- **Retraining per PDE class**: they train a **separate model per PDE** (one Poisson UGrid, one Helmholtz UGrid, one diffusion-convection-reaction UGrid). If our D_i+D_e depends on tissue geometry, we either retrain per mesh or build a conditioning mechanism — **not offered by the paper**.
- **Coupled Vm↔φ_e**: UGrid solves the elliptic step in isolation; the RHS `f = −∇·(D_i ∇V_m)` would be a separate precompute. This is actually fine for our decoupled Gauss-Seidel splitting but worth noting.
- **No preconditioner usage shown**: paper presents UGrid as a **full iterative solver**, not as a preconditioner for PCG. Worth testing both — a learned V-cycle preconditioner could inherit Krylov's robustness while exploiting learned speedups.

### Actionable Insights
- **HIGH — Prototype UGrid as drop-in for Tier 3 GMG elliptic solver**: the architecture matches our problem class and claims 10× wall-clock speedup. Specifically, build a `UGridEllipticSolver` sibling to `DecoupledBidomainDiffusionSolver`'s Tier 3 path, trained offline against residual loss.
- **HIGH — Adopt the residual loss**: `L_abs = E‖(1−M)(f − Ax)‖₂` is exactly what we want — trains without needing φ_e ground truth from Bidomain V1, just the assembled operator A and RHS. This removes the data-generation cost that would dominate supervised training.
- **HIGH — Adapt masked iterator to Neumann BCs**: this is the single biggest engineering task for adoption. Prototype a `NeumannMask` variant that enforces `∂u/∂n = g` instead of `u = b`. Validate against the analytical test in `Bidomain/Engine_V1/tests/`.
- **MEDIUM — Extend stencils to anisotropic tensor D**: replace the hardcoded 5-point Laplacian `L` in Eq. 16 with the 9-point anisotropic FD stencil derived from D_i+D_e. Convergence proof (Theorem 4.1) should still go through as long as P remains full-rank diagonal.
- **MEDIUM — Test as preconditioner for PCG**: even if standalone UGrid doesn't hit tolerance, a learned V-cycle preconditioner could beat ICC/Jacobi. Our Tier 2 (`PCG + Spectral`) is a natural slot for a `PCG + UGrid` variant.
- **MEDIUM — Generalization test**: they prove the method generalizes to unseen geometries of similar class. For us this means one UGrid could handle multiple cardiac mesh geometries (different infarct scars, different fiber fields) if trained on a diverse set.
- **LOW — Reference implementation available**: https://github.com/AXIHIXA/UGrid (open-source). PyTorch, so direct integration with our stack. Use as a starting point, not as a canonical API.
- **LOW — Scalability**: they scale from 1M to 16M DOF without retraining. Cardiac meshes are typically <1M nodes; we are well within their demonstrated regime.

## Limitations / Caveats
- **Linear PDEs only** — Theorem 4.1 does not cover nonlinear PDEs. Fits our elliptic step (linear in φ_e) but not the full reaction-diffusion Vm equation. Authors flag this as future work.
- **No rate guarantee**: "no mathematical guarantee on *how fast* UGrid will converge" — may not beat small-scale legacy solvers, only large-scale ones. Bidomain V1 on 512×512 grids is likely in the regime where UGrid wins, but tiny problems (e.g. 64×64) may not benefit.
- **Dirichlet-only in paper**: Neumann/mixed BC handling is an assumption we'd need to prove separately. No experimental results exist for flux BCs.
- **One model per PDE family**: coefficients are baked into the network. Varying D(x) across simulations requires either retraining or a conditioning mechanism the paper does not provide.
- **2D only in the paper**: 3D extension is presumably straightforward (swap 2D convs for 3D) but not demonstrated. Our current Bidomain V1 is 2D so this matches for now.
- **Training data distribution matters**: they train on "donut-like" geometries; generalization to arbitrary cardiac meshes (thin fibers, sharp infarct boundaries) is plausible but unverified. Would need an ablation on our domain.
- **Ablation shows vanilla U-Net with residual loss diverges** on most test cases — the multigrid-inspired masked convolutional structure, not just the loss, is load-bearing for correctness. Don't naively swap in any U-Net.
