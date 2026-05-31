---
paper: hsieh_2019_convergence_guarantees
title: "Learning Neural PDE Solvers with Convergence Guarantees"
authors: "Hsieh J, Zhao S, Eismann S, Mirabella L, Ermon S"
year: 2019
journal: "ICLR 2019"
doi: "arxiv:1906.01200"
pmid: ""
pdf: ../papers/hsieh_2019_convergence_guarantees.pdf
questions: [surrogate_pipeline]
---

## Key Findings
- **Provably convergent neural PDE solver** — the learned update is designed so that the resulting iterative method inherits strong correctness and convergence guarantees from a classical baseline. Among the first neural-PDE papers to provide such a guarantee.
- **Single model generalizes across geometries and BCs** after training on just one geometry. Achieves **2–3× speedup** vs classical iterative solvers (Jacobi-style) on test cases.
- **Architecturally minimalist**: the NN modifies the *update rule* of a classical iterative solver — does not replace it. This is the "learn a solver component, not the full inverse" pattern that later animates Greenfeld, UGrid, NPO, and Lan 2023.
- **Foundational paper** for the preconditioner-first design philosophy. Establishes that learned corrections to classical updates can be rigorously safe.

## Method
- **Core idea**: a classical iterative solver (e.g., Jacobi, Gauss-Seidel) has an update step `u_{k+1} = u_k + α·r_k` where `r_k` is the residual. Hsieh et al. replace the scalar α with a **learned CNN-based correction term** that modifies the direction and magnitude of the update — but the method is structured so that the correction's effect on convergence is provably bounded.
- **Provable guarantees**: the key trick is to enforce that the learned update's eigenvalues in the error-propagation matrix remain inside the unit disk (or whatever stability region the classical method provides). This is done architecturally — e.g., by compositing the NN output with a projection onto the convergent subspace.
- **Training**: unsupervised on residual norms. The network is rewarded for reducing the residual faster than the classical baseline, subject to the convergence constraint.
- **Architecture**: small CNN operating on the residual field `r_k` and the current iterate `u_k`, producing a local correction. Convolutional receptive field is local — similar to a smoother, not a direct solver.
- **BC handling**: the paper claims generalization across "various boundary conditions" after training on a single geometry, but the method of handling BCs isn't a special mechanism — they're part of the operator `A` that assembles the residual. So: **BC generalization comes via operator input, not architectural modification.**

## Connections to Our Models

### Relevant Engine Components
Directly relevant to the **Bidomain V1 elliptic solver** as a prototype "safe learned acceleration" pattern. The design philosophy here — "learn a correction to a classical iterative solver's update, with provable convergence" — maps cleanly onto what we want for the bidomain elliptic step.

### Agreements
- **Convergence guarantees preserved.** Exactly the safety story we need for a preconditioner-first hybrid bidomain.
- **Unsupervised training via residual-loss objective.** Matches the Greenfeld/UGrid/NPO pattern — no ground-truth φ_e needed, just the operator A and RHS f.
- **Small, local CNN.** Cheap per-iteration, deployable at real-time bidomain rollout speeds.
- **Generalization across BCs and geometries.** The paper's demonstration that one training generalizes to unseen BCs is directly what our tissue-config variance would need.

### Disagreements or Gaps
- **Only 2–3× speedup.** Modest compared to UGrid (10×) or NPO (order-of-magnitude). The conservative "preserve convergence" constraint costs speed. For Bidomain V1 where the elliptic solve is 94% of wall-time, 2–3× is still meaningful but not transformative. Balance safety vs aggressive speedup.
- **2D only, simple scalar PDEs** (Laplace, Poisson) demonstrated. No tensor anisotropy. Same gap as every other learned-elliptic paper.
- **Iterative, not multigrid.** The method accelerates a single-grid iterative solver; doesn't exploit multiscale structure. UGrid's V-cycle is strictly more powerful at scale (1M+ DOF). For small bidomain grids (<256²) the difference may not matter.
- **Old paper (2019)** — predates most neural-operator infrastructure. Architecture is vanilla CNN; modern attention-based or multigrid-structured variants would likely outperform.
- **No preconditioner framing.** The method is a standalone iterative solver with guarantees, not a preconditioner for external Krylov methods. Adapting to preconditioner role is plausible but not discussed.

### Actionable Insights
- **HIGH — Read for the design philosophy**, not for direct code adoption. The convergence-guarantee pattern is what we want for a preconditioner-first hybrid bidomain surrogate, and Hsieh 2019 is the clearest articulation of it.
- **HIGH — Reference when proposing our own convergence story.** When we validate our dual-tower bidomain surrogate (specifically the elliptic tower), we should argue for a similar safety property: either (a) wrap in PCG so convergence is guaranteed, (b) prove a Hsieh-style bounded-error property, or (c) both.
- **MEDIUM — Fallback architecture**: if UGrid + NPO both prove too aggressive for our bidomain use case, Hsieh's single-grid learned-update pattern is a safer fallback with known-good properties.
- **LOW — Code adoption**: the paper is old enough that reimplementation from the paper is probably cleaner than adapting released code.

## Limitations / Caveats
- **Modest speedup (2–3×)** — safety-first design costs performance. Not competitive with UGrid at large scales.
- **No tensor anisotropy, no mixed BCs, no cardiac use case.** Adaptation required for all of these to apply to bidomain.
- **Jacobi-style iteration** — doesn't leverage multigrid hierarchy, FFT, or Krylov acceleration. Baselines against which the 2–3× speedup is measured may not be the best-available classical solvers.
- **2019 vintage architectures** — vanilla CNN backbone. Modern backbones (U-Net, attention, multigrid-structured CNNs) would likely push the method further but the convergence proofs are tied to the architecture choices.
- **"Convergence guarantee" means iterative convergence to a fixed point, not necessarily to the true PDE solution at a given tolerance.** The fixed point is the correct solution because the NN modifies an already-correct classical update, but the rate of convergence (beyond "converges") is not guaranteed to be fast.
