---
paper: nie_2026_jaws
title: "JAWS: Enhancing Long-term Rollout of Neural PDE Solvers via Spatially-Adaptive Jacobian Regularization"
authors: "Nie F, Suzuki Y"
year: 2026
journal: "arXiv preprint (Mar 2026)"
doi: "arxiv:2603.05538"
pmid: ""
pdf: ../papers/jaws_2026.pdf
questions: [surrogate_pipeline]
---

## Key Findings
- **Addresses the universal autoregressive failure mode**: step-wise error accumulation leads to "spectral blow-up and unphysical divergence" during long rollouts. Same problem every cardiac NN surrogate has (AGATA, LFLDNet, Lydon PINO all document it).
- **Previous fixes are too blunt**: uniform Jacobian regularization damps high-frequencies uniformly, which over-smooths everywhere. JAWS solves this by making the regularization strength **spatially adaptive** — strong in smooth regions, relaxed near sharp features (shocks, gradients).
- **MAP estimation framework with spatially heteroscedastic uncertainty**: principled probabilistic formulation. Loss-term reformulation — not an architectural constraint, not an inference-time trick. Trains any architecture to be more stable under rollout.
- **Memory-efficient training for long-horizon accuracy**: "short-horizon, memory-efficient training to match the accuracy of long-horizon baselines." Removes the TBPTT / adjoint-gradient memory bottleneck that usually constrains long-rollout training.
- **Tested on shocks / gradient-rich problems** (1D viscous Burgers, 2D flow past cylinder at out-of-distribution Re=400). Wavefront-like dynamics — directly transferable to cardiac upstrokes which are "shocks" in the excitable-media sense.
- **Open-source code**: `github.com/jyohosyo-dot/JAWS_2D`.

## Method
- **Core reformulation**: operator learning as **Maximum A Posteriori (MAP) estimation with spatially heteroscedastic uncertainty**. Each spatial location has its own uncertainty parameter σ(x), which the model learns from data. Regularization weight at each location is inversely related to σ(x) — smooth regions get strong regularization (small σ → large weight), sharp regions get relaxed regularization (large σ → small weight).
- **Jacobian penalty**: penalize the Jacobian of the one-step operator (gradient of output w.r.t. input) to enforce contractive dynamics. Contractive = stable under iteration.
- **Spatial adaptivity**: instead of one global regularization weight, per-location weights `λ(x)` learned alongside the model. Data-driven rather than hand-tuned.
- **Training**: short-horizon training (few rollout steps) with the spatially-adaptive Jacobian penalty. Claims match long-horizon training accuracy at a fraction of the memory.
- **Architecture-agnostic**: JAWS is a training/loss technique. Wraps any architecture.
- **Benchmarks**: 1D Burgers (shock formation — classic reaction-diffusion shock benchmark), 2D flow past cylinder (Re=400, out-of-distribution). Both exhibit sharp-gradient features.

## Connections to Our Models

### Relevant Engine Components
**Training-stability technique for the full bidomain rollout**, where V_m dynamics have sharp wavefronts (cardiac "shocks"). If our dual-tower surrogate is trained on short bidomain segments and then deployed autoregressively over 30K steps, JAWS could prevent the spectral-blow-up failure mode that cripples AGATA-style GNN surrogates at long horizons.

### Agreements
- **Sharp features matter** — uniform regularization is wrong for cardiac wavefronts. JAWS's spatial-adaptivity is the right abstraction.
- **Training-time fix, not inference-time** — no per-step inference cost (unlike PDE-Refiner). Important because our 30K-step bidomain rollout can't afford K× per-step cost.
- **Works with any architecture** — composable with PDE-Transformer, CNO, UGrid, etc. Same pattern as NOWS for deployment.
- **Memory-efficient short-horizon training** — matches our computational constraints (single GPU, can't TBPTT through 30K steps).
- **Open-source code** — adoption-friction low.

### Disagreements or Gaps
- **Tested on fluid-dynamics shocks, not cardiac wavefronts.** Math is analogous (sharp gradient fronts in RD / conservation laws) but empirical transfer is untested.
- **MAP estimation framework** requires probabilistic interpretation of the output. Our dual-tower surrogate would need minor architectural extension to output per-location uncertainty σ(x) alongside the field prediction. Not trivial — adds a head per tower.
- **Short-horizon training guarantees long-horizon stability *empirically***, not theoretically. If our bidomain dynamics have failure modes absent from Burgers / cylinder flow, JAWS may not cover them.
- **No reaction-diffusion benchmark explicitly**. Burgers is close (shock formation, scalar conservation) but not exactly RD. FitzHugh-Nagumo or Fisher-KPP would be better validation.
- **Abstract March 2026** — just-released paper, no peer review yet.
- **Additional loss components to tune**: the MAP framework introduces hyperparameters (prior, uncertainty parameterization) that need careful tuning.

### Actionable Insights
- **MEDIUM — Reserve for Phase 2.** Get the dual-tower working on short bidomain rollouts first. If we see drift at long horizons, JAWS is the right next tool.
- **MEDIUM — Read PDF for architectural details** on how σ(x) is parameterized. Extending our dual-tower to output uncertainty fields is cheap if parameterized well (one extra output channel per tower).
- **MEDIUM — Benchmark against uniform-Jacobian-regularization baseline** to validate the spatial-adaptivity claim on our specific dynamics. Ablation.
- **LOW — Code availability**: `github.com/jyohosyo-dot/JAWS_2D` — study the loss implementation; the rest of the architecture is interchangeable.
- **LOW — Combination with PDE-Refiner**: JAWS (training-time) + PDE-Refiner (inference-time) are orthogonal. Both could be applied if drift is severe. Cost: K× inference for refiner, zero extra for JAWS.

## Limitations / Caveats
- **Fluid-dynamics benchmarks only** (Burgers, cylinder flow). Cardiac RD transfer is an open question.
- **Very recent paper (Mar 2026)** — no peer review, methodology may refine.
- **Probabilistic output requirement** — architecture extension needed to produce uncertainty alongside field prediction.
- **Hyperparameter tuning** — MAP framework introduces new knobs (prior, uncertainty param) that need care.
- **Empirical long-horizon stability, not theoretical** — no guarantee if our dynamics are qualitatively different from their benchmarks.
- **Memory-efficient claim** is critical; verify it on our specific 30K-step horizon before committing. Paper's "long-horizon" may mean thousands, not tens of thousands.
- **Orthogonal to BC handling** — doesn't fix FNO's periodic assumption if used with FNO backbone.
