---
paper: lu_2021_deeponet
title: "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators"
authors: "Lu L, Jin P, Pang G, Zhang Z, Karniadakis GE"
year: 2021
journal: "Nature Machine Intelligence 3, 218–229 (2021)"
doi: "10.1038/s42256-021-00302-5"
pmid: ""
pdf: ../papers/lu_2021_deeponet.pdf
questions: [surrogate_pipeline]
---

## Key Findings
- **Universal approximation theorem for operators** — DeepONet can approximate any continuous nonlinear operator `G: U → V` where `U, V` are function spaces. Theoretical foundation that extends the classical Cybenko / Hornik universal approximation from finite-dim to infinite-dim.
- **Branch-trunk architecture**: two networks. **Branch net** encodes the input function `u` evaluated at fixed sensor locations `{x_1, ..., x_m}` into a vector `b ∈ ℝ^p`. **Trunk net** encodes the output query location `y` into another vector `t ∈ ℝ^p`. Output = `b · t` (inner product).
- **BC handling is fundamentally different from FNO** — DeepONet evaluates at arbitrary query points `y` via the trunk net, which is a coordinate MLP. BCs are encoded **through the input function u sampled at sensors**, not through a periodic-assumption architecture. This makes DeepONet much more flexible for non-periodic problems.
- **Small-data regime works well**. Generalization error vs data curves flatter than fully-connected baselines.
- **Broad validation**: dynamical systems (Lorenz, chemical reactions), PDEs (diffusion-reaction, Burgers, advection), stochastic ODEs.

## Method
- **Input/output**: maps input function `u: U → ℝ` to output function `v = G(u): V → ℝ`. At inference, `G(u)(y)` for a specific query `y` is computed as `branch(u@sensors) · trunk(y)`.
- **Branch net**: fully-connected or CNN, takes the m-dimensional vector `[u(x_1), ..., u(x_m)]` as input, outputs a p-dim vector `b`.
- **Trunk net**: fully-connected MLP, takes the query coordinate `y ∈ ℝ^d` as input, outputs a p-dim vector `t`.
- **Prediction**: `G(u)(y) ≈ Σ_k b_k(u) · t_k(y)`. Inner-product structure — equivalent to a finite-rank kernel expansion.
- **Training**: supervised L² loss on pairs `(u_i, y_j, G(u_i)(y_j))`. Queries can be arbitrary — they don't need to lie on a fixed grid.
- **BC treatment**: BCs are part of the input function `u`. If `u` describes initial condition plus boundary values, the branch net learns to use them. **No periodic-BC architectural assumption**, unlike FNO.

## Connections to Our Models

### Relevant Engine Components
DeepONet is architecturally more BC-friendly than FNO for our **bidomain elliptic step**. The coordinate-based trunk net can evaluate φ_e at arbitrary query points, and BC info is fed as input rather than baked into Fourier math.

However, for dense-grid bidomain prediction (we want the entire φ_e(x, y) field, not sparse queries), the branch-trunk design is inefficient compared to a full CNN. DeepONet's strength is **query-at-arbitrary-points**, which we don't need — we want the full field on a regular Cartesian grid.

### Agreements
- **Non-periodic BC handling**: the coordinate trunk net avoids FNO's periodic-BC trap. DeepONet can represent Neumann or mixed BCs as long as they're encoded in `u`.
- **Universal approximation**: theoretical guarantee we can approximate the bidomain elliptic operator to arbitrary accuracy with sufficient capacity.
- **Small-data regime**: if Bidomain V1 data generation is expensive, DeepONet's small-data efficiency is appealing.

### Disagreements or Gaps
- **Inefficient for dense-grid outputs.** To reconstruct a 256×256 φ_e field, we need 65,536 trunk-net evaluations per solve — much more expensive than a single CNN forward pass producing the full field. **This is the dominant reason DeepONet is a poor fit for our bidomain elliptic problem** vs CNN or UGrid.
- **Branch net sees u only at fixed sensors.** The sensor layout is a hyperparameter. Moving to a different tissue geometry with different resolution or sensor pattern requires retraining. FNO / CNN preserve grid flexibility better.
- **Inner-product structure limits expressivity.** `G(u)(y) ≈ Σ b_k(u) t_k(y)` is a finite-rank decomposition. For rapidly-varying inputs (sharp V_m wavefronts driving the elliptic RHS), might need large `p`.
- **No convergence guarantees under iterative rollout.** Like FNO, DeepONet produces a direct approximation with no error-correction fallback. Not preconditioner-ready out of the box.
- **No built-in awareness of PDE structure.** DeepONet is purely data-driven; doesn't exploit the elliptic operator's known mathematical structure (symmetry, discrete Green's function). UGrid / Lan 2023's multigrid or spatially-varying kernels encode more prior knowledge.

### Actionable Insights
- **LOW — Not a primary adoption target** for dense-grid bidomain φ_e prediction. CNN/UGrid-style architectures are more efficient for full-field output.
- **MEDIUM — Reference for understanding what "operator learning" means** — DeepONet and FNO are the two canonical formulations. Useful context for our literature chapter.
- **MEDIUM — Potential niche use case**: if we later want to evaluate φ_e at arbitrary (non-grid) query points — e.g., for computing effective conductance at electrode locations — DeepONet's trunk-net design is well-suited. Not a primary use case.
- **LOW — The sensor-layout brittleness means DeepONet is less attractive than UGrid/CNN for clinical deployment** across variable mesh resolutions.

## Limitations / Caveats
- **Inefficient dense-field inference** — `O(|grid|)` trunk evaluations to recover the full field. For our 256² Bidomain V1 grid: 65K evaluations per solve.
- **Fixed sensor layout**: branch net input dimensionality is hard-coded. Hard to redeploy across grid resolutions.
- **Finite-rank kernel**: `G(u)(y) = Σ_k b_k t_k` — limited by `p`. For sharp-wavefront-driven RHS, may need large `p`, which increases training cost.
- **Supervised learning** — same ground-truth-requirement issue as FNO.
- **No BC-enforcement mechanism** — BCs are implicit in the input function `u`. If the training data covers a narrow BC distribution, test-time BC shifts produce errors without warning.
- **No iterative refinement / preconditioner framing** — pure forward approximation. Errors don't self-correct.
