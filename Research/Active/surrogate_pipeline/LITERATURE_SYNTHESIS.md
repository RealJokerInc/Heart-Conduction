# Neural Surrogates for Cardiac Bidomain Electrophysiology: A Review of the Architectural Design Space

> **Scope.** This document synthesizes the 27 papers currently filed in `literature/` into a structured review of the neural-PDE landscape relevant to a hybrid bidomain cardiac-electrophysiology surrogate. It presents each architectural family with its intuition, the problem it solves, a layer-level mathematical sketch, design philosophy, and concrete limitations, then cross-cuts by recurring themes (boundary conditions, long-horizon stability, convergence guarantees) and closes with the adopted design stack. Written in the style of a Nature review, not a tutorial.

---

## 1. Abstract

Cardiac electrophysiology simulation is dominated by a single numerical bottleneck: the elliptic solve for extracellular potential φ_e, which consumes ~94% of wall-clock time in decoupled bidomain solvers. Two decades of classical work (multigrid, Krylov methods, spectral solvers) have brought this cost down by perhaps two orders of magnitude; neural surrogates promise another one or two, but only if they target the right sub-problem with the right architecture. The existing cardiac ML surrogate literature is narrow and monodomain-focused: every published method either replaces the full PDE (AGATA, LFLDNet, Lydon PINO, Centofanti FNO/KOL) or skips the spatial field entirely (Ziarelli ECG forward, Salvador LNODE). **Nobody co-solves φ_e as a learned spatial field.** Meanwhile, a much deeper literature on learned elliptic solvers from computational fluid dynamics (Greenfeld multigrid, Özbay Poisson CNN, Hsieh convergence-guaranteed solvers, Lan mixed-BC preconditioner, NPO, UGrid, MGCNN) solves exactly the problem the bidomain needs, but has never been ported. This review maps that cross-domain opportunity, identifies the architectural families that are BC-correct and convergence-safe for bidomain, and argues for a hybrid design: **classical ionic scaffold + dual CNN-transformer towers with channel-wise cross-attention, deployed as a neural warm-start for the existing PCG solver**.

---

## 2. The Problem Space

### 2.1 Bidomain electrophysiology in one equation block

The standard (parabolic-elliptic) bidomain system couples a reaction-diffusion equation for transmembrane potential V_m with a quasi-static constraint for extracellular potential φ_e:

```
Parabolic:   χ C_m ∂V_m/∂t = ∇·(D_i ∇(V_m + φ_e)) − χ I_ion(V_m, s)
Elliptic:    0 = ∇·((D_i + D_e) ∇φ_e) + ∇·(D_i ∇V_m)
Gating:      ds/dt = f_s(V_m, s)
```

Here χ is cell surface-to-volume ratio, C_m is membrane capacitance, D_i and D_e are intracellular and extracellular conductivity tensors (SPD 2×2 in 2D, varying with fiber orientation), s is the ionic state vector (18 components for TTP06, 41 for ORd), and I_ion is a nonlinear function parameterized by ~50 conductances and rate constants.

The decoupled Gauss-Seidel splitting used in our Bidomain V1 solves these three stages sequentially at each timestep. Wall-time profiling gives approximately:

| Step | Structure | % wall-time |
|------|-----------|------------:|
| Ionic (I_ion + gating) | pointwise ODE | ~3% |
| Parabolic (diffusion of V_m) | sparse tridiagonal / spectral | ~3% |
| Elliptic (solve for φ_e) | sparse SPD linear system | **~94%** |

The elliptic step is the natural target for a surrogate. It is linear (in φ_e), time-local (one solve per timestep), globally coupled (Green's function has full-domain support), and numerically well-studied.

### 2.2 What makes bidomain elliptic different from textbook Poisson

Three structural features set it apart from the toy problems most learned-Poisson papers benchmark:

1. **Anisotropic tensor coefficients.** D_i + D_e is a full 2×2 SPD tensor at every point, varying with fiber orientation. Discretization produces a 9-point (or larger) stencil, not the 5-point Laplacian `[[0,1,0],[1,-4,1],[0,1,0]]` that every paper hardcodes.
2. **Mixed Neumann/Dirichlet boundary conditions.** Zero-flux Neumann at tissue-bath interfaces, occasional Dirichlet pinning at grounding electrodes, Robin-like coupling at certain anatomical boundaries. Pure-Neumann cases have a constant null space (φ_e defined up to a constant), breaking convergence assumptions that require full-rank preconditioners.
3. **Non-smooth right-hand side.** The source term ∇·(D_i ∇V_m) concentrates sharply at cardiac wavefronts (~1 mm wide in a ~10 cm domain), producing high-frequency spatial content that stresses any surrogate with a spectral bias.

All three features are absent from the standard learned-Poisson benchmarks (Laplace on a square, Helmholtz with smooth source, Darcy flow with smooth coefficients). Cross-domain adoption is the opportunity, not the automatic win.

### 2.3 The cardiac-ML literature has looked elsewhere

The five pre-filed cardiac NN surrogate papers (Morier AGATA, Salvador LFLDNet, Salvador LNODE, Salvador CMAME branched LNM, Lydon PINO, Centofanti FNO/KOL, Ogbomo-Harmitt ECG forward) occupy three clear camps: whole-PDE monodomain replacement (GNN, FNO, CfC, PINO), zero-D organ-level surrogates (LNODE), and post-hoc lead-field ECG maps (Ogbomo-Harmitt). None learn φ_e as a spatial field, and none operate on bidomain. The gap is not accidental — it is structural: cardiac simulator researchers lean on the monodomain approximation (which eliminates φ_e entirely at ~10% accuracy cost) because the bidomain elliptic solve was considered too hard to learn. It isn't.

---

## 3. Taxonomy: Where a Surrogate Can Enter a Classical PDE Solver

Neural surrogates for PDE simulation fall into four architectural regimes, distinguished by how they interact with the classical solver:

### 3.1 Full-PDE replacement

The neural model maps input fields directly to future field states, entirely bypassing the numerical solver. Autoregressive iteration extends to long horizons. **Pros:** maximum speedup, end-to-end differentiability. **Cons:** no convergence guarantees, error compounds across rollout steps, BC handling must be learned from data, clinical defense is hard. Examples: Morier AGATA, Salvador LFLDNet, Lydon PINO.

### 3.2 Split-step hybrid (one stage classical, one stage neural)

The simulator retains some stages in classical form (often the stiff, well-understood, or analytically tractable ones) and replaces others with neural surrogates. Our hybrid-bidomain pivot is an instance: classical ionic (TTP06 with compiled Rush-Larsen), classical parabolic (well-studied), neural elliptic. **Pros:** inherits classical guarantees for the classical stages, targets only the problematic step, differentiable by stage. **Cons:** engineering complexity of dual paths, cross-stage coupling can be fragile, residual 6% of wall-time from the classical stages caps speedup. Example: this project.

### 3.3 Learned preconditioner for Krylov / multigrid

The neural model produces an approximate inverse `M⁻¹` of the discrete operator `A`, used inside a Krylov iteration `x_{k+1} = x_k + M⁻¹ r_k` or multigrid smoother. The Krylov method still converges to the exact solution regardless of preconditioner quality; the net only affects iteration count. **Pros:** convergence guarantees preserved, graceful OOD degradation, mathematically defensible. **Cons:** per-step cost must be low enough that fewer iterations × NN-overhead < classical iteration cost; difficult to learn efficient preconditioners for highly anisotropic operators. Examples: Greenfeld prolongation, Lan 2023 spatially-varying preconditioner, NPO.

### 3.4 Warm-start for iterative refinement

The neural model produces an initial guess `x₀ = NN(A, b)`, then a classical iterative solver refines to tolerance. A degenerate case of preconditioning, applied once at the start of each solve rather than inside every iteration. **Pros:** maximal separation between net and solver — any NN architecture works, classical solver is untouched. **Cons:** no runtime feedback from solver to NN; speedup bounded by how close `NN(A,b)` can get to `x*`. Example: NOWS (Eshaghi 2025).

The four regimes form a safety-speedup tradeoff curve. Regime 1 is fastest but riskiest; regime 4 is safest but extracts less speedup. Our hybrid bidomain surrogate sits at regime 2 (split-step), with regime 4 (warm-start) as the deployment wrapper. Preconditioner (regime 3) is a fallback if warm-start turns out to not extract enough speedup.

---

## 4. Architectural Families

This section surveys the twelve architectural families represented in `literature/`, presenting each with a uniform structure: **intuition**, **problem solved**, **mathematical form** (layer-level digest), **design philosophy**, **our take**.

### 4.1 Graph Neural Networks (GNN)

**Intuition.** Cardiac tissue is naturally a graph — cells (or mesh nodes) are vertices, conductive couplings are edges. A GNN updates each node's state based on its neighbors' states via message passing, which is exactly the discretized form of a diffusion operator.

**Problem solved.** Geometric generality. Unstructured meshes, patient-specific anatomy, arbitrary connectivity — all handled by the same message-passing machinery. Training on one geometry generalizes to others.

**Mathematical form.** Morier's AGATA uses GATv2Conv layers with edge-aware attention:

```
x_i^{(l+1)} = σ( Σ_{j ∈ N(i)} α_{ij}^{(l)} W_n^{(l)} (x_i^{(l)} + x_j^{(l)}) + W_e^{(l)} e_{ij} )

α_{ij} = softmax( LeakyReLU( a^T [W_n (x_i^s + x_j^s) + W_e e_{ij}] ) )
```

Three such layers, autoregressive window T_w = 5 timesteps, concatenate-linear-sigmoid decoder. ~50k parameters.

**Design philosophy.** Inductive bias of locality (message passing) + geometric invariance (permutation-equivariant). The mesh topology is the inductive bias; the network just learns the local update rule.

**Our take.** GNNs are right for patient-specific unstructured meshes but wrong for structured Cartesian Bidomain V1. The GNN formalism reduces to convolution on a regular grid (same thing, different notation) so we pay GNN's engineering overhead without its geometric benefit. Reserve for Phase B if we ever extend past structured grids. See `literature/morier_2025_agata_gnn.md`.

### 4.2 Fourier Neural Operators (FNO)

**Intuition.** Classical PDE operators are often simple in Fourier space: the Laplacian becomes pointwise multiplication by −k². Learning an operator in Fourier space — multiplying by a learnable complex tensor on truncated modes — captures global spatial coupling efficiently.

**Problem solved.** Parameter efficiency for smooth-field operators. Resolution invariance (same weights, any grid size, via continuous-function interpretation). Theoretical parametric-PDE generalization.

**Mathematical form.** An FNO layer is:

```
v(x) → F̂(v) ∈ C^{N×d}    (FFT, d = channel count)
     → R_θ · F̂(v) ∈ C^{k×d}  (truncate to k modes, multiply by learned complex tensor R_θ ∈ C^{k×d×d})
     → F̂⁻¹(result) ∈ R^{N×d} (inverse FFT)
Final:  y(x) = F̂⁻¹(R_θ · F̂(v))(x) + W·v(x)   (skip with learned linear map W)
```

Stacked 4–8 layers, GELU between, lift/project MLPs for channel count.

**Design philosophy.** Low-mode spectral bias as regularization. Global context for free via FFT. Discretization invariance as a continuous-function-space property.

**Our take.** **FNO is a non-starter for bidomain.** The FFT at its heart assumes periodic boundary conditions — mathematically, not as a hyperparameter. Bidomain is Neumann-dominant with mixed BCs. Non-periodic inputs fed to FFT produce Gibbs-like artifacts that propagate through the network and accumulate across autoregressive timesteps, destabilizing long rollouts. Cardiac papers that have used FNO (Lydon PINO, Centofanti FNO) work around this either with artificial periodic padding (introducing O(grid-width) boundary error) or by reformulating the problem as BC-implicit (Centofanti's stimulus→AT/RT is single-shot, which hides the rollout instability). For our 30K-step autoregressive bidomain rollout, neither workaround is acceptable. See `literature/li_2020_fno.md`, and the same BC critique applies to `literature/lydon_2025_pino_cardiac.md` and `literature/centofanti_2025_fno_kol_cardiac.md`.

### 4.3 DeepONet

**Intuition.** An operator maps input functions to output functions. Approximate this map by a finite-rank decomposition: `G(u)(y) ≈ Σ_k b_k(u) · t_k(y)` where b_k encodes the input function and t_k encodes the output location.

**Problem solved.** Universal approximation theorem for operators (Cybenko-Hornik extended to function spaces). Arbitrary output-location query — evaluate the solution at any point without re-running the network.

**Mathematical form.** Two sub-networks:

```
Branch net:   b(u) = MLP_b([u(x_1), u(x_2), …, u(x_m)]) ∈ R^p
Trunk net:    t(y) = MLP_t(y) ∈ R^p
Output:       G(u)(y) ≈ <b(u), t(y)> = Σ_k b_k · t_k
```

`p` is the rank of the learned operator approximation; universality requires p → ∞.

**Design philosophy.** Decouple the "what does the operator do" (branch) from the "where do I want the answer" (trunk). Inner-product structure enables arbitrary query points with no retraining.

**Our take.** Wrong fit for bidomain. The strength — arbitrary query locations — is unused because we want the entire φ_e field on a regular grid. The cost — one trunk-net evaluation per query point — scales as O(N²) for a full grid (65K trunk evals on a 256² bidomain domain). A plain CNN produces the full field in one forward pass. Worth knowing as the canonical "other" operator-learning paradigm (vs FNO), but not an adoption target. See `literature/lu_2021_deeponet.md`.

### 4.4 Convolutional Neural Operators (CNO)

**Intuition.** Plain CNNs are mesh-sample-dependent (grid-refinement changes what the network does). Treat convolutions as discretizations of continuous integral operators, use explicit anti-aliasing filters before downsampling, and you recover discretization invariance — a proper neural operator with a CNN backbone.

**Problem solved.** Non-periodic operator learning at parity with FNO on accuracy but without FNO's BC limitation. Discretization invariance with a universality theorem. Rehabilitates CNNs as proper operator-learning tools.

**Mathematical form.** A CNO block is a standard convolutional encoder-bottleneck-decoder, but with:

```
Downsample:   x_coarse = (filter * x) ↓ 2    (anti-aliasing filter BEFORE stride-2)
Block:        x_new = σ( Conv(x) + Conv(Skip(x)) )
Upsample:     x_fine = Interp(filter * x)    (anti-aliasing filter, then interpolation)
```

Filter is chosen so the coarse representation is a proper projection of the fine one; no aliasing.

**Design philosophy.** Respect the continuous-function interpretation at every architectural choice. Prove universality as a design constraint, not an afterthought. CNN inductive bias (translation equivariance, locality) for free.

**Our take.** **One of two primary backbone candidates** for our dual-tower elliptic surrogate. Universality theorem gives formal defensibility; BC handling is clean (no periodic assumption); open-source code is available and maintained by Mishra's group at ETH (same lab as Poseidon). Trade-off vs PDE-Transformer (§4.9): CNO is simpler and lighter, PDE-Transformer is more expressive and has the "channels-as-tokens" cross-attention pattern natively. See `literature/raonic_2023_cno.md`.

### 4.5 Learned Multigrid Solvers

**Intuition.** Classical multigrid gets its speed from its architecture: smooth high frequencies on the fine grid, restrict to coarser grids where low frequencies look high and are smoothed cheaply, prolong back up. Each multigrid component (smoother, prolongation, restriction, coarse-grid operator) is a design choice; learning replaces hand-crafted choices with data-driven ones.

**Problem solved.** Optimal solver-structure match to a specific PDE family without manual tuning. Provable convergence when the learned components preserve multigrid's structural guarantees. Scales to large grids.

**Mathematical form.** A V-cycle iteration is (recursive over levels):

```
V(A, f, u):
    u ← Smooth_ν₁(A, f, u)              (pre-smooth: Jacobi-like conv layers)
    r ← f − A u                          (compute residual)
    e_coarse ← V(A_coarse, R r, 0)       (recursive solve on coarse grid)
    u ← u + P e_coarse                   (prolongate correction)
    u ← Smooth_ν₂(A, f, u)              (post-smooth)
    return u
```

Learnable: R (restriction), P (prolongation), coarse-grid operator A_coarse, smoother weights. UGrid learns the full V-cycle as a CNN with masked iterator for BCs; MGCNN learns a subset with linearity preservation; Greenfeld learns only P and demonstrates this is sufficient.

**Design philosophy.** Structural inductive bias = multigrid's V-cycle topology. Learned components fill in the "how to do it right for this PDE" that hand-crafting gets wrong on anisotropic or heterogeneous coefficients.

**Our take.** **The structural match for bidomain elliptic.** UGrid, MGCNN, and Greenfeld are all candidates. UGrid is most open-source-adoption-friendly and has the masked-iterator BC mechanism; MGCNN claims the concrete 3–8× speedup vs classical GMG on heterogeneous coefficients (cardiac's fiber-dependent D analog); Greenfeld is foundational for understanding the lineage. Expect to adapt UGrid's masked iterator from Dirichlet to Neumann for bidomain. See `literature/ugrid_2024_li.md`, `literature/xie_2023_mgcnn.md`, `literature/greenfeld_2019_multigrid.md`.

### 4.6 Neural Preconditioners

**Intuition.** A preconditioner `M⁻¹ ≈ A⁻¹` accelerates Krylov iteration. Learning the preconditioner sidesteps the need to learn the full inverse, which is lossy and unstable. The Krylov method still converges to the true answer — the NN only decides *how fast*.

**Problem solved.** Speedup without sacrificing convergence guarantees. Graceful out-of-distribution degradation (worst case: more iterations). Bounded worst-case behavior under distribution shift.

**Mathematical form.** Inside preconditioned conjugate gradient (PCG) for `A x = b`:

```
Standard PCG:  z_k = M⁻¹ r_k    (preconditioner applied each iteration)
Learned M⁻¹:   z_k = NN(r_k, A, BCs, parameters)

Training loss (Lan 2023 style): L = E_{A,b} ‖(r - A NN(r, A)) / ‖r‖‖²
                                (residual reduction per application)
NPO style loss:                   L = κ(M⁻¹ A) + residual_loss
                                  (minimize condition number of preconditioned system)
```

Lan 2023's architecture uses **spatially-varying convolution kernels** — at each grid point, the kernel is a function of local BC indicator values — explicitly encoding BC-dependent behavior that standard CNNs cannot.

**Design philosophy.** Trust the Krylov method's correctness properties; only the preconditioner is learned. Safety by construction.

**Our take.** **Primary deployment pattern** for the hybrid bidomain surrogate. Lan 2023's spatially-varying kernels are directly applicable to our Neumann-dominant mixed-BC case (BC indicator fields as input channels). NPO provides an alternative loss formulation (condition number + residual) that ensures the learned operator is a genuinely good preconditioner rather than a lossy approximate inverse. Combined with NOWS's warm-start framing, the whole hybrid bidomain becomes "replace Bidomain V1's current preconditioner with a learned one; keep everything else." See `literature/lan_2023_neural_preconditioner_mixed_bc.md`, `literature/npo_2025_cai.md`, `literature/eshaghi_2025_nows.md`, `literature/hsieh_2019_convergence_guarantees.md`.

### 4.7 Liquid / Closed-form Continuous (CfC) Networks

**Intuition.** Neural ODEs `dz/dt = f_θ(z)` require a numerical integrator at inference, inheriting its step-size constraints (CFL for stiff systems). Replace the ODE with a closed-form gated update that produces `z(t+Δt)` in one step for any Δt. The gating function naturally saturates at large Δt, giving bounded dynamics at any scale.

**Problem solved.** Stiffness-free temporal dynamics. Training at small Δt, inference at large Δt (or continuous-time). Elimination of per-step ODE solver calls.

**Mathematical form.** CfC update at arbitrary Δt:

```
s(t + Δt) = g(s, μ, Δt) ⊙ h(s, μ, Δt) + (1 − g(s, μ, Δt)) ⊙ f(s, μ, Δt)
```

where f, g, h are feedforward networks and ⊙ is elementwise product. Gating function g ∈ [0, 1] acts as a learned "how much to update," naturally saturating for large Δt (bounded dynamics). No numerical integration; evaluation is one forward pass.

**Design philosophy.** Bounded dynamics through architectural gating. No CFL condition, no step-size controller. Closed-form convergence guarantees when gating is bounded.

**Our take.** Used successfully in cardiac monodomain by Salvador LFLDNet (30× speedup, dt=10 ms at inference vs 0.1 ms for FEM). Highly relevant for the **ionic side** of our hybrid design — if we eventually revisit the ionic surrogate path — because cardiac ionic dynamics are stiff (gate timescales span 0.1–100 ms). Not directly relevant for the elliptic side (which is instantaneous, no time integration). Keep in back pocket for ionic optimization. See `literature/salvador_2025_lfldnet.md`.

### 4.8 Latent Neural ODEs (LNODE)

**Intuition.** High-dimensional cardiac state lives on a low-dimensional manifold. Encode to latent, integrate the manifold dynamics there, decode back. The latent ODE `dz/dt = f_θ(z)` is small, fast, and differentiable.

**Problem solved.** Parameter estimation with uncertainty quantification. 0-D hemodynamic surrogate at 300× speedup. Differentiable ODE flow for patient-specific parameter fitting.

**Mathematical form.**

```
z₀ = Encoder(x_initial)                     (reduce to low-dim latent)
dz/dt = f_θ(z, μ)                           (latent ODE, μ = model parameters)
z(t) = ODESolve(f_θ, z₀, t, μ)              (integrate with adjoint)
x(t) = Decoder(z(t))                        (reconstruct output)
```

Salvador's electromechanics LNODE uses 3 hidden layers × 13 neurons = 39-neuron latent ODE for whole-heart electromechanics. Tiny.

**Design philosophy.** Dimension reduction as compression. Solve the easy problem (small ODE) instead of the hard one (high-D PDE). Differentiability for inverse problems.

**Our take.** Solves a different problem than ours. LNODE targets 0-D hemodynamic outputs (pressure-volume loops), not spatial field evolution. For our bidomain elliptic we need the full 2D φ_e(x, y), which has no obvious low-dimensional manifold structure (global Green's function coupling). Useful conceptual precedent for the latent approach but not architectural template. See `literature/salvador_2024_lnode_cardiac.md`.

### 4.9 Transformers for PDEs

**Intuition.** Self-attention computes pairwise interactions over a set of tokens. On a 2D grid, each pixel is a token; attention captures global coupling naturally. Cost is O((HW)²) for full attention, intractable beyond small grids — hence the windowed, shifted, and factorized variants.

**Problem solved.** Global context at scale. Cross-field coupling (different physics channels attend to each other). Foundation-model pretraining (one backbone, many PDEs).

**Mathematical form.** The three main variants:

- **Shifted-window (Swin-style, used in Poseidon and PDE-Transformer):**
  ```
  Partition grid into W×W windows.
  Attention within each window: O(W²) per window × (H/W × W/W) windows = O(HW × W²)
  Shift windows by W/2 at alternate layers to enable cross-window flow.
  ```
- **Channel-wise (PDE-Transformer):**
  ```
  Tokens = (spatial_position, physical_channel)
  Spatial attention within channel (windowed), channel attention across fields at each position.
  ```
- **Physics-attention slices (Transolver):**
  ```
  Soft-assign each mesh point to one of M slices via learned clustering.
  Compute attention over M slice-tokens: O(M²) with M ≪ N.
  Broadcast slice features back to mesh points.
  ```

**Design philosophy.** Generality > specialization. Replace hand-crafted operators (conv kernels, FFT) with learned attention that adapts to data. Linear-cost variants make scaling tractable.

**Our take.** **Second primary backbone candidate** (the first being CNO). PDE-Transformer's "channels-as-tokens with channel-wise self-attention" is literally the dual-tower-with-cross-talk architecture we've been sketching. Poseidon provides a pretrained checkpoint we could fine-tune. DRIFT-Net offers a dual-branch variant with bandwise-weighted cross-mixing. Transolver is Phase-B relevant (unstructured meshes). Windowed attention is the clean answer to "attention without FNO's periodic trap." See `literature/holzschuh_2025_pde_transformer.md`, `literature/herde_2024_poseidon.md`, `literature/wu_2024_transolver.md`, `literature/li_2026_driftnet.md`.

### 4.10 Diffusion Refinement (PDE-Refiner)

**Intuition.** Autoregressive PDE surrogates fail because MSE training under-represents non-dominant frequencies; small high-frequency errors compound into visible long-horizon divergence. Diffusion model training forces the network to predict all frequencies uniformly (Gaussian noise has flat spectrum), so a diffusion-style refinement chain at each timestep recovers the suppressed frequencies.

**Problem solved.** Long-horizon autoregressive stability without architectural modification. Architecture-agnostic wrapper.

**Mathematical form.** At each rollout step t → t+1:

```
u_{t+1}^{(0)} = BaseNN(u_t)                           (initial prediction)
for k in 1..K:
    u_{t+1}^{(k)} = u_{t+1}^{(k-1)} + DenoiseNN(u_{t+1}^{(k-1)}, k, u_t)
return u_{t+1}^{(K)}                                  (refined prediction)
```

K refinement steps per rollout step. Denoise net trained on diffusion-style noise-injection schedule.

**Design philosophy.** Use the frequency-uniform training signal of diffusion models to fix neural PDE surrogates' spectral bias. Wrap any base model.

**Our take.** Reserve for later stability issues. The K× inference cost multiplier is expensive for our 30K-step bidomain rollouts. McCabe's architectural fixes (§4.12) are cheaper; try those first. PDE-Refiner is a backup if drift persists and architectural surgery isn't enough. Strongly **incompatible with FNO backbones** — it cannot fix periodic-BC artifacts, only spectral bias. See `literature/lippe_2023_pde_refiner.md`.

### 4.11 Jacobian Regularization (JAWS)

**Intuition.** If the one-step neural operator has a Jacobian with spectral radius > 1 anywhere, errors at that location amplify over rollout. Regularize the Jacobian to enforce contractive dynamics. Uniform regularization over-smooths sharp features; make it spatially adaptive.

**Problem solved.** Long-horizon stability via a training-time penalty rather than inference-time cost. Preserves sharp fronts (cardiac wavefronts) while damping noise elsewhere.

**Mathematical form.** MAP estimation with spatially-adaptive prior:

```
L_rollout = ‖u_pred − u_true‖² + λ(x) · ‖∂u_pred/∂u_prev‖²
λ(x) = 1 / σ²(x)                                       (learned uncertainty at each location)
```

σ(x) is a learned per-location uncertainty map; smooth regions have small σ (strong regularization), sharp regions have large σ (weak regularization).

**Design philosophy.** Operator-level stability via Jacobian contraction. Spatial adaptation prevents the over-smoothing failure mode of uniform penalties.

**Our take.** Composable with any backbone (CNO, PDE-Transformer, UGrid). No per-step inference cost. Excellent candidate for the training regime when the full bidomain rollout is assembled — prevents spectral-blow-up without smearing wavefronts. See `literature/nie_2026_jaws.md`.

### 4.12 Autoregressive-Stability Architectural Fixes

**Intuition.** Certain architectural operations (unnormalized activations, BatchNorm over feature dims, unbounded feedback paths) generically amplify errors during autoregressive iteration. Identify and replace them; stability follows.

**Problem solved.** Long-rollout stability via architectural constraints rather than training tricks or inference-time refinement. Zero runtime cost.

**Mathematical form.** No single equation — the paper (McCabe 2023) is prescriptive about architectural patterns:

- Replace BatchNorm with GroupNorm or LayerNorm when the feature dimension is small.
- Ensure activations have bounded outputs or bounded gradients in the relevant regime.
- Audit feedback connections for contractive vs amplifying behavior in the Jacobian sense.
- Place residual connections such that the zero-function is an achievable initial state.

**Design philosophy.** Architecture as a first-class stability concern. Safety by structural design; debugging via the failure-mode checklist.

**Our take.** Pre-screen our dual-tower design against McCabe's checklist. Cheap, paper-guided audit. Combine with Hsieh 2019's convergence-guarantee design pattern for a belt-and-suspenders stability story. See `literature/mccabe_2023_autoregressive_stability.md`, `literature/hsieh_2019_convergence_guarantees.md`.

---

## 5. Cross-Cutting Themes

These themes recur across architectural families and dominate design decisions.

### 5.1 Boundary conditions: the silent deal-breaker

FNO's periodic-BC assumption is the single most consequential architectural constraint in the entire learned-PDE literature. It is rarely flagged as a limitation in FNO papers themselves — the bias is mathematical (inherent to the FFT), not a hyperparameter — but it disqualifies FNO from any application with non-periodic BCs. Bidomain is firmly non-periodic (Neumann-dominant with mixed Dirichlet/Robin).

**Architectures that handle mixed BCs correctly:**
- **CNO, UGrid, MGCNN**: CNN-based, no spectral assumption, BCs enter via padding or masked iterators.
- **Lan 2023**: spatially-varying kernels explicitly conditioned on BC indicator fields. Most principled of the group.
- **Swin-windowed transformers (Poseidon, PDE-Transformer)**: windowed attention makes no periodicity assumption; boundary windows naturally see boundary features.
- **DeepONet**: BC enters as input function to the branch net.
- **Physics-attention (Transolver)**: BCs are per-mesh-point features.

**Architectures that struggle with mixed BCs:**
- **FNO and FNO-based variants (Lydon PINO, Centofanti FNO)**: structural periodicity assumption.
- **DRIFT-Net**: spectral branch *may* inherit the problem, depending on bandwise weighting specifics — pending PDF verification.

For cardiac bidomain, **mixed-BC correctness is a hard architectural constraint**, not a nice-to-have. A model that produces 5% boundary artifacts per timestep compounds catastrophically across 30K autoregressive steps.

### 5.2 Long-horizon stability

Every autoregressive cardiac NN surrogate paper reports drift or divergence beyond some rollout horizon. The sources are multiple:

1. **Spectral bias** (non-dominant frequencies under-trained): PDE-Refiner addresses directly.
2. **Non-contractive Jacobian** (one-step errors amplify): JAWS addresses.
3. **Amplifying architectural operations** (BN on small feature dims, unbounded activations): McCabe addresses.
4. **BC-induced accumulation** (FNO's periodic artifacts): the only fix is to not use FNO.
5. **Ionic instability** (stiff gate dynamics blow up): CfC addresses, or keep classical TTP06.

Our split-step hybrid design sidesteps (1)–(3) partly by not being fully autoregressive — the classical ionic and parabolic stages are exact, only the elliptic is learned — and handles (4) by architecture choice. Issue (5) is moot because we keep TTP06 classical. But the elliptic solve still appears inside a 30K-step loop (one solve per timestep), so elliptic-stage drift still matters. JAWS + McCabe's architectural audit are the reasonable defaults; PDE-Refiner is the escape hatch.

### 5.3 Convergence guarantees

Three tiers of safety are available:

- **Tier 1 (strongest)**: Krylov-wrapped learned operator. The classical solver converges to the true answer regardless of what the network produces. Examples: Lan 2023, NPO, NOWS, Hsieh 2019.
- **Tier 2 (medium)**: Architecturally-bounded dynamics. The learned operator has provably contractive properties (Hsieh's eigenvalue constraint, CfC's gating). Convergence to a fixed point is guaranteed, though the fixed point's quality depends on training.
- **Tier 3 (empirical only)**: Direct approximation. FNO, DeepONet, CNO, PDE-Transformer, Poseidon. "It works in practice" is the only guarantee.

For clinical deployment — even in a research context — tier 1 is the responsible choice. The net can be arbitrarily wrong and the solver still returns a correct answer; the only consequence of poor net quality is slower convergence. For Bidomain V1's elliptic step, this translates to: a learned preconditioner or warm-start that wraps the existing PCG solver.

### 5.4 Discretization invariance

FNO's resolution invariance (train at one grid, evaluate at any) is its main marketing point. CNO achieves it via explicit anti-aliasing; DeepONet via continuous-coordinate trunk net; GNNs via mesh-agnostic message passing. Purely convolutional architectures without anti-aliasing (plain ResNets, U-Nets) are mesh-sample-dependent and brittle across resolutions.

For cardiac bidomain, resolution invariance is **nice but not essential**. Our typical grid sizes (128–512 per side) are well within any architecture's validated range. Cross-resolution deployment would matter for clinical integration (varying tissue-model resolutions) but is Phase-B work.

### 5.5 Training-data cost

Three cost regimes:

- **Supervised**: Requires pairs (input, ground-truth solution). Generating ground truth requires running a classical solver N times. Cost: O(N × classical solve time). Used by FNO, DeepONet, CNO, PDE-Transformer, Poseidon, DeepONet.
- **Residual loss**: Requires only the operator `A` and RHS `f`, not solutions. Loss is `‖f − Ax‖` directly. Cost: O(N × operator assembly). Much cheaper. Used by Greenfeld, UGrid, NPO, Geng Allen-Cahn.
- **Unsupervised / physics loss**: Pure PDE residual, no data. Cost: O(N × loss evaluation). Cheapest but harder to train stably. Used by PINN-family (out of scope).

For Bidomain V1, residual loss is the preferred regime: we have the operator (discrete anisotropic Laplacian) and can compute residuals directly. Supervised training would require thousands of Bidomain V1 runs to generate labels. Not prohibitive but unnecessary.

### 5.6 Heterogeneous coefficients

Most learned-Poisson benchmarks use scalar or simple-variable coefficients. Bidomain has anisotropic SPD tensor D(x) varying with fiber orientation — a harder class that few existing methods explicitly handle.

- **Best handled**: MGCNN (heterogeneous scalar coefficients explicitly tested), Lan 2023 (spatially-varying kernels can encode D(x) as indicator fields), Geng Allen-Cahn (nonlocal-kernel-as-input-channel pattern).
- **Untested**: UGrid (isotropic Laplacian in paper), FNO (isotropic benchmarks), Poisson CNN (isotropic).
- **Architecturally open**: passing D(x) as input channels is plausible for most CNN-based architectures; validation needed.

The pattern to adopt from Geng Allen-Cahn: pass variable-coefficient fields (D_i, D_e) as input tensor channels alongside V_m. The network learns to condition its operator on these fields without explicit mathematical structure.

---

## 6. Design Implications for Hybrid Bidomain Surrogate

Synthesizing the above produces a concrete design stack.

### 6.1 Deployment pattern (NOWS-style)

```
x₀ = NN(Vm_intermediate, D_i, D_e, BC_mask, stim_mask)      (neural warm-start)
φ_e = Bidomain_V1_PCG(A, b, initial_guess=x₀, tol=1e-6)     (classical refinement)
```

- **Speedup target**: 50–90% PCG iteration reduction (NOWS reports up to 90%).
- **Safety guarantee**: PCG's convergence guarantee is preserved. Bad net → more iterations, never wrong answer.
- **Clinical defense**: Bidomain V1's existing correctness properties unchanged; only speed.

### 6.2 Backbone architecture (dual-tower, preconditioner/warm-start)

Two leading candidates, to be A/B-tested empirically:

**Option A: Dual CNO towers with cross-attention at bottleneck**
```
Vm-tower: CNO encoder → bottleneck → CNO decoder → Vm features
φ_e-tower: CNO encoder → bottleneck → CNO decoder → φ_e guess
Cross-talk: full self-attention at V-cycle bottleneck (smallest feature map);
            1×1 cross-conv at full-resolution levels (cheap, preserves spatial information)
```

**Option B: PDE-Transformer with channels-as-tokens**
```
Per-channel tokenization: {Vm, φ_e, D_i_xx, D_i_xy, D_i_yy, D_e_xx, …, BC_mask}
Spatial attention: shifted-window (Swin) within each channel
Channel attention: across channels at every layer
Output: φ_e field
```

### 6.3 Input conditioning (nonlocal-kernel-as-channel pattern)

From Geng Allen-Cahn: pass variable-coefficient fields as input channels:

```
Inputs = [Vm_intermediate,              # driving signal
          D_i_xx, D_i_xy, D_i_yy,        # intracellular conductivity tensor
          D_e_xx, D_e_xy, D_e_yy,        # extracellular conductivity tensor
          BC_type_mask,                  # 0=interior, 1=Neumann, 2=Dirichlet
          BC_value_mask,                 # prescribed value where BC_type ≠ 0
          previous_phi_e]                # optional: warm-start from previous timestep
```

The network learns to condition its operator on these fields without explicit mathematical encoding.

### 6.4 Training loss (residual-based, unsupervised)

```
L = ‖(1 − M_BC)(f − A_discrete NN_output)‖²                (interior residual)
  + ‖M_BC NN_output − BC_values‖²                          (boundary enforcement)
  + λ_rest ‖NN(Vm=V_rest, D_rest)‖²                        (rest-attractor regularizer)
  + λ_jaws L_JAWS                                          (optional: Jacobian regularization)
```

No ground-truth φ_e required. Operator `A_discrete` is the standard FEM/FD discretization; residual is cheap.

### 6.5 Validation strategy

Three tiers:

1. **Direct Bidomain V1 reference**: Run the hybrid surrogate on held-out tissue configurations, compare φ_e field L² error, iteration count, wall-clock.
2. **APEBench external validation**: Deploy architecture on Fisher-KPP, Gray-Scott, Allen-Cahn benchmarks; verify cross-domain accuracy on reaction-diffusion adjacent problems.
3. **Niederer 2011 N-version benchmark**: Final-product validation on CV (conduction velocity) and APD (action potential duration) metrics against the canonical cardiac simulator benchmark.

Success criteria:
- CV within 5% of Bidomain V1 ground truth
- APD within 5 ms
- φ_e L² relative error < 1%
- Total wall-clock speedup > 3× (conservative, given 94% elliptic → < 30% residual + net overhead)
- Kleber boundary-layer effect reproduced qualitatively

### 6.6 What we are explicitly NOT doing

- **Not replacing TTP06**: Benchmark shows compiled TTP06 beats our 8k-param neural ionic by 8× on GPU. Wrong fight.
- **Not using FNO-family backbones**: Periodic-BC assumption is structural, not fixable without replacing the Fourier layer.
- **Not learning the full inverse φ_e = NN(Vm, D, BC)**: Error compounds across autoregressive timesteps. Preconditioner or warm-start only.
- **Not using full O(N²) self-attention**: Memory-prohibitive at cardiac grid sizes. Windowed or factorized.
- **Not doing hyperbolic bidomain now**: Deferred to future Phase B. Requires a hyperbolic simulator we don't have.

---

## 7. Open Problems and Future Directions

### 7.1 Anisotropic-tensor Poisson

No learned-Poisson paper currently handles full 2×2 SPD tensor coefficients cleanly. Lan 2023's spatially-varying kernels are the most plausible route (pass tensor components as input channels) but untested. A successful bidomain surrogate will be the first to handle this explicitly.

### 7.2 Hyperbolic bidomain

The Maxwell-Cattaneo hyperbolic bidomain (second time derivative, finite propagation speed) is more physically correct than parabolic-elliptic and would play to NN strengths (no CFL constraint on learned effective dt). But no hyperbolic cardiac simulator exists in the project, and TTP06's coupling to a hyperbolic V_m is nontrivial. Phase B work.

### 7.3 Geometric variability (patient-specific meshes)

Extension from structured Cartesian grids to unstructured cardiac meshes. Transolver (physics-attention slices) and Salvador's branched LNM are the leading candidates. Would require re-architecting the entire surrogate; Phase B.

### 7.4 Foundation-model scaling

Poseidon's pretraining paradigm (train on many PDEs, fine-tune per task) could apply to "one backbone for TTP06, ORd, Mitchell-Schaeffer ionic models, across multiple tissue configurations." Requires substantial pretraining compute. Deferred unless we partner with a foundation-model effort.

### 7.5 Cardiac contribution to PDE benchmarks

APEBench currently has 46 PDEs but no cardiac. Contributing bidomain as a benchmark PDE would extend the benchmark's value and anchor our method publicly.

---

## 8. Summary Table of Reviewed Works

| Family | Paper | Year | BC handling | Our verdict |
|--------|-------|------|-------------|-------------|
| **Cardiac whole-PDE (monodomain)** | Morier AGATA | 2025 | Unstructured-mesh GNN, BC implicit | Wrong grid class for Phase A |
| | Salvador LFLDNet | 2025 | Fourier coord encoding | Ionic-side precedent, not elliptic |
| | Lydon PINO | 2025 | FNO backbone | **Periodic-BC disqualified** |
| | Centofanti FNO/KOL | 2025 | FNO backbone, single-shot | **Periodic-BC disqualified** |
| | Salvador LNODE | 2024 | 0-D, no spatial BC | Wrong output space |
| | Salvador branched LNM | 2025 | Atlas-based geometry | Phase B reference |
| | Ogbomo-Harmitt ECG forward | 2025 | Skips φ_e field | Confirms the gap we're filling |
| **Generic learned elliptic** | UGrid | 2024 | Masked iterator, Dirichlet demonstrated | **Primary elliptic candidate** |
| | MGCNN | 2023 | Not specified in abstract | Alternative, needs BC verification |
| | Greenfeld multigrid | 2019 | Via operator input | Foundational, cite |
| | Ozbay Poisson CNN | 2019 | Dirichlet superposition | Prototype baseline |
| | Lan mixed-BC | 2023 | **Spatially-varying kernels, Dirichlet+Neumann** | **Direct adoption for preconditioner** |
| | Hsieh convergence | 2019 | Via operator | Cite for safety philosophy |
| **Neural operators** | FNO | 2021 | **Periodic only (structural)** | **Disqualified** |
| | DeepONet | 2021 | Via branch input | Wrong fit (dense grid inefficient) |
| | CNO | 2023 | CNN-based, BC-friendly | **Primary backbone candidate** |
| **Transformers** | PDE-Transformer | 2025 | Shifted-window, BC-agnostic | **Primary backbone candidate** |
| | Poseidon | 2024 | scOT Swin-attention | Pretraining source |
| | Transolver | 2024 | Physics-attention slices | Phase B (unstructured) |
| | DRIFT-Net | 2026 | Dual-branch, spectral branch uncertain | Needs BC verification |
| **Deployment** | NOWS | 2025 | Inherits from classical solver | **Deployment pattern** |
| | NPO | 2025 | Multi-BC preconditioner | Alternative preconditioner |
| **Stability** | PDE-Refiner | 2023 | Inherits backbone | Reserve for Phase 2 |
| | JAWS | 2026 | Inherits backbone | Training-time safety net |
| | McCabe AR stability | 2023 | Architectural, BC-agnostic | Pre-screen checklist |
| **Benchmark** | APEBench | 2024 | Varies per PDE | External validation |
| **Sharp-front RD** | Geng Allen-Cahn | 2024 | Fourier collocation (check) | **Nonlocal-channel pattern adopted** |

---

## 9. Conclusion

The neural-PDE surrogate design space for cardiac bidomain electrophysiology has matured to the point where a defensible architecture choice can be made from published evidence alone. The cardiac-specific literature is narrow (monodomain whole-PDE replacement dominates) and contains no published work on bidomain elliptic surrogates. The cross-domain learned-Poisson and reaction-diffusion literature is deep and directly applicable after adaptation for anisotropic tensor coefficients and Neumann-dominant mixed BCs. Four cross-cutting constraints — BC correctness, long-horizon stability, convergence guarantees, heterogeneous coefficient handling — eliminate FNO-family backbones and argue for a CNN/transformer hybrid with classical warm-start wrapping. The concrete adopted stack (dual CNO or PDE-Transformer towers, nonlocal-channel conditioning, residual-loss training, JAWS stability regularization, NOWS deployment) follows from these constraints and exploits the largest unaddressed gap in the published literature: bidomain elliptic as a learnable sub-operator. Validation against APEBench, the Niederer 2011 N-version benchmark, and direct Bidomain V1 ground truth establishes publishability. Phase B extensions (hyperbolic bidomain, unstructured meshes, foundation-model scaling) are preserved as optional directions but not required for the primary scientific contribution: the first learned bidomain elliptic solver, demonstrating that the 94%-wall-time bottleneck in cardiac bidomain simulation can be accelerated by ≥3× with neural warm-start while preserving classical correctness guarantees.

---

*Document version 1.0, 2026-04-22. Synthesized from 27 papers in `literature/`. For individual paper details see the corresponding `literature/<slug>.md` files. For project status and design decisions-in-flight see `IDEALOG.md`. For data specifications see `DATA_V2_SPEC.md`.*
