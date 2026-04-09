---
paper: salvador_2025_lfldnet
title: "Liquid Fourier Latent Dynamics Networks for fast GPU-based numerical simulations in computational cardiology"
authors: "Salvador M, Marsden AL"
year: 2025
journal: "Computers in Biology and Medicine"
doi: "10.1016/j.compbiomed.2025.111355"
pmid: "41338030"
pmcid: "PMC12691700"
pdf: ../papers/liquid_fourier_latent_dynamics_networks_for_fast_gpu_based_numerical_simulations_in_computational_cardiology.pdf
questions: [surrogate_pipeline]
---

## Key Findings

- LFLDNets achieve 30x speedup over high-fidelity monodomain EP simulation (3 min vs 1.5 hr on 24 CPU cores) on a 3D pediatric biventricular geometry with ~240K DOFs, using the ten Tusscher-Panfilov ionic model.
- The surrogate operates at dt=10ms — two orders of magnitude larger than the FEM requirement (~0.1ms) — by replacing the ODE integrator with a closed-form continuous (CfC) liquid neural network that is stable at arbitrary timesteps without CFL constraints.
- Latent dynamics are encoded in a Closed-form Continuous Neural Circuit Policy (CfC NCP): a sparse, neurologically-inspired architecture that propagates a low-dimensional latent state s(t) through gated layers, avoiding all ODE solver calls at inference.
- A Fourier embedding with a trainable kernel B maps spatial coordinates to (cos(2πBx), sin(2πBx)) before the reconstruction network, enabling learning of high-frequency spatial patterns (fiber orientations, activation wavefronts) faster and more accurately than raw coordinate input.
- Generalization to 7-parameter family (Ca/Na/K conductances, tissue diffusivities, stimulus timing) with val MSE 3.15e-3 normalized, capturing qualitative bifurcations between healthy sinus rhythm and bundle branch block.

## Method

**Architecture:** Two-network design.
1. **Dynamics network (CfC NCP):** Evolves latent state s(t) ∈ ℝ^d at irregular, large timesteps. No numerical ODE solver. Gating functions f, g, h with sigmoid activations propagate state in closed form given elapsed time Δt and parametric inputs μ. Sparse wiring (NCP) limits parameter count and improves stability.
2. **Reconstruction network:** Fully-connected MLP mapping [s(t), x_fourier, μ] → u(x,t). Takes Fourier-embedded spatial coordinate x_fourier = [cos(2πBx), sin(2πBx)] with learned B.

**Training:**
- MSE loss with no regularization terms.
- Adam optimizer, single precision, up to 10,000 epochs.
- Hyperparameter search: Tree-structured Parzen Estimator Bayesian optimization over 20 trials.
- Hardware: single Nvidia A40 GPU.
- Spatial subsampling: 1,000 random spatial points per epoch.
- Dataset: 150 monodomain EP simulations (100 train / 50 val), 7 varied parameters.
- Model size: ~715K parameters for EP surrogate.

**Cardiac test case:** 3D biventricular geometry (pediatric HLHS patient), monodomain + ten Tusscher-Panfilov ionic model, anisotropic conductivity + Purkinje tree. Mesh: 1,016,192 tetrahedral cells, 240,555 DOFs.

## Key Equations / Results

**CfC dynamics (replacing ODE integration):**

The standard neural ODE `ds/dt = f_NN(s, t, μ)` is replaced by a closed-form layer propagation:

```
s(t + Δt) = g(s, μ, Δt) ⊙ h(s, μ, Δt) + (1 - g(s, μ, Δt)) ⊙ f(s, μ, Δt)
```

where f, g, h are feedforward networks and ⊙ is elementwise multiplication. This is evaluated directly without any numerical integration step. No adaptive solver, no step-size control, no CFL condition.

**Speedup numbers:**
- Cardiac EP: 3 min (LFLDNet, 1 GPU) vs 1.5 hr (FEM, 24 CPU cores) → ~30x (wall-clock, different hardware basis).
- CFD hemodynamics: 45 min (LFLDNet) vs 3-4 hr (FEM, 128 CPU cores) → ~4-5x.
- Inference dt: 10ms (EP) vs 0.1ms (FEM) → 100x fewer time steps in the latent.

**Accuracy:**
- EP: training normalized MSE = 9.12e-4, validation = 3.15e-3.
- CFD: training = 3.32e-4, validation = 5.25e-4.

## Why They Moved Away From Neural ODEs

Standard neural ODEs (using feedforward fully-connected networks as the dynamics function) have three concrete failure modes for cardiac EP:

1. **Require a numerical ODE solver at inference time.** Explicit solvers (RK4, etc.) impose step-size constraints tied to the stiffness of the system. For cardiac ionic models, stiffness forces dt ~ 0.01-0.1ms, giving O(10K-30K) solver calls per second of simulation — exactly the same bottleneck as the physics code.
2. **Vanishing/exploding gradients during training.** Long rollouts through a stiff, feedforward dynamics function accumulate gradient pathologies. Standard neural ODEs provide no architectural protection against this.
3. **Require larger networks for equivalent accuracy.** Feedforward networks lack the inherent bounded dynamics that liquid/CfC networks provide through their gating structure, necessitating more parameters to achieve the same expressivity.

**How CfC liquid NNs address stiffness:** The CfC formulation provides an explicit closed-form solution for s(t + Δt) given s(t) and Δt as an input — not as a numerical integration step count. The gating mechanism (g in the equation above) acts like a learned "how much to update" signal that naturally saturates at large Δt, providing bounded dynamics for any Δt. No CFL condition, no stability analysis, no adaptive step-size controller.

## Connections to Our Models

### Relevant Engine Components

**IonicSurrogateV3 (`Surrogate/surrogate/model/`):** Our surrogate solves per-cell ionic dynamics — an ODE system for carried_state (36 dims: 32 ionic latent + 4 concentrations). The v3 cross-attention + MLP was originally designed for discrete autoregressive stepping at dt=0.01ms. A4 failed because 30K autoregressive steps accumulate prediction errors faster than the model corrects them. LFLDNet's CfC dynamics function is the direct architectural answer to this failure mode.

**Stage 1 (n×1 cross-attention + MLP):** The v3 attention block (36 dims attend to [Vm, dt]) is conceptually similar to the CfC gating — both compute "how much to update each latent dim given elapsed time and input." The key difference: our attention block is trained as a discrete-step recurrence, while CfC is trained to accept Δt as a continuous input and produce s(t+Δt) directly. Retrofitting Δt-conditioning into our existing attention head is architecturally feasible.

### Agreements

- **Latent state approach:** Both encode dynamics in a low-dimensional latent vector rather than explicitly simulating all ODE states. Their s(t) is our carried_state.
- **Decoupled dynamics + readout:** Their dynamics network (latent evolution) + reconstruction network (spatial output) mirrors our Stage 1 (state evolution) + Stage 2 (current readout) split.
- **Parameterization:** They condition on parameter vectors μ (conductances, diffusivities); we plan the same (model ID token, per-cell conductance parameters).
- **ten Tusscher-Panfilov ionic model:** Directly relevant — this is TTP06, the same ionic model our surrogate currently targets.

### Disagreements or Gaps

- **Output space:** LFLDNets output global spatio-temporal fields u(x,t) via the reconstruction network — they learn a spatial surrogate over the full 3D mesh. We are learning per-cell ionic dynamics (IonicSurrogateV3) and a separate spatial diffusion surrogate (Cross-Skip ResNet). Their reconstruction network is not applicable to our ionic component at all.
- **Scale of spatial task:** Their reconstruction MLP takes spatial coordinates x and latent s(t) to produce Vm(x,t) at arbitrary query points. Our diffusion ResNet is convolutional — it needs to resolve wavefront structure. The coordinate-based approach would require a NeRF-style or FNO approach for our diffusion step, not a direct substitute.
- **CfC addresses temporal stiffness, not error accumulation per se:** Our A4 failure was fundamentally latent instability from recurrence — CfC helps by providing bounded dynamics and Δt-as-input, but does not remove recurrence. For 300ms trajectories at dt=10ms that is only 30 latent steps, which is tractable. At dt=0.01ms it is still 30K steps.
- **Their stiffness solution operates at the ODE solver level:** CfC eliminates the solver calls by replacing them with a closed-form expression. This is most valuable when the solver is the bottleneck (their case: FEM EP solver). In our case, the discrete error accumulation is the bottleneck, which is a training pathology — CfC's bounded dynamics help but are not a complete fix without also increasing Δt dramatically.
- **No per-cell heterogeneity in their spatial model:** Their reconstruction network maps coordinates to output — it implicitly assumes smooth spatial variation. Our ionic surrogate must handle per-cell state trajectories for heterogeneous tissue (infarct regions, APD gradients).

### Actionable Insights

1. **Adopt CfC-style Δt-conditioning for the Neural ODE pivot (Priority: HIGH).** Replace our discrete-step attention recurrence with a CfC-inspired gating: accept Δt as an explicit input, compute gating coefficients g(s, Vm, Δt), and produce s(t+Δt) as a convex-combination update. This directly addresses A4's error accumulation by enabling large Δt (e.g., 0.1ms or 1ms latent steps even if ground truth is 0.01ms).
2. **Use Fourier embedding for the spatial diffusion component (Priority: MEDIUM).** If we later explore coordinate-based spatial surrogates (NeRF/INR-style), the trainable Fourier kernel B is straightforwardly applicable to encoding fiber orientation or geometry into the reconstruction network.
3. **Latent dimensionality:** Their EP surrogate uses a latent of unspecified but likely small dimension (their model is ~715K params total). Our 36-dim carried_state is in the same ballpark and likely sufficient.
4. **Training data budget:** 100 simulations was sufficient for their 7-parameter EP family. For our ionic surrogate (single-cell, parameterized by cell type), this is very achievable.

**Priority: HIGH** — the CfC temporal dynamics pattern directly addresses our pivot from discrete autoregressive stepping to Neural ODE formulation.

## Limitations / Caveats

- **Global latent, not per-cell:** Their s(t) encodes the entire spatial field at once, relying on the reconstruction network for spatial resolution. This does not scale to per-cell heterogeneous ionic dynamics (each cell has its own trajectory). Our per-cell IonicSurrogateV3 architecture cannot be replaced by their approach directly.
- **Monodomain only:** Their cardiac case uses the monodomain equation. No bidomain, no phi_e. Their surrogate does not target the elliptic solve (94% of bidomain wall time) — the Cross-Skip ResNet remains our approach for that.
- **30x speedup vs different hardware baseline:** FEM ran on 24 CPU cores; LFLDNet on 1 GPU. GPU-vs-CPU speedup is mixed in. GPU-to-GPU comparison would likely show a smaller ratio.
- **3.15e-3 normalized val MSE is not negligible:** For quantitative EP (APD, CV, restitution), this error level may be unacceptable depending on normalization range. The paper does not report APD error or CV error explicitly.
- **Small dataset (100 training sims):** Generalization may be limited for out-of-distribution parameters. Their 7-parameter space is relatively low-dimensional.
- **Reconstruction network is not a PDE solver:** No boundary conditions, no conservation laws enforced. Physics consistency relies entirely on training data coverage.

---

*Retrieved from PubMed (PMID: [41338030](https://doi.org/10.1016/j.compbiomed.2025.111355)). Full text via arXiv preprint [2408.09818](https://arxiv.org/abs/2408.09818).*
