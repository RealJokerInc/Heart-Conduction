---
paper: li_2020_fno
title: "Fourier Neural Operator for Parametric Partial Differential Equations"
authors: "Li Z, Kovachki N, Azizzadenesheli K, Liu B, Bhattacharya K, Stuart A, Anandkumar A"
year: 2021
journal: "ICLR 2021"
doi: "arxiv:2010.08895"
pmid: ""
pdf: ../papers/li_2020_fno.pdf
questions: [surrogate_pipeline]
---

## Key Findings
- **FNO parameterizes the integral kernel in Fourier space** — each Fourier layer does `K(x) = F⁻¹(R_θ · F(x))` where R_θ is a learned complex-valued tensor on truncated Fourier modes.
- **Up to three orders of magnitude faster** than classical PDE solvers on parametric Burgers, Darcy flow, Navier-Stokes. **First neural operator to successfully model turbulence with zero-shot super-resolution.**
- **Resolution-invariant training**: a network trained at one grid resolution evaluates at any other resolution without retraining. Ostensibly architecture-level generalization.
- **CRITICAL LIMITATION (implicit, unadvertised)**: **FNO's Fourier layer assumes periodic boundary conditions.** The FFT operation used inside the Fourier layer is mathematically a DFT, which assumes periodicity. For non-periodic PDEs (Dirichlet, Neumann, mixed BCs), FNO either requires artificial padding/periodization tricks or produces systematic artifacts near boundaries.
- **Reduced-order + resolution-invariant** — dramatically fewer parameters than U-Net-style solvers because the network operates in low-mode Fourier space.

## Method
- **Fourier layer**: input `v(x)` → FFT → keep only lowest `k` modes → multiply by learned `R_θ ∈ ℂ^{k × d × d}` (complex tensor, `d` channels) → IFFT → result. Skip connection with a pointwise linear layer adds high-frequency contributions directly in physical space.
- **Architecture**: typically 4 Fourier layers with GELU activations. Plus input/output linear lifts for channel dimension.
- **Training**: supervised L² loss on ground-truth PDE solutions. Needs a classical solver to produce labels.
- **PDE families**: 1D Burgers, 2D Darcy flow, 2D Navier-Stokes (incompressible). All have either periodic BCs (Navier-Stokes in the paper) or mostly-smooth Dirichlet BCs where periodic extension is a mild approximation.
- **Resolution invariance**: because Fourier modes are continuous-space objects (the grid is just a sampling), the same `R_θ` tensor applies at any resolution. Truncation to `k` modes gives discretization invariance as long as `k` captures the relevant physics.

## Connections to Our Models

### Relevant Engine Components
**FNO is a NON-STARTER for our bidomain elliptic surrogate as written.** Bidomain has Neumann-dominant mixed BCs (zero-flux at tissue-bath, sometimes Dirichlet pinning, sometimes Robin). FNO's periodic-BC assumption produces systematic boundary artifacts that would compound across autoregressive rollout steps.

Read for the ambient context of "how operator learning works," not as a direct adoption target. Architectural alternatives (UGrid, Lan 2023, DeepONet) handle BCs correctly.

### Agreements
- **Operator-learning paradigm**: learning a family of PDE solutions parameterized by inputs (coefficients, RHS, BCs) is directionally what we want for a bidomain surrogate that generalizes across tissue configs. FNO is the seminal paper here.
- **Resolution invariance**: if achievable, extremely valuable for clinical deployment across mesh sizes. Though: FNO's resolution invariance is subtly brittle — see limitations.
- **Spectral bias is real**: training pressure on low-frequency modes is a feature, not a bug, for smooth-fields (φ_e is smoother than V_m after elliptic filtering). For sharp V_m wavefronts, spectral bias is a liability.

### Disagreements or Gaps
- **Periodic-BC assumption is the deal-breaker.** Bidomain has:
  - Zero-flux Neumann at tissue-bath interface (tissue insulated from external bath except at grounding)
  - Dirichlet pinning at grounding electrodes
  - Mixed/Robin at certain anatomical boundaries
  None are periodic. Padding-with-periodization (a common workaround) introduces ~O(grid-width) error at boundaries. In a 30K-step autoregressive rollout, these errors compound into wavefront-position drift, APD errors, and potentially dispersion.
- **Single-field operator**. FNO maps one function to one function. Bidomain elliptic is "given V_m, solve for φ_e" — single-field from the elliptic perspective — so FNO is structurally OK for the one-shot problem, but the autoregressive loop still fails for BC reasons.
- **Truncation to low modes fails for sharp wavefronts.** V_m has a ~1 mm upstroke over a 2 cm domain — narrow spatial support, high-frequency content. FNO's mode truncation would smear the upstroke unless `k` is very large, which defeats the parameter-efficiency argument.
- **No convergence guarantees.** Unlike Hsieh 2019 / Lan 2023, FNO is a direct inverse approximation. Errors do not fall back to a PCG loop. For our use case where phi_e feeds V_m feeds phi_e, this is structurally risky.
- **Anisotropic tensor**: unclear whether mode-by-mode multiplication in Fourier space generalizes to tensor-anisotropic operators. Paper's examples are isotropic.

### Actionable Insights
- **HIGH — Do NOT build primary architecture on FNO.** The periodic-BC assumption is structural, not a hyperparameter. No amount of engineering fixes it without replacing the Fourier layer.
- **HIGH — When reading Lydon 2025 PINO and Centofanti 2025 (both FNO-backbone cardiac papers), apply this same BC critique.** Their accuracy on monodomain may partly reflect the monodomain PDE's more benign BC structure; their approach does not obviously extend to bidomain.
- **MEDIUM — If tempted to use FNO as a component**: consider it only for subproblems where periodicity is genuinely acceptable, e.g., a periodic tissue patch used as a test domain. Not production.
- **MEDIUM — Spectral bias as a diagnostic tool.** FNO's mode truncation gives a clean way to measure "how much high-frequency content does our field need?" If we can reproduce Bidomain V1's φ_e with ≤16 Fourier modes, we know the elliptic response is smooth and a low-mode CNN suffices. If not, we need full receptive field (UGrid / V-cycle).
- **LOW — Neural-Operator textbook framing (Kovachki 2021)**: FNO inspired a whole family of neural operators. Understanding FNO helps parse the zoo.

## Limitations / Caveats
- **PERIODIC BC ASSUMPTION.** Cannot be overstated for our use case. Papers in the FNO line that claim BC generalization typically rely on either (a) padded/periodic extensions (boundary artifacts), (b) BC as an input channel (weak, doesn't fix the Fourier-layer math), or (c) constrained problem classes where BCs are implicit.
- **Resolution invariance caveat**: mode truncation `k` is fixed at training time. Moving to higher resolution works; moving to resolution where previously-negligible modes become relevant produces errors.
- **Supervised training** — requires classical solver runs for labels. Same data-cost issue as DeepONet.
- **No iterative-refinement structure.** Single forward pass, no PCG fallback. Errors don't self-correct.
- **High-frequency content gets squashed.** For our sharp-wavefront V_m dynamics, this is a real concern. Bidomain is less affected for φ_e (smoother field after elliptic filtering), more affected for V_m.
- **2021 paper, pre-dates most architectural fixes.** Later FNO variants (FFNO, F-FNO, GINO) address some limitations. But the fundamental periodic-BC issue remains in the core Fourier-layer construction.
