---
paper: lippe_2023_pde_refiner
title: "PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers"
authors: "Lippe P, Veeling BS, Perdikaris P, Turner RE, Brandstetter J"
year: 2023
journal: "NeurIPS 2023"
doi: "arxiv:2308.05732"
pmid: ""
pdf: ../papers/lippe_2023_pde_refiner.pdf
questions: [surrogate_pipeline]
---

## Key Findings
- **Identifies the root cause of autoregressive PDE-surrogate failure**: standard MSE training neglects non-dominant spatial frequencies. The model learns the dominant (large-scale) structure well, but fails to capture high-frequency detail — and these small errors compound catastrophically during long rollouts.
- **Diffusion-style multistep refinement as the fix**: at each time step, the initial prediction is refined through a short denoising chain. The denoising objective inherently forces the model to learn all frequency components (the spectral content of Gaussian noise is uniform).
- **Outperforms both neural and hybrid neural-numerical baselines** on fluid-dynamics benchmarks, specifically for long rollouts where alternatives diverge.
- **Architecture-agnostic**: the refinement wrapper can sit on top of any base neural PDE solver (FNO, U-Net, transformer). **This means PDE-Refiner inherits the BC limitations of whatever backbone it uses** — e.g., FNO's periodic-BC assumption is not fixed by PDE-Refiner.
- **Denoising provides spectral data augmentation for free** — the noise-injection step ensures all frequencies see gradient signal during training, with no additional loss terms.

## Method
- **Core mechanism**: at each rollout step, instead of predicting `u_{t+1}` directly from `u_t`, the model:
  1. Produces an initial prediction `u_{t+1}^{(0)}`.
  2. Refines via `K` denoising steps: `u_{t+1}^{(k+1)} = u_{t+1}^{(k)} + NN_denoise(u_{t+1}^{(k)}, k, u_t)` where the denoiser is trained to remove Gaussian noise injected at each step.
  3. Final prediction `u_{t+1}^{(K)}` becomes the rollout output.
- **Training**: noise levels sampled from a schedule (like DDPM). At each training step, noise is added to the target, the model predicts the noise, loss is MSE on noise prediction. Matches diffusion-model training exactly.
- **Inference cost**: K forward passes per time step instead of 1. Typical K = 3–10.
- **Base architecture**: tested with FNO and U-Net backbones. Both benefit from the refinement wrapper.
- **Training data**: synthetic trajectories from classical PDE solvers (Kolmogorov flow, Navier-Stokes).

## Connections to Our Models

### Relevant Engine Components
**Most relevant for the dual-tower bidomain rollout** if we see long-horizon error accumulation in V_m and/or φ_e. The bidomain integration is 30K steps at dt=0.01ms — far longer than the rollouts demonstrated in the paper, so the failure mode it addresses is a real risk for us.

### Agreements
- **Autoregressive long-horizon failure is the primary risk** for the dual-tower design. PDE-Refiner explicitly targets this failure mode and provides a principled fix.
- **Architecture-agnostic wrapper**: compatible with CNN-V-cycle towers (UGrid-per-domain), plain CNN encoders, or Swin-attention backbones. We can adopt it regardless of tower architecture choice.
- **Unsupervised denoising loss**: no additional ground-truth required beyond what we already plan (V_m, φ_e from Bidomain V1).

### Disagreements or Gaps
- **Inference cost × K.** Each time step becomes K forward passes. For a 30K-step rollout with K=5, that's 150K forward passes vs 30K without refinement. Speedup over classical PCG may evaporate. **Must benchmark carefully.**
- **Demonstrated on fluid dynamics, not cardiac.** The spectral content of cardiac EP fields (sharp wavefronts with relatively narrow spectrum) may differ significantly from turbulence benchmarks (broad spectrum). The "non-dominant frequencies matter" argument may be weaker or stronger for us; empirical question.
- **Does NOT fix BC limitations of backbones.** If we used FNO as the base architecture, PDE-Refiner still inherits the periodic-BC assumption. **This matters** because the paper's headline wins on fluid benchmarks may come partly from refining FNO's BC-induced errors; we can't rely on it to fix a structurally-wrong backbone. Stick with CNN/U-Net backbones where BCs are correctly modeled.
- **Inherits single-field assumption.** The paper is scalar PDEs (Kolmogorov flow, vorticity). Our bidomain has two coupled fields (V_m + φ_e). Refinement on the joint field is architecturally fine but not demonstrated.

### Actionable Insights
- **MEDIUM — Reserve for Phase-2 stability work.** First, get the dual-tower surrogate working at short rollout horizons (say 300ms / 30K steps of our bidomain rollout). If we see drift or wavefront dispersion, add PDE-Refiner as a wrapper.
- **MEDIUM — Benchmark inference-cost tradeoff.** If K=3 is enough to fix drift, the 3× inference cost is likely worth it (PCG+GMG is still more expensive). If K=10 needed, reconsider.
- **HIGH — Do NOT combine with FNO backbone.** FNO's periodic-BC failure is a structural problem PDE-Refiner cannot fix. For bidomain with Neumann BCs, we need a backbone that handles BCs correctly (CNN with explicit BC layers, UGrid-style masked iterator, or Lan-2023-style spatially-varying kernels).
- **LOW — Full diffusion-model vs few-step refinement.** PDE-Refiner uses K << typical diffusion step counts (1000). Worth exploring whether even fewer steps suffice for cardiac EP specifically, where the signal is more structured than turbulence.

## Limitations / Caveats
- **K× inference cost** is the primary deployment penalty. Headline speedup numbers in the paper are on a per-step basis; the total wall-time story depends on K.
- **No BC-awareness built in.** If the base architecture has BC limitations (FNO-periodic), PDE-Refiner cannot fix them. It can only refine predictions consistent with the base architecture's biases.
- **Denoising objective requires careful noise scheduling.** Too much noise → model learns trivial denoising; too little → limited spectral augmentation benefit. Hyperparameter-sensitive.
- **Demonstrated on fluid benchmarks only.** Cardiac EP characteristics (wavefront sharpness, operator splitting, coupled fields) differ enough that empirical validation is necessary before relying on the method.
- **Accumulates K cycles per rollout step** — if one cycle produces errors, K cycles may amplify rather than correct. The paper's stability proofs rely on the denoising objective's contraction properties, which hold for well-trained networks but may fail under distribution shift.
- **Not a fundamental fix for model capacity issues.** If the base model can't represent the physics (e.g., missing sharp wavefronts due to resolution), refinement won't add capacity.
