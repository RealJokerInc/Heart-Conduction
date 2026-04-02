---
paper: nietoramos_2023_bayesian_hmc
title: "Bayesian inference for fitting cardiac models to experiments: estimating parameter distributions using Hamiltonian Monte Carlo and approximate Bayesian computation"
authors: "Nieto Ramos A, Cherry EM, Fenton FH"
year: 2023
journal: "Medical & Biological Engineering & Computing"
doi: "10.1007/s11517-022-02685-y"
pmid: "36322242"
pdf: ../papers/bayesian_hmc_abc_cardiac_fitting_2023_nietoramos.pdf
questions: [Q8]
---

## Key Findings
- Both **HMC (Hamiltonian Monte Carlo via NUTS/Stan)** and **ABC-SMC (Approximate Bayesian Computation)** produce well-shaped posterior distributions for cardiac AP model parameters
- HMC produces narrower (more precise) distributions than ABC-SMC for synthetic data
- Multi-CL (cycle length) data including near-alternans dynamics constrains parameters far better than single-CL fitting
- Strong bivariate parameter correlations reveal compensatory mechanisms (e.g., tau_in/tau_out correlation in Mitchell-Schaeffer)
- Some Fenton-Karma parameters (tau_d, tau_0, tau_si) are structurally non-identifiable from limited CL data
- Estimated noise σ serves as a diagnostic: if σ >> expected measurement noise, the model structure is inadequate

## Method
- **HMC via NUTS** (No-U-Turn Sampler) implemented in Stan
- **ABC-SMC**: Sequential Monte Carlo with approximate likelihood
- **Models tested**: Mitchell-Schaeffer (5 params), Fenton-Karma (13 params)
- **Training data**: Synthetic (perturbed true params + Gaussian noise σ=0.03) and experimental (zebrafish AP recordings)
- **Protocol**: 3 cycle lengths near alternans bifurcation (e.g., 400, 350, 310 ms), last 2 APs of 6 per CL
- **Priors**: Folded normal centered at true values, σ = 30% of true value
- **Sample size**: 500 posterior samples, 1000 warm-up

## Key Equations / Results
- Gaussian likelihood: p(data | params, σ) ∝ exp(-Σ(V_model - V_data)² / 2σ²)
- Synthetic MS: posterior mode errors < 9% for all 5 parameters
- Synthetic FK: voltage trace error < 0.6%
- Experimental FK: voltage trace error ~3.4%
- Estimated σ for experimental data ~10× larger than synthetic → model inadequacy diagnostic
- Forward Euler with adaptive dt (0.1 ms first 4 ms, 0.5 ms otherwise)

## Connections to Our Models

### Relevant Engine Components
- **Monodomain V5.4**: Our Rush-Larsen ionic solver could be wrapped in a differentiable framework (PyTorch autograd) to enable HMC
- **Ionic models**: `cardiac_sim/ionic/ttp06/model.py` and `cardiac_sim/ionic/ord/model.py` — these would need to be differentiable for HMC
- Our TTP06/ORd are already PyTorch-based, which is advantageous for gradient computation

### Agreements
- Confirms that multi-rate pacing is essential for constraining ionic parameters — aligns with Groenendaal 2015
- The noise diagnostic concept is useful for validating our model fits

### Disagreements or Gaps
- Tested on 5-13 parameter models, NOT on TTP06 (17+ conductances) or ORd (40+ state variables)
- TTP06/ORd have stiff ODEs with Rush-Larsen integrator — getting gradients through this is non-trivial
- Stan requires explicit model code; our PyTorch models would need a different HMC implementation (e.g., NumPyro, Pyro, or custom)

### Actionable Insights
- **HIGH**: Implement HMC for TTP06 conductance tuning using PyTorch's autograd. Our ionic models are already in PyTorch — wrap them with `torch.autograd` and use a NUTS sampler (e.g., from NumPyro or Pyro)
- **HIGH**: Use 3+ cycle lengths near alternans threshold as fitting targets — this maximally constrains rate-dependent parameters
- **MEDIUM**: The noise σ diagnostic should be standard: if estimated σ >> 0.01 (our simulation noise), the model needs structural changes, not just parameter tuning
- **MEDIUM**: Start with MS or FK models to validate the pipeline before scaling to TTP06

## Limitations / Caveats
- Scaling HMC to TTP06 (17+ params) or ORd is unvalidated — may require more sophisticated priors and longer chains
- Gradient computation through stiff ODE solvers (Rush-Larsen) may have numerical issues
- Folded normal priors centered at "true" values assume prior knowledge — in practice, priors would be less informative
