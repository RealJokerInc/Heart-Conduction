---
paper: nietoramos_2022_hmc_cinc
title: "Quantifying Distributions of Parameters for Cardiac Action Potential Models Using the Hamiltonian Monte Carlo Method"
authors: "Nieto Ramos A, Cherry EM, Fenton FH"
year: 2022
journal: "Computing in Cardiology (CinC)"
doi: "10.23919/cinc53138.2021.9662836"
pmid: "35754520"
pdf: ../papers/hamiltonian_monte_carlo_ap_parameters_2022_nietoramos.pdf
questions: [Q8]
---

## Key Findings
- **Proof-of-concept** that HMC (NUTS/Stan) scales to 13-parameter cardiac models (Fenton-Karma)
- Unimodal posteriors for MS (5 params); broader posteriors for FK (13 params) reflecting identifiability limits
- Estimated noise σ diagnostic: ~0.025 for synthetic (true 0.03), ~10× larger for experimental → model inadequacy signal
- Conference precursor to the full Nieto Ramos 2023 paper

## Method
- HMC via NUTS in Stan, folded normal priors, 3 cycle lengths near alternans
- Mitchell-Schaeffer (5 params) and Fenton-Karma (13 params)
- Synthetic data with Gaussian noise + zebrafish experimental recordings

## Connections to Our Models
- Validates that gradient-based Bayesian methods work for cardiac ODE systems
- Our PyTorch ionic models could interface with NumPyro/Pyro for HMC sampling
- **Actionable**: Use this as validation — implement HMC on MS model first, verify against their results, then scale to TTP06

## Limitations
- Conference paper — less detail than the 2023 journal version
- 13 parameters max; TTP06 has 17+ tunable conductances
