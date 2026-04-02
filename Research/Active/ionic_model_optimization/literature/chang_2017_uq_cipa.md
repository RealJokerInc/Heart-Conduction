---
paper: chang_2017_uq_cipa
title: "Uncertainty Quantification Reveals the Importance of Data Variability and Experimental Design Considerations for Proarrhythmia Risk Assessment"
authors: "Chang KC, Dutta S, Mirams GR, et al."
year: 2017
journal: "Frontiers in Physiology"
doi: "10.3389/fphys.2017.00917"
pmid: "29209226"
pdf: ../papers/uncertainty_quantification_proarrhythmia_cipa_2017_chang.pdf
questions: [Q8]
---

## Key Findings
- **CiPA ORdv1.0**: ORd model with explicitly rescaled conductances — demonstrates the kind of tuning we need
- Drug-hERG binding: Ku and EC50 are strongly correlated; only the ratio Ku/EC50 is identifiable
- **>60% block rule**: IC50 can only be reliably estimated if >60% block is achieved experimentally
- TdP risk stratification via qNet metric is robust at 1-4× therapeutic concentration
- UQ (bootstrapping + MCMC DRAM) reveals which parameter uncertainties propagate to predictions

## Method
- **UQ methods**: Bootstrapping (2000 samples from voltage-clamp data) + Bayesian MCMC (DRAM algorithm)
- **Model**: ORd (O'Hara-Rudy) with CiPA conductance scaling
- **Drug model**: hERG Markov model + Hill-equation IC50 block for 6 other currents
- **Tested**: 12 CiPA training compounds (High/Intermediate/Low TdP risk)

## Connections to Our Models
- Operates directly on **ORd model** — our `cardiac_sim/ionic/ord/` implementation
- CiPA ORdv1.0 conductance scaling factors are a validated starting point for our ORd tuning
- The qNet metric (integral of net current) could serve as an additional optimization objective
- **Actionable**: Use CiPA ORdv1.0 scaling as our baseline, then optimize further for target CV/APD
- **Actionable**: Always pair parameter optimization with UQ — report confidence intervals, not just point estimates

## Limitations
- Drug-focused (proarrhythmia risk), not CV/APD tuning per se
- UQ methods computationally expensive (2000 bootstrap + MCMC)
- No tissue-level validation (single-cell qNet only)
