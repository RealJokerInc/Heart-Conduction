---
paper: zhang_2024_gradient_two_waveform
title: "Cell-specific models of hiPSC-CMs developed by the gradient-based parameter optimization method fitting two different action potential waveforms"
authors: "Zhang Y, Kohjitani H, Bhatt SM, et al."
year: 2024
journal: "Scientific Reports"
doi: "10.1038/s41598-024-63413-0"
pmid: "38849433"
pdf: ../papers/gradient_based_hipsc_two_waveform_fitting_2024_zhang.pdf
questions: [Q8]
---

## Key Findings
- **Two-waveform fitting** (control AP + IKr-blocked AP) dramatically reduces parameter correlations vs single-AP fitting
- Gradient-based Parameter Search method efficiently tunes 6 conductance scaling factors (GK1, GKr, GCaL, GNaL, Gf, Gb)
- IKr/GCaL correlation during plateau is broken by removing IKr in the second waveform
- Model-to-model accuracy: all 6 parameters within ~2% of true values
- Experimental (hiPSC-CM) accuracy: 16/21 measurements within 10% deviation

## Method
- **Gradient-based PO**: dG_i = -ε · dMSE/dG_i at each iteration
- **Model**: hiPSC-CM Kohjitani model (6 tunable conductances)
- **Innovation**: Simultaneous MSE minimization over control + E-4031 (IKr block) waveforms
- **Multi-run**: randomized initial params, top 20 by MSE selected

## Connections to Our Models
- Gradient-based approach is directly applicable to our PyTorch TTP06/ORd (autograd available)
- The two-waveform strategy could be implemented: run TTP06 at baseline + with GKr scaled to 0.4 (60% block)
- **Actionable**: For quick TTP06 tuning of ~6 conductances, gradient descent with paired waveforms is faster than GA/Bayesian
- **Limitation**: Only 6 parameters; scaling to 16 untested

## Limitations
- hiPSC-CM model, not TTP06/ORd directly
- Gradient methods find local optima — needs multi-start or global search first
- Single-cell only, no tissue-level CV optimization
