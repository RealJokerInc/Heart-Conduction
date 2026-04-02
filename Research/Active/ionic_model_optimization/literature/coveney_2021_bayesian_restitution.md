---
paper: coveney_2021_bayesian_restitution
title: "Bayesian Calibration of Electrophysiology Models Using Restitution Curve Emulators"
authors: "Coveney S, Clayton C, Oakley J, et al."
year: 2021
journal: "Frontiers in Physiology"
doi: "10.3389/fphys.2021.693015"
pmid: "34366883"
pdf: ../papers/bayesian_calibration_restitution_emulators_2021_coveney.pdf
questions: [Q8]
---

## Key Findings
- **Restitution Curve Emulators (RCEs)** use PCA + Gaussian Processes to create a fast surrogate of the parameter → restitution curve mapping, enabling 10^5 predictions per second after 500 training simulations
- Just 3 PCA components capture >99% of variance in CV(S2) and APD(S2) restitution curves
- **Combining CV + APD + ERP restitution** is critical for identifiability — any single curve leaves parameters poorly constrained
- Even with only 5 parameters (modified Mitchell-Schaeffer), some parameters show strip-shaped posterior degeneracies
- Diffusion coefficient D primarily controls CV; repolarization time constants control APD — these are partially separable

## Method
- **Surrogate model**: Latin Hypercube Design (500 simulations via openCARP) → PCA on restitution curves → GP regression on PCA coordinates
- **Calibration**: Bayesian MCMC via `emcee` package, using RCE as the fast forward model
- **Tissue simulations**: Monodomain on 24×0.6mm strip, S1S2 pacing protocol, bisection for ERP
- **Model**: Modified Mitchell-Schaeffer (5 params: D, tau_in, tau_out, tau_close, tau_open)
- **GP kernel**: Squared exponential with automatic relevance determination + linear mean function
- Reparameterization to CV_max and APD_max for better-conditioned parameter space

## Key Equations / Results
- CV_max = 0.5 * sqrt((1/2 - V_gate)^2 * D / tau_in)
- APD_max = tau_close * log(1/4 + tau_out/tau_in * (1 - V_gate)^2)
- RCE prediction: F(S2) ≈ Φ₀(S2) + Σ_c f_c · Φ_c(S2), where f_c ~ GP(parameter_vector)
- Cross-validation R² > 0.999 for ERP emulator
- 500 training simulations sufficient for 5-parameter space
- S1S2 restitution surfaces (varying S1) add minimal information — single S1 + ERP sufficient

## Connections to Our Models

### Relevant Engine Components
- **Bidomain V1 / Monodomain V5.4**: We already compute CV from tissue simulations — this is the measurement that would feed the emulator
- **Monodomain V5.4 S1S2 protocol**: `examples/spiral_wave_s1s2.py` already implements S1S2 pacing — could be adapted to generate restitution curves
- Our spectral solver could generate the 500 training simulations efficiently on GPU

### Agreements
- Confirms our experience that CV is primarily controlled by the diffusion coefficient (and by extension GNa which sets the wavefront speed)
- The monodomain tissue simulation approach matches our Engine V5.4

### Disagreements or Gaps
- Only tested on 5-parameter phenomenological model — NOT on TTP06 or ORd. The authors acknowledge this as "a logical next step"
- For TTP06/ORd, 500 training simulations may not suffice — the parameter space is higher-dimensional (6-16 conductances)
- Rule of thumb: need ~10× parameters training points, so TTP06 with 10 tunable conductances would need ~5000 simulations

### Actionable Insights
- **HIGH**: Build an RCE for our TTP06 model. Generate Latin Hypercube training data (vary GNa, GCaL, GKr, GKs, GK1 + D), compute CV/APD restitution, fit PCA+GP. This gives us a fast surrogate for BayesOpt.
- **HIGH**: Use the combined CV + APD + ERP restitution approach for calibration — any single measurement is insufficient
- **MEDIUM**: The reparameterization to CV_max / APD_max improves GP fitting — apply analogous transformations to TTP06 outputs
- **LOW**: S1S2 pacing at a single S1 is sufficient — no need for full restitution surfaces

## Limitations / Caveats
- 5-parameter model only — scaling to TTP06/ORd is unvalidated
- PCA assumes linear modes of variation — may need more components for detailed ionic models
- 500 training simulations took ~hours on openCARP — our GPU engine should be much faster
- No model discrepancy treatment (assumes noise but not systematic bias)
