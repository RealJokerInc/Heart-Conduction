---
paper: salvador_2024_lnode_cardiac
title: "Whole-heart electromechanical simulations using Latent Neural Ordinary Differential Equations"
authors: "Salvador M, Strocchi M, Regazzoni F, Dedè L, Niederer S, Quarteroni A"
year: 2024
journal: "npj Digital Medicine"
doi: "10.1038/s41746-024-01084-x"
pmid: "38605089"
pdf:
questions: [surrogate_pipeline]
---

## Key Findings
- Extremely compact surrogate (3 hidden layers, 13 neurons each = 39 neurons total) for whole-heart electromechanics
- **300x real-time acceleration** on a single laptop CPU
- Trained from only 400 3D-0D closed-loop simulations
- 43 model parameters (cell-to-organ electromechanics + hemodynamics)
- Successful parameter estimation with uncertainty quantification in 3 hours

## Method
- **Equations**: 3D-0D coupled electromechanical model (electrophysiology + mechanics + hemodynamics)
- **Architecture**: Latent Neural ODEs — encode high-dimensional cardiac dynamics into low-dimensional latent space, solve latent ODE, decode back
- **Autoregressive**: YES (in latent space — latent ODE is integrated forward)
- **Input/Output**: Model parameters → pressure-volume dynamics over cardiac cycles
- **NOT a spatial field predictor** — learns lumped (0D) hemodynamic outputs, not spatiotemporal (Vm, phi_e) fields

## Connections to Our Models

### Relevant Engine Components
- Fundamentally different scope: they model organ-level hemodynamics, we model spatiotemporal field evolution
- Their approach reduces the problem to 0D (pressure-volume loops), losing all spatial information
- Not applicable to our goal of predicting (Vm, phi_e) fields on a grid

### Agreements
- Extreme compression is possible in cardiac modeling (39 neurons → 300x speedup)
- Latent space representations of cardiac dynamics work well

### Disagreements or Gaps
- **0D vs 2D/3D**: They predict integrated outputs (PV loops), not spatial fields. Our surrogate must produce full (Nx, Ny) fields at every timestep.
- **Different use case**: Digital twins for patient calibration vs fast PDE surrogate for research simulation

### Actionable Insights
- **Latent space concept**: Their success with tiny latent ODE validates the concept of learned latent dynamics. Our Ionic Transformer's latent state is analogous — abstract representation of ionic dynamics. Priority: informational.
- **Training efficiency**: 400 simulations was sufficient for their scope. Our data generation budget can be guided by this — start small, scale as needed. Priority: medium.

## Limitations / Caveats
- No spatial field prediction — cannot produce Vm or phi_e maps
- Only captures lumped cardiac function (EF, pressures, volumes)
- Not a PDE surrogate in the traditional sense — more like a learned reduced-order model
