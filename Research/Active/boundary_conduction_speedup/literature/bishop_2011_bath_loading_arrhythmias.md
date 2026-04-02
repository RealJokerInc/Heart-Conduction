---
paper: bishop_2011_bath_loading_arrhythmias
title: "Cardiac Bidomain Bath-Loading Effects during Arrhythmias: Interaction with Anatomical Heterogeneity"
authors: "Bishop MJ, Vigmond E, Plank G"
year: 2011
journal: "Biophys J"
doi: "10.1016/j.bpj.2011.10.052"
pmid: "22208185"
pmc: "PMC3244060"
pdf: ../papers/bath_loading_arrhythmia_wavefront_curvature_2011_bishop.pdf
questions: [Q5]
---

## Key Findings
- Bath loading increases CV close to the tissue-fluid interface, inducing transmural wavefront curvature
- Without fiber rotation, bath-loading curvature dominates activation pattern, increasing arrhythmia complexity
- With fiber rotation, bath-loading effects are direction-dependent: accentuates concave wavefronts, attenuates convex ones
- Increased surface CV reduces arrhythmia inducibility by increasing the wavelength at the surface
- Important to include bath-loading effects when comparing simulations with experimental data (optical mapping, electrode recordings)

## Method
- Full 3D bidomain with explicit bath domain
- Anatomically detailed ventricular geometry with fiber rotation
- Arrhythmia induction protocols (S1-S2 cross-field stimulation)
- Compared bath-coupled bidomain vs no-bath bidomain vs monodomain
- Likely CARP/openCARP solver framework (same group as Bishop & Plank 2011)

## Key Equations / Results
- Transmural CV gradient creates curvature: surface faster than midwall
- Fiber rotation interacts with bath effect: when wavefront propagation aligns with fiber direction at the surface, bath speedup is maximally effective
- Arrhythmia vulnerability window changes with bath coupling
- Wavelength = CV x APD; increased surface CV increases wavelength, making reentry harder to sustain at the surface

## Connections to Our Models

### Relevant Engine Components
- **Bidomain V1**: Our `bc_type='bath_tb'` bath coupling directly relates to the transmural gradient studied here (bath on top/bottom surfaces)
- **Triangle merger experiment**: The V-shaped wavefront we observe is the 2D analog of the transmural curvature this paper studies in 3D
- **Anisotropic test** (`experiments/anisotropic_test.py`): Tests bath loading with anisotropic conductivities, directly relevant to their fiber rotation analysis

### Agreements
- Confirms that bath-loading curvature is a dominant effect that must be included in simulations comparing to experiment
- The transmural gradient (surface faster than bulk) is exactly what our triangle merger shows in 2D cross-section
- Direction-dependent effects are consistent with our anisotropic test results showing different Kleber ratios in longitudinal vs transverse directions

### Disagreements or Gaps
- We do not model fiber rotation in our current Bidomain V1 (2D only)
- Our simulations focus on planar propagation, not arrhythmia dynamics
- We have not tested how bath loading interacts with reentry or spiral waves

### Actionable Insights
- **HIGH**: When extending to 3D or anatomical models, bath loading must be included — monodomain without augmentation will give wrong arrhythmia dynamics
- **MEDIUM**: The wavelength argument (CV x APD) provides a theoretical framework for predicting how boundary effects influence reentry stability
- **LOW**: Consider testing S1-S2 protocols in our 2D bidomain to see if bath loading affects vulnerability window even in simplified geometry

## Limitations / Caveats
- Full 3D anatomical model makes results geometry-specific; quantitative numbers may not transfer to simpler geometries
- Does not provide a simple analytical formula for the direction-dependent interaction with fiber rotation
- Computational cost of full bidomain with bath domain is high; the augmented monodomain from their companion paper addresses this
