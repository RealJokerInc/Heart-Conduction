---
paper: bishop_2011_augmented_monodomain
title: "Representing Cardiac Bidomain Bath-Loading Effects by an Augmented Monodomain Approach"
authors: "Bishop MJ, Plank G"
year: 2011
journal: "IEEE Trans Biomed Eng"
doi: "10.1109/TBME.2010.2096425"
pmid: "21292591"
pmc: "PMC3075562"
pdf: ../papers/augmented_monodomain_bath_loading_2011_bishop.pdf
questions: [Q5]
---

## Key Findings
- Bath-loading induces V-shaped wavefront curvature in bidomain simulations: surface leads bulk
- Edge CV is 11.1% faster than bulk for isotropic conductivities, 15.2% for anisotropic
- Effect depends STRONGLY on conductivity parameter set: Clerc gives only 2.4% speedup, Roberts & Scher gives 47.6%
- Key parameter is the effective conductivity ratio R = sigma_eff_edge / sigma_eff_bulk, where at the edge sigma_e effectively becomes sigma_bath (low resistance path)
- Even a 0.1mm bath layer produces significant wavefront curvature; bath conductivity above 0.5 S/m has minimal additional effect
- Developed "augmented monodomain" (MDMEQ): tag 3 elements at tissue-bath interface, boost their conductivity by factor R, achieving ~3% activation time agreement with full bidomain at ~7x speedup

## Method
- Full 3D bidomain with explicit bath domain (BDM) as ground truth
- Bidomain no-bath (BDMNB) and monodomain (MDM) as controls
- Augmented monodomain (MDMEQ): monodomain with conductivity boosted by factor R in 3 boundary elements
- Tissue slab geometry with bath on surfaces
- FEM solver (likely CARP/openCARP framework)

## Key Equations / Results
- R = [(sigma_i * sigma_bath) / (sigma_i + sigma_bath)] / [(sigma_i * sigma_e) / (sigma_i + sigma_e)]
- For isotropic conductivities: R = 1.111, giving 11.1% edge speedup
- For anisotropic conductivities: R = 1.152 (longitudinal), giving 15.2% edge speedup
- "Wavefront in the BDMNB case is fully planar... In the BDM case the wavefront exhibits significant curvature... a V-shaped wavefront morphology"
- "Initially (25 ms images) the wavefront at the centre of the tissue in the BDM case propagates at approximately the same speed as the BDMNB wavefront"
- MDMEQ activation time error vs full bidomain: ~3%

## Connections to Our Models

### Relevant Engine Components
- **Bidomain V1**: `tests/cv_shared.py` with `bc_type='bath_tb'` implements bath coupling via Dirichlet phi_e=0 at top/bottom edges
- **Triangle merger experiment** (`experiments/triangle_merger.py`): directly observes the V-shaped wavefront this paper characterizes, with edge leading center by 1.65-1.70 cm
- **Conductivity sweep** (`experiments/conductivity_sweep.py`): tests how the edge lead depends on sigma_i/sigma_e ratio, directly comparable to this paper's R parameter analysis
- **Kleber ratio**: `research/BOUNDARY_SPEEDUP_ANALYSIS.md` derives the theoretical ratio sqrt((sigma_i + sigma_e)/sigma_e) = 1.131 for our parameter set

### Agreements
- Our triangle merger experiment produces the same V-shaped wavefront morphology described here
- Our Kleber ratio of 1.0714 at dx=0.025 is converging toward the theoretical value, consistent with their 11.1% isotropic result
- The strong dependence on conductivity parameter set explains why our conductivity sweep shows the edge lead depends on the Kleber ratio, not D_eff alone

### Disagreements or Gaps
- Our Bidomain V1 uses simplified bath coupling (phi_e=0 Dirichlet BC) rather than an explicit bath domain with finite conductivity. This is equivalent to assuming infinite bath conductivity (sigma_bath >> sigma_e), which this paper shows is a reasonable approximation above 0.5 S/m
- We do not implement the augmented monodomain (MDMEQ) approach. This could be useful as a fast approximation in Monodomain V5.4
- Our theoretical Kleber ratio sqrt((D_i+D_e)/D_e) = 1.131 is slightly different from their R formula because they account for finite bath conductivity

### Actionable Insights
- **HIGH**: The R formula provides a more general prediction than our sqrt formula when bath conductivity is finite. Consider implementing finite-conductivity bath in Bidomain V1 to test sensitivity
- **HIGH**: The augmented monodomain approach (boost conductivity by R in boundary elements) could be implemented in Monodomain V5.4 as a fast alternative to full bidomain for boundary studies
- **MEDIUM**: Test our simulations with different conductivity parameter sets (Clerc, Roberts & Scher) to verify the strong parameter dependence they report
- **LOW**: The 3-element boundary layer depth is a useful calibration point for our mesh resolution requirements

## Limitations / Caveats
- FEM-based results; direct quantitative comparison with our FDM implementation requires accounting for discretization differences
- The augmented monodomain requires knowing R a priori, which depends on bath conductivity and tissue conductivities
- Only tested with planar wavefront stimulation; does not address point stimuli or complex wavefront geometries
- Does not address the overdetermined boundary condition problem (see Patel & Roth 2005)
