---
paper: tranquillo_2005_analytical_bath
title: "Analytical model of extracellular potentials in a tissue slab with a finite bath"
authors: "Tranquillo JV, Burwell DO, Henriquez CS"
year: 2005
journal: "IEEE Trans Biomed Eng"
doi: "10.1109/TBME.2004.840467"
pmid: "15709672"
pdf: ../papers/tranquillo_2005_analytical_bath.pdf
questions: [Q5]
---

## Key Findings
- Provides an analytical solution for extracellular potentials arising from a planar wavefront in a 3D tissue slab with variable bath thickness
- Starting from a known transmembrane potential profile, yields phi_e at any point in both the bath and tissue domains
- Matches full reaction-diffusion bidomain results across different bath thicknesses
- Also works for abrupt ionic inhomogeneity (e.g., ischemic boundary) within the tissue
- Computational cost is trivial compared to full numerical bidomain

## Method
- Analytical / Fourier series solution to the passive bidomain equations
- Assumes transmembrane potential Vm is KNOWN (prescribed wavefront shape)
- Solves for phi_e in tissue and phi in bath as a linear boundary value problem
- 3D tissue slab geometry: finite thickness tissue surrounded by bath of variable thickness
- Validated against full reaction-diffusion bidomain simulations

## Key Equations / Results
- Given Vm(x,y,z,t), solve Laplace/Poisson equation for phi_e with tissue-bath boundary conditions
- Bath thickness affects the phi_e distribution: thicker bath provides lower-resistance current return path
- phi_e variation through tissue thickness is significant even for thin tissue
- Abrupt ionic inhomogeneity (e.g., sigma_i step change) creates additional phi_e features
- Analytical solution expressed as Fourier series with exponential depth dependence

## Connections to Our Models

### Relevant Engine Components
- **Bidomain V1 elliptic solver**: Our elliptic equation A_ellip * phi_e = L_i * Vm is exactly the type of problem this paper solves analytically (given Vm, find phi_e)
- **DST-based solver**: Our spectral solver for phi_e with Dirichlet BCs uses the same mathematical framework (eigenfunction expansion) as their Fourier series solution
- **`bc_type='bath_tb'`**: Our bath coupling with phi_e=0 at boundaries is the infinite bath thickness limit of their variable-thickness solution

### Agreements
- The analytical framework confirms that our elliptic phi_e solve with Dirichlet BCs is mathematically well-posed
- The Fourier series approach is mathematically equivalent to our DST-based spectral solver
- Their finding that phi_e varies through tissue thickness is consistent with the boundary layer effects that drive the Kleber speedup

### Disagreements or Gaps
- Their solution assumes Vm is KNOWN, so it does not capture the feedback loop where phi_e influences Vm propagation (which is what creates the Kleber boundary speedup). Our full bidomain simulation captures this feedback
- Our phi_e=0 BC assumes infinite bath, while they analyze finite bath thickness effects that we do not model
- We do not implement variable bath thickness

### Actionable Insights
- **HIGH**: The analytical solution could serve as a MANUFACTURED SOLUTION for verifying our elliptic phi_e solver — prescribe Vm, compute phi_e analytically, compare with numerical result
- **MEDIUM**: The finite bath thickness analysis could inform whether our infinite-bath assumption (phi_e=0) is adequate for realistic experimental setups where bath depth is limited
- **LOW**: The abrupt inhomogeneity solution could be useful for future scar/ischemia boundary studies (connects to Q6)

## Limitations / Caveats
- Assumes known Vm profile (no reaction-diffusion feedback), so cannot predict CV changes
- Planar wavefront assumption limits applicability to complex wavefront geometries
- Uniform tissue properties (except for the inhomogeneity extension)
- Does not directly address the boundary speedup effect — provides the phi_e field but not its influence on subsequent propagation
