---
paper: johnston_2008_approximate_bidomain
title: "Approximate solutions for certain bidomain problems in electrocardiography"
authors: "Johnston PR"
year: 2008
journal: "Phys Rev E"
doi: "10.1103/PhysRevE.78.041904"
pmid: "18999452"
pdf: ../papers/johnston_2008_approximate_bidomain.pdf
questions: [Q5]
---

## Key Findings
- Extends the Patel & Roth (2005) approximate solution method for bidomain boundary problems
- Shows that equivalent mathematical approaches to the overdetermined BC problem existed ~20 years earlier in the literature
- Provides additional functional forms for the boundary layer correction that are more appropriate to specific boundary geometries
- Generally confirms Patel & Roth's results but identifies situations where geometry-specific functional forms give improvements over the simple exponential decay ansatz

## Method
- Analytical / perturbation approach to the bidomain equations at tissue-bath interfaces
- Extension of the exponential decay ansatz from Patel & Roth (2005) to other functional forms
- Comparison of different approximate boundary layer corrections
- Testing against numerical bidomain solutions for various geometries

## Key Equations / Results
- Patel & Roth used Vm ~ exp(-x/delta) as boundary layer correction; Johnston shows this is not unique
- Alternative forms (e.g., polynomial decay, error-function profiles) may better match specific geometries
- The key insight is that the overdetermined BC system (3 conditions for 2 unknowns) requires a boundary layer to reconcile
- Different functional forms converge to the same result as delta -> 0, but at finite delta (finite mesh resolution) the choice matters
- Confirms the general validity of the approximate approach

## Connections to Our Models

### Relevant Engine Components
- **Bidomain V1**: Our handling of the tissue-bath boundary (phi_e Dirichlet + Vm Neumann) encounters exactly the overdetermined system that both Patel & Roth and Johnston address
- **Mesh convergence** (`tests/test_phase6d_convergence.py`): Our convergence of the Kleber ratio with decreasing dx is directly related to resolving the boundary layer that these approximate methods describe
- **Elliptic solver**: The boundary layer structure affects how many grid points are needed near the boundary for accurate phi_e computation

### Agreements
- Confirms that the overdetermined BC problem is a fundamental issue in bidomain tissue-bath coupling, not specific to our implementation
- The boundary layer concept explains our observed mesh-dependent Kleber ratio convergence
- The delta -> 0 limit (which all functional forms agree on) is what our simulations approach as dx -> 0

### Disagreements or Gaps
- We do not implement any explicit boundary layer correction — we rely on mesh refinement to resolve it
- The choice of functional form for the boundary layer correction is not relevant to our FDM approach (we solve the full system numerically)
- Our convergence rate (1.0385 -> 1.0714 -> 1.131) may be improved if we used a boundary-layer-aware discretization

### Actionable Insights
- **MEDIUM**: The boundary layer width (delta) from these analytical solutions could predict the mesh resolution needed for a given accuracy target in the Kleber ratio
- **LOW**: If implementing a boundary-layer-corrected scheme, Johnston's geometry-specific forms would be more accurate than Patel & Roth's simple exponential
- **LOW**: Could use these analytical solutions to construct high-order boundary corrections in our FDM stencils

## Limitations / Caveats
- Purely analytical work — no full reaction-diffusion simulations with these corrections
- The "improvement" over Patel & Roth is incremental, mainly relevant for finite delta
- Does not provide a practical numerical algorithm; the value is theoretical understanding
- Limited to simple geometries where analytical forms can be derived
