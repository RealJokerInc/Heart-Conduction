---
paper: patel_2005_bidomain_boundary
title: "Approximate solution to the bidomain equations for electrocardiogram problems"
authors: "Patel SG, Roth BJ"
year: 2005
journal: "Phys Rev E"
doi: "10.1103/PhysRevE.72.051931"
pmid: "16383669"
pdf: ../papers/overdetermined_boundary_condition_bidomain_2005_patel.pdf
questions: [Q5]
---

## Key Findings
- Identifies the OVERDETERMINED BOUNDARY CONDITION problem at tissue-bath interface: two potentials (phi_e in tissue, phi in bath) must satisfy THREE boundary conditions simultaneously
- The three BCs are: (1) phi_e = phi_bath at interface, (2) n . sigma_e . grad(phi_e) = n . sigma_bath . grad(phi_bath) (current continuity), (3) n . sigma_i . grad(Vm) = 0 (no intracellular current leaves tissue)
- This overdetermination means the standard bidomain formulation is ill-posed at tissue-bath interfaces
- Their fix: add an exponential decay term to Vm that falls off with depth into tissue, satisfying BC (3) while allowing BCs (1) and (2) to be imposed normally
- Taking the limit as the decay length goes to zero yields two effective BCs that approximately handle all three conditions

## Method
- Analytical / semi-analytical approach to the bidomain equations
- 1D and 2D tissue-bath interface geometry
- Perturbation expansion in the decay length parameter
- Comparison with full numerical bidomain solutions

## Key Equations / Results
- Three BCs at tissue-bath boundary: phi_e = phi_bath, n . sigma_e . grad(phi_e) = n . sigma_bath . grad(phi_bath), n . sigma_i . grad(Vm) = 0
- The third BC (no intracellular current flux) is automatically violated when imposing Dirichlet on phi_e, because Vm = phi_i - phi_e and constraining phi_e while phi_i has no-flux creates a boundary layer
- Solution: Vm(x) = Vm_bulk + A * exp(-x/delta) near the boundary, where delta is the decay length
- In the limit delta -> 0, effective BCs reduce to two conditions that can be imposed directly

## Connections to Our Models

### Relevant Engine Components
- **Bidomain V1**: Our `bc_type='bath_tb'` imposes phi_e = 0 (Dirichlet) at bath boundaries and Vm Neumann (no intracellular flux). This is exactly the overdetermined system this paper analyzes
- **Elliptic solver**: Our DST-based solver for the phi_e equation with Dirichlet BCs implicitly handles condition (1). Condition (3) is imposed on the parabolic equation for Vm
- **`tests/test_phase6c_boundary_cv.py`**: Tests the Kleber effect under these boundary conditions — the boundary layer described in this paper explains why our measured ratio (1.0714) has not yet converged to the theoretical value at dx=0.025
- **`research/BOUNDARY_SPEEDUP_ANALYSIS.md`**: Our Kleber derivation implicitly assumes the three BCs are satisfied

### Agreements
- The overdetermined BC problem is exactly what our Bidomain V1 encounters when coupling tissue to bath
- The exponential boundary layer in Vm explains the mesh-dependent convergence of our Kleber ratio (1.0385 at dx=0.05 -> 1.0714 at dx=0.025 -> 1.131 theory)
- Our approach of imposing phi_e=0 (Dirichlet) + Vm no-flux (Neumann) is a common practical approximation that this paper formally justifies in the delta->0 limit

### Disagreements or Gaps
- We do not explicitly resolve the exponential boundary layer in Vm — our FDM discretization at dx=0.025 may not adequately resolve it if the decay length is comparable to or smaller than dx
- Our simplified phi_e=0 BC assumes infinite bath conductivity, which avoids BC (2) entirely. This paper's analysis is more general
- The slow mesh convergence of our Kleber ratio may be partly due to inadequate resolution of this boundary layer

### Actionable Insights
- **HIGH**: The boundary layer decay length sets a minimum mesh resolution for accurate Kleber ratio. Estimate delta from tissue parameters and ensure dx << delta for proper convergence
- **HIGH**: This paper provides the theoretical foundation for why our phi_e=0 Dirichlet BC is a valid approximation — it is the delta->0 limit of the full three-BC problem
- **MEDIUM**: Consider implementing an adaptive mesh refinement near bath boundaries to better resolve the boundary layer without increasing global resolution
- **LOW**: The analytical solution could provide a manufactured solution for verifying our boundary implementation

## Limitations / Caveats
- Approximate solution — the delta->0 limit may not be accurate for all parameter regimes
- 1D/2D analysis; extension to 3D and complex geometries is not straightforward
- Does not quantify the error introduced by the approximation for specific conductivity parameter sets
- Does not address time-dependent effects (wavefront arrival at boundary)
