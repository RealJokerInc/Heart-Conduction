# Boundary Conduction Speedup (Kleber Effect) — Knowledge File

> This file is a running synthesis. Updated as findings accumulate.
> When the question is complete, a copy is promoted to `Research/Knowledge/`.

## Current Understanding

At an insulated (no-flux) tissue boundary in a bath-perfused preparation, conduction velocity increases by approximately 7-13% compared to the tissue interior. This is the **Kleber boundary speedup**, a real biophysical effect confirmed both experimentally and computationally.

### Mechanism (8-link argument chain)

The effect arises from the asymmetric boundary conditions in the bidomain model at a tissue-bath interface:

1. **Intracellular domain terminates** (Neumann BC): gap junctions end at the tissue surface, so no intracellular current crosses the boundary.
2. **Extracellular domain is continuous with the bath** (Dirichlet BC): the interstitial fluid connects to the low-resistance bath solution, effectively clamping phi_e to zero at the surface.

This asymmetry short-circuits the extracellular return path for propagating wavefronts. In bulk tissue, the wavefront current traverses both intracellular resistance r_i (forward) and extracellular resistance r_e (return), giving sigma_eff = sigma_i * sigma_e / (sigma_i + sigma_e). Near the boundary, the bath provides a parallel low-resistance return path, reducing effective resistance to approximately r_i alone, so sigma_eff_boundary approaches sigma_i.

**Theoretical CV ratio** (longitudinal human ventricular tissue):
```
CV_boundary / CV_interior = sqrt((sigma_i + sigma_e) / sigma_e)
                          = sqrt((1.74 + 6.25) / 6.25) = sqrt(1.278) = 1.131
```

The enhancement decays exponentially into the tissue interior with characteristic length lambda (electrotonic space constant, approximately 1.4 mm at rest).

### Our results

- **Bidomain V1**: CV ratio = 1.0714 at dx=0.025 cm, converging toward 1.131 with mesh refinement (confirmed via mesh convergence study)
- **LBM V1**: D2Q9 with Dirichlet BC also captures the speedup, though with a ~35% CV baseline offset due to LBM numerical dispersion
- **Monodomain FDM control**: No boundary speedup (as expected, since monodomain lacks the two-domain BC asymmetry)

### Where the effect is present

| Geometry | Intracellular | Extracellular | Kleber effect? |
|----------|--------------|---------------|----------------|
| Tissue submerged in Tyrode's | Terminates | Bath-coupled | YES |
| Laser-cut void (bath fills void) | Terminates | Bath-coupled | YES |
| Tissue on glass substrate | Terminates | Insulated (glass) | NO |
| In vivo (blood contact) | Terminates | Bath-coupled (blood) | YES |

### Key literature

- **Kleber et al. 2021** (PMID 34296210): Most recent comprehensive review on safety factor, coupling, and boundaries
- **Kucera et al. 1998** (PMID 9776726): Foundational work on geometry-CV relationship and branching effects
- **Connolly et al. 2015** (PMID 25872206): Direct evidence of electrotonic load gradients at infarct border zones
- **Shaw & Rudy 1997** (PMID 9351447): Ionic mechanisms linking safety factor to conduction
- **Roth 1991** (Ann Biomed Eng 19:669-678): Bidomain boundary condition equivalence
- **Patel & Roth 2005** (Phys Rev E 72:051931): Matched-asymptotic solution showing exponential boundary layer

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Validation approach | Mesh convergence toward 1.131 | Theoretical ratio provides analytical target |
| Boundary treatment | Face-based FDM (not ghost-node) | Symmetric SPD Laplacian required for bidomain |
| Control experiment | Monodomain FDM with same grid | Isolates bidomain boundary effect from curvature |
| Bath coupling model | Dirichlet phi_e = 0 at surface | Standard approximation; exact in infinite bath limit |

## Open Questions

- What is the convergence rate with mesh refinement? (Is it O(dx) or O(dx^2)?)
- How does the speedup interact with wavefront curvature at obstacle corners? (Curvature speedup is a separate geometric effect)
- Does the speedup magnitude change with anisotropic conductivity tensors (fiber orientation at boundary)?
- At what tissue thickness does the boundary layer span the entire preparation (transitioning from surface effect to full-thickness effect)?
- Does the EMI model (cell-resolved, no homogenization) reproduce the same speedup magnitude?

## Connections
- **Engines**: Bidomain V1 (primary validation), LBM V1 (secondary confirmation)
- **Related research**: scar_bc_validity (Q6 -- Neumann not Dirichlet at scar), lbm_cardiac (Q4 -- LBM can capture the effect with D2Q9)
- **Pipelines**: Triangle merger experiments (bidomain vs monodomain CV comparison on realistic geometries)
