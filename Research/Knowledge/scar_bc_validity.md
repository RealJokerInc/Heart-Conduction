# Scar Boundary Condition Validity — Knowledge File

> This file is a running synthesis. Updated as findings accumulate.
> When the question is complete, a copy is promoted to `Research/Knowledge/`.

## Current Understanding

At scar tissue boundaries, the physically correct boundary condition is **Neumann (no-flux)**, not Dirichlet (voltage clamping). Applying Dirichlet BC at a scar boundary is unphysical because it implies the scar acts as a voltage source or ground, which contradicts the biology.

### Physical reasoning

Scar tissue (fibrosis, infarct core) is electrically inert:
- **No ion channels**: Fibroblasts and collagen do not express cardiac ion channels. There is no active membrane current.
- **No gap junctions to viable myocardium**: Connexin-43 (Cx43) is absent or non-functional at the scar-myocardium interface. No intracellular current can flow into or out of the scar.
- **No voltage source**: Dead tissue cannot maintain or impose a potential. Clamping V = constant at a scar border implies an infinite current source, which is unphysical.

Therefore, the correct BC is **zero normal flux** (Neumann):
```
n . sigma_i . grad(phi_i) = 0   (no intracellular current crosses into scar)
n . sigma_e . grad(phi_e) = 0   (no extracellular current crosses into scar)
```

Both domains have Neumann BCs at scar interfaces. This is symmetric -- there is no BC asymmetry, and therefore no Kleber boundary speedup at scar boundaries (in contrast to tissue-bath interfaces where the asymmetry drives the speedup).

### Contrast with tissue-bath interface

| Interface | phi_i BC | phi_e BC | Asymmetry? | Kleber speedup? |
|-----------|----------|----------|------------|-----------------|
| **Tissue-bath** | Neumann (cells end) | Dirichlet (bath shorts r_e) | YES | YES (~13%) |
| **Tissue-scar** | Neumann (no gap junctions) | Neumann (no current path) | NO | NO |
| **Tissue-glass** | Neumann (cells end) | Neumann (insulator) | NO | NO |

### ML-DO implications

An Oxford ML-Directed Optimization proposal used Dirichlet BCs at laser-cut scar boundaries in a D2Q9 LBM simulation. This introduces a computational artifact: the Dirichlet BC creates a fictitious voltage clamp that short-circuits the extracellular return path, producing a boundary speedup that does not exist at real scar tissue. The optimization objective (arrhythmogenicity of scar geometry) is therefore confounded by an unphysical speedup at every scar border.

The correct model for laser-cut voids in a bath-perfused monolayer is:
- **Void edges**: Dirichlet on phi_e (bath fills the void) -- Kleber speedup IS present
- **Scar edges** (if modeling fibrotic tissue): Neumann on both domains -- no speedup

The distinction matters because the two BC types produce qualitatively different conduction patterns at obstacle borders.

### Two distinct boundary CV phenomena

1. **Curvature effect** (monodomain, geometric): v(kappa) = v_0 - D * kappa. At obstacle corners, wavefront curvature changes CV. Present in any consistent discretization. This is a geometric effect, not a BC effect.
2. **Kleber effect** (bidomain, BC-driven): Bath shorts the extracellular return path at tissue-bath interfaces. Only present with bidomain + asymmetric BCs. NOT present at scar (symmetric BCs).

These must not be conflated. The curvature effect exists everywhere; the Kleber effect exists only at tissue-bath interfaces.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scar BC | Neumann (no-flux) on both domains | Scar has no ion channels, no gap junctions, no voltage source |
| Bath-tissue BC | Dirichlet (extracellular only) | Bath is a conductive volume at known potential |
| Implementation | Harmonic mean D at cell faces | D=0 at scar face gives zero flux automatically |
| EMI alternative | Not pursued | Extracellular-Membrane-Intracellular model is more accurate but much more complex |
| Validation | Compare CV at scar border vs interior -- should show NO speedup | Confirms correct BC choice |

## Open Questions

- None remaining for the core question (Neumann is definitively correct for scar). The question is marked Complete.
- Secondary question: How does the border zone (partially coupled fibroblasts) transition from Neumann (pure scar) to full coupling (healthy tissue)? This would require a Robin BC or a graded conductivity model.

## Connections
- **Engines**: Bidomain V1 (validates that Neumann at scar gives no speedup), LBM V1 (bounce-back BC implements no-flux)
- **Related research**: boundary_conduction_speedup (Q5 -- the effect that Dirichlet incorrectly produces at scar), lbm_cardiac (Q4 -- BC implementation in LBM)
- **Pipelines**: None directly; informs Surrogate design (correct BC encoding)
- **Impact**: Validates Bidomain V1's use of Neumann BCs at tissue edges; identifies a flaw in competitor approaches using Dirichlet at scar boundaries
