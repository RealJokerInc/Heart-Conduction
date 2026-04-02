# Bidomain Parabolic-Parabolic Formulation

## Question
Does replacing the elliptic equation in the standard bidomain with a parabolic equation (adding a capacitive ∂Ve/∂t term) eliminate the non-physical triangular boundary artifacts caused by instantaneous extracellular information propagation, and can we implement this as an alternative solver in Bidomain V1?

## Status: Active

## Why It Matters
The standard parabolic-elliptic bidomain propagates extracellular potential instantaneously (elliptic = infinite speed). At tissue-bath interfaces, this creates triangular wavefront artifacts instead of the physically expected parabolic shapes. Since our research program depends on accurate boundary effects (Kleber speedup, geometry-induced pacemaking), we need a formulation that captures microscopic bath-loading dynamics faithfully. The parabolic-parabolic formulation adds finite propagation speed to the extracellular domain, which should produce more physical boundary behavior.

## Engines
- **Bidomain V1**: Primary target — add parabolic-parabolic as alternative solver pathway alongside existing parabolic-elliptic

## Completion Criteria
- [ ] Parabolic-parabolic bidomain equations derived and documented (what ∂Ve/∂t term looks like, physical basis, literature precedent)
- [ ] Mathematical/computational requirements identified (what changes vs current decoupled GS approach — both equations now need time-stepping, no elliptic solve)
- [ ] Audit of existing Bidomain V1 code for reusable components (parabolic solver, stencils, spectral tools)
- [ ] Working parabolic-parabolic solver implemented in Bidomain V1
- [ ] Cross-validation: same Kleber boundary setup produces parabolic (not triangular) wavefront shape at bath interface
- [ ] Performance and accuracy comparison vs parabolic-elliptic on standard benchmarks

## Sub-Questions

| Sub-Question | Status | Key Finding |
|-------------|--------|-------------|
| — | — | — |

## Key Findings So Far
The triangular artifact was observed during boundary conduction speedup research (Phase 6). The artifact shape is a direct consequence of the elliptic equation's instantaneous propagation — the extracellular potential adjusts globally at each timestep rather than diffusing with finite speed.

## Literature
| Paper | Summary | Key Insight |
|-------|---------|-------------|
| (to be populated) | | |

## Related Research
- [Boundary conduction speedup](../boundary_conduction_speedup/) — where the triangular artifact was first observed; cross-reference target for validation

## Future Work
{No deferred items yet.}
