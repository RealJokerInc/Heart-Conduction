# LBM for Cardiac Electrophysiology

## Question
Can Lattice Boltzmann methods solve cardiac electrophysiology equations?

## Status: Complete (2026-03-16)

## Key Answer
**Yes for monodomain, feasible but complex for bidomain.** LBM replaces PDE discretization with a kinetic scheme: distribute → collide → stream. D2Q5 handles isotropic diffusion; D2Q9 is needed for anisotropic tensors (Dxy cross-terms). BGK collision is simplest; MRT gives better stability and accuracy.

Key advantage: naturally parallel, GPU-friendly, no matrix assembly or linear solve. Key disadvantage: ~35% higher CV than FDM at same resolution (numerical dispersion), tau calibration non-trivial.

For bidomain LBM, three architectures are viable (dual-lattice recommended) but none are production-ready.

## Engines
- **LBM V1**: 8 phases complete, 34 tests. D2Q5/D2Q9, BGK/MRT, 3 BC types, `@torch.compile` fused steps.
- **Monodomain V5.4**: Minimal LBM path (D2Q5/D3Q7, BGK only).

## Experiments

| Experiment | Engine | Result | Location |
|-----------|--------|--------|----------|
| Planar wave CV | LBM V1 | CV=75.4 cm/s (D2Q5 BGK) | `Monodomain/LBM_V1/tests/` |
| D2Q5 vs D2Q9 match | LBM V1 | CV match within tolerance | `Monodomain/LBM_V1/tests/` |
| Boundary CV (Neumann) | LBM V1 | Uniform CV (ratio=1.0000), no boundary speedup | `Monodomain/LBM_V1/tests/` |

## Literature
See `literature/` for paper summaries. Key files:
- `LBM_BIDOMAIN.md` (1046 lines) — 3 bidomain architectures, coupling strategies
- `04_LBM_EP_Implementation.md` — LBM-EP algorithm details
- `SUMMARY.md` — Key findings and recommendations

## Connected Research
- **boundary_conduction_speedup** — D2Q9 with Dirichlet BC captures Kleber effect
- **engine_consolidation** — LBM V1 chosen as canonical LBM
