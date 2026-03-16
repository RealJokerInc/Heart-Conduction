# Q5: Does conduction velocity increase at inert tissue boundaries?

## Short Answer

**Yes.** At an insulated (no-flux) tissue boundary, the wavefront has fewer downstream cells to charge, reducing the electrotonic load. This increases the safety factor for propagation, which increases conduction velocity. The effect was first characterized by Kleber et al. and is on the order of 7-13% faster CV near the boundary, converging to the theoretical ratio of ~1.131 as mesh resolution increases.

This is a real biophysical effect, not a numerical artifact. It has been confirmed experimentally in optical mapping studies and computationally across FDM, FEM, and LBM.

Our Bidomain Engine V1 reproduces this: CV ratio = 1.0714 at dx=0.025, converging toward 1.131 with mesh refinement.

## Key Files in This Folder

| File | Contents |
|------|----------|
| `CARDIAC_BOUNDARY_CONDUCTION_BIBLIOGRAPHY.md` | 21 core papers, organized by topic |
| `ADDITIONAL_RELATED_PAPERS.md` | 30+ supplementary papers |
| `QUICK_REFERENCE.txt` | Executive summary of findings |
| `Experimental_Validation.md` | Proposed validation experiments |
| `Infarct_Boundary_Speedup_Analysis.pdf` | Full analysis document |
| `proofs/No_Flux_BC_Proof.md` | Mathematical proof of no-flux BC |
| `papers/` | Download metadata and access recommendations for 50+ papers |

## Connected Questions

- **Q6** — What BC to use at scar boundaries (Neumann, not Dirichlet)
- **Q1** — Boundary discretization must be face-based to correctly capture this effect
- **Q4** — LBM with D2Q9 Dirichlet BC can simulate reduced edge loading
