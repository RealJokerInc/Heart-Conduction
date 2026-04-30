# Boundary Conduction Speedup (Kleber Effect)

## Question
Does conduction velocity increase at inert tissue boundaries? By how much, and under what conditions?

## Status: Active

## Why It Matters
The Kleber boundary speedup is a real biophysical effect: at a bath-perfused tissue boundary, the extracellular return path is short-circuited by the bath, reducing electrotonic loading and increasing CV by 7-13%. This affects any simulation of tissue with free boundaries, scar borders, or bath-coupled surfaces. Getting the boundary physics wrong confounds CV measurements, optimization objectives, and arrhythmia predictions.

## Engines
- **Bidomain V1**: Primary validation engine. CV ratio = 1.0714 at dx=0.025cm, converging toward theoretical 1.131.
- **LBM V1**: Secondary confirmation via D2Q9 with Dirichlet BC. Captures speedup despite ~35% CV baseline offset from numerical dispersion.
- **Monodomain V5.4**: Control — no boundary speedup (expected, since monodomain lacks two-domain BC asymmetry).

## Completion Criteria
- [x] Isotropic CV ratio measured and validated against theory (1.131)
- [x] Mesh convergence study (CV ratio → 1.131 as dx → 0)
- [x] Monodomain control confirms no speedup (under standard 5-pt cardinal)
- [x] LBM independently validates the effect
- [x] Triangle merger wavefront characterization (5pt vs Mehrstellen)
- [x] Conductivity sweep (edge lead ~ 1/sqrt(D_eff))
- [x] **Connectivity-mediated boundary deficit reproduced in monodomain (2026-04-30)** — Moore-8 stencil + face_mirror BC produces +486 µs LAT shift in TTP06 EPI line-stim. Eliminated by cardinal-only OR face_mirror_iso (LBM bounce-back analog).
- [x] **Cross-engine bridge claim confirmed (2026-04-30)** — same connectivity mechanism in storage tank, monodomain V5.4, LBM V1. Cardinal-only or iso+bounce-back fixes it in all three.
- [ ] Anisotropic boundary study (fiber-parallel vs perpendicular)
- [ ] 3D validation
- [ ] Tissue thickness study (when boundary layer spans full thickness)

## Sub-Questions

| Sub-Question | Status | Key Finding |
|-------------|--------|-------------|
| Isotropic CV ratio | Complete | 1.0714 at dx=0.025, converges to 1.131 |
| Triangle merger wavefront | Complete | Bath-coupled boundaries produce triangular wavefront merger |
| Stencil comparison | Complete | Mehrstellen sharper than 5pt, same CV ratio |
| Conductivity sweep | Complete | Edge lead ~ 1/sqrt(D_eff); 4x sigma_i increases Kleber ratio |
| Connectivity-mediated boundary deficit | **Complete (2026-04-30)** | Moore-8 stencil + face_mirror reproduces John's crescent in monodomain (+486 µs LAT). Bridge claim confirmed across storage-tank, monodomain V5.4, LBM V1. |
| Anisotropic boundaries | Active | Testing fiber-parallel vs perpendicular effects |

## Experiments

| Experiment | Engine | Result | Location |
|-----------|--------|--------|----------|
| Triangle merger | Bidomain V1 | Triangular wavefront, 3 stencil configs | `Bidomain/Engine_V1/experiments/triangle_merger/` |
| Anisotropic test | Bidomain V1 | 2:1 anisotropy → sharper triangle (eikonal confirmed) | `Bidomain/Engine_V1/experiments/anisotropic_test/` |
| Conductivity sweep | Bidomain V1 | 5 configs, 1/sqrt(D_eff) scaling confirmed | `Bidomain/Engine_V1/experiments/conductivity_sweep/` |

## Literature
See `literature/` for paper summaries. Key references:
- Bishop 2011 (augmented monodomain, bath loading)
- Rossi 2018 (thickness, curvature, atrial CV)
- Patel & Roth 2005 (overdetermined BC, analytical solution)
- Johnston 2008 (approximate bidomain, boundary layer)

## Engine References

Files to read when resuming work on this question:

| File | What it tells you |
|------|-------------------|
| `Bidomain/Engine_V1/REVIEW.md` §6.5 | Diffusion tensor encoding, FDM stencil comparison |
| `Bidomain/Engine_V1/PROGRESS.md` | Current engine status, Phase 6 boundary CV tests |
| `Bidomain/Engine_V1/cardiac_sim/simulation/classical/discretization/fdm.py` | Face-based stencil implementation |
| `Bidomain/Engine_V1/tests/test_phase6c_boundary_cv.py` | Boundary CV validation tests |
| `Bidomain/Engine_V1/cardiac_sim/simulation/classical/solver/diffusion_stepping/decoupled.py` | Decoupled GS solver (parabolic + elliptic) |
| `LBM/Engine_V1/src/collision/bgk.py` | LBM collision with source term |
| `LBM/Engine_V1/PROGRESS.md` | LBM engine status |
| `Research/Knowledge/bidomain_simulation.md` | Discretization and solver knowledge |
| `Research/Knowledge/lbm_cardiac.md` | LBM knowledge (D2Q5/D2Q9, MRT) |
| `Research/Complete/scar_bc_validity/KNOWLEDGE.md` | Scar BC analysis (contrast with bath-coupled) |

## Future Work
{No deferred items yet.}

## Connected Research
- **scar_bc_validity** — Neumann at scar means NO speedup there (contrast with bath-coupled)
- **lbm_cardiac** — LBM with D2Q9 Dirichlet BC captures the effect
- **engine_consolidation** — Face-based FDM stencil required for correct boundary treatment
