---
paper: sambelashvili_2004_scroll_wave
title: "Dynamics of virtual electrode-induced scroll-wave reentry in a 3D bidomain model"
authors: "Sambelashvili A, Efimov IR"
year: 2004
journal: "Am J Physiol Heart Circ Physiol"
doi: "10.1152/ajpheart.01108.2003"
pmid: "15371264"
pdf: ../papers/scroll_wave_reentry_boundary_dynamics_2004_sambelashvili.pdf
questions: [Q5]
---

## Key Findings
- Scroll wave filaments tend to attach to air boundaries (no-flux, Neumann) but NOT to bath boundaries (phi_e = 0, Dirichlet)
- Filaments detach from electroporated boundaries (Vm = 0, Dirichlet on Vm)
- Bath presence generates only U-shaped filaments; I-shaped (transmural) filaments were not stable with bath coupling
- U-shaped filaments survived for 1.5mm wall thickness but not for 0.5mm or 3mm, showing a narrow stability window
- Boundary conditions are the primary determinant of reentry type, dynamics, and stability

## Method
- 3D bidomain model with virtual electrode polarization (VEP)
- Scroll wave induction via S1-S2 field stimulation protocol
- Tissue slab of varying thickness (0.5mm, 1.5mm, 3mm)
- Three boundary types compared: air (Neumann), bath (Dirichlet phi_e=0), electroporated (Dirichlet Vm=0)
- Filament tracking and stability analysis

## Key Equations / Results
- Air boundary: n . sigma_i . grad(phi_i) = 0 AND n . sigma_e . grad(phi_e) = 0 (both no-flux)
- Bath boundary: phi_e = 0 (Dirichlet) AND n . sigma_i . grad(phi_i) = 0 (no intracellular flux)
- Electroporated boundary: Vm = 0 (membrane short-circuited)
- I-shaped filaments (transmural, both ends attached) form at air boundaries
- U-shaped filaments (both ends on same surface) form at bath boundaries
- Stability window for U-shaped filaments: ~1.5mm wall thickness only

## Connections to Our Models

### Relevant Engine Components
- **Bidomain V1**: Our `bc_type='bath_tb'` (phi_e=0 at top/bottom) and `bc_type='neumann'` (insulated) correspond exactly to the "bath" and "air" boundary types in this paper
- **`tests/test_phase6c_boundary_cv.py`**: Compares insulated (Config A) vs bath-coupled (Config D), which map to this paper's air vs bath boundaries
- **Boundary speedup analysis** (`research/BOUNDARY_SPEEDUP_ANALYSIS.md`): Our analysis focuses on CV effects; this paper adds the reentry/stability dimension

### Agreements
- The qualitative difference between air (Neumann) and bath (Dirichlet) boundaries is fundamental — our Bidomain V1 correctly distinguishes these two cases
- Bath boundaries (phi_e=0) create fundamentally different electrophysiological behavior than insulated boundaries, consistent with our finding that bath coupling changes CV by ~7-13%

### Disagreements or Gaps
- Our work focuses exclusively on CV and wavefront shape, not on reentry dynamics
- We do not model scroll waves (2D only, no spiral wave studies yet)
- The narrow stability window for U-shaped filaments (only at ~1.5mm) suggests that tissue thickness is a critical parameter we have not explored
- We do not implement electroporated boundary conditions (Vm=0 Dirichlet)

### Actionable Insights
- **MEDIUM**: When extending to 3D reentry studies, the boundary condition choice (bath vs air) will be critical for filament dynamics — our existing bc_type infrastructure supports this
- **MEDIUM**: The Vm=0 (electroporated) BC is a third boundary type we could implement in Bidomain V1 for studying ablation lesion effects
- **LOW**: The 1.5mm stability window for U-shaped filaments could be a validation target for future 3D bidomain simulations

## Limitations / Caveats
- Specific to scroll wave dynamics; may not generalize to all propagation scenarios
- Simple slab geometry without fiber rotation
- VEP-induced reentry is a specific induction mechanism; other induction methods may show different boundary sensitivity
- Only three wall thicknesses tested (0.5, 1.5, 3mm) — the stability window may be broader or narrower
- Does not quantify the CV changes at bath boundaries (focuses on filament topology, not conduction velocity)
