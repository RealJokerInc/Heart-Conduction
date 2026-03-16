---
paper: rossi_2018_thickness_curvature
title: "Muscle Thickness and Curvature Influence Atrial Conduction Velocities"
authors: "Rossi S, Gaeta S, Griffith BE, et al."
year: 2018
journal: "Front Physiol"
doi: "10.3389/fphys.2018.01344"
pmid: "30420809"
pmc: "PMC6215968"
pdf: ../papers/rossi_2018_thickness_curvature.pdf
questions: [Q5]
---

## Key Findings
- Transverse conductivities and boundary conditions speed up or slow down propagation depending on wall curvature
- A planar wavefront propagating parallel to a straight surface normal does NOT remain normal in curved domains
- The 2D manifold assumption (thin-shell model) fails for tissue thicker than 0.5mm with significant curvature
- Full 3D bidomain with explicit bath domain is required to correctly capture curvature-dependent CV changes
- Atrial tissue thickness (~1-3mm) is in the regime where these effects matter

## Method
- Full 3D bidomain equations with explicit bath domain
- Curved tissue geometries (cylinders, spheres) of varying thickness
- Compared 3D volumetric model vs 2D manifold (thin-shell) approximation
- Systematic variation of curvature radius and wall thickness
- Bath-loading boundary conditions on endocardial and epicardial surfaces

## Key Equations / Results
- Critical thickness threshold: ~0.5mm — below this, 2D manifold is adequate
- Curvature creates asymmetric loading: concave side (endocardium) has different electrotonic environment than convex side (epicardium)
- Transverse conductivity becomes important when thickness > 0.5mm because transmural current flow redistributes charge
- CV variation due to curvature can be comparable in magnitude to the bath-loading effect itself

## Connections to Our Models

### Relevant Engine Components
- **Bidomain V1**: Our flat slab geometry avoids curvature effects entirely, which is appropriate for isolating the pure bath-loading effect
- **Triangle merger experiment**: Uses a flat 2D domain, so curvature effects are absent — this is actually a strength for cleanly measuring the Kleber effect
- Our `bc_type='bath_tb'` (bath on top/bottom) creates a geometry analogous to their flat slab limit

### Agreements
- Confirms that bath-loading (boundary conditions) significantly affect CV, consistent with our Kleber effect measurements
- The 3D nature of the effect is consistent with our finding that the bidomain (which couples intracellular and extracellular domains in 3D) is needed to capture boundary speedup

### Disagreements or Gaps
- We work in 2D (no curvature effects), so this paper's main finding about curvature is not directly testable in our current setup
- Our Bidomain V1 does not model explicit tissue thickness in a 3D sense
- The 0.5mm thickness threshold suggests that for realistic atrial simulations, our 2D approach may miss important physics

### Actionable Insights
- **MEDIUM**: When extending to anatomical models, curvature must be considered alongside bath loading — the two effects can reinforce or counteract each other
- **MEDIUM**: The 0.5mm threshold is a useful design criterion: for tissue thinner than this, a 2D manifold + augmented monodomain may suffice
- **LOW**: Consider testing curved geometries in a future 3D extension of Bidomain V1 to quantify curvature-bath interaction

## Limitations / Caveats
- Idealized geometries (cylinders, spheres) rather than anatomically realistic shapes
- Does not provide a simple correction factor for curvature effects (unlike Bishop's R ratio for flat bath loading)
- Atrial-specific results; ventricular wall thickness (~10mm) may show different behavior
- Does not address how curvature interacts with fiber orientation
