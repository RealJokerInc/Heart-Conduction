---
paper: bishop_plank_2011_augmented_monodomain
title: "Representing Cardiac Bidomain Bath-Loading Effects by an Augmented Monodomain Approach: Application to Complex Ventricular Models"
authors: "Martin J. Bishop, Gernot Plank"
year: 2011
journal: "IEEE Transactions on Biomedical Engineering"
doi: "10.1109/TBME.2010.2096425"
pmid: "21292591"
pmcid: "PMC3075562"
pdf: null
questions: [bidomain_parabolic_parabolic, boundary_conduction_speedup]
---

## Key Findings

- **Bath-loading produces V-shaped wavefront curvature** at tissue–bath interfaces in the bidomain but is absent from monodomain. The bath shunts extracellular current, reducing effective extracellular resistance locally and increasing tissue-edge CV by 11–48% depending on the conductivity set.
- **Augmented monodomain (MDMEQ) recovers bidomain wavefront curvature with ~7× speedup.** The trick: tag a thin layer of elements at the tissue–bath interface and multiply their conductivity eigenvalues by a ratio `R = (σ_i·σ_bath)/(σ_i + σ_bath) / (σ_i·σ_e)/(σ_i + σ_e)` (the ratio of edge-effective to bulk-effective bulk conductivity). Activation times match bidomain to within 0.5–7% across standard conductivity sets.
- **Bath-loading is highly sensitive to which conductivity set is used.** Clerc (1976) gives R ≈ 1.06–1.12 (small curvature). Roberts & Scher (1982) gives R ≈ 1.15–1.48 (pronounced V-shape). Roberts (1979) is intermediate. The huge spread in literature conductivity values means bath-loading magnitude is highly uncertain.
- **Bath-loading happens even for thin baths.** Even 0.1 mm of surrounding fluid produces significant wavefront curvature. Most of the effect saturates by ~1 mm bath thickness.
- **Bath-loading happens for any bath conductivity above ~0.5 S/m.** Blood (~0.6) and Tyrode's (~1.0–2.0) both fall on the plateau, so in-vivo and ex-vivo simulations give similar curvature.

## Method

- **Governing equations**: Standard parabolic-elliptic bidomain + monodomain (the paper is not about PP vs PE — it's about bridging them efficiently).
- **Augmented conductivity prescription**: For the 3 tissue elements nearest the bath interface (or 2 in high-resolution meshes), scale the monodomain conductivity eigenvalues by `R` along the fiber direction and the cross-fiber direction separately. Computed from the standard conductivity formulas, no tuning needed.
- **Geometry**: 3D slab (3.0 × 0.01 × 1.0 cm) and a 4M-node MRI-derived rabbit ventricular mesh.
- **Ionic**: rabbit ventricular cell model (Mahajan).
- **Software**: CARP simulator (Cardiac Arrhythmia Research Package).

## Key Equations / Results

- **Edge conductivity ratio:**
  ```
  σ_edge^ζ ≈ (σ_i^ζ · σ_bath^ζ) / (σ_i^ζ + σ_bath^ζ)     (one-sided bath shunt)
  R^ζ = (σ_i^ζ · σ_bath^ζ)/(σ_i^ζ + σ_bath^ζ)  /  (σ_i^ζ · σ_e^ζ)/(σ_i^ζ + σ_e^ζ)
      = (σ_bath · (σ_i + σ_e)) / (σ_e · (σ_i + σ_bath))
  ```
- **Conduction velocity ratios (BDM/BDMNB, Roberts & Scher 1982):**
  - Fiber direction: 1.48 (48% speedup)
  - Cross-fiber: 1.15 (15% speedup)
- **MDMEQ accuracy on rabbit ventricle**: total activation time 78.7 ms vs BDM 76.1 ms (3.4% slow).

## Connections to Our Models

### Relevant Engine Components

- **Bidomain V1** (`Bidomain/Engine_V1/`): the bath-loading analysis in this paper is the direct biophysical basis for our Kleber boundary speedup research. `boundary_conduction_speedup/` uses the same ratio R as its theoretical prediction.
- **Monodomain V5.4** / **LBM V1**: MDMEQ is a recipe for reproducing bidomain boundary effects with a cheaper engine — relevant if we want to validate LBM boundary behavior against a "bidomain-equivalent" monodomain baseline.

### Agreements

- The edge-conductivity formula matches the Kleber derivation we use in `boundary_conduction_speedup/KNOWLEDGE.md`. Our 1.0714 observed ratio at dx=0.025 (Kleber MEMORY note) is in the ballpark predicted by the conductivity-set-dependent ratios Bishop reports.
- The thin-bath saturation is consistent with Bidomain V1's observation that bath thickness beyond ~0.1 cm has minimal effect on CV.

### Disagreements or Gaps

- **MDMEQ does NOT model finite extracellular propagation.** It still uses the instantaneous-elliptic extracellular equation — it just tweaks local conductivity to fake the curvature. It cannot predict **temporal** differences in wavefront shape at the tissue–bath interface, only the steady-state spatial curvature.
- **The V-shape itself is not a wave; it is a conductivity gradient.** The curvature in MDMEQ arises because faster tissue leads bulk tissue — but both are solved with the same elliptic φ_e. This is precisely the artifact we want to test with finite-propagation models (ε-regularization or Cattaneo hyperbolic).

### Actionable Insights

- **High priority: MDMEQ is the PE-world ceiling to beat.** If our hyperbolic bidomain at τ_i ≠ τ_e produces a *smoother* (non-triangular) wavefront than MDMEQ, that's evidence the artifact is real and the dual-evolving formulation fixes it. If the wavefront still looks V-shaped, the artifact might be genuine physics (boundary shunting) rather than numerical-discretization-caused, and the PP hyperbolic correction is cosmetic.
- **High priority: Use this paper's edge-conductivity formula as the null hypothesis.** Any finite-propagation experiment should first verify the standard PE bidomain reproduces Bishop's R values at the tissue edge — then compare how the hyperbolic/PP result deviates.
- **Medium priority: Roberts & Scher conductivity set gives the strongest signal.** Use R ≈ 1.48 as the test case; the effect is too small to measure reliably with Clerc 1976.
- **Medium priority: MDMEQ could serve as a "bidomain-in-LBM" shortcut.** If we want to validate LBM bidomain boundary behavior without a full LBM bidomain implementation, we can run LBM monodomain with locally scaled edge conductivity and compare to a reference bidomain run. This would be a cheap sanity check before the full dual-lattice LBM build.

## Limitations / Caveats

- **Steady-state spatial effect only.** MDMEQ cannot reproduce time-dependent boundary phenomena (virtual electrode, defibrillation shocks).
- **Requires knowing the bath conductivity.** For unusual preparations the R formula needs per-element adjustment.
- **Does not extend to anisotropy rotation or orthotropic (fiber+sheet+normal) tissue cleanly.** The paper extrapolates but does not validate.
- **Uses a quasi-static elliptic extracellular.** Inherits the infinite-propagation assumption of standard bidomain — the thing we are trying to move past.
