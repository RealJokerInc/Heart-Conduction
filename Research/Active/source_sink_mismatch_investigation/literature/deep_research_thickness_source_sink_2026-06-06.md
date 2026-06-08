# Deep-Research Report — Thickness-Driven Source-Sink Curvature & the Augmented Monodomain Fix
**Date:** 2026-06-06 · **Method:** deep-research harness (6 angles, 21 sources fetched, 100 claims extracted, 25 adversarially verified → 24 confirmed / 1 killed) · run `wf_8ef309b4-7d4`

## Verdict (summary)
The augmented/thickness-weighted monodomain `∂V/∂t = (1/T)∇·(T·D∇V) − I_ion/Cm` IS the correct reduced model to reproduce the Ciaccio Fig-4 source-sink wavefront-curvature artifact — with an attribution correction. **Biktasheva/Dierckx/Biktashev (arXiv:1408.3654, PRL 2019)** rigorously DERIVE exactly this form as the leading-order O(μ²) 2-D reduction of 3-D RD in a thin layer with no-flux top/bottom (H = local thickness), validate it against full 3-D simulation, and show the thickness-gradient term = `D·∇(ln H)·∇u` = `∇T/T` — precisely Ciaccio's phenomenological eikonal `θ=θ₀−D(∇T/T)`. **Bishop & Plank's "augmented monodomain" (2011) is a DISTINCT tool** (bath-loading edge-conductivity modifier; ~3 boundary elements × R_ζ), NOT a thickness field — same name, different mechanism. Regime: the varying dimension must be thin vs the electrotonic space constant (the O(μ²) thin-layer limit).

## Confirmed findings (tag: theory / simulation / experiment)

1. **[theory+sim] The thickness-weighted monodomain is the rigorously-derived reduced model.** Biktasheva/Dierckx/Biktashev (arXiv:1408.3654, PRL 2019) Eq.(4): `u_t = f(u) + D(1/H(x,y))∇·(H(x,y)∇u) + O(μ²)`, thin layer z∈[z_min,z_max], H≡thickness, no-flux boundaries. For constant D ≡ `(1/T)∇·(T·D∇V)`. Source: arxiv.org/pdf/1408.3654. (Caveat: generic FHN/Oregonator kinetics, scroll-wave drift — not cardiac TTP06 / not the Ciaccio block artifact.)
2. **[theory] The thickness-gradient term is a log-thickness coupling.** Eq.(5): `h = D(∇K)·∇u, K=lnH` → `∇K=∇T/T`. Independently checkable: `(1/H)∇·(H∇u)=∇²u+(∇lnH)·∇u`. Rigorous origin of Ciaccio's ∇T/T.
3. **[sim] 3-D validation.** Authors compare the 2-D thickness-reduced system (Eq.4) vs full 3-D (Eq.1) for FHN + Oregonator (BeatBox); "quantitatively confirmed by direct numerical simulations" for a thickness step. (Caveat: non-cardiac, scroll-wave drift, not the CV/block artifact.)
4. **[theory] Bishop & Plank distinct.** PMC3075562: bath-loading curvature via current shunted into a conducting bath; implemented as `g_aug=g_dft·R_ζ`, `R_ζ=g_b(g_iζ+g_eζ)/(g_eζ(g_iζ+g_b))` on ~3 edge elements; "less dependent upon the thickness of the surrounding bath" — a conductivity/interface phenomenon, NOT a thickness field.
5. **[theory] Ciaccio eikonal relation.** Ciaccio 2018 (PMC5874259): `θ=θ₀−DΔT/(c·T)`, θ₀=0.4 mm/ms, **D=0.2 mm²/ms**; Ciaccio 2015 (PMC4533242) identical form with **D=0.1 mm²/ms**. (Authors' own simplified eikonal estimate — a model, corroborated against canine mapping.)
6. **[theory/sim] Block threshold.** D=0.2 → block at `ΔT/T≈2` per mm (400→1200 µm over 1 mm gives θ=0); D=0.1 → `ΔT/T≈4`. Canine empirical ~1.55 per mm ≈ predicted 2.
7. **[theory+exp+sim] Fig-4 wavefront shape law.** "concave + speeds up when distal volume lesser; rectilinear when no change; convex with slowing or block when distal volume greater" — source-sink (insufficient source current to charge larger sink).
8. **[theory] Generalizes to any volumetric change** — fibrosis, fiber-bundle discontinuity — not only post-infarction scar (authors' theoretical extension).
9. **[experiment] Canine foundation.** Ciaccio 2007 (PMC2626544): isthmus IBZ 231±140 µm vs outer 1440±770 µm (~6×, p<0.001); CV slower at entrance/exit (0.32±0.05 vs 0.42±0.13 mm/ms); fastest CV at minimum thickness gradient; block lines coincide with sharp thin→thick; model predicts circuit features at 75% sens / 97% spec.
10. **[experiment] Electrogram fractionation** at IBZ lateral boundaries from wavefront discontinuity, depending on rate and orientation (Ciaccio 2014, 10.1161/CIRCEP.113.000840).
11. **[sim, ORTHOGONAL] In-plane source-sink curvature exists too** — Romero/Trenor/Ferrero/Starmer 2013 (PLoS ONE 10.1371/journal.pone.0078328): curvature from non-uniform cellular source-sink dispersion along the wavefront, in a STANDARD uniform-conductivity 2-D monodomain (TTP06), no thickness weighting. Confirms general curvature mechanism but does NOT validate the thickness fix.

## Refuted (0-3)
- A variant of the eikonal claim transcribing D's units as "mm/ms" — D is a diffusion coefficient, **mm²/ms**.

## Caveats
- **Parameter discrepancy:** 2018 paper D=0.2 (thr≈2) vs 2015 paper D=0.1 (thr≈4); both internally consistent — cite the specific paper.
- **Validation gap (the project's opening):** the thickness reduction is derived + 3-D-validated only in FHN/Oregonator scroll-wave drift, NOT against the Ciaccio cardiac CV/block artifact with a physiological ionic model. No source runs thickness-weighted monodomain + cardiac ionics to reproduce Fig-4.
- **Ciaccio eikonal is a model**, not first-principles; corroborated against canine experiment.
- **Scholarly debate** (JACC Clin EP 2023; Heart Rhythm 2021) on thickness vs fibrosis/scar/fat as the dominant isthmus determinant — about relative weighting, not whether block lines sit on thin→thick transitions (they do, in canine).

## Open questions
1. Implement thickness-weighted monodomain with TTP06/ORd → reproduce Fig-4 A–D + block, validated vs full 3-D? (the gap / likely original contribution)
2. Adopt 2018 (D=0.2, thr≈2) or 2015 (D=0.1, thr≈4)?
3. Precise μ (thickness / space-constant) threshold where the thin-layer reduction breaks?
4. Mesh-resolution pitfalls at a sharp thickness transition (does under-resolution spuriously create/suppress block)?

## Primary sources
- Ciaccio 2018 JACEP — DOI 10.1016/j.jacep.2017.08.019 (PMC5874259)
- Ciaccio 2015 Comput Biol Med — PMC4533242
- Ciaccio 2007 Heart Rhythm — PMC2626544
- Ciaccio 2014 Circ Arrhythm Electrophysiol — DOI 10.1161/CIRCEP.113.000840
- Biktasheva/Dierckx/Biktashev — arXiv:1408.3654 (PRL 2019)
- Bishop & Plank 2011 — PMC3075562 (bath-loading; the DISTINCT tool)
- Romero/Starmer 2013 — DOI 10.1371/journal.pone.0078328
- Goldstein & Rall varying-cross-section cable lineage (PubMed 7585837, 7606760, 7525101)
- Non-uniform source-sink dispersion — PMC3817246
