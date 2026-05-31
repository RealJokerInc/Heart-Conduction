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
| [rossi_2017_hyperbolic_bidomain](literature/rossi_2017_hyperbolic_bidomain.md) ([PDF](papers/rossi_griffith_2017_hyperbolic_bidomain.pdf)) | Rossi & Griffith, Chaos 27:093926 | **THE key paper.** Cattaneo-type flux replaces Ohm's law → hyperbolic bidomain. τ_e ≠ τ_i is the regime where φ_e equation becomes genuinely non-elliptic (what Rossi's code calls `ParabolicParabolicHyperbolic`). CV *increases* with small τ in nonlinear ionic models. Only one bidomain figure ever published. |
| [colli_franzone_2013_coupled_uncoupled](literature/colli_franzone_2013_coupled_uncoupled.md) ([PDF](papers/esaim_m2an_2013_coupled_uncoupled_bidomain.pdf)) | Colli Franzone, Pavarino, Scacchi, ESAIM M2AN 47:1017 | Uncoupled (Gauss-Seidel) PE bidomain is 2.5–3× faster than coupled with equal accuracy. Reference for why Bidomain V1's decoupled GS architecture is well-founded — and why that architecture survives the move to hyperbolic formulation. |
| [bishop_plank_2011_augmented_monodomain](literature/bishop_plank_2011_augmented_monodomain.md) (PDF via [PMC3075562](https://pmc.ncbi.nlm.nih.gov/articles/PMC3075562/)) | Bishop & Plank, IEEE TBME 58(4):1066 | Bath-loading produces V-shaped wavefront curvature at tissue–bath interface — reproducible with a monodomain that scales edge-element conductivities. 11–48% CV speedup at the edge (conductivity-set dependent). This is the PE-world ceiling: if our hyperbolic bidomain produces a smoother wavefront than MDMEQ, the dual-evolving formulation is fixing the artifact. |
| Bourgault, Coudière, Pierre 2009 (PDF **pending** — HAL behind anti-bot wall) | Nonlinear Analysis: RWA 10:458 | PP well-posedness via "bidomain operator." Establishes that the literature "parabolic-parabolic bidomain" is just a reformulation of the same physics as PE — same degenerate system, different variables. |
| Colli Franzone, Pavarino, Scacchi 2011 (paywalled, SIAM JSC) | SIAM J. Sci. Comput. 33(4) | Parallel Schwarz preconditioners for both PP and PE formulations. Reports PP block system is harder to precondition (not scalable); PE is preferred numerically. |
| Bendahmane & Karlsen 2006 | Uses ε-regularization as a **proof technique** only (ε → 0 recovers true bidomain). Not a computational method. Frequently mis-cited as if it were a numerical scheme. | |
| **[PNP Landscape Survey](literature/PNP_LANDSCAPE_SURVEY.md)** (2026-04-23) | Survey of PNP-cardiac literature — 10+ papers categorized | Homogenized PNP-bidomain exists (Okada 2013, Whiteley 2020). Tissue-level V_c remains quasi-static after homogenization; dynamical fields are ion concentrations per species per compartment. Cell-resolved PNP (Mori 2008, Jæger 2023, Tveito group) preserves Debye-layer physics at nm scale. No published LBM-PNP for cardiac. |
| Whiteley 2020, Math Med Biol 37:262 ([PDF](papers/whiteley_2020_bidomain_debye_homogenization.pdf)) | PNP + Debye-layer microscale → homogenize to tissue | Standard PE bidomain IS a valid homogenization of PNP in normal cardiac conditions. Gives explicit Debye-layer-to-C_m derivation and names conditions where the homogenization fails. |
| Okada, Sugiura, Hisada 2013, Phys Rev E 87:062701 (paywalled — pending acquisition) | "Rational bidomain" derived from NP + electroneutrality + homogenization | Cardiac-specific homogenized PNP-bidomain. Addresses the self-contradiction in standard bidomain re: ion-concentration updates. **Single most relevant paper; needs institutional access.** |
| Mori, Fishman, Peskin 2008, PNAS 105:6463 (PMC2359793, abstract only) | Full PNP in 3D cardiac strand, cell-resolved | Foundational cardiac PNP paper. Demonstrates Debye-layer and ion-concentration physics matter for ephaptic conduction. Cell-resolved, not homogenized. |
| Jæger-Ivanović-Kučera-Tveito 2023, PLOS Comp Biol | Nano-scale PNP in two neighboring cardiac cells | Intercellular electrochemical waves via PNP explain ephaptic conduction at low gap-junctional coupling. Cell-resolved, nm-scale, ns-timestep. |
| Jæger-Tveito 2023, npj Sys Bio Apps (KNM) | Kirchhoff-network cell-based cardiac | Efficient cell-based EP without full PNP. Each myocyte a network node. Not PNP; useful comparison. |
| Tveito group 2022, eNeuro (KNP-EMI validation) | KNP-EMI applied to cortical spreading depression | Framework = electroneutral Nernst-Planck + cell-resolved EMI. Tveito-school answer to "PNP on cells affordably." Applicable to cardiac. |
| Pods-Schönke-Bastian 2013, Biophys J 105:242 | 3D PNP for a neural axon | Methodological template for PNP finite-element codes. Explicit Debye-layer resolution. Neural, not cardiac, but transferable. |
| Pods 2017, J Integr Neurosci | VC vs PNP vs EN model comparison | Three-way comparison of extracellular potential models for neurons. No cardiac equivalent exists — a parallel study for cardiac would be a contribution. |

## Related Research
- [Boundary conduction speedup](../boundary_conduction_speedup/) — where the triangular artifact was first observed; cross-reference target for validation

## Future Work
{No deferred items yet.}
