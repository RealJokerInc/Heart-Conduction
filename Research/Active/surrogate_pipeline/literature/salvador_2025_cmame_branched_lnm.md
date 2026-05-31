---
paper: salvador_2025_cmame_branched_lnm
title: "Full-field surrogate modeling of cardiac function encoding geometric variability"
authors: "Martinez E, Moscoloni B, Salvador M, Kong F, Peirlinck M, Marsden AL"
year: 2025
journal: "arXiv 2504.20479 (targeting CMAME)"
doi: "arxiv:2504.20479"
pmid: ""
pdf: ../papers/salvador_2025_cmame_branched_lnm.pdf
questions: [surrogate_pipeline]
---

> Salvador is third author (senior representation of the broader LFLDNet / BLNM project line at Marsden lab). First author is Martinez. Filename kept as `salvador_2025_cmame_branched_lnm.md` for continuity with prior project references.

## Key Findings
- **Branched Latent Neural Maps (BLNMs)** as a full-field cardiac EP surrogate that explicitly encodes **geometric variability** across patient-specific meshes. Directly targets the "surrogates are geometry-specific" gap cited across cardiac-ML literature.
- **13 pediatric Tetralogy of Fallot patients** used as the geometric base set, plus **52 synthetic geometries** generated via z-score sampling in a statistical shape space built by diffeomorphic atlas mapping. Total training cohort: 65 geometries.
- **MSE = 0.0034** on held-out test geometries. (Normalization range not specified in abstract; proper interpretation requires reading the PDF.)
- **Open-source implementation** (MIT License), making it reproducibility-friendly and potentially a direct adoption target.
- **Physics-based data generation** via "a complex multi-scale mathematical model coupling partial and ordinary differential equations" — i.e., a proper cardiac EP simulator generates ground truth. Model type (monodomain vs bidomain) not disclosed in abstract.

## Method
- **Architecture**: Branched Latent Neural Maps. The branching structure is not specified in the abstract; likely a hierarchical neural-maps design where one branch encodes the geometry (via atlas coordinates) and another encodes the dynamics. Needs PDF read for details.
- **Training data**: 13 patient geometries + 52 synthetic → 65 total. Ground truth from physics-based simulation.
- **Geometry encoding**: diffeomorphic anatomical atlas — the base geometry is a reference mesh, and variation is parameterized via diffeomorphic deformation maps. This is how "geometric variability" enters the input space.
- **Output**: "activation maps" — not full V_m field, just per-node activation times (and presumably repolarization times from the broader BLNM project line).
- **Evaluation**: MSE = 0.0034 on held-out geometries.

## Connections to Our Models

### Relevant Engine Components
**Phase-B relevant, not Phase-A.** Our immediate target (structured-grid 2D Bidomain V1 elliptic surrogate) is on a regular Cartesian grid with no geometric variability. This paper is the load-bearing reference for **if/when we extend the surrogate to patient-specific anatomies** — the geometric-generalization step.

### Agreements
- **Full-field surrogate** as the right framing for cardiac EP (vs. 0D electromechanics like their own LNODE 2024 paper). Aligns with our approach.
- **Supervised from physics-based simulator** — matches our planned pipeline (train on Bidomain V1 output).
- **Open-source code** is an adoption-friendly signal; we should inspect their repo before Phase B planning.

### Disagreements or Gaps
- **Activation maps, not full V_m field over time**: they predict scalar per-node activation times, not the spatiotemporal evolution. Our dual-tower bidomain surrogate needs the full V_m(x, t) and φ_e(x, t) fields. Different output space — architecturally incompatible with our design.
- **Monodomain implied** (not explicit): the paper doesn't call out bidomain, and the BLNM line (including salvador_2025_lfldnet) is all monodomain. Same gap as every other cardiac NN surrogate we've reviewed: **nobody predicts φ_e as a field**.
- **Unstructured mesh + atlas**: their training data is patient-specific unstructured meshes with diffeomorphic deformation. Our Bidomain V1 is structured Cartesian. Bridge not trivial — their architecture design is tailored to the mesh/atlas setup.
- **Small dataset (65 geometries)**: may or may not be sufficient for the real variability in cardiac anatomy. Open question for Phase B.
- **Tetralogy of Fallot cohort specifically**: pediatric congenital heart disease; their generalization to adult hearts, arrhythmia patients, etc. is not tested. Our target domain may be different.

### Actionable Insights
- **LOW (Phase A) — Defer.** Not relevant to our structured-grid elliptic surrogate target. Revisit only when we're confident Phase A works and we're planning patient-specific deployment.
- **MEDIUM (Phase B) — Must-read when extending to patient meshes.** The BLNM + atlas pattern is the cleanest precedent for handling geometric variability in cardiac NN surrogates.
- **LOW — Inspect the open-source repo.** Architecture details are not in the abstract; the repo is the fastest route to understanding what "branched" means.
- **LOW — Cite as state-of-the-art for geometric generalization in cardiac EP surrogates** in our eventual write-up, whether or not we adopt their architecture.

## Limitations / Caveats
- **Abstract-only info at time of filing**. Full PDF read needed to confirm: (a) exact architecture of BLNMs, (b) monodomain vs bidomain, (c) what loss function, (d) what inference cost/speedup.
- **65 geometries** is a small training set for anatomy-generalizing models; claims of robust generalization should be read with caution.
- **Activation-map output** is a lossy projection of the full spatiotemporal EP solution. Errors accumulated during the map construction (upstroke timing in the simulator) are inherited by the surrogate.
- **Pediatric Tetralogy of Fallot**: the specific patient cohort may not be representative of other cardiac disease states.
- **arXiv preprint (April 2025)**, not yet peer-reviewed at CMAME; methodology claims should be treated as draft-quality until the published version is available.
- **Project-line homonyms**: distinct from `salvador_2025_lfldnet` (CfC liquid-NN monodomain TTP06) and `salvador_2024_lnode_cardiac` (0D electromechanics LNODE). Marsden lab has three concurrent surrogate lines — don't conflate them.
