# Fetal Heart Development — Knowledge File

> This file is a running synthesis. Updated as findings accumulate.
> When the question is complete, a copy is promoted to `Research/Knowledge/`.

## Current Understanding

This is a literature review question with no active simulation component. The cardiac conduction system develops through a series of morphogenetic events, and understanding these developmental transitions is relevant to how tissue geometry, connectivity, and gap junction expression affect conduction patterns -- the same principles that govern boundary speedup and scar interaction in the adult heart.

### Developmental sequence

1. **Heart tube formation**: Cardiac progenitor cells in anterior lateral mesoderm form a linear heart tube. First heart field (FHF) contributes left ventricle and parts of the atria.
2. **Looping**: The linear tube undergoes rightward looping (dextral), establishing left-right asymmetry. Second heart field (SHF) cells are added progressively at both poles.
3. **Chamber morphogenesis**: Ballooning of chambers from the outer curvature of the looped tube. Trabeculation (finger-like projections) increases surface area for oxygen exchange before coronary vessels form.
4. **Septation**: Division of the common chambers into four-chambered heart. Neural crest cells contribute to outflow tract septation.
5. **Conduction system differentiation**: Specialized conduction cells (SA node, AV node, His bundle, Purkinje network) differentiate from working myocardium, establishing the mature activation sequence.

### Gap junction transitions

Gap junction expression changes dramatically during development, directly affecting conduction velocity and pattern:

| Connexin | Early development | Mature heart | Functional role |
|----------|-------------------|--------------|-----------------|
| **Cx45** | Widespread, dominant early | SA/AV node only | Low conductance (~30 pS), slow conduction |
| **Cx40** | Expressed in developing chambers | Atria, His-Purkinje | High conductance (~180 pS), fast conduction |
| **Cx43** | Appears later in working myocardium | Ventricles (dominant) | Medium conductance (~60 pS), moderate speed |

The developmental transition from Cx45-dominant (slow, uniform conduction) to Cx40/Cx43 (fast, anisotropic conduction) parallels the morphological transition from peristaltic heart tube to coordinated chamber contraction. This transition also mirrors the hiPSC-CM maturation pathway (PHAS13 to MHAS13): immature hiPSC-CMs have low IK1 and high If (resembling embryonic cardiomyocytes), while matured variants gain IK1 and lose If (resembling adult cells).

### Key insight for simulation

The absence of Cx43 at scar-myocardium interfaces (Q6) mirrors the developmental state: scar tissue reverts to an electrically disconnected state analogous to pre-gap-junction embryonic tissue. Understanding how conduction patterns emerge during development (from uniform slow conduction to organized fast conduction) informs how conduction patterns are disrupted by scar tissue (reversion to slow/blocked conduction).

### Extracardiac cell populations

Two critical extracardiac populations contribute to heart development:
- **Neural crest cells**: Migrate from dorsal neural tube to heart; required for outflow tract septation, pharyngeal arch artery remodeling. Their absence causes persistent truncus arteriosus.
- **Epicardium-derived cells (EPDCs)**: From proepicardial organ; contribute to coronary vasculature, myocardial maturation, and fibroblast population. The adult cardiac fibroblasts (relevant to scar formation) are largely EPDC-derived.

### Literature scope

The literature review (fetal_heart_development_literature.md) covers 14 categories with 50+ papers:
1. Overview and comprehensive reviews
2. Heart tube formation and looping
3. Second heart field and cardiac progenitors
4. Cardiac neural crest and outflow tract
5. Conduction system development
6. Signaling pathways (Wnt, Notch, BMP)
7. Epicardium and coronary vessel development
8. Chamber maturation, trabeculation, and valves
9. Congenital heart defects (genetic/molecular basis)
10. Single-cell transcriptomics and atlases
11. Fetal-to-neonatal transition
12. Macrophages, immune cells, and regeneration
13. Other factors (cilia, RNA-binding proteins)
14. Recent preprints

A separate ML-focused review (ml_in_fetal_heart_development.md) covers machine learning applications in developmental cardiology.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Status | Backlog (literature review only) | No simulation component planned currently |
| Scope | Broad developmental biology review | Informs understanding of conduction system formation and gap junction biology |
| Priority | Low | Relevant background knowledge but not directly actionable for current engines |
| Trigger to activate | When boundary speedup is validated in 3D | 3D geometry makes developmental questions more tractable |

## Open Questions

- Could a developmental model of gap junction expression transitions be simulated in V5.4? (Would require spatially/temporally varying conductivity)
- How do trabeculation geometry and Purkinje network formation affect conduction patterns? (Potential future simulation question)
- Is there a quantitative model of Cx45-to-Cx43 transition timing that could be validated against fetal ECG recordings?
- Can the maturation pathway in hiPSC-CM models be validated against fetal development data?
- How do boundary speedup effects (Q5) manifest during developmental tissue growth?

## Connections
- **Engines**: None currently (backlog item)
- **Related research**: boundary_conduction_speedup (Q5 -- same boundary physics applies during tissue growth), scar_bc_validity (Q6 -- gap junction absence at scar mirrors developmental patterns), mature_hipsc_cm_models (hiPSC-CMs recapitulate immature/fetal phenotype)
- **Pipelines**: None
