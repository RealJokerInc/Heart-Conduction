---
paper: roberts_2008_discrete_multidomain
title: "Effect of Nonuniform Interstitial Space Properties on Impulse Propagation: A Discrete Multidomain Model"
authors: "Roberts SF, Stinstra JG, Henriquez CS"
year: 2008
journal: "Biophys J"
doi: "10.1529/biophysj.108.137349"
pmid: "18641070"
pmc: "PMC2553133"
pdf: ../papers/roberts_2008_discrete_multidomain.pdf
questions: [Q5]
---

## Key Findings
- Discrete multidomain model with spatially distinct intracellular and extracellular volumes reveals microscale effects on propagation
- CV is relatively insensitive to confining 50% of the membrane by narrow extracellular depths (down to 2nm), as long as some extracellular path remains
- Action potential morphology varies greatly around the fiber perimeter even when CV remains constant
- When tight-space conductivity is sufficiently reduced, the membrane adjacent to the tight space is eliminated from active propagation, and CV actually INCREASES
- Results can be used to determine appropriate tissue-level properties for macroscopic bidomain models

## Method
- Discrete multidomain model: explicitly represents individual cells with separate intracellular, extracellular, and tight-space (cleft) domains
- Dynamic and static boundary conditions electrically couple neighboring spaces
- Systematic variation of extracellular cleft width and conductivity
- Comparison with standard bidomain to bridge microscale and macroscale descriptions

## Key Equations / Results
- Extracellular cleft widths varied from 2nm to bulk extracellular spacing
- CV remains within ~5% even with 50% of membrane confined to 2nm clefts
- When cleft conductivity drops below a critical threshold, the cleft-adjacent membrane becomes passive -> CV increases because the active wavefront has less membrane to charge
- This is a microscale analog of the macroscale Kleber boundary speedup: reducing electrotonic load increases CV

## Connections to Our Models

### Relevant Engine Components
- **Bidomain V1**: Our macroscale bidomain treats sigma_e as a homogeneous parameter. This paper shows that the microscale heterogeneity of the extracellular space can be averaged into effective macroscale conductivities
- **Conductivity sweep** (`experiments/conductivity_sweep.py`): Our sweep varies macroscale sigma_i and sigma_e, which are the effective parameters that this paper's microscale model derives
- **`tests/cv_shared.py`**: Our SIGMA_I=1.74 and SIGMA_E=6.25 mS/cm are macroscale averages; this paper shows what microscale structure produces such averages

### Agreements
- The finding that reducing extracellular loading increases CV is the microscale version of the Kleber boundary speedup we observe at the macroscale
- CV being insensitive to extracellular geometry over a wide range supports our use of homogeneous sigma_e in the macroscale bidomain
- The principle that eliminating membrane from active propagation increases CV is consistent with our boundary speedup mechanism

### Disagreements or Gaps
- We operate entirely at the macroscale bidomain level; the microscale cleft effects are averaged out in our formulation
- Our model cannot capture the AP morphology variation around the fiber perimeter that this paper demonstrates
- The "critical conductivity threshold" for cleft elimination does not have a direct analog in our continuous bidomain

### Actionable Insights
- **MEDIUM**: The microscale justification for the boundary speedup effect strengthens our confidence that the macroscale Kleber effect is physically real, not a numerical artifact
- **LOW**: The effective conductivity derivation could inform how we set sigma_e for different tissue types (atrial vs ventricular, healthy vs diseased)
- **LOW**: If we ever implement a discrete-cell model (e.g., for gap junction effects), this paper provides the multidomain framework

## Limitations / Caveats
- 2D cross-section model (one cell diameter); does not capture 3D tissue-level effects
- Assumes cylindrical cell geometry
- The conductivity thresholds for cleft elimination may be model-dependent
- Does not directly address the macroscale bath-loading effect — the connection to Kleber boundary speedup is by analogy, not direct derivation
