---
paper: centofanti_2025_fno_kol_cardiac
title: "Learning cardiac activation and repolarization times with operator learning"
authors: "Centofanti E, Ziarelli G, Parolini N, et al."
year: 2025
journal: "PLOS Computational Biology"
doi: "10.1371/journal.pcbi.1013920"
pmid: ""
pdf:
questions: [surrogate_pipeline]
---

## Key Findings
- FNO and KOL learn the operator mapping applied stimulus → activation/repolarization time distributions
- Single-shot prediction (NOT autoregressive timestep simulation)
- Both approaches are robust to hyperparameter choices
- Evaluated on 2D/3D synthetic domains and realistic left ventricle geometry

## Method
- **Equations**: Monodomain (comparison model); learned map corresponds to Eikonal model for activation times
- **Architecture**: Two operator learning approaches compared — Fourier Neural Operator (FNO) and Kernel Operator Learning (KOL)
- **Input/Output**: Applied stimulus location → activation time map + repolarization time map
- **NOT autoregressive** — predicts summary statistics (AT, RT) directly, not field evolution over time

## Connections to Our Models

### Relevant Engine Components
- Conceptually different from our approach — they learn a mapping from stimulus to output metrics, not a timestep-by-timestep simulation
- Our surrogate predicts (Vm, phi_e) fields at each dt, enabling rollout of full AP propagation

### Agreements
- FNO is effective for cardiac EP problems — supports our deferred upgrade path (Phase 6: FNO spectral layer for phi_e)
- Operator learning generalizes across conditions

### Disagreements or Gaps
- **Different problem scope**: They predict activation/repolarization times (scalar maps), not full spatiotemporal field evolution. Our surrogate must predict the full (Vm, phi_e) field at every timestep.
- **No ionic dynamics**: Their approach bypasses ionic modeling entirely. Our Ionic Transformer explicitly models ionic state.
- **Monodomain only**: No bidomain, no phi_e.

### Actionable Insights
- **FNO for phi_e**: Their success with FNO on cardiac EP supports our Phase 6 upgrade path (FNO spectral layer for phi_e if cross-skip ResNet has insufficient receptive field). Priority: medium (Phase 6).
- **Could be complementary**: Their stimulus→AT/RT mapping could serve as a fast pre-filter in optimization, with our full surrogate used for detailed validation. Priority: low (future).

## Limitations / Caveats
- Not a timestep-by-timestep simulator — cannot produce AP waveforms or phi_e fields
- Single-shot prediction limits applicability to tasks requiring full field evolution
