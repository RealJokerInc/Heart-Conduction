---
paper: lydon_2025_pino_cardiac
title: "Physics-Informed Neural Operators for Cardiac Electrophysiology"
authors: "Lydon H, Kazemi M, Bishop M, Paoletti N"
year: 2025
journal: "arXiv preprint"
doi: "arxiv:2511.08418"
pmid: ""
pdf:
questions: [surrogate_pipeline]
---

## Key Findings
- First neural operator (PINO) applied to cardiac EP — FNO backbone with physics constraints
- Autoregressive rollouts over extended time horizons
- Generalizes to 10x training resolution (zero-shot upscaling)
- Significant reduction in simulation time vs numerical PDE solvers

## Method
- **Equations**: Cardiac EP PDEs (monodomain implied, not bidomain)
- **Architecture**: Physics-Informed Neural Operator (PINO) — FNO backbone with soft physics constraints (PDE residual loss)
- **Autoregressive**: YES — predictions recursively fed back as inputs for long rollouts
- **Key distinction from PINNs**: PINO learns mappings between function spaces (generalizes across resolutions/ICs), whereas PINNs learn a single solution

## Connections to Our Models

### Relevant Engine Components
- Their FNO backbone is relevant to our Phase 6 upgrade path for phi_e
- Physics constraints (PDE residual loss) could inform our Stage C end-to-end fine-tuning

### Agreements
- Autoregressive rollout with neural operators works for cardiac EP
- Resolution generalization (10x upscaling) is achievable with operator learning

### Disagreements or Gaps
- **Physics constraints**: They use PDE residual loss (PINN-adjacent). Our approach is purely data-driven — the physics is encoded in the training data from Bidomain V1, not in the loss function.
- **No operator splitting**: Monolithic architecture, no ionic/diffusion separation.
- **Monodomain only**: No bidomain, no phi_e.

### Actionable Insights
- **Resolution generalization**: Their 10x zero-shot upscaling is impressive. If we want variable grids later, FNO is the path. Priority: low (deferred).
- **Physics-informed loss**: Could consider adding a PDE residual term to Stage C fine-tuning if purely data-driven training produces poor generalization. Priority: low (only if needed).
- **GitHub available**: Code at `github.com/janet-9/CardiacEP-PINOS`. Could reference their FNO implementation. Priority: medium.

## Limitations / Caveats
- PINN-adjacent (user specifically excluded PINNs from search, but this is operator-based not PINN)
- Monodomain only
- Specific accuracy metrics and speedup factors not detailed in abstract
