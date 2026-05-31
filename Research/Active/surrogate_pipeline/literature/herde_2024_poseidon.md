---
paper: herde_2024_poseidon
title: "Poseidon: Efficient Foundation Models for PDEs"
authors: "Herde M, Raonic B, Rohner T, Kappeli R, Molinaro R, de Bezenac E, Mishra S"
year: 2024
journal: "NeurIPS 2024"
doi: "arxiv:2405.19101"
pmid: ""
pdf: ../papers/poseidon_2024_herde.pdf
questions: [surrogate_pipeline]
---

## Key Findings
- **Foundation-model scaling for PDEs**: single pretrained model generalizes to **15 diverse downstream PDE tasks** including ones unseen during pretraining.
- **scOT backbone (scalable Operator Transformer)**: multiscale Swin-attention ViT — directly BC-friendly, does not assume periodicity.
- **Time-conditioned layer norms** (adaLN-style): enables continuous-time evaluation for time-dependent PDEs. Similar pattern to DiT / PDE-Transformer.
- **Semi-group property of time-dependent PDEs for data scaling**: elegant augmentation trick — a trajectory x(t₁), x(t₂), ..., x(tₙ) gives not just one training pair but O(n²) pairs by composition. Multiplies effective training data without generating new simulations.
- **Strong out-of-distribution generalization**: pretrained on fluid dynamics, beats task-specific baselines on unseen PDE classes. Validates the foundation-model framing for physics.
- **CamLab (ETH Zurich, Mishra group)**: same lineage as CNO. Consistent methodology; well-maintained code.
- **Open-source**: `github.com/camlab-ethz/poseidon`, pretrained models on HuggingFace `huggingface.co/camlab-ethz`.

## Method
- **scOT architecture**: hierarchical Swin-Transformer with PDE-specific adaptations. Multiscale attention (different patch sizes at different levels) captures both local wavefront structure and global field interactions.
- **Time conditioning**: adaptive layer normalization conditioned on a time embedding — the same token stream evaluates at different times based on the embedding.
- **Semi-group augmentation**: for a PDE solution `u(t)` = S(t) u(0), the semi-group property says `S(t₁+t₂) = S(t₁) ∘ S(t₂)`. Training pairs (u(0), u(t₁)), (u(t₁), u(t₂)), and (u(0), u(t₁+t₂)) are all consistent. Poseidon trains on all of them, getting O(n²) pairs from one trajectory.
- **Pretraining**: large-scale on fluid-dynamics trajectories (compressible Euler, incompressible Navier-Stokes). Pretrained weights released.
- **Fine-tuning**: on 15 downstream tasks across various PDE families. Claims strong generalization even to unseen physics.
- **BC handling**: not explicitly discussed in abstract, but Swin attention is BC-agnostic by construction (no periodicity assumption).

## Connections to Our Models

### Relevant Engine Components
**Scaling / pretraining template for the long-term version of the hybrid bidomain surrogate.** Phase A (single-geometry bidomain) doesn't need foundation-model scale. But Phase B (generalizing across tissue configurations, infarct masks, fiber orientations) is exactly what Poseidon's pretraining approach is designed for.

### Agreements
- **Swin-attention backbone** — agrees with PDE-Transformer and our BC-awareness analysis. Consistent evidence that shifted-window attention is the right choice for non-periodic PDEs.
- **Multiscale hierarchy** — cardiac fields have 100× scale separation (wavefront vs domain). Poseidon's multiscale design handles this directly.
- **Open-source pretrained models** — we can attempt transfer learning rather than training from scratch.
- **Same group as CNO** (Mishra ETH) — consistent methodology; trust-building.
- **Time-conditioned LayerNorm** — if we extend from elliptic-only to full-bidomain time-dependent surrogate, this is the right time-conditioning pattern.

### Disagreements or Gaps
- **Fluid-dynamics pretraining**: Poseidon is pretrained on compressible/incompressible flow. Transfer to reaction-diffusion cardiac EP is plausible but not guaranteed. Our wavefront dynamics are different from turbulent flow.
- **Foundation-model compute cost**: pretraining Poseidon required substantial resources (not stated explicitly, but implicit in the foundation-model framing — likely 1000s of GPU-hours). We can use their pretrained weights; we cannot practically train our own foundation model.
- **Not elliptic-specialized**: Poseidon is general-purpose. For the elliptic solve specifically, UGrid or Lan-2023-style multigrid may outperform.
- **Swin window choice is a hyperparameter**: window size affects receptive field. For bidomain, the wavefront width must fit within window scope or neighboring windows, else cross-window information flow becomes a bottleneck.
- **Semi-group augmentation works for time-dependent PDEs** — the elliptic step we want to learn is time-instantaneous. The semi-group trick doesn't apply to our primary target.

### Actionable Insights
- **MEDIUM — Use Poseidon's pretrained weights as warm-start** when/if we scale to multi-configuration training. Skip pretraining from scratch.
- **MEDIUM — Adopt scOT architecture as an alternative to PDE-Transformer** for the dual-tower if PDE-Transformer's implementation maturity disappoints. Both are Swin-based; PDE-Transformer is more recent but Poseidon has more maturity.
- **MEDIUM — Semi-group augmentation**: applicable if we extend beyond the elliptic-only surrogate to full-trajectory prediction. Not needed for Phase A.
- **LOW — Phase B / foundation-model framing**: preserve the option. If we decide later to build a "bidomain simulator for any tissue config" product, this is the path.
- **LOW — Benchmark transfer learning from Poseidon's checkpoints to our bidomain task** as an experiment. Low-cost, potentially informative.

## Limitations / Caveats
- **Fluid-dynamics-trained backbone**: cardiac EP is distant from Poseidon's training distribution. Transfer may be lossy.
- **Foundation-model compute footprint**: pretraining requires significant resources. We'd use their weights, not reproduce them.
- **General-purpose rather than elliptic-specialized**: UGrid / Lan 2023 may win on the elliptic-only task.
- **Time-conditioning not relevant** for our elliptic-only Phase A target.
- **Swin window size is a hyperparameter** with physics implications — must be tuned per problem.
- **BC handling not explicit in abstract**: need PDF read to verify. Swin is BC-agnostic by design, but boundary-feature injection is an implementation detail.
- **15 downstream tasks don't include cardiac EP** — we'd be out-of-distribution for the published results.
