---
paper: holzschuh_2025_pde_transformer
title: "PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations"
authors: "Holzschuh B, Liu Q, Kohl G, Thuerey N"
year: 2025
journal: "ICML 2025"
doi: "arxiv:2505.24717"
pmid: ""
pdf: ../papers/pdetransformer_2025_tumpbs.pdf
questions: [surrogate_pipeline]
---

## Key Findings
- **The architectural template for our dual-tower-with-cross-talk design.** PDE-Transformer embeds "different physical channels individually as spatio-temporal tokens, which interact via channel-wise self-attention" — this is structurally the same as our Vm-tower + φ_e-tower cross-communication proposal.
- **Shifted-window attention (Swin-style)** gives O(N) complexity and natively handles non-periodic BCs because translational invariance is never assumed. Directly sidesteps FNO's periodic-BC trap.
- **Pre-training + fine-tuning demonstrated across 16 PDE types**. Shows that a single transformer backbone can transfer across PDE classes — relevant if we later want to train across multiple tissue configurations or extend to ORd/Mitchell-Schaeffer ionic models.
- **"Consistent token information density"** across different PDE types: when learning multiple physics channels simultaneously, the tokens stay balanced. Avoids the "one channel dominates" failure mode.
- **Integrates diffusion transformer advancements** — inherits the training stability improvements from the DiT line (time-conditioned adaLN, etc.).
- **Code available**: `github.com/tum-pbs/pde-transformer`. TUM Physics-based Simulation group (Thuerey lab).

## Method
- **Architecture**: enhanced transformer built on diffusion transformer (DiT) foundations. Key physics-specific modifications:
  1. **Per-channel tokenization**: each physical field (V_m, φ_e, D_i, D_e in our case) gets its own token stream over spatial grid positions. Contrast with standard vision transformers where all channels collapse into one RGB-style input.
  2. **Channel-wise self-attention**: tokens from different physical channels attend to each other, enabling explicit coupling between fields.
  3. **Spatial attention via shifted windows** (Swin-style): non-periodic BC friendly, O(N) cost.
  4. **Time-conditioned layer norms** (adaLN from diffusion transformers): for time-dependent PDEs, enables continuous-time evaluation.
- **Training**: pre-train on a broad PDE corpus, fine-tune on downstream tasks. Both multi-PDE pre-training and single-task training demonstrated.
- **Benchmarked**: 16 different PDE types from a held-out benchmark. Outperforms SOTA CV transformers (Swin, DiT) adapted to PDE tasks.
- **BC handling**: not explicitly detailed in the abstract, but shifted-window attention inherently does not assume periodicity. Boundary features are injected via padding / indicator channels (details in the paper).

## Connections to Our Models

### Relevant Engine Components
**This is the highest-priority architectural reference for the dual-tower bidomain surrogate design.** The "channels-as-tokens with channel-wise self-attention" pattern is essentially what we're building, but framed more elegantly and with an existing open-source implementation. Rather than inventing our own cross-talk mechanism, we can adopt this pattern.

### Agreements
- **Channels as independent token streams**: V_m and φ_e get separate tokens, not fused into one feature map. Matches our dual-tower intuition perfectly.
- **Channel-wise attention for cross-coupling**: the cross-talk mechanism we've been sketching.
- **Shifted-window spatial attention**: O(N) cost, non-periodic BC-friendly. The scan's consensus choice for our needs.
- **DiT-derived stability**: inherits time-conditioned LayerNorm, scale-shift modulation, and training stability improvements from the diffusion-transformer line.
- **Open-source code**: direct adoption path. Not reinventing.

### Disagreements or Gaps
- **Not explicitly elliptic-focused**: PDE-Transformer is a general physics-simulation backbone. Its strengths for time-dependent parabolic/hyperbolic may not all transfer to the instantaneous elliptic solve we need. Specifically, the time-conditioned layer norm is moot for a one-shot elliptic solve.
- **BC handling underspecified in the abstract.** Need to read the PDF for details. Swin windows are BC-friendly in principle, but specific injection of Neumann/Dirichlet indicator fields needs verification.
- **Quadratic-in-channel-count self-attention**: if we have many physical channels (V_m, φ_e, D_i_xx, D_i_yy, D_i_xy, D_e_xx, ..., stim, boundary mask), channel-wise attention scales as O(C²). Manageable for small C, unbounded for large.
- **Large-scale pretraining setup**: the paper's foundation-model framing requires substantial pretraining compute. For a single-laboratory project, running it from scratch is prohibitive. Use their pretrained weights if compatible.
- **Demonstrated on regular grids only**: our Bidomain V1 is on a regular grid, so this matches. But if we later extend to unstructured meshes, PDE-Transformer is not directly applicable (Transolver would be the alternative).

### Actionable Insights
- **HIGH — Adopt PDE-Transformer as the dual-tower architecture.** Use their `github.com/tum-pbs/pde-transformer` as the starting implementation. Replace the PDE-agnostic pretraining with a bidomain-specific training regime.
- **HIGH — Use channels-as-tokens framing explicitly.** V_m is one channel stream, φ_e is another, conductivity fields are auxiliary channels fed through channel-wise attention.
- **HIGH — Windowed spatial attention** is the defensible choice per our BC constraints. No more debating Swin vs full vs linear.
- **MEDIUM — Start from pretrained weights if transferable.** Their pretraining on 16 PDEs likely includes reaction-diffusion systems close to ours — fine-tuning may converge much faster than training from scratch.
- **MEDIUM — Channel count management**: if we want to include full anisotropic D tensors as input channels, budget for the O(C²) channel-attention cost. Consider grouping (e.g., D_i components as one "conductivity" token) rather than one-hot per component.
- **LOW — Time-conditioning is optional**: for the elliptic solve (instantaneous) we can strip the time-conditioning. For the full autoregressive bidomain rollout it may help with step-size awareness.

## Limitations / Caveats
- **ICML 2025 paper, very recent**: implementation maturity is early. Expect code instabilities / breaking changes.
- **Foundation-model framing**: the headline benefits (transfer across PDEs, improved downstream) require substantial pretraining compute. A single-lab project would adapt rather than pretrain from scratch.
- **Channel-wise attention cost** scales quadratically with channel count — may need to be thoughtful about what counts as a "channel."
- **BC details not in abstract** — need PDF read to verify Neumann/Dirichlet handling is clean.
- **Regular grid only** — same limitation as FNO, UGrid, etc. Good for Bidomain V1; not for unstructured cardiac meshes.
- **Not benchmarked on cardiac EP specifically** — we'd be porting a general-physics architecture to our domain. Expect some adaptation work.
