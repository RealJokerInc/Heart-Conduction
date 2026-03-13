# Surrogate Pipeline -- Implementation Progress

> **This file is the single source of truth for what's done, what's in-progress, and what's next.**
> Read this FIRST at the start of every session or after compaction.

---

## Current Status

**Active Phase:** Documentation
**Last Updated:** 2026-03-12

---

## Documentation Phase -- DONE

| Document | Status | Notes |
|----------|--------|-------|
| `README.md` | DONE | Overview, ionic Transformer data flow, cross-skip ResNet, build order, upgrade path |
| `improvement.md` | DONE | Full architecture spec: gate AE, ionic Transformer (Vm-only input → gates → I_ion), cross-skip coupled ResNet, composed model, temporal schedule, 14 design decisions |
| `IMPLEMENTATION.md` | DONE | 7-phase plan: data → AE → Transformer → mono ResNet → bi cross-skip → e2e → inference |
| `PROGRESS.md` | DONE | This file |

### Key Design Decisions (summary)

- **Ionic Transformer**: Vm history only → attention → MLP → **latent_state** (model-agnostic) → **MLP_ion** (universal) → I_ion
- **Latent space**: abstract ionic state, not compressed gates. Shared across all ionic models. Only Attention block is model-specific
- **Gate decoder**: training scaffold only — provides auxiliary loss to bootstrap latent space, removed for production
- **MLP_ion**: universal current calculator — same network for TTP06, ORd, any ionic model
- **Cross-model training**: TTP06 + ORd gate decoders train in parallel on shared latent space
- **Diffusion ResNet**: cross-skip coupled dual-path — bidirectional 1×1 conv skip connections between Vm and phi_e paths
- **Build order**: monodomain single-path first → bidomain cross-skip upgrade
- **Training stages**: A(scaffolded ionic → remove scaffold) → B1(mono) → B2(bi) → C(end-to-end)
- **Upgrade path for phi_e**: cross-skip → dilated conv → U-Net → local Transformer → FNO
- dt = 0.01ms, fixed grid, TTP06 first, 300-point non-uniform lookback, predict I_ion (not skip to Vm)

---

## Phase 1A: Single-Cell Data Generation -- NOT STARTED

## Phase 1B: Gate Decoder (Training Scaffold) -- NOT STARTED

## Phase 2: Ionic Transformer (Stage A) -- NOT STARTED

## Phase 3A: Monodomain Single-Path ResNet (Stage B1) -- NOT STARTED

## Phase 3B: Bidomain Cross-Skip Upgrade (Stage B2) -- NOT STARTED

## Phase 4: End-to-End Integration (Stage C) -- NOT STARTED

## Phase 5: Inference & Evaluation -- NOT STARTED

## Phase 6: Upgrade Path -- NOT STARTED (conditional)

## Phase 7: ML-Directed Optimization -- NOT STARTED (future)

---

## Key Line Numbers in improvement.md

- Design principles: L9
- Gate Autoencoder: L19
- Ionic Transformer: L54
- Transformer data flow diagram: L71
- Non-uniform temporal sampling: L96
- Transformer architecture details: L118
- Cross-Skip Coupled ResNet: L145
- Cross-skip block diagram: L163
- Full data flow: L178
- Monodomain simplification: L196
- Receptive field + upgrade path: L206
- Full composed model: L233
- History buffer: L264
- Sequential training strategy: L275
- Temporal schedule: L316
- Design decisions log: L341

---

## Session Log

| Date | Session | Work Done |
|------|---------|-----------|
| 2026-03-11 | 1 | Created scaffold documents |
| 2026-03-11 | 2 | Initial architecture: two-component model, gate AE, ionic Transformer, dual-path ResNet |
| 2026-03-12 | 3 | Refined architecture: Vm-only Transformer input, cross-skip coupled ResNet, monodomain-first build order, elliptic upgrade path |
| 2026-03-12 | 4 | Refined ionic architecture: universal latent space + universal MLP_ion (model-agnostic), gate decoder as training scaffold (removed for production), parallel TTP06/ORd decoders, 18 design decisions logged |
