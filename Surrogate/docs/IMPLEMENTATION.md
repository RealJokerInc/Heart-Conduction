# Surrogate Pipeline: Phase-by-Phase Implementation Guide

This document provides an ordered implementation plan. Phases follow the sequential training strategy: data first, then each component in isolation, then integration. Monodomain baseline before bidomain.

**Cross-Reference Key:**
- `improvement.md` — Architecture specification (this repo)
- `Bidomain/Engine_V1/` — Bidomain simulator (training data)
- `Monodomain/Engine_V5.4/` — Monodomain simulator (monodomain baseline data)
- `Research/code_examples/torchcor/` — Reference GNN/DeepONet implementations

---

## Phase 1: Data Generation & Gate Autoencoder

**Goal:** Generate single-cell training data and build the gate latent space.

### Phase 1A: Single-Cell Data Generation

Generate TTP06 single-cell trajectories: (Vm, 18 gate states, I_ion) at dt=0.01ms.

| # | File | Action | Notes |
|---|------|--------|-------|
| 1A.1 | `surrogate/__init__.py` | Create | Package init |
| 1A.2 | `surrogate/data/__init__.py` | Create | Data subpackage |
| 1A.3 | `surrogate/data/single_cell_generator.py` | Create | Run TTP06 ODE with varied pacing, save trajectories |
| 1A.4 | `surrogate/data/temporal_sampler.py` | Create | Non-uniform 300-point lookback schedule |

**Validation:**

| Test | Criteria |
|------|----------|
| 1A-V1 | Generator produces (Vm, 18 gates, I_ion) at dt=0.01ms for 1000ms |
| 1A-V2 | Multiple pacing rates (BCL=300,500,800,1000ms) generate distinct AP morphologies |
| 1A-V3 | Temporal sampler produces 300 indices: dense near t=0, sparse for distant past |
| 1A-V4 | Sampled points resolve upstroke (~100 points in first 1ms) and cover ~300ms total |

### Phase 1B: Gate Decoder (Training Scaffold)

The gate decoder is not a standalone autoencoder — it is trained jointly with the ionic Transformer
in Phase 2 as an auxiliary loss branch. Phase 1B creates the module and validates it can reconstruct
gates from a known latent space (sanity check before joint training).

| # | File | Action | Notes |
|---|------|--------|-------|
| 1B.1 | `surrogate/models/__init__.py` | Create | Models subpackage |
| 1B.2 | `surrogate/models/gate_decoder.py` | Create | Decoder: (d_latent,) → (n_gates,) per ionic model |
| 1B.3 | `surrogate/training/__init__.py` | Create | Training subpackage |

**Validation:**

| Test | Criteria |
|------|----------|
| 1B-V1 | Decoder forward pass: (batch, d_latent) → (batch, 18) for TTP06 |
| 1B-V2 | Decoder forward pass: (batch, d_latent) → (batch, 40) for ORd |
| 1B-V3 | Given PCA-reduced gate states as "latent", decoder can reconstruct (sanity check) |

---

## Phase 2: Ionic Transformer (Stage A Training)

**Goal:** Train per-node Transformer on single-cell data. Input: Vm history only. Output: latent ionic state → I_ion. Gate decoder provides auxiliary scaffold loss during training.

| # | File | Action | Notes |
|---|------|--------|-------|
| 2.1 | `surrogate/models/ionic_transformer.py` | Create | Attention + MLP → latent_state |
| 2.2 | `surrogate/models/mlp_ion.py` | Create | Universal MLP_ion: (latent, Vm) → I_ion |
| 2.3 | `surrogate/data/ionic_dataset.py` | Create | Vm history sequences → (gates, I_ion) targets |
| 2.4 | `surrogate/training/stage_a_transformer.py` | Create | Scaffolded training with gate decoder + I_ion loss |
| 2.5 | `surrogate/training/losses.py` | Create | Gate scaffold loss + I_ion MSE + rollout loss |

**Training curriculum (scaffolded → production):**
1. Single-step: Vm history → latent → I_ion, with gate scaffold (high weight)
2. Short rollout (N=10): accumulate Vm error, gate scaffold active
3. Medium rollout (N=100): anneal gate scaffold weight toward zero
4. Long rollout (N=1000): gate scaffold removed, I_ion + rollout only

**Parallel cross-model training (optional):**
- Train Attention_TTP06 + Decoder_TTP06 on TTP06 data simultaneously with
  Attention_ORd + Decoder_ORd on ORd data. Shared latent space + shared MLP_ion.

**Validation:**

| Test | Criteria |
|------|----------|
| 2-V1 | Forward pass: (batch, 300, 1) → latent (batch, d_latent) + I_ion (batch, 1) |
| 2-V2 | Single-step I_ion prediction MSE decreases during training |
| 2-V3 | With scaffold: decoded gates match TTP06 ground truth within tolerance |
| 2-V4 | Without scaffold: I_ion still accurate after gate loss removed |
| 2-V5 | Autoregressive AP rollout (1000ms): Vm trajectory qualitatively correct |
| 2-V6 | APD prediction error < 5ms on held-out pacing rates |
| 2-V7 | Rollout stable for 5000ms (5 beats at BCL=1000ms) without divergence |
| 2-V8 | (Optional) TTP06 and ORd latent trajectories are comparable in shared space |

---

## Phase 3: Diffusion ResNet (Stage B Training)

**Goal:** Train spatial conv network on pure diffusion data. Monodomain first, then bidomain.

### Phase 3A: Monodomain Single-Path (Stage B1)

Single ResNet on Vm-only diffusion. Uses Monodomain V5.4 as data source.

| # | File | Action | Notes |
|---|------|--------|-------|
| 3A.1 | `surrogate/data/diffusion_generator.py` | Create | Generate mono/bidomain diffusion-only snapshots |
| 3A.2 | `surrogate/models/diffusion_resnet.py` | Create | Start with single-path ResNet (Vm only) |
| 3A.3 | `surrogate/data/diffusion_dataset.py` | Create | Paired (input, target) spatial field dataset |
| 3A.4 | `surrogate/training/stage_b_diffusion.py` | Create | Stage B training loop |

**Validation:**

| Test | Criteria |
|------|----------|
| 3A-V1 | Forward pass: (batch, 1, Nx, Ny) → (batch, 1, Nx, Ny) correct shapes |
| 3A-V2 | Gaussian diffusion: predicted variance growth matches 2*D_eff*dt |
| 3A-V3 | 100-step autoregressive rollout matches monodomain simulator |
| 3A-V4 | Boundary conditions respected (Neumann: no flux at edges) |

### Phase 3B: Bidomain Cross-Skip Upgrade (Stage B2)

Add phi_e path + bidirectional cross-skip connections. Initialize Vm path from Phase 3A.

| # | File | Action | Notes |
|---|------|--------|-------|
| 3B.1 | `surrogate/models/diffusion_resnet.py` | Extend | Add phi_e path, cross-skip connections |
| 3B.2 | `surrogate/data/diffusion_generator.py` | Extend | Add bidomain diffusion-only data generation |

**Validation:**

| Test | Criteria |
|------|----------|
| 3B-V1 | Forward pass: (batch, 2, Nx, Ny) → (batch, 2, Nx, Ny) correct shapes |
| 3B-V2 | Gaussian diffusion: both Vm and phi_e fields evolve correctly |
| 3B-V3 | Cross-domain coupling: phi_e responds to Vm gradient (not independent channels) |
| 3B-V4 | 100-step rollout: fields match bidomain simulator within tolerance |
| 3B-V5 | Monodomain limit: with D_e→∞, phi_e→0, Vm matches monodomain result |

---

## Phase 4: End-to-End Integration (Stage C Training)

**Goal:** Combine trained ionic Transformer + cross-skip ResNet, fine-tune on full bidomain data.

| # | File | Action | Notes |
|---|------|--------|-------|
| 4.1 | `surrogate/data/bidomain_generator.py` | Create | Full bidomain simulation → training snapshots |
| 4.2 | `surrogate/models/surrogate.py` | Create | Composed model: ionic → diffusion + history buffer |
| 4.3 | `surrogate/training/stage_c_finetune.py` | Create | End-to-end fine-tuning with rollout loss |
| 4.4 | `surrogate/training/metrics.py` | Create | Vm/phi_e MSE, AT error, CV error |

**Fine-tuning schedule:**
1. Freeze both components, train only the connection (Vm_post_ionic handoff)
2. Unfreeze diffusion ResNet with small LR
3. Unfreeze ionic Transformer with smaller LR
4. Full end-to-end with autoregressive rollout loss

**Validation:**

| Test | Criteria |
|------|----------|
| 4-V1 | Single-step prediction: Vm MSE < threshold vs simulator |
| 4-V2 | 100-step rollout: wavefront position error < 1 grid cell |
| 4-V3 | Full AP propagation: CV within 5% of simulator |
| 4-V4 | phi_e field: spatial pattern qualitatively matches simulator |
| 4-V5 | Error accumulation: rollout error grows sub-linearly over 1000 steps |
| 4-V6 | Kleber boundary speedup: bath-coupled CV_ratio within 10% of simulator's |

---

## Phase 5: Inference & Evaluation

**Goal:** Fast prediction API and systematic accuracy/speed benchmarking.

| # | File | Action | Notes |
|---|------|--------|-------|
| 5.1 | `surrogate/inference/__init__.py` | Create | Inference subpackage |
| 5.2 | `surrogate/inference/predictor.py` | Create | Autoregressive rollout wrapper |
| 5.3 | `tests/test_accuracy.py` | Create | Surrogate vs simulator comparison suite |
| 5.4 | `tests/test_speed.py` | Create | Timing benchmarks (per-step and full sim) |

**Validation:**

| Test | Criteria |
|------|----------|
| 5-V1 | Per-step inference time < simulator per-step time |
| 5-V2 | Full 500ms simulation accuracy within tolerance |
| 5-V3 | Speedup factor reported (wall time surrogate / wall time simulator) |
| 5-V4 | Memory footprint < simulator memory footprint |

---

## Phase 6: Upgrade Path (If Needed)

**Goal:** Improve phi_e accuracy if cross-skip ResNet receptive field is insufficient.

Only proceed if Phase 4/5 shows phi_e errors are the bottleneck.

| Upgrade | File Changes | When |
|---------|-------------|------|
| Dilated convolutions | Modify ResBlocks in `diffusion_resnet.py` | First attempt |
| U-Net bottleneck | Add encoder-decoder in `diffusion_resnet.py` | If dilated insufficient |
| Local Transformer | New block in `diffusion_resnet.py` | If U-Net insufficient |
| FNO spectral layer | New module, insert in phi_e path | Alternative to local Transformer |

---

## Phase 7: ML-Directed Optimization (Future)

**Goal:** Close the loop — use trained surrogate to guide parameter space exploration.

Deferred until Phases 1-5 are stable.

---

## Summary

| Phase | Goal | Data Source | Priority |
|-------|------|------------|----------|
| 1A | Single-cell data + temporal sampler | TTP06 ODE | Critical |
| 1B | Gate autoencoder | Single-cell trajectories | Critical |
| 2 | Ionic Transformer (Stage A) | Single-cell Vm sequences | Critical |
| 3A | Monodomain single-path ResNet (Stage B1) | Monodomain V5.4 diffusion | Critical |
| 3B | Bidomain cross-skip upgrade (Stage B2) | Bidomain V1 diffusion | Critical |
| 4 | End-to-end integration (Stage C) | Full bidomain sims | Critical |
| 5 | Inference & evaluation | Held-out sims | High |
| 6 | Upgrade path (dilated/UNet/Transformer/FNO) | Same as Phase 4 | Conditional |
| 7 | ML-directed optimization | Surrogate predictions | Future |
