# Training Strategy — Ionic Surrogate v3

> Authoritative reference for all training decisions. Single loss per phase, no weighting.
> Model: IonicSurrogateV3 (1,534 inference params). First run: TTP06, small dims.

---

## Overview

> **UPDATED 2026-04-03**: A1 (encoder autoencoder), A3 (encoder-fed conductance), and teacher forcing removed. Model discovers its own latent through dynamics training. Encoder deleted. Phase order reduced from 11 to 9.

9 phases: **A2, B1-B5, C, D, E**. Each phase trains specific components with all others frozen. Combined ionic + concentration loss from B1 onward (no weighting). All rollouts start from zeros (steady state), purely autoregressive.

```
A2: Attention concentration tracking (optional, can skip — conc trains in B1)
B1: Latent discovery (rollout=1) — attention+MLP+decoder learn what 16 dims mean
B2: Short rollout (rollout=10) — model handles own errors
B2_decoder: Decoder recalibration — freeze attention+MLP, train decoders on stable latent
B2_cond: Conductance bootstrap — freeze attention+MLP, train conductance compression+decoders
B3: Medium rollout (rollout=100) — only rollout length changes, all params already trained
B4: Long rollout (rollout=1000)
B5: Very long rollout (rollout=10000) — full action potential
C: Concentration dynamics refinement
D: Stage 2 current readout (frozen Stage 1)
E: End-to-end fine-tune
```

**Training progression — two halves of Stage 1, then combine:**

Stage 1 has two sequential halves:
```
Half 1:  carried_state → VoltageAttention → split → ionic_mixing_mlp → ionic_new
Half 2:  carried_state_new → gate_conductance_mlp + linear → conductance_latent
```

Training order:
1. **B1** (rollout=1): Train Half 1 (attention + ionic_mixing_mlp + ionic_state_decoder). Half 2 frozen. Model discovers latent from (zeros, Vm).
2. **B2** (rollout=10): Same params. Model handles own errors over short sequences.
3. **B2_cond** (rollout=1): Freeze Half 1. Train Half 2 (gate_conductance_mlp + linear + logit + gate_conductance_decoder). Conductance compression learns gate products from stable latent.
4. **B2_cond_r10** (rollout=10): Same as B2_cond but with rollout=10. Conductance tracks across steps.
5. **B3** (rollout=100): Unfreeze ALL Stage 1 params. Only variable changing = rollout length.

**Rule**: Always train encoder and decoder together. A decoder trained on a frozen latent
becomes miscalibrated the moment the latent representation shifts. Each decoder must be
unfrozen whenever the components that produce the latent it reads are unfrozen.

**Principle**: Train each half of Stage 1 to stability before combining. Extend rollout gradually for each half independently. When combining, the only new variable is rollout length — no newly unfrozen params, no new data.

**Removed phases**: A1 (encoder autoencoder — model should discover own latent, not have it imposed), A3 (encoder-fed conductance — needs stable latent from B1/B2 first). See IDEALOG.md Failed Approaches.

---

## Phase Details

### Phase A1 — Ionic Autoencoder

| Property | Value |
|----------|-------|
| **Trains** | Temporary encoder (14→16) + `ionic_state_decoder` (16→14) |
| **Frozen** | Everything else (attention, MLP, compression, Stage 2) |
| **Loss** | `MSE(ionic_state_decoder(encoder(true_14_states)), true_14_states)` |
| **Data** | T1 only. Random state snapshots (no temporal structure). |
| **Batch size** | 4096 |
| **LR** | 1e-3, cosine decay |
| **Rollout** | N/A (single-step) |
| **Scheduled sampling** | N/A |
| **Transition** | Val reconstruction MSE < 1e-4 or plateau for 10 epochs |
| **Checkpoint** | `best_A1.pt` — encoder weights (temporary), decoder weights (transfer to scaffold) |

**Purpose**: Bootstrap the latent space. Encoder learns to map 14 true ionic states (12 HH gates + RR + Ca_SR) into the 16-dim ionic latent. Decoder (= `ionic_state_decoder` in Stage 1) learns to reconstruct. After A1, the encoder provides initial latent mappings for teacher forcing in Phase B.

**Encoder is TEMPORARY** — used only for teacher forcing in Phase B, then discarded.

### Phase A2 — Attention Concentration Tracking

| Property | Value |
|----------|-------|
| **Trains** | `voltage_attention` (W_q, W_k, W_v, W_out) — concentration dims only |
| **Frozen** | Encoder, decoder, MLP, compression, Stage 2. Ionic dims of attention receive gradients but do not need to converge. |
| **Loss** | `MSE(conc_attention_output, true_conc_next)` — 4 concentration dims only |
| **Data** | T1 sequential pairs (state_t, conc_t+1) |
| **Batch size** | 2048 |
| **LR** | 1e-3, cosine decay |
| **Rollout** | N/A (single-step pairs) |
| **Transition** | Val concentration MSE < 1e-6 or plateau for 10 epochs |
| **Checkpoint** | `best_A2.pt` — attention weights |

**Purpose**: Teach attention to track slow concentration dynamics. Concentrations split off after attention (before MLP), so attention is their only update mechanism.

### Phase A3 — Gate Conductance Projection

| Property | Value |
|----------|-------|
| **Trains** | `gate_conductance_mlp`, `gate_conductance_linear`, `gate_conductance_logit`, `gate_conductance_decoder` |
| **Frozen** | Attention, ionic MLP, encoder, ionic_state_decoder, Stage 2 |
| **Loss** | `MSE(gate_conductance_decoder(conductance_latent), true_5_products)` |
| **Data** | T1 carried_state vectors → conductance products. Carried_state constructed from encoder(true_14_states) + true_concentrations. |
| **Batch size** | 4096 |
| **LR** | 1e-3, cosine decay |
| **Rollout** | N/A (single-step) |
| **Transition** | Val conductance product MSE < 1e-4 or plateau for 10 epochs |
| **Checkpoint** | `best_A3.pt` — compression + conductance decoder weights |

**Conductance products** (5 targets): G_Na(m^3hj), G_CaL(dff2fCass), G_to(rs), G_Kr(Xr1Xr2), G_Ks(Xs^2).

### Phase B — Stage 1 Dynamics (Rollout Curriculum)

5 sub-phases with increasing rollout length and decreasing teacher forcing:

| Sub-phase | Rollout | Sched. sampling p | Data | Batch | LR |
|-----------|---------|-------------------|------|-------|----|
| B1 | 1 | 0.1 (10% model) | T1 | 1024 | 5e-4 |
| B2 | 10 | 0.3 | T1 | 512 | 5e-4 |
| B3 | 100 | 0.5 | T1+T2 | 256 | 5e-4 |
| B4 | 1000 | 0.8 | T1-T3 | 128 | 3e-4 |
| B5 | 10000 | 1.0 (full autoregressive) | T1-T3 | 64 | 2e-4 |

| Property | Value |
|----------|-------|
| **Trains** | `voltage_attention`, `ionic_mixing_mlp`, `ionic_mixing_logit` (all Stage 1 dynamics) |
| **Frozen** | Compression, conductance decoder, Stage 2. Encoder frozen but used for teacher forcing. |
| **Loss** | `MSE(ionic_state_pred, true_14_states)` via scaffold decoder, averaged over rollout |
| **Data** | T1 (B1-B2), T1+T2 (B3), T1-T3 (B4-B5). T12 (celltypes) enters at B1. |
| **LR schedule** | Cosine decay within each sub-phase, reset at sub-phase transition |
| **Transition** | Val loss plateau for 15 epochs within sub-phase |
| **Checkpoint** | `best_B1.pt` through `best_B5.pt` |

**Scheduled sampling**: At each rollout step, with probability p use model's own predicted latent, with probability (1-p) use `encoder(true_states)`. Ramps from mostly teacher-forced to fully autoregressive.

**Teacher forcing mechanism**: Encoder maps true_14_states → 16-dim latent. This replaces the model's predicted ionic latent at each step (concatenated with true concentrations to form carried_state). The encoder is NOT used for initialization — only for mid-rollout correction.

### Phase C — Concentration Dynamics

| Property | Value |
|----------|-------|
| **Trains** | `voltage_attention` (concentration dims unfrozen), all Stage 1 params |
| **Frozen** | Stage 2 |
| **Loss** | `MSE(conc_pred, true_conc)` averaged over rollout |
| **Data** | T1-T3, rollout=10000 |
| **Batch size** | 64 |
| **LR** | 1e-4, cosine decay |
| **Transition** | Val concentration MSE < 1e-5 or plateau for 20 epochs |
| **Checkpoint** | `best_C.pt` |

**Purpose**: After B trains ionic dynamics at long rollout, C ensures concentration tracking remains accurate under autoregressive drift.

### Phase D — Stage 2 Current Readout

| Property | Value |
|----------|-------|
| **Trains** | All Stage 2 params (`conductance_attention`: e_q, e_k, e_v; `output_mlp`: W1, b1, W2, b2) |
| **Frozen** | All Stage 1 (including scaffold decoders) |
| **Loss** | `MSE(I_ion_pred, I_ion_true)` |
| **Data** | T1-T4, rollout=10000 |
| **Batch size** | 64 |
| **LR** | 1e-3, cosine decay |
| **Transition** | Val I_ion MSE < 1e-3 or plateau for 20 epochs |
| **Checkpoint** | `best_D.pt` |

**Key**: Stage 1 is completely frozen. Stage 2 learns to read the conductance latent and environment that Stage 1 produces. Stage 2 uses PREVIOUS step's conductance_latent and concentrations (operator splitting convention).

### Phase E — End-to-End Fine-Tune

| Property | Value |
|----------|-------|
| **Trains** | All parameters (Stage 1 + Stage 2). Scaffold decoders still present but loss not used. |
| **Frozen** | Nothing |
| **Loss** | `MSE(I_ion_pred, I_ion_true)` averaged over rollout |
| **Data** | T1-T4, rollout=10000→100000 |
| **Batch size** | 32 |
| **LR** | 5e-5, cosine decay |
| **Transition** | Val I_ion MSE converged. Final model. |
| **Checkpoint** | `best_E.pt` (final), `best_E_no_scaffold.pt` (scaffold removed) |

**After E**: Remove scaffold decoders (`model.remove_scaffold()`). Save production model. Validate on T5-T11 held-out data.

---

## Initialization

| Component | Initial value | Rationale |
|-----------|--------------|-----------|
| Ionic latent (16 dims) | zeros | Model discovers own representation, not TTP06-imprinted |
| Concentrations (4 dims) | [Na_i=10, K_i=138, Ca_i=0.0001, Ca_ss=0.0002] | Real resting values (Layer 0 physics) |
| Conductance latent (8 dims) | zeros | Derived from carried_state via compression |
| Attention (W_q, W_k, W_v, W_out) | Xavier uniform | Standard |
| MLP weights | Xavier uniform | Standard |
| Interpolation logits (alpha, beta) | -5.0 | sigmoid(-5) ~ 0.007, near-pure residual at init |

**After A1**: Encoder provides latent mapping for teacher forcing. But encoder output is NOT used to initialize the latent at t=0 — the model always starts from zeros and self-corrects within ~1 step (attention gate ~ 1 when latent far from target).

---

## Optimizer

| Parameter | Value |
|-----------|-------|
| Algorithm | AdamW |
| Gradient clipping | max_norm=1.0 |
| LR schedule | Cosine decay within each phase, reset at phase transitions |
| Weight decay | 1e-4 (A1-A3, B1-B2), 5e-4 (B3-C), 1e-3 (D-E) |

**Per-phase LR**:

| Phase | LR |
|-------|-----|
| A1, A2, A3 | 1e-3 |
| B1-B3 | 5e-4 |
| B4-B5 | 2e-4 to 3e-4 |
| C | 1e-4 |
| D | 1e-3 (fresh Stage 2 params) |
| E | 5e-5 (fine-tune) |

---

## Data Curriculum

| Phase | Tiers used |
|-------|-----------|
| A1, A2, A3 | T1 only |
| B1-B2 | T1, T12 (celltypes) |
| B3 | T1, T2, T12 |
| B4-B5 | T1-T3, T12 |
| C | T1-T3, T12 |
| D | T1-T4 |
| E | T1-T4 |
| Post-E validation | T5-T11 (held out, never trained on) |

**T12 (celltypes)** enters at Phase B — same protocols as T1-T3 but with ENDO and M_CELL configurations. Forces the model to generalize across cell types early.

**T5-T11** (tissue-mimicking, voltage clamp, concentration perturbation, long-duration, corruption recovery, tissue-specific, combined stressors) are held out until after Phase E to test generalization.

---

## Validation

- **Split by protocol**: validation set contains unseen BCL values (e.g., train on BCL={300,500,700,1000,1500}, val on BCL={400,600,800,2000})
- **Metrics per phase**:
  - Phase A1: reconstruction MSE
  - Phase A2: concentration tracking MSE
  - Phase A3: conductance product MSE
  - Phase B: ionic state prediction MSE, APD error (%)
  - Phase C: concentration tracking MSE over rollout
  - Phase D-E: I_ion MSE, APD error (%), dVm/dt_max error (%)
- **Early stopping**: per phase, patience = 15-20 epochs depending on phase
- **APD error**: computed by running full-beat rollout, measuring APD90 vs ground truth
- **dVm/dt_max error**: max upstroke velocity, sensitive to Na channel dynamics

---

## Hardware

- **GPU**: NVIDIA RTX PRO 4500 Blackwell (33.7 GB VRAM)
- **VRAM**: Not a constraint. Model is 1,534 inference params (~6 KB). Rollout=100K at batch=8 uses ~500 MB.
- **Bottleneck**: Training wall time, not memory. Estimated ~40 GPU-hours total across all phases.
- **Data loading**: Shards are pre-chunked float32 .pt files (~200 MB each), load directly to GPU.

---

## Model Dimensions (TTP06 First Run)

| Dim | Value | Notes |
|-----|-------|-------|
| ionic_dim | 16 | Latent ionic state |
| conc_dim | 4 | [Na_i, K_i, Ca_i, Ca_ss] |
| carried_dim | 20 | ionic + conc |
| attn_dim | 4 | Attention projection |
| cond_dim | 8 | Conductance latent |
| mlp_hidden | 16 | Ionic mixing MLP |
| comp_h1, comp_h2 | 12, 12 | Compression MLP |
| n_env | 9 | Environment tokens for Stage 2 |
| stage2_attn | 4 | Stage 2 Q/K dim |
| stage2_dv | 1 | Stage 2 value dim |
| stage2_mlp_h | 4 | Stage 2 output MLP hidden |

| Component | Inference params | Scaffold params |
|-----------|-----------------|----------------|
| Stage 1 (attention + MLP + compression) | 1,416 | +243 (decoders) |
| Stage 2 (cross-attention + output MLP) | 118 | 0 |
| **Total** | **1,534** | **+243** |
