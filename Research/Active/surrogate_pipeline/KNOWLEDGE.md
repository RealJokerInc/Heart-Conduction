# Surrogate Pipeline — Knowledge File

> This file is a running synthesis. Updated as findings accumulate.
> When the question is complete, a copy is promoted to `Research/Knowledge/`.

## Overall Architecture

Two-component autoregressive surrogate mirroring the bidomain simulator's operator splitting:

1. **Ionic Surrogate** (per-node, local): pure autoregressive latent — carried latent state updated each step via n×1 cross-attention to Vm. Predicts I_ion from latent + Vm. No Vm history buffer.
2. **Cross-Skip Coupled ResNet** (spatial, global): dual conv paths for Vm and phi_e with bidirectional 1×1 cross-skip connections. Replaces the elliptic solve (PCG/GMG) which dominates 94% of bidomain wall time.

Per-step inference: `Rush-Larsen(or ionic surrogate) → Vm_post_ionic → ResNet → (Vm_next, phi_e_next)`

## Ionic Surrogate — Design

### Architecture (3 stages, 705 FLOPs, 2.9× Rush-Larsen)

**Stage 1 — n×1 Cross-Attention** (~496 FLOPs): Each of 16 latent dimensions independently queries the voltage. Produces a per-dimension gate (how much to update) and per-dimension target (where to move toward). Contractive by construction.
```
Q = W_q · latent_prev            (16, 8)     per-dim query (state-dependent)
K = W_k · [Vm, I_stim, dt]      (1, 8)      voltage key (shared)
V = W_v · [Vm, I_stim, dt]      (1, 8)      voltage value (shared)
gate = σ(Q · K^T / √8)          (16, 1)     per-dim attention score
target = V · W_out               (16,)       per-dim voltage-dependent target
delta = gate · (target - latent_prev)         contraction toward target
latent_mid = latent_prev + delta
```
Note: W_out is (8, 16) — each latent dim gets its OWN target. Without per-dim targets, all dims converge to the same equilibrium (useless).

**Stage 2 — Split GELU Cross-Channel** (~176 FLOPs): Half the latent dims gate the other half (SwiGLU-style), then a spectrally-normalized linear projection mixes channels. Captures Ca↔CaL and other inter-channel coupling. Rank-8 bottleneck matches real coupling rank (~3 concentration variables).
```
gated = GELU(latent_mid[:8]) ⊙ latent_mid[8:]    (8,)
correction = spectral_norm(W_cc) · gated + b      (16,)    Lipschitz-bounded
latent_new = latent_mid + correction
```
Spectral normalization constrains ||W_cc||₂ ≤ 1, bounding the correction magnitude and preserving Stage 1 contractivity. Implementation: `torch.nn.utils.spectral_norm()`.

**Stage 3 — Linear Current Readout** (~33 FLOPs): Weighted sum of latent + Vm. No activation — I_ion is unbounded (~-10 to +50 µA/cm²). Physically correct: real ionic current is a sum of per-channel contributions, inherently linear.
```
I_ion = w · latent_new + b_vm · Vm + b
```

**Scaffold — Gate Decoder** (training only, ~324 FLOPs): Single linear (16→18) + sigmoid → 18 TTP06 gates. Forces latent to encode physically meaningful ionic state. Removed for production inference. If linear decoder is insufficient, upgrade to MLP (16→32→18).

### Why Carried Latent (Not Vm History)

The ionic surrogate uses pure autoregressive latent with no Vm history buffer. Three reasons:
1. **Stimulus artifacts**: Current injection creates Vm spikes that confuse history-based models. Carried latent handles I_stim as an explicit input.
2. **Rush-Larsen doesn't use history either**: The real simulator carries 18 gate states forward. Our latent is the learned analog.
3. **No buffer management**: No non-uniform temporal schedule, no resampling, no variable-dt issues.

### Why n×1 Cross-Attention (Not Full Transformer)

The n×1 cross-attention IS a 1D hybrid Transformer: Q=latent dims, K=V=voltage, no self-attention between dims. Mathematically equivalent to learned Rush-Larsen but with state-dependent gating (the attention score depends on BOTH Vm AND the current latent value, while RL's rate depends on Vm only). Cost: 496 FLOPs vs 200M for a temporal Transformer.

### I_stim as Input — Design Note

I_stim is included as a model input alongside Vm and dt. Biophysically, real gates respond to Vm only — they don't know about I_stim. However, I_stim helps the model during training by distinguishing stimulus-driven from intrinsic depolarization. Evaluate during Phase B whether removing I_stim hurts or helps. If removed, W_k and W_v reduce from (3,8) to (2,8), saving 16 params.

### Initialization

`latent(0) = zeros` or `V·W_out evaluated at Vm_rest`. Self-corrects in ~1 step: gate ≈ 1 when latent is far from target, delta → (target - latent), latent snaps to equilibrium.

### Simplification Spectrum

| Level | FLOPs | vs RL | Description |
|-------|-------|-------|-------------|
| 0: Scalar HH | 176 | 0.7× | σ(wV+b) target, const rate, Nernst current. Start here. |
| 1: + Vm rates | 240 | 1.0× | Sigmoid rate |
| 2: + coupling | 416 | 1.7× | + split GELU cross-channel |
| 3: Full design | 705 | 2.9× | n×1 cross-attn + split GELU + linear readout |

Strategy: start Level 0, see what breaks. When upgrading levels, restart from Phase B (Phase A autoencoder checkpoint is shared). Save per-phase-per-level checkpoints.

Strategy: start Level 0, see what breaks (restitution? calcium? APD accuracy?), add features from the modification menu.

### Modification Menu

**Accuracy upgrades**: A1 multi-head (+200), A2 stacked attention (+464/layer), A3 nonlinear readout (+300), A5 explicit concentrations (+100), A6 structured Nernst current (+50).

**Speed downgrades**: S1 remove cross-channel (-176), S2 scalar attention (-350), S3 smaller latent d=8 (~half), S6 drop to scalar HH (-497).

### Losses and Regularization

- `L_ion = MSE(I_ion_pred, I_ion_true)` — weight 1.0, always active
- `L_gate = MSE(gates_pred, gates_true)` — weight λ_gate, annealed to 0 over training curriculum
- `L_roll = (1/T) Σ_t MSE(Vm_pred(t), Vm_true(t))` — weight λ_roll, introduced in rollout phases
- Weight decay 1e-4 (AdamW), gradient clipping max_norm=1.0

### Parameters

| Component | Detail | Params (inference) | Params (training) |
|-----------|--------|-------------------|-------------------|
| Stage 1: W_q | (16, 8) per-dim query | 128 | 128 |
| Stage 1: W_k | (3, 8) voltage key | 24 | 24 |
| Stage 1: W_v | (3, 8) voltage value | 24 | 24 |
| Stage 1: W_out | (8, 16) per-dim target | 128 | 128 |
| Stage 2: W_cc + b | (16, 8) + (16,) | 144 | 144 |
| Stage 3: w + b_vm + b | (16,) + (1,) + (1,) | 18 | 18 |
| Scaffold: W_dec + b_dec | (18, 16) + (18,) | — | 306 |
| **Total** | | **466** | **772** |

Note: Stage 1 W_out was originally documented as (8,1)=8 params — this was wrong. Must be (8,16)=128 for per-dim targets. Without per-dim targets, all latent dims converge to the same equilibrium.

## Advanced Upgrades — Ideas from Modern ML

### Connection to State Space Models (Mamba)

Our ionic surrogate IS a selective SSM — independently derived but structurally identical. The n×1 cross-attention produces input-dependent state transitions (gate and target depend on Vm), which is Mamba's core "selectivity" mechanism. Mapping: latent=state x, [Vm,I_stim]=input u, I_ion=output y, cross-attention=selective A(u)·x + B(u)·u dynamics.

Two borrowable techniques:

**Parallel scan for training**: Mamba trains over sequences in O(N) using parallel prefix scan instead of sequential rollout. Stage 1 alone is an affine recurrence (z = (1-gate)·z + gate·target) which supports associative scan. **However, Stage 2 (split GELU) is nonlinear and breaks the affine structure required for parallel scan.** Options: (1) apply Stage 2 every K steps, not every step; (2) train Stage 1 with scan, fine-tune with Stage 2 sequentially; (3) accept sequential backprop. For our tiny model (705 FLOPs), sequential backprop through 100K steps is feasible: ~240MB memory per sample (gradient checkpointing reduces to O(√N)), ~210M FLOPs backward per sample. Not a bottleneck on Blackwell GPU.

**Zero-order hold discretization**: Replace our Euler update `latent += delta` with the exact exponential discretization `latent = exp(A·dt)·latent + (exp(A·dt)-1)·A⁻¹·B·u`. Unconditionally stable for any dt (matches Rush-Larsen's exponential exactness). Cost: +16 FLOPs (one exp per dim). Enables variable dt with guaranteed stability.

### KAN (Kolmogorov-Arnold Networks) for Current Readout

Replace linear readout `I_ion = Σ w_k · z_k` with learned 1D spline functions per dim: `I_ion = Σ φ_k(z_k)`. Each latent dim gets its own learned activation-to-current curve. This captures the multiplicative gate structure (m³·h·j) that linear readout cannot, without adding a hidden layer.

Cost: ~80 FLOPs (vs 33 linear, vs 320 MLP with hidden). The nonlinearity is per-dimension (no cross-dim interaction in the readout), which matches the physics: each channel's current contribution is an independent function of its activation.

Could also apply KAN to Stage 2 cross-channel: replace the linear W_cc with spline-parameterized edges. Would capture nonlinear coupling (Ca²⁺-dependent CaL inactivation) more efficiently than split GELU.

### Mixture of Experts (MoE) — AP Phase Specialization

Different AP phases have fundamentally different dominant dynamics (upstroke: I_Na, 0.1ms; plateau: I_CaL vs I_Kr, 100ms; repolarization: I_Kr/I_Ks, 50ms; diastole: I_K1/recovery, 500ms). A single weight set handling all four phases is a compromise.

MoE solution: 4 tiny expert copies of Stage 1, each specialized for one phase. A router (single linear layer on [Vm, I_stim], ~8 FLOPs) selects top-1 expert per step. Only one expert runs per step → zero extra inference cost, 4× effective model capacity. Each expert learns its phase's dynamics without interference from other phases.

The router naturally learns phase boundaries (Vm thresholds correspond to AP phase transitions). During training, load balancing ensures all experts get used.

### Summary of Advanced Upgrades

| Upgrade | Cost Impact | Training Impact | What It Gives |
|---------|-------------|-----------------|---------------|
| Mamba parallel scan | 0 inference | O(log N) vs O(N) | Train over full traces without sequential rollout |
| Mamba ZOH discretization | +16 FLOPs | — | Exact exponential update, stable any dt |
| KAN readout | +47 FLOPs | — | Per-dim nonlinearity (m³·h·j) without hidden layer |
| MoE (4 experts) | +8 FLOPs | ~4× data needed | 4× capacity, phase-specialized, zero cost |

## Training Data Generation

### Protocol Hierarchy

Data generated from TTP06 single-cell ODE. All protocols produce (Vm, 18 gates, I_ion) at each timestep. dt fed as 3rd input to model: K = W_k · [Vm, I_stim, dt].

**Tier 1 — Steady-state pacing**: BCL ∈ {300, 400, 500, 600, 700, 800, 1000, 1500, 2000} ms. 20 beats each (10 warmup + 10 training). Baseline AP morphology at each rate.

**Tier 2 — S1-S2 restitution**: S1=1000ms × 10 beats, S2 at DI ∈ {50, 75, 100, 150, 200, 300, 500, 800} ms. Includes sub-ERP DIs (failed captures). Maps restitution curve.

**Tier 3 — Dynamic protocols**: BCL ramp down (1000→300ms/30 beats), ramp up (300→1000ms), burst (5@300ms + 2000ms pause ×5), alternans (20 beats @ BCL=330ms).

**Tier 4 — Random intervals**: Inter-beat interval ~ LogUniform(200, 2000) ms. 5-200 beats/protocol (variable length) × 200 protocols. Tests arbitrary pacing and generalization to arbitrary trace lengths.

**Tier 5 — Tissue-mimicking current injection**: Simulates what a cell experiences in tissue (diffusion current from neighbors), without running full tissue simulation.
- Ornstein-Uhlenbeck noise: dI/dt = -I/τ + σ·dW, τ ∈ {1, 2, 5, 10, 20} ms, σ ∈ {5, 10, 20} µA/cm². Smooth random current mimicking fluctuating neighbor voltages.
- Smooth depolarizing ramp: 0 → -30 µA/cm² over 2-5ms, then decay. Mimics approaching wavefront.
- Sub-threshold blips: -10 to -20 µA/cm² for 1-3ms at random times. Model learns "don't fire."
- Sustained current offsets: constant ±5 µA/cm² for 10-100ms. Shifts resting potential.
- Biphasic pulses: depolarizing then hyperpolarizing. Mimics wavefront passage.
- Random telegraph: I switches between 0 and -I_max at Poisson times (λ ~ 1-10/ms). Irregular neighbor activity.
- **Real tissue profiles**: Extract I_diff(t) = D·∇²Vm from actual Bidomain V1 runs at representative nodes. Most realistic injection profiles.

**Tier 6 — Voltage clamp**: Vm is externally controlled, model must predict correct I_ion and update latent.
- Step clamp: hold -80mV → step to V_test ∈ {-60,-40,-20,0,+20,+40}mV → hold 500ms. Gates converge to gate_inf(V_test) — exact supervision target.
- Ramp clamp: linear -80→+40mV over 100-500ms. Continuous I-V curve.
- Staircase clamp: -80→-60→-40→-20→0→+20→+40mV, hold 50-200ms each step.
- AP clamp: play back recorded AP waveform as Vm command. Decouples ionic prediction from voltage evolution — clean gradient signal.
- Partial clamp: Vm = α·V_command + (1-α)·V_free, α ∈ {0.3, 0.5, 0.7}. Mimics electrotonic coupling.

Voltage clamp is uniquely valuable: errors in I_ion don't propagate back through Vm (voltage is fixed), giving clean gradients. Gate decoder gets exact targets (gate_inf at clamp voltage).

**Tier 7 — Concentration perturbation**: Perturb initial ionic concentrations to simulate tissue variability.
- K_o ∈ {4.0, 5.0, 5.4, 6.0, 8.0, 10.0} mM (normal=5.4). Hyperkalemia shifts resting Vm.
- Na_i ∈ {6, 8, 10, 12, 15} mM. Affects E_Na and Na current amplitude.
- Ca_i ∈ {0.5×, 1×, 1.5×, 2×} baseline. Affects I_CaL, contractility.
- ~20 representative combos × Tier 1 protocols.

**Tier 8 — Long-duration stability**: Tests that the model doesn't drift over extended simulation.
- Very long steady pacing: 200+ beats at constant BCL (1000ms). Track slow APD drift from Na_i/K_o accumulation. Model must capture this cumulative effect.
- Very long quiescence: 5-30 seconds of rest (no pacing). Latent must hold resting state without drift. Then stimulus → AP must still be correct.
- Long blank → burst: 5-10s rest → sudden fast pacing (BCL=300ms). Tests cold-start after extended quiescence.
- Variable-length traces: same protocol at 10, 50, 100, 200, 500 beats. Model must work at any trace length.

**Tier 9 — Recovery from corruption**: Deliberately push TTP06 into non-physiological gate states.
- Perturb individual gates to random values (e.g., set m=0.9 during diastole, or h=0.0 during rest).
- Run TTP06 forward and record recovery trajectory back to physiological state.
- Teaches self-correction: if the latent drifts to a weird state during tissue inference, the model has seen recovery from similar states.
- Also: sudden Vm jumps (e.g., -86 → -20 mV without stimulus) — what do the gates do? The model must handle this because tissue diffusion can impose arbitrary Vm changes.

**Tier 10 — Tissue-specific scenarios**: Conditions experienced by cells at special tissue locations.
- **Boundary cells**: Reduced electrotonic load (fewer neighbors). Simulate by reducing injection current magnitude by 50%. Longer APD, slightly different repolarization. Relates to Kleber boundary speedup.
- **Infarct border zone**: Cells adjacent to non-conducting scar. Zero current from one side, normal from others. Asymmetric loading. Simulate by injecting current from only one direction (half the normal magnitude).
- **Inert tissue interface**: Cells next to non-excitable tissue. Electrotonic sink during plateau (current drains into inert region). Simulate by adding a sustained small positive (repolarizing) current during plateau phase.
- **Stimulus site**: Cells at the pacing electrode receive strong I_stim AND strong electrotonic current from freshly activated neighbors. Double depolarization. Simulate by combining sharp I_stim pulse with smooth ramp injection.
- **Wavefront tip (spiral)**: Very short DI, rapid re-excitation, curved wavefront. Simulate by pacing at DI=30-80ms (near and below ERP) with varying I_stim amplitude.

**Tier 11 — Combined stressors and stitched protocols**: Real tissue conditions involve multiple simultaneous effects. Also, stitching different protocol segments with variable-length rest periods creates diverse training sequences from existing data.
- Random pacing + OU noise + hyperkalemia (K_o=8mM) in the same trace.
- Fast pacing (BCL=350ms) + sustained depolarizing current (-3 µA/cm²) + elevated Ca_i.
- S1-S2 protocol + tissue injection noise + concentration perturbation.
- **Stitched traces**: Random protocols concatenated with variable-length rest breaks. E.g., [10 beats BCL=1000] → [3-15s rest] → [S1-S2] → [5-20s rest] → [burst pacing]. Breaks ∈ LogUniform(1s, 30s). Tests latent stability through long quiescence then correct response to new protocol.
- Generate 500+ combined/stitched protocols from random combinations.

**Tier 12 — Celltype variants**: TTP06 epi, endo, and M cell configurations.
- Different I_to, I_Ks, I_CaL conductances per celltype.
- Run all Tier 1-3 protocols for each celltype (×3 data).
- In tissue, these celltypes are adjacent — surrogate must handle all three.

### Identified Coverage Gaps and Mitigations

| Gap | Risk | Mitigation |
|-----|------|------------|
| Celltype variants (epi/endo/M) | Model fails on specific celltype in tissue | Tier 12: run protocols for all 3 celltypes |
| Recovery from weird states | Latent drift during tissue inference | Tier 9: corruption → recovery trajectories |
| Wavefront-specific Vm shapes | Model fails at spiral tips, boundaries | Tier 10 + AP clamp from Bidomain V1 node traces |
| Very long simulations | Slow Na_i/K_o drift over minutes | Tier 8: 200+ beat protocols + long quiescence |
| Not enough random protocols | Poor generalization to arbitrary pacing | Tier 4: increased to 200 protocols |
| Partial depolarization zones | Cells near block at intermediate Vm | Tier 10 (infarct border) + Tier 5 (sustained current) |
| Boundary/inert tissue effects | Reduced loading, APD changes | Tier 10: boundary, infarct, inert tissue scenarios |
| Simultaneous effects | Individual tiers tested in isolation | Tier 11: combined stressor protocols |

### Variable dt

Train with dt ∈ {0.005, 0.01, 0.02, 0.05, 0.1} ms. Each protocol run at all 5 dt values. With ZOH discretization (z_new = z·exp(-gate·dt) + target·(1-exp(-gate·dt))), dt is explicit in the formula — exact for any value.

Multi-scale dt within traces: dt=0.005ms during upstroke, dt=0.1ms during diastole. Model must handle dt transitions mid-simulation.

### Data Augmentation (on-the-fly during training)

- **Vm noise injection**: ε ~ N(0, σ²), σ ∈ {0.1, 0.5, 1.0} mV. Teaches robustness to autoregressive prediction errors.
- **Conductance scaling**: g_X × U(0.5, 2.0) for each major current. Simulates biological variability. Directly useful for Optimizer pipeline.
- **Stimulus variation**: amplitude ∈ {-30, -40, -52, -70, -100} µA/cm², duration ∈ {0.5, 1, 2, 5} ms. Include sub-threshold.
- **Random initial conditions**: gate_k ~ N(gate_k_rest, 0.1·gate_k_rest). Non-equilibrium starts.

### Data Storage Format

Two-layer storage: raw HDF5 for reproducibility, pre-chunked .pt shards for training speed.

**Storage location**: External HDD `/media/norepinephrine/Elements-ext4/` (5.5TB ext4, ~1.1TB needed). Too large for NVMe root partition (244GB).

**Generation layer (source of truth):**
```
/media/norepinephrine/Elements-ext4/surrogate_data/raw/
├── tier01_steady_state.h5       one file per tier
├── tier02_s1s2.h5               full metadata per protocol
├── ...                          float64 (simulator precision)
└── tier12_celltypes.h5
```
HDF5 stores complete traces with metadata (BCL, celltype, conductances, protocol type, tier). This is the archival format — never deleted, always reproducible.

**Training layer (optimized for speed):**
```
/media/norepinephrine/Elements-ext4/surrogate_data/train/
├── shard_0000.pt                ~200MB each
├── shard_0001.pt                pre-shuffled
├── ...                          float32 (ML precision)
└── shard_NNNN.pt

/media/norepinephrine/Elements-ext4/surrogate_data/val/
└── ...                          split by PROTOCOL, not timestep
```
Each shard is a single PyTorch tensor: `(N_segments, segment_length, 23)` at float32. Loads directly to GPU via `torch.load(map_location='cuda')` — zero parsing, zero dtype conversion.

**Segment format (23 columns):**

| Columns | Content | Role |
|---------|---------|------|
| 0 | Vm | model input |
| 1 | I_stim | model input |
| 2 | dt | model input |
| 3-20 | 18 gate states | gate decoder target |
| 21 | I_ion | primary prediction target |
| 22 | clamp_mask | 0.0 = free-running, 1.0 = voltage clamped |

Segment lengths: 100, 500, 1000, 5000 steps (matching rollout curriculum stages).

**Speed decisions:**

| Decision | Rationale |
|----------|-----------|
| float32 not float64 | 2× memory, 2× compute. ML doesn't need float64. Generate at float64, convert during pre-processing. |
| Pre-chunk segments offline | No windowing/slicing during training. Segment lengths match curriculum. |
| Shard size ~200MB | Fits CPU page cache. Fast sequential read. ~2000 segments of 1000 steps per shard. |
| Pre-shuffle within shards | No shuffling overhead in DataLoader. |
| Split train/val by protocol | Val sees unseen pacing patterns, not unseen timesteps from seen patterns. |
| Augmentation: hybrid | Conductance scaling, stitching, dt variation → offline (need ODE re-run). Vm noise → on-the-fly (just add N(0,σ²), trivial). |

**Estimated dataset size:** ~500GB at float32 across all tiers and augmentations. Fits on disk easily. Training loads 1-2 shards into GPU at a time (~400MB). Blackwell GPU has ample memory for model + data + gradients.

## Training Strategy

### Principle

Bootstrap the latent space first (clean data, autoencoder), then teach dynamics (simple → complex data, teacher forcing → rollout), then stress-test (noise, tissue, corruption). The latent must be well-structured before dynamics training begins.

### Phase A — Latent Space Bootstrap

Train a standalone gate autoencoder to establish a meaningful latent coordinate system before any dynamics training.

```
encoder: (18 gates) → (16 latent)     temporary, discarded after Phase A
decoder: (16 latent) → (18 gates)     becomes scaffold decoder
```

Data: gate state vectors from Tier 1 (steady-state pacing at all BCLs). Each timestep gives one (18,) vector. No dynamics, no Vm, no I_stim — just gate snapshots.

Loss: MSE(decoder(encoder(gates)), gates)

After Phase A: decoder weights transfer to scaffold. Encoder used for initial latent computation and as a training target reference in Phase B. Latent space is a PCA-like compression of gate states — well-structured and interpretable.

**Why Phase A matters**: Without it, Phase B starts with random decoder AND random dynamics weights. The latent could collapse to something that minimizes I_ion but doesn't encode meaningful ionic state. Phase A anchors the latent space first. Cheap (trains in minutes on 18-dim vectors).

### Phase B — Simple Dynamics

Train the dynamics model (Stages 1-2-3) on clean, simple data with pre-trained decoder frozen.

**Data**: Tier 1 only (steady-state pacing). Single celltype (epi). Single dt=0.01ms. Clean, no noise.

**B1 — Teacher forcing (single-step)**: At each step, provide TRUE latent from encoder(true_gates) as latent_prev. Model predicts latent_new and I_ion. No error accumulation. λ_gate=1.0 (dominant), L_ion secondary. Until gate reconstruction error < threshold.

**B2 — Short rollout (N=10)**: Model uses own predictions as input for 10 steps, then resets to ground truth. Introduce scheduled sampling: randomly replace ground-truth latent with model prediction (start 10% → increase). λ_gate=1.0, λ_roll=0.1. Until 10-step rollout stable.

**B3 — Medium rollout (N=100 = 1ms)**: Autoregressive for 100 steps. λ_gate=0.5 (start annealing), λ_roll=0.5. Until 100-step rollout stable.

### Phase C — Full Dynamics

Add complex data, anneal scaffold, increase rollout.

**Data**: Add Tiers 2-4 (S1-S2, dynamic, random intervals). Add all celltypes. Add multiple dt values. Gradual data mixing: start 90% Tier 1 + 10% new tiers, increase new tier fraction over epochs.

**Unfreeze decoder** — fine-tune jointly with dynamics. The decoder adapts as latent space evolves.

**C1 — Long rollout (N=1000 = 10ms)**: λ_gate=0.3→0.1. λ_roll=1.0. LR reduced. Add voltage clamp protocols (scaffold still active — clamp gives exact gate targets at clamped Vm, maximizing scaffold value).

**C2 — Very long rollout (N=10000 = 100ms)**: λ_gate=0.1→0.01. Covers full upstroke + plateau. Mixed free-running + clamped batches (use clamp mask per segment).

**C3 — Full beat rollout (N=100000 = 1000ms)**: λ_gate=0.01→0.0 (scaffold effectively removed). Model must reproduce complete AP autoregressively. Validation: APD error < 5ms, dVm/dt_max within 10%.

**Training loop detail for rollout**: Model predicts I_ion, not Vm. To compute L_roll and autoregressive Vm:
```
For each step in rollout:
  I_ion_pred = model(latent, Vm, I_stim, dt)
  if clamp_mask[t]:
    Vm_next = V_clamp[t+1]           # voltage clamp: override Vm
  else:
    Vm_next = Vm + dt·(-I_ion_pred + I_stim)/Cm   # free-running: Vm update
  latent = model.latent_new           # carry latent forward
```
L_roll only applies to free-running segments (clamped Vm is externally set, not predicted).

### Phase D — Robustness

Hard data, no scaffold, stress testing.

**Data**: Add Tiers 5-12 (tissue injection, voltage clamp, concentrations, long-duration, corruption recovery, tissue-specific, combined stressors, stitched protocols).

**Scaffold removed** (λ_gate=0). Monitor gate reconstruction error passively — if it diverges, re-enable scaffold at low weight.

**D1 — Tissue-mimicking**: OU noise, sustained currents, biphasic pulses. Full-beat rollout.

**D2 — Stress testing**: Corruption recovery, combined stressors, stitched protocols. 5-beat rollout (5000ms). Validation: 5-beat rollout stable, restitution curve correct, recovery from perturbation within 1 beat.

### Training Timeline

| Phase | Step | Data | Rollout | λ_gate | λ_roll | LR | Weight Decay | Batch |
|-------|------|------|---------|--------|--------|-------|-------------|-------|
| A | — | gate states | — | — | — | 1e-3 | 1e-5 | 4096 |
| B | B1 | Tier 1 | 1 | 1.0 | 0 | 1e-3 | 1e-4 | 1024 |
| B | B2 | Tier 1 | 10 | 1.0 | 0.1 | 1e-3 | 1e-4 | 512 |
| B | B3 | Tier 1 | 100 | 0.5 | 0.5 | 5e-4 | 1e-4 | 256 |
| C | C1 | Tier 1-4 + clamp | 1000 | 0.3→0.1 | 1.0 | 5e-4 | 5e-4 | 128 |
| C | C2 | Tier 1-4 + clamp | 10000 | 0.1→0.01 | 1.0 | 2e-4 | 5e-4 | 64 |
| C | C3 | Tier 1-4 + clamp | 100000 | 0.01→0 | 1.0 | 1e-4 | 5e-4 | 32 |
| D | D1 | Tier 1-12 | 100000 | 0 | 1.0 | 5e-5 | 1e-3 | 32 |
| D | D2 | + stress | 500000 | 0 | 1.0 | 2e-5 | 1e-3 | 8 |

Estimated total: ~40 GPU-hours on Blackwell.

### Optimizer

- **AdamW** with variable weight decay (1e-5 → 1e-3 across phases, increase when overfitting detected)
- **LR schedule**: cosine decay within each phase, reset at phase transitions
- **Gradient clipping**: max_norm=1.0
- **Batch size**: decreases as rollout length increases (memory-bound)
- **Early stopping**: per-phase, based on validation I_ion error
- **Checkpointing**: save best model per phase, resume from best if training destabilizes

### Transition Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Phase B→C: complex data destabilizes latent | Gradual data mixing (90/10 → 50/50 → 10/90) |
| Scaffold removal: latent drifts without gate loss | Monitor gate reconstruction passively; re-enable at low weight if divergent |
| Rollout length jumps: error accumulation spikes | Double rollout length each time (100→200→400→800→1000), not 10× jumps |
| Teacher forcing → autoregressive gap (exposure bias) | Scheduled sampling: mix ground-truth and predicted latent inputs (10%→100% model predictions) |
| Overfitting at any phase | Increase weight decay 2-3×; if persistent, add data or reduce model capacity |

### Advanced Data Strategies

- **Phase space coverage monitor**: Track (Vm, dVm/dt, Ca_i) coverage. Add protocols for uncovered regions.
- **Curriculum over complexity**: Train Tier 1 → add Tier 2 when loss plateaus → Tier 3 → etc.
- **Adversarial protocol generation**: After initial training, gradient-search for pacing patterns that maximize model error. Generate those protocols, retrain.
- **Interpolation ground truth**: Generate at dt=0.001ms (gold standard), subsample to training dt values. Validates dt-invariance.

### Data Budget

All single-cell ODE — runs in seconds per protocol on GPU. Estimated ~50,000 AP beats across all tiers and augmentations. Fits in memory trivially. Bidomain V1 tissue profiles (Tier 5) require a few tissue runs (~minutes each).

## Diffusion Component — Cross-Skip Coupled ResNet

Architecture unchanged from original design (not yet revisited this session):
- Dual conv paths (Vm, phi_e) with bidirectional 1×1 cross-skip connections at each block
- Monodomain single-path baseline first, then bidomain upgrade
- Upgrade path if phi_e accuracy insufficient: dilated conv → U-Net → local Transformer → FNO

## Known Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Stage 2 breaks Stage 1 contractivity | HIGH | Spectral normalization on W_cc (||W_cc||₂ ≤ 1) |
| Error accumulation over 100K+ autoregressive steps | HIGH | Gate decoder scaffold per-step; gradient checkpointing for memory; validate with 5000ms+ rollouts |
| Parallel scan incompatible with Stage 2 | HIGH | Sequential backprop feasible for tiny model; or train Stage 1 with scan, fine-tune Stage 2 |
| Ca handling is compartmental, not gate-like | MEDIUM | Monitor Ca-related gate predictions; add explicit Ca dims (mod A5) if needed |
| Charge/concentration drift | MEDIUM | Monitor Na/K/Ca equivalents; add conservation penalty if needed |
| Overfitting in Phase B (466 params, ~180 APs) | MEDIUM | Weight decay, extend Tier 1 data, monitor train/val gap |
| Encoder-dynamics latent mismatch | MEDIUM | Monitor encoder(gates) vs dynamics latent in Phase B validation; gate decoder loss ensures functional equivalence |
| Concentration tracking implicit | MEDIUM | Model infers concentrations from AP shape; if fails, add explicit inputs (mod A5) |
| Linear readout too weak for m³·h·j | LOW | Upgrade to KAN readout (+47 FLOPs) or nonlinear MLP (+300 FLOPs) |
| Linear gate decoder insufficient | LOW | Upgrade to MLP decoder (16→32→18) if reconstruction error high |

## Competitive Landscape (as of 2026-03-18)

No existing bidomain surrogate models. Our approach is unique.

| Approach | Paper | Key difference from ours |
|----------|-------|------------------------|
| **AGATA** (GNN) | Morier et al., FIMH 2025 | Monodomain, Mitchell-Schaeffer (2-var), no phi_e, 12× speedup vs FEM |
| **FNO/KOL** | Centofanti et al., PLOS CB 2025 | Single-shot AT/RT maps, not timestep simulation |
| **PINO** | Lydon et al., arXiv 2025 | PINN-adjacent, monodomain, 10× resolution generalization |
| **LNODE** | Salvador et al., npj Dig Med 2024 | 0D hemodynamic outputs, 300× speedup, different scope |
| **BLNMs** | Martinez et al., CMAME 2025 | Single-shot activation maps, geometry atlas |

Our differentiators: (1) only bidomain surrogate, (2) only biophysically detailed ionic model, (3) only physics-aware architecture, (4) universal ionic latent space, (5) designed for calcium imaging transfer learning.

## Open Questions

- How does autoregressive error accumulation scale with rollout length? (empirical, no theory)
- Is the split GELU 8/8 split optimal, or does a different ratio work better?
- How many ResNet blocks are needed for adequate phi_e receptive field?
- Will cross-skip 1×1 convolutions suffice for Vm↔phi_e coupling?
- What pacing protocols are needed in training data for robust restitution?
- Is the monodomain → bidomain transfer (reusing Vm path weights) effective?

## Connections
- **Engines**: Bidomain V1 (training data source), Monodomain V5.4 (monodomain baseline)
- **Related research**: [Boundary conduction speedup](../boundary_conduction_speedup/) — surrogate must reproduce Kleber effect; [Engine consolidation](../engine_consolidation/) — unified API would simplify data generation
- **Pipelines**: Optimizer V1 (surrogate could replace simulator in optimization loop)
- **Future**: Calcium imaging transfer learning — fine-tune ionic surrogate on real Ca²⁺ fluorescence data
