# Surrogate Pipeline: Architecture Specification

## Overview

Two-component autoregressive surrogate for bidomain cardiac simulation. Mirrors operator splitting:

1. **Ionic Transformer** — per-node, temporal: voltage history → attention → predicted gates → I_ion
2. **Cross-Skip Coupled ResNet** — spatial, 2D: dual conv paths with bidirectional skip connections for Vm ↔ phi_e coupling

Built in stages: monodomain baseline (single conv path) first, then bidomain upgrade (add phi_e path + cross-skips).

---

## Design Principles

1. **Mirror the physics** — two-component split mirrors operator splitting (ionic → diffusion), enabling independent training and interpretability
2. **Voltage-only input to Transformer** — gates are predicted (intermediate output), not provided as input; the AP waveform encodes ionic state
3. **Latent gate encoding** — gate states compressed to shared latent space for cross-model comparison and dimensionality reduction
4. **Cross-skip coupling** — bidirectional skip connections between Vm and phi_e paths learn the asymmetric bidomain coupling terms
5. **Sequential training** — train each component in isolation on cheap data before expensive end-to-end fine-tuning
6. **Monodomain first** — validate full pipeline with single conv path before adding bidomain coupling complexity
7. **Progressive upgrade path** — if cross-skip ResNet insufficient for elliptic phi_e, upgrade to dilated conv → U-Net → local Transformer → FNO

---

## Scaffold vs Production Architecture

The ionic component has two modes:

### Production Architecture (final goal)

```
Vm history → Attention → MLP → latent_state → MLP_ion → I_ion
```

No gates anywhere. The latent space is an abstract, model-agnostic representation of ionic state.
MLP_ion is **universal** — the same network regardless of which ionic model generated the training data.
The Attention block is **model-specific** — different weights for TTP06 vs ORd, but same architecture.

### Scaffolded Architecture (training)

```
Vm history → Attention → MLP → latent_state ──→ MLP_ion → I_ion   (I_ion loss)
                                     |
                                     └──→ Gate Decoder → gates     (auxiliary gate loss)
```

The gate decoder is **training scaffolding** — it forces the latent space to encode physically
meaningful ionic state information during early training. Benefits:
1. Provides supervised signal that bootstraps the latent space
2. Interpretability during development (decode → inspect predicted gates)
3. Can be removed once the model trains well without it

The gate decoder is **not** part of the production model.

### Parallel Gate Decoders for Cross-Model Comparison

Multiple gate decoders can be trained in parallel on the same latent space:

```
                              ┌→ Decoder_TTP06 → 18 gates  (trained on TTP06 data)
latent_state ─────────────────┤
                              └→ Decoder_ORd   → 40 gates  (trained on ORd data)
```

If TTP06 and ORd trajectories map to similar latent regions, the latent space has learned a
universal ionic representation. If they map differently, the differences reveal how the models
diverge — a scientific insight tool.

---

## Component 1: Gate Decoder (Training Scaffold)

### Purpose

Decode the latent ionic state back to full gate variables for a specific ionic model.
Used as an auxiliary training loss to bootstrap the latent space. Removed after training.

### Architecture

Per ionic model:
```
Decoder_TTP06: (d_latent,) → MLP(d_latent, h, h, 18) → (18,)
Decoder_ORd:   (d_latent,) → MLP(d_latent, h, h, 40) → (40,)
```

Hidden dim h TBD (likely 32-64). Activation: ReLU or GELU.

### Training

- Data: single-cell trajectories from the target ionic model
- Loss: reconstruction MSE on gate states
- Trained jointly with the ionic Transformer (not as a separate pre-training step)
- Multiple decoders can train in parallel on different ionic model data

### Interface

```python
class GateDecoder(nn.Module):
    """Training scaffold — decodes latent state to model-specific gates."""
    def __init__(self, d_latent: int, n_gates: int, hidden: int = 64):
        ...
    def forward(self, latent: Tensor[..., d_latent]) -> Tensor[..., n_gates]
```

### Lifecycle

| Stage | Gate Decoder Role |
|-------|-------------------|
| Stage A (single-cell training) | Active — provides auxiliary loss to shape latent space |
| Stage C (end-to-end fine-tuning) | Optional — can keep for monitoring or remove |
| Production inference | Removed — only Attention → MLP → latent → MLP_ion remains |

---

## Component 2: Ionic Transformer

### Purpose

Predict ionic current at t+1 for each spatial node independently. Takes **voltage history only** as input, produces a **latent ionic state** (abstract, model-agnostic), then uses a **universal MLP_ion** to compute I_ion from latent state + current Vm.

### Why Voltage-Only Input

The AP waveform shape encodes which phase the cell is in (upstroke, plateau, repolarization), which determines ionic state. This means:
- Input is simpler: just Vm history (1 value per time point)
- No need to carry gate history in the lookback buffer
- The network learns to *infer* ionic state from the voltage trace — a form of state estimation

### Data Flow

```
Per node:

  Vm[t0, t1, ..., tm-1]                     Voltage history (300 time points)
         |                                   (t0 = current time, tm-1 = most distant)
         v
  Positional Encoding (non-uniform)          Encodes temporal spacing
         |
         v
  Transformer Encoder (N layers)             Self-attention over temporal sequence
         |
         v
  Final token representation
         |
         v
  MLP → latent_state(t)                     Abstract ionic state (d_latent dims)
         |
         +-----> [Gate Decoder → gates]      SCAFFOLD ONLY: auxiliary training loss
         |
  Vm(tN) + latent_state(t)                  tN = current time = t0 in the buffer
         |
         v
  MLP_ion → I_ion(tN+1)                     Universal current calculator
         |
         v
  Vm_post_ionic = Vm(tN) + dt*(-I_ion + I_stim)/Cm
```

### What is the Latent State?

The latent state is **not** a compressed gate vector — it is an abstract representation of
"where in the ionic dynamics cycle this cell is." During scaffolded training, the gate decoder
forces it to correlate with physical gate states. But the latent space can evolve beyond gates
once the scaffold is removed.

The key property: **the same latent space and the same MLP_ion are used regardless of which
ionic model (TTP06, ORd, etc.) generated the training data.** The Attention block is the only
model-specific component.

### What is Model-Specific vs Universal?

| Component | Scope | Why |
|-----------|-------|-----|
| Attention block | Model-specific | Different ionic models have different temporal dynamics |
| MLP (→ latent) | Model-specific | Part of the Attention → latent pipeline |
| Latent space | Universal | Shared representation across all ionic models |
| MLP_ion | Universal | Maps (latent, Vm) → I_ion regardless of model |
| Gate Decoder | Model-specific (scaffold) | TTP06 decoder has 18 outputs, ORd has 40 |

### Input

Per node, a sequence of 300 tokens:

```
Token at lookback index k:
  - Vm[t - tau_k]              (1 value)
  - positional_encoding(tau_k)  (d_pe values, encodes non-uniform temporal position)
```

Where tau_k is the non-uniform lookback offset (dense near current time, sparse for distant past).

### Non-Uniform Temporal Sampling

300 lookback points with spacing that increases with distance from current time:

```
tau_0 = 0 (current time)
tau_1 = dt
tau_2 = 2*dt
...
tau_k = f(k)  where f grows sub-linearly
```

Key constraint: must resolve the fast upstroke (~1ms, ~100 points at dt=0.01ms) while covering the full AP duration (~300ms, ~30000 simulator steps).

Candidate schedules:
- **Logarithmic**: tau_k = dt * exp(k * log(T_max/dt) / 299)
- **Power law**: tau_k = dt * (k/299)^alpha * T_max/dt
- **Piecewise linear**: three segments with different densities

Exact schedule TBD — will be tuned empirically.

### Output

Per node:
- latent_state(t): abstract ionic state (d_latent values) — intermediate, model-agnostic
- I_ion(tN+1): predicted ionic current (scalar) — final output

### Vm Update (Ionic Step)

```
Vm_post_ionic = Vm(tN) + dt * (-I_ion(tN+1) + I_stim(tN)) / Cm
```

This mirrors the forward Euler ionic step in the simulator.

### Architecture Details

```
Input: (batch, 300, 1)  — voltage history per node
  → Linear embedding: (batch, 300, d_model)
  → Add positional encoding: non-uniform-aware sinusoidal or learned
  → Transformer encoder: N_layers blocks (self-attention + FFN)
  → Take final token (or CLS token): (batch, d_model)
  → MLP_latent: (batch, d_model) → (batch, d_latent)  [latent ionic state]
  → Concat [latent_state, Vm(tN)]: (batch, d_latent + 1)
  → MLP_ion: (batch, d_latent + 1) → (batch, 1)  [I_ion prediction]

During scaffolded training only:
  → Gate Decoder: (batch, d_latent) → (batch, n_gates)  [auxiliary loss]
```

Hyperparameters TBD: d_model, N_layers, n_heads, d_latent.

### Per-Node Operation

The Transformer has **no spatial coupling** — it processes each node independently. During training, nodes across all spatial locations can be batched together. This mirrors the fact that ionic dynamics are local ODEs with no diffusion coupling.

### Training Losses

**Scaffolded training (with gate decoder):**
- **Gate loss (scaffold)**: MSE(GateDecoder(latent_state), true_gates) — forces latent to encode gate info
- **I_ion loss**: MSE(MLP_ion(latent_state, Vm), true_I_ion) — primary objective
- **Autoregressive rollout loss**: accumulated Vm error over N-step rollout

**Production training (scaffold removed):**
- **I_ion loss**: MSE(MLP_ion(latent_state, Vm), true_I_ion) — sole objective
- **Autoregressive rollout loss**: accumulated Vm error

The gate loss can be weighted down or removed as training progresses (curriculum: high gate weight early → zero late).

### Cross-Model Training

TTP06 and ORd can train in parallel with shared latent space + shared MLP_ion:

```
TTP06 data → Attention_TTP06 → MLP → latent → MLP_ion → I_ion  (shared)
                                        |
                                        └→ Decoder_TTP06 → 18 gates  (scaffold)

ORd data   → Attention_ORd   → MLP → latent → MLP_ion → I_ion  (shared)
                                        |
                                        └→ Decoder_ORd   → 40 gates  (scaffold)
```

Both models share the same latent space and MLP_ion. Only the Attention blocks and Gate Decoders
differ. This enables direct comparison of latent trajectories between ionic models.

---

## Component 3: Cross-Skip Coupled ResNet

### Purpose

Predict the diffusion-coupled field update. Takes post-ionic Vm and current phi_e as 2D fields, processes them through dual conv paths with bidirectional cross-skip connections, outputs updated (Vm, phi_e).

### Why Cross-Skip Architecture

In the bidomain equations:
- **Parabolic** (Vm evolution): involves L_i·Vm (local) + L_i·phi_e (coupling from phi_e → Vm)
- **Elliptic** (phi_e determination): involves (L_i+L_e)·phi_e (local) + L_i·Vm (coupling from Vm → phi_e)

The coupling is **asymmetric** — phi_e influences Vm differently than Vm influences phi_e. Cross-skip connections at each block level let the network learn this asymmetric coupling naturally:

- **Same-path skip**: standard ResNet residual — learns the Laplacian correction
- **Cross-path skip (phi_e → Vm)**: learns the coupling term L_i · phi_e
- **Cross-path skip (Vm → phi_e)**: learns the source term L_i · Vm

### Architecture

```
Block k:

  Vm_feat[k-1] ──→ ResBlock_i(·) ──→ (+) ──→ Vm_feat[k]
                                       ↑
                                  Conv_cross_e2i(phi_e_feat[k-1])    # phi_e → Vm coupling

  phi_e_feat[k-1] → ResBlock_e(·) ──→ (+) ──→ phi_e_feat[k]
                                       ↑
                                  Conv_cross_i2e(Vm_feat[k-1])       # Vm → phi_e coupling
```

Expanded:
```python
# At each block k:
Vm_feat[k]    = ResBlock_i(Vm_feat[k-1])    + Conv_1x1(phi_e_feat[k-1])
phi_e_feat[k] = ResBlock_e(phi_e_feat[k-1]) + Conv_1x1(Vm_feat[k-1])
```

### Full Data Flow

```
Input:
  Vm_post_ionic  (Nx, Ny)  →  Conv_stem_i  →  Vm_feat_0  (Nx, Ny, C)
  phi_e(t)       (Nx, Ny)  →  Conv_stem_e  →  phi_e_feat_0  (Nx, Ny, C)

K coupled blocks:
  (Vm_feat_0, phi_e_feat_0) → Block_1 → Block_2 → ... → Block_K → (Vm_feat_K, phi_e_feat_K)

Output heads:
  Vm_feat_K    →  Conv_head_i  →  Vm(t+1)     (Nx, Ny)
  phi_e_feat_K →  Conv_head_e  →  phi_e(t+1)  (Nx, Ny)
```

### ResBlock Structure

Each ResBlock is a standard pre-activation residual block:
```
input → BN → ReLU → Conv3x3 → BN → ReLU → Conv3x3 → (+input) → output
```

### Cross-Skip Conv

The cross-path connections use 1×1 convolutions (learned channel mixing, no spatial kernel) to keep the coupling lightweight. The spatial coupling is already handled by each path's own ResBlocks — the cross-skip just injects the other domain's information.

If 1×1 proves insufficient, can upgrade to 3×3 cross-skip convs (learns local spatial coupling between domains).

### Monodomain Simplification

For the monodomain baseline (Step 1), the architecture reduces to a single path with no cross-skips:

```
Vm_post_ionic → Conv_stem → ResBlock ×K → Conv_head → Vm(t+1)
```

This is just a standard ResNet. The Vm path becomes `ResBlock_i` in the bidomain version — no wasted work.

### Receptive Field Consideration

With K blocks of 3×3 convolutions, the effective receptive field is ~(2K+1) grid points per side. The elliptic equation for phi_e is **global** — change Vm at one point, phi_e updates everywhere.

For most practical cases, the coupling decays within a few electrotonic length constants (λ ≈ 56 grid points at dx=0.025cm). With K=10-20 blocks, receptive field covers ~21-41 grid points — within one λ. This may be sufficient.

### Upgrade Path (If Cross-Skip ResNet Insufficient)

If phi_e accuracy is poor due to limited receptive field:

1. **Dilated convolutions**: replace some 3×3 convs with dilated 3×3 (dilation=2,4,8). Receptive field grows exponentially. Minimal cost increase.

2. **U-Net bottleneck**: add downsample → bottleneck → upsample in the joint processing. Global context through spatial compression. Proven for PDE surrogates.

3. **Local Transformer**: windowed cross-attention (W=8) between Vm and phi_e features within spatial patches. Cost O(W²·N) vs O(N²) for global. Rich pairwise interactions.

4. **FNO spectral layer**: FFT → learned filter → IFFT on the phi_e path. Directly mirrors the simulator's spectral elliptic solver. Global receptive field in one layer, O(N log N).

Each upgrade is additive — the cross-skip ResNet skeleton remains, with the upgrade inserted at specific points.

---

## Full Surrogate: Composed Model

### Forward Pass (One Time Step)

```python
def step(self, Vm_t, phi_e_t, history_buffer):
    # 1. Ionic Transformer (per-node)
    #    Input: Vm history from buffer (300 non-uniform samples)
    #    Output: latent ionic state → I_ion
    latent_state = self.ionic_transformer.predict_latent(history_buffer)
    I_ion = self.mlp_ion(latent_state, Vm_t)  # universal current calculator
    Vm_post_ionic = Vm_t + self.dt * (-I_ion + I_stim) / self.Cm

    # 2. Cross-Skip Coupled ResNet (spatial)
    Vm_next, phi_e_next = self.diffusion_resnet(Vm_post_ionic, phi_e_t)

    # 3. Update history buffer (push new Vm, drop oldest)
    history_buffer.push(Vm_next)

    return Vm_next, phi_e_next, latent_state
```

### Autoregressive Rollout

```python
def rollout(self, Vm_0, phi_e_0, n_steps):
    history = HistoryBuffer(Vm_0, schedule=self.temporal_schedule)
    Vm, phi_e = Vm_0, phi_e_0

    for step in range(n_steps):
        Vm, phi_e, latent_state = self.step(Vm, phi_e, history)
        yield Vm, phi_e, latent_state
```

### History Buffer

Maintains the 300-point non-uniform temporal lookback of **Vm only**. The latent state is
predicted fresh at each step from the voltage history — it is not stored or carried forward.

On each step:
- New Vm is pushed to index 0 (current time)
- Existing entries shift according to the non-uniform schedule
- Entries that fall between schedule points are discarded or interpolated

---

## Sequential Training Strategy

### Stage A: Ionic Component (Single-Cell Data)

**Data source**: Single-cell ODE simulations (no spatial coupling)
- Cheap to generate: one ODE integration per sample
- Unlimited data: vary initial conditions, stimulus timing, pacing rate, BCL
- TTP06 and ORd can train in parallel with shared latent space

**Training approach**:
- **Scaffolded phase**: Attention → MLP → latent → MLP_ion → I_ion, with gate decoder providing
  auxiliary loss. Gate loss weighted high initially, annealed toward zero.
- **Production phase**: Remove gate decoder, fine-tune on I_ion loss + rollout loss only.

**Curriculum**:
1. Single-step: Vm history → (latent_state, I_ion) with gate scaffold loss
2. Short rollout (N=10 steps): accumulate Vm error, gate scaffold still active
3. Medium rollout (N=100): anneal gate loss weight toward zero
4. Long rollout (N=1000): gate scaffold removed, I_ion + rollout loss only

**Parallel cross-model training** (optional, can defer):
- Train Attention_TTP06 + Decoder_TTP06 on TTP06 data
- Train Attention_ORd + Decoder_ORd on ORd data
- Both share the same MLP_ion and latent space
- Compare latent trajectories for scientific insight

**Validation**: single-cell AP reproduction over 1000ms autoregressive rollout

### Stage B: Diffusion Component (Pure Diffusion Data)

**Data source**: Diffusion-only simulations (ionic activity disabled)

**Sub-stages**:
1. **B1: Monodomain single-path** — single ResNet on Vm-only diffusion (from Monodomain V5.4)
2. **B2: Bidomain cross-skip** — add phi_e path + cross-skip connections (from Bidomain V1)

B1 validates the spatial conv pipeline. B2 adds the coupling. The Vm path from B1 initializes ResBlock_i in B2.

**Validation**: diffusion of Gaussian blob matches simulator within O(h²)

### Stage C: End-to-End Fine-Tuning (Full Simulation Data)

**Data source**: Full bidomain simulations (ionic + diffusion coupled)
- Start with frozen ionic + diffusion components (trained in A, B)
- Gradually unfreeze with small learning rate
- Autoregressive rollout loss over increasing horizon

**Validation**: full AP propagation, conduction velocity within 5% of simulator

---

## Temporal Schedule

The 300-point non-uniform lookback schedule for the history buffer.

### Requirements

- Must resolve fast upstroke (~1ms, ~100 points at dt=0.01ms)
- Must cover full AP duration (~300ms, ~30000 simulator steps)
- Dense near current time, sparse for distant past
- Total buffer stores only 300 Vm values per node (memory-efficient)

### Candidate Schedules

**Logarithmic**: tau_k = dt * exp(k * log(T_max/dt) / 299)
- Uniform in log-time
- Good balance of near/far coverage

**Power law**: tau_k = T_max * (k/299)^alpha
- alpha=1: uniform. alpha=2: quadratic concentration near t=0
- Tunable knob for density distribution

**Piecewise linear**: three segments
- [0, 99]: dt spacing (first 1ms)
- [100, 199]: 10*dt spacing (1-10ms)
- [200, 299]: 100*dt spacing (10-300ms+)

Exact schedule TBD — will be determined empirically based on AP dynamics.

---

## Key Design Decisions Log

| # | Decision | Choice | Rationale | Date |
|---|----------|--------|-----------|------|
| D1 | Component structure | Two-component (ionic + diffusion) | Mirrors operator splitting; enables staged training | 2026-03-11 |
| D2 | Ionic architecture | Transformer (per-node, temporal) | Sequence modeling; no spatial coupling needed | 2026-03-11 |
| D3 | Transformer input | Voltage history only | Latent state predicted, not input; AP waveform encodes ionic state | 2026-03-12 |
| D4 | Ionic prediction flow | Vm history → attention → MLP → latent_state → MLP_ion → I_ion | Latent state is model-agnostic; MLP_ion is universal | 2026-03-12 |
| D5 | Latent space | Abstract ionic state (not compressed gates) | Universal across ionic models; enables cross-model comparison | 2026-03-12 |
| D6 | Gate decoder | Training scaffold only — removed for production | Forces latent to encode meaningful state; provides supervised bootstrap | 2026-03-12 |
| D7 | MLP_ion universality | Same MLP_ion for TTP06, ORd, any model | Latent space is the shared interface; only Attention block is model-specific | 2026-03-12 |
| D8 | Cross-model training | Parallel decoders (TTP06 + ORd) on shared latent | Scientific insight into ionic model similarity + validates universal latent | 2026-03-12 |
| D9 | Diffusion architecture | Cross-skip coupled ResNet | Bidirectional skip connections learn asymmetric Vm↔phi_e coupling | 2026-03-12 |
| D10 | Cross-skip mechanism | 1×1 conv on other path's features, added to own path | Lightweight; spatial coupling handled by each path's own ResBlocks | 2026-03-12 |
| D11 | Temporal input | 300-point non-uniform lookback (Vm only) | Resolve upstroke + cover full AP; no gate/latent storage needed | 2026-03-12 |
| D12 | Training strategy | Staged: A(scaffolded ionic) → B1(mono) → B2(bi) → C(e2e) | Decompose complexity; cheap data first; monodomain baseline | 2026-03-12 |
| D13 | Build order | Monodomain single-path first, then bidomain upgrade | Validates pipeline; Vm path reused as ResBlock_i | 2026-03-12 |
| D14 | Surrogate dt | Same as simulator (0.01ms) | Speedup from compute, not temporal compression | 2026-03-11 |
| D15 | Grid size | Fixed (Nx × Ny) | Simplifies conv architecture; variable grids deferred | 2026-03-11 |
| D16 | Elliptic upgrade path | Cross-skip → dilated conv → U-Net → local Transformer → FNO | Progressive if phi_e accuracy insufficient | 2026-03-12 |
| D17 | Mask inputs (BC, infarct) | Deferred | Extra input channels when needed | 2026-03-11 |
| D18 | Predict I_ion (not skip to Vm) | Keep explicit I_ion prediction for now | Interpretable; skip-to-Vm is future option | 2026-03-12 |
