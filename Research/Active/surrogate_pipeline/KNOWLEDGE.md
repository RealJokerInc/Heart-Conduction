# Surrogate Pipeline -- Knowledge File

> Reference document. Updated as findings accumulate.
> When the question is complete, a copy is promoted to `Research/Knowledge/`.

---

## 1. Overall Architecture

Two-component autoregressive surrogate mirroring the bidomain simulator's operator splitting:

1. **Ionic Surrogate** (per-node, local): pure autoregressive latent -- carried state updated each step via n x 1 cross-attention to Vm. Predicts I_ion from latent + Vm + reversal potentials + concentrations. No Vm history buffer.
2. **Cross-Skip Coupled ResNet** (spatial, global): dual conv paths for Vm and phi_e with bidirectional 1x1 cross-skip connections. Replaces the elliptic solve (PCG/GMG) which dominates 94% of bidomain wall time.

Per-step inference: `IonicSurrogate (or Rush-Larsen) -> Vm_post_ionic -> ResNet -> (Vm_next, phi_e_next)`

### Compute Insight -- Critical Path vs Background

The diffusion step (94% of wall time) only needs I_ion. Stage 1 outputs (updated latent, compressed conductance) are needed at the NEXT timestep. On GPU, Stage 1 runs on a separate CUDA stream in parallel with the diffusion solve -- effectively free compute. Only Stage 2 (current readout) is on the critical path.

```
carried_state(t), Vm, dt
  |-> Stage 1 (off critical path): state evolution -> carried_state(t+1), conductance_latent(t+1)
  |                                  Nernst -> reversal_potentials(t+1)
  |                                  [runs during diffusion step]
  |
  +-> Stage 2 (ON critical path):  current readout -> I_ion(t)
                                    [must complete before diffusion step begins]
```

Stage 1 design prioritizes accuracy (can be as expressive as needed). Stage 2 design prioritizes speed.

---

## 2. Design Philosophy

### Layer 0 Reasoning Framework

Architectural decisions are derived from three layers, prioritized top-down:

**Layer 0 -- Physical reality (neurophysiology):** What actually happens at the cardiac membrane. Ground truth. The architecture must capture these truths. Future goal: train from real arc light (optical mapping) data, so the architecture must not be locked to simulator assumptions.

**Layer 1 -- Biophysics models (TTP06, ORd):** Human-constructed theories explaining Layer 0. Useful, validated, contain assumptions. HH gating, independent gates, compartmental Ca cycling, specific Markov state diagrams -- all are modeling choices, not measured physics.

**Layer 2 -- Neural architecture:** Should be inspired by what Layer 1 got right about Layer 0, but not enslaved to Layer 1's specific formulations. Where a neural network can learn the relationship more naturally than the biophysics equation prescribes, let it.

### Design Maxims

- "Ground truth is reality, not models" -- don't bake in TTP06/ORd assumptions
- "Physics provides the skeleton, ML fills the flesh"
- Physics-informed elements: Nernst equation (fixed), operator splitting structure, explicit concentrations, Kirchhoff's law (I_ion = sum of channel contributions)
- ML-learned elements: attention dynamics, MLP coupling, compression, readout routing

### Layer 0 Physical Truths

| Reality | What biophysics models say | Assessment |
|---------|---------------------------|------------|
| Channels respond to voltage (conformational changes) | HH: dg/dt = (g_inf(V)-g)/tau(V). Markov: rate matrix. | Good pattern (relax toward equilibrium), but independence of gates and specific kinetics are assumptions |
| Current = conductance x driving force | I = g_max x P_open x (V-E) | Solid physics (Ohm's law). But P_open = m^3*h*j is HH-specific; Markov uses state occupancy |
| Concentrations change slowly, creating memory | dCa/dt = -I_Ca/(2FV_cell) + fluxes | Physics is right, compartment structure is assumed. TTP06: 3 Ca compartments. ORd: 4. Reality: continuous gradients |
| Multi-timescale dynamics | Separate tau per gate (0.1ms to 100ms+) | Real phenomenon, but clean separation into named gates is artificial |
| Cross-channel coupling | Hardcoded equations (I_CaL->Ca_i->fCa, CaMKII) | Coupling exists, but wiring diagram differs between models. All coupling is SLOW (timescale >> dt) |

### Physical Task Decomposition

| Physical task | What happens | Timescale within 1 dt (0.01ms) | Architecture |
|---------------|-------------|-------------------------------|--------------|
| A: Conformational dynamics | Channel proteins change shape in response to Vm | Per-state, independent within dt | Stage 1 (per-dim cross-attention) |
| B: Cross-state coupling | Ca cycling, CaMKII, concentration feedback | Negligible within dt (~0.0001% change). All coupling is TEMPORAL -- accumulates over many steps | Stage 1 latent memory (no dedicated stage) |
| C: Current summation | Static function: state + Vm -> I_ion. Cross-dim products for open probability | Instantaneous (no dynamics) | Stage 2 (readout with cross-dim interaction) |

---

## 3. v3 Stage 1 -- State Evolution (off critical path)

### Carried State

```
carried_state = [ionic_state(32), Na_i, K_i, Ca_i, Ca_ss] = (36,)
                 |-- latent ---|  |---- explicit, named ---|
```

- **ionic_state** (32 dims): latent, learned, encodes channel conformational states (gates, Markov occupancies)
- **concentrations** (4 dims): explicit named variables [Na_i, K_i, Ca_i, Ca_ss]
- Ca_SR dropped -- purely internal SR variable, not used in any reversal potential or readout feature. SR calcium load tracked implicitly by ionic_state.
- **Working dim = 32**: TTP06 has 18 states, ORd has 41. Working dim must accommodate both. 32 is comfortable for TTP06, sufficient for ORd with learned compression. Increase to 48+ for ORd if needed -- cost stays in background.

### Full Pipeline

```
carried_state(t) = [ionic_state(32), concentrations(4)]     (36,)
  -> n x 1 cross-attention to [Vm, dt], attn_dim=4          per-dim voltage response (contractive)
      36 dims all attend to Vm; concentration dims learn slow tracking
  -> SPLIT after attention:
      ionic_mid = z_mid[:32]                                 (32,) -> continues to MLP
      conc(t+1) = z_mid[32:36]                               (4,) -> DONE (attention only, no MLP)

  ionic_mid:
  -> Pre-RMSNorm                                             stabilize MLP input scale (0 params)
  -> learned-mix MLP + GELU (32->32->32), 1 hidden layer     cross-dim Markov correction
      correction = W2 @ GELU(W1 @ RMSNorm(ionic_mid) + b1) + b2
      alpha = sigmoid(w)                                     per-dim mixing weight (32 params)
      ionic(t+1) = (1 - alpha) * ionic_mid + alpha * correction

  carried_state(t+1) = [ionic(t+1), conc(t+1)]             (36,)

  -> learned-mix compression (ionic_state only):
      linear_path = W_lin @ ionic_state                      (32->16)
      nonlinear_path = MLP(ionic_state)                      (32->24->24->16) gate products
      beta = sigmoid(w_beta)                                 per-dim mixing weight (16 params)
      conductance_latent = (1-beta) * linear_path + beta * nonlinear_path
  -> conductance_latent(t+1)                                 (16,)

  -> Nernst equation (fixed physics, 0 learned params):
      concentrations(t+1) -> [E_Na, E_K, E_Ca, E_Ks]       (4,) reversal potentials

Scaffold decoders (training only, all annealed to zero in Phase D):
  ionic_state(t+1)        -> full gate decoder (32->12) -> gate predictions (12 HH gates only)
  conductance_latent(t+1) -> compressed gate decoder (16->12) -> gate predictions
  concentrations(t+1)     -> direct MSE vs true concentrations (no decoder needed)
  Losses:
    L_gate_full = MSE(full_decoder(ionic_state), true_gates)
    L_gate_comp = MSE(comp_decoder(conductance_latent), true_gates)
    L_conc = MSE(concentrations, true_concentrations)     <- direct, no decoder
```

### Design Rationale

**n x 1 cross-attention:**
- Each of 36 carried_state dims independently queries the voltage [Vm, dt]. Produces a per-dim gate (how much to update) and per-dim target (where to move toward).
- Mathematically equivalent to learned Rush-Larsen but with state-dependent gating (the attention score depends on BOTH Vm AND the current latent value, while RL's rate depends on Vm only).
- **attn_dim = 4** (down from 8 in v2): Gate is one scalar from 2 inputs (Vm, dt). Smooth sigmoid-like function -- 4 basis functions suffice.
- **1 attention layer**: One voltage response per dt is physically correct. Stacking applies contraction twice to same Vm -- numerically redundant.

**Concentrations split off after attention, before MLP:**
- The MLP handles intra-protein Markov corrections between ionic dims only.
- Passing concentrations through the MLP would create artificial ionic<->concentration coupling that doesn't exist within one dt.
- Attention is sufficient for concentration tracking: Vm strongly correlates with gate state during normal pacing, concentration changes are ~0.0001% per step, self-regulation via own-value feedback, end-to-end training through I_ion -> Nernst catches systematic errors.

**Markov MLP (32->32->32, 1 hidden layer, 1 GELU):**
- Cross-dim correction over ionic_state dims only (32 dims). No bottleneck -- architecture ALLOWS coupling but does not FORCE it.
- HH gates that need no coupling pass through residual unchanged.
- Pre-RMSNorm on MLP input stabilizes input scale across 100K+ recurrent steps.
- One hidden layer captures pairwise interactions. Add second layer if insufficient.

**Learned residual mixing (inspired by DeepSeek hyper-connections):**

```python
alpha = sigmoid(w)                         # w learned per-dim (32,), alpha in (0,1)
correction = MLP(RMSNorm(ionic_mid))
ionic_new = (1 - alpha) * ionic_mid + alpha * correction   # convex combination
```

No amplification by construction (convex combination bounded between inputs). Each ionic dim learns how much MLP correction to accept: alpha->0 for HH dims (pure residual), alpha>0 for Markov-coupled dims. Init: w = large negative -> alpha~0 -> starts as near-pure residual. 32 learned params.

Replaces three mechanisms from earlier design iterations:
- ~~Spectral norm on MLP weights~~ -- not needed, convex combination can't amplify
- ~~Zero-init of MLP output~~ -- not needed, alpha init handles it
- ~~Gate-modulated correction~~ -- not needed, per-dim alpha is the learned gate

**Compression MLP (CARRIED_DIM->COMP_H1->COMP_H2->COND_DIM, 2 hidden layers, 2 GELUs):**
- Takes FULL carried_state (ionic + concentrations) as input, NOT just ionic_state. Gives compression access to concentration context (e.g., Ca_ss for fCass-dependent conductances). Recomputed every step because gate products change and attention cannot compute cross-dim products (structural limitation).
- Two hidden layers: layer 1 computes pairwise products (m*h), layer 2 composes into triple products (m*h*j).
- Residual compression with learned mixing: `compressed = (1-beta) * W_lin @ carried + beta * MLP(carried)`. Linear path for direct projection, nonlinear path for gate products. Per-dim beta (COND_DIM params) controls linear vs nonlinear.
- Total GELUs in Stage 1 pipeline: 3 (1 in Markov MLP + 2 in compression MLP). Sufficient for triple product composition.

**Dual scaffold decoders (training only):**
- Full decoder (32->12): ensures full latent encodes HH gate information (12 gates: m,h,j,r,s,d,f,f2,fCass,Xr1,Xr2,Xs). NOT 18 — RR has no gate_inf/tau, concentrations have their own direct MSE loss.
- Compressed decoder (16->12): ensures compression preserves gate information.
- Both use L_gate = MSE(gates_pred, gates_true), annealed to zero in Phase D.
- Both removed for production inference.

### Nernst Computation (fixed physics, 0 learned params)

Reads concentration dims directly from carried_state. No learned components.

```
concentrations(t+1) = carried_state(t+1)[32:36]             (4,) [Na_i, K_i, Ca_i, Ca_ss]
  -> Nernst equation:
      E_Na = (RT/F) x ln(Na_o / Na_i)                       Na_o = 140 mM
      E_K  = (RT/F) x ln(K_o / K_i)                         K_o = 5.4 mM
      E_Ca = (RT/(2F)) x ln(Ca_o / Ca_i)                    Ca_o = 2.0 mM
      E_Ks = (RT/F) x ln((K_o + P*Na_o)/(K_i + P*Na_i))    mixed permeability
  -> [E_Na, E_K, E_Ca, E_Ks]                                (4,) reversal potentials
```

Differentiable (log and division). Gradients flow: I_ion -> readout -> E -> Nernst -> concentration dims -> attention.

**Why both E and raw concentrations feed Stage 2:**

| Current type | What it needs | Example |
|---|---|---|
| Ohmic channels | E (reversal potential) | I_Na uses Vm - E_Na |
| GHK (I_CaL) | Raw Ca_ss + Vm | Nonlinear flux equation |
| Pumps/exchangers | Raw Na_i, Ca_i + Vm | I_NaCa, I_NaK |

### Effective Variables for I_ion Computation

| Ionic model | State variables | Effective I_ion inputs | Compression ratio |
|---|---|---|---|
| TTP06 | 18 (13 gates + 5 concentrations) | ~11 (5 conductance products + 5 concentrations + Vm) | 1.6x |
| ORd | 41 (20 gates + 11 Markov + 9 concentrations + CaMKII) | ~13 (6 conductance terms + 6 concentrations + Vm) | 3.2x |

16 compressed dims comfortably exceeds both models' effective input requirements.

### Information Flow

- Only `carried_state` (36 values) is carried forward between timesteps
- `conductance_latent` (16 values) is derived each step as the last part of Stage 1 pipeline
- Stage 2 reads `conductance_latent(t)` (the OLD compressed, before this step's Stage 1 updates)
- Reversal potentials are derived, not carried

---

## 4. v3 Stage 2 -- Current Readout (ON critical path)

### Inputs

```
conductance_latent(t) + Vm + reversal_potentials(t) + concentrations(t) -> readout -> I_ion(t)
        (16,)          (1,)         (4,)                   (4,)                       (1,)
```

25 inputs total: 16 conductance tokens (queries) + 9 environment tokens (keys/values).

### Architecture: Cross-Attention + MLP

**Cross-attention (no softmax):** Each conductance dim queries the environment to determine its current contribution. No softmax because I_ion is unbounded and negative scores are physically meaningful (the E term in Vm - E is subtracted).

```
Q_k = conductance_k × e_q[k]       # 16 queries, each d-dim (d=4)
K_j = environment_j × e_k[j]       # 9 keys, each d-dim
V_j = environment_j × e_v[j]       # 9 values, each d_v-dim (d_v=1)

scores = Q @ K^T / sqrt(d)          # (16, 9) attention matrix
attended = scores @ V                # (16,) per-channel current contributions
```

**Output MLP:** 16 → h → GELU → 1 (h=4). Allows nonlinear channel interactions beyond pure Kirchhoff summation.

```
I_ion = W2 @ GELU(W1 @ attended + b1) + b2     # scalar
```

**Physical interpretation:**
- Each conductance token "asks" the environment: what driving force should I apply?
- Na conductance attends strongly to Vm and E_Na → learns (Vm - E_Na)
- CaL conductance attends to Vm, Ca_ss → learns GHK-like nonlinear flux
- Pump conductance attends to Na_i, Vm → learns I_NaK dependence
- The MLP combines the 16 per-channel contributions, allowing interactions (e.g., I_NaCa involves both Na and Ca)

### Parameters and FLOPs

| Component | Params | FLOPs |
|---|---|---|
| Q embeddings e_q (16, 4) | 64 | 64 |
| K embeddings e_k (9, 4) | 36 | 36 |
| V embeddings e_v (9, 1) | 9 | 9 |
| Q @ K^T (16×9, d=4) | -- | 1152 |
| Scale 1/√4 | -- | 144 |
| scores @ V (16×9, d_v=1) | -- | 288 |
| MLP W1 (16→4) + b1 | 68 | 128 |
| GELU (4) | -- | 20 |
| MLP W2 (4→1) + b2 | 5 | 9 |
| **Total** | **182** | **~1850** |

### Comparison to Ionic Model Steps

| | Total FLOPs | vs TTP06 (~600) | vs ORd (~1700) |
|---|---|---|---|
| Stage 2 readout | ~1850 | 3.1× | 1.1× |
| TTP06 full step | ~600 | 1× | -- |
| ORd full step | ~1700 | -- | 1× |

Total FLOPs are ~1× ORd. Acceptable because Stage 2 is the ONLY thing on the critical path (Stage 1 is free).

**Optional optimization (defer to later):** Queries and 8/9 of keys/values can be precomputed off critical path (only Vm is new each step). Reduces critical path to ~330 FLOPs (0.55× TTP06). Implement if readout becomes the actual bottleneck after diffusion surrogate is built.

### Design Decisions

- **d=4 (attention dim):** 16 queries attending to 9 keys. Each dot product is 4-dimensional -- enough to learn channel-to-environment routing. Larger d adds capacity for diminishing returns.
- **d_v=1 (value dim):** Each environment token provides one scalar value. Sufficient for single-output (I_ion). Upgrade to d_v=2-4 if training shows limitation.
- **h=4 (MLP hidden):** Conservative. Start small, upgrade to h=8 if channel interactions need more capacity (~120 extra FLOPs).
- **No softmax:** I_ion is unbounded. Negative attention scores are physically meaningful (subtraction in driving force). Softmax would force positive weights and add ~1600 FLOPs (144 exp operations).
- **GELU in output MLP:** Allows nonlinear channel interaction. Kirchhoff says currents sum linearly, but our 16 conductance dims are learned (not literal channels) -- their combination may benefit from nonlinearity.
- **Input normalization (required):** Environment tokens have wildly different scales (K_i ~ 138 mM vs Ca_i ~ 0.0001 mM -- six orders of magnitude). Without normalization, low-magnitude tokens (Ca_i, Ca_ss) are invisible to attention (key magnitude ≈ 0). All 9 environment inputs normalized to ~[-1, 1] using known physiological ranges before embedding. 18 fixed constants (9 shifts + 9 scales), not learned. Conductance latent scale is controlled by compression -- normalize if needed.

### Training Strategy (v3, revised)

Single loss per phase, no weighting. Each phase trains one thing.

| Phase | What trains | Loss | Data |
|---|---|---|---|
| A1 | Ionic autoencoder (encoder 14→16, decoder=ionic_state_decoder 16→14) | MSE(decoded, true_14_states) | T1 state snapshots |
| A2 | Attention concentration tracking | MSE(conc_attention_output, true_conc_next) | T1 traces |
| A3 | Gate conductance projection (gate_conductance_mlp + linear + logit + decoder) | MSE(decoded, true_5_products) | T1 carried_state vectors |
| B1-B5 | Stage 1 dynamics (attention + ionic_mixing_mlp) | MSE(ionic_state_pred, true_14_states) | T1→T1+T2→T1-T3 |
| C | Concentration dynamics added | MSE(conc, true_conc) | T1-T3 |
| D | Stage 2 (frozen Stage 1) | MSE(I_ion_pred, I_ion_true) | T1-T4 |
| E | End-to-end fine-tune | MSE(I_ion) | T1-T4 |

Rollout curriculum in Phase B: 1→10→100→1000→10000 steps. Scheduled sampling ramps model predictions from 10%→100%. Transition when val loss plateaus.

Initialization: ionic latent = zeros (model discovers own representation), concentrations = real resting values [Na_i≈10, K_i≈138, Ca_i≈0.0001, Ca_ss≈0.0002] (Layer 0 physics). Not from TTP06 encoder — avoids imprinting model assumptions on latent.

Optimizer: AdamW, cosine LR decay per phase, gradient clipping max_norm=1.0. T12 (celltypes) enters at Phase B.

### Multi-Model Conditioning (planned)

One model learns both TTP06 and ORd, conditioned on a model ID label token fed to the attention. TTP06 uses ~18 of 32 ionic dims; ORd uses all 32. Shared structure (attention, compression, readout) is identical -- only latent usage differs. Benefits: richer latent for arc light fine-tuning, single architecture, natural curriculum (TTP06 first, add ORd later).

### First Training Run (TTP06, small dims)

| Hyperparameter | Full (ORd-ready) | First run (TTP06) |
|---|---|---|
| ionic_state | 32 | 16 |
| conductance_latent | 16 | 8 |
| MLP | 32→32 | 16→16 |
| Compression | 32→24→24→16 | 16→12→12→8 |
| Stage 2 queries | 16 | 8 |
| Total inference params | ~4,950 | ~1,200 |

Same architecture, smaller hyperparameters. Validate that the design trains and reproduces TTP06 I_ion before scaling up.

### Approaches Explored Then Scrapped

**Bilinear form with hand-crafted features:** `phi = C @ [1,Vm,Vm^2,Vm^3,E,conc]`, then `I_ion = (conductance * phi).sum()`. Psi factorization moved matmul off critical path (8 FLOPs on path), but the feature vector is arbitrary and the pipeline overly complex.

**Ohmic/non-Ohmic split:** Scrapped because Ohm's law is a Layer 1 model assumption, not Layer 0 ground truth. Rectification, voltage-dependent conductance, surface charge effects all violate Ohm's law.

**KAN Chebyshev readout (v2):** Per-dim Chebyshev on the latent -- wrong place. Compression handles state nonlinearity. Readout's job is driving force, not state transformation.

**Eight ML architectures surveyed:** MLP, two-tower, bilinear, FiLM, hypernetwork, gated two-pathway, cross-attention, Chebyshev. Cross-attention selected for learned routing, physical interpretability, factorizability, and FLOP budget.

---

## 5. Naming Convention

| Name | Dims | What it encodes | Type |
|---|---|---|---|
| **carried_state** | 36 | Full state: ionic + concentrations | Carried between timesteps |
| **ionic_state** | 32 | Channel conformational states (gates, Markov occupancies) | Latent, learned |
| **concentrations** | 4 | [Na_i, K_i, Ca_i, Ca_ss] -- dims 32-35 of carried_state | Explicit, physically named |
| **conductance_latent** | 16 | Effective conductances (gate products) | Latent, compressed from ionic_state |
| **reversal_potentials** | 4 | [E_Na, E_K, E_Ca, E_Ks] | Derived from concentrations via Nernst (fixed physics) |

---

## 6. Normalization and Regularization

### v3 Strategy

| Technique | Where | Purpose | Params |
|-----------|-------|---------|--------|
| 1/sqrt(4) scaling | Attention score | Prevents score saturation (attn_dim=4) | 0 |
| Sigmoid gate | Attention update | Bounds update rate to (0,1), guarantees contraction | 0 |
| Pre-RMSNorm | Before Markov MLP input | Stabilizes MLP input scale across 100K+ recurrent steps | 0 |
| Learned residual mixing (alpha) | Markov MLP residual (ionic dims only) | Per-dim alpha=sigmoid(w), convex combination. No amplification by construction. | 32 |
| Learned compression mixing (beta) | Compression residual | Per-dim beta=sigmoid(w), controls linear vs nonlinear path per compressed dim. | 16 |
| Weight decay | All parameters (AdamW) | Soft regularization, pushes unused coupling toward zero | 0 |
| Gradient clipping | Training | max_norm=1.0, prevents gradient explosions during rollout | 0 |

### Rejected Techniques

- **Spectral norm** (was in v2 and early v3): superseded by learned residual mixing. Convex combination provides a harder guarantee (output bounded between inputs) than spectral norm (bounds amplification ratio but allows growth).
- **RMSNorm on MLP corrections** (was in v2 for split GELU): v3's MLP+GELU has no quadratic blowup risk, and learned mixing alpha already bounds the correction's influence.
- **LayerNorm**: removes per-dim magnitude, which IS information (state distance from equilibrium).
- **BatchNorm**: unstable for autoregressive inference (batch=1 during tissue simulation).
- **Dropout**: injects noise that compounds over 100K+ autoregressive steps. Train/inference mismatch.
- **MLP bottleneck as regularizer**: forces coupling on HH dims that should remain independent. Full-rank 32->32 with learned alpha provides constraint through learning, not geometry.
- **Sigmoid output bounding** (evaluated in v2): vanishing gradients at saturation, breaks residual identity at initialization, triple sigmoid path.

---

## 7. Cross-State Coupling Analysis

### Within One dt (0.01ms)

| Coupling mechanism | Physical timescale | Change within 0.01ms | Conclusion |
|-------------------|-------------------|---------------------|------------|
| Ion concentration changes | Ca_i: 10-100ms, Na_i: minutes | ~0.0001% | Negligible -- temporal, handled by latent memory |
| Ca-induced Ca release (CICR) | RyR activation: 1-2ms | ~0.5-1% | Fast but not instantaneous -- temporal |
| CaMKII phosphorylation | Seconds | ~0.001% | Negligible -- temporal |
| Direct channel-channel interaction | No evidence for conformational coupling | Zero | Non-existent |

**Conclusion**: All cross-state coupling is temporal (many-step accumulation), not instantaneous. No physical basis for a dedicated per-step coupling stage. Coupling emerges naturally through the carried latent's temporal memory.

### Markov Chain Coupling

| Coupling mechanism | Operates through | Timescale | Within one dt | Architecture handles via |
|---|---|---|---|---|
| Intra-protein Markov transitions | Voltage + own conformational state | 0.01-10ms | YES -- primary dynamics | n x 1 attention (voltage) + MLP (cross-dim correction) |
| Inter-channel via Vm | Shared membrane voltage | Instantaneous | YES -- but already captured | Vm as model input |
| Inter-channel via dyadic Ca | Local nanodomain concentration | 10-100us | Comparable to dt | Latent memory + MLP correction |
| Inter-channel via bulk [Ca], [Na], [K] | Cytoplasmic concentration | 10-100ms | Negligible (~0.0001%) | Latent temporal memory |
| CaMKII / phosphorylation | Enzyme kinetics | Seconds | Negligible | Latent temporal memory |
| Direct conformational coupling | None known | N/A | Non-existent | N/A |

**Key finding**: No physical mechanism exists for one channel type's conformational state to directly affect another channel type's transition rates. All inter-channel coupling is mediated through shared variables (Vm, concentrations).

### n x 1 Attention + MLP Is Sufficient for Markov Dynamics

For the Markov update S_new = expm(Q(V)*dt)*S, first-order Taylor expansion gives:

```
S_new ~ S + [D(V) + C(V)]*dt*S = [I + D(V)*dt]*S + C(V)*dt*S
          per-dim voltage (attn)    cross-dim correction (MLP)
```

Splitting error is O(dt^2) ~ 10^-8 at dt=0.01ms. Full self-attention would capture higher-order terms but they are physically negligible.

### Honest Unknowns

- Whether discrete Markov states are the right description (single-channel recordings show fractal gating, memory effects inconsistent with Markov)
- Whether real conformational landscapes are continuous rather than discrete
- Whether macroscopic whole-cell behavior requires Markov-level pathway detail or can be captured by simpler learned dynamics
- Whether undiscovered fast inter-channel coupling mechanisms exist

These unknowns reinforce the design philosophy: provide capacity and inductive bias, but let the data teach the model the actual dynamics.

### Ionic Model Complexity

| Model | State variables | HH gates | Markov states | Concentrations | Other |
|-------|----------------|----------|--------------|----------------|-------|
| TTP06 | 18 | 13 (0-1 bounded) | 0 | 5 (unbounded) | -- |
| ORd | 41 | ~20 | 11 (Na Markov) | 9 | 1 (CaMKII) |

---

## 8. Foundational Design Decisions

### Why Carried Latent (Not Vm History)

The ionic surrogate uses pure autoregressive latent with no Vm history buffer. Three reasons:
1. **Stimulus artifacts**: Current injection creates Vm spikes that confuse history-based models. Carried latent is robust because it only sees Vm (I_stim applied externally in Vm update).
2. **Rush-Larsen doesn't use history either**: The real simulator carries 18 gate states forward. Our latent is the learned analog.
3. **No buffer management**: No non-uniform temporal schedule, no resampling, no variable-dt issues.

### Why n x 1 Cross-Attention (Not Full Transformer)

The n x 1 cross-attention IS a 1D hybrid Transformer: Q=latent dims, K=V=voltage, no self-attention between dims. Cost: ~464 FLOPs vs 200M for a temporal Transformer. Mathematically equivalent to learned Rush-Larsen but with state-dependent gating.

### I_stim Removed

I_stim has been removed from model inputs. The model receives only [Vm, dt]. Three reasons:
1. **Biophysically correct**: real ion channel gates respond to Vm only -- they have no mechanism to sense I_stim.
2. **Matches operator splitting**: in the simulator, the ionic step sees only Vm. I_stim is applied externally in the Vm update equation.
3. **Simpler model**: W_k and W_v are (2, attn_dim) instead of (3, attn_dim).

### Initialization

`latent(0) = zeros` or `V*W_out evaluated at Vm_rest`. Self-corrects in ~1 step: gate ~ 1 when latent is far from target, delta -> (target - latent), latent snaps to equilibrium.

---

## 9. Losses and Training

### Loss Functions

- `L_ion = MSE(I_ion_pred, I_ion_true)` -- weight 1.0, always active
- `L_gate_full = MSE(full_decoder(ionic_state), true_gates)` -- weight lambda_gate, annealed to 0
- `L_gate_comp = MSE(comp_decoder(conductance_latent), true_gates)` -- weight lambda_gate, annealed to 0
- `L_conc = MSE(concentrations, true_concentrations)` -- direct, no decoder
- `L_roll = (1/T) sum_t MSE(Vm_pred(t), Vm_true(t))` -- weight lambda_roll, introduced in rollout phases
- Weight decay 1e-4 (AdamW), gradient clipping max_norm=1.0

### Training Strategy

Bootstrap the latent space first (clean data, autoencoder), then teach dynamics (simple -> complex data, teacher forcing -> rollout), then stress-test (noise, tissue, corruption). The latent must be well-structured before dynamics training begins.

**Phase A -- Latent Space Bootstrap:**
Train a standalone gate autoencoder (encoder: 12 HH gates -> ionic_dim latent, decoder: ionic_dim latent -> 12 HH gates). Data: gate state vectors from Tier 1 (steady-state pacing). Decoder weights transfer to scaffold. Encoder used for initial latent computation in Phase B. Cheap (trains in minutes). Note: 12 gates (not 18) — RR excluded (no gate_inf/tau), concentrations excluded (separate explicit dims with direct MSE loss).

**Phase B -- Simple Dynamics:**
Train dynamics on Tier 1 only (steady-state, single celltype, single dt). Three sub-phases: B1 teacher forcing (single-step) -> B2 short rollout (N=10) -> B3 medium rollout (N=100). Scheduled sampling (gradually replace ground-truth latent with model prediction). Lambda_gate=1.0 initially.

**Phase C -- Full Dynamics:**
Add Tiers 2-4 (S1-S2, dynamic, random intervals). All celltypes. Multiple dt values. Gradual data mixing (90% T1 -> 50/50 -> 10% T1). Unfreeze decoder. Sub-phases: C1 long rollout (N=1000) -> C2 very long rollout (N=10,000) -> C3 full beat (N=100,000). Lambda_gate annealed from 0.3 to 0. Add voltage clamp protocols.

**Phase D -- Robustness:**
Add Tiers 5-12. Scaffold removed (lambda_gate=0). Sub-phases: D1 tissue-mimicking -> D2 stress testing (5-beat rollout, 5000ms). Validation: restitution curve correct, recovery from perturbation within 1 beat.

### Training Timeline

| Phase | Step | Data | Rollout | lambda_gate | lambda_roll | LR | Weight Decay | Batch |
|-------|------|------|---------|-------------|-------------|-------|-------------|-------|
| A | -- | gate states | -- | -- | -- | 1e-3 | 1e-5 | 4096 |
| B | B1 | Tier 1 | 1 | 1.0 | 0 | 1e-3 | 1e-4 | 1024 |
| B | B2 | Tier 1 | 10 | 1.0 | 0.1 | 1e-3 | 1e-4 | 512 |
| B | B3 | Tier 1 | 100 | 0.5 | 0.5 | 5e-4 | 1e-4 | 256 |
| C | C1 | Tier 1-4 + clamp | 1000 | 0.3->0.1 | 1.0 | 5e-4 | 5e-4 | 128 |
| C | C2 | Tier 1-4 + clamp | 10000 | 0.1->0.01 | 1.0 | 2e-4 | 5e-4 | 64 |
| C | C3 | Tier 1-4 + clamp | 100000 | 0.01->0 | 1.0 | 1e-4 | 5e-4 | 32 |
| D | D1 | Tier 1-12 | 100000 | 0 | 1.0 | 5e-5 | 1e-3 | 32 |
| D | D2 | + stress | 500000 | 0 | 1.0 | 2e-5 | 1e-3 | 8 |

Estimated total: ~40 GPU-hours on Blackwell.

### Optimizer

- **AdamW** with variable weight decay (1e-5 -> 1e-3 across phases)
- **LR schedule**: cosine decay within each phase, reset at phase transitions
- **Gradient clipping**: max_norm=1.0
- **Batch size**: decreases as rollout length increases (memory-bound)
- **Early stopping**: per-phase, based on validation I_ion error
- **Checkpointing**: save best model per phase, resume from best if training destabilizes

### Transition Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Phase B->C: complex data destabilizes latent | Gradual data mixing (90/10 -> 50/50 -> 10/90) |
| Scaffold removal: latent drifts without gate loss | Monitor gate reconstruction passively; re-enable at low weight if divergent |
| Rollout length jumps: error accumulation spikes | Double rollout length each time, not 10x jumps |
| Teacher forcing -> autoregressive gap (exposure bias) | Scheduled sampling: mix ground-truth and predicted latent inputs |
| Overfitting at any phase | Increase weight decay 2-3x; add data or reduce model capacity |

### Training Loop (rollout detail)

```
For each step in rollout:
  I_ion_pred = model(latent, Vm, dt)
  if clamp_mask[t]:
    Vm_next = V_clamp[t+1]           # voltage clamp: override Vm
  else:
    Vm_next = Vm + dt*(-I_ion_pred + I_stim)/Cm   # free-running: Vm update
  latent = model.latent_new           # carry latent forward
```

L_roll only applies to free-running segments (clamped Vm is externally set, not predicted).

---

## 10. Training Data

### Protocol Hierarchy (12 Tiers)

Data generated from TTP06 single-cell ODE. All protocols produce (Vm, 18 gates, I_ion) at each timestep. dt fed as 2nd input to model: K = W_k * [Vm, dt].

**Tier 1 -- Steady-state pacing**: BCL in {300, 400, 500, 600, 700, 800, 1000, 1500, 2000} ms. 20 beats each (10 warmup + 10 training). Baseline AP morphology at each rate.

**Tier 2 -- S1-S2 restitution**: S1=1000ms x 10 beats, S2 at DI in {50, 75, 100, 150, 200, 300, 500, 800} ms. Includes sub-ERP DIs (failed captures). Maps restitution curve.

**Tier 3 -- Dynamic protocols**: BCL ramp down (1000->300ms/30 beats), ramp up (300->1000ms), burst (5@300ms + 2000ms pause x5), alternans (20 beats @ BCL=330ms).

**Tier 4 -- Random intervals**: Inter-beat interval ~ LogUniform(200, 2000) ms. 5-200 beats/protocol (variable length) x 200 protocols. Tests arbitrary pacing and generalization.

**Tier 5 -- Tissue-mimicking current injection**: Simulates what a cell experiences in tissue.
- Ornstein-Uhlenbeck noise: dI/dt = -I/tau + sigma*dW, tau in {1, 2, 5, 10, 20} ms, sigma in {5, 10, 20} uA/cm2.
- Smooth depolarizing ramp, sub-threshold blips, sustained current offsets, biphasic pulses, random telegraph.
- **Real tissue profiles**: Extract I_diff(t) = D*nabla^2 Vm from actual Bidomain V1 runs.

**Tier 6 -- Voltage clamp**: Vm externally controlled.
- Step clamp, ramp clamp, staircase clamp, AP clamp, partial clamp.
- Uniquely valuable: errors in I_ion don't propagate back through Vm, giving clean gradients.

**Tier 7 -- Concentration perturbation**: Perturb initial ionic concentrations (K_o, Na_i, Ca_i). ~20 combos x Tier 1 protocols.

**Tier 8 -- Long-duration stability**: 200+ beat pacing, 5-30s quiescence, long blank -> burst. Tests slow drift (Na_i/K_o accumulation).

**Tier 9 -- Recovery from corruption**: Deliberate non-physiological gate states. Teaches self-correction if latent drifts during tissue inference.

**Tier 10 -- Tissue-specific scenarios**: Boundary cells, infarct border zone, inert tissue interface, stimulus site, wavefront tip (spiral).

**Tier 11 -- Combined stressors and stitched protocols**: Multiple simultaneous effects. Stitched traces with variable-length rest. 500+ protocols.

**Tier 12 -- Celltype variants**: TTP06 epi, endo, and M cell configurations across Tier 1-3 protocols.

### Coverage Gaps and Mitigations

| Gap | Risk | Mitigation |
|-----|------|------------|
| Celltype variants (epi/endo/M) | Model fails on specific celltype | Tier 12 |
| Recovery from weird states | Latent drift during tissue inference | Tier 9 |
| Wavefront-specific Vm shapes | Model fails at spiral tips, boundaries | Tier 10 + AP clamp |
| Very long simulations | Slow Na_i/K_o drift | Tier 8 |
| Partial depolarization zones | Cells near block at intermediate Vm | Tier 10 + Tier 5 |
| Simultaneous effects | Individual tiers in isolation | Tier 11 |

### Variable dt

Train with dt in {0.005, 0.01, 0.02, 0.05, 0.1} ms. Each protocol run at all 5 dt values. Multi-scale dt within traces: dt=0.005ms during upstroke, dt=0.1ms during diastole.

### Data Augmentation (on-the-fly during training)

- **Vm noise injection**: epsilon ~ N(0, sigma^2), sigma in {0.1, 0.5, 1.0} mV. Robustness to autoregressive errors.
- **Conductance scaling**: g_X x U(0.5, 2.0) for each major current. Biological variability.
- **Stimulus variation**: amplitude in {-30, -40, -52, -70, -100} uA/cm2, duration in {0.5, 1, 2, 5} ms.
- **Random initial conditions**: gate_k ~ N(gate_k_rest, 0.1*gate_k_rest).

### Data Storage Format

Two-layer storage: raw HDF5 for reproducibility, pre-chunked .pt shards for training speed.

**Storage location**: External HDD `/media/HDD/` (5.5TB ext4).

**Generation layer (source of truth):**
```
/media/HDD/surrogate_data/raw/
  tier01_steady_state.h5       one file per tier
  tier02_s1s2.h5               full metadata per protocol
  ...                          float64 (simulator precision)
  tier12_celltypes.h5
```

**Training layer (optimized for speed):**
```
/media/HDD/surrogate_data/train/
  shard_0000.pt                ~200MB each, pre-shuffled
  shard_0001.pt                float32 (ML precision)
  ...
/media/HDD/surrogate_data/val/
  ...                          split by PROTOCOL, not timestep
```

Each shard: `(N_segments, segment_length, 47)` at float32. Loads directly to GPU.

**Segment format (47 columns):**

| Columns | Content | Role |
|---------|---------|------|
| 0 | Vm | model input |
| 1 | I_stim | model input |
| 2 | dt | model input |
| 3-20 | 18 gate states | gate decoder target |
| 21 | I_ion | primary prediction target |
| 22 | clamp_mask | 0.0 = free-running, 1.0 = voltage clamped |
| 23-34 | 12 gate_inf values | steady-state gate targets (computed post-hoc from Vm) |
| 35-46 | 12 gate_tau values | time constants (computed post-hoc from Vm, ms) |

Column indices match `TraceData` constants in `single_cell_generator.py`: GATE_INF_START=23, GATE_INF_END=35, GATE_TAU_START=35, GATE_TAU_END=47.

### Generation Status

| Tier | Status | Size | Notes |
|------|--------|------|-------|
| T1 (steady-state) | Done | ~3.5GB | |
| T2 (S1-S2) | Done | ~5.1GB | |
| T3 (dynamic) | Done | ~2.3GB | 47-col format |
| T4 (random intervals) | Done | ~551GB | 200 protocols, chunked |
| T5-T10 | Done | ~15GB | |
| T11 (combined/stitched) | Done | ~18GB | 50 stitched protocols |
| T12 (celltypes) | Done | ~22GB | ENDO/M_CELL variants |
| **Total** | **Done** | **~608GB** | All tiers complete on HDD |

### Generation Benchmarks

| Mode | Throughput | Per-beat time | Speedup vs CPU single |
|------|-----------|---------------|----------------------|
| CPU single-cell sequential | 1,637 steps/s | 61 s/beat | 1x |
| CPU batch (n=200) | 269,806 steps/s | 0.36 s/beat/proto | 171x |
| GPU torch.compile (n=10K) | 77.5M steps/s | 0.0013 s/beat/proto | 47,227x |

GPU batch requires large n (>=1000) to overcome kernel launch overhead.

**Chunked processing for long protocols:** Tier 4 (200 protocols x 16.3M steps each) produces 1.2TB monolithically. Solution: chunk_steps=500K. Each chunk ~0.9GB. Eliminates OOM.

---

## 11. Advanced Upgrades

### Connection to State Space Models (Mamba)

Our ionic surrogate IS a selective SSM -- independently derived but structurally identical. The n x 1 cross-attention produces input-dependent state transitions, which is Mamba's core "selectivity" mechanism.

**Parallel scan for training**: Stage 1 alone is an affine recurrence that supports associative scan. For our tiny model, sequential backprop through 100K steps is feasible (~240MB memory per sample with gradient checkpointing). Not a bottleneck on Blackwell GPU.

**Zero-order hold discretization**: Replace Euler update with exact exponential: unconditionally stable for any dt. Cost: +16 FLOPs (one exp per dim). Enables variable dt with guaranteed stability.

### Mixture of Experts (MoE)

4 tiny expert copies of Stage 1, each specialized for one AP phase. Router (single linear on [Vm, dt], ~8 FLOPs) selects top-1 per step. Zero extra inference cost, 4x effective capacity.

### Summary of Advanced Upgrades

| Upgrade | Cost Impact | Training Impact | What It Gives |
|---------|-------------|-----------------|---------------|
| Mamba parallel scan | 0 inference | O(log N) vs O(N) | Train over full traces without sequential rollout |
| Mamba ZOH discretization | +16 FLOPs | -- | Exact exponential update, stable any dt |
| MoE (4 experts) | +8 FLOPs | ~4x data needed | 4x capacity, phase-specialized, zero cost |

---

## 12. Diffusion Component -- Cross-Skip Coupled ResNet

Architecture unchanged from original design (not yet revisited -- focus has been on ionic surrogate):
- Dual conv paths (Vm, phi_e) with bidirectional 1x1 cross-skip connections at each block
- Monodomain single-path baseline first, then bidomain upgrade
- Upgrade path if phi_e accuracy insufficient: dilated conv -> U-Net -> local Transformer -> FNO

---

## 13. Known Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Error accumulation over 100K+ autoregressive steps | HIGH | Gate decoder scaffold per-step; gradient checkpointing; validate with 5000ms+ rollouts |
| Stage 2 readout insufficient for cross-dim products | HIGH | Start with cross-attention, add GELU/MLP if needed. This is where all cross-dim interaction lives now |
| Ca handling is compartmental, not gate-like | MEDIUM | Monitor Ca-related predictions; explicit concentrations help |
| Charge/concentration drift | MEDIUM | Monitor Na/K/Ca; add conservation penalty if needed |
| Encoder-dynamics latent mismatch | MEDIUM | Monitor encoder(gates) vs dynamics latent in Phase B; gate decoder loss ensures functional equivalence |
| Stage 1 too slow even with diffusion hiding | LOW | Stage 1 has entire diffusion step (~ms of GPU time). Only a risk if extremely deep |
| Linear gate decoder insufficient | LOW | Upgrade to MLP decoder if reconstruction error high |

---

## 14. Competitive Landscape (as of 2026-03-18)

No existing bidomain surrogate models. Our approach is unique.

| Approach | Paper | Key difference from ours |
|----------|-------|------------------------|
| **AGATA** (GNN) | Morier et al., FIMH 2025 | Monodomain, Mitchell-Schaeffer (2-var), no phi_e, 12x speedup vs FEM |
| **FNO/KOL** | Centofanti et al., PLOS CB 2025 | Single-shot AT/RT maps, not timestep simulation |
| **PINO** | Lydon et al., arXiv 2025 | PINN-adjacent, monodomain, 10x resolution generalization |
| **LNODE** | Salvador et al., npj Dig Med 2024 | 0D hemodynamic outputs, 300x speedup, different scope |
| **BLNMs** | Martinez et al., CMAME 2025 | Single-shot activation maps, geometry atlas |

Our differentiators: (1) only bidomain surrogate, (2) only biophysically detailed ionic model, (3) only physics-aware architecture, (4) universal ionic latent space, (5) designed for calcium imaging transfer learning.

---

## 15. Open Questions

### Stage 2 Readout (active design)
- What d_model for cross-attention? Token embedding dimension?
- How to embed scalar conductance dims as query tokens?
- How to embed the 9 environment values as key/value tokens?
- Output aggregation: sum of 16 attended values, or concat -> linear?
- Should the 16->1 output projection have a GELU, or trust linear (Kirchhoff's law)?

### Architecture Scaling
- How expressive should Stage 1 be now that it's off the critical path? Multi-head? Stacked layers?
- Does ORd (41 states) require working dim > 32?

### Training
- How does autoregressive error accumulation scale with rollout length? (empirical, no theory)
- Is Phase A autoencoder (designed for 16-dim latent) still the right bootstrap for 32-dim ionic_state?

### Diffusion ResNet
- How many ResNet blocks needed for adequate phi_e receptive field?
- Will cross-skip 1x1 convolutions suffice for Vm<->phi_e coupling?
- Is the monodomain -> bidomain transfer (reusing Vm path weights) effective?

---

## 16. Implementation Status

| Component | Code Location | Tests | Status |
|-----------|--------------|-------|--------|
| Data generation (TTP06) | `Surrogate/surrogate/data/` | 32 tests | Done |
| V3Preprocessor | `Surrogate/surrogate/data/preprocessor.py` | 7 tests | Done |
| NernstComputer | `Surrogate/surrogate/model/nernst.py` | 3 tests | Done |
| IonicStage1 | `Surrogate/surrogate/model/stage1.py` | 9 tests | Done (1,416 inference params) |
| IonicStage2 | `Surrogate/surrogate/model/stage2.py` | 6 tests | Done (118 params) |
| IonicSurrogateV3 | `Surrogate/surrogate/model/ionic_surrogate_v3.py` | 7 tests | Done (1,534 total inference) |
| ORd infrastructure | `Surrogate/surrogate/data/ord_*.py` | 19 tests | Done |
| ORd T1 EPI data | `/media/HDD/surrogate_data/raw_ord/` | -- | Done (9/9 protocols, 12GB) |
| v2 code | `Surrogate/surrogate/model/v2_archive/` | -- | Archived |
| Training pipeline | -- | -- | Not started (next) |
| ARCHITECTURE.md | `Surrogate/ARCHITECTURE.md` | -- | Done (explainer, code-consistent) |

**Test summary**: 51/51 passing (25 model + 7 preprocessor + 19 ORd). v2 tests removed.

---

## 17. Connections

- **Engines**: Bidomain V1 (training data source), Monodomain V5.4 (monodomain baseline)
- **Related research**: [Boundary conduction speedup](../boundary_conduction_speedup/) -- surrogate must reproduce Kleber effect; [Engine consolidation](../engine_consolidation/) -- unified API would simplify data generation
- **Pipelines**: Optimizer V1 (surrogate could replace simulator in optimization loop)
- **Future**: Calcium imaging transfer learning -- fine-tune ionic surrogate on real Ca2+ fluorescence data

---

## Appendix A: v2 Architecture (reference, superseded)

*The v2 design was implemented and tested (18/18 tests passing) but superseded by v3 before training began. Code exists at `Surrogate/surrogate/model/` and diagram at `Images/ionic_surrogate_v2.tex`.*

### v2 Architecture Summary (3 stages, 886 FLOPs, 3.7x Rush-Larsen)

**Stage 1 -- n x 1 Cross-Attention** (~464 FLOPs): 16 latent dims independently query [Vm, dt]. attn_dim=8. Per-dim gate + target. Contractive by construction.

**Stage 2 -- Split GELU Cross-Channel (x2 rounds)** (~352 FLOPs): Two rounds of split GELU gating with spectrally-normalized W_cc (8->16) + RMSNorm. Captures cross-channel coupling. Rank-8 bottleneck.

**Stage 3 -- KAN Chebyshev K=3** (~70 FLOPs): Per-dim Chebyshev polynomial readout. 16 dims x 4 coefficients = 64 params + b_vm + b = 66 total.

**Scaffold -- Gate Decoder** (~324 FLOPs): Single linear (16->18) + sigmoid -> 18 TTP06 gates. Training only.

### v2 Parameter Table

| Component | Params (inference) | Params (training) |
|-----------|-------------------|-------------------|
| Stage 1: W_q (16,8) + W_k (2,8) + W_v (2,8) + W_out (8,16) | 288 | 288 |
| Stage 2a: W_cc1 (8,16) + b_1 (16) | 144 | 144 |
| Stage 2b: W_cc2 (8,16) + b_2 (16) | 144 | 144 |
| Stage 3: C (16,4) + b_vm + b | 66 | 66 |
| Scaffold: W_dec (18,16) + b_dec (18) | -- | 306 |
| **Total** | **642** | **948** |

### v2 Normalization Stack

1/sqrt(8) attention scaling, sigmoid gate, spectral norm on W_cc, RMSNorm on split GELU corrections (inline, zero params), Chebyshev normalization on readout input.

### v2 Simplification Spectrum

| Level | FLOPs | vs RL | Description |
|-------|-------|-------|-------------|
| 0: Scalar HH | 176 | 0.7x | sigma(wV+b) target, const rate |
| 1: + Vm rates | 240 | 1.0x | Sigmoid rate |
| 2: + coupling | 416 | 1.7x | + split GELU cross-channel |
| 3: Full design | 886 | 3.7x | n x 1 cross-attn + 2x split GELU + KAN Chebyshev |

### v2 Modification Menu

**Accuracy upgrades**: A1 multi-head (+200), A2 stacked attention (+464/layer), A3 nonlinear MLP readout (+300), A5 explicit concentrations (+100), A6 structured Nernst current (+50), A7 higher-degree KAN K=5 (+32 params).

**Speed downgrades**: S1 remove one cross-channel round (-176), S2 scalar attention (-350), S3 smaller latent d=8, S4 linear readout (-37 FLOPs), S6 drop to scalar HH (-497).

---

## Appendix B: Failed Approaches (all versions)

| Approach | Why it failed |
|----------|---------------|
| Temporal Transformer (300-pt history) | 200M FLOPs = 1Mx Rush-Larsen. Buffer management, resampling, variable dt issues. |
| Vm history buffer (any size) | Stimulus artifacts contaminate history. Non-uniform schedule is a hyperparameter maze. |
| Fourier decomposition of Vm | Sliding DFT is O(K) and clever, but unnecessary once carried latent eliminates history need. |
| Learned Rush-Larsen | Too constrained -- forces HH exponential relaxation and independent dimensions. Can't represent Markov or Ca dynamics. |
| Neural ODE (dz/dt = MLP(z,Vm)) | Too unconstrained -- multi-timescale learning is notoriously hard without structural priors. |
| GRU cell | Works but gating mechanism adds cost (10x RL) without clear benefit over residual formulation. |
| 17x17 self-attention over latent dims | 47x RL. Cross-channel coupling not worth the cost. |
| Deep MLP for cross-channel | Overkill. Real coupling is rank-3. |
| Dedicated cross-coupling stage (v2 Stage 2) | Cross-state coupling is temporal, not instantaneous within one dt. No physical basis for per-step coupling layer. |
| Full self-attention for Markov | More elegant but splitting error at dt=0.01ms is negligible. Not physically justified over simpler MLP. |
| Softmax for cross-channel gating | Conservation constraint (sum=1) doesn't match independent gate biology. |
| Ohmic/non-Ohmic readout split | Ohmic behavior is a Layer 1 model assumption, not Layer 0 ground truth. |
| Bilinear readout with hand-crafted features | Feature vector [1,Vm,Vm^2,...] is arbitrary. Psi factorization overly complex. Cross-attention learns routing naturally. |
| Concentration decoder from ionic state | Replaced by explicit concentration dims. No decoder that can go wrong. |
| Sigmoid output bounding | Vanishing gradients, breaks residual identity, triple sigmoid path. |
