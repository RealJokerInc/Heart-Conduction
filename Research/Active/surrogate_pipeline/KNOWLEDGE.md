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
carried_state(t), Vm
  |-> Stage 1 (off critical path): dzdt(z, V) -> dz/dt rate -> euler_step -> carried_state(t+1)
  |                                  _compress(carried_state) -> conductance_latent(t+1)
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

### Full Pipeline — PLANNED changes noted inline (code still reflects pre-pivot state)

```
carried_state(t) = [ionic_state(16), concentrations(4)]     (20,)  [full-size: 32+4=36]
  -> n x 1 cross-attention to [Vm], attn_dim=4              per-dim voltage response (contractive)
      NOTE: dt removed for NODE pivot (Section 5b). Field must be dt-independent.
      20 dims all attend to Vm; concentration dims learn slow tracking
  -> SPLIT after attention:
      ionic_mid = z_mid[:16]                                 (16,) -> continues to MLP  [full: :32]
      conc_delta = z_mid[16:20] - z[16:20]                   (4,) -> concentration rate  [full: 32:36]

  ionic_mid:
  -> Pre-RMSNorm                                             stabilize MLP input scale (0 params)
  -> learned-mix MLP + GELU (16->16->16), 1 hidden layer     cross-dim Markov correction  [full: 32->32->32]
      correction = W2 @ GELU(W1 @ RMSNorm(ionic_mid) + b1) + b2
      alpha = sigmoid(w)                                     per-dim mixing weight (16 params)  [full: 32]
      ionic_rate = ionic_delta + alpha * correction          residual_bypass (additive, not convex)

  dz/dt = [ionic_rate, conc_delta]                           (20,)  [full: 36]
  NOTE: dzdt() returns a RATE, not a displacement. ODE solver integrates; Euler does z + dt*dzdt.
  IonicStage1.forward() is removed as a discrete stepper. All callers use dzdt() + euler_step().

  -> learned-mix compression (full carried_state):
      linear_path = W_lin @ carried_state                    (20->8)  [full: 36->16]
      nonlinear_path = MLP(carried_state)                    (20->12->12->8) gate products  [full: 36->24->24->16]
      beta = sigmoid(w_beta)                                 per-dim mixing weight (8 params)  [full: 16]
      conductance_latent = linear_path + beta * nonlinear_path   residual_bypass (additive)
  -> conductance_latent(t+1)                                 (8,)  [full: 16]

  -> Nernst equation (fixed physics, 0 learned params):
      concentrations(t+1) -> [E_Na, E_K, E_Ca, E_Ks]       (4,) reversal potentials

Scaffold decoders (training only, all annealed to zero in Phase D):
  ionic_state(t+1)        -> ionic_state_decoder (16->14) -> 12 HH gates + RR + Ca_SR  [full: 32->14]
  conductance_latent(t+1) -> gate_conductance_decoder (8->5) -> effective gate products  [full: 16->5]
  concentrations(t+1)     -> direct MSE vs true concentrations (no decoder needed)
  Losses:
    L_gate_full = MSE(full_decoder(ionic_state), true_gates)
    L_gate_comp = MSE(comp_decoder(conductance_latent), true_gates)
    L_conc = MSE(concentrations, true_concentrations)     <- direct, no decoder
```

### Design Rationale

**n x 1 cross-attention:**
- Each of 20 carried_state dims [full: 36] independently queries Vm. Produces a per-dim gate (how much to update) and per-dim target (where to move toward).
- NOTE: dt removed from attention input for NODE pivot (Section 5b). The vector field f(z,V) must be dt-independent.
- Mathematically equivalent to learned Rush-Larsen but with state-dependent gating (the attention score depends on BOTH Vm AND the current latent value, while RL's rate depends on Vm only). In the ODE formulation, this gives `dz/dt = gate * (target(V) - z)` — a linear attractor (see Section 5b "Inherent Contraction").
- **attn_dim = 4** (down from 8 in v2): Gate is one scalar from 1 input (Vm). Smooth sigmoid-like function -- 4 basis functions suffice.
- **1 attention layer**: One voltage response is physically correct. Stacking applies contraction twice to same Vm -- numerically redundant.

**Concentrations split off after attention, before MLP:**
- The MLP handles intra-protein Markov corrections between ionic dims only.
- Passing concentrations through the MLP would create artificial ionic<->concentration coupling that doesn't exist within one dt.
- Attention is sufficient for concentration tracking: Vm strongly correlates with gate state during normal pacing, concentration changes are ~0.0001% per step, self-regulation via own-value feedback, end-to-end training through I_ion -> Nernst catches systematic errors.

**Markov MLP (32->32->32, 1 hidden layer, 1 GELU):**
- Cross-dim correction over ionic_state dims only (32 dims). No bottleneck -- architecture ALLOWS coupling but does not FORCE it.
- HH gates that need no coupling pass through residual unchanged.
- Pre-RMSNorm on MLP input stabilizes input scale across 100K+ recurrent steps.
- One hidden layer captures pairwise interactions. Add second layer if insufficient.

**Residual bypass (evolved from DeepSeek hyper-connections):**

```python
alpha = sigmoid(w)                         # w learned per-dim (16,) [full: 32], alpha in (0,1)
correction = MLP(RMSNorm(ionic_mid))
ionic_rate = ionic_delta + alpha * correction   # additive bypass — identity always flows through
```

NOTE: Changed from convex combination `(1-alpha)*base + alpha*correction` to additive bypass `base + alpha*correction` for the NODE pivot. In the ODE formulation, the derivative is `dz/dt = attention_rate + alpha * mlp_correction` — a clean additive sum. Identity path is unconditional; MLP adds corrections on top. This breaks existing checkpoint compatibility (acceptable — A4 failed, no checkpoints worth preserving).

Each ionic dim learns how much MLP correction to accept: alpha->0 for HH dims (pure attention attractor), alpha>0 for Markov-coupled dims. Init: w = large negative -> alpha~0 -> starts as near-pure attractor. 16 learned params [full: 32].

Replaces three mechanisms from earlier design iterations:
- ~~Spectral norm on MLP weights~~ -- not needed, attention contraction provides stability
- ~~Zero-init of MLP output~~ -- not needed, alpha init handles it
- ~~Gate-modulated correction~~ -- not needed, per-dim alpha is the learned gate

**Compression MLP (CARRIED_DIM->COMP_H1->COMP_H2->COND_DIM, 2 hidden layers, 2 GELUs):**
- Takes FULL carried_state (ionic + concentrations) as input, NOT just ionic_state. Gives compression access to concentration context (e.g., Ca_ss for fCass-dependent conductances). Recomputed every step because gate products change and attention cannot compute cross-dim products (structural limitation).
- Two hidden layers: layer 1 computes pairwise products (m*h), layer 2 composes into triple products (m*h*j).
- Residual compression with learned bypass: `compressed = W_lin @ carried + beta * MLP(carried)` [PLANNED: additive bypass, currently convex in code]. Linear path for direct projection, nonlinear path for gate products. Per-dim beta (COND_DIM params) controls nonlinear contribution.
- Total GELUs in Stage 1 pipeline: 3 (1 in Markov MLP + 2 in compression MLP). Sufficient for triple product composition.

**Dual scaffold decoders (training only):**
- ionic_state_decoder (16->14): 12 HH gates + RR + Ca_SR = 14 targets. NOT 18 — concentrations have their own direct MSE loss. [full-size: 32->14]
- gate_conductance_decoder (8->5): 5 effective gate products (G_Na(m³hj), G_CaL(dff2fCass), G_to(rs), G_Kr(Xr1·Xr2), G_Ks(Xs²)). [full-size: 16->5]
- Both use MSE loss, annealed to zero in Phase D. No sigmoid (Ca_SR is unbounded). Linear only — weak decoder forces strong latent.
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

### Training Strategy (v3) — Status: Discrete Autoregressive ABANDONED → Neural ODE Pivot

> The discrete autoregressive approach was abandoned after A4 (native dt, 30K steps) failed to converge.
> **Neural ODE pivot active.** Stage1.dzdt() is the dynamics function, trained via odeint_adjoint (dopri8).
> PLAN.md has implementation steps. See Section 5b for full design, Section 10 for revised training phases.
> NOTE: TRAINING_STRATEGY.md predates the NODE pivot and recommends batch=32768 (a known failure — see IDEALOG.md). Treat PLAN.md as the authoritative current plan for NODE training.

**What was tried and what worked:**

| Approach | Result |
|---|---|
| dt curriculum (subsample T1 at 3ms→1ms→0.1ms→0.01ms, fixed 300ms coverage) | A1-A3 worked. A3 (dt=0.1ms, r=3000) val=0.92. A4 (dt=0.01ms, r=30000) stuck at val~720. |
| Per-dim min-max loss normalization | Essential. Without it, K_i (138 mM) drowns Ca_i (0.0001 mM). |
| Truncated BPTT (window=500 on 30K rollout) | Did not help A4. Window may be too narrow (5ms of gradient for 300ms trajectory). |
| Cosine warm restarts (T_0=50, T_mult=2) | No breakthrough after 2 full cycles at A4. |
| No encoder — model discovers own latent | Correct decision. Encoder imposed wrong latent space. |
| Two-half training (ionic then conductance) | Sound approach. Not fully tested due to A4 failure. |

**Core issue: latent instability over long discrete rollouts.** At native dt (0.01ms), 30K autoregressive steps compound prediction errors faster than the model learns to correct them. Error at step N corrupts steps N+1 through 30K, producing incoherent gradients. This is structural — not fixable by hyperparameter tuning, TBPTT, or LR schedules. A continuous ODE formulation (dz/dt = f(z, Vm)) avoids discrete error accumulation entirely.

Initialization: ionic latent = zeros (model discovers own representation), concentrations = real resting values [Na_i≈10, K_i≈138, Ca_i≈0.0001, Ca_ss≈0.0002] (Layer 0 physics).

Optimizer: AdamW, batch=4096, LR=5e-4 (warmup) or 1e-4 (rollout), gradient clipping max_norm=1.0.

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
| Total inference params | ~4,950 | ~1,534 (verified) |

Same architecture, smaller hyperparameters. Validate that the design trains and reproduces TTP06 I_ion before scaling up.

### Approaches Explored Then Scrapped

**Bilinear form with hand-crafted features:** `phi = C @ [1,Vm,Vm^2,Vm^3,E,conc]`, then `I_ion = (conductance * phi).sum()`. Psi factorization moved matmul off critical path (8 FLOPs on path), but the feature vector is arbitrary and the pipeline overly complex.

**Ohmic/non-Ohmic split:** Scrapped because Ohm's law is a Layer 1 model assumption, not Layer 0 ground truth. Rectification, voltage-dependent conductance, surface charge effects all violate Ohm's law.

**KAN Chebyshev readout (v2):** Per-dim Chebyshev on the latent -- wrong place. Compression handles state nonlinearity. Readout's job is driving force, not state transformation.

**Eight ML architectures surveyed:** MLP, two-tower, bilinear, FiLM, hypernetwork, gated two-pathway, cross-attention, Chebyshev. Cross-attention selected for learned routing, physical interpretability, factorizability, and FLOP budget.

---

## 5. Neural ODE Pivot -- CfC Liquid Networks (from LFLDNet, Salvador & Marsden 2025)

> **SUPERSEDED by Section 5b.** CfC's Δt-as-input approach was evaluated but rejected — it bakes in a structural prior (convex gating form) that prevents learning non-convex dynamics. Section 5b adopts an unconstrained vector field instead. This section retained for the LFLDNet literature summary and the A4 post-mortem.

> Source: [salvador_2025_lfldnet](literature/salvador_2025_lfldnet.md) — DOI: 10.1016/j.compbiomed.2025.111355

### Why the Discrete Autoregressive Approach Failed (A4 Post-Mortem)

At native dt=0.01ms, 30K autoregressive steps accumulate prediction errors faster than the model corrects them. Error at step N corrupts steps N+1 through 30K, producing incoherent gradients. TBPTT (window=500), cosine warm restarts, and dt curriculum did not fix this — it is structural. The discrete-step recurrence must be replaced.

### What LFLDNets Do

LFLDNets replace the ODE integration step with a **Closed-form Continuous (CfC) Neural Circuit Policy (NCP)**: a sparse, neurologically-inspired liquid network that propagates a latent state s(t) using gating functions f, g, h evaluated directly at arbitrary Δt:

```
s(t + Δt) = g(s, μ, Δt) ⊙ h(s, μ, Δt) + (1 - g(s, μ, Δt)) ⊙ f(s, μ, Δt)
```

- `g` is a gating network that learns "how much to update each latent dim given Δt and input."
- No ODE solver. No CFL condition. No step-size controller. Δt is an explicit scalar input.
- Operates at dt=10ms (100× larger than FEM dt=0.1ms) on cardiac EP.
- Bounded dynamics by construction: g ∈ (0,1) by sigmoid, output is convex combination of f and h.

**Speedup on 3D monodomain + TTP06 (HLHS biventricular geometry, 240K DOFs, 7 parameters):**
- LFLDNet inference: 3 min (1 GPU) vs FEM: 1.5 hr (24 CPU cores) → ~30× wall-clock speedup.
- Validation normalized MSE: 3.15e-3.

### Connection to Our Stage 1 Architecture

Our n×1 cross-attention block already resembles CfC gating:

| | Our Stage 1 (pre-pivot) | CfC layer |
|---|---|---|
| Input | [Vm, dt] (dt removed in Section 5b) | [s, μ, Δt] |
| Gating | sigmoid(attention score) per dim | g(s, μ, Δt) per dim |
| Update | delta = attention × (target - state) | convex combination of f and h |
| Stability | bounded by attention contraction | bounded by g ∈ (0,1) |

The key difference: CfC trains Δt as a continuous input. Our Section 5b approach instead removes dt entirely and learns an unconstrained vector field f(z,V), relying on the ODE solver to handle time.

### Actionable Design Change — SUPERSEDED by Section 5b

> The recommendation below to feed Δt as a continuous input was **rejected** in favor of the unconstrained vector field approach (Section 5b). dt is removed from the model entirely. The ODE solver handles time stepping, and the vector field f(z,V) is dt-independent by construction. Kept for historical context on the CfC evaluation path.

### What Does NOT Transfer

- Their **reconstruction network** (maps latent + spatial coordinate → field) is not applicable to per-cell ionic dynamics. We need per-cell trajectories, not a spatial field surrogate.
- Their **global latent** (one s(t) for the entire spatial domain) cannot handle per-cell heterogeneity (infarct, APD gradients). Our per-cell carried_state architecture is correct.
- Their **monodomain-only** surrogate does not address the elliptic solve (94% of bidomain wall time). Cross-Skip ResNet is still needed for the diffusion component.
- The 30× speedup claim mixes GPU vs CPU hardware. GPU-to-GPU comparison would show a smaller ratio.

---

## 5b. Neural ODE Pivot -- Unconstrained Vector Field (2026-04-06 Session)

### The Core Decision: No Structural Prior on the Dynamics

The CfC/liquid NN approach (Section 5) provides bounded updates and Δt-as-input, but it still bakes in a structural form for the dynamics (convex combination of learned gates). Rush-Larsen is even more constrained — it assumes HH exponential relaxation toward a V-dependent equilibrium.

**Decision: learn the unconstrained vector field `dz/dt = f_θ(z, V)`.** No structural prior on what shape f takes. If Rush-Larsen is the right description of the underlying physics, the learned vector field will look like Rush-Larsen when examined. If it doesn't, that's a discovery. This is the Layer 0 principle applied to the dynamics function: don't bake in TTP06/ORd assumptions at the temporal integration level either.

### Why the Discrete Model Was Fundamentally Wrong

The discrete model trained `z_{t+1} = F(z_t, V_t)` — a transition operator. The implicit vector field `f(z,V) = F(z,V) - z` existed in the weights but was:

- **Only reliable on the true trajectory.** The model only ever saw states lying exactly on `z_true(t)`. It never saw a perturbed state, so it has no reason to point perturbed states back toward the trajectory.
- **Full of holes.** Everywhere in phase space except the training trajectory, the vector field is undefined/unreliable.
- **No attractor.** Without a restoring force for off-trajectory states, errors at step N compound through steps N+1 to 30K. The discrete model memorized a path, not a landscape.

### Why the Neural ODE Learns an Attractor

The adjoint method computes gradients against the continuous trajectory. Every time the solver takes an internal step that drifts slightly off the true path, the loss pulls the field back. Over training, the vector field gets shaped not just along the trajectory but in its neighborhood:

```
Discrete model:   f(z, V) reliable only ON trajectory   →  no restoring force, errors compound
Neural ODE model: f(z, V) shaped in basin around it      →  attractor forms, errors decay
```

The rest fixed point, the AP limit cycle, and the basin of attraction are not designed in — they emerge from training the vector field to consistently integrate to the correct trajectory from any starting point near the data.

**Intuition:** the discrete model is a marble rolling along a groove painted on a flat table. The neural ODE model carves a valley. Same path, but the geometry of the valley actively pulls the marble back if it drifts. The valley lives in the weights — not as new parameters, but as the shape the existing parameters were trained to encode.

### Phase Space Geometry

The AP attractor lives in `(V(t), z_1(t), ..., z_32(t)) ∈ ℝ^{33}`, where V is the external driving input and z ∈ ℝ^{32} is the ODE state (+ 4 concentration dims = ℝ^{36} full):

- **Rest fixed point**: z* where `dz/dt = 0` at resting V ≈ -85mV
- **AP limit cycle**: closed curve in ℝ^{36}, traced once per beat
- **Basin of attraction**: the neighborhood from which trajectories converge to the cycle

The attractor manifold is likely much lower-dimensional than 36 — maybe 3-5 effective dimensions (fast V-m subsystem, slow n-h subsystem). The remaining dims are approximately slaved. **Testable prediction:** compute intrinsic dimensionality of learned z(t) trajectories after training. If ~4, matches known HH slow manifold structure.

**Publication-worthy diagnostic:** after training, plot the learned vector field in (V, z_i) projection for the most informative dims. If some dims recover V vs n_K phase portrait structure without being told to, that validates the approach.

### Architectural Changes (Minimal) — PLANNED (PLAN.md Phases 1-2, not yet in code)

> Code still reflects pre-pivot state. These changes are specified in PLAN.md and will be implemented in Phases 1-2.

Zero new learned parameters. Net change: -8 parameters (dt removed from W_k, W_v).

| Change | What | Status |
|--------|------|--------|
| Remove dt from attention input | W_k, W_v: input dim 2 → 1. dt handled by solver, not model. The vector field f(z,V) must be dt-independent. | PLAN Phase 1 |
| `interpolate` → `residual_bypass` | Alpha is internal residual bypass, not external blend. `base + alpha * correction` not `(1-alpha)*base + alpha*correction`. Applies to both ionic mixing and conductance compression. | PLAN Phase 1 |
| `forward()` repurposed | No longer advances state. Runs compression + scaffold only. New `dzdt()` method returns rate. | PLAN Phase 1 |
| `IonicNODE` wrapper | Thin wrapper: holds V(t) linear interpolant, calls `odeint_adjoint`. Euler step for inference. No learned parameters. | PLAN Phase 2 |

**Critical: ODE solver sits after the full Stage 1 pipeline (attention + MLP), not between them.** Both blocks compute f_θ together. Cutting the solver between attention and MLP would deprive the MLP of the full derivative context.

Corrected forward:
```python
def forward(self, t, z):                          # signature for torchdiffeq
    V = self.v_interp(t)                          # linear interp from training data
    z_mid = self.voltage_attention(z, V)          # returns z + gate*(target-z), full state
    delta = z_mid - z                             # attention rate for all dims
    ionic_delta = delta[:, :self.ionic_dim]
    conc_delta  = delta[:, self.ionic_dim:]
    correction = self.mlp(rms_norm(z_mid[:, :self.ionic_dim]))
    alpha = sigmoid(self.mixing_logit)
    ionic_rate = ionic_delta + alpha * correction  # residual bypass on ionic dims
    dz_dt = torch.cat([ionic_rate, conc_delta], dim=-1)
    return dz_dt                                  # → odeint_adjoint integrates this
```

### Solver and Training

| Decision | Choice | Reason |
|----------|--------|--------|
| Solver | `dopri8` + `odeint_adjoint` | GPU-native, 8th order adaptive RK (fewer steps than dopri5 for smooth regions), O(1) memory backward |
| V(t) interpolation | Linear between training grid points | Smooth, no discontinuities, cheap |
| t_eval | 20 AP landmarks (10 in upstroke 0-5ms, 10 in plateau/repol/diastole) | Adjoint shapes full trajectory; dense upstroke sampling matches solver attention |
| Gradients | Adjoint only | Exact gradients against continuous ODE; no BPTT chain |

The latent space may be less stiff than the original ionic ODE — the learned coordinates can implicitly "unroll" stiff directions into smoother latent coordinates. Using `dopri8` (8th order, fewer steps for smooth dynamics). Fall back to implicit solvers only if NFE >1000.

### Inference: ODE Solver NOT Required

The ODE solver is a training tool only. At inference, z is already on or near the attractor. Simple Euler (or RK4 for accuracy) suffices:

```python
# Inference — no torchdiffeq, no adaptive stepping:
dzdt = stage1(z, V)                  # one forward pass, returns dz/dt
z_next = z + dt * dzdt               # Euler; attractor geometry handles drift
```

Inference cost is identical to the original discrete model. The stability gain comes entirely from the landscape learned during training, not from the solver at inference.

**Summary:** training is more expensive (adjoint solve), inference is unchanged. The only thing that changed is what the network was asked to learn during training — a landscape instead of a path.

### Attractor Basin Analysis (2026-04-06 Review)

**The real win of the NODE pivot is gradient tractability, not emergent attractors.** The practical gain:

1. **Gradient chain length:** dopri8 takes ~150-800 internal adaptive steps for a 300ms AP (see "Stiffness Analysis" below), where discrete took 30K. The adjoint backprops through ~200-1000 steps, not 30K. That's a 30-150× reduction in gradient chain length — directly fixes A4's incoherent gradients.
2. **Decoupled resolution:** Loss at ~20 AP landmarks (dense during upstroke) gives the solver freedom between them. Training resolution decoupled from inference resolution.
3. **dt-independent field:** Network learns a rate; inference works at any dt (0.01ms to 1ms).

**Basin width concern:** The adjoint method shapes the field near the trajectory, but only within the ODE solver's local error tolerance (rtol=1e-4). The solver's internal substeps are not adversarial perturbations — they're numerical artifacts the solver minimizes. Without explicit off-trajectory training, the basin of attraction is narrow.

### Inherent Contraction in VoltageAttention

The attention block already provides attractor structure. VoltageAttention computes:

```python
z_new = z + gate * (target(V) - z)    # gate ∈ (0,1) via sigmoid
```

Reinterpreted as a rate for the ODE:

```
dz/dt = gate * (target(V) - z) = -gate * z + gate * target(V)
```

This is a **linear attractor** with decay rate `gate` toward V-dependent equilibrium `target(V)`. This is precisely the Rush-Larsen form `dg/dt = (g_inf(V) - g) / tau(V)` — it fell out of the architecture without being designed in, consistent with the Layer 0 principle (if Rush-Larsen is the right description of reality, the model will discover it).

The MLP correction (`alpha * correction`) perturbs this pure attractor. At init (alpha ≈ 0.007), the pure contractive attention dominates. As alpha grows during training, the MLP adds cross-dimensional coupling corrections on top of the attractor base. The attractor is never lost — it's the foundation the MLP builds on.

**Contraction is preserved at inference** as long as `dt * gate < 1` for all dims. Since gate ∈ (0,1) and typical inference dt is 0.01-0.1ms, this holds comfortably.

### Techniques for Widening the Attractor Basin

**1. Noise injection on z0 (recommended for v1, trivial to implement):**
```python
z0_noisy = z0 + sigma * torch.randn_like(z0)   # sigma schedule: 1e-3 → 1e-2
```
Train the field to reach the correct trajectory from a neighborhood of initial conditions. Same principle as denoising score matching — learn the field in a ball around the data, not just on it. Zero architectural changes, one line in node_rollout.

**2. Mid-trajectory perturbation (follow-up if z0 noise insufficient):**
At random t_eval landmarks, inject noise into z before continuing the ODE solve. Like "scheduled sampling" for continuous dynamics. Forces restoring force everywhere along the AP, not just at t=0. More expensive (multiple ODE solves per segment).

**3. Jacobian spectral radius regularization (insurance):**
```python
v = torch.randn_like(z)
Jv = torch.autograd.functional.jvp(lambda z: f(z, V), z, v)[1]
reg = (Jv.norm() / v.norm()) ** 2   # ≈ ||J||²
loss = trajectory_loss + lambda_reg * reg
```
Explicitly pushes Jacobian eigenvalues toward negative real parts (contraction). Sample at a few points per trajectory, not every t_eval.

**4. Lyapunov co-training (research-grade, not for v1):**
Co-train scalar L(z) with dL/dt < 0 constraint. Formally certifies attractor. Publication material.

**Implementation priority:** z0 noise injection in node_rollout (Phase 3). Mid-trajectory perturbation deferred to first training iteration review. Jacobian regularization if needed after initial results.

### Stiffness Analysis and Training Strategy (2026-04-06)

**Source of stiffness:** VoltageAttention gives each dim its own dynamics: `dz_i/dt ≈ gate_i * (target_i(V) - z_i)`. Jacobian eigenvalues ≈ -gate_i. Gate values range from ~0.001 (slow inactivation) to ~0.999 (m gate during upstroke) → condition number ~1000. The linear scaffold decoder (`nn.Linear(z) ≈ gates`) forces z to be a linear function of gate values, inheriting the full physical stiffness.

**Stiffness is state-dependent, not constant:**

| AP phase | Duration | Fast modes | Eigenvalue spread | dopri8 behavior |
|----------|----------|------------|-------------------|-----------------|
| Upstroke | 0-5ms | m gate active | Large (~1000×) | Small steps, ~100-500 NFE |
| Plateau | 5-100ms | m gate at equilibrium | Small | Large steps, ~20-50 NFE |
| Repolarization | 100-300ms | Moderate | Moderate | ~50-200 NFE |

Estimated total NFE for 300ms AP: 200-1000. This is 30-150× fewer than 30K discrete steps. Adjoint backward through 200-1000 steps is tractable.

**Segmented training was considered and rejected.** Breaking the AP into separate ODE solves per phase would contain stiffness to short segments. But segment B (plateau) needs an initial condition z at t=5ms — there is no ground truth for this. The latent is learned; only resting state z0 = [zeros, resting_conc] is known. Sequential solves with detach at boundaries reduce to TBPTT, which already failed (A4). Without segmentation, the single solve from resting state is the honest approach.

**Dense upstroke landmarks instead:** Focus the loss signal where dynamics are stiff. 10 of 20 t_eval points are in the first 5ms (upstroke). The solver naturally takes small adaptive steps there — matching loss resolution to solver resolution. No segmentation, no mid-AP initial conditions.

**Scaffold-stiffness interaction:** The linear scaffold constrains z ≈ W⁺ × gates, importing gate timescales into z. A nonlinear decoder would free z to use non-stiff coordinates, but violates the "weak decoder → strong latent" design principle. The linear scaffold is correct for structuring the latent — the stiffness it introduces is manageable with dopri8 on a 300ms solve.

**Contingency:** If NFE consistently >1000, add a learned diagonal preconditioner to IonicNODE (20 params, training-only): integrate in w = z × scale coordinates where rates are equalized, convert back to z for loss. Inference bypasses preconditioner entirely.

---

## 6. Naming Convention

| Name | Dims | What it encodes | Type |
|---|---|---|---|
| **carried_state** | 36 | Full state: ionic + concentrations | Carried between timesteps |
| **ionic_state** | 32 | Channel conformational states (gates, Markov occupancies) | Latent, learned |
| **concentrations** | 4 | [Na_i, K_i, Ca_i, Ca_ss] -- dims 32-35 of carried_state | Explicit, physically named |
| **conductance_latent** | 16 | Effective conductances (gate products) | Latent, compressed from ionic_state |
| **reversal_potentials** | 4 | [E_Na, E_K, E_Ca, E_Ks] | Derived from concentrations via Nernst (fixed physics) |

---

## 7. Normalization and Regularization

### v3 Strategy

| Technique | Where | Purpose | Params |
|-----------|-------|---------|--------|
| 1/sqrt(4) scaling | Attention score | Prevents score saturation (attn_dim=4) | 0 |
| Sigmoid gate | Attention update | Bounds update rate to (0,1), guarantees contraction | 0 |
| Pre-RMSNorm | Before Markov MLP input | Stabilizes MLP input scale across ODE solver evaluations (training) and Euler inference steps | 0 |
| Learned residual bypass (alpha) | Markov MLP residual (ionic dims only) | Per-dim alpha=sigmoid(w), additive bypass: `base + alpha*correction`. Identity unconditional. [PLANNED: changed from convex combination for NODE pivot] | 16 [full: 32] |
| Learned compression bypass (beta) | Compression residual | Per-dim beta=sigmoid(w), additive: `linear + beta*nonlinear`. [PLANNED: same change] | 8 [full: 16] |
| Weight decay | All parameters (AdamW) | Soft regularization, pushes unused coupling toward zero | 0 |
| Gradient clipping | Training | max_norm=1.0, prevents gradient explosions during rollout | 0 |

### Rejected Techniques

- **Spectral norm** (was in v2 and early v3): superseded by learned residual bypass. Attention contraction provides stability (Section 5b); additive bypass preserves the attractor base.
- **RMSNorm on MLP corrections** (was in v2 for split GELU): v3's MLP+GELU has no quadratic blowup risk, and learned mixing alpha already bounds the correction's influence.
- **LayerNorm**: removes per-dim magnitude, which IS information (state distance from equilibrium).
- **BatchNorm**: unstable for autoregressive inference (batch=1 during tissue simulation).
- **Dropout**: injects noise that compounds over 100K+ autoregressive steps. Train/inference mismatch.
- **MLP bottleneck as regularizer**: forces coupling on HH dims that should remain independent. Full-rank 32->32 with learned alpha provides constraint through learning, not geometry.
- **Sigmoid output bounding** (evaluated in v2): vanishing gradients at saturation, breaks residual identity at initialization, triple sigmoid path.

---

## 8. Cross-State Coupling Analysis

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

## 9. Foundational Design Decisions

### Why Carried Latent (Not Vm History)

The ionic surrogate uses pure autoregressive latent with no Vm history buffer. Three reasons:
1. **Stimulus artifacts**: Current injection creates Vm spikes that confuse history-based models. Carried latent is robust because it only sees Vm (I_stim applied externally in Vm update).
2. **Rush-Larsen doesn't use history either**: The real simulator carries 18 gate states forward. Our latent is the learned analog.
3. **No buffer management**: No non-uniform temporal schedule, no resampling, no variable-dt issues.

### Why n x 1 Cross-Attention (Not Full Transformer)

The n x 1 cross-attention IS a 1D hybrid Transformer: Q=latent dims, K=V=voltage, no self-attention between dims. Cost: ~464 FLOPs vs 200M for a temporal Transformer. Mathematically equivalent to learned Rush-Larsen but with state-dependent gating.

### I_stim and dt Removed

I_stim was removed from model inputs early in v3 design. dt was subsequently removed for the NODE pivot (Section 5b). The model receives only [Vm]. Three reasons for I_stim removal:
1. **Biophysically correct**: real ion channel gates respond to Vm only -- they have no mechanism to sense I_stim.
2. **Matches operator splitting**: in the simulator, the ionic step sees only Vm. I_stim is applied externally in the Vm update equation.
3. **Simpler model**: W_k and W_v are (1, attn_dim).

dt removal rationale: the vector field `f(z,V)` must be dt-independent for ODE integration. The solver handles time stepping — dt is not a model input. See Section 5b.

### Initialization

`latent(0) = zeros` or `V*W_out evaluated at Vm_rest`. Self-corrects in ~1 step: gate ~ 1 when latent is far from target, delta -> (target - latent), latent snaps to equilibrium.

---

## 10. Losses and Training

> **NOTE:** Sections 10 below describe the **original discrete autoregressive** training strategy, which was abandoned after A4 failure. The NODE pivot uses `odeint_adjoint` with loss at AP landmarks (not per-step). See Section 5b for NODE training design and PLAN.md Phase 3 for implementation. The loss functions and scaffold structure remain the same; what changes is how the trajectory is produced (ODE integration vs discrete loop).

### Loss Functions

- `L_ion = MSE(I_ion_pred, I_ion_true)` -- weight 1.0, always active
- `L_ionic_state = MSE(ionic_state_decoder(ionic_state), true_ionic_states)` -- scaffold, annealed
- `L_conductance = MSE(gate_conductance_decoder(cond_lat), true_conductance_products)` -- scaffold, annealed
- `L_conc = MSE(concentrations, true_concentrations)` -- direct, no decoder
- Weight decay 1e-4 (AdamW), gradient clipping max_norm=1.0

### Training Strategy (SUPERSEDED — discrete approach)

> The phased strategy below was designed for discrete autoregressive rollout. The NODE pivot replaces rollout-length curriculum with ODE integration at AP landmarks. Phase names (A1, B1, etc.) are reused in node_rollout.py but the training loop is fundamentally different. Kept for historical reference.

**Phase A -- Latent Space Bootstrap:**
Train a standalone gate autoencoder (encoder: 12 HH gates -> ionic_dim latent, decoder: ionic_dim latent -> 12 HH gates). Data: gate state vectors from Tier 1 (steady-state pacing). Decoder weights transfer to scaffold. Encoder used for initial latent computation in Phase B. Cheap (trains in minutes). Note: 12 gates (not 18) — RR excluded (no gate_inf/tau), concentrations excluded (separate explicit dims with direct MSE loss).

**Phase B -- Simple Dynamics:**
Train dynamics on Tier 1 only (steady-state, single celltype, single dt). Three sub-phases: B1 teacher forcing (single-step) -> B2 short rollout (N=10) -> B3 medium rollout (N=100). Scheduled sampling (gradually replace ground-truth latent with model prediction). Lambda_gate=1.0 initially.

**Phase C -- Full Dynamics:**
Add Tiers 2-4 (S1-S2, dynamic, random intervals). All celltypes. Multiple dt values. Gradual data mixing (90% T1 -> 50/50 -> 10% T1). Unfreeze decoder. Sub-phases: C1 long rollout (N=1000) -> C2 very long rollout (N=10,000) -> C3 full beat (N=100,000). Lambda_gate annealed from 0.3 to 0. Add voltage clamp protocols.

**Phase D -- Robustness:**
Add Tiers 5-12. Scaffold removed (lambda_gate=0). Sub-phases: D1 tissue-mimicking -> D2 stress testing (5-beat rollout, 5000ms). Validation: restitution curve correct, recovery from perturbation within 1 beat.

### Training Timeline (SUPERSEDED — discrete approach)

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

Estimated total: ~40 GPU-hours on Blackwell (discrete approach — NODE training cost TBD, likely higher per-epoch due to adjoint ODE solve).

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

## 11. Training Data

### Protocol Hierarchy (12 Tiers)

Data generated from TTP06 single-cell ODE. All protocols produce (Vm, 18 gates, I_ion) at each timestep. Model input is Vm only (dt removed for NODE pivot). dt stored in segments for building cumulative time grids during ODE integration.

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
| 0 | Vm | model input (only model input post-NODE pivot) |
| 1 | I_stim | removed from model input (Section 9) |
| 2 | dt | time grid construction (not model input post-NODE pivot) |
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

## 12. Advanced Upgrades

### Connection to State Space Models (Mamba)

Our ionic surrogate IS a selective SSM -- independently derived but structurally identical. The n x 1 cross-attention produces input-dependent state transitions, which is Mamba's core "selectivity" mechanism.

**Parallel scan for training**: Stage 1 alone is an affine recurrence that supports associative scan. For our tiny model, sequential backprop through 100K steps is feasible (~240MB memory per sample with gradient checkpointing). Not a bottleneck on Blackwell GPU.

**Zero-order hold discretization**: Replace Euler update with exact exponential: unconditionally stable for any dt. Cost: +16 FLOPs (one exp per dim). Enables variable dt with guaranteed stability.

### Mixture of Experts (MoE)

4 tiny expert copies of Stage 1, each specialized for one AP phase. Router (single linear on [Vm], ~4 FLOPs) selects top-1 per step. Zero extra inference cost, 4x effective capacity. NOTE: dt removed from router input for NODE pivot — router sees Vm only.

### Summary of Advanced Upgrades

| Upgrade | Cost Impact | Training Impact | What It Gives |
|---------|-------------|-----------------|---------------|
| Mamba parallel scan | 0 inference | O(log N) vs O(N) | Train over full traces without sequential rollout |
| Mamba ZOH discretization | +16 FLOPs | -- | Exact exponential update, stable any dt |
| MoE (4 experts) | +8 FLOPs | ~4x data needed | 4x capacity, phase-specialized, zero cost |

---

## 13. Diffusion Component -- Cross-Skip Coupled ResNet

Architecture unchanged from original design (not yet revisited -- focus has been on ionic surrogate):
- Dual conv paths (Vm, phi_e) with bidirectional 1x1 cross-skip connections at each block
- Monodomain single-path baseline first, then bidomain upgrade
- Upgrade path if phi_e accuracy insufficient: dilated conv -> U-Net -> local Transformer -> FNO

---

## 14. Known Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Error accumulation over 100K+ autoregressive steps | ~~HIGH~~ MITIGATED | NODE pivot eliminates discrete error compounding. Adjoint trains through ~200-1000 solver steps, not 30K (30-150× reduction). Residual risk: Euler inference drift at very long horizons — mitigated by attractor geometry (Section 5b) and z0 noise injection during training. |
| Adjoint ODE solve stiffness | HIGH | If dopri8 takes too many internal steps during upstroke (fast m-gate dynamics), training becomes slow. Mitigation: learned latent coordinates may unfold stiffness. Fallback: implicit_adams or stiff solver. Monitor NFE (number of function evaluations) per segment. |
| Narrow attractor basin from training | MEDIUM | Adjoint shapes field near trajectory but only within solver tolerance. Mitigation: z0 noise injection (primary), mid-trajectory perturbation (secondary), Jacobian regularization (if needed). See Section 5b. |
| Stage 2 readout insufficient for cross-dim products | HIGH | Start with cross-attention, add GELU/MLP if needed. This is where all cross-dim interaction lives now |
| Ca handling is compartmental, not gate-like | MEDIUM | Monitor Ca-related predictions; explicit concentrations help |
| Charge/concentration drift | MEDIUM | Monitor Na/K/Ca; add conservation penalty if needed |
| ~~Encoder-dynamics latent mismatch~~ | ~~MEDIUM~~ REMOVED | Encoder deleted 2026-04-03. Model discovers own latent from zeros. |
| Stage 1 too slow even with diffusion hiding | LOW | Stage 1 has entire diffusion step (~ms of GPU time). Only a risk if extremely deep |
| Linear gate decoder insufficient | LOW | Upgrade to MLP decoder if reconstruction error high |

---

## 15. Competitive Landscape (as of 2026-03-18)

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

## 16. Open Questions

### Stage 2 Readout (RESOLVED — see Section 4)
- ~~What d_model for cross-attention?~~ d=4 (Section 4)
- ~~How to embed scalar conductance dims as query tokens?~~ Per-dim embedding e_q (Section 4)
- ~~How to embed the 9 environment values as key/value tokens?~~ Per-dim e_k, e_v (Section 4)
- ~~Output aggregation?~~ attended → MLP(16→4→1) with GELU (Section 4)
- ~~GELU or linear?~~ GELU — conductance dims are learned, not literal channels (Section 4)

### Architecture Scaling
- How expressive should Stage 1 be now that it's off the critical path? Multi-head? Stacked layers?
- Does ORd (41 states) require working dim > 32?

### Training
- ~~How does autoregressive error accumulation scale with rollout length?~~ ANSWERED: compounds structurally at 30K steps. NODE pivot eliminates this. (Section 5b)
- How many dopri8 internal steps (NFE) will the learned latent require per AP? (determines training cost)
- Is z0 noise injection sufficient for attractor basin, or do we need mid-trajectory perturbation?

### Diffusion ResNet
- How many ResNet blocks needed for adequate phi_e receptive field?
- Will cross-skip 1x1 convolutions suffice for Vm<->phi_e coupling?
- Is the monodomain -> bidomain transfer (reusing Vm path weights) effective?

---

## 17. Implementation Status

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
| IonicNODE wrapper | `Surrogate/surrogate/model/node.py` | -- | Not started (PLAN Phase 2) |
| NODE training loop | `Surrogate/surrogate/training/node_rollout.py` | -- | Not started (PLAN Phase 3) |
| Discrete training archive | `Surrogate/surrogate/training/archive/` | -- | Not started (PLAN Phase 0) |
| Training pipeline (discrete) | `Surrogate/surrogate/training/` | 44 tests | Done (superseded by NODE) |
| ARCHITECTURE.md | `Surrogate/ARCHITECTURE.md` | -- | Done (explainer, code-consistent) |

**Test summary**: 51/51 passing (25 model + 7 preprocessor + 19 ORd). v2 tests removed.

---

## 18. Connections

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
| Neural ODE with plain MLP (dz/dt = MLP(z,Vm)) | Too unconstrained without structural priors (Session 7). **Reconsidered**: using existing attention+MLP as f_θ provides inherent contraction via attention gating. Now the active approach — see Section 5b. The key difference: plain MLP has no attractor structure; attention+MLP has built-in contraction (`gate*(target-z)`). |
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
