# Ionic Surrogate v3: Design Rationale

> Paper-resolution documentation of every architectural and training decision.
> Each section maps to a potential Methods subsection. Written to support
> a full publication without requiring conversation history reconstruction.

---

## 1. Problem Statement

The bidomain cardiac simulation requires solving an ionic ODE at every mesh node at every timestep. The ionic step (TTP06: 18 state variables, ~600 FLOPs; ORd: 40 state variables, ~1700 FLOPs) accounts for ~6% of total wall time, while the elliptic solve for extracellular potential dominates at ~94%. A neural surrogate replacing both components enables real-time simulation.

The ionic surrogate must:
- Predict total ionic current I_ion from membrane voltage and internal state
- Maintain an autoregressive state across 100,000+ timesteps without drift
- Generalize across pacing protocols, cell types, and eventually ionic models
- Run faster than the Rush-Larsen analytical ODE solver it replaces

The key architectural insight: the ionic step and diffusion step can run in parallel via operator splitting. I_ion(t) depends on state(t), while state(t+1) is computed simultaneously. Only the current readout is on the critical path; state evolution runs in the background during the diffusion solve.

## 2. Reasoning Framework: Layer 0 Analysis

Architectural decisions are derived from physical reality (Layer 0), not from biophysics model assumptions (Layer 1). This distinction is critical because:

- TTP06 and ORd disagree on internal structure (HH gates vs Markov chains, 3 vs 4 Ca compartments, presence/absence of CaMKII)
- The surrogate should capture what both models agree on (physical reality) without being locked to either model's specific parameterization
- The future goal is training from experimental optical mapping data, which follows physics, not any particular model

Layer 0 physical truths encoded in the architecture:
1. Channel proteins change conformation in response to voltage (→ per-dim attention to Vm)
2. Different processes operate at different timescales, 0.1ms to 100ms+ (→ per-dim sigmoid gate)
3. Ionic concentrations change slowly, creating electrochemical memory (→ explicit concentration dims with Nernst physics)
4. Current = conductance × driving force, summed over channels (→ cross-attention readout)
5. Cross-channel coupling within one timestep is negligible (→ no dedicated coupling stage; temporal coupling via carried state memory)

Layer 1 assumptions explicitly NOT encoded:
- HH gating formalism (m³hj products) — the latent can represent state differently
- Ohm's law for driving force — some channels are non-Ohmic (GHK, pumps)
- Specific gate independence — the MLP can learn coupling the model doesn't prescribe
- Named gate variables — the 16-dim latent is not forced to correspond to specific gates

## 3. Two-Stage Parallel Architecture

### 3.1 Critical Path Analysis

In the full bidomain surrogate pipeline:
```
I_ion(t) → Vm_update → diffusion_solve → Vm(t+1)
```

The diffusion solve dominates wall time (94%). The ionic surrogate's I_ion must complete before diffusion begins (critical path), but state(t+1) is only needed at the NEXT timestep's readout. Therefore:

- Stage 1 (state evolution): runs on a separate CUDA stream during the diffusion solve. Effectively free compute. Can be arbitrarily expressive.
- Stage 2 (current readout): on the critical path. Must be fast.

This decomposition was derived from the operator splitting structure of the parent simulator, not from ML architecture conventions.

### 3.2 Why Not One Stage?

A single network mapping (state, Vm) → (state_new, I_ion) would entangle the critical-path readout with the background state update. Any increase in state evolution complexity (larger latent, deeper MLP) would slow the readout proportionally. The two-stage design decouples these concerns.

## 4. Stage 1: State Evolution

### 4.1 VoltageAttention

Each of the 20 carried_state dimensions independently queries [Vm, dt] to produce a per-dim update gate and target. The update is contractive: z_new = z + gate × (target - z), with gate ∈ (0,1).

Why n×1 cross-attention (not full self-attention): Cross-state coupling within one dt=0.01ms is physically negligible. The Markov matrix exponential decomposes as expm(Q·dt) ≈ I + [D(V) + C(V)]·dt, where D is diagonal (per-dim voltage response) and C is off-diagonal (cross-dim coupling). The splitting error is O(dt²) ≈ 10⁻⁸. Full self-attention would capture higher-order terms but at 20× cost for negligible physical benefit.

Why state-dependent gating (vs Rush-Larsen): The attention gate depends on both Vm AND the current latent value (via the per-dim query q_d = z_d · W_q[d,:]). Rush-Larsen's rate depends on Vm only. State-dependent gating allows the model to allocate more update to dimensions far from equilibrium — a form of adaptive computation.

Why attn_dim=4: The gate is a scalar computed from a 2-input function (Vm, dt). Voltage-dependent steady-state curves are smooth sigmoids — 4 basis dimensions in the query-key dot product suffice. Increasing to 8 doubles Stage 1 attention parameters without theoretical benefit.

### 4.2 Concentration Split

Concentrations (Na_i, K_i, Ca_i, Ca_ss) are explicit, physically named state dimensions. They split off AFTER attention, BEFORE the ionic mixing MLP. Rationale:

- Concentrations change ~0.0001% per timestep. The MLP's cross-dimensional coupling is designed for intra-protein conformational dynamics (gate-to-gate interactions), not for the slow accumulation of ionic concentrations.
- Routing concentrations through the MLP creates artificial coupling between gates and concentrations that does not exist within one dt.
- Attention alone provides sufficient Vm-dependent tracking for concentrations: at each Vm, there is an approximate equilibrium concentration (where ion influx = efflux). The attention target encodes this. The slow gate (small sigmoid output) controls the rate of approach.
- Concentration tracking is validated by direct MSE against ground truth — no decoder needed.

Why 4 explicit dims (not 5): Ca_SR (sarcoplasmic reticulum calcium) was excluded because it does not appear in any reversal potential computation or readout feature. It is a purely internal variable tracked implicitly in the 16-dim ionic latent.

### 4.3 Ionic Mixing MLP

A two-layer MLP (Linear → GELU → Linear) operating on the 16 ionic dims only, with learned per-dim interpolation between the MLP correction and the residual (attention output).

Why cross-dimensional MLP (not per-dim only): While the attention update is per-dim, real ionic dynamics include intra-protein coupling. In Markov channel models (e.g., ORd Na channel), transition rates between conformational states create off-diagonal elements in the state evolution matrix. The MLP captures these to first order: expm(Q·dt)·S ≈ [I + D·dt]·S + C·dt·S, where C·dt·S is the cross-dim correction the MLP learns.

Why no bottleneck: A bottleneck (16→8→16) would force all dims through shared hidden neurons, imposing coupling on dimensions that may be independent (e.g., Na gates and K gates). The full-rank MLP (16→16→16) allows coupling where the data says it exists while the per-dim interpolation logit naturally suppresses coupling where it is not needed (logit stays near its initial value of -5.0, giving sigmoid ≈ 0.007).

Why learned interpolation (not spectral norm or zero-init): The interpolation function `(1-sigmoid(logit))·residual + sigmoid(logit)·correction` provides a per-dim learned gate that:
- Cannot amplify (output bounded between residual and correction)
- Starts as near-pure residual (logit initialized to -5.0)
- Allows each dim to independently control how much MLP correction to accept
- Replaces three separate mechanisms (spectral norm, zero-init, gate-modulated correction) with one

### 4.4 Gate Conductance Projection

The full carried_state (20 dims: 16 ionic + 4 concentration) is projected to an 8-dim conductance latent via parallel linear and nonlinear paths with learned per-dim interpolation.

Why full carried_state input (not ionic only): Some effective conductances depend on concentrations (e.g., fCass depends on Ca_ss). Providing the full state gives the projection access to this context.

Why recompute every step (not carry forward): The projection must compute effective gate products (m³·h·j, d·f·f2·fCass, etc.) — nonlinear cross-dimensional operations. The attention mechanism is per-dim and structurally cannot compute these products. Carrying the conductance latent forward and updating it incrementally via attention would accumulate errors in the gate products. Recomputation from the current state guarantees consistency.

Why two GELU hidden layers: Gate products require compositional nonlinearity. Layer 1 computes pairwise products (m·h). Layer 2 computes products of products (m·h·j). One hidden layer can approximate these via universal approximation but requires excessive width; two layers match the compositional structure naturally.

### 4.5 Nernst Computation

Reversal potentials are computed from concentrations using the exact Nernst equation with zero learned parameters. RTONF = RT/F computed from physical constants (R=8314.472 J/mol·K, T=310K, F=96485.3415 C/mol).

Why fixed physics (not learned): The Nernst equation is thermodynamics — not a model assumption. Learning it would add parameters that can only converge to the known answer or learn something wrong. The computation is differentiable (log and division), so gradients flow through it to train concentration tracking.

## 5. Stage 2: Current Readout

### 5.1 Cross-Attention Without Softmax

Each conductance latent dimension queries the 9-element normalized environment vector [Vm, E_Na, E_K, E_Ca, E_Ks, Na_i, K_i, Ca_i, Ca_ss] to determine its current contribution.

Why cross-attention (not MLP on concatenated inputs): The physical equation I_ion = Σ G_ch × driving_force_ch(Vm, E) has multiplicative structure — each channel's conductance multiplies a driving force that depends on specific environment tokens. Cross-attention learns this routing: the Na conductance dim attends strongly to Vm and E_Na, learning something approximating (Vm - E_Na). An MLP would need to discover this multiplicative structure from data.

Why no softmax: The total ionic current I_ion is unbounded (~-200 to +50 µA/cm²). Softmax forces attention weights to be positive and sum to 1, which would prevent the model from learning negative contributions (the E term in Vm - E). Without softmax, scores can be negative, which is physically meaningful — the sign of the driving force determines current direction.

### 5.2 Environment Normalization

All 9 environment tokens are normalized to approximately [-1, 1] using fixed physiological ranges (18 constants: 9 shifts + 9 scales). This is required because concentration magnitudes span 6 orders of magnitude (K_i ~ 138 mM vs Ca_i ~ 0.0001 mM). Without normalization, low-magnitude tokens are invisible to the attention mechanism.

### 5.3 Output MLP

A two-layer MLP (Linear(8,4) → GELU → Linear(4,1)) combines the 8 attended values into scalar I_ion. Biases are zero-initialized so zero conductance produces zero current.

Why MLP (not linear sum): Kirchhoff's current law says channel currents sum linearly. But the 8 conductance latent dimensions are learned abstractions, not literal channel currents. Their combination may benefit from nonlinearity. The hidden dimension of 4 is conservative — expandable if needed.

## 6. Scaffold Decoders

### 6.1 Design Principle

Scaffold decoders are intentionally weak (single linear layer, no activation) so they cannot compensate for a poor latent. If the decoder cannot reconstruct the target, the latent must improve. A strong decoder (MLP) could reconstruct from fragments, hiding latent quality issues.

### 6.2 Ionic State Decoder

Linear(16, 14) mapping ionic latent to 14 targets: 12 HH gates (m through Xs) + RR + Ca_SR. No activation — targets include both [0,1]-bounded gates and unbounded concentrations. Removed for production inference.

### 6.3 Gate Conductance Decoder

Linear(8, 5) mapping conductance latent to 5 effective gate products: G_Na(m³hj), G_CaL(dff2fCass), G_to(rs), G_Kr(Xr1·Xr2), G_Ks(Xs²). No activation — products are unbounded. Validates that the gate conductance projection correctly computes the cross-dimensional products.

## 7. Training Strategy

### 7.1 Single Loss Per Phase

Each training phase optimizes exactly one loss function. No multi-objective balancing, no weighting hyperparameters. The complexity is in the curriculum (which data, which components frozen), not in the loss function.

This was a deliberate design choice to avoid the well-known difficulty of multi-task loss balancing, where relative weighting of losses can dominate training dynamics and requires extensive tuning.

### 7.2 Phase Progression

The progression A1→A2→A3→B→C→D→E follows a bootstrap logic:
- A1 establishes the latent space (what the 16 dims mean)
- A2 teaches concentration tracking (slow dynamics)
- A3 trains the gate conductance projection (given a well-structured latent)
- B teaches dynamics (how the latent evolves, with increasing rollout)
- C adds concentration dynamics to the dynamics model
- D trains Stage 2 in isolation (simple regression on frozen features)
- E fine-tunes end-to-end

Each phase depends on the previous: A3 needs a meaningful latent from A1; B needs pre-trained scaffold decoders from A1/A3; D needs stable Stage 1 features from B/C.

### 7.3 Initialization Philosophy

Ionic latent initialized to zeros — the model discovers its own internal representation. This avoids imprinting TTP06's gate structure (a Layer 1 assumption) onto a latent that should be free to represent ionic state in whatever way best serves I_ion prediction.

Concentrations initialized to real resting values (Na_i≈10, K_i≈138, Ca_i≈0.0001, Ca_ss≈0.0002 mM). These are measurable physical quantities (Layer 0), not model parameters.

The attention's contractive update self-corrects the ionic latent within a few steps: when the latent is far from target, gate ≈ 1, and the update snaps toward the voltage-dependent target.

### 7.4 Rollout Curriculum

Teacher forcing (rollout=1) trains single-step accuracy. Progressive rollout extension (10→100→1000→10000) teaches the model to handle its own prediction errors. Scheduled sampling (mixing teacher-forced and autoregressive steps) bridges the gap.

The rollout length determines what temporal phenomena the model must capture:
- 1 step (0.01ms): instantaneous dynamics
- 100 steps (1ms): sodium upstroke
- 1000 steps (10ms): early repolarization
- 10000 steps (100ms): full plateau
- 100000 steps (1000ms): complete action potential

## 8. Scaling Design

The architecture is parameterized by hyperparameters (ionic_dim, cond_dim, attn_dim, etc.) that scale from TTP06 (small: 16+4=20 carried state) to ORd (full: 32+4=36 carried state) without code changes. The same model code handles both — only the numbers change.

Multi-model conditioning is planned via a label token, enabling one trained model to serve both ionic models and potentially generalize to real experimental data.
