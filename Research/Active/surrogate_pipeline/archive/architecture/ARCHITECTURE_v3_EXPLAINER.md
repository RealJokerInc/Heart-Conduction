# IonicSurrogateV3 Architecture Explainer

Logic-based explainer organized by flow, then decisions, then per-block detail.
All variable names, shapes, and defaults match the implemented code (small TTP06 config).

---

## Part 1: Architecture Flow

The model replaces the TTP06 ionic model's per-cell computation. Each timestep, it takes a 20-dimensional carried state (16 ionic + 4 concentration dims) and membrane voltage, and produces an updated state plus a scalar ionic current I_ion.

**VoltageAttention** receives the full `carried_state` (B, 20) along with `Vm` and `dt`. Each of the 20 dims independently queries the voltage to decide how much to update and where to move toward. The output `z_mid` is the same shape (B, 20) -- every dim has been nudged by voltage.

**Concentration Split** cleaves `z_mid` into `ionic_mid` (B, 16) and `conc_new` (B, 4). Concentrations are done -- they only needed the voltage-gated attention update. Ionic dims continue through the MLP.

**IonicMixingMLP** applies cross-dimensional correction to `ionic_mid` only. RMSNorm stabilizes the input, the MLP computes a correction, and `interpolate` blends the correction with the residual via a learned per-dim alpha. The result is `ionic_new` (B, 16).

**Recombine** concatenates `ionic_new` and `conc_new` back into `carried_state_new` (B, 20). This is the state carried to the next timestep.

**GateConductanceCompression** reads the full `carried_state_new` (B, 20) and compresses it into `conductance_latent` (B, 8) through parallel linear and MLP paths, blended by a learned per-dim beta. This latent encodes effective gate products (like m^3*h*j) needed for current computation.

**NernstComputer** takes the 4 concentrations from the PREVIOUS timestep and computes reversal potentials [E_Na, E_K, E_Ca, E_Ks] using fixed physics (zero learned params). Then `normalize_environment` packs 9 tokens [Vm, E_Na, E_K, E_Ca, E_Ks, Na_i, K_i, Ca_i, Ca_ss] and normalizes them to roughly [-2, 2].

**Stage 2 ConductanceAttention** uses the PREVIOUS step's `conductance_latent` as queries and `env_normalized` as keys/values. No softmax -- scores can be negative because driving forces (Vm - E) can be negative. Produces `attended` (B, 8).

**OutputMLP** maps `attended` (B, 8) to scalar `I_ion` (B,).

The critical insight: Stage 1 advances state to t+1 (off critical path, can run during diffusion), while Stage 2 computes I_ion from the PREVIOUS step's conductance and concentrations (on critical path, must finish before diffusion begins).

```
carried_state(t), Vm, dt
  |
  +---> Stage 1 (off critical path):
  |       VoltageAttention -> Split -> IonicMixingMLP -> Recombine
  |         -> GateConductanceCompression -> conductance_latent(t+1)
  |         -> Nernst(conc(t+1)) -> reversal_potentials(t+1)
  |         -> carried_state(t+1)
  |
  +---> Stage 2 (ON critical path):
          ConductanceAttention(cond_lat(t), env_norm(t)) -> OutputMLP -> I_ion(t)
```

---

## Part 2: Design Decisions

### VoltageAttention
- **Alternatives**: GRU cell (works but gating overhead unnecessary), standard self-attention over dims (47x Rush-Larsen, overkill), pure MLP (no state-dependent gating).
- **Why this**: n x 1 cross-attention IS learned Rush-Larsen with state-dependent gating. Each dim queries voltage independently, producing a gate and target. The gate depends on both Vm and the current latent value (RL's rate depends on Vm only). Contractive by construction: update is `old + gate * (target - old)` with gate in (0,1).
- **Layer 0 reality**: Channel proteins change conformation in response to membrane voltage. Each channel type responds independently within one dt.
- **Does NOT**: apply cross-dim coupling (that is the MLP's job), use softmax (gate is per-dim sigmoid, not competitive), or stack multiple layers (one voltage response per dt is physically correct).

### Concentration Split
- **Alternatives**: Pass concentrations through the MLP alongside ionic dims.
- **Why split**: The MLP handles intra-protein Markov corrections between ionic dims. Concentrations change ~0.0001% per step -- routing them through the MLP creates artificial coupling that does not exist within one dt. Attention alone is sufficient for their slow Vm-dependent tracking.
- **Does NOT**: apply any MLP correction to concentrations. They get attention only.

### RMSNorm
- **Alternatives**: LayerNorm (adds centering + learned scale, unnecessary), BatchNorm (unstable for autoregressive inference with batch=1).
- **Why this**: Stabilizes MLP input scale across 100K+ recurrent steps. Zero parameters. Instance-level (no batch statistics).
- **Does NOT**: learn any parameters or center the input. Pure scale normalization.

### IonicMixingMLP + interpolate
- **Alternatives**: Direct residual add (no blending control), spectral norm + zero-init (heavier machinery for same goal), full self-attention (overkill for cross-dim coupling).
- **Why this**: DeepSeek-inspired learned residual mixing. Convex combination `(1-alpha)*residual + alpha*correction` cannot amplify by construction. Each ionic dim learns how much MLP correction to accept. HH-like dims learn alpha near 0 (pure residual); Markov-coupled dims learn alpha > 0.
- **Layer 0 reality**: Cross-channel coupling (e.g., Ca cycling affecting gate kinetics) exists but is slow -- accumulates over many steps. One MLP layer captures pairwise interactions per step.
- **Does NOT**: force coupling on dims that do not need it (residual bypass). Does not touch concentration dims.

### GateConductanceCompression
- **Alternatives**: Carry conductance_latent forward from previous step (attention cannot compute cross-dim products like m^3*h*j, so conductance must be recomputed). Use ionic_state only as input (misses Ca_ss needed for fCass-dependent conductances).
- **Why this**: Two GELU layers compose triple products (m*h in layer 1, m*h*j in layer 2). Linear bypass provides direct projection path. Beta mixing controls linear vs nonlinear per compressed dim.
- **Does NOT**: use the previous conductance_latent as input. Recomputes from scratch each step.

### NernstComputer
- **Alternatives**: Learn reversal potentials (unnecessary -- Nernst is exact physics with known constants).
- **Why fixed**: The Nernst equation is Layer 0 physics. RT/F, extracellular concentrations, and the log relationship are not model assumptions -- they are thermodynamics. Zero learned parameters means zero chance of learning something wrong.
- **Does NOT**: contribute any learned parameters. Gradients still flow through it (log and division are differentiable).

### Environment Normalization
- **Alternatives**: Learned normalization (adds parameters for something that is known a priori), no normalization (Ca_i ~ 0.0001 mM vs K_i ~ 138 mM makes calcium invisible to attention).
- **Why fixed shifts/scales**: Physiological ranges are known. 18 fixed constants. Normalizes all 9 tokens to approximately [-2, 2].
- **Does NOT**: learn the normalization. Ranges come from TTP06 physiology.

### Stage 2 CrossAttention (no softmax)
- **Alternatives**: MLP on concatenated inputs (no routing structure), bilinear form (arbitrary feature engineering), softmax attention (forces positive weights, physically wrong for driving forces).
- **Why this**: Each conductance dim "asks" the environment what driving force to apply. Na conductance attends to Vm and E_Na, learning (Vm - E_Na). CaL attends to Vm and Ca_ss, learning GHK-like flux. Negative scores are physically meaningful -- the sign of (Vm - E) determines current direction.
- **Layer 0 reality**: Current = conductance x driving force. The driving force depends on which ion and which voltage/concentration tokens matter for that channel.
- **Does NOT**: use softmax (I_ion is unbounded, negative attention is meaningful), or share parameters across query dims.

### OutputMLP
- **Alternatives**: Pure linear sum (Kirchhoff's law says currents sum linearly, but our 8 conductance dims are learned abstractions, not literal channels).
- **Why MLP**: Allows nonlinear interaction between the 8 attended values. Hidden dim of 4 is conservative.

---

## Part 3: Detailed Logic Blocks

### 1. VoltageAttention

**Purpose**: Per-dim voltage-gated update of the carried state.

**Input -> Output**: `carried_state` (B, 20), `Vm` (B,), `dt` (B,) -> `z_mid` (B, 20)

**How it works**:
1. Stack `[Vm, dt]` into `x` (B, 2).
2. Compute key `k = W_k(x)` (B, 4) and value `v = W_v(x)` (B, 4).
3. Per-dim query: `q = einsum('ij,jk->ijk', carried_state, W_q)` (B, 20, 4). Each of 20 dims gets its own 4-dim query by multiplying its scalar value with its row of `W_q`.
4. Score: `score = einsum('ijk,ik->ij', q, k) * scale` (B, 20). Dot product of each dim's query with the shared key.
5. Gate: `gate = sigmoid(score)` (B, 20). Bounds update rate to (0,1).
6. Target: `target = v @ W_out` (B, 20). Shared target projected to all dims.
7. Update: `carried_state + gate * (target - carried_state)` (B, 20). Contractive: moves toward target, never overshoots.

**Key parameters**: `W_q` (20, 4), `W_k` Linear(2, 4), `W_v` Linear(2, 4), `W_out` (4, 20). `scale` = 1/sqrt(4) = 0.5.

**Scaffold supervision**: None directly. Validated indirectly through scaffold decoders on the downstream ionic state and conductance latent.

**Failure mode**: If attention learns gate near 0 everywhere, state never updates. If gate near 1 everywhere, state snaps to a shared target (loses per-dim information). Both are mitigated by per-dim query depending on current state value.

### 2. Concentration Split

**Purpose**: Separate ionic dims (which need MLP coupling) from concentration dims (which do not).

**Input -> Output**: `z_mid` (B, 20) -> `ionic_mid` (B, 16), `conc_new` (B, 4)

**How it works**: `ionic_mid = z_mid[:, :16]`, `conc_new = z_mid[:, 16:]`. Pure slicing.

**Key parameters**: None. Split point determined by `ionic_dim` (16).

**Scaffold supervision**: `conc_new` gets direct MSE loss against true concentrations.

**Failure mode**: If the split point is wrong (concentrations encoded in wrong dims), both branches get corrupted. Mitigated by initialization and scaffold losses.

### 3. RMSNorm

**Purpose**: Normalize MLP input scale to prevent drift over 100K+ recurrent steps.

**Input -> Output**: `ionic_mid` (B, 16) -> normalized (B, 16)

**How it works**: `x / (x.pow(2).mean(-1, keepdim=True).sqrt() + 1e-8)`. Divides each sample by its root-mean-square. No learned scale or shift.

**Key parameters**: None. Epsilon = 1e-8.

**Scaffold supervision**: None (transparent to gradients).

**Failure mode**: If input is already well-scaled, RMSNorm is a no-op (divides by ~1). Cannot hurt. If input drifts to extreme scale, RMSNorm rescues the MLP input.

### 4. IonicMixingMLP + interpolate

**Purpose**: Cross-dimensional correction on ionic dims (Markov coupling, pairwise gate interactions).

**Input -> Output**: `ionic_mid` (B, 16) -> `ionic_new` (B, 16)

**How it works**:
1. `correction = ionic_mixing_mlp(rms_norm(ionic_mid))` (B, 16). Two-layer MLP: Linear(16, 16) -> GELU -> Linear(16, 16).
2. `ionic_new = interpolate(ionic_mid, correction, ionic_mixing_logit)` (B, 16). Computes `alpha = sigmoid(ionic_mixing_logit)` per dim, then `(1-alpha)*ionic_mid + alpha*correction`.

**Key parameters**: `ionic_mixing_mlp` weights (16x16 + 16x16 + biases), `ionic_mixing_logit` (16,) initialized to -5.0 (sigmoid(-5) ~ 0.007, near-pure residual at start).

**Scaffold supervision**: Downstream `ionic_state_decoder` on `ionic_new` ensures the MLP preserves gate information.

**Failure mode**: If alpha stays near 0, MLP has no effect (pure residual -- safe but no coupling). If alpha goes to 1 for all dims, residual path is lost. Sigmoid bounds prevent divergence.

### 5. GateConductanceCompression

**Purpose**: Compress full carried state into effective conductance latent (gate products like m^3*h*j).

**Input -> Output**: `carried_state_new` (B, 20) -> `conductance_latent` (B, 8)

**How it works**:
1. `linear_path = gate_conductance_linear(carried_state_new)` (B, 8). Linear(20, 8, bias=False).
2. `nonlinear_path = gate_conductance_mlp(carried_state_new)` (B, 8). Three-layer MLP: Linear(20, 12) -> GELU -> Linear(12, 12) -> GELU -> Linear(12, 8).
3. `conductance_latent = interpolate(linear_path, nonlinear_path, gate_conductance_logit)` (B, 8). Per-dim beta mixing between linear projection and nonlinear gate-product composition.

**Key parameters**: `gate_conductance_linear` (20, 8), `gate_conductance_mlp` layers (20->12->12->8), `gate_conductance_logit` (8,) initialized to -5.0.

**Scaffold supervision**: `gate_conductance_decoder` maps `conductance_latent` (8) -> (5) predicting effective gate products [G_Na(m^3hj), G_CaL(dff2fCass), G_to(rs), G_Kr(Xr1Xr2), G_Ks(Xs^2)].

**Failure mode**: If compression loses gate-product information, Stage 2 cannot reconstruct the correct current magnitude. The scaffold decoder catches this during training.

### 6. NernstComputer

**Purpose**: Compute reversal potentials from ion concentrations using fixed thermodynamic equations.

**Input -> Output**: `Na_i`, `K_i`, `Ca_i` each (B,) -> `E_Na`, `E_K`, `E_Ca`, `E_Ks` each (B,)

**How it works**:
1. `E_Na = RTONF * log(140 / Na_i)` -- monovalent sodium.
2. `E_K = RTONF * log(5.4 / K_i)` -- monovalent potassium.
3. `E_Ca = 0.5 * RTONF * log(2.0 / Ca_i)` -- divalent calcium (factor 0.5 = 1/z for z=2).
4. `E_Ks = RTONF * log((5.4 + 0.03*140) / (K_i + 0.03*Na_i))` -- mixed Na/K permeability.
5. All concentrations clamped at 1e-12 to prevent log(0).

**Key parameters**: None learned. `RTONF` = RT/F ~ 26.713 mV. Extracellular: Na_o=140, K_o=5.4, Ca_o=2.0 mM. PRNaK=0.03.

**Scaffold supervision**: None needed (exact physics).

**Failure mode**: Cannot fail (fixed computation). If concentrations drift to unphysical values, the clamp prevents NaN but the reversal potentials will be wrong -- upstream concentration tracking must be accurate.

### 7. Environment Normalization

**Purpose**: Scale 9 environment tokens to comparable magnitudes for attention.

**Input -> Output**: `Vm`, `E_Na`, `E_K`, `E_Ca`, `E_Ks`, `Na_i`, `K_i`, `Ca_i`, `Ca_ss` each (B,) -> `env_normalized` (B, 9)

**How it works**:
1. Stack all 9 scalars into `env` (B, 9).
2. `(env - norm_shift) / norm_scale` using fixed shift/scale buffers.
3. Shifts and scales derived from TTP06 physiological ranges (e.g., Vm: shift=-25, scale=65; Ca_i: shift=0.001, scale=0.001).

**Key parameters**: `norm_shift` (9,) and `norm_scale` (9,) registered as buffers (not learned).

**Scaffold supervision**: None.

**Failure mode**: If physiological ranges are wrong, some tokens will be much larger than others, dominating attention. Ranges are conservatively set from known TTP06 bounds.

### 8. Stage 2 CrossAttention

**Purpose**: Each conductance dim queries the environment to determine its current contribution.

**Input -> Output**: `conductance_latent` (B, 8), `env_normalized` (B, 9) -> `attended` (B, 8)

**How it works**:
1. `Q = einsum('ij,jk->ijk', conductance_latent, e_q)` (B, 8, 4). Each of 8 conductance dims scaled by its row of `e_q`.
2. `K = einsum('il,lk->ilk', env_normalized, e_k)` (B, 9, 4). Each of 9 env tokens scaled by its row of `e_k`.
3. `V = einsum('il,lm->ilm', env_normalized, e_v)` (B, 9, 1). Each env token scaled by its row of `e_v`.
4. `scores = einsum('ijk,ilk->ijl', Q, K) * scale` (B, 8, 9). No softmax.
5. `attended = einsum('ijl,ilm->ijm', scores, V).squeeze(-1)` (B, 8).

**Key parameters**: `e_q` (8, 4), `e_k` (9, 4), `e_v` (9, 1). `scale` = 1/sqrt(4) = 0.5. All initialized with small random values (*0.1).

**Scaffold supervision**: None directly on attention. Validated through I_ion output.

**Failure mode**: If scores collapse to uniform, all conductance dims see the same environment (loses channel specificity). If scores saturate, gradients vanish. Small init (*0.1) and no softmax mitigate both.

### 9. Output MLP

**Purpose**: Combine 8 attended conductance contributions into scalar I_ion.

**Input -> Output**: `attended` (B, 8) -> `I_ion` (B,)

**How it works**:
1. `h = GELU(W1 @ attended + b1)` (B, 4). Linear(8, 4) + GELU.
2. `I_ion = W2 @ h + b2` (B, 1) -> squeeze -> (B,). Linear(4, 1).
3. Biases zero-initialized so zero conductance -> zero output.

**Key parameters**: `output_mlp[0]` Linear(8, 4), `output_mlp[2]` Linear(4, 1). Biases zero-init.

**Scaffold supervision**: I_ion is the primary training target (MSE against true ionic current).

**Failure mode**: If hidden dim 4 is too small, the MLP cannot capture nonlinear channel interactions. Upgrade to 8 if needed.

### 10. Scaffold: ionic_state_decoder

**Purpose**: Ensure the ionic latent encodes recoverable gate information during training.

**Input -> Output**: `ionic_new` (B, 16) -> `ionic_state_pred` (B, 15)

**How it works**: Single `nn.Linear(16, 15)`. No activation -- targets include both [0,1]-bounded gates and unbounded concentrations (Ca_SR, RR). Targets: 13 HH gates + Ca_SR + RR.

**Key parameters**: Linear(16, 15). Removed by `remove_scaffold()` for inference.

**Failure mode**: If the decoder cannot reconstruct gates, the ionic latent is not encoding useful gate information. Training adjusts Stage 1 to fix this.

### 11. Scaffold: gate_conductance_decoder

**Purpose**: Ensure compression preserves effective gate product information.

**Input -> Output**: `conductance_latent` (B, 8) -> `conductance_pred` (B, 5)

**How it works**: Single `nn.Linear(8, 5)`. No activation -- products are unbounded. Targets: G_Na(m^3hj), G_CaL(dff2fCass), G_to(rs), G_Kr(Xr1Xr2), G_Ks(Xs^2).

**Key parameters**: Linear(8, 5). Removed by `remove_scaffold()` for inference.

**Failure mode**: If the decoder cannot reconstruct gate products, the conductance latent has lost information needed for current computation. Training adjusts compression to fix this.

---

## Part 4: Naming Convention

| Code variable | Shape | Meaning |
|---|---|---|
| `carried_state` | (B, 20) | Full state carried between timesteps: 16 ionic + 4 concentration |
| `ionic_dim` | 16 | Number of latent ionic state dimensions |
| `conc_dim` | 4 | Number of explicit concentration dimensions [Na_i, K_i, Ca_i, Ca_ss] |
| `carried_dim` | 20 | `ionic_dim + conc_dim` |
| `attn_dim` | 4 | Attention projection dimension (Stage 1 VoltageAttention) |
| `cond_dim` | 8 | Conductance latent dimension after compression |
| `mlp_hidden` | 16 | IonicMixingMLP hidden layer width |
| `comp_h1` | 12 | Compression MLP first hidden layer |
| `comp_h2` | 12 | Compression MLP second hidden layer |
| `z_mid` | (B, 20) | Attention output before split |
| `ionic_mid` | (B, 16) | Ionic dims after attention, before MLP |
| `conc_new` | (B, 4) | Updated concentrations (attention only) |
| `ionic_new` | (B, 16) | Ionic dims after MLP + interpolation |
| `carried_state_new` | (B, 20) | Recombined `[ionic_new, conc_new]` |
| `ionic_mixing_mlp` | -- | Two-layer MLP for cross-dim ionic correction |
| `ionic_mixing_logit` | (16,) | Per-dim alpha logit for MLP interpolation (init -5.0) |
| `gate_conductance_linear` | -- | Linear(20, 8) bypass for compression |
| `gate_conductance_mlp` | -- | Three-layer MLP (20->12->12->8) for nonlinear compression |
| `gate_conductance_logit` | (8,) | Per-dim beta logit for compression interpolation (init -5.0) |
| `conductance_latent` | (B, 8) | Compressed effective conductances |
| `cond_lat_prev` | (B, 8) | PREVIOUS step's conductance latent (Stage 2 input) |
| `conc_prev` | (B, 4) | PREVIOUS step's concentrations (Nernst input) |
| `E_Na, E_K, E_Ca, E_Ks` | (B,) | Nernst reversal potentials |
| `env_normalized` | (B, 9) | Normalized environment tokens for Stage 2 |
| `norm_shift` | (9,) | Fixed normalization shifts (buffer) |
| `norm_scale` | (9,) | Fixed normalization scales (buffer) |
| `e_q` | (8, 4) | Stage 2 query embeddings |
| `e_k` | (9, 4) | Stage 2 key embeddings |
| `e_v` | (9, 1) | Stage 2 value embeddings |
| `scores` | (B, 8, 9) | Stage 2 attention scores (no softmax) |
| `attended` | (B, 8) | Stage 2 attention output per conductance dim |
| `output_mlp` | -- | Stage 2 final MLP: Linear(8,4) -> GELU -> Linear(4,1) |
| `I_ion` | (B,) | Output: total ionic current |
| `ionic_state_decoder` | -- | Scaffold: Linear(16, 15), training only |
| `gate_conductance_decoder` | -- | Scaffold: Linear(8, 5), training only |
| `n_ionic_targets` | 15 | Scaffold targets: 13 HH gates + Ca_SR + RR |
| `n_conductance_targets` | 5 | Scaffold targets: 5 effective gate products |
| `ALPHA_INIT` | -5.0 | Initial logit for ionic mixing (sigmoid(-5) ~ 0.007) |
| `BETA_INIT` | -5.0 | Initial logit for compression mixing |
| `n_env` | 9 | Number of environment tokens [Vm, E_Na, E_K, E_Ca, E_Ks, Na_i, K_i, Ca_i, Ca_ss] |
| `d_v` | 1 | Stage 2 value dimension |
| `stage2_attn` | 4 | Stage 2 attention dimension |
| `stage2_mlp_h` | 4 | Stage 2 output MLP hidden dimension |
