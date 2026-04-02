# Ionic Surrogate v3

## Overview

We present a two-stage autoregressive neural surrogate for cardiac ionic dynamics that replaces the Rush-Larsen ODE solver while learning the full nonlinear state-current mapping from data. The architecture decomposes ionic computation into two parallel stages -- state evolution (Stage 1) and current readout (Stage 2) -- with a key structural insight: only Stage 2 lies on the simulation critical path. Stage 1 runs off critical path on a separate CUDA stream during the diffusion solve, making its compute effectively free. The model uses 1,454 parameters at inference (1,766 during training) with 970 critical-path FLOPs (Stage 2) and 3,292 background FLOPs (Stage 1) per forward step.

## 1. Design Philosophy: The Layer 0 Framework

Architectural decisions derive from a three-layer reasoning hierarchy, prioritized top-down:

**Layer 0 -- Physical reality.** What actually happens at the cardiac membrane. Channels undergo voltage-dependent conformational changes. Current obeys Ohm's law modulated by open probability. Concentrations create electrochemical gradients. Multi-timescale dynamics span sub-millisecond activation to minutes-long concentration drift. These are ground truths -- the architecture must capture them.

**Layer 1 -- Biophysics models (TTP06, ORd).** Human-constructed theories explaining Layer 0. Hodgkin-Huxley gating, independent gates, compartmental calcium cycling, specific Markov state diagrams -- all useful, validated, but contain assumptions. The $m^3hj$ sodium gating product is a modeling choice, not a measured physical law.

**Layer 2 -- Neural architecture.** Inspired by what Layer 1 got right about Layer 0, but not enslaved to Layer 1's specific formulations. Where a neural network can learn the relationship more naturally than the biophysics equation prescribes, let it.

The resulting maxim: *physics provides the skeleton, ML fills the flesh.* Fixed physics (Nernst equation, operator splitting, Kirchhoff summation) is hardcoded with zero learned parameters. Learned components (attention dynamics, MLP coupling, compression, readout routing) handle relationships that differ between ionic models or that reality may implement differently than any single biophysics model assumes.

## 2. Problem Formulation

The TTP06 ventricular ionic model evolves 18 state variables according to:


$$
\frac{dg_i}{dt} = \frac{g_{\infty,i}(V_m) - g_i}{\tau_i(V_m)}, \quad I_{ion} = \sum_j \bar{g}_j \prod_i g_i^{a_{ij}} (V_m - E_j)
$$


The surrogate replaces this with a 20-dimensional carried state $\mathbf{z} = [\mathbf{s}_{ion}, \mathbf{c}] \in \mathbb{R}^{20}$ (16 latent ionic + 4 explicit concentrations $[\text{Na}_i, \text{K}_i, \text{Ca}i, \text{Ca}{ss}]$) updated autoregressively. TTP06's fifth concentration $\text{Ca}_{SR}$ is excluded from the explicit set — it is a purely internal SR variable not needed by any reversal potential or readout feature, and is tracked implicitly within the 16-dim latent ionic state:


$$
\mathbf{z}^{n+1}, \mathbf{g}^{n+1} = f_{\text{Stage1}}(\mathbf{z}^n, V_m^n, \Delta t) \quad \text{(off critical path)}
$$


$$
I_{ion}^n = f_{\text{Stage2}}(\mathbf{g}^n, \text{env}^n) \quad \text{(on critical path)}
$$


where $\mathbf{g}^n$ is the conductance latent (8 dims, compressed from ionic state) and $\text{env}^n$ is the 9-token normalized environment vector $[V_m, E_{Na}, E_K, E_{Ca}, E_{Ks}, \text{Na}_i, \text{K}_i, \text{Ca}i, \text{Ca}{ss}]$. Crucially, Stage 2 reads the previous step's conductance latent and concentrations, not Stage 1's current-step output. This mirrors the operator splitting in the bidomain simulator where $I_{ion}$ at time $t$ depends on the state at time $t$, while the state advances to $t+1$ in parallel:

```
carried_state(t), Vm, dt
  |-> Stage 1 (off critical path):  carried_state(t+1), conductance_latent(t+1)
  |                                  [runs during diffusion step on separate CUDA stream]
  |
  +-> Stage 2 (ON critical path):   I_ion(t) from conductance_latent(t) + env(t)
  |                                  Nernst(conc(t)) -> reversal_potentials(t)
                                     [must complete before diffusion step begins]
```

**Implementation note:** In the sequential reference code, Nernst runs inside the orchestrator's `forward()` on `conc_prev` (time $t$), producing reversal potentials for Stage 2's current-step computation. In the CUDA-stream deployment, Nernst on Stage 1's output `conc(t+1)` would be precomputed during the diffusion step for the *next* timestep's readout — mathematically equivalent, but amortized over the diffusion compute.

$I_{stim}$ is excluded from model inputs. Ion channels respond to membrane voltage, not to external stimulus current directly. The stimulus affects $V_m$, which the model receives at the next time step. This preserves the operator-splitting structure.

## 3. Stage 1: State Evolution (Off Critical Path)

Stage 1 is an expressive state evolution engine whose compute cost is amortized by running on a separate CUDA stream during the diffusion solve. It comprises five sub-stages: cross-attention, concentration split, Markov MLP, compression, and Nernst computation.

### 3a. Cross-Attention (20 dims attend to $[V_m, \Delta t]$, attn_dim=4)

**Motivation.** In the Rush-Larsen scheme, each gate's update rate $\alpha_i = 1 - \exp(-\Delta t / \tau_i(V_m))$ depends only on voltage. A gate far from equilibrium updates at the same rate as one near equilibrium -- physically unrealistic during the rapid sodium upstroke where activation must respond on sub-millisecond timescales while plateau-phase gates should change minimally. We introduce state-dependent gating via per-dimension cross-attention.

**Formulation.** The voltage and time step are projected to shared key and value vectors:


$$
\mathbf{x} = [V_m, \Delta t] \in \mathbb{R}^2, \quad \mathbf{k} = W_k \mathbf{x} \in \mathbb{R}^4, \quad \mathbf{v} = W_v \mathbf{x} \in \mathbb{R}^4
$$


Each of the 20 carried dimensions independently queries the voltage representation:


$$
\mathbf{q}_d = z_d \cdot W_q[d, :] \in \mathbb{R}^4, \quad \text{score}_d = \frac{\mathbf{q}_d \cdot \mathbf{k}}{\sqrt{4}}, \quad \alpha_d = \sigma(\text{score}_d)
$$



$$
\text{target}_d = \mathbf{v} \cdot W_{out}[:, d], \quad z_d^{mid} = z_d + \alpha_d \cdot (\text{target}_d - z_d)
$$


**Implementation.** $W_q \in \mathbb{R}^{20 \times 4}$ (80 params), $W_k \in \mathbb{R}^{2 \times 4}$ (8), $W_v \in \mathbb{R}^{2 \times 4}$ (8), $W_{out} \in \mathbb{R}^{4 \times 20}$ (80). All bias-free, Xavier-initialized. Total: 176 parameters, 572 FLOPs.

**Physical interpretation.** The sigmoid gate $\alpha_d \in (0,1)$ is the learned analog of the Rush-Larsen rate $1 - \exp(-\Delta t/\tau)$, but now depends on both voltage *and* the current state value. A dimension far from its target produces a large query, receives a high attention score, and updates aggressively. A dimension near equilibrium produces a small query and updates gently. This state-dependent gating is impossible in Rush-Larsen.

**Why attn_dim=4 (down from 8 in v2).** The gate is a single scalar computed from two inputs ($V_m$, $\Delta t$). The voltage-dependent steady-state curves $g_\infty(V_m)$ are smooth sigmoidal functions -- 4 basis directions in the query-key dot product suffice to represent them. Increasing to 8 would double Stage 1's attention parameter count (176 → 352) without theoretical justification — a 2-input smooth function does not require 8 basis dimensions.

**Contraction guarantee.** For any $\alpha_d \in (0,1)$: $|z_d^{mid} - \text{target}_d| = (1-\alpha_d)|z_d - \text{target}_d| < |z_d - \text{target}_d|$. The update is strictly contractive toward the voltage-dependent target, preventing unbounded latent drift over arbitrarily long rollouts.

### 3b. Concentration Split

After attention, the 20-dimensional intermediate $\mathbf{z}^{mid}$ is split:


$$
\mathbf{s}^{mid}_{ion} = \mathbf{z}^{mid}_{[1:16]}, \quad \mathbf{c}^{n+1} = \mathbf{z}^{mid}_{[17:20]}
$$


Concentrations are *done* -- they receive no further processing. This is a deliberate design choice grounded in Layer 0 analysis.

**Why concentrations skip the MLP.** The Markov MLP (Stage 1c) handles intra-protein conformational corrections between ionic state dimensions. Passing concentrations through it would create artificial ionic-concentration coupling that does not exist within one $\Delta t = 0.01$ ms. Concentration changes are $\sim$0.0001% per step -- the cross-attention's contractive update toward a voltage-dependent target is sufficient for their slow tracking dynamics. End-to-end training through $I_{ion} \rightarrow$ Nernst catches systematic concentration drift.

**Concentration MSE loss (training only).** Concentrations have a direct MSE loss against ground truth with no decoder: $L_{conc} = \text{MSE}(\mathbf{c}^{n+1}, \mathbf{c}^{n+1}_{true})$. This is possible because concentrations are explicit, physically-named variables -- unlike the latent ionic state which requires a scaffold decoder for supervision.

### 3c. Markov MLP (16 $\rightarrow$ 16, GELU, Pre-RMSNorm, Learned $\alpha$ Mixing)

**Motivation.** The cross-attention treats each ionic dimension independently, mirroring Hodgkin-Huxley gate independence. In reality, Markov models of individual channels have transition rates that couple multiple conformational states -- the Na channel's open state depends on both activation and inactivation pathways simultaneously. The MLP introduces controlled cross-dimensional coupling.

**Formulation.**


$$
\hat{\mathbf{s}} = \text{RMSNorm}(\mathbf{s}^{mid}_{ion}), \quad \text{correction} = W_2 \cdot \text{GELU}(W_1 \hat{\mathbf{s}} + \mathbf{b}_1) + \mathbf{b}_2
$$



$$
\alpha = \sigma(\mathbf{w}_\alpha), \quad \mathbf{s}^{n+1}_{ion} = (1 - \alpha) \odot \mathbf{s}^{mid}_{ion} + \alpha \odot \text{correction}
$$


where $\mathbf{w}_\alpha \in \mathbb{R}^{16}$ is a learned per-dimension mixing parameter initialized to $-5.0$ ($\sigma(-5) \approx 0.007$, near-pure residual at initialization).

**Pre-RMSNorm.** Inline, zero parameters: $\hat{s}_d = s_d / (\sqrt{\frac{1}{D}\sum_d s_d^2} + \epsilon)$, where $\epsilon = 10^{-8}$ is added *outside* the square root (matching the implementation: `x / (x.pow(2).mean(-1, keepdim=True).sqrt() + 1e-8)`). Stabilizes MLP input scale across 100,000+ recurrent steps. Unlike LayerNorm, RMSNorm does not center (remove mean), preserving per-dimension magnitude information that the state-dependent query mechanism exploits.

**Learned $\alpha$ mixing.** The convex combination $(1-\alpha) \cdot \text{residual} + \alpha \cdot \text{correction}$ provides a harder stability guarantee than spectral normalization (used in v2): the output is bounded between the two inputs regardless of correction magnitude. Each ionic dimension learns independently how much MLP correction to accept -- $\alpha \rightarrow 0$ for HH-like dimensions that need no coupling (pure residual), $\alpha > 0$ for Markov-coupled dimensions. This replaces three mechanisms from v2: spectral norm on weights, zero-initialization of output layers, and gate-modulated corrections.

**Implementation.** $W_1 \in \mathbb{R}^{16 \times 16}$ + $\mathbf{b}_1$ (272 params), $W_2 \in \mathbb{R}^{16 \times 16}$ + $\mathbf{b}*2$ (272), $\mathbf{w}_\alpha$ (16). Total: 560 parameters, 1,330 FLOPs.

**Physical interpretation.** For the Markov matrix exponential $\mathbf{S}^{n+1} = \exp(Q(V)\Delta t) \mathbf{S}^n$, the first-order Taylor expansion gives:


$$
\mathbf{S}^{n+1} \approx \mathbf{S}^n + [D(V) + C(V)] \Delta t \cdot \mathbf{S}^n
$$


where $D(V)$ captures per-state diagonal (voltage-dependent) rates and $C(V)$ captures cross-state coupling. The attention handles $D(V) \cdot \Delta t \cdot \mathbf{S}^n$ (per-dimension voltage response); the MLP handles $C(V) \cdot \Delta t \cdot \mathbf{S}^n$ (cross-dimensional correction). The splitting error is $O(\Delta t^2) \sim 10^{-8}$ at $\Delta t = 0.01$ ms.

### 3d. Compression (16 $\rightarrow$ 12 $\rightarrow$ 12 $\rightarrow$ 8, Dual-Path with Learned $\beta$ Mixing)

**Motivation.** The ionic current $I_{ion} = \sum_j \bar{g}_j \prod_i g_i^{a_{ij}} (V_m - E_j)$ depends on the state through *effective conductances* -- products of gating variables like $m^3hj$, $df \cdot f_2 \cdot f_{Ca}$. The 16-dimensional ionic state contains more information than is needed for current computation. Compression learns the mapping from raw state to the effective quantities that Stage 2 needs.

**Formulation.** A linear bypass path preserves direct information:


$$
\text{linear} = W_{lin} \cdot \mathbf{s}^{n+1}_{ion} \in \mathbb{R}^8
$$


A nonlinear path computes gate products via two GELU layers:


$$
\mathbf{h}_1 = \text{GELU}(W_{c1} \cdot \mathbf{s}^{n+1}_{ion} + \mathbf{b}_{c1}), \quad \mathbf{h}_2 = \text{GELU}(W_{c2} \cdot \mathbf{h}_1 + \mathbf{b}_{c2})
$$


$$
\text{nonlinear} = W_{c3} \cdot \mathbf{h}_2 + \mathbf{b}_{c3} \in \mathbb{R}^8
$$


The two paths combine via learned per-dimension mixing:


$$
\beta = \sigma(\mathbf{w}_\beta), \quad \mathbf{g} = (1 - \beta) \odot \text{linear} + \beta \odot \text{nonlinear}
$$


**Why two hidden layers.** Layer 1 computes pairwise products (e.g., $m \cdot h$). Layer 2 composes into triple products (e.g., $m \cdot h \cdot j$ or $d \cdot f \cdot f_2$). This matches the maximum polynomial degree in TTP06 current equations. The GELU activation enables sign-dependent gating -- dimensions near zero are suppressed, allowing selective product computation.

**Why dual-path mixing.** Some information passes through linearly (concentrations embedded in ionic state, single-gate conductances). Other information requires nonlinear transformation (multi-gate products). The learned $\beta$ per dimension controls the blend, initialized near zero ($\beta$-init $= -5.0$) so the model starts with the simpler linear path and discovers nonlinearity as needed.

**Implementation.** $W_{lin}$ (128 params), $W_{c1}$ + $\mathbf{b}$ (204), $W_{c2}$ + $\mathbf{b}$ (156), $W_{c3}$ + $\mathbf{b}$ (104), $\mathbf{w}_\beta$ (8). Total: 600 parameters, 1,360 FLOPs.

**Compression ratio.** TTP06 has $\sim$11 effective current inputs; ORd has $\sim$13. The 8-dimensional conductance latent achieves a 2:1 compression from 16 ionic dims, matching the empirical effective dimensionality.

### 3e. Nernst Computation (Fixed Physics, 0 Learned Parameters)

The Nernst module computes reversal potentials from the updated concentrations using textbook electrochemistry:


$$
E_{Na} = \frac{RT}{F} \ln\frac{[\text{Na}^+]_o}{[\text{Na}^+]_i}, \quad E_K = \frac{RT}{F} \ln\frac{[\text{K}^+]_o}{[\text{K}^+]_i}
$$



$$
E_{Ca} = \frac{RT}{2F} \ln\frac{[\text{Ca}^{2+}]_o}{[\text{Ca}^{2+}]*i}, \quad E_{Ks} = \frac{RT}{F} \ln\frac{[\text{K}^+]*o + P_{NaK}[\text{Na}^+]_o}{[\text{K}^+]*i + P_{NaK}[\text{Na}^+]_i}
$$


Constants match TTP06: $R = 8314.472$ J/(mol$\cdot$K), $T = 310$ K, $F = 96485.3415$ C/mol, $[\text{Na}^+]_o = 140$ mM, $[\text{K}^+]_o = 5.4$ mM, $[\text{Ca}^{2+}]*o = 2.0$ mM, $P_{NaK} = 0.03$. All constants registered as buffers for device portability; concentration inputs clamped above $\epsilon = 10^{-12}$ to prevent $\log(0)$.

This is a Layer 0 design choice: the Nernst equation is thermodynamic law, not a model assumption. Hardcoding it with zero learned parameters prevents the model from "learning around" electrochemistry and ensures physically correct reversal potential tracking when concentrations change.

The Nernst module is differentiable: gradients flow $I_{ion} \rightarrow$ Stage 2 $\rightarrow E \rightarrow$ Nernst $\rightarrow$ concentration dims $\rightarrow$ attention. This end-to-end path is critical for teaching concentrations to track correctly during autoregressive rollout.

### 3f. Scaffold Decoders (Training Only)

Two linear decoders supervise the latent during training:


$$
\hat{\mathbf{g}}_{full} = \sigma(W_{full} \cdot \mathbf{s}^{n+1}_{ion} + \mathbf{b}_{full}), \quad L_{gate,full} = \text{MSE}(\hat{\mathbf{g}}_{full}, \mathbf{g}_{true})
$$


$$
\hat{\mathbf{g}}_{comp} = \sigma(W_{comp} \cdot \mathbf{g} + \mathbf{b}_{comp}), \quad L_{gate,comp} = \text{MSE}(\hat{\mathbf{g}}_{comp}, \mathbf{g}_{true})
$$


The full decoder (16 $\rightarrow$ 12, 204 params) ensures the ionic latent encodes biophysically meaningful gate information. The compressed decoder (8 $\rightarrow$ 12, 108 params) verifies that compression preserves it. Targets are 12 HH gates only (m, h, j, r, s, d, f, f$*2$, f$_{Cass}$, X$_{r1}$, X$_{r2}$, X$*s$) -- RR is excluded (no $g*\infty/\tau$), concentrations have their own direct MSE loss.

Both decoders are annealed to zero weight during Phase D training and removed for production inference via `remove_scaffold()`.

## 4. Stage 2: Current Readout (On Critical Path)

Stage 2 is the only computation that must complete before the diffusion solve begins. Its design prioritizes speed.

### 4a. Cross-Attention without Softmax

**Motivation.** The total ionic current is a sum of channel contributions: $I_{ion} = \sum_j I_j$, where each channel current involves a conductance term (from the state) multiplied by a driving force term (from the electrochemical environment). Each conductance latent dimension must "query" the environment to determine what driving force to apply. Cross-attention naturally expresses this routing.

**Formulation.** The 8 conductance tokens form queries; the 9 normalized environment tokens form keys and values:


$$
Q_c = g_c \cdot \mathbf{e}_q[c,:] \in \mathbb{R}^4, \quad K_j = \text{env}_j \cdot \mathbf{e}_k[j,:] \in \mathbb{R}^4, \quad V_j = \text{env}_j \cdot \mathbf{e}_v[j] \in \mathbb{R}^1
$$



$$
\text{scores}_{cj} = \frac{Q_c \cdot K_j}{\sqrt{4}}, \quad \text{attended}*c = \sum_j \text{scores}_{cj} \cdot V_j
$$



$$
I_{ion} = W_2 \cdot \text{GELU}(W_1 \cdot \text{attended} + \mathbf{b}_1) + b_2
$$


**Why no softmax.** $I_{ion}$ is unbounded and can be negative. Softmax forces attention weights to be positive and sum to 1, which would prevent the model from representing subtraction in the driving force $(V_m - E_{rev})$. A conductance token attending to $V_m$ and $E_{Na}$ needs to compute their *difference*, requiring a negative weight on $E_{Na}$. Raw scores preserve this physics.

**Physical interpretation.** Each conductance token "asks" the environment: what driving force should I apply?

- A learned Na-conductance token attends strongly to $V_m$ and $E_{Na}$ $\rightarrow$ discovers $(V_m - E_{Na})$
- A learned CaL-conductance token attends to $V_m$ and $\text{Ca}_{ss}$ $\rightarrow$ discovers GHK-like nonlinear flux
- A learned pump-conductance token attends to $\text{Na}*i$ and $V_m$ $\rightarrow$ discovers $I_{NaK}$ voltage/sodium dependence

### 4b. Environment Normalization

The 9 environment tokens span six orders of magnitude: $\text{K}_i \sim 138$ mM vs $\text{Ca}_i \sim 0.0001$ mM. Without normalization, low-magnitude tokens are invisible to the attention mechanism (their key vectors have negligible norm). Each token is normalized to approximately $[-2, 2]$ using fixed physiological ranges:


$$
\text{env}^{norm}_j = \frac{\text{env}_j - \text{shift}_j}{\text{scale}_j}
$$


The 18 constants (9 shifts + 9 scales) are derived from TTP06 physiological bounds and registered as non-learnable buffers. This is preprocessing, not a learned operation.

### 4c. Output MLP (8 $\rightarrow$ 4 $\rightarrow$ GELU $\rightarrow$ 1)

The MLP combines the 8 per-channel current contributions into the scalar $I_{ion}$. Kirchhoff's law says currents sum linearly, but our 8 conductance dimensions are learned (not literal channels) -- their combination may benefit from the nonlinearity that GELU provides. The hidden dimension of 4 is conservative; upgradable to 8 if training indicates insufficient capacity.

**Zero-bias initialization.** Both bias vectors are initialized to zero so that zero conductance input produces zero $I_{ion}$ output. The model starts as a near-zero current predictor and gradually learns channel contributions during training.

**Implementation.** $\mathbf{e}_q$ (32 params), $\mathbf{e}_k$ (36), $\mathbf{e}_v$ (9), $W_1$ + $\mathbf{b}_1$ (36), $W_2$ + $b_2$ (5). Total: 118 parameters, 970 FLOPs.

## 5. Normalization and Stability

A central challenge for autoregressive models is preventing activation drift over 100,000+ time steps (a single cardiac beat at $\Delta t = 0.01$ ms). v3 uses a layered normalization strategy where each sub-stage has its own stability mechanism matched to its specific failure mode:


| Sub-stage       | Failure Mode      | Mechanism                                    | Guarantee                                         |
| --------------- | ----------------- | -------------------------------------------- | ------------------------------------------------- |
| 1a. Attention   | Latent drift      | Contractive sigmoid gate                     | $|z^{mid} - \text{target}| < |z - \text{target}|$ |
| 1c. MLP         | Correction blowup | Learned $\alpha$ mixing (convex combination) | Output bounded between residual and correction    |
| 1c. MLP input   | Input scale drift | Pre-RMSNorm                                  | Stable input magnitude across rollout             |
| 1d. Compression | Path imbalance    | Learned $\beta$ mixing (convex combination)  | Output bounded between linear and nonlinear paths |
| 2. Readout      | Score magnitude   | $1/\sqrt{d}$ scaling                         | Prevents attention saturation                     |


**Why convex combination supersedes spectral norm (v2).** Spectral normalization (used in v2's split GELU stage) bounds the amplification ratio $W_2 \leq 1$ but still allows growth when the correction is added via residual connection. Convex combination provides a strictly tighter bound: the output $\mathbf{y} = (1-\alpha)\mathbf{x} + \alpha \cdot f(\mathbf{x})$ satisfies $\mathbf{y} \leq \max(\mathbf{x}, f(\mathbf{x}))$ regardless of $f$. No amplification is possible by construction, even for adversarial inputs.

**Rejected approaches.** Sigmoid output bounding (vanishing gradients, breaks residual identity), LayerNorm (removes per-dimension magnitude which is meaningful state information), BatchNorm (unstable for batch=1 autoregressive inference), dropout (noise compounds over 100K+ steps).

## 6. Training Strategy

### Phase 1: Stage 1 Isolation with Scaffolds

Stage 1 trains alone using scaffold losses. A curriculum progresses through increasing rollout lengths:

- **Phase A**: Latent space bootstrap via gate autoencoder (encoder: 12 gates $\rightarrow$ 16 ionic dims; decoder weights transfer to scaffold). Cheap, trains in minutes.
- **Phase B**: Simple dynamics on Tier 1 data. B1: single-step teacher forcing. B2: short rollout ($N=10$). B3: medium rollout ($N=100$). Scheduled sampling gradually replaces ground-truth with model predictions.
- **Phase C**: Full dynamics on Tiers 1--4 with gradual data mixing. $\lambda_{gate}$ annealed from 0.3 to 0. Rollout lengths increase to $N=100{,}000$ (one full beat).
- **Phase D**: Robustness on Tiers 5--12. Scaffolds removed. Stress testing with 5-beat rollouts.

### Phase 2: Stage 2 Regression

Stage 1 frozen. Stage 2 trains as a supervised regression: 25 precomputed inputs (8 conductance + 9 environment + padding) $\rightarrow$ $I_{ion}$. No autoregressive rollout required. Trains in minutes because the mapping $(\mathbf{g}, \text{env}) \rightarrow I_{ion}$ is a static function.

### Phase 3: End-to-End Fine-Tuning

Both stages unfrozen. Full autoregressive rollout with $I_{ion}$ loss plus reduced scaffold weight. This phase corrects any distributional mismatch between Stage 1's latent output and Stage 2's training distribution. AdamW optimizer, cosine LR decay within each phase, gradient clipping at maxnorm=1.0.

## 7. Parameter Budget


| Component                                  | Parameters (Inference) | Parameters (Training) | FLOPs                    |
| ------------------------------------------ | ---------------------- | --------------------- | ------------------------ |
| **Stage 1a: Attention**                    |                        |                       |                          |
| $W_q$ (20$\times$4)                        | 80                     | 80                    | --                       |
| $W_k$ (2$\times$4)                         | 8                      | 8                     | --                       |
| $W_v$ (2$\times$4)                         | 8                      | 8                     | --                       |
| $W_{out}$ (4$\times$20)                    | 80                     | 80                    | --                       |
| **1a subtotal**                            | **176**                | **176**               | **572**                  |
| **Stage 1c: Markov MLP**                   |                        |                       |                          |
| $W_1$ (16$\times$16) + $\mathbf{b}_1$      | 272                    | 272                   | --                       |
| $W_2$ (16$\times$16) + $\mathbf{b}_2$      | 272                    | 272                   | --                       |
| $\mathbf{w}_\alpha$ (16)                   | 16                     | 16                    | --                       |
| **1c subtotal**                            | **560**                | **560**               | **1,330**                |
| **Stage 1d: Compression**                  |                        |                       |                          |
| $W_{lin}$ (8$\times$16)                    | 128                    | 128                   | --                       |
| $W_{c1}$ (12$\times$16) + $\mathbf{b}$     | 204                    | 204                   | --                       |
| $W_{c2}$ (12$\times$12) + $\mathbf{b}$     | 156                    | 156                   | --                       |
| $W_{c3}$ (8$\times$12) + $\mathbf{b}$      | 104                    | 104                   | --                       |
| $\mathbf{w}_\beta$ (8)                     | 8                      | 8                     | --                       |
| **1d subtotal**                            | **600**                | **600**               | **1,360**                |
| **Stage 1e: Nernst**                       | **0**                  | **0**                 | **~30**                  |
| **Stage 1 total**                          | **1,336**              | **1,336**             | **3,292**                |
| **Stage 1f: Scaffolds**                    |                        |                       |                          |
| Full decoder (16$\times$12) + $\mathbf{b}$ | --                     | 204                   | 444                      |
| Comp decoder (8$\times$12) + $\mathbf{b}$  | --                     | 108                   | 252                      |
| **Scaffold total**                         | **--**                 | **312**               | **696**                  |
| **Stage 2: Readout**                       |                        |                       |                          |
| $\mathbf{e}_q$ (8$\times$4)                | 32                     | 32                    | --                       |
| $\mathbf{e}_k$ (9$\times$4)                | 36                     | 36                    | --                       |
| $\mathbf{e}_v$ (9$\times$1)                | 9                      | 9                     | --                       |
| $W_1$ (4$\times$8) + $\mathbf{b}_1$        | 36                     | 36                    | --                       |
| $W_2$ (1$\times$4) + $b_2$                 | 5                      | 5                     | --                       |
| **Stage 2 total**                          | **118**                | **118**               | **970**                  |
| **Grand total**                            | **1,454**              | **1,766**             | **4,262 (970 critical)** |


At 970 critical-path FLOPs, Stage 2 is 1.6$\times$ the cost of a TTP06 Rush-Larsen step (~~600 FLOPs) and 0.57$\times$ the cost of an ORd step (~~1,700 FLOPs). Stage 1's 3,292 FLOPs run in parallel with the diffusion solve (which dominates 94% of bidomain wall time) and contribute zero additional latency.

## 8. Scaling: Small TTP06 vs Full ORd-Ready

The architecture is parameterized, not fixed. The small TTP06 configuration validates the design; the full configuration targets multi-model capability:


| Hyperparameter    | Small (TTP06)                                  | Full (ORd-ready)                                |
| ----------------- | ---------------------------------------------- | ----------------------------------------------- |
| ionicstate        | 16                                             | 32                                              |
| concentrations    | 4                                              | 4                                               |
| carriedstate      | 20                                             | 36                                              |
| conductancelatent | 8                                              | 16                                              |
| attndim           | 4                                              | 4                                               |
| MLP hidden        | 16                                             | 32                                              |
| Compression       | 16$\rightarrow$12$\rightarrow$12$\rightarrow$8 | 32$\rightarrow$24$\rightarrow$24$\rightarrow$16 |
| Stage 2 queries   | 8                                              | 16                                              |
| Inference params  | ~1,454                                         | ~4,950                                          |


**Multi-model conditioning (planned).** A single model learns both TTP06 and ORd, conditioned on a model ID label token fed to the cross-attention. TTP06 uses $\sim$16 of 32 ionic dims; ORd uses all 32. Shared architecture (attention, compression, readout) is identical -- only latent utilization differs. Benefits: richer latent representation for future arc light fine-tuning, single deployable model, natural curriculum (TTP06 first, add ORd later).

## 9. Comparison to v2


| Aspect                     | v2                                 | v3                                                                          | Why                                                                             |
| -------------------------- | ---------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Stages**                 | 3 sequential                       | 2 parallel                                                                  | Stage 1 off critical path (free compute during diffusion)                       |
| **Carried state**          | 16 latent (opaque)                 | 20 = 16 ionic + 4 explicit concentrations                                   | Concentrations named, supervised directly, feed Nernst                          |
| **Cross-channel coupling** | Split GELU + spectral norm         | Markov MLP + learned $\alpha$ mixing                                        | Convex combination gives tighter stability bound; MLP allows full-rank coupling |
| **Compression**            | None (16 $\rightarrow$ readout)    | 16 $\rightarrow$ 12 $\rightarrow$ 12 $\rightarrow$ 8                        | Separates state evolution (16 dims) from readout input (8 dims)                 |
| **Current readout**        | Chebyshev KAN (per-dim polynomial) | Cross-attention (conductance queries environment)                           | Chebyshev applied nonlinearity in wrong place; cross-attention learns routing   |
| **Reversal potentials**    | None (implicit in latent)          | Explicit Nernst (fixed physics, 0 params)                                   | Correct electrochemistry is hardcoded, not learned                              |
| **Environment tokens**     | $V_m$ only                         | 9 tokens: $V_m$, 4 reversal potentials, 4 concentrations                    | Readout sees driving forces and raw concentrations (GHK, pumps)                 |
| **Normalization**          | Spectral norm + RMSNorm            | Learned $\alpha/\beta$ mixing + Pre-RMSNorm                                 | Convex combination cannot amplify; spectral norm can (via residual)             |
| **Scaffold**               | 1 decoder (16 $\rightarrow$ 18)    | 2 decoders (16 $\rightarrow$ 12 gates, 8 $\rightarrow$ 12 gates) + conc MSE | Dual supervision ensures compression preserves information                      |
| **Softmax**                | Avoided (degenerate for 1 token)   | Absent by design                                                            | Negative scores physically meaningful in readout                                |
| **Attention dim**          | 8                                  | 4                                                                           | 2 inputs, smooth targets -- 4 suffices                                          |
| **Inference params**       | 642                                | 1,454                                                                       | Additional capacity in MLP + compression, but critical-path cost is comparable  |
| **Critical-path FLOPs**    | 886 (all stages)                   | 970 (Stage 2 only)                                                          | v3 critical path is similar despite 2.3$\times$ total params                    |


The central insight driving v3 is the decoupling of state evolution from current readout. In v2, all three stages ran sequentially on the critical path. In v3, the expensive work (attention, MLP, compression) runs in the background, and only the lightweight current readout (118 params, 970 FLOPs) gates the simulation timestep.