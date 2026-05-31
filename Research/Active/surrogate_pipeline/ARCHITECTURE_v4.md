# Ionic Surrogate v4

## Abstract

We present the fourth iteration of a neural surrogate for the ionic step of cardiac electrophysiology simulation. The architecture is a Neural Ordinary Differential Equation (NODE) in which a five-layer multilayer perceptron with pre-LayerNorm and a gated full-path residual defines a continuous rate field `dz/dt = f_θ(z, V_m)` over a 24-dimensional state. State consists of a 20-dimensional learned ionic latent together with four physically named concentrations. A frozen scaffold decoder pins the latent origin `z = 0` to the TTP06 physiological rest state by construction, so the latent encodes deviation from rest. Training uses adjoint integration through `odeint` with two physics-informed attractor regularizers: one anchoring the rest fixed point, and one generalizing to the full voltage-clamp steady-state manifold. The design is the direct response to a Session 27 (2026-04-19) diagnostic that demonstrated the previous architecture's inference error was entirely capacity-bound rather than integrator-bound, with calcium in the sarcoplasmic reticulum and two slow gate variables dominating the residual. The inference parameter count is approximately 7,600, a 3.6× increase over v3 for matched dynamics and approximately one-fifteenth the cost of a first-principles Rush-Larsen step on modern tensor hardware.

## 1. Design Lineage

v2 (2026-03-23) introduced the two-stage factorization: state evolution off the simulation critical path, current readout on it. Stage 1 combined a 20-dimensional cross-attention (ionic and concentration dimensions sharing weights) with a two-round GELU coupling stage and a Chebyshev readout. The architecture achieved 18-test validation but was never trained at scale; it was superseded before Phase A.

v3 (2026-04-01) factored the ionic and concentration rate paths. A per-dimension cross-attention (VoltageAttention) drove the 16 ionic latent dimensions; a B-spline KAN drove the four concentration dimensions; the two were concatenated into `dz/dt`. Concentrations were excluded from the attention to avoid a feedback instability observed in an earlier MLP-based concentration network. Training reached `val_loss = 0.0084` on multi-BCL T1 data after seven epochs (checkpoint `multi_bcl_002/best.pt`).

v3 had two substantive flaws exposed by the Session 27 diagnostic. First, VoltageAttention on scalar per-dimension inputs collapses algebraically to a 32-parameter switched linear ODE; the full 136-parameter attention machinery is redundant, and direct inspection of trained weights confirmed the MLP correction path was essentially unused (see Failed Approaches, IDEALOG). Second, the concentration KAN is additively universal in its inputs but cannot represent multiplicative cross-terms such as the sodium-calcium exchanger current `I_NaCa ∝ exp(γV_mF/RT)·Na_i^3·Ca_i`, which is an unavoidable feature of TTP06 physics.

v4 resolves both issues by unifying the ionic and concentration rate paths into a single multilayer perceptron, restoring concentration self-input that v3 had dropped as collateral damage, and adding compositional depth sufficient for quartic cross-products. The ionic latent dimension is expanded from 16 to 20 to provide explicit capacity for the slow variables that v3's diagnostic identified as chronically under-tracked. The scaffold decoder acquires a frozen bias term that pins the latent origin to the physiological rest state, giving the latent the semantic interpretation of deviation from rest. Two physics-informed attractor regularizers train the rate field to vanish on the full voltage-clamp steady-state manifold — the defining property of Hodgkin–Huxley dynamics.

## 2. Design Philosophy

### 2.1 Layer 0 Framework

Architectural decisions derive from a three-layer reasoning hierarchy, prioritized top-down.

**Layer 0** is physical reality at the cardiac membrane. Ion channels are multi-state proteins whose open probability depends on membrane voltage; the current through a channel is the product of conductance and electrochemical driving force; concentrations change slowly relative to the action potential timescale and create memory of past activity; dynamics span three orders of magnitude in timescale. These are the ground truths the architecture must capture.

**Layer 1** comprises biophysics models — TTP06, ORd, and their descendants. These are human-constructed theories explaining Layer 0. Hodgkin–Huxley gate independence, specific Markov state diagrams, three-compartment calcium cycling, the `m^3hj` product for sodium open probability: all are modeling choices, not measured physical laws. A model's validation on experiment does not elevate its parameterization to truth.

**Layer 2** is the neural architecture. It should be informed by what Layer 1 got right about Layer 0 but not enslaved to Layer 1's specific formulations. Where a neural network can learn a relationship more naturally than a biophysics equation prescribes, we let it.

The operative maxim is that physics provides the skeleton and machine learning fills the flesh. Fixed physics — Nernst's equation, operator splitting, Kirchhoff's current summation — is hardcoded with zero learned parameters. Learned components handle relationships that differ between ionic models, or that reality may implement differently than any single biophysics model assumes.

### 2.2 Consequences of the Maxim

The v4 design decisions each trace back to this philosophy.

The latent is not hardcoded to Hodgkin–Huxley gate variables even though we have that data. Hardcoding would commit the architecture to TTP06's specific parameterization and preclude generalizing to models with different state sets (ORd has 41 states against TTP06's 18) or to experimental data that contains no gate observables. See IDEALOG Session 27 Q10 for the full argument against hardcoding.

The scaffold decoder is linear. Any nonlinearity in the decoder would launder learning into the training-only scaffold; at deployment, when the scaffold is removed, that knowledge would disappear. A linear map's role is rotation: the decoder converts an abstract latent basis to the basis of observable gates without adding expressivity the latent must earn.

Concentrations are explicit rather than latent. They have identical meaning across ionic models, they have closed-form evolution via Nernst for their role in reversal potentials, and they are the physical quantities measured experimentally. Storing them as-is is the simplest correct choice.

`Ca_SR` is an exception. It is internal to the sarcoplasmic reticulum, does not appear in any reversal potential or in the Stage 2 environment token, and has no cross-model semantic constancy (TTP06 has one SR compartment; ORd has two). It is tracked implicitly inside the 20-dimensional latent and decoded through the scaffold during training.

## 3. Problem Formulation

The TTP06 ventricular ionic model evolves 18 state variables (12 Hodgkin–Huxley gates, the CaMKII-like state variable `RR`, four intracellular concentrations, and the SR calcium) according to

$$
\frac{dg_i}{dt} = \frac{g_{\infty,i}(V_m) - g_i}{\tau_i(V_m)}, \qquad I_{ion} = \sum_j \bar{g}_j \prod_i g_i^{a_{ij}} (V_m - E_j).
$$

The surrogate replaces this system with a 24-dimensional carried state

$$
\mathbf{z}^n = [\mathbf{s}_{ion}^n, \mathbf{c}^n] \in \mathbb{R}^{24}, \qquad \mathbf{s}_{ion} \in \mathbb{R}^{20}, \quad \mathbf{c} = [\text{Na}_i, \text{K}_i, \text{Ca}_i, \text{Ca}_{ss}] \in \mathbb{R}^4,
$$

evolved as a continuous-time NODE and sampled as needed by the solver. Stage 2 reads a conductance latent `g ∈ ℝ^8` derived from `z` by a compression step, plus a normalized environment vector of reversal potentials and concentrations, and produces the ionic current:

$$
\frac{d\mathbf{z}}{dt} = f_\theta(\mathbf{z}, V_m) \quad \text{(Stage 1, off critical path)},
$$

$$
I_{ion}^n = h_\phi\big(\text{compress}(\mathbf{z}^n), \text{env}(\mathbf{z}^n, V_m^n)\big) \quad \text{(Stage 2, on critical path)}.
$$

The factorization is deliberate. Stage 2's output feeds the diffusion step that advances `V_m`; its latency is on the critical path. Stage 1's output becomes the next timestep's latent and can be computed in parallel with the diffusion solve on a separate CUDA stream. Stage 1 can therefore be as expressive as physics demands without paying a wall-clock penalty.

## 4. Constants and Dimensions

Every constant in v4 plays one of four roles: a latent capacity that the model fills at will; an explicit physical quantity that the simulator and the surrogate share; an internal width for the rate MLP; or a scaffold target whose value we compare against ground truth during training. The table below lists every constant, its value, and its role.

| Constant | Value | Role | Notes |
|---|---:|---|---|
| `IONIC_DIM` | 20 | latent capacity | abstract, learned; represents deviation from rest after bias freeze |
| `CONC_DIM` | 4 | explicit physical | `[Na_i, K_i, Ca_i, Ca_ss]` in simulator units |
| `CARRIED_DIM` | 24 | total state | `IONIC_DIM + CONC_DIM`, the dimension `z` the ODE solver integrates |
| `H` (hidden) | 32 | MLP width | inside `StateRateMLP`, slightly above max(in,out) for mixing headroom |
| `COND_DIM` | 8 | latent capacity | compressed conductance, input to Stage 2; spans 5 effective gate products with slack |
| `N_IONIC_TARGETS` | 14 | scaffold output | 12 HH gates + RR + CaSR; free linear decoder from `IONIC_DIM` |
| `N_COND_TARGETS` | 5 | scaffold output | effective gate products: `G_Na(m³hj), G_CaL(dff2fCass), G_to(rs), G_Kr(Xr1·Xr2), G_Ks(Xs²)` |
| `V_rest` | −85.23 mV | physical constant | TTP06 resting membrane voltage |
| Stage 1 input dim | 25 | —  | `CARRIED_DIM + 1` (Vm is the only external driver) |
| Stage 1 output dim | 24 | — | same as `CARRIED_DIM`; one rate per state dim |

Two pairings deserve explicit commentary because they are the single most common source of confusion.

First, `IONIC_DIM = 20` and `N_IONIC_TARGETS = 14` are not equal and never should be. The former is the *computational capacity* of the latent, chosen to give the model headroom for slow variables, stiff Markov-like states, and cross-model universality. The latter is the number of *named physical observables* we can regress against during training. The free linear decoder `W_d \in \mathbb{R}^{14 \times 20}` rotates the latent basis to the observable basis; four of the twenty latent dimensions are slack the model can use for internal tracking that has no direct observable counterpart. This was 16 and 14 in v3, which gave two slack dimensions; the Session 27 diagnostic attributed the slow-variable tracking failure in part to insufficient slack, and v4 doubles it.

Second, `CONC_DIM = 4` and `CARRIED_DIM = 24` differ because concentrations are stored and evolved natively, not through the latent. The rate predictor sees the full 24-dim state as input and outputs a 24-dim rate, so the rate for `Na_i` is a genuine `dNa_i/dt` in millimolar per millisecond, not a rotated latent rate. No decoder is involved for concentrations; their loss is a direct MSE against simulator concentrations.

## 5. Stage 1 — Rate Predictor

Stage 1 defines the continuous rate field `f_θ : \mathbb{R}^{24} \times \mathbb{R} \to \mathbb{R}^{24}` that the ODE solver integrates. Its realization in v4 is a single 5-layer perceptron with pre-LayerNorm and a gated full-path linear skip, plus a zero-initialized linear readout. The concentration pathway is not separated from the ionic pathway; all 24 output rates come from the same network.

### 5.1 Input and Output

The input is the concatenation

$$
\mathbf{x} = [\mathbf{z}, V_m] \in \mathbb{R}^{25}.
$$

`V_m` is the only external driver; stimulus current is excluded because it affects the membrane only through `V_m`, and the model sees `V_m` at the next timestep via the diffusion step. Removing `I_{stim}` preserves the operator-splitting structure and removes a time-varying input that would complicate the vector field.

The output is

$$
\frac{d\mathbf{z}}{dt} = f_\theta(\mathbf{x}) \in \mathbb{R}^{24}.
$$

The solver integrates this continuously; during inference, a forward Euler step `z_{n+1} = z_n + \Delta t \cdot f_\theta(z_n, V_m)` suffices at the native `\Delta t = 0.01 \text{ ms}`, as the Session 27 integrator diagnostic confirmed that Euler truncation error is 215× smaller than model-capacity error at that timestep.

### 5.2 Forward Pass

The forward pass is

$$
\mathbf{h}_1 = \text{GELU}(W_1 \mathbf{x} + \mathbf{b}_1),
$$

$$
\mathbf{h}_{k+1} = \text{GELU}(W_{k+1} \, \text{LayerNorm}_k(\mathbf{h}_k) + \mathbf{b}_{k+1}), \quad k = 1, 2, 3, 4,
$$

$$
\mathbf{r}_\text{deep} = W_r \, \text{LayerNorm}_5(\mathbf{h}_5) + \mathbf{b}_r,
$$

$$
\mathbf{r}_\text{skip} = W_s \mathbf{x},
$$

$$
\alpha = \sigma(\ell), \qquad f_\theta(\mathbf{x}) = \mathbf{r}_\text{deep} + \alpha \odot \mathbf{r}_\text{skip},
$$

where `ℓ ∈ ℝ^{24}` is a per-dimension learned logit, `α = σ(ℓ)` is the gated mixing weight, and `⊙` denotes elementwise multiplication.

The stem `W_1` projects the unnormalized 25-dimensional input up to the hidden width. LayerNorm is applied only to hidden representations, not to the raw input, because `[z, V_m]` has no common scale — gates are bounded near unity, `V_m` ranges over ±100 mV, concentrations span six orders of magnitude between `Ca_i` and `K_i`. Normalizing the input would destroy physical magnitude information the stem needs for gating.

The LayerNorm modules follow the pre-norm convention popularized by LLaMA: each hidden Linear sees a normalized input. Pre-norm is preferred over post-norm for depth-5 plain stacks because it produces a more stable gradient landscape and permits higher learning rates without explosion at deep layers.

### 5.3 Gated Full-Path Skip

The skip `W_s : \mathbb{R}^{25} \to \mathbb{R}^{24}` is a linear map without bias. Its role is to give the model an explicit choice between representing the rate field as a predominantly linear function of `(z, V_m)` or as a predominantly deep-MLP function. The choice is controlled by `α = σ(ℓ)`, a per-output-dimension gate; large `α` routes more of the rate through the linear skip, small `α` routes more through the deep MLP.

This is not the same as a per-layer residual connection and is not the same as a Transformer-FFN block residual. Per-layer residuals protect gradient flow through depth but do not give the model a transparent knob for the linear-vs-nonlinear decomposition. A single full-path skip makes this decomposition explicit and trainable: after training, `||W_s||` and the distribution of `α` directly report how much of the dynamics the model preferred to capture linearly.

The logit `ℓ` is initialized to zero, so `α = 0.5` at initialization — half linear, half deep. The deep readout `W_r` and its bias are zero-initialized so the deep contribution is zero at initialization; the rate field at initialization is therefore `\frac{1}{2} W_s \mathbf{x}`, a small Xavier-initialized linear map of the input. This is sufficient to produce a non-vanishing gradient through both paths on the first training step, avoiding the `fc3`-stuck pathology observed in the current checkpoint (Session 27 diagnostic, IDEALOG Q9).

### 5.4 Parameter Breakdown

| Component | Shape | Params |
|---|---|---:|
| `W_1, b_1` (stem) | `(32, 25) + (32,)` | 832 |
| `W_2, b_2` (hidden 1) | `(32, 32) + (32,)` | 1,056 |
| `W_3, b_3` (hidden 2) | `(32, 32) + (32,)` | 1,056 |
| `W_4, b_4` (hidden 3) | `(32, 32) + (32,)` | 1,056 |
| `W_5, b_5` (hidden 4) | `(32, 32) + (32,)` | 1,056 |
| `W_r, b_r` (readout) | `(24, 32) + (24,)` | 792 |
| `W_s` (skip) | `(24, 25)` | 600 |
| `ℓ` (gate logit) | `(24,)` | 24 |
| `γ_k, β_k` (five LN) | `5 × 2 × 32` | 320 |
| **Stage 1 dzdt total** | | **6,792** |

### 5.5 Initialization

The stem, hidden Linears, skip, and readout are Xavier-uniform-initialized except where noted. The readout `W_r` and bias `b_r` are zero-initialized to ensure the deep contribution to the rate field vanishes at step zero. LayerNorm `γ` is one-initialized, `β` zero-initialized. The gate logit `ℓ` is zero-initialized, giving `α = 0.5` and an initial rate field equal to half the skip's Xavier-magnitude linear map.

### 5.6 Rationale for Depth 5, Width 32, and the Absence of KAN

Depth-5 is calibrated to TTP06 and ORd cross-product physics. Each GELU layer contributes one level of nonlinear composition; `I_{NaCa} \propto \exp(\gamma V_m F/RT)\, \text{Na}_i^3\, \text{Ca}_i` and `I_{CaL}` flux require quartic composition, which is at the edge of 3-layer MLP capability but comfortably within 4-5 layer range. Five layers also provide headroom for ORd, which has 41 states including a Markov `I_{NaL}` and additional calcium compartments; in single-model TTP06 use, four layers would suffice, but the additional layer costs only 1,056 parameters.

Width `H = 32` sits slightly above the 25-dimensional input and 24-dimensional output. Wider hidden dims were considered and rejected because the KAN readout from v3, had it been retained, scales parameters as `H \cdot H_{\text{out}} \cdot (G + K)` in the grid and order constants; widening to 48 would have cost more than two additional layers of depth. With the KAN eliminated in v4, this consideration no longer applies, but the empirical rationale — that cross-product physics favors compositional depth over mixing width — stands.

The KAN readout was eliminated for cost reasons. The Session 27 trade-off analysis showed that a grid-5 order-3 KAN `(32 \to 24)` would contribute 6,912 parameters, exceeding 60% of the Stage 1 budget. A plain zero-initialized linear readout recovers nearly all the expressivity a 5-layer MLP with GELU activations already provides, because compositional nonlinearity lives upstream in the MLP and per-dimension rate shaping is adequately representable through row-specific weights of `W_r`. If post-training diagnostics show that slow variables specifically require spline-basis per-dim shaping, a small KAN with grid 2 or 3 can be re-introduced at roughly 3,000 extra parameters.

## 6. Stage 1 — Compression

The conductance latent `g \in \mathbb{R}^8` is a learned compression of the 24-dimensional state, intended to represent the effective gate products that enter the ionic current summation. Compression runs once per timestep on the integrated state `z(t+\Delta t)` and produces a latent input for Stage 2 at the next step.

The compression retains v3's architecture unchanged:

$$
\ell(\mathbf{z}) = W_{\text{lin}} \mathbf{z},
$$

$$
n(\mathbf{z}) = W_{c3} \, \text{GELU}(W_{c2} \, \text{GELU}(W_{c1} \mathbf{z} + \mathbf{b}_{c1}) + \mathbf{b}_{c2}) + \mathbf{b}_{c3},
$$

$$
\mathbf{g} = \ell(\mathbf{z}) + \sigma(\boldsymbol{\lambda}) \odot n(\mathbf{z}),
$$

where `W_{\text{lin}} : \mathbb{R}^{24} \to \mathbb{R}^{8}` is a linear bypass without bias, the two-hidden-layer MLP computes cross-dimensional products, and `λ ∈ ℝ^8` is a per-output learned logit initialized to `−5` so that `σ(λ) ≈ 0.007` at initialization, making the MLP contribution essentially zero and leaving the linear path as the initial compression. Hidden widths are 12 and 12.

Two hidden GELU layers are adequate for the triple products (`m^3hj`) and quartic products (`dff2fCass`) that appear in TTP06's effective conductances. The first hidden layer computes pairwise products; the second composes into triples. The initial-zero contribution of the nonlinear path protects early training from compression-driven gradient pathologies.

Compression parameter count: `W_{\text{lin}}` = 192, nonlinear path = 512, logit = 8, total = 712.

## 7. Stage 1 — Nernst Reversal Potentials

Nernst is closed-form physics with zero learned parameters. It converts the explicit concentration dimensions of `z(t+\Delta t)` into four reversal potentials used by Stage 2:

$$
E_{\text{Na}} = \tfrac{RT}{F} \ln \frac{[\text{Na}]_o}{[\text{Na}]_i}, \quad E_K = \tfrac{RT}{F} \ln \frac{[K]_o}{[K]_i}, \quad E_{\text{Ca}} = \tfrac{RT}{2F} \ln \frac{[\text{Ca}]_o}{[\text{Ca}]_i},
$$

$$
E_{\text{Ks}} = \tfrac{RT}{F} \ln \frac{[K]_o + P [\text{Na}]_o}{[K]_i + P [\text{Na}]_i}, \qquad P = 0.03.
$$

External concentrations are constants (`[\text{Na}]_o = 140 \text{ mM}`, `[K]_o = 5.4 \text{ mM}`, `[\text{Ca}]_o = 2.0 \text{ mM}`). The operation is differentiable, so gradients flow from Stage 2's current loss back through the reversal potentials into the concentration dimensions of the latent and onward through the rate predictor.

## 8. Scaffold Decoders

Scaffold decoders exist only during training. They are removed at deployment via `remove_scaffold()` because they produce quantities the production inference pipeline does not consume. Their role is to ground the learned latent in observable physical quantities during training so that the latent carries the information Stage 2 and downstream users need.

### 8.1 Ionic State Decoder

The ionic state decoder is a free linear map

$$
\hat{\mathbf{y}} = W_d \mathbf{z}_{ion} + \mathbf{b}_d, \qquad W_d \in \mathbb{R}^{14 \times 20}, \quad \mathbf{b}_d \in \mathbb{R}^{14}
$$

where `\mathbf{y}` concatenates the 12 Hodgkin–Huxley gates, the CaMKII-like variable `RR`, and `Ca_{SR}` in a fixed order.

**The bias is frozen** (Session 27, 2026-04-19). At initialization, `\mathbf{b}_d` is set to the TTP06 physiological rest ionic state — the values taken by the 14 observables when the cell has been at rest for sufficiently long — and `\mathbf{b}_d.\text{requires\_grad}` is set to false. The rest values (to four decimal places) are `(m, h, j, r, s, d, f, f_2, f_{\text{Cass}}, X_{r1}, X_{r2}, X_s, RR, Ca_{\text{SR}}) = (0.0017, 0.7444, 0.7045, 0.0000, 1.0000, 0.0000, 0.7888, 0.9755, 0.9953, 0.0062, 0.4712, 0.0017, 0.9073, 3.6400)`.

The freeze has a precise semantic consequence. At `\mathbf{z}_{ion} = \mathbf{0}`, the decoder produces exactly the rest state, by construction:

$$
\hat{\mathbf{y}}(\mathbf{z}_{ion} = \mathbf{0}) = \mathbf{b}_d = \mathbf{y}_{\text{rest}}.
$$

The latent therefore has the interpretation of *deviation from rest*. A non-zero `\mathbf{z}_{ion}` encodes how far the cell is from its resting conformation, in a learned basis. This resolves the t=0 prediction cliff observed in v3, where the latent's zero origin decoded to arbitrary scaffold-bias values unrelated to physiological rest.

The weight `W_d` remains free and learnable (280 parameters of 294). Because `W_d` is unconstrained, the latent basis is free to rotate in any way the rate predictor finds convenient; the decoder is not an identity map, and the latent dimensions do not correspond one-to-one with observable gates. This is the *linear span* property: the twenty-dimensional latent subspace is required to span the fourteen observables through `W_d`, nothing stronger.

### 8.2 Gate Conductance Decoder

The gate conductance decoder is a free linear map

$$
\hat{\mathbf{G}} = W_G \mathbf{g} + \mathbf{b}_G, \qquad W_G \in \mathbb{R}^{5 \times 8}, \quad \mathbf{b}_G \in \mathbb{R}^5,
$$

regressing the compressed conductance latent against five effective gate products: `G_{\text{Na}}(m^3hj)`, `G_{\text{CaL}}(dff_2f_{\text{Cass}})`, `G_{\text{to}}(rs)`, `G_{\text{Kr}}(X_{r1}X_{r2})`, and `G_{\text{Ks}}(X_s^2)`. These are the quantities the analytic TTP06 current equations multiply by the driving force to obtain individual ionic currents. Pinning the conductance latent to them during training guarantees that Stage 2 receives an interpretable 8-dimensional input and that downstream fine-tuning does not have to re-discover the compression's role. Parameter count: 45 (none frozen).

### 8.3 Linear Decoder Interpretation

The choice of a linear decoder without activation for both scaffolds is load-bearing and deserves elaboration.

Any nonlinearity added to the decoder would *launder* learning into a training-only module. A sigmoid on gate outputs would force gates into `[0, 1]`, improving surface metrics during training — but at deployment, when the scaffold is removed, the latent remains free to produce values the decoder would have clipped, and downstream Stage 2 consumes whatever the latent produces. Training-time improvements from decoder nonlinearity are therefore deceptive. The correct place for the clipping constraint, if it is wanted, is a soft regularizer on the latent's decoded output, not the decoder's forward pass.

The linear decoder *spans* the observables without *aligning* to them. A free `W_d` can recover any 14 linearly independent directions from a 20-dimensional latent, but it does so through arbitrary rotations — the latent is not required to encode `m` in its first dimension, or to dedicate any single dimension to any single observable. The user-legible interpretation of the latent is accessible only by inverting `W_d` post-training; it is not present by construction.

If alignment — not just span — is ever desired, the path is to constrain `W_d` to a diagonal or block-diagonal form, or to add an orthogonality regularizer on its rows. v4 takes neither path; the latent is a free basis, and the model is permitted to discover whatever rotation serves its training objective.

## 9. Stage 2 — Current Readout

Stage 2 is unchanged in v4 from the v3 formulation and is described here for completeness.

### 9.1 Environment Tokens

Stage 2's inputs are the conductance latent `\mathbf{g} \in \mathbb{R}^8` and a nine-dimensional environment vector

$$
\text{env} = [V_m, E_{\text{Na}}, E_K, E_{\text{Ca}}, E_{\text{Ks}}, \text{Na}_i, K_i, \text{Ca}_i, \text{Ca}_{ss}],
$$

normalized to approximately `[-1, 1]` using known physiological ranges. Normalization is fixed at eighteen physiological constants (nine means, nine ranges) pulled from the loss-normalization module; these are not learned.

### 9.2 Conductance Attention

A cross-attention with eight queries and nine keys/values, without softmax:

$$
Q = \mathbf{g} \odot e_q, \quad K = \text{env} \odot e_k, \quad V = \text{env} \odot e_v,
$$

$$
\text{scores} = \frac{QK^\top}{\sqrt{d}}, \qquad \text{attended} = \text{scores} \cdot V.
$$

Here `e_q \in \mathbb{R}^{8 \times 4}`, `e_k \in \mathbb{R}^{9 \times 4}`, `e_v \in \mathbb{R}^{9 \times 1}`, `d = 4`. The output is an 8-dimensional vector. The absence of softmax is physical: scores can be negative because the driving force `V_m - E_j` can change sign, and softmax would erase sign information in the score distribution.

### 9.3 Output MLP and Assembly

A two-layer MLP produces the scalar current:

$$
I_{ion} = W_2 \cdot \text{GELU}(W_1 \cdot \text{attended} + \mathbf{b}_1) + b_2, \qquad W_1 \in \mathbb{R}^{4 \times 8}, W_2 \in \mathbb{R}^{1 \times 4}.
$$

Total Stage 2 parameters: 118 (attention: 77; MLP: 41).

Stage 2 has not been trained in any v3 checkpoint. It enters the training pipeline during Phase C (see §11).

## 10. Physics-Informed Attractors

A central innovation of v4 is the addition of two attractor regularizers to the training loss. Both enforce that the rate field vanishes on the known fixed-point manifold of Hodgkin–Huxley dynamics.

### 10.1 Rest Attractor

The rest attractor regularizer is

$$
\mathcal{L}_{\text{rest}} = \left\| f_\theta(\mathbf{z}_{\text{rest}}, V_{\text{rest}}) \right\|^2,
$$

where `\mathbf{z}_{\text{rest}} = [\mathbf{0}_{20}, \mathbf{c}_{\text{rest}}]`, `\mathbf{c}_{\text{rest}} = [10.0, 138.0, 10^{-4}, 2 \times 10^{-4}]`, and `V_{\text{rest}} = -85.23 \text{ mV}`. This pressures the rate field to be exactly zero at the physiological rest state, making `\mathbf{z}_{\text{rest}}` a true fixed point of the learned dynamics.

Coupled with the frozen-decoder-bias convention of §8.1, this gives the rest state a privileged role: the decoder places it at `\mathbf{z}_{ion} = \mathbf{0}` by construction, and the regularizer keeps the dynamics from drifting from it. Both are necessary. The decoder freeze alone guarantees that decoding `\mathbf{z} = \mathbf{0}` produces rest, but does not prevent the rate field from pushing `\mathbf{z}` away from zero when the cell is at rest; the regularizer prevents that drift.

### 10.2 Voltage-Clamp Steady-State Attractor

The rest attractor is a single point on a richer manifold. For any voltage `V` held constant, Hodgkin–Huxley dynamics relax deterministically to a fixed point `\mathbf{z}_{ss}(V)` where every gate takes its steady-state value `g_{\infty,i}(V)`. The voltage-clamp steady-state attractor regularizer generalizes the rest attractor to this manifold:

$$
\mathcal{L}_{\text{vclamp}} = \mathbb{E}_{V \sim \text{grid}} \left\| f_\theta(\mathbf{z}_{ss}(V), V) \right\|^2,
$$

evaluated over a discrete voltage grid `\{-90, -60, -40, -20, 0, +20, +40, +60\} \text{ mV}`. At each grid voltage, `\mathbf{z}_{ss}(V)` is precomputed by simulating TTP06 with the voltage clamped and integrating until `\dot{\mathbf{z}} \approx 0` (approximately 500 ms).

`\mathcal{L}_{\text{rest}}` is the `V = V_{\text{rest}}` special case. In practice the two losses can be combined into a single `\mathcal{L}_{\text{vclamp}}` evaluated on the grid `\{V_{\text{rest}}\} \cup \{-90, -60, -40, -20, 0, +20, +40, +60\}`.

This regularizer directly encodes the defining property of Hodgkin–Huxley dynamics — that voltage-clamped cells asymptotically reach the fixed-point manifold `\{\mathbf{z}_{ss}(V) : V \in \mathbb{R}\}` — into the learned rate field. It is physics injected at the correct level of abstraction: not committing to any specific parameterization of `g_{\infty}(V)` or `\tau(V)`, but requiring the correct asymptotic behavior.

### 10.3 Multi-Model Extension

Per-model rest states and steady-state grids are required for multi-model training. TTP06 and ORd have different rest values (ORd has eight concentrations including `Na_{ss}`, `Ca_{nsr}`, `Ca_{jsr}`, and a CaMKII trophic state `CaMKt`), different gate sets, and different voltage-dependent steady-state curves. The architecture handles this by selecting the appropriate per-model rest constant and `z_{ss}(V)` table via a model-identifier input; the rate predictor itself remains shared across models.

### 10.4 Optional Tier-2 Attractors

Two additional attractors were analyzed in Session 27 and deferred. Both are cheap to add if post-training diagnostics indicate a need.

A *contraction-toward-target* regularizer

$$
\mathcal{L}_{\text{contract}} = \lambda \sum_d \text{relu}\big( -\text{sign}(\mathbf{z}_{ss}(V)_d - \mathbf{z}_d) \cdot \dot{\mathbf{z}}_d \big)
$$

softly penalizes rate components pointing away from the local fixed point at any instantaneous state. This is the Rush-Larsen contraction principle expressed as a soft directional constraint rather than a rigid exponential form.

A *decoded-gate bounds* penalty

$$
\mathcal{L}_{\text{bounds}} = \sum_{d \in \text{HH}} \text{relu}(-\hat{y}_d) + \text{relu}(\hat{y}_d - 1)
$$

penalizes decoded gate predictions falling outside `[0, 1]`. The sum is over the twelve Hodgkin–Huxley gate rows of the decoder output; `RR` and `Ca_{SR}` are excluded because their ranges differ.

## 11. Training Pipeline

Training proceeds through a phase hierarchy that introduces supervision progressively, mirroring the architecture's factorization.

### 11.1 Phase Hierarchy

| Phase | Trains | Notes |
|---|---|---|
| A (ionic-only) | `StateRateMLP` + `W_d` + rest/vclamp regularizers | first pass; scaffold-supervised |
| `conc_only` (optional) | `StateRateMLP` concentration subset, frozen ionic | alternative to Phase A if concentration drift appears |
| B (ionic + conductance) | + `compress` + `W_G` | introduces compression and conductance targets |
| C (current readout) | + Stage 2; Stage 1 frozen | first time Stage 2 sees gradient |
| D (end-to-end fine-tune) | all; scaffold loss annealed to zero | decoder bias remains frozen |
| E (multi-tier, multi-model) | all; curriculum over BCL, tier, and model ID | operational training for deployment |

At the close of Phase D the scaffold decoders can be structurally removed via `remove_scaffold()`; subsequent training and inference operate without them.

### 11.2 Loss per Phase

Let `\mathcal{L}_{\text{ionic}} = \text{norm\_mse}(W_d \mathbf{z}_{ion} + \mathbf{b}_d, \mathbf{y}_{\text{true}})`, `\mathcal{L}_{\text{conc}} = \text{norm\_mse}(\mathbf{z}_{[20:24]}, \mathbf{c}_{\text{true}})`, `\mathcal{L}_{\text{cond}} = \text{norm\_mse}(W_G \mathbf{g} + \mathbf{b}_G, \mathbf{G}_{\text{true}})`, and `\mathcal{L}_{I_{ion}} = \text{mse}(\hat{I}_{ion}, I_{ion}^{\text{true}})`. `norm\_mse` is per-dimension min-max normalization to `[0, 1]` using physiological ranges, so every target dimension contributes equally regardless of physical scale.

Phase A:
$$
\mathcal{L}_A = \mathcal{L}_{\text{ionic}} + \lambda_{\text{vclamp}} \cdot \mathcal{L}_{\text{vclamp}}
$$

Phase B:
$$
\mathcal{L}_B = \mathcal{L}_{\text{ionic}} + \mathcal{L}_{\text{conc}} + \mathcal{L}_{\text{cond}} + \lambda_{\text{vclamp}} \cdot \mathcal{L}_{\text{vclamp}}
$$

Phase C:
$$
\mathcal{L}_C = \mathcal{L}_{I_{ion}}
$$

Phase D:
$$
\mathcal{L}_D = \mathcal{L}_C + (1 - \alpha_t) \cdot \mathcal{L}_B, \qquad \alpha_t: 0 \to 1 \text{ over training}
$$

Phase E: same as Phase D, aggregated over the T1–T12 data curriculum and across ionic models.

Starting values for the Lagrange multipliers are `\lambda_{\text{vclamp}} = 10^{-2}`, subject to ablation. `\mathcal{L}_{\text{rest}}` is subsumed into `\mathcal{L}_{\text{vclamp}}` as the `V = V_{\text{rest}}` grid point.

### 11.3 Integration and Adjoint

Training uses `odeint` from `torchdiffeq` with the Dormand–Prince 5-stage method (`dopri5`) and tolerances `rtol = atol = 10^{-3}`. Backpropagation is through-solver (direct gradient through the solver's internal state) in early training and can be switched to adjoint integration (O(1) memory) once the rate field is sufficiently trained to avoid chaotic-gradient divergence.

AP landmark evaluation times densely sample the upstroke (10 points in the first 5 ms) and sparsely cover the plateau and repolarization (10 points spanning 10–300 ms), for 20 landmarks per training segment. This non-uniform schedule reflects the stiffness of the dynamics: the first 5 ms contains the fastest rates and deserves proportional loss coverage.

### 11.4 Current Training State

All current checkpoints (`multi_bcl_002/best.pt` and its predecessors) were trained on the v3 architecture and are unloadable against v4 because the state dictionary keys change: `ionic_rate_mlp.*` and `conc_kan.*` are replaced by the unified `state_rate_mlp.*`. Phase A must restart from initialization when v4 lands. The compression layers, gate conductance decoder, Stage 2, and all of Phases B through E have never been trained; their weights will remain at initialization until their respective phases begin.

## 12. Known Issues and Open Questions

### 12.1 The Slow-Variable Problem (Session 27 Diagnostic)

The v3 architecture, trained to `val_loss = 0.0084`, exhibited three coupled failure modes, all visible on a held-out BCL=2000 trajectory: a 27% normalized-RMSE error on `Ca_{SR}` against a physiological range of 4.5 mM, an order-of-magnitude over-oscillation of `RR` and `Ca_{SR}` during diastolic intervals, and a ~0.70 normalized-RMSE error during the upstroke dominated by initial-state mismatch at `t = 0`. The v4 design is a direct response: latent expansion creates capacity for slow-variable tracking, the rest-bias freeze resolves the initial-state mismatch, and the voltage-clamp attractors damp spurious AP-frequency content in slow dimensions by enforcing the correct asymptotic fixed points. Whether these changes close the slow-variable gap is an empirical question that will be answered in the first Phase A run on v4.

### 12.2 Integrator Choice at Inference

The Session 27 integrator diagnostic measured forward-Euler inference against a tight-tolerance `dopri5` on the same trained model, finding a 215× advantage for `dopri5` in truncation but a 0.002 RMSE against ground truth compared to `dopri5`'s 0.346 — demonstrating that integrator truncation is negligible relative to model-capacity error at native `\Delta t = 0.01 \text{ ms}`. V4 retains forward Euler at inference; if future higher-`\Delta t` deployment is needed, the integrator choice should be revisited but the rate field itself is integrator-agnostic.

### 12.3 Hybrid Explicit-Slow-Variable Representation

Promoting `Ca_{SR}` and ORd's slow reservoirs (`jSR`, `nSR`, `CaMKt`) out of the latent into an explicit named set was analyzed in Session 27 as an alternative route to the slow-variable problem. It was deferred in favor of first attempting the attractor regularizers on the pure-latent design. If `\mathcal{L}_{\text{vclamp}}` does not close the CaSR tracking gap, the hybrid is the next intervention to try; it is architecturally compatible with v4 and requires adding one extra named concentration dimension.

### 12.4 Neural Rush-Larsen Rejection

A more aggressive alternative — hardcoding the latent to the TTP06 Hodgkin–Huxley gate variables and eliminating the scaffold — was analyzed and rejected (IDEALOG Session 27 Q10). It violates the multi-model goal because TTP06 and ORd have fundamentally different state sets, it commits to HH gating as ground truth in violation of the Layer 0 maxim, and it precludes transfer to optical mapping data that contains no gate observables. V4 does not take this path.

### 12.5 Dropout

Dropout inside the rate predictor was considered and rejected (Session 27). Dropout corrupts the vector field with Bernoulli noise at every `odeint` call, making the adjoint backward pass unstable and the learned dynamics stochastic rather than deterministic. Physical ionic dynamics are deterministic functions of state and voltage; dropout is the wrong regularizer for this class of model. Weight decay (`1 \times 10^{-4}` in `AdamW`) is the retained regularizer. If overfitting between training and validation appears, dropout may be added to the scaffold decoders only — training-only modules with no inference footprint.

## 13. Comparison with Prior Versions

| | v2 (2026-03-23) | v3 (2026-04-01 → 04-07) | v4 (this document) |
|---|---|---|---|
| Ionic latent dim | 20 (shared) | 16 | **20** |
| Rate path | attention + GELU + Chebyshev | VoltageAttention + MLP (ionic) + KAN (conc) | **unified StateRateMLP, 5 GELU + LN + gated skip** |
| Cross-product capacity | limited | single KAN: additively universal, mult-blind | **compositional via depth** |
| Decoder bias | free | free | **frozen at rest** |
| Physics-informed attractors | — | — | **rest + voltage-clamp SS** |
| Concentration self-input to rate | — | dropped | **restored** |
| Inference params | 948 | 2,124 | **7,622** |
| Training params | 1,008 | 2,407 | **7,961** |
| Multi-model provision | not analyzed | model-ID conditioning | **per-model rest + z_ss grid** |
| Validated training phases | none | A only | **none; Phase A restarts on v4** |

The parameter growth from v3 to v4 is 3.6×, attributable almost entirely to compositional depth: the Stage 1 dzdt network grew from 1,444 parameters (IonicRateMLP + conc_kan) to 6,792 parameters (five-layer MLP + LN + skip). The remaining architecture is minimally changed.

## 14. References

### 14.1 Internal

- `Research/Active/surrogate_pipeline/IDEALOG.md`, Session 27 (2026-04-19). Full design discussion for v4, questions Q1–Q12 including integrator error budget, decoder interpretation, hard-coding analysis, and attractor selection.
- `Research/Active/surrogate_pipeline/KNOWLEDGE.md`, Section 3b (Stage 1 Pending Pivot) and Section 3c (Model Capacity Diagnostic). Parallel narrative for the v4 pivot and the diagnostic that motivated it.
- `Research/Active/surrogate_pipeline/archive/architecture/ARCHITECTURE_v3.md`. Superseded. Retained for historical comparison.
- `Research/Active/surrogate_pipeline/archive/architecture/ARCHITECTURE_v2.md`. Superseded. Retained for historical comparison.
- `Surrogate/diagnostics/integrator_error_budget.py` and its artifact `Surrogate/diagnostics/artifacts/integrator_error_budget.pt`. Reproducible Session 27 diagnostic script.
- `Surrogate/TRAINING_STRATEGY.md`. Training phase details (will be updated in parallel with v4 implementation).

### 14.2 External (informative, not directly instantiated in the architecture)

- Chen, R. T. Q. et al. *Neural Ordinary Differential Equations*, NeurIPS 2018. Foundational adjoint-integration framework used by `odeint`.
- Salvador, M. et al. *Fast and robust prediction of cardiac electromechanics using latent neural ODE*, 2024. Motivating precedent for NODE-based cardiac ionic modeling.
- He, K. et al. *Deep Residual Learning for Image Recognition*, CVPR 2016. Origin of the block-residual design followed in Section 5.3's full-path skip.
- Vaswani, A. et al. *Attention Is All You Need*, NeurIPS 2017. Origin of the pre-LayerNorm convention followed in Section 5.2.
