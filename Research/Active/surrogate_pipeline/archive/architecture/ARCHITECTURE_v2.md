# Ionic Surrogate v2: A Physics-Inspired Neural Ionic Model

## Overview

We present a compact autoregressive neural surrogate for cardiac ionic dynamics that replaces the Rush-Larsen (RL) ordinary differential equation solver at 3.7x its computational cost while learning the full nonlinear gate-current mapping from data. The architecture comprises three stages — cross-attention gate update, split GELU cross-channel coupling, and Chebyshev polynomial current readout — each motivated by a specific biophysical mechanism in the Hodgkin-Huxley formalism. The model uses 642 parameters at inference (948 during training) and requires 886 floating-point operations per forward step.

## 1. Problem Formulation

The TTP06 ventricular ionic model evolves 18 state variables (13 gating variables, 5 ionic concentrations) according to:

$$\frac{dg_i}{dt} = \frac{g_{\infty,i}(V_m) - g_i}{\tau_i(V_m)}, \quad I_{ion} = \sum_j \bar{g}_j \prod_i g_i^{a_{ij}} (V_m - E_j)$$

The Rush-Larsen method discretizes the gate ODE analytically:

$$g_i^{n+1} = g_{\infty,i}(V_m) + (g_i^n - g_{\infty,i}(V_m)) \cdot \exp(-\Delta t / \tau_i(V_m))$$

This can be rewritten in the interpolation form that motivates our Stage 1:

$$g_i^{n+1} = g_i^n + \alpha_i(V_m) \cdot (g_{\infty,i}(V_m) - g_i^n), \quad \alpha_i = 1 - \exp(-\Delta t / \tau_i(V_m)) \in (0,1)$$

The surrogate replaces this three-step process (gate update, current computation, voltage advance) with a learned 16-dimensional latent state $\mathbf{h} \in \mathbb{R}^{16}$ that is updated autoregressively:

$$\mathbf{h}^{n+1}, I_{ion}^{n+1} = f_\theta(\mathbf{h}^n, V_m^n, \Delta t)$$

The membrane voltage is then updated externally via operator splitting:

$$V_m^{n+1} = V_m^n + \Delta t \cdot (-I_{ion} + I_{stim}) / C_m$$

We deliberately exclude $I_{stim}$ from the model input. Ionic gates respond to membrane voltage, not to the external stimulus directly. The stimulus affects $V_m$, which the model receives at the next time step. This separation preserves the operator-splitting structure of the parent PDE solver and ensures the surrogate is agnostic to stimulus protocol.

## 2. Stage 1: Per-Dimension Cross-Attention Gate Update

### 2.1 Motivation

In the Rush-Larsen scheme, each gate updates independently toward a voltage-dependent steady state at a voltage-dependent rate. Two fundamental limitations constrain this approach: (1) the update rate $\alpha_i$ depends only on $V_m$, not on the current gate state, and (2) gates update independently with no coupling between channels.

Limitation (1) is biophysically significant. During the rapid sodium upstroke ($\dot{V}_m > 300$ V/s), the activation gate $m$ must respond on a sub-millisecond timescale, while during the plateau phase the same gate should change minimally. A state-dependent rate would allow the model to allocate computational "attention" to dimensions that are far from equilibrium.

We address (1) with a per-dimension cross-attention mechanism that computes state-dependent update rates. Limitation (2) is deferred to Stage 2.

### 2.2 Architecture

The input voltage and time step are concatenated and projected to produce a shared key $\mathbf{k}$ and value $\mathbf{v}$:

$$\mathbf{x} = [V_m, \Delta t] \in \mathbb{R}^2, \quad \mathbf{k} = W_k \mathbf{x} \in \mathbb{R}^8, \quad \mathbf{v} = W_v \mathbf{x} \in \mathbb{R}^8$$

where $W_k, W_v \in \mathbb{R}^{2 \times 8}$ are bias-free linear projections. The 8-dimensional attention space is shared across all 16 latent dimensions.

Each latent dimension $d$ independently queries the voltage representation:

$$\mathbf{q}_d = h_d \cdot W_q[d, :] \in \mathbb{R}^8$$

where $W_q \in \mathbb{R}^{16 \times 8}$ is a learnable parameter matrix. This per-dimension query is distinct from standard multi-head attention: each latent dimension acts as an independent "head" that scales a learned query direction by its current value. Dimensions far from zero produce large queries; dimensions near zero produce small queries.

The attention score and update rate are:

$$\text{score}_d = \frac{\sum_j q_{d,j} \cdot k_j}{\sqrt{8}}, \quad \alpha_d = \sigma(\text{score}_d) \in (0, 1)$$

The per-dimension target is computed from the value projection:

$$\text{target}_d = \sum_j v_j \cdot W_{out}[j, d]$$

where $W_{out} \in \mathbb{R}^{8 \times 16}$. The latent update follows the Rush-Larsen interpolation form:

$$h_d^{mid} = h_d + \alpha_d \cdot (\text{target}_d - h_d)$$

### 2.3 Design Rationale

**Sigmoid gate, not softmax.** Standard attention uses softmax over sequence positions to produce a probability distribution. Our "sequence" has length 1 (a single voltage token), making softmax degenerate. Instead, the sigmoid produces a per-dimension update rate in $(0,1)$, directly analogous to $1 - \exp(-\Delta t / \tau)$ in Rush-Larsen. This guarantees contraction: $\|h^{mid} - \text{target}\| < \|h - \text{target}\|$ for all non-zero gates.

**Per-dimension query, not linear projection.** A standard linear query $Q = h \cdot W_q$ mixes information across latent dimensions. Our scalar-times-row formulation $q_d = h_d \cdot W_q[d,:]$ preserves dimensional independence — each gate queries the voltage autonomously, mirroring the biophysical independence of Hodgkin-Huxley gates. Cross-dimensional coupling is handled explicitly in Stage 2.

**Shared key-value, per-dimension query.** All 16 latent dimensions observe the same voltage representation ($\mathbf{k}$, $\mathbf{v}$) but attend to it differently via their per-dimension queries. This reflects the biophysics: all gates see the same membrane voltage, but each has its own voltage-dependent kinetics ($g_\infty$, $\tau$).

**8-dimensional attention space.** The choice of 8 dimensions balances expressiveness against parameter cost. The voltage input is 2-dimensional ($V_m$, $\Delta t$), so the key/value projections expand from 2 to 8, providing sufficient capacity to represent nonlinear voltage dependencies (sigmoid-shaped $g_\infty$ curves, bell-shaped $\tau$ curves). The 8-dimensional query-key dot product can represent up to 8 independent selectivity features per latent dimension.

**No layer normalization.** Standard transformers apply LayerNorm before attention to stabilize training across 6–96 stacked layers. Our architecture has a single attention step per time step with no layer stacking. The 1/$\sqrt{8}$ scaling normalizes scores, the sigmoid bounds the gate, and the contractive update prevents latent drift. Layer normalization would remove per-dimension magnitude information that the state-dependent query mechanism deliberately exploits.

### 2.4 Biophysical Correspondence

| Rush-Larsen | Stage 1 |
|---|---|
| $\alpha_i = 1 - e^{-\Delta t / \tau_i(V_m)}$ | $\alpha_d = \sigma(h_d W_q[d,:] \cdot k^\top / \sqrt{8})$ |
| Rate depends on $V_m$ only | Rate depends on $h$, $V_m$, $\Delta t$ |
| $g_{\infty,i}(V_m)$ — fixed function | $\text{target}_d(V_m, \Delta t)$ — learned function |
| Independent per gate | Shared K/V, per-dim Q |
| Contraction guaranteed | Contraction guaranteed |

The key upgrade is state-dependent gating: a latent dimension that is far from its equilibrium target will produce a large query, receive a large attention score, and update aggressively. A dimension near equilibrium produces a small query, receives a moderate score (~0.5), and updates gently. Rush-Larsen fundamentally cannot achieve this — its update rate is a fixed function of voltage regardless of the current gate state.

## 3. Stage 2: Split GELU Cross-Channel Coupling

### 3.1 Motivation

Rush-Larsen treats each gate as an independent ODE. In reality, ionic channels couple through shared ion concentrations, voltage-dependent co-activation, and direct protein interactions. The $m^3hj$ product in the sodium current, the calcium-induced calcium release cascade, and the $I_{Ks}$ slow potassium current all exhibit inter-gate dependencies that independent gating cannot capture.

Stage 2 introduces two rounds of cross-channel mixing that allow information to flow between latent dimensions, enabling the model to learn these coupling patterns from data.

### 3.2 Architecture

Each round splits the 16-dimensional latent into two 8-dimensional halves, applies GELU gating, projects back to 16 dimensions, and adds a residual connection:

$$\mathbf{g} = \text{GELU}(\mathbf{h}_{[:8]}) \odot \mathbf{h}_{[8:]}, \quad \mathbf{h} \leftarrow \mathbf{h} + \text{RMSNorm}(W_{cc} \cdot \mathbf{g} + \mathbf{b})$$

where $W_{cc} \in \mathbb{R}^{8 \times 16}$ is spectrally normalized ($\|W_{cc}\|_2 \leq 1$) and RMSNorm normalizes the correction to consistent scale. This is repeated twice with separate parameters ($W_{cc1}, \mathbf{b}_1$) and ($W_{cc2}, \mathbf{b}_2$).

### 3.3 Design Rationale

**Split GELU gating, not standard feedforward.** The split-and-multiply operation $\text{GELU}(\mathbf{h}_{[:8]}) \odot \mathbf{h}_{[8:]}$ computes pairwise products between two groups of latent dimensions. This is biophysically motivated: ionic currents involve gate products ($m \cdot h$, $m^3 \cdot h \cdot j$, $d \cdot f \cdot f_2 \cdot f_{Ca}$). The GELU activation on the left half allows sign-dependent gating — dimensions near zero are suppressed (GELU(0) = 0), while large positive dimensions pass through (GELU(x) ≈ x for x >> 0). This selectively activates coupling based on gate state.

**Two rounds (×2).** A single round of split GELU mixing captures pairwise interactions between the two halves. Two rounds allow information to propagate across all dimensions: the first round mixes halves A and B, the second round mixes the result — allowing dimension $i$ in half A to influence dimension $j$ in half A through their shared interactions with half B. This provides the minimum depth for full cross-channel communication.

**Spectral normalization on $W_{cc}$.** The spectral norm constraint $\|W_{cc}\|_2 \leq 1$ ensures that the linear projection is a contraction in operator norm. Combined with the residual connection, this means:

$$\|\mathbf{h}^{new}\| \leq \|\mathbf{h}^{old}\| + \|W_{cc}\|_2 \cdot \|\text{RMSNorm}(\mathbf{g})\| + \|\mathbf{b}\|$$

The correction magnitude is bounded regardless of training dynamics. This is critical for autoregressive stability over thousands of time steps.

**RMSNorm on the correction.** The split GELU product is quadratic in the latent magnitude — if latent values grow, the product grows faster. RMSNorm normalizes the correction to consistent RMS before the residual addition, preventing positive feedback loops. Unlike LayerNorm, RMSNorm does not center (remove mean), preserving the correction's ability to shift the latent mean. Zero learnable parameters.

**RMSNorm + spectral norm together.** These address different failure modes. RMSNorm bounds what the operator *sees* (input scale). Spectral norm bounds what the operator *does* (output/input ratio). Together they provide a hard guarantee on correction magnitude: $\|\text{correction}\| \leq \sqrt{16} + \|\mathbf{b}\| \approx 4 + \|\mathbf{b}\|$. Neither can be "learned away" during training.

**Residual connection.** The additive skip connection ensures that when the correction is zero, the latent passes through unchanged. This is essential during early training when weights are near zero — the model starts as an identity mapping plus the Stage 1 contractive update, and gradually learns coupling patterns as training progresses.

### 3.4 Biophysical Correspondence

The split GELU cross-channel mixer has no direct Rush-Larsen equivalent — RL gates update independently. Stage 2 captures the inter-gate coupling that RL entirely neglects:

- $m \cdot h$ (sodium activation-inactivation product)
- $d \cdot f \cdot f_2 \cdot f_{Ca}$ (L-type calcium multi-gate product)
- $Ca^{2+}_i \leftrightarrow Ca^{2+}_{SR}$ (calcium-induced calcium release)
- $[K^+]_i$ dependence of $I_{Ks}$ slow potassium current

These couplings are the primary source of complex cardiac dynamics (action potential morphology, restitution, alternans) that independent gate models struggle to reproduce accurately.

## 4. Stage 3: KAN Chebyshev Current Readout

### 4.1 Motivation

The ionic current in Hodgkin-Huxley models is a sum of channel currents, each involving a conductance (product of gating variables raised to integer powers) multiplied by a driving force (voltage minus reversal potential):

$$I_{ion} = \sum_j \bar{g}_j \cdot \prod_i g_i^{a_{ij}} \cdot (V_m - E_j)$$

This is a polynomial function of the gate states with a linear dependence on $V_m$. A standard linear readout $I_{ion} = \mathbf{w}^\top \mathbf{h} + b$ cannot capture the nonlinear gate products. A multilayer perceptron could approximate them but adds unnecessary parameters and obscures the structure.

We adopt a Kolmogorov-Arnold Network (KAN) inspired readout using Chebyshev polynomial basis functions, which can represent arbitrary nonlinear functions of each latent dimension while maintaining interpretability and parameter efficiency.

### 4.2 Architecture

Each latent dimension is independently mapped through a degree-$K$ Chebyshev polynomial, then summed with a direct voltage term:

$$I_{ion} = \sum_{d=1}^{16} \varphi_d(h_d) + b_{vm} \cdot V_m + b$$

where $\varphi_d(h_d) = \sum_{k=0}^{K} C[d, k] \cdot T_k(\tilde{h}_d)$ and $T_k$ are Chebyshev polynomials of the first kind computed via the three-term recurrence:

$$T_0(x) = 1, \quad T_1(x) = x, \quad T_n(x) = 2x \cdot T_{n-1}(x) - T_{n-2}(x)$$

The normalized input $\tilde{h}_d = 2(h_d - z_{min,d}) / (z_{max,d} - z_{min,d}) - 1$ maps each latent dimension to $[-1, 1]$, where Chebyshev polynomials are orthogonal and bounded ($|T_k(x)| \leq 1$). The bounds $z_{min}$, $z_{max}$ are registered buffers set from Phase A training statistics.

### 4.3 Design Rationale

**Chebyshev basis, degree K=3.** The $m^3hj$ sodium gate product involves cubic terms — the highest polynomial degree in the TTP06 current equations. Chebyshev degree 3 can represent up to cubic nonlinearities per dimension: $T_0 = 1$ (constant), $T_1 = x$ (linear), $T_2 = 2x^2 - 1$ (quadratic), $T_3 = 4x^3 - 3x$ (cubic). This matches the biophysical complexity without overfitting.

**Per-dimension polynomials (KAN structure).** In a standard MLP, all input dimensions interact through weight matrices. In our KAN-style readout, each dimension has its own polynomial — the 16-dimensional function decomposes as a sum of 16 univariate functions. This mirrors the Hodgkin-Huxley current structure where each channel contributes independently to the total current. Cross-dimensional products (e.g., $m^3 \cdot h$) are captured by Stage 2's cross-channel mixing, which embeds these products into the latent representation before Stage 3 reads them out.

**Direct voltage term $b_{vm} \cdot V_m$.** Every Hodgkin-Huxley channel current contains a linear driving force term $(V_m - E_{rev})$. The direct voltage pathway allows the model to represent this without consuming Chebyshev capacity. The learned $b_{vm}$ aggregates the net voltage sensitivity across all channels.

**Zero initialization of $C$.** The Chebyshev coefficient matrix $C$ is initialized to zero, so the model starts as $I_{ion} = b_{vm} \cdot V_m + b$ (a simple linear voltage-current relationship). Nonlinear gate-current dependencies are learned gradually during training. This prevents early-training instability from large random $I_{ion}$ predictions.

**Normalization to $[-1,1]$.** Chebyshev polynomials diverge rapidly outside $[-1, 1]$: $T_3(2) = 26$, $T_3(5) = 485$. The per-dimension normalization with clamping ensures numerical stability. The bounds are set from Phase A training data statistics and frozen — this is a data-dependent preprocessing step, not a learned operation.

### 4.4 Biophysical Correspondence

| Rush-Larsen | Stage 3 |
|---|---|
| $I_{ion} = \sum_j \bar{g}_j \prod_i g_i^{a_{ij}} (V_m - E_j)$ | $I_{ion} = \sum_d \varphi_d(h_d) + b_{vm} V_m + b$ |
| Gate products: $m^3 h j$ | Chebyshev cubics: $T_3(\tilde{h}_d)$ |
| Driving force: $(V_m - E_{rev})$ | Direct voltage: $b_{vm} V_m$ |
| Fixed conductances $\bar{g}$ | Learned coefficients $C[d,k]$ |
| 15 explicit currents | 16 per-dim polynomials (summed) |

## 5. Scaffold Decoder (Training Only)

During Phase A training, a linear decoder maps the latent state to the 18 TTP06 gate values:

$$\hat{\mathbf{g}} = \sigma(W_{dec} \cdot \mathbf{h} + \mathbf{b}_{dec}), \quad W_{dec} \in \mathbb{R}^{16 \times 18}$$

This auxiliary loss ensures the latent representation is biophysically meaningful — each dimension must encode information relevant to gate prediction, not arbitrary features. The decoder adds 306 parameters (288 weights + 18 biases) that are removed after training via `remove_scaffold()`.

The sigmoid activation bounds gate predictions to $(0, 1)$, matching the biological range of gating variables.

## 6. Normalization Strategy

A central design question for autoregressive models is preventing activation drift over long rollouts (1000+ time steps). We considered and rejected several approaches:

**Sigmoid output bounding** was rejected because: (1) sigmoid saturation causes vanishing gradients during long-rollout training, (2) the triple sigmoid path (Stage 1 gate → output → scaffold) compounds gradient compression, (3) sigmoid breaks the residual identity — $\sigma(x + 0) \neq x$, and (4) not all biophysical quantities are bounded to $[0, 1]$ (ionic concentrations span orders of magnitude).

**LayerNorm** was rejected because: (1) centering (mean removal) prevents the Stage 2 correction from shifting the latent mean, which is a legitimate operation, and (2) the learnable affine parameters ($\gamma$, $\beta$) can reintroduce arbitrary scale during training, undermining the normalization.

**BatchNorm** was rejected because: (1) our inference is per-node autoregressive (batch=1), making batch statistics meaningless, and (2) the train/eval discrepancy from running statistics creates inconsistency during long rollouts.

The adopted strategy uses **architectural stability** — each stage has its own normalization mechanism matched to its specific risk:

| Stage | Risk | Mechanism | Guarantee |
|---|---|---|---|
| 1 | Latent drift | Contractive update ($\sigma$ gate) | $\|\mathbf{h}^{mid} - \text{target}\| < \|\mathbf{h} - \text{target}\|$ |
| 2 | Quadratic blowup | RMSNorm + spectral norm | $\|\text{correction}\| \leq \sqrt{16} + \|\mathbf{b}\|$ |
| 3 | Polynomial divergence | Chebyshev normalization + clamp | $\tilde{h}_d \in [-1, 1]$ |

No activation bounding (sigmoid/tanh/clamp) is applied to the latent state. Stability emerges from the architecture itself.

## 7. Parameter Budget

| Component | Parameters (Inference) | Parameters (Training) | FLOPs |
|---|---|---|---|
| $W_q$ (16×8) | 128 | 128 | — |
| $W_k$ (2×8) | 16 | 16 | — |
| $W_v$ (2×8) | 16 | 16 | — |
| $W_{out}$ (8×16) | 128 | 128 | — |
| **Stage 1 total** | **288** | **288** | **464** |
| $W_{cc1}$ (8×16) + $\mathbf{b}_1$ | 144 | 144 | — |
| $W_{cc2}$ (8×16) + $\mathbf{b}_2$ | 144 | 144 | — |
| **Stage 2 total** | **288** | **288** | **352** |
| $C$ (16×4) | 64 | 64 | — |
| $b_{vm}$, $b$ | 2 | 2 | — |
| **Stage 3 total** | **66** | **66** | **70** |
| $W_{dec}$ (16×18) + $\mathbf{b}_{dec}$ | — | 306 | — |
| **Scaffold** | **—** | **306** | **324** |
| **Total** | **642** | **948** | **886 (1210)** |

At 886 FLOPs per forward step, the ionic surrogate is 3.7× the cost of a single Rush-Larsen step (240 FLOPs). For a tissue simulation with 10,000+ nodes, the per-node overhead is amortized by GPU parallelism and the elimination of lookup table memory access patterns that limit Rush-Larsen throughput.
