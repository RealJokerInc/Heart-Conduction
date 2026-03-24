# PLAN: Ionic Surrogate nn.Module Implementation

Created: 2026-03-24
Engine(s): None (standalone ML model)
Research question: [surrogate_pipeline](README.md)
Source: [IDEALOG.md](IDEALOG.md) — Session 12 (Architecture v2 settled)

## Objective
Implement the ionic surrogate model v2 as a `torch.nn.Module`: ChebyshevReadout layer, IonicSurrogate (3 stages + scaffold decoder). Verify parameter counts, shapes, contractivity, spectral norm, and gradient flow. Model code only — no data loading, no training.

## Success Criteria
- [ ] IonicSurrogate matches spec: 642 inference params, 948 training params
- [ ] Spectral norms on W_cc1, W_cc2 ≤ 1.0
- [ ] Stage 1 contractivity verified
- [ ] Gradient flow through 10-step rollout without NaN/Inf
- [ ] All 18 new tests pass, all existing 32 data generation tests pass

## Architecture Changes
- NEW: `Surrogate/surrogate/model/__init__.py` — model subpackage
- NEW: `Surrogate/surrogate/model/chebyshev.py` — Chebyshev polynomial readout layer
- NEW: `Surrogate/surrogate/model/ionic_surrogate.py` — full 3-stage model + scaffold
- NEW: `Surrogate/tests/test_model.py` — unit tests

## Known Failures (from IDEALOG)
- Temporal Transformer — 1M× RL
- Vm history buffer — stimulus artifacts
- Learned Rush-Larsen — too constrained
- Neural ODE — too unconstrained
- GRU cell — unnecessary gating overhead
- 17×17 self-attention — 47× RL overkill
- Deep MLP cross-channel — overkill
- I_stim as model input — biophysically wrong

---

## Phase 1: ChebyshevReadout Layer

**Goal**: Self-contained per-dim Chebyshev polynomial readout. Tested in isolation before integration.
**Tier**: small
**Estimated scope**: 1 file (~80 lines), 5 tests

### Phase Context
- KNOWLEDGE.md:45-52 — KAN Chebyshev K=3 spec
- Chebyshev recurrence: T₀=1, T₁=z̃, Tₙ=2z̃Tₙ₋₁-Tₙ₋₂
- z̃ normalized to [-1,1] via registered buffers z_min, z_max (set later from Phase A stats)
- Default bounds [-1,1] (identity normalization) until set_bounds() called
- C initialized to zeros → model starts as I_ion = b_vm*Vm + b
- float32 throughout

### Step 1.1: Implement ChebyshevReadout
**Model**: opus

#### Read First
- KNOWLEDGE.md:45-52 — KAN Chebyshev spec
- KNOWLEDGE.md:45-52 — Stage 3 KAN Chebyshev spec (WHITEBOARD.md is ephemeral, use KNOWLEDGE as authoritative)

#### Why
Isolated component — the recurrence relation and normalization are easy to get wrong (off-by-one in degree, wrong broadcast shape for C×T product). Testing independently catches these before they hide inside the full model.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/model/__init__.py`, `Surrogate/surrogate/model/chebyshev.py`

```python
class ChebyshevReadout(nn.Module):
    def __init__(self, n_dims: int = 16, degree: int = 3):
        # C: Parameter (n_dims, degree+1) — init zeros
        # b_vm: Parameter (1,) — init zero
        # b: Parameter (1,) — init zero
        # z_min: buffer (n_dims,) — default -1
        # z_max: buffer (n_dims,) — default +1

    def set_bounds(self, z_min: Tensor, z_max: Tensor):
        """Update normalization buffers."""

    def forward(self, z: Tensor, Vm: Tensor) -> Tensor:
        """z: (batch, n_dims), Vm: (batch,) → I_ion: (batch,)"""
```

#### Pseudocode
```
forward(z, Vm):
    eps = 1e-8
    z_norm = 2 * (z - self.z_min) / (self.z_max - self.z_min + eps) - 1
    z_norm = z_norm.clamp(-1, 1)

    # Chebyshev basis: list of (batch, n_dims) tensors
    T = [torch.ones_like(z_norm), z_norm]
    for j in range(2, self.degree + 1):
        T.append(2 * z_norm * T[-1] - T[-2])
    T = torch.stack(T, dim=-1)          # (batch, n_dims, degree+1)

    phi = (self.C * T).sum(dim=-1)      # (batch, n_dims)
    I_ion = phi.sum(dim=-1) + self.b_vm * Vm + self.b  # (batch,)
    return I_ion
```

#### Test Spec
- `test_chebyshev_shape` — Explicitly construct `ChebyshevReadout(n_dims=16, degree=3)`. Input (32, 16), Vm (32,). Output: (32,). Assert output dtype is float32.
- `test_chebyshev_params` — Construct `ChebyshevReadout(n_dims=16, degree=3)`. Verify: 16*(3+1) + 1 + 1 = 66 trainable params.
- `test_chebyshev_zero_init` — C=0 → I_ion = b_vm*Vm + b (= 0 at init)
- `test_chebyshev_linear_recovery` — Set C[:,0]=w, rest=0. Since T₀=1, output = Σw_k + b_vm*Vm + b. Verify numerically.
- `test_chebyshev_cubic` — 1 dim, C=[0,0,0,1]. At z̃=0.5: T₃(0.5)=4(0.125)-1.5=-1.0. Verify.
- `test_chebyshev_set_bounds` — Call set_bounds(z_min, z_max), verify buffers updated, normalization changes output.
- `test_chebyshev_constant_dim` — Set z_min[0]=z_max[0]=0.5 (constant dim). Verify no NaN/Inf — eps in denominator + clamp should produce valid output.
- `test_chebyshev_out_of_bounds` — Input z values outside [z_min, z_max]. Verify output is valid (clamp to [-1,1] prevents divergence).

#### Checklist
- [ ] Create `surrogate/model/__init__.py` (empty or with imports)
- [ ] Implement ChebyshevReadout
- [ ] Register z_min, z_max as buffers
- [ ] eps in denominator, safety clamp
- [ ] Zero init C, b_vm, b
- [ ] Write 8 tests in `tests/test_model.py` (5 original + set_bounds + constant_dim + out_of_bounds)

#### Verify
```bash
cd Surrogate && conda run -n heart-conduction python -m pytest tests/test_model.py -v -k chebyshev
```

#### Exit Criteria
- [ ] 8/8 chebyshev tests pass
- [ ] 66 params confirmed

#### Risk
Broadcast shape mismatch in `self.C * T` — C is (n_dims, K+1), T is (batch, n_dims, K+1). Need C.unsqueeze(0) or rely on broadcast. Mitigation: shape assertion in test.

---

### Phase 1 Verification
```bash
cd Surrogate && conda run -n heart-conduction python -m pytest tests/test_model.py -v -k chebyshev
```

### Phase 1 Exit Criteria
- [ ] 8/8 tests pass
- [ ] 66 params

### Phase 1 Cleanup
- [ ] float32 consistency
- [ ] Docstring with parameter table

**-> Commit point: git commit after Phase 1 passes**

---

## Phase 2: IonicSurrogate nn.Module

**Goal**: Full 3-stage model + scaffold decoder. All architectural properties verified.
**Tier**: medium
**Estimated scope**: 1 file (~200 lines), 9 tests

### Phase Context
- KNOWLEDGE.md:17-54 — Architecture spec
- KNOWLEDGE.md:106-118 — Parameter table
- Stage 1: n×1 cross-attention. W_q is (16,8) Parameter, NOT nn.Linear. Per-dim query: `latent[k] * W_q[k,:]`. W_k, W_v are nn.Linear(2,8,bias=False). W_out is (8,16) Parameter.
- Stage 2: two rounds of split GELU. cc1, cc2 are nn.Linear(8,16) wrapped in spectral_norm. Residual connections.
- Stage 3: ChebyshevReadout from Phase 1.
- Scaffold: nn.Linear(16,18) + sigmoid. Optional (scaffold=True default).
- Handle batched (B,16) and unbatched (16,) inputs transparently.
- scale = 1/√8 stored as constant.
- Init: Xavier uniform for W_q, W_k, W_v, W_out. **CRITICAL: init weights BEFORE applying spectral_norm** (spectral_norm replaces `.weight` with a computed property from `.weight_orig`; after wrapping, must use `.weight_orig` not `.weight`). Zeros for readout. Random for scaffold (will be overwritten by Phase A).
- Store `self.split = split` in __init__ for use in forward().
- **Do NOT modify top-level `surrogate/__init__.py`** — only update `surrogate/model/__init__.py`. Keeps model imports isolated from data imports, prevents import cascade breaking existing 32 tests.

### Step 2.1: Implement IonicSurrogate
**Model**: opus

#### Read First
- KNOWLEDGE.md:17-54 — all 3 stages + scaffold
- KNOWLEDGE.md:106-118 — parameter table
- `surrogate/model/chebyshev.py` from Phase 1

#### Why
Core model. The per-dim query broadcast in Stage 1 is the trickiest part — it's NOT a standard linear layer. The shape flow through 3 stages must be verified carefully. Spectral norm must be applied at construction (not post-hoc).

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/model/ionic_surrogate.py`

```python
class IonicSurrogate(nn.Module):
    def __init__(self, latent_dim=16, attn_dim=8, cheby_degree=3,
                 split=8, n_gates=18, scaffold=True):

    def forward(self, latent_prev, Vm, dt):
        """(batch,16) + (batch,) + (batch,) → (batch,16), (batch,), (batch,18)|None"""

    def inference_param_count(self) -> int:
        """Params excluding scaffold."""

    def remove_scaffold(self):
        """Delete decoder attribute for production."""
```

#### Pseudocode
```
__init__:
    self.split = split

    # Stage 1
    W_q = Parameter(empty(latent_dim, attn_dim))       # (16, 8)
    W_k = Linear(2, attn_dim, bias=False)               # 2→8
    W_v = Linear(2, attn_dim, bias=False)               # 2→8
    W_out = Parameter(empty(attn_dim, latent_dim))      # (8, 16)
    scale = 1 / sqrt(attn_dim)

    # Stage 2 — CRITICAL: init weights BEFORE spectral_norm wrapping
    _cc1 = Linear(split, latent_dim)                     # 8→16 + bias
    _cc2 = Linear(split, latent_dim)                     # 8→16 + bias
    xavier_uniform_(_cc1.weight)
    xavier_uniform_(_cc2.weight)
    cc1 = spectral_norm(_cc1)                            # wraps .weight → .weight_orig
    cc2 = spectral_norm(_cc2)

    # Stage 3
    readout = ChebyshevReadout(latent_dim, cheby_degree)

    # Scaffold (optional)
    if scaffold: decoder = Linear(latent_dim, n_gates)  # 16→18 + bias

    _init_weights()

_init_weights:
    xavier_uniform_(W_q)
    xavier_uniform_(W_k.weight)
    xavier_uniform_(W_v.weight)
    xavier_uniform_(W_out)
    # cc1, cc2: already initialized BEFORE spectral_norm wrapping (see above)
    # readout: already zero-init in ChebyshevReadout
    # decoder: default init is fine (will be overwritten by Phase A)

forward(latent_prev, Vm, dt):
    # Handle unbatched
    squeezed = latent_prev.dim() == 1
    if squeezed:
        latent_prev = latent_prev.unsqueeze(0)
        Vm = Vm.view(1)
        dt = dt.view(1)

    # Stage 1: n×1 cross-attention
    x = torch.stack([Vm, dt], dim=-1)             # (B, 2)
    k = self.W_k(x)                                # (B, 8)
    v = self.W_v(x)                                # (B, 8)
    q = latent_prev.unsqueeze(-1) * self.W_q       # (B, 16, 1) * (16, 8) → (B, 16, 8)
    score = (q * k.unsqueeze(1)).sum(-1) * self.scale  # (B, 16)
    gate = torch.sigmoid(score)                     # (B, 16)
    target = v @ self.W_out                         # (B, 16)
    latent_mid = latent_prev + gate * (target - latent_prev)

    # Stage 2: two-round split GELU
    s = self.split
    g1 = F.gelu(latent_mid[:, :s]) * latent_mid[:, s:]
    latent_a = latent_mid + self.cc1(g1)
    g2 = F.gelu(latent_a[:, :s]) * latent_a[:, s:]
    latent_new = latent_a + self.cc2(g2)

    # Stage 3
    I_ion = self.readout(latent_new, Vm)

    # Scaffold
    gates_pred = None
    if hasattr(self, 'decoder'):
        gates_pred = torch.sigmoid(self.decoder(latent_new))

    # Unsqueeze if needed
    if squeezed:
        latent_new, I_ion = latent_new.squeeze(0), I_ion.squeeze(0)
        if gates_pred is not None:
            gates_pred = gates_pred.squeeze(0)

    return latent_new, I_ion, gates_pred
```

#### Test Spec
- `test_surrogate_shapes` — (32,16)+(32,)+(32,) → (32,16), (32,), (32,18). Assert all output dtypes are float32.
- `test_surrogate_single` — (16,)+scalar+scalar → (16,), scalar, (18,)
- `test_surrogate_param_count_inference` — Construct with default args. Compute expected: latent_dim*attn_dim + 2*2*attn_dim + attn_dim*latent_dim + 2*(split*latent_dim + latent_dim) + latent_dim*(degree+1) + 2 = 642. Derive from constructor args, not hardcoded.
- `test_surrogate_param_count_training` — inference + latent_dim*n_gates + n_gates = 948. Derive from constructor args.
- `test_surrogate_stage1_contractivity` — Isolate Stage 1 ONLY: manually compute latent_mid = prev + gate*(target - prev). Verify ||mid - target|| < ||prev - target|| for 100 random samples. Name clearly indicates Stage 1 only. NOTE: full-model contractivity (with Stage 2) is a known open risk — NOT tested here, monitored during training.
- `test_surrogate_spectral_norm` — Verify σ_max(cc1.weight_orig) and σ_max(cc2.weight_orig) are renormalized. After forward pass, check `torch.linalg.matrix_norm(cc1.weight, ord=2) ≤ 1.0 + 1e-6`.
- `test_surrogate_no_scaffold` — scaffold=False → gates is None, inference_param_count()=642
- `test_surrogate_remove_scaffold` — remove_scaffold() drops decoder, param count changes. Call twice → second call is idempotent (no error).
- `test_surrogate_gradient_flow` — 10-step autoregressive rollout, loss.backward() completes, no NaN in grads
- `test_surrogate_no_import_cascade` — `from surrogate.data.storage import TraceStorage` still works after model package exists. Verifies top-level `surrogate/__init__.py` not broken.

#### Checklist
- [ ] Implement IonicSurrogate.__init__ with all weight matrices
- [ ] Implement forward with correct broadcast for per-dim query
- [ ] spectral_norm on cc1, cc2 at construction
- [ ] Xavier init on all non-readout weights
- [ ] Handle batched/unbatched transparently
- [ ] Implement inference_param_count()
- [ ] Implement remove_scaffold()
- [ ] Update `surrogate/model/__init__.py` with exports (do NOT modify top-level `surrogate/__init__.py`)
- [ ] Write 10 tests

#### Verify
```bash
cd Surrogate && conda run -n heart-conduction python -m pytest tests/test_model.py -v
cd Surrogate && conda run -n heart-conduction python -m pytest tests/test_data_generation.py -v  # regression
```

#### Exit Criteria
- [ ] 18/18 tests pass (8 chebyshev + 10 surrogate)
- [ ] 32/32 existing tests pass (no regression)
- [ ] Forward pass works on both CPU and CUDA

#### Risk
Per-dim query broadcast: `latent_prev.unsqueeze(-1) * self.W_q` — shape (B,16,1)*(16,8). PyTorch broadcasts (B,16,1)*(16,8) → need to verify this broadcasts as (B,16,8) not (B,16,16,8). Mitigation: explicit shape assertion in test. If broadcast fails, use `einsum('bd,dh->bdh', latent_prev, W_q)`.

---

### Phase 2 Verification
```bash
cd Surrogate && conda run -n heart-conduction python -m pytest tests/test_model.py -v
cd Surrogate && conda run -n heart-conduction python -m pytest tests/test_data_generation.py -v
```

### Phase 2 Exit Criteria
- [ ] 18/18 model tests pass
- [ ] 32/32 data gen tests pass
- [ ] Forward pass on CUDA verified
- [ ] 642 inference / 948 training params confirmed
- [ ] All outputs float32 verified

### Phase 2 Cleanup
- [ ] float32 consistency
- [ ] Docstrings with parameter table from KNOWLEDGE.md
- [ ] `__init__.py` exports: `IonicSurrogate`, `ChebyshevReadout`

**-> Commit point: git commit after Phase 2 passes**

---

## Final Cleanup
- [ ] float32 consistency — no float64 leaks
- [ ] V5.3 not modified
- [ ] No code duplication
- [ ] Archive plan:
```bash
mkdir -p Research/Active/surrogate_pipeline/plans
cp Research/Active/surrogate_pipeline/PLAN.md "Research/Active/surrogate_pipeline/plans/$(date +%Y-%m-%d)_ionic-surrogate-model-impl.md"
```
- [ ] Revert tmux pane:
```bash
tmux send-keys -t 2 C-c
sleep 0.3
tmux send-keys -t 2 'W=$(tput cols); H=""; while true; do N=$(md5sum Research/Active/surrogate_pipeline/WHITEBOARD.md 2>/dev/null | cut -d" " -f1); if [ "$N" != "$H" ]; then clear; glow -s .glow-style.json -w $W Research/Active/surrogate_pipeline/WHITEBOARD.md 2>/dev/null; H=$N; fi; sleep 1; done' Enter
```

## Mutation Log
(Initially empty — populated during execution)
