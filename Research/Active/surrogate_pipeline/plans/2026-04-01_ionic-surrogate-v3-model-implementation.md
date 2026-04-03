# PLAN: Ionic Surrogate v3 Model Implementation

Created: 2026-04-01
Engine(s): None (standalone ML model, uses Bidomain V1 for data gen)
Research question: [surrogate_pipeline](README.md)
Source: [IDEALOG.md](IDEALOG.md) — Sessions 15-17 (v3 architecture redesign)

## Objective
Implement the v3 ionic surrogate model (2-stage parallel architecture), data preprocessing pipeline, and validate with tests. Replace the v2 model code (3-stage sequential, 642 params) which was superseded before training began. Small TTP06 dims first (ionic=16, cond_latent=8, ~1200 params).

## Success Criteria
- [ ] v3 model code: Stage 1 (attention + MLP + compression) + Stage 2 (cross-attention readout)
- [ ] All new model tests pass (~25 tests)
- [ ] Data preprocessing: TTP06 47-col → v3 format with Nernst + normalization
- [ ] 32 existing data generation tests still pass (no regressions)

## Architecture Changes
- DEL: `Surrogate/surrogate/model/chebyshev.py` — replaced by cross-attention readout
- DEL: `Surrogate/surrogate/model/ionic_surrogate.py` — replaced by v3 model
- NEW: `Surrogate/surrogate/model/stage1.py` — Stage 1: attention + MLP + compression
- NEW: `Surrogate/surrogate/model/stage2.py` — Stage 2: cross-attention readout
- NEW: `Surrogate/surrogate/model/nernst.py` — Nernst equation (fixed physics)
- NEW: `Surrogate/surrogate/model/ionic_surrogate_v3.py` — orchestrator combining Stage 1 + Stage 2
- MOD: `Surrogate/surrogate/model/__init__.py` — update exports
- NEW: `Surrogate/surrogate/data/preprocessor.py` — TTP06 47-col → v3 training format
- MOD: `Surrogate/tests/test_model.py` — full rewrite for v3 tests
- NEW: `Surrogate/tests/test_preprocessing.py` — preprocessor tests

## Known Failures (from IDEALOG)
- Split GELU + W_cc coupling — cross-state coupling is temporal, not instantaneous
- KAN Chebyshev readout on latent — wrong place, compression handles state nonlinearity
- Bilinear readout with hand-crafted features — arbitrary feature vector, overly complex
- Ohmic/non-Ohmic split — Ohm's law is Layer 1 assumption, not Layer 0 truth
- Concentration decoder from ionic state — replaced by explicit dims
- Sigmoid output bounding — vanishing gradients, breaks residual identity
- LayerNorm — removes per-dim magnitude which IS information
- BatchNorm — unstable for autoregressive inference (batch=1)
- Dropout — noise compounds over 100K+ steps
- MLP bottleneck — forces coupling on dims that should stay independent
- Spectral norm — superseded by learned α mixing (convex combination stronger)
- Do NOT modify top-level `Surrogate/surrogate/__init__.py` — breaks 32 data gen tests

---

## Phase 1: v3 Model Core

**Goal**: Implement Stage 1, Stage 2, Nernst, and orchestrator. All model code.
**Tier**: medium
**Estimated scope**: 5 files, ~500 lines, ~25 tests

### Phase Context
Working directory: `Surrogate/` (tests run with `cd Surrogate && conda run -n heart-conduction python -m pytest tests/`). Pure PyTorch. W_q is a raw nn.Parameter — per-dim query requires manual broadcasting (pattern from v2). Keep v2 files until Phase 3 cleanup. All dims are hyperparameters defaulting to small TTP06 config.

### Hyperparameters (small TTP06 config — all tunable knobs defined here)

```python
# === DIMENSIONS ===
IONIC_DIM     = 16    # latent ionic state dims (KNOWLEDGE full: 32)
CONC_DIM      = 4     # explicit concentration dims [Na_i, K_i, Ca_i, Ca_ss]
CARRIED_DIM   = 20    # IONIC_DIM + CONC_DIM (KNOWLEDGE full: 36)
COND_DIM      = 8     # conductance latent after compression (KNOWLEDGE full: 16)
N_ENV         = 9     # environment tokens [Vm, 4×E, 4×conc]
N_GATES       = 12    # scaffold target: HH gates only (m,h,j,r,s,d,f,f2,fCass,Xr1,Xr2,Xs)

# === STAGE 1: ATTENTION ===
ATTN_DIM      = 4     # attention projection dimension

# === STAGE 1: MARKOV MLP ===
MLP_HIDDEN    = 16    # Markov MLP hidden dim = IONIC_DIM (no bottleneck, no expansion)
                      # W1: (IONIC_DIM, MLP_HIDDEN), W2: (MLP_HIDDEN, IONIC_DIM)

# === STAGE 1: COMPRESSION ===
COMP_H1       = 12    # first hidden layer: IONIC_DIM → COMP_H1
COMP_H2       = 12    # second hidden layer: COMP_H1 → COMP_H2
                      # output: COMP_H2 → COND_DIM

# === STAGE 2: CROSS-ATTENTION ===
STAGE2_ATTN   = 4     # attention dim for Q/K embeddings
STAGE2_DV     = 1     # value dim (scalar values)
STAGE2_MLP_H  = 4     # output MLP hidden dim: COND_DIM → STAGE2_MLP_H → 1

# === TRAINING ===
DTYPE         = torch.float32  # ML standard. Data gen uses float64, converted in preprocessing.
ALPHA_INIT    = -5.0  # sigmoid(-5) ≈ 0.007, near-pure residual at init
BETA_INIT     = -5.0  # same for compression mixing
```

**Dimension mapping (small → full)**: KNOWLEDGE specifies the ORd-ready full config. This plan uses small TTP06 dims. To scale up: IONIC_DIM=32, COND_DIM=16, MLP_HIDDEN=32, COMP_H1=24, COMP_H2=24. Same architecture, same code, just bigger numbers.

**Scaffold decoder targets**: 12 HH gates (m, h, j, r, s, d, f, f2, fCass, Xr1, Xr2, Xs). NOT 13 — RR is a RyR release fraction with no gate_inf/tau, not an HH gate. NOT 18 — concentrations have their own direct MSE loss, not decoded from ionic_state. The 12 gates match TTP06's `gate_indices` (12 entries) and the 12 gate_inf/12 gate_tau columns in training data.

### Step 1.1: Nernst module
**Model**: sonnet

#### Read First
- `Bidomain/Engine_V1/cardiac_sim/ionic/ttp06/currents.py:33-54` — Nernst formulas

#### Why
Fixed physics module, zero learned params. Separate because it's pure physics. Needed by orchestrator for both E computation and environment normalization.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/model/nernst.py`
**Interfaces:**
```python
class NernstComputer(nn.Module):
    # Constants: RTONF = R*T/F = 8314.472*310.0/96485.3415 (~26.713 mV). Compute, don't hardcode.
    def forward(self, Na_i, K_i, Ca_i) -> (E_Na, E_K, E_Ca, E_Ks)
    # Note: param order is [Na_i, K_i, Ca_i] matching v3 concentration ordering.
    # TTP06 source uses [Ki, Nai] order — do NOT copy that convention.
    def normalize_environment(Vm, E_Na, E_K, E_Ca, E_Ks, Na_i, K_i, Ca_i, Ca_ss) -> (B, 9)
```
Normalization: 9 fixed shift/scale constants from physiological ranges. Registered as buffers.

#### Pseudocode
```
E_Na = RTONF * log(140 / Na_i.clamp(min=1e-12))
E_K  = RTONF * log(5.4 / K_i.clamp(min=1e-12))
E_Ca = 0.5 * RTONF * log(2.0 / Ca_i.clamp(min=1e-12))
E_Ks = RTONF * log((5.4 + 0.03*140) / (K_i + 0.03*Na_i).clamp(min=1e-12))
normalize: (x - shift) / scale per token, shift/scale as registered buffers
```

#### Test Spec
- `test_nernst_values` — known concentrations → known E (compare TTP06 currents.py)
- `test_nernst_differentiable` — backward through log, no NaN
- `test_nernst_normalization_range` — physiological inputs → output in [-2, 2]

#### Checklist
- [ ] NernstComputer with constants matching TTP06
- [ ] Clamp before log (eps=1e-12)
- [ ] Normalization buffers for 9 tokens

#### Verify
```bash
cd Surrogate && conda run -n heart-conduction python -m pytest tests/test_model.py::TestNernst -v
```

#### Exit Criteria
- [ ] 3 tests pass
- [ ] E values match TTP06 reference

#### Risk
Log of near-zero Ca_i. Mitigation: clamp to 1e-12.

---

### Step 1.2: Stage 1 — Attention + MLP + Compression
**Model**: opus

#### Read First
- `Surrogate/surrogate/model/ionic_surrogate.py:54-174` — v2 attention pattern (W_q raw Parameter, scale, squeeze)
- `Research/Active/surrogate_pipeline/KNOWLEDGE.md` §3 "v3 Stage 1" — attention, MLP, compression, α/β mixing spec

#### Why
Stage 1 is the state evolution engine. Most complex component. Key: attention over all 20 dims (16 ionic + 4 conc), concentrations split off AFTER attention (no MLP), learned α/β mixing for stability.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/model/stage1.py`
**Interfaces:**
```python
class IonicStage1(nn.Module):
    def __init__(self, ionic_dim=IONIC_DIM, conc_dim=CONC_DIM, attn_dim=ATTN_DIM,
                 cond_dim=COND_DIM, mlp_hidden=MLP_HIDDEN,
                 comp_h1=COMP_H1, comp_h2=COMP_H2,
                 n_gates=N_GATES, scaffold=True):
    def forward(self, carried_state, Vm, dt):
        # Returns: (carried_state_new, conductance_latent, concentrations_new,
        #           gates_pred_full, gates_pred_comp)
    def remove_scaffold(self) -> None
    def inference_param_count(self) -> int
```

Architecture: attention(CARRIED_DIM=20, d=ATTN_DIM=4) → split → ionic(IONIC_DIM=16) through Pre-RMSNorm → MLP(IONIC_DIM→MLP_HIDDEN→IONIC_DIM) → α mixing → compression(IONIC_DIM→COMP_H1→COMP_H2→COND_DIM) with β mixing. Concentrations(CONC_DIM=4) pass through directly.

#### Pseudocode
```
# Attention over all dims
q = carried.unsqueeze(-1) * W_q       # (B, 20, 4)
k = W_k([Vm, dt])                     # (B, 4)
score = (q * k.unsqueeze(1)).sum(-1) * (1/√4)
gate = sigmoid(score)                  # (B, 20)
target = W_v([Vm, dt]) @ W_out        # (B, 20)
z_mid = carried + gate * (target - carried)

# Split
ionic_mid = z_mid[:, :16]
conc_new = z_mid[:, 16:]              # DONE for concentrations

# MLP on ionic only (W1: ionic_dim→mlp_hidden, W2: mlp_hidden→ionic_dim)
normed = RMSNorm(ionic_mid)           # inline
correction = W2(GELU(W1(normed)))     # W1: (16→16), W2: (16→16) at small config
alpha = sigmoid(w_alpha)              # (16,) learned, init w=ALPHA_INIT=-5
ionic_new = (1-alpha) * ionic_mid + alpha * correction

# Recombine
carried_new = cat([ionic_new, conc_new])

# Compression (ionic only)
linear_path = W_lin(ionic_new)        # 16→8
h1 = GELU(comp_W1(ionic_new))        # 16→12
h2 = GELU(comp_W2(h1))               # 12→12
nonlinear_path = comp_W3(h2)          # 12→8
beta = sigmoid(w_beta)                # (8,) learned, init w=-5
cond_lat = (1-beta) * linear_path + beta * nonlinear_path
```

#### Test Spec
- `test_stage1_shapes` — batched (32, 20) → correct shapes
- `test_stage1_unbatched` — (20,) → 1D shapes
- `test_stage1_contractivity` — ||z_mid - target|| < ||prev - target||
- `test_stage1_alpha_zero` — w_alpha=-100 → ionic_new ≈ ionic_mid
- `test_stage1_beta_zero` — w_beta=-100 → cond_lat ≈ W_lin @ ionic
- `test_stage1_conc_no_mlp` — concentrations unchanged by MLP modifications
- `test_stage1_param_count` — matches expected
- `test_stage1_remove_scaffold` — idempotent
- `test_stage1_gradient_flow` — 5-step rollout, backward no NaN

#### Checklist
- [ ] W_q as raw nn.Parameter (20, 4)
- [ ] Pre-RMSNorm: inline `x / (x.pow(2).mean(-1, keepdim=True).sqrt() + 1e-8)`
- [ ] w_alpha init to -5.0 (sigmoid≈0.007)
- [ ] w_beta init to -5.0
- [ ] Compression: 2 GELU hidden layers + linear bypass
- [ ] Scaffolds: full(16→12) + compressed(8→12)

#### Verify
```bash
cd Surrogate && conda run -n heart-conduction python -m pytest tests/test_model.py::TestStage1 -v
```

#### Exit Criteria
- [ ] 9 tests pass
- [ ] Contractivity verified
- [ ] α/β extremes verified

#### Risk
Shape mismatch in W_q broadcasting with 20-dim. Mitigation: assert shapes in forward().

---

### Step 1.3: Stage 2 — Cross-Attention Readout
**Model**: opus

#### Read First
- `Research/Active/surrogate_pipeline/KNOWLEDGE.md:215-295` — Stage 2 spec

#### Why
Only component on critical path. Cross-attention (d=4, d_v=1, no softmax) with MLP output (8→4→GELU→1). Input normalization critical (Ca_i vs K_i: 6 orders of magnitude).

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/model/stage2.py`
**Interfaces:**
```python
class IonicStage2(nn.Module):
    def __init__(self, cond_dim=COND_DIM, n_env=N_ENV, attn_dim=STAGE2_ATTN,
                 d_v=STAGE2_DV, mlp_hidden=STAGE2_MLP_H):
    def forward(self, conductance_latent, env_normalized) -> I_ion
```
env_normalized: (B, 9) pre-normalized by NernstComputer.

#### Pseudocode
```
Q = cond_lat.unsqueeze(-1) * e_q      # (B, C, d)
K = env_norm.unsqueeze(-1) * e_k      # (B, 9, d)
V = env_norm.unsqueeze(-1) * e_v      # (B, 9, 1)
scores = bmm(Q, K.T) / √d            # (B, C, 9) no softmax
attended = bmm(scores, V).squeeze(-1) # (B, C)
I_ion = W2(GELU(W1(attended)))        # (B, 1) → (B,)
```

#### Test Spec
- `test_stage2_shapes` — (32, 8) + (32, 9) → (32,)
- `test_stage2_unbatched` — (8,) + (9,) → scalar
- `test_stage2_zero_cond` — zero conductance → I_ion ≈ bias only (MLP bias terms). Use atol for approximate zero, or init MLP bias to zero.
- `test_stage2_param_count` — e_q(8×4) + e_k(9×4) + e_v(9×1) + W1(8×4+4) + W2(4×1+1)
- `test_stage2_gradient_flow` — backward no NaN
- `test_stage2_negative_scores` — verify scores can be negative

#### Checklist
- [ ] e_q, e_k, e_v as nn.Parameter
- [ ] No softmax
- [ ] Scale 1/√d
- [ ] MLP: Linear(C, h) → GELU → Linear(h, 1)

#### Verify
```bash
cd Surrogate && conda run -n heart-conduction python -m pytest tests/test_model.py::TestStage2 -v
```

#### Exit Criteria
- [ ] 6 tests pass
- [ ] Zero conductance → zero current

#### Risk
bmm shape issues with unbatched. Mitigation: squeeze/unsqueeze pattern from v2.

---

### Step 1.4: Orchestrator — IonicSurrogateV3
**Model**: opus

#### Read First
- Steps 1.1-1.3 outputs
- `Research/Active/surrogate_pipeline/KNOWLEDGE.md:22-30` — top-level flow

#### Why
Combines all components. Key: Stage 2 reads OLD state (prev conductance + prev concentrations), not Stage 1 output.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/model/ionic_surrogate_v3.py`
**Interfaces:**
```python
class IonicSurrogateV3(nn.Module):
    def __init__(self, ionic_dim=IONIC_DIM, conc_dim=CONC_DIM,
                 attn_dim=ATTN_DIM, cond_dim=COND_DIM, ...):
    def forward(self, carried_state, Vm, dt, cond_lat_prev, conc_prev) -> dict
    def remove_scaffold(self) -> None
    def inference_param_count(self) -> int
```

#### Pseudocode
```
# Stage 1: produces t+1 outputs
cs_new, cond_new, conc_new, gf, gc = stage1(carried_state, Vm, dt)

# Nernst on PREV concentrations: [Na_i, K_i, Ca_i, Ca_ss]
E_Na, E_K, E_Ca, E_Ks = nernst(conc_prev[:, 0], conc_prev[:, 1], conc_prev[:, 2])
env_norm = nernst.normalize_environment(
    Vm, E_Na, E_K, E_Ca, E_Ks,
    conc_prev[:, 0], conc_prev[:, 1], conc_prev[:, 2], conc_prev[:, 3]
)

# Stage 2: reads PREV conductance + environment
I_ion = stage2(cond_lat_prev, env_norm)

return dict(carried_state=cs_new, conductance_latent=cond_new,
            concentrations=conc_new, I_ion=I_ion, gates_full=gf, gates_comp=gc)
```

#### Test Spec
- `test_v3_shapes` — full forward, all output shapes correct
- `test_v3_autoregressive` — 5-step rollout, pass outputs as next inputs
- `test_v3_stage2_reads_old` — I_ion depends on prev cond, not current
- `test_v3_nernst_uses_prev` — Nernst uses prev concentrations, not new. Change prev conc → I_ion changes. Change new conc → I_ion unchanged.
- `test_v3_remove_scaffold` — idempotent
- `test_v3_param_count` — stage1 + stage2 + nernst(0)
- `test_v3_no_import_cascade` — model import doesn't break data imports

#### Checklist
- [ ] Stage 2 reads PREV state
- [ ] Nernst on PREV concentrations
- [ ] remove_scaffold delegates to stage1
- [ ] Update model/__init__.py (ADD exports, don't remove v2 yet)

#### Verify
```bash
cd Surrogate && conda run -n heart-conduction python -m pytest tests/ -v
```

#### Exit Criteria
- [ ] ~25 new tests pass
- [ ] 32 data gen tests pass
- [ ] 5-step rollout no NaN

#### Risk
Import cascade. Mitigation: only add exports, don't remove v2 until Phase 3.

---

### Phase 1 Verification
```bash
cd Surrogate && conda run -n heart-conduction python -m pytest tests/ -v
```

### Phase 1 Exit Criteria
- [ ] ~25 new model tests pass
- [ ] 32 data gen tests pass
- [ ] v3 model instantiates at small config
- [ ] 5-step autoregressive rollout + backward no NaN

### Phase 1 Cleanup
- [ ] No debug prints
- [ ] Docstrings on public methods
- [ ] Shape assertions in forward()

**→ Commit point: git commit after Phase 1 passes**

---

## Phase 2: Data Preprocessing

**Goal**: Convert TTP06 47-col → v3 format with Nernst, column mapping, normalization.
**Tier**: small
**Estimated scope**: 1 file, ~200 lines, ~5 tests

### Phase Context
Raw data: `/media/norepinephrine/Elements-ext4/surrogate_data/raw/`. TTP06 StateIndex order: [Ki(0), Nai(1), Cai(2), CaSR(3), CaSS(4), m(5)...RR(17)]. Must reorder concentrations to [Na_i, K_i, Ca_i, Ca_ss] = state indices [1, 0, 2, 4].

### Step 2.1: Preprocessor
**Model**: opus

#### Read First
- `Surrogate/surrogate/data/single_cell_generator.py` — TraceData class, N_COLUMNS=47, column constants
- `Bidomain/Engine_V1/cardiac_sim/ionic/ttp06/parameters.py:20-60` — StateIndex (state variable ordering)
- `Surrogate/surrogate/data/storage.py` — TraceStorage (HDF5 I/O) and ShardProcessor (segmentation/sharding)

#### Why
v3 training needs explicit column separation (gates vs concentrations), precomputed Nernst reversal potentials, and normalization statistics. The existing 47-col format mixes gates and concentrations in TTP06's internal order.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/data/preprocessor.py`
**Interfaces:**
```python
class V3Preprocessor:
    # TTP06 StateIndex column mapping (within cols 3-20):
    CONC_REORDER = [1, 0, 2, 4]   # Nai, Ki, Cai, CaSS → [Na_i, K_i, Ca_i, Ca_ss]
    GATE_INDICES = list(range(5, 17))  # m through Xs = 12 HH gates (NOT RR at 17)

    def process_segment(self, raw: Tensor) -> dict:
        """47-col segment → dict of named tensors."""
    def compute_normalization_stats(self, data_dir: str) -> dict:
        """First pass over training data → per-token min/max/mean/std."""
```

#### Pseudocode
```
states = raw[:, 3:21]                    # 18 states
conc = states[:, [1, 0, 2, 4]]          # reorder to [Na_i, K_i, Ca_i, Ca_ss]
gates = states[:, 5:17]                  # 12 HH gates (m through Xs, NOT RR)
E_Na = RTONF * log(140 / conc[:, 0].clamp(min=1e-12))   # Na_i
E_K  = RTONF * log(5.4 / conc[:, 1].clamp(min=1e-12))   # K_i
E_Ca = 0.5 * RTONF * log(2.0 / conc[:, 2].clamp(min=1e-12))
E_Ks = RTONF * log((5.4+0.03*140) / (conc[:,1]+0.03*conc[:,0]).clamp(min=1e-12))
```

#### Test Spec
- `test_preprocessor_column_mapping` — correct concentration reordering [Na_i, K_i, Ca_i, Ca_ss]
- `test_preprocessor_nernst` — Nernst output matches NernstComputer module
- `test_preprocessor_gate_count` — 12 gates (not 13), correct order matching gate_indices
- `test_preprocessor_no_nan` — physiological inputs → no NaN in any output
- `test_preprocessor_normalization` — stats have correct keys and reasonable ranges

#### Checklist
- [ ] Column reordering verified against StateIndex
- [ ] 12 gates (indices 5-16), NOT 13 (exclude RR at 17)
- [ ] Nernst uses same constants as nernst.py module
- [ ] Ca_i clamped to eps before log
- [ ] Normalization stats computation

#### Verify
```bash
cd Surrogate && conda run -n heart-conduction python -m pytest tests/test_preprocessing.py -v
```

#### Exit Criteria
- [ ] 5 tests pass
- [ ] Column mapping verified against TTP06 StateIndex

#### Risk
Wrong column ordering → silent data corruption. Mitigation: test with known TTP06 output where specific gate values can be verified.

### Phase 2 Exit Criteria
- [ ] 5 preprocessor tests pass
- [ ] All previous tests still pass

**→ Commit point: git commit after Phase 2 passes**

---

## Phase 3: Cleanup + v2 Removal

**Goal**: Remove v2 code, update exports.
**Tier**: trivial
**Estimated scope**: delete 2 files, update 1 file

### Step 3.1: Remove v2
**Model**: sonnet

#### Read First
- `Surrogate/surrogate/model/__init__.py` — current exports (v2 + v3 coexist after Phase 1)
- `Surrogate/surrogate/__init__.py` — top-level init (DO NOT MODIFY)

#### Why
v2 code is dead. Remove to avoid confusion. Archive first for reference.

#### Implementation Spec
**Files to delete:** `chebyshev.py`, `ionic_surrogate.py` (after archiving)
**Files to modify:** `model/__init__.py` — remove v2 exports, keep v3 only

#### Pseudocode
```bash
cd Surrogate/surrogate/model
mkdir -p v2_archive
mv chebyshev.py ionic_surrogate.py v2_archive/
# Update __init__.py: remove ChebyshevReadout, IonicSurrogate imports
```

#### Test Spec
- Existing v3 tests + data gen tests all pass (no new tests needed)

#### Checklist
- [ ] Archive v2 files to `v2_archive/`
- [ ] Update `model/__init__.py` — export v3 only
- [ ] Do NOT touch `surrogate/__init__.py`
- [ ] Grep for stale v2 imports
- [ ] Full test suite

#### Verify
```bash
cd Surrogate && conda run -n heart-conduction python -m pytest tests/ -v
grep -r "ChebyshevReadout\|from.*ionic_surrogate import" tests/ surrogate/ --include="*.py" | grep -v v2_archive | grep -v __pycache__
```

#### Exit Criteria
- [ ] All tests pass
- [ ] No v2 references outside archive
- [ ] `from surrogate.model import IonicSurrogateV3` works

#### Risk
Data gen tests might import from old model indirectly. Mitigation: grep before deleting.

### Phase 3 Exit Criteria
- [ ] All tests pass
- [ ] No v2 references outside archive

**→ Commit point: git commit after Phase 3 passes**

---

## Final Cleanup

1. Archive completed plan:
```bash
mkdir -p Research/Active/surrogate_pipeline/plans
cp Research/Active/surrogate_pipeline/PLAN.md "Research/Active/surrogate_pipeline/plans/$(date +%Y-%m-%d)_v3-model-implementation.md"
```

2. Revert bottom tmux pane:
```bash
tmux send-keys -t "0:Surrogate Pipeline.2" C-c
sleep 0.3
tmux send-keys -t "0:Surrogate Pipeline.2" 'W=$(tput cols); H=""; while true; do N=$(md5sum Research/Active/surrogate_pipeline/WHITEBOARD.md 2>/dev/null | cut -d" " -f1); if [ "$N" != "$H" ]; then clear; glow -s .glow-style.json -w $W Research/Active/surrogate_pipeline/WHITEBOARD.md 2>/dev/null; H=$N; fi; sleep 1; done' Enter
```

## Mutation Log
(Empty — populated during execution)
