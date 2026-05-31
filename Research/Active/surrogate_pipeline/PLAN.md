# PLAN: Session 27 — v4 Architecture Pivot (StateRateMLP + Physics-Informed Attractors)

Created: 2026-04-19
Engines: Surrogate (Stage 1 only; Stage 2 unchanged, never trained)
Research question: [Surrogate Pipeline](README.md)
Source: [IDEALOG.md](IDEALOG.md) Session 27 (2026-04-19), Q1–Q12. Full architecture spec: [ARCHITECTURE_v4.md](ARCHITECTURE_v4.md).

## Objective

Replace the v3 ionic + concentration split rate paths (`IonicRateMLP` + `conc_kan`, ~1,444 params) with a unified `StateRateMLP` (5-layer MLP with pre-LayerNorm and a gated full-path linear skip, ~6,792 params). Expand the ionic latent dimension from 16 to 20. Freeze the scaffold decoder bias at TTP06 physiological rest so `decoder(z=0) = rest` by construction. Add a rest-state attractor regularizer to the Phase A training loss, backed by a precomputed voltage-clamp steady-state grid for later extension. Addresses the CaSR capacity bottleneck identified by the Session 27 integrator error-budget diagnostic (CaSR NRMSE 27.4% vs any-other-gate median ~11%).

## Success Criteria

- [ ] `StateRateMLP` replaces `IonicRateMLP` + `conc_kan`; all existing and new tests pass
- [ ] `decoder(z=0) = TTP06_REST_IONIC_STATE` by construction, verified on fresh init and after `load_state_dict`
- [ ] `L_rest` attractor regularizer computes without error and produces finite gradient
- [ ] Phase A training converges: `val_loss` decreases monotonically over ≥10 epochs on T1 multi-BCL AND final `val_loss < 0.02` (see Phase 3 Exit Criteria for rationale on the 0.02 v4 threshold vs. the v3 0.0088 oracle)
- [ ] Post-Phase-A re-run of `integrator_error_budget.py` shows CaSR NRMSE **< 20%** (baseline: v3 = 27.4%)
- [ ] No regressions in pre-existing tests (`Surrogate/tests/`)

## Architecture Changes

- MOD: `Surrogate/surrogate/model/stage1.py` — constants (IONIC_DIM 16→20), replace `IonicRateMLP` + `conc_kan` with `StateRateMLP`, rewrite `dzdt` to use it, update compression input dim (20→24) and all references, add `pin_rest_bias()` + frozen `TTP06_REST_IONIC_STATE`
- MOD: `Surrogate/surrogate/training/node_rollout.py` — add `L_rest` term with configurable `λ_rest`; verify `INIT_CONC` indexing auto-adapts
- MOD: `Surrogate/tests/test_model.py` — update any dim fixtures referencing 16 or 20 to match new CARRIED_DIM=24
- MOD: `Surrogate/tests/test_training.py` — same
- NEW: `Surrogate/surrogate/data/voltage_clamp_ss.py` — TTP06 voltage-clamp integrator to compute `z_ss(V)` grid
- NEW: `Surrogate/diagnostics/artifacts/z_ss_grid.pt` — cached `(V_grid, z_ss_grid)` tensor produced by the above

## Harness

Training is run through the **`cardiac_ml`** Hydra+MLflow harness (landed 2026-04-20). The v4 model is picked up automatically by `cardiac_ml.model.ionic_node_factory.make_node` because it instantiates `IonicStage1(scaffold=True)` — the v4 class post-Phase-1. No harness changes are required; the existing `conf/experiment/ionic_node_t1.yaml` composes model + data + training for Session 25 oracle parity, and `Surrogate/surrogate/training/node_step.py` already wraps `node_rollout` in the harness-compatible `train_step_fn(trainer, batch) -> dict` protocol. `train_node.py`'s argparse CLI is the archived pre-harness entry; **do not use it for v4 training**. See MEMORY.md "Session 25 parity oracle is `Surrogate/run_multi_bcl.py`" and "cardiac_ml Harness".

## Known Failures (from IDEALOG — do NOT retry)

- **Learned `g_φ` integrator head** (rate + dt → Δz trained against dopri5) — Session 27 diagnostic: Euler vs. dopri5 RMSE is 215× smaller than capacity error. Integrator is not the bottleneck. (IDEALOG Session 27 Q1.)
- **Single KAN layer for concentration rates** — additively universal, multiplicatively blind; cannot represent `I_NaCa = exp(γVF/RT)·Na_i³·Ca_i` or `I_CaL` flux products. Fixed by unified MLP. (IDEALOG Session 27 Q5.)
- **Separate ionic and concentration rate paths** — scar tissue from `conc_mlp` Xavier-init feedback explosion (1e237); the no-cross-talk fix was KAN, not path separation. (IDEALOG Session 27 Q6.)
- **Per-layer residuals** (residual around each single `Linear+GELU`) — not standard practice; full-path or block-level is preferred.
- **KAN readout at final layer** — cost-prohibitive (6,912 params = 60% of model budget); plain Linear readout is minimum-viable given 5-layer MLP upstream.
- **Dropout inside the rate predictor** — corrupts the vector field with Bernoulli noise during `odeint`, destabilizes adjoint gradient. (IDEALOG Session 27.)
- **Hard-coded latent to HH gate variables** — violates multi-model goal (TTP06 vs. ORd state set mismatch) and Layer 0 maxim. (IDEALOG Session 27 Q10.)
- **Nonlinear decoder** (sigmoid/MLP/KAN on readout) — launders knowledge into training-only scaffold. (IDEALOG Session 27 Q3.)
- **Adjoint integration with random weights** — diverges. Start with `adjoint=False` (backprop-through-solver), switch to adjoint after vector field is partially trained. (IDEALOG 2026-04-07.)
- **`dopri8` early in NODE training** — diverges on untrained vector field. Use `dopri5` with `rtol=atol=1e-3`. (IDEALOG 2026-04-07.)
- **Xavier init on last layer of rate MLP** — ODE immediately produces NaN. Zero-init last layer is mandatory. Carried forward in v4 as `readout.weight = 0`.

---

## Phase 1: Core Architecture Pivot

**Goal**: Replace `IonicRateMLP` + `conc_kan` with unified `StateRateMLP`. Expand `IONIC_DIM` 16→20. Update compression and decoder input dims. Freeze `ionic_state_decoder.bias` at TTP06 rest. After this phase the model instantiates cleanly, forward+backward compile, all tests pass, and decoder(z=0) decodes to rest — but nothing is trained yet.

**Tier**: large
**Estimated scope**: 1 file heavy-modified (`stage1.py`), 2 test files updated.

### Phase 1 Context

- **Partial-progress flag (2026-04-20 — per-item):** a flash-test pass of Phase 1 has been applied to the working tree. Run `git diff 8f191f77 -- Surrogate/surrogate/model/stage1.py` first. Expected state per step:
  - Step 1.1 — **DONE**. `IONIC_DIM=20`, `CARRIED_DIM=24`, `H_STATE_MLP=32`, no `MLP_HIDDEN`. Verify via the Step 1.1 grep / import check; if it passes, tick the Checklist without making new edits.
  - Step 1.2 — **DONE**. `TTP06_REST_IONIC_STATE` constant present at module level with the 14 values. Verify via import.
  - Step 1.3 — **DONE including skip_logit fix**. `StateRateMLP` class exists; `skip_logit = BETA_INIT` (verify via `grep BETA_INIT Surrogate/surrogate/model/stage1.py`).
  - Step 1.4 — **DONE**. `dzdt` rewired to `state_rate_mlp`; `IonicRateMLP`, `conc_kan`, `KANLayer` imports removed. Compression layers unchanged.
  - Step 1.5 — **DONE**. `pin_rest_bias` method exists and is called from `_init_weights`.
  - Step 1.6 — **PARTIAL**. Most dim literals and stale attribute references were removed during the flash-test pass, but the full negative-grep assertion (`grep -rn "ionic_rate_mlp\|conc_kan\|MLP_HIDDEN\|VoltageAttention" Surrogate/tests/`) has NOT been re-run since; re-run it before marking Phase 1 complete. Also verify Test Specs added in this plan (e.g., `test_skip_logit_init_beta`, `test_rate_magnitude_at_rest_bounded`, `test_no_stale_attrs`, `test_mlp_hidden_removed`) are actually present in `test_model.py`. If absent, ADD them; they were specified post-flash-test and are not in the working tree yet.
- v3 checkpoints (`multi_bcl_002/best.pt` etc.) become unloadable because state-dict keys change. This is acceptable; Phase A restarts fresh.
- All tensors float64 (`torch.float64`). Project convention.
- Conda env: `heart-conduction`. All verify commands: `conda run -n heart-conduction pytest ...`.
- Compression layers (`gate_conductance_linear`, `gate_conductance_mlp`, `gate_conductance_logit`) and Nernst are STRUCTURALLY unchanged from v3 — only their input dim changes (20 → 24). Do not rewrite them; just update dims. See Step 1.4's compression-capacity note for when to revisit `COMP_H1`/`COMP_H2`.
- Stage 2 is completely unchanged and untrained. Do not modify it.
- TTP06 rest ionic state (14 dims, order `[m, h, j, r, s, d, f, f2, fCass, Xr1, Xr2, Xs, RR, CaSR]`): `[0.001720, 0.744400, 0.704500, 0.000000, 0.999998, 0.000034, 0.788800, 0.975500, 0.995300, 0.006210, 0.471200, 0.001720, 0.907300, 3.640000]`. Extracted from `tier01.h5 / steady_bcl2000_dt0.01` at t=0 (verified 2026-04-19).

### Step 1.1: Update constants
**Model**: opus

#### Read First
- `Surrogate/surrogate/model/stage1.py:27-41` — current constants block
- `Research/Active/surrogate_pipeline/ARCHITECTURE_v4.md` Section 4 — table of all constants and their physical role

#### Why
Centralizing the dim change in the constants block means downstream modules read the new values without further changes. `H_STATE_MLP=32` is new in v4 (it was `MLP_HIDDEN=16` in v3, which is removed with `IonicRateMLP`).

#### Implementation Spec
**Files to modify:** `Surrogate/surrogate/model/stage1.py:27-41`

Changes:
- `IONIC_DIM`: 16 → 20
- `CARRIED_DIM = IONIC_DIM + CONC_DIM` auto-becomes 24
- Add `H_STATE_MLP = 32` (new StateRateMLP hidden width)
- Remove `MLP_HIDDEN = 16` (only used by old IonicRateMLP, which is deleted in Step 1.3)
- `CONC_KAN_GRID`, `CONC_KAN_ORDER` — keep; harmless unused constants, may be reused if KAN is ever reintroduced

#### Pseudocode
```python
# Surrogate/surrogate/model/stage1.py (constants block)
IONIC_DIM = 20                               # was 16
CONC_DIM = 4                                 # unchanged
CARRIED_DIM = IONIC_DIM + CONC_DIM           # = 24 (was 20)
COND_DIM = 8                                 # unchanged
H_STATE_MLP = 32                             # NEW — StateRateMLP hidden width
# (MLP_HIDDEN = 16 removed)
COMP_H1 = 12                                 # unchanged
COMP_H2 = 12                                 # unchanged
N_IONIC_TARGETS = 14                         # unchanged
N_CONDUCTANCE_TARGETS = 5                    # unchanged
CONC_KAN_GRID = 5                            # unchanged (harmless)
CONC_KAN_ORDER = 3                           # unchanged (harmless)
BETA_INIT = -5.0                             # unchanged
```

#### Test Spec
- `test_constants_v4`: import and assert `(IONIC_DIM, CONC_DIM, CARRIED_DIM, H_STATE_MLP) == (20, 4, 24, 32)`.
- `test_mlp_hidden_removed`: `import surrogate.model.stage1 as m`; assert `not hasattr(m, 'MLP_HIDDEN')` — enforces the removal rather than a silent rename.

#### Checklist
- [ ] `IONIC_DIM = 20`
- [ ] `CARRIED_DIM = 24` (via derivation)
- [ ] `H_STATE_MLP = 32` added
- [ ] `MLP_HIDDEN` removed (grep returns no hit in stage1.py)
- [ ] Module still imports cleanly

#### Verify
```
conda run -n heart-conduction python -c "
from surrogate.model import stage1 as m
assert (m.IONIC_DIM, m.CONC_DIM, m.CARRIED_DIM, m.H_STATE_MLP) == (20, 4, 24, 32)
assert not hasattr(m, 'MLP_HIDDEN'), 'MLP_HIDDEN must be removed'
print('ok')
"
```

#### Exit Criteria
- [ ] Constants updated; `MLP_HIDDEN` is gone; import works

#### Risk
Hardcoded `16` or `20` literals elsewhere — mitigation: `grep -rn "= 16\|= 20" Surrogate/surrogate/ --include="*.py"` after change, review each hit. Expected hits are few and contextual (batch sizes, local loop counters). Review each one rather than trying to predict the list here. Test files are handled separately by Step 1.6.

---

### Step 1.2: Add `TTP06_REST_IONIC_STATE` constant
**Model**: opus

#### Read First
- `Surrogate/surrogate/model/stage1.py:27-41` — location to insert (after existing constants block)
- `Research/Active/surrogate_pipeline/ARCHITECTURE_v4.md` Section 8.1 — rest values to four decimal places

#### Why
Used by `pin_rest_bias()` (Step 1.5) to freeze the decoder bias. Module-level constant makes it accessible both to the model's `_init_weights` path and to external test code.

#### Implementation Spec
**Files to modify:** `Surrogate/surrogate/model/stage1.py` — add after the existing constants block, before helper functions.

```python
# === Physics-informed rest state (Session 27, 2026-04-19) ===
# TTP06 physiological rest ionic_state (14 dims, from BCL=2000 t=0 sample).
# Frozen as ionic_state_decoder.bias so decoder(z=0) = rest by construction.
# Latent semantics: z represents deviation from rest.
# Order: [m, h, j, r, s, d, f, f2, fCass, Xr1, Xr2, Xs, RR, CaSR].
TTP06_REST_IONIC_STATE = torch.tensor([
    0.001720,  # m
    0.744400,  # h
    0.704500,  # j
    0.000000,  # r
    0.999998,  # s
    0.000034,  # d
    0.788800,  # f
    0.975500,  # f2
    0.995300,  # fCass
    0.006210,  # Xr1
    0.471200,  # Xr2
    0.001720,  # Xs
    0.907300,  # RR
    3.640000,  # CaSR
], dtype=torch.float64)
```

#### Pseudocode
(Verbatim constant above; no logic.)

#### Test Spec
- `test_ttp06_rest_shape_dtype`: import `TTP06_REST_IONIC_STATE`; assert `.shape == (14,)` and `.dtype == torch.float64`.
- `test_ttp06_rest_values_bounded`: assert all gate values in `[0, 1]` (rows 0–12) and `CaSR ∈ [0, 10]` (row 13) — guards typos in the hand-transcribed tensor.
- `test_ttp06_rest_matches_tier01`: optional sanity — if `/media/HDD/norepinephrine/surrogate_data/raw/tier01.h5` is mounted, load `steady_bcl2000_dt0.01/data[0]`, apply `V3Preprocessor`, assert the first `ionic_states[0]` row matches `TTP06_REST_IONIC_STATE` within 1e-3 tolerance. This is the extraction source; guards against stale values if the simulator protocol ever changes.

#### Checklist
- [ ] Constant added at module level
- [ ] Dtype is `torch.float64`
- [ ] Shape is `(14,)`
- [ ] Order matches scaffold decoder output (preprocessor.py:149-157 sets the canonical order)

#### Verify
```
conda run -n heart-conduction python -c "from surrogate.model.stage1 import TTP06_REST_IONIC_STATE as R; import torch; assert R.shape == (14,) and R.dtype == torch.float64 and abs(R[0].item() - 0.00172) < 1e-6; print('ok', R.shape)"
```

#### Exit Criteria
- [ ] Constant importable, correct shape/dtype

#### Risk
Wrong ordering vs. decoder output — mitigation: preprocessor.py:149-157 shows the decoder target order (`m,h,j,r,s,d,f,f2,fCass,Xr1,Xr2,Xs,RR` then `CaSR`); match exactly.

---

### Step 1.3: Implement `StateRateMLP` module
**Model**: opus

#### Read First
- `Surrogate/surrogate/model/stage1.py:49-67` — existing `IonicRateMLP` for reference pattern
- `Research/Active/surrogate_pipeline/ARCHITECTURE_v4.md` Section 5 — full StateRateMLP spec (forward pass, param breakdown, initialization)

#### Why
Core architectural change. Single unified rate network replaces the v3 ionic + conc split. 5-layer depth is calibrated for TTP06/ORd cross-product physics (IDEALOG Session 27 Q7); pre-LayerNorm is for gradient stability at depth; gated full-path skip gives the model an explicit linear-vs-nonlinear choice.

#### Implementation Spec
**Files to modify:** `Surrogate/surrogate/model/stage1.py`

Replace `IonicRateMLP` class with `StateRateMLP`:

```python
class StateRateMLP(nn.Module):
    """Unified rate predictor: (z_full, Vm) → dz/dt. Replaces IonicRateMLP + conc_kan.

    5-layer MLP with pre-LayerNorm and gated full-path linear skip.
    Depth is calibrated for TTP06/ORd cross-product physics (quartic products
    like I_NaCa, I_CaL flux). Linear readout (no KAN); zero-init at readout
    guarantees rate ≈ 0 at init → ODE stable. Gated skip lets the model choose
    linear vs. deep rate decomposition (logit init = 0 → α = 0.5).

    Input dim: carried_dim + 1 (for Vm). Output dim: carried_dim.
    Default shape for TTP06 v4: 25 → 32 → 32 → 32 → 32 → 32 → 24.

    NOTE (CRITICAL): skip_logit MUST be initialized to BETA_INIT (-5), NOT zero.
    Zero init gives alpha = 0.5 at start; combined with Xavier skip weights and
    V = -85.23 mV, rate magnitude at rest is ~20/dim, which the ODE solver
    amplified to 1e90 on the 2026-04-19 flash training. Keep BETA_INIT.
    """

    def __init__(self, carried_dim: int = CARRIED_DIM, hidden: int = H_STATE_MLP):
        super().__init__()
        in_dim = carried_dim + 1  # carried + Vm
        self.fc1 = nn.Linear(in_dim, hidden)   # stem
        self.fc2 = nn.Linear(hidden, hidden)   # hidden 1
        self.fc3 = nn.Linear(hidden, hidden)   # hidden 2
        self.fc4 = nn.Linear(hidden, hidden)   # hidden 3
        self.fc5 = nn.Linear(hidden, hidden)   # hidden 4
        # Pre-norm on each hidden Linear's input; five norms total (one before readout)
        self.ln1 = nn.LayerNorm(hidden)   # applied to fc1 output before fc2
        self.ln2 = nn.LayerNorm(hidden)   # before fc3
        self.ln3 = nn.LayerNorm(hidden)   # before fc4
        self.ln4 = nn.LayerNorm(hidden)   # before fc5
        self.ln5 = nn.LayerNorm(hidden)   # before readout
        # Readout: zero-init weights and bias for ODE stability
        self.readout = nn.Linear(hidden, carried_dim)
        # Gated full-path skip: input → rate, linear map, per-dim gate
        self.skip = nn.Linear(in_dim, carried_dim, bias=False)
        self.skip_logit = nn.Parameter(torch.full((carried_dim,), BETA_INIT))  # σ(-5)≈0.007, near-dormant

    def forward(self, z: Tensor, Vm: Tensor) -> Tensor:
        x = torch.cat([z, Vm.unsqueeze(-1)], dim=-1)
        h = F.gelu(self.fc1(x))
        h = F.gelu(self.fc2(self.ln1(h)))
        h = F.gelu(self.fc3(self.ln2(h)))
        h = F.gelu(self.fc4(self.ln3(h)))
        h = F.gelu(self.fc5(self.ln4(h)))
        rate_deep = self.readout(self.ln5(h))
        rate_skip = self.skip(x)
        alpha = torch.sigmoid(self.skip_logit)
        return rate_deep + alpha * rate_skip
```

Delete `IonicRateMLP` class entirely.

**Callers must `.double()` the model after construction** (all project tensors are float64 per CLAUDE.md); `nn.LayerNorm`'s default parameters are float32 and promote on `.double()`. The existing `IonicStage1` construction pattern in `ionic_surrogate_v3.py` already does this, but new call sites must follow suit. **Runtime enforcement is weak** — there is no `assert dtype == float64` inside `dzdt`. If the model is float32 and `z`/`Vm` are float64, PyTorch will raise a dtype-mismatch RuntimeError at matmul time, which is the current safety net. Consider adding an assertion at the top of `dzdt` only if a silent-promotion bug is ever observed; do not add it prophylactically.

#### Pseudocode
```python
# Forward pass (symbolic)
x       = cat([z, Vm]).float64               # (B, carried_dim + 1)
h1      = GELU(Linear(in_dim -> hidden)(x))                        # stem
h2      = GELU(Linear(hidden -> hidden)(LayerNorm(h1)))            # pre-norm block 1
h3      = GELU(Linear(hidden -> hidden)(LayerNorm(h2)))            # block 2
h4      = GELU(Linear(hidden -> hidden)(LayerNorm(h3)))            # block 3
h5      = GELU(Linear(hidden -> hidden)(LayerNorm(h4)))            # block 4
deep    = Linear(hidden -> carried_dim)(LayerNorm(h5))             # readout, W+b zero-init
skip    = Linear(in_dim -> carried_dim, bias=False)(x)             # Xavier
alpha   = sigmoid(skip_logit)                                      # σ(BETA_INIT) ≈ 0.007 at init
return deep + alpha * skip
```

#### Test Spec
Add to `Surrogate/tests/test_model.py`:

- `test_state_rate_mlp_shape`: instantiate `StateRateMLP(24, 32).double()`, pass `(z=torch.randn(3, 24, dtype=torch.float64), Vm=torch.randn(3, dtype=torch.float64))`, assert output shape `(3, 24)` and dtype float64.
- `test_state_rate_mlp_param_count`: instantiate defaults, `n = sum(p.numel() for p in m.parameters())`, assert `n` ∈ `[6700, 6900]` (expected 6,792; small bracket for LayerNorm initialization quirks).
- `test_state_rate_mlp_grad_flow`: forward + `.sum().backward()`, verify all params have non-None, finite gradient.
- `test_state_rate_mlp_zero_readout_contribution_at_init`: at init, manually compute `rate_deep` by running fc1–fc5 then `readout(ln5(h))`; assert it's near zero (< 1e-10). Verifies readout zero-init.
- `test_skip_logit_init_beta`: instantiate model; compute `alpha = torch.sigmoid(module.state_rate_mlp.skip_logit)` and assert `torch.allclose(alpha, torch.sigmoid(torch.tensor(BETA_INIT, dtype=alpha.dtype)), atol=1e-6)` where `BETA_INIT` is imported from `surrogate.model.stage1` (expected value: `sigmoid(-5) ≈ 0.006693`). Guards against regression of the 2026-04-19 divergence bug. Computing the expected value from `BETA_INIT` rather than hardcoding the numeric literal keeps the test tied to the single source of truth.
- `test_rate_magnitude_at_rest_bounded`: with fresh init, pass `z_rest = [zeros(20), 10, 138, 1e-4, 2e-4]` and `Vm = -85.23`; assert `rate.abs().max() < 1.0`. Rate at rest must be small for ODE stability.

#### Checklist
- [ ] `StateRateMLP` class implemented per spec
- [ ] `IonicRateMLP` class deleted
- [ ] All Test Spec items pass

#### Verify
```
conda run -n heart-conduction pytest Surrogate/tests/test_model.py -v -k "state_rate_mlp or skip_logit or rate_magnitude_at_rest"
```

#### Exit Criteria
- [ ] `StateRateMLP` instantiates, runs forward+backward
- [ ] Parameter count ≈ 6,792
- [ ] Zero-init readout verified
- [ ] `sigmoid(skip_logit) ≈ 0.007` at init (not 0.5)
- [ ] Rate at rest `||rate||_∞ < 1.0`

#### Risk
- LayerNorm dtype: default `nn.LayerNorm` creates float32 weights. Mitigation: call `.double()` on the full model after construction; the Test Spec and Verify steps both cover this.
- Skip logit mis-init: `nn.Parameter(torch.zeros(...))` would give `α = 0.5` at init → 2026-04-19 flash training divergence (loss 1e90). **MUST use `torch.full((carried_dim,), BETA_INIT)`**. Test `test_skip_logit_init_beta` guards this.

---

### Step 1.4: Rewire `IonicStage1.dzdt` to use `StateRateMLP`; update compression input dims
**Model**: opus

#### Read First
- `Surrogate/surrogate/model/stage1.py:71-200` — existing `IonicStage1` class
- `Research/Active/surrogate_pipeline/ARCHITECTURE_v4.md` Sections 5–6 — Stage 1 dzdt and compression

#### Why
`dzdt` currently computes ionic + conc rates separately and concatenates. With `StateRateMLP` as a unified rate predictor, `dzdt` is a single forward pass. Compression layers take full carried_state as input (24 dims now, was 20); they are structurally unchanged but dim references must update.

#### Implementation Spec
**Files to modify:** `Surrogate/surrogate/model/stage1.py` — `IonicStage1.__init__`, `dzdt`, `_init_weights`, `forward`.

In `IonicStage1.__init__`:
- Delete `self.ionic_rate_mlp = IonicRateMLP(...)` → replace with `self.state_rate_mlp = StateRateMLP(self.carried_dim, H_STATE_MLP)`
- Delete `self.conc_kan = KANLayer(...)` entirely
- Delete `conc_kan_in` local variable and any `CONC_KAN_GRID`/`CONC_KAN_ORDER` references inside the class (keep module-level constants untouched)
- `self.gate_conductance_linear`, `self.gate_conductance_mlp`, `self.gate_conductance_logit` — unchanged in code (already read `self.carried_dim`)
- `self.ionic_state_decoder = nn.Linear(ionic_dim, n_ionic_targets)` — unchanged in code (reads `ionic_dim` attr)

In `IonicStage1.dzdt`:
```python
def dzdt(self, z: Tensor, Vm: Tensor) -> Tensor:
    squeezed = z.dim() == 1
    if squeezed:
        z = z.unsqueeze(0)
        Vm = Vm.view(1)
    dz_dt = self.state_rate_mlp(z, Vm)
    if squeezed:
        dz_dt = dz_dt.squeeze(0)
    return dz_dt
```

In `IonicStage1._init_weights`, replace `IonicRateMLP` block with:

```python
# StateRateMLP hidden Linears: Xavier uniform
for fc in [self.state_rate_mlp.fc1, self.state_rate_mlp.fc2,
          self.state_rate_mlp.fc3, self.state_rate_mlp.fc4,
          self.state_rate_mlp.fc5]:
    xavier_uniform_(fc.weight)
# StateRateMLP readout: zero-init for ODE stability
nn.init.zeros_(self.state_rate_mlp.readout.weight)
nn.init.zeros_(self.state_rate_mlp.readout.bias)
# StateRateMLP skip: Xavier on skip weight; skip_logit is constructor-initialized
# to BETA_INIT (-5) — sigmoid(-5)≈0.007. MUST NOT re-init to zero here (that
# would give alpha=0.5 and reproduce the 2026-04-19 divergence; see Step 1.3 Risk).
xavier_uniform_(self.state_rate_mlp.skip.weight)
# Compression (unchanged from v3)
xavier_uniform_(self.gate_conductance_linear.weight)
xavier_uniform_(self.gate_conductance_mlp[0].weight)
xavier_uniform_(self.gate_conductance_mlp[2].weight)
xavier_uniform_(self.gate_conductance_mlp[4].weight)
```

In `IonicStage1.forward`: currently may reference the old rate paths for scaffold prediction logic. Review and update — only compression + scaffold decoding happens here, not rate computation.

**Compression capacity sanity check (new in v4).** With `carried_dim` bumped 20 → 24 and `cond_dim` kept at 8, the compression ratio goes from 2.5× to 3.0×. `COMP_H1=12` and `COMP_H2=12` are retained unchanged: the MLP's hidden width (12) is below `carried_dim` (24), so the compression IS a bottleneck — intended. Do not expand `COMP_H1/H2` without first observing empirical degradation on `gate_conductance_decoder` regression in Phase B training. If Phase B shows conductance loss plateau above v3 baseline, revisit (bump to `COMP_H1 = COMP_H2 = 16`, +~150 params).

#### Pseudocode
```python
# IonicStage1.__init__ — the diff
- self.ionic_rate_mlp = IonicRateMLP(ionic_dim, MLP_HIDDEN)
- conc_kan_in = ionic_dim + 1
- self.conc_kan = KANLayer(conc_kan_in, conc_dim, grid_size=CONC_KAN_GRID, ...)
+ self.state_rate_mlp = StateRateMLP(self.carried_dim, H_STATE_MLP)
# Compression (unchanged code; self.carried_dim is now 24):
  self.gate_conductance_linear = nn.Linear(self.carried_dim, cond_dim, bias=False)
  self.gate_conductance_mlp = nn.Sequential(
      nn.Linear(self.carried_dim, comp_h1), nn.GELU(),
      nn.Linear(comp_h1, comp_h2),           nn.GELU(),
      nn.Linear(comp_h2, cond_dim),
  )

# IonicStage1.dzdt — the new body
def dzdt(self, z: Tensor, Vm: Tensor) -> Tensor:
    squeezed = z.dim() == 1
    if squeezed: z = z.unsqueeze(0); Vm = Vm.view(1)
    dz_dt = self.state_rate_mlp(z, Vm)
    return dz_dt.squeeze(0) if squeezed else dz_dt
```

#### Test Spec
- Update `test_stage1_dzdt_shape`: construct `IonicStage1(scaffold=True).double()`, pass `z=torch.zeros(3, 24, dtype=torch.float64), Vm=torch.zeros(3, dtype=torch.float64)`, assert output shape `(3, 24)`.
- Update `test_stage1_forward_compress_shape`: compression still produces `(batch, 8)` regardless of carried_dim change.
- `test_stage1_backward_finite`: `rate.sum().backward()` on the dzdt output; assert every **trainable** (`requires_grad=True`) param has finite grad. Frozen `ionic_state_decoder.bias` is skipped.
- `test_no_stale_attrs`: `stage1 = IonicStage1()`; assert `not hasattr(stage1, 'ionic_rate_mlp') and not hasattr(stage1, 'conc_kan')` — enforces removal rather than quiet rename.

#### Checklist
- [ ] `IonicRateMLP` references removed
- [ ] `conc_kan` references removed
- [ ] `state_rate_mlp` instance attribute added
- [ ] `dzdt` uses unified path
- [ ] `_init_weights` updated
- [ ] `forward` method still works (compression + scaffold)
- [ ] Updated dzdt shape/backward tests pass

#### Verify
```
conda run -n heart-conduction pytest Surrogate/tests/test_model.py -v -k "stage1 or dzdt or compress or forward"
```

#### Exit Criteria
- [ ] `IonicStage1` instantiates with `IONIC_DIM=20`
- [ ] `dzdt` returns `(B, 24)` rate vector
- [ ] Backward pass produces finite gradients for all trainable params

#### Risk
- Hidden references to `self.ionic_rate_mlp` or `self.conc_kan` anywhere in the file — mitigation: grep `ionic_rate_mlp\|conc_kan` after edit.
- `dzdt`'s removal of per-path concatenation logic could accidentally drop unbatched-shape handling — mitigation: the `squeezed` branch is preserved explicitly.

---

### Step 1.5: Add `pin_rest_bias()` method; call from `_init_weights`
**Model**: opus

#### Read First
- `Surrogate/surrogate/model/stage1.py:_init_weights` — after Step 1.4 state
- `Research/Active/surrogate_pipeline/ARCHITECTURE_v4.md` Section 8.1 — bias-freeze semantics and reload caveat

#### Why
Rest-attractor mechanism requires `decoder(z=0) = rest` by construction, enforced via a frozen bias. Method must be callable independently of init because `load_state_dict` silently overwrites the bias.

#### Implementation Spec
**Files to modify:** `Surrogate/surrogate/model/stage1.py`

Add method on `IonicStage1`:

```python
def pin_rest_bias(self) -> None:
    """Pin ionic_state_decoder.bias to TTP06 physiological rest and freeze.

    Makes decoder(z=0) = TTP06_REST_IONIC_STATE by construction. Latent
    semantics become "deviation from rest".

    CALL AFTER load_state_dict() — torch.nn.Module.load_state_dict overwrites
    the bias with the checkpoint value, silently breaking the rest-attractor
    guarantee. Idempotent; no-op if scaffold is absent.
    """
    if not hasattr(self, "ionic_state_decoder"):
        return
    assert self.ionic_state_decoder.out_features == TTP06_REST_IONIC_STATE.numel(), (
        f"Decoder out_features ({self.ionic_state_decoder.out_features}) "
        f"does not match TTP06_REST_IONIC_STATE length "
        f"({TTP06_REST_IONIC_STATE.numel()})"
    )
    with torch.no_grad():
        self.ionic_state_decoder.bias.copy_(
            TTP06_REST_IONIC_STATE.to(
                dtype=self.ionic_state_decoder.bias.dtype,
                device=self.ionic_state_decoder.bias.device,
            )
        )
    self.ionic_state_decoder.bias.requires_grad_(False)
```

In `_init_weights`, at the very end, add:
```python
# Physics-informed rest attractor: pin decoder bias to TTP06 rest
self.pin_rest_bias()
```

#### Pseudocode
```python
# Bias-freeze invariant (called at end of _init_weights and after every load_state_dict)
with torch.no_grad():
    decoder.bias.copy_(TTP06_REST_IONIC_STATE.to(decoder.bias))
decoder.bias.requires_grad_(False)
# Result: decoder(0) = W @ 0 + bias = rest
```

#### Test Spec
Add to `Surrogate/tests/test_model.py`:

- `test_decoder_bias_frozen_at_init`: fresh `IonicStage1(scaffold=True).double()`; verify `ionic_state_decoder.bias.requires_grad == False` and `torch.allclose(bias.cpu(), TTP06_REST_IONIC_STATE)`.
- `test_decoder_zero_maps_to_rest_with_nonzero_weight`: pass `z_ionic = torch.zeros(1, 20, dtype=torch.float64)` through decoder, assert output ≈ `TTP06_REST_IONIC_STATE` within 1e-12. ALSO assert `ionic_state_decoder.weight.abs().max() > 1e-4` — Xavier init should give W noticeably non-zero, so the allclose equality at `z=0` genuinely isolates the bias-freeze (rather than passing trivially because weight happens to be zero).
- `test_pin_rest_bias_after_reload`: instantiate, replace `bias` with `torch.randn(14, dtype=torch.float64)` in-place AND set `requires_grad = True`, call `pin_rest_bias()`, verify bias restored AND `requires_grad is False` again.
- `test_trainable_param_count_excludes_bias`: assert `(total - trainable) == 14` — precisely the 14 bias elements, not more (which would mean something else got silently frozen) and not fewer (which would mean the freeze failed).

#### Checklist
- [ ] `pin_rest_bias` method added
- [ ] `_init_weights` calls it last
- [ ] Four new tests pass

#### Verify
```
conda run -n heart-conduction pytest Surrogate/tests/test_model.py -v -k "rest_bias or decoder_zero or pin or trainable"
```

#### Exit Criteria
- [ ] `decoder(z=0) = TTP06_REST_IONIC_STATE` on fresh init
- [ ] Bias frozen (`requires_grad=False`)
- [ ] `pin_rest_bias` restores frozen state after bias mutation

#### Risk
- `.double()` timing: bias dtype must be float64. Mitigation: `bias.dtype == torch.float64` assertion in test.
- Device placement: `.to(device=bias.device)` inside `pin_rest_bias` handles CUDA moves.

---

### Step 1.6: Update test fixtures for new dimensions, remove stale-architecture tests
**Model**: opus

#### Read First
- `Surrogate/tests/test_model.py` — all test functions; hardcoded `16`, `20`, carried_dim literals AND references to removed attributes (`ionic_rate_mlp`, `conc_kan`, `MLP_HIDDEN`, `VoltageAttention`).
- `Surrogate/tests/test_training.py` — same, plus the freeze-mask tests which reference v3 component names and will pass vacuously under v4 if not updated.
- `Surrogate/tests/test_preprocessing.py`, `test_data_generation.py`, `test_ord_data.py` — quick scan; if they never touch ionic/conc dims, leave alone.

#### Why
Tests with hardcoded dim values break under v4. Tests referencing removed attributes (`ionic_rate_mlp`, `conc_kan`) pass vacuously — a silent regression that would let a broken model ship. Both must be enumerated and fixed deliberately, not pattern-greped.

#### Implementation Spec
**Files to modify:** `Surrogate/tests/test_model.py`, `Surrogate/tests/test_training.py`

Three passes:

1. **Dim literal sweep** (`test_model.py`): run `grep -n "= 16\|= 20\|(16,\|(20,\|,16)\|,20)" Surrogate/tests/test_model.py`; replace with imported constants or instance attributes. Use the programmatic sed script pattern already established (this session used Python regex for 34 substitutions).

2. **Stale-attribute enumeration (verified against HEAD 2026-04-20)** — most v3-specific tests in `test_model.py` were already renamed/deleted during the Session 27 flash-test pass. Current state (grep-verified):
   - `test_stage1_ionic_rate_mlp` — **ALREADY DELETED/RENAMED**. Replaced by `test_stage1_state_rate_mlp` at line 137. No action needed; tick the Checklist.
   - `test_stage1_ionic_conc_separate` — **ALREADY DELETED**. No action.
   - `test_stage1_conc_kan` — **ALREADY DELETED/RENAMED** to `test_stage1_conc_rate_shape` at line 175. No action.
   - `test_stage1_param_count` (line 191) — **ALREADY UPDATED** to band `[7800, 8100]` and scaffold `== 339`. Verify with `grep -n '7800 <= total' Surrogate/tests/test_model.py`; tick if confirmed.
   - `test_stage1_remove_scaffold` (line 206) — **ALREADY UPDATED**. Verify with `grep -n 'total_before - 339' Surrogate/tests/test_model.py`.
   - `test_stage1_dzdt_numerical` (line 274) — **ALREADY UPDATED** to compare against `state_rate_mlp`. Verify with `grep -n 'state_rate_mlp' Surrogate/tests/test_model.py`.
   - `test_v3_param_count` (line 619) — **ALREADY UPDATED** to band `[7450, 7650]`. Verify with `grep -n '7450' Surrogate/tests/test_model.py`.
   - `test_stage1_gradient_flow` (line 234) + `test_v3_autoregressive` (line 484) — frozen-param skip. Verify via `grep -n 'requires_grad' Surrogate/tests/test_model.py` returns hits at lines 248 and 518 (the `if not p.requires_grad: continue` pattern). If confirmed, already applied; if missing, patch as described.
   - `test_rollout_gradient_flow` in `test_training.py` — mark `@pytest.mark.xfail` if not already. Verify with `grep -n 'xfail' Surrogate/tests/test_training.py`.
   - `test_training.py` `ionic_rate_mlp` freeze-mask hits — run `grep -n 'ionic_rate_mlp' Surrogate/tests/test_training.py`; for each hit, if it's in a freeze-mask pattern list (`['ionic_rate_mlp', ...]`), replace with `['state_rate_mlp']`; otherwise delete.

   The key directive is: **run grep first, then ONLY patch what's actually stale**. The enumeration above may show entirely "no action" after the flash-test pass, in which case Step 1.6's actual work is (a) running the final negative-grep assertion, and (b) adding any NEW tests from the `Test Spec` sections of Steps 1.1–1.5 that haven't landed yet.

3. **Negative assertion**: `grep -rn "ionic_rate_mlp\|conc_kan\|MLP_HIDDEN\|VoltageAttention\|KANLayer" Surrogate/tests/` must return ZERO hits after this step. This is the discovery guarantee — pattern-based enumeration alone missed these in Step 1.6's original spec. `KANLayer` is included because Phase 1 Cleanup removes the `.kan` import from `stage1.py`; residual test references would break on reimport.

#### Pseudocode
```bash
# 1. Dim-literal substitution (reuse the flash-test one-liner pattern)
python3 -c "
import re
pairs = [
    (r'\btorch\.randn\(B, 20\)',      'torch.randn(B, 24)'),
    (r'\btorch\.randn\(20\)',          'torch.randn(24)'),
    (r'\[:, :16\]',                    '[:, :20]'),
    (r'\[:, 16:\]',                    '[:, 20:]'),
    # ... (full list per flash-test session)
]
for path in ['Surrogate/tests/test_model.py', 'Surrogate/tests/test_training.py']:
    text = open(path).read()
    for p,r in pairs: text = re.sub(p, r, text)
    open(path, 'w').write(text)
"
# 2. Manual review each test enumerated above.
# 3. Final check:
grep -rn "ionic_rate_mlp\|conc_kan\|MLP_HIDDEN\|VoltageAttention" Surrogate/tests/
#    -> must return empty
```

#### Test Spec
This step IS the test spec for Phase 1. The exit criterion is "full suite passes with zero stale-architecture references".

#### Checklist
- [ ] All dim literals replaced or use imported constants
- [ ] Each of the 10 enumerated tests above individually updated or deleted (NOT left in a "passes vacuously" state)
- [ ] Grep for `ionic_rate_mlp|conc_kan|MLP_HIDDEN|VoltageAttention` in `Surrogate/tests/` returns empty
- [ ] Gradient-flow assertions skip frozen params

#### Verify
```
conda run -n heart-conduction pytest Surrogate/tests/ -v
grep -rn "ionic_rate_mlp\|conc_kan\|MLP_HIDDEN\|VoltageAttention" Surrogate/tests/ && echo "FAIL: stale refs" || echo "ok: no stale refs"
```

#### Exit Criteria
- [ ] Full test suite passes (xfail allowed for legacy discrete rollout)
- [ ] `grep` for stale attributes returns empty

#### Risk
Vacuous passes — a freeze-mask loop that can't find `ionic_rate_mlp` in a v4 model has an empty iteration and passes without testing anything. Mitigation: explicit enumeration above, plus the negative grep assertion.

---

### Phase 1 Verification

```
conda run -n heart-conduction pytest Surrogate/tests/ -v
```

Smoke test. **Populates INIT_CONC in z_rest** (required for the rest-attractor invariant — `z` with zero concentrations is unphysical and would give a large misleading rate magnitude at init):
```
conda run -n heart-conduction python -c "
import torch, sys
sys.path.insert(0, 'Surrogate')
from surrogate.model.stage1 import IonicStage1, TTP06_REST_IONIC_STATE
from surrogate.training.node_rollout import INIT_CONC

m = IonicStage1(scaffold=True).to(torch.float64)
assert m.ionic_dim == 20 and m.carried_dim == 24

# 1. Decoder(z=0) = rest (by construction)
z_ionic_zero = torch.zeros(m.ionic_dim, dtype=torch.float64)
assert torch.allclose(m.ionic_state_decoder(z_ionic_zero).cpu(), TTP06_REST_IONIC_STATE)

# 2. Rest attractor sanity: z_rest = [zeros(20), INIT_CONC], V=V_rest => rate is small
B = 3
z_rest = torch.zeros(B, 24, dtype=torch.float64)
z_rest[:, m.ionic_dim:] = INIT_CONC
V = torch.full((B,), -85.23, dtype=torch.float64)
rate = m.dzdt(z_rest, V)
assert rate.shape == (B, 24) and torch.isfinite(rate).all()
assert rate.abs().max().item() < 1.0, f'rate at rest ||_inf = {rate.abs().max():.4f}, expected < 1.0'

# 3. Backward flows through all trainable params
rate.sum().backward()

# 4. Param count band (single source of truth: [7800, 8100])
total = sum(p.numel() for p in m.parameters())
trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
assert 7800 <= total <= 8100, f'total {total} out of band'
assert total - trainable == 14, f'expected 14 frozen (bias), got {total - trainable}'
print('Phase 1 smoke: OK', total, 'total,', trainable, 'trainable, rate_inf=', rate.abs().max().item())
"
```

### Phase 1 Exit Criteria

- [ ] All existing + new tests pass (no regressions)
- [ ] Smoke test above passes
- [ ] Model parameter count in `[7800, 8100]` (expected 7,891; bounds match Step 1.6 and the Phase 1 Verification smoke — single source of truth). Trainable count = total − 14 (the 14 decoder-bias elements are frozen).
- [ ] Forward/backward pass produces finite values

### Phase 1 Cleanup

- [ ] `IonicRateMLP` class deleted
- [ ] `KANLayer` import from `.kan` deleted if `conc_kan` was the only usage (grep first)
- [ ] `CONC_KAN_GRID`, `CONC_KAN_ORDER` constants retained as-is (future-proofing)
- [ ] No float32 leaks — grep `\.float()` in `stage1.py` (should be none outside typing)
- [ ] No residual commented-out code from v3 paths
- [ ] V5.3 untouched — trivially satisfied (Surrogate/ is separate)

**→ Commit point: `git commit` after Phase 1 passes. Suggested message: "Surrogate v4: StateRateMLP replaces IonicRateMLP+conc_kan; latent 16→20; decoder bias pinned at rest"**

---

## Phase 2: Rest Attractor in Training Loop + z_ss(V) Precomputation

**Goal**: Precompute TTP06 steady-state ionic state on a voltage grid via a voltage-clamped simulator run; cache the result. Add `L_rest = ||f_θ(z=0, V=V_rest)||²` regularizer to `node_rollout`'s loss with a tunable `λ_rest`. The full voltage-clamp attractor over the V grid is deferred (see Phase 2 Context); MVP is rest-only because it does not require inverting the decoder to produce a latent target at non-rest voltages.

**Tier**: medium
**Estimated scope**: 1 new file (`voltage_clamp_ss.py`, ~80 lines), 1 cached artifact, node_rollout.py ~25 lines added.

### Phase 2 Context

- **TTP06 simulator — MANDATORY import, no hand-rolled RHS.** Use `Monodomain/Engine_V5.4/cardiac_sim/ionic/ttp06/` directly. A standalone TTP06 RHS is ~14 coupled ODEs with ryanodine release, SERCA, CaMKII leak — NOT an 80-line job; the original plan's "or write a minimal standalone rhs" escape hatch is incorrect and must not be taken. Concrete import pattern: `from cardiac_sim.ionic.ttp06.model import TTP06Model` (or the equivalent per V5.4's actual class name — verify at implementation time). If sys.path hygiene from `Surrogate/` is awkward, add `Monodomain/Engine_V5.4/` to `sys.path` in `voltage_clamp_ss.py`, not via a rewrite.
- **V grid decision — rest only for MVP**: `[-85.23]`. Full 9-voltage grid is DEFERRED; see "MVP scope decision" below. Rationale: the full V grid is not consumed by Phase 2's L_rest, and computing `z_ss(V)` at extreme voltages (+60 mV held) for a reduced-state system where concentrations are frozen hits convergence pathologies (CaSR drains unbounded, some gates never settle). Compute the full grid only when Phase 2+ extends L_vclamp beyond rest and actually needs it.
- **Convergence criterion (single rest voltage)**: per-dim relative tolerance. `|dg_i/dt| / (1 + |g_i|) < 1e-6` for all i, OR `t ≥ 500 ms`. Per-dim relative avoids the uniform-1e-6 bias that would favor gates (native range [0,1]) over CaSR (range 0–10).
- **Output artifact**: `Surrogate/diagnostics/artifacts/z_ss_grid.pt`, containing `{'V_grid': tensor(1,), 'z_ss_grid': tensor(1, 14)}` — only the rest row for MVP. Schema is forward-compatible for a full grid later. The artifact's role in MVP is diagnostic/validation: confirm that integrating TTP06 from `TTP06_REST_IONIC_STATE` at `V_rest` stays within tolerance of the starting state (the tensor was extracted from BCL=2000 steady-state, so `z_ss(V_rest) ≈ TTP06_REST_IONIC_STATE` is the correctness check).
- **MVP scope decision**: the training loop implements ONLY `L_rest = ||f_θ(z=0, V=V_rest)||²`. Extending to the full V grid requires computing the correct latent `z_latent(V)` such that `decoder(z_latent) = z_ss_observable(V)`, which in turn requires inverting the (random-at-init, evolving-during-training) decoder `W_d`. This per-step pseudo-inverse computation is expensive and deferred. The rest-only attractor is sufficient to validate the physics-informed regularizer pattern. If Phase 3 validation shows CaSR NRMSE is still >20%, follow-up work (tracked as Phase 2.5 in IDEALOG — to be opened) will extend the V grid.

### Step 2.1: Precompute `z_ss(V)` grid from TTP06 simulator
**Model**: opus

#### Read First
- `Monodomain/Engine_V5.4/cardiac_sim/ionic/ttp06/` — if importable, use its rhs
- `Monodomain/Engine_V5.3/` — validated baseline for TTP06 reference values
- `Surrogate/surrogate/data/preprocessor.py:149-157` — scaffold decoder target ordering

#### Why
Cached steady-state values serve two future purposes: (1) data point for extending `L_vclamp` beyond rest after decoder-pseudo-inverse logic is built, (2) diagnostic reference for plots. The phase 2 MVP does not consume the full grid but records the precomputation for completeness.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/data/voltage_clamp_ss.py`

Functionality:
- `def compute_z_ss_grid(V_grid=(-85.23,), *, initial_state: torch.Tensor | None = None, sim_dt_ms=0.01, max_t_ms=500.0, rel_tol=1e-6) -> dict`:
  - `initial_state` (optional, shape `(14,)` float64): initial ionic-state vector. Defaults to `TTP06_REST_IONIC_STATE.clone()` when None. Exposing this arg is REQUIRED for the perturbed-start test in Test Spec (`test_z_ss_from_perturbed_initial_converges_to_rest`).
  - For each V, initialize from `initial_state` (+ the hardcoded resting concentrations `[Na_i=10, K_i=138, Ca_i=1e-4, Ca_ss=2e-4]`, passed to the V5.4 RHS).
  - At each step, call V5.4's TTP06 RHS with V clamped.
  - Per-dim relative convergence: stop when `max_i |dg_i/dt| / (1 + |g_i|) < rel_tol`, OR when `t >= max_t_ms`.
  - Record the final 14-observable slice (gate/RR/CaSR rows per `preprocessor.py:149-157` order).
  - Return dict also includes `'converged': bool` per voltage, so `test_z_ss_converged_early` can assert convergence rather than timeout.
- `def main()`: Runs with default V grid (rest only for MVP) and default `initial_state=None`; saves to `Surrogate/diagnostics/artifacts/z_ss_grid.pt`.
- TTP06 RHS: MUST use V5.4 primitives, no reimplementation of equations. V5.4's `TTP06Model` class (at `Monodomain/Engine_V5.4/cardiac_sim/ionic/ttp06/model.py`) does not export a single pure-RHS function, but it exports the primitives we need (verified with grep, 2026-04-20):
  - `compute_gate_steady_states(V, ionic_states)` → g_inf per HH gate
  - `compute_gate_time_constants(V, ionic_states)` → tau per HH gate
  - `compute_concentration_rates(V, ionic_states)` → dConc/dt
  - From these, the HH gate rate is the direct formula `dg_i/dt = (g_inf_i - g_i) / tau_i`. CaSR and RR are handled in `compute_concentration_rates` (or an equivalent calcium-handling method — verify at implementation time; see `calcium.py`).
  - The adapter `_ttp06_rhs_clamped(g_14, V) -> dg_14/dt` assembles these three helpers into one 14-dim rate call. **This is not equation re-implementation — it is a thin composition of V5.4 methods**. ~30 lines.
  - If V5.4's model class expects batched states of shape `(n_cells, n_states)` (likely, per `n_cells=1` default in `get_initial_state`), feed it a single-cell state: `ionic_states = g_14.unsqueeze(0)`. No mesh fixture required.
  - Do NOT copy RHS equations into `voltage_clamp_ss.py`. Do NOT finite-difference `step()` (it's Rush-Larsen, gives the WRONG rate).

#### Pseudocode
```python
import torch
from surrogate.model.stage1 import TTP06_REST_IONIC_STATE

def _ttp06_rhs_clamped(state_14, V):
    # Implement or import TTP06 rhs for 14 ionic-state observables only.
    # Takes (14,) state and scalar V; returns (14,) rate.
    ...

def compute_z_ss_grid(V_grid=(-85.23,), *, initial_state=None,
                      sim_dt_ms=0.01, max_t_ms=500.0, rel_tol=1e-6):
    start = (initial_state if initial_state is not None
             else TTP06_REST_IONIC_STATE.clone()).double()
    results, converged_flags = [], []
    for V in V_grid:
        g = start.clone()
        n_steps = int(max_t_ms / sim_dt_ms)
        converged = False
        for step in range(n_steps):
            dg = _ttp06_rhs_clamped(g, V)          # from Monodomain V5.4
            # Per-dim relative convergence (avoids uniform-threshold bias)
            if (dg.abs() / (1.0 + g.abs())).max().item() < rel_tol:
                converged = True
                break
            g = g + sim_dt_ms * dg
        results.append(g); converged_flags.append(converged)
    return {
        'V_grid': torch.tensor(list(V_grid), dtype=torch.float64),
        'z_ss_grid': torch.stack(results, dim=0),
        'converged': torch.tensor(converged_flags),
    }

def main():
    # MVP: rest only. Extend the V grid when Phase 2.5+ wires L_vclamp.
    out = compute_z_ss_grid([-85.23])
    out_path = 'Surrogate/diagnostics/artifacts/z_ss_grid.pt'
    torch.save(out, out_path)
    print(f"Saved {out_path}: V_grid {out['V_grid'].tolist()}, z_ss shape {tuple(out['z_ss_grid'].shape)}")

if __name__ == '__main__':
    main()
```

#### Test Spec
Add to `Surrogate/tests/test_voltage_clamp_ss.py`:

- `test_z_ss_from_perturbed_initial_converges_to_rest`: start from `TTP06_REST_IONIC_STATE + 0.1 * torch.randn(14, dtype=torch.float64)` (perturbed by ~10% of unit range), clamp V at -85.23, integrate via `compute_z_ss_grid`; assert the FINAL state is within 1e-3 of `TTP06_REST_IONIC_STATE`. This validates that the integrator actually RELAXES to rest (not just "does not drift when started at rest"). A broken RHS or wrong clamp would fail this; the identity-start test would not. NOTE: this requires exposing `compute_z_ss_grid` with an `initial_state` argument, or a helper that accepts it — extend the API.
- `test_z_ss_from_rest_is_stable`: identity check — start at rest, V_rest, assert final ≈ initial within 1e-4. Weak but cheap.
- `test_z_ss_dtype_shape`: out tensors are float64, shapes `(1,)` and `(1, 14)` for MVP.
- `test_z_ss_converged_early`: assert the solver returned before hitting `max_t_ms` (i.e., converged via tolerance) — guards against silent timeout returning a drifting-state artifact.
- `test_z_ss_uses_v54_rhs`: import `surrogate.data.voltage_clamp_ss as m`; assert `'cardiac_sim.ionic.ttp06' in sys.modules` (V5.4 import happened) AND assert that the module DOES NOT define any private symbol starting with `_TTP06_RHS_` or similar (a reinlined RHS would need its own function). Stronger than a grep because it checks import chain at runtime. If the adapter wrapper class name is known (e.g., `_TTP06Adapter`), assert it calls `TTP06Model` (or whatever the V5.4 class is named) using `unittest.mock.patch` and `assert_called_once`.

#### Checklist
- [ ] File created and imports `cardiac_sim.ionic.ttp06` (no standalone RHS)
- [ ] `compute_z_ss_grid` computes for rest voltage only
- [ ] Rest-voltage output matches `TTP06_REST_IONIC_STATE` to 1e-3
- [ ] Artifact cached at expected path
- [ ] Per-dim relative tolerance used, not uniform absolute

#### Verify
```
conda run -n heart-conduction python -m surrogate.data.voltage_clamp_ss
conda run -n heart-conduction python -c "
import torch
d = torch.load('Surrogate/diagnostics/artifacts/z_ss_grid.pt', weights_only=False)
assert d['V_grid'].shape == (1,) and d['z_ss_grid'].shape == (1, 14)
assert d['V_grid'].dtype == torch.float64 and d['z_ss_grid'].dtype == torch.float64
print('z_ss grid cached:', d['V_grid'].tolist())
"
```

#### Exit Criteria
- [ ] `z_ss_grid.pt` exists with schema `{V_grid: (1,), z_ss_grid: (1, 14)}`
- [ ] Rest voltage's steady state ≈ `TTP06_REST_IONIC_STATE` within 1e-3
- [ ] No standalone TTP06 RHS in the file (grep `def _ttp06_rhs` should only show one thin adapter, no equations inlined)

#### Risk
- Cross-import from `Monodomain/Engine_V5.4` may be awkward (different package root). Mitigation: add `Monodomain/Engine_V5.4/` to `sys.path` at the top of `voltage_clamp_ss.py`. Do NOT fall back to a hand-rolled RHS — see "MANDATORY import" in Phase 2 Context.
- If the V5.4 TTP06 class requires a mesh fixture, wrap with a dummy single-cell mesh. The overhead is one class instantiation.
- **Scope-justification note**: Step 2.1 is the diagnostic pre-req for a Phase 2.5 extension of `L_vclamp` beyond rest. In MVP (rest-only), the artifact IS technically unused by the training loss but IS used by `test_z_ss_from_perturbed_initial_converges_to_rest` (in the Test Spec) to validate the V5.4 TTP06 import worked correctly. That test alone justifies the step. If after Phase 3.2 the CaSR gap is closed without needing a full V-grid L_vclamp, disposition is: KEEP the file (as an integrator-correctness validation artifact) but DEFER extending the V grid. If CaSR gap persists, Phase 2.5 extends the grid and wires decoder pseudo-inverse. Either way, Step 2.1 as-scoped delivers a verified single-voltage steady-state point.

---

### Step 2.2: Add `L_rest` regularizer to `node_rollout`
**Model**: opus

#### Read First
- `Surrogate/surrogate/training/node_rollout.py:54-137` — current loss computation in `node_rollout`
- `Research/Active/surrogate_pipeline/ARCHITECTURE_v4.md` Sections 10.1, 11.2 — attractor formalism and per-phase loss

#### Why
Rest attractor gives the model an explicit training signal: `f_θ(z=0, V=V_rest) = 0`. With the decoder bias frozen at rest, z=0 decodes to rest; the regularizer anchors it as a fixed point of the dynamics. Directly addresses the Session 27 finding that v3 failed to treat z=0 as meaningful.

#### Implementation Spec
**Files to modify:** `Surrogate/surrogate/training/node_rollout.py`

At module level, add constants:
```python
V_REST_MV = -85.23
LAMBDA_REST = 1e-2
```

In `node_rollout` function, after `odeint` integration and before the final `result` return, add:

```python
# Rest attractor regularizer (Session 27 physics-informed)
# f_θ(z_rest, V_rest) should be zero. z_rest = [zeros(ionic_dim), INIT_CONC].
B = segment['Vm'].shape[0]
z_rest = torch.zeros(B, node.stage1.carried_dim, dtype=torch.float64, device=device)
z_rest[:, node.stage1.ionic_dim:] = INIT_CONC.to(device)
V_rest = torch.full((B,), V_REST_MV, dtype=torch.float64, device=device)
rate_at_rest = node.stage1.dzdt(z_rest, V_rest)
L_rest = rate_at_rest.pow(2).mean()

mean_loss = torch.stack(losses_per_eval).mean() + LAMBDA_REST * L_rest
```

Expose `L_rest` in the returned `result` dict:
```python
result['L_rest'] = L_rest.detach()
```

#### Pseudocode
```python
# node_rollout.py additions (symbolic):
V_REST_MV   = -85.23     # TTP06 resting Vm
LAMBDA_REST = 1e-2       # balance vs normalized ionic_state_mse (~0.01 at v3)

# Inside node_rollout(), after odeint(...):
B = segment['Vm'].shape[0]
z_rest = zeros(B, carried_dim, dtype=float64, device=device)
z_rest[:, ionic_dim:] = INIT_CONC.to(device)           # resting concentrations
V_rest_batch = full((B,), V_REST_MV, dtype=float64, device=device)
rate_at_rest = node.stage1.dzdt(z_rest, V_rest_batch)  # MUST be ≈ 0 for trained model
L_rest = rate_at_rest.pow(2).mean()

mean_loss = mean(stack(losses_per_eval)) + LAMBDA_REST * L_rest
result['L_rest'] = L_rest.detach()
return result
```

#### Test Spec
Add to `Surrogate/tests/test_training.py`:

- `test_L_rest_computed`: mini-batch rollout with `node.stage1.to(torch.float64)` explicit; verify `'L_rest' in result` and `torch.isfinite(result['L_rest'])`.
- `test_L_rest_float64_contract`: construct `z_rest` at `dtype=torch.float64` as `node_rollout` does; assert `rate_at_rest.dtype == torch.float64` — locks the contract that the rest-attractor code path operates on float64. Current `node_rollout.py` creates `z_rest` with explicit `dtype=torch.float64` (lines 99, 109), so this test passes unconditionally and prevents a later edit from silently demoting.
- `test_total_loss_contains_rest`: verify loss actually sums the rest term by comparing two runs. Run the rollout twice — once normally, once with `surrogate.training.node_rollout.LAMBDA_REST` monkey-patched to `0.0`. Assert `torch.isclose(result_normal['loss'] - result_zero_lambda['loss'], LAMBDA_REST_ORIGINAL * result_normal['L_rest'], rtol=1e-4)`. If the implementation drops the summation, `result_normal['loss'] == result_zero_lambda['loss']` and the assertion fails. This is the independent-recomputation check (compare-against-base) that a single-run algebraic identity cannot give you.
- `test_L_rest_goes_to_zero`: directly construct `z_rest` and `V_rest` as `node_rollout` does, call a STUB `stage1.dzdt` that returns `torch.zeros_like(z_rest)`, compute `L_rest = rate.pow(2).mean()`, assert `L_rest.item() == 0.0`. Test the L_rest computation isolatedly, NOT through the full `node_rollout` (avoids ambiguity about when to monkey-patch vs when odeint sees the patched version).
- `test_init_conc_not_leaky`: assert `INIT_CONC.requires_grad is False` — pre-condition; if the constant ever gets accidentally promoted to a parameter, `L_rest.backward()` would update it.

#### Checklist
- [ ] `V_REST_MV` and `LAMBDA_REST` module constants added
- [ ] `z_rest` constructed per-batch (correct device, dtype, float64 explicit)
- [ ] `L_rest` computed via `stage1.dzdt`
- [ ] Total loss includes `LAMBDA_REST * L_rest`
- [ ] `L_rest` returned in result dict
- [ ] All five tests pass

#### Verify
```
conda run -n heart-conduction pytest Surrogate/tests/test_training.py -v -k "rest or L_rest or rollout"
```

#### Exit Criteria
- [ ] `L_rest` appears in training output
- [ ] Total loss is finite, positive
- [ ] Existing rollout tests pass

#### Risk
- Gradient leakage into `INIT_CONC` — `INIT_CONC` is a module-level constant tensor without `requires_grad`. Verify via `INIT_CONC.requires_grad == False` in a test.
- `L_rest` unit: it's raw MSE (not normalized). `LAMBDA_REST = 1e-2` is the starting value to balance against normalized ionic_state_mse (~0.01 at v3 convergence). Tunable via training logs.

---

### Phase 2 Verification

```
conda run -n heart-conduction python -m surrogate.data.voltage_clamp_ss
conda run -n heart-conduction pytest Surrogate/tests/test_training.py -v -k "rest or rollout"
```

Smoke test (one epoch of training with a tiny dataset):
```
conda run -n heart-conduction python -c "
# Minimal rollout smoke test: confirms L_rest is finite and contributes to loss.
import torch, sys
sys.path.insert(0, 'Surrogate')
from surrogate.model.stage1 import IonicStage1
from surrogate.model.node import IonicNODE
from surrogate.training.node_rollout import node_rollout
m = IonicStage1(scaffold=True).to(torch.float64).cuda()
node = IonicNODE(m).cuda()
seg = {
    'Vm': torch.full((2, 300), -85.23, dtype=torch.float64).cuda(),
    'dt': torch.full((2, 300), 0.01, dtype=torch.float64).cuda(),
    'ionic_states': torch.zeros(2, 300, 14, dtype=torch.float64).cuda(),
    'concentrations': torch.zeros(2, 300, 4, dtype=torch.float64).cuda(),
    'conductance_products': torch.zeros(2, 300, 5, dtype=torch.float64).cuda(),
}
result = node_rollout(node, seg, phase_name='A1', device=torch.device('cuda'))
print('loss:', result['loss'].item(), 'L_rest:', result['L_rest'].item())
assert torch.isfinite(result['loss']) and torch.isfinite(result['L_rest'])
"
```

### Phase 2 Exit Criteria

- [ ] `z_ss_grid.pt` cached (1 × 14 — MVP rest-only grid)
- [ ] `L_rest` term in rollout loss
- [ ] New tests pass, no regressions
- [ ] Smoke test prints finite `L_rest`

### Phase 2 Cleanup

- [ ] `LAMBDA_REST = 1e-2` is a named constant (not a magic literal scattered in loss code)
- [ ] `V_REST_MV = -85.23` is a named constant
- [ ] No float32 leaks in `voltage_clamp_ss.py` — all tensors float64
- [ ] If standalone TTP06 rhs was implemented in `voltage_clamp_ss.py`, add a comment linking to the TTP06 paper or `Monodomain/Engine_V5.4` source for cross-reference
- [ ] V5.3 untouched

**→ Commit point: `git commit` after Phase 2 passes. Suggested message: "Surrogate v4: rest-attractor regularizer + z_ss grid precomputation"**

---

## Phase 3: Phase A Training Validation (via cardiac_ml harness)

**Goal**: Train the v4 architecture from scratch on T1 multi-BCL via the **cardiac_ml Hydra+MLflow harness** (`scripts/train.py +experiment=ionic_node_t1`). Re-run the Session 27 integrator error-budget diagnostic against the new checkpoint. Confirm CaSR NRMSE improves from v3's 27.4% baseline toward the <20% success-criterion target.

**Tier**: medium
**Estimated scope**: no code changes (harness exists, model v4 instantiates via existing factory). ~15 min–1 hour compute depending on epoch count.

### Phase 3 Context

- **Training is invoked through `cardiac_ml`.** The oracle is `run_multi_bcl.py` (Session 25 multi_bcl_002: 8 epochs, val=0.00838 at ep 6). `train_node.py`'s CLI path is the archived pre-harness entry and is NOT the oracle — do not fall back to it. See MEMORY.md "Session 25 parity oracle is `Surrogate/run_multi_bcl.py`".
- **Entry point**: `conda run -n heart-conduction python scripts/train.py +experiment=ionic_node_t1` from repo root.
- **Config composition** (set by `conf/experiment/ionic_node_t1.yaml`):
  - model: `cardiac_ml.model.ionic_node_factory.make_node` → `IonicNODE(IonicStage1(scaffold=True))`. v4 model is picked up automatically since `IonicStage1` now IS v4.
  - data: `cardiac_ml.data.multi_bcl_loader` on T1 cache. Train BCLs `{300,500,700,1000,1500}`, val BCLs `{400,600,800,2000}`, last 5/3 beats each.
  - training: `conf/training/node.yaml` — 30 epochs, AdamW lr=1e-4 wd=1e-4, dopri5 rtol=atol=1e-3, adjoint=False, phase_name=A1.
  - step functions: `surrogate.training.node_step.{node_train_step,node_val_step}` — the harness-compatible adapter around `node_rollout`.
- **Cache env vars**: the data config reads `SURROGATE_T1_CACHE_TRAIN` / `SURROGATE_T1_CACHE_VAL` with `/tmp/surrogate_cache/tier01_{train,val}.pt` defaults. Build cache first if absent (see Step 3.1 prereq).
- **Outputs**: Hydra working dir `outputs/YYYY-MM-DD/HH-MM-SS/` holds per-run state_dict checkpoints (`best.pt`, `final.pt`). MLflow `mlruns/` duplicates the checkpoints as artifacts and logs per-epoch metrics. Checkpoints via the `ModelCheckpoint` callback.
- **v4 parity caveat**: the oracle `val=0.00838 at ep 6` was measured against the v3 model tree. v4 (3.6× params, fresh init) may converge more slowly; aim for `val < 0.02` at 30 epochs as the v4 smoke threshold, not the 0.0088 oracle. The oracle's purpose was *harness validation* (already done by cardiac_ml's own tests), not v4-arch validation.
- **Divergence criteria (abort)**: val_loss > 10 at epoch 3 or any NaN. Flash test on 2026-04-19 showed an explosion to 1e90 when a gated branch is initialized at `α = 0.5`; that's been fixed by `skip_logit = BETA_INIT` (see Phase 1 Step 1.3). If divergence recurs, suspect a new gated init regression and inspect `sigmoid(skip_logit)` at the start of training.

### Step 3.0: Patch `ionic_node_factory.make_node` to call `pin_rest_bias()` after warm-start
**Model**: opus

#### Read First
- `cardiac_ml/model/ionic_node_factory.py` — current factory body is in `make_node()` around lines 24–50 (file total ~51 lines, verify with `wc -l`)
- `Surrogate/surrogate/model/stage1.py` — `pin_rest_bias` docstring warning

#### Why
The factory calls `stage1.load_state_dict(ckpt["stage1_state_dict"])` and returns immediately. `pin_rest_bias`'s docstring explicitly warns: "CALL AFTER load_state_dict() — torch.nn.Module.load_state_dict overwrites the bias with the checkpoint value, silently breaking the rest-attractor guarantee". Any warm-started training — including resuming from a v3 ckpt or an earlier v4 run — silently corrupts the rest anchor, invalidates `L_rest` training signal (which assumes `decoder(0) = rest`), and bakes noise into the attractor term before v4 even trains. Cold-start from-scratch training is safe (no load) but the whole point of the WARM_START_CKPT env var is to enable warm starts, which are currently broken.

#### Implementation Spec
**Files to modify:** `cardiac_ml/model/ionic_node_factory.py` — body of `make_node`.

Add a single call after the `load_state_dict` block:
```python
# Post-load rest-attractor pin (Session 27, PLAN Step 1.5). load_state_dict
# overwrites the frozen rest bias; re-pin before returning. Idempotent; no-op
# on models without a scaffold decoder.
if hasattr(stage1, "pin_rest_bias"):
    stage1.pin_rest_bias()
```

This applies to BOTH warm-start and cold-start paths (since cold-start `IonicStage1(scaffold=True)` already pins in `_init_weights`, the second call is a no-op).

#### Pseudocode
Diff against the existing factory (preserves the FileNotFoundError + KeyError paths already in the code — see `ionic_node_factory.py`):

```diff
   def make_node(scaffold=True, stage1_ckpt=None):
       stage1 = IonicStage1(scaffold=scaffold)
       if stage1_ckpt and str(stage1_ckpt).lower() != "null":
           path = Path(stage1_ckpt)
           if not path.is_file():
               raise FileNotFoundError(f"stage1_ckpt not found: {path}...")
           ckpt = torch.load(path, weights_only=False, map_location="cpu")
           if "stage1_state_dict" not in ckpt:
               raise KeyError(f"warm-start ckpt {path} missing 'stage1_state_dict' key...")
           stage1.load_state_dict(ckpt["stage1_state_dict"])
+      # Post-load rest-attractor pin (Session 27). load_state_dict overwrites the
+      # frozen rest bias; re-pin BEFORE returning so the attractor contract holds
+      # on both warm-start and cold-start (cold-start call is a no-op).
+      if hasattr(stage1, "pin_rest_bias"):
+          stage1.pin_rest_bias()
       return IonicNODE(stage1)
```

Apply as a literal insertion after the `load_state_dict` block, BEFORE `return IonicNODE(stage1)`. Do not rewrite the surrounding error handling.

#### Test Spec
Add to `cardiac_ml/tests/` (new file `test_factory_rest_bias.py` or extend existing node-factory test):

- `test_factory_pins_rest_bias_cold_start`: call `make_node()` without warm-start; assert `decoder.bias.requires_grad is False` AND `decoder.bias` matches `TTP06_REST_IONIC_STATE`.
- `test_factory_pins_rest_bias_after_warm_start`: craft a ckpt in the v3 wrapper format the factory expects — `torch.save({"stage1_state_dict": stage1_sd}, tmp_ckpt)` where `stage1_sd` contains a random 14-element bias — NOT a flat state dict. Call `make_node(stage1_ckpt=tmp_ckpt)`; assert bias is restored to `TTP06_REST_IONIC_STATE` and frozen. This is the critical test — without the patch, this test fails. (Note: the flat `trainer.model.state_dict()` format from cardiac_ml's `ModelCheckpoint` is NOT what the current factory accepts; warm-starting from that format is out of scope for Step 3.0 and tracked as a follow-up.)

#### Checklist
- [ ] `pin_rest_bias()` call added after `load_state_dict` in factory
- [ ] Both tests pass; warm-start test FAILS without the patch (sanity-check the test itself by reverting the patch locally, running, confirming failure, reapplying)

#### Verify
```
conda run -n heart-conduction pytest cardiac_ml/tests/test_factory_rest_bias.py -v
```

#### Exit Criteria
- [ ] Warm-started model has `decoder.bias == TTP06_REST_IONIC_STATE` and `requires_grad is False`

#### Risk
- The flat-state-dict format saved by `cardiac_ml.training.callbacks.ModelCheckpoint` is the `IonicNODE` top-level dict, not wrapped in `stage1_state_dict`. Warm-starting from one of THOSE checkpoints via the current factory would fail with `KeyError('stage1_state_dict')` — independent issue. Out of scope for Step 3.0; file as follow-up if warm-start-from-v4 is ever needed.

---

### Step 3.1: Phase A training run via Hydra
**Model**: opus (monitoring)

#### Read First
- `scripts/train.py` — Hydra entry, thin wrapper around `Trainer(cfg).fit()`
- `conf/experiment/ionic_node_t1.yaml` — the composed recipe
- `conf/training/node.yaml` — epochs, optimizer, callbacks, ODE solver config
- `Surrogate/surrogate/training/node_step.py` — adapter binding `node_rollout` to Trainer protocol
- MEMORY.md "Session 25 multi_bcl_002 oracle" entry for exact parity numbers

#### Why
First end-to-end training pass on v4 through the new harness. Validates that (a) v4 model instantiates and integrates correctly through the cardiac_ml factory, (b) training converges monotonically, and (c) checkpoint artifacts land in a known location for Step 3.2. Smoke before the CaSR diagnostic.

#### Implementation Spec

Prereq — ensure T1 cache exists. **Cache path caveat**: `data_cache.py:86` has an outdated default `/media/HDD/surrogate_data/raw` which does NOT exist. The real path is `/media/HDD/norepinephrine/surrogate_data/raw` (verified 2026-04-19; see MEMORY.md). Always pass `raw_dir` explicitly.

**Cache dtype caveat**: `data_cache.py:137` writes the cache as `float32` via `torch.cat(v, dim=0).float()`. Every downstream consumer (`datasets.py`, `node_rollout.py`, the factory) then `.double()`s back to float64 at use time. This is silently violating the project's float64 default, wasting cache disk size (~2×), and forcing a GPU-side upcast on every batch. Not a blocker for Phase 3 Step 3.1 (the existing cache predates this PLAN), but:
  - Add a follow-up plan step to fix `data_cache.py:137` to `.double()` or leave as the raw-read dtype (which is already float64 from HDF5).
  - When rebuilding the cache per the prereq script below, the cache will still be float32; the v4 smoke will still work because downstream code upcasts. Document this in the Mutation Log and revisit post-Phase 3.

```
conda run -n heart-conduction python -c "
from pathlib import Path
p = Path('/tmp/surrogate_cache/tier01_train.pt')
if not p.exists():
    import sys; sys.path.insert(0, 'Surrogate')
    from surrogate.training.data_cache import CacheBuilder
    CacheBuilder(raw_dir='/media/HDD/norepinephrine/surrogate_data/raw',
                 cache_dir='/tmp/surrogate_cache').build_tier_cache(1)
print('cache ready')
"
```

Launch training:
```
conda run -n heart-conduction python scripts/train.py +experiment=ionic_node_t1
```

**LR/epoch policy — choose ONE starting point, lock it.** `conf/training/node.yaml:16` specifies `lr=1e-4` (matches oracle multi_bcl_002 epoch-0 value 9.997e-05 via cosine decay). **Use the oracle's 1e-4 as the starting point for v4, unmodified.** Previous sections that mentioned 5e-4 as an override are now superseded — do not run that first. If by epoch 10 `val_loss` is still > 0.1, THEN override via `training.optimizer.lr=3e-4` for a retry. Do not chase aggressive LR; the v3→v4 capacity jump already slows per-step effectiveness.

Override knob syntax (verify before issuing — Hydra's `@package _global_` experiment composition puts training config under the `training` key): `training.optimizer.lr=...`, `training.epochs=...`, `training.phase_name=...`. Dump the composed config with `--cfg job` to confirm key layout if uncertain:
```
conda run -n heart-conduction python scripts/train.py +experiment=ionic_node_t1 --cfg job | head -40
```

For a quick 2-epoch smoke before committing to the full run:
```
conda run -n heart-conduction python scripts/train.py +experiment=ionic_node_smoke
```

Monitor: Hydra per-run log at `outputs/{date}/{time}/train.log`. MLflow UI is a manual *monitoring aid*, NOT a verification step — launched in a separate terminal if desired:
```
mlflow ui --backend-store-uri ./mlruns --port 5001
```

Watch per epoch (metrics surfaced by cardiac_ml callbacks — verified against `cardiac_ml/training/callbacks.py`):
- `val_loss` — non-increasing trend, final < 0.02 (v4 threshold; not the 0.0088 oracle)
- `grad_norm` — bounded, shrinking (emitted by `GradNormMonitor` callback already in `conf/training/node.yaml:33`)

`nfe_mean` is NOT surfaced by the harness (no NFE counter callback exists in `cardiac_ml/training/callbacks.py`). If NFE tracking is needed for stiffness diagnosis, either (a) attach a custom callback that reads `trainer.model.nfe` (the `IonicNODE.nfe` attribute added 2026-04-19 in `node.py`) on `on_epoch_end`, or (b) skip it — NFE impacts wall time and can be inferred from epoch duration. Do not pretend the metric is reported when it isn't.

If divergence (val_loss > 10 at ep 3 or NaN): abort, reduce LR override to `training.optimizer.lr=3e-5`, retry.

**DO NOT use** `python -m surrogate.training.train_node ...`. The argparse CLI at `Surrogate/surrogate/training/train_node.py` is the pre-harness entry; it does NOT route through MLflow, does NOT use the multi-BCL loader, and bypasses the cardiac_ml callback suite. File has not been physically moved to `archive/` yet (follow-up cleanup) but is considered deprecated as of 2026-04-20.

#### Pseudocode
```python
# Effectively (inside cardiac_ml.Trainer.fit()):
for epoch in range(cfg.training.epochs):
    for batch in train_loader:
        batch = _to_device_and_dtype(batch, device, dtype)   # float64
        result = train_step_fn(trainer, batch)                # node_train_step
        if not result.get("_backward_done"):
            result["loss"].backward()
        if cb := result.get("_on_after_backward"):            # clear_v_trajectory
            cb()
        optimizer.step(); optimizer.zero_grad()
        mlflow.log_metric(...)
    # val pass identical but under torch.no_grad
    callbacks.on_epoch_end(epoch, trainer)  # saves best.pt, early-stop check
```

#### Test Spec
- **Verify first, then reuse**: `grep -l 'test_ionic_node' cardiac_ml/tests/*.py` to find existing NODE end-to-end coverage. cardiac_ml landed with `test_end_to_end.py` covering `test_synthetic_end_to_end` (synthetic-MLP path, not NODE); there is NO pre-existing `test_ionic_node_smoke_e2e`. If one exists after the harness is updated, use it; otherwise add a new one mirroring `test_synthetic_end_to_end` but with `+experiment=ionic_node_smoke`.
- NEW `test_v4_stage1_loads_via_factory` (in `cardiac_ml/tests/test_node_configs.py` or `test_end_to_end.py`): call `cardiac_ml.model.ionic_node_factory.make_node(scaffold=True)`; assert the returned IonicNODE wraps an IonicStage1 with `ionic_dim=20`, has a `state_rate_mlp` attribute, and `decoder.bias.requires_grad is False`. Guards against the factory silently regressing to v3 dims OR dropping the Step 3.0 rest-bias-pin.

#### Checklist
- [ ] T1 cache present at `/tmp/surrogate_cache/tier01_{train,val}.pt`
- [ ] `scripts/train.py +experiment=ionic_node_t1` launches without error
- [ ] Training completes ≥ 10 epochs without NaN
- [ ] `best.pt` exists in `outputs/{date}/{time}/`
- [ ] MLflow run visible under `mlruns/`
- [ ] Final `val_loss` < 0.02
- [ ] LR locked at 1e-4 unless Phase 3.1 checklist shows loss plateau by ep 10

#### Verify
```
conda run -n heart-conduction python -c "
import glob
run_dir = sorted(glob.glob('outputs/*/*'))[-1]
log = f'{run_dir}/train.log'
print(f'Run dir: {run_dir}')
try:
    print(chr(10).join(open(log).read().splitlines()[-10:]))
except FileNotFoundError:
    print('train.log not present — check Hydra output layout (cardiac_ml may use app.log or similar)')
"
```

#### Exit Criteria
- [ ] Training completed ≥ 10 epochs
- [ ] `val_loss` monotonically decreasing (transient spikes allowed)
- [ ] `best.pt` saved in Hydra working dir
- [ ] MLflow tags include git SHA + dirty flag

#### Risk
- v4's larger model may need more epochs than the 30-epoch oracle budget. Mitigation: inspect loss trend at ep 30; if still dropping fast, bump `training.epochs=60` and resume via `WARM_START_CKPT`.
- Hydra working-dir lookup from scripts may be brittle; explicit run-dir via `hydra.run.dir=runs/v4_A1` override if cleaner.
- Hydra `.log` file name: the default is `train.log` (matches `@hydra.main` decorator pattern); if cardiac_ml's `config.yaml` overrode to `hydra.job.name` or similar, the tail path differs. Check `outputs/{date}/{time}/*.log` if `train.log` is missing.

---

### Step 3.2: Re-run integrator error-budget diagnostic
**Model**: opus

#### Read First
- `Surrogate/diagnostics/integrator_error_budget.py` — diagnostic script (path constant is at the top; loader is at `stage1.load_state_dict(ckpt["stage1_state_dict"])` — this assumes the v3-era wrapper format)
- Session 27 baseline values: CaSR NRMSE 27.4%, gate median 11% (KNOWLEDGE.md §3c)
- `cardiac_ml/training/callbacks.py:118` `ModelCheckpoint` — saves `trainer.model.state_dict()` DIRECTLY (where `trainer.model` is the `IonicNODE`). Resulting checkpoint is a flat dict with keys prefixed `stage1.*`, NO `stage1_state_dict` wrapper. The diagnostic's current loader will KeyError without adaptation.

#### Why
Direct empirical test: does v4 close the CaSR capacity gap? Re-using the held-out trajectory makes v3 → v4 comparison unambiguous.

#### Implementation Spec

**Files to modify:** `Surrogate/diagnostics/integrator_error_budget.py`

Two changes:

1. **Checkpoint path** (line 19) — point at the latest Hydra run:
```python
import glob
_CANDIDATES = sorted(glob.glob(str(REPO / "outputs/*/*/best.pt")))
if not _CANDIDATES:
    raise FileNotFoundError("No Hydra output found. Run Step 3.1 first.")
CKPT = Path(_CANDIDATES[-1])
```

2. **Loader adaptation** (around line 108) — handle both checkpoint formats AND guard the print statements that reference wrapper-only keys (`ckpt.get("epoch")`, `ckpt.get("val_loss")`). Cardiac_ml flat dicts don't have these keys; `f"{None:.4f}"` raises `TypeError`. Fix in one block:
```python
ckpt = torch.load(CKPT, weights_only=False, map_location=device)
epoch = ckpt.get("epoch", "unknown")
val_loss = ckpt.get("val_loss")
val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else "unknown"
print(f"Loaded checkpoint at epoch {epoch}, val_loss={val_loss_str}")
# -- also replace the existing print line at integrator_error_budget.py:111 accordingly --

if "stage1_state_dict" in ckpt:
    # Legacy v3 format (run_multi_bcl.py, old train_node.py)
    stage1.load_state_dict(ckpt["stage1_state_dict"])
else:
    # cardiac_ml harness: flat IonicNODE state dict, strip "stage1." prefix
    stage1_sd = {
        k[len("stage1."):]: v
        for k, v in ckpt.items()
        if k.startswith("stage1.")
    }
    assert stage1_sd, f"Checkpoint at {CKPT} has no stage1.* keys; format unknown"
    stage1.load_state_dict(stage1_sd)
# Physics-informed invariant: re-pin decoder bias AFTER load (load_state_dict
# overwrites the frozen rest bias — see stage1.py:pin_rest_bias docstring)
if hasattr(stage1, "pin_rest_bias"):
    stage1.pin_rest_bias()
```

3. **Artifact save path** — avoid overwriting v3 baseline:
```python
torch.save({...}, out_dir / "integrator_error_budget_v4.pt")
```

4. **Add NRMSE (normalized RMSE) output.** The current script only computes RAW RMSE via `rmse()` and `per_dim_rmse()` at lines 80–84 (verified by grep). Success Criteria specify **NRMSE < 20%**, which requires normalization by physiological range. Add this block after the per-dim RMSE table:

```python
from surrogate.training.loss_normalization import _RANGES
_R = _RANGES['ionic_states']
_range_denom = (_R['max'] - _R['min']).to(ionic_true.device)      # (14,)
nrmse_euler_truth = per_dim_rmse(ionic_euler, ionic_true) / _range_denom
nrmse_dopri_truth = per_dim_rmse(ionic_dopri, ionic_true) / _range_denom
print()
print("=== NRMSE (% of physiological range) — compare to Session 27 v3 baseline ===")
for i, name in enumerate(lbl):
    print(f"{name:>6}  Euler-Truth  {100*nrmse_euler_truth[i]:7.2f}%   "
          f"dopri5-Truth  {100*nrmse_dopri_truth[i]:7.2f}%")
print(f"  CaSR target: < 20% (v3 baseline: 27.4%)")
```

Run:
```
conda run -n heart-conduction python Surrogate/diagnostics/integrator_error_budget.py
```

Inspect stdout for per-dim **NRMSE** table — focus on CaSR row (v3 baseline: 27.4%, v4 target: <20%).

#### Pseudocode
```python
# integrator_error_budget.py (diff vs current)
# (1) Find latest Hydra checkpoint
CKPT = Path(sorted(glob.glob(str(REPO / "outputs/*/*/best.pt")))[-1])

# (2) Dual-format loader
ckpt = torch.load(CKPT, weights_only=False, map_location=device)
stage1 = IonicStage1(scaffold=True).double().to(device)
if "stage1_state_dict" in ckpt:
    stage1.load_state_dict(ckpt["stage1_state_dict"])            # v3 wrapper
else:
    stage1_sd = {k[len("stage1."):]: v
                 for k, v in ckpt.items() if k.startswith("stage1.")}
    stage1.load_state_dict(stage1_sd)                             # cardiac_ml flat
stage1.pin_rest_bias()                                            # restore frozen bias

# (3) rest of script unchanged (integrate Euler/dopri5, compute per-dim NRMSE)

# (4) Save artifact under v4 suffix
torch.save({...}, out_dir / "integrator_error_budget_v4.pt")
```

#### Test Spec
- `test_diagnostic_loads_cardiac_ml_ckpt`: create a minimal `IonicNODE` fixture, save its state dict via `torch.save(node.state_dict(), tmp_path)` (cardiac_ml format), run the loader path from the diagnostic, assert stage1 weights round-trip without error.
- `test_diagnostic_loads_legacy_ckpt`: same test but save as `{"stage1_state_dict": stage1.state_dict()}` (v3 format); assert legacy loader branch is taken and works.
- `test_diagnostic_pin_rest_bias_after_load`: after loading a checkpoint whose bias differs from rest, assert the diagnostic's re-pin call restores `TTP06_REST_IONIC_STATE` (prevents silent regression of the rest-attractor contract).

#### Checklist
- [ ] `CKPT` glob updated to `outputs/*/*/best.pt`
- [ ] Dual-format loader implemented (both `stage1_state_dict` wrapper and flat `stage1.*` prefix cases)
- [ ] `pin_rest_bias()` called after load
- [ ] Artifact output path updated (`_v4` suffix)
- [ ] Diagnostic runs without error
- [ ] Per-dim NRMSE table printed

#### Verify
Output CaSR row reads `CaSR: X%` where X is the per-dim normalized RMSE. Target: X < 20. Excellent: X < 15.

#### Exit Criteria
- [ ] Diagnostic completes on v4 checkpoint
- [ ] Per-dim table printed
- [ ] `integrator_error_budget_v4.pt` saved

#### Risk
- If CaSR NRMSE ≥ 20%, the rest-attractor alone was insufficient. Escalate to Tier-2 attractors (contraction-toward-target, gate bounds) or the hybrid explicit-slow-variable approach (ARCHITECTURE_v4.md §12.3). Document and raise as a follow-up.
- If per-dim gate error INCREASED from v3, something is wrong with the training — investigate before declaring success.

---

### Step 3.3: Document results in IDEALOG and KNOWLEDGE
**Model**: opus

#### Read First
- `Research/Active/surrogate_pipeline/IDEALOG.md` — Current Direction + Session 27 entry
- `Research/Active/surrogate_pipeline/KNOWLEDGE.md` §3c — Model Capacity Diagnostic

#### Why
Close the Session 27 loop: design was motivated by the diagnostic; validation results belong with the design for future readers. Also updates the Current Direction with post-implementation state.

#### Implementation Spec
**Files to modify:**
- `IDEALOG.md` — add a new top-of-Thread entry (Session 28) with: (a) v4 implementation completion, (b) Phase A training summary (epochs, val_loss), (c) diagnostic CaSR NRMSE delta from v3 baseline.
- `KNOWLEDGE.md` §3c — add a "Post-v4 baseline" row to the per-dim NRMSE table so v3 and v4 sit side-by-side.
- `IDEALOG.md` Current Direction — update to reflect v4 validation status: "done" or "partial" or "failed".

Content template for IDEALOG Session 28 (replace the `YYYY-MM-DD` placeholder with the actual date returned by `date +%F` at write time — do not commit the literal placeholder):
```markdown
### YYYY-MM-DD (Session 28): v4 validation

Completed v4 architecture + rest attractor per PLAN Session 27.
Phase A1 trained on T1 multi-BCL via cardiac_ml harness (scripts/train.py +experiment=ionic_node_t1).
N epochs, batch 1 (multi_bcl_loader default), LR 1e-4.
Final val_loss: X.XXXX (v3 multi_bcl_002 baseline: 0.00838 at epoch 6).
Integrator diagnostic on held-out BCL=2000:
- CaSR NRMSE: Y.YY% (v3 baseline: 27.4%)
- Gates median NRMSE: Z.ZZ% (v3 baseline: ~11%)
Gap [closed / partially closed / did not close]: {interpretation}

Next step: [optional Phase B, or iterate on arch if CaSR still > 20%].
```

#### Pseudocode
(Pure documentation; no code.)

#### Test Spec
No unit tests — this is a documentation step. Post-hoc consistency check:
- `grep -c "Session 28" Research/Active/surrogate_pipeline/IDEALOG.md` returns ≥ 1 (entry exists)
- `grep -c "v4 baseline" Research/Active/surrogate_pipeline/KNOWLEDGE.md` returns ≥ 1 (§3c row added)

#### Checklist
- [ ] IDEALOG Session 28 entry written with numbers from Step 3.1 and 3.2
- [ ] IDEALOG Current Direction updated (done/partial/failed)
- [ ] KNOWLEDGE §3c updated with v4 numbers alongside v3 baseline

#### Verify
```
grep -c "Session 28" Research/Active/surrogate_pipeline/IDEALOG.md
grep -c "v4 baseline" Research/Active/surrogate_pipeline/KNOWLEDGE.md
```

#### Exit Criteria
- [ ] Both documents updated
- [ ] Session 27's "pending implementation" status resolved
- [ ] Grep counts both return ≥ 1

#### Risk
None.

---

### Phase 3 Verification

Shell block — run from repo root. Each `conda run` subcommand is self-contained; the final `ls -lh` is a bare shell command:

```
# 1. Locate the most recent Hydra run (produced by scripts/train.py).
conda run -n heart-conduction python -c "
import glob, json
run = sorted(glob.glob('outputs/*/*'))[-1]
print(f'Run dir: {run}')
# train.log is a plain-text log; MLflow run data is in mlruns/. Show tail:
try:
    tail = open(f'{run}/train.log').read().splitlines()[-20:]
    print(chr(10).join(tail))
except FileNotFoundError:
    print('train.log not found — check Hydra working dir layout')
"

# 2. Confirm best.pt exists in that run dir:
conda run -n heart-conduction python -c "
import glob; from pathlib import Path
ckpt = sorted(glob.glob('outputs/*/*/best.pt'))[-1]
assert Path(ckpt).stat().st_size > 0, f'empty ckpt {ckpt}'
print('ckpt ok:', ckpt)
"

# 3. Diagnostic artifact saved:
ls -lh Surrogate/diagnostics/artifacts/integrator_error_budget_v4.pt
```

### Phase 3 Exit Criteria

- [ ] Training completed ≥ 10 epochs without NaN (verify via MLflow metrics or train.log tail)
- [ ] Final `val_loss` < 0.02
- [ ] `best.pt` exists in `outputs/{date}/{time}/`
- [ ] CaSR NRMSE < 20% OR documented next-step if > 20%
- [ ] IDEALOG and KNOWLEDGE updated with post-v4 results

### Phase 3 Cleanup

- [ ] `best.pt` retained in the Hydra working dir (`outputs/*/*/`). Verify `outputs/` is already in `.gitignore` (it should be — cardiac_ml landed with that ignore). MLflow's `mlruns/best.pt` copy is also gitignored.
- [ ] `integrator_error_budget_v4.pt` artifact preserved (in `Surrogate/diagnostics/artifacts/`, NOT in `outputs/`)
- [ ] `train.log` inside the Hydra run dir retained for debugging
- [ ] V5.3 untouched

**→ Commit point: `git commit` after Phase 3 passes. Suggested message: "Surrogate v4: Phase A trained; CaSR NRMSE X% (v3 baseline: 27.4%)"**

---

## Final Cleanup

- [ ] float64 consistency — `grep -n "\.float()\|torch.float32" Surrogate/surrogate/` is expected to return the following hits only (all pre-existing, not introduced by Session 27):
  - `data/storage.py:170` — HDF5 read buffering (legitimate)
  - `model/nernst.py:58,70,100` — Nernst equation constants (legitimate, scalar constants in physics module)
  - `training/shard_loader.py:67` — HDD streamer (legitimate)
  - `training/data_cache.py:137` — **KNOWN BUG** (cache write in float32; Step 3.1 flags this as a follow-up; do not fix in Session 27)

  Any NEW hit in `Surrogate/surrogate/model/stage1.py`, `node.py`, `ionic_surrogate_v3.py`, or new files like `data/voltage_clamp_ss.py` is a regression and must be fixed before completing Phase 1/2.
- [ ] V5.3 not modified — `git status Monodomain/Engine_V5.3/` is clean
- [ ] No code duplication across engines — `StateRateMLP` is Surrogate-specific by design; no `cardiac_core/` extraction needed
- [ ] `train_node.py` deprecation decision: either (a) physically move `Surrogate/surrogate/training/train_node.py` to `Surrogate/surrogate/training/archive/train_node.py` with a stub redirect, or (b) add a top-of-file `DeprecationWarning` + banner in the docstring pointing at `scripts/train.py +experiment=ionic_node_t1`. Option (a) preferred. Do not leave the file as-is with the PLAN claiming it is "archived".
- [ ] Archive the completed plan. Use `cp` (not `mv`) so `PLAN.md` remains canonical at its live location; the archived copy is an immutable historical snapshot:
  ```
  mkdir -p Research/Active/surrogate_pipeline/plans
  cp Research/Active/surrogate_pipeline/PLAN.md "Research/Active/surrogate_pipeline/plans/$(date +%Y-%m-%d)_session-27-state-rate-mlp-physics-attractors.md"
  ```
- [ ] Backlinks: PLAN header (line 5) already references `README.md` (the research question's README). PLAN lives inside `Research/Active/surrogate_pipeline/` so the research-question backlink is implicit via filesystem location. No separate EXPERIMENT.md is needed for this plan (no new engine experiments are created by Phase 1–3; the diagnostic already exists at `Surrogate/diagnostics/`). If Phase B+ later creates an experiment folder in an engine, that folder's EXPERIMENT.md must backlink here.

---

## Mutation Log

- **MUTATED 2026-04-20**: Phase 3 Step 3.1 REWRITTEN — training entry point changed from `python -m surrogate.training.train_node --phase A1 ...` to `python scripts/train.py +experiment=ionic_node_t1` via the cardiac_ml Hydra+MLflow harness (landed 2026-04-20). Added Harness section near plan top. Oracle clarified: `run_multi_bcl.py` (val=0.00838 at ep 6), not `train_node.py`. v4 smoke threshold relaxed to `val_loss < 0.02` at 30 epochs — the 0.0088 oracle is a harness-validation target already met by cardiac_ml's own tests on v3, not a v4-arch requirement.
- **MUTATED 2026-04-20**: Phase 3 Step 3.2 path update — diagnostic checkpoint glob now points at `outputs/*/*/best.pt` (Hydra working dir) instead of `Surrogate/runs/*/best_A1.pt`.
- **MUTATED 2026-04-20**: Flash test insight added to Phase 3 Context — gated branch `α = 0.5` at init caused 1e90 divergence; `skip_logit = BETA_INIT` in Phase 1 Step 1.3 is the fix.
- **MUTATED 2026-04-20 (audit round 1)**: 20 issues identified, 16 mutation bullets below (several bullets bundle related issues, e.g., HIGH #3 "Test Spec section missing" covers 6 steps in one bullet).
  - CRITICAL: Step 3.2 checkpoint loader now handles BOTH `stage1_state_dict` wrapper (legacy) AND flat `stage1.*` prefix (cardiac_ml `ModelCheckpoint`). Re-pins rest bias after load.
  - CRITICAL: Step 1.3 Implementation Spec `skip_logit` init corrected to `torch.full((carried_dim,), BETA_INIT)` — was `torch.zeros` which caused 1e90 divergence; new tests `test_skip_logit_init_beta` and `test_rate_magnitude_at_rest_bounded` guard against regression.
  - HIGH: Added Test Spec + Pseudocode sections to every step (previously missing from 1.1, 1.2, 1.6, 2.2, 3.1, 3.2, 3.3; Pseudocode missing from all steps except 2.1).
  - HIGH: Phase 3 Verification + Cleanup paths corrected from `Surrogate/runs/v4_A1/log_A1.jsonl` / `best_A1.pt` to `outputs/*/*/train.log` / `best.pt` (cardiac_ml harness layout).
  - HIGH: Step 1.4 adds explicit compression-capacity note — `COMP_H1/H2=12` retained even though carried_dim grew 20→24; bump only if Phase B shows degradation.
  - HIGH: Step 1.6 now enumerates 10 specific stale tests by name instead of relying on a dim-literal grep that misses `ionic_rate_mlp` / `conc_kan` references in test_training.py freeze-mask loops. Adds negative grep assertion as exit criterion.
  - MEDIUM: Step 1.1 Verify + Checklist now explicitly assert `MLP_HIDDEN` removal.
  - MEDIUM: Step 1.5 decoder-zero test strengthened with `weight.abs().max() > 1e-4` side-check so it does not pass trivially against an all-zero W.
  - MEDIUM: Step 2.1 convergence switched to per-dim relative tolerance `|dg_i| / (1 + |g_i|) < rel_tol` (not uniform absolute). V grid reduced to `[-85.23]` MVP only — the full 9-voltage grid is deferred as Phase 2.5, removing the +60 mV unbounded-integration risk. TTP06 RHS MUST import from `Monodomain/Engine_V5.4/cardiac_sim/ionic/ttp06/`; the standalone-RHS escape hatch is explicitly removed.
  - MEDIUM: Step 3.1 cache-path discrepancy flagged — `data_cache.py:86` default `/media/HDD/surrogate_data/raw` does NOT exist; always pass `/media/HDD/norepinephrine/surrogate_data/raw` explicitly.
  - MEDIUM: Final Cleanup adds concrete `train_node.py` deprecation step (physical archive or Deprecation banner).
  - LOW: LR policy locked at `1e-4` (oracle value, from `conf/training/node.yaml`). The previously-floating 5e-4 suggestion is explicitly labeled as a "do not run first" fallback.
  - LOW: Step 1.3 Implementation Spec now instructs callers to `.double()` after construction (previously only implicit in Test Spec).
  - LOW: Step 2.2 adds `test_L_rest_requires_float64_model` and `test_init_conc_not_leaky` guards.
  - LOW: Archive snippet retains `cp` semantics but adds rationale (snapshot, not move).
  - LOW: Phase 1 Context flags that Steps 1.1–1.5 are PARTIALLY already implemented in HEAD (per flash-test commit) — agent must diff against `8f191f77` before redoing work.
  - LOW: Phase 2 Exit Criteria z_ss shape changed from `9 × 14` to `1 × 14` (MVP rest-only).
- **MUTATED 2026-04-20 (audit round 2)**: 16 issues addressed.
  - CRITICAL: Step 3.2 diagnostic extended with NRMSE computation (raw RMSE alone does not produce the "X%" values Success Criteria / Exit Criteria reference). Added post-table block that divides per-dim RMSE by physiological ranges from `loss_normalization._RANGES`.
  - HIGH: Step 1.4 pseudocode comment "gate logit stays at zero" corrected to reference BETA_INIT; stale wording removed.
  - HIGH: Step 3.1 `nfe_mean` metric removed from the watch list — no NFE callback exists in `cardiac_ml/training/callbacks.py`. Replaced with an explicit note on how to add one if needed.
  - HIGH: Phase 1 Context partial-progress flag rewritten per-step, listing exact expected state of each Step 1.1–1.6 item in HEAD so agents don't redo completed work.
  - HIGH: NEW Step 3.0 added — patches `cardiac_ml/model/ionic_node_factory.py` to call `pin_rest_bias()` after `load_state_dict`. Without this, warm-starts silently break the rest-attractor invariant.
  - HIGH: Step 2.1 `test_z_ss_rest_matches_known` strengthened — added `test_z_ss_from_perturbed_initial_converges_to_rest` (perturb then integrate back) as the real integrator smoke. Identity-start test kept as a cheap sanity check but labeled weak.
  - MEDIUM: Phase 3 Step 3.1 cache-dtype caveat added — `data_cache.py:137` writes cache as float32, forcing runtime upcast. Non-blocker; follow-up to fix noted.
  - MEDIUM: Step 2.2 removed the conditional `test_L_rest_requires_float64_model` (it asked the agent to decide semantics at implementation time). Replaced with `test_L_rest_float64_contract` which simply locks the current explicit-float64 behavior.
  - MEDIUM: Phase 1 param-count bands unified to `[7800, 8100]` across Step 1.6, Phase 1 Verification smoke, and Phase 1 Exit Criteria (previously three distinct ranges).
  - MEDIUM: Phase 1 Verification smoke test now populates INIT_CONC in z_rest (matches the rest-attractor invariant; previously used unphysical zero concentrations).
  - MEDIUM: Step 1.6 `test_v3_param_count` bullet flagged as "already updated in HEAD — verify via grep, don't re-edit".
  - MEDIUM: Step 2.1 Risk bullet clarified — artifact IS used (by `test_z_ss_from_perturbed_initial_converges_to_rest`); disposition after Phase 3.2 is "keep the file, maybe defer extending the V grid", not "remove".
  - LOW: Step 1.4 `.double()` prescription softened — documented runtime safety net (PyTorch dtype-mismatch RuntimeError) rather than adding a speculative assert.
  - LOW: Step 3.3 IDEALOG template placeholder changed from `MM-DD` to `YYYY-MM-DD` with explicit instruction to replace via `date +%F`.
  - LOW: Step 3.1 MLflow UI block labeled as "manual monitoring aid, not a verification step".
  - LOW: `test_skip_logit_init_beta` now computes expected value from imported `BETA_INIT` constant, not a hardcoded 0.00669285 literal.
  - LOW: Round-1 mutation log accounting clarified — 20 issues identified, 16 mutation bullets (several bundled).
- **MUTATED 2026-04-20 (audit round 3)**: 17 issues addressed (0 critical, 4 high, 7 medium, 6 low).
  - HIGH: `test_total_loss_contains_rest` assertion math corrected — compares `loss ≈ base + LAMBDA_REST * L_rest` (was `loss > L_rest`, wrong when L_rest is large and LAMBDA_REST is small).
  - HIGH: `test_ionic_node_smoke_e2e` claim corrected — no such test pre-exists in cardiac_ml; instruction is now to grep for it, create if absent, mirror `test_synthetic_end_to_end`.
  - HIGH: Stale docstrings in `stage1.py` lines 81 and 181 fixed directly — both said "logit init 0 / alpha=0.5" which contradicts the BETA_INIT fix; now explicitly reference BETA_INIT and warn against re-init to zero.
  - HIGH: Step 2.1 API updated — `compute_z_ss_grid` signature now exposes `initial_state` kwarg (required for `test_z_ss_from_perturbed_initial_converges_to_rest`) and returns a `converged` boolean array.
  - MEDIUM: `test_stage1_param_count` in `test_model.py` updated to band `[7800, 8100]` — now matches PLAN single-source-of-truth.
  - MEDIUM: Step 3.0 `test_factory_pins_rest_bias_after_warm_start` clarified — ckpt must be in the v3 `{"stage1_state_dict": ...}` wrapper format, not a flat dict.
  - MEDIUM: Final Cleanup float-grep expectation enumerated explicitly — 4 legitimate hits and 1 known-bug hit (`data_cache.py:137`), not a vague "any explicit promotion".
  - MEDIUM: `test_L_rest_goes_to_zero` disambiguated — test L_rest computation isolated from odeint, don't monkey-patch through a rollout.
  - MEDIUM: Step 1.6 negative-grep now includes `KANLayer` (consistent with Phase 1 Cleanup's `.kan` import removal).
  - MEDIUM: Step 3.2 diagnostic loader now guards `ckpt.get("epoch")` / `ckpt.get("val_loss")` against the `f"{None:.4f}"` TypeError that would occur on flat cardiac_ml checkpoints.
  - LOW: Step 1.1 Risk's prediction about `loss_normalization.py` hits removed (was wrong — grep returns empty).
  - LOW: `test_z_ss_uses_v54_rhs` strengthened from a grep to an import-chain + mock assertion.
  - LOW: Step 3.0 pseudocode changed to diff format to preserve the factory's existing error-handling (FileNotFoundError + KeyError).
  - LOW: Phase 3 Verification block labeled "shell block" to prevent agents from treating it as a single Python one-liner.
  - LOW: Skipped — stale "Read First" line number refs left as-is (cheap to absorb at implementation; line-number drift is expected in evolving files).
- **MUTATED 2026-04-20 (audit round 4)**: 5 issues addressed (1 critical + 4 high). Halting iteration — remaining MEDIUM/LOW issues are sub-threshold cruft that are being introduced by the audit process itself faster than they are resolved (round 3 added 20 issues to fix 17; round 4 would add more). Plan is implementation-ready as-is.
  - CRITICAL: `test_total_loss_contains_rest` rewritten to compare two runs (normal vs `LAMBDA_REST=0` monkey-patched) rather than a tautological single-run algebraic identity. Catches regression where implementer drops `LAMBDA_REST * L_rest` from `mean_loss`.
  - HIGH: Step 1.6 stale-test enumeration rewritten to reflect ACTUAL file state (HEAD 2026-04-20). Most tests are already v4-compliant from the flash-test pass; the real Step 1.6 work is (a) running the negative-grep and (b) adding any NEW tests from Test Spec sections that haven't landed.
  - HIGH: Step 3.0 Read First line range updated; file is only 51 lines total, body at 24–50.
  - HIGH: Step 2.1 TTP06 adapter disambiguated — V5.4 does NOT export a pure RHS, but exports `compute_gate_steady_states`, `compute_gate_time_constants`, `compute_concentration_rates` as primitives. Adapter composes these via `dg_i/dt = (g_inf_i - g_i) / tau_i` — composition, not reimplementation. ~30 lines. Explicitly warns NOT to finite-difference `step()` (Rush-Larsen, wrong rate).
  - HIGH: Success Criteria `val_loss < 0.02` now explicitly folded in (was only in Exit Criteria). Single-standard for passing Phase 3.
  - Not addressed (sub-threshold after 3 rounds):
    - Stale line numbers in Read First blocks (cost of keeping them accurate > cost of agent re-locating sections at read time)
    - Hardcoded `.cuda()` in Phase 2 smoke (works on the intended environment; non-blocker)
    - `test_z_ss_uses_v54_rhs` lazy-import limitation (test is advisory, not load-bearing)
    - Step 3.3 grep-for-strings verification form-vs-content (inherent limitation of text-pattern verification)
    - Objective's "~1,444 v3 params" trivia claim — left as stated, documented in IDEALOG anyway.
