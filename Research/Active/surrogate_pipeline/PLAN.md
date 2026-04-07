# PLAN: Neural ODE Pivot for IonicSurrogateV3

Created: 2026-04-06
Engine(s): Surrogate
Research question: [surrogate_pipeline](README.md)
Source: [IDEALOG.md](IDEALOG.md) — 2026-04-06 session: A4 failed, pivot to unconstrained vector field f(z,V)

## Objective
Replace the discrete autoregressive rollout in IonicSurrogateV3 with a Neural ODE formulation: Stage 1 becomes a dynamics function `dz/dt = f_θ(z, V)`, trained via `odeint_adjoint` (dopri8). Inference remains Euler — no solver overhead at deploy time. The adjoint method shapes attractor geometry during training; inference exploits it for free at any dt. Archive all discrete training code cleanly before modifying anything.

## Success Criteria
- [ ] `stage1.py`: no dt in VoltageAttention, `residual_bypass` replaces `interpolate`, `dzdt()` method added
- [ ] `model/node.py`: IonicNODE wrapper passes torchdiffeq integration tests
- [ ] `training/node_rollout.py`: odeint_adjoint training loop validated on synthetic AP
- [ ] All existing tests pass after updating call sites (dt removal requires test updates)
- [ ] Euler inference at dt=0.01ms, dt=0.1ms, dt=1ms all produce finite outputs
- [ ] Archive: discrete rollout and phases moved to `training/archive/` with explanatory headers

## Architecture Changes
- NEW: `Surrogate/surrogate/model/node.py` — IonicNODE: torchdiffeq wrapper around Stage1.dzdt(), plus Euler inference step
- MOD: `Surrogate/surrogate/model/stage1.py:49-56,78-86,199-210` — remove dt from VoltageAttention; `interpolate` → `residual_bypass`; add `dzdt(z, V)` method
- MOD: `Surrogate/surrogate/model/ionic_surrogate_v3.py:155-160` — remove dt from Stage1 call
- NEW: `Surrogate/surrogate/training/node_rollout.py` — odeint_adjoint training loop replacing rollout.py
- ARCHIVE: `Surrogate/surrogate/training/archive/rollout_discrete.py` — copy of rollout.py pre-NODE
- ARCHIVE: `Surrogate/surrogate/training/archive/phases_discrete.py` — copy of phases.py A1-A4 configs

## Known Failures (from IDEALOG)
- **Discrete autoregressive at dt=0.01ms (A1-A4)** — failed because 30K steps compound errors faster than model corrects. TBPTT window=500, warm restarts, dt curriculum all failed at A4. Do NOT retry discrete rollout at native dt.
- **Batch=32768** — too few optimizer steps, LR scaling explodes. Use batch=4096.
- **Variance normalization** — K_i drowns Ca_i. Use per-dim min-max.
- **External encoder for latent bootstrapping** — imposes wrong latent space. Model discovers own latent.
- **Standalone decoder recalibration** — decoder becomes stale when latent shifts. Co-train.
- **dt as attention input for NODE** — makes vector field dt-dependent, corrupts ODE integration. Remove dt from W_k/W_v entirely.
- **forward() as discrete stepper with dzdt()** — `z + dzdt(z,V)` implicitly assumes dt=1, incompatible with `euler_step(z,V,dt) = z + dt*dzdt(z,V)`. Decision: `dzdt()` returns a RATE. `forward()` repurposed to run compression + scaffold only (no dynamics). All state advancement through `euler_step()` or `odeint_adjoint`.
- **Existing checkpoint compatibility** — `residual_bypass` (additive) differs from `interpolate` (convex). All A1-A4 checkpoints are unusable. Acceptable — A4 failed, fresh training required.

---

## Phase 0: Archive Discrete Training Code

**Goal**: Preserve the discrete training pipeline before any modifications. Clean archive with explanatory headers — not deleted, clearly labeled as superseded.
**Tier**: trivial
**Estimated scope**: copy 2 files, create archive dir

### Phase Context
Archive target: `Surrogate/surrogate/training/archive/`. Source files: `rollout.py` (TBPTT discrete rollout) and `phases.py` (A1-A4 phase configs). Archive is a copy — originals stay until Phase 1 confirms all tests pass. Do not modify any test files.

---

### Step 0.1: Create archive and copy discrete training files
**Model**: sonnet

#### Read First
- `Surrogate/surrogate/training/rollout.py:1-10` — confirm file exists
- `Surrogate/surrogate/training/phases.py:1-10` — confirm file exists

#### Why
Archiving before touching anything means the working discrete pipeline is always recoverable. The archive files serve as historical reference for comparing training behavior.

#### Implementation Spec
**Files to create:**
- `Surrogate/surrogate/training/archive/__init__.py` — empty
- `Surrogate/surrogate/training/archive/rollout_discrete.py` — copy of rollout.py with header
- `Surrogate/surrogate/training/archive/phases_discrete.py` — copy of phases.py with header

**Archive header (prepend to each):**
```python
# ARCHIVED: 2026-04-06 — Discrete autoregressive training pipeline.
# Superseded by Neural ODE pivot (node_rollout.py).
# A4 (dt=0.01ms, 30K steps) failed after 155+ epochs — val stuck at ~720.
# Root cause: error compounding over long discrete rollouts. Not a hyperparameter issue.
# Kept for historical reference. Do NOT import in production code.
# See KNOWLEDGE.md Section 5b for full analysis.
```

#### Test Spec
- No new tests — archive is passive (importability check in Verify suffices)

#### Checklist
- [ ] `Surrogate/surrogate/training/archive/` directory created
- [ ] `__init__.py` created (empty)
- [ ] `rollout_discrete.py` created with archive header prepended
- [ ] `phases_discrete.py` created with archive header prepended
- [ ] Original `rollout.py` and `phases.py` untouched

#### Risk
Archive copy fails silently (permissions, wrong path). Mitigation: verify importability + ls in Verify step.

#### Verify
```bash
conda run -n heart-conduction bash -c "cd Surrogate && python -c 'import surrogate.training.archive.rollout_discrete; print(\"ok\")'
"
ls Surrogate/surrogate/training/archive/
```

#### Exit Criteria
- [ ] Archive dir exists with 3 files
- [ ] Both archive files importable
- [ ] Originals unchanged

---

### Phase 0 Verification
```bash
conda run -n heart-conduction bash -c "cd Surrogate && python -m pytest tests/ -q 2>&1 | tail -5"
```

### Phase 0 Exit Criteria
- [ ] All 51 existing tests pass (nothing touched yet)
- [ ] Archive files exist and importable

### Phase 0 Cleanup
- [ ] float64 consistency — archive files inherited from originals (no change needed)
- [ ] No V5.3 modifications

**-> Commit point: `git commit -m "Archive discrete training pipeline before NODE pivot"`**

---

## Phase 1: Modify Stage1 for Neural ODE

**Goal**: Four targeted changes to `stage1.py`: remove dt from VoltageAttention, replace `interpolate` with `residual_bypass`, add `dzdt()` method (returns rate), repurpose `forward()` to compression+scaffold only (no dynamics). Existing tests updated for new signatures.
**Tier**: small
**Estimated scope**: ~30 lines changed in one file + minor update to ionic_surrogate_v3.py

### Phase Context
Three changes, all in `Surrogate/surrogate/model/stage1.py`:

**1. `interpolate` → `residual_bypass`** (lines ~49-56 and both call sites):
```python
# OLD: (1-alpha)*base + alpha*correction  — blend (identity suppressed at alpha=1)
# NEW: base + alpha*correction            — additive (identity always flows through)
```
This makes the derivative `dz/dt = delta_attn + alpha * mlp_correction` — clean sum. Applies to ionic mixing (~line 207) AND conductance compression (~line 215).

**2. Remove dt from VoltageAttention** (lines ~66-86):
- `__init__`: `W_k = nn.Linear(2, attn_dim)` → `nn.Linear(1, attn_dim)`. Same for `W_v`. Loses 8 params.
- `forward(carried_state, Vm, dt)` → `forward(carried_state, Vm)`. `x = Vm.unsqueeze(-1)` instead of `torch.stack([Vm, dt])`.

**3. Add `IonicStage1.dzdt(z, V)` method**: computes dz/dt without running compression or scaffold. Used by IonicNODE ODE wrapper. Returns tensor of same shape as z (carried_dim).

**Update `IonicStage1.forward()`**: remove dt from signature. Call site in `ionic_surrogate_v3.py` line 157 must be updated too.

Read ALL test call patterns before editing — tests may pass dt explicitly.

---

### Step 1.1: Remove dt from VoltageAttention and add residual_bypass
**Model**: opus

#### Read First
- `Surrogate/surrogate/model/stage1.py:1-260` — full file
- `Surrogate/tests/test_model.py:1-200` — all Stage1 test call patterns
- `Surrogate/surrogate/model/ionic_surrogate_v3.py:118-160` — forward signature and Stage1 call

#### Why
VoltageAttention takes `[Vm, dt]` as 2-dim input. For NODE, `f(z,V)` must be dt-independent — if dt is inside the network, the vector field changes with step size and the ODE solver cannot integrate it correctly. `residual_bypass` ensures the derivative is `delta_attn + alpha*correction` (unambiguous rate), not a blend that can suppress the base signal.

#### Implementation Spec
**Files to modify:**
1. `Surrogate/surrogate/model/stage1.py`
2. `Surrogate/surrogate/model/ionic_surrogate_v3.py:157`

**Change 1 — residual_bypass function:**
```python
def residual_bypass(base: Tensor, correction: Tensor, logit: Tensor) -> Tensor:
    """Additive residual bypass. Identity path unconditional; correction adds on top.
    alpha=0: pass base unchanged. alpha=1: add full correction to base.
    For NODE: derivative = attention_rate + alpha * mlp_correction (clean additive sum).
    """
    alpha = torch.sigmoid(logit)
    return base + alpha * correction
```

**Change 2 — VoltageAttention (init + forward):**
```python
# __init__: change both Linear dims from 2 to 1
self.W_k = nn.Linear(1, attn_dim, bias=False)
self.W_v = nn.Linear(1, attn_dim, bias=False)

# forward: remove dt param, use Vm.unsqueeze(-1)
def forward(self, carried_state: Tensor, Vm: Tensor) -> Tensor:
    x = Vm.unsqueeze(-1)          # (B, 1)
    k = self.W_k(x)               # (B, d)
    v = self.W_v(x)               # (B, d)
    # rest unchanged
```

**Change 3 — IonicStage1._compress() helper:**
```python
def _compress(self, carried_state: Tensor) -> Tensor:
    """Run gate conductance compression on carried_state. No dynamics."""
    linear_path = self.gate_conductance_linear(carried_state)
    nonlinear_path = self.gate_conductance_mlp(carried_state)
    return residual_bypass(linear_path, nonlinear_path, self.gate_conductance_logit)
```
Needed by `forward()` (compression+scaffold) and `node_rollout.py` (scaffold loss at landmarks).

**Change 4 — IonicStage1.dzdt():**
```python
def dzdt(self, z: Tensor, Vm: Tensor) -> Tensor:
    """Compute dz/dt for ODE integration. No compression, no scaffold decoders.

    Args:
        z: carried_state (B, carried_dim) or (carried_dim,)
        Vm: membrane voltage (B,) or scalar
    Returns:
        dz_dt: rate of change, same shape as z
    """
    squeezed = z.dim() == 1
    if squeezed:
        z = z.unsqueeze(0)
        Vm = Vm.view(1)

    z_mid = self.voltage_attention(z, Vm)           # z + delta_attn
    delta = z_mid - z                               # attention rate for all dims

    ionic_delta = delta[:, :self.ionic_dim]
    conc_delta  = delta[:, self.ionic_dim:]

    correction = self.ionic_mixing_mlp(rms_norm(z_mid[:, :self.ionic_dim]))
    ionic_rate = residual_bypass(ionic_delta, correction, self.ionic_mixing_logit)

    dz_dt = torch.cat([ionic_rate, conc_delta], dim=-1)

    if squeezed:
        dz_dt = dz_dt.squeeze(0)
    return dz_dt
```

**Change 4 — Remove IonicStage1.forward() as a discrete stepper.**

`forward()` is removed as a state-advancing method. All callers use `dzdt()` (for ODE integration) or `euler_step()` via IonicNODE (for inference). The old `forward()` implicitly assumed dt=1 which is semantically wrong — `dzdt()` returns a rate (dz/dt), not a displacement.

Replace `forward()` with a thin wrapper that runs compression and scaffold decoders on a given state (for use after ODE integration produces the new state):

```python
def forward(self, carried_state: Tensor, Vm: Tensor) -> Tuple[...]:
    """Run compression + scaffold on carried_state. Does NOT advance state.
    State advancement is done by IonicNODE.euler_step() or odeint_adjoint.
    """
    squeezed = carried_state.dim() == 1
    if squeezed:
        carried_state = carried_state.unsqueeze(0)
        Vm = Vm.view(1)

    ionic_new = carried_state[:, :self.ionic_dim]
    conc_new  = carried_state[:, self.ionic_dim:]

    # Compression and scaffold only — no dynamics
    conductance_latent = self._compress(carried_state)

    ionic_state_pred = None
    conductance_pred = None
    if hasattr(self, "ionic_state_decoder"):
        ionic_state_pred = self.ionic_state_decoder(ionic_new)
        conductance_pred = self.gate_conductance_decoder(conductance_latent)

    if squeezed:
        carried_state = carried_state.squeeze(0)
        conductance_latent = conductance_latent.squeeze(0)
        conc_new = conc_new.squeeze(0)
        if ionic_state_pred is not None:
            ionic_state_pred = ionic_state_pred.squeeze(0)
            conductance_pred = conductance_pred.squeeze(0)
    
    return (carried_state, conductance_latent, conc_new, ionic_state_pred, conductance_pred)
```

**ionic_surrogate_v3.py line 157:**
```python
# OLD: cs_new, cond_new, conc_new, ... = self.stage1(carried_state, Vm, dt)
# NEW: Stage1 no longer advances state. IonicNODE.euler_step() handles dynamics.
#      Stage1.forward() now runs compression + scaffold on the already-advanced state.
cs_new, cond_new, conc_new, ionic_state_pred, conductance_pred = self.stage1(
    carried_state, Vm   # dt removed; this just does compression + scaffold
)
```
**Keep `dt` in `IonicSurrogateV3.forward()` signature** but don't pass it to Stage1. This avoids breaking `rollout.py` and `trainer.py` (superseded but not deleted). Stage1 no longer uses dt internally.

#### Test Spec
- All existing `test_model.py` Stage1 tests pass (update any that pass dt explicitly)
- New `test_stage1_dzdt_shape` — `dzdt(z, V)` returns same shape as z, float64, finite values
- New `test_stage1_dzdt_no_dt` — `dzdt` does not accept dt argument (TypeError)
- New `test_stage1_dzdt_numerical` — at alpha≈0 (init), `dzdt(z, V)` ≈ `voltage_attention(z, V) - z` (the pure attention rate). Verify with atol=1e-6.
- New `test_stage1_forward_no_dynamics` — `forward(z, V)` returns same carried_state (no advancement), plus compression + scaffold outputs
- New `test_residual_bypass_near_zero_alpha` — at logit=-100.0, output ≈ base (atol=1e-4)
- New `test_residual_bypass_near_one_alpha` — at logit=100.0, output ≈ base + correction (atol=1e-4)

#### Checklist
- [ ] `interpolate` renamed to `residual_bypass`, formula changed to `base + alpha * correction`
- [ ] Both call sites updated: ionic mixing and conductance compression
- [ ] `VoltageAttention.__init__`: W_k and W_v are `Linear(1, attn_dim)`
- [ ] `VoltageAttention.forward`: dt removed from signature, `x = Vm.unsqueeze(-1)`
- [ ] `IonicStage1.dzdt()` method added, returns correct shape
- [ ] `IonicStage1._compress()` helper added (runs gate conductance compression on given state)
- [ ] `IonicStage1.forward()`: dt removed, repurposed to compression+scaffold only (no dynamics, uses _compress)
- [ ] `ionic_surrogate_v3.py`: dt removed from Stage1 call
- [ ] All test call sites updated
- [ ] Module docstring updated: `[Vm, dt]` → `[Vm]`, remove dt references from architecture description

#### Verify
```bash
conda run -n heart-conduction bash -c "cd Surrogate && python -m pytest tests/test_model.py -v 2>&1 | tail -20"
conda run -n heart-conduction bash -c "cd Surrogate && python -m pytest tests/ -q 2>&1 | tail -5"
```

#### Exit Criteria
- [ ] All model tests pass
- [ ] `dzdt()` produces finite float64 output of correct shape
- [ ] `residual_bypass` in stage1.py, `interpolate` gone

#### Risk
Test files call Stage1 with dt — read test_model.py fully before editing. Tests that need updating:
- `test_stage1_contractivity` — calls `voltage_attention(carried, Vm, dt)` AND reconstructs target via `torch.stack([Vm, dt])`. **Must be redesigned** (not just signature-updated): `residual_bypass` is additive, not contractive in the convex sense. Rewrite to test that dzdt output is finite and bounded.
- `test_stage1_param_count` — hardcodes `W_k: 2*4=8` and `W_v: 2*4=8`. After dt removal: `1*4=4` each. Inference params: 1416→1408, total: 1699→1691.
- `test_stage1_alpha_zero`, `test_stage1_beta_zero` — assertions still hold under `residual_bypass` (alpha=0 → output=base), but test structure changes because `forward()` no longer advances state.
- All `TestV3` tests that pass `dt` to the full surrogate — keep dt in V3 signature so these pass with minimal changes.

`IonicSurrogateV3.forward()` keeps dt in signature but doesn't pass it to Stage1. `rollout.py` and `trainer.py` are superseded but not deleted — keeping dt avoids breaking them.

---

### Phase 1 Verification
```bash
conda run -n heart-conduction bash -c "cd Surrogate && python -m pytest tests/ -v 2>&1 | tail -15"
grep -n "interpolate(" Surrogate/surrogate/model/stage1.py && echo "FAIL: interpolate still present" || echo "OK: interpolate removed"
grep -n "dt" Surrogate/surrogate/model/stage1.py | grep "Linear(2" && echo "FAIL: dt still in Linear" || echo "OK"
```

### Phase 1 Exit Criteria
- [ ] All tests pass (existing tests updated for dt removal + forward() change)
- [ ] `stage1.dzdt(z, V)` method exists and returns rate (not displacement)
- [ ] `stage1.forward(z, V)` does compression + scaffold only (no dynamics)
- [ ] `interpolate` not present in stage1.py
- [ ] No `Linear(2,` in VoltageAttention

### Phase 1 Cleanup
- [ ] float64: `dzdt()` produces float64 output
- [ ] No dt argument anywhere in stage1.py (grep confirm)
- [ ] KNOWLEDGE.md Section 5b matches implemented code

**-> Commit point: `git commit -m "NODE pivot Phase 1: remove dt from attention, residual_bypass, add dzdt()"`**

---

## Phase 2: IonicNODE Wrapper

**Goal**: New `model/node.py` — `IonicNODE` wraps Stage1.dzdt as a torchdiffeq-compatible ODE function. Also provides Euler inference step for any dt.
**Tier**: medium
**Estimated scope**: ~120 lines, new file

### Phase Context
torchdiffeq ODE function signature: `(t: Tensor, z: Tensor) -> Tensor` — scalar time, state. V(t) is interpolated from a stored trajectory. Two interfaces:
1. `IonicNODE.integrate(z0, t_eval)` → trajectory via `odeint_adjoint` (training)
2. `IonicNODE.euler_step(z, V, dt)` → `z + dt * dzdt(z, V)` (inference, any dt)

V(t) interpolation: linear interpolation over the stored t_grid. Set via `set_v_trajectory()` before `integrate()`, cleared after.

No new learned parameters — all dynamics live in stage1.

---

### Step 2.1: Install torchdiffeq and create IonicNODE
**Model**: opus

#### Read First
- `Surrogate/surrogate/model/stage1.py` — confirm `dzdt()` from Phase 1
- `Surrogate/surrogate/model/__init__.py` — current exports

#### Why
IonicNODE is a thin bridge. The only complexity is the V(t) interpolation (must map scalar t to batch-aligned V) and the odeint_adjoint wiring. euler_step is the critical inference path — must be simple and fast.

#### Implementation Spec
**Check/install torchdiffeq first:**
```bash
conda run -n heart-conduction python -c "import torchdiffeq; print('ok')" 2>/dev/null || \
conda run -n heart-conduction pip install torchdiffeq
```

**File to create:** `Surrogate/surrogate/model/node.py`

```python
"""IonicNODE: wraps IonicStage1.dzdt as a torchdiffeq ODE function.

Training:  odeint_adjoint(node, z0, t_eval)  — adjoint shapes vector field geometry
Inference: node.euler_step(z, V, dt)         — no solver, works for any dt

Zero new learned parameters. All dynamics in stage1.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from torchdiffeq import odeint_adjoint

from .stage1 import IonicStage1


class IonicNODE(nn.Module):

    def __init__(self, stage1: IonicStage1):
        super().__init__()
        self.stage1 = stage1
        self._V_traj: Optional[Tensor] = None   # (B, T) or (T,)
        self._t_grid: Optional[Tensor] = None   # (T,) cumulative times

    def set_v_trajectory(self, V_traj: Tensor, t_grid: Tensor) -> None:
        """Store V(t) for interpolation during integrate(). Call before integrate()."""
        self._V_traj = V_traj
        self._t_grid = t_grid

    def clear_v_trajectory(self) -> None:
        self._V_traj = None
        self._t_grid = None

    def _interpolate_V(self, t: Tensor) -> Tensor:
        """Linear interpolation of V at continuous scalar time t.
        
        V_traj has T points (one per segment step). t_grid has T+1 points
        (cumulative dt, including t=0). Interpolation maps t to V values using
        t_grid[:-1] as the knot times (each V[i] corresponds to t_grid[i]).
        At t >= t_grid[T-1], clamp to V_traj[T-1] (last value).
        """
        assert self._V_traj is not None, "Call set_v_trajectory() before integrate()"
        t_grid = self._t_grid
        T = self._V_traj.shape[-1]  # number of V samples
        t_c = t.clamp(t_grid[0], t_grid[T - 1])  # clamp to V_traj range, NOT t_grid[-1]
        idx = (torch.searchsorted(t_grid[:T].contiguous(), t_c) - 1).clamp(0, T - 2)
        t0, t1 = t_grid[idx], t_grid[idx + 1]
        frac = ((t_c - t0) / (t1 - t0 + 1e-12)).clamp(0.0, 1.0)
        if self._V_traj.dim() == 1:
            return self._V_traj[idx] + frac * (self._V_traj[idx + 1] - self._V_traj[idx])
        else:
            return self._V_traj[:, idx] + frac * (self._V_traj[:, idx + 1] - self._V_traj[:, idx])

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        """torchdiffeq interface: (scalar t, state z) -> dz/dt."""
        V = self._interpolate_V(t)
        return self.stage1.dzdt(z, V)

    def integrate(
        self,
        z0: Tensor,
        t_eval: Tensor,
        method: str = "dopri8",
        rtol: float = 1e-4,
        atol: float = 1e-5,
    ) -> Tensor:
        """Integrate from t_eval[0] to t_eval[-1], return z at each t_eval point.
        Uses odeint_adjoint for O(1) memory backward pass.
        Returns: (N, B, carried_dim)
        """
        return odeint_adjoint(
            self, z0, t_eval,
            method=method, rtol=rtol, atol=atol,
            adjoint_params=list(self.stage1.parameters()),
        )

    def euler_step(self, z: Tensor, V: Tensor, dt: float) -> Tensor:
        """Euler inference step. No solver. Works for any dt value.
        Args:
            z: (B, carried_dim) or (carried_dim,)
            V: (B,) or scalar
            dt: timestep in ms (float — dt-independence guaranteed by training)
        Returns:
            z_next: same shape as z
        """
        return z + dt * self.stage1.dzdt(z, V)
```

**Update `Surrogate/surrogate/model/__init__.py`**: add `from .node import IonicNODE`.

#### Test Spec
- `test_ionic_node_euler_shape` — `euler_step(z, V, 0.01)` returns same shape as z, float64, finite
- `test_ionic_node_euler_variable_dt` — dt=0.01, 0.1, 1.0 all produce finite output (variable dt works)
- `test_ionic_node_integrate_shape` — `integrate(z0, t_eval)` returns (N, B, D), correct dims
- `test_ionic_node_adjoint_backward` — `z_traj[-1].sum().backward()` succeeds, grads exist on stage1 params
- `test_ionic_node_v_interpolate_endpoints` — V at t_grid[0] matches V_traj[:,0]; at t_grid[-1] matches V_traj[:,-1]
- `test_ionic_node_no_learned_params` — `sum(p.numel() for p in node.parameters() if id(p) not in {id(q) for q in node.stage1.parameters()}) == 0`

#### Checklist
- [ ] torchdiffeq installed and importable
- [ ] `node.py` created with full `IonicNODE` class
- [ ] `set_v_trajectory` / `clear_v_trajectory` implemented
- [ ] `_interpolate_V` handles batched (B,T) and unbatched (T,) V_traj
- [ ] `forward(t, z)` — torchdiffeq signature
- [ ] `integrate()` uses `odeint_adjoint`
- [ ] `euler_step()` is pure Euler, no torchdiffeq import needed at inference
- [ ] `__init__.py` exports `IonicNODE`
- [ ] All 6 new tests pass

#### Verify
```bash
conda run -n heart-conduction bash -c "cd Surrogate && python -m pytest tests/test_model.py -v -k 'NODE or node' 2>&1"
conda run -n heart-conduction bash -c "cd Surrogate && python -m pytest tests/ -q 2>&1 | tail -5"
```

#### Exit Criteria
- [ ] All NODE model tests pass
- [ ] Adjoint backward produces gradients on stage1 parameters
- [ ] All 51+ tests pass

#### Risk
torchdiffeq not installed — check first, install via pip. `odeint_adjoint` requires `adjoint_params` to be explicitly listed (not just `model.parameters()`) — use `list(self.stage1.parameters())`.

---

### Phase 2 Verification
```bash
conda run -n heart-conduction bash -c "cd Surrogate && python -m pytest tests/ -v 2>&1 | tail -15"
```

### Phase 2 Exit Criteria
- [ ] `model/node.py` exists
- [ ] Euler inference works for dt=0.01, 0.1, 1.0
- [ ] odeint_adjoint backward pass succeeds
- [ ] All tests pass

### Phase 2 Cleanup
- [ ] float64: all tensors in node.py are float64 (no float32 creep)
- [ ] `_interpolate_V` edge case: t at exact boundary of t_grid works
- [ ] No global mutable state — V trajectory per-instance only

**-> Commit point: `git commit -m "NODE pivot Phase 2: IonicNODE with odeint_adjoint + Euler inference"`**

---

## Phase 3: NODE Training Loop

**Goal**: New `training/node_rollout.py` — odeint_adjoint training loop with sparse AP landmark t_eval. Replaces discrete `rollout.py` for NODE training. Original rollout.py NOT deleted.
**Tier**: medium
**Estimated scope**: ~150 lines, new file

### Phase Context
Key differences from discrete rollout:
1. No step-by-step for loop — single `node.integrate(z0, t_eval)` call
2. V(t) interpolation built from full segment before solve
3. Loss at 20 AP landmarks (dense during upstroke) — adjoint backprops through full trajectory regardless

**t_eval landmarks (ms) — dense during upstroke where dynamics are stiff:**
```python
NODE_T_EVAL_MS = [0, 0.1, 0.3, 0.5, 0.8, 1, 1.5, 2, 3, 5,  # 10 points in 5ms (upstroke)
                  10, 20, 40, 80,                              # 4 points in plateau
                  120, 160, 200, 240, 270, 300]                # 6 points in repol+diastole
```
Dense upstroke sampling focuses loss signal where fast gate dynamics live. The ODE solver naturally takes small adaptive steps there — matching loss resolution to solver resolution. Single solve from resting state (0→300ms); no segmentation (no ground truth z for mid-AP initial conditions).

**Stiffness strategy:** dopri8 (8th order Dormand-Prince) handles the full 300ms solve. Higher order = fewer steps than dopri5 for smooth regions (plateau/repol), more function evaluations per step but net fewer total. GPU-native. Estimated NFE: 150-800. Monitor NFE on first training run — if consistently >1000, add diagonal preconditioner to IonicNODE (20 params, training-only coordinate rescaling).

Segment format (from SegmentDataset): `segment['Vm']` shape (B, T), `segment['dt']` shape (B, T). Build t_grid as cumulative sum: `[0, dt[0], dt[0]+dt[1], ...]` shape (T+1,).

At each t_eval point, find nearest index in t_grid for supervised loss targets.

Return dict with same keys as discrete `rollout()` for trainer compatibility.

---

### Step 3.1: Implement node_rollout.py
**Model**: opus

#### Read First
- `Surrogate/surrogate/training/rollout.py:1-160` — full discrete rollout for reference on loss logic and return format
- `Surrogate/surrogate/training/loss_normalization.py:1-60` — LossNormalizer interface
- `Surrogate/surrogate/training/datasets.py:1-80` — SegmentDataset output format (key names, shapes)
- `Surrogate/surrogate/model/node.py` — IonicNODE interface from Phase 2
- `Surrogate/tests/test_training.py:1-60` — fake segment helper `_make_fake_47col` for test writing

#### Why
The training loop is the only remaining change — all model-side work is done in Phases 1-2. Reusing the same loss normalization, phase names, and return dict format ensures node_rollout.py plugs into the existing trainer and monitoring infrastructure with no changes to those files.

#### Implementation Spec
**File to create:** `Surrogate/surrogate/training/node_rollout.py`

```python
"""NODE training rollout via odeint_adjoint.

Replaces discrete rollout.py for Neural ODE training.
Loss computed at AP landmark t_eval points; adjoint backprops through full trajectory.
Original rollout.py preserved in archive/ for reference.
"""
import torch
from torch import Tensor
from typing import Optional
from ..model.node import IonicNODE
from .loss_normalization import LossNormalizer
# Resting concentrations (Layer 0 physics) — duplicated from rollout.py to avoid
# dependency on discrete training code. Same values: [Na_i, K_i, Ca_i, Ca_ss].
INIT_CONC = torch.tensor([10.0, 138.0, 0.0001, 0.0002], dtype=torch.float64)

# AP landmark evaluation times (ms) — covers upstroke, plateau, repol, diastole
NODE_T_EVAL_MS = torch.tensor(
    [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0,   # dense upstroke
     10.0, 20.0, 40.0, 80.0,                                 # plateau
     120.0, 160.0, 200.0, 240.0, 270.0, 300.0],              # repol + diastole
    dtype=torch.float64
)

_normalizer = LossNormalizer()


def build_t_grid(segment_dt: Tensor) -> Tensor:
    """Build cumulative time grid from dt values.
    segment_dt: (B, T) or (T,) in ms.
    Returns: (T+1,) cumulative times starting at 0.
    NOTE: Uses first batch element's dt — assumes uniform dt across batch.
    If dt varies per batch element, this must be revised.
    """
    dt_1d = segment_dt[0] if segment_dt.dim() == 2 else segment_dt
    return torch.cat([
        torch.zeros(1, dtype=torch.float64, device=dt_1d.device),
        dt_1d.double().cumsum(0)
    ])


def node_rollout(
    node: IonicNODE,
    segment: dict,
    phase_name: str = "A1",
    device: Optional[torch.device] = None,
    t_eval_ms: Optional[Tensor] = None,
    z0_noise_sigma: float = 0.0,
) -> dict:
    """NODE training rollout: integrate z via odeint_adjoint, compute loss at landmarks.

    Args:
        node: IonicNODE instance (wraps stage1.dzdt)
        segment: dict with Vm (B,T), dt (B,T), ionic_states (B,T,14), etc.
        phase_name: A1/A2/A3/A4/B1/B2/B3/B4/C/D — same as discrete rollout
        z0_noise_sigma: Gaussian noise std on z0 for attractor basin widening (0=off)
        device: target device
        t_eval_ms: override landmark times (default: NODE_T_EVAL_MS)

    Returns:
        dict with 'loss' and per-component losses (same keys as discrete rollout)
    """
    if device is None:
        device = segment['Vm'].device

    B = segment['Vm'].shape[0]
    T = segment['Vm'].shape[1]

    # Build time grid and t_eval
    t_grid = build_t_grid(segment['dt'].to(device))  # (T+1,)
    T_max = t_grid[-1]

    t_eval = (t_eval_ms if t_eval_ms is not None else NODE_T_EVAL_MS).to(device)
    t_eval = t_eval[t_eval <= T_max]   # clamp to trajectory length
    if t_eval[0] > 0:
        t_eval = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), t_eval])

    # Initialize state: ionic=zeros, conc=resting
    z0 = torch.zeros(B, node.stage1.carried_dim, dtype=torch.float64, device=device)
    z0[:, node.stage1.ionic_dim:] = INIT_CONC.to(device)

    # Attractor basin widening: add Gaussian noise to z0 during training
    if z0_noise_sigma > 0 and node.training:
        z0 = z0 + z0_noise_sigma * torch.randn_like(z0)

    # Set V trajectory for interpolation during ODE solve
    # NOTE: V_traj has T points, t_grid has T+1 points. _interpolate_V handles this.
    node.set_v_trajectory(segment['Vm'].double().to(device), t_grid)

    try:
        z_traj = node.integrate(z0, t_eval)  # (N_eval, B, carried_dim)
    finally:
        node.clear_v_trajectory()

    # Compute loss at each t_eval point
    losses_per_eval = []
    component_sums: dict = {}

    for i, t_i in enumerate(t_eval):
        # Find nearest segment index for ground truth
        idx = int((torch.searchsorted(t_grid, t_i) - 1).clamp(0, T - 1).item())
        z_pred = z_traj[i]  # (B, carried_dim)

        step_losses = _compute_node_loss(phase_name, z_pred, segment, idx, node)
        losses_per_eval.append(step_losses['loss'])

        for k, v in step_losses.items():
            if k != 'loss':
                component_sums[k] = component_sums.get(k, 0.0) + v.detach()

    mean_loss = torch.stack(losses_per_eval).mean()

    result = {'loss': mean_loss}
    N_eval = len(t_eval)
    for k, v in component_sums.items():
        result[k] = v / N_eval
    return result


def _compute_node_loss(phase_name, z_pred, segment, idx, node) -> dict:
    """Loss at one t_eval point. Mirrors compute_phase_loss from rollout.py."""
    losses = {}
    ionic_pred = z_pred[:, :node.stage1.ionic_dim]
    conc_pred  = z_pred[:, node.stage1.ionic_dim:]

    # Scaffold predictions via decoders (if present)
    ionic_state_pred = None
    conductance_pred = None
    if hasattr(node.stage1, 'ionic_state_decoder'):
        ionic_state_pred = node.stage1.ionic_state_decoder(ionic_pred)
    if hasattr(node.stage1, 'gate_conductance_decoder'):
        # Need conductance latent — run compression on z_pred
        cond_lat = node.stage1._compress(z_pred)   # add _compress helper
        conductance_pred = node.stage1.gate_conductance_decoder(cond_lat)

    if phase_name in ("A1", "A2", "A3", "A4", "ionic_state"):
        losses['ionic_state_mse'] = _normalizer.normalized_mse(
            ionic_state_pred, segment['ionic_states'][:, idx, :], 'ionic_states')
        losses['conc_mse'] = _normalizer.normalized_mse(
            conc_pred, segment['concentrations'][:, idx, :], 'concentrations')
        losses['loss'] = losses['ionic_state_mse'] + losses['conc_mse']

    elif phase_name in ("B1", "B2", "B3", "B4", "ionic_state_and_conductance"):
        losses['ionic_state_mse'] = _normalizer.normalized_mse(
            ionic_state_pred, segment['ionic_states'][:, idx, :], 'ionic_states')
        losses['conc_mse'] = _normalizer.normalized_mse(
            conc_pred, segment['concentrations'][:, idx, :], 'concentrations')
        losses['conductance_mse'] = _normalizer.normalized_mse(
            conductance_pred, segment['conductance_products'][:, idx, :], 'conductance_products')
        losses['loss'] = losses['ionic_state_mse'] + losses['conc_mse'] + losses['conductance_mse']

    elif phase_name in ("C", "D", "I_ion"):
        # I_ion requires Stage2 — not yet wired in node_rollout; raise clear error
        raise NotImplementedError(
            "I_ion phase requires Stage2 in node_rollout — wire IonicNODE with full surrogate first"
        )
    else:
        raise ValueError(f"Unknown phase: {phase_name}")

    return losses
```

**`_compress()` helper**: Already added in Phase 1 (Step 1.1, Change 3). Used here by `_compute_node_loss` for scaffold losses at landmarks.

#### Test Spec
- `test_node_rollout_runs` — `node_rollout(node, fake_seg, 'A1')` completes, returns dict with 'loss'
- `test_node_rollout_backward` — `loss.backward()` runs, gradients exist on stage1 params, norm finite
- `test_node_rollout_loss_finite` — loss is finite float64 scalar
- `test_node_rollout_phase_names` — A1, B1 phases accepted; C raises NotImplementedError
- `test_node_rollout_z0_noise` — `node_rollout(..., z0_noise_sigma=0.01)` runs, loss differs from sigma=0
- `test_build_t_grid_cumulative` — t_grid[0]=0, t_grid[-1]=sum(dt), shape=(T+1,)
- `test_interpolate_v_boundary` — V at t beyond V_traj range clamps correctly (no index out of bounds)

#### Checklist
- [ ] `NODE_T_EVAL_MS` defined at module level (20 points, dense during upstroke)
- [ ] `build_t_grid()` builds cumulative time correctly, prepends 0
- [ ] `node_rollout()` calls `set_v_trajectory` before and `clear_v_trajectory` in finally block
- [ ] z0 initialized: ionic=zeros, conc=INIT_CONC resting values
- [ ] t_eval clamped to [0, T_max], 0 prepended if missing
- [ ] `_compute_node_loss` mirrors phase logic from discrete rollout
- [ ] `IonicStage1._compress()` helper added
- [ ] Return dict has same keys as discrete rollout
- [ ] C/D phase raises NotImplementedError (not silently wrong)

#### Verify
```bash
conda run -n heart-conduction bash -c "cd Surrogate && python -m pytest tests/test_training.py -v -k 'node or NODE' 2>&1"
conda run -n heart-conduction bash -c "cd Surrogate && python -m pytest tests/ -q 2>&1 | tail -5"

# Smoke test on real data shapes
conda run -n heart-conduction bash -c "cd Surrogate && python -c \"
import torch
from surrogate.model.stage1 import IonicStage1
from surrogate.model.node import IonicNODE
from surrogate.training.node_rollout import node_rollout
stage1 = IonicStage1(scaffold=True).double()
node = IonicNODE(stage1)
B, T = 4, 3000
seg = {
    'Vm': torch.randn(B, T, dtype=torch.float64) * 10 - 80,
    'dt': torch.full((B, T), 0.1, dtype=torch.float64),  # 3000 * 0.1ms = 300ms coverage
    'ionic_states': torch.rand(B, T, 14, dtype=torch.float64),
    'concentrations': torch.rand(B, T, 4, dtype=torch.float64).abs() + 0.0001,
    'conductance_products': torch.rand(B, T, 5, dtype=torch.float64),
}
out = node_rollout(node, seg, 'A1')
print('loss:', out['loss'].item())
out['loss'].backward()
gnorm = sum(p.grad.norm().item() for p in stage1.parameters() if p.grad is not None)
print('grad norm:', gnorm)
assert gnorm > 0, 'zero gradients!'
print('PASS')
\""
```

#### Exit Criteria
- [ ] node_rollout runs end-to-end on fake data
- [ ] Backward pass succeeds, gradient norm > 0
- [ ] All tests pass

#### Risk
`odeint_adjoint` may be slow on first call (JIT compilation). Smoke test may take 10-30s — expected. If solver fails with stiffness errors on random V(t) data, reduce rtol/atol or use `method='euler'` for testing only.

**NFE monitoring (stiffness diagnostic):** After smoke test passes, print NFE (number of function evaluations) from the integrate call. If NFE > 1000 for 300ms synthetic data, stiffness is a concern — add diagonal preconditioner to IonicNODE as contingency (see KNOWLEDGE.md Section 5b "Stiffness Analysis"). Expected NFE: 200-1000.

---

### Phase 3 Verification
```bash
conda run -n heart-conduction bash -c "cd Surrogate && python -m pytest tests/ -v 2>&1 | tail -20"
```

### Phase 3 Exit Criteria
- [ ] node_rollout runs end-to-end on synthetic data
- [ ] Backward pass produces non-zero gradients on stage1
- [ ] All tests pass (51+ including new NODE tests)

### Phase 3 Cleanup
- [ ] float64: all tensors in node_rollout.py are float64
- [ ] `clear_v_trajectory()` called in `finally` block (not just happy path)
- [ ] No imports from `rollout.py` in `node_rollout.py` (INIT_CONC duplicated locally)
- [ ] `_compress()` helper in stage1.py covered by a test

**-> Commit point: `git commit -m "NODE pivot Phase 3: node_rollout.py with odeint_adjoint training loop"`**

---

## Final Cleanup

1. Archive this completed plan:
```bash
mkdir -p Research/Active/surrogate_pipeline/plans
cp Research/Active/surrogate_pipeline/PLAN.md "Research/Active/surrogate_pipeline/plans/2026-04-06_neural-ode-pivot.md"
```

2. Update `Research/Active/surrogate_pipeline/IDEALOG.md` Current Direction and Next Step:
```
Current Direction: Neural ODE pivot complete. Stage1.dzdt() is the dynamics function.
IonicNODE provides odeint_adjoint training and Euler inference at any dt.
Next Step: First NODE training run on T1 data. Phase A1 (scaffold loss at AP landmarks).
Hyperparams: batch=4096, LR=5e-4, adjoint rtol=1e-4, atol=1e-5.
Monitor: loss finite + decreasing, gradient norm non-zero and stable, AP shape visible at t_eval landmarks.
```

3. Update `Surrogate/PROGRESS.md` — add NODE pivot as completed phase.

4. Cross-check:
```bash
# No float32 leaks in new files
grep -r "float32" Surrogate/surrogate/model/node.py Surrogate/surrogate/model/stage1.py Surrogate/surrogate/training/node_rollout.py
# Should return nothing

# dt fully removed from stage1
grep "def forward.*dt\|def dzdt.*dt\|W_k.*Linear(2\|W_v.*Linear(2" Surrogate/surrogate/model/stage1.py
# Should return nothing

# interpolate fully replaced
grep "interpolate(" Surrogate/surrogate/model/stage1.py
# Should return nothing

# Archive files exist
ls Surrogate/surrogate/training/archive/

# V5.3 untouched
git diff --name-only | grep Engine_V5.3
# Should return nothing
```

5. Revert tmux pane 2 from PLAN.md to WHITEBOARD.md:
```bash
tmux send-keys -t 2 C-c
sleep 0.3
tmux send-keys -t 2 'W=$(tput cols); H=""; while true; do N=$(md5sum Research/Active/surrogate_pipeline/WHITEBOARD.md 2>/dev/null | cut -d" " -f1); if [ "$N" != "$H" ]; then clear; glow -s .glow-style.json -w $W Research/Active/surrogate_pipeline/WHITEBOARD.md 2>/dev/null; H=$N; fi; sleep 1; done' Enter
```

---

## Mutation Log

*(populated during execution — mark each step as SKIPPED, SPLIT, or INSERTED with date and reason)*
