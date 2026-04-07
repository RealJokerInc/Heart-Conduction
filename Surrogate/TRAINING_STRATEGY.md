# Training Strategy — Neural ODE Ionic Surrogate v3

> Authoritative reference for NODE training decisions.
> Model: IonicSurrogateV3 (1,408 inference params). First run: TTP06, small dims.
> Supersedes discrete autoregressive strategy (archived in `archive/TRAINING_STRATEGY_discrete.md`).

---

## Overview

**What changed:** The discrete autoregressive rollout (30K sequential steps, TBPTT) failed at native dt (A4: val stuck ~720, 155+ epochs). The Neural ODE pivot replaces the step-by-step loop with a single `odeint_adjoint(dopri8)` call. The model learns `dz/dt = f_θ(z, V)` — a continuous vector field. Training shapes the field via adjoint gradients; inference uses simple Euler steps.

**What stayed the same:** Same model architecture (attention + MLP), same loss normalization (per-dim min-max), same scaffold decoders, same data pipeline (SegmentDataset provides V(t) trajectory). Phase names (A1, B1) are reused but the training loop is fundamentally different.

---

## Key Differences from Discrete Training

| | Discrete (failed) | NODE (current) |
|---|---|---|
| Forward pass | 30K sequential model calls | 1 `odeint_adjoint` call |
| Gradient method | BPTT through 30K steps (or truncated) | Adjoint ODE backward (~200-1000 solver steps) |
| Loss points | Every timestep | 20 AP landmarks (dense upstroke) |
| dt handling | Model input (`[Vm, dt]`) | Not a model input. Solver handles time. |
| Step size | Fixed dt=0.01ms | Adaptive (dopri8 chooses internally) |
| V(t) | Fed step-by-step | Linear interpolation from stored trajectory |
| z0 | zeros + resting conc | Same, plus optional Gaussian noise |

---

## Phase Structure

Same phase names, different execution. Each phase uses `node_rollout()` instead of `rollout()`.

| Phase | What trains | Loss components | Data |
|-------|------------|----------------|------|
| **A1** | Half 1 (attention + ionic MLP + ionic decoder) | ionic_state_mse + conc_mse | T1 |
| **A2-A4** | Same as A1 | Same | T1 (reserved for curriculum if A1 insufficient) |
| **B1** | Half 2 (conductance compression + decoder), Half 1 frozen | + conductance_mse | T1 |
| **B2-B4** | Same as B1 | Same | T1 (reserved) |
| **C** | Stage 2 (current readout), Stage 1 frozen | I_ion_mse | T1 — NOT YET IMPLEMENTED |
| **D** | All params | I_ion_mse | T1-T4 — NOT YET IMPLEMENTED |

**Major simplification vs discrete:** No dt curriculum needed. The ODE solver handles temporal resolution adaptively. A single phase (A1) covers the full AP at whatever resolution dopri8 needs. No rollout length, no subsample factor, no TBPTT window.

---

## First Training Run: Phase A1

### Data Setup

```python
# SegmentDataset provides V(t) trajectory for ODE interpolation
# subsample=10: take every 10th timestep → effective dt=0.1ms, segment covers 300ms
# This gives the V trajectory; the ODE solver interpolates between points
segment_ds = SegmentDataset(
    cached_data=tier1_train,
    segment_length=3000,     # 3000 points at dt=0.1ms = 300ms
    subsample=10,            # every 10th raw timestep
    stride=15000,            # non-overlapping 300ms windows
)
```

Why subsample=10 (not subsample=1): The V trajectory only needs enough resolution for linear interpolation. At dt=0.1ms, consecutive Vm values differ by <1mV during plateau and <5mV during upstroke. The ODE solver's adaptive stepping handles the fine resolution — it doesn't need 30K trajectory points to interpolate from.

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch size | 4096 | Proven in discrete A1-A3. Enough optimizer steps per epoch. |
| Learning rate | 5e-4 | Standard for small models with AdamW |
| Weight decay | 1e-4 | Soft regularization |
| Gradient clip | max_norm=1.0 | Prevent adjoint gradient explosions |
| dopri8 rtol | 1e-4 | Standard for neural ODEs |
| dopri8 atol | 1e-5 | Standard for neural ODEs |
| z0_noise_sigma | 0.0 initially | Enable (1e-3 → 1e-2) if validation loss plateaus |
| Epochs | 200 (patience=50) | Same as discrete A1 |
| Float | float64 | Project convention |

### t_eval Landmarks (20 points)

```python
[0, 0.1, 0.3, 0.5, 0.8, 1, 1.5, 2, 3, 5,   # 10 in upstroke (0-5ms)
 10, 20, 40, 80,                               # 4 in plateau
 120, 160, 200, 240, 270, 300]                  # 6 in repol+diastole
```

Dense upstroke = more loss signal where dynamics are stiff. Solver naturally takes small steps there — matching loss resolution to solver attention.

### Freeze Mask

Same as discrete A1 — Half 1 params only:
```
stage1.voltage_attention.*
stage1.ionic_mixing_mlp.*
stage1.ionic_mixing_logit
stage1.ionic_state_decoder.*
```

### Training Loop (pseudocode)

```python
for epoch in range(max_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        out = node_rollout(node, batch, phase_name='A1')
        out['loss'].backward()
        # IMPORTANT: clear V trajectory AFTER backward (adjoint needs it)
        node.clear_v_trajectory()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Validation
    with torch.no_grad():
        for val_batch in val_loader:
            val_out = node_rollout(node, val_batch, phase_name='A1')
            node.clear_v_trajectory()
```

### What Success Looks Like

| Metric | Target | Why |
|--------|--------|-----|
| Train loss | Decreasing over epochs | Basic sanity |
| Val loss | Decreasing, < 10.0 by epoch 50 | Discrete A3 reached val=0.92; NODE should do better |
| Gradient norm | Finite, stable (no explosions/vanishing) | Adjoint gradients should be well-behaved |
| NFE per segment | < 1000 | If >1000, stiffness is a problem → add preconditioner |
| ionic_state_mse | Dominant early, decreasing | Scaffold decoder learning gate structure |
| conc_mse | Small, stable | Concentrations change slowly; easy target |

### What Failure Looks Like

| Symptom | Diagnosis | Action |
|---------|-----------|--------|
| Loss NaN/Inf | Solver divergence | Reduce rtol/atol to 1e-5/1e-6. Check for div-by-zero in rms_norm. |
| Loss stuck high (>100 after 50 epochs) | Vector field not learning | Check gradient norms. Try LR=1e-3. Verify data pipeline produces correct segments. |
| NFE > 2000 | Latent dynamics are stiff | Add diagonal preconditioner (20 params). Or increase rtol to 1e-3. |
| Val loss increasing while train decreases | Overfitting | Increase weight decay to 1e-3. Add z0_noise_sigma=1e-3. |
| Gradient norm in millions | Adjoint instability | Reduce rtol/atol. Try method='dopri5'. Clip more aggressively (max_norm=0.1). |

---

## Monitoring

### Per-Epoch Logging

```python
{
    'epoch': int,
    'train_loss': float,
    'val_loss': float,
    'ionic_state_mse': float,     # scaffold: decoded gates vs true gates
    'conc_mse': float,            # concentrations vs true
    'grad_norm': float,           # global gradient norm
    'nfe_mean': float,            # mean solver function evaluations per segment
    'lr': float,
    'z0_noise_sigma': float,
}
```

### NFE Tracking

`torchdiffeq` tracks NFE internally. Access after integration:
```python
# After node.integrate():
nfe = node.nfe  # attribute set by torchdiffeq wrapper
```

If mean NFE > 1000 for 3 consecutive epochs, consider:
1. Loosening rtol/atol to 1e-3/1e-3
2. Adding diagonal preconditioner
3. Switching to dopri5 (fewer evals per step, more steps)

---

## Phase Progression

### A1 → B1

Transition when: val_ionic_state_mse plateaus (patience=50 epochs, no improvement).

B1 setup:
- Freeze Half 1 (attention + MLP + ionic decoder)
- Unfreeze Half 2 (conductance compression + decoder)
- Same data (T1), same batch size, LR=1e-3 (fresh optimizer for new params)
- Loss adds conductance_mse

### B1 → C (NOT YET IMPLEMENTED)

Requires wiring Stage 2 into node_rollout. The ODE produces z at landmarks; Stage 1 forward() runs compression; Stage 2 produces I_ion from conductance_latent + environment.

### Later Phases

- **Multi-tier data**: Add T2 (restitution), T3 (dynamic), T12 (celltypes) in later A/B reruns
- **z0 noise schedule**: Start at 0, increase to 1e-3 then 1e-2 if val loss plateaus
- **Solver tolerance schedule**: Start tight (1e-4/1e-5), loosen if NFE is prohibitive

---

## Data Pipeline

### Existing Infrastructure (reusable)

| Component | Status | Notes |
|-----------|--------|-------|
| CacheBuilder | Ready | T1-T3, T12 cached on SSD |
| V3Preprocessor | Ready | 47-col → named tensors |
| SegmentDataset | Ready | subsample + stride for V trajectory |
| LossNormalizer | Ready | Per-dim min-max, fixed ranges |

### NODE-Specific Data Notes

- **SegmentDataset subsample**: Use subsample=10 (dt=0.1ms) for V trajectory. The ODE solver doesn't need every 0.01ms point — it interpolates linearly between trajectory points. Coarser trajectory = less memory, same physics.
- **Segment length**: 3000 at subsample=10 = 300ms. One full AP. Longer segments (2+ beats) deferred to later phases.
- **V trajectory for ODE**: `segment['Vm']` shape (B, T) fed to `node.set_v_trajectory()`. `segment['dt']` used to build cumulative t_grid via `build_t_grid()`.
- **Ground truth at landmarks**: `segment['ionic_states'][:, idx, :]` where idx is the nearest timestep to each t_eval point. Scaffold loss at these 20 points.

---

## Hardware

- GPU: NVIDIA RTX PRO 4500 Blackwell (33.7 GB VRAM)
- Model: 1,691 total params (1,408 inference + 283 scaffold). VRAM never a constraint.
- Training cost bottleneck: ODE solver compute (dopri8 function evaluations), not memory.
- Estimated: ~200-1000 NFE per 300ms segment. Each NFE = 1 forward pass of stage1.dzdt().

---

## References

- Model code: `Surrogate/surrogate/model/` (stage1.py, node.py, ionic_surrogate_v3.py)
- Training code: `Surrogate/surrogate/training/node_rollout.py`
- Data cache: `/tmp/surrogate_cache/` (tier01-03, tier12)
- Raw data: `/media/HDD/surrogate_data/raw/` (tier01-12)
- Architecture: KNOWLEDGE.md Sections 3, 5b
- Stiffness analysis: KNOWLEDGE.md Section 5b "Stiffness Analysis"
- Discrete training post-mortem: KNOWLEDGE.md Section 5 "Why the Discrete Approach Failed"
- Archived discrete strategy: `Surrogate/archive/TRAINING_STRATEGY_discrete.md`
