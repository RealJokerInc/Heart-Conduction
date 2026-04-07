# Training Monitor — Ionic Surrogate v3

> Design for training monitoring with pause/resume, logging, and agent-readable status.

---

## Pause/Resume via Flag File

```
/home/norepinephrine/Documents/Heart-Conduction/Surrogate/training_control.json
```

```json
{
    "status": "running",
    "current_phase": "B3",
    "current_epoch": 42,
    "current_rollout": 100,
    "best_val_loss": 0.00123,
    "scheduled_sampling_p": 0.5,
    "message": ""
}
```

**Status values**: `"running"`, `"pause_requested"`, `"paused"`, `"stop_requested"`

**Training loop behavior** (checked every 50 batches):

```python
def check_control(control_path):
    ctrl = json.loads(control_path.read_text())
    if ctrl['status'] == 'pause_requested':
        save_checkpoint(path='checkpoints/pause_checkpoint.pt')
        ctrl['status'] = 'paused'
        control_path.write_text(json.dumps(ctrl, indent=2))
        while True:
            time.sleep(5)
            ctrl = json.loads(control_path.read_text())
            if ctrl['status'] == 'running':
                break
            if ctrl['status'] == 'stop_requested':
                raise TrainingStoppedError()
    elif ctrl['status'] == 'stop_requested':
        save_checkpoint(path='checkpoints/stop_checkpoint.pt')
        raise TrainingStoppedError()
```

**To pause**: Set `"status": "pause_requested"` in the JSON file. Training saves checkpoint and blocks.

**To resume**: Set `"status": "running"`. Training continues from where it paused.

**To stop**: Set `"status": "stop_requested"`. Training saves checkpoint and exits.

---

## Training Log Format

```
/home/norepinephrine/Documents/Heart-Conduction/Surrogate/runs/{run_name}/
├── training_control.json      # symlinked from Surrogate/ root
├── training_log.jsonl         # one JSON line per batch
├── phase_summary.json         # per-phase best metrics
├── config.json                # all hyperparameters (frozen at run start)
├── checkpoints/
│   ├── best_A1.pt
│   ├── best_A2.pt
│   ├── best_A3.pt
│   ├── best_B1.pt
│   ├── ...
│   ├── best_E.pt
│   ├── latest.pt             # most recent (any phase)
│   └── pause_checkpoint.pt   # saved on pause
└── tensorboard/              # TensorBoard event files
```

### training_log.jsonl

One JSON object per line, one line per batch:

```json
{"phase": "B3", "epoch": 12, "batch": 340, "step": 15340, "loss": 0.00234, "lr": 3.2e-4, "grad_norm": 0.87, "rollout": 100, "sched_p": 0.5, "wall_s": 0.34, "timestamp": "2026-03-30T14:22:01"}
```

| Field | Type | Description |
|-------|------|-------------|
| phase | str | Current training phase |
| epoch | int | Epoch within current phase |
| batch | int | Batch within current epoch |
| step | int | Global step counter (never resets) |
| loss | float | Batch loss value |
| lr | float | Current learning rate |
| grad_norm | float | Pre-clip gradient norm |
| rollout | int | Current rollout length |
| sched_p | float | Scheduled sampling probability |
| wall_s | float | Wall-clock seconds for this batch |
| timestamp | str | ISO 8601 |

### phase_summary.json

Updated at the end of each epoch:

```json
{
    "A1": {
        "best_val_loss": 3.2e-5,
        "best_epoch": 23,
        "total_epochs": 30,
        "total_steps": 4500,
        "wall_hours": 0.12,
        "metrics": {"recon_mse": 3.2e-5}
    },
    "B3": {
        "best_val_loss": 0.00123,
        "best_epoch": 42,
        "total_epochs": 55,
        "total_steps": 8200,
        "wall_hours": 2.3,
        "metrics": {"ionic_state_mse": 0.00123, "apd_error_pct": 1.2, "dvdt_max_error_pct": 3.4}
    }
}
```

---

## Checkpoint Contents

```python
checkpoint = {
    # Model
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),

    # Training state
    'phase': current_phase,           # e.g., "B3"
    'epoch': epoch,                   # within current phase
    'step': global_step,              # never resets
    'best_val_loss': best_val_loss,

    # Phase-specific state
    'rollout_length': rollout_length,
    'scheduled_sampling_p': p,

    # Encoder (temporary, Phase A-B only)
    'encoder_state_dict': encoder.state_dict() if encoder else None,

    # Reproducibility
    'config': {
        'ionic_dim': 16,
        'conc_dim': 4,
        'cond_dim': 8,
        'attn_dim': 4,
        'mlp_hidden': 16,
        'comp_h1': 12,
        'comp_h2': 12,
        'n_env': 9,
        'stage2_attn': 4,
        'stage2_dv': 1,
        'stage2_mlp_h': 4,
        'weight_decay': ...,
        'max_grad_norm': 1.0,
        'data_tiers': [...],
        'val_protocols': [...],
    },
    'rng_state': {
        'torch': torch.random.get_rng_state(),
        'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    },
}
```

**Naming**: `best_{phase}.pt` for best validation loss per phase. `latest.pt` overwritten every epoch.

---

## Claude Agent Monitoring

A Claude agent can be manually invoked to inspect training status. Not automatic — the user asks Claude to check.

**Agent reads**:
1. `training_control.json` — current phase, epoch, rollout, status
2. `training_log.jsonl` (tail) — recent loss values, gradient norms, wall time
3. `phase_summary.json` — cross-phase comparison, total wall time

**Agent can**:
- Plot loss curves (read JSONL, identify trends)
- Detect plateaus ("val loss hasn't improved for 20 epochs in B3")
- Detect divergence ("loss increased 5x over last 100 batches")
- Suggest phase transition ("B2 converged, ready for B3")
- Pause training (write `"status": "pause_requested"` to control file)
- Compare metrics across phases (is B3 better than B2?)

**Example agent query**: "Check surrogate training status" triggers reading the three files and producing a summary.

---

## Divergence Detection Heuristics

Built into the training loop (automatic, no agent needed):

| Condition | Action |
|-----------|--------|
| `loss > 3 * running_avg(last_100)` | Log WARNING, continue |
| `loss` is NaN or Inf | Auto-pause, save checkpoint, set status="paused", log ERROR |
| `grad_norm > 10 * running_avg(last_100)` | Log WARNING, continue (clipping handles it) |
| Val loss no improvement for 20 epochs | Log SUGGEST: "Consider transitioning to next phase" |
| Val loss no improvement for 40 epochs | Auto-pause, log "Plateau detected" |

**Running average**: Exponential moving average with alpha=0.01 (smooth, slow-moving baseline).

**NaN recovery**: On NaN/Inf, roll back to `latest.pt` checkpoint (last epoch). If NaN persists after rollback, escalate: load `best_{phase}.pt` and reduce LR by 2x.

---

## Phase Transition Protocol

When the training loop detects convergence (or the agent suggests transition):

1. Save `best_{phase}.pt` checkpoint
2. Update `phase_summary.json` with final metrics
3. Load `best_{phase}.pt` as starting point for next phase
4. Reset optimizer and scheduler (new LR, new cosine schedule)
5. Update `training_control.json` with new phase info
6. Update DataLoader for new tiers/rollout length
7. Reset early stopping counter
8. Log phase transition to `training_log.jsonl`: `{"event": "phase_transition", "from": "B2", "to": "B3", ...}`

**Manual override**: Set `"message": "transition_to:B3"` in control file to force a phase transition.

---

## TensorBoard Integration

Standard PyTorch `SummaryWriter` logging:

| Tag | Frequency | Description |
|-----|-----------|-------------|
| `loss/train` | Every batch | Training loss |
| `loss/val` | Every epoch | Validation loss |
| `lr` | Every batch | Current learning rate |
| `grad_norm` | Every batch | Pre-clip gradient norm |
| `metrics/apd_error` | Every val epoch | APD90 error % |
| `metrics/dvdt_max_error` | Every val epoch | dVm/dt_max error % |
| `phase` | On transition | Phase marker (for vertical lines) |

TensorBoard is supplementary — the JSONL log is the primary record for agent monitoring and reproducibility.
