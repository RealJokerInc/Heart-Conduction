# Phase 4 Reality Check

Date: 2026-04-19
Agent: Claude Opus 4.7 (Phase 4.0 exploratory step)
Scope: verify PLAN Phase 4 assumptions against actual codebase + runtime before Steps 4.1–4.4 get re-specified via `/blueprint-revise`.

> **Purpose**: Round-3 audit (30 issues, 4 critical, 15 Phase-4-scoped deferred) showed that Phase 4's step specifications rested on invented or incorrect assumptions. This doc reads the real code, documents what's there, and ends with a concrete "revisions needed" section that drives the next revise pass.

---

## 1. Dataset API

**File**: `Surrogate/surrogate/training/datasets.py` (128 lines)

`SegmentDataset.__init__(cached_data, segment_length, stride=None, subsample=1)`

- `cached_data` is a dict of tensors already loaded into memory (`torch.load` of a `.pt` file).
- `segment_length` = number of timesteps in one segment AFTER subsampling.
- `stride` in raw timesteps; defaults to `raw_length // 2`.
- `subsample` = take every Nth raw timestep; `dt` multiplied by `subsample` in the output.
- **There is NO `train_bcls` / `val_bcls` filter.** Round-3 HIGH-1 confirmed: those are fabricated.

`SegmentDataset.__getitem__(i)` returns a dict with up to 12 keys, float64:
```python
{'Vm', 'dt', 'I_stim', 'I_ion', 'clamp_mask', 'concentrations',
 'gates', 'ionic_states', 'conductance_products', 'E',
 'gate_inf', 'gate_tau'}
```
Only keys present in `cached_data` are included (see `SegmentDataset.KEYS` filter at line 102).

**Implication for Phase 4**: a Trainer batch coming from SegmentDataset is a **dict of (B, T, ...) tensors**, NOT a `(x, y)` tuple. Our `_to_device_and_dtype` in cardiac_ml/training/trainer.py already handles dict batches correctly (float-only promotion, non-tensor pass-through).

## 2. Dataloader construction

**Files**: `Surrogate/surrogate/training/train_node.py:48-72` (`make_dataloaders`), `Surrogate/surrogate/training/data_cache.py:1-80`

The real tier-cache pattern:
- `make_dataloaders(cache_dir, tier, segment_length, subsample, stride, batch_size)` loads two `.pt` files:
  - `cache_dir/tier{N:02d}_train.pt`
  - `cache_dir/tier{N:02d}_val.pt`
- Each `.pt` is a dict produced by `data_cache.CacheBuilder` from the raw tier HDF5 at `raw_dir/tier{N:02d}.h5`.
- Train/val SPLIT is by protocol name (not by timestep). `data_cache.py:25-48` `DEFAULT_VAL_PROTOCOLS` maps each tier to the held-out BCL/DI names:
  - Tier 1: `steady_bcl400/600/800/2000_dt0.01` → val set
  - Tier 2: `s1s2_di75/200/800_dt0.01` → val set
  - Tier 3: `ramp_300to1000_dt0.01` → val
  - Tier 12: mix
- The DataLoader in `train_node.py:66-68` uses `shuffle=True, drop_last=False` for train, `shuffle=False` for val.

**Implication for Phase 4**: the `conf/data/t1_multi_bcl.yaml` must be rewritten to:
1. Call `make_dataloaders` via a thin factory helper, OR
2. Explicitly instantiate two `SegmentDataset`s from pre-loaded `.pt` dicts + wrap in DataLoader.

Option 1 is cleaner — expose `make_dataloaders` as a Hydra `_target_` returning a `{"train": ..., "val": ...}` dict the Trainer consumes. OR split into `data.train._target_: make_tier_loader(tier, split='train')` and `data.val._target_: make_tier_loader(tier, split='val')`.

## 3. Phase naming for multi_bcl_002

**Critical finding**: `Surrogate/runs/multi_bcl_002/log.jsonl` was **NOT produced by `train_node.py`**. The log schema includes `val_per_bcl` (per-BCL validation breakdown) which `train_node.py`'s `train_phase()` does not emit.

The actual producer is `Surrogate/run_multi_bcl.py` (at Surrogate/ root, not under surrogate/training/), which is a fully independent script with its own training loop. It:
- Manually extracts individual beats via `extract_beats(data, bcls, n_beats=20)` (run_multi_bcl.py:25)
- Subsamples by `SUBSAMPLE=10` constant → dt=0.1ms
- Calls `node_rollout(node, batch, phase_name=?, ...)` directly
- Logs `val_per_bcl[bcl]` for each validation BCL

**Phase name for the oracle run**: grep for `phase_name=` in run_multi_bcl.py to find the exact value. This was not done in this reality check — flagged as an open Step 4.2 task.

Most likely candidates based on phases.py:
- `"A1"` → `loss_fn="ionic_state"` (Half 1 training: ionic rate MLP + ionic state decoder)
- `"ionic_state"` → same as A1 via `_compute_node_loss` phase_name matching (node_rollout.py:160)

Our current plan placeholder `phase_name: "A1"` is defensible. **Action for Phase 4 revise**: read run_multi_bcl.py's call to `node_rollout` and record the literal value there.

## 4. Parity gate source — JSONL vs MLflow mismatch (Round-3 CRITICAL-2)

**The Trainer logs to MLflow. `SessionParityGate` as sketched reads JSONL. Mismatch.**

Options:
- **(a)** Add a JSONL shadow-log callback to `cardiac_ml/training/` that writes per-epoch `{epoch, train_loss, val_loss}` to a file. Gate reads this file.
- **(b)** Rewrite `compare_to_session25` to query MLflow's metric history via the MLflow client instead of reading JSONL.
- **(c)** Skip the gate entirely — rely on EarlyStopping + post-hoc parity verification at Phase 4 end.

**Recommendation**: (c) for simplicity. The log.jsonl range is only 8 epochs (§5), so a mid-flight gate is of limited value anyway. Post-hoc parity check after fit() completes is sufficient. If future parity runs with longer reference logs are needed, add (b) since it avoids duplicating MLflow's built-in metric storage.

## 5. Reference log coverage

**File**: `Surrogate/runs/multi_bcl_002/log.jsonl` — **8 lines, epochs 0–7 only**.

Schema per line: `{epoch, train_loss, val_loss, val_per_bcl: {bcl: loss}, lr, elapsed_s}`.

- Epoch 0 val_loss: 0.01281
- Epoch 1 val_loss: 0.00842 (best at this epoch)
- Epoch 6 val_loss: 0.00838 (**global best**)
- Epoch 7: early stop or manual interrupt (last recorded epoch)

**Per-BCL breakdown at epoch 6 (oracle best)**:
- BCL 400: 0.00283
- BCL 600: 0.01218
- BCL 800: 0.01188
- BCL 2000: 0.00678

**Implication**: PLAN.md parity threshold `val_loss ≤ 0.008` is BELOW the oracle's actual best (0.00838). A 5% envelope `≤ 1.05 × 0.00838 = 0.0088` was added in Round 2 (M-7). This is still the right approach.

**Implication for gate**: `min_epoch=10` locks out the entire reference (epochs 0–7). Either:
- Drop `min_epoch` to 0 (but then early-epoch transient spikes may trip the gate falsely), OR
- Drop the gate (recommendation §4 above).

`elapsed_s` in the log says ~130s per epoch → 7 epochs = ~15 minutes total training time. This is not a long-running job; parity gate overhead isn't justified.

## 6. `cfg.experiment.name` population

**No verification performed in this reality check.** Phase 3 MLflowLoggerCallback already has the `_derive_run_name` fallback (Round-3 C-4) that handles:
- `cfg.experiment` is a string → use directly
- `cfg.experiment.name` exists → use it
- Fall back to Hydra `runtime.choices["experiment"]`
- Fall back to `"cardiac_ml_run"`

`conf/experiment/ionic_node_t1.yaml` has `# @package _global_` but no top-level `name` field. MLflow run name will resolve to `ionic_node_t1_<sha>` via the HydraConfig fallback. **No action needed** — verified robust in Phase 3 tests (`test_derive_run_name_fallback_without_hydra_context`).

## 7. Wall-time budget

From log.jsonl: each epoch takes ~130s. Oracle converged in 7 epochs (~15 minutes total). With `patience=50` early-stopping (run_multi_bcl.py presumably uses similar), a fresh parity run that hits oracle quality in ~20 epochs would take ~45 minutes — much less than the "500 epochs / hours of GPU time" the PLAN assumed.

**Implication for PLAN**:
- Step 4.4 scope is overstated. "500 epochs" should be "20–50 epochs with early stop, ~15–45 min". Update success criteria accordingly.
- No mid-flight gate needed (run is short enough that a failed config wastes < 15 min).

## 8. V-trajectory lifecycle

**File**: `Surrogate/surrogate/model/node.py:26-33`, `Surrogate/surrogate/training/train_node.py:131, 160`

- `node.set_v_trajectory(V_traj, t_grid)` is called INSIDE `node_rollout()` at node_rollout.py:109. Caller does NOT need to call it directly.
- `node.clear_v_trajectory()` **MUST** be called by the caller AFTER `loss.backward()` completes (confirmed: train_node.py:131 train path, train_node.py:160 val path — both clear after the rollout).
- Under `adjoint=False` (the oracle config), clearing isn't strictly required for correctness (backward doesn't re-enter forward), but the existing pattern clears on both paths for safety.

**Implication for Phase 4**: `_on_after_backward` hook is the right mechanism — our Trainer already dispatches it on both train AND val paths. `node_step.py` adapter should set `_on_after_backward = lambda: trainer.model.clear_v_trajectory()`. This was the PLAN Step 4.1 intent and it's correct.

## 9. SHA pin robustness — `8f191f77`

**Verified 2026-04-19**: `git log --oneline 8f191f77 -1` returns:
> `8f191f77 Surrogate: NODE pivot validated, dense MLP replaces VoltageAttention`

Commit message explicitly says: *"Multi-BCL T1: val 0.008 across 5 train + 4 val BCLs"*. So the parity target is associated with this exact SHA. Good.

`git log --oneline 8f191f77..HEAD -- Surrogate/surrogate/model/` returns **empty** (model tree unchanged since 8f191f77). The pin is coincidentally current. Any future work in `Surrogate/surrogate/model/` will require updating the pin or deferring Phase 4.

**Action for Phase 4 revise**: keep the `git diff --quiet 8f191f77 -- Surrogate/surrogate/model/` precondition check from R2. It's correct and currently passes.

## 10. Scaffold decoder presence at 8f191f77

Not directly inspected via `git show` in this check. But PLAN §7.2 of REQUIREMENTS references `ionic_state_decoder` and `gate_conductance_decoder` as scaffold decoders that exist on `IonicStage1`. Per `Surrogate/surrogate/training/phases.py:38-53` at HEAD:

```python
_HALF1_PARAMS = [
    "stage1.ionic_rate_mlp.*",
    "stage1.ionic_state_decoder.*",
]
_HALF2_PARAMS = [
    "stage1.gate_conductance_mlp.*",
    "stage1.gate_conductance_linear.*",
    "stage1.gate_conductance_logit",
    "stage1.gate_conductance_decoder.*",
]
```

The decoders exist at HEAD. Since model tree is unchanged since 8f191f77, they exist at the pin too.

**Per-component scaffold metrics**: `_compute_node_loss` in `node_rollout.py:160-191` produces `ionic_state_mse`, `conc_mse`, `conductance_mse` per phase. These will flow through `node_rollout`'s return dict. The Phase 4 adapter should surface all of them as detached extras so Phase 4 parity debugging has per-component visibility (matches oracle log format).

---

## 11. Bonus finding — `node.nfe` is a latent bug

`train_node.py:138` reads `node.nfe`. `grep -rn "nfe" Surrogate/surrogate/` shows NO `.nfe` assignment anywhere in `node.py`. Running `train_node.py` as-is would `AttributeError` at the first batch.

This is a latent bug in the frozen-input code. **Action**: Phase 4 must NOT rely on `node.nfe`. Our Step 4.1 adapter was already rewritten to drop NFE (Round-3 C-2). Keep that.

## 12. Bonus finding — data path

- Actual raw data: `/media/HDD/norepinephrine/surrogate_data/raw/` (tier01/02/03.h5 verified)
- Hardcoded in `data_cache.py:7` docstring and `Surrogate/run_multi_bcl.py`: `/media/HDD/surrogate_data/raw` (WRONG — doesn't exist)
- SSD cache `/tmp/surrogate_cache/` does NOT exist currently

**Action**: Step 4.2 YAML's OmegaConf default must be `/media/HDD/norepinephrine/surrogate_data/raw`. Also: the cache build step must run (via `data_cache.CacheBuilder.build_all(tiers=[1])`) before training can start, OR direct-from-HDD training (slower I/O) can be used.

## 13. Bonus finding — `run_multi_bcl.py` is the real oracle, not `train_node.py`

PLAN consistently references `train_node.py` as the oracle. But the parity target run (multi_bcl_002) was produced by `Surrogate/run_multi_bcl.py`:
- Different dir (Surrogate/ root, not under surrogate/training/)
- Different training loop (manual beat extraction, no SegmentDataset)
- Different log schema (includes `val_per_bcl`)

This changes Phase 4 architecture significantly:
- `node_step.py` adapter can still call `node_rollout()` — that is shared.
- But the DATA PATH is different: multi_bcl_002 uses `extract_beats()` from the raw T1 cache, not `SegmentDataset` via `make_dataloaders(tier=1)`.
- To replicate parity, we either replicate `run_multi_bcl.py`'s beat extraction in the harness, OR adapt the data config.

---

## Revisions needed before Step 4.1 is executable

### Addresses Round-3 CRITICAL + HIGH deferred items:

1. **[C-1, §5]** Drop the mid-flight parity gate entirely (recommendation in §4). Reference log is too short to support it, and run time (~15 min) doesn't justify gate overhead. Remove `SessionParityGate`, remove `cardiac_ml/reference/session25_log.jsonl` copy step, remove `compare_to_session25` utility from PLAN. Replace Step 4.4 verification with a simple post-fit threshold check: `best val_loss ≤ 1.05 × 0.00838 = 0.0088`.

2. **[C-2, §4]** Resolved by (1) — no gate means no JSONL-vs-MLflow mismatch.

3. **[C-3, §3]** Read `Surrogate/run_multi_bcl.py` during Step 4.2 to find the literal `phase_name=` argument. Set `conf/training/node.yaml:phase_name` to that value. Most likely `"A1"`; verify.

4. **[C-4]** Already fixed in Phase 3 via `_derive_run_name` fallback. No action.

5. **[HIGH-1, §1 and §2]** Rewrite `conf/data/t1_multi_bcl.yaml`:
   - Remove fictional `train_bcls` / `val_bcls` fields.
   - Instead, point `data.train` at a factory that calls `make_dataloaders(cache_dir, tier=1, ...)` and returns the train DataLoader. Same for `data.val`.
   - OR (better for multi_bcl_002 parity): replicate `run_multi_bcl.py`'s `extract_beats()` flow in the harness via a dedicated `cardiac_ml/data/multi_bcl_loader.py` that yields beat segments.

6. **[HIGH-3, §8]** `_on_after_backward` hook pattern is correct. Step 4.1 pseudocode should set `_on_after_backward = lambda: trainer.model.clear_v_trajectory()`. No phase-ordering issue — Trainer already dispatches the hook (Phase 3 commit).

7. **[HIGH-5, §9]** SHA pin 8f191f77 still valid. `git diff --quiet 8f191f77 -- Surrogate/surrogate/model/` passes at HEAD. Keep the precondition as-is.

### Addresses Round-4 HIGH deferred items:

8. **[R4-HIGH-2, §5]** PLAN.md 0.008 vs 0.0088 drift: update Success Criteria line 17, Phase 4 Context, Phase 4 Exit, Step 4.4 Read-First, Step 4.4 Why — all to use `0.0088` or explicit "1.05 × oracle best = 0.0088".

### Addresses Round-4 MED deferred items:

9. **[R4-MED-1]** Phase 4 Verification block currently has `# parity verification command from Step 4.4` as a comment. Replace with the actual mlflow-search command from Step 4.4 verify block.

10. **[R4-MED-8]** Add checklist item to Step 4.1: *"Remove `surrogate.training.node_step` from `cardiac_ml/tests/_deferred_targets.py::DEFERRED`"*. (Actually I've already been removing entries as each step lands — see _deferred_targets.py state.)

11. **[R4-MED-12]** Step 5.4 grep `--exclude-dir=runs` over-excludes top-level `runs/`. Replace with `--exclude-dir=archive`.

### New issues discovered during reality check:

12. **[§12]** Data path: set `SURROGATE_CACHE_DIR` default to `/media/HDD/norepinephrine/surrogate_data/raw/`, NOT `/media/HDD/surrogate_data/raw/`. Update all PLAN references.

13. **[§13]** Oracle is `Surrogate/run_multi_bcl.py`, NOT `train_node.py`. Update all Step 4.x "Read First" references. The data-loading path in particular differs.

14. **[§7]** Wall-time: Step 4.4 currently scopes "500 epochs, hours". Actual: ~7–20 epochs, ~15–45 min. Update scope + threshold to match.

15. **[§11]** `node.nfe` is a latent bug — do not attempt to surface NFE as a metric. Already dropped in Round-3 C-2. No regression.

---

## Summary for the next `/blueprint-revise` pass

The Phase 4 architecture was framed around parity against an assumed "train_node.py 500-epoch training with JSONL+MLflow shadow log". Reality:

- Oracle is `Surrogate/run_multi_bcl.py`, which produces an 8-epoch, ~15-minute run.
- `train_node.py` has a latent `nfe` AttributeError bug.
- Parity threshold is `≤ 0.0088` (oracle best 0.00838 + 5% envelope), not `0.008`.
- Mid-flight gate isn't justified (short run) and had a JSONL/MLflow mismatch anyway — drop it.
- `conf/data/t1_multi_bcl.yaml` must be rewritten around `make_dataloaders(tier=1)` OR a custom multi-BCL beat extractor.
- Data path is `/media/HDD/norepinephrine/surrogate_data/raw/`, not `/media/HDD/surrogate_data/raw/`.

Phase 4 simplifies substantially: no gate, no reference log copy, shorter run time, one clearly-specified data config. Remaining complexity is the data-loading adapter choice (§5 of "Revisions needed"). That's the one real design question for the next revise pass.

**Confidence level**: high for §1, §2, §5, §7, §8, §9, §10, §12. Medium for §3 (phase_name literal value — requires reading run_multi_bcl.py in full). Medium for §13 (whether to use make_dataloaders or replicate extract_beats — depends on what run_multi_bcl.py actually does). Both can be resolved cheaply during the next revise pass.
