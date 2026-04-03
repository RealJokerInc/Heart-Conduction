# PLAN: Training Pipeline for Ionic Surrogate v3

Created: 2026-04-02
Engine(s): None (standalone ML pipeline, reads Bidomain V1 HDF5 data)
Research question: [surrogate_pipeline](README.md)
Source: [IDEALOG.md](IDEALOG.md) — 2026-04-02 pre-blueprint decisions + Session 19-20 training strategy

## Objective
Build the complete training pipeline for IonicSurrogateV3: data cache preparation, phase-aware training loop (A1 through E), monitoring/checkpointing infrastructure, and a project-based Claude agent for agentic training oversight. The pipeline trains a 1,534-param ionic surrogate on 608 GB of TTP06 single-cell data across 7 training phases with progressive rollout curriculum.

## Hard Cutoff
Phases 1-4 (through Phase C integration test) are the **minimum viable deliverable**. Phase 5 (T4 shard streaming), Phase 6 (training agent) are stretch goals. If the session runs out of context after Phase 4, the pipeline is usable for Phases A1→C on T1-T3+T12 data. T4 streaming (needed for Phases D-E) can be a follow-up PLAN.

## Success Criteria
- [ ] Data cache builder converts raw HDF5 (T1-T3, T12) to preprocessed `.pt` on SSD
- [ ] Training loop executes all phases A1→A2→A3→B1-B5→C→D→E with correct freeze/unfreeze
- [ ] Checkpoint save/load enables resume from any phase
- [ ] Pause/resume via `training_control.json` works
- [ ] Divergence detection (NaN, plateau) triggers automatic responses
- [ ] Project-based Claude agent can read logs, diagnose, and intervene
- [ ] Phase A1 completes: ionic autoencoder reconstruction MSE < 1e-4
- [ ] All existing tests pass (no regressions) — 83 tests across 4 test files

## Architecture Changes
- NEW: `Surrogate/surrogate/training/data_cache.py` — HDF5→preprocessed .pt cache builder
- NEW: `Surrogate/surrogate/training/datasets.py` — Per-phase Dataset classes (snapshot, pair, segment)
- NEW: `Surrogate/surrogate/training/encoder.py` — Temporary encoder (14→16) for A1 + teacher forcing
- NEW: `Surrogate/surrogate/training/trainer.py` — Phase-aware training loop orchestrator
- NEW: `Surrogate/surrogate/training/phases.py` — Per-phase config (freeze masks, loss, data, hyperparams)
- NEW: `Surrogate/surrogate/training/rollout.py` — Autoregressive rollout with scheduled sampling
- NEW: `Surrogate/surrogate/training/checkpoint.py` — Save/load/resume logic
- NEW: `Surrogate/surrogate/training/monitor.py` — JSONL logging, divergence detection, control file
- NEW: `Surrogate/surrogate/training/metrics.py` — APD, dVm/dt_max, per-phase validation metrics
- NEW: `Surrogate/surrogate/training/shard_loader.py` — T4 shard streaming with prefetch (Phase 5)
- NEW: `Surrogate/train.py` — CLI entry point
- NEW: `Surrogate/tests/test_training.py` — Training pipeline tests
- NEW: `.claude/agents/training-monitor.md` — Project-based Claude agent definition
- MOD: `Surrogate/surrogate/data/storage.py` — Pass `raw_dir` explicitly (default hardcodes stale path)

## Known Failures (from IDEALOG)
- Temporal Transformer (300-pt history): 200M FLOPs, buffer management nightmare — do NOT retry
- Vm history buffer: stimulus artifacts, coverage issues — do NOT retry
- Sigmoid output bounding: vanishing gradients, triple sigmoid path — do NOT use
- Softmax in cross-attention: forces positive weights, breaks driving force sign — do NOT add
- Ohmic/non-Ohmic split: Layer 1 assumption, not Layer 0 — do NOT encode
- Spectral norm: superseded by learned residual mixing — do NOT reintroduce
- BatchNorm: unstable for autoregressive inference (batch=1) — do NOT use
- Dropout: compounds noise over 100K+ steps — do NOT use
- HDD speed: measured 7-246 MB/s depending on block size (USB 3.0, WD Elements). Old 1.26 MB/s figure was stale. Pre-caching to SSD still preferred for repeated loads but HDD is not catastrophically slow.

---

## Phase 1: Data Cache Builder

**Goal**: Convert raw HDF5 tiers on HDD to preprocessed `.pt` files on SSD, ready for instant GPU loading. Compute normalization statistics.
**Tier**: medium
**Estimated scope**: 2 files, ~300 lines, 6 tests

### Phase Context
- Raw data: `/media/HDD/surrogate_data/raw/tier{01-12}.h5` — 47-col float64
- **IMPORTANT**: TraceStorage default path is `/media/norepinephrine/Elements-ext4/surrogate_data/raw/` which is stale. Always pass `raw_dir='/media/HDD/surrogate_data/raw'` explicitly to CacheBuilder, which passes it to TraceStorage.
- Target cache: `/tmp/surrogate_cache/` (or configurable SSD path)
- V3Preprocessor already exists at `Surrogate/surrogate/data/preprocessor.py` — use it as-is
- TraceStorage already exists at `Surrogate/surrogate/data/storage.py` — use for HDF5 reads, pass raw_dir explicitly
- Cache tiers: T1-T3 + T12 (celltypes). T12 required from Phase B1 per TRAINING_STRATEGY.md.
  - T1-T3 preprocessed ≈ 5.5 GB float32. T12 ≈ 5.5 GB float32. Total ≈ 11 GB.
  - SSD has 47 GB free. Fits easily.
- T4 (551 GB) stays on HDD — shard-streamed in Phase 5 of this plan.
- Val split by protocol name: train BCL={300,500,700,1000,1500}, val BCL={400,600,800,2000}
- HDD speed: ~7-246 MB/s (USB 3.0). T1 load from HDD: ~22s. Not a critical bottleneck.
- All tensors float64 in the training pipeline (project convention). Only raw cache files use float32 for storage efficiency; convert back to float64 on load.
- Conda env: `heart-conduction`

### Step 1.1: Data Cache Builder
**Model**: opus

#### Read First
- `Surrogate/surrogate/data/preprocessor.py` — V3Preprocessor.process_segment() interface
- `Surrogate/surrogate/data/storage.py` — TraceStorage.load_trace(), list_protocols()
- `Surrogate/HARDWARE_CONSTRAINTS.md` — SSD/HDD constraints, cache strategy
- `Surrogate/DATA_PIPELINE.md` — Raw format (47-col), preprocessing output dict
- `Surrogate/TRAINING_STRATEGY.md:186-198` — Data curriculum (which tiers per phase)

#### Why
The cache serves two purposes: (1) pre-applies V3Preprocessor once instead of every epoch, and (2) stores preprocessed data on faster SSD for repeated loads across training restarts. HDD is ~7 MB/s with direct I/O (USB 3.0), so T1 loads in ~22s — tolerable but wasteful to repeat.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/training/__init__.py` — empty package init
**Files to create:** `Surrogate/surrogate/training/data_cache.py` — CacheBuilder class

**Interfaces / Signatures:**
```python
class CacheBuilder:
    def __init__(self, raw_dir: str, cache_dir: str, val_protocols: dict[int, list[str]] | None = None):
        """
        raw_dir: path to HDD raw HDF5 directory
        cache_dir: path to SSD cache directory (e.g., /tmp/surrogate_cache)
        val_protocols: {tier_num: [protocol_names]} for validation split
        """

    def build_tier_cache(self, tier: int) -> dict[str, Path]:
        """Preprocess one tier -> train.pt + val.pt on SSD. Returns paths."""

    def build_all(self, tiers: list[int] = [1, 2, 3, 12]) -> dict:
        """Build cache for multiple tiers. Returns summary dict."""

    def is_cached(self, tiers: list[int] = [1, 2, 3, 12]) -> bool:
        """Check if all tiers are already cached on SSD."""

    def compute_normalization_stats(self, tiers: list[int] = [1, 2, 3]) -> dict:
        """Compute env token normalization stats from cached data. Save to cache_dir/norm_stats.pt."""

    def load_tier(self, tier: int, split: str = 'train') -> dict[str, Tensor]:
        """Load preprocessed tier from cache. Returns dict of named tensors."""
```

#### Pseudocode
```
build_tier_cache(tier):
    storage = TraceStorage(raw_dir)
    preprocessor = V3Preprocessor()
    protocols = storage.list_protocols(tier)

    train_data = {key: [] for key in preprocessor output keys}
    val_data = {key: [] for key in preprocessor output keys}

    for proto in protocols:
        trace = storage.load_trace(tier, proto)
        processed = preprocessor.process_segment(trace.data)
        target = val_data if proto in val_protocols[tier] else train_data
        for key, tensor in processed.items():
            target[key].append(tensor)

    # Concatenate along time dimension
    train_tensors = {k: torch.cat(v, dim=0).float() for k, v in train_data.items()}
    val_tensors = {k: torch.cat(v, dim=0).float() for k, v in val_data.items()}

    # Save as .pt
    torch.save(train_tensors, cache_dir / f'tier{tier:02d}_train.pt')
    torch.save(val_tensors, cache_dir / f'tier{tier:02d}_val.pt')

compute_normalization_stats(tiers):
    # Load cached train splits, stack env tokens [Vm, E_Na, E_K, E_Ca, E_Ks, Na_i, K_i, Ca_i, Ca_ss]
    # Compute min, max, mean, std across all timesteps
    # Save shift/scale to cache_dir/norm_stats.pt
    # These are VALIDATION stats, not used in model (model has fixed nernst.py constants)
    # But useful for monitoring and diagnostics
```

#### Test Spec
- `test_training.py::test_cache_builder_creates_files` — Setup: fake tier HDF5 in /tmp. Expected: train.pt and val.pt created, loadable.
- `test_training.py::test_cache_builder_split` — Setup: fake tier with known protocols. Expected: val protocols in val.pt, rest in train.pt.
- `test_training.py::test_cache_builder_shapes` — Setup: build cache from fake data. Expected: all tensors have correct shapes, consistent T dimension.
- `test_training.py::test_normalization_stats` — Setup: build cache, compute stats. Expected: shift/scale tensors of shape (9,), reasonable ranges.

#### Checklist
- [ ] Create `Surrogate/surrogate/training/__init__.py`
- [ ] Implement CacheBuilder in `data_cache.py`
- [ ] Pass `raw_dir` explicitly to TraceStorage (do NOT use its default `/media/norepinephrine/Elements-ext4/...`)
- [ ] Default tiers: [1, 2, 3, 12]. T12 required from Phase B1.
- [ ] Default val_protocols: T1 val={steady_bcl400, steady_bcl600, steady_bcl800, steady_bcl2000}. T12 val protocols: same BCL pattern for ENDO/M_CELL.
- [ ] Validate protocol names against actual HDF5 group names (warn if val_protocol not found in tier)
- [ ] Implement `is_cached()` method
- [ ] Handle HDD not mounted gracefully (clear error message)
- [ ] Store metadata (tier, n_train_steps, n_val_steps, protocol_names) alongside tensors
- [ ] Write tests

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate
conda run -n heart-conduction python -m pytest tests/test_training.py -v -k "cache"
```

#### Exit Criteria
- [ ] Cache builder creates .pt files from fake HDF5
- [ ] Val/train split works correctly by protocol name
- [ ] Normalization stats computed and saved
- [ ] All cache tests pass

#### Risk
HDD mount point: currently `/media/HDD/` (ext4 partition). TraceStorage default path is stale (`/media/norepinephrine/Elements-ext4/`). Mitigation: always pass raw_dir explicitly. CacheBuilder constructor validates the path exists before proceeding.

### Step 1.2: Per-Phase Dataset Classes
**Model**: opus

#### Read First
- `Surrogate/surrogate/training/data_cache.py` — CacheBuilder.load_tier() output format
- `Surrogate/TRAINING_STRATEGY.md:26-147` — Phase A1/A2/A3/B data requirements
- `Surrogate/DATA_PIPELINE.md:86-180` — Per-phase sample shapes, segment windowing

#### Why
Each training phase needs different data: A1 needs random state snapshots, A2 needs consecutive pairs, A3 needs single timesteps with carried_state, B-E need contiguous segments of rollout_length. A unified Dataset interface lets the trainer swap data without changing the training loop.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/training/datasets.py`

**Interfaces / Signatures:**
```python
class SnapshotDataset(torch.utils.data.Dataset):
    """Phase A1/A3: random single-timestep samples. Returns (ionic_states, concentrations, conductance_products)."""
    def __init__(self, cached_data: dict[str, Tensor]):
        # cached_data from CacheBuilder.load_tier()
    def __len__(self) -> int: ...
    def __getitem__(self, idx) -> dict[str, Tensor]: ...

class PairDataset(torch.utils.data.Dataset):
    """Phase A2: consecutive (t, t+1) pairs. Returns (state_t, conc_t, Vm_t, dt_t, conc_t+1)."""
    def __init__(self, cached_data: dict[str, Tensor]):
    def __len__(self) -> int: ...  # T - 1
    def __getitem__(self, idx) -> dict[str, Tensor]: ...

class SegmentDataset(torch.utils.data.Dataset):
    """Phase B-E: contiguous segments of rollout_length. Returns dict of (rollout_length, ...) tensors."""
    def __init__(self, cached_data: dict[str, Tensor], segment_length: int, stride: int | None = None):
        # stride defaults to segment_length // 2 (50% overlap)
    def __len__(self) -> int: ...
    def __getitem__(self, idx) -> dict[str, Tensor]: ...

def merge_tier_datasets(datasets: list[Dataset]) -> ConcatDataset:
    """Merge datasets from multiple tiers."""
```

#### Pseudocode
```
SnapshotDataset.__getitem__(idx):
    return {
        'ionic_states': self.ionic_states[idx],       # (14,)
        'concentrations': self.concentrations[idx],    # (4,)
        'conductance_products': self.cond_products[idx], # (5,)
        'Vm': self.Vm[idx],                            # scalar
        'dt': self.dt[idx],                            # scalar
    }

PairDataset.__getitem__(idx):
    return {
        'ionic_states_t': self.ionic_states[idx],
        'concentrations_t': self.concentrations[idx],
        'Vm_t': self.Vm[idx],
        'dt_t': self.dt[idx],
        'concentrations_t1': self.concentrations[idx + 1],  # target
    }

SegmentDataset.__getitem__(idx):
    start = self.starts[idx]  # precomputed start indices
    end = start + self.segment_length
    return {key: self.data[key][start:end] for key in self.data}
```

#### Test Spec
- `test_training.py::test_snapshot_dataset_shapes` — Expected: each sample has correct tensor shapes
- `test_training.py::test_pair_dataset_consecutive` — Expected: t+1 sample follows t sample
- `test_training.py::test_segment_dataset_contiguous` — Expected: segment is contiguous slice
- `test_training.py::test_segment_dataset_overlap` — Expected: 50% overlap produces correct count

#### Checklist
- [ ] Implement SnapshotDataset
- [ ] Implement PairDataset
- [ ] Implement SegmentDataset with configurable stride
- [ ] Implement merge_tier_datasets
- [ ] All datasets return float64 tensors (convert from float32 cache on load)
- [ ] Write tests

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate
conda run -n heart-conduction python -m pytest tests/test_training.py -v -k "dataset"
```

#### Exit Criteria
- [ ] All three dataset classes work with cached data format
- [ ] Shapes match DATA_PIPELINE.md specs
- [ ] All dataset tests pass

#### Risk
SegmentDataset with large rollout_length (10K+) may produce few segments from short protocols — mitigation: merge across tiers via ConcatDataset.

### Phase 1 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate
conda run -n heart-conduction python -m pytest tests/test_training.py -v -k "cache or dataset"
conda run -n heart-conduction python -m pytest tests/ -v  # ALL tests, no regressions (83 across 4 files)
```

### Phase 1 Exit Criteria
- [ ] All new tests pass
- [ ] All existing 83 tests pass (no regressions: test_model 25 + test_preprocessing 7 + test_data_generation 32 + test_ord_data 19)
- [ ] CacheBuilder can process a real tier from HDD (manual test with tier01.h5)
- [ ] T12 cached alongside T1-T3
- [ ] Dataset classes produce correct shapes for all three patterns

### Phase 1 Cleanup
- float64 consistency — datasets return float64, cache stores float32 (storage only)
- V5.3 not modified
- No code duplication — reuse V3Preprocessor and TraceStorage, do not copy

**-> Commit point: git commit after Phase 1 passes**

---

## Phase 2: Temporary Encoder + Training Core

**Goal**: Build the temporary encoder (Phase A1) and the core training loop that orchestrates all 7 phases with correct freeze/unfreeze, loss functions, and optimizer management.
**Tier**: large
**Estimated scope**: 4 files, ~800 lines, 12 tests

### Phase Context
- IonicSurrogateV3 model is at `Surrogate/surrogate/model/ionic_surrogate_v3.py`
- IonicStage1 has scaffold decoders: `ionic_state_decoder` (16→14), `gate_conductance_decoder` (8→5)
- Encoder is TEMPORARY: trained in A1, used for teacher forcing in B, discarded after B
- Single loss per phase — NO multi-objective weighting
- Phase B uses scheduled sampling: probability p of using model's own prediction ramps from 0.1→1.0
- Stage 2 reads PREVIOUS step's conductance_latent and concentrations (operator splitting)
- Initialization: ionic latent = zeros, concentrations = [Na_i=10, K_i=138, Ca_i=0.0001, Ca_ss=0.0002]
- AdamW optimizer, cosine LR decay per phase, gradient clipping max_norm=1.0
- LR resets at each phase transition
- TRAINING_STRATEGY.md is the authoritative reference for all hyperparameters

### Step 2.1: Temporary Encoder
**Model**: opus

#### Read First
- `Surrogate/surrogate/model/stage1.py:38-39` — N_IONIC_TARGETS=14, N_CONDUCTANCE_TARGETS=5
- `Surrogate/TRAINING_STRATEGY.md:26-43` — Phase A1 spec (encoder 14→16, decoder 16→14)
- `Surrogate/DESIGN_RATIONALE.md:167-173` — Initialization philosophy (zeros, not encoder output)

#### Why
Phase A1 trains an autoencoder (encoder: 14→16, decoder: existing ionic_state_decoder 16→14) to bootstrap the latent space. The encoder maps true ionic states to the 16-dim latent, creating a meaningful initial mapping for teacher forcing in Phase B. The encoder is TEMPORARY — never part of inference.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/training/encoder.py`

**Interfaces / Signatures:**
```python
class TemporaryEncoder(nn.Module):
    """Maps true ionic states (14) to ionic latent (16). Training scaffold only."""
    def __init__(self, n_ionic_targets: int = 14, ionic_dim: int = 16):
    def forward(self, ionic_states: Tensor) -> Tensor:
        """(B, 14) -> (B, 16)"""

def make_carried_state_from_encoder(encoder: TemporaryEncoder, ionic_states: Tensor, concentrations: Tensor) -> Tensor:
    """Convenience: encoder(states) + cat(conc) -> carried_state (B, 20)"""
```

#### Pseudocode
```
TemporaryEncoder:
    self.net = nn.Sequential(
        nn.Linear(n_ionic_targets, ionic_dim),
        nn.GELU(),
        nn.Linear(ionic_dim, ionic_dim),
    )
    # Xavier init

forward(ionic_states):
    return self.net(ionic_states)

make_carried_state_from_encoder(encoder, ionic_states, concentrations):
    latent = encoder(ionic_states)          # (B, 16)
    return torch.cat([latent, concentrations], dim=-1)  # (B, 20)
```

#### Test Spec
- `test_training.py::test_encoder_shape` — Expected: (B, 14) → (B, 16)
- `test_training.py::test_encoder_differentiable` — Expected: gradients flow through encoder
- `test_training.py::test_make_carried_state` — Expected: (B, 20) output, correct dim split

#### Checklist
- [ ] Implement TemporaryEncoder (simple 2-layer MLP with GELU)
- [ ] Implement make_carried_state_from_encoder helper
- [ ] Xavier init on weights
- [ ] Write tests

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate
conda run -n heart-conduction python -m pytest tests/test_training.py -v -k "encoder"
```

#### Exit Criteria
- [ ] Encoder produces correct shapes
- [ ] Gradients flow

#### Risk
Encoder too expressive → latent not learnable by attention — mitigation: keep it simple (2-layer, no bottleneck). If A1 converges but B fails, the encoder may be creating a latent space that attention can't reproduce. Monitor attention-reconstructed vs encoder-reconstructed latent in Phase B.

### Step 2.2: Phase Configuration
**Model**: opus

#### Read First
- `Surrogate/TRAINING_STRATEGY.md` — Full document, all phase specs
- `Surrogate/surrogate/model/ionic_surrogate_v3.py:73-116` — Model constructor params
- `Surrogate/surrogate/model/stage1.py:89-159` — IonicStage1 parameter groups

#### Why
Each phase has different frozen/unfrozen parameters, loss function, data requirements, batch size, LR, and transition criteria. Centralizing this as declarative config objects keeps the training loop clean and makes phase transitions a simple config swap.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/training/phases.py`

**Interfaces / Signatures:**
```python
@dataclass
class PhaseConfig:
    name: str                           # "A1", "A2", ..., "E"
    trainable_params: list[str]         # parameter name patterns to unfreeze
    loss_fn: str                        # "autoencoder", "concentration", "conductance", "ionic_state", "I_ion"
    data_tiers: list[int]              # which tiers to load
    batch_size: int
    lr: float
    weight_decay: float
    rollout_length: int                # 1 for A phases, 10-10000 for B-E
    scheduled_sampling_p: float        # 0.0 = all teacher forcing, 1.0 = all autoregressive
    transition_metric: str             # metric name to monitor for convergence
    transition_threshold: float        # val metric below this → converged
    patience: int                      # epochs without improvement before transition
    max_epochs: int                    # hard cap

def get_phase_config(phase_name: str) -> PhaseConfig: ...
def get_all_phases() -> list[PhaseConfig]: ...
def get_freeze_mask(model: IonicSurrogateV3, phase: PhaseConfig) -> dict[str, bool]: ...
```

#### Pseudocode
```
PHASE_CONFIGS = {
    "A1": PhaseConfig(
        name="A1",
        trainable_params=["encoder.*", "stage1.ionic_state_decoder.*"],
        loss_fn="autoencoder",
        data_tiers=[1],
        batch_size=4096, lr=1e-3, weight_decay=1e-4,
        rollout_length=1, scheduled_sampling_p=0.0,
        transition_metric="val_recon_mse", transition_threshold=1e-4,
        patience=10, max_epochs=100,
    ),
    "A2": PhaseConfig(
        name="A2",
        trainable_params=["stage1.voltage_attention.*"],
        # NOTE: TRAINING_STRATEGY.md says "concentration dims only" but W_q is a single
        # (carried_dim, attn_dim) parameter — cannot selectively freeze rows.
        # Unfreezing all attention params is correct. The loss only backprops through
        # concentration dims, so ionic dims get gradients but the signal is weak.
        # TRAINING_STRATEGY.md acknowledges: "Ionic dims of attention receive gradients
        # but do not need to converge."
        loss_fn="concentration",
        data_tiers=[1],
        batch_size=2048, lr=1e-3, weight_decay=1e-4,
        rollout_length=1, scheduled_sampling_p=0.0,
        transition_metric="val_conc_mse", transition_threshold=1e-6,
        patience=10, max_epochs=100,
    ),
    # B1 example — includes T12 (celltypes) per TRAINING_STRATEGY.md:
    "B1": PhaseConfig(
        name="B1",
        trainable_params=["stage1.voltage_attention.*", "stage1.ionic_mixing_mlp.*", "stage1.ionic_mixing_logit"],
        loss_fn="ionic_state",
        data_tiers=[1, 12],  # T12 enters at B1
        batch_size=1024, lr=5e-4, weight_decay=1e-4,
        rollout_length=1, scheduled_sampling_p=0.1,
        transition_metric="val_ionic_state_mse", transition_threshold=None,
        patience=15, max_epochs=200,
    ),
    # ... A3, B2-B5, C, D, E from TRAINING_STRATEGY.md
}

get_freeze_mask(model, phase):
    # Freeze all params
    # Unfreeze only those matching phase.trainable_params patterns
    # Return dict of {param_name: requires_grad}
```

#### Test Spec
- `test_training.py::test_phase_configs_complete` — Expected: all 11 phases defined (A1,A2,A3,B1-B5,C,D,E)
- `test_training.py::test_freeze_mask_A1` — Expected: only encoder + ionic_state_decoder unfrozen
- `test_training.py::test_freeze_mask_B3` — Expected: Stage 1 dynamics unfrozen, Stage 2 frozen
- `test_training.py::test_freeze_mask_E` — Expected: everything unfrozen

#### Checklist
- [ ] Define all 11 phase configs matching TRAINING_STRATEGY.md exactly
- [ ] Implement get_freeze_mask with pattern matching on param names
- [ ] Encoder params only trainable in A1 (and used for teacher forcing in B)
- [ ] Stage 2 frozen until Phase D
- [ ] Write tests

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate
conda run -n heart-conduction python -m pytest tests/test_training.py -v -k "phase"
```

#### Exit Criteria
- [ ] All phase configs match TRAINING_STRATEGY.md
- [ ] Freeze masks are correct for every phase
- [ ] All phase tests pass

#### Risk
Parameter name patterns may not match actual model parameter names — mitigation: test against real IonicSurrogateV3 instance, print mismatches.

### Step 2.3: Rollout Engine
**Model**: opus

#### Read First
- `Surrogate/surrogate/model/ionic_surrogate_v3.py:118-193` — forward() signature, output dict
- `Surrogate/TRAINING_STRATEGY.md:77-101` — Phase B rollout details, scheduled sampling
- `Surrogate/DATA_PIPELINE.md:127-183` — Rollout execution, teacher forcing, segment windowing
- `Surrogate/DESIGN_RATIONALE.md:179-188` — Rollout curriculum rationale

#### Why
Phases B-E require autoregressive unrolling: the model feeds its own outputs back as inputs for the next step. Scheduled sampling mixes teacher-forced and autoregressive steps. The rollout engine encapsulates this loop, accumulating per-step losses and handling the teacher forcing mechanism.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/training/rollout.py`

**Interfaces / Signatures:**
```python
# Default initial state
INIT_CONC = torch.tensor([10.0, 138.0, 0.0001, 0.0002])  # [Na_i, K_i, Ca_i, Ca_ss]

def rollout(
    model: IonicSurrogateV3,
    segment: dict[str, Tensor],      # from SegmentDataset: (rollout_len, ...)
    encoder: TemporaryEncoder | None, # for teacher forcing (Phase B only)
    scheduled_sampling_p: float,      # probability of using model's own prediction
    loss_fn: callable,                # per-step loss function
    phase_name: str,                  # determines what loss targets to use
) -> dict[str, Tensor]:
    """Execute autoregressive rollout over a segment, accumulating loss.

    Returns dict with:
        'loss': scalar (mean over rollout steps)
        'per_step_losses': (rollout_len,) individual step losses
        'predictions': dict of accumulated predictions for metrics
    """

def compute_phase_loss(
    phase_name: str,
    model_out: dict[str, Tensor],
    segment: dict[str, Tensor],
    t: int,
) -> Tensor:
    """Phase-specific loss at step t. Single MSE, no weighting.
    Dispatch:
        A1: MSE(ionic_state_decoder(encoder(true_14)), true_14)  -- handled outside rollout
        A2: MSE(model_out['concentrations'], segment['concentrations'][:, t+1, :])
        A3: MSE(gate_cond_decoder(cond_lat), segment['conductance_products'][:, t, :])
        B:  MSE(model_out['ionic_state_pred'], segment['ionic_states'][:, t, :])
        C:  MSE(model_out['concentrations'], segment['concentrations'][:, t+1, :])
        D/E: MSE(model_out['I_ion'], segment['I_ion'][:, t])
    """
```

#### Pseudocode
```
rollout(model, segment, encoder, p, loss_fn, phase_name):
    # After DataLoader collation, tensors are (B, T) or (B, T, D)
    B = segment['Vm'].shape[0]
    T = segment['Vm'].shape[1]

    # Initialize state
    carried = zeros(B, 20)  # ionic latent = 0, conc = resting
    carried[:, 16:] = INIT_CONC
    cond_lat_prev = zeros(B, 8)
    conc_prev = INIT_CONC.expand(B, 4)

    total_loss = 0
    for t in range(T):
        Vm_t = segment['Vm'][..., t]
        dt_t = segment['dt'][..., t]

        # Forward pass
        out = model(carried, Vm_t, dt_t, cond_lat_prev, conc_prev)

        # Compute per-step loss
        step_loss = loss_fn(phase_name, out, segment, t)
        total_loss = total_loss + step_loss

        # Update state for next step
        cond_lat_prev = out['conductance_latent']
        conc_prev = out['concentrations']

        # Scheduled sampling: use model output or teacher forcing?
        if encoder is not None and random() > p:
            # Teacher forcing
            true_ionic = segment['ionic_states'][..., t, :]
            true_conc = segment['concentrations'][..., t, :]
            latent = encoder(true_ionic)
            carried = torch.cat([latent, true_conc], dim=-1)
            # Recompute conductance latent from teacher-forced carried_state
            # NOTE: IonicStage1 has no standalone conductance method — replicate the
            # inline logic from stage1.py:212-215 (linear + nonlinear + interpolate):
            s1 = model.stage1
            linear_path = s1.gate_conductance_linear(carried)
            nonlinear_path = s1.gate_conductance_mlp(carried)
            cond_lat_prev = interpolate(linear_path, nonlinear_path, s1.gate_conductance_logit)
            conc_prev = true_conc
        else:
            # Autoregressive
            carried = out['carried_state']

    return {'loss': total_loss / T, ...}
```

#### Test Spec
- `test_training.py::test_rollout_shapes` — Setup: fake segment, rollout_len=10. Expected: loss is scalar, predictions have correct shapes.
- `test_training.py::test_rollout_teacher_forcing` — Setup: p=0.0 (all teacher forcing). Expected: model receives ground truth at every step.
- `test_training.py::test_rollout_autoregressive` — Setup: p=1.0 (all model). Expected: model feeds own output.
- `test_training.py::test_rollout_gradient_flow` — Setup: rollout_len=5. Expected: loss.backward() produces non-zero gradients on model params.

#### Checklist
- [ ] Implement rollout function with scheduled sampling
- [ ] Import `interpolate` from `surrogate.model.stage1` for teacher-forcing conductance recomputation
- [ ] Implement compute_phase_loss for all phase types (A1: MSE(decoded, true_14), A2: MSE(conc_pred, conc_true), A3: MSE(cond_decoded, true_5), B: MSE(ionic_state_pred, true_14), C: MSE(conc, true_conc), D/E: MSE(I_ion_pred, I_ion_true))
- [ ] Handle both single-step (A phases, rollout=1) and multi-step (B-E)
- [ ] PREVIOUS step convention: Stage 2 reads t-1 conductance/concentrations
- [ ] Initialize carried_state correctly (zeros + resting concentrations)
- [ ] Teacher forcing resets conductance latent AND concentrations (not just carried_state)
- [ ] Write tests

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate
conda run -n heart-conduction python -m pytest tests/test_training.py -v -k "rollout"
```

#### Exit Criteria
- [ ] Rollout executes without error for all phase types
- [ ] Gradients flow through the full rollout
- [ ] Teacher forcing correctly replaces model state
- [ ] All rollout tests pass

#### Risk
Gradient accumulation over long rollouts (10K steps) may cause OOM — mitigation: gradient checkpointing if needed, but model is tiny (1,534 params) so likely fine. Test with rollout=100 first.

### Step 2.4: Training Loop Orchestrator
**Model**: opus

#### Read First
- `Surrogate/surrogate/training/phases.py` — PhaseConfig, get_freeze_mask
- `Surrogate/surrogate/training/rollout.py` — rollout function
- `Surrogate/surrogate/training/datasets.py` — Dataset classes
- `Surrogate/surrogate/training/data_cache.py` — CacheBuilder.load_tier
- `Surrogate/TRAINING_STRATEGY.md:165-183` — Optimizer config, LR schedule
- `Surrogate/TRAINING_MONITOR.md:215-229` — Phase transition protocol

#### Why
The trainer orchestrates the full A1→E pipeline: loading data for each phase, configuring optimizer/scheduler, running train/val epochs, detecting convergence, transitioning between phases, and delegating to the checkpoint and monitor subsystems.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/training/trainer.py`

**Interfaces / Signatures:**
```python
class SurrogateTrainer:
    def __init__(
        self,
        model: IonicSurrogateV3,
        cache_dir: str,
        run_dir: str,             # e.g., Surrogate/runs/run_001
        device: str = 'cuda',
        start_phase: str = 'A1',  # for resuming
    ):

    def train(self) -> None:
        """Run full training pipeline from start_phase through E."""

    def train_phase(self, phase: PhaseConfig) -> dict:
        """Train one phase. Returns metrics dict."""

    def train_epoch(self, phase: PhaseConfig, dataloader, optimizer, scheduler) -> float:
        """One training epoch. Returns mean loss."""

    def validate(self, phase: PhaseConfig, dataloader) -> dict:
        """Validation pass. Returns metrics dict."""

    def should_transition(self, phase: PhaseConfig, val_metrics: dict, epochs_no_improve: int) -> bool:
        """Check if phase should transition to next."""

    def transition_phase(self, from_phase: str, to_phase: str) -> None:
        """Handle phase transition: save checkpoint, reset optimizer, update data."""
```

#### Pseudocode
```
train():
    phases = get_all_phases()
    start_idx = index of start_phase in phases
    for phase in phases[start_idx:]:
        train_phase(phase)

train_phase(phase):
    # Setup
    apply freeze_mask(model, phase)
    setup optimizer (AdamW, phase.lr, phase.weight_decay)
    setup scheduler (CosineAnnealingLR, phase.max_epochs)
    setup dataloader (phase-appropriate Dataset, phase.batch_size)

    best_val_loss = inf
    epochs_no_improve = 0

    for epoch in range(phase.max_epochs):
        train_loss = train_epoch(phase, dataloader, optimizer, scheduler)
        val_metrics = validate(phase, val_dataloader)

        # Logging (delegate to monitor)
        monitor.log_epoch(phase, epoch, train_loss, val_metrics)

        # Best model tracking
        if val_metrics[phase.transition_metric] < best_val_loss:
            best_val_loss = val_metrics[phase.transition_metric]
            epochs_no_improve = 0
            checkpoint.save_best(phase.name)
        else:
            epochs_no_improve += 1

        # Transition check
        if should_transition(phase, val_metrics, epochs_no_improve):
            transition_phase(phase.name, next_phase.name)
            break

        # Control file check (pause/resume/stop)
        monitor.check_control()
```

#### Test Spec
- `test_training.py::test_trainer_freeze_unfreeze` — Setup: create trainer, apply Phase A1 mask. Expected: only encoder+decoder params have requires_grad=True.
- `test_training.py::test_trainer_phase_transition` — Setup: mock convergence. Expected: optimizer reset, scheduler reset, freeze mask updated.
- `test_training.py::test_trainer_A1_one_epoch` — Setup: fake cached data, run 1 epoch of A1. Expected: loss decreases, encoder+decoder params updated.
- `test_training.py::test_trainer_B1_rollout_one_epoch` — Setup: fake segment data, run 1 epoch of B1 with rollout=1. Expected: Stage 1 dynamics params updated, Stage 2 frozen.

#### Checklist
- [ ] Implement SurrogateTrainer class
- [ ] Phase A1: train encoder + ionic_state_decoder, MSE reconstruction
- [ ] Phase A2: train voltage_attention (conc dims), MSE concentration tracking
- [ ] Phase A3: train gate_conductance_mlp/linear/logit/decoder, MSE conductance products
- [ ] Phase B1-B5: train Stage 1 dynamics, rollout with scheduled sampling
- [ ] Phase C: train all Stage 1 with concentration focus
- [ ] Phase D: train Stage 2 only (frozen Stage 1), MSE I_ion
- [ ] Phase E: train everything, MSE I_ion
- [ ] Optimizer + scheduler reset at each phase transition
- [ ] Encoder created at A1 start, used through B, discarded at C start
- [ ] DataLoader swaps at phase transitions (new tiers, new dataset type)
- [ ] Write tests

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate
conda run -n heart-conduction python -m pytest tests/test_training.py -v -k "trainer"
```

#### Exit Criteria
- [ ] Trainer can execute A1 for 1 epoch with fake data
- [ ] Freeze/unfreeze correct for every phase
- [ ] Phase transition logic works
- [ ] All trainer tests pass

#### Risk
Complex state management across phases — mitigation: PhaseConfig is declarative, trainer is just a loop that applies configs. Keep trainer thin.

### Phase 2 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate
conda run -n heart-conduction python -m pytest tests/test_training.py -v -k "encoder or phase or rollout or trainer"
conda run -n heart-conduction python -m pytest tests/ -v  # ALL tests, no regressions (83+)
```

### Phase 2 Exit Criteria
- [ ] All new tests pass
- [ ] All existing 83 tests pass (no regressions)
- [ ] Trainer can run A1 for 1+ epochs on fake data
- [ ] Rollout engine handles all phase types
- [ ] Freeze masks verified for all 11 phases

### Phase 2 Cleanup
- float64 consistency — all training tensors float64 (model and data)
- V5.3 not modified
- No code duplication — rollout reuses model.forward(), not reimplemented

**-> Commit point: git commit after Phase 2 passes**

---

## Phase 3: Monitoring, Checkpointing, and Metrics

**Goal**: Build the monitoring infrastructure: JSONL logging, TensorBoard, checkpoint save/load/resume, divergence detection, pause/resume control, and per-phase validation metrics.
**Tier**: medium
**Estimated scope**: 3 files, ~500 lines, 8 tests

### Phase Context
- TRAINING_MONITOR.md is the authoritative spec for all monitoring behavior
- Checkpoint format defined in TRAINING_MONITOR.md:127-171
- Control file: `Surrogate/training_control.json`
- Run directory: `Surrogate/runs/{run_name}/` with training_log.jsonl, phase_summary.json, checkpoints/, tensorboard/
- Divergence detection: NaN→auto-pause, plateau→suggest transition, spike→log warning
- NaN recovery: rollback to latest.pt, if persists rollback to best_{phase}.pt + halve LR

### Step 3.1: Checkpoint Manager
**Model**: opus

#### Read First
- `Surrogate/TRAINING_MONITOR.md:125-171` — Checkpoint contents spec
- `Surrogate/surrogate/training/encoder.py` — Encoder state dict (temporary, saved in A-B only)
- `Surrogate/surrogate/training/phases.py` — PhaseConfig for state reconstruction

#### Why
Training may be interrupted (GPU crash, user pause, context compaction mid-session). Checkpoint save/load must capture ALL state needed to resume exactly: model weights, optimizer, scheduler, phase, epoch, encoder, RNG state.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/training/checkpoint.py`

**Interfaces / Signatures:**
```python
class CheckpointManager:
    def __init__(self, run_dir: str):

    def save(self, tag: str, model, optimizer, scheduler, encoder, phase, epoch, step, best_val_loss, config, extra: dict = None) -> Path:
        """Save checkpoint. tag: 'best_A1', 'latest', 'pause_checkpoint', etc."""

    def load(self, tag: str, model, optimizer=None, scheduler=None, encoder=None) -> dict:
        """Load checkpoint. Returns metadata dict. Optionally loads optimizer/scheduler/encoder."""

    def get_best(self, phase: str) -> Path | None:
        """Get path to best checkpoint for a phase, or None."""

    def list_checkpoints(self) -> list[str]:
        """List all checkpoint tags."""
```

#### Pseudocode
```
save(tag, ...):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'encoder_state_dict': encoder.state_dict() if encoder else None,
        'phase': phase, 'epoch': epoch, 'step': step,
        'best_val_loss': best_val_loss,
        'config': config,
        'rng_state': { torch, cuda, numpy, python states },
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, run_dir / 'checkpoints' / f'{tag}.pt')

load(tag, model, ...):
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    # ... restore RNG state
    return {k: v for k, v in ckpt.items() if k not in state_dicts}
```

#### Test Spec
- `test_training.py::test_checkpoint_save_load` — Expected: model weights identical after save/load
- `test_training.py::test_checkpoint_resume_training` — Expected: optimizer state restored, training continues from same point

#### Checklist
- [ ] Implement CheckpointManager
- [ ] Save/load all state: model, optimizer, scheduler, encoder, RNG
- [ ] Handle encoder=None (phases C-E)
- [ ] Overwrite latest.pt every epoch
- [ ] Write tests

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate
conda run -n heart-conduction python -m pytest tests/test_training.py -v -k "checkpoint"
```

#### Exit Criteria
- [ ] Round-trip save/load preserves all state
- [ ] Resume produces same training trajectory

#### Risk
torch.save with pickle can be fragile across PyTorch versions — mitigation: save config as plain dict, not model class references.

### Step 3.2: Training Monitor
**Model**: opus

#### Read First
- `Surrogate/TRAINING_MONITOR.md` — Full document: control file, log format, divergence detection
- `Surrogate/surrogate/training/checkpoint.py` — CheckpointManager for NaN recovery

#### Why
The monitor handles three concerns: (1) JSONL + TensorBoard logging for agent readability, (2) control file polling for pause/resume, (3) automatic divergence detection. These are separate from the training loop logic.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/training/monitor.py`

**Interfaces / Signatures:**
```python
class TrainingMonitor:
    def __init__(self, run_dir: str, control_path: str | None = None):

    def log_batch(self, phase, epoch, batch, step, loss, lr, grad_norm, rollout, sched_p, wall_s) -> None:
        """Append one JSONL line."""

    def log_epoch(self, phase, epoch, train_loss, val_metrics) -> None:
        """Update phase_summary.json, TensorBoard scalars."""

    def log_phase_transition(self, from_phase, to_phase, metrics) -> None:
        """Log transition event."""

    def check_control(self) -> str:
        """Poll training_control.json. Returns 'running', 'paused', or raises TrainingStoppedError."""

    def update_control(self, **kwargs) -> None:
        """Update fields in training_control.json."""

    def check_divergence(self, loss, grad_norm) -> str | None:
        """Returns 'nan', 'spike', 'plateau', or None."""

class TrainingStoppedError(Exception): ...
```

#### Test Spec
- `test_training.py::test_monitor_jsonl_format` — Expected: JSONL line has all required fields
- `test_training.py::test_monitor_control_pause` — Expected: pause_requested → saves checkpoint, blocks
- `test_training.py::test_monitor_nan_detection` — Expected: NaN loss triggers 'nan' response
- `test_training.py::test_monitor_plateau_detection` — Expected: 20 epochs no improvement triggers 'plateau'

#### Checklist
- [ ] JSONL logging (one line per batch)
- [ ] phase_summary.json (updated per epoch)
- [ ] TensorBoard writer (loss/train, loss/val, lr, grad_norm, metrics/*)
- [ ] Control file polling (every 50 batches)
- [ ] Divergence detection: NaN, spike (3x running avg), plateau (configurable patience)
- [ ] EMA running average for loss baseline (alpha=0.01)
- [ ] Write tests

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate
conda run -n heart-conduction python -m pytest tests/test_training.py -v -k "monitor"
```

#### Exit Criteria
- [ ] JSONL log is machine-readable and has all fields from TRAINING_MONITOR.md
- [ ] Control file pause/resume works
- [ ] Divergence detection fires correctly

#### Risk
TensorBoard import may fail if not installed — mitigation: optional import, degrade gracefully to JSONL-only.

### Step 3.3: Validation Metrics
**Model**: sonnet

#### Read First
- `Surrogate/TRAINING_STRATEGY.md:206-218` — Per-phase metrics, APD, dVm/dt_max
- `Surrogate/surrogate/model/ionic_surrogate_v3.py:118-193` — Model output dict

#### Why
Beyond loss, we need physiologically meaningful metrics: APD90 error (action potential duration) and dVm/dt_max error (max upstroke velocity). These are computed on full-beat rollouts during validation.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/training/metrics.py`

**Interfaces / Signatures:**
```python
def compute_apd90(Vm_trace: Tensor, dt: Tensor | float = 0.01) -> Tensor:
    """APD90 from a Vm trace. dt can be scalar or per-step tensor. Returns APD in ms. NaN if no AP detected."""

def compute_dvdt_max(Vm_trace: Tensor, dt: Tensor | float = 0.01) -> Tensor:
    """Max upstroke velocity dVm/dt. dt can be scalar or per-step tensor. Returns mV/ms."""

def compute_phase_metrics(phase_name: str, predictions: dict, targets: dict) -> dict:
    """Compute all relevant metrics for a phase. Returns {metric_name: value}."""
```

#### Pseudocode
```
compute_apd90(Vm_trace, dt):
    # Find threshold crossings at -40 mV (upstroke)
    # Find Vm_max, compute 90% repolarization level
    # Find crossing of 90% level on downstroke
    # APD90 = (t_repol - t_upstroke) * dt
    # Return NaN if no clean AP detected

compute_dvdt_max(Vm_trace, dt):
    dVdt = diff(Vm_trace) / dt
    return dVdt.max()
```

#### Test Spec
- `test_training.py::test_apd90_known_trace` — Setup: synthetic AP trace. Expected: APD within 1ms of expected.
- `test_training.py::test_dvdt_max_known_trace` — Setup: synthetic upstroke. Expected: correct max slope.

#### Checklist
- [ ] Implement APD90 computation
- [ ] Implement dVm/dt_max computation
- [ ] Implement phase_metrics dispatcher
- [ ] Handle edge cases: no AP detected, flat trace
- [ ] Write tests

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate
conda run -n heart-conduction python -m pytest tests/test_training.py -v -k "metrics"
```

#### Exit Criteria
- [ ] APD90 correct on synthetic traces
- [ ] dVm/dt_max correct
- [ ] Graceful handling of degenerate traces

#### Risk
Low — these are simple 1D signal processing operations.

### Phase 3 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate
conda run -n heart-conduction python -m pytest tests/test_training.py -v -k "checkpoint or monitor or metrics"
conda run -n heart-conduction python -m pytest tests/ -v  # ALL tests, no regressions (83+)
```

### Phase 3 Exit Criteria
- [ ] All new tests pass
- [ ] All existing 83 tests pass (no regressions)
- [ ] Checkpoint round-trip preserves all state
- [ ] Monitor produces correct JSONL and detects divergence
- [ ] APD90 and dVm/dt_max metrics work

### Phase 3 Cleanup
- float64 consistency — metric computations in float64
- V5.3 not modified

**-> Commit point: git commit after Phase 3 passes**

---

## Phase 4: CLI Entry Point + Integration Test

**Goal**: Wire everything together with a CLI entry point and run an integration test that executes Phase A1 on real T1 data.
**Tier**: medium
**Estimated scope**: 2 files, ~200 lines, 2 integration tests

### Phase Context
- All training components exist from Phases 1-3
- Real T1 data on HDD: `/media/HDD/surrogate_data/raw/tier01.h5` (5.5 GB, 9 protocols, 2M steps each)
- SSD has 47 GB free for cache
- GPU has 33.7 GB VRAM — model + T1 preprocessed (~1.8 GB float32) fits easily

### Step 4.1: CLI Entry Point
**Model**: opus

#### Read First
- `Surrogate/surrogate/training/trainer.py` — SurrogateTrainer interface
- `Surrogate/surrogate/training/data_cache.py` — CacheBuilder
- `Surrogate/TRAINING_MONITOR.md:56-74` — Run directory structure

#### Why
A single `python train.py` command should set up the cache (if not present), create the run directory, initialize the model, and start training. Configurable via command-line args for run name, start phase, device, cache dir.

#### Implementation Spec
**Files to create:** `Surrogate/train.py`

**Interfaces / Signatures:**
```python
def main():
    parser = argparse.ArgumentParser(description='Train IonicSurrogateV3')
    parser.add_argument('--run-name', default=None, help='Run name (auto-generated if not set)')
    parser.add_argument('--start-phase', default='A1', help='Phase to start from')
    parser.add_argument('--cache-dir', default='/tmp/surrogate_cache')
    parser.add_argument('--raw-dir', default='/media/HDD/surrogate_data/raw')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--dry-run', action='store_true', help='Create run dir and exit without training')
```

#### Pseudocode
```
main():
    args = parse_args()
    run_name = args.run_name or f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = f"Surrogate/runs/{run_name}"
    os.makedirs(run_dir, exist_ok=True)

    # Build cache if needed (T1-T3 + T12)
    cache = CacheBuilder(args.raw_dir, args.cache_dir)
    tiers = [1, 2, 3, 12]
    if not cache.is_cached(tiers=tiers):
        print("Building data cache from HDD...")
        cache.build_all(tiers=tiers)
        cache.compute_normalization_stats(tiers=[1, 2, 3])

    # Create model
    model = IonicSurrogateV3(scaffold=True).to(args.device)

    # Create trainer
    trainer = SurrogateTrainer(
        model=model,
        cache_dir=args.cache_dir,
        run_dir=run_dir,
        device=args.device,
        start_phase=args.start_phase,
    )

    # Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()
```

#### Test Spec
- `test_training.py::test_cli_help` — Expected: `python train.py --help` exits 0, prints usage.
- `test_training.py::test_cli_creates_run_dir` — Setup: run with `--dry-run` flag. Expected: run dir created, no training started.

#### Checklist
- [ ] CLI with argparse
- [ ] Auto-generate run name from timestamp
- [ ] Create run directory structure
- [ ] Build cache if missing (use `is_cached()`)
- [ ] Default tiers: [1, 2, 3, 12]
- [ ] Model creation with scaffold=True
- [ ] Resume from checkpoint support

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate
conda run -n heart-conduction python train.py --help
```

#### Exit Criteria
- [ ] `python train.py --help` works
- [ ] Dry-run with `--start-phase A1` initializes everything without error

#### Risk
HDD may not be mounted at training time — mitigation: clear error message, check mount before starting.

### Step 4.2: Integration Test — Phase A1 on Real Data
**Model**: opus

#### Read First
- All Phase 1-3 code
- `Surrogate/TRAINING_STRATEGY.md:26-43` — Phase A1 expected behavior

#### Why
The ultimate test: run Phase A1 on real T1 data for a few epochs and verify the autoencoder loss decreases. This catches integration issues between all components.

#### Implementation Spec
**Files to modify:** `Surrogate/tests/test_training.py` — add integration test class

#### Test Spec
- `test_training.py::TestIntegration::test_A1_real_data` — Setup: build cache from tier01.h5 (may take a few minutes from HDD). Run A1 for 3 epochs. Expected: val loss decreases epoch-over-epoch. Mark as `@pytest.mark.slow`.
- `test_training.py::TestIntegration::test_A1_checkpoint_resume` — Setup: run A1 for 2 epochs, save checkpoint, load, run 1 more epoch. Expected: loss trajectory consistent.

#### Checklist
- [ ] Integration test class with `@pytest.mark.slow` marker
- [ ] Test A1 converges on real data (loss decreasing)
- [ ] Test checkpoint resume produces consistent results
- [ ] Skip if HDD not mounted (graceful skip, not failure)

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate
conda run -n heart-conduction python -m pytest tests/test_training.py -v -k "integration" -m "slow"
```

#### Exit Criteria
- [ ] Phase A1 runs on real T1 data
- [ ] Loss decreases over 3 epochs
- [ ] Checkpoint resume works

#### Risk
First-run cache build reads T1 (5.5 GB) from HDD at ~7 MB/s = ~13 min with direct I/O. Subsequent runs use SSD cache. Mitigation: skip cache build if already cached (`is_cached()`).

### Phase 4 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate
conda run -n heart-conduction python -m pytest tests/test_training.py -v
conda run -n heart-conduction python -m pytest tests/ -v  # ALL tests, no regressions (83+)
```

### Phase 4 Exit Criteria
- [ ] All tests pass (unit + integration)
- [ ] All existing 83 tests pass (no regressions)
- [ ] `python train.py` runs A1 successfully
- [ ] Checkpoint save/load works end-to-end

### Phase 4 Cleanup
- float64 consistency
- V5.3 not modified
- Run directory structure matches TRAINING_MONITOR.md spec

**-> Commit point: git commit after Phase 4 passes**

---

## Phase 5: T4 Shard Streaming

**Goal**: Build shard-based streaming loader for T4 (551 GB, cannot fit in RAM/SSD). Required for training Phases D and E.
**Tier**: medium
**Estimated scope**: 1 file, ~250 lines, 4 tests

### Phase Context
- T4: `/media/HDD/surrogate_data/raw/tier04.h5` — 551 GB, 200 protocols, float64
- HDD speed: ~7 MB/s (direct I/O) to ~246 MB/s (buffered). A 200 MB shard loads in 1-30s depending on access pattern.
- Strategy from HARDWARE_CONSTRAINTS.md: pre-convert T4 to .pt shards (~200 MB each) on HDD, load one shard at a time, train until exhausted, swap next shard. Double-buffering: prefetch shard N+1 in background while training on shard N.
- T4 enters at Phase D (TRAINING_STRATEGY.md line 196). Not needed for A-C.
- Shards are preprocessed via V3Preprocessor (same as cached tiers), stored as float32 .pt on HDD.

### Step 5.1: Shard Converter
**Model**: opus

#### Read First
- `Surrogate/HARDWARE_CONSTRAINTS.md:57-76` — Shard streaming strategy, prefetch design
- `Surrogate/surrogate/data/storage.py` — TraceStorage, ShardProcessor (existing but needs update)
- `Surrogate/surrogate/data/preprocessor.py` — V3Preprocessor
- `Surrogate/DATA_PIPELINE.md:68-80` — Shard storage format

#### Why
T4 protocols are long (up to 16M timesteps). Loading entire protocols into RAM is feasible (~600 MB each at float64) but loading ALL 200 protocols at once is not (551 GB total). Pre-converting to preprocessed .pt shards (~200 MB each, float32) enables sequential streaming.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/training/shard_loader.py`

**Interfaces / Signatures:**
```python
class ShardConverter:
    """Convert T4 HDF5 protocols to preprocessed .pt shards on HDD."""
    def __init__(self, raw_dir: str, shard_dir: str, shard_size_mb: float = 200.0):
    def convert_tier(self, tier: int = 4) -> int:
        """Convert tier to shards. Returns number of shards created."""

class ShardStreamLoader:
    """Stream preprocessed shards from HDD with double-buffering."""
    def __init__(self, shard_dir: str, segment_length: int, device: str = 'cuda'):
    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        """Yields batches of segments from current shard. Auto-swaps shards."""
    def prefetch_next(self) -> None:
        """Start loading next shard in background thread."""
```

#### Pseudocode
```
ShardConverter.convert_tier(tier):
    storage = TraceStorage(raw_dir)
    preprocessor = V3Preprocessor()
    protocols = storage.list_protocols(tier)

    current_shard = []
    current_size = 0
    shard_idx = 0

    for proto in protocols:
        trace = storage.load_trace(tier, proto)
        processed = preprocessor.process_segment(trace.data)
        # Convert to float32 dict of tensors
        processed_f32 = {k: v.float() for k, v in processed.items()}
        current_shard.append(processed_f32)
        current_size += sum(v.nbytes for v in processed_f32.values())

        if current_size >= shard_size_mb * 1e6:
            # Concatenate and save shard
            merged = {k: torch.cat([s[k] for s in current_shard], dim=0) for k in current_shard[0]}
            torch.save(merged, shard_dir / f'shard_{shard_idx:04d}.pt')
            shard_idx += 1
            current_shard = []
            current_size = 0

    # Save remaining
    if current_shard:
        merged = {k: torch.cat([s[k] for s in current_shard], dim=0) for k in current_shard[0]}
        torch.save(merged, shard_dir / f'shard_{shard_idx:04d}.pt')

ShardStreamLoader.__iter__():
    shards = sorted(glob(shard_dir / 'shard_*.pt'))
    shuffle(shards)  # random shard order each epoch

    # Start prefetching first shard
    prefetch_thread = Thread(target=load_shard, args=(shards[0],))
    prefetch_thread.start()

    for i, shard_path in enumerate(shards):
        # Wait for current shard
        prefetch_thread.join()
        shard_data = prefetch_result  # loaded in background

        # Start prefetching next shard
        if i + 1 < len(shards):
            prefetch_thread = Thread(target=load_shard, args=(shards[i+1],))
            prefetch_thread.start()

        # Extract segments from this shard
        dataset = SegmentDataset(shard_data, segment_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch in loader:
            yield {k: v.to(device).double() for k, v in batch.items()}  # float64 on GPU
```

#### Test Spec
- `test_training.py::test_shard_converter_creates_shards` — Setup: fake tier with 2 protocols. Expected: .pt shard files created.
- `test_training.py::test_shard_stream_loader_yields_batches` — Setup: create 2 small shards. Expected: loader yields batches with correct shapes.
- `test_training.py::test_shard_prefetch_overlap` — Setup: 3 shards, mock timing. Expected: shard N+1 loading starts before shard N training ends.
- `test_training.py::test_shard_loader_float64_output` — Setup: float32 shard on disk. Expected: yielded tensors are float64.

#### Checklist
- [ ] Implement ShardConverter (HDF5 → preprocessed .pt shards on HDD)
- [ ] Implement ShardStreamLoader with double-buffering
- [ ] Background thread for shard prefetch
- [ ] Shard shuffle per epoch (different shard order each time)
- [ ] Output float64 tensors (convert from float32 shards)
- [ ] Integrate with SurrogateTrainer: swap from CacheBuilder to ShardStreamLoader at Phase D
- [ ] Write tests

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate
conda run -n heart-conduction python -m pytest tests/test_training.py -v -k "shard"
```

#### Exit Criteria
- [ ] Shard converter creates valid .pt files
- [ ] Stream loader yields correct batches
- [ ] Prefetch hides HDD latency
- [ ] All shard tests pass

#### Risk
T4 shard conversion is slow (551 GB through HDD + V3Preprocessor) — mitigation: one-time cost, can run overnight. Estimated ~2 hours at 7 MB/s.

### Phase 5 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate
conda run -n heart-conduction python -m pytest tests/test_training.py -v -k "shard"
conda run -n heart-conduction python -m pytest tests/ -v  # ALL tests, no regressions
```

### Phase 5 Exit Criteria
- [ ] All new shard tests pass
- [ ] All existing tests pass (no regressions)
- [ ] ShardStreamLoader can yield batches from test shards

### Phase 5 Cleanup
- float64 consistency — shards store float32, loader outputs float64
- V5.3 not modified

**-> Commit point: git commit after Phase 5 passes**

---

## Phase 6: Training Agent

**Goal**: Create a project-based Claude agent that monitors training logs, diagnoses issues, and can intervene (pause, adjust LR, transition phase, rollback). Discarded after training succeeds.
**Tier**: small
**Estimated scope**: 1 agent definition file, 1 helper script, ~150 lines

### Phase Context
- Agent reads: training_control.json, training_log.jsonl, phase_summary.json
- Agent writes: training_control.json (intervention field)
- Training loop checks control file every 50 batches and applies interventions
- Agent is invoked manually: user asks Claude "check training" or runs the agent directly
- Agent definition lives in `.claude/agents/` — project-scoped, not committed long-term
- TRAINING_MONITOR.md:178-195 describes agent capabilities

### Step 6.1: Agent Definition
**Model**: sonnet

#### Read First
- `Surrogate/TRAINING_MONITOR.md:178-248` — Agent monitoring spec, divergence heuristics
- `.claude/agents/` — existing agent definitions (if any)

#### Why
A Claude agent with full LLM reasoning can diagnose subtle training issues that heuristic thresholds miss: unusual loss curve shapes, cross-phase metric comparisons, gradient norm trends suggesting architectural issues. The agent reads structured logs (JSONL) and writes structured interventions (JSON).

#### Implementation Spec
**Files to create:** `.claude/agents/training-monitor.md` — Agent definition with instructions

The agent definition should instruct the agent to:
1. Read `training_control.json` for current state
2. Read tail of `training_log.jsonl` (last 200 lines) for recent batch metrics
3. Read `phase_summary.json` for cross-phase comparison
4. Analyze: loss trends, gradient norms, convergence rate, val/train gap
5. Diagnose: plateau, divergence, overfitting, LR too high/low, phase ready to transition
6. Act: write intervention to training_control.json OR report to user

**Autonomous interventions** (agent can do without asking):
- Pause training (loss spike, NaN, gradient explosion)
- Reduce LR by 0.5x (plateau detected, grad norm stable)
- Transition to next phase (convergence criteria met)
- Increase rollout within phase (B sub-phase curriculum)
- Rollback to best checkpoint (divergence after change)

**Must-ask interventions** (agent reports to user, doesn't act):
- Skip a phase entirely
- Adjust batch size
- Abort training
- Architectural changes

#### Test Spec
- `test_training.py::test_agent_definition_exists` — Expected: `.claude/agents/training-monitor.md` exists and contains required sections (Analysis checklist, Intervention protocol, Output format).

#### Pseudocode
```markdown
# Training Monitor Agent

You monitor the IonicSurrogateV3 training pipeline.

## Files to read
- `Surrogate/runs/{latest_run}/training_control.json`
- `Surrogate/runs/{latest_run}/training_log.jsonl` (tail -200)
- `Surrogate/runs/{latest_run}/phase_summary.json`

## Analysis checklist
1. Current phase and epoch
2. Loss trend (last 50 batches): improving / plateaued / diverging
3. Gradient norm trend: stable / growing / exploding
4. Val vs train gap: underfitting / good / overfitting
5. Phase convergence: below threshold? / approaching? / stuck?
6. Cross-phase comparison: is current phase learning faster/slower than previous?

## Intervention protocol
[structured decision tree for each diagnosis]

## Output format
[structured report + optional intervention JSON]
```

#### Checklist
- [ ] Write agent definition in `.claude/agents/training-monitor.md`
- [ ] Include analysis checklist
- [ ] Include intervention decision tree
- [ ] Include example invocation and output format
- [ ] Document how training loop reads interventions from control file

#### Verify
```bash
# Verify agent definition exists and is readable
cat .claude/agents/training-monitor.md | head -20
```

#### Exit Criteria
- [ ] Agent definition file exists
- [ ] Instructions are clear enough for a cold-start Claude to follow
- [ ] Intervention JSON format matches what training loop expects

#### Risk
Agent may make poor decisions with limited context — mitigation: conservative by default (only pause, reduce LR). Destructive actions require user confirmation.

### Step 6.2: Intervention Handler in Training Loop
**Model**: sonnet

#### Read First
- `Surrogate/surrogate/training/monitor.py` — TrainingMonitor.check_control()
- `Surrogate/surrogate/training/trainer.py` — Where control checks happen

#### Why
The training loop needs to read and apply interventions that the agent writes to training_control.json. This extends the existing pause/resume mechanism with LR adjustment, phase transition, and checkpoint rollback.

#### Implementation Spec
**Files to modify:** `Surrogate/surrogate/training/monitor.py` — add intervention handling

#### Test Spec
- `test_training.py::test_intervention_reduce_lr` — Setup: write reduce_lr intervention to control file. Expected: LR halved after next check cycle.
- `test_training.py::test_intervention_cleared_after_apply` — Setup: apply intervention. Expected: intervention field cleared in control file.

**New method:**
```python
def apply_intervention(self, intervention: dict, trainer) -> str:
    """Apply an agent intervention. Returns description of action taken."""
    action = intervention.get('action')
    if action == 'reduce_lr':
        factor = intervention.get('factor', 0.5)
        # Reduce current LR by factor
    elif action == 'transition_phase':
        # Signal phase transition
    elif action == 'rollback':
        # Load best checkpoint for current phase
    elif action == 'pause':
        # Already handled by status field
    return f"Applied: {action}"
```

#### Checklist
- [ ] Parse intervention field from training_control.json
- [ ] Implement reduce_lr action
- [ ] Implement transition_phase action
- [ ] Implement rollback action
- [ ] Clear intervention field after applying
- [ ] Log all interventions to JSONL

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate
conda run -n heart-conduction python -m pytest tests/test_training.py -v -k "intervention"
```

#### Exit Criteria
- [ ] Intervention handler applies all action types
- [ ] Interventions logged to JSONL
- [ ] Control file cleared after intervention applied

#### Risk
Race condition if agent writes while training loop reads — mitigation: atomic JSON write (write to temp file, rename). Training loop reads are non-blocking.

### Phase 6 Verification
```bash
cat .claude/agents/training-monitor.md | wc -l  # agent definition exists
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate
conda run -n heart-conduction python -m pytest tests/test_training.py -v -k "intervention or agent"
conda run -n heart-conduction python -m pytest tests/ -v  # all tests
```

### Phase 6 Exit Criteria
- [ ] Agent definition file exists and is comprehensive
- [ ] Intervention handler works for all action types
- [ ] All tests pass (no regressions)

### Phase 6 Cleanup
- Agent file is in `.claude/agents/` (project-scoped, temporary)
- Add comment: "TEMPORARY — discard after training pipeline validated"

**-> Commit point: git commit after Phase 6 passes**

---

## Final Cleanup

1. Archive the completed plan:
```bash
mkdir -p Research/Active/surrogate_pipeline/plans
cp Research/Active/surrogate_pipeline/PLAN.md "Research/Active/surrogate_pipeline/plans/$(date +%Y-%m-%d)_training-pipeline.md"
```

2. Revert the bottom tmux pane from PLAN.md back to WHITEBOARD.md:
```bash
tmux send-keys -t 2 C-c
sleep 0.3
tmux send-keys -t 2 'W=$(tput cols); H=""; while true; do N=$(md5sum Research/Active/surrogate_pipeline/WHITEBOARD.md 2>/dev/null | cut -d" " -f1); if [ "$N" != "$H" ]; then clear; glow -s .glow-style.json -w $W Research/Active/surrogate_pipeline/WHITEBOARD.md 2>/dev/null; H=$N; fi; sleep 1; done' Enter
```

3. Cross-phase cleanup:
- [ ] float64 consistency — no float32 leaks in training code (cache stores float32, everything else float64)
- [ ] V5.3 not modified
- [ ] No code duplication — reuse V3Preprocessor, TraceStorage, NernstComputer
- [ ] All tests pass: `conda run -n heart-conduction python -m pytest Surrogate/tests/ -v`
- [ ] Run directory structure matches TRAINING_MONITOR.md
- [ ] Agent definition clearly marked as temporary

## Mutation Log

(Initially empty — populated during execution)

