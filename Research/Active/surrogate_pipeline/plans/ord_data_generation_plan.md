# PLAN: ORd Data Generation for Surrogate Training

Created: 2026-03-30
Engine(s): Bidomain V1 (ORd ionic model)
Research question: [surrogate_pipeline](../README.md)
Source: Audit of existing TTP06 pipeline + ORd model code in `Bidomain/Engine_V1/cardiac_sim/ionic/ord/`

## Objective
Extend the surrogate data generation pipeline to produce ORd (O'Hara-Rudy 2011) single-cell traces. ORd uses its own 101-column format with all 40 state variables logged. Generate T1-T13 data for all 3 celltypes. CaMKII buildup requires 100-beat warmup for all protocols plus a dedicated T13 recording the full buildup. Format unification with TTP06 is deferred to the training-time dataloader.

## Success Criteria
- [ ] `ORdTraceData` with 101-col format, column constants verified by test
- [ ] ORd `SingleCellGenerator` produces correct traces (I_ion matches, AP shape valid)
- [ ] ORd `BatchGenerator` runs on GPU (throughput within 3x of TTP06 -- ORd is 2.8x heavier)
- [ ] `corrupt_states()` handles both TTP06 and ORd gate index ranges
- [ ] ORd-aware `TraceStorage` and `ShardProcessor` for 101-col HDF5 files
- [ ] T1-T13 generated for ENDO, EPI, M_CELL at dt=0.01 and dt=0.005
- [ ] All 32 existing TTP06 data generation tests still pass (zero regressions)
- [ ] No modifications to any existing TTP06 code paths

## Hyperparameters

```python
# === ORd MODEL ===
N_STATES_ORD     = 40      # Total state variables (excludes V)
N_GATES_RL_ORD   = 28      # Rush-Larsen gates (nca excluded -- Forward Euler)
V_REST_ORD       = -87.5   # mV

# === ORd COLUMN FORMAT (101 columns) ===
#   0:     Vm (mV)
#   1:     I_stim (sign-flipped: positive = depolarizing)
#   2:     dt (ms)
#   3-42:  40 ionic states (StateIndex order)
#   43:    I_ion (pure ionic current, no stimulus)
#   44:    clamp_mask (0.0 = free-running, 1.0 = voltage clamped)
#   45-72: 28 gate_inf values (gate_indices order)
#   73-100: 28 gate_tau values (gate_indices order, ms)
N_COLUMNS_ORD    = 101     # 3 + 40 + 2 + 28 + 28

# === ORd STATE INDICES (from parameters.py StateIndex) ===
# Concentrations: nai(0), ki(1), cai(2), cansr(3), nass(4), kss(5), cass(6), cajsr(7)
# RL gates:       m(8), hf(9), hs(10), j(11), hsp(12), jp(13),
#                 mL(14), hL(15), hLp(16),
#                 a(17), iF(18), iS(19), ap(20), iFp(21), iSp(22),
#                 d(23), ff(24), fs(25), fcaf(26), fcas(27), jca(28), ffp(30), fcafp(31),
#                 xrf(32), xrs(33), xs1(34), xs2(35), xk1(36)
# FE gate:        nca(29)  -- Forward Euler, NOT Rush-Larsen
# SR release:     Jrelnp(37), Jrelp(38)
# CaMKII:         CaMKt(39)
ORD_GATE_INDICES = [8,9,10,11,12,13, 14,15,16, 17,18,19,20,21,22,
                    23,24,25,26,27,28, 30,31, 32,33, 34,35, 36]  # 28 total

# === CaMKII WARMUP ===
WARMUP_BEATS     = 100     # Standard for all ORd protocols
WARMUP_BCL       = 1000.0  # ms (BCL=1000 is ORd standard)
WARMUP_STIM_AMP  = -80.0   # uA/uF (TTP06-convention negative)
WARMUP_STIM_DUR  = 0.5     # ms

# === CONCENTRATION PERTURBATION (ORd) ===
# ORd uses params_override (lowercase): ko, nao, cao
# NOT TTP06 convention: Ko, Nai_init, Cai_scale
ORD_CONC_PERTURB_KEYS = ['ko', 'nao', 'cao']

# === STORAGE ===
ORD_RAW_DIR      = '/media/HDD/surrogate_data/raw_ord'
ORD_SHARD_DIR    = '/media/HDD/surrogate_data/train_ord'

# === T13: CaMKII BUILDUP PROTOCOL ===
T13_N_BEATS      = 1000    # Record full 1000-beat buildup (CaMKt from 0 to steady-state)
T13_BCL          = 1000.0  # ms
T13_DT           = 0.01    # ms
```

## Known Failures (from audit)

1. **`corrupt_states()` hardcodes TTP06 gates**: `range(5, 17)` is TTP06. ORd gates are at `[8:37]` but that range includes nca(29) which is FE. Must use `ORD_GATE_INDICES` list. Also `extreme_ca` uses index 2 which happens to be `cai` in BOTH models -- but document this explicitly.

2. **`TraceData.N_COLUMNS = 47`**: Hardcoded in `single_cell_generator.py`. `ShardProcessor` uses it for shard size estimation. ORd needs 101 columns. Create `ORdTraceData` with its own column constants. Do NOT modify `TraceData`.

3. **`TraceStorage.load_trace()` returns `TraceData`**: Returns TTP06 `TraceData` object. ORd storage must return `ORdTraceData`. Separate class or parameterized approach needed.

4. **`ConcentrationPerturbation` is TTP06-specific**: Uses `Ko`, `Nai_init`, `Cai_scale` fields. ORd uses `params_override` with lowercase keys (`ko`, `nao`, `cao`). Need `ORdConcentrationPerturbation` wrapper.

5. **`CorruptionRecovery` delegates to `corrupt_states()`**: If `corrupt_states()` is fixed to be model-aware, `CorruptionRecovery` itself needs no change -- just pass a model identifier down.

6. **ORd `model.step()` uses `Istim` kwarg, not `I_stim`**: TTP06 uses `I_stim`. The ORd generator must use `Istim=`.

7. **ORd has no `from_config()` for conductance scaling**: TTP06 has `CellTypeConfig` + `from_config()`. ORd uses `params_override` dict on `__init__`. Generator must translate conductance scaling accordingly.

---

## Phase 1: ORd Data Infrastructure

**Goal**: Create ORdTraceData (101-col format), ORd-aware storage/sharding, and ORd-aware corrupt_states.
**Tier**: medium
**Estimated scope**: 3 files created, 1 file modified, ~300 lines, ~15 tests

### Phase Context
Working directory: `Surrogate/`. Tests: `cd Surrogate && conda run -n heart-conduction python -m pytest tests/ -v`. The ORd ionic model lives in `Bidomain/Engine_V1/cardiac_sim/ionic/ord/`. ORd has 40 state variables, 28 Rush-Larsen gates (nca is FE), and CaMKII dynamics. All new files are ORd-specific -- never modify existing TTP06 files.

### Step 1.1: ORdTraceData class
**Model**: sonnet

#### Read First
- `Surrogate/surrogate/data/single_cell_generator.py:32-63` -- TTP06 `TraceData` class and column constants
- `Bidomain/Engine_V1/cardiac_sim/ionic/ord/parameters.py:16-98` -- `StateIndex`, `STATE_NAMES` (40 states)
- `Bidomain/Engine_V1/cardiac_sim/ionic/ord/model.py:135-156` -- `gate_indices` property (28 RL gates)

#### Why
ORd uses a different column layout (101 vs 47). Column constants must be defined as a dataclass with named indices, mirroring `TraceData` but with ORd-specific values. This is the data contract that all downstream code references.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/data/ord_trace.py`
**Interfaces:**
```python
@dataclass
class ORdTraceData:
    """Container for recorded ORd single-cell trace data.

    Attributes:
        data: (T, 101) float64 tensor with columns:
            0: Vm (mV)
            1: I_stim (sign-flipped: positive = depolarizing)
            2: dt (ms)
            3-42: 40 ionic states (StateIndex order)
            43: I_ion
            44: clamp_mask
            45-72: 28 gate_inf (gate_indices order)
            73-100: 28 gate_tau (gate_indices order, ms)
        metadata: dict
    """
    data: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Column indices
    VM = 0
    I_STIM = 1
    DT = 2
    STATES_START = 3
    STATES_END = 43       # exclusive: 40 states at indices 3..42
    I_ION = 43
    CLAMP_MASK = 44
    GATE_INF_START = 45
    GATE_INF_END = 73     # exclusive: 28 gate_inf values
    GATE_TAU_START = 73
    GATE_TAU_END = 101    # exclusive: 28 gate_tau values
    N_COLUMNS = 101
    N_STATES = 40
    N_GATES_RL = 28
```

#### Pseudocode
```
# Dataclass with column constants as class attributes
# Mirror TraceData pattern exactly, just different numbers
# Include GATE_INDICES list as class-level constant for reference
GATE_INDICES = [8,9,10,11,12,13, 14,15,16, 17,18,19,20,21,22,
                23,24,25,26,27,28, 30,31, 32,33, 34,35, 36]
```

#### Test Spec
- `test_ord_trace_column_count` -- `ORdTraceData.N_COLUMNS == 101`
- `test_ord_trace_column_arithmetic` -- `STATES_END - STATES_START == 40`, `GATE_INF_END - GATE_INF_START == 28`, `GATE_TAU_END - GATE_TAU_START == 28`, total = 3 + 40 + 2 + 28 + 28 = 101
- `test_ord_trace_no_overlap` -- column ranges don't overlap: `STATES_END <= I_ION < CLAMP_MASK < GATE_INF_START`
- `test_ord_trace_gate_indices_count` -- `len(GATE_INDICES) == 28`
- `test_ord_trace_gate_indices_no_nca` -- `29 not in GATE_INDICES` (nca excluded)

#### Checklist
- [ ] 101 columns verified: 3 + 40 + 2 + 28 + 28
- [ ] Column indices are contiguous with no gaps
- [ ] nca (index 29) NOT in GATE_INDICES
- [ ] Column names match ORd StateIndex ordering
- [ ] Dataclass, not regular class (matches TraceData pattern)

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/test_ord_data_generation.py::TestORdTraceData -v
```

#### Exit Criteria
- [ ] 5 tests pass
- [ ] Column arithmetic verified

#### Risk
Off-by-one in column ranges. Mitigation: arithmetic test `3 + 40 + 2 + 28 + 28 == 101` is first test written.

---

### Step 1.2: ORd-aware corrupt_states
**Model**: sonnet

#### Read First
- `Surrogate/surrogate/data/augmentation.py:26-46` -- current `corrupt_states()` with hardcoded TTP06 gate indices
- `Bidomain/Engine_V1/cardiac_sim/ionic/ord/parameters.py:16-98` -- ORd StateIndex

#### Why
`corrupt_states()` hardcodes `range(5, 17)` for TTP06 gate indices. ORd gates span indices 8-36 (with gap at nca=29). Must parameterize by model type. Approach: add `model_type` parameter defaulting to `'ttp06'` for backward compatibility. No breaking changes to existing callers.

#### Implementation Spec
**Files to modify:** `Surrogate/surrogate/data/augmentation.py`
**Changes:**
```python
# Add model-specific gate index maps at module level
_GATE_INDICES = {
    'ttp06': list(range(5, 17)),  # 12 gates: m,h,j,r,s,d,f,f2,fCass,Xr1,Xr2,Xs
    'ord': [8,9,10,11,12,13, 14,15,16, 17,18,19,20,21,22,
            23,24,25,26,27,28, 30,31, 32,33, 34,35, 36],  # 28 RL gates (no nca)
}

_EXTREME_CA_INDEX = {
    'ttp06': 2,  # Cai at TTP06 StateIndex position 2
    'ord': 2,    # cai at ORd StateIndex position 2 (same position, document explicitly)
}

def corrupt_states(states, corruption_type, severity=0.5, seed=0,
                   model_type='ttp06'):
    # Use _GATE_INDICES[model_type] instead of hardcoded range(5, 17)
    # Use _EXTREME_CA_INDEX[model_type] for extreme_ca
```

#### Pseudocode
```
def corrupt_states(states, corruption_type, severity=0.5, seed=0,
                   model_type='ttp06'):
    gate_indices = _GATE_INDICES[model_type]
    rng = np.random.RandomState(seed)
    states = states.clone()
    if corruption_type == 'random_gates':
        for idx in gate_indices:
            if states.dim() > 1:
                states[:, idx] = rand(1) * severity + states[:, idx] * (1 - severity)
            else:
                states[idx] = float(rand(1) * severity + states[idx] * (1 - severity))
    elif corruption_type == 'extreme_ca':
        ca_idx = _EXTREME_CA_INDEX[model_type]
        if states.dim() > 1:
            states[:, ca_idx] *= (1 + 10 * severity)
        else:
            states[ca_idx] *= (1 + 10 * severity)
    return states
```

#### Test Spec
- `test_corrupt_ttp06_unchanged` -- calling with `model_type='ttp06'` produces same result as before (regression test)
- `test_corrupt_ord_random_gates` -- ORd: gates at indices 8-36 (excluding 29) are modified, concentrations (0-7) are NOT modified
- `test_corrupt_ord_extreme_ca` -- ORd: cai at index 2 is scaled up
- `test_corrupt_default_is_ttp06` -- omitting `model_type` defaults to TTP06 behavior (backward compat)
- `test_corrupt_unknown_model_raises` -- `model_type='unknown'` raises KeyError

#### Checklist
- [ ] Default `model_type='ttp06'` -- all existing callers unaffected
- [ ] ORd gate list has exactly 28 entries
- [ ] nca (index 29) excluded from ORd gate list
- [ ] `extreme_ca` index documented for both models
- [ ] No changes to `StitchedProtocol` class

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/ -v -k "corrupt"
```

#### Exit Criteria
- [ ] 5 tests pass
- [ ] 32 existing TTP06 tests still pass (regression check)

#### Risk
Accidentally breaking TTP06 callers. Mitigation: default argument preserves existing behavior; explicit regression test.

---

### Step 1.3: ORd TraceStorage and ShardProcessor
**Model**: opus

#### Read First
- `Surrogate/surrogate/data/storage.py` -- full file: `TraceStorage` (HDF5 I/O) and `ShardProcessor` (segmentation)
- `Surrogate/surrogate/data/ord_trace.py` -- `ORdTraceData` from Step 1.1

#### Why
`TraceStorage.load_trace()` returns `TraceData` objects (TTP06). `ShardProcessor` hardcodes `TraceData.N_COLUMNS` for shard size estimation. ORd data needs separate HDF5 files (different base_dir) and must use `ORdTraceData`. Create `ORdTraceStorage` and `ORdShardProcessor` as thin subclasses -- they share the same logic, just different data class and column count.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/data/ord_storage.py`
**Interfaces:**
```python
class ORdTraceStorage(TraceStorage):
    """HDF5 storage for ORd traces. Separate directory from TTP06."""

    def __init__(self, base_dir: str = ORD_RAW_DIR):
        super().__init__(base_dir=base_dir)

    def save_trace(self, trace: ORdTraceData, tier: int, protocol_name: str):
        # Identical to parent, but accepts ORdTraceData

    def load_trace(self, tier: int, protocol_name: str) -> ORdTraceData:
        # Returns ORdTraceData instead of TraceData


class ORdShardProcessor(ShardProcessor):
    """Shard processor for ORd 101-col data."""

    def __init__(self, raw_dir=ORD_RAW_DIR, shard_dir=ORD_SHARD_DIR, ...):
        # Override: use ORdTraceData.N_COLUMNS for shard size calc

    def process_tier(self, tier: int) -> List[torch.Tensor]:
        # Uses ORdTraceStorage instead of TraceStorage
```

#### Pseudocode
```
class ORdTraceStorage(TraceStorage):
    def __init__(self, base_dir=ORD_RAW_DIR):
        super().__init__(base_dir=base_dir)

    def load_trace(self, tier, protocol_name):
        path = self.base_dir / f'tier{tier:02d}.h5'
        with h5py.File(path, 'r') as f:
            data = torch.tensor(f[protocol_name]['data'][:], dtype=torch.float64)
            metadata = dict(f[protocol_name].attrs)
        return ORdTraceData(data=data, metadata=metadata)

class ORdShardProcessor(ShardProcessor):
    def __init__(self, raw_dir=ORD_RAW_DIR, shard_dir=ORD_SHARD_DIR,
                 segment_length=1000, shard_size_mb=200.0):
        self.raw_dir = Path(raw_dir)
        self.shard_dir = Path(shard_dir)
        self.segment_length = segment_length
        self.shard_size_mb = shard_size_mb
        bytes_per_segment = segment_length * ORdTraceData.N_COLUMNS * 4
        self._segments_per_shard = max(1, int(shard_size_mb * 1e6 / bytes_per_segment))

    def process_tier(self, tier):
        storage = ORdTraceStorage(str(self.raw_dir))
        protocols = storage.list_protocols(tier)
        all_segments = []
        for proto_name in protocols:
            trace = storage.load_trace(tier, proto_name)
            segments = self._extract_segments(trace)
            all_segments.extend(segments)
        return all_segments

    def process_all(self):
        """Override parent: use ORdTraceStorage instead of TTP06 TraceStorage.
        Parent hardcodes TraceStorage -- we must use ORdTraceStorage."""
        all_segments = []
        for tier_dir in sorted(self.raw_dir.iterdir()):
            if tier_dir.is_dir() and tier_dir.name.startswith('tier'):
                tier = int(tier_dir.name.replace('tier', ''))
                segments = self.process_tier(tier)
                all_segments.extend(segments)
                if len(all_segments) >= self._segments_per_shard:
                    self._write_shards(all_segments[:self._segments_per_shard])
                    all_segments = all_segments[self._segments_per_shard:]
        if all_segments:
            self._write_shards(all_segments)
```

#### Test Spec
- `test_ord_storage_save_load_roundtrip` -- save ORdTraceData, load back, verify data and metadata match
- `test_ord_storage_returns_ord_trace` -- `load_trace()` returns `ORdTraceData`, not `TraceData`
- `test_ord_storage_101_columns` -- saved/loaded data has 101 columns
- `test_ord_shard_segment_shape` -- segments have shape (segment_length, 101)
- `test_ord_shard_float32` -- shards are float32

#### Checklist
- [ ] Separate base_dir from TTP06 (different HDF5 files)
- [ ] `load_trace` returns `ORdTraceData`
- [ ] Shard size estimation uses 101 columns, not 47
- [ ] `_extract_segments` inherited from parent (no override needed)
- [ ] `_write_shards` inherited from parent (generic tensor stacking)

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/test_ord_data_generation.py::TestORdStorage -v
```

#### Exit Criteria
- [ ] 5 tests pass
- [ ] HDF5 roundtrip preserves all 101 columns

#### Risk
Parent class `__init__` checks for external HDD mount. Mitigation: tests use `tmp_path`, not real mount point.

---

### Phase 1 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/ -v
```

### Phase 1 Exit Criteria
- [ ] ~15 new ORd infrastructure tests pass
- [ ] 32 existing TTP06 tests pass (zero regressions)
- [ ] `ORdTraceData`, `ORdTraceStorage`, `ORdShardProcessor` importable
- [ ] `corrupt_states(model_type='ord')` works

### Phase 1 Cleanup
- [ ] No debug prints
- [ ] Docstrings on all public classes and methods
- [ ] Type hints on all function signatures

**Commit point: git commit after Phase 1 passes**

---

## Phase 2: ORd Generators

**Goal**: Implement ORd SingleCellGenerator and BatchGenerator, adapted from TTP06 versions.
**Tier**: large
**Estimated scope**: 2 files created, 1 file modified, ~500 lines, ~20 tests

### Phase Context
The ORd model API differs from TTP06 in several ways: `step()` uses `Istim=` not `I_stim=`; no `from_config()`; conductance scaling via `params_override`; CaMKII requires 100-beat warmup; concentration perturbation uses lowercase `ko`/`nao`/`cao` as `params_override` keys. All protocols (T1-T12) from `protocols.py` are reusable -- they only define timing and stimulus, not model-specific behavior. The model-specific parts are: (a) constructing the model, (b) the step call, (c) concentration perturbation setup, (d) CaMKII warmup.

### Step 2.1: ORd SingleCellGenerator
**Model**: opus

#### Read First
- `Surrogate/surrogate/data/single_cell_generator.py` -- full file: TTP06 generator (pattern to mirror)
- `Bidomain/Engine_V1/cardiac_sim/ionic/ord/model.py:88-103` -- ORdModel `__init__` (params_override)
- `Bidomain/Engine_V1/cardiac_sim/ionic/ord/model.py:753-858` -- `step()` method (Istim kwarg)
- `Bidomain/Engine_V1/cardiac_sim/ionic/ord/model.py:602-731` -- `compute_gate_steady_states`, `compute_gate_time_constants`
- `Bidomain/Engine_V1/cardiac_sim/ionic/ord/parameters.py:280-325` -- `get_celltype_parameters`, cell-type scaling

#### Why
Core generator for ORd single-cell traces. Must handle: CaMKII warmup before every protocol, ORd-specific `step()` signature, `params_override` for conductance scaling, ORd-specific concentration perturbation (lowercase keys, no Cai_scale). Records 101-column ORdTraceData.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/data/ord_single_cell_generator.py`
**Interfaces:**
```python
class ORdSingleCellGenerator:
    """Runs ORd single-cell ODE for any protocol, recording full state.

    Automatically performs CaMKII warmup (100 beats at BCL=1000)
    before every protocol to reach quasi-steady-state CaMKt.
    """

    _CELLTYPE_MAP = {
        'EPI': CellType.EPI,
        'ENDO': CellType.ENDO,
        'M_CELL': CellType.M_CELL,
    }

    def __init__(self, cell_type: str = 'EPI', device: str = 'cuda',
                 conductance_scaling: Optional[Dict[str, float]] = None,
                 warmup_beats: int = 100):
        # ORdModel(cell_type=..., params_override=...) -- no from_config()

    def _build_params_override(self, conductance_scaling):
        """Convert {name: scale} to {param_name: absolute_value} for params_override."""

    def _warmup(self, model, V, states) -> Tuple[Tensor, Tensor]:
        """Run 100-beat BCL=1000 warmup for CaMKII buildup. No recording."""

    def run_protocol(self, protocol) -> ORdTraceData:
        """Execute protocol with CaMKII warmup. Returns 101-col ORdTraceData."""

    def _run_loop(self, protocol, model, V, states) -> ORdTraceData:
        """Core simulation loop (mirrors TTP06 _run_loop)."""

    def _setup_concentration_perturbation(self, protocol, model):
        """ORd: create new model with params_override for ko/nao/cao."""

    def run_pacing(self, bcl, n_beats, dt=0.01, ...) -> ORdTraceData:
        """Convenience method."""
```

#### Pseudocode
```
def __init__(self, cell_type, device, conductance_scaling, warmup_beats=100):
    self.warmup_beats = warmup_beats
    if conductance_scaling:
        override = self._build_params_override(conductance_scaling)
        self.model = ORdModel(cell_type=self.cell_type_enum, device=device,
                              params_override=override)
    else:
        self.model = ORdModel(cell_type=self.cell_type_enum, device=device)

def _build_params_override(self, scaling):
    """ORd conductance names: GNa, GNaL, Gto, GKr, GKs, GK1, PCa, Gncx, Pnak, GpCa, GKb.
    Scale factors applied to the _scale parameters, not the base conductances.
    E.g. GNa_scale=2.0 means double GNa. But params_override sets absolute values,
    so we compute: override = {f'{name}_scale': value} for scale-type params,
    or {name: base * scale} for base conductances.
    """
    # ORd params are: GNa(75.0), GNaL(0.0075), Gto(0.02), etc.
    # The celltype-specific scaling is via _scale fields (GNaL_scale, etc.)
    # For data gen, apply ADDITIONAL scaling on top of celltype defaults
    # Strategy: multiply into the _scale fields
    override = {}
    # Map user-facing names to ORdParameters scale field names
    _NAME_TO_SCALE = {
        'GNa': None,         # No _scale field; override GNa directly
        'GNaL': 'GNaL_scale',
        'Gto': 'Gto_scale',
        'GKr': 'GKr_scale',
        'GKs': 'GKs_scale',
        'GK1': 'GK1_scale',
        'PCa': 'PCa_scale',
        'Gncx': 'Gncx_scale',
        'Pnak': 'Pnak_scale',
        'GpCa': None,        # Override directly
        'GKb': 'GKb_scale',
    }
    base_params = get_celltype_parameters(self.cell_type_enum)
    for name, user_scale in scaling.items():
        scale_field = _NAME_TO_SCALE.get(name)
        if scale_field is None:
            # Direct override: base_value * user_scale
            base_val = getattr(base_params, name)
            override[name] = base_val * user_scale
        else:
            # Scale field: existing_scale * user_scale
            existing_scale = getattr(base_params, scale_field)
            override[scale_field] = existing_scale * user_scale
    return override

def _warmup(self, model, V, states):
    """Silent 100-beat warmup. No recording."""
    dt = 0.01
    bcl = WARMUP_BCL
    for beat in range(self.warmup_beats):
        for step in range(int(bcl / dt)):
            t = step * dt
            I_stim = WARMUP_STIM_AMP if t < WARMUP_STIM_DUR else 0.0
            stim_tensor = torch.tensor(I_stim, dtype=torch.float64, device=self.device)
            V, states = model.step(V, states, dt, Istim=stim_tensor)
    return V, states

def run_protocol(self, protocol):
    model = self.model
    V = torch.tensor(V_REST_ORD, dtype=torch.float64, device=self.device)
    states = model.get_initial_state(n_cells=1).to(self.device)

    # Handle ORdConcentrationPerturbation BEFORE warmup
    # (warmup steady-state depends on extracellular concentrations)
    if isinstance(protocol, ORdConcentrationPerturbation):
        model = self._setup_concentration_perturbation(protocol, model)
        protocol = protocol.base_protocol

    # CaMKII warmup (always, unless warmup_beats=0)
    # Uses the potentially-modified model (correct ko/nao/cao during warmup)
    if self.warmup_beats > 0:
        V, states = self._warmup(model, V, states)

    # Handle initial_states override (CorruptionRecovery, custom starts)
    if hasattr(protocol, 'initial_states') and protocol.initial_states is not None:
        states = protocol.initial_states.to(self.device)

    # Handle StitchedProtocol
    if isinstance(protocol, StitchedProtocol):
        return self._run_stitched(protocol, model, V, states)

    # Handle CorruptionRecovery
    if isinstance(protocol, CorruptionRecovery):
        from .augmentation import corrupt_states
        states = corrupt_states(states, protocol.corruption_type,
                                protocol.severity, model_type='ord')
        protocol = protocol.base_protocol

    return self._run_loop(protocol, model, V, states)

def _run_loop(self, protocol, model, V, states):
    records = []
    t = 0.0
    while t < protocol.duration_ms:
        dt = protocol.get_dt(t)
        I_stim = protocol.get_I_stim(t)
        I_ext = protocol.get_I_ext(t)
        clamped = 1.0 if protocol.is_clamped(t) else 0.0

        I_ion = model.compute_Iion(V, states)
        recorded_stim = -(I_stim + I_ext)

        state_flat = states.squeeze(0) if states.dim() > 1 else states
        record = torch.cat([
            V.reshape(1),
            torch.tensor([recorded_stim, dt], dtype=torch.float64, device=self.device),
            state_flat,                   # 40 states
            I_ion.reshape(1),
            torch.tensor([clamped], dtype=torch.float64, device=self.device),
        ])  # shape: (45,)  -- gate_inf/tau added post-hoc
        records.append(record)

        # Advance (same clamp/free-running logic as TTP06, but Istim= kwarg)
        if protocol.is_clamped(t):
            if hasattr(protocol, 'alpha'):
                V_cmd = torch.tensor(protocol.get_clamp_voltage(t), ...)
                V_free, states = model.step(V, states, dt, Istim=None)
                V = protocol.alpha * V_cmd + (1 - protocol.alpha) * V_free
            else:
                V_clamp = torch.tensor(protocol.get_clamp_voltage(t), ...)
                _, states = model.step(V_clamp, states, dt, Istim=None)
                V = torch.tensor(protocol.get_clamp_voltage(t + dt), ...)
        else:
            total_stim = torch.tensor(I_stim + I_ext, ...)
            V_new, states = model.step(V, states, dt, Istim=total_stim)
            V = V_new
            # Adaptive dt tracking (ported from TTP06)
            if hasattr(protocol, '_last_dvdt'):
                protocol._last_dvdt = float((V_new - V) / dt) if dt > 0 else 0.0
        t += dt

    data_core = torch.stack(records)  # (T, 45)

    # Post-hoc: gate_inf (28) and gate_tau (28) vectorized
    Vm_all = data_core[:, ORdTraceData.VM]
    states_all = data_core[:, ORdTraceData.STATES_START:ORdTraceData.STATES_END]
    gate_inf = model.compute_gate_steady_states(Vm_all, states_all)   # (T, 28)
    gate_tau = model.compute_gate_time_constants(Vm_all, states_all)  # (T, 28)
    data = torch.cat([data_core, gate_inf, gate_tau], dim=1)          # (T, 101)

    metadata = {...}
    return ORdTraceData(data=data, metadata=metadata)

def _run_stitched(self, protocol, model, V, states):
    """Run StitchedProtocol: concatenate sub-protocol traces, carry state forward.
    CRITICAL: use ORdTraceData column constants (40 states), NOT TTP06's (18 states)."""
    all_traces = []
    for sub_protocol in protocol.sub_protocols:
        trace = self._run_loop(sub_protocol, model, V, states)
        all_traces.append(trace)
        # Carry forward final state using ORd column offsets
        final_row = trace.data[-1]
        V = final_row[ORdTraceData.VM].unsqueeze(0)
        states = final_row[ORdTraceData.STATES_START:ORdTraceData.STATES_END].unsqueeze(0)
    combined_data = torch.cat([t.data for t in all_traces], dim=0)
    metadata = {**all_traces[0].metadata, 'stitched': True, 'n_sub': len(all_traces)}
    return ORdTraceData(data=combined_data, metadata=metadata)

def _setup_concentration_perturbation(self, protocol, model):
    """Create new ORdModel with modified ko/nao/cao via params_override."""
    import copy
    new_model = copy.deepcopy(model)
    if protocol.ko is not None:
        new_model.params.ko = protocol.ko
    if protocol.nao is not None:
        new_model.params.nao = protocol.nao
    if protocol.cao is not None:
        new_model.params.cao = protocol.cao
    return new_model
```

#### Test Spec
- `test_ord_gen_creates_trace` -- run 50ms, verify shape (T, 101), initial Vm near -87.5
- `test_ord_gen_produces_ap` -- BCL=1000 x 1 beat post-warmup produces AP (Vm > 0)
- `test_ord_gen_iion_matches` -- I_ion column matches `model.compute_Iion()` at first timestep
- `test_ord_gen_celltypes` -- EPI, ENDO, M_CELL produce different APD (M_CELL longest)
- `test_ord_gen_warmup_camkt` -- after warmup, CaMKt > 0 (steady-state ~0.01-0.02)
- `test_ord_gen_no_warmup` -- `warmup_beats=0`: CaMKt stays near 0
- `test_ord_gen_gate_inf_tau_shape` -- gate_inf has 28 cols, gate_tau has 28 cols
- `test_ord_gen_gate_inf_physiological` -- gate_inf values in [0, 1] range
- `test_ord_gen_clamp_protocol` -- step clamp works, Vm follows command
- `test_ord_gen_istim_sign_flip` -- stimulus recorded as positive when depolarizing
- `test_ord_gen_conductance_scaling` -- scaled GKr produces different APD vs unscaled
- `test_ord_gen_stitched_state_carryforward` -- stitched protocol: final state of sub-protocol N = initial state of sub-protocol N+1, all 40 states carried
- `test_ord_gen_conc_perturbation` -- ORdConcentrationPerturbation with ko=8.0 shifts resting Vm
- `test_ord_gen_initial_states_override` -- protocol with custom initial_states uses them

#### Checklist
- [ ] ORdModel constructed with `params_override`, not `from_config()`
- [ ] `model.step()` called with `Istim=`, not `I_stim=`
- [ ] CaMKII warmup runs 100 beats at BCL=1000 before every protocol
- [ ] 101 columns: 3 header + 40 states + 2 meta + 28 gate_inf + 28 gate_tau
- [ ] gate_inf/tau computed post-hoc via `compute_gate_steady_states` / `compute_gate_time_constants`
- [ ] Concentration perturbation uses `params.ko`, `params.nao`, `params.cao` (lowercase)
- [ ] Corruption recovery passes `model_type='ord'` to `corrupt_states()`
- [ ] Metadata includes `model='ord'`, `warmup_beats`, `cell_type`

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/test_ord_data_generation.py::TestORdSingleCellGenerator -v
```

#### Exit Criteria
- [ ] 11 tests pass
- [ ] AP shape physiologically valid (peak > 0 mV, rest < -80 mV)
- [ ] CaMKt > 0 after warmup

#### Risk
CaMKII warmup is slow (~100K steps at dt=0.01). Mitigation: tests use `warmup_beats=5` or CPU with short protocols. Production runs use GPU. Consider `warmup_beats` parameter for test flexibility.

---

### Step 2.2: ORd ConcentrationPerturbation protocol
**Model**: sonnet

#### Read First
- `Surrogate/surrogate/data/protocols.py:238-264` -- TTP06 `ConcentrationPerturbation`
- `Bidomain/Engine_V1/cardiac_sim/ionic/ord/parameters.py:172-174` -- `nao`, `cao`, `ko` fields

#### Why
TTP06 `ConcentrationPerturbation` uses `Ko`, `Nai_init`, `Cai_scale` -- TTP06-specific field names and semantics. ORd uses `params_override` with lowercase `ko`, `nao`, `cao`. Need a separate protocol class that the ORd generator recognizes via `isinstance()`.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/data/ord_protocols.py`
**Interfaces:**
```python
class ORdConcentrationPerturbation(Protocol):
    """Modify extracellular concentrations for ORd model. Tier 7.

    ORd uses params_override with lowercase keys (ko, nao, cao).
    ORd SingleCellGenerator detects this type and modifies ORdModel.params directly.
    """

    def __init__(self, base_protocol: Protocol,
                 ko: Optional[float] = None,
                 nao: Optional[float] = None,
                 cao: Optional[float] = None):
        # Delegates all protocol methods to base_protocol

class ORdProtocolLibrary:
    """ORd-specific protocol factory. Adds T13 and ORd-specific tier variants."""

    @staticmethod
    def tier7_concentration(base_protos: List[Protocol]) -> List[Protocol]:
        """Wrap base protocols with ORd concentration perturbations."""

    @staticmethod
    def tier13_camkii_buildup() -> Protocol:
        """T13: 1000-beat recording of full CaMKII buildup from zero."""
```

#### Pseudocode
```
class ORdConcentrationPerturbation(Protocol):
    def __init__(self, base_protocol, ko=None, nao=None, cao=None):
        super().__init__(name=f'ord_conc_ko{ko}_{base_protocol.name}', tier=7,
                         duration_ms=base_protocol.duration_ms,
                         dt_default=base_protocol.dt_default)
        self.base_protocol = base_protocol
        self.ko = ko
        self.nao = nao
        self.cao = cao
    # Delegate get_I_stim, get_I_ext, get_dt, is_clamped, get_clamp_voltage

class ORdProtocolLibrary:
    @staticmethod
    def tier7_concentration(base_protos):
        result = []
        # Hyperkalemia: ko = 7.0, 9.0
        for ko in [7.0, 9.0]:
            for bp in base_protos[:2]:  # first 2 base protos
                result.append(ORdConcentrationPerturbation(bp, ko=ko))
        # Hyponatremia: nao = 120
        for bp in base_protos[:2]:
            result.append(ORdConcentrationPerturbation(bp, nao=120.0))
        # Low Ca: cao = 1.0
        for bp in base_protos[:2]:
            result.append(ORdConcentrationPerturbation(bp, cao=1.0))
        return result

    @staticmethod
    def tier13_camkii_buildup():
        """1000-beat pacing recording full CaMKt buildup.
        Generator must use warmup_beats=0 for this protocol."""
        return SteadyStatePacing(bcl=T13_BCL, n_beats=T13_N_BEATS, dt_default=T13_DT)
```

#### Test Spec
- `test_ord_conc_perturb_creates` -- construct with ko=7.0, verify attributes
- `test_ord_conc_perturb_delegates` -- get_I_stim, is_clamped match base protocol
- `test_ord_conc_perturb_isinstance` -- isinstance(p, ORdConcentrationPerturbation) is True
- `test_ord_protocol_tier13` -- T13 has 1000 beats, BCL=1000

#### Checklist
- [ ] Uses lowercase `ko`, `nao`, `cao` (NOT TTP06 `Ko`, `Nai_init`)
- [ ] Delegates all timing methods to base_protocol
- [ ] T13 uses `warmup_beats=0` (generator handles this at call site)
- [ ] ORd generator detects via `isinstance(protocol, ORdConcentrationPerturbation)`

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/test_ord_data_generation.py::TestORdProtocols -v
```

#### Exit Criteria
- [ ] 4 tests pass
- [ ] ORdConcentrationPerturbation is distinct from TTP06 ConcentrationPerturbation

#### Risk
Confusing TTP06 and ORd concentration protocols at call site. Mitigation: distinct class names, isinstance check.

---

### Step 2.3: ORd BatchGenerator
**Model**: opus

#### Read First
- `Surrogate/surrogate/data/batch_generator.py` -- full file: TTP06 BatchGenerator (pattern to mirror)
- `Surrogate/surrogate/data/ord_single_cell_generator.py` -- Step 2.1 output
- `Bidomain/Engine_V1/cardiac_sim/ionic/ord/model.py:88-103` -- ORdModel constructor
- `Bidomain/Engine_V1/cardiac_sim/ionic/ord/model.py:602-731` -- gate_steady_states, gate_time_constants

#### Why
GPU-batched generation for ORd. Key differences from TTP06 BatchGenerator: (1) uses ORdModel, (2) records 40 states instead of 18, (3) post-hoc computes 28 gate_inf/tau instead of 12, (4) CaMKII warmup must be done in batch before recording begins, (5) `model.step()` uses `Istim=` kwarg.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/data/ord_batch_generator.py`
**Interfaces:**
```python
class ORdBatchGenerator:
    """Batched ORd generator -- runs N protocols in parallel.

    CaMKII warmup is done in batch before recording.
    """

    def __init__(self, cell_type: str = 'EPI', device: str = 'cuda',
                 use_compile: bool = True, warmup_beats: int = 100):

    def _warmup_batch(self, n: int) -> Tuple[Tensor, Tensor]:
        """Batch warmup: n cells x 100 beats at BCL=1000. Returns (V, states)."""

    def run_batch(self, protocols: List[Protocol],
                  progress_interval: float = 10.0,
                  record_every: int = 1) -> List[ORdTraceData]:
```

#### Pseudocode
```
def __init__(self, cell_type, device, use_compile, warmup_beats=100):
    self.model = ORdModel(cell_type=self.cell_type_enum, device=device)
    self.warmup_beats = warmup_beats
    if use_compile and device == 'cuda':
        self._step_fn = torch.compile(self.model.step)
        # Warmup compile
        V = torch.full((2,), V_REST_ORD, ...)
        s = self.model.get_initial_state(n_cells=2).to(device)
        self._step_fn(V, s, 0.01, Istim=torch.zeros(2, ...))
    else:
        self._step_fn = self.model.step

def _warmup_batch(self, n):
    V = torch.full((n,), V_REST_ORD, dtype=torch.float64, device=self.device)
    states = self.model.get_initial_state(n_cells=n).to(self.device)
    dt = 0.01
    steps_per_beat = int(WARMUP_BCL / dt)
    for beat in range(self.warmup_beats):
        for step in range(steps_per_beat):
            t_in_beat = step * dt
            I_stim = WARMUP_STIM_AMP if t_in_beat < WARMUP_STIM_DUR else 0.0
            stim = torch.full((n,), I_stim, dtype=torch.float64, device=self.device)
            V, states = self._step_fn(V, states, dt, Istim=stim)
    return V, states

def run_batch(self, protocols, progress_interval=10.0, record_every=1):
    n = len(protocols)
    # Batch warmup
    V, states = self._warmup_batch(n)

    # Pre-allocate: 40 states instead of 18
    all_states = torch.zeros((n_records, n, 40), ...)

    # Main loop (same structure as TTP06 BatchGenerator)
    for step in range(n_steps):
        ...
        V_new, states_new = self._step_fn(V, states, dt, Istim=total_stim)
        ...

    # Post-hoc: 28 gate_inf, 28 gate_tau
    Vm_flat = all_Vm[:record_idx].reshape(-1)
    states_flat = all_states[:record_idx].reshape(-1, 40)
    gate_inf_flat = self.model.compute_gate_steady_states(Vm_flat, states_flat)   # (N, 28)
    gate_tau_flat = self.model.compute_gate_time_constants(Vm_flat, states_flat)  # (N, 28)
    all_gate_inf = gate_inf_flat.reshape(record_idx, n, 28)
    all_gate_tau = gate_tau_flat.reshape(record_idx, n, 28)

    # Assemble (T, 101) per-protocol
    for i, proto in enumerate(protocols):
        data = torch.cat([
            all_Vm[:pr, i].unsqueeze(1),                          # col 0
            all_stim[:pr, i].unsqueeze(1),                        # col 1
            torch.full((pr, 1), dt, dtype=torch.float64),         # col 2
            all_states[:pr, i],                                   # cols 3-42 (40 states)
            all_Iion[:pr, i].unsqueeze(1),                        # col 43
            all_clamp[:pr, i].unsqueeze(1),                       # col 44
            all_gate_inf[:pr, i],                                 # cols 45-72 (28)
            all_gate_tau[:pr, i],                                 # cols 73-100 (28)
        ], dim=1)  # (T, 101)
        traces.append(ORdTraceData(data=data, metadata={...}))
```

#### Test Spec
- `test_ord_batch_single` -- batch of 1 protocol, verify 101 columns
- `test_ord_batch_multiple` -- batch of 3 protocols with different durations
- `test_ord_batch_matches_single` -- batch result (n=1) matches single-cell generator for first 100 steps (within float64 tolerance)
- `test_ord_batch_warmup_camkt` -- CaMKt > 0 in output traces
- `test_ord_batch_gate_inf_shape` -- 28 gate_inf, 28 gate_tau per step
- `test_ord_batch_clamp` -- clamp protocol works in batch mode

#### Checklist
- [ ] `ORdModel` instantiated, not `TTP06Model`
- [ ] `step()` called with `Istim=`, not `I_stim=`
- [ ] CaMKII batch warmup before recording loop
- [ ] Pre-allocated `all_states` has shape `(n_records, n, 40)`, not `(n_records, n, 18)`
- [ ] Post-hoc gate_inf/tau: 28 columns each
- [ ] Output: `ORdTraceData` with 101 columns
- [ ] `torch.compile` warmup uses ORdModel.step, not TTP06

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/test_ord_data_generation.py::TestORdBatchGenerator -v
```

#### Exit Criteria
- [ ] 6 tests pass
- [ ] Batch output has 101 columns
- [ ] CaMKt > 0 in all output traces

#### Risk
Batch warmup is expensive (100 beats x 100K steps per beat x n cells). For large n, this dominates runtime. Mitigation: warmup is embarassingly parallel -- same code that runs the recording loop. GPU handles it. For tests, use `warmup_beats=2` and `n_beats=1`.

---

### Phase 2 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/ -v
```

### Phase 2 Exit Criteria
- [ ] ~20 new ORd generator tests pass
- [ ] 32 existing TTP06 tests still pass
- [ ] Single-cell and batch generators produce physiologically valid ORd APs
- [ ] CaMKII warmup verified (CaMKt > 0)

### Phase 2 Cleanup
- [ ] No debug prints
- [ ] All public methods have docstrings
- [ ] Warmup progress messages (print) for long runs

**Commit point: git commit after Phase 2 passes**

---

## Phase 3: Generate Data

**Goal**: Generate T1-T13 ORd training data for all celltypes and dt values.
**Tier**: medium (mostly scripting, long GPU runtime)
**Estimated scope**: 1 script file, ~200 lines. Runtime: ~24-48h GPU time.

### Phase Context
With Phase 1-2 complete, generating data is a matter of scripting the protocol library calls. The existing TTP06 generation scripts in `Surrogate/` are the pattern. ORd generation is more expensive per step (2.8x TTP06) and requires CaMKII warmup (100 beats). Use batch generator for T1-T4 (same-dt protocols), single-cell for T5-T12 (mixed protocols).

### Step 3.1: ORd data generation script
**Model**: opus

#### Read First
- `Surrogate/surrogate/data/protocols.py:299-330` -- `ProtocolLibrary` (tier1-tier4 factories)
- `Surrogate/surrogate/data/ord_protocols.py` -- `ORdProtocolLibrary`, `ORdConcentrationPerturbation` (from Step 2.2)
- `Surrogate/surrogate/data/ord_single_cell_generator.py` -- ORd single-cell gen (Step 2.1)
- `Surrogate/surrogate/data/ord_batch_generator.py` -- ORd batch gen (Step 2.3)
- `Surrogate/surrogate/data/ord_storage.py` -- `ORdTraceStorage` (Step 1.3)
- `Surrogate/surrogate/data/injection.py` -- OUNoiseInjection, RampInjection, etc. (model-agnostic)
- `Surrogate/surrogate/data/clamp.py` -- StepClamp, RampClamp, etc. (model-agnostic)

#### Why
Master script that generates all ORd training data. Must cover T1-T13 for 3 celltypes and 2 dt values. T13 is ORd-specific (CaMKII buildup recording with warmup_beats=0). Tier assignments follow TTP06 pattern where protocols are model-agnostic, except T7 (concentration) and T9 (corruption) which use ORd-specific wrappers.

#### Implementation Spec
**Files to create:** `Surrogate/run_ord_datagen.py`
**Structure:**
```python
"""ORd data generation script.

Usage:
    conda run -n heart-conduction python run_ord_datagen.py --tier 1 --celltype EPI
    conda run -n heart-conduction python run_ord_datagen.py --tier all --celltype all
    conda run -n heart-conduction python run_ord_datagen.py --tier 13 --celltype ENDO
"""

def generate_tier(tier: int, cell_type: str, dt: float, device: str):
    """Generate one tier for one celltype at one dt."""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tier', default='all')
    parser.add_argument('--celltype', default='all')
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--device', default='cuda')
    ...
```

#### Pseudocode
```
TIER_CONFIG = {
    # Batch-compatible tiers (same dt, can parallelize)
    1: ('batch', ProtocolLibrary.tier1),
    2: ('batch', ProtocolLibrary.tier2),
    3: ('batch', ProtocolLibrary.tier3),
    4: ('batch', ProtocolLibrary.tier4),

    # Single-cell tiers (model-agnostic protocols)
    5: ('single', make_tier5_injection_protocols),
    6: ('single', make_tier6_clamp_protocols),

    # ORd-specific
    7: ('single', ORdProtocolLibrary.tier7_concentration),
    8: ('batch', make_tier8_variable_dt),     # model-agnostic

    # Augmentation tiers
    9: ('single', make_tier9_corruption),      # uses corrupt_states(model_type='ord')
    10: ('single', make_tier10_conductance),    # conductance_scaling
    11: ('single', make_tier11_stitched),       # model-agnostic
    12: ('single', make_tier12_boundary_cell),  # model-agnostic

    # ORd exclusive
    13: ('single', make_tier13_camkii),         # warmup_beats=0, record full buildup
}

def generate_tier(tier, cell_type, dt, device):
    mode, factory = TIER_CONFIG[tier]
    storage = ORdTraceStorage()

    if mode == 'batch':
        gen = ORdBatchGenerator(cell_type=cell_type, device=device)
        protos = factory() if tier != 7 else factory(ProtocolLibrary.tier1()[:2])
        # Adjust dt
        for p in protos:
            p.dt_default = dt
        traces = gen.run_batch(protos)
        storage.save_tier(traces, tier)
    else:
        gen = ORdSingleCellGenerator(cell_type=cell_type, device=device)
        if tier == 13:
            gen = ORdSingleCellGenerator(cell_type=cell_type, device=device,
                                          warmup_beats=0)  # Record from zero CaMKt
        protos = factory() if callable(factory) else factory
        for proto in protos:
            trace = gen.run_protocol(proto)
            storage.save_trace(trace, tier, trace.metadata['protocol_name'])

    print(f'  Tier {tier} ({cell_type}): {len(protos)} protocols saved')
```

#### Test Spec
- `test_ord_datagen_tier1_smoke` -- generate T1 EPI with 1 BCL, 1 beat, verify HDF5 written
- `test_ord_datagen_tier13_no_warmup` -- T13 starts with CaMKt=0, builds up over 1000 beats
- `test_ord_datagen_tier7_conc_perturb` -- T7 with ko=7.0 modifies AP shape
- `test_ord_datagen_tier9_corruption` -- T9 corrupts ORd gates (not TTP06 gates)

#### Checklist
- [ ] All 13 tiers defined in TIER_CONFIG
- [ ] T7 uses `ORdConcentrationPerturbation`, not TTP06 `ConcentrationPerturbation`
- [ ] T9 passes `model_type='ord'` to `corrupt_states()`
- [ ] T13 uses `warmup_beats=0` to record full CaMKII buildup from zero
- [ ] Storage writes to `ORD_RAW_DIR` (separate from TTP06)
- [ ] CLI supports `--tier all`, `--celltype all`, `--dt 0.005`
- [ ] Progress printing for long-running tiers

#### Verify
```bash
# Smoke test (fast, CPU)
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python run_ord_datagen.py --tier 1 --celltype EPI --dt 0.1 --device cpu --quick

# Full test suite
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/test_ord_data_generation.py::TestORdDataGen -v
```

#### Exit Criteria
- [ ] 4 smoke tests pass
- [ ] T1 produces valid HDF5 with 101-col data
- [ ] T13 CaMKt monotonically increases from 0

#### Risk
Long runtime (~48h for full dataset). Mitigation: script supports `--tier` and `--celltype` for incremental generation. `--quick` flag for smoke tests (1 beat, large dt).

---

### Phase 3 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/ -v
```

### Phase 3 Exit Criteria
- [ ] ~4 datagen smoke tests pass
- [ ] All previous tests pass (~47 total: 32 TTP06 + ~15 infra + ~20 generator)
- [ ] T1 HDF5 file exists and is loadable
- [ ] T13 CaMKII buildup recorded correctly

### Phase 3 Cleanup
- [ ] Script has `--help` with usage examples
- [ ] Progress messages for each tier/celltype
- [ ] Error handling for missing external HDD

**Commit point: git commit after Phase 3 passes**

---

## Final Cleanup

1. Update `Surrogate/surrogate/data/__init__.py` to export ORd generators:
```python
from .ord_trace import ORdTraceData
from .ord_single_cell_generator import ORdSingleCellGenerator
from .ord_storage import ORdTraceStorage
```

2. Archive plan:
```bash
# Plan is already in plans/ directory -- no move needed
```

## File Summary

| File | Action | Phase |
|------|--------|-------|
| `Surrogate/surrogate/data/ord_trace.py` | CREATE | 1.1 |
| `Surrogate/surrogate/data/augmentation.py` | MODIFY | 1.2 |
| `Surrogate/surrogate/data/ord_storage.py` | CREATE | 1.3 |
| `Surrogate/surrogate/data/ord_single_cell_generator.py` | CREATE | 2.1 |
| `Surrogate/surrogate/data/ord_protocols.py` | CREATE | 2.2 |
| `Surrogate/surrogate/data/ord_batch_generator.py` | CREATE | 2.3 |
| `Surrogate/run_ord_datagen.py` | CREATE | 3.1 |
| `Surrogate/tests/test_ord_data_generation.py` | CREATE | 1-3 |
| `Surrogate/surrogate/data/__init__.py` | MODIFY | final |

## Mutation Log
(Empty -- populated during execution)
