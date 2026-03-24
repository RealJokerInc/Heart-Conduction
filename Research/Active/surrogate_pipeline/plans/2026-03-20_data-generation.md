# PLAN: Ionic Surrogate Training Data Generation

Created: 2026-03-20
Engine(s): Bidomain V1 (TTP06 source)
Research question: [surrogate_pipeline](README.md)
Source: [IDEALOG.md](IDEALOG.md) — Session 8: Training data generation plan

## Objective
Build the complete data generation pipeline for the ionic surrogate model. Generate TTP06 single-cell ODE trajectories across 12 protocol tiers, store as HDF5 (archival) and pre-chunked .pt shards (training speed). This provides all training data needed for Phases A–D of the training curriculum.

## Success Criteria
- [ ] Single-cell data generation pipeline (TTP06 ODE trajectories) — README.md completion criterion #1
- [ ] All 12 protocol tiers generate valid data
- [ ] HDF5 raw storage with full metadata per protocol
- [ ] .pt shard pre-processor produces training-ready float32 segments
- [ ] Validation: generated AP traces match known TTP06 behavior (V_rest, APD, restitution curve)
- [ ] All existing Bidomain V1 tests pass (no regressions)
- [ ] All 32 tests pass (Phase 1: 10, Phase 2: 8, Phase 3: 8, Phase 4: 6)

## Architecture Changes
- NEW: `Surrogate/surrogate/__init__.py` — package init
- NEW: `Surrogate/surrogate/data/__init__.py` — data subpackage
- NEW: `Surrogate/surrogate/data/single_cell_generator.py` — TTP06 ODE wrapper for protocol execution
- NEW: `Surrogate/surrogate/data/protocols.py` — Protocol definitions (Tiers 1-12)
- NEW: `Surrogate/surrogate/data/injection.py` — Current injection profiles (Tier 5 OU noise, ramps, etc.)
- NEW: `Surrogate/surrogate/data/clamp.py` — Voltage clamp protocols (Tier 6)
- NEW: `Surrogate/surrogate/data/augmentation.py` — Conductance scaling, stitching, corruption
- NEW: `Surrogate/surrogate/data/storage.py` — HDF5 write/read + .pt shard conversion
- NEW: `Surrogate/tests/test_data_generation.py` — Validation suite

## Known Failures (from IDEALOG)
- Non-uniform temporal sampling schedule — abandoned (carried latent eliminates need for history buffer)
- 300-point lookback buffer — abandoned (stimulus artifacts, coverage issues)
- These are architecture decisions, not data generation failures. No data generation approaches have been tried yet.

**Note**: Corrected test counts and column counts are canonical in this PLAN. IDEALOG may contain stale numbers from earlier sessions.

---

## Phase 1: Core Generator + Basic Protocols (Tiers 1-3)

**Goal**: Build the TTP06 wrapper, implement basic pacing protocols, store to HDF5. This produces enough data to start training Phase A (autoencoder bootstrap) and Phase B (simple dynamics).
**Tier**: medium
**Estimated scope**: 4 files, ~500 lines, 10 tests

### Phase Context
- TTP06 lives at `Bidomain/Engine_V1/cardiac_sim/ionic/ttp06/`. Use `TTP06Model` class directly — do NOT copy ionic code.
- Key interface: `model.step(V, ionic_states, dt, Istim)` → `(V_new, states_new)`. Also `model.compute_Iion(V, states)` for I_ion, `model.get_initial_state()` for initial conditions.
- 18 state variables: indices 0-4 are concentrations (Ki, Nai, Cai, CaSR, CaSS), indices 5-16 are gates (m, h, j, r, s, d, f, f2, fCass, Xr1, Xr2, Xs), index 17 is RR.
- 3 celltypes: `CellType.EPI`, `CellType.ENDO`, `CellType.M_CELL` — different Gto, GKs conductances.
- All tensors float64 for generation (convert to float32 in Phase 4 shard processor).
- Device: use `'cuda'` when available.
- conda env: `heart-conduction`.
- Output segment format: `(timesteps, 23)` where columns = [Vm, I_stim, dt, 18 gate/concentration states, I_ion, clamp_mask].
- **Data storage**: External HDD at `/media/norepinephrine/Elements-ext4/` (5.2TB free, ext4). Raw HDF5 and .pt shards go here — dataset is ~1.1TB, too large for the 244GB NVMe root partition.
  - Raw: `/media/norepinephrine/Elements-ext4/surrogate_data/raw/`
  - Train shards: `/media/norepinephrine/Elements-ext4/surrogate_data/train/`
  - Val shards: `/media/norepinephrine/Elements-ext4/surrogate_data/val/`
- **I_stim sign convention**: TTP06 uses `dV = -(I_ion + I_stim) * dt`, where depolarizing I_stim is negative (e.g., -80 µA/µF). The surrogate training formula is `Vm_next = Vm + dt * (-I_ion + I_stim) / Cm`. To make these consistent, **negate I_stim when recording**: `recorded_I_stim = -(I_stim_ttp06 + I_ext)`. This way the surrogate sees depolarizing stimulus as positive, and the formula produces the same Vm update as TTP06. Both I_stim_ttp06 and I_ext follow TTP06 sign convention (negative = depolarizing).

  **Worked example (sign math)**:
  - TTP06 step(): `dV = -(I_ion + total_stim) * dt` where `total_stim = I_stim + I_ext`
  - With I_stim = -80 (depolarizing): `dV = -(I_ion + (-80)) * dt = (-I_ion + 80) * dt` → Vm goes UP (correct)
  - Recorded: `recorded_stim = -(I_stim + I_ext) = -(-80 + 0) = 80` (positive = depolarizing in surrogate convention)
  - Surrogate: `dV = (-I_ion_pred + recorded_stim) * dt / Cm = (-I_ion_pred + 80) * dt / 1.0` → matches TTP06 when Cm=1
  - Key: `compute_Iion()` returns PURE ionic current (no stimulus). Column 21 (I_ion) = pure ionic. Column 1 (I_stim) = sign-flipped total external current.

### Step 1.1: Package Scaffold
**Model**: sonnet

#### Read First
- `Surrogate/README.md:66-101` — planned package structure

#### Why
Need the Python package structure before any code. The `surrogate` package will eventually hold models, training, and inference — but this plan only creates `data/`.

#### Implementation Spec
**Files to create:**
- `Surrogate/surrogate/__init__.py` — empty, marks package
- `Surrogate/surrogate/data/__init__.py` — exports SingleCellGenerator, ProtocolLibrary

#### Pseudocode
```python
# surrogate/__init__.py
"""Surrogate model for bidomain cardiac simulation."""

# surrogate/data/__init__.py
"""Data generation for ionic surrogate training."""
from .single_cell_generator import SingleCellGenerator
from .protocols import ProtocolLibrary
```

#### Test Spec
- `test_data_generation.py::test_package_scaffold` — Verify import: `from surrogate.data import SingleCellGenerator`. Expected: import succeeds without error.

#### Checklist
- [ ] Create `Surrogate/surrogate/__init__.py`
- [ ] Create `Surrogate/surrogate/data/__init__.py`
- [ ] No `__init__.py` for `Surrogate/tests/` (pytest discovers without it)
- [ ] Verify import from project root

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -c "from surrogate.data import SingleCellGenerator; print('OK')"
```

#### Exit Criteria
- [ ] `surrogate.data` importable

#### Risk
Import path issues if Surrogate/ not on PYTHONPATH — mitigation: use `PYTHONPATH=. python` or add sys.path in tests.

---

### Step 1.2: Single-Cell Generator
**Model**: opus

#### Read First
- `Bidomain/Engine_V1/cardiac_sim/ionic/ttp06/model.py` — TTP06Model class, `step()` and `compute_Iion()` interface
- `Bidomain/Engine_V1/cardiac_sim/ionic/ttp06/parameters.py` — StateIndex enum, initial state values
- `Bidomain/Engine_V1/cardiac_sim/ionic/ttp06/celltypes/standard.py` — CellType enum, EPI/ENDO/M_CELL, CellTypeConfig

#### Why
Central class that runs TTP06 ODE for any pacing protocol and records (Vm, I_stim, dt, 18 states, I_ion, clamp_mask) at every timestep. All 12 tiers use this as the execution engine. Must handle: variable I_stim profiles, variable dt, voltage clamp mode, and arbitrary initial conditions.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/data/single_cell_generator.py`

**Key class:**
```python
class SingleCellGenerator:
    def __init__(self, cell_type: str = 'EPI', device: str = 'cuda',
                 conductance_scaling: Optional[Dict[str, float]] = None):
        """
        Args:
            cell_type: 'EPI', 'ENDO', or 'M_CELL'
            device: torch device
            conductance_scaling: optional {conductance_name: scale_factor} for augmentation.
                Scale factors are relative (e.g., 0.5 = half, 2.0 = double).
                Applied by looking up base conductance values from the default
                CellTypeConfig for the cell type, multiplying by scale factors
                to get absolute values, then creating a CellTypeConfig with
                those absolute values and passing to TTP06Model.from_config().
                Example: {'GKr': 0.5} halves I_Kr conductance.
        """
        # Map string to CellType enum
        _celltype_map = {'EPI': CellType.EPI, 'ENDO': CellType.ENDO, 'M_CELL': CellType.M_CELL}
        self.cell_type_enum = _celltype_map[cell_type]

        if conductance_scaling:
            config = self._make_scaled_config(conductance_scaling)
            self.model = TTP06Model.from_config(config, base_cell_type=self.cell_type_enum,
                                                 device=device)
        else:
            self.model = TTP06Model(cell_type=self.cell_type_enum, device=device)

    @staticmethod
    def _get_base_conductances(cell_type_enum: CellType) -> Dict[str, float]:
        """Look up default conductance values for this cell type.

        Returns dict mapping conductance names to their absolute base values,
        sourced from get_celltype_parameters(). Used to convert scale factors
        to absolute values for CellTypeConfig.
        """
        from cardiac_sim.ionic.ttp06.parameters import get_celltype_parameters
        base_params = get_celltype_parameters(cell_type_enum)
        return {
            'GNa': base_params.GNa, 'GK1': base_params.GK1,
            'Gto': base_params.Gto, 'GKr': base_params.GKr,
            'GKs': base_params.GKs, 'GCaL': base_params.PCa,  # PCa is the L-type param
            'GpCa': base_params.GpCa, 'GpK': base_params.GpK,
            'GbNa': base_params.GbNa, 'GbCa': base_params.GbCa,
            'PCa': base_params.PCa, 'KNaCa': base_params.KNaCa,
            'PNaK': base_params.PNaK,
        }

    def _make_scaled_config(self, conductance_scaling: Dict[str, float]) -> CellTypeConfig:
        """Convert scale factors to absolute conductances and build CellTypeConfig.

        Args:
            conductance_scaling: {name: scale_factor}, e.g. {'GKr': 0.5}

        Returns:
            CellTypeConfig with absolute conductance values
        """
        base = self._get_base_conductances(self.cell_type_enum)
        absolute_values = {}
        for name, scale in conductance_scaling.items():
            if name not in base:
                raise ValueError(f"Unknown conductance '{name}'. "
                                 f"Valid: {list(base.keys())}")
            absolute_values[name] = base[name] * scale

        return CellTypeConfig(name=f'scaled_{self.cell_type_enum.value}',
                              **absolute_values)

    def run_protocol(self, protocol: Protocol) -> TraceData:
        """Execute a protocol and return full trace data.

        Returns TraceData with:
            data: Tensor (T, 23) — [Vm, I_stim, dt, 18 states, I_ion, clamp_mask]
            metadata: dict with protocol info, cell_type, conductances

        Note: for PartialClamp protocols, this method handles the partial
        clamp as a special case — instead of fully overriding Vm, it blends
        V_cmd with V_free: Vm = alpha*V_cmd + (1-alpha)*V_free.
        """

    def run_pacing(self, bcl: float, n_beats: int, dt: float = 0.01,
                   stim_amplitude: float = -80.0, stim_duration: float = 1.0,
                   I_ext: Optional[Callable] = None) -> TraceData:
        """Convenience for simple pacing protocols."""
```

**Data container:**
```python
@dataclass
class TraceData:
    data: torch.Tensor          # (T, 23) float64
    metadata: Dict[str, Any]    # protocol info, cell_type, etc.

    # Column indices
    VM = 0
    I_STIM = 1
    DT = 2
    STATES_START = 3    # columns 3-20: 18 ionic states
    STATES_END = 21     # exclusive end (Python convention): states are columns 3..20 inclusive
    I_ION = 21
    CLAMP_MASK = 22     # 0.0 = free-running, 1.0 = clamped
```

#### Pseudocode
```python
def run_protocol(self, protocol):
    V = torch.tensor(self.model.V_rest, dtype=torch.float64, device=self.device)
    states = self.model.get_initial_state(n_cells=1).to(self.device)
    if protocol.initial_states is not None:
        states = protocol.initial_states.clone()

    records = []
    t = 0.0
    while t < protocol.duration_ms:
        dt = protocol.get_dt(t)
        I_stim = protocol.get_I_stim(t)
        I_ext = protocol.get_I_ext(t)  # tissue-mimicking injection
        clamped = 1.0 if protocol.is_clamped(t) else 0.0

        I_ion = self.model.compute_Iion(V, states)

        # Record: [Vm, -I_stim-I_ext (sign-flipped), dt, 18 states, I_ion, clamp_mask]
        # Sign flip: TTP06 uses dV = -(I_ion+I_stim), surrogate uses dV = (-I_ion+I_stim)/Cm
        # So recorded I_stim = -(I_stim_ttp06 + I_ext)
        recorded_stim = -(I_stim + I_ext)
        record = torch.cat([V.unsqueeze(0),
                           torch.tensor([recorded_stim, dt], device=self.device),
                           states.squeeze(0),
                           I_ion.unsqueeze(0),
                           torch.tensor([clamped], device=self.device)])
        records.append(record)

        if protocol.is_clamped(t):
            # PartialClamp: blend V_cmd with V_free instead of full override
            if hasattr(protocol, 'alpha'):
                V_cmd = protocol.get_clamp_voltage(t)
                # Step freely to get V_free
                V_free, states = self.model.step(V, states, dt, I_stim=torch.tensor(0.0))
                V = protocol.alpha * V_cmd + (1 - protocol.alpha) * V_free
            else:
                V = protocol.get_clamp_voltage(t)
                # Still update states (gates respond to clamped Vm)
                _, states = self.model.step(V, states, dt, I_stim=torch.tensor(0.0))
                V = protocol.get_clamp_voltage(t + dt)  # re-clamp
        else:
            total_stim = I_stim + I_ext
            V_new, states = self.model.step(V, states, dt, I_stim=torch.tensor(total_stim))
            V = V_new

        t += dt

    data = torch.stack(records)  # (T, 23)
    return TraceData(data=data, metadata={...})
```

#### Test Spec
- `test_data_generation.py::test_generator_creates_trace` — Run 100ms pacing at BCL=1000ms. Expected: trace shape (T, 23) where T = int(100/0.01), V starts at ~-85 mV. Column 22 (clamp_mask) is all 0.0.
- `test_data_generation.py::test_generator_produces_ap` — Run 500ms with stimulus at t=10ms. Expected: V reaches >0 mV during upstroke, returns to <-80 mV.
- `test_data_generation.py::test_generator_iion_matches_ttp06` — Compare I_ion column with `model.compute_Iion()` at same (V, states). Expected: exact match.
- `test_data_generation.py::test_generator_celltypes` — Run with EPI, ENDO, M_CELL. Expected: different APD (EPI~340, ENDO~300, M~400 ms).

#### Checklist
- [ ] Implement SingleCellGenerator with TTP06Model wrapper
- [ ] Implement TraceData container with 23 columns (including clamp_mask)
- [ ] Cell_type string→enum conversion in __init__
- [ ] Conductance scaling via `_make_scaled_config()` helper: convert scale factors to absolute values using base conductances from `get_celltype_parameters()`, build CellTypeConfig, pass to `TTP06Model.from_config(config, base_cell_type=...)`
- [ ] Handle variable dt per step via time-based loop (`while t < duration_ms`)
- [ ] Handle external current injection (I_ext)
- [ ] Handle voltage clamp mode (protocol.is_clamped) — record clamp_mask=1.0
- [ ] Handle PartialClamp as special case: blend V_cmd with V_free
- [ ] I_stim sign flip when recording (negate TTP06 convention for surrogate formula)
- [ ] Handle custom initial conditions
- [ ] HDD pre-flight check: verify `/media/norepinephrine/Elements-ext4/` is mounted before writing
- [ ] 4 tests passing

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/test_data_generation.py::test_generator_creates_trace tests/test_data_generation.py::test_generator_produces_ap tests/test_data_generation.py::test_generator_iion_matches_ttp06 tests/test_data_generation.py::test_generator_celltypes -v
```

#### Exit Criteria
- [ ] SingleCellGenerator runs TTP06 and produces (T, 23) traces
- [ ] I_ion column exactly matches TTP06 model output
- [ ] I_stim column is sign-flipped relative to TTP06 convention
- [ ] clamp_mask column is 0.0 for free-running protocols
- [ ] All 3 celltypes produce physiological APs
- [ ] 4 tests pass

#### Risk
TTP06 import path — the Bidomain engine is not a pip-installed package. Mitigation: add `Bidomain/Engine_V1` to sys.path in the generator, or use relative imports.

---

### Step 1.3: Protocol Definitions (Tiers 1-3)
**Model**: opus

#### Read First
- `Research/Active/surrogate_pipeline/KNOWLEDGE.md:149-153` — Tier 1-3 specifications
- Step 1.2's SingleCellGenerator interface

#### Why
Define Protocol objects for Tiers 1-3 (steady-state, S1-S2, dynamic). These are the simplest protocols — clean pacing with known I_stim patterns. Used for training Phases A and B.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/data/protocols.py`

**Protocol base class — regular class (not @dataclass) to avoid required-field inheritance issues:**
```python
class Protocol:
    """Base class for pacing protocols.

    NOT a dataclass. Subclasses define their own __init__ and call super().__init__().
    The run loop uses `while t < protocol.duration_ms`, not a fixed step count.
    """
    def __init__(self, name: str, tier: int, duration_ms: float,
                 dt_default: float = 0.01, initial_states=None):
        self.name = name
        self.tier = tier
        self.duration_ms = duration_ms
        self.dt_default = dt_default
        self.initial_states = initial_states  # Optional[torch.Tensor]

    def get_I_stim(self, t: float) -> float: ...
    def get_I_ext(self, t: float) -> float: return 0.0
    def get_dt(self, t: float) -> float: return self.dt_default
    def is_clamped(self, t: float) -> bool: return False
    def get_clamp_voltage(self, t: float) -> float: raise NotImplementedError
```

**Tier 1 — SteadyStatePacing:**
```python
class SteadyStatePacing(Protocol):
    """BCL-paced protocol. Tier 1."""
    def __init__(self, bcl: float, n_beats: int = 20,
                 stim_amplitude: float = -80.0, stim_duration: float = 1.0,
                 dt_default: float = 0.01):
        duration_ms = bcl * n_beats
        super().__init__(name=f'steady_bcl{int(bcl)}', tier=1,
                         duration_ms=duration_ms, dt_default=dt_default)
        self.bcl = bcl
        self.n_beats = n_beats
        self.stim_amplitude = stim_amplitude
        self.stim_duration = stim_duration
```

**Tier 2 — S1S2Restitution:**
```python
class S1S2Restitution(Protocol):
    """S1 train + single S2 premature beat. Tier 2."""
    def __init__(self, s2_di: float, s1_bcl: float = 1000.0, s1_beats: int = 10,
                 ...):
        # duration_ms = s1_bcl * s1_beats + s2_di + extra_tail
        ...
```

**Tier 3 — DynamicProtocol variants:**
```python
class BCLRamp(Protocol): ...       # linear BCL decrease over N beats
class BurstPacing(Protocol): ...   # fast burst + pause
class AlternansProtocol(Protocol): ...  # constant fast BCL for alternans
```

**ProtocolLibrary — generates all protocols for a tier:**
```python
class ProtocolLibrary:
    @staticmethod
    def tier1() -> List[Protocol]:
        """9 BCLs x 20 beats each."""
        return [SteadyStatePacing(bcl=b, n_beats=20)
                for b in [300, 400, 500, 600, 700, 800, 1000, 1500, 2000]]

    @staticmethod
    def tier2() -> List[Protocol]:
        """S1-S2 at 8 DI values."""
        return [S1S2Restitution(s2_di=di)
                for di in [50, 75, 100, 150, 200, 300, 500, 800]]

    @staticmethod
    def tier3() -> List[Protocol]: ...
```

#### Pseudocode
```python
# SteadyStatePacing.get_I_stim(t):
# NOTE: float modulo can cause edge-case misses at beat boundaries.
# Use tolerance-based comparison instead of exact modulo:
#   beat_phase = t % self.bcl
#   if beat_phase < self.stim_duration or (self.bcl - beat_phase) < 1e-9:
# Alternatively, pre-compute stimulus onset times and use comparison windows.
beat_phase = t % self.bcl
if beat_phase < self.stim_duration:
    return self.stim_amplitude  # e.g., -80.0 (TTP06 convention: negative depolarizes)
return 0.0

# S1S2Restitution.get_I_stim(t):
# First s1_beats beats at s1_bcl, then one S2 beat at s1_bcl*s1_beats + s2_di
s1_end = self.s1_bcl * self.s1_beats
if t < s1_end:
    return SteadyStatePacing.get_I_stim(self, t)  # S1 pacing
s2_onset = s1_end + self.s2_di
if s2_onset <= t < s2_onset + self.stim_duration:
    return self.stim_amplitude
return 0.0

# BCLRamp: linearly interpolate BCL from bcl_start to bcl_end over n_beats
# BurstPacing: n_burst beats at fast BCL, then pause, repeat n_cycles times
# AlternansProtocol: constant fast BCL (e.g., 330ms) for n_beats
```

#### Test Spec
- `test_data_generation.py::test_tier1_protocols` — Generate all 9 Tier 1 protocols. Expected: 9 protocols, BCLs match, each produces valid APs.
- `test_data_generation.py::test_tier2_restitution` — Run S1-S2 at DI=200ms. Expected: S2 APD shorter than S1 APD (restitution).
- `test_data_generation.py::test_tier2_failed_capture` — Run S1-S2 at DI=30ms (below ERP). Expected: S2 does not produce full AP (V_max < -20 mV).
- `test_data_generation.py::test_tier3_alternans` — Run 20 beats at BCL=330ms. Expected: beat-to-beat APD variation > 5ms.

#### Checklist
- [ ] Protocol base class (regular class, not dataclass) with get_I_stim, get_dt, is_clamped interface
- [ ] duration_ms property on all protocols (no n_steps property)
- [ ] Tier 1: SteadyStatePacing for 9 BCLs
- [ ] Tier 2: S1S2Restitution with configurable DI, includes sub-ERP
- [ ] Tier 3: BCLRamp, BurstPacing, AlternansProtocol
- [ ] ProtocolLibrary with tier1(), tier2(), tier3() class methods
- [ ] 4 tests passing

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/test_data_generation.py::test_tier1_protocols tests/test_data_generation.py::test_tier2_restitution tests/test_data_generation.py::test_tier2_failed_capture tests/test_data_generation.py::test_tier3_alternans -v
```

#### Exit Criteria
- [ ] All Tier 1-3 protocols generate physiologically valid traces
- [ ] S1-S2 restitution curve is monotonically increasing (APD vs DI)
- [ ] Failed capture correctly identified at sub-ERP DIs
- [ ] 4 tests pass

#### Risk
Alternans may not appear at BCL=330ms with default TTP06 EPI parameters. Mitigation: try BCL=300-350ms range; if no alternans, adjust BCL or note as TTP06 limitation.

---

### Step 1.4: HDF5 Storage
**Model**: opus

#### Read First
- `Research/Active/surrogate_pipeline/KNOWLEDGE.md:238-287` — storage format specification

#### Why
Store generated traces in HDF5 for archival (float64, full metadata). This is the "source of truth" layer — raw data that can always be reprocessed into training shards.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/data/storage.py`

```python
class TraceStorage:
    """HDF5 storage for generated traces."""

    def __init__(self, base_dir: str = '/media/norepinephrine/Elements-ext4/surrogate_data/raw'):
        self.base_dir = Path(base_dir)
        self._check_storage_mounted()

    def _check_storage_mounted(self):
        """Verify external HDD is mounted before writing.
        Raises RuntimeError if /media/norepinephrine/Elements-ext4/ does not exist
        or is not a mount point.
        """
        mount_point = Path('/media/norepinephrine/Elements-ext4')
        if not mount_point.is_dir():
            raise RuntimeError(
                f"External HDD not mounted at {mount_point}. "
                f"Mount it with: sudo mount /dev/sdX1 {mount_point}")

    def save_trace(self, trace: TraceData, tier: int, protocol_name: str):
        """Append trace to tier HDF5 file."""

    def save_tier(self, traces: List[TraceData], tier: int):
        """Save all traces for a tier."""

    def load_trace(self, tier: int, protocol_name: str) -> TraceData:
        """Load a single trace."""

    def list_protocols(self, tier: int) -> List[str]:
        """List all protocols in a tier file."""
```

**HDF5 structure:**
```
tier01_steady_state.h5
+-- protocol_bcl300/
|   +-- data          (T, 23) float64
|   +-- metadata      {bcl, n_beats, cell_type, dt, ...}
+-- protocol_bcl400/
|   +-- data
|   +-- metadata
+-- ...
```

#### Pseudocode
```python
def save_trace(self, trace, tier, protocol_name):
    path = self.base_dir / f'tier{tier:02d}.h5'
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'a') as f:
        grp = f.create_group(protocol_name)
        grp.create_dataset('data', data=trace.data.cpu().numpy())  # (T, 23) float64
        for k, v in trace.metadata.items():
            grp.attrs[k] = v

def load_trace(self, tier, protocol_name):
    path = self.base_dir / f'tier{tier:02d}.h5'
    with h5py.File(path, 'r') as f:
        data = torch.tensor(f[protocol_name]['data'][:], dtype=torch.float64)
        metadata = dict(f[protocol_name].attrs)
    return TraceData(data=data, metadata=metadata)
```

#### Test Spec
- `test_data_generation.py::test_hdf5_roundtrip` — Save a trace, load it back. Expected: data matches exactly (float64).
- `test_data_generation.py::test_hdf5_metadata` — Save with metadata, load. Expected: metadata fields preserved.

#### Checklist
- [ ] TraceStorage class with save/load/list interface
- [ ] HDF5 file per tier with protocol groups
- [ ] Metadata stored as HDF5 attributes
- [ ] Float64 preservation
- [ ] 23-column traces stored correctly
- [ ] Directory auto-creation
- [ ] HDD mount pre-flight check in constructor
- [ ] 2 tests passing

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/test_data_generation.py::test_hdf5_roundtrip tests/test_data_generation.py::test_hdf5_metadata -v
```

#### Exit Criteria
- [ ] Traces round-trip through HDF5 with exact float64 fidelity
- [ ] Metadata preserved
- [ ] 2 tests pass

#### Risk
h5py not installed — mitigation: `conda install h5py` in heart-conduction env. Check first.

---

### Phase 1 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/test_data_generation.py -v
```

### Phase 1 Exit Criteria
- [ ] 10 tests pass (0 scaffold + 4 generator + 4 protocol + 2 storage)
- [ ] Can generate and store Tier 1-3 data end-to-end
- [ ] Gate state vectors extractable from HDF5 for Phase A autoencoder training
- [ ] Bidomain V1 tests still pass: `cd Bidomain/Engine_V1 && conda run -n heart-conduction python -m pytest tests/ -x -q`

### Phase 1 Cleanup
- [ ] float64 consistency — all generated data is float64 (float32 conversion is Phase 4)
- [ ] No code duplication — TTP06 imported from Bidomain/Engine_V1, not copied
- [ ] All functions have docstrings with parameter types
- [ ] No hardcoded paths — base directories configurable

**-> Commit point: git commit after Phase 1 passes**

---

## Phase 2: Advanced Protocols (Tiers 4-6)

**Goal**: Add random intervals, tissue-mimicking current injection, and voltage clamp protocols. These provide the harder training data needed for Phases C-D.
**Tier**: medium
**Estimated scope**: 3 files, ~600 lines, 8 tests

### Phase Context
- SingleCellGenerator and Protocol base class from Phase 1 are the foundation.
- Tier 5 (injection) requires new current profile generators — these are functions of time, not pacing protocols.
- Tier 6 (voltage clamp) requires the `is_clamped` / `get_clamp_voltage` interface in Protocol.
- **Real tissue I_diff extraction is deferred.** Synthetic profiles (OU noise, ramps, blips, telegraph) are sufficient for initial training. Real tissue profiles from Bidomain V1 I_diff can be added later if synthetic profiles prove insufficient.
- For Tier 4 (random intervals): beat counts vary 5-200 per protocol. This supersedes KNOWLEDGE.md's fixed 50-beat specification — variable lengths provide better training diversity.

### Step 2.1: Random Interval Protocols (Tier 4)
**Model**: opus

#### Read First
- `Research/Active/surrogate_pipeline/KNOWLEDGE.md:155` — Tier 4 spec
- Step 1.3's Protocol base class

#### Why
Random inter-beat intervals test generalization to arbitrary pacing patterns. LogUniform(200, 2000) ms covers the full physiological range. 200 protocols with varying trace lengths (5-200 beats) provide diversity.

#### Implementation Spec
**Files to modify:** `Surrogate/surrogate/data/protocols.py` — add RandomIntervalPacing class

```python
class RandomIntervalPacing(Protocol):
    """Random inter-beat intervals. Tier 4.

    Not a dataclass. Uses regular __init__ to generate random intervals
    upfront and compute duration_ms from their sum.
    Pre-computes cumulative interval sums for O(log n) get_I_stim lookup.
    """
    def __init__(self, n_beats: int, interval_min: float = 200.0,
                 interval_max: float = 2000.0, seed: Optional[int] = None,
                 stim_amplitude: float = -80.0, stim_duration: float = 1.0,
                 dt_default: float = 0.01):
        rng = np.random.RandomState(seed)
        self.intervals = np.exp(rng.uniform(
            np.log(interval_min), np.log(interval_max), n_beats))
        self.cumulative = np.cumsum(self.intervals)  # pre-compute for O(log n) lookup
        duration_ms = float(self.cumulative[-1])
        super().__init__(name=f'random_seed{seed}', tier=4,
                         duration_ms=duration_ms, dt_default=dt_default)
        self.n_beats = n_beats
        self.stim_amplitude = stim_amplitude
        self.stim_duration = stim_duration
```

**Add to ProtocolLibrary:**
```python
@staticmethod
def tier4(n_protocols: int = 200) -> List[Protocol]:
    rng = np.random.RandomState(42)
    return [RandomIntervalPacing(
        n_beats=rng.randint(5, 201), seed=i)
        for i in range(n_protocols)]
```

#### Pseudocode
```python
def get_I_stim(self, t):
    # O(log n) lookup using pre-computed cumulative sums
    # Find which beat we're in via binary search
    # Beat 0 starts at t=0, beat k starts at cumulative[k-1]
    beat_idx = np.searchsorted(self.cumulative, t, side='right')
    if beat_idx >= self.n_beats:
        return 0.0
    beat_start = 0.0 if beat_idx == 0 else float(self.cumulative[beat_idx - 1])
    beat_phase = t - beat_start
    if beat_phase < self.stim_duration:
        return self.stim_amplitude
    return 0.0
```

#### Test Spec
- `test_data_generation.py::test_tier4_random_intervals` — Generate 5 protocols. Expected: each has different intervals, all intervals in [200, 2000] ms.
- `test_data_generation.py::test_tier4_variable_length` — Generate protocols with n_beats=5, 50, 200. Expected: trace lengths proportional to n_beats.

#### Checklist
- [ ] RandomIntervalPacing with LogUniform interval sampling
- [ ] duration_ms computed from sum of intervals (not fixed)
- [ ] Pre-computed cumulative sums in __init__ for O(log n) lookup
- [ ] get_I_stim uses np.searchsorted (binary search), not O(n) loop
- [ ] Reproducible with seed
- [ ] Variable trace lengths (5-200 beats)
- [ ] ProtocolLibrary.tier4() generates 200 protocols
- [ ] 2 tests passing

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/test_data_generation.py::test_tier4_random_intervals tests/test_data_generation.py::test_tier4_variable_length -v
```

#### Exit Criteria
- [ ] 200 random protocols generated with valid AP traces
- [ ] 2 tests pass

#### Risk
Very short intervals (DI < ERP) may cause failed captures mid-trace. This is expected and desirable — the model needs to handle this.

---

### Step 2.2: Tissue-Mimicking Current Injection (Tier 5)
**Model**: opus

#### Read First
- `Research/Active/surrogate_pipeline/KNOWLEDGE.md:157-164` — Tier 5 injection profiles
- SingleCellGenerator's `I_ext` handling from Step 1.2

#### Why
In tissue, cells experience diffusion current from neighbors — smooth, continuous, not sharp pulses. Training on these profiles bridges single-cell training to tissue inference. OU noise is the most realistic synthetic profile.

**Note**: Real tissue I_diff extraction from Bidomain V1 is deferred. Synthetic profiles are sufficient for initial surrogate training. If the surrogate underperforms on tissue inference, real profiles can be added as a follow-up.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/data/injection.py`

```python
class OUNoiseInjection:
    """Ornstein-Uhlenbeck noise current injection."""
    def __init__(self, tau: float, sigma: float, seed: int = 0): ...
    def __call__(self, t: float) -> float: ...

class RampInjection:
    """Smooth depolarizing ramp mimicking approaching wavefront."""
    def __init__(self, peak: float = -30.0, ramp_time: float = 3.0,
                 onset: float = 50.0): ...

class SubThresholdBlips:
    """Random sub-threshold current blips."""
    def __init__(self, amplitude: float = -15.0, duration: float = 2.0,
                 rate: float = 0.01, seed: int = 0): ...

class SustainedOffset:
    """Constant current offset."""
    def __init__(self, amplitude: float = -5.0, start: float = 0.0,
                 end: float = None): ...

class BiphasicPulse:
    """Depolarizing then hyperpolarizing pulse."""
    def __init__(self, depol_amp: float = -20.0, hyperpol_amp: float = 10.0,
                 pulse_duration: float = 3.0, onset: float = 50.0): ...

class RandomTelegraph:
    """Poisson-switching current between 0 and -I_max."""
    def __init__(self, I_max: float = -20.0, rate: float = 5.0,
                 seed: int = 0): ...

class CompositeInjection:
    """Combine multiple injection profiles additively."""
    def __init__(self, *injections): ...
```

**Integration with Protocol:**
```python
class InjectedPacing(Protocol):
    """Standard pacing with additional current injection. Tier 5."""
    def __init__(self, base_protocol: Protocol, injection: Callable, **kwargs):
        super().__init__(name=f'injected_{base_protocol.name}', tier=5,
                         duration_ms=base_protocol.duration_ms,
                         dt_default=base_protocol.dt_default)
        self.base_protocol = base_protocol
        self.injection = injection

    def get_I_stim(self, t: float) -> float:
        return self.base_protocol.get_I_stim(t)

    def get_I_ext(self, t: float) -> float:
        return self.injection(t)
```

#### Pseudocode
```python
# OUNoiseInjection: Ornstein-Uhlenbeck process
# dI = -I/tau * dt_ou + sigma * sqrt(2*dt_ou/tau) * N(0,1)
# Pre-generate full trajectory at construction time for reproducibility
class OUNoiseInjection:
    def __init__(self, tau, sigma, duration_ms, dt_ou=0.01, seed=0):
        rng = np.random.RandomState(seed)
        n = int(duration_ms / dt_ou)
        self.trajectory = np.zeros(n)
        I = 0.0
        for i in range(1, n):
            I += -I/tau * dt_ou + sigma * np.sqrt(2*dt_ou/tau) * rng.randn()
            self.trajectory[i] = I
        self.dt_ou = dt_ou
    def __call__(self, t):
        idx = int(t / self.dt_ou)
        return float(self.trajectory[min(idx, len(self.trajectory)-1)])

# RampInjection: 0 -> peak over ramp_time starting at onset
# SubThresholdBlips: Poisson-timed rectangular pulses
# RandomTelegraph: Poisson-switching between 0 and I_max
```

#### Test Spec
- `test_data_generation.py::test_ou_noise_statistics` — Generate OU noise (tau=5ms, sigma=10uA). Expected: mean ~ 0, std ~ sigma*sqrt(tau/2).
- `test_data_generation.py::test_injection_modifies_ap` — Run same protocol with and without sustained -5uA offset. Expected: different resting Vm and APD.
- `test_data_generation.py::test_subthreshold_no_ap` — Inject sub-threshold blips without pacing. Expected: Vm fluctuates but never exceeds -40mV (no AP).

#### Checklist
- [ ] 6 injection profile classes (OU, ramp, blips, sustained, biphasic, telegraph)
- [ ] CompositeInjection for combining profiles
- [ ] InjectedPacing protocol wrapper
- [ ] ProtocolLibrary.tier5() generates injection protocols
- [ ] 3 tests passing

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/test_data_generation.py::test_ou_noise_statistics tests/test_data_generation.py::test_injection_modifies_ap tests/test_data_generation.py::test_subthreshold_no_ap -v
```

#### Exit Criteria
- [ ] All 6 injection types produce valid current profiles
- [ ] Injection modifies AP dynamics as expected
- [ ] 3 tests pass

#### Risk
OU noise with large sigma might cause numerical instability in TTP06. Mitigation: clip I_ext to physiological range (+/-100 uA/cm^2).

---

### Step 2.3: Voltage Clamp Protocols (Tier 6)
**Model**: opus

#### Read First
- `Research/Active/surrogate_pipeline/KNOWLEDGE.md:166-173` — Tier 6 clamp specs
- SingleCellGenerator's `is_clamped` / `get_clamp_voltage` handling from Step 1.2

#### Why
Voltage clamp provides the cleanest training signal — I_ion errors don't propagate through Vm. Gates converge to exact gate_inf(V_clamp) values. Used in training Phase C with scaffold active for maximum gate supervision.

#### Implementation Spec
**Files to create:** `Surrogate/surrogate/data/clamp.py`

```python
class StepClamp(Protocol):
    """Hold -> step to V_test -> hold. Tier 6."""
    def __init__(self, v_hold=-80.0, v_test=-20.0, hold_time=500.0,
                 test_time=500.0, dt_default=0.01):
        duration_ms = hold_time + test_time
        super().__init__(name=f'step_clamp_{v_test}', tier=6,
                         duration_ms=duration_ms, dt_default=dt_default)
        self.v_hold = v_hold
        self.v_test = v_test
        self.hold_time = hold_time

    def is_clamped(self, t): return True
    def get_clamp_voltage(self, t):
        return self.v_hold if t < self.hold_time else self.v_test

class RampClamp(Protocol):
    """Linear voltage ramp. Tier 6."""
    def __init__(self, v_start=-80.0, v_end=40.0, ramp_duration=300.0,
                 dt_default=0.01):
        super().__init__(name='ramp_clamp', tier=6,
                         duration_ms=ramp_duration, dt_default=dt_default)
        self.v_start = v_start
        self.v_end = v_end

    def is_clamped(self, t): return True

class StaircaseClamp(Protocol):
    """Multi-step voltage staircase. Tier 6."""
    def __init__(self, voltages=None, step_duration=100.0, dt_default=0.01):
        voltages = voltages or [-80, -60, -40, -20, 0, 20, 40]
        duration_ms = step_duration * len(voltages)
        super().__init__(name='staircase_clamp', tier=6,
                         duration_ms=duration_ms, dt_default=dt_default)
        self.voltages = voltages
        self.step_duration = step_duration

    def is_clamped(self, t): return True

class APClamp(Protocol):
    """Play back recorded AP waveform as Vm command. Tier 6."""
    def __init__(self, vm_waveform, dt_waveform=0.01, dt_default=0.01):
        """
        Args:
            vm_waveform: required — 1D array/tensor of Vm values to play back.
                         No default; must be explicitly provided.
            dt_waveform: temporal resolution of vm_waveform (ms)
            dt_default: simulation dt (ms)
        """
        duration_ms = len(vm_waveform) * dt_waveform
        super().__init__(name='ap_clamp', tier=6,
                         duration_ms=duration_ms, dt_default=dt_default)
        self.vm_waveform = vm_waveform
        self.dt_waveform = dt_waveform

    def is_clamped(self, t): return True

class PartialClamp(Protocol):
    """Partially clamped: Vm = alpha*V_cmd + (1-alpha)*V_free. Tier 6.

    Note: is_clamped() returns True. SingleCellGenerator handles partial
    clamp as a special case: instead of fully overriding Vm, it detects
    `hasattr(protocol, 'alpha')` and blends V_cmd with V_free.
    """
    def __init__(self, alpha=0.5, command_protocol=None, dt_default=0.01):
        super().__init__(name=f'partial_clamp_a{alpha}', tier=6,
                         duration_ms=command_protocol.duration_ms,
                         dt_default=dt_default)
        self.alpha = alpha
        self.command_protocol = command_protocol

    def is_clamped(self, t): return True

    def get_clamp_voltage(self, t):
        return self.command_protocol.get_clamp_voltage(t)
```

#### Pseudocode
```python
# StepClamp.get_clamp_voltage(t):
return self.v_hold if t < self.hold_time else self.v_test

# RampClamp.get_clamp_voltage(t):
frac = t / self.duration_ms
return self.v_start + frac * (self.v_end - self.v_start)

# StaircaseClamp.get_clamp_voltage(t):
step_idx = min(int(t / self.step_duration), len(self.voltages) - 1)
return self.voltages[step_idx]

# APClamp.get_clamp_voltage(t):
idx = min(int(t / self.dt_waveform), len(self.vm_waveform) - 1)
return float(self.vm_waveform[idx])

# PartialClamp: handled in SingleCellGenerator.run_protocol():
#   if protocol.is_clamped(t) and hasattr(protocol, 'alpha'):
#       V_cmd = protocol.get_clamp_voltage(t)
#       V_free = model.step(V, states, dt, I_stim=0) -> V_free
#       V = alpha * V_cmd + (1-alpha) * V_free
```

#### Test Spec
- `test_data_generation.py::test_step_clamp_gate_convergence` — Step clamp at V=-20mV for 500ms. Expected: m gate converges to m_inf(-20) within 1%, h and j converge to h_inf(-20).
- `test_data_generation.py::test_ap_clamp_iion` — AP clamp with recorded waveform. Expected: I_ion trace has correct shape (inward during upstroke, outward during repolarization).
- `test_data_generation.py::test_partial_clamp_interpolation` — Partial clamp at alpha=0.5. Expected: Vm is between V_cmd and V_free.

#### Checklist
- [ ] StepClamp, RampClamp, StaircaseClamp, APClamp, PartialClamp
- [ ] All implement is_clamped/get_clamp_voltage interface
- [ ] PartialClamp.is_clamped() returns True; blending handled in SingleCellGenerator
- [ ] All use regular __init__ with super().__init__() (not dataclass)
- [ ] APClamp requires vm_waveform (no default value)
- [ ] APClamp loads from previously generated AP trace
- [ ] ProtocolLibrary.tier6() generates clamp protocols
- [ ] 3 tests passing

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/test_data_generation.py::test_step_clamp_gate_convergence tests/test_data_generation.py::test_ap_clamp_iion tests/test_data_generation.py::test_partial_clamp_interpolation -v
```

#### Exit Criteria
- [ ] Gates converge to gate_inf(V_clamp) under step clamp
- [ ] AP clamp produces valid I_ion traces
- [ ] 3 tests pass

#### Risk
Voltage clamp with TTP06: model.step() applies stimulus AND advances gates. Under clamp, we need to advance gates at clamped Vm but NOT apply the Vm update. Must manually call gate update and concentration update separately, or use step() then override V. Check that model.step() returns correct states when V is externally set.

---

### Phase 2 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/test_data_generation.py -v
```

### Phase 2 Exit Criteria
- [ ] 8 new tests pass (2 + 3 + 3)
- [ ] All Phase 1 tests still pass (18 total: 10 Phase 1 + 8 Phase 2)
- [ ] Tiers 4-6 generate valid traces

### Phase 2 Cleanup
- [ ] float64 consistency
- [ ] No code duplication
- [ ] All injection profiles documented with units

**-> Commit point: git commit after Phase 2 passes**

---

## Phase 3: Stress Protocols + Augmentation (Tiers 7-12)

**Goal**: Add concentration perturbation, long-duration stability, corruption recovery, tissue-specific scenarios, combined stressors with stitching, celltype variants, and variable dt. These provide stress-test data for training Phase D.
**Tier**: medium
**Estimated scope**: 2 files, ~500 lines, 8 tests

### Phase Context
- Tiers 7-12 all build on the SingleCellGenerator and Protocol classes from Phases 1-2.
- Tier 7 has two distinct modification paths:
  - **Conductance scaling** → CellTypeConfig + TTP06Model.from_config() (for Gto, GKs, GNa, etc.). CellTypeConfig holds absolute conductance values and kinetic flags.
  - **Concentration perturbation** → deepcopy model.params and set Ko, Nao, Cao directly on the TTP06Parameters object. These are model-level parameters, NOT CellTypeConfig fields — CellTypeConfig only handles conductances/permeabilities.
- Tier 9 (corruption recovery): deliberately set non-physiological gate states.
- Tier 10 (tissue-specific): combine injection profiles from Tier 5 in specific patterns.
- Tier 11 (stitching): concatenate TraceData from different protocols with rest periods.
- Tier 12 (celltypes): same protocols, different CellType enum.
- Variable dt: Protocol.get_dt(t) can return different values based on simulation phase.

### Step 3.1: Tiers 7-10 (Perturbation + Tissue Scenarios)
**Model**: opus

#### Read First
- `Research/Active/surrogate_pipeline/KNOWLEDGE.md:175-198` — Tiers 7-10 specs
- `Bidomain/Engine_V1/cardiac_sim/ionic/ttp06/parameters.py` — TTP06Parameters class, Ko/Nao/Cao fields
- `Bidomain/Engine_V1/cardiac_sim/ionic/ttp06/celltypes/standard.py` — CellTypeConfig (conductances only, NOT concentrations)
- TTP06 parameters.py for initial state values and valid ranges

#### Why
These tiers cover edge cases: unusual concentrations, extended simulations, corrupted states, and tissue-specific electrotonic loading patterns. Each addresses a specific gap identified in the audit.

#### Implementation Spec
**Files to modify:** `Surrogate/surrogate/data/protocols.py` — add Tier 7-10 protocol classes
**Files to create:** `Surrogate/surrogate/data/augmentation.py` — corruption, conductance scaling, stitching

**Tier 7 — ConcentrationPerturbation:**
K_o, Na_o, Ca_o are model-level parameters (fields of TTP06Parameters), not CellTypeConfig fields. CellTypeConfig only handles conductances/permeabilities. To modify concentrations, deepcopy model.params and set Ko/Nao/Cao directly.
```python
class ConcentrationPerturbation(Protocol):
    """Modify extracellular concentrations. Wraps another protocol. Tier 7.

    K_o, Na_o, Ca_o are fields of TTP06Parameters, NOT CellTypeConfig.
    CellTypeConfig only handles conductances (GNa, GKr, etc.).
    SingleCellGenerator detects this protocol type and modifies
    TTP06Parameters via deepcopy before running.
    """
    def __init__(self, base_protocol: Protocol,
                 Ko: Optional[float] = None,     # default 5.4 mM
                 Nai_init: Optional[float] = None,  # initial Nai state
                 Cai_scale: float = 1.0):
        super().__init__(name=f'conc_Ko{Ko}_{base_protocol.name}', tier=7,
                         duration_ms=base_protocol.duration_ms,
                         dt_default=base_protocol.dt_default)
        self.base_protocol = base_protocol
        self.Ko = Ko            # Model parameter override (TTP06Parameters.Ko)
        self.Nai_init = Nai_init  # Initial state override
        self.Cai_scale = Cai_scale  # Scale initial Cai state

    # Delegate I_stim etc. to base_protocol
```

**SingleCellGenerator handling for ConcentrationPerturbation:**
```python
# In run_protocol():
if isinstance(protocol, ConcentrationPerturbation):
    # Concentration perturbation: modify TTP06Parameters directly (not CellTypeConfig)
    # CellTypeConfig only handles conductances; Ko/Nao/Cao are TTP06Parameters fields
    if protocol.Ko is not None:
        modified_params = copy.deepcopy(self.model.params)
        modified_params.Ko = protocol.Ko
        # Create new model with modified params
        temp_model = TTP06Model(cell_type=self.cell_type_enum, device=self.device)
        temp_model.params = modified_params
        # Use temp_model for this protocol run
    # Modify initial states for Nai, Cai
    if protocol.Nai_init is not None:
        states[StateIndex.Nai] = protocol.Nai_init
    if protocol.Cai_scale != 1.0:
        states[StateIndex.Cai] *= protocol.Cai_scale
    # Run base_protocol with modified model/states
```

**Tier 8 — LongDuration:**
```python
class LongPacing(Protocol): ...         # 200+ beats at constant BCL
class LongQuiescence(Protocol): ...     # 5-30s rest, then stimulus
class LongBlankBurst(Protocol): ...     # long rest -> fast pacing
```

**Tier 9 — Corruption:**
```python
def corrupt_states(states: torch.Tensor, corruption_type: str,
                   severity: float = 0.5, seed: int = 0) -> torch.Tensor:
    """Perturb gate states to non-physiological values."""

class CorruptionRecovery(Protocol):
    """Start from corrupted states, record recovery. Tier 9."""
    def __init__(self, base_protocol: Protocol,
                 corruption_type: str, severity: float = 0.5):
        super().__init__(name=f'corrupt_{corruption_type}', tier=9,
                         duration_ms=base_protocol.duration_ms,
                         dt_default=base_protocol.dt_default,
                         initial_states=None)  # set in run_protocol
        self.base_protocol = base_protocol
        self.corruption_type = corruption_type  # 'random_gates', 'vm_jump', 'extreme_ca'
        self.severity = severity
```

**Tier 10 — Tissue-specific:** Combine injection profiles from Tier 5 in patterns:
```python
class BoundaryCell(InjectedPacing): ...     # reduced injection magnitude
class InfarctBorder(InjectedPacing): ...    # asymmetric injection
class InertInterface(InjectedPacing): ...   # repolarizing sink during plateau
class StimulusSite(InjectedPacing): ...     # sharp pulse + ramp
class SpiralTip(Protocol): ...              # very short DI pacing
```

#### Pseudocode
```python
# ConcentrationPerturbation — handled in SingleCellGenerator.run_protocol():
# NOTE: Ko/Nao/Cao are TTP06Parameters fields, NOT CellTypeConfig fields.
# Use deepcopy of model.params to modify concentrations.
# Conductance scaling uses a different path (CellTypeConfig + from_config).
if isinstance(protocol, ConcentrationPerturbation):
    model_to_use = self._make_modified_model(protocol)  # deep-copy params, set Ko
    states = self.model.get_initial_state(n_cells=1)
    if protocol.Nai_init: states[StateIndex.Nai] = protocol.Nai_init
    if protocol.Cai_scale != 1.0: states[StateIndex.Cai] *= protocol.Cai_scale
    # run loop using model_to_use instead of self.model

# corrupt_states():
if corruption_type == 'random_gates':
    gate_indices = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    for idx in gate_indices:
        states[idx] = torch.rand(1) * severity + states[idx] * (1 - severity)
elif corruption_type == 'extreme_ca':
    states[StateIndex.Cai] *= (1 + 10 * severity)
# ...

# Tier 10 BoundaryCell: InjectedPacing with SustainedOffset(amplitude * 0.5)
# InfarctBorder: InjectedPacing with one-sided injection profile
```

#### Test Spec
- `test_data_generation.py::test_hyperkalemia` — K_o=8.0mM (modified model parameter via deepcopy of TTP06Parameters, not CellTypeConfig). Expected: resting Vm > -80mV (depolarized vs normal -85mV).
- `test_data_generation.py::test_long_quiescence` — 10s rest then stimulus. Expected: normal AP after rest (no drift).
- `test_data_generation.py::test_corruption_recovery` — Set m=0.9 at rest, run 100ms. Expected: m recovers to m_inf(V_rest) ~ 0.0017.
- `test_data_generation.py::test_boundary_cell` — Reduced injection. Expected: APD different from normal pacing.

#### Checklist
- [ ] Tier 7: ConcentrationPerturbation — deepcopy model.params, set Ko/Nao/Cao (TTP06Parameters fields, NOT CellTypeConfig)
- [ ] Tier 7: Initial state overrides for Nai, Cai_scale
- [ ] Tier 8: LongPacing, LongQuiescence, LongBlankBurst
- [ ] Tier 9: corrupt_states() function + CorruptionRecovery protocol
- [ ] Tier 10: BoundaryCell, InfarctBorder, InertInterface, StimulusSite, SpiralTip
- [ ] 4 tests passing

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/test_data_generation.py::test_hyperkalemia tests/test_data_generation.py::test_long_quiescence tests/test_data_generation.py::test_corruption_recovery tests/test_data_generation.py::test_boundary_cell -v
```

#### Exit Criteria
- [ ] All Tier 7-10 protocols generate valid traces
- [ ] K_o perturbation correctly modifies model parameters (TTP06Parameters.Ko, not CellTypeConfig)
- [ ] Corrupted states recover to physiological values
- [ ] 4 tests pass

#### Risk
Long-duration protocols (Tier 8) are slow — 200 beats at BCL=1000ms = 200s simulation at dt=0.01ms = 20M steps. On GPU, each step is fast but total wall time may be minutes. Mitigation: accept the time cost, run in batch.

---

### Step 3.2: Tiers 11-12 + Variable dt + Augmentation
**Model**: opus

#### Read First
- `Research/Active/surrogate_pipeline/KNOWLEDGE.md:200-229` — Tiers 11-12, variable dt, augmentation specs

#### Why
Tier 11 combines multiple stressors and stitches protocols for diversity. Tier 12 runs celltypes. Variable dt enables adaptive timestepping. Conductance scaling is the primary offline augmentation.

#### Implementation Spec
**Files to modify:** `Surrogate/surrogate/data/augmentation.py` — add stitching, conductance scaling
**Files to modify:** `Surrogate/surrogate/data/protocols.py` — add Tier 11-12, variable dt

**Tier 11 — StitchedProtocol:**
StitchedProtocol is NOT a Protocol subclass (it runs multiple protocols sequentially with rest breaks). SingleCellGenerator.run_protocol() detects it via isinstance() and handles it specially.
```python
class StitchedProtocol:
    """Concatenate protocols with rest breaks. Tier 11.

    Not a Protocol subclass — SingleCellGenerator detects this type
    and handles it with sequential protocol execution.
    """
    def __init__(self, protocols: List[Protocol],
                 rest_durations: List[float]):
        """
        Args:
            protocols: list of Protocol objects to run sequentially
            rest_durations: ms between protocols, LogUniform(1000, 30000)
        """
        self.protocols = protocols
        self.rest_durations = rest_durations
        self.name = 'stitched'
        self.tier = 11
        self.duration_ms = sum(p.duration_ms for p in protocols) + sum(rest_durations)

    # SingleCellGenerator.run_protocol() handles this:
    # if isinstance(protocol, StitchedProtocol):
    #     traces = []
    #     for i, sub_proto in enumerate(protocol.protocols):
    #         trace = self._run_single(sub_proto, V, states)  # carry state forward
    #         traces.append(trace)
    #         V, states = <final state from trace>
    #         if i < len(protocol.rest_durations):
    #             rest_trace = self._run_rest(protocol.rest_durations[i], V, states)
    #             traces.append(rest_trace)
    #             V, states = <final state from rest>
    #     return TraceData(data=torch.cat([t.data for t in traces]), ...)
```

**Tier 12 — run existing protocols with different celltypes:**
```python
# No new Protocol class — ProtocolLibrary.tier12() returns Tier 1-3 protocols
# with cell_type='ENDO' and cell_type='M_CELL' (EPI already covered)
```

**Variable dt:**
```python
class AdaptiveDtProtocol(Protocol):
    """Variable dt based on dVm/dt. Wraps another protocol.

    duration_ms inherited from base_protocol. The run loop uses
    `while t < duration_ms` and accumulates time with variable dt.
    Tier is inherited from base_protocol (not hardcoded).
    """
    def __init__(self, base_protocol: Protocol,
                 dt_fast: float = 0.005, dt_slow: float = 0.1,
                 dvdt_threshold: float = 1.0):
        super().__init__(name=f'adaptive_dt_{base_protocol.name}',
                         tier=base_protocol.tier,
                         duration_ms=base_protocol.duration_ms,
                         dt_default=base_protocol.dt_default)
        self.base_protocol = base_protocol
        self.dt_fast = dt_fast
        self.dt_slow = dt_slow
        self.dvdt_threshold = dvdt_threshold
        self._last_dvdt = 0.0  # updated by generator each step

    def get_dt(self, t):
        if abs(self._last_dvdt) > self.dvdt_threshold:
            return self.dt_fast
        return self.dt_slow

    # Delegate other methods to base_protocol
```

**Conductance scaling:**
```python
def generate_scaled_conductances(n_variants: int, seed: int = 0) -> List[Dict[str, float]]:
    """Generate random conductance scaling dicts.
    Each dict maps conductance parameter name -> scale factor ~ U(0.5, 2.0).

    These are SCALE FACTORS (relative), not absolute values.
    SingleCellGenerator._make_scaled_config() converts them to absolute values
    by looking up base conductances from get_celltype_parameters() for the
    selected cell type and multiplying.

    These dicts are passed to SingleCellGenerator(conductance_scaling=...) which
    creates a CellTypeConfig with the absolute scaled values and instantiates
    TTP06Model via TTP06Model.from_config().

    Example output: {'GKr': 0.73, 'GKs': 1.45, 'GNa': 1.12, ...}
    """
```

**Multi-dt sweep utility:**
```python
def generate_multi_dt(protocol: Protocol, dt_values: List[float],
                      generator: SingleCellGenerator) -> List[TraceData]:
    """Run the same protocol at each dt value. Returns one TraceData per dt.

    Used for training dt-invariance: model sees identical dynamics at different
    temporal resolutions.

    Args:
        protocol: base protocol to run
        dt_values: e.g., [0.005, 0.01, 0.02, 0.05, 0.1]
        generator: SingleCellGenerator instance
    Returns:
        List of TraceData, one per dt value
    """
    results = []
    for dt in dt_values:
        p = copy.deepcopy(protocol)
        p.dt_default = dt
        results.append(generator.run_protocol(p))
    return results
```

#### Pseudocode
```python
# StitchedProtocol execution in SingleCellGenerator:
if isinstance(protocol, StitchedProtocol):
    all_records = []
    V, states = self._get_initial(protocol)
    for i, sub_proto in enumerate(protocol.protocols):
        trace = self._run_sub_protocol(sub_proto, V, states)
        all_records.append(trace.data)
        V, states = self._extract_final_state(trace)
        if i < len(protocol.rest_durations):
            rest_trace = self._run_quiescent(protocol.rest_durations[i], V, states)
            all_records.append(rest_trace.data)
            V, states = self._extract_final_state(rest_trace)
    return TraceData(data=torch.cat(all_records, dim=0), ...)

# AdaptiveDtProtocol: generator updates _last_dvdt each step
# In run_protocol loop:
    dt = protocol.get_dt(t)
    # ... run step ...
    protocol._last_dvdt = (V_new - V) / dt  # update for next step's dt decision

# generate_multi_dt:
for dt in dt_values:
    modified = copy.deepcopy(protocol)
    modified.dt_default = dt
    traces.append(generator.run_protocol(modified))

# generate_scaled_conductances:
conductance_names = ['GNa', 'Gto', 'GKr', 'GKs', 'GK1', 'GpCa', 'GpK', 'GbNa', 'GbCa']
for i in range(n_variants):
    rng = np.random.RandomState(seed + i)
    scaling = {name: float(rng.uniform(0.5, 2.0)) for name in conductance_names}
    results.append(scaling)
```

#### Test Spec
- `test_data_generation.py::test_stitched_protocol` — Stitch 3 protocols with 2s breaks. Expected: trace contains 3 AP segments separated by rest periods.
- `test_data_generation.py::test_celltype_apd_differs` — Run BCL=1000 for EPI, ENDO, M_CELL. Expected: M_CELL APD > EPI APD > ENDO APD.
- `test_data_generation.py::test_variable_dt` — Run with adaptive dt. Expected: dt column has different values, small dt during upstroke.
- `test_data_generation.py::test_conductance_scaling` — Scale g_Kr by 0.5. Expected: APD prolongation (reduced repolarization current).

#### Checklist
- [ ] Tier 11: StitchedProtocol (not a Protocol subclass, detected via isinstance)
- [ ] Tier 11: CombinedStressor protocol (pacing + injection + concentration)
- [ ] Tier 12: celltypes via ProtocolLibrary.tier12()
- [ ] AdaptiveDtProtocol with dVm/dt-based switching (time-based loop, no n_steps); tier inherited from base_protocol
- [ ] generate_scaled_conductances() returns scale factors; SingleCellGenerator._make_scaled_config() converts to absolute values via base conductance lookup
- [ ] generate_multi_dt() utility for dt sweep
- [ ] ProtocolLibrary.tier11(), tier12() methods
- [ ] 4 tests passing

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/test_data_generation.py::test_stitched_protocol tests/test_data_generation.py::test_celltype_apd_differs tests/test_data_generation.py::test_variable_dt tests/test_data_generation.py::test_conductance_scaling -v
```

#### Exit Criteria
- [ ] Stitched protocols produce valid multi-segment traces
- [ ] All 3 celltypes produce distinct AP morphologies
- [ ] Variable dt works without numerical instability
- [ ] Conductance scaling modifies AP as expected
- [ ] 4 tests pass

#### Risk
Variable dt with TTP06: very large dt (0.1ms) during upstroke would miss fast Na dynamics. The adaptive switching must correctly detect upstroke onset. Mitigation: use dVm/dt threshold from the PREVIOUS step, and clamp dt_max to 0.1ms (still 10x the normal dt=0.01ms).

---

### Phase 3 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/test_data_generation.py -v
```

### Phase 3 Exit Criteria
- [ ] 8 new tests pass (4 + 4)
- [ ] All Phase 1-2 tests still pass (26 total)
- [ ] All 12 tiers produce valid data
- [ ] Variable dt and conductance scaling work

### Phase 3 Cleanup
- [ ] float64 consistency
- [ ] All protocol classes documented with tier number and purpose
- [ ] No duplicate protocol definitions

**-> Commit point: git commit after Phase 3 passes**

---

## Phase 4: Shard Processor + Full Pipeline Validation

**Goal**: Build the .pt shard pre-processor that converts HDF5 -> training-ready float32 shards. Validate the full end-to-end pipeline: generate -> store -> convert -> load.
**Tier**: medium
**Estimated scope**: 1 file, ~300 lines, 6 tests

### Phase Context
- HDF5 files from Phase 1-3 contain float64 traces with full metadata.
- .pt shards are float32 tensors: `(N_segments, segment_length, 23)`, pre-shuffled, ~200MB each.
- Segment lengths: 100, 500, 1000, 5000 steps (matching training rollout curriculum).
- Train/val split by PROTOCOL (unseen pacing patterns), not by timestep.
- All traces already have 23 columns (column 22 = clamp_mask). No special handling needed — clamp_mask flows through naturally.
- **50% overlap note**: Segment extraction uses 50% overlap (stride = segment_length / 2), which approximately doubles the number of segments compared to non-overlapping extraction. Account for this 2x inflation in storage estimates. Estimated total: ~1.1TB raw HDF5 + ~1.1TB .pt shards (2.2TB total, fits within 5.2TB HDD).

### Step 4.1: Shard Pre-Processor
**Model**: opus

#### Read First
- `Research/Active/surrogate_pipeline/KNOWLEDGE.md:238-287` — shard format specification

#### Why
.pt shards are the training-speed layer. Native PyTorch tensors load directly to GPU with zero parsing overhead. Pre-chunking into fixed-length segments eliminates windowing during training. Pre-shuffling eliminates DataLoader shuffling overhead.

#### Implementation Spec
**Files to modify:** `Surrogate/surrogate/data/storage.py` — add ShardProcessor class

```python
class ShardProcessor:
    """Convert HDF5 raw traces to .pt training shards."""

    def __init__(self, raw_dir: str = '/media/norepinephrine/Elements-ext4/surrogate_data/raw',
                 shard_dir: str = '/media/norepinephrine/Elements-ext4/surrogate_data/train',
                 segment_length: int = 1000,
                 shard_size_mb: float = 200.0):
        ...

    def process_tier(self, tier: int):
        """Convert all traces in a tier to segments, add to shards."""

    def process_all(self, tiers: List[int] = None,
                    val_protocols: List[str] = None):
        """Full pipeline: HDF5 -> segments -> shuffled shards.
        val_protocols: list of protocol names held out for validation."""

    def _extract_segments(self, trace: TraceData,
                         segment_length: int) -> List[torch.Tensor]:
        """Extract overlapping segments from a trace.
        Uses 50% overlap (stride = segment_length // 2).
        Note: this approximately doubles the segment count vs non-overlapping."""

    def _write_shard(self, segments: List[torch.Tensor],
                     shard_path: Path):
        """Stack segments, convert to float32, save as .pt."""
```

**Segment format (23 columns — same as generation):**
```
Column 0:  Vm
Column 1:  I_stim (sign-flipped: positive = depolarizing)
Column 2:  dt
Column 3-20: 18 ionic states
Column 21: I_ion
Column 22: clamp_mask (0.0 = free-running, 1.0 = clamped)

Total: 23 columns
```

#### Pseudocode
```python
def process_tier(self, tier):
    storage = TraceStorage(self.raw_dir)
    protocols = storage.list_protocols(tier)
    all_segments = []
    for proto_name in protocols:
        trace = storage.load_trace(tier, proto_name)
        segments = self._extract_segments(trace, self.segment_length)
        all_segments.extend(segments)
    # Shuffle
    random.shuffle(all_segments)
    # Write shards
    shard_idx = 0
    for batch in chunk(all_segments, self._segments_per_shard):
        tensor = torch.stack(batch).to(torch.float32)  # (N, seg_len, 23)
        torch.save(tensor, self.shard_dir / f'shard_{shard_idx:04d}.pt')
        shard_idx += 1

def _extract_segments(self, trace, segment_length):
    data = trace.data  # (T, 23)
    segments = []
    stride = segment_length // 2  # 50% overlap — ~2x segment count vs non-overlapping
    for start in range(0, len(data) - segment_length + 1, stride):
        segments.append(data[start:start + segment_length])
    return segments
```

#### Test Spec
- `test_data_generation.py::test_shard_float32` — Process a Tier 1 trace. Expected: shard tensor dtype is float32.
- `test_data_generation.py::test_shard_segment_shape` — Expected: each segment is (segment_length, 23).
- `test_data_generation.py::test_shard_roundtrip_accuracy` — Load shard, compare to original HDF5. Expected: values match within float32 precision.
- `test_data_generation.py::test_train_val_split` — Hold out one protocol. Expected: it appears in val/ not train/.

#### Checklist
- [ ] ShardProcessor with process_tier() and process_all()
- [ ] Float64 -> float32 conversion
- [ ] Segment extraction with configurable length
- [ ] 50% overlap with stride = segment_length // 2 (note: ~2x dataset inflation)
- [ ] 23-column segments (clamp_mask included from source)
- [ ] Train/val split by protocol name
- [ ] Shard size targeting (~200MB)
- [ ] Pre-shuffling within shards
- [ ] 4 tests passing

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/test_data_generation.py::test_shard_float32 tests/test_data_generation.py::test_shard_segment_shape tests/test_data_generation.py::test_shard_roundtrip_accuracy tests/test_data_generation.py::test_train_val_split -v
```

#### Exit Criteria
- [ ] Shards are float32 .pt files
- [ ] Segments have correct shape (seg_len, 23)
- [ ] Train/val split works
- [ ] 4 tests pass

#### Risk
Memory during shard creation — loading all segments for a tier into memory before writing. Mitigation: stream segments to disk, shuffle shard files after (not in-memory).

---

### Step 4.2: End-to-End Pipeline Test
**Model**: opus

#### Read First
- All previous steps

#### Why
Validate the full pipeline: protocol definition -> TTP06 execution -> HDF5 storage -> shard conversion -> GPU loading. This is the integration test that proves the data generation system works end-to-end.

#### Implementation Spec
**Files to modify:** `Surrogate/tests/test_data_generation.py` — add integration tests

#### Pseudocode
```python
def test_full_pipeline_tier1():
    # Generate
    gen = SingleCellGenerator(cell_type='EPI')
    protocols = ProtocolLibrary.tier1()[:2]  # just 2 for speed
    traces = [gen.run_protocol(p) for p in protocols]
    # Store
    storage = TraceStorage(tmp_dir)
    for t, p in zip(traces, protocols):
        storage.save_trace(t, tier=1, protocol_name=p.name)
    # Shard
    processor = ShardProcessor(raw_dir=tmp_dir, shard_dir=shard_dir, segment_length=1000)
    processor.process_tier(1)
    # Load to GPU
    shard = torch.load(shard_dir / 'shard_0000.pt', map_location='cuda')
    assert shard.dtype == torch.float32
    assert shard.shape[-1] == 23

def test_ap_shape_validation():
    gen = SingleCellGenerator(cell_type='EPI')
    trace = gen.run_protocol(SteadyStatePacing(bcl=1000, n_beats=2))
    data = trace.data
    assert data[:, TraceData.VM].min() < -80   # V_rest
    assert data[:, TraceData.VM].max() > 0      # V_max
    # Check APD, dVm/dt_max...
```

#### Test Spec
- `test_data_generation.py::test_full_pipeline_tier1` — Generate Tier 1, store HDF5, convert to shards, load shard to GPU. Expected: shard loads, data is valid float32, shapes correct.
- `test_data_generation.py::test_ap_shape_validation` — Check generated APs have: V_rest < -80mV, V_max > 0mV, APD between 150-500ms, dVm/dt_max > 100 mV/ms.

#### Checklist
- [ ] Full pipeline integration test (generate -> store -> shard -> GPU)
- [ ] AP shape validation (V_rest, V_max, APD, dVm/dt_max)
- [ ] 2 tests passing

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/test_data_generation.py -v
```

#### Exit Criteria
- [ ] Full pipeline runs end-to-end without error
- [ ] Generated data passes physiological validation
- [ ] 2 tests pass

#### Risk
GPU memory for loading shards — should be fine for single shard (~200MB). Test with `map_location='cuda'`.

---

### Phase 4 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Surrogate && conda run -n heart-conduction python -m pytest tests/test_data_generation.py -v
# Also check Bidomain tests still pass:
cd /home/norepinephrine/Documents/Heart-Conduction/Bidomain/Engine_V1 && conda run -n heart-conduction python -m pytest tests/ -x -q
```

### Phase 4 Exit Criteria
- [ ] All 32 tests pass (10 + 8 + 8 + 6)
- [ ] Full pipeline: protocol -> TTP06 -> HDF5 -> .pt shard -> GPU
- [ ] Generated APs pass physiological validation
- [ ] Bidomain V1 tests pass (no regressions)

### Phase 4 Cleanup
- [ ] float64 consistency in generation (float32 only in shards)
- [ ] Remove any temp files created during tests
- [ ] All files have module-level docstrings

**-> Commit point: git commit after Phase 4 passes**

---

## Final Cleanup

- [ ] float64 consistency — generation layer is float64, shard layer is float32, no leaks
- [ ] V5.3 not modified
- [ ] No code duplication — TTP06 imported from Bidomain/Engine_V1, not copied
- [ ] All 32 tests pass
- [ ] Surrogate PROGRESS.md updated with data generation status
- [ ] Revert bottom tmux pane from PLAN.md to WHITEBOARD.md:
```bash
tmux send-keys -t 2 C-c
sleep 0.3
tmux send-keys -t 2 'W=$(tput cols); H=""; while true; do N=$(md5sum Research/Active/surrogate_pipeline/WHITEBOARD.md 2>/dev/null | cut -d" " -f1); if [ "$N" != "$H" ]; then clear; glow -s .glow-style.json -w $W Research/Active/surrogate_pipeline/WHITEBOARD.md 2>/dev/null; H=$N; fi; sleep 1; done' Enter
```

## Mutation Log
- 2026-03-21: Audit fixes applied (18 items). Key changes: test counts corrected (32 total, not 36), all traces 23 columns (clamp_mask always present), time-based run loop (no n_steps), Protocol is regular class (not dataclass), I_stim sign convention documented (negate for surrogate formula), K_o perturbation via model re-instantiation, StitchedProtocol handled via isinstance, conductance scaling via CellTypeConfig, generate_multi_dt utility added, KNOWLEDGE.md line references corrected.
- 2026-03-21: Second audit fixes (16 items). Changes by severity:
  - CRITICAL (3): CellTypeConfig constructor fixed — removed `base_cell_type` field (not a CellTypeConfig param), added `_make_scaled_config()` helper to convert scale factors to absolute conductance values via base lookup from `get_celltype_parameters()`, pass config to `TTP06Model.from_config(config, base_cell_type=...)`. I_stim sign convention clarified with worked numerical example showing exact sign math. K_o vs conductance paths clearly separated — concentrations via `deepcopy(model.params)`, conductances via CellTypeConfig.
  - HIGH (4): Phase 2 running total "(18 total)" disambiguated to "(18 total: 10 Phase 1 + 8 Phase 2)". RandomIntervalPacing.get_I_stim fixed from O(n) loop to O(log n) via pre-computed cumulative sums + `np.searchsorted`. PartialClamp.is_clamped() now returns True with note about SingleCellGenerator special-case handling. Step 1.1 Test Spec added formal named test.
  - MEDIUM (6): APClamp vm_waveform default removed — now required parameter. AdaptiveDtProtocol tier inherits from base_protocol instead of hardcoding 0. 50% overlap note added with 2x dataset inflation and updated storage estimate. No `__init__.py` for tests/ added to Step 1.1 checklist. Conductance scaling helper `_make_scaled_config()` with base lookup added to pseudocode. HDD pre-flight check added to TraceStorage constructor.
  - LOW (3): Tier 2 DI values `[50, 75, 100, 150, 200, 300, 500, 800]` added to ProtocolLibrary.tier2() spec. Float modulo edge case note added to SteadyStatePacing pseudocode. STATES_END documented as "(exclusive end, Python convention)".
