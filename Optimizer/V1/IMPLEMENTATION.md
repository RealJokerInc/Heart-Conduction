# Engine Tuner V1 — Implementation Plan

**Scope**: Monodomain V5.4 only. Validate that the BayesOpt pipeline works end-to-end on GPU.

**Engine config**: FDM discretization, Godunov splitting, Rush-Larsen ionic, ForwardEuler diffusion.

---

## Implementation Phases

| Phase | Deliverable | Validates |
|-------|-------------|-----------|
| **I. Foundation** | Config dataclasses, AP metrics, single-cell runner | Can run TTP06 single-cell and measure APD/restitution |
| **Ib. Baseline** | Run default TTP06/epi, measure all targets, quantify gap | Setup works, gap to hiPSC-CM targets is known |
| **II. Cell Fitter** | BayesOpt loop over ionic conductances | Single-cell targets met (APD, restitution) |
| **III. Tissue Fitter** | CV measurement, BayesOpt over D_L/D_T | Tissue CV targets met |
| **IV. Joint Refinement** | GP emulator, NSGA-II on surrogate | Tissue APD + CV + restitution jointly optimized |
| **V. Validation** | Automated test suite | Tuned params generalize to novel protocols |
| **VI. Integration** | CLI entry point, save/load results | Full pipeline runnable as single command |

---

## Tunable Parameter Tiers

The user selects a tier (1, 2, or 3). Higher tiers include all lower-tier parameters. Default is Tier 2. Alternatively, run Sobol sensitivity pre-screen to auto-select.

### TTP06 Parameter Tiers

```
TIER 1 — Core (6 params)
Always tune. Controls the dominant features of AP shape and CV.
┌────────────┬─────────────┬────────────────────────────────────────────┐
│ Parameter  │ Published   │ Controls                                   │
├────────────┼─────────────┼────────────────────────────────────────────┤
│ GNa        │ 14.838      │ Upstroke velocity (dvdt_max), CV           │
│ PCa (GCaL) │ 3.98e-5     │ L-type Ca²⁺, plateau height/duration, APD │
│ GKr        │ 0.153       │ Rapid delayed rectifier, phase 3 repol     │
│ GKs        │ 0.392       │ Slow delayed rectifier, rate-dependent APD │
│ GK1        │ 5.405       │ Inward rectifier, resting Vm, terminal rep │
│ Gto        │ 0.294       │ Transient outward, phase 1 notch           │
└────────────┴─────────────┴────────────────────────────────────────────┘
Bounds: [0.3, 3.0] for all (scaling factor relative to published)
GP training sims needed: ~600 (100 per param)

TIER 2 — Extended (+5 = 11 params)
Add when fitting rate-dependent behavior, Ca²⁺ transient shape, or
pump/exchanger contributions. RECOMMENDED DEFAULT.
┌────────────┬─────────────┬────────────────────────────────────────────┐
│ Parameter  │ Published   │ Controls                                   │
├────────────┼─────────────┼────────────────────────────────────────────┤
│ KNaCa      │ 1000.0      │ Na/Ca exchanger rate, late plateau, Ca²⁺   │
│ PNaK       │ 2.724       │ Na/K pump, resting Vm (electrogenic 3:2)   │
│ GpCa       │ 0.1238      │ Sarcolemmal Ca²⁺ pump                      │
│ Vmax_up    │ 0.006375    │ SERCA uptake rate, Ca²⁺ transient decay    │
│ GpK        │ 0.0146      │ Plateau potassium, late repolarization     │
└────────────┴─────────────┴────────────────────────────────────────────┘
Bounds: [0.3, 3.0] for all
GP training sims needed: ~1100 (100 per param)

TIER 3 — Full (+6 = 17 params)
Use only if Tier 2 fails to match targets, or if matching calcium
dynamics or very specific AP morphology features.
┌────────────┬─────────────┬────────────────────────────────────────────┐
│ Parameter  │ Published   │ Controls                                   │
├────────────┼─────────────┼────────────────────────────────────────────┤
│ GbNa       │ 0.00029     │ Background Na⁺ leak                        │
│ GbCa       │ 0.000592    │ Background Ca²⁺ leak                       │
│ Vrel       │ 0.102       │ RyR release rate, Ca²⁺ spark amplitude     │
│ Vleak      │ 0.00036     │ SR leak rate, diastolic Ca²⁺               │
│ Vxfer      │ 0.0038      │ Subspace→cytosol Ca²⁺ transfer rate        │
│ Kup        │ 0.00025     │ SERCA affinity (Km), Ca²⁺ sensitivity      │
└────────────┴─────────────┴────────────────────────────────────────────┘
Bounds: [0.1, 5.0] (wider range for small currents)
GP training sims needed: ~1700 (100 per param)
```

### ORd Parameter Tiers

```
TIER 1 — Core (8 params)
ORd has more major currents than TTP06 (INaL, IKb are significant).
┌────────────┬─────────────┬────────────────────────────────────────────┐
│ Parameter  │ Published   │ Controls                                   │
├────────────┼─────────────┼────────────────────────────────────────────┤
│ GNa        │ 75.0        │ Fast sodium, upstroke, CV                  │
│ GNaL       │ 0.0075      │ Late sodium, plateau duration (major in ORd│
│ PCa        │ 0.0001      │ L-type Ca²⁺ permeability, plateau, APD     │
│ GKr        │ 0.046       │ Rapid delayed rectifier, phase 3 repol     │
│ GKs        │ 0.0034      │ Slow delayed rectifier, rate-dependent APD │
│ GK1        │ 0.1908      │ Inward rectifier, resting Vm               │
│ Gto        │ 0.02        │ Transient outward, phase 1 notch           │
│ GKb        │ 0.003       │ Background potassium, plateau balance      │
└────────────┴─────────────┴────────────────────────────────────────────┘
Bounds: [0.3, 3.0] for all
GP training sims needed: ~800

TIER 2 — Extended (+7 = 15 params)
Pumps, exchangers, and calcium handling. RECOMMENDED DEFAULT.
┌────────────┬─────────────┬────────────────────────────────────────────┐
│ Parameter  │ Published   │ Controls                                   │
├────────────┼─────────────┼────────────────────────────────────────────┤
│ Gncx       │ 0.0008      │ Na/Ca exchanger, late plateau, Ca²⁺ balance│
│ Pnak       │ 30.0        │ Na/K pump, resting Vm                      │
│ GpCa       │ 0.0005      │ Sarcolemmal Ca²⁺ pump                      │
│ Jup_max    │ 0.004375    │ SERCA uptake, Ca²⁺ transient decay         │
│ a_rel      │ 0.5         │ RyR release amplitude                      │
│ PNab       │ 3.75e-10    │ Background Na⁺ permeability                │
│ PCab       │ 2.5e-8      │ Background Ca²⁺ permeability               │
└────────────┴─────────────┴────────────────────────────────────────────┘
Bounds: [0.3, 3.0] for conductances, [0.1, 5.0] for Ca²⁺ handling
GP training sims needed: ~1500

TIER 3 — Full (+10 = 25 params)
CaMKII modulation, SR dynamics, diffusion time constants.
Use for rate-dependent alternans fitting or phosphorylation studies.
┌────────────┬─────────────┬────────────────────────────────────────────┐
│ Parameter  │ Published   │ Controls                                   │
├────────────┼─────────────┼────────────────────────────────────────────┤
│ CaMKo      │ 0.05        │ Total CaMKII, phosphorylation magnitude   │
│ KmCaM      │ 0.0015      │ CaM affinity, CaMKII activation threshold │
│ aCaMK      │ 0.05        │ CaMKII trapping rate                       │
│ bCaMK      │ 0.00068     │ CaMKII release rate                        │
│ bt         │ 4.75        │ SR release time constant                   │
│ tau_tr      │ 100.0       │ NSR→JSR Ca²⁺ transfer time                │
│ tau_diff_Ca │ 0.2         │ Subspace Ca²⁺ diffusion time              │
│ tau_diff_Na │ 2.0         │ Subspace Na⁺ diffusion time               │
│ tau_diff_K  │ 2.0         │ Subspace K⁺ diffusion time                │
│ Kmup       │ 0.00092     │ SERCA Ca²⁺ affinity                       │
└────────────┴─────────────┴────────────────────────────────────────────┘
Bounds: [0.1, 5.0] for all
GP training sims needed: ~2500
```

### Tier Selection Guide

```
Use Tier 1 when:
  → Quick calibration for CV + APD only
  → Testing the pipeline on a new target
  → Computational budget is limited

Use Tier 2 when (RECOMMENDED):
  → Matching restitution curve shape
  → Fitting to experimental AP traces
  → Need Ca²⁺ transient to be physiological
  → Want rate-dependent APD to be accurate

Use Tier 3 when:
  → Tier 2 fails to match targets (> 5% error)
  → Fitting alternans threshold or CaMKII effects (ORd)
  → Matching very specific AP morphology features
  → Reproducing drug block experiments

Use Sobol pre-screen when:
  → Unsure which tier to use
  → Working with modified/custom cell types
  → Want data-driven parameter selection
```

### Sobol Sensitivity Pre-Screen (Optional)

Before choosing a tier, run a global sensitivity analysis to identify which parameters
actually matter for the specific targets:

```python
# ~9000 single-cell evals for 17 params (TTP06), ~45 min on GPU
# ~13000 single-cell evals for 25 params (ORd), ~1.5 hr on GPU
from SALib.sample import saltelli
from SALib.analyze import sobol

# Result: Sobol total-order index S_T per parameter per output
# Keep params with S_T > 0.01 for any target (APD, restit, dvdt_max)
# Typically reduces 17 → 8-12 (TTP06) or 25 → 10-15 (ORd)
```

### Impact on Pipeline Cost

| Tier | Ionic Params | + Tissue | Total Dims | Training Sims | Phase 3 GPU Time |
|------|-------------|----------|-----------|---------------|------------------|
| 1    | 6 (TTP06) / 8 (ORd) | 2 | 8-10 | 500-800 | ~3-4 hr |
| 2    | 11 (TTP06) / 15 (ORd) | 2 | 13-17 | 1100-1500 | ~8-12 hr |
| 3    | 17 (TTP06) / 25 (ORd) | 2 | 19-27 | 1700-2500 | ~14-20 hr |
| Sobol | auto-selected | 2 | varies | ~100 per dim | varies |

---

## Phase I: Foundation

### I-1: Config dataclasses (`tuner/config.py`)

```python
from enum import IntEnum

class ParamTier(IntEnum):
    TIER_1 = 1   # Core conductances only
    TIER_2 = 2   # + pumps, exchangers, SERCA (recommended)
    TIER_3 = 3   # + background currents, SR dynamics, CaMKII


# Parameter registry: maps param name → (attribute_path, published_value, tier)
TTP06_PARAMS = {
    # Tier 1 — Core
    'GNa':     ('GNa',     14.838,    1),
    'PCa':     ('PCa',     3.98e-5,   1),
    'GKr':     ('GKr',     0.153,     1),
    'GKs':     ('GKs',     0.392,     1),
    'GK1':     ('GK1',     5.405,     1),
    'Gto':     ('Gto',     0.294,     1),
    # Tier 2 — Extended
    'KNaCa':   ('KNaCa',   1000.0,    2),
    'PNaK':    ('PNaK',    2.724,     2),
    'GpCa':    ('GpCa',    0.1238,    2),
    'Vmax_up': ('Vmax_up', 0.006375,  2),
    'GpK':     ('GpK',     0.0146,    2),
    # Tier 3 — Full
    'GbNa':    ('GbNa',    0.00029,   3),
    'GbCa':    ('GbCa',    0.000592,  3),
    'Vrel':    ('Vrel',    0.102,     3),
    'Vleak':   ('Vleak',   0.00036,   3),
    'Vxfer':   ('Vxfer',   0.0038,    3),
    'Kup':     ('Kup',     0.00025,   3),
}

ORD_PARAMS = {
    # Tier 1 — Core
    'GNa':     ('GNa',     75.0,      1),
    'GNaL':    ('GNaL',    0.0075,    1),
    'PCa':     ('PCa',     0.0001,    1),
    'GKr':     ('GKr',     0.046,     1),
    'GKs':     ('GKs',     0.0034,    1),
    'GK1':     ('GK1',     0.1908,    1),
    'Gto':     ('Gto',     0.02,      1),
    'GKb':     ('GKb',     0.003,     1),
    # Tier 2 — Extended
    'Gncx':    ('Gncx',    0.0008,    2),
    'Pnak':    ('Pnak',    30.0,      2),
    'GpCa':    ('GpCa',    0.0005,    2),
    'Jup_max': ('Jup_max', 0.004375,  2),
    'a_rel':   ('a_rel',   0.5,       2),
    'PNab':    ('PNab',    3.75e-10,  2),
    'PCab':    ('PCab',    2.5e-8,    2),
    # Tier 3 — Full
    'CaMKo':   ('CaMKo',   0.05,      3),
    'KmCaM':   ('KmCaM',   0.0015,    3),
    'aCaMK':   ('aCaMK',   0.05,      3),
    'bCaMK':   ('bCaMK',   0.00068,   3),
    'bt':      ('bt',      4.75,      3),
    'tau_tr':  ('tau_tr',  100.0,     3),
    'tau_diff_Ca': ('tau_diff_Ca', 0.2, 3),
    'tau_diff_Na': ('tau_diff_Na', 2.0, 3),
    'tau_diff_K':  ('tau_diff_K',  2.0, 3),
    'Kmup':    ('Kmup',    0.00092,   3),
}

def get_param_names(model: str, tier: int) -> list[str]:
    """Return parameter names up to the given tier."""
    registry = TTP06_PARAMS if model == 'ttp06' else ORD_PARAMS
    return [name for name, (_, _, t) in registry.items() if t <= tier]

# Per-parameter bounds (scaling factor relative to published).
# Tuned for hiPSC-CM targets: low GNa (slow upstroke), low GK1 (depolarized V_rest).
# Bounds must be wide enough to reach hiPSC-CM regime from adult published values.
# Per-parameter bounds (scaling factor relative to published).
# Calibrated for matured hiPSC-CM targets (CV=25, APD=250, dvdt=150, V_rest=-85).
# All scalings stay within [0.3, 2.0] — no extreme perturbation of TTP06.
TTP06_BOUNDS = {
    # Tier 1
    'GNa':     (0.3,  1.5),   # 0.5 gives dvdt_max ~150 V/s at V_rest=-85
    'PCa':     (0.3,  2.0),   # Plateau height/duration → APD
    'GKr':     (0.3,  2.5),   # May need >1.0 for shorter APD (250 vs 280)
    'GKs':     (0.3,  2.5),   # Rate-dependent APD
    'GK1':     (0.5,  1.5),   # V_rest stays near -85 → narrow range
    'Gto':     (0.3,  2.0),   # Phase 1 notch
    # Tier 2
    'KNaCa':   (0.3,  2.5),
    'PNaK':    (0.5,  2.0),
    'GpCa':    (0.3,  2.5),
    'Vmax_up': (0.3,  2.5),
    'GpK':     (0.3,  3.0),
    # Tier 3
    'GbNa':    (0.2,  3.0),
    'GbCa':    (0.2,  3.0),
    'Vrel':    (0.2,  3.0),
    'Vleak':   (0.2,  3.0),
    'Vxfer':   (0.2,  3.0),
    'Kup':     (0.2,  3.0),
}

ORD_BOUNDS = {
    # Tier 1
    'GNa':     (0.3,  1.5),
    'GNaL':    (0.3,  3.0),   # Late Na is variable
    'PCa':     (0.3,  2.0),
    'GKr':     (0.3,  2.5),
    'GKs':     (0.3,  2.5),
    'GK1':     (0.5,  1.5),
    'Gto':     (0.3,  2.0),
    'GKb':     (0.3,  3.0),
    # Tier 2
    'Gncx':    (0.3,  2.5),
    'Pnak':    (0.5,  2.0),
    'GpCa':    (0.3,  2.5),
    'Jup_max': (0.3,  2.5),
    'a_rel':   (0.2,  3.0),
    'PNab':    (0.2,  3.0),
    'PCab':    (0.2,  3.0),
    # Tier 3
    'CaMKo':   (0.2,  3.0),
    'KmCaM':   (0.2,  3.0),
    'aCaMK':   (0.2,  3.0),
    'bCaMK':   (0.2,  3.0),
    'bt':      (0.2,  3.0),
    'tau_tr':  (0.2,  3.0),
    'tau_diff_Ca': (0.2, 3.0),
    'tau_diff_Na': (0.2, 3.0),
    'tau_diff_K':  (0.2, 3.0),
    'Kmup':    (0.2,  3.0),
}

# Tissue diffusion bounds (cm²/ms)
# CV ∝ √D. To get CV=25 cm/s (vs adult ~54), D ≈ (25/54)² × D_adult ≈ 0.21× adult.
# Adult D ≈ 0.001 cm²/ms → target D ≈ 0.0002–0.0004
TISSUE_BOUNDS = {
    'D_long':  (0.0001, 0.002),   # hiPSC-CM: ~0.0002-0.0005
    'D_trans': (0.00005, 0.001),   # AR=2.0 → D_T = D_L/4
}

def get_default_bounds(model: str, tier: int) -> dict[str, tuple[float, float]]:
    """Return default scaling factor bounds for each parameter."""
    names = get_param_names(model, tier)
    all_bounds = TTP06_BOUNDS if model == 'ttp06' else ORD_BOUNDS
    return {name: all_bounds[name] for name in names}


@dataclass
class TuningTargets:
    """Target electrophysiology values. See TARGET_VALUES.md for derivation."""
    apd_90: float = 250.0              # ms, midrange mature hiPSC-CM
    cv_longitudinal: float = 25.0      # cm/s, matured aligned hiPSC-CM
    cv_transverse: float = 12.5        # cm/s, AR = 2.0
    tissue_apd_90: float = 225.0       # ms, estimated (cell APD - 25 ms loading)
    restitution: list[tuple[float, float]] = field(default_factory=lambda: [
        (50, 150), (100, 190), (200, 230), (500, 248)
    ])
    # Optional
    apd_50: float | None = None
    v_rest: float = -85.0              # mV, TTP06 native (don't fight the model)
    dvdt_max: float = 150.0            # V/s, matured hiPSC-CM in 3D EHT
    ap_morphology: torch.Tensor | None = None
    erp: float | None = None

@dataclass
class TuningConfig:
    ionic_model: str = 'ttp06'         # 'ttp06' or 'ord'
    cell_type: str = 'epi'
    param_tier: int = 2                # 1, 2, or 3 (default: Tier 2)
    device: str = 'cuda'
    dx: float = 0.02                   # cm (tissue mesh spacing)
    dt: float = 0.01                   # ms
    param_bounds: dict | None = None   # overrides tier defaults if provided
    bayesopt_budget: int = 300         # Phase 1 eval budget
    tissue_budget: int = 30            # Phase 2 eval budget
    joint_training_sims: int | None = None  # auto-calculated from tier if None

    def __post_init__(self):
        if self.param_bounds is None:
            self.param_bounds = get_default_bounds(self.ionic_model, self.param_tier)
        if self.joint_training_sims is None:
            n_ionic = len(get_param_names(self.ionic_model, self.param_tier))
            n_total = n_ionic + 2  # + D_long, D_trans
            self.joint_training_sims = n_total * 100  # 100 per dimension

    @property
    def ionic_param_names(self) -> list[str]:
        return get_param_names(self.ionic_model, self.param_tier)

    @property
    def n_ionic_params(self) -> int:
        return len(self.ionic_param_names)

@dataclass
class TuningResult:
    theta_ionic: dict[str, float]      # conductance scaling factors
    theta_tissue: dict[str, float]     # D_long, D_trans
    pareto_front: list[dict] | None    # all non-dominated solutions
    validation: dict | None            # validation report
    emulator: object | None            # GP emulator for reuse
    config: TuningConfig | None = None # config used (for reproducibility)
```

### I-2: AP metrics (`tuner/objectives/ap_metrics.py`)

Functions that take a voltage trace V(t) and return scalar metrics:

```python
def measure_apd(V: Tensor, t: Tensor, repol_frac: float = 0.9) -> float
def measure_dvdt_max(V: Tensor, t: Tensor) -> float
def measure_v_rest(V: Tensor) -> float
def measure_v_peak(V: Tensor) -> float
def ap_morphology_rmse(V_model: Tensor, V_target: Tensor) -> float
```

### I-3: Single-cell runner (`tuner/objectives/ap_metrics.py` or `tuner/cell_fitter.py`)

```python
def run_single_cell(
    ionic_model: str,
    cell_type: str,
    theta_ionic: dict[str, float],
    cls: list[float],             # cycle lengths to pace at
    n_beats: int = 20,
    device: str = 'cuda',
) -> dict:
    """Run single-cell pacing protocol, return {apd_90, dvdt_max, v_rest, V_trace, ...}"""
```

Uses `TTP06Model.step()` directly — no spatial solver, no mesh. Each eval should take ~0.2-0.5s on GPU.

### I-4: Conductance scaling (`tuner/objectives/conductance.py`)

```python
def apply_theta_ionic(
    params: TTP06Parameters | ORdParameters,
    theta: dict[str, float],
    model_name: str = 'ttp06',
):
    """Scale model parameters by theta (scaling factors relative to published).

    theta = {'GKr': 1.2} means GKr is set to 120% of its published value.
    Uses the parameter registry to map names to attributes.
    """
    registry = TTP06_PARAMS if model_name == 'ttp06' else ORD_PARAMS
    for name, scale in theta.items():
        attr_name, published_value, _ = registry[name]
        setattr(params, attr_name, published_value * scale)
```

### I-5: Restitution protocol (`tuner/objectives/restitution.py`)

```python
def run_s1s2_restitution(
    ionic_model: str,
    cell_type: str,
    theta_ionic: dict[str, float],
    s1_cl: float,                  # e.g., 1000 ms
    di_values: list[float],        # e.g., [50, 100, 150, 200, 300, 500]
    n_s1_beats: int = 10,
    device: str = 'cuda',
) -> list[tuple[float, float]]:
    """Return [(DI, APD_90), ...] restitution curve."""
```

### Validation Criteria (Phase I)

| Test | Criterion |
|------|-----------|
| I-V1 | `TuningTargets` and `TuningConfig` instantiate with defaults |
| I-V2 | `TuningConfig(param_tier=1)` selects 6 params (TTP06) / 8 params (ORd) |
| I-V3 | `TuningConfig(param_tier=2)` selects 11 params (TTP06) / 15 params (ORd) |
| I-V4 | `TuningConfig(param_tier=3)` selects 17 params (TTP06) / 25 params (ORd) |
| I-V5 | `joint_training_sims` auto-calculated: Tier 2 TTP06 → 1300 sims |
| I-V6 | `measure_apd()` returns 270-290 ms for default TTP06/epi at CL=1000 |
| I-V7 | `measure_dvdt_max()` returns ~300 V/s for default TTP06/epi |
| I-V8 | `run_single_cell()` completes in <1s on GPU for 20 beats at CL=1000 |
| I-V9 | `run_s1s2_restitution()` returns monotonically increasing APD with DI |
| I-V10 | Conductance scaling (GKr=0.5 → longer APD, GKr=2.0 → shorter APD) |
| I-V11 | `apply_theta_ionic()` correctly scales both TTP06 and ORd parameters |
| I-V12 | Multi-CL pacing: APD(CL=1000) > APD(CL=500) > APD(CL=350) |

---

## Phase Ib: Baseline

Run default TTP06/epi with no scaling to quantify the gap between published adult parameters and hiPSC-CM targets. This takes 5 minutes and catches setup bugs before the optimizer runs.

```python
def run_baseline(config: TuningConfig) -> dict:
    """Run default model with no scaling, measure all targets."""
    theta_identity = {name: 1.0 for name in config.ionic_param_names}
    result = run_single_cell('ttp06', 'epi', theta_identity, cls=[1000, 500, 350])
    restit = run_s1s2_restitution('ttp06', 'epi', theta_identity, s1_cl=1000,
                                  di_values=[50, 100, 200, 500])
    return {
        'apd_90': result['apd_90'],       # expect ~280 ms (adult)
        'dvdt_max': result['dvdt_max'],   # expect ~300 V/s (adult)
        'v_rest': result['v_rest'],       # expect ~-85 mV (adult)
        'restitution': restit,
        'gap': {
            'apd_gap': result['apd_90'] - 250.0,        # target: 250 (expect ~+30)
            'dvdt_gap': result['dvdt_max'] - 150.0,       # target: 150 (expect ~+150)
            'vrest_gap': result['v_rest'] - (-85.0),       # target: -85 (expect ~0)
        }
    }
```

### Validation Criteria (Phase Ib)

| Test | Criterion |
|------|-----------|
| Ib-V1 | Baseline runs without error |
| Ib-V2 | Default APD90 ~280 ms (confirms model works) |
| Ib-V3 | Default dvdt_max ~300 V/s (confirms upstroke is measured correctly) |
| Ib-V4 | Gap to targets: APD gap ~30 ms, dvdt gap ~150 V/s, V_rest gap ~0 mV |
| Ib-V5 | GNa=0.5 produces dvdt_max ~120-180 V/s (target=150 is reachable) |
| Ib-V6 | GKr=1.5 shortens APD below 260 ms (target=250 is reachable) |

---

## Phase II: Cell Fitter

### II-1: BayesOpt wrapper (`tuner/cell_fitter.py`)

```python
class CellFitter:
    def __init__(self, targets: TuningTargets, config: TuningConfig): ...

    def fit(self) -> list[dict]:
        """
        Run multi-objective BayesOpt (qNEHVI) over 8 ionic conductances.
        Returns Pareto front of non-dominated solutions.
        """
```

Inner loop:
1. BoTorch `SingleTaskGP` per objective (3 GPs: AP_RMSE, APD_error, restitution_RMSE)
2. `qNoisyExpectedHypervolumeImprovement` acquisition function
3. Batch size 10-20, total budget ~300 evaluations
4. Each eval: `run_single_cell()` at CL=1000,500,350 + `run_s1s2_restitution()`

### II-2: CMA-ES fallback

If BoTorch GP underperforms (poor surrogate fit in 8D):
```python
def fit_cmaes(self) -> dict:
    """Scalarized objective with CMA-ES. Returns single best solution."""
```
Weighted sum: `0.4·AP_RMSE + 0.3·APD_error + 0.3·restitution_RMSE`

### Validation Criteria (Phase II)

| Test | Criterion |
|------|-----------|
| II-V1 | CellFitter converges within 300 evals (objective decreases) |
| II-V2 | Best solution has APD_90 within 5% of target |
| II-V3 | Best solution has restitution RMSE < 10 ms |
| II-V4 | Pareto front has ≥10 non-dominated solutions |
| II-V5 | GKr and GKs values differ across Pareto front (not locked together) |
| II-V6 | Total Phase 1 runtime < 10 min on GPU |

---

## Phase III: Tissue Fitter

### III-1: CV measurement (`tuner/objectives/cv_metrics.py`)

```python
def measure_cv(
    ionic_model: str,
    cell_type: str,
    theta_ionic: dict[str, float],
    D: float,
    cable_length: float = 2.0,     # cm
    dx: float = 0.02,              # cm
    device: str = 'cuda',
) -> float:
    """
    Run 1D cable simulation with MonodomainSimulation (V5.4).
    Measure CV from activation times at two probe points.
    """
```

Uses:
```python
from cardiac_sim.simulation.classical import MonodomainSimulation
from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
```

### III-2: Tissue APD measurement

```python
def measure_tissue_apd(
    ionic_model: str,
    cell_type: str,
    theta_ionic: dict[str, float],
    D_long: float,
    D_trans: float,
    slab_size: float = 1.0,       # cm
    dx: float = 0.02,
    device: str = 'cuda',
) -> float:
    """
    Run 2D slab simulation, measure APD at center node.
    Tissue APD < single-cell APD due to electrotonic loading.
    """
```

### III-3: BayesOpt over D_long, D_trans (`tuner/tissue_fitter.py`)

```python
class TissueFitter:
    def __init__(self, targets: TuningTargets, config: TuningConfig,
                 theta_ionic: dict[str, float]): ...

    def fit(self) -> dict[str, float]:
        """
        BayesOpt (qEI) over D_long, D_trans.
        Analytical warm-start: D ≈ (CV_target/CV_ref)² × D_ref.
        Returns {'D_long': ..., 'D_trans': ...}.
        """
```

### Validation Criteria (Phase III)

| Test | Criterion |
|------|-----------|
| III-V1 | `measure_cv()` returns ~54 cm/s for default TTP06/epi with published D |
| III-V2 | CV scales as ~√D (doubling D increases CV by ~41%) |
| III-V3 | `measure_tissue_apd()` returns APD shorter than single-cell APD |
| III-V4 | TissueFitter converges within 30 evals |
| III-V5 | Final CV_long and CV_trans within 2% of targets |
| III-V6 | Analytical warm-start is within 15% of final optimized D |

---

## Phase IV: Joint Refinement

### IV-1: Latin Hypercube training data generator

```python
def generate_training_data(
    pareto_front: list[dict],
    D_long: float, D_trans: float,
    config: TuningConfig,
    n_samples: int = 500,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    LHS around Phase 1 Pareto front × Phase 2 D values.
    Run full 2D tissue sim for each.
    Returns (X: [N, n_params+2], Y: {cv_L, cv_T, tissue_apd, restitution_pca}).

    IMPORTANT: Each training sim measures CV_L, CV_T, and tissue_APD from a
    short 2D tissue simulation (~5 beats, ~2 min/sim). Restitution is measured
    at the SINGLE-CELL level (not tissue) to keep cost manageable — tissue S1S2
    would cost ~1 hour per sample, making 500 samples infeasible.
    Single-cell restitution shape is primarily an ionic property and transfers
    well to tissue. Tissue restitution is only verified in Phase V validation.
    """
```

### IV-2: GP emulator (`tuner/surrogate/emulator.py`)

```python
class RestitutionEmulator:
    def __init__(self, X: Tensor, Y: dict): ...

    def fit(self):
        """PCA on restitution curves, GP per PCA component + scalar targets."""

    def predict(self, X_new: Tensor) -> dict:
        """Return predicted CV_L, CV_T, tissue_APD, restitution with uncertainty."""
```

### IV-3: NSGA-II on surrogate (`tuner/joint_refiner.py`)

```python
class JointRefiner:
    def __init__(self, targets: TuningTargets, config: TuningConfig,
                 pareto_front: list[dict], D: dict[str, float]): ...

    def refine(self) -> dict:
        """
        1. Generate 500 training sims
        2. Build GP emulator
        3. NSGA-II on emulator (100K evals in seconds)
        4. Validate top 10 on real simulator
        5. Active learning if emulator error > 5%
        Returns final θ* = {theta_ionic + theta_tissue}.
        """
```

### Validation Criteria (Phase IV)

| Test | Criterion |
|------|-----------|
| IV-V1 | GP emulator cross-validation R² > 0.9 for CV and APD predictions |
| IV-V2 | PCA with 3 components captures >95% restitution variance |
| IV-V3 | NSGA-II Pareto front has ≥20 non-dominated solutions |
| IV-V4 | Top candidate: real sim CV within 5% of emulator prediction |
| IV-V5 | Final θ*: CV_L and CV_T within 2% of target |
| IV-V6 | Final θ*: tissue APD within 3% of target |
| IV-V7 | Phase 3 total runtime < 6 hours on GPU |

---

## Phase V: Validation

### V-1: Automated validation suite (`tuner/validator.py`)

```python
class Validator:
    def __init__(self, theta: dict, targets: TuningTargets, config: TuningConfig): ...

    def validate(self) -> dict:
        """Run all validation tests, return report."""
```

### Validation Tests

| Test | What | Pass Criterion |
|------|------|----------------|
| V-V1 | Novel CL pacing (2000, 800, 600, 400, 300 ms) | APD error < 5% at all CLs |
| V-V2 | CV in 1D cable | Within ±2% of Phase 2 result |
| V-V3 | CV in 2D plane wave | Within ±2% of target |
| V-V4 | 2× threshold stimulus | Propagation succeeds |
| V-V5 | 0.5× threshold stimulus | Propagation fails (confirms threshold) |
| V-V6 | 20 beats at CL=1000 | No APD drift (steady state) |
| V-V7 | ERP measurement | Within physiological range (200-300 ms) |

---

## Phase VI: Integration

### VI-1: Pipeline orchestrator (`tuner/pipeline.py`)

```python
class EngineTuner:
    def __init__(self, targets: TuningTargets, config: TuningConfig): ...

    def tune(self) -> TuningResult:
        """Full pipeline: cell fit → tissue fit → joint refine → validate."""

    def save(self, path: str): ...
    def load(cls, path: str) -> TuningResult: ...
```

### VI-2: CLI entry point

```bash
cd Optimizer/V1
python -m tuner --ionic-model ttp06 --cell-type epi \
    --cv-long 65 --cv-trans 25 --apd 280 --tissue-apd 260 \
    --device cuda --output results/run_001.json
```

### Validation Criteria (Phase VI)

| Test | Criterion |
|------|-----------|
| VI-V1 | `EngineTuner.tune()` runs end-to-end without error |
| VI-V2 | Results save/load round-trip preserves all values |
| VI-V3 | Total pipeline runtime < 8 hours on GPU |
| VI-V4 | Output JSON contains all θ values + validation report |

---

## Implementation Order

```
Phase I   ████████░░░░░░░░░░░░  Foundation (config, metrics, single-cell runner)
Phase II  ░░░░░░░░████████░░░░  Cell fitter (BayesOpt loop)
Phase III ░░░░░░░░░░░░████░░░░  Tissue fitter (CV measurement + BayesOpt)
Phase IV  ░░░░░░░░░░░░░░██████  Joint refinement (GP emulator + NSGA-II)
Phase V   ░░░░░░░░░░░░░░░░░░██  Validation suite
Phase VI  ░░░░░░░░░░░░░░░░░░░█  Integration (CLI, save/load)
```

Each phase depends on the previous. Implement → validate → commit → next phase.

---

## V2 Scope (deferred)

- Engine adapter abstraction (EngineAdapter ABC) — see `improvement.md`
- Bidomain V1 integration (D_i, D_e per axis → 4 tissue params)
- LBM V1 integration (tau_L, tau_T via tau_from_D)
- Cross-engine validation (Bidomain ↔ Monodomain within 1%)
- HMC/NUTS on GP surrogate for full posterior distributions
- Sobol sensitivity pre-screen for automatic parameter selection
- Full anisotropic diffusion tensor with fiber orientation
