# Computational Pipeline Architecture: Critical Analysis & Improvement Recommendations

**Project:** Cardiac Electrophysiology Simulation
**Date:** 2025-12-05
**Reviewer:** Claude Code (Anthropic)

---

## Executive Summary

The codebase demonstrates solid scientific understanding and functional implementations, but suffers from **architectural fragmentation**, **code duplication**, **inconsistent abstractions**, and **poor separation of concerns**. The pipeline would benefit significantly from refactoring into a more modular, testable, and maintainable structure.

**Overall Grade: C+ (Functional but needs significant refactoring)**

---

## 1. CRITICAL ISSUES

### 1.1 Version Proliferation Without Clear Purpose ⚠️ SEVERE

**Problem:**
- `version1.py`, `version2.py`, `version3.py` exist with nearly identical code
- `version2.py` and `version3.py` are **byte-for-byte identical** (305 lines each)
- No clear documentation explaining the differences between versions
- Violates DRY (Don't Repeat Yourself) principle catastrophically

**Impact:**
- Bug fixes must be applied to multiple files
- Maintenance nightmare
- Confusion about which version to use
- Increased risk of divergence and inconsistencies

**Recommended Fix:**
```python
# Proposed structure:
# models/
#   __init__.py
#   base.py              # Abstract base class
#   aliev_panfilov.py    # Full ionic model (current version1)
#   relaxed_cable.py     # Simplified model (version2/3)
#   model_factory.py     # Factory for model selection
```

**Example:**
```python
# models/base.py
from abc import ABC, abstractmethod

class CardiacModel(ABC):
    @abstractmethod
    def step(self, V, w, mesh, params, I_stim, dt):
        """Advance one time step"""
        pass

    @abstractmethod
    def default_params(self):
        """Return default parameter dictionary"""
        pass

# Then users select:
from models import AlievPanfilovModel, RelaxedCableModel
model = AlievPanfilovModel()  # or RelaxedCableModel()
```

---

### 1.2 Duplicated `run_simulation()` Functions ⚠️ SEVERE

**Problem:**
- `run_simulation()` appears in:
  - `plot_simulation.py` (lines 22-52)
  - `plot_simulation_circular.py` (lines 15-40)
  - `plot_simulation_infarct_flow` (lines 24-61)
  - `debug_infarct_snapshots.py` (lines 13-28)
- Nearly identical logic with minor variations
- ~150 lines of duplicated code across files

**Impact:**
- Cannot easily modify simulation algorithm globally
- Inconsistent behavior across different entry points
- Testing requires checking multiple implementations

**Recommended Fix:**
```python
# simulation/runner.py
class SimulationRunner:
    def __init__(self, mesh, model, params):
        self.mesh = mesh
        self.model = model
        self.params = params

    def run(self, dt=0.02, t_stop=60.0, output_stride=5,
            stim_fn=None, callbacks=None):
        """Unified simulation loop with callback support"""
        if stim_fn is None:
            stim_fn = self.mesh.periodic_left_edge_stimulus()

        V, w = self.mesh.empty_state()
        frames, times = [], []

        n_steps = int(np.ceil(t_stop / dt))
        for n in range(n_steps):
            t = n * dt

            if n % output_stride == 0:
                frames.append(V.copy())
                times.append(t)
                if callbacks:
                    for cb in callbacks:
                        cb(t, V, w)

            I_stim = stim_fn(t)
            V, w = self.model.step(V, w, self.mesh,
                                   self.params, I_stim, dt)

        return np.array(frames), np.array(times)
```

---

### 1.3 Plotting Code Duplication ⚠️ MODERATE

**Problem:**
- `_color_limits()` duplicated in 3 files
- `plot_geometry()` duplicated in 2 files with slight variations
- `animate_voltage()` has 3 nearly-identical implementations
- Inconsistent function signatures and behavior

**Recommended Fix:**
```python
# visualization/
#   __init__.py
#   colormaps.py     # Color scaling utilities
#   geometry.py      # Geometry plotting
#   animations.py    # Animation helpers
#   snapshots.py     # Static snapshot plots
```

---

### 1.4 Inconsistent Mesh Interface Usage ⚠️ MODERATE

**Problem:**
Compare these two calls to the time-stepping function:

```python
# plot_simulation.py (line 44-45):
V, w = step_monodomain(
    V, w, mesh.Dxx, mesh.Dxy, mesh.Dyy, mesh.dx, mesh.dy, dt, params, I_stim
)

# plot_simulation_circular.py (line 34):
V, w = step_relaxed_monodomain(V, w, mesh, params, I_stim, dt)
```

**Issues:**
- First version extracts 5 parameters from mesh individually
- Second version passes entire mesh object
- **Inconsistent API design** between `version1` and `version2/3`
- Makes code harder to understand and refactor

**Recommended Fix:**
```python
# All models should use consistent interface:
def step(self, V, w, mesh, params, I_stim, dt):
    """All models accept mesh object directly"""
    pass
```

---

## 2. ARCHITECTURAL PROBLEMS

### 2.1 Poor Separation of Concerns ⚠️ MODERATE

**Problem:**
Files mix multiple responsibilities:
- `plot_simulation.py`: simulation logic + visualization + CLI
- `mesh_setup.py`: mesh construction + stimulus functions + data container
- Tight coupling between components

**Recommended Structure:**
```
heart_conduction/
├── models/              # Electrophysiology models
│   ├── aliev_panfilov.py
│   └── relaxed_cable.py
├── geometry/            # Mesh and geometry
│   ├── mesh.py
│   └── fiber_fields.py
├── simulation/          # Simulation engine
│   ├── runner.py
│   └── stimulus.py
├── analysis/            # Post-processing
│   ├── conduction_velocity.py
│   └── activation_maps.py
├── visualization/       # All plotting
│   ├── animations.py
│   └── geometry.py
└── scripts/             # Entry points
    ├── run_baseline.py
    └── run_infarct.py
```

---

### 2.2 Hardcoded Magic Numbers ⚠️ MODERATE

**Problem:**
```python
# version2.py line 199:
n_sub = int(np.ceil(dt / dt_crit))

# mesh_setup.py line 206:
boost = 1.0 + boundary_boost * np.exp(-(distance_from_boundary / boost_width) ** 2)

# plot_simulation_circular.py line 77:
ax.quiver(X, Y, U, V, color="red", scale=20, width=0.003)
```

Magic numbers appear throughout: `20`, `0.003`, `0.9`, `3.0`, etc.

**Recommended Fix:**
```python
# config/constants.py
class VisualizationConfig:
    FIBER_QUIVER_SCALE = 20
    FIBER_QUIVER_WIDTH = 0.003
    FIBER_QUIVER_COLOR = 'red'
    GEOMETRY_DOWNSAMPLE_FACTOR = 20

class NumericalConfig:
    STABILITY_SAFETY_FACTOR = 0.9
    MIN_SUBSTEPS = 1
    MAX_SUBSTEPS = 1000
```

---

### 2.3 Missing Error Handling ⚠️ MODERATE

**Problem:**
No validation or error handling in critical functions:

```python
def step_relaxed_monodomain(V, w, mesh, params, I_stim, dt):
    # No checks for:
    # - V, w same shape as mesh?
    # - dt > 0?
    # - params contain required keys?
    # - I_stim matches mesh dimensions?
    # - Numerical stability violations?
```

**Recommended Fix:**
```python
def step_relaxed_monodomain(V, w, mesh, params, I_stim, dt):
    # Validate inputs
    if V.shape != (mesh.ny, mesh.nx):
        raise ValueError(f"V shape {V.shape} != mesh shape ({mesh.ny}, {mesh.nx})")
    if dt <= 0:
        raise ValueError(f"Time step must be positive, got {dt}")

    required_params = ['C_m', 'tau_v', 'tau_w', 'V_rest']
    missing = [p for p in required_params if p not in params]
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")

    # Warn about numerical issues
    if dt > 0.1:
        warnings.warn(f"Large time step {dt} ms may cause instability")

    # ... actual computation ...
```

---

### 2.4 No Unit Testing ⚠️ SEVERE

**Problem:**
- Zero test files in the repository
- No regression tests
- No validation against analytical solutions
- Difficult to verify correctness

**Recommended Fix:**
```python
# tests/
#   test_models.py
#   test_mesh.py
#   test_diffusion.py
#   test_integration.py
#   fixtures/
#     reference_solutions.npz

# Example test:
def test_conduction_velocity_homogeneous():
    """Verify CV matches theoretical prediction in uniform tissue"""
    mesh = create_left_to_right_mesh(nx=200, ny=50)
    model = RelaxedCableModel()

    cv_measured = measure_conduction_velocity(mesh, model)
    cv_expected = calculate_theoretical_cv(mesh.Dxx[0,0], params)

    assert abs(cv_measured - cv_expected) / cv_expected < 0.05  # 5% tolerance
```

---

## 3. CODE QUALITY ISSUES

### 3.1 Inconsistent Naming Conventions ⚠️ MINOR

**Problems:**
```python
# Mixed conventions:
D_parallel      # Snake_case with capital
D_perp          # Abbreviated
epsilon_tissue  # Full descriptive name
eps_core        # Abbreviated version
nx, ny          # Lowercase
Lx, Ly          # Uppercase
```

**Recommended:**
```python
# Use consistent style:
d_parallel      # or diffusion_parallel
d_perpendicular # Full name
epsilon_tissue  # Full name
epsilon_core    # Consistent naming
n_x, n_y        # or num_points_x
length_x, length_y  # Full descriptive
```

---

### 3.2 Missing Type Hints in Critical Places ⚠️ MINOR

**Problem:**
```python
# version2.py has some type hints but inconsistent:
def build_D_components(
    theta: np.ndarray,      # Has type hint
    D_parallel,             # Missing!
    D_perp,                 # Missing!
    epsilon_tissue: np.ndarray,
):
```

**Recommended:**
```python
from typing import Union
import numpy.typing as npt

def build_D_components(
    theta: npt.NDArray[np.float64],
    D_parallel: Union[float, npt.NDArray[np.float64]],
    D_perp: Union[float, npt.NDArray[np.float64]],
    epsilon_tissue: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Construct diffusion tensor components.

    Returns
    -------
    Dxx, Dxy, Dyy : arrays of shape (ny, nx)
    """
```

---

### 3.3 Docstring Inconsistencies ⚠️ MINOR

**Problem:**
- `version1.py` has extensive docstrings
- `version2.py` has some docstrings
- Diagnostic scripts have minimal documentation
- Mixed docstring styles (NumPy vs Google style)

**Recommendation:** Standardize on NumPy style throughout:
```python
def function(param1, param2):
    """
    Brief one-line description.

    More detailed explanation if needed, describing the algorithm,
    assumptions, and important implementation details.

    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type
        Description of param2

    Returns
    -------
    result : type
        Description of return value

    Raises
    ------
    ValueError
        When and why this is raised

    Examples
    --------
    >>> result = function(1, 2)
    >>> print(result)
    3
    """
```

---

## 4. PERFORMANCE CONCERNS

### 4.1 Inefficient Array Operations ⚠️ MODERATE

**Problem:**
```python
# version2.py lines 274-287:
for _ in range(n_sub):
    # Multiple array operations per substep
    diff = _anisotropic_diffusion(V, mesh)
    reaction = -(V - V_rest) / tau_v
    dVdt = (diff + reaction + I_stim) / C_m
    V = V + dt_sub * dVdt
    # ... more operations
```

Each operation creates temporary arrays. For 160×160 grid over 3000 steps, this creates millions of temporary arrays.

**Recommended:**
```python
# Use in-place operations where possible:
def step_relaxed_monodomain_inplace(V, w, mesh, params, I_stim, dt,
                                    scratch_arrays=None):
    """Version that reuses buffers to reduce allocations"""
    if scratch_arrays is None:
        scratch_arrays = {
            'diff': np.zeros_like(V),
            'reaction': np.zeros_like(V),
            'dVdt': np.zeros_like(V),
        }

    # Reuse buffers
    diff = scratch_arrays['diff']
    # ... etc
```

---

### 4.2 No Profiling or Optimization Markers ⚠️ MINOR

**Problem:**
- No indication of performance bottlenecks
- No profiling data
- Unclear which parts are slow

**Recommendation:**
```python
# Add profiling decorators for key functions:
import time
from functools import wraps

def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__}: {elapsed*1000:.2f} ms")
        return result
    return wrapper

@profile
def step_relaxed_monodomain(...):
    ...
```

Or use line_profiler:
```python
# Add @profile decorator and run:
# kernprof -l -v script.py
```

---

### 4.3 Potential for Vectorization/JIT Compilation ⚠️ MINOR

**Current:** Pure NumPy implementation
**Potential:** Could benefit from:
- Numba JIT compilation for hot loops
- Cython for critical paths
- JAX for GPU acceleration (future)

**Example:**
```python
from numba import jit

@jit(nopython=True)
def _anisotropic_diffusion_jit(V, Dxx, Dxy, Dyy, dx, dy):
    """JIT-compiled version for speed"""
    # Implementation here
    pass
```

---

## 5. MISSING FEATURES

### 5.1 No Configuration Management ⚠️ MODERATE

**Problem:**
- Parameters hardcoded or passed via CLI args
- No way to save/load configuration files
- Difficult to reproduce experiments

**Recommended:**
```python
# config/
#   default.yaml
#   experiment_infarct_3mm.yaml
#   experiment_anisotropy_study.yaml

# config/default.yaml
simulation:
  dt: 0.02
  t_stop: 60.0
  output_stride: 5

mesh:
  type: "left_to_right"
  nx: 160
  ny: 120
  Lx: 20.0
  Ly: 15.0

model:
  type: "aliev_panfilov"
  params:
    k: 8.0
    a: 0.1
    epsilon0: 0.01

# Then load:
import yaml
with open('config/experiment.yaml') as f:
    config = yaml.safe_load(f)
```

---

### 5.2 No Result Serialization ⚠️ MODERATE

**Problem:**
- Simulations produce frames but no persistent storage
- Cannot analyze results offline
- Cannot share results with collaborators

**Recommended:**
```python
# simulation/output.py
import h5py

class SimulationOutput:
    def save(self, filename, frames, times, mesh, params, metadata=None):
        """Save simulation results to HDF5"""
        with h5py.File(filename, 'w') as f:
            f.create_dataset('frames', data=frames)
            f.create_dataset('times', data=times)

            # Mesh data
            mesh_grp = f.create_group('mesh')
            mesh_grp.create_dataset('Dxx', data=mesh.Dxx)
            mesh_grp.create_dataset('theta', data=mesh.theta)
            mesh_grp.attrs['nx'] = mesh.nx
            mesh_grp.attrs['Lx'] = mesh.Lx

            # Parameters
            param_grp = f.create_group('params')
            for key, val in params.items():
                param_grp.attrs[key] = val

            # Metadata
            if metadata:
                meta_grp = f.create_group('metadata')
                meta_grp.attrs['timestamp'] = metadata.get('timestamp')
                meta_grp.attrs['git_commit'] = metadata.get('commit')

    def load(self, filename):
        """Load previous simulation results"""
        # Implementation
```

---

### 5.3 No Analysis Tools ⚠️ MODERATE

**Problem:**
- Can visualize, but no quantitative analysis
- No conduction velocity measurement
- No activation time maps
- No reentry detection

**Recommended:**
```python
# analysis/
#   activation_maps.py
#   conduction_velocity.py
#   reentry_detection.py
#   wave_propagation.py

# Example:
class ActivationAnalyzer:
    def __init__(self, frames, times, threshold=0.5):
        self.frames = frames
        self.times = times
        self.threshold = threshold

    def compute_activation_map(self):
        """Compute time of first activation for each point"""
        activation_time = np.full(self.frames[0].shape, np.nan)

        for i in range(self.frames.shape[1]):  # ny
            for j in range(self.frames.shape[2]):  # nx
                signal = self.frames[:, i, j]
                activated = np.where(signal > self.threshold)[0]
                if len(activated) > 0:
                    activation_time[i, j] = self.times[activated[0]]

        return activation_time

    def measure_conduction_velocity(self, path):
        """Measure CV along a specified path"""
        # Implementation
```

---

### 5.4 No Logging Infrastructure ⚠️ MINOR

**Problem:**
- Print statements scattered throughout
- No structured logging
- Difficult to debug issues

**Recommended:**
```python
import logging

# setup_logging.py
def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simulation.log'),
            logging.StreamHandler()
        ]
    )

# Then in modules:
logger = logging.getLogger(__name__)

def run_simulation(...):
    logger.info(f"Starting simulation: t_stop={t_stop}, dt={dt}")
    logger.debug(f"Mesh dimensions: {mesh.nx}×{mesh.ny}")
    # ...
    logger.warning(f"Large time step detected: dt={dt}")
```

---

## 6. DIAGNOSTIC SCRIPTS ARE POORLY INTEGRATED

### 6.1 Diagnostic Scripts Reinvent the Wheel ⚠️ MODERATE

**Problem:**
- `conduction_diagnostic.py`, `snapshot_probe.py`, etc. each create their own simulation loops
- Could use the unified `SimulationRunner` with callbacks
- No systematic test suite

**Recommended:**
```python
# Instead of custom simulation loops, use callbacks:
class ActivationProbe:
    """Callback to detect activation times"""
    def __init__(self, columns, threshold=0.2):
        self.columns = columns
        self.threshold = threshold
        self.crossing_times = {col: None for col in columns}

    def __call__(self, t, V, w):
        for col in self.columns:
            if self.crossing_times[col] is None:
                if np.max(V[:, col]) >= self.threshold:
                    self.crossing_times[col] = t

# Usage:
probe = ActivationProbe(columns=[0, 10, 20, 40, 60, 79])
runner.run(..., callbacks=[probe])
print(probe.crossing_times)
```

---

## 7. POSITIVE ASPECTS (To Preserve)

Despite criticisms, the code has strengths:

✅ **Good mathematical documentation** in version1.py header
✅ **Clear physical meaning** of parameters
✅ **Functional implementations** that produce reasonable results
✅ **Multiple geometry options** (left-right, circular, spiral, flow)
✅ **Flexible stimulus functions** in TissueMesh class
✅ **Automatic stability checking** via `_stable_substep()`
✅ **CLI interfaces** for all entry points
✅ **Reasonable use of NumPy** for numerical operations

---

## 8. PRIORITIZED REFACTORING ROADMAP

### Phase 1: Critical (Do First) 🔴
1. **Consolidate version2.py and version3.py** (identical files)
2. **Create unified SimulationRunner** to eliminate `run_simulation()` duplication
3. **Add input validation** to all public functions
4. **Create basic test suite** with at least 10 tests

**Estimated effort:** 2-3 days

---

### Phase 2: High Priority (Do Soon) 🟠
1. **Refactor into modular package structure** (models/, geometry/, visualization/)
2. **Consolidate plotting utilities** into visualization module
3. **Implement configuration file system** (YAML/JSON)
4. **Add result serialization** (HDF5 or NetCDF)
5. **Add logging infrastructure**

**Estimated effort:** 4-5 days

---

### Phase 3: Medium Priority (Nice to Have) 🟡
1. **Add analysis tools** (CV measurement, activation maps)
2. **Improve performance** (profiling, in-place operations)
3. **Complete type hints** throughout
4. **Standardize docstrings**
5. **Add continuous integration** (GitHub Actions for tests)

**Estimated effort:** 3-4 days

---

### Phase 4: Low Priority (Future) 🟢
1. **Add JIT compilation** (Numba/Cython)
2. **Create Jupyter notebook tutorials**
3. **Add interactive visualization** (Plotly/Bokeh)
4. **GPU acceleration** (JAX/CuPy)
5. **Parameter sensitivity analysis tools**

**Estimated effort:** 5-10 days

---

## 9. IMMEDIATE ACTIONABLE STEPS

If you can only do 5 things right now:

### 1. Delete version3.py (it's identical to version2.py)
```bash
rm version3.py
# Update imports in other files to use version2
```

### 2. Create a unified runner
```bash
mkdir simulation
# Move consolidated run_simulation() here
```

### 3. Add basic validation
```python
# Add to beginning of step functions:
assert V.shape == w.shape, "V and w must have same shape"
assert dt > 0, "Time step must be positive"
```

### 4. Create tests/ directory with one test
```python
# tests/test_basic.py
def test_left_to_right_propagation():
    """Smoke test that simulation runs without crashing"""
    mesh = create_left_to_right_mesh(nx=40, ny=20)
    # ... minimal test that runs successfully
```

### 5. Add a PARAMETERS_REFERENCE.txt usage example
```python
# Add to one main script:
# Example: Use parameters from PARAMETERS_REFERENCE.txt
# D_parallel = 1.0 mm²/ms (Section 2)
# D_perp = 0.5 mm²/ms (Section 2)
# See PARAMETERS_REFERENCE.txt for physiological justification
```

---

## 10. CONCLUSION

The current codebase is **scientifically sound but structurally fragile**. The physics is correct and the implementations work, but the architecture makes maintenance, testing, and extension unnecessarily difficult.

**Key Takeaway:** You have a working research prototype that urgently needs refactoring before adding more features. The duplication and fragmentation will compound as the project grows.

**Recommendation:** Invest 1-2 weeks in architectural refactoring now to save months of headaches later. Focus on eliminating duplication, adding tests, and creating clear module boundaries.

**Bottom Line:** This is typical research code that evolved organically. It's not bad—it just needs intentional engineering to become sustainable for long-term development.

---

## APPENDIX: Proposed Final Structure

```
heart_conduction/
├── README.md
├── setup.py
├── requirements.txt
├── PARAMETERS_REFERENCE.txt          # ✅ Already exists
├── ARCHITECTURE_CRITIQUE.md          # ✅ This document
│
├── heart_conduction/                 # Main package
│   ├── __init__.py
│   │
│   ├── models/                       # Electrophysiology models
│   │   ├── __init__.py
│   │   ├── base.py                   # Abstract base
│   │   ├── aliev_panfilov.py        # version1 → here
│   │   └── relaxed_cable.py         # version2/3 → here
│   │
│   ├── geometry/                     # Mesh and spatial setup
│   │   ├── __init__.py
│   │   ├── mesh.py                   # TissueMesh class
│   │   ├── fiber_fields.py          # Theta field generators
│   │   └── diffusion.py             # D_components builder
│   │
│   ├── simulation/                   # Simulation engine
│   │   ├── __init__.py
│   │   ├── runner.py                 # Unified SimulationRunner
│   │   ├── stimulus.py               # Stimulus functions
│   │   └── callbacks.py              # Diagnostic callbacks
│   │
│   ├── analysis/                     # Post-processing
│   │   ├── __init__.py
│   │   ├── activation_maps.py
│   │   ├── conduction_velocity.py
│   │   └── wave_analysis.py
│   │
│   ├── visualization/                # All plotting
│   │   ├── __init__.py
│   │   ├── animations.py
│   │   ├── snapshots.py
│   │   ├── geometry.py
│   │   └── colormaps.py
│   │
│   └── io/                           # Data I/O
│       ├── __init__.py
│       ├── config.py                 # YAML config loading
│       └── serialization.py          # HDF5 save/load
│
├── scripts/                          # Entry point scripts
│   ├── run_baseline.py
│   ├── run_infarct_simulation.py
│   └── run_parameter_sweep.py
│
├── tests/                            # Test suite
│   ├── test_models.py
│   ├── test_mesh.py
│   ├── test_simulation.py
│   ├── test_integration.py
│   └── fixtures/
│       └── reference_solutions.npz
│
├── configs/                          # Configuration files
│   ├── default.yaml
│   ├── infarct_3mm.yaml
│   └── anisotropy_study.yaml
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_tutorial.ipynb
│   ├── 02_parameter_exploration.ipynb
│   └── 03_validation.ipynb
│
└── docs/                             # Documentation
    ├── api_reference.md
    ├── user_guide.md
    └── developer_guide.md
```

---

**End of Critique**
