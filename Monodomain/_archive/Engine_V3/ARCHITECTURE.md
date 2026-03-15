# Engine_V3 Architecture Proposal

**Date**: 2025-12-10
**Purpose**: Clean, modular Python structure for Fenton-Karma cardiac simulation

---

## 1. V2 Problems to Fix

| Issue | V2 Problem | V3 Solution |
|-------|-----------|-------------|
| Code duplication | Diffusion code in 3 files | Single `diffusion.py` module |
| Mixed concerns | Ionic model + simulation mixed | Separate modules |
| File clutter | Debug files everywhere | Organized `tests/` folder |
| Animation embedded | Animation in simulation files | Separate `visualization.py` |
| No package structure | Flat files, relative imports | Proper Python package |

---

## 2. Proposed Directory Structure

```
Engine_V3/
│
├── __init__.py              # Package init, version info
├── README.md                # Usage documentation
│
├── # ============== CORE MODULES ==============
│
├── ionic/                   # Ionic models (swappable)
│   ├── __init__.py
│   ├── base.py              # Abstract base class for ionic models
│   ├── fenton_karma.py      # FK model implementation
│   └── aliev_panfilov.py    # AP model (optional, for comparison)
│
├── diffusion.py             # Anisotropic diffusion solver
│                            # - flux-based method
│                            # - boundary conditions
│                            # - Numba-accelerated
│
├── mesh.py                  # 2D mesh and geometry
│                            # - grid creation
│                            # - fiber fields
│                            # - tissue masks (infarct)
│                            # - coordinate transforms
│
├── parameters.py            # Parameter management
│                            # - FK parameters
│                            # - Physical constants
│                            # - Spatial scaling
│
├── # ============== SIMULATION ==============
│
├── simulation.py            # Core simulation engine
│                            # - Simulation class
│                            # - Time stepping
│                            # - Operator splitting
│                            # - State management
│
├── stimulus.py              # Stimulus protocols
│                            # - Point stimulus
│                            # - Pulse trains
│                            # - S1-S2 protocols
│                            # - Custom stimulus functions
│
├── # ============== VISUALIZATION ==============
│
├── visualization.py         # All plotting/animation
│                            # - Voltage heatmaps
│                            # - Animation creation
│                            # - Phase portraits
│                            # - Time traces
│
├── interactive.py           # Real-time interactive mode
│                            # - Click-to-stimulate
│                            # - Keyboard controls
│                            # - Live updates
│
├── # ============== ANALYSIS ==============
│
├── analysis.py              # Post-processing tools
│                            # - APD measurement
│                            # - CV measurement
│                            # - Activation maps
│                            # - Restitution curves
│
├── # ============== TESTS ==============
│
├── tests/
│   ├── __init__.py
│   ├── test_ionic.py        # Single-cell AP tests
│   ├── test_diffusion.py    # Diffusion solver tests
│   ├── test_cv.py           # Conduction velocity tests
│   ├── test_restitution.py  # APD/CV restitution tests
│   └── test_integration.py  # Full simulation tests
│
├── # ============== EXAMPLES ==============
│
├── examples/
│   ├── single_cell.py       # Single cell AP demo
│   ├── planar_wave.py       # 1D wave propagation
│   ├── spiral_wave.py       # 2D spiral initiation
│   ├── reentry.py           # Reentry around obstacle
│   └── interactive_demo.py  # Interactive click demo
│
└── # ============== DOCS ==============

    docs/
    ├── FENTON_KARMA_RESEARCH.md   # (already created)
    ├── PARAMETER_COMPARISON.md    # (already created)
    └── API.md                     # API documentation
```

---

## 3. Module Responsibilities

### 3.1 `ionic/base.py` - Abstract Base Class

```python
from abc import ABC, abstractmethod
import numpy as np

class IonicModel(ABC):
    """Abstract base class for cardiac ionic models."""

    @property
    @abstractmethod
    def n_state_variables(self) -> int:
        """Number of state variables (e.g., 2 for AP, 3 for FK)."""
        pass

    @property
    @abstractmethod
    def state_names(self) -> list[str]:
        """Names of state variables (e.g., ['u', 'v', 'w'])."""
        pass

    @abstractmethod
    def initialize_state(self, shape: tuple) -> dict[str, np.ndarray]:
        """Create initial state arrays at rest."""
        pass

    @abstractmethod
    def ionic_step(self, state: dict, dt: float, I_stim: np.ndarray) -> None:
        """Update state variables for one ionic time step (in-place)."""
        pass

    @abstractmethod
    def voltage_to_physical(self, u: np.ndarray) -> np.ndarray:
        """Convert normalized voltage to mV."""
        pass

    @abstractmethod
    def physical_to_voltage(self, V_mV: np.ndarray) -> np.ndarray:
        """Convert mV to normalized voltage."""
        pass
```

### 3.2 `ionic/fenton_karma.py` - FK Implementation

```python
class FentonKarmaModel(IonicModel):
    """Fenton-Karma 3-variable ionic model."""

    def __init__(self, params: FKParameters):
        self.params = params
        # Store Numba-compatible parameter tuple
        self._numba_params = self._pack_params()

    @property
    def n_state_variables(self) -> int:
        return 3  # u, v, w

    @property
    def state_names(self) -> list[str]:
        return ['u', 'v', 'w']

    def initialize_state(self, shape: tuple) -> dict[str, np.ndarray]:
        return {
            'u': np.zeros(shape),  # Voltage
            'v': np.ones(shape),   # Fast gate (starts open)
            'w': np.ones(shape),   # Slow gate (starts open)
        }

    def ionic_step(self, state: dict, dt: float, I_stim: np.ndarray) -> None:
        # Call Numba kernel
        _fk_ionic_kernel(
            state['u'], state['v'], state['w'],
            dt, I_stim, self._numba_params
        )
```

### 3.3 `diffusion.py` - Diffusion Solver

```python
"""Anisotropic diffusion solver with Numba acceleration."""

import numpy as np
import numba

@numba.jit(nopython=True, parallel=True, cache=True)
def compute_diffusion(u, Dxx, Dyy, Dxy, dx, dy):
    """
    Compute ∇·(D∇u) using flux-based finite differences.

    Supports anisotropic diffusion tensor:
    D = [[Dxx, Dxy],
         [Dxy, Dyy]]

    Returns div_J = ∂(Dxx*∂u/∂x + Dxy*∂u/∂y)/∂x
                  + ∂(Dxy*∂u/∂x + Dyy*∂u/∂y)/∂y
    """
    # ... implementation ...

def apply_diffusion_step(state: dict, D_tensor: dict, dx: float, dy: float, dt: float):
    """Apply diffusion to voltage field."""
    div_J = compute_diffusion(
        state['u'],
        D_tensor['Dxx'], D_tensor['Dyy'], D_tensor['Dxy'],
        dx, dy
    )
    state['u'] += dt * div_J
```

### 3.4 `mesh.py` - Geometry and Mesh

```python
"""2D cardiac mesh with fiber fields and tissue masks."""

@dataclass
class Mesh2D:
    """2D structured mesh for cardiac simulation."""

    Lx: float          # Domain width [mm]
    Ly: float          # Domain height [mm]
    nx: int            # Grid points in x
    ny: int            # Grid points in y
    dx: float          # Grid spacing x [mm]
    dy: float          # Grid spacing y [mm]

    # Coordinate arrays
    X: np.ndarray      # X coordinates (ny, nx)
    Y: np.ndarray      # Y coordinates (ny, nx)

    # Fiber field (angles in radians)
    fiber_angle: np.ndarray  # Fiber direction at each point

    # Tissue mask (True = healthy, False = scar)
    tissue_mask: np.ndarray

    # Diffusion tensor components
    Dxx: np.ndarray
    Dyy: np.ndarray
    Dxy: np.ndarray

def create_mesh(Lx, Ly, dx, D_parallel, D_perp, fiber_angle=0.0) -> Mesh2D:
    """Create uniform mesh with constant fiber direction."""
    ...

def create_mesh_with_infarct(Lx, Ly, dx, D_parallel, D_perp,
                              infarct_center, infarct_radius) -> Mesh2D:
    """Create mesh with circular infarct region."""
    ...
```

### 3.5 `simulation.py` - Core Engine

```python
"""Main simulation engine with operator splitting."""

class Simulation:
    """2D cardiac electrophysiology simulation."""

    def __init__(self, mesh: Mesh2D, ionic_model: IonicModel, dt: float = 0.01):
        self.mesh = mesh
        self.ionic = ionic_model
        self.dt = dt
        self.t = 0.0

        # Initialize state
        self.state = ionic_model.initialize_state((mesh.ny, mesh.nx))

    def step(self, I_stim: np.ndarray = None):
        """Advance simulation by one time step."""
        if I_stim is None:
            I_stim = np.zeros((self.mesh.ny, self.mesh.nx))

        # Operator splitting: Diffusion then Reaction
        apply_diffusion_step(self.state, self.mesh.D_tensor,
                            self.mesh.dx, self.mesh.dy, self.dt)
        self.ionic.ionic_step(self.state, self.dt, I_stim)

        self.t += self.dt

    def run(self, t_end: float, stim_func=None, save_every_ms=2.0):
        """Run simulation and return history."""
        ...

    def reset(self):
        """Reset to initial conditions."""
        self.state = self.ionic.initialize_state((self.mesh.ny, self.mesh.nx))
        self.t = 0.0
```

### 3.6 `stimulus.py` - Stimulus Protocols

```python
"""Stimulus protocols for cardiac simulation."""

def point_stimulus(mesh: Mesh2D, center: tuple, radius: float,
                   amplitude: float, t: float, t_start: float, duration: float) -> np.ndarray:
    """Create circular point stimulus."""
    ...

def pulse_train(mesh: Mesh2D, location: str, amplitude: float,
                duration: float, times: list[float]) -> callable:
    """Create pulse train stimulus function."""
    def stim_func(t):
        ...
    return stim_func

def s1s2_protocol(mesh: Mesh2D, s1_times: list, s2_time: float,
                  s1_location: str, s2_location: str) -> callable:
    """Create S1-S2 stimulation protocol for reentry induction."""
    ...
```

### 3.7 `visualization.py` - Plotting and Animation

```python
"""Visualization tools for cardiac simulation."""

def plot_voltage_snapshot(sim: Simulation, ax=None, cmap='turbo', **kwargs):
    """Plot current voltage state."""
    ...

def plot_time_trace(times: np.ndarray, V_history: np.ndarray,
                    points: list[tuple], ax=None):
    """Plot voltage traces at specific points."""
    ...

def create_animation(times: np.ndarray, V_history: np.ndarray,
                     mesh: Mesh2D, ionic: IonicModel,
                     skip_frames: int = 5, interval: int = 30) -> FuncAnimation:
    """Create matplotlib animation from simulation history."""
    ...

def save_animation(anim: FuncAnimation, path: str, fps: int = 30, dpi: int = 100):
    """Save animation to file (gif, mp4, etc.)."""
    ...

def plot_phase_portrait(u: np.ndarray, v: np.ndarray, ax=None):
    """Plot phase portrait of voltage vs gate variable."""
    ...
```

### 3.8 `interactive.py` - Real-time Mode

```python
"""Interactive real-time simulation with click-to-stimulate."""

class InteractiveSimulation:
    """Real-time cardiac simulation with mouse interaction."""

    def __init__(self, mesh: Mesh2D, ionic_model: IonicModel, **kwargs):
        self.sim = Simulation(mesh, ionic_model, **kwargs)
        self.stim_amplitude = 30.0
        self.stim_radius = 3.0
        self.pending_stimuli = []

    def run(self):
        """Start interactive simulation loop."""
        # Setup matplotlib figure with callbacks
        # Main animation loop
        ...

    def on_click(self, event):
        """Handle mouse click - add stimulus."""
        ...

    def on_key(self, event):
        """Handle keyboard input."""
        ...
```

### 3.9 `analysis.py` - Post-processing

```python
"""Analysis tools for cardiac simulation results."""

def measure_apd(t: np.ndarray, V: np.ndarray, threshold: float = 0.9) -> float:
    """Measure APD90 from voltage trace."""
    ...

def measure_cv(times: np.ndarray, V_history: np.ndarray,
               mesh: Mesh2D, direction: str = 'x') -> float:
    """Measure conduction velocity from activation times."""
    ...

def compute_activation_map(times: np.ndarray, V_history: np.ndarray,
                           threshold: float = 0.5) -> np.ndarray:
    """Compute activation time at each grid point."""
    ...

def compute_apd_restitution(sim: Simulation, di_range: np.ndarray) -> tuple:
    """Compute APD restitution curve (APD vs DI)."""
    ...

def compute_cv_restitution(sim: Simulation, di_range: np.ndarray) -> tuple:
    """Compute CV restitution curve (CV vs DI)."""
    ...
```

---

## 4. Module Dependency Graph

```
                    parameters.py
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
     ionic/base.py    mesh.py    diffusion.py
            │            │            │
            ▼            │            │
  ionic/fenton_karma.py  │            │
            │            │            │
            └────────────┼────────────┘
                         ▼
                   simulation.py
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    stimulus.py   visualization.py   analysis.py
         │               │
         └───────┬───────┘
                 ▼
          interactive.py
```

---

## 5. Implementation Order

### Phase 1: Core (Foundation)
```
1. parameters.py       - FK parameters + physical constants
2. ionic/base.py       - Abstract ionic model interface
3. ionic/fenton_karma.py - FK implementation + Numba kernel
4. diffusion.py        - Diffusion solver (port from V2)
5. mesh.py             - 2D mesh (port from V2)
```

### Phase 2: Simulation
```
6. simulation.py       - Core simulation class
7. stimulus.py         - Stimulus protocols
8. tests/test_ionic.py - Validate single-cell AP
```

### Phase 3: Visualization & Interaction
```
9. visualization.py    - Plotting and animation
10. interactive.py     - Click-to-stimulate mode
11. tests/test_cv.py   - Validate conduction velocity
```

### Phase 4: Analysis & Examples
```
12. analysis.py        - APD/CV measurement, restitution
13. examples/          - Demo scripts
14. tests/test_restitution.py - Validate restitution curves
```

---

## 6. Key Design Decisions

### 6.1 Swappable Ionic Models
The `IonicModel` base class allows easy swapping:
```python
# Use FK model
ionic = FentonKarmaModel(fk_params)
sim = Simulation(mesh, ionic)

# Or use AP model for comparison
ionic = AlievPanfilovModel(ap_params)
sim = Simulation(mesh, ionic)
```

### 6.2 State as Dictionary
State stored as dict for flexibility:
```python
state = {
    'u': np.ndarray,  # Voltage
    'v': np.ndarray,  # Gate 1
    'w': np.ndarray,  # Gate 2 (FK only)
}
```

### 6.3 Numba Kernels Separate
Heavy computation in Numba-jitted functions, called from class methods:
```python
@numba.jit(nopython=True, cache=True)
def _fk_ionic_kernel(u, v, w, dt, I_stim, params):
    # Pure Numba - no Python objects
    ...

class FentonKarmaModel:
    def ionic_step(self, state, dt, I_stim):
        _fk_ionic_kernel(state['u'], state['v'], state['w'],
                        dt, I_stim, self._numba_params)
```

### 6.4 Clean Separation
- **ionic/**: Only ionic model math
- **diffusion.py**: Only spatial coupling
- **simulation.py**: Orchestration only
- **visualization.py**: Only plotting

---

## 7. File Count Summary

| Category | Files | Purpose |
|----------|-------|---------|
| Core | 6 | Foundation modules |
| Simulation | 2 | Engine + stimulus |
| Visualization | 2 | Plotting + interactive |
| Analysis | 1 | Post-processing |
| Tests | 5 | Validation |
| Examples | 5 | Demos |
| Docs | 3 | Documentation |
| **Total** | **24** | |

This is a significant reduction from V2's 25+ Python files (many debug scripts) while adding proper organization.

---

## 8. Quick Start Example

```python
# examples/planar_wave.py
from engine_v3 import (
    create_mesh,
    FentonKarmaModel,
    FKParameters,
    Simulation,
    pulse_train,
    create_animation
)

# Setup
params = FKParameters()
mesh = create_mesh(Lx=80, Ly=80, dx=0.5, D_parallel=1.0, D_perp=0.5)
ionic = FentonKarmaModel(params)

# Create simulation
sim = Simulation(mesh, ionic, dt=0.01)

# Run with left-edge stimulus
stim = pulse_train(mesh, location='left', amplitude=30,
                   duration=2, times=[5.0])
times, history = sim.run(t_end=200, stim_func=stim, save_every_ms=2)

# Visualize
anim = create_animation(times, history, mesh, ionic)
anim.save('planar_wave.gif')
```

---

**Author**: Generated with Claude Code
**Date**: 2025-12-10
**Status**: Architecture Proposed - Ready for Review
