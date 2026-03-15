# Engine V5.1: PyTorch GPU Cardiac Simulation

High-performance cardiac electrophysiology simulation using PyTorch GPU acceleration.

## Features

- **O'Hara-Rudy 2011 (ORd)** ventricular myocyte model
- **Monodomain** tissue-level electrical propagation
- **GPU-accelerated** using PyTorch CUDA (5-10x speedup over CPU)
- **Float64 precision** for numerical stability
- **CV-based parameters** for mesh-independent conduction velocity

## Requirements

- Python 3.9+
- PyTorch 2.0+ with CUDA support
- NVIDIA GPU with CUDA capability

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy matplotlib
```

## Quick Start

### Single Cell Simulation

```python
from ionic import ORdModel, CellType

# Create model
model = ORdModel(celltype=CellType.ENDO)

# Get initial state
state = model.get_initial_state()

# Run simulation
dt = 0.01  # ms
for t in range(50000):  # 500 ms
    Istim = -80.0 if 10.0 <= t*dt < 11.0 else 0.0
    state = model.step(state, dt, Istim)
```

### Tissue Simulation (Using MeshBuilder)

```python
from tissue import MeshBuilder
from ionic import CellType

# Create 15x15 cm tissue with pre-tuned diffusion coefficients
# Default: anisotropic (CV_long=0.6 m/s, CV_trans=0.3 m/s)
mesh = MeshBuilder.create_default()

# Or for spiral waves, use isotropic (CV=0.6 m/s all directions)
mesh = MeshBuilder.create_default(anisotropic=False)

# Create simulation from mesh
sim = mesh.create_simulation(celltype=CellType.ENDO)

# Add stimulus at left edge
sim.add_stimulus(
    region=(slice(None), slice(0, 3)),
    start_time=1.0,
    duration=2.0
)

# Run simulation
t, V = sim.run(t_end=100.0, dt=0.02, save_interval=1.0)
```

### Pre-tuned Diffusion Coefficients

The MeshBuilder uses empirically calibrated diffusion coefficients:

| Target CV | D (cm²/ms) | Calibration |
|-----------|------------|-------------|
| 0.6 m/s | 0.002161 | 1D cable simulation |
| 0.3 m/s | 0.000819 | 1D cable simulation |

These values were tuned at dx=0.02 cm, dt=0.02 ms using the ORd model.

## Architecture

```
Engine_V5.1/
├── ionic/           # Cellular ionic model
│   ├── model.py     # ORdModel class
│   ├── gating.py    # Voltage-dependent gating
│   ├── currents.py  # Ion current calculations
│   ├── calcium.py   # Ca2+ handling
│   └── camkii.py    # CaMKII signaling
├── tissue/          # Tissue-level simulation
│   ├── mesh.py      # MeshBuilder & CableMesh (CV tuning)
│   ├── diffusion.py # FVM diffusion operator
│   └── simulation.py# MonodomainSimulation
├── tests/           # Validation tests
└── examples/        # Example scripts
    └── spiral_wave_s1s2.py  # S1-S2 spiral induction
```

## Performance

| Grid Size | Cells | V5 (CPU) | V5.1 (GPU) | Speedup |
|-----------|-------|----------|------------|---------|
| 100x100 | 10K | 1.2 ms | 0.2 ms | 6x |
| 250x250 | 62.5K | 7.5 ms | 1.0 ms | 7.5x |
| 500x500 | 250K | 30 ms | 4 ms | 7.5x |
| 1000x1000 | 1M | 120 ms | 12 ms | 10x |

## Validation

V5.1 is validated against V5 (CPU Numba implementation):

- Single cell AP: < 1 mV difference
- APD90: < 1% difference
- Conduction velocity: < 5% error from target

## Backward Compatibility

Load V5 states for comparison:

```python
import numpy as np
from utils.validation import load_v5_state

# Load V5 state (numpy array)
v5_state = np.load('v5_checkpoint.npy')

# Convert to V5.1 (PyTorch GPU tensor)
v51_state = load_v5_state(v5_state)
```

## Implementation Details

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for:

- Mathematical formulation
- Numerical methods (Rush-Larsen, FVM)
- Validation procedures
- Stage-by-stage implementation plan

## References

1. O'Hara T, et al. (2011). "Simulation of the Undiseased Human Cardiac Ventricular Action Potential." PLoS Comput Biol.

2. Rush S, Larsen H (1978). "A practical algorithm for solving dynamic membrane equations." IEEE Trans Biomed Eng.

## License

MIT License
