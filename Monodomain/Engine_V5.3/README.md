# Engine V5.3: Cardiac Electrophysiology Simulation Engine

High-performance cardiac electrophysiology simulation with **multiple ionic models**, **CPU/GPU backend abstraction**, **LUT optimization**, and PyTorch acceleration.

## Key Features

| Feature | Description |
|---------|-------------|
| **Ionic Models** | O'Hara-Rudy 2011 (41 states) + ten Tusscher-Panfilov 2006 (19 states) |
| **Backend System** | Unified CPU/GPU toggle with auto-detection |
| **LUT Optimization** | Pre-computed gating functions (~4× speedup) |
| **Spatial Methods** | FDM (explicit) for 2D tissue simulations |
| **Time Integration** | Godunov operator splitting with Rush-Larsen |
| **Visualization** | OpenCV-based real-time animation |

## Recent Updates (V5.3.1)

### Backend Abstraction System

New unified CPU/GPU backend system (`utils/backend.py`):

```python
from utils import Backend, get_backend, cuda_available

# Auto-detect best device (CUDA if available)
backend = Backend(device='auto')

# Force specific device
cpu_backend = Backend(device='cpu')
gpu_backend = Backend(device='cuda')

# Create tensors on backend
x = backend.zeros(100, 100)
y = backend.linspace(-80, 40, 1000)

# Global backend access
device = get_device()
dtype = get_dtype()
```

**Backend Features:**
- Auto-detection of CUDA availability
- Device-aware tensor creation (`zeros`, `ones`, `linspace`, `tensor`, etc.)
- Memory management utilities (`synchronize`, `empty_cache`, `memory_allocated`)
- Device info reporting (`DeviceInfo` dataclass)

### TTP06 Model with LUT

The ten Tusscher-Panfilov 2006 model now includes LUT acceleration:

```python
from ionic import TTP06Model, CellType

# With LUT (default, ~4× faster)
model = TTP06Model(celltype=CellType.EPI, device='cuda', use_lut=True)

# Without LUT (for debugging/validation)
model_direct = TTP06Model(celltype=CellType.EPI, use_lut=False)
```

### Spiral Wave Examples

Two interactive spiral wave simulations using S1-S2 cross-field protocol:

| Script | Model | Description |
|--------|-------|-------------|
| `examples/spiral_wave_s1s2.py` | TTP06 | GPU-accelerated with LUT |
| `examples/spiral_wave_ord.py` | ORd | 41-state model, direct computation |

**Controls:**
- `S` - Start S1 stimulus (plane wave from left edge)
- `SPACE` - Apply S2 in vulnerable window
- `R` - Reset simulation
- `+/-` - Adjust speed
- `Q/ESC` - Quit

## Architecture

```
Engine_V5.3/
├── ionic/                    # Cellular ionic models
│   ├── base.py              # IonicModel abstract base class
│   ├── lut.py               # Lookup table implementation
│   ├── ord/                 # O'Hara-Rudy 2011 model
│   │   ├── model.py         # ORdModel class
│   │   ├── gating.py        # Voltage-dependent gating
│   │   ├── currents.py      # Ion current calculations
│   │   ├── calcium.py       # Ca²⁺ handling
│   │   ├── camkii.py        # CaMKII signaling
│   │   └── parameters.py    # StateIndex, CellType, ORdParameters
│   └── ttp06/               # ten Tusscher-Panfilov 2006 model
│       ├── model.py         # TTP06Model class (with LUT support)
│       ├── gating.py        # Voltage-dependent gating
│       ├── currents.py      # Ion current calculations
│       ├── calcium.py       # Ca²⁺ handling
│       └── parameters.py    # StateIndex, CellType, TTP06Parameters
│
├── utils/                   # Utility modules
│   ├── __init__.py          # Public API exports
│   └── backend.py           # CPU/GPU backend abstraction
│
├── fem/                     # Finite Element infrastructure (future)
│   ├── mesh.py              # TriangularMesh class
│   └── assembly.py          # Mass & stiffness matrix assembly
│
├── solver/                  # Time integration
│   ├── linear.py            # Linear solvers
│   └── time_stepping.py     # Time stepping schemes
│
├── tissue/                  # Tissue-level simulation
│   ├── simulation.py        # MonodomainSimulation
│   └── stimulus.py          # Stimulus protocols
│
├── examples/                # Example scripts
│   ├── spiral_wave_s1s2.py  # TTP06 spiral wave (GPU + LUT)
│   ├── spiral_wave_ord.py   # ORd spiral wave
│   └── validate_stage1.py   # Single-cell validation
│
├── tests/                   # Test suite
│   ├── test_backend.py      # Backend CPU/GPU tests
│   ├── test_lut_comprehensive.py  # LUT validation
│   └── ...
│
├── README.md
└── IMPLEMENTATION.md
```

## Requirements

- Python 3.9+
- PyTorch 2.0+ (with CUDA support recommended)
- NumPy, SciPy
- OpenCV (for visualization)
- Matplotlib (optional)

## Installation

```bash
# From project root
cd "Monodomain/Engine_V5.3"
source ../../venv/bin/activate

# Install dependencies (if needed)
pip install torch numpy scipy opencv-python matplotlib
```

## Quick Start

### Single Cell Simulation

```python
from ionic import ORdModel, TTP06Model, CellType

# ORd model (41 states)
model = ORdModel(celltype=CellType.ENDO, device='cuda')
state = model.get_initial_state()

# TTP06 model with LUT (19 states, faster)
model = TTP06Model(celltype=CellType.EPI, device='cuda', use_lut=True)
state = model.get_initial_state()

# Run simulation
dt = 0.01  # ms
for t in range(50000):
    I_stim = -80.0 if 10.0 <= t*dt < 11.0 else 0.0
    state = model.step(state, dt, I_stim)
    V = model.get_voltage(state)
```

### 2D Tissue Simulation (FDM)

```python
# Run interactive spiral wave simulation
# TTP06 version:
python examples/spiral_wave_s1s2.py --domain 16.0 --dx 0.02

# ORd version:
python examples/spiral_wave_ord.py --domain 16.0 --dx 0.02
```

### Backend Testing

```python
# Run backend tests
python tests/test_backend.py

# Output:
# Engine V5.3 Backend Test Suite
# ============================================================
# Available Devices:
#   CPU: Available
#   CUDA:0: NVIDIA GeForce RTX 4080 (12.4 GB)
# ...
# ALL TESTS PASSED!
```

## Numerical Methods

### Monodomain Equation

```
χ·Cm·∂V/∂t = -χ·Iion(V, u) + ∇·(D·∇V) + Istim
```

### Current Spatial Method (FDM)

The current implementation uses **Finite Difference Method (FDM)** with explicit Euler:

- **5-point stencil** Laplacian for isotropic diffusion
- **Neumann (no-flux)** boundary conditions
- **Stability limit**: dt ≤ dx²/(4D)

```python
# FDM diffusion step
V_new = V + dt * D * laplacian(V)
```

### Operator Splitting (Godunov)

```
1. Ionic step:  states = ionic_model.step(states, dt, I_stim)
2. Diffusion:   V = V + dt * D * ∇²V
```

## Performance

### Backend Comparison (Single Cell, 1000 steps)

| Backend | ms/step | Notes |
|---------|---------|-------|
| CPU | ~0.15 | Baseline |
| CUDA | ~0.02 | ~7× faster |

### LUT Speedup (TTP06)

| Mode | Relative Speed |
|------|----------------|
| Direct computation | 1.0× |
| LUT enabled | ~4× |

## Validation

### Single Cell AP Characteristics

| Model | APD90 (ms) | V_max (mV) | V_rest (mV) |
|-------|------------|------------|-------------|
| ORd ENDO | ~280 | +40 | -87 |
| TTP06 EPI | ~300 | +35 | -86 |

### Conduction Velocity

Target: 40-60 cm/s (isotropic, D ≈ 0.00154 cm²/ms)

## References

1. O'Hara T, et al. (2011). "Simulation of the Undiseased Human Cardiac Ventricular Action Potential." PLoS Comput Biol.

2. ten Tusscher KHWJ, Panfilov AV (2006). "Alternans and spiral breakup in a human ventricular tissue model." Am J Physiol Heart Circ Physiol.

3. Rush S, Larsen H (1978). "A practical algorithm for solving dynamic membrane equations." IEEE Trans Biomed Eng.

4. openCARP. "The openCARP Simulation Environment for Cardiac Electrophysiology." [Website](https://opencarp.org/)

## License

MIT License
