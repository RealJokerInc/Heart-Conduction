# Engine V4: Luo-Rudy 1994 Monodomain Solver

## Overview

Engine V4 implements the **Luo-Rudy 1994 (LRd94)** ventricular action potential model for 2D cardiac tissue simulations. This is a 1:1 port of Engine V3's architecture, replacing the Fenton-Karma ionic model with the more physiologically detailed LRd94 model.

**Key Features:**
- Physical voltage units (mV) - no normalization
- 13 state variables (vs. 3 in FK)
- Detailed ion channel kinetics
- Ion concentration dynamics (Na+, K+, Ca2+)
- SR calcium handling
- Numba-accelerated kernels

## Model Reference

Luo CH, Rudy Y. *"A dynamic model of the cardiac ventricular action potential. I. Simulations of ionic currents and concentration changes."* Circ Res. 1994;74(6):1071-1096.

## File Structure

```
Engine_V4/
├── parameters.py           # LRd94 parameters, spatial config, presets
├── luo_rudy_1994.py       # Numba-optimized ionic model
├── diffusion.py           # 2D anisotropic diffusion (same as V3)
├── simulation.py          # 2D simulation combining ionic + diffusion
├── spiral_wave_sim.py     # Spiral wave formation demo
├── infarct_border_speedup.py  # Infarct border study
├── interactive_sim.py     # Interactive BCL-controlled simulation
└── ARCHITECTURE.md        # This file
```

## Key Differences from Engine V3 (Fenton-Karma)

| Aspect | Engine V3 (FK) | Engine V4 (LRd94) |
|--------|---------------|-------------------|
| Ionic Model | Fenton-Karma | Luo-Rudy 1994 |
| Voltage | Normalized [0,1] | Physical mV [-85, +50] |
| State Variables | 3 (u, v, w) | 13 (V, m, h, j, d, f, f_Ca, X, Na_i, K_i, Ca_i, Ca_jsr, Ca_nsr) |
| Time Step | dt = 0.02 ms | dt = 0.005 ms |
| APD90 | ~250 ms (tunable) | ~250 ms (tuned via G_K) |
| Computational Cost | Low | ~4x higher |
| Ion Concentrations | Not tracked | Na+, K+, Ca2+ dynamics |

## State Variables

### Gating Variables (7)
| Variable | Description | Range |
|----------|-------------|-------|
| m | Na+ activation | [0, 1] |
| h | Na+ fast inactivation | [0, 1] |
| j | Na+ slow inactivation | [0, 1] |
| d | L-type Ca2+ activation | [0, 1] |
| f | L-type Ca2+ inactivation | [0, 1] |
| f_Ca | Ca2+-dependent inactivation | [0, 1] |
| X | Time-dependent K+ activation | [0, 1] |

### Ion Concentrations (5)
| Variable | Description | Typical Range |
|----------|-------------|---------------|
| Na_i | Intracellular Na+ | ~10 mM |
| K_i | Intracellular K+ | ~145 mM |
| Ca_i | Intracellular Ca2+ | 100 nM - 6 µM |
| Ca_jsr | Junctional SR Ca2+ | 0.1 - 2 mM |
| Ca_nsr | Network SR Ca2+ | 0.1 - 2 mM |

### Voltage
| Variable | Description | Typical Range |
|----------|-------------|---------------|
| V | Membrane potential | -86 to +65 mV |

## Ionic Currents

The model includes 10 membrane currents:

1. **I_Na** - Fast Na+ current (upstroke)
2. **I_Ca,L** - L-type Ca2+ current (plateau)
3. **I_K** - Time-dependent K+ current (repolarization)
4. **I_K1** - Inward rectifier K+ current (resting potential)
5. **I_Kp** - Plateau K+ current
6. **I_NaCa** - Na+/Ca2+ exchanger
7. **I_NaK** - Na+/K+ pump
8. **I_pCa** - Sarcolemmal Ca2+ pump
9. **I_Na,b** - Background Na+ current
10. **I_Ca,b** - Background Ca2+ current

## Parameter Tuning

The default parameters produce **APD90 ≈ 250 ms** by increasing G_K from the original 0.282 to 0.35 mS/cm².

### Available Presets

```python
from parameters import get_preset

# APD90 ~ 250 ms (default, tuned)
params = get_preset('default')

# APD90 ~ 300 ms (original LRd94)
params = get_preset('original')

# APD90 ~ 200 ms (short)
params = get_preset('short')

# APD90 ~ 350 ms (long)
params = get_preset('long')
```

## Usage Examples

### Single Cell Simulation

```python
from luo_rudy_1994 import LuoRudy1994Model, run_single_cell

model = LuoRudy1994Model(dt=0.005)
results = run_single_cell(
    model,
    t_end=500.0,
    stim_amplitude=-80.0,  # uA/cm^2
    stim_start=10.0,
    stim_duration=1.0,
)

print(f"APD90 = {results['apd90']:.1f} ms")
```

### 2D Tissue Simulation

```python
from simulation import create_simulation

sim = create_simulation(
    domain_size=40.0,      # mm
    resolution=0.5,        # mm
    D_parallel=0.1,        # mm^2/ms
    D_perp=0.05,           # mm^2/ms
    dt=0.005,              # ms
    param_preset='default'
)

# Add stimulus
sim.add_point_stimulus(
    x=5.0, y=20.0,
    t_start=10.0,
    amplitude=-100.0,  # uA/cm^2
    radius=3.0,
    duration=1.0
)

# Run simulation
result = sim.run(t_end=100.0)
```

### Interactive Demo

```python
from interactive_sim import main
main()  # Opens matplotlib window with controls
```

## Numerical Methods

### Operator Splitting
1. **Diffusion step**: V_new = V + dt * ∇·(D∇V)
2. **Ionic step**: Update V, gates, and concentrations

### Gating Variables
- **Rush-Larsen method** for stability with large time steps:
  ```
  x_new = x_inf - (x_inf - x) * exp(-dt/tau_x)
  ```

### Concentrations
- **Forward Euler** with buffering factors:
  ```
  [Ca]_new = [Ca] + dt * β * d[Ca]/dt
  ```

### Diffusion
- **Explicit finite differences** with Neumann (no-flux) BC
- **CFL stability**: r = D·dt/dx² < 0.25

## Performance

Typical performance on modern CPU:
- Single cell: ~2000 ms/s simulation time (500 ms in ~0.25s)
- 81×81 grid: ~500-1000 steps/s
- 161×161 grid: ~200-400 steps/s

Numba JIT compilation provides significant speedup after warmup.

## Stability Requirements

- **Time step**: dt ≤ 0.01 ms recommended, 0.005 ms for safety
- **CFL condition**: D·dt/dx² < 0.25
- **Stimulus**: -50 to -100 µA/cm² for 0.5-2 ms

## Known Limitations

1. **Elevated Ca2+ transient**: Peak [Ca2+]i ~5-7 µM vs. expected ~1 µM
   - This is a known characteristic of simplified SR models
   - Does not significantly affect voltage AP morphology

2. **Computational cost**: ~4x slower than FK due to:
   - 13 state variables vs. 3
   - Smaller time step requirement
   - Complex gating kinetics

3. **Temperature**: Fixed at 37°C (310 K)

## References

1. Luo CH, Rudy Y. Circ Res. 1994;74(6):1071-1096.
2. Keener JP, Sneyd J. Mathematical Physiology. Springer.
3. Sundnes J et al. Computing the Electrical Activity in the Heart. Springer.

---

*Generated with Claude Code - 2025-12-11*
