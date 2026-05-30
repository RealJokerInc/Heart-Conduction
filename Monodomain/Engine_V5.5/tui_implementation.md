# TUI Implementation Design Document

> Design specification for the `cardiac_sim` interactive terminal interface.
> A guided wizard that walks users through simulation setup, shows live progress, and summarizes results.
> Wraps the existing V5.4 Python API — no engine modifications needed.

---

## 1. Vision & Interface Philosophy

### What It Is

An **interactive terminal wizard** — not a flag-based CLI. The user launches it, gets guided through choices with helpful descriptions, watches the simulation run with a live progress display, and gets a clean results summary at the end.

Three screens: **Setup → Run → Results**

### What It Feels Like

```
$ cardiac-sim

╭─ Cardiac Simulation Engine v5.4 ─────────────────────────────╮
│                                                               │
│  Welcome! This wizard will guide you through setting up       │
│  and running a cardiac electrophysiology simulation.          │
│                                                               │
│  Navigate: [Enter] next  [b] back  [q] quit                  │
│                                                               │
╰───────────────────────────────────────────────────────────────╯

? Choose simulation paradigm:
  ❯ Classical (FEM / FDM / FVM)
    LBM (Lattice-Boltzmann)
```

### Who It's For

1. **Researchers** — compare solver configurations without writing boilerplate Python
2. **Students** — run textbook scenarios with guided parameter choices
3. **Quick exploration** — faster than scripting when you want to try one run

### Design Principles

| Principle | Rationale |
|-----------|-----------|
| **Guided, not guessed** | Every choice has a description explaining what it means and what the default does |
| **Sensible defaults** | Pressing Enter through every prompt produces a valid, reasonable simulation |
| **Always reversible** | User can go back to any previous step and change their mind |
| **Show what you chose** | A review panel before running displays the full config — no surprises |
| **Clean output** | Rich-rendered panels, progress bars, and tables — not raw print statements |

---

## 2. Application Flow

### Three-Screen Architecture

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│  SETUP  │────▶│   RUN   │────▶│ RESULTS │
│ (wizard)│     │(progress)│     │(summary)│
└─────────┘     └─────────┘     └─────────┘
     ▲  │
     └──┘  (back/forward between steps)
```

### Setup Screen — Step Flow

The setup wizard is a linear sequence of prompted steps. The user can go back to any previous step at any time.

**Classical path:**
```
Step 1: Paradigm        → Classical / LBM
Step 2: Geometry         → Domain size (Lx, Ly), resolution (Nx, Ny)
Step 3: Discretization   → FEM / FDM / FVM
Step 4: Physics          → D, chi, Cm (with smart defaults)
Step 5: Ionic Model      → TTP06 / ORd, cell type (EPI/ENDO/M_CELL)
Step 6: Solvers          → Splitting, ionic solver, diffusion solver, linear solver
Step 7: Stimulus         → Region shape, timing, amplitude (repeatable — add multiple)
Step 8: Time & Output    → t_end, dt, save_every, output file
Step 9: Review           → Full config panel — confirm or go back
```

**LBM path** (branches at Step 1):
```
Step 1: Paradigm        → LBM
Step 2: Geometry         → Ny, Nx, dx
Step 3: Physics          → D, chi, Cm, dt
Step 4: Collision        → BGK / MRT
Step 5: Ionic Model      → TTP06 / ORd, cell type
Step 6: Ionic Solver     → Rush-Larsen / Forward Euler
Step 7: Stimulus         → Region shape, timing, amplitude
Step 8: Time & Output    → t_end, save_every, output file
Step 9: Review           → Full config panel
```

### Navigation

| Key | Action |
|-----|--------|
| `Enter` | Accept current choice / value, go to next step |
| `b` | Go back one step |
| `q` | Quit (with confirmation) |
| `↑` / `↓` | Navigate within a selection list |
| Number keys | Quick-select from numbered options |

### Run Screen

Once the user confirms at the Review step, the interface transitions to the Run screen:

```
╭─ Running Simulation ─────────────────────────────────────────╮
│                                                               │
│  Classical · FDM · TTP06 (EPI) · Strang · CN+PCG             │
│  Grid: 100×100 (2.5cm × 2.5cm)  dt=0.02ms                   │
│                                                               │
│  Progress ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸━━━━━  72%           │
│  t = 360.0 / 500.0 ms    elapsed: 01:23    ETA: 00:32        │
│                                                               │
╰───────────────────────────────────────────────────────────────╯
```

- Uses `rich.progress` (not raw tqdm) for the progress bar
- Shows elapsed time and ETA
- Config summary line at top so user remembers what's running
- `Ctrl+C` gracefully stops and still saves whatever frames have been collected

### Results Screen

```
╭─ Simulation Complete ────────────────────────────────────────╮
│                                                               │
│  Duration:  500.0 ms in 01:55 wall time                       │
│  Frames:    500 saved (every 1.0 ms)                          │
│  Output:    ./sim_output_20260205_143022.npz                  │
│                                                               │
│  Voltage range:  -86.2 mV → +32.1 mV                         │
│  Activation:     first node at t=1.2 ms                       │
│                                                               │
│  Contents of .npz:                                            │
│    times      (500,)         float64                          │
│    voltages   (500, 10000)   float64                          │
│    config     dict           simulation parameters            │
│                                                               │
╰───────────────────────────────────────────────────────────────╯

? What next?
  ❯ Exit
    Run again with same settings
    Run again with modifications (back to Setup)
```

---

## 3. Classical Simulation — Setup Steps in Detail

Each step below describes: what the user sees, what they choose, the default, and how it maps to the engine API.

### Step 1: Paradigm

```
? Choose simulation paradigm:
  ❯ Classical (FEM / FDM / FVM)
      Solve the monodomain equation using traditional numerical methods.
      Supports multiple discretization schemes and linear solvers.

    LBM (Lattice-Boltzmann)
      Solve diffusion via streaming and collision on a lattice.
      No linear system solves — GPU-friendly, explicit method.
```

**Default:** Classical
**Maps to:** Decides which setup path (Steps 2-8) to follow.

### Step 2: Geometry

```
? Domain size
  Lx (cm) [2.5]: █
  Ly (cm) [2.5]: █

? Grid resolution
  Nx (nodes in x) [100]: █
  Ny (nodes in y) [100]: █

  → Grid spacing: dx=0.0253 cm, dy=0.0253 cm  (2,500 µm)
  → Total nodes: 10,000
```

- Numeric text input with defaults in brackets
- Grid spacing and total node count computed and shown immediately as feedback
- Validation: Nx, Ny >= 3; Lx, Ly > 0

**Defaults:** Lx=2.5, Ly=2.5, Nx=100, Ny=100
**Maps to:** `StructuredGrid.create_rectangle(Lx, Ly, Nx, Ny)` or `TriangularMesh.create_rectangle(Lx, Ly, Nx, Ny)` depending on Step 3.

### Step 3: Discretization Scheme

```
? Choose spatial discretization:
  ❯ FDM — Finite Difference Method
      9-point stencil on structured grid. Fast, simple.
      Best for: rectangular domains with uniform spacing.

    FVM — Finite Volume Method
      TPFA with harmonic mean at interfaces. Conservative.
      Best for: heterogeneous tissue (scar, fibrosis).

    FEM — Finite Element Method
      P1 triangular elements. Flexible geometry.
      Best for: complex domain shapes (future mesh import).
```

**Default:** FDM
**Maps to:** `FDMDiscretization(grid, D, chi, Cm)` / `FVMDiscretization(grid, D, chi, Cm)` / `FEMDiscretization(mesh, D, chi, Cm)`

**Note:** FEM creates a `TriangularMesh` from the same Lx/Ly/Nx/Ny. FDM and FVM create a `StructuredGrid`.

### Step 4: Physics Parameters

```
? Tissue physics
  Diffusion coefficient D (cm²/ms) [0.001]: █
  Surface-to-volume ratio χ (cm⁻¹) [1400]: █
  Membrane capacitance Cm (µF/cm²) [1.0]: █

  These defaults model healthy ventricular myocardium.
  Reduce D to simulate ischemic or fibrotic tissue.
```

**Defaults:** D=0.001, chi=1400, Cm=1.0
**Maps to:** Passed directly to discretization constructor.

### Step 5: Ionic Model

```
? Choose ionic model:
  ❯ TTP06 — ten Tusscher-Panfilov 2006
      17 state variables, 12 gates. Ventricular action potential.
      Fast, well-validated. Recommended for most uses.

    ORd — O'Hara-Rudy 2011
      41 state variables. Detailed CaMKII signaling.
      Slower. Use for drug block or restitution studies.

? Cell type:
  ❯ EPI (epicardial)
    ENDO (endocardial)
    M_CELL (mid-myocardial)
```

**Defaults:** TTP06, EPI
**Maps to:** `ionic_model='ttp06'`, `cell_type='EPI'`

### Step 6: Solver Configuration

```
? Operator splitting:
  ❯ Strang (second-order)  [recommended]
    Godunov (first-order)

? Ionic solver:
  ❯ Rush-Larsen (exponential integrator)  [recommended]
    Forward Euler

? Diffusion solver:
  ❯ Crank-Nicolson (implicit, O(dt²))  [recommended]
    BDF1 (implicit, O(dt))
    BDF2 (implicit, O(dt²))
    Forward Euler (explicit — requires small dt)
    RK2 (explicit, O(dt²))
    RK4 (explicit, O(dt⁴))

? Linear solver:                          ← only shown for implicit diffusion
  ❯ PCG (preconditioned conjugate gradient)  [recommended]
    Chebyshev (polynomial, no sync — GPU-friendly)
    DCT (spectral — uniform grid only)
```

**Defaults:** Strang, Rush-Larsen, Crank-Nicolson, PCG
**Maps to:** `splitting='strang'`, `ionic_solver='rush_larsen'`, `diffusion_solver='crank_nicolson'`, `linear_solver='pcg'`

**Conditional logic:**
- If diffusion solver is explicit (FE/RK2/RK4) → skip linear solver question
- If discretization is FEM → hide DCT/FFT options (spectral solvers need structured grid)

### Step 7: Stimulus

This is the most complex step — users need to define spatial regions, timing, and amplitude, and they may want multiple stimuli.

```
? Add a stimulus? [Y/n]: █

? Stimulus region shape:
  ❯ Left edge
      Depolarize the left boundary. Classic planar wave setup.
    Rectangle
      Define a rectangular region by corner coordinates.
    Circle
      Define a circular region by center and radius.
    Point
      Small focal stimulus at a single location.

? Left edge width (cm) [0.1]: █

? Timing
  Start time (ms) [0.0]: █
  Duration (ms) [1.0]: █
  Amplitude (µA/µF) [-52.0]: █

  ℹ Negative amplitude depolarizes (convention: outward current positive)

╭─ Stimuli defined ─────────────────────────╮
│  1. Left edge (w=0.1cm) at t=0.0ms, 1.0ms │
│     duration, amplitude=-52.0 µA/µF       │
╰───────────────────────────────────────────╯

? Add another stimulus? [y/N]: █
```

**Region types and their parameters:**

| Region | Parameters | Maps to |
|--------|-----------|---------|
| Left edge | `width` | `left_edge_region(width)` |
| Rectangle | `x0, y0, x1, y1` | `rectangular_region(x0, y0, x1, y1)` |
| Circle | `cx, cy, radius` | `circular_region(cx, cy, radius)` |
| Point | `cx, cy` | `point_stimulus(cx, cy)` |

**Pacing protocols** (shown if user adds a stimulus):
```
? Pacing protocol:
  ❯ Single stimulus
    Regular pacing (repeat at BCL)
        → additional: BCL (ms), number of beats
    S1-S2 protocol
        → additional: n_s1, BCL, S2 coupling interval
```

**Default:** One left-edge stimulus at t=0, duration 1ms, amplitude -52.0 µA/µF.

### Step 8: Time & Output

```
? Simulation timing
  End time (ms) [500.0]: █
  Time step dt (ms) [0.02]: █
  Save interval (ms) [1.0]: █

  → 25,000 time steps, 500 output frames
  → Estimated output size: ~38 MB

? Output file [./sim_output.npz]: █

? Device:
  ❯ CPU
    CUDA (GPU)      ← only shown if torch.cuda.is_available()
    MPS (Apple GPU)  ← only shown if torch.backends.mps.is_available()
```

**Defaults:** t_end=500, dt=0.02, save_every=1.0, output=`./sim_output.npz`, device=auto-detect
**Maps to:** `sim.run(t_end, save_every)`, `np.savez()`

### Step 9: Review & Confirm

```
╭─ Simulation Configuration ───────────────────────────────────╮
│                                                               │
│  Paradigm       Classical                                     │
│  Discretization FDM (100×100, dx=0.025cm)                     │
│  Domain         2.5cm × 2.5cm (10,000 nodes)                  │
│  Physics        D=0.001 cm²/ms, χ=1400 cm⁻¹, Cm=1.0 µF/cm²  │
│                                                               │
│  Ionic model    TTP06 (EPI)                                   │
│  Splitting      Strang                                        │
│  Ionic solver   Rush-Larsen                                   │
│  Diffusion      Crank-Nicolson + PCG                          │
│                                                               │
│  Stimulus       1× left edge (w=0.1cm, t=0ms, -52 µA/µF)     │
│  Duration       500ms (dt=0.02ms, save every 1.0ms)           │
│  Output         ./sim_output.npz                              │
│  Device         CPU                                           │
│                                                               │
╰───────────────────────────────────────────────────────────────╯

? Ready to run? [Y/n/b(ack to edit)]: █
```

---

## 4. LBM Simulation — Setup Steps in Detail

After choosing LBM at Step 1, the wizard follows a shorter path.

### Step 2: Geometry

```
? Grid dimensions
  Nx (columns) [100]: █
  Ny (rows) [100]: █
  Grid spacing dx (cm) [0.025]: █

  → Domain: 2.475cm × 2.475cm
  → Total nodes: 10,000
```

**Defaults:** Nx=100, Ny=100, dx=0.025
**Maps to:** `LBMSimulation(Ny, Nx, dx, ...)`

### Step 3: Physics

```
? Physics parameters
  Diffusion coefficient D (cm²/ms) [0.001]: █
  Time step dt (ms) [0.01]: █
  χ (cm⁻¹) [1400]: █
  Cm (µF/cm²) [1.0]: █

  → τ = 0.525 (stable: τ > 0.5 ✓)
```

- Computes and shows τ immediately as feedback
- Warns (and blocks proceeding) if τ ≤ 0.5

**Maps to:** `LBMSimulation(..., D=D, dt=dt, chi=chi, Cm=Cm)`

### Step 4: Collision Operator

```
? Collision operator:
  ❯ BGK (single relaxation time)
      Simple and fast. Good for isotropic diffusion.

    MRT (multiple relaxation time)
      Better numerical stability. Required for anisotropic diffusion.
```

**Default:** BGK
**Maps to:** `collision='bgk'` or `collision='mrt'` (uses `create_isotropic_bgk()` / MRT factory)

### Steps 5-9

Same as Classical Steps 5, 7, 8, 9 (Ionic Model, Stimulus, Time & Output, Review), except:
- No diffusion solver / linear solver questions
- Stimulus masks are 2D grid-shaped (Ny, Nx) instead of coordinate-based

---

## 5. Stimulus Specification — Design Details

Stimulus is the hardest UX problem because it combines spatial geometry, temporal protocol, and amplitude — all from text input.

### Strategy: Preset Shapes + Protocols

Rather than asking for raw mask tensors, the wizard offers **named shapes** with intuitive parameters:

```python
# What the wizard builds internally:
region_fn = left_edge_region(width=0.1)       # from regions.py
stim = Stimulus(region=region_fn, start_time=0.0, duration=1.0, amplitude=-52.0)
protocol.add_stimulus(region_fn, start_time=0.0, duration=1.0, amplitude=-52.0)
```

### Multiple Stimuli

The wizard loops: after each stimulus definition, it shows a numbered list of all defined stimuli and asks "Add another?". The user can also **delete** a previously added stimulus by number:

```
╭─ Stimuli (2 defined) ────────────────────────────────────────╮
│  1. Left edge (w=0.1cm) — single at t=0ms, -52 µA/µF        │
│  2. Circle (0.5,0.5 r=0.2cm) — pacing BCL=1000ms ×10 beats  │
╰───────────────────────────────────────────────────────────────╯

? [a]dd / [d]elete / [Enter] done: █
```

### Pacing Protocol Integration

Each stimulus gets a protocol choice:

```
? Pacing pattern for this stimulus:
  ❯ Single pulse
    Regular pacing
        BCL (ms) [1000]: █
        Number of beats [1]: █
    S1-S2 protocol
        Number of S1 beats [10]: █
        S1 BCL (ms) [1000]: █
        S2 coupling interval (ms) [300]: █
```

**Maps to:** `protocol.add_stimulus()`, `protocol.add_regular_pacing()`, or `protocol.add_s1s2_protocol()`

---

## 6. Output Specification

### .npz Format

Every simulation saves a single `.npz` file containing:

```python
np.savez(
    output_path,
    times=times,           # (n_frames,)       float64 — time in ms
    voltages=voltages,     # (n_frames, n_dof)  float64 — membrane voltage in mV
    config=config_dict,    # dict — all simulation parameters (JSON-serializable)
)
```

### Config Dict Contents

```python
config = {
    'paradigm': 'classical',          # or 'lbm'
    'discretization': 'fdm',
    'Lx': 2.5, 'Ly': 2.5, 'Nx': 100, 'Ny': 100,
    'dx': 0.02525..., 'dy': 0.02525...,
    'D': 0.001, 'chi': 1400.0, 'Cm': 1.0,
    'ionic_model': 'ttp06', 'cell_type': 'EPI',
    'splitting': 'strang',
    'ionic_solver': 'rush_larsen',
    'diffusion_solver': 'crank_nicolson',
    'linear_solver': 'pcg',
    'dt': 0.02, 't_end': 500.0, 'save_every': 1.0,
    'stimuli': [
        {'type': 'left_edge', 'width': 0.1, 'start': 0.0,
         'duration': 1.0, 'amplitude': -52.0, 'protocol': 'single'}
    ],
    'device': 'cpu',
    'engine_version': '5.4.0',
}
```

This dict is stored as a numpy object array inside the .npz so it can be loaded back:
```python
data = np.load('sim_output.npz', allow_pickle=True)
config = data['config'].item()
```

### Auto-naming

If the user doesn't specify a filename:
```
sim_output_20260205_143022.npz     # timestamp-based
```

---

## 7. Progress & Feedback (Run Screen)

### Rich Progress Display

```python
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.live import Live
from rich.panel import Panel

# During simulation:
with Progress(
    SpinnerColumn(),
    "[progress.description]{task.description}",
    BarColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    TimeElapsedColumn(),
    TimeRemainingColumn(),
) as progress:
    task = progress.add_task("Simulating...", total=n_steps)
    for state in sim.run(t_end, save_every):
        # save frame
        progress.advance(task, advance=steps_per_save)
```

### Graceful Ctrl+C

```python
import signal

# Register handler that sets a flag
# On next yield from sim.run(), check flag → break → save partial results
```

Output on interrupt:
```
╭─ Simulation Interrupted ─────────────────────────────────────╮
│  Stopped at t=234.5 ms (46.9% complete)                      │
│  Saved 234 frames to ./sim_output.npz                         │
╰───────────────────────────────────────────────────────────────╯
```

---

## 8. Wizard State Machine

### Implementation Pattern

The wizard is a **state machine** with numbered steps. Each step is a function that:
1. Renders its prompt(s) using Rich
2. Collects user input
3. Returns the collected value(s) or a "back" signal

```python
class WizardState:
    """Holds all config choices made so far."""
    paradigm: str = 'classical'
    Lx: float = 2.5
    Ly: float = 2.5
    Nx: int = 100
    Ny: int = 100
    # ... all other fields with defaults
```

```python
def run_wizard():
    state = WizardState()
    steps = get_steps_for_paradigm(state.paradigm)  # list of step functions
    current = 0

    while current < len(steps):
        result = steps[current](state)

        if result == 'back':
            current = max(0, current - 1)
        elif result == 'quit':
            return None
        else:
            # Step updates state in-place
            current += 1

            # If paradigm changed at step 0, rebuild step list
            if current == 1:
                steps = get_steps_for_paradigm(state.paradigm)

    return state  # Complete config
```

### Step Functions

Each step follows a consistent pattern:

```python
def step_geometry(state: WizardState) -> str:
    """Step 2: Geometry configuration."""
    console.print(Panel("Step 2 of 9: Geometry", style="bold blue"))

    state.Lx = FloatPrompt.ask("Lx (cm)", default=state.Lx)
    state.Ly = FloatPrompt.ask("Ly (cm)", default=state.Ly)
    state.Nx = IntPrompt.ask("Nx", default=state.Nx)
    state.Ny = IntPrompt.ask("Ny", default=state.Ny)

    # Show computed feedback
    dx = state.Lx / (state.Nx - 1)
    console.print(f"  → Grid spacing: {dx:.4f} cm, Total: {state.Nx * state.Ny:,} nodes")

    return prompt_nav()  # returns 'next', 'back', or 'quit'

def prompt_nav() -> str:
    """Common navigation prompt."""
    choice = Prompt.ask("[Enter] next / [b]ack / [q]uit", default="next")
    if choice.lower().startswith('b'):
        return 'back'
    elif choice.lower().startswith('q'):
        return 'quit'
    return 'next'
```

---

## 9. Implementation Plan

### File Structure

```
cardiac_sim/
├── tui/
│   ├── __init__.py          # Entry point: main()
│   ├── wizard.py            # Wizard state machine + WizardState dataclass
│   ├── steps_classical.py   # Step functions for classical path
│   ├── steps_lbm.py         # Step functions for LBM path
│   ├── steps_common.py      # Shared steps (ionic model, stimulus, time/output, review)
│   ├── runner.py            # Build simulation from WizardState, run with progress
│   └── display.py           # Rich rendering helpers (panels, tables, prompts)
├── __main__.py              # python -m cardiac_sim → tui.main()
```

### Build Order

| Order | File | Purpose | Test |
|-------|------|---------|------|
| 1 | `display.py` | Rich helpers, prompt wrappers, navigation | Manual visual check |
| 2 | `wizard.py` | WizardState dataclass, state machine loop | Unit test: step navigation |
| 3 | `steps_common.py` | Shared steps (ionic, stimulus, time, review) | Unit test: defaults populate |
| 4 | `steps_classical.py` | Classical-specific steps (geometry, scheme, solver) | Unit test: state mutation |
| 5 | `steps_lbm.py` | LBM-specific steps | Unit test: state mutation |
| 6 | `runner.py` | WizardState → Simulation → run with Rich progress → save .npz | Integration test: runs to completion |
| 7 | `__init__.py` + `__main__.py` | Entry points | `python -m cardiac_sim` works |

### Entry Points

```python
# cardiac_sim/__main__.py
from .tui import main
main()

# cardiac_sim/tui/__init__.py
from rich.console import Console

console = Console()

def main():
    """Entry point for cardiac-sim command."""
    from .wizard import run_wizard
    from .runner import run_simulation

    config = run_wizard()
    if config is None:
        return  # User quit

    run_simulation(config)
```

```python
# setup.py
entry_points={
    'console_scripts': [
        'cardiac-sim=cardiac_sim.tui:main',
    ],
},
```

---

## 10. Dependencies

### New Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `rich` | >=13.0 | Terminal rendering (panels, tables, prompts, progress) |

### Dropped from Original Plan

| Package | Why dropped |
|---------|-------------|
| `click` | Not needed — no flag-based CLI, wizard handles all input |
| `tqdm` | Replaced by `rich.progress` (same library handles rendering + progress) |

### Already Required

- `torch` (engine core)
- `numpy` (output)

---

## 11. Future Enhancements (Not in V1)

- **Config file support**: `cardiac-sim --config my_run.yaml` to skip wizard (batch mode)
- **Preset scenarios**: `cardiac-sim --preset 1d_cable` for common textbook setups
- **Live voltage heatmap**: ASCII or sixel rendering of V during simulation
- **Multi-run comparisons**: Side-by-side results from different solver configs

---

*This document is the design spec. Implementation follows after user approval.*
