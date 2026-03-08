"""Shared parameters and utilities for cross-validation Phase 6 tests.

All Phase 6 tests import from here to guarantee identical parameters.
"""

import sys
import os
import time
import math

# Path setup: add both engine roots
_HERE = os.path.dirname(os.path.abspath(__file__))
_BIDOMAIN_ROOT = os.path.join(_HERE, '..')
_LBM_ROOT = os.path.join(_HERE, '..', '..', '..', 'Monodomain', 'LBM_V1')

if _BIDOMAIN_ROOT not in sys.path:
    sys.path.insert(0, _BIDOMAIN_ROOT)
if _LBM_ROOT not in sys.path:
    sys.path.insert(0, _LBM_ROOT)

import torch
torch.set_default_dtype(torch.float64)


# ============================================================
# Physical Parameters
# ============================================================
SIGMA_I = 1.74       # mS/cm (intracellular longitudinal)
SIGMA_E = 6.25       # mS/cm (extracellular)
CHI_REAL = 1400.0    # cm^-1 (real surface-to-volume ratio)
CM_REAL = 1.0        # uF/cm^2 (real membrane capacitance)

D_I = SIGMA_I / (CHI_REAL * CM_REAL)  # 0.00124 cm^2/ms
D_E = SIGMA_E / (CHI_REAL * CM_REAL)  # 0.00446 cm^2/ms
D_EFF = D_I * D_E / (D_I + D_E)      # 0.000970 cm^2/ms

# Formulation B: chi is absorbed into D = sigma/(chi*Cm).
# Only Cm appears in the numerical operators (source term scaling).
CM_NUM = 1.0


# ============================================================
# Domain Parameters
# ============================================================
NX, NY = 150, 40
DX = 0.025            # cm (250 um)
DT = 0.01             # ms
LX = DX * (NX - 1)    # 3.725 cm
LY = DX * (NY - 1)    # 0.975 cm


# ============================================================
# Stimulus Parameters
# ============================================================
STIM_COLS = 5          # Left 5 columns
STIM_WIDTH = STIM_COLS * DX  # 0.125 cm
STIM_START = 1.0       # ms
STIM_DUR = 2.0         # ms
STIM_AMP = -80.0       # uA/uF (strong depolarizing)


# ============================================================
# Simulation Parameters
# ============================================================
T_END = 40.0           # ms
SAVE_EVERY = 0.5       # ms


# ============================================================
# CV Measurement Parameters
# ============================================================
X1, X2 = 30, 80       # x-indices for CV measurement
Y_CENTER = NY // 2     # y=20 (interior)
Y_EDGE = 1            # y=1 (first interior row)
THRESHOLD = -30.0      # mV (activation threshold)


# ============================================================
# Derived Constants
# ============================================================
KLEBER_RATIO = math.sqrt((D_I + D_E) / D_E)  # ~1.131


# ============================================================
# CV Measurement Utility
# ============================================================
def measure_cv_from_history(V_history, times, y, dx=DX, x1=X1, x2=X2,
                            threshold=THRESHOLD):
    """Measure conduction velocity between two x-positions at row y.

    Parameters
    ----------
    V_history : list of (Nx, Ny) tensors
        Voltage snapshots
    times : list of float
        Corresponding time points (ms)
    y : int
        Row index for measurement
    dx : float
        Spatial resolution (cm)
    x1, x2 : int
        x-indices for measurement (x2 > x1)
    threshold : float
        Activation threshold (mV)

    Returns
    -------
    cv : float
        Conduction velocity (cm/ms). NaN if wave didn't reach.
    """
    t1 = None
    t2 = None
    for V, t in zip(V_history, times):
        if t1 is None and V[x1, y].item() > threshold:
            t1 = t
        if t2 is None and V[x2, y].item() > threshold:
            t2 = t
        if t1 is not None and t2 is not None:
            break

    if t1 is None or t2 is None or t2 <= t1:
        return float('nan')

    distance = abs(x2 - x1) * dx
    return distance / (t2 - t1)


def is_nan(x):
    """Check if a float is NaN (works without numpy)."""
    return x != x


# ============================================================
# LBM Simulation Builder
# ============================================================
def build_lbm_sim(nx, ny, dx, dt, D, lattice, stim_cols=STIM_COLS,
                  stim_start=STIM_START, stim_dur=STIM_DUR,
                  stim_amp=STIM_AMP):
    """Build an LBMSimulation with standard parameters.

    Returns the LBMSimulation instance (call .run() to execute).
    """
    from ionic.ttp06.model import TTP06Model
    from ionic.base import CellType
    from src.simulation import LBMSimulation

    model = TTP06Model(cell_type=CellType.ENDO, device=torch.device('cpu'))
    sim = LBMSimulation(nx, ny, dx, dt, D, model, lattice=lattice)

    stim_mask = torch.zeros(nx, ny, dtype=torch.bool)
    stim_mask[:stim_cols, :] = True
    sim.add_stimulus(stim_mask, start=stim_start, duration=stim_dur,
                     amplitude=stim_amp)
    return sim


def run_lbm(nx=NX, ny=NY, dx=DX, dt=DT, D=D_EFF, lattice='d2q5',
            t_end=T_END, save_every=SAVE_EVERY, **kwargs):
    """Build and run LBM simulation, returning (times, V_history)."""
    sim = build_lbm_sim(nx, ny, dx, dt, D, lattice, **kwargs)
    return sim.run(t_end=t_end, save_every=save_every)


# ============================================================
# Bidomain Simulation Builder
# ============================================================
def build_bidomain_sim(nx, ny, dx, dt, D_i, D_e, bc_type='insulated',
                       Cm=CM_NUM, stim_width=STIM_WIDTH,
                       stim_start=STIM_START, stim_dur=STIM_DUR,
                       stim_amp=STIM_AMP, theta=0.5):
    """Build a BidomainSimulation with standard parameters.

    Parameters
    ----------
    bc_type : str
        'insulated' (Neumann all) or 'bath' (phi_e Dirichlet)

    Returns
    -------
    sim : BidomainSimulation
    grid : StructuredGrid (needed for flat_to_grid conversion)
    """
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec
    from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
    from cardiac_sim.simulation.classical.discretization.fdm import (
        BidomainFDMDiscretization)
    from cardiac_sim.tissue_builder.stimulus import (
        StimulusProtocol, left_edge_region)
    from cardiac_sim.simulation.classical.bidomain import BidomainSimulation

    Lx = dx * (nx - 1)
    Ly = dx * (ny - 1)

    if bc_type == 'insulated':
        boundary_spec = BoundarySpec.insulated()
    elif bc_type == 'bath':
        boundary_spec = BoundarySpec.bath_coupled()
    else:
        raise ValueError(f"Unknown bc_type: {bc_type}")

    grid = StructuredGrid(Nx=nx, Ny=ny, Lx=Lx, Ly=Ly,
                          boundary_spec=boundary_spec)
    cond = BidomainConductivity(D_i=D_i, D_e=D_e)
    spatial = BidomainFDMDiscretization(grid, cond, Cm=Cm)

    stimulus = StimulusProtocol()
    stimulus.add_stimulus(
        region=left_edge_region(width=stim_width),
        start_time=stim_start,
        duration=stim_dur,
        amplitude=stim_amp,
    )

    sim = BidomainSimulation(
        spatial=spatial,
        ionic_model='ttp06',
        stimulus=stimulus,
        dt=dt,
        splitting='strang',
        parabolic_solver='pcg',
        elliptic_solver='auto',
        theta=theta,
    )
    return sim, grid


def run_bidomain(nx=NX, ny=NY, dx=DX, dt=DT, D_i=D_I, D_e=D_E,
                 bc_type='insulated', t_end=T_END, save_every=SAVE_EVERY,
                 **kwargs):
    """Build and run bidomain simulation, returning (times, V_history).

    V_history is a list of (Nx, Ny) tensors (reshaped from flat).
    """
    sim, grid = build_bidomain_sim(nx, ny, dx, dt, D_i, D_e,
                                   bc_type=bc_type, **kwargs)
    times = []
    V_hist = []
    for state in sim.run(t_end=t_end, save_every=save_every):
        times.append(state.t)
        V_grid = grid.flat_to_grid(state.Vm)
        V_hist.append(V_grid.clone())
    return times, V_hist


# ============================================================
# Monodomain FDM Control (V5.3-style explicit Euler)
# ============================================================
def run_monodomain_fdm(nx=NX, ny=NY, dx=DX, dt=DT, D=D_EFF,
                       t_end=T_END, save_every=SAVE_EVERY,
                       stim_cols=STIM_COLS, stim_start=STIM_START,
                       stim_dur=STIM_DUR, stim_amp=STIM_AMP):
    """Run a simple monodomain FDM simulation (explicit diffusion + Rush-Larsen).

    Uses V5.3-style operator splitting:
      1. Ionic step (Rush-Larsen)
      2. Diffusion step (explicit Euler, 5-point stencil)

    Direct equation: dV/dt = D * nabla^2(V) - I_ion + I_stim
    """
    from cardiac_sim.ionic.ttp06.model import TTP06Model

    model = TTP06Model(device='cpu')
    V = torch.full((nx, ny), model.V_rest, dtype=torch.float64)
    S = model.get_initial_state(nx * ny).reshape(nx, ny, -1)

    # Stability check
    dt_max = dx * dx / (4 * D)
    if dt > dt_max:
        print(f"    WARNING: dt={dt} > CFL limit={dt_max:.4f}, using dt={dt_max*0.8:.4f}")
        dt = dt_max * 0.8

    alpha = D / (dx * dx)

    # Stimulus mask
    stim_mask = torch.zeros(nx, ny, dtype=torch.bool)
    stim_mask[:stim_cols, :] = True

    times = []
    V_hist = []
    t = 0.0
    next_save = save_every
    n_steps = int(t_end / dt + 0.5)

    for step_i in range(n_steps):
        # --- Ionic step (Rush-Larsen) ---
        V_flat = V.reshape(-1)
        S_flat = S.reshape(-1, S.shape[-1])

        Iion = model.compute_Iion(V_flat, S_flat)

        # Stimulus
        Istim = torch.zeros_like(V_flat)
        if stim_start <= t < stim_start + stim_dur:
            Istim[stim_mask.reshape(-1)] = stim_amp

        # Update V with ionic + stimulus
        V_flat = V_flat + dt * (-(Iion + Istim))

        # Rush-Larsen gate update
        gate_inf = model.compute_gate_steady_states(V_flat, S_flat)
        gate_tau = model.compute_gate_time_constants(V_flat, S_flat)
        for k, gi in enumerate(model.gate_indices):
            tau_k = gate_tau[:, k].clamp(min=1e-6)
            inf_k = gate_inf[:, k]
            S_flat[:, gi] = inf_k + (S_flat[:, gi] - inf_k) * torch.exp(-dt / tau_k)

        # Forward Euler concentration update
        conc_rates = model.compute_concentration_rates(V_flat, S_flat)
        for k, ci in enumerate(model.concentration_indices):
            S_flat[:, ci] = S_flat[:, ci] + dt * conc_rates[:, k]

        V = V_flat.reshape(nx, ny)
        S = S_flat.reshape(nx, ny, -1)

        # --- Diffusion step (explicit Euler, 5-pt stencil, Neumann BCs) ---
        V_pad = torch.nn.functional.pad(
            V.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate'
        ).squeeze()
        lap = (V_pad[1:-1, 2:] + V_pad[1:-1, :-2] +
               V_pad[2:, 1:-1] + V_pad[:-2, 1:-1] -
               4 * V_pad[1:-1, 1:-1])
        V = V + dt * alpha * lap

        t += dt

        if t >= next_save - 1e-12:
            next_save += save_every
            times.append(t)
            V_hist.append(V.clone())

    return times, V_hist


# ============================================================
# Result Formatting
# ============================================================
def format_cv(cv):
    """Format CV value as cm/s string."""
    if is_nan(cv):
        return "N/A"
    return f"{cv * 1000:.1f}"


def print_config_result(name, cv_center, cv_edge, expected_ratio="?"):
    """Print a single config's results."""
    ratio = cv_edge / cv_center if not (is_nan(cv_center) or is_nan(cv_edge)) \
        else float('nan')
    ratio_str = f"{ratio:.4f}" if not is_nan(ratio) else "N/A"
    print(f"  {name:<35} {format_cv(cv_center):>8} cm/s  "
          f"{format_cv(cv_edge):>8} cm/s  ratio={ratio_str}  "
          f"(expect {expected_ratio})")
    return ratio


def timed_run(name, run_fn):
    """Execute a run function with timing and error handling."""
    print(f"\n--- {name} ---")
    t0 = time.time()
    try:
        result = run_fn()
        elapsed = time.time() - t0
        print(f"    Completed in {elapsed:.1f}s")
        return result
    except Exception as e:
        elapsed = time.time() - t0
        print(f"    FAILED after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return None
