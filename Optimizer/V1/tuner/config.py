"""
Optimizer V1 — Configuration and Parameter Registry

Core dataclasses for tuning targets, parameter registries, and scaling.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch


# ============================================================================
# Tuning Targets
# ============================================================================

@dataclass
class TuningTargets:
    """Target biomarker values for optimization."""
    apd_90: float = 350.0           # ms (single-cell)
    cv_longitudinal: float = 15.0   # cm/s
    cv_transverse: float = 7.5      # cm/s
    dvdt_max: float = 25.0          # V/s
    v_rest: float = -74.0           # mV (PHAS13 native, don't tune)
    v_peak_min: float = 20.0        # mV (constraint, not objective)
    spontaneous_cl: Optional[float] = 1200.0  # ms (optional target)
    restitution: Optional[List[Tuple[float, float]]] = None  # (DI_ms, APD_ms)
    # Hard constraints (A1) — reject solutions outside these bounds
    dvdt_max_upper: float = 60.0    # V/s — reject if dV/dt > this
    v_peak_max: float = 60.0        # mV — reject if V_peak > this
    v_rest_range: Tuple[float, float] = (-92.0, -70.0)  # mV


# ============================================================================
# Tuning Config
# ============================================================================

@dataclass
class TuningConfig:
    """Configuration for an optimization run."""
    ionic_model: str = 'mhas13'     # 'mhas13' | 'phas13'
    # Backend for single-cell AP evaluation (P-1 backend unification):
    #   'cardiac_core' (default) — AP on the cardiac_core hook path, one model with
    #                              the tissue-CV path (kinetics axis identifiable).
    #   'cardiac_sim'  — legacy V5.4 batched path (cell_runner), kept for parity tests.
    ionic_backend: str = 'cardiac_core'
    tier: int = 2                   # Parameter tier (1=6, 2=10, 3=14 params)
    seed: int = 42                  # Reproducibility seed
    device: str = 'cuda'
    dt: float = 0.02                # ms — tissue dt (CFL: dx²/4D)
    dt_cell: float = 0.2            # ms — single-cell dt (validated: <1% Vpeak drift vs 0.05)
    n_beats: int = 10               # Beats to pace before measuring
    pacing_cl: float = 1000.0       # ms (pacing cycle length)
    stim_amplitude: float = -40.0   # A/F (MHAS13 threshold ~-15 A/F)
    stim_duration: float = 2.0      # ms
    n_initial: int = 0              # BoTorch initial points (0 = auto)
    n_iterations: int = 200         # BoTorch evaluations
    dtype: torch.dtype = torch.float64
    # Tissue parameters (from spiral_wave_s1s2)
    dx_cm: float = 0.04             # cm (400 μm, matches spiral_wave_s1s2)
    cable_length_cm: float = 1.5    # cm (1D cable for CV — shorter = faster)
    # Engine selection
    engine: str = 'monodomain'      # 'monodomain' | 'bidomain' | 'lbm'
    # Bidomain parameters (from cv_shared.py: sigma_i=1.74, sigma_e=6.25, chi=1400, Cm=1.0)
    De_Di_ratio: float = 3.597      # D_e / D_i ratio (physiological default)
    bc_type: str = 'insulated'      # 'insulated' | 'bath'
    bidomain_splitting: str = 'strang'
    elliptic_solver: str = 'auto'
    # cardiac_core / chip fields (Phase 1+). NOTE: cc_runner builds effective-D
    # meshes with chi=1.0 (the FDM operator divides by chi); use dx_cm=0.01 for
    # cardiac_core's CN+PCG (coarser over-depolarizes the stim site).
    stim_start: float = 1.0          # ms — stimulus onset
    dt_lbm: float = 0.01             # ms — LBM time step (may exceed dt to keep MRT tau off 0.5)
    anisotropy_ratio: float = 2.0    # CV_L / CV_T (Bursac & Parker ~2.1)
    domain_mm: float = 16.0          # chip square side (within the 25 mm coverslip)
    dx_mm: float = 0.1               # chip resolution (0.1 mm)
    baseline: str = 'nrvm'           # 'nrvm' | 'hipsc'


# ============================================================================
# PHAS13 Parameter Registry
# ============================================================================

@dataclass
class ParamSpec:
    """Specification for a single tunable parameter."""
    attr_name: str          # Attribute name on PHAS13Parameters
    published: float        # Published (default) value
    tier: int               # 1, 2, or 3
    bounds: Tuple[float, float]  # Scaling factor bounds


# Registry: param_name -> ParamSpec
PHAS13_REGISTRY: Dict[str, ParamSpec] = {
    # Tier 1 — Core conductances (6 params)
    'g_Na':   ParamSpec('g_Na',   3.6712302,    1, (0.5, 2.0)),
    'g_CaL':  ParamSpec('g_CaL',  8.635702e-5,  1, (0.3, 2.0)),
    'g_Kr':   ParamSpec('g_Kr',   0.0298667,    1, (0.5, 3.0)),
    'g_Ks':   ParamSpec('g_Ks',   0.002041,     1, (0.3, 2.5)),
    'g_K1':   ParamSpec('g_K1',   0.0281492,    1, (0.3, 2.0)),
    'g_to':   ParamSpec('g_to',   0.0299038,    1, (0.3, 2.5)),

    # Tier 2 — Extended (+4 = 10 params)
    'kNaCa':  ParamSpec('kNaCa',  4900.0,       2, (0.3, 2.5)),
    'PNaK':   ParamSpec('PNaK',   1.841424,     2, (0.5, 2.0)),
    'g_pCa':  ParamSpec('g_pCa',  0.4125,       2, (0.3, 2.5)),
    'VmaxUp': ParamSpec('VmaxUp', 5.6064e-4,    2, (0.3, 2.5)),

    # Tier 3 — Full (+4 = 14 params)
    'g_f':    ParamSpec('g_f',    0.03010312,   3, (0.3, 3.0)),
    'g_bNa':  ParamSpec('g_bNa',  0.0009,       3, (0.2, 3.0)),
    'g_bCa':  ParamSpec('g_bCa',  0.00069264,   3, (0.2, 3.0)),
    'V_leak': ParamSpec('V_leak', 4.4444e-7,    3, (0.2, 3.0)),
}


# Tissue parameters (always included, separate from ionic tiers)
TISSUE_PARAMS = {
    'D_long': (0.00005, 0.001),     # cm^2/ms
    'D_trans': (0.000025, 0.0005),   # cm^2/ms
}


# Na-kinetic axes (P1.5) — per-INSTANCE MHAS13 attributes (identity 1.0 / 0.0),
# applied in the gate hooks (compute_gate_*), NOT step(). These reshape I_Na in TIME
# so the joint fit can decouple dV/dt (peak I_Na) from CV (charge-to-sink), which
# conductance scaling alone cannot (architecture §5). Registered here so
# decision_space imports FROM config (never the reverse). PHAS13 is unaffected.
# Bounds are on the multiplier (tau_*_scale) / mV shift (v_half_shift). τ_m is the
# primary decoupling knob; the rest are the "if needed" set (architecture §9-P1.5).
KINETIC_REGISTRY = {
    'tau_m_scale':  (0.5, 3.0),
    'tau_h_scale':  (0.5, 3.0),
    'tau_j_scale':  (0.5, 3.0),
    'v_half_shift': (-10.0, 10.0),   # mV
}


def get_params_for_tier(tier: int) -> Dict[str, ParamSpec]:
    """Return parameter specs for a given tier (cumulative)."""
    return {k: v for k, v in PHAS13_REGISTRY.items() if v.tier <= tier}


def get_bounds_tensor(tier: int, dtype=torch.float64) -> torch.Tensor:
    """Return (2, n_params) bounds tensor for BoTorch."""
    params = get_params_for_tier(tier)
    lowers = [spec.bounds[0] for spec in params.values()]
    uppers = [spec.bounds[1] for spec in params.values()]
    return torch.tensor([lowers, uppers], dtype=dtype)


def get_param_names(tier: int) -> List[str]:
    """Return ordered list of parameter names for a given tier."""
    return list(get_params_for_tier(tier).keys())


# ============================================================================
# Scaling Application
# ============================================================================

def apply_scaling(params, theta: Dict[str, float]):
    """
    Apply scaling factors to a PHAS13Parameters instance.

    Parameters
    ----------
    params : PHAS13Parameters
        Model parameters (modified in-place).
    theta : dict
        Mapping of param_name -> scaling_factor (e.g. {'g_Na': 0.8}).

    Returns
    -------
    PHAS13Parameters
        The modified params (same object, for chaining).
    """
    for name, scale in theta.items():
        if name not in PHAS13_REGISTRY:
            raise ValueError(f"Unknown parameter: {name}")
        spec = PHAS13_REGISTRY[name]
        published = spec.published
        setattr(params, spec.attr_name, published * scale)
    return params


def theta_to_dict(theta_tensor: torch.Tensor, tier: int) -> Dict[str, float]:
    """Convert a scaling factor tensor to a named dict."""
    names = get_param_names(tier)
    return {name: theta_tensor[i].item() for i, name in enumerate(names)}


def dict_to_theta(theta_dict: Dict[str, float], tier: int,
                  dtype=torch.float64) -> torch.Tensor:
    """Convert a named dict to a scaling factor tensor."""
    names = get_param_names(tier)
    return torch.tensor([theta_dict.get(n, 1.0) for n in names], dtype=dtype)
