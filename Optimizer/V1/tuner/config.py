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


# ============================================================================
# Tuning Config
# ============================================================================

@dataclass
class TuningConfig:
    """Configuration for an optimization run."""
    ionic_model: str = 'phas13'     # 'phas13' | 'ttp06'
    tier: int = 1                   # Parameter tier (1, 2, or 3)
    device: str = 'cuda'
    dt: float = 0.02                # ms — tissue dt (CFL: dx²/4D)
    dt_cell: float = 0.2            # ms — single-cell dt (validated: <1% Vpeak drift vs 0.05)
    n_beats: int = 10               # Beats to pace before measuring
    pacing_cl: float = 1000.0       # ms (pacing cycle length)
    stim_amplitude: float = -5.0    # A/F
    stim_duration: float = 2.0      # ms
    n_initial: int = 0              # BoTorch initial points (0 = auto)
    n_iterations: int = 200         # BoTorch evaluations
    dtype: torch.dtype = torch.float64
    # Tissue parameters (from spiral_wave_s1s2)
    dx_cm: float = 0.04             # cm (400 μm, matches spiral_wave_s1s2)
    cable_length_cm: float = 1.5    # cm (1D cable for CV — shorter = faster)


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
    'g_Ks':   ParamSpec('g_Ks',   0.002041,     1, (0.3, 3.0)),
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
