"""
Lookup Table (LUT) Acceleration for Ionic Models

Precomputes voltage-dependent gating functions over a voltage range
and provides fast linear interpolation for runtime evaluation.

Achieves 2-5x speedup for ionic model evaluation by replacing
expensive exp() and division operations with table lookups.

Reference: Sherwin et al. (2022) "Resource-Efficient Use of Modern
Processor Architectures For Numerically Solving Cardiac Ionic Cell Models"
"""

import torch
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass


@dataclass
class LUTConfig:
    """Configuration for lookup table."""
    V_min: float = -100.0      # Minimum voltage (mV)
    V_max: float = 80.0        # Maximum voltage (mV)
    n_points: int = 2001       # Number of table points (0.09 mV resolution)
    device: torch.device = None
    dtype: torch.dtype = torch.float64


class LookupTable:
    """
    GPU-optimized lookup table for voltage-dependent functions.

    Precomputes function values over voltage range and uses linear
    interpolation for fast evaluation. Handles edge cases by clamping
    to table bounds.

    Parameters
    ----------
    config : LUTConfig
        Configuration specifying voltage range and resolution
    """

    def __init__(self, config: Optional[LUTConfig] = None):
        if config is None:
            config = LUTConfig()
        if config.device is None:
            config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.config = config
        self.V_min = config.V_min
        self.V_max = config.V_max
        self.n_points = config.n_points
        self.device = config.device
        self.dtype = config.dtype

        # Precompute voltage grid
        self.dV = (self.V_max - self.V_min) / (self.n_points - 1)
        self.inv_dV = 1.0 / self.dV

        # Voltage grid for table construction
        self.V_grid = torch.linspace(
            self.V_min, self.V_max, self.n_points,
            device=self.device, dtype=self.dtype
        )

        # Table storage
        self.tables: Dict[str, torch.Tensor] = {}

    def add_function(self, name: str, func: Callable[[torch.Tensor], torch.Tensor]):
        """
        Add a voltage-dependent function to the LUT.

        Parameters
        ----------
        name : str
            Function identifier
        func : callable
            Function that takes voltage tensor and returns values
        """
        with torch.no_grad():
            self.tables[name] = func(self.V_grid)

    def lookup(self, name: str, V: torch.Tensor) -> torch.Tensor:
        """
        Look up interpolated values for given voltages.

        Parameters
        ----------
        name : str
            Function identifier
        V : torch.Tensor
            Voltage values (mV)

        Returns
        -------
        torch.Tensor
            Interpolated function values
        """
        table = self.tables[name]

        # Compute index and fractional part
        idx_float = (V - self.V_min) * self.inv_dV
        idx_float = torch.clamp(idx_float, 0, self.n_points - 1.0001)

        idx = idx_float.long()
        frac = idx_float - idx.float()

        # Linear interpolation
        return table[idx] * (1.0 - frac) + table[torch.clamp(idx + 1, max=self.n_points - 1)] * frac

    def lookup_batch(self, names: List[str], V: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Look up multiple functions at once (more efficient).

        Parameters
        ----------
        names : list of str
            Function identifiers
        V : torch.Tensor
            Voltage values

        Returns
        -------
        dict
            Function name -> interpolated values
        """
        # Compute index once
        idx_float = (V - self.V_min) * self.inv_dV
        idx_float = torch.clamp(idx_float, 0, self.n_points - 1.0001)

        idx = idx_float.long()
        idx_p1 = torch.clamp(idx + 1, max=self.n_points - 1)
        frac = idx_float - idx.float()
        one_minus_frac = 1.0 - frac

        return {
            name: self.tables[name][idx] * one_minus_frac + self.tables[name][idx_p1] * frac
            for name in names
        }

    @property
    def memory_bytes(self) -> int:
        """Total memory used by tables (bytes)."""
        return sum(t.element_size() * t.numel() for t in self.tables.values())

    @property
    def memory_mb(self) -> float:
        """Total memory used by tables (MB)."""
        return self.memory_bytes / (1024 * 1024)


class TTP06LUT(LookupTable):
    """
    Precomputed lookup tables for TTP06 model gating functions.

    Supports both EPI and ENDO cell types with cell-type specific
    Ito inactivation kinetics.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64
    ):
        config = LUTConfig(device=device, dtype=dtype)
        super().__init__(config)

        # Import gating functions
        from .ttp06.gating import (
            INa_m_inf, INa_m_tau, INa_h_inf, INa_h_tau, INa_j_inf, INa_j_tau,
            Ito_r_inf, Ito_r_tau,
            Ito_s_inf_endo, Ito_s_inf_epi, Ito_s_tau_endo, Ito_s_tau_epi,
            ICaL_d_inf, ICaL_d_tau, ICaL_f_inf, ICaL_f_tau,
            ICaL_f2_inf, ICaL_f2_tau,
            IKr_Xr1_inf, IKr_Xr1_tau, IKr_Xr2_inf, IKr_Xr2_tau,
            IKs_Xs_inf, IKs_Xs_tau
        )

        # Build all tables
        self.add_function('m_inf', INa_m_inf)
        self.add_function('m_tau', INa_m_tau)
        self.add_function('h_inf', INa_h_inf)
        self.add_function('h_tau', INa_h_tau)
        self.add_function('j_inf', INa_j_inf)
        self.add_function('j_tau', INa_j_tau)

        self.add_function('r_inf', Ito_r_inf)
        self.add_function('r_tau', Ito_r_tau)
        self.add_function('s_inf_endo', Ito_s_inf_endo)
        self.add_function('s_inf_epi', Ito_s_inf_epi)
        self.add_function('s_tau_endo', Ito_s_tau_endo)
        self.add_function('s_tau_epi', Ito_s_tau_epi)

        self.add_function('d_inf', ICaL_d_inf)
        self.add_function('d_tau', ICaL_d_tau)
        self.add_function('f_inf', ICaL_f_inf)
        self.add_function('f_tau', ICaL_f_tau)
        self.add_function('f2_inf', ICaL_f2_inf)
        self.add_function('f2_tau', ICaL_f2_tau)

        self.add_function('Xr1_inf', IKr_Xr1_inf)
        self.add_function('Xr1_tau', IKr_Xr1_tau)
        self.add_function('Xr2_inf', IKr_Xr2_inf)
        self.add_function('Xr2_tau', IKr_Xr2_tau)

        self.add_function('Xs_inf', IKs_Xs_inf)
        self.add_function('Xs_tau', IKs_Xs_tau)

    def get_all_gating(self, V: torch.Tensor, celltype_is_endo: bool = True) -> Dict[str, torch.Tensor]:
        """
        Get all gating steady-state and time constants at once.

        Parameters
        ----------
        V : torch.Tensor
            Voltage values
        celltype_is_endo : bool
            True for ENDO, False for EPI/M

        Returns
        -------
        dict
            All gating parameters
        """
        # Compute index once for all lookups
        idx_float = (V - self.V_min) * self.inv_dV
        idx_float = torch.clamp(idx_float, 0, self.n_points - 1.0001)

        idx = idx_float.long()
        idx_p1 = torch.clamp(idx + 1, max=self.n_points - 1)
        frac = idx_float - idx.float()
        one_minus_frac = 1.0 - frac

        def interp(name):
            t = self.tables[name]
            return t[idx] * one_minus_frac + t[idx_p1] * frac

        result = {
            'm_inf': interp('m_inf'),
            'm_tau': interp('m_tau'),
            'h_inf': interp('h_inf'),
            'h_tau': interp('h_tau'),
            'j_inf': interp('j_inf'),
            'j_tau': interp('j_tau'),
            'r_inf': interp('r_inf'),
            'r_tau': interp('r_tau'),
            'd_inf': interp('d_inf'),
            'd_tau': interp('d_tau'),
            'f_inf': interp('f_inf'),
            'f_tau': interp('f_tau'),
            'f2_inf': interp('f2_inf'),
            'f2_tau': interp('f2_tau'),
            'Xr1_inf': interp('Xr1_inf'),
            'Xr1_tau': interp('Xr1_tau'),
            'Xr2_inf': interp('Xr2_inf'),
            'Xr2_tau': interp('Xr2_tau'),
            'Xs_inf': interp('Xs_inf'),
            'Xs_tau': interp('Xs_tau'),
        }

        # Cell-type specific Ito inactivation
        if celltype_is_endo:
            result['s_inf'] = interp('s_inf_endo')
            result['s_tau'] = interp('s_tau_endo')
        else:
            result['s_inf'] = interp('s_inf_epi')
            result['s_tau'] = interp('s_tau_epi')

        return result


# Global LUT cache to avoid rebuilding
_lut_cache: Dict[Tuple[str, str], LookupTable] = {}


def get_ttp06_lut(device: Optional[torch.device] = None) -> TTP06LUT:
    """
    Get (or create) cached TTP06 lookup table.

    Parameters
    ----------
    device : torch.device, optional
        Device for LUT (default: cuda if available)

    Returns
    -------
    TTP06LUT
        Lookup table instance
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    key = ('TTP06', str(device))
    if key not in _lut_cache:
        _lut_cache[key] = TTP06LUT(device=device)

    return _lut_cache[key]


def clear_lut_cache():
    """Clear the global LUT cache."""
    _lut_cache.clear()
