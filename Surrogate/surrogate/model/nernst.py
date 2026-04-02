"""Nernst equation computation and environment normalization.

Fixed-physics module with zero learned parameters. Computes reversal potentials
from intracellular ion concentrations and normalizes the 9-token environment
vector [Vm, E_Na, E_K, E_Ca, E_Ks, Na_i, K_i, Ca_i, Ca_ss] for Stage 2 input.

Constants match TTP06 (ten Tusscher-Panfilov 2006):
  R = 8314.472 J/(mol*K), T = 310 K, F = 96485.3415 C/mol
  Extracellular: Na_o = 140 mM, K_o = 5.4 mM, Ca_o = 2.0 mM
  Na/K permeability ratio: PRNaK = 0.03
"""

import torch
import torch.nn as nn

# Physical constants (TTP06, identical to Bidomain/Engine_V1/.../currents.py)
R = 8314.472       # Gas constant (J/(mol*K))
T = 310.0          # Temperature (K)
F = 96485.3415     # Faraday constant (C/mol)
RTONF = R * T / F  # ~26.713 mV

# Extracellular concentrations (mM)
Na_o = 140.0
K_o = 5.4
Ca_o = 2.0
PRNaK = 0.03  # Na/K permeability ratio for IKs

# Concentration clamp floor (prevents log(0))
EPS = 1e-12

# ---------------------------------------------------------------------------
# Normalization constants (physiological ranges)
# ---------------------------------------------------------------------------
# These are fixed shift/scale pairs for the 9 environment tokens:
#   [Vm, E_Na, E_K, E_Ca, E_Ks, Na_i, K_i, Ca_i, Ca_ss]
#
# Ranges derived from TTP06 physiological bounds:
#   Vm:   [-90, +40] mV          -> shift=-25, scale=65
#   E_Na: [+50, +80] mV          -> shift=65,  scale=15
#   E_K:  [-100, -80] mV         -> shift=-90, scale=10
#   E_Ca: [+100, +140] mV        -> shift=120, scale=20
#   E_Ks: [-90, -70] mV          -> shift=-80, scale=10
#   Na_i: [4, 20] mM             -> shift=12,  scale=8
#   K_i:  [130, 145] mM          -> shift=137.5, scale=7.5
#   Ca_i: [0.00005, 0.002] mM    -> shift=0.001, scale=0.001
#   Ca_ss:[0.00005, 0.002] mM    -> shift=0.001, scale=0.001

_NORM_SHIFT = torch.tensor([
    -25.0,    # Vm
    65.0,     # E_Na
    -90.0,    # E_K
    120.0,    # E_Ca
    -80.0,    # E_Ks
    12.0,     # Na_i
    137.5,    # K_i
    0.001,    # Ca_i
    0.001,    # Ca_ss
], dtype=torch.float32)

_NORM_SCALE = torch.tensor([
    65.0,     # Vm
    15.0,     # E_Na
    10.0,     # E_K
    20.0,     # E_Ca
    10.0,     # E_Ks
    8.0,      # Na_i
    7.5,      # K_i
    0.001,    # Ca_i
    0.001,    # Ca_ss
], dtype=torch.float32)


class NernstComputer(nn.Module):
    """Compute Nernst reversal potentials and normalize environment tokens.

    Zero learned parameters. All constants are registered as buffers so they
    follow device/dtype transfers via .to() and .cuda().

    Parameters
    ----------
    None (all fixed physics).

    Inputs to forward
    -----------------
    Na_i : (B,) intracellular sodium concentration (mM)
    K_i  : (B,) intracellular potassium concentration (mM)
    Ca_i : (B,) intracellular calcium concentration (mM)

    Returns
    -------
    E_Na, E_K, E_Ca, E_Ks : each (B,) reversal potentials in mV
    """

    def __init__(self):
        super().__init__()
        # Register normalization constants as buffers (not parameters)
        self.register_buffer("norm_shift", _NORM_SHIFT.clone())
        self.register_buffer("norm_scale", _NORM_SCALE.clone())
        # Store RTONF as buffer for device portability
        self.register_buffer("rtonf", torch.tensor(RTONF, dtype=torch.float32))

    def forward(
        self,
        Na_i: torch.Tensor,
        K_i: torch.Tensor,
        Ca_i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute four Nernst reversal potentials.

        Parameters
        ----------
        Na_i : (B,) or scalar — intracellular sodium (mM)
        K_i  : (B,) or scalar — intracellular potassium (mM)
        Ca_i : (B,) or scalar — intracellular calcium (mM)

        Returns
        -------
        (E_Na, E_K, E_Ca, E_Ks) : each same shape as input, in mV.
        """
        rtonf = self.rtonf

        E_Na = rtonf * torch.log(Na_o / Na_i.clamp(min=EPS))
        E_K = rtonf * torch.log(K_o / K_i.clamp(min=EPS))
        E_Ca = 0.5 * rtonf * torch.log(Ca_o / Ca_i.clamp(min=EPS))
        E_Ks = rtonf * torch.log(
            (K_o + PRNaK * Na_o) / (K_i + PRNaK * Na_i).clamp(min=EPS)
        )

        return E_Na, E_K, E_Ca, E_Ks

    def normalize_environment(
        self,
        Vm: torch.Tensor,
        E_Na: torch.Tensor,
        E_K: torch.Tensor,
        E_Ca: torch.Tensor,
        E_Ks: torch.Tensor,
        Na_i: torch.Tensor,
        K_i: torch.Tensor,
        Ca_i: torch.Tensor,
        Ca_ss: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize 9 environment tokens to approximately [-2, 2].

        Parameters
        ----------
        Vm, E_Na, E_K, E_Ca, E_Ks : (B,) voltages in mV
        Na_i, K_i, Ca_i, Ca_ss    : (B,) concentrations in mM

        Returns
        -------
        env_normalized : (B, 9) normalized environment vector
        """
        # Stack into (B, 9) — all inputs are (B,) or scalar
        env = torch.stack([Vm, E_Na, E_K, E_Ca, E_Ks, Na_i, K_i, Ca_i, Ca_ss], dim=-1)
        return (env - self.norm_shift) / self.norm_scale
