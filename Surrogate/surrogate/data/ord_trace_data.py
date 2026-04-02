"""ORd trace data format — 101-column layout for O'Hara-Rudy model.

Column layout:
    0:     Vm (mV)
    1:     I_stim (sign-flipped: positive = depolarizing)
    2:     dt (ms)
    3-42:  40 state variables (ORd StateIndex order)
    43:    I_ion (pure ionic current, no stimulus)
    44:    clamp_mask (0.0 = free-running, 1.0 = voltage clamped)
    45-72: 28 gate_inf values (Rush-Larsen gates only, gate_indices order)
    73-100: 28 gate_tau values (same order)
    Total: 101 columns

State variable order (indices 0-39 within state block, cols 3-42):
    0-3:   Bulk concentrations: nai, ki, cai, cansr
    4-7:   Subspace concentrations: nass, kss, cass, cajsr
    8-13:  INa gates: m, hf, hs, j, hsp, jp
    14-16: INaL gates: mL, hL, hLp
    17-22: Ito gates: a, iF, iS, ap, iFp, iSp
    23-31: ICaL gates: d, ff, fs, fcaf, fcas, jca, nca, ffp, fcafp
    32-33: IKr gates: xrf, xrs
    34-35: IKs gates: xs1, xs2
    36:    IK1 gate: xk1
    37-38: SR release: Jrelnp, Jrelp
    39:    CaMKII: CaMKt

28 Rush-Larsen gates (gate_inf/tau columns):
    m, hf, hs, j, hsp, jp, mL, hL, hLp,
    a, iF, iS, ap, iFp, iSp,
    d, ff, fs, fcaf, fcas, jca, ffp, fcafp,
    xrf, xrs, xs1, xs2, xk1
    (nca EXCLUDED — it is Forward Euler)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import torch
from torch import Tensor


@dataclass
class ORdTraceData:
    """Container for a single ORd protocol trace (101 columns)."""

    data: Tensor  # (T, 101) float64
    metadata: Dict[str, Any] = field(default_factory=dict)

    # === Column indices ===
    VM = 0
    I_STIM = 1
    DT = 2
    STATES_START = 3
    STATES_END = 43       # exclusive: 3 + 40
    I_ION = 43
    CLAMP_MASK = 44
    GATE_INF_START = 45
    GATE_INF_END = 73     # exclusive: 45 + 28
    GATE_TAU_START = 73
    GATE_TAU_END = 101    # exclusive: 73 + 28
    N_COLUMNS = 101

    # === State indices (within cols 3-42) ===
    # Concentrations
    STATE_NAI = 0
    STATE_KI = 1
    STATE_CAI = 2
    STATE_CANSR = 3
    STATE_NASS = 4
    STATE_KSS = 5
    STATE_CASS = 6
    STATE_CAJSR = 7

    # First RL gate
    STATE_M = 8

    # Last RL gate
    STATE_XK1 = 36

    # Non-RL states
    STATE_NCA = 29       # Forward Euler gate (excluded from gate_inf/tau)
    STATE_JRELNP = 37
    STATE_JRELP = 38
    STATE_CAMKT = 39

    N_STATES = 40
    N_RL_GATES = 28      # gate_inf/tau columns

    # RL gate indices within the state block (for scaffold/corrupt_states)
    # These are the 28 gates that have gate_inf and gate_tau
    RL_GATE_INDICES = [
        8, 9, 10, 11, 12, 13,     # INa: m, hf, hs, j, hsp, jp
        14, 15, 16,                 # INaL: mL, hL, hLp
        17, 18, 19, 20, 21, 22,    # Ito: a, iF, iS, ap, iFp, iSp
        23, 24, 25, 26, 27, 28,    # ICaL: d, ff, fs, fcaf, fcas, jca
        30, 31,                     # ICaL phosph: ffp, fcafp
        32, 33,                     # IKr: xrf, xrs
        34, 35,                     # IKs: xs1, xs2
        36,                         # IK1: xk1
    ]

    # Concentration indices for v3 model [Na_i, K_i, Ca_i, Ca_ss]
    CONC_INDICES = [0, 1, 2, 6]   # nai, ki, cai, cass (within state block)

    def __post_init__(self):
        if self.data.shape[-1] != self.N_COLUMNS:
            raise ValueError(
                f"ORdTraceData expects {self.N_COLUMNS} columns, "
                f"got {self.data.shape[-1]}"
            )
