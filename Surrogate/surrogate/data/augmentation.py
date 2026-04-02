"""Augmentation utilities: corruption, conductance scaling, stitching.

Implemented in Phase 3. Stubs provided for Phase 1 imports.
"""

import torch
import numpy as np
from typing import List, Optional, Dict


class StitchedProtocol:
    """Concatenate protocols with rest breaks. Tier 11.

    Not a Protocol subclass — SingleCellGenerator detects via isinstance.
    Full implementation in Phase 3.
    """

    def __init__(self, protocols: list, rest_durations: list):
        self.protocols = protocols
        self.rest_durations = rest_durations
        self.name = 'stitched'
        self.tier = 11
        self.duration_ms = sum(p.duration_ms for p in protocols) + sum(rest_durations)


# Model-specific gate indices for corrupt_states
_GATE_INDICES = {
    'ttp06': list(range(5, 17)),   # m, h, j, r, s, d, f, f2, fCass, Xr1, Xr2, Xs (12 gates)
    'ord': list(range(8, 37)),     # m through xk1 (28 RL gates + nca at 29)
}
_CAI_INDEX = {
    'ttp06': 2,   # Cai in TTP06 StateIndex
    'ord': 2,     # cai in ORd StateIndex
}


def corrupt_states(states: torch.Tensor, corruption_type: str,
                   severity: float = 0.5, seed: int = 0,
                   model_type: str = 'ttp06') -> torch.Tensor:
    """Perturb gate states to non-physiological values.

    Args:
        states: State tensor (..., N_STATES).
        corruption_type: 'random_gates' or 'extreme_ca'.
        severity: 0.0 = no corruption, 1.0 = full random.
        seed: Random seed for reproducibility.
        model_type: 'ttp06' or 'ord' — selects correct gate indices.
    """
    if model_type not in _GATE_INDICES:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'ttp06' or 'ord'.")

    rng = np.random.RandomState(seed)
    states = states.clone()
    if corruption_type == 'random_gates':
        gate_indices = _GATE_INDICES[model_type]
        for idx in gate_indices:
            if states.dim() > 1:
                states[:, idx] = torch.rand(1) * severity + states[:, idx] * (1 - severity)
            else:
                states[idx] = float(torch.rand(1) * severity + states[idx] * (1 - severity))
    elif corruption_type == 'extreme_ca':
        cai_idx = _CAI_INDEX[model_type]
        if states.dim() > 1:
            states[:, cai_idx] *= (1 + 10 * severity)
        else:
            states[cai_idx] *= (1 + 10 * severity)
    return states
