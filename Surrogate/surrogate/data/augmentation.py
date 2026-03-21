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


def corrupt_states(states: torch.Tensor, corruption_type: str,
                   severity: float = 0.5, seed: int = 0) -> torch.Tensor:
    """Perturb gate states to non-physiological values.

    Full implementation in Phase 3.
    """
    rng = np.random.RandomState(seed)
    states = states.clone()
    if corruption_type == 'random_gates':
        gate_indices = list(range(5, 17))  # m, h, j, r, s, d, f, f2, fCass, Xr1, Xr2, Xs
        for idx in gate_indices:
            if states.dim() > 1:
                states[:, idx] = torch.rand(1) * severity + states[:, idx] * (1 - severity)
            else:
                states[idx] = float(torch.rand(1) * severity + states[idx] * (1 - severity))
    elif corruption_type == 'extreme_ca':
        if states.dim() > 1:
            states[:, 2] *= (1 + 10 * severity)  # Cai
        else:
            states[2] *= (1 + 10 * severity)
    return states
