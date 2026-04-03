"""Per-dimension min-max normalization for loss computation.

Maps all target dimensions to [0, 1] before computing MSE, so every dimension
contributes equally regardless of physical scale. A 10% error on Ca_i (0.0001 mM)
gets the same loss as a 10% error on K_i (138 mM).

Ranges are computed from T1 training data with small padding to avoid division
by zero on near-constant dimensions. These are FIXED constants, not learned.

Usage:
    norm = LossNormalizer()
    loss = norm.normalized_mse(pred_ionic, target_ionic, 'ionic_states')
         + norm.normalized_mse(pred_conc, target_conc, 'concentrations')
         + norm.normalized_mse(pred_cond, target_cond, 'conductance_products')
"""

import torch
from torch import Tensor

# ============================================================================
# Per-dimension ranges from T1 training data (8M timesteps, EPI celltype)
#
# Ionic states (14 dims): 12 HH gates [0,1] + RR [0,1] + CaSR [0.1, 4.7]
# Concentrations (4 dims): Na_i [8.4, 8.6], K_i [136.7, 137.3], Ca_i [6.8e-5, 1.8e-4], Ca_ss [1.2e-4, 6.8e-2]
# Conductance products (5 dims): G_Na [0, 0.30], G_CaL [0, 0.51], G_to [0, 0.47], G_Kr [0, 0.37], G_Ks [0, 0.02]
# ============================================================================

_RANGES = {
    'ionic_states': {
        'min': torch.tensor([
            0.0015, 0.0, 0.0, 0.0, 0.0009, 0.0, 0.206, 0.331,
            0.400, 0.0002, 0.0135, 0.0017, 0.881, 1.69,
        ]),
        'max': torch.tensor([
            1.0, 0.764, 0.764, 0.794, 1.0, 0.990, 0.999, 1.0,
            0.995, 0.958, 0.478, 0.142, 0.998, 4.73,
        ]),
    },
    'concentrations': {
        'min': torch.tensor([8.389, 136.70, 6.79e-5, 1.20e-4]),
        'max': torch.tensor([8.646, 137.30, 1.76e-4, 6.82e-2]),
    },
    'conductance_products': {
        'min': torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
        'max': torch.tensor([0.305, 0.513, 0.475, 0.366, 0.0201]),
    },
}

# Add small padding to ranges to avoid division by zero
_EPS = 1e-8


class LossNormalizer:
    """Per-dimension min-max normalization for loss computation.

    Caches normalized range tensors on the correct device/dtype on first use.
    """

    def __init__(self):
        self._cache: dict[str, dict[str, Tensor]] = {}

    def _get_range(self, name: str, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """Get cached min/range tensors on the right device."""
        key = f"{name}_{device}_{dtype}"
        if key not in self._cache:
            r = _RANGES[name]
            mn = r['min'].to(device=device, dtype=dtype)
            mx = r['max'].to(device=device, dtype=dtype)
            rng = (mx - mn).clamp(min=_EPS)
            self._cache[key] = {'min': mn, 'range': rng}
        return self._cache[key]['min'], self._cache[key]['range']

    def normalize(self, x: Tensor, name: str) -> Tensor:
        """Normalize tensor to [0, 1] using known physiological ranges."""
        mn, rng = self._get_range(name, x.device, x.dtype)
        return (x - mn) / rng

    def normalized_mse(self, pred: Tensor, target: Tensor, name: str) -> Tensor:
        """MSE on min-max normalized values. All dims contribute equally."""
        pred_n = self.normalize(pred, name)
        target_n = self.normalize(target, name)
        return torch.nn.functional.mse_loss(pred_n, target_n)
