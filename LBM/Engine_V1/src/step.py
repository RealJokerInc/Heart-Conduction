"""Compiled LBM step functions.

Each function fuses: collide → clone(f_star) → stream → BC → recover_V
into a single callable suitable for @torch.compile.

Layer 2: pure functions, no state.
"""

import torch
from torch import Tensor

from .collision.bgk import bgk_collide
from .collision.mrt.d2q9 import mrt_collide_d2q9
from .streaming.d2q5 import stream_d2q5
from .streaming.d2q9 import stream_d2q9
from .boundary.neumann import apply_neumann_d2q5, apply_neumann_d2q9
from .state import recover_voltage


def lbm_step_d2q5_bgk(f: Tensor, V: Tensor, R: Tensor,
                       dt: float, omega: float, w: Tensor,
                       bounce_masks: dict) -> tuple:
    """One D2Q5-BGK step: collide → stream → Neumann BC → recover V."""
    f = bgk_collide(f, V, R, dt, omega, w)
    f_star = f.clone()
    f = stream_d2q5(f)
    f = apply_neumann_d2q5(f, f_star, bounce_masks)
    V = recover_voltage(f)
    return f, V


def lbm_step_d2q9_bgk(f: Tensor, V: Tensor, R: Tensor,
                       dt: float, omega: float, w: Tensor,
                       bounce_masks: dict) -> tuple:
    """One D2Q9-BGK step: collide → stream → Neumann BC → recover V."""
    f = bgk_collide(f, V, R, dt, omega, w)
    f_star = f.clone()
    f = stream_d2q9(f)
    f = apply_neumann_d2q9(f, f_star, bounce_masks)
    V = recover_voltage(f)
    return f, V


def lbm_step_d2q9_mrt(f: Tensor, V: Tensor, R: Tensor,
                       dt: float, w: Tensor,
                       s_e: float, s_eps: float,
                       s_jx: float, s_q: float,
                       s_pxx: float, s_pxy: float,
                       bounce_masks: dict,
                       s_jy: float = None) -> tuple:
    """One D2Q9-MRT step: collide → stream → Neumann BC → recover V."""
    f = mrt_collide_d2q9(f, V, R, dt, s_e, s_eps, s_jx, s_q,
                          s_pxx, s_pxy, w, s_jy=s_jy)
    f_star = f.clone()
    f = stream_d2q9(f)
    f = apply_neumann_d2q9(f, f_star, bounce_masks)
    V = recover_voltage(f)
    return f, V
