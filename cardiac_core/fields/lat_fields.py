"""LAT-based named fields: the conduction-velocity vector field and wavefront geometry, all
derived from ONE canonical activation-time map.

Method: the Bayly (1998) local-quadratic CV fit on a uniform grid IS a fixed
2-D Savitzky–Golay convolution — ``Tx = K_x * T`` with ``K_x = s_y ⊗ g_x`` (a 5-point SG derivative
``g`` across the propagation axis, a 5-point SG smoother ``s`` across the other), ``Ty = K_y * T``.
The smoother's own residual ``|T − Ŝ*T|`` is a free per-node ``quality`` (high at collisions /
block edges / bad data). From ``∇T``:

    direction n̂ = ∇T/|∇T|,   speed = 1/|∇T|,   velocity = ∇T/|∇T|² = speed·n̂,
    curvature κ = ∇·n̂  (Osher–Sethian; also the collision/focus signal `divergence`),
    vorticity = curl(velocity)  (rotor cores).

**Divergence gating (mandatory).** Never return ``1/|∇T|`` at a collision: a boolean ``mask`` flags
low ``|∇T|`` (foci/collisions), a SG window that touches a non-activating node (no bleed), and high
fit residual; ``velocity``/``speed``/``direction`` are NaN there. Speeds are in cm/s (matching
``r.cv()``); a first-crossing LAT is invalid under reentry — use ``phase_map`` there.
"""

import torch
import torch.nn.functional as F

from . import derivatives

# 5-point Savitzky–Golay kernels: smoother (quadratic/cubic) and 1st-derivative (linear/quad).
_S = (-3.0, 12.0, 17.0, 12.0, -3.0)     # ÷35
_G = (-2.0, -1.0, 0.0, 1.0, 2.0)        # ÷(10·h)
_R = 2                                   # kernel half-width (5-point)

# Default gate: |∇T| below this (ms/cm) ⇒ CV above ~1000 cm/s ⇒ a collision/singularity, not tissue.
_MAG_FLOOR = 1e-3


def _sep_conv(field: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
    """Cross-correlate a ``(Nx, Ny)`` field with the separable kernel ``kx ⊗ ky`` (``kx`` along x,
    ``ky`` along y), whole-sample-mirror (reflect) padded — matches the ``face_mirror`` edge rule.
    (torch ``conv2d`` is cross-correlation, so the antisymmetric derivative kernel keeps its sign.)"""
    K = torch.outer(kx, ky)[None, None]                  # (1,1,5,5)
    f = F.pad(field[None, None], (_R, _R, _R, _R), mode='reflect')
    return F.conv2d(f, K)[0, 0]


def bayly_gradient(lat: torch.Tensor, dx: float, dy: float, mask=None):
    """Bayly/Savitzky–Golay gradient of a LAT map. Returns ``(Tx, Ty, quality, full_window)``.

    ``lat`` is ``(Nx, Ny)`` with NaN at non-activating nodes. NaN (and masked) nodes are held out:
    ``full_window`` is True only where the entire 5×5 SG window is valid (so gradients never bleed
    across a block edge). ``quality`` is the SG smoothing residual ``|lat − Ŝ*lat|`` (0 on a smooth
    front, large at discontinuities)."""
    dtype, device = lat.dtype, lat.device
    s = torch.tensor(_S, dtype=dtype, device=device) / 35.0
    gx = torch.tensor(_G, dtype=dtype, device=device) / (10.0 * dx)
    gy = torch.tensor(_G, dtype=dtype, device=device) / (10.0 * dy)
    ones = torch.ones(5, dtype=dtype, device=device)

    valid = torch.isfinite(lat)
    if mask is not None:
        valid = valid & mask
    lat0 = torch.where(valid, lat, torch.zeros_like(lat))

    Tx = _sep_conv(lat0, gx, s)                          # ∂T/∂x (smoothed in y)
    Ty = _sep_conv(lat0, s, gy)                          # ∂T/∂y (smoothed in x)
    smooth = _sep_conv(lat0, s, s)                       # Ŝ*T (the fitted centre value)
    n_valid = _sep_conv(valid.to(dtype), ones, ones)     # count of valid nodes in each 5×5 window
    full_window = n_valid >= (25.0 - 0.5)                # every window node valid
    quality = torch.where(valid, (lat - smooth).abs(), torch.full_like(lat, float('nan')))
    return Tx, Ty, quality, (full_window & valid)


def lat_field_bundle(r, *, mag_floor: float = _MAG_FLOOR) -> dict:
    """Compute every LAT-based field once (they share ∇T). Returns a dict of tensors.

    ``r`` is a ``SimulationResult``; uses the CANONICAL LAT (``activation_time``, interp/−40) routed
    through ``r.lat()`` so it agrees with ``r.cv()``. Speeds in cm/s.
    """
    from .. import analysis   # lazy — analysis is torch/numpy only, but keep import local
    lat = analysis.activation_time(r.Vm, r.times)        # (Nx, Ny), NaN where unactivated
    return bundle_from_lat(lat, r.dx, r.dy,
                           boundary_mode=getattr(r, 'boundary_mode', 'face_mirror'),
                           mask=getattr(r, 'domain_mask', None), mag_floor=mag_floor)


def bundle_from_lat(lat, dx, dy, *, boundary_mode='face_mirror', mask=None,
                    mag_floor: float = _MAG_FLOOR) -> dict:
    """The LAT-geometry core: raw LAT map → velocity/direction/speed/curvature/vorticity/quality/mask.

    Split out from :func:`lat_field_bundle` so the geometry is unit-testable on a synthetic (planar /
    radial) LAT with a known ``|∇T|`` and curvature, independent of a full simulation."""
    Tx, Ty, quality, ok = bayly_gradient(lat, dx, dy, mask=mask)
    magsq = Tx * Tx + Ty * Ty
    mag = torch.sqrt(magsq)

    nan = torch.full_like(lat, float('nan'))
    gate = ok & (mag >= mag_floor)                       # confident nodes
    gate_mask = ~gate                                    # the boolean `mask` field (True = gated)

    inv_mag = torch.where(gate, 1.0 / torch.where(gate, mag, torch.ones_like(mag)), nan)
    inv_magsq = torch.where(gate, 1.0 / torch.where(gate, magsq, torch.ones_like(magsq)), nan)

    Txg = torch.where(gate, Tx, nan)
    Tyg = torch.where(gate, Ty, nan)
    gradient = torch.stack([Txg, Tyg], dim=-1)                       # ∇T  (Nx,Ny,2)
    direction = torch.stack([Txg * inv_mag, Tyg * inv_mag], dim=-1)  # n̂
    speed = 1000.0 * inv_mag                                          # cm/s
    velocity = torch.stack([Txg * inv_magsq, Tyg * inv_magsq], dim=-1) * 1000.0  # cm/s vector

    # curvature κ = ∇·n̂ (Osher–Sethian) via the Phase-2 div operator on the unit-normal field.
    # NaN direction/velocity at gated nodes → NaN curvature/vorticity there (no spurious value).
    opkw = dict(boundary_mode=boundary_mode, mask=mask)
    curvature = derivatives.div(direction, dx, dy, **opkw)
    vorticity = derivatives.curl(velocity, dx, dy, **opkw)

    return dict(lat=lat, gradient=gradient, velocity=velocity, direction=direction,
                speed=speed, curvature=curvature, divergence=curvature, vorticity=vorticity,
                quality=quality, mask=gate_mask)
