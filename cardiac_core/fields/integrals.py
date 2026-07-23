"""Global reductions — ``fields.integrals``.

Each is the Stokes / divergence-theorem partner of a ``fields.derivatives`` operator, so the two
tiers cross-check for free:

    region_integral   ∬_Ω f dA                        (midpoint rule; the workhorse)
    net_flux          ∮_∂Ω F·n̂ ds = ∬_Ω div F dA      (telescoping — EXACT to ~1e-12)
    circulation       ∮_C v·dl = ∬ curl v dA          (Stokes)
    winding_number    ∮ ∇φ·dl / 2π                    (enclosed rotors; the shared loop-sum)
    conduction_time   ∫ ∇T·dl = T(b) − T(a)           (integrate the SLOWNESS ∇T, NOT velocity)
    activated_area    ∬ 𝟙[V≥θ] dA                     (recruitment curve)
    isochrone family  wavefront_length ∮ds, global_curvature ∮κ ds (Gauss–Bonnet ≈ 2π)

Sign conventions (asserted, not incidental): flux uses the OUTWARD normal (positive = efflux =
source inside); circulation / winding are counter-clockwise positive. Regions and boundaries ARE
``(Nx, Ny)`` masks — the same objects used for scars/stimuli.
"""

import math

import numpy as np
import torch

from . import derivatives
from .derivatives import winding_loop_sum


def _as_raw(F):
    from . import VectorField
    return F.components if isinstance(F, VectorField) else F


def region_integral(f: torch.Tensor, dx: float, dy: float, over=None) -> torch.Tensor:
    """∬_Ω f dA (midpoint rule, ``dA = dx·dy``). ``f`` is ``(Nx, Ny)`` → scalar, or ``(T, Nx, Ny)`` →
    ``(T,)``. ``over`` is an ``(Nx, Ny)`` bool region (default = the whole grid). NaN nodes are
    skipped (``nansum``)."""
    if over is not None:
        m = over
        while m.dim() < f.dim():
            m = m.unsqueeze(0)
        f = torch.where(m, f, torch.zeros_like(f))
    return torch.nansum(f, dim=(-2, -1)) * (dx * dy)


def _divergence_flux(F: torch.Tensor, dx: float, dy: float, mask=None) -> torch.Tensor:
    """Staggered divergence of a NODE vector ``F`` (``…, Nx, Ny, 2``): average F to faces, then the
    backward face-div. Its region-sum telescopes EXACTLY to the boundary face flux."""
    Fx, Fy = F[..., 0], F[..., 1]
    gx = 0.5 * (Fx[..., 1:, :] + Fx[..., :-1, :])       # x-faces  (…, Nx-1, Ny)
    gy = 0.5 * (Fy[..., :, 1:] + Fy[..., :, :-1])       # y-faces  (…, Nx, Ny-1)
    if mask is not None:
        gx = torch.where(mask[1:, :] & mask[:-1, :], gx, torch.zeros_like(gx))
        gy = torch.where(mask[:, 1:] & mask[:, :-1], gy, torch.zeros_like(gy))
    return derivatives._div_face(gx, dx, -2) + derivatives._div_face(gy, dy, -1)


def net_flux(F, dx: float, dy: float, region=None, *, mask=None) -> torch.Tensor:
    """Net outward flux ∮_∂Ω F·n̂ ds = ∬_Ω div F dA (divergence theorem). ``F`` is a node vector
    (``VectorField`` or ``(…,Nx,Ny,2)``); ``region`` is the ``(Nx,Ny)`` mask Ω. Positive = net
    efflux (a source inside Ω). Exact to telescoping — see ``test_net_flux_equals_boundary_faces``."""
    div = _divergence_flux(_as_raw(F), dx, dy, mask=mask)
    return region_integral(div, dx, dy, over=region)


def circulation(v, dx: float, dy: float, region=None, *,
                boundary_mode: str = 'face_mirror', mask=None) -> torch.Tensor:
    """Circulation ∮_C v·dl = ∬ curl(v) dA (Stokes). ``region`` = enclosed area Ω. CCW positive →
    enclosed vorticity (a rotor)."""
    c = derivatives.curl(_as_raw(v), dx, dy, boundary_mode=boundary_mode, mask=mask)
    return region_integral(c, dx, dy, over=region)


def winding_number(phase: torch.Tensor, region=None) -> int:
    """Number of enclosed phase singularities (rotors) = ∮∇φ·dl / 2π, via the shared 2×2 loop-sum.

    ``region``, if given, is an ``(Nx, Ny)`` mask; plaquettes whose lower-left node is in the region
    are counted. Returns the rounded integer charge."""
    charge = winding_loop_sum(phase) / (2 * math.pi)       # (Nx-1, Ny-1)
    if region is not None:
        charge = torch.where(region[:-1, :-1], charge, torch.zeros_like(charge))
    return int(round(float(torch.nansum(charge).item())))


def conduction_time(lat: torch.Tensor, a, b) -> float:
    """Traversal time between two sites = ``LAT(b) − LAT(a)`` (the gradient theorem: ∫∇T·dl is
    path-independent, so it is just ΔT). Integrate the SLOWNESS ∇T — NEVER the ``velocity`` field
    (``∫v·dl ≠ ΔT``; only ∇T is curl-free). NaN if either site never activates."""
    ta = float(lat[a[0], a[1]])
    tb = float(lat[b[0], b[1]])
    return tb - ta


def activated_area(V: torch.Tensor, dx: float, dy: float, *, threshold: float = -40.0,
                   over=None) -> torch.Tensor:
    """∬ 𝟙[V ≥ θ] dA per frame — the depolarized area (recruitment curve). ``V`` is ``(T, Nx, Ny)``
    → ``(T,)``; ``(Nx, Ny)`` → scalar."""
    act = (V >= threshold).to(V.dtype)
    return region_integral(act, dx, dy, over=over)


def state_fraction(V: torch.Tensor, *, threshold: float = -40.0, over=None) -> torch.Tensor:
    """Fraction of the region that is excited (``V ≥ θ``) per frame. ``(T, Nx, Ny)`` → ``(T,)``."""
    exc = (V >= threshold).to(V.dtype)
    if over is not None:
        m = over
        while m.dim() < exc.dim():
            m = m.unsqueeze(0)
        num = torch.nansum(torch.where(m, exc, torch.zeros_like(exc)), dim=(-2, -1))
        den = float(over.sum().item())
    else:
        num = torch.nansum(exc, dim=(-2, -1))
        den = float(exc.shape[-2] * exc.shape[-1])
    return num / max(den, 1.0)


# ------------------------------------------------------------------------------- isochrone family

def isochrone(lat: torch.Tensor, level: float, dx: float, dy: float):
    """Marching-squares contour(s) of the LAT level set ``T = level``. Returns a list of ``(n, 2)``
    numpy arrays of ``(x, y)`` points in cm. Non-activating (NaN) nodes are lifted above the level so
    the contour tracks the activated boundary."""
    from skimage.measure import find_contours
    a = lat.detach().cpu().numpy().astype(float)
    if np.isnan(a).any():
        finite = a[np.isfinite(a)]
        fill = (finite.max() + abs(finite.max()) + 1.0) if finite.size else level + 1.0
        a = np.where(np.isfinite(a), a, fill)
    return [c * np.array([dx, dy]) for c in find_contours(a, level)]


def wavefront_length(lat: torch.Tensor, level: float, dx: float, dy: float) -> float:
    """Front perimeter ∮ ds (cm) of the ``T = level`` isochrone (summed over all contour pieces)."""
    total = 0.0
    for c in isochrone(lat, level, dx, dy):
        seg = np.diff(c, axis=0)
        total += float(np.hypot(seg[:, 0], seg[:, 1]).sum())
    return total


def global_curvature(lat: torch.Tensor, level: float, dx: float, dy: float) -> float:
    """∮ κ ds — the net turning of the ``T = level`` isochrone (Gauss–Bonnet: ``≈ 2π`` for a single
    convex closed loop). Sum of signed exterior turning angles over the polyline vertices."""
    total = 0.0
    for c in isochrone(lat, level, dx, dy):
        closed = np.allclose(c[0], c[-1])
        pts = c[:-1] if closed else c
        n = len(pts)
        if n < 3:
            continue
        t = np.diff(np.vstack([pts, pts[:1]]) if closed else pts, axis=0)
        seg_len = np.hypot(t[:, 0], t[:, 1])
        ang = np.arctan2(t[:, 1], t[:, 0])
        dang = np.diff(np.concatenate([ang, ang[:1]]) if closed else ang)
        dang = (dang + np.pi) % (2 * np.pi) - np.pi          # wrap turning angle to (−π, π]
        total += float(dang.sum())
    return total
