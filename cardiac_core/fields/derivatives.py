"""Discrete vector-calculus operators on a uniform structured grid — the machinery every
named field composes from (Phase 2 of the ``analysis.fields`` branch).

Torch, on-device, float64. Scalars are ``(..., Nx, Ny)`` (leading ``...`` absorbs a time axis
or batch); vectors are ``(..., Nx, Ny, 2)`` with x/y on the trailing axis. Every operator honours
the run's ``boundary_mode`` (``"face_mirror"`` = no-flux/Neumann mirror) and an optional
``domain_mask`` (interior holes are no-flux edges; masked nodes come back NaN).

Two families, on purpose (DESIGN § "operator toolkit" / § Calculations A1–A8):

* **Collocated central** ``grad`` / ``div`` / ``curl`` (A1–A2) — the user-facing operators for
  ``velocity``/``direction``/``curvature``/``vorticity`` and display. Whole-sample mirror boundary
  ⇒ the normal derivative is 0 at a no-flux edge and ``curl(grad f) ≡ 0`` in the interior.
* **Staggered adjoint** ``laplacian`` via the forward-grad / backward-div face pair (A8) — the
  COMPACT 5-point ``∇²`` (NOT the wide checkerboard stencil that collocated ``div(grad)`` gives),
  so ``laplacian(V)`` equals the solver's diffusion term and the discrete divergence theorem is
  exact. ``_grad_face`` / ``_div_face`` are exposed for the region-integral flux (Phase 5).
"""

import torch

_VALID_MODES = ("face_mirror",)


def _check_mode(boundary_mode: str) -> None:
    if boundary_mode not in _VALID_MODES:
        raise ValueError(
            f"boundary_mode {boundary_mode!r} not supported; expected one of {_VALID_MODES} "
            "(the no-flux/Neumann mirror the solver uses)."
        )


def _pad1_mirror(a: torch.Tensor, axis: int) -> torch.Tensor:
    """Whole-sample mirror pad by 1 on a spatial ``axis`` (``-2``=x or ``-1``=y).

    ghost[-1] = a[1], ghost[N] = a[N-2] (reflection about the EDGE NODE). For a scalar this makes
    the central-difference normal derivative 0 at the boundary (the no-flux Neumann condition); it
    is applied identically to the bool mask so a full-rectangle outer edge mirrors (stays no-flux)
    while an interior hole severs (see ``_central``). Works for any leading dims.
    """
    n = a.shape[axis]
    if n < 2:
        # Degenerate axis: replicate the single slice so the pad is well-defined (grad→0 there).
        edge = a
        return torch.cat([edge, a, edge], dim=axis)
    lo = a.index_select(axis, a.new_tensor([1], dtype=torch.long))       # ghost_left  = a[1]
    hi = a.index_select(axis, a.new_tensor([n - 2], dtype=torch.long))   # ghost_right = a[N-2]
    return torch.cat([lo, a, hi], dim=axis)


def _slice_axis(a: torch.Tensor, axis: int, start: int, length: int) -> torch.Tensor:
    return a.narrow(axis, start, length)


def _central(f: torch.Tensor, h: float, axis: int, mask) -> torch.Tensor:
    """2nd-order central difference ∂f/∂axis (A1) with mirror boundary + mask severing.

    axis is ``-2`` (x) or ``-1`` (y). With ``mask`` (``(Nx, Ny)`` bool), a differencing partner that
    is masked-out is replaced by the CENTRE value (sever the connection ⇒ zero flux across that
    face); the outer rectangle edge stays a mirror (no-flux). Masked centres are NOT NaN'd here —
    the public operators do that once at the end.
    """
    n = f.shape[axis]
    fp = _pad1_mirror(f, axis)                      # (..., n+2, ...) along `axis`
    plus = _slice_axis(fp, axis, 2, n)              # f[i+1] (mirror at the border)
    minus = _slice_axis(fp, axis, 0, n)             # f[i-1]
    if mask is not None:
        mp = _pad1_mirror(mask, axis)
        plus_ok = _slice_axis(mp, axis, 2, n)
        minus_ok = _slice_axis(mp, axis, 0, n)
        plus = torch.where(plus_ok, plus, f)        # sever: use the centre value
        minus = torch.where(minus_ok, minus, f)
    return (plus - minus) / (2.0 * h)


def _apply_mask_nan(out: torch.Tensor, mask, *, vector: bool = False) -> torch.Tensor:
    """NaN-fill masked-out nodes. ``mask`` is ``(Nx, Ny)``; broadcasting handles any leading dims
    (and the trailing component axis for a vector output when ``vector=True``)."""
    if mask is None:
        return out
    m = mask.unsqueeze(-1) if vector else mask       # (Nx,Ny,1) for vectors, (Nx,Ny) for scalars
    return torch.where(m, out, torch.full_like(out, float('nan')))


def grad(f: torch.Tensor, dx: float, dy: float = None, *,
         boundary_mode: str = "face_mirror", mask=None) -> torch.Tensor:
    """∇f — gradient of a scalar. ``(..., Nx, Ny)`` → ``(..., Nx, Ny, 2)`` (x-, y-component)."""
    _check_mode(boundary_mode)
    dy = dx if dy is None else dy
    gx = _central(f, dx, -2, mask)
    gy = _central(f, dy, -1, mask)
    out = torch.stack([gx, gy], dim=-1)
    return _apply_mask_nan(out, mask, vector=True)


def div(F: torch.Tensor, dx: float, dy: float = None, *,
        boundary_mode: str = "face_mirror", mask=None) -> torch.Tensor:
    """∇·F — divergence of a vector. ``(..., Nx, Ny, 2)`` → ``(..., Nx, Ny)`` (A2, collocated)."""
    _check_mode(boundary_mode)
    dy = dx if dy is None else dy
    Fx = F[..., 0]
    Fy = F[..., 1]
    out = _central(Fx, dx, -2, mask) + _central(Fy, dy, -1, mask)
    return _apply_mask_nan(out, mask)


def curl(F: torch.Tensor, dx: float, dy: float = None, *,
         boundary_mode: str = "face_mirror", mask=None) -> torch.Tensor:
    """∇×F (2-D z-component) = ∂Fy/∂x − ∂Fx/∂y. ``(..., Nx, Ny, 2)`` → ``(..., Nx, Ny)``.

    The CROSS pattern (x-derivative of the y-component minus y-derivative of the x-component) —
    distinct from ``div``'s straight (add) pattern.
    """
    _check_mode(boundary_mode)
    dy = dx if dy is None else dy
    Fx = F[..., 0]
    Fy = F[..., 1]
    out = _central(Fy, dx, -2, mask) - _central(Fx, dy, -1, mask)
    return _apply_mask_nan(out, mask)


# --------------------------------------------------------------------------- staggered core (A8)

def _grad_face(f: torch.Tensor, h: float, axis: int, mask):
    """Forward difference cell→face along ``axis``: g[i+½] = (f[i+1]−f[i])/h.

    Returns the ``N-1`` interior faces. Faces touching a masked cell are zeroed (no-flux).
    """
    n = f.shape[axis]
    fp1 = _slice_axis(f, axis, 1, n - 1)   # f[i+1], i=0..N-2
    f0 = _slice_axis(f, axis, 0, n - 1)    # f[i]
    g = (fp1 - f0) / h
    if mask is not None:
        both = _slice_axis(mask, axis, 1, n - 1) & _slice_axis(mask, axis, 0, n - 1)
        g = torch.where(both, g, torch.zeros_like(g))
    return g


def _div_face(g: torch.Tensor, h: float, axis: int):
    """Backward difference face→cell along ``axis`` with zero-flux boundary faces:
    (div g)[i] = (g[i+½] − g[i−½])/h, boundary faces g[−½]=g[N−½]=0.

    ``g`` carries the ``N-1`` interior faces; the two boundary faces are added as zeros so the
    telescoping is exact (the divergence theorem holds to machine precision)."""
    shape = list(g.shape)
    shape[axis] = 1
    zero = g.new_zeros(shape)
    gfull = torch.cat([zero, g, zero], dim=axis)      # N+1 faces: −½ .. N−½
    n = gfull.shape[axis] - 1                          # = original N
    return (_slice_axis(gfull, axis, 1, n) - _slice_axis(gfull, axis, 0, n)) / h


def laplacian(f: torch.Tensor, dx: float, dy: float = None, *,
              boundary_mode: str = "face_mirror", mask=None) -> torch.Tensor:
    """∇²f via the staggered forward-grad / backward-div pair (A8) — the COMPACT 5-point Laplacian.

    ``div(grad)`` of the collocated central operators would be the WIDE 2h stencil with a
    checkerboard null-space; the staggered adjoint pair gives ``(f[i+1]−2f[i]+f[i−1])/dx²`` exactly
    and makes the discrete divergence theorem exact. No-flux (``face_mirror``) at every domain /
    mask boundary face; masked nodes → NaN.
    """
    _check_mode(boundary_mode)
    dy = dx if dy is None else dy
    lap_x = _div_face(_grad_face(f, dx, -2, mask), dx, -2)
    lap_y = _div_face(_grad_face(f, dy, -1, mask), dy, -1)
    out = lap_x + lap_y
    return _apply_mask_nan(out, mask)


def winding_loop_sum(phase: torch.Tensor, *, ccw: bool = True) -> torch.Tensor:
    """Sum of wrapped phase differences around each 2×2 plaquette — the topological-charge core.

    ``W(Δφ) = atan2(sin Δφ, cos Δφ)`` wraps each edge difference to ``(−π, π]``; the CCW loop sum is
    ``≈ 2π · (integer charge)`` — ``±2π`` at a phase singularity / rotor tip. The ONE primitive
    behind ``phase_singularities``, ``circulation``/``winding_number``, and Gauss–Bonnet (DESIGN § 5).

    Parameters
    ----------
    phase : ``(Nx, Ny)`` angle field in radians (a Hilbert phase, ``atan2`` of a vector, …).
    ccw : orientation (``True`` = counter-clockwise positive).

    Returns
    -------
    ``(Nx-1, Ny-1)`` loop sum per plaquette (radians). Divide by ``2π`` for the integer charge.
    """
    def W(d):
        return torch.atan2(torch.sin(d), torch.cos(d))
    d1 = W(phase[1:, :-1] - phase[:-1, :-1])     # bottom edge (+x)
    d2 = W(phase[1:, 1:] - phase[1:, :-1])       # right edge  (+y)
    d3 = W(phase[:-1, 1:] - phase[1:, 1:])       # top edge    (−x)
    d4 = W(phase[:-1, :-1] - phase[:-1, 1:])     # left edge   (−y)
    s = d1 + d2 + d3 + d4
    return s if ccw else -s


def _face_harm(D: torch.Tensor, axis: int) -> torch.Tensor:
    """Harmonic mean of ``D`` to the interior faces along ``axis``: ``D_face = 2·D_L·D_R/(D_L+D_R)``.

    This is the FDM cardinal-4 interface scheme (``fdm.py`` ``_harm``): it is ``D`` for uniform D and
    ``0`` at a ``D=0`` interface (zero flux into a scar), so ``diffusion_term`` reproduces the solver's
    own operator for uniform AND heterogeneous-isotropic D (and a D=0 block)."""
    n = D.shape[axis]
    a = _slice_axis(D, axis, 0, n - 1)
    b = _slice_axis(D, axis, 1, n - 1)
    s = a + b
    safe = torch.where(s > 0, s, torch.ones_like(s))
    return torch.where(s > 0, 2.0 * a * b / safe, torch.zeros_like(s))


def diffusion_term(f: torch.Tensor, D, dx: float, dy: float = None, *,
                   boundary_mode: str = "face_mirror", mask=None) -> torch.Tensor:
    """Conservative isotropic diffusion operator ∇·(D∇f) — the ``source_sink`` numerical core.

    Staggered face pair (A8) with HARMONIC face-averaged ``D`` (matching the solver's cardinal-4
    operator), so it reduces to ``D·laplacian(f)`` for uniform D and equals the engine's
    ``apply_diffusion`` (incl. a masked/scar no-flux rim). ``D`` is a scalar or an ``(Nx, Ny)`` field.
    Masked nodes → NaN.
    """
    _check_mode(boundary_mode)
    dy = dx if dy is None else dy
    D = torch.as_tensor(D, dtype=f.dtype, device=f.device)
    if D.dim() == 0:
        D = D.expand(f.shape[-2], f.shape[-1])
    fx = _face_harm(D, -2) * _grad_face(f, dx, -2, mask)     # face flux, x
    fy = _face_harm(D, -1) * _grad_face(f, dy, -1, mask)     # face flux, y
    out = _div_face(fx, dx, -2) + _div_face(fy, dy, -1)
    return _apply_mask_nan(out, mask)
