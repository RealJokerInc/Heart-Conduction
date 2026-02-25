"""
version2.py
-----------

Lightweight, numerically robust variant of the tissue solver tailored to the
plot_simulation_circular.py workflow.

Compared to the original "full" monodomain + Aliev–Panfilov system this file
implements a *relaxed* passive cable model:

    C_m dV/dt = ∇·(D ∇V) - (V - V_rest)/tau_v + I_stim

with an auxiliary low-pass filtered recovery variable

    dw/dt = (V - w)/tau_w.

The goal is not to reproduce the full upstroke / plateau / repolarisation
shape of an action potential, but to give a well-behaved anisotropic conduction
equation that:

  * respects the diffusion tensor D defined on the mesh,
  * propagates stimuli in the correct preferred directions, and
  * is stable for the relatively large time–steps used by the plotting scripts.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def build_D_components(
    theta: np.ndarray,
    D_parallel,
    D_perp,
    epsilon_tissue: np.ndarray,
):
    """
    Construct the diffusion-tensor components (Dxx, Dxy, Dyy) from a field of
    fiber angles `theta` and scalar (or array-valued) diffusivities along and
    across the fibers.

    Parameters
    ----------
    theta : array, shape (ny, nx)
        Fiber orientation angle in radians. theta = 0 means fibers aligned with
        the +x axis.
    D_parallel : float or array broadcastable to theta.shape
        Diffusivity along the fiber direction.
    D_perp : float or array broadcastable to theta.shape
        Diffusivity orthogonal to the fibers.
    epsilon_tissue : array, shape (ny, nx)
        Dimensionless "presence of tissue" factor. Typically 1.0 in healthy
        tissue and 0.0 inside an infarct or hole. This simply scales the
        diffusion tensor: if epsilon_tissue == 0, diffusion is shut off.

    Returns
    -------
    Dxx, Dxy, Dyy : arrays, each shape (ny, nx)
        Components of the symmetric 2×2 tensor

            D = [[Dxx, Dxy],
                 [Dxy, Dyy]]

        in the lab (x, y) coordinates.
    """
    theta = np.asarray(theta, dtype=float)
    eps = np.asarray(epsilon_tissue, dtype=float)

    D_par = np.broadcast_to(D_parallel, theta.shape).astype(float)
    D_perp = np.broadcast_to(D_perp, theta.shape).astype(float)

    c = np.cos(theta)
    s = np.sin(theta)

    # Standard rotation formula: R diag(D_par, D_perp) R^T
    Dxx = eps * (D_par * c * c + D_perp * s * s)
    Dyy = eps * (D_par * s * s + D_perp * c * c)
    Dxy = eps * (D_par - D_perp) * c * s

    return Dxx, Dxy, Dyy


# ---------------------------------------------------------------------------
# Finite difference building blocks
# ---------------------------------------------------------------------------

def _centered_diff(field: np.ndarray, axis: int, spacing: float) -> np.ndarray:
    """
    Second–order centred finite difference with first–order one-sided stencils
    on the outer boundary.

    This approximates df/dx (or df/dy) assuming "natural" (approximately
    no-flux) behaviour at the outer edges of the domain.
    """
    f = np.asarray(field, dtype=float)
    df = np.zeros_like(f)

    if axis == 0:  # derivative with respect to y
        # interior points
        df[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2.0 * spacing)
        # top and bottom: first-order one-sided
        df[0, :] = (f[1, :] - f[0, :]) / spacing
        df[-1, :] = (f[-1, :] - f[-2, :]) / spacing
    elif axis == 1:  # derivative with respect to x
        df[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * spacing)
        df[:, 0] = (f[:, 1] - f[:, 0]) / spacing
        df[:, -1] = (f[:, -1] - f[:, -2]) / spacing
    else:
        raise ValueError("axis must be 0 (y) or 1 (x)")

    return df


def _anisotropic_diffusion(V: np.ndarray, mesh) -> np.ndarray:
    """
    Discrete diffusion operator L[V] ≈ ∇·(D ∇V) on the rectangular mesh.

    We compute:
        grad V = (V_x, V_y) via centred finite differences,
        F      = D grad V   (flux vector field),
        L[V]   = div F      via centred finite differences.
    """
    dx = float(mesh.dx)
    dy = float(mesh.dy)

    Vx = _centered_diff(V, axis=1, spacing=dx)  # ∂V/∂x
    Vy = _centered_diff(V, axis=0, spacing=dy)  # ∂V/∂y

    Dxx = mesh.Dxx
    Dxy = mesh.Dxy
    Dyy = mesh.Dyy

    Fx = Dxx * Vx + Dxy * Vy
    Fy = Dxy * Vx + Dyy * Vy

    dFx_dx = _centered_diff(Fx, axis=1, spacing=dx)
    dFy_dy = _centered_diff(Fy, axis=0, spacing=dy)

    return dFx_dx + dFy_dy


def _max_diffusivity(mesh) -> float:
    """
    Return a conservative upper bound on the local diffusion eigenvalue λ_max(D).

    For each cell we form the 2×2 tensor

        D = [[Dxx, Dxy],
             [Dxy, Dyy]]

    and take its largest eigenvalue. The global maximum is then used in a CFL-
    type condition for the explicit Euler time stepping.
    """
    Dxx = np.asarray(mesh.Dxx, dtype=float)
    Dxy = np.asarray(mesh.Dxy, dtype=float)
    Dyy = np.asarray(mesh.Dyy, dtype=float)

    # eigenvalues of 2×2 symmetric matrix
    half_trace = 0.5 * (Dxx + Dyy)
    diff = 0.5 * (Dxx - Dyy)
    radicand = diff * diff + Dxy * Dxy
    radicand = np.maximum(radicand, 0.0)  # numerical safety

    lambda_max = half_trace + np.sqrt(radicand)

    return float(np.max(lambda_max))


def _stable_substep(mesh, dt: float, safety: float = 0.9) -> tuple[int, float]:
    """
    Compute how many internal substeps are needed to make the explicit
    diffusion update stable for a desired global step `dt`.

    We use the standard explicit-Euler stability condition for a 2-D diffusion
    equation on a rectangular grid:

        dt <= 1 / ( 2 * λ_max * (1/dx^2 + 1/dy^2) )

    where λ_max is the largest eigenvalue of D.
    """
    Dmax = _max_diffusivity(mesh)
    if Dmax <= 0.0:
        return 1, dt  # completely non-diffusive domain

    dx = float(mesh.dx)
    dy = float(mesh.dy)

    inv_dx2_dy2 = 1.0 / (dx * dx) + 1.0 / (dy * dy)
    dt_crit = safety / (2.0 * Dmax * inv_dx2_dy2)

    if dt <= dt_crit:
        return 1, dt

    n_sub = int(np.ceil(dt / dt_crit))
    return n_sub, dt / n_sub


# ---------------------------------------------------------------------------
# Time stepping
# ---------------------------------------------------------------------------

def default_params():
    """
    Default parameter set for the relaxed monodomain model.

    C_m      : effective membrane capacitance (arbitrary units)
    tau_v    : relaxation time back to V_rest (ms)
    tau_w    : time scale for the low-pass recovery variable w (ms)
    V_rest   : resting potential (mV)
    V_min,
    V_max    : soft clamping bounds for numerical safety
    """
    return dict(
        C_m=1.0,
        tau_v=20.0,
        tau_w=80.0,
        V_rest=0.0,
        V_min=-20.0,
        V_max=40.0,
    )


def step_relaxed_monodomain(
    V: np.ndarray,
    w: np.ndarray,
    mesh,
    params: dict,
    I_stim: np.ndarray,
    dt: float,
):
    """
    Advance the relaxed monodomain system by one time step of length `dt`.

    Parameters
    ----------
    V : array (ny, nx)
        Transmembrane voltage field at the current time.
    w : array (ny, nx)
        Recovery / adaptation variable. It acts here as a simple low-pass
        filtered copy of V; keeping it separate lets us later swap back in a
        more biophysically detailed reaction term if we want.
    mesh : TissueMesh
        Provides Dxx, Dxy, Dyy and the grid spacing dx, dy.
    params : dict
        Parameter dictionary. See `default_params`.
    I_stim : array (ny, nx)
        External stimulus current density applied over this time step.
    dt : float
        Global time step length in ms.

    Returns
    -------
    V_next, w_next : arrays
        Updated fields after one step.
    """
    C_m = float(params.get("C_m", 1.0))
    tau_v = float(params.get("tau_v", 20.0))
    tau_w = float(params.get("tau_w", 80.0))
    V_rest = float(params.get("V_rest", 0.0))
    V_min = params.get("V_min", None)
    V_max = params.get("V_max", None)

    V = np.asarray(V, dtype=float)
    w = np.asarray(w, dtype=float)
    I_stim = np.asarray(I_stim, dtype=float)

    # Decide whether to sub-step for stability.
    n_sub, dt_sub = _stable_substep(mesh, float(dt))

    for _ in range(n_sub):
        # Diffusion term
        diff = _anisotropic_diffusion(V, mesh)

        # Linear relaxation towards V_rest.
        reaction = -(V - V_rest) / tau_v

        # Explicit Euler update for V.
        dVdt = (diff + reaction + I_stim) / C_m
        V = V + dt_sub * dVdt

        # Simple low-pass dynamics for w following V.
        dw_dt = (V - w) / tau_w
        w = w + dt_sub * dw_dt

        # Soft numerical clamping of V to avoid runaway modes.
        if V_min is not None or V_max is not None:
            V = np.clip(
                V,
                V_min if V_min is not None else -np.inf,
                V_max if V_max is not None else np.inf,
            )

    return V, w


__all__ = [
    "build_D_components",
    "default_params",
    "step_relaxed_monodomain",
]

