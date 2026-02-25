"""
monodomain_2d.py
================

GROUND-TRUTH MODEL SPEC (MATH CONTRACT FOR CURSOR)

We model a 2D sheet of cardiac tissue with an anisotropic monodomain equation
coupled to the Aliev–Panfilov ionic model.

----------------------------------------------------------------------
0. VARIABLES AND UNITS
----------------------------------------------------------------------

State variables:
  V(x, y, t)  [mV] : transmembrane voltage
  w(x, y, t)  [-]  : recovery variable (dimensionless)

Parameters:
  C_m                 : membrane capacitance per unit area (scalar)
  D_parallel, D_perp  : diffusion coefficients along / across fibers [mm^2/ms]
  epsilon_tissue(x,y) : conduction mask in [0, 1]
  epsilon_AP(V, w)    : AP recovery time-scale function
  I_stim(x,y,t)       : external stimulus current density

Units for simulation:
  t in ms, x,y in mm, V in mV.

----------------------------------------------------------------------
1. GOVERNING EQUATIONS
----------------------------------------------------------------------

Monodomain PDE with heterogeneous conduction:

  C_m * dV/dt = div( epsilon_tissue(x,y) * D(x,y) * grad(V) )
                - I_ion(V, w)
                + I_stim(x, y, t)

Where:
  D(x,y) is a 2x2 diffusion tensor (anisotropic),
  epsilon_tissue(x,y) is a scalar field in [0,1].

Define effective tensor:
  D_eff(x,y) = epsilon_tissue(x,y) * D(x,y)

So equivalently:

  C_m * dV/dt = div( D_eff(x,y) * grad(V) )
                - I_ion(V, w)
                + I_stim(x, y, t)

----------------------------------------------------------------------
2. IONIC MODEL: ALIEV–PANFILOV
----------------------------------------------------------------------

Ionic current:
  I_ion(V, w) = k * V * (V - a) * (V - 1) + V * w

Recovery dynamics:
  dw/dt = epsilon_AP(V, w) * ( -k * V * (V - a - 1) - w )

Time-scale function:
  epsilon_AP(V, w) = epsilon0 + mu1 * w / (V + mu2)

Notes:
  - Use epsilon_AP(V, w) for the AP time-scale.
  - Use epsilon_tissue(x, y) for the spatial conduction mask.
  They are different concepts even though both are called "epsilon".

----------------------------------------------------------------------
3. TISSUE CONDUCTION MASK epsilon_tissue(x,y)
----------------------------------------------------------------------

epsilon_tissue(x,y) is a scalar field in [0,1]:

  ~1 : normal conduction
  ~0 : non-conductive infarct core
  between 0 and 1 : border zone / reduced conduction

It appears ONLY in the diffusion term:

  C_m * dV/dt = div( epsilon_tissue * D * grad(V) ) - I_ion + I_stim

We will use TWO options for epsilon_tissue(x,y):

(1) Binary epsilon (sharp infarct)
----------------------------------

Let Omega_scar be the infarct core. Then

  epsilon_tissue(x,y) =
      1          if (x,y) is not in Omega_scar (healthy)
      eps_core   if (x,y) is in Omega_scar (scar)

with 0 <= eps_core << 1.

  eps_core = 0   -> fully non-conductive core
  0 < eps_core < 1 -> very weak conduction

Interpretation:
  A simple "on/off" conduction model with a sharp border.

(2) Sigmoid epsilon (smooth border zone)
----------------------------------------

We define a signed distance d(x,y) relative to the infarct boundary:

  d(x,y) < 0 : inside the core
  d(x,y) = 0 : on the core boundary
  d(x,y) > 0 : outside (healthy)

Then

  epsilon_tissue(x,y) =
      eps_core + (1 - eps_core) * S( d(x,y) / delta )

where:
  eps_core in [0, 1) as above
  delta > 0 is the border thickness scale (mm)
  S(z) is a sigmoid, e.g. S(z) = 1 / (1 + exp(-k_sig * z))

Intuition:
  deep inside core (d << 0)   -> epsilon_tissue ~ eps_core
  far in healthy (d >> 0)     -> epsilon_tissue ~ 1
  around boundary |d| ~ delta -> smooth ramp between eps_core and 1

Implementation idea:
  z = d / delta
  S = 1 / (1 + exp(-k_sig * z))
  epsilon_tissue = eps_core + (1 - eps_core) * S

----------------------------------------------------------------------
4. DIFFUSION TENSOR AND FIBER ORIENTATION
----------------------------------------------------------------------

Let theta(x,y) be the fiber angle (radians from x-axis).
Define fiber unit vector:

  f = [cos(theta), sin(theta)]^T

Then the diffusion tensor is:

  D(x,y) = D_parallel * (f f^T) + D_perp * (I - f f^T)

where I is the 2x2 identity matrix.

Effective tensor used in the PDE:

  D_eff(x,y) = epsilon_tissue(x,y) * D(x,y)

----------------------------------------------------------------------
5. BOUNDARY CONDITIONS (BCs)
----------------------------------------------------------------------

We use no-flux (Neumann) BCs on the boundary of the domain:

  n · ( D_eff(x,y) * grad(V) ) = 0   on the outer boundary

where n is the outward unit normal.

Interpretation:
  No current leaves the computational domain; electrically isolated patch.

Finite difference:
  Enforce zero normal flux using ghost nodes or one-sided differences.

----------------------------------------------------------------------
6. INITIAL CONDITIONS (ICs)
----------------------------------------------------------------------

At t = 0:

  V(x,y,0) = V_rest
  w(x,y,0) = w_rest

Typical AP choices:
  V_rest ~ 0
  w_rest ~ 0

All cells are initially at rest; stimulus triggers depolarization.

----------------------------------------------------------------------
7. STIMULUS CURRENT I_stim(x,y,t)
----------------------------------------------------------------------

The stimulus is a source term in the V-equation:

  C_m * dV/dt = div(D_eff * grad(V)) - I_ion(V,w) + I_stim(x,y,t)

Simple rectangular stimulus:

  I_stim(x,y,t) = I0             if (x,y) in Omega_stim and
                                             t0 <= t <= t0 + dt_stim
                   0             otherwise

Where:
  Omega_stim : set of grid points being stimulated
  I0         : amplitude
  t0         : onset time
  dt_stim    : duration

----------------------------------------------------------------------
8. GEOMETRY AND GRID (NUMERICAL)
----------------------------------------------------------------------

Domain:
  x in [0, Lx], y in [0, Ly]

Grid:
  Nx points along x, Ny points along y
  dx = Lx / (Nx - 1)
  dy = Ly / (Ny - 1)

Grid indices:
  i = 0..Nx-1, j = 0..Ny-1
  x_i = i * dx
  y_j = j * dy

We store:
  V[i, j](t)  ~ V(x_i, y_j, t)
  w[i, j](t)  ~ w(x_i, y_j, t)
  epsilon_tissue[i, j]
  theta[i, j]

----------------------------------------------------------------------
9. DISCRETIZATION SUMMARY (IMPLEMENTATION HINTS)
----------------------------------------------------------------------

Spatial:
  Use second-order central finite differences for the diffusion term.
  Implement anisotropic diffusion by discretizing components of D_eff:
    D_xx, D_xy, D_yy.

Time:
  Start with explicit Euler:
    V^{n+1} = V^n + dt * ( (1/C_m) * [ div(D_eff grad V^n)
                                       - I_ion(V^n, w^n)
                                       + I_stim^n ] )
    w^{n+1} = w^n + dt * ( epsilon_AP(V^n, w^n) *
                           ( -k * V^n * (V^n - a - 1) - w^n ) )

  dt must satisfy stability constraints based on max diffusion and dx, dy.

Data structures:
  Numpy arrays, shape (Ny, Nx) or (Nx, Ny), but be consistent.
  epsilon_tissue can be generated with either:
    - "binary" mode
    - "sigmoid" mode

This file's docstring is the ground-truth math specification.
Cursor should treat it as the source of truth and must NOT change the math,
only the numerical implementation or parameters.

======================================================================
END OF SPECIFICATION
======================================================================
"""

from __future__ import annotations

import numpy as np


# -------------------------------------------------------------------
#  Helper functions to build epsilon_tissue
# -------------------------------------------------------------------

def build_epsilon_tissue_binary(shape, scar_mask, eps_core=0.0):
    """
    Build epsilon_tissue for the BINARY option.

    Parameters
    ----------
    shape : tuple (ny, nx)
        Grid shape.
    scar_mask : 2D boolean array (ny, nx)
        True where tissue is scar core.
    eps_core : float
        Conduction level in scar core (0 = non-conductive).

    Returns
    -------
    epsilon_tissue : 2D array
        Values in [eps_core, 1.0].
    """
    ny, nx = shape
    eps = np.ones((ny, nx), dtype=float)
    eps[scar_mask] = eps_core
    return eps


def build_epsilon_tissue_sigmoid(d, eps_core=0.0, delta=1.0, k_sig=4.0):
    """
    Build epsilon_tissue for the SIGMOID option from a distance field d(x,y).

    Parameters
    ----------
    d : 2D array
        Signed distance to infarct boundary:
        d < 0 core, d = 0 boundary, d > 0 healthy.
    eps_core : float
        Conduction level deep in core.
    delta : float
        Border thickness scale.
    k_sig : float
        Sigmoid steepness.

    Returns
    -------
    epsilon_tissue : 2D array in [eps_core, 1.0]
    """
    z = d / delta
    S = 1.0 / (1.0 + np.exp(-k_sig * z))
    eps = eps_core + (1.0 - eps_core) * S
    return eps


# -------------------------------------------------------------------
#  Diffusion tensor construction
# -------------------------------------------------------------------

def build_D_components(theta, D_parallel, D_perp, epsilon_tissue):
    """
    Construct D_eff components (Dxx, Dxy, Dyy) for anisotropic diffusion.

    D(x,y) = D_parallel * f f^T + D_perp * (I - f f^T)
    D_eff(x,y) = epsilon_tissue(x,y) * D(x,y)

    Parameters
    ----------
    theta : 2D array
        Fiber angle in radians.
    D_parallel : float
        Diffusion along fibers.
    D_perp : float
        Diffusion across fibers.
    epsilon_tissue : 2D array
        Conduction mask in [0,1].

    Returns
    -------
    Dxx, Dxy, Dyy : 2D arrays
        Components of D_eff.
    """
    c = np.cos(theta)
    s = np.sin(theta)

    # f f^T components
    fxf = c * c
    fyf = s * s
    fxy = c * s

    Dxx = D_parallel * fxf + D_perp * (1.0 - fxf)
    Dyy = D_parallel * fyf + D_perp * (1.0 - fyf)
    Dxy = D_parallel * fxy - D_perp * fxy  # (D_parallel - D_perp)*fxy

    Dxx *= epsilon_tissue
    Dyy *= epsilon_tissue
    Dxy *= epsilon_tissue

    return Dxx, Dxy, Dyy


# -------------------------------------------------------------------
#  Aliev–Panfilov ionic model
# -------------------------------------------------------------------

def epsilon_AP(V, w, epsilon0, mu1, mu2):
    """
    Time-scale function for AP: epsilon_AP(V,w) = epsilon0 + mu1 * w / (V + mu2)

    Assumes V + mu2 != 0 (use small epsilon if needed).
    """
    return epsilon0 + mu1 * w / (V + mu2)


def I_ion(V, w, k, a):
    """
    Aliev–Panfilov ionic current:
      I_ion = k * V * (V - a) * (V - 1) + V * w
    """
    return k * V * (V - a) * (V - 1) + V * w


def reaction_step(V, w, params, dt):
    """
    Update (V,w) due ONLY to the reaction (ionic) terms using explicit Euler.

    This does NOT include diffusion or stimulus; those are handled elsewhere.

    dV/dt (reaction) = - I_ion(V, w) / C_m
    dw/dt            = epsilon_AP(V,w) * ( -k * V * (V - a - 1) - w )

    Parameters
    ----------
    V, w : 2D arrays
        Voltage and recovery variable at time n.
    params : dict
        Contains keys:
          'C_m', 'k', 'a', 'epsilon0', 'mu1', 'mu2'
    dt : float
        Time step.

    Returns
    -------
    V_new, w_new : 2D arrays
        Updated fields after reaction step.
    """
    C_m = params["C_m"]
    k = params["k"]
    a = params["a"]
    epsilon0 = params["epsilon0"]
    mu1 = params["mu1"]
    mu2 = params["mu2"]

    # Ionic current
    I = I_ion(V, w, k, a)

    # dV/dt (reaction only)
    dVdt_react = -I / C_m

    # dw/dt
    eps_ap = epsilon_AP(V, w, epsilon0, mu1, mu2)
    dwdt = eps_ap * (-k * V * (V - a - 1.0) - w)

    V_new = V + dt * dVdt_react
    w_new = w + dt * dwdt

    return V_new, w_new


# -------------------------------------------------------------------
#  Diffusion step (finite differences, explicit)
# -------------------------------------------------------------------

def diffusion_step(V, Dxx, Dxy, Dyy, dx, dy, dt, C_m):
    """
    Explicit Euler update for the diffusion term:

      C_m * dV/dt = div( D_eff * grad(V) )

    Implementation: approximate div(D grad V) using central differences and
    anisotropic tensor components (Dxx, Dxy, Dyy).

    This is a simple reference implementation; can be optimized or replaced
    later without changing the math.
    """
    def _centered_diff_x(field, spacing):
        padded = np.pad(field, ((0, 0), (1, 1)), mode="edge")
        return (padded[:, 2:] - padded[:, :-2]) / (2.0 * spacing)

    def _centered_diff_y(field, spacing):
        padded = np.pad(field, ((1, 1), (0, 0)), mode="edge")
        return (padded[2:, :] - padded[:-2, :]) / (2.0 * spacing)

    V_new = V.copy()

    dVdx = _centered_diff_x(V, dx)
    dVdy = _centered_diff_y(V, dy)

    # Flux components with zero-flux BC enforced by edge padding.
    Jx = Dxx * dVdx + Dxy * dVdy
    Jy = Dxy * dVdx + Dyy * dVdy

    dJxdx = _centered_diff_x(Jx, dx)
    dJydy = _centered_diff_y(Jy, dy)

    div_term = dJxdx + dJydy
    V_new += dt * (div_term / C_m)

    return V_new


# -------------------------------------------------------------------
#  Main update function (reaction + diffusion + stimulus)
# -------------------------------------------------------------------

def step_monodomain(V, w, Dxx, Dxy, Dyy, dx, dy, dt, params, I_stim):
    """
    Advance (V, w) one time-step:

      C_m dV/dt = div(D_eff grad V) - I_ion(V,w) + I_stim
      dw/dt     = epsilon_AP(V,w) * ( -k V (V - a - 1) - w )

    We split into:
      1) diffusion + stimulus in V-equation
      2) reaction update for V and w

    This is just one possible operator splitting; the math stays the same.
    """
    C_m = params["C_m"]

    # 1) Diffusion + stimulus on V
    V_diff = diffusion_step(V, Dxx, Dxy, Dyy, dx, dy, dt, C_m)
    V_diff += dt * (I_stim / C_m)

    # 2) Reaction (ionic) on (V, w)
    V_new, w_new = reaction_step(V_diff, w, params, dt)
    return V_new, w_new


# -------------------------------------------------------------------
#  Example: setup a tiny 3x3 conceptual block (for sanity checks)
# -------------------------------------------------------------------

def build_3x3_test():
    """
    Tiny conceptual example: 3x3 tissue block, center cell stimulated.

    This is NOT meant for real simulations; it's mainly to visualize how
    coupling between blocks would work under the governing equation.
    """
    nx = ny = 3
    dx = dy = 1.0

    V = np.zeros((ny, nx), dtype=float)
    w = np.zeros_like(V)

    # Fiber direction constant (e.g. horizontal)
    theta = np.zeros_like(V)  # 0 rad -> along x

    # No scar: epsilon_tissue = 1 everywhere
    epsilon_tissue = np.ones_like(V)

    # Diffusion parameters
    D_parallel = 1.0
    D_perp = 0.3
    Dxx, Dxy, Dyy = build_D_components(theta, D_parallel, D_perp, epsilon_tissue)

    # Params for AP
    params = dict(
        C_m=1.0,
        k=8.0,
        a=0.1,
        epsilon0=0.01,
        mu1=0.2,
        mu2=0.3,
    )

    # Center stimulus region: only grid (1,1)
    I_stim = np.zeros_like(V)
    I_stim[1, 1] = 1.0  # arbitrary pulse for one step

    return V, w, Dxx, Dxy, Dyy, dx, dy, params, I_stim


if __name__ == "__main__":
    # Minimal smoke test: one step on 3x3 grid.
    V, w, Dxx, Dxy, Dyy, dx, dy, params, I_stim = build_3x3_test()
    dt = 0.01

    V_new, w_new = step_monodomain(V, w, Dxx, Dxy, Dyy, dx, dy, dt, params, I_stim)
    print("V_new:\n", V_new)
    print("w_new:\n", w_new)
