"""
Shared test helpers for diffusion solver validation.

Provides:
- _make_spatial: Create a small grid + spatial discretization
- _make_cosine_state: Create state with cosine mode initial condition
- validate_deff_cosine: Tight D_eff validation via cosine mode decay
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import math
torch.set_default_dtype(torch.float64)

from cv_shared import D_I, D_E, D_EFF, DX, DT


def _make_spatial(nx, ny):
    """Create a BidomainFDMDiscretization for testing."""
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec
    from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
    from cardiac_sim.simulation.classical.discretization.fdm import BidomainFDMDiscretization

    Lx = DX * (nx - 1)
    Ly = DX * (ny - 1)
    grid = StructuredGrid(Nx=nx, Ny=ny, Lx=Lx, Ly=Ly,
                          boundary_spec=BoundarySpec.insulated())
    cond = BidomainConductivity(D_i=D_I, D_e=D_E)
    return BidomainFDMDiscretization(grid, cond, Cm=1.0)


def _make_cosine_state(nx, ny, spatial, kx=1, ky=1):
    """Create state with exact eigenmode of the face-based discrete Laplacian.

    The face-based FDM stencil (graph Laplacian of the grid) has eigenfunctions:
        cos(pi*kx*(2i+1)/(2*nx)) * cos(pi*ky*(2j+1)/(2*ny))
    with eigenvalues:
        lambda = -(2/dx^2)(1-cos(pi*kx/nx)) - (2/dy^2)(1-cos(pi*ky/ny))

    This is a pure eigenmode — it decays as exp(-D_eff * |lambda| * t)
    with no multi-mode contamination.
    """
    from cardiac_sim.simulation.classical.state import BidomainState

    x, y = spatial.coordinates

    i_idx = torch.arange(nx, dtype=torch.float64)
    j_idx = torch.arange(ny, dtype=torch.float64)
    ii, jj = torch.meshgrid(i_idx, j_idx, indexing='ij')

    Vm_grid = (torch.cos(math.pi * kx * (2*ii + 1) / (2*nx)) *
               torch.cos(math.pi * ky * (2*jj + 1) / (2*ny)))
    Vm = Vm_grid.reshape(-1)

    return BidomainState(
        spatial=spatial, n_dof=spatial.n_dof, x=x, y=y,
        Vm=Vm.clone(), phi_e=torch.zeros_like(Vm),
        ionic_states=torch.zeros(spatial.n_dof, 1),
        gate_indices=[], concentration_indices=[],
    )


def validate_deff_cosine(solver, spatial, dt, nx, ny, n_steps=200,
                          kx=1, ky=1, tol=0.05, label=""):
    """Validate D_eff using discrete eigenmode exponential decay.

    The face-based FDM eigenmode cos(pi*kx*(2i+1)/(2*nx))*cos(...) decays as:
        amplitude(t) = amplitude(0) * exp(-D_eff * |lambda| * t)
    where lambda = (2/dx^2)(1-cos(pi*kx/nx)) + (2/dy^2)(1-cos(pi*ky/ny))
    is the discrete eigenvalue of the graph Laplacian.

    This measures D_eff to ~0.1% on a 30x30 grid because the initial
    condition is an exact eigenmode of the discrete operator.

    Returns (measured_D_eff, expected_D_eff, rel_err).
    """
    state = _make_cosine_state(nx, ny, spatial, kx=kx, ky=ky)
    amp_0 = state.Vm.abs().max().item()

    for _ in range(n_steps):
        solver.step(state, dt)

    amp_final = state.Vm.abs().max().item()
    t_total = n_steps * dt

    # Discrete eigenvalue of face-based Laplacian (graph Laplacian of grid)
    lam = ((2.0 / DX**2) * (1.0 - math.cos(math.pi * kx / nx)) +
           (2.0 / DX**2) * (1.0 - math.cos(math.pi * ky / ny)))

    if amp_final > 1e-15 and amp_0 > 1e-15:
        D_meas = -math.log(amp_final / amp_0) / (lam * t_total)
    else:
        D_meas = 0.0

    rel_err = abs(D_meas - D_EFF) / D_EFF

    print(f"    {label}Amplitude: {amp_0:.6f} -> {amp_final:.6f}")
    print(f"    {label}D_eff measured:  {D_meas:.6e} cm^2/ms")
    print(f"    {label}D_eff expected:  {D_EFF:.6e} cm^2/ms")
    print(f"    {label}Relative error:  {rel_err:.4f}")

    assert rel_err < tol, f"D_eff validation failed: rel_err={rel_err:.4f} > {tol}"
    return D_meas, D_EFF, rel_err
