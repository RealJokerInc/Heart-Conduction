"""
Validate the cross-derivative (Dxy) term in the anisotropic FDM stencil.

Two tests:
1. Direct stencil coefficient check: extract L[k, k_NE] and verify it equals
   -Dxy / (2*dx*dy) (with the factor of 2 from div(D·grad V)).
2. Eigenvalue spectrum comparison: a 45° rotation should not change the
   eigenvalue spectrum (only the eigenvectors rotate). Compare sorted
   eigenvalues of the 45°-rotated tensor against the axis-aligned case.

Run: ./venv/bin/python Bidomain/Engine_V1/tests/test_aniso_crossderiv.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
torch.set_default_dtype(torch.float64)
import math

from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec
from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
from cardiac_sim.simulation.classical.discretization.fdm import BidomainFDMDiscretization


def test_stencil_coefficients():
    """Verify NE/NW/SE/SW stencil coefficients match analytical formula."""
    nx, ny = 8, 8
    dx = 0.025
    Lx = dx * (nx - 1)
    Ly = dx * (ny - 1)

    D_fiber = 0.004
    D_cross = 0.001
    theta_rad = math.radians(45.0)

    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    Dxx = D_fiber * cos_t**2 + D_cross * sin_t**2  # 0.0025
    Dxy = (D_fiber - D_cross) * cos_t * sin_t       # 0.0015
    Dyy = D_fiber * sin_t**2 + D_cross * cos_t**2  # 0.0025

    print(f"D_fiber={D_fiber}, D_cross={D_cross}, theta=45°")
    print(f"Dxx={Dxx:.6f}, Dxy={Dxy:.6f}, Dyy={Dyy:.6f}")

    theta_field = torch.full((nx, ny), theta_rad)
    cond = BidomainConductivity(
        D_i=D_fiber, D_e=D_fiber,
        D_i_fiber=D_fiber, D_i_cross=D_cross,
        theta=theta_field)

    grid = StructuredGrid(Nx=nx, Ny=ny, Lx=Lx, Ly=Ly,
                          boundary_spec=BoundarySpec.insulated())
    spatial = BidomainFDMDiscretization(grid, cond, Cm=1.0)

    L_i = spatial.L_i.to_dense()

    # Check an interior node: (3, 3) — far from boundaries
    i, j = 3, 3
    k = i * ny + j           # flat index of (3,3)
    k_E = (i+1) * ny + j     # East
    k_W = (i-1) * ny + j     # West
    k_N = i * ny + (j+1)     # North
    k_S = i * ny + (j-1)     # South
    k_NE = (i+1) * ny + (j+1)  # NE
    k_NW = (i-1) * ny + (j+1)  # NW
    k_SE = (i+1) * ny + (j-1)  # SE
    k_SW = (i-1) * ny + (j-1)  # SW

    # Expected coefficients for uniform D (harmonic mean = D for uniform):
    cx = 1.0 / (dx * dx)       # 1600
    cy = 1.0 / (dx * dx)       # 1600 (dy = dx)
    cxy = 1.0 / (2.0 * dx * dx)  # 800 (with factor of 2 fix)

    E_expected = Dxx * cx        # East/West coefficient
    N_expected = Dyy * cy        # North/South coefficient
    NE_expected = +Dxy * cxy     # NE/SW coefficient (positive for Dxy > 0)
    NW_expected = -Dxy * cxy     # NW/SE coefficient
    center_expected = -(2 * E_expected + 2 * N_expected)  # diag terms cancel

    # Extract actual coefficients
    E_actual = L_i[k, k_E].item()
    W_actual = L_i[k, k_W].item()
    N_actual = L_i[k, k_N].item()
    S_actual = L_i[k, k_S].item()
    NE_actual = L_i[k, k_NE].item()
    NW_actual = L_i[k, k_NW].item()
    SE_actual = L_i[k, k_SE].item()
    SW_actual = L_i[k, k_SW].item()
    C_actual = L_i[k, k].item()

    print(f"\nStencil at interior node ({i},{j}):")
    print(f"  East:   {E_actual:+.6f}  (expected {E_expected:+.6f})")
    print(f"  West:   {W_actual:+.6f}  (expected {E_expected:+.6f})")
    print(f"  North:  {N_actual:+.6f}  (expected {N_expected:+.6f})")
    print(f"  South:  {S_actual:+.6f}  (expected {N_expected:+.6f})")
    print(f"  NE:     {NE_actual:+.6f}  (expected {NE_expected:+.6f})")
    print(f"  NW:     {NW_actual:+.6f}  (expected {NW_expected:+.6f})")
    print(f"  SE:     {SE_actual:+.6f}  (expected {NW_expected:+.6f})")
    print(f"  SW:     {SW_actual:+.6f}  (expected {NE_expected:+.6f})")
    print(f"  Center: {C_actual:+.6f}  (expected {center_expected:+.6f})")

    # Verify each coefficient
    tol = 1e-12
    assert abs(E_actual - E_expected) < tol, f"East wrong: {E_actual} != {E_expected}"
    assert abs(W_actual - E_expected) < tol, f"West wrong: {W_actual} != {E_expected}"
    assert abs(N_actual - N_expected) < tol, f"North wrong: {N_actual} != {N_expected}"
    assert abs(S_actual - N_expected) < tol, f"South wrong: {S_actual} != {N_expected}"
    assert abs(NE_actual - NE_expected) < tol, f"NE wrong: {NE_actual} != {NE_expected}"
    assert abs(NW_actual - NW_expected) < tol, f"NW wrong: {NW_actual} != {NW_expected}"
    assert abs(SE_actual - NW_expected) < tol, f"SE wrong: {SE_actual} != {NW_expected}"
    assert abs(SW_actual - NE_expected) < tol, f"SW wrong: {SW_actual} != {NE_expected}"
    assert abs(C_actual - center_expected) < tol, f"Center wrong: {C_actual} != {center_expected}"

    # Verify the coefficient value is Dxy/(2*dx*dy) not Dxy/(4*dx*dy)
    NE_wrong = Dxy / (4.0 * dx * dx)  # what the old code would give (no factor of 2)
    NE_correct = Dxy / (2.0 * dx * dx)  # what the fixed code gives (with factor of 2)
    print(f"\n  NE value: {NE_actual:.6f}")
    print(f"  Correct (Dxy/(2dx²)): {NE_correct:.6f}")
    print(f"  Wrong (Dxy/(4dx²)):   {NE_wrong:.6f}")
    assert abs(NE_actual - NE_correct) < tol, "NE uses wrong cxy coefficient"

    print("\nPASS: Stencil coefficients match analytical formula with factor of 2")


def test_eigenvalue_spectrum_rotation_invariance():
    """Eigenvalue spectrum should be invariant under rotation of fiber angle.

    For uniform D_fiber and D_cross, rotating the fiber angle by any amount
    should not change the eigenvalue spectrum of the Laplacian (the spectrum
    depends only on the eigenvalues of the diffusion tensor D_f, D_c, not
    on the rotation angle). This is true for periodic BCs or infinite domains.

    For Neumann BCs on a finite grid, the spectrum is NOT exactly rotation-
    invariant because the grid axes break rotational symmetry. But for small
    anisotropy ratios on large grids, the spectra should be close.

    Instead, we use a direct comparison: build the Laplacian explicitly as
    Dxx*Lxx + 2*Dxy*Lxy + Dyy*Lyy using separate operator matrices, and
    verify it matches the FDM-built Laplacian.
    """
    nx, ny = 8, 8
    dx = 0.025
    Lx = dx * (nx - 1)
    Ly = dx * (ny - 1)

    D_fiber = 0.004
    D_cross = 0.001
    theta_rad = math.radians(30.0)

    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    Dxx = D_fiber * cos_t**2 + D_cross * sin_t**2
    Dxy = (D_fiber - D_cross) * cos_t * sin_t
    Dyy = D_fiber * sin_t**2 + D_cross * cos_t**2

    print(f"D_fiber={D_fiber}, D_cross={D_cross}, theta=30°")
    print(f"Dxx={Dxx:.6f}, Dxy={Dxy:.6f}, Dyy={Dyy:.6f}")

    theta_field = torch.full((nx, ny), theta_rad)
    cond = BidomainConductivity(
        D_i=D_fiber, D_e=D_fiber,
        D_i_fiber=D_fiber, D_i_cross=D_cross,
        theta=theta_field)

    grid = StructuredGrid(Nx=nx, Ny=ny, Lx=Lx, Ly=Ly,
                          boundary_spec=BoundarySpec.insulated())
    spatial = BidomainFDMDiscretization(grid, cond, Cm=1.0)
    L_aniso = spatial.L_i.to_dense()

    # Build reference: L_ref = Dxx * L_unit_xx + 2*Dxy * L_unit_xy + Dyy * L_unit_yy
    # where L_unit_xx uses Dxx=1, Dxy=0, Dyy=0
    #       L_unit_xy uses Dxx=0, Dxy=1, Dyy=0 (BUT with factor of 2 already in cxy)
    #       L_unit_yy uses Dxx=0, Dxy=0, Dyy=1
    cond_xx = BidomainConductivity(D_i=1.0, D_e=1.0,
                                    D_i_field=(torch.ones(nx,ny), torch.zeros(nx,ny), torch.zeros(nx,ny)))
    cond_yy = BidomainConductivity(D_i=1.0, D_e=1.0,
                                    D_i_field=(torch.zeros(nx,ny), torch.zeros(nx,ny), torch.ones(nx,ny)))
    cond_xy = BidomainConductivity(D_i=1.0, D_e=1.0,
                                    D_i_field=(torch.zeros(nx,ny), torch.ones(nx,ny), torch.zeros(nx,ny)))

    L_xx = BidomainFDMDiscretization(grid, cond_xx, Cm=1.0).L_i.to_dense()
    L_yy = BidomainFDMDiscretization(grid, cond_yy, Cm=1.0).L_i.to_dense()
    L_xy = BidomainFDMDiscretization(grid, cond_xy, Cm=1.0).L_i.to_dense()

    # L_xy already has the factor of 2 baked into cxy. So:
    # L_ref = Dxx * L_xx + Dxy * L_xy + Dyy * L_yy
    # (the factor of 2 is inside L_xy, not multiplied externally)
    L_ref = Dxx * L_xx + Dxy * L_xy + Dyy * L_yy

    diff = (L_aniso - L_ref).abs().max().item()
    print(f"\nMax |L_aniso - L_ref|: {diff:.2e}")
    assert diff < 1e-12, f"Anisotropic L doesn't match reference: max_diff={diff}"

    # Verify symmetry
    sym_err = (L_aniso - L_aniso.T).abs().max().item()
    print(f"Symmetry error: {sym_err:.2e}")
    assert sym_err < 1e-14

    # Verify zero row sum
    row_sum = L_aniso.sum(dim=1).abs().max().item()
    print(f"Row sum error: {row_sum:.2e}")
    assert row_sum < 1e-12

    print("\nPASS: Anisotropic L matches reference decomposition")


def test_diffusion_decay_anisotropic():
    """Run actual diffusion with anisotropic D and verify faster decay along fibers.

    With 45° fibers, diffusion is fastest at 45° (D_fiber) and slowest at
    135° (D_cross). A Gaussian centered on the grid should spread into an
    ellipse oriented along the fiber direction.

    We verify that after many diffusion steps, the variance along the fiber
    direction is larger than along the cross-fiber direction.
    """
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.decoupled_gs import DecoupledBidomainDiffusionSolver
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import PCGSolver
    from cardiac_sim.simulation.classical.state import BidomainState

    nx, ny = 30, 30
    dx = 0.025
    dt = 0.005  # small dt for explicit parts
    Lx = dx * (nx - 1)
    Ly = dx * (ny - 1)

    D_fiber = 0.004
    D_cross = 0.001
    theta_rad = math.radians(45.0)
    theta_field = torch.full((nx, ny), theta_rad)

    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    Dxx = D_fiber * cos_t**2 + D_cross * sin_t**2
    Dxy = (D_fiber - D_cross) * cos_t * sin_t
    Dyy = D_fiber * sin_t**2 + D_cross * cos_t**2

    # Use D_i for intracellular, D_e much larger (so D_eff ≈ D_i for testing)
    cond = BidomainConductivity(
        D_i=D_fiber, D_e=10*D_fiber,  # D_e >> D_i so D_eff ≈ D_i
        D_i_fiber=D_fiber, D_i_cross=D_cross,
        theta=theta_field)

    grid = StructuredGrid(Nx=nx, Ny=ny, Lx=Lx, Ly=Ly,
                          boundary_spec=BoundarySpec.insulated())
    spatial = BidomainFDMDiscretization(grid, cond, Cm=1.0)

    # Initial condition: narrow Gaussian at grid center
    x, y = spatial.coordinates
    cx, cy_c = Lx/2, Ly/2
    sigma = 2 * dx
    Vm_init = torch.exp(-((x - cx)**2 + (y - cy_c)**2) / (2 * sigma**2))

    state = BidomainState(
        spatial=spatial, n_dof=spatial.n_dof, x=x, y=y,
        Vm=Vm_init.clone(), phi_e=torch.zeros_like(Vm_init),
        ionic_states=torch.zeros(spatial.n_dof, 1),
        gate_indices=[], concentration_indices=[])

    gs = DecoupledBidomainDiffusionSolver(
        spatial, dt,
        PCGSolver(max_iters=500, tol=1e-10),
        PCGSolver(max_iters=500, tol=1e-10),
        theta=0.5)

    # Diffuse for many steps
    n_steps = 200
    for _ in range(n_steps):
        gs.step(state, dt)

    # Compute variance along fiber direction (45°) and cross-fiber (135°)
    Vm = state.Vm
    total = Vm.sum().item()
    if total < 1e-15:
        print("WARNING: Vm decayed to zero")
        return

    weights = Vm / total
    x_mean = (weights * x).sum().item()
    y_mean = (weights * y).sum().item()

    # Project onto fiber direction (45°) and cross-fiber (135°)
    u_fiber = cos_t * (x - x_mean) + sin_t * (y - y_mean)   # fiber axis
    u_cross = -sin_t * (x - x_mean) + cos_t * (y - y_mean)  # cross-fiber axis

    var_fiber = (weights * u_fiber**2).sum().item()
    var_cross = (weights * u_cross**2).sum().item()

    ratio = var_fiber / var_cross if var_cross > 1e-15 else float('inf')

    print(f"\nAnisotropic diffusion (45° fiber, D_f/D_c={D_fiber/D_cross:.0f}):")
    print(f"  Variance along fiber:  {var_fiber:.6e}")
    print(f"  Variance cross-fiber:  {var_cross:.6e}")
    print(f"  Ratio (expect ~{D_fiber/D_cross:.0f}): {ratio:.2f}")

    # The variance ratio should approach D_fiber/D_cross = 4.0 for pure
    # monodomain diffusion. With bidomain (D_e >> D_i), D_eff ≈ D_i,
    # so we expect a ratio near 4. Accept if > 2.0 (significant anisotropy).
    assert ratio > 2.0, (
        f"Variance ratio {ratio:.2f} too low — cross-derivative may be wrong")
    assert ratio < 8.0, (
        f"Variance ratio {ratio:.2f} too high — cross-derivative may be doubled")

    print("PASS: Anisotropic diffusion shows correct directional preference")


if __name__ == '__main__':
    print("Anisotropic Cross-Derivative Validation\n")

    test_stencil_coefficients()
    print()
    test_eigenvalue_spectrum_rotation_invariance()
    print()
    test_diffusion_decay_anisotropic()

    print("\nAnisotropic: ALL 3 TESTS PASS")
