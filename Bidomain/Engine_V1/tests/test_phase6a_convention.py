"""Phase 6A: Formulation B Convention Verification.

Lightweight tests (seconds) that verify Formulation B (D-based, 1/dt mass term)
gives correct operator scaling and diffusion rates. Chi is fully absorbed into
D = sigma/(chi*Cm) and does NOT appear in any operators.

Tests:
  6A-T1: Operator diagonal scaling (chi param is ignored, mass = 1/dt)
  6A-T2: L_i applied to quadratic field (verify D_i in stencil)
  6A-T3: Bidomain Gaussian diffusion (D_eff variance growth)
  6A-T4: LBM Gaussian diffusion (cross-check against bidomain)

Run: ./venv/bin/python Bidomain/Engine_V1/tests/test_phase6a_convention.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'Monodomain', 'LBM_V1'))

import torch
torch.set_default_dtype(torch.float64)

from cv_shared import D_I, D_E, D_EFF, DX, DT, CM_NUM


# ============================================================
# 6A-T1: Operator diagonal scaling
# ============================================================
def test_6a_t1_operator_scaling():
    """Verify A_para uses Formulation B: mass term = 1/dt (chi is ignored).

    Formulation B: A_para = 1/dt * I - theta * L_i
    Chi is fully absorbed into D = sigma/(chi*Cm). Passing chi to the
    constructor triggers a deprecation warning but has no effect.

    Diagonal at interior node = 1/dt + theta * |L_i center weight|
    """
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec
    from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
    from cardiac_sim.simulation.classical.discretization.fdm import (
        BidomainFDMDiscretization)
    import warnings

    nx, ny = 20, 10
    Lx, Ly = DX * (nx - 1), DX * (ny - 1)
    grid = StructuredGrid(Nx=nx, Ny=ny, Lx=Lx, Ly=Ly,
                          boundary_spec=BoundarySpec.insulated())
    cond = BidomainConductivity(D_i=D_I, D_e=D_E)

    dt = DT
    theta = 0.5

    # Build without chi (Formulation B)
    sp1 = BidomainFDMDiscretization(grid, cond, Cm=1.0)
    A1, _ = sp1.get_parabolic_operators(dt, theta)
    A1d = A1.to_dense()

    # Build with chi=1400 — should produce identical operators (chi ignored)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sp2 = BidomainFDMDiscretization(grid, cond, chi=1400.0, Cm=1.0)
    A2, _ = sp2.get_parabolic_operators(dt, theta)
    A2d = A2.to_dense()

    # Pick an interior node
    i_interior = (nx // 2) * ny + (ny // 2)

    diag_no_chi = A1d[i_interior, i_interior].item()
    diag_chi1400 = A2d[i_interior, i_interior].item()

    # Key check: chi parameter is ignored — both operators identical
    assert abs(diag_no_chi - diag_chi1400) < 1e-12, \
        f"chi should be ignored but diag differs: {diag_no_chi} vs {diag_chi1400}"
    print(f"    chi=None and chi=1400 produce identical operators (diff < 1e-12)")

    # Verify mass term = 1/dt (Formulation B)
    mass_term = 1.0 / dt  # = 100
    ratio = diag_no_chi / mass_term

    print(f"    A_para diagonal = {diag_no_chi:.2f} (mass term 1/dt = {mass_term:.0f})")
    print(f"    Ratio diag/mass: {ratio:.4f}")

    # Stencil adds D_i/dx^2 * theta ~ 0.00124/0.000625 * 0.5 ~ 1.0
    # so ratio ~ (100 + 1) / 100 = 1.01
    assert ratio > 1.0, f"Diagonal should exceed mass term, got {ratio}"
    assert ratio < 1.1, f"Stencil correction unexpectedly large: {ratio}"

    stencil_contribution = diag_no_chi - mass_term
    print(f"    Stencil contribution: {stencil_contribution:.4f}")
    print(f"    Diffusion/mass ratio: {stencil_contribution/mass_term:.4f}")

    print("6A-T1 PASS: Formulation B operator scaling correct (chi ignored)")


# ============================================================
# 6A-T2: L_i applied to quadratic
# ============================================================
def test_6a_t2_laplacian_quadratic():
    """Verify L_i * x^2 = 2*D_i at interior nodes.

    The Laplacian of f(x,y) = x^2 is exactly 2. With the diffusion
    coefficient D_i built into L_i, we expect L_i * (x^2) = 2 * D_i.
    """
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec
    from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
    from cardiac_sim.simulation.classical.discretization.fdm import (
        BidomainFDMDiscretization)

    nx, ny = 30, 20
    Lx, Ly = DX * (nx - 1), DX * (ny - 1)
    grid = StructuredGrid(Nx=nx, Ny=ny, Lx=Lx, Ly=Ly,
                          boundary_spec=BoundarySpec.insulated())
    cond = BidomainConductivity(D_i=D_I, D_e=D_E)
    spatial = BidomainFDMDiscretization(grid, cond, Cm=1.0)

    # Create V = x^2 (flat vector)
    x, y = spatial.coordinates
    V_quad = x ** 2

    # Apply L_i
    LiV = spatial.apply_L_i(V_quad)

    # Reshape to grid for interior extraction
    LiV_grid = grid.flat_to_grid(LiV)

    # Interior nodes (avoid boundaries where Neumann stencil differs)
    interior = LiV_grid[3:-3, 3:-3]
    expected = 2.0 * D_I

    max_err = (interior - expected).abs().max().item()
    mean_val = interior.mean().item()
    rel_err = abs(mean_val - expected) / expected

    print(f"    L_i * x^2: interior mean = {mean_val:.6f}, "
          f"expected = {expected:.6f}")
    print(f"    Max error = {max_err:.2e}, rel error = {rel_err:.4f}")

    assert rel_err < 0.01, f"L_i quadratic test failed: rel_err = {rel_err}"
    print("6A-T2 PASS: L_i contains D_i correctly")


# ============================================================
# 6A-T3: Bidomain Gaussian diffusion
# ============================================================
def test_6a_t3_bidomain_gaussian():
    """Verify Gaussian spreads at rate D_eff in bidomain (Formulation B).

    Pure diffusion (no ionic): initialize Vm as Gaussian, phi_e=0,
    run DecoupledBidomainDiffusionSolver for N steps, measure variance.

    Expected: sigma^2(t) = sigma_0^2 + 2*D_eff*t (2D isotropic)
    The factor is 2*D per dimension, but we measure total 2D variance.
    For 2D: Var_x(t) = sigma_0^2 + 2*D*t, same for Var_y.
    """
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec
    from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
    from cardiac_sim.simulation.classical.discretization.fdm import (
        BidomainFDMDiscretization)
    from cardiac_sim.simulation.classical.solver.diffusion_stepping.decoupled_gs import (
        DecoupledBidomainDiffusionSolver)
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import PCGSolver
    from cardiac_sim.simulation.classical.state import BidomainState

    nx, ny = 50, 50
    dx = DX
    dt = DT
    Lx, Ly = dx * (nx - 1), dx * (ny - 1)

    grid = StructuredGrid(Nx=nx, Ny=ny, Lx=Lx, Ly=Ly,
                          boundary_spec=BoundarySpec.insulated())
    cond = BidomainConductivity(D_i=D_I, D_e=D_E)
    spatial = BidomainFDMDiscretization(grid, cond, Cm=1.0)

    # Gaussian initial condition
    x, y = spatial.coordinates
    x_center = Lx / 2
    y_center = Ly / 2
    sigma_0 = 5 * dx  # Initial width

    Vm = torch.exp(-((x - x_center)**2 + (y - y_center)**2) / (2 * sigma_0**2))
    phi_e = torch.zeros_like(Vm)

    # Create a minimal state (no ionic — we won't use ionic_states)
    state = BidomainState(
        spatial=spatial,
        n_dof=spatial.n_dof,
        x=x, y=y,
        Vm=Vm.clone(),
        phi_e=phi_e,
        ionic_states=torch.zeros(spatial.n_dof, 1),  # dummy
        gate_indices=[],
        concentration_indices=[],
    )

    # Build diffusion solver directly
    para_solver = PCGSolver(max_iters=500, tol=1e-10)
    ellip_solver = PCGSolver(max_iters=500, tol=1e-10)
    diff_solver = DecoupledBidomainDiffusionSolver(
        spatial, dt, para_solver, ellip_solver, theta=0.5)

    # Measure initial variance (x-direction)
    def measure_variance_x(Vm_flat):
        Vm_grid = grid.flat_to_grid(Vm_flat)
        # Marginal in x: sum over y
        marginal = Vm_grid.sum(dim=1)  # (nx,)
        marginal = marginal / marginal.sum()
        x_1d = torch.linspace(0, Lx, nx)
        mean_x = (x_1d * marginal).sum()
        var_x = ((x_1d - mean_x)**2 * marginal).sum()
        return var_x.item()

    var_0 = measure_variance_x(state.Vm)

    # Run 200 diffusion-only steps (t = 200 * 0.01 = 2.0 ms)
    n_steps = 200
    for _ in range(n_steps):
        diff_solver.step(state, dt)

    var_final = measure_variance_x(state.Vm)
    t_total = n_steps * dt

    # Expected variance growth: 2 * D_eff * t (per dimension)
    expected_growth = 2 * D_EFF * t_total
    actual_growth = var_final - var_0

    rel_err = abs(actual_growth - expected_growth) / expected_growth

    print(f"    Initial variance: {var_0:.6f} cm^2")
    print(f"    Final variance:   {var_final:.6f} cm^2")
    print(f"    Actual growth:    {actual_growth:.6f} cm^2")
    print(f"    Expected growth:  {expected_growth:.6f} cm^2 "
          f"(2*D_eff*t = 2*{D_EFF:.6f}*{t_total})")
    print(f"    Relative error:   {rel_err:.4f}")

    # Tolerance is 15% to account for operator splitting error:
    # the decoupled solver uses phi_e^n (lagged) in the parabolic RHS,
    # which slightly over-estimates effective diffusion. This is O(dt)
    # splitting error, NOT a convention bug. With chi=1400, growth would
    # be ~1400x too small (~2.8e-6 instead of ~0.004).
    assert rel_err < 0.15, f"Gaussian variance growth error: {rel_err:.4f}"
    print("6A-T3 PASS: Bidomain Gaussian diffuses at D_eff rate")


# ============================================================
# 6A-T4: LBM Gaussian diffusion cross-check
# ============================================================
def test_6a_t4_lbm_gaussian():
    """Verify LBM D2Q5 Gaussian spreads at same rate as bidomain.

    Uses the same D_eff, same grid, same initial Gaussian.
    Pure diffusion (no ionic model) — uses existing LBM Phase 5 approach.
    """
    from src.lattice import D2Q5
    from src.diffusion import tau_from_D
    from src.state import create_lbm_state, recover_voltage
    from src.collision.bgk import bgk_collide
    from src.streaming.d2q5 import stream_d2q5
    from src.boundary.neumann import apply_neumann_d2q5

    nx, ny = 50, 50
    dx = DX
    dt = DT

    lattice = D2Q5()
    w = torch.tensor(lattice.w, dtype=torch.float64)
    tau = tau_from_D(D_EFF, dx, dt)
    omega = 1.0 / tau

    # Same Gaussian
    Lx = dx * (nx - 1)
    Ly = dx * (ny - 1)
    x_1d = torch.linspace(0, Lx, nx, dtype=torch.float64)
    y_1d = torch.linspace(0, Ly, ny, dtype=torch.float64)
    xx, yy = torch.meshgrid(x_1d, y_1d, indexing='ij')
    x_center = Lx / 2
    y_center = Ly / 2
    sigma_0 = 5 * dx

    V_init = torch.exp(-((xx - x_center)**2 + (yy - y_center)**2) /
                        (2 * sigma_0**2))

    # Initialize LBM distributions
    f = w[:, None, None] * V_init[None, :, :]
    V = V_init.clone()

    # Bounce masks for rectangular domain
    bounce_masks = {}
    for a in range(1, lattice.Q):
        m = torch.zeros(nx, ny, dtype=torch.bool)
        ex, ey = lattice.e[a]
        if ex == 1:   m[-1, :] = True
        if ex == -1:  m[0, :] = True
        if ey == 1:   m[:, -1] = True
        if ey == -1:  m[:, 0] = True
        bounce_masks[a] = m

    # Measure initial x-variance
    def measure_variance_x(V_grid):
        marginal = V_grid.sum(dim=1)
        marginal = marginal / marginal.sum()
        mean_x = (x_1d * marginal).sum()
        var_x = ((x_1d - mean_x)**2 * marginal).sum()
        return var_x.item()

    var_0 = measure_variance_x(V)

    # Run 200 pure diffusion steps (no source term)
    n_steps = 200
    R = torch.zeros(nx, ny, dtype=torch.float64)

    for _ in range(n_steps):
        f = bgk_collide(f, V, R, dt, omega, w)
        f_star = f.clone()
        f = stream_d2q5(f)
        f = apply_neumann_d2q5(f, f_star, bounce_masks)
        V = recover_voltage(f)

    var_final = measure_variance_x(V)
    t_total = n_steps * dt

    expected_growth = 2 * D_EFF * t_total
    actual_growth = var_final - var_0
    rel_err = abs(actual_growth - expected_growth) / expected_growth

    print(f"    LBM D2Q5 variance growth: {actual_growth:.6f} cm^2")
    print(f"    Expected:                  {expected_growth:.6f} cm^2")
    print(f"    Relative error:            {rel_err:.4f}")

    assert rel_err < 0.05, f"LBM Gaussian variance growth error: {rel_err:.4f}"
    print("6A-T4 PASS: LBM Gaussian diffuses at D_eff rate")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Phase 6A: Convention Fix Verification\n")

    test_6a_t1_operator_scaling()
    print()
    test_6a_t2_laplacian_quadratic()
    print()
    test_6a_t3_bidomain_gaussian()
    print()
    test_6a_t4_lbm_gaussian()

    print("\nPhase 6A: ALL 4 TESTS PASS")
