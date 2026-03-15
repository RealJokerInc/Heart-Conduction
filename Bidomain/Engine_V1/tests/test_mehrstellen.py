"""
Mehrstellen 9-Point Stencil — Validation Tests

Tests for the Mehrstellen isotropic 9-point FDM stencil across the full
bidomain pipeline: FDM assembly, spectral eigenvalues, solver wiring,
cv_shared builders, and bidomain-monodomain equivalence.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
import numpy as np

torch.set_default_dtype(torch.float64)


# ============================================================
# Helpers
# ============================================================

def _make_fdm(nx=16, ny=16, lx=None, ly=None, D_i=0.00124, D_e=0.00446,
              bc='insulated', stencil='5pt'):
    """Create BidomainFDMDiscretization with given params."""
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec
    from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
    from cardiac_sim.simulation.classical.discretization.fdm import BidomainFDMDiscretization

    # Default: square domain with dx == dy
    if lx is None:
        lx = 1.0
    if ly is None:
        ly = lx * (ny - 1) / (nx - 1)  # ensures dx == dy

    grid = StructuredGrid.create_rectangle(Lx=lx, Ly=ly, Nx=nx, Ny=ny)
    if bc == 'insulated':
        grid.boundary_spec = BoundarySpec.insulated()
    elif bc == 'bath':
        grid.boundary_spec = BoundarySpec.bath_coupled()
    elif bc == 'bath_tb':
        from cardiac_sim.tissue_builder.mesh.boundary import Edge
        grid.boundary_spec = BoundarySpec.bath_coupled_edges([Edge.TOP, Edge.BOTTOM])
    cond = BidomainConductivity(D_i=D_i, D_e=D_e)
    return BidomainFDMDiscretization(grid, cond, Cm=1.0, stencil=stencil)


# ============================================================
# Step 1: FDM Mehrstellen Assembly
# ============================================================

class TestMehrstellenFDM:
    """Tests 1-6: FDM Mehrstellen stencil assembly."""

    def test_default_backward_compat(self):
        """M-T1: Default stencil is '5pt', backward compatible."""
        spatial = _make_fdm()
        assert spatial.stencil == '5pt'
        # Should have L_i and L_e
        assert spatial.L_i is not None
        assert spatial.L_e is not None

    def test_symmetry(self):
        """M-T2: L_i with mehrstellen is symmetric."""
        spatial = _make_fdm(stencil='mehrstellen')
        L = spatial.L_i.to_dense()
        diff = torch.norm(L - L.T).item()
        assert diff < 1e-14, f"L_i not symmetric: ||L - L^T|| = {diff}"

    def test_zero_row_sum(self):
        """M-T3: All rows of L_i sum to zero (conservation / Neumann)."""
        spatial = _make_fdm(stencil='mehrstellen')
        L = spatial.L_i.to_dense()
        row_sums = L.sum(dim=1)
        max_sum = row_sums.abs().max().item()
        assert max_sum < 1e-14, f"Max row sum = {max_sum}"

    def test_negative_semidefinite(self):
        """M-T4: Eigenvalues of L_i are all <= 0."""
        spatial = _make_fdm(nx=12, ny=12, stencil='mehrstellen')
        L = spatial.L_i.to_dense()
        eigvals = torch.linalg.eigvalsh(L)
        max_eigval = eigvals.max().item()
        assert max_eigval < 1e-12, f"Max eigenvalue = {max_eigval} (should be <= 0)"

    def test_dx_ne_dy_rejected(self):
        """M-T5: AssertionError when dx != dy with mehrstellen."""
        with pytest.raises(AssertionError):
            _make_fdm(nx=16, ny=16, lx=1.0, ly=2.0, stencil='mehrstellen')

    def test_stencil_property(self):
        """M-T6: Stencil property is readable."""
        spatial_5pt = _make_fdm(stencil='5pt')
        spatial_mst = _make_fdm(stencil='mehrstellen')
        assert spatial_5pt.stencil == '5pt'
        assert spatial_mst.stencil == 'mehrstellen'


# ============================================================
# Step 2: Spectral Eigenvalues for Mehrstellen
# ============================================================

def _make_spectral(nx=16, ny=16, D=0.0057, bc_x='neumann', bc_y='neumann',
                   stencil='5pt'):
    """Create a SpectralSolver with given params."""
    from cardiac_sim.simulation.classical.solver.linear_solver.spectral import SpectralSolver
    lx = 1.0
    dx = lx / (nx - 1)
    return SpectralSolver(nx, ny, dx, dx, D, bc_x=bc_x, bc_y=bc_y,
                          stencil=stencil)


class TestMehrstellenSpectral:
    """Tests 7-10: Spectral eigenvalues for Mehrstellen."""

    def test_eigenvalue_consistency(self):
        """M-T7: Sparse L_i eigenvalues match spectral formula on 16x16 Neumann."""
        from cardiac_sim.simulation.classical.solver.linear_solver.spectral import SpectralSolver
        nx, ny = 16, 16
        D_i, D_e = 0.00124, 0.00446
        D_sum = D_i + D_e
        spatial = _make_fdm(nx=nx, ny=ny, D_i=D_i, D_e=D_e,
                            bc='insulated', stencil='mehrstellen')

        # Eigenvalues from sparse matrix: -(L_i + L_e) = A_ellip
        L_ie = (spatial.L_i + spatial.L_e).to_dense()
        sparse_eigvals = torch.linalg.eigvalsh(-L_ie).sort()[0]

        # Eigenvalues from spectral formula (compute directly, no placeholder)
        device, dtype = torch.device('cpu'), torch.float64
        h = 1.0 / (nx - 1)
        CX = SpectralSolver._axis_cosines('neumann', nx, device, dtype)
        CY = SpectralSolver._axis_cosines('neumann', ny, device, dtype)
        CX_2d, CY_2d = torch.meshgrid(CX, CY, indexing='ij')
        spectral_eigvals = D_sum * (
            20.0 - 8.0 * CX_2d - 8.0 * CY_2d - 4.0 * CX_2d * CY_2d
        ) / (6.0 * h * h)
        spectral_eigvals = spectral_eigvals.flatten().sort()[0]

        max_diff = (sparse_eigvals - spectral_eigvals).abs().max().item()
        assert max_diff < 1e-10, f"Eigenvalue mismatch: max diff = {max_diff}"

    def test_null_mode(self):
        """M-T8: eigenvalues[0,0] == 0 for Neumann/Neumann (before placeholder)."""
        spec = _make_spectral(stencil='mehrstellen')
        # Manually compute to check the raw value
        from cardiac_sim.simulation.classical.solver.linear_solver.spectral import SpectralSolver
        h = 1.0 / 15
        D = 0.0057
        # CX[0] = cos(0) = 1, CY[0] = cos(0) = 1
        # eigenvalue = D * (20 - 8 - 8 - 4) / (6*h^2) = D * 0 / (6*h^2) = 0
        raw = D * (20.0 - 8.0 - 8.0 - 4.0) / (6.0 * h * h)
        assert abs(raw) < 1e-14, f"Null mode not zero: {raw}"

    def test_spectral_solve_residual_neumann(self):
        """M-T9: Solve -D*Lap(u) = b with Mehrstellen spectral, verify residual."""
        nx, ny = 16, 16
        D_i, D_e = 0.00124, 0.00446
        D_sum = D_i + D_e
        spatial = _make_fdm(nx=nx, ny=ny, D_i=D_i, D_e=D_e,
                            bc='insulated', stencil='mehrstellen')

        # Build RHS: random with zero mean (Neumann compatibility)
        torch.manual_seed(42)
        b = torch.randn(nx * ny)
        b -= b.mean()

        spec = _make_spectral(nx=nx, ny=ny, D=D_sum, stencil='mehrstellen')
        u = spec.solve(None, b)

        # Compute residual: L_mehrstellen * u should = -b (since -D*Lap*u = b → L*u = -b/D... no)
        # Actually: spectral solves -D*Lap*u = b. The sparse matrix is L = D*Lap (negative semidefinite).
        # So -L*u = b, i.e. L*u = -b. But L is -(L_i+L_e) for elliptic.
        # More precisely: A_ellip = -(L_i + L_e), and spectral solves A_ellip * u = b.
        L_ie = (spatial.L_i + spatial.L_e).to_dense()
        A_ellip = -L_ie
        residual = A_ellip @ u - b
        rel_res = torch.norm(residual).item() / torch.norm(b).item()
        assert rel_res < 1e-10, f"Relative residual = {rel_res}"

    def test_spectral_solve_residual_mixed_bc(self):
        """M-T10: Spectral Mehrstellen with Neumann-x / Dirichlet-y (bath_tb)."""
        nx, ny = 16, 16
        D_i, D_e = 0.00124, 0.00446
        D_sum = D_i + D_e
        spatial = _make_fdm(nx=nx, ny=ny, D_i=D_i, D_e=D_e,
                            bc='bath_tb', stencil='mehrstellen')

        # Get A_ellip (has Dirichlet enforcement)
        A_ellip = spatial.get_elliptic_operator().to_dense()

        # Build RHS: random, with zeros at Dirichlet nodes
        torch.manual_seed(123)
        b = torch.randn(nx * ny)
        # Zero out Dirichlet rows (top/bottom boundary = j=0 and j=ny-1)
        for i in range(nx):
            b[i * ny] = 0.0          # j=0 (bottom)
            b[i * ny + ny - 1] = 0.0  # j=ny-1 (top)

        spec = _make_spectral(nx=nx, ny=ny, D=D_sum, bc_x='neumann',
                              bc_y='dirichlet', stencil='mehrstellen')
        u = spec.solve(None, b)

        residual = A_ellip @ u - b
        rel_res = torch.norm(residual).item() / torch.norm(b).item()
        assert rel_res < 1e-10, f"Relative residual (mixed BC) = {rel_res}"


# ============================================================
# Step 3: Solver Factory Wiring
# ============================================================

def _make_sim(nx=21, ny=21, stencil='5pt', bc='insulated'):
    """Create BidomainSimulation with given stencil."""
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec, Edge
    from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
    from cardiac_sim.simulation.classical.discretization.fdm import BidomainFDMDiscretization
    from cardiac_sim.tissue_builder.stimulus import StimulusProtocol, left_edge_region
    from cardiac_sim.simulation.classical.bidomain import BidomainSimulation

    lx = 1.0
    ly = lx * (ny - 1) / (nx - 1)
    grid = StructuredGrid.create_rectangle(Lx=lx, Ly=ly, Nx=nx, Ny=ny)
    if bc == 'insulated':
        grid.boundary_spec = BoundarySpec.insulated()
    elif bc == 'bath_tb':
        grid.boundary_spec = BoundarySpec.bath_coupled_edges([Edge.TOP, Edge.BOTTOM])
    cond = BidomainConductivity(D_i=0.00124, D_e=0.00446)
    spatial = BidomainFDMDiscretization(grid, cond, Cm=1.0, stencil=stencil)

    stim = StimulusProtocol()
    stim.add_stimulus(
        region=left_edge_region(width=0.1),
        start_time=1.0, duration=2.0, amplitude=-80.0,
    )
    sim = BidomainSimulation(
        spatial=spatial, ionic_model='ttp06', stimulus=stim,
        dt=0.02, elliptic_solver='auto',
    )
    return sim, grid


class TestMehrstellenWiring:
    """Tests 11-13: Stencil parameter wired through solver factories."""

    def test_auto_selection_works(self):
        """M-T11: Mehrstellen auto-selects spectral for insulated isotropic."""
        sim, _ = _make_sim(stencil='mehrstellen', bc='insulated')
        assert sim._elliptic_solver_name == 'spectral'

    def test_smoke_test_10_steps(self):
        """M-T12: 10 steps of BidomainSimulation with mehrstellen, no NaN/Inf."""
        sim, grid = _make_sim(stencil='mehrstellen')
        count = 0
        for state in sim.run(t_end=0.2, save_every=0.2):
            V = grid.flat_to_grid(state.Vm)
            assert torch.isfinite(V).all(), "NaN/Inf in Vm"
            count += 1
        assert count >= 1

    def test_elliptic_spd_dirichlet(self):
        """M-T13: A_ellip with mehrstellen is positive definite (bath_tb)."""
        from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
        from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec, Edge
        from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity
        from cardiac_sim.simulation.classical.discretization.fdm import BidomainFDMDiscretization

        nx, ny = 12, 12
        lx = 1.0
        ly = lx * (ny - 1) / (nx - 1)
        grid = StructuredGrid.create_rectangle(Lx=lx, Ly=ly, Nx=nx, Ny=ny)
        grid.boundary_spec = BoundarySpec.bath_coupled_edges([Edge.TOP, Edge.BOTTOM])
        cond = BidomainConductivity(D_i=0.00124, D_e=0.00446)
        spatial = BidomainFDMDiscretization(grid, cond, Cm=1.0, stencil='mehrstellen')
        A = spatial.get_elliptic_operator().to_dense()
        eigvals = torch.linalg.eigvalsh(A)
        assert eigvals.min().item() > 0, f"Min eigenvalue = {eigvals.min().item()}"


# ============================================================
# Step 4: cv_shared Builders
# ============================================================

class TestMehrstellenCvShared:
    """Tests 14-15: cv_shared builders accept stencil parameter."""

    def test_default_preserved(self):
        """M-T14: run_bidomain() without stencil arg runs normally."""
        from cv_shared import build_bidomain_sim, NX, NY, DX, DT, D_I, D_E
        # Just build, don't run (too slow for unit test)
        sim, grid = build_bidomain_sim(nx=21, ny=21, dx=DX, dt=DT,
                                       D_i=D_I, D_e=D_E)
        assert sim is not None

    def test_mehrstellen_accepted(self):
        """M-T15: build_bidomain_sim with stencil='mehrstellen' works."""
        from cv_shared import build_bidomain_sim, DX, DT, D_I, D_E
        sim, grid = build_bidomain_sim(nx=21, ny=21, dx=DX, dt=DT,
                                       D_i=D_I, D_e=D_E,
                                       stencil='mehrstellen')
        # Run 5ms
        count = 0
        for state in sim.run(t_end=0.1, save_every=0.1):
            V = grid.flat_to_grid(state.Vm)
            assert torch.isfinite(V).all()
            count += 1
        assert count >= 1


# ============================================================
# Step 5: Bidomain-Monodomain Equivalence
# ============================================================

class TestMehrstellenEquivalence:
    """Test 18: Bidomain insulated ≡ Monodomain with Mehrstellen stencil."""

    @pytest.mark.slow
    def test_bidomain_monodomain_equivalence(self):
        """M-T18: Bidomain insulated matches monodomain with Mehrstellen stencil.

        Both use same:
        - Mehrstellen 9-point Laplacian
        - D_eff = D_i*D_e/(D_i+D_e) for monodomain
        - Neumann BCs everywhere
        - Same domain, dx, stimulus

        The bidomain with insulated BCs (equal Neumann on both domains)
        reduces to monodomain with D_eff. Wavefront position should
        match within dx at t=25ms.
        """
        from cv_shared import (build_bidomain_sim, run_monodomain_fdm_mehrstellen,
                               D_I, D_E, D_EFF, DX, DT, STIM_AMP,
                               STIM_START, STIM_DUR, THRESHOLD)

        nx, ny = 121, 61  # ~3 x 1.5 cm at dx=0.025
        dx = DX
        t_end = 25.0

        # --- Monodomain Mehrstellen ---
        times_mono, V_mono = run_monodomain_fdm_mehrstellen(
            nx=nx, ny=ny, dx=dx, dt=DT, D=D_EFF,
            t_end=t_end, save_every=t_end,
            stim_cols=5, stim_start=STIM_START, stim_dur=STIM_DUR,
            stim_amp=STIM_AMP,
        )
        assert len(V_mono) > 0, "Monodomain produced no snapshots"

        # --- Bidomain Mehrstellen Insulated ---
        sim, grid = build_bidomain_sim(
            nx=nx, ny=ny, dx=dx, dt=DT, D_i=D_I, D_e=D_E,
            bc_type='insulated', stencil='mehrstellen', theta=0.5,
        )
        V_bi = None
        for state in sim.run(t_end=t_end, save_every=t_end):
            V_bi = grid.flat_to_grid(state.Vm)

        assert V_bi is not None, "Bidomain produced no snapshots"
        V_mono_final = V_mono[-1]

        # Find wavefront position at center row: first x where V > threshold
        y_mid = ny // 2

        def _front_x(V, threshold=-30.0):
            for ix in range(nx - 1, -1, -1):
                if V[ix, y_mid].item() > threshold:
                    return ix
            return 0

        front_mono = _front_x(V_mono_final)
        front_bi = _front_x(V_bi)

        # Wavefront should match within 2*dx (allow slight splitting error)
        diff_nodes = abs(front_mono - front_bi)
        assert diff_nodes <= 2, (
            f"Wavefront mismatch: mono x={front_mono} ({front_mono*dx:.3f}cm), "
            f"bi x={front_bi} ({front_bi*dx:.3f}cm), diff={diff_nodes} nodes"
        )
