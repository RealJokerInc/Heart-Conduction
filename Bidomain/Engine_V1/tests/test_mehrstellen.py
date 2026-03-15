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
