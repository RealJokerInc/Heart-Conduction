"""Tests for cardiac_core.grid.Grid — the structured-only geometry descriptor."""

import math

import torch

from cardiac_core import Grid
from cardiac_core.geometry import circle_mask


class TestGridDims:
    def test_dims(self):
        g = Grid(150, 40, 0.025)
        assert g.Nx == 150 and g.Ny == 40
        assert math.isclose(g.Lx, 0.025 * 149)
        assert math.isclose(g.Ly, 0.025 * 39)
        assert g.dy == g.dx == 0.025
        assert g.n_dof == 150 * 40  # 6000

    def test_dy_independent(self):
        g = Grid(10, 20, dx=0.1, dy=0.05)
        assert g.dy == 0.05
        assert math.isclose(g.Ly, 0.05 * 19)


class TestGridCoordinates:
    def test_coordinates(self):
        g = Grid(150, 40, 0.025)
        x, y = g.coordinates
        assert tuple(x.shape) == (150, 40)
        assert tuple(y.shape) == (150, 40)
        # ij orientation: x varies along axis 0, y along axis 1.
        assert x[0, 0].item() == 0.0
        assert math.isclose(x[-1, 0].item(), g.Lx, rel_tol=1e-12)
        assert y[0, 0].item() == 0.0
        assert math.isclose(y[0, -1].item(), g.Ly, rel_tol=1e-12)
        assert x.dtype == torch.float64

    def test_coordinates_match_engine_orientation(self):
        # The engine StructuredGrid uses the same linspace/meshgrid('ij') convention;
        # confirm the wrapper's grid-shaped coords reduce to the engine's flat coords.
        g = Grid(8, 5, 0.1)
        sg = g._structured_grid()
        fx, fy = sg.coordinates  # flat (n_dof,)
        xx, yy = g.coordinates
        assert torch.allclose(xx.flatten(), fx)
        assert torch.allclose(yy.flatten(), fy)


class TestGridMask:
    def test_mask_ndof(self):
        mask = circle_mask(50, 50, 0.05, center=(1.25, 1.25), radius=0.5)
        g = Grid(50, 50, 0.05, mask=mask)
        assert g.n_dof == int(mask.sum())
        assert g.n_dof < 50 * 50

    def test_mask_shape_validation(self):
        import pytest
        with pytest.raises(ValueError):
            Grid(10, 10, 0.1, mask=torch.ones(5, 5, dtype=torch.bool))


class TestGridStructured:
    def test_structured_roundtrip(self):
        g = Grid(30, 12, 0.05)
        sg = g._structured_grid()
        assert sg.Nx == 30 and sg.Ny == 12
        # cached — second call returns the same object.
        assert g._structured_grid() is sg

    def test_structured_from_mask(self):
        mask = circle_mask(40, 40, 0.05, center=(1.0, 1.0), radius=0.4)
        g = Grid(40, 40, 0.05, mask=mask)
        sg = g._structured_grid()
        assert sg.Nx == 40 and sg.Ny == 40
        assert sg.domain_mask is not None
