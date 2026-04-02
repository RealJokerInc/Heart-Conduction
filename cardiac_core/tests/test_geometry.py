"""Tests for cardiac_core.geometry — mask and distance field helpers."""

import numpy as np
import pytest

from cardiac_core.geometry import (
    circle_mask, rectangle_mask, annulus_mask,
    left_edge_mask, right_edge_mask,
    point_distance, boundary_distance,
    fiber_field_uniform, fiber_field_transmural,
)


class TestCircleMask:
    def test_basic(self):
        mask = circle_mask(41, 41, dx=0.025, center=(0.5, 0.5), radius=0.2)
        assert mask.shape == (41, 41)
        assert mask.dtype == bool
        # Center should be inside
        assert mask[20, 20] == True
        # Far corner should be outside
        assert mask[0, 0] == False

    def test_radius_zero(self):
        mask = circle_mask(21, 21, dx=0.05, center=(0.5, 0.5), radius=0.0)
        # Only the exact center node (if it lands on grid)
        assert mask.sum() <= 1


class TestRectangleMask:
    def test_basic(self):
        mask = rectangle_mask(41, 21, dx=0.025, x0=0.1, y0=0.1, x1=0.4, y1=0.3)
        assert mask.shape == (41, 21)
        # Point inside
        assert mask[10, 6] == True  # x=0.25, y=0.15
        # Point outside
        assert mask[0, 0] == False


class TestAnnulusMask:
    def test_ring(self):
        mask = annulus_mask(41, 41, dx=0.025, center=(0.5, 0.5),
                           inner_radius=0.1, outer_radius=0.3)
        # Center should be outside (inside hole)
        assert mask[20, 20] == False
        # Ring region should be inside
        assert mask[20, 14] == True  # x=0.5, y=0.35 → r=0.15


class TestEdgeMasks:
    def test_left_edge(self):
        mask = left_edge_mask(41, 21, dx=0.025, width=0.1)
        assert mask[:4, :].all()    # x < 0.1
        assert not mask[10:, :].any()  # x >= 0.25

    def test_right_edge(self):
        mask = right_edge_mask(41, 21, dx=0.025, width=0.1)
        assert mask[37:, :].all()   # x > 0.9
        assert not mask[:30, :].any()


class TestPointDistance:
    def test_distance_at_origin(self):
        dist = point_distance(21, 21, dx=0.05, x0=0.0, y0=0.0)
        assert dist[0, 0] == pytest.approx(0.0)
        assert dist[20, 0] == pytest.approx(1.0)  # x=1.0, y=0

    def test_distance_symmetry(self):
        dist = point_distance(21, 21, dx=0.05, x0=0.5, y0=0.5)
        assert dist[10, 10] == pytest.approx(0.0)
        # Symmetric around center
        assert dist[0, 10] == pytest.approx(dist[20, 10], abs=1e-10)


class TestBoundaryDistance:
    def test_full_rectangle(self):
        mask = np.ones((21, 11), dtype=bool)
        dist = boundary_distance(mask, dx=0.025)
        # Edge nodes have distance = 1*dx (EDT counts from outside, edge is 1 pixel from border)
        assert dist[0, 0] == pytest.approx(0.025, rel=0.01)
        # Interior should be larger
        assert dist[10, 5] > dist[0, 0]

    def test_outside_is_nan(self):
        mask = np.ones((21, 11), dtype=bool)
        mask[0, :] = False
        dist = boundary_distance(mask, dx=0.025)
        assert np.isnan(dist[0, 0])


class TestFiberFields:
    def test_uniform(self):
        field = fiber_field_uniform(21, 11, angle=0.5)
        assert field.shape == (21, 11)
        np.testing.assert_allclose(field, 0.5)

    def test_transmural(self):
        field = fiber_field_transmural(21, 11, angle_endo=-1.0, angle_epi=1.0)
        assert field.shape == (21, 11)
        # Endo edge (y=0)
        np.testing.assert_allclose(field[:, 0], -1.0)
        # Epi edge (y=10)
        np.testing.assert_allclose(field[:, 10], 1.0)
        # Middle
        np.testing.assert_allclose(field[:, 5], 0.0, atol=0.01)
        # Uniform along x
        assert field[0, 3] == pytest.approx(field[20, 3])
