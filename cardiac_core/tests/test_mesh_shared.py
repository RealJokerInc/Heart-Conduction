"""Phase 1: shared cardiac_core.mesh + cardiac_core.stimulus packages import and work."""

import torch

from cardiac_core.mesh import StructuredGrid, BoundarySpec  # noqa: F401
from cardiac_core.stimulus import StimulusProtocol


class TestSharedMesh:
    def test_structured_builds(self):
        g = StructuredGrid.create_rectangle(1.0, 0.5, 21, 11, device='cpu')
        assert g.Nx == 21 and g.Ny == 11
        x, y = g.coordinates
        assert x.numel() == 21 * 11

    def test_boundary_spec_superset(self):
        # boundary_spec is the bidomain superset addition; must be assignable on the shared grid.
        g = StructuredGrid.create_rectangle(1.0, 0.5, 11, 11, device='cpu')
        g.boundary_spec = BoundarySpec.insulated()
        assert g.boundary_spec is not None

    def test_flat_to_grid_roundtrip(self):
        g = StructuredGrid.create_rectangle(1.0, 0.5, 21, 11, device='cpu')
        flat = torch.arange(21 * 11, dtype=torch.float64)
        grid = g.flat_to_grid(flat)
        assert tuple(grid.shape) == (21, 11)


class TestSharedStimulus:
    def test_protocol_imports(self):
        p = StimulusProtocol()
        assert hasattr(p, 'add_stimulus')
