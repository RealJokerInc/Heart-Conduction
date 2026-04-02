"""
cardiac_core — Unified cardiac simulation API.

Usage:
    from cardiac_core import monodomain, create_cardiac_mesh, run_monodomain
    from cardiac_core import activation_time, apd_map, conduction_velocity
"""

from .file_format import CardiacMeshData, save_cardiac_mesh, load_cardiac_mesh, create_cardiac_mesh
from .api import monodomain, bidomain, lbm, CardiacSimulation, SimulationSnapshot, Distribution
from .run import run_monodomain, run_bidomain, run_lbm, simulate, SimulationResult
from .analysis import (
    activation_time, conduction_velocity, apd_at, apd_map,
    dominant_frequency, wavefront_mask,
    phase_map, phase_singularities, restitution_curve,
)
from .geometry import (
    circle_mask, rectangle_mask, annulus_mask,
    left_edge_mask, right_edge_mask,
    point_distance, boundary_distance,
    fiber_field_uniform, fiber_field_transmural,
)
from .io import save_result, load_result
