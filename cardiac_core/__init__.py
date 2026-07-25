"""
cardiac_core — Unified cardiac simulation API.

Usage:
    from cardiac_core import monodomain, create_cardiac_mesh, run_monodomain
    from cardiac_core import activation_time, apd_map, conduction_velocity
    from cardiac_core.ionic import TTP06Model, ORdModel, CellType

Lazy package: public names are resolved on first access via PEP 562 ``__getattr__``.
This is deliberate — importing a light submodule like ``cardiac_core.ionic`` must NOT
drag in ``api`` (which imports the heavy vendored solver packages). ``cardiac_core.ionic``
is a self-contained subpackage; import it directly.
"""

import importlib

# public name -> submodule that defines it (kept in sync with the submodules' exports)
_LAZY = {
    # file_format
    'CardiacMeshData': 'file_format', 'save_cardiac_mesh': 'file_format',
    'load_cardiac_mesh': 'file_format', 'create_cardiac_mesh': 'file_format',
    # conductivity (the chi/Cm + Formulation-A/B firewall)
    'ConductivityConfig': 'conductivity',
    # grid (structured-only geometry)
    'Grid': 'grid',
    # simulation (engine-agnostic runtime Protocol)
    'Simulation': 'simulation',
    # api (engine wrappers — heavy: imports the vendored _monodomain/_bidomain/_lbm packages on first use)
    'monodomain': 'api', 'bidomain': 'api', 'lbm': 'api',
    'CardiacSimulation': 'api', 'SimulationSnapshot': 'api', 'Distribution': 'api',
    # run
    'run_monodomain': 'run', 'run_bidomain': 'run', 'run_lbm': 'run',
    'simulate': 'run', 'SimulationResult': 'run',
    # analysis
    'activation_time': 'analysis', 'max_dvdt_time': 'analysis',
    'conduction_velocity': 'analysis', 'apd_at': 'analysis',
    'apd_map': 'analysis', 'dominant_frequency': 'analysis', 'wavefront_mask': 'analysis',
    'phase_map': 'analysis', 'phase_singularities': 'analysis', 'restitution_curve': 'analysis',
    'dominant_frequency_map': 'analysis', 'cv_between': 'analysis', 'radial_cv': 'analysis',
    'apd_per_beat': 'analysis', 'restitution_slope': 'analysis',
    'wavelength': 'analysis', 'di': 'analysis',
    # protocols (run simulations)
    'erp': 'protocols', 'erp_proxy': 'protocols', 'post_repol_refractoriness': 'protocols',
    # single-cell (0-D) + safety factor.
    # The module is PRIVATE (`_single_cell`) on purpose: a submodule named `single_cell` would be
    # bound as a package attribute on import and shadow this lazy `single_cell` function export
    # (PEP 562 __getattr__ only fires when normal lookup fails), so `cc.single_cell` would silently
    # become the MODULE. Never name a submodule the same as a public export here.
    'single_cell': '_single_cell', 'safety_factor': '_single_cell', 'threshold_charge': '_single_cell',
    # stimulus object
    'Stim': 'stimulus.stim',
    # geometry
    'circle_mask': 'geometry', 'rectangle_mask': 'geometry', 'annulus_mask': 'geometry',
    'left_edge_mask': 'geometry', 'right_edge_mask': 'geometry',
    'top_edge_mask': 'geometry', 'bottom_edge_mask': 'geometry', 'point_distance': 'geometry',
    'boundary_distance': 'geometry', 'fiber_field_uniform': 'geometry',
    'fiber_field_transmural': 'geometry',
    # io
    'save_result': 'io', 'load_result': 'io',
    # viz (standardized result figures/videos)
    'propagation_video': 'viz', 'apd_map_figure': 'viz', 'activation_isochrones': 'viz',
    # video (spec-first rendering: Video + Gradient + render)
    'Video': 'video', 'Gradient': 'video', 'render': 'video',
    'render_video': 'video', 'VideoInfo': 'video', 'ImagePath': 'video',
    # image (spec-first still figures: Image + draw). The submodule holding `draw` is `_draw`,
    # not `draw`, for the same shadowing reason `_single_cell` is private.
    'Image': 'image', 'Trace': 'image', 'draw': 'image', 'ImageInfo': 'image',
}


def __getattr__(name):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    submodule = importlib.import_module(f".{mod}", __name__)
    return getattr(submodule, name)


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY))
