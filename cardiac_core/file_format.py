"""
Cardiac Mesh File Format — save/load/create cardiac tissue meshes.

Format version 1: .npz files containing grid, conductivity, stimulus, and
optional bidomain fields. Designed to be the single input to monodomain(),
bidomain(), and lbm() API functions.
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class CardiacMeshData:
    """Everything needed to construct any cardiac simulation engine.

    Attributes
    ----------
    dx, dy : float
        Grid spacing (cm).
    mask : np.ndarray
        (Nx, Ny) bool — True = active tissue.
    D_xx, D_yy, D_xy : np.ndarray
        (Nx, Ny) float64 — RAW conductivity-like tensor components. The
        membrane-effective diffusivity is ``D/(χ·Cm)`` (cm²/ms), computed
        identically by every engine.
    chi : float
        Surface-to-volume ratio (cm⁻¹).
    Cm : float
        Membrane capacitance (µF/cm²).
    ionic_model : str
        Name of ionic model ('ttp06', 'ord', 'phas13', 'mhas13', 'paci').
    dt : float
        Time step (ms).
    group_labels : list[str]
        Label per tissue group (e.g. ['myocardium']).
    group_cell_types : list[str]
        Cell type per group (e.g. ['ENDO']).
    stimuli : list[dict]
        Each dict: {mask, label, amplitude, duration, start_time, bcl, num_pulses}.
    sigma_i : tuple[np.ndarray, ...] | None
        (xx, yy, xy) intracellular conductivity — bidomain only.
    sigma_e : tuple[np.ndarray, ...] | None
        (xx, yy, xy) extracellular conductivity — bidomain only.
    boundary : str
        'insulated' or 'bath'.
    """
    # Grid
    dx: float
    dy: float
    mask: np.ndarray

    # Conductivity (monodomain effective)
    D_xx: np.ndarray
    D_yy: np.ndarray
    D_xy: np.ndarray

    # Physics
    chi: float = 1400.0
    Cm: float = 1.0
    ionic_model: str = 'ttp06'
    dt: float = 0.02

    # Tissue metadata
    group_labels: list = field(default_factory=lambda: ['myocardium'])
    group_cell_types: list = field(default_factory=lambda: ['ENDO'])

    # Stimulus regions
    stimuli: list = field(default_factory=list)

    # Bidomain (optional)
    sigma_i: Optional[tuple] = None
    sigma_e: Optional[tuple] = None

    # Boundary conditions
    boundary: str = 'insulated'


def save_cardiac_mesh(path: str, data: CardiacMeshData) -> None:
    """Save CardiacMeshData to .npz with format_version=1."""
    d = {
        'format_version': np.int64(1),
        'dx': np.float64(data.dx),
        'dy': np.float64(data.dy),
        'chi': np.float64(data.chi),
        'Cm': np.float64(data.Cm),
        'ionic_model': np.array(data.ionic_model),
        'dt': np.float64(data.dt),
        'mask': data.mask.astype(bool),
        'D_xx': data.D_xx.astype(np.float64),
        'D_yy': data.D_yy.astype(np.float64),
        'D_xy': data.D_xy.astype(np.float64),
        'group_labels': np.array(data.group_labels, dtype=str),
        'group_cell_types': np.array(data.group_cell_types, dtype=str),
        'boundary': np.array(data.boundary),
        'n_stim_regions': np.int64(len(data.stimuli)),
    }

    # Stimulus regions
    for i, stim in enumerate(data.stimuli):
        d[f'stim_mask_{i}'] = stim['mask'].astype(bool)
        d[f'stim_label_{i}'] = np.array(stim.get('label', f'stim_{i}'))
        d[f'stim_amplitude_{i}'] = np.float64(stim['amplitude'])
        d[f'stim_duration_{i}'] = np.float64(stim['duration'])
        d[f'stim_start_time_{i}'] = np.float64(stim['start_time'])
        d[f'stim_bcl_{i}'] = np.float64(stim.get('bcl', 0.0))
        d[f'stim_num_pulses_{i}'] = np.int64(stim.get('num_pulses', 1))

    # Optional bidomain fields
    if data.sigma_i is not None:
        d['sigma_i_xx'] = data.sigma_i[0].astype(np.float64)
        d['sigma_i_yy'] = data.sigma_i[1].astype(np.float64)
        d['sigma_i_xy'] = data.sigma_i[2].astype(np.float64)
    if data.sigma_e is not None:
        d['sigma_e_xx'] = data.sigma_e[0].astype(np.float64)
        d['sigma_e_yy'] = data.sigma_e[1].astype(np.float64)
        d['sigma_e_xy'] = data.sigma_e[2].astype(np.float64)

    np.savez(path, **d)


def load_cardiac_mesh(path: str) -> CardiacMeshData:
    """Load .npz, return CardiacMeshData."""
    f = np.load(path, allow_pickle=False)

    version = int(f['format_version'])
    if version != 1:
        raise ValueError(f"Unsupported format version: {version}")

    # Stimulus regions
    n_stim = int(f['n_stim_regions'])
    stimuli = []
    for i in range(n_stim):
        stimuli.append({
            'mask': f[f'stim_mask_{i}'],
            'label': str(f[f'stim_label_{i}']),
            'amplitude': float(f[f'stim_amplitude_{i}']),
            'duration': float(f[f'stim_duration_{i}']),
            'start_time': float(f[f'stim_start_time_{i}']),
            'bcl': float(f[f'stim_bcl_{i}']),
            'num_pulses': int(f[f'stim_num_pulses_{i}']),
        })

    # Optional bidomain
    sigma_i = None
    sigma_e = None
    if 'sigma_i_xx' in f:
        sigma_i = (f['sigma_i_xx'], f['sigma_i_yy'], f['sigma_i_xy'])
    if 'sigma_e_xx' in f:
        sigma_e = (f['sigma_e_xx'], f['sigma_e_yy'], f['sigma_e_xy'])

    return CardiacMeshData(
        dx=float(f['dx']),
        dy=float(f['dy']),
        mask=f['mask'],
        D_xx=f['D_xx'],
        D_yy=f['D_yy'],
        D_xy=f['D_xy'],
        chi=float(f['chi']),
        Cm=float(f['Cm']),
        ionic_model=str(f['ionic_model']),
        dt=float(f['dt']),
        group_labels=list(f['group_labels']),
        group_cell_types=list(f['group_cell_types']),
        stimuli=stimuli,
        sigma_i=sigma_i,
        sigma_e=sigma_e,
        boundary=str(f['boundary']),
    )


def create_cardiac_mesh(
    Lx: float,
    Ly: float,
    dx: float,
    D: float = 1.4,
    D_yy: float = None,
    ionic_model: str = 'ttp06',
    dt: float = 0.02,
    chi: float = 1400.0,
    Cm: float = 1.0,
    stim_width: float = 0.1,
    stim_amplitude: float = -80.0,
    stim_duration: float = 2.0,
    stim_start: float = 1.0,
    mask: np.ndarray = None,
) -> CardiacMeshData:
    """Create a CardiacMeshData programmatically (no Builder needed).

    Default: rectangular tissue, isotropic D, left-edge stimulus.

    CONDUCTIVITY CONVENTION — `D` is RAW; every engine divides by χ·Cm
    -----------------------------------------------------------------
    ``D`` is a RAW conductivity-like value. The **membrane-effective diffusivity
    is ``D/(χ·Cm)``**, computed identically by monodomain, bidomain, and LBM.
    The default ``D=1.4, chi=1400, Cm=1`` → effective ``1e-3 cm²/ms`` (physiological).
    To pass an EFFECTIVE diffusivity directly (e.g. ``D=1e-3``), set ``chi=1.0``
    (as the Optimizer chip code does, ``cc_runner``/``chip.chip_mesh``): then
    ``D/(χ·Cm) = D``. A ``D/(χ·Cm)`` outside [1e-4, 1e-1] cm²/ms emits a warning.

    Parameters
    ----------
    Lx, Ly : float
        Domain size (cm).
    dx : float
        Grid spacing (cm). dy = dx.
    D : float
        RAW x-axis conductivity-like term; the membrane sees ``D/(chi*Cm)`` in
        every engine. Default 1.4 → effective 1e-3 at chi=1400. Pass an effective
        diffusivity with chi=1.0.
    D_yy : float, optional
        y-axis term (per-axis anisotropy). None → isotropic (D_yy = D).
    ionic_model : str
        Ionic model name.
    dt : float
        Time step (ms).
    chi : float
        Surface-to-volume ratio (cm⁻¹). Divides D in every engine
        (effective = D/(χ·Cm)). Default 1400. Set chi=1.0 to treat `D` as effective.
    Cm : float
        Membrane capacitance (µF/cm²).
    stim_width : float
        Width of left-edge stimulus region (cm).
    stim_amplitude : float
        Stimulus amplitude (µA/µF).
    stim_duration : float
        Stimulus duration (ms).
    stim_start : float
        Stimulus start time (ms).
    mask : np.ndarray, optional
        Custom domain mask (Nx, Ny) bool. If None, full rectangle.
    """
    Nx = round(Lx / dx) + 1
    Ny = round(Ly / dx) + 1
    dy = dx

    if mask is None:
        mask = np.ones((Nx, Ny), dtype=bool)

    # Per-axis anisotropy: D = x-axis (longitudinal); D_yy = y-axis (transverse).
    # Default D_yy=None -> isotropic (D_yy = D), preserving prior behaviour.
    D_yy_scalar = D if D_yy is None else D_yy
    D_xx = np.full((Nx, Ny), D, dtype=np.float64)
    D_yy = np.full((Nx, Ny), D_yy_scalar, dtype=np.float64)
    D_xy = np.zeros((Nx, Ny), dtype=np.float64)

    # Left-edge stimulus mask
    x_coords = np.arange(Nx) * dx
    stim_mask = np.zeros((Nx, Ny), dtype=bool)
    stim_mask[x_coords < stim_width, :] = True
    stim_mask &= mask  # intersect with tissue

    stimuli = [{
        'mask': stim_mask,
        'label': 'S1_left',
        'amplitude': stim_amplitude,
        'duration': stim_duration,
        'start_time': stim_start,
        'bcl': 0.0,
        'num_pulses': 1,
    }]

    D_eff = D / (chi * Cm)
    if not (1e-4 <= D_eff <= 1e-1):
        warnings.warn(
            f"create_cardiac_mesh: effective diffusivity D/(χ·Cm) = {D_eff:.2e} cm²/ms "
            f"is outside the physiological band [1e-4, 1e-1]. `D` is RAW (effective = "
            f"D/(χ·Cm)); pass chi=1.0 to treat D as an effective diffusivity.",
            stacklevel=2,
        )

    return CardiacMeshData(
        dx=dx,
        dy=dy,
        mask=mask,
        D_xx=D_xx,
        D_yy=D_yy,
        D_xy=D_xy,
        chi=chi,
        Cm=Cm,
        ionic_model=ionic_model,
        dt=dt,
        stimuli=stimuli,
    )
