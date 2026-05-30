"""
StimBuilder export functionality.

Exports a StimBuilderSession to a .npz file containing per-region
masks and protocol parameters for the simulation engine loader.
"""

import numpy as np
from pathlib import Path

from .session import StimBuilderSession


def export_stim(session: StimBuilderSession, output_path: str) -> str:
    """
    Export a configured StimBuilderSession to a .npz file.

    The .npz contains:
        n_regions           : int       — number of stimulus regions
        mask_{i}            : (Nx, Ny) bool — spatial mask for region i (grid convention)
        label_{i}           : str       — region label
        stim_type_{i}       : str       — "current_injection" or "voltage_clamp"
        amplitude_{i}       : float     — uA/cm2 or mV
        duration_{i}        : float     — pulse duration (ms)
        start_time_{i}      : float     — first pulse onset (ms)
        bcl_{i}             : float     — basic cycle length (ms), 0 = single pulse
        num_pulses_{i}      : int       — pulse count, 0 = infinite

    Parameters
    ----------
    session : StimBuilderSession
        A fully configured session.
    output_path : str
        Path for the output .npz file.

    Returns
    -------
    str
        The resolved output path.
    """
    if session.image_array is None:
        raise ValueError("No image loaded in session.")

    active = [r for r in session.active_regions if r.is_configured]
    if not active:
        raise ValueError("No configured stimulus regions.")

    data = {}
    data['n_regions'] = np.int32(len(active))

    for i, region in enumerate(active):
        # Get spatial mask from image (Ny, Nx) and transpose to grid convention (Nx, Ny)
        mask = session.get_region_mask(region.color)
        data[f'mask_{i}'] = mask.astype(bool).T
        data[f'label_{i}'] = str(region.label or "")
        data[f'stim_type_{i}'] = region.stim_type.value if region.stim_type else ""
        data[f'amplitude_{i}'] = np.float64(region.amplitude or 0.0)

        if region.protocol is not None:
            data[f'duration_{i}'] = np.float64(region.protocol.duration)
            data[f'start_time_{i}'] = np.float64(region.protocol.start_time)
            data[f'bcl_{i}'] = np.float64(region.protocol.bcl)
            data[f'num_pulses_{i}'] = np.int32(
                region.protocol.num_pulses if region.protocol.num_pulses is not None else 0
            )
        else:
            data[f'duration_{i}'] = np.float64(1.0)
            data[f'start_time_{i}'] = np.float64(0.0)
            data[f'bcl_{i}'] = np.float64(0.0)
            data[f'num_pulses_{i}'] = np.int32(1)

    output_path = str(Path(output_path).resolve())
    np.savez(output_path, **data)

    return output_path
