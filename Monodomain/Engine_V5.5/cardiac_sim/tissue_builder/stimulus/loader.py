"""
Stimulus loader — reads stim.npz files exported by stim_builder.

Converts the .npz contents into a StimulusProtocol with precomputed
mask tensors aligned to the mesh's active-node ordering.
"""

import numpy as np
import torch

from .protocol import StimulusProtocol


def load_stimulus(
    path: str,
    mesh_mask: np.ndarray,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float64,
) -> StimulusProtocol:
    """
    Load a stim.npz file and return a StimulusProtocol.

    Each stimulus region mask is intersected with the mesh mask and
    flattened to active-node ordering (matching StructuredGrid.from_mask).

    Parameters
    ----------
    path : str
        Path to the stim.npz file.
    mesh_mask : np.ndarray
        Boolean mask from mesh.npz, shape (Nx, Ny) in grid convention.
        Used to intersect stim regions with active tissue and to determine
        flat ordering.
    device : str
        Torch device.
    dtype : torch.dtype
        Float precision.

    Returns
    -------
    StimulusProtocol
        Protocol with precomputed mask tensors for each region/pulse.
    """
    data = np.load(path, allow_pickle=True)
    n_regions = int(data['n_regions'])

    protocol = StimulusProtocol()

    for i in range(n_regions):
        # Load region data — masks already in grid convention (Nx, Ny)
        stim_mask = data[f'mask_{i}'].astype(bool)
        label = str(data[f'label_{i}'])
        stim_type = str(data[f'stim_type_{i}'])
        amplitude = float(data[f'amplitude_{i}'])
        duration = float(data[f'duration_{i}'])
        start_time = float(data[f'start_time_{i}'])
        bcl = float(data[f'bcl_{i}'])
        num_pulses = int(data[f'num_pulses_{i}'])

        # Intersect with mesh mask (only active tissue nodes)
        intersected = stim_mask & mesh_mask

        # Flatten to active-node ordering: extract where mesh_mask is True
        flat_mask = torch.tensor(
            intersected[mesh_mask],
            dtype=torch.bool,
            device=device,
        )

        if flat_mask.sum() == 0:
            continue  # Skip regions with no overlap with tissue

        # Determine number of pulses
        if num_pulses == 0:
            # 0 means infinite — use a large number
            n_beats = 1000
        else:
            n_beats = num_pulses

        # Add to protocol
        if bcl > 0 and n_beats > 1:
            protocol.add_regular_pacing(
                region=flat_mask,
                bcl=bcl,
                n_beats=n_beats,
                start_time=start_time,
                duration=duration,
                amplitude=amplitude,
            )
        else:
            # Single pulse
            protocol.add_stimulus(
                region=flat_mask,
                start_time=start_time,
                duration=duration,
                amplitude=amplitude,
            )

    return protocol
