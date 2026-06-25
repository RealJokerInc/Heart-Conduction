"""
Result I/O — save and load simulation results to disk.

Saves voltage history, times, and metadata as .npz files.
Separate from the mesh file format — this is for post-simulation data.

    save_result("output.npz", times, V)
    times, V, phi_e, meta = load_result("output.npz")
"""

import warnings
from typing import Optional
import numpy as np
import torch


def save_result(
    path: str,
    times: torch.Tensor,
    Vm: Optional[torch.Tensor] = None,
    phi_e: Optional[torch.Tensor] = None,
    *,
    V: Optional[torch.Tensor] = None,
    **metadata,
) -> None:
    """Save simulation results to .npz.

    Parameters
    ----------
    path : str
        Output file path.
    times : torch.Tensor
        (n_saves,) time points in ms.
    Vm : torch.Tensor
        (n_saves, Nx, Ny) voltage history. Canonical name. The legacy positional/keyword
        ``V`` is still accepted (with a DeprecationWarning when passed by keyword); the
        voltage in positional slot 3 binds to ``Vm``.
    phi_e : torch.Tensor, optional
        (n_saves, Nx, Ny) extracellular potential (bidomain). Positional-or-keyword.
    V : torch.Tensor, optional
        Deprecated keyword alias for ``Vm``.
    **metadata
        Additional scalar/string metadata stored as numpy scalars.
        e.g. dx=0.025, engine='monodomain', ionic_model='ttp06'
    """
    voltage = Vm if Vm is not None else V
    if Vm is None and V is not None:
        warnings.warn(
            "save_result(V=...) is deprecated; pass Vm=... instead.",
            DeprecationWarning, stacklevel=2,
        )
    if voltage is None:
        raise TypeError("save_result requires the voltage history (Vm=...).")
    d = {
        'times': times.cpu().numpy(),
        'V': voltage.cpu().numpy(),  # npz key stays 'V' so load_result is unchanged
    }
    if phi_e is not None:
        d['phi_e'] = phi_e.cpu().numpy()

    for key, val in metadata.items():
        d[f'meta_{key}'] = np.array(val)

    np.savez_compressed(path, **d)


def load_result(
    path: str,
    device: str = 'cpu',
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], dict]:
    """Load simulation results from .npz.

    Parameters
    ----------
    path : str
        Input file path.
    device : str
        Device for output tensors.

    Returns
    -------
    times : torch.Tensor
        (n_saves,) time points.
    V : torch.Tensor
        (n_saves, Nx, Ny) voltage history.
    phi_e : torch.Tensor | None
        Extracellular potential if present.
    metadata : dict
        Any additional metadata that was saved.
    """
    f = np.load(path)
    dev = torch.device(device)

    times = torch.tensor(f['times'], dtype=torch.float64, device=dev)
    V = torch.tensor(f['V'], dtype=torch.float64, device=dev)

    phi_e = None
    if 'phi_e' in f:
        phi_e = torch.tensor(f['phi_e'], dtype=torch.float64, device=dev)

    metadata = {}
    for key in f.files:
        if key.startswith('meta_'):
            val = f[key]
            # Convert 0-d arrays back to scalars
            metadata[key[5:]] = val.item() if val.ndim == 0 else val
    return times, V, phi_e, metadata
