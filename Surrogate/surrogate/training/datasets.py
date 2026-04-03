"""Per-phase Dataset classes for IonicSurrogateV3 training.

Three dataset types matching the training phases:
- SnapshotDataset: Phase A1/A3 — random single-timestep samples
- PairDataset: Phase A2 — consecutive (t, t+1) pairs
- SegmentDataset: Phase B-E — contiguous segments of rollout_length
"""

from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset, ConcatDataset


class SnapshotDataset(Dataset):
    """Random single-timestep samples for Phase A1 (autoencoder) and A3 (conductance).

    Returns dict with ionic_states (14,), concentrations (4,),
    conductance_products (5,), Vm (scalar), dt (scalar).
    All tensors returned as float64.
    """

    def __init__(self, cached_data: dict[str, Tensor]):
        self.ionic_states = cached_data['ionic_states'].double()
        self.concentrations = cached_data['concentrations'].double()
        self.conductance_products = cached_data['conductance_products'].double()
        self.Vm = cached_data['Vm'].double()
        self.dt = cached_data['dt'].double()

    def __len__(self) -> int:
        return self.ionic_states.shape[0]

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            'ionic_states': self.ionic_states[idx],
            'concentrations': self.concentrations[idx],
            'conductance_products': self.conductance_products[idx],
            'Vm': self.Vm[idx],
            'dt': self.dt[idx],
        }


class PairDataset(Dataset):
    """Consecutive (t, t+1) pairs for Phase A2 (concentration tracking).

    Returns dict with state at t and concentration target at t+1.
    All tensors returned as float64.
    """

    def __init__(self, cached_data: dict[str, Tensor]):
        self.ionic_states = cached_data['ionic_states'].double()
        self.concentrations = cached_data['concentrations'].double()
        self.Vm = cached_data['Vm'].double()
        self.dt = cached_data['dt'].double()

    def __len__(self) -> int:
        return self.ionic_states.shape[0] - 1

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            'ionic_states_t': self.ionic_states[idx],
            'concentrations_t': self.concentrations[idx],
            'Vm_t': self.Vm[idx],
            'dt_t': self.dt[idx],
            'concentrations_t1': self.concentrations[idx + 1],
        }


class SegmentDataset(Dataset):
    """Contiguous segments of rollout_length for Phases B-E.

    Extracts overlapping segments with configurable stride.
    Returns dict of (segment_length, ...) tensors, all float64.
    """

    KEYS = [
        'Vm', 'dt', 'I_stim', 'I_ion', 'clamp_mask',
        'concentrations', 'gates', 'ionic_states',
        'conductance_products', 'E', 'gate_inf', 'gate_tau',
    ]

    def __init__(
        self,
        cached_data: dict[str, Tensor],
        segment_length: int,
        stride: Optional[int] = None,
    ):
        self.segment_length = segment_length
        self.stride = stride if stride is not None else max(1, segment_length // 2)

        # Store data as float64
        self.data = {}
        for key in self.KEYS:
            if key in cached_data:
                self.data[key] = cached_data[key].double()

        # Precompute valid start indices
        T = self.data['Vm'].shape[0]
        self.starts = list(range(0, T - segment_length + 1, self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        start = self.starts[idx]
        end = start + self.segment_length
        return {key: tensor[start:end] for key, tensor in self.data.items()}


def merge_tier_datasets(datasets: list[Dataset]) -> ConcatDataset:
    """Merge datasets from multiple tiers into one."""
    return ConcatDataset(datasets)
