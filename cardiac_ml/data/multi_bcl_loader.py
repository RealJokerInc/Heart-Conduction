"""Multi-BCL beat loader — replicates Surrogate/run_multi_bcl.py:27-81.

Reads a pre-cached tier `.pt` dict (Vm, dt, ionic_states, concentrations,
conductance_products at 0.01-ms resolution), extracts `n_beats` per BCL with
SUBSAMPLE=10 (dt = 0.1 ms), optionally filters to the last-N beats per BCL
(`min_beat`), and returns a DataLoader yielding per-beat dict segments
matching node_rollout's `segment` contract.

Oracle parity (Session 25 multi_bcl_002):
- train BCLs: [300, 500, 700, 1000, 1500], `min_beat=15` (keeps beats 15-19).
- val BCLs:   [400, 600, 800, 2000],        `min_beat=17` (keeps beats 17-19).
- SUBSAMPLE=10, n_beats=20 per BCL before filtering.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

SUBSAMPLE = 10  # dt = 0.1 ms after subsample (matches run_multi_bcl.py:24)


def _extract_beats(
    data: dict, bcls: Sequence[int], n_beats: int = 20, min_beat: int = 0,
) -> list[dict]:
    """Mirror run_multi_bcl.py:27-47 extract_beats, with optional min_beat filter."""
    beats: list[dict] = []
    offset = 0
    for bcl in bcls:
        steps_per_beat = int(bcl / 0.01)  # raw 0.01-ms steps
        for beat_idx in range(n_beats):
            start = offset + beat_idx * steps_per_beat
            indices = list(range(start, start + steps_per_beat, SUBSAMPLE))
            seg: dict[str, Any] = {}
            for k, v in data.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 1:
                    seg[k] = v[indices]
            if "dt" in seg:
                seg["dt"] = seg["dt"] * SUBSAMPLE
            seg["_bcl"] = bcl
            seg["_beat"] = beat_idx
            seg["_tier"] = "T1"
            if beat_idx >= min_beat:
                beats.append(seg)
        offset += steps_per_beat * n_beats
    return beats


class MultiBCLBeatDataset(Dataset):
    """Dataset wrapper around _extract_beats output.

    `cache_path` points at a torch.load-compatible `.pt` produced by
    surrogate.training.data_cache.CacheBuilder.build_all(tiers=[N]).
    """

    def __init__(
        self,
        cache_path: str | Path,
        bcls: Sequence[int],
        n_beats: int = 20,
        min_beat: int = 0,
    ):
        cache_path = Path(cache_path)
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"MultiBCL cache not found: {cache_path}. Build via "
                "`surrogate.training.data_cache.CacheBuilder(raw_dir, cache_dir)"
                ".build_all(tiers=[1])`."
            )
        data = torch.load(cache_path, weights_only=False)
        self.beats = _extract_beats(data, bcls, n_beats=n_beats, min_beat=min_beat)

    def __len__(self) -> int:
        return len(self.beats)

    def __getitem__(self, i: int) -> dict:
        return self.beats[i]


def _single_beat_collate(batch: list[dict]) -> dict:
    """batch_size=1 collate — unsqueeze leading batch dim + cast tensors to float64.

    Mirrors run_multi_bcl.py:78-81 `beat_to_batch`. Non-tensor metadata
    (`_bcl`, `_beat`, `_tier`) passes through so the trainer adapter can
    build oracle-matching t_eval from `_bcl`.
    """
    assert len(batch) == 1, "multi_bcl loader uses batch_size=1"
    seg = batch[0]
    out: dict[str, Any] = {}
    for k, v in seg.items():
        if torch.is_tensor(v):
            out[k] = v.unsqueeze(0).to(dtype=torch.float64)
        else:
            out[k] = v
    return out


def make_loader(
    cache_path: str | Path,
    bcls: Sequence[int],
    n_beats: int = 20,
    min_beat: int = 0,
    batch_size: int = 1,
    shuffle: bool = True,
) -> DataLoader:
    """Hydra `_target_` factory. `batch_size` MUST be 1 (oracle contract)."""
    ds = MultiBCLBeatDataset(cache_path, bcls, n_beats=n_beats, min_beat=min_beat)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, collate_fn=_single_beat_collate,
    )
