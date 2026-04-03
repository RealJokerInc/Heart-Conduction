"""Checkpoint manager for training state persistence.

Saves/loads: model weights, optimizer, scheduler, encoder, RNG state, phase info.
Handles best-per-phase tracking, latest overwrites, and pause checkpoints.
"""

import random as pyrandom
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class CheckpointManager:
    def __init__(self, run_dir: str):
        self.ckpt_dir = Path(run_dir) / 'checkpoints'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(self, tag: str, model: nn.Module, optimizer, scheduler,
             encoder: Optional[nn.Module], phase: str, epoch: int, step: int,
             best_val_loss: float, config: Optional[dict] = None) -> Path:
        ckpt = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'encoder_state_dict': encoder.state_dict() if encoder else None,
            'phase': phase, 'epoch': epoch, 'step': step,
            'best_val_loss': best_val_loss,
            'config': config or {},
            'rng_state': {
                'torch': torch.random.get_rng_state(),
                'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                'numpy': np.random.get_state(),
                'python': pyrandom.getstate(),
            },
        }
        path = self.ckpt_dir / f'{tag}.pt'
        torch.save(ckpt, path)
        return path

    def load(self, tag: str, model: nn.Module, optimizer=None,
             scheduler=None, encoder=None, device='cpu') -> dict:
        path = self.ckpt_dir / f'{tag}.pt'
        ckpt = torch.load(path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if encoder and ckpt.get('encoder_state_dict'):
            encoder.load_state_dict(ckpt['encoder_state_dict'])
        # Restore RNG state
        rng = ckpt.get('rng_state', {})
        if 'torch' in rng:
            torch.random.set_rng_state(rng['torch'])
        if 'numpy' in rng:
            np.random.set_state(rng['numpy'])
        if 'python' in rng:
            pyrandom.setstate(rng['python'])
        return {k: v for k, v in ckpt.items()
                if k not in ('model_state_dict', 'optimizer_state_dict',
                             'scheduler_state_dict', 'encoder_state_dict', 'rng_state')}

    def get_best(self, phase: str) -> Optional[Path]:
        path = self.ckpt_dir / f'best_{phase}.pt'
        return path if path.exists() else None

    def list_checkpoints(self) -> list[str]:
        return sorted(p.stem for p in self.ckpt_dir.glob('*.pt'))
