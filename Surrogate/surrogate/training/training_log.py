"""HDF5 training log: comprehensive per-batch and per-epoch metrics.

Hierarchical structure:
    training_log.h5
    ├── {run_name}/                          (one group per training run)
    │   ├── attrs: {model_params, start_phase, device, created, ...}
    │   ├── {phase_name}/                    (one group per phase within run)
    │   │   ├── attrs: {rollout, batch_size, lr, weight_decay, data_tiers,
    │   │   │           loss_type, trainable_params, max_epochs, patience,
    │   │   │           transition_metric, transition_threshold,
    │   │   │           best_val_loss, best_epoch, total_epochs, wall_seconds}
    │   │   ├── epochs                       (dataset, resizable)
    │   │   │   columns: [epoch, train_loss, val_loss, lr, wall_s]
    │   │   ├── batches                      (dataset, resizable)
    │   │   │   columns: [step, epoch, batch, loss, grad_norm, lr, wall_s]
    │   │   └── events                       (dataset, resizable, variable-length string)
    │   │       e.g., "epoch=5 divergence=spike loss=4803.99"

Usage:
    log = TrainingLog('runs/training_log.h5')

    # Start a run
    log.start_run('run_001', model_params=1534, device='cuda')

    # Start a phase
    log.start_phase('run_001', 'B1', rollout=1, batch_size=4096, lr=5e-4, ...)

    # Log per-batch (inside training loop)
    log.log_batch('run_001', 'B1', step=100, epoch=2, batch=50,
                  loss=0.05, grad_norm=0.3, lr=4.5e-4, wall_s=0.02)

    # Log per-epoch (after validation)
    log.log_epoch('run_001', 'B1', epoch=2, train_loss=0.05,
                  val_loss=0.04, lr=4.5e-4, wall_s=120.0)

    # Log events (spikes, transitions, interventions)
    log.log_event('run_001', 'B1', 'epoch=4 divergence=spike loss=4803.99')

    # End phase
    log.end_phase('run_001', 'B1', best_val_loss=0.56, best_epoch=28,
                  total_epochs=30, wall_seconds=3449)

    # Read back for plotting
    epochs = log.get_epochs('run_001', 'B1')  # numpy array (N, 5)
    batches = log.get_batches('run_001', 'B1')  # numpy array (M, 7)
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


# Column definitions for fixed-width datasets
EPOCH_COLUMNS = ['epoch', 'train_loss', 'val_loss', 'lr', 'wall_s']
EPOCH_DTYPE = np.float64

BATCH_COLUMNS = ['step', 'epoch', 'batch', 'loss', 'grad_norm', 'lr', 'wall_s']
BATCH_DTYPE = np.float64


class TrainingLog:
    """HDF5-backed training log with per-batch and per-epoch metrics.

    Thread-safe for single-writer (training loop). Readers can open
    independently for monitoring/plotting.

    Args:
        path: Path to HDF5 file. Created if it doesn't exist.
    """

    def __init__(self, path: str = 'runs/training_log.h5'):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def start_run(self, run_name: str, **attrs) -> None:
        """Create a run group with metadata attributes."""
        with h5py.File(self.path, 'a') as f:
            if run_name in f:
                return  # already exists, resuming
            grp = f.create_group(run_name)
            grp.attrs['created'] = datetime.now().isoformat()
            for k, v in attrs.items():
                if v is not None:
                    grp.attrs[k] = v

    def start_phase(self, run_name: str, phase_name: str, **attrs) -> None:
        """Create a phase group within a run, with config attributes."""
        with h5py.File(self.path, 'a') as f:
            run_grp = f.require_group(run_name)
            if phase_name in run_grp:
                return  # resuming
            phase_grp = run_grp.create_group(phase_name)
            phase_grp.attrs['started'] = datetime.now().isoformat()

            # Store config
            for k, v in attrs.items():
                if v is None:
                    continue
                if isinstance(v, (list, tuple)):
                    phase_grp.attrs[k] = str(v)
                else:
                    phase_grp.attrs[k] = v

            # Create resizable datasets
            phase_grp.create_dataset(
                'epochs',
                shape=(0, len(EPOCH_COLUMNS)),
                maxshape=(None, len(EPOCH_COLUMNS)),
                dtype=EPOCH_DTYPE,
                chunks=(64, len(EPOCH_COLUMNS)),
            )
            phase_grp.create_dataset(
                'batches',
                shape=(0, len(BATCH_COLUMNS)),
                maxshape=(None, len(BATCH_COLUMNS)),
                dtype=BATCH_DTYPE,
                chunks=(1024, len(BATCH_COLUMNS)),
            )
            # Variable-length string dataset for events
            dt = h5py.special_dtype(vlen=str)
            phase_grp.create_dataset(
                'events',
                shape=(0,),
                maxshape=(None,),
                dtype=dt,
                chunks=(64,),
            )

    def log_batch(self, run_name: str, phase_name: str,
                  step: int, epoch: int, batch: int,
                  loss: float, grad_norm: float, lr: float,
                  wall_s: float = 0.0) -> None:
        """Append one batch row."""
        row = np.array([[step, epoch, batch, loss, grad_norm, lr, wall_s]],
                       dtype=BATCH_DTYPE)
        with h5py.File(self.path, 'a') as f:
            ds = f[run_name][phase_name]['batches']
            n = ds.shape[0]
            ds.resize(n + 1, axis=0)
            ds[n] = row

    def log_epoch(self, run_name: str, phase_name: str,
                  epoch: int, train_loss: float, val_loss: float,
                  lr: float, wall_s: float = 0.0) -> None:
        """Append one epoch row."""
        row = np.array([[epoch, train_loss, val_loss, lr, wall_s]],
                       dtype=EPOCH_DTYPE)
        with h5py.File(self.path, 'a') as f:
            ds = f[run_name][phase_name]['epochs']
            n = ds.shape[0]
            ds.resize(n + 1, axis=0)
            ds[n] = row

    def log_event(self, run_name: str, phase_name: str, event: str) -> None:
        """Append a timestamped event string."""
        ts = datetime.now().isoformat()
        entry = f"[{ts}] {event}"
        with h5py.File(self.path, 'a') as f:
            ds = f[run_name][phase_name]['events']
            n = ds.shape[0]
            ds.resize(n + 1, axis=0)
            ds[n] = entry

    def end_phase(self, run_name: str, phase_name: str,
                  best_val_loss: float, best_epoch: int,
                  total_epochs: int, wall_seconds: float) -> None:
        """Write final phase summary attributes."""
        with h5py.File(self.path, 'a') as f:
            grp = f[run_name][phase_name]
            grp.attrs['best_val_loss'] = best_val_loss
            grp.attrs['best_epoch'] = best_epoch
            grp.attrs['total_epochs'] = total_epochs
            grp.attrs['wall_seconds'] = wall_seconds
            grp.attrs['ended'] = datetime.now().isoformat()

    # ------------------------------------------------------------------
    # Read API (for plotting / monitoring)
    # ------------------------------------------------------------------

    def get_epochs(self, run_name: str, phase_name: str) -> np.ndarray:
        """Return epoch data as (N, 5) array. Columns: epoch, train_loss, val_loss, lr, wall_s."""
        with h5py.File(self.path, 'r') as f:
            return f[run_name][phase_name]['epochs'][:]

    def get_batches(self, run_name: str, phase_name: str) -> np.ndarray:
        """Return batch data as (M, 7) array. Columns: step, epoch, batch, loss, grad_norm, lr, wall_s."""
        with h5py.File(self.path, 'r') as f:
            return f[run_name][phase_name]['batches'][:]

    def get_events(self, run_name: str, phase_name: str) -> list[str]:
        """Return all events for a phase."""
        with h5py.File(self.path, 'r') as f:
            return list(f[run_name][phase_name]['events'][:])

    def get_phase_attrs(self, run_name: str, phase_name: str) -> dict:
        """Return phase config + summary attributes."""
        with h5py.File(self.path, 'r') as f:
            return dict(f[run_name][phase_name].attrs)

    def list_runs(self) -> list[str]:
        """List all run names."""
        if not self.path.exists():
            return []
        with h5py.File(self.path, 'r') as f:
            return list(f.keys())

    def list_phases(self, run_name: str) -> list[str]:
        """List all phases in a run."""
        with h5py.File(self.path, 'r') as f:
            return [k for k in f[run_name].keys() if isinstance(f[run_name][k], h5py.Group)]

    # ------------------------------------------------------------------
    # Backfill (import from existing runs)
    # ------------------------------------------------------------------

    def backfill_epochs(self, run_name: str, phase_name: str,
                        epoch_data: list[dict]) -> None:
        """Import epoch data from a list of dicts.

        Each dict should have: epoch, train_loss, val_loss, lr.
        Optional: wall_s (defaults to 0).
        """
        rows = []
        for d in epoch_data:
            rows.append([
                d['epoch'],
                d['train_loss'],
                d['val_loss'],
                d['lr'],
                d.get('wall_s', 0.0),
            ])
        arr = np.array(rows, dtype=EPOCH_DTYPE)

        with h5py.File(self.path, 'a') as f:
            ds = f[run_name][phase_name]['epochs']
            n = ds.shape[0]
            ds.resize(n + len(rows), axis=0)
            ds[n:] = arr
