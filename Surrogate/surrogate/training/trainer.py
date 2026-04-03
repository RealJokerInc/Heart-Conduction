"""Phase-aware training loop orchestrator for IonicSurrogateV3.

Manages the full A2->E pipeline: data loading per phase, optimizer/scheduler
creation, freeze/unfreeze, train/val epochs, convergence detection, and
phase transitions. Delegates to checkpoint and monitor subsystems.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..model.ionic_surrogate_v3 import IonicSurrogateV3
from .datasets import SnapshotDataset, PairDataset, SegmentDataset, merge_tier_datasets
from .phases import PhaseConfig, get_phase_config, get_all_phases, apply_freeze_mask, PHASE_ORDER
from .rollout import rollout, INIT_CONC, LOSS_NORM

logger = logging.getLogger(__name__)


class SurrogateTrainer:
    """Orchestrates the full training pipeline from Phase A2 through E.

    Args:
        model: IonicSurrogateV3 with scaffold=True.
        cache_dir: Path to SSD cache with preprocessed .pt files.
        run_dir: Path to run output directory for logs, checkpoints, etc.
        device: 'cuda' or 'cpu'.
        start_phase: Phase name to start from (for resuming).
    """

    def __init__(
        self,
        model: IonicSurrogateV3,
        cache_dir: str,
        run_dir: str,
        device: str = 'cuda',
        start_phase: str = 'A2',
    ):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.run_dir = Path(run_dir)
        self.device = torch.device(device)
        self.start_phase = start_phase

        # Create run directory structure
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / 'checkpoints').mkdir(exist_ok=True)

        # Global step counter (never resets across phases)
        self.global_step = 0

        # Move model to device with float64 (project convention)
        self.model = self.model.to(dtype=torch.float64, device=self.device)

    def train(self) -> None:
        """Run full training pipeline from start_phase through E."""
        all_phases = get_all_phases()
        start_idx = PHASE_ORDER.index(self.start_phase)

        for phase_config in all_phases[start_idx:]:
            logger.info(f"=== Starting Phase {phase_config.name} ===")
            metrics = self.train_phase(phase_config)
            logger.info(
                f"=== Phase {phase_config.name} complete: "
                f"{phase_config.transition_metric}={metrics.get(phase_config.transition_metric, 'N/A')} ==="
            )

        logger.info("Training complete.")

    def train_phase(self, phase: PhaseConfig) -> dict:
        """Train one phase to convergence. Returns final val metrics."""
        # Apply freeze mask
        apply_freeze_mask(self.model, phase)

        # Collect trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if not trainable_params:
            logger.warning(f"Phase {phase.name}: no trainable parameters!")
            return {}

        # Create optimizer and scheduler
        optimizer = AdamW(trainable_params, lr=phase.lr, weight_decay=phase.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=phase.max_epochs)

        # Create data loaders
        train_loader = self._make_dataloader(phase, split='train')
        val_loader = self._make_dataloader(phase, split='val')

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_metrics = {}

        for epoch in range(phase.max_epochs):
            # Train epoch
            train_metrics = self._train_epoch(phase, train_loader, optimizer)
            train_loss = train_metrics['train_loss']
            scheduler.step()

            # Validate
            val_metrics = self._validate(phase, val_loader)
            val_loss = val_metrics.get(phase.transition_metric, float('inf'))

            # Build component log string
            components = []
            for k in ['train_ionic_state_mse', 'train_conc_mse', 'train_conductance_mse', 'train_grad_norm']:
                if k in train_metrics:
                    short = k.replace('train_', '')
                    components.append(f"{short}={train_metrics[k]:.4f}")
            for k in ['val_ionic_state_mse', 'val_conc_mse', 'val_conductance_mse']:
                if k in val_metrics:
                    short = k.replace('val_', 'v_')
                    components.append(f"{short}={val_metrics[k]:.4f}")
            comp_str = ' | '.join(components) if components else ''

            logger.info(
                f"Phase {phase.name} Epoch {epoch}: "
                f"train={train_loss:.6f}, val={val_loss:.6f}, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
                + (f" | {comp_str}" if comp_str else "")
            )

            # Best model tracking
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_metrics = val_metrics.copy()
                self._save_checkpoint(f'best_{phase.name}', phase, epoch, optimizer, scheduler, best_val_loss)
            else:
                epochs_no_improve += 1

            # Save latest checkpoint every epoch
            self._save_checkpoint('latest', phase, epoch, optimizer, scheduler, best_val_loss)

            # Check transition criteria
            if self._should_transition(phase, val_metrics, epochs_no_improve):
                logger.info(f"Phase {phase.name} converged at epoch {epoch}")
                break

        return best_metrics

    def _train_epoch(self, phase: PhaseConfig, dataloader: DataLoader, optimizer: AdamW) -> float:
        """Run one training epoch. Returns mean loss."""
        self.model.train()

        total_loss = 0.0
        n_batches = 0

        component_sums: dict[str, float] = {}

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            optimizer.zero_grad()

            if phase.loss_fn == "concentration":
                loss_val = self._single_step_loss(phase, batch)
                loss = loss_val  # scalar tensor
                batch_components = {'conc_mse': loss_val.item()}
            elif phase.rollout_length <= 1:
                result = self._snapshot_step(phase, batch)
                loss = result['loss']
                batch_components = {k: v.item() for k, v in result.items() if k != 'loss'}
            else:
                result = self._rollout_step(phase, batch)
                loss = result['loss']
                batch_components = {k: v.item() if hasattr(v, 'item') else v
                                    for k, v in result.items()
                                    if k not in ('loss', 'per_step_losses')}

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0,
            ).item()
            optimizer.step()

            total_loss += loss.item()
            for k, v in batch_components.items():
                component_sums[k] = component_sums.get(k, 0.0) + v
            component_sums['grad_norm'] = component_sums.get('grad_norm', 0.0) + grad_norm
            n_batches += 1
            self.global_step += 1

        n = max(n_batches, 1)
        metrics = {'train_loss': total_loss / n}
        for k, v in component_sums.items():
            metrics[f'train_{k}'] = v / n
        return metrics

    def _single_step_loss(self, phase: PhaseConfig, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Phase A2: single-step concentration loss (no rollout).

        Build carried_state from zeros + true concentrations, then forward
        to get concentration output.
        """
        conc_t = batch['concentrations_t']       # (B, 4)
        Vm = batch['Vm_t']
        dt = batch['dt_t']
        B = Vm.shape[0]

        # Build carried_state: zeros for ionic latent + true concentrations
        carried = torch.zeros(B, self.model.carried_dim, dtype=torch.float64, device=self.device)
        carried[:, self.model.ionic_dim:] = conc_t

        out = self.model(
            carried, Vm, dt,
            torch.zeros(B, self.model.cond_dim, device=self.device, dtype=torch.float64),
            conc_t,
        )
        return nn.functional.mse_loss(out['concentrations'], batch['concentrations_t1'])

    def _snapshot_step(self, phase: PhaseConfig, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Single-step forward from zeros for rollout=1 phases (B1).

        SnapshotDataset provides (B, ...) tensors without time dimension.
        Returns dict with 'loss' (total) and per-component losses.
        """
        B = batch['Vm'].shape[0]
        Vm = batch['Vm']
        dt = batch['dt']

        # Start from zeros (steady state)
        carried = torch.zeros(B, self.model.carried_dim, dtype=torch.float64, device=self.device)
        carried[:, self.model.ionic_dim:] = INIT_CONC.to(self.device)
        cond_lat_prev = torch.zeros(B, self.model.cond_dim, dtype=torch.float64, device=self.device)
        conc_prev = INIT_CONC.to(self.device).unsqueeze(0).expand(B, -1).clone()

        out = self.model(carried, Vm, dt, cond_lat_prev, conc_prev)

        ionic_mse = nn.functional.mse_loss(out['ionic_state_pred'], batch['ionic_states'])
        conc_mse = nn.functional.mse_loss(out['concentrations'], batch['concentrations'])

        losses = {'ionic_state_mse': ionic_mse, 'conc_mse': conc_mse}
        total = ionic_mse / LOSS_NORM['ionic_state'] + conc_mse / LOSS_NORM['conc']

        if phase.loss_fn == "ionic_state_and_conductance" and out['conductance_pred'] is not None:
            cond_mse = nn.functional.mse_loss(out['conductance_pred'], batch['conductance_products'])
            losses['conductance_mse'] = cond_mse
            total = total + cond_mse / LOSS_NORM['conductance']

        losses['loss'] = total
        return losses

    def _rollout_step(self, phase: PhaseConfig, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Phase B2-E: autoregressive rollout over segments."""
        return rollout(
            model=self.model,
            segment=batch,
            phase_name=phase.name,
            device=self.device,
        )

    @torch.no_grad()
    def _validate(self, phase: PhaseConfig, dataloader: DataLoader) -> dict:
        """Validation pass. Returns metrics dict with component breakdowns."""
        self.model.eval()

        total_loss = 0.0
        component_sums: dict[str, float] = {}
        n_batches = 0

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            if phase.loss_fn == "concentration":
                loss_val = self._single_step_loss(phase, batch)
                loss = loss_val.item()
                batch_components = {'conc_mse': loss}
            elif phase.rollout_length <= 1:
                result = self._snapshot_step(phase, batch)
                loss = result['loss'].item()
                batch_components = {k: v.item() for k, v in result.items() if k != 'loss'}
            else:
                result = self._rollout_step(phase, batch)
                loss = result['loss'].item()
                batch_components = {k: v.item() if hasattr(v, 'item') else v
                                    for k, v in result.items()
                                    if k not in ('loss', 'per_step_losses')}

            total_loss += loss
            for k, v in batch_components.items():
                component_sums[k] = component_sums.get(k, 0.0) + v
            n_batches += 1

        n = max(n_batches, 1)
        metrics = {'val_loss': total_loss / n}
        for k, v in component_sums.items():
            metrics[f'val_{k}'] = v / n

        # Add transition metric alias
        metric_map = {
            "concentration": "val_conc_mse",
            "ionic_state": "val_ionic_state_mse",
            "ionic_state_and_conductance": "val_ionic_state_mse",
            "concentration_rollout": "val_conc_mse",
            "I_ion": "val_I_ion_mse",
        }
        metric_name = metric_map.get(phase.loss_fn, phase.transition_metric)
        metrics[metric_name] = total_loss / n

        return metrics

    def _should_transition(self, phase: PhaseConfig, val_metrics: dict, epochs_no_improve: int) -> bool:
        """Check if phase should transition to next."""
        val_loss = val_metrics.get(phase.transition_metric, float('inf'))

        # Threshold-based transition
        if phase.transition_threshold is not None and val_loss < phase.transition_threshold:
            return True

        # Patience-based transition
        if epochs_no_improve >= phase.patience:
            return True

        return False

    def _make_dataloader(self, phase: PhaseConfig, split: str = 'train') -> DataLoader:
        """Create DataLoader for a phase."""
        datasets = []
        for tier in phase.data_tiers:
            try:
                path = self.cache_dir / f'tier{tier:02d}_{split}.pt'
                data = torch.load(path, weights_only=False)
            except FileNotFoundError:
                logger.warning(f"Cache for tier {tier} split '{split}' not found, skipping")
                continue

            if phase.loss_fn == "concentration":
                datasets.append(PairDataset(data))
            elif phase.rollout_length <= 1:
                # For rollout=1 phases (B1), use SnapshotDataset to avoid
                # creating millions of 1-step segments in memory
                datasets.append(SnapshotDataset(data))
            else:
                datasets.append(SegmentDataset(data, segment_length=phase.rollout_length))

        if not datasets:
            raise RuntimeError(f"No data found for phase {phase.name} split '{split}'")

        merged = merge_tier_datasets(datasets) if len(datasets) > 1 else datasets[0]
        shuffle = (split == 'train')
        return DataLoader(merged, batch_size=phase.batch_size, shuffle=shuffle, drop_last=True)

    def _save_checkpoint(self, tag: str, phase: PhaseConfig, epoch: int,
                         optimizer: AdamW, scheduler: CosineAnnealingLR, best_val_loss: float) -> None:
        """Save checkpoint."""
        ckpt = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'phase': phase.name,
            'epoch': epoch,
            'step': self.global_step,
            'best_val_loss': best_val_loss,
        }
        path = self.run_dir / 'checkpoints' / f'{tag}.pt'
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str) -> dict:
        """Load checkpoint and restore state."""
        ckpt = torch.load(path, weights_only=False, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.global_step = ckpt.get('step', 0)
        self.start_phase = ckpt.get('phase', 'A2')
        return ckpt
