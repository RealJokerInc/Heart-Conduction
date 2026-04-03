"""Phase-aware training loop orchestrator for IonicSurrogateV3.

Manages the full A1->E pipeline: data loading per phase, optimizer/scheduler
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
from .encoder import TemporaryEncoder, make_carried_state
from .phases import PhaseConfig, get_phase_config, get_all_phases, apply_freeze_mask, PHASE_ORDER
from .rollout import rollout, INIT_CONC

logger = logging.getLogger(__name__)


class SurrogateTrainer:
    """Orchestrates the full training pipeline from Phase A1 through E.

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
        start_phase: str = 'A1',
    ):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.run_dir = Path(run_dir)
        self.device = torch.device(device)
        self.start_phase = start_phase

        # Create run directory structure
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / 'checkpoints').mkdir(exist_ok=True)

        # Temporary encoder — created at A1, discarded after B
        self.encoder: Optional[TemporaryEncoder] = None

        # Global step counter (never resets across phases)
        self.global_step = 0

        # Move model to device
        self.model = self.model.to(self.device)

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
        # Create encoder if needed
        if phase.uses_encoder and self.encoder is None:
            self.encoder = TemporaryEncoder().to(self.device).to(torch.float64)
            logger.info("Created temporary encoder")

        # Discard encoder when no longer needed
        if not phase.uses_encoder and self.encoder is not None:
            self.encoder = None
            logger.info("Discarded temporary encoder")

        # Apply freeze mask
        apply_freeze_mask(self.model, phase)

        # Collect trainable parameters (model + encoder if applicable)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if phase.uses_encoder and self.encoder is not None:
            trainable_params.extend(self.encoder.parameters())

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
            train_loss = self._train_epoch(phase, train_loader, optimizer)
            if epoch > 0 or train_loss is not None:
                scheduler.step()

            # Validate
            val_metrics = self._validate(phase, val_loader)
            val_loss = val_metrics.get(phase.transition_metric, float('inf'))

            logger.info(
                f"Phase {phase.name} Epoch {epoch}: "
                f"train_loss={train_loss:.6f}, "
                f"{phase.transition_metric}={val_loss:.6f}, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
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
        if self.encoder is not None:
            self.encoder.train()

        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            optimizer.zero_grad()

            if phase.loss_fn == "autoencoder":
                loss = self._autoencoder_step(batch)
            elif phase.loss_fn in ("concentration", "conductance"):
                loss = self._single_step_loss(phase, batch)
            else:
                # Rollout-based phases (B-E)
                loss = self._rollout_step(phase, batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad] +
                ([p for p in self.encoder.parameters()] if self.encoder is not None else []),
                max_norm=1.0,
            )
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

        return total_loss / max(n_batches, 1)

    def _autoencoder_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Phase A1: encode true states -> decode -> reconstruct."""
        ionic_states = batch['ionic_states']  # (B, 14)
        latent = self.encoder(ionic_states)    # (B, 16)
        decoded = self.model.stage1.ionic_state_decoder(latent)  # (B, 14)
        return nn.functional.mse_loss(decoded, ionic_states)

    def _single_step_loss(self, phase: PhaseConfig, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Phase A2/A3: single-step forward pass (no rollout)."""
        if phase.loss_fn == "concentration":
            # A2: attention concentration tracking
            ionic_states = batch['ionic_states_t']  # (B, 14)
            conc_t = batch['concentrations_t']       # (B, 4)
            Vm = batch['Vm_t']
            dt = batch['dt_t']

            carried = make_carried_state(self.encoder, ionic_states, conc_t)
            out = self.model(
                carried, Vm, dt,
                torch.zeros(carried.shape[0], self.model.cond_dim, device=self.device, dtype=torch.float64),
                conc_t,
            )
            return nn.functional.mse_loss(out['concentrations'], batch['concentrations_t1'])

        elif phase.loss_fn == "conductance":
            # A3: gate conductance projection
            ionic_states = batch['ionic_states']   # (B, 14)
            conc = batch['concentrations']          # (B, 4)
            carried = make_carried_state(self.encoder, ionic_states, conc)
            out = self.model(
                carried, batch['Vm'], batch['dt'],
                torch.zeros(carried.shape[0], self.model.cond_dim, device=self.device, dtype=torch.float64),
                conc,
            )
            return nn.functional.mse_loss(out['conductance_pred'], batch['conductance_products'])

        raise ValueError(f"Unknown single-step loss: {phase.loss_fn}")

    def _rollout_step(self, phase: PhaseConfig, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Phase B-E: autoregressive rollout."""
        result = rollout(
            model=self.model,
            segment=batch,
            encoder=self.encoder if phase.uses_encoder else None,
            scheduled_sampling_p=phase.scheduled_sampling_p,
            phase_name=phase.name,
            device=self.device,
        )
        return result['loss']

    @torch.no_grad()
    def _validate(self, phase: PhaseConfig, dataloader: DataLoader) -> dict:
        """Validation pass. Returns metrics dict."""
        self.model.eval()
        if self.encoder is not None:
            self.encoder.eval()

        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            if phase.loss_fn == "autoencoder":
                loss = self._autoencoder_step(batch)
            elif phase.loss_fn in ("concentration", "conductance"):
                loss = self._single_step_loss(phase, batch)
            else:
                loss = self._rollout_step(phase, batch)

            total_loss += loss.item()
            n_batches += 1

        mean_loss = total_loss / max(n_batches, 1)

        # Map to transition metric name
        metric_map = {
            "autoencoder": "val_recon_mse",
            "concentration": "val_conc_mse",
            "conductance": "val_cond_mse",
            "ionic_state": "val_ionic_state_mse",
            "concentration_rollout": "val_conc_mse",
            "I_ion": "val_I_ion_mse",
        }
        metric_name = metric_map.get(phase.loss_fn, phase.transition_metric)
        return {metric_name: mean_loss}

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

            if phase.loss_fn == "autoencoder" or phase.loss_fn == "conductance":
                datasets.append(SnapshotDataset(data))
            elif phase.loss_fn == "concentration":
                datasets.append(PairDataset(data))
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
            'encoder_state_dict': self.encoder.state_dict() if self.encoder else None,
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
        if ckpt.get('encoder_state_dict') is not None:
            if self.encoder is None:
                self.encoder = TemporaryEncoder().to(self.device).to(torch.float64)
            self.encoder.load_state_dict(ckpt['encoder_state_dict'])
        self.global_step = ckpt.get('step', 0)
        self.start_phase = ckpt.get('phase', 'A1')
        return ckpt
