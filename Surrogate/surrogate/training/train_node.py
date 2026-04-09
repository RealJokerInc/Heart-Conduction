"""NODE training script for IonicSurrogateV3.

Usage:
    conda run -n heart-conduction python -m surrogate.training.train_node

Phase A1: Half 1 (attention + ionic MLP + ionic decoder) on T1 data.
See TRAINING_STRATEGY.md for full plan.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..model.stage1 import IonicStage1
from ..model.node import IonicNODE
from .datasets import SegmentDataset
from .node_rollout import node_rollout
from .phases import _HALF1_PARAMS, _HALF2_PARAMS, _ALL_STAGE1

logger = logging.getLogger(__name__)


def apply_freeze(stage1: IonicStage1, trainable_patterns: list[str]) -> None:
    """Freeze all params, then unfreeze those matching patterns."""
    from fnmatch import fnmatch
    for name, p in stage1.named_parameters():
        p.requires_grad = any(fnmatch(name, pat) for pat in trainable_patterns)


PHASE_PARAMS = {
    "A1": _HALF1_PARAMS,
    "A2": _HALF1_PARAMS,
    "A3": _HALF1_PARAMS,
    "A4": _HALF1_PARAMS,
    "B1": _HALF2_PARAMS,
    "B2": _HALF2_PARAMS,
    "B3": _HALF2_PARAMS,
    "B4": _HALF2_PARAMS,
}


def make_dataloaders(
    cache_dir: Path,
    tier: int,
    segment_length: int,
    subsample: int,
    stride: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Create train and val DataLoaders from cached tier data."""
    train_data = torch.load(cache_dir / f'tier{tier:02d}_train.pt', weights_only=False)
    val_data = torch.load(cache_dir / f'tier{tier:02d}_val.pt', weights_only=False)

    train_ds = SegmentDataset(train_data, segment_length=segment_length,
                              subsample=subsample, stride=stride)
    val_ds = SegmentDataset(val_data, segment_length=segment_length,
                            subsample=subsample, stride=stride)

    effective_bs = min(batch_size, len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=effective_bs, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, len(val_ds)),
                            shuffle=False, drop_last=False)

    logger.info(f"Train: {len(train_ds)} segments, {len(train_loader)} batches")
    logger.info(f"Val:   {len(val_ds)} segments, {len(val_loader)} batches")
    return train_loader, val_loader


def train_phase(
    node: IonicNODE,
    phase_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    run_dir: Path,
    device: torch.device,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 50,
    z0_noise_sigma: float = 0.0,
) -> dict:
    """Train one phase. Returns best validation metrics."""

    # Freeze/unfreeze
    patterns = PHASE_PARAMS.get(phase_name, _ALL_STAGE1)
    # Patterns reference "stage1.X" but named_parameters on stage1 gives "X"
    # Strip the "stage1." prefix for matching against stage1's own params
    stage1_patterns = [p.replace("stage1.", "", 1) for p in patterns]
    apply_freeze(node.stage1, stage1_patterns)

    trainable = [p for p in node.stage1.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    logger.info(f"Phase {phase_name}: {n_trainable} trainable params")

    if not trainable:
        logger.warning("No trainable parameters!")
        return {}

    optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    log_path = run_dir / f'log_{phase_name}.jsonl'

    for epoch in range(max_epochs):
        t0 = time.time()

        # === Train ===
        node.train()
        train_loss_sum = 0.0
        train_components: dict[str, float] = {}
        grad_norm_sum = 0.0
        nfe_sum = 0
        n_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            optimizer.zero_grad()

            result = node_rollout(node, batch, phase_name=phase_name,
                                  device=device, z0_noise_sigma=z0_noise_sigma)
            result['loss'].backward()

            # IMPORTANT: clear V trajectory AFTER backward (adjoint needs it)
            node.clear_v_trajectory()

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0).item()
            optimizer.step()

            train_loss_sum += result['loss'].item()
            grad_norm_sum += grad_norm
            nfe_sum += node.nfe
            for k, v in result.items():
                if k != 'loss':
                    val = v.item() if hasattr(v, 'item') else v
                    train_components[k] = train_components.get(k, 0.0) + val
            n_batches += 1

        n = max(n_batches, 1)
        train_loss = train_loss_sum / n
        mean_grad_norm = grad_norm_sum / n
        mean_nfe = nfe_sum / n

        # === Validate ===
        node.eval()
        val_loss_sum = 0.0
        val_components: dict[str, float] = {}
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                result = node_rollout(node, batch, phase_name=phase_name, device=device)
                node.clear_v_trajectory()

                val_loss_sum += result['loss'].item()
                for k, v in result.items():
                    if k != 'loss':
                        val = v.item() if hasattr(v, 'item') else v
                        val_components[k] = val_components.get(k, 0.0) + val
                val_batches += 1

        vn = max(val_batches, 1)
        val_loss = val_loss_sum / vn

        elapsed = time.time() - t0

        # === Log ===
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'grad_norm': mean_grad_norm,
            'nfe_mean': mean_nfe,
            'lr': optimizer.param_groups[0]['lr'],
            'z0_noise_sigma': z0_noise_sigma,
            'elapsed_s': round(elapsed, 1),
        }
        for k, v in train_components.items():
            log_entry[f'train_{k}'] = v / n
        for k, v in val_components.items():
            log_entry[f'val_{k}'] = v / vn

        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # === Console ===
        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            marker = ' *'
            torch.save({
                'stage1_state_dict': node.stage1.state_dict(),
                'phase': phase_name,
                'epoch': epoch,
                'val_loss': val_loss,
            }, run_dir / f'best_{phase_name}.pt')
        else:
            epochs_no_improve += 1

        logger.info(
            f"Ep {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f} | "
            f"gnorm={mean_grad_norm:.2f} | nfe={mean_nfe:.0f} | "
            f"{elapsed:.1f}s{marker}"
        )

        # === Early stopping ===
        if epochs_no_improve >= patience:
            logger.info(f"Patience ({patience}) reached. Stopping.")
            break

        # === NaN guard ===
        if not (train_loss == train_loss):  # NaN check
            logger.error("NaN loss detected! Stopping.")
            break

    # Save final checkpoint
    torch.save({
        'stage1_state_dict': node.stage1.state_dict(),
        'phase': phase_name,
        'epoch': epoch,
        'val_loss': val_loss,
    }, run_dir / f'final_{phase_name}.pt')

    return {'best_val_loss': best_val_loss}


def main():
    parser = argparse.ArgumentParser(description='NODE training for IonicSurrogateV3')
    parser.add_argument('--phase', default='A1', help='Phase name (A1-A4, B1-B4)')
    parser.add_argument('--cache-dir', default='/tmp/surrogate_cache', help='Path to cached data')
    parser.add_argument('--run-dir', default='runs/node_001', help='Output directory')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--tier', type=int, default=1, help='Data tier')
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--segment-length', type=int, default=3000, help='Points per segment')
    parser.add_argument('--subsample', type=int, default=10, help='Subsample factor for V trajectory')
    parser.add_argument('--stride', type=int, default=15000, help='Stride in raw timesteps')
    parser.add_argument('--z0-noise', type=float, default=0.0, help='z0 noise sigma')
    parser.add_argument('--checkpoint', default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Create model
    stage1 = IonicStage1(scaffold=True).to(dtype=torch.float64, device=device)
    node = IonicNODE(stage1)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
        stage1.load_state_dict(ckpt['stage1_state_dict'])
        logger.info(f"Loaded checkpoint: {args.checkpoint}")

    logger.info(f"Inference params: {stage1.inference_param_count()}")
    logger.info(f"Total params: {sum(p.numel() for p in stage1.parameters())}")

    # Data
    train_loader, val_loader = make_dataloaders(
        cache_dir=Path(args.cache_dir),
        tier=args.tier,
        segment_length=args.segment_length,
        subsample=args.subsample,
        stride=args.stride,
        batch_size=args.batch_size,
    )

    # Train
    result = train_phase(
        node=node,
        phase_name=args.phase,
        train_loader=train_loader,
        val_loader=val_loader,
        run_dir=run_dir,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        z0_noise_sigma=args.z0_noise,
    )

    logger.info(f"Done. Best val loss: {result.get('best_val_loss', 'N/A')}")


if __name__ == '__main__':
    main()
