#!/usr/bin/env python
"""Train IonicSurrogateV3.

Usage:
    python train.py                           # Full training from Phase A1
    python train.py --start-phase B1          # Resume from Phase B1
    python train.py --resume runs/run_001/checkpoints/latest.pt
    python train.py --dry-run                 # Create run dir, verify setup, exit
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from surrogate.model.ionic_surrogate_v3 import IonicSurrogateV3
from surrogate.training.data_cache import CacheBuilder
from surrogate.training.trainer import SurrogateTrainer


def main():
    parser = argparse.ArgumentParser(description='Train IonicSurrogateV3')
    parser.add_argument('--run-name', default=None,
                        help='Run name (auto-generated from timestamp if not set)')
    parser.add_argument('--start-phase', default='A1',
                        help='Phase to start from (default: A1)')
    parser.add_argument('--cache-dir', default='/tmp/surrogate_cache',
                        help='SSD cache directory for preprocessed data')
    parser.add_argument('--raw-dir', default='/media/HDD/surrogate_data/raw',
                        help='HDD raw HDF5 directory')
    parser.add_argument('--device', default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--resume', default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--dry-run', action='store_true',
                        help='Create run dir and verify setup without training')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    logger = logging.getLogger('train')

    # Run name and directory
    run_name = args.run_name or f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path(__file__).parent / 'runs' / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run directory: {run_dir}")

    # Build data cache if needed
    tiers = [1, 2, 3, 12]
    cache = CacheBuilder(raw_dir=args.raw_dir, cache_dir=args.cache_dir)
    if not cache.is_cached(tiers=tiers):
        logger.info(f"Building data cache for tiers {tiers} from {args.raw_dir}...")
        cache.build_all(tiers=tiers)
        cache.compute_normalization_stats(tiers=[1, 2, 3])
        logger.info("Cache built successfully.")
    else:
        logger.info("Data cache already exists.")

    # Create model
    model = IonicSurrogateV3(scaffold=True)
    logger.info(f"Model: {model.inference_param_count()} inference params")

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    if args.dry_run:
        logger.info("Dry run complete. Exiting.")
        return

    # Create trainer
    trainer = SurrogateTrainer(
        model=model,
        cache_dir=args.cache_dir,
        run_dir=str(run_dir),
        device=args.device,
        start_phase=args.start_phase,
    )

    # Resume from checkpoint if requested
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    logger.info(f"Starting training from Phase {args.start_phase}")
    trainer.train()
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
