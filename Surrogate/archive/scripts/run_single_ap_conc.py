"""Single-AP concentration training. Loads best ionic checkpoint, freezes all, trains conc only."""

import json
import logging
import time
from pathlib import Path
from fnmatch import fnmatch

import torch
from torch.optim import AdamW

from surrogate.model.stage1 import IonicStage1
from surrogate.model.node import IonicNODE
from surrogate.training.node_rollout import node_rollout
from surrogate.training.datasets import SegmentDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load best ionic checkpoint
    stage1 = IonicStage1(scaffold=True).to(dtype=torch.float64, device=device)
    ckpt = torch.load('runs/single_ap_001/best.pt', weights_only=False, map_location=device)
    stage1.load_state_dict(ckpt['stage1_state_dict'])
    logger.info(f"Loaded checkpoint from epoch {ckpt['epoch']}, ionic loss={ckpt['loss']:.4f}")

    node = IonicNODE(stage1)

    # Freeze everything, unfreeze only conc_kan
    for p in stage1.parameters():
        p.requires_grad = False
    for name, p in stage1.named_parameters():
        if fnmatch(name, 'conc_kan.*'):
            p.requires_grad = True

    trainable = [p for p in stage1.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    logger.info(f"Trainable: {n_train} (conc_kan only)")

    # Data
    ap_data = torch.load('/tmp/surrogate_cache/single_ap_bcl2000.pt', weights_only=False)
    ds = SegmentDataset(ap_data, segment_length=5000, subsample=10, stride=50000)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)))
    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    t_eval = torch.tensor(
        [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0,
         10.0, 20.0, 40.0, 80.0, 120.0, 160.0, 200.0, 240.0, 300.0, 350.0, 400.0, 450.0, 500.0],
        dtype=torch.float64,
    )

    optimizer = AdamW(trainable, lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)
    run_dir = Path('runs/single_ap_conc_001')
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / 'log.jsonl'

    max_epochs = 500
    best_loss = float('inf')

    for epoch in range(max_epochs):
        t0 = time.time()
        node.train()
        optimizer.zero_grad()

        result = node_rollout(node, batch, phase_name='conc_only', device=device,
                              t_eval_ms=t_eval, adjoint=False,
                              method='dopri5', rtol=1e-3, atol=1e-3)
        loss = result['loss']
        loss.backward()
        node.clear_v_trajectory()

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0).item()
        optimizer.step()
        scheduler.step()

        elapsed = time.time() - t0
        loss_val = loss.item()
        ionic = result.get('ionic_state_mse', 0)
        conc = result.get('conc_mse', 0)
        if hasattr(ionic, 'item'): ionic = ionic.item()
        if hasattr(conc, 'item'): conc = conc.item()

        log_entry = {
            'epoch': epoch, 'loss': loss_val,
            'ionic_state_mse': ionic, 'conc_mse': conc,
            'grad_norm': grad_norm, 'nfe': 0, 'elapsed_s': round(elapsed, 2),
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        marker = ''
        if loss_val < best_loss:
            best_loss = loss_val
            marker = ' *'
            if epoch % 50 == 0 or epoch < 10:
                torch.save({
                    'stage1_state_dict': stage1.state_dict(),
                    'epoch': epoch, 'loss': loss_val,
                }, run_dir / 'best.pt')

        logger.info(
            f"Ep {epoch:3d} | conc={conc:.4f} | ionic={ionic:.4f} | "
            f"gnorm={grad_norm:.3f} | {elapsed:.1f}s{marker}"
        )

        if loss_val != loss_val:
            logger.error("NaN! Stopping.")
            break

    torch.save({
        'stage1_state_dict': stage1.state_dict(),
        'epoch': epoch, 'loss': loss_val,
    }, run_dir / 'final.pt')
    logger.info(f"Done. Best conc loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
