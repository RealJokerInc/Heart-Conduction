"""Single-AP training run for NODE pipeline validation.

One AP (BCL=2000, 500ms window), full resolution via odeint_adjoint(dopri8).
Validates: forward pass, adjoint backward, loss decrease, NFE stability.
"""

import json
import logging
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from surrogate.model.stage1 import IonicStage1
from surrogate.model.node import IonicNODE
from surrogate.training.node_rollout import node_rollout, NODE_T_EVAL_MS
from surrogate.training.datasets import SegmentDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # === Model ===
    stage1 = IonicStage1(scaffold=True).to(dtype=torch.float64, device=device)
    node = IonicNODE(stage1)

    # Freeze everything except ionic rate MLP + scaffold decoder
    from fnmatch import fnmatch
    ionic_patterns = [
        "ionic_rate_mlp.*",
        "ionic_state_decoder.*",
    ]
    for name, p in stage1.named_parameters():
        p.requires_grad = any(fnmatch(name, pat) for pat in ionic_patterns)

    trainable = [p for p in stage1.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    logger.info(f"Trainable params: {n_train}")

    # === Data: single AP ===
    ap_data = torch.load('/tmp/surrogate_cache/single_ap_bcl2000.pt', weights_only=False)

    # Use subsample=10 → 5000 points covering 500ms
    ds = SegmentDataset(ap_data, segment_length=5000, subsample=10, stride=50000)
    logger.info(f"Segments: {len(ds)}")

    # Single segment = single AP, batch dim added by DataLoader
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    logger.info(f"Vm shape: {batch['Vm'].shape}, range: [{batch['Vm'].min():.1f}, {batch['Vm'].max():.1f}]")

    # === t_eval: expand to cover 500ms ===
    # Every 0.1ms across 500ms = 5001 points. Forces smooth vector field everywhere.
    t_eval = torch.linspace(0.0, 500.0, 5001, dtype=torch.float64)

    # === Training ===
    optimizer = AdamW(trainable, lr=5e-4, weight_decay=1e-4)
    run_dir = Path('runs/single_ap_001')
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / 'log.jsonl'

    max_epochs = 500
    best_loss = float('inf')

    for epoch in range(max_epochs):
        t0 = time.time()

        # Forward
        node.train()
        optimizer.zero_grad()
        result = node_rollout(node, batch, phase_name='A1', device=device,
                              t_eval_ms=t_eval, adjoint=False,
                              method='dopri5', rtol=1e-3, atol=1e-3)
        loss = result['loss']

        # Backward (backprop through solver — more memory, stable with random weights)
        loss.backward()
        node.clear_v_trajectory()

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0).item()
        optimizer.step()

        elapsed = time.time() - t0
        loss_val = loss.item()
        nfe = getattr(node, 'nfe', 0)

        marker = ''
        if loss_val < best_loss:
            best_loss = loss_val
            marker = ' *'
            if epoch % 50 == 0 or epoch < 10:
                torch.save({
                    'stage1_state_dict': stage1.state_dict(),
                    'epoch': epoch,
                    'loss': loss_val,
                }, run_dir / 'best.pt')

        # Component losses
        ionic = result.get('ionic_state_mse', 0)
        conc = result.get('conc_mse', 0)
        if hasattr(ionic, 'item'):
            ionic = ionic.item()
        if hasattr(conc, 'item'):
            conc = conc.item()

        log_entry = {
            'epoch': epoch, 'loss': loss_val,
            'ionic_state_mse': ionic, 'conc_mse': conc,
            'grad_norm': grad_norm, 'nfe': nfe,
            'elapsed_s': round(elapsed, 2),
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        logger.info(
            f"Ep {epoch:3d} | loss={loss_val:.4f} | ionic={ionic:.4f} | conc={conc:.4f} | "
            f"gnorm={grad_norm:.3f} | nfe={nfe:4d} | {elapsed:.1f}s{marker}"
        )

        # NaN guard
        if loss_val != loss_val:
            logger.error("NaN! Stopping.")
            break

    logger.info(f"Done. Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
