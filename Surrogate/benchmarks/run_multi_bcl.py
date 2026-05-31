"""Multi-BCL + S1S2 ionic training. T1 (steady-state) + T2 (restitution).

T1: Variable-length segments per BCL, full 0.1ms resolution.
T2: S1S2 protocols — last 2 beats (S1 + S2) extracted per DI.
"""

import json
import logging
import random
import time
from pathlib import Path
from fnmatch import fnmatch

import torch
from torch.optim import AdamW

from surrogate.model.stage1 import IonicStage1
from surrogate.model.node import IonicNODE
from surrogate.training.node_rollout import node_rollout

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

SUBSAMPLE = 10  # dt = 0.1ms


def extract_beats(data: dict, bcls: list[int], n_beats: int = 20) -> list[dict]:
    """Extract individual beats from concatenated T1 data, subsampled."""
    beats = []
    offset = 0
    for bcl in bcls:
        steps_per_beat = int(bcl / 0.01)  # raw steps
        for beat_idx in range(n_beats):
            start = offset + beat_idx * steps_per_beat
            indices = list(range(start, start + steps_per_beat, SUBSAMPLE))
            seg = {}
            for k, v in data.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 1:
                    seg[k] = v[indices]
            if 'dt' in seg:
                seg['dt'] = seg['dt'] * SUBSAMPLE
            seg['_bcl'] = bcl
            seg['_beat'] = beat_idx
            seg['_tier'] = 'T1'
            beats.append(seg)
        offset += steps_per_beat * n_beats
    return beats


def extract_s1s2_full(data: dict, dis: list[int]) -> list[dict]:
    """Extract full S1S2 protocol per DI (all 10 S1 beats + DI + S2).

    ODE integrates from z=0 through all S1 beats, naturally building
    the correct ionic state before the S2 response.
    """
    segments = []
    offset = 0
    for di in dis:
        total_steps = int((11000 + di) / 0.01)
        total_ms = 11000 + di

        indices = list(range(offset, offset + total_steps, SUBSAMPLE))
        seg = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                seg[k] = v[indices]
        if 'dt' in seg:
            seg['dt'] = seg['dt'] * SUBSAMPLE
        seg['_bcl'] = total_ms  # for variable t_eval
        seg['_beat'] = 0
        seg['_tier'] = f'T2_DI{di}'
        segments.append(seg)

        offset += total_steps
    return segments


def beat_to_batch(seg: dict, device: torch.device) -> dict:
    """Add batch dim, cast to float64, move to device."""
    return {k: v.unsqueeze(0).to(dtype=torch.float64, device=device)
            for k, v in seg.items() if isinstance(v, torch.Tensor)}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # === Model ===
    stage1 = IonicStage1(scaffold=True).to(dtype=torch.float64, device=device)
    node = IonicNODE(stage1)

    # Warm start: prefer latest multi-BCL checkpoint
    ckpt_path = Path('runs/multi_bcl_002/best.pt')
    if not ckpt_path.exists():
        ckpt_path = Path('runs/multi_bcl_001/best.pt')
    if not ckpt_path.exists():
        ckpt_path = Path('runs/single_ap_001/best.pt')
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        stage1.load_state_dict(ckpt['stage1_state_dict'])
        logger.info(f"Warm start from {ckpt_path} (epoch {ckpt.get('epoch', '?')}, loss {ckpt.get('val_loss', ckpt.get('loss', '?'))})")

    # Freeze: ionic_rate_mlp + scaffold decoder only
    for name, p in stage1.named_parameters():
        p.requires_grad = any(fnmatch(name, pat)
                              for pat in ["ionic_rate_mlp.*", "ionic_state_decoder.*"])

    trainable = [p for p in stage1.parameters() if p.requires_grad]
    logger.info(f"Trainable: {sum(p.numel() for p in trainable)}")

    # === Data: T1 (steady-state) + T2 (restitution) ===
    t1_train = torch.load('/tmp/surrogate_cache/tier01_train.pt', weights_only=False)
    t1_val = torch.load('/tmp/surrogate_cache/tier01_val.pt', weights_only=False)
    t2_train = torch.load('/tmp/surrogate_cache/tier02_train.pt', weights_only=False)
    t2_val = torch.load('/tmp/surrogate_cache/tier02_val.pt', weights_only=False)

    # T1: last 5 beats per BCL for train, last 3 for val
    train_bcls = [300, 500, 700, 1000, 1500]
    val_bcls = [400, 600, 800, 2000]
    t1_train_beats = [b for b in extract_beats(t1_train, train_bcls) if b['_beat'] >= 15]
    t1_val_beats = [b for b in extract_beats(t1_val, val_bcls) if b['_beat'] >= 17]

    # T2: S1S2 segments (last S1 + S2 per DI)
    # Need to figure out which DIs are in train vs val
    # Train: ~53600ms, Val: ~32575ms. Just extract all DIs from each split.
    all_dis = [50, 75, 100, 150, 200, 300, 500, 800]
    # Try extracting — if a DI doesn't exist in the split, it'll be out of bounds
    # Safer: extract based on total length
    t2_train_steps = t2_train['Vm'].shape[0]
    t2_val_steps = t2_val['Vm'].shape[0]

    # Figure out which DIs fit in each split by cumulative steps
    def find_dis_in_split(total_steps, all_dis):
        dis = []
        cumul = 0
        for di in all_dis:
            proto_steps = int((11000 + di) / 0.01)
            if cumul + proto_steps <= total_steps:
                dis.append(di)
                cumul += proto_steps
            else:
                break
        return dis

    train_dis = find_dis_in_split(t2_train_steps, all_dis)
    val_dis = find_dis_in_split(t2_val_steps, [d for d in all_dis if d not in train_dis])
    # If val_dis empty, remaining DIs might start from different offset
    # Simpler: just try all DIs not in train
    remaining_dis = [d for d in all_dis if d not in train_dis]
    val_dis = find_dis_in_split(t2_val_steps, remaining_dis)

    logger.info(f"T2 train DIs: {train_dis}, val DIs: {val_dis}")

    t2_train_segs = extract_s1s2_full(t2_train, train_dis) if train_dis else []
    t2_val_segs = extract_s1s2_full(t2_val, val_dis) if val_dis else []

    # Combine T1 + T2
    train_beats = t1_train_beats + t2_train_segs
    val_beats = t1_val_beats + t2_val_segs

    n_t1_train = len(t1_train_beats)
    n_t2_train = len(t2_train_segs)
    n_t1_val = len(t1_val_beats)
    n_t2_val = len(t2_val_segs)
    logger.info(f"Train: {len(train_beats)} segments (T1:{n_t1_train} + T2:{n_t2_train})")
    logger.info(f"Val: {len(val_beats)} segments (T1:{n_t1_val} + T2:{n_t2_val})")

    # === Training ===
    optimizer = AdamW(trainable, lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
    run_dir = Path('runs/multi_bcl_t2_001')
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / 'log.jsonl'

    max_epochs = 200
    best_val_loss = float('inf')

    for epoch in range(max_epochs):
        t0 = time.time()

        # === Train ===
        node.train()
        random.shuffle(train_beats)
        train_loss_sum = 0.0
        n_train = 0

        for seg in train_beats:
            batch = beat_to_batch(seg, device)
            bcl = seg['_bcl']
            T_ms = float(bcl)
            n_pts = int(T_ms / 0.1) + 1
            t_eval = torch.linspace(0.0, T_ms, n_pts, dtype=torch.float64, device=device)

            optimizer.zero_grad()
            result = node_rollout(node, batch, phase_name='A1', device=device,
                                  t_eval_ms=t_eval, adjoint=False,
                                  method='dopri5', rtol=1e-3, atol=1e-3)
            result['loss'].backward()
            node.clear_v_trajectory()

            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            train_loss_sum += result['loss'].item()
            n_train += 1

        scheduler.step()
        train_loss = train_loss_sum / max(n_train, 1)

        # === Val ===
        node.eval()
        val_loss_sum = 0.0
        n_val = 0
        val_per_bcl = {}

        with torch.no_grad():
            for seg in val_beats:
                batch = beat_to_batch(seg, device)
                bcl = seg['_bcl']
                T_ms = float(bcl)
                n_pts = int(T_ms / 0.1) + 1
                t_eval = torch.linspace(0.0, T_ms, n_pts, dtype=torch.float64, device=device)

                result = node_rollout(node, batch, phase_name='A1', device=device,
                                      t_eval_ms=t_eval, adjoint=False,
                                      method='dopri5', rtol=1e-3, atol=1e-3)
                node.clear_v_trajectory()

                loss_val = result['loss'].item()
                val_loss_sum += loss_val
                n_val += 1
                val_per_bcl.setdefault(bcl, []).append(loss_val)

        val_loss = val_loss_sum / max(n_val, 1)
        elapsed = time.time() - t0

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            marker = ' *'
            torch.save({
                'stage1_state_dict': stage1.state_dict(),
                'epoch': epoch, 'val_loss': val_loss,
            }, run_dir / 'best.pt')

        # Per-BCL val breakdown
        bcl_str = ' '.join(f'{bcl}:{sum(v)/len(v):.3f}' for bcl, v in sorted(val_per_bcl.items()))

        log_entry = {
            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            'val_per_bcl': {str(k): sum(v)/len(v) for k, v in val_per_bcl.items()},
            'lr': optimizer.param_groups[0]['lr'], 'elapsed_s': round(elapsed, 1),
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        logger.info(
            f"Ep {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f} | "
            f"{bcl_str} | {elapsed:.0f}s{marker}"
        )

        if train_loss != train_loss:
            logger.error("NaN! Stopping.")
            break

    logger.info(f"Done. Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
