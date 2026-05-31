#!/usr/bin/env python3
"""
A1 - F3: Training curves (train vs val loss) for v3 and v4 surrogate stages.

v3 was the Layer-0 + Neural ODE architecture (1,444 trainable params after
the NODE pivot). v4 expanded the StateRateMLP capacity to 7,891 params.

Stages logged:
- v3 / multi_bcl_001     : T1 multi-BCL fresh start, 74 epochs
- v3 / multi_bcl_002     : T1 multi-BCL parity oracle (warm-started from 001), 8 epochs
- v3 / multi_bcl_t2_001  : T2 multi-BCL extension, 112 epochs
- v4 / v4_A1             : v4 architecture, A1 stage (ionic-only loss), 30 epochs

Earlier stages (single_ap_001, single_ap_conc_001) had train-only loss
and are not shown here.

Output: figures/A/A1_F3_training_curves.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ARCHIVE = os.path.join(REPO_ROOT, 'archive', 'runs_legacy')
OUT_DIR = os.path.join(REPO_ROOT, 'MonthlyReport', 'April', 'figures', 'A')

# ordered: 3 v3 stages + 1 v4 stage  (rows = model version)
STAGES = [
    {
        'log': os.path.join(ARCHIVE, 'multi_bcl_001', 'log.jsonl'),
        'title': 'v3 — multi_bcl_001\n(T1 multi-BCL, fresh start)',
        'note': '1,444 params · NODE / dopri8',
        'color': '#1f77b4',
        'group': 'v3',
    },
    {
        'log': os.path.join(ARCHIVE, 'multi_bcl_002', 'log.jsonl'),
        'title': 'v3 — multi_bcl_002 (parity oracle)\n(T1, warm-start from 001)',
        'note': 'best val = 0.00838 at epoch 6',
        'color': '#1f77b4',
        'group': 'v3',
    },
    {
        'log': os.path.join(ARCHIVE, 'multi_bcl_t2_001', 'log.jsonl'),
        'title': 'v3 — multi_bcl_t2_001\n(T2 restitution extension)',
        'note': 'generalization to S1S2 protocol set',
        'color': '#1f77b4',
        'group': 'v3',
    },
    {
        'log': os.path.join(ARCHIVE, 'v4_A1', 'log_A1.jsonl'),
        'title': 'v4 — v4_A1\n(7,891 params, ionic-only A1 stage)',
        'note': 'capacity test on T1 — converged to ≈ same val',
        'color': '#d62728',
        'group': 'v4',
    },
]


def load(log_path):
    epochs, train, val = [], [], []
    with open(log_path) as f:
        for line in f:
            d = json.loads(line)
            epochs.append(d['epoch'])
            train.append(d.get('train_loss'))
            val.append(d.get('val_loss'))
    return np.array(epochs), np.array(train, dtype=float), np.array(val, dtype=float)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), dpi=300)
    axes = axes.flatten()

    for ax, stage in zip(axes, STAGES):
        epoch, tr, va = load(stage['log'])
        ax.plot(epoch, tr, label='train',
                color=stage['color'], linewidth=1.6)
        ax.plot(epoch, va, label='val',
                color=stage['color'], linewidth=1.6,
                linestyle='--', alpha=0.85)

        ax.set_yscale('log')
        ax.set_xlabel('epoch', fontsize=9)
        ax.set_ylabel('loss (log scale)', fontsize=9)
        ax.set_title(stage['title'], fontsize=10, loc='left',
                     color=stage['color'], fontweight='bold', pad=4)
        ax.grid(True, alpha=0.25, which='both')
        ax.legend(loc='upper right', fontsize=8.5, framealpha=0.95)
        ax.tick_params(labelsize=8)

        # Annotate final / best val
        best_val = float(np.nanmin(va))
        last_train = float(tr[-1])
        last_val = float(va[-1])
        ax.text(
            0.02, 0.04,
            f"final train={last_train:.4g}\nfinal val={last_val:.4g}\n"
            f"best val={best_val:.4g}\n{stage['note']}",
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=7.6, color='#37474f',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='lightgray', alpha=0.92),
        )

    # Group banners
    fig.suptitle(
        'Surrogate training curves — train vs val loss across stages',
        fontsize=13, fontweight='bold', y=0.995, color='#1a237e',
    )
    fig.text(
        0.5, -0.01,
        'Top row: v3 architecture (1,444 trainable params) — fresh T1 fit, parity-oracle continuation, T2 extension.  '
        'Bottom-right: v4 architecture (7,891 params) capacity test on the same A1 ionic-state target.  '
        'v4 reached comparable validation loss to v3 — extra capacity did not unlock new fit quality on T1.',
        ha='center', fontsize=8.7, style='italic', color='#37474f',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = os.path.join(OUT_DIR, 'A1_F3_training_curves.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == '__main__':
    main()
