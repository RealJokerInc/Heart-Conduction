#!/usr/bin/env python3
"""
A1 - F1: Surrogate vs classical TTP06 throughput (CPU + GPU paired bars).

The honest negative result for the ionic-replacement direction:
- GPU @ N=10,000:  classical TTP06 is 8x FASTER (surrogate 0.124x)
- CPU @ N=10-1000: surrogate is 3-7x faster (matters less; bidomain runs on GPU)

The takeaway: ionic step was the wrong bottleneck. KNOWLEDGE Sec.1 logged
that 94% of bidomain wall-time lives in the elliptic solve, not the ionic
step. Effort is now redirected to a learned elliptic sub-operator.

Sources:
- Surrogate/benchmarks/results/ttp06_vs_surrogate_gpu.json   (full sweep)
- Surrogate/benchmarks/results/ttp06_vs_surrogate_cpu.log    (parsed inline)

Output: figures/A1_F1_throughput.png
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(REPO_ROOT, 'MonthlyReport', 'April', 'figures')
SURR_BENCH = os.path.join(REPO_ROOT, 'Surrogate', 'benchmarks', 'results')


def load_gpu():
    with open(os.path.join(SURR_BENCH, 'ttp06_vs_surrogate_gpu.json')) as f:
        d = json.load(f)
    n = [r['n_cells'] for r in d['rows']]
    ttp = [r['ttp06_cell_steps_per_sec'] for r in d['rows']]
    sur = [r['surrogate_cell_steps_per_sec'] for r in d['rows']]
    spd = [r['speedup_surr_over_ttp06'] for r in d['rows']]
    return n, ttp, sur, spd


def load_cpu():
    # Parse the truncated CPU log inline (run terminated at N=1000).
    rows = [
        (10,    16257.0,    119870.0),
        (100,   158871.0,   837741.0),
        (1000,  1170867.0,  3934451.0),
    ]
    n   = [r[0] for r in rows]
    ttp = [r[1] for r in rows]
    sur = [r[2] for r in rows]
    spd = [s / t for t, s in zip(ttp, sur)]
    return n, ttp, sur, spd


def main():
    n_g, ttp_g, sur_g, spd_g = load_gpu()
    n_c, ttp_c, sur_c, spd_c = load_cpu()

    fig, (axc, axg) = plt.subplots(1, 2, figsize=(13, 5.4), dpi=300,
                                   sharey=True)

    TTP_COLOR = '#1f77b4'
    SUR_COLOR = '#ff7f0e'

    # ----- Left panel: CPU -----
    x_c = np.arange(len(n_c))
    w = 0.36
    axc.bar(x_c - w/2, ttp_c, w, color=TTP_COLOR, label='Classical TTP06',
            edgecolor='black', linewidth=0.5)
    axc.bar(x_c + w/2, sur_c, w, color=SUR_COLOR, label='Surrogate v4 Euler',
            edgecolor='black', linewidth=0.5)
    axc.set_yscale('log')
    axc.set_xticks(x_c)
    axc.set_xticklabels([f'N={n:,}' for n in n_c])
    axc.set_ylabel('Throughput (cell-steps / sec, log scale)', fontsize=10)
    axc.set_title('CPU (no compile)\nSurrogate WINS — but not where it matters',
                  fontsize=11, color='#2e7d32', loc='left', fontweight='bold')
    axc.grid(True, axis='y', alpha=0.25, which='both')
    axc.legend(loc='upper left', fontsize=9, framealpha=0.95)

    # speedup callouts
    for xi, s in zip(x_c, spd_c):
        ymax = max(ttp_c[xi], sur_c[xi])
        axc.text(xi, ymax * 1.5, f'{s:.1f}×',
                 ha='center', fontsize=9.5, color='#2e7d32', fontweight='bold')

    # ----- Right panel: GPU -----
    x_g = np.arange(len(n_g))
    axg.bar(x_g - w/2, ttp_g, w, color=TTP_COLOR, label='Classical TTP06',
            edgecolor='black', linewidth=0.5)
    axg.bar(x_g + w/2, sur_g, w, color=SUR_COLOR, label='Surrogate v4 Euler',
            edgecolor='black', linewidth=0.5)
    axg.set_yscale('log')
    axg.set_xticks(x_g)
    axg.set_xticklabels([f'N={n:,}' for n in n_g])
    axg.set_title('GPU (torch.compile, RTX PRO 4500)\n'
                  'Classical TTP06 WINS at tissue scale',
                  fontsize=11, color='#c62828', loc='left', fontweight='bold')
    axg.grid(True, axis='y', alpha=0.25, which='both')
    axg.legend(loc='upper left', fontsize=9, framealpha=0.95)

    for xi, s in zip(x_g, spd_g):
        ymax = max(ttp_g[xi], sur_g[xi])
        axg.text(xi, ymax * 1.5, f'{s:.2f}×',
                 ha='center', fontsize=9.5,
                 color='#c62828' if s < 1 else '#2e7d32',
                 fontweight='bold')

    # Highlight the headline N=10000 GPU result
    axg.annotate(
        'classical 8× faster\nat tissue scale',
        xy=(x_g[-1] - w/2, ttp_g[-1]),
        xytext=(x_g[-1] - 1.6, ttp_g[-1] * 0.35),
        fontsize=10, color='#c62828', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#c62828', lw=1.5),
        ha='center',
    )

    # Suptitle + caption
    fig.suptitle(
        'Ionic-step throughput: surrogate Euler vs classical TTP06\n'
        '(30k steps, dt=0.01 ms; speedup labels = surrogate / TTP06)',
        fontsize=12.5, y=1.0, fontweight='bold', color='#1a237e',
    )
    fig.text(
        0.5, -0.02,
        'Surrogate v4 Euler: 7,552 inference parameters.  '
        'Bidomain V1 runs on GPU; the surrogate loses 8× there. '
        'Ionic step is also only ~6% of bidomain wall-time — 94% lives in the elliptic solve. '
        'The ionic step was never the bottleneck.',
        ha='center', fontsize=8.7, style='italic', color='#37474f',
    )

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, 'A1_F1_throughput.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == '__main__':
    main()
