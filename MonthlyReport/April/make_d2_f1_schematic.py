#!/usr/bin/env python3
"""
D2 - F1: Optimizer V1 schematic — qLogNEHVI multi-objective Bayesian optimization.

Cartoon flowchart explaining the BO loop:

    Forward (top):  theta -> simulator -> V(t) -> biomarkers -> Pareto residual
    Constraints sidegate:    V_rest range, V_peak max  (reject infeasible)
    Targets sidebar:         APD90 ~= 200 ms, dV/dt ~= 250 V/s (adult-like)
    BO loop (bottom):  Pareto residual -> GP surrogate (per obj.)
                       -> qLogNEHVI acquisition -> next theta batch (loop)

Design choices match Optimizer V1 source:
- tuner/cell_fitter.py:17,175  qLogNoisyExpectedHypervolumeImprovement
- tuner/cell_fitter.py:182     q_batch = min(4, ...)
- tuner/cell_fitter.py:138     ref_point = -500 per objective
- run_mhas13.py:34             tier=2 -> 10 params

Output: figures/D2_F1_optimizer_schematic.png
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(REPO_ROOT, 'MonthlyReport', 'April', 'figures')


def box(ax, x, y, w, h, text, fc='#e8eaf6', ec='#1a237e',
        textcolor=None, fontsize=9, fontweight='normal'):
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle='round,pad=0.05',
        linewidth=1.5, facecolor=fc, edgecolor=ec, zorder=2,
    )
    ax.add_patch(rect)
    if textcolor is None:
        textcolor = ec
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, color=textcolor,
            zorder=3)


def arrow(ax, x1, y1, x2, y2, color='#37474f', label=None,
          lstyle='-', curve=0.0, lw=1.6, label_offset=(0.0, 0.18),
          label_fontsize=8.5, label_color=None):
    style = f'arc3,rad={curve}' if curve else 'arc3'
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>', mutation_scale=14,
        connectionstyle=style,
        linewidth=lw, color=color, linestyle=lstyle, zorder=1,
    )
    ax.add_patch(a)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if label_color is None:
            label_color = color
        ax.text(mx + label_offset[0], my + label_offset[1], label,
                ha='center', fontsize=label_fontsize, color=label_color,
                style='italic', zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(13, 7), dpi=300)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    # ----- Top row: forward simulation -----
    THETA_FC, THETA_EC = '#fff3e0', '#e65100'
    SIM_FC,   SIM_EC   = '#e3f2fd', '#0d47a1'
    BIO_FC,   BIO_EC   = '#e8eaf6', '#1a237e'
    LOSS_FC,  LOSS_EC  = '#fce4ec', '#880e4f'

    # 1: theta batch
    box(ax, 1.4, 5.7, 2.0, 1.1,
        r'$\theta \in \mathbb{R}^{10}$' + '\nbatch q ≤ 4\n[g_Na, g_CaL, g_Kr,\ng_Ks, g_K1, g_to,\nkNaCa, PNaK, g_pCa, VmaxUp]',
        fc=THETA_FC, ec=THETA_EC, fontsize=7.2, fontweight='bold')

    # 2: simulator
    box(ax, 4.3, 5.7, 2.2, 1.1,
        'Single-cell ODE\n(PHAS13 / MHAS13)\n8 beats @ 1 Hz\nRush-Larsen, dt = 0.05 ms',
        fc=SIM_FC, ec=SIM_EC, fontsize=7.7, fontweight='bold')

    # 3: V(t)
    box(ax, 7.0, 5.7, 1.5, 1.1,
        'V(t)\nlast 3 beats',
        fc=SIM_FC, ec=SIM_EC, fontsize=8.5)

    # 4: biomarker extraction
    box(ax, 9.6, 5.7, 2.2, 1.1,
        'Biomarker extract\n' + r'$\mathrm{APD}_{90}$, $|dV/dt|_{\max}$' + '\n' + r'$V_{\mathrm{rest}}$, $V_{\mathrm{peak}}$',
        fc=BIO_FC, ec=BIO_EC, fontsize=8, fontweight='bold')

    # 5: targets sidebar (top-right)
    box(ax, 12.1, 5.7, 1.7, 1.1,
        'Targets\n(adult-like)\n' + r'$\mathrm{APD}_{90}$ ≈ 280 ms' + '\n' + r'$|dV/dt| \approx 200$ V/s',
        fc='#e8f5e9', ec='#2e7d32', fontsize=7.7, fontweight='bold')

    # ----- Constraint gate -----
    box(ax, 9.6, 3.85, 2.4, 1.0,
        'Hard constraints\n' + r'$V_{\mathrm{rest}} \in [-90,-75]$ mV' + '\n' + r'$V_{\mathrm{peak}} \leq +50$ mV' + '\n(else → −500 penalty)',
        fc='#fff8e1', ec='#ff6f00', fontsize=7.5)

    # ----- Loss / Pareto -----
    box(ax, 9.6, 2.1, 2.4, 1.0,
        'Pareto residual\n' + r'$y_i = -|m_i - m^*_i|$ for $i \in \{\mathrm{APD}, dV/dt\}$',
        fc=LOSS_FC, ec=LOSS_EC, fontsize=8, fontweight='bold')

    # ----- GP surrogate -----
    box(ax, 5.8, 2.1, 2.2, 1.0,
        'GP surrogate\nper objective\n(BoTorch SingleTaskGP)',
        fc='#ede7f6', ec='#311b92', fontsize=8, fontweight='bold')

    # ----- qLogNEHVI acquisition -----
    box(ax, 2.1, 2.1, 2.4, 1.0,
        'qLogNEHVI acquisition\n' + r'$\arg\max_{\theta_{1..q}} \mathbb{E}[\Delta\mathrm{HV}]$' + '\nref pt $= (-500,-500)$',
        fc=THETA_FC, ec=THETA_EC, fontsize=7.7, fontweight='bold')

    # ----- Arrows -----
    # Forward chain
    arrow(ax, 2.4, 5.7, 3.2, 5.7)
    arrow(ax, 5.4, 5.7, 6.25, 5.7)
    arrow(ax, 7.75, 5.7, 8.5, 5.7)
    # to constraint gate
    arrow(ax, 9.6, 5.15, 9.6, 4.35, label='biomarkers')
    # constraint -> loss (feasible path)
    arrow(ax, 9.6, 3.35, 9.6, 2.6, label='feasible',
          label_color='#1b5e20')
    # targets dashed -> residual (clean curved path on the far right)
    arrow(ax, 12.1, 5.15, 10.85, 2.1, color='#2e7d32', lstyle='--',
          label='', curve=-0.35, lw=1.3)
    ax.text(12.4, 3.6, 'targets', fontsize=8.5, color='#2e7d32',
            style='italic', ha='center', fontweight='bold')
    # BO loop, right -> left
    arrow(ax, 8.4, 2.1, 6.9, 2.1)
    arrow(ax, 4.7, 2.1, 3.3, 2.1)
    # qLogNEHVI loops back to theta (curved up-left)
    arrow(ax, 2.1, 2.6, 1.4, 5.15,
          color='#e65100', label='next batch', curve=-0.25, lw=2.0,
          label_color='#e65100', label_fontsize=9, label_offset=(-0.05, 0.0))

    # ----- Title + caption -----
    fig.suptitle(
        'Optimizer V1 — qLogNEHVI multi-objective Bayesian optimization for ionic-model fitting',
        fontsize=12.5, y=0.965, fontweight='bold', color='#1a237e',
    )
    fig.text(
        0.5, 0.045,
        'Each iteration: per-objective GP surrogates fit on observed '
        r'$(\theta, y)$' + ' pairs; qLogNEHVI then selects up to 4 new '
        r'$\theta$' + ' vectors maximizing expected hypervolume improvement on the Pareto front.  '
        'Infeasible candidates (V_rest / V_peak out of range) receive a uniform −500 penalty rather than poisoning the GP.',
        ha='center', fontsize=8.7, style='italic', color='#37474f',
    )

    out_png = os.path.join(OUT_DIR, 'D2_F1_optimizer_schematic.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == '__main__':
    main()
