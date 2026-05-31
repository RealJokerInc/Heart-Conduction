#!/usr/bin/env python3
"""
A1 - F2: Showcase the 12 BCL/protocol tiers from the data-generation pipeline.

Each tier is a thematic batch of TTP06 simulations on different protocols
(steady-state pacing, S1S2 restitution, alternans, randomized parameters,
voltage-clamp ramps, drug-injected variants, etc.). The 12 tiers together
span ~608 GB of raw HDF5 traces on the project HDD.

For each tier, we load the FIRST protocol's voltage trace and plot a
representative window (~3-4 seconds) from the steady portion.

Layout: 4 rows x 3 cols = 12 panels.

Output: figures/A/A1_F2_tier_showcase.png
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DIR = '/mnt/HDD/norepinephrine/surrogate_data/raw'
OUT_DIR = os.path.join(REPO_ROOT, 'MonthlyReport', 'April', 'figures', 'A')

# Per-tier representative windows (start_idx, n_steps).  dt = 0.01 ms.
# Pick windows from the steady / interesting portion of each protocol.
WINDOWS = {
    1:  (200000, 400000),     # steady_bcl1000: 2-6 s
    2:  (50000,  400000),     # s1s2_di100: 0.5-4.5 s
    3:  (100000, 400000),     # alternans_bcl330: 1-5 s
    4:  (50000,  400000),     # random_seed0: 0.5-4.5 s
    5:  (100000, 350000),     # injected_biphasic
    6:  (5000,   25000),      # ramp_clamp: short tier (300 ms total)
    7:  (200000, 400000),     # steady_bcl1000
    8:  (200000, 400000),     # quiescent
    9:  (1000,   8000),       # quiescent (small tier, 100 ms)
    10: (100000, 350000),     # injected_boundary
    11: (100000, 400000),     # stitched
    12: (100000, 400000),     # alternans_bcl330
}

# Friendly labels per tier (theme + sample protocol)
TIER_LABELS = {
    1:  'T1 — Steady-state pacing\n(BCLs 400–2000 ms)',
    2:  'T2 — S1S2 restitution\n(diastolic intervals 75–800 ms)',
    3:  'T3 — Alternans regime\n(short-BCL pacing)',
    4:  'T4 — Randomized seeds\n(200 random parameter draws)',
    5:  'T5 — Injected biphasic\n(drug-like current pulses)',
    6:  'T6 — Voltage-clamp ramps\n(I-V characterization)',
    7:  'T7 — Steady-state (canonical)\n(BCL 1000 ms)',
    8:  'T8 — Quiescent\n(no stimulus, long rest)',
    9:  'T9 — Quiescent (short)\n(100 ms rest)',
    10: 'T10 — Injected boundary\n(boundary-current perturbations)',
    11: 'T11 — Stitched protocols\n(50 sequential patterns)',
    12: 'T12 — Alternans + restitution mix\n(21 protocols, generalization set)',
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(4, 3, figsize=(13, 10), dpi=300, sharey=True)
    axes = axes.flatten()

    for tier in range(1, 13):
        ax = axes[tier - 1]
        path = os.path.join(RAW_DIR, f'tier{tier:02d}.h5')
        if not os.path.exists(path):
            ax.text(0.5, 0.5, f'tier{tier:02d}.h5\nnot found',
                    transform=ax.transAxes, ha='center', va='center')
            continue

        with h5py.File(path, 'r') as f:
            protos = list(f.keys())
            proto_name = protos[0]
            attrs = dict(f[proto_name].attrs)
            n_total = f[proto_name]['data'].shape[0]
            i0, n = WINDOWS[tier]
            i1 = min(i0 + n, n_total)
            V = f[proto_name]['data'][i0:i1, 0]
            dt = float(attrs.get('dt_default', 0.01))

        t = np.arange(len(V)) * dt / 1000.0  # convert ms -> s

        # Color shift across tiers for visual variety
        c = plt.cm.viridis(0.05 + 0.85 * (tier - 1) / 11)
        ax.plot(t, V, color=c, linewidth=0.8)
        ax.set_ylim(-100, 60)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)

        # Title block per panel
        ax.set_title(TIER_LABELS[tier], fontsize=8.5, loc='left',
                     fontweight='bold', color='#1a237e', pad=4)

        # Annotation: protocol name + n_protocols in tier
        ax.text(
            0.98, 0.04,
            f'{proto_name}\n({len(protos)} protocols in tier)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=6.5, color='#37474f',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='lightgray', alpha=0.85),
        )

        # X label only on bottom row
        if tier > 9:
            ax.set_xlabel('time (s)', fontsize=8)
        # Y label only on left column
        if tier in (1, 4, 7, 10):
            ax.set_ylabel('V (mV)', fontsize=8)

    fig.suptitle(
        'Surrogate-pipeline data generation — 12 protocol tiers (T1–T12, 608 GB raw)',
        fontsize=13, fontweight='bold', y=0.995, color='#1a237e',
    )
    fig.text(
        0.5, -0.005,
        'BatchGenerator + torch.compile produced 47,000× speedup over sequential single-cell '
        'simulation; each tier captures a thematically distinct slice of the TTP06 protocol space '
        '(pacing, restitution, alternans, drug-like injections, voltage clamps, randomized parameters, quiescence).',
        ha='center', fontsize=9, style='italic', color='#37474f',
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])

    out_png = os.path.join(OUT_DIR, 'A1_F2_tier_showcase.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == '__main__':
    main()
