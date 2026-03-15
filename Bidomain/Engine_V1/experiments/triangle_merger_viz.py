#!/usr/bin/env python
"""
Triangle Merger Visualizations — 7 plots comparing 5pt vs Mehrstellen

Reads saved data from triangle_merger.py output.
All saved to Research/Q5_boundary_conduction_speedup/triangle_merger/.

Layout convention: top row = 5pt, bottom row = Mehrstellen 9pt.
"""

import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ============================================================
# Parameters (must match triangle_merger.py)
# ============================================================
NX, NY = 1001, 161
DX = 0.05
SAVE_EVERY = 25.0
THRESHOLD = -30.0

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', 'Research',
    'Q5_boundary_conduction_speedup', 'triangle_merger')


# ============================================================
# Data loading
# ============================================================
def load_config(name):
    """Load fronts, times, and activation times for a config."""
    fronts = torch.load(os.path.join(OUTPUT_DIR, f'{name}_fronts.pt'),
                        weights_only=True)
    act_time = torch.load(os.path.join(OUTPUT_DIR, f'{name}_act_time.pt'),
                          weights_only=True)
    with open(os.path.join(OUTPUT_DIR, f'{name}_times.json'), 'r') as f:
        times = json.load(f)

    # Load key Vm snapshots
    Vm = {}
    for t in [200, 400, 600, 800]:
        path = os.path.join(OUTPUT_DIR, f'{name}_Vm_{t}ms.pt')
        if os.path.exists(path):
            Vm[t] = torch.load(path, weights_only=True)

    return {'fronts': fronts, 'times': times, 'act_time': act_time, 'Vm': Vm}


def front_deviation(fronts, mono_fronts, times):
    """Compute wavefront deviation relative to monodomain flat reference.

    Returns deviation_cm as (n_snaps, NY) tensor.
    """
    dev = (fronts.float() - mono_fronts.float()) * DX
    return dev


def find_snap_idx(times, target):
    """Find snapshot index closest to target time."""
    for i, t in enumerate(times):
        if abs(t - target) < SAVE_EVERY / 2 + 0.1:
            return i
    return None


# ============================================================
# Plot 1: Wavefront evolution
# ============================================================
def plot_wavefront_evolution(mono, bi5, bi9):
    """2 rows x 4 cols: wavefront deviation at t=100,300,500,800ms."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    plot_times = [100, 300, 500, 800]
    y_cm = np.arange(NY) * DX

    for col, pt in enumerate(plot_times):
        for row, (bi, label) in enumerate([(bi5, '5pt'), (bi9, '9pt Mehrstellen')]):
            ax = axes[row, col]
            idx = find_snap_idx(bi['times'], pt)
            mono_idx = find_snap_idx(mono['times'], pt)
            if idx is None or mono_idx is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            dev = (bi['fronts'][idx].float() - mono['fronts'][mono_idx].float()) * DX
            ax.plot(y_cm, dev.numpy(), 'b-', linewidth=1.5)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.set_title(f'{label}, t={pt}ms')
            ax.set_xlabel('y (cm)')
            if col == 0:
                ax.set_ylabel('Deviation (cm)')
            ax.set_xlim(0, y_cm[-1])

    fig.suptitle('Wavefront Deviation vs Monodomain Flat Reference', fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'wavefront_evolution.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Plot 2: Vm heatmaps
# ============================================================
def plot_vm_heatmaps(bi5, bi9):
    """2 rows x 4 cols: Vm at t=200,400,600,800ms."""
    fig, axes = plt.subplots(2, 4, figsize=(24, 6))
    plot_times = [200, 400, 600, 800]

    for col, pt in enumerate(plot_times):
        for row, (bi, label) in enumerate([(bi5, '5pt'), (bi9, '9pt')]):
            ax = axes[row, col]
            if pt not in bi['Vm']:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            Vm = bi['Vm'][pt].numpy()
            # Find wavefront center for zoom
            front = bi['fronts'][find_snap_idx(bi['times'], pt)]
            front_x = front.float().mean().item() * DX

            # Zoom to wavefront ± 5cm
            x0 = max(0, int((front_x - 5) / DX))
            x1 = min(NX, int((front_x + 5) / DX))
            Vm_zoom = Vm[x0:x1, :].T  # transpose for imshow (y on y-axis)

            im = ax.imshow(Vm_zoom, aspect='auto', origin='lower',
                          extent=[x0 * DX, x1 * DX, 0, (NY - 1) * DX],
                          cmap='RdBu_r', vmin=-90, vmax=30)
            # Wavefront contour
            front_cm = front.float().numpy() * DX
            y_cm = np.arange(NY) * DX
            valid = (front_cm > x0 * DX) & (front_cm < x1 * DX)
            if valid.any():
                ax.plot(front_cm[valid], y_cm[valid], 'w-', linewidth=1.5)

            ax.set_title(f'{label}, t={pt}ms')
            if col == 0:
                ax.set_ylabel('y (cm)')
            ax.set_xlabel('x (cm)')

    fig.suptitle('Vm Heatmaps (zoomed to wavefront ± 5cm)', fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'Vm_heatmaps.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Plot 3: Lead vs time
# ============================================================
def plot_lead_vs_time(mono, bi5, bi9):
    """2 rows: edge/quarter lead vs time for 5pt and Mehrstellen."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    y_center = NY // 2
    y_edge = 1
    y_quarter = NY // 4

    for row, (bi, label) in enumerate([(bi5, '5pt'), (bi9, '9pt Mehrstellen')]):
        ax = axes[row]
        times = np.array(bi['times'])
        fronts = bi['fronts'].float().numpy()
        mono_fronts = mono['fronts'].float().numpy()

        # Only plot where wave has progressed
        center = fronts[:, y_center] * DX
        edge = fronts[:, y_edge] * DX
        quarter = fronts[:, y_quarter] * DX

        edge_lead = edge - center
        quarter_lead = quarter - center

        valid = center > 0.5  # at least 0.5cm of propagation
        if valid.any():
            ax.plot(times[valid], edge_lead[valid], 'r-', label='Edge lead', linewidth=1.5)
            ax.plot(times[valid], quarter_lead[valid], 'b--', label='Quarter lead', linewidth=1)

        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_ylabel('Lead distance (cm)')
        ax.set_title(f'{label}')
        ax.legend(loc='upper left')

    axes[1].set_xlabel('Time (ms)')
    fig.suptitle('Edge/Quarter Lead vs Center', fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'lead_vs_time.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Plot 4: Front range vs time
# ============================================================
def plot_front_range_vs_time(bi5, bi9):
    """Single plot: total deviation max(front)-min(front) vs time."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for bi, label, color in [(bi5, '5pt', 'blue'), (bi9, '9pt Mehrstellen', 'red')]:
        times = np.array(bi['times'])
        fronts = bi['fronts'].float().numpy()
        # Only use rows where wave has arrived
        ranges = []
        for i in range(len(times)):
            f = fronts[i]
            active = f > 0
            if active.sum() > 2:
                ranges.append((f[active].max() - f[active].min()) * DX)
            else:
                ranges.append(0.0)
        ranges = np.array(ranges)
        valid = ranges > 0
        if valid.any():
            ax.plot(times[valid], ranges[valid], color=color, label=label, linewidth=2)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Front range max-min (cm)')
    ax.set_title('Wavefront Deviation: Growth → Merger → Steady State')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'front_range_vs_time.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Plot 5: CV profile (steady state)
# ============================================================
def plot_cv_profile_steady(bi5, bi9):
    """2 rows: local CV as f(y) at late time (600-800ms)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    y_cm = np.arange(NY) * DX

    for row, (bi, label) in enumerate([(bi5, '5pt'), (bi9, '9pt Mehrstellen')]):
        ax = axes[row]
        times = np.array(bi['times'])
        fronts = bi['fronts'].float().numpy()

        # Find late-time indices (600-800ms)
        late_mask = (times >= 600) & (times <= 800)
        if late_mask.sum() < 2:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        late_idx = np.where(late_mask)[0]
        # CV from front advancement rate between consecutive snapshots
        cv_profiles = []
        for k in range(len(late_idx) - 1):
            i1, i2 = late_idx[k], late_idx[k + 1]
            dx_front = (fronts[i2] - fronts[i1]) * DX  # cm
            dt_front = times[i2] - times[i1]  # ms
            cv = dx_front / dt_front * 1000  # cm/s
            cv_profiles.append(cv)

        if cv_profiles:
            cv_mean = np.mean(cv_profiles, axis=0)
            ax.plot(y_cm, cv_mean, 'b-', linewidth=1.5)
            ax.axhline(np.median(cv_mean), color='gray', linestyle='--', alpha=0.5,
                       label=f'Median={np.median(cv_mean):.1f} cm/s')
            ax.set_ylabel('CV (cm/s)')
            ax.set_title(f'{label} — Late-time CV profile (600-800ms)')
            ax.legend()

    axes[1].set_xlabel('y (cm)')
    fig.suptitle('Local CV as Function of y (Transverse Position)', fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'cv_profile_steady.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Plot 6: Stencil comparison
# ============================================================
def plot_stencil_comparison(bi5, bi9):
    """Single plot: |front_5pt - front_9pt| at key times."""
    fig, ax = plt.subplots(figsize=(10, 6))
    y_cm = np.arange(NY) * DX
    colors = ['blue', 'green', 'orange', 'red']
    plot_times = [200, 400, 600, 800]

    for pt, color in zip(plot_times, colors):
        idx5 = find_snap_idx(bi5['times'], pt)
        idx9 = find_snap_idx(bi9['times'], pt)
        if idx5 is None or idx9 is None:
            continue
        diff = (bi5['fronts'][idx5].float() - bi9['fronts'][idx9].float()).abs() * DX
        ax.plot(y_cm, diff.numpy(), color=color, label=f't={pt}ms', linewidth=1.5)

    ax.set_xlabel('y (cm)')
    ax.set_ylabel('|front_5pt - front_9pt| (cm)')
    ax.set_title('Stencil Isotropy Difference: 5pt vs Mehrstellen')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'stencil_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Plot 7: Isochrone map
# ============================================================
def plot_isochrone_map(bi5, bi9):
    """2 rows: isochrone contours every 25ms."""
    fig, axes = plt.subplots(2, 1, figsize=(20, 8))
    x_cm = np.arange(NX) * DX
    y_cm = np.arange(NY) * DX

    for row, (bi, label) in enumerate([(bi5, '5pt'), (bi9, '9pt Mehrstellen')]):
        ax = axes[row]
        act = bi['act_time'].numpy()
        # Replace inf with NaN for contour plotting
        act_plot = np.where(np.isfinite(act), act, np.nan)

        levels = np.arange(0, 800, 25)
        cs = ax.contour(x_cm, y_cm, act_plot.T, levels=levels, cmap='viridis')
        ax.clabel(cs, inline=True, fontsize=6, fmt='%.0f')
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_title(f'Isochrone Map — {label} (contours every 25ms)')
        ax.set_aspect('equal')

    fig.suptitle('Activation Time Isochrones', fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'isochrone_map.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================
def main():
    print("Loading data...")
    mono = load_config('monodomain_mehrstellen')
    bi5 = load_config('bidomain_5pt')
    bi9 = load_config('bidomain_mehrstellen')
    print(f"  Loaded {len(mono['times'])} snapshots per config")

    print("\nGenerating plots...")
    plot_wavefront_evolution(mono, bi5, bi9)
    plot_vm_heatmaps(bi5, bi9)
    plot_lead_vs_time(mono, bi5, bi9)
    plot_front_range_vs_time(bi5, bi9)
    plot_cv_profile_steady(bi5, bi9)
    plot_stencil_comparison(bi5, bi9)
    plot_isochrone_map(bi5, bi9)

    # Verify all 7 PNGs exist and > 10KB
    print("\nVerifying output files...")
    expected = [
        'wavefront_evolution.png', 'Vm_heatmaps.png', 'lead_vs_time.png',
        'front_range_vs_time.png', 'cv_profile_steady.png',
        'stencil_comparison.png', 'isochrone_map.png',
    ]
    all_ok = True
    for fname in expected:
        path = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(path):
            print(f"  MISSING: {fname}")
            all_ok = False
        else:
            size = os.path.getsize(path)
            if size < 10240:
                print(f"  TOO SMALL: {fname} ({size} bytes)")
                all_ok = False
            else:
                print(f"  OK: {fname} ({size // 1024} KB)")

    if all_ok:
        print("\nAll 7 visualizations generated successfully.")
    else:
        print("\nSome visualizations missing or too small!")


if __name__ == '__main__':
    main()
