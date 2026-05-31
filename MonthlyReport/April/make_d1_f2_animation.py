#!/usr/bin/env python3
"""
D1 — F2 (animated): Quiescence comparison as an animated GIF.

Loads traces from D1_F2_traces.npz (produced by make_d1_f2_quiescence.py).
Two cells side-by-side, filled circles colored by V (coolwarm colormap).
PHAS13 visibly pulses on each spontaneous AP; MHAS13 stays at V_rest.

Output:
    MonthlyReport/April/figures/D1_F2_quiescence.gif

PPTX-only deliverable. PDF readers will see the still cover frame only;
spec requires explicit "video" labeling on the slide.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.cm import coolwarm
from matplotlib.animation import FuncAnimation, PillowWriter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(REPO_ROOT, 'MonthlyReport', 'April', 'figures')


def main():
    npz_path = os.path.join(OUT_DIR, 'D1_F2_traces.npz')
    data = np.load(npz_path)
    t_p = data['phas13_t']  # ms
    V_p = data['phas13_V']
    t_m = data['mhas13_t']
    V_m = data['mhas13_V']

    # GIF parameters
    sim_window_ms = 10000.0     # animate full 10 s
    target_n_frames = 200       # ~ 30 fps × 6.7 s GIF
    fps = 30

    # Subsample to target_n_frames frames spanning sim_window_ms
    # Both traces start at t=0; t_m might start later (saved-window).
    # Re-base both onto a common time grid by subsampling from their own arrays.

    # PHAS13 — straightforward: t_p starts at 0
    p_mask = t_p <= sim_window_ms
    t_p = t_p[p_mask]
    V_p = V_p[p_mask]
    p_stride = max(1, len(t_p) // target_n_frames)
    t_p_anim = t_p[::p_stride]
    V_p_anim = V_p[::p_stride]

    # MHAS13 — re-base to start at 0 if it doesn't already
    t_m_rebased = t_m - t_m[0]
    m_mask = t_m_rebased <= sim_window_ms
    t_m_anim_full = t_m_rebased[m_mask]
    V_m_anim_full = V_m[m_mask]
    m_stride = max(1, len(t_m_anim_full) // target_n_frames)
    t_m_anim = t_m_anim_full[::m_stride]
    V_m_anim = V_m_anim_full[::m_stride]

    # Match frame counts (use min)
    n_frames = min(len(t_p_anim), len(t_m_anim))
    t_p_anim = t_p_anim[:n_frames]
    V_p_anim = V_p_anim[:n_frames]
    t_m_anim = t_m_anim[:n_frames]
    V_m_anim = V_m_anim[:n_frames]

    print(f"GIF: {n_frames} frames @ {fps} fps "
          f"({n_frames/fps:.1f} s real-time, {sim_window_ms/1000.0:.0f} s sim)")

    # ----- Build figure -----
    norm = Normalize(vmin=-95.0, vmax=50.0)
    cmap = coolwarm

    fig = plt.figure(figsize=(8.0, 6.0), dpi=120)
    gs = fig.add_gridspec(
        3, 2, height_ratios=[6, 1, 0.4],
        left=0.06, right=0.94, top=0.92, bottom=0.10,
        wspace=0.15, hspace=0.30,
    )

    ax_p = fig.add_subplot(gs[0, 0]); ax_p.set_aspect('equal')
    ax_m = fig.add_subplot(gs[0, 1]); ax_m.set_aspect('equal')
    ax_t_p = fig.add_subplot(gs[1, 0])
    ax_t_m = fig.add_subplot(gs[1, 1])
    ax_cb = fig.add_subplot(gs[2, :])

    # Cell axes — set up
    for ax, title, color in [
        (ax_p, 'PHAS13 (immature hiPSC-CM)', '#2ca02c'),
        (ax_m, 'MHAS13 (matured + fitted)',  '#d62728'),
    ]:
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11, color=color, pad=10)
        for s in ax.spines.values():
            s.set_visible(False)

    # Cells
    cell_p = Circle((0, 0), 0.85, facecolor=cmap(norm(V_p_anim[0])),
                    edgecolor='black', linewidth=1.5)
    cell_m = Circle((0, 0), 0.85, facecolor=cmap(norm(V_m_anim[0])),
                    edgecolor='black', linewidth=1.5)
    ax_p.add_patch(cell_p)
    ax_m.add_patch(cell_m)

    # V text overlay on each cell
    txt_p = ax_p.text(0, 0, '', ha='center', va='center',
                       fontsize=14, fontweight='bold', color='white')
    txt_m = ax_m.text(0, 0, '', ha='center', va='center',
                       fontsize=14, fontweight='bold', color='white')

    # AP counters
    ap_count_p = [0]
    ap_count_m = [0]
    cnt_p = ax_p.text(0, -1.1, '', ha='center', va='center',
                       fontsize=10, color='#2ca02c', fontweight='bold')
    cnt_m = ax_m.text(0, -1.1, '', ha='center', va='center',
                       fontsize=10, color='#d62728', fontweight='bold')

    # Voltage trace strips (show full trace, vertical line marks current time)
    ax_t_p.plot(t_p_anim / 1000.0, V_p_anim, color='#2ca02c', linewidth=1.0)
    ax_t_m.plot(t_m_anim / 1000.0, V_m_anim, color='#d62728', linewidth=1.0)
    for ax in (ax_t_p, ax_t_m):
        ax.axhline(-40.0, color='gray', linestyle=':', linewidth=0.6, alpha=0.6)
        ax.set_ylim(-95, 50); ax.grid(alpha=0.25)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.tick_params(labelsize=8)
    ax_t_p.set_ylabel('V (mV)', fontsize=8)

    line_p = ax_t_p.axvline(0, color='black', linewidth=1.3, alpha=0.7)
    line_m = ax_t_m.axvline(0, color='black', linewidth=1.3, alpha=0.7)

    # Title with running time
    title = fig.suptitle('', fontsize=12, y=0.985)

    # Colorbar at bottom
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb, orientation='horizontal')
    cb.set_label('Membrane potential V (mV)', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    def update(i):
        Vp = V_p_anim[i]
        Vm = V_m_anim[i]
        t_now = t_p_anim[i] / 1000.0  # seconds

        cell_p.set_facecolor(cmap(norm(Vp)))
        cell_m.set_facecolor(cmap(norm(Vm)))
        txt_p.set_text(f'{Vp:+.0f} mV')
        txt_m.set_text(f'{Vm:+.0f} mV')

        # AP detection (rising edge above -40 mV)
        if i > 0:
            if V_p_anim[i - 1] <= -40.0 < Vp:
                ap_count_p[0] += 1
            if V_m_anim[i - 1] <= -40.0 < Vm:
                ap_count_m[0] += 1
        cnt_p.set_text(f'spontaneous APs: {ap_count_p[0]}')
        cnt_m.set_text(f'spontaneous APs: {ap_count_m[0]}')

        line_p.set_xdata([t_now, t_now])
        line_m.set_xdata([t_now, t_now])

        title.set_text(
            f'Quiescence comparison — free-run, no stimulus     t = {t_now:.2f} s'
        )

        return cell_p, cell_m, txt_p, txt_m, cnt_p, cnt_m, line_p, line_m, title

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000.0/fps,
                        blit=False)

    out_gif = os.path.join(OUT_DIR, 'D1_F2_quiescence.gif')
    print(f"Encoding GIF... ({out_gif})")
    ani.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved: {out_gif}")


if __name__ == '__main__':
    main()
