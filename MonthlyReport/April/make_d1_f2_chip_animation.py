#!/usr/bin/env python3
"""
D1 - F2 (chip animation): COUPLED monodomain 3x3, PHAS13 vs fitted MHAS13.

Two 3x3 wells, side-by-side. Each chip is a real 3x3 monodomain mesh:
batch_step over M=9 cells with diffusion injected as I_stim = -D * Lap(V)
(5-point stencil, zero-flux Neumann BC on all four edges).

Each well starts from a stochastically perturbed initial state (multiplicative
noise on gating + Ca buffers, additive +/- mV jitter on V) so the wells are NOT
identical at t=0. Coupling then drives them through diffusion: PHAS13 wells
synchronize their spontaneous beats; MHAS13 wells decay back to V_rest together.

Center well (row 1, col 1, batch index 4) is highlighted with a red ring; the
voltage traces below each chip plot ONLY that center cell.

Workflow (per user request):
    1. Render MP4 via FFMpegWriter (libx264).
    2. Convert MP4 -> GIF via ffmpeg subprocess (palettegen + paletteuse).

Outputs:
    figures/D1_F2_chip.mp4
    figures/D1_F2_chip.gif
    figures/D1_F2_chip_coupled_traces.npz   (raw 9-cell V traces)
"""

import os
import sys
import json
import time
import subprocess
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.cm import coolwarm
from matplotlib.animation import FuncAnimation, FFMpegWriter

import imageio_ffmpeg
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
matplotlib.rcParams['animation.ffmpeg_path'] = FFMPEG_EXE

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'Optimizer', 'V1'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'Monodomain', 'Engine_V5.4'))

from tuner.batch_ionic import batch_step, build_conductance_tensor
from tuner.config import TuningConfig
from cardiac_sim.ionic.phas13.parameters import (
    get_initial_state as _get_initial_state, V_REST,
)

OUT_DIR = os.path.join(REPO_ROOT, 'MonthlyReport', 'April', 'figures')

N_GRID = 3
M_CELLS = N_GRID * N_GRID
CENTER_IDX = 4  # (row 1, col 1) -> 1*3+1 = 4


def laplacian_2d_neumann(V_2d, dx):
    """5-point Laplacian with zero-flux Neumann BC (mirror padding)."""
    Vp = torch.zeros((V_2d.shape[0] + 2, V_2d.shape[1] + 2),
                      dtype=V_2d.dtype, device=V_2d.device)
    Vp[1:-1, 1:-1] = V_2d
    Vp[0,    1:-1] = V_2d[0,  :]
    Vp[-1,   1:-1] = V_2d[-1, :]
    Vp[1:-1, 0]    = V_2d[:,  0]
    Vp[1:-1, -1]   = V_2d[:, -1]
    return (Vp[:-2, 1:-1] + Vp[2:, 1:-1]
            + Vp[1:-1, :-2] + Vp[1:-1, 2:]
            - 4.0 * V_2d) / (dx ** 2)


def run_coupled(theta_array, ionic_model,
                t_total_ms=10000.0, dt=0.05,
                dx=0.025, D=0.001,
                seed=42, state_perturb=0.02, v_perturb=3.0,
                save_every=10):
    """Run coupled 3x3 monodomain with stochastic initial conditions.

    Returns
    -------
    t : (T_save,) ms
    V_hist : (T_save, 9) mV — column index = batch index = row*3 + col
    """
    torch.manual_seed(seed)
    dtype = torch.float64
    device = 'cpu'

    config = TuningConfig(
        ionic_model=ionic_model, tier=2, device=device,
        dt=0.02, dt_cell=dt, dx_cm=dx, cable_length_cm=1.5,
        n_beats=1, pacing_cl=1000.0,
        stim_amplitude=-40.0, stim_duration=2.0,
    )
    theta_t = torch.tensor(theta_array, dtype=dtype, device=device)
    theta_batch = theta_t.unsqueeze(0).expand(M_CELLS, -1).contiguous()
    cond = build_conductance_tensor(theta_batch, config.tier, dtype,
                                     device, ionic_model=ionic_model)

    # Initial state: shared baseline + stochastic per-cell multiplicative jitter
    init_state = _get_initial_state(device=torch.device(device), dtype=dtype)
    states = init_state.unsqueeze(0).expand(M_CELLS, -1).clone()
    noise = state_perturb * torch.randn_like(states)
    states = states * (1.0 + noise)
    states.clamp_(min=1e-8)  # gates / concentrations stay non-negative

    v_rest = V_REST  # ~ -83.7
    V = torch.full((M_CELLS,), v_rest, dtype=dtype, device=device)
    V = V + v_perturb * torch.randn(M_CELLS, dtype=dtype, device=device)

    n_steps = int(t_total_ms / dt)
    n_save = n_steps // save_every + 1
    V_hist = np.zeros((n_save, M_CELLS))
    t_arr = np.zeros(n_save)

    V_hist[0] = V.cpu().numpy()
    t_arr[0] = 0.0
    save_idx = 1

    for step_i in range(n_steps):
        V_2d = V.reshape(N_GRID, N_GRID)
        Lap = laplacian_2d_neumann(V_2d, dx).flatten()
        I_diff = -D * Lap                              # (M,)
        V, states = batch_step(V, states, dt, cond, I_diff,
                                ionic_model=ionic_model)

        if (step_i + 1) % save_every == 0 and save_idx < n_save:
            V_hist[save_idx] = V.cpu().numpy()
            t_arr[save_idx] = (step_i + 1) * dt
            save_idx += 1

        if not torch.isfinite(V).all():
            print(f"  ! diverged at step {step_i}")
            break

    return t_arr[:save_idx], V_hist[:save_idx]


def count_aps(V_center, threshold=-40.0):
    above = V_center > threshold
    edges = np.diff(above.astype(int))
    return int(np.sum(edges == 1))


def main():
    theta_path = os.path.join(OUT_DIR, 'D1_mhas13_theta.json')
    with open(theta_path) as f:
        meta = json.load(f)
    mhas13_theta = meta['theta_array']
    n_tier_params = len(mhas13_theta)
    phas13_theta = [1.0] * n_tier_params

    print("=" * 70)
    print("D1 F2 — COUPLED 3x3 chip (stochastic ICs)")
    print("=" * 70)
    print(f"PHAS13 theta = baseline (all-ones, {n_tier_params} params)")
    print(f"MHAS13 theta = fitted (g_K1={mhas13_theta[4]:.3f}, "
          f"g_Kr={mhas13_theta[2]:.3f}, kNaCa={mhas13_theta[6]:.3f})")

    print("\n[1/2] PHAS13 coupled chip (3x3, dt=0.05, dx=0.025, D=0.001)...")
    t0 = time.perf_counter()
    t_p, V_p = run_coupled(phas13_theta, 'phas13', t_total_ms=10000.0,
                            seed=42, save_every=20)
    print(f"      done ({time.perf_counter() - t0:.1f}s); shape {V_p.shape}")
    print(f"      center-well APs: {count_aps(V_p[:, CENTER_IDX])}")

    print("\n[2/2] MHAS13 fitted coupled chip...")
    t0 = time.perf_counter()
    t_m, V_m = run_coupled(mhas13_theta, 'mhas13', t_total_ms=10000.0,
                            seed=42, save_every=20)
    print(f"      done ({time.perf_counter() - t0:.1f}s); shape {V_m.shape}")
    print(f"      center-well V_max: {V_m[:, CENTER_IDX].max():.1f} mV")
    print(f"      center-well APs: {count_aps(V_m[:, CENTER_IDX])}")

    # Save raw traces for reproducibility
    npz_path = os.path.join(OUT_DIR, 'D1_F2_chip_coupled_traces.npz')
    np.savez(npz_path, phas13_t=t_p, phas13_V=V_p,
             mhas13_t=t_m, mhas13_V=V_m,
             phas13_theta=np.array(phas13_theta),
             mhas13_theta=np.array(mhas13_theta))
    print(f"\nSaved coupled traces: {npz_path}")

    # ----- Subsample to animation frames -----
    target_n_frames = 240
    fps = 30
    n_avail = min(len(t_p), len(t_m))
    stride = max(1, n_avail // target_n_frames)
    t_anim = t_p[::stride][:target_n_frames]
    V_p_anim = V_p[::stride][:target_n_frames]
    V_m_anim = V_m[::stride][:target_n_frames]
    n_frames = len(t_anim)
    print(f"\nAnimation: {n_frames} frames @ {fps} fps "
          f"({n_frames/fps:.1f}s real-time, {t_anim[-1]/1000:.1f}s sim)")

    # ----- Build figure -----
    norm = Normalize(vmin=-95.0, vmax=50.0)
    cmap = coolwarm

    fig = plt.figure(figsize=(11.5, 6.8), dpi=120)
    gs = fig.add_gridspec(
        3, 2, height_ratios=[6, 1.2, 0.4],
        left=0.05, right=0.96, top=0.91, bottom=0.09,
        wspace=0.12, hspace=0.34,
    )
    ax_p  = fig.add_subplot(gs[0, 0]); ax_p.set_aspect('equal')
    ax_m  = fig.add_subplot(gs[0, 1]); ax_m.set_aspect('equal')
    ax_tp = fig.add_subplot(gs[1, 0])
    ax_tm = fig.add_subplot(gs[1, 1])
    ax_cb = fig.add_subplot(gs[2, :])

    for ax, title, color in [
        (ax_p, 'PHAS13 chip (immature, 3x3 monodomain, coupled)', '#2ca02c'),
        (ax_m, 'MHAS13 chip (matured + fitted, 3x3 monodomain, coupled)',
              '#d62728'),
    ]:
        ax.set_xlim(-0.6, N_GRID - 0.4)
        ax.set_ylim(-0.6, N_GRID - 0.4)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11, color=color, pad=10)
        ax.invert_yaxis()
        for s in ax.spines.values():
            s.set_visible(False)

    # Per-well circles
    cells_p, cells_m = [], []
    txts_p,  txts_m  = [], []
    radius = 0.34
    center_ring_p = None
    center_ring_m = None
    for r in range(N_GRID):
        for c in range(N_GRID):
            cp = Circle((c, r), radius,
                        facecolor=cmap(norm(V_p_anim[0, r * N_GRID + c])),
                        edgecolor='black', linewidth=1.0)
            cm = Circle((c, r), radius,
                        facecolor=cmap(norm(V_m_anim[0, r * N_GRID + c])),
                        edgecolor='black', linewidth=1.0)
            ax_p.add_patch(cp); ax_m.add_patch(cm)
            cells_p.append(cp); cells_m.append(cm)
            tp = ax_p.text(c, r, '', ha='center', va='center',
                           fontsize=8, fontweight='bold', color='white')
            tm = ax_m.text(c, r, '', ha='center', va='center',
                           fontsize=8, fontweight='bold', color='white')
            txts_p.append(tp); txts_m.append(tm)

    # Red highlight ring around center well on each chip
    ring_p = Circle((1, 1), radius * 1.18, facecolor='none',
                    edgecolor='red', linewidth=2.8, zorder=5)
    ring_m = Circle((1, 1), radius * 1.18, facecolor='none',
                    edgecolor='red', linewidth=2.8, zorder=5)
    ax_p.add_patch(ring_p); ax_m.add_patch(ring_m)
    ax_p.text(1, 1 + radius * 1.55, 'center well',
              ha='center', va='center', fontsize=8, color='red',
              fontweight='bold')
    ax_m.text(1, 1 + radius * 1.55, 'center well',
              ha='center', va='center', fontsize=8, color='red',
              fontweight='bold')

    # Voltage strips: CENTER cell only
    Vp_center = V_p_anim[:, CENTER_IDX]
    Vm_center = V_m_anim[:, CENTER_IDX]
    ax_tp.plot(t_anim / 1000.0, Vp_center, color='#2ca02c', linewidth=1.0)
    ax_tm.plot(t_anim / 1000.0, Vm_center, color='#d62728', linewidth=1.0)
    for ax in (ax_tp, ax_tm):
        ax.axhline(-40.0, color='gray', linestyle=':', linewidth=0.6, alpha=0.6)
        ax.set_ylim(-95, 50); ax.grid(alpha=0.25)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.tick_params(labelsize=8)
    ax_tp.set_ylabel('V (mV)\ncenter well', fontsize=8)
    ax_tp.set_title('center-well voltage trace', fontsize=8.5,
                     color='red', loc='right', pad=2)
    ax_tm.set_title('center-well voltage trace', fontsize=8.5,
                     color='red', loc='right', pad=2)

    line_p = ax_tp.axvline(0, color='black', linewidth=1.3, alpha=0.7)
    line_m = ax_tm.axvline(0, color='black', linewidth=1.3, alpha=0.7)

    # Title + colorbar
    title_obj = fig.suptitle('', fontsize=12, y=0.985)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb, orientation='horizontal')
    cb.set_label('Membrane potential V (mV)', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    def update(i):
        Vp_now = V_p_anim[i]
        Vm_now = V_m_anim[i]
        t_now_s = t_anim[i] / 1000.0

        for k, (cp, tp) in enumerate(zip(cells_p, txts_p)):
            cp.set_facecolor(cmap(norm(Vp_now[k])))
            tp.set_text(f'{Vp_now[k]:+.0f}')
        for k, (cm, tm) in enumerate(zip(cells_m, txts_m)):
            cm.set_facecolor(cmap(norm(Vm_now[k])))
            tm.set_text(f'{Vm_now[k]:+.0f}')

        line_p.set_xdata([t_now_s, t_now_s])
        line_m.set_xdata([t_now_s, t_now_s])

        title_obj.set_text(
            'Quiescence comparison - 3x3 coupled monodomain, '
            'stochastic ICs, free-run     '
            f't = {t_now_s:.2f} s'
        )

        return (cells_p + cells_m + txts_p + txts_m
                + [line_p, line_m, title_obj])

    ani = FuncAnimation(fig, update, frames=n_frames,
                        interval=1000.0 / fps, blit=False)

    out_mp4 = os.path.join(OUT_DIR, 'D1_F2_chip.mp4')
    print(f"\nEncoding MP4 -> {out_mp4}")
    writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=2400,
                          extra_args=['-pix_fmt', 'yuv420p'])
    ani.save(out_mp4, writer=writer)
    plt.close(fig)
    print(f"Saved MP4: {out_mp4} "
          f"({os.path.getsize(out_mp4) / 1e6:.1f} MB)")

    out_gif = os.path.join(OUT_DIR, 'D1_F2_chip.gif')
    print(f"Converting MP4 -> GIF -> {out_gif}")
    cmd = [
        FFMPEG_EXE, '-y', '-i', out_mp4,
        '-vf',
        f'fps={fps},scale=860:-1:flags=lanczos,'
        'split[s0][s1];[s0]palettegen=max_colors=128[p];'
        '[s1][p]paletteuse=dither=bayer:bayer_scale=5',
        '-loop', '0',
        out_gif,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg stderr:")
        print(result.stderr[-2000:])
        raise RuntimeError("ffmpeg conversion failed")
    print(f"Saved GIF: {out_gif} "
          f"({os.path.getsize(out_gif) / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
