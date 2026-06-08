"""
Measure the electrotonic FOOT length (how far the subthreshold depolarization
leaks AHEAD of the propagating upstroke) in both engines, plus a pure-diffusion
baseline (how far injected charge spreads in one upstroke time).

Step 1: active foot lambda from a propagating plane wave (LBM and monodomain).
Step 2: pure-diffusion baseline (LBM, ionics off): inject a step, measure spread.

Compare to physiological electrotonic length ~0.5-1 mm. No tuning yet — just
the numbers, to decide whether the front is mistuned-sharp or physiologically
sharp before touching D.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

REPO = Path("/home/norepinephrine/Documents/Heart-Conduction")
sys.path.insert(0, str(REPO / "LBM/Engine_V1"))
sys.path.insert(0, str(REPO / "Monodomain/Engine_V5.4"))
sys.path.insert(0, str(REPO))
from cardiac_core.media import media_path

LX, LY, DX = 4.0, 0.3, 0.025
NX = int(round(LX / DX)) + 1     # 161
NY = int(round(LY / DX)) + 1     # 13  (thin quasi-1D strip)
DT = 0.02
T_END = 70.0
SAVE_EVERY = 0.1
D = 0.001
VTHR = -40.0


def run_lbm_wave():
    import torch
    from src.collision.bgk import bgk_collide
    from src.streaming.d2q9 import stream_d2q9
    from src.boundary.masks import precompute_bounce_masks
    from src.boundary.neumann import apply_neumann_d2q9
    from src.state import recover_voltage
    from src.solver.rush_larsen import compute_source_term, ionic_step
    from src.diffusion import tau_from_D
    from src.lattice import D2Q9
    from ionic.ttp06.model import TTP06Model
    from ionic.base import CellType
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lat = D2Q9()
    domain = torch.ones(NX, NY, dtype=torch.bool, device=dev)
    obm = precompute_bounce_masks(domain, lat); bm = {}
    for a in range(1, 9):
        m = torch.zeros(NX, NY, dtype=torch.bool, device=dev); ex, ey = lat.e[a]
        if ex == 1: m[-1, :] = True
        if ex == -1: m[0, :] = True
        if ey == 1: m[:, -1] = True
        if ey == -1: m[:, 0] = True
        bm[a] = obm[a] | m
    ionic = TTP06Model(cell_type=CellType.EPI, device=dev); V_rest = float(ionic.V_rest)
    w = torch.tensor(lat.w, dtype=torch.float64, device=dev)
    omega = 1.0 / tau_from_D(D, DX, DT, cs2=lat.cs2)
    V = torch.full((NX, NY), V_rest, dtype=torch.float64, device=dev); V[:2, :] = 20.0
    f = w[:, None, None] * V[None, :, :]
    states = ionic.get_initial_state(n_cells=NX * NY)
    Iz = torch.zeros(NX * NY, dtype=torch.float64, device=dev)
    n = int(round(T_END / DT)); siv = int(round(SAVE_EVERY / DT))
    rows = [V[:, NY // 2].cpu().numpy()]; tt = [0.0]
    for k in range(1, n + 1):
        I_ion = ionic.compute_Iion(V.reshape(-1), states)
        R = compute_source_term(I_ion, Iz, 1.0).reshape(NX, NY)
        f = bgk_collide(f, V, R, DT, omega, w); fs = f.clone()
        f = stream_d2q9(f); f = apply_neumann_d2q9(f, fs, bm); V = recover_voltage(f)
        states = ionic_step(ionic, V.reshape(-1), states, DT)
        if k % siv == 0:
            rows.append(V[:, NY // 2].cpu().numpy()); tt.append(k * DT)
    return np.array(tt), np.array(rows), V_rest


def run_lbm_diffusion(tau_probe):
    """Pure diffusion (ionics off): step IC, measure spread after tau_probe ms."""
    import torch
    from src.collision.bgk import bgk_collide
    from src.streaming.d2q9 import stream_d2q9
    from src.boundary.masks import precompute_bounce_masks
    from src.boundary.neumann import apply_neumann_d2q9
    from src.state import recover_voltage
    from src.diffusion import tau_from_D
    from src.lattice import D2Q9
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lat = D2Q9(); V_rest = -85.23
    domain = torch.ones(NX, NY, dtype=torch.bool, device=dev)
    obm = precompute_bounce_masks(domain, lat); bm = {}
    for a in range(1, 9):
        m = torch.zeros(NX, NY, dtype=torch.bool, device=dev); ex, ey = lat.e[a]
        if ex == 1: m[-1, :] = True
        if ex == -1: m[0, :] = True
        if ey == 1: m[:, -1] = True
        if ey == -1: m[:, 0] = True
        bm[a] = obm[a] | m
    w = torch.tensor(lat.w, dtype=torch.float64, device=dev)
    omega = 1.0 / tau_from_D(D, DX, DT, cs2=lat.cs2)
    V = torch.full((NX, NY), V_rest, dtype=torch.float64, device=dev)
    V[:4, :] = 0.0                     # injected depolarized block at left
    f = w[:, None, None] * V[None, :, :]
    Rz = torch.zeros(NX, NY, dtype=torch.float64, device=dev)
    n = int(round(tau_probe / DT)); x_edge = 4 * DX
    for k in range(n):
        f = bgk_collide(f, V, Rz, DT, omega, w); fs = f.clone()
        f = stream_d2q9(f); f = apply_neumann_d2q9(f, fs, bm); V = recover_voltage(f)
    row = V[:, NY // 2].cpu().numpy()
    return row, V_rest, x_edge


def run_mono_wave():
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol
    from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
    from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation
    grid = StructuredGrid.create_rectangle(LX, LY, NX, NY)
    fdm = FDMDiscretization(grid, D=D, chi=1.0, Cm=1.0, stencil='cardinal4',
                            boundary_mode='face_mirror')
    proto = StimulusProtocol()
    proto.add_stimulus(region=lambda x, y: x < 0.05, start_time=0.0, duration=2.0, amplitude=-52.0)
    sim = MonodomainSimulation(spatial=fdm, ionic_model='ttp06', stimulus=proto, dt=DT,
                               splitting='strang', ionic_solver='rush_larsen',
                               diffusion_solver='forward_euler', cell_type='EPI')
    times, Vflat = sim.run_to_array(t_end=T_END, save_every=SAVE_EVERY)
    V = np.asarray(Vflat).reshape(len(times), NX, NY)
    return np.asarray(times), V[:, :, NY // 2], -85.23


def measure_foot(times, rows, V_rest, name):
    x = np.arange(NX) * DX
    # leading edge = RIGHTMOST above-threshold cell (not the stim at x=0)
    fronts = []
    for r in rows:
        above = r >= VTHR
        fronts.append(x[np.where(above)[0].max()] if above.any() else np.nan)
    fronts = np.array(fronts)
    # pick snapshot where the leading edge is nearest x=2 cm (mid, clear of stim & wall)
    k = int(np.nanargmin(np.abs(fronts - 2.0)))
    r = rows[k]; xf = fronts[k]
    # upstroke time (foot time) at the cell at xf: time it spent between rest+5 and VTHR
    ic = int(xf / DX)
    tr = rows[:, ic]
    t_rest = times[np.argmax(tr >= V_rest + 5)] if (tr >= V_rest + 5).any() else np.nan
    t_thr = times[np.argmax(tr >= VTHR)] if (tr >= VTHR).any() else np.nan
    tau_foot = t_thr - t_rest
    # spatial foot: ahead of front, fit ln(V - V_rest) vs x over subthreshold band
    ahead = (x > xf) & (r > V_rest + 1.0) & (r < VTHR - 2.0)
    lam = np.nan
    if ahead.sum() >= 4:
        coeff = np.polyfit(x[ahead] - xf, np.log(r[ahead] - V_rest), 1)
        lam = -1.0 / coeff[0] if coeff[0] < 0 else np.nan
    # foot width AHEAD: distance from front to where V drops back to rest+2 ahead
    xs_ahead = x[(x > xf) & (r >= V_rest + 2)]
    foot_w = (xs_ahead.max() - xf) if len(xs_ahead) else 0.0
    print(f"[{name}] front@x={xf:.2f}cm  foot lambda={lam*10:.3f} mm ({lam/DX:.1f} cells)  "
          f"foot-width(rest+5->thr)={foot_w*10:.3f} mm  tau_foot={tau_foot:.2f} ms")
    return x, r, xf, lam, tau_foot


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    print(f"Strip {NX}x{NY} ({LX}x{LY}cm) dx={DX}, D={D}")
    print("physiological electrotonic length ~0.5-1 mm (~2-4 cells at this dx)\n")

    tL, rL, vrL = run_lbm_wave()
    xL, rowL, xfL, lamL, tauL = measure_foot(tL, rL, vrL, "LBM wave")
    tM, rM, vrM = run_mono_wave()
    xM, rowM, xfM, lamM, tauM = measure_foot(tM, rM, vrM, "MONO wave")

    # pure-diffusion baseline: spread over one foot-time (use the measured tau_foot)
    tau_probe = max(0.5, np.nanmean([tauL, tauM]))
    rowD, vrD, xedge = run_lbm_diffusion(tau_probe)
    x = np.arange(NX) * DX
    # diffusion decay length from the injected edge
    ahead = (x > xedge) & (rowD > vrD + 1.0) & (rowD < vrD + 40.0)
    lamD = np.nan
    if ahead.sum() >= 4:
        c = np.polyfit(x[ahead] - xedge, np.log(rowD[ahead] - vrD), 1)
        lamD = -1.0 / c[0] if c[0] < 0 else np.nan
    print(f"[pure diffusion] over tau={tau_probe:.2f} ms: spread decay length="
          f"{lamD*10:.3f} mm ({lamD/DX:.1f} cells);  sqrt(2*D*tau)={np.sqrt(2*D*tau_probe)*10:.3f} mm")

    # figure: foot profiles
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(xL - xfL, rowL, label=f"LBM wave (λ_foot={lamL*10:.2f}mm)", lw=1.5)
    ax.plot(xM - xfM, rowM, label=f"mono wave (λ_foot={lamM*10:.2f}mm)", lw=1.5)
    ax.axhline(VTHR, color='gray', ls=':', lw=1, label='threshold -40mV')
    ax.set_xlim(-1.0, 0.5); ax.set_xlabel("x - front (cm)  [negative = behind, positive = ahead]")
    ax.set_ylabel("V (mV)"); ax.set_title("Electrotonic foot: subthreshold ramp ahead of the upstroke")
    ax.legend(); ax.grid(alpha=0.3)
    p = media_path("source_sink_mismatch_investigation", "images", "foot-lambda-profiles")
    fig.savefig(p, dpi=120); plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    main()
