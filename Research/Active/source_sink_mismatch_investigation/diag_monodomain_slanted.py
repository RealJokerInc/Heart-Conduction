"""
Monodomain V5.4 (FDM) version of the slanted-plane-wave boundary experiment.
Compares boundary_mode face_mirror (forward crescent / deficit) vs
face_mirror_iso (zero bias), with the moore8_uniform stencil (so the diagonal
connectivity deficit is present), in an EMPTY box.

Two stimuli: straight (control) and slanted 20 deg (bottom-leading). For each we
measure the wall LAT deficit (mirror - iso) at the top and bottom walls vs x.

PDE note: there is no PDE analog of inverse-crescent specular. mirror = forward
crescent (wall LAGS = deficit), iso = zero bias. So expect mirror to lag iso at
the wall; positive (mirror - iso) = mirror slower at the wall.

Outputs: media images (LAT grid) + side-by-side mirror|iso videos.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path("/home/norepinephrine/Documents/Heart-Conduction")
sys.path.insert(0, str(REPO / "Monodomain/Engine_V5.4"))
sys.path.insert(0, str(REPO))

from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol
from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation
from cardiac_core.media import media_path

LX, LY, DX = 4.0, 2.0, 0.025
NX = int(round(LX / DX)) + 1     # 161
NY = int(round(LY / DX)) + 1     # 81
DT = 0.02
T_END = 120.0
SAVE_EVERY = 0.1                 # ms (LAT resolution 100 us; deficit is ~0.5 ms/cm)
STENCIL = "moore8_uniform"
ANGLE = 20.0                     # deg from y-axis, bottom-leading
THR = -40.0
TAN = np.tan(np.radians(ANGLE))


def stim_region(slanted: bool):
    if not slanted:
        return lambda x, y: x < 0.05
    # bottom (small y) leads: cutoff larger at the bottom
    return lambda x, y: x < 0.05 + TAN * (LY - y)


def run(boundary_mode: str, slanted: bool):
    grid = StructuredGrid.create_rectangle(LX, LY, NX, NY)
    fdm = FDMDiscretization(grid, D=0.001, chi=1.0, Cm=1.0,
                            stencil=STENCIL, boundary_mode=boundary_mode)
    proto = StimulusProtocol()
    proto.add_stimulus(region=stim_region(slanted), start_time=0.0,
                       duration=2.0, amplitude=-52.0)
    sim = MonodomainSimulation(spatial=fdm, ionic_model='ttp06', stimulus=proto,
                               dt=DT, splitting='strang', ionic_solver='rush_larsen',
                               diffusion_solver='forward_euler', cell_type='EPI')
    times, V_hist = sim.run_to_array(t_end=T_END, save_every=SAVE_EVERY)
    times = np.asarray(times)
    V = V_hist.reshape(len(times), NX, NY)
    return times, V


def lat_field(V, times, thr=THR):
    above = V >= thr
    ever = above.any(0)
    idx = np.argmax(above, axis=0)
    idxc = np.clip(idx, 1, len(times) - 1)
    v1 = np.take_along_axis(V, idxc[None], 0)[0]
    v0 = np.take_along_axis(V, (idxc - 1)[None], 0)[0]
    t1 = times[idxc]; t0 = times[idxc - 1]
    denom = np.where(v1 == v0, 1.0, v1 - v0)
    lat = t0 + (thr - v0) * (t1 - t0) / denom
    lat[idx == 0] = times[0]
    lat[~ever] = np.nan
    return lat


def main():
    cases = {}
    for slanted in (False, True):
        for bc in ("face_mirror", "face_mirror_iso"):
            key = ("slanted" if slanted else "straight", bc)
            print(f"[run] {key} ...", flush=True)
            t, V = run(bc, slanted)
            cases[key] = (t, V)
            print(f"      frames={len(t)}  Vmax={V.max():.1f}")

    x = np.arange(NX) * DX
    # wall deficit: mirror - iso  (+ = mirror slower at wall)
    print("\n=== wall LAT deficit  (face_mirror - face_mirror_iso), ms  (+ = mirror slower) ===")
    lat = {}
    for key, (t, V) in cases.items():
        lat[key] = lat_field(V, t)
    for stim in ("straight", "slanted"):
        Lm = lat[(stim, "face_mirror")]; Li = lat[(stim, "face_mirror_iso")]
        d = Lm - Li
        print(f"[{stim}]")
        for xc in (1, 2, 3):
            i = int(xc / DX)
            print(f"   x={xc}cm  bottom(j=0)={d[i,0]*1000:+7.1f} us   "
                  f"top(j={NY-1})={d[i,NY-1]*1000:+7.1f} us   "
                  f"center={d[i,NY//2]*1000:+7.1f} us")

    # ---- LAT-difference figure: rows straight/slanted, cols mirror/iso/diff ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ext = [0, LX, 0, LY]
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.5), constrained_layout=True)
    for r, stim in enumerate(("straight", "slanted")):
        Lm = lat[(stim, "face_mirror")]; Li = lat[(stim, "face_mirror_iso")]
        for c, (L, ttl, cm, vlim) in enumerate([
                (Lm, "face_mirror LAT", "viridis", None),
                (Li, "face_mirror_iso LAT", "viridis", None),
                ((Lm - Li) * 1000, "mirror - iso (us)\n(+ = mirror slower)", "RdBu_r", 600)]):
            ax = axes[r, c]
            kw = dict(origin="lower", extent=ext, aspect="equal")
            if vlim is not None:
                kw.update(cmap=cm, vmin=-vlim, vmax=vlim)
            else:
                kw.update(cmap=cm)
            im = ax.imshow(L.T, **kw)
            ax.set_title(f"{stim}: {ttl}", fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.7)
            ax.tick_params(labelsize=7)
    fig.suptitle(f"Monodomain V5.4 FDM ({STENCIL}) — mirror vs iso, straight vs slanted {ANGLE:.0f}deg",
                 fontsize=12)
    p = media_path("source_sink_mismatch_investigation", "images", "monodomain-slanted-mirror-vs-iso-lat")
    fig.savefig(p, dpi=110); plt.close(fig); print("wrote", p)

    # ---- videos: side-by-side mirror | iso, for straight and slanted ----
    import imageio_ffmpeg
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    vr = float(min(V.min() for _, V in cases.values()))
    for stim in ("straight", "slanted"):
        tm, Vm = cases[(stim, "face_mirror")]
        ti, Vi = cases[(stim, "face_mirror_iso")]
        nf = min(len(Vm), len(Vi))
        step = max(1, nf // 220)
        frames = range(0, nf, step)
        figv, (axA, axB) = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
        cmap = plt.cm.inferno.copy()
        imA = axA.imshow(Vm[0].T, origin="lower", extent=ext, cmap=cmap, vmin=vr, vmax=40,
                         aspect="equal", interpolation="bilinear")
        imB = axB.imshow(Vi[0].T, origin="lower", extent=ext, cmap=cmap, vmin=vr, vmax=40,
                         aspect="equal", interpolation="bilinear")
        axA.set_title("face_mirror (forward / deficit)"); axB.set_title("face_mirror_iso (zero bias)")
        for ax in (axA, axB):
            ax.set_xlabel("x (cm)"); ax.tick_params(labelsize=7)
        axA.set_ylabel("y (cm)")
        figv.colorbar(imA, ax=(axA, axB), shrink=0.75, label="V (mV)")
        ttl = figv.suptitle(f"Monodomain {STENCIL} — {stim} wave   t=0 ms")

        def upd(fr, imA=imA, imB=imB, Vm=Vm, Vi=Vi, tm=tm, ttl=ttl, stim=stim):
            imA.set_array(Vm[fr].T); imB.set_array(Vi[fr].T)
            ttl.set_text(f"Monodomain {STENCIL} — {stim} wave   t={tm[fr]:.0f} ms")
            return imA, imB, ttl
        anim = FuncAnimation(figv, upd, frames=frames, blit=False)
        out = media_path("source_sink_mismatch_investigation", "videos",
                         f"monodomain-{stim}-mirror-vs-iso", ext="mp4")
        anim.save(out, writer=FFMpegWriter(fps=20, bitrate=3500), dpi=120)
        plt.close(figv); print("wrote", out)


if __name__ == "__main__":
    main()
