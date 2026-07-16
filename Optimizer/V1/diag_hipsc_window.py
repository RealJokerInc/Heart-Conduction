"""
Optimizer V1 — Step 1.3 diagnostic: hiPSC-θ propagating window + high-D NaN cause.

Two architecture open questions this answers with data:
  (1) The propagating window (~5e-5–1e-4) was measured at NRVM θ; the UNREACHABLE
      target is hiPSC — so re-measure the window at the saved hiPSC warm-start θ.
  (2) The high-D NaN (D≈1e-3) was unexplained. Classify each NaN as an
      over-depolarization blow-up (Vmax non-physical → CN accuracy/over-depol, a real
      upper window edge) vs a source-sink block (Vmax physiological, wave dies → the
      lower window edge). This sets the TRUE window width (architecture §4/§9-P0).

Run: conda run -n heart-conduction python Optimizer/V1/diag_hipsc_window.py
Saves a CV & r*/dx vs D figure under media/ionic_model_optimization/images/{date}/.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from tuner.cc_runner import run_1d_cable          # noqa: E402
from tuner.cv_estimator import rstar_cm           # noqa: E402

# Peak V above this (mV) is non-physical (AP peaks ~+15..+55) → over-depolarization.
VMAX_OVERDEPOL = 70.0
# Peak V below the activation threshold means the tissue NEVER fired anywhere: the
# stimulus was drained before it could depolarize the tissue (high-D SINK OVERLOAD /
# no capture) — distinct from a source-sink block, where the tissue DOES fire locally
# (Vmax above threshold) but the wave dies before crossing both probes. -30 mV matches
# cc_runner's activation threshold (_ACT_THRESHOLD).
VMAX_CAPTURE = -30.0


def classify_nan(cv, vmax):
    """propagates | over_depolarization | no_capture (high-D sink overload) |
    source_sink_block (fires locally, wave dies).

    Empirically (Step 1.3, hiPSC θ): the HIGH-D NaN is `no_capture` (Vmax stays
    sub-threshold — the sink drains the stimulus), NOT over-depolarization as the
    architecture §4 hypothesized. The LOW-D NaN is `source_sink_block` (Vmax fires
    positive but the wave blocks)."""
    if math.isfinite(cv) and cv > 0:
        return "propagates"
    if vmax is None or not math.isfinite(vmax):
        return "source_sink_block"
    if vmax > VMAX_OVERDEPOL:
        return "over_depolarization"
    if vmax < VMAX_CAPTURE:
        return "no_capture"
    return "source_sink_block"


def diag_hipsc_window(theta, config, D_grid, *, save_media=True, media_slug="hipsc-window"):
    """Sweep D at fixed θ; classify each point; return window + high-D NaN cause.

    Returns dict with keys: 'window' ((D_lo, D_hi) propagating range or None),
    'nan_cause' (cause of the NaN just ABOVE the window), 'rows', and (if save_media)
    'figure'.
    """
    rows = []
    for D in D_grid:
        cv, vmax = run_1d_cable(theta, D, config, return_vmax=True)
        rs = rstar_cm(D, cv)
        rows.append({
            "D": D, "cv": cv, "vmax": vmax,
            "rstar_over_dx": (rs / config.dx_cm) if math.isfinite(rs) else float("nan"),
            "cause": classify_nan(cv, vmax),
        })

    prop = [r for r in rows if r["cause"] == "propagates"]
    window = (min(r["D"] for r in prop), max(r["D"] for r in prop)) if prop else None
    # NaN cause at the HIGH-D end (largest non-propagating D above the window) — this
    # is the architecture's open question (is the high-D NaN over-depol or artifact?).
    above = sorted(
        (r for r in rows if r["cause"] != "propagates"
         and (window is None or r["D"] > window[1])),
        key=lambda r: r["D"],
    )
    nan_cause = above[-1]["cause"] if above else "none"

    result = {"window": window, "nan_cause": nan_cause, "rows": rows}
    if save_media:
        result["figure"] = _plot(rows, window, config, media_slug)
    return result


def _plot(rows, window, config, slug):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cardiac_core.media import media_path

    Ds = [r["D"] for r in rows]
    cvs = [r["cv"] if math.isfinite(r["cv"]) else float("nan") for r in rows]
    rox = [r["rstar_over_dx"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.set_xscale("log")
    ax1.plot(Ds, cvs, "o-", color="tab:blue", label="CV (cm/s)")
    ax1.set_xlabel("D  (cm²/ms)")
    ax1.set_ylabel("CV (cm/s)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(Ds, rox, "s--", color="tab:red", label="r*/dx")
    ax2.axhline(3.0, color="tab:red", ls=":", lw=1, alpha=0.7)
    ax2.set_ylabel("r*/dx  (dotted = k=3 floor)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    _cause_color = {"over_depolarization": "orange", "no_capture": "purple",
                    "source_sink_block": "gray"}
    for r in rows:
        c = _cause_color.get(r["cause"])
        if c:
            ax1.axvline(r["D"], color=c, ls=":", lw=1, alpha=0.5)
    if window:
        ax1.axvspan(window[0], window[1], color="green", alpha=0.12)

    ax1.set_title(f"hiPSC-θ window @ dx={config.dx_cm*10:.2f} mm  "
                  f"(green=propagates · purple=no-capture/sink · gray=block)")
    fig.tight_layout()
    path = media_path("ionic_model_optimization", "images", slug)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    import warnings
    warnings.filterwarnings("ignore")
    from tuner.config import TuningConfig
    from tuner.presets import load_record

    rec = load_record("chip_hipsc")
    theta = rec["theta_ionic"]
    cfg = TuningConfig(device="cpu", ionic_model="mhas13", tier=2,
                       dx_cm=0.01, cable_length_cm=0.5, dt=0.02,
                       stim_amplitude=-40.0, stim_start=1.0, engine="monodomain")
    D_grid = [4e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2.5e-5, 1e-5]

    res = diag_hipsc_window(theta, cfg, D_grid)
    print(f"hiPSC-θ propagating window (cm²/ms): {res['window']}")
    print(f"high-D NaN cause: {res['nan_cause']}")
    print(f"{'D':>10} {'CV':>9} {'Vmax':>7} {'r*/dx':>7}  cause")
    for r in res["rows"]:
        cv = f"{r['cv']:.3f}" if math.isfinite(r["cv"]) else "nan"
        rox = f"{r['rstar_over_dx']:.2f}" if math.isfinite(r["rstar_over_dx"]) else "nan"
        print(f"{r['D']:>10.2e} {cv:>9} {r['vmax']:>7.1f} {rox:>7}  {r['cause']}")
    print(f"figure: {res.get('figure')}")


if __name__ == "__main__":
    main()
