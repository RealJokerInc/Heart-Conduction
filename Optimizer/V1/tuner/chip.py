"""
Optimizer V1 — Kit Parker tissue-chip mesh + EP targets (Phase 2).

"Simulate the entire chip": a 25 mm coverslip → ~16 mm usable square tissue at
dx = 0.1 mm (161² grid). Anisotropic ~2:1 (Bursac & Parker 2002 ratio ≈2.1):
CV_T = CV_L / ratio, hence D_T = D_L / ratio² (CV ∝ √D).

Provenance caveat: CV_L 9.33 (NRVM) / 5.2 (hiPSC) are MacQueen 2018 engineered
*ventricle* values; the 2:1 ratio is from NRVM/monolayer — a cross-construct
assumption applied to CV_L (flag if it matters).

NOTE (Phase 0 convention): meshes carry an already-effective D, so chi=1.0 (the
FDM operator divides by chi). λ exceeds the 16 mm chip at APD≈350 — that tension
is the reentry application's to resolve (rapid pacing / larger domain), not the fit's.
"""
from cardiac_core import create_cardiac_mesh

from .config import TuningTargets

# Anisotropy ratio CV_L/CV_T ≈ 2:1 (Bursac & Parker 2002 ≈2.1; de Diego 2010 2.1±0.8).
ANISOTROPY_RATIO = 2.0

# dvdt_max_upper=120 (NOT the default 60, which rejects MHAS13's ~110 V/s — Known Failure).
PARKER_NRVM = TuningTargets(
    cv_longitudinal=9.33, cv_transverse=9.33 / ANISOTROPY_RATIO,
    apd_90=350.0, dvdt_max=110.0, dvdt_max_upper=120.0,
)
PARKER_HIPSC = TuningTargets(
    cv_longitudinal=5.2, cv_transverse=5.2 / ANISOTROPY_RATIO,
    apd_90=350.0, dvdt_max=110.0, dvdt_max_upper=120.0,
)
PARKER = {"nrvm": PARKER_NRVM, "hipsc": PARKER_HIPSC}


# Resolved chip resolution (architecture lock-3 / P0 feasibility): the SLOW corner
# needs r*/dx ≳ 3 to RESOLVE source-sink. At the transverse corner r* ≈ 62 µm so
# k=3 ⇒ dx ≲ 20 µm ≈ 0.02 mm (0.03 mm gives only r*/dx≈2). The P0 feasibility map
# confirmed dx=0.1 mm is unresolved (r*/dx≈1.9–2.2) while dx≤0.03 mm resolves the
# fast axis (r*/dx≈7). ≈25× more cells than the old 0.1 mm — the reentry campaign
# inherits this heavier grid (documented in the hand-off).
RESOLVED_DX_MM = 0.02


def chip_mesh(domain_mm: float = 16.0, dx_mm: float = RESOLVED_DX_MM,
              D_long: float = None, D_trans: float = None,
              ionic_model: str = "mhas13", dt: float = 0.02):
    """Build the chip CardiacMeshData (per-axis D, chi=1.0).

    D_long/D_trans are effective diffusion (cm²/ms) from the fit. If omitted, a
    placeholder isotropic-ish pair is used (real D comes from Phase 3/4). ``dx_mm``
    defaults to the RESOLVED 0.02 mm (r*/dx≥3 at the slow corner); pass 0.1 mm only
    for the legacy/coarse behaviour.
    """
    L = domain_mm / 10.0          # mm -> cm
    dx = dx_mm / 10.0
    if D_long is None:
        D_long = 1.0e-4
    if D_trans is None:
        D_trans = D_long / ANISOTROPY_RATIO ** 2   # CV_T=CV_L/ratio -> D_T=D_L/ratio²
    return create_cardiac_mesh(
        Lx=L, Ly=L, dx=dx, D=D_long, D_yy=D_trans,
        chi=1.0, Cm=1.0, ionic_model=ionic_model, dt=dt,
    )


# --- Lateral boundary-speedup guide (boundary_conduction_speedup, 2026-06-25) ---
# The side-wall isochrone crescent is controlled by the LBM relaxation number
#   β = D·dt/dx²  ⟺  τ = 0.5 + β/c_s²  (c_s² = 1/3), matching the engine's tau_from_D.
# CV is INERT to it. CRUCIALLY the regime is BC-SPECIFIC (bcs KNOWLEDGE 2026-06-25):
#   • HBB (halfway bounce-back) — FORWARD (slow-down) at ALL τ; |C| grows with τ;
#     NO speed-up ever. ← this is what cardiac_core's LBM actually runs (the chip).
#   • same-cell specular — the only rule that FLIPS sign with τ:
#       τ ≲ 0.67 inverse → wall SPEED-UP · τ ≈ 0.75 flat (C≈0) · τ ≳ 0.84 forward.
#       NOT wired into cardiac_core's _lbm (HBB-only) → speed-up is unreachable
#       on the chip until the specular/α-blend BC is added to the engine.
#   • neighbour-cell ("zero") — flat (C≈0) at all τ.
# Distinct from the source-sink dx/r* number (geometry/CV-driven; comes later). No
# converged scalar curvature metric yet → use this number to GUIDE the (free) dt knob.
# NOTE: dt is SHARED with the Rush-Larsen ionic step, so moving dt to shift τ also
# coarsens the ionic upstroke/APD — bounds how far dt can be pushed.
CS2 = 1.0 / 3.0
TAU_SPEEDUP_MAX = 0.67     # specular only — τ ≤ this: inverse crescent (speed-up)
TAU_FORWARD_MIN = 0.84     # specular only — τ ≥ this: forward crescent (slow-down)

# BC aliases → the three named wall families. Names match cardiac_core's WALL_MODES
# ('neumann','hbb','specular_nextcell','specular_samecell','combined') so the same
# string passed to run_lbm(boundary=...) can be handed straight to this guide.
_HBB_BC = {"hbb", "neumann", "insulated", "bounce", "bounce_back", "no_flux"}
# sign-flipping family (same-cell specular; 'combined' with alpha<1 rides this branch).
_SPECULAR_BC = {"specular_samecell", "scs", "specular", "same_cell", "same-cell",
                "combined", "alpha0"}
_ZERO_BC = {"specular_nextcell", "ncs", "zero", "neighbour", "neighbor", "neighbour_cell"}


def boundary_number(D: float, dt: float, dx_mm: float, bc: str = "hbb") -> dict:
    """Lateral-wall crescent guide from (D, dt, dx, bc). τ matches LBM `tau_from_D`.

    β = D·dt/dx²,  τ = 0.5 + β/c_s²  (c_s² = 1/3). D in cm²/ms, dt in ms, dx_mm in
    mm. `bc` selects the wall family — the regime is BC-SPECIFIC:
      - "hbb" (DEFAULT — cardiac_core's actual LBM wall): forward at all τ, no speed-up.
      - "specular" (same-cell; NOT in cardiac_core's _lbm): flips sign at τ≈0.75.
      - "zero" (neighbour-cell): flat (C≈0) at all τ.
    Returns {"beta", "tau", "bc", "regime"}. Guide only — no curvature is fit.
    """
    dx = dx_mm / 10.0                          # mm -> cm (D is cm²/ms)
    beta = D * dt / (dx * dx)
    tau = 0.5 + beta / CS2                      # == 0.5 + 3·β  (c_s² = 1/3)
    bc_l = bc.lower()
    if bc_l in _HBB_BC:
        regime = "forward (wall slow-down; |C|∝τ, no speed-up under HBB)"
    elif bc_l in _ZERO_BC:
        regime = "flat (C≈0 at all τ)"
    elif bc_l in _SPECULAR_BC:
        if tau <= TAU_SPEEDUP_MAX:
            regime = "inverse (wall speed-up)"
        elif tau >= TAU_FORWARD_MIN:
            regime = "forward (wall slow-down)"
        else:
            regime = "crossover (~flat)"
    else:
        raise ValueError(f"unknown bc {bc!r}; expected one of hbb/specular/zero")
    return {"beta": beta, "tau": tau, "bc": bc_l, "regime": regime}


def wavelength_mm(cv_cm_s: float, apd_ms: float) -> float:
    """λ = CV·APD, in mm. (cm/s · ms = cm/1000 ·... ) -> cv*apd/100 mm."""
    return cv_cm_s * apd_ms / 100.0


def points_per_wavelength(cv_cm_s: float, apd_ms: float, dx_mm: float) -> float:
    """Spatial resolution check; warn (caller) if < ~25."""
    return wavelength_mm(cv_cm_s, apd_ms) / dx_mm
