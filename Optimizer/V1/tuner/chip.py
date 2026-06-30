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


def chip_mesh(domain_mm: float = 16.0, dx_mm: float = 0.1,
              D_long: float = None, D_trans: float = None,
              ionic_model: str = "mhas13", dt: float = 0.02):
    """Build the chip CardiacMeshData (per-axis D, chi=1.0).

    D_long/D_trans are effective diffusion (cm²/ms) from the fit. If omitted, a
    placeholder isotropic-ish pair is used (real D comes from Phase 3/4).
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


def wavelength_mm(cv_cm_s: float, apd_ms: float) -> float:
    """λ = CV·APD, in mm. (cm/s · ms = cm/1000 ·... ) -> cv*apd/100 mm."""
    return cv_cm_s * apd_ms / 100.0


def points_per_wavelength(cv_cm_s: float, apd_ms: float, dx_mm: float) -> float:
    """Spatial resolution check; warn (caller) if < ~25."""
    return wavelength_mm(cv_cm_s, apd_ms) / dx_mm
