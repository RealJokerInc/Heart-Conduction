"""
Tests for the resolved chip mesh + joint-fit record schema (PLAN Step 4.1).

Sim-free: checks the resolved chip dx (0.02 mm) actually resolves the source-sink at a
representative fitted (D, CV), and that the record carries the joint-fit provenance
(kinetics, D_trans, achieved r*/dx, dx-ladder) the reentry campaign inherits.
"""

import pytest


def test_chip_mesh_default_dx_resolved():
    """chip_mesh defaults to the resolved 0.02 mm (0.002 cm) grid."""
    from tuner.chip import chip_mesh, RESOLVED_DX_MM
    assert RESOLVED_DX_MM == 0.02
    mesh = chip_mesh(domain_mm=1.0)              # tiny domain, just check spacing
    assert mesh.dx == pytest.approx(0.002)       # 0.02 mm -> 0.002 cm


def test_dx_resolves():
    """At the resolved dx, a representative fitted (D, CV) gives r*/dx ≥ 3 — whereas
    the old 0.1 mm grid did NOT (r*/dx ≈ 2). r* = D/(CV/1000) [cm]."""
    from tuner.cv_estimator import rstar_cm
    # Representative slow-corner fit: D_trans ~ 1.25e-4, CV_T resolved ~ 5.8 (P0 map).
    D, cv = 1.25e-4, 5.8
    rstar = rstar_cm(D, cv)                       # ~0.0216 cm
    assert (rstar / 0.002) >= 3.0                 # resolved at 0.02 mm
    assert (rstar / 0.010) < 3.0                  # NOT resolved at 0.1 mm (the failure)


def test_record_carries_joint_provenance():
    """A joint-fit record carries kinetics + per-axis D + resolved-grid provenance."""
    from tuner.presets import make_record
    rec = make_record(
        name="chip_hipsc", baseline="hipsc",
        theta_ionic={"g_Na": 0.5}, kinetics={"tau_m_scale": 2.1, "v_half_shift": 3.0},
        tissue={"monodomain": {"D_long": 1.4e-4, "D_trans": 4.0e-5, "dt_ms": 0.02}},
        targets={"cv_longitudinal": 5.2, "cv_transverse": 2.6, "apd_90": 350},
        validation={"achieved_rstar_over_dx": {"long": 5.4, "trans": 3.1},
                    "dx_ladder": [0.004, 0.002, 0.001]},
    )
    assert rec["kinetics"]["tau_m_scale"] == 2.1
    assert rec["tissue"]["monodomain"]["D_trans"] == 4.0e-5
    assert rec["mesh"]["dx_mm"] == 0.02                    # resolved default
    assert rec["validation"]["achieved_rstar_over_dx"]["trans"] >= 3.0
    assert rec["validation"]["dx_ladder"] == [0.004, 0.002, 0.001]
