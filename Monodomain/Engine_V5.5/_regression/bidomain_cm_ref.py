"""
Generate the Bidomain V1 cross-validation reference for V5.5's Cm fix.

RUNS IN THE BIDOMAIN ENGINE (separate process — Bidomain V1 and Monodomain V5.5 both use
the `cardiac_sim` package name and cannot be imported together). Writes JSON consumed by
Monodomain/Engine_V5.5/test_phase10_cm_scaling.py::test_bidomain_cross_validation.

Bidomain V1 is an independent, Formulation-B (Cm-correct) engine. With isotropic scalar
D_i/D_e and insulated (Neumann) boundaries, its bulk propagation equals monodomain with
D_eff = D_i*D_e/(D_i+D_e) (equal-anisotropy reduction), measured at the CENTER row (away
from any edge effect). We run it at Cm=1 and Cm=2.

Formulation-B input rule: D_i/D_e are physical diffusivities sigma/(chi*Cm). To hold sigma
fixed while scaling Cm->k, rescale D_i,D_e -> /k. (V5.5 monodomain instead holds its input
D fixed and lets D_phys = D/(chi*Cm) shrink — see the test. Both give D_eff = D_EFF/k.)

NOTE: this is NOT a time-dilation test. Cm does not dilate the AP (gate kinetics carry no
Cm). It is an absolute cross-engine agreement test: two independently-correct engines must
report the same CV/APD at each Cm.
"""

import json
import os
import sys

import numpy as np
import torch

BIDOMAIN_ROOT = "/home/norepinephrine/Documents/Heart-Conduction/Bidomain/Engine_V1"
sys.path.insert(0, BIDOMAIN_ROOT)
sys.path.insert(0, os.path.join(BIDOMAIN_ROOT, "tests"))

from cv_shared import (run_bidomain, measure_cv_from_history,
                       D_I, D_E, NX, NY, DX, DT, Y_CENTER, X1, X2, THRESHOLD)

# CV is the robust cross-engine metric here. APD is dropped: capturing full repolarization
# (~300+ ms) at this grid is very expensive, and a partial-window APD is meaningless.
# Larger Cm slows conduction (D_phys and reaction both shrink), so the wave needs longer
# to cross x2 -> use a per-Cm horizon.
T_END_BY_CM = {1.0: 60.0, 2.0: 160.0}


def run_one(Cm):
    # Hold sigma fixed -> rescale physical diffusivities by 1/Cm (Formulation B).
    times, V_hist = run_bidomain(
        nx=NX, ny=NY, dx=DX, dt=DT, D_i=D_I / Cm, D_e=D_E / Cm,
        bc_type='insulated', t_end=T_END_BY_CM[Cm], Cm=Cm,
    )
    cv = measure_cv_from_history(V_hist, times, y=Y_CENTER, threshold=THRESHOLD)  # cm/ms
    return float(cv)


def main():
    out = {'engine': 'bidomain_v1', 'D_EFF_input': float(D_I * D_E / (D_I + D_E)),
           'NX': NX, 'NY': NY, 'DX': DX, 'DT': DT,
           'X1': X1, 'X2': X2, 'Y_CENTER': Y_CENTER, 'THRESHOLD': THRESHOLD,
           'cases': {}}
    for Cm in (1.0, 2.0):
        cv = run_one(Cm)
        out['cases'][str(Cm)] = {'cv_cm_per_s': cv * 1000.0}
        print(f"  bidomain Cm={Cm}: CV={cv*1000:.2f} cm/s")
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bidomain_cm_ref.json')
    with open(dst, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"wrote {dst}")


if __name__ == '__main__':
    main()
