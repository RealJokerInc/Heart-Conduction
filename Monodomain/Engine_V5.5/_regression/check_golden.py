"""Regression guard: re-run the reference sim and assert it matches the Cm=1 golden."""

import os
import sys

import numpy as np

from _ref_sim import run_reference

golden_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden_cm1.npz')
if not os.path.exists(golden_path):
    sys.exit(f"FAIL: golden not found at {golden_path} (run make_golden.py first)")

golden = np.load(golden_path)
times, voltages = run_reference()

dV = float(np.abs(voltages - golden['voltages']).max())
dt = float(np.abs(times - golden['times']).max())
ok = np.allclose(voltages, golden['voltages'], atol=1e-12, rtol=0.0) and \
     np.allclose(times, golden['times'], atol=1e-12, rtol=0.0)

print(f"check_golden: max|dV|={dV:.3e}  max|dt|={dt:.3e}  -> {'PASS' if ok else 'FAIL'}")
if not ok:
    sys.exit(1)
