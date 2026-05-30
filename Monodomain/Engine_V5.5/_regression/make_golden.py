"""Capture the Cm=1 regression golden. Run once from the pristine V5.5 copy."""

import os

import numpy as np

from _ref_sim import run_reference

times, voltages = run_reference()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden_cm1.npz')
np.savez(out, times=times, voltages=voltages)
print(f"golden saved: {out}  times={times.shape}  voltages={voltages.shape}")
