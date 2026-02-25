#!/usr/bin/env python
"""Run V5 EPI cell simulation and save state."""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ionic import ORdModel, CellType, StateIndex
from ionic.camkii import compute_CaMKa, fCaMKp

model = ORdModel(celltype=CellType.EPI)
y = model.get_initial_state()
dt = 0.01

# Run to t=15ms
for i in range(1500):
    t = i * dt
    stim = -80.0 if 10.0 <= t < 11.0 else 0.0
    y = model.step(y, dt, stim)

# Get currents at t=15ms
CaMKt = y[StateIndex.CaMKt]
cass = y[StateIndex.cass]
CaMKb, CaMKa = compute_CaMKa(CaMKt, cass, model.params.CaMKo, model.params.KmCaM)
currents = model.compute_currents(y, CaMKa)

print("V5 EPI at t=15ms:")
print(f"V = {y[StateIndex.V]:.6f}")
print(f"iF = {y[StateIndex.iF]:.9f}")
print(f"iS = {y[StateIndex.iS]:.9f}")
print(f"a = {y[StateIndex.a]:.9f}")
print()
print("Currents:")
for name, val in sorted(currents.items()):
    print(f"  {name}: {val:.9f}")

# Save state
np.save('/tmp/v5_epi_state.npy', y)
