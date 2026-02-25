#!/usr/bin/env python
"""Run V5.1 EPI cell simulation and compare with V5."""
import sys
import os
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ionic import ORdModel, CellType, StateIndex
from ionic.camkii import compute_CaMKa, fCaMKp

model = ORdModel(celltype=CellType.EPI, device='cpu')
y = model.get_initial_state()
dt = 0.01

# Run to t=15ms
for i in range(1500):
    t = i * dt
    stim = -80.0 if 10.0 <= t < 11.0 else 0.0
    stim_t = torch.tensor(stim, dtype=torch.float64)
    y = model.step(y, dt, stim_t)

# Get currents at t=15ms
p = model.params
CaMKb, CaMKa = compute_CaMKa(y[StateIndex.CaMKt], y[StateIndex.cass], p.CaMKo, p.KmCaM)
fCaMKp_val = fCaMKp(CaMKa, p.KmCaMK)
currents = model.compute_currents(y, fCaMKp_val)

print("V5.1 EPI at t=15ms:")
print(f"V = {y[StateIndex.V].item():.6f}")
print(f"iF = {y[StateIndex.iF].item():.9f}")
print(f"iS = {y[StateIndex.iS].item():.9f}")
print(f"a = {y[StateIndex.a].item():.9f}")
print()
print("Currents:")
for name, val in sorted(currents.items()):
    v = val.item() if isinstance(val, torch.Tensor) else val
    print(f"  {name}: {v:.9f}")

# Load V5 state and compare
y_v5 = np.load('/tmp/v5_epi_state.npy')
print()
print("="*60)
print("Comparison at t=15ms:")
print("="*60)
print(f"{'Variable':<12} {'V5':<18} {'V5.1':<18} {'Diff':<15}")
print("-"*60)

vars_to_check = [
    ('V', StateIndex.V),
    ('iF', StateIndex.iF),
    ('iS', StateIndex.iS),
    ('a', StateIndex.a),
    ('nai', StateIndex.nai),
    ('ki', StateIndex.ki),
    ('cai', StateIndex.cai),
    ('cass', StateIndex.cass),
]
for name, idx in vars_to_check:
    v5_val = y_v5[idx]
    v51_val = y[idx].item()
    diff = abs(v5_val - v51_val)
    print(f"{name:<12} {v5_val:<18.9f} {v51_val:<18.9f} {diff:<15.2e}")
