"""Cheatsheet canary — uses ONLY patterns from cardiac_core/API_CHEATSHEET.md.

If this breaks, the cheatsheet has drifted from the shipped API; fix the cheatsheet.
Run: conda run -n heart-conduction python Lab/_validate/smoke.py
"""

import cardiac_core as cc

# §8 full example — measure conduction velocity in a 2.0 × 0.5 cm strip.
g = cc.Grid(201, 51, 0.01)
cond = cc.ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)
stim = {"region": lambda x, y: x < 0.05, "start_time": 1.0, "duration": 2.0, "amplitude": -80.0}

sim = cc.monodomain(g, "ttp06", cond, stim)
r = sim.run(t_end=40.0, save_every=0.5)
cv = r.cv(x1=20, x2=100, y=25)

print(f"conduction velocity = {cv:.1f} cm/s")
assert 10.0 < cv < 100.0, f"CV {cv} out of physiological band — cheatsheet/API mismatch"

# §6 — the result hooks the cheatsheet advertises must exist and return the documented shapes.
assert r.Vm.ndim == 3 and r.Vm.shape[1:] == (201, 51), r.Vm.shape
assert r.apd().shape == (201, 51)
assert r.lat().shape == (201, 51)

print("SMOKE OK — cheatsheet matches the shipped API")
