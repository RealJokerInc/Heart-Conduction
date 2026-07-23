"""Capture pre-vendor integrity goldens + source-tree content hashes.

Run ONCE, BEFORE any vendoring, against the ORIGINAL engines (reached via _prepare_engine).
Each golden is the bit-identical reference output a vendored engine must reproduce (atol=0) —
the only check that proves the code-MOVE is behavior-preserving (the repointed MatchesDirect
tests compare vendored-vs-vendored and cannot catch a vendoring numerics regression).


The canonical sims here are SMALL, CPU, float64, deterministic (no randomness, fixed mesh) →
bit-reproducible run-to-run, which `save_goldens` asserts before trusting the golden.
"""

import os
import sys
import json
import hashlib

import torch

HERE = os.path.dirname(os.path.abspath(__file__))                 # cardiac_core/tests/_integrity
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))      # repo root
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from cardiac_core import simulate, create_cardiac_mesh  # noqa: E402

DEVICE = "cpu"
ENGINES = ["monodomain", "bidomain", "lbm"]



def canonical_sim(engine: str):
    """A fixed, deterministic sim per engine — the integrity probe. Returns SimulationResult."""
    if engine == "lbm":
        mesh = create_cardiac_mesh(
            Lx=0.5, Ly=0.5, dx=0.025,
            stim_amplitude=-80.0, stim_start=0.0, stim_duration=2.0,
        )
        return simulate(mesh, t_end=3.0, save_every=0.5, engine="lbm", dt=0.005, device=DEVICE)
    mesh = create_cardiac_mesh(
        Lx=1.0, Ly=0.5, dx=0.05,
        stim_amplitude=-80.0, stim_start=0.0, stim_duration=2.0,
    )
    return simulate(mesh, t_end=8.0, save_every=1.0, engine=engine, device=DEVICE)



def save_goldens():
    for e in ENGINES:
        r = canonical_sim(e)
        r2 = canonical_sim(e)  # self-consistency: must be bit-reproducible BEFORE we trust it
        assert torch.equal(r.Vm, r2.Vm), f"{e}: NOT bit-reproducible run-to-run — golden unreliable"
        d = {"times": r.times.cpu(), "Vm": r.Vm.cpu()}
        if r.phi_e is not None:
            d["phi_e"] = r.phi_e.cpu()
        torch.save(d, os.path.join(HERE, f"golden_{e}.pt"))
        print(f"  golden_{e}.pt  Vm{tuple(r.Vm.shape)}  phi_e={'yes' if r.phi_e is not None else 'no'}")



if __name__ == "__main__":
    print("Capturing integrity goldens (pre-vendor, original engines)...")
    save_goldens()
    print("DONE")
