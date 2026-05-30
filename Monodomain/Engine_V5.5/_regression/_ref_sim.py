"""
Shared reference simulation for the V5.5 Cm-regression golden.

Deterministic Cm=1.0 monodomain run (FDM + Strang + Rush-Larsen + Crank-Nicolson/PCG)
on a small CPU/float64 grid. Both make_golden.py and check_golden.py call run_reference()
so the captured golden and the checked run use byte-identical configuration.

The golden is captured from the PRISTINE V5.5 copy (code identical to V5.4 at fork time),
so it encodes V5.4's Cm=1 behavior without touching the V5.4 tree. After the Phase-1 Cm
fix, this run must still match (Cm=1 -> reaction /1.0 is an exact no-op).
"""

import os
import sys

# Make the engine root importable regardless of cwd (script dir is _regression/).
_ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ENGINE_ROOT not in sys.path:
    sys.path.insert(0, _ENGINE_ROOT)

import numpy as np
import torch

from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation
from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol


def run_reference():
    """Run the fixed reference sim. Returns (times, voltages) as numpy arrays."""
    torch.set_grad_enabled(False)

    grid = StructuredGrid.create_rectangle(
        Lx=1.2, Ly=0.08, Nx=60, Ny=4, device='cpu', dtype=torch.float64
    )
    spatial = FDMDiscretization(grid, D=0.001, chi=1.0, Cm=1.0)  # Cm=1 baseline

    stim = StimulusProtocol()
    x, _y = grid.coordinates
    stim.add_stimulus(region=(x < 0.05), start_time=0.0, duration=1.0, amplitude=-52.0)

    sim = MonodomainSimulation(
        spatial=spatial,
        ionic_model='ttp06',
        stimulus=stim,
        dt=0.02,
        splitting='strang',
        ionic_solver='rush_larsen',
        diffusion_solver='crank_nicolson',
        linear_solver='pcg',
    )
    times, voltages = sim.run_to_array(t_end=50.0, save_every=1.0)
    return np.asarray(times), np.asarray(voltages)
