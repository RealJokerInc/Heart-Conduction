"""Rush-Larsen ionic state update for LBM workflow.

In the LBM framework:
- V is recovered from distributions (sum of f), NOT updated by the ionic solver
- The ionic solver updates only gates and concentrations
- I_ion is used as the source term R in the collision operator

This module provides the ionic step that decouples V evolution from ionic states.

Layer 2: pure function, no state.
"""

import torch
from torch import Tensor


def ionic_step(model, V: Tensor, ionic_states: Tensor, dt: float) -> Tensor:
    """Update ionic states (gates + concentrations) only.

    V is NOT modified — it's only used to compute currents and gating.
    For LBM, V comes from distributions, not the ionic ODE.

    Note: model.step() internally computes I_ion and V_new. We discard V_new.
    To get I_ion for the LBM source term, call model.compute_Iion() separately
    (done in LBMSimulation.step() before this function).

    Args:
        model: IonicModel instance (e.g., TTP06Model)
        V: (n_cells,) or (Nx*Ny,) membrane potential from distributions
        ionic_states: (n_cells, n_states) ionic state tensor
        dt: time step (ms)

    Returns:
        ionic_states_new: (n_cells, n_states) updated ionic states
    """
    # model.step() updates gates (Rush-Larsen) and concentrations (Forward Euler).
    # It also computes V_new internally, which we discard since LBM V comes from
    # distributions. I_stim is not passed because it only affects V_new (discarded).
    _, ionic_states_new = model.step(V, ionic_states, dt)

    return ionic_states_new


def compute_source_term(I_ion: Tensor, I_stim: Tensor,
                        Cm: float) -> Tensor:
    """Convert ionic current to LBM source term R.

    From the monodomain equation divided by chi*Cm:
        dV/dt = D*laplacian(V) + R
        R = -(I_ion + I_stim) / Cm

    Note: chi is already absorbed into D = sigma/(chi*Cm). The source term
    only involves Cm, not chi.

    Args:
        I_ion: (n_cells,) ionic current (pA/pF)
        I_stim: (n_cells,) stimulus current (pA/pF), or 0
        Cm: membrane capacitance (uF/cm^2)

    Returns:
        R: (n_cells,) source term for collision operator
    """
    return -(I_ion + I_stim) / Cm
