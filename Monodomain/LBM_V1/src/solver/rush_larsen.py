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


def ionic_step(model, V: Tensor, ionic_states: Tensor, dt: float,
               I_stim: Tensor = None) -> tuple:
    """Compute I_ion and update ionic states (gates + concentrations).

    V is NOT modified — it's only used to compute currents and gating.

    Args:
        model: IonicModel instance (e.g., TTP06Model)
        V: (n_cells,) or (Nx*Ny,) membrane potential from distributions
        ionic_states: (n_cells, n_states) ionic state tensor
        dt: time step (ms)
        I_stim: (n_cells,) stimulus current (pA/pF), or None

    Returns:
        I_ion: (n_cells,) total ionic current
        ionic_states_new: (n_cells, n_states) updated ionic states
    """
    # Compute I_ion before updating anything
    I_ion = model.compute_Iion(V, ionic_states)

    # Use model.step() to update both V and ionic states,
    # then discard V_new (LBM V comes from distributions)
    _, ionic_states_new = model.step(V, ionic_states, dt, I_stim)

    return I_ion, ionic_states_new


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
