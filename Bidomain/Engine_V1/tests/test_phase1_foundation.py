"""
Phase 1 Validation Tests — Foundation

Tests 1-T1 through 1-T6 from PROGRESS.md.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest


# === 1-T1: Package import ===

def test_package_import():
    """1-T1: import cardiac_sim succeeds, __version__ == '1.0.0'."""
    import cardiac_sim
    assert cardiac_sim.__version__ == '1.0.0'


# === 1-T2: BidomainConductivity ===

def test_bidomain_conductivity():
    """1-T2: Construct with defaults, check D_i, D_e, get_effective_monodomain_D()."""
    from cardiac_sim.tissue_builder.tissue.conductivity import BidomainConductivity

    cond = BidomainConductivity()
    assert cond.D_i == pytest.approx(0.00124, abs=1e-6)
    assert cond.D_e == pytest.approx(0.00446, abs=1e-6)

    # D_eff = D_i * D_e / (D_i + D_e) = 0.00124 * 0.00446 / 0.00570 ≈ 0.000970
    D_eff = cond.get_effective_monodomain_D()
    assert D_eff == pytest.approx(0.000970, abs=1e-5)

    # Boundary enhanced D
    assert cond.get_boundary_enhanced_D() == pytest.approx(0.00124, abs=1e-6)

    # Isotropy check
    assert cond.is_isotropic is True

    # D_sum
    assert cond.D_sum == pytest.approx(0.00570, abs=1e-5)


# === 1-T3: BidomainState ===

def test_bidomain_state():
    """1-T3: Construct state, check Vm/phi_e shapes, Vm_flat, V alias."""
    from cardiac_sim.simulation.classical.state import BidomainState

    n_dof = 100
    Vm = torch.zeros(n_dof, dtype=torch.float64)
    phi_e = torch.zeros(n_dof, dtype=torch.float64)
    ionic_states = torch.zeros(n_dof, 18, dtype=torch.float64)

    state = BidomainState(
        spatial=None,
        n_dof=n_dof,
        x=torch.zeros(n_dof),
        y=torch.zeros(n_dof),
        Vm=Vm,
        phi_e=phi_e,
        ionic_states=ionic_states,
        gate_indices=[0, 1, 2],
        concentration_indices=[3, 4],
    )

    assert state.Vm.shape == (n_dof,)
    assert state.phi_e.shape == (n_dof,)
    assert state.ionic_states.shape == (n_dof, 18)
    assert state.Vm_flat.shape == (n_dof,)
    assert state.phi_e_flat.shape == (n_dof,)

    # V alias for ionic solver compatibility
    assert state.V is state.Vm
    state.V = torch.ones(n_dof, dtype=torch.float64)
    assert state.Vm[0].item() == 1.0

    # Default stim data
    assert len(state.stim_starts) == 0
    assert state.t == 0.0


# === 1-T4: BoundarySpec ===

def test_boundary_spec():
    """1-T4: Test all BoundarySpec factory methods and properties."""
    from cardiac_sim.tissue_builder.mesh.boundary import (
        BoundarySpec, BCType, Edge, EdgeBC
    )

    # Insulated: all Neumann
    bs = BoundarySpec.insulated()
    assert bs.phi_e_has_null_space is True
    assert bs.phi_e_spectral_eligible is True
    assert bs.phi_e_uniform_bc == BCType.NEUMANN
    assert bs.spectral_transform == 'dct'

    # Bath-coupled: phi_e Dirichlet everywhere
    bs = BoundarySpec.bath_coupled()
    assert bs.phi_e_has_null_space is False
    assert bs.phi_e_spectral_eligible is True
    assert bs.phi_e_uniform_bc == BCType.DIRICHLET
    assert bs.spectral_transform == 'dst'

    # Vm is always Neumann
    for edge in Edge:
        assert bs.Vm[edge].bc_type == BCType.NEUMANN

    # Bath-coupled edges (mixed)
    bs = BoundarySpec.bath_coupled_edges([Edge.TOP, Edge.BOTTOM])
    assert bs.phi_e_has_null_space is False  # Has Dirichlet
    assert bs.phi_e_spectral_eligible is False  # Mixed
    assert bs.phi_e_uniform_bc is None
    assert bs.spectral_transform is None

    assert bs.phi_e[Edge.TOP].bc_type == BCType.DIRICHLET
    assert bs.phi_e[Edge.BOTTOM].bc_type == BCType.DIRICHLET
    assert bs.phi_e[Edge.LEFT].bc_type == BCType.NEUMANN
    assert bs.phi_e[Edge.RIGHT].bc_type == BCType.NEUMANN

    # get_bc helper
    assert bs.get_bc('phi_e', Edge.TOP).bc_type == BCType.DIRICHLET
    assert bs.get_bc('Vm', Edge.TOP).bc_type == BCType.NEUMANN

    # Bath-coupled edges with string names
    bs2 = BoundarySpec.bath_coupled_edges(['top', 'bottom'])
    assert bs2.phi_e[Edge.TOP].bc_type == BCType.DIRICHLET


# === 1-T5: Ionic model reuse ===

def test_ionic_model_reuse():
    """1-T5: Import TTP06, call compute_Iion with bidomain state.Vm."""
    from cardiac_sim.ionic import TTP06Model

    model = TTP06Model(device='cpu')
    n_cells = 10
    V = torch.full((n_cells,), model.V_rest, dtype=torch.float64)
    ionic_states = model.get_initial_state(n_cells)

    Iion = model.compute_Iion(V, ionic_states)
    assert Iion.shape == (n_cells,)
    # At rest, Iion should be near zero
    assert torch.abs(Iion).max().item() < 5.0  # Less than 5 uA/uF at rest


# === 1-T6: Device abstraction ===

def test_device_abstraction():
    """1-T6: Backend CPU detection works."""
    from cardiac_sim.utils.backend import Backend

    backend = Backend(device='cpu', verbose=False)
    assert backend.device.type == 'cpu'
    assert backend.is_cpu is True
    assert backend.is_cuda is False

    # Tensor creation
    x = backend.zeros(10)
    assert x.shape == (10,)
    assert x.device.type == 'cpu'


# === Additional: StructuredGrid + BoundarySpec integration ===

def test_structured_grid_boundary_spec():
    """Verify StructuredGrid stores and uses BoundarySpec."""
    from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
    from cardiac_sim.tissue_builder.mesh.boundary import BoundarySpec, BCType, Edge

    grid = StructuredGrid.create_rectangle(Lx=1.0, Ly=1.0, Nx=10, Ny=10)

    # Default: insulated
    assert grid.boundary_spec.phi_e_has_null_space is True

    # Set bath-coupled
    grid.boundary_spec = BoundarySpec.bath_coupled()
    assert grid.boundary_spec.phi_e_has_null_space is False

    # Dirichlet mask
    dmask = grid.dirichlet_mask_phi_e
    assert dmask.shape == (10, 10)
    # All boundary nodes should be Dirichlet
    assert dmask[0, :].all()
    assert dmask[-1, :].all()
    assert dmask[:, 0].all()
    assert dmask[:, -1].all()
    # Interior nodes should NOT be Dirichlet
    assert not dmask[5, 5].item()

    # Edge masks
    em = grid.edge_masks
    assert em[Edge.LEFT][0, 5].item() is True
    assert em[Edge.LEFT][5, 5].item() is False

    # Neumann mask (should be empty for all-Dirichlet)
    nmask = grid.neumann_mask_phi_e
    assert nmask.sum().item() == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
