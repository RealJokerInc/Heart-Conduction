"""Tests for cardiac_core.api.bidomain() — simplified bidomain API."""

import numpy as np
import pytest
import torch

from cardiac_core import bidomain, create_cardiac_mesh, save_cardiac_mesh
from cardiac_core.file_format import CardiacMeshData


class TestBidomainFromData:
    """Test bidomain() with CardiacMeshData directly."""

    def test_basic_run_ratio_fallback(self):
        """Create mesh without sigma → bidomain derives D_i/D_e from ratio → runs."""
        mesh = create_cardiac_mesh(Lx=1.0, Ly=0.5, dx=0.05, D=0.001, chi=1.0)
        sim = bidomain(mesh, device='cpu')

        snapshots = list(sim.snapshots(5.0, save_every=1.0))
        assert len(snapshots) >= 4

        Nx = round(1.0 / 0.05) + 1
        Ny = round(0.5 / 0.05) + 1
        for snap in snapshots:
            assert snap.V.shape == (Nx, Ny)
            assert snap.phi_e is not None
            assert snap.phi_e.shape == (Nx, Ny)

    def test_stimulus_activates(self):
        """Stimulus should depolarize tissue and produce phi_e."""
        mesh = create_cardiac_mesh(
            Lx=1.0, Ly=0.5, dx=0.05,
            stim_amplitude=-80.0,
            stim_start=0.0,
            stim_duration=2.0,
        )
        sim = bidomain(mesh, device='cpu')

        for snap in sim.snapshots(5.0, save_every=1.0):
            pass
        V_max = snap.V.max().item()
        assert V_max > -50.0, f"V_max={V_max}, stimulus didn't activate"

        # phi_e should be non-trivially different from zero
        phi_e_range = snap.phi_e.max().item() - snap.phi_e.min().item()
        assert phi_e_range > 0.1, f"phi_e range={phi_e_range}, expected significant phi_e variation"


class TestBidomainWithSigma:
    """Test bidomain() when file provides sigma_i and sigma_e."""

    def test_with_sigma_fields(self):
        """Provide sigma_i/sigma_e directly → bidomain uses them."""
        Nx, Ny = 21, 11
        mesh = create_cardiac_mesh(Lx=1.0, Ly=0.5, dx=0.05, D=0.001, chi=1.0)

        # Add sigma fields
        chi_Cm = mesh.chi * mesh.Cm
        mesh.sigma_i = (
            np.full((Nx, Ny), 1.74),   # xx
            np.full((Nx, Ny), 0.174),   # yy
            np.zeros((Nx, Ny)),          # xy
        )
        mesh.sigma_e = (
            np.full((Nx, Ny), 6.25),    # xx
            np.full((Nx, Ny), 2.36),     # yy
            np.zeros((Nx, Ny)),          # xy
        )

        sim = bidomain(mesh, device='cpu')
        snapshots = list(sim.snapshots(3.0, save_every=1.0))
        assert len(snapshots) >= 2
        assert snapshots[0].phi_e is not None


class TestBidomainFromFile:
    """Test bidomain() with .npz file path."""

    def test_from_file(self, tmp_path):
        mesh = create_cardiac_mesh(Lx=1.0, Ly=0.5, dx=0.05)
        path = str(tmp_path / "mesh.npz")
        save_cardiac_mesh(path, mesh)

        sim = bidomain(path, device='cpu')
        snapshots = list(sim.snapshots(3.0, save_every=1.0))
        assert len(snapshots) >= 2
        assert snapshots[0].phi_e is not None


class TestBidomainBoundary:
    """Test boundary condition override."""

    def test_insulated_default(self):
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.05)
        sim = bidomain(mesh, device='cpu')
        # Should not raise
        list(sim.snapshots(2.0, save_every=2.0))

    def test_bath_override(self):
        mesh = create_cardiac_mesh(Lx=0.5, Ly=0.5, dx=0.05)
        sim = bidomain(mesh, boundary='bath', device='cpu')
        snapshots = list(sim.snapshots(2.0, save_every=2.0))
        assert len(snapshots) >= 1


class TestBidomainMatchesDirect:
    """Verify wrapper produces same result as direct Bidomain V1 construction."""

    def test_matches_direct(self):
        """Same params → wrapper vs direct → identical V and phi_e after 3ms."""
        from cardiac_core._bidomain import (
            BidomainSimulation, BidomainFDMDiscretization, BidomainConductivity,
        )
        from cardiac_core.mesh.structured import StructuredGrid
        from cardiac_core.mesh.boundary import BoundarySpec
        from cardiac_core.stimulus.protocol import StimulusProtocol

        dx = 0.05
        Lx, Ly = 1.0, 0.5
        Nx = round(Lx / dx) + 1
        Ny = round(Ly / dx) + 1

        # D_eff and sigma_ratio → D_i, D_e
        D_eff = 0.001
        r = 3.59
        D_i = D_eff * (1 + r) / r
        D_e = D_eff * (1 + r)

        # --- Direct construction ---
        grid_direct = StructuredGrid.create_rectangle(Lx, Ly, Nx, Ny, device='cpu')
        grid_direct.boundary_spec = BoundarySpec.insulated()
        cond = BidomainConductivity(D_i=D_i, D_e=D_e)
        spatial_direct = BidomainFDMDiscretization(grid_direct, cond, Cm=1.0)

        x, y = grid_direct.coordinates
        stim_direct = StimulusProtocol()
        stim_direct.add_stimulus(
            region=(x < 0.1),
            start_time=1.0,
            duration=2.0,
            amplitude=-80.0,
        )

        sim_direct = BidomainSimulation(
            spatial=spatial_direct,
            ionic_model='ttp06',
            stimulus=stim_direct,
            dt=0.02,
            elliptic_solver='auto',
            theta=0.5,
            device='cpu',
        )

        # --- Wrapper construction ---
        mesh = create_cardiac_mesh(Lx=Lx, Ly=Ly, dx=dx, D=D_eff, chi=1.0)  # D_eff is effective
        sim_wrapper = bidomain(mesh, sigma_ratio=r, device='cpu')

        # Run both for 3ms
        Vm_direct = phi_e_direct = None
        for state in sim_direct.run(3.0, save_every=3.0):
            Vm_direct = state.Vm.clone()
            phi_e_direct = state.phi_e.clone()

        V_wrapper = phi_e_wrapper = None
        for snap in sim_wrapper.snapshots(3.0, save_every=3.0):
            V_wrapper = snap.V.clone()
            phi_e_wrapper = snap.phi_e.clone()

        Vm_direct_grid = grid_direct.flat_to_grid(Vm_direct)
        phi_e_direct_grid = grid_direct.flat_to_grid(phi_e_direct)

        assert torch.allclose(V_wrapper, Vm_direct_grid, atol=1e-10), \
            f"Vm max diff: {(V_wrapper - Vm_direct_grid).abs().max()}"
        assert torch.allclose(phi_e_wrapper, phi_e_direct_grid, atol=1e-10), \
            f"phi_e max diff: {(phi_e_wrapper - phi_e_direct_grid).abs().max()}"
