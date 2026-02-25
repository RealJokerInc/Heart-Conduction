"""
Phase 8 Validation Tests — Per-Node Conductivity

Tests 8-V1 through 8-V7.
"""

import sys
import os
import tempfile
import numpy as np
import torch

# Ensure Engine_V5.4 is on path
sys.path.insert(0, os.path.dirname(__file__))

from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
from cardiac_sim.simulation.classical.discretization_scheme.fvm import FVMDiscretization
from cardiac_sim.simulation.classical.discretization_scheme.base import sparse_mv


def test_8v1_fdm_uniform_pernode_matches_scalar():
    """8-V1: FDM uniform per-node D produces identical Laplacian as scalar D."""
    D = 0.001
    grid = StructuredGrid.create_rectangle(1.0, 1.0, 21, 21)

    # Scalar D
    fdm_scalar = FDMDiscretization(grid, D=D, chi=1.0, Cm=1.0)

    # Per-node D (uniform, same value)
    Dxx = torch.full((21, 21), D, dtype=torch.float64)
    Dxy = torch.zeros(21, 21, dtype=torch.float64)
    Dyy = torch.full((21, 21), D, dtype=torch.float64)
    fdm_pernode = FDMDiscretization(grid, D_field=(Dxx, Dxy, Dyy), chi=1.0, Cm=1.0)

    # Compare Laplacians
    L_scalar = fdm_scalar.L.to_dense()
    L_pernode = fdm_pernode.L.to_dense()
    diff = (L_scalar - L_pernode).abs().max().item()

    assert diff == 0.0, f"FDM Laplacians differ: max diff = {diff}"
    assert fdm_scalar.n_dof == fdm_pernode.n_dof

    # Also test with a random V vector
    V = torch.randn(grid.n_dof, dtype=torch.float64)
    dVdt_scalar = fdm_scalar.apply_diffusion(V)
    dVdt_pernode = fdm_pernode.apply_diffusion(V)
    v_diff = (dVdt_scalar - dVdt_pernode).abs().max().item()
    assert v_diff < 1e-14, f"apply_diffusion diff = {v_diff}"

    print(f"  Laplacian max diff: {diff}")
    print(f"  apply_diffusion max diff: {v_diff}")
    print("  PASS")


def test_8v2_fvm_uniform_pernode_matches_scalar():
    """8-V2: FVM uniform per-node D produces identical flux matrix as scalar D."""
    D = 0.001
    grid = StructuredGrid.create_rectangle(1.0, 1.0, 21, 21)

    # Scalar D
    fvm_scalar = FVMDiscretization(grid, D=D, chi=1.0, Cm=1.0)

    # Per-node D (uniform, same value)
    D_field = torch.full((21, 21), D, dtype=torch.float64)
    fvm_pernode = FVMDiscretization(grid, D_field=D_field, chi=1.0, Cm=1.0)

    # Also test anisotropic tuple with same D for both
    fvm_aniso = FVMDiscretization(grid, D_field=(D_field, D_field), chi=1.0, Cm=1.0)

    F_scalar = fvm_scalar.F.to_dense()
    F_pernode = fvm_pernode.F.to_dense()
    F_aniso = fvm_aniso.F.to_dense()

    diff1 = (F_scalar - F_pernode).abs().max().item()
    diff2 = (F_scalar - F_aniso).abs().max().item()

    assert diff1 == 0.0, f"FVM scalar vs per-node diff = {diff1}"
    assert diff2 == 0.0, f"FVM scalar vs aniso diff = {diff2}"

    print(f"  Scalar vs per-node max diff: {diff1}")
    print(f"  Scalar vs aniso max diff: {diff2}")
    print("  PASS")


def test_8v3_fdm_scar_blocks_diffusion():
    """8-V3: FDM scar strip (D=0) blocks diffusion completely."""
    # 21x21 grid with horizontal scar strip in the middle
    nx, ny = 21, 21
    D = 0.001
    grid = StructuredGrid.create_rectangle(1.0, 1.0, nx, ny)

    Dxx = torch.full((nx, ny), D, dtype=torch.float64)
    Dxy = torch.zeros(nx, ny, dtype=torch.float64)
    Dyy = torch.full((nx, ny), D, dtype=torch.float64)

    # Scar strip: nodes i=9,10,11 (3 rows in x-direction)
    Dxx[9:12, :] = 0.0
    Dyy[9:12, :] = 0.0

    fdm = FDMDiscretization(grid, D_field=(Dxx, Dxy, Dyy), chi=1.0, Cm=1.0)

    # Initial condition: step function — V=1 for i < 9, V=0 elsewhere
    V = torch.zeros(grid.n_dof, dtype=torch.float64)
    V_grid_init = V.reshape(nx, ny)
    V_grid_init[:9, :] = 1.0
    V = V_grid_init.flatten()

    # Run 100 explicit FE steps
    dt = 0.001
    for _ in range(100):
        dVdt = fdm.apply_diffusion(V)
        V = V + dt * dVdt

    # Check: voltage inside scar strip should remain zero
    V_grid = V.reshape(nx, ny)
    scar_max = V_grid[9:12, :].abs().max().item()
    # Right side (i >= 12) should also be zero (no diffusion past scar)
    right_max = V_grid[12:, :].abs().max().item()

    assert scar_max < 1e-10, f"Voltage leaked into scar: max = {scar_max}"
    assert right_max < 1e-10, f"Voltage leaked past scar: max = {right_max}"

    # Left side should have nonzero voltage
    left_max = V_grid[:9, :].abs().max().item()
    assert left_max > 0.1, f"Left side voltage too small: {left_max}"

    print(f"  Scar max V: {scar_max:.2e}")
    print(f"  Right-of-scar max V: {right_max:.2e}")
    print(f"  Left-of-scar max V: {left_max:.4f}")
    print("  PASS")


def test_8v4_fvm_scar_blocks_diffusion():
    """8-V4: FVM scar strip (D=0) blocks diffusion via harmonic mean → D_face=0."""
    nx, ny = 21, 21
    D = 0.001
    grid = StructuredGrid.create_rectangle(1.0, 1.0, nx, ny)

    D_field = torch.full((nx, ny), D, dtype=torch.float64)
    # Scar strip
    D_field[9:12, :] = 0.0

    fvm = FVMDiscretization(grid, D_field=D_field, chi=1.0, Cm=1.0)

    # Initial condition: step function — V=1 for i < 9, V=0 elsewhere
    V = torch.zeros(grid.n_dof, dtype=torch.float64)
    V_grid_init = V.reshape(nx, ny)
    V_grid_init[:9, :] = 1.0
    V = V_grid_init.flatten()

    dt = 0.001
    for _ in range(100):
        dVdt = fvm.apply_diffusion(V)
        V = V + dt * dVdt

    V_grid = V.reshape(nx, ny)
    scar_max = V_grid[9:12, :].abs().max().item()
    right_max = V_grid[12:, :].abs().max().item()

    assert scar_max < 1e-10, f"FVM: Voltage leaked into scar: max = {scar_max}"
    assert right_max < 1e-10, f"FVM: Voltage leaked past scar: max = {right_max}"

    left_max = V_grid[:9, :].abs().max().item()
    assert left_max > 0.1, f"Left side voltage too small: {left_max}"

    print(f"  Scar max V: {scar_max:.2e}")
    print(f"  Right-of-scar max V: {right_max:.2e}")
    print(f"  Left-of-scar max V: {left_max:.4f}")
    print("  PASS")


def test_8v5_anisotropic_propagation():
    """8-V5: Anisotropic D produces elliptical diffusion pattern."""
    nx, ny = 41, 41
    grid = StructuredGrid.create_rectangle(2.0, 2.0, nx, ny)

    # Fiber along x: D_fiber = 4*D_cross
    D_fiber = 0.1
    D_cross = 0.025

    Dxx = torch.full((nx, ny), D_fiber, dtype=torch.float64)
    Dxy = torch.zeros(nx, ny, dtype=torch.float64)
    Dyy = torch.full((nx, ny), D_cross, dtype=torch.float64)

    fdm = FDMDiscretization(grid, D_field=(Dxx, Dxy, Dyy), chi=1.0, Cm=1.0)

    # Point source at center (narrow Gaussian)
    x, y = grid.coordinates
    V = torch.exp(-((x - 1.0) ** 2 + (y - 1.0) ** 2) / 0.002).to(torch.float64)

    # Run enough steps for diffusion to spread several nodes
    dt = 0.0005
    for _ in range(2000):
        dVdt = fdm.apply_diffusion(V)
        V = V + dt * dVdt

    # Measure spread in x vs y direction
    V_grid = V.reshape(nx, ny)
    cx, cy = nx // 2, ny // 2

    # Find extent where V > threshold along x and y axes
    threshold = V_grid[cx, cy].item() * 0.1
    x_extent = 0
    for di in range(1, cx):
        if V_grid[cx + di, cy].item() > threshold:
            x_extent = di
    y_extent = 0
    for dj in range(1, cy):
        if V_grid[cx, cy + dj].item() > threshold:
            y_extent = dj

    # With D_fiber/D_cross = 4, spread ratio should be ~2
    # (diffusion distance ~ sqrt(D*t))
    if y_extent > 0:
        ratio = x_extent / y_extent
    else:
        ratio = float('inf')

    assert x_extent > 2, f"x_extent too small: {x_extent} (need diffusion to spread)"
    assert y_extent > 1, f"y_extent too small: {y_extent} (need diffusion to spread)"
    assert ratio > 1.3, f"Anisotropy ratio too small: {ratio:.2f} (expected > 1.3)"
    assert ratio < 3.0, f"Anisotropy ratio too large: {ratio:.2f} (expected < 3.0)"

    print(f"  x_extent: {x_extent}, y_extent: {y_extent}, ratio: {ratio:.2f}")
    print(f"  Expected ratio ~2.0 for D_fiber/D_cross=4")
    print("  PASS")


def test_8v6_end_to_end_scar_simulation():
    """8-V6: End-to-end: mesh.npz with scar → FDM per-node D → MonodomainSimulation."""
    from mesh_builder import MeshBuilderSession
    from mesh_builder.export import export_mesh
    from cardiac_sim.tissue_builder.mesh.loader import load_mesh
    from cardiac_sim.simulation.classical import MonodomainSimulation
    from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol

    # Create image: tissue (red) with scar strip (green)
    img = np.zeros((40, 40, 4), dtype=np.uint8)
    img[:, :, 3] = 255
    img[:, :, :3] = [255, 0, 0]            # all red tissue
    img[15:25, :, :3] = [0, 255, 0]        # green scar strip

    session = MeshBuilderSession()
    session.image_array = img
    session.image_size = (40, 40)
    session.detect_colors()

    for color, group in session.color_groups.items():
        if group.is_background:
            continue
        if color[1] == 255:  # green = scar
            session.configure_group(color, 'Scar', 'non_conductive', 0.0, 0.0, 0.0)
        else:  # red = tissue
            session.configure_group(color, 'Myocardial', 'myocardial', 0.001, 0.001, 0.0)

    session.set_dimensions(0.4, 0.4, 0.01)

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        mesh_path = f.name

    try:
        export_mesh(session, mesh_path)
        mesh_data = load_mesh(mesh_path)
        grid = mesh_data.grid

        # Build FDM with per-node D from loader
        fdm = FDMDiscretization(
            grid,
            D_field=(mesh_data.D_xx_grid, mesh_data.D_xy_grid, mesh_data.D_yy_grid),
            chi=1400.0,
            Cm=1.0,
        )

        # Stimulus on the left edge — exclude scar nodes (D_xx=0)
        stimulus = StimulusProtocol()
        x, y = grid.coordinates
        left_mask = (x < 0.05) & (mesh_data.D_xx > 0)
        stimulus.add_stimulus(region=left_mask, start_time=0.0, duration=1.0, amplitude=-52.0)

        sim = MonodomainSimulation(
            spatial=fdm,
            ionic_model='ttp06',
            stimulus=stimulus,
            dt=0.02,
            splitting='godunov',
            ionic_solver='rush_larsen',
            diffusion_solver='forward_euler',
            linear_solver='none',
        )

        # Run for 2ms
        for _ in range(100):
            sim.step(0.02)

        V = sim.get_voltage()
        V_min = float(V.min())
        V_max = float(V.max())

        # Stimulus should initiate depolarization
        assert V_max > -50.0, f"No depolarization: V_max={V_max:.1f} mV"

        # Check that scar region has NOT depolarized
        # Need to identify scar nodes: D_xx = 0
        scar_mask = mesh_data.D_xx == 0.0
        if scar_mask.sum() > 0:
            V_scar_max = float(V[scar_mask].max())
            V_scar_min = float(V[scar_mask].min())
            # Scar should remain near resting potential
            assert V_scar_max < -80.0, f"Scar depolarized: V_scar_max={V_scar_max:.1f} mV"

        print(f"  Grid: {grid}")
        print(f"  After 2ms: V_min={V_min:.1f} mV, V_max={V_max:.1f} mV")
        if scar_mask.sum() > 0:
            print(f"  Scar region: V_min={V_scar_min:.1f}, V_max={V_scar_max:.1f} mV")
        print("  PASS")
    finally:
        os.unlink(mesh_path)


def test_8v7_backward_compatibility():
    """8-V7: Scalar D path still works correctly (no regression)."""
    # Test FDM with scalar D — basic convergence check
    D = 0.001

    errors = []
    for n in [11, 21, 41]:
        grid = StructuredGrid.create_rectangle(1.0, 1.0, n, n)
        fdm = FDMDiscretization(grid, D=D, chi=1.0, Cm=1.0)

        # Apply to cos(pi*x)*cos(pi*y) — should give -2*pi^2*D*cos(pi*x)*cos(pi*y)
        x, y = grid.coordinates
        V = torch.cos(np.pi * x) * torch.cos(np.pi * y)
        expected = -2.0 * np.pi ** 2 * D * V

        LV = sparse_mv(fdm.L, V)
        err = (LV - expected).abs().max().item()
        errors.append(err)

    # Should converge at O(h^2)
    ratio1 = errors[0] / errors[1] if errors[1] > 0 else 0
    ratio2 = errors[1] / errors[2] if errors[2] > 0 else 0

    assert ratio1 > 3.0, f"FDM convergence ratio 1 too low: {ratio1:.2f}"
    assert ratio2 > 3.0, f"FDM convergence ratio 2 too low: {ratio2:.2f}"

    # Test FVM with scalar D — row sums = 0
    grid = StructuredGrid.create_rectangle(1.0, 1.0, 11, 11)
    fvm = FVMDiscretization(grid, D=D, chi=1.0, Cm=1.0)
    F_dense = fvm.F.to_dense()
    row_sums = F_dense.sum(dim=1)
    max_row_sum = row_sums.abs().max().item()
    assert max_row_sum < 1e-14, f"FVM row sums nonzero: {max_row_sum}"

    # Test FDM row sums = 0
    fdm = FDMDiscretization(grid, D=D, chi=1.0, Cm=1.0)
    L_dense = fdm.L.to_dense()
    fdm_row_sums = L_dense.sum(dim=1)
    max_fdm_row = fdm_row_sums.abs().max().item()
    assert max_fdm_row < 1e-14, f"FDM row sums nonzero: {max_fdm_row}"

    print(f"  FDM convergence: errors = {[f'{e:.2e}' for e in errors]}")
    print(f"  FDM convergence ratios: {ratio1:.2f}, {ratio2:.2f}")
    print(f"  FDM row sums max: {max_fdm_row:.2e}")
    print(f"  FVM row sums max: {max_row_sum:.2e}")
    print("  PASS")


if __name__ == '__main__':
    tests = [
        ("8-V1: FDM uniform per-node D matches scalar D", test_8v1_fdm_uniform_pernode_matches_scalar),
        ("8-V2: FVM uniform per-node D matches scalar D", test_8v2_fvm_uniform_pernode_matches_scalar),
        ("8-V3: FDM scar blocks diffusion", test_8v3_fdm_scar_blocks_diffusion),
        ("8-V4: FVM scar blocks diffusion", test_8v4_fvm_scar_blocks_diffusion),
        ("8-V5: Anisotropic propagation", test_8v5_anisotropic_propagation),
        ("8-V6: End-to-end scar → simulation", test_8v6_end_to_end_scar_simulation),
        ("8-V7: Backward compatibility", test_8v7_backward_compatibility),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n{name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
