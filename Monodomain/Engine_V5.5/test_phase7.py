"""
Phase 7 Validation Tests — Builder Integration Backend Pipeline

Tests 7-V1 through 7-V7.
"""

import sys
import os
import tempfile
import numpy as np

# Ensure Engine_V5.4 is on path
sys.path.insert(0, os.path.dirname(__file__))


def test_7v1_mesh_export_roundtrip():
    """7-V1: mesh_builder loads image → exports .npz → verify keys, shapes, dtypes."""
    from mesh_builder import MeshBuilderSession
    from mesh_builder.export import export_mesh

    # Create a synthetic test image: 100x80 with 2 colors
    # White background (255,255,255,255) and red tissue (255,0,0,255)
    img = np.ones((80, 100, 4), dtype=np.uint8) * 255  # all white
    img[10:70, 10:90, :] = [255, 0, 0, 255]  # red rectangle in center

    session = MeshBuilderSession()
    session.image_array = img
    session.image_size = (100, 80)

    # Detect and configure
    session.detect_colors()
    for color, group in session.color_groups.items():
        if group.is_background:
            continue
        session.configure_group(color, 'Myocardial', 'myocardial', 0.001, 0.0003, 0.0)

    session.set_dimensions(1.0, 0.8, 0.01)

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        path = f.name

    try:
        export_mesh(session, path)
        data = np.load(path, allow_pickle=True)

        # Check all expected keys exist
        expected_keys = ['mask', 'dx', 'dy', 'Lx', 'Ly', 'D_xx', 'D_yy', 'D_xy',
                         'label_map', 'group_labels', 'group_cell_types']
        for key in expected_keys:
            assert key in data, f"Missing key: {key}"

        nx, ny = session.get_mesh_resolution()
        assert data['mask'].shape == (nx, ny), f"mask shape {data['mask'].shape} != ({nx}, {ny})"
        assert data['mask'].dtype == bool
        assert data['D_xx'].shape == (nx, ny)
        assert data['D_xx'].dtype == np.float64
        assert float(data['dx']) == 0.01
        assert data['mask'].sum() > 0, "No active tissue pixels"

        print(f"  mask shape: {data['mask'].shape}, active: {data['mask'].sum()}/{data['mask'].size}")
        print("  PASS")
    finally:
        os.unlink(path)


def test_7v2_mesh_loader_roundtrip():
    """7-V2: Load mesh.npz → StructuredGrid with correct n_dof, dx, mask."""
    import torch
    from mesh_builder import MeshBuilderSession
    from mesh_builder.export import export_mesh
    from cardiac_sim.tissue_builder.mesh.loader import load_mesh

    # Create a simple test image
    img = np.ones((50, 50, 4), dtype=np.uint8) * 255
    img[5:45, 5:45, :] = [0, 0, 255, 255]  # blue tissue

    session = MeshBuilderSession()
    session.image_array = img
    session.image_size = (50, 50)
    session.detect_colors()

    for color, group in session.color_groups.items():
        if not group.is_background:
            session.configure_group(color, 'Tissue', 'myocardial', 0.001, 0.0003, 0.0)

    session.set_dimensions(0.5, 0.5, 0.01)

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        path = f.name

    try:
        export_mesh(session, path)
        mesh_data = load_mesh(path)

        grid = mesh_data.grid
        assert grid.n_dof > 0, "n_dof should be > 0"
        assert grid.dx > 0
        assert mesh_data.D_xx.shape == (grid.n_dof,)
        assert mesh_data.D_yy.shape == (grid.n_dof,)
        assert mesh_data.D_xy.shape == (grid.n_dof,)

        # Verify coordinates exist and have right shape
        x, y = grid.coordinates
        assert x.shape == (grid.n_dof,)
        assert y.shape == (grid.n_dof,)

        print(f"  grid: {grid}")
        print(f"  n_dof: {grid.n_dof}, D_xx range: [{mesh_data.D_xx.min():.6f}, {mesh_data.D_xx.max():.6f}]")
        print("  PASS")
    finally:
        os.unlink(path)


def test_7v3_conductivity_mapping():
    """7-V3: Multiple tissue groups with different D → verify per-node mapping."""
    import torch
    from mesh_builder import MeshBuilderSession
    from mesh_builder.export import export_mesh
    from cardiac_sim.tissue_builder.mesh.loader import load_mesh

    # Image with 3 regions: background (white), tissue1 (red), tissue2 (blue)
    img = np.ones((60, 60, 4), dtype=np.uint8) * 255  # white bg
    img[5:30, 5:55, :] = [255, 0, 0, 255]   # red: top half
    img[30:55, 5:55, :] = [0, 0, 255, 255]  # blue: bottom half

    session = MeshBuilderSession()
    session.image_array = img
    session.image_size = (60, 60)
    session.detect_colors()

    for color, group in session.color_groups.items():
        if group.is_background:
            continue
        if color[0] == 255 and color[2] == 0:  # red
            session.configure_group(color, 'Fast', 'myocardial', 0.002, 0.001, 0.0)
        else:  # blue
            session.configure_group(color, 'Slow', 'endocardial', 0.0005, 0.0002, 0.0)

    session.set_dimensions(0.6, 0.6, 0.01)

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        path = f.name

    try:
        export_mesh(session, path)
        mesh_data = load_mesh(path)

        # Should have nodes with D=0.002 and nodes with D=0.0005
        unique_dxx = torch.unique(mesh_data.D_xx)
        assert len(unique_dxx) >= 2, f"Expected ≥2 unique D_xx values, got {len(unique_dxx)}: {unique_dxx}"

        has_fast = (mesh_data.D_xx == 0.002).any()
        has_slow = (mesh_data.D_xx == 0.0005).any()
        assert has_fast, "Missing fast tissue (D_xx=0.002)"
        assert has_slow, "Missing slow tissue (D_xx=0.0005)"

        print(f"  Unique D_xx values: {unique_dxx.tolist()}")
        print(f"  Fast nodes: {(mesh_data.D_xx == 0.002).sum()}, Slow nodes: {(mesh_data.D_xx == 0.0005).sum()}")
        print("  PASS")
    finally:
        os.unlink(path)


def test_7v4_scar_region_d_zero():
    """7-V4: Non-conductive scar region has D=0."""
    import torch
    from mesh_builder import MeshBuilderSession
    from mesh_builder.export import export_mesh
    from cardiac_sim.tissue_builder.mesh.loader import load_mesh

    # Image: tissue (red) with scar strip (green)
    img = np.zeros((40, 40, 4), dtype=np.uint8)
    img[:, :, 3] = 255
    img[:, :, :3] = [255, 0, 0]            # all red tissue
    img[15:25, :, :3] = [0, 255, 0]        # green scar strip

    # Add white border as background
    img[0, :] = [255, 255, 255, 255]
    img[-1, :] = [255, 255, 255, 255]
    img[:, 0] = [255, 255, 255, 255]
    img[:, -1] = [255, 255, 255, 255]

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
            session.configure_group(color, 'Myocardial', 'myocardial', 0.001, 0.0003, 0.0)

    session.set_dimensions(0.4, 0.4, 0.01)

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        path = f.name

    try:
        export_mesh(session, path)
        mesh_data = load_mesh(path)

        scar_mask = mesh_data.D_xx == 0.0
        tissue_mask = mesh_data.D_xx > 0.0

        assert scar_mask.sum() > 0, "No scar nodes (D=0)"
        assert tissue_mask.sum() > 0, "No tissue nodes (D>0)"
        # Scar should also have D_yy = 0 and D_xy = 0
        assert (mesh_data.D_yy[scar_mask] == 0).all(), "Scar D_yy != 0"
        assert (mesh_data.D_xy[scar_mask] == 0).all(), "Scar D_xy != 0"

        print(f"  Scar nodes: {scar_mask.sum()}, Tissue nodes: {tissue_mask.sum()}")
        print("  PASS")
    finally:
        os.unlink(path)


def test_7v5_stim_export_roundtrip():
    """7-V5: stim_builder exports → verify .npz keys and shapes."""
    from stim_builder import StimBuilderSession
    from stim_builder.models import StimType, StimTarget
    from stim_builder.export import export_stim

    # Create a synthetic image with 2 stimulus regions
    img = np.ones((50, 50, 4), dtype=np.uint8) * 255  # white bg
    img[5:15, 5:15, :] = [255, 0, 0, 255]   # red = S1
    img[35:45, 35:45, :] = [0, 255, 0, 255]  # green = S2

    session = StimBuilderSession()
    session.image_array = img
    session.image_size = (50, 50)
    session.detect_colors()

    # Configure regions
    for color, region in session.stim_regions.items():
        if region.is_background:
            continue
        if color[0] == 255:  # red = S1
            region.configure(
                label='S1', stim_type=StimType.CURRENT_INJECTION,
                amplitude=-52.0, duration=1.0, start_time=0.0, bcl=1000.0, num_pulses=5
            )
        else:  # green = S2
            region.configure(
                label='S2', stim_type=StimType.CURRENT_INJECTION,
                amplitude=-52.0, duration=1.0, start_time=300.0, bcl=1000.0, num_pulses=1
            )

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        path = f.name

    try:
        export_stim(session, path)
        data = np.load(path, allow_pickle=True)

        n_regions = int(data['n_regions'])
        assert n_regions == 2, f"Expected 2 regions, got {n_regions}"

        for i in range(n_regions):
            assert f'mask_{i}' in data
            assert f'label_{i}' in data
            assert f'amplitude_{i}' in data
            assert f'bcl_{i}' in data
            assert data[f'mask_{i}'].shape == (50, 50)

        print(f"  {n_regions} regions exported")
        for i in range(n_regions):
            print(f"    Region {i}: {str(data[f'label_{i}'])}, amp={float(data[f'amplitude_{i}'])}, bcl={float(data[f'bcl_{i}'])}")
        print("  PASS")
    finally:
        os.unlink(path)


def test_7v6_stim_loader_roundtrip():
    """7-V6: stim.npz + mesh mask → StimulusProtocol with correct masks and timing."""
    from stim_builder import StimBuilderSession
    from stim_builder.models import StimType
    from stim_builder.export import export_stim
    from cardiac_sim.tissue_builder.stimulus.loader import load_stimulus

    # Create image: full tissue with one stim region
    img = np.zeros((30, 30, 4), dtype=np.uint8)
    img[:, :, :] = [0, 0, 255, 255]       # all blue tissue
    img[2:8, 2:8, :] = [255, 0, 0, 255]   # red stim region

    session = StimBuilderSession()
    session.image_array = img
    session.image_size = (30, 30)
    session.detect_colors()

    for color, region in session.stim_regions.items():
        if region.is_background:
            continue
        if color[0] == 255:  # red = stim
            region.configure(
                label='S1', stim_type=StimType.CURRENT_INJECTION,
                amplitude=-52.0, duration=1.0, start_time=0.0, bcl=500.0, num_pulses=3
            )
        else:
            session.mark_as_background(color)

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        stim_path = f.name

    try:
        export_stim(session, stim_path)

        # Create a mesh mask that covers the whole image (all tissue)
        mesh_mask = np.ones((30, 30), dtype=bool)

        protocol = load_stimulus(stim_path, mesh_mask)
        assert len(protocol.stimuli) > 0, "No stimuli loaded"

        # Regular pacing with 3 beats at bcl=500 should produce 3 stimuli
        assert len(protocol.stimuli) == 3, f"Expected 3 stimuli, got {len(protocol.stimuli)}"

        # Check timing
        times = sorted([s.start_time for s in protocol.stimuli])
        assert times == [0.0, 500.0, 1000.0], f"Unexpected times: {times}"

        print(f"  {len(protocol.stimuli)} stimuli loaded")
        print(f"  Times: {times}")
        print("  PASS")
    finally:
        os.unlink(stim_path)


def test_7v7_end_to_end_npz_to_simulation():
    """7-V7: .npz → loaders → FDM → MonodomainSimulation → AP initiates."""
    import torch
    from mesh_builder import MeshBuilderSession
    from mesh_builder.export import export_mesh
    from cardiac_sim.tissue_builder.mesh.loader import load_mesh
    from cardiac_sim.simulation.classical.discretization_scheme import FDMDiscretization
    from cardiac_sim.simulation.classical import MonodomainSimulation
    from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol

    # Create a small uniform tissue (all same color, no background needed)
    # 40x40 pixels, single tissue type
    img = np.zeros((40, 40, 4), dtype=np.uint8)
    img[:, :, :] = [200, 100, 50, 255]  # uniform tissue color

    session = MeshBuilderSession()
    session.image_array = img
    session.image_size = (40, 40)
    session.detect_colors()

    for color, group in session.color_groups.items():
        if not group.is_background:
            session.configure_group(color, 'Tissue', 'myocardial', 0.001, 0.001, 0.0)

    session.set_dimensions(0.4, 0.4, 0.01)

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        mesh_path = f.name

    try:
        export_mesh(session, mesh_path)
        mesh_data = load_mesh(mesh_path)
        grid = mesh_data.grid

        # Use scalar D for now (Phase 8 will add per-node D)
        D_avg = float(mesh_data.D_xx.mean())

        # Build discretization with scalar D
        from cardiac_sim.tissue_builder.tissue.isotropic import IsotropicTissue
        tissue = IsotropicTissue(D=D_avg)
        spatial = FDMDiscretization(grid, D=tissue.D, chi=tissue.chi, Cm=tissue.Cm)

        # Simple stimulus: left edge
        stimulus = StimulusProtocol()
        x, y = grid.coordinates
        left_mask = x < 0.05
        stimulus.add_stimulus(region=left_mask, start_time=0.0, duration=1.0, amplitude=-52.0)

        # Create and run simulation for 2ms
        sim = MonodomainSimulation(
            spatial=spatial,
            ionic_model='ttp06',
            stimulus=stimulus,
            dt=0.02,
            splitting='godunov',
            ionic_solver='rush_larsen',
            diffusion_solver='forward_euler',
            linear_solver='none',
        )

        # Run a few steps
        for _ in range(100):
            sim.step(0.02)

        V = sim.get_voltage()
        V_min = float(V.min())
        V_max = float(V.max())

        # The stimulus should have initiated depolarization (V_max > -50 mV)
        assert V_max > -50.0, f"No depolarization: V_max={V_max:.1f} mV"

        print(f"  Grid: {grid}")
        print(f"  After 2ms: V_min={V_min:.1f} mV, V_max={V_max:.1f} mV")
        print("  PASS")
    finally:
        os.unlink(mesh_path)


if __name__ == '__main__':
    tests = [
        ("7-V1: mesh_builder export round-trip", test_7v1_mesh_export_roundtrip),
        ("7-V2: mesh loader round-trip", test_7v2_mesh_loader_roundtrip),
        ("7-V3: conductivity mapping (multi-group)", test_7v3_conductivity_mapping),
        ("7-V4: scar region D=0", test_7v4_scar_region_d_zero),
        ("7-V5: stim_builder export round-trip", test_7v5_stim_export_roundtrip),
        ("7-V6: stim loader round-trip", test_7v6_stim_loader_roundtrip),
        ("7-V7: end-to-end .npz → simulation", test_7v7_end_to_end_npz_to_simulation),
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
