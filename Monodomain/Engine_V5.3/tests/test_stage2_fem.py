#!/usr/bin/env python3
"""
Stage 2 Validation: FEM Core

Tests:
2.1: Single triangle M matrix - Match hand calculation
2.2: Single triangle K matrix - Match hand calculation
2.3: Mass matrix symmetry
2.4: Stiffness matrix symmetry
2.5: Mass matrix positive-definite
2.6: Stiffness matrix semi-definite
2.7: Row sum property (Neumann BC: sum(K_row) ≈ 0)
2.8: Mesh generation consistency
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


def print_test_header(test_id: str, description: str):
    """Print test header."""
    print(f"\n[Test {test_id}] {description}")
    print("-" * 60)


def print_result(passed: bool, message: str = ""):
    """Print test result."""
    status = "PASS" if passed else "FAIL"
    symbol = "✓" if passed else "✗"
    if message:
        print(f"  {symbol} {status}: {message}")
    else:
        print(f"  {symbol} {status}")
    return passed


def test_single_triangle_mass():
    """
    Test 2.1: Single triangle mass matrix against hand calculation.

    Triangle: (0,0), (1,0), (0,1)
    Area = 0.5
    With chi=1, Cm=1:
    M = Area/12 * [2,1,1; 1,2,1; 1,1,2] = 1/24 * [2,1,1; 1,2,1; 1,1,2]
    """
    print_test_header("2.1", "Single triangle mass matrix")

    from fem.mesh import TriangularMesh
    from fem.assembly import assemble_mass_matrix

    # Create single-triangle mesh
    nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    elements = torch.tensor([[0, 1, 2]], dtype=torch.long)
    boundary = torch.tensor([0, 1, 2], dtype=torch.long)

    mesh = TriangularMesh(
        nodes=nodes,
        elements=elements,
        boundary_nodes=boundary,
        n_nodes=3,
        n_elements=1,
        device=torch.device('cpu'),
        dtype=torch.float64
    )

    # Assemble with chi=1, Cm=1
    M = assemble_mass_matrix(mesh, chi=1.0, Cm=1.0)
    M_dense = M.to_dense()

    # Expected: M = Area/12 * [2,1,1; 1,2,1; 1,1,2] where Area = 0.5
    M_expected = torch.tensor([
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2]
    ], dtype=torch.float64) / 24.0

    error = torch.max(torch.abs(M_dense - M_expected)).item()
    passed = error < 1e-14

    print(f"  Expected M (scaled by 24):\n{M_expected * 24}")
    print(f"  Computed M (scaled by 24):\n{M_dense * 24}")
    print(f"  Max error: {error:.2e}")

    return print_result(passed, f"Error = {error:.2e}")


def test_single_triangle_stiffness():
    """
    Test 2.2: Single triangle stiffness matrix against hand calculation.

    Triangle: (0,0), (1,0), (0,1)
    Area = 0.5
    b = [y2-y3, y3-y1, y1-y2] = [0-1, 1-0, 0-0] = [-1, 1, 0]
    c = [x3-x2, x1-x3, x2-x1] = [0-1, 0-0, 1-0] = [-1, 0, 1]

    With D=1:
    K[i,j] = D * (b_i*b_j + c_i*c_j) / (4*Area)
           = (b_i*b_j + c_i*c_j) / 2

    K = 0.5 * [b⊗b + c⊗c]
      = 0.5 * [[1+1, -1+0, 0-1], [-1+0, 1+0, 0+0], [0-1, 0+0, 0+1]]
      = 0.5 * [[2, -1, -1], [-1, 1, 0], [-1, 0, 1]]
      = [[1, -0.5, -0.5], [-0.5, 0.5, 0], [-0.5, 0, 0.5]]
    """
    print_test_header("2.2", "Single triangle stiffness matrix")

    from fem.mesh import TriangularMesh
    from fem.assembly import assemble_stiffness_matrix

    # Create single-triangle mesh
    nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    elements = torch.tensor([[0, 1, 2]], dtype=torch.long)
    boundary = torch.tensor([0, 1, 2], dtype=torch.long)

    mesh = TriangularMesh(
        nodes=nodes,
        elements=elements,
        boundary_nodes=boundary,
        n_nodes=3,
        n_elements=1,
        device=torch.device('cpu'),
        dtype=torch.float64
    )

    # Assemble with D=1
    K = assemble_stiffness_matrix(mesh, D=1.0)
    K_dense = K.to_dense()

    # Expected from hand calculation
    K_expected = torch.tensor([
        [1.0, -0.5, -0.5],
        [-0.5, 0.5, 0.0],
        [-0.5, 0.0, 0.5]
    ], dtype=torch.float64)

    error = torch.max(torch.abs(K_dense - K_expected)).item()
    passed = error < 1e-14

    print(f"  Expected K:\n{K_expected}")
    print(f"  Computed K:\n{K_dense}")
    print(f"  Max error: {error:.2e}")

    return print_result(passed, f"Error = {error:.2e}")


def test_mass_matrix_symmetry():
    """Test 2.3: Mass matrix is symmetric."""
    print_test_header("2.3", "Mass matrix symmetry")

    from fem import TriangularMesh, assemble_mass_matrix

    mesh = TriangularMesh.create_rectangle(1.0, 1.0, 11, 11, device='cpu')
    M = assemble_mass_matrix(mesh, chi=1400.0, Cm=1.0)
    M_dense = M.to_dense()

    asymmetry = torch.max(torch.abs(M_dense - M_dense.T)).item()
    passed = asymmetry < 1e-14

    print(f"  Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    print(f"  Max asymmetry: {asymmetry:.2e}")

    return print_result(passed, f"Asymmetry = {asymmetry:.2e}")


def test_stiffness_matrix_symmetry():
    """Test 2.4: Stiffness matrix is symmetric."""
    print_test_header("2.4", "Stiffness matrix symmetry")

    from fem import TriangularMesh, assemble_stiffness_matrix

    mesh = TriangularMesh.create_rectangle(1.0, 1.0, 11, 11, device='cpu')
    K = assemble_stiffness_matrix(mesh, D=0.001)
    K_dense = K.to_dense()

    asymmetry = torch.max(torch.abs(K_dense - K_dense.T)).item()
    passed = asymmetry < 1e-14

    print(f"  Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    print(f"  Max asymmetry: {asymmetry:.2e}")

    return print_result(passed, f"Asymmetry = {asymmetry:.2e}")


def test_mass_matrix_positive_definite():
    """Test 2.5: Mass matrix is positive definite."""
    print_test_header("2.5", "Mass matrix positive-definiteness")

    from fem import TriangularMesh, assemble_mass_matrix

    mesh = TriangularMesh.create_rectangle(1.0, 1.0, 11, 11, device='cpu')
    M = assemble_mass_matrix(mesh, chi=1400.0, Cm=1.0)
    M_dense = M.to_dense()

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(M_dense)
    min_eig = eigenvalues.min().item()
    max_eig = eigenvalues.max().item()

    passed = min_eig > 0

    print(f"  Min eigenvalue: {min_eig:.6e}")
    print(f"  Max eigenvalue: {max_eig:.6e}")
    print(f"  Condition number: {max_eig/min_eig:.2e}")

    return print_result(passed, f"Min eigenvalue = {min_eig:.6e}")


def test_stiffness_matrix_semidefinite():
    """Test 2.6: Stiffness matrix is positive semi-definite."""
    print_test_header("2.6", "Stiffness matrix positive semi-definiteness")

    from fem import TriangularMesh, assemble_stiffness_matrix

    mesh = TriangularMesh.create_rectangle(1.0, 1.0, 11, 11, device='cpu')
    K = assemble_stiffness_matrix(mesh, D=0.001)
    K_dense = K.to_dense()

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(K_dense)
    min_eig = eigenvalues.min().item()
    max_eig = eigenvalues.max().item()

    # Semi-definite: min eigenvalue >= 0 (with tolerance for numerical errors)
    passed = min_eig >= -1e-12

    print(f"  Min eigenvalue: {min_eig:.6e}")
    print(f"  Max eigenvalue: {max_eig:.6e}")
    print(f"  # near-zero eigenvalues: {(eigenvalues.abs() < 1e-10).sum().item()}")

    return print_result(passed, f"Min eigenvalue = {min_eig:.6e}")


def test_stiffness_row_sum():
    """
    Test 2.7: Stiffness matrix row sums are zero (Neumann BC).

    For Neumann BC, ∑_j K_ij = 0 for interior nodes.
    This property holds when all elements share nodes properly.
    """
    print_test_header("2.7", "Stiffness matrix row sum (Neumann BC)")

    from fem import TriangularMesh, assemble_stiffness_matrix

    mesh = TriangularMesh.create_rectangle(1.0, 1.0, 11, 11, device='cpu')
    K = assemble_stiffness_matrix(mesh, D=0.001)
    K_dense = K.to_dense()

    row_sums = K_dense.sum(dim=1)
    max_row_sum = torch.max(torch.abs(row_sums)).item()

    passed = max_row_sum < 1e-14

    print(f"  Max absolute row sum: {max_row_sum:.2e}")

    return print_result(passed, f"Max row sum = {max_row_sum:.2e}")


def test_mesh_generation():
    """Test 2.8: Mesh generation produces correct structure."""
    print_test_header("2.8", "Mesh generation consistency")

    from fem import TriangularMesh

    # Test various mesh sizes
    test_cases = [
        (1.0, 1.0, 5, 5),
        (2.0, 1.0, 11, 6),
        (5.0, 5.0, 51, 51),
    ]

    all_passed = True

    for Lx, Ly, nx, ny in test_cases:
        mesh = TriangularMesh.create_rectangle(Lx, Ly, nx, ny, device='cpu')

        # Check node count
        expected_nodes = nx * ny
        nodes_ok = mesh.n_nodes == expected_nodes

        # Check element count: 2 triangles per quad, (nx-1)*(ny-1) quads
        expected_elements = 2 * (nx - 1) * (ny - 1)
        elements_ok = mesh.n_elements == expected_elements

        # Check that elements reference valid nodes
        max_node_idx = mesh.elements.max().item()
        valid_refs = max_node_idx < mesh.n_nodes

        # Check boundary nodes
        n_boundary_expected = 2 * (nx + ny - 2)
        boundary_ok = len(mesh.boundary_nodes) == n_boundary_expected

        # Check total area
        areas = mesh.compute_element_areas()
        total_area = areas.sum().item()
        area_ok = abs(total_area - Lx * Ly) < 1e-10

        case_passed = nodes_ok and elements_ok and valid_refs and boundary_ok and area_ok

        status = "PASS" if case_passed else "FAIL"
        print(f"  {nx}x{ny} mesh on [{Lx}x{Ly}]: {status}")
        print(f"    Nodes: {mesh.n_nodes} (expected {expected_nodes})")
        print(f"    Elements: {mesh.n_elements} (expected {expected_elements})")
        print(f"    Boundary nodes: {len(mesh.boundary_nodes)} (expected {n_boundary_expected})")
        print(f"    Total area: {total_area:.6f} (expected {Lx*Ly})")

        all_passed = all_passed and case_passed

    return print_result(all_passed)


def test_gpu_assembly():
    """Test FEM assembly on GPU if available."""
    print_test_header("2.9", "GPU assembly (if available)")

    from fem import TriangularMesh, assemble_matrices

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping GPU test")
        return print_result(True, "Skipped - no GPU")

    # Create mesh on GPU
    mesh = TriangularMesh.create_rectangle(1.0, 1.0, 51, 51, device='cuda')
    M, K = assemble_matrices(mesh, D=0.001, chi=1400.0, Cm=1.0)

    # Check tensors are on GPU
    on_gpu = M.device.type == 'cuda' and K.device.type == 'cuda'

    # Check basic properties
    M_dense = M.to_dense()
    K_dense = K.to_dense()

    M_symmetric = torch.max(torch.abs(M_dense - M_dense.T)).item() < 1e-12
    K_symmetric = torch.max(torch.abs(K_dense - K_dense.T)).item() < 1e-12

    passed = on_gpu and M_symmetric and K_symmetric

    print(f"  Device: {M.device}")
    print(f"  M symmetric: {M_symmetric}")
    print(f"  K symmetric: {K_symmetric}")

    return print_result(passed)


def main():
    print("=" * 70)
    print("Stage 2 Validation: FEM Core")
    print("=" * 70)

    results = []

    results.append(("2.1", "Single triangle M", test_single_triangle_mass()))
    results.append(("2.2", "Single triangle K", test_single_triangle_stiffness()))
    results.append(("2.3", "M symmetry", test_mass_matrix_symmetry()))
    results.append(("2.4", "K symmetry", test_stiffness_matrix_symmetry()))
    results.append(("2.5", "M positive-definite", test_mass_matrix_positive_definite()))
    results.append(("2.6", "K semi-definite", test_stiffness_matrix_semidefinite()))
    results.append(("2.7", "K row sum", test_stiffness_row_sum()))
    results.append(("2.8", "Mesh generation", test_mesh_generation()))
    results.append(("2.9", "GPU assembly", test_gpu_assembly()))

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    n_passed = sum(1 for _, _, p in results if p)
    n_total = len(results)

    for test_id, name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {test_id}: {name} - {status}")

    print("-" * 70)
    print(f"Passed: {n_passed}/{n_total}")
    print("=" * 70)

    return 0 if n_passed == n_total else 1


if __name__ == '__main__':
    sys.exit(main())
