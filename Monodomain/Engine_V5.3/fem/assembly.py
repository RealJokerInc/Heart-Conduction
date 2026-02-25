"""
FEM Matrix Assembly

Vectorized assembly of mass and stiffness matrices for P1 (linear)
triangular elements. All operations are GPU-compatible via PyTorch.

Mathematical Background
-----------------------
For P1 elements on triangle with vertices (x1,y1), (x2,y2), (x3,y3):

Shape functions:
    N_i(x,y) = (a_i + b_i*x + c_i*y) / (2*Area)

where:
    b = [y2-y3, y3-y1, y1-y2]
    c = [x3-x2, x1-x3, x2-x1]

Local stiffness matrix (for D=1):
    K_e[i,j] = (b_i*b_j + c_i*c_j) / (4*Area)

Local mass matrix (consistent):
    M_e[i,j] = Area/12 * [2,1,1; 1,2,1; 1,1,2]

Reference: Zienkiewicz & Taylor, "The Finite Element Method", 7th ed.
"""

from typing import Tuple
import torch
from .mesh import TriangularMesh


def assemble_matrices(
    mesh: TriangularMesh,
    D: float = 0.001,
    chi: float = 1400.0,
    Cm: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Assemble both mass and stiffness matrices.

    Parameters
    ----------
    mesh : TriangularMesh
        The computational mesh
    D : float
        Diffusion coefficient (cm²/ms)
    chi : float
        Surface-to-volume ratio (cm⁻¹)
    Cm : float
        Membrane capacitance (µF/cm²)

    Returns
    -------
    M : torch.Tensor
        Mass matrix (sparse COO), scaled by chi*Cm
    K : torch.Tensor
        Stiffness matrix (sparse COO), scaled by D
    """
    M = assemble_mass_matrix(mesh, chi, Cm)
    K = assemble_stiffness_matrix(mesh, D)
    return M, K


def assemble_mass_matrix(
    mesh: TriangularMesh,
    chi: float = 1400.0,
    Cm: float = 1.0
) -> torch.Tensor:
    """
    Assemble the consistent mass matrix.

    M_ij = χ·Cm·∫ φ_i·φ_j dΩ

    For P1 elements:
    M_e = χ·Cm·Area/12 · [2,1,1; 1,2,1; 1,1,2]

    Parameters
    ----------
    mesh : TriangularMesh
        The computational mesh
    chi : float
        Surface-to-volume ratio (cm⁻¹)
    Cm : float
        Membrane capacitance (µF/cm²)

    Returns
    -------
    M : torch.Tensor
        Sparse COO mass matrix
    """
    device = mesh.device
    dtype = mesh.dtype

    # Get element vertex coordinates: (n_elem, 3, 2)
    coords = mesh.get_element_coordinates()

    # Extract coordinates
    x1, y1 = coords[:, 0, 0], coords[:, 0, 1]
    x2, y2 = coords[:, 1, 0], coords[:, 1, 1]
    x3, y3 = coords[:, 2, 0], coords[:, 2, 1]

    # Compute element areas
    areas = 0.5 * torch.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    # Local mass matrix reference (unscaled)
    # M_ref = [2,1,1; 1,2,1; 1,1,2] / 12
    M_ref = torch.tensor(
        [[2, 1, 1],
         [1, 2, 1],
         [1, 1, 2]],
        dtype=dtype, device=device
    ) / 12.0

    # Scale: M_local = chi * Cm * Area * M_ref
    # Shape: (n_elem, 3, 3)
    scaling = chi * Cm * areas
    M_local = scaling.view(-1, 1, 1) * M_ref.unsqueeze(0)

    # Assemble into global sparse matrix
    return _assemble_local_to_global(mesh, M_local)


def assemble_stiffness_matrix(
    mesh: TriangularMesh,
    D: float = 0.001
) -> torch.Tensor:
    """
    Assemble the stiffness matrix.

    K_ij = D·∫ ∇φ_i·∇φ_j dΩ

    For P1 elements:
    K_e[i,j] = D·(b_i·b_j + c_i·c_j) / (4·Area)

    where b = [y2-y3, y3-y1, y1-y2], c = [x3-x2, x1-x3, x2-x1]

    Parameters
    ----------
    mesh : TriangularMesh
        The computational mesh
    D : float
        Diffusion coefficient (cm²/ms)

    Returns
    -------
    K : torch.Tensor
        Sparse COO stiffness matrix
    """
    device = mesh.device
    dtype = mesh.dtype

    # Get element vertex coordinates: (n_elem, 3, 2)
    coords = mesh.get_element_coordinates()

    # Extract coordinates
    x1, y1 = coords[:, 0, 0], coords[:, 0, 1]
    x2, y2 = coords[:, 1, 0], coords[:, 1, 1]
    x3, y3 = coords[:, 2, 0], coords[:, 2, 1]

    # Shape function gradient coefficients
    # b_i = y_{i+1} - y_{i+2} (cyclic)
    # c_i = x_{i+2} - x_{i+1} (cyclic)
    b = torch.stack([y2 - y3, y3 - y1, y1 - y2], dim=1)  # (n_elem, 3)
    c = torch.stack([x3 - x2, x1 - x3, x2 - x1], dim=1)  # (n_elem, 3)

    # Element areas
    areas = 0.5 * torch.abs(b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])

    # Local stiffness: K_e[i,j] = D * (b_i*b_j + c_i*c_j) / (4*Area)
    # Use einsum for outer products: (n_elem, 3, 3)
    K_local = (D / (4.0 * areas.unsqueeze(1).unsqueeze(2))) * (
        torch.einsum('ei,ej->eij', b, b) +
        torch.einsum('ei,ej->eij', c, c)
    )

    # Assemble into global sparse matrix
    return _assemble_local_to_global(mesh, K_local)


def _assemble_local_to_global(
    mesh: TriangularMesh,
    local_matrices: torch.Tensor
) -> torch.Tensor:
    """
    Assemble local element matrices into global sparse matrix.

    Parameters
    ----------
    mesh : TriangularMesh
        The computational mesh
    local_matrices : torch.Tensor
        Local matrices, shape (n_elements, 3, 3)

    Returns
    -------
    torch.Tensor
        Global sparse COO matrix
    """
    device = mesh.device
    dtype = mesh.dtype
    n_elem = mesh.n_elements

    # Local-to-global index mapping
    # For each element, we have 9 entries (3x3 local matrix)
    i_local = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], device=device)
    j_local = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], device=device)

    # Global row indices: elements[:, i_local] gives (n_elem, 9)
    global_i = mesh.elements[:, i_local].flatten()  # (n_elem * 9,)
    global_j = mesh.elements[:, j_local].flatten()  # (n_elem * 9,)

    # Values from local matrices
    values = local_matrices[:, i_local, j_local].flatten()  # (n_elem * 9,)

    # Create sparse COO tensor
    indices = torch.stack([global_i, global_j])
    sparse_matrix = torch.sparse_coo_tensor(
        indices,
        values,
        size=(mesh.n_nodes, mesh.n_nodes),
        dtype=dtype,
        device=device
    )

    # Coalesce to sum duplicate entries (from shared nodes)
    return sparse_matrix.coalesce()


def lump_mass_matrix(M: torch.Tensor) -> torch.Tensor:
    """
    Convert consistent mass matrix to lumped (diagonal) mass matrix.

    Lumping is done by row-sum: M_ii^L = Σ_j M_ij

    This trades accuracy for computational speed (diagonal solve).

    Parameters
    ----------
    M : torch.Tensor
        Consistent mass matrix (sparse COO)

    Returns
    -------
    torch.Tensor
        Lumped mass matrix (sparse diagonal)
    """
    if M.is_sparse:
        # Sum each row
        M_lumped = torch.sparse.sum(M, dim=1).to_dense()
    else:
        M_lumped = M.sum(dim=1)

    # Create diagonal sparse matrix
    n = M.shape[0]
    indices = torch.arange(n, device=M.device).unsqueeze(0).repeat(2, 1)

    return torch.sparse_coo_tensor(
        indices,
        M_lumped,
        size=(n, n),
        dtype=M.dtype,
        device=M.device
    ).coalesce()


def extract_diagonal(A: torch.Tensor) -> torch.Tensor:
    """
    Extract diagonal from sparse COO matrix.

    Parameters
    ----------
    A : torch.Tensor
        Sparse COO matrix

    Returns
    -------
    torch.Tensor
        Diagonal elements as dense vector
    """
    if A.is_sparse:
        indices = A.indices()
        values = A.values()
        mask = indices[0] == indices[1]
        diag = torch.zeros(A.shape[0], dtype=A.dtype, device=A.device)
        diag[indices[0, mask]] = values[mask]
        return diag
    else:
        return A.diag()
