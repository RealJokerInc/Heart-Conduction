"""
Finite Element Method Discretization

Wraps V5.3 FEM assembly (P1 triangular elements) into the
SpatialDiscretization interface.

Ref: improvement.md:L810-846
Ref: V5.3 fem/assembly.py
"""

from typing import Tuple
import torch

from .base import SpatialDiscretization, MassType, DiffusionOperators, sparse_mv
from ....tissue_builder.mesh.triangular import TriangularMesh


# =============================================================================
# FEM Assembly Functions (migrated from V5.3 fem/assembly.py)
# =============================================================================

def assemble_mass_matrix(
    mesh: TriangularMesh,
    chi: float = 1400.0,
    Cm: float = 1.0
) -> torch.Tensor:
    """
    Assemble the consistent mass matrix.

    M_ij = chi * Cm * integral(phi_i * phi_j) dOmega

    For P1 elements:
    M_e = chi * Cm * Area/12 * [2,1,1; 1,2,1; 1,1,2]
    """
    device = mesh._device
    dtype = mesh._dtype

    coords = mesh.get_element_coordinates()
    x1, y1 = coords[:, 0, 0], coords[:, 0, 1]
    x2, y2 = coords[:, 1, 0], coords[:, 1, 1]
    x3, y3 = coords[:, 2, 0], coords[:, 2, 1]

    areas = 0.5 * torch.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    M_ref = torch.tensor(
        [[2, 1, 1],
         [1, 2, 1],
         [1, 1, 2]],
        dtype=dtype, device=device
    ) / 12.0

    scaling = chi * Cm * areas
    M_local = scaling.view(-1, 1, 1) * M_ref.unsqueeze(0)

    return _assemble_local_to_global(mesh, M_local)


def assemble_stiffness_matrix(
    mesh: TriangularMesh,
    D: float = 0.001
) -> torch.Tensor:
    """
    Assemble the stiffness matrix.

    K_ij = D * integral(grad(phi_i) . grad(phi_j)) dOmega

    For P1 elements:
    K_e[i,j] = D * (b_i*b_j + c_i*c_j) / (4*Area)
    """
    device = mesh._device
    dtype = mesh._dtype

    coords = mesh.get_element_coordinates()
    x1, y1 = coords[:, 0, 0], coords[:, 0, 1]
    x2, y2 = coords[:, 1, 0], coords[:, 1, 1]
    x3, y3 = coords[:, 2, 0], coords[:, 2, 1]

    b = torch.stack([y2 - y3, y3 - y1, y1 - y2], dim=1)
    c = torch.stack([x3 - x2, x1 - x3, x2 - x1], dim=1)

    areas = 0.5 * torch.abs(b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])

    K_local = (D / (4.0 * areas.unsqueeze(1).unsqueeze(2))) * (
        torch.einsum('ei,ej->eij', b, b) +
        torch.einsum('ei,ej->eij', c, c)
    )

    return _assemble_local_to_global(mesh, K_local)


def assemble_matrices(
    mesh: TriangularMesh,
    D: float = 0.001,
    chi: float = 1400.0,
    Cm: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assemble both mass and stiffness matrices."""
    M = assemble_mass_matrix(mesh, chi, Cm)
    K = assemble_stiffness_matrix(mesh, D)
    return M, K


def _assemble_local_to_global(
    mesh: TriangularMesh,
    local_matrices: torch.Tensor
) -> torch.Tensor:
    """Assemble local element matrices into global sparse matrix."""
    device = mesh._device
    dtype = mesh._dtype

    i_local = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], device=device)
    j_local = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], device=device)

    global_i = mesh.elements[:, i_local].flatten()
    global_j = mesh.elements[:, j_local].flatten()
    values = local_matrices[:, i_local, j_local].flatten()

    indices = torch.stack([global_i, global_j])
    sparse_matrix = torch.sparse_coo_tensor(
        indices, values,
        size=(mesh.n_nodes, mesh.n_nodes),
        dtype=dtype, device=device
    )
    return sparse_matrix.coalesce()


# =============================================================================
# FEMDiscretization
# =============================================================================

class FEMDiscretization(SpatialDiscretization):
    """
    Finite Element Method spatial discretization.

    Uses P1 (linear) triangular elements. Wraps the assembly functions
    from V5.3 and provides the SpatialDiscretization interface.

    Parameters
    ----------
    mesh : TriangularMesh
        The computational mesh
    D : float
        Diffusion coefficient (cm^2/ms)
    chi : float
        Surface-to-volume ratio (cm^-1)
    Cm : float
        Membrane capacitance (uF/cm^2)
    """

    def __init__(
        self,
        mesh: TriangularMesh,
        D: float = 0.001,
        chi: float = 1400.0,
        Cm: float = 1.0
    ):
        self._mesh = mesh
        self._n_dof = mesh.n_nodes
        self._x = mesh.nodes[:, 0]
        self._y = mesh.nodes[:, 1]

        # Assemble FEM matrices
        self.M = assemble_mass_matrix(mesh, chi, Cm)
        self.K = assemble_stiffness_matrix(mesh, D)

    @property
    def n_dof(self) -> int:
        return self._n_dof

    @property
    def coordinates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._x, self._y

    @property
    def mass_type(self) -> MassType:
        return MassType.SPARSE

    @property
    def mesh(self) -> TriangularMesh:
        """Access the underlying mesh for visualization."""
        return self._mesh

    def get_diffusion_operators(self, dt: float, scheme: str) -> DiffusionOperators:
        """
        Build operators for implicit time stepping.

        For FEM: M*dV/dt = -K*V
        - CN:   (M + 0.5*dt*K)*V^{n+1} = (M - 0.5*dt*K)*V^n
        - BDF1: (M + dt*K)*V^{n+1} = M*V^n
        """
        scheme = scheme.upper()

        if scheme == "CN":
            A = (self.M + 0.5 * dt * self.K).coalesce()
            B = (self.M - 0.5 * dt * self.K).coalesce()
        elif scheme == "BDF1":
            A = (self.M + dt * self.K).coalesce()
            B = self.M.clone()
        elif scheme == "BDF2":
            # BDF2: (3M + 2*dt*K)*V^{n+1} = 4M*V^n - M*V^{n-1}
            # A_lhs for the LHS, B_rhs encodes 4M (caller manages V^{n-1})
            A = (3.0 * self.M + 2.0 * dt * self.K).coalesce()
            B = (4.0 * self.M).coalesce()
        else:
            raise ValueError(f"Unknown scheme: {scheme}. Use 'CN', 'BDF1', or 'BDF2'.")

        M_ref = self.M

        def apply_mass(f: torch.Tensor) -> torch.Tensor:
            return sparse_mv(M_ref, f)

        return DiffusionOperators(A_lhs=A, B_rhs=B, apply_mass=apply_mass)

    def apply_diffusion(self, V: torch.Tensor) -> torch.Tensor:
        """
        Compute -K*V (negative stiffness application) for explicit methods.

        For FEM explicit stepping, the full operation is:
        M * dV/dt = -K * V  =>  dV/dt = M^{-1} * (-K * V)

        This method returns -K*V. The caller must handle the mass solve.
        """
        return -sparse_mv(self.K, V)
