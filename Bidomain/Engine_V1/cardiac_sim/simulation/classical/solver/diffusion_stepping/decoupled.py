"""
Decoupled Bidomain Diffusion Solver

Gauss-Seidel splitting: parabolic solve for Vm, then elliptic solve for phi_e.
Two sequential N x N SPD solves per time step.

Step 1 (Parabolic):
    (chi*Cm/dt * I - theta*L_i) * Vm^{n+1} =
        B_para * Vm^n + theta * L_i * phi_e^n

Step 2 (Elliptic):
    -(L_i + L_e) * phi_e^{n+1} = L_i * Vm^{n+1}

Ref: improvement.md L960-1044
"""

from typing import TYPE_CHECKING

from .base import BidomainDiffusionSolver
from ..linear_solver.pcg import sparse_mv

if TYPE_CHECKING:
    from ...discretization.base import BidomainSpatialDiscretization
    from ...state import BidomainState
    from ..linear_solver.base import LinearSolver


class DecoupledBidomainDiffusionSolver(BidomainDiffusionSolver):
    """
    Decoupled (Gauss-Seidel splitting) bidomain diffusion solver.

    Parameters
    ----------
    spatial : BidomainSpatialDiscretization
        Provides L_i, L_e stencils and parabolic/elliptic operators.
    dt : float
        Time step (ms)
    parabolic_solver : LinearSolver
        Solver for Vm sub-problem
    elliptic_solver : LinearSolver
        Solver for phi_e sub-problem
    theta : float
        Implicitness (0.5 = CN, 1.0 = BDF1)
    pin_node : int
        Node for null space pinning (Neumann only)
    """

    def __init__(self, spatial, dt, parabolic_solver, elliptic_solver,
                 theta=0.5, pin_node=0):
        super().__init__(spatial, dt)
        self.theta = theta
        self.parabolic_solver = parabolic_solver
        self.elliptic_solver = elliptic_solver

        # Read BCs from mesh
        self._needs_pinning = spatial.grid.boundary_spec.phi_e_has_null_space
        self._pin_node = pin_node

        # Build operators
        self._build_operators(spatial, dt)

    def _build_operators(self, spatial, dt):
        """Build parabolic and elliptic operators."""
        self.A_para, self.B_para = spatial.get_parabolic_operators(dt, self.theta)
        self.A_ellip = spatial.get_elliptic_operator()

        # Null space pinning for Neumann phi_e
        if self._needs_pinning:
            self._apply_pinning(self.A_ellip, self._pin_node)

    def step(self, state, dt):
        """
        Advance Vm and phi_e by one diffusion time step.

        Step 1: Parabolic solve for Vm
        Step 2: Elliptic solve for phi_e
        """
        # --- Step 1: Parabolic (Vm) ---
        # Coupling: L_i * phi_e^n (NOT theta * L_i * phi_e^n)
        # From CN discretization of dVm/dt = L_i*(Vm + phi_e):
        #   theta*L_i*phi_e^{n+1} + (1-theta)*L_i*phi_e^n
        #   ≈ L_i*phi_e^n (since phi_e is lagged)
        rhs_para = sparse_mv(self.B_para, state.Vm) \
                   + self._spatial.apply_L_i(state.phi_e)
        Vm_new = self.parabolic_solver.solve(self.A_para, rhs_para)

        # --- Step 2: Elliptic (phi_e) ---
        rhs_ellip = self._spatial.apply_L_i(Vm_new)
        if self._needs_pinning:
            rhs_ellip[self._pin_node] = 0.0
        phi_e_new = self.elliptic_solver.solve(self.A_ellip, rhs_ellip)

        # Post-subtract pinning for spectral solvers (Neumann)
        if self._needs_pinning:
            phi_e_new = phi_e_new - phi_e_new[self._pin_node]

        # --- Update state in-place ---
        state.Vm.copy_(Vm_new)
        state.phi_e.copy_(phi_e_new)

    def _apply_pinning(self, A, pin_node):
        """Enforce phi_e(pin_node) = 0 by modifying the elliptic matrix."""
        if A.is_sparse:
            A_coal = A.coalesce()
            indices = A_coal.indices()
            values = A_coal.values()

            # Zero out row and column for pin_node
            row_mask = indices[0] != pin_node
            col_mask = indices[1] != pin_node
            keep = row_mask & col_mask

            new_row = [indices[0, keep], indices[0].new_tensor([pin_node])]
            new_col = [indices[1, keep], indices[1].new_tensor([pin_node])]
            new_val = [values[keep], values.new_tensor([1.0])]

            import torch
            new_indices = torch.stack([torch.cat(new_row), torch.cat(new_col)])
            new_values = torch.cat(new_val)
            pinned = torch.sparse_coo_tensor(new_indices, new_values, A.shape)

            # In-place replacement via data_ptr isn't possible for sparse.
            # Store as attribute.
            self.A_ellip = pinned.coalesce()
        else:
            import torch
            A[pin_node, :] = 0
            A[:, pin_node] = 0
            A[pin_node, pin_node] = 1.0

    def rebuild_operators(self, spatial, dt):
        """Rebuild operators when dt changes."""
        self._spatial = spatial
        self._dt = dt
        self._build_operators(spatial, dt)
