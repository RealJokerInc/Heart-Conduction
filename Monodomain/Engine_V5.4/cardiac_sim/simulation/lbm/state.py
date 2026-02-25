"""
LBM Simulation State

Holds distribution functions, macroscopic variables, and domain mask.
Provides streaming and boundary condition methods.

Grid convention: (Nx, Ny) with indexing='ij', matching StructuredGrid.
  - Axis 0 = x (columns), Axis 1 = y (rows)
  - V.shape = (Nx, Ny), f.shape = (Q, Nx, Ny)

V is stored separately from ionic_states (gates + concentrations only).

Ref: Research/04_LBM_EP:L873-934
"""

import torch
from typing import Optional, Tuple
from dataclasses import dataclass, field

from .d2q5 import D2Q5, d2q5


@dataclass
class LBMState:
    """
    State container for LBM simulation.

    Attributes
    ----------
    f : torch.Tensor
        Distribution functions, shape (Q, Nx, Ny)
    V : torch.Tensor
        Macroscopic voltage, shape (Nx, Ny)
    ionic_states : torch.Tensor
        Ionic state variables (gates + concentrations), shape (Nx*Ny, n_states)
    mask : torch.Tensor
        Domain mask (1 = active, 0 = outside), shape (Nx, Ny)
    t : float
        Current simulation time (ms)
    lattice : D2Q5
        Lattice definition
    """
    f: torch.Tensor
    V: torch.Tensor
    ionic_states: torch.Tensor
    mask: torch.Tensor
    t: float
    lattice: D2Q5 = field(default=d2q5)

    @property
    def Q(self) -> int:
        """Number of velocity directions."""
        return self.lattice.Q

    @property
    def Nx(self) -> int:
        """Grid size in x (axis 0)."""
        return self.V.shape[0]

    @property
    def Ny(self) -> int:
        """Grid size in y (axis 1)."""
        return self.V.shape[1]

    @property
    def n_dof(self) -> int:
        """Total degrees of freedom (active cells)."""
        return int(self.mask.sum().item())

    @property
    def device(self) -> torch.device:
        """Device of tensors."""
        return self.V.device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of tensors."""
        return self.V.dtype

    @property
    def V_flat(self) -> torch.Tensor:
        """Return V as 1D (Nx*Ny,) — zero-copy view via reshape."""
        return self.V.reshape(-1)

    def recover_voltage(self) -> None:
        """
        Recover macroscopic voltage from distributions.

        V = Σ f_i (sum over all directions)
        """
        self.V = self.f.sum(dim=0)

    def init_equilibrium(self) -> None:
        """
        Initialize distributions to equilibrium.

        f_i = w_i * V
        """
        Q = self.Q
        w = self.lattice.get_w_tensor(self.device, self.dtype)
        self.f = w.view(Q, 1, 1) * self.V.unsqueeze(0)

    def stream(self) -> None:
        """
        Streaming step: f_i(x + e_i, t+dt) = f*_i(x, t)

        Uses torch.roll for each velocity direction.
        Convention: (Nx, Ny) — axis 0 = x, axis 1 = y
        """
        f_new = torch.zeros_like(self.f)

        # Direction 0: rest (no movement)
        f_new[0] = self.f[0]

        # Direction 1: +x -> roll by -1 in x (dim=0, axis 0 = x)
        f_new[1] = torch.roll(self.f[1], shifts=-1, dims=0)

        # Direction 2: -x -> roll by +1 in x
        f_new[2] = torch.roll(self.f[2], shifts=1, dims=0)

        # Direction 3: +y -> roll by -1 in y (dim=1, axis 1 = y)
        f_new[3] = torch.roll(self.f[3], shifts=-1, dims=1)

        # Direction 4: -y -> roll by +1 in y
        f_new[4] = torch.roll(self.f[4], shifts=1, dims=1)

        self.f = f_new

    def apply_bounce_back(self) -> None:
        """
        Apply bounce-back boundary conditions.

        For no-flux (Neumann) BC at domain boundaries:
        - If neighbor is outside domain, reflect distribution back
        - f_opposite(x) = f_i(x) when x+e_i is outside

        Uses shifted masks to identify boundary nodes.
        Convention: (Nx, Ny) — axis 0 = x, axis 1 = y
        """
        mask = self.mask.float()
        opposite = self.lattice.opposite

        # +x boundary: if node at (x+1, y) is outside, bounce f[1] -> f[2]
        mask_px = torch.roll(mask, shifts=-1, dims=0)
        bounce_px = (mask > 0) & (mask_px == 0)
        self.f[opposite[1]][bounce_px] = self.f[1][bounce_px]

        # -x boundary
        mask_mx = torch.roll(mask, shifts=1, dims=0)
        bounce_mx = (mask > 0) & (mask_mx == 0)
        self.f[opposite[2]][bounce_mx] = self.f[2][bounce_mx]

        # +y boundary
        mask_py = torch.roll(mask, shifts=-1, dims=1)
        bounce_py = (mask > 0) & (mask_py == 0)
        self.f[opposite[3]][bounce_py] = self.f[3][bounce_py]

        # -y boundary
        mask_my = torch.roll(mask, shifts=1, dims=1)
        bounce_my = (mask > 0) & (mask_my == 0)
        self.f[opposite[4]][bounce_my] = self.f[4][bounce_my]

        # Zero out distributions outside domain
        for i in range(self.Q):
            self.f[i] = self.f[i] * mask

    def total_voltage(self) -> float:
        """
        Compute total voltage (for conservation check).

        Returns sum of V over active domain.
        """
        return (self.V * self.mask).sum().item()


def create_lbm_state(
    Nx: int,
    Ny: int,
    V_init: torch.Tensor,
    ionic_states: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    lattice: D2Q5 = d2q5,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float64
) -> LBMState:
    """
    Create and initialize LBM state.

    Parameters
    ----------
    Nx, Ny : int
        Grid dimensions (Nx = axis 0 = x, Ny = axis 1 = y)
    V_init : torch.Tensor
        Initial voltage, shape (Nx, Ny)
    ionic_states : torch.Tensor
        Ionic state variables (gates + concentrations), shape (Nx*Ny, n_states)
    mask : torch.Tensor, optional
        Domain mask. If None, entire domain is active.
    lattice : D2Q5
        Lattice definition
    device : torch.device
        Computation device
    dtype : torch.dtype
        Data type

    Returns
    -------
    state : LBMState
    """
    if mask is None:
        mask = torch.ones(Nx, Ny, device=device, dtype=torch.bool)
    else:
        mask = mask.to(device=device, dtype=torch.bool)

    V = V_init.to(device=device, dtype=dtype)
    ionic_states = ionic_states.to(device=device, dtype=dtype)

    # Initialize distributions to equilibrium
    Q = lattice.Q
    w = lattice.get_w_tensor(device, dtype)
    f = w.view(Q, 1, 1) * V.unsqueeze(0)

    # Zero out outside domain
    f = f * mask.float().unsqueeze(0)

    return LBMState(
        f=f,
        V=V,
        ionic_states=ionic_states,
        mask=mask,
        t=0.0,
        lattice=lattice
    )
