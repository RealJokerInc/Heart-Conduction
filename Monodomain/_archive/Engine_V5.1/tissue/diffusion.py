"""
Finite Volume Method (FVM) Diffusion Operator for Monodomain Equation

GPU-accelerated implementation using PyTorch.

The monodomain equation:
    χ * (Cm * ∂V/∂t + Iion) = ∇·(D·∇V)

Where D is the conductivity tensor:
    D = R(θ) * [[D_L, 0], [0, D_T]] * R(θ)^T

CV-based diffusion scaling:
    Empirical relationship: CV = k * sqrt(D)
    where k ≈ 1.514 cm^0.5/ms^0.5 (calibrated from ORd simulations)
    So: D = (CV/k)^2

Human ventricular CV targets:
    - Longitudinal: 0.5-0.7 m/s (0.05-0.07 cm/ms)
    - Transverse: 0.17-0.25 m/s (0.017-0.025 cm/ms)
    - Anisotropy ratio: ~3:1
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union
import math


# Physical constants
CHI = 1400.0      # Surface-to-volume ratio (cm^-1)
CM = 1.0          # Membrane capacitance (µF/cm^2)

# Empirical calibration from ORd model simulations
# Measured: D_L = 0.001498 cm^2/ms at dx = 0.01 cm gives CV ≈ 0.0586 cm/ms
# This gives: k = CV / sqrt(D) = 0.0586 / sqrt(0.001498) = 1.514
CV_EMPIRICAL_CONSTANT = 1.514  # cm^0.5 / ms^0.5

# ============================================================================
# D_min(dx, APD) - Mesh and APD dependent minimum diffusion coefficient
# ============================================================================
#
# VALIDATED FORMULA (Phase 1 experiments, 2024-12-23):
#
#   D_min(dx, APD) = k_base × dx² × (APD_ref / APD)^alpha
#
# Where:
#   - k_base = 0.92: Base mesh stability constant
#   - APD_ref = 280 ms: Reference APD (normal ORd endocardial)
#   - alpha = 0.25: APD scaling exponent (weak dependence)
#   - dx = mesh spacing in cm
#   - APD = action potential duration in ms
#
# The minimum D has TWO components:
#   1. D_min_numerical(dx): Mesh-dependent numerical stability limit
#   2. D_min_physical(APD): APD-dependent source-sink balance limit
#
# The formula combines both effects into a single expression.
#
# See DIFFUSION_BUG.md and IMPLEMENTATION_PLAN.md for full derivation.
# ============================================================================

# D_min formula constants
K_BASE = 0.92           # Base mesh stability constant
APD_REF = 280.0         # Reference APD (ms) - normal ORd endocardial
APD_ALPHA = 0.25        # APD scaling exponent
SAFETY_MARGIN_DEFAULT = 1.5  # Recommended D/D_min ratio

# Legacy constant (kept for backwards compatibility, but D_min is now mesh-dependent)
D_MINIMUM_STABLE = 0.00080  # cm^2/ms - approximate for dx=0.03cm (0.3mm)


def compute_D_min(dx: float, apd_ms: float = APD_REF) -> float:
    """
    Compute minimum stable diffusion coefficient for given mesh and APD.

    VALIDATED FORMULA (Phase 1 experiments, 2024-12-23):
        D_min(dx, APD) = 0.92 × dx² × (280/APD)^0.25

    Parameters
    ----------
    dx : float
        Mesh spacing in cm (e.g., 0.02 for 0.2mm)
    apd_ms : float
        Action potential duration in ms (default 280 for normal ORd)

    Returns
    -------
    D_min : float
        Minimum stable diffusion coefficient in cm²/ms
    """
    return K_BASE * (dx ** 2) * (APD_REF / apd_ms) ** APD_ALPHA


def validate_D_for_mesh(
    D: float,
    dx: float,
    apd_ms: float = APD_REF,
    safety_margin: float = SAFETY_MARGIN_DEFAULT,
    raise_error: bool = False,
    direction: str = "transverse"
) -> tuple:
    """
    Validate that diffusion coefficient D is sufficient for mesh spacing dx.

    Uses the experimentally-validated D_min(dx, APD) formula to determine
    if the requested D value will support stable propagation.

    Parameters
    ----------
    D : float
        Diffusion coefficient (cm²/ms)
    dx : float
        Mesh spacing (cm)
    apd_ms : float
        Action potential duration in ms (default 280)
    safety_margin : float
        Multiplier above D_min for safe operation (default 1.5)
    raise_error : bool
        If True, raise ValueError when D < D_min (default False)
    direction : str
        Which direction this D is for (used in error messages)

    Returns
    -------
    status : str
        "OK", "WARNING", or "CRITICAL"
    message : str
        Descriptive message about the validation result

    Raises
    ------
    ValueError
        If raise_error=True and D < D_min
    """
    D_min = compute_D_min(dx, apd_ms)
    D_safe = D_min * safety_margin

    dx_mm = dx * 10  # Convert to mm for messages

    if D < D_min:
        status = "CRITICAL"
        msg = (f"{direction.upper()} DIFFUSION VALIDATION FAILED\n"
               f"\n"
               f"  Problem: D = {D:.6f} cm²/ms is BELOW minimum stable value\n"
               f"           D_min = {D_min:.6f} cm²/ms for mesh dx = {dx_mm:.2f} mm, APD = {apd_ms:.0f} ms\n"
               f"\n"
               f"  Consequence: Propagation in {direction} direction WILL FAIL.\n"
               f"\n"
               f"  Solutions (choose one):\n"
               f"    1. Use finer mesh: dx <= {math.sqrt(D / K_BASE) * 10:.2f} mm\n"
               f"    2. Increase {direction} CV to >= {CV_EMPIRICAL_CONSTANT * math.sqrt(D_min):.4f} cm/ms\n"
               f"    3. Use normal APD parameters (avoid APD-shortening drugs)\n")

        if raise_error:
            raise ValueError(msg)

    elif D < D_safe:
        status = "WARNING"
        msg = (f"{direction.upper()} diffusion is MARGINAL:\n"
               f"  D = {D:.6f} < D_safe = {D_safe:.6f} (D_min × {safety_margin})\n"
               f"  Propagation may be unreliable. Consider using finer mesh or higher CV.")

    else:
        status = "OK"
        ratio = D / D_min
        msg = f"{direction.upper()} diffusion OK: D = {D:.6f} ({ratio:.1f}× D_min)"

    return status, msg

# Human ventricular CV targets (cm/ms)
CV_LONGITUDINAL_DEFAULT = 0.06   # 0.6 m/s
CV_TRANSVERSE_DEFAULT = 0.02    # 0.2 m/s


def compute_D_from_cv(
    cv: float,
    dx: float,
    dx_ref: float = 0.01,
    enforce_minimum: bool = True,
    warn: bool = True
) -> float:
    """
    Compute diffusion coefficient to achieve target CV at given mesh size.

    Uses empirical calibration from ORd model simulations.

    IMPORTANT: For very low CV values (below ~0.025 cm/ms), the computed D
    may be below the minimum required for stable propagation. This function
    enforces a minimum D to prevent conduction block.

    Parameters
    ----------
    cv : float
        Target conduction velocity (cm/ms). Note: 1 m/s = 0.1 cm/ms
    dx : float
        Mesh spacing (cm)
    dx_ref : float
        Reference mesh size for correction (cm)
    enforce_minimum : bool
        If True, clamp D to minimum stable value (default True)
    warn : bool
        If True, print warning when D is clamped (default True)

    Returns
    -------
    D : float
        Diffusion coefficient (cm^2/ms)
    """
    # Base D from empirical relationship: CV = k * sqrt(D)
    D_base = (cv / CV_EMPIRICAL_CONSTANT) ** 2

    # Mesh correction factor (empirically tuned)
    ratio = dx / dx_ref
    if ratio > 1.0:
        # Coarser mesh: reduce D to compensate for numerical dispersion
        correction = 1.0 / (1.0 + 0.04 * (ratio - 1.0))
    elif ratio < 1.0:
        # Finer mesh: slight increase
        correction = 1.0 + 0.02 * (1.0 - ratio)
    else:
        correction = 1.0

    D = D_base * correction

    # Enforce minimum D for stable propagation
    if enforce_minimum and D < D_MINIMUM_STABLE:
        if warn:
            cv_min = CV_EMPIRICAL_CONSTANT * math.sqrt(D_MINIMUM_STABLE)
            print(f"WARNING: Requested CV={cv:.4f} cm/ms requires D={D:.6f} cm²/ms, "
                  f"which is below minimum stable value D_min={D_MINIMUM_STABLE:.6f}.")
            print(f"         Clamping D to minimum. Actual CV will be ~{cv_min:.4f} cm/ms.")
        D = D_MINIMUM_STABLE

    return D


def compute_cv_from_D(D: float) -> float:
    """
    Estimate CV from diffusion coefficient using empirical relationship.

    Parameters
    ----------
    D : float
        Diffusion coefficient (cm^2/ms)

    Returns
    -------
    cv : float
        Estimated conduction velocity (cm/ms)
    """
    return CV_EMPIRICAL_CONSTANT * math.sqrt(D)


def get_diffusion_params(
    dx: float,
    cv_long: float = CV_LONGITUDINAL_DEFAULT,
    cv_trans: float = CV_TRANSVERSE_DEFAULT,
    apd_ms: float = APD_REF,
    validate: bool = True,
    warn: bool = True
) -> Tuple[float, float]:
    """
    Get diffusion coefficients for target CVs at given mesh size.

    Uses the mesh-dependent D_min(dx, APD) formula to validate that the
    requested CV values are achievable. If validation fails, provides
    actionable guidance.

    Parameters
    ----------
    dx : float
        Mesh spacing (cm)
    cv_long : float
        Target longitudinal CV (cm/ms), default 0.06 (0.6 m/s)
    cv_trans : float
        Target transverse CV (cm/ms), default 0.02 (0.2 m/s)
    apd_ms : float
        Expected APD in ms (default 280 for normal ORd). Use ~150 for
        APD-shortening scenarios.
    validate : bool
        If True, check D values against D_min(dx, APD) and warn (default True)
    warn : bool
        If True, print warnings for marginal/failing configurations (default True)

    Returns
    -------
    D_L, D_T : float
        Longitudinal and transverse diffusion coefficients (cm^2/ms)
    """
    # Compute D from CV using empirical relationship
    D_L = compute_D_from_cv(cv_long, dx, enforce_minimum=False, warn=False)
    D_T = compute_D_from_cv(cv_trans, dx, enforce_minimum=False, warn=False)

    if validate:
        # Validate both D values against mesh-dependent D_min
        status_L, msg_L = validate_D_for_mesh(D_L, dx, apd_ms, direction="longitudinal")
        status_T, msg_T = validate_D_for_mesh(D_T, dx, apd_ms, direction="transverse")

        if warn:
            if status_L == "CRITICAL":
                print(f"\n{'='*70}")
                print(msg_L)
                print(f"{'='*70}\n")
            elif status_L == "WARNING":
                print(f"WARNING: {msg_L}")

            if status_T == "CRITICAL":
                print(f"\n{'='*70}")
                print(msg_T)
                print(f"{'='*70}\n")
            elif status_T == "WARNING":
                print(f"WARNING: {msg_T}")

    return D_L, D_T


def compute_diffusion_tensor(
    theta: float,
    D_L: float,
    D_T: float
) -> Tuple[float, float, float]:
    """
    Compute full 2x2 diffusion tensor from fiber angle.

    Parameters
    ----------
    theta : float
        Fiber angle in radians (0 = fibers along x-axis)
    D_L : float
        Longitudinal diffusion coefficient (along fiber)
    D_T : float
        Transverse diffusion coefficient (across fiber)

    Returns
    -------
    D_xx, D_yy, D_xy : float
        Components of symmetric diffusion tensor
    """
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    cos2 = cos_t * cos_t
    sin2 = sin_t * sin_t

    D_xx = D_L * cos2 + D_T * sin2
    D_yy = D_L * sin2 + D_T * cos2
    D_xy = (D_L - D_T) * cos_t * sin_t

    return D_xx, D_yy, D_xy


class DiffusionOperator:
    """
    FVM-based diffusion operator for monodomain equation.

    GPU-accelerated using PyTorch conv2d for efficient computation.

    Supports:
    - Isotropic diffusion (D_L = D_T)
    - Anisotropic with uniform fiber angle
    - Anisotropic with spatially varying fiber field

    Parameters
    ----------
    ny, nx : int
        Grid dimensions
    dx, dy : float
        Grid spacing (cm)
    D_L : float
        Longitudinal diffusion coefficient (cm^2/ms)
    D_T : float
        Transverse diffusion coefficient (cm^2/ms)
    fiber_angle : float or Tensor
        Fiber angle in radians. If scalar, uniform orientation.
        If Tensor (ny, nx), spatially varying.
    device : str
        PyTorch device ('cuda' or 'cpu')
    dtype : torch.dtype
        Data type (default float64)
    """

    def __init__(
        self,
        ny: int,
        nx: int,
        dx: float,
        dy: float,
        D_L: float = 0.001,
        D_T: float = 0.00025,
        fiber_angle: Union[float, torch.Tensor] = 0.0,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float64
    ):
        self.ny = ny
        self.nx = nx
        self.dx = dx
        self.dy = dy
        self.D_L = D_L
        self.D_T = D_T
        self.device = device
        self.dtype = dtype

        # Check for isotropic case
        self.isotropic = (abs(D_L - D_T) < 1e-12)

        # Handle fiber angle
        if isinstance(fiber_angle, (int, float)):
            self.uniform_fiber = True
            self.fiber_angle = float(fiber_angle)
            # Precompute tensor components for uniform case
            self.D_xx, self.D_yy, self.D_xy = compute_diffusion_tensor(
                self.fiber_angle, D_L, D_T
            )
            self.D_xx_field = None
            self.D_yy_field = None
            self.D_xy_field = None
        else:
            self.uniform_fiber = False
            self.fiber_angle = fiber_angle.to(device=device, dtype=dtype)
            self._precompute_tensor_field()

        # Build convolution kernels for isotropic case
        if self.isotropic:
            self._build_isotropic_kernel()

    def _build_isotropic_kernel(self):
        """Build 5-point Laplacian kernel for isotropic diffusion."""
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2

        # Standard 5-point Laplacian stencil
        # [  0,      1/dy²,     0    ]
        # [1/dx², -2/dx²-2/dy², 1/dx²]
        # [  0,      1/dy²,     0    ]
        kernel = torch.zeros(3, 3, dtype=self.dtype, device=self.device)
        kernel[0, 1] = 1.0 / dy2  # North
        kernel[2, 1] = 1.0 / dy2  # South
        kernel[1, 0] = 1.0 / dx2  # West
        kernel[1, 2] = 1.0 / dx2  # East
        kernel[1, 1] = -2.0 / dx2 - 2.0 / dy2  # Center

        # Scale by diffusion coefficient
        self.kernel = (self.D_L * kernel).unsqueeze(0).unsqueeze(0)

    def _precompute_tensor_field(self):
        """Precompute diffusion tensor at each cell for varying fiber field."""
        cos_t = torch.cos(self.fiber_angle)
        sin_t = torch.sin(self.fiber_angle)
        cos2 = cos_t ** 2
        sin2 = sin_t ** 2

        self.D_xx_field = self.D_L * cos2 + self.D_T * sin2
        self.D_yy_field = self.D_L * sin2 + self.D_T * cos2
        self.D_xy_field = (self.D_L - self.D_T) * cos_t * sin_t

    def apply(self, V: torch.Tensor) -> torch.Tensor:
        """
        Apply diffusion operator to voltage field.

        Parameters
        ----------
        V : Tensor (ny, nx)
            Membrane potential field

        Returns
        -------
        diff : Tensor (ny, nx)
            ∇·(D·∇V)
        """
        if self.isotropic:
            return self._apply_isotropic(V)
        elif self.uniform_fiber:
            return self._apply_uniform_anisotropic(V)
        else:
            return self._apply_varying_anisotropic(V)

    def _apply_isotropic(self, V: torch.Tensor) -> torch.Tensor:
        """Apply isotropic diffusion using conv2d."""
        # Add batch and channel dimensions: (ny, nx) -> (1, 1, ny, nx)
        V_4d = V.unsqueeze(0).unsqueeze(0)

        # Pad with replicate for no-flux boundary conditions
        V_padded = F.pad(V_4d, (1, 1, 1, 1), mode='replicate')

        # Apply convolution
        diff = F.conv2d(V_padded, self.kernel)

        # Remove batch/channel dims
        return diff[0, 0]

    def _apply_uniform_anisotropic(self, V: torch.Tensor) -> torch.Tensor:
        """Apply anisotropic diffusion with uniform fiber angle."""
        ny, nx = V.shape
        dx, dy = self.dx, self.dy
        D_xx, D_yy, D_xy = self.D_xx, self.D_yy, self.D_xy

        # Pad V for boundary handling
        V_padded = F.pad(V.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')[0, 0]

        # Compute gradients using central differences
        # V_padded indices: original (i,j) is at (i+1, j+1) in padded

        # dV/dx at cell centers
        dVdx = (V_padded[1:-1, 2:] - V_padded[1:-1, :-2]) / (2 * dx)
        # dV/dy at cell centers
        dVdy = (V_padded[2:, 1:-1] - V_padded[:-2, 1:-1]) / (2 * dy)

        # For FVM, compute fluxes at faces and balance

        # East face flux: F_e = D_xx * dV/dx + D_xy * dV/dy
        # Use forward difference for dV/dx at east face
        dVdx_e = (V_padded[1:-1, 2:] - V_padded[1:-1, 1:-1]) / dx

        # dV/dy at east face (average of adjacent cells)
        # North-south gradient at (i, j) and (i, j+1)
        dVdy_left = (V_padded[2:, 1:-1] - V_padded[:-2, 1:-1]) / (2 * dy)
        dVdy_right = (V_padded[2:, 2:] - V_padded[:-2, 2:]) / (2 * dy)
        dVdy_e = 0.5 * (dVdy_left + dVdy_right)

        flux_e = D_xx * dVdx_e + D_xy * dVdy_e

        # West face flux
        dVdx_w = (V_padded[1:-1, 1:-1] - V_padded[1:-1, :-2]) / dx
        dVdy_left_w = (V_padded[2:, :-2] - V_padded[:-2, :-2]) / (2 * dy)
        dVdy_right_w = (V_padded[2:, 1:-1] - V_padded[:-2, 1:-1]) / (2 * dy)
        dVdy_w = 0.5 * (dVdy_left_w + dVdy_right_w)

        flux_w = D_xx * dVdx_w + D_xy * dVdy_w

        # North face flux: F_n = D_xy * dV/dx + D_yy * dV/dy
        dVdy_n = (V_padded[2:, 1:-1] - V_padded[1:-1, 1:-1]) / dy

        dVdx_bottom = (V_padded[1:-1, 2:] - V_padded[1:-1, :-2]) / (2 * dx)
        dVdx_top = (V_padded[2:, 2:] - V_padded[2:, :-2]) / (2 * dx)
        dVdx_n = 0.5 * (dVdx_bottom + dVdx_top)

        flux_n = D_xy * dVdx_n + D_yy * dVdy_n

        # South face flux
        dVdy_s = (V_padded[1:-1, 1:-1] - V_padded[:-2, 1:-1]) / dy

        dVdx_bottom_s = (V_padded[:-2, 2:] - V_padded[:-2, :-2]) / (2 * dx)
        dVdx_top_s = (V_padded[1:-1, 2:] - V_padded[1:-1, :-2]) / (2 * dx)
        dVdx_s = 0.5 * (dVdx_bottom_s + dVdx_top_s)

        flux_s = D_xy * dVdx_s + D_yy * dVdy_s

        # No-flux boundary conditions (zero flux at edges)
        # Create masks for boundary cells
        diff = torch.zeros_like(V)

        # Interior: full flux balance
        diff[1:-1, 1:-1] = ((flux_e[1:-1, 1:-1] - flux_w[1:-1, 1:-1]) / dx +
                            (flux_n[1:-1, 1:-1] - flux_s[1:-1, 1:-1]) / dy)

        # Edges: adjust for no-flux BC
        # Left edge (j=0): no west flux
        diff[1:-1, 0] = (flux_e[1:-1, 0] - 0.0) / dx + (flux_n[1:-1, 0] - flux_s[1:-1, 0]) / dy
        # Right edge (j=nx-1): no east flux
        diff[1:-1, -1] = (0.0 - flux_w[1:-1, -1]) / dx + (flux_n[1:-1, -1] - flux_s[1:-1, -1]) / dy
        # Bottom edge (i=0): no south flux
        diff[0, 1:-1] = (flux_e[0, 1:-1] - flux_w[0, 1:-1]) / dx + (flux_n[0, 1:-1] - 0.0) / dy
        # Top edge (i=ny-1): no north flux
        diff[-1, 1:-1] = (flux_e[-1, 1:-1] - flux_w[-1, 1:-1]) / dx + (0.0 - flux_s[-1, 1:-1]) / dy

        # Corners
        diff[0, 0] = flux_e[0, 0] / dx + flux_n[0, 0] / dy
        diff[0, -1] = -flux_w[0, -1] / dx + flux_n[0, -1] / dy
        diff[-1, 0] = flux_e[-1, 0] / dx - flux_s[-1, 0] / dy
        diff[-1, -1] = -flux_w[-1, -1] / dx - flux_s[-1, -1] / dy

        return diff

    def _apply_varying_anisotropic(self, V: torch.Tensor) -> torch.Tensor:
        """Apply anisotropic diffusion with spatially varying fiber field."""
        # Similar to uniform case but using tensor fields
        # This is a simplified implementation; for full accuracy would need
        # face-averaged tensor components

        ny, nx = V.shape
        dx, dy = self.dx, self.dy

        # Pad V
        V_padded = F.pad(V.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')[0, 0]

        # Compute gradients
        dVdx_e = (V_padded[1:-1, 2:] - V_padded[1:-1, 1:-1]) / dx
        dVdx_w = (V_padded[1:-1, 1:-1] - V_padded[1:-1, :-2]) / dx
        dVdy_n = (V_padded[2:, 1:-1] - V_padded[1:-1, 1:-1]) / dy
        dVdy_s = (V_padded[1:-1, 1:-1] - V_padded[:-2, 1:-1]) / dy

        # Cross-derivatives (averaged)
        dVdy_center = (V_padded[2:, 1:-1] - V_padded[:-2, 1:-1]) / (2 * dy)
        dVdx_center = (V_padded[1:-1, 2:] - V_padded[1:-1, :-2]) / (2 * dx)

        # Compute fluxes using cell-centered tensor
        D_xx = self.D_xx_field
        D_yy = self.D_yy_field
        D_xy = self.D_xy_field

        # East flux (use average of cell and east neighbor)
        flux_e = D_xx * dVdx_e + D_xy * dVdy_center
        flux_w = D_xx * dVdx_w + D_xy * dVdy_center
        flux_n = D_xy * dVdx_center + D_yy * dVdy_n
        flux_s = D_xy * dVdx_center + D_yy * dVdy_s

        # Flux balance
        diff = (flux_e - flux_w) / dx + (flux_n - flux_s) / dy

        # Apply no-flux BC by zeroing boundary fluxes
        # This is approximate; more accurate would reconstruct from scratch

        return diff

    def get_stability_limit(self, safety: float = 0.9) -> float:
        """
        Compute maximum stable time step for explicit integration.

        Uses von Neumann stability analysis for 2D diffusion equation.

        Parameters
        ----------
        safety : float
            Safety factor (0 < safety <= 1)

        Returns
        -------
        dt_max : float
            Maximum stable time step (ms)
        """
        # For anisotropic diffusion, use maximum eigenvalue
        D_max = max(self.D_L, self.D_T)

        # Stability criterion: dt <= dx² / (4 * D_max) for 2D
        dt_max = 0.25 * min(self.dx ** 2, self.dy ** 2) / D_max

        return safety * dt_max
