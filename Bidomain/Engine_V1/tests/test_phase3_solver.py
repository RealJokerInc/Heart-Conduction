"""
Phase 3 Validation Tests — Linear Solvers

Tests 3-T1 through 3-T5 from PROGRESS.md.
(3-T6, 3-T7 skipped — GMG is a stub.)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
import numpy as np


# === 3-T1: SpectralSolver DCT (Neumann) ===

def test_spectral_neumann():
    """3-T1: Self-consistency test — solve(-D*Lap*u=b) via DCT round-trip."""
    from cardiac_sim.simulation.classical.solver.linear_solver.spectral import SpectralSolver

    nx, ny = 32, 32
    dx, dy = 0.05, 0.05
    D = 0.0057

    solver = SpectralSolver(nx, ny, dx, dy, D, bc_type='neumann')
    solver._compute_eigenvalues(torch.device('cpu'), torch.float64)

    # Pick u with zero mean (Neumann null space)
    torch.manual_seed(42)
    u = torch.randn(nx, ny, dtype=torch.float64)
    u -= u.mean()

    # Forward operator: b = IDCT(eigenvalues * DCT(u))
    import torch_dct
    u_hat = torch_dct.dct_2d(u, norm='ortho')
    b_hat = u_hat * solver._eigenvalues
    b_hat[0, 0] = 0.0  # Zero-mean RHS consistent with zero-mean solution
    b = torch_dct.idct_2d(b_hat, norm='ortho')

    # Solve
    u_solved = solver.solve(None, b.flatten()).reshape(nx, ny)

    err = torch.abs(u_solved - u).max().item()
    assert err < 1e-10, f"Spectral Neumann self-consistency error: {err}"


# === 3-T2: SpectralSolver DST (Dirichlet) ===

def test_spectral_dirichlet():
    """3-T2: Self-consistency test — solve via DST round-trip on interior grid."""
    from cardiac_sim.simulation.classical.solver.linear_solver.spectral import SpectralSolver

    nx, ny = 32, 32
    dx, dy = 0.05, 0.05
    D = 0.0057

    solver = SpectralSolver(nx, ny, dx, dy, D, bc_type='dirichlet')
    solver._compute_eigenvalues(torch.device('cpu'), torch.float64)

    # Pick u on interior grid (nx-2 x ny-2)
    mx, my = nx - 2, ny - 2
    torch.manual_seed(42)
    u_int = torch.randn(mx, my, dtype=torch.float64)

    # Forward operator on interior: b_int = IDST(eigenvalues * DST(u_int))
    import scipy.fft
    u_np = u_int.numpy()
    u_hat_np = scipy.fft.dstn(u_np, type=1)
    b_hat_np = u_hat_np * solver._eigenvalues.numpy()
    b_int_np = scipy.fft.idstn(b_hat_np, type=1)
    b_int = torch.from_numpy(b_int_np)

    # Pad to full grid for the solver
    b_full = torch.zeros(nx, ny, dtype=torch.float64)
    b_full[1:-1, 1:-1] = b_int

    # Solve
    u_solved = solver.solve(None, b_full.flatten()).reshape(nx, ny)

    # Check interior
    err = torch.abs(u_solved[1:-1, 1:-1] - u_int).max().item()
    assert err < 1e-10, f"Spectral Dirichlet self-consistency error: {err}"

    # Boundary should be zero
    assert u_solved[0, :].abs().max() < 1e-14
    assert u_solved[-1, :].abs().max() < 1e-14
    assert u_solved[:, 0].abs().max() < 1e-14
    assert u_solved[:, -1].abs().max() < 1e-14


# === 3-T3: SpectralSolver FFT (Periodic) ===

def test_spectral_periodic():
    """3-T3: Self-consistency test — solve via FFT round-trip."""
    from cardiac_sim.simulation.classical.solver.linear_solver.spectral import SpectralSolver

    nx, ny = 32, 32
    dx, dy = 0.05, 0.05
    D = 0.0057

    solver = SpectralSolver(nx, ny, dx, dy, D, bc_type='periodic')
    solver._compute_eigenvalues(torch.device('cpu'), torch.float64)

    # Pick u with zero mean
    torch.manual_seed(42)
    u = torch.randn(nx, ny, dtype=torch.float64)
    u -= u.mean()

    # Forward operator: b = IFFT(eigenvalues * FFT(u))
    u_hat = torch.fft.fft2(u)
    b_hat = u_hat * solver._eigenvalues
    b_hat[0, 0] = 0.0
    b = torch.fft.ifft2(b_hat).real

    # Solve
    u_solved = solver.solve(None, b.flatten()).reshape(nx, ny)

    err = torch.abs(u_solved - u).max().item()
    assert err < 1e-10, f"Spectral Periodic self-consistency error: {err}"


# === 3-T2b: SpectralSolver Dirichlet with analytical function ===

def test_spectral_dirichlet_analytical():
    """SpectralSolver Dirichlet gives O(h^2) convergence for analytical Poisson."""
    from cardiac_sim.simulation.classical.solver.linear_solver.spectral import SpectralSolver

    D = 0.0057
    Lx, Ly = 1.0, 1.0
    errors = []

    for nx in [16, 32, 64]:
        ny = nx
        dx = Lx / (nx - 1)
        dy = Ly / (ny - 1)
        solver = SpectralSolver(nx, ny, dx, dy, D, bc_type='dirichlet')

        kx, ky = np.pi / Lx, np.pi / Ly
        x = torch.arange(nx, dtype=torch.float64) * dx
        y = torch.arange(ny, dtype=torch.float64) * dy
        X, Y = torch.meshgrid(x, y, indexing='ij')
        u_exact = torch.sin(kx * X) * torch.sin(ky * Y)
        f = D * (kx**2 + ky**2) * u_exact

        u_solved = solver.solve(None, f.flatten()).reshape(nx, ny)

        # Check interior
        err = torch.abs(u_solved[1:-1, 1:-1] - u_exact[1:-1, 1:-1]).max().item()
        errors.append(err)

    ratio1 = errors[0] / errors[1]
    ratio2 = errors[1] / errors[2]
    assert ratio1 > 3.0, f"Dirichlet convergence ratio 1: {ratio1:.2f}"
    assert ratio2 > 3.0, f"Dirichlet convergence ratio 2: {ratio2:.2f}"


# === 3-T4: PCGSpectralSolver (anisotropic Neumann) ===

def test_pcg_spectral_neumann():
    """3-T4: PCG+Spectral on anisotropic problem, Neumann BCs.

    Uses unpinned PSD matrix — null space handled consistently by
    spectral preconditioner (zero-mean projection).
    """
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg_spectral import PCGSpectralSolver
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import sparse_mv

    nx, ny = 20, 20
    Lx, Ly = 1.0, 1.0
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    D_xx, D_yy = 0.006, 0.004
    D_avg = (D_xx + D_yy) / 2.0

    # Build anisotropic -Lap with Neumann BCs (face-based: skip out-of-domain)
    N = nx * ny
    rows, cols, vals = [], [], []
    for i in range(nx):
        for j in range(ny):
            k = i * ny + j
            center = 0.0
            if i > 0:
                w = D_xx / dx**2
                rows.append(k); cols.append(k - ny); vals.append(w)
                center -= w
            if i < nx - 1:
                w = D_xx / dx**2
                rows.append(k); cols.append(k + ny); vals.append(w)
                center -= w
            if j > 0:
                w = D_yy / dy**2
                rows.append(k); cols.append(k - 1); vals.append(w)
                center -= w
            if j < ny - 1:
                w = D_yy / dy**2
                rows.append(k); cols.append(k + 1); vals.append(w)
                center -= w
            rows.append(k); cols.append(k); vals.append(center)

    # A = -L (positive semi-definite for Neumann, no pinning)
    A = -torch.sparse_coo_tensor(
        torch.tensor([rows, cols], dtype=torch.long),
        torch.tensor(vals, dtype=torch.float64),
        size=(N, N)
    ).coalesce()

    # Known solution with zero mean (in range of A)
    x = torch.arange(nx, dtype=torch.float64) * dx
    y = torch.arange(ny, dtype=torch.float64) * dy
    X, Y = torch.meshgrid(x, y, indexing='ij')
    u_exact = torch.cos(np.pi * X / Lx) * torch.cos(np.pi * Y / Ly)
    u_exact_flat = u_exact.flatten()
    u_exact_flat = u_exact_flat - u_exact_flat.mean()  # Zero mean

    # RHS is in range of A (sum = 0 since A has zero row sum)
    b = sparse_mv(A, u_exact_flat)

    solver = PCGSpectralSolver(nx, ny, dx, dy, D_avg, bc_type='neumann',
                                max_iters=50, tol=1e-10)
    u_solved = solver.solve(A, b)

    # Compare zero-mean solutions (unique up to constant for Neumann)
    u_solved_zm = u_solved - u_solved.mean()

    err = torch.abs(u_solved_zm - u_exact_flat).max().item()
    assert err < 1e-6, f"PCGSpectral Neumann error: {err}"
    assert solver.last_iters <= 20, f"Too many iterations: {solver.last_iters}"


# === 3-T5: PCGSpectralSolver DST mode (anisotropic Dirichlet) ===

def test_pcg_spectral_dirichlet():
    """3-T5: PCG+Spectral on anisotropic problem, Dirichlet BCs."""
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg_spectral import PCGSpectralSolver
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg import sparse_mv

    nx, ny = 20, 20
    Lx, Ly = 1.0, 1.0
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    D_xx, D_yy = 0.006, 0.004
    D_avg = (D_xx + D_yy) / 2.0

    # Build anisotropic -Lap with Dirichlet BCs
    N = nx * ny
    rows, cols, vals = [], [], []
    for i in range(nx):
        for j in range(ny):
            k = i * ny + j
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                rows.append(k); cols.append(k); vals.append(1.0)
                continue
            center = 0.0
            w = D_xx / dx**2
            rows.append(k); cols.append(k - ny); vals.append(w)
            rows.append(k); cols.append(k + ny); vals.append(w)
            center -= 2 * w
            w = D_yy / dy**2
            rows.append(k); cols.append(k - 1); vals.append(w)
            rows.append(k); cols.append(k + 1); vals.append(w)
            center -= 2 * w
            rows.append(k); cols.append(k); vals.append(-center)

    A = torch.sparse_coo_tensor(
        torch.tensor([rows, cols], dtype=torch.long),
        torch.tensor(vals, dtype=torch.float64),
        size=(N, N)
    ).coalesce()

    # Known solution: sin(pi*x)*sin(pi*y) (zero on boundary)
    x = torch.arange(nx, dtype=torch.float64) * dx
    y = torch.arange(ny, dtype=torch.float64) * dy
    X, Y = torch.meshgrid(x, y, indexing='ij')
    u_exact = torch.sin(np.pi * X / Lx) * torch.sin(np.pi * Y / Ly)
    u_exact_flat = u_exact.flatten()

    b = sparse_mv(A, u_exact_flat)

    solver = PCGSpectralSolver(nx, ny, dx, dy, D_avg, bc_type='dirichlet',
                                max_iters=50, tol=1e-10)
    u_solved = solver.solve(A, b)

    err = torch.abs(u_solved - u_exact_flat).max().item()
    assert err < 1e-6, f"PCGSpectral Dirichlet error: {err}"
    assert solver.last_iters <= 20, f"Too many iterations: {solver.last_iters}"


# === 3-T6, 3-T7: GMG stubs — skip ===

def test_gmg_stub_raises():
    """3-T6/T7: GMG and PCG+GMG stubs raise NotImplementedError."""
    from cardiac_sim.simulation.classical.solver.linear_solver.multigrid import GeometricMultigridPreconditioner
    from cardiac_sim.simulation.classical.solver.linear_solver.pcg_gmg import EllipticPCGMGSolver

    with pytest.raises(NotImplementedError):
        GeometricMultigridPreconditioner(32, 32, 0.01, 0.01, 0.005)

    with pytest.raises(NotImplementedError):
        EllipticPCGMGSolver(32, 32, 0.01, 0.01, 0.005)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
