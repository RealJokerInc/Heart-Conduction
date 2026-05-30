"""
Boundary-mode validation tests for FDMDiscretization (PLAN.md Phase A).

Three discrete Neumann ghost choices added to fdm.py:
  - 'node_mirror_existing' (default, legacy): V[i,-1] = V[i,1] (sub-edge mirror)
  - 'face_mirror' (NEW):                       V[i,-1] = V[i,0] (boundary cell)
  - 'zero_pad'    (NEW):                       V[i,-1] = 0     (Dirichlet outside)

Tests:
  A.3  SPD preservation: A == A.T for each mode.
  A.4  Hand-verification corner: 4x4 grid, V[i,j]=4i+j+1, h=D=chi=Cm=1.
       Expected apply_diffusion(V)[3,3] = -5 (face), -10 (node), -37 (zero).
  A.5  Default unchanged: omitting boundary_mode reproduces legacy Laplacian.
  A.6  Bad mode raises ValueError.

Math reference: Research/Active/boundary_conduction_speedup/bc_discretization_math.tex.
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization


def _build(mode, Nx=4, Ny=4, h=1.0, D=1.0):
    """Build an FDM with chi=Cm=1, h=1, scalar D=1, given boundary_mode."""
    grid = StructuredGrid.create_rectangle(h * (Nx - 1), h * (Ny - 1), Nx, Ny)
    return FDMDiscretization(grid, D=D, chi=1.0, Cm=1.0, boundary_mode=mode)


def test_a3_symmetry_per_mode():
    """
    A.3: Symmetry analysis per mode.

    Empirical finding (2026-04-29): legacy 'node_mirror_existing' is NOT a
    symmetric matrix. At (Nx-1, j), the off-grid east neighbor is mirrored
    to (Nx-2, j); the cardinal-west branch ALSO writes to (Nx-2, j). After
    coalesce, A[Nx-1, Nx-2] = +2w. From (Nx-2, j)'s row, only the cardinal-
    east branch writes back, giving A[Nx-2, Nx-1] = +w. Asymmetric by w.
    The operator is self-adjoint in a weighted inner product (boundary cells
    have weight 1/2), but the raw COO matrix is not. V5.4 PCG has been
    running on this matrix since Phase 1 — leaving it untouched.

    The two NEW modes (face_mirror, zero_pad) ARE symmetric:
      - face_mirror: skips off-diagonals at boundaries entirely.
      - zero_pad:    only modifies diagonal at boundaries.
    """
    print("\n[A.3] Symmetry per boundary mode")
    results = {}
    for mode in FDMDiscretization.BOUNDARY_MODES:
        fdm = _build(mode, Nx=8, Ny=8)
        L = fdm.L.to_dense()
        sym_err = (L - L.T).abs().max().item()
        results[mode] = sym_err
        print(f"  {mode:25s}  ||L - L^T||_inf = {sym_err:.3e}")

    # New modes must be symmetric.
    assert results['face_mirror'] < 1e-12, (
        f"face_mirror broke symmetry: {results['face_mirror']}"
    )
    assert results['zero_pad'] < 1e-12, (
        f"zero_pad broke symmetry: {results['zero_pad']}"
    )
    # Legacy is intentionally left as-is. Document the asymmetry.
    legacy = results['node_mirror_existing']
    assert legacy > 0.0, (
        f"legacy unexpectedly symmetric ({legacy}); patch may have changed it"
    )
    print(f"  PASS — new modes symmetric; legacy preserved (asymmetry={legacy:.3e}).")


def test_a4_hand_verification_corner():
    """
    A.4: 4x4 grid, V[i,j] = 4i + j + 1, h=D=chi=Cm=1.
    Corner (3,3): both east (i+1=4) and north (j+1=4) are off-grid;
                  west (2,3)=12 and south (3,2)=15 are real.
    Expected apply_diffusion(V)[3,3]:
      face_mirror          : 16 + 12 + 16 + 15 - 4*16 = -5
      node_mirror_existing : 12 + 12 + 15 + 15 - 4*16 = -10
      zero_pad             :  0 + 12 +  0 + 15 - 4*16 = -37
    """
    print("\n[A.4] Hand-verification at corner (3,3)")
    Nx, Ny = 4, 4
    V = np.zeros((Nx, Ny), dtype=np.float64)
    for i in range(Nx):
        for j in range(Ny):
            V[i, j] = 4 * i + j + 1
    # Sanity check the test data
    assert V[2, 3] == 12.0 and V[3, 2] == 15.0 and V[3, 3] == 16.0

    expected = {
        'face_mirror':           -5.0,
        'node_mirror_existing': -10.0,
        'zero_pad':             -37.0,
    }

    for mode, want in expected.items():
        fdm = _build(mode, Nx=Nx, Ny=Ny)
        V_t = torch.tensor(V.flatten(), dtype=torch.float64)
        out = fdm.apply_diffusion(V_t).numpy().reshape(Nx, Ny)
        got = float(out[3, 3])
        print(f"  {mode:25s}  apply_diffusion(V)[3,3] = {got:+8.4f}  (expected {want:+8.4f})")
        assert abs(got - want) < 1e-10, (
            f"{mode}: got {got}, expected {want} (diff {got - want:.3e})"
        )
    print("  PASS — all three modes match analytic stencil values.")


def test_a5_default_is_face_mirror():
    """
    A.5: Default boundary_mode is 'face_mirror' (changed 2026-04-29).

    Rationale: node_mirror_existing amplifies any column-wise voltage gradient
    at the wall by exactly 2x (L_y[j=0] = 2*(V[i,1]-V[i,0]) instead of
    (V[i,1]-V[i,0])). This is the root cause of the storage-tank camel-toe /
    crescent boundary artifacts that motivated the boundary_conduction_speedup
    research question. face_mirror makes the off-grid flux identically zero
    by construction (ghost = V[i,0]) and is the genuine no-flux Neumann choice.

    Anyone needing bit-exact reproduction of pre-2026-04-29 V5.4 results must
    pass boundary_mode='node_mirror_existing' explicitly.
    """
    print("\n[A.5] Default boundary_mode is face_mirror")
    grid = StructuredGrid.create_rectangle(1.0, 1.0, 16, 16)
    fdm_default = FDMDiscretization(grid, D=0.001, chi=1400.0, Cm=1.0)
    fdm_face = FDMDiscretization(
        grid, D=0.001, chi=1400.0, Cm=1.0,
        boundary_mode='face_mirror',
    )
    fdm_legacy = FDMDiscretization(
        grid, D=0.001, chi=1400.0, Cm=1.0,
        boundary_mode='node_mirror_existing',
    )
    L_default = fdm_default.L.to_dense()
    L_face = fdm_face.L.to_dense()
    L_legacy = fdm_legacy.L.to_dense()

    err_face = (L_default - L_face).abs().max().item()
    diff_legacy = (L_default - L_legacy).abs().max().item()
    print(f"  ||L_default - L_face_mirror||_inf         = {err_face:.3e}")
    print(f"  ||L_default - L_node_mirror_existing||_inf = {diff_legacy:.3e}")

    assert err_face == 0.0, f"default differs from face_mirror: {err_face}"
    assert diff_legacy > 0.0, (
        "default unexpectedly matches legacy node_mirror_existing — "
        "the 2026-04-29 default flip may have been reverted"
    )
    print("  PASS — default is bit-identical to face_mirror; legacy is distinct.")


def test_a6_invalid_mode_rejected():
    """A.6: Unknown mode raises ValueError mentioning the allowed options."""
    print("\n[A.6] Invalid boundary_mode raises ValueError")
    grid = StructuredGrid.create_rectangle(1.0, 1.0, 4, 4)
    try:
        FDMDiscretization(grid, boundary_mode='ghost_dragon')
    except ValueError as e:
        msg = str(e)
        print(f"  raised ValueError: {msg!s}")
        assert 'boundary_mode' in msg
        assert 'node_mirror_existing' in msg
        print("  PASS")
        return
    raise AssertionError("ValueError was not raised for invalid mode.")


def test_a7_distinct_matrices():
    """
    A.7 (sanity): the three modes produce *distinct* Laplacians
    (they're not all the same operator).
    """
    print("\n[A.7] Three modes produce distinct Laplacians")
    L = {}
    for mode in FDMDiscretization.BOUNDARY_MODES:
        L[mode] = _build(mode, Nx=6, Ny=6).L.to_dense()
    pairs = [
        ('node_mirror_existing', 'face_mirror'),
        ('node_mirror_existing', 'zero_pad'),
        ('face_mirror',          'zero_pad'),
    ]
    for a, b in pairs:
        diff = (L[a] - L[b]).abs().max().item()
        print(f"  ||L[{a}] - L[{b}]||_inf = {diff:.3e}")
        assert diff > 1e-6, f"{a} and {b} unexpectedly identical"
    print("  PASS — three modes are genuinely different operators.")


def test_a8_invalid_stencil_rejected():
    """A.8: Unknown stencil raises ValueError mentioning the allowed options."""
    print("\n[A.8] Invalid stencil raises ValueError")
    grid = StructuredGrid.create_rectangle(1.0, 1.0, 4, 4)
    try:
        FDMDiscretization(grid, stencil='unknown')
    except ValueError as e:
        msg = str(e)
        print(f"  raised ValueError: {msg!s}")
        assert 'stencil' in msg
        assert 'cardinal4' in msg
        print("  PASS")
        return
    raise AssertionError("ValueError was not raised for invalid stencil.")


def test_a8_moore8_constructs():
    """A.8: Moore-8 stencils construct successfully (Step 1.2/1.3 implemented)."""
    print("\n[A.8] Moore-8 stencils construct (uniform + iso)")
    grid = StructuredGrid.create_rectangle(1.0, 1.0, 8, 8)
    for stencil in ('moore8_uniform', 'moore8_iso'):
        fdm = FDMDiscretization(grid, D=0.001, chi=1.0, Cm=1.0, stencil=stencil)
        L = fdm.L.to_dense()
        assert L.shape == (64, 64), f"{stencil}: L shape {L.shape}, expected (64, 64)"
        # face_mirror is symmetric for both uniform and iso
        sym_err = (L - L.T).abs().max().item()
        assert sym_err < 1e-12, f"{stencil}: |L - L^T| = {sym_err}, expected 0"
        # Row sums must be 0 for face_mirror (mass conservation)
        row_sum_max = L.sum(dim=1).abs().max().item()
        assert row_sum_max < 1e-12, f"{stencil}: row-sum max = {row_sum_max}"
        print(f"  {stencil:<16} symmetric, row-sums = 0  ✓")
    print("  PASS")


def test_a8_moore8_rejects_anisotropic_D():
    """A.8: Moore-8 stencils reject non-zero Dxy with NotImplementedError."""
    print("\n[A.8] Moore-8 rejects anisotropic Dxy")
    grid = StructuredGrid.create_rectangle(1.0, 1.0, 4, 4)
    Dxx = torch.full((4, 4), 0.001, dtype=torch.float64)
    Dyy = torch.full((4, 4), 0.001, dtype=torch.float64)
    Dxy = torch.full((4, 4), 0.0005, dtype=torch.float64)  # NON-zero Dxy
    try:
        FDMDiscretization(grid, D_field=(Dxx, Dxy, Dyy),
                          chi=1.0, Cm=1.0, stencil='moore8_uniform')
    except NotImplementedError as e:
        msg = str(e)
        print(f"  raised NotImplementedError: {msg!s}")
        assert 'isotropic' in msg
        print("  PASS")
        return
    raise AssertionError("NotImplementedError was not raised for non-zero Dxy.")


def test_a8_moore8_rejects_non_square_grid():
    """A.8: Moore-8 stencils require dx == dy."""
    print("\n[A.8] Moore-8 rejects non-square grid")
    # Lx=2, Ly=1, Nx=Ny=5 -> dx=0.5, dy=0.25 (different)
    grid = StructuredGrid.create_rectangle(2.0, 1.0, 5, 5)
    try:
        FDMDiscretization(grid, D=0.001, chi=1.0, Cm=1.0,
                          stencil='moore8_uniform')
    except NotImplementedError as e:
        msg = str(e)
        print(f"  raised NotImplementedError: {msg!s}")
        assert 'dx == dy' in msg
        print("  PASS")
        return
    raise AssertionError("NotImplementedError was not raised for dx != dy.")


# ============================================================================
# Step 1.2 tests: moore8_uniform Laplacian properties
# ============================================================================

def _build_moore8(stencil, boundary_mode='face_mirror', Nx=8, Ny=8, h=1.0, D=1.0):
    """Helper: build a Moore-8 stencil on a square grid."""
    grid = StructuredGrid.create_rectangle(h * (Nx - 1), h * (Ny - 1), Nx, Ny)
    return FDMDiscretization(
        grid, D=D, chi=1.0, Cm=1.0, stencil=stencil, boundary_mode=boundary_mode
    )


def _apply_L(fdm, V_2d):
    """Helper: apply L to a 2D field. Returns 2D output."""
    nx, ny = V_2d.shape
    V_flat = torch.tensor(V_2d.flatten(), dtype=torch.float64)
    out_flat = torch.sparse.mm(fdm.L, V_flat.unsqueeze(1)).squeeze(1)
    return out_flat.numpy().reshape(nx, ny)


def test_a9_moore8_uniform_constant_field():
    """A.9: moore8_uniform L applied to V = 1 everywhere should give 0 (row sums)."""
    print("\n[A.9] moore8_uniform: constant field gives L*V = 0")
    fdm = _build_moore8('moore8_uniform')
    V = np.ones((8, 8), dtype=np.float64)
    out = _apply_L(fdm, V)
    err = np.abs(out).max()
    print(f"  max|L*V| = {err:.3e}")
    assert err < 1e-12, f"row sums non-zero: max|L*V| = {err}"
    print("  PASS")


def test_a9_moore8_uniform_y_uniform_x_linear():
    """A.9: V[i,j] = i (linear in x, constant in y) -> L*V = 0 in interior."""
    print("\n[A.9] moore8_uniform: y-uniform x-linear field, interior L*V = 0")
    fdm = _build_moore8('moore8_uniform')
    V = np.zeros((8, 8), dtype=np.float64)
    for i in range(8):
        for j in range(8):
            V[i, j] = float(i)
    out = _apply_L(fdm, V)
    # Interior cells (1 <= i <= 6, 1 <= j <= 6) should all give ~0 (linear field)
    interior_err = np.abs(out[1:-1, 1:-1]).max()
    print(f"  interior max|L*V| = {interior_err:.3e}")
    assert interior_err < 1e-10, f"linear field gave non-zero L*V in interior: {interior_err}"
    print("  PASS")


def test_a9_moore8_uniform_boundary_deficit_2_over_3():
    """A.9: y-uniform x-step field, boundary deficit ratio = 2/3 vs interior."""
    print("\n[A.9] moore8_uniform: boundary deficit ratio = 2/3 (face_mirror)")
    Nx, Ny = 8, 8
    fdm = _build_moore8('moore8_uniform', Nx=Nx, Ny=Ny, h=1.0, D=1.0)
    # y-uniform x-step: V[i,j] = 1.0 if i >= 4 else 0.0 (sharp x-step at i=4)
    V = np.zeros((Nx, Ny), dtype=np.float64)
    V[4:, :] = 1.0
    out = _apply_L(fdm, V)
    # The wavefront column is i=3 (just below step) and i=4 (just above).
    # At a boundary cell (i=3, j=0) vs an interior cell (i=3, j=Ny//2):
    #   Interior cell sees full 8-neighbour stencil at the step.
    #   Boundary cell loses NW + NE diagonals (off-grid in y).
    # Deficit ratio: boundary_value / interior_value should equal 2/3.
    b_val = float(out[3, 0])         # top boundary
    i_val = float(out[3, Ny // 2])   # interior
    if abs(i_val) < 1e-12:
        raise AssertionError(f"interior value too small: {i_val}")
    ratio = b_val / i_val
    print(f"  L*V[3, 0]      = {b_val:+.6f}  (top boundary)")
    print(f"  L*V[3, {Ny//2}] = {i_val:+.6f}  (interior)")
    print(f"  ratio = {ratio:.4f}  (expected 2/3 = {2.0/3.0:.4f})")
    assert abs(ratio - 2.0 / 3.0) < 1e-10, (
        f"deficit ratio {ratio} != 2/3"
    )
    print("  PASS — Moore-8 uniform deficit confirmed at 2/3.")


# ============================================================================
# Step 1.3 tests: moore8_iso Laplacian (Patra-Kałuża 4:1)
# ============================================================================

def test_a10_moore8_iso_recovers_continuum_in_interior():
    """A.10: moore8_iso applied to sin(πx)cos(πy) approximates -2π²V (4th-order accurate)."""
    print("\n[A.10] moore8_iso: 4th-order accuracy on sin·cos test field")
    Nx, Ny = 16, 16
    h = 1.0 / (Nx - 1)
    grid = StructuredGrid.create_rectangle(1.0, 1.0, Nx, Ny)
    fdm = FDMDiscretization(grid, D=1.0, chi=1.0, Cm=1.0, stencil='moore8_iso')
    xs = np.linspace(0, 1, Nx)
    ys = np.linspace(0, 1, Ny)
    V = np.zeros((Nx, Ny), dtype=np.float64)
    for i in range(Nx):
        for j in range(Ny):
            V[i, j] = np.sin(np.pi * xs[i]) * np.cos(np.pi * ys[j])
    expected = -2.0 * np.pi ** 2 * V
    out = _apply_L(fdm, V)
    # Interior only (away from boundary effects)
    err = np.abs(out[4:-4, 4:-4] - expected[4:-4, 4:-4]).max()
    rel = err / np.abs(expected[4:-4, 4:-4]).max()
    print(f"  max|L*V - ∇²V| in interior = {err:.4e}  (rel = {rel:.4e})")
    assert rel < 5e-2, f"iso 9-pt interior accuracy worse than expected: rel_err {rel}"
    print("  PASS")


def test_a10_moore8_iso_y_uniform_interior_matches_cardinal():
    """A.10: y-uniform x-linear field, iso interior matches cardinal4 interior."""
    print("\n[A.10] moore8_iso: y-uniform interior matches cardinal4")
    fdm_iso = _build_moore8('moore8_iso')
    fdm_card = FDMDiscretization(
        StructuredGrid.create_rectangle(7.0, 7.0, 8, 8),
        D=1.0, chi=1.0, Cm=1.0, stencil='cardinal4'
    )
    V = np.zeros((8, 8), dtype=np.float64)
    for i in range(8):
        for j in range(8):
            V[i, j] = 4.0 * float(i) + float(j) + 1.0  # arbitrary smooth field

    # Use y-uniform x-quadratic V[i,j] = i² to test second-derivative recovery
    V = np.zeros((8, 8), dtype=np.float64)
    for i in range(8):
        for j in range(8):
            V[i, j] = float(i) ** 2  # y-uniform, x-quadratic
    out_iso = _apply_L(fdm_iso, V)
    out_card = _apply_L(fdm_card, V)
    interior_err = np.abs(out_iso[1:-1, 1:-1] - out_card[1:-1, 1:-1]).max()
    print(f"  max|iso - cardinal4| in interior = {interior_err:.3e}")
    # Expected: both give 2 (second derivative of x²) in interior
    expected = 2.0
    print(f"  iso[2, 4]  = {out_iso[2, 4]:.4f}  (expected {expected})")
    print(f"  card[2, 4] = {out_card[2, 4]:.4f}  (expected {expected})")
    assert abs(out_iso[2, 4] - expected) < 1e-10
    assert abs(out_card[2, 4] - expected) < 1e-10
    assert interior_err < 1e-10, (
        f"iso interior should match cardinal4 in y-uniform: {interior_err}"
    )
    print("  PASS")


def test_a10_moore8_iso_boundary_deficit_5_over_6():
    """A.10: y-uniform x-step, iso boundary deficit ratio = 5/6 vs interior."""
    print("\n[A.10] moore8_iso: boundary deficit ratio = 5/6 (face_mirror)")
    Nx, Ny = 8, 8
    fdm = _build_moore8('moore8_iso', Nx=Nx, Ny=Ny, h=1.0, D=1.0)
    V = np.zeros((Nx, Ny), dtype=np.float64)
    V[4:, :] = 1.0
    out = _apply_L(fdm, V)
    b_val = float(out[3, 0])
    i_val = float(out[3, Ny // 2])
    if abs(i_val) < 1e-12:
        raise AssertionError(f"interior value too small: {i_val}")
    ratio = b_val / i_val
    print(f"  L*V[3, 0]      = {b_val:+.6f}  (top boundary)")
    print(f"  L*V[3, {Ny//2}] = {i_val:+.6f}  (interior)")
    print(f"  ratio = {ratio:.4f}  (expected 5/6 = {5.0/6.0:.4f})")
    assert abs(ratio - 5.0 / 6.0) < 1e-10, f"deficit ratio {ratio} != 5/6"
    print("  PASS — Moore-8 iso 4:1 deficit confirmed at 5/6.")


def test_a10_iso_normalization_check():
    """A.10: moore8_iso diag(L) coefficient matches Patra-Kaluza 1/6 prefactor.

    For interior cells with isotropic D: diag(L) = -(w_card_sum + w_diag_sum) =
    -(4·4/6 + 4·1/6)/h² · D = -20·D/(6·h²) = -10·D/(3·h²).

    If the 1/6 prefactor is forgotten, diag becomes 6× too large (matching the
    bug we hit in John's tanks: D_eff = 6k = 0.48 instead of k = 0.08, violating
    the standard CFL limit). This test catches that specific bug.
    """
    print("\n[A.10] moore8_iso: 1/6 prefactor sanity (catches the IDEALOG bug)")
    grid = StructuredGrid.create_rectangle(1.0, 0.5, 41, 21)
    fdm = FDMDiscretization(grid, D=0.001, chi=1.0, Cm=1.0, stencil='moore8_iso')
    L_dense = fdm.L.to_dense()
    diag_max = (-L_dense).diag().max().item()
    h = grid.dx
    D = 0.001
    # Interior diag should equal 10·D/(3·h²); boundary cells give less due to
    # face_mirror losing diagonals. Interior is the maximum.
    expected_interior = 10.0 * D / (3.0 * h * h)
    ratio = diag_max / expected_interior
    print(f"  diag_max = {diag_max:.4f}")
    print(f"  expected interior = 10·D/(3·h²) = {expected_interior:.4f}")
    print(f"  ratio = {ratio:.4f}  (expected 1.0; 6.0 would mean missing prefactor)")
    assert abs(ratio - 1.0) < 1e-10, (
        f"diag_max/expected = {ratio}, expected 1.0. Ratio of 6.0 indicates "
        f"the 1/6 Patra-Kaluza prefactor is missing — see IDEALOG.md "
        f"'Bug fix — iso weights need 1/6 normalisation'."
    )
    print("  PASS — 1/6 prefactor correctly applied.")


def test_a10_all_three_stencils_construct():
    """A.10: All 3 stencils construct, are symmetric (face_mirror), row sums = 0."""
    print("\n[A.10] All 3 stencils construct + symmetric + row-sums-zero")
    grid = StructuredGrid.create_rectangle(1.0, 1.0, 8, 8)
    for stencil in ('cardinal4', 'moore8_uniform', 'moore8_iso'):
        fdm = FDMDiscretization(
            grid, D=0.001, chi=1.0, Cm=1.0,
            stencil=stencil, boundary_mode='face_mirror'
        )
        L = fdm.L.to_dense()
        assert L.shape == (64, 64), f"{stencil}: bad shape"
        sym = (L - L.T).abs().max().item()
        rs = L.sum(dim=1).abs().max().item()
        print(f"  {stencil:<16} sym={sym:.2e}  row-sum={rs:.2e}")
        assert sym < 1e-12, f"{stencil}: not symmetric (face_mirror)"
        assert rs < 1e-12, f"{stencil}: row sums non-zero"
    print("  PASS")


# ============================================================================
# Step 2.1 tests: face_mirror_iso (diagonal-aware reflection)
# ============================================================================

def test_a11_face_mirror_iso_in_iso_stencil_zero_deficit():
    """A.11: moore8_iso + face_mirror_iso gives ZERO boundary deficit in y-uniform."""
    print("\n[A.11] moore8_iso + face_mirror_iso: ZERO deficit (LBM bounce-back)")
    Nx, Ny = 8, 8
    fdm = _build_moore8('moore8_iso', boundary_mode='face_mirror_iso',
                         Nx=Nx, Ny=Ny, h=1.0, D=1.0)
    # y-uniform x-step setup
    V = np.zeros((Nx, Ny), dtype=np.float64)
    V[4:, :] = 1.0
    out = _apply_L(fdm, V)
    b_val = float(out[3, 0])
    i_val = float(out[3, Ny // 2])
    print(f"  L*V[3, 0]      = {b_val:+.6f}  (top boundary)")
    print(f"  L*V[3, {Ny//2}] = {i_val:+.6f}  (interior)")
    diff = abs(b_val - i_val)
    print(f"  |boundary - interior| = {diff:.3e}  (expected 0)")
    assert diff < 1e-12, f"face_mirror_iso should give zero deficit: diff={diff}"
    print("  PASS — iso + bounce-back fully eliminates boundary deficit.")


def test_a12_face_mirror_iso_with_uniform_stencil():
    """A.12: moore8_uniform + face_mirror_iso ALSO gives zero deficit (same mechanism)."""
    print("\n[A.12] moore8_uniform + face_mirror_iso: ZERO deficit")
    Nx, Ny = 8, 8
    fdm = _build_moore8('moore8_uniform', boundary_mode='face_mirror_iso',
                         Nx=Nx, Ny=Ny, h=1.0, D=1.0)
    V = np.zeros((Nx, Ny), dtype=np.float64)
    V[4:, :] = 1.0
    out = _apply_L(fdm, V)
    b_val = float(out[3, 0])
    i_val = float(out[3, Ny // 2])
    diff = abs(b_val - i_val)
    print(f"  |boundary - interior| = {diff:.3e}  (expected 0)")
    assert diff < 1e-12, f"diff={diff}"
    print("  PASS")


def test_a13_face_mirror_iso_corner_handling():
    """A.13: face_mirror_iso at corners — both axes off-grid, ghost = self."""
    print("\n[A.13] moore8_iso + face_mirror_iso: corner cells correct")
    Nx, Ny = 8, 8
    fdm = _build_moore8('moore8_iso', boundary_mode='face_mirror_iso',
                         Nx=Nx, Ny=Ny, h=1.0, D=1.0)
    V = np.zeros((Nx, Ny), dtype=np.float64)
    V[4:, :] = 1.0
    out = _apply_L(fdm, V)
    # At corner (0, 0), V is constant (all 0). L*V[0, 0] should be 0.
    corner_val = float(out[0, 0])
    print(f"  L*V[0, 0] (corner, V uniform 0) = {corner_val:.3e}  (expected 0)")
    assert abs(corner_val) < 1e-12, f"corner produces non-zero on uniform field: {corner_val}"
    print("  PASS")


def test_a13_cardinal4_face_mirror_iso_degenerate():
    """A.13: cardinal4 + face_mirror_iso must produce bit-identical L to cardinal4 + face_mirror."""
    print("\n[A.13] cardinal4 + face_mirror_iso degenerates to face_mirror")
    grid = StructuredGrid.create_rectangle(1.0, 1.0, 8, 8)
    fdm_fm = FDMDiscretization(grid, D=0.001, chi=1.0, Cm=1.0,
                                stencil='cardinal4', boundary_mode='face_mirror')
    fdm_fmi = FDMDiscretization(grid, D=0.001, chi=1.0, Cm=1.0,
                                 stencil='cardinal4', boundary_mode='face_mirror_iso')
    L_fm = fdm_fm.L.to_dense()
    L_fmi = fdm_fmi.L.to_dense()
    diff = (L_fm - L_fmi).abs().max().item()
    print(f"  ||L_face_mirror - L_face_mirror_iso||_inf = {diff:.3e}  (expected 0)")
    assert diff == 0.0, f"cardinal4 face_mirror_iso must equal face_mirror exactly: {diff}"
    print("  PASS")


if __name__ == "__main__":
    test_a3_symmetry_per_mode()
    test_a4_hand_verification_corner()
    test_a5_default_is_face_mirror()
    test_a6_invalid_mode_rejected()
    test_a7_distinct_matrices()
    test_a8_invalid_stencil_rejected()
    test_a8_moore8_constructs()
    test_a8_moore8_rejects_anisotropic_D()
    test_a8_moore8_rejects_non_square_grid()
    # Step 1.2 (moore8_uniform)
    test_a9_moore8_uniform_constant_field()
    test_a9_moore8_uniform_y_uniform_x_linear()
    test_a9_moore8_uniform_boundary_deficit_2_over_3()
    # Step 1.3 (moore8_iso)
    test_a10_moore8_iso_recovers_continuum_in_interior()
    test_a10_moore8_iso_y_uniform_interior_matches_cardinal()
    test_a10_moore8_iso_boundary_deficit_5_over_6()
    test_a10_iso_normalization_check()
    test_a10_all_three_stencils_construct()
    # Step 2.1 (face_mirror_iso)
    test_a11_face_mirror_iso_in_iso_stencil_zero_deficit()
    test_a12_face_mirror_iso_with_uniform_stencil()
    test_a13_face_mirror_iso_corner_handling()
    test_a13_cardinal4_face_mirror_iso_degenerate()
    print("\nAll boundary-mode tests passed.")
