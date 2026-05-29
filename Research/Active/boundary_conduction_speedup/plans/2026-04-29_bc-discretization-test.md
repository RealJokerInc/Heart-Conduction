# PLAN — Boundary BC Discretization Test

> Goal: classify whether the persistent boundary artifacts seen in John's
> storage-tank model (camel toe under reflect_y, crescent under zero_pad)
> reflect a real continuum effect or a numerical artifact of the specific
> Neumann discretization scheme. Verify by adding boundary-discretization
> variants to the V5.4 monodomain FDM and the LBM Engine V1, measuring
> per-column LAT deviation, and checking h-scaling.
>
> Created: 2026-04-29.  Revised twice (2026-04-29 audit-1, 2026-04-29 audit-2)
> — see Mutation Log. All API calls in this version are grounded against
> verbatim engine source via the API-reference card produced by Explore.
>
> See `IDEALOG.md` (2026-04-29 entry) and `KNOWLEDGE.md` ("Boundary BC
> discretization") for the analysis and predictions this plan operationalizes.

## Hypothesis being tested

**H_BC**: Storage-tank `zero_pad` and `reflect_y` use boundary discretizations
(amputated stencil and node-centered mirror with mass-conserving fold,
respectively) that don't cancel missing-channel asymmetry. A *face-centered*
mirror Neumann BC DOES cancel the asymmetry exactly. This plan adds
`face_mirror`, `zero_pad`, and (deferred) `node_mirror_with_fold` modes to
the V5.4 monodomain FDMDiscretization, then verifies that:

1. `face_mirror` is asymmetry-free (Δ ≈ 0 at all h).
2. `zero_pad` reproduces a crescent that vanishes as h → 0.
3. (Deferred) `node_mirror_with_fold` reproduces a camel toe that vanishes
   as h → 0.
4. The bidomain bath-coupled Kleber speedup is constant in h (real continuum
   effect).

## Predictions (revised, with chi standardized)

| variant                                                 | Δ@x_mid (sign / mag) | h-scaling         |
|---------------------------------------------------------|----------------------|-------------------|
| monodomain `face_mirror` (NEW)                          | ≈ 0                  | flat at all h     |
| monodomain `node_mirror_existing` (V5.4 current default)| TBD — measure        | TBD — measure     |
| monodomain `zero_pad` (NEW)                             | + (crescent expected)| empirical scaling exponent |
| monodomain `node_mirror_with_fold` (DEFERRED to Phase F)| − (camel expected)   | n/a in this round |
| LBM `bounce_back` (existing)                            | ≈ 0                  | flat              |
| LBM `vacuum` (NEW, V_outside=0)                         | + (crescent expected)| empirical scaling |
| Bidomain V1 monodomain mode (Mehrstellen)               | ≈ 0 (already measured at 0.000 cm) | flat |
| bidomain bath-coupled                                   | − ~7% CV (Kleber)    | ~constant in h (continuum) |

The h-scaling test is the diagnostic: artifacts that vanish as h → 0 are
discretization choices; artifacts that survive are real physics. **Empirical
measurement is required — no a-priori O(h¹) assumption.**

**Standardized parameters** (all variants must use these so that cross-engine
comparison is valid):
- **chi = 1400** cm⁻¹ (V5.4 default; matches FDMDiscretization)
- **Cm = 1.0** µF/cm²
- **D = 0.001** cm²/ms (intracellular conductivity / chi*Cm)
- **dx ∈ {0.05, 0.025, 0.0125}** cm
- **dt = 0.02** ms (ionic step), implicit Crank-Nicolson for diffusion
- **TTP06 EPI** ionic model, V_rest = -85.23 mV
- **Stimulus**: line-source at x ∈ [0, 0.1] cm, depolarizing current
  `amplitude = -52.0` µA/µF, duration = 5.0 ms, start_time = 0.0 ms

---

## Phase A — Add `boundary_mode` to V5.4 FDMDiscretization

**Goal**: extend `Monodomain/Engine_V5.4/cardiac_sim/simulation/classical/
discretization_scheme/fdm.py` with a `boundary_mode` parameter on the
`FDMDiscretization` class. Add two NEW modes (`face_mirror`, `zero_pad`)
alongside the existing default (renamed to `node_mirror_existing`). Defer
`node_mirror_with_fold` to a follow-up phase F because V5.4 has no
non-symmetric solver (PCG/Chebyshev/DCT/FFT only — no GMRES; verified
against `cardiac_sim/simulation/classical/solver/diffusion_time_stepping/
linear_solver/`).

### Step A.0 — Locate and audit current Neumann handling [READ-ONLY]

**Read first**: API reference card (verified 2026-04-29); `KNOWLEDGE.md`
"Why face-centered cancels boundary asymmetry exactly" section.

**Why**: confirm the exact assembly logic before modification.

**Implementation spec**:
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.4
sed -n '260,360p' cardiac_sim/simulation/classical/discretization_scheme/fdm.py
```

Confirm:
- Line ~272: `if i+1 >= nx: A[k, idx(i-1, j)] += w_east` (East ghost mirrors to sub-edge — node-centered mirror)
- Line ~288: same for West ghost: `A[k, idx(i+1, j)] += w_west`
- Symmetric handling for North (j+1>=ny → mirror to j-1) and South (j-1<0 → mirror to j+1)
- Line 322-349: 9-point diagonal corners are SKIPPED at rectangle boundary (a known stencil incompleteness — comment says "negligible correction term")
- Line 282-286 ALREADY adds the cardinal-west entry on a separate code path. The mirror code at line 271-278 adds a SECOND entry to the SAME `(i-1, j)` matrix slot. After `A.coalesce()`, the boundary cell's matrix entry to `(i-1, j)` is `2 * w_east` (audit #2 N3 finding).

**Test spec**: none (read-only).

**Verify**: written notes capturing the above details.

**Exit criteria**: ready to write Step A.1 against the real assembly loop.

**Risk**: none.

### Step A.1 — Add `boundary_mode` parameter

**Read first**: Step A.0 notes; `cardiac_sim/simulation/classical/discretization_scheme/base.py` (SpatialDiscretization ABC).

**Why**: provide a single switch on the FDMDiscretization constructor.

**Implementation spec**: in `fdm.py`, modify the `FDMDiscretization` class:

```python
# In FDMDiscretization class (around line 29):

class FDMDiscretization(SpatialDiscretization):
    """Finite Difference Method (FDM) spatial discretization."""

    BOUNDARY_MODES = (
        'node_mirror_existing',  # legacy default — V_ghost = V[i-1] (sub-edge mirror)
        'face_mirror',           # NEW — V_ghost = V[i] (boundary itself); face-centered Neumann
        'zero_pad',              # NEW — V_ghost = 0 (amputated; equivalent to V_outside = 0)
    )

    def __init__(
        self,
        grid: StructuredGrid,
        D: float = 0.001,
        chi: float = 1400.0,
        Cm: float = 1.0,
        D_field: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        boundary_mode: str = 'node_mirror_existing',  # NEW kwarg
    ):
        if boundary_mode not in self.BOUNDARY_MODES:
            raise ValueError(
                f"boundary_mode must be one of {self.BOUNDARY_MODES}; got {boundary_mode!r}"
            )
        self.boundary_mode = boundary_mode
        # ... existing __init__ body ...
```

**Test spec**: a smoke test that constructing each variant doesn't crash:
```python
import pytest
import torch
from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization

def test_construct_each_boundary_mode():
    grid = StructuredGrid.create_rectangle(Lx=1.0, Ly=1.0, Nx=11, Ny=11, device='cpu')
    for mode in FDMDiscretization.BOUNDARY_MODES:
        s = FDMDiscretization(grid=grid, D=0.001, chi=1400.0, Cm=1.0, boundary_mode=mode)
        assert s.boundary_mode == mode

def test_invalid_boundary_mode_raises():
    grid = StructuredGrid.create_rectangle(Lx=1.0, Ly=1.0, Nx=11, Ny=11, device='cpu')
    with pytest.raises(ValueError):
        FDMDiscretization(grid=grid, boundary_mode='nonsense')
```

**Verify**: existing V5.4 test suite still passes:
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.4
python -m pytest tests/ -v
```

**Exit criteria**: ✅ all 9 V5.4 phase tests still pass; new smoke tests pass; default behavior unchanged.

**Risk**: changing the class signature with `boundary_mode='node_mirror_existing'` as default is safe (default preserves legacy behavior). However, if any V5.4 test was inadvertently relying on a specific stencil under the legacy "node-mirror" assumption, that test will continue to pass — meaning the legacy is implicitly canonical.

### Step A.2 — Implement `face_mirror` and `zero_pad` modes in the assembly loop

**Read first**: Step A.0 notes; `Research/Q1_spatial_discretization/01_FDM_Stencils_and_Implementation.md` for stencil derivation reference.

**Why**: this is the experiment's core. Wrong assembly = invalid experiment.

**Implementation spec** — modify the assembly loop in `_build_laplacian` (~line 261-320). For the EAST edge (`i+1 >= nx`), branch on `self.boundary_mode`:

```python
# East neighbor handling — boundary cell at i = nx-1
if i + 1 < nx:
    # Interior east neighbor — full stencil entry
    if _is_active(i+1, j):
        D_face = harmonic_mean(D_xx[i, j], D_xx[i+1, j])
        w = D_face * cx
        A_rows.append(k); A_cols.append(idx(i+1, j)); A_vals.append(w)
        A_rows.append(k); A_cols.append(k);            A_vals.append(-w)
elif self.boundary_mode == 'node_mirror_existing':
    # LEGACY default — V_ghost[nx,j] = V[nx-2,j], so the East stencil entry's
    # contribution piles onto the West neighbor entry. Net: A[k, idx(i-1,j)]
    # gets += 2*w_east (the West cardinal entry adds w; this mirror adds w again).
    if _is_active(i-1, j):
        D_face = harmonic_mean(D_xx[i, j], D_xx[i-1, j])
        w = D_face * cx
        A_rows.append(k); A_cols.append(idx(i-1, j)); A_vals.append(w)
        A_rows.append(k); A_cols.append(k);            A_vals.append(-w)
elif self.boundary_mode == 'face_mirror':
    # NEW — V_ghost[nx,j] = V[i,j]. The East flux = D*(V_ghost - V[i,j])/dx² = 0.
    # In matrix form: NOTHING is added (no contribution to any entry). The
    # diagonal coefficient is REDUCED by w because one neighbor is missing.
    # Equivalent to: face-centered Neumann places the wall at i = nx - 0.5 and
    # cell (nx-1, j) has only 3 active neighbors (W, N, S) — its diagonal is
    # -3w instead of -4w. This is the standard "ghost = self" Neumann.
    pass  # nothing to add — diagonal stays at the running sum (one less -w contribution)
elif self.boundary_mode == 'zero_pad':
    # NEW — V_ghost[nx,j] = 0. The East flux = D*(0 - V[i,j])/dx² = -D*V[i,j]/dx².
    # In matrix form: the diagonal entry gets the -w contribution as if a real
    # neighbor existed at V=0. NO off-diagonal entry is added.
    A_rows.append(k); A_cols.append(k); A_vals.append(-w_east)
    # where w_east = D * cx (using D at the boundary cell, NOT harmonic mean
    # since the "ghost" has no D)
```

Apply symmetric logic to the WEST (`i-1 < 0`), NORTH (`j+1 >= ny`), and
SOUTH (`j-1 < 0`) branches.

For the 9-point diagonal corners (line 322-349), the existing code already
SKIPS them at rectangle boundary. Document this as a known limitation
(diagonals not handled in any boundary mode). Phase C uses the 9-point
stencil but the diagonal-skip behavior is identical across all three
boundary modes, so the comparison is valid.

**Test spec** — see Step A.4.

**Verify** — hand computation on a 4×4 grid:

For `V = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]`, indexed as
`V[i,j]` where `i` is x-direction (column index in fdm.py) and `j` is
y-direction. So V[2,3]=12, V[3,2]=15, V[3,3]=16. Use D=1, chi=Cm=1, dx=1.

Laplacian at corner (i=3, j=3) [bottom-right]:

- node_mirror_existing: East ghost mirrors to (i-1, j)=(2,3), V=12.
  North ghost mirrors to (i, j-1)=(3,2), V=15. West real (2,3)=12 and
  South real (3,2)=15. Stencil:
  L = (V_E_ghost + V_W_real + V_N_ghost + V_S_real − 4·V[3,3])
    = (12 + 12 + 15 + 15 − 64) = -10
  In matrix form: the East mirror entry adds w to the (i-1,j)=(2,3) slot,
  which the West cardinal already populated → after coalesce(), entry to
  (2,3) becomes 2w. Same for North/South → (3,2) entry is 2w. So the
  matrix-vector product: 2·12 + 2·15 − 4·16 = 24 + 30 − 64 = -10. ✓

- face_mirror: V_ghost_east = V[3,3] = 16, V_ghost_north = V[3,3] = 16.
  West real (2,3)=12, South real (3,2)=15. Stencil:
  L = (16 + 12 + 16 + 15 − 64) = 59 − 64 = -5
  In matrix form: face_mirror adds NOTHING to off-diagonal entries; the
  East stencil contribution reduces to (V_C − V_C)/dx² = 0. The diagonal
  entry stays at the running sum. With 2 missing neighbors (East, North),
  diagonal accumulator = -2w (only West and South contribute -w each).
  L = w·V[2,3] + w·V[3,2] − 2w·V[3,3] = 12 + 15 − 32 = -5. ✓

- zero_pad: V_ghost_east = 0, V_ghost_north = 0. West real (2,3)=12,
  South real (3,2)=15. Stencil:
  L = (0 + 12 + 0 + 15 − 64) = -37
  In matrix form: zero_pad adds -w to the diagonal for each off-grid
  neighbor (the ghost contributes -w·V_C with V_ghost = 0). Diagonal
  accumulator = -4w (West and South cardinals contribute -w each, plus
  East and North zero-pad add -w each). Off-diagonal: w·V[2,3] (West)
  and w·V[3,2] (South).
  L = w·12 + w·15 − 4w·16 = 27 − 64 = -37. ✓

Interior cell (i=1, j=1) under all modes (no boundary contribution):
  L = (V[0,1] + V[2,1] + V[1,0] + V[1,2] − 4·V[1,1])
    = (2 + 10 + 5 + 7 − 24) = 0. ✓

Hand-verified. The Step A.4 test code uses these exact expected values
(face_mirror corner = -5, zero_pad corner = -37); node_mirror_existing
hand-value = -10 (added test should also check this).

**Exit criteria**: ✅ hand-computation matches numerical for at least one
corner cell in each mode. Document the values in a code comment.

**Risk**:
- `harmonic_mean` of `D_xx[i, j]` with itself doesn't apply at the ghost
  for `zero_pad` — instead, just use `D = D_xx[i, j]`. Mitigation: the
  pseudocode above does this. Verify in implementation.
- The 9-point diagonal-skip is preserved (existing behavior), so the
  comparison is valid but the absolute Laplacian values may have a
  small (~O(D_xy)) systematic difference from a hypothetical diagonal-
  aware implementation. Document this caveat in the experiment write-up.
- **SPD properties of the three modes** (per audit-3 N6):
  - `node_mirror_existing` PRESERVES SPD (matrix is symmetric — both East mirror and West cardinal write to the same off-diagonal entry; symmetric).
  - `face_mirror` PRESERVES SPD (no off-diagonal additions; only the diagonal accumulator changes; symmetric).
  - `zero_pad` PRESERVES SPD (only the diagonal accumulates the missing-channel sink term, which is symmetric).
  - All three modes are compatible with V5.4's PCG / Chebyshev / DCT / FFT
    solvers (which assume SPD). Only `node_mirror_with_fold` (deferred to
    Phase F) breaks SPD by writing asymmetric off-diagonal entries.
  - Practical consequence: PCG convergence behavior may differ slightly
    across the three modes due to changed eigenvalue spectrum (boundary
    diagonal entries differ), but `pcg_max_iter=500` and `pcg_tol=1e-8`
    have ample headroom; convergence should not be a Phase C blocker.

### Step A.3 — Defer `node_mirror_with_fold` to Phase F (follow-up)

**Read first**: Step A.0 notes (V5.4 has no GMRES).

**Why**: `node_mirror_with_fold` makes the operator A non-symmetric
(audit-2 N1 + audit-1 H2). V5.4's available solvers (PCG, Chebyshev, DCT,
FFT — verified from `cardiac_sim/simulation/classical/solver/
diffusion_time_stepping/linear_solver/`) all assume SPD. Implementing
`node_mirror_with_fold` would require either (a) an explicit time stepper
that doesn't need a matrix solve, or (b) a new GMRES/BiCGStab solver. Both
are substantial side-projects.

**Implementation spec**:
- Mark `node_mirror_with_fold` as DEFERRED in this revision.
- Phase C tests three modes (face_mirror, node_mirror_existing, zero_pad)
  and compares against the existing storage-tank reflect_y empirical result
  (Δ@x=18 = -270 step camel toe at gradient rule, from prior IDEALOG).
- A future "Phase F" will add an explicit-Euler diffusion solver that doesn't
  need SPD, then retrofit `node_mirror_with_fold` against it.

**Test spec**: none.

**Verify**: documented decision in IDEALOG.

**Exit criteria**: ✅ written entry. Phase C scope reduced from 4 modes to 3.

**Risk**: storing the camel-toe reproduction question for Phase F means
the immediate experiment cannot fully verify H_BC's "node-mirror-with-fold
reproduces storage-tank reflect_y in continuum form" claim. Consequence:
the conclusion of this round will be "if H_BC holds, zero_pad reproduces
crescent; node-mirror-with-fold camel toe is testable in Phase F". This is
acceptable and explicitly noted in Done Criteria.

### Step A.4 — Tests

**Read first**: Step A.2 implementation; existing V5.4 test files for
conventions.

**Why**: lock in correctness before expensive Phase C runs.

**Implementation spec**: `Monodomain/Engine_V5.4/test_phase_boundary_modes.py`:

```python
"""Tests for boundary_mode parameter on FDMDiscretization.

Verifies:
  - constant field gives zero Laplacian for face_mirror, node_mirror_existing
  - constant field gives nonzero (sink) Laplacian for zero_pad at boundaries
  - uniform-y wave gives identical boundary/interior Laplacian for face_mirror
  - corner cells produce expected hand-computed values
  - dtype is preserved as float64
"""
import pytest
import torch
import numpy as np
from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization

DTYPE = torch.float64
DEVICE = 'cpu'


def _make_fdm(boundary_mode: str, Nx: int = 11, Ny: int = 11, dx: float = 1.0):
    grid = StructuredGrid.create_rectangle(
        Lx=(Nx - 1) * dx, Ly=(Ny - 1) * dx,
        Nx=Nx, Ny=Ny,
        device=DEVICE, dtype=DTYPE,
    )
    return FDMDiscretization(
        grid=grid, D=1.0, chi=1.0, Cm=1.0,  # chi=1, Cm=1, D=1 for simple math
        boundary_mode=boundary_mode,
    )


@pytest.mark.parametrize("mode", ['face_mirror', 'node_mirror_existing'])
def test_constant_field_laplacian_zero(mode):
    """Mirror Neumann (face or node): ∇²(constant) should be 0."""
    fdm = _make_fdm(mode)
    V = torch.full((fdm.n_dof,), 5.0, dtype=DTYPE, device=DEVICE)
    L_V = fdm.apply_diffusion(V)
    assert torch.allclose(L_V, torch.zeros_like(L_V), atol=1e-12), \
        f"{mode}: ∇²(const) = {L_V.abs().max().item()} (expected 0)"


def test_constant_field_zero_pad_drains_at_boundary():
    """Zero-pad: boundary cells should see V=0 ghosts, draining the constant."""
    fdm = _make_fdm('zero_pad', Nx=11, Ny=11)
    V = torch.full((fdm.n_dof,), 5.0, dtype=DTYPE, device=DEVICE)
    L_V = fdm.apply_diffusion(V)
    L_2d = L_V.reshape(11, 11)
    # Interior should have ∇²(const) = 0
    assert torch.allclose(L_2d[1:-1, 1:-1], torch.zeros_like(L_2d[1:-1, 1:-1]), atol=1e-12)
    # Boundary should be < 0 (drains to V=0)
    assert (L_2d[0, 1:-1] < -0.01).all(), "top edge should drain"
    assert (L_2d[-1, 1:-1] < -0.01).all(), "bottom edge should drain"
    assert (L_2d[1:-1, 0] < -0.01).all(), "left edge should drain"
    assert (L_2d[1:-1, -1] < -0.01).all(), "right edge should drain"


def test_uniform_y_wave_face_mirror_is_1d():
    """face_mirror: for V(x,y) = f(x), boundary row Laplacian == interior row Laplacian."""
    fdm = _make_fdm('face_mirror', Nx=11, Ny=11)
    Nx, Ny = 11, 11
    x_grid = torch.arange(Nx, dtype=DTYPE) * 1.0  # dx=1
    f_x = 100.0 * torch.exp(-((x_grid - 4.0) ** 2) / 2.0)
    V_2d = f_x[:, None].expand(Nx, Ny).clone()  # uniform in y
    V = V_2d.flatten()
    L_V = fdm.apply_diffusion(V).reshape(Nx, Ny)
    # Boundary row (y=0) should equal interior row (y=Ny//2)
    assert torch.allclose(L_V[:, 0], L_V[:, Ny // 2], atol=1e-10), \
        "face_mirror: boundary row Laplacian differs from interior row"


def test_uniform_y_wave_zero_pad_breaks_symmetry():
    """zero_pad: boundary row should DIFFER from interior row (extra sink)."""
    fdm = _make_fdm('zero_pad', Nx=11, Ny=11)
    Nx, Ny = 11, 11
    x_grid = torch.arange(Nx, dtype=DTYPE) * 1.0
    f_x = 100.0 * torch.exp(-((x_grid - 4.0) ** 2) / 2.0)
    V_2d = f_x[:, None].expand(Nx, Ny).clone()
    V = V_2d.flatten()
    L_V = fdm.apply_diffusion(V).reshape(Nx, Ny)
    diff = (L_V[:, 0] - L_V[:, Ny // 2]).abs().max().item()
    assert diff > 1e-3, f"zero_pad: boundary should differ from interior (got max diff {diff})"


def test_corner_cell_hand_check_face_mirror():
    """4×4 grid, V=arange(16).reshape(4,4)+1, dx=1, D=chi=Cm=1, face_mirror.
    Bottom-right corner (3,3) = 16. Real neighbors: (2,3)=12, (3,2)=15.
    face_mirror: V_ghost_east = 16, V_ghost_south = 16.
    L = (16 + 12 + 16 + 15 - 4·16) = -5.   <-- two ghosts cancel two real neighbors
    Equivalent: L = (V[2,3] + V[3,2] - 2·V[3,3]) = (12 + 15 - 32) = -5."""
    fdm = _make_fdm('face_mirror', Nx=4, Ny=4, dx=1.0)
    V = torch.arange(1, 17, dtype=DTYPE).reshape(4, 4)  # V[i,j] = i*4 + j + 1
    L_V = fdm.apply_diffusion(V.flatten()).reshape(4, 4)
    # Note: V[3, 3] = 16, V[2, 3] = 12, V[3, 2] = 15
    # Expected face_mirror Laplacian at (3,3) with chi*Cm=1: -5
    expected = -5.0
    actual = L_V[3, 3].item()
    assert abs(actual - expected) < 1e-10, \
        f"face_mirror corner: expected {expected}, got {actual}"


def test_corner_cell_hand_check_zero_pad():
    """4×4 grid, V=arange(16).reshape(4,4)+1, dx=1, D=chi=Cm=1, zero_pad.
    At (3,3): V_ghost_east = 0, V_ghost_north = 0.
    L = (0 + V[2,3] + 0 + V[3,2] - 4·V[3,3]) = (12 + 15 - 64) = -37."""
    fdm = _make_fdm('zero_pad', Nx=4, Ny=4, dx=1.0)
    V = torch.arange(1, 17, dtype=DTYPE).reshape(4, 4)
    L_V = fdm.apply_diffusion(V.flatten()).reshape(4, 4)
    expected = -37.0
    actual = L_V[3, 3].item()
    assert abs(actual - expected) < 1e-10, \
        f"zero_pad corner: expected {expected}, got {actual}"


def test_corner_cell_hand_check_node_mirror_existing():
    """4×4 grid, V=arange(16).reshape(4,4)+1, dx=1, D=chi=Cm=1, node_mirror_existing.
    At (3,3): East ghost mirror = V[2,3]=12, North ghost mirror = V[3,2]=15.
    West real = V[2,3]=12, South real = V[3,2]=15.
    Matrix accumulates 2w on (2,3) entry (East mirror + West cardinal) and
    2w on (3,2) entry (North mirror + South cardinal); diagonal -4w.
    L = 2·12 + 2·15 - 4·16 = 24 + 30 - 64 = -10."""
    fdm = _make_fdm('node_mirror_existing', Nx=4, Ny=4, dx=1.0)
    V = torch.arange(1, 17, dtype=DTYPE).reshape(4, 4)
    L_V = fdm.apply_diffusion(V.flatten()).reshape(4, 4)
    expected = -10.0
    actual = L_V[3, 3].item()
    assert abs(actual - expected) < 1e-10, \
        f"node_mirror_existing corner: expected {expected}, got {actual}"


def test_dtype_preserved():
    """All boundary modes must preserve float64."""
    for mode in FDMDiscretization.BOUNDARY_MODES:
        fdm = _make_fdm(mode)
        V = torch.zeros(fdm.n_dof, dtype=DTYPE, device=DEVICE)
        L_V = fdm.apply_diffusion(V)
        assert L_V.dtype == DTYPE, f"{mode}: dtype was {L_V.dtype}"
```

**Test spec** — pytest passes.

**Verify**:
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.4
python -m pytest test_phase_boundary_modes.py -v
```

**Exit criteria**: ✅ all 7 tests pass. Commit.

**Risk**: hand-computed values assume D_xx isotropic and dx=1 for clean math.
If the existing implementation uses a more complex D_field, the simple
`D=1` constructor should bypass that. Verify by reading
`FDMDiscretization.__init__` to confirm `D_field=None` falls back to
isotropic `D`.

---

## Phase B — Add `vacuum` BC variant to LBM

**Goal**: add a new BC variant in `LBM/Engine_V1` that sets populations
streaming into the wall to zero (`f_in = 0`). Name it `vacuum` to avoid
collision with the existing `absorbing.py` (which uses
`f_in = w · V`, equilibrium-based — different semantics).

**Files to modify**:
- `LBM/Engine_V1/src/boundary/vacuum.py` (NEW)
- `LBM/Engine_V1/src/boundary/__init__.py` (export)
- `LBM/Engine_V1/tests/test_vacuum_bc.py` (NEW)

### Step B.0 — Read existing BC implementations [READ-ONLY]

**Read first**: API reference card section 7. Verify:
- `precompute_bounce_masks(domain_mask: Tensor, lattice) -> dict[int, Tensor]` (signature)
- `make_rect_bounce_masks(Nx, Ny, lattice) -> dict` (test helper in `tests/test_phase4.py:29`)

**Why**: don't reinvent the helper or use the wrong signature.

**Implementation spec** (read-only):
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/LBM/Engine_V1
sed -n '1,50p' src/boundary/masks.py
sed -n '20,55p' tests/test_phase4.py  # find make_rect_bounce_masks
```

**Test spec**: none.

**Verify**: written notes confirming the helper signatures.

**Exit criteria**: ready to write Step B.1 with correct conventions.

**Risk**: none.

### Step B.1 — Implement `vacuum` BC

**Read first**: Step B.0 notes; `LBM/Engine_V1/src/boundary/absorbing.py` as template (closest semantically).

**Why**: this is the LBM analog of monodomain `zero_pad`.

**Implementation spec**: `LBM/Engine_V1/src/boundary/vacuum.py`:

```python
"""Vacuum boundary condition — incoming distributions set to zero.

Formula:
    f[opp[a]](x) = 0   at cells where direction `a` points into the wall

Equivalent to V_outside = 0 (an "amputated" boundary, as opposed to the
standard bounce-back which mirrors interior populations and thereby cancels
boundary asymmetry; or the existing absorbing.py which uses w·V incoming).

This is the LBM analog of monodomain zero_pad: the boundary cell loses
mass to a virtual V=0 outside.

WARNING: Strongly non-conservative. Total mass decreases over time. Intended
only for the BC discretization comparison study (see Research/Active/
boundary_conduction_speedup/PLAN.md Phase B).

Layer 2: pure functions, torch.compile compatible.
"""

import torch
from torch import Tensor


def apply_vacuum_d2q5(f: Tensor, bounce_masks: dict[int, Tensor]) -> Tensor:
    """Apply vacuum BC for D2Q5 (incoming = 0).

    Args:
        f: (5, Nx, Ny) post-streaming distributions
        bounce_masks: dict from precompute_bounce_masks; bounce_masks[a] is
                      True at cells where direction `a` points OUT of the domain.

    Returns:
        f with incoming distributions zeroed at boundary cells.
    """
    # D2Q5 opposite indices: (0, 2, 1, 4, 3)
    # At cells where direction 1 (E) points out, the opposite direction 2 (W)
    # is the INCOMING direction — set to zero.
    f[2] = torch.where(bounce_masks[1], torch.zeros_like(f[2]), f[2])
    f[1] = torch.where(bounce_masks[2], torch.zeros_like(f[1]), f[1])
    f[4] = torch.where(bounce_masks[3], torch.zeros_like(f[4]), f[4])
    f[3] = torch.where(bounce_masks[4], torch.zeros_like(f[3]), f[3])
    return f


def apply_vacuum_d2q9(f: Tensor, bounce_masks: dict[int, Tensor]) -> Tensor:
    """Apply vacuum BC for D2Q9 (incoming = 0)."""
    # D2Q9 opposite indices: (0, 2, 1, 4, 3, 7, 8, 5, 6)
    # Cardinals
    f[2] = torch.where(bounce_masks[1], torch.zeros_like(f[2]), f[2])
    f[1] = torch.where(bounce_masks[2], torch.zeros_like(f[1]), f[1])
    f[4] = torch.where(bounce_masks[3], torch.zeros_like(f[4]), f[4])
    f[3] = torch.where(bounce_masks[4], torch.zeros_like(f[3]), f[3])
    # Diagonals
    f[7] = torch.where(bounce_masks[5], torch.zeros_like(f[7]), f[7])
    f[8] = torch.where(bounce_masks[6], torch.zeros_like(f[8]), f[8])
    f[5] = torch.where(bounce_masks[7], torch.zeros_like(f[5]), f[5])
    f[6] = torch.where(bounce_masks[8], torch.zeros_like(f[6]), f[6])
    return f
```

Add to `LBM/Engine_V1/src/boundary/__init__.py`:
```python
from .vacuum import apply_vacuum_d2q5, apply_vacuum_d2q9
```

**Test spec**: see Step B.2.

**Verify**:
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/LBM/Engine_V1
python -c "from src.boundary import apply_vacuum_d2q5; print('import ok')"
```

**Exit criteria**: ✅ import succeeds; tests pass (Step B.2).

**Risk**: dtype handled correctly via `torch.zeros_like(f[2])` (preserves dtype).

### Step B.2 — Tests for `vacuum` BC

**Read first**: `LBM/Engine_V1/tests/test_phase4.py:29-47` for `make_rect_bounce_masks` helper.

**Why**: lock in correctness; use the right helper for full-grid rectangular domains.

**Implementation spec**: `LBM/Engine_V1/tests/test_vacuum_bc.py`:

```python
"""Tests for vacuum BC: incoming populations = 0 at wall cells."""
import pytest
import torch
from src.boundary.vacuum import apply_vacuum_d2q5, apply_vacuum_d2q9
from src.lattice import D2Q5, D2Q9

DTYPE = torch.float64


def make_rect_bounce_masks(Nx, Ny, lattice):
    """Create bounce masks for rectangular domain filling full grid.
    Copied from tests/test_phase4.py:29-47 for self-containment."""
    bounce_masks = {}
    for a in range(1, lattice.Q):
        m = torch.zeros(Nx, Ny, dtype=torch.bool)
        ex, ey = lattice.e[a]
        if ex == 1:   m[-1, :] = True   # east wall
        if ex == -1:  m[0, :] = True    # west wall
        if ey == 1:   m[:, -1] = True   # north wall
        if ey == -1:  m[:, 0] = True    # south wall
        bounce_masks[a] = m
    return bounce_masks


def test_vacuum_d2q5_zeros_incoming_at_walls():
    """At each wall cell, the incoming-direction population should be zero."""
    Nx, Ny = 8, 6
    f = torch.ones((5, Nx, Ny), dtype=DTYPE)
    masks = make_rect_bounce_masks(Nx, Ny, D2Q5())
    f_out = apply_vacuum_d2q5(f.clone(), masks)
    # At east wall (x=Nx-1), direction 1 (E) points out; incoming is 2 (W) → 0.
    assert (f_out[2, -1, :] == 0).all()
    # At west wall (x=0), direction 2 (W) points out; incoming is 1 (E) → 0.
    assert (f_out[1, 0, :] == 0).all()
    # At north wall (y=Ny-1, since ey=1 → m[:, -1]), direction 3 (N) out; incoming 4 (S) → 0.
    assert (f_out[4, :, -1] == 0).all()
    # At south wall (y=0), direction 4 (S) out; incoming 3 (N) → 0.
    assert (f_out[3, :, 0] == 0).all()
    # Center direction (0) untouched
    assert (f_out[0] == 1).all()


def test_vacuum_d2q5_loses_mass_at_boundary():
    """Vacuum is non-conservative. Wall cells should lose mass; interior unchanged."""
    Nx, Ny = 8, 6
    f = torch.ones((5, Nx, Ny), dtype=DTYPE)
    rho_before = f.sum(dim=0)
    masks = make_rect_bounce_masks(Nx, Ny, D2Q5())
    f_out = apply_vacuum_d2q5(f.clone(), masks)
    rho_after = f_out.sum(dim=0)
    # Wall cells: density decreases
    assert (rho_after[0, :] < rho_before[0, :]).all()
    assert (rho_after[-1, :] < rho_before[-1, :]).all()
    assert (rho_after[:, 0] < rho_before[:, 0]).all()
    assert (rho_after[:, -1] < rho_before[:, -1]).all()
    # Interior: unchanged
    assert torch.allclose(rho_after[1:-1, 1:-1], rho_before[1:-1, 1:-1])


def test_vacuum_d2q9_diagonals_at_corners():
    """At corner cells, all 4 diagonal incoming should be zeroed."""
    Nx, Ny = 8, 6
    f = torch.ones((9, Nx, Ny), dtype=DTYPE)
    masks = make_rect_bounce_masks(Nx, Ny, D2Q9())
    f_out = apply_vacuum_d2q9(f.clone(), masks)
    # At east wall, direction 1 (E)=(1,0) is OUT; opp 2 (W) → 0.
    assert (f_out[2, -1, :] == 0).all()
    # Diagonal NE (5)=(1,1) out at east+north; opp SW (7) → 0.
    assert (f_out[7, -1, -1] == 0).all() or f_out[7, -1, -1] == 0
    # Center untouched
    assert (f_out[0] == 1).all()


def test_vacuum_dtype_preserved():
    f = torch.ones((5, 8, 6), dtype=DTYPE)
    masks = make_rect_bounce_masks(8, 6, D2Q5())
    f_out = apply_vacuum_d2q5(f, masks)
    assert f_out.dtype == DTYPE
```

**Test spec**: pytest passes.

**Verify**:
```bash
python -m pytest tests/test_vacuum_bc.py -v
```

**Exit criteria**: ✅ all green. Commit.

**Risk**: D2Q9 corner test diagonals may not include all 4 cells in `make_rect_bounce_masks`. Verify by inspecting the helper output before relying on the assertion.

---

## Phase C — Mesh-refinement scan in V5.4 monodomain (3 modes)

**Goal**: run line-source propagation for the three monodomain variants
(`face_mirror`, `node_mirror_existing`, `zero_pad`) at three mesh resolutions
(dx ∈ {0.05, 0.025, 0.0125}). Measure per-column LAT deviation `Δ(x_mid)`.
Fit empirical h-scaling exponent. Verify against predictions.

**Files**:
- `Monodomain/Engine_V5.4/experiments/bc_discretization_test/EXPERIMENT.md` (NEW)
- `Monodomain/Engine_V5.4/experiments/bc_discretization_test/run.py` (NEW)
- `Monodomain/Engine_V5.4/experiments/bc_discretization_test/outputs/` (created at runtime)
- `simulation/configs.py` (ADD GRADIENT_BIDIR_REFLECT for Step C.4)

### Step C.0 — Set up experiment directory and EXPERIMENT.md

**Read first**: `Monodomain/Engine_V5.4/experiments/` for existing experiment conventions.

**Why**: standardize documentation and traceability.

**Implementation spec**:
```bash
mkdir -p /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.4/experiments/bc_discretization_test
cd $_
```

Create `EXPERIMENT.md`:

```markdown
# Experiment: BC Discretization H-Scaling

## Research question
Test whether boundary artifacts under non-standard Neumann discretizations
(zero_pad, node_mirror_existing) are numerical artifacts that vanish in the
continuum limit, vs. real continuum effects (Kleber under bath-coupled BC).

## Backlinks
- ../../../../Research/Active/boundary_conduction_speedup/{KNOWLEDGE,IDEALOG,PLAN}.md
- ../../../../MASTER_KNOWLEDGE_INDEX.md

## Setup (standardized — must match across all variants)
- Geometry: 2D rectangle, 4 cm × 2 cm
- Initial condition: V = V_rest = -85.23 mV (TTP06 EPI)
- Stimulus: line source at x ∈ [0, 0.1] cm, depolarizing current
  - region: lambda x, y: x < 0.1
  - amplitude: -52.0 µA/µF
  - duration: 5.0 ms
  - start_time: 0.0 ms
- Domain mesh: dx ∈ {0.05, 0.025, 0.0125} cm
- Time: 200 ms simulated
- chi = 1400 cm⁻¹ (V5.4 default; standardized)
- Cm = 1.0 µF/cm²
- D = 0.001 cm²/ms
- Time integration: Crank-Nicolson (V5.4 default for FDM diffusion)
- Ionic step: dt = 0.02 ms
- Splitting: Strang
- Linear solver: PCG with pcg_tol=1e-8

## Variants (3 modes, deferred 4th to Phase F)
- monodomain `face_mirror`             — true Neumann (predicted: Δ ≈ 0 at all h)
- monodomain `node_mirror_existing`    — V5.4 current default (predicted: Δ ≠ 0; sign TBD)
- monodomain `zero_pad`                — V_outside=0 (predicted: crescent; scaling empirical)

## Measurements
- Per-column LAT deviation Δ(x) = ½(LAT[0, x] + LAT[Ny-1, x]) − LAT[Ny/2, x]
  Convention: positive Δ = boundary fires later than middle (crescent).
  Negative Δ = boundary fires first (camel toe).
- |Δ|_max across x for each (variant, dx)
- Log-log fit of |Δ|_max vs h to extract scaling exponent p (where |Δ| ~ h^p)

## Pass/fail criteria
- face_mirror: |Δ|_max / max(|other variants' |Δ|_max|) < 0.02 at all h
- zero_pad and node_mirror_existing: scaling exponent p ∈ [0.5, 2.5]
  (i.e., consistent with O(h^p) for some reasonable p; FAIL if Δ stays
  constant or grows as h decreases, which would indicate p ≤ 0 and falsify
  the discretization-artifact hypothesis)
- Bidomain Kleber comparison (Step C.2): scaling |p| < 0.2 (constant in h)
```

**Test spec**: file created, EXPERIMENT.md readable, backlinks resolve.

**Verify**:
```bash
ls ../../../../MASTER_KNOWLEDGE_INDEX.md
ls ../../../../Research/Active/boundary_conduction_speedup/KNOWLEDGE.md
```

**Exit criteria**: file exists; backlinks point to actual files.

**Risk**: relative paths depend on the directory depth. Verify with `ls`.

### Step C.1 — Implement `run.py` using verbatim-correct V5.4 API

**Read first**:
- API reference card sections 1, 2, 3, 4, 6 (verified APIs)
- An existing V5.4 experiment (if any in `Monodomain/Engine_V5.4/experiments/`)

**Why**: use real APIs that the prior audit's CRITICAL N1 finding flagged
were invented. Every API call below is grounded.

**Implementation spec**: `run.py`:

```python
"""BC discretization h-scaling experiment for Monodomain V5.4.

Sweeps {face_mirror, node_mirror_existing, zero_pad} × {dx=0.05, 0.025, 0.0125}.
Outputs per-column LAT deviation, h-scaling plot, and summary table.

All API calls grounded against verified engine source (API reference card 2026-04-29).
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Verified imports
from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation
from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol

DTYPE = torch.float64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def run_one(boundary_mode: str, dx: float, T_max: float = 200.0):
    """Run a single (boundary_mode, dx) configuration. Returns (lat, Nx, Ny)."""
    Lx, Ly = 4.0, 2.0  # cm
    Nx = int(round(Lx / dx)) + 1
    Ny = int(round(Ly / dx)) + 1

    # Standardized parameters
    grid = StructuredGrid.create_rectangle(
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        device=DEVICE, dtype=DTYPE,
    )
    spatial = FDMDiscretization(
        grid=grid,
        D=0.001,        # cm²/ms (standardized)
        chi=1400.0,     # cm⁻¹ (V5.4 default; standardized)
        Cm=1.0,         # µF/cm²
        boundary_mode=boundary_mode,  # NEW kwarg from Phase A
    )

    # Line-source stimulus at x ∈ [0, 0.1] cm
    stim = StimulusProtocol()
    stim.add_stimulus(
        region=lambda x, y: x < 0.1,    # callable: returns boolean mask over grid
        start_time=0.0,                 # ms
        duration=5.0,                   # ms
        amplitude=-52.0,                # µA/µF (depolarizing, standard sign)
    )

    sim = MonodomainSimulation(
        spatial=spatial,
        ionic_model='ttp06',            # string accepted by _build_ionic_model
        stimulus=stim,
        dt=0.02,                        # ms (NOT time_step=)
        splitting='strang',
        ionic_solver='rush_larsen',
        diffusion_solver='crank_nicolson',
        linear_solver='pcg',            # NOT 'gmres' — only pcg/chebyshev/dct/fft/none exist
        cell_type='EPI',
        pcg_tol=1e-8,                   # NOT linear_rtol=
        pcg_max_iter=500,
    )

    # Run simulation, collecting V history
    times, V_history = sim.run_to_array(t_end=T_max, save_every=1.0)
    # V_history shape: (n_saves, n_dof) = (~200, Nx*Ny)

    # Compute LAT (Local Activation Time) at each node
    lat = sim.compute_activation_time(V_history, times, threshold=-20.0)
    # lat shape: (n_dof,) with NaN where unfired
    lat_2d = lat.reshape(Nx, Ny)

    return lat_2d, Nx, Ny


def per_column_delta(lat_2d: np.ndarray, x_idx: int) -> float:
    """Δ(x) = ½(LAT[0, x] + LAT[-1, x]) - LAT[middle, x]; NaN if any unfired."""
    Ny = lat_2d.shape[1]
    col = lat_2d[x_idx, :].astype(np.float64)
    if np.isnan(col).any():
        return float('nan')
    top, mid, bot = col[0], col[Ny // 2], col[-1]
    return float(0.5 * (top + bot) - mid)


def main():
    modes = ['face_mirror', 'node_mirror_existing', 'zero_pad']
    dxs = [0.05, 0.025, 0.0125]
    results = {}

    for mode in modes:
        results[mode] = {}
        for dx in dxs:
            print(f"running {mode} at dx={dx} cm...", flush=True)
            lat_2d, Nx, Ny = run_one(mode, dx)
            x_mid = Nx // 2
            delta = per_column_delta(lat_2d, x_mid)
            np.savez(
                OUT_DIR / f"lat_{mode}_dx{dx}.npz",
                lat=lat_2d, dx=dx, Nx=Nx, Ny=Ny, x_mid=x_mid,
            )
            results[mode][f"{dx}"] = delta
            print(f"  Δ@x_mid = {delta:.4f} ms", flush=True)

    # Save summary
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # H-scaling plot (log-log)
    fig, ax = plt.subplots(figsize=(8, 6))
    fit_results = {}
    for mode in modes:
        ds_keys = sorted(results[mode].keys(), key=float)
        ds = [float(k) for k in ds_keys]
        deltas = []
        ds_with_data = []
        for d_key, d in zip(ds_keys, ds):
            v = results[mode][d_key]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                deltas.append(abs(v))
                ds_with_data.append(d)
        if len(deltas) < 2: continue
        ax.loglog(ds_with_data, deltas, 'o-', label=mode, lw=1.8, ms=8)
        slope, intercept = np.polyfit(np.log(ds_with_data), np.log(deltas), 1)
        fit_results[mode] = float(slope)
        print(f"  {mode}: log-log slope p = {slope:.3f}", flush=True)

    ax.set_xlabel("dx (cm)", fontsize=12)
    ax.set_ylabel("|Δ(x_mid)|  (ms)", fontsize=12)
    ax.set_title("Boundary-LAT-deviation magnitude vs mesh size", fontsize=12)
    ax.grid(alpha=0.4, which='both')
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "h_scaling.png", dpi=200)
    print(f"saved {OUT_DIR / 'h_scaling.png'}", flush=True)

    with open(OUT_DIR / "h_scaling_fits.json", "w") as f:
        json.dump(fit_results, f, indent=2)

    # Per-column delta plot at finest h
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for mode in modes:
        d = np.load(OUT_DIR / f"lat_{mode}_dx{dxs[-1]}.npz")
        lat_2d = d["lat"]
        Nx_, Ny_ = int(d["Nx"]), int(d["Ny"])
        deltas_x = [per_column_delta(lat_2d, x) for x in range(Nx_)]
        ax2.plot(deltas_x, label=mode, lw=1.6)
    ax2.axhline(0, color='gray', lw=0.6)
    ax2.set_xlabel("x (column)", fontsize=12)
    ax2.set_ylabel("Δ(x)  (ms)   negative = camel toe / boundary leads", fontsize=11)
    ax2.set_title(f"Per-column LAT deviation at dx={dxs[-1]} cm", fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=11)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "per_column_delta.png", dpi=200)
    print(f"saved {OUT_DIR / 'per_column_delta.png'}", flush=True)

    # Pass/fail evaluation
    print("\n=== Pass/fail evaluation ===", flush=True)
    SIGNIFICANCE_FLOOR_MS = 0.5  # |Δ| below this is considered "no boundary effect"

    def _max_abs(d):
        vs = [abs(v) for v in d.values()
              if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return max(vs) if vs else 0.0

    fm_max = _max_abs(results.get('face_mirror', {}))
    others_max = max(
        _max_abs(results[m]) for m in modes if m != 'face_mirror'
    ) if any(m != 'face_mirror' for m in modes) else 0.0

    print(f"face_mirror max |Δ| = {fm_max:.5f} ms", flush=True)
    print(f"others max |Δ|     = {others_max:.5f} ms", flush=True)

    # Three-way verdict (handles divide-by-zero edge case)
    if max(fm_max, others_max) < SIGNIFICANCE_FLOOR_MS:
        print(
            f"VERDICT: no boundary effect detected anywhere "
            f"(all |Δ| < {SIGNIFICANCE_FLOOR_MS} ms).\n"
            "         Either the test setup is insensitive at this mesh / time, "
            "or all three boundary modes happen to give near-zero asymmetry.\n"
            "         This is WEAKLY consistent with H_BC (face-centered cancels) "
            "but doesn't discriminate among modes. Recommend: extend domain or run "
            "longer to amplify the signal.",
            flush=True,
        )
    elif others_max < SIGNIFICANCE_FLOOR_MS:
        print(
            "VERDICT: face_mirror max |Δ| dominates other variants. "
            "Unexpected — face_mirror was supposed to be near-zero. H_BC may be "
            "FALSIFIED as currently posed.",
            flush=True,
        )
    else:
        ratio = fm_max / others_max
        print(
            f"ratio (face_mirror / others) = {ratio:.4f}  "
            f"({'PASS' if ratio < 0.02 else 'FAIL'} — face-centered hypothesis)",
            flush=True,
        )

    for mode in ('zero_pad', 'node_mirror_existing'):
        p = fit_results.get(mode)
        if p is not None:
            ok = 0.5 < p < 2.5
            print(f"{mode}: scaling p = {p:.3f}  ({'PASS' if ok else 'FAIL'})", flush=True)


if __name__ == "__main__":
    main()
```

**Test spec**:
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.4/experiments/bc_discretization_test
python run.py 2>&1 | tee run.log
```

**Verify**: `outputs/summary.json`, `outputs/h_scaling.png`,
`outputs/per_column_delta.png`, `outputs/h_scaling_fits.json` exist.
Numerical pass/fail criteria evaluated and printed.

**Exit criteria**: ✅ all four numerical checks pass:
- face_mirror max |Δ| / others max |Δ| < 0.02
- zero_pad scaling p ∈ [0.5, 2.5]
- node_mirror_existing scaling p ∈ [0.5, 2.5]
If pass: H_BC confirmed for monodomain. If any fail: log falsification in
IDEALOG and STOP — do not proceed to Phase D until reconciled.

**Risk**:
- Computational cost at dx=0.0125: Crank-Nicolson is unconditionally stable
  but each PCG solve scales. Estimate ~10-30 minutes per run on the GPU,
  9 runs ≈ 1.5-4.5 hours total. Run in background.
- TTP06 may have a long upstroke (~5 ms). LAT threshold of -20 mV captures
  mid-upstroke; verify by looking at V_history at one cell to confirm
  threshold crossing happens cleanly.
- Stimulus amplitude -52 µA/µF for 5 ms is the V5.4 default — should
  reliably initiate a wave. If it fails to fire, increase duration or
  amplitude (more negative).

### Step C.2 — Bidomain Kleber h-scaling (verify continuum effect)

**Read first**: `Bidomain/Engine_V1/PROGRESS.md`; `Bidomain/Engine_V1/experiments/triangle_merger.py` (the existing dx=0.05 result).

**Why**: confirm that bath-coupled bidomain Kleber speedup is constant in h.

**Implementation spec**: extend the existing Phase 6c boundary CV test.
Run at three mesh resolutions to establish h-scaling. Save to:
`Bidomain/Engine_V1/experiments/kleber_h_scaling/outputs/summary.json`
with structure:
```json
{
  "bath_coupled": {"0.05": cv_ratio_05, "0.025": cv_ratio_025, "0.0125": cv_ratio_0125}
}
```

The existing dx=0.025 test (per `Bidomain/Engine_V1/PROGRESS.md`) gives
CV ratio = 1.0714. Run dx=0.05 (faster) and dx=0.0125 (slower) to bracket.

**Test spec**: outputs/summary.json contains three entries.

**Verify**: log-log fit |CV_ratio - 1.0| vs dx gives |slope| < 0.2.

**Exit criteria**: ✅ scaling exponent close to zero (continuum-limit convergence).

**Risk**: bidomain at dx=0.0125 may take >2 hours per run. Schedule with
`run_in_background: true`. If intractable, defer the dx=0.0125 point and
report 2-point scaling instead with a caveat.

### Step C.3 — Diagnostic: storage-tank reflect_y vs monodomain node_mirror_existing

**Read first**: `simulation/tanks_vec.py` for storage-tank `reflect_y` semantics; Phase A Step A.0 notes for V5.4's `node_mirror_existing` semantics.

**Why**: prior audit-1 finding H7 — node_mirror_existing (without fold) may
not reproduce storage-tank reflect_y dynamics. Test directly. (Note: V5.4
node_mirror_existing has NO fold; storage-tank reflect_y HAS fold. So they
are NOT semantically identical, but their stencils on uniform-y inputs are
similar; this test verifies how similar.)

**Implementation spec**: a one-off diagnostic script
`Monodomain/Engine_V5.4/experiments/bc_discretization_test/diagnostic_proxy.py`:

```python
"""Diagnostic: compare storage-tank reflect_y per-column LAT shape against
monodomain node_mirror_existing on a matched grid (Nx=80, Ny=50). If the
shapes are qualitatively similar (same sign of Δ at all sampled x), the
proxy is reasonable. If different, the cascade requires the fold and Phase F
(node_mirror_with_fold) is the only path to reproducing storage-tank reflect_y.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

# Storage-tank
sys.path.insert(0, str(Path("/home/norepinephrine/Documents/Heart-Conduction/simulation")))
import tanks_vec, configs

# V5.4 monodomain
from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation
from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol

OUT_DIR = Path(__file__).parent / "outputs_diagnostic"
OUT_DIR.mkdir(exist_ok=True)

# Storage-tank reflect_y (gradient rule, Nx=80, Ny=50, 4000 steps)
inlet, outlet = configs.resolve_geometry(configs.GRADIENT['geometry'])
out = tanks_vec.run(
    Nx=80, Ny=50, mode='gradient', steps=4000,
    inlet_cells=inlet, outlet_cells=outlet,
    threshold=45, max_volume=100, max_pump=10, gradient_k=0.08,
    directionality='one_way', boundary='reflect_y',
    damping_cap=True,   # MATCH the GRADIENT config default (audit-3 H2);
                        # IDEALOG.md:148 reference Δ@x=18=-270 used this default
    record_history=False, snap_every=100,
)
iso_st = out['iso']
delta_st = []
for x in range(80):
    col = iso_st[:, x].astype(float)
    col[col < 0] = np.nan
    if np.isnan(col).any(): delta_st.append(np.nan); continue
    top, mid, bot = col[0], col[25], col[-1]
    delta_st.append(0.5 * (top + bot) - mid)

# V5.4 node_mirror_existing (Nx=80, Ny=50 → match storage-tank dimensions)
# dx chosen to match (4 cm domain / 80 cells = 0.05 cm)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float64
grid = StructuredGrid.create_rectangle(Lx=4.0, Ly=2.5, Nx=80, Ny=50, device=DEVICE, dtype=DTYPE)
spatial = FDMDiscretization(grid=grid, D=0.001, chi=1400.0, Cm=1.0, boundary_mode='node_mirror_existing')
stim = StimulusProtocol()
stim.add_stimulus(region=lambda x, y: x < 0.1, start_time=0.0, duration=5.0, amplitude=-52.0)
sim = MonodomainSimulation(
    spatial=spatial, ionic_model='ttp06', stimulus=stim, dt=0.02,
    splitting='strang', ionic_solver='rush_larsen',
    diffusion_solver='crank_nicolson', linear_solver='pcg',
    cell_type='EPI', pcg_tol=1e-8,
)
times, V_history = sim.run_to_array(t_end=200.0, save_every=1.0)
lat = sim.compute_activation_time(V_history, times, threshold=-20.0)
lat_2d = lat.reshape(80, 50)
delta_md = []
for x in range(80):
    col = lat_2d[x, :].astype(float)
    if np.isnan(col).any(): delta_md.append(np.nan); continue
    top, mid, bot = col[0], col[25], col[-1]
    delta_md.append(0.5 * (top + bot) - mid)

# Plot side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(delta_st, lw=1.6); ax1.axhline(0, color='gray', lw=0.5); ax1.grid(alpha=0.3)
ax1.set_title("storage-tank reflect_y (gradient rule)"); ax1.set_xlabel("x (column)"); ax1.set_ylabel("Δ (steps)")
ax2.plot(delta_md, lw=1.6); ax2.axhline(0, color='gray', lw=0.5); ax2.grid(alpha=0.3)
ax2.set_title("monodomain V5.4 node_mirror_existing"); ax2.set_xlabel("x (column)"); ax2.set_ylabel("Δ (ms)")
fig.tight_layout()
fig.savefig(OUT_DIR / "storage_tank_vs_monodomain.png", dpi=200)
print(f"saved {OUT_DIR / 'storage_tank_vs_monodomain.png'}")

# Compare signs
print(f"storage-tank Δ@x=18: {delta_st[18] if len(delta_st)>18 else None}")
print(f"monodomain   Δ@x=18: {delta_md[18] if len(delta_md)>18 else None}")
sign_match = (np.sign(delta_st[18]) == np.sign(delta_md[18])
              if (len(delta_st)>18 and len(delta_md)>18 and not np.isnan(delta_st[18]) and not np.isnan(delta_md[18]))
              else False)
print(f"signs match: {sign_match}")
```

**Test spec**: visual comparison of two plots.

**Verify**: signs of Δ@x=18 match (both negative or both positive) in the
storage-tank reflect_y vs monodomain node_mirror_existing comparison.

**Exit criteria**: ✅ written conclusion in IDEALOG: "node_mirror_existing
is/is not a valid proxy for storage-tank reflect_y." If invalid, add to
the README's `Future Work` section and prioritize Phase F.

**Risk**: medium — the magnitudes differ inherently (storage-tank is steps,
monodomain is ms). Compare SIGNS only, not magnitudes.

### Step C.4 — Discriminating test: bidirectional × reflect_y × gradient

**Read first**: `KNOWLEDGE.md` "Boundary-operator dominance" subsection;
`simulation/configs.py` for harness conventions.

**Why**: prior audit H7. Existing storage-tank evidence: `gradient + reflect_y`
gave Δ@x=18 = -270 (camel toe); `constant + bidirectional + zero_pad` gave
+25.5 (crescent). The new test combines BOTH: `gradient + bidirectional + reflect_y`.

**Implementation spec**:
- Add to `simulation/configs.py`:
```python
GRADIENT_BIDIR_REFLECT = make({
    "name": "gradient_bidir_reflect",
    "description": "Discriminating test — gradient rule, bidirectional pipes, reflect_y BC",
    "tags": ["line", "gradient", "bidirectional", "reflect_y"],
    "rule": {"type": "gradient"},
    "pipes": {"directionality": "bidirectional"},
    "boundary": {"type": "reflect_y"},
})
```
And add `GRADIENT_BIDIR_REFLECT` to the REGISTRY tuple.

- Run via:
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/simulation
python experiment.py gradient_bidir_reflect
```

**Test spec**: per-column Δ@x=18 logged from the harness output.

**Verify**: classify the result based on the four-cell decision table:
- If Δ@x=18 < -10 (clear camel): BC dominates rule directionality
- If Δ@x=18 > +10 (clear crescent): rule directionality dominates BC
- If -10 ≤ Δ@x=18 ≤ +10 (near zero): they cancel
- If Δ@x=18 sign is opposite to either of (a)/(b): inconclusive — log
  needs deeper analysis

**Exit criteria**: ✅ result with sign, magnitude, and classification logged in IDEALOG.md.

**Risk**: low. ~30 second storage-tank run.

---

## Phase D — Final summary figure

**Goal**: side-by-side classification figure showing all monodomain and LBM
variants and their h-scaling, suitable for the writeup.

**File**: `Research/Active/boundary_conduction_speedup/figures/bc_classification.py` (NEW)

### Step D.0 — Create figures/ directory

```bash
mkdir -p /home/norepinephrine/Documents/Heart-Conduction/Research/Active/boundary_conduction_speedup/figures
```

**Verify**: `ls -d figures/`.

**Exit criteria**: ✅ trivially.

### Step D.1 — Implement plot script

**Read first**: Phase C save scheme (`lat_{mode}_dx{dx}.npz` in monodomain
outputs; `summary.json` in bidomain outputs). Verify both schemas before
implementing.

**Why**: assemble cross-engine comparison.

**Implementation spec**:

```python
"""Final classification figure for the BC discretization study.

Loads results from:
  - Monodomain/Engine_V5.4/experiments/bc_discretization_test/outputs/
  - Bidomain/Engine_V1/experiments/kleber_h_scaling/outputs/

Layout:
  Left panel:  per-column LAT deviation Δ(x) at finest h, all variants overlaid.
  Right panel: |Δ|_max vs h on log-log, fitted scaling slopes annotated.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
MD_OUT = ROOT / "Monodomain/Engine_V5.4/experiments/bc_discretization_test/outputs"
BD_OUT = ROOT / "Bidomain/Engine_V1/experiments/kleber_h_scaling/outputs"
OUT = Path(__file__).parent / "bc_classification.png"

# Map labels to (output_dir, summary_key, npz_prefix). Explicit, no inference.
SOURCES = {
    "monodomain face_mirror":           (MD_OUT, "face_mirror",           "lat_face_mirror"),
    "monodomain node_mirror_existing":  (MD_OUT, "node_mirror_existing",  "lat_node_mirror_existing"),
    "monodomain zero_pad":              (MD_OUT, "zero_pad",              "lat_zero_pad"),
    "bidomain bath-coupled":            (BD_OUT, "bath_coupled",          None),  # no per-x npz
}

colors = {
    "monodomain face_mirror":          "gray",
    "monodomain node_mirror_existing": "C5",
    "monodomain zero_pad":             "C0",
    "bidomain bath-coupled":           "C3",
}

# Load summaries
all_results = {}
for label, (out_dir, key, _) in SOURCES.items():
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        print(f"skipping {label}: no summary at {summary_path}", flush=True)
        continue
    with open(summary_path) as f:
        data = json.load(f)
    all_results[label] = data.get(key, {})

if not all_results:
    raise SystemExit("No experiment results found. Run Phases C first.")

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 5.5),
                                          gridspec_kw={"width_ratios": [3, 2]})

# Left panel: per-column Δ(x) at finest h (monodomain only — bidomain has no per-x npz)
finest_h = 0.0125
for label in [k for k in SOURCES if "monodomain" in k]:
    out_dir, key, npz_prefix = SOURCES[label]
    if npz_prefix is None: continue
    npz_path = out_dir / f"{npz_prefix}_dx{finest_h}.npz"
    if not npz_path.exists():
        print(f"  no per-column data for {label} at h={finest_h}", flush=True)
        continue
    d = np.load(npz_path)
    lat_2d = d["lat"]
    Nx, Ny = int(d["Nx"]), int(d["Ny"])
    deltas_x = []
    for x in range(Nx):
        col = lat_2d[x, :].astype(np.float64)
        if np.isnan(col).any():
            deltas_x.append(np.nan); continue
        top, mid, bot = col[0], col[Ny // 2], col[-1]
        deltas_x.append(0.5 * (top + bot) - mid)
    ax_left.plot(deltas_x, label=label, color=colors[label], lw=1.6)

ax_left.axhline(0, color='gray', lw=0.6)
ax_left.set_xlabel("x  (column)", fontsize=12)
ax_left.set_ylabel("Δ(x)  (ms,  negative = camel toe / boundary leads)", fontsize=11)
ax_left.set_title(f"Per-column LAT deviation at dx={finest_h} cm (monodomain variants)", fontsize=11)
ax_left.grid(alpha=0.3)
ax_left.legend(fontsize=9, loc='best')

# Right panel: h-scaling (all variants)
for label, data in all_results.items():
    if not data: continue
    hs_str = sorted(data.keys(), key=float)
    hs = []
    deltas = []
    for h_key in hs_str:
        v = data[h_key]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        hs.append(float(h_key))
        deltas.append(abs(v))
    if len(deltas) < 2: continue
    ax_right.loglog(hs, deltas, 'o-', label=label, color=colors[label], lw=1.8, ms=8)
    slope = float(np.polyfit(np.log(hs), np.log(deltas), 1)[0])
    ax_right.annotate(f"  p={slope:.2f}", xy=(hs[0], deltas[0]),
                      fontsize=9, color=colors[label])

ax_right.set_xlabel("dx  (cm)", fontsize=12)
ax_right.set_ylabel("|Δ(x_mid)|  (ms)", fontsize=12)
ax_right.set_title("h-scaling of boundary artifact", fontsize=12)
ax_right.grid(alpha=0.3, which='both')
ax_right.legend(fontsize=9, loc='best')

fig.suptitle("Boundary BC discretization: numerical artifact vs continuum effect", fontsize=13)
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"saved {OUT}", flush=True)
```

**Test spec**: figure exists; visual inspection.

**Verify**:
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Research/Active/boundary_conduction_speedup/figures
python bc_classification.py
ls -la bc_classification.png
```

**Exit criteria**: ✅ figure exists with all available variants and h-scaling slopes annotated.

**Risk**: file naming MUST match — `lat_{mode}_dx{dx}.npz` from Phase C; `bath_coupled` key in bidomain summary.json. Verify before running.

---

## Phase E — Documentation and conclusion

### Step E.1 — Update IDEALOG.md with results

**Read first**: existing IDEALOG.md.

**Why**: close the experimental loop.

**Implementation spec**: append to `## Thread`:

```markdown
### YYYY-MM-DD: BC discretization H-scaling — empirical results

| variant                          | Δ@x_mid h=0.05 | h=0.025 | h=0.0125 | log-log p |
|----------------------------------|----------------|---------|----------|-----------|
| monodomain face_mirror           | ___            | ___     | ___      | ___       |
| monodomain node_mirror_existing  | ___            | ___     | ___      | ___       |
| monodomain zero_pad              | ___            | ___     | ___      | ___       |
| bidomain bath-coupled            | ___            | ___     | ___      | ___       |

Findings:
- [ ] face_mirror gave Δ ≈ 0 at all h?  (PASS criterion: ratio < 0.02)
- [ ] zero_pad and node_mirror_existing showed h-dependent decline? (p ∈ [0.5, 2.5])
- [ ] bidomain bath-coupled stayed ~constant in h? (|p| < 0.2)
- [ ] C.3 diagnostic: node_mirror_existing as proxy for storage-tank reflect_y → ___
- [ ] C.4 discriminating test: gradient_bidir_reflect Δ@x=18 = ___, classification = ___

Conclusion: H_BC {confirmed | falsified | partially confirmed — see notes}.
```

**Verify**: `git status -s IDEALOG.md` shows the file as modified.

**Exit criteria**: ✅ all entries filled in.

### Step E.2 — Update KNOWLEDGE.md with empirical results

**Read first**: KNOWLEDGE.md "Boundary BC discretization" section.

**Implementation spec**: add an "Empirical results (2026-MM-DD)" subsection
to the existing section. Promote prediction → confirmed/falsified.

**Verify**: file modified.

**Exit criteria**: ✅ done.

### Step E.3 — Update README.md and MASTER_KNOWLEDGE_INDEX.md

**Read first**: `Research/Active/boundary_conduction_speedup/README.md` —
verified sections are: Question, Status, Why It Matters, Engines, Completion
Criteria, Sub-Questions, Experiments, Literature, Engine References,
**Future Work**, Connected Research. README does NOT have an "Open Questions"
section; use "Completion Criteria" (to mark sub-finding as DONE) and
"Sub-Questions" (to add BC discretization as a settled sub-question).
Also update `MASTER_KNOWLEDGE_INDEX.md` row.

**Why**: communicate the finding at project scope.

**Implementation spec**:
- README "Completion Criteria": add `[x] BC discretization sub-question resolved (PLAN.md Phase C)`.
- README "Sub-Questions": optionally add a row for BC discretization linking to KNOWLEDGE.md "Empirical results" subsection from Step E.2.
- README "Connected Research": link to MASTER_KNOWLEDGE_INDEX.md row for cross-reference.
- `MASTER_KNOWLEDGE_INDEX.md`: update the boundary_conduction_speedup row with the BC sub-finding (e.g., "BC-discretization classification: numerical artifact under non-standard Neumann; physical under bath-coupled Dirichlet").

**Verify**: `git status -s README.md MASTER_KNOWLEDGE_INDEX.md` shows both as modified.

**Exit criteria**: ✅ both updated; the "Open Questions" reference from
prior PLAN drafts is replaced with the actual README structure (Completion
Criteria + Sub-Questions).

### Step E.4 — Knowledge promotion checkpoint

Decide whether BC discretization is now a SETTLED sub-question and whether
to promote to `Research/Knowledge/`, or whether to keep open pending Phase F.

**Verify**: written decision in IDEALOG.

**Exit criteria**: ✅ next-step direction is clear.

---

## Phase F — DEFERRED: node_mirror_with_fold (future work)

Implement an explicit-Euler diffusion path in V5.4 that doesn't need an SPD
solver, then add `node_mirror_with_fold` mode. Test against storage-tank
reflect_y for direct camel-toe reproduction. Deferred from this round because
V5.4 has no GMRES/BiCGStab and the implicit Crank-Nicolson path requires SPD.

---

## Risk register (revised)

- **R1 — V5.4 default discretization is node-centered.** Confirmed by audit.
  Phase A includes `node_mirror_existing` as a labeled mode. The previously
  reported "0.000 cm deviation from flat monodomain control" was on a
  DIFFERENT code path (Bidomain V1 monodomain mode + Mehrstellen + Neumann)
  per the API reference card section 9. PLAN.md's predictions only assume
  face_mirror gives ≈0; the V5.4 default's behavior is an open question
  empirically resolved by Phase C.

- **R2 — V5.4 has no non-symmetric solver.** Verified: only PCG, Chebyshev,
  DCT, FFT exist. node_mirror_with_fold (which would break SPD) is therefore
  deferred to Phase F.

- **R3 — Stencil correctness errors.** Mitigated by hand-computed corner
  tests in Step A.4 (face_mirror corner = -5; zero_pad corner = -37 for the
  prescribed 4×4 V grid).

- **R4 — Empirical h-scaling exponent unknown.** Plan now MEASURES rather
  than asserts. Accept p ∈ [0.5, 2.5] as confirming "discretization-vanishing".
  If p < 0.5 or > 2.5 → unexpected, requires deeper analysis but doesn't
  block reporting.

- **R5 — Computational cost at fine mesh.** Crank-Nicolson is unconditionally
  stable (no CFL on diffusion). PCG cost scales mostly linearly with n_dof.
  Estimate: ~10-30 min per (mode, dx) at dx=0.0125; 9 monodomain runs ≈
  1.5-4.5 hours; bidomain at dx=0.0125 ~2 hours. Total Phase C ~4-7 hours.
  Run in background.

- **R6 — Storage-tank ↔ monodomain proxy validity.** Step C.3 directly
  tests this. If proxy fails, escalate to Phase F.

- **R7 — Stimulus may not reliably initiate a wave on small grids.** -52
  µA/µF for 5 ms is the V5.4 default and should work. If wave fails to
  initiate (LAT all NaN), increase amplitude to -80 µA/µF or duration to
  10 ms.

- **R8 — TTP06 V_rest is -85.23 mV; LAT threshold of -20 mV is a safe choice
  (well above resting and below peak ~+30 mV).** Confirmed by reading
  ttp06/parameters.py.

---

## Estimated effort (revised)

```
   Phase  Description                            Time
   ────────────────────────────────────────────────────
   A      monodomain boundary_mode + tests       6–10 hours
   B      LBM vacuum BC + tests                  2–4 hours
   C      mesh-refinement experiments            6–10 hours (compute) + 2 hours setup
   D      summary figure                         1–2 hours
   E      documentation + promotion              2–3 hours
   F      DEFERRED (future work)                 N/A
   ────────────────────────────────────────────────────
   total                                         17–29 hours
```

Critical path: A → C.1 → C.3 → D. B and C.4 can run in parallel with A.

---

## Done criteria

This plan is COMPLETE when:
1. ✓ Phase A tests pass; three modes (face_mirror, node_mirror_existing,
   zero_pad) produce expected behavior on hand-computed corner cases.
2. ✓ Phase B tests pass; vacuum BC zeros incoming populations correctly.
3. ✓ Phase C produces summary.json + h_scaling.png + per_column_delta.png +
   h_scaling_fits.json with empirical scaling slopes.
4. ✓ Phase C numerical pass criteria are evaluated and recorded:
   - face_mirror max |Δ| / others max |Δ| < 0.02 OR explicit FAIL note
   - zero_pad and node_mirror_existing scaling p ∈ [0.5, 2.5] OR FAIL note
   - bidomain Kleber scaling |p| < 0.2 OR caveat note
5. ✓ Step C.3 diagnostic concluded with written verdict on proxy validity.
6. ✓ Step C.4 discriminating test concluded with sign of Δ@x=18 logged.
7. ✓ Phase D figure exists with all available variants and slopes annotated.
8. ✓ IDEALOG.md, KNOWLEDGE.md, README.md (open questions section verified),
   MASTER_KNOWLEDGE_INDEX.md updated.

If results CONFIRM H_BC: storage-tank artifacts are discretization-specific;
standard cardiac monodomain/LBM with face-centered mirror Neumann don't
reproduce them; bath-coupled bidomain Kleber is the only continuum-level
boundary effect. Mark the corresponding item in the README's `Completion
Criteria` as done; update `MASTER_KNOWLEDGE_INDEX.md` with the BC
sub-finding.

If results FALSIFY H_BC: BC discretization explanation has a hole;
KNOWLEDGE.md gets a "what we got wrong" subsection. Falsification is itself
a publishable finding.

---

## Mutation Log

### Audit-1 mutations (2026-04-29)

**MUTATED**: Step A.0 — corrected engine path from `src/diffusion/operators/laplacian_5pt.py` to `cardiac_sim/.../discretization_scheme/fdm.py`. (audit-1 C1)

**MUTATED**: Step A.1 — replaced invented `Laplacian5pt` with FDM class modification preserving legacy default. (audit-1 C2)

**MUTATED**: Step A.2 — rewrote stencil pseudocode at matrix-construction level. Removed buggy corner formula. (audit-1 H1, M2)

**MUTATED**: Step A.3 — clarified `node_mirror_with_fold` requires non-SPD solver; added decision step.

**MUTATED**: Step A.4 — expanded test suite with corner cases and float64 dtype check. (audit-1 M3, L2)

**MUTATED**: Step B renamed and rewritten — LBM `absorbing` already exists; new BC named `vacuum`. Array shape corrected to `(N_pop, Nx, Ny)`. (audit-1 C4, C5)

**MUTATED**: Step C.1 — replaced invented MonodomainSimulation API with real one. (audit-1 C3)

**MUTATED**: Step C.0 EXPERIMENT.md — fixed `MASTER.md` → `MASTER_KNOWLEDGE_INDEX.md`. (audit-1 M6)

**ADDED**: Step C.3 — proxy validity diagnostic. (audit-1 H2)

**ADDED**: Step C.4 — discriminating test. (audit-1 H7)

**ADDED**: Step D.0 — explicit `mkdir figures/`. (audit-1 M8)

**MUTATED**: Step D.1 — explicit SOURCES dict mapping labels to file naming. (audit-1 M7)

**MUTATED**: Predictions table — added empirical-measurement disclaimer. (audit-1 H3)

### Audit-2 mutations (2026-04-29 — same day, after second pass)

**MUTATED**: Step C.1 `run.py` — every API call replaced with verbatim-correct
calls per the API reference card. Kwargs corrected: `dt=` (not `time_step=`),
`pcg_tol=` (not `linear_rtol=`), `linear_solver='pcg'` (not `'gmres'` —
GMRES doesn't exist in V5.4), `ionic_model='ttp06'` (string, not class
import). API methods corrected: `sim.run_to_array(t_end=...)` +
`sim.compute_activation_time(V_history, times, threshold=-20.0)` (not
fictional `run_record_iso`). Tissue building corrected:
`StructuredGrid.create_rectangle(Lx, Ly, Nx, Ny, device, dtype)` (not
`build_rectangle`). Stimulus corrected: `StimulusProtocol().add_stimulus(
region=callable, start_time=, duration=, amplitude=)` with units µA/µF
(NOT `amplitude_mV=20.0` which doesn't exist; depolarizing is NEGATIVE,
default -52). (audit-2 N1, N9)

**MUTATED**: Step B.2 tests — added `make_rect_bounce_masks` test helper
inline (copied from `tests/test_phase4.py:29-47`). `precompute_bounce_masks`
takes `(domain_mask: Tensor, lattice)`, NOT `(D2Q5, Nx, Ny, wall_axes=...)`.
Tests now use the helper for full-grid rectangular domains. (audit-2 N2)

**MUTATED**: Step A.2 matrix description — clarified that
`node_mirror_existing` produces matrix entry `2 * w_east` at boundary cells
(after coalesce()) by piling the mirror entry onto the existing cardinal-west
entry. The pseudocode now shows the full assembly logic, not a paraphrase.
(audit-2 N3)

**MUTATED**: Step D.1 SOURCES dict — eliminated phantom LBM entries since
Phase B doesn't include a mesh-refinement run for LBM. SOURCES now has only
3 monodomain modes + 1 bidomain entry (4 total). LBM h-scaling deferred to
a separate future experiment. The "adapt if naming differs" trailing
comments removed. (audit-2 N4, N22)

**MUTATED**: Predictions table + EXPERIMENT.md setup — `chi=1400`, `Cm=1.0`,
`D=0.001` standardized across all variants. CV-ratio comparison to bidomain
in Step C.2 must use matching chi/Cm. (audit-2 N5)

**MUTATED**: Step A.4 fixture — uses `FDMDiscretization` (real class name)
not `FDMScheme` (fictional). Fixtures use `_make_fdm` helper that calls
`StructuredGrid.create_rectangle` then `FDMDiscretization` with real kwargs.
(audit-2 N7)

**MUTATED**: Step A.4 corner-cell tests — arithmetic justifications now
match the matrix-construction semantics, not handwaved 4-neighbor sums.
Hand values: face_mirror corner = -5; zero_pad corner = -37. (audit-2 N8)

**MUTATED**: Step C.1 stimulus — `amplitude=-52.0` µA/µF (real kwarg name,
real units, depolarizing sign correct), `region=lambda x, y: x < 0.1`
(callable accepted by `add_stimulus`). NO voltage-clamp at 0 mV (V5.4 has
no voltage-clamp Stimulus). (audit-2 N9)

**MUTATED**: Step A.2 "Read first" reference — `Research/Q1_spatial_discretization/01_FDM_Stencils_and_Implementation.md` (real path) NOT `Research/openCARP_FDM_FVM/01_FDM/` (fictional). (audit-2 N10)

**MUTATED**: Phase A scope — Mehrstellen 9-point default has known
diagonal-skip incompleteness at corners (lines 322-349 in fdm.py). Phase C
uses 9-point but ALL boundary modes share the same diagonal-skip behavior,
so the comparison is internally valid. Documented in Step A.2 Risk note.
(audit-2 N11)

**MUTATED**: KNOWLEDGE.md classification table — separated
`Bidomain V1 monodomain mode` (face-centered, 0.000 cm result) from
`Monodomain V5.4 FDMDiscretization` (node-centered, TBD). Table no longer
falsely conflates them. The "Monodomain FDM control: No boundary speedup"
text was qualified to specify which engine produced that result. Done in
companion edit, not in PLAN.md. (audit-2 N12, N13)

**MUTATED**: Step C.4 interpretation — added 4th outcome ("inconclusive —
log needs deeper analysis") for cases where the sign is opposite to either
single-axis result. (audit-2 N14)

**MUTATED**: Step A.3 grep target — `cardiac_sim/simulation/classical/solver/diffusion_time_stepping/linear_solver/` (real path). Verified: only PCG/Chebyshev/DCT/FFT exist, no GMRES. node_mirror_with_fold deferred to Phase F. (audit-2 N15)

**MUTATED**: Step C.2 Risk note — added explicit timeboxing: "If dx=0.0125 bidomain >2 hours, defer the dx=0.0125 point and report 2-point scaling with a caveat." (audit-2 N16)

**MUTATED**: Predictions table qualifier — "5-point stencil only is the immediate target; Mehrstellen has diagonal-skip caveat." (audit-2 N17)

**MUTATED**: Steps E.1-E.3 verify — replaced bare `git diff` with `git status -s <file>` checks (no working-tree assumption). (audit-2 N18)

**MUTATED**: Effort estimate — refined per-phase, separated Phase F as deferred. (audit-2 N19)

**MUTATED**: Mutation Log — split into audit-1 and audit-2 sections for traceability. (audit-2 N20)

**MUTATED**: Step E.3 — added "verify exists" check before edit instruction for README.md. (audit-2 N21)

**ADDED**: Phase F — explicit "future work" placeholder for node_mirror_with_fold pending an explicit-Euler diffusion path in V5.4.

### Audit-3 mutations (2026-04-29 — third pass, same day)

**MUTATED**: Step A.2 hand-verification — corrected arithmetic. The verify
prose previously claimed `V[2,3]=11` (wrong) and computed `node_mirror = -12,
face_mirror = -6, zero_pad = -38`. Correct values with `V[2,3]=12, V[3,2]=15`:
`node_mirror_existing = -10, face_mirror = -5, zero_pad = -37`. Step A.4
test code already had the correct face_mirror=-5 and zero_pad=-37; now the
prose matches. Also added matrix-form derivations to make the doubling
mechanics in node_mirror_existing explicit. (audit-3 H1)

**ADDED**: `test_corner_cell_hand_check_node_mirror_existing` to Step A.4 —
verifies the node_mirror corner value (-10) that wasn't covered by the
existing face_mirror/zero_pad corner tests. (audit-3 H1)

**MUTATED**: Step C.3 storage-tank invocation — `damping_cap=True` (was
`False`) to match the IDEALOG empirical anchor `Δ@x=18 = -270` which used
the GRADIENT config default (damping_cap=True). Comment cites IDEALOG.md:148.
(audit-3 H2)

**MUTATED**: V_rest cited as -85.23 mV throughout (was -86.2 mV). Verified
against `cardiac_sim/ionic/ttp06/parameters.py:264` (`V_REST = -85.23`).
LAT threshold of -20 mV remains safe (well above either value). (audit-3 H3)

**MUTATED**: Step E.3 — corrected README.md section reference. The README
has "Completion Criteria", "Sub-Questions", "Future Work", "Connected
Research" — but NO "Open Questions" section. Step E.3 now uses the actual
section structure (Completion Criteria + Sub-Questions). (audit-3 M3)

**MUTATED**: Step C.1 run.py pass/fail evaluation — replaced
`ratio = fm_max / max(others_max, 1e-12)` (which silently masked the
divide-by-zero edge case) with a three-way verdict that handles
"all-near-zero" and "face-mirror-dominates" as separate explicit outcomes.
Includes a SIGNIFICANCE_FLOOR_MS = 0.5 threshold below which |Δ| is
considered "no effect detected". (audit-3 M4)

**MUTATED**: Step A.2 Risk note — added explicit "SPD properties of the
three modes" sub-bullet documenting that all three Phase A modes preserve
SPD and are compatible with V5.4's existing PCG/Chebyshev/DCT/FFT solvers.
(audit-3 N6 from audit-2 partial)

### Audit-3 issues NOT addressed in this revision (deferred / acceptable)

- **N17 (audit-2)**: predictions table doesn't carry per-row Mehrstellen
  caveat. Step A.2 risk-note covers it; acceptable.

- **L1, L2, L3, L4, L5 (audit-3)**: minor doc/UX issues (GPU-only assumption,
  no progress callback, BOUNDARY_MODES tuple docstring polish, REGISTRY
  wording, effort-estimate inconsistency). Not blocking. Acceptable as-is.

- **M1, M2 (audit-3)**: Step A.2 doesn't show diagonal handling under the
  three boundary modes; Step C.2 spec is light on `kleber_h_scaling/run.py`.
  Both deferred — implementer reads existing code for context. Not blocking
  for cold-start execution but worth tightening if the implementer has
  questions.

### Future companion-doc fixes

The following corrections belong in IDEALOG.md and KNOWLEDGE.md, not PLAN.md, but flagged here for traceability:

- KNOWLEDGE.md and IDEALOG.md: cascade story already amended with "Caveat — the cascade requires SEEDED asymmetry" subsection (audit-1 H4 H5). PLAN.md's Step C.3 verifies this empirically.

- KNOWLEDGE.md classification table: corrected to separate Bidomain V1 monodomain mode from V5.4 FDMDiscretization (audit-2 N12, applied 2026-04-29).

- KNOWLEDGE.md "Monodomain FDM control: No boundary speedup" text: qualified to specify Bidomain V1 origin and noted V5.4 status as TBD pending Phase C (audit-2 N13, applied 2026-04-29).
