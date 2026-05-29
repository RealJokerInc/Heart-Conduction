# PLAN: Moore-8 / iso-9pt stencil extension to monodomain V5.4 + LBM

Created: 2026-04-30
Engine(s): Monodomain V5.4, LBM V1
Research question: [boundary_conduction_speedup](README.md)
Source: [IDEALOG.md](IDEALOG.md) — 2026-04-29 thread "Connectivity × threshold-gate ablation",
2026-04-30 thread "Wave-slowing dilation"
Predecessor: [plans/2026-04-29_bc-discretization-test.md](plans/2026-04-29_bc-discretization-test.md) — completed (face_mirror default flip, ablation R1-R6, normalised diagnostics)

## Objective

Extend the storage-tank connectivity ablation finding (Moore-8 = boundary deficit;
cardinal-4 = no deficit; iso 4:1 reduces deficit by half but does not eliminate it
without diagonal-aware bounce-back) into the production cardiac solvers. Add
configurable stencil to `Monodomain V5.4 FDMDiscretization` and configurable
direction weights to `LBM V1 D2Q9`. Implement the diagonal-aware face_mirror
reflection that closes the LBM bounce-back analogy. Validate by porting the
y-uniform line-stim column diagnostic to monodomain and predicting the same
crescent-presence pattern we observed in John's tanks.

### Two BC schemes, both delivered:

This PLAN delivers TWO inequivalent diagonal-handling schemes for the new
Moore-8 stencils — both serve the research narrative:

- **Scheme B** (`boundary_mode='face_mirror'`, default): for ALL off-grid
  pipes (cardinal AND diagonal), ghost = boundary cell value → flux = 0.
  This is the faithful translation of John's `zero_pad`/`valid` mask
  semantics into FDM ghost-cell language: boundary cells genuinely have
  fewer effective neighbours, so the deficit is REAL. Tells the story
  *"the artifact appears in monodomain too if you discretise it the way
  John's tanks do."*

- **Scheme A** (`boundary_mode='face_mirror_iso'`, NEW in Phase 2): for
  cardinal off-grid, ghost = self (same as Scheme B). For diagonal
  off-grid, mirror only the off-grid axis: ghost(i+di, -1) = V[i+di, 0].
  This is the LBM-bounce-back analog and the higher-order Patra-Kałuża-
  consistent reflection. Tells the story *"the same connectivity can give
  ZERO boundary effect with proper higher-order numerics."*

The two schemes coexist in the codebase. Phase 3's column diagnostic
exercises all 5 (stencil, BC) pairings to show:
  cardinal4 + B → 0     (baseline, no diagonals to lose)
  moore8_uniform + B → 1/3  (John-equivalent — bridge claim CONFIRMED)
  moore8_uniform + A → 0    (Scheme A hides the deficit)
  moore8_iso + B     → 5/6  (iso weighting attenuates but doesn't eliminate)
  moore8_iso + A     → 0    (LBM bounce-back analog — full elimination)

## Success Criteria

- [ ] `FDMDiscretization` accepts `stencil ∈ {cardinal4, moore8_uniform, moore8_iso}` keyword
- [ ] All three stencils satisfy row-sum = 0 (mass conservation) within 1e-12 in y-uniform fields
- [ ] `boundary_mode='face_mirror_iso'` correctly handles diagonal off-grid neighbours so iso 4:1 + face_mirror_iso gives boundary == interior in y-uniform fields (max |L*V_boundary - L*V_interior| < 1e-12)
- [ ] Monodomain column diagnostic empirically distinguishes the 3 stencil regimes:
  - cardinal4 + face_mirror: ~1e-13 mV deviation, ~0 µs LAT shift (already verified, baseline)
  - moore8_uniform + face_mirror: deviation **above floating-point noise** (> 1e-9 mV) and LAT shift > 0 µs (magnitude TBD — depends on how I_ion's V_advantage clamp interacts with the 2/3 deficit; could be sub-µs and visible only with sub-ms save_every)
  - moore8_iso + face_mirror_iso: same low-noise floor as cardinal4 (< 1e-12 mV deviation)
  - **The success criterion is the ORDERING and the qualitative split into "no deficit" vs "deficit", NOT a specific LAT shift magnitude.** A measurable but small shift for moore8_uniform is a positive result.
- [ ] LBM V1 D2Q9 accepts `weights_mode ∈ {'canonical', 'uniform_8'}` keyword; canonical is default
- [ ] LBM under uniform_8 weights produces visible crescent in line-stim setup; LBM under canonical weights does not
- [ ] All existing V5.4 tests pass (`test_phase7.py` 7/7, `test_phase8.py` 7/7, `test_boundary_modes.py` 5/5)
- [ ] All existing LBM V1 tests pass (`tests/` pytest suite)
- [ ] Final figure `connectivity_cross_engine.png` shows side-by-side: John's tanks / monodomain V5.4 / LBM V1, all under matched conditions, demonstrating the same Moore-8-deficit / cardinal-no-deficit / iso-with-bounce-back pattern in all three model classes

## Architecture Changes

- **MOD**: `Monodomain/Engine_V5.4/cardiac_sim/simulation/classical/discretization_scheme/fdm.py`
  - Add `STENCILS = ('cardinal4', 'moore8_uniform', 'moore8_iso')` class constant
  - Add `stencil: str = 'cardinal4'` parameter to `__init__`
  - Add `'face_mirror_iso'` to `BOUNDARY_MODES`
  - Refactor `_build_laplacian` to dispatch on stencil (rename existing body to `_build_laplacian_cardinal`)
  - New helpers: `_build_laplacian_moore8(self, Dxx, Dyy, mask, weighting)` covering both uniform and iso branches
- **MOD**: `Monodomain/Engine_V5.4/test_boundary_modes.py`
  - Step 1.1 dispatch tests: `test_a8_invalid_stencil_rejected`, `test_a8_moore8_stub_raises_not_implemented`, `test_a8_moore8_rejects_anisotropic_D`, `test_a8_moore8_rejects_non_square_grid`
  - Step 1.2 uniform tests: `test_a9_moore8_uniform_constant_field`, `test_a9_moore8_uniform_y_uniform_x_linear`, `test_a9_moore8_uniform_boundary_deficit_2_over_3`
  - Step 1.3 iso tests: `test_a10_moore8_iso_recovers_continuum_in_interior`, `test_a10_moore8_iso_y_uniform_interior_matches_cardinal`, `test_a10_moore8_iso_boundary_deficit_5_over_6`, `test_a10_cfl_stability_check`, `test_a10_all_three_stencils_construct`
  - Step 2.1 face_mirror_iso tests: `test_a11_face_mirror_iso_in_iso_stencil_zero_deficit`, `test_a12_face_mirror_iso_with_uniform_stencil`, `test_a13_face_mirror_iso_corner_handling`, `test_a13_cardinal4_face_mirror_iso_degenerate`
- **NEW**: `Research/Active/boundary_conduction_speedup/diag_monodomain_connectivity.py`
  - Port of `simulation/connectivity_threshold_ablation.py` to monodomain V5.4
- **NEW**: `LBM/Engine_V1/src/lattice/d2q9_uniform.py`
  - Defines `D2Q9_uniform` lattice variant with weights `[0, 1/8, ..., 1/8]` and `cs2 = 0.75`
- **MOD**: `LBM/Engine_V1/src/lattice/__init__.py`
  - Export `D2Q9_uniform` alongside existing `D2Q5`, `D2Q9`
- **MOD**: `LBM/Engine_V1/src/simulation.py`
  - Add `weights_mode: str = 'canonical'` parameter to `LBMSimulation.__init__`
  - Dispatch to `D2Q9()` (canonical) or `D2Q9_uniform()` based on weights_mode
  - **CRITICAL FIX**: change `tau = tau_from_D(D, dx, dt)` on line 73 to `tau = tau_from_D(D, dx, dt, cs2=self.lattice.cs2)` — currently uses default `cs2=1/3` regardless of lattice, which is a latent bug for any non-canonical lattice
  - Reject `weights_mode='uniform_8'` if `lattice='d2q5'` (uniform_8 is D2Q9-only). NO MRT-guard required: `LBMSimulation` always uses BGK in current code, so a guard would be dead code (see Step 4.1 Implementation Spec for the explicit no-action decision)
- **MOD**: `LBM/Engine_V1/src/diffusion.py`
  - No code change needed: `tau_from_D` already accepts `cs2` parameter; the bug is in simulation.py not passing it through
- **NEW**: `LBM/Engine_V1/tests/test_d2q9_uniform_weights.py`
  - 7 tests: `test_canonical_weights_unchanged`, `test_uniform_weights_construction`, `test_uniform_cs2_self_consistency`, `test_uniform_fourth_moment_not_isotropic`, `test_tau_from_D_uses_lattice_cs2`, `test_uniform_propagation_creates_boundary_artifact`, `test_d2q5_rejects_uniform_weights_mode`
  - File naming follows the existing LBM convention (`test_phaseN.py` is used for engine-phase tests; per-feature tests like `test_d2q9_uniform_weights.py` are acceptable per the existing `tests/` layout — verify against `LBM/Engine_V1/tests/` before committing)
- **NEW**: `Research/Active/boundary_conduction_speedup/figures/connectivity_cross_engine.py`
  - Cross-engine validation figure script

## Known Failures (from IDEALOG)

- **Iso 4:1 weights without 1/6 normalisation prefactor** — produces D_eff = 6k = 0.48, violates 2D explicit-diffusion CFL limit of 0.25, manifests as grid-scale mosaic instability. Diagnostic: check D_eff·Δt/Δx² ≤ 1/4 before reading scientific conclusions.
- **Iso 4:1 weights without diagonal-aware face_mirror reflection** — boundary cell still has 5/6 deficit even with proper 1/6 normalisation, because the off-grid diagonals contribute 0 (zero_pad-style) instead of mirroring to the in-grid cell at the boundary row. Crescent reduces ~12% but does NOT vanish. Phase 2 of this plan is the fix.
- **`apply_diffusion` V-shift trick for non-zero ghost values** — only works for `rest_pad` mode where ghost is constant. For `face_mirror_iso` where the diagonal ghost is V[i±1, 0] (varies with x), V-shift cannot be applied. Implement directly in matrix construction instead.
- **`node_mirror_existing` matrix asymmetry** — empirically 1.0 mV asymmetry on 8x8 grid. Operator is self-adjoint in a weighted inner product but raw COO is not. New stencils should preserve symmetry (face_mirror, face_mirror_iso, zero_pad all give symmetric matrices).

---

## Phase 1: V5.4 FDM Moore-8 stencils (no diagonal mirror yet)

**Goal**: Add `moore8_uniform` and `moore8_iso` stencils to `FDMDiscretization`. Initial implementation uses existing `face_mirror` BC handling (diagonal off-grid contributes 0 — same as `valid` mask in John's code). This will reproduce John's R5 result (~12% reduction, NOT zero deficit) in monodomain.

**Tier**: medium
**Estimated scope**: ~150 lines of new code, ~80 lines of new tests, ~6-8 hours (revised up from initial 3-hour estimate; the COO assembly with 8-direction enumeration + boundary-mode dispatch + symmetry verification across 3 weighting variants is more involved than the cardinal-only version)

### Phase Context

- V5.4 FDMDiscretization currently uses 5-point cardinal Laplacian with optional Dxy cross-derivative diagonals. The Dxy diagonals are NOT the same as the iso 9-pt diagonals — they carry sign according to NE/SW vs NW/SE for the cross-derivative term. The new isotropic stencils need symmetric all-positive diagonal weights for symmetric matrix.
- Existing boundary modes are `face_mirror` (default since 2026-04-29), `node_mirror_existing`, `zero_pad`, `rest_pad`. All four operate on cardinal pipes; this phase extends them to also handle diagonal pipes consistently.
- Float64 only. Conda env `heart-conduction`. Tests run as standalone scripts: `python test_boundary_modes.py`.
- New stencils support **isotropic scalar D only** for Phase 1. If `D_field` is provided with non-zero Dxy, raise NotImplementedError for moore8 stencils. The existing cardinal4 stencil retains its full Dxy support.
- Target weights:
  - `cardinal4` (default): existing 5-point + Dxy correction (unchanged)
  - `moore8_uniform`: cardinals weight `1/(3·h²)`, diagonals weight `1/(3·h²)`. In y-uniform interior gives `(V_E + V_W − 2·V_C)/h²` matching continuum.
  - `moore8_iso`: cardinals weight `4/(6·h²)`, diagonals weight `1/(6·h²)`. Patra-Kałuża isotropic.
- After Phase 1, iso 4:1 + face_mirror gives ~5/6 boundary deficit (the same 12% reduction we saw in John's R5). Phase 2 adds the diagonal-aware reflection to fix this.

### Step 1.1: Add stencil parameter and dispatch
**Model**: opus

#### Read First
- `Monodomain/Engine_V5.4/cardiac_sim/simulation/classical/discretization_scheme/fdm.py:60-130` — current docstring, BOUNDARY_MODES, __init__
- `Monodomain/Engine_V5.4/cardiac_sim/simulation/classical/discretization_scheme/fdm.py:230-435` — current `_build_laplacian` body

#### Why
The dispatch layer must be added before any new stencil implementation, so subsequent steps can plug in cleanly. Doing this cleanly upfront avoids tangling the existing cardinal+Dxy logic with the new isotropic Moore-8 logic.

#### Implementation Spec
**Files to modify:** `fdm.py:88-132`
**Interfaces:**
```python
class FDMDiscretization(SpatialDiscretization):
    BOUNDARY_MODES = ('face_mirror', 'node_mirror_existing', 'zero_pad', 'rest_pad')
    STENCILS = ('cardinal4', 'moore8_uniform', 'moore8_iso')

    def __init__(
        self, grid, D=0.001, chi=1400.0, Cm=1.0,
        D_field=None,
        boundary_mode='face_mirror', pad_value=0.0,
        stencil: str = 'cardinal4',  # NEW
    ):
        if stencil not in self.STENCILS:
            raise ValueError(f"stencil must be one of {self.STENCILS}, got {stencil!r}")
        self._stencil = stencil
        # ... rest unchanged ...
```

In `_build_laplacian`, add a top-level dispatch. Rename the existing body to `_build_laplacian_cardinal`.

For Moore-8 builders, raise `NotImplementedError` if `D_field` includes non-zero Dxy.

#### Pseudocode
```python
def _build_laplacian(self, Dxx, Dxy, Dyy, mask):
    if self._stencil == 'cardinal4':
        return self._build_laplacian_cardinal(Dxx, Dxy, Dyy, mask)
    if self._stencil in ('moore8_uniform', 'moore8_iso'):
        if Dxy is not None and torch.as_tensor(Dxy).abs().max().item() > 0:
            raise NotImplementedError(
                "Moore-8 stencils currently support isotropic scalar D only. "
                "For anisotropic Dxy, use stencil='cardinal4'."
            )
        if abs(self._dx - self._dy) > 1e-12:
            raise NotImplementedError(
                f"Moore-8 stencils require dx == dy "
                f"(got dx={self._dx}, dy={self._dy})"
            )
        weighting = 'uniform' if self._stencil == 'moore8_uniform' else 'iso'
        return self._build_laplacian_moore8(Dxx, Dyy, mask, weighting=weighting)
```

#### Test Spec
- `test_boundary_modes.py::test_a8_invalid_stencil_rejected` — Setup: `FDMDiscretization(grid, stencil='unknown')`. Expected: `ValueError` mentioning STENCILS allowed-list.
- `test_boundary_modes.py::test_a8_moore8_stub_raises_not_implemented` — Setup: build with `stencil='moore8_uniform'`, isotropic D scalar. Expected: `NotImplementedError` from the stub `_build_laplacian_moore8`. Will be replaced with positive tests in Step 1.2/1.3 once the builder is implemented.
- `test_boundary_modes.py::test_a8_moore8_rejects_anisotropic_D` — Setup: `stencil='moore8_uniform'`, `D_field=(D, Dxy_nonzero, D)`. Expected: `NotImplementedError` mentioning anisotropic D not supported (rejection happens BEFORE the stub raises, in the dispatch validation).
- `test_boundary_modes.py::test_a8_moore8_rejects_non_square_grid` — Setup: `stencil='moore8_uniform'`, grid with dx != dy. Expected: `NotImplementedError` mentioning dx == dy required.

(The full "all 3 stencils build successfully" dispatch-validity test is deferred to Step 1.3, after the builder is implemented.)

#### Checklist
- [ ] Add `STENCILS` constant
- [ ] Add `stencil` parameter to `__init__` with allowed-list validation
- [ ] Rename existing `_build_laplacian` body to `_build_laplacian_cardinal`
- [ ] Add top-level dispatch in `_build_laplacian` (validates Dxy and dx==dy before calling stub)
- [ ] Add stub `_build_laplacian_moore8(self, Dxx, Dyy, mask, weighting)` that raises `NotImplementedError("Moore-8 builder not yet implemented; awaiting Step 1.2/1.3")`
- [ ] Add 4 dispatch tests above

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.4
conda run -n heart-conduction python test_boundary_modes.py
conda run -n heart-conduction python test_phase7.py
conda run -n heart-conduction python test_phase8.py
```

#### Exit Criteria
- [ ] Dispatch layer in place; tests pass
- [ ] cardinal4 (default) gives bit-identical Laplacian to before flip
- [ ] Moore-8 stencils raise NotImplementedError when called

#### Risk
**Risk**: When `D_field=None`, the constructor auto-builds `Dxy = zeros(nx, ny)`. So a `Dxy is None` check would NOT distinguish "isotropic default" from "user passed zero anisotropy". The real protection is the magnitude check `Dxy.abs().max().item() > 0`, which is True only when the user provides a non-zero Dxy field. Without this distinction the dispatch would either always reject (false positive on the default isotropic case) or never reject (false negative on user-supplied zero Dxy → silent wrong stencil). — mitigation: use `Dxy is None or Dxy.abs().max().item() == 0` as the "isotropic" predicate; rejection fires only when both branches fail.

---

### Step 1.2: Implement moore8_uniform Laplacian builder
**Model**: opus

#### Read First
- `fdm.py:230-435` (renamed `_build_laplacian_cardinal`) — to follow the same construction style (active-mask, harmonic-mean D_face, COO assembly, boundary-mode branches)
- IDEALOG.md `2026-04-29 (cont): Connectivity × threshold-gate ablation` — for the deficit-ratio prediction (2/3 in y-uniform)
- `simulation/tanks_vec.py:165-200` — for reference implementation of the 8-direction enumeration with valid mask

#### Why
The "uniform Moore-8" stencil weights each of the 8 neighbours equally. In y-uniform interior, this gives 3× the cardinal Laplacian (1 cardinal pair × 1 + 2 diagonal pairs × 1 = 3). We normalise by 1/3 so the operator approximates the continuum Laplacian magnitude correctly. Without this normalisation, D_eff scales with the stencil and we get unintended fast propagation (the same trap we hit with iso 4:1 by a different factor).

The boundary handling for Phase 1 is "off-grid pipes contribute 0" (matching what `face_mirror` already does for cardinals: ghost = self → flux = 0). For diagonals at the boundary, the off-grid diagonal contributes 0 just like a cardinal — same handling. This gives the 2/3 boundary deficit predicted by the ablation analysis.

#### Implementation Spec
**Files to modify:** `fdm.py` — fill in `_build_laplacian_moore8` body (uniform branch)

**Algorithm:**
- 8 Moore-neighbour offsets: `MOORE_8 = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]`
- For each cell `(i, j)`, for each offset `(di, dj)`:
  - Determine direction kind: cardinal (exactly one of di/dj is 0) or diagonal (both non-zero)
  - Determine if neighbour `(i+di, j+dj)` is in-grid AND active (mask)
  - If in-grid: harmonic mean D_face from D[i,j] and D[i+di, j+dj]; weight `w_card` or `w_diag`; add `+w` to off-diagonal `L[k, k']`, subtract `w` from diagonal `L[k, k]`
  - If off-grid: respect boundary_mode same as cardinals do today (face_mirror skips → 0 contribution; zero_pad subtracts w from diagonal; node_mirror_existing mirrors; rest_pad treats as zero_pad in matrix)

For h = dx = dy and weighting='uniform': `w_card = w_diag = 1.0 / (3.0 * h2)`.

**Convention for diagonal harmonic-mean D**: use harmonic mean of D[i,j] and D[i+di,j+dj] (geometric mean would also be defensible, but harmonic is consistent with cardinal pipe convention; document this choice in code comment).

#### Pseudocode
```python
def _build_laplacian_moore8(self, Dxx, Dyy, mask, weighting):
    nx, ny = self._nx, self._ny
    h = self._dx
    h2 = h * h
    if weighting == 'uniform':
        w_card_base = 1.0 / (3.0 * h2)
        w_diag_base = 1.0 / (3.0 * h2)
    elif weighting == 'iso':
        w_card_base = 4.0 / (6.0 * h2)
        w_diag_base = 1.0 / (6.0 * h2)

    # Build active-node index mapping (same as cardinal builder)
    # ... (re-use _is_active, _idx, _harm helpers)

    MOORE_8 = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]

    for i in range(nx):
        for j in range(ny):
            if mask_np is not None and not mask_np[i, j]:
                continue
            k = _idx(i, j)
            d_self = float(Dxx[i, j])  # Dxx == Dyy for isotropic
            center = 0.0
            for di, dj in MOORE_8:
                is_cardinal = (di == 0) ^ (dj == 0)
                w_base = w_card_base if is_cardinal else w_diag_base
                ni, nj = i + di, j + dj
                if _is_active(ni, nj):
                    D_face = _harm(d_self, float(Dxx[ni, nj]))
                    w = D_face * w_base
                    center -= w
                    _add(k, _idx(ni, nj), w)
                else:
                    # Off-grid neighbour: BC handling
                    if self._boundary_mode == 'face_mirror':
                        pass  # ghost = self -> flux = 0
                    elif self._boundary_mode == 'node_mirror_existing':
                        # mirror across the wall: replace ni or nj with in-grid mirror
                        ni_m = ni if 0 <= ni < nx else (i - di)  # reflect off x wall
                        nj_m = nj if 0 <= nj < ny else (j - dj)  # reflect off y wall
                        if _is_active(ni_m, nj_m) and (ni_m, nj_m) != (i, j):
                            D_face = _harm(d_self, float(Dxx[ni_m, nj_m]))
                            w = D_face * w_base
                            center -= w
                            _add(k, _idx(ni_m, nj_m), w)
                    elif self._boundary_mode in ('zero_pad', 'rest_pad'):
                        w = d_self * w_base
                        center -= w
                    elif self._boundary_mode == 'face_mirror_iso':
                        # Phase 2: implemented later
                        raise NotImplementedError(
                            "face_mirror_iso requires Phase 2 implementation"
                        )
            _add(k, k, center)
    # COO assembly (same as cardinal)
    # ...
```

#### Test Spec
- `test_boundary_modes.py::test_a9_moore8_uniform_constant_field` — Setup: V uniform = 1.0. Expected: L*V = 0 everywhere within 1e-12.
- `test_boundary_modes.py::test_a9_moore8_uniform_y_uniform_x_linear` — Setup: V[i,j] = i (linear in x, constant in y). Expected: L*V at INTERIOR cells (i in [1, nx-2], j in [1, ny-2]) approximates `D · ∂²(i)/∂x² = 0` within 1e-10.
- `test_boundary_modes.py::test_a9_moore8_uniform_boundary_deficit_2_over_3` — Setup: 8x8, isotropic D=1, h=1, **y-uniform x-step field**: `V[i, j] = 1.0 if i >= 4 else 0.0` for all j (so V is uniform in y, has a sharp x-step at i=4). Compute `L*V` at the wavefront column. Expected: at boundary cells (j=0 or j=Ny-1), `|L*V|` is exactly **2/3** of `|L*V|` at interior cells (j=Ny//2) within 1e-12. Justification: this matches the deficit-ratio prediction for moore8_uniform in y-uniform fields with an x-direction wavefront, the exact analog of John's tank ablation setup.

#### Checklist
- [ ] Implement the 8-direction loop with weighting='uniform' branch
- [ ] Apply harmonic-mean D averaging for in-grid neighbours
- [ ] Apply boundary-mode branches (face_mirror, node_mirror_existing, zero_pad, rest_pad)
- [ ] Verify row-sum = 0 in interior cells (constant-field test)
- [ ] Verify symmetric matrix (face_mirror, zero_pad branches; node_mirror is asymmetric as documented)
- [ ] Add 3 tests above

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.4
conda run -n heart-conduction python test_boundary_modes.py
```

#### Exit Criteria
- [ ] moore8_uniform tests pass
- [ ] Constant-field row-sum = 0 verified
- [ ] Boundary deficit 2/3 verified

#### Risk
**Risk**: Mistakenly applying harmonic-mean D averaging in a way that double-counts diagonal pipes. The harmonic mean is defined for cells sharing a face. For diagonal pipes (corner-touching) there is no shared face — geometric mean might be more appropriate physically, but harmonic preserves the convention from cardinal4. — mitigation: use harmonic mean for both, document the convention choice in code comment, defer revisiting.

---

### Step 1.3: Implement moore8_iso Laplacian builder (Patra-Kałuża 4:1)
**Model**: opus

#### Read First
- IDEALOG.md `2026-04-29 (cont): Bug fix — iso weights need 1/6 normalisation` — for the prefactor analysis
- Step 1.2 implementation in `fdm.py` — the iso branch reuses most of it

#### Why
Patra-Kałuża isotropic 9-point Laplacian has ~4th-order accuracy (vs 2nd-order for the 5-point cardinal). Cardinals weight 4/6, diagonals weight 1/6 — the 1/6 prefactor is the canonical normalisation (NOT optional). Without it, D_eff is 6× the intended coefficient and CFL stability is violated.

In y-uniform fields, the iso 9-pt with proper normalisation gives `(V_E + V_W − 2·V_C)/h² = ∂²V/∂x²` exactly, matching cardinal-4. So in interior, behaviour is unchanged. The DIFFERENCE shows up at the boundary, where iso 4:1 gives 5/6 deficit (vs 1/3 for uniform Moore-8 and 1.0 for cardinal-4).

#### Implementation Spec
Within `_build_laplacian_moore8`, the `weighting='iso'` branch sets:
```python
w_card_base = 4.0 / (6.0 * h2)
w_diag_base = 1.0 / (6.0 * h2)
```

The rest of the algorithm is identical to uniform.

#### Pseudocode
```python
# Inside _build_laplacian_moore8, after the weighting dispatch:
if weighting == 'iso':
    # Patra-Kałuża isotropic 9-pt: ∇²V ≈ (1/6h²)·[4·cards + diags - 20·V_self]
    # The 1/6 prefactor IS the canonical normalization; without it, D_eff = 6k
    # which violates 2D-explicit CFL limit (0.25) and produces grid-scale
    # mosaic instability — see IDEALOG.md "Bug fix — iso weights need 1/6
    # normalisation" for the specific failure we hit on John's tanks.
    w_card_base = 4.0 / (6.0 * h2)   # NOT 4.0 / h2
    w_diag_base = 1.0 / (6.0 * h2)   # NOT 1.0 / h2

# Then: identical 8-direction enumeration loop as Step 1.2 (uniform branch).
# For each cell (i, j) and each (di, dj) in MOORE_8:
#   is_cardinal = (di == 0) ^ (dj == 0)
#   w_base      = w_card_base if is_cardinal else w_diag_base
#   ... in-grid: harm-mean D_face × w_base added to off-diagonal ...
#   ... off-grid: boundary_mode dispatch (face_mirror/zero_pad/rest_pad/
#                  node_mirror_existing — face_mirror_iso comes in Phase 2)

# Verify CFL stability after construction:
#   D_eff = (w_card_base + 2*w_diag_base) * h2 * D    (in y-uniform interior)
#         = (4/6 + 2/6) * D = D    (matches cardinal-4)
#   CFL = D_eff * dt / h² = D * dt / h²   (same stability as cardinal-4, OK)
```

#### Test Spec
- `test_boundary_modes.py::test_a10_moore8_iso_recovers_continuum_in_interior` — Setup: 16x16 grid, isotropic D=1e-3, V[i,j] = sin(π·xi)·cos(π·yj). Compute L_iso·V at interior cells (i in [4, 12], j in [4, 12]). Compare to analytic ∇²V = -2π²·V at those points. Expected: relative error < 1e-2.
- `test_boundary_modes.py::test_a10_moore8_iso_y_uniform_interior_matches_cardinal` — y-uniform V; L_iso*V at interior cells matches L_cardinal*V output to 1e-12.
- `test_boundary_modes.py::test_a10_moore8_iso_boundary_deficit_5_over_6` — Setup as in 1.2's deficit test. Expected boundary deficit = 5/6 of interior charging rate within numerical tolerance.
- `test_boundary_modes.py::test_a10_cfl_stability_check` — Setup: build moore8_iso L on 41x21 grid, D=0.001, dx=0.025. Compute `D_eff = (-L).to_dense().diag().max() * dx² / D`. Expected: D_eff ≤ 0.25 (CFL safe). FAIL fast if violated. (Stronger spectral check via eigenvalue is overkill for this diagnostic; the diagonal-magnitude check catches the 6× normalization bug we hit before.)
- `test_boundary_modes.py::test_a10_all_three_stencils_construct` — Setup: 8×8 grid, isotropic D=1e-3. Build all 3 stencils (`cardinal4`, `moore8_uniform`, `moore8_iso`) with default `boundary_mode='face_mirror'`. Expected: each returns `torch.sparse_coo_tensor` of shape (64, 64); the COO is coalesced; row sums all <= 1e-12; matrix symmetric (face_mirror branch). This is the "all 3 build successfully" test deferred from Step 1.1.

#### Checklist
- [ ] Add 'iso' branch with 4/6, 1/6 weights
- [ ] Verify interior y-uniform behaviour matches cardinal4
- [ ] Verify 5/6 boundary deficit
- [ ] Add CFL diagnostic test
- [ ] Inline code comment cites IDEALOG entry documenting the 1/6 bug

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.4
conda run -n heart-conduction python test_boundary_modes.py
```

#### Exit Criteria
- [ ] All iso tests pass
- [ ] CFL test confirms D_eff ≤ 0.25

#### Risk
**Risk**: Forgetting the 1/6 prefactor (the same bug we hit on John's storage tank). — mitigation: explicit unit test `test_a10_cfl_stability_check` that fails LOUDLY if D_eff violates CFL, plus inline comment.

---

### Phase 1 Verification

```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.4
conda run -n heart-conduction python test_boundary_modes.py    # 5 -> 12+ tests
conda run -n heart-conduction python test_phase7.py            # 7/7
conda run -n heart-conduction python test_phase8.py            # 7/7
```

Quick analytical verification (manual, optional):
```bash
conda run -n heart-conduction python -c "
import sys; sys.path.insert(0, '/home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.4')
from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
g = StructuredGrid.create_rectangle(1.0, 0.5, 41, 21)
for stencil in ('cardinal4', 'moore8_uniform', 'moore8_iso'):
    fdm = FDMDiscretization(g, D=0.001, chi=1.0, Cm=1.0, stencil=stencil)
    L = fdm.L.to_dense()
    print(f'{stencil}: max|L sym err| = {(L - L.T).abs().max().item():.3e}, '
          f'row-sum max = {L.sum(dim=1).abs().max().item():.3e}')
"
```

### Phase 1 Exit Criteria

- [ ] All boundary-mode tests pass (5 original + ~5 new)
- [ ] Phase 7/8 regression tests pass (7+7)
- [ ] Three stencils each produce SPD or near-SPD operators (face_mirror gives SPD; node_mirror_existing has 1.0 asymmetry as documented in test_a3)
- [ ] Iso CFL stability verified

### Phase 1 Cleanup

- [ ] Run `grep -r 'float32' Monodomain/Engine_V5.4/cardiac_sim/simulation/classical/discretization_scheme/fdm.py` — must return nothing
- [ ] V5.3 not modified — `git diff Monodomain/Engine_V5.3/` is empty
- [ ] FDMDiscretization docstring updated to document `stencil` parameter and 3 options
- [ ] Inline comment in `_build_laplacian_moore8` cites the 1/6 prefactor + CFL diagnostic
- [ ] No code duplication — boundary-mode logic shared between cardinal and moore8 via helper if natural

**-> Commit point: git commit after Phase 1 passes**

---

## Phase 2: Diagonal-aware face_mirror reflection (face_mirror_iso BC)

**Goal**: Add a new boundary mode `face_mirror_iso` that handles diagonal off-grid cells by reflecting the y-component to the in-grid cell at the boundary row. Combined with iso 4:1 stencil, this gives boundary == interior in y-uniform fields (zero deficit). This is the LBM bounce-back trick generalised to 9-point.

**Tier**: medium
**Estimated scope**: ~80 lines of new code in fdm.py, 3 new tests

### Phase Context

The current `face_mirror` BC in `fdm.py` (Phase 1) handles cardinal off-grid by setting ghost = boundary cell value → flux = 0. For diagonal off-grid pipes (e.g., NW at (i−1, −1) for top wall), the Phase 1 implementation also gives ghost = self equivalence → flux = 0. This produces the 5/6 deficit observed in storage-tank R5.

The correct face-centered diagonal reflection is: ghost(i+di, −1) = V[i+di, 0] (mirror only the y-component, keep x-component). This makes the off-grid diagonal "fire" against the in-grid cell at the boundary row (column i+di), contributing `(V[i+di, 0] − V[i, 0])` to the Laplacian. In y-uniform fields where V[i+di, 0] = V[i+di, 1] = ..., this contribution equals what an in-grid diagonal would contribute, restoring boundary == interior.

After Phase 2, the matrix L for moore8_iso + face_mirror_iso has:
- Interior row-sum = 0
- Boundary row-sum = 0
- L*V at boundary equals L*V at interior in y-uniform fields, exactly

### Step 2.1: Add face_mirror_iso to BOUNDARY_MODES + handler
**Model**: opus

#### Read First
- `fdm.py` (Phase 1 builders for moore8) — to know where to plug the new BC
- KNOWLEDGE.md "Connectivity is the smoking gun" section — for the LBM bounce-back analogy

#### Why
This is the structural fix that makes iso 4:1 actually deliver on its theoretical promise. Without it, iso reduces deficit from 1/3 to 1/6 but doesn't eliminate it. With it, deficit goes to 0 — same as cardinal-4, but with the higher-order accuracy of 9-pt iso.

#### Implementation Spec
**Files to modify:** `fdm.py:88` (BOUNDARY_MODES), `_build_laplacian_moore8`

```python
BOUNDARY_MODES = ('face_mirror', 'face_mirror_iso', 'node_mirror_existing', 'zero_pad', 'rest_pad')
```

Add to `_build_laplacian_moore8` for off-grid diagonals when boundary_mode == 'face_mirror_iso':
```python
elif self._boundary_mode == 'face_mirror_iso':
    if is_cardinal:
        pass  # cardinal off-grid: ghost = self, contribute 0 (same as face_mirror)
    else:  # diagonal off-grid: mirror y-component to in-grid at boundary row
        ni_m = ni if 0 <= ni < nx else i  # x off-grid -> mirror to self.x
        nj_m = nj if 0 <= nj < ny else j  # y off-grid -> mirror to self.y
        if (ni_m, nj_m) == (i, j):
            pass  # corner case: both axes off-grid -> ghost = self, contribute 0
        elif _is_active(ni_m, nj_m):
            D_face = _harm(d_self, float(Dxx[ni_m, nj_m]))
            w = D_face * w_base
            center -= w
            _add(k, _idx(ni_m, nj_m), w)
```

**`face_mirror_iso` for cardinal4 stencil**: must NOT silently fall through. The cardinal4 builder's existing `if/elif` chains for off-grid handling have no `face_mirror_iso` branch, so without explicit handling the off-grid contribution would silently default to "do nothing" (equivalent to face_mirror by accident). To make the degeneracy explicit and correct, add `elif self._boundary_mode == 'face_mirror_iso': pass` to each of the 4 wall branches in `_build_laplacian_cardinal` (east, west, north, south at the `i+1 >= nx` / `i-1 < 0` / `j+1 >= ny` / `j-1 < 0` else-blocks). This makes the cardinal4 + face_mirror_iso combination behaviorally identical to cardinal4 + face_mirror, by design rather than by oversight. Add a test that asserts this equivalence.

#### Pseudocode
For diagonal direction (di != 0 AND dj != 0):
```
ni, nj = i + di, j + dj
if 0 <= ni < nx and 0 <= nj < ny:
    # In-grid diagonal — handle normally
    add diagonal pipe with weight w_diag
else:
    # Off-grid — apply boundary mode
    if face_mirror_iso:
        # Mirror only the off-grid axis back to boundary
        ni_m = ni if 0 <= ni < nx else i
        nj_m = nj if 0 <= nj < ny else j
        if (ni_m, nj_m) == (i, j):
            pass  # corner: both axes off-grid, ghost = self
        else:
            add pipe to (ni_m, nj_m) with weight w_diag
```

#### Test Spec
- `test_boundary_modes.py::test_a11_face_mirror_iso_in_iso_stencil_zero_deficit` — Setup: 8x8, isotropic D, V[i,j] = i (linear in x, constant in y). Build L with `stencil='moore8_iso'`, `boundary_mode='face_mirror_iso'`. Compute L*V at boundary cell (i, 0) and interior cell (i, 4) for i in [2, 5]. Expected: |L*V[i, 0] − L*V[i, 4]| < 1e-12 for all interior i.
- `test_boundary_modes.py::test_a12_face_mirror_iso_with_uniform_stencil` — moore8_uniform + face_mirror_iso also gives zero deficit (the mechanism is the same, just with different weights).
- `test_boundary_modes.py::test_a13_face_mirror_iso_corner_handling` — Setup: build at corner cell (0, 0). Apply L to V uniform-in-x. Expected: L*V at corner = same as interior (no asymmetry from corner ghost).
- `test_boundary_modes.py::test_a13_cardinal4_face_mirror_iso_degenerate` — Setup: build cardinal4 with `boundary_mode='face_mirror_iso'`, then with `boundary_mode='face_mirror'`. Expected: bit-identical L matrices (face_mirror_iso must explicitly degenerate to face_mirror for cardinal4, not silently break).

#### Checklist
- [ ] Add 'face_mirror_iso' to BOUNDARY_MODES
- [ ] Implement diagonal-aware reflection in `_build_laplacian_moore8`
- [ ] Validate for cardinal-only stencil (degenerates to face_mirror)
- [ ] Add 3 tests
- [ ] Update fdm.py docstring with face_mirror_iso explanation

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.4
conda run -n heart-conduction python test_boundary_modes.py
```

#### Exit Criteria
- [ ] Boundary == interior verified for y-uniform fields with iso + face_mirror_iso
- [ ] Corner handling correct
- [ ] No regressions in existing tests

#### Risk
**Risk**: Corner handling — for cell (0, 0), both NW and W and N (some of them) go off-grid. The reflection logic detects the case `(ni_m, nj_m) == (i, j)` (both axes off-grid → mirror back to self) and skips. — mitigation: explicit corner test (a13).

---

### Phase 2 Verification

```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.4
conda run -n heart-conduction python test_boundary_modes.py
conda run -n heart-conduction python test_phase7.py
conda run -n heart-conduction python test_phase8.py
```

### Phase 2 Exit Criteria

- [ ] iso + face_mirror_iso gives zero deficit in y-uniform fields (verified at 1e-12 tolerance)
- [ ] All other tests pass

### Phase 2 Cleanup

- [ ] fdm.py docstring documents face_mirror_iso
- [ ] Diagonal reflection logic is in a clearly-named helper function
- [ ] V5.3 not modified

**-> Commit point: git commit after Phase 2 passes**

---

## Phase 3: Monodomain column diagnostic — port John's ablation

**Goal**: Port the y-uniform line-stim column diagnostic from John's storage-tank ablation to monodomain V5.4. Validate predictions (qualitative ordering, not hard magnitudes — see Success Criteria for the binding criteria): cardinal4 + face_mirror → ~0 LAT shift (round-off floor), moore8_uniform + face_mirror → measurable structural deviation (LAT shift magnitude TBD; could be sub-µs depending on I_K1 V_advantage clamp), moore8_iso + face_mirror_iso → ~0 LAT shift.

**Tier**: small
**Estimated scope**: ~150 lines, mostly experiment glue

### Phase Context

The diagnostic in `Research/Active/boundary_conduction_speedup/diag_column_boundary_vs_center.py` already runs face_mirror + cardinal4 and shows V[boundary] = V[center] to 1e-13 mV. We extend it to sweep over (stencil, boundary_mode) combinations.

Caveat: monodomain TTP06 has a strong I_ion clamp on V_advantage. **Source for the ~7 mV / ~50 µs prediction**: IDEALOG.md 2026-04-29 thread "Wave-slowing dilation as the dominant apparent-curvature artifact" (~lines 618-639) — quantifies V_advantage at boundary cells under TTP06's I_K1 clamp at ~7 mV, translating to ~50 µs LAT shift over a ~135 mV/ms upstroke. The KNOWLEDGE.md "Wave-slowing dilation" section discusses the storage-tank crescent-deceleration analysis (different topic). For moore8_uniform, the V_advantage might exceed 7 mV due to threshold-amplified compounding, yielding LAT shifts barely above our save_every = 0.025 ms resolution.

### Step 3.1: Sweep diagnostic across stencils
**Model**: opus

#### Read First
- `Research/Active/boundary_conduction_speedup/diag_column_boundary_vs_center.py` — current single-config diagnostic
- `fdm.py` (after Phase 1+2) — for the new stencil/BC combos

#### Why
The diagnostic confirms the structural prediction: the boundary effect is determined by stencil connectivity, modulated by boundary handling. Five configurations cover the predicted outcomes:
1. cardinal4 + face_mirror (existing baseline) → 0 deficit
2. moore8_uniform + face_mirror → ~33% deficit, visible LAT shift
3. moore8_uniform + face_mirror_iso → 0 deficit (bounce-back fixes uniform too)
4. moore8_iso + face_mirror → ~17% deficit (Patra-Kałuża without bounce-back)
5. moore8_iso + face_mirror_iso → 0 deficit (LBM-equivalent)

#### Implementation Spec
**Files to create:** `Research/Active/boundary_conduction_speedup/diag_monodomain_connectivity.py`

```python
"""
Monodomain V5.4 column diagnostic across (stencil, boundary_mode) combinations.
Predicts: only stencils with proper boundary handling give zero LAT shift.

Save_every = 0.025 ms (40x finer than typical) to capture sub-ms shifts.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

ENGINE = Path("/home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.4")
sys.path.insert(0, str(ENGINE))

from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
from cardiac_sim.tissue_builder.stimulus.protocol import StimulusProtocol
from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
from cardiac_sim.simulation.classical.monodomain import MonodomainSimulation

# Configurations — each (stencil, boundary_mode) combination
CASES = [
    ("cardinal4_face_mirror",      "cardinal4",      "face_mirror"),
    ("moore8_uniform_face_mirror", "moore8_uniform", "face_mirror"),
    ("moore8_uniform_face_iso",    "moore8_uniform", "face_mirror_iso"),
    ("moore8_iso_face_mirror",     "moore8_iso",     "face_mirror"),
    ("moore8_iso_face_iso",        "moore8_iso",     "face_mirror_iso"),
]

# Setup matches diag_column_boundary_vs_center.py
LX, LY = 1.0, 0.5
DX = 0.025
NX, NY = 41, 21
DT = 0.02
T_END = 25.0
SAVE_EVERY = 0.025
LAT_THRESH = -40.0

def run_one(stencil, boundary_mode):
    grid = StructuredGrid.create_rectangle(LX, LY, NX, NY)
    fdm = FDMDiscretization(grid, D=0.001, chi=1.0, Cm=1.0,
                            stencil=stencil, boundary_mode=boundary_mode)
    proto = StimulusProtocol()
    proto.add_stimulus(region=lambda x, y: x < 0.05,
                       start_time=0.0, duration=2.0, amplitude=-52.0)
    sim = MonodomainSimulation(spatial=fdm, ionic_model='ttp06',
                               stimulus=proto, dt=DT, splitting='strang',
                               ionic_solver='rush_larsen',
                               diffusion_solver='forward_euler',
                               cell_type='EPI')
    times, V_hist = sim.run_to_array(t_end=T_END, save_every=SAVE_EVERY)
    return times, V_hist.reshape(len(times), NX, NY)

# Run all 5, extract V at j=0/center/Ny-1 in mid column, compute LAT/dev
# Plot 5 rows: V(t) traces for each case
# Print summary table: V_max_top, V_max_ctr, max|top-ctr|, LAT_top, LAT_ctr, ΔLAT
```

#### Pseudocode
The Implementation Spec above contains the full skeleton (CASES list + run_one() function + extraction loop). Per-case post-processing pseudocode:
```python
import numpy as np
def lat(V_t, times, threshold=-40.0):
    above = V_t >= threshold
    if not above.any():
        return float('nan')
    idx = int(np.argmax(above))
    if idx == 0:
        return times[0]
    v0, v1 = V_t[idx-1], V_t[idx]
    t0, t1 = times[idx-1], times[idx]
    return t0 + (threshold - v0) * (t1 - t0) / (v1 - v0)

# For each (label, stencil, bc) in CASES:
times, V_field = run_one(stencil, bc)
i_mid = NX // 2
V_top = V_field[:, i_mid, 0]
V_ctr = V_field[:, i_mid, NY // 2]
V_bot = V_field[:, i_mid, NY - 1]
lat_top = lat(V_top, times)
lat_ctr = lat(V_ctr, times)
delta_lat_us = (lat_top - lat_ctr) * 1000  # ms -> µs
max_dev_mV = np.abs(V_top - V_ctr).max()
print(f"{label:<32}  max|top-ctr|={max_dev_mV:.3e} mV   ΔLAT={delta_lat_us:+.4f} µs")
# Save 5-row plot of V(t) traces and dev(t) curves; one row per case.
```

#### Test Spec
- The script itself is the test. Run it and inspect output.
- Optional unit test in test_boundary_modes.py: `test_a14_iso_with_iso_bc_zero_deficit_in_monodomain` — quick analytical check via L matrix only (might already be covered by Phase 2 tests).

#### Checklist
- [ ] Create diagnostic script
- [ ] Run all 5 cases
- [ ] Confirm cardinal4_face_mirror gives ≤ 1e-13 mV deviation (matches earlier result)
- [ ] Confirm moore8_uniform_face_mirror gives >> 0 deviation
- [ ] Confirm moore8_iso_face_iso gives ≤ 1e-12 mV deviation
- [ ] Save output figure to `figures/diag_monodomain_connectivity.png`

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Research/Active/boundary_conduction_speedup
conda run -n heart-conduction python diag_monodomain_connectivity.py
```

Expected output table (qualitative split — exact magnitudes TBD empirically):
```
case                          max|V_top - V_ctr|      ΔLAT(top - ctr)
─────────────────────────────────────────────────────────────────────
cardinal4_face_mirror         ~1e-13 mV (round-off)   0 µs (round-off)
moore8_uniform_face_mirror    > 1e-9 mV               > 0 µs (magnitude TBD)
moore8_uniform_face_iso       ~1e-12 mV (round-off)   0 µs (round-off)
moore8_iso_face_mirror        > 1e-10 mV (deficit ~½) > 0 µs (≈ ½ of moore8_uniform)
moore8_iso_face_iso           ~1e-12 mV (round-off)   0 µs (round-off)
```
The empirical verification is the ORDERING: cases with proper boundary handling
(cardinal4 + any, moore8 + face_mirror_iso) cluster at the floating-point floor;
cases with deficit (moore8 + face_mirror) are clearly above noise.

#### Exit Criteria
- [ ] Predictions confirmed numerically
- [ ] Figure saved with 5 panel V(t) traces

#### Risk
**Risk 1 (CFL)**: Forward-Euler dt may be unstable for moore8 stencils since their effective diffusion magnitude differs from cardinal4. — mitigation: dt = 0.02 ms (V5.4 default) is conservative for D = 0.001 cm²/ms, dx = 0.025 cm. Verify CFL: D·dt/dx² = 0.001*0.02/0.000625 = 0.032 << 0.25. Safe even for moore8_uniform's 3× factor (= 0.096) and moore8_iso (= 0.032 same as cardinal). If unstable, reduce dt to 0.005 ms.

**Risk 2 (I_ion clamp may suppress the LAT shift below visibility — primary scientific risk)**: KNOWLEDGE.md "Wave-slowing dilation" documents that I_K1 in TTP06 clamps subthreshold V_advantage to ~7 mV regardless of where it comes from. With moore8_uniform's 2/3 deficit, the boundary cell's V_advantage may be smaller than the cardinal-4 V_max overshoot (which is itself bounded to ~7 mV) → LAT shift may be deeply sub-µs, undetectable even at save_every=0.025 ms. — mitigation: (a) accept that the test passes if `max|V_top - V_ctr|` is above the floating-point noise floor (1e-9 mV), even if LAT shift itself is undetectable; (b) for stronger amplification, optionally re-run with `g_K1` scaled down (cardiac_sim allows ionic-parameter override) to disable the clamp and confirm the deficit grows. Don't fail the phase if LAT shift is < 1 µs; the floating-point deviation IS the structural signal.

**Risk 3 (line-stim symmetry breaking)**: Strictly y-uniform line-stim in monodomain produces y-uniform V at every step (verified in `diag_column_boundary_vs_center.py`: 1e-13 mV deviation under cardinal4). Even with moore8_uniform's structural deficit, the V field may stay y-uniform unless something breaks symmetry (round-off, asymmetric corner). — mitigation: explicitly perturb the stim by ±1e-6 mV at the boundary y-rows to seed the asymmetry, OR rely on the structural mathematical asymmetry of the L matrix (which IS present at j=0 vs j=Ny//2 even before time-stepping). The L matrix asymmetry alone (test_a9 / test_a10) confirms the bridge claim regardless of whether V dynamics develop the deviation.

---

### Phase 3 Verification

```bash
conda run -n heart-conduction python diag_monodomain_connectivity.py
```

### Phase 3 Exit Criteria

- [ ] All 5 predictions confirmed
- [ ] Figure saved
- [ ] Cross-engine consistency check: monodomain results match the storage-tank R1-R6 ablation pattern qualitatively

### Phase 3 Cleanup

- [ ] Diagnostic script has docstring explaining what it tests
- [ ] Output figure committed to `figures/`

**-> Commit point: git commit after Phase 3 passes**

---

## Phase 4: LBM V1 D2Q9 customizable weights

**Goal**: Add a `weights_mode` argument to LBM V1 LBMSimulation so we can compare canonical D2Q9 (1/9 cardinals, 1/36 diagonals — IS the iso 4:1) against a "uniform D2Q9" variant (1/8 each direction, no rest particle weight). Predict: canonical gives no crescent (LBM bounce-back is the equivalent of face_mirror_iso), uniform_8 gives a crescent (matching John's R3 / R6 in the no-threshold regime).

**Tier**: medium
**Estimated scope**: ~80 lines code, 4 tests

### Phase Context

LBM D2Q9 lattice already uses the 4:1 cardinal:diagonal ratio:
- w[0] = 4/9 (rest particle)
- w[1..4] = 1/9 (cardinals)
- w[5..8] = 1/36 (diagonals)

The lattice weights enter the LBM equilibrium distribution: `f_eq[i] = w[i] · ρ`. The diffusion coefficient relates to weights and τ via `tau_from_D(D, dx, dt, cs2)` where `cs2 = 1/3` for D2Q9.

For the "uniform D2Q9" variant, w[1..8] = 1/8 each, w[0] = 0 (no rest particle). cs2 must be re-derived from the second moment: `cs2 · δ_αβ = Σ w_i · e_iα · e_iβ`.

For Moore-8 directions on a unit lattice (cardinals: e²=1, diagonals: e²=2):
- xx component: (1/8)·(2·1 + 4·1) = 6/8 = 0.75 (cardinals contribute (1)² for E, W; diagonals contribute (1)² for NE, NW, SE, SW — the e_x component squared = 1 for each)
- Wait: NE has e_x = 1, NW has e_x = -1, so e_x² = 1 each. So xx = (1/8)·(1²+1²+0²+0²+1²+1²+1²+1²) = (1/8)·6 = 0.75
- Cross-component xy: should be 0 by symmetry. (1/8)·Σ e_x·e_y. Cardinals contribute 0. Diagonals: (1)(1)+(-1)(1)+(1)(-1)+(-1)(-1) = 1-1-1+1 = 0. ✓
- So cs2 = 0.75. Verify numerically in test.

This is non-standard LBM — it breaks rotational invariance the canonical weights are designed for, but it's a controlled non-isotropy that lets us test the connectivity argument across model classes.

### Step 4.1: Implement weights_mode parameter
**Model**: opus

#### Read First
- `LBM/Engine_V1/src/lattice/base.py` — `Lattice` ABC declaring required class attributes (Q, e, w, opposite, cs2)
- `LBM/Engine_V1/src/lattice/d2q9.py` — canonical D2Q9 (template for D2Q9_uniform: same direction order, same opposite tuple, only w and cs2 differ)
- `LBM/Engine_V1/src/lattice/d2q5.py` — D2Q5 (template for the d2q5 / weights_mode rejection logic)
- `LBM/Engine_V1/src/lattice/__init__.py` — current exports (need to add D2Q9_uniform here)
- `LBM/Engine_V1/src/simulation.py:49-92` — current LBMSimulation construction (the line 73 cs2 plumbing fix lives here)
- `LBM/Engine_V1/src/diffusion.py:29-34` — `tau_from_D` signature (already accepts cs2 with default 1/3; no code change here)

#### Why
LBM D2Q9 with canonical weights IS the Patra-Kałuża iso 9-pt with bounce-back. To bridge to John's storage-tank result, we want a non-isotropic uniform variant that exhibits the boundary deficit. This validates the bridge claim across all three engines (storage tank / monodomain / LBM).

#### Implementation Spec
**Files to create:**
- `LBM/Engine_V1/src/lattice/d2q9_uniform.py`: new class `D2Q9_uniform` inheriting from `Lattice` (ABC at `LBM/Engine_V1/src/lattice/base.py`). Uses class-level constants matching the style of `D2Q9`/`D2Q5` — **NOT a dataclass**. Required attributes: `Q=9`, `cs2=0.75`, `e` (same direction order as canonical D2Q9 — copy verbatim), `w` (rest particle weight 0, 8 moving particles each 1/8), `opposite=(0, 2, 1, 4, 3, 7, 8, 5, 6)` (same as canonical D2Q9). See Pseudocode below for the exact layout.

**Files to modify:**
- `LBM/Engine_V1/src/lattice/__init__.py`: export `D2Q9_uniform` alongside existing `D2Q5`, `D2Q9`.
- `LBM/Engine_V1/src/simulation.py`:
  - Line 49 — add `weights_mode: str = 'canonical'` parameter to `LBMSimulation.__init__`.
  - Lines 63-70 — extend lattice dispatch:
    ```python
    if lattice == 'd2q5':
        if weights_mode != 'canonical':
            raise ValueError(f"weights_mode={weights_mode!r} only valid for d2q9 (d2q5 has no diagonals)")
        self.lattice = D2Q5()
        self._step_fn = lbm_step_d2q5_bgk
    elif lattice == 'd2q9':
        if weights_mode == 'canonical':
            self.lattice = D2Q9()
        elif weights_mode == 'uniform_8':
            self.lattice = D2Q9_uniform()
        else:
            raise ValueError(f"unknown weights_mode: {weights_mode!r}")
        self._step_fn = lbm_step_d2q9_bgk
    else:
        raise ValueError(f"Unknown lattice: {lattice}")
    ```
  - **Line 73 (CRITICAL FIX)** — change `tau = tau_from_D(D, dx, dt)` → `tau = tau_from_D(D, dx, dt, cs2=self.lattice.cs2)`. Currently the cs2 default is hardcoded as 1/3 in diffusion.py, which is WRONG for any non-canonical lattice (uniform_8 has cs2=0.75). This fix is required even if uniform_8 is never selected — the canonical D2Q9 also has cs2=1/3 so behavior is identical for the existing default, but the plumbing must be correct so uniform_8 produces a stable τ.
  - **MRT path scope note**: `LBMSimulation` currently always uses BGK (lines 65, 68 hard-wire `lbm_step_d2qN_bgk`). The MRT collision (`collision/mrt/`) is not exposed through `LBMSimulation`. So this Phase only needs to wire uniform_8 into BGK. **DO NOT add a speculative MRT-guard** — it would be dead code referencing a non-existent code path. When (if) MRT becomes user-selectable in `LBMSimulation`, the new `weights_mode` plumbing must be revisited; until then, no action needed.

**Files NOT modified:**
- `LBM/Engine_V1/src/diffusion.py`: `tau_from_D` already accepts `cs2` parameter with default 1/3. The bug was simulation.py not passing it through. No change needed here — the fix is in simulation.py.

#### Pseudocode
```python
# lattice/d2q9_uniform.py
"""
Non-canonical D2Q9 variant with uniform weights on the 8 moving particles.

Used to compare against canonical D2Q9 for the boundary-deficit study;
NOT a standard LBM scheme. Notes on the deviations from canonical:
  - The rest particle (i=0) has weight 0, so f[0] is driven to zero each
    step. This makes the lattice effectively 8-velocity (D2Q8-like) but
    we keep the 9-velocity index space for code-path compatibility.
  - The fourth moment Σ w_i e_iα²·e_iβ² is NOT 2·cs2²·δ_αβ as required for
    Galilean-isotropic Navier-Stokes. For pure DIFFUSION (no advection),
    only the second moment matters, so this lattice still recovers the
    heat equation correctly. For coupled NS work it would be invalid.
"""
from .base import Lattice   # NOT LatticeBase; existing class is `Lattice`


class D2Q9_uniform(Lattice):
    # Class-level constants only — match the style of D2Q9 / D2Q5 (not @dataclass).
    Q = 9
    cs2 = 0.75    # derived: Σ w_i · e_iα² = (1/8)(2·1 + 4·1) = 6/8 = 0.75 for both x and y

    # IMPORTANT: direction order MUST match canonical D2Q9 exactly so that
    # bounce_masks (simulation.py:97-105) and `opposite` indexing remain
    # consistent across both lattices. Copy the layout from D2Q9 verbatim.
    e = (
        (0, 0),     # 0: rest
        (1, 0),     # 1: east
        (-1, 0),    # 2: west
        (0, 1),     # 3: north
        (0, -1),    # 4: south
        (1, 1),     # 5: NE
        (-1, 1),    # 6: NW
        (-1, -1),   # 7: SW
        (1, -1),    # 8: SE
    )

    # Uniform weights on the 8 moving particles; rest = 0 (deliberately dead).
    w = (
        0.0,                                # rest
        1.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0,  # cardinals (E, W, N, S)
        1.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0,  # diagonals (NE, NW, SW, SE)
    )

    # MUST match canonical D2Q9.opposite for bounce-back to work correctly.
    opposite = (0, 2, 1, 4, 3, 7, 8, 5, 6)


# simulation.py changes — see Implementation Spec for the lattice dispatch.
# Critical line 73 fix: tau = tau_from_D(D, dx, dt, cs2=self.lattice.cs2)
```

#### Test Spec
- `LBM/Engine_V1/tests/test_d2q9_uniform_weights.py::test_canonical_weights_unchanged` — D2Q9() returns canonical weights (4/9 rest, 1/9 cardinals, 1/36 diagonals); cs2 = 1/3.
- `test_uniform_weights_construction` — D2Q9_uniform() returns (0.0, 1/8, ..., 1/8); cs2 = 0.75.
- `test_uniform_cs2_self_consistency` — compute second moment numerically: `M_αβ = Σ w_i · e_iα · e_iβ` for α, β ∈ {x, y}. Expected: M_xx = M_yy = 0.75 (matches lattice.cs2), M_xy = 0. Tolerance 1e-12.
- `test_uniform_fourth_moment_not_isotropic` — DOCUMENTING test (asserts the non-isotropy as a known property): `M4_αβγδ = Σ w_i · e_iα · e_iβ · e_iγ · e_iδ`. Expected: M4_xxxx != cs2² · 3 (the canonical fourth-moment requirement). Test asserts the failure to validate the docstring's claim that this lattice is diffusion-only.
- `test_tau_from_D_uses_lattice_cs2` — Setup: build LBMSimulation with `weights_mode='uniform_8'`. Expected: `self.omega == 1.0 / (0.5 + D*dt/(0.75 * dx²))` (using cs2=0.75, NOT 1/3). Tolerance 1e-12.
- `test_uniform_propagation_creates_boundary_artifact` — Setup: 10x10 grid, line-stim at x=0, BGK collision, run 200 steps with both weight modes. Measure `|V[boundary, mid_x] − V[mid_y, mid_x]|`. Expected: > 1e-6 (well above LBM round-off) with uniform_8, ~1e-12 (round-off) with canonical.
- `test_d2q5_rejects_uniform_weights_mode` — Setup: `LBMSimulation(lattice='d2q5', weights_mode='uniform_8', ...)`. Expected: ValueError mentioning d2q5 has no diagonals.

#### Checklist
- [ ] Create `LBM/Engine_V1/src/lattice/d2q9_uniform.py` with `D2Q9_uniform` class (inherits `Lattice` ABC, class-level constants — NOT a dataclass; see Pseudocode for exact layout including `opposite` tuple)
- [ ] Update `LBM/Engine_V1/src/lattice/__init__.py` to export `D2Q9_uniform`
- [ ] Add `weights_mode` parameter to `LBMSimulation.__init__` with validation (rejects bad combos with d2q5)
- [ ] **CRITICAL**: change `simulation.py:73` to pass `cs2=self.lattice.cs2` to `tau_from_D`
- [ ] Verify behavior unchanged for canonical (cs2=1/3 same as default)
- [ ] Add 7 tests in `tests/test_d2q9_uniform_weights.py` (per Test Spec above)
- [ ] Document `D2Q9_uniform` in `__init__.py` and module docstring noting (a) non-canonical, diffusion-only, (b) fourth-moment non-isotropic, (c) rest-particle weight is 0 by design

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/LBM/Engine_V1
conda run -n heart-conduction python -m pytest tests/ -v
```

#### Exit Criteria
- [ ] All LBM tests pass (existing + 7 new per the Test Spec headcount)
- [ ] Uniform variant creates measurable boundary deficit (> 1e-6 dimensionless V) in line-stim setup
- [ ] Canonical variant continues to give the existing zero-deficit behavior (regression check)

#### Risk
**Risk 1 (cs2 derivation)**: Second moment must satisfy `Σ w_i e_iα e_iβ = c_s² δ_αβ`. The plan derives cs2=0.75 for uniform_8 (verified: cardinals contribute (1/8)·(1²+1²) = 0.25 in xx, diagonals contribute (1/8)·(1²+1²+1²+1²) = 0.5 in xx, total 0.75). If miscoded, the LBM scheme diverges from the heat equation. — mitigation: `test_uniform_cs2_self_consistency` numerically verifies the second moment matches the constant.

**Risk 2 (cs2 plumbing — the original CRITICAL issue from audit)**: `simulation.py:73` currently calls `tau_from_D(D, dx, dt)` with NO cs2 argument, so it always uses default 1/3 regardless of lattice. For canonical D2Q9 this is coincidentally correct (cs2=1/3 IS the default), but the code path silently breaks for any non-canonical lattice. The fix in this Phase makes the code lattice-aware, which is the right structural change even ignoring uniform_8. — mitigation: `test_tau_from_D_uses_lattice_cs2` asserts the new code path explicitly.

**Risk 3 (literal 4/9 / 1/9 / 1/36 in sources)**: Other LBM source files might hardcode lattice constants (e.g., `compute_source_term` in `solver/rush_larsen.py` or `step.py`). — mitigation: grep for literals `4/9`, `1/9`, `1/36`, `1/3` in `LBM/Engine_V1/src/`; replace with `lattice.w[i]` / `lattice.cs2` references where they appear.

**Risk 4 (MRT-path future-compat)**: The MRT collision modules (`collision/mrt/`) use cs2 internally, but `LBMSimulation` currently always uses BGK (no user-facing collision selector). So Phase 4's BGK-only wiring is correct for current code. The risk is future: if a `collision='mrt'` parameter is added, the new `weights_mode` plumbing must be re-checked. — mitigation: code comment in `simulation.py` near the lattice-dispatch block flagging the BGK assumption + a `# TODO(weights_mode-MRT)` marker, so future MRT work doesn't silently break uniform_8.

**Risk 5 (D2Q9_uniform with w[0]=0)**: Equilibrium `f_eq = w · ρ`, so f[0] is always driven to 0 by collision. `recover_voltage(f) = f.sum(dim=0)` still works (f[0]=0 contributes 0). But streaming + BC bookkeeping for direction 0 could behave unexpectedly. — mitigation: docstring + test `test_uniform_propagation_creates_boundary_artifact` runs full step and verifies stable behavior.

---

### Phase 4 Verification

```bash
cd /home/norepinephrine/Documents/Heart-Conduction/LBM/Engine_V1
conda run -n heart-conduction python -m pytest tests/ -v
```

### Phase 4 Exit Criteria

- [ ] LBM canonical (default) gives no boundary deficit
- [ ] LBM uniform_8 gives measurable boundary deficit
- [ ] All existing LBM tests pass

### Phase 4 Cleanup

- [ ] No code duplication between canonical and uniform paths
- [ ] Document non-standard nature of uniform_8 in `lattice/d2q9_uniform.py` module docstring + `lattice/__init__.py` package note
- [ ] Verify no hardcoded `cs2 = 1/3` or literals like `4/9`, `1/9`, `1/36` anywhere in `step.py`, `collision/`, or `simulation.py` paths — replace with `lattice.w[i]` / `lattice.cs2` references

**-> Commit point: git commit after Phase 4 passes**

---

## Phase 5: Cross-engine validation figure

**Goal**: Produce a single figure showing John's storage-tank R1/R5 results next to monodomain V5.4 (cardinal4, moore8_uniform, moore8_iso) next to LBM V1 (canonical, uniform_8) under matched conditions. Demonstrates the same connectivity-mediated boundary effect across all three model classes.

**Tier**: small
**Estimated scope**: ~80 lines, no new infrastructure

### Step 5.1: Cross-engine figure
**Model**: opus

#### Read First
- `simulation/connectivity_threshold_ablation.py` — for storage-tank reference results
- `Research/Active/boundary_conduction_speedup/diag_monodomain_connectivity.py` — for monodomain results
- LBM tests from Phase 4 — for LBM example invocation

#### Why
The bridge claim ("Moore-8 connectivity is the smoking gun across all model classes") is most convincing as a single side-by-side plot. This is the deliverable for closing the research thread.

#### Implementation Spec
**Files to create:** `Research/Active/boundary_conduction_speedup/figures/connectivity_cross_engine.py`

3-row × 3-col figure with EXACT (stencil, boundary_mode) pairings — the canonical pairings that demonstrate the bridge claim:

```
Row 1: STORAGE TANK
  Col 1: R1 — moore8 (uniform) + zero_pad + threshold + gradient + one_way
         (visible crescent — full deficit)
  Col 2: R2 — cardinal4 + zero_pad + threshold + gradient + one_way
         (no crescent — cardinal-only baseline)
  Col 3: R5 — moore8_iso (4:1 normalized) + zero_pad + threshold + gradient + one_way
         (partial reduction — iso WITHOUT diagonal-aware bounce-back)

Row 2: MONODOMAIN V5.4
  Col 1: moore8_uniform + face_mirror   (visible deficit)
  Col 2: cardinal4      + face_mirror   (no deficit, baseline)
  Col 3: moore8_iso     + face_mirror_iso (no deficit — bounce-back fix)

Row 3: LBM V1
  Col 1: D2Q9 weights_mode='uniform_8' + bounce-back (visible deficit)
  Col 2: D2Q5                         + bounce-back (cardinal-only baseline)
  Col 3: D2Q9 weights_mode='canonical'+ bounce-back (canonical iso 4:1 — should give no deficit because LBM bounce-back IS the diagonal-aware reflection)
```

**Note on Row 3 Col 3 vs Row 2 Col 3**: both should show NO crescent. The row 2 case explicitly uses our newly-implemented `face_mirror_iso`; the row 3 case uses LBM's standard bounce-back which is mathematically equivalent. This intentional redundancy is the bridge proof.

**Data sourcing:**
- Row 1 (storage tank): re-run `simulation/connectivity_threshold_ablation.py` and load isochrones from `simulation/outputs/connectivity_threshold/iso_R{1,2,5}.png` (already present from prior session) OR re-execute and pull `iso` arrays directly from the result dict. PREFERRED: re-run inside the figure script for reproducibility, importing `tanks_vec` and calling `run()` directly.
- Row 2 (monodomain): re-run the 5-case sweep from `diag_monodomain_connectivity.py` (Phase 3) and load V_field at `t = T_END / 2` for the 3 selected cases.
- Row 3 (LBM): run `LBMSimulation` with each weight_mode + lattice combination, line-stim setup matching the storage-tank geometry as closely as possible (Nx, Ny, dx, dt picked to give comparable propagation distance per displayed time).

Output: `figures/connectivity_cross_engine.png` and `.pdf`. Both saved with matplotlib `bbox_inches='tight'`, dpi=200.

#### Pseudocode
```python
import sys, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SIMULATION = ROOT.parent.parent.parent / "simulation"   # storage tank
ENGINE_V54 = ROOT.parent.parent.parent / "Monodomain" / "Engine_V5.4"
LBM_V1 = ROOT.parent.parent.parent / "LBM" / "Engine_V1"

# Row 1: storage tank
sys.path.insert(0, str(SIMULATION))
from configs import GRADIENT, resolve_geometry
import tanks_vec
inlet, outlet = resolve_geometry(GRADIENT["geometry"])
tank_results = {}
for label, conn in [("R1_uniform", "moore8"), ("R2_cardinal", "cardinal4"), ("R5_iso", "moore8_iso")]:
    out = tanks_vec.run(...connectivity=conn, threshold_gate=True...)
    tank_results[label] = out["iso"]

# Row 2: monodomain V5.4
sys.path.insert(0, str(ENGINE_V54))
from cardiac_sim... import FDMDiscretization, MonodomainSimulation, StructuredGrid, StimulusProtocol
mono_results = {}
for label, stencil, bc in [
    ("uniform", "moore8_uniform", "face_mirror"),
    ("cardinal", "cardinal4",      "face_mirror"),
    ("iso_with_iso_bc", "moore8_iso", "face_mirror_iso"),
]:
    grid = StructuredGrid.create_rectangle(...)
    fdm = FDMDiscretization(grid, ..., stencil=stencil, boundary_mode=bc)
    sim = MonodomainSimulation(spatial=fdm, ...)
    times, V_hist = sim.run_to_array(t_end=..., save_every=...)
    mono_results[label] = V_hist[len(V_hist)//2]   # mid-propagation snapshot

# Row 3: LBM V1
sys.path.insert(0, str(LBM_V1))
from src import LBMSimulation, Stimulus
lbm_results = {}
for label, lattice, weights_mode in [
    ("uniform_8", "d2q9", "uniform_8"),
    ("d2q5",      "d2q5", "canonical"),
    ("canonical_d2q9", "d2q9", "canonical"),
]:
    sim = LBMSimulation(lattice=lattice, weights_mode=weights_mode, ...)
    sim.add_stimulus(...)
    times, V_history = sim.run(t_end=..., save_every=...)
    lbm_results[label] = V_history[len(V_history)//2]

# Render 3x3 figure
fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
# ... iso contours for row 1, V heatmaps for rows 2-3 ...
fig.suptitle("Connectivity-mediated boundary deficit across model classes ...")
fig.savefig(out_dir / "connectivity_cross_engine.png", dpi=200, bbox_inches='tight')
fig.savefig(out_dir / "connectivity_cross_engine.pdf", dpi=200, bbox_inches='tight')
```

#### Test Spec
Visual inspection. The figure self-validates by being internally consistent. Optional: add a `tests/test_cross_engine_figure.py` that verifies the figure script runs to completion and produces both output files (smoke test only, not numerical).

#### Checklist
- [ ] Implement the Pseudocode-skeleton script
- [ ] Run all 3 engines under the (stencil, BC) pairings spelled out in Implementation Spec
- [ ] Build 3×3 figure with row labels (Storage Tank / Monodomain V5.4 / LBM V1) and column labels (Deficit / Baseline / Bounce-back-fix)
- [ ] Save PNG + PDF

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Research/Active/boundary_conduction_speedup
conda run -n heart-conduction python figures/connectivity_cross_engine.py
ls -la figures/connectivity_cross_engine.png figures/connectivity_cross_engine.pdf
```

#### Exit Criteria
- [ ] Figure exists in both PNG and PDF formats
- [ ] Visual inspection: column 1 (deficit) has visible boundary asymmetry in all 3 rows; columns 2-3 (baseline + fix) do not. The bridge claim is established.

#### Risk
**Risk 1 (engine-comparability)**: The three engines have different time scales, dx/dt conventions, and even different state variables (storage tank: dimensionless V ∈ [0, 100]; monodomain: V ∈ [-85, +50] mV; LBM: same as monodomain). Direct visual comparison requires careful column-normalization or per-panel colorbars. — mitigation: each panel gets its own colorbar; rows use the same colormap (e.g., `inferno` for V fields, `viridis` for isochrone steps); panel titles include scale info.

**Risk 2 (data-availability)**: Phase 5 depends on outputs from Phases 1-4. If any earlier phase didn't produce the required arrays (e.g., the moore8_uniform LAT data from Phase 3, or LBM uniform_8 from Phase 4), Phase 5 fails. — mitigation: the script re-runs everything end-to-end (slower but reproducible) rather than depending on cached outputs; document expected runtime (~5 min for the 9-panel sweep).

**Risk 3 (storage-tank LBM-incomparability)**: Storage tank uses one_way pumps + threshold + zero_pad — all of which are absent in cardiac PDE. The figure makes a SHAPE comparison (where does deficit appear?), not a MAGNITUDE comparison. The user should not be surprised to see different absolute crescent sizes; the bridge claim is qualitative ordering, not quantitative match. — mitigation: figure caption explicitly says "qualitative bridge: same connectivity → same boundary-deficit shape; absolute magnitudes are model-class dependent and not compared".

---

### Phase 5 Verification + Exit + Cleanup

```bash
ls Research/Active/boundary_conduction_speedup/figures/connectivity_cross_engine.{png,pdf}
```

- [ ] Figure exists, both formats
- [ ] Figure is self-explanatory (legend, panel titles)

**-> Commit point: git commit after Phase 5 passes**

---

## Final Cleanup

- [ ] No float32 leaks anywhere in new code (`grep -r 'float32' Monodomain/Engine_V5.4/cardiac_sim LBM/Engine_V1/src` — empty)
- [ ] V5.3 not modified — `git diff Monodomain/Engine_V5.3/` empty
- [ ] No duplicated logic across `tanks_vec.py` (storage tank), `fdm.py` (monodomain), `lattice/d2q9.py` + `lattice/d2q9_uniform.py` (LBM); each has its own implementation per the engine's conventions, with cross-references in code comments
- [ ] KNOWLEDGE.md updated with the cross-engine validation result under the "Connectivity is the smoking gun" section
- [ ] IDEALOG.md gets a final thread entry summarising the closing of the bridge claim
- [ ] Archive the completed plan:
```bash
mkdir -p Research/Active/boundary_conduction_speedup/plans
cp Research/Active/boundary_conduction_speedup/PLAN.md \
   "Research/Active/boundary_conduction_speedup/plans/$(date +%Y-%m-%d)_moore8-iso-9pt-extension.md"
```
- [ ] Revert tmux pane back to WHITEBOARD.md:
```bash
tmux send-keys -t 2 C-c
sleep 0.3
tmux send-keys -t 2 'W=$(tput cols); H=""; while true; do N=$(md5sum Research/Active/boundary_conduction_speedup/WHITEBOARD.md 2>/dev/null | cut -d" " -f1); if [ "$N" != "$H" ]; then clear; glow -s .glow-style.json -w $W Research/Active/boundary_conduction_speedup/WHITEBOARD.md 2>/dev/null; H=$N; fi; sleep 1; done' Enter
```

## Mutation Log

**MUTATED 2026-04-30**: pre-execution revision based on `/audit` (20 issues, 1 critical / 5 high / 9 medium / 5 low). Fixed:

- **CRITICAL**: Phase 4 architecture entry now explicitly fixes `simulation.py:73` (`tau_from_D` was called without `cs2`, defaulting to 1/3 regardless of lattice — latent bug for non-canonical lattices). Documented in Implementation Spec, added test `test_tau_from_D_uses_lattice_cs2`.

- **HIGH 1 (file paths)**: Replaced all references to `LBM/Engine_V1/src/lattice.py` (which is actually a directory) with `LBM/Engine_V1/src/lattice/d2q9_uniform.py` (NEW) and `LBM/Engine_V1/src/lattice/__init__.py` (MOD).

- **HIGH 2 (Step 1.3 missing Pseudocode)**: Added Pseudocode section to Step 1.3 with the iso-branch weights, 1/6 prefactor justification, and CFL stability verification.

- **HIGH 3 (Step 5.1 missing Pseudocode + Risk)**: Added both sections; Pseudocode is a full skeleton script, Risk addresses engine-comparability, data-availability, and storage-tank-vs-LBM model-class incomparability.

- **HIGH 4 (Phase 3 prediction overconfident)**: Replaced "> 1 µs LAT shift" hard prediction with "ordering above floating-point noise" qualitative criterion, plus added Risk 2 documenting the I_K1 V_advantage clamp could suppress LAT shift to sub-µs and the success criterion is the structural deviation, not a specific magnitude.

- **HIGH 5 (Step 1.1 contradiction)**: Test Spec previously said "all 3 stencils build successfully" while Checklist required Moore-8 stubs that raise NotImplementedError. Resolved by deferring the all-3-stencils-build test to Step 1.3 (after the builder is implemented), keeping only NotImplementedError-raising tests in Step 1.1.

- **MEDIUM 1 (face_mirror_iso silent fall-through in cardinal4)**: Phase 2 Step 2.1 now explicitly mandates adding `elif self._boundary_mode == 'face_mirror_iso': pass` to the 4 wall branches in `_build_laplacian_cardinal`, plus test `test_a13_cardinal4_face_mirror_iso_degenerate` verifying bit-identity with face_mirror.

- **MEDIUM 2 (uniform_8 fourth moment)**: Added documentation in d2q9_uniform.py docstring noting the lattice is diffusion-only (fourth moment is not Galilean-isotropic), and a `test_uniform_fourth_moment_not_isotropic` test that documents this property.

- **MEDIUM 3 (harmonic-mean diagonal D)**: Acknowledged in Step 1.2 Risk as a convention choice; deferred since Phase 1 only supports isotropic D where the choice doesn't matter.

- **MEDIUM 4 (D2Q9_uniform dead rest particle)**: Documented in d2q9_uniform.py docstring + Risk 5 in Step 4.1.

- **MEDIUM 5 (MRT path)**: Added explicit guard rejecting `weights_mode='uniform_8'` if MRT is selected, plus Risk 4 in Step 4.1 documenting BGK-only scope.

- **MEDIUM 6 (Phase 5 storage-tank source unclear)**: Pseudocode now explicitly re-runs `tanks_vec.run()` rather than loading PNGs.

- **MEDIUM 7 (Phase 5 figure inconsistency)**: Replaced ambiguous "default for each" with explicit (stencil, boundary_mode) pairings in a 3-row × 3-col layout, with cardinal-only baseline + bounce-back-fix as columns.

- **MEDIUM 8 (Step 1.2 deficit-test setup)**: Replaced step-function setup with y-uniform x-step setup matching the success criterion.

- **MEDIUM 9 (Step 1.1 Risk wording)**: Clarified that the predicate is `Dxy is None or Dxy.abs().max().item() == 0` (the magnitude check is the real protection, not the None check).

- **LOW (Phase 1 estimate)**: Revised from 3 hours to 6-8 hours.

**MUTATED 2026-04-30 (round 5 — narrative clarification)**: user alignment check on the diagonal off-grid handling for `face_mirror`. Confirmed both schemes are already in the codebase under separate BC names: `face_mirror` is Scheme B (uniform self-mirror, faithful to John's zero_pad), `face_mirror_iso` is Scheme A (y-axis reflection for diagonals, LBM bounce-back analog). Added explicit "Two BC schemes, both delivered" section to the Objective so future-me reads the design intent correctly: Scheme B is the bridge-claim-confirming case (artifact appears in monodomain); Scheme A is the higher-order theoretical case (artifact can be hidden). Phase 3's 5-case sweep already exercises both. No code change needed — purely a framing clarification.

**MUTATED 2026-04-30 (round 4)**: final-pass audit found 1 leftover MEDIUM issue from round 1 — Phase 3 Goal line still asserted "> 1 µs LAT shift" hard prediction even though the round-1 fix had softened the Success Criteria. Updated Phase 3 Goal text to match the qualitative-ordering criterion. Round-4 audit verdict: PLAN.md ready for execution.

**MUTATED 2026-04-30 (round 3)**: re-audit found 4 contradictions where prose hadn't been updated to match the corrected pseudocode + 1 missing Pseudocode subheading. Fixed:

- **HIGH (Architecture Changes / Step 4.1 line 60)**: Removed the speculative MRT-guard requirement that contradicted Step 4.1's "DO NOT add a speculative MRT-guard" line.
- **HIGH (Step 4.1 Implementation Spec line 745)**: Replaced "new dataclass `D2Q9_uniform`...Inherits from `LatticeBase`" with "new class inheriting from `Lattice` ABC...class-level constants — NOT a dataclass"; aligned with the corrected Pseudocode.
- **MEDIUM (Step 4.1 Checklist line 839)**: Same dataclass→class fix in checklist wording.
- **MEDIUM (Step 3.1 missing Pseudocode subheading)**: Added an explicit `#### Pseudocode` block with the LAT extraction logic + per-case post-processing skeleton, restoring Step 3.1 to the 9-section structure.

**MUTATED 2026-04-30 (round 2)**: re-audit of round-1 revisions found 8/9 resolved + 11 new issues introduced (mostly from a wrong-stencil pseudocode for D2Q9_uniform). Fixed:

- **HIGH H-NEW1/H-NEW2/H-NEW3 (D2Q9_uniform pseudocode bugs)**: Rewrote Step 4.1 Pseudocode to (a) inherit from `Lattice` ABC (was `LatticeBase`), (b) use class-level constants instead of `@dataclass`, (c) match canonical D2Q9's exact direction order (rest, E, W, N, S, NE, NW, SW, SE — NOT the rest-E-N-W-S order I had originally), (d) include the required `opposite = (0, 2, 1, 4, 3, 7, 8, 5, 6)` attribute that the `Lattice` ABC requires for bounce-back to work.

- **PARTIAL: HIGH 1 (file paths)**: Fixed the 3 remaining stale `lattice.py` references in Step 4.1 Read First (now correctly cites `lattice/base.py`, `lattice/d2q9.py`, `lattice/d2q5.py`, `lattice/__init__.py`), Phase 4 Cleanup checklist (`lattice/d2q9_uniform.py` + `lattice/__init__.py` package note), and Final Cleanup (replaced `lattice.py` with `lattice/d2q9.py + lattice/d2q9_uniform.py`).

- **MEDIUM M-NEW1 (misdirected citation)**: Step 3.1 Phase Context now correctly cites IDEALOG.md 2026-04-29 thread "Wave-slowing dilation as the dominant apparent-curvature artifact" (lines ~618-639) for the ~7 mV / ~50 µs prediction, instead of the unrelated KNOWLEDGE.md "Wave-slowing dilation" section.

- **MEDIUM M-NEW3 (Architecture Changes stale test name)**: Replaced the bullet `test_a8_stencil_dispatch_validity — all 3 stencils build successfully` with the actual per-step test list grouped by Step (1.1 dispatch / 1.2 uniform / 1.3 iso / 2.1 face_mirror_iso).

- **MEDIUM M-NEW4 (Phase 4 test count inconsistency)**: Architecture Changes header now says "7 tests"; Step 4.1 Test Spec lists 7 tests; Phase 4 Exit Criteria says "7 new". Internal headcount unified.

- **MEDIUM M-NEW5 (MRT guard would be dead code)**: Removed the speculative MRT-guard requirement; replaced with a code comment + TODO marker for future MRT exposure.

(empty — populated during execution)
