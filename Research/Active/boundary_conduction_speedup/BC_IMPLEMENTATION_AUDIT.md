# Boundary-Wall Handling — Cross-Engine Implementation Audit

> **Backlinks:** research question [README](./README.md) · [KNOWLEDGE](./KNOWLEDGE.md) · cross-cuts
> [engine_consolidation](../engine_consolidation/KNOWLEDGE.md) (the "API → engine mode" / API-debt theme) ·
> [MASTER.md](../../../MASTER.md)
>
> **Created:** 2026-06-30. **Type:** reference / adversarial code audit (read-only; no code changed).
> **Method:** every layer read end-to-end and each claim verified against source. The
> conceptual taxonomy (HBB/specular, face_mirror/iso, cardinal/Moore) is in
> [KNOWLEDGE.md](./KNOWLEDGE.md) §"Three BC families", §"Clean inverse-crescent BC",
> §"Curvature control: the α-blend", §"Connectivity is the smoking gun". This doc audits the
> **implementation** of those modes and **whether they are reachable from an API**.

---

## 0. TL;DR — what "boundary mode" *means* differs per engine, and almost none of it reaches the API

The single most important finding: **"boundary mode" is three different objects across the three
engines, with no shared vocabulary and no shared API.**

| Engine | "Boundary mode" = | Lives at | Validated? | Reachable via orchestrator? | Reachable via `cardiac_core`? |
|--------|-------------------|----------|------------|------------------------------|-------------------------------|
| **LBM** | population reflection rule at the wall (HBB / specular variants / α-blend) | **research script only** (`diag_lbm_specular.py`) — bypasses the engine | manual `print` checks only | **NO** — step fns hardcode `apply_neumann` | **NO** |
| **LBM** | dirichlet / absorbing | `src/boundary/` | d2q5 only (d2q9 untested) | **NO** — orphaned, never wired | **NO** |
| **Monodomain FDM** | discrete-Neumann **ghost choice** × **stencil** (`face_mirror`/`face_mirror_iso`/`node_mirror_existing`/`zero_pad`/`rest_pad` × `cardinal4`/`moore8_uniform`/`moore8_iso`) | `FDMDiscretization` | **YES** — `test_boundary_modes.py`, ~20 asserts | **NO** — orchestrator takes a pre-built `spatial` (DI), no param | **NO** — `monodomain()` builds defaults only |
| **Bidomain FDM** | Vm Neumann discretization | **fixed** = symmetric face-based | via bidomain suite | n/a (only one) | n/a |
| **Bidomain FDM** | phi_e elliptic BC (`insulated` / `bath_coupled`) — **the genuine Kléber knob** | `BoundarySpec` on the grid | **YES** | **YES** (`grid.boundary_spec`) | **YES** (`bidomain(boundary=...)`) |
| **Bidomain FDM** | stencil (`5pt` / `mehrstellen`) | `BidomainFDMDiscretization` | YES | YES (ctor arg) | **NO** (no `stencil` kwarg) |

**The only boundary knob that travels all the way to the unified public API is the bidomain phi_e
BC (`bath_coupled` vs `insulated`)** — which happens to be the one *physically convergent* effect
(mesh-convergent CV ratio → 1.131). Every single-field analog (all LBM wall rules; all monodomain
ghost/stencil choices) is stuck **below the API line**.

**Severity legend:** ⛔ critical · ⚠️ correctness/footgun · 🟡 limitation/debt · ✅ verified-correct.

---

## 1. LBM (`LBM/Engine_V1/`, vendored verbatim into `cardiac_core/_lbm/`)

### 1.1 Lattice convention — ✅ verified
`src/lattice/d2q9.py`: `e[5]=(1,1)=NE, e[6]=(-1,1)=NW, e[7]=(-1,-1)=SW, e[8]=(1,-1)=SE`;
`opposite=(0,2,1,4,3,7,8,5,6)`; weights 1/9 cardinal : 1/36 diagonal. Streaming (`streaming/d2q9.py`)
is a periodic `torch.roll`, so at the top wall the slots needing repair after streaming are exactly
the −y-incoming `f4,f7,f8`. **Every slot map in the research kernels (`f5→f8` etc.) is therefore
geometrically valid.** This matters: a uniform-field no-op test only proves *weight-matching*, not
geometric correctness, so the convention had to be checked directly.

### 1.2 The mode kernels (all in `diag_lbm_specular.py`)
Top wall (j=NY-1), non-corner cells; bottom is the y-mirror. `f*` = pre-stream (post-collision):

| Mode | Function | Rule (receiving slots) | Crescent | Status |
|------|----------|------------------------|----------|--------|
| HBB (baseline) | `apply_neumann_d2q9` | `f7←f*5 (NE→SW)`, `f8←f*6 (NW→SE)` — reverse x & y | forward (slowdown) | ✅ |
| specular-at-neighbour ("zero") | `apply_specular_top_bottom_d2q9` | `f8[i+1]←f*5[i]`, `f7[i-1]←f*6[i]` — flip y, displace 1 cell | zero bias | ✅ (HBB corner fallback) |
| same-cell specular | (α-blend at α=0) | `f8←f*5 (NE→SE)`, `f7←f*6 (NW→SW)` — flip y, keep x | inverse (speedup) | ✅ |
| **α-blend** | `apply_combined_top_bottom_d2q9(α)` | `f7 = α·f*5 + (1−α)·f*6`; `f8 = α·f*6 + (1−α)·f*5` | α=1 forward → α≈0.91 flat → α→0 inverse | ✅ |
| αβγ "weighted" simplex | `apply_weighted_top_bottom_d2q9` | α→HBB, β→neighbour-specular, γ→horizontal-cardinal | deprecated | ⚠️ see 1.5 |
| horizontal redirect (+ 4 variants) | `apply_horizontal_*` | diagonal mass → adjacent **cardinal** slot | inverse (ARTIFACT) | ⚠️ wall pre-charge |

**The α-blend is correct — ✅ verified both endpoints + invariants.** At α=1 it is exactly HBB; at
α=0 it is exactly the clean same-cell-specular rule (`f8←NE`, `f7←NW`). Both endpoints map
diagonal→diagonal (1/36→1/36), so the blend is **rest-neutral at every α** (`α·V/36 + (1−α)·V/36 =
V/36 = feq`) and **mass-conserving** (`Σ_in = f*5+f*6 = Σ_out` since `α+(1−α)=1`). Bottom wall is the
correct y-mirror. This is the function the [KNOWLEDGE α-sweep](./KNOWLEDGE.md) (curvature knob, flat
crossover α≈0.91) is built on, and the code matches the spec exactly.

### 1.3 ⛔ Headline — there is no "API → engine mode" path
- **Engine has no BC selector.** `LBMSimulation.__init__` (`src/simulation.py:55`) exposes
  `lattice`, `weights_mode`, `collision`, `bounce_masks` — **no `boundary`/`bc_mode`**. All three
  step fns (`src/step.py:20,32,44`) **hardcode `apply_neumann` (HBB)**.
- **dirichlet & absorbing are orphaned.** Implemented in `src/boundary/`, imported by **nothing** in
  the runtime path; `src/boundary/__init__.py` is **0 bytes**. `test_phase4` exercises only the
  *d2q5* variants — `apply_dirichlet_d2q9` / `apply_absorbing_d2q9` are imported-but-never-called.
  (Both are internally correct anti-bounce-back / equilibrium kernels — just unreachable.)
- **HBB-as-selectable, specular, combined, weighted, horizontal** live **only** in
  `diag_lbm_specular.py` / `run_oblique_wall_incidence.py`, which *bypass* `LBMSimulation.step()` —
  they import `bgk_collide`/`stream_d2q9`/`apply_neumann`/`recover_voltage` as loose functions and
  hand-roll the loop (`lbm_step_specular`, `lbm_step_combined`, …).
- **cardiac_core inherited the limitation verbatim.** `cardiac_core/_lbm/step.py` has **identical**
  hardcoded-neumann wiring (diff confirmed); `cardiac_core.lbm()` (`api.py:1292`) exposes only
  `lattice` — no boundary parameter. (The `boundary=` arg at `api.py:1146` is the bidomain/monodomain
  `BoundarySpec`, unrelated to LBM walls.)

**Consequence:** the entire LBM BC research program is un-productized — reproducible only by running
the specific research scripts, which is where the next findings bite.

### 1.4 ⛔ `--bc specular` means **opposite things** in the two entry scripts
- `diag_lbm_specular.py --bc specular` → `apply_specular_top_bottom_d2q9` = **neighbour-cell** = **ZERO** bias.
- `run_oblique_wall_incidence.py --bc specular` → `apply_combined_top_bottom_d2q9(α)` = **same-cell** = **INVERSE** (and there `--bc zero` is the neighbour one).

Both scripts claim the kernels are "copied VERBATIM," yet bind them to rotated flag names. The
`caseN_..._specular_natural.h5` outputs (diag) are the **zero-bias** BC; an "oblique specular" run is
the **inverse** BC. Cross-reading them silently compares two different boundary conditions.

### 1.5 ⚠️ `weighted` (αβγ) simplex = deprecated *artifact* path, still runnable
`apply_weighted_top_bottom_d2q9` routes a γ share of diagonal mass (1/36) into **cardinal** slots
(1/9). For **γ>0** it is **not rest-neutral**: at uniform field `f7=(1−γ)·V/36 ≠ V/36`, leaving a
non-equilibrium distribution the next collision pumps — the +18 mV wall pre-charge artifact
[KNOWLEDGE](./KNOWLEDGE.md) documents and explicitly replaced with the 2-vertex α-blend. The
`(α,β,0)` sub-family (HBB↔neighbour-specular) is fine; only γ>0 breaks it. Its β vertex is
*neighbour*-specular while the α-blend's β endpoint is *same-cell* — two incompatible "blend" axes
sharing the word "specular".

### 1.6 🟡 Other LBM findings
- **Geometry restriction.** Same-cell specular is hardcoded to axis-aligned top/bottom walls;
  `audit_specular_every_surface.py` shows strict eligibility selects ~0 % of slanted-wall cells →
  curved/oblique walls **silently fall back to HBB** (no speedup on the slanted segments). East/west
  walls and all 4 corners are always HBB.
- **Redundant double-apply.** Every custom step runs full `apply_neumann` (HBB on *all* walls) then
  overwrites the y-wall diagonals — wasteful and relies on an unchecked invariant (HBB happens not to
  touch the y-wall cardinals `f1/f2`, so the horizontal `+=` lands on a valid value).
- **Zero engine-level tests** for specular/combined/weighted/horizontal.

### 1.7 Scientific caveat (do not adjudicate — flag at any API surface)
[KNOWLEDGE §τ-correction](./KNOWLEDGE.md): the same-cell-specular inverse crescent has **no
dt-converged limit** (C ∝ −1/dt, diverges as dt→0 → numerical-artifact-by-definition for *that*
vertex; HBB converges to +60 µs, neighbour-specular ≡0). Whether it is a real resolution-dependent
effect or an artifact is **DISPUTED/OPEN (PI vs Claude, 2026-06-24)**. The α-blend is explicitly a
*phenomenological curvature descriptor*, parked from the artifact-vs-physical question.

---

## 2. Monodomain FDM (`FDMDiscretization`) — "iso mirror vs mirror", "cardinal vs Moore"

This is the **inverse of the LBM situation**: the boundary modes are first-class, validated,
**well-tested** constructor parameters — they just don't reach any higher-level API.

Code: `Monodomain/Engine_V5.5/cardiac_sim/.../discretization_scheme/fdm.py` (vendored identically into
`cardiac_core/_monodomain/...`). Tests: `Monodomain/Engine_V5.5/test_boundary_modes.py`
(test_a3 … test_a13).

### 2.1 The two axes
```
FDMDiscretization(grid, D, chi, Cm, boundary_mode=..., stencil=..., pad_value=...)

  boundary_mode ∈ {'face_mirror'(default), 'face_mirror_iso', 'node_mirror_existing',
                   'zero_pad', 'rest_pad'}            # the discrete-Neumann ghost choice
  stencil       ∈ {'cardinal4'(default), 'moore8_uniform', 'moore8_iso'}   # the neighbour set
```
Both are validated in `__init__` (`fdm.py:123,128`) → `ValueError` on bad input (test_a6, test_a8).

**boundary_mode** — the off-grid **ghost** value (what sits at the wall plane):

| Mode | Ghost rule | Effect | Wall placement |
|------|-----------|--------|----------------|
| `face_mirror` (default since 2026-04-29) | ghost = the boundary cell itself (`V[i,-1]=V[i,0]`) | off-grid flux ≡ 0 → genuine no-flux Neumann | face-centered (y=−h/2) |
| `face_mirror_iso` (2026-04-30) | cardinals = self (as face_mirror); **diagonals**: mirror only the off-grid axis to the in-grid boundary-row cell (`ghost(i+1,−1)=V[i+1,0]`) | re-injects the boundary-row neighbour → **zero** y-deficit | face-centered |
| `node_mirror_existing` (LEGACY) | geometric reflection across the wall plane (`V[i,-1]=V[i,1]`) | combines with the interior cardinal → **2w** → amplifies wall gradient 2× | node-centered (wall AT the node) |
| `zero_pad` | ghost = 0 | Dirichlet-to-zero outside (current sink) | — |
| `rest_pad` | ghost = `pad_value` (e.g. V_rest) | Dirichlet-to-constant; handled via V-shift in `apply_diffusion` | — |

**stencil** — the neighbour set & weights (the "cardinal vs Moore" axis):

| Stencil | Neighbours | Weights | Anisotropy | Notes |
|---------|-----------|---------|------------|-------|
| `cardinal4` (default) | 4 cardinals (harmonic-mean faces) + 4 diagonals **carrying only the Dxy cross-term** | `cx=1/dx²`, `cy=1/dy²`, `cxy=1/(4 dx dy)` | full | the legacy 5-pt-+cross stencil |
| `moore8_uniform` | all 8 | `w_card=w_diag=1/(3h²)` | isotropic only | raises `NotImplementedError` if Dxy≠0 or dx≠dy |
| `moore8_iso` | all 8 | Patra-Kałuża 4:1: `w_card=4/(6h²)`, `w_diag=1/(6h²)` | isotropic only | 4th-order accurate; **1/6 normalisation is mandatory (CFL)** |

### 2.2 The key result: mirror-vs-iso-mirror only matters **with diagonals** — ✅ verified
The boundary deficit (the "crescent" / forward slowdown at a wall) is a **diagonal-connectivity**
phenomenon. With no diagonal pipes there is nothing to starve:

| stencil × boundary_mode | y-uniform boundary deficit (boundary/interior charging) | Tested |
|--------------------------|---------------------------------------------------------|--------|
| `cardinal4` × any mirror | **1.0** (no deficit — no isotropic diagonals to lose) | implicit |
| `cardinal4` × `face_mirror_iso` | **bit-identical to `face_mirror`** (degenerate) | test_a13 (`diff==0`) |
| `moore8_uniform` × `face_mirror` | **2/3** | test_a9 (`ratio==2/3`) |
| `moore8_iso` (4:1) × `face_mirror` | **5/6** | test_a10 (`ratio==5/6`) |
| `moore8_*` × `face_mirror_iso` | **0 deficit** (boundary==interior to 1e-12) | test_a11, test_a12 |

So **"iso mirror vs mirror" is a no-op on the default `cardinal4` stencil** (🟡 footgun: a user who
sets `face_mirror_iso` expecting a change on the default stencil gets nothing — silent, though
code-documented). It is meaningful **only** with Moore-8. And `node_mirror_existing` is the
*intentional artifact* (2× wall-gradient amplification = the storage-tank "camel-toe"/crescent); the
default was correctly flipped from it to `face_mirror` on 2026-04-29 (test_a5).

### 2.3 ⛔/🟡 API → engine exposure (the same gap as LBM, one layer up)
- `boundary_mode` + `stencil` exist and are **well-tested at the `FDMDiscretization` layer** — this
  is genuinely good engineering (symmetry, hand-verified stencil values, deficit ratios, 1/6
  normalisation, corner handling, degeneration all asserted). ✅
- **But the orchestrator does not surface them.** `MonodomainSimulation.__init__(spatial, …)`
  (`monodomain.py:254`) takes a **pre-built** `SpatialDiscretization` (dependency injection) — there
  is **no `boundary_mode`/`stencil` parameter**. The string-config factories build ionic/diffusion/
  splitting solvers, never the discretization.
- **`cardiac_core.monodomain()` builds `FDMDiscretization` with defaults only** (`api.py:1093,1106`
  — no `boundary_mode`/`stencil` passed) → always `face_mirror` + `cardinal4`.
- **Net:** to use `moore8_iso` or `face_mirror_iso`, a user must hand-construct `FDMDiscretization`
  and inject it into `MonodomainSimulation` directly, **bypassing cardiac_core**. The unified API
  offers no path. (Same theme as the LBM gap and the engine_consolidation `create_cardiac_mesh`
  API-debt finding.)

### 2.4 🟡 Correctness limitations (cardinal4)
- **Boundary cross-derivative is dropped.** In `_build_laplacian_cardinal` (`fdm.py:668-695`) the
  diagonal entries carry **only** the Dxy cross-term, and at the rectangle boundary the diagonal
  ghosts are **omitted entirely** — no `boundary_mode` branch (docstring: "acceptable: Dxy is a small
  correction term"). So for **anisotropic** boundary studies (the README open criterion *"Anisotropic
  boundary study — fiber-parallel vs perpendicular"*), the boundary cross-flux is silently zeroed,
  independent of `boundary_mode`. This can bias fiber-oblique boundary CV. (Bidomain has the same drop
  — see 3.4.)
- **Pure-Python assembly.** Both builders are per-cell × per-neighbour Python loops emitting COO —
  O(Nx·Ny·{4,8}) Python iterations. Fine for research grids; slow at scale.

---

## 3. Bidomain FDM (`BidomainFDMDiscretization`) — symmetric face-based + the real Kléber knob

Code: `Bidomain/Engine_V1/cardiac_sim/.../discretization/fdm.py` (vendored into
`cardiac_core/_bidomain/...`).

### 3.1 ✅ Vm Neumann is **fixed** — symmetric face-based, by necessity
Bidomain does **not** offer a mirror/iso/Moore choice for the Vm field. It uses one stencil: a
**symmetric face-based** Laplacian where each interior face contributes equally to both adjacent
nodes and out-of-domain faces are skipped (`fdm.py:477-504`, the `if _is_active(...)` with no `else`)
→ zero-flux Neumann, **symmetric, zero row sum**.

The design rationale is sound and documented (`fdm.py:7-27`): the monodomain
`node_mirror_existing` 2w-asymmetry is *harmless* there (operator `A = χCm/dt·I − θL` is
identity-dominated, ratio ~10⁶), but the bidomain **elliptic** operator `A_ellip = −(L_i+L_e)` has
**no identity term**, so an asymmetric L would be non-SPD and PCG would fail. Hence the symmetric
face-based stencil is *required*. Trade-off: boundary nodes get half the strong-form stiffness (half
control volume) — variationally correct, and it cancels in the elliptic coupling.

### 3.2 ✅ The genuine boundary mode = phi_e elliptic BC (`BoundarySpec`)
The actual bidomain "boundary mode" is the **extracellular** BC, carried by the grid's `BoundarySpec`
(`cardiac_core/mesh/boundary.py`), applied **only** in `get_elliptic_operator()` via symmetric
row+column elimination (`_enforce_dirichlet`, `fdm.py:331`) — **not** baked into L_i/L_e:

| `BoundarySpec` | phi_e BC | Physics |
|----------------|----------|---------|
| `insulated()` | all-Neumann phi_e → `phi_e_has_null_space=True` | floating extracellular reference; no boundary speedup |
| `bath_coupled(bath_value=0.0)` | Dirichlet phi_e at the bath surface | grounds φₑ = the extracellular short-circuit → **the genuine Kléber speedup** |

**This is the one boundary knob exposed end-to-end:** `grid.boundary_spec` (engine) →
`cardiac_core.bidomain(boundary='bath_coupled'|'insulated')` (`api.py:1188-1202`). It is also the
**only physically convergent** boundary effect in the whole audit — mesh-convergent CV ratio → 1.131
(README, validated). The mono/LBM single-field crescents are discretization-connectivity *analogs*
of this (the confound the research question studies), not the same object.

### 3.3 🟡 Stencil options & cross-engine inconsistency
Bidomain offers `stencil ∈ {'5pt'(default), 'mehrstellen'}` (`fdm.py:148,159`). `mehrstellen` is an
isotropic 9-point built by tensor products of 1-D face-based Neumann Laplacians (`fdm.py:551`) →
spectral-compatible (DCT/DST diagonalises it), requires dx==dy + isotropic D.
- **Not exposed by `cardiac_core`** (no `stencil` kwarg on `bidomain()`) — same API gap (minor).
- **Cross-engine drift:** the two FDM engines use **different isotropic-9-pt stencils** — monodomain
  has `moore8_uniform`/`moore8_iso`, bidomain has `mehrstellen`. A consolidation concern: the
  "9-point isotropic" concept is implemented twice, differently, with no shared code or naming.

### 3.4 🟡 Same boundary cross-derivative drop as monodomain
`_build_laplacian` (`fdm.py:512-534`) includes the Dxy diagonal cross-terms but **skips them at the
boundary** (the diagonal `if _is_active(...)` has no boundary branch) — identical limitation to
monodomain cardinal4 (2.4). Relevant to anisotropic boundary work in **both** FDM engines.

---

## 4. Cross-engine synthesis

### 4.1 The PDE ↔ LBM taxonomy is real and now **code-verified**
[KNOWLEDGE §"Three BC families"](./KNOWLEDGE.md) maps the two engines' bookkeeping to the same
physical families. Confirmed against source + tests:

```
  Family                    LBM realization          PDE (monodomain FDM) realization        Deficit (bdry/interior)
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Forward crescent (slow)   HBB                       face_mirror  (on Moore-8)               uniform 2/3 · iso 5/6
  Zero bias (transparent)   specular-at-neighbour     face_mirror_iso (on Moore-8)            0
  Inverse crescent (fast)   same-cell specular        [NO PDE analog yet — open]              < 1 (speedup)
  No deficit (no diagonals) D2Q5                      cardinal4 (any mirror)                  1.0
```
The deficit numbers match across engines: LBM `uniform_8 + HBB` → 2/3 and `canonical(4:1) + HBB` →
5/6 (KNOWLEDGE) === monodomain `moore8_uniform + face_mirror` → 2/3 (test_a9) and `moore8_iso +
face_mirror` → 5/6 (test_a10). The diagonal **weight ratio sets the deficit**, in both engines.

**Open item (README criterion):** the inverse-crescent (same-cell specular) has **no PDE analog**.
A "face stencil preserving the tangential gradient" is the missing monodomain construction — and
since the LBM version is dt-divergent (§1.7), building the PDE analog is also the cleanest way to
settle the artifact-vs-physical dispute (a convergent FDM crescent would refute the artifact reading;
a dt/diffusion-number-dependent one would confirm it).

### 4.2 The API → engine exposure gap is the unifying defect
Reading down the TL;DR table: the boundary physics is **richest and best-tested in monodomain**,
**most physically meaningful in bidomain**, and **most experimentally explored in LBM** — yet only the
bidomain phi_e BC reaches `cardiac_core`. The mono ghost/stencil matrix and *all* LBM wall rules are
research-/engine-internal. This is the same shape as the engine_consolidation API-debt findings: the
unified construction API under-exposes engine capability.

### 4.3 Vocabulary collisions to fix
- LBM `--bc specular` = two different rules across two scripts (§1.4).
- "specular" spans neighbour-cell (zero) **and** same-cell (inverse) (§1.2, §1.5).
- "9-point isotropic" = monodomain `moore8_*` **and** bidomain `mehrstellen` (§3.3).
- "mirror" spans face-centered (`face_mirror`) and node-centered (`node_mirror_existing`) — opposite
  sign behaviour (§2.2).

---

## 5. Consolidated findings & recommendations

### Findings (by severity)
- ⛔ **LBM:** no API→engine boundary-mode path at all; modes are research-script-only; `--bc specular`
  collides across scripts (§1.3, §1.4).
- ⛔/🟡 **Monodomain:** modes are first-class + well-tested at `FDMDiscretization` but **not** surfaced
  by the orchestrator or `cardiac_core` (defaults only) (§2.3).
- ⚠️ **LBM:** `weighted` αβγ simplex is the deprecated, rest-noop-violating (γ>0) artifact path, still
  runnable (§1.5).
- ⚠️ **Monodomain:** `face_mirror_iso` is a silent no-op on the default `cardinal4` stencil (§2.2).
- 🟡 **Both FDM engines:** boundary cross-derivative (Dxy) is silently dropped at the wall → biases
  anisotropic boundary studies (§2.4, §3.4).
- 🟡 **Consolidation drift:** two different isotropic-9-pt stencils (`moore8_*` vs `mehrstellen`); no
  shared boundary vocabulary across engines (§3.3, §4.3).
- ✅ **Bidomain** Vm symmetric face-based stencil + phi_e `BoundarySpec` exposure is the **model** for
  how the others should be done (correct, justified, exposed end-to-end) (§3.1, §3.2).
- ✅ **The α-blend, HBB, specular variants, and all mono ghost/stencil kernels are individually
  correct** against the spec (§1.2, §2.2).

### Recommendations (priority order)
1. **Unify "boundary mode" as a first-class, engine-spanning concept** on the unified API. Bidomain's
   `BoundarySpec` exposure is the template. Surface monodomain `boundary_mode`/`stencil` on
   `cardiac_core.monodomain(...)`; lift the LBM wall rules into the engine (a `boundary=` selector +
   registry in `src/boundary`, dispatched in `step.py` instead of the hardcoded `apply_neumann`), then
   onto `cardiac_core.lbm(...)`.
2. **Fix the vocabulary collisions** (§4.3): one canonical name per rule — `hbb`,
   `specular_nextcell`(≡zero), `specular_samecell`(≡inverse), `combined(α)`; reconcile
   `moore8`/`mehrstellen` naming.
3. **Retire / hard-gate** the LBM `weighted`+`horizontal*` artifact paths (§1.5).
4. **Promote LBM BC tests into the engine suite** (rest no-op + mass conservation per mode), matching
   the monodomain `test_boundary_modes.py` standard; wire the dormant LBM `*_d2q9` dirichlet/absorbing
   tests.
5. **Decide the boundary cross-derivative drop** (§2.4/§3.4) before the anisotropic-boundary criterion.
6. **Build the PDE analog of same-cell specular** (§4.1) — the open README item and the cleanest path
   to settle the artifact-vs-physical dispute.
7. Carry the **flat-wall-only + dt/τ-divergence** caveats wherever the LBM α-blend is exposed (§1.7).

---

## Appendix — files read for this audit

**LBM:** `src/lattice/d2q9.py`, `src/streaming/d2q9.py`, `src/boundary/{neumann,dirichlet,absorbing,masks,__init__}.py`,
`src/step.py`, `src/simulation.py`, `diag_lbm_specular.py`, `run_oblique_wall_incidence.py`,
`audit_specular_every_surface.py`, `tests/test_phase4.py`, `cardiac_core/_lbm/step.py`, `cardiac_core/api.py::lbm`.
**Monodomain:** `Engine_V5.5/.../discretization_scheme/fdm.py`, `.../classical/monodomain.py`,
`test_boundary_modes.py`, `cardiac_core/api.py` (mono construction).
**Bidomain:** `Engine_V1/.../discretization/fdm.py`, `cardiac_core/mesh/boundary.py`, `cardiac_core/api.py::bidomain`.
**Spec:** `boundary_conduction_speedup/KNOWLEDGE.md` §§ "Three BC families", "Clean inverse-crescent BC",
"Curvature control: the α-blend", "CORRECTION — control parameter is τ", "Connectivity is the smoking gun".
