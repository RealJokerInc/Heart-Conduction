# Design: `cardiac_core.analysis.fields` — named fields (+ derivatives/integrals toolkits)

> Status: DESIGNED (2026-07-21, design conversation). NOT yet implemented. Separate from the
> probe feature (deferred) and from the `r.grid()`/`r.coord()` ergonomics (small, separate).

## Purpose & structure
The whole branch **operates on fields**. `fields` is the PARENT namespace, and the user-facing thing
it does is **hold the named, pre-saved physical fields you reach for** — one dot, plain-language,
explicit about the quantity:

    r.fields.voltage_flux      r.fields.velocity      r.fields.source_sink
    r.fields.electric_field    r.fields.curvature     r.fields.vorticity        (full catalog below)

Under that convenience surface sit two toolkits (one gradient implementation + one set of boundary
rules, shared):

- **`fields.derivatives`** — LOCAL operators, field → field: `grad`/`div`/`curl`/`laplacian`, the
  machinery the named fields are built from. Tucked away; most users never call it directly.
- **`fields.integrals`** — GLOBAL reductions, field → number over a region/contour: line and region
  integrals, each the Stokes/divergence-theorem PARTNER of a `derivatives` operator (built-in
  consistency check). Consume the named fields.

Both toolkits are torch-native and on-device (fixes the GPU gap: `front_metrics` is numpy/CPU and
crashes on a cuda tensor). **Naming rule: names say WHAT they act on** — `voltage_flux`/`voltage_
gradient` (not bare `flux`/`grad`), `current_flux` for the bidomain current, `electric_field` (not
`efield`). The common case is `r.fields.<name>` (one dot, cached); raw operators are under
`.derivatives` for power users.

## The precision principle (load-bearing)
Operators are **typed by the field they consume**, and the branch is explicit about *which* field
each named quantity differentiates — because the math is NOT interchangeable:

| call | on field | result | physical meaning |
|------|----------|--------|------------------|
| `div(grad(Vm))` = `laplacian(Vm)` | Vm | ∇²Vm | **electrotonic source–sink** (the diffusion term the solver used) |
| `curl(grad(Vm))` | Vm | **≡ 0** | vector-calculus identity — a NULL, guarded/not exposed as a metric |
| `curl(velocity_field)` | v (from LAT) | vorticity | **rotation → rotor cores** |
| `div(velocity_field)` / `div(n_hat)` | v / n̂ | κ | **wavefront curvature** (what `front_metrics` computes) |
| `grad(phi_e)` (× −1) | φ_e | E-field | current flow (bidomain) |

**Do not** offer a single ambiguous `curl(field)` — `∇×(∇V) ≡ 0` for any scalar V, so curl-of-a-
gradient must be a distinct, guarded call (or refused), while curl-of-velocity is the meaningful one.
"curl of the velocity field" and "curl of ∇V" are DIFFERENT calls with different return semantics.

## `fields.derivatives` — API (local operators, field → field)
**Primitive operators only** (the machinery; named physical fields live in `fields.derived`). Torch,
on-device, per-snapshot; accept `(Nx,Ny)` or `(T,Nx,Ny)`:
- `grad(scalar) -> vector`  (returns `(...,2)`; components ∂/∂x, ∂/∂y)
- `div(vector) -> scalar`
- `curl(vector) -> scalar`  (2-D curl = ∂vy/∂x − ∂vx/∂y, the z-component)
- `laplacian(scalar) -> scalar`  (= `div(grad(·))`)

### Vector-field representation (DECISION)
A vector field is stored as **components on the LAST axis, `(..., 2)`** — `grad(Vm (T,Nx,Ny))` →
`(T,Nx,Ny,2)`; a LAT-based `velocity` → `(Nx,Ny,2)` (no time axis). Same rule whether or not a time
axis is present (the leading `...` absorbs it). Rationale: it's what `grad` naturally returns,
`[...,0]/[...,1]` are x/y, `norm(v, dim=-1)` is the magnitude, and the integral dot-products
(`(v*n).sum(-1)` for flux/circulation) are clean. Wrap it in a light `VectorField` so users never
index a raw axis: `.x`, `.y`, `.magnitude`, `.angle`, `.components` (the raw `(...,2)` tensor).
On disk / cache: store the `(...,2)` tensor (npz-friendly). NOT chosen: separate `vx,vy` tensors or
`(2,...)` first-axis (both fight the scalar shape and broadcasting).

## Named fields — `r.fields.<name>` (pre-saved, cached)
The canonical fields worth naming — the ones you visualize AND feed to `integrals`. Names say WHAT
they act on; each **commits to its base field + operator** (so the identity-zero trap can't happen):

| `r.fields.…` | definition | base | meaning |
|--------------|-----------|------|---------|
| `voltage_gradient` | ∇Vm | Vm | steepest-ascent of V (large at the front) |
| `voltage_flux` | D∇Vm | Vm | diffusion flux of voltage; `div(voltage_flux)` = `source_sink` |
| `source_sink` | ∇·(D∇Vm) = D∇²Vm | Vm | **electrotonic source–sink map** (the source–sink research field) |
| `current_flux` | −σ∇φ_e | φ_e | current field (bidomain); `div` = current source density |
| `electric_field` | −∇φ_e | φ_e | extracellular E-field (bidomain) |
| `velocity` | ∇LAT / \|∇LAT\|² (= CV·n̂) | LAT | conduction-velocity vector field |
| `direction` | ∇LAT / \|∇LAT\| = n̂ | LAT | unit propagation direction |
| `speed` | 1/\|∇LAT\| | LAT | front-normal conduction speed (`front_metrics.cv_n`) |
| `curvature` | ∇·n̂ | LAT | **wavefront curvature** (`front_metrics.kappa`) |
| `vorticity` | curl(velocity) | LAT | rotation → **rotor cores** |

**Cached / pre-saved (the point):** each is computed ONCE and cached (lazily on first `r.fields.<name>`
access), because it's expensive-ish (gradients over all frames) and reused — plot `velocity` and
compute its `circulation` from the same array; take `voltage_flux` and its `net-flux` integral from
one flux array. Cache is invalidated if the underlying `Vm`/`LAT`/`boundary_mode` changes (a
`scale_conductance`/`reset` clears it). Optional eager mode: record chosen named fields alongside
`Vm` during the run (heavier; connects to the deferred probe).

## GATE: scrutinize LAT *before* trusting the LAT-based fields (TODO, later)
Half the named fields (`velocity`, `direction`, `speed`, `curvature`, `vorticity`) are built on the
activation-time map `LAT`, so they INHERIT every definitional choice in `LAT`. Pin + document these
first, or the fields are precise numbers on a shaky base.

**CONCRETE FINDING (2026-07-21): there are already TWO disagreeing LAT definitions.**
- `activation_time` (default; `r.lat()`): first frame where `V ≥ −20 mV`, `times[first_idx]` —
  **nearest save-point, NO interpolation** (LAT resolution = `save_every`). torch.
- `activation_time_interp`: **linearly-interpolated** crossing at `V = −40 mV`, sub-frame accurate,
  numpy/CPU — exists because eikonal `CV = 1/|∇LAT|` needs sub-frame accuracy.
They differ on **threshold (−20 vs −40)** and **accuracy (nearest-frame vs interpolated)**, so `r.lat()`
and the eikonal path yield DIFFERENT CV/curvature from the same run. Neither uses max-`dV/dt`. Before
the LAT fields, pick ONE canonical LAT (recommend: interpolated crossing, single agreed threshold,
torch/on-device) and route `r.lat()` + eikonal + the named fields all through it.

**BLAST RADIUS — the two conventions are split across the DEFAULT hooks vs the RESEARCH path:**
- `−20 mV, nearest-frame`: `r.lat()` (`activation_time`), `r.cv()` (`conduction_velocity` computes its
  OWN nearest-frame crossing at −20), and `apd_map` (uses `activation_time` as reference). = what a
  casual user gets.
- `−40 mV, interpolated`: `activation_time_interp` → `test_eikonal_metrics`, `front_metrics`, the
  **`source_sink_mismatch_investigation`** research + the **`fig4c_sourcesink`** experiments. = what
  the source–sink / curvature figures were actually made with.
So `r.cv()` (−20 nearest) ≠ the eikonal CV (−40 interp) on the SAME run — a silent, undocumented
discrepancy. Documented as a real finding in engine_consolidation KNOWLEDGE + IDEALOG.

Full scrutiny checklist:
- **Activation criterion** — threshold crossing (V > θ) vs max-`dV/dt` (upstroke) vs interpolated
  crossing. Changes `LAT`, hence CV and (especially) curvature. Confirm what `activation_time` uses.
- **Sub-frame interpolation** — `LAT` resolution is capped by `save_every` unless the crossing time
  is interpolated between frames; coarse `LAT` → noisy gradient → noisy curvature. (`activation_time_interp`
  exists — is it the default?)
- **Non-activating nodes** — scar/block never crosses → NaN; how do grad/div behave with NaN
  neighbors at a block edge? (ties to the domain_mask boundary rule.)
- **Multi-beat / re-activation** — with pacing or REENTRY a node activates many times; `LAT` = first
  crossing is ill-defined. For reentry, `LAT` breaks down → use **phase** (`phase_map` /
  `phase_singularities`), not `LAT`-based velocity/curvature. Document this limit loudly.
- **Threshold sensitivity** — CV and curvature are sensitive to θ; expose it, don't hardcode.
This is a review GATE, not a blocker for the Vm/φ_e fields (which don't touch `LAT`).

## Boundary handling — SAME as the tissue edge boundary (DECISION)
The derivative stencils MUST use the **same edge treatment as the simulation** (its `boundary_mode`,
default `face_mirror` = no-flux / Neumann mirror), so a post-hoc `∇²Vm` equals the electrotonic
source the solver actually saw — not a numpy-default one-sided edge. Consequences to implement:
- **Carry the boundary mode**: `SimulationResult` currently holds only `dx/dy/Vm/times`; add the
  `boundary_mode` (and grid/`domain_mask`) so field ops can honor it, or take it as an argument.
- **Internal boundaries too**: a scar/hole (`domain_mask`) is a no-flux edge — stencils must respect
  the mask (mirror / one-side at hole borders, NaN masked-out nodes) or divergence/curvature blows
  up at hole edges. Reuse the engine's `boundary_mode`/`stencil` convention, not a generic edge rule.

## `fields.integrals` — API (global reductions, field → number)
Global line/region integrals. Each is a Stokes/divergence-theorem partner of a `fields.derivatives`
operator, so the two tiers cross-check (see Consistency test).

### Line / contour integrals ("global curvature" family)
| quantity | integral | = (theorem) | meaning |
|----------|----------|-------------|---------|
| global curvature | `∮ κ ds` on isochrone | Gauss–Bonnet: net turning | wavefront integrated curvature |
| circulation | `∮ v · dl` around loop | `∬ curl(v) dA` (Stokes) | enclosed vorticity → **rotor** |
| conduction time | `∫ ∇LAT · dl` on path | `LAT(end) − LAT(start)` | traversal time; ÷ arc-length = path CV |
| wavefront length | `∮ ds` on isochrone | — | front perimeter (source size) |
| winding number | `∮ ∇φ · dl / 2π` | count of enclosed singularities | **# rotors** in the loop |

### Region / area ("volumetric flux") family
| quantity | integral | = (theorem) | meaning |
|----------|----------|-------------|---------|
| net flux through boundary | `∮ F · n dl` over ∂region | `∬ div(F) dA` (divergence thm) | **net source–sink inside** (source–sink balance) |
| activated area | `∬ 𝟙[V>θ] dA` | — | depolarized-area(t) + recruitment rate |
| total current / load | `∬ I_ion dA`, `∬ ∇²V dA` | — | region source / integrated electrotonic load |
| state fractions | region occupancy | — | excited / refractory fraction |

**2-D → areal** (per unit thickness, `dA`); generalizes verbatim to `dV` on a 3-D grid — the API
must NOT hardcode 2-D, but today's numbers are per-area.

### Ergonomics — regions & boundaries ARE mesh/mask objects (DECISION)
The user never hand-builds a contour or a measure — integration regions and boundaries are the
mesh/mask objects they already have (the SAME masks used for scars/stimuli). `mesh in → number out`:
- **Region integral**: `over=mask` — an `(Nx,Ny)` bool from `circle_mask`/`rectangle_mask`/
  `annulus_mask`/`domain_mask`. Default = the whole domain (`domain_mask`). Measure `dA = dx·dy`
  taken from the mesh.
- **Flux / boundary integral**: pass the SAME `region=mask` → the branch DERIVES the boundary
  `∂(mask)`, the OUTWARD normals, and the arc-length `ds` from the mesh geometry. Or
  `boundary="domain"` = the tissue's outer edge. No contour is hand-built.
- **Isochrone integrals** (global curvature, wavefront length): the contour is a `LAT` level set —
  select by `at_time=t` / `level=…`; extracted internally (marching-squares).
- Everything honors the SAME `boundary_mode` + `domain_mask` as the differential tier — a scar/hole
  edge is a real boundary, so its outward normal is included in a flux integral. Build a region
  ONCE, pass it as `over=`/`region=`, get the number.

### Orientation / sign conventions (a SPEC, not a detail)
- **Flux**: OUTWARD normal — positive = net efflux (source inside).
- **Circulation / winding**: counter-clockwise positive.
Documented AND asserted, so a source never reads as a sink and a rotor never flips its charge.

### Consistency test (free validation asset)
`∮ v·dl` vs `∬ curl(v)`, and `∮ F·n dl` vs `∬ div(F)`, must agree to discretization error. One unit
test per theorem validates BOTH tiers and the boundary handling at once.

## Relationship to existing code
- `analysis.front_metrics(lat, dx)` already computes cv_n, propagation direction (n_x,n_y), and
  κ = div(n̂) — i.e. it IS the LAT-based named fields (`speed`, `direction`, `curvature`), just
  numpy/CPU and standalone. `fit_eikonal` fits CV_n = CV0 − D·κ. Both STAY as-is for now.
- **Migrate later (documented intent, NOT now):** re-express `front_metrics`'s outputs as the
  `r.fields.{speed,direction,curvature}` named fields over the torch `fields.derivatives` primitives
  (one boundary-aware gradient implementation, not a numpy one and a torch one drifting apart); keep
  `fit_eikonal` as a thin consumer of `r.fields.speed` + `r.fields.curvature`. Do this once the
  primitive layer is proven — then front_metrics becomes a compatibility shim, not a 2nd impl.

## Adjacent analysis additions (SEPARATE TRACK — scalar EP metrics, not fields)
High-level clinical/EP metrics that COMPOSE field + point measurements into one number. Different
category from the field operators; live under top-level `analysis`, not `analysis.fields`. Wishlist:
- **`analysis.wavelength`** — the big one. **λ = CV · ERP** (the reentry master variable; `CV · APD`
  is a common proxy). Computing it by hand today is a pain: get CV, get ERP/APD, reconcile units,
  handle NaN/block. Make it one call with the choices EXPOSED: `λ = CV·ERP` vs `CV·APD`; CV local
  (at a site) vs global; which APD% / ERP definition. (ERP ≈ APD but rate-dependent — `CV·ERP` is
  the physiologically-correct reentry form; note in the ionic-optimization work λ is the master var.)
- **`analysis.apd`** — consolidate + complete: APD at % (APD90/50/30), APD restitution, per-beat.
  Partly exists (`apd_at`/`apd_map`/`apd_per_beat`) — unify + fill gaps.
- **`erp`, `di`, safety factor** — effective refractory period, diastolic interval, source–sink
  safety factor ("... or stuff").
Separate build track from the `fields` branch; captured here so it isn't lost.

## Out of scope here (separate items)
- **Probe** (point + dt-resolution recorder that these operators evaluate on for local-property
  time-series) — DEFERRED, dealt with later.
- **`r.grid(x,y)` / `r.coord(ix,iy)`** coord↔index ergonomics — small, separate, can land anytime.

## Open decisions
- Exact home/name: a `Fields` accessor on the result exposing the named fields directly
  (`r.fields.voltage_flux`) plus the `.derivatives`/`.integrals` toolkits — vs plain
  `cardiac_core/analysis/fields/` submodules. The user-facing target is `r.fields.<name>`.
- **Final field names** (adjustable): the catalog uses explicit names (`voltage_flux`, `voltage_
  gradient`, `current_flux`, `electric_field`, `speed`, …); confirm the exact set before building —
  esp. `velocity` vs `conduction_velocity`, `vorticity` vs `rotation`, `speed` vs `cv`.
- **Named-field cache**: lazy memoize on the result (`r.fields.velocity`) — decide the cache key /
  invalidation (recompute if `Vm`/`LAT`/`boundary_mode` changed; `scale_conductance`/`reset` clears
  it). Plus whether to offer an EAGER record-during-run mode (heavier; overlaps the probe).
- Whether operators default to `face_mirror` when no boundary is supplied, or require it explicitly.
- Second-order-interior vs matching the engine's exact `stencil` (`cardinal4` vs `moore8`) so
  curvature at edges is bit-consistent with the solved physics.
- **Where the mesh/boundary comes from for `.integrals`**: the result must expose `dx/dy`,
  `domain_mask`, and `boundary_mode` (the same addition the differential tier needs) so `over=`/
  `region=` can be a bare mask and the branch supplies the measure + normals. Decide: pass the mesh
  explicitly, or have `SimulationResult` carry it.
- **Contour extraction** for isochrone integrals (`∮κ ds`, wavefront length): marching-squares vs a
  co-area / level-set formulation; arc-length weighting to keep `∮κ ds` from being grid-noisy.
- **Mask-boundary normals**: deriving outward normals + `ds` from a discrete `(Nx,Ny)` mask edge
  (staircase) needs a convention (face-based vs smoothed) — pick one and pin it with the
  consistency test.
