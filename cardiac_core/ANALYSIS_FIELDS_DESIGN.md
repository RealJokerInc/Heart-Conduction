# Design: `cardiac_core.analysis.fields` — derivatives + integrals

> Status: DESIGNED (2026-07-21, design conversation). NOT yet implemented. Separate from the
> probe feature (deferred) and from the `r.grid()`/`r.coord()` ergonomics (small, separate).

## Purpose & structure
The whole branch **operates on fields** — scalar fields (`Vm`, `phi_e`, `LAT`) or vector fields
derived from them. Because that is the shared input, `fields` is the PARENT namespace, with three
layers under it (one gradient implementation + one set of boundary rules shared across all):

- **`fields.derivatives`** — LOCAL operators, field → field: gradient, divergence, curl, Laplacian,
  computed per-snapshot (e.g. `grad Vm` at every frame → `(T, Nx, Ny, 2)`). The *verbs* / machinery.
- **`fields.derived`** — NAMED physical fields built from the operators and **cached/pre-saved**:
  flux, velocity, source–sink, E-field, curvature, vorticity. The *nouns* — what you actually
  visualize AND feed to the integrals. (Full catalog below.)
- **`fields.integrals`** — GLOBAL reductions, field → number over a region/contour: line and region
  integrals, each the Stokes/divergence-theorem PARTNER of a `derivatives` operator (built-in
  consistency check). Consume `derived` fields.

Both computed layers are torch-native and on-device (fixes the GPU gap: `front_metrics` is numpy/CPU
and crashes on a cuda tensor). Naming rule: `derivatives.*` returns fields, `derived.*` returns
(cached) fields, `integrals.*` returns numbers; **all three take a field as their first argument.**
(Naming nit to settle: `derivatives` vs `derived` read alike — candidate rename `fields.physical` or
`fields.maps` for the middle layer; see Open decisions.)

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

## `fields.derived` — named physical fields (cached / pre-saved)
The canonical derived fields worth naming and caching — the ones you visualize AND feed to
`integrals`. Each **commits to its base field + operator** (so the identity-zero trap can't happen)
and each is the physically-loaded map, not a raw operator:

| field | definition | base | meaning |
|-------|-----------|------|---------|
| `voltage_gradient` | ∇Vm | Vm | steepest-ascent of V (large at the front) |
| `source_sink` | ∇·(D∇Vm) = D∇²Vm | Vm | **electrotonic source–sink map** (the source–sink research field) |
| `flux` | D∇Vm (diffusion) / −σ∇φ_e (current) | Vm / φ_e | flux/current field; `div(flux)` = source–sink |
| `efield` | −∇φ_e | φ_e | extracellular E-field (bidomain) |
| `velocity` | ∇LAT / \|∇LAT\|² (= CV·n̂) | LAT | conduction-velocity vector field |
| `direction` | ∇LAT / \|∇LAT\| = n̂ | LAT | unit propagation direction |
| `cv_magnitude` | 1/\|∇LAT\| | LAT | front-normal speed (`front_metrics.cv_n`) |
| `curvature` | ∇·n̂ | LAT | **wavefront curvature** (`front_metrics.kappa`) |
| `vorticity` | curl(velocity) | LAT | rotation → **rotor cores** |

**Caching / pre-save (the point):** a derived field is computed ONCE and cached (lazily on first
access), because it's expensive-ish (gradients over all frames) and reused — plot `velocity` and
compute its `circulation` from the same array; take `div(flux)` for the source–sink map and its
`net-flux` integral from the same flux. Home: a lazy cache on the result, e.g. `r.fields.velocity`
memoized (invalidated if the underlying `Vm`/`LAT` changes). Optional eager mode: record chosen
derived fields alongside `Vm` during the run (heavier; connects to the deferred probe).

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
  κ = div(n̂) — i.e. it IS the LAT subset of `fields.derived` (`cv_magnitude`, `direction`,
  `curvature`), just numpy/CPU and standalone. `fit_eikonal` fits CV_n = CV0 − D·κ. Both STAY as-is
  for now.
- **Migrate later (documented intent, NOT now):** re-express `front_metrics`'s outputs as the LAT
  `fields.derived` fields on top of the torch `fields.derivatives` primitives (one boundary-aware
  gradient implementation, not a numpy one and a torch one drifting apart); keep `fit_eikonal` as a
  thin consumer of `derived.cv_magnitude` + `derived.curvature`. Do this once the primitive layer is
  proven — then front_metrics becomes a compatibility shim, not a second implementation.

## Out of scope here (separate items)
- **Probe** (point + dt-resolution recorder that these operators evaluate on for local-property
  time-series) — DEFERRED, dealt with later.
- **`r.grid(x,y)` / `r.coord(ix,iy)`** coord↔index ergonomics — small, separate, can land anytime.

## Open decisions
- Exact home/name: a `fields` SUBPACKAGE — `cardiac_core/analysis/fields/{derivatives,derived,
  integrals}.py` (`from cardiac_core.analysis.fields import ...`) vs a `Fields` facade exposing
  `.derivatives`/`.derived`/`.integrals`. Either way all three layers live under one `fields` parent.
- **Middle-layer name**: `fields.derived` reads too close to `fields.derivatives` — candidates:
  `fields.physical`, `fields.maps`, `fields.quantities`. Pick before implementing.
- **Derived-field cache**: lazy memoize on the result (`r.fields.velocity`) — decide the cache key /
  invalidation (recompute if `Vm`/`LAT`/`boundary_mode` changed; a `scale_conductance`/reset must
  clear it). Plus whether to offer an EAGER record-during-run mode (heavier; overlaps the probe).
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
