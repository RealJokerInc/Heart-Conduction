# Design: `cardiac_core.analysis` field branch — differential + integral tiers

> Status: DESIGNED (2026-07-21, design conversation). NOT yet implemented. Separate from the
> probe feature (deferred) and from the `r.grid()`/`r.coord()` ergonomics (small, separate).

## Purpose
A dedicated analysis branch with **two tiers**, sharing one gradient implementation and one set of
boundary rules:
1. **Differential tier** (`analysis.fields`) — LOCAL operators on time-resolved fields: gradient,
   divergence, curl, Laplacian, computed on `Vm`, `phi_e`, or `LAT`, per-snapshot
   (e.g. `grad Vm` at every frame → `(T, Nx, Ny, 2)`).
2. **Integral tier** (`analysis.integrals`) — GLOBAL line/region integrals, each the Stokes/
   divergence-theorem PARTNER of a local operator (which gives a built-in consistency check).

Both torch-native and on-device (fixes the GPU gap: `front_metrics` is numpy/CPU and crashes on a
cuda tensor).

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

## API shape
1. **Primitive operators** (torch, on-device, per-snapshot; accept `(Nx,Ny)` or `(T,Nx,Ny)`):
   - `grad(scalar) -> vector`  (returns `(...,2)`; components ∂/∂x, ∂/∂y)
   - `div(vector) -> scalar`
   - `curl(vector) -> scalar`  (2-D curl = ∂vy/∂x − ∂vx/∂y, the z-component)
   - `laplacian(scalar) -> scalar`  (= `div(grad(·))`)
2. **Named physical quantities** (each commits to its source field, so the identity-zero trap can't
   happen): `voltage_gradient(Vm)`, `electrotonic_source(Vm)` = ∇²Vm, `efield(phi_e)`,
   `velocity_field(lat)`, `wavefront_curvature(lat)` = div(n̂), `vorticity(velocity_field)` =
   curl(v).

## Boundary handling — SAME as the tissue edge boundary (DECISION)
The derivative stencils MUST use the **same edge treatment as the simulation** (its `boundary_mode`,
default `face_mirror` = no-flux / Neumann mirror), so a post-hoc `∇²Vm` equals the electrotonic
source the solver actually saw — not a numpy-default one-sided edge. Consequences to implement:
- **Carry the boundary mode**: `SimulationResult` currently holds only `dx/dy/Vm/times`; add the
  `boundary_mode` (and grid/`domain_mask`) so field ops can honor it, or take it as an argument.
- **Internal boundaries too**: a scar/hole (`domain_mask`) is a no-flux edge — stencils must respect
  the mask (mirror / one-side at hole borders, NaN masked-out nodes) or divergence/curvature blows
  up at hole edges. Reuse the engine's `boundary_mode`/`stencil` convention, not a generic edge rule.

## Integral tier — `cardiac_core.analysis.integrals`
Global line/region integrals. Each is a Stokes/divergence-theorem partner of a local operator, so
the two tiers cross-check (see Consistency test).

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
  κ = div(n̂). `fit_eikonal` fits CV_n = CV0 − D·κ. These STAY as-is for now.
- **Migrate later (documented intent, NOT now):** refactor `front_metrics` / `fit_eikonal` to sit ON
  TOP of the new primitive layer, so there is ONE gradient implementation (torch, boundary-aware),
  not a numpy one and a torch one drifting apart. Do this when the primitive layer is proven.

## Out of scope here (separate items)
- **Probe** (point + dt-resolution recorder that these operators evaluate on for local-property
  time-series) — DEFERRED, dealt with later.
- **`r.grid(x,y)` / `r.coord(ix,iy)`** coord↔index ergonomics — small, separate, can land anytime.

## Open decisions
- Exact home/name: `cardiac_core/analysis/{fields,integrals}.py` (submodules) vs a `FieldAnalysis`/
  `Integrals` class pair.
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
