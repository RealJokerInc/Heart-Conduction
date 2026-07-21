# Design: `cardiac_core.analysis.fields` — spatial differential-operator branch

> Status: DESIGNED (2026-07-21, design conversation). NOT yet implemented. Separate from the
> probe feature (deferred) and from the `r.grid()`/`r.coord()` ergonomics (small, separate).

## Purpose
A dedicated analysis branch for **spatial differential operators on time-resolved fields** —
gradient, divergence, curl, Laplacian — computed on `Vm`, `phi_e`, or `LAT`. Torch-native and
on-device (fixes the GPU gap: `front_metrics` is numpy/CPU and crashes on a cuda tensor). Computes
per-snapshot by construction (e.g. `grad Vm` at every frame → `(T, Nx, Ny, 2)`).

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
- Exact home/name: `cardiac_core/analysis/fields.py` (submodule) vs a `FieldAnalysis` class.
- Whether operators default to `face_mirror` when no boundary is supplied, or require it explicitly.
- Second-order-interior vs matching the engine's exact `stencil` (`cardinal4` vs `moore8`) so
  curvature at edges is bit-consistent with the solved physics.
