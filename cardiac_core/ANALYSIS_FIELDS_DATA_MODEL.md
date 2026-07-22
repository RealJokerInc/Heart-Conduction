# Analysis Fields — Data Model & Shapes Reference

> The object hierarchy, the terminology, and **every torch object's shape** for the `analysis.fields` layer.
> Companion to [ANALYSIS_FIELDS_DESIGN.md](./ANALYSIS_FIELDS_DESIGN.md) (design + math), the
> [ANALYSIS_METHODS_PRIOR_ART.md](./ANALYSIS_METHODS_PRIOR_ART.md) (methods), and
> [ANALYSIS_FIELDS_PLAN.md](./ANALYSIS_FIELDS_PLAN.md) (implementation). Use this to remember what you're holding.
> Everything is a **`torch` tensor**, float64, on the run's device (cuda if the run was on GPU), except masks (bool)
> and `winding_number` (int). Created 2026-07-22.

## 1. Dimension vocabulary (used everywhere)

| Symbol | Meaning |
|--------|---------|
| `T` | number of saved time frames (`len(r.times)`) — the **time axis** |
| `Nx` | grid nodes along x |
| `Ny` | grid nodes along y |
| `2` | vector components on the **trailing** axis: `[...,0]` = x-component, `[...,1]` = y-component |
| `n_states` | ionic state variables per node (TTP06 ≈ 18, ORd ≈ 40) |

**Two rules generate every shape below:**
1. **Time axis (`T`) present ⇔ the quantity is per-frame** (Vm/φ_e-based). **Absent ⇔ LAT-based** — the activation-time
   map collapses time to one number per node, so everything derived from it (`velocity`/`direction`/`speed`/
   `curvature`/`vorticity`) has **no `T`**.
2. **Trailing `2` ⇔ vector**; no trailing `2` ⇔ scalar. Integrals reduce to a **number** (or a per-frame/per-node map).

Mental shortcut: **start from `(Nx, Ny)`** → prepend `T` if per-frame → append `2` if a vector → collapse to a scalar
if it's an integral reduction.

## 2. Terminology (property vs function/method vs operator)

| Term | What it is | How you use it | Example |
|------|-----------|----------------|---------|
| **property** | a value you read (attribute-style) | dotted access, **no parens** | `r.fields.velocity`, `.magnitude`, `.x` |
| **method** | a function attached to an object | **call with parens** | `r.cv(x1,x2,y)`, `r.lat()` |
| **function** | a standalone callable | **call with parens** | `grad(f, dx, dy)`, `wavelength(cv, erp)` |
| **operator** | a *math* concept (∇, ∇·, ∇×, ∇²) that maps field→field | realized in code as a **function** | `grad`=∇, `div`=∇·, `curl`=∇×, `laplacian`=∇² |

- Descend `r.fields` with dots/no-parens until you hit a **named field** (a value — done) or a **toolkit**
  (`.derivatives` / `.integrals`, whose entries are **functions** you then call).
- **`Fields`** = the accessor/container (`r.fields`) holding the named fields + the two toolkits (one per result,
  lazily cached). **`VectorField`** = the return *type* for a vector-valued field: a thin wrapper over a `(...,2)`
  torch tensor exposing `.x`/`.y`/`.magnitude`/`.angle`/`.components`. Scalar fields return a plain tensor, not a
  `VectorField`.

## 3. Object hierarchy

```
r = sim.run(...)                         →  SimulationResult
│
├─ DATA (plain attributes — read-only)
│   ├─ r.times · r.Vm · r.phi_e · r.ionic_states · r.dx · r.dy
│   └─ r.domain_mask · r.boundary_mode · r.Cm · r.chi · r.conductivity · r.ionic_model · r.cell_type
│
├─ METHODS (scalar analysis — call with parens)
│   └─ r.lat() · r.cv() · r.cv_between() · r.radial_cv() · r.apd() · r.apd_per_beat()
│       r.restitution() · r.restitution_slope() · r.df_map()
│
└─ r.fields                              →  Fields  (accessor — a property)
    ├─ NAMED FIELDS (properties — read, cached)
    │   voltage_gradient · voltage_flux · source_sink · electric_field · current_flux
    │   velocity · direction · speed · curvature · vorticity · divergence · quality · mask
    ├─ r.fields.derivatives              →  OPERATORS (functions): grad · div · curl · laplacian
    └─ r.fields.integrals                →  REDUCTIONS (functions): conduction_time · net_flux ·
        circulation · winding_number · region_integral · activated_area · wavefront_length · global_curvature

VectorField (wraps any (...,2) field):  .x · .y · .magnitude · .angle · .components
Top-level (imported / called directly): cardiac_core.single_cell() · protocols.erp() ·
                                         analysis.wavelength() · .di() · .safety_factor() · .activation_time()
```

## 4. Shapes — raw data on the result

| Object | shape | dtype | dims |
|--------|-------|-------|------|
| `r.times` | `(T,)` | f64 | one time (ms) per frame |
| `r.Vm` (`r.V` alias) | `(T, Nx, Ny)` | f64 | frame × x × y |
| `r.phi_e` | `(T, Nx, Ny)` or `None` | f64 | frame × x × y (bidomain only) |
| `r.ionic_states` | `(T, n_states, Nx, Ny)` or `None` | f64 | frame × state-var × x × y (opt-in `record=`) |
| `r.domain_mask` | `(Nx, Ny)` or `None` | bool | x × y (geometry — no time) |
| `r.conductivity.D_eff` | `(Nx, Ny)` (scalar if uniform) | f64 | per-node effective diffusivity `D_raw/(χ·Cm)` |
| `r.dx`, `r.dy`, `r.Cm`, `r.chi` | scalar | float | Python floats, NOT tensors |
| `r.boundary_mode`, `r.ionic_model`, `r.cell_type` | scalar | str | edge rule / model identity |

## 5. Shapes — derivative operators (`fields.derivatives`)

Shape transforms (the leading `...` absorbs `T` if present, or nothing for a single `(Nx,Ny)` frame):

| Call | input | output |
|------|-------|--------|
| `grad(scalar)` | `(..., Nx, Ny)` | `(..., Nx, Ny, 2)` — adds the component axis |
| `div(vector)` | `(..., Nx, Ny, 2)` | `(..., Nx, Ny)` — removes it |
| `curl(vector)` | `(..., Nx, Ny, 2)` | `(..., Nx, Ny)` |
| `laplacian(scalar)` | `(..., Nx, Ny)` | `(..., Nx, Ny)` |

## 6. Shapes — named fields

**WITH time axis** (Vm/φ_e-based, per-frame):

| Field | definition | shape |
|-------|-----------|-------|
| `r.fields.voltage_gradient` | ∇Vm | `(T, Nx, Ny, 2)` |
| `r.fields.voltage_flux` | D∇Vm | `(T, Nx, Ny, 2)` |
| `r.fields.source_sink` | ∇·(D∇Vm) | `(T, Nx, Ny)` (scalar) |
| `r.fields.electric_field` | −∇φ_e | `(T, Nx, Ny, 2)` (bidomain) |
| `r.fields.current_flux` | −σ∇φ_e | `(T, Nx, Ny, 2)` (bidomain) |

**WITHOUT time axis** (LAT-based — time already collapsed to the activation map):

| Field | definition | shape |
|-------|-----------|-------|
| `r.lat()` (the LAT map) | first-crossing activation time | `(Nx, Ny)` (NaN where unactivated) |
| `r.fields.velocity` | ∇T/\|∇T\|² | `(Nx, Ny, 2)` |
| `r.fields.direction` | n̂ = ∇T/\|∇T\| | `(Nx, Ny, 2)` |
| `r.fields.speed` | 1/\|∇T\| | `(Nx, Ny)` (scalar) |
| `r.fields.curvature` | ∇·n̂ | `(Nx, Ny)` (scalar) |
| `r.fields.vorticity` | curl(velocity) | `(Nx, Ny)` (scalar) |
| `r.fields.divergence` | ∇·n̂ (gating) | `(Nx, Ny)` (scalar) |
| `r.fields.quality` | fit residual | `(Nx, Ny)` (scalar) |
| `r.fields.mask` | \|∇T\|<floor / collision / high-residual gate | `(Nx, Ny)` (bool) |

## 7. Shapes — `VectorField` accessors

For any `(..., 2)` field (e.g. `r.fields.velocity` = `(Nx, Ny, 2)`):

| Access | shape | note |
|--------|-------|------|
| `.components` | `(..., 2)` | the raw tensor (npz-friendly; this is what's cached/stored) |
| `.x` | `(...)` | `[..., 0]` — x-component, last axis dropped |
| `.y` | `(...)` | `[..., 1]` — y-component |
| `.magnitude` | `(...)` | `norm(v, dim=-1)`; e.g. `velocity.magnitude` == `speed` |
| `.angle` | `(...)` | `atan2(y, x)` |

## 8. Shapes — integrals & scalar metrics (reductions)

| Call | shape | dims |
|------|-------|------|
| `conduction_time(a,b)`, `net_flux`, `circulation`, `region_integral`, `wavefront_length`, `global_curvature` | scalar (0-d) | one number |
| `winding_number(loop)` | scalar | **int** — count of enclosed rotors |
| `activated_area` (over all t) | `(T,)` | one area per frame (recruitment curve) |
| `safety_factor` | `(Nx, Ny)` | per-node SF map |
| `wavelength`, `di`, `erp`, `apd_at(ix,iy)` | scalar | one number |
| `apd_map`, `df_map` | `(Nx, Ny)` | per-node map |
| `phase_map(t_idx)` | `(Nx, Ny)` | phase (rad) at one frame |

## 9. How the calculations are done (storage + compute)

- **Storage:** a vector field is ONE contiguous torch tensor with components on the trailing `(...,2)` axis — never
  split into separate `vx`/`vy`, never `(2,...)` first-axis. `VectorField` is only an ergonomic shell over it.
- **Spatial ops = real-space finite-difference convolution stencils**, NOT spatial FFT (central differences /
  compact 5-point Laplacian via the staggered `div=−grad*` / Savitzky–Golay kernel for the CV gradient — see DESIGN
  § "Calculations on a uniform grid"). Reasons: honor the solver's `boundary_mode` + `domain_mask` (FFT assumes
  periodicity, dies on masks), reproduce the solver's own diffusion term for `source_sink`, and keep the CV gradient
  local + noise-robust. All run as `conv2d`/slice-subtract on device.
- **FFT is used ONLY for the intrinsically spectral, per-pixel *temporal* ops:** `dominant_frequency`/`df_map`
  (`rfft` along t) and `phase_map` (Hilbert via `fft` along t). The engine's spectral elliptic solvers are a
  separate concern (the PDE solve, not this layer).
- **curl is a CROSS pattern** (easy to flip vs divergence): `curl(F) = Dx⊛Fy − Dy⊛Fx` (x-deriv on the y-component
  minus y-deriv on the x-component); `div(F) = Dx⊛Fx + Dy⊛Fy` (straight pattern, add). Both convolutions run on the
  ghost-padded array so the boundary/mask edge rule applies.
- **Central-difference denominators are `2·dx` / `2·dy`** because the stencil spans two cells (`i−1`→`i+1`,
  distance `2·dx`); this 2nd-order (O(h²)) form falls out of the Taylor subtraction (the even-order error terms
  cancel).

## 10. Status
Reference for the DESIGNED-but-NOT-yet-built `fields` layer (implementation gated; see PLAN). Shapes reflect the
audited plan. Open naming items (`velocity`/`vorticity`/`speed` vs alternatives) tracked in DESIGN § "Open decisions".
