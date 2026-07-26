# `cardiac_core` — API Reference

> **Scope.** This is the reference for the unified, engine-agnostic simulation API (North-Star Goal 1) —
> the target public surface of `cardiac_core`. It is written as library documentation: every public
> class and function, its signature, parameters, return value, and a usage example.
>
> **Companion docs.** `GLOSSARY.md` (why each name was chosen) · `API_DESIGN.md` (rationale + decisions).
> All names here are the canonical glossary names.
>
> **Conventions used throughout**
> - Transmembrane potential is **`Vm`**, grid-shaped **`(Nx, Ny)`**, `torch.Tensor` **float64**.
>   Time series are **`(T, Nx, Ny)`**.
> - Geometry is a **structured grid** (FEM/unstructured dropped, 2026-06-24) — `(Nx, Ny)` `ij` indexing.
> - All physical scaling (`sigma → D`, `chi`, `Cm`) is confined to `ConductivityConfig`.
> - Negative stimulus amplitude = depolarizing. Default `-52.0`. Stimuli accumulate (`+=`).
>
> **Implementation-status legend** (the API is partly built, partly designed):
> `[now]` exists in `cardiac_core` today (wrapper level) · `[P3]`/`[P4]` lands in that consolidation
> phase · `[design]` specified here, not yet coded.

---

## Quickstart

```python
from cardiac_core import monodomain, Grid, ConductivityConfig, TTP06Model, CellType, Stim

g = Grid(Nx=200, Ny=50, dx=0.01)                                   # structured grid
sim = monodomain(
    g,
    TTP06Model(cell_type=CellType.EPI),                            # ionic model
    ConductivityConfig.bidomain(sigma_i=1.74, sigma_e=6.25),       # physics -> D_eff inside
    stimulus=Stim.boundary(g, "left", start_time=1.0),             # canonical Stim (dicts still work)
)
result = sim.run(t_end=50.0)        # eager -> SimulationResult
print(result.cv())                  # ~54 cm/s

# parameter sweep — functional, original untouched
faster = sim.with_(conductivity=ConductivityConfig.isotropic(sigma=2.0))
```

---

## Module map

| Symbol | Kind | Purpose |
|---|---|---|
| `Grid` | class | structured-grid geometry descriptor |
| `ConductivityConfig` | class | physics → diffusivity; the χ/Cm + Formulation-A/B firewall |
| `IonicModel` / `TTP06Model` / `ORdModel` / `CellType` | class / enum | cell electrophysiology (shared ABC) |
| `Stim` | class | **canonical** stimulus — presets + current/clamp modes (lowers to the dict/protocol) |
| `Stimulus` / `StimulusProtocol` | class | lower-level pacing events + protocols `Stim` lowers onto |
| region helpers (`left_edge_region`, `circular_region`, …) | function | build stimulus regions |
| `monodomain` / `bidomain` / `lbm` | factory fn | DECLARE a `Simulation` |
| `Simulation` | protocol | the runtime object: `run` / `step` / `with_` / `stimulate` |
| `SimulationResult` | class | the only public output (times + `Vm` + analysis hooks) |
| `BoundarySpec` | class | bidomain-only boundary conditions |
| `SimulationSpec` / `GridSpec` / `PacingSpec` / `create_simulation` | class / fn | declarative layer (LLM-intake bridge) |
| `cardiac_core.analysis` | module | pure-tensor analysis (`conduction_velocity`, `apd_map`, …) |

---

## Geometry

### `class Grid` `[design]`

Structured rectangular grid. The single geometry type (unstructured/FEM removed).

```python
Grid(Nx, Ny, dx, dy=None, *, mask=None, device="cpu", dtype=torch.float64)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `Nx`, `Ny` | int | — | grid points in x, y (`ij` convention; fields are `(Nx, Ny)`) |
| `dx` | float | — | spacing in x (cm) |
| `dy` | float | `dx` | spacing in y (cm); defaults to `dx` (isotropic spacing) |
| `mask` | `Tensor (Nx,Ny) bool` | `None` | tissue domain; `False` cells are non-tissue. `None` = full rectangle |
| `device`, `dtype` | — | `cpu`, `f64` | tensor placement |

**Properties:** `Lx = dx*(Nx-1)`, `Ly = dy*(Ny-1)`, `coordinates -> (x, y)` meshgrid tensors,
`n_dof` (number of tissue nodes).

```python
g = Grid(Nx=150, Ny=40, dx=0.025)                 # 3.725 x 0.975 cm sheet
x, y = g.coordinates
```

> **Note.** Internally `StructuredGrid.create_rectangle(Lx, Ly, Nx, Ny, …)` is the existing engine
> constructor; `Grid` is the unified front for it. LBM (which today takes `Nx/Ny/dx` inline) gains it.

---

## Conductivity

### `class ConductivityConfig` `[P3]`

The single owner of conductivity, `chi`, and `Cm`. Stores **physics** (`sigma_*`, `chi`, `Cm`,
`fiber_angle`) and emits the diffusion input each engine needs. **It is the only place the
Formulation-A/B asymmetry lives** — the user never sees it. (See `API_DESIGN.md` §4; firewall gate
verified 2026-06-24 to machine precision + live-engine CV.)

**Constructors** (use these — they hide the field sprawl):

```python
ConductivityConfig.isotropic(sigma, chi=1400.0, Cm=1.0)
ConductivityConfig.bidomain(sigma_i, sigma_e, chi=1400.0, Cm=1.0)
ConductivityConfig.anisotropic(sigma_l, sigma_t, fiber_angle, chi=1400.0, Cm=1.0)
```

| Constructor | Use when | Inputs |
|---|---|---|
| `.isotropic` | single-domain effective conductivity, isotropic | `sigma` (mS/cm) |
| `.bidomain` | paired intra/extra conductivities (most physical) | `sigma_i`, `sigma_e` (mS/cm) |
| `.anisotropic` | fiber-aligned tensor | `sigma_l`, `sigma_t`, `fiber_angle` (rad) |

> ⚠️ **Units.** `sigma*` are raw **conductivities in mS/cm**. The effective diffusivity
> `D_eff = sigma_eff/(chi·Cm)` (cm²/ms) is **derived** — do not pass a pre-divided `D` as `sigma`.

**Read-only properties**

| Property | Returns | Meaning |
|---|---|---|
| `sigma_eff` | float | harmonic i/e collapse `sigma_i·sigma_e/(sigma_i+sigma_e)` |
| `D_eff` | float / tensor | true physical effective diffusivity `sigma_eff/(chi·Cm)` |

**Per-engine emitters** (consumed by the factories — you normally never call these)

| Method | Returns | Notes |
|---|---|---|
| `for_monodomain()` | `{D, chi, Cm}` | Form A: `D = D_eff·Cm = sigma_eff/chi` (Cm-**un**scaled), `chi=1` (inert), real `Cm`. **Temporary** — deleted when monodomain converts to Form B in Phase 4. |
| `for_bidomain()` | `{D_i, D_e, Cm}` | Form B: `D_i,D_e = sigma_*/(chi·Cm)`, real `Cm` |
| `for_lbm()` | `{D, Cm}` | Form B: `D = D_eff`, real `Cm` |

```python
cfg = ConductivityConfig.bidomain(sigma_i=1.74, sigma_e=6.25, chi=1400)
cfg.D_eff                     # 0.000972 cm^2/ms
cfg.for_monodomain()          # {'D': 0.000972, 'chi': 1.0, 'Cm': 1.0}
```

---

## Ionic models

### `class IonicModel` (ABC)  `[now]`

Byte-identical across all engines — the proof that unification works. You instantiate a concrete model
and pass it to a factory; you rarely call its methods directly.

| Member | Signature | Description |
|---|---|---|
| `compute_Iion` | `(V, ionic_states) -> Tensor` | total ionic current |
| `compute_gate_steady_states` | `(V, ionic_states) -> Tensor` | gate `∞` values |
| `compute_gate_time_constants` | `(V, ionic_states) -> Tensor` | gate `τ` values |
| `compute_concentration_rates` | `(V, ionic_states) -> Tensor` | `d[ion]/dt` |
| `step` | `(V, ionic_states, dt, Istim=None) -> (V_new, states_new)` | one cell-model advance |
| `get_initial_state` | `(n_cells=1) -> Tensor` | resting state `(n_cells, n_states)` |

**Properties:** `name`, `n_states`, `V_rest`, `state_names`, `gate_indices`, `concentration_indices`.

> The ABC's positional voltage argument is `V` (not `Vm`) and is left as-is — renaming it is a separate,
> wide-blast follow-up. `Vm` is canonical only at the State/API layer.

### `class CellType` (enum)  `[now]`
`ENDO = 0`, `EPI = 1`, `M_CELL = 2`.

### Concrete models  `[now]`

```python
TTP06Model(cell_type=CellType.EPI, device=None)     # ten Tusscher–Panfilov 2006 (18-state)
ORdModel(cell_type=CellType.ENDO, device=None)      # O'Hara–Rudy (40-state)
# superset also: PHAS13Model, MHAS13Model, PaciModel
```

Factories also accept a **string** alias (`"ttp06"`, `"ord"`) for convenience.

---

## Stimulus

`Stim` is the **canonical, factory-facing** stimulus object (the form the engine factories and `.npz`
serialization consume via a normalized dict). `Stimulus`/`StimulusProtocol` (below) are the lower-level
per-engine protocol `Stim` lowers onto. The bare **dict** form is legacy (soft-deprecated — still works,
emits a `DeprecationWarning`).

### `class Stim`  `[now]`

A masked, timed stimulus with two modes, inferred from the keyword: **current injection** (`amplitude`,
default `-52.0` µA/µF, negative = depolarizing) or **voltage clamp** (`clamp=<mV>`, a hard V override on
mono/bidomain/LBM — routed to `clamp_voltage`, never serialized as a current). One fixed mask per `Stim`;
pass a **list** for multi-site/moving stimuli (overlaps sum on FDM/FEM, LBM overwrites).

```python
Stim(mask, *, amplitude=None, clamp=None, start_time=0.0, duration=2.0,
     bcl=0.0, num_pulses=1, label="stim")            # base: an explicit (Nx, Ny) bool mask

# eager classmethod factories over a Grid (NOT subclasses — one Stim type):
Stim.boundary(grid, side, *, width=None, **kw)       # side ∈ "left"/"right"/"top"/"bottom"
Stim.point(grid, (x, y), *, radius=None, **kw)       # a blob at an (x, y) cm point
Stim.center(grid, *, radius=None, **kw)              # a blob at the domain centre
Stim.from_region(grid, region, **kw)                 # any callable (x,y)->bool mask OR an (Nx,Ny) mask
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mask` | `(Nx,Ny)` bool (torch/numpy) | — | where the stimulus applies (factories build it from the grid) |
| `amplitude` | float | `-52.0` | current µA/µF; **negative = depolarizing** (current mode) |
| `clamp` | float / `(Nx,Ny)` field / `callable(t)` | `None` | clamp voltage (mV); presence selects voltage-clamp mode |
| `start_time` / `duration` | float | `0.0` / `2.0` | active window (ms) |
| `bcl` / `num_pulses` | float / int | `0.0` / `1` | pacing train (current mode only — rejected for a clamp) |

Methods: `to_dict()` (current-mode 7-key lowering; raises on a clamp Stim), `from_dict(d)`, `times()`,
`n_nodes()`. Passing `amplitude` **and** `clamp` together raises.

```python
sim = cc.monodomain(g, "ttp06", cond, cc.Stim.boundary(g, "left", bcl=1000, num_pulses=5))  # pace
sim = cc.lbm(g, "ttp06", cond, cc.Stim.boundary(g, "left", clamp=-20, duration=50))          # clamp
```

### `class Stimulus`  `[now]`

Lower-level per-engine stimulus event (what `Stim` lowers onto).

```python
Stimulus(region, start_time, duration=1.0, amplitude=-52.0)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `region` | callable `(x,y)->bool` **or** `(Nx,Ny)` bool mask | — | where the stimulus is applied |
| `start_time` | float | — | onset (ms) |
| `duration` | float | `1.0` | pulse width (ms) |
| `amplitude` | float | `-52.0` | current; **negative = depolarizing** |

### `class StimulusProtocol`  `[now]`

A collection of `Stimulus` events with high-level pacing builders. Overlapping stimuli **accumulate**.

| Method | Signature | Description |
|---|---|---|
| `add_stimulus` | `(region, start_time, duration=1.0, amplitude=-52.0) -> self` | add one event (chainable) |
| `add_s1s2_protocol` | `(region, n_s1, bcl, s2_ci, …) -> self` | S1–S2 restitution train |
| `add_regular_pacing` | `(region, bcl, n_beats, start_time=0.0, …) -> self` | periodic pacing |

```python
stim = (StimulusProtocol()
        .add_regular_pacing(left_edge_region, bcl=500.0, n_beats=5))
```

### Region helpers  `[now]`
Callables returning a boolean over `(x, y)`: `left_edge_region`, `right_edge_region`,
`circular_region(cx, cy, r)`, `rectangle_region(...)`, … Pass either a helper or your own
`lambda x, y: …`, or a precomputed `(Nx, Ny)` mask.

---

## DECLARE — engine factories

Each returns a `Simulation`. Shared positional args: `(geometry, ionic_model, conductivity, stimulus=None)`.
`stimulus` accepts a `Stim`, a list of `Stim`s, or the legacy dict/list-of-dicts (soft-deprecated). A
`clamp`-mode `Stim` is split out and applied via `clamp_voltage` (all three engines) rather than injected as a
current. Engine-specific knobs are keyword-only and never leak across engines.

### `monodomain(...)`  `[now]`

```python
monodomain(geometry, ionic_model, conductivity, stimulus=None, *,
           dt=0.02, Cm=1.0,
           splitting="strang", ionic_solver="rush_larsen",
           diffusion_solver="crank_nicolson", linear_solver="pcg") -> Simulation
```

| Knob | Options | Default |
|---|---|---|
| `splitting` | `strang`, `godunov` | `strang` |
| `ionic_solver` | `rush_larsen`, `forward_euler` | `rush_larsen` |
| `diffusion_solver` | `crank_nicolson`, `bdf1`, `bdf2`, `forward_euler`, `rk2`, `rk4` | `crank_nicolson` |
| `linear_solver` | `pcg`, `chebyshev`, `dct`, `fft`, `none` | `pcg` |

> FEM dropped (2026-06-24): there is no `discretization` knob — monodomain is FDM on the structured grid
> (FVM available internally; not a user default).

### `bidomain(...)`  `[now]`

```python
bidomain(geometry, ionic_model, conductivity, stimulus=None, *,
         dt=0.02, Cm=1.0,
         splitting="strang", ionic_solver="rush_larsen",
         parabolic_solver="pcg", elliptic_solver="auto", theta=0.5,
         boundary=None) -> Simulation
```

| Knob | Options | Default | Note |
|---|---|---|---|
| `parabolic_solver` | `pcg`, `chebyshev`, `spectral` | `pcg` | Vm half-step |
| `elliptic_solver` | `auto`, `spectral`, `pcg_spectral`, `pcg`, `pcg_gmg` | `auto` | φ_e solve; **use `auto`** |
| `theta` | `0.5` (CN) / `1.0` (BDF1) | `0.5` | implicitness |
| `boundary` | `BoundarySpec` | insulated | bath coupling (Kleber effect) |

### `lbm(...)`  `[now]`

```python
lbm(geometry, ionic_model, conductivity, stimulus=None, *,
    dt, Cm=1.0, lattice="d2q5", weights_mode="canonical") -> Simulation
```

| Knob | Options | Default | Note |
|---|---|---|---|
| `dt` | float | **required** | `τ` derived from `D·dt`; not defaulted |
| `lattice` | `d2q5`, `d2q9`, `d2q9_uniform` | `d2q5` | `d2q5` isotropic; `d2q9` for `Dxy` |
| `weights_mode` | `canonical`, `uniform_8` | `canonical` | |

---

## The `Simulation` object

The engine-agnostic runtime handle returned by the factories. Program against this; the concrete engine
is hidden.

**Introspection (read-only):** `Nx`, `Ny`, `dx`, `dy`, `dt`, `Cm`, `ionic_model`,
`Vm -> Tensor (Nx,Ny)` (current voltage), `t -> float` (current time, ms).

### `run(...)`  `[now eager / design batch]`

```python
run(t_end, save_every=1.0, *, batch=None, record=("Vm",), callback=None)
    -> SimulationResult | Iterator[SimulationResult]
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `t_end` | float | — | integrate to this time (ms) |
| `save_every` | float | `1.0` | save-point interval (ms) |
| `batch` | `None` \| int≥1 | `None` | `None` = **eager** (one `SimulationResult`). `k` = **streaming**: yields `SimulationResult` chunks of ≤ `k` save-points (`k=1` = frame-by-frame) |
| `record` | tuple[str] | `("Vm",)` | fields to keep. `phi_e` auto-added for bidomain; `"ionic_states"` opt-in. Bounds memory |
| `callback` | callable | `None` | eager early-stop / progress; return falsy to stop |

**Returns:** one `SimulationResult` when `batch is None`; otherwise an iterator of `SimulationResult`.

```python
res = sim.run(t_end=300.0, save_every=1.0)                 # eager
for chunk in sim.run(t_end=300.0, save_every=1.0, batch=50):  # streaming
    render(chunk.Vm[-1])
```

### `step()`  `[now]`
Advance one configured `dt`, mutating internal live state. `-> None`.

### `reset()`  `[design]`
Restore the initial state (re-run from `t = 0`). `-> None`.

### `with_(**overrides)`  `[design]`
Return a **new** `Simulation` with overridden parameters; the original is untouched (immutable —
sweep-safe; the canonical CHANGE idiom). `-> Simulation`.

```python
sim2 = sim.with_(dt=0.01)
sim3 = sim.with_(conductivity=ConductivityConfig.isotropic(sigma=2.0))
```

### `stimulate(region, start_time=0.0, duration=1.0, amplitude=-52.0)`  `[now]`
Add a stimulus after construction and rebuild from t=0. `region` is a `Stim`, a callable `(x,y)->bool mask`,
or an `(Nx,Ny)` mask (a `Stim` carries its own timing/amplitude; a clamp `Stim` routes to `clamp_voltage`).
`-> None`.

---

## Results

### `class SimulationResult`  `[now]`

The **only** public output type (eager and streamed both return this).

| Field | Type | Description |
|---|---|---|
| `times` | `Tensor (T,)` | save-point times (ms) |
| `Vm` | `Tensor (T, Nx, Ny)` f64 | transmembrane potential history |
| `phi_e` | `Tensor (T, Nx, Ny)` \| `None` | extracellular potential (bidomain only) |
| `ionic_states` | `Tensor (T, Nx, Ny, n_states)` \| `None` | only if `"ionic_states"` in `record` |
| `dx`, `dy` | float | grid spacing (for analysis) |

**Analysis hooks** (thin wrappers over `cardiac_core.analysis`):

| Method | Returns | Description |
|---|---|---|
| `cv(**kw)` | float / tensor | conduction velocity |
| `apd(**kw)` | tensor | action-potential duration map |
| `lat(**kw)` | tensor | local activation-time map |
| `restitution(**kw)` | (CI, APD) arrays | restitution curve |

```python
res = sim.run(t_end=50.0)
res.Vm.shape          # (51, 200, 50)
res.cv()              # cm/s
```

---

## Engine-specific types

### `class BoundarySpec` (bidomain)  `[now]`
Boundary conditions for the elliptic (φ_e) solve.

```python
BoundarySpec.insulated()                       # Neumann all edges (default)
BoundarySpec.bath_coupled()                    # Dirichlet bath on all edges
BoundarySpec.bath_coupled_edges(edges=...)     # bath on selected edges (Kleber boundary effect)
```

---

## Declarative layer (Goal-2 bridge)

### `class SimulationSpec`  `[design]`

A declarative, serializable description of a whole simulation. Fields are **self-describing**
(`{tier, prompt, options, default}`) — the field metadata *is* the LLM intake questionnaire, so it
can't drift from what the engines need. Three tiers:

| Tier | Behavior | Example fields |
|---|---|---|
| `required` | the LLM/user must supply | `engine`, `geometry`, `pacing`, `measure` |
| `defaulted` | silent physiological default | `ionic`, `dt`, `Cm`, `conductivity`, solvers, `chi` |
| `derived` | computed, never asked | `save_every`, `t_end`, `D_eff`, LBM `tau` |

Helper specs: `GridSpec(Lx, Ly, dx)` and `PacingSpec.single(...) / .s1s2(...) / .regular(bcl, n_beats)`
(expand into a `Grid` / `StimulusProtocol`).

### `create_simulation(spec) -> Simulation`  `[design]`
Dispatches on `spec.engine` to the matching factory, filling defaulted/derived fields. Thin glue — no
new physics. This is the seam the conversational builder (Goal 2) drives.

```python
sim = create_simulation(SimulationSpec(
    engine="monodomain",
    geometry=GridSpec(Lx=2.0, Ly=0.5, dx=0.01),
    pacing=PacingSpec.single(region="left_edge"),
    measure=["cv"],
))                                   # everything else defaulted/derived
res = sim.run(t_end=50.0)
```

---

## `cardiac_core.analysis`  `[now]`

Pure-tensor analysis functions (operate on `(T, Nx, Ny)` voltage histories; `SimulationResult` hooks
call these): `activation_time`, `conduction_velocity`, `apd_at` / `apd_map`, `dominant_frequency`,
`wavefront_mask`, `phase_map`, `phase_singularities`, `restitution_curve`.

---


## Video (`cardiac_core.video`)

Spec-first video rendering. A `Video` holds the description and `render()` turns it into frames.
**Rendering displays; naming a destination saves** — with no `path=` and no `media/` convention
keyword the result is returned in memory and plays inline in a notebook. **Status: implemented**
(Phases 1-3 of
`cardiac_core/VIDEO_OBJECT_PLAN.md`).

```python
from cardiac_core import Video, Gradient, render, VideoInfo
```

| Object | Signature (abridged) | Notes |
|---|---|---|
| `SimulationResult.video` | `r.video(slug="video", **kw) -> VideoInfo` | The one-liner; displays inline unless a destination is named. Video-level kwargs (`gradient`, `style`, `front`, `isochrones`, `mask`, `field`, `label`, `aspect`, `units`) are split out and passed to `Video`; the rest go to `render`. |
| `Video` | `Video(data, field="Vm", gradient=Gradient.physiological(), label=None, front=None, isochrones=False, mask=None, style="bare", aspect="equal", units="auto")` | `data` = `SimulationResult` \| `(times, V)` \| `(T,Nx,Ny)` array \| `.npz` path. `Video.bare()` / `Video.annotated()` presets. `mask=False` disables masking. `.preview(t_ms=…)` renders one frame to PNG. |
| `Gradient` | `Gradient(cmap="viridis", value_range="physiological", gamma=1.0, levels=None, bad="0.55", interpolation="nearest", v_rest=None, rest_vmax=40.0, zoom_span=8.0, zoom_below=0.3)` | Frozen. Presets: `physiological()` `rest_anchored(vmax=40)` `zoom(span, below)` `diverging()` `autoscale()`. `cmap` accepts a name, a `Colormap`, or a list of colours. |
| `render` | `render(video_or_list, slug="video", *, path=None, question=None, bulk=None, resolution="1080p", fit="contain", fps=20.0, speed=None, max_frames=300, format=None, bitrate=None, show_time=None, colorbar=None, title=None, figsize=None, dpi=None, units=None, progress=False, labels=None, rows=None, cols=None, date=None, root=None) -> VideoInfo` | A LIST renders N panels sharing one colorbar + one time stamp. `render_video` is an alias. A file is written ONLY when `path=` or a convention keyword is given; `bulk` then defaults to True. `format` follows `path`'s extension. |
| `VideoInfo` | `.path .saved .data .n_frames .fps .backend .codec .bitrate .width .height .duration_s .vmin .vmax .stride .size_bytes`, `.read()` `.save(path)` | `path` is `None` when nothing was saved. Str-like only when saved (`os.fspath` raises otherwise). Displays inline in a notebook. `.backend` makes any encoder fallback visible. |

Defaults worth knowing: the zero-argument call is **bare, unlabelled, 1080p**, `Gradient.physiological()`
(viridis, -90..40 mV), aspect preserved with letterbox padding. Masking routes through `domain_mask`
(**True = ACTIVE**) so LBM's *finite* obstacle nodes are excluded from both the display and the colour
range. Figure-only knobs raise on a bare clip rather than silently doing nothing.

## Appendix — what exists today vs. designed

| Area | Status |
|---|---|
| `monodomain/bidomain/lbm` factories, `SimulationResult`, eager `run`, `analysis`, `geometry`, `io` | **built** in `cardiac_core` (wrapper level, 77 tests) — voltage currently shipped as `V` (to be renamed `Vm`) |
| `Vm` rename, grid-only `Vm` output | rename pending (glossary #2/#7) |
| `Grid`, `ConductivityConfig` (incl. firewall emitters), `with_`, `reset`, `stimulate`, `batch`/`record`, `SimulationSpec`/`create_simulation` | **designed here** — land in Phases 1–5 |
| Monodomain Form-A → Form-B convergence; delete `for_monodomain()`; drop FEM | **Phase 4** |

> This reference describes the **target** surface. As consolidation phases land, `[design]` symbols
> become `[now]`; the firewall's `for_monodomain()` disappears at Phase 4 (Form-B convergence).
