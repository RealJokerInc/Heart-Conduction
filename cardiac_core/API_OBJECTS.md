# cardiac_core — Object Atlas

```python
import cardiac_core as cc
```

## The object map

| Object | Meaning | You get it from |
|---|---|---|
| `Grid` | structured geometry — node counts + spacing | `cc.Grid(...)` |
| `ConductivityConfig` | tissue physics → diffusivity (the χ/Cm firewall) | `cc.ConductivityConfig.bidomain(...)` |
| `Stim` | a masked, timed stimulus (current **or** voltage clamp) | `cc.Stim.boundary(g, "left")` |
| `IonicModel` | cell electrophysiology (TTP06 / ORd / hiPSC) | a name string, or `TTP06Model(...)` |
| `CardiacSimulation` | the live simulation handle | `cc.monodomain(...)` / `bidomain` / `lbm` |
| `SimulationResult` | the run output + analysis hooks | `sim.run(...)` |
| `Fields` / `VectorField` | spatial field calculus on a result | `r.fields` |
| `SingleCellResult` | 0-D action potential | `cc.single_cell(...)` |
| `Video` / `Gradient` / `VideoInfo` | render spec → rendered file | `cc.Video(r)` → `cc.render(...)` |
| `Image` / `Trace` / `ImageInfo` | still-figure spec → rendered figure | `r.image()` / `r.trace()` / `cc.draw(...)` |
| `CardiacMeshData` | the on-disk mesh (`.npz`) | `cc.create_cardiac_mesh(...)` |
| `Distribution` | per-node random parameter spec | `cc.Distribution(...)` |
| `SimulationSnapshot` | one frame from a generator run | `sim.snapshots(...)` |

**The pipeline:**

```
Grid + ConductivityConfig + Stim → monodomain() → CardiacSimulation → .run() → SimulationResult → .fields / .video()
```

---

## 1. `Grid`

```python
g = cc.Grid(Nx, Ny, dx, dy=None, *, mask=None, boundary_mode="face_mirror",
            device="cpu", dtype=torch.float64)
```

**Attributes**

| Access | Meaning |
|---|---|
| `g.Nx`, `g.Ny` | node counts (int) |
| `g.dx`, `g.dy` | spacing, **cm** (`dy` defaults to `dx`) |
| `g.mask` | `(Nx, Ny)` bool tissue mask, or `None` |
| `g.boundary_mode` | ghost/mirror edge rule (`"face_mirror"`) |
| `g.device`, `g.dtype` | tensor device / float dtype |
| `g.Lx`, `g.Ly` | *property* — domain size **cm**: `dx*(Nx-1)`, `dy*(Ny-1)` |
| `g.coordinates` | *property* — `(x, y)` tensors `(Nx, Ny)`, `ij` indexing; `x[0,0]==0`, `x[-1,0]==Lx` |
| `g.n_dof` | *property* — active nodes: `mask.sum()` if masked, else `Nx*Ny` |

**Methods** — none.

For an exact **Lx × Ly cm** domain: `Nx = round(Lx/dx) + 1`. A 1-D cable is `Ny=1`.

---

## 2. `ConductivityConfig`

Never constructed raw — use a classmethod.

**Constructors**

| Call | Does |
|---|---|
| `cc.ConductivityConfig.isotropic(sigma, chi=1400.0, Cm=1.0)` | one effective σ |
| `cc.ConductivityConfig.bidomain(sigma_i, sigma_e, chi=1400.0, Cm=1.0)` | intra + extracellular — most physical |
| `cc.ConductivityConfig.anisotropic(sigma_l, sigma_t, fiber_angle, chi=1400.0, Cm=1.0)` | fibers (angle in **radians**) |

**Attributes**

| Access | Meaning |
|---|---|
| `cond.sigma_i`, `cond.sigma_e` | intra / extracellular conductivity, **mS/cm** |
| `cond.sigma_l`, `cond.sigma_t` | longitudinal / transverse (anisotropic) |
| `cond.sigma_iso` | the isotropic value |
| `cond.chi` | surface-to-volume, **cm⁻¹** |
| `cond.Cm` | capacitance, **µF/cm²** |
| `cond.fiber_angle` | fiber angle, **radians** |
| `cond.sigma_eff` | *property* — effective σ **mS/cm**; scalar, but an `(xx, yy, xy)` 3-tuple for anisotropic |
| `cond.D_eff` | *property* — diffusivity **cm²/ms** = `sigma_eff/(chi*Cm)`; same scalar-or-tuple shape |

**Methods**

| Call | Does |
|---|---|
| `cond.for_monodomain()` | emit Form A — `D=sigma_eff/chi`, `chi=1`, real `Cm` |
| `cond.for_bidomain()` | emit Form B for the bidomain factory |
| `cond.for_lbm()` | emit Form B with fully-scaled `D` |

---

## 3. `Stim`

```python
s = cc.Stim(mask, *, amplitude=None, clamp=None, start_time=0.0, duration=2.0,
            bcl=0.0, num_pulses=1, label="stim")
```

**Constructors**

| Call | Does |
|---|---|
| `cc.Stim.boundary(grid, side, *, width=None, **kw)` | border strip — `"left"/"right"/"top"/"bottom"` |
| `cc.Stim.point(grid, center, *, radius=None, **kw)` | blob at an `(x, y)` cm point |
| `cc.Stim.center(grid, *, radius=None, **kw)` | blob at the domain centre |
| `cc.Stim.from_region(grid, region, **kw)` | from a `callable(x,y)->mask` or an `(Nx,Ny)` array |
| `cc.Stim.from_dict(d)` | rebuild from the normalized dict |

**Attributes**

| Access | Meaning |
|---|---|
| `s.mask` | `(Nx, Ny)` numpy bool |
| `s.mode` | `"current"` or `"clamp"` — inferred from which keyword you used |
| `s.amplitude` | **µA/µF**, negative = depolarizing (default −52) |
| `s.clamp` | clamp level **mV**, or `None` |
| `s.start_time`, `s.duration` | active window, **ms** |
| `s.bcl`, `s.num_pulses` | pacing train |
| `s.label` | free-text tag |

**Methods**

| Call | Does |
|---|---|
| `s.times()` | list the pulse onsets — `[t0 + k·bcl]` |
| `s.n_nodes()` | count the nodes the mask actually hits |
| `s.to_dict()` | lower to the dict engines/`.npz` consume — **current mode only** (raises on a clamp) |

```python
cc.Stim.boundary(g, "left", amplitude=-52, bcl=1000, num_pulses=5)   # CURRENT injection
cc.Stim.boundary(g, "left", clamp=-20, duration=50)                  # VOLTAGE clamp, all 3 engines
```

---

## 4. `IonicModel`

Usually just a **name string**: `"ttp06"` (default), `"ord"` (**LBM only**), `"paci"`, `"phas13"`, `"mhas13"`.

```python
from cardiac_core.ionic import TTP06Model, ORdModel, CellType
m = TTP06Model(cell_type=CellType.EPI)      # ENDO (default) / EPI / M
```

**Constructors**

| Call | Does |
|---|---|
| `TTP06Model.from_config(config, device=None, dtype=torch.float64, use_lut=False, base_cell_type=CellType.EPI)` | build from a `CellTypeConfig` |

**Attributes**

| Access | Meaning |
|---|---|
| `m.cell_type` | the `CellType` member — `ENDO` / `EPI` / `M` |
| `m.params` | the parameter object — maximal conductances live here |
| `m.params.GNa`, `.GK1`, `.GKr`, `.GKs`, `.Gto`, `.GbCa`, `.GbNa`, `.GpCa`, `.GpK`, `.PCa`, `.PNaK` | TTP06 scalable conductances |
| `m.V_rest` | *property* — resting potential, **mV** |
| `m.name` | *property* — model name |
| `m.n_states` | *property* — state count (18 for TTP06) |
| `m.state_names` | *property* — ordered ionic state names |
| `m.gate_indices`, `m.concentration_indices` | *property* — state-column indices by kind |
| `m.use_epi_ito_kinetics` | *property* — cell-type Ito switch |

**Methods**

| Call | Does |
|---|---|
| `m.compute_Iion(V, ionic_states)` | total ionic current |
| `m.compute_gate_steady_states(V, ionic_states)` | gate `x∞` |
| `m.compute_gate_time_constants(V, ionic_states)` | gate `τ` |
| `m.compute_concentration_rates(V, ionic_states)` | concentration derivatives |
| `m.get_initial_state(n_cells=1)` | build the resting state tensor |
| `m.step(V, ionic_states, dt, I_stim=None)` | advance one Rush–Larsen ionic step |
| `m.run(t_end, dt=0.01, stim_times=None, stim_duration=1.0, stim_amplitude=-80.0, save_interval=None)` | standalone 0-D run |

Factories accept a **name or a pre-built instance**, so a scaled model can be passed straight in.

---

## 5. `CardiacSimulation`

```python
sim = cc.monodomain(g, "ttp06", cond, stim)     # fast, single potential — default
sim = cc.bidomain(g, "ttp06", cond, stim)       # when bath / edge loading matters
sim = cc.lbm(g, "ttp06", cond, stim, dt=0.005)  # lattice-Boltzmann
```

**Attributes** (all properties)

| Access | Meaning |
|---|---|
| `sim.Nx`, `sim.Ny` | grid size |
| `sim.dx`, `sim.dy` | spacing, **cm** |
| `sim.dt` | timestep, **ms** |
| `sim.t` | current time, **ms** |
| `sim.Vm` (alias `sim.V`) | live `(Nx, Ny)` voltage, **mV** |
| `sim.phi_e` | extracellular potential — **bidomain only** |
| `sim.mask` | tissue mask |
| `sim.engine_type` | `"monodomain"` / `"bidomain"` / `"lbm"` |
| `sim.ionic_model` | the live model instance |
| `sim.ionic_states` | live `(n_dof, n_states)` tensor |
| `sim.state_names` | the ionic state names |
| `sim.Cm` | capacitance, **µF/cm²** |
| `sim.boundary_mode` | the stencil ghost rule |

**Methods — run & control**

| Call | Does |
|---|---|
| `sim.run(t_end, save_every=1.0, *, batch=None, record=("Vm",), callback=None)` | run it — eager → `SimulationResult`; `batch=k` → iterator of chunks |
| `sim.snapshots(t_end, save_every=1.0, *, record=("Vm",), callback=None)` | yield `SimulationSnapshot` frames lazily |
| `sim.step()` | advance one timestep |
| `sim.reset()` | rebuild at t=0 |
| `sim.with_(**overrides)` | return a NEW sim with new knobs — sweep-safe, original untouched |

**Methods — stimulus & clamp**

| Call | Does |
|---|---|
| `sim.stimulate(region, start_time=0.0, duration=1.0, amplitude=-52.0)` | append a stimulus (`Stim`, callable, or mask) and rebuild from t=0 |
| `sim.add_stimulus(mask, start_time, duration, amplitude=-80.0)` | same, mask-form alias |
| `sim.add_pacing(mask, bcl=1000.0, n_beats=10, start_time=0.0, duration=2.0, amplitude=-80.0)` | add a pacing train |
| `sim.clamp_voltage(mask, voltage, start_time=None, duration=None)` | hold V (scalar, field, or `callable(t)`) |
| `sim.add_clamp_protocol(mask, steps, start_time=0.0)` | multi-step VC — `steps=[(mV, ms), …]` |
| `sim.release_clamp()` | remove the clamp |
| `sim.inject_current(mask, amplitude)` | inject current mid-run |

**Methods — heterogeneity**

| Call | Does |
|---|---|
| `sim.scale_conductance(current_name, factor, mask=None)` | drug block / upregulation; compounds, rebuilds from t=0 |
| `sim.set_conductivity(mask, D)` | set a region's diffusivity (scar: `D=0.0`) |
| `sim.scale_conductivity(mask, factor)` | multiply a region's diffusivity |
| `sim.set_parameter(name, value, mask=None)` | **raises `NotImplementedError`** |
| `sim.get_parameter_field(name)` | **raises `NotImplementedError`** |

**Methods — state injection** (mono/bidomain only; LBM raises)

| Call | Does |
|---|---|
| `sim.get_state(name)` | read one ionic state as `(Nx, Ny)` |
| `sim.set_state(name, values)` | overwrite one ionic state |
| `sim.set_voltage(V)` | overwrite the membrane voltage |

**Methods — probes & shortcuts**

| Call | Does |
|---|---|
| `sim.add_probe(name, x, y)` | record a trace at an `(x, y)` cm point |
| `sim.get_traces()` | return the recorded probe traces |
| `sim.clear_traces()` | discard recorded traces |
| `sim.compute_cv(x1, x2, y, threshold=-20.0)` | conduction velocity between two x-indices |
| `sim.compute_apd(x, y, repol=0.9)` | APD at a point |
| `sim.compute_activation_time(threshold=-20.0)` | activation-time map |

---

## 6. `SimulationResult`

```python
r = sim.run(t_end=40.0, save_every=0.5)
```

**Attributes**

| Access | Meaning |
|---|---|
| `r.times` | `(T,)` **ms** |
| `r.Vm` (alias property `r.V`) | `(T, Nx, Ny)` voltage — masked-out nodes are **NaN** |
| `r.phi_e` | `(T, Nx, Ny)`, or `None` |
| `r.dx`, `r.dy` | spacing carried from the sim |
| `r.ionic_states` | present if `record=("Vm","ionic_states")` |
| `r.domain_mask` | tissue mask |
| `r.boundary_mode`, `r.Cm`, `r.chi` | analysis context |
| `r.conductivity`, `r.ionic_model`, `r.cell_type` | provenance |
| `r.fields` | *cached property* — the `Fields` accessor (§7) |

**Methods**

| Call | Does |
|---|---|
| `r.cv(x1, x2, y, **kw)` | conduction velocity **cm/s** between two x-indices |
| `r.cv_between(p1, p2, **kw)` | CV along the line between two `(ix,iy)` nodes |
| `r.radial_cv(center, **kw)` | outward CV map from a point source |
| `r.apd(**kw)` | APD map |
| `r.apd_per_beat(ix, iy, **kw)` | APD of each beat at a node |
| `r.lat(**kw)` | `(Nx, Ny)` local activation time map |
| `r.df_map()` | dominant-frequency map |
| `r.restitution(ix, iy, **kw)` | `(DI, APD)` restitution curve |
| `r.restitution_slope(ix, iy, **kw)` | max slope + DI\* (alternans onset) |
| `r.video(slug="video", **kw)` | render a video — displays inline; saves only when `path=` or a convention keyword is given |

---

## 7. `Fields` and `VectorField`

**`Fields` attributes** — reached as `r.fields.<name>`, all properties, computed lazily.

| Access | Meaning |
|---|---|
| `.voltage_gradient` | ∇V |
| `.voltage_flux` | voltage flux |
| `.current_flux` | current flux |
| `.electric_field` | E = −∇V |
| `.velocity` | wavefront velocity |
| `.speed` | \|velocity\| |
| `.direction` | propagation direction |
| `.divergence` | ∇·F |
| `.curvature` | wavefront curvature |
| `.vorticity` | ∇×v |
| `.source_sink` | source–sink balance |
| `.quality` | per-node reliability of the derived fields |
| `.mask` | the tissue mask used |
| `.derivatives` | the raw derivative sub-accessor |
| `.integrals` | the integral sub-accessor (`region_integral`, `net_flux`, `circulation`) |

**Methods** — none.

**`VectorField` attributes** — returned by the vector-valued fields above.

| Access | Meaning |
|---|---|
| `v.x`, `v.y` | components |
| `v.components` | both, stacked |
| `v.magnitude` | length per node |
| `v.angle` | direction, **radians** |
| `v.shape`, `v.device`, `v.dtype` | tensor metadata |

**Methods** — none.

---

## 8. `SingleCellResult`

```python
sc = cc.single_cell("ttp06", celltype="EPI", pre_pace=5)
```

**Attributes**

| Access | Meaning |
|---|---|
| `sc.times` | `(T,)` **ms** |
| `sc.V` | `(T,)` voltage trace, **mV** |
| `sc.final_state` | end ionic state — carry into a tissue run |
| `sc.model` | the model name |
| `sc.dt`, `sc.Cm` | timestep, capacitance |
| `sc.v_peak` | *property* — overshoot, **mV** |
| `sc.v_rest` | *property* — diastolic rest, **mV** |

**Methods**

| Call | Does |
|---|---|
| `sc.apd(repol=0.9, threshold=-20.0)` | action potential duration, **ms** |

Full signature: `single_cell(model="ttp06", *, celltype="ENDO", dt=None, bcl=1000.0, n_beats=1,
pre_pace=0, stim_amplitude=-52.0, stim_duration=2.0, t0=10.0, Cm=1.0, save_every=None,
conductances=None, device="cpu")`

`conductances={name: factor}` applies a drug at 0-D — multiplicative (`<1` block, `>1` upregulation),
name-validated (a typo raises), applied BEFORE `pre_pace` so pre-pacing settles the drugged cell. Same
names and semantics as tissue `scale_conductance`: `cc.single_cell("ttp06", conductances={"GKr": 0.5})`.

---

## 9. `Video`, `Gradient`, `VideoInfo`

```python
v = cc.Video(r, field="Vm", gradient=cc.Gradient(cmap="inferno"))
info = cc.render(v)                    # displays inline in a notebook; writes NO file
info = cc.render(v, path="wave.mp4")   # writes ./wave.mp4, exactly there
info = cc.render(v, "spiral-wave", bulk=True)   # -> media/lab/_sim_outputs/videos/{date}/…
```

**Rendering displays; naming a destination saves** — the matplotlib contract. A file appears only
when you pass `path=` or one of the `media/` convention keywords (`question=`/`bulk=`/`root=`/
`date=`). Otherwise the encoded bytes are returned on the object and play inline via a data URI,
which needs neither a file server nor a persistent disk (so it works on Colab).

### `Video`

**Constructors**

| Call | Does |
|---|---|
| `cc.Video.bare(data, **kw)` | no chrome — raw raster |
| `cc.Video.annotated(data, **kw)` | with title / colourbar / time |

**Attributes**

| Access | Meaning |
|---|---|
| `v.data` | the result (or array) to draw |
| `v.field` | which field — `"Vm"`, `"phi_e"`, or an array |
| `v.gradient` | the `Gradient` colour spec |
| `v.label` | panel label |
| `v.front` | front-contour level, or `None` |
| `v.isochrones` | draw activation isochrones |
| `v.mask` | tissue mask |
| `v.style`, `v.aspect`, `v.units` | chrome style, aspect ratio, unit display |

**Methods**

| Call | Does |
|---|---|
| `v.preview(t_ms=None, *, frame=None, slug="preview", path=None, question=None, bulk=None, **kw)` | render ONE still — displays inline; saves only when a destination is named |
| `v.display_values(t)` | return the `(Nx, Ny)` array actually drawn at frame `t` |
| `v.masked_iter(idx)` | iterate masked frames |
| `v.requires_figure()` | report whether it needs a matplotlib figure vs a raw raster |

### `Gradient`

**Constructors**

| Call | Does |
|---|---|
| `cc.Gradient.physiological(**kw)` | fixed physiological mV range (default) |
| `cc.Gradient.autoscale(**kw)` | scale to the data |
| `cc.Gradient.diverging(**kw)` | diverging map, for signed fields |
| `cc.Gradient.rest_anchored(vmax=40.0, **kw)` | anchor at rest, cap at `vmax` |
| `cc.Gradient.zoom(span=8.0, below=0.3, **kw)` | narrow window — reveals sub-threshold structure |

**Attributes**

| Access | Meaning |
|---|---|
| `gr.cmap` | colormap name / list / `Colormap` |
| `gr.value_range` | `"physiological"` or an explicit `(vmin, vmax)` |
| `gr.gamma` | gamma correction |
| `gr.levels` | discrete contour levels, or `None` |
| `gr.bad` | colour for NaN / masked nodes |
| `gr.interpolation` | raster interpolation |
| `gr.v_rest`, `gr.rest_vmax` | rest anchor and cap |
| `gr.zoom_span`, `gr.zoom_below` | zoom-window width and lower offset |

**Methods**

| Call | Does |
|---|---|
| `gr.key()` | return a cache key for this spec |
| `gr.resolve(masked_values, *, field="Vm")` | resolve to a concrete norm + colormap |

### `VideoInfo`

What `render` produced. Displays itself in a notebook; str-like when a file was written.

**Attributes**

| Access | Meaning |
|---|---|
| `info.path` | the written file — `None` when no destination was named |
| `info.saved` | whether a file was written |
| `info.data` | encoded bytes, retained only when nothing was saved |
| `info.n_frames`, `info.fps`, `info.duration_s` | timing |
| `info.backend`, `info.codec` | encoder actually used |
| `info.width`, `info.height` | pixels |
| `info.vmin`, `info.vmax` | colour range used |
| `info.stride`, `info.size_bytes`, `info.bitrate` | frame stride, file size, bitrate |

**Methods**

| Call | Does |
|---|---|
| `info.read()` | the encoded bytes, from memory or from the saved file |
| `info.save(path)` | write it out after the fact and mark this object saved (releases `data`); returns the path |
| `info.show()` | display it explicitly, like `plt.show()` — inline in a notebook, else the OS player from a terminal; prints the path on a headless box, never raises. Returns `None`; does NOT save |
| `info._repr_html_()` | the inline `<video>` (called by Jupyter/Colab, not by you) |

`os.fspath(info)` and `str(info)` give the path when one exists; `os.fspath` raises a
guiding `TypeError` when nothing was saved.

`render(video_or_list, slug="video", *, path=None, question=None, bulk=None, fps=20.0,
max_frames=300, format=None, ...)` — pass a **list** of `Video`s for side-by-side panels.
`format` defaults to `path`'s extension, else `"mp4"`.

### `ImagePath`

Returned by `Video.preview(...)`. A `str` subclass — every existing path use keeps working — that
also carries `.data`/`.saved`/`.format`, a `.read()`/`.save(path)` pair, and displays inline
(the format follows `path=`'s extension, so it is not always PNG). `.save()` returns the path
`str` like the other two objects; a `str` cannot change its own value, so the ORIGINAL stays
unsaved — use the return value. `.show()` displays it explicitly (inline in a notebook, else the
OS image viewer; returns `None`) — and when unsaved it materialises from the in-memory bytes, so
the human-sentence string value is never handed to the opener as a path.

---

## 10. `Image`, `Trace`, `ImageInfo`

Still figures. Same contract as the video layer — **drawing displays; naming a destination saves** —
and the same reusable `Gradient`. Two spec types, because a map and a series do not share a
description. Verified by introspection over each class.

### `Image` — one spatial map

`Image` is **resolved at construction**: mutating a field afterwards has no effect (unlike `Video`,
whose fields are live). Rebuild the spec instead.

| Access | Meaning |
|---|---|
| `data` | a `SimulationResult`, or a bare `(Nx, Ny)` array |
| `what` | `"snapshot"` (default) · `"activation"` · `"apd"` · `"frequency"` · any `r.fields` name |
| `at` | a **TIME in ms** (on `Trace` the same keyword is a NODE). Raises on a static map |
| `field` | `"Vm"` / `"phi_e"` — legal only with `what="snapshot"` |
| `what_kwargs` | forwarded to the selector's own analysis function |
| `gradient` | a `Gradient`; `None` → a per-`what` default (a map in ms is not put on a mV scale) |
| `label` | panel title |
| `front` | mV isoline drawn over the map |
| `isochrones` | LAT contours; `None` → on for `what="activation"` unless `filled` |
| `filled` | `contourf` bands instead of an image — the bands ARE the isochrones |
| `contour_levels` | isoline count (NOT `Gradient.levels`, which is colormap quantization) |
| `mask` | `None` → the run's `domain_mask` · an array · `False` → none |
| `style` | `"annotated"` (default) or `"bare"` |
| `aspect`, `units` | `"equal"`/`"auto"`; `"auto"`/`"cm"`/`"nodes"` |
| `value_label` | colorbar label; an explicit value always wins over the derived one |

| Call | Does |
|---|---|
| `display_values()` | the single frame with inactive tissue set to NaN, `(Nx, Ny)` |
| `requires_figure()` | True when the spec needs the matplotlib producer |

### `Trace` — one series panel

| Access | Meaning |
|---|---|
| `data` | `SimulationResult` · `SingleCellResult` · `(x, y)` · `{label: (x, y)}` |
| `what` | `"trace"` (default) · `"restitution"` · `"apd_per_beat"` |
| `at` | a **NODE** `(ix, iy)`, a list, or `{label: node}` whose keys become the legend |
| `series` | explicit `[(label, x, y), …]` override |
| `label`, `xlabel`, `ylabel` | title and axis labels (derived from `what` when unset) |
| `hline`, `vline` | reference lines: a scalar, a list, or `(value, label)` pairs |
| `legend` | `None` → on when there is more than one series |
| `marker`, `linestyle` | `None` → per-`what` defaults; restitution is marker-only |
| `xlim`, `ylim`, `logx`, `logy`, `colors` | axis limits, log scales, explicit colours |
| `what_kwargs` | forwarded to `restitution_curve` / `apd_per_beat` |

### `ImageInfo` — what `draw()` produced

| Access | Meaning |
|---|---|
| `path` | where it was written, or `None` when nothing was saved (the default) |
| `data` | the encoded bytes — the sole copy when unsaved |
| `format` | `png` · `svg` · `pdf` · `jpg` · `jpeg` · `webp` |
| `width`, `height` | pixel size read back from the file; `None` for vector formats |
| `n_panels` | 1, or the panel count for a layout |
| `vmin`, `vmax` | the resolved colour range; `None` for a trace-only figure |
| `size_bytes` | size of the encoded figure |
| `saved` | property — `path is not None` |

| Call | Does |
|---|---|
| `read()` | the bytes, from memory or from the saved file |
| `save(path)` | write after the fact; returns the path |
| `show()` | display it explicitly, like `plt.show()` — inline in a notebook, else the OS image viewer; prints the path on a headless box, never raises. Returns `None`; does NOT save |

### `draw(spec, slug="figure", …)`

Takes an `Image`, a `Trace`, a `Video`, or a **list** of `Image`/`Trace` for a multi-panel layout.
Destination: `path=` or the `media/` keywords (`question=`/`bulk=`/`date=`/`root=`); with none of
them the figure is returned in memory and displays inline. Also accepts `format`, `frame`
(`Video` only), `figsize`, `dpi`, `tight`, `title`, `colorbar`, `show_time`, `units`,
`transparent`, `resolution`/`fit` (bare producer only), and `labels`/`rows`/`cols` (lists only).

## 11. `CardiacMeshData`

```python
mesh = cc.create_cardiac_mesh(Lx, Ly, dx, D=1.4, chi=1400.0)   # lengths first, THEN spacing (cm)
cc.save_cardiac_mesh(path, mesh);  mesh = cc.load_cardiac_mesh(path)
sim  = cc.monodomain(mesh)
```

**Attributes**

| Access | Meaning |
|---|---|
| `mesh.dx`, `mesh.dy` | spacing, **cm** |
| `mesh.mask` | tissue mask |
| `mesh.D_xx`, `mesh.D_yy`, `mesh.D_xy` | per-node diffusivity tensor (**raw** — pre χ·Cm) |
| `mesh.chi`, `mesh.Cm` | **cm⁻¹**, **µF/cm²** |
| `mesh.ionic_model` | model NAME (a string) |
| `mesh.dt` | timestep, **ms** |
| `mesh.stimuli` | list of normalized stimulus dicts |
| `mesh.sigma_i`, `mesh.sigma_e` | bidomain conductivity field tuples |
| `mesh.group_labels`, `mesh.group_cell_types` | region tags and their cell types |
| `mesh.boundary` | `"insulated"` / `"bath"` |

**Methods** — none.

> **Argument-order trap.** `Grid(Nx, Ny, dx)` leads with **node counts**; `create_cardiac_mesh(Lx, Ly, dx)`
> leads with **lengths in cm**. Both end in `dx`. Tied by `L = (N-1)·dx`.

---

## 12. `Distribution` and `SimulationSnapshot`

### `Distribution`

`cc.Distribution(kind, **kwargs)` — a per-node random parameter spec, accepted by
`set_conductivity` / `scale_conductivity`.

**Attributes** — `d.kind` and the kind-specific kwargs.

**Methods**

| Call | Does |
|---|---|
| `d.sample(shape, device="cpu", dtype=torch.float64)` | draw a `shape` tensor of values |

### `SimulationSnapshot`

One frame yielded by `sim.snapshots(...)`.

**Attributes**

| Access | Meaning |
|---|---|
| `snap.t` | frame time, **ms** |
| `snap.Vm` (alias property `snap.V`) | `(Nx, Ny)` voltage |
| `snap.phi_e` | `(Nx, Ny)`, or `None` |
| `snap.Nx`, `snap.Ny`, `snap.dx`, `snap.dy` | geometry |
| `snap.ionic_states` | if requested via `record=` |

**Methods** — none.

---

## Free functions

- **Analysis** — `activation_time` `max_dvdt_time` `conduction_velocity` `cv_between` `radial_cv` `apd_at`
  `apd_map` `apd_per_beat` `restitution_curve` `restitution_slope` `wavelength` `di` `dominant_frequency`
  `dominant_frequency_map` `phase_map` `phase_singularities` `wavefront_mask`
- **Geometry masks** — `circle_mask` `rectangle_mask` `annulus_mask` `left_edge_mask` `right_edge_mask`
  `top_edge_mask` `bottom_edge_mask` `point_distance` `boundary_distance` `fiber_field_uniform`
  `fiber_field_transmural`
- **Protocols (these RUN simulations)** — `erp(...)` · `erp_proxy(apd90)` · `post_repol_refractoriness(erp, apd90)`
- **0-D / safety** — `single_cell(...)` · `safety_factor(r, ...)` · `threshold_charge(...)`
- **One-shot runners** — `simulate(mesh, t_end, ..., engine=...)` · `run_monodomain` / `run_bidomain` / `run_lbm`
- **I/O** — `save_result` · `load_result` · `save_cardiac_mesh` · `load_cardiac_mesh` · `create_cardiac_mesh`
- **Figures & video** — `propagation_video` · `apd_map_figure` · `activation_isochrones` · `render` / `render_video`

---

## Units

| Quantity | Where | Unit |
|---|---|---|
| Spacing / lengths | `dx`, `dy`, `Lx`, `Ly`, stim `width`/`radius`, point `(x,y)` | **cm** |
| Node counts | `Nx`, `Ny` | integer count |
| Time | `dt`, `start_time`, `duration`, `t_end`, `save_every`, `bcl` | **ms** |
| Voltage | `Vm`, `clamp=` | **mV** |
| Stimulus current | `amplitude` | **µA/µF** (negative = depolarizing) |
| Diffusivity | `D_eff` | **cm²/ms** |
| Conductivity | `sigma_i`, `sigma_e`, `sigma_eff` | **mS/cm** |
| Surface-to-volume | `chi` | **cm⁻¹** |
| Capacitance | `Cm` | **µF/cm²** |
| Conduction velocity | `r.cv()` | **cm/s** |
| Fiber angle | `fiber_angle` | **radians** |

`float64` on CPU by default; pass `device="cuda"` to the factories. The **Optimizer** is the only subsystem
that speaks millimetres (`dx_mm`, `domain_mm`) and converts at its edge.
