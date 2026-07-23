# PLAN: cardiac_core `Stim` object — mask-first stimulus, non-breaking, → canonical

Created: 2026-07-22
Engine(s): cardiac_core (construction/analysis layer; no solver changes)
Research question: [engine_consolidation](../Research/Active/engine_consolidation/README.md)
Source: [engine_consolidation IDEALOG](../Research/Active/engine_consolidation/IDEALOG.md) — the
"Stim-as-object (DESIGN LOCKED 2026-07-22)" bullet.

> **Design LOCKED (user, 2026-07-22).** Mask-first public `Stim` (+`region=` callable convenience); one fixed mask
> per Stim; **non-breaking COEXISTENCE** (`Stim | dict | list[either]` everywhere the dict is accepted today, dict
> soft-deprecated) that **steers toward `Stim` as the final canonical form**; **built-in location presets**
> (`Stim.boundary(grid, side)` / `Stim.point/center/from_region`, eager classmethod factories). Depth = FRONT-DOOR: `Stim` lowers to the
> existing normalized-dict at `_normalize_stimulus`, so the `.npz` format + engines + `CardiacMeshData.stimuli` are
> UNCHANGED. Plus TWO modes: current injection + voltage clamp (mono/bidomain hard-write `v[mask]=value`; a NEW
> additive native LBM clamp `f[:,mask]+=w·(value−Σf)`). **This is a PLAN — implementation is a separate, explicit user
> go (hard gate).**
>
> **AUDIT: CONVERGED (2026-07-22, 3 rounds, Opus adversarial, code-verified).** R1 1B/6H/8L → R2 1B/3H/6L →
> **R3 0B/0H/5L (verdict: converged)**. R1 caught the clamp-routing under-spec (clamp silently injected as current);
> R2 caught the fixes' tail (`add_clamp_protocol` signature, LBM clamp lifecycle — mask device, `reset()` drop,
> `self.V` re-sync); R3 confirmed 0 blocker/0 major and only localized minors, all applied (`_resolve_where` delegates
> to `geometry.py`; overlap-sum scoped mono/bidomain — LBM overwrites; `_end` None-safe; field-value cast; stale
> `*_edge` purged). Every fix code-verified against `api.py`/`_lbm/simulation.py`/`stimulus/protocol.py`.

## Objective
Promote stimulus from an opaque dict/lambda to a public, mask-first `Stim` object that is serializable,
inspectable, composable, and visualizable — with built-in presets for common stimulation sites. Land it
**without breaking** the existing dict path (~19 cardiac_core tests + Surrogate/Optimizer/MCP/Lab consumers +
the `.npz` serialization all keep working), then steer cardiac_core + each consumer toward `Stim` over later,
independently-deliverable phases. `Stim` carries BOTH modes — **current injection** (the stimulus protocol) and
**voltage clamp** (mono/bidomain's existing clamp + a NEW native LBM clamp, Step 1.4). **Solver changes are limited to
the additive, opt-in LBM clamp** — a no-clamp run is byte-identical, so the per-engine integrity goldens stay atol=0.

## Success Criteria
- [ ] A public `Stim` dataclass: `mask:(Nx,Ny)bool` primary (+ `region=` callable resolved at build),
      `amplitude/start_time/duration/bcl/num_pulses/label`; **one fixed mask** (multi-site = a list of `Stim`s,
      overlaps sum — the existing `StimulusProtocol.get_current` behaviour).
- [ ] **EAGER-only, classmethod factory constructors** (NOT subclasses — one `Stim` type): the primary API is
      `Stim.boundary(grid, side, **params)` (side∈left/right/top/bottom — the sole edge API), plus
      `Stim.point(grid, pt, **params)`, `Stim.center(grid, **params)`, `Stim.from_region(grid, callable, **params)`;
      `Stim(mask, **params)` is the base. `width/radius` default to a thin strip so `Stim.boundary(grid,"left")` needs
      no other args. (`top/bottom_edge_mask` added to `geometry.py` as ground-truth for tests + other consumers.)
- [ ] **Coexistence:** the 3 declarative factories `monodomain()/bidomain()/lbm()` + `CardiacSimulation.stimulate()` accept
      `Stim | dict | list[either]`; the dict path is UNCHANGED behaviourally (soft-deprecated only).
- [ ] `Stim ⇄ dict` lowering at `_normalize_stimulus` — the `.npz` format, `CardiacMeshData.stimuli`, and the
      `_build_stimulus_protocol_*` engines are byte-for-byte unchanged (a `Stim` run == the equivalent dict run).
- [ ] `Stim` is the DOCUMENTED canonical form (cheatsheet + docstrings); the dict path carries a soft
      `DeprecationWarning`.
- [ ] Per-consumer migrations (Surrogate/Optimizer/MCP/Lab) are each independently deliverable + test-gated
      (Phase 3) — NOT a blocker for the feature (coexistence lets them migrate at their own pace).
- [ ] **Two modes:** current injection (default, `amplitude` µA/µF → stimulus protocol) AND voltage clamp
      (`clamp=<mV>` → the clamp mechanism) — clamp works on mono, bidomain, AND lbm (a NEW native `f=w·V_clamp` LBM
      clamp; Step 1.4). Mode inferred from whether `clamp` is passed.
- [ ] All existing cardiac_core tests pass (no regressions); per-engine integrity goldens bit-identical (atol=0 —
      incl. the LBM golden: the clamp is opt-in, a no-clamp run is byte-identical).

## Architecture Changes
- NEW: `cardiac_core/stimulus/stim.py` — the public `Stim` dataclass + presets + `to_dict()`/`from_dict()`/`times()`.
  (Co-located with the internal `stimulus/protocol.py` it lowers onto.)
- MOD: `cardiac_core/geometry.py` — add `top_edge_mask`/`bottom_edge_mask` (mirror `left/right_edge_mask` on the
  y-axis); export both.
- MOD: `cardiac_core/api.py:1235` (`_normalize_stimulus`) — accept a CURRENT-mode `Stim | dict | list` (lower a Stim
  via `.to_dict()`; a dict resolves its region as today); RAISE on a clamp-mode Stim. The single seam for CURRENT
  stims — the 3 factories + `stimulate()` route through it. (Clamp stims do NOT flow here — see the factory split.)
- NEW: `cardiac_core/api.py` `_partition_stimulus(stimulus) -> (current, clamp)` — each factory splits the list;
  current → `_build_mesh_data`/`data.stimuli`; clamp → applied post-build via `sim.clamp_voltage` (single window).
- MOD: `cardiac_core/api.py:569` (`stimulate()`) — accept a `Stim` (current → `data.stimuli`; clamp → `clamp_voltage`).
- MOD: `cardiac_core/api.py:654` (`clamp_voltage`) — dispatch on `_engine_type`: LBM skips `_require_stateful`(686),
  casts the mask to torch on-device, stores the clamp on the WRAPPER (`_lbm_clamp`); `reset()`(327) re-pushes it
  (mono/bidomain path UNCHANGED). (`add_clamp_protocol` is NOT used — periodic VC is rejected at `Stim` construction.)
- NEW (Step 1.4): a native additive per-step voltage clamp in `cardiac_core/_lbm/simulation.py` (`LBMSimulation.step()`
  end: `f[:,mask]+=w·(value−Σf)`, re-sync `self.V`) + `set_clamp` on the engine — opt-in, LBM integrity golden atol=0.
- MOD: `cardiac_core/__init__.py` — export `Stim` (+ `top_edge_mask`/`bottom_edge_mask`) via the `_LAZY` map.
- MOD (Phase 2): cardiac_core tests/examples/`API_CHEATSHEET.md`/docstrings → `Stim`; add the dict soft-deprecation.
- MOD (Phase 3, per-consumer, optional): Surrogate datagen ×5, Optimizer tuner ×2, cardiac_mcp `core.py`, Lab ×3.

## Known Failures / gotchas
- **Do NOT rip out the dict path in one PR.** It reaches ~19 cardiac_core tests + Surrogate datagen ×5 + Optimizer
  tuner ×2 + cardiac_mcp + Lab ×3 + the `.npz` serialization. A big-bang removal breaks live cross-project consumers
  — the exact pattern the consolidation track always defers (the ionic migration is per-consumer for this reason).
  Coexist; migrate per-consumer; keep the dict form working.
- **Keep the `.npz`/`CardiacMeshData.stimuli` format UNCHANGED.** The lowering target is the existing normalized dict
  (`{mask, label, amplitude, duration, start_time, bcl, num_pulses}` — file_format.py:102-143). `Stim.to_dict()` MUST
  emit exactly that shape (numpy bool mask) so save/load + `_build_stimulus_protocol_*` are untouched.
- **`_normalize_stimulus` already resolves callables + numpy-casts the mask** (api.py:1251-1258). A `Stim` arrives with
  the mask ALREADY resolved (it built it from a Grid) — do not re-evaluate a callable on a Stim; just `.to_dict()` it.
- **Overlap semantics differ by engine (existing behaviour — do not change):** mono/bidomain `StimulusProtocol.get_current`
  ADDS overlapping amplitudes (`Istim[mask] += amplitude`, protocol.py:183); **LBM OVERWRITES** (`I_stim[s.mask] =
  s.amplitude`, simulation.py:204 — last-writer-wins). Scope the overlap-SUM test to mono/bidomain; note the LBM caveat.
- **The mask is grid-bound.** A `Stim` built from `Stim.boundary(grid, …)` carries a concrete `(Nx,Ny)` mask; it is
  NOT reusable on a different-sized grid. Document this (the `region=` callable form stays grid-agnostic for the rare
  reuse case). The factory must validate a passed `Stim.mask` shape == the grid `(Nx,Ny)` and raise a clear error.
- **`top/bottom` axis convention.** In the `ij` grid, x=axis-0 (`left/right`), y=axis-1. Define `top_edge`=high-y,
  `bottom_edge`=low-y; DOCUMENT it in the docstrings (users will otherwise guess).
- **A `DeprecationWarning` on the dict path must land AFTER cardiac_core's own tests migrate** (Phase 2 orders this),
  or `filterwarnings('error')` / the warning-as-error tests break on the ~19 dict-using tests.
- **No solver changes on EXISTING paths.** The only solver touch is the additive, opt-in LBM clamp (Step 1.4, a NEW
  branch in `LBMSimulation.step()`); a no-clamp run is byte-identical. If any `test_integrity.py` golden moves, STOP.
- **`stimulus/regions.py` already exists** (callable region helpers `rectangular_region`/`circular_region`/
  `left_edge_region`/`point_stimulus`) — a PARALLEL region system to the `geometry.py` mask builders the presets use.
  Reconcile: the `Stim` presets should build on ONE of them (prefer the `geometry.py` mask builders, which return the
  `(Nx,Ny)` bool the mask path needs); do NOT silently create a third. Document the choice in `stim.py`.
- **Naming: `Stim` (public) vs the co-located internal `Stimulus`** (`stimulus/protocol.py:15`) are confusingly
  similar. `Stim` is the public front door; `Stimulus`/`StimulusProtocol` stay internal (the engine lowering target).
  Note the distinction in both docstrings; do not rename `Stimulus` (it is used across the engines).
- **`simulate()` (run.py) does NOT accept a Stim** — it is mesh-only (its `stimulus=` kwarg is dropped on the mesh
  path, pre-existing). Do not claim it; the Stim entry points are the 3 declarative factories + `stimulate()`.

---

## Phase 1: The `Stim` object + presets + coexistence (front-door)

**Goal**: a public mask-first `Stim` with named-constructor presets and BOTH modes (current injection + voltage clamp),
accepted everywhere the dict is; current-mode lowers to the unchanged internal dict, clamp-mode routes to the existing
clamp mechanism — non-breaking. Independently deliverable (delivers the whole user-facing feature).
**Tier**: large
**Estimated scope**: 1 new module + 2 geometry helpers + normalizer extension + clamp routing + `stimulate()` + tests.

### Phase Context
The stimulus flow today: public `stimulus=` (dict | list[dict]) → `_normalize_stimulus(stimulus, coords)`
(api.py:1235; resolves a `region` callable against the grid coords, casts the mask to numpy bool, emits the
normalized dict `{mask,label,amplitude,duration,start_time,bcl,num_pulses}`) → stored on `CardiacMeshData.stimuli`
→ serialized in `.npz` (file_format.py:102-143) → lowered to the engine `StimulusProtocol` by
`_build_stimulus_protocol_v54`/`_bidomain` (api.py:1395/1435; expands `bcl`/`num_pulses` into per-pulse
`add_stimulus`) and the LBM factory's OWN stim loop (api.py:1917-1932). `stimulus/protocol.py` holds `Stimulus` (region can be a
mask; `is_active`, `get_mask`) + `StimulusProtocol` (list; `get_current` SUMS amplitudes on overlap;
`add_regular_pacing`/`add_s1s2_protocol`). Geometry helpers (`geometry.py`) return `(Nx,Ny)` numpy bool masks;
`left_edge_mask(Nx,Ny,dx,width)` / `right_edge_mask(...)` exist, top/bottom do NOT. **The ONLY coexistence point we
need is `_normalize_stimulus`** — extend it to accept `Stim`, and every factory + `stimulate()` inherits it. Keep
everything downstream (mesh, `.npz`, engines) byte-identical. **No solver changes.**

### Step 1.1: `top_edge_mask` + `bottom_edge_mask` in `geometry.py`
**Model**: sonnet (trivial)
#### Read First
- `cardiac_core/geometry.py` — `left_edge_mask`/`right_edge_mask` (mirror them on the y-axis).
- `cardiac_core/__init__.py` — the `_LAZY` geometry exports block (add the two names).
#### Why
`Stim.boundary(grid, "top"/"bottom")` needs y-axis edge masks; only x-axis (`left/right_edge_mask`) exist. Symmetry.
#### Implementation Spec
**Files to modify:** `geometry.py`, `__init__.py`.
**Signatures:** `top_edge_mask(Nx, Ny, dx, width, dy=None) -> np.ndarray`; `bottom_edge_mask(...)` same.
Convention: `top` = high y (`y > Ly - width`), `bottom` = low y (`y < width`), `Ly = (Ny-1)*dy`. `dy` defaults to `dx`.
#### Pseudocode
```
dy = dx if dy is None else dy
y = arange(Ny)*dy ; Ly = y[-1]
m = zeros((Nx,Ny), bool)
top:    m[:, y > Ly - width] = True
bottom: m[:, y < width]      = True
return m
```
#### Test Spec
- `test_geometry.py::test_top_bottom_edge_masks` — `top_edge_mask(10,10,0.1,0.15)` selects the 2 highest-y columns
  for ALL x; `bottom` selects the lowest; disjoint; shapes `(10,10)`; dtype bool.
#### Checklist
- [ ] Two functions; `dy` defaults to `dx`; documented axis convention.
- [ ] Added to `_LAZY` geometry exports.
#### Verify
`conda run -n heart-conduction python -m pytest cardiac_core/tests/test_geometry.py -q -k edge`
#### Exit Criteria
- [ ] Both masks correct + exported.
#### Risk
Axis confusion (x vs y) — the test pins high-y=top. Mitigation: assert exact columns.

### Step 1.2: the `Stim` dataclass + presets + `to_dict`/`from_dict`/`times`
**Model**: opus
#### Read First
- `cardiac_core/stimulus/protocol.py` (full) — `Stimulus`/`StimulusProtocol`; `Stim` LOWERS onto this, do not
  duplicate the engine logic.
- `cardiac_core/api.py:1259-1267` — the EXACT normalized-dict shape `Stim.to_dict()` must emit.
- `cardiac_core/file_format.py:102-143` — the `.npz` stim fields (same shape — do not diverge).
- `cardiac_core/geometry.py` — the mask builders the presets call.
#### Why
`Stim` is the public front door. It must (a) carry an explicit `(Nx,Ny)` mask, (b) offer ergonomic presets over a
`Grid`, and (c) lower to the byte-identical internal dict so nothing downstream changes.
#### Implementation Spec
**Files to create:** `cardiac_core/stimulus/stim.py`.
**Signatures:**
```
class Stim:
    # A resolved stimulus = an explicit (Nx,Ny) mask + timing params. EAGER-ONLY (no deferred). Build it with a
    # NAMED CONSTRUCTOR below (grid + a location + any timing params) or directly from a mask. These are @classmethod
    # FACTORY constructors on ONE class (the datetime.fromtimestamp pattern), NOT subclasses: once the mask is
    # resolved every Stim is structurally identical (mask + params), so a boundary/point/center type hierarchy would
    # add nothing.
    def __init__(self, mask, *, amplitude=None, clamp=None, start_time=0.0, duration=2.0,
                 bcl=0.0, num_pulses=1, label="stim"):
        # MODE inferred: clamp=<mV> ⇒ VOLTAGE CLAMP (hold V=clamp on the mask); else CURRENT injection
        # (amplitude µA/µF, default −52). Both are a masked, timed condition — they differ only in what they
        # impose and which engine mechanism they lower to (current → stimulus protocol / Istim; clamp → the
        # per-step clamp mechanism — mono/bidomain hard-write `v[mask]=value`, LBM native additive
        # `f[:,mask]+=w·(value−Σf)` (holds Σf=value, preserves the flux f^neq), all three engines, Step 1.4).
        if clamp is not None and amplitude is not None:
            raise ValueError("pass amplitude (current) OR clamp (voltage), not both — the mode is inferred")
        if clamp is not None and (bcl > 0 or num_pulses > 1):
            raise ValueError("periodic pacing (bcl/num_pulses) is not supported for a clamp Stim — the clamp "
                             "mechanism holds ONE window; for repeated clamps pass one clamp Stim per window")
        self.mask = _as_bool_mask(mask)          # concrete (Nx,Ny) bool (accepts torch or numpy) — a small helper
        self.mode = "clamp" if clamp is not None else "current"
        self.amplitude = -52.0 if amplitude is None else amplitude
        self.clamp = clamp
        self.start_time, self.duration = start_time, duration
        self.bcl, self.num_pulses, self.label = bcl, num_pulses, label
    # --- named constructors: grid + a location; each a FULL constructor (pass bcl/amplitude/… via **kw) ---
    @classmethod
    def boundary(cls, grid, side, *, width=None, **kw) -> "Stim"   # side ∈ "left"/"right"/"top"/"bottom" (the edge API)
    @classmethod
    def point(cls, grid, center, *, radius=None, **kw) -> "Stim"   # a blob at an (x,y) cm point
    @classmethod
    def center(cls, grid, *, radius=None, **kw) -> "Stim"          # a blob at the domain centre
    @classmethod
    def from_region(cls, grid, region, **kw) -> "Stim"            # any callable (x,y)->bool mask
    # --- lowering / helpers ---
    def to_dict(self) -> dict          # CURRENT-mode only → {mask(np bool),label,amplitude,duration,start_time,bcl,
                                       # num_pulses}; RAISES on a clamp Stim (clamp routes to clamp_voltage, Step 1.4)
    @classmethod
    def from_dict(cls, d) -> "Stim"    # inverse of to_dict (current-mode)
    def times(self) -> list[float]     # [start_time + k*bcl for k in range(num_pulses)] (or [start_time])
    def n_nodes(self) -> int           # mask.sum() — quick inspectability
```
**EAGER-ONLY, classmethod factory constructors** (user: "i like the .boundary stuff"; the deferred/grid-free path is
SCRAPPED). The primary API is `Stim.boundary(grid, "left", bcl=1000, num_pulses=5, amplitude=-52)` — a full
constructor: grid + side + any timing params via `**kw`. Likewise `Stim.point(grid, (x,y), **kw)`,
`Stim.center(grid, **kw)`, `Stim.from_region(grid, callable, **kw)`; `Stim(mask, **kw)` is the base for an
already-built explicit mask. Each classmethod builds the concrete `(Nx,Ny)` mask via `_resolve_where(grid, where,
width, radius)`, which **DELEGATES to the `geometry.py` builders** (`{left,right,top,bottom}_edge_mask`, `circle_mask`)
— ONE mask system (do NOT reimplement the edge rule inline; that would be a third system and could go off-by-one vs
the strict-`<` geometry convention) — then returns `cls(mask, **kw)`.
`width/radius=None` → thin strip (`~2*dx`) so `Stim.boundary(grid, "left")` needs no other args. **NOT subclasses —
one `Stim` type** (a resolved boundary vs point differ only in their mask). A Stim is always self-contained (mask
present), so `to_dict()` needs no coords and `_normalize_stimulus` just lowers it — **NO deferred path, NO
`.on()`/`.resolve(coords)`**. Validate: `side ∈ {left,right,top,bottom}`; explicit mask is 2-D bool `(Nx,Ny)`.
`Stim.boundary` is the sole edge API (no `*_edge`).
**Two MODES, one object** (`clamp=<mV>` ⇒ voltage clamp, else current injection): a current-mode Stim lowers to the
stimulus protocol (`data.stimuli`/`Istim`) as above; a clamp-mode Stim is NOT lowered here — it routes to the per-step
CLAMP mechanism (Step 1.4). `to_dict()` is CURRENT-mode only (the 7-key dict); a clamp Stim is applied to the sim, not
serialized (`.npz` clamp persistence is a follow-on). `_normalize_stimulus` only handles current-mode Stims.
#### Pseudocode
```
def to_dict(self):                                     # CURRENT-mode only — the exact 7-key normalized dict
    if self.mode == "clamp":
        raise ValueError("to_dict() is for current-mode Stims; a clamp Stim routes to clamp_voltage (Step 1.4)")
    m = self.mask; m = m.cpu().numpy() if hasattr(m,'cpu') else np.asarray(m)
    return {'mask': m.astype(bool), 'label': self.label, 'amplitude': float(self.amplitude),
            'duration': float(self.duration), 'start_time': float(self.start_time),
            'bcl': float(self.bcl), 'num_pulses': int(self.num_pulses)}

@classmethod
def boundary(cls, grid, side, *, width=None, **kw):    # the SOLE edge API (no *_edge classmethods)
    return cls(_resolve_where(grid, side, width=width), **kw)   # _resolve_where builds the (Nx,Ny) mask
# point/center: cls(_resolve_where(grid, center_or_"center", radius=...)); from_region: pass a callable.
# _resolve_where(grid, where, width=None, radius=None) DELEGATES to geometry.py (ONE system, matches strict-< edges):
#   "left"/"right"/"top"/"bottom" → geometry.{left,right,top,bottom}_edge_mask(grid.Nx,grid.Ny,grid.dx[,grid.dy],width)
#   "center"/(x,y) point → geometry.circle_mask(...); callable → where(*grid.coordinates); mask → passthrough (validated).
#   width/radius None → ~2*dx.
```
#### Test Spec
- `test_stim.py::test_boundary_constructor` — `Stim.boundary(grid, "left", bcl=1000, num_pulses=3)` → a concrete
  leftmost-strip mask (cross-check vs `left_edge_mask`), `to_dict()` numpy bool, `.times()==[0,1000,2000]`;
  `"right"/"top"/"bottom"` hit the right cols/rows; `width=None` → a non-empty thin strip.
- `::test_point_center_region` — `Stim.point(grid,(cx,cy))` a blob at that cm point; `Stim.center(grid)` a middle blob;
  `Stim(mask)` passes an explicit mask through; `Stim.from_region(grid, lambda x,y:x<width)` == `Stim.boundary(grid,"left")`.
- `::test_to_from_dict` — for a CURRENT Stim, `Stim.from_dict(s.to_dict())` reproduces `s` (mask + params, numpy bool);
  `to_dict()` on a CLAMP Stim RAISES (current-mode only).
- `::test_validation` — `Stim.boundary(grid, "north")` raises (bad side); a wrong-shape explicit mask raises;
  `Stim(mask, amplitude=-80, clamp=-85)` raises (mode ambiguous — pass one, not both).
- `::test_is_one_type` — `type(Stim.boundary(grid,"left")) is type(Stim.point(grid,(0,0))) is Stim` (no subclasses).
#### Checklist
- [ ] `__init__(mask, **params)` + classmethods `boundary/point/center/from_region` (each grid + location, EAGER).
- [ ] `_resolve_where(grid, where, width, radius)` DELEGATES to `geometry.py` builders (no third mask system); side/center/point/callable; thin-strip default (~2*dx); side validated.
- [ ] `to_dict()` (numpy bool, EXACT normalized shape) / `from_dict` / `times` / `n_nodes`. NO deferred / `.on` / `.resolve`.
#### Verify
`conda run -n heart-conduction python -m pytest cardiac_core/tests/test_stim.py -q`
#### Exit Criteria
- [ ] `Stim` + presets + lowering correct; no engine/mesh code touched yet.
#### Risk
Diverging `to_dict` from the normalized-dict shape → `.npz`/engine breakage later. Mitigation: assert equality against
a hand-built normalized dict in the test.

### Step 1.3: accept `Stim` at `_normalize_stimulus` + `stimulate()` + export; coexistence tests
**Model**: opus
#### Read First
- `cardiac_core/api.py:1235-1268` (`_normalize_stimulus`, incl. the INLINE dict-normalization at 1247-1267 — there is
  NO `_normalize_one_dict` helper; inline or extract it) + its callers (each factory `stimulus=` →
  `_build_mesh_data`:1333 → `_normalize_stimulus`); `api.py:569-590` (`stimulate()`); `api.py:1395`/`1435`
  (`_build_stimulus_protocol_v54`/`_bidomain`) + `api.py:1917-1932` (the LBM factory's OWN stim loop) — the downstream
  CURRENT consumers, MUST stay unchanged.
- `cardiac_core/__init__.py:58-63` (the `_LAZY` loader — dotted `'stimulus.stim'` DOES resolve via
  `importlib.import_module`, no shim needed); `cardiac_core/stimulus/__init__.py` (add `Stim` to its `__all__`).
- **NOTE: `simulate()` (run.py:313) is NOT a Stim entry point** — it is mesh-only; a positional `CardiacMeshData` is
  type-sniffed as `mesh=` → the `_resolve_mesh` branch, which SKIPS `_build_mesh_data`/`_normalize_stimulus`. Its
  `stimulus=` kwarg is dropped (pre-existing for dicts too). Stim entry points are the 3 declarative factories
  (`monodomain`/`bidomain`/`lbm`) + `stimulate()`. Do NOT claim `simulate()` accepts a Stim.
#### Why
`_normalize_stimulus` is the single seam through which the 3 declarative factories + `stimulate()` lower a
CURRENT-mode stimulus to `data.stimuli`; making it accept a `Stim` there makes all of them accept current Stims for
free, byte-identically. CLAMP-mode Stims do NOT lower here (they impose voltage, not current) — they are split out and
routed to `clamp_voltage` in Step 1.4; this step GUARDS `_normalize_stimulus` to REJECT a clamp Stim so one can never
silently land in `data.stimuli` as a bogus current.
#### Implementation Spec
Extend `_normalize_stimulus(stimulus, coords)` to accept `Stim | dict | list[either]`. A current-mode `Stim` carries a
resolved mask (eager-only), so it lowers via `.to_dict()` (no coords). A clamp-mode Stim reaching here is a routing bug
→ raise:
```
if stimulus is None: return []
if isinstance(stimulus, (Stim, dict)): stimulus = [stimulus]
for s in stimulus:
    if isinstance(s, Stim):
        if s.mode == "clamp":
            raise ValueError("a clamp-mode Stim must be applied via the factory's clamp routing (Step 1.4), "
                             "not lowered as a current — this is an internal routing error")
        d = s.to_dict()                              # the exact 7-key normalized dict
    else:
        d = _normalize_dict(s, coords)               # the EXISTING inline dict logic (api.py:1247-1267)
    out.append(d)
```
`stimulate()`: a current-mode `Stim` → append `stim.to_dict()` (mask `& data.mask`) + `reset()`; a clamp-mode Stim →
`self.clamp_voltage(...)` (Step 1.4); else the current `region`+params path. Export `Stim` in `_LAZY`
(`'Stim': 'stimulus.stim'`) AND add it to `stimulus/__init__.py.__all__`.
#### Pseudocode
```
# api.py — local import inside _normalize_stimulus / the factories to avoid an api↔stimulus.stim import cycle
from .stimulus.stim import Stim
# factory-level clamp split lives in Step 1.4 (_partition_stimulus); _normalize_stimulus only sees CURRENT stims.
```
#### Test Spec
- `test_stim.py::test_lbm_accepts_stim` — `lbm(g,'ttp06',cond, Stim.boundary(g,"left", bcl=1000, num_pulses=2))` runs;
  a probe captures 2 beats (mirror the verified 1 Hz check).
- `::test_all_engines_accept_stim` — mono + bidomain + lbm each accept a `Stim` and a `[Stim, Stim]` list.
- `::test_stim_equals_dict` — a `Stim` run and the EQUIVALENT dict run produce identical `Vm` (byte-identical LAT/CV)
  — proves the lowering is faithful.
- `::test_overlap_sums[mono|bidomain]` — two overlapping current `Stim`s → the overlap region gets SUMMED amplitude
  (stronger depolarization than a single stim). **mono/bidomain only** — LBM overwrites (last-writer-wins,
  simulation.py:204); a separate `test_overlap_lbm_overwrites` documents that.
- `::test_stimulate_accepts_stim` — `sim.stimulate(Stim.point(g, (0.2,0.2)))` (current-mode) works.
- `::test_clamp_stim_rejected_by_normalizer` — passing a clamp-mode `Stim` DIRECTLY to `_normalize_stimulus` raises
  (guards the "clamp silently lowered as current" blocker); the factory-level clamp routing is tested in Step 1.4.
- `::test_dict_path_unchanged` — the existing dict form still runs identically (regression).
- `::test_npz_roundtrip_with_stim` — build a mesh via a CURRENT `Stim`, `save_cardiac_mesh`/`load_cardiac_mesh`, rerun →
  identical (the `.npz` 7-key format is untouched).
#### Checklist
- [ ] `_normalize_stimulus` accepts current Stim/dict/list; RAISES on a clamp-mode Stim; downstream unchanged.
- [ ] `stimulate()` accepts a current Stim (clamp Stim → `clamp_voltage`, Step 1.4); dict path kept.
- [ ] `Stim` exported in `_LAZY` + `stimulus/__init__.__all__`; import-cycle-safe (local import).
- [ ] Coexistence + faithful-lowering + overlap + npz tests green; `simulate()` NOT claimed as a Stim entry.
#### Verify
`conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q -k "stim or normalize or construction or integrity"`
#### Exit Criteria
- [ ] `Stim` works on all engines + `stimulate()`; dict path unchanged; `.npz` unchanged; integrity goldens atol=0.
#### Risk
Import cycle (`api` ↔ `stimulus.stim` if `stim` imports `api`/`geometry` at module load). Mitigation: `stim.py`
imports only `geometry` (numpy) + torch; `api` imports `Stim` locally inside `_normalize_stimulus`.

### Step 1.4: voltage-clamp mode — factory routing + a NEW native LBM clamp (all three engines)
**Model**: opus  ·  **Tier note**: touches `_lbm/simulation.py` — additive/opt-in, so the LBM integrity golden MUST stay atol=0.
#### Read First
- **Clamp routing (the seam is NOT `_normalize_stimulus`):** the factories build `CardiacSimulation` at
  `api.py:1600`/`1781`/`1937` (mono/bidomain/lbm) AFTER `_build_mesh_data` lowers `stimulus=` → `data.stimuli`. A clamp
  must be applied to the built `sim` object, so it is SPLIT OUT before `_build_mesh_data` and applied after.
- mono/bidomain clamp (already exists): `api.py:654` (`clamp_voltage`), `:686` (`_require_stateful` — **RAISES
  `NotImplementedError` for LBM at :466-471**, must be bypassed for the LBM branch), `:688` (mono CASTS the mask to a
  torch bool on-device — mirror this), `:689` (`_clamp_mask`/`_clamp_value`), `:710` (`_apply_clamp`: `v[fm]=value`,
  resolves scalar/field/callable at :714-722), `:724` (`add_clamp_protocol(mask, steps:list[tuple], …)` — NOTE the
  signature; NOT used by this plan), `:1112`/`:1131-1133` (`_stepping_run` calls `_apply_clamp` at the END of each
  step, checking the POST-step `t`), `:304` (the `!=lbm` gate keeping LBM off `_stepping_run`), `:327-335` (`reset()`
  rebuilds `_engine` — MUST re-push an LBM clamp), `:443` (`_as_grid_mask` → numpy bool).
- LBM internals for the NEW clamp: **`_lbm/simulation.py:207-256` (`LBMSimulation.step()` — THE per-step loop with
  reaction: collide→stream→BC→recover_V→`ionic_step`@:251; `step.py` is stateless fused kernels, NOT the loop),** the
  reported/reused field `self.V` (saved by `_run_lbm` at `api.py:1183`, read by the next collision at
  `simulation.py:210/214`), `_lbm/collision/bgk.py:30` (`f_eq=w·V`), `_lbm/state.py:47/58` (`f=w·V_init`, `V=Σf`),
  the lattice weights `w` (confirm `Σw_i=1` for the D2Q5/D2Q9 sets used) + `recover_voltage` (state.py:57).
- an existing `clamp_voltage` test (test_advanced_features.py) — mirror its "V is held" assertion.
#### Why
A `clamp`-mode Stim imposes a VOLTAGE by a **HARD OVERRIDE, never via Istim**. It CANNOT go through `data.stimuli`
(that path is a current, default amplitude −52 — a clamp value there is silently wrong). Mono/bidomain already
hard-write `v[mask]=value` at the END of each step (`_apply_clamp`, api.py:720, via `_stepping_run`); the gates keep
integrating and V is re-pinned. LBM has NO stored V field (`V=Σf_i`), and the distribution `f_i=w_i·V+f_i^neq` carries
the FLUX in `f^neq`, so the LBM clamp forces `Σf=value` WHILE PRESERVING `f^neq` — additive `f[:,mask]+=w·(value−Σf)`,
NOT a pure reset `f=w·value` (zeros the flux) nor a rescale (V is signed → blows up at V≈0). We BUILD it rather than
raise. All three: "set V and let the sim run," never solving backward.
#### Implementation Spec
1. **Factory-level split (the routing seam):** add `_partition_stimulus(stimulus) -> (current, clamp)` — accepts
   `None | Stim | dict | list[either]` (normalize to a list; a `Stim` with `mode=="clamp"` → `clamp`, else → `current`).
   Each of the 3 factories, **ON THE DECLARATIVE PATH ONLY (`geometry` is a `Grid`)**, calls it at the top, passes ONLY
   `current` to `_build_mesh_data` (→ `data.stimuli`, Step 1.3), and — right before `return sim` — applies each clamp
   Stim: `sim.clamp_voltage(cs.mask, cs.clamp, start_time=cs.start_time, duration=cs.duration)` (ONE window;
   `bcl`/`num_pulses` are rejected at `Stim` construction, so NO `add_clamp_protocol` — the existing clamp API can't
   express a gapped periodic clamp, and `add_clamp_protocol(mask, steps:list[tuple], …)` (api.py:724) has a different
   signature). On the LEGACY `mesh=` path (positional `CardiacMeshData`), `stimulus=` is dropped (pre-existing, for
   BOTH current and clamp — do NOT apply clamp there, else clamp applies while a current Stim is silently ignored).
2. **`clamp_voltage` dispatch (LBM support):** branch on `self._engine_type`. mono/bidomain → the existing
   `_require_stateful`/`_clamp_mask` path (UNCHANGED). **LBM →** skip `_require_stateful`; CAST the mask to a torch
   bool tensor on `self._engine.device` (an `_as_grid_mask` numpy mask indexing the torch `f` would CUDA-crash — cf.
   the mono path api.py:688 and existing LBM masks simulation.py:204); STORE the clamp ON THE WRAPPER
   (`self._lbm_clamp = (mask_t, value, start, end)`, NOT only on the engine) and push it via `self._engine.set_clamp(*
   self._lbm_clamp)`. **`reset()` (api.py:327) must RE-PUSH `self._lbm_clamp`** to the freshly-rebuilt engine — else a
   clamp vanishes after `stimulate`/`scale_conductance`/`set_conductivity` (all call `reset()`), whereas the mono
   clamp survives on the wrapper. `value` may be scalar / `(Nx,Ny)` field / `callable(t)` (same as mono, api.py:714-722).
3. **LBM native clamp (additive, in `LBMSimulation.step()`):** add `self._clamp = None` in `LBMSimulation.__init__`
   (else `step()` AttributeErrors on step 1 — even the no-clamp golden run) + a `set_clamp(mask, value, start, end)`
   setter. At the END of `step()` — **AFTER `self.t += self.dt` (simulation.py:256)** so the window checks the POST-step
   `t`, matching mono (api.py:1131-1133) — while `start<=t<end`:
   ```
   val = value(self.t) if callable(value) else value              # scalar | (Nx,Ny) field (torch OR numpy) | callable(t)
   if not isinstance(val, (int, float)):                          # a field → per-node, cast to torch on the f device
       val = torch.as_tensor(val, dtype=self.f.dtype, device=self.f.device)[mask]
   self.f[:, mask] += self.w[:, None] * (val - self.f[:, mask].sum(dim=0))   # additive: Σf→val, f^neq preserved
   self.V = recover_voltage(self.f)                               # RE-SYNC the reported/next-step V (else stale!)
   ```
   Gates integrate at the re-pinned ≈`value` (the prior step re-pinned V) — same semantics as mono's end-of-step clamp.
4. **`stimulate()`** routes a clamp Stim to `self.clamp_voltage(...)` (current Stim → `data.stimuli`, Step 1.3).
5. **`to_dict()` is CURRENT-mode only** (the 7-key normalized dict for `data.stimuli`/`.npz`). A clamp Stim is NOT
   serialized to `data.stimuli` (it's applied to the sim), so it does NOT round-trip through `to_dict`/`.npz` — that
   is a documented follow-on (clamp persistence in `.npz`), NOT this step. Do NOT add `mode`/`clamp` to the 7-key dict.
#### Pseudocode
```
# api.py — factory (each of mono/bidomain/lbm): DECLARATIVE path only (geometry is a Grid)
current, clamp = _partition_stimulus(stimulus)          # (current list, clamp list); clamp = Stims with mode=='clamp'
data = _build_mesh_data(geometry, ionic, cond, current, dt, engine)   # only CURRENT → data.stimuli
...
sim = CardiacSimulation(engine, engine_type, grid, data, build_kwargs, boundary_mode=...)
for cs in clamp:                                        # (the mesh= path drops stimulus= entirely — no clamp there)
    sim.clamp_voltage(cs.mask, cs.clamp, start_time=cs.start_time, duration=cs.duration)   # ONE window per clamp Stim
return sim

# api.py — CardiacSimulation.clamp_voltage, dispatch:
def clamp_voltage(self, mask, value, start_time=None, duration=None):
    if self._engine_type == 'lbm':
        m = torch.as_tensor(self._as_grid_mask(mask), dtype=torch.bool, device=self._engine.device)  # torch, on device
        self._lbm_clamp = (m, value, start_time, _end(start_time, duration))   # store on WRAPPER → survives reset()
        self._engine.set_clamp(*self._lbm_clamp)
        return
    self._require_stateful("voltage clamp"); ...        # existing mono/bidomain path, UNCHANGED

# api.py — reset(): re-push the LBM clamp to the rebuilt engine
def reset(self):
    ... rebuild self._engine from self._data ...
    if self._engine_type == 'lbm' and getattr(self, '_lbm_clamp', None) is not None:
        self._engine.set_clamp(*self._lbm_clamp)

# _lbm/simulation.py: __init__ sets `self._clamp = None`; `set_clamp(mask,value,start,end)` stores it.
# LBMSimulation.step(), appended AFTER `self.t += self.dt`:
if self._clamp is not None:
    m, value, start, end = self._clamp
    if (start is None or self.t >= start) and (end is None or self.t < end):
        val  = value(self.t) if callable(value) else value
        vloc = val[m] if torch.is_tensor(val) and val.shape == self.V.shape else val
        self.f[:, m] += self.w[:, None] * (vloc - self.f[:, m].sum(dim=0))
        self.V = recover_voltage(self.f)
```
_Helpers to deliver: `_partition_stimulus(stimulus) -> (current, clamp)` (handles None/single/list); `_as_bool_mask(mask)`
(torch|numpy → `(Nx,Ny)` numpy bool, shape-validated); `_end(start, duration)` (= `(start or 0.0)+duration` if
`duration` else None — handle `start=None` like mono at api.py:692, else `None+X` TypeErrors)._
#### Test Spec
- `test_stim.py::test_clamp_holds_voltage[mono|bidomain|lbm]` — `<engine>(g,'ttp06',cond, Stim.boundary(g,"left",
  clamp=-20, start_time=0, duration=50))` → the left nodes sit near −20 mV during [0,50] ms, on ALL THREE engines.
- `::test_clamp_vs_current` — `Stim.boundary(g,"left")` (current) drives an AP; `Stim.boundary(g,"left", clamp=-85)`
  holds rest (no AP).
- `::test_mixed_current_and_clamp` — `[Stim.boundary(g,"left"), Stim.point(g,c, clamp=20)]` applies the current stim via
  the protocol AND the clamp via the clamp mechanism (both visible).
- `::test_lbm_clamp_matches_mono` — **the arbiter.** The SAME clamp on a small cable in LBM vs monodomain (mono's
  hard-write `v[mask]=val` is ground truth: it pins V on-node and lets current flow THROUGH — `in≠out`). Compute BOTH
  candidate LBM schemes and compare to mono: (B) additive `f+=w·(val−Σf)` (preserves f^neq → conducts through, on-node,
  O(h²)) should MATCH mono; (A) pure reset `f=w·val` (zeros f^neq → insulating flat point, O(h) value slip) should
  DEVIATE. Assert B tracks mono (V held + neighbour spread) and A does not → confirms B, retires A. Ship only B; this
  A/B is a one-time test-side comparison, not a shipped toggle.
- `::test_lbm_clamp_preserves_nonequilibrium` — after a clamp step, `Σf==value` to ~1e-12 AND `f^neq` at the clamped
  node is UNCHANGED from pre-clamp (additive), i.e. not forced to zero.
- `::test_clamp_stim_not_in_data_stimuli` — after `monodomain(g,'ttp06',cond, Stim.boundary(g,"left", clamp=-85))`,
  `sim._data.stimuli == []` (the clamp went to the clamp mechanism, NOT `data.stimuli` as a current) AND
  `sim._clamp_mask is not None` — guards the BLOCKER (clamp-as-current) directly.
- `::test_lbm_clamp_survives_reset` — an LBM clamp still holds V AFTER `sim.stimulate(...)` (which calls `reset()`) —
  guards the wrapper-storage/re-push fix.
- `::test_clamp_periodic_rejected` — `Stim.boundary(g,"left", clamp=-20, bcl=1000, num_pulses=5)` RAISES (periodic VC
  unsupported).
- `::test_lbm_clamp_cuda` (skipif no cuda) — an LBM clamp on `device='cuda'` holds V (guards the numpy-mask-indexing-a-
  CUDA-tensor crash).
- `::test_legacy_mesh_stim_dropped` — a `Stim` via `stimulus=` on the `mesh=` path is dropped for BOTH current and
  clamp (no asymmetric clamp application).
- **Regolden guard:** `test_integrity.py` LBM golden UNCHANGED (atol=0) — the clamp is opt-in; a no-clamp LBM run is
  byte-identical. If it moves, STOP.
#### Checklist
- [ ] `_partition_stimulus`; each factory (DECLARATIVE path) routes clamp → `sim.clamp_voltage` (single window,
  post-build), current → `data.stimuli`; `_normalize_stimulus` raises on a clamp Stim; `mesh=` path drops all `stimulus=`.
- [ ] `clamp_voltage` LBM branch: skip `_require_stateful`; mask cast to torch bool on-device; store on the WRAPPER
  (`_lbm_clamp`); `reset()` re-pushes it; scalar/field/callable value; mono/bidomain path UNCHANGED.
- [ ] `LBMSimulation.__init__` sets `self._clamp=None` + `set_clamp`; clamp additive `f[:,mask]+=w·(val−Σf)` AFTER
  `self.t+=self.dt`, `self.V` re-synced; opt-in (no clamp → byte-identical).
- [ ] `to_dict()` current-mode 7-key only; clamp+`bcl`/`num_pulses` rejected at `Stim` construction; LBM golden atol=0.
#### Verify
`conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q -k "clamp or integrity"`
#### Exit Criteria
- [ ] Clamp-mode Stims hold voltage on mono, bidomain, AND lbm (additive), survive `reset()`, work on cuda; clamp NEVER
  lands in `data.stimuli`; periodic-clamp rejected; current-mode path unchanged; LBM golden atol=0.
#### Risk
Failure modes (all now guarded): (a) clamp lowered as current (BLOCKER) → `_partition_stimulus` + `_normalize_stimulus`
raise + `test_clamp_stim_not_in_data_stimuli`; (b) `f` mutated without re-syncing `self.V` → `test_clamp_holds_voltage[lbm]`;
(c) numpy mask on a CUDA `f` → the torch-cast + `test_lbm_clamp_cuda`; (d) clamp dropped on `reset()` →
`test_lbm_clamp_survives_reset`; (e) `self._clamp` uninit → AttributeError even the golden → the `__init__` init;
(f) pure `f=w·value`/rescale → additive + `test_lbm_clamp_matches_mono`; (g) clamp before `ionic_step`/wrong `t` window →
after `self.t+=dt`; (h) any no-clamp step change → golden atol=0 guard.

### Phase 1 Verification
`conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q`
### Phase 1 Exit Criteria
- [ ] `Stim` + presets shipped; accepted on all engines + `stimulate()`; dict path + `.npz` byte-identical; integrity
  goldens atol=0; no existing test breaks.
### Phase 1 Cleanup
- float64/bool consistency; V5.3 untouched; `Stim` lowers onto the existing `Stimulus`/`StimulusProtocol` (no
  duplicated engine logic); no solver files touched.
**-> Commit point: git commit after Phase 1 passes.**

---

## Phase 2: Steer cardiac_core to `Stim` + soft-deprecate the dict path

**Goal**: `Stim` is the DOCUMENTED canonical form; cardiac_core's own tests/examples/cheatsheet use it; the dict path
warns (softly) but still works. Independently deliverable.
**Tier**: medium

### Phase Context
Coexistence (Phase 1) means this is optional polish that sets direction — nothing external breaks. **Ordering is
load-bearing:** migrate cardiac_core's OWN dict callers (tests + the INTERNAL `stimulate()`/`add_stimulus` dict) FIRST,
THEN turn on the `DeprecationWarning` — else the ~19 dict tests + any `filterwarnings('error')` trip, AND the warning
would fire on `stimulate()`'s own internal dict (a KEPT API, not deprecated). The warning must be a
`DeprecationWarning` (quiet by default) fired only where a USER passes a dict to a public factory.

### Step 2.1: migrate cardiac_core tests + the internal `stimulate()` dict + `API_CHEATSHEET.md` → `Stim`
**Model**: opus
#### Read First
- `api.py:569-590` (`stimulate()` — builds an internal dict via `_normalize_stimulus`; migrate it to construct a
  `Stim` internally so Step 2.2's warning won't fire on it); the ~19 dict-using test files (grep `'region'`/`start_time`).
#### Why
`stimulate()` calls `_normalize_stimulus` with a raw dict — if the deprecation lands there before this, every
`sim.stimulate(...)` warns/errors. Migrating the internal callers first makes Step 2.2 safe.
#### Implementation Spec
Mechanical `dict → Stim`/`Stim.<preset>` across the ~19 cardiac_core test files (KEEP a few dict-form cases under
`test_stim.py::test_dict_path_unchanged` as the back-compat guard). Rewrite `stimulate()`/`add_stimulus` to build a
`Stim` (current-mode) internally rather than a dict. Update `API_CHEATSHEET.md` §stimulus + the `lbm`/`monodomain`
examples to `Stim`; update `API_REFERENCE.md` if it exists.
#### Pseudocode — `{"region":r,"amplitude":a,...}` → `Stim.from_region(g, r, amplitude=a, ...)`; inside `stimulate()`:
`self._data.stimuli.append(Stim.from_region(self._grid, region, amplitude=…, …).to_dict()); self.reset()`.
#### Test Spec — full suite green after migration; the retained dict-form regression test still passes.
#### Checklist — [ ] ~19 test files migrated (few dict guards kept); [ ] `stimulate()`/`add_stimulus` build a Stim; [ ] cheatsheet updated.
#### Verify — `conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q`
#### Exit Criteria — [ ] suite green; no internal caller passes a raw dict to `_normalize_stimulus`.
#### Risk — a missed internal dict caller → Step 2.2's warning breaks it. Mitigation: grep every `_normalize_stimulus` caller before 2.2.

### Step 2.2: soft-deprecate the dict path + docstrings
**Model**: opus
#### Read First — `api.py:1235` (`_normalize_stimulus`, now only user dicts reach it); the factory `stimulus=` docstrings.
#### Why — steer users to `Stim` without breaking the dict path; the warning is the signal.
#### Implementation Spec — emit `warnings.warn("stimulus dicts are deprecated; use cardiac_core.Stim — dicts still work",
DeprecationWarning, stacklevel=…)` in `_normalize_stimulus` when a raw dict (not a Stim) is passed. Present `Stim` as
canonical in the factory docstrings + cheatsheet; dict = legacy.
#### Pseudocode — in `_normalize_stimulus`, per item: `if isinstance(s, dict): warnings.warn("stimulus dicts are
deprecated; use cardiac_core.Stim (dicts still work)", DeprecationWarning, stacklevel=3)` (Stims never warn).
#### Test Spec — `test_stim.py::test_dict_warns` — a dict to a factory raises `DeprecationWarning`; a `Stim` does not;
`sim.stimulate(Stim...)` does not (guards MAJOR-5).
#### Checklist — [ ] `DeprecationWarning` on user dicts only; [ ] docstrings/cheatsheet steer to Stim.
#### Verify — `conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q -k "dict_warns or stim"`
#### Exit Criteria — [ ] dict warns, Stim doesn't, `stimulate()` doesn't; suite green.
#### Risk — warning fires on an internal dict → breaks a kept API. Mitigation: Step 2.1 removed internal dicts; the `stimulate()` no-warn test.
**Phase 2 commit point.**

---

## Phase 3: Per-consumer migration (each independently deliverable, test-gated, OPTIONAL)

**Goal**: move the external consumers onto `Stim`. Coexistence means these are NON-blocking — schedule at will.
**Tier**: medium (per step)

### Phase Context
Each consumer has its OWN tests; migrate one per PR, gate on that consumer's suite, never break it. Do NOT touch a
consumer without running its tests. `Monodomain/Engine_V5.3/` stays read-only. These steps are follow-on and may be
deferred indefinitely (the dict path keeps working).

Each Phase-3 step shares this shape: **Read First** the consumer's dict-stimulus call sites (grep `'region'`/`stimulus=`)
+ its test suite · **Why** move it off the soft-deprecated dict onto `Stim` · **Impl** mechanical `dict → Stim`/preset
· **Test** the consumer's OWN suite green (never break a live consumer) · **Verify** run that suite · **Exit** consumer
green on `Stim`, no dict-stimulus left · **Risk** breaking a live consumer → gate on its suite, migrate one PR at a time.
(`Monodomain/Engine_V5.3/` stays read-only; `Surrogate/archive/scripts/run_datagen_cpu.py` is archived — EXCLUDE it.)

### Step 3.1: Surrogate datagen (×5) → `Stim`  ·  **Model**: opus
- `Surrogate/surrogate/data/{batch_generator,injection,protocols,single_cell_generator,ord_single_cell_generator}.py`.
- **Verify**: `conda run -n heart-conduction python -m pytest Surrogate/ -q` (Surrogate datagen tests) green.
### Step 3.2: Optimizer tuner (×2) → `Stim`  ·  **Model**: opus
- `Optimizer/V1/tuner/{presets,tissue_runner_bidomain}.py`.
- **Verify**: `conda run -n heart-conduction python -m pytest Optimizer/V1/tests -q` green.
### Step 3.3: cardiac_mcp `core.py` + Lab experiments (×3) → `Stim`  ·  **Model**: opus
- `cardiac_mcp/core.py` (+ any manifest/preset text); `Lab/2026-06-25_cv-strip-*/run.py`, `Lab/_validate/smoke.py`.
- **Verify**: `conda run -n heart-conduction python -m pytest cardiac_mcp/tests -q` green; Lab smoke runs.
**Each step is its own commit point.**

---

## Final Cleanup
- float64/bool consistency; V5.3 untouched; `Stim` lowers onto `Stimulus`/`StimulusProtocol` (no engine-logic dup);
  no solver files touched; per-engine integrity goldens bit-identical (atol=0).
- `API_CHEATSHEET.md` (+ `API_REFERENCE.md` if built) present `Stim` as canonical; `_LAZY` exports current.
- Archive this plan: `mkdir -p Research/Active/engine_consolidation/plans && cp cardiac_core/STIM_OBJECT_PLAN.md
  "Research/Active/engine_consolidation/plans/$(date +%Y-%m-%d)_stim-object.md"`.

## Mutation Log
- **MUTATED 2026-07-22**: Step 1.4 INSERTED a test edit — `test_advanced_features.py::test_lbm_clamp_and_injection_raise`
  asserted LBM `clamp_voltage` raises `NotImplementedError`. Step 1.4 makes the LBM clamp a supported native op, so
  that assertion is now stale. Renamed → `test_lbm_clamp_supported_but_injection_raises`: `clamp_voltage` no longer
  raises (registers `_lbm_clamp`), while `set_voltage` still raises (V is a lattice moment). Not a scope change —
  a foreseen consequence of adding the LBM clamp.

## Phase 1 — DONE (2026-07-22, commit c087b8c)
Steps 1.1–1.4 implemented on branch `stim-object`. `test_stim.py` (24) + `test_geometry` edge-mask pass;
`test_advanced_features` (14) pass after the mutation above; integrity goldens **atol=0** (opt-in clamp is
byte-identical on no-clamp runs). Full `cardiac_core/tests/` green except `test_video.py` (a PARALLEL cloud
session's in-progress video pipeline — `run.py`/`viz.py`/`video.py`/`test_video.py`, NOT this feature; left untouched).

## Phase 2 — DONE (2026-07-22)
Step 2.1: internal dict callers migrated to `Stim` — `stimulate()`/`add_stimulus` build a `Stim` internally
(no raw dict through `_normalize_stimulus`); `protocols.py::_captures` builds `Stim.from_region` S1/S2 trains;
the 11 dict-using cardiac_core test files migrated (module `STIM` const → `_stim(g)`; dict helpers → grid-threaded
`Stim.from_region`; test_usability_fixes keeps a dict `_stim()` for the non-warning mesh= path + a Stim `_gstim(g)`
for the Grid path); `API_CHEATSHEET.md` §3 + the §11 full-example + the §12 runnable-canary all present `Stim` as
canonical. Step 2.2: `_normalize_stimulus` emits a `DeprecationWarning` (stacklevel=4 → user's call) for a raw dict
ONLY; Stims/`stimulate()`/mesh= never warn. Guards: `test_stim.py::TestDeprecation::{test_dict_warns,
test_dict_path_unchanged}`. All 11 migrated files + canary pass under `-W error::DeprecationWarning` (141 passed);
integrity goldens still atol=0.

**Phase 3 (cross-project consumer migration) NOT started** — optional/deferrable, one-PR-per-consumer, gate on each
consumer's own suite. Checkpoint with the user before starting.
