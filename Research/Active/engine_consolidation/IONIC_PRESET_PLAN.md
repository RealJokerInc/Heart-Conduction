# PLAN: `IonicPreset` — a first-class, savable ionic-model config

Created: 2026-07-23
Engine(s): cardiac_core (all three engines, via the shared `build_ionic_model` registry seam)
Research question: [engine_consolidation](../Research/Active/engine_consolidation/README.md)
Source: design session 2026-07-23 (follow-on to the `Stim` object; the "no home for a tuned conductance set" gap)

## Objective
A first-class **`IonicPreset`** object: a named, savable ionic-model configuration — a base model plus a map of
parameter **scalings** — that (a) works for TTP06 and every registered model, (b) round-trips to/from JSON, and (c)
is accepted anywhere `ionic_model=` is. It closes the gap surfaced this session: a tuned conductance set has no
home — `scale_conductance` mutates the live model in memory, and `.npz`/`CardiacMeshData` stores only the model
NAME string, so a tuned TTP06 cannot be saved or reused as a unit.

## Locked design decisions (user, 2026-07-23)
- **Representation = BOTH.** `scalings` (`{param: factor}` relative to the base model's published defaults) is the
  CANONICAL, serialized form; resolved absolute values are exposed on demand via `.values` / `.resolve(cell_type)`.
  You may *input* either way (`scalings=` or `IonicPreset.from_values(...)`), but scalings is what's stored.
- **Breadth = ANY named parameter** (conductances **+** concentrations **+** kinetics). A scaling name is validated to
  EXIST on the model's `params`; it is NOT restricted by category. Multiplicative scaling only makes physical sense
  for magnitude params (G*/P*, concentrations, rates) — a small denylist (`gamma_ncx`, `pkna`, additive/shape/reversal
  params) **warns** (does not error) so a nonsensical scaling is loud, not silent.
- **Scope = CORE OBJECT ONLY.** This plan delivers the object + JSON save/load + factory acceptance through the single
  shared `build_ionic_model` seam. Explicitly **DEFERRED** (NOT this plan):
  - **`.npz` persistence of the scalings** — a preset-built sim currently saves its mesh with the BASE-NAME string
    (scalings dropped from the `.npz`, but PRESERVED across `reset()`/`with_()` via `build_kwargs`). A follow-on phase
    would extend the file format to carry the scalings. Documented limitation, not a bug.
  - **The Optimizer tuner bridge** (`from_tuning_result()`, reconciling the tuner's `g_Na` names ↔ cardiac_core's
    `GNa`) — entangles with the gated ionic-tuner redesign; a separate PR.
  - **Absolute-SET of arbitrary params on a LIVE sim** — `set_parameter` stays stubbed; a preset applies at BUILD only.
  - **Per-node heterogeneity** — uniform scalar scalings only (same limitation as `scale_conductance`).

## Success Criteria
- [ ] `IonicPreset("ttp06", {"GNa": 0.8})` — `.scalings`, `.values`, `.resolve(cell_type)`, `.build()`; `from_values`,
      `from_model`; `save`/`load` JSON round-trip; works for ttp06/ord/phas13/mhas13/paci.
- [ ] `cc.monodomain/bidomain/lbm(g, IonicPreset(...), cond, stim)` runs on ALL THREE engines; survives `reset()`/
      `with_()`; a preset-built sim's `.npz`-save uses the base-name string (documented Phase-2 limitation).
- [ ] All existing tests pass; integrity goldens **atol=0** (a preset is opt-in — the no-preset path is byte-identical).

## Architecture Changes
- **NEW** `cardiac_core/ionic/preset.py` — the `IonicPreset` class + `_apply_scalings` + `_resolve_values`.
- **MOD** `cardiac_core/ionic/registry.py` — `build_ionic_model` resolves an `IonicPreset` (function-level import,
  breaks the preset↔registry cycle); add `model_name(instance) -> str` (reverse type→name, for `from_model`).
- **MOD** `cardiac_core/ionic/__init__.py` (+ `__all__`) and `cardiac_core/__init__.py` `_LAZY` — export `IonicPreset`.
- **MOD** `cardiac_core/api.py` — `_build_mesh_data` stores the base-name string when `ionic_model` is a preset
  (keeps `data.ionic_model` a str → `.npz` back-compat); verify the factory passes the preset through as `ionic` so
  `build_kwargs` carries it (reset/with_ re-resolve). **MOD** `cardiac_core/single_cell.py:90` (guarded caller — route
  a non-`IonicModel` through `build_ionic_model`).
- **NEW** `cardiac_core/tests/test_ionic_preset.py`.
- **MOD docs** `cardiac_core/API_CHEATSHEET.md` + `Research/Active/engine_consolidation/API_REFERENCE.md`.

## Known Failures / traps (from the codebase)
- `build_ionic_model` is the single seam BUT some callers guard on `isinstance(model, str)` and pass a non-str THROUGH
  without resolving (`single_cell.py:90`, possibly `api.py:1957`) — a preset would leak past unresolved. Every caller
  must route a preset through the seam. AUDIT them (Step 1.2).
- preset↔registry **import cycle**: `preset.py` imports `build_ionic_model` at module load; `registry.py` must import
  `IonicPreset` at FUNCTION level (inside `build_ionic_model`), never module-level.
- `data.ionic_model` must stay a **string** (for `.npz`) — storing a preset object there breaks `save_cardiac_mesh`.
- cell-type dependence: EPI/ENDO/M change the published defaults, so `.values`/`.build` MUST resolve at a specific
  cell type — the preset OWNS its `cell_type` (a preset for EPI builds EPI regardless of the mesh default).

---

## Phase 1: the `IonicPreset` object (core)

**Goal**: a shippable, savable `IonicPreset` accepted anywhere `ionic_model=` is, on all three engines.
**Tier**: large (touches the shared ionic seam + all-engine acceptance; golden-guarded).

### Phase Context
Mirrors the `Stim` pattern (a public config object with classmethod constructors + `to_dict`/`from_dict` lowering,
coexisting with the existing string/instance paths — non-breaking). The resolution seam is
`cardiac_core/ionic/registry.py::build_ionic_model` (used by all three engines); a preset resolves there to a scaled
`IonicModel` instance, exactly like a tuner-scaled instance already does. `scalings` is canonical; absolutes derive.
Do NOT restrict scaling to conductances (breadth = any named param) — validate existence, warn on the denylist. Keep
`data.ionic_model` a string. A no-preset run must stay byte-identical (goldens atol=0).

### Step 1.1: the `IonicPreset` class + JSON save/load (self-contained)
**Model**: opus
#### Read First
- `cardiac_core/ionic/registry.py` (the whole file — `build_ionic_model(name, cell_type, device)`, the `_CELLTYPE_MODELS`
  / `_DEVICE_ONLY_MODELS` maps; a preset builds its base via this).
- `cardiac_core/ionic/scaling.py` (`scale_ionic_conductances` — the deep-copy + `setattr(params, name, val*factor)`
  firewall, shared by tissue `scale_conductance` and `single_cell(conductances=)`; `IonicPreset` generalizes its
  APPLY to any-named-param, and reuses its `_NON_CONDUCTANCE` set as the WARN denylist — do NOT hard-restrict to G/P).
- `cardiac_core/ionic/ttp06/model.py:74-106` (`TTP06Model.__init__` — `self.cell_type`, `self.params =
  get_celltype_parameters(cell_type)`, and the `param_overrides` loop — confirms params live as attributes on
  `self.params`). `cardiac_core/stimulus/stim.py` (mirror its `to_dict`/`from_dict`/`__repr__` shape + docstring density).
#### Why
A preset must be model-agnostic and depend only on introspection (`vars(model.params)`), so it works for
TTP06/ORd/PHAS13/MHAS13/paci without per-model code. Scalings are canonical because that is the tuner's idiom AND
`scale_conductance`'s, and because absolute values are cell-type-dependent (so they must be *resolved*, not stored).
#### Implementation Spec
**Create** `cardiac_core/ionic/preset.py`:
- `class IonicPreset(ionic_model: str, scalings: dict | None = None, *, cell_type: str = 'ENDO', label: str =
  'preset', meta: dict | None = None)` — store `ionic_model` (lower-cased base name), `scalings` (copy), `cell_type`
  (a string), `label`, `meta` (freeform provenance, e.g. tuner targets).
- `.build(cell_type=None, device='cpu') -> IonicModel` — `build_ionic_model(self.ionic_model, cell_type=cell_type or
  self.cell_type, device=device)` → a fresh base, then `_apply_scalings(model, self.scalings)` in place; return it.
- `.resolve(cell_type=None) -> dict` — `_resolve_values(base_params, self.scalings)` = `{n: default_n * f}` (validates
  each name exists). `.values` property = `.resolve()`.
- `@classmethod from_values(ionic_model, values, *, cell_type='ENDO', **kw)` — instantiate the base, convert each
  absolute to a scaling `v / default` (raise on an unknown name or a zero default), store scalings canonical.
- `@classmethod from_model(model, *, ionic_model=None, **kw)` — snapshot a LIVE/scaled model: `name =
  ionic_model or model_name(model)`; `ct = model.cell_type.name` (fallback 'ENDO'); build a fresh base at `ct`;
  `scalings = {n: cur/def for n in vars(base.params) if def and cur != def}` (numeric params only).
- `to_dict()` → `{'ionic_model', 'cell_type', 'scalings', 'label', 'meta'}`; `@classmethod from_dict(d)`; `save(path)`
  (`json.dump(indent=2)`); `@classmethod load(path)`; `__repr__` (model, n scalings, cell_type).
- module helpers: `_apply_scalings(model, scalings)` (validate `hasattr(params, name)` else raise listing
  `sorted(vars(params))`; WARN if `name.lower()` in the denylist `{'gamma_ncx','pkna'}`; `setattr(p, name, getattr(p,
  name)*float(factor))`); `_resolve_values(params, scalings)` (same validation, returns absolutes).
**Modify** `cardiac_core/ionic/registry.py`: add `model_name(instance) -> str` (reverse lookup over the two maps;
PHAS13Model → canonical `'phas13'`). **Modify** `cardiac_core/ionic/__init__.py` (import + `__all__`) and
`cardiac_core/__init__.py` `_LAZY` (`'IonicPreset': 'ionic.preset'`).
#### Pseudocode
```
# preset.py
from .registry import build_ionic_model, model_name          # module-level OK (registry imports preset FN-level)
_SCALE_WARN = {'gamma_ncx', 'pkna'}
def _apply_scalings(model, scalings):
    p = model.params
    for name, f in scalings.items():
        if not hasattr(p, name): raise ValueError(f"{name!r} not a param of {type(model).__name__}; "
                                                  f"available: {sorted(vars(p))}")
        if name.lower() in _SCALE_WARN: warnings.warn(f"scaling {name!r} multiplicatively is likely nonsensical")
        setattr(p, name, getattr(p, name) * float(f))
    return model
class IonicPreset:
    def __init__(self, ionic_model, scalings=None, *, cell_type='ENDO', label='preset', meta=None):
        self.ionic_model=ionic_model.lower(); self.scalings=dict(scalings or {}); self.cell_type=cell_type
        self.label=label; self.meta=dict(meta or {})
    def build(self, cell_type=None, device='cpu'):
        return _apply_scalings(build_ionic_model(self.ionic_model, cell_type=cell_type or self.cell_type,
                                                 device=device), self.scalings)
    def resolve(self, cell_type=None):
        p = build_ionic_model(self.ionic_model, cell_type=cell_type or self.cell_type).params
        return _resolve_values(p, self.scalings)
    @property
    def values(self): return self.resolve()
    # from_values / from_model / to_dict / from_dict / save / load / __repr__  (as Spec)
```
#### Test Spec — `cardiac_core/tests/test_ionic_preset.py`
- `test_build_scales_conductance` — `IonicPreset("ttp06", {"GNa": 0.8}).build()` has `params.GNa ==
  0.8 * TTP06Model(ENDO).params.GNa`; other params untouched; the base TTP06Model is UNMUTATED (fresh each build).
- `test_values_and_resolve` — `.values["GNa"] == default*0.8`; `.resolve("EPI")` uses EPI defaults (≠ ENDO for `Gto`).
- `test_from_values_roundtrip` — `from_values("ttp06", {"GNa": 11.87})` → `.scalings["GNa"] ≈ 11.87/default`; `.values
  ["GNa"] ≈ 11.87`.
- `test_from_model` — take a `scale_conductance`-scaled model (or a hand-scaled `TTP06Model`), `from_model(m)` recovers
  the scalings (factor within 1e-12); round-trips through `build()`.
- `test_save_load` — `save`→`load` reproduces `ionic_model`/`cell_type`/`scalings`/`meta`; JSON is plain (no numpy).
- `test_breadth_any_param` — scaling a CONCENTRATION (`"Ko"`/`"Nao"` — confirm the real attr name) and a kinetic param
  builds without error; an unknown name RAISES listing available params; a denylist name (`"gamma_ncx"` if present)
  WARNS.
- `test_other_models` — `IonicPreset("ord", {...}).build()` and `IonicPreset("phas13", {...}).build()` succeed
  (device-only models ignore cell_type).
#### Checklist
- [ ] `IonicPreset` (build/resolve/values/from_values/from_model/to_dict/from_dict/save/load/repr); `_apply_scalings`
  + `_resolve_values`; `registry.model_name`; exports (`cc.IonicPreset`, `cardiac_core.ionic.IonicPreset`).
- [ ] No hard category restriction (any existing param scalable); denylist WARNS only; unknown name RAISES.
- [ ] Fresh base every `build()` (never mutate a shared/cached model); cell_type owned by the preset.
#### Verify
`conda run -n heart-conduction python -m pytest cardiac_core/tests/test_ionic_preset.py -q`
#### Exit Criteria
- [ ] All Step-1.1 tests pass; `cc.IonicPreset` imports; JSON round-trips; works across ttp06/ord/phas13.
#### Risk
Import cycle (preset↔registry) → registry imports preset FUNCTION-level. Mutating a cached base model → build a fresh
instance each call. Concentration attr names differ per model (`Ko` vs `K_o`) → the test confirms the real name via
`vars(params)` rather than assuming.

### Step 1.2: factory acceptance — `build_ionic_model` resolves a preset (all engines) + `data.ionic_model` stays a str
**Model**: opus
#### Read First
- `cardiac_core/ionic/registry.py:38-45` (add the preset branch BEFORE the `isinstance(name, IonicModel)` check).
- Every `build_ionic_model` caller (grep): `cardiac_core/_monodomain/simulation/classical/monodomain.py:79,363`
  (mono — already routes any `name` through the seam → a preset flows through: GOOD), the bidomain + lbm equivalents
  (VERIFY each engine's ionic construction routes through `build_ionic_model`), `cardiac_core/single_cell.py:90`
  (`... if isinstance(model, str) else model` — a preset would leak past UNRESOLVED → FIX), `cardiac_core/api.py:1957`
  (read context — ensure a preset reaches the seam), `cardiac_core/_result_context.py:131` (record-only; check it
  doesn't choke on a non-string model identity).
- `cardiac_core/api.py:_build_mesh_data` (the `ionic_model=ionic_model or 'ttp06'` line — must store a STRING) and the
  three factories' `ionic = ionic_model or data.ionic_model` + `build_kwargs=dict(..., ionic_model=ionic)` (confirm a
  preset passed as `ionic_model=` becomes `ionic`, lands in `build_kwargs`, and thus survives `reset()`/`with_()`).
#### Why
One seam (`build_ionic_model`) resolves a preset to a scaled instance for all three engines — but only if every path
actually routes non-string ionic models through it. `data.ionic_model` must remain a string so `.npz` save is
unaffected (the scalings are intentionally NOT in the mesh yet — the deferred Phase-2 gap); the preset itself rides in
`build_kwargs`, so `reset()`/`with_()` rebuild it identically.
#### Implementation Spec
1. `registry.build_ionic_model`: at the top, `from .preset import IonicPreset` (function-level); `if isinstance(name,
   IonicPreset): return name.build(device=device)` (the preset owns cell_type; the caller's `cell_type` arg is ignored
   for a preset — document it).
2. `single_cell.py:90`: change to `m = model if isinstance(model, IonicModel) else build_ionic_model(model, celltype,
   device=device)` so a preset (or string) resolves; an instance passes through.
3. `api.py:1957` + bidomain/lbm ionic construction: ensure a preset reaches `build_ionic_model` (if a branch guards on
   `isinstance(..., str)`, widen it to also route a preset — or just always call the seam for a non-`IonicModel`).
4. `_build_mesh_data`: `data.ionic_model = ionic_model.ionic_model if isinstance(ionic_model, IonicPreset) else
   (ionic_model or 'ttp06')` — store the base-name string (add a local `from .ionic.preset import IonicPreset`).
   Leave the factory's `ionic = ionic_model or data.ionic_model` + `build_kwargs['ionic_model']=ionic` untouched
   (a preset is truthy → `ionic = preset`, carried in build_kwargs).
#### Pseudocode
```
# registry.build_ionic_model
def build_ionic_model(name, cell_type='ENDO', device='cuda'):
    from .preset import IonicPreset
    if isinstance(name, IonicPreset): return name.build(device=device)   # preset owns cell_type
    if isinstance(name, IonicModel):  return name
    ... existing string path ...
# api._build_mesh_data
from .ionic.preset import IonicPreset
data_ionic = ionic_model.ionic_model if isinstance(ionic_model, IonicPreset) else (ionic_model or 'ttp06')
# ...CardiacMeshData(ionic_model=data_ionic, ...)  ;  factory keeps ionic = ionic_model or data.ionic_model
```
#### Test Spec (append to `test_ionic_preset.py`)
- `test_preset_drives_sim[monodomain|bidomain|lbm]` — `engine(g, IonicPreset("ttp06", {"GNa": 0.6}), cond, stim)` runs;
  a strong `GNa` block MEASURABLY slows CV vs the unscaled `engine(g, "ttp06", ...)` (front reaches a probe later, or
  CV lower) — proves the scaling actually took effect through the seam on each engine.
- `test_preset_survives_reset` — after `sim.stimulate(...)` (calls `reset()`), the sim's live model still carries the
  scaling (`sim._live_ionic_model().params.GNa` == scaled) — guards the build_kwargs round-trip.
- `test_preset_npz_uses_base_name` — a preset-built mono sim: `sim._data.ionic_model == "ttp06"` (a string, not the
  preset) → `save_cardiac_mesh` works; document that the scalings are NOT in the `.npz` (Phase-2).
- `test_single_cell_accepts_preset` — `cc.single_cell(IonicPreset("ttp06", {"GNa":0.5}))` runs (guards the widened
  caller).
- **Regolden guard**: `test_integrity.py` (all three goldens) UNCHANGED (atol=0) — a preset is opt-in.
#### Checklist
- [ ] `build_ionic_model` resolves a preset (fn-level import); the guarded callers (`single_cell`, `api.py:1957`,
  bidomain/lbm) route a preset through the seam; `data.ionic_model` stays a string; reset/with_ carry the preset.
- [ ] goldens atol=0.
#### Verify
`conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q -k "preset or integrity"`
#### Exit Criteria
- [ ] A preset drives a sim on mono/bidomain/lbm (scaling visibly takes effect); survives reset; `.npz`-save uses the
  base name; goldens byte-identical.
#### Risk
A missed caller leaks a preset past the seam → an engine gets an `IonicPreset` where it expects a model → AttributeError.
Mitigation: the caller audit + the per-engine `test_preset_drives_sim`. A preset stored in `data.ionic_model` would
break `.npz` → the string-store guard + `test_preset_npz_uses_base_name`.

### Step 1.3: docs + full-suite verify + commit
**Model**: opus
#### Read First
- `cardiac_core/API_CHEATSHEET.md` §8 (heterogeneity — `scale_conductance` lives there) and the `Stim` §3 (mirror its
  "canonical object + legacy note" shape).
- `Research/Active/engine_consolidation/API_REFERENCE.md` (the `Stim` subsection added this session — mirror it for
  IonicPreset in the ionic-model section).
#### Why
The cheatsheet is the surface the Goal-2 skills generate against; a savable ionic config is exactly a wet-lab need
(drug-block / channelopathy presets). Keep it discoverable.
#### Implementation Spec
- `API_CHEATSHEET.md`: in §8 (or a short new bullet), add `cc.IonicPreset("ttp06", {"GNa": 0.8}).save("p.json")` /
  `cc.monodomain(g, cc.IonicPreset.load("p.json"), cond, stim)`; cross-note `scale_conductance` (live, in-memory) vs
  `IonicPreset` (savable, applied at build). Note the Phase-2 `.npz` limitation in one line.
- `API_REFERENCE.md`: an `### class IonicPreset [now]` subsection near the ionic-model section (constructor,
  `build/values/resolve/from_values/from_model/save/load`, the "scalings canonical, cell_type owned, `.npz` deferred"
  notes).
#### Test Spec — full `cardiac_core/tests/` green; goldens atol=0.
#### Checklist — [ ] cheatsheet + API_REFERENCE updated; [ ] full suite green; [ ] goldens atol=0.
#### Verify
`conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q`  (ignore the parallel session's
`test_video.py` if still un-merged)
#### Exit Criteria — [ ] docs present IonicPreset as canonical-for-a-saved-config; full suite green.
#### Risk — none (docs + verify).

### Phase 1 Verification
`conda run -n heart-conduction python -m pytest cardiac_core/tests/ -q`
### Phase 1 Exit Criteria
- [ ] `IonicPreset` shipped: builds scaled models for every registered model; save/load JSON; accepted `ionic_model=`
  on all three engines; survives reset/with_; `.npz`-save uses the base name (documented); integrity goldens atol=0;
  no existing test breaks.
### Phase 1 Cleanup
- float64/bool consistency; V5.3 untouched; the preset reuses `build_ionic_model` (no per-model or duplicated engine
  logic); no solver files touched; `_LAZY`/`__all__` exports current.
**-> Commit point: git commit after Phase 1 passes (feature branch, e.g. `ionic-preset`).**

---

## Deferred (future phases, NOT this plan)
- **Phase 2 — `.npz` round-trip**: extend `CardiacMeshData`/`save_cardiac_mesh`/`load_cardiac_mesh` to carry an
  optional scalings map alongside `ionic_model` (back-compat: a plain string still loads); the loader rebuilds an
  `IonicPreset`. Closes the "tuned model travels with the mesh" gap.
- **Phase 3 — tuner bridge**: `IonicPreset.from_tuning_result(json)` + reconcile the Optimizer tuner's `g_Na`
  (PHAS13_REGISTRY attr names) ↔ cardiac_core's `GNa`; optionally read `Optimizer/V1/presets/*.json`. Entangles with
  the gated ionic-tuner redesign — schedule with it.
- **Absolute-set on a live sim** (`set_parameter`) and **per-node conductance heterogeneity** — both remain stubbed.

## Final Cleanup
- float64/bool consistency; V5.3 untouched; no solver files touched; goldens bit-identical.
- `API_CHEATSHEET.md` + `API_REFERENCE.md` present `IonicPreset`; `_LAZY`/`__all__` current.
- Archive this plan: `mkdir -p Research/Active/engine_consolidation/plans && cp Research/Active/engine_consolidation/IONIC_PRESET_PLAN.md
  "Research/Active/engine_consolidation/plans/$(date +%Y-%m-%d)_ionic-preset.md"`.

## Mutation Log
{empty — populate during execution: `**MUTATED {date}**: Step X.Y {SKIPPED|SPLIT|INSERTED} — {reason}`}
