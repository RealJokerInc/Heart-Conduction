# Engine Consolidation — Unified API Design (Goal 1)

**Status:** DRAFT — 2026-06-24. Builds directly on `GLOSSARY.md` (the ubiquitous language) and the
Cross-Engine Capability Census in `KNOWLEDGE.md`. This is the **`Simulation` interface/Protocol +
idioms** layer — North-Star Goal 1. The declarative `SimulationSpec` here is the bridge the LLM
wrapper (Goal 2) will sit on.

**Reading order:** GLOSSARY (vocabulary) → this doc (interface in that vocabulary) → later, the
LLM-wrapper skill bundle (Goal 2).

All names here are the **canonical glossary names**. Every decision below traces to a glossary row.

---

## 0. The four idioms

A simulation has exactly four verbs. Everything the API exposes is one of these:

| Idiom | What it does | Surface |
|---|---|---|
| **DECLARE** | build a simulation from geometry + ionic + conductivity + stimulus | `monodomain()/bidomain()/lbm()` factories; `create_simulation(spec)` |
| **STIMULATE** | add a pacing event / protocol | `Stimulus`/`StimulusProtocol` at declare-time, or `sim.stimulate(...)` |
| **RUN** | integrate to `t_end`, collect results | `sim.run(t_end, ...)` → `SimulationResult` |
| **CHANGE** | re-parameterize for a sweep | `sim.with_(**overrides)` → new `Simulation` (functional) |

---

## 1. `Simulation` Protocol (the runtime interface)

The single engine-agnostic interface. Optimizer / Surrogate / the LLM wrapper all program against
this; the concrete engine (mono/bidomain/LBM) is hidden behind the factory.

```python
@runtime_checkable
class Simulation(Protocol):
    # --- introspection (read-only) ---
    Nx: int
    Ny: int
    dx: float
    dy: float
    dt: float
    Cm: float
    ionic_model: IonicModel

    @property
    def Vm(self) -> Tensor: ...        # current transmembrane potential, grid (Nx,Ny) torch f64
    @property
    def t(self) -> float: ...          # current sim time (ms)

    # --- STIMULATE ---
    def stimulate(self, region, start_time, duration=1.0, amplitude=-52.0) -> None: ...
    #   region: callable (x,y)->bool  OR  (Nx,Ny) bool mask. Accumulates (+=) — glossary #9.

    # --- RUN ---
    def run(self, t_end, save_every=1.0, *, batch=None,
            record=("Vm",), callback=None) -> "SimulationResult | Iterator[SimulationResult]": ...
    def step(self) -> None: ...        # advance one configured dt, mutate live state
    def reset(self) -> None: ...       # restore initial state (re-run from t=0)

    # --- CHANGE (functional; clean for sweeps / Optimizer) ---
    def with_(self, **overrides) -> "Simulation": ...
    #   returns a NEW Simulation with overridden params; original untouched.
    #   e.g. sim.with_(conductivity=cfg2)  /  sim.with_(dt=0.01)
```

**Decisions baked in:**
- `Vm` is the canonical voltage name (glossary #2), grid-shaped `(Nx,Ny)` torch f64 (glossary #7, P2).
- `run()` is **eager by default**; `batch=k` switches to streaming (glossary #6). One public output
  type only: `SimulationResult` (glossary #5 public).
- `with_()` is the **only** re-parameterization idiom (user, 2026-06-24): immutable, sweep-safe, no
  hidden mutable state between runs.

---

## 2. DECLARE — per-model factories

Structured grid is the primary representation (P2). Geometry is the LBM-simple descriptor; the
unstructured `TriangularMesh` (FEM) is the explicit secondary path, not the default.

```python
def monodomain(geometry, ionic_model, conductivity, stimulus=None, *,
               dt=0.02, Cm=1.0,
               splitting="strang", ionic_solver="rush_larsen",
               diffusion_solver="crank_nicolson", linear_solver="pcg") -> Simulation: ...

def bidomain(geometry, ionic_model, conductivity, stimulus=None, *,
             dt=0.02, Cm=1.0,
             splitting="strang", ionic_solver="rush_larsen",
             parabolic_solver="pcg", elliptic_solver="auto", theta=0.5,
             boundary=None) -> Simulation: ...        # boundary: BoundarySpec (engine-specific)

def lbm(geometry, ionic_model, conductivity, stimulus=None, *,
        dt, Cm=1.0,
        lattice="d2q5", weights_mode="canonical") -> Simulation: ...
```

Shared arguments (the universal tier — same name, same meaning across all three):

| Arg | Type | Notes |
|---|---|---|
| `geometry` | `Grid` | `Grid(Nx, Ny, dx, dy=None, mask=None)` — structured, P2. `dy` defaults to `dx`. |
| `ionic_model` | `IonicModel` | the byte-identical ABC (glossary #3); e.g. `TTP06Model(cell_type=EPI)` |
| `conductivity` | `ConductivityConfig` | §4 — the single place chi/Cm and Form-A/B live |
| `stimulus` | `StimulusProtocol \| None` | §3; `None` = no stimulus (declare then `sim.stimulate(...)`) |
| `dt` | float | default 0.02 (mono/bidomain); **required for LBM** (τ derived from D·dt) |
| `Cm` | float | default 1.0 |

Engine-specific knobs stay on their own factory (glossary Tier 2) — never leak into the others.

---

## 3. STIMULATE

```python
Stimulus(region, start_time, duration=1.0, amplitude=-52.0)   # one event; accumulate (+=)
StimulusProtocol()                                            # collection
  .add_stimulus(region, start_time, duration=1.0, amplitude=-52.0)
  .add_s1s2_protocol(region, n_s1, bcl, s2_ci, ...)
  .add_regular_pacing(region, bcl, n_beats, start_time=0.0, ...)
```

- `region` accepts a **callable `(x,y)->bool`** or a **`(Nx,Ny)` bool mask** (LBM gains the callable
  path + the protocol/pacing helpers it lacks today — glossary #9/#10).
- **Default amplitude `-52.0`** (RESOLVED 2026-06-24, glossary #9): classical-engine majority; LBM's
  `-80` retired. Identical depolarization since the ionic model is byte-identical across engines.
- `start_time` is the canonical field name (LBM's `start` renamed).

The North-Star high-level pacing (`single` / `s1s2` / `regular(bcl, n_beats)`) **expands** into this
list — `add_s1s2_protocol` / `add_regular_pacing` are the expansion targets (glossary #10).

---

## 4. `ConductivityConfig` — the chi/Cm + Formulation-A/B firewall (glossary #12, #13)

Stores **physics**; emits the **physically-scaled diffusivity** each engine needs. This is the *only*
place `chi` appears (decision #13) — and that is exactly what makes the Form-A/B asymmetry vanish.

```python
@dataclass(frozen=True)
class ConductivityConfig:
    # physical inputs (one construction mode below sets these)
    sigma_i: ... = None       # intracellular conductivity (scalar or (sig_l, sig_t))
    sigma_e: ... = None       # extracellular
    sigma_eff: ... = None     # effective single-domain (if given directly)
    chi: float = 1400.0
    Cm: float = 1.0
    fiber_angle: "float | Tensor" = 0.0

    # construction (hide the field sprawl — "one obvious way")
    @classmethod
    def isotropic(cls, sigma, chi=1400.0, Cm=1.0): ...            # single-domain effective, scalar
    @classmethod
    def bidomain(cls, sigma_i, sigma_e, chi=1400.0, Cm=1.0): ...  # paired
    @classmethod
    def anisotropic(cls, sigma_l, sigma_t, fiber_angle, chi=1400.0, Cm=1.0): ...  # fiber tensor

    @property
    def D_eff(self):                 # TRUE physical effective diffusivity = reduce(sigma_i,sigma_e)/(chi*Cm)
        ...                          #   reduce = harmonic i/e collapse sigma_i*sigma_e/(sigma_i+sigma_e)

    # per-engine emitters — the factory splats these; the user never calls them
    def for_monodomain(self):        # Form A: diffusion scaled by Cm INTERNALLY -> feed it Cm-UNscaled D
        return dict(D=self.D_eff * self.Cm,   # = sigma_eff/chi ; engine's mass-term Cm redoes the /Cm
                    chi=1.0, Cm=self.Cm)       # chi folded into D (chi=1 inert); real Cm drives mass term + reaction
    def for_bidomain(self):          # Form B: no internal Cm scaling -> feed fully-scaled D + real Cm
        return dict(D_i=self.sigma_i/(self.chi*self.Cm),
                    D_e=self.sigma_e/(self.chi*self.Cm), Cm=self.Cm)
    def for_lbm(self):               # Form B
        return dict(D=self.D_eff, Cm=self.Cm)
```

**Why the asymmetry stays hidden (the corrected mechanic — verified against `fdm.py:195–238`):**

The FDM implicit solve is `(χ·Cm·I − ½dt·L)Vⁿ⁺¹ = (χ·Cm·I + ½dt·L)Vⁿ` with `L` built from the input
`D` (NOT χ·Cm). Dividing by χ·Cm ⟹ **physical diffusivity = `D_input/(χ·Cm)`**; the reaction divides by
the same tissue `Cm` (`state.Cm`, V5.5 fix). So both halves use the *real* `Cm` — it must reach the engine.

| Engine | Form | Diffusion input fed | `chi` | `Cm` | ⇒ engine diffusion | reaction |
|---|---|---|---|---|---|---|
| monodomain | A | `D_eff·Cm` (= `sigma_eff/chi`, **Cm-unscaled**) | `1` | `Cm` | `input/(1·Cm)=sigma_eff/(chi·Cm)` ✓ | `/Cm` ✓ |
| bidomain | B | `D_i,D_e = sigma_*/(chi·Cm)` (fully scaled) | — | `Cm` | used directly ✓ | `/Cm` ✓ |
| LBM | B | `D_eff = sigma_eff/(chi·Cm)` (fully scaled) | — | `Cm` | used directly ✓ | `/Cm` ✓ |

Only the **diffusion input's Cm-scaling** differs by formulation: Form-A monodomain scales diffusion by
`Cm` *internally* (mass term), so it must be fed a Cm-**un**scaled `D`; Form-B feeds it Cm-**pre**scaled.
`chi` is still ConductivityConfig-only (#13): folded into the monodomain `D` with the engine's `chi`
pinned inert at `1`, never user-facing. At the project's pinned **`Cm=1`, all three collapse to
`D_eff = sigma_eff/chi`** — exactly what the 2026-05-30 cross-engine test used (`D=D_EFF, chi=1`).

> 🎯 **`for_monodomain()` is TEMPORARY** (decision 2026-06-24): **Form B is the canonical target.**
> When Phase 4 converts monodomain's diffusion to Form B (drop χ·Cm from the mass term + spectral
> denominators, consume pre-scaled `D`), this Cm-unscaled special-case is **deleted** and
> `ConductivityConfig` collapses to one emitter (physical `D` + `Cm`). The asymmetry above is the
> price paid only until that rewire. See KNOWLEDGE "Canonical formulation = B".

> ⚠️ **Corrected (2026-06-24, source-verified).** The earlier draft "feed `D_eff` with `chi=1, Cm=1`
> no-op" was **wrong for Cm≠1**: pinning the engine's `Cm=1` makes the *reaction* divide by 1 instead of
> the real `Cm` — invisible at the pinned `Cm=1`, silently wrong otherwise (same Cm-trap family as the
> false time-dilation invariant). The real `Cm` must always reach the engine.
> **Build-time gate CLOSED (2026-06-24).** Probe `Monodomain/Engine_V5.5/_probe_conductivity_firewall.py`
> drives raw `sigma_i=1.74, sigma_e=6.25, chi=1400` → `for_monodomain()` → the live V5.5 cable:
> arithmetic `D = 0.0009721973895941` (= reference `D_EFF` to 1.1e-19, Cm-independent ✓); CV(Cm=1)=**54.35**
> (0.00% vs bidomain reference), CV(Cm=2)=**28.09** vs 27.77 (**1.15%**, < 5% tol). The Cm≠1 path is
> numerically correct — not just on paper.

---

## 5. RUN — `SimulationResult` (the only public output, glossary #5/#6)

```python
@dataclass
class SimulationResult:
    times: Tensor        # (T,)
    Vm: Tensor           # (T, Nx, Ny) torch f64   (glossary #7, P2)
    phi_e: "Tensor|None" # (T, Nx, Ny) — auto-populated for bidomain, else None
    ionic_states: "Tensor|None"   # (T, Nx, Ny, n_states) — opt-in via record=
    dx: float
    dy: float
    # thin analysis hooks (delegate to cardiac_core.analysis)
    def cv(self, **kw): ...      def apd(self, **kw): ...
    def lat(self, **kw): ...     def restitution(self, **kw): ...
```

`run(t_end, save_every, *, batch, record, callback)`:

| `batch` | mode | returns |
|---|---|---|
| `None` (default) | eager — integrate to `t_end`, collect all save-points | one `SimulationResult` |
| `k` (int ≥ 1) | streaming — yield chunks of ≤ k save-points (`k=1` = frame-by-frame) | `Iterator[SimulationResult]` |

- `record=("Vm",)` decides which fields are kept — "outputs drive the run". `Vm` always; `phi_e`
  auto-added for bidomain; `ionic_states` opt-in (Surrogate ground-truth gen). Bounds memory.
- `callback` retained for eager early-stop / progress; in streaming mode the caller's loop `break`s.
- Accepted wart (glossary #6): return type varies with `batch`. Deliberate — one verb, not two.

---

## 6. DECLARE (declarative) — `SimulationSpec` + `create_simulation` (the Goal-2 bridge)

> **Spec schema = the intake questionnaire.** Each field is self-describing
> (`{tier, prompt, options, default}`); the LLM "gather" step = ask the prompt of each unfilled
> **required** field. Because the spec and the factories share one schema, the questionnaire **cannot
> drift** from what the engines actually need. This is the cross-goal leverage point.

```python
create_simulation(spec: SimulationSpec) -> Simulation
#   dispatches on spec.engine to monodomain()/bidomain()/lbm(); pure thin glue, no new logic.
```

Three field tiers (the LLM only ever asks about **required**):

| Tier | Meaning | Examples |
|---|---|---|
| **required** | LLM asks the user | `engine`, `geometry` (Nx,Ny,dx), `pacing`, `measure` (what to compute) |
| **defaulted** | silent physiological good value | `ionic`=TTP06/EPI, `dt`=0.02, `Cm`=1, `splitting`=strang, solvers, `chi`=1400 |
| **derived** | computed, never asked | `save_every`/`t_end` from `measure`+`pacing`; `D_eff` from `sigma`; LBM `tau` from `D·dt` |

Sketch (illustrative — field metadata is what the questionnaire reads):

```python
@dataclass
class SimulationSpec:
    engine: str = field(metadata={"tier": "required", "options": ["monodomain","bidomain","lbm"],
        "prompt": "Which physics? monodomain (fast, single potential), bidomain (bath/boundary "
                  "effects, two potentials), or LBM (lattice-Boltzmann)?"})
    geometry: GridSpec = field(metadata={"tier": "required",
        "prompt": "Tissue size and resolution — width, height, grid spacing dx?"})
    pacing:   PacingSpec = field(metadata={"tier": "required",
        "prompt": "How is it paced — single beat, S1-S2, or regular(bcl, n_beats)?"})
    measure:  list[str] = field(default_factory=lambda: ["cv"], metadata={"tier": "required",
        "options": ["cv","apd","lat","reentry"],
        "prompt": "What do you want to measure?"})
    # defaulted (silent):
    ionic: str = field(default="ttp06_epi", metadata={"tier": "defaulted"})
    dt: float = field(default=0.02, metadata={"tier": "defaulted"})
    Cm: float = field(default=1.0,  metadata={"tier": "defaulted"})
    conductivity: ConductivityConfig = field(default=None, metadata={"tier": "defaulted"})  # isotropic default
    # ... splitting/solvers/chi all defaulted ...
    # derived: save_every, t_end, D_eff — computed in create_simulation, never in the spec
```

**"Outputs drive the run"** (glossary #6, KNOWLEDGE): `measure` feeds `derived` numerics — e.g.
`measure=["reentry"]` → longer `t_end`, finer `save_every`; `measure=["cv"]` → enough save-points to
fit an activation front. `measure` also sets `record=` (e.g. ionic_states only if asked).

**Engine inference (auditable):** the LLM may *infer* `engine` from the scientific question
(bath/boundary → bidomain; fast/simple → monodomain) and record the rationale, but `engine` stays
explicit + overridable in the spec.

---

## 7. Minimal-spec smoke test (the "one obvious way" bar)

A non-coder's minimal ask must run on pure defaults:

```python
sim = create_simulation(SimulationSpec(
    engine="monodomain",
    geometry=GridSpec(Lx=2.0, Ly=0.5, dx=0.01),        # required
    pacing=PacingSpec.single(region="left_edge"),       # required
    measure=["cv"],                                      # required
))                                                       # everything else defaulted/derived
result = sim.run(t_end=50.0)                             # eager
print(result.cv())                                       # ~54 cm/s with default TTP06/EPI, sigma
```

Programmatic equivalent (no spec, factory direct):

```python
sim = monodomain(
    Grid(Nx=200, Ny=50, dx=0.01),
    TTP06Model(cell_type=EPI),
    # sigma is the raw CONDUCTIVITY (mS/cm); D_eff = sigma/(chi*Cm) is DERIVED inside the config.
    # human-ventricle default: sigma_i=1.74, sigma_e=6.25 -> sigma_eff=1.361 -> D_eff=0.000972 cm^2/ms
    ConductivityConfig.bidomain(sigma_i=1.74, sigma_e=6.25, chi=1400.0),   # mono uses the D_eff collapse
    stimulus=StimulusProtocol().add_stimulus(left_edge_region, start_time=1.0),
)
cv = sim.run(t_end=50.0).cv()                                              # ~54 cm/s (matches cv_shared)
sim_hi = sim.with_(conductivity=ConductivityConfig.isotropic(sigma=2.0))  # sweep (sigma_eff=2 mS/cm)
```

> Units note: `ConductivityConfig.isotropic(sigma=...)` / `.bidomain(sigma_i, sigma_e)` take **raw
> conductivity in mS/cm**; the effective physical diffusivity `D_eff = sigma_eff/(chi·Cm)` (cm²/ms) is
> derived. Do **not** pass a pre-divided `D` as `sigma` — that's the unit trap (0.00097 is the *D_eff*,
> not the conductivity). The named-tissue → sigma mapping is a Goal-2 questionnaire convenience.

---

## 8. Open / deferred

| Item | State |
|---|---|
| **Ditch FEM → structured-grid ONLY** (user, 2026-06-24) | **PENDING DISCUSSION.** Strengthens P2 from "structured primary / FEM secondary" to "structured is the *only* standard." Knock-on (talk later): drop the unstructured/flat-`(n_dof,)` secondary path, drop `TriangularMesh`, drop monodomain's `FEMDiscretization` knob; `Vm` is always grid-native (no flat path to support); may simplify how the engines themselves are implemented. See §9. |
| #5 internal live-`State` unification (one dataclass; LBM adopts, drops free attrs) | **defer to code phase** (no public-contract impact); FEM-ditch makes the flat path moot, simplifying this further |
| `ConductivityConfig` `D_eff` reduction for unequal anisotropy ratios | use equal-ratio harmonic mean now; flag exact tensor reduction at build time |
| Geometry input (Fiji drawing → Builder image→mesh; drawings inbox; export→mask contract) | **deferred** — assume geometry provided |
| `SimulationSpec` field-metadata exact schema (the questionnaire renderer) | sketched here; finalize when Goal-2 skill bundle is built |
| Optimizer / Surrogate migration onto `Simulation` Protocol | deferred (per-consumer, post-rewire) |

## 9. CONFIRMED: structured-grid is the ONLY standard — FEM dropped (user, 2026-06-24)

**Decision: ditch FEM. Every simulation runs on a structured grid.** This *simplifies* the design — a
whole branch is removed, not demoted:

- **P2 → P2′** — "structured grid is the *only* standard." The "unstructured/complex-mesh secondary
  path" (FEM, `TriangularMesh`, flat `(n_dof,)` as a first-class geometry) **disappears** rather than
  being demoted. There is no secondary geometry path to maintain.
- **`Vm` is unconditionally grid-shaped** `(Nx,Ny)` — no flat-geometry case, so reshape-at-the-API is
  the *only* representation (FDM/FVM may still index flat *internally* on the structured grid, but that
  never surfaces as a geometry the user picks).
- **State unification (#5) gets simpler** — no unstructured node container to accommodate.
- **Geometry shrinks to one constructor** — `Grid(Nx, Ny, dx[, dy], mask)`. `TriangularMesh` is removed.
- **Engine knobs shrink** — monodomain drops the `FEMDiscretization` scheme. **FDM stays primary; FVM
  survives** (it is structured-grid-native — TPFA on the grid, not unstructured). Whether to *also*
  collapse FVM into FDM (on a uniform structured grid with standard stencils they largely coincide) is a
  **separate, later** question — not part of the FEM ditch.
- This is the concrete content of "reconsider how we implement the engines" — Phase-4 rewire scope, and
  it composes with the Form-A→B convergence (§4): monodomain loses both its `χ·Cm` mass term *and* its
  FEM scheme as it moves into `cardiac_core`.

---

## Provenance
2026-06-24. Decisions confirmed this session: stim amplitude **−52**; **CHANGE = functional
`.with_()`**; **construction = factories + `SimulationSpec`/`create_simulation` layered**. All other
names inherit from `GLOSSARY.md` resolved rows (#2, #5, #6, #7, #12, #13).
