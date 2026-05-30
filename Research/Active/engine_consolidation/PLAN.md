# PLAN: Monodomain Engine V5.5 — Cm-Correct Reaction Fork

Created: 2026-05-29
Engine(s): Monodomain V5.5 (NEW, forked from V5.4)
Research question: [engine_consolidation](README.md)
Source: [IDEALOG.md](IDEALOG.md) — 2026-05-29 "V5.5 detour decided — fix the Formulation-A reaction Cm bug in an independent fork"

## Objective
Fork `Monodomain/Engine_V5.4` → `Monodomain/Engine_V5.5` as a full independent copy (V5.4 stays the frozen validated baseline). Apply ONE functional change: divide the operator-split reaction (ionic) voltage update by the tissue Cm, so simulations with Cm != 1.0 are physically correct and consistent with Bidomain V1 / LBM V1 (Formulation B). The diffusion half is untouched — its `chi*Cm` mass term already handles arbitrary Cm. This unblocks the eventual `cardiac_core` consolidation with a Cm-robust monodomain.

## Success Criteria
- [ ] `Monodomain/Engine_V5.5/` exists as a faithful copy of V5.4 (Phase 0)
- [ ] At Cm=1.0, V5.5 is bit-identical to V5.4 (regression golden, atol=1e-12)
- [ ] At Cm=k, the reaction divides by Cm in both ionic steppers (rush_larsen + forward_euler)
- [ ] Cm-scaling time-dilation invariant holds: V(x, t; Cm=k) == V(x, t/k; Cm=1) — 0D and tissue-level
- [ ] Cross-validated against Bidomain V1 (independent Formulation-B engine): V5.5 reproduces bidomain's CV/APD Cm-scaling ratios
- [ ] V5.5's dead, unused internal LBM path is removed (verified: no importer, no test; boundary work uses the separate LBM V1 engine)
- [ ] All existing V5.4 tests pass unchanged in V5.5 (no regressions)

## Architecture Changes
- NEW: `Monodomain/Engine_V5.5/` — full copy of `Engine_V5.4/`
- MOD: `…/cardiac_sim/simulation/classical/state.py` — add `Cm: float = 1.0` field to `SimulationState`
- MOD: `…/classical/discretization_scheme/{fdm,fem,fvm}.py` — add public `Cm` read-only property returning `self._Cm`. **FDM/FVM already store `self._Cm`; FEM does NOT** (it bakes Cm into `self.M` at `fem.py:163` and keeps only `_mesh,_n_dof,_x,_y`) — so FEM also needs `self._Cm = Cm` and `self._chi = chi` added to its `__init__`.
- MOD: `…/classical/monodomain.py:324` — pass `Cm=spatial.Cm` as a kwarg in the `SimulationState(...)` constructor call (read directly, NOT via `getattr(spatial,'Cm',1.0)` — a missing attribute must fail loud, not silently fall back to 1.0 and re-introduce the bug)
- MOD: `…/classical/solver/ionic_time_stepping/rush_larsen.py:83` — `state.V = V + dt * (-(Iion + Istim))` → `state.V = V + dt * (-(Iion + Istim) / state.Cm)`
- MOD: `…/classical/solver/ionic_time_stepping/forward_euler.py:64` — same
- DEL (Phase 0): `…/cardiac_sim/simulation/lbm/` — remove the entire unused LBM package (collision/d2q5/d3q7/monodomain/state/__init__). Verified: zero importers, zero standing tests, boundary research uses the separate `LBM/Engine_V1` engine. Also drop the now-dead `ionic_time_stepping/base.py::step_with_V` (its only caller was the LBM path) and the stale `lbm/` docstring line in `simulation/__init__.py`.
- NEW: `Monodomain/Engine_V5.5/test_phase10_cm_scaling.py` — Cm-scaling validation suite (0D, tissue, bidomain cross-val)
- NEW: `Monodomain/Engine_V5.5/_regression/bidomain_cm_ref.py` — generates the Bidomain V1 reference (separate process, Bidomain engine)

## Known Failures (from IDEALOG)
- **Converting V5.4 in-place to Formulation B** — rejected: risks V5.4's 77 passing tests for zero benefit at Cm=1. The fork preserves V5.4 as baseline; only V5.5 changes.
- **Full Formulation-B structural conversion of V5.5** (rewriting diffusion operators to 1/dt mass term + pre-scaled D) — NOT this plan. Diffusion already handles arbitrary Cm; touching it is scope creep and contradicts "only thing to fix."
- **Dividing by the ionic model's internal `Cm`** (e.g. `ttp06/model.py:548 p.Cm=0.185`, paci/ord calcium `inv_VcF`) — WRONG. That is a fixed per-cell capacitance for Ca/Na concentration-flux bookkeeping, NOT the cable Cm. The fix uses the TISSUE Cm (`spatial.Cm` / `state.Cm`) only. Do not touch the ionic models.

---

## Phase 0: Clone V5.4 → V5.5 (faithful copy, zero logic change)

**Goal**: A byte-faithful copy of Engine_V5.4 living at Engine_V5.5 that passes the entire existing test suite identically. This is the Cm=1 baseline and the backup guarantee.
**Tier**: small
**Estimated scope**: directory copy + doc-header updates + capture regression golden

### Phase Context
- Both engines use the internal package name `cardiac_sim` with **relative imports** (`from .....ionic.base import …`). A copy therefore needs NO import-path rewrites for internal code — relative imports resolve within whichever directory you run from.
- Consequence of the shared `cardiac_sim` name: V5.4 and V5.5 **cannot both be imported in the same Python process**. All cross-engine comparisons must use **separate processes** (run V5.4, dump arrays to `.npy`; run V5.5, load + compare). Do not attempt `import` of both in one test.
- V5.4 tests are **standalone scripts run with `python`**, not pytest. Run each from the engine root. Known test files: `test_phase7.py`, `test_phase8.py`, `test_boundary_modes.py`, `tests/test_mhas13.py`, `tests/test_paci.py`, `tests/test_phas13.py`. Phases 1-6 validation lives inside these / the scripts' sub-tests.
- Do NOT modify `Monodomain/Engine_V5.4/` or `Monodomain/Engine_V5.3/` at any point.

### Step 0.1: Copy the engine tree
**Model**: sonnet

#### Read First
- `CLAUDE.md` § "Running Tests" — confirm how V5.4 tests are invoked.

#### Why
A literal copy (not a refactor) is the backup mechanism. Stripping caches avoids stale `.pyc` confusion when the copy runs.

#### Implementation Spec
**Files to create:** `Monodomain/Engine_V5.5/` (recursive copy of `Engine_V5.4/`)

#### Pseudocode
N/A — pure file operation (see Verify for the exact commands).

#### Test Spec
N/A — verified structurally by the tree-parity `diff` (no behavioral assertion at this step).

#### Checklist
- [ ] `cp -r Monodomain/Engine_V5.4 Monodomain/Engine_V5.5`
- [ ] Delete all `__pycache__/` and `*.pyc` under `Engine_V5.5/`
- [ ] Confirm tree parity (file count/names match V5.4 modulo caches)

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction
cp -r Monodomain/Engine_V5.4 Monodomain/Engine_V5.5
find Monodomain/Engine_V5.5 -name __pycache__ -type d -prune -exec rm -rf {} +
find Monodomain/Engine_V5.5 -name '*.pyc' -delete
# parity (should print nothing but cache/pyc differences):
diff <(cd Monodomain/Engine_V5.4 && find . -type f ! -path '*__pycache__*' ! -name '*.pyc' | sort) \
     <(cd Monodomain/Engine_V5.5 && find . -type f ! -path '*__pycache__*' ! -name '*.pyc' | sort)
```

#### Exit Criteria
- [ ] `Engine_V5.5/` exists, no `__pycache__`/`.pyc`, file tree matches V5.4.

#### Risk
Accidental edit to V5.4 — mitigation: only `cp`/`rm` under `Engine_V5.5/`; never touch V5.4 paths.

### Step 0.2: Re-label V5.5 docs (lineage only, no code)
**Model**: sonnet

#### Read First
- `Monodomain/Engine_V5.5/README.md` (top) and `Monodomain/Engine_V5.5/PROGRESS.md` (top) — current V5.4 headers.

#### Why
The copy must announce what it is, or a future cold-start agent will mistake it for V5.4. Code/docstrings stay; only the top-level identity docs change.

#### Implementation Spec
**Files to modify:**
- `Engine_V5.5/README.md` (header) — title → "Monodomain Engine V5.5", add one line: "Fork of V5.4. ONLY change: reaction divides by tissue Cm (Cm != 1.0 correct). V5.4 = frozen baseline."
- `Engine_V5.5/PROGRESS.md` (header) — note V5.5 lineage + this PLAN as the active work; do NOT alter V5.4's phase-history tables.

#### Pseudocode
N/A — documentation-only edit.

#### Test Spec
N/A — no behavioral change (Checklist item "No `.py` files touched" is the guard).

#### Checklist
- [ ] README header updated
- [ ] PROGRESS header updated (history tables left intact)
- [ ] No `.py` files touched in this step

#### Exit Criteria
- [ ] V5.5 README/PROGRESS clearly identify it as the V5.4 fork + Cm fix.

#### Risk
Over-editing docs into V5.4 history — mitigation: change only header/identity lines.

### Step 0.3: Capture the Cm=1 regression golden + confirm copy passes suite
**Model**: opus

#### Read First
- `Monodomain/Engine_V5.4/test_phase7.py` — see how a `MonodomainSimulation` is built (grid → FDM → sim → run).

#### Why
We need an objective, process-isolated regression oracle: a voltage trajectory from V5.4 saved to disk, that V5.5 must reproduce to atol=1e-12 BEFORE and AFTER the fix (at Cm=1). Relying only on "tests pass" is weaker than a saved golden array.

#### Implementation Spec
**Files to create:**
- `Monodomain/Engine_V5.5/_regression/make_golden.py` — runs a fixed reference sim and saves `(times, voltages)` to `_regression/golden_cm1.npz`. Deterministic: fixed grid (e.g. 1D cable 100x1 or small 2D), TTP06 EPI, rush_larsen + crank_nicolson, Cm=1.0, fixed dt, fixed t_end, single stimulus. CPU + float64 for reproducibility.
- `Monodomain/Engine_V5.5/_regression/check_golden.py` — runs the SAME sim, loads `golden_cm1.npz`, asserts `np.allclose(V, golden_V, atol=1e-12, rtol=0)`.

#### Pseudocode
```
# make_golden.py (run under Engine_V5.4 AND copied identically to Engine_V5.5)
grid = StructuredGrid.create_rectangle(Lx, Ly, Nx, Ny, device='cpu')
fdm  = FDMDiscretization(grid, D=0.001, chi=1.0, Cm=1.0)   # Cm=1 baseline
sim  = MonodomainSimulation(fdm, ionic_model='ttp06', stimulus=stim,
                            dt=0.02, splitting='strang',
                            ionic_solver='rush_larsen',
                            diffusion_solver='crank_nicolson', linear_solver='pcg')
times, V = sim.run_to_array(t_end=50.0, save_every=1.0)
np.savez('golden_cm1.npz', times=times, voltages=V)
```
NOTE: this golden uses `chi=1.0` (matching the test convention), so it does NOT exercise the `chi*Cm` mass term with a realistic chi — it is blind to a chi-only regression. Acceptable here because this fork changes ONLY the Cm reaction term (not chi handling); the golden's job is to lock Cm=1 reaction+diffusion behavior, which it does. (If a future change touches chi, add a `chi=1400` golden.)

#### Test Spec
- `_regression/check_golden.py` — Setup: identical sim config; Cm=1.0. Expected: `allclose(V, golden, atol=1e-12)`.

#### Checklist
- [ ] Write `make_golden.py`; run it in **Engine_V5.4** to produce the golden `.npz`
- [ ] Copy the golden `.npz` into `Engine_V5.5/_regression/`
- [ ] Run `make_golden.py` in **Engine_V5.5** (unmodified copy) → produces identical arrays
- [ ] Run all existing V5.5 test scripts; confirm same PASS output as V5.4

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.4
conda run -n heart-conduction python _regression/make_golden.py   # writes golden_cm1.npz here
cp _regression/golden_cm1.npz ../Engine_V5.5/_regression/
cd ../Engine_V5.5
conda run -n heart-conduction python _regression/check_golden.py  # PASS expected (copy == V5.4)
conda run -n heart-conduction python test_phase7.py
conda run -n heart-conduction python test_phase8.py
conda run -n heart-conduction python test_boundary_modes.py
conda run -n heart-conduction python tests/test_mhas13.py
conda run -n heart-conduction python tests/test_paci.py
conda run -n heart-conduction python tests/test_phas13.py
```

#### Exit Criteria
- [ ] `check_golden.py` passes in V5.5 (unmodified copy reproduces V5.4 exactly).
- [ ] All existing test scripts pass in V5.5 with the same results as V5.4.

#### Risk
`make_golden.py` lives under V5.4 — it's a NEW file, not a modification of V5.4 logic, but to be safe keep it under a `_regression/` subdir and do not commit it into V5.4 history if V5.4 must stay pristine. Mitigation: run it from a temp copy or `git checkout -- .` V5.4 afterward if policy requires V5.4 untouched. (The golden `.npz` is the only artifact that needs to persist, in V5.5.)

### Step 0.4: Remove the dead, unused LBM path
**Model**: opus

#### Read First
- `Monodomain/Engine_V5.5/cardiac_sim/simulation/__init__.py` — confirm the `lbm` reference is a docstring comment, NOT an import.
- `…/ionic_time_stepping/base.py:85-125` — `step_with_V` (the LBM-only method).

#### Why
V5.5's internal LBM (`cardiac_sim/simulation/lbm/`) is **dead code**: verified across the whole repo to have ZERO importers, ZERO standing tests, and ZERO experiments. The boundary-conduction research (the only active LBM work) runs on the **separate `LBM/Engine_V1` engine** (`from src.simulation import LBMSimulation`), where all new boundary conditions live. Removing it makes V5.5 a clean classical-only monodomain and eliminates the ONLY place the chi/Cm source-term entanglement existed — so there is nothing left to guard. (V5.4 retains the LBM path; it remains the faithful backup.)

#### Implementation Spec
**Files to delete:** entire `Monodomain/Engine_V5.5/cardiac_sim/simulation/lbm/` directory (`collision.py`, `d2q5.py`, `d3q7.py`, `monodomain.py`, `state.py`, `__init__.py`).
**Files to modify:**
- `…/ionic_time_stepping/base.py` — remove `step_with_V` (dead after LBM removal; its only caller was `lbm/monodomain.py:237`). Leaving it is harmless but it references a now-nonexistent path in its docstring; remove for cleanliness.
- `…/simulation/__init__.py` — drop the stale `- lbm/: Lattice-Boltzmann method` docstring line.

#### Pseudocode
N/A — deletion + two small doc/dead-code removals (see Verify for commands).

#### Test Spec
- Re-run of the full existing suite (Verify block) must stay green — the behavioral assertion is "removal changed nothing," since nothing imported the LBM path.

#### Checklist
- [ ] `cardiac_sim/simulation/lbm/` deleted from V5.5
- [ ] `step_with_V` removed from `ionic_time_stepping/base.py`. NOTE on `_update_gates`: the BASE `_update_gates` becomes dead after this removal (RushLarsen *overrides* it at `rush_larsen.py:101`; ForwardEuler *inlines* its own gate update at `forward_euler.py:77`) — so it is no longer "used by Rush-Larsen." It is harmless to keep; leave it to minimize churn. Do NOT remove RushLarsen's override.
- [ ] `simulation/__init__.py` docstring updated
- [ ] V5.4's `lbm/` is UNTOUCHED (deletion is V5.5-only)

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.5
rm -rf cardiac_sim/simulation/lbm
# nothing should still reference it:
grep -rn "simulation.lbm\|import.*lbm\|step_with_V\|LBMState\|create_lbm_state" cardiac_sim test_*.py tests/ --include=*.py | grep -v __pycache__ || echo "clean: no LBM references remain"
# full suite still green (nothing depended on LBM):
conda run -n heart-conduction python _regression/check_golden.py
conda run -n heart-conduction python test_phase7.py
conda run -n heart-conduction python test_phase8.py
conda run -n heart-conduction python test_boundary_modes.py   # NOTE: FDM boundary-mode test, not LBM
conda run -n heart-conduction python tests/test_mhas13.py
conda run -n heart-conduction python tests/test_paci.py
conda run -n heart-conduction python tests/test_phas13.py
```

#### Exit Criteria
- [ ] `lbm/` gone from V5.5; no dangling references; full suite green; golden still matches.

#### Risk
Hidden import surfacing at runtime — mitigation: the grep above + running the full suite catches any reference. Remove ONLY `step_with_V`; do not touch RushLarsen's `_update_gates` override (`rush_larsen.py:101`) or ForwardEuler's inlined gate update.

### Phase 0 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.5
conda run -n heart-conduction python _regression/check_golden.py
conda run -n heart-conduction python test_phase7.py && conda run -n heart-conduction python test_phase8.py && conda run -n heart-conduction python test_boundary_modes.py
```

### Phase 0 Exit Criteria
- [ ] V5.5 copy exists, docs re-labeled, dead LBM path removed, regression golden captured, full suite green.

### Phase 0 Cleanup
- float64: confirm `make_golden`/`check_golden` force `dtype=torch.float64`, `device='cpu'`.
- V5.3/V5.4 not modified (run `git status` — only `Engine_V5.5/` + golden artifact should be new/changed; if V5.4 shows a diff, revert it).
- V5.5 is intentionally a near-full copy MINUS the dead LBM path — this is the documented backup, not shared-code duplication.

**-> Commit point: git commit after Phase 0 ("Monodomain V5.5: fork of V5.4, drop dead LBM path, + Cm=1 regression golden")**

---

## Phase 1: Cm-correct reaction (the fix)

**Goal**: Plumb the tissue Cm into the ionic step and divide the reaction by it. At Cm=1 the change is a numerical no-op (`/1.0`), so the Phase 0 golden + full suite must still pass bit-identically.
**Tier**: medium
**Estimated scope**: 1 dataclass field, FEM `_Cm`/`_chi` store + 3 trivial properties, 1 orchestrator line, 2 one-line stepper edits

### Phase Context
- **Single source of truth for Cm**: the spatial discretization. The fix reads the SAME Cm the diffusion operator uses, so both operator-split halves agree.
- **Cm storage differs by scheme — VERIFY, don't assume:**
  - `FDMDiscretization` stores `self._Cm` and `self._chi`; operator uses `chi_Cm = self._chi * self._Cm` (`fdm.py:144,199`). ✓ has it.
  - `FVMDiscretization` stores `self._chi`, `self._Cm` (`fvm.py:85-86`). ✓ has it.
  - **`FEMDiscretization` does NOT store `_Cm` or `_chi`** — it bakes them into `self.M = assemble_mass_matrix(mesh, chi, Cm)` at `fem.py:163` and keeps only `_mesh,_n_dof,_x,_y`. ✗ MUST add `self._Cm = Cm` and `self._chi = chi` to its `__init__` (Step 1.1) or the `Cm` property raises `AttributeError`.
- The reaction update appears in TWO steppers: `rush_larsen.py:83` and `forward_euler.py:64`, both currently `state.V = V + dt * (-(Iion + Istim))`.
- `SimulationState` (`state.py`) currently has NO Cm field; `_build_ionic_solver(name, ionic_model)` (`monodomain.py:91`) does not pass Cm. We add `state.Cm` and read it inside the stepper. **Read `spatial.Cm` directly (NOT `getattr(spatial,'Cm',1.0)`)** — a silent fallback to 1.0 would mask a missing-attribute bug (e.g. the FEM defect above) and re-introduce the very Cm bug we're fixing, undetected by the FDM-only golden. Bidomain V1 uses `getattr(state,'Cm',1.0)` at the *stepper* (defensive read of an optional state field, which is fine); here at *construction* we want it to fail loud.
- DO NOT touch any ionic model's internal `Cm` (calcium-flux constant). See Known Failures.

### Step 1.1: Plumb tissue Cm → SimulationState
**Model**: opus

#### Read First
- `…/classical/state.py:24-102` — `SimulationState` dataclass + `__post_init__`.
- `…/classical/discretization_scheme/fdm.py:112-145` — confirms `self._Cm`/`self._chi` stored.
- `…/classical/discretization_scheme/fvm.py:69-86` — confirms `self._chi`/`self._Cm` stored.
- `…/classical/discretization_scheme/fem.py:150-164` — confirms FEM keeps only `_mesh,_n_dof,_x,_y` and bakes Cm into `self.M` (the gap to fix).
- `…/classical/monodomain.py:296-340` — where `spatial` is received and `SimulationState` is built.

#### Why
The stepper needs Cm at `step()` time. Carrying it on `state` (rather than the solver constructor) matches the existing data-on-state design and the Bidomain convention, and keeps `_build_ionic_solver`'s signature unchanged. Failing loud on a missing `spatial.Cm` is deliberate: a silent 1.0 fallback would let FEM (or any future scheme that forgets `_Cm`) divide the reaction by 1.0 while its diffusion uses the real Cm — a half-fixed, silently-wrong engine.

#### Implementation Spec
**Files to modify:**
- `state.py` — add field `Cm: float = 1.0` to `SimulationState` (place after `t: float = 0.0`, before stimulus fields, to keep dataclass default ordering valid).
- `discretization_scheme/fem.py` — in `__init__` (~line 157, alongside `self._mesh = mesh`), add `self._Cm = Cm` and `self._chi = chi` (currently absent — Cm is only baked into `self.M`).
- `discretization_scheme/fdm.py`, `fem.py`, `fvm.py` — add the property (FDM/FVM already have the backing field; FEM now does too after the line above):
  ```python
  @property
  def Cm(self) -> float:
      """Membrane capacitance (uF/cm^2) — single source of truth, shared with reaction."""
      return self._Cm
  ```
- `monodomain.py` (in `__init__`, the `SimulationState(...)` construction ~line 324) — add `Cm=spatial.Cm,` to the kwargs (direct attribute read, NO `getattr` fallback — fail loud if absent).

#### Checklist
- [ ] `SimulationState.Cm` field added (default 1.0)
- [ ] FEM `__init__` stores `self._Cm = Cm` AND `self._chi = chi`
- [ ] `Cm` property on FDM, FEM, FVM (all three return `self._Cm`)
- [ ] `MonodomainSimulation` sets `Cm=spatial.Cm` (direct read, no silent fallback)

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.5
conda run -n heart-conduction python -c "
import torch
from cardiac_sim.tissue_builder.mesh.structured import StructuredGrid
from cardiac_sim.simulation.classical.discretization_scheme.fdm import FDMDiscretization
from cardiac_sim.simulation.classical.discretization_scheme.fvm import FVMDiscretization
g=StructuredGrid.create_rectangle(1.0,0.01,50,1,device='cpu')
for cls in (FDMDiscretization, FVMDiscretization):
    s=cls(g, D=0.001, chi=1.0, Cm=2.0); assert s.Cm==2.0, (cls.__name__, s.Cm)
print('FDM/FVM Cm property OK')
"
# FEM needs a triangular mesh — exercise via the FEM constructor used in test_phase8.py;
# assert FEMDiscretization(...).Cm == 2.0 with chi=1.0, Cm=2.0 (adapt mesh build from test_phase8).
```

#### Exit Criteria
- [ ] `spatial.Cm` returns the configured value for ALL THREE schemes (FDM, FEM, FVM); `state.Cm` populated at construction with no `getattr` fallback.

#### Risk
- Dataclass field-ordering error (non-default after default) — mitigation: `Cm` has a default, place it among the defaulted fields.
- FEM `Cm` property silently returning 1.0 via a `getattr` fallback (the audit CRITICAL) — mitigation: direct `spatial.Cm` read + the FEM `_Cm` storage + the per-scheme Verify above; an `AttributeError` here is the desired loud failure.

### Step 1.2: Divide the reaction by Cm in both steppers
**Model**: opus

#### Read First
- `rush_larsen.py:80-83` and `forward_euler.py:62-64` — the exact reaction lines + their comments.

#### Why
This is the bug fix. The true reaction ODE is `dV/dt = -(I_ion + I_stim)/Cm`; the current code hardcodes the Cm=1 case. Reading Cm off `state` keeps both halves on the same Cm.

#### Implementation Spec
**Files to modify:**
- `rush_larsen.py:83`:
  `state.V = V + dt * (-(Iion + Istim))`  →  `state.V = V + dt * (-(Iion + Istim) / state.Cm)`
- `forward_euler.py:64`: identical change.
- Update the adjacent comment to: `# dV = -(Iion + Istim)/Cm  (Cm from tissue; =1.0 reproduces V5.3/V5.4)`

#### Pseudocode
```
Cm = state.Cm            # tissue capacitance, single source of truth
state.V = V + dt * (-(Iion + Istim) / Cm)
```

#### Test Spec
- Regression: `_regression/check_golden.py` (Cm=1.0) — Expected: `allclose(V, golden, atol=1e-12)` STILL holds (since `/1.0` is exact).

#### Checklist
- [ ] `rush_larsen.py` reaction divides by `state.Cm`
- [ ] `forward_euler.py` reaction divides by `state.Cm`
- [ ] Comments updated
- [ ] `state.Cm` read locally (not `self.ionic_model.Cm` — that's the wrong Cm)

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.5
conda run -n heart-conduction python _regression/check_golden.py   # MUST still pass (Cm=1)
conda run -n heart-conduction python test_phase7.py
conda run -n heart-conduction python test_phase8.py
```

#### Exit Criteria
- [ ] Cm=1 golden + full suite unchanged (no regression).
- [ ] Reaction now scales with Cm (validated in Phase 2).

#### Risk
Division by an integer/None Cm or a tensor-shape surprise — mitigation: `state.Cm` is a Python float; `/float` broadcasts over the (n_dof,) tensor cleanly.

### Phase 1 Verification
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.5
conda run -n heart-conduction python _regression/check_golden.py
conda run -n heart-conduction python test_phase7.py && conda run -n heart-conduction python test_phase8.py
```

### Phase 1 Exit Criteria
- [ ] Cm=1 bit-identical to V5.4 (golden atol=1e-12).
- [ ] All existing tests pass — explicitly including `test_phase8.py` (FEM/FVM per-node conductivity) so the FEM `_Cm` change is exercised at Cm=1.
- [ ] Reaction divides by `state.Cm` in both steppers.
- [ ] FEM and FVM construct with Cm=2.0 and report `spatial.Cm == 2.0` (per-scheme check from Step 1.1 Verify) — confirms the fix reaches all three schemes, not just FDM.

### Phase 1 Cleanup
- float64: no float32 leaks introduced.
- Confirm V5.4/V5.3 untouched (`git status`).
- Confirm NO ionic-model file was edited (the internal-Cm trap).

**-> Commit point: git commit after Phase 1 ("Monodomain V5.5: reaction divides by tissue Cm; Cm=1 regression preserved")**

---

## Phase 2: Cm-scaling validation (the discriminating test)

**Goal**: Prove the fix is physically correct via the time-dilation invariant — scaling tissue Cm by k slows the whole solution by k (and ONLY then). This is the test V5.4 fails and V5.5 passes.
**Tier**: large
**Estimated scope**: one new test script with a 0D test and a tissue-level CV/APD test

### Phase Context
**The invariant.** Normalizing the monodomain PDE by chi*Cm gives `dV/dt = div(D grad V) - (I_ion+I_stim)/Cm` with `D = sigma/(chi*Cm)`. Scaling `Cm -> k*Cm` (sigma, chi fixed) scales BOTH `D -> D/k` and the reaction `-> reaction/k`, i.e. multiplies the whole RHS by `1/k` — exactly a time rescaling `t -> t/k`. Therefore:
```
V(x, t; Cm=k) == V(x, t/k; Cm=1)          (identical spatial structure, time-dilated)
  => Conduction velocity:  CV(Cm=k) == CV(Cm=1) / k
  => Action potential dur:  APD(Cm=k) == k * APD(Cm=1)
```
This requires BOTH halves to scale with Cm. V5.4 (reaction missing /Cm) breaks it; V5.5 satisfies it. (Note: this dilation invariant is what makes the test sharp — it does not depend on any specific ionic model behaviour, only on the structure of the equation.)

**CRITICAL — Formulation A vs B input asymmetry (governs how you set up the "scale Cm" experiment in each engine):**
- **Monodomain V5.4/V5.5 (Formulation A):** the input `D` to `FDMDiscretization(grid, D=…, chi, Cm)` plays the role of **sigma** (conductivity). The engine forms the physical diffusivity internally as `D_input/(chi*Cm)` (the operator is `chi*Cm*I ± θ·dt·L(D_input)`). Therefore to run "fix sigma, scale Cm": **hold `D` fixed and change `Cm`** — the diffusion half then dilates automatically. Do NOT rescale D in the monodomain runs.
- **Bidomain V1 (Formulation B):** the input `D_i, D_e` are the **already-scaled physical diffusivities** `sigma/(chi*Cm)`. So to run the SAME experiment with sigma fixed, you MUST **rescale `D_i, D_e -> D_i/k, D_e/k` when `Cm -> k*Cm`**. If you hold D_i/D_e fixed in bidomain, only the reaction dilates and the cross-check will spuriously fail. (Use `chi=1` so the bookkeeping is clean; `cv_shared.py` uses chi=1.)
- Always compare the dimensionless RATIO (CV(Cm=k)/CV(Cm=1) -> 1/k, APD ratio -> k) rather than absolute CV, so the two engines need not match in absolute units.
- Use CPU + float64 for determinism. Keep grids small so the suite runs fast.
- **Comparison strategy (robust, not pointwise-on-upstroke):** the invariant is exact in continuous time, but a naive whole-trace `allclose@1e-2 mV` between the Cm=k trace and the Cm=1 trace resampled at `t/k` will spuriously FAIL on the upstroke — TTP06 dV/dt exceeds 200 mV/ms there, so a sub-`dt` resampling misalignment produces >>1e-2 mV pointwise error even when the physics is correct. Instead, assert on **robust scalars and a slow-phase window**: (1) **APD90 ratio ≈ k** (the primary, interpolation-insensitive scalar); (2) optionally a pointwise `allclose` restricted to the **plateau + repolarization window** (exclude the upstroke, e.g. from peak+5 ms onward) at a modest atol. Do NOT gate the test on a whole-trace upstroke-inclusive `allclose`.

### Step 2.1: 0D single-cell time-dilation test (isolates the reaction)
**Model**: opus

#### Read First
- `…/ionic/ttp06/model.py` — `get_initial_state`, `V_rest`, `compute_Iion`.
- `rush_larsen.py:45-99` — the stepper you'll drive directly.
- `state.py` — building a 1-dof `SimulationState`.

#### Why
With no diffusion, the only Cm-dependence is the reaction. If a single cell's AP at Cm=k is exactly the Cm=1 AP slowed by k, the reaction fix is correct in isolation — the cleanest possible discriminator, independent of the diffusion solver.

#### Implementation Spec
**Files to create:** `Monodomain/Engine_V5.5/test_phase10_cm_scaling.py` (this step adds `test_0d_time_dilation`).

#### Pseudocode
```
def run_0d(Cm, dt, t_end, k_for_stim_scaling):
    model = TTP06Model(EPI, device='cpu')          # float64
    S = model.get_initial_state(n_cells=1)
    V = full((1,), model.V_rest, float64)
    state = SimulationState(spatial=None, n_dof=1, x=..,y=.., V=V,
                            ionic_states=S, gate_indices=.., concentration_indices=..,
                            Cm=Cm, t=0.0)
    # add a brief suprathreshold stimulus via stim_masks (mask=[1.0]) at t in [t0,t0+dur]
    solver = RushLarsenSolver(model)
    record (t, V) every save step while stepping with dt
    return times, Vtrace

# Cm=1 reference and Cm=k
t1, V1 = run_0d(Cm=1.0, dt, t_end=T)
tk, Vk = run_0d(Cm=k,   dt, t_end=k*T)        # run k× longer to cover the dilated AP
# PRIMARY assertion: robust scalar, interpolation-insensitive
assert isclose(apd90(tk, Vk), k * apd90(t1, V1), rtol=0.02)
# OPTIONAL secondary: pointwise on the SLOW phase only (exclude upstroke)
t_peak_k = tk[int(argmax(Vk))]                 # time of peak V in the Cm=k trace
mask = tk >= (t_peak_k + 5.0)                  # plateau + repolarization window
V1_at = interp(tk[mask] / k, t1, V1)
assert allclose(Vk[mask], V1_at, atol=1e-2)    # NOT applied across the upstroke
```
NOTE on stimulus: the stimulus is itself a current; under the dilation, the stim window would also dilate (`start,dur -> k*start, k*dur`). To avoid that bookkeeping, drive both runs with the SAME early stimulus and compare only the post-stimulus free AP (the chosen variant — document it in the test). The APD90-ratio assertion is the primary gate; the windowed pointwise check is a secondary sanity check that deliberately excludes the fast upstroke.

#### Test Spec
- `test_phase10_cm_scaling.py::test_0d_time_dilation` — Setup: TTP06 EPI, dt=0.01 ms, k=2.0, identical stimulus, post-stim comparison. Expected (PRIMARY): `APD90(Cm=2) ≈ 2·APD90(Cm=1)` within 2%. Expected (SECONDARY, optional): plateau/repolarization-window `allclose@1e-2 mV` (upstroke excluded).

#### Checklist
- [ ] `run_0d` helper drives `RushLarsenSolver` on a 1-dof state with `state.Cm` set
- [ ] PRIMARY gate: `APD90(Cm=k)/APD90(Cm=1) ≈ k` (rtol 2%)
- [ ] SECONDARY (optional): windowed pointwise compare on plateau+repolarization only (upstroke excluded)
- [ ] `apd90` helper implemented (shared with Step 2.2)
- [ ] Test PRINTS a clear PASS/FAIL line (standalone-script convention)

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.5
conda run -n heart-conduction python test_phase10_cm_scaling.py
```

#### Exit Criteria
- [ ] `test_0d_time_dilation` passes on V5.5.

#### Risk
Stimulus dilation confusion — mitigation: use the post-stimulus free-AP comparison and document it. Whole-trace pointwise `allclose` failing on the upstroke despite correct physics — mitigation: gate on the APD90 ratio (robust scalar); restrict any pointwise check to the slow phase.

### Step 2.2: Tissue-level CV/APD scaling test
**Model**: opus

#### Read First
- `monodomain.py:511-591` — `compute_activation_time`, `compute_cv` helpers (reuse them).
- `test_phase7.py` — a working 1D/2D `MonodomainSimulation` setup to copy.

#### Why
The 0D test proves the reaction; this proves the FULL split (reaction + diffusion) scales coherently, i.e. CV and APD obey the dilation. This is the end-to-end physical check and the one that distinguishes V5.5 from V5.4 at the tissue level.

**Why the un-dilated stimulus is acceptable here (unlike 0D):** both cables get the SAME early stimulus rather than a dilated one. Strictly, I_stim is a current that should dilate; but CV is measured by threshold-crossing between two INTERIOR nodes (x1=0.5, x2=1.5) of a wave already propagating freely — the stimulus-site upstroke perturbation has been left behind and largely cancels in the (t2−t1) difference. APD90 is measured at a mid node, also away from the stimulus. The residual stimulus-timing error is well within the 3% tolerance. (If a tighter bound is ever needed, dilate the stim window `start,dur → k·start,k·dur`.)

#### Implementation Spec
**Files to modify:** add `test_tissue_cv_apd_scaling` to `test_phase10_cm_scaling.py`.

#### Pseudocode
```
def run_cable(Cm, t_end):
    grid = StructuredGrid.create_rectangle(Lx=2.0, Ly=0.01, Nx=200, Ny=1, device='cpu')
    fdm  = FDMDiscretization(grid, D=0.001, chi=1.0, Cm=Cm)
    sim  = MonodomainSimulation(fdm, 'ttp06', stimulus=left_edge_stim,
                                dt=0.01, splitting='strang',
                                ionic_solver='rush_larsen',
                                diffusion_solver='crank_nicolson', linear_solver='pcg')
    times, V = sim.run_to_array(t_end, save_every=0.5)   # V shape (n_saves, n_dof)
    cv  = sim.compute_cv(V, times, x1=0.5, x2=1.5, threshold=-20)
    mid_node = 200 // 2                                   # interior node (Nx=200, Ny=1)
    apd = apd90(times, V[:, mid_node])
    return cv, apd

cv1, apd1 = run_cable(Cm=1.0, t_end=80)
cvk, apdk = run_cable(Cm=2.0, t_end=160)   # dilated horizon
assert isclose(cvk, cv1/2.0, rtol=0.03)
assert isclose(apdk, 2.0*apd1, rtol=0.03)
```

#### Test Spec
- `test_phase10_cm_scaling.py::test_tissue_cv_apd_scaling` — Setup: 1D cable, TTP06, k=2, FDM. Expected: `CV(Cm=2) ≈ CV(Cm=1)/2` and `APD90(Cm=2) ≈ 2·APD90(Cm=1)`, both within 3%.
- `test_phase10_cm_scaling.py::test_fem_fvm_cm_scaling_smoke` — Setup: same cable on FEM and on FVM at Cm∈{1,2}. Expected: CV ratio ≈ 1/k within 5% for each scheme (end-to-end guard for the FEM `_Cm` fix).

#### Checklist
- [ ] Cable sim at Cm=1 and Cm=2 (dilated t_end), FDM + crank_nicolson
- [ ] CV ratio ≈ 1/k, APD ratio ≈ k asserted
- [ ] Reuses `sim.compute_cv`; implement a small `apd90` helper (shared with Step 2.1)
- [ ] **FEM + FVM Cm≠1 smoke check**: run the same cable with `diffusion_solver`/discretization swapped to FEM and to FVM at Cm=2, assert CV ratio ≈ 1/k (looser rtol, e.g. 5%, since FEM/FVM CV differs from FDM in absolute terms) — this guards the FEM `_Cm` fix end-to-end, not just at construction. Parametrize `run_cable(scheme=...)`.
- [ ] PRINTS PASS/FAIL + the measured ratios for all three schemes

#### Verify
```bash
cd /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.5
conda run -n heart-conduction python test_phase10_cm_scaling.py
```

#### Exit Criteria
- [ ] `test_tissue_cv_apd_scaling` passes on V5.5 (FDM).
- [ ] FEM and FVM Cm-scaling smoke check passes (CV ratio ≈ 1/k) — confirms the fix works end-to-end on all three discretizations.

#### Risk
Boundary/threshold artifacts in CV — mitigation: measure CV between interior nodes (x1,x2 well inside the cable), threshold -20 mV (matches helper default). If APD ratio is noisy, increase save resolution near repolarization. FEM/FVM explicit-vs-implicit stability at Cm=2 (effective diffusivity = D/(chi·Cm) shrinks, so stability is easier, not harder) — use crank_nicolson as in the FDM case.

### Step 2.3: Bidomain V1 cross-validation (independent Formulation-B oracle)
**Model**: opus

#### Read First
- `Bidomain/Engine_V1/cardiac_sim/simulation/classical/solver/ionic_stepping/rush_larsen.py:81-84` — confirm bidomain's reaction is `Cm=getattr(state,'Cm',1.0); state.V = V + dt*(-(Iion+Istim)/Cm)`. This is EXACTLY the form V5.5 adopts (Formulation B).
- `Bidomain/Engine_V1/tests/cv_shared.py` — reuse `run_bidomain(..., Cm=…, D_i=…, D_e=…)` and `measure_cv_from_history(...)`. Note constants `D_I`, `D_E`, `CM_NUM`, `THRESHOLD`, grid params. **TWO gotchas verified against the source:** (1) `run_bidomain` returns `(times, V_history)` IN THAT ORDER (`cv_shared.py:248`) — and `V_history` is a list of `(Nx,Ny)` tensors. (2) cv_shared has **NO APD utility** — only `measure_cv_from_history` (returns a single CV float). So `bidomain_cm_ref.py` must define its OWN `apd90` over a mid-node voltage trace extracted from `V_history` (or omit APD from the cross-check — CV alone is a sufficient dilation oracle).

#### Why
Bidomain V1 is a SEPARATE engine that shares NO solver code with monodomain, and it is *already* Cm-correct (Formulation B). It is therefore the strongest independent oracle: (1) a **code-parity** anchor — V5.5's fixed line should match bidomain's verbatim; (2) a **physics cross-check** — bidomain run as the same physical "fix sigma, scale Cm" experiment must exhibit the same 1/k CV dilation, and V5.5 must reproduce bidomain's ratio. If both independent engines agree on the dilation, the fix is trusted.

#### Implementation Spec
**Files to create:**
- `Monodomain/Engine_V5.5/_regression/bidomain_cm_ref.py` — RUNS IN THE BIDOMAIN ENGINE (separate process; bidomain also uses the `cardiac_sim` name, so it CANNOT be imported alongside V5.5). It runs a matched isotropic cable in Bidomain V1 at (Cm=1, D_i, D_e) and (Cm=k, **D_i/k, D_e/k**), measures CV at each via `measure_cv_from_history`, computes APD via its OWN `apd90` helper over a mid-node trace, and writes `bidomain_cm_ref.json` = `{k, cm1: {cv, apd}, cmk: {cv, apd}}`. (CV is the primary dilation signal; APD is secondary — if you omit the `apd90` helper, drop the `apd` keys from the JSON rather than leaving them undefined.)
- Add `test_bidomain_cross_validation` to `test_phase10_cm_scaling.py` — loads `bidomain_cm_ref.json`, asserts (a) bidomain itself obeys `cv_cmk ≈ cv_cm1/k` within 3% (sanity: the oracle is correct), and (b) V5.5's own CV ratio (from Step 2.2) matches bidomain's CV ratio within 5%.

#### Pseudocode
```
# bidomain_cm_ref.py — executed with the BIDOMAIN engine on sys.path
from cv_shared import run_bidomain, measure_cv_from_history, D_I, D_E, NX, NY, T_END

def apd90(times, v_trace):   # local helper — cv_shared has none
    vmin, vmax = v_trace.min(), v_trace.max()
    thr = vmin + 0.10 * (vmax - vmin)            # 90% repolarization
    above = [t for t, v in zip(times, v_trace) if v >= thr]
    return (above[-1] - above[0]) if above else float('nan')

k = 2.0
nx, ny = NX, NY                                  # grid dims from cv_shared
mid    = ny // 2                                 # y-index for CV row / mid-node
mid_x  = nx // 2                                 # x-index for the mid-node APD trace
T      = T_END
# Cm=1 baseline — NOTE return order is (times, V_history); V_history is list of (Nx,Ny)
t1, V1 = run_bidomain(Cm=1.0, D_i=D_I,   D_e=D_E,   t_end=T)
cv1  = measure_cv_from_history(V1, t1, y=mid)
apd1 = apd90(t1, [v[mid_x, mid].item() for v in V1])     # mid-node trace
# Cm=k WITH diffusivities rescaled by 1/k (hold sigma fixed)
tk, Vk = run_bidomain(Cm=k,   D_i=D_I/k, D_e=D_E/k, t_end=k*T)
cvk  = measure_cv_from_history(Vk, tk, y=mid)
apdk = apd90(tk, [v[mid_x, mid].item() for v in Vk])
json.dump({'k': k, 'cm1': {'cv': cv1, 'apd': apd1}, 'cmk': {'cv': cvk, 'apd': apdk}}, ...)

# test_bidomain_cross_validation (in V5.5 process)
ref = json.load('bidomain_cm_ref.json')
assert isclose(ref['cmk']['cv'], ref['cm1']['cv']/ref['k'], rtol=0.03)   # oracle is self-consistent
mono_ratio = cvk_v55 / cv1_v55      # from Step 2.2 cable runs (chi=1, D fixed, Cm scaled)
bido_ratio = ref['cmk']['cv'] / ref['cm1']['cv']
assert isclose(mono_ratio, bido_ratio, rtol=0.05)                        # cross-engine agreement
```

#### Test Spec
- `test_phase10_cm_scaling.py::test_bidomain_cross_validation` — Setup: load bidomain ref JSON; k=2. Expected: bidomain self-consistent (CV ratio ≈ 0.5, 3%); V5.5 ratio matches bidomain ratio (5%).

#### Checklist
- [ ] `bidomain_cm_ref.py` rescales `D_i, D_e` by 1/k when scaling Cm (Formulation B requirement — see Phase Context)
- [ ] Uses the correct `run_bidomain` return order `(times, V_history)` (NOT `(V, t)`)
- [ ] Defines its own `apd90` (cv_shared has none); or drops `apd` from the JSON
- [ ] Reference JSON generated in a SEPARATE process (bidomain engine), checked into `_regression/`
- [ ] V5.5 test loads JSON, asserts oracle self-consistency AND cross-engine ratio agreement
- [ ] (Optional secondary) note absolute CV(Cm=1) agreement vs bidomain if parameters matched (`run_monodomain_fdm` already matches bidomain at 54.3 cm/s per project memory)
- [ ] PRINTS PASS/FAIL + both ratios

#### Verify
```bash
# 1) generate bidomain reference (separate process, bidomain engine on path)
cd /home/norepinephrine/Documents/Heart-Conduction/Bidomain/Engine_V1
conda run -n heart-conduction python /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.5/_regression/bidomain_cm_ref.py
# (script writes bidomain_cm_ref.json into Engine_V5.5/_regression/)
# 2) run the cross-validation in the V5.5 process
cd /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.5
conda run -n heart-conduction python test_phase10_cm_scaling.py
```

#### Exit Criteria
- [ ] Bidomain reference is self-consistent (obeys 1/k dilation).
- [ ] V5.5's CV dilation ratio matches Bidomain V1's within tolerance.

#### Risk
Forgetting the D_i/D_e rescale in bidomain (Formulation B) → bidomain won't dilate → false failure. Mitigation: the checklist + Phase Context call this out explicitly; the test's "oracle self-consistency" assertion catches a mis-set bidomain run before the cross-engine compare. Cross-engine absolute CV mismatch from discretization differences (stencil, threshold, dx) — mitigation: compare RATIOS, not absolute CV.

### Phase 2 Verification
```bash
# generate bidomain reference once (separate process / bidomain engine):
cd /home/norepinephrine/Documents/Heart-Conduction/Bidomain/Engine_V1
conda run -n heart-conduction python /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.5/_regression/bidomain_cm_ref.py
# run the full Cm-scaling suite (0D + tissue + bidomain cross-val):
cd /home/norepinephrine/Documents/Heart-Conduction/Monodomain/Engine_V5.5
conda run -n heart-conduction python test_phase10_cm_scaling.py
```

### Phase 2 Exit Criteria
- [ ] All three Cm-scaling tests pass on V5.5 (0D, tissue, bidomain cross-val).
- [ ] Bidomain V1 reference generated and self-consistent; V5.5 ratio matches it.
- [ ] (Documentation) Note in IDEALOG that V5.4 fails `test_tissue_cv_apd_scaling` (run it against a V5.4 cable in a separate process to confirm the discriminator, optional but recommended — do NOT add the test file to V5.4).

### Phase 2 Cleanup
- float64 + CPU in all new tests for determinism.
- No EXPERIMENT.md needed (these are engine unit tests, not a research experiment); if a CV/APD-vs-Cm sweep figure is produced, that WOULD need an `experiments/` folder + EXPERIMENT.md backlink.
- V5.4/V5.3 untouched.

**-> Commit point: git commit after Phase 2 ("Monodomain V5.5: Cm-scaling time-dilation validation (0D + tissue)")**

---

## Final Cleanup (cross-phase de-sloppify)
- [ ] `git status` shows changes ONLY under `Monodomain/Engine_V5.5/` (+ this PLAN/IDEALOG/README/MASTER updates). V5.4 and V5.3 pristine.
- [ ] No `float32` leaks anywhere new (`grep -rn "float32" Engine_V5.5/test_phase10_cm_scaling.py _regression/`).
- [ ] No ionic-model file modified (internal-Cm trap): `git -C Monodomain/Engine_V5.5 diff --stat` touches only state.py, the 3 discretization schemes, monodomain.py, the 2 steppers, the deleted `lbm/` package + `step_with_V` removal, and new test/regression files.
- [ ] Update `Research/Active/engine_consolidation/README.md` — add a "V5.5 Cm-correct monodomain fork (prerequisite)" line under the **## Completion Criteria** section (README.md:15-26; there is no section literally named "migration plan").
- [ ] Update `Engine_V5.5/PROGRESS.md` with V5.5 phase status.
- [ ] Archive this plan:
```bash
mkdir -p Research/Active/engine_consolidation/plans
cp Research/Active/engine_consolidation/PLAN.md \
   "Research/Active/engine_consolidation/plans/2026-05-29_monodomain-v5.5-cm-correct-reaction-fork.md"
```
- [ ] Append an IDEALOG Thread entry recording outcomes (CV/APD ratios measured, any tolerance choices) and confirm the V5.4-fails-the-invariant cross-check.

## Mutation Log
_(execution-time mutations go here: `**MUTATED {date}**: Step X.Y SKIPPED/SPLIT/INSERTED — reason`)_

### Revision 2026-05-30 — adversarial audit findings (11 issues)
- **MUTATED 2026-05-30**: Step 1.1 MODIFIED (audit CRITICAL) — FEMDiscretization does NOT store `self._Cm`/`self._chi` (bakes Cm into `self.M` at `fem.py:163`); the prescribed `Cm` property would `AttributeError`, silently masked by the old `getattr(spatial,'Cm',1.0)` fallback → FEM would divide reaction by 1.0 while diffusion used real Cm. Fix: add `self._Cm = Cm` + `self._chi = chi` to FEM `__init__`; change plumbing to read `spatial.Cm` directly (fail loud, no fallback); per-scheme Verify for FDM/FEM/FVM. Phase Context + Architecture-line-23 updated.
- **MUTATED 2026-05-30**: Step 2.3 MODIFIED (audit HIGH) — `run_bidomain` returns `(times, V_history)`; pseudocode had `V1, t1 = ...` reversed. Fixed to `t1, V1 = ...`, added the `(Nx,Ny)`-list note and the correct `measure_cv_from_history(V_hist, times, y)` order.
- **MUTATED 2026-05-30**: Step 2.3 MODIFIED (audit HIGH) — cv_shared has NO APD utility; `apd1/apdk` were undefined. Added a local `apd90` helper in `bidomain_cm_ref.py` (or drop `apd` from JSON). Read-First gotcha added.
- **MUTATED 2026-05-30**: Architecture line 25 MODIFIED (audit MEDIUM) — shorthand `/(Iion+Istim)` → `/((Iion+Istim)/state.Cm)` mis-implied the code already divided; replaced with the real before/after.
- **MUTATED 2026-05-30**: Step 0.3 MODIFIED (audit MEDIUM) — documented that the `chi=1.0` golden is blind to chi-only regressions (acceptable for this Cm-only fork).
- **MUTATED 2026-05-30**: Step 1.2/Phase-1 Exit + Step 2.2 MODIFIED (audit MEDIUM) — added FEM/FVM regression coverage: Phase 1 explicitly runs `test_phase8.py` (FEM/FVM) + per-scheme `spatial.Cm` check; Phase 2 adds `test_fem_fvm_cm_scaling_smoke` (CV ratio ≈ 1/k on FEM & FVM).
- **MUTATED 2026-05-30**: Step 2.2 MODIFIED (audit MEDIUM) — documented why the un-dilated stimulus is acceptable at tissue level (interior-node CV measurement cancels stimulus-site perturbation; 3% tolerance covers it).
- **MUTATED 2026-05-30**: Step 2.1 + Phase-2 Context MODIFIED (audit LOW) — replaced whole-trace `allclose@1e-2 mV` (fails on >200 mV/ms upstroke) with APD90-ratio primary gate + optional plateau/repolarization-window pointwise check.
- **MUTATED 2026-05-30**: Step 0.4 MODIFIED (audit LOW) — corrected the inaccurate "`_update_gates` used by Rush-Larsen" note (RushLarsen overrides it; ForwardEuler inlines; base becomes dead but harmless to keep).
- **MUTATED 2026-05-30**: Steps 0.1/0.2/0.4 MODIFIED (audit LOW) — added explicit "N/A — pure file op" Pseudocode/Test Spec markers for the 9-section structural rule.
- **MUTATED 2026-05-30**: Final Cleanup MODIFIED (audit LOW) — README target corrected from "migration plan" to the actual "## Completion Criteria" section.

### Revision 2026-05-30 (pass 2) — second-audit findings (4 issues)
- **MUTATED 2026-05-30**: Step 2.1 pseudocode MODIFIED (audit MEDIUM) — defined `t_peak_k = tk[argmax(Vk)]` (was referenced in the slow-phase mask but undefined; introduced by pass-1's own LOW rewrite).
- **MUTATED 2026-05-30**: Step 2.2 pseudocode MODIFIED (audit MEDIUM) — defined `mid_node = 200 // 2` in `run_cable` (was undefined).
- **MUTATED 2026-05-30**: Step 2.3 pseudocode MODIFIED (audit MEDIUM) — imported `NX, NY, T_END` from cv_shared and defined `nx, ny, mid, mid_x, T` (lowercase `ny`/`mid_x`/`T` were undefined; `V_history` elements are `(Nx,Ny)` so both x- and y-indices are needed).
- **MUTATED 2026-05-30**: Architecture line 24 MODIFIED (audit LOW) — reworded to "pass `Cm=spatial.Cm` as a kwarg in the `SimulationState(...)` call" to match Step 1.1's constructor form (was "set state.Cm = spatial.Cm", which read like a post-hoc assignment).
