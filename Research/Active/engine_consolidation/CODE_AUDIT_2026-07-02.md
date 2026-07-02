# cardiac_core Deep Code Audit — Math Integrity + API (2026-07-02)

> **Backlinks:** [KNOWLEDGE.md](./KNOWLEDGE.md) · [IDEALOG.md](./IDEALOG.md) ·
> [API_CONSISTENCY_AUDIT.md](./API_CONSISTENCY_AUDIT.md) (the prior API-surface audit) ·
> [MASTER.md](../../../MASTER.md)
>
> **Scope.** A full-surface agentic walkthrough of `cardiac_core/`: numerical math integrity of each
> engine (monodomain / bidomain / LBM) + shared ionic models + conductivity, then the API layer
> (factories, simulation wrapper, run/io/analysis/mesh). Distinct from the prior
> `API_CONSISTENCY_AUDIT.md` (which was API-surface parity only) — this one checks the **numerics**.

## Method
6 parallel deep auditors, each grounding findings in the **actual code AND numerical checks it ran**
(float64, `heart-conduction` env), not the test suite (explicitly: "a passing suite ≠ correct math").
Lanes: monodomain math · bidomain math · LBM math · ionic+conductivity · API factories/simulation ·
run/io/analysis/mesh. The orchestrator then **independently re-verified the two cheapest actionable
majors** (M1, M3) and characterized M1's trigger; M2/M4 rest on the auditors' detailed reproductions.

## Verdict

**0 BLOCKERS · 4 MAJORS · ~22 MINORS.**

The **default paths and all recently-shipped work are sound.** Independently confirmed correct:
- **Default-path math, all 3 engines** — monodomain (PCG + Strang + Rush-Larsen + cardinal4 FDM +
  face_mirror), bidomain (decoupled GS + spectral/PCG elliptic + face-based SPD stencil), LBM (BGK/MRT +
  bounce-back). CV matches reference (bidomain 54.14 vs 54.35 cm/s; mono dx-independent).
- **The prior cross-derivative bug is genuinely fixed** — mono `cxy = 1/(2·dx·dy)` gives exactly `2·Dxy`
  (sign + magnitude), matches the bidomain builder to 1e-13.
- **Time-stepper orders exact** — CN 2.00, BDF2 2.05, RK4 4.01, FE/BDF1 1.00.
- **Reaction Cm-scaling (V5.5 fix)** correct on mono + bidomain (`dV·Cm` invariant to ~5e-14, Cm∈{0.5,1,2,4}).
- **LBM Chapman–Enskog** recovers the intended D to sub-percent (BGK D2Q5/D2Q9 + MRT per-axis anisotropy);
  the D_xy-discarded limitation is real and honestly documented.
- **The Phase-1–4 hardening** (this branch): `lbm_step_d2q9_mrt_wall` rest-neutral/mass-conserving/
  collision-agnostic; masked-bounce union seals hole rim + outer walls (zero leak); build_kwargs replay
  complete for every new param; `_resolve_mesh` deepcopy isolation; warnings correctly scoped; P1/I3.
- **Ionic models** — TTP06/ORd/PHAS13/MHAS13 all stable at rest + physiological APs with correct
  cell-type APD ordering; Rush–Larsen unconditionally stable; LUT <1e-6; registry behavior-preserving.
- **Conductivity firewall** — D_eff=9.72e-4 to machine precision; Form-A/B Cm-trap correct at Cm≠1.
- **Analysis math** — CV/APD/restitution/dominant-freq/phase-singularity correct on synthetic fronts.

---

## MAJORS

### M1 — [monodomain] Chebyshev solver: Gershgorin bounds on raw `A`, not the preconditioned `D⁻¹A`
- **Location:** `cardiac_core/_monodomain/simulation/classical/solver/diffusion_time_stepping/linear_solver/chebyshev.py:146-160` (`_estimate_eigenvalues`→`_gershgorin_bounds(A)`).
- **Problem:** With Jacobi preconditioning on (the default), the iteration acts on `D⁻¹A` (spectrum ~[0.04, 1.96]) but the Chebyshev interval is taken from Gershgorin circles on the raw `A` (~[1.4, 83]) → the polynomial is optimized for the wrong interval and barely damps the error.
- **Evidence (orchestrator-reproduced):** stiff CN system (effD=0.5, dt=0.05): `max|V_pcg − V_cheb| = 94.4 mV` — completely wrong. Physiological (effD=1e-3, dt=0.02): agree to 2.6e-6. **Config-dependent: severe for wide operator spectrum (high `D·dt/dx²`), benign otherwise.** Auditor: 27% error @200 iters; correct preconditioned bounds → 2.8e-7 @60 iters.
- **Cross-engine inconsistency:** the **bidomain** Chebyshev already has this fix (`_gershgorin_bounds_preconditioned`, the "CH-1 FIX"); the monodomain copy never got it.
- **Reachability:** opt-in only (`linear_solver='chebyshev'`; default is `pcg`). No test exercises Chebyshev accuracy → passed CI unnoticed.
- **Fix:** port `_gershgorin_bounds_preconditioned` from bidomain (centers=1, radius = off-diag row-sum/|diag| when Jacobi is on). Also fix `set_eigenvalue_bounds` (sets `_A_id=None` → bounds get clobbered on next solve; never extracts `_diag_inv` → manual bounds silently disable preconditioning).

### M2 — [monodomain] FFT periodic spectral solver inverts continuum `−k²`, not the discrete 5-point Laplacian
- **Location:** `.../linear_solver/fft.py:299-317` (`FFTSolver._compute_eigenvalues`: `_eigenvalues = -K2`).
- **Problem:** the assembled operator is the discrete 5-point stencil (periodic eigenvalues `−4·sin²(πk/N)/dx²`); the solver uses the continuum `−k²`. They agree only as k→0; ~10% off by mid-band, growing with frequency → the FFT solver does not invert the operator the engine assembles.
- **Evidence (auditor):** mode (2,3) on 16×16: solver eigenvalue −513.22 vs true discrete −466.03; RHS built from the discrete eigenvalue → 1.3e-15 recovery, from continuum → 7.5e-2. The sibling **DCT** (Neumann) solver uses the correct discrete eigenvalue (matches assembled operator to 2e-12) — only FFT is wrong.
- **Reachability:** opt-in periodic spectral solve (not the cardiac default DCT/Neumann).
- **Fix:** use `λ = −4·sin²(π·i/N)/dx²` (≡ `(2/dx²)(cos(2π·i/N) − 1)`).

### M3 — [API] `CardiacSimulation.step()` raises `AttributeError` on bidomain sims
- **Location:** `cardiac_core/api.py:232-237`.
- **Problem:** the non-LBM branch dispatches `self._engine.step()`, but `BidomainSimulation` has **no `step()`** (only a `run()` generator). The if/else is also dead code (both branches identical).
- **Evidence (orchestrator-reproduced):** `cc.bidomain(mesh).step()` → `AttributeError: 'BidomainSimulation' object has no attribute 'step'`; `mono.step()` and `lbm.step()` both OK.
- **Reachability:** live public method, pre-existing (not from Phases 1–4), untested (no test calls `.step()`).
- **Fix:** for bidomain, advance one step via the splitting strategy (`self._engine.splitting.step(state, dt)` + advance `state.t`) and collapse the dead if/else. NOTE: stepping the splitter directly does not advance `state.t` — the fix must set it (two auditors independently hit this).

### M4 — [bidomain] Spatially-varying anisotropy → non-symmetric elliptic operator → silent ~13% wrong `phi_e`
- **Location:** `cardiac_core/_bidomain/simulation/classical/discretization/fdm.py:506-534` (9-pt cross-derivative term).
- **Problem:** the cross-term weights use **only the center node's `D_xy`**. Fine when `D_xy` is spatially uniform (weights cancel → operator stays symmetric, verified `rel_sym_err=0.0`), but a per-node fiber-angle / `D_xy` field gives `(k, k±NE)` and its transpose different `D_xy` → symmetry breaks.
- **Evidence (auditor):** 25×25 varying fiber field: A_ellip `rel_sym_err = 0.85–1.3%` (macroscopic), complex eigenvalues (`max|Im|=6e-3`), symmetric-part min-eig slightly negative (not SPD). Consequence: plain `PCGSolver` → `converged=False` after 5000 iters; the auto-selected `PCGSpectralSolver` certifies a tiny internal residual but returns `phi_e` **13.5% wrong vs the exact dense solve** (independent of preconditioner D → it's the nonsymmetry, a classic silent CG-on-nonSPD failure).
- **Reachability:** the uniform-angle public API (`ConductivityConfig.anisotropic(σ_l, σ_t, fiber_angle)` scalar angle) is **SAFE**. Hit by **per-node fiber fields** — the legacy `bidomain(mesh=...)` path with a fiber-angle map, or `BidomainConductivity(D_i_field=…, theta=<field>)`. **Most serious finding: a silent wrong result in a reachable regime.**
- **Fix:** build the cross-term in flux form (average `D_xy` across the diagonal face pair so `(k,m)` and `(m,k)` share one weight), or symmetrize the assembled matrix; AND guard the auto-selector to refuse CG-family solvers when `L_i+L_e` isn't symmetric (fail loud, not silent-wrong).

---

## MINORS (by area)

**API / factories (`api.py`)**
- Declarative pre-built ionic **instance** leaks into `data.ionic_model` (`_build_mesh_data:963-971`) → `.ionic_model` returns an object not a name, and `save_cardiac_mesh` fails (object-array pickle). Legacy `mesh=` path is fine (instance stays in build_kwargs). Fix: coerce to a string name.
- `combined`/`specular_*` boundary modes are unusable with the default `lattice='d2q5'` (raise) and the `boundary` docstring doesn't say to set `lattice='d2q9'`. Fix: auto-upgrade (like anisotropic does) or document.
- `dt` property + `timestep = dt or data.dt` (api.py:744, 1130, 1258, 1440) use `or`-on-numeric → a legitimate `dt=0.0` silently falls back to the mesh dt. Prefer `... if x is not None else`.
- `add_stimulus` default amplitude −80 vs `stimulate`/`_normalize_stimulus` −52 (documented as aliases). Align.
- `record=(…)` tuple not validated (unknown keys silently ignored; `record=("ionic",)` typo → no data, no error). [INFO]

**LBM (`_lbm/`)**
- Misleading MRT comment: `collision/mrt/d2q9.py:11,44` say `p_xx` "encodes D_xx − D_yy" — but anisotropy is carried by the flux rates `s_jx/s_jy`; `p_xx` is a free stability rate (verified: varying it changes D by ≤2.4e-10). Code correct, comment wrong (the function docstring :63-69 is already right).
- No stability gate on the BGK construction path (`simulation.py:125-131`) — MRT calls `check_stability_tensor` and raises on τ≤0.5, BGK doesn't. `check_stability` exists but is never called. Fix: call it symmetrically.
- Interior hole cells are sealed (0.0 leak, verified) but not zeroed — `V[hole]` holds evolving garbage and total-grid mass grows. Cosmetic; document that V is valid only on `data.mask`, or zero hole distributions at init.

**Monodomain (`_monodomain/`)**
- DCT/FFT spectral solvers silently ignore `A` and assume `face_mirror`/periodic + uniform scalar D → **34% error** for `node_mirror_existing` (and masked/heterogeneous D). Fix: guard (permit spectral only for the matching boundary_mode + uniform D + no mask, else raise).
- `Grid`/`structured.py` `__post_init__`: `dy = Ly/(Ny-1)` → `ZeroDivisionError` on 1-D (Ny=1) / single-cell (1×1) grids. Fix: guard `Nx==1`/`Ny==1`.
- Dead/redundant line `fft.py:177` (recomputes `u_dct[0,0]`) with a misleading "set mean to zero" comment.

**Bidomain (`_bidomain/`)**
- Stale scalar `D_i`/`D_e` used to build the spectral preconditioner for field-based conductivity (`bidomain.py:163,179`) — default scalars regardless of the actual field. Harmless when it only weakens the preconditioner; fix: derive from the trace of the actual tensor field.
- Inconsistent PCG breakdown thresholds: `pcg.py:203` (`pAp <= 0`) vs `pcg_spectral.py:111` (`|pAp| < 1e-14·b²`) → the two diagnose a borderline system differently. Unify + add a hard SPD/symmetry assertion at operator-build.

**Ionic + conductivity (`ionic/`, `conductivity.py`)**
- LUT interpolation error spikes to 1.2% at the `V=−40 mV` piecewise kink in `INa_h_tau/j_tau` (~2e-5 elsewhere); physically negligible (0.67 ms on 53 ms). Optional: place a grid node at V=−40.
- No guards on non-physical conductivity (`sigma≤0` silent; `sigma_i+sigma_e=0` → ZeroDivisionError). User-error inputs; a validating classmethod would fail louder.
- `ORdModel.step()` mutates the input `ionic_states` in place (TTP06/PHAS13 build a fresh stack) — safe in the reassignment pattern, but a hazard for adjoint/checkpointing wrappers.
- `ORdModel.compute_concentration_rates` raises `NotImplementedError` (by design — SR/CaMKII coupling; use `step()`). Flag only so ABC callers know ORd doesn't support the external-solver path.

**run / io / analysis / mesh (support layer)**
- LBM `save_every` non-integer multiple of `dt` silently quantizes the save cadence (`int(round(save_every/dt))`), while classical engines use a running accumulator → the two disagree on the time axis for identical `(save_every, dt)`. Fix: warn or use an accumulator in `_run_lbm`.
- `activation_time` (`analysis.py:47`) raises `IndexError` on a 0-save history instead of returning NaN (siblings `conduction_velocity`/`apd_*` degrade gracefully). Reachable via `t_end < save_every` + `SimulationResult.lat()`. Fix: guard `V.shape[0]==0 → NaN`.
- `dominant_frequency` (`analysis.py:220`) `ZeroDivisionError` on a single frame. Fix: early-return NaN for n<2.
- Empty eager result `Vm` shape is `(0,)` not `(T,Nx,Ny)` (`api.py:983`, `run.py:88`) — upstream cause of the `activation_time` crash. Fix: `torch.empty(0, Nx, Ny)`.
- `boundary_distance` docstring says "chamfer (city-block)" but the code is Euclidean (EDT); also "0 at boundary" is inaccurate (boundary nodes are `1·dx`). Docstring fix.
- `group_labels` round-trips as `numpy.str_` not `str` (`file_format.py:165`). Cosmetic.

---

## Cross-cutting themes
1. **Opt-in solvers carry the real math bugs; the default PCG path is sound.** M1 (Chebyshev), M2 (FFT), and the DCT/FFT-ignore-`A` minor all bite only when a user selects a non-default `linear_solver` — and all fail *silently* (plausible output, wrong numbers). Consider guarding/eliminating the fragile spectral/polynomial paths, or gating them behind validated preconditions.
2. **Silent wrong-results under anisotropy (M4)** — the one finding that hits a physically-motivated regime (fiber fields). Same family as the deferred C7/oblique theme: the FDM cross-derivative is the recurring weak spot.
3. **Degenerate-input handling is a crash-instead-of-NaN family** — `activation_time`, `dominant_frequency`, `Grid` 1-D/single-cell, empty-result shape. A small "0/1-element robustness" sweep would close all four.
4. **Cross-engine inconsistencies** — Chebyshev fix in bidomain-not-mono (M1); `step()` in mono/lbm-not-bidomain (M3); `save_every` accumulator vs quantized; PCG breakdown thresholds; stimulus default amplitude. The consolidation left per-engine copies that have drifted.
5. **Docstring/behavior mismatches** — `boundary_distance` (Euclidean vs city-block), MRT `p_xx`, DCT "mean to zero". Low-risk but erode trust.

## Recommended fix priority (if pursued — NOT yet actioned)
- **P1 (silent-wrong in a reachable regime):** M4 (symmetrize the bidomain cross-term + guard CG on non-SPD).
- **P2 (opt-in correctness):** M1 (port bidomain's preconditioned Gershgorin to mono Chebyshev + fix `set_eigenvalue_bounds`), M2 (discrete FFT eigenvalue), DCT/FFT precondition guard.
- **P3 (robustness / footguns):** M3 (bidomain `step()`), the degenerate-input NaN guards (analysis + Grid + empty shape), BGK stability gate, `dt or` → `is not None`, ionic-instance-leak coercion.
- **P4 (cleanliness):** docstrings, `add_stimulus` amplitude, hole-cell zeroing, LUT kink, conductivity input guards, PCG-threshold unification, `save_every` cadence.

## Status
Audit only — **no source modified.** Findings recorded here for triage. The default paths, the prior
API-consistency hardening (Phases 0–5), and all four ionic models are verified sound; the 4 majors are
2 opt-in-solver bugs, 1 pre-existing untested public method, and 1 anisotropic-fiber silent-wrong-result.
Verification scripts under the session scratchpad (`verify_majors.py`, `verify_cheb.py`, per-auditor scripts).
