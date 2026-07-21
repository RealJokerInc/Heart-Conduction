# Engine Consolidation — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan this in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
**SOLVER HARDENING — SHIPPED (2026-07-21, branch `solver-hardening`, 4 commits, NOT merged).** The audit findings
below drove a fix campaign ("work through all"): Step 1 (non-convergence signal + Chebyshev M1 fix), Step 2 (mid-run
voltage clamp + state injection), opt-in solver fixes (pcg_spectral mixed-BC, IMEX-SBDF2, RKC doc-defer). All
test-gated, integrity goldens bit-identical, full suite 260/2. **DEFERRED awaiting user decision:** #13 GPU sync-free
PCG (regolden or GPU-only) + #14 mono-ionic V5.3 alignment (regolden default — user flagged doubt it's a bug). See
the 2026-07-21 Session-Log entry + the "Solver hardening — SHIPPED" callout in KNOWLEDGE. **NEW SIDEQUEST:** an
"Intro to Cardiac Core" 8-lesson Jupyter tutorial series — plan at `cardiac_core/tutorials/PLAN.md` (prep-first;
L1 single-cell → L8 bidomain infarct + mixed BC). Also assessed library packaging: importable (cardiac-core 0.1.0)
but deps under-declared (only mcp; torch/numpy/scipy/torch_dct missing) + no README/LICENSE/__version__.

**Audit provenance (2026-07-16). No solver code changed at audit time (audit + measure only).**
6-lane adversarial audit + empirical GPU benchmark; every HIGH/MED finding independently
reproduced. Full ranked table + GPU characterization are in **KNOWLEDGE.md → "Solver +
GPU audit — 2026-07-16"** (the reference; scan there). Scratchpad artifacts: `gpu_bench.py`/`gpu_bench_results.json`,
`FINDINGS_task6_gpu.md`, `AUDIT_collation.md`, `cheby_repro.py` + agent repros.

**Verdicts (30-sec scan):**
- **device='cuda' IS using the GPU** — full residency cuda:0/float64 across all 3 engines; result hooks on cuda.
- **The "crossover weirdness" = per-iteration GPU→CPU host syncs in the iterative solvers, NOT a CPU-compute
  fallback** (the user's mental model was close but the mechanism is a pipeline stall, not offload). Syncs/step:
  explicit 0, mono CN+pcg 24, CN+dct 1; bidomain default (pcg+pcg_spectral) syncs the heaviest; LBM = 0. GPU
  per-step is launch-latency bound (~6-10 ms flat) → wins over CPU only above ~10k dof. float64 on a 1:64-FP64 card.
- **2 HIGH silent-wrong bugs** — (1) mono Chebyshev-Jacobi tunes to raw A not D⁻¹A → 46% err at high diffusion-number
  (opt-in, machine-precision at default dt); (2) bidomain pcg_spectral singular Neumann precond on anisotropic
  mixed-BC → stalls, wrong phi_e. **Systemic MED:** ALL iterative solvers silently return unconverged as converged.
  IMEX-SBDF2 silently 1st-order; RKC refinement-immune ~0.8% err; mono ionic conc-currents use post-RL gates
  (diverges from V5.3, inherited from V5.4 — bidomain copy is correct). **The DEFAULT mono (pcg+CN) and bidomain
  (pcg auto) paths are correctness-solid; risk is opt-in solvers failing silently.** LBM audit = clean.
- **Highest-value single fix (future):** a shared non-convergence signal (warn/raise + surface residual) — closes
  the systemic finding across 4 lanes and makes the HIGH bugs fail LOUD instead of silent.

**Next-direction plan recorded (task #9, detail in KNOWLEDGE):** advanced features (masked per-step voltage clamp +
mid-run state injection, both via one `_stepping_run` per-step hook in CardiacSimulation) → GPU opt follow-ups (PCG
sync-free convergence, auto-dct-on-GPU, isotropic-bidomain→Tier-1 spectral, COO→CSR, torch.compile LBM) →
consolidation Phase 2-5 (mesh/stimulus/ConductivityConfig unify → engine rewire+delete [blocker: Surrogate/Optimizer
consumers] → clean namespace). Advanced features are independent of the dedup phases and can land first.

**What NOT to retry / gotchas learned this session:**
- A quick end-to-end pcg-vs-chebyshev compare at the DEFAULT config shows them AGREEING (both ~machine precision) —
  do NOT conclude Chebyshev is fine from that. The Jacobi-bounds bug only bites at high diffusion-number (off/diag
  ≳0.24); you must sweep dt/dx to see it. Goldens don't catch it (they froze the safe regime).
- cardiac_core faithfully copied V5.4; the mono-ionic conc-ordering deviation from V5.3 is a V5.4-lineage defect, NOT
  a copy bug. Self-goldens can't catch V5.3 divergence (they're self-referential).
- Bidomain declarative path ALWAYS builds D_i_field (even isotropic) → is_isotropic=False → never auto-selects the
  fast Tier-1 direct 'spectral' solver.

---
**PLAN.md EXECUTED — P0/P1/P2 usability fixes SHIPPED (2026-07-16, branch `usability-fixes-p0-p1`, NOT yet merged to main).**
All 5 PLAN phases done + a 5-lane adversarial audit of the whole branch (see the 2026-07-16 final-audit Thread
entry — 4 more real bugs found + fixed), each test-gated + per-engine integrity goldens bit-identical (atol=0).
Commits P1 `a37d325` → P2 `d78a86d` → P3 `d94aa6d` → P4 `d6a3237` → P5 `99e1fa3` → audit-remediation `c0306d2` → round-2 remediation `9f387ef`.
Suite **260 passed / 2 xfailed** (218 baseline + 42 tests in `cardiac_core/tests/test_usability_fixes.py`).
- **P1 six P0 bugs:** B1 GPU device-mismatch (`_result_from` builds `times` on Vm's device); B8 NaN-fill masked
  nodes (`StructuredGrid.flat_to_grid`, mono+bidomain; LBM has no flat_to_grid, untouched); B3/B4 `apd_at`
  beat-bounded peak + dome-aware LAST-crossing (spike-and-dome safe); B5 `Grid(N,1)` degenerate-axis guard;
  B6 `forward_euler` CFL warn (FDM retains `_D_max`); B7 `record=` key validation.
- **P2 (B2):** DCT/FFT wired through `_build_linear_solver(spatial,dt,scheme)` + `_spectral_kwargs` — fast path
  restored (was TypeError→silent PCG fallback = the runtime wall). DCT CV matches PCG; default pcg untouched;
  full-rectangle only (masked → pcg).
- **P3:** de-trapped ~18 `NotImplementedError` stubs (informative errors, removed misleading `>>>` examples);
  IMPLEMENTED `scale_conductance` / `set_conductivity` / `scale_conductivity` (rebuild-from-t=0). Adversarial
  audit of the diff caught 2 CROSS-ENGINE bugs (my mono-only tests missed them): (i) declarative bidomain uses
  `sigma_i/sigma_e` fields not `D_xx` → scar was a SILENT no-op → fix applies the mask op to sigma too, and a
  nonzero absolute D on sigma-bidomain now RAISES; (ii) `scale_conductance` re-derived the model from
  name+mesh-cell_type but bidomain/LBM build ENDO by default → CELL-TYPE FLIP (Gto 0.073→0.294) → fix
  deep-copies the LIVE engine model (`_live_ionic_model()`), preserving cell type + prior scalings.
- **P4:** `API_CHEATSHEET.md` rewrite — Solver&dt section, drug/conductance map (**PCa = ICaL, NOT "GCaL"**;
  ORd adds GNaL), **ORd LBM-only** (raises on mono/bidomain) + paci/phas13/mhas13 on mono, `record=`/
  save_result arg-order/df/two-step phase_map, fiber_angle radians, bath cost, LBM CV ~+30–47%; +
  `test_cheatsheet_examples_execute` canary (execs a tagged runnable block).
- **P5 (optional):** analysis aggregates — `dominant_frequency_map`/`df_map`, `cv_between`, `radial_cv`,
  `apd_per_beat`, `restitution_slope` (+ result hooks + top-level exports) + DF-resolution warning +
  zero-node-stimulus warning.
Key facts verified empirically this session: `monodomain('ord')` RAISES (SR-release/CaMKII concentration path
unwired for classical splitting) but `lbm('ord')` runs; `paci`/`phas13`/`mhas13` run on monodomain; the FDM
`_harm` guards 0/0 (`s>0 else 0.0`) so a D=0 scar BLOCK is NaN-safe.

**API usability audit ROUND 2 (full-solve-and-run, +30 tasks) — 2026-07-16 → same report, "ROUND 2" section.**
10 agents, 30 new tasks (25–54) + full-scale re-run of the prior 24; agents had to actually SOLVE+RUN each to
completion. Running fully LOWERED the grade — it surfaced a class of defects a smoke test hides. **13 concrete
bugs (B1–B13)**, the load-bearing ones: **B1** GPU `device="cuda"` crashes ALL analysis/viz (`_result_from`
puts `times` on CPU, `Vm` on CUDA — one-line fix, mine); **B2** `linear_solver='fft'/'dct'` broken via factory
(`FFTSolver.__init__` missing args) → everything stuck on slow PCG = root of the runtime wall; **B3/B4**
`apd_at` peak-over-remaining + notch bugs → silently wrong multi-beat/low-repol APD; **B5** `Grid(N,1)` crash;
**B6** `forward_euler` silent blowup past dt-stability; **B7** `record=` silently ignores unknown keys; **B8**
masked nodes returned at 0.0 mV → 23% silent CV error on every scar/fibrosis study; **B9** dead `stim_amplitudes_e`
(no defibrillation). PERF: fixed per-step wall (~1.5–3 ms/step CPU, ~13 ms/step GPU, grid-independent) → long
protocols 5–11 min; escape = `forward_euler`+`none`+`dt`≈0.04 (undocumented). Verdict flips: T3 automaticity
No→Yes (undoc `paci`), T6 non-hole scar No→Possible (per-node D=0), T8 isthmus block Yes→No-via-geometry,
T15 5/5→broken-on-GPU. Reentry ACHIEVED (anchored CL=296 ms, figure-8 CL≈344 ms, ring min-circ≈λ=2.82 cm) with
the solver workarounds; blockers = runtime wall + no rotor-seeding/mid-run-state API (all `set_*`/`get_state`/
`scale_conductance`/`clamp_voltage`/`add_pacing` are NotImplementedError stubs). Report ends with a **MERGED
P0–P4 fix list** = the blueprint target. Contention caveat: box oversubscribed, absolute wall-times inflated ~2–4×.

**Task-based API USABILITY audit — 2026-07-15 → [API_USABILITY_AUDIT_2026-07-15.md](./API_USABILITY_AUDIT_2026-07-15.md).**
Agentic walkthrough: 24 realistic scientist tasks across 7 categories, 6 parallel agents, each WRITING +
RUNNING the minimal cardiac_core script and rating Possible?/Ease(1–5) empirically. **Verdict: "possible
but painful," mean ease ≈2.7/5**; strong at *expressing* a sim, weak at *parameterizing* + *measuring* it.
**2 tasks IMPOSSIBLE** via the public API (transmural cell-type gradient; clean single-cell automaticity);
the motivating category (ionic tuning/pharmacology) is the worst-served (1.5/5). Ranked themes: **(CRIT)**
the parameter/heterogeneity layer (`scale_conductance`/`set_parameter`/`set_conductivity`/`scale_conductivity`/
`clamp_voltage`) is `NotImplementedError` stubs WITH inviting worked-example docstrings → #1 hallucination
trap, blocks 7 tasks; **(HIGH)** no documented conductance knob (only working route = inject a model
instance, undocumented + inconsistent TTP06-vs-ORd); **(HIGH)** cheatsheet ERRORS (lists `ord` as a
monodomain model but ORd runs LBM-only; omits paci/hiPSC; save/load + dominant_frequency + phase_map absent;
`phase_singularities` mis-primed); **(HIGH)** analysis is single-point/x-axis-only (no cv-map/radial/
restitution, no DF map, no tip tracking); **(MED)** silent failures (nan CV on block, masked nodes returned
at 0.0 mV → counted "activated", zero-node stim = silent no-op); **(MED)** no 0-D single-cell mode; no
sweep/fitting helper; uniform-only conductivity + global-only cell type (MID crashes). Bright spots: viz
(5/5), the LBM d2q9 boundary errors (validates the same-day F2 change), stimulus expression, masks. Fix
priority: kill the stub trap → cheatsheet correctness pass → NaN-fill masked Vm → analysis aggregates →
0-D + sweep → per-node fields. NOTE: connects to the CODE_AUDIT #7/#12 "planned-not-shipped methods" note —
usability audit shows that's an active runtime trap, not a benign doc gap.

**API failure-mode sweep + F1/F2 hardening — 2026-07-15.** Ran a full public-API failure-mode check
(all ~40 `_LAZY` exports; both construction paths — declarative factory + file-format mesh; all 3
engines; analysis/io/geometry/viz; + degenerate-input and expected-raise probes; 103 checks). **Verdict:
the whole public surface is complete and working** — every documented call completes or raises its
documented error; all 4 contract guards and 6/7 degenerate inputs already degrade gracefully. Two real
gaps found and FIXED (commit `2938cf9`, main; 218 passed / 2 xfailed): **F1** — an empty run
(`t_end < save_every`) crashed the analysis hooks (`.apd()`/`.lat()`/`.cv()`) because `_collect`/
`_result_from` returned a rank-1 `(0,)` Vm; now they return rank-3 `(0,Nx,Ny)` and `activation_time`/
`apd_map` guard the zero-length time axis → NaN maps / NaN, no crash. **F2** — `hbb` reclassified as
**D2Q9-only** (joins ncs/scs/combined in `wall_modes.D2Q9_ONLY`; `hbb`+d2q5 now raises instead of
silently acting as a neumann no-op), and the LBM boundary **default is now lattice-aware**: `neumann`
on d2q5 (UNCHANGED — tuner calls `run_lbm` with no lattice/boundary, and goldens pin d2q5, so both are
untouched), `hbb` on d2q9 (label-only; neumann≡hbb numerically on d2q9). Cheatsheet §4 documents the
d2q9 requirement (the prior doc gap). **User decisions (2026-07-15):** keep the global d2q5/neumann
default (do NOT flip to d2q9); merge only the 12 committed tuner commits to main. Two cosmetic nits
ALSO FIXED (commit `e707fe1`): **F3** `point_distance` now takes `center=(x,y)` matching
`circle_mask`/`annulus_mask` (was scalar `x0,y0`; only 2 internal test callers); **F4** cheatsheet §2
now notes `ConductivityConfig.sigma_eff`/`D_eff` return a scalar (iso/bi) but a 3-tuple `(xx,yy,xy)`
(aniso). Also landed the `engine-tuner-v2-joint` branch (12 commits) onto main (merge
`9d82f56`; one MASTER_KNOWLEDGE_INDEX.md conflict resolved keeping both the β-dt-guide bullet and the
SCS-gate-decontamination correction).

**Deep code audit — math integrity + API — 2026-07-02.** After pushing the API-consistency work to
`main`, ran a 6-lane agentic walkthrough of ALL of cardiac_core with per-lane NUMERICAL verification →
[CODE_AUDIT_2026-07-02.md](./CODE_AUDIT_2026-07-02.md). **0 blockers / 4 majors / ~22 minors.** Default
paths + all Phase-0–5 hardening + the 4 ionic models verified SOUND (cross-derivative fix confirmed;
time-stepper orders exact CN=2.00/RK4=4.01; LBM Chapman–Enskog + `mrt_wall` + masked-bounce correct; CV
matches ref 54.14/54.35; D_eff=9.72e-4 + Cm-trap correct). The 4 majors: **mono Chebyshev** (Gershgorin
bounds on raw A not preconditioned D⁻¹A → 94 mV wrong at a stiff config, opt-in solver; bidomain already
has this fix), **mono FFT** (continuum −k² vs discrete 5-pt eigenvalue, opt-in), **bidomain `step()`**
AttributeError (pre-existing, untested public method), and — most serious — **bidomain spatially-varying
anisotropy** breaks elliptic-operator symmetry → CG-family solvers silently return **~13% wrong phi_e**
(per-node fiber fields; the uniform-angle public API is SAFE). M1+M3 orchestrator-reproduced; M2/M4 on the
auditors' numerical evidence. **Audit only — NO source changed**; findings triaged for later (fix priority
P1–P4 in the doc). Commit `619460c`. Recurring weak spot = the **FDM cross-derivative** (M4 is the same
family as the deferred C7 / boundary anisotropy).

**API-consistency hardening + contract-matrix harness SHIPPED (2026-07-01).** The two post-ship boundary
gaps turned out to be a *class* of API-surface fragility. Ran a 4-lens adversarial audit
([API_CONSISTENCY_AUDIT.md](./API_CONSISTENCY_AUDIT.md): 7 HIGH/8 MED/6 LOW), **audited the fix PLAN to
convergence over 4 rounds** (R1 5blk/10maj → R2 1blk/5maj → R3 1blk/1maj → R4 1blk-mechanical →
CONVERGED — each round narrower: code-bugs → coverage → mechanism text → a namedtuple field order), then
executed 6 phases on `engine-tuner-cardiac-core` (`1a65d3d`→`9702bb7`). Keystone =
`tests/test_api_contract.py`, the contract matrix **written FIRST** with `xfail(strict=True)` forcing each
fix's in-phase flip (the post-mortem's process cure, mechanized). **217 passed / 2 xfailed** (C2 oblique
capability + C7, documented-deferred; the audit caught that oblique LBM is real numerics/Audit #46, not
wiring — would've shipped silent-wrong). Goldens bit-identical every phase. See KNOWLEDGE "API-consistency
hardening". **This also RESOLVES** the prior Next-Step item "mono `boundary_mode`/`stencil` not surfaced"
(C6, now exposed) and the ionic-registry parity #6/paci-alias #12 (C3/C8). **Remaining:** Form-A→B
monodomain convergence; oblique-LBM moment-space rotation (Audit #46) if ever wanted.

**Foundation cleanup + boundary modes SHIPPED (2026-07-01).** A cardiac_core+cardiac_mcp adversarial audit (46 findings → [CARDIAC_CORE_AUDIT.md](./CARDIAC_CORE_AUDIT.md); 8 lanes, find→refute→completeness) drove a 3-phase, audit-to-convergence-hardened cleanup ([PLAN.md](./PLAN.md); R1 5blk/6maj → R2 2/2 → R3 SOUND): **P1** fixed a real mono FDM anisotropic cross-derivative BUG (wrong sign + half magnitude; `V=x·y` now `+2·Dxy`) + unified the chi/D convention (`D_xx` RAW everywhere, effective = `D/(χ·Cm)` in every engine, default `D=1.4`, blocked-default fixed) + ionic-override replay + MCP path-traversal; **P2** removed FEM/TriangularMesh + dead code + API footguns; **P3** productized the LBM flat-wall boundary modes as `cardiac_core.lbm(boundary=, alpha=)` — hbb / `specular_nextcell` (NCS) / `specular_samecell` (SCS) / `combined`-α (the β curvature knob), 'ncs'/'scs' aliases (default `neumann` bit-identical → goldens safe). Suite 148/1 → **196 passed / 0 failed**; 5 commits pushed to `origin/engine-tuner-cardiac-core`. **RESOLVES** the boundary-mode API→engine gap (for LBM), the `create_cardiac_mesh` chi firewall-bypass (cluster #1), and FEM removal.

**cardiac_mcp MCP server — STANDARDIZED + SHIPPED to `main` (2026-06-28→30).** The `cardiac_mcp/` server (built 2026-06-26) was audited against the OFFICIAL MCP spec **2025-11-25** (4 spec-research agents, verified vs live spec + installed SDK source) → a 4-tier PLAN → hardened through **3 adversarial audit rounds (12→5→0, CONVERGENCE CLEAR)** → executed **Tiers 1–3** (T1 honest annotations + `serverInfo.version` + MIME + two path-traversal guards; T2 typed `outputSchema`/`structuredContent` + 2 prompts + README + Option-B installable `cardiac-mcp` console script; T3 provenance + CPU/FSIZE-limited `run_experiment` + localhost HTTP transport + `REMOTE_DEPLOY.md`). **16 cardiac_mcp + 140 cardiac_core tests green; merged --no-ff → `main`, pushed (`41d17f4`).** Phase 4 (registry publish) SKIPPED. Detail: KNOWLEDGE "Goal-2 MCP server — standardization audit" + the 2026-06-28 Session Log + PLAN.md Mutation Log.

**Goal-2 MCP server — `cardiac-core` — SHIPPED local (2026-06-26).** Built `cardiac_mcp/` — an MCP (Model Context Protocol) server exposing `cardiac_core` to ANY MCP host (Claude Desktop/Code/IDE), breaking the skills' Claude-Code-terminal-only ceiling — the real reach step for the wet-lab audience. **Two-track tool surface** (user decision): a DIRECT `simulate()` (ephemeral CV, no record, coarse-dx fast ~8s) + the GATED `build_manifest`→`commit_experiment`→`run_experiment` chain that ports the `/sim-experiment` accountability gate STRUCTURALLY — `build_manifest` returns a self-signed `experiment_token` embedding the exact manifest+params; `commit_experiment` refuses unless that token verifies AND `confirmed=True`, so the committed `Lab/{date}_{slug}/` script is provably what the scientist reviewed. **Local stdio now, designed for remote-HTTP later** (user decision): all logic in transport-agnostic `cardiac_mcp/core.py`, `server.py` only wires FastMCP, `__main__` picks transport (HTTP = one-line swap). Registered via `.mcp.json`; `mcp` SDK (1.28.0) installed in the env. Validated: 10 core tests + server boot (5 tools/2 resources) + real stdio client↔server roundtrip. See KNOWLEDGE "Goal-2 MCP server". **Next:** activate in Claude Code (approve the project server), then optionally add resources/prompts (presets, glossary), media tool, and the HTTP transport for remote scientists.

**Goal 2 — the LLM layer — SHIPPED (2026-06-25).** The script-generating skill suite for wet-lab scientists is built + committed (`/sim-experiment` keystone + `/sim-preset` + `/sim-media` + `/sim-notebook` + `cardiac_core/API_CHEATSHEET.md` + `cardiac_core/viz.py`); 140 tests green; validated end-to-end (control/knockdown CV series). See KNOWLEDGE "Goal-2 LLM layer — SHIPPED". **Both north-star goals now delivered.** Remaining options: Layer-A `SimulationSpec`; programmatic claude-api; Form-A→B convergence; FEM removal. Keystone `/sim-experiment` to be `/audit`'d (the double-check gate).

**Consolidation (A2 vendoring) SHIPPED (2026-06-25).** `cardiac_core` is one self-contained package — 3 engines vendored `_monodomain`/`_bidomain`/`_lbm` + shared `ionic`/`mesh`/`stimulus`, `_prepare_engine()` hack deleted, bit-identical goldens, originals frozen. Phases 0–5 `935160b`→`37dc381`. → KNOWLEDGE "cardiac_core unified ground-up package". *(Predecessors condensed — full detail in Thread: V5.5 Cm-correct fork + consolidation Phase-1 copy-only, 2026-05-30/31; the Goal-2 design reframe to wet-lab code-gen, 2026-06-25; the "ditch FEM → structured-grid only" pending constraint — RESOLVED by the 2026-07-01 FEM/TriangularMesh removal. The deferred code-dedup [engines import from cardiac_core + delete copies] stays per-consumer — big-bang breaks Surrogate/Optimizer.)*

## Next Step
**▶ MERGE `usability-fixes-p0-p1` → `main`** (commits `a37d325`→`c0306d2`: 5 phases + `562a7a0`/`257b2c1` docs + `c0306d2`/`9f387ef` audit-remediation; PLAN EXECUTED + 5-lane final audit clean, see Current Direction). Suite 260/2xfail,
goldens bit-identical. The branch is review-ready; nothing else in the plan is outstanding. After the merge, the OPEN threads below are unchanged (none were touched this session). Two
usability items that were NOT in this PLAN and remain P3/future work: **B9** dead `stim_amplitudes_e` (no
defibrillation) and the rotor-seeding / mid-run-state API (`set_voltage`/`get_state`/`clamp_voltage`/`add_pacing`
are now HONEST stubs, not implemented) + a 0-D single-cell mode + `cc.sweep`/`fit_conductivity` (all documented in
PLAN.md "Future work").

The A2 unification, Goal-2 skill suite, cardiac_mcp server, AND the 2026-07-01 foundation cleanup + LBM boundary modes are all SHIPPED (see Current Direction). RESOLVED this session: `create_cardiac_mesh` chi firewall-bypass (P1 cluster #1 — D_xx RAW convention + default D=1.4 + band guard); boundary-mode API→engine gap for LBM (P3 — `lbm(boundary=, alpha=)`); FEM/TriangularMesh removal (P2). Remaining open threads:
- **Code-audit fix backlog (2026-07-02, [CODE_AUDIT_2026-07-02.md](./CODE_AUDIT_2026-07-02.md), NOT yet actioned):** P1 = bidomain M4 (symmetrize the FDM cross-term + guard CG on non-SPD — silent ~13% phi_e error on per-node fiber fields); P2 = mono Chebyshev M1 (port bidomain's preconditioned-Gershgorin + fix `set_eigenvalue_bounds`) + mono FFT M2 (discrete 5-pt eigenvalue) + DCT/FFT precondition guard; P3 = bidomain `step()` M3, degenerate-input NaN guards (`activation_time`/`dominant_frequency`/`Grid` 1-D/empty-result shape), BGK stability gate, `dt or`→`is not None`, declarative ionic-instance-leak; P4 = docstrings + `add_stimulus` amplitude + hole-cell zeroing + LUT kink + conductivity guards + PCG-threshold unify + LBM `save_every` cadence.
- **Form-A→B convergence** (convert monodomain diffusion in `_monodomain`, delete `ConductivityConfig.for_monodomain()`) — confirmed-but-deferred.
- **Deferred audit backlog** (in PLAN.md "Findings Coverage"): Research/ doc reconciliation (#9/#10/#11/#24/#25); wiring the orphaned LBM `dirichlet`/`absorbing` as selectable modes (#37, needs a bc-value); ionic-registry parity (#6) + the `paci→PHAS13` alias check (#12); the FDM boundary-Dxy Neumann (#14) + bidomain bath≠insulated assert (#39); the 12 completeness gaps (analysis.py, io round-trip, LBM masked grids, device/dtype, second mesh loader).
- ~~**Monodomain/Bidomain boundary-mode API exposure**~~ **DONE (2026-07-01, Phase-4/C6):** mono `stencil`/`boundary_mode` + bidomain `stencil` now surfaced by the factories. (A fully-unified cross-engine boundary *concept* using bidomain `BoundarySpec` as template is still a future nicety, not blocking.)
- **Surrogate/Optimizer ionic migration** off engine-local `cardiac_sim.ionic` (per-consumer, test-gated; never delete out from under a live consumer).
- **MCP follow-ups** (optional): media tool wrapping `cardiac_core.viz`; more resources/prompts; reentry/restitution recipes; the remote-HTTP auth stack (`REMOTE_DEPLOY.md`) + Phase-4 registry publish when wanted.
- Deferred Goal-1 Layer-A `SimulationSpec`; programmatic claude-api; `API_REFERENCE.md` `[design]`→`[now]` tags.

--- prior (consolidation track, still valid) ---
**cardiac_core drift RECONCILED (2026-05-30):** the post-Phase-0 additions (`run.py`/`analysis.py`/`geometry.py`/`io.py`) are a benign wrapper-level convenience layer (77 tests now, not 34); no shared-code packages yet, so Phase 1 is unblocked. `Engines/` symlink index fixed (cardiac_core un-broken; lbm_v1 → real `LBM/Engine_V1`; monodomain_v5.5 added). See KNOWLEDGE "cardiac_core drift reconciled".

**Phase 1 (copy) DONE (2026-05-31):** `cardiac_core/ionic/` is the canonical superset copy (from V5.5; latent LUT keyword `cell_type_is_endo`→`celltype_is_endo` fixed); `cardiac_core/__init__` made lazy (PEP 562 — `import cardiac_core.ionic` is engine-free, no `_prepare_engine`); `pyproject.toml` + `pip install -e .` make cardiac_core a real importable package (cwd-independent, scoped to `cardiac_core*` — does NOT expose Builder/cardiac_ml/engines). 77 cardiac_core tests green; V5.5 golden still exact (engines untouched).

**Scope pivot (post-audit, 2026-05-30):** the engine rewire+delete was DROPPED to copy-only after the audit found big-bang deletion breaks engine tests/examples AND active cross-project consumers (`Surrogate/surrogate/data/*_generator.py`, `Optimizer/V1/tuner/tissue_runner_bidomain.py` import `cardiac_sim.ionic` via the Bidomain path). User: "don't delete the originals — just copy them over."

**Next Step:** the DEFERRED migration (PLAN.md "Deferred" section) — when resumed, migrate consumers REPO-WIDE (engines' tests/examples + Surrogate datagen + Optimizer + `cv_shared` bare `from ionic`) to `cardiac_core.ionic`, per-consumer with test gates, never deleting out from under a live consumer; exclude V5.3/V5.4/_archive/torchcor from any survivor check. cardiac_core is now editable-installed (engines/consumers gain `import cardiac_core` for free once rewired).

## Thread

### 2026-07-16 (audit round 2): audit-the-fix + completeness critic → 2 more real bugs + 2 masked-data gaps
Ran a SECOND round (4 lanes: gate-correctness, analysis-fixes, P3-validation, and a whole-branch completeness
critic) — the discipline that a fix round which changed code must itself be audited. Fixed (commit `9f387ef`):
- **hiPSC regression I introduced in round 1.** The round-1 conductance allow-list keyed on UPPERCASE `G*/P*`,
  but paci/phas13/mhas13 name conductances LOWERCASE `g_*` → scale_conductance rejected ALL their conductances
  (and regressed `g_Na` scaling that worked pre-fix). Now case-insensitive first-letter + explicit denylist for the
  two dimensionless params that merely start with g/p (`gamma_ncx` NCX-partition, `PkNa` IKs Nernst ratio). Verified
  by enumerating g/G/p/P params across ALL 5 models.
- **Bidomain masked default crash** (completeness critic — the cross-engine symmetry both per-file lanes missed):
  `elliptic_solver='auto'` picked spectral on a hole → cryptic `shape [15,15] invalid for size 216`. Monodomain got
  the DCT gate in round 1; bidomain's parallel path didn't. Auto now falls back to `pcg` on a masked domain (golden-
  safe — full-rect unaffected).
- **P5 silent-wrong on masked/NaN data** (all P5 tests had used clean synthetic tensors): `dominant_frequency_map`
  returned a phantom low freq at NaN holes → now NaN; `radial_cv` silently all-NaN on a dead center → now warns.
- Lesser: `restitution_slope` → LAST descending crossing (alternans boundary on noisy curves); `_rebuild_with_
  conductivity` transactional; cheatsheet count/LBM-record/masked notes.
- **Round-2 VERIFIED SOUND:** the DCT gate is complete (no silent-wrong slips through; allowed set is an exact
  subset of the match-set, residual 7e-15); both round-1 analysis fixes; flat_to_grid guard (all dtype branches);
  LBM guard; the two Phase-3 cross-engine fixes. Accepted/left: B1 CPU-coverage (GPU-only bug, GPU test covers it);
  apd_per_beat/apd_at shared `V_rest=trace[0]`; bidomain scar φ_e pcg_spectral-vs-pcg ~533 mV (PRE-EXISTING elliptic
  accuracy = the deferred 2026-07-02 code-audit M4; Vm/cv/apd unaffected, only φ_e/ECG).
**Convergence signal:** round 2's findings were 1 self-inflicted regression + cross-cutting completeness gaps (not
a new bug class in the core logic); the completeness critic called the per-lane work "genuinely thorough." Suite
**260 passed / 2 xfailed**; goldens bit-identical.

### 2026-07-16 (final audit): 5-lane adversarial audit of the whole branch → 4 more real bugs fixed
After executing all 5 phases, ran a **5-lane parallel adversarial audit** (general-purpose subagents; A=P1,
B=P2, C=P3, D=P5, E=tests+docs) over the whole `main..HEAD` diff. Verdict: default paths sound, but **reachable
non-default configs were silently wrong**. Fixed (commit `c0306d2`):
- **B2 was the worst — my own regression.** Wiring dct/fft removed the TypeError but let users SELECT them in
  configs where they're silently wrong: DCT/FFT ignore the assembled matrix and invert an idealized scalar-D
  Neumann eigen-operator. Measured: anisotropic D → up to **68% CV error**; scar D=0 → **invisible**; bdf2 →
  **CV=nan** (BDF2's BDF1-bootstrap step gets the BDF2 denom → DC/3); fft on any (Neumann) mono mesh → **CV=nan**.
  Fix: `_check_spectral_preconditions` gates dct/fft to iso-uniform + full-rect + face_mirror + cardinal4 + CN/BDF1,
  rejects fft entirely; FDM exposes `_is_iso_uniform`. GOTCHA: the mono factory ALWAYS builds `from_mask` (full-rect
  gets an all-True mask), so the "masked" check must be `not domain_mask.all()`, not `is not None` (first gate cut
  falsely rejected the valid uniform dct path). Default `pcg` untouched; dct on the valid path is exact (Vm≈CN 1e-3).
- **restitution_slope.DI_star** returned the steep short-DI end, not the slope=1 crossing → now interpolates the
  descending crossing. **apd_per_beat** emitted 0.0 for a beat in progress at t=0 → now only measures clean-upstroke
  beats. **_scale_ionic_conductances** accepted any attr (F/T/Cm/concentrations/`*_scale`) → restricted to G*/P*
  non-`*_scale`. **flat_to_grid** NaN-fill guarded for int/bool flats. **LBM** regional set/scale_conductivity now
  raises a clear error (was a misleading "oblique fibers"). **Cheatsheet** `load_result` fixed to the real 4-tuple.
- **Verified SOUND by the audit:** B1 device fix (empty-branch consistent, batch path covered), B3/B4 apd_at exact
  reduction to old behavior on monotonic APs, B5 all four degenerate cases, B6 CFL formula/attrs/non-FDM-skip, B7
  eager both entry points; the two Phase-3 cross-engine fixes (sigma no-op, cell-type flip); every cheatsheet claim
  except load_result (ORd-LBM-only, PCa=ICaL, paci-on-mono, save_result arg order all TRUE).
- **Accepted/left (minor):** B1 CPU test is tautological (real coverage = the GPU-gated test, runs on this box; the
  bug is GPU-only anyway); df_map flat-node→0 Hz (consistent w/ per-node fn); radial_cv all-NaN on a bad center;
  transposed square-mask silent-accept. Documented, not fixed.
Suite after remediation: **254 passed / 2 xfailed**; goldens bit-identical.

### 2026-07-16 (exec): PLAN.md usability fixes EXECUTED — 5 phases, audit-hardened
Cold-started from PLAN.md and ran all 5 phases on branch `usability-fixes-p0-p1`, each implement → targeted test
→ goldens (bit-identical, atol=0) → commit. Details in Current Direction. **Execution catches beyond the plan:**
- **Bidomain masked runs can't use the default spectral elliptic solver** (`SpectralSolver.solve` reshapes to the
  full `nx*ny`; a hole makes `n_dof < nx*ny` → RuntimeError). The B8 bidomain masked test uses
  `elliptic_solver='pcg'`. Documented in the cheatsheet.
- **The Phase-3 adversarial audit was the load-bearing step.** My scale_conductance/set_conductivity tests were
  monodomain-only and passed, but the audit (a general-purpose subagent, since the Opus `/audit` path was rate-
  limited) found 2 REAL cross-engine bugs on the paths I hadn't tested: the declarative-bidomain sigma no-op and
  the cell-type flip. Both fixed + regression-tested (declarative bidomain scar, cell-type-preservation, sigma
  scaling). Lesson (again): a green mono test says nothing about the bidomain/LBM surface — test at the level a
  USER reaches the feature (this is the SAME class as the 2026-07-01 Phase-3 boundary-gap post-mortem).
- **B2 DCT is exact, not approximate:** the DCT solver's denom (`chi*Cm - 0.5*dt*D*λ`) matches the FDM CN operator
  when `D=_D_max` (raw) — so CV matches PCG to tolerance, not just "close". Confirmed the FDM's raw-D convention
  flows correctly into the spectral solve.
- **apd_at B3/B4 fix is regression-safe:** existing `test_analysis.py` synthetic APs are monotonic/dome-free, so
  beat-windowing + dome-aware last-crossing reduce to the old first-crossing there (verified: values unchanged).
**Next:** merge to main (see Next Step). Optional: an independent Opus `/audit` of the whole branch once the rate
limit clears (this session's Phase-3 audit used a general-purpose subagent).

### 2026-07-02 Session — deep code audit (math integrity + API)
**Worked on:** after pushing the API-consistency work to `main` (fast-forward, `abc54db`→`94a2689`), the
user asked for further rounds of audit over ALL of cardiac_core — math integrity per engine, then the API
— as an agentic walkthrough.
**Accomplished:** 6 parallel deep auditors (mono math · bidomain math · LBM math · ionic+conductivity ·
API factories · run/io/analysis/mesh), each verifying NUMERICALLY (not trusting the suite). Verdict **0
blockers / 4 majors / ~22 minors**, written up in [CODE_AUDIT_2026-07-02.md](./CODE_AUDIT_2026-07-02.md)
(committed `619460c`). Orchestrator independently reproduced M1 (Chebyshev 94 mV wrong at stiff effD=0.5/
dt=0.05, benign at physiological) and M3 (bidomain `step()` AttributeError). Verified SOUND: default-path
math on all 3 engines, the prior cross-derivative fix (2·Dxy exact), time-stepper orders, reaction Cm,
LBM Chapman–Enskog + the new `mrt_wall`/masked-bounce, all 4 ionic models (stable rest + physiological AP,
correct APD ordering), the conductivity firewall, analysis math, AND every Phase-0–5 change. 4 majors:
mono Chebyshev (opt-in, wrong Gershgorin operator), mono FFT (opt-in, continuum vs discrete eigenvalue),
bidomain `step()` (pre-existing untested), bidomain per-node-anisotropy non-symmetric elliptic (silent
~13% phi_e). **No source modified** — findings triaged only.
**Next:** if the user wants, action the P1–P4 fix backlog (Next Step above), starting with M4 (symmetrize
the bidomain FDM cross-term). Otherwise the audit stands as a recorded triage. Could also do further audit
rounds (e.g. adversarial re-verification of M2/M4, or a 3D/thickness-path sweep).

### 2026-07-01 (exec): API-consistency hardening EXECUTED — 6 phases, contract-first harness, 217/2
Executed the audit-converged PLAN phase-by-phase, each test+golden-gated and committed. **Contract-first
worked exactly as designed:** Phase 0 wrote all 22 cells as `xfail(strict=True)` (green — expected-fails);
each phase's fix then made its cells XPASS → strict-fail → *forced* me to flip `to_fix→landed` in that same
commit. No cell could be silently left behind. Ended 20 landed + 2 deferred (C2-capability, C7).
**Execution catches (beyond the plan):**
- The round-2 ENDO-vs-EPI warning paid off: the shared ionic builder MUST default `cell_type='ENDO'`
  (verified ORd/TTP06 ctors default ENDO; mono derives from the mesh) — an EPI default would've flipped the
  bidomain+LBM goldens. Shipped ENDO; goldens held every phase.
- `mrt_collide_d2q9`'s `w` is 5th (right after `dt`); `lbm_step_d2q9_mrt_wall` mirrors it exactly (the
  round-2 signature fix was right).
- `_lbm_bounce_masks` had to UNION hole rim ∪ outer rect edges because `precompute_bounce_masks` uses
  periodic `torch.roll` (the B2 finding) — wired to BOTH LBM construction branches.
- Minor self-inflicted: passed a stencil value (`moore8_iso`) as a `boundary_mode` in one named test —
  distinct vocabularies; caught on first run, fixed.
Commits `1a65d3d`(P0)→`40cd2ca`(P1)→`1dda8f6`(P2)→`35327f5`(P3)→`9702bb7`(P4); PLAN archived to
`plans/2026-07-01_api-consistency-hardening-stress-harness.md`; audit docs marked RESOLVED.

### 2026-07-01 (Phase-3 boundary gaps found post-ship): two real gaps + a testing-failure post-mortem → [PHASE3_BOUNDARY_GAPS.md](./PHASE3_BOUNDARY_GAPS.md)
User caught two gaps in the shipped LBM boundary feature (commit `736296d`) that my 17 green tests missed:
- **Gap A — `run_lbm()` doesn't forward `boundary`/`alpha`** (still HBB-only). The params reached only the declarative `lbm()` factory + `simulate(**kwargs)`; the explicit one-shot `run.py::run_lbm` forwards only ionic/dt/lattice/device. Exactly the one-shot-vs-factory asymmetry the audit flagged as #18/#19 (deferred) and re-opened.
- **Gap B — wall modes are BGK-only; anisotropy is blocked.** `lbm(aniso, boundary='ncs')` raises, because anisotropic D forces MRT and a guard requires `collision='bgk'`. But the overlay is POST-STREAM → collision-agnostic; the guard was over-conservative and `test_lbm_rejects_oblique_Dxy` *codified* the over-restriction (conflating oblique `D_xy≠0` (real limit) with per-axis anisotropy (should work)).
**Root cause (post-mortem):** tested/audited the feature at the level it was BUILT (`lbm()` factory, isotropic BGK), not at the level a USER reaches it (`run_lbm`/`simulate`; anisotropic fibers). The contract (entry points × physics) was never written down, so nothing checked against it — and the 3-round plan-audit couldn't catch NEW-feature under-specification (it checks the plan vs the 46 existing findings, not surface-coverage of a not-yet-written feature). **Fix approach (settled, in the gaps doc):** (A) add `boundary`/`alpha` to `run_lbm`; (B) add `lbm_step_d2q9_mrt_wall`, remove the bgk-only guard, dispatch mrt+special→mrt_wall, keep the oblique-`D_xy≠0` rejection. Tests must span `{lbm, run_lbm, simulate} × {isotropic-BGK, per-axis-aniso-MRT} × modes`. → blueprint the fix.

### 2026-06-30 (system audit + cleanup decisions): adversarial audit of cardiac_core + cardiac_mcp + docs → [CARDIAC_CORE_AUDIT.md](./CARDIAC_CORE_AUDIT.md)
Ran an 8-lane adversarial multi-agent audit (find → refute → completeness; 64 agents, ~3.1M tokens) over the shipped surface, to tidy the foundation BEFORE building the boundary-mode work (which the PI has now adopted as REAL, β/discreteness-driven, unreported physics — stop treating the α-blend/specular speedup as artifact). **55 raised → 46 confirmed** (4 HIGH, 11 MED, 27 LOW, 4 INFO), 9 refuted, 12 completeness gaps. Full report: `CARDIAC_CORE_AUDIT.md`.
**Standout:** a real numerics bug (#4, HIGH) — the monodomain FDM anisotropic cross-derivative has the WRONG SIGN and HALF MAGNITUDE vs the validated bidomain builder (`cxy=1/(4dxdy)` should be `1/(2dxdy)`; diagonal signs flipped) → oblique-fiber runs quantitatively wrong, untested. Directly on the anisotropic-boundary path; my own boundary audit missed it.
**Systemic cluster:** the `CardiacMeshData` chi/D ambiguity (#2/#8/#21) — same mesh runs at ~1400× different D across engines; default `create_cardiac_mesh(D=0.001, chi=1400)` is conduction-blocked. The firewall has a hole at the file-format layer.
**DECISION (user, 2026-06-30): `CardiacMeshData.D_xx` = RAW conductivity-like; `chi·Cm` divides in EVERY engine → physical effective D = `D_xx/(χ·Cm)` uniformly.** Monodomain FDM already does this (χ·Cm in mass term = the "correct" one); **the LBM factory (api.py:1375-1395) currently passes `D_xx` straight through as effective and MUST be changed to divide by χ·Cm first**; bidomain already divides. **Corollary:** the `create_cardiac_mesh` DEFAULTS must also change (raw-σ-scale default D, or default `chi=1.0`, or a guard) — "raw everywhere" makes the convention consistent but leaves the blocked default. **DECISION (user): blueprint the whole cleanup first** (blockers → tidy → boundary build), no code until the PLAN is reviewed. Sequence: fix #4 + cluster-#1 + #1 (ionic-override lost on reset) + #3 (MCP path traversal) BEFORE the boundary-mode build; fold #29/#37/#38/#14/#39/#40 INTO the boundary build (they are its surface); batch the dead-code/docs tidy; the 12 completeness gaps are the next audit round.

### 2026-06-30 (API-debt finding): `create_cardiac_mesh` BYPASSES the Formulation-A/B firewall → silent ~1400× D mis-scale
Surfaced while implementing the `ionic_model_optimization` ↔ chip-fit cross-plan (tuning cardiac_core to Kit Parker chip EP). **Symptom:** building a mesh with `create_cardiac_mesh(D=<effective diffusivity ~1e-3>)` at the **default `chi=1400`** gave NO propagation on every engine config — the stimulus pooled the source nodes to a non-physical **Vmax ≈ 80–123 mV** with zero downstream activation (CV=NaN). Isotropic *and* anisotropic; not the anisotropy/MRT work — a mesh-assembly issue. `chi=1.0` → clean propagation, CV=59 cm/s.
**Mechanism (verified in `_monodomain/.../fdm.py:37,159`):** the FDM operator solves `χ·Cm·∂V/∂t = ∇·(D·∇V)` — the stiffness Laplacian is built from `D` alone, `χ·Cm` sits only in the mass/time term, so the **membrane-effective diffusivity is `D/(χ·Cm)`**. An effective `D≈1e-3` with `chi=1400` → effective ≈7e-7 (1400× too low) → CV ∝ √D drops ~37× → below the discrete source–sink launch threshold at chip `dx=0.01` (the space constant shrinks below `dx`) → conduction block. This is *faithful physics of the wrong number*, not a solver bug: `(D=1e-3, χ=1400)` is exactly degenerate with `(D=7.14e-7, χ=1)`.
**The API debt (this is the report):** we built the **Formulation-A/B firewall** (`conductivity.py`, ConductivityConfig) precisely to prevent σ-vs-D_eff confusion (see 2026-03-16 / 2026-06-24 entries) — but `create_cardiac_mesh` is a **second, unguarded entry point** that re-exposes raw `D` + `chi`, and its default (`chi=1400`) contradicts its own docstring ("D : diffusion coefficient cm²/ms"). So the trap the firewall closed is reopened by the convenience builder.
**Recommended fix (pick one):** (a) docstring warning — DONE (`file_format.py` + `ionic_model_optimization/KNOWLEDGE.md`); (b) default `chi=1.0` + treat `D` as a true diffusivity — *breaks* the ConductivityConfig/declarative callers that pass conductivity-style with `chi=1400`; (c) add a non-breaking `mode={'diffusivity'|'conductivity'}` flag that validates the `D`/`chi` pairing; (d) route `create_cardiac_mesh` through `ConductivityConfig` so there is ONE firewall. **Recommend (c) or (d).** Mitigated in the chip pipeline already (`cc_runner`/`chip.chip_mesh` pin `chi=1.0`). LBM unaffected (maps D→τ directly, no χ division).
**Empirical (2026-06-30, TTP06, dx in cm):** (1) **Degeneracy CONFIRMED** — `(D=1e-3, χ=1400)` and `(D=7.14e-7, χ=1)` are bit-identical (Vmax 81.6, CV=None, both blocked at dx=0.01). χ is a pure rescaling in monodomain. (2) **The block is DISCRETIZATION, not "merely slow"** — the same effD=7e-7 at 4× finer dx=0.0025 *does* propagate (CV 0.72 cm/s); at dx=0.01 the ~37× thinner front is unresolved → block. (3) **The REAL chip regime is FINE** — effD=2.5e-5 (NRVM target) propagates cleanly at dx=0.01, CV≈6.06 cm/s (~36× above the block threshold); no artificial block. The tuner's secant fit dials D to the exact target (9.33). Minor caveat: at chip-slow CV the stim-site upstroke runs a bit hot (Vmax~81 vs ~50) at dx=0.01 — finer dx / the points-per-λ guard cleans it up if needed.

### 2026-06-28 (MCP audit + blueprint): standardized vs spec 2025-11-25 → 4-tier PLAN
After shipping `cardiac_mcp` (2026-06-26), the user flagged that a working server isn't a *standardized* one — they intuited "a list of supporting documents" an MCP server needs and wanted an audit against proper guidelines (not my from-memory build). Ran **4 parallel spec-research agents** against the OFFICIAL spec (modelcontextprotocol.io + schema.ts), confirmed current revision **2025-11-25**. Key learnings: (1) **Tool annotations** are the headline gap — unset → spec defaults make ALL our tools advertise `destructiveHint=true/openWorldHint=true`, dishonest for the read-only ones and under-flagging the code-runner; set `readOnlyHint`/`destructiveHint=false`/`openWorldHint=false` per tool. (2) `serverInfo.version` falls back to the SDK version (1.28.0) unless set. (3) **outputSchema/structuredContent** SHOULD back structured dict returns (type the returns). (4) Errors: recoverable → `isError` tool-result (FastMCP already does this for raised exceptions); reserve JSON-RPC errors for protocol faults. (5) **Two path-traversal bugs** (MUST validate inputs): `run_experiment` `(REPO_ROOT/experiment_dir).resolve()` escapes on absolute/`..`; `commit_experiment` uses unsanitized `date` in the folder name. (6) Distribution layer = `server.json` (reverse-DNS name, immutable semver), README + ownership marker, `pyproject` console-script, LICENSE/Dockerfile — REQUIRED only to publish to the registry. (7) **stdio→HTTP is a big delta**: OAuth 2.1+PKCE(S256), RFC 9728 PRM, RFC 8707 resource indicators, `Origin`→403, secure session IDs (never auth-via-session), SSRF defenses, no token passthrough; AND the spec SHOULD-sandboxes a code-executing tool (our `run_experiment` remote-without-sandbox = RCE-as-a-service). All audited features (annotations/outputSchema/mime) are available in the installed SDK 1.28.0 (they predate 2025-11-25). Synthesized into a 4-tier remediation order (T1 now → T4 optional) → `/blueprint` → PLAN.md.

### 2026-06-25 (Goal 2 design): skill suite for wet-lab scientists — settled, ready to blueprint
**Audience REFRAMED** (corrected the README "non-coder conversational builder" wording twice): target = WET-LAB scientists (cell culture, tissue-on-chip / lab-chip) WITHOUT computational-simulation exposure. Goal 2 = a SKILL SUITE that **lowers coding complexity by GENERATING runnable `cardiac_core` scripts** (code-gen, not an interactive wizard; no auto-teaching). Drives the shipped `cardiac_core` API directly — Layer-A `SimulationSpec`/`create_simulation` DEFERRED; programmatic claude-api comes later.

**The suite (user's order):**
1. `/sim-experiment` — free-form description → runnable `cardiac_core` script. (KEYSTONE)
2. `/sim-preset` — save / store / reuse named parameter sets.
3. `/sim-media` — standardized figures & videos (canonical `media/`).
4. `/sim-notebook` — lab-notebook organization (master log + per-experiment folders).

**`/sim-experiment` protocol (settled with user):**
- RECEIVE free-form input (any shape — a sentence, a paragraph, a chip protocol).
- INTERPRET → build `cardiac_core` params (infer engine + map to API); ask ONLY for genuine gaps.
- MANIFEST → present a plain-**TEXT** summary of ALL params: goal, engine (+why), ionic model, geometry (Nx/Ny/dx), tissue (σ/χ/Cm), **delivery/stimulus method**, **sim length** (t_end/dt/save_every), measure, outputs, script path.
- ⛔ **DOUBLE-CHECK GATE** — scientist confirms or corrects; the skill **NEVER runs without it**. THE accountability principle (user: "no crazy vibe coding runoff").
- ON "GO" → create a **dedicated experiment folder**, write `MANIFEST.md` (the confirmed text, verbatim = the record) + `run.py`, append a one-line entry to the **master log**.
- RUN (offer) → verify results are sane, save standardized media, write results back to manifest/log.

**Folder structure:** `Lab/` (new top-level) — `Lab/NOTEBOOK.md` master log + per-experiment `Lab/{date}_{slug}/` (`MANIFEST.md`, `run.py`, `outputs/`). Skills 1 & 4 share this home (the notebook organizes itself as each run drops a folder + a log line).

**Manifest fields:** the listed ones are core; extra fields (scientist/initials, hypothesis, expected runtime) are **OPTIONAL** (user: "make it optional") — include when relevant, never required.

**Key asset = a MAINTAINED `api-cheatsheet`** (current, correct `cardiac_core` calls) in the skill bundle. This is what prevents the #1 LLM-sim-code failure mode (hallucinated API). Refresh it now that `cardiac_core` just shipped (137 tests). The existing `API_REFERENCE.md` predates the consolidation and is design-oriented — distill a CURRENT cheatsheet from the shipped API (`cc.Grid`, `cc.ConductivityConfig`, `cc.monodomain/bidomain/lbm`, `cc.simulate`, `result.run().cv()/.apd()`, `cc.media`).

**Skill format:** `.claude/skills/{name}/SKILL.md` + a `reference/` folder — the FIRST *bundled* skill in this repo (all existing skills are single `SKILL.md`). Output = `.py` script (notebook option left open). Build ON existing conventions: `media/` + `cardiac_core.media.media_path`, the experiments pattern, the `Research/` doc architecture.

**Next:** `/blueprint` the build — phased: api-cheatsheet → `/sim-experiment` (keystone) → `/sim-preset` → `/sim-media` → `/sim-notebook`. (User: "think we should blueprint.")

### 2026-06-25 (exec): consolidation EXECUTED — cardiac_core self-contained (Phases 0–5, 137 tests)
Ran the vendoring plan phase-by-phase, each test+integrity-gated and committed. Backup first (tag `pre-consolidation-vendoring` + 739M bundle). Result: `cardiac_core` owns all 3 engines (`_monodomain`/`_bidomain`/`_lbm`) + shared `ionic`/`mesh`/`stimulus`; hack deleted; 137 green.

**Two real bugs hit + fixed during execution (worth remembering):**
1. **Cross-ref rewrite regex corrupted internal imports.** First pass used `from \.+(ionic|tissue_builder)` WITHOUT `\b` → matched the `ionic` prefix of `ionic_time_stepping` → rewrote 4 solver-INTERNAL imports to `cardiac_core.ionic_time_stepping` (broken). Caught immediately (source untouched), re-copied clean, re-ran with `\b`-anchored regex → exactly 8 cross-refs, internals intact. Lesson: ALWAYS `\b`-anchor `ionic`/`tissue_builder` in the rewrite.
2. **Solver package name shadowed the factory.** Named it `cardiac_core/monodomain/` → `from cardiac_core import monodomain` returned the PACKAGE (real submodule beats the lazy `__getattr__`), so `monodomain(mesh)` → `'module' object is not callable` (ordering-dependent: only failed once the package was imported). Fix: underscore-prefix the solver packages (`_monodomain` etc.) so they don't collide with the public factory names. This is exactly the "don't import _* directly" design I'd floated; now enforced.

**Other notes:** the self-containment guard initially false-positived on PROSE (comments/docstrings saying "no `_prepare_engine()`") — refined to match the call form `_prepare_engine(` with inline-comments stripped. `stimulus/protocol.py` reconciliation = bidomain's `+=` (canonical; the V5.5 `=` overwrite differs only for overlapping stims — goldens single-stim so bit-identical). The `conda run python - <<heredoc` inline form silently no-op'd the rewrite (stdin issue) → wrote the rewrite script to a file and ran it (per CLAUDE.md temp-file guidance).

### 2026-06-25: DECISION — consolidation = unified ground-up package (Approach A2), source-verified
After the Goal-1 API shipped, did a "final alignment" with the user on what cardiac_core actually is. Confirmed it's a WRAPPER that references the 3 engines from `Monodomain/Engine_V5.5/`, `Bidomain/Engine_V1/`, `LBM/Engine_V1/` via `_prepare_engine()` (sys.modules flush + sys.path swap) — NOT a single unified codebase. The `_prepare_engine` hack exists because V5.5 and Bidomain BOTH name their top package `cardiac_sim` (collision); LBM uses `src`.

**Key source finding (reshaped the whole plan):** the 3 engine trees are **100% relative-import internally with ZERO gotchas** — `grep` over all of them: 0 absolute `cardiac_sim`/`src` imports, 0 `sys.path`, 0 `importlib`, 0 `__file__`, 0 name-as-string-literal. (V5.5 85 .py / 186 relative-import lines; Bidomain 78 / 165; LBM 25 / 17.) So the engines are hermetic, relocatable bricks. My earlier estimate that removing the hack meant "~70+ absolute-import rewrites" was WRONG — relative imports are name/location agnostic, so renaming/moving a tree changes nothing inside it. The solver code reaches OUT to shared code only ~7×/engine (`from .....ionic`, `from ....tissue_builder` in V5.5's `simulation/`); everything else is solver-internal (`from ..base`, `from ....state`).

**Decisions (user, 2026-06-25):**
- **Approach A2 — unified, flat, ground-up.** NOT Approach B (relocate-but-keep-hack, leaves nested `cardiac_core/engines/X/cardiac_sim/...` blobs + packaging-exclusion). NOT rename-only-A1 (silos with triplicated shared code). User: "I wanted a unified simple ground up end product not nested reference and import."
- **`cardiac_sim` DISSOLVES** (user OK'd renaming mono+bidomain's package). No `cardiac_sim`, no `engines/` nesting. Shared parts → top-level siblings; solver parts → `cardiac_core/{monodomain,bidomain,lbm}/`.
- **Unify ionic + mesh + stimulus** this pass (the "do the first" option). `ionic/` already extracted (canonical copy from Phase-1). `mesh/` needs a SUPERSET (bidomain `StructuredGrid` adds `boundary_spec`; mono's doesn't). `stimulus/` already aligned (all accumulate `+=`). `tissue`/conductivity *internals* stay per-engine (solver-specific, already fronted by `ConductivityConfig`).
- **Copy, don't delete.** The 3 engine folders (`Monodomain/Engine_V5.5/` etc.) stay untouched as-is (their own dev/tests). cardiac_core gets the unified copy. Drift is the accepted cost; note a re-vendor strategy.
- **Delete `_prepare_engine()`** — once each piece has a unique `cardiac_core.*` dotted name there's no collision, so the hack and the packaging-exclusion both go away. Both engines become importable simultaneously (normal Python).

**Target layout** (audit-ready): `cardiac_core/{ionic,mesh,stimulus}` (shared) + `cardiac_core/{monodomain,bidomain,lbm}` (slim solvers) + the existing api/run/conductivity/grid/simulation/analysis/geometry/io/file_format/media. Internal imports: solver-internal stay RELATIVE (untouched as the subtree moves intact); the ~10–20 solver→shared cross-refs rewrite to absolute `from cardiac_core.{ionic,mesh,stimulus}...`. `api.py` swaps the `_prepare_engine + from cardiac_sim...` blocks for direct `from cardiac_core.{monodomain,bidomain,lbm}... import ...`.

**Risk (bounded):** step that merges the slightly-divergent shared modules — only `mesh` (StructuredGrid superset incl. `boundary_spec`) is non-trivial; ionic done, stimulus aligned. FEM/TriangularMesh: keep as-is during the move (FEM-removal is a SEPARATE confirmed-but-deferred cleanup; don't entangle). **Verification each phase: all 121 cardiac_core tests green + a guard that no `cardiac_core/**` file references `Monodomain/`,`Bidomain/`,`LBM/` paths or the `_prepare_engine` hack.**

**DECISION (user, 2026-06-25): `cardiac_core` is the CENTRALIZED home.** Future engine improvements happen in `cardiac_core/{monodomain,bidomain,lbm}` — the original engine folders (`Engine_V5.5`, `Engine_V1`×2) become frozen/legacy. This resolves the only real objection to copy-vendoring (drift): there is no drift if the vendored copy is the single living source going forward. The vendoring consolidation is greenlit. Also confirmed: public API is engine-as-parameter (`cardiac_core.simulate(engine='lbm')` / the `monodomain()`/`lbm()` factories) — the `cardiac_core.monodomain` etc. subpackages are INTERNAL plumbing the user never imports (mark them private). Optional thin add: a single `build(engine=…)` live-sim entry.

**Next:** `/blueprint` [DONE — PLAN.md 2026-06-25] → `/audit` → execute per-phase (test-gated) → audit the final folder structure.

### 2026-06-24 (impl): PLAN.md executed — Goal-1 construction API shipped (Phases 0–5, 121 tests)
After a 3rd `/audit` pass on PLAN.md (5 findings folded in: the HIGH `save_result` positional-`phi_e` break + 2 MED missing-section + 2 LOW), executed all 6 API-track phases end-to-end. Each phase: implement → targeted test → full-suite gate. No engine source touched (V5.3/V5.4/V5.5/Bidomain/LBM unchanged); cardiac_core grew `conductivity.py`, `grid.py`, `simulation.py` + refactored `api.py`/`run.py`/`io.py`. Result: **121 cardiac_core tests pass** (80→121), incl. the live-CV firewall gate.

Key implementation decisions / gotchas (beyond the design):
- **`ConductivityConfig.sigma_eff` is the PUBLIC property** (per API_REFERENCE), so the isotropic stored conductivity lives in a `sigma_iso` field (the plan's sketch had a `sigma_eff` field + `sigma_eff_value` property — would clash). Arithmetic mirrors the probe exactly.
- **Live-CV gate runs in a SUBPROCESS** (`tests/_live_cv_gate_driver.py`) — running test_phase10's V5.5 cable inside the cardiac_core pytest session would collide on the shared `cardiac_sim` namespace (flushed by `_prepare_engine`). Subprocess isolates it; ~2 min, skips cleanly if the V5.5 dir / ref JSON is absent. (First run hit a `numpy.bool_` not-JSON-serializable bug → cast `bool()/float()`.)
- **Bidomain σ-tuples must be `(Nx,Ny)` FIELDS, not scalars** — the bidomain FDM indexes `dxx[i,j]`; passing scalar σ gave a 0-d-array IndexError. `_build_mesh_data` now emits `np.full((Nx,Ny), σ)` tuples.
- **`stimulate()`/`reset()`/`with_()` unified across both construction paths** by routing ALL stimuli through `data.stimuli` and replaying the factory with `mesh=self._data` + stored `_build_kwargs`. This made the audit-MEDIUM "must work for both paths" trivially true and sidestepped the LBM `start`/`start_time` audit-LOW (the existing LBM factory loop already reads `data.stimuli` positionally).
- **`run()` eager flip needed a 34-site migration** `*.run(`→`*.snapshots(` across 6 test files + the production `run.py::_collect` (engine-direct `sim_direct.run` left alone). Done with a word-boundary regex script.
- **CV smoke needed t_end=40ms** (front ~50 cm/s = 0.05 cm/ms; x2=1.0cm activates ~21ms) — at t_end=20 the far probe never activated → CV=nan.

OPEN (handed back to user): git — all work is uncommitted on `main` alongside pre-existing unrelated changes (source_sink, MASTER.md, the design docs). Per harness rule I did NOT auto-commit to the default branch. The plan's per-phase commit points are ready to apply once a branch/commit strategy is chosen.

### 2026-06-24: Unified API drafted — `Simulation` Protocol + 4 idioms + `SimulationSpec`
Picked up from the glossary's 3 open items and produced **`API_DESIGN.md`** (Goal-1 interface in the resolved vocabulary). Settled the open items + drafted the interface.

**Open glossary items resolved:**
- **#9 default stim amplitude → −52** (user). Rationale: ionic model is byte-identical across engines and the stimulus enters the same `R=-(Iion+Istim)/Cm` term in the same units → an amplitude that depolarizes in M/B depolarizes identically in L. LBM's −80 was author drift; retire it. The glossary's "verify under L" is automatically satisfied.
- **#5 internal live-State → unify + defer.** One internal `State(t, Vm, ionic_states, …, Cm, coords)` with optional `phi_e` (bidomain) / `f` (LBM); LBM adopts it. Zero public-contract impact (only `SimulationResult` is user-facing) → land in a code phase, not now.
- **#12 ConductivityConfig interface — drafted, then source-verified + CORRECTED.** The class stores **physics** (`sigma_i/sigma_e/sigma_eff, chi, Cm, fiber_angle`) and emits per-engine inputs. Construction via classmethods `.isotropic/.bidomain/.anisotropic`. **#13 (chi only in ConductivityConfig) makes #12 work.**

### 2026-06-24 (cont.): VERIFIED ConductivityConfig vs source — caught a Cm≠1 bug in my own draft
Read the actual operators instead of trusting the KNOWLEDGE summary. Confirmed:
- `fdm.py:195–238` (V5.5): implicit solve `(χ·Cm·I − ½dt·L)Vⁿ⁺¹ = (χ·Cm·I + ½dt·L)Vⁿ`, `L` built from input `D` (NOT χ·Cm). ⟹ **physical diffusivity = `D_input/(χ·Cm)`** (Form A confirmed). Reaction divides by `state.Cm` (V5.5 fix, test_phase10 @3.55e-15).
- `BidomainConductivity` (`conductivity.py`): `D_i,D_e` PRE-scaled `=σ/(χ·Cm)`; has `get_effective_monodomain_D()=D_i·D_e/(D_i+D_e)` (the harmonic i/e collapse — the `D_eff` reduction). LBM `sigma_to_D`: `D=σ/(χ·Cm)` pre-scaled. (Form B confirmed.)
- **BUG in my first §4 draft:** "feed `D_eff` with `chi=1, Cm=1` no-op" is correct ONLY at Cm=1. At Cm≠1, pinning the engine's `Cm=1` makes the **reaction** divide by 1 instead of the real Cm → silently wrong. **Same Cm-trap family as the false time-dilation invariant** (invisible at pinned Cm=1, bites otherwise). The real Cm must reach EVERY engine.
- **Corrected mechanic:** only the *diffusion input's* Cm-scaling differs by formulation. Form-A monodomain scales diffusion by Cm internally (mass term) → feed it Cm-**un**scaled `D = D_eff·Cm = sigma_eff/chi`, with engine `chi=1` (chi folded in) and the **real Cm** (drives mass term + reaction). Form-B (bidomain/LBM) → feed fully-scaled `D = σ/(χ·Cm)` + real Cm. ConductivityConfig now exposes per-engine emitters `for_monodomain()/for_bidomain()/for_lbm()` so this arithmetic lives in ONE place. At Cm=1 all collapse to `D_eff = sigma_eff/chi` (= what the 2026-05-30 cross-engine test used: `D=D_EFF, chi=1`).
- **Also fixed a units trap in the §7 smoke test:** `ConductivityConfig.isotropic(sigma=...)` takes raw CONDUCTIVITY (mS/cm), not pre-divided `D`. `0.00097` is the *D_eff*, not σ. Standard tissue σ_i=1.74, σ_e=6.25 → D_eff=0.000972.
- **Build-time gate (still open):** confirm `for_monodomain()` reproduces test_phase10 CVs at Cm∈{1,2} once coded.

### 2026-06-24 (cont.): DECISION — canonical formulation = B (converge in Phase 4)
User asked: now that both A and B are physically correct, which is "good"? Reasoned it through: **the V5.5 reaction fix made the physics a TIE, so this is now a pure software-engineering decision** — and B wins on every axis: (1) consolidation alignment — B confines all χ/Cm scaling to `ConductivityConfig` (decision #13); A's engine is a *second* scaling authority (its χ·Cm mass term); (2) non-fragile — A scatters χ·Cm across FDM mass + FEM M + FVM Vol + DCT/FFT denominators + reaction (that scattering caused the V5.4 bug); B has one σ→D line; (3) majority — 2/3 engines already B; (4) clean operator `(I−θ·dt·L)` vs A carrying χ·Cm=1400 into the linear algebra. A's only edge (operator reads like the PDE) is a *docs* value, neutralized at the API (user passes σ/χ/Cm to ConductivityConfig either way).
**DECISION (user): Form B target; converge in Phase 4.** Two-phase: keep both now (firewall `for_monodomain()` absorbs the asymmetry); convert monodomain Form-A diffusion→B *as part of* the Phase-4 rewire into cardiac_core (no new fork) → then DELETE `for_monodomain()`, ConductivityConfig collapses to one emitter (physical D + Cm). Recorded in KNOWLEDGE Key Decisions ("Canonical formulation = B") + migration-plan Phase 4 + API_DESIGN §4 + glossary.

### 2026-06-24 (cont.): GATE CLOSED + FEM ditch CONFIRMED + API reference doc
- **ConductivityConfig firewall gate — CLOSED (numerically).** Wrote `Monodomain/Engine_V5.5/_probe_conductivity_firewall.py`: raw `sigma_i=1.74, sigma_e=6.25, chi=1400` → `for_monodomain()` → live V5.5 cable (reuses `test_phase10.run_cable_v55`). Result: arithmetic `D=0.0009721973895941` = reference `D_EFF` to **1.1e-19** (Cm-independent ✓); CV(Cm=1)=**54.35** (0.00% vs bidomain ref), CV(Cm=2)=**28.09** vs 27.77 (**1.15%** < 5%). The Cm≠1 firewall path is correct in the live engine, not just on paper. (Probe is a keep-or-toss artifact; the permanent test lands in `cardiac_core/tests` when ConductivityConfig is built in Phase 3.)
- **FEM ditch — CONFIRMED (user).** Structured grid is now the ONLY standard (P2→P2′). Drops the unstructured/flat-`(n_dof,)` geometry path, `TriangularMesh`, monodomain's `FEMDiscretization`. **FDM primary; FVM survives** (structured-grid-native); collapsing FVM→FDM is a SEPARATE later question. Composes with the Form-A→B convergence in the Phase-4 rewire. API_DESIGN §9 marked CONFIRMED.
- **Deliverable: `API_REFERENCE.md`** — library-style reference (every class + function, signatures, params, returns, examples). Built from API_DESIGN.

**New decisions (user, 2026-06-24):**
- **CHANGE idiom = functional `sim.with_(**overrides)` → new Simulation** (immutable, sweep-safe; no mutable setters in the public API — clean for Optimizer).
- **Construction = factories + spec, layered.** Per-model `monodomain()/bidomain()/lbm()` factories (programmer surface) AND `create_simulation(SimulationSpec)` on top (LLM-intake surface). Spec→factory; building the spec now keeps the questionnaire from drifting from engine needs ("spec schema = the intake questionnaire" — the cross-goal leverage point).

**`API_DESIGN.md` structure:** §0 four idioms · §1 `Simulation` Protocol · §2 factories · §3 stimulus · §4 ConductivityConfig (the chi/Form-A/B firewall) · §5 `SimulationResult` (only public output; eager + `batch=`) · §6 `SimulationSpec`/`create_simulation` (3 tiers required/defaulted/derived; Goal-2 bridge) · §7 minimal-spec smoke test · §8 open/deferred · §9 the FEM-ditch pending note.

**FEM-ditch flagged (user, discuss later):** all sims on structured grid → P2 strengthens to "structured ONLY." Drops the unstructured/flat path, `TriangularMesh`, FEM knob; simplifies State #5; touches engine implementation (Phase-4). Recorded as pending in `API_DESIGN.md` §9, not yet committed.

**Next:** discuss the FEM-ditch (its knock-on for engine implementation) → then either (a) finalize `ConductivityConfig` class shape against source (verify the chi=1 no-op), or (b) start turning `API_DESIGN.md` into the actual `cardiac_core` `Simulation`/Protocol + factory code.

### 2026-05-31: Session vision — unified API + LLM wrapper → conversational simulation builder
North star (now the question's main goal, see README). A non-coder converses with Claude to build cardiac sims and learn how conduction works. Two goals:
1. **Unified construction API (Goal 1)** — one standardized, engine-agnostic, easy-to-construct way to declare + run: a declarative, validated, serializable **SimulationSpec** → run → **SimulationResult** → analysis. Consolidates today's split config (`CardiacMeshData` fields + `simulate()` call-args) into ONE object. Three field tiers: **required** (LLM asks) / **defaulted** (silent good values) / **derived** (computed).
2. **Self-contained LLM wrapper (Goal 2)** — Claude skills + reference docs driving Goal 1 under a strict protocol (gather → validate → construct → run → verify → present).

Key design insights (settled-ish; revisit when building Goal 2):
- **Spec schema = the intake questionnaire.** Make spec fields self-describing (`{required?, prompt, options, default}`); the LLM "gather" step = ask the prompt of each unfilled required field. The questionnaire can't drift from what engines need (same schema). THE cross-goal leverage point.
- **Pacing abstraction.** High-level protocol (`single` / `s1s2` / `regular(bcl, n_beats)`) that EXPANDS into the low-level stimulus list engines consume. Non-coder speaks in beats, not timestamps.
- **Outputs drive the run.** What the user wants to MEASURE (CV / APD / LAT / reentry) feeds back into numerics/run (`save_every`, `t_end`), not just post-hoc analysis.
- **Engine = explicit in spec, but LLM-inferred from the scientific question** + records rationale (auditable, overridable). e.g. bath/boundary effects → bidomain; fast/simple → monodomain.
- **Defaults philosophy.** A minimal spec ("pace this sheet, measure CV") must RUN via physiological defaults (TTP06/EPI, dt=0.02, strang, CN/pcg, chi=1400, Cm=1) — "one obvious way".

Deferred: user geometry input (Fiji drawing → Builder image→mesh; a designated drawings inbox; the Fiji-export→mask format contract). Assume geometry is provided for now.

**Decided focus (2026-05-31):** build the foundational **API** FIRST — everything (the spec questionnaire, the LLM wrapper) is contingent on a clean, standardized construct + run + results surface in `cardiac_core`.

### 2026-05-30: Phase 1 scoped — ionic is ~unified across engines; full direct migration decided
Verified the plan's "ionic identical across engines" assumption before executing. Findings:
- **Classical engines (V5.5 ↔ Bidomain V1): shared model files byte-identical** — `base.py`, `lut.py`, `ttp06`, `ord`, `mhas13`, `phas13`. V5.5 only adds `paci` + `__init__` exports.
- **LBM V1 ionic is NOT a fork** (initial diff misled): same `IonicModel` ABC (base.py byte-identical), same `ttp06/` structure (calcium/celltypes/currents/gating/model/parameters), `ord/model.py` byte-identical. Differs only by: (1) one keyword rename `cell_type_is_endo` (V5.5) vs `celltype_is_endo` (LBM) in a lut call; (2) top-level `ionic/` namespace vs `cardiac_sim.ionic`; (3) model subset (ttp06+ord only); (4) a dead stray `LBM/Engine_V1/ionic/ionic/` (imported by nothing).
- **Rewire surface:** ~23 consumer sites (14 V5.5, 9 Bidomain V1) + LBM's, all relative imports of varying depth (`from ...ionic`, `from .....ionic.base`).

DECISIONS (2026-05-30):
- **Scope: all three engines** (divergence is trivial, not a fork). cardiac_core/ionic/ = canonical SUPERSET (include paci/mhas13/phas13; reconcile the one keyword rename to a single canonical name).
- **Strategy: Option B — direct rewire, EXACT migration. NOT shims, NOT a sys.path trick.** User directive: "exact migration of all engines, not as path." Engine-local ionic copies DELETED; all consumers import `cardiac_core.ionic.*` absolutely; cardiac_core made a properly importable package (editable install / real package), not per-engine `sys.path.insert`.
- **Sequencing: 1a** build canonical `cardiac_core/ionic/` superset → **1b** rewire+delete classical engines (V5.5 + Bidomain V1), run their suites → **1c** rewire+delete LBM V1 (handle top-level namespace, keyword, drop stray ionic/ionic), run its suite. Verify cardiac_core 77 + every engine suite green.
- Rejected: re-export shim (A) and A-then-B staging — user wants the clean end-state directly.

Baseline before Phase 1: cardiac_core 77/77 pass; classical/LBM suites green (this session).

### 2026-03-16: The core tension is engine-centric vs. research-centric layout
Engines serve multiple research questions, but the directory structure forces navigation by engine. A single experiment may touch Bidomain V1 and Monodomain V5.4, yet there is no natural place for it. Proposed restructuring: `Engines/` top-level with `cardiac_core/` as the shared package, `Pipelines/` for optimizer/surrogate/builder, `Research/` for writing only.

### 2026-03-16: Three proposed structures, iterated to final form
First proposal grouped everything under `Engines/` and `Pipelines/`. User pushed back on groupings. Second revision separated concerns more cleanly. Third iteration established the principle: Research = writing (no .py files), Engines = code, cross-linked via MASTER.md and EXPERIMENT.md backlinks.

### 2026-03-16: Identified the experiment gap
Traced the research cycle (Hypothesis -> Script -> Run -> Outputs -> Analysis -> Finding -> Knowledge) and found each step lived in a different location with no links between them. Scripts lived in engine test directories, outputs were ephemeral, and analysis was manual. Solved by adding `experiments/` directories inside engines with EXPERIMENT.md backlinks to research questions.

### 2026-03-16: Chi/Cm audit revealed two valid formulations
Audited chi/Cm handling across all three engines. Found Formulation A (V5.4: chi*Cm in mass term, ionic solver does NOT divide by Cm) and Formulation B (Bidomain V1, LBM V1: D pre-scaled, ionic solver divides by Cm). Both produce identical results when Cm=1.0. Decision: keep both, unify at the API level. Converting V5.4 would risk 77 tests for no practical benefit since Cm is always 1.0 (ionic models output pA/pF).

### 2026-03-16: Diffusion tensor encoding differs by method but is reconcilable
Mapped what each discretization method receives (FDM 5pt gets Dxx/Dyy, FDM 9pt gets full tensor, LBM D2Q5 gets scalar, D2Q9 MRT gets full tensor). ConductivityConfig can be the single entry point: user provides sigma, it converts to D with chi/Cm in one place, then each engine extracts what it needs.

### 2026-03-17: Phase 0 completed — API wrapper with 34 tests
Built `cardiac_core/` as an API wrapper: `monodomain()`, `bidomain()`, `lbm()` functions return `CardiacSimulation` with `.run()` generator. File format: `CardiacMeshData` dataclass with `.npz` save/load. Verified wrapper output matches direct engine construction exactly (atol=1e-10). The `_prepare_engine()` hack flushes `sys.modules` because both engines use `cardiac_sim` namespace. This is temporary; goes away in Phase 1.

### 2026-03-17: Code duplication inventory — 15+ files across 3 engines
IonicModel ABC + lut.py (3 copies, identical), TTP06 5 files (3 copies, identical), ORd 6 files (3 copies, identical), PCG solver (2 copies, minor divergence), splitting strategies (2 copies, identical logic), stimulus protocol (2 copies, semantic difference: += vs =), StructuredGrid (2 copies, bidomain adds boundary_spec).

### 2026-03-17: Decided LBM V1 is canonical over V5.4's LBM
LBM V1 has more features: MRT collision, D2Q9 lattice, 3 boundary condition types, torch.compile kernel fusion. V5.4's LBM is simpler but less capable. Decision: LBM V1 is the canonical implementation.

### 2026-03-17: Full project document/folder map established
Mapped every document and folder in the project, clarifying what belongs where. Established conventions: Research/Active for open questions, Research/Complete for answered questions (read-only KNOWLEDGE.md), Research/Knowledge for promoted findings, experiments inside engine directories.

### 2026-05-29: V5.5 detour decided — fix the Formulation-A reaction Cm bug in an independent fork
Revisited the chi/Cm audit. Derived *why* chi is safe but Cm is not: normalizing the parent PDE by chi·Cm cancels chi out of the reaction term entirely (chi·I_ion / (chi·Cm) = I_ion/Cm), so chi lives only in D (one half → can't break splitting). Cm appears in BOTH halves (D = sigma/(chi·Cm) AND the reaction /Cm), so Formulation A — which handles Cm in the diffusion mass term but drops it in the reaction — is silently wrong for any Cm != 1.0. NOT a "change-Cm-midway" hazard; it's wrong at t=0 for any Cm != 1. Safe today only because the project pins Cm=1.0 (ionic models output pA/pF).

Decision: create **Monodomain Engine_V5.5** as a full independent copy of V5.4 (rationale: backup — V5.4 stays the frozen validated baseline; don't risk its 77 tests). The ONLY change is making the reaction Cm-correct. Diffusion is NOT touched (its chi·Cm mass term already handles arbitrary Cm — verified: dividing the implicit theta-solve by chi·Cm yields effective D = sigma/(chi·Cm) for any Cm). This is a reaction-only minimal fix, NOT a full Formulation-B structural conversion.

Code facts located (V5.4):
- Bug sites (operator-split ionic steppers, both miss /Cm):
  - `cardiac_sim/simulation/classical/solver/ionic_time_stepping/rush_larsen.py:83` → `state.V = V + dt * (-(Iion + Istim))`
  - `cardiac_sim/simulation/classical/solver/ionic_time_stepping/forward_euler.py:64` → same
  - Fix: `... / Cm`, dividing by the TISSUE Cm.
- Plumbing gap: `SimulationState` (`state.py`) has no Cm field; `_build_ionic_solver(name, ionic_model)` (`monodomain.py:91`) doesn't pass Cm. Need to wire tissue Cm → ionic step. Preferred: add `Cm` to SimulationState and read it in the stepper, mirroring Bidomain V1's `Cm = getattr(state, 'Cm', 1.0)`.
- TWO-Cm hazard: tissue/cable Cm (`tissue/isotropic.py:23`, the `chi·Cm` in `fft.py` denominators) is the one to divide by. The ionic models' internal `Cm` (`ttp06/model.py:548 p.Cm=0.185`, paci/ord calcium `inv_VcF`) is a fixed per-cell constant for Ca/Na concentration flux — DO NOT touch or conflate.
- Aside: ORd standalone `model.py:802` already does `/p.Cm`, and TTP06 `model.py:297 dV=-I_ion` does not — but neither standalone path is used by the classical operator-split solver, so the steppers are the real fix sites.

Test protocol (settled intent, blueprint to formalize):
- **Regression (Cm=1):** V5.5 reproduces V5.4 bit-identically across the existing 77-test suite (the copy must not change Cm=1 behavior). atol ~1e-12.
- **Cm-scaling invariant (Cm=k):** scaling tissue Cm by k is equivalent to time-dilation by k — solution V(x, t; Cm=k) == V(x, t/k; Cm=1); observable as CV → CV/k and APD → k·APD, identical spatial structure. Requires BOTH halves to scale with Cm, so it fails on V5.4 (broken reaction) and passes on V5.5. This is the discriminating test.
- **0D single-cell version:** no diffusion; verify dV/dt = -(I_ion)/Cm directly (trajectory at Cm=k is the Cm=1 trajectory slowed by k).

**Bidomain V1 as independent oracle (added 2026-05-29):** Bidomain V1 `rush_larsen.py:81-84` is ALREADY the exact target form — `Cm=getattr(state,'Cm',1.0); state.V = V + dt*(-(Iion+Istim)/Cm)` (Formulation B). Use it two ways: (1) code-parity anchor for V5.5's fixed line; (2) cross-engine dilation oracle — run a matched cable in bidomain (reuse `cv_shared.py` `run_bidomain`, `measure_cv_from_history`) at Cm=1 and Cm=k, confirm CV ratio→1/k, assert V5.5's ratio matches. Bidomain shares NO solver code with monodomain → strong independent check. Process-isolated (bidomain also uses `cardiac_sim`): generate a `bidomain_cm_ref.json` in a separate process, load it in the V5.5 test.

**V5.4 internal LBM is dead code — DROP it in V5.5 (decided 2026-05-29):** Verified across the repo: `cardiac_sim/simulation/lbm/` has ZERO importers outside itself, ZERO standing tests (PROGRESS "Phase 5 LBM DONE" was historical; no `test_phase5.py` ships; the live suite test_phase7/8 + test_boundary_modes + tissue tests never touch it), ZERO experiments. `simulation/__init__.py` only names it in a docstring. `step_with_V` (ionic base) is the LBM path's only hook and is called solely from `lbm/monodomain.py:237`. The boundary-conduction research — the only active LBM work — runs on the SEPARATE `LBM/Engine_V1` engine (`diag_lbm_specular.py` → `sys.path.insert(LBM/Engine_V1)`, `from src.simulation import LBMSimulation`); all new BCs (same-cell specular, HBB, bounce-back, 27-rule enumeration) live in LBM V1's `src/`. So instead of guarding V5.5's LBM at Cm!=1, just DELETE the `lbm/` package + dead `step_with_V`. V5.5 becomes clean classical-only + Cm-correct; the chi/Cm source-term entanglement (the reason a guard was considered) vanishes with it. V5.4 keeps its LBM as the faithful backup. (Earlier plan had a fail-loud guard phase; superseded.) NOTE: `test_boundary_modes.py` is an FDM boundary-mode test, NOT an LBM test — don't use it to "verify LBM".

**Formulation A vs B D-input asymmetry (CRITICAL for the comparison):** Monodomain V5.4/V5.5 (Form. A) takes input `D` = sigma; engine forms physical diffusivity = D/(chi·Cm) internally. So "scale Cm" = hold D fixed, change Cm (diffusion dilates automatically). Bidomain V1 (Form. B) takes input D_i/D_e = already-scaled sigma/(chi·Cm); so the SAME experiment requires rescaling D_i,D_e→/k when Cm→k·Cm. Compare dimensionless ratios (CV→1/k), not absolute CV, so engines needn't match in absolute units.

Open: exact plumbing (state.Cm field vs constructor arg) — recommend state.Cm. Whether V5.5 keeps the `cardiac_sim` package name (collides with V5.4 if both imported, like the existing Bidomain/V5.4 collision) — for standalone test runs it's fine; note for the eventual cardiac_core consolidation.

### 2026-05-30: Phase 2 physics correction — the Cm time-dilation invariant is FALSE
While executing Phase 2, the 0D test empirically failed: APD90(Cm=2)/APD90(Cm=1) = 1.34, not the predicted 2.0. Root cause: the tissue Cm divides ONLY the voltage update (`dV = -(Iion+Istim)/Cm`). The gate kinetics (`tau` from `compute_gate_time_constants(V,S)`) and the concentration rates carry NO Cm — they are intrinsic membrane kinetics. So scaling Cm→k·Cm does NOT rescale the whole system in time: V slows but gates keep their kinetics, so the AP MORPHOLOGY changes; it is not a `t→t/k` stretch. Substitution proof: for `W(t)=V(t/k)` to satisfy the Cm=k system you'd need `k·tau = tau` ⇒ only k=1. The invariant (and `CV→CV/k`, `APD→k·APD`) is wrong. Both `/reason` and BOTH audit passes missed this (the audit even "verified the physics sound"); the empirical run caught it. The FIX itself is correct — `dV=-(Iion+Istim)/Cm` is the right cable equation; only the *validation strategy* was flawed.

Corrected, rigorous validation (passing): (1) **exact one-step scaling** — from an identical state, `dV·Cm` is invariant across Cm∈{0.5,1,2,4} to 3.55e-15 (machine precision); this directly proves the reaction divides by Cm exactly, independent of morphology. (2) **direction** — larger Cm slows the upstroke (peak dV/dt 368→211 mV/ms) and changes APD (218→292 ms, NOT 2×). Together with the Cm=1 golden (max|dV|=0) and the full existing suite, the fix is rigorously validated. Test file: `Engine_V5.5/test_phase10_cm_scaling.py`.

Step 2.3 RESOLVED (user chose the proper cross-engine check): implemented absolute CV agreement vs Bidomain V1 (independent Formulation-B engine; isotropic + insulated BC reduces to monodomain with D_eff in the bulk). Reference generated by `Engine_V5.5/_regression/bidomain_cm_ref.py` (runs in the Bidomain engine, separate process — both use `cardiac_sim`). Matched physical diffusivity: bidomain D_i,D_e -> /Cm; V5.5 holds input D=D_EFF fixed with chi=1 (so D_phys=D_EFF/Cm) — both give D_eff=D_EFF/Cm. RESULT: Cm=1 V5.5 54.35 vs bidomain 54.35 cm/s (0.0%, exact threshold-grid match; also reproduces the historical 54.3 benchmark); Cm=2 V5.5 28.09 vs bidomain 27.77 cm/s (1.1%). Both << 5% tol. Phase 2 PASSES.

Refinement on CV vs APD scaling: empirically CV(Cm=2)/CV(Cm=1) ≈ 0.51 in BOTH engines — i.e. CV ~ 1/Cm. This is eikonal scaling, NOT dilation: CV ∝ sqrt(D_phys · upstroke_rate), and both D_phys ∝ 1/Cm and the upstroke rate (dV/dt = -Iion/Cm) ∝ 1/Cm, so CV ∝ 1/Cm. APD does NOT scale (set by repolarization gate kinetics, no Cm — measured 218→292 ms, not 2x). So the original plan's "CV→CV/k" was approximately right for the wrong reason; "APD→k·APD" was simply wrong. The cross-engine test does not depend on either — it compares two correct engines' absolute CVs.

NOTE: `cv_shared.run_monodomain_fdm` is NOT Cm-aware (line 303 has no /Cm, takes no Cm arg) — it cannot serve as a Cm!=1 reference. Only Bidomain V1 (run_bidomain) is a confirmed Cm-correct independent engine. cv_shared SIGMA_I=1.74, SIGMA_E=6.25, chi=1400 -> D_EFF=0.000972 (the test reads D_EFF_input from the ref JSON to avoid hardcoding drift).

## Failed Approaches
- **Flat engine-centric structure** (2026-03-16) — failed because: engines serve multiple research questions, making it impossible to find all work related to a single question. No natural place for cross-engine experiments.
- **First proposed restructure** (2026-03-16) — failed because: user wanted different groupings; initial Pipelines/Research separation didn't match actual workflow.
- **Converting V5.4 to Formulation B IN PLACE** (2026-03-16) — rejected: would risk V5.4's 77 passing tests. RESOLUTION (2026-05-30): instead FORKED V5.5 with the Formulation-B reaction; V5.4 stays frozen. (So Formulation B was the right target — just not destructively on V5.4.)
- **Cm time-dilation invariant for validation** (2026-05-30) — FALSE. Assumed `V(x,t;Cm=k)==V(x,t/k;Cm=1)` (⇒ CV→CV/k, APD→k·APD). Tissue Cm divides only the voltage update; gate kinetics/concentration rates carry no Cm, so Cm changes AP morphology, not timescale (APD 218→292 ms at k=2, not 2×). Asserted by the plan AND both audit passes; caught empirically (0D APD ratio 1.34). Replaced with exact 1/Cm one-step scaling (machine precision) + Bidomain V1 absolute-CV cross-check. The fix was always correct; only this validation premise was wrong.
- **`cv_shared.run_monodomain_fdm` as a Cm≠1 reference** (2026-05-30) — won't work: it has no `/Cm` and takes no Cm arg (hardcoded Cm=1). Used Bidomain V1 (`run_bidomain`) instead.
- **Merging solver internals into cardiac_core** (2026-03-16) — rejected because: solvers are engine-specific (decoupled GS for bidomain, CN/BDF for monodomain, BGK/MRT for LBM). Only shared code (ionic, mesh, stimulus) should be unified.
- **sys.modules hack as permanent solution** (2026-03-17) — recognized as temporary: `_prepare_engine()` flushes modules because both engines use `cardiac_sim` namespace. Acceptable for Phase 0 wrapper but must be eliminated when shared code moves into `cardiac_core/`.

## Session Log

### 2026-07-21 Session (solver hardening SHIPPED "work through all" + tutorial-plan sidequest)
**Worked on**: Executed the audit-driven fix roadmap on branch `solver-hardening`; discussed cardiac_core as a
proper importable+documented library; started a Jupyter tutorial-series sidequest (plan only).
**Accomplished**:
- **Step 1 — make failure loud + Chebyshev M1** (`60acfbe`): shared `SolverConvergenceWarning` (warn by default;
  `filterwarnings('error', ...)` to escalate) at every non-convergence exit across mono pcg, bidomain pcg,
  pcg_spectral, and both Chebyshev solvers. Ported `_gershgorin_bounds_preconditioned` (CH-1) to the mono Chebyshev
  → the 07-02 **M1** bug fixed (regime sweep: 46% err → 3e-15). Immediately surfaced a real under-solved elliptic
  solve in scar-bidomain (M4-family), previously silent. Integrity bit-identical.
- **Step 2 — advanced features** (`91ac993`): `clamp_voltage`/`add_clamp_protocol`/`release_clamp` (per-step
  wrapper-driven `_stepping_run`; scalar/field/callable value; verified holds a strip at exactly 10 mV every frame
  while the rest evolves), `set_voltage`/`set_state`/`get_state`/`state_names` wired to live state. Gave the bidomain
  engine a `step()` (was missing → wrapper `step()` had been broken for bidomain). LBM raises (V is a lattice moment).
  17 tests. Integrity bit-identical.
- **Opt-in solver fixes** (`68b5847`): pcg_spectral mixed-BC → falls back to plain PCG (1.8e-2 stall → 4.2e-8);
  IMEX-SBDF2 2nd-order coupling extrapolation; RKC documented-and-deferred. 6 tests.
- **HONEST correction (verified on the REAL solver, not the toy repro):** the IMEX "fix to 2nd order" only HALVES
  the error — the decoupled parabolic→elliptic *staggering* imposes its own O(dt) floor that extrapolation can't
  lift (self-convergence order stayed ~1.0 before AND after; error 6.5e-3 → 2.2e-3). Documented, not overclaimed.
- **Full suite 260 passed / 2 xfailed** after each step; branch `solver-hardening` (4 code commits) NOT merged.
- **Library-as-product assessment**: `cardiac-core` 0.1.0 IS pip-installed + importable with a clean lazy `__init__`;
  BLOCKER = deps under-declared (only `mcp`; torch/numpy/scipy/torch_dct missing → clean install can't import);
  gaps = no README/LICENSE/`__version__`, API_REFERENCE stranded in Research/ not shipped. ~65% to "importable+documented".
- **Tutorial sidequest**: `cardiac_core/tutorials/PLAN.md` — 8 standalone lessons (L1 import+single-cell → L8 bidomain
  infarct+mixed BC), prep-first (P0.1-P0.6). Recon: single-cell = small uniform grid + whole-domain stim (`Grid(1,1)`
  fails, `Grid(2,2)`+ works); `ipykernel` present, must confirm `nbformat`/`nbconvert`; execute-all gate in Phase W.
- **ADVERSARIAL AUDIT of the solver-hardening branch → CONVERGED (4 rounds).** R1 (3 lanes) found HIGH (the new
  non-convergence warning FLOODED the default bidomain path — declarative-isotropic stores conductivity as a field →
  is_isotropic=False → pcg_spectral breaks down at ~1e-4, warns every step, 437 in the suite; fixed via warn-ONCE-
  per-instance), MED (clamp `_stepping_run` hardcoded mono save-cadence → extra bidomain frame when save_every==dt),
  LOW (reason mislabel). R2 (audit-the-fix + completeness) found MED (warn-once was per-LIFETIME not per-run → reused
  sim silent on run 2+ → added `_reset_solver_diagnostics` per-run re-arm) + 2 LOW (chebyshev check perf, clamp
  docstring overclaim); completeness sweep found NO CRIT/HIGH/MED (GPU clamp/injection, batch/callback run-modes,
  degenerate inputs all verified clean). R3 (audit-the-fix) found MED (my R2 "chebyshev check once per run" was
  UNSOUND — residual is b-dependent + A not fixed across a run for bdf2/IMEX → reverted to check-every-solve, warn-
  once-per-run). R4 CONVERGED — 1 LOW only (two vendored `SolverConvergenceWarning` classes, pre-existing/dedup).
  Severity decayed HIGH→MED→MED→∅. Remediation commits `4003ab5`→`1b66939` + stale-stub-test fix; integrity
  bit-identical throughout; full suite green. Total solver-hardening branch = 10 commits (4 hardening + 6 audit).
**Next**: (1) get the user's call on #13 (GPU sync-free — GPU-only vs regolden vs skip) and #14 (mono-ionic V5.3
align + regolden vs document-only); (2) OR start tutorial Phase P0; (3) merge `solver-hardening` → main (user's call);
(4) optional library packaging pass (declare deps, README/LICENSE/__version__, ship docs).

### 2026-07-16→17 Session (solver + GPU audit → roadmap deliverable; NO code changed)
**Worked on**: New user direction — audit EVERY solver + a dedicated GPU-implementation audit, empirically test
whether `device='cuda'` uses the GPU and is optimized (user's "explicit→GPU, implicit→CPU crossover weirdness"),
then set the forward path (advanced features + Phase 2-5). Audit-only per user ("measure first, don't change
solver code").
**Accomplished**:
- **GPU benchmark** (`scratchpad/gpu_bench.py`): device='cuda' IS on GPU (21/21 mono + bidomain + LBM tensors
  cuda:0/float64; results on cuda). The "crossover" = **per-iteration host syncs** (mono CN+pcg 24/step, explicit
  0, dct 1; bidomain heaviest; LBM 0), NOT a CPU-compute fallback. GPU per-step latency-bound (~6-10 ms flat), CPU
  scales with dof, crossover ~10k dof. float64 on a 1:64-FP64 card.
- **6-lane adversarial solver audit** (mono diffusion/linear, mono ionic/splitting, bidomain diffusion/splitting,
  bidomain elliptic, LBM, GPU-impl). **Every HIGH/MED finding independently reproduced by me** (ran the agents'
  repros + my own regime sweeps). 2 HIGH silent-wrong (mono Chebyshev-Jacobi 46% err at high diffusion-number;
  bidomain pcg_spectral singular precond on anisotropic mixed-BC), a systemic silent-non-convergence across ALL
  iterative solvers, IMEX-SBDF2 silently 1st-order, RKC refinement-immune ~0.8% err, mono ionic conc-currents use
  post-RL gates (diverges from V5.3, V5.4-lineage; bidomain copy is correct). **LBM clean. Default mono/bidomain
  paths solid.** Full table in KNOWLEDGE "Solver + GPU audit — 2026-07-16".
- **Cross-ref win:** finding #1 (mono Chebyshev) = the 07-02 CODE_AUDIT **M1** — KNOWN + UNFIXED 2 weeks; #9 FFT
  overlaps **M2**; bidomain silent-phi_e relates to **M4**. New this session: systemic framing, IMEX/RKC/mono-ionic,
  all GPU. Argues for landing the shared non-convergence signal, not re-auditing.
- **Decisions:** dedup (Phase 2-5) DEPRIORITIZED to backlog with the user's universal-vs-engine-specific framing +
  the internal-vs-repo-wide split. RK: mono has correct `rk2`/`rk4` (diffusion sub-step only, the GPU-clean 0-sync
  path); bidomain RK-family = the buggy `explicit_rkc`; **LBM has no RK4 and it's a category mismatch** (collide-
  stream, not an ODE march; its upgrade axis is BGK→MRT). Commits 91b52a7 (audit results) + the dedup-backlog note.
**Next**: build the roadmap deliverable (audit state + advanced-features + build path). Then, when the user greenlights
CODE changes: land the shared non-convergence signal (closes the systemic finding + M1 port) as the cheap first fix,
then the advanced features (masked per-step voltage clamp + mid-run state injection via one `_stepping_run` hook).
Still unmerged: `usability-fixes-p0-p1` branch → main (user's call).

### 2026-06-28 Session (cardiac_mcp standardization — audited → blueprinted → executed Tiers 1–3 → merged)
**Worked on**: Took the just-built `cardiac_mcp` server from "working" to "standardized" against the official MCP spec — the user flagged that a working server ≠ a standardized one and wanted its supporting materials audited against proper guidelines, then iterated audit↔revise to convergence, then executed.
**Accomplished**:
- **Audited the supporting materials** via 4 parallel spec-research agents (primitives/metadata · lifecycle/transport/errors · security/authorization · packaging/distribution), verified against the live spec (revision **2025-11-25**) + the installed `mcp` 1.28.0 source — not memory. Surfaced two real path-traversal input-validation bugs (`run_experiment` ran any `run.py`; `commit_experiment` used the unsanitized `date`) plus the annotations / `serverInfo.version` / `outputSchema` / packaging gaps.
- **Blueprinted** a 4-tier PLAN (`/blueprint`), then **audited it to convergence** — 3 adversarial Opus rounds, **12 → 5 low → 0** findings (CONVERGENCE CLEAR), all FastMCP/SDK claims source-verified, every step 9/9 sections. Folded findings via `/blueprint-revise` with mutation-log tracing. Key revisions: **Option B** packaging (extend the root pyproject, not a 2nd editable); **drop `RLIMIT_AS`** (caps virtual AS → aborts torch); a mandatory subprocess-limits test.
- **Executed Tiers 1–3** phase-by-phase, test-gated and committed on branch `mcp-standardization` (6 commits): T1 honest `ToolAnnotations` + `serverInfo.version=0.1.0` + markdown MIME + the two path-traversal guards; T2 typed `TypedDict` returns → `outputSchema`/`structuredContent` + 2 prompts + README + installable `cardiac-mcp` console script; T3 provenance-marker + CPU/FSIZE-limited `run_experiment` + `CARDIAC_MCP_TRANSPORT=http` localhost transport + `REMOTE_DEPLOY.md`. **Phase 4 (registry publishing) SKIPPED** by user.
- **16 cardiac_mcp + 140 cardiac_core tests green; HTTP mode live-verified (406, uvicorn 127.0.0.1).** One execution deviation: `RLIMIT_CPU = timeout_s*ncpu` (multi-threaded torch sums CPU-time → the plan's `≈timeout_s` would false-kill a real run). **Merged `--no-ff` → `main` and pushed to `origin` (`41d17f4`).**
**Next**: optional MCP follow-ups (media tool wrapping `cardiac_core.viz`; more resources/prompts; reentry/restitution recipes; the remote-HTTP auth stack per `REMOTE_DEPLOY.md` when a real deploy target exists; Phase-4 registry publish if public discoverability is wanted — needs a GitHub handle + license). Other engine_consolidation threads in "Next Step" (the `create_cardiac_mesh` chi firewall-bypass, Form-A→B convergence, FEM removal, Surrogate/Optimizer ionic migration).

### 2026-06-26 Session (cardiac-core MCP server — Goal-2 portability layer)
**Worked on**: Explained MCP to the user, then built `cardiac_mcp/` — an MCP server exposing `cardiac_core` to any MCP host (the reach step beyond the Claude-Code-only `/sim-*` skills).
**Accomplished**:
- **`cardiac_mcp/` shipped + validated** (uncommitted on `main`): `core.py` (transport-agnostic logic), `server.py` (FastMCP wrapper), `__main__.py` (stdio), `tests/test_core.py` (10 tests), `.mcp.json` (Claude Code registration). `mcp` SDK 1.28.0 installed in env.
- **Two-track tool surface** (user decision "both, separate tools"): DIRECT `simulate()` (ephemeral CV, coarse-dx fast) + GATED `build_manifest`→`commit_experiment`→`run_experiment` (the `/sim-experiment` gate made STRUCTURAL via a self-signed `experiment_token`; commit refuses w/o `confirmed=True` + intact token). Resources: `cardiac://cheatsheet`, `cardiac://notebook`.
- **Local stdio now, remote-HTTP later** (user decision): all logic in `core.py`; transport swap is one line in `__main__`.
- **Validated**: 10 tests green; server boots (5 tools/2 resources); real stdio client↔server roundtrip (build_manifest + cheatsheet read over the wire). Timing probe sized the defaults (coarse dx=0.02 ~8s, fine dx=0.01 ~38s). See KNOWLEDGE "Goal-2 MCP server".
**Next**: user approves the project server in Claude Code (then tools go live); optionally add a media tool (wraps `cardiac_core.viz`), more resources/prompts, reentry/restitution recipes, and the streamable-HTTP transport + hosting for remote scientists. Commit decision handed back to user (on `main`, uncommitted).

### 2026-06-25 Session (consolidation executed + Goal-2 skill suite shipped — BOTH north-star goals done)
**Worked on**: Made `cardiac_core` a single self-contained package (A2 vendoring), then built the Goal-2 LLM layer — a script-generating skill suite for **wet-lab scientists** (audience reframed from "non-coder conversational builder").
**Accomplished**:
- **Consolidation SHIPPED** (Phases 0–5, commits `935160b`→`37dc381` on `main`): vendored the 3 engines into `cardiac_core/_monodomain/_bidomain/_lbm` + shared `ionic/mesh/stimulus`; **deleted the `_prepare_engine()` sys.modules hack**; no cross-folder imports. Per-engine **bit-identical integrity goldens** (atol=0, captured pre-vendor) + source-hash baseline; 3 engine originals byte-untouched. Backup before vendoring: tag `pre-consolidation-vendoring` + bundle. **137 tests green.** Two real bugs hit+fixed: (1) cross-ref rewrite regex missing `\b` corrupted internal `ionic_time_stepping` imports; (2) naming the package `cardiac_core/monodomain/` SHADOWED the `monodomain()` factory → underscore-prefix the solver packages. See KNOWLEDGE "cardiac_core unified ground-up package — SHIPPED".
- **Goal-2 skill suite SHIPPED** (Phases 1–5, commits `126ff25`→`7635404`): `/sim-experiment` (keystone, manifest + double-check accountability gate) · `/sim-preset` · `/sim-media` · `/sim-notebook`, backed by `cardiac_core/API_CHEATSHEET.md` (anti-hallucination asset, canary-tested) + `cardiac_core/viz.py` (tested). Validated end-to-end with a control/knockdown CV series (59.3 → 41.0 cm/s, eikonal √D). `/audit` of the keystone: 0 critical (gate holds); all 11 findings folded in (notably the slug-overwrite guard + `failed`-status recording). README north-star Goal-2 wording corrected. **140 tests green.**
- Both shipped, committed, **pushed** to `origin/main`. Memory `project-goal2-skill-suite` records the audience reframe.
**Next**: Layer-A `SimulationSpec`/`create_simulation` declarative bridge, OR the programmatic claude-api wrapper, OR the deferred consolidation cleanups (Form-A→B convergence + delete `for_monodomain()`; FEM/`TriangularMesh` removal) — all now easy since the code is in one place.

### 2026-06-02 Session (glossary draft)
**Worked on**: Built the **unified glossary** off the 2026-06-01 capability census, source-verified.
**Accomplished**:
- **`GLOSSARY.md` created** (new artifact for this question). Four parallel read-only census agents (one per surface: Monodomain V5.5, Bidomain V1, LBM V1, `cardiac_core`) harvested every public identifier with `file:line`. Synthesized into a 3-tier doc: Tier 1 universal concepts (one enforced name), Tier 2 engine-specific (canonical where applicable), Tier 3 internal-module rename targets. Each row tagged ✅ aligned / 🟡 minor / 🔴 decision / ⚙️ engine-specific.
- **Census confirmed in source**: `IonicModel` ABC byte-identical across all 3 (the unification proof); `dt`/`Cm`/`(Nx,Ny)` ij/stimulus-accumulate(`+=`)/pacing-helpers(M+B) already aligned. Divergence concentrated in: voltage name, State container, run/result contract + output shape, conductivity input, χ handling, LBM-as-outlier (no grid obj, raw-mask stim, state on free attributes).
- **New evidence surfaced**: `cardiac_core` already shipped **`V`** end-to-end (snapshot/result/analysis + 77 tests) — i.e. the existing wrapper had already (silently) decided the #1 contested name against the IDEALOG `Vm` lean.
- **DECISION (user): voltage = `Vm`** (revokes the CC `V`). Glossary §2 RESOLVED. Migration: rename `V`→`Vm` in M+L, revert CC's `V` + tests, keep read-only `.V` alias then deprecate. Ionic-ABC positional `V` param left as-is (wide-blast rename = separate follow-up).
- **DECISION (user): two naming principles** added to glossary (govern all rows):
  - **P1 — mixed/subscripted notation** for any intra/extra/membrane quantity; bare symbol only when the concept is *identical across engines*. `Vm` qualifies (membrane potential is the same thing in mono+bidomain → bare `V` rejected). Bare `D` does NOT — mono/LBM carry the **effective** diffusivity, bidomain the **components**.
  - **P2 — structured grid is the primary standard** (grid-shaped `(Nx,Ny)`, LBM-simple notation), since almost all our sims are structured; the unstructured/complex-mesh path (FEM, `TriangularMesh`, flat `(n_dof,)`) becomes the explicit *secondary* standard, not the default.
- **Cascaded resolutions from P1/P2**:
  - **#7 output shape RESOLVED → grid `(Nx,Ny)` torch f64** (structured primary; flat reserved for FEM path).
  - **#12 conductivity NAMING RESOLVED → `D_eff` (mono/LBM), `D_i`/`D_e` (bidomain); inputs `sigma_*`** (bare `D` banned; user: "`D_i` is not true for monodomain, `D_eff` is more correct"). Interface (the `ConductivityConfig` class shape) still 🟡.
  - **#13 chi RESOLVED → lives only inside `ConductivityConfig`**, never a free solver knob.
  - **#6 partially resolved**: voltage field name/shape/type now fixed (`Vm`, grid, f64); only delivery style (eager `Result` vs generator vs both) still open.
- **DECISION (user): #6 run/result contract RESOLVED** — `run()` is **eager by default** (returns one `SimulationResult`); streaming folded into the SAME method via **`batch=k`** (yields `Iterator[SimulationResult]` in chunks of ≤k save-points; k=1 = frame-by-frame). User: "rename stream as run(,,batch=x)". Consequences: **no separate `stream()` method, no `Snapshot` type** — a streamed chunk is just a `SimulationResult` with T≤k, so the ONLY public output type is `SimulationResult`. Added `record=("Vm",)` knob (phi_e auto for bidomain, ionic_states opt-in for Surrogate) + kept `callback` for eager early-stop. Accepted wart: return type varies with `batch`.
- **#5 cascaded**: the batch model collapsed the old 3-object view (live State / Snapshot / Result) to **2** — public = `SimulationResult` ONLY (✅ RESOLVED; live mutable state never exposed, killing the yield-the-mutable-object footgun); internal live `State` = recommend one unified dataclass (LBM adopts, drops free attrs) but it's an internal refactor, **deferrable**.
**Still open (🔴)**: #5 internal live-State unification (deferrable), #9 default stim amplitude (−52 vs −80; likely author drift), #12 ConductivityConfig interface shape.
**Toolchain note**: a parallel research session's in-progress hook (`enforce-media-path.py`, project-root-relative path) deadlocked Write/Edit/Bash whenever cwd drifted out of repo root (python exit-2 = block). Recovered by user `cd` back to root. Not this question's bug.
**Next (resume cold)**: settle #6 delivery style + #5 (coupled — the State the generator yields); then #9, #12 interface; then start Goal 1's `Simulation` interface/Protocol + idioms in this vocabulary. Also noted: most divergences trace to 4 root causes — (A) M+B shared lineage vs LBM independent, (B) physics dimensionality (bidomain's extra potential/conductivity), (C) Formulation A/B chi bookkeeping, (D) plain naming drift.

### 2026-06-01 Session (handoff)
**Worked on**: Finished the V5.5 detour + consolidation Phase 1, then pivoted to the north-star (conversational simulation builder) and began designing the unified vocabulary/API — including a full 3-engine capability census.
**Accomplished**:
- **V5.5 Cm-correct fork** — Phases 0–2 done + committed (`ac30af55`→`5171bbce`); exact 1/Cm scaling (3.55e-15), Bidomain cross-check (0.0%/1.1%), Cm=1 golden exact. (Earlier this session.)
- **cardiac_core drift reconciled** + committed (`8f032687`); Engines/ symlink index fixed.
- **Consolidation Phase 1 — COPY-ONLY** + committed (`1f6c72e`): canonical `cardiac_core/ionic/` superset (keyword fix `cell_type_is_endo`→`celltype_is_endo`), lazy `__init__` (engine-free `import cardiac_core.ionic`), `pyproject.toml` + `pip install -e .`. Engine rewire + downstream (Surrogate/Optimizer) migration DEFERRED — audit found big-bang deletion breaks repo-wide consumers. README Phase-1 marked PARTIAL.
- **North-star set** (now the question's main goal in README): Goal 1 unified construction API + Goal 2 self-contained LLM wrapper (skills+docs, strict protocol) → conversational builder for non-coders. Key insight: **spec schema = the intake questionnaire**. Build order REFRAMED: **vocabulary first** (a ubiquitous language across the 3 engines), **then** the unified API (interface/Protocol + idioms).
- **3-engine capability census** run (read-only Explore agents) + synthesized into the cross-engine comparison (see KNOWLEDGE "Cross-engine capability census"). Found: ionic ABC + physical conventions + stimulus `+=` already aligned; divergence concentrated in construction, voltage naming (V/Vm), state, and the run/result contract; LBM is the consistent outlier.
**Next (resume cold)**: build the **glossary** off the census — settle the highest-leverage divergences first: (1) voltage `V` vs `Vm` [lean `Vm`], (2) the `State` concept (dataclass vs LBM on-object), (3) the `run()`/result contract (generator vs `(times,V_history)` + flat-vs-grid output). Then the rest of the universal-tier vocabulary, then Goal 1's interface/idioms. Geometry input (Fiji→Builder) and the Optimizer downstream migration both remain DEFERRED.

### 2026-05-30 Session
**Worked on**: Reasoned through the chi/Cm audit (why chi is safe but Cm is the troublemaker — Cm couples to both operator-split halves, chi to only diffusion); decided the V5.5 detour; blueprinted it; ran two adversarial audit passes (11 + 4 findings, all applied); executed Phases 0–2.
**Accomplished**:
- **Engine_V5.5** forked from V5.4 (Phase 0): faithful clone, dead internal LBM path removed (zero importers; boundary work uses LBM/Engine_V1), Cm=1 regression golden captured (`_regression/`, max|dV|=0).
- **Cm fix** (Phase 1): `SimulationState.Cm` plumbed from `spatial.Cm` (fail-loud, no getattr fallback); reaction divides by Cm in rush_larsen + forward_euler; FEM `_Cm`/`_chi` storage added (audit-CRITICAL — FEM only baked them into `self.M`). Cm=1 stays bit-identical; FDM/FEM/FVM all expose `.Cm`.
- **Validation** (Phase 2): `test_phase10_cm_scaling.py` 3/3 — exact 1/Cm reaction scaling to 3.55e-15; Cm-direction; Bidomain V1 cross-check (CV 54.35 vs 54.35 cm/s @Cm=1, 28.09 vs 27.77 @Cm=2).
- **Physics correction**: the Cm time-dilation invariant (assumed by the plan AND both audits) is FALSE — gate kinetics/concentrations carry no Cm, so Cm changes AP morphology, not timescale (APD 218→292 ms, not 2×). CV~1/Cm holds by eikonal scaling, not dilation. Caught empirically by the 0D test. The fix was correct throughout; only the validation strategy was wrong.
- 4 commits on `main` (`ac30af55`→`5171bbce`) + plan archived; README/KNOWLEDGE updated.
**Next**: Consolidation Phase 1 — move ionic models into `cardiac_core/ionic/` (build against V5.5). First reconcile the live `cardiac_core/` drift (added geometry/io/analysis/run; `Engines/lbm_v1` symlink deleted).

### 2026-07-01 Session — cardiac_core+mcp audit → 3-phase cleanup → LBM boundary modes SHIPPED
**Worked on**: Tidy the cardiac_core+cardiac_mcp foundation (user: "no bug or loophole… everything tidied up") BEFORE productizing the LBM boundary modes. Ran an 8-lane adversarial multi-agent audit, blueprinted a 3-phase plan, hardened it audit-to-convergence, executed all 3 phases, committed + pushed.
**Accomplished**:
- **Audit** (64 agents, ~3.1M tok): 46 confirmed (4 HIGH), 9 refuted, 12 gaps → [CARDIAC_CORE_AUDIT.md](./CARDIAC_CORE_AUDIT.md). Plus a cross-engine boundary-handling audit → [BC_IMPLEMENTATION_AUDIT.md](../boundary_conduction_speedup/BC_IMPLEMENTATION_AUDIT.md). Plan audited to convergence (R1 5blk/6maj → R2 2/2 → R3 SOUND).
- **P1** (`a3915d1`): mono FDM anisotropic cross-derivative BUG fixed (sign+½-magnitude; the audit's standout — my own boundary audit missed it) in cardiac_core+V5.5, pinned by `test_fdm_anisotropy`; chi/D=RAW unified — the depth: the DECLARATIVE `_build_mesh_data` LBM branch stores already-effective D, so the fix stores raw there too (no double-divide, Cm-safe); default D=1.4 (old default was conduction-BLOCKED); ionic-override replay; MCP path-traversal. Goldens regenerated (mono golden had captured the blocked sim; bidomain/lbm unchanged since they already ran at 1e-3).
- **P2** (`945350f`): FEM/TriangularMesh + orphaned fft.py/d2q5-mrt deleted; pcg_gmg warn; zero-save guard; positional-mesh clash; NOTEBOOK escape; stale docstrings.
- **P3** (`736296d` + terminology fix): `cardiac_core.lbm(boundary=, alpha=)` — hbb / specular_nextcell (NCS) / specular_samecell (SCS) / combined-α; kernels lifted from `diag_lbm_specular.py`; default neumann bit-identical → goldens safe; 17 boundary tests. User-corrected terminology: next-cell/same-cell specular, 'ncs'/'scs' aliases.
- Suite **148/1 → 196 passed / 0 failed**. 5 commits, pushed to `origin/engine-tuner-cardiac-core`.
- 2 own-test bugs surfaced real LBM subtleties (precompute_bounce_masks → all-False on a full periodic domain; modes only differ off-equilibrium).
**Next**: deferred backlog only (see Next Step). Nothing blocking. Candidate next real work: Form-A→B convergence, or surface mono `boundary_mode`/`stencil` through `cardiac_core.monodomain()` to match the LBM productization.

### 2026-07-15→16 Session — API failure-mode + two-round usability audit + fix blueprint
**Worked on**: (1) an API failure-mode check → F1/F2 fixes; (2) merged the `engine-tuner-v2-joint` branch (12 commits) to `main`; (3) F3/F4 cosmetic nits; (4) a **task-based usability audit** — round 1 (24 tasks, light) then round 2 (30 new tasks + full-scale re-run of the 24, **actually solved & run to completion** via 10 parallel agents); (5) a machine-targeted **PLAN.md** for the fixes, audit-revised to convergence.
**Accomplished**:
- **F1** empty-run analysis hooks no longer crash (rank-3 `(0,Nx,Ny)` + T=0 guards); **F2** `hbb`→D2Q9-only + lattice-aware LBM boundary default (d2q5/neumann kept — tuner/goldens safe); **F3** `point_distance(center=)`; **F4** cheatsheet scalar/3-tuple note. Commits `2938cf9`/`e707fe1`/`2d241af`. A clean adversarial correctness audit (0 crit/high/med) + inline hardening of 2 LOWs.
- **Tuner→main**: merge `9d82f56` (resolved 1 MASTER_KNOWLEDGE_INDEX conflict); working tree left untouched.
- **Usability audit** (report [API_USABILITY_AUDIT_2026-07-15.md](./API_USABILITY_AUDIT_2026-07-15.md), commits `09ee644` R1, `04611ed` R2): verdict "possible but painful," mean ease ≈2.7/5; 2 tasks impossible; **13 concrete bugs (B1–B13)** — B1 GPU crash-all-analysis, B2 broken fft/dct fast path, B3/B4 apd_at, B8 masked-node 23% CV error, + a fixed per-step runtime wall. Full running FLIPPED 2 verdicts up (paci automaticity; per-node-D scar) and several down.
- **PLAN.md** (`9e6a0e7`) audit-converged (inline — see caveat): 5 phases, test-gated, golden-guarded; P3/P4 as future work.
**Next**: **EXECUTE PLAN.md Phase 1** (see the ▶ HANDOVER at the top of Next Step). Optional: independent-subagent audit of PLAN.md after the session rate limit resets (~4:50am ET).
