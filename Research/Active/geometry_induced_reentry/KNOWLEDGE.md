# Geometry-Induced Reentry — Knowledge File

> This file is a running synthesis. Updated as findings accumulate.
> When the question is complete, a copy is promoted to `Research/Knowledge/`.

## Current Understanding
The setup is fixed: a **uniform planar wave** propagating across a 2-D sheet meets an **inexcitable infarct of simple shape** (circle first), modeled with a **no-flux (Neumann) boundary** — bounce-back in LBM. The wave diffracts around the obstacle; the question is whether the diffracted branches heal (no arrhythmia) or break into rotating wavefronts behind the obstacle (reentry). **Twofold literature search done 2026-06-24** — see `## Literature Synthesis` below; full indexes in `literature/{engineered_tissue_reentry,parker_lab,hipsc_wavelength_geometry}/INDEX.md`.

The classical physics (to be verified against literature, then promoted here):
- A wavefront passing an obstacle develops high curvature at the obstacle's downstream "shoulder." By the eikonal relation `θ = θ₀ − D·κ` (curvature κ), a highly convex front slows; beyond a critical curvature it detaches (wavebreak) instead of wrapping cleanly around the corner.
- Whether the broken tips curl into a sustained rotor depends on obstacle size relative to the **wavelength** (`λ = CV · APD`) and the **refractory tail**: too small an obstacle → branches reconnect immediately; large enough → a free tip survives long enough to rotate. This is the "vortex shedding" picture (Cabo 1996).
- Sharp corners concentrate curvature and source-sink mismatch more than smooth ones — shape, not just size, should matter.

## Key Decisions
- **Infarct boundary condition = Neumann no-flux (bounce-back), not Dirichlet.** Settled by `scar_bc_validity` (Complete): scar tissue is electrically inert; Dirichlet voltage-clamping is unphysical and injects spurious current. *Do not relitigate.*
- **Planar stimulus, not point source** — isolates obstacle geometry from source curvature.
- **Simple 2-D obstacle, fully inexcitable, initial scope.** Partial-thickness / gray-zone infarcts (graded excitability) are deferred to the thickness-weighted operator developed in `source_sink_mismatch_investigation`.
- **LBM primary; cardiac_core the intended cross-engine driver** once consolidation Phase 1+ lands.
- **Report results on a `d/λ` axis** (obstacle size / wavelength) — λ = CV·APD is the only intrinsic length scale; absolute mm are engine-dependent.
- **Obstacle-diameter sweep** (from the hiPSC-CM wavelength data): operate at CV ≈ 20–25 cm/s, APD ≈ 150–250 ms → λ ≈ 1.5–4 cm; sweep **d/λ ∈ {0.1, 0.3, 0.5, 1, 1.5, 2, 3}**. Expect: heal below d/λ ≈ 0.3; wavebreak/vortex-shedding window d/λ ≈ 0.3–1 (resolve finely); stable anchored reentry d/λ ≳ 1.5. Confirm the transition tracks λ by re-running at a second pacing rate.

## Open Questions
- What is the critical obstacle-size / wavelength ratio for wavebreak in our LBM at our CV/APD?
- How sharp does a corner need to be before shape (vs size) dominates inducibility?
- Transient wavebreak vs sustained rotor — what separates self-termination from a stable tip?
- Does the LBM bounce-back obstacle reproduce the same threshold as a Neumann obstacle in V5.4/Bidomain (numerics check)?
- Is a single planar wave enough, or is a premature S2 / restitution-driven wavelength shortening needed to actually induce sustained reentry?

## Literature Synthesis (2026-06-24)

### The "laser circular ablation → reentry" paper: identified (with a caveat)
The remembered experiment — *a custom/engineered heart where a circular ablation makes a passing wave reenter* — is almost certainly **MacQueen et al. 2018, "A tissue-engineered scale model of the heart ventricle," Nat Biomed Eng 2:930–941** (DOI 10.1038/s41551-018-0271-5, PMC6774355, **open access**; PDF in `papers/macqueen_2018_tissue_engineered_ventricle.pdf`). **Parker lab (Disease Biophysics Group).** They punched **1-mm circular holes** (inexcitable voids) into engineered NRVM ventricular tissue, and a **plane wave became a spiral pinned to the hole**; *two holes 5 mm apart → counter-propagating spirals*; tissue regrowth filling the hole **abolished** pinning. They did **not** quantify an obstacle-size-vs-wavelength threshold. Caveat: the obstacle was a **punched hole, not a laser ablation** — the search found *no* 2-D cardiac paper that makes the obstacle specifically by laser and gets reentry (the lone laser-disc-in-hiPSC-monolayer paper, Saraithong 2025 Comm Biol, induces no reentry). The lab is likely conflating the fabrication method; the *phenomenon* is MacQueen 2018. SI movies S12/S13/S16/S17 (optical-mapping spiral clips) are on PMC (URLs in `parker_lab/INDEX.md`; behind an anti-bot portal — fetch via browser).

### Core experimental prior art (2-D, simple obstacle → reentry)
| Paper | Prep | Obstacle | Result | Access |
|---|---|---|---|---|
| **Lim, …, Tung 2006** Circulation 114:2113 | NRVM monolayer | circular hole **0.6–2.6 mm** | spiral *anchors* to obstacle ≥0.6 mm; attachment prob ↑ with size | paywalled (DOI 10.1161/CIRCULATIONAHA.105.598631) |
| **Cysyk & Tung 2008** Biophys J | NRVM monolayer | 2–4 mm hole | field stim pins/unpins spiral | OA, `papers/cysyk_2008_*.pdf` |
| **Kadota et al. 2012** | NRVM monolayer | needle-cut corner/isthmus | front detaches → spiral **only under reduced excitability (lidocaine)**; core grows with refractoriness | PMC free |
| **Cabo, Pertsov, …, Jalife 1996** Biophys J | sheep epicardium (2-D) | sharp-edged obstacle | **vortex shedding** at edge under reduced excitability / fast pacing | OA, `papers/cabo_1996_vortex_shedding.pdf` |
| **MacQueen 2018** (above) | engineered NRVM + hiPSC-CM | 1-mm punched hole | plane wave → pinned spiral | OA, in `papers/` |

Note: most "anchoring" papers *pin a pre-existing spiral*; **breaking a genuinely planar wave into a new rotor** at a simple obstacle is rarer and needs reduced excitability or fast forcing (Cabo, Kadota, Agladze).

### Theory / modelling counterparts (the threshold law)
- **Agladze, Keener, Müller & Panfilov 1994** (Science) — sharp corner + high-frequency forcing → spiral creation by geometry.
- **Panfilov & Keener 1995** — minimum obstacle size for spiral formation (~2.5 cm in FHN; *decreases as excitability drops*).
- **Fenton & Cherry 2002** (Chaos; `papers/fenton-cherry-2002-*.pdf`) — circus movement sustains only when **obstacle perimeter ≳ wavelength**.
- **Pandit–Jalife** curvature/eikonal companion — a *planar* front breaks at an obstacle when edge radius ≥ R_crit (critical curvature, `θ = θ₀ − Dκ`).

### Quantitative anchor — wavelength & the critical-size rule
λ = CV · APD (≈ CV · refractory period). Measured 2-D values:
| Tissue | CV (cm/s) | APD (ms) | λ | Source |
|---|---|---|---|---|
| hiPSC-CM monolayer (optimal ECM) | **43.6 ± 7.0** | — | — | Herron 2016, CircEP |
| hiPSC-CM monolayer (glass) | ~22 | ~250→400 | ~2–8 cm (spontaneous) | Herron 2016; Slotvitsky/Agladze 2019 |
| hiPSC-CM engineered patch | ~25–29 | **APD80 424–471** | — | Shadrin/Bursac 2017 (`papers/shadrin-2017-*.pdf`) |
| NRVM monolayer (reentry baseline) | **12.9** | **APD80 118** | **≈ 1.5 cm** | Iravanian 2003 |
| Parker DBG ventricle | NRVM **9.33**, hiPSC-CM **5.2** | (APD not reported) | — | MacQueen 2018 |
| Bursac & Parker 2002 anisotropic monolayer | CV_L 23.5–37.2, CV_T 9.2–18.1 | — | — | Circ Res |

**Critical-size rule** (theory ↔ experiment converge): circus reentry sustains when **obstacle perimeter ≥ λ**, i.e. (inference, not a quoted constant) **d_crit ≈ λ/π ≈ 0.32 λ**. Experimentally spirals pin to obstacles ≥0.6 mm (Lim/Tung). Maturity caveat: hiPSC-CMs are immature → slow CV + long/variable APD → large spontaneous λ (2–8 cm), shrinking to ~1–2 cm only during fast activity/reentry. (Vaidya 1999: paced λ *overestimates* the minimum size — don't truncate the sweep below d/λ ≈ 0.2.)

### The gap this question fills (the niche)
No single 2-D engineered/cultured preparation has done a **controlled sweep of obstacle BORDER SHAPE** (circle vs square vs slab vs sharp corner) at **fixed area and fixed wavelength**, nor mapped the full **heal → wavebreak → anchor transition as a function of d/λ** at controlled CV/APD. Vortex shedding from a planar wave has been shown only in sheep epicardium (Cabo 1996), never replicated in a cultured monolayer with simple geometry. That — plus the source-sink mismatch reading of the obstacle "shoulder" — is precisely this question's contribution.

## Parameter-Fitting Strategy — shared cross-plan with `ionic_model_optimization` (2026-06-29)

We ground the campaign in a **real preparation — Kit Parker's tissue chip — and simulate the entire chip.** The fit splits into two independent problems:
- **(a) Discretization** (dx, domain L, obstacle in cells): a *meshing* decision from the chip footprint + a points-per-λ rule. **Not** the Engine Tuner's job.
- **(b) Physics** (D/σ/τ + ionic conductances → CV, APD, restitution, hence λ): **exactly** the Engine Tuner's job (`Optimizer/V1`).

They couple via λ (dx must resolve the tuned λ) and, in LBM, via `D = cs²·(τ−0.5)·dx²/dt` — at fixed chip dx + dt, **τ is the diffusion knob** (`LBM/Engine_V1/src/diffusion.py`: `sigma_to_D`/`tau_from_D`).

**Chip-fitted mesh (decision):** L = 16 mm square inside the 25 mm Parker coverslip; dx = 0.1 mm (161² grid, ~26k nodes). At λ ≈ 1.5–4 cm → ~150–400 pts/λ; a 1 mm obstacle = 10 cells across.

**This is a cross-plan.** The *build* is owned by `ionic_model_optimization` (the Engine Tuner question); this question is the *application* that drives it. **End goal: a finished Engine Tuner adapted to `cardiac_core`, spanning monodomain + bidomain + LBM.** The tuner Q already lists "Cross-engine validation (V5.4 vs Bidomain vs LBM)" as open and holds the V2 adapter design (`Optimizer/improvement.md`). The canonical PLAN lives with the tuner question; this question links to it.

**Engine Tuner (`Optimizer/V1`) — state:** implemented (~2.5k LOC, 6 test files; README still says "Planning"). 4 phases: (1) qNEHVI on ionic conductances → AP + restitution; (2) qEI on D → CV (warm-start D ∝ CV²); (3) PCA+GP emulator → NSGA-II joint refine; (4) validate. Works on **monodomain + bidomain** today.

**Limitations for this question:**
1. **No LBM adapter** (deferred to V2) — LBM is our primary engine, and LBM CV runs ~35% higher than FDM for the same D, so a V5.4-tuned set needs LBM recalibration or a native LBM fit.
2. **Ionic model** — solved on the tuner side: **MHAS13** is a quiescent *mature hiPSC-CM* model (Paci 2013 → PHAS13 → MHAS13; V_rest −83.7 mV, APD 349 ms). Use MHAS13, not adult TTP06. (See `hipsc_cm_ionic_models`.)
3. **Parker reports CV only** — no APD/λ. Supply CV from Parker (NRVM 9.3 / hiPSC 5.2 cm/s) + APD/restitution from broader hiPSC lit (Herron, Shadrin APD80 424–471; Paci ~350 ms).

**Targets (decision):** fit **both** NRVM (CV 9.3) and hiPSC-CM (CV 5.2) baselines; pick the reentry baseline after seeing which yields a tractable on-chip wavebreak window.

**λ caveat:** MacQueen's "~5 mm plane-wave spatial period" is NOT λ — λ = CV·APD ≈ 1.5–4 cm; the 5 mm is a small-geometry artifact. Fit/sweep against λ = CV·APD (see IDEALOG → Failed Approaches).

## Connections
- **Engines**: LBM V1 (primary), cardiac_core (intended driver), Monodomain V5.4 + Bidomain V1 (cross-validation), Builder (geometry authoring).
- **Related research**: `scar_bc_validity` (infarct BC — foundational), `source_sink_mismatch_investigation` (functional block / source-sink at obstacle shoulder), `boundary_conduction_speedup` (Kléber boundary physics), `geometry_induced_pacemaking` (sibling), `engine_consolidation` (cardiac_core driver).
- **Pipelines**: Builder (mesh + stimulus authoring).
