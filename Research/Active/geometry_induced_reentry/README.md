# Geometry-Induced Reentry

## Question
When a **uniform (planar) propagating wave** encounters an **inexcitable infarct of simple 2-D geometry** (circle, square, ellipse, slab — a non-conducting scar with no-flux boundaries), under what geometric conditions does the interaction break the wave and seed **self-sustaining reentry** (vortex shedding / rotor formation), rather than the wave simply diffracting around the obstacle and healing?

## Status: Active (created 2026-06-24)

## Why It Matters
Reentry around an anatomical obstacle is the canonical mechanism of post-infarction ventricular tachycardia. The open scientific question here is how much of that is **geometry alone**: given a planar wave and an inert scar, the obstacle's *shape and size relative to the wavelength/refractory tail* determine whether the diffracting wave branches reconnect cleanly or break into rotating wavefronts behind the obstacle. This is the arrhythmogenic payoff of the lab's geometry program — it connects the Kléber boundary-speedup physics (`boundary_conduction_speedup`), the source-sink functional-block mechanism (`source_sink_mismatch_investigation`), and the correct infarct boundary condition (`scar_bc_validity`) to a clinically relevant arrhythmia endpoint. It is the arrhythmia-mechanism sibling of `geometry_induced_pacemaking`.

## Engines
This question targets **all engines**, with LBM as the primary workhorse and `cardiac_core` as the intended driver once its consolidation is complete.
- **LBM V1**: *Primary.* Boundary/geometry effects are its research focus; bounce-back BC implements the no-flux infarct correctly (see `scar_bc_validity`). Vortex-shedding studies want fine, cheap 2-D timesteps — LBM's strength.
- **cardiac_core**: *Intended driver.* Once engine consolidation finishes, drive the campaign through the unified `lbm()`/`monodomain()`/`bidomain()` API + shared mesh format so the same infarct geometry runs on every engine.
- **Monodomain V5.4**: Cross-validation; reaction-diffusion ground truth, and the thickness-weighted operator if partial-thickness (gray-zone) infarcts enter scope.
- **Bidomain V1**: Full bidomain validation; correct asymmetric BCs at any tissue-bath interface vs the symmetric Neumann at the scar.
- **Builder**: PNG/SVG → StructuredGrid to author the simple 2-D infarct geometries (circle, square, sharp corners) and the planar stimulus.

## Completion Criteria
- [ ] **Baseline**: a clean planar wave propagates across a homogeneous 2-D sheet in LBM (no obstacle), with measured CV and wavelength.
- [ ] **Diffraction (non-arrhythmic)**: insert a simple inexcitable infarct (circle) with no-flux/bounce-back BC; confirm the wave diffracts around it and the two branches heal — the *control* case, no reentry.
- [ ] **Wavebreak / vortex shedding**: identify a parameter regime where the diffracting branches fail to reconnect and curl into rotating wavefronts behind the obstacle.
- [ ] **Critical geometry**: characterize the obstacle size / wavelength (and obstacle size / refractory-tail length) threshold for wavebreak; relate to wavefront curvature and source-sink at the obstacle "shoulder".
- [ ] **Shape dependence**: vary infarct geometry (circle vs square vs sharp-cornered) and quantify how shape changes reentry inducibility.
- [ ] **Self-sustaining vs transient**: distinguish a transient broken wave that self-terminates from a sustained rotor (≥ N rotations / stable tip trajectory).
- [ ] **Cross-engine**: reproduce the inducibility threshold on at least one second engine via `cardiac_core` to rule out LBM-specific numerics.

## Sub-Questions

| Sub-Question | Status | Key Finding |
|-------------|--------|-------------|
| — | — | — |

## Key Findings So Far
*(Literature only — no simulations yet. Full synthesis in [KNOWLEDGE.md](KNOWLEDGE.md); twofold search indexes in `literature/*/INDEX.md`.)*
- **The remembered "laser circular ablation → reentry" experiment = MacQueen et al. 2018 (Parker lab), *Nat Biomed Eng***: 1-mm circular *punched holes* in engineered NRVM ventricle turn a plane wave into a pinned spiral. Caveat — obstacle was a punched hole, **not** a laser ablation; no 2-D laser-ablation→reentry paper exists. They did **not** quantify an obstacle-size/wavelength threshold.
- **Critical-size law** (theory ↔ experiment): circus reentry sustains when **obstacle perimeter ≳ wavelength** (Fenton-Cherry 2002), i.e. d_crit ≈ λ/π ≈ 0.32 λ; spirals pin to obstacles ≥0.6 mm (Lim/Tung 2006). Breaking a *planar* wave (vs pinning an existing spiral) needs reduced excitability / fast pacing (Cabo 1996 vortex shedding; Kadota 2012; Agladze 1994).
- **hiPSC-CM wavelength** ≈ 1.5–4 cm at relevant rates (CV ~20–44 cm/s, APD ~250–470 ms; immature → slow CV, long/variable APD). → simulate at λ ≈ 1.5–4 cm and **sweep d/λ ∈ {0.1…3}**.
- **The niche/gap**: nobody has swept obstacle **border shape** at fixed area & λ, nor mapped the full heal→wavebreak→anchor transition vs d/λ in a 2-D prep. Vortex shedding never replicated in a cultured monolayer.

## Scope Notes (mechanism, fixed 2026-06-24)
- **Stimulus**: a uniform *planar* wavefront (not a point source) — we study the wave–obstacle *interaction*, isolating geometry from source curvature.
- **Infarct model**: a *simple* 2-D shape (circle first), fully inexcitable, **no-flux (Neumann) boundary** — bounce-back in LBM. This BC choice is settled by `scar_bc_validity` (Dirichlet at scar is unphysical). Partial-thickness / gray-zone infarcts are out of initial scope (defer to the thickness-weighted operator from `source_sink_mismatch_investigation`).
- **Endpoint**: does the interaction *induce reentry* — i.e. wavebreak → rotor — and what geometry makes it happen.

## Literature
Twofold exhaustive search done 2026-06-24 — indexes: [`literature/LITERATURE_SEARCH.md`](literature/LITERATURE_SEARCH.md) (framing), [`engineered_tissue_reentry/INDEX.md`](literature/engineered_tissue_reentry/INDEX.md), [`parker_lab/INDEX.md`](literature/parker_lab/INDEX.md), [`hipsc_wavelength_geometry/INDEX.md`](literature/hipsc_wavelength_geometry/INDEX.md). 10 OA PDFs in `papers/`.

| Paper | Where | Key Insight |
|-------|-------|-------------|
| MacQueen et al. 2018 (*Nat Biomed Eng*) | `papers/macqueen_2018_*.pdf` | **Closest prior art**: 1-mm punched holes in engineered ventricle → plane wave pins to a spiral (Parker lab); size/λ threshold NOT quantified |
| Lim …, Tung 2006 (*Circulation*) | `engineered_tissue_reentry/INDEX.md` | NRVM monolayer; spiral anchors to circular obstacle ≥0.6 mm; prob ↑ with size (paywalled) |
| Cabo …, Jalife 1996 (*Biophys J*) | `papers/cabo_1996_vortex_shedding.pdf` | Vortex shedding at a sharp obstacle edge under reduced excitability — the planar-wave→rotor mechanism |
| Kadota et al. 2012 | `engineered_tissue_reentry/INDEX.md` | NRVM corner/isthmus → front detaches → spiral only with reduced excitability; core grows with refractoriness |
| Agladze, Keener, Müller, Panfilov 1994 (*Science*) | `engineered_tissue_reentry/INDEX.md` | Spiral creation by obstacle geometry (sharp corner + fast forcing) |
| Fenton & Cherry 2002 (*Chaos*) | `papers/fenton-cherry-2002-*.pdf` | Circus reentry sustains when obstacle perimeter ≳ wavelength |
| Herron 2016; Shadrin/Bursac 2017; Iravanian 2003 | `hipsc_wavelength_geometry/INDEX.md`, `papers/` | CV/APD/wavelength numbers for hiPSC-CM & NRVM tissue |
| *Re-read through reentry lens*: Fast & Kléber 1995, Gonzalez-Rajal 2018, Zemlin 2018 | `../geometry_induced_pacemaking/literature/` | expansion block / geometry→dynamics / curvature ectopy |

## Connected Research
- **scar_bc_validity** (Complete) — *foundational*: the infarct must use Neumann no-flux (bounce-back), not Dirichlet. Settles the obstacle BC for this whole question.
- **source_sink_mismatch_investigation** — functional-block lines and source-sink mismatch at the obstacle shoulder; "prerequisite to any geometry-driven arrhythmia study." This question is that study.
- **boundary_conduction_speedup** — the Kléber boundary/source-sink physics underlying wavefront behavior at the obstacle edge.
- **geometry_induced_pacemaking** — sibling: same source-sink geometry principle applied to pacemaking rather than reentry.
- **engine_consolidation** — `cardiac_core` is the intended driver once Phase 1+ lands.
- **ionic_model_optimization** — *shared cross-plan*: the Engine Tuner fits this question's Kit Parker chip-EP parameter set (both NRVM + hiPSC-CM) across monodomain/bidomain/LBM via cardiac_core. Plan: [`../ionic_model_optimization/PLAN.md`](../ionic_model_optimization/PLAN.md). This question is the *application*; that one owns the *build*.

## Experiments
| Experiment | Engine | Result | Location |
|-----------|--------|--------|----------|
| — | — | — | — |

## Engine References
| File | What it tells you |
|------|-------------------|
| `LBM/Engine_V1/src/` | LBM path; bounce-back BC = no-flux infarct boundary |
| `LBM/Engine_V1/` | Primary engine; D2Q5/D2Q9, BGK/MRT |
| `cardiac_core/` | Unified `lbm()`/`monodomain()`/`bidomain()` API + shared mesh format (intended driver) |
| `Builder/` | PNG/SVG → StructuredGrid for infarct geometry + planar stimulus |
| `Research/Complete/scar_bc_validity/` | Why the infarct BC is Neumann no-flux |

## Future Work
{No deferred items yet.}
