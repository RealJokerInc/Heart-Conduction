# Geometry-Induced Reentry — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan this in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
Brand-new question (2026-06-24), spun up as the arrhythmia-mechanism sibling to `geometry_induced_pacemaking`. Scope fixed at creation: a **uniform planar wave hitting a simple 2-D inexcitable infarct** (no-flux boundary), asking under what geometry the wave breaks into self-sustaining reentry. LBM is the primary engine; `cardiac_core` becomes the driver once consolidation finishes.

**Update 2026-06-29 — parameter fitting via the Engine Tuner.** Decided to ground the whole campaign in a **real preparation: Kit Parker's tissue chip**, and simulate the *entire chip*. This splits into (a) a meshing decision — chip footprint (25 mm coverslip → 16 mm tissue) + points-per-λ → dx = 0.1 mm — and (b) a physics-parameter fit (D/σ/τ + ionic conductances → target CV/APD/λ), which is exactly the **Engine Tuner's** (`Optimizer/V1`) job. The Tuner does NOT fit dx (that's meshing); it fits the physics on whatever mesh we give it.

## Next Step
`/blueprint` the **shared cross-plan** "Engine Tuner V2 → cardiac_core (monodomain + bidomain + LBM)" — co-owned with `ionic_model_optimization` (the build) and driven by this question (the application: fit a Kit Parker chip EP set, then run reentry sweeps). Plan only, no coding (user gate, 2026-06-29). Canonical PLAN lives with the engine-tuner question; this question links to it.

## Thread
- 2026-06-24: Created. Confirmed not a duplicate — distinct from `geometry_induced_pacemaking` (pacing, not reentry), `source_sink_mismatch_investigation` (thickness-driven curvature/block, which is the *prerequisite* this study builds on), and `scar_bc_validity` (settles the infarct BC). The mechanism choice — planar wave + simple obstacle → wavebreak — is deliberately narrower than "any geometry → reentry"; keeps the first campaign tractable.
- Settled at creation: infarct BC is Neumann no-flux / bounce-back (from `scar_bc_validity`); planar (not point) stimulus; fully-inexcitable simple obstacle; partial-thickness deferred to the thickness-weighted operator.

## Failed Approaches
- **Do NOT treat MacQueen's "~5 mm plane-wave spatial period" as the wavelength λ.** λ = CV·APD = 0.093 mm/ms × ~300 ms ≈ 28 mm for NRVM; the 5 mm is a small-geometry / wave-train artifact on a 9 mm ventricle. Fit and sweep against **λ = CV·APD ≈ 1.5–4 cm** instead. Implication: MacQueen's 1 mm hole was d/λ ≈ 0.03–0.07 — a tiny obstacle pinning a fast spontaneous rotor, not a de-novo planar wavebreak.
- **Do NOT tune adult TTP06 down to immature-chip CV** ("fighting the adult phenotype"). The engine-tuner question already built **MHAS13** (quiescent mature hiPSC-CM); use it as the ionic target.

## Session Log
- **2026-06-24** — Question scaffolded via `/research-new`. README + KNOWLEDGE + IDEALOG created, MASTER.md updated. No code or experiments yet.
- **2026-06-24** — Twofold exhaustive lit search (3 parallel agents → `literature/{engineered_tissue_reentry,parker_lab,hipsc_wavelength_geometry}/INDEX.md`, 10 OA PDFs in `papers/`). Headlines:
  - **The "laser circular ablation → reentry" paper resolved**: it's **MacQueen 2018 (Parker lab, Nat Biomed Eng)** — 1-mm *punched holes* (not laser), plane wave → pinned spiral. No 2-D *laser*-ablation→reentry paper exists; user likely conflated fabrication method. Phenomenon confirmed, threshold NOT quantified by them (= our gap).
  - **Threshold law**: obstacle perimeter ≳ λ → d_crit ≈ λ/π ≈ 0.32λ (Fenton-Cherry); pinning ≥0.6 mm (Lim/Tung). Breaking a *planar* wave (vs pinning existing spiral) needs reduced excitability/fast pacing (Cabo vortex shedding, Kadota, Agladze).
  - **hiPSC-CM λ ≈ 1.5–4 cm** → sim plan: CV 20–25, APD 150–250, **sweep d/λ ∈ {0.1…3}**, report on d/λ axis, confirm transition tracks λ via 2nd pacing rate.
  - **Gap/niche**: no controlled obstacle-SHAPE sweep at fixed area & λ; no full heal→break→anchor map vs d/λ in a 2-D prep; vortex shedding never replicated in a cultured monolayer. Source-sink reading of the obstacle shoulder is ours to add.
  - **Caveat logged**: SI optical-mapping movies (MacQueen S12/S13/S16/S17; Lee 2022) are behind anti-bot SI portals — recorded URLs, not auto-downloaded. "J. Parker" in de Diego 2010 = John Parker (UCLA), NOT Kit Parker — don't misfile.
- **2026-06-29** — Pivoted to **parameter fitting grounded in a real prep (Kit Parker tissue chip)**, done via the **Engine Tuner** (`Optimizer/V1`); reframed as a **cross-plan** with `ionic_model_optimization`.
  - **Chip size** (web + local PDFs): Parker "Heart on a Chip" = **25 mm circular coverslip**; MTF tissue films ~1–3.5 mm × 4 mm; MacQueen 2018 ventricle (the reentry prep) = half-ellipsoid a=b=4.5/c=9 mm, **1 mm punch holes 5 mm apart**, CV **9.33 cm/s NRVM / 5.2 hiPSC-CM**. → simulate the whole chip: **L = 16 mm, dx = 0.1 mm** (161² grid).
  - **Engine Tuner read** (`Optimizer/V1`): 4-phase (cell BayesOpt qNEHVI → tissue CV qEI → GP+NSGA-II joint → validate). Implemented (~2.5k LOC + tests) for **monodomain + bidomain**; README stale ("Planning"). Fits *physics* params (D/σ/τ + conductances), NOT dx (that's meshing).
  - **Limitations vs our job**: (1) **no LBM adapter** (V2-deferred) though LBM is our primary engine, and LBM CV ~35% > FDM; (2) ionic target = **MHAS13** (mature quiescent hiPSC-CM, V_rest −83.7, APD 349) — already built by the tuner Q, so no TTP06/Paci detour; (3) Parker reports CV only → APD/restitution from broader hiPSC lit.
  - **Cross-plan reframe (user)**: shared with `ionic_model_optimization`. That Q already lists "Cross-engine validation (V5.4 vs Bidomain vs LBM)" open + holds `Optimizer/improvement.md` (V2 multi-engine adapter). **End goal = finished Engine Tuner adapted to `cardiac_core` (monodomain + bidomain + LBM)**; this question is the first application (chip-EP fit → reentry).
  - **Targets**: fit **both** NRVM (9.3) and hiPSC-CM (5.2); choose reentry baseline after seeing which gives a tractable on-chip wavebreak window.
  - **Gate**: blueprint only, no build.
