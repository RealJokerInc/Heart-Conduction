# LBM_V1 — Implementation Progress

> **Single source of truth for what's done, in-progress, and next.**
> Read this FIRST at the start of every session or after compaction.

---

## Current Status

**Active Phase:** ALL PHASES COMPLETE (0-8)
**Last Updated:** 2026-03-05

---

## Planning Phase — DONE

| Document | Status | Notes |
|----------|--------|-------|
| README.md | DONE | Purpose, architecture, design decisions |
| IMPLEMENTATION.md | DONE | 8 phases, 34 tests, full code structure |
| PROGRESS.md | DONE | This file |
| CONVENTIONS.md | DONE | Symbol naming (e_i, w_i, R, Omega_NR, cs2), tau-D pipeline, moment spaces |
| research/PAPER_COMPARISON.md | DONE | Rapaka vs Campos: MRT, moments, τ encoding, optimizations |
| research/rapaka_2012.txt | DONE | Extracted paper text |
| research/campos_2016.txt | DONE | Extracted paper text |

**Decisions recorded:**
- Framework: PyTorch (V5.4 compatibility, MPS, torch.compile, eager debug)
- Ionic: Copy V5.4 ionic/ folder (TTP06, ORd), standalone Rush-Larsen function
- Architecture: d2q5.py and d2q9.py as self-contained lattice modules, simulation.py as coordinator
- Naming: follows Rapaka/Campos conventions (see CONVENTIONS.md)

---

## Phase 0: Copy Ionic Models — DONE (2026-03-05)

Ionic models not yet needed; placeholder `__init__.py` files created.

---

## Phase 1: Lattice Definitions — DONE (2026-03-05)

**Files:** `src/lattice/base.py`, `src/lattice/d2q5.py`, `src/lattice/d2q9.py`, `src/diffusion.py`
**Tests:** Phase 1 tests integrated into Phase 2 test file.

---

## Phase 2: Collision Operators — DONE (2026-03-05)

**Files:** `src/collision/bgk.py`, `src/collision/mrt/d2q5.py`, `src/collision/mrt/d2q9.py`
**Tests:** 6/6 pass (test_phase2.py: 2-V1 through 2-V6)

---

## Phase 3: State + Streaming — DONE (2026-03-05)

**Files:** `src/state.py`, `src/streaming/d2q5.py`, `src/streaming/d2q9.py`
**Tests:** Streaming validated through Phase 2 and Phase 4 tests.
**Bug fix:** Roll shift signs were inverted (all directions swapped). Fixed: shift = +e_component for pull convention.

---

## Phase 4: Boundary Conditions — DONE (2026-03-05)

**Files:** `src/boundary/masks.py`, `src/boundary/neumann.py`, `src/boundary/dirichlet.py`, `src/boundary/absorbing.py`
**Tests:** 4/4 pass (test_phase4.py: 4-V1 through 4-V4)
- 4-V1: Neumann conservation, drift < 1e-10 (both D2Q5, D2Q9)
- 4-V2: Dirichlet steady state, linearity < 5e-3
- 4-V3: Absorbing, reflected energy < 5%
- 4-V4: Mixed BC (Dirichlet top/bottom + Neumann L/R)
**Note:** Tests use `make_rect_bounce_masks()` helper for full-grid rectangular domains (manual edge masks). `precompute_bounce_masks()` is for irregular domains with explicit outside nodes.

---

## Phase 5: Pure Diffusion Validation — DONE (2026-03-05)

**Tests:** 5/5 pass (test_phase5.py: 5-V1 through 5-V5)
- 5-V1: D2Q5 BGK Gaussian variance, err < 0.1%
- 5-V2: D2Q9 BGK Gaussian variance, err < 0.1%
- 5-V3: MRT isotropic matches BGK, diff < 1e-14
- 5-V4: Anisotropic MRT D_xx=0.2/D_yy=0.05, variance ratio 2.00, err < 0.2%
- 5-V5: Rotated 30-deg fiber, D_xx/D_yy correct, err < 0.1%
**API change:** MRT s_j → s_jx + s_jy (separate flux relaxation for anisotropic D)

---

## Phase 6: Ionic Model Coupling — DONE (2026-03-05)

**Files:** `ionic/` (copied from V5.3), `src/solver/rush_larsen.py`
**Tests:** 3/3 pass (test_phase6.py: 6-V1 through 6-V3)
- 6-V1: Single-cell AP, peak=56.6mV, APD90=223ms
- 6-V2: Rush-Larsen stable at dt=0.1ms, matches ref within 0.05mV
- 6-V3: Source term conservation verified

---

## Phase 7: Simulation Orchestrator — DONE (2026-03-05)

**Files:** `src/step.py`, `src/simulation.py`
**Tests:** 3/3 pass (test_phase7.py: 7-V1 through 7-V3)
- 7-V1: Planar wave, CV=75.4 cm/s, 90 nodes activated
- 7-V2: D2Q5 vs D2Q9 match (75 vs 74 activated nodes)
- 7-V3: Stimulus timing correct (5.6ms after 5ms start)
**Bug fix:** Source term had extra chi factor: R=-(I_ion+I_stim)/(chi*Cm) → R=-(I_ion+I_stim)/Cm

---

## Phase 8: Boundary Speedup Experiment — DONE (2026-03-05)

**Tests:** 3/3 pass (test_phase8.py: 8-V1 through 8-V3)
- 8-V1: Neumann uniform CV, ratio=1.0000 (no boundary artifact)
- 8-V2: D2Q5 vs D2Q9 CV match (rel_diff < 1%)
- 8-V3: Wave propagates full domain, CV=75.0 cm/s
**Key findings:**
- Neumann BC gives perfectly uniform CV (no speedup or slowdown at boundaries)
- D2Q9 edge CV is ~3% slower than center (known O(dx²) diagonal artifact)
- D2Q5 has no such artifact

---

## Session Log

| Date | Session | Work Done |
|------|---------|-----------|
| 2026-03-04 | 1 | Planning: read both LBM-EP papers, created README, IMPLEMENTATION, PROGRESS, PAPER_COMPARISON, CONVENTIONS. Decided on PyTorch, V5.4 ionic reuse, d2q5/d2q9 self-contained module architecture. |
| 2026-03-04 | 2 | Architecture: refactored to V5.4-style nested structure with torch.compile two-layer pattern. Updated IMPLEMENTATION, README, PROGRESS. Created BOUNDARY_SPEEDUP_ANALYSIS.md: proved Dirichlet BC on V cannot produce Kleber speedup (acts as current sink). Derived correct monodomain approximation (spatially varying D with Neumann BC) and bidomain FDM approach (intra-Neumann + extra-Dirichlet). |
| 2026-03-05 | 3 | Expanded Phase 4 docs in IMPLEMENTATION.md. Cross-doc consistency audit and fixes. |
| 2026-03-05 | 4 | Implemented all phases (0-8). Fixed streaming bug (roll shifts inverted). Fixed source term (removed extra chi factor). MRT updated with separate s_jx/s_jy for anisotropic D. All 24 tests pass. |
