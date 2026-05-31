# May 2026 Scout — Monthly Report Pipeline

> Period: 2026-05-01 → 2026-05-28 (since April report submitted 2026-05-01).
> Generated for the May progress report (due 2026-05-28, last Thursday of May).
> Inputs: git working tree (NO May commits — history ends 2026-04-30), IDEALOG dated
> entries, memory/, PI email thread metadata. Calibration anchor: April deck (11 slides).

## SCOUT NOTE — git is blind this period
The last commit is dated **2026-04-30** (`abd62388`). There are **0 commits in May**, but
**187 uncommitted tracked files** in the working tree (a large reorg: deleted all
`Surrogate/*.md` top-level docs + run scripts, removed `Research/Complete/lbm_cardiac/`,
new `cardiac_core/` package). **May signal lives in the dirty tree + IDEALOG, not git log.**
(Friction obs #2 confirmed at scale: git log and IDEALOG are complementary; this month git
contributes almost nothing.)

---

## Project inventory (PI-visibility weighted)

### A. Boundary Conduction Speedup — HEADLINE (PI anchor)
**Activity**: heaviest of the month. Two sessions (May 2, May 14), 14 HDF5 diagnostic cases.
- **May 2**: Theoretical — why Fickian (gradient) flux is *sign-locked* to forward crescent.
  k-sweep (sign locked positive at every k), equilibrium argument V*(y)=N_in/(N_in+N_out)·V_up
  identical at edge and interior → Effect B structurally dead in Fickian. Capacitor-vs-resistor
  mnemonic. Storage-tank cardinal4 video + figure cleanup.
- **May 14** (big): high-res dV/dt diagnostics + mechanism isolation + cross-engine closure.
  - Cases 1-4: dV/dt decomposition. **2/3 structural deficit confirmed at step 1** (bdry 92 vs
    ctr 138 mV/ms) for moore8_uniform + face_mirror. face_mirror_iso = 0 to FP precision.
  - Cases 5-6: synchronized-AP tests prove the source-effect imbalance is **local per-column**,
    generated fresh wherever a column charges from rest under face_mirror.
  - Case 7: **imposed inverse crescent is eaten by face_mirror in 2 columns**, flips to forward,
    asymptotes +568 µs by col 40. face_mirror sign-locks to forward crescent.
  - Case 8: **LBM cross-engine verification** — HBB eats inverse crescent (~20 cols, milder 5/6).
  - **Major correction**: `HBB ≡ face_mirror` (NOT face_mirror_iso, as earlier claimed).
    `specular ≡ face_mirror_iso`. Weight ratio sets deficit magnitude (2/3 vs 5/6).
  - Cases 9-12: 4-way LBM (canonical/uniform × HBB/specular). Specular = zero structural bias.
  - **Cases 13-14: NOVEL "horizontal redirect" BC → sustained INVERSE crescent (boundary
    SPEEDUP)**. Canonical −1146 µs, uniform −3106 µs by col 38 (grows with distance).
  - **Three BC families established**: HBB/fm (forward slowdown) · specular/fmi (zero) ·
    horizontal (inverse speedup, novel). Future: weighted (α,β,γ) 3-simplex BC family,
    fittable to experimental tissue LAT — "wall personality" axis distinguishing pathology.
- **PI signals**: (1) May-1 John forwarded **Andre's Cardiac Conduction handbook** quoting
  source-sink loading ("large depolarization... due to the large 'load' of the four adjacent
  cells") — direct mechanism for boundary speedup, his April comment #5 follow-through.
  (2) John is **grant-writing** and requested Charley's slides (May 17). (3) The novel
  inverse-crescent / boundary-speedup result directly answers his April "John Artifact"
  reframing (#3): the curved wavefront may be physiology, and a BC family can produce *either*
  direction.
- **Continuity**: continues April slides 3-6 (crescent / camel toe / inverse crescent).
- **Proposed**: 2-3 slides (HEADLINE).

### B. Surrogate Pipeline — HYBRID PIVOT (honest negative → redirect)
**Activity** (Session 29 late, working-tree model edits + doc cleanup):
- Benchmark: v4 ionic-surrogate Euler path = 4.2 M cell-steps/s vs **classical TTP06 on GPU =
  34.1 M cs/s → surrogate is 8× SLOWER** at tissue scale. 94% of bidomain wall time is the
  **elliptic solve**, not the ionic step → ionic surrogate was never the speedup lever.
- **PIVOT**: keep classical TTP06 as ionic scaffold; build neural surrogate for the bidomain
  **elliptic step** instead. Dual CNN towers (Vm / φ_e) with cross-communication. Parabolic-
  elliptic v1 first; hyperbolic deferred.
- Ionic v4 demoted to secondary roles (CPU deployment 3-7× win, differentiable coupling, param
  optimization). Big doc/script cleanup (deleted Surrogate top-level docs + run_*.py post
  cardiac_ml cutover).
- **Continuity**: continues April slides 9-10 (surrogate). Partially answers John's #11
  (train on richer model — related but distinct).
- **Proposed**: 1-2 slides. Honest-failure-with-redirect framing (the format John praised).

### C. Engine Consolidation → cardiac_core/
**Activity**: Phase 0 done (API wrapper layer, 34 tests). New `cardiac_core/` package in tree
(api.py + tests). Goal: single shared copy of ionic/mesh/stimulus/conductivity across 3 engines,
unified API, kill the `sys.modules` hack + 15 duplicated files. Next: move ionic models in (Phase 1).
- **Proposed**: 1 slide OR a summary/general-lab-activities bullet (infrastructure, lower PI pull).

### D. Research Environment Optimization
**Activity**: May-11 entry — two-system architecture + Obsidian vault as sibling repo, coarse-first.
Claude Code workflow tooling.
- **Proposed**: "General lab activities" bullet, NOT a slide (PI is science-focused).

### E. Bidomain Parabolic-Parabolic
**Activity**: README + IDEALOG touched (uncommitted); latest dated entry is 2026-04-23. Status:
"Research complete; 3-phase plan settled (ε-regularization → Cattaneo → boundary validation),
ready for /blueprint." John folded this into April's "LBM hyperbolic derivation" future bullet.
- **Proposed**: Future Outlook bullet, not a slide.

### F. lbm_ep reopened / lbm_cardiac merged
**Activity**: `Research/Complete/lbm_cardiac/` deleted in tree; lbm_ep reopened 2026-04-19 for
engine maturation (anisotropy, boundary artifacts). LBM was the vehicle for cases 8-14 above —
folds into the boundary headline rather than standing alone.
- **Proposed**: subsumed into A; mention in Future Outlook (LBM hyperbolic derivation).

---

## Existing asset to leverage
**May-15 lab-meeting deck** ("Group Meeting Slides", ~15-min project update) uploaded to
SharePoint `/Charley Chang/Presentation` on May 17 at John's request for his grant. This deck
likely already presents the May boundary work and is the strongest base asset for the report —
**confirm with user whether to build the May report on top of it.**

---

## PI email digest (May, john.f.zimmerman@cornell.edu)
| Date | Subject | Signal |
|------|---------|--------|
| 05-01 | Cardiac Conduction handbook | Andre's handbook; source-sink loading quote → boundary mechanism (comment #5 follow-up) |
| 05-02 | Re: April Progress Report | "Good job overall" + 11 comments (April round-trip, closed) |
| 05-04 | 5/15 Project Update | John invited Charley to present ~15 min at lab meeting |
| 05-12 | Handoff meeting | "hand off of Charlie's project" (to bmi25 + lc836) |
| 05-15 | (lab meeting) | Charley presented project update |
| 05-17 | Group Meeting Slides | John grant-writing, requested slides → uploaded to SharePoint |

---

## Proposed slide allocation (rank-driven, ~10-11 slides, April = floor)
1. Title
2. Summary (TOC + reporting period)
3-5. **Boundary speedup** (3 slides): inverse-crescent discovery + 3 BC families + cross-engine
     closure + weighted-simplex future. Foreground Andre-handbook source-sink connection.
6-7. **Surrogate hybrid pivot** (2 slides): benchmark (8× slower) honest result + elliptic-surrogate
     redirect (dual CNN towers).
8. **Engine consolidation / cardiac_core** (1 slide, or demote to summary bullet).
9. **Future Outlook**: weighted-BC simplex + experimental fitting · elliptic surrogate v1 ·
   bidomain PP (Cattaneo) / LBM hyperbolic · cardiac_core Phase 1.
(General lab activities bullet on summary: research-environment / Obsidian tooling.)

**Open triage decisions for user** (human-in-the-loop, per friction obs #6, #11):
- Project ranking (ordinal) → slide allocation.
- Build on the May-15 group-meeting deck, or fresh?
- Include cardiac_core as a slide or a bullet?
