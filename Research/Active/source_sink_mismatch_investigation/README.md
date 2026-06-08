# Source-Sink Mismatch Investigation

## Question
Does a propagating wavefront, crossing a transition in the conducting **cross-section** of the tissue (a change in viable-tissue **thickness**), reshape and slow/block the way Ciaccio et al. (2018) Figure 4 describes — concave + speedup when the downstream conducting volume shrinks, rectilinear when unchanged, convex + slowing/functional-block when it grows? Can our engines reproduce that effect, and with what model?

## Status: Active (spun out of `boundary_conduction_speedup`, 2026-06-06)

## Why It Matters
Source-sink (current-to-load) mismatch at conducting-cross-section transitions is the mechanism Ciaccio/Coromilas/Wit propose for the lateral functional-block lines that bound the re-entry isthmus in post-infarction VT. Reproducing it faithfully is prerequisite to any geometry-driven arrhythmia study. This question exists because a session-long attempt to elicit the effect with 2-D in-plane geometry **failed**, and the cause turned out to be a model-structure issue (see KNOWLEDGE).

## The Target Artifact (Ciaccio 2018, Fig 4)
`θ = θ₀ − D·(ΔT)/(c·T)`  — θ=CV, θ₀≈0.4 mm/ms, D≈0.1–0.2 mm²/ms, T=tissue thickness, ΔT/c=thickness gradient.
- thickness DECREASES ahead (sink<source) → wavefront concave, accelerates
- no change → rectilinear
- thickness INCREASES ahead (sink>source) → convex, slows; functional block when `ΔT/(c·T) ≳ 2–4 per mm`.
Paper: `papers/ciaccio_2018_source_sink_functional_block.pdf` (JACC Clin EP 2018;4(1):1-16, DOI 10.1016/j.jacep.2017.08.019).

## The Verified Fix (deep-research, 2026-06-06)
Thickness-weighted ("augmented") monodomain: `∂V/∂t = (1/T)∇·(T·D∇V) − I_ion/Cm`, with T(x,y) a 2-D thickness/cross-section **coefficient field** (NOT a 3rd grid dimension). Rigorously derived as the O(μ²) thin-layer reduction of 3-D RD (Biktasheva/Dierckx/Biktashev, PRL 2015 114:068302, arXiv:1408.3654) and validated vs full 3-D (BeatBox, vendored at `Research/code_examples/beatbox/`). Identity `(1/T)∇·(T∇V)=∇²V+(∇lnT)·∇V` gives the `∇T/T` term = Ciaccio's eikonal relation. (NB: distinct from Bishop & Plank's bath-loading "augmented monodomain".)

## Completion Criteria
- [ ] Implement the thickness-weighted operator `(1/T)∇·(T·D∇V)` in V5.4 FDM (and/or LBM) with a thickness field T(x,y)
- [ ] Reproduce Ciaccio Fig-4 A–D (concave / rectilinear / convex / block) by sweeping `ΔT/(c·T)` through the ~2–4 threshold
- [ ] Recover the eikonal relation `θ = θ₀ − D·(∇T/T)` quantitatively from the simulation
- [ ] Validate against a full-3-D varying-thickness reference (the cardiac-ionic validation gap — likely original contribution)
- [ ] Confirm the regime boundary: where the thin-layer (cable) limit breaks down
- [ ] Numerical-pitfall check: mesh resolution needed at a sharp thickness transition (does under-resolution spuriously create/suppress block?)

## Active Test Plan
**[FIG4C_BLOCK_TEST_PLAN.md](./FIG4C_BLOCK_TEST_PLAN.md)** — systematic 2-D campaign (no
thickness) to reproduce Ciaccio Fig-4C/D from in-plane source-sink mismatch. Tests the
four necessary conditions our hourglass runs broke (resolution `r*`, current-limited
source, excitability, measurement), each yielding a tuned parameter or a prove/disprove
verdict. Supersedes the thickness-weighted direction below as the primary line.

## Sub-Questions
| Sub-Question | Status | Note |
|---|---|---|
| Is augmented monodomain the right model? | **Resolved (lit)** | Yes — Biktasheva PRL 2015; derived + 3D-validated (non-cardiac) |
| Reproduce Fig-4 with cardiac ionics | Open | the project's opening; not done in literature |
| Which (θ₀, D, threshold) to adopt | Open | 2018 (D=0.2, thr≈2) vs 2015 (D=0.1, thr≈4) |
| Regime boundary (thin-layer μ) | Open | no explicit μ-threshold located |

## Engine References
| File | What it tells you |
|---|---|
| `Monodomain/Engine_V5.4/.../discretization_scheme/fdm.py` | FDM Laplacian; `apply_diffusion`; needs the (1/T)∇·(T·) operator added |
| `Monodomain/Engine_V5.4/.../monodomain.py` | orchestrator |
| `LBM/Engine_V1/src/` | LBM path; thickness-weighting = drift `u_drift=D·∇(lnT)` in equilibrium (Ru et al. 2021 recipe) |
| `Research/code_examples/beatbox/` | **Vendored** BeatBox (GPL-3.0) — Biktasheva-2015 3-D validation engine; FHN thickness-drift example scripts in `data/scripts/**/FitzHughNagumo_model/` |
| `Research/Active/source_sink_mismatch_investigation/literature/lbm_thickness_analog_and_code_2026-06-07.md` | LBM analog (Ru 2021) + bidomain pieces + code provenance |
| `Research/Knowledge/bidomain_simulation.md` | discretization/solver knowledge |

## Connected Research
- **boundary_conduction_speedup** (parent) — the same-cell specular boundary-speedup BC, crescent/HBB taxonomy, κ-accumulation; this question split off from its source-sink thread.

## Future Work
{none yet — see Completion Criteria}
