# Source-Sink Mismatch Investigation — Idea Log

> Thinking trail. Spun out of `boundary_conduction_speedup` 2026-06-06.
> Scan in 30 s to remember where we are and how we got here.

## Current Direction
Target = Ciaccio 2018 Fig 4: source-sink wavefront curvature (concave-speedup /
rectilinear / convex-slow / block) driven by viable-tissue **thickness** transitions.
Deep-research (2026-06-06) CONFIRMED the fix: the thickness-weighted "augmented"
monodomain `∂V/∂t = (1/T)∇·(T·D∇V) − I_ion/Cm` (T a 2-D coefficient field), rigorously
derived + 3D-validated by Biktasheva/Dierckx/Biktashev (PRL 2015 114:068302). The open prize is
to run it with a cardiac ionic model (TTP06/ORd) and reproduce Fig-4 + block — not
yet done in the literature.

## Next Step
Implement the `(1/T)∇·(T·D∇V)` operator in V5.4 FDM with a thickness field T(x,y)
(start: a 1-D/2-D thin strip with a thickness STEP and a thickness RAMP), then:
(1) recover `θ = θ₀ − D·(∇T/T)`; (2) reproduce Fig-4 A–D by sweeping `ΔT/(c·T)`
through ~2; (3) validate vs a full-3-D varying-thickness reference. Decide params:
2018 (θ₀=0.4, D=0.2, thr≈2) vs 2015 (D=0.1, thr≈4). → /blueprint when settled.

## Failed Approaches (do NOT retry)
- **2-D in-plane width geometry for source-sink curvature** — wedge/hourglass/isthmus
  in a WIDE channel: a planar wave is width-independent (∂²V/∂y²≈0 → 1-D CV₀); a width
  change is a boundary deformation, not a cross-section load. Flat CV, no concave speedup.
- **Pure-diffusion probe** — category error: eikonal/source-sink curvature is a
  reaction-diffusion WAVE property; pure diffusion (√t creep, no front) can't show it.
- **Expecting conduction block at healthy excitability** — high safety factor; no block
  even at a 1-cell neck. Block needs the thickness mechanism and/or reduced excitability.
- **Attributing the fix to Bishop & Plank "augmented monodomain"** — that's bath-loading
  (edge-conductivity modifier), a DIFFERENT mechanism. Correct ref: Biktasheva PRL 2015.

## Thread

### 2026-06-08: Phase 1 DONE — eikonal coupling confirmed; proved it was the parameters
S0 (expanding circle, V5.4 FDM, `moore8_iso`, dx=50µm): engine obeys `CV_n=CV0−D·κ` —
CV0=62.4cm/s, D_eik=0.00084 (0.84·D), r*≈134µm, R²=0.99997. Key: fit the SMOOTH
integrated `LAT(r)=r/CV0+(D/CV0²)ln r+c`, not differentiation (R²≈0.13) or per-cell
div(n̂) (R²≈0.04) — the curvature signal is only ~8–13%. S0b (same circle, vary only
params): 50µm/iso D_eik/D=+0.83; 250µm/iso +0.72; **250µm/cardinal4 (exact hourglass)
D_eik/D=−0.93 (sign inverted)**. → resolution+stencil decisively corrupted the
hourglass curvature physics; r*/dx went 0.5→2.7. Scope: isolates resolution/stencil;
regime+excitability (block) still S2/S3. New code: `cardiac_core.analysis`
(activation_time_interp/front_metrics/fit_eikonal, 3 self-tests); experiments in
`Monodomain/Engine_V5.4/experiments/fig4c_sourcesink/`. Next: Phase 2 = S1 dx sweep
(critical nucleus) → lock dx_resolved (target ≤~45µm; 50µm borderline).

### 2026-06-10: CONTROL PARAMETER = dx/r* (= dx·CV/D). Not wavelength, not constriction.
Closed the "why couldn't original recreate it" arc. Converging-half wall-center crescent
(clean metric, no fan):
- S0f dx sweep (fixed geom): −115→−236µs as dx 250→25µm — CONVERGES, doesn't die → physical
  under-resolved effect (opposite of a vanishing artifact).
- S0g (Step 2) wavelength via APD (GKr/GKs, CV fixed): crescent EXACTLY −175µs across 3.9×
  λ (APD 280→72ms). λ INERT — crescent is an activation/LAT feature; APD is repolarization,
  absent from r*=D/CV.
- S0h discriminator (scale geom+dx together, dx/constriction FIXED): crescent still moves
  −117→−180µs tracking r*/dx → NOT dx/constriction. Constriction sets magnitude; dx/r* sets
  resolution.
- S0i CV channel (GNa, fixed dx,D): crescent −338→−90µs as CV 49→81 → CV operative because
  it enters r*=D/CV. "λ effect" only ever acts through CV, never APD. CV is grid-fudgable.
Answer: original failed because dx(250–500µm) > r*(~134µm); criterion dx≲r*/3≈45µm. Caveat:
no single collapse on r*/dx across routes (1/CV time-scaling); spatial lead ~0.6–0.8 r* is
cleaner. Scripts: run_s0f/g/h/i_*.py. Full synthesis in KNOWLEDGE "CONSOLIDATED CONCLUSION".

### 2026-06-10: S0d — hourglass inverse crescent is RESOLUTION-dependent, not stencil (visually confirmed)
Re-ran actual hourglass at controlled params. PI visually confirmed: at dx=50µm BOTH
cardinal4 and moore8_iso show the dilation-wall inverse crescent; at 250µm neither does.
→ inverse crescent needs r* resolved (dx~50µm), NOT diagonal connectivity (axis-aligned
geometry; matches S0c). Original orig-vs-fixed video was confounded (cardinal4@250 vs
moore8@50) — apparent stencil effect was actually dx. CAVEAT: all my LAT-derived crescent
metrics (centerline CV, wall-center, edge-inner) FAILED — read "edge lags" while the front
visibly shows the inverse crescent. Visual front inspection (isochrones/video) is the
ground truth for boundary crescents; don't trust the derived scalar. Scripts: run_s0d*,
render_s0d_matched_video.

### 2026-06-08: S0c — diagonal connectivity ≠ full story; r*/dx (diffusion-param) tuning is the other part
Obstacle + planar wave, `moore8_iso`, leading/trailing boundary crescent vs r*/dx.
Diagonal connectivity NOT decisive here: cardinal4≈moore8_iso at coarse dx (−110 vs
−111µs lead) — opposite of S0b — because the planar wave runs along-axis (grid
anisotropy bites off-axis only; S0b's radial wave samples all angles). The operative
knob is **r*/dx = D/CV0 / dx**: leading inverse-crescent −110→−123→−163µs as r*/dx
0.8→1.6→3.2, growing whether via D×4 OR finer dx. So YES there's a diffusion-parameter
tuning process. Original semicircle ran dx=0.05 → r*/dx≈0.27 (under-resolved even with
specular) → leading read "linear". diag_eikonal_circle had a connectivity mismatch
(mono cardinal4 / LBM D2Q9). Story: need BOTH (A) diagonal connectivity (decisive for
bulk off-axis curvature, S0b) AND (B) r* resolved vs dx (decisive for boundary crescent,
S0c). Script: run_s0c_obstacle_tuning.py.

### 2026-06-07: agentic search — LBM & bidomain extensions + code hunt
Fired 3 agentic searches on extending the thickness-weighted operator to LBM/bidomain.
Findings (full synthesis in KNOWLEDGE.md "Extension to LBM & Bidomain"; sources in
`literature/lbm_thickness_analog_and_code_2026-06-07.md`):
- **LBM**: `(1/T)∇·(T·D∇V)` = ADE with drift `u_drift=D·∇(lnT)` folded into the
  equilibrium. Direct physical analog = variable-water-depth depth-averaged ADE LBM
  (**Ru et al. 2021, CMAME 379:113745**) — same `(1/h)∇·(h·D∇C)` math, with a
  well-balanced linked-scheme that cancels the spurious `V·∇·u` reaction term. NO
  public code → re-implement from paper. Cardiac version unpublished = our opening.
- **Bidomain**: thickness-weighted bidomain object is unwritten (strongest novelty);
  pieces exist (Chapelle-Collin-Gerbeau 2013 thin-layer reduction; Biasi 2023
  smoothed-boundary `∇·(ψσ∇)` template). T-gradient enters both parabolic & elliptic.
- **Code copied**: BeatBox (the Biktasheva-2015 validation engine) vendored at
  `Research/code_examples/beatbox/` (GPL-3.0, 13M after trimming geometry binaries;
  113 .bbs example scripts kept incl. the FHN NegativeTension/PositiveTension cases).
- **CORRECTIONS**: original reduction paper is **PRL 2015 114:068302**, not "2019"
  (fixed in README/KNOWLEDGE/IDEALOG). Curvilinear-CDE LBM author is **Yoshida &
  Nagaoka 2014** (JCP 257:884), not "Yang". Ru et al. is **CMAME 379:113745**, not
  "374:113563".

### 2026-06-06: spun off from boundary_conduction_speedup
This question was extracted from the source-sink/eikonal/thickness thread of
`boundary_conduction_speedup`. Migrated here: the 8 source-sink diagnostic scripts,
the Ciaccio paper + Rossi(thickness)/Bishop(augmented) refs, the source-sink media,
and the findings below. The boundary-BC thread (specular speedup, crescent taxonomy,
κ-accumulation) stayed in the parent question.

### 2026-06-06: BREAKTHROUGH — target effect is THICKNESS (cross-section) source-sink, not 2-D in-plane geometry
PI identified Ciaccio 2018 Fig 4 as the exact target. Reading the PDF (Fig 4 + Fig 5 +
Eq 1) revealed the driver is IBZ **thickness** T (out-of-plane conducting volume),
governed by `θ = θ₀ − D·ΔT/(c·T)`, block when `ΔT/(c·T) ≳ 2`. Realization: every
2-D in-plane geometry we'd tried this session was a domain-SHAPE change, which is
CV-independent for a planar wave — that's why we saw flat CV / radial-collapse only.
The effect needs the varying dimension as a PDE COEFFICIENT (thickness) or in the
thin/cable regime.

### 2026-06-06 (correction): "2-D structurally cannot" was an over-claim
PI pushback (correct): area is the 2-D analog of volume; the source-sink parameter is
the relative cross-section gradient `(1/A)dA/dx` — thickness in 3-D, width in 2-D,
same math; no literal extra dimension needed. Corrected statement: the cross-section
change loads the BULK wave only in the THIN/CABLE regime (`W ≲ sqrt(D·δ/CV) ≈ 1 cell`
at our params). Wide channels are width-blind. Two valid 2-D routes: (a) genuinely
thin varying-width strip, (b) augmented monodomain with a thickness coefficient field
(practical). Both 2-D, neither adds a grid dimension.

### 2026-06-06: deep-research verdict (24/25 claims confirmed)
Augmented/thickness-weighted monodomain `(1/T)∇·(T·D∇V)` confirmed as the correct,
rigorously-derived (Biktasheva PRL 2019, arXiv:1408.3654), 3D-validated reduced model;
`(1/T)∇·(T∇V)=∇²V+(∇lnT)·∇V` → exactly Ciaccio's `∇T/T`. Bishop & Plank's
"augmented monodomain" is a DISTINCT bath-loading tool (attribution corrected).
Params θ₀≈0.4 mm/ms, D=0.1–0.2 mm²/ms, thr ΔT/T≈2–4. Cardiac-ionic Fig-4 reproduction
validated vs 3-D is the gap = our original contribution. Full synthesis in KNOWLEDGE.md.

### 2026-06-03 → 06-06: the negative-result trail that led here (summary)
eikonal circle (leading inverse crescent only −0.14 ms; trailing crescent = diffraction
shadow) → isthmus/strand→expansion (radial collapse, no block to 1 cell) → converging
wedge (flat CV) → hourglass (constriction flat, dilation fan) → diffusion-only (√t creep,
both engines identical) → foot/λ (~physiological, not mistuned) → mirror-vs-iso slanted
(wall-parallel BC effect, angle-dependent). All consistent once we understood the
in-plane-width-is-CV-blind / thickness-is-the-real-variable resolution. Details + figures:
this folder's scripts and `media/source_sink_mismatch_investigation/`.

## Session Log

### 2026-06-07 Session
**Worked on**: Closed out the multi-day source-sink arc (run under boundary_conduction_speedup), identified the exact target artifact, verified the fix by literature, and migrated everything into this dedicated question.
**Accomplished**:
- Ran the full source-sink/curvature suite in LBM V1 + Monodomain V5.4 (eikonal circle, isthmus, strand→expansion, converging wedge, hourglass, diffusion-only, foot/λ, mirror-vs-iso slanted) — all weak/null; root-caused to 2-D in-plane width being CV-blind for a planar wave (width change = boundary deformation, not a cross-section load).
- PI identified the target: Ciaccio 2018 JACEP Fig 4 — thickness-driven source-sink wavefront curvature, θ=θ₀−D·ΔT/(c·T), block at ΔT/(c·T)≈2. Read the PDF directly.
- Corrected an over-claim: "2-D structurally cannot" → it is a thin/cable-regime condition; area is the 2-D analog of volume.
- Deep-research (6 angles, 21 sources, 24/25 claims adversarially verified): CONFIRMED the thickness-weighted "augmented" monodomain (1/T)∇·(T·D∇V) is the rigorously-derived (Biktasheva/Dierckx/Biktashev, PRL 2019, arXiv:1408.3654) + 3-D-validated fix; (1/H)∇·(H∇u)=∇²u+(∇lnH)·∇u gives the ∇T/T term. Attribution corrected: NOT Bishop & Plank (that's bath-loading). Full report in literature/.
- Created this question; migrated 8 scripts, 3 papers, 24 media, the full deep-research report, and the foundational source-sink theory (copied from parent). Registered in MASTER.md + MASTER_KNOWLEDGE_INDEX.md; cross-linked the parent.
**Next**: /blueprint the thickness-weighted (1/T)∇·(T·D∇V) operator in V5.4 FDM (start: thin strip with a thickness step + ramp) → recover θ=θ₀−D·∇T/T → reproduce Ciaccio Fig-4 A–D + block by sweeping ΔT/(c·T) through ~2 → validate vs a full-3-D varying-thickness reference (the cardiac-ionic validation gap = the project's original contribution). Decide D: 2018 (0.2, thr≈2) vs 2015 (0.1, thr≈4).
