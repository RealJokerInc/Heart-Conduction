# Source-Sink Mismatch Investigation — Knowledge File

> Running synthesis. Spun out of `boundary_conduction_speedup` on 2026-06-06.
> Promoted to `Research/Knowledge/` on completion.

## ⚑ UPDATE 2026-06-08 — Premise corrected + Phase 1 results

**The earlier thickness framing below was wrong** (mechanism vs proxy error — see
`feedback_ciaccio_fig4_mechanism_not_thickness` and the error-trace in IDEALOG).
Corrected: the Ciaccio source-sink effect is **2-D in-plane cross-section curvature
(Fig 4)**, governed by the eikonal relation `CV_n = CV0 − D·κ`. Thickness (Fig 5,
`θ=θ0−D·ΔT/(c·T)`) is just the IBZ **measurable proxy** for the conducting
cross-section; in 2-D the analog is width, `θ=θ0−D·ΔW/(c·W)`. No 3rd dimension is
needed. Active line is now [FIG4C_BLOCK_TEST_PLAN.md](FIG4C_BLOCK_TEST_PLAN.md) /
[PLAN.md](PLAN.md). (Thickness-weighted/augmented monodomain + LBM/bidomain extension
work is retained below as a separate optional 3-D-fidelity sub-question, NOT the main line.)

## ⚑ CONSOLIDATED CONCLUSION 2026-06-10 — the control parameter is dx/r* (= dx·CV/D)

The source-sink curvature/crescent effect is recreated correctly **iff the grid resolves
the wavefront's own electrotonic length** `r* = D/CV0 ≈ 134–160 µm` (the eikonal-front /
foot thickness). The single control parameter is

```
   dx / r*  =  dx · CV / D          (NOT dx/wavelength, NOT dx/constriction)
```

**Why the original engine failed:** it ran at dx = 250–500 µm > r* ≈ 134 µm, so the
r*-thick boundary layer was sub-grid. Concrete criterion: **dx ≲ r*/3 ≈ 45 µm.**

Evidence (converging-half wall-center crescent = clean metric, no geometric fan;
hourglass, moore8_iso):
- **S0f — dx sweep (fixed geometry):** crescent −115 → −236 µs as dx 250 → 25 µm
  (r*/dx 0.5 → 5.4). **Converges, does NOT die** — physical under-resolved effect, the
  opposite of a vanishing artifact.
- **S0g (Step 2) — wavelength via APD (GKr/GKs, fixed CV, fixed dx):** crescent
  **exactly −175 µs across a 3.9× λ swing** (APD 280→72 ms, λ 17→4.5 cm). **λ is inert.**
  Mechanism: the crescent is an *activation/LAT* feature (set by the upstroke); APD is
  repolarization and is absent from `r* = D/CV`. (So my earlier "resolution-dependent"
  was right in direction; "wavelength-dependent" is wrong via the APD channel.)
- **S0h — discriminator (scale geometry+dx together, dx/constriction FIXED at 0.17):**
  crescent still moves −117 → −180 µs as r*/dx 0.67 → 2.68. **So it is NOT dx/constriction;**
  the constriction sets the *magnitude* of the source-sink mismatch, but the *resolution
  criterion* is dx/r*, independent of feature size.
- **S0i — CV channel (GNa, fixed dx, fixed D):** crescent −338 → −90 µs as CV 49 → 81
  cm/s (r* = D/CV 202 → 124 µm). **CV is the operative knob** because it enters r*. This
  is why a "wavelength effect" only ever shows through CV, never APD: `λ = CV·APD`, and
  only the CV factor reaches `r* = D/CV`. CV is itself grid-dependent ("fudgable" — sags
  62→49 cm/s at coarse dx, S0b), so `dx/r* = dx·CV/D` couples dx and CV.

**Caveat — not a single clean collapse across routes.** At the same r*/dx ≈ 2.4–2.7 the
CV-route (S0i, −338 µs) and the dx-route (S0f, ~−210 µs) differ: the crescent is a *time*
and a slower wave (lower CV) inflates the same spatial lead. The cleaner invariant is the
**spatial lead ≈ 0.6–0.8·r\***. Headline (CV operative, APD inert, control = dx/r*) is
solid; the exact non-dimensional form carries a 1/CV time-scaling.

**Methodological lessons (see `feedback_visual_front_over_derived_metric`):** (1) the
converging half is the real test — the diverging half makes a forward crescent trivially
(geometric fan) that masks the boundary signal; (2) lead with the isochrone/video front,
not derived scalars (my centerline/edge-inner metrics were blind to the inverse crescent
the PI saw); (3) always match dx/dt/D when comparing configs — the first orig-vs-fixed
video was dx-confounded.

### Phase 1 (S0) — the engine DOES obey the eikonal relation
Expanding circular wave, Monodomain V5.4 FDM, TTP06 EPI, `moore8_iso`, dx=50µm:
- **CV0 = 62.4 cm/s**, **D_eik = 0.00084 cm²/ms** (= 0.84·operator D=0.001; the ~16%
  deficit is the expected leading-order eikonal correction), **r* = D_eik/CV0 ≈ 134 µm**,
  fit **R² = 0.99997**.
- Recovered by fitting the SMOOTH integrated form `LAT(r) = r/CV0 + (D/CV0²)·ln r + c`
  — NOT by differentiating binned LAT (R²≈0.13) nor per-cell `div(n̂)` (R²≈0.04); the
  curvature signal is only ~8–13% here and drowns in differentiation/point noise.
- Figure: `media/.../images/2026-06-08/s0-eikonal-cv-vs-kappa-mono_06.png`;
  video `videos/2026-06-08/s0-eikonal-expanding-circle_02.mp4`.

### What we changed vs the hourglass — and proof it was the parameters (S0b)
Three changes turned a failed measurement into a clean one:
1. **Resolution**: dx 250 µm → **50 µm** (5×). r*/dx went 0.5 → 2.7 — the curvature
   zone went from sub-grid-cell to resolved.
2. **Stencil — directional diagonal connectivity**: `cardinal4` (no diagonal
   connectivity) → **`moore8_iso`** (4:1-weighted diagonal connectivity, Patra-Kałuża).
   Modeled diffusion is isotropic (scalar D) in BOTH; without diagonal channels the
   5-point Laplacian carries flux only along axes → direction-dependent CV → the
   "circle" becomes a rounded square, and that directional scatter (~few %) swamps the
   ~8% curvature signal. NOT material/tensor anisotropy. See "The crucial cause:
   directional diagonal connectivity" below.
3. **Measurement**: integrated `LAT(r)` fit instead of differentiation/per-cell div.

**S0b controlled isolation** — SAME expanding circle, vary only (dx, stencil):

| Params | D_eik/D | r*/dx | result |
|---|---|---|---|
| 50 µm, `moore8_iso` (S0) | **+0.83** | 2.66 | curvature resolved, correct sign |
| 250 µm, `moore8_iso` | +0.72 | 0.52 | degraded (under-resolved) |
| 250 µm, `cardinal4` (**exact hourglass**) | **−0.93** | −0.76 | **sign INVERTED — anti-eikonal artifact** |

At the hourglass params the recovered curvature coefficient is **negative** — the
discretization makes convex fronts appear to *speed up*, the opposite of correct
physics. (CV0 also sags: 62→55→49 cm/s as dx coarsens — under-resolved upstroke.)
Figure: `media/.../images/2026-06-08/s0b-param-degradation_01.png`.

**Conclusion:** resolution + stencil were a *real, decisive* cause of the hourglass's
inability to show source-sink curvature — confirmed by reproducing the corruption on a
clean geometry. **Scope:** this isolates resolution/stencil only; the hourglass also
had a reservoir-fed over-driven source and healthy excitability (Conditions 2 & 3),
which are separate factors still to be tested in S2/S3 (they govern *block*, not the
curvature signal itself).

### S0c — diagonal connectivity is necessary but NOT sufficient; r*/dx tuning is the other part
Planar wave past a circular obstacle (the infarct-boundary geometry of
diag_eikonal_circle / sim_semicircle), `moore8_iso`, measuring boundary-adjacent LAT
minus bulk-at-same-x (negative = LEADS = inverse crescent = source-sink speedup at the
obstacle edge; positive = LAGS = path-length shadow):

| r*/dx | lead (µs) | trail (µs) | case |
|---|---|---|---|
| 0.80 | −110 | +1576 | coarse `cardinal4` (no diag) |
| 0.80 | −111 | +1711 | coarse `moore8_iso` (diag only) |
| 1.60 | −123 | +831 | coarse `moore8_iso`, **D×4 (tune D)** |
| 3.21 | −163 | +1250 | fine `moore8_iso` (resolved) |

Two findings:
1. **Diagonal connectivity is NOT decisive here** — cardinal4 vs moore8_iso at the same
   coarse dx are ≈identical (−110 vs −111 µs). Opposite of S0b (where the stencil flipped
   the sign). **Why:** missing-diagonal grid anisotropy bites *off-axis* propagation; the
   S0b radial wave samples all angles, but this planar wave runs *along the grid axis*
   past the obstacle, where cardinal4 ≈ moore8_iso. So the stencil barely matters and the
   resolution/diffusion knob dominates.
2. **r*/dx (the diffusion-parameter / resolution knob) is the operative factor** — the
   leading inverse-crescent strengthens monotonically −110 → −123 → −163 µs as r*/dx goes
   0.8 → 1.6 → 3.2, and it grows whether reached by **raising D** (D×4 at fixed coarse
   grid) OR **refining dx**. r* = D/CV0 (CV0 ∝ √D, so r* ∝ √D). This is the
   "is-there-a-tuning-process-for-the-diffusion-parameter" question: **yes** — resolve r*
   (raise D or refine dx) and the source-sink boundary effect emerges/sharpens.

**Note on the original semicircle test:** it ran at **dx=0.05 cm → r*/dx ≈ 0.27** (even
*with* D2Q9 specular diagonal connectivity) — badly under-resolved, so the leading
source-sink curvature was sub-grid (read "linear") and only the gross trailing
path-length shadow survived. Also `diag_eikonal_circle.py` had a **connectivity mismatch**
(mono `cardinal4` = no diagonals; LBM `D2Q9` = diagonals), confounding that comparison.
Figure: `media/.../images/2026-06-08/s0c-obstacle-rstar-tuning_01.png`.

**Net corrected story:** the source-sink/curvature effect needs (A) directional diagonal
connectivity — decisive for *bulk off-axis* curvature (S0b); AND (B) r* = D/CV0 resolved
relative to dx — decisive for the *obstacle/boundary* crescent (S0c), tunable via D or dx.
Neither alone is the full story.

### S0d — hourglass confirmation: the inverse crescent is RESOLUTION-dependent, not stencil-dependent (visually confirmed)
Re-ran the actual hourglass (scaled) at controlled params. **Settled by direct visual
inspection of the front (video + isochrones), which is the reliable diagnostic here:**
- At **dx = 50 µm, BOTH `cardinal4` and `moore8_iso` show the dilation-wall inverse
  crescent** (boundary speedup, edge leads). At **dx = 250 µm, neither does.**
  → the inverse crescent is **resolution-dependent (needs r* resolved), NOT
  stencil/diagonal-connectivity-dependent** for this axis-aligned geometry.
- The original "orig vs fixed" video was **confounded**: `cardinal4`@250µm vs
  `moore8_iso`@50µm — the apparent stencil difference was actually the dx difference.
- Consistent with S0c: for axis-aligned boundary/obstacle source-sink, **r*/dx
  resolution is the operative knob**; diagonal connectivity matters for *off-axis/radial*
  curvature (S0b), not here.
- Centerline dilation CV dip (convex fan, ~10%) was present even at 250µm; the *wall
  inverse crescent* specifically required ~50µm. So at coarse dx the original missed the
  boundary inverse-crescent due to resolution.

**Measurement caveat (important):** every LAT-derived crescent metric I built — centerline
CV, wall-minus-center, local edge-minus-inner — read "edge lags" while the front *visibly*
shows the inverse crescent. **They did not capture the effect; visual front inspection
(isochrones / V-field video) was the ground truth.** Trust the front, not the derived
scalar, for boundary-crescent questions. Figures: `s0d4-isochrones-clean_01.png`;
videos `s0d-hourglass-orig-vs-fixed_01.mp4`, `s0d-hourglass-matched-dx-cardinal-vs-moore8_01.mp4`.

### The crucial cause: directional DIAGONAL CONNECTIVITY (not "anisotropy")
The `cardinal4` failure is the **diagonal-connectivity** axis established in the
boundary work (see §"Connectivity is the smoking gun: 8-neighbour vs 4-neighbour
ablation") — the SAME mechanism, here in the bulk. It is NOT material/tensor
anisotropy: with scalar D both stencils model the isotropic equation
`∂V/∂t = D∇²V − I_ion`.

- **`cardinal4` has no diagonal connectivity** → the discrete Laplacian can only carry
  flux along the grid axes, so its truncation error is axis-locked and not rotationally
  invariant:
  ```
     ∇²_h u = ∇²u + (h²/12)(u_xxxx + u_yyyy) + O(h⁴)
     u_xxxx + u_yyyy = ∇⁴u − 2·u_xxyy        (∇⁴ = biharmonic, rotationally invariant)
  ```
  For a plane wave at angle θ the error scales as `cos⁴θ+sin⁴θ = 1 − ½sin²2θ` →
  coefficient 1 on-axis, ½ on-diagonal: a factor-2 direction dependence → direction-
  dependent numerical CV → **rounded-square wave**.
- **`moore8_iso` adds directional diagonal connectivity** (4:1 cardinal:diagonal,
  1/6 prefactor, Patra-Kałuża) → the diagonal channels carry the off-axis flux the
  cardinal-only stencil cannot, making the leading error `∝ ∇⁴u` so `cos⁴θ+sin⁴θ → 1`
  (θ-independent) → **circular wave**.

So **directional diagonal connectivity is the crucial cause**; the
non-rotationally-invariant truncation error ("grid/numerical anisotropy") is its
mathematical manifestation, and direction-dependent CV is the effect. The S0b sign
inversion (cardinal4 −0.93 vs moore8_iso +0.72 at the SAME 250µm) is this artifact:
binning a rounded-square (diagonal-blind) wave by radius mixes fast-diagonal/slow-axial
cells in r-varying proportion, faking an inverted curvature response.

**Terminology rule:** reserve **"anisotropic" (unqualified) for the conductivity tensor
`D_xy≠0`** (the codebase's own convention: `cardinal4` "anisotropic D support" = tensor;
reduces to 5-point when `D_xy=0`). For the discretization effect, name the cause —
**"(directional) diagonal connectivity"** — not "anisotropy".

---

## Current Understanding

When a wavefront crosses a transition in the **conducting cross-section** of the
tissue — concretely, a change in viable-tissue **thickness** T — source-sink
(current-to-load) mismatch reshapes the front: **concave + speedup** when the
downstream conducting volume shrinks, **rectilinear** when unchanged, **convex +
slowing/functional-block** when it grows. This is Ciaccio et al. (2018) Fig 4 and
the mechanism they propose for the lateral functional-block lines bounding the
post-infarction re-entry isthmus.

The correct reduced model that reproduces this is the **thickness-weighted
("augmented") monodomain**:

```
   ∂V/∂t = (1/T) ∇·(T·D ∇V) − I_ion/Cm        T = T(x,y) thickness/cross-section field
```

Identity `(1/T)∇·(T∇V) = ∇²V + (∇lnT)·∇V`, so the extra term is `D·(∇T/T)·∇V`,
which yields exactly the Ciaccio eikonal relation `θ = θ₀ − D·(∇T/T)`. T is a
2-D **coefficient field**, NOT a literal 3rd grid dimension.

## The Target Artifact — Ciaccio 2018 Figure 4 / 5

Source: Ciaccio EJ, Coromilas J, Wit AL, Peters NS, Garan H, "Source-Sink
Mismatch Causing Functional Conduction Block in Re-Entrant Ventricular
Tachycardia," JACC Clin Electrophysiol 2018;4(1):1-16. Based on PubMed/PMC5874259.
DOI: https://doi.org/10.1016/j.jacep.2017.08.019. PDF in `papers/`.

- **Fig 4** (wave Source→Sink): A sink<source → concave, speeds up; B equal →
  rectilinear; C sink>source → convex, slows; D sink≫source → convex + functional
  block (double line). [theory/schematic]
- **Fig 5 + Eq 1** (the mechanism is THICKNESS): `θ = θ₀ − D·ΔT/(c·T)`,
  θ₀≈0.4 mm/ms, **D≈0.2 mm²/ms** (note: mm² /ms, a diffusion coefficient),
  T=thickness, c=space step. Worked example 400 µm→1200 µm over 1 mm → θ=0 (block).
  Block threshold `ΔT/(c·T) ≳ 2 per mm` (2018 paper). [theory/model]
- Companion Ciaccio 2015 (Comput Biol Med, PMC4533242) uses identical form with
  **D=0.1 mm²/ms → threshold ΔT/T ≈ 4**. Pick the paper when quoting a number.

## The Verified Fix (deep-research 2026-06-06; 24/25 claims adversarially confirmed)

**The augmented/thickness-weighted monodomain IS the correct, rigorously-derived
reduced model** — with an attribution correction:

- **Derivation + 3D validation [theory+simulation]:** Biktasheva, Dierckx,
  Biktashev (**PRL 2015, 114:068302**; DOI 10.1103/PhysRevLett.114.068302;
  arXiv:1408.3654). [Year corrected 2026-06-07 — was mis-recorded as "PRL 2019".]
  From 3-D RD in a thin layer
  z∈[z_min(x,y),z_max(x,y)], H≡thickness, no-flux top/bottom, the leading-order
  reduction is `u_t = f(u) + D(1/H)∇·(H∇u) + O(μ²)` (their Eq 4) — algebraically
  identical to `(1/T)∇·(T·D∇V)` for constant D. Their Eq 5: `h = D(∇K)·∇u, K=lnH`,
  so the correction scales as `∇(lnT)=∇T/T` — the rigorous origin of Ciaccio's
  phenomenological `∇T/T`. **Validated against full 3-D** (BeatBox, FitzHugh-Nagumo
  + Oregonator, thickness step) with quantitative agreement.
- **Attribution correction:** Bishop & Plank 2011 "augmented monodomain"
  (PMC3075562) is a **different** tool — a bath-LOADING edge-conductivity modifier
  (tag ~3 boundary elements, scale conductivity by R_ζ=g_b(g_i+g_e)/(g_e(g_i+g_b))),
  for current shunting into a conductive bath, NOT a global thickness field. The
  name overlaps; the mechanism differs. (We mis-attributed this initially.)
- **Parameters:** θ₀≈0.4 mm/ms; D = 0.1–0.2 mm²/ms; block threshold ΔT/T ≈ 2–4
  per mm. Canine empirical block threshold ~1.55 per mm ≈ predicted ~2.
- **Regime condition:** the varying dimension must be thin relative to the
  electrotonic space constant (the O(μ²) thin-layer limit) for the cross-section
  change to load the bulk wave.

### THE GAP / original contribution
No source was found that runs the thickness-weighted monodomain with a
**physiological cardiac ionic model (TTP06/ORd)** and reproduces the Ciaccio
Fig-4 concave/convex/block sequence validated against a full-3-D cardiac
reference. The reduction is validated only in generic FHN/Oregonator media for
scroll-wave drift. **This direct cardiac validation is the project's opening.**

## Extension to LBM & Bidomain (agentic search, 2026-06-07)

Full detail: `literature/lbm_thickness_analog_and_code_2026-06-07.md`. Both extensions
are OPEN GAPS (potential original contributions).

### LBM route — implementable; cardiac version is novel
The operator expands to `D∇²V + D·(∇T/T)·∇V` = advection-diffusion with **drift
velocity `u_drift = D·∇(lnT)`**. Recipe: fold the drift into the equilibrium,
`f_i^eq = w_i·V·[1 + (e_i·u_drift)/c_s²]` (D2Q5 BGK, single global τ — do NOT use
variable-τ; it gives the non-conservative form and breaks stability at sharp T steps).

- **Direct physical analog [the recipe to copy]:** depth-averaged shallow-water
  transport with variable water depth, `(1/h)∇·(h·D∇C)` — h is literally a
  thickness/cross-section field. **Ru/Liu/Xing/Ding 2021 (CMAME 379:113745)** do
  exactly this: product-rule split → pseudo-velocity `u_pseudo = D·∇(lnh)` in the
  equilibrium, with a **"well-balanced linked-scheme"** correction that cancels the
  spurious `C·∇·u` source. **No public code** — must re-implement from the paper.
- **Conservative-vs-advective pitfall:** adding `u_drift` to `f_i^eq` yields
  `∇·(V·u_drift)`, differing from `u_drift·∇V` by `V·∇·u_drift`, and
  `∇·u_drift = D∇²(lnT) ≠ 0` → spurious reaction term if uncorrected. The Ru et al.
  well-balanced construction is precisely the fix to port.
- **CFL:** `|u_drift|/c_s ≪ 1`; `u_drift` spikes at sharp thickness steps (the block
  regime) → refine Δx or cap ∇(lnT) there.
- Cardiac LBM papers (Rapaka/Mansi LBM-EP 2012; Zettinig 2014; Campos 2015) handle
  heterogeneity only via the relaxation matrix / regional scalar conductivity — none
  implement the conservative thin-layer operator. **Cardiac thickness-weighted LBM is
  unpublished.**

### Bidomain route — pieces exist, the coupled object is unwritten (strongest novelty)
- Closest prior art: **Chapelle–Collin–Gerbeau 2013** (MMMAS 23:2749, DOI
  10.1142/S0218202513500450) — rigorous thin-layer 3D→2D bidomain reduction, but
  fixed-thickness midsurface, not a varying scalar T(x,y) weight.
- Operator form already implemented: **Biasi/Sega 2023** smoothed-boundary bidomain
  (PLOS ONE, PMC10256234) multiplies every operator by a phase field ψ →
  `∇·(ψσ_i∇·)`, `∇·(ψσ_e∇·)` — algebraically identical if ψ→T. Discretization template.
- Eikonal-bidomain with thickness: **Colli Franzone–Guerri–Rovida 1990** (JMB 28:121,
  DOI 10.1007/BF00163143) — eikonal-curvature from bidomain via singular perturbation
  in wall thickness.
- **New content:** in bidomain the thickness gradient enters BOTH parabolic & elliptic
  operators AND the φ_e coupling source; the single monodomain `(∇lnT)·∇V` drift
  becomes a coupled pair whose T-weights must stay consistent for V_m = φ_i − φ_e to
  close. Sanity check: at equal anisotropy ratios bidomain→monodomain, must collapse to
  Biktasheva. Bishop & Plank bath-loading is thickness-INDEPENDENT — a distinct
  mechanism, not this.

### Cross-engine effort ranking
| Engine | Status | Effort |
|---|---|---|
| Monodomain FDM | model settled, 3D-validated (Biktasheva 2015) — our planned next step | lowest |
| LBM | recipe exists (drift in equilibrium); cardiac version novel | medium |
| Bidomain | coupled-T-weight object unwritten — strongest original-contribution claim | highest |

## Theory of the Artifact

- **Source-sink / current-to-load mismatch [theory]:** the wavefront shape is set
  by available source current vs downstream load; insufficient source to charge a
  larger sink → convex/slow/block (Ciaccio 2018).
- **Varying-cross-section cable [theory]:** the cable equation with non-uniform
  cross-section produces impedance/load mismatch at expansions and branch points
  (Goldstein & Rall 1974 lineage). The thickness reduction is the 2-D realization.
- **In-plane source-sink curvature [simulation, ORTHOGONAL]:** Romero/Trenor/
  Ferrero/Starmer 2013 (PLoS ONE, DOI 10.1371/journal.pone.0078328) show curvature
  from non-uniform cellular source-sink dispersion along the wavefront in a STANDARD
  uniform-conductivity 2-D monodomain (TTP06) — confirms the general curvature
  mechanism but does NOT validate the thickness fix.
- **Experimental foundation [experiment]:** Ciaccio 2007 (Heart Rhythm, PMC2626544):
  isthmus IBZ thickness 231±140 µm vs outer pathway 1440±770 µm (~6×, p<0.001);
  CV slower at entrance/exit (0.32 vs 0.42 mm/ms); functional-block lines coincide
  with sharp thin→thick transitions; model predicts circuit features at 75% sens /
  97% spec.

## This Session's Experiments — why 2-D in-plane geometry FAILED (the empirical journey)

All conducted in LBM V1 + Monodomain V5.4 (TTP06 EPI). Scripts in this folder;
figures/videos in `media/source_sink_mismatch_investigation/`.

| Geometry (script) | Result | Why |
|---|---|---|
| eikonal circle (`diag_eikonal_circle.py`) | leading inverse crescent only −0.14 ms; strong trailing crescent | trailing = diffraction shadow (path-length), not source-sink mirror |
| isthmus / strand→expansion (`diag_sourcesink_isthmus.py`) | radial collapse at expansion; **no block** down to 1-cell neck | wide-upstream reservoir feeds neck; high safety factor |
| converging wedge (`diag_triangle_pinch.py`) | **flat CV** (LBM ~76, mono ~56) | planar wave in a wide channel is width-independent |
| hourglass (`diag_hourglass.py`) | constriction flat; dilation radial fan | same |
| diffusion-only (`diag_hourglass_diffusion.py`) | √t creep, no propagation; both engines identical | pure diffusion has no regeneration |
| foot/λ (`diag_foot_lambda.py`) | foot ~0.25–0.5 mm ≈ diffusion-limited / physiological | not mistuned |
| mirror vs iso slanted (`diag_monodomain_slanted.py`) | wall-parallel effect, angle-dependent | boundary BC, not source-sink |

**Root cause:** all of these varied 2-D **in-plane width** — a domain-SHAPE change.
For a planar wave in a wide channel, `∂²V/∂y²≈0` in the bulk → locally 1-D →
**CV = CV₀, width-blind**; a width change is then a boundary deformation, not a
cross-section load. The Ciaccio effect needs the varying dimension to enter the PDE
as a **coefficient** (thickness), or to be in the **thin/cable regime**.

## Regime & Numerical Pitfalls

- **Thin/cable requirement.** Cross-section change loads the bulk wave only when the
  varying transverse dimension is thin vs the electrotonic length: roughly
  `W ≲ sqrt(D·δ/CV) ≈ 1 cell` at our params (δ=foot≈0.05 cm, CV≈0.06 cm/ms,
  D=0.001) — impractically thin in-plane, hence the coefficient-field (thickness)
  route is the practical one. [theory; our derivation, consistent with O(μ²) limit]
- **Open numerical questions** (from deep-research, unanswered in literature):
  the explicit μ-threshold where the thin-layer reduction breaks; the mesh
  resolution needed to resolve a sharp `∇T/T` transition without spuriously
  creating/suppressing block.

## Key Decisions
| Decision | Choice | Rationale |
|---|---|---|
| Model for the artifact | thickness-weighted monodomain `(1/T)∇·(T·D∇V)` | rigorously derived + 3D-validated (Biktasheva PRL 2019); gives Ciaccio's ∇T/T |
| Thickness as coefficient field, not 3rd dimension | 2-D T(x,y) | area is the cross-section analog; no extra grid axis (PI's framing, corrected) |
| Parameter set | adopt 2018 (θ₀=0.4, D=0.2, thr≈2) as primary | matches canine empirical ~1.55; revisit vs 2015 (D=0.1) |
| Citation | Biktasheva/Dierckx/Biktashev, NOT Bishop & Plank | B&P is bath-loading, a different mechanism |

## Open Questions
- Implement thickness-weighted monodomain with TTP06/ORd and reproduce Ciaccio
  Fig-4 A–D + block, validated vs full 3-D? (the gap / original contribution)
- Adopt 2018 (D=0.2) or 2015 (D=0.1) parameterization — fitted assumption or model revision?
- Precise μ (thickness/space-constant) threshold where the thin-layer reduction fails?
- Mesh-resolution pitfalls at a sharp thickness transition?

## Sources (deep-research, primary peer-reviewed)
- Ciaccio 2018 JACEP — DOI 10.1016/j.jacep.2017.08.019 (PMC5874259); Ciaccio 2015 Comput Biol Med (PMC4533242); Ciaccio 2007 Heart Rhythm (PMC2626544); Ciaccio 2014 Circ AE (10.1161/CIRCEP.113.000840)
- Biktasheva/Dierckx/Biktashev — PRL 2015, 114:068302, DOI 10.1103/PhysRevLett.114.068302 (arXiv:1408.3654) — thickness reduction + 3D validation [year corrected from "2019"]
- **Ru/Liu/Xing/Ding 2021** — CMAME 379:113745, DOI 10.1016/j.cma.2021.113745 — well-balanced LBM for depth-averaged ADE with variable depth `(1/h)∇·(h·D∇C)`: the direct LBM analog (no public code)
- Yoshida & Nagaoka 2014 — JCP 257:884, DOI 10.1016/j.jcp.2013.09.035 — curvilinear-coordinate CDE LBM (backup; was mis-cited as "Yang 2014")
- BeatBox: Antonioletti et al. 2017, PLOS ONE 12(5):e0172292, DOI 10.1371/journal.pone.0172292 — validation engine, vendored at `Research/code_examples/beatbox/`
- Bishop & Plank 2011 — PMC3075562 — bath-loading augmented monodomain (the DISTINCT tool)
- Romero/Starmer 2013 — DOI 10.1371/journal.pone.0078328 — in-plane source-sink curvature (orthogonal)

## Connections
- **boundary_conduction_speedup** (parent): specular BC speedup, crescent/HBB taxonomy, κ-accumulation — the boundary-BC thread, kept there.

---

## ── Source-Sink Theory (copied from boundary_conduction_speedup, 2026-06-07) ──

> Foundational source-sink-mismatch theory developed in the parent
> `boundary_conduction_speedup` question (storage-tank / John Zimmerman model
> thread). Copied here verbatim as the conceptual foundation of this
> investigation; it also remains in the parent (intertwined with the
> storage-tank boundary work). Original dates retained inline.

## PDE Formulations of the Effect

### Storage-tank analog (Zimmerman, 2026-04-24)

PI John Zimmerman shared a discrete storage-tank simulation (`../../../simulation/storagetanks.py`, filed at repo root) that
exhibits a qualitatively similar boundary speedup. Each tank on a 2D Moore-neighbourhood grid
pumps a source-state-dependent amount `max_pump·√((u−θ)/(u_max−θ))` through every channel
leading to a lower-volume neighbour, gated on `u > θ`. Interior tanks have 8 open channels,
edge tanks 5, corner tanks 3 — fewer channels at the boundary means a fired tank retains
potential longer, sustaining drive to the remaining neighbours along the edge.

This rule is **non-Fickian**: the flux across each link depends only on the source state,
not on the gradient (u_i − u_j). The acceptor's state appears only through a Heaviside gate.
So the total outflow from a source scales with its number of sinks — the geometric source
of the boundary asymmetry.

### Why the plain heat equation cannot reproduce it

For standard reaction–diffusion with Neumann BC:

```
∂u/∂t = D ∇²u + f(u),     ∂u/∂n = 0 on ∂Ω
```

In the *continuum*, the wall reflects flux but preserves tangential symmetry. 1D wave
speed c = 2√(D·f'(0)) is set by bulk coefficients and is the same at the boundary as in
the interior. The monodomain control experiment (0.000 cm deviation from flat) is the
empirical confirmation.

In the *discrete* setting, however, the no-flux Neumann BC is implemented either as
zero-pad (boundary cells have fewer non-zero stencil entries) or as ghost-reflection
(boundary cells get duplicated upstream values) — and these produce *different signs*
of boundary effect. See `Discrete-lattice boundary effects` below for the full
decomposition. The continuum result lives in between these two discrete extremes.

### Three candidate modifications

In rising fidelity:

**(A) Heat + state-dependent loss** — pedagogical / transparent
```
∂u/∂t = D ∇²u + f(u) − γ(x) u
γ(x) = γ₀ · [1 − exp(−d(x)/λ)]
```
`d(x)` = distance to boundary, `λ` = electrotonic length. Bulk: full dissipation. Edge: γ→0.
Speed ratio ≈ √(1 + γ₀·τ_rxn), parameter-dependent (doesn't land on 1.131 by itself).

**(B) Heat with spatially-varying diffusivity** — canonical Kleber model
```
∂u/∂t = ∇·(D(x) ∇u) + f(u)

D(x) = D_bulk + (D_bdry − D_bulk) · exp(−d(x)/λ)
D_bulk = σᵢσₑ / (σᵢ+σₑ)        (harmonic mean)
D_bdry = σᵢ                     (bath shorts out σₑ)
```
This is precisely the monodomain reduction of bidomain under the quasi-static assumption,
with the boundary correction encoded in D(x). Linearised traveling-wave speed c ∝ √D gives:
```
CV_boundary / CV_interior = √(σᵢ / D_bulk) = √((σᵢ + σₑ) / σₑ)
```
For human ventricle longitudinal (σᵢ=1.74, σₑ=6.25): = √(7.99/6.25) = **1.131** — matches
the theoretical target the Bidomain V1 engine has been converging toward (1.0714 at dx=0.025).

**(C) Non-local (peridynamic) heat** — most faithful to John's tank rule
```
∂u/∂t = ∫_{B(x,R)} K(|x−x'|) [u(x') − u(x)] · 𝟙_Ω(x') dx' + f(u)
```
The domain indicator `𝟙_Ω(x')` truncates the kernel at ∂Ω. In the interior this reduces to
`D∇²u` with D = (∫|x'|² K dx')/2d. At the boundary the self-coefficient on u(x) shrinks
(less local dissipation) and the truncated support's centroid shifts inward (inward drift).
Direct continuum analog of "8 nbrs → 5 → 3" in John's toy.

### Recommendation

**Use (B) as the single-field PDE that captures the Kleber effect.** It is:
- The simplest modification of the heat equation,
- Mechanistically interpretable (extracellular short-circuit = local diffusivity jump),
- Quantitatively correct (reproduces the measured 1.131 ratio),
- Derivable as the quasi-static bidomain reduction (Bishop & Plank 2011 augmented monodomain).

For the anisotropic sub-question, (B) generalises by replacing scalar D(x) with a spatial
diffusivity **tensor D(x)**. The eikonal-limit wave speed becomes direction-dependent:
`c(x, n̂) = 2√(n̂ · D(x) · n̂ · f'(0))`, which predicts the fiber-parallel vs perpendicular
boundary-layer profiles we're about to measure.

### Discrete-lattice boundary effects: Effect A, Effect B, Effect B′

Investigation of John's storage-tank model on a 2D 80×50 Moore-neighbourhood grid with
inlet/outlet line geometry (`simulation/`) decomposes the boundary asymmetry into two
intrinsic effects plus one that's induced by the boundary operator.

**Effect A — geometric inflow deficit.** When a planar wavefront propagates rightward,
an interior tank at column N receives drive from 3 fired upstream neighbours (the (N-1)
column at y-1, y, y+1). An edge tank at column N receives from only 2 upstream neighbours
(one of the diagonals doesn't exist). This deficit is purely geometric — it appears in
*every* nearest-neighbour-coupled lattice with a no-flux wall, regardless of pump rule.
Effect A pushes the per-column LAT shape toward an *inverted-U / crescent* (interior
ahead of edge), i.e. boundary slowdown.

**Effect B — outflow dividend / sustained source.** Once a tank has fired, an interior
tank drains into 5 unfired downstream neighbours, an edge tank into only 3. The edge
therefore retains volume better, stays above threshold longer, and integrates more total
drive into whatever comes next. Effect B pushes toward *U-shape / camel toe* (edge ahead),
i.e. boundary speedup. **Effect B requires a non-self-limiting flux rule**: under a
gradient-driven (Fickian) rule, the receiver's rising V suppresses the per-channel pump
rate before the integrated-drive advantage can accumulate, and Effect B is killed.

**Effect B′ — mirror-duplication enhancement (induced by reflection BC).** When the
boundary operator is *reflection-padded* (`np.pad(V, mode='reflect')`), each boundary
cell sees 8 channels but only 5 unique upstream cells: the y=1 row contributes both via
the real channel and via the ghost channel that mirrors it. The boundary cell receives
*double drive* from 3 of its real neighbours. This is far stronger than Effect A and
overwhelms it. Both pump rules then produce a massive camel toe under reflection BC.

**Boundary-operator dominance.** The boundary operator (how the wall handles missing
neighbours) determines the sign of the boundary effect *more strongly than the pump rule*:

| BC choice                       | constant rule LAT shape   | gradient rule LAT shape   |
|---------------------------------|---------------------------|---------------------------|
| zero-pad (no-flux Neumann)      | crescent + transient camel | pure crescent (mono.)    |
| reflection (mirror enhancement) | massive camel toe         | massive camel toe         |

**Pump-speed Goldilocks zone (parameter sensitivity).** The drainage effect's
ability to overcome the inflow effect depends on *timescale matching*. The camel
toe magnitude is non-monotonic in `max_pump` for the constant rule (line geometry,
80×50, 4000 steps):

| max_pump | Δ@x=18 | shape         |
|----------|--------|---------------|
| 2        | +53    | crescent      |
| 5        | −8     | camel         |
| 10       | −12.5  | camel (peak)  |
| 15       | −1.5   | weak camel    |
| 20       | 0      | flat          |
| 30       | +3     | weak crescent |

Mechanism: camel toe requires the *drainage timescale* (steps a fired source
stays above threshold) to be comparable to the *inflow timescale* (steps the
downstream tank takes to fire). When the two match, the edge's slightly slower
drain (5 outflow channels vs 8) accumulates a meaningful integrated head-start.
Below the resonance, the wavefront passes before the drainage advantage has any
effect; above it, the column fires nearly simultaneously so the drainage delta is
negligible. John's effective `max_pump = 10` is at the camel-toe peak.

For the gradient rule, magnitude scales monotonically with 1/k (faster k →
smaller absolute time delays) but the *shape stays crescent at every k*. The
self-limiting flux makes the drainage advantage impossible in principle.
See "Equilibrium argument" below for the structural proof and empirical
k-sweep confirmation.

### Equilibrium argument: why Fickian is sign-locked to crescent at every k (2026-05-02)

Take a fired source cell mid-wavefront. Upstream column at V_up, downstream
column ≈ 0, lateral N/S gaps ≈ 0 in the y-uniform regime. Under Fickian:

```
dV/dt = k·N_in·(V_up − V) − k·N_out·V

V*(y) = [N_in / (N_in + N_out)] · V_up   (steady state)
τ(y)  = 1 / [k·(N_in + N_out)]            (time constant)
```

In the moore8 + zero_pad lattice with one_way pipes and uniform y, both edge
and interior cells have N_in = N_out (3,3 interior; 2,2 edge), so:

```
V*(edge)     = (2/4)·V_up = V_up/2
V*(interior) = (3/6)·V_up = V_up/2
ratio        = 1            ←  IDENTICALLY, independent of k
```

**Edge plateau equals interior plateau.** No y-asymmetric stockpile exists
to discharge. The time constant τ does differ — τ_edge = 1/(4k) takes 50%
longer to reach V* than τ_interior = 1/(6k) — but that is just Effect A
re-stated as a charging time, not a differential downstream-pumping
advantage. Once V*(y) is reached, the per-pipe outflow rate k·(V* − V_down)
is the same at edge and interior; total downstream pumping
= k·V*·N_out scales with N_out, so the edge actually pumps LESS total
fluid into the next column than interior does. Effect B doesn't merely
vanish under Fickian — it inverts and *reinforces* Effect A.

Replacing k → α·k everywhere rescales τ uniformly (isochrones stretch by
1/α) but leaves the V*(y) ratio at 1. **No k can produce a non-unity
ratio.** This is structural, not parametric.

### Capacitor vs resistor mnemonic

```
                    John's constant rule        Fickian gradient rule
─────────────────────────────────────────────────────────────────────────
Phase structure     Fill → dwell → drain        No phases — asymptotic
Stockpile high V?   YES (parks near V_max)      NO (asymptotes to V_up/2)
Drain depends on V_down?  No (until cap kicks)  Yes (linear in gap)
Edge advantage      Slower drain → more         No advantage — V* is
                    integrated downstream       y-independent and total
                    pumping → camel toe         pumping ∝ N_out
                    possible                    favours interior
Behaves like…       Capacitor discharging       Resistor in steady state
─────────────────────────────────────────────────────────────────────────
```

Capacitors hold a stockpile that asymmetric drainage can release on its
own schedule. Resistors carry no stockpile to release asymmetrically.
Effect B is the capacitor's discharge bonus; it cannot exist in a
resistor network.

### Empirical confirmation (2026-05-02 k-sweep)

`Nx=80`, `Ny=50`, `gradient` mode, `moore8` connectivity, `zero_pad`,
`one_way`, line stim, `threshold=45`, sweeping k. Edge−center LAT (steps,
positive = edge fires LATER = crescent):

```
   k     x=10   x=20   x=30   x=40   x=50   x=60
 0.200    +12    +15    +22    +28    +33    +38
 0.120    +18    +43    +72   +105   +132   +159
 0.080    +22    +56    +92   +127   +161   +192
 0.040    +41   +100   +160   +214   +265    --
 0.020    +79   +185   +290    --     --     --
 0.010   +155   +357    --     --     --     --
 0.005   +307    --     --     --     --     --
```

Sign locked positive at every k from 0.005 to 0.20 (40× range).
Magnitude scales as 1/k as predicted by Effect A combined with
wave-slowing dilation. (`--` columns indicate the wavefront stalled
before reaching that x — see Finding 1 in IDEALOG re finite-distance
propagation under threshold gating.)

### Generalisation

The equilibrium argument generalises to **any** flux law of the form
`q-dot = f(V_src − V_dst)` with f(0) = 0 and f monotonic. Solving
dV/dt = 0 in the y-uniform regime gives V*(y) = N_in/(N_in+N_out)·V_up,
which is y-independent whenever N_in = N_out at every cell. Any such law
can only manifest Effect A (geometric inflow deficit) and is sign-locked
to crescent. The bias is baked into the diffusion operator's structure,
not the rate-law parameters.

**Pipe directionality is a third axis.** Even within the constant rule under zero-pad
BC, the *transient* camel toe in mid columns disappears if pipes are made bidirectional
(both A→B and B→A fire when their respective sources are above threshold). With
bidirectional pipes the rule becomes self-limiting (net flow = f(V_A) − f(V_B), which
vanishes as V_B catches up), which kills Effect B. Only one-way pipes preserve Effect B.

So the full causal picture:

| axis                    | options                          | controls                 |
|-------------------------|----------------------------------|--------------------------|
| boundary operator       | zero-pad / reflect-y / reflect-all | sign of effect (A vs B′) |
| pipe directionality     | one-way / bidirectional          | existence of Effect B    |
| pump-rule rate law      | constant / gradient / other      | magnitude only           |

Camel toe in this model requires (zero-pad BC) AND (one-way pipes). Either modification
on its own removes it. The pump rate law (sqrt vs linear vs other) only affects the
size of the effect.

Mapped to the cardiac dichotomy:
- Zero-pad ↔ Neumann ↔ monodomain control → boundary slowdown (matches our 0.000 cm
  monodomain control).
- Reflection / enhanced inflow ↔ partial bidomain analog ↔ Kleber-style camel toe.

The *pump rule* (constant vs gradient, source-limited vs Fickian) modulates the
*magnitude* of the effect but the *boundary-operator choice* fixes its *sign*.

**Operator-level argument (state-independent).** A perfectly uniform initial wavefront
should not produce any boundary effect by symmetry — but it does. The asymmetry doesn't
come from initial conditions; it comes from the discrete update operator U being
*not translation-invariant in y* near the wall. Even with perfect ICs, edge rows of U
have fewer non-zero entries than interior rows, so uniform input → non-uniform output
on the very first step. Effect A is baked into U at the operator level. Effect B′ is
baked into U via reflection padding. Both are state-independent properties of the
boundary operator, not transients of the simulation.

## John's per-cell physics derived from first principles

John's pump rate `max_pump · √((V_C − θ)/(V_max − θ))` is **textbook Torricelli**
for a single tank with outlet hole at height θ draining to atmosphere. From Bernoulli
(free surface at h_C, P=P_atm, v≈0) to pipe outlet (height θ, P=P_atm, velocity v):

```
g·h_C = g·θ + v²/2  ⇒  v = √(2g·(h_C − θ))
Q = a · v = a · √(2g · (h_C − θ))
```

Identifying `a · √(2g) ≡ max_pump / √(V_max − θ)` (with unit cross-section) gives John's
normalized form. The √ and threshold are not arbitrary modeling choices — they are
energy conservation (potential head → outlet kinetic energy) and outlet-hole geometry
respectively. **Single-cell physics is correct; no modification needed.**

For a *submerged pipe* between two tanks (h_C > θ AND h_i > θ), Bernoulli on
free-surface to free-surface gives v = √(2g·(h_C − h_i)) — **gap-driven, not source-driven**.
John applies his single-cell law to this regime too, which is incorrect: he over-estimates
the rate by √((h_C − θ)/(h_C − h_i)), which can be huge at small gaps. His quarter-gap
damping clamp is a crude LINEAR approximation to the missing Bernoulli √-of-gap law.

**Unified hydrostatic-faithful form:**

```
rate(C → i) = max_pump · √( max(V_C − max(θ, V_i), 0) / (V_max − θ) )
```

Reduces to John's law when V_i ≤ θ (Torricelli, atmospheric outlet); reduces to Bernoulli
√-gap when both above θ (submerged outlet). Single √-formula, no clamp needed, monotone
equilibrium approach (no overshoot).

**Continuous physics, numerically discretized.** John's per-cell ODE
`dV/dt = −max_pump · √((V−θ)/(V_max−θ))` for V > θ is analytically solvable —
empties V₀=V_max to threshold in `t* = 2·(V_max−θ)/max_pump = 11` steps for the
default parameters. His simulation code is forward Euler with dt=1; the damping clamp
is a stability hack for the multi-cell coupled regime, not a physics feature.

## Single-cell mechanism: what role does V_C play?

For one cell C with 8 Moore neighbors numbered 1..8, each link (C, i) carries two
populations: φ_i⁺ (C→i outflow) and φ_i⁻ (i→C inflow). All rule variants share the
same Jacobi-buffered update; they differ only in *firing conditions* and *rate*:

| variant | φ_i⁺ fires iff | φ_i⁻ fires iff | rate (φ_i⁺) |
|---|---|---|---|
| **John** (const+1way+damp) | V_C>θ ∧ V_C>V_i | V_i>θ ∧ V_i>V_C | min(max_pump·f(V_C), (V_C−V_i)/4) |
| const + 1way + no-damp | V_C>θ ∧ V_C>V_i | V_i>θ ∧ V_i>V_C | max_pump·f(V_C) (uncapped) |
| const + bidirectional | V_C>θ | V_i>θ | max_pump·f(V_C) |
| gradient + 1way | V_C>θ ∧ V_C>V_i | V_i>θ ∧ V_i>V_C | k·(V_C − V_i) |
| gradient + bidirectional | V_C>θ | V_i>θ | k·(V_C − V_i) |

where f(V) = √((V−θ)/(V_max−θ)). The one-way gate enforces mutual exclusion of
{φ_i⁺, φ_i⁻} per link; bidirectional drops it.

**The non-LBM ingredient.** Standard LBM diffusion has equilibrium f_i^eq = w_i · ρ —
*linear* in density. John has rate ∝ √(V_C − θ) — *concave* in source state. The
concavity is the necessary ingredient for Effect 1 (drainage advantage): high-V cells
pump disproportionately hard, so a boundary cell that retains V longer keeps driving
downstream pumps even after equivalent interior cells have equilibrated. Linear rate
laws (LBM, gradient rule) cannot produce camel toe by this mechanism. To give an LBM
setup John-like speedup, the recipe is **f_i^eq concave in ρ** (not standard cardiac
LBM, but a clean test target).

**Threshold step function as asymmetry amplifier.** The hard step at θ creates two
clean regimes per cell — accumulation (V_C below θ: inflow only, no outflow) and
pumping (V_C above θ: full √-rate). The switch at threshold-crossing is binary, not
gradual. This separation lets each effect express cleanly:

- Accumulation phase: inflow channel deficit (5 boundary vs 8 interior) acts unopposed
  by self-leakage. Effect 2 amplitude maximized.
- Pumping phase: outflow channel deficit lets boundary cells retain V to higher levels.
  Effect 1 amplitude maximized.

Smooth-onset variants (sigmoid, leaky integrator) weaken both effects because cells
continuously self-leak while accumulating, never reaching as high a stored V to release
at firing. **Predicted camel-toe magnitude ordering: step > smooth-ramp > no-threshold**.

## John's axiom set: cardiac claims vs model implementation

A central distinction for evaluating his "boundary speedup" claim: not every feature
of his Colab simulation represents what he would defend as a property of cardiac
tissue. Separated into two tiers.

### Tier I — Genuine cardiac axioms (defendable in heart literature)

**I.1  Discreteness matters at the cell scale.** Cardiac tissue is a network of
discrete coupled cells, and that discreteness has consequences not captured by the
continuum PDE limit. (Aligned with Spach group's microscopic-discontinuity tradition.)

**I.2  Sub-threshold accumulation.** Cells integrate input over time below their
firing threshold, with the integrated state persisting between events. Functional
form is open — could be hard step, soft sigmoid, leaky integrate-and-fire — the
commitment is to *integration*, not which functional form.

### Tier II — Model implementation features (NOT cardiac claims)

These are choices made for tractability or borrowed from the water-tank metaphor.
John would not defend any of them as biological.

| # | feature | source / motivation |
|---|---------|---------------------|
| II.1  | Torricelli √-law `√(V−θ)` | water-tank hydrostatics |
| II.2  | Source-state-only coupling | water-tank metaphor |
| II.3  | Hard step function threshold | implementation simplification |
| II.4  | Hard one-way valve at gap-junction level | implementation simplification |
| II.5  | Moore-8 dense connectivity | lattice convenience |
| II.6  | Square lattice geometry | geometric convenience |
| II.7  | Synchronous Jacobi update | numerical scheme |
| II.8  | Quarter-gap damping clamp | numerical stability hack |
| II.9  | Memoryless cells (no recovery) | radical simplification |
| II.10 | No-flux Neumann boundary | default |

### Three-question evaluation program

```
   Q1 — SENSITIVITY: does each axiom (Tier I or II) produce a boundary artifact
        in the toy model?
        (Already partly characterized for II.4 bidirectional, II.10 reflect-y BC,
        and the rate-law axes via gradient rule.)

   Q2 — ROBUSTNESS: does Tier I ALONE, with cardiac-realistic Tier II replacements,
        still produce camel toe?

        Cardiac-realistic replacements:
          II.1 + II.2 → linear ohmic gap junction:  I_ij = g·(V_i − V_j)
          II.3        → smooth sigmoid threshold (or FHN cubic recovery)
          II.4        → bidirectional coupling (refractoriness lives in membrane
                         kinetics, not the gap junction)
          II.5 + II.6 → anisotropic sparse connectivity (along-fiber dense, sparse
                         cross-fiber)

        IF YES → boundary speedup is a Tier-I consequence; cardiac defense reduces
                 to defending I.1 + I.2.
        IF NO  → speedup depends on Tier-II artifacts John doesn't claim as cardiac.
                 The boundary effect is a model artifact, not a cardiac prediction.

   Q3 — CARDIAC TRUTH of I.1 + I.2: defend or reject from biology (gap-junction
        density, optical mapping at tissue edges, Spach/Kleber literature).
```

**Prior on Q2 outcome.** The biophysically suspect axioms (II.1 Torricelli √, II.2
source-state-only) are exactly what produce Effect 1 (drainage advantage). The
defensible Tier-I axioms (I.1, I.2) plus cardiac-realistic Tier-II at most support
Effect 2 (inflow deficit → crescent / slowdown). Predict: under cardiac-realistic
Tier-II, the boundary effect *flips sign* relative to John's setup — slowdown,
not speedup.


## Connectivity is the smoking gun: 8-neighbour vs 4-neighbour ablation (2026-04-29/30)

The boundary-effect mechanism in John's storage-tank model was localised by
running a 6-way ablation on the user's "Fickian-modified" John setup
(`gradient` mode + `one_way` + `zero_pad` + line geometry).

### Setup

Two ablation knobs added to `tanks_vec.run()`:
- `connectivity` ∈ {`moore8`, `cardinal4`, `moore8_iso`}
- `threshold_gate` ∈ {True, False} — if False, drops the `fired_p` (V > θ)
  gate from pipe-firing condition.

### Findings

```
Run                      Connectivity    Threshold    max|LAT-meanY|    cols_full
─────────────────────────────────────────────────────────────────────────────────
R1   baseline            moore8          True         91.8 steps         42
R2   cardinal-4 only     cardinal4       True          0.0 steps         25
R3   no threshold only   moore8          False        11.5 steps         33
R4   both off            cardinal4       False         0.0 steps         20
R5   iso 4:1             moore8_iso      True         81.0 steps         25
R6   iso, no thresh      moore8_iso      False        16.5 steps         20
```

**Two clean conclusions:**

1. **Cardinal-4 connectivity gives EXACTLY ZERO crescent** in y-uniform line
   stim, regardless of threshold gate (R2/R4 both 0.0 to floating-point
   precision). The "missing N pipe at the boundary" contributes gap=0 in
   y-uniform fields, so losing it costs nothing.

2. **Moore-8 connectivity ALWAYS produces a crescent** (R1/R3/R5/R6 all
   non-zero). Threshold gate amplifies by ~8× (91.8/11.5) under uniform
   weights. **Moore-8 is the necessary structural ingredient; threshold
   amplifies but is not required.**

### Mechanism

In y-uniform field with wavefront at column k, boundary cell at (0, k) loses
its NW and NE diagonals (off-grid). Each interior cell has 3 firing inflow
pipes (NW, W, SW from column k-1) and 3 firing outflow pipes (NE, E, SE
to column k+1). Boundary loses one inflow + one outflow pipe → 2/3 charging
rate of interior. Crescent forms.

In cardinal-4: boundary only loses N pipe (which has gap=0 in y-uniform), so
no deficit. The diagonals carry x-direction flux even in y-uniform fields
because they span both axes simultaneously — that is exactly the
mechanism the cardinal-only stencil cannot create.

### Iso 4:1 (Patra-Kałuża) reduces but does not eliminate

Implementing the Patra-Kałuża isotropic 9-point stencil weights (cardinal × 4,
diagonal × 1, with the canonical 1/6 normalisation prefactor — initial
implementation forgot the prefactor and produced D_eff = 6k = 0.48,
violating the 2D-explicit CFL limit of 0.25, manifesting as grid-scale
mosaic instability) gives:

```
Boundary deficit ratio = (w_c + w_d) / (w_c + 2·w_d)

  Equal-weight Moore-8 (1, 1):    2/3   = 0.667    (33% deficit)
  LBM / iso 4:1     (4/6, 1/6):   5/6   = 0.833    (17% deficit)
  Cardinal-only        (1, 0):    1     = 1.000     (0% deficit)
```

Empirically R5 vs R1: 81.0 vs 91.8 steps — **only ~12% reduction in
crescent magnitude**, exactly matching the 5/6 deficit prediction modulo
threshold-amplified compounding. **Iso 4:1 weighting alone is necessary
but not sufficient** to eliminate the boundary effect. Full elimination
requires either cardinal-4 (no diagonals at all) OR a custom boundary
treatment (face_mirror_iso on the PDE side, specular reflection on the
LBM side) that restores real upstream-V information to the diagonal
ghost / population slots at the wall.

### LBM connection (corrected 2026-05-14)

LBM D2Q9 weights are 4/9 (rest), 1/9 (cardinals), 1/36 (diagonals).
Cardinal:diagonal ratio = 4:1, identical to Patra-Kałuża isotropic 9-point.
**LBM halfway bounce-back (HBB) is structurally equivalent to face_mirror**
(not face_mirror_iso). Both kill upstream-V contribution from diagonal
channels at the wall:
- face_mirror: `V_NW_ghost = V_self` → gap = 0 → no diagonal Laplacian contribution
- HBB: `f_SE(C, after) = f_NW(C, before)` → diagonal slot gets C's own pre-stream NW emission, which is local equilibrium (≈ `w_NW · ρ_C`) and carries no upstream V information

Different bookkeeping, same physical Neumann boundary condition, same structural deficit. The LBM bounce-back family inherits the forward sign-lock from face_mirror, and the magnitude of the deficit follows the cardinal:diagonal weight ratio (canonical 4:1 → 5/6 mild; uniform_8 1:1 → 2/3 full). The LBM analog of face_mirror_iso (zero deficit, no sign-lock) is **specular reflection**, in which diagonal mass crosses the wall to the adjacent cell's row-aligned slot and carries real upstream-V information from the upstream-boundary's diagonal emission. See §"Three BC families" for the full mapping.

