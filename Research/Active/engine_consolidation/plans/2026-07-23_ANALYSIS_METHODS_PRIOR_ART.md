# Analysis Fields & EP Metrics — Standard Math & Prior Art

> Companion to [ANALYSIS_FIELDS_DESIGN.md](./ANALYSIS_FIELDS_DESIGN.md). For **every** planned feature
> (the `analysis.fields` branch, the scalar EP metrics, `single_cell()`, and the canonical LAT), this doc
> records the **standard mathematical definition**, **how others already implement it** (tools + primary
> literature), and a **verdict for our design** (matches / adjust / adopt-new). Compiled 2026-07-21 from a
> six-lane literature + implementation survey (LAT/CV · curvature/rotors · source–sink/safety-factor ·
> discrete vector calculus · scalar EP metrics · 0-D single-cell). Equations in ASCII/Unicode.
>
> **Bottom line:** most of our planned definitions are exactly the field-standard ones. The survey changed
> five things — see § 0.

---

## 0. Cross-cutting conclusions (the load-bearing changes)

These five results cut across many features and should be settled before building:

1. **Canonical LAT = linearly-interpolated V_m crossing at ONE frozen threshold (recommend −40 mV), torch/on-device,
   first-crossing-per-beat.** The physiological gold standard for *transmembrane* activation is max(dV/dt), but
   for the inverse-gradient CV field a quantized nearest-frame LAT produces the flat-neighbour singularities
   Cantwell 2015 flags as the dominant finite-difference CV error. Keep `activation_time_interp`'s machinery,
   drop `activation_time`'s nearest-frame default, route `r.lat()` + `r.cv()` + eikonal + all LAT-based fields
   through it. Expose max(dV/dt) as a secondary "physiological reference" at probe points. (Closes the OPEN LAT
   BIG ISSUE. Detail § 1.)

2. **One staggered adjoint numerical core (`div = −grad*`).** `div(grad)` of naïve *collocated* central differences
   is NOT the compact 5-point Laplacian — it's the wide 2h stencil with a checkerboard null-space. A staggered
   half-step pair (forward-difference to faces / backward-difference from faces) makes `div(grad) == laplacian`
   bitwise AND makes the discrete divergence theorem exact, so `∬div = ∮F·n` is a *free* consistency test, not
   just a convergent one. (Detail § 4.)

3. **`boundary_mode` MUST be a ghost/mirror pad matching the solver — never `numpy.gradient` one-sided edges.**
   A one-sided edge derivative silently assumes nonzero normal flux, contradicting the solver's Neumann BC exactly
   at the reduced-sink tissue edge where the boundary-conduction physics lives. Default `face_mirror` ≙ scipy
   `reflect` (half-sample symmetric). Internal scar/hole = openCARP's "sever-the-connection" no-flux. (Detail § 4.)

4. **One winding-number primitive, reused three ways.** `∮(loop) = ±2π·integer` is the SAME operator on three
   fields: phase φ (→ phase singularities / rotors), velocity v (→ circulation Γ / vorticity), and the isochrone
   tangent (→ Gauss–Bonnet ∮κ ds = 2π = rotor count). Implement one wrapped-loop-sum; CCW = positive. This
   collapses `vorticity` + `circulation` + `winding_number` + Gauss-Bonnet into a single core. (Detail § 2.)

5. **Our `source_sink = ∇·(D∇Vm)` field IS the numerator of the modern safety factor.** Boyle & Vigmond 2010
   define SF from exactly this diffusion term over a single-cell threshold-charge curve. So building the field
   first gives us (a) the source–sink map, (b) the divergence-theorem boundary integral + its self-check, and
   (c) most of a safety-factor implementation, for free. (Detail § 3.)

### Feature → verdict at a glance

| Planned feature | Standard? | Verdict |
|---|---|---|
| `voltage_gradient` ∇Vm, `voltage_flux` D∇Vm | yes | ✅ as designed |
| `source_sink` ∇·(D∇Vm) | yes | ✅ as designed — also the SF numerator |
| `current_flux` −σ∇φ_e, `electric_field` −∇φ_e | yes | ✅ exact standard bidomain defs |
| `velocity` ∇LAT/\|∇LAT\|², `direction` n̂, `speed` 1/\|∇LAT\| | yes (eikonal) | ✅ correct; **§ 7** — Bayly k=5 patch + residual `quality`, divergence gating, GP `velocity_std` option |
| `conduction_time` ∫∇LAT·dl, `isochrone_spacing_CV`, path-CV | yes | ✅ **§ 7.5** — integrate SLOWNESS ∇LAT (not `velocity`); ∮∇LAT·dl=0/=CL self-checks; Δs/Δt≡1/\|∇LAT\| |
| `curvature` ∇·n̂ | yes | ✅ correct; validate vs CV=CV0−D·κ |
| `vorticity` curl(v) | yes | ✅ ; fold into the winding-number primitive |
| `grad`/`div`/`curl`/`laplacian` | yes | ⚠️ ADJUST — staggered adjoint core, not collocated |
| circulation / winding / Gauss-Bonnet / net-flux integrals | yes | ✅ ; one loop-sum + exact divergence theorem |
| isochrone extraction | yes | ✅ marching squares (skimage semantics) |
| canonical **LAT** | — | ⚠️ ADOPT-NEW — interp −40 mV, one threshold, +max-dV/dt reference |
| `wavelength` λ=CV·ERP | yes | ✅ ; ERP default, CV·APD warned proxy, ÷1000 unit fix |
| `apd` (%-excursion, dome-aware) | yes | ✅ ; fix multi-beat baseline |
| `erp` | protocol | ⚠️ ARCHITECTURAL FORK — S1S2+bisection, must run sims |
| `di` = BCL−APD | yes | ✅ |
| `safety_factor` | yes | ➕ NEW — Boyle–Vigmond SF_VB first, needs 1× single-cell Q_thr curve |
| `single_cell()` | yes | ✅ ; route through the tissue reaction step (fixes ORd bug) |

---

## 1. Local Activation Time (LAT) & Conduction Velocity

### Standard math
Activation-time criteria, in order of physiological correctness for a **transmembrane** signal:
- **max(dV/dt)** (peak upstroke) — the reference standard; coincides with peak I_Na (true depolarization
  instant). For unipolar electrograms the equivalent is **max(−dφ_e/dt)**.
- **Threshold crossing** V > θ — pragmatic; valid only if θ sits on the steep Na⁺ upstroke. Bias vs max(dV/dt)
  is a near-common offset that largely **cancels in the spatial gradient**, so it is fine for a CV field *if the
  same iso-level is used at every node*.
- **Interpolated threshold crossing** — linear sub-frame estimate of the crossing time:
  `t* = t_k + Δt·(θ − V_k)/(V_{k+1} − V_k)`.

CV from the activation surface T(x,y) via the eikonal identity `|∇T| = 1/c`:
`speed = 1/|∇T|`, `direction n̂ = ∇T/|∇T|`, `velocity v = ∇T/|∇T|²`. (Our no-minus sign is correct: with T
increasing along propagation, v already points forward.)

CV-estimation methods (Cantwell 2015 error ranking): raw finite-difference is least robust (fails on flat/coarse
LAT); **local polynomial-surface fitting (Bayly 1998)** is the recommended general method — robust to noise,
smooths through flat pixels, yields a full vector field + a fit-residual quality metric. Triangulation / cosine /
RBF are for sparse/irregular electrode layouts.

### Prior implementations
- **openCARP** `LATs`: threshold crossing, default **V_m = −10 mV, positive slope** (repolarization −70 mV);
  derivative (max dV/dt) mode available; `-lats[i].all=1` records every crossing. `tuneCV` measures 1-D CV as
  `(x₁−x₀)/(T₁−T₀)` and uses `CV ∝ √σ`.
- **Finitewave** (Python FDM, our closest analog): `ActivationTimeTracker` = threshold **−40 mV**, first crossing,
  no interp (≈ our `activation_time` but at −40); `LocalActivationTimeTracker` = multi-beat with back-cross re-arm;
  `Velocity2DCalculation` = ellipse-fit front speed (not inverse-gradient).
- **OpenEP/openep-py**: CV by triangulation / Bayly-polynomial / RBF over *interpolated* LAT; a **divergence gate**
  excludes wave-collision (div<0) and focal (div>0) regions where single-wave CV is invalid.
- **ElectroMap** (optical): LAT = max dV/dt default; CV = multi-vector 5×5 window.

### Verdict for our design
Converge the two LAT defs onto **one canonical LAT = linearly-interpolated V_m = −40 mV crossing, first crossing
per beat, torch/on-device**; route `r.lat()`, `r.cv()`, eikonal, and every LAT-based field through it. Compute the
CV-field ∇LAT with a **Bayly-style local quadratic fit** (or at least Gaussian-smoothed FD), not raw centred
differences; expose the fit residual as a confidence map. **Mask non-activating nodes (NaN)** out of ∇T. Expose
**max(dV/dt) LAT** (parabolic sub-frame peak) as a probe-point "physiological reference." For pacing/reentry use
**multi-crossing LAT + phase analysis** (§ 2), not first-crossing. Consider an OpenEP-style divergence gate to
blank CV at collisions/foci. Keep θ a config knob but ship −40 mV canonical.

---

## 2. Wavefront Curvature, Vorticity & Rotors (phase singularities)

### Standard math
Curvature = divergence of the normalized activation-time gradient (= curvature of the T level-set isochrone):
```
n̂ = ∇T/|∇T|
κ  = ∇·n̂ = (T_xx·T_y² − 2·T_x·T_y·T_xy + T_yy·T_x²) / (T_x² + T_y²)^(3/2)
```
Validated against the **eikonal curvature law** `CV_n = CV0 − D·κ` (Keener 1991; Colli-Franzone 1990; mechanism
Fast–Kléber 1997). Convex/expanding front κ>0 → slowed; concave/converging κ<0 → accelerated. **Critical curvature**
`κ_c = CV0/D` (radius `r_c ≈ D/CV0`) → source–sink block, the wavebreak/reentry seed (Cabo 1994). Pitfall:
`|∇T|→0` at foci, collisions, and the rotor core blows up κ and speed — clip below a `|∇T|` floor.

Rotor detection, phase route:
1. Build phase φ(x,t) ∈ (−π,π] from a single signal — Hilbert transform (`φ=atan2(H[f],f)`, Umapathy 2010),
   state-space/time-delay embedding (`φ=atan2(R−R*, V−V*)`, R=V(t−τ), Gray 1998), or V–dV/dt.
2. **Phase singularity = topological charge ±1**: `n_t = (1/2π)∮_C ∇φ·dl = ±1`. Discrete, on a CCW 2×2 plaquette:
   ```
   n_t = (1/2π) · Σ_{i=1..4} W(φ_{i+1} − φ_i),   φ₅≡φ₁,   W(x) = atan2(sin x, cos x)
   ```
   +1 = CCW rotor, −1 = CW, 0 = none. (Bray–Wikswo convolution-kernel form for whole-frame PS maps.) Complement
   with a **phase-defect-line** check for linear-core rotors (Arno 2021).

The **same loop-sum** is: circulation `Γ = ∮v·dl = ∬curl(v)dA` on the velocity field (vorticity
`ω = ∂v_y/∂x − ∂v_x/∂y`), and **Gauss–Bonnet** `∮κ ds = 2π·(enclosed rotor count)` on the isochrone tangent.

Rotor tip, isoline route (what production sims ship): `tip = {V=V_iso} ∩ {dV/dt=0}` (Fenton–Karma 1998).

### Prior implementations
- **openCARP `GlFilament`**: isoline intersection — activation isosurface (V=−20 mV) ∩ recovery (dV_m/dt=0) over
  an `iso_intv` window; outputs filament points.
- **Finitewave `spiral_wave_core_tracker.py`**: isoline crossing (threshold 0.5) in two successive frames +
  bilinear refinement. Isoline, not phase.
- **Phase toolchain**: `scipy.signal.hilbert` + topological-charge convolution (Umapathy 2010 / Bray–Wikswo);
  standardized params in Li 2020.
- **Isochrone extraction**: `skimage.measure.find_contours` (marching squares, sub-pixel, mask-aware).

### Verdict for our design
`curvature = ∇·n̂` and `vorticity = curl(v)` as designed. Compute κ two ways and cross-check: divergence of the
normalized LAT gradient vs curvature of the marching-squares isochrone — disagreement flags under-resolution.
Regress `CV_n = CV0 − D·κ` as the validation asset (`fit_eikonal` already does). **Implement one wrapped-loop-sum
primitive** and expose it as `circulation` (velocity), `winding_number`/phase-singularity (phase), and
`∮κ ds`/Gauss-Bonnet (isochrone). Ship **both** tip definitions: isoline-intersection for clean sim data (default),
phase-singularity for noisy/experimental, + a phase-defect-line guard. Guard `|∇T|→0` everywhere. Note: reentry is
where LAT-based fields break down → this is the "use phase not LAT" boundary the design doc already calls out.

---

## 3. Source–Sink, Safety Factor & Bidomain Current Fields

### Standard math
The monodomain diffusion term IS the local source–sink map:
```
voltage_flux = D∇Vm
source_sink  = ∇·(D∇Vm) = D∇²Vm     ( <0 local source / front crest;  >0 local sink / being charged )
```
Divergence theorem (feature + free self-check): `∬_Ω ∇·(D∇Vm) dA = ∮_∂Ω (D∇Vm)·n̂ dl`, i.e.
`sum(source_sink over Ω) == line-integral of voltage_flux·n over ∂Ω`.

**Safety factor** (SF>1 propagates, SF<1 blocks), competing definitions:
- **Shaw–Rudy 1997** (charge-based, 1-D): `SF_SR = ∫_A(I_c + I_out)dt / ∫_A I_in dt` over the activation interval
  A = [t at 1% of max dV/dt, t at V_max]; `I_c` capacitive, `I_out`/`I_in` axial to downstream/upstream. **Breaks
  in 2-D/at junctions** (I_in→0 → SF maximal exactly where block occurs).
- **Boyle–Vigmond 2010** (mesh-compatible, recommended): numerator is exactly our `source_sink` integrated:
  ```
  SF_VB = [ (1/β) ∫_A ∇·(σ̄_m∇Vm) dt ] / Q_thr(t_A)
        = [ C_m·ΔVm + Q_ion − Q_s ] / Q_thr(t_A),   A = [t_1%, t_Im0]
  ```
  `Q_thr(t_A)` = min charge to trigger an AP for a pulse of duration t_A — a **one-time single-cell calibration
  curve** per ionic model (≈linear in t_A). Works unchanged in 1-D/2-D/3-D + unstructured; drops below 1 exactly
  at conduction failure.

Bidomain current fields (standard, Henriquez / Plonsey-Barr): `electric_field = −∇φ_e`,
`current_flux = −σ_e∇φ_e`, current source density `∇·(σ_e∇φ_e) = β·I_m` (the bidomain analogue of source_sink;
collapses to ∇·(D∇Vm) in the monodomain limit).

### Prior implementations
- **openCARP** assembles the `div(σ∇φ)` stiffness operator and can output V_m/φ_e/φ_i/I_ion/I_c, but has **no native
  SF or source–sink field** — users derive it in post-processing. (Confirms the ingredients are exactly what we
  already have.)
- **Shaw–Rudy** on the LRd fiber; **Boyle–Vigmond** in the CARP ecosystem; **Romero** SFm2 (2-D generalization,
  calibrated critical profile ~6.6–7). Source–sink & curvature: Kléber–Rudy 2004 (canonical review).

### Verdict for our design
Build **`source_sink = ∇·(D∇Vm)`** first, with `voltage_flux = D∇Vm` (so `div(voltage_flux)==source_sink`) and the
**divergence-theorem boundary integral** as both feature and self-check. Sign: negative = local source. Then layer
**`safety_factor` = Boyle–Vigmond SF_VB** on top (needs only a one-time single-cell `Q_thr(t_A)` curve per model —
which `single_cell()` § 6 can produce). Keep **Shaw–Rudy as a 1-D-fiber-only validation** option. Add the bidomain
`electric_field`/`current_flux` (our `−σ∇φ_e`/`−∇φ_e` are the exact standard defs) reusing the same `div(σ∇·)`
operator.

---

## 4. Discrete Vector Calculus (grad/div/curl/laplacian + integral theorems)

### Standard math
Interior 2nd-order central differences; compact 5-point Laplacian
`∇²f = (f_{i+1,j}+f_{i−1,j}+f_{i,j+1}+f_{i,j−1}−4f_{i,j})/h²`. **The trap:** `div(grad)` of collocated central
operators = the *wide* 2h Laplacian (checkerboard null-space), NOT the compact 5-point. Fix = **staggered half-step
adjoint pair**: `grad` cell→face (forward diff), `div` face→cell (backward diff), so `div = −grad*` exactly. This
one property gives simultaneously (a) `div(grad) == laplacian`, (b) the **exact discrete divergence theorem**
(`∬div F dA = ∮F·n ds` telescopes to round-off — interior faces cancel pairwise), and (c) the free integral-tier
cross-check. This is the mimetic (Hyman–Shashkov) / SBP / MAC / finite-volume construction.

No-flux (Neumann) boundary = **ghost-node mirror**: `f_ghost = f_edge` ⇒ the outward face contributes zero flux ⇒
boundary node charged by fewer neighbours (the reduced electrotonic sink the solver saw). `face_mirror` ≙ scipy
`reflect` (half-sample symmetric). Internal scar/hole = **sever-the-connection** (openCARP: drop any stencil edge
crossing into a masked node; masked nodes → NaN); staircase outward normal = axis-aligned face normal, `ds` = face
length (O(h^1.5) in flux — documented; cut-cell/IIM only if higher accuracy ever needed). Isochrone line integrals:
marching squares → arc-length-weighted sum with bilinear field sampling; normal from `∇V/|∇V|` cross-checked
against the −90°-rotated polyline tangent.

### Prior implementations
- **scipy.ndimage** — boundary `mode` (`reflect`/`mirror`/`nearest`/`wrap`/`constant`) IS our `boundary_mode`
  abstraction; `reflect` = no-flux face mirror. Best conceptual match.
- **numpy.gradient** — de-facto collocated convention, but `div∘grad` = wide 2h stencil and one-sided edges are
  NOT no-flux → wrong for a boundary-matching `∇²V`. Use for interior only.
- **findiff** — closest Python vector-calc API (`Gradient/Divergence/Curl/Laplacian`), but collocated + Curl is
  3-D only + no torch.
- **OpenFOAM** `fvc::grad/div/curl/laplacian` — Gauss-theorem face sums, no-flux via `zeroGradient` patch (exact
  by construction). **FEniCS/scikit-fem** — Neumann is the "natural" BC (drops out of the weak form). **MOLE /
  PyDEC** — exact-theorem mimetic reference libraries. **skimage.find_contours** — the isochrone extractor.

### Verdict for our design
**Collocated public API backed by staggered internals.** Implement the operators as fixed conv/slice-subtract
stencils on device (torch, batches over leading `(T,…)`). Keep the exact staggered adjoint pair for the internal
Laplacian and the integral tiers (so `∬div == ∮F·n` is exact); average faces back to nodes only for user-facing
vector fields. Route **every** edge through `boundary_mode` as a ghost/mirror pad identical to the solver's
diffusion operator (default `face_mirror` = scipy `reflect`) — **never** `numpy.gradient` one-sided edges. Treat
`domain_mask` as internal no-flux by sever-the-connection; masked → NaN; reuse the *same* boundary-face set for the
region (`∬div`) and contour (`∮F·n`) tiers. This requires `SimulationResult` to carry `dx/dy`, `domain_mask`, and
`boundary_mode` (the addition both the differential and integral tiers need).

---

## 5. Scalar EP Metrics: Wavelength, APD, ERP, DI, Restitution

### Standard math
- **Wavelength** `λ = CV · ERP` [cm] (Wiener–Rosenblueth 1946; Allessie leading-circle 1977) — the reentry master
  variable = minimum sustainable circuit length. **ERP is the physiologically correct term; CV·APD90 is a proxy
  that UNDERESTIMATES λ whenever post-repolarization refractoriness (PRR = ERP−APD90 > 0) is present.** Unit trap:
  CV in cm/s, ERP in ms → `λ_cm = CV_cm_s · ERP_ms / 1000`.
- **APD at x%**: `V_repol(x) = V_peak − (x/100)(V_peak − V_rest)`, `APDx = t_repol(x) − t_act`. APD90 standard.
  Pitfalls: window V_peak/V_rest to *that beat*; spike-and-dome notch → use **last** above→below crossing
  (dome-aware).
- **Restitution**: `APD_{n+1} = f(DI_n)`, `DI_n = BCL − APD_n` (algebraic) or measured recovery interval. **Slope > 1
  → alternans** (Nolasco–Dahlen 1968; the DI where slope=1 is the alternans-onset boundary). Two protocols: **S1S2
  extrastimulus** (shallower) vs **dynamic/Koller** (steeper, alternans-relevant).
- **ERP**: an **active protocol** quantity — S1S2 extrastimulus, ERP = longest S1S2 coupling interval that fails to
  capture, found by **bisection**; defined at a stated stimulus amplitude (≈2× diastolic threshold). ERP ≈ APD90 in
  healthy tissue; ERP > APD90 = PRR.

### Prior implementations
- **openCARP**: `bench`/`limpet` APD restitution (S1S2 & dynamic protocols; APD90 from max-upstroke; output columns
  incl. triangulation APD90−APD30); ERP-restitution tutorial (S1S2 + bisection, ERP=longest non-capturing S1S2).
- **Myokit**: `run(apd_variable, apd_threshold)` (CVODE root-find, **fixed** threshold — differs from %-excursion);
  restitution scripted around `Simulation`.
- **Source papers**: ten Tusscher 2004/06 and O'Hara-Rudy 2011 report APD90 from max-upstroke with both S1S2 and
  dynamic restitution.

### Verdict for our design
Our existing `apd_at`/`restitution_slope` already match openCARP/Myokit conventions (APD90, dome-aware last-crossing,
slope-1 `DI_star`). Concrete work: **(a)** fix the multi-beat baseline (`V_rest = trace[0]` → per-beat pre-upstroke
diastolic V); **(b)** `wavelength(cv, refractory, kind='erp')` with the ÷1000 unit conversion, `kind='erp'` default,
`kind='apd'` a **warned** proxy, and `cv_scope='local'|'global'`; **(c)** `di(bcl, apd)` = both algebraic and measured;
**(d)** `erp(...)` as a **protocol-based** function (S1S2 + capture-detection bisection) — flag clearly that it
**runs simulations**, unlike the pure trace-analysis metrics (this is a real architectural fork), plus a cheap
`erp_proxy = apd90` and a `post_repol_refractoriness = erp − apd90` helper. Expose `activation='threshold'|'dvdt_max'`.

---

## 6. Single-Cell 0-D Mode (`single_cell()`)

### Standard math
`Cm·dV/dt = −(I_ion(V,s) + I_stim)` with the full gating + concentration ODE system and **no diffusion term** —
i.e. the monodomain reaction term with `∇·(D∇V)` set to zero. **Rush–Larsen** for gates (exponential integrator,
unconditionally stable, bounded):
```
g_{n+1} = g∞(V_n) + (g_n − g∞(V_n))·exp(−Δt/τ(V_n))
```
forward Euler for V and concentrations. Adaptive stiff solvers (CVODE/LSODA/`ode15s`, rtol=atol≈1e-6) are the
single-cell accuracy reference. **Steady-state pre-pacing is mandatory** (Na_i/Ca_i drift over many beats):
1000 beats for ORd-class (Dutta/CiPA, ToR-ORd), 100–1000 for TTP06, run-to-limit-cycle for spontaneous Paci hiPSC.

### Prior implementations & the consistency subtlety
- **Myokit**: CVODES + `s.pre(t)` (pace without logging/clock) — the idiom for steady state. **openCARP `bench`**:
  fixed-step FE+RL, `--imp/--dt/--bcl/--stim-*`, freeze/restore state. **Chaste**: shared `GetIIonic()` for both
  single-cell and tissue, FE+RL Δt=0.01 ms. **CellML/OpenCOR**: CVODE.
- **0-D↔tissue divergence points**: (1) operator split reorders reaction/diffusion → the reaction substep must call
  the *identical* ionic step as 0-D; (2) **post-Rush-Larsen gate ordering in concentration currents** — using new
  gates `g_{n+1}` vs old `g_n` in the concentration flux drifts Na_i/Ca_i at O(Δt) (**this is exactly our ORd
  "concentration path unwired in tissue splitting" bug**); (3) stimulus charge — inject I_stim into dV/dt only, not
  any concentration balance, or K_i drifts (Hund 2001 charge conservation).

### Verdict for our design
**Route `single_cell()` through the SAME per-node ionic step the monodomain reaction substep uses** (diffusion
omitted), not a parallel re-implementation — this closes the 0-D gap AND sidesteps the ORd bug because both modes
exercise one fixed code path. Pin the order: compute all currents/concentration-RHS from the start-of-step state →
Rush–Larsen the gates → forward-Euler V and concentrations from those start-of-step currents. **Default FE+RL**
(tissue-consistent, GPU-friendly); Δt = 0.02 ms TTP06 / 0.005–0.01 ms ORd / 0.01 ms cross-model default; optional
`solver='adaptive'` (SciPy LSODA/Radau, rtol=atol=1e-6) for validation. Stimulus = current pulse into dV/dt only
(−80 A/F×0.5 ms ORd, −52 A/F×1 ms TTP06; Paci spontaneous, no stim). `pre_pace(n_beats, bcl)` Myokit-style,
default 1000 beats ORd-class, save/reuse the paced state. Per-model defaults:

| Model | Stimulus | Δt (RL) | Pre-pace | Mode |
|---|---|---|---|---|
| TTP06 (18-state) | −52 A/F, 1 ms | 0.02 ms | 100–1000 beats | paced |
| ORd (40-state) | −80 A/F, 0.5 ms | 0.005–0.01 ms | 1000 beats @ 1000 ms | paced |
| Paci hiPSC (+variants) | none (spontaneous) | 0.01 ms | run to limit cycle | self-pacing |

---

## 7. DEEP DIVE — the CV velocity field (the payoff of LAT) & the LAT integrals

> Focused second pass (2026-07-21) on the two things § 1 skimmed: **how people actually compute the
> conduction-velocity vector field from LAT** (the reason LAT exists), and **the LAT line/contour integrals**
> and how they tie back to — and validate — that field.

### 7.0 Framing: velocity is the deliverable, LAT is the means
LAT `T(x,y)` is a bookkeeping scalar — its absolute value depends on where you started the clock. The
physiological payload is the CV vector field, reached via the **eikonal identity** `|∇T| = 1/c`; all three named
fields are algebraic functions of the *single* gradient `∇T`:
```
speed c = 1/|∇T|        direction n̂ = ∇T/|∇T|        velocity v = ∇T/|∇T|²
```
The load-bearing consequence: **the whole payload lives in a derivative of a noisy, discretely-sampled scalar.**
Coveney 2020 quantifies it — LAT interpolation is easy (nRMSE ≈ 0.5%), its *gradient* is ~14× harder
(nRMSE ≈ 7%), and `c = 1/|∇T|` is singular exactly where the front is fast or flat. So "compute LAT then
differentiate" is not a formality: *how* you represent T and differentiate it is the entire ballgame.

### 7.1 How the CV vector field is computed — method families
| # | Family | Core math | Uncertainty? | Anisotropy? | Robustness | Notes |
|---|--------|-----------|:---:|:---:|-----------|-------|
| 1 | Finite difference of T | central diff, `c=1/√(Gx²+Gy²)` | no | no | low | noise↑, staircase, flat-LAT singular |
| 1s | **Smoothed FD** | Gaussian/Savitzky-Golay + FD | no | no | med-high | Cantwell's "best simple"; smoothing biases CV flatter |
| 2 | **Polynomial fit (Bayly 1998)** | LS quadratic patch → analytic ∇T | via residual | per-vector | **high** | de-facto gold standard; window size is the knob; residual = free quality metric |
| 3 | RBF / thin-plate spline | global RBF interp of T → analytic ∇ | no (λ bias) | no | med-high | OpenEP route; smoothing λ flattens CV |
| 4 | **GP / GPMI (Coveney 2020)** | GP prior on T → gradient-GP | **yes, calibrated** | tensor ext | high | the modern frontier; error bars for free; MC through 1/\|∇T\| |
| 5 | Triangulation (3-pt) | local linear system for ∇T | no | via binning | low-med | for discrete electrodes; noisiest |
| 6 | **Eikonal-fit / PINN** | fit `√(∇T·M∇T)=1` | some | **yes (tensor)** | high | fits the field not pointwise ∇; yields fibers + CV_L/CV_T; PIEMAP, Sahli-Costabal |
| 7 | Omnipolar / EGF | E-field loop / optical flow | no | direction | med | **no LAT step** — but needs raw EGMs, not a T-map |
| 8 | **Velocity ellipse (Roney 2019)** | ellipse fit to CV vectors, ≥3 maps | via fit | **yes** | high | recovers CV_L, CV_T, fiber angle |

Detail on the two that matter most for us:
- **Polynomial (Bayly)** — fit `T = ax²+by²+cxy+dx+ey+f` over a k×k window (ElectroMap uses **5×5**), differentiate
  analytically at the center: `Tx=2ax+cy+d`, `Ty=2by+cx+e`, then `v=∇T/|∇T|²`. The least-squares over-determination
  *is* the noise suppression; the **residual is a built-in quality metric** (high = collision/bad data). Bigger
  window = more noise rejection but more curvature over-smoothing near foci/pivots. On a structured grid this is a
  **fixed precomputed pseudo-inverse applied as a depthwise conv** — fast, batched, GPU-native.
- **GP / GPMI (Coveney)** — a GP prior on T, conditioned on uncertain LATs; since a linear operator on a GP is a GP,
  the **gradient is a GP too** → `velocity_mean` + `velocity_std`. Use Matérn-ν=3/2 (so T is once-differentiable),
  ≥~250 effective samples, and Monte-Carlo (~2000 draws) through the nonlinear `1/|∇T|`. Its principled advantage:
  at `|∇T|→0` it **reports huge variance instead of a confident wrong number**. Reference impl: `quLATi`.

### 7.2 The `|∇T|→0` problem & divergence gating (mandatory)
`speed=1/|∇T|` and `velocity=∇T/|∇T|²` are singular exactly at the interesting sites — **collisions**,
**breakthrough/foci**, **rotor cores**. Every gradient method fails there (FD spikes; poly/RBF/GP flatten and
over-read CV; GPMI at least flags it as variance). Standard fix (OpenEP EP Workbench; Masè 2021) = **divergence of
the unit direction field**:
```
D = ∇·n̂ = ∇·(∇T/|∇T|)      D>0 → focal source (exclude/flag)   D<0 → collision (exclude/flag)   D≈0 → planar (trust)
```
Note `∇·n̂ = κ_isochrone`, so this is the isochrone-curvature map. Mask CV where `|D|>thresh`, `|∇T|<floor`, or the
fit residual is high — and expose `D` as a **feature** (focus/collision detector), not just a gate.

### 7.3 Anisotropy — the velocity ellipse (an identifiability fact)
CV is fast along fibers (CV_L) and slow across (CV_T) — the speed-vs-direction locus is an ellipse. **A single
activation map only gives the *apparent* CV in the direction the front happened to travel; you cannot separate
CV_L from CV_T without multiple wave directions or a fiber prior** (Roney 2019: 3 maps → ~70% of fibers within 20°,
median angle error ~11°). Encode this in the API — do NOT advertise CV_L/CV_T from one map. The tensor route
(eikonal-fit § 7.1-#6) is the single-map alternative, but only with a regularizing model.

### 7.4 Accuracy benchmarks — what's actually most accurate
- **Linnenbank 2014** (1.6M sims): at high anisotropy / coarse sampling, **model-guided (single-vector / ellipse)
  beats brute averaging**; averaging over-reads CV_L (near-empty fiber-direction bins keep accidentally-large vectors).
- **Vigmond, Roney, Bayer, Nanthakumar 2024/25** (the sobering modern benchmark): the two dominant error sources are
  **ignoring 3-D propagation** (surface-projected gradient mis-reads a transmural front) and **coarse sampling**. With
  a 10 cm/s tolerance, **CV is reliable only within ~2 cm of the pacing site and only at ≤~1 mm sampling** — much
  clinical CV mapping measures artifact.
- **Circle method (Siles-Paredes 2022)**: works in a wavefront-aligned local frame → dodges the coordinate-axis
  singularity, robust at low SNR (beats FD's ±20% and polynomial fitting).
- **Bottom line:** no universal winner; consistent messages — (1) local model-fitting (polynomial/circle/eikonal/
  ellipse) beats raw differencing; (2) smoothing is mandatory but biases CV flatter/higher; (3) fine sampling +
  proximity to one clean wavefront dominate all algorithmic differences; (4) **GP is the only family that tells you
  when to distrust the number.**

### 7.5 The LAT integrals — and the slowness-vs-velocity trap
**Path-independence identity.** `σ = ∇T` (slowness) is by construction a gradient → conservative → for ANY path C:
```
∫_C ∇T · dl = T(B) − T(A)      (exact, path-INDEPENDENT — the Fermat/eikonal travel-time integral)
```
So **conduction time between two sites is just `ΔT = T(B)−T(A)`** — return it directly; the line-integral form is
only needed for a *prescribed* anatomical path. Path CV = `L(C)/(T(B)−T(A))`.

⚠️ **CRITICAL CODING TRAP:** conduction time integrates the **slowness `σ=∇T`, NOT the velocity `v=∇T/|∇T|²`.** Only
`∇T` is curl-free; `∫v·dl ≠ ΔT`. Our design doc's `conduction time = ∫∇LAT·dl` is correct (that's the slowness), but
the code must not accidentally line-integrate the stored `velocity` field.

**Free consistency checks from the same identity:**
- `∮∇T·dl = 0` on any closed loop (curl-free) — a discrete ∇T from bad interpolation FAILS this → a direct detector
  of annotation noise / unresolved isochrones. (Fix: differentiate a smooth *scalar* interpolant → curl-free to
  machine precision by construction.)
- `∮∇T·dl = CL` (one cycle length) around a loop enclosing a **reentry** circuit — T is multivalued there; the
  *nonzero* circulation is the topological reentry signature (the LAT analogue of the phase winding number). Nonzero
  is the intended signal, not a bug.

**Isochrone-spacing CV = 1/|∇T| (the classical method, proven equivalent).** Draw isochrones at increment Δt, measure
normal spacing Δs; then `CV = Δs/Δt`. Proof: stepping Δs along n̂ changes T by `ΔT = |∇T|·Δs`; one isochrone increment
means `ΔT=Δt`, so `Δs = Δt/|∇T|` → `CV = Δs/Δt = 1/|∇T|`. "Isochrone crowding" ⇔ large `|∇T|` ⇔ slow CV (the clinical
ILAM/deceleration-zone method; Raiman-Tung: 3 isochrones within 1 cm ⇒ CV<0.6 m/s). **This equivalence is the single
best regression test** that the geometric and differential CV pipelines agree. Measure Δs along `n̂`, not an arbitrary
axis.

**Co-area identity (ties front length ↔ speed ↔ area):** with wavefront length `L(t)=∮_{T=t}ds` and activated area
`A(t)=∬𝟙[T≤t]dA`,
```
dA/dt = ∮_{T=t} (1/|∇T|) ds = ∮_{T=t} CV ds = L(t)·⟨CV⟩_front
```
Independently computing `dA/dt`, `∮c ds`, and `L·⟨CV⟩` and checking agreement is a second whole-field consistency
gate. (Front length `L` = the source-size for source–sink; `dA/dt` = recruitment rate, the quantity a defib shock
must drive ≤0.)

**Gauss–Bonnet / total turning:** on a simple closed isochrone `∮κ_g ds = 2π` (Hopf), `= 2π·m` for m enclosed
rotational cores — a topological rotor count that must be an integer (non-integer ⇒ contour noise / self-intersection)
and must agree with the `∮∇T·dl/CL` circulation count and the winding-number rotor map.

**Geodesic / fast-marching:** the forward map is the eikonal `√(∇T·M∇T)=1`; solving it = a weighted geodesic-distance
computation (Sethian FMM; Wallman 2012), and conduction time between sites = the weighted shortest path. Exposing an
eikonal solve gives a **forward/inverse round-trip test**: re-solve T from an estimated `c=1/|∇T|` field and check it
reproduces the measured LAT.

### 7.6 The unifying design principle
**Build every LAT field and integral on ONE smooth, single-valued interpolant of T (RBF or local polynomial), and
derive everything analytically from its gradient.** Then: the fields are mutually consistent *by construction*;
`curl(∇T)=0` to machine precision so **only the divergence carries information** (which is what makes the
collision/focus diagnostics trustworthy); and the path-independence, isochrone-spacing≡1/|∇T|, co-area, and
Gauss–Bonnet identities all become built-in regression tests rather than separate code.

### 7.7 Verdict for our design (velocity field + LAT integrals)
- **`velocity`/`direction`/`speed`**: default **local polynomial (Bayly) k=5 patch** (fixed conv on the grid) →
  analytic ∇T; ship the **per-window residual as a `quality` field**. Smoothed-FD as a cheap fallback/validator.
  Guard `|∇T|<floor`.
- **Divergence gating ON by default**: emit `divergence = ∇·n̂` + a boolean `mask` flagging foci (D>0), collisions
  (D<0), low-|∇T|, and high-residual nodes. Never silently return `1/|∇T|` at a collision.
- **Uncertainty option (the differentiator)**: a Matérn-3/2 **GP gradient head** → `velocity_std`/`speed_ci`; the one
  thing no lightweight CV tool offers, and it turns the `|∇T|→0` blow-up into reported variance.
- **`conduction_time`** = `T(B)−T(A)` directly (integrate **slowness ∇T**, never `velocity`); expose `∫_C∇T·dl` for
  user paths and use path-independence (`∮∇T·dl≈0`; `≈CL`⇒reentry) as a self-test.
- **`isochrone_spacing_CV`** as a regression test asserting `Δs/Δt == 1/|∇T|` (measure Δs along n̂).
- **Co-area, Gauss–Bonnet, circulation** as whole-field consistency gates (the "free validation asset" the design
  doc wanted).
- **Anisotropy tensor** (CV_L/CV_T/fiber angle) only as an opt-in that requires multiple maps (or an eikonal-fit);
  document the single-map identifiability limit.

## 8. STRUCTURED-GRID REGIME — the optical-mapping / simulation analog (concrete defaults)

> Focused pass (2026-07-21) on the literature for a **dense uniform Cartesian grid** — our actual input. The § 7
> velocity-field methods were drawn largely from the *clinical* world (sparse scattered electrodes on a curved
> manifold → RBF, GP-manifold, triangulation, dominated by *reconstruction* error). Our regime is the opposite:
> **optical mapping** (a CMOS/CCD camera → a regular pixel grid of a Vm-proxy signal over time) and **PDE-solver
> output on its solve grid**. Every node populated, neighbors on a known lattice, time a regular axis → the correct
> ops are **fixed convolution stencils, windowed polynomial fits, per-pixel Hilbert phase** — dense, vectorizable,
> GPU-native. The governing tradeoff is not sparsity but **frame-rate quantization vs SNR-vs-blur**.

### 8.1 `optimap` is the direct code-level analog — mirror its API
`optimap` (Cardiac Vision Lab, pure-Python `{t,x,y}` arrays; Lebert & Christoph) is literally our library's twin:
- **CV**: `compute_velocity_field(method="bayly"|"circle"|"gradient")` → per-pixel `(rows,cols,2)`. `bayly` fits exactly
  our `T=ax²+by²+cxy+dx+ey+f` (`scipy.lstsq`, odd window default ≈ 10% of grid) → `v=∇T/|∇T|²`; `gradient` =
  Gaussian σ2 then `np.gradient`; `circle` = Siles-Paredes (radius 5px).
- **LAT**: `compute_activation_map(method="maximum_derivative"|"threshold_crossing", interpolate=False)` — dV/dt peak
  with **3-point parabolic** sub-frame refine (`offset = 0.5*(y0−y2)/(y0−2y1+y2)`), or threshold with linear
  sub-frame. Note `interpolate` **defaults off** — same gap as our `activation_time`.
- **Phase / PS**: `compute_phase = angle(hilbert(video−0.5, axis=0))`; `detect_phase_singularities` = the **2×2-plaquette
  topological charge** (`Σ = Δx_top − Δy_right − Δx_bottom + Δy_left`; PS where `π<|Σ|<3π`, charge = sign) — the
  grid-concrete instance of § 2's winding-number primitive.
- **Filtering**: NaN/mask-aware separable Gaussian (`smooth_gaussian`, `smooth_spatiotemporal`) — smooth data + mask,
  divide, so tissue edges don't bleed (directly relevant to our boundary-CV work).

Other tools & their conventions: **ElectroMap** (4×4 Gaussian σ=1.5 spatial, 3rd-order Savitzky-Golay temporal, N×N
multi-vector CV — 5×5 typical, LAT = dF/dt_max); **RHYTHM/Laughner 2012** (Bayly polynomial CV, 3×3 box bin, 50th-order
zero-phase FIR 0–100 Hz); **COSMAS** (the "comb" algorithm for known-pacing-rate segmentation); **KairoSight** (radial
single-vector); **finitewave** (sim-native: −40 mV threshold LAT, Fenton-Karma isoline-intersection tip); **openCARP**
(`.igb` + `igbextract`).

### 8.2 The Bayly fit on a uniform grid IS a fixed Savitzky-Golay convolution (the implementation win)
On a *uniform* grid the window's local coordinates are identical at every node, so the design matrix `X` — and the
pseudo-inverse `C = (XᵀX)⁻¹Xᵀ` — are **data-independent**. Every polynomial coefficient is then a **fixed convolution
kernel** applied to T (this is exactly the 2-D Savitzky-Golay filter; Krumm 2001, Savitzky-Golay 1964). So:
`T̂ = C₀₀*T` (smoothed value / residual reference), `Tx = a₁₀ = K_x*T`, `Ty = a₀₁ = K_y*T`, and the curvature inputs
`Txx=2a₂₀`, `Tyy=2a₀₂`, `Txy=a₁₁` — **one window fit yields velocity AND curvature**. Precompute `C` once at import,
reshape the coefficient rows to `k×k` torch buffers, apply with `F.conv2d` (scale `K_x*=1/dx`). Fast separable order-2
path: derivative weights `g = [−2,−1,0,1,2]/(10·dx)` along the differentiated axis, SG-quadratic smoother
`s = [−3,12,17,12,−3]/35` across it, `K_x = s_y ⊗ g_x`. The least-squares **residual `‖d−Xa‖²` is a free per-node
confidence map** (mask block lines, collisions, border) — the grid-native version of clinical PSF weighting, cheap
because `C` is precomputed.

### 8.3 Match the solver's operators for source reconstruction (ties § 4 ↔ § 7)
Two distinct operator needs: **display CV** can use the generic SG/Sobel gradient, but **reconstructing the
electrotonic source `∇·(D∇V)`** must use the *same discrete operators the solver used*, or the reconstructed source
carries a stencil-mismatch error that looks like spurious source/sink. That means the mimetic staggered `div = −gradᵀ`
pair (§ 4) / SBP operators — expose a shared stencil module so the analysis `laplacian` equals the solver's. Concrete
5-pt / isotropic 9-pt Laplacians as in § 4. So: **SG kernels for the CV *field*; the solver's own staggered operators
for the `source_sink` reconstruction.**

### 8.4 Resolution requirements — what grid/frame-rate you actually need
- **The dx-vs-dt error law:** since `c = Δx/Δt_LAT`, propagating a timing uncertainty δt gives **`δc/c ≈ c·δt/dx`.**
  Error grows with speed and shrinks with the spatial baseline — so **you cannot fix a coarse `dt` by refining `dx`**
  (finer dx = smaller baseline = worse timing-limited error). Fast fronts need fine temporal sampling *or* a longer
  differencing baseline.
- **Frame rate (optical-mapping practice):** the AP upstroke (~1–2 ms) is often faster than the frame → dV/dt_max is
  quantization-limited and *underestimates* at low frame rate. Fix = **interpolate the upstroke** (1 kHz → ~16 kHz-
  equivalent captures nearly all the gain; parabolic peak-refine is the cheap version); **F50/midpoint LAT is more
  quantization-robust** than dV/dt_max. Keep the LAT step across one cell (`Δt_LAT = dx/c`) spanning ≫1 solver step.
- **Spatial filtering:** Gaussian σ≈1, kernel **3×3–5×5**; at ≥5×5 further smoothing gives NO SNR gain, only blur.
  **Avoid temporal filtering before dV/dt** (it depresses the upstroke, frame-size-dependently) — prefer spatial.
- **Grid convergence:** measured CV converges as `dx→0` only once the upstroke width (~`c·τ_upstroke`) is resolved by
  several nodes; **below that, under-resolution numerically DEPRESSES apparent CV** (a numerical block, not physics —
  the project's own known failure mode). **Ship a `dx`-ladder CV-convergence test as the acceptance gate.**
- **Linnenbank 2014 (the canonical regular-grid validation):** grids 8×8→32×32 at 0.5 mm; **30° angle bins** (15° bins
  leave empties and overestimate CV_L except on ~22×22 grids); **5×5 subgrid** lowers variance but needs a larger total
  grid; the average-vector method **systematically overestimates CV_L** on small grids — report it, don't hide it.

### 8.5 Grid-native isochrones, integrals, anisotropy
- **Isochrones**: marching squares (`skimage.measure.find_contours`) — sub-pixel, but handle the **saddle ambiguity**
  (can flip topology near collisions) and open boundary/mask contours before any `∮`. Contour CV = **normal** isochrone
  spacing `Δs/Δt_iso` (a gradient-free second CV estimator, robust where `∇T`-methods are ill-conditioned).
- **Grid integrals**: discrete divergence theorem via mimetic/SBP `div`; area by shoelace `A=½∮(x dy−y dx)`; line
  integrals **arc-length-weighted** (`Σ f_k·|Δr_k|`, unequal segments); SBP-norm quadrature keeps `∬` consistent with
  `div`.
- **Anisotropy**: fit the velocity ellipse `c(θ)² = CV_L²·cos²(θ−φ) + CV_T²·sin²(θ−φ)` (⇔ eikonal `∇Tᵀ M ∇T = 1`,
  eigenvalues `1/CV_L²`,`1/CV_T²`). **Identifiability: a single map is under-determined** near the source and on
  near-planar segments (oblique/curved front mimics anisotropy) — require an assumed fiber axis or multiple pacing
  directions.

### 8.6 Verdict — concrete structured-grid defaults for our library
- **CV**: default **local quadratic (p=2) Bayly fit on a 5×5 window as a precomputed SG `conv2d` kernel** (emit the
  residual/R² `quality` map); separable order-2 fast path; `gradient` (Gaussian σ≈2) as the cheap smooth-field option;
  `circle` (r≈5) and isochrone-spacing as robustness cross-checks. Expose `window=3|5|7`, `order=1|2|3` (order≥2 for
  curvature). Mirror `optimap`'s method names.
- **LAT**: dV/dt_max with **parabolic sub-frame refine** as primary; F50/threshold (linear sub-frame) as the
  quantization-robust alternative; make the definition an explicit named parameter (resolves the LAT triple-def issue).
- **Filtering**: NaN/mask-aware Gaussian σ≈1 (≤5×5); no temporal filter before dV/dt.
- **Phase/PS**: normalize→(−0.5)→Hilbert along t→2×2-plaquette topological charge; cross-check against a finitewave-
  style isoline-intersection tip on sim data.
- **Consistency mode**: SG kernels for display CV, the **solver's own staggered `div=−gradᵀ`/laplacian** for
  `source_sink` reconstruction.
- **Validation**: `dx`-ladder CV convergence as the acceptance gate; report CV_L overestimation on small grids; use
  30° bins for directional CV.

## Recommended build order (dependency-sorted)

1. **Canonical LAT** (§ 1) — everything LAT-based inherits it; closes the OPEN BIG ISSUE. Ship interp −40 mV +
   max-dV/dt reference; regate `r.lat()`/`r.cv()`.
2. **`fields.derivatives` staggered core** (§ 4) — grad/div/curl/laplacian with `boundary_mode` + mask; the
   `div(grad)==laplacian` + exact-divergence-theorem foundation everything else stands on. Requires
   `SimulationResult` to carry `dx/dy/domain_mask/boundary_mode`.
3. **Vm/φ_e named fields** (§ 3) — `voltage_gradient`, `voltage_flux`, `source_sink`, `electric_field`,
   `current_flux`; no LAT dependency, immediately useful, and `source_sink` unlocks the SF.
4. **LAT-based named fields** (§ 1–2, **§ 7.1–7.4**) — `velocity`/`direction`/`speed` via a Bayly k=5 patch
   (analytic ∇LAT) + `quality` residual, **divergence gating on by default** (foci/collision/low-|∇T| mask), optional
   GP `velocity_std`; `curvature`, `vorticity`; guard `|∇T|→0`; migrate `front_metrics` onto these. Anisotropy
   tensor (CV_L/CV_T) only with multiple maps.
5. **`fields.integrals`** (§ 2, 4, **§ 7.5**) — `conduction_time`=ΔLAT (integrate slowness ∇LAT, not `velocity`) +
   `isochrone_spacing_CV`≡1/|∇LAT| regression test; the one winding-number primitive (circulation/winding/Gauss-
   Bonnet, with ∮∇LAT·dl=0 curl-free / =CL reentry checks) + the divergence-theorem net-flux + co-area gate;
   isochrone extraction via marching squares. Build all of these on ONE smooth interpolant (§ 7.6).
6. **Scalar EP metrics** (§ 5) — `wavelength`, `apd` consolidation + baseline fix, `di`; then the protocol-based
   `erp` (separate track — runs sims).
7. **`single_cell()`** (§ 6) — shared ionic step; also produces the `Q_thr(t_A)` curve for the safety factor.
8. **`safety_factor`** (§ 3) — Boyle–Vigmond SF_VB on top of `source_sink` + the single-cell Q_thr curve.

---

## References (consolidated)

**LAT / CV**
- Cantwell CD, et al. 2015. Techniques for automated LAT annotation and CV estimation in cardiac mapping.
  *Comput Biol Med* 65:229–242. doi:10.1016/j.compbiomed.2015.04.027. PMC4593301.
- Bayly PV, et al. 1998. Estimation of conduction velocity vector fields from epicardial mapping data.
  *IEEE TBME* 45(5):553–562. doi:10.1109/10.668746.
- O'Shea C, et al. 2019. ElectroMap. *Sci Rep* 8:38263. doi:10.1038/s41598-018-38263-2.
- Williams SE, et al. 2021. OpenEP. *Front Physiol* 12:646023.
- openCARP LAT / tuneCV docs: opencarp.org/documentation/examples/02_ep_tissue/08_lats & …/03a_study_prep_tunecv.
- Finitewave: github.com/finitewave/Finitewave (`activation_time_tracker.py`, `velocity_2d_calculation.py`).

**Velocity field & LAT integrals — deep dive (§ 7)**
- Coveney S, Corrado C, Roney CH, et al. 2020. Gaussian process manifold interpolation for probabilistic atrial
  activation maps and uncertain conduction velocity. *Phil Trans R Soc A* 378:20190345. doi:10.1098/rsta.2019.0345.
  (GPMI; gradient-GP; CV uncertainty; ≥250-obs rule. Code: quLATi, Zenodo 3758043.) + IEEE TBME 67(1):99–109, 2020.
- Roney CH, Whitaker J, Sim I, et al. 2019. A technique for measuring anisotropy in atrial conduction… *Comput Biol
  Med* 104:278–290. doi:10.1016/j.compbiomed.2018.10.019. (velocity ellipse; CV_L/CV_T/fiber angle; 3-map identifiability.)
- Linnenbank AC, de Bakker JMT, Coronel R. 2014. How to measure propagation velocity in cardiac tissue: a simulation
  study. *Front Physiol* 5:267. doi:10.3389/fphys.2014.00267.
- Vigmond EJ, Roney CH, Bayer J, Nanthakumar K. 2024/25. The accuracy of cardiac surface conduction velocity
  measurements. *JACC Clin Electrophysiol*. doi:10.1016/j.jacep.2024.11.004; medRxiv 2024.01.26.24301849.
- Masè M, Cristoforetti A, Del Greco M, Ravelli F. 2021. A divergence-based approach for identification of AF focal
  drivers. *Front Physiol* 12:749430. (D=∇·n̂; D>0 focal, D<0 collision.) + Masè & Ravelli, IEEE EMBC 2010 (RBF CV).
- Vigneswaran V, et al. 2024. Enhancing OpenEP: atrial CV & CV heterogeneity via EP Workbench. *Europace*
  26(Suppl 1):euae102.626. (triangulation + polynomial + RBF; divergence gating; slow-CV <0.3 m/s.)
- Raiman M, Tung R. 2018. Automated isochronal late activation mapping (ILAM) to identify deceleration zones.
  *Comput Biol Med* 102:336–340. (isochrone crowding = slow CV; 3 isochrones/1 cm ⇒ CV<0.6 m/s.)
- Deno DC, Massé S, Nanthakumar K, et al. 2017. Novel omnipolar electrograms / resolving myocardial activation.
  *Circ Arrhythm Electrophysiol* 10(6):e004107. (E-field-loop velocity, no LAT step.)
- Bhatt N, Narayan SM, et al. 2022. Electrographic flow mapping for AF: theoretical basis. *J Interv Card
  Electrophysiol*. doi:10.1007/s10840-022-01308-8. (Horn–Schunck optical flow; sources = divergence poles.)
- Grandits T, Pezzuto S, et al. 2021. PIEMAP: personalized inverse-eikonal model from cardiac EAMs. *FIMH/STACOM*.
  arXiv:2008.10724. + Sahli Costabal F, et al. 2020. PINNs for cardiac activation mapping. *Front Phys* 8:42.
- Siles-Paredes JG, et al. 2022. Circle method for robust local CV from optical mapping. *Front Physiol* 13:794761.
- Sethian JA. 1996. Fast marching level-set method. *PNAS* 93(4):1591–1595. + Kimmel R, Sethian JA. 1998. Geodesic
  paths on manifolds. *PNAS* 95(15):8431–8435. + Wallman M, Smith NP, Rodriguez B. 2012. Graph/eikonal/monodomain
  activation-time estimation. *IEEE TBME* 59(6):1739–1748.
- O'Shea C, et al. 2019. ElectroMap (multi-vector 5×5 polynomial CV, single-vector, activation constant). *Sci Rep*
  9:1389. · Bayly 3-D extension: Barnette AR, et al. 2000. *IEEE TBME* 47(8):1027–1035.

**Structured-grid regime — optical mapping / simulation (§ 8)**
- Lebert J, Christoph J, et al. **optimap** — open-source Python library for fluorescence video (LAT/CV/phase/PS on
  `{t,x,y}` grids). cardiacvision.github.io/optimap · github.com/cardiacvision/optimap (`activation/_cv.py`,
  `phase/_singularities.py`). The direct code-level analog.
- Laughner JI, Ng FS, Sulkin MS, Arthur RM, Efimov IR. 2012. Processing and analysis of cardiac optical mapping data
  obtained with potentiometric dyes. *Am J Physiol Heart Circ Physiol* 303(7):H753–H765. doi:10.1152/ajpheart.00404.2012.
  (RHYTHM pipeline conventions: Bayly CV, 3×3 box bin, zero-phase FIR.)
- Gloschat C, et al. 2018. RHYTHM: open-source panoramic optical-mapping toolkit. *Sci Rep* 8:2921.
- Tomek J, Wang ZJ, Burton RB, Herring N, Bub G. 2021. COSMAS: lightweight cardiac optical-mapping analysis
  ("comb" algorithm). *Sci Rep* 11:9147.
- Haq KT, et al. 2023. KairoSight-3.0. *J Mol Cell Cardiol Plus* 5:100043.
- Nezlobinsky T, et al. 2026. Finitewave: lightweight cardiac EP simulation framework. *JOSS* 11(122):9310.
  github.com/finitewave/Finitewave (grid-native trackers: −40 mV threshold LAT, Fenton-Karma tip).
- Savitzky A, Golay MJE. 1964. Smoothing and differentiation of data by simplified least-squares procedures.
  *Anal Chem* 36(8):1627–1639. + Krumm J. 2001. Savitzky-Golay filters for 2-D images (MSR note; `C=(XᵀX)⁻¹Xᵀ`,
  explicit 5×5 derivative kernels).
- Kay MW, Gray RA. 2005. Measuring curvature and velocity vector fields for waves of cardiac excitation in 2-D media.
  *IEEE TBME* 52(1):50–63. doi:10.1109/TBME.2004.839798.
- Osher S, Sethian JA. 1988. Fronts propagating with curvature-dependent speed. *J Comput Phys* 79(1):12–49. +
  Sethian JA. 1999. *Level Set Methods and Fast Marching Methods*, CUP. (κ = ∇·(∇T/|∇T|) stencil.)
- (§ 4 mimetic/SBP refs — Hyman-Shashkov, Mattsson-Nordström, Svärd-Nordström — also ground § 8.3's match-the-solver
  operator requirement.) van der Walt S, et al. 2014. scikit-image. *PeerJ* 2:e453 (`measure.find_contours`).

**Curvature / rotors / phase**
- Fast VG, Kléber AG. 1997. Role of wavefront curvature in propagation. *Cardiovasc Res* 33(2):258–271.
- Cabo C, et al. 1994. Wave-front curvature as a cause of slow conduction and block. *Circ Res* 75(6):1014–1028.
- Keener JP. 1991. An eikonal-curvature equation for AP propagation in myocardium. *J Math Biol* 29:629–651.
- Colli Franzone P, Guerri L, Rovida S. 1990. Wavefront propagation in anisotropic cardiac tissue. *J Math Biol*
  28:121–176.
- Gray RA, Pertsov AM, Jalife J. 1998. Spatial and temporal organization during cardiac fibrillation. *Nature*
  392:75–78. doi:10.1038/32164.
- Iyer AN, Gray RA. 2001. Accurate localization of phase singularities during re-entry. *Ann Biomed Eng* 29(1):47–59.
- Bray M-A, Wikswo JP. 2002. Phase plane analysis for nonstationary reentrant cardiac behavior. *Phys Rev E*
  65:051902.
- Umapathy K, et al. 2010. Phase mapping of cardiac fibrillation. *Circ Arrhythm Electrophysiol* 3:105–114.
- Arno L, et al. 2021. A phase-defect framework for cardiac arrhythmia patterns. *Front Physiol* 12:690453.
- Li X, et al. 2020. Standardizing single-frame PS identification. *Front Physiol* 11:869.
- Fenton F, Karma A. 1998. Vortex dynamics in 3-D continuous myocardium. *Chaos* 8(1):20–47.
- Zhuchkova E, Clayton RH. 2005. Identifying & tracking phase singularities. *FIMH* LNCS 3504:246–255.
- openCARP GlFilament; skimage.measure.find_contours.

**Source–sink / safety factor / bidomain**
- Shaw RM, Rudy Y. 1997. Ionic mechanisms of propagation… *Circ Res* 81(5):727–741.
- Boyle PM, Vigmond EJ. 2010. An intuitive safety factor for cardiac propagation. *Biophys J* 98(12):L57–L59.
  doi:10.1016/j.bpj.2010.03.018.
- Kléber AG, Rudy Y. 2004. Basic mechanisms of cardiac impulse propagation. *Physiol Rev* 84(2):431–488.
- Romero L, et al. Safety factor in simulated 2-D cardiac tissue (SFm2); PLoS One PMC3817246 (curvature/source-sink).
- Henriquez CS. 1993. Simulating cardiac tissue with the bidomain model. *Crit Rev Biomed Eng* 21(1):1–77.

**Discrete vector calculus**
- Hyman JM, Shashkov M. 1997. Natural discretizations for div/grad/curl. *Comput Math Appl* 33(4):81–104; and
  Adjoint operators. *Appl Numer Math* 25(4):413–442.
- Ranocha H, et al. 2020. Discrete vector calculus & Helmholtz–Hodge for SBP operators. arXiv:1908.08732.
- Mattsson K, Nordström J. 2004. SBP for second derivatives. *J Comput Phys* 199:503–540.
- Perot B. 2000. Conservation properties of unstructured staggered mesh schemes. *J Comput Phys* 159:58–89.
- Harlow FH, Welch JE. 1965. MAC method. *Phys Fluids* 8:2182.
- Johansen H, Colella P. 1998. Cartesian embedded-boundary Poisson on irregular domains. *J Comput Phys* 147:60–85.
- LeVeque RJ. 2007. *Finite Difference Methods for ODEs and PDEs*, SIAM; 2002. *Finite Volume Methods*, CUP.
- Bell N, Hirani AN. 2012. PyDEC. *ACM TOMS* 39(1):3. · MOLE: github.com/csrc-sdsu/mole.
- Library docs: numpy.gradient, scipy.ndimage.correlate1d (boundary modes), findiff, OpenFOAM fvSchemes,
  FEniCS/scikit-fem (natural Neumann), skimage.measure.

**Scalar EP metrics**
- Wiener N, Rosenblueth A. 1946. Mathematical formulation of impulse conduction. *Arch Inst Cardiol Mex* 16:205–265.
- Allessie MA, Bonke FIM, Schopman FJG. 1977. Leading-circle reentry. *Circ Res* 41(1):9–18.
- Nolasco JB, Dahlen RW. 1968. Graphic method for alternation in cardiac APs. *J Appl Physiol* 25(2):191–196.
- Koller ML, Riccio ML, Gilmour RF. 1998. Dynamic restitution. *Am J Physiol* 275(5):H1635–H1642.
- Qu Z, Weiss JN, Garfinkel A. 1999. Restitution & spiral-wave stability. *Am J Physiol* 276(1):H269–H283.
- Weiss JN, et al. 2006. From pulsus to pulseless: cardiac alternans. *Circ Res* 98(10):1244–1253.
- Rensma PL, et al. 1988. Wavelength & reentrant atrial arrhythmias. *Circ Res* 62(2):395–410.
- Franz MR, Ravens U. 2014. Drug-induced post-repolarization refractoriness. *Europace* 16(suppl4):iv39–iv45.
- Comtois P, Kneller J, Nattel S. 2005. Leading circle vs spiral wave. *Europace* 7(s2):S10–S20.
- openCARP APD/ERP restitution tutorials; Myokit `Simulation` docs.

**Single-cell 0-D**
- Rush S, Larsen H. 1978. A practical algorithm for solving dynamic membrane equations. *IEEE TBME* 25(4):389–392.
- Marsh ME, Ziaratgahi ST, Spiteri RJ. 2012. Secrets to the success of Rush–Larsen. *IEEE TBME* 59(9):2506–2515.
- ten Tusscher KHWJ, Panfilov AV. 2006. Alternans and spiral breakup (TP06). *Am J Physiol* 291(3):H1088–H1100;
  and ten Tusscher et al. 2004, *Am J Physiol* 286(4):H1573–H1589.
- O'Hara T, Virág L, Varró A, Rudy Y. 2011. Undiseased human ventricular AP (ORd). *PLoS Comput Biol* 7(5):e1002061.
- Dutta S, et al. 2017. CiPA-ORd optimization. *Front Physiol* 8:616. · Tomek J, et al. 2019. ToR-ORd. *eLife*
  8:e48890.
- Paci M, et al. 2013. Ventricular/atrial hiPSC-CM models. *Ann Biomed Eng* 41(11):2334–2348 (+ 2018/2020).
- Hund TJ, Kucera JP, Otani NF, Rudy Y. 2001. Ionic charge conservation & steady state. *Biophys J* 81(6):3324–3331.
- Clerx M, et al. 2016. Myokit. *Prog Biophys Mol Biol* 120:100–114. · Plank G, et al. 2021. openCARP. *CMPB*
  208:106223. · Mirams GR, et al. 2013. Chaste. *PLoS Comput Biol* 9:e1002970.
