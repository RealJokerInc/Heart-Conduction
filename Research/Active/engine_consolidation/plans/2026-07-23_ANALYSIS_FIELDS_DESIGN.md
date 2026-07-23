# Design: `cardiac_core.analysis.fields` — named fields (+ derivatives/integrals toolkits)

> Status: DESIGNED (2026-07-21, design conversation). NOT yet implemented. Separate from the
> probe feature (deferred) and from the `r.grid()`/`r.coord()` ergonomics (small, separate).
>
> **Data model / object + shapes reference → [ANALYSIS_FIELDS_DATA_MODEL.md](./ANALYSIS_FIELDS_DATA_MODEL.md)**
> (2026-07-22): the object hierarchy (`r` → data / methods / `r.fields` → named fields + `.derivatives`/`.integrals`),
> the terminology (property vs method vs operator; `Fields` vs `VectorField`), and **every torch object's shape with
> each dimension defined** (the `T`/`Nx`/`Ny`/`2`/`n_states` axes; the "time axis present for Vm/φ_e-based, absent for
> LAT-based" rule). Read it when you lose track of what a call returns.
>
> **Prior-art / standard-math survey → [ANALYSIS_METHODS_PRIOR_ART.md](./ANALYSIS_METHODS_PRIOR_ART.md)**
> (2026-07-21): for every feature here + the scalar EP metrics + `single_cell()` + the canonical LAT, the
> standard math, how others implement it (openCARP / Myokit / Finitewave / OpenEP / scipy / skimage / primary
> literature), and a per-feature verdict. Most of our definitions match the field standard; five things changed
> — canonical LAT (interp −40 mV + max-dV/dt reference), a **staggered adjoint** numerical core (`div=−grad*` →
> `div(grad)==laplacian` + exact divergence theorem), `boundary_mode` as a ghost/mirror pad (not numpy one-sided),
> ONE winding-number primitive (circulation/vorticity/Gauss-Bonnet/phase-singularity), and `source_sink` IS the
> Boyle–Vigmond safety-factor numerator. See that doc's § 0 + the recommended build order.
>
> **MATH DERIVATIONS are now documented inline** below in two sibling sections: **(i) the operator toolkit**
> (`grad`/`div`/`curl`/`laplacian` + line/region integrals) — the machinery, tied together by `div=−grad*`, the
> `curl∘grad≡0`/`div∘curl≡0` identities, and the gradient/Stokes/divergence theorems (each a free discrete
> consistency check); and **(ii) per field** — for every field + every existing `analysis.py` op, the fundamental
> definition, the *equivalent* computational routes + why they're equal (velocity = LAT-gradient ≡ optical-flow;
> `source_sink` 3 ways; curvature 4 ways; scalar CV = the secant of the field), the traps, and the canonical choice.
> All descend from one master equation `Cm ∂V/∂t = ∇·(D∇V) − I_ion`; each equivalence is also a free consistency
> check. This is the blueprint's math layer — ready to `/blueprint` into a PLAN.md.

## Purpose & structure
The whole branch **operates on fields**. `fields` is the PARENT namespace, and the user-facing thing
it does is **hold the named, pre-saved physical fields you reach for** — one dot, plain-language,
explicit about the quantity:

    r.fields.voltage_flux      r.fields.velocity      r.fields.source_sink
    r.fields.electric_field    r.fields.curvature     r.fields.vorticity        (full catalog below)

Under that convenience surface sit two toolkits (one gradient implementation + one set of boundary
rules, shared):

- **`fields.derivatives`** — LOCAL operators, field → field: `grad`/`div`/`curl`/`laplacian`, the
  machinery the named fields are built from. Tucked away; most users never call it directly.
- **`fields.integrals`** — GLOBAL reductions, field → number over a region/contour: line and region
  integrals, each the Stokes/divergence-theorem PARTNER of a `derivatives` operator (built-in
  consistency check). Consume the named fields.

Both toolkits are torch-native and on-device (fixes the GPU gap: `front_metrics` is numpy/CPU and
crashes on a cuda tensor). **Naming rule: names say WHAT they act on** — `voltage_flux`/`voltage_
gradient` (not bare `flux`/`grad`), `current_flux` for the bidomain current, `electric_field` (not
`efield`). The common case is `r.fields.<name>` (one dot, cached); raw operators are under
`.derivatives` for power users.

## The precision principle (load-bearing)
Operators are **typed by the field they consume**, and the branch is explicit about *which* field
each named quantity differentiates — because the math is NOT interchangeable:

| call | on field | result | physical meaning |
|------|----------|--------|------------------|
| `div(grad(Vm))` = `laplacian(Vm)` | Vm | ∇²Vm | **electrotonic source–sink** (the diffusion term the solver used) |
| `curl(grad(Vm))` | Vm | **≡ 0** | vector-calculus identity — a NULL, guarded/not exposed as a metric |
| `curl(velocity_field)` | v (from LAT) | vorticity | **rotation → rotor cores** |
| `div(velocity_field)` / `div(n_hat)` | v / n̂ | κ | **wavefront curvature** (what `front_metrics` computes) |
| `grad(phi_e)` (× −1) | φ_e | E-field | current flow (bidomain) |

**Do not** offer a single ambiguous `curl(field)` — `∇×(∇V) ≡ 0` for any scalar V, so curl-of-a-
gradient must be a distinct, guarded call (or refused), while curl-of-velocity is the meaningful one.
"curl of the velocity field" and "curl of ∇V" are DIFFERENT calls with different return semantics.

## Mathematical foundations — the operator toolkit (grad / div / curl / laplacian + integrals)

> The **machinery** every named field is built from (`fields.derivatives` + `fields.integrals`). Same method as
> the per-field section. **The load-bearing structural fact:** the four differential operators and the integral
> operators are NOT independent — they are tied by (a) the adjoint relation `div = −grad*`, (b) the identities
> `curl∘grad ≡ 0` and `div∘curl ≡ 0`, and (c) the three integral theorems (gradient theorem, Stokes, divergence
> theorem). Build them so these hold **discretely** (mimetic / staggered) and each identity becomes a **free
> consistency test**. Grid spacing `dx, dy`; scalars on `(...,Nx,Ny)`, vectors stored `(...,2)`. Full numerical
> prior art: [ANALYSIS_METHODS_PRIOR_ART.md](./ANALYSIS_METHODS_PRIOR_ART.md) § 4.

#### grad — `grad(scalar) → vector`
**Definition.** `∇f = (∂f/∂x, ∂f/∂y)`; central interior stencil `∂f/∂x ≈ (f[i+1,j]−f[i−1,j])/(2dx)` (y analogous).
**Meaning / key facts.** Points along steepest increase and is **⟂ to the level sets of f** — so `grad(T)`'s
direction is the wavefront normal (basis of § 2), and `grad(V)` at activation is likewise normal to the front.
`grad` of any scalar is **curl-free** (`curl(grad f) ≡ 0`), which is what makes line integrals of gradients
path-independent (§ 7).
**Trap.** Loses a border row/col; the boundary stencil MUST match the solver's `boundary_mode` (ghost/mirror), not
`numpy.gradient` one-sided edges, or a post-hoc ∇ implies a spurious edge flux exactly at the reduced-sink edge.
**Canonical.** Central-difference interior; `face_mirror` ghost pad at edges; forward-difference to faces in the
staggered core (see laplacian).

#### div — `div(vector) → scalar`
**Definition.** `∇·F = ∂Fx/∂x + ∂Fy/∂y`.
**Meaning.** Net outward flux **per unit area** — a **source** (`div>0`) or **sink** (`div<0`) of F. This one
operator is behind `source_sink` (`div(D∇V)`), `curvature` (`div(n̂)`), and the collision/focus detector (`div(v)`).
**Key identity — divergence theorem.** `∬_Ω ∇·F dA = ∮_∂Ω F·n̂ dl` (region source = boundary flux) → the
`net_flux` integral and its self-check (§ integrals).
**Canonical.** Backward-difference from faces (staggered), so `div = −grad*` (see laplacian).

#### curl — `curl(vector) → scalar` (2-D z-component)
**Definition.** `∇×F = ∂Fy/∂x − ∂Fx/∂y` (the out-of-plane scalar).
**Meaning.** Local rotation rate (vorticity) of F — behind `vorticity = curl(velocity)` (§ 5).
**Key identities.** `curl(grad f) ≡ 0` (a gradient has no rotation → why `curl(∇V)` is a guarded NULL, the precision
principle); **Stokes** `∮_C F·dl = ∬_A (∇×F) dA` (circulation = enclosed vorticity).
**Trap.** NEVER expose `curl(grad(scalar))` as a metric — it is identically zero; the meaningful curl is of a genuine
vector field (velocity), not a gradient. (`curl(velocity)` and `curl(∇V)` are different calls — precision principle.)
**Canonical.** Central-difference of the two components; reuse the § 5 winding-number loop-sum for the integral form.

#### laplacian — `laplacian(scalar) → scalar` (= `div(grad)`)
**Definition.** `∇²f = ∂²f/∂x² + ∂²f/∂y²`; compact 5-point
`∇²f ≈ (f[i+1,j]+f[i−1,j]+f[i,j+1]+f[i,j−1]−4f[i,j])/h²`.
**The trap (load-bearing).** `div(grad f)` from **collocated** central differences is NOT the compact 5-point
Laplacian — it is the **wide 2h stencil** `(f[i+2]−2f[i]+f[i−2])/(2h)²` with a checkerboard null-space. To get
`div(grad) == laplacian` bitwise, use a **staggered half-step adjoint pair**: `grad` cell→face (forward diff),
`div` face→cell (backward diff), giving `div = −grad*` exactly.
**Why it matters (three payoffs from one property).** `div = −grad*` gives simultaneously: (a) `div(grad)==laplacian`;
(b) the discrete divergence theorem **exact** (interior faces telescope) → `∬div = ∮F·n` is a *free* check, not just
convergent; (c) `laplacian(V) = source_sink` equals the diffusion term the solver actually stepped.
**Canonical.** Compact 5-point (isotropic 9-point optional); staggered `div=−grad*` core for exactness.

#### Line integral — `∮ F·dl` (circulation) and `∮ F·n ds` (flux through a curve)
**Definition.** Tangential `∮_C F·dl = Σ_k F(x̄_k)·Δl_k`; normal (flux) `∮_C F·n̂ ds = Σ_k (F·n̂)_k |Δl_k|` — both
**arc-length-weighted** (segments are unequal).
**The three theorems that make them cross-checks.**
- `∮ F·dl = ∬ curl(F) dA` (Stokes) → `circulation`, `winding number`, `vorticity` (§ 5).
- `∮ F·n̂ ds = ∬ div(F) dA` (divergence thm) → `net source–sink`, flux balance (§ 7).
- `∫ ∇T·dl = T(B) − T(A)` (gradient theorem) → conduction time, path-independent; **integrate the SLOWNESS ∇T, not
  the velocity** (§ 7).
**Ergonomics.** The contour is a mesh/mask boundary or a marching-squares isochrone; normals from `∇/|∇|`
cross-checked against the −90°-rotated tangent; **CCW positive**, outward normal = efflux positive.
**Trap.** Must arc-length-weight; the mask-edge normal is axis-aligned (staircase, O(h^1.5) flux accuracy); sample F
by bilinear interpolation onto sub-pixel contour points.

#### Region / volumetric integral — `∬ f dA` (→ `∭ f dV` in 3-D)
**Definition.** Quadrature over a region: `∬_Ω f dA ≈ Σ_{cells∈Ω} f·dA`, `dA = dx·dy` (use the SBP norm as the
weight so `∬div` matches the boundary `∮F·n` exactly).
**Uses.** Activated area `∬𝟙[V≥θ]`, integrated load `∬∇²V`, net flux `∬div F` (= `∮F·n`), state fractions.
**Region = a mask.** The domain is the SAME `(Nx,Ny)` bool mask used for scars/stimuli (`over=mask`); the branch
derives `∂(mask)`, outward normals, and `ds` from it. **Do NOT hardcode 2-D** — the API generalizes verbatim to
`∭ dV` on a 3-D grid (today's numbers are per-area).
**Trap.** The interior-face telescoping that makes `∬div = ∮F·n` **exact** needs the SAME boundary-face set for both
tiers AND the staggered operators; a generic quadrature + generic gradient agree only to truncation error.

#### The unifying structure (why these are ONE system)
```
   grad                 div                         curl
 f ----> ∇f (vector) ----> ∇·F (scalar)     F ----> ∇×F (scalar, 2-D)
 identities:   curl∘grad ≡ 0        div∘curl ≡ 0
 theorems:     ∮∇f·dl = f(B)−f(A)   ∮F·dl = ∬curl F     ∮F·n ds = ∬div F
                 (gradient)           (Stokes)            (divergence)
 adjoint:      div = −grad*  ⟹  div(grad)=laplacian  AND  all three theorems hold DISCRETELY (exact self-checks)
```
**One gradient implementation + one boundary rule.** Every named field and every integral below is a *composition*
of these operators, so their mutual consistency (the theorems) is guaranteed **by construction**, not re-derived
per field. That is the entire reason to build the operator toolkit before the named fields.

## Calculations on a uniform grid — standard discrete stencils (cited)

> The concrete numerical realization of the operator toolkit. Every stencil / quadrature is the **standard textbook
> form** with its truncation order and canonical citation (verified 2026-07-22). Grid spacing `h` (or `dx, dy`);
> scalar `f[i,j]`, vector `F=(Fx,Fy)`; interior stencils, boundary variants noted. This is the **implementation
> contract**: pick one order, one boundary rule, one quadrature, and hold the operators to `div=−grad*` so the
> integral theorems (B5) are exact self-checks.

### A. Differential operators
**A1 — First derivative (central).**
```
2nd order:  Dx f = (f[i+1] − f[i−1]) / (2h)                              O(h²)
4th order:  Dx f = (−f[i+2] + 8f[i+1] − 8f[i−1] + f[i−2]) / (12h)        O(h⁴)
```
Arbitrary order/spacing: the **Fornberg (1988)** weight recursion (reproduces every central/one-sided coefficient).
*Fornberg 1988 Math.Comp. 51(184):699–706; LeVeque 2007 §1; NIST DLMF §3.4.*

**A2 — grad / div / curl** (compose A1 per axis, all `O(h²)` with the central stencil):
```
grad f = (Dx f, Dy f)      div F = Dx Fx + Dy Fy      curl F = Dx Fy − Dy Fx   (2-D z-comp)
```
*LeVeque 2007 §1–2.*

**A3 — Second & cross derivatives.**
```
Dxx f = (f[i+1] − 2f[i] + f[i−1]) / h²                                   O(h²)
   4th: (−f[i+2] + 16f[i+1] − 30f[i] + 16f[i−1] − f[i−2]) / (12h²)       O(h⁴)
∂²f/∂x∂y = (f[i+1,j+1] − f[i+1,j−1] − f[i−1,j+1] + f[i−1,j−1]) / (4·dx·dy)   O(h²)
```
*LeVeque 2007 §1; DLMF §3.4.*

**A4 — Laplacian.**
```
5-point (compact):   (f[i+1,j]+f[i−1,j]+f[i,j+1]+f[i,j−1] − 4f[i,j]) / h²      O(h²)  [1;1 −4 1;1]/h²
   leading error (h²/12)(f_xxxx + f_yyyy)  → ANISOTROPIC
9-point isotropic (Mehrstellen/Collatz):  (1/6h²)·[1 4 1; 4 −20 4; 1 4 1]      O(h²) bare
   leading error ∝ ∇⁴ (rotationally invariant = "isotropic"); O(h⁴) Poisson solve ONLY with
   modified RHS:  Lap9 u = f + (h²/12)·Lap5 f
```
⚠️ **Collocated `div(grad)` = the WIDE stencil** `(f[i+2]−2f[i]+f[i−2])/(4h²)` → decouples even/odd nodes
(checkerboard), NOT the compact 5-point. Fix = the staggered A8. *LeVeque 2007 §3.5, eqs 3.17–3.19; Collatz 1960;
Patra–Karttunen 2006; Harlow–Welch 1965 (checkerboard).*

**A5 — Level-set curvature (Osher–Sethian).**
```
κ = (φxx·φy² − 2·φx·φy·φxy + φyy·φx²) / (φx² + φy² + ε)^(3/2)      O(h²)   (partials from A1/A3; ε guards |∇φ|→0)
```
*Osher–Sethian 1988 JCP 79(1):12–49; Sethian 1999.*

**A6 — Boundary stencils.**
```
one-sided 2nd-order:  f'[0]  = (−3f0 + 4f1 − f2)/(2h)     f''[0] = (2f0 − 5f1 + 4f2 − f3)/h²
ghost-node Neumann / no-flux mirror:  f_ghost = f_edge  →  reduced-neighbor Laplacian at the edge
```
`f_ghost=f_edge` ⇔ scipy.ndimage `mode='reflect'` (half-sample) / `'nearest'`. *LeVeque 2007 §2.12; scipy.ndimage docs.*

**A7 — Savitzky–Golay derivative kernels (noise-robust, grid-native).** Local LS polynomial fit → a FIXED convolution
`a = (XᵀX)⁻¹Xᵀ d` on a uniform grid. 5-point kernels:
```
smoother (quad/cubic):        [−3, 12, 17, 12, −3] / 35
1st deriv (linear/quadratic): [−2, −1, 0, 1, 2] / (10h)
1st deriv (cubic/quartic):    [1, −8, 0, 8, −1] / (12h)      (≡ the 4th-order central A1)
2-D separable:  K_∂x = d1_x ⊗ s_y   (rows: derivative kernel; cols: smoother)
```
⚠️ **Label nuance:** `[−2,−1,0,1,2]/10h` is the *linear/quadratic* first-derivative kernel, NOT cubic. *Savitzky–Golay
1964 Anal.Chem. 36:1627 (+ Steinier 1972 corrections; Gorry 1990); Krumm 2001 (2-D).*

**A8 — Mimetic / staggered `div=−grad*` (the exact-Laplacian construction).**
```
GRAD cell→face (forward):  (grad f)[i+½] = (f[i+1] − f[i]) / h
DIV  face→cell (backward): (div g)[i]   = (g[i+½] − g[i−½]) / h
⇒  div(grad f) = the COMPACT 5-point Laplacian EXACTLY;   ⟨DIV g, f⟩ = −⟨g, GRAD f⟩ + bdy  ⇒  DIV = −GRAD*
```
*Harlow–Welch 1965 Phys.Fluids 8:2182; Hyman–Shashkov 1997 (CMA 33:81 & ANM 25:413); Lipnikov–Manzini–Shashkov 2014.*

### B. Integral operators (quadrature)
**B1 — 1-D building blocks** (nodes `x_k=a+kh`):
```
midpoint (open):  h·Σ f_{k+½}                                              O(h²)
trapezoidal:      h·(½f0 + f1 + … + f_{N−1} + ½fN)                         O(h²)   [DLMF 3.5.2/3.5.3]
Simpson (N even): (h/3)(f0 + 4f1 + 2f2 + … + 4f_{N−1} + fN)               O(h⁴)   [DLMF 3.5.7/3.5.8]
```
*NIST DLMF §3.5 (error terms fetched-verified); Press NR §4.1; Atkinson 1989 §5.2.*

**B2 — 2-D region integral `∬_Ω f dA`** (tensor products):
```
midpoint/Riemann:  Σ_{i,j∈Ω} f[i,j]·dx·dy                                  O(h²)   ← workhorse for a MASKED Ω
trapezoidal:       node weight dx·dy·{interior 1, edge ½, corner ¼}        O(h²)
Simpson (even int. each axis): per 3×3 patch (dx·dy/9)·[1 4 1; 4 16 4; 1 4 1]   O(h⁴)  ← loses order on a
                                                                                  staircased mask; rect sub-blocks only
```
*Davis–Rabinowitz 1984 Ch.5; LeVeque 2002 Ch.4 (masked/cut cells).*

**B3 — Polygon area (shoelace / Green).**
```
A = ½·∮(x dy − y dx) = ½·|Σ_k (x_k·y_{k+1} − x_{k+1}·y_k)|      exact for polygons; sign = orientation (+ = CCW)
```
*Green's theorem; Braden 1986 (surveyor's/shoelace).*

**B4 — Line / flux integral along a polyline** (`P_k`, `dP_k=P_{k+1}−P_k`, midpoints `P̄_k`):
```
circulation:  ∮F·dl   ≈ Σ_k F(P̄_k)·dP_k                                    O(h²)
flux:         ∮F·n̂ ds ≈ Σ_k (F·n̂)_k·|dP_k|
outward normal: n̂ = (t_y, −t_x)  (−90° of unit tangent, CCW contour)  or  ∇φ/|∇φ|
F at sub-pixel points by BILINEAR interpolation.
```
*Atkinson 1989 §5.2 (arc-length midpoint); standard contour practice (skimage.measure).*

**B5 — Discrete integral theorems — EXACT by telescoping.**
```
finite-volume div:  (div F)[i,j] = (Fx[i+½,j]−Fx[i−½,j])/dx + (Fy[i,j+½]−Fy[i,j−½])/dy
  ⇒  Σ_{Ω}(div F)·dx·dy  ==  Σ_{∂Ω} F·n̂·(face length)     EXACT  (interior faces cancel pairwise)
Stokes: same telescoping in circulation form (corner curls) → only the boundary loop survives.
SBP quadrature: the diagonal norm H = h·diag(½,1,…,1,½) (= trapezoid) is the weight making ⟨u,v⟩=uᵀHv
  consistent with the SBP derivative → discrete integration-by-parts / divergence theorem exact.
```
*LeVeque 2002 Ch.4; Perot 2000 JCP 159:58; Kreiss–Scherer 1974; Svärd–Nordström 2014; Hicken–Zingg 2013.*

**B6 — Isochrone extraction — marching squares.**
```
per 2×2 cell: 4-bit corner classify vs level L → 16-case lookup; linear edge crossing
  t* = (L − f_a)/(f_b − f_a),  P = P_a + t*·(P_b − P_a)     O(h²) position
saddle (cases 5/10): resolve by the cell-center bilinear value (skimage `fully_connected`).
```
*Lorensen–Cline 1987 (marching cubes, 2-D case), DOI 10.1145/37402.37422; skimage.measure.find_contours.*

**B7 — Winding number / topological charge** (2×2 plaquette, phase φ):
```
N = (1/2π)·Σ_links W(Δφ),   W(x) = atan2(sin x, cos x)     PS where π < |Σ| < 3π, charge = sign(Σ)
```
*Iyer–Gray 2001 Ann.Biomed.Eng. 29:47; Bray–Wikswo 2002 Phys.Rev.E 65:051902.*

### Operator → stencil → theorem/check map
| field / integral | stencil | theorem / cross-check |
|---|---|---|
| `source_sink` = ∇·(D∇V) | A4 / A8 (staggered) | B5 divergence theorem (`∬div = ∮F·n`) |
| `speed`/`velocity`/`direction` | A1 or A7 (grad of LAT) | eikonal `|∇T|=1/c`; isochrone-spacing B6 |
| `curvature` κ=∇·n̂ | A5 (or A2 of n̂) | B6 isochrone curvature; Gauss–Bonnet `∮κ ds=2π` |
| `vorticity` / winding | A2 curl / B7 | B5 Stokes (`∮v·dl=∬curl v`) |
| `conduction_time` | B4 line-integral of ∇T (A1) | path-independence `∮∇T·dl=0` / `=CL` reentry |
| `net_flux`, activated area | B2 / B5 | divergence theorem |

### Implementation contract (one line)
2nd-order central (A1) with the **compact 5-point Laplacian realized via the staggered `div=−grad*` pair (A8)**,
`face_mirror` ghost boundary (A6), **Bayly/Savitzky–Golay kernels (A7)** for the noise-robust CV field, **midpoint
region quadrature (B2)** + **arc-length line integrals (B4)** on **marching-squares contours (B6)** — and hold the
operators to the adjoint so the divergence/Stokes theorems (B5) hold to machine precision as free self-checks.
Everything else composes from these.

### Calculation references
Fornberg 1988 *Math.Comp.* 51:699 (DOI 10.1090/S0025-5718-1988-0935077-0) · LeVeque 2007 *Finite Difference
Methods for ODEs/PDEs* (SIAM) · LeVeque 2002 *Finite Volume Methods* (CUP, DOI 10.1017/CBO9780511791253) ·
Collatz 1960 *The Numerical Treatment of Differential Equations* (Mehrstellen) · Patra–Karttunen 2006 *NMPDE*
22:936 · Osher–Sethian 1988 *JCP* 79:12 (DOI 10.1016/0021-9991(88)90002-2) · Sethian 1999 *Level Set Methods…*
(CUP) · Savitzky–Golay 1964 *Anal.Chem.* 36:1627 (DOI 10.1021/ac60214a047) · Steinier 1972 *Anal.Chem.* 44:1906 ·
Gorry 1990 *Anal.Chem.* 62:570 · Krumm 2001 (2-D SG, MSR) · Harlow–Welch 1965 *Phys.Fluids* 8:2182 (DOI
10.1063/1.1761178) · Hyman–Shashkov 1997 *CMA* 33:81 & *ANM* 25:413 · Lipnikov–Manzini–Shashkov 2014 *JCP* 257:1163
· NIST DLMF §3.4/§3.5 · Press et al. 2007 *Numerical Recipes* 3e §4.1 · Atkinson 1989 *Intro to Numerical Analysis*
§5 · Davis–Rabinowitz 1984 *Methods of Numerical Integration* Ch.5 · Braden 1986 *Coll.Math.J.* 17:326 · Perot 2000
*JCP* 159:58 · Kreiss–Scherer 1974 · Svärd–Nordström 2014 *JCP* 268:17 · Hicken–Zingg 2013 *JCAM* 237:111 ·
Lorensen–Cline 1987 *SIGGRAPH* (DOI 10.1145/37402.37422) · Iyer–Gray 2001 *Ann.Biomed.Eng.* 29:47 · Bray–Wikswo
2002 *Phys.Rev.E* 65:051902.

## Mathematical foundations — per field (the derivations)

> Full prior-art + method survey: [ANALYSIS_METHODS_PRIOR_ART.md](./ANALYSIS_METHODS_PRIOR_ART.md). This section
> is the **math each field rests on**, written to one method: for every quantity we give **(a)** the fundamental
> definition, **(b)** the *equivalent* ways to compute it, **(c)** why they are equal, **(d)** the trap where they
> diverge, **(e)** the canonical choice. The equivalences are the payload — they separate what is *fundamental*
> from what is *incidental* (e.g. LAT is incidental to the velocity field), and every equivalence doubles as a
> **free consistency check** in the test suite. Math is ASCII/Unicode.

### 0. The master relation (everything descends from one equation)
The monodomain reaction–diffusion equation is the parent of every field here:
```
Cm ∂V/∂t = ∇·(D∇V) − I_ion               (+ I_stim)
             \_______/    \___/
              diffusion   reaction
```
Read the terms:
- **reaction** `I_ion(V, states)` → the **`single_cell`** 0-D mode (§ 9) is this term with diffusion set to 0.
- **diffusion** `∇·(D∇V)` → the **`source_sink`** field (§ 3): the net axial current arriving at a point.
- the **eikonal reduction** of the whole PDE (asymptotics of a sharp front) → **`velocity`/`direction`/`speed`** (§ 2).
- the **next-order** eikonal term → **`curvature`** via `CV = CV0 − D·κ` (§ 4).
- the **topology** of the resulting field → **`vorticity`/rotors** (§ 5).

So the fields are not a grab-bag: they are the terms of this equation and the successive reductions of it. That is
why `source_sink` and `single_cell` are the two halves of the same operator split, and why the velocity field and
the source–sink map are the same physics seen at two scales.

### 1. LAT — activation time `T(x)`
**Definition.** The instant the wavefront arrives: `T(x)` is defined implicitly by `V(x, T(x)) = V_thr` (threshold
route), or by `argmax_t ∂V/∂t` (max-upstroke route).
**Equivalent routes & the key identity.** Differentiate the defining relation `V(x, T(x)) = V_thr` in space:
```
∂V/∂x + (∂V/∂t)·(∂T/∂x) = 0     ⇒     ∇T = −∇V / (∂V/∂t)
```
This is the bridge between the *time-collapsed* view (LAT) and the *raw-field* view (V and its derivatives). The
threshold and max-dV/dt routes differ only by a near-constant offset along the upstroke, which **cancels in ∇T** —
so for anything built on ∇T (all of § 2, 4) the absolute criterion barely matters; using **one** criterion
everywhere is what matters.
**Trap.** Nearest-save-frame LAT is quantized to `save_every` → `∇T` staircases and `|∇T|→0` between equal-LAT
neighbors → CV blows up. Under reentry `T` is multivalued (§ 5).
**Canonical.** Linearly-interpolated crossing at ONE frozen threshold (−40 mV), torch/on-device; max(dV/dt) with
parabolic sub-frame refine offered as a probe-point reference.

### 2. `velocity` / `direction` / `speed` — and why LAT is *incidental*
**Definition (eikonal).** A wavefront is the level set `{T(x)=const}`; its normal is `n̂ = ∇T/|∇T|` and the
front-normal speed satisfies the eikonal identity `|∇T| = 1/c`. Hence:
```
speed  c = 1/|∇T|        direction n̂ = ∇T/|∇T|        velocity v = ∇T/|∇T|²
```
(The `1/|∇T|²` — not `1/|∇T|` — because the front moves *along* ∇T but its *speed* is `1/|∇T|`.)
**Two equivalent routes — one uses LAT, one does not.**
- *LAT-gradient* (uses LAT): form `T(x)`, then `v = ∇T/|∇T|²`.
- *Optical flow* (no LAT): treat `V(x,y,t)` as a moving image; if the pattern is advected with velocity `v`, its
  total derivative vanishes → the **brightness-constancy** equation, which contains only V and its derivatives:
  ```
  ∂V/∂t + ∇V·v = 0      ⇒      normal speed  v_n = −(∂V/∂t)/|∇V|
  ```
**Why they are equal.** Substitute the § 1 identity `∇T = −∇V/(∂V/∂t)`:
```
c = 1/|∇T| = |∂V/∂t| / |∇V| = v_n           ← identical
```
So **LAT is incidental**: the fundamental quantity is the ratio of the field's *time* derivative to its *space*
derivative. LAT-gradient computes it by first collapsing the time axis to one number then differentiating in space;
optical flow differentiates the raw movie in space AND time. Same relation, two factorizations. The consequence for
the API: the LAT route gives **one** vector per pixel (assumes steady propagation) and needs an activation
criterion; the optical-flow route gives a velocity field **per frame** and needs none — so it, not LAT, is the
fallback for **fibrillation/reentry** where "arrival time" is undefined.
**Trap.** `|∇T|→0` at collisions, foci, rotor cores → `c,v` singular (mask it — § 5); optical flow has the
**aperture problem** (recovers only the normal component; needs Horn–Schunck/Lucas–Kanade regularization for the
tangential part).
**Canonical.** LAT-gradient via a Bayly local-polynomial fit (structured-grid Savitzky–Golay kernel) with divergence
gating; optical-flow as the documented reentry-regime alternative.

### 3. `source_sink` = ∇·(D∇V) — computable THREE ways
**Definition.** `voltage_flux = D∇V` (diffusion flux of voltage); `source_sink = ∇·(D∇V) = D∇²V` (isotropic) — the
net axial/gap-junctional current density arriving at a point (`<0` = local source / front crest, `>0` = local sink).
**Three equivalent routes (this is the "process" in its purest form):**
1. **Spatial** — `∇·(D∇V)` directly (the diffusion stencil).
2. **Boundary flux** (divergence theorem) — over a region Ω: `∬_Ω ∇·(D∇V) dA = ∮_∂Ω (D∇V)·n̂ dl`. The region source
   equals the net flux through its boundary.
3. **Temporal + ionic** — rearrange the master equation (§ 0): `∇·(D∇V) = Cm ∂V/∂t + I_ion`. You can get the
   source–sink map from the *trace* (`Cm·V_t`) plus the ionic current the engine already computes — **no spatial
   derivative at all.**
**Why they are equal.** (1)≡(2) is Gauss's theorem; (1)≡(3) is the governing PDE itself. All three agreeing is a
whole-pipeline consistency test (and `source_sink` is literally the diffusion term the solver stepped, so route (1)
must reproduce the solver's own operator — see the boundary-mode rule).
**Downstream.** The time-integral of `source_sink` over the upstroke IS the numerator of the Boyle–Vigmond safety
factor (§ 8): `SF = [(1/β)∫_A ∇·(σ∇V) dt] / Q_thr = [Cm·ΔV + Q_ion − Q_s]/Q_thr`.
**Trap.** Route (1) must use the **solver's own stencil** (mimetic `div=−gradᵀ`) or it carries a stencil-mismatch
error that reads as spurious source/sink; route (3) needs the ionic current at the same instant.
**Canonical.** Route (1) with the solver's operators; expose route (2) as `net_flux` and route (3) as the
consistency check.

### 4. `curvature` κ = ∇·n̂ — four routes to one number
**Definition.** Curvature of the wavefront = divergence of the unit normal:
```
κ = ∇·n̂ = ∇·(∇T/|∇T|) = (Txx·Ty² − 2·Tx·Ty·Txy + Tyy·Tx²) / (Tx² + Ty²)^{3/2}
```
**Equivalent routes.** (a) divergence of the normalized LAT gradient; (b) the level-set formula above (same thing,
expanded — all five partials come free from ONE Bayly window fit, `Tx=a10, Ty=a01, Txx=2a20, Tyy=2a02, Txy=a11`);
(c) the curvature of the marching-squares isochrone (a geometric contour, no gradient); (d) inverted from measured
speed via the eikonal-curvature law `CV = CV0 − D·κ`.
**Why equal.** `∇·n̂` of a level set *is* the geometric curvature of that level set — (a)=(b)=(c) identically; (d) is
the physics (Keener/Colli-Franzone asymptotics) linking κ to how much the front is slowed.
**Trap.** κ is a ratio of second derivatives to `|∇T|³` → high-pass → amplifies noise, and blows up at `|∇T|→0`.
**Canonical.** Route (b) from the Bayly fit; cross-check against (c) as the free geometric test.

### 5. `vorticity` / circulation / winding — ONE primitive, three fields
**Definition.** The loop integral of a field around a closed contour equals `2π ×` (an integer topological charge):
```
N = (1/2π) ∮_C  (field)·dl
```
**The same primitive on three fields:**
- **phase** φ (from Hilbert transform of V): `(1/2π)∮∇φ·dl = ±1` per rotor = **phase singularity** count.
- **velocity** v: `∮v·dl = ∬ (∇×v) dA` (Green/Stokes) → **circulation** = enclosed **vorticity** → rotor.
- **isochrone tangent**: `∮κ ds = 2π·m` (Gauss–Bonnet) → **m enclosed rotational cores**.
- **LAT gradient**: `∮∇T·dl = 0` (curl-free) normally, `= CL` (one cycle length) around a **reentry** loop — because
  `∇×(∇T)≡0` unless `T` is multivalued.
**Why equal.** All four are the winding number / topological charge; Stokes ties the velocity line-integral to the
area integral of its curl; `∇×(∇anything)≡0` is why `∮∇T·dl` and `∮∇φ·dl` measure topology, not local flow.
**Trap.** `curl(grad(V))≡0` — never expose it as a metric (the precision principle). LAT-based winding fails under
reentry (T multivalued) → use the **phase** field there.
**Canonical.** Implement one wrapped-loop-sum (2×2 plaquette: `Σ = Δφ_top − Δφ_right − Δφ_bottom + Δφ_left`, PS
where `π<|Σ|<3π`, charge = sign, CCW positive); reuse it for vorticity, circulation, Gauss–Bonnet, and the reentry
check.

### 6. Bidomain current fields — `electric_field`, `current_flux`
**Definition.** `electric_field = −∇φ_e`; `current_flux = −σ_e∇φ_e` (Ohm in tissue); current source density
`∇·(σ_e∇φ_e) = β·I_m`. In the monodomain limit `∇·(σ_e∇φ_e)` collapses to `source_sink` (§ 3) — the same operator
on the extracellular potential. So these reuse the § 3–4 machinery with `φ_e` in place of `Vm` and `σ` in place of
`D`.

### 7. LAT integrals — the consistency web (and the slowness-vs-velocity trap)
**Conduction time.** `∇T` is a gradient ⇒ conservative ⇒ for ANY path C:
```
∫_C ∇T·dl = T(B) − T(A)      (exact, path-INDEPENDENT — the Fermat/eikonal travel-time integral)
```
so conduction time between two sites is just `ΔT`; path CV = `|C| / (T(B)−T(A))`.
**⚠️ The trap.** This integrates the **slowness `σ = ∇T`**, NOT the velocity `v = ∇T/|∇T|²`. Only `∇T` is curl-free;
`∫v·dl ≠ ΔT`. Conduction-time code must line-integrate ∇T, never the stored `velocity`.
**Free checks from the same identity.** `∮∇T·dl = 0` on any loop (curl-free — a discrete ∇T from bad interpolation
FAILS this, so it detects bad fields); `∮∇T·dl = CL` around a reentry loop (§ 5).
**Isochrone-spacing CV ≡ 1/|∇T|.** Stepping `Δs` along `n̂` changes T by `|∇T|·Δs`; one isochrone increment means
`ΔT=Δt`, so `Δs = Δt/|∇T|` and `CV = Δs/Δt = 1/|∇T|`. The classical geometric method and the eikonal speed are the
SAME number — a mutual regression test.
**Co-area identity.** `dA/dt = ∮_{T=t}(1/|∇T|)ds = ∮_{T=t} CV ds = L(t)·⟨CV⟩` ties activated-area rate ↔ front
length ↔ speed — a third whole-field consistency gate.

### 8. Scalar EP metrics
- **Wavelength** `λ = CV·ERP` (reentry master variable = minimum circuit length); `CV·APD90` is a proxy that
  *underestimates* λ under post-repolarization refractoriness (ERP>APD90). Units: `λ_cm = CV_cm/s · ERP_ms / 1000`.
- **APD** `APDx = t_repol(x) − t_act`, `V_repol(x) = V_peak − (x/100)(V_peak−V_rest)`; dome-aware last-crossing.
- **Restitution** `APD_{n+1} = f(DI_n)`, `DI_n = BCL − APD_n`; `max|dAPD/dDI| > 1` ⇒ alternans (the slope-1 boundary).
- **ERP** — a *protocol*, not a trace read: S1S2 extrastimulus, ERP = longest S1S2 that fails to capture (bisection).
  This is the one metric that must **run simulations**.
- **Safety factor** `SF_VB = [Cm·ΔV + Q_ion − Q_s]/Q_thr(t_A)` — built on the § 3 source–sink integral + a one-time
  single-cell threshold-charge curve; SF<1 ⇒ block.

### 9. `single_cell` — the reaction term alone
**Definition.** The master equation (§ 0) with diffusion removed: `Cm dV/dt = −(I_ion + I_stim)`, integrated with
**Rush–Larsen** gates `g_{n+1} = g∞ + (g_n − g∞)·exp(−Δt/τ)` + forward-Euler V and concentrations.
**The unifying point.** `single_cell` (reaction) and `source_sink` (diffusion) are the two terms of § 0 — together
they *are* the operator split the tissue solver uses. Routing `single_cell` through the **same per-node ionic step**
the tissue reaction substep calls is what makes them consistent (and sidesteps the ORd concentration-ordering bug).
It also produces the `Q_thr(t_A)` curve the safety factor (§ 8) needs.
**Canonical.** Shared ionic step, FE+RL default (Δt 0.02 ms TTP06 / 0.005–0.01 ms ORd), stimulus into dV/dt only,
`pre_pace` to steady state (~1000 beats ORd-class).

### Other analysis operations (existing `analysis.py`, non-field) — same method
> § 1–9 are the fields branch. These are the remaining `analysis.py` operations, documented the same way. Several
> are the *point/scalar* or *frequency-domain* counterparts of a field quantity. (Already covered above:
> `activation_time`/`_interp`→§1; `front_metrics`/`fit_eikonal`→§2,4; `phase_singularities`→§5; `apd_at`/`apd_map`/
> `apd_per_beat`/`restitution_curve`/`restitution_slope`→§8.)

#### 10. Dominant frequency — `dominant_frequency`, `dominant_frequency_map`
**Definition.** Frequency-domain summary of a trace: `DF = argmax_f |rfft(V − mean)(f)|`, DC bin zeroed; `df_map`
runs it batched per node.
**The process / relation.** LAT/CV/APD assume you can *annotate discrete activations*; when activation is too
complex to annotate (fibrillation, multiple fronts) the trace is summarized by its dominant **rate** instead. For a
(quasi-)periodic signal `DF ≈ 1/period`, and for reentry `DF ≈ 1000/CL_ms` Hz — so DF is the **spectral proxy for
the activation rate** that cycle length CL and (indirectly) the wavelength express in the time domain. Equivalent
routes: FFT peak (ours) vs autocorrelation-lag vs mean inter-activation interval — all estimate the same rate; they
diverge on broadband/irregular signals (FFT picks the strongest spectral line, interval-based the mean).
**Trap.** Frequency resolution `1/(n·dt)` — differences below it are unresolved (we warn); a strong **harmonic** can
beat the fundamental; masked/NaN nodes must return NaN, not a phantom low frequency; DC must be removed.
**Canonical.** rfft of the detrended trace, DC zeroed, argmax; warn when `1/(n·dt) > 0.5 Hz`.

#### 11. Phase map — `phase_map`
**Definition.** Instantaneous phase of the oscillation at every node: form the analytic signal by Hilbert transform
(one-sided spectrum `h[0]=1, h[1:n/2]=2, h[n/2]=1`) of the detrended trace, then `φ(x,t) = atan2(Im, Re) ∈ (−π,π]`.
**The process / relation.** Phase turns "where in the AP cycle is this pixel *right now*" into one angle that
advances 2π per beat — and it is defined **even when LAT is not** (fibrillation, re-excitation), which is exactly
why the rotor machinery (§ 5) is built on φ, not T. Equivalent routes to a phase: Hilbert (ours), state-space
embedding `φ = atan2(V(t−τ)−R*, V−V*)`, or the V–dV/dt plane — all wind 2π per cycle; they differ in noise
sensitivity and the τ/offset choices. **Relation to LAT:** on the upstroke φ sweeps through 0, so the `φ=0`
iso-contour at instant t ≈ the wavefront ≈ the LAT isochrone `{T=t}` — φ is the LAT-free generalization.
**Trap.** Hilbert assumes a narrowband analytic signal → detrend first; non-stationary/transient traces distort φ;
FFT wrap-around at the record ends. Feeds `phase_singularities` (§ 5).
**Canonical.** FFT-based analytic signal on the detrended trace; `φ = atan2(Im, Re)`.

#### 12. Wavefront mask — `wavefront_mask`
**Definition.** The instantaneous front: `front = {V ≥ θ} ∩ {∃ 4-neighbor with V < θ}` — super-threshold nodes
touching sub-threshold tissue = the discrete boundary `∂{V ≥ θ}` of the depolarized region.
**The process / relation.** This is the **instantaneous spatial level set** of V (boundary of the activated set at
fixed t) — the sibling of the *isochrone* (a level set of T). So `wavefront_mask` at time t ≈ the isochrone `{T=t}`;
the activated **area** `A(t)=|{V≥θ}|·dA` and **front length** `L(t)=|wavefront_mask|·spacing` are the § 7 co-area
quantities (`dA/dt = L·⟨CV⟩`). It is the discrete, threshold version of marching-squares isochrone extraction.
**Trap.** Front *thickness* is threshold- and resolution-dependent (a discrete boundary is ≥1 px thick, not a
curve) → for arc-length/curvature use the sub-pixel marching-squares contour, not this mask; θ-sensitive.
**Canonical.** Boundary of the super-threshold set for masking/area; marching-squares for geometric integrals.

#### 13. Scalar / two-point CV — `conduction_velocity`, `cv_between`, `radial_cv`
**Definition (all one formula — the secant CV).** `CV = distance / (LAT difference)`:
- `conduction_velocity`: two x-indices on a row, `CV = |x2−x1|·dx / (T2−T1)`.
- `cv_between`: any two nodes, `CV = ‖p2−p1‖ / (T2−T1)` (Euclidean).
- `radial_cv`: from a point source `c`, each node's `CV = ‖x−c‖ / (T(x)−T(c))`.
**The process / relation.** These are the **secant** (finite-baseline, two-point) estimate of conduction speed; the
fields-branch `velocity = ∇T/|∇T|²` is the **differential** (local-gradient) estimate — the **same eikonal quantity
at two granularities**: as the baseline → 0 the secant `Δs/ΔT → 1/|∂T/∂s|` = the gradient speed along that line
(exactly the § 7 path-CV `|C|/(T(B)−T(A))` for a straight chord).
**Why/when they differ.** The secant returns the **projected** speed along the chord (component along the line), so
it under-reads unless the chord follows the propagation direction; and it **averages** over any curvature between
the endpoints (the differential value is local). `radial_cv` additionally assumes a single point source at `c`.
So: secant for a quick *directed spot* measurement or a point-source spread map; the vector field (§ 2) for the
full, direction-resolved map.
**Trap.** Nearest-frame LAT → quantized ΔT → the § 1 staircase; a chord off the propagation axis gives a projected
(too-slow) CV; `radial_cv` is meaningless if `center` isn't the actual source.
**Canonical.** Keep as the lightweight two-point/point-source tools, but route their LAT through the canonical
interpolated LAT (§ 1) so ΔT isn't frame-quantized; document that they return a **projected/secant** speed, not the
local vector.

## `fields.derivatives` — API (local operators, field → field)
**Primitive operators only** (the machinery; named physical fields live in `fields.derived`). Torch,
on-device, per-snapshot; accept `(Nx,Ny)` or `(T,Nx,Ny)`:
- `grad(scalar) -> vector`  (returns `(...,2)`; components ∂/∂x, ∂/∂y)
- `div(vector) -> scalar`
- `curl(vector) -> scalar`  (2-D curl = ∂vy/∂x − ∂vx/∂y, the z-component)
- `laplacian(scalar) -> scalar`  (= `div(grad(·))`)

### Vector-field representation (DECISION)
A vector field is stored as **components on the LAST axis, `(..., 2)`** — `grad(Vm (T,Nx,Ny))` →
`(T,Nx,Ny,2)`; a LAT-based `velocity` → `(Nx,Ny,2)` (no time axis). Same rule whether or not a time
axis is present (the leading `...` absorbs it). Rationale: it's what `grad` naturally returns,
`[...,0]/[...,1]` are x/y, `norm(v, dim=-1)` is the magnitude, and the integral dot-products
(`(v*n).sum(-1)` for flux/circulation) are clean. Wrap it in a light `VectorField` so users never
index a raw axis: `.x`, `.y`, `.magnitude`, `.angle`, `.components` (the raw `(...,2)` tensor).
On disk / cache: store the `(...,2)` tensor (npz-friendly). NOT chosen: separate `vx,vy` tensors or
`(2,...)` first-axis (both fight the scalar shape and broadcasting).

## Named fields — `r.fields.<name>` (pre-saved, cached)
The canonical fields worth naming — the ones you visualize AND feed to `integrals`. Names say WHAT
they act on; each **commits to its base field + operator** (so the identity-zero trap can't happen):

| `r.fields.…` | definition | base | meaning |
|--------------|-----------|------|---------|
| `voltage_gradient` | ∇Vm | Vm | steepest-ascent of V (large at the front) |
| `voltage_flux` | D∇Vm | Vm | diffusion flux of voltage; `div(voltage_flux)` = `source_sink` |
| `source_sink` | ∇·(D∇Vm) = D∇²Vm | Vm | **electrotonic source–sink map** (the source–sink research field) |
| `current_flux` | −σ∇φ_e | φ_e | current field (bidomain); `div` = current source density |
| `electric_field` | −∇φ_e | φ_e | extracellular E-field (bidomain) |
| `velocity` | ∇LAT / \|∇LAT\|² (= CV·n̂) | LAT | conduction-velocity vector field |
| `direction` | ∇LAT / \|∇LAT\| = n̂ | LAT | unit propagation direction |
| `speed` | 1/\|∇LAT\| | LAT | front-normal conduction speed (`front_metrics.cv_n`) |
| `curvature` | ∇·n̂ | LAT | **wavefront curvature** (`front_metrics.kappa`) |
| `vorticity` | curl(velocity) | LAT | rotation → **rotor cores** |

**Cached / pre-saved (the point):** each is computed ONCE and cached (lazily on first `r.fields.<name>`
access), because it's expensive-ish (gradients over all frames) and reused — plot `velocity` and
compute its `circulation` from the same array; take `voltage_flux` and its `net-flux` integral from
one flux array. **The cache lives for the result's lifetime — `SimulationResult` is an IMMUTABLE post-run
snapshot, so there is nothing to invalidate** (`scale_conductance`/`reset` are SIM methods; they produce a NEW
run → a NEW result → a fresh accessor; a stale cache is impossible because the Vm/LAT can't change under a result).
Optional eager mode: record chosen named fields alongside `Vm` during the run (heavier; connects to the deferred probe).

## GATE: scrutinize LAT *before* trusting the LAT-based fields (TODO, later)
Half the named fields (`velocity`, `direction`, `speed`, `curvature`, `vorticity`) are built on the
activation-time map `LAT`, so they INHERIT every definitional choice in `LAT`. Pin + document these
first, or the fields are precise numbers on a shaky base.

**CONCRETE FINDING (2026-07-21): there are already TWO disagreeing LAT definitions.**
- `activation_time` (default; `r.lat()`): first frame where `V ≥ −20 mV`, `times[first_idx]` —
  **nearest save-point, NO interpolation** (LAT resolution = `save_every`). torch.
- `activation_time_interp`: **linearly-interpolated** crossing at `V = −40 mV`, sub-frame accurate,
  numpy/CPU — exists because eikonal `CV = 1/|∇LAT|` needs sub-frame accuracy.
They differ on **threshold (−20 vs −40)** and **accuracy (nearest-frame vs interpolated)**, so `r.lat()`
and the eikonal path yield DIFFERENT CV/curvature from the same run. Neither uses max-`dV/dt`. Before
the LAT fields, pick ONE canonical LAT (recommend: interpolated crossing, single agreed threshold,
torch/on-device) and route `r.lat()` + eikonal + the named fields all through it.

**BLAST RADIUS — the two conventions are split across the DEFAULT hooks vs the RESEARCH path:**
- `−20 mV, nearest-frame`: `r.lat()` (`activation_time`), `r.cv()` (`conduction_velocity` computes its
  OWN nearest-frame crossing at −20), and `apd_map` (uses `activation_time` as reference). = what a
  casual user gets.
- `−40 mV, interpolated`: `activation_time_interp` → `test_eikonal_metrics`, `front_metrics`, the
  **`source_sink_mismatch_investigation`** research + the **`fig4c_sourcesink`** experiments. = what
  the source–sink / curvature figures were actually made with.
So `r.cv()` (−20 nearest) ≠ the eikonal CV (−40 interp) on the SAME run — a silent, undocumented
discrepancy. Documented as a real finding in engine_consolidation KNOWLEDGE + IDEALOG.

Full scrutiny checklist:
- **Activation criterion** — threshold crossing (V > θ) vs max-`dV/dt` (upstroke) vs interpolated
  crossing. Changes `LAT`, hence CV and (especially) curvature. Confirm what `activation_time` uses.
- **Sub-frame interpolation** — `LAT` resolution is capped by `save_every` unless the crossing time
  is interpolated between frames; coarse `LAT` → noisy gradient → noisy curvature. (`activation_time_interp`
  exists — is it the default?)
- **Non-activating nodes** — scar/block never crosses → NaN; how do grad/div behave with NaN
  neighbors at a block edge? (ties to the domain_mask boundary rule.)
- **Multi-beat / re-activation** — with pacing or REENTRY a node activates many times; `LAT` = first
  crossing is ill-defined. For reentry, `LAT` breaks down → use **phase** (`phase_map` /
  `phase_singularities`), not `LAT`-based velocity/curvature. Document this limit loudly.
- **Threshold sensitivity** — CV and curvature are sensitive to θ; expose it, don't hardcode.
This is a review GATE, not a blocker for the Vm/φ_e fields (which don't touch `LAT`).

## Boundary handling — SAME as the tissue edge boundary (DECISION)
The derivative stencils MUST use the **same edge treatment as the simulation** (its `boundary_mode`,
default `face_mirror` = no-flux / Neumann mirror), so a post-hoc `∇²Vm` equals the electrotonic
source the solver actually saw — not a numpy-default one-sided edge. Consequences to implement:
- **Carry the boundary mode**: `SimulationResult` currently holds only `dx/dy/Vm/times`; add the
  `boundary_mode` (and grid/`domain_mask`) so field ops can honor it, or take it as an argument.
- **Internal boundaries too**: a scar/hole (`domain_mask`) is a no-flux edge — stencils must respect
  the mask (mirror / one-side at hole borders, NaN masked-out nodes) or divergence/curvature blows
  up at hole edges. Reuse the engine's `boundary_mode`/`stencil` convention, not a generic edge rule.

## `fields.integrals` — API (global reductions, field → number)
Global line/region integrals. Each is a Stokes/divergence-theorem partner of a `fields.derivatives`
operator, so the two tiers cross-check (see Consistency test).

### Line / contour integrals ("global curvature" family)
| quantity | integral | = (theorem) | meaning |
|----------|----------|-------------|---------|
| global curvature | `∮ κ ds` on isochrone | Gauss–Bonnet: net turning | wavefront integrated curvature |
| circulation | `∮ v · dl` around loop | `∬ curl(v) dA` (Stokes) | enclosed vorticity → **rotor** |
| conduction time | `∫ ∇LAT · dl` on path | `LAT(end) − LAT(start)` | traversal time; ÷ arc-length = path CV |
| wavefront length | `∮ ds` on isochrone | — | front perimeter (source size) |
| winding number | `∮ ∇φ · dl / 2π` | count of enclosed singularities | **# rotors** in the loop |

### Region / area ("volumetric flux") family
| quantity | integral | = (theorem) | meaning |
|----------|----------|-------------|---------|
| net flux through boundary | `∮ F · n dl` over ∂region | `∬ div(F) dA` (divergence thm) | **net source–sink inside** (source–sink balance) |
| activated area | `∬ 𝟙[V>θ] dA` | — | depolarized-area(t) + recruitment rate |
| total current / load | `∬ I_ion dA`, `∬ ∇²V dA` | — | region source / integrated electrotonic load |
| state fractions | region occupancy | — | excited / refractory fraction |

**2-D → areal** (per unit thickness, `dA`); generalizes verbatim to `dV` on a 3-D grid — the API
must NOT hardcode 2-D, but today's numbers are per-area.

### Ergonomics — regions & boundaries ARE mesh/mask objects (DECISION)
The user never hand-builds a contour or a measure — integration regions and boundaries are the
mesh/mask objects they already have (the SAME masks used for scars/stimuli). `mesh in → number out`:
- **Region integral**: `over=mask` — an `(Nx,Ny)` bool from `circle_mask`/`rectangle_mask`/
  `annulus_mask`/`domain_mask`. Default = the whole domain (`domain_mask`). Measure `dA = dx·dy`
  taken from the mesh.
- **Flux / boundary integral**: pass the SAME `region=mask` → the branch DERIVES the boundary
  `∂(mask)`, the OUTWARD normals, and the arc-length `ds` from the mesh geometry. Or
  `boundary="domain"` = the tissue's outer edge. No contour is hand-built.
- **Isochrone integrals** (global curvature, wavefront length): the contour is a `LAT` level set —
  select by `at_time=t` / `level=…`; extracted internally (marching-squares).
- Everything honors the SAME `boundary_mode` + `domain_mask` as the differential tier — a scar/hole
  edge is a real boundary, so its outward normal is included in a flux integral. Build a region
  ONCE, pass it as `over=`/`region=`, get the number.

### Orientation / sign conventions (a SPEC, not a detail)
- **Flux**: OUTWARD normal — positive = net efflux (source inside).
- **Circulation / winding**: counter-clockwise positive.
Documented AND asserted, so a source never reads as a sink and a rotor never flips its charge.

### Consistency test (free validation asset)
`∮ v·dl` vs `∬ curl(v)`, and `∮ F·n dl` vs `∬ div(F)`, must agree to discretization error. One unit
test per theorem validates BOTH tiers and the boundary handling at once.

## Relationship to existing code
- `analysis.front_metrics(lat, dx)` already computes cv_n, propagation direction (n_x,n_y), and
  κ = div(n̂) — i.e. it IS the LAT-based named fields (`speed`, `direction`, `curvature`), just
  numpy/CPU and standalone. `fit_eikonal` fits CV_n = CV0 − D·κ. Both STAY as-is for now.
- **Migrate later (documented intent, NOT now):** re-express `front_metrics`'s outputs as the
  `r.fields.{speed,direction,curvature}` named fields over the torch `fields.derivatives` primitives
  (one boundary-aware gradient implementation, not a numpy one and a torch one drifting apart); keep
  `fit_eikonal` as a thin consumer of `r.fields.speed` + `r.fields.curvature`. Do this once the
  primitive layer is proven — then front_metrics becomes a compatibility shim, not a 2nd impl.

## Adjacent analysis additions (SEPARATE TRACK — scalar EP metrics, not fields)
High-level clinical/EP metrics that COMPOSE field + point measurements into one number. Different
category from the field operators; live under top-level `analysis`, not `analysis.fields`. Wishlist:
- **`analysis.wavelength`** — the big one. **λ = CV · ERP** (the reentry master variable; `CV · APD`
  is a common proxy). Computing it by hand today is a pain: get CV, get ERP/APD, reconcile units,
  handle NaN/block. Make it one call with the choices EXPOSED: `λ = CV·ERP` vs `CV·APD`; CV local
  (at a site) vs global; which APD% / ERP definition. (ERP ≈ APD but rate-dependent — `CV·ERP` is
  the physiologically-correct reentry form; note in the ionic-optimization work λ is the master var.)
- **`analysis.apd`** — consolidate + complete: APD at % (APD90/50/30), APD restitution, per-beat.
  Partly exists (`apd_at`/`apd_map`/`apd_per_beat`) — unify + fill gaps.
- **`erp`, `di`, safety factor** — effective refractory period, diastolic interval, source–sink
  safety factor ("... or stuff").
Separate build track from the `fields` branch; captured here so it isn't lost.

## Out of scope here (separate items)
- **Probe** (point + dt-resolution recorder that these operators evaluate on for local-property
  time-series) — DEFERRED, dealt with later.
- **`r.grid(x,y)` / `r.coord(ix,iy)`** coord↔index ergonomics — small, separate, can land anytime.

## Open decisions
- Exact home/name: a `Fields` accessor on the result exposing the named fields directly
  (`r.fields.voltage_flux`) plus the `.derivatives`/`.integrals` toolkits — vs plain
  `cardiac_core/analysis/fields/` submodules. The user-facing target is `r.fields.<name>`.
- **Final field names** (adjustable): the catalog uses explicit names (`voltage_flux`, `voltage_
  gradient`, `current_flux`, `electric_field`, `speed`, …); confirm the exact set before building —
  esp. `velocity` vs `conduction_velocity`, `vorticity` vs `rotation`, `speed` vs `cv`.
- **Named-field cache**: lazy memoize on the result accessor (`r.fields.velocity`). **RESOLVED:** the result is
  immutable → no invalidation logic (mutating the sim yields a new result); just memoize per-name for the
  accessor's lifetime. Plus whether to offer an EAGER record-during-run mode (heavier; overlaps the probe).
- Whether operators default to `face_mirror` when no boundary is supplied, or require it explicitly.
- Second-order-interior vs matching the engine's exact `stencil` (`cardinal4` vs `moore8`) so
  curvature at edges is bit-consistent with the solved physics.
- **Where the mesh/boundary comes from for `.integrals`**: the result must expose `dx/dy`,
  `domain_mask`, and `boundary_mode` (the same addition the differential tier needs) so `over=`/
  `region=` can be a bare mask and the branch supplies the measure + normals. Decide: pass the mesh
  explicitly, or have `SimulationResult` carry it.
- **Contour extraction** for isochrone integrals (`∮κ ds`, wavefront length): marching-squares vs a
  co-area / level-set formulation; arc-length weighting to keep `∮κ ds` from being grid-noisy.
- **Mask-boundary normals**: deriving outward normals + `ds` from a discrete `(Nx,Ny)` mask edge
  (staircase) needs a convention (face-based vs smoothed) — pick one and pin it with the
  consistency test.
