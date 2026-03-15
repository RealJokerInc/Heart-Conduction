# The Case Against Bidomain Neumann+Dirichlet, and the Case for EMI

---

## Part I: Seven Arguments Against the Bidomain Boundary Treatment

The previous document (WHY_NEUMANN_DIRICHLET.md) established a seemingly airtight
logical chain: physical premises → bidomain equations → Neumann-intra + Dirichlet-extra
→ Kleber speedup. Here we attack each link in that chain.

---

### Argument 1: The Bidomain Is Not Valid at Tissue Boundaries

The bidomain is derived from cellular microstructure by homogenisation — spatial
averaging over many cells. Neu & Krassowska (1993, PMID 8243090) are explicit about
the limitations of this procedure. From their abstract:

> "The validity of the homogenized syncytium model is assured deep in the tissue
> for autonomous processes, such as propagation, and in the presence of external
> fields that are nearly uniform and limited in strength. **The derived model is
> not formally valid on the surface of tissue**, in the proximity of sources, and
> under strong or rapidly changing electrical fields."

The Kleber boundary speedup happens AT THE TISSUE SURFACE — precisely where the
bidomain's own mathematical derivation says it is not valid.

**Why it's invalid at the surface:** Homogenisation requires a periodic
representative volume element (RVE) that repeats throughout the domain. At the
tissue boundary, there is no cell on one side. The RVE is cut in half. The
periodicity assumption fails. The scale separation assumption (field varies slowly
over one RVE) also fails because phi_e transitions from its bulk value to phi_bath
over a distance comparable to one or two cells.

**The consequence:** The bidomain's sigma_i, sigma_e, and chi are bulk-averaged
quantities. At the tissue surface, these quantities are not physically meaningful
in the same way. The "intracellular conductivity" of the last half-cell at the
boundary is not the same as sigma_i averaged over many complete cells in the
interior. The boundary conditions we impose on the bidomain at the tissue surface
are BCs on a model that is not formally valid there.

**How serious is this?** The homogenisation error is O(epsilon) where epsilon =
cell_size / domain_size. For typical cells (~100 um) and domains (~1 cm),
epsilon ~ 0.01. The boundary layer is ~lambda ~ 1.4 mm ~ 14 cells wide. The
first 1-2 cells at the boundary have O(1) homogenisation error (the RVE is
completely disrupted), but the bulk of the boundary layer (cells 3-14) has the
normal O(epsilon) error. So the overall effect is probably captured to ~10-20%
accuracy, but the fine structure near the actual boundary is unreliable.

---

### Argument 2: Dirichlet phi_e = 0 Is an Approximation, Not the Exact Physics

The actual physics at the tissue-bath interface involves THREE boundary conditions
for TWO unknowns:

```
1. n . sigma_i . grad(phi_i) = 0           (intracellular no-flux)
2. phi_e|_tissue = phi_bath|_interface      (potential continuity)
3. n . sigma_e . grad(phi_e) = n . sigma_bath . grad(phi_bath)   (current continuity)
```

Plus the bath is governed by Laplace's equation:
```
4. div(sigma_bath . grad(phi_bath)) = 0     in the bath
5. phi_bath → 0                              far from tissue
```

The Dirichlet approximation phi_e = 0 collapses conditions 2-5 into a single
condition by ASSUMING phi_bath ≈ 0 at the interface. Patel & Roth (2005,
[DOI](https://doi.org/10.1103/PhysRevE.72.051931)) showed this is the
leading-order outer solution of a matched-asymptotic expansion. The inner solution
— the correction near the actual interface — has structure on the scale of the
cell radius and is discarded.

**When does the approximation break down?**

The key ratio is sigma_bath / sigma_e. For Tyrode's and cardiac tissue:
```
sigma_bath ~ 15-20 mS/cm
sigma_e_longitudinal ~ 6.25 mS/cm
sigma_e_transverse ~ 2.36 mS/cm
```

The ratio is only 2.5-8x, NOT infinity. The bath is more conductive but not
overwhelmingly so. For the approximation to be exact, we would need
sigma_bath/sigma_e → ∞.

**For confined geometries, it's worse.** Inside a narrow laser-cut channel
(width w), the bath is geometrically confined. The return current from the tissue
enters a thin strip of bath, and the potential inside this strip is NOT zero — it
rises according to:
```
phi_bath_channel ~ I_return * w / (sigma_bath * cross_section)
```

For a narrow isthmus (w ~ 0.5 mm, cross_section ~ 0.1 mm^2), phi_bath can be a
significant fraction of phi_e. The Dirichlet approximation overestimates the bath
shorting effect, and the Kleber speedup inside narrow channels is LESS than the
theory predicts.

This matters for ML-DO because the geometries most likely to support reentry
(narrow isthmuses, tight channels around the scar) are exactly the ones where the
Dirichlet approximation is worst.

---

### Argument 3: The Boundary Layer Thickness Is Dynamic, Not Static

The spatially varying D profile uses a static length scale:
```
lambda = sqrt(D_eff / G_m_rest)
```

where G_m_rest is the resting membrane conductance (~0.05 mS/cm^2). This gives
lambda ~ 1.4 mm.

But G_m is NOT constant during the action potential:

| Phase | G_m (mS/cm^2) | lambda (mm) | Boundary layer |
|-------|---------------|-------------|----------------|
| Resting | ~0.05 | ~1.4 | Wide |
| Na+ peak (upstroke) | ~100 | ~0.03 | Extremely thin |
| Plateau | ~0.5 | ~0.44 | Moderate |
| Repolarisation | ~5 | ~0.14 | Thin |

During the upstroke (when CV is determined), the membrane conductance increases
~2000-fold. The effective boundary layer COLLAPSES to ~30 um — less than one cell
width. At this instant, the Kleber effect is confined to the very last cell at
the boundary.

**The implication:** The static exponential profile D(x,y) = D_eff + (D_i - D_eff) *
exp(-d/lambda) with lambda = 1.4 mm dramatically overestimates the spatial extent
of the speedup during the critical moment (the upstroke) when CV is actually
determined. The true Kleber speedup may be much more localised — confined to the
last 1-2 cells — and its magnitude may differ from the static prediction.

The bidomain solver handles this correctly (the elliptic equation is solved at
every timestep with the current membrane conductance distribution), but the
analytical prediction CV_ratio = sqrt((sigma_i+sigma_e)/sigma_e) = 1.131 is
derived assuming the RESTING state boundary layer. The true ratio during the
dynamic upstroke may be different.

---

### Argument 4: The Last Cell Is Geometrically Asymmetric

In the continuum bidomain, every point has membrane distributed uniformly
(chi = 1400 cm^-1, the surface-to-volume ratio). At the tissue boundary, the
last physical cell is different:

```
Interior cell:
  ┌──────────────────┐
  │   cytoplasm       │  Membrane on ALL sides
  │   (gap junctions  │  Gap junctions at BOTH ends
  │    at both ends)  │
  └──────────────────┘

Boundary cell:
  ┌──────────────────┐
  │   cytoplasm       │  Membrane on top, bottom, right
  │   (gap junctions  │  Gap junction on LEFT end only
  │    on left only)  │  RIGHT end: membrane directly exposed to bath
  └──────────────────┘
```

The boundary cell has:
1. **Gap junctions on only one side** → reduced intracellular coupling in the
   direction toward the void. The effective sigma_i at this cell is anisotropic
   and asymmetric in a way the bidomain doesn't represent.

2. **Membrane directly exposed to bath** → the membrane on the void-facing end
   sees bath potential directly, not through the interstitial cleft. The
   transmembrane potential on this face is V = phi_i - phi_bath ≈ phi_i (since
   phi_bath ≈ 0). This membrane patch is capacitively coupled directly to the
   bath, acting as a current source/sink that the bidomain doesn't resolve.

3. **Asymmetric extracellular geometry** → the extracellular cleft on the
   void-facing side opens into the bath. The cleft width effectively becomes
   infinite. The extracellular potential drops to phi_bath over a distance of
   one cell, not smoothly over lambda.

The bidomain averages over all of this. It sees a smooth sigma_i, sigma_e, chi
at the boundary with smooth BCs. The real cell-level physics has discrete
transitions and geometric asymmetries that the bidomain cannot represent.

---

### Argument 5: The "Correct" Bidomain Result Has Never Been Experimentally Validated

The Kleber speedup of ~13% is a THEORETICAL PREDICTION. From our own
Experimental_Validation.md:

> "**The direct experiment has not been done.** No published study measures
> spatially-resolved conduction velocity along the edge of a clean insulating
> obstacle in cardiac tissue and compares it to CV in the interior."

The theory predicts 13%. But the theory:
- Uses bulk-averaged conductivities (which break down at boundaries, Argument 1)
- Uses the Dirichlet approximation (which is not exact, Argument 2)
- Assumes a static boundary layer (which is dynamic, Argument 3)
- Ignores discrete cell geometry (Argument 4)

The actual speedup could be 5%, 13%, 20%, or even negative (if source-sink effects
from the exposed membrane dominate). We do not know. We have never measured it.

Our Phase 6 validation shows the bidomain FDM CONVERGES TO ITS OWN PREDICTION
(ratio → 1.131). This proves the numerical implementation is correct. It does NOT
prove the prediction matches physical reality. A perfect implementation of an
approximate model gives exact solutions to the wrong equations.

---

### Argument 6: The 2D Bidomain Misrepresents Monolayer Geometry

A hiPSC monolayer is NOT a 2D sheet. It is 3-5 cell layers thick (~50-100 um),
submerged in a bath that covers BOTH the top and bottom surfaces.

The 2D bidomain treats the tissue as:
```
    bath (phi_e = 0)
    ─────────────────    ← 1D boundary (Dirichlet)
    tissue (2D bidomain)
    ─────────────────    ← 1D boundary (could be Neumann or Dirichlet)
```

The real 3D geometry is:
```
    bath (3D Laplace)
    ═════════════════    ← top surface (2D, bath-coupled)
    tissue (3-5 layers)
    ═════════════════    ← bottom surface (substrate + bath seeping underneath)
    substrate / bath
```

The bath couples to the tissue through BOTH surfaces over the ENTIRE tissue area,
not just at the edges. The "interior" of the 2D bidomain (where we assumed
sigma_eff applies) actually has bath coupling from above and below. The effective
D in the "interior" is already elevated above D_eff — it's not the same as
D_eff for a thick slab.

This means the RATIO of boundary D to interior D is SMALLER than the thick-slab
prediction because the interior D is already partially enhanced. The Kleber
speedup ratio would be reduced.

For a true single-cell-thick monolayer, the 3D bath coupling is even stronger:
every cell is directly bath-coupled from both surfaces. The distinction between
"boundary" and "interior" cells becomes smaller. In the extreme case of a
single-cell-wide strand (1D), every cell is bath-coupled and there is no "interior"
at all.

---

### Argument 7: The Bidomain Assumes Linear, Ohmic Conductors

The bidomain treats both intracellular and extracellular spaces as linear ohmic
conductors (J = sigma * E). This is accurate for the bulk electrolyte but less
so for:

1. **Gap junctions**: Gap junction conductance is voltage-dependent (transjunctional
   voltage-gating). At the tissue boundary, the last gap junction connection sees
   different loading than interior junctions (only one cell on one side). The
   voltage across this junction may enter the nonlinear regime, reducing its
   conductance. This would reduce sigma_i at the boundary, partially counteracting
   the Kleber enhancement.

2. **Narrow extracellular clefts**: In the tight spaces between cells (cleft
   width ~20 nm), the continuum Ohm's law approximation may break down. Debye
   layer effects, ion crowding, and fixed charges on cell surfaces can alter the
   effective conductivity. The EMI model by Roberts, Stinstra & Henriquez (2008,
   [DOI](https://doi.org/10.1529/biophysj.108.137349)) showed that when
   extracellular cleft conductivity is sufficiently reduced, the membrane adjacent
   to the tight space is "eliminated from participating in propagation" — a
   phenomenon the bidomain cannot capture.

---

### Summary of Arguments Against

| # | Argument | Effect on Kleber prediction | Severity |
|---|----------|---------------------------|----------|
| 1 | Homogenisation invalid at surface | Unknown — could increase or decrease | Fundamental |
| 2 | Dirichlet phi_e=0 is approximate | Overestimates speedup in confined voids | Moderate-High |
| 3 | Dynamic boundary layer | Static profile overestimates spatial extent during upstroke | Moderate |
| 4 | Discrete cell asymmetry at edge | Adds effects not in continuum model | Moderate |
| 5 | No experimental validation | Cannot verify prediction | Critical |
| 6 | 2D misrepresents 3D monolayer | Overestimates boundary-interior contrast | Moderate |
| 7 | Nonlinear conductors at boundary | May reduce effective sigma_i at edge | Low-Moderate |

The bidomain with Neumann+Dirichlet is the best CONTINUUM model, but it is still
an APPROXIMATE representation of the discrete, 3D, nonlinear, dynamic reality.
The approximation is strongest exactly where it matters most — at the tissue boundary.

---

## Part II: The EMI Model — Can It Capture the Physics More Correctly?

### What Is the EMI Model?

The EMI (Extracellular-Membrane-Intracellular) model, developed by Tveito, Jæger
and colleagues at Simula Research Laboratory, explicitly resolves all three spaces
in cardiac tissue:

- **E (Extracellular)**: The interstitial fluid between cells, resolved as a
  continuous geometric domain Omega_e with its own mesh
- **M (Membrane)**: The cell membrane, represented as explicit boundaries Gamma_k
  between each cell k and the extracellular space
- **I (Intracellular)**: The cytoplasm of each individual cell, resolved as
  separate geometric domains Omega_i_k

> Based on articles retrieved from PubMed: Jæger et al. (2021, Front Physiol
> 12:763584, [DOI](https://doi.org/10.3389/fphys.2021.763584)) introduced the EMI
> framework. Jæger & Tveito (2022, Front Physiol 12:811029,
> [DOI](https://doi.org/10.3389/fphys.2021.811029)) formally derived the bidomain
> as a homogenisation of the EMI, quantifying the approximation error. Jæger et al.
> (2024, Sci Rep 14:16954, [DOI](https://doi.org/10.1038/s41598-024-67431-w))
> benchmarked EMI against bidomain and monodomain for computational cost vs
> physiological resolution.

### The EMI Equations

In each intracellular domain (cell k):
```
div(sigma_i . grad(phi_i_k)) = 0     in Omega_i_k     ... (EMI-1)
```

In the extracellular domain:
```
div(sigma_e . grad(phi_e)) = 0       in Omega_e        ... (EMI-2)
```

On the membrane of cell k:
```
n . sigma_i . grad(phi_i_k) = Cm * dV_k/dt + I_ion(V_k, gates_k)    on Gamma_k   ... (EMI-3)
n . sigma_e . grad(phi_e) = -(Cm * dV_k/dt + I_ion(V_k, gates_k))   on Gamma_k   ... (EMI-4)
```

where V_k = phi_i_k - phi_e on the membrane of cell k.

Between adjacent cells j and k (gap junction):
```
I_gap = g_gap * (phi_i_j - phi_i_k)    ... (EMI-5)
```

**Key structural differences from the bidomain:**

1. **No homogenisation.** Each cell is resolved. There is no sigma_i_bulk or
   chi parameter. The intracellular conductivity is the ACTUAL cytoplasmic
   conductivity (~6-10 mS/cm), not the homogenised tissue conductivity
   (~1.74 mS/cm). The homogenised value is lower because gap junctions
   add resistance.

2. **No chi.** The surface-to-volume ratio chi = 1400 cm^-1 is a homogenisation
   artifact — it converts the membrane (a 2D surface) into a volumetric source
   term for the 3D continuum. In the EMI model, the membrane IS a surface. No
   conversion needed.

3. **Equations (EMI-1) and (EMI-2) are LAPLACE equations**, not reaction-diffusion.
   All the reaction (ion channels, capacitance) lives on the membrane boundary
   conditions (EMI-3, EMI-4). The bulk domains are passive conductors. This is
   physically correct — there are no current sources inside the cytoplasm or the
   interstitial fluid.

### How EMI Handles the Tissue-Void Boundary

At the edge of the tissue where cells terminate at a laser-cut void:

```
Cell interior boundary cell:
  ┌─────────────────────┐
  │ Omega_i_k            │  ← Laplace equation
  │                      │
  │ gap junction ←───────│──── gap junction to cell k-1
  │                      │
  └──────────┬───────────┘
             │ Gamma_k (membrane)
             │ V_k = phi_i_k - phi_e
             │ n.sigma_i.grad(phi_i_k) = Cm*dV/dt + I_ion
  ───────────┴──────────────
  Omega_e (extracellular)
  ──────────────────────────
  │          │              │
  │  cleft   │   OPENS TO   │
  │  between │   BATH       │  ← phi_e transitions to phi_bath
  │  cells   │              │
  ──────────────────────────
```

The void-facing end of the boundary cell has:

1. **No gap junction** on the void side → phi_i_k has a Neumann BC at this end:
   n . sigma_i . grad(phi_i_k) = membrane current (EMI-3). This is not an
   imposed BC — it EMERGES from the fact that there is no cell k+1 to connect to.

2. **The extracellular cleft opens into the bath** → Omega_e is geometrically
   continuous with the bath volume. No "Dirichlet phi_e = 0" needs to be imposed.
   The extracellular potential naturally transitions from its cleft value to the
   bath potential through the GEOMETRY of the domain. If the cleft opens into a
   wide bath, phi_e → 0 naturally. If the cleft opens into a narrow channel,
   phi_e remains elevated — and this is AUTOMATICALLY captured.

3. **The membrane on the void-facing end is EXPLICITLY RESOLVED** → The exposed
   membrane patch has its own Cm, I_ion, and sees bath potential on the outside.
   The current through this membrane is computed, not averaged.

**The crucial point:** In the EMI model, the Kleber-type boundary speedup (if it
exists) would EMERGE from the geometry and the physics without being imposed
through BCs. It would arise because:

- The extracellular cleft resistance near the tissue edge is reduced (geometry
  opens into bath)
- This allows more return current to flow through the low-resistance bath path
- The circuit resistance decreases → more current → faster depolarisation of
  downstream cells

The EMI model computes this from first principles. The bidomain ASSUMES it through
BCs.

---

### What EMI Would Resolve That the Bidomain Cannot

#### Resolution 1: Is the speedup 13%, or something else?

The bidomain predicts CV_ratio = sqrt((sigma_i + sigma_e)/sigma_e) = 1.131 using
HOMOGENISED conductivities. In the EMI model, the actual ratio would emerge from
the discrete cell geometry. Possible outcomes:

- **Ratio ≈ 1.13**: Bidomain prediction is accurate. Homogenisation error at
  the boundary is small. → Validates the bidomain approach.

- **Ratio < 1.13** (e.g., 1.05-1.10): The discrete geometry partially shields
  the interior cells from the bath. The exposed membrane on the boundary cell
  acts as a current sink (capacitive coupling to bath) that partially offsets
  the conductive enhancement. → Bidomain overestimates.

- **Ratio > 1.13** (e.g., 1.15-1.20): The discrete gap junction asymmetry
  creates additional current focusing that the smooth bidomain misses. → Bidomain
  underestimates.

- **Ratio ≈ 1.00**: The discrete effects (source-sink from exposed membrane,
  gap junction asymmetry, cleft geometry) largely cancel the conductive
  enhancement. → The Kleber speedup is a continuum artifact that doesn't
  survive discretisation.

Without running the EMI simulation, we genuinely do not know which of these
outcomes is correct. The bidomain predicts 1.13, but the bidomain is not valid
at the boundary (Argument 1).

#### Resolution 2: What is the TRUE boundary layer profile?

The bidomain predicts a smooth exponential:
```
D(d) = D_eff + (D_i - D_eff) * exp(-d / lambda)
```

The EMI model would show:
- Cell-by-cell staircase modulation (each cell is a discrete conductivity step)
- Possible non-monotonic behaviour (the exposed membrane on the boundary cell
  creates a local dip in effective D before the enhancement kicks in)
- Dynamic variation (the profile changes shape during the action potential as
  membrane conductance varies)

If the true profile is significantly different from the smooth exponential, the
monodomain D(x,y) approximation may be less accurate than assumed.

#### Resolution 3: Does gap junction voltage-gating matter at the boundary?

The boundary cell has gap junctions on only one side. During wave propagation,
the voltage difference across this junction differs from interior junctions
(because the boundary cell has different loading). If the voltage difference
enters the nonlinear regime of gap junction gating, the effective coupling
changes.

The EMI model with voltage-dependent g_gap (EMI-5 with g_gap = g_gap(V_j - V_k))
would reveal whether this is significant. The bidomain uses a constant sigma_i
and cannot capture this.

#### Resolution 4: Confined-void geometry effects

For a narrow laser-cut channel (width ~ 0.5 mm), the bidomain with Dirichlet
phi_e = 0 assumes the bath perfectly shorts the extracellular space. The EMI
model would naturally capture:
- Elevated phi_bath inside the narrow channel (finite bath conductivity)
- Reduced Kleber effect in narrow channels vs wide-open boundaries
- Geometry-dependent speedup that varies along the scar perimeter

This is DIRECTLY relevant to ML-DO: the scar geometries with narrow isthmuses
(reentry-supporting) are exactly where the Dirichlet approximation is worst.

#### Resolution 5: The exposed membrane as current pathway

The boundary cell has membrane patches directly facing the bath. During the
action potential:

- **Phase 0 (upstroke):** Large inward Na+ current. This current crosses the
  membrane into the extracellular space. On the bath-facing membrane, the
  extracellular side IS the bath. The transmembrane current flows directly into
  the bath, not through the interstitial cleft. This is an additional current
  pathway not present in interior cells.

- **Phase 2 (plateau):** Smaller L-type Ca2+ current flows into the bath through
  the exposed membrane.

- **Phase 3 (repolarisation):** K+ current flows out through the exposed membrane
  into the bath.

These currents through the exposed membrane are NOT the same as the averaged
chi * I_ion term in the bidomain. They represent a DIRECT coupling between the
intracellular space and the bath that bypasses the normal interstitial extracellular
pathway. The bidomain lumps this into the general phi_e = 0 BC and does not
resolve the difference between membrane facing another cell and membrane facing
the bath.

---

### What EMI Cannot Do (and Where Bidomain Remains Necessary)

#### Scale limitation

Based on PubMed: Jæger et al. (2024, [DOI](https://doi.org/10.1038/s41598-024-67431-w))
explicitly evaluated computational cost:

> "Cell-based models [...] offer both efficiency and precision for simulating
> small tissue samples (comprising thousands of cardiomyocytes). Conversely,
> the traditional bidomain model and its simplified counterpart, the monodomain
> model, are more appropriate for larger tissue masses (encompassing millions
> to billions of cardiomyocytes)."

For ML-DO, a single 2D tissue with a scar has ~10,000-100,000 cells. The ML-DO
loop requires ~1000-10,000 simulation evaluations. This is:

| Engine | Cost per evaluation | 10,000 evaluations | Feasible? |
|--------|--------------------|--------------------|-----------|
| Monodomain LBM | ~1 second | ~3 hours | Yes |
| Bidomain FDM | ~10 seconds | ~1 day | Marginal |
| EMI | ~1000 seconds | ~4 months | No |

EMI cannot serve as the ML-DO production engine.

#### Parameter uncertainty

The EMI model requires:
- Individual cell dimensions (length, width, height)
- Gap junction distribution (end-to-end vs lateral, number per cell)
- Gap junction conductance (single-channel and density)
- Cytoplasmic conductivity
- Extracellular cleft width
- Membrane properties per patch (Cm, I_ion may vary around the cell)

For hiPSC-CMs, most of these are poorly characterised. The cells are smaller,
rounder, and have less organised gap junction distribution than adult myocytes.
Using adult-myocyte parameters in an EMI model of hiPSC tissue would be
misleading.

#### The validation strategy, not the production engine

The correct role for EMI in the ML-DO project is as a VALIDATION TOOL:

```
Phase 1: Run EMI simulation of a small tissue strip (100-200 cells)
         with a clean tissue-void boundary, in a bath.
         Measure CV profile vs distance from boundary.

Phase 2: Run bidomain FDM on the same geometry and parameters.
         Compare CV profiles.

Phase 3: If they agree to within 5%:
           → Bidomain is validated. Use bidomain (or D(x,y) approximation)
             for ML-DO production runs.
         If they disagree:
           → Quantify the discrepancy. Determine whether it is:
             (a) A constant offset (all geometries affected equally → ranking
                 preserved, just rescale)
             (b) Geometry-dependent (narrow channels ≠ open edges → ranking
                 may change → need to correct)

Phase 4: For case (b), derive a CORRECTION to the bidomain prediction based
         on EMI calibration. Incorporate as a geometry-dependent modifier
         to D(x,y).
```

---

### Can We Build an EMI Model With Our Current Infrastructure?

#### What we have:
1. **Bidomain FDM Engine V1** — fully functional, validated. Provides the operator
   discretisation patterns (Laplace on irregular domains, membrane BCs, sparse
   solvers).

2. **LBM V1** — fully functional. Provides the streaming/collision framework
   for diffusion on Cartesian grids.

3. **TTP06 ionic model** — validated. Provides I_ion(V, gates) for membrane BCs.

4. **FDM stencils** — 5-point and 9-point, with harmonic mean at interfaces,
   face-based symmetric stencil for SPD operators.

#### What we would need:

1. **Cell geometry generator** — Create rectangular cells (or rounded rectangles
   for hiPSC-CMs) arranged in a brick-like pattern on a 2D grid. Each cell gets
   its own intracellular domain. The extracellular space is the complement.

2. **Mesh generation** — Triangular or Cartesian mesh that resolves both the
   cell interiors and the narrow extracellular clefts (~20 nm width, requiring
   ~5 nm mesh resolution). This is the expensive part — the mesh must resolve
   features spanning 4 orders of magnitude (nm clefts to mm tissue).

3. **Membrane interface handling** — At each cell boundary node, compute the
   transmembrane current from V_k = phi_i_k - phi_e and apply it as a flux BC
   on both the intracellular and extracellular Laplace equations.

4. **Gap junction coupling** — At cell-cell interfaces, add a conductance term
   g_gap * (phi_i_j - phi_i_k) to the boundary conditions.

5. **Bath domain** — Extend the extracellular mesh into the void/bath region.
   No membrane BCs in the bath, just Laplace with far-field phi → 0.

#### Implementation complexity:

The core physics (Laplace + membrane BCs) is simpler than the bidomain (no
reaction-diffusion in the bulk). The complexity is in the GEOMETRY:
- Generating and meshing thousands of cells with nm-scale clefts
- Handling the multi-domain problem (thousands of independent intracellular
  domains, one global extracellular domain)
- Efficiently solving the coupled system (block structure: each cell's
  intracellular solve is independent given phi_e; phi_e depends on all cells)

**Estimated effort:** 4-6 weeks for a 2D EMI prototype capable of simulating
100-200 cells with a tissue-void boundary. Not production-quality, but
sufficient for the Phase 1-2 validation above.

---

## Part III: Synthesis — What Should We Actually Do?

### The honest assessment:

The bidomain with Neumann+Dirichlet is the best available continuum model, but
it has real limitations at tissue boundaries (Arguments 1-7). The Kleber speedup
prediction of 13% is theoretically derived but experimentally unvalidated.

The EMI model can, in principle, resolve every limitation of the bidomain at
boundaries. But it is too expensive for ML-DO production runs and requires
parameters we don't have for hiPSC tissue.

### The pragmatic path:

1. **For ML-DO production:** Use monodomain LBM with spatially varying D(x,y)
   (the D(x,y) approximation). It captures the qualitative boundary effect, is
   fast enough for the optimisation loop, and the curvature/source-sink effects
   (which dominate geometry ranking) are exactly correct.

2. **For quantitative validation:** Build a small-scale EMI model (100-200 cells)
   and measure the actual boundary CV profile. Compare against bidomain. This
   answers the question "is 13% correct?" and determines whether a correction
   factor is needed.

3. **For the Oxford proposal:** The key message is:
   - D2Q9 Dirichlet on V is wrong (produces slowdown, not speedup) — PROVEN
   - The correct mechanism requires bidomain-level physics (two domains, asymmetric
     BCs) — ESTABLISHED
   - Monodomain with D(x,y) is a validated approximation — AVAILABLE
   - The exact magnitude of the boundary effect is uncertain because the bidomain
     prediction has not been experimentally verified and has known theoretical
     limitations at boundaries — HONEST
   - An EMI model could provide the missing ground truth — PROPOSED

### What does this mean for the research narrative?

The strongest position is not "we have the exact answer" but rather:

> "We identified a fundamental error in the proposed D2Q9 Dirichlet mechanism,
> we implemented the correct bidomain physics, we showed it converges to the
> theoretical prediction, AND we recognise the limitations of the continuum
> prediction. We propose EMI validation as a pathway to ground truth."

This is intellectually honest and positions the work at the frontier of the
field — nobody has measured the Kleber speedup at a clean inert boundary, in any
system, and nobody has compared the bidomain prediction against a cell-resolved
EMI model at a tissue-void interface.

---

## References

Based on articles retrieved from PubMed:

1. Neu JC, Krassowska W (1993). Homogenization of syncytial tissues. Crit Rev Biomed Eng 21(2):137-199. [PMID 8243090](https://pubmed.ncbi.nlm.nih.gov/8243090/)

2. Patel SG, Roth BJ (2005). Approximate solution to the bidomain equations for electrocardiogram problems. Phys Rev E 72:051931. [DOI](https://doi.org/10.1103/PhysRevE.72.051931)

3. Roberts SF, Stinstra JG, Henriquez CS (2008). Effect of nonuniform interstitial space properties on impulse propagation: a discrete multidomain model. Biophys J 95(8):3724-3737. [DOI](https://doi.org/10.1529/biophysj.108.137349)

4. Jæger KH, Edwards AG, Giles WR, Tveito A (2021). From Millimeters to Micrometers; Re-introducing Myocytes in Models of Cardiac Electrophysiology. Front Physiol 12:763584. [DOI](https://doi.org/10.3389/fphys.2021.763584)

5. Jæger KH, Tveito A (2022). Deriving the Bidomain Model of Cardiac Electrophysiology From a Cell-Based Model; Properties and Comparisons. Front Physiol 12:811029. [DOI](https://doi.org/10.3389/fphys.2021.811029)

6. Jæger KH, Trotter JD, Cai X, Arevalo H, Tveito A (2024). Evaluating computational efforts and physiological resolution of mathematical models of cardiac tissue. Sci Rep 14:16954. [DOI](https://doi.org/10.1038/s41598-024-67431-w)

7. Roth BJ (1991). A comparison of two boundary conditions used with the bidomain model of cardiac tissue. Ann Biomed Eng 19(6):669-678. [DOI](https://doi.org/10.1007/BF02368075)

8. Roth BJ (1996). Effect of a perfusing bath on the rate of rise of an action potential propagating through a slab of cardiac tissue. Ann Biomed Eng 24(6):639-646. [DOI](https://doi.org/10.1007/BF02684177)

9. Kleber AG, Rudy Y (2004). Basic mechanisms of cardiac impulse propagation. Physiol Rev 84(2):431-488. [DOI](https://doi.org/10.1152/physrev.00025.2003)
