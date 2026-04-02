# Literature Review: CV Effects at Clean Inert Boundaries

> Scope: conduction velocity changes at sharp, clean tissue-void interfaces with
> uniform healthy tissue on one side and electrically inert obstacle on the other.
> No border zone heterogeneity, no fibrosis remodeling, no ion channel changes.
> All citations retrieved from PubMed.

---

## Category 1: Eikonal-Curvature Theory

### 1. Keener (1991) — Eikonal-curvature equation for AP propagation in myocardium
[DOI](https://doi.org/10.1007/BF00163916) | J Math Biol 29(7):629-651

**Core finding:** Derives v(kappa) = v_0 - D*kappa from matched asymptotics of
the reaction-diffusion PDE. Wavefront curvature directly modulates local CV:
convex fronts (kappa > 0) slow, concave fronts (kappa < 0) speed up.

**Connection:** The mathematical foundation for all CV changes at inert obstacle
corners. Along flat inert edges, kappa = 0, so the eikonal equation predicts
NO CV change. Speedup occurs ONLY where the wavefront wraps around corners
(developing negative curvature). This is captured by any solver (FDM, FEM, LBM)
that correctly discretises the Laplacian.

### 2. Fast & Kleber (1997) — Role of wavefront curvature in propagation of cardiac impulse
[DOI](https://doi.org/10.1016/s0008-6363(96)00216-7) | Cardiovasc Res 33(2):258-271

**Core finding:** Review establishing that curvature effects matter in four
cardiac settings: point stimulation, narrow-to-wide transitions, sharp obstacle
edges, and spiral wave cores. Estimates critical curvature for block at radii
~0.1-0.5 mm, setting a lower bound on reentrant circuit size.

**Connection:** For inert scar geometry optimisation, the scar shapes that produce
the tightest concave curvature at corners (sharp notches, narrow isthmuses) will
produce the largest local CV changes. This geometric effect is well-captured by
monodomain LBM. The open question is what happens along flat inert edges where
kappa = 0 — eikonal theory says nothing happens, but the bidomain says otherwise.

---

## Category 2: Experimental Validation of Curvature Effects at Inert Obstacles

### 3. Cabo et al (1994) — Wavefront curvature as cause of slow conduction and block
[DOI](https://doi.org/10.1161/01.res.75.6.1014) | Circ Res 75(6):1014-1028

**Core finding:** Planar waves diffracted through narrow isthmuses cut in sheep
epicardium and simulated in the Luo-Rudy model. After exiting the isthmus, the
wavefront becomes elliptical with pronounced curvature. CV decreases proportional
to curvature. Critical isthmus width for block: ~200 um longitudinal, ~600 um
transverse (model). Block threshold is rate-dependent.

**Connection:** Definitive experimental proof that v(kappa) operates in cardiac
tissue on the slowing side (convex fronts exiting isthmuses). The isthmus-width
vs block relationship is exactly the geometric parameter that ML-DO should
optimise. Importantly, the experiment uses clean-cut isthmuses — true inert
boundaries — validating the curvature mechanism in isolation from any border
zone effects.

### 4. Rohr & Salzberg (1994) — Impulse propagation across geometric expansions
[DOI](https://doi.org/10.1085/jgp.104.2.287) | J Gen Physiol 104(2):287-309

**Core finding:** First microscopic characterisation of propagation across abrupt
tissue expansions in patterned NRVM cultures. Rectangular (sharp) expansions cause
delays and AP upstroke distortions; funneled (gradual) expansions support smooth
propagation. At high rates, delays at abrupt expansions convert to intermittent
unidirectional block.

**Connection:** The geometry around an inert obstacle IS a series of expansions
(isthmus exit) and compressions (isthmus entrance). This paper shows the
expansion geometry determines whether conduction succeeds or blocks — with clean,
sharp edges being more prone to block than tapered ones. For ML-DO, this means
rounded vs angular scar corners have fundamentally different arrhythmogenic
potential, even though rounded corners produce less curvature speedup.

### 5. Entcheva et al (2000) — Reentry in monolayers around inert obstacles
[DOI](https://doi.org/10.1111/j.1540-8167.2000.tb00029.x) | J Cardiovasc Electrophysiol 11(6):665-676

**Core finding:** First sustained anatomic reentry in NRVM monolayers around a
small inert obstacle (6 x 0.5 mm). CV varies along the reentrant pathway due to
curvature effects. Reentry terminated by field stimulation activating tissue at
obstacle borders.

**Connection:** Direct proof-of-concept for the ML-DO experimental design. A
small inert obstacle in a monolayer IS sufficient to anchor reentry. The variable
CV along the circuit confirms curvature effects are physiologically significant.
No border zone — pure geometry driving reentry around a clean inert void.

### 6. Fast, Darrow, Saffitz & Kleber (1996) — Anisotropic activation in heart cell monolayers
[DOI](https://doi.org/10.1161/01.res.79.1.115) | Circ Res 79(1):115-127

**Core finding:** High-resolution optical mapping (7-15 um) in NRVM monolayers.
At structural discontinuities (intercellular clefts), Vmax INCREASES when the
wavefront faces cleft borders. Narrow isthmuses between clefts are susceptible
to conduction block.

**Connection:** Even without bath coupling, clean structural edges produce local
electrophysiological changes through source-sink redistribution. When a wavefront
approaches a void, the cells at the tissue edge lose lateral coupling partners
on the void side. This reduces their electrotonic load and increases Vmax. This
is a monodomain geometric source-sink effect distinct from the bidomain Kleber
effect but operating in the same direction (increased excitability at borders).

---

## Category 3: The Kleber Boundary Speedup (Bidomain at Tissue-Bath Interface)

### 7. Kleber & Rudy (2004) — Basic mechanisms of cardiac impulse propagation
[DOI](https://doi.org/10.1152/physrev.00025.2003) | Physiol Rev 84(2):431-488

**Core finding:** The definitive review on cardiac propagation. Section on
tissue-bath interface establishes: at a clean tissue-bath boundary, the
intracellular space terminates (Neumann BC) while the extracellular space is
continuous with the low-resistance bath (effectively Dirichlet, phi_e -> 0).
This BC asymmetry shorts the extracellular resistance near the boundary,
increasing effective conductivity from sigma_eff = sigma_i*sigma_e/(sigma_i+sigma_e)
to sigma_boundary = sigma_i. Predicted CV ratio = sqrt((sigma_i+sigma_e)/sigma_e)
~ 1.13 for human ventricular tissue.

**Connection:** This is the central theoretical prediction for the ML-DO problem.
At any clean tissue-void boundary in a bath-perfused monolayer, the Kleber
speedup should produce ~13% CV increase along FLAT edges (kappa = 0). This is
invisible to monodomain simulations with any uniform BC. Standard FDM/FEM with
no-flux treats the boundary as invisible to planar waves. The Oxford proposal
correctly identified this gap but proposed the wrong fix (D2Q9 Dirichlet on V
instead of bidomain or spatially varying D).

### 8. Roth (1991) — Comparison of two bidomain boundary conditions
[DOI](https://doi.org/10.1007/BF02368075) | Ann Biomed Eng 19(6):669-678

**Core finding:** Two alternative BC formulations for the bidomain at the tissue-bath
interface are mathematically equivalent when the transverse space constant >> cell
radius. Both produce the same Kleber-type boundary effect.

**Connection:** Confirms that the Kleber speedup is robust to the specific BC
formulation used in the bidomain model. Any correct bidomain implementation (FDM
or LBM, either BC formulation) will capture it. The issue is structural: monodomain
has only one domain and therefore cannot represent the asymmetric BCs at all.

### 9. Roth (1996) — Bath effect on AP rate of rise in a bidomain slab
[DOI](https://doi.org/10.1007/BF02684177) | Ann Biomed Eng 24(6):639-646

**Core finding:** Bidomain simulation of a tissue slab perfused by a bath. The
bath REDUCES dV/dtmax at the tissue surface, and the reduction depends on
propagation direction. Analytical solution derived assuming intracellular potential
is depth-independent.

**Connection:** Shows that the perfusing bath affects AP morphology (not just CV)
at the tissue surface. In a bath-perfused hiPSC monolayer, optical mapping at
inert scar borders will see both the Kleber CV speedup AND altered AP upstroke
shape. The simulation engine must account for both to match experimental recordings.

---

## Category 4: Source-Sink Balance at Inert Boundaries

### 10. Shaw & Rudy (1997) — Ionic mechanisms of propagation
[DOI](https://doi.org/10.1161/01.res.81.5.727) | Circ Res 81(5):727-741

**Core finding:** Defines the safety factor (SF) for conduction. At geometric
expansions (source-sink mismatch), SF determines whether the wavefront survives.
Reduced excitability lowers SF (block at ~17 cm/s); reduced coupling paradoxically
raises SF (conduction sustained to ~0.26 cm/s).

**Connection:** Cited in the Oxford proposal. At inert scar corners and isthmuses,
the source-sink balance determines block vs propagation. The SF framework predicts
that in well-coupled, fully excitable tissue (the ML-DO scenario with clean inert
boundaries and healthy tissue), block occurs at relatively high CV (~17 cm/s). This
sets the critical isthmus width. The ML-DO objective function is essentially
searching for geometries that push the local SF below 1 during S2 delivery.

### 11. Rohr, Kucera, Fast & Kleber (1997) — Paradoxical improvement by partial uncoupling
[DOI](https://doi.org/10.1126/science.275.5301.841) | Science 275(5301):841-844

**Core finding:** Spatially uniform reduction of gap junction coupling can
paradoxically IMPROVE conduction across discontinuous structures that exhibit
unidirectional block. Mechanism: uncoupling reduces current drain into the expansion
(reduces sink), allowing the reduced source to overcome the load.

**Connection:** Though this paper studies coupling changes, the core insight is
about source-sink balance at geometric discontinuities — which is exactly what
happens at inert scar borders. At an abrupt expansion past a scar corner, the
wavefront faces a source-sink mismatch. The paper shows that this mismatch is
exquisitely sensitive to the geometry of the transition. For ML-DO, scar shapes
that create specific expansion ratios will have sharply nonlinear effects on
whether propagation succeeds, fails, or becomes unidirectional.

---

## Category 5: Bidomain vs Monodomain — What the Monodomain Misses at Clean Boundaries

### 12. Pollard, Hooke & Henriquez (1992) — Cardiac propagation simulation
PMID: 1478091 | Crit Rev Biomed Eng 20(3-4):171-210

**Core finding:** Comprehensive simulation review comparing 2D bidomain and 3D
monodomain. The bidomain explicitly characterises intracellular, interstitial,
and extracellular volumes. Each volume conductor has unique identifiable
consequences on activation patterns.

**Connection:** Establishes the structural reason why monodomain cannot capture
the Kleber speedup: it lumps three conductors into one, destroying the ability
to impose asymmetric BCs (intracellular Neumann + extracellular Dirichlet) at a
clean tissue-bath interface. This is not a discretisation error — it is a
fundamental model limitation.

### 13. Hubbard & Henriquez (2010) — Interstitial loading effects
[DOI](https://doi.org/10.1152/ajpheart.00689.2009) | Am J Physiol Heart Circ Physiol 298(4):H1209-1218

**Core finding:** Increasing effective interstitial resistivity makes propagation
more continuous at the microscale. Uses a monodomain correction derived from
bidomain simulations. Shows that interstitial space properties modulate propagation
independently of intracellular structure.

**Connection:** Demonstrates that even in a monodomain framework, interstitial
effects CAN be approximated via modified effective parameters. This supports the
spatially-varying-D approach for capturing the Kleber effect in monodomain LBM:
near the inert boundary, increase D from D_eff to D_i over the electrotonic space
constant lambda. The paper validates that such corrections produce physically
meaningful results.

---

## Category 6: LBM as Simulation Engine

### 14. Rapaka et al (2012) — LBM-EP: Lattice-Boltzmann for fast cardiac EP
[DOI](https://doi.org/10.1007/978-3-642-33418-4_5) | MICCAI 15(Pt2):33-40

**Core finding:** First LBM cardiac EP solver. Monodomain on Cartesian grid with
level-set geometry. 10-45x faster than FEM with comparable accuracy. Demonstrated
on a patient with myocardial scar.

**Connection:** Establishes LBM as viable for cardiac EP. The speed advantage
(10-45x over FEM) is critical for ML-DO, which requires thousands of simulation
evaluations. The level-set scar representation naturally handles arbitrary inert
boundary shapes. However, this is monodomain — the Kleber boundary speedup at
the level-set interface is not captured.

### 15. Zettinig et al (2014) — Data-driven LBM diffusivity estimation from ECG
[DOI](https://doi.org/10.1016/j.media.2014.04.011) | Med Image Anal 18(8):1361-1376

**Core finding:** Polynomial regression predicts LBM diffusivity from QRS duration
and electrical axis. 9500 simulations (~3s each) on 19 patients. Quantifies
inherent uncertainty in diffusion parameters for given ECG features.

**Connection:** Demonstrates LBM at the scale needed for ML-DO (thousands of
evaluations). The uncertainty quantification approach could be adapted: for each
scar geometry, quantify how sensitive the arrhythmogenicity score is to diffusion
parameter uncertainty. If the Kleber boundary speedup (~13%) is smaller than the
parameter uncertainty, it may not matter for ranking.

### 16. Villar-Valero et al (2025) — LBM for doxorubicin cardiotoxicity simulation
[DOI](https://doi.org/10.1113/JP288819) | J Physiol (in press)

**Core finding:** GPU-optimised LBM for 3D LV models with diffuse fibrosis. Virtual
programmed stimulation (96 simulations). Reentry required >= 70% diffusion reduction
in fibrotic zones to induce arrhythmia despite unchanged excitability.

**Connection:** Most recent and directly comparable to the ML-DO setup. Uses GPU-LBM
with programmed stimulation protocol around geometric scar/fibrosis. Their 70%
diffusion reduction threshold for reentry provides a reference point. Notably, this
paper uses diffusion REDUCTION (heterogeneity-induced slowing), NOT boundary
speedup, as the arrhythmogenic mechanism — consistent with the real-infarct
literature but different from the clean-inert-boundary question.

---

## Category 7: Reentry Around Inert Obstacles — Geometry Determines Circuit

### 17. Gonzales et al (2014) — Structural contributions to fibrillatory rotors
[DOI](https://doi.org/10.1093/europace/euu251) | Europace 16(S4):iv3-iv10

**Core finding:** In a 3D biatrial model, inexcitable (inert) regions near the
rotor tip ANCHOR the rotor, converting fibrillation to macro-reentry. Increasing
inert-obstacle size decreases rotation frequency and widens the excitable gap.

**Connection:** Directly relevant to ML-DO. Inert obstacle geometry (size, shape,
position) determines rotor anchoring behaviour. The finding that increasing obstacle
size converts fibrillation to macro-reentry (potentially more treatable) suggests
a non-monotonic relationship between scar size and arrhythmogenicity. ML-DO should
explore this design space — there may be a "worst-case" scar size that is neither
too small (no anchoring) nor too large (stable macro-reentry with wide excitable gap).

---

## Summary Table — Inert Boundary Focus

| # | Paper | Year | Finding Relevant to Inert Boundaries | Captured by Mono LBM? |
|---|-------|------|--------------------------------------|----------------------|
| 1 | Keener | 1991 | v(kappa) = v_0 - D*kappa at corners | YES |
| 2 | Fast & Kleber | 1997 | Critical curvature sets minimum reentry size | YES |
| 3 | Cabo et al | 1994 | Isthmus width determines block (clean cuts) | YES |
| 4 | Rohr & Salzberg | 1994 | Sharp vs funneled expansions: block threshold | YES |
| 5 | Entcheva et al | 2000 | Sustained reentry around inert obstacle in monolayer | YES |
| 6 | Fast et al | 1996 | Vmax increase at structural edges (source-sink) | YES |
| 7 | Kleber & Rudy | 2004 | Kleber speedup ~13% at tissue-bath interface | NO — needs bidomain |
| 8 | Roth | 1991 | Bidomain BCs produce boundary speedup | NO — needs bidomain |
| 9 | Roth | 1996 | Bath alters AP morphology at surface | NO — needs bidomain |
| 10 | Shaw & Rudy | 1997 | Safety factor at geometric expansions | YES |
| 11 | Rohr et al | 1997 | Source-sink balance at discontinuities | YES |
| 12 | Pollard et al | 1992 | Monodomain structurally cannot do asymmetric BCs | Explains limitation |
| 13 | Hubbard et al | 2010 | Monodomain correction from bidomain is valid | Supports D(x,y) fix |
| 14 | Rapaka et al | 2012 | LBM is 10-45x faster than FEM | Engine choice |
| 15 | Zettinig et al | 2014 | LBM scales to thousands of evaluations | ML-DO feasibility |
| 16 | Villar-Valero | 2025 | GPU-LBM with programmed stimulation works | Engine validation |
| 17 | Gonzales et al | 2014 | Inert obstacle size controls rotor anchoring | YES |

---

## Key Findings for ML-DO

### What monodomain LBM DOES capture at inert boundaries:
1. **Curvature-velocity relation** at corners: v(kappa) = v_0 - D*kappa
2. **Source-sink mismatch** at isthmuses and expansions: block thresholds
3. **Rate-dependent block** through narrow channels
4. **Rotor anchoring** to inert obstacles
5. **Reentry inducibility** via S1-S2 protocol

These are the dominant geometric determinants of arrhythmogenicity. They are
correctly captured by any monodomain solver including D2Q5 or D2Q9 LBM with
bounce-back (Neumann) BC at inert borders.

### What monodomain LBM MISSES at inert boundaries:
1. **Kleber boundary speedup** (~13% CV increase along flat edges in bath-perfused
   tissue). Requires bidomain or spatially varying D approximation.

This is a single, well-characterised effect with a known analytical form:
```
D(x,y) = D_eff + (D_i - D_eff) * exp(-d / lambda)
lambda = sqrt(D_eff / G_m_rest) ~ 1.4 mm
```

### What the Oxford proposal's D2Q9 Dirichlet approach INCORRECTLY produces:
- Dirichlet V = V_rest at scar boundary creates a current SINK (drains current
  from depolarised cells), producing CV SLOWDOWN — opposite to Kleber.
- D2Q9 diagonal bounce-back at Neumann boundaries produces an O(dx^2) slowdown
  artifact (~3%) — also opposite to Kleber and vanishes with refinement.

### The fix is straightforward:
- Use **Neumann BC** (bounce-back) at inert boundaries — NOT Dirichlet
- To capture the Kleber speedup, either:
  (a) implement spatially varying D(x,y) near scar borders (monodomain approx), or
  (b) use a bidomain LBM (two lattices, no global solve needed)
- Option (a) adds negligible cost; option (b) adds ~2x memory and compute

### Impact on ML-DO geometry ranking:
The Kleber effect is a constant ~13% speedup along ALL flat scar edges regardless
of shape. It modifies the effective wavelength (CV x APD) near boundaries uniformly.
This likely shifts the absolute critical scar size but preserves the RELATIVE
ranking of geometries (since all shapes are affected similarly). The curvature and
source-sink effects, which ARE geometry-dependent and ARE captured by monodomain
LBM, likely dominate the ranking.
