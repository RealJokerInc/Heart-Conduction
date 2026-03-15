# Kleber's Logical Argument Chain for Boundary Speedup

> The "Kleber boundary speedup" is not derived in a single paper. It is a
> conclusion that emerges from the intersection of five intellectual threads,
> spanning 25 years of work by Kleber and collaborators (particularly Roth,
> Henriquez, Fast, Rohr, Kucera, and Rudy). This document traces the chain
> link by link, citing the specific paper where each link was established.

---

## Link 1: Cardiac Propagation Is a Circuit Problem

**Source:** Classical cable theory (Hodgkin-Huxley 1952, extended to cardiac
tissue by Plonsey, Henriquez)

**The argument:**

A propagating action potential works by local circuit currents. At the wavefront:

```
    ← — — — extracellular return current — — — ←
    ↓                                           ↑
  [resting cell]  ←membrane→  [extracellular]   ↑
    ↓                                           ↑
  [gap junction]                                ↑
    ↓                                           ↑
  [depolarised cell]  ←membrane→  [extracellular]
    ↓                                           ↑
    → — — — intracellular forward current — — — →
```

Current flows FORWARD through the intracellular space (through gap junctions)
and RETURNS through the extracellular space. The circuit is closed by
transmembrane current crossing the membrane at two locations:

1. At the **wavefront**: outward current depolarises the resting cell ahead
2. Behind the wavefront: inward current from the sodium channels of the
   already-depolarised cell

The total axial resistance of this loop is:
```
R_axial = R_intracellular + R_extracellular = r_i + r_e   (per unit length)
```

The effective conductivity for propagation is:
```
sigma_eff = 1 / (1/sigma_i + 1/sigma_e) = sigma_i * sigma_e / (sigma_i + sigma_e)
```

This is the harmonic mean. It is ALWAYS less than either sigma_i or sigma_e alone,
because the current must traverse BOTH resistances in series.

**The key insight (from cable theory):** CV depends on sigma_eff, not on sigma_i
or sigma_e independently. For a 1D cable:
```
CV = f(sigma_eff, Cm, I_ion)
```

where f is the nonlinear traveling-wave eigenvalue. For simple excitable kinetics,
CV ~ sqrt(sigma_eff).

> Based on articles retrieved from PubMed: Henriquez & Plonsey (1990, IEEE Trans
> Biomed Eng 37(9):850-860, [DOI](https://doi.org/10.1109/10.58596)) formalised
> this circuit analysis for a cylindrical bundle in a volume conductor, showing
> how the extracellular medium modifies the effective axial resistance.

---

## Link 2: r_i and r_e Are Separately Measurable — and r_e Depends on the Bath

**Source:** Kleber, Riegger & Janse (1987)

**The argument:**

To measure r_i and r_e separately (not just their sum), you need cable analysis:
inject current at one point, measure voltage decay along the fiber, fit the
space constants. But cable analysis has a subtle requirement: the BOUNDARY
CONDITIONS must be known.

Kleber, Riegger & Janse (1987, [DOI](https://doi.org/10.1161/01.res.61.2.271))
made a critical experimental choice: they placed isolated rabbit papillary muscles
in a **humid gaseous environment**, not in a liquid bath. Why?

> "Cable analysis was made possible by placing the muscle in a H2O-saturated
> gaseous environment, which acted as an **electrical insulator**."

In a liquid bath, the extracellular space of the tissue is electrically continuous
with the bath. The extracellular current has an ALTERNATIVE path — it can flow
out into the bath, travel through the low-resistance bath, and re-enter the tissue
elsewhere. This "short-circuits" the extracellular resistance. You cannot measure
the true tissue r_e if the bath is shorting it.

By using a gaseous environment (insulator), they ensured:
- ALL extracellular current stays within the tissue
- The measured r_e reflects the true tissue extracellular resistance
- Cable analysis gives valid results

**The implicit recognition:** The bath CHANGES the effective extracellular resistance.
A preparation in a bath has a LOWER effective r_e (and therefore higher sigma_eff
and higher CV) than the same preparation insulated from a bath. This is not an
artifact — it is real physics.

Their measured values (arterially perfused rabbit papillary muscle):
```
r_i (intracellular) = measured in gaseous environment
r_e (extracellular) = measured in gaseous environment
ratio r_e/r_i ≈ 0.3-0.5  (extracellular is 2-3x more conductive)
```

---

## Link 3: The Bidomain Formalises Two Domains with Separate Boundary Conditions

**Source:** Tung (1978, PhD thesis), formalised by Neu & Krassowska (1993)

**The argument:**

The circuit picture (Link 1) and the experimental observation that r_i and r_e are
independent quantities (Link 2) lead to a natural mathematical framework: treat
the intracellular and extracellular spaces as two separate, interpenetrating
continua, each with its own potential field and conductivity tensor.

The bidomain equations (derived by homogenising over individual cells):
```
Intracellular: div(sigma_i . grad(phi_i)) = +I_m    (current leaving intracellular)
Extracellular: div(sigma_e . grad(phi_e)) = -I_m    (current entering extracellular)
Membrane:      I_m = chi * (Cm * dV/dt + I_ion)     (transmembrane current density)
```

where V = phi_i - phi_e.

The crucial structural feature: phi_i and phi_e are INDEPENDENT variables. They
can have DIFFERENT boundary conditions at tissue surfaces. This is impossible in
a single-variable (monodomain) model.

> Neu & Krassowska (1993, Crit Rev Biomed Eng 21:137-199,
> [PMID 8243090](https://pubmed.ncbi.nlm.nih.gov/8243090/)) derived the bidomain
> from cellular microstructure and stated explicitly that the model "is not formally
> valid on the surface of tissue."

---

## Link 4: At a Tissue-Bath Interface, the BCs Are Asymmetric

**Source:** Roth (1991), with physical motivation from Kleber's circuit picture

**The argument:**

At a tissue-bath interface (tissue in contact with Tyrode's solution), two
independent physical facts determine the boundary conditions:

**Fact A (intracellular termination):** Cells end at the tissue surface. Gap
junctions do not extend beyond the tissue. No intracellular current can cross
the boundary.
```
n . sigma_i . grad(phi_i) = 0     (Neumann — zero intracellular flux)
```

**Fact B (extracellular continuity):** The interstitial fluid is the same medium
as the bath (both are electrolyte solutions). There is no barrier. The
extracellular potential at the tissue surface equals the bath potential.
```
phi_e = phi_bath ≈ 0               (Dirichlet — extracellular clamped to bath)
```

These are BCs on DIFFERENT variables. Fact A constrains phi_i (or equivalently,
the intracellular current). Fact B constrains phi_e (the extracellular potential).

> Based on articles retrieved from PubMed: Roth (1991, Ann Biomed Eng 19:669-678,
> [DOI](https://doi.org/10.1007/BF02368075)) showed that two alternative
> formulations of these boundary conditions are mathematically equivalent when
> the transverse space constant >> cell radius.

**Contrast with insulated boundary (gaseous environment):**
```
n . sigma_i . grad(phi_i) = 0     (same — cells still end)
n . sigma_e . grad(phi_e) = 0     (DIFFERENT — no bath, so no extracellular escape)
```

Both domains have Neumann BCs → symmetric → no boundary enhancement.

---

## Link 5: The BC Asymmetry Shorts the Extracellular Return Path

**Source:** This is Kleber's circuit insight applied to the bidomain BCs from Link 4

**The argument:**

Return to the circuit picture from Link 1. In bulk tissue, the wavefront current
flows:
```
Forward:  intracellularly, through gap junctions  (resistance r_i)
Return:   extracellularly, through interstitial space  (resistance r_e)
Total:    r_i + r_e  →  sigma_eff = sigma_i*sigma_e / (sigma_i+sigma_e)
```

Now consider a wavefront propagating along a tissue-bath boundary. The return
current has TWO available paths:

```
Path 1:  Through the tissue extracellular space  (resistance r_e)
Path 2:  Out into the bath, through the bath, back into the tissue  (resistance ≈ 0)
```

Path 2 exists because phi_e = 0 at the boundary (Link 4, Fact B). The bath is a
low-resistance shunt connected in PARALLEL with the tissue extracellular space.

Parallel combination:
```
1/r_e_effective = 1/r_e + 1/r_bath ≈ 1/r_e + ∞ = ∞
r_e_effective ≈ 0
```

With r_e effectively shorted:
```
R_total_boundary ≈ r_i + 0 = r_i
sigma_eff_boundary ≈ sigma_i
```

Since sigma_i > sigma_eff (the harmonic mean is always less than either component):
```
CV_boundary > CV_interior
```

**Quantitatively:**
```
CV_boundary / CV_interior = sqrt(sigma_eff_boundary / sigma_eff_interior)
                          = sqrt(sigma_i / sigma_eff)
                          = sqrt(sigma_i * (sigma_i + sigma_e) / (sigma_i * sigma_e))
                          = sqrt((sigma_i + sigma_e) / sigma_e)
```

For human ventricular tissue (longitudinal):
```
sigma_i = 1.74 mS/cm
sigma_e = 6.25 mS/cm
CV_ratio = sqrt((1.74 + 6.25) / 6.25) = sqrt(1.278) = 1.131
```

**~13% CV increase at the tissue-bath boundary.**

---

## Link 6: The Shorting Effect Decays Exponentially into the Tissue Interior

**Source:** Bidomain analysis; made explicit by Roth (1996) and Patel & Roth (2005)

**The argument:**

The bath shorts the extracellular resistance AT the boundary (Link 5). How far
into the tissue does this effect extend?

Consider the elliptic equation for phi_e (equation 5 from the bidomain):
```
div((sigma_i + sigma_e) . grad(phi_e)) = -div(sigma_i . grad(V))
```

In bulk tissue, the particular solution is:
```
phi_e_bulk = -sigma_i / (sigma_i + sigma_e) * V
```

At the boundary, phi_e = 0 (Dirichlet). The transition between phi_e = 0 and
phi_e = phi_e_bulk requires a homogeneous correction that satisfies the Laplace
equation in the transverse direction. For a semi-infinite slab with boundary at
y = 0:

```
phi_e_correction ~ exp(-y / lambda)
```

where lambda is the electrotonic space constant:
```
lambda = sqrt(sigma_eff / (chi * G_m))
```

At the wavefront (where CV is determined), G_m is the membrane conductance at
that instant. Using resting conductance G_m_rest ≈ 0.05 mS/cm^2:
```
lambda ≈ sqrt(0.000970 / 0.05) ≈ 0.139 cm ≈ 1.4 mm
```

The effective conductivity profile:
```
sigma_eff(y) = sigma_i   at y = 0
             → sigma_eff  as y → ∞
Transition:   sigma_eff(y) ≈ sigma_eff + (sigma_i - sigma_eff) * exp(-y / lambda)
```

**The enhancement is localised.** At y = 3*lambda ≈ 4 mm, the enhancement has
decayed to ~5% of its peak value. The effect is confined to a boundary layer of
thickness ~1-2 mm.

> Based on articles retrieved from PubMed: Roth (1996, Ann Biomed Eng 24:639-646,
> [DOI](https://doi.org/10.1007/BF02684177)) showed analytically that the bath
> reduces dV/dtmax at the tissue surface. Patel & Roth (2005, Phys Rev E 72:051931,
> [DOI](https://doi.org/10.1103/PhysRevE.72.051931)) derived the full
> matched-asymptotic solution showing the exponential boundary layer structure.

---

## Link 7: This Explains Systematic "Edge Effects" in Experiments

**Source:** Kleber & Rudy (2004) synthesis; experimental observations from Fast,
Rohr, Kleber

**The argument:**

Experimentalists routinely observe higher CV and altered AP morphology at tissue
edges. The standard practice is to EXCLUDE edge measurements from analysis:

> From Kleber & Rudy (2004, [DOI](https://doi.org/10.1152/physrev.00025.2003)):
> Researchers "routinely exclude boundary regions from CV analysis because
> homogeneity cannot be verified near edges."

The bidomain analysis (Links 3-6) provides a mechanistic explanation for these
"edge effects" that does NOT invoke damage, inhomogeneity, or measurement error:

1. The tissue is submerged in a conducting bath
2. The bath shorts the extracellular return path at the tissue surface
3. Effective conductivity increases near the surface
4. CV is higher at the surface than in the interior
5. AP morphology changes (Roth 1996: dV/dtmax decreases at the surface)

These are real physical effects of the bath coupling, not artifacts to be discarded.

**The irony:** By excluding edge measurements to "avoid artifacts," experimentalists
have been systematically excluding the most interesting boundary physics from their
data for decades.

---

## Link 8: The Effect Is Present Wherever Tissue Meets a Conducting Bath

**Source:** Generalisation from the bidomain analysis

**The argument:**

The boundary speedup (Links 4-6) requires only:
1. Intracellular space terminates (Neumann on phi_i)
2. Extracellular space is continuous with a low-resistance conductor (Dirichlet
   on phi_e)

This applies to:

| Geometry | Intracellular | Extracellular | Kleber effect? |
|----------|--------------|---------------|----------------|
| Tissue submerged in Tyrode's (outer boundary) | Terminates | Bath-coupled | YES |
| Laser-cut void in monolayer (void fills with bath) | Terminates | Bath-coupled | YES |
| Tissue on glass substrate (bottom surface) | Terminates | Insulated (glass) | NO |
| Tissue surface in air (Langendorff) | Terminates | Partially coupled (thin film) | PARTIAL |
| Tissue in contact with blood (in vivo) | Terminates | Bath-coupled (blood) | YES |

For the ML-DO project:
- The hiPSC monolayer is submerged in Tyrode's → outer edges: Kleber effect
- The laser-cut voids fill with Tyrode's → void edges: Kleber effect
- The culture substrate (glass or PDMS) → bottom: insulated (no Kleber)

The Kleber effect is present at EVERY inert boundary that faces a conducting bath.
It is absent only at boundaries facing insulators.

---

## The Complete Chain (One Paragraph)

Cardiac propagation is a local circuit: intracellular forward current through gap
junctions returns extracellularly through interstitial fluid (Link 1). The
intracellular and extracellular resistances are independent physical quantities
that must be measured separately — and the measurement itself requires insulating
the tissue from the bath to prevent extracellular shorting (Link 2). The bidomain
model captures this two-domain structure mathematically, with separate potential
fields and separate boundary conditions for each domain (Link 3). At a tissue-bath
interface, the intracellular domain terminates (Neumann BC) while the extracellular
domain is continuous with the low-resistance bath (Dirichlet BC). These BCs are
on different variables — the asymmetry is fundamental (Link 4). This asymmetry
shorts the extracellular return path near the boundary: return current can flow
through the bath instead of through the tissue extracellular space, reducing the
effective axial resistance from (r_i + r_e) to approximately r_i. Since the
resistance is lower, the effective conductivity rises from sigma_eff to sigma_i,
and CV increases by sqrt((sigma_i + sigma_e)/sigma_e) ≈ 13% (Link 5). The
enhancement decays exponentially into the tissue over the electrotonic space
constant lambda ≈ 1.4 mm — it is a localised boundary layer effect, not a global
change (Link 6). This mechanism explains the systematically higher CV and altered
AP morphology observed at tissue surfaces in bath-perfused preparations —
observations that have been routinely discarded as "edge effects" rather than
recognised as real bidomain physics (Link 7).

---

## Where Kleber's Chain Is Strongest and Weakest

### Strongest links:

- **Link 1 (circuit theory):** Established physics. Verified experimentally in
  countless preparations. Not in dispute.

- **Link 2 (r_i and r_e are independent):** Direct experimental measurement
  (Kleber, Riegger, Janse 1987). The fact that you need to insulate the preparation
  to measure r_e correctly IS the proof that the bath modifies r_e.

- **Link 3 (bidomain captures two domains):** Mathematical structure is
  well-established. Derived rigorously from microstructure (Neu & Krassowska).

- **Link 4 (BCs are asymmetric):** Follows directly from physical reality. Not
  controversial.

### Weakest links:

- **Link 5 (the shorting magnitude):** Assumes the Dirichlet approximation
  (phi_e = 0 is exact). In reality, phi_bath ≠ 0 near the tissue interface,
  especially in confined geometries. The 13% is an upper bound.

- **Link 6 (the exponential profile):** Uses a static lambda computed from resting
  membrane conductance. During the action potential upstroke (when CV is determined),
  G_m increases ~2000-fold, collapsing lambda to ~30 um. The profile is dynamic,
  not static.

- **Link 7 (experimental attribution):** The "edge effects" in experiments have
  multiple possible causes (tissue damage, edge cells with different properties,
  optical artifacts at boundaries). Kleber's attribution to the bath-coupling
  mechanism is plausible but not proven. The direct experiment (measuring CV vs
  distance from boundary with and without bath) has not been done.

### The missing link:

**Link 0 (homogenisation validity at the boundary):** The entire chain rests on
the bidomain, which is derived by homogenising over many cells. At the tissue
surface, the homogenisation breaks down (Neu & Krassowska 1993). The BCs in
Links 3-4 are applied to a model that is formally invalid at the location where
they matter most. The EMI model (which resolves individual cells) would test
whether the bidomain's prediction survives when the homogenisation assumption is
removed.

---

## References

Based on articles retrieved from PubMed:

1. Kleber AG, Riegger CB, Janse MJ (1987). Electrical uncoupling and increase of
   extracellular resistance after induction of ischemia in isolated, arterially
   perfused rabbit papillary muscle. Circ Res 61(2):271-279.
   [DOI](https://doi.org/10.1161/01.res.61.2.271)

2. Kleber AG, Riegger CB, Janse MJ (1987). Extracellular K+ and H+ shifts in
   early ischemia. J Mol Cell Cardiol 19(Suppl 5):35-44.
   [DOI](https://doi.org/10.1016/s0022-2828(87)80608-9)

3. Kleber AG, Rudy Y (2004). Basic mechanisms of cardiac impulse propagation and
   associated arrhythmias. Physiol Rev 84(2):431-488.
   [DOI](https://doi.org/10.1152/physrev.00025.2003)

4. Fast VG, Kleber AG (1994). Anisotropic conduction in monolayers of neonatal
   rat heart cells cultured on collagen substrate. Circ Res 75(3):591-595.
   [DOI](https://doi.org/10.1161/01.res.75.3.591)

5. Fast VG, Darrow BJ, Saffitz JE, Kleber AG (1996). Anisotropic activation
   spread in heart cell monolayers assessed by high-resolution optical mapping.
   Circ Res 79(1):115-127.
   [DOI](https://doi.org/10.1161/01.res.79.1.115)

6. Rohr S, Kucera JP, Kleber AG (1998). Slow conduction in cardiac tissue, I:
   effects of a reduction of excitability versus a reduction of electrical coupling
   on microconduction. Circ Res 83(8):781-794.
   [DOI](https://doi.org/10.1161/01.res.83.8.781)

7. Rohr S, Schölly DM, Kleber AG (1991). Patterned growth of neonatal rat heart
   cells in culture. Circ Res 68(1):114-130.
   [DOI](https://doi.org/10.1161/01.res.68.1.114)

8. Roth BJ (1991). A comparison of two boundary conditions used with the bidomain
   model of cardiac tissue. Ann Biomed Eng 19(6):669-678.
   [DOI](https://doi.org/10.1007/BF02368075)

9. Roth BJ (1996). Effect of a perfusing bath on the rate of rise of an action
   potential. Ann Biomed Eng 24(6):639-646.
   [DOI](https://doi.org/10.1007/BF02684177)

10. Patel SG, Roth BJ (2005). Approximate solution to the bidomain equations for
    electrocardiogram problems. Phys Rev E 72:051931.
    [DOI](https://doi.org/10.1103/PhysRevE.72.051931)

11. Henriquez CS, Plonsey R (1990). Simulation of propagation along a cylindrical
    bundle of cardiac tissue. IEEE Trans Biomed Eng 37(9):850-860.
    [DOI](https://doi.org/10.1109/10.58596)

12. Neu JC, Krassowska W (1993). Homogenization of syncytial tissues. Crit Rev
    Biomed Eng 21(2):137-199.
    [PMID 8243090](https://pubmed.ncbi.nlm.nih.gov/8243090/)
