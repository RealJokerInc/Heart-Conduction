# PNP-Cardiac Literature Survey

**Date:** 2026-04-23
**Scope:** What's been done with Poisson-Nernst-Planck (PNP) equations in cardiac and cellular electrophysiology. Maps the landscape of work that sits between standard quasi-static bidomain and full ion-level electrodiffusion.

---

## TL;DR

- **The homogenized PNP-bidomain already exists** in the literature. Two principal references: **Okada, Sugiura & Hisada 2013** (cardiac-specific) and **Whiteley 2020** (general, with Debye-layer treatment).
- **But these do not produce a 2-DOF-in-V formulation.** Homogenization of PNP in the normal cardiac regime still recovers the standard PE bidomain for the tissue-level potentials V_i, V_e. The quasi-static form is physically stable under homogenization.
- **Where PNP does add DOFs: ion concentrations.** Each ion species (Na⁺, K⁺, Ca²⁺, anions) becomes a dynamical tissue-level field with its own conservation equation. This is where PNP-bidomain is richer than standard bidomain.
- **Cell-resolved PNP** (Mori 2008, Jæger 2023/2025) preserves full Debye-layer and ephaptic physics at the μm–nm scale but does not produce a homogenized continuum model.
- **Tveito group's KNP-EMI** combines electroneutral Nernst-Planck (KNP) with cell-resolved EMI geometry. KNM is a simpler Kirchhoff-network approximation.
- **No published LBM-PNP for cardiac.** Existing LBM-PNP work is exclusively electrokinetic (microfluidics, porous media). This is the principal gap our project could fill.

---

## Classification of approaches

| Approach | Level | V dynamics | Ion species | Geometry | Key reference |
|---------|--------|------------|-------------|-----------|----------------|
| **Standard bidomain** | Continuum | 1-DOF (V), φ_e elliptic | Parametric (constants) | Homogenized | Tung 1978 |
| **Cattaneo bidomain** | Continuum | 1-DOF-with-memory (Q) | Parametric | Homogenized | Rossi-Griffith 2017 |
| **PNP-bidomain (Okada)** | Continuum | V_i, V_e elliptic-coupled | **Dynamical (per species)** | Homogenized | Okada 2013 (Phys Rev E) |
| **Whiteley PNP homogenization** | Continuum | Elliptic after hmg | Dynamical at microscale | Almost-periodic | Whiteley 2020 |
| **EMI** | Cell-resolved | 1-DOF (v) on membrane, bulk elliptic | Parametric | Explicit cells | Tveito et al. |
| **KNP-EMI** | Cell-resolved | Elliptic bulk, coupled ions | **Electroneutral dynamical** | Explicit cells | Solbrå/Tveito et al. |
| **KNM** | Cell-resolved | Kirchhoff network | Parametric | Discrete cell-nodes | Jæger-Tveito 2023 npj Sys Bio |
| **Full PNP (Mori, Jæger)** | Cell-resolved | Full PNP | **Fully dynamical** | Explicit cells, Debye-layer-resolved | Mori 2008 PNAS; Jæger 2023 PLOS CB |

---

## Key papers — what they do and where they sit

### Okada, Sugiura, Hisada 2013 (Phys Rev E 87:062701) ⭐ **most relevant**

**DOI:** 10.1103/PhysRevE.87.062701 — paywalled, no open preprint found.

"Modeling for cardiac excitation propagation based on the Nernst-Planck equation
and homogenization."

**What they do.** Start from Nernst-Planck + ionic conservation + electroneutrality
(`Σ z_α n_α = 0`) at the cellular level. Introduce a homogenization method and assume
microscopic uniformity. Derive **"rational bidomain equations at the macroscopic
level."** Claim this fixes a "self-contradiction" in the standard bidomain: standard
bidomain updates ion concentrations without accounting for electric-drift or capacitive
current contributions, despite those being used to assemble Kirchhoff's law.

**Why it matters for us.** This IS the homogenized PNP-bidomain for cardiac. Probably
the closest thing in literature to what the "2-DOF bidomain" question lands on once you
commit to preserving ion-level physics. The paper needs to be acquired (library
request, institutional download, or ask an author).

**What I don't know until I read the PDF.**
- Does Okada's "rational bidomain" keep V_e elliptic, or does it have genuine 2-DOF in
  potentials?
- What electroneutrality assumption does he invoke, and does it still collapse V_c
  dynamics via charge neutrality?
- What are the tissue-level dynamical variables — is it just ion concentrations, or
  also ρ_c?

**Action item.** Acquire this PDF. It may answer the "does homogenized PNP give us 2
DOF in V?" question conclusively.

### Whiteley 2020 (Math Med Biol 37:262-302) ⭐⭐ **have the PDF, most critically useful**

**DOI:** 10.1093/imammb/dqz014 — Oxford ORA open preprint acquired.

"An evaluation of some assumptions underpinning the bidomain equations of
electrophysiology."

**What he does.** Starts from microscale PNP (§2.1 eqs 2.1–2.11): Nernst-Planck for
Na⁺/K⁺/Ca²⁺/A⁻ in Ω_i, symmetric in Ω_e, plus Poisson in each compartment and Laplace
in the membrane. Explicitly resolves Debye layers (§3, limit analysis). Derives the
capacitor relation Q = C_m V as a **consequence** of Debye-layer physics (under
reasonable permittivity assumptions), not as a postulate — improvement over Richardson
2009's squid-axon asymptotic analysis. Then homogenizes to tissue-level with an
almost-periodic microstructure technique (Richardson-Chapman 2011 method).

**Final result.** Homogenization recovers the **standard bidomain form** (eq 1.1:
parabolic-elliptic with C_m ∂V/∂t + I_m source). Investigates assumption that
conductivity tensors are diagonal and constant — **concludes the assumption is usually
valid but names situations where it isn't** (non-cardiac orientations of sheets,
non-periodic microstructure variations, etc.).

**Why it matters for us.** Shows that the PE bidomain **is the correct homogenization
of PNP under normal assumptions**. This is bad news for "2-DOF-in-V via homogenized
PNP" — the quasi-static form is structurally robust. Good news: he specifies the cases
where the assumption fails, which gives us a targeted list of conditions under which a
2-DOF correction might be real.

**Two further findings to track down in the paper's §3–4.**
1. The explicit form of the Debye-layer correction to the capacitor relation (the Q–V
   relationship beyond Q = C_m V). If nonlinear, this is a window onto genuinely
   non-quasi-static bidomain behavior near the membrane.
2. Whether the homogenized bulk equations include any ion-concentration dynamics at
   the tissue level (analogous to Okada 2013) or whether he treats them as parameters.

### Mori, Fishman & Peskin 2008 (PNAS 105:6463) ⭐

**DOI:** 10.1073/pnas.0801089105 — PDF blocked by PMC anti-scraping; abstract in hand.

"Ephaptic conduction in a cardiac strand model with 3D electrodiffusion."

**What they do.** Full PNP in a 3D cardiac strand model (cell-resolved, not
homogenized). Study cardiac AP propagation under severe gap-junction reduction.
Compare to 1D ephaptic models. Identify a mode where ephaptic and gap-junction-mediated
propagation alternate.

**Why it matters.** Foundational cardiac PNP paper. Demonstrates that explicit
Debye-layer and ion-concentration physics matter for ephaptic conduction — a regime
where standard bidomain is known to fail. Predates the modern cell-by-cell frameworks
but established the modeling technique.

**Limitation for us.** Cell-resolved, not homogenized. Cannot give us a
continuum-level dual-evolving bidomain directly; serves as validation target for
continuum models.

### Jæger, Ivanović, Kučera, Tveito 2023 (PLOS Comp Biol 19:e1010895)

**DOI:** 10.1371/journal.pcbi.1010895 — PMC9974139, full text extracted via MCP.

"Nano-scale solution of the Poisson-Nernst-Planck (PNP) equations in a fraction of
two neighboring cells reveals the magnitude of intercellular electrochemical waves."

**What they do.** Full PNP at nano-meter resolution in a small piece of two adjacent
cells + their extracellular cleft. Show that when Na⁺ channels open in one cell, an
intercellular electrochemical wave can propagate to the neighbor through the cleft —
potentially explaining cardiac conduction at low gap-junctional coupling.

**Why it matters.** Modern confirmation of Mori 2008 at higher resolution. Tveito
group's use of PNP is at sub-cellular scale; they do **not** attempt continuum
homogenization. The resolution is nano-meters, timestep is nano-seconds.

**What this rules out.** Using full PNP directly for tissue-level simulations — cost
is prohibitive.

### Jæger & Tveito 2023 (npj Sys Bio Apps 9:29 — "KNM")

**DOI:** 10.1038/s41540-023-00288-3 — PMC10267147.

"Efficient, cell-based simulations of cardiac electrophysiology; The Kirchhoff Network
Model (KNM)."

**What they do.** Each cardiomyocyte becomes a node in an electrical network linked by
gap-junction resistances. Kirchhoff current conservation at each node + membrane ODEs.
Cell-level resolution without the 3D continuum cost. Simpler than EMI; not PNP
(concentrations not tracked).

**Why it matters.** Sits orthogonal to our direction — cell-resolved but
quasi-static/reduced. Useful as a comparison point for conduction velocity under
reduced coupling, not a formulation we'd adopt.

### Tveito group 2022 (eNeuro — "KNP-EMI validation")

**DOI:** 10.1523/ENEURO.0408-21.2022 — PMC9045477.

"Validating a Computational Framework for Ionic Electrodiffusion with Cortical
Spreading Depression as a Case Study."

**What they do.** KNP-EMI = Kirchhoff-Nernst-Planck + EMI. Combines electroneutral
ion-concentration electrodiffusion with cell-resolved geometry. Applied to cortical
spreading depression (brain), validated against experiments. Not cardiac-specific but
the framework extends to cardiac.

**Why it matters.** This is the Tveito-school answer to "how do we add PNP to EMI
affordably." Electroneutral assumption drops one DOF (ρ_c = 0 constraint at macroscale)
but keeps ion-species dynamics. This is very close to what Okada does at the continuum
level.

### Pods, Schönke, Bastian 2013 (Biophys J 105:242)

**DOI:** 10.1016/j.bpj.2013.05.041 — PMC3703912.

"Electrodiffusion models of neurons and extracellular space using the PNP equations."

**What they do.** 3D PNP for a single axon exploiting cylinder symmetry. Explicitly
resolve the Debye layer with a fine computational mesh. Compare to the line-source
approximation and identify an "action potential echo" signal component.

**Why it matters.** Methodological template. If we ever need a PNP code for
comparison/validation, this is a well-documented finite-element reference (neural, not
cardiac, but transferable).

### Pods 2017 (J Integr Neurosci 16:19–32)

**DOI:** 10.3233/JIN-170009.

"A comparison of computational models for the extracellular potential of neurons."

**What he does.** Theoretical and numerical comparison of **VC (volume conductor, =
elliptic bidomain)**, **PNP (full electrodiffusion)**, and **EN (electroneutral)** for
neural extracellular potentials.

**Why it matters for us.** This is the three-way comparison we want for cardiac. No
cardiac equivalent exists — a parallel of this study for cardiac tissue would be a
genuine contribution. Pods gives the framework and tells you what to look for.

### Ivanović et al. 2025 (J Physiol) & Horgmo Jæger et al. 2025 (Biophys J)

Two very recent papers continuing Tveito-school cardiac PNP. Ivanović focuses on
intercalated-disc geometry effects on ephaptic coupling; Horgmo Jæger examines ionic
electrodiffusion in cardiac dyads at nano-scale. Both use KNP/PNP frameworks, both
cell-resolved.

---

## Papers to acquire (priority order)

1. **Okada, Sugiura, Hisada 2013** — the cardiac homogenized PNP-bidomain. Paywalled
   (Phys Rev E). Institutional download needed.
2. **Pods 2013 Biophys J** — PMC blocked; try direct institutional access via Biophys J.
3. **Jæger-Tveito-Ivanović 2023** (PLOS CB, intercellular waves) — open access but blocked
   by scraping. Manual download via biorxiv mirror likely works.
4. **Richardson 2009** — cited by Whiteley as precursor work on squid axon Debye-layer
   asymptotic analysis. If Whiteley builds on it, we should read it.
5. **Mori 2009** (CPA) — the electroneutral limit of electrodiffusion paper.
   Mathematically substantial. Paywalled.

## The LBM-PNP gap

Extensive search found LBM-PNP work in electrokinetics (microfluidics, porous media,
electro-osmotic flow) but **no LBM-PNP paper for cardiac electrophysiology**. The
closest adjacent work:

- LBM for ion transport in nanochannels (Kim et al. 2018, Zou et al. 2008, etc.)
- LBM for electro-osmotic flow (Wang & Kang 2009, Chai & Shi 2008)
- LBM for Poisson-Boltzmann (Chai-Zhao-Wang multiple papers)

The technical machinery for LBM-PNP exists. Adapting it to the cardiac ionic-model
coupling (gating variables, I_ion on the membrane) would be the specific contribution.

---

## What this lets us conclude

1. **The "dual-evolving bidomain" project does not have an obvious literature
   precedent.** The existing homogenized PNP papers (Okada 2013, Whiteley 2020)
   derive something close to standard quasi-static PE bidomain as the homogenized
   limit. Cell-resolved PNP work (Mori, Jæger, Tveito) preserves the 2-DOF richness
   of PNP but at nano-scale, not tissue-level.

2. **The actionable 2-DOF direction is: homogenized PNP with ion-species tracking.**
   V_i, V_e stay elliptic-coupled (quasi-static) but Na⁺, K⁺, Ca²⁺ concentrations in
   each compartment become tissue-level dynamical fields. This is the Okada 2013
   framework. Each ion species maps to one LBM lattice (drift-diffusion with source).
   Total lattice count for cardiac: ~8 (4 ions × 2 compartments).

3. **"Two independent V_c lattices that each do Cattaneo kinetics and carry their own
   time evolution" is a formulation that doesn't exist in the literature.** If we build
   it, it's novel. But it needs a defensible physical basis — options discussed earlier
   (phenomenological C_c, lattice-scale freedom) remain on the table but are weak.

4. **The strongest position to write papers from is probably a hybrid:** implement
   Okada-style homogenized PNP-bidomain (ion species dynamical at tissue level, V_c
   quasi-static), solve via LBM with one lattice per (species × compartment), and
   demonstrate it captures the boundary artifacts / ephaptic effects that cell-resolved
   PNP models predict but standard bidomain misses. This is closer to Okada's framework
   but novel at the LBM implementation level.

5. **What to do before writing any more reformulation docs:** acquire Okada 2013 PDF.
   If Okada's "rational bidomain" is what we want, the formulation is already settled
   and we focus on implementation. If it isn't, we know precisely what's missing and
   can write the gap from first principles.
