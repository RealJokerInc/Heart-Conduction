# LBM Thickness-Weighting: Analogous Physics, Source Papers & Vendored Code

> Agentic-search synthesis, 2026-06-07. Focus: implementing the thickness-weighted
> operator `(1/T)∇·(T·D∇V)` in our LBM cardiac engine, the published *physical
> analog* to copy from, and the actual code we could vendor.
> See also: `deep_research_thickness_source_sink_2026-06-06.md` (the model itself),
> KNOWLEDGE.md (synthesis).

## TL;DR

- The operator expands to `D∇²V + D·(∇T/T)·∇V` — an advection-diffusion equation
  with **drift velocity `u_drift = D·∇(lnT)`**. Standard LBM-ADE machinery; the drift
  goes in the equilibrium distribution.
- The **direct physical analog is depth-averaged shallow-water transport with variable
  water depth** `h(x)`: `(1/h)∇·(h·D∇C)` — h is literally a thickness/cross-section
  field, identical math.
- **No public code exists for the analogous LBM paper** (verified negative). The scheme
  must be re-implemented from the paper's equations.
- **The clonable code is BeatBox** — the validation engine behind the *original*
  monodomain thickness-reduction paper. Vendored at `Research/code_examples/beatbox/`.

---

## 1. Original monodomain reduction paper (the model)

> **Biktasheva, I.V., Dierckx, H., Biktashev, V.N. (2015).** "Drift of scroll waves
> in thin layers caused by thickness features: asymptotic theory and numerical
> simulations." *Physical Review Letters* **114**, 068302.
> DOI: 10.1103/PhysRevLett.114.068302 · arXiv: **1408.3654**

**CORRECTION carried into KNOWLEDGE.md:** this is a **2015 PRL (114:068302)**, NOT
"PRL 2019" as previously recorded. The arXiv id 1408.3654 we already had is correct
and resolves to this 2015 paper. (The only 2019 Dierckx/Biktasheva PRE paper is an
unrelated response-function work.)

Content: thin-layer thickness-reduction asymptotics yielding the augmented operator
`u_t = f(u) + D(1/H)∇·(H∇u) + O(μ²)` (their Eq 4), with the correction scaling as
`∇(lnH) = ∇H/H` (their Eq 5, `h = D(∇K)·∇u, K=lnH`) — the rigorous origin of
Ciaccio's phenomenological `∇T/T`. Validated against full 3-D in **FitzHugh-Nagumo +
Oregonator** media with a thickness step.

**Validation code = BeatBox** (confirmed). HPC C/MPI cardiac EP simulator.
Reference: Antonioletti, Biktashev, Jackson, Kharche, Stary, Biktasheva (2017),
"BeatBox—HPC simulation environment for biophysically and anatomically realistic
cardiac electrophysiology," *PLOS ONE* **12(5)**:e0172292,
DOI 10.1371/journal.pone.0172292 (arXiv:1605.06015).

---

## 2. LBM analogous-physics paper (the recipe to copy)

> **Ru, Z., Liu, H., Xing, L., Ding, Y. (2021).** "A well-balanced lattice Boltzmann
> model for the depth-averaged advection–diffusion equation with variable water depth."
> *Computer Methods in Applied Mechanics and Engineering* **379**, 113745.
> DOI: 10.1016/j.cma.2021.113745

**CORRECTION:** earlier notes cited "CMAME 374:113563" — the verified record is
**vol 379, art 113745**.

Why it's the best analog: it solves exactly `(1/h)∇·(h·D∇C)` with `h(x)` a literal
variable depth/thickness field — same math as `(1/T)∇·(T·D∇V)`.

**The scheme (what to lift):**
- Product-rule split: `(1/h)∇·(h·D∇C) = D∇²C + (D/h)∇h·∇C`. The second term is recast
  as a **"pseudo-velocity"** `u_pseudo = D·∇h/h = D·∇(ln h)` and **folded into the
  equilibrium's advection term**: `f_i^eq = w_i·C·[1 + (e_i·u)/c_s²]` with
  `u → u_phys + u_pseudo`. This is the literal water-physics counterpart of our
  `u_drift = D·∇(lnT)`.
- Two equilibrium discretizations of `∇h`: a **"center scheme"** and a
  **"linked scheme"** (∇h at node vs. on links). The **linked scheme is the
  well-balanced one**.
- Lattice/collision: the **solute (ADE) lattice is D2Q5 + BGK** (τ independent of D);
  the flow lattice is D2Q9 + central-moments (irrelevant to us — we supply zero bulk
  flow, drift only).
- **Well-balanced construction = the key contribution.** The naive advective LBM-ADE
  produces a spurious source `∝ C·∇·u` (our feared `V·∇·u_drift` artifact, nonzero
  since `∇·u_drift = D∇²(lnT) ≠ 0`). Their linked-scheme correction cancels it so a
  uniform field over variable depth stays uniform ("C-property" for transport).
  **This is the single most valuable thing to port.**
- Stability/CFL: D2Q5 BGK needs `τ > 0.5`; the pseudo-velocity must obey the low-Mach
  ADE limit `|u_total|/c_s ≪ 1`, i.e. `D·|∇lnT|` small vs lattice speed. Steep `∇lnT`
  (sharp thickness steps — exactly our block regime) violates low-Mach → refine Δx or
  cap the gradient there.

**Secondary/backup references (theory only, NOT primary analog):**
- Yoshida & Nagaoka (2014), "LBM for the convection–diffusion equation in curvilinear
  coordinate systems," *J. Comput. Phys.* **257**:884–900, DOI 10.1016/j.jcp.2013.09.035.
  (MRT; apparent anisotropy from the Jacobian. NB: authors are Yoshida & Nagaoka — our
  earlier "Yang 2014" attribution was wrong.) Useful if we later add anisotropy.
- Guo & Zhao (2002) porous-media LBM, *Phys. Rev. E* **66**:036304 — porosity ε(x) in
  equilibrium + force; looser analog, reference-only.

---

## 3. Code repositories

**Verified negative:** the analogous-physics LBM papers (Ru 2021; Yoshida–Nagaoka 2014)
have **NO public code** (checked GitHub/GitLab/Zenodo/OpenAlex). The well-balanced
variable-depth scheme must be re-implemented from the manuscript.

| Repo | Contents | License | Verdict |
|---|---|---|---|
| **BeatBox** `github.com/beatbox-heart/bb2-public` | C/MPI HPC cardiac EP (0D–3D, FHN/Oregonator/ttp06, scroll/filament tracking). Validation engine for Biktasheva 2015. | GPL-3.0 | **VENDORED** — 3-D ground-truth reference |
| Kuzmin LBM examples `github.com/shurikkuzmin/LatticeBoltzmannMethod` | Python/Jupyter D2Q5 ADE-with-drift (TRT, Gaussian hill). Clean skeleton. | **none (all-rights-reserved)** | reference-only; **do NOT vendor** (no license) |
| Palabos `gitlab.com/unigespc/palabos` | C++ LBM framework, ADE + forcing infra | AGPL-3.0 (viral) | reference-only; heavyweight |

### Vendored: `Research/code_examples/beatbox/`
- Cloned 2026-06-07 from `github.com/beatbox-heart/bb2-public` (main @ a629c420), GPL-3.0.
- Nested `.git` stripped (vendored as plain files, matching the other code_examples repos).
- Heavy `.bbg/.bbg.gz` geometry binaries (>200K) deleted — 153M → 13M. Full `src/`
  (7.2M) and all **113 `.bbs` example scripts** retained.
- Directly relevant scripts: `data/scripts/**/FitzHughNagumo_model/*.bbs`, especially
  `fhn3_NegativeTension.bbs` / `fhn3_PositiveTension.bbs` — scroll-wave filament-tension
  examples, the FHN medium Biktasheva 2015 used for the thickness-drift validation.
- GPL-3.0 is fine for an internal validation tool; do not statically link into a
  redistributed closed engine.

### Implementation note
Our LBM thickness-weighted operator stays **clean-room**: port the Ru et al. (2021)
linked-scheme pseudo-velocity + well-balanced cancellation from the paper's equations
(no copyable source exists), validate against BeatBox's 3-D FHN thickness-drift result.
