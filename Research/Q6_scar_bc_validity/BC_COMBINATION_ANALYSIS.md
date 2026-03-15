# Exhaustive BC Combination Analysis for Inert Boundary Physics

## 0. What "Correct Physics" Means

At a clean tissue-void interface in a bath-perfused hiPSC monolayer:

**Two domains, two different BCs:**
```
Intracellular (phi_i):  Neumann  (cell network terminates — no flux)
Extracellular (phi_e):  Dirichlet (continuous with bath — phi_e = phi_bath ≈ 0)
```

**What this produces:**
1. The bath shorts the extracellular return path near the boundary
2. Effective diffusivity transitions from D_eff to D_i over length scale lambda
3. CV increases ~13% along flat edges, independent of geometry
4. The transition is smooth (exponential decay into tissue bulk)
5. The boundary cell's AP morphology changes (reduced dV/dtmax per Roth 1996)
6. The effect is BIDOMAIN-STRUCTURAL — it arises from having two domains with
   asymmetric BCs. It cannot arise from any single-domain manipulation.

**Additionally (geometry-dependent effects, captured by any solver):**
7. Curvature at corners: v(kappa) = v_0 - D*kappa
8. Source-sink mismatch at expansions: safety factor determines block/propagation
9. Rate-dependent block through isthmuses

---

## 1. Monodomain FDM Combinations

The monodomain equation is:
```
chi*Cm * dV/dt = div(D*grad(V)) - chi*I_ion
```
with a single variable V and a single BC at tissue-void interface.

### Available knobs:
- BC type on V: Neumann (implemented), Dirichlet (not implemented but trivial), Robin (not implemented)
- D profile: uniform scalar, spatially varying field (Dxx, Dxy, Dyy per node)
- Stencil: 9-point with ghost-node mirror

### Exhaustive combinations:

| # | BC on V | D profile | Mechanism at boundary | Correct physics? |
|---|---------|-----------|----------------------|-------------------|
| M1 | Neumann | Uniform D_eff | Flat edge invisible to planar wave. kappa=0 → no effect. | Captures 7,8,9. Misses 1-6. |
| M2 | Dirichlet V=V_rest | Uniform D_eff | Current SINK: boundary drains current during depolarisation. Electrotonic loading INCREASES. | **WRONG direction.** Slowdown, not speedup. |
| M3 | Robin (alpha*V + beta*dV/dn = 0) | Uniform D_eff | Weak current sink for any alpha/beta > 0. Interpolates between M1 (alpha=0) and M2 (beta=0). | **WRONG direction** for alpha > 0. Reduces to M1 for alpha=0. |
| M4 | Robin (dV/dn = +alpha*(V - V_rest)) | Uniform D_eff | Current SOURCE when V > V_rest. Would speed up conduction. | **Wrong mechanism.** Where does the current come from? Nonphysical — violates conservation. No bath domain to supply it. |
| M5 | Neumann | D(x,y) = D_eff + (D_i - D_eff)*exp(-d/lambda) | Enhanced diffusion near boundary, decaying into bulk. Correct transition profile. | **Best monodomain approximation.** Captures 1-3 approximately, plus 7-9. Misses 5 (AP morphology). |
| M6 | Dirichlet V=V_rest | D(x,y) varying | D enhancement (speedup) + current sink (slowdown). Competing effects. Net direction depends on magnitudes. | **Unstable compromise.** Even if magnitudes balance, mechanism is wrong. |
| M7 | Robin | D(x,y) varying | D enhancement + weak sink. | **Worse than M5** — Robin adds artifact on top of the correct D-based mechanism. |

### Verdict: M5 is the only monodomain FDM combination worth pursuing.

**Why no BC on V can produce speedup:** The monodomain PDE with uniform D admits a
1D traveling wave solution for planar fronts. The y-BC is mathematically irrelevant
(dV/dy = 0 everywhere satisfies the PDE and any BC). The ONLY way to make the
boundary "visible" to a planar wave is to make D depend on y — this modifies the
PDE itself, not the BC. Any BC on V either does nothing (Neumann), drains current
(Dirichlet/Robin with positive conductance), or injects nonphysical current (Robin
with negative conductance).

---

## 2. Monodomain LBM Combinations

The LBM solves the same monodomain PDE but via distribution functions f_i. Additional
knobs: stencil (D2Q5/D2Q9), collision (BGK/MRT), and LBM-specific BC implementations.

### Available knobs:
- Stencil: D2Q5, D2Q9
- Collision: BGK, MRT (with per-moment relaxation rates)
- BC type: Bounce-back (Neumann), Anti-bounce-back (Dirichlet), Absorbing
- BC can be applied per-direction (different treatment for cardinal vs diagonal in D2Q9)
- tau/omega: uniform or spatially varying (controls D)
- Source term: standard -(I_ion + I_stim)/Cm
- Weights: fixed per lattice (not per-node configurable)

### Exhaustive combinations:

#### 2A. Standard BC variants (uniform tau)

| # | Stencil | Collision | BC | Effect at flat boundary | Correct? |
|---|---------|-----------|-----|------------------------|----------|
| L1 | D2Q5 | BGK | Full BB | Exact Neumann. No artifact. Planar wave = 1D. | M1 equivalent. Misses 1-6. |
| L2 | D2Q5 | MRT | Full BB | Same as L1 (MRT and BGK identical for Neumann on D2Q5). | Same as L1. |
| L3 | D2Q9 | BGK | Full BB | Neumann + O(dx^2) diagonal artifact → ~3% SLOWDOWN. | **Worse than M1.** Wrong direction artifact. |
| L4 | D2Q9 | MRT | Full BB | Same artifact as L3 (MRT doesn't eliminate bounce-back error). | Same as L3. |
| L5 | D2Q5 | BGK | Full ABB (V=V_rest) | Dirichlet. Current sink. Slowdown. | M2 equivalent. **WRONG.** |
| L6 | D2Q5 | MRT | Full ABB | Same as L5. | **WRONG.** |
| L7 | D2Q9 | BGK | Full ABB (V=V_rest) | Dirichlet + diagonal artifacts. Current sink. | M2 equivalent + artifacts. **WRONG.** |
| L8 | D2Q9 | MRT | Full ABB | Same as L7. | **WRONG.** |
| L9 | D2Q5 | BGK | Absorbing | Open boundary. Non-conservative. V approaches local equilibrium at boundary. | **WRONG context.** Tissue doesn't terminate openly — cells end. |
| L10 | D2Q9 | BGK | Absorbing | Same issue as L9. | **WRONG context.** |

#### 2B. Hybrid BC variants (D2Q9 only — mix cardinal and diagonal treatment)

In D2Q9 at a flat y=0 boundary, 3 directions hit the wall:
- Cardinal: f_4 (south, weight 1/9)
- Diagonal: f_7 (SW, weight 1/36), f_8 (SE, weight 1/36)

We can treat cardinal and diagonal differently:

| # | Cardinal (f_4) | Diagonals (f_7, f_8) | Net effect | Correct? |
|---|---------------|---------------------|------------|----------|
| L11 | BB (Neumann) | BB (Neumann) | = L3/L4. Standard Neumann with diagonal artifact. | No. |
| L12 | BB (Neumann) | ABB (Dirichlet) | Cardinal: correct no-flux. Diagonals: drain current via x-component. Net: current sink in x-direction at boundary. | **WRONG.** Directional current drain → anisotropic slowdown. |
| L13 | ABB (Dirichlet) | BB (Neumann) | Cardinal: current drain in y. Diagonals: Neumann. Net: y-directed drain. | **WRONG.** y-drain pulls current away from propagation direction. |
| L14 | ABB (Dirichlet) | ABB (Dirichlet) | = L7/L8. Full Dirichlet. | **WRONG.** |
| L15 | BB (Neumann) | Absorbing | Cardinal: no-flux. Diagonals: equilibrium replacement. Net: partial current loss through diagonals. | **WRONG.** Still a net drain. |
| L16 | Absorbing | BB (Neumann) | Cardinal: open. Diagonals: Neumann. | **WRONG.** Non-conservative at boundary. |
| L17 | BB (Neumann) | Pass-through (no BC) | Cardinal: Neumann. Diagonals: stream normally from void (where V = 0 or undefined). | **WRONG.** Streaming from void = Dirichlet V=0. Current drain. |

**Why all hybrid BCs fail:** The fundamental issue is that EVERY non-Neumann treatment
of ANY direction at the boundary introduces either a current drain (if the void is at
lower potential) or a nonphysical source (if the void is at higher potential). There is
no intermediate option that produces enhanced diffusion. Enhanced diffusion is a BULK
property (tau/D), not a boundary condition property.

#### 2C. Modified relaxation at boundary (spatially varying tau)

| # | Stencil | Collision | BC | tau profile | Effect | Correct? |
|---|---------|-----------|-----|-------------|--------|----------|
| L18 | D2Q5 | BGK | Full BB | tau(x,y) from D(x,y) | Enhanced D near boundary + exact Neumann. | **M5 equivalent. Best LBM monodomain.** |
| L19 | D2Q5 | MRT | Full BB | s_jx(x,y), s_jy(x,y) from D(x,y) | Same as L18 but anisotropic D profile possible. | **M5 equivalent with anisotropy.** |
| L20 | D2Q9 | BGK | Full BB | tau(x,y) from D(x,y) | Enhanced D near boundary + O(dx^2) artifact. | M5 + small artifact. **Acceptable.** |
| L21 | D2Q9 | MRT | Full BB | s_jx(x,y), s_jy(x,y) from D(x,y) | Same as L20 with anisotropy. | **Acceptable.** |
| L22 | D2Q9 | MRT | Full BB | s_jx, s_jy, PLUS modified s_pxx/s_pxy at boundary | Changes stress relaxation at boundary. Alters higher-order error terms. | **Marginal.** Doesn't change leading-order D, only error structure. |
| L23 | D2Q5 | BGK | Full ABB | tau(x,y) from D(x,y) | Enhanced D + current sink. Competing. | **WRONG.** Sink partially cancels enhancement. |
| L24 | D2Q9 | BGK | Full ABB | tau(x,y) from D(x,y) | Enhanced D + current sink + artifact. | **WRONG.** Three competing effects. |

#### 2D. Source term modification

| # | Base config | Source modification at boundary | Effect | Correct? |
|---|------------|-------------------------------|--------|----------|
| L25 | L1 (D2Q5 BB uniform) | Add R_bath = +(D_i - D_eff)*d²V/dx² at boundary nodes | Injects extra diffusive current equivalent to enhanced D. | **Mathematically equivalent to M5/L18.** But fragile: requires computing d²V/dx² from the distribution functions and is sensitive to noise. |
| L26 | L1 (D2Q5 BB uniform) | Add R_bath = +g_bath*(V_boundary_ahead - V) | Artificial lateral coupling enhancement. | **Wrong mechanism.** This is a made-up coupling, not derived from physics. Direction of effect depends on sign of g_bath. |

### Verdict for monodomain LBM:
**L18 or L19 (D2Q5 + BB + spatially varying tau) is the best.** L20/L21 (D2Q9) is
acceptable with the understanding that the O(dx^2) diagonal artifact adds ~3% error
at coarse grids.

No combination of BC types alone (without spatially varying D) can produce the Kleber
speedup. The effect is fundamentally about enhanced BULK diffusion near the boundary,
not about what happens AT the boundary node.

---

## 3. Bidomain FDM Combinations

The bidomain has TWO independent BCs — one on phi_i (or V) and one on phi_e. This is
the critical structural advantage.

### Available knobs:
- Intracellular BC: Neumann (implemented), Dirichlet (implementable via row elimination)
- Extracellular BC: Neumann (implemented), Dirichlet (implemented), Robin (not implemented but straightforward)
- Per-edge control: each of 4 edges can have independent BC type
- Solver: 5 strategies (decoupled_gs, semi_implicit, etc.)
- D_i, D_e: scalar or spatially varying fields

### Exhaustive combinations (intracellular × extracellular):

| # | Intra BC (phi_i) | Extra BC (phi_e) | Physical meaning | Produces Kleber? |
|---|-----------------|-----------------|------------------|-----------------|
| B1 | Neumann | Neumann | Fully insulated tissue (no bath). | **No.** No BC asymmetry → no Kleber. CV ratio = 1.00. Reference case. |
| B2 | Neumann | Dirichlet (phi_e = 0) | Cells terminate, bath shorts extracellular. | **YES.** Correct physics. CV ratio → 1.131. |
| B3 | Neumann | Dirichlet (phi_e = V_bath ≠ 0) | Bath at nonzero potential (e.g., stimulating electrode in bath). | **YES** with offset. Same mechanism as B2 but phi_e → V_bath. Relevant if bath potential is non-trivial. |
| B4 | Neumann | Robin (alpha*phi_e + beta*d(phi_e)/dn = 0) | Partial bath coupling. Models resistive layer between tissue and bath (e.g., connective tissue capsule, culture substrate with partial perfusion). | **Partial Kleber.** Speedup magnitude depends on alpha/beta ratio. Reduces to B2 (beta→0) or B1 (alpha→0). |
| B5 | Dirichlet (phi_i = V_rest_i) | Neumann | Clamp intracellular at rest, insulate extracellular. | **Nonphysical.** Where does the intracellular current drain to? Cells can't "connect to ground" without an extracellular path. Would create artificial current sink in phi_i. |
| B6 | Dirichlet (phi_i = V_rest_i) | Dirichlet (phi_e = 0) | Clamp both domains. V = phi_i - phi_e is clamped. | **Nonphysical.** Equivalent to Dirichlet V = V_rest at the boundary PLUS Dirichlet phi_e = 0. Double constraint. Over-determined. |
| B7 | Robin (intra) | Neumann | Partial intracellular leakage, insulated extra. Models damaged cells at cut edge with incomplete membrane seal. | **Nonphysical for clean cut.** Laser cut kills cells cleanly → Neumann. Robin would model a damaged border zone, which we explicitly exclude. |
| B8 | Robin (intra) | Dirichlet (phi_e = 0) | Partial intra leakage + bath coupling. | **Over-specified.** For clean inert boundaries, intra should be Neumann. Robin on intra would model damage, contradicting "clean cut." |
| B9 | Robin (intra) | Robin (extra) | Partial everything. | **Vague.** Neither domain has clean physics. Not justified for laser-cut voids in bath. |

### Per-edge mixed BCs:

The bidomain engine supports different BCs on different edges. For a rectangular
domain with a central inert void:

| # | Void edges (intra) | Void edges (extra) | Domain edges (intra) | Domain edges (extra) | Notes |
|---|-------------------|-------------------|---------------------|---------------------|-------|
| B10 | Neumann | Dirichlet | Neumann | Neumann | Kleber at void. Insulated outer boundary. | **Standard ML-DO setup.** |
| B11 | Neumann | Dirichlet | Neumann | Dirichlet | Kleber everywhere (void + outer boundary both bath-coupled). | Only if monolayer is fully submerged and domain boundary = tissue edge in bath. |
| B12 | Neumann | Robin | Neumann | Neumann | Partial Kleber at void. Insulated outer. | For imperfect bath coupling at void. |

### Verdict for bidomain FDM:
**B2 (Neumann intra + Dirichlet extra) is the correct and unique physical combination**
for clean tissue-void interfaces in bath-perfused monolayers. No other combination is
justified.

B4 (Robin on extra) is worth considering only if the bath-tissue coupling is known to
be imperfect — e.g., if the laser-cut void doesn't fill completely with Tyrode's.

---

## 4. Bidomain LBM Combinations (Not Yet Implemented)

A bidomain LBM would use two independent lattices — one for phi_i (or V), one for phi_e —
coupled through the transmembrane current source term.

### Available knobs (per lattice):
- Stencil: D2Q5, D2Q9
- Collision: BGK, MRT
- BC: BB (Neumann), ABB (Dirichlet), Absorbing, hybrid
- tau: from D_i (intra lattice) or D_e (extra lattice)

### Key combinations (stencil × BC on each lattice):

| # | Intra stencil | Intra BC | Extra stencil | Extra BC | Notes |
|---|--------------|---------|--------------|---------|-------|
| BL1 | D2Q5 | BB (Neumann) | D2Q5 | ABB (phi_e=0) | **Cleanest.** No diagonal artifacts on either lattice. Correct BCs. |
| BL2 | D2Q9 | BB (Neumann) | D2Q9 | ABB (phi_e=0) | Correct BCs but O(dx^2) diagonal artifact on BOTH lattices. Intra artifact: ~3% V slowdown. Extra artifact: ~3% phi_e error. Effects partially cancel in the CV ratio? Unknown — needs testing. |
| BL3 | D2Q5 | BB | D2Q9 | ABB | Mixed stencils. Intra clean, extra has diagonal artifact. | Possible but awkward — different isotropy per domain. |
| BL4 | D2Q9 | BB | D2Q5 | ABB | Mixed stencils. Intra has artifact, extra clean. | Same issue as BL3 reversed. |
| BL5 | D2Q5 | BB | D2Q5 | BB (Neumann) | Both Neumann = insulated. No Kleber. | B1 equivalent. Reference case. |
| BL6 | D2Q9 | BB | D2Q9 | BB | Both Neumann = insulated + both have diagonal artifact. | B1 + artifacts. |
| BL7 | D2Q5 | ABB | D2Q5 | ABB | Both Dirichlet. Nonphysical. | B6 equivalent. **WRONG.** |
| BL8 | D2Q5 | BB | D2Q5 | Absorbing | Intra: Neumann. Extra: open boundary. | Absorbing doesn't clamp phi_e to zero; it sets incoming f to local equilibrium. Might approximate a "soft" Dirichlet if phi_e is small. **Untested, potentially interesting.** |

### Collision × stencil × BC cross-products:

For the physically correct case (intra BB + extra ABB), the collision and stencil
choices affect accuracy, not correctness:

| # | Intra collision | Extra collision | Stencil | Artifact level | Anisotropy support |
|---|----------------|----------------|---------|---------------|-------------------|
| BL1a | BGK | BGK | D2Q5 | Zero | Isotropic only |
| BL1b | MRT | BGK | D2Q5 | Zero | Intra anisotropic, extra isotropic |
| BL1c | BGK | MRT | D2Q5 | Zero | Intra isotropic, extra anisotropic |
| BL1d | MRT | MRT | D2Q5 | Zero | Both anisotropic |
| BL2a | BGK | BGK | D2Q9 | O(dx^2) both | Isotropic only |
| BL2b | MRT | MRT | D2Q9 | O(dx^2) both | Full tensor D |

**All BL1x variants produce correct physics.** The choice between them is about
anisotropy support and computational cost, not correctness.

### Coupling mechanism:

Both lattices share the same source term:
```
R = chi * (Cm * dV/dt + I_ion)
```
where V is reconstructed from both lattices: V = sum(f_intra) - sum(f_extra).

The coupling is LOCAL (per-node, per-timestep). No global solve. Each lattice streams
independently. This is the key advantage of bidomain LBM over bidomain FDM.

### Verdict for bidomain LBM:
**BL1 (D2Q5/D2Q5, BB/ABB) is optimal.** Zero artifacts, correct physics, cleanest
implementation. D2Q9 adds nothing for isotropic tissue and introduces artifacts.

For anisotropic tissue (fiber rotation in 3D), BL1d (MRT/MRT D2Q5) handles D_xx ≠ D_yy
but not D_xy. For full tensor, need D2Q9 MRT (BL2b) — accept the O(dx^2) artifact.

---

## 5. Hybrid Approaches

### H1: Monodomain bulk + bidomain boundary layer

Run monodomain in the tissue interior (where bidomain ≈ monodomain) and switch to
bidomain within lambda (~1.4 mm) of any inert boundary.

**Knobs:** boundary layer width, interpolation between mono/bidomain regions.

**Assessment:** Complicated to implement, minimal benefit over M5 (spatially varying D
achieves the same thing more simply) or over full bidomain (which eliminates the
elliptic solve via LBM anyway).

### H2: LBM with per-node source term correction

Standard monodomain LBM (L1) plus an additional source term at boundary-adjacent nodes
that represents the enhanced diffusive current:
```
R_correction(x,y) = (D_i - D_eff) * Laplacian(V) * exp(-d(x,y)/lambda)
```

**Assessment:** Mathematically identical to L18 (spatially varying tau). But numerically
inferior: requires computing Laplacian(V) from the distribution functions at every
timestep, which is noisy. The tau-based approach (L18) encodes the same physics in the
relaxation rate, which is set once at initialisation. **Strictly worse than L18.**

### H3: Monodomain FDM with anisotropic boundary stiffness

Modify the FDM stencil at boundary-adjacent nodes to increase the effective conductance
in the propagation direction. Equivalent to spatially varying D but implemented via
stencil coefficients instead of D field.

**Assessment:** This IS M5, just implemented differently. Same physics, same result.

---

## 6. Complete Ranking

Ranking all physically viable combinations by fidelity to the correct physics:

### Tier 1: Correct physics (captures mechanism, not just magnitude)

| Rank | Combination | What it captures | What it misses | Implementation status |
|------|-------------|-----------------|---------------|---------------------|
| 1 | **B2: Bidomain FDM, Neumann intra + Dirichlet extra** | Full Kleber mechanism. Correct BCs. Correct spatial profile. Correct AP morphology at boundary. Correct parameter dependence (sigma_i, sigma_e, chi, Cm). | Nothing — this IS the correct physics (within continuum bidomain). | **IMPLEMENTED. Validated Phase 6.** |
| 2 | **BL1: Bidomain LBM D2Q5, BB intra + ABB extra** | Same as B2 but via LBM. No elliptic solve. | Possible time-stepping differences (LBM diffusion = parabolic-parabolic, not parabolic-elliptic). | **NOT IMPLEMENTED.** |

### Tier 2: Correct direction, approximate mechanism

| Rank | Combination | What it captures | What it misses | Implementation status |
|------|-------------|-----------------|---------------|---------------------|
| 3 | **L18: LBM D2Q5 BB + spatially varying tau** | Kleber speedup magnitude and spatial profile (exponential decay). Curvature. Source-sink. | AP morphology at boundary. Parameter coupling (D profile must be precomputed analytically, doesn't self-consistently emerge from BCs). | **NOT IMPLEMENTED** (tau is currently uniform). Requires: distance-to-scar computation + D(x,y) field + per-node tau. |
| 4 | **M5: Monodomain FDM Neumann + spatially varying D field** | Same as L18. | Same as L18. | **PARTIALLY IMPLEMENTED** (D field supported in V5.4, but no distance-to-scar module or Kleber D-profile function). |
| 5 | **L20: LBM D2Q9 BB + spatially varying tau** | Same as L18 + O(dx^2) diagonal artifact (~3% at coarse grids). | Same as L18 + artifact. | **NOT IMPLEMENTED.** |

### Tier 3: Captures geometry only (no Kleber)

| Rank | Combination | What it captures | What it misses | Implementation status |
|------|-------------|-----------------|---------------|---------------------|
| 6 | **L1: LBM D2Q5 BB uniform** | Curvature, source-sink, block thresholds, rotor anchoring. Zero artifact. | Kleber speedup entirely (~13% on flat edges). | **IMPLEMENTED.** |
| 7 | **M1: Monodomain FDM Neumann uniform** | Same as L1. | Same as L1. | **IMPLEMENTED.** |
| 8 | **L3: LBM D2Q9 BB uniform** | Curvature, source-sink, etc. + O(dx^2) artifact (~3% slowdown). | Kleber + has wrong-direction artifact. | **IMPLEMENTED.** |

### Tier 4: Wrong physics (all Dirichlet/Robin/hybrid BC on V)

| Rank | Combination | Why wrong |
|------|-------------|-----------|
| — | M2, M3, M6, M7 (FDM Dirichlet/Robin) | Current sink → slowdown |
| — | L5-L10 (LBM ABB/Absorbing) | Current sink → slowdown |
| — | L12-L17 (D2Q9 hybrid BCs) | Partial drain → directional slowdown |
| — | L23-L24 (ABB + varying tau) | Enhancement + sink = competing effects |
| — | B5-B9 (bidomain with intra Dirichlet/Robin) | Nonphysical intra BCs |

---

## 7. Do We Need More Advanced Tools?

### What our current best (B2: Bidomain FDM) gets right:
- Kleber speedup mechanism ✓
- Correct spatial profile ✓
- Correct parameter dependence ✓
- AP morphology at boundary ✓
- Curvature + source-sink ✓

### What even our best tools CANNOT capture:

#### 7A. The parabolic-parabolic vs parabolic-elliptic question

The bidomain admits two formulations:
- **Parabolic-elliptic** (standard): dV/dt = parabolic, phi_e = elliptic (instantaneous)
- **Parabolic-parabolic** (LBM-native): both V and phi_e evolve diffusively

In FDM bidomain, we use parabolic-elliptic. The elliptic solve for phi_e assumes the
extracellular potential adjusts instantaneously — valid because electromagnetic
propagation (speed of light) >> action potential propagation (~1 m/s).

In bidomain LBM, both lattices propagate at the LBM characteristic speed c = dx/dt.
This means phi_e propagates at FINITE speed in the LBM, not instantaneously. For
typical LBM parameters (dx=0.025 cm, dt=0.01 ms), c = 2.5 cm/ms = 25 m/s. Since
AP speed is ~0.05 cm/ms, c/v_AP ~ 500. The LBM "speed of light" is 500x faster
than the AP — probably sufficient, but this is an unvalidated assumption.

**Do we need a tool to validate this?** Yes — compare bidomain FDM (elliptic phi_e)
against bidomain LBM (parabolic phi_e) with identical parameters and measure whether
the Kleber speedup ratio differs.

#### 7B. 3D effects in nominally 2D monolayers

hiPSC monolayers are 3-5 cell layers thick. The bath covers BOTH surfaces (top and
bottom). The bidomain in 2D treats the boundary as a line. In reality:
- Top surface: bath-coupled (phi_e = 0 at z = top)
- Bottom surface: substrate (insulated, Neumann at z = bottom? Or bath if perfused from below?)
- The "2D" effective D depends on the 3D bath coupling geometry

A true 2D monolayer (single cell layer) is correctly modeled by 2D bidomain with
bath BCs. But a 3-5 layer "monolayer" has depth-dependent phi_e that the 2D model
misses.

**Do we need a 3D tool?** Possibly, for quantitative comparison with experiments.
For the ML-DO RANKING question (which geometry is most arrhythmogenic), the 3D
correction is likely a constant scaling factor that doesn't affect relative ranking.

#### 7C. Finite bath conductivity

Our bidomain uses phi_e = 0 (Dirichlet) at the tissue-bath interface. This assumes
the bath conductivity is infinite (or at least >> sigma_e). In reality:

```
sigma_bath ~ 15-20 mS/cm (Tyrode's solution)
sigma_e ~ 6.25 mS/cm (extracellular tissue)
sigma_bath / sigma_e ~ 2.5-3.2
```

The bath conductivity is NOT infinite — it's only ~3x higher than the extracellular.
The correct BC is a continuity condition:
```
sigma_e * d(phi_e)/dn |_tissue = sigma_bath * d(phi_bath)/dn |_bath
phi_e |_tissue = phi_bath |_boundary
```

This requires solving the Laplace equation in the bath (volume conductor problem)
coupled to the tissue bidomain. phi_e at the boundary is NOT zero — it's the
solution of the bath Laplace equation.

For a large bath (bath volume >> tissue volume), phi_bath ≈ 0 far from the tissue,
and phi_e at the boundary is small. But inside a narrow void (e.g., a thin laser-cut
channel), the bath potential may NOT be negligible. The void is a confined space where
return currents from the tissue can elevate phi_bath significantly.

**Do we need a bath solver?** YES, for narrow/confined inert voids. The Dirichlet
phi_e = 0 assumption is good for tissue edges facing large bath volumes but
potentially wrong for narrow laser-cut channels where the bath is geometrically
confined.

This would require:
- A Laplace solver on the void domain (bath region)
- Continuity BC at tissue-bath interface (replaces Dirichlet)
- Solution at every timestep (or quasi-static — bath equilibrates fast)

**Impact on ML-DO:** For scar geometries with narrow channels and isthmuses (exactly
the geometries most likely to support reentry), the confined-bath effect could be
significant. The effective Kleber speedup would be REDUCED in narrow channels compared
to open edges — geometry-dependent correction that could affect rankings.

#### 7D. Discrete gap junction effects at the boundary

The continuum bidomain assumes smooth, homogeneous tissue. At the cellular scale,
gap junctions are discrete — the last row of cells at the inert boundary has coupling
partners missing on the void side. This creates:

1. **Reduced lateral coupling** (fewer gap junction connections → lower effective sigma_i
   in the y-direction at the boundary). The continuum model smooths this.
2. **Discrete source-sink effects**: The last cell is a full cell (not half a cell as the
   continuum boundary implies). Current redistribution at the cellular scale may differ
   from the continuum prediction.
3. **Cell geometry effects**: Cells are elongated (~100 μm × 20 μm). At a boundary
   parallel to the fiber direction, the last row of cells loses side-to-side coupling.
   At a boundary perpendicular, cells lose end-to-end coupling. These have different
   electrophysiological consequences.

**Do we need a discrete model?** For quantitative accuracy at specific boundaries,
possibly. For ML-DO ranking across many geometries, probably not — the discrete effects
are small corrections to the continuum solution.

#### 7E. Electrotonic interaction between nearby boundaries

The Kleber effect extends ~lambda = 1.4 mm into the tissue. If two inert boundaries
are closer than ~2*lambda = 2.8 mm, their Kleber boundary layers overlap. The
spatially varying D approximation (M5/L18) handles this correctly (the exponentials
from each boundary simply add), but the bidomain solution may be more complex
(phi_e boundary layers interact nonlinearly through the elliptic equation).

For narrow isthmuses (width < 2*lambda), the ENTIRE isthmus has enhanced D, not just
the edges. This would significantly affect the CV through the isthmus and the block
threshold.

**Do we need a tool for this?** No — both the bidomain FDM (B2) and the spatially
varying D approximation (M5/L18) handle this. The bidomain handles it exactly; the
monodomain approximation handles it approximately (sum of exponentials instead of
the true nonlinear interaction). The error is small for most geometries.

---

## 8. Conclusions

### The answer to "best combination with current tools":

**Bidomain FDM (B2)** — already implemented and validated. Neumann on intracellular,
Dirichlet on extracellular. This produces the correct physics through the correct
mechanism. All 16 Phase 6 tests pass. Mesh convergence to theoretical 1.131 ratio
confirmed.

For ML-DO at scale (thousands of evaluations), use **monodomain LBM with spatially
varying tau (L18/L19)** — not yet implemented but requires minimal code changes (add
distance-to-scar computation and per-node tau initialisation). This captures ~90% of
the Kleber effect at ~10% of the bidomain cost.

### The answer to "do we need more advanced tools":

**Yes, for two specific scenarios:**

1. **Confined-void bath solver (7C):** For narrow laser-cut channels and isthmuses,
   the Dirichlet phi_e = 0 assumption breaks down. The bath potential inside a narrow
   void is elevated by return currents. This requires coupling a Laplace solver on
   the void domain to the tissue bidomain. This is a tissue-bath coupling problem that
   none of our current tools handle.

   **Impact:** Geometry-dependent. Affects exactly the geometries ML-DO cares about most
   (narrow isthmuses near the reentry threshold). Could change geometry rankings.

2. **Bidomain LBM (BL1) with validation against FDM (B2) (7A):** Before trusting
   bidomain LBM for ML-DO at scale, we need to verify that the parabolic-parabolic
   formulation (LBM-native) matches the parabolic-elliptic formulation (FDM standard).
   The characteristic speed ratio c/v_AP ~ 500 suggests convergence, but this is
   unvalidated.

   **Impact:** If the formulations disagree, bidomain LBM is not trustworthy and we
   must use bidomain FDM (with its expensive elliptic solve) or the monodomain
   approximation (L18).

### What we definitively do NOT need:

- More LBM BC variants (all non-Neumann BCs on V produce wrong physics)
- Mixed/hybrid bounce-back schemes (partial drains don't help)
- Robin BCs on monodomain V (can only drain, never enhance)
- More complex monodomain stencils (the PDE is the limitation, not the discretisation)
- D2Q9 for the Kleber problem (D2Q5 is strictly better — no diagonal artifact, same physics)
- 3D modeling for ranking purposes (3D correction is geometry-independent, doesn't affect ranking)

---

## 9. Decision Matrix for ML-DO Engine Selection

| Criterion | B2 (Bidomain FDM) | L18 (Mono LBM + D(x,y)) | BL1 (Bidomain LBM) |
|-----------|-------------------|------------------------|---------------------|
| Physics correctness | Exact | ~90% (analytical approx) | Exact (if parabolic-parabolic OK) |
| Confined-void accuracy | Needs bath solver | Wrong (assumes phi_e=0) | Needs bath solver |
| Speed per evaluation | Slow (elliptic solve) | Fast (LBM streaming) | Medium (2 lattices, no solve) |
| GPU parallelism | Poor (global solve) | Excellent | Good (2x memory) |
| ML-DO scalability | 100s of evals | 10,000s of evals | 1,000s of evals |
| Implementation effort | Done | Low (add D field + tau) | Medium (new engine) |
| Ranking reliability | High | High (if Kleber ≈ uniform) | High |
| Absolute threshold accuracy | High | Moderate | High |
