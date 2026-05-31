# Hyperbolic-Hyperbolic (τ_i ≠ τ_e) Code Analysis

**Scope.** This document isolates the hyperbolic-hyperbolic-specific code paths in the
BeatIt reference implementation (Rossi & Griffith 2017, arXiv:1706.08490). It complements
`HYPERBOLIC_BIDOMAIN_MAPPING.md` (which covers the generic hyperbolic solver) by focusing
on what is *different* when τ_i ≠ τ_e vs τ_i = τ_e > 0 — the regime where **both** the
Vm and φ_e equations carry second-order time dynamics and we get a genuinely dual-evolving
bidomain.

**Sources (all in `code_examples/beatit/`):** `Bidomain.cpp` (1034 L), `Bidomain.hpp`,
`BidomainWithBath.cpp` (1493 L), `BidomainWithBath.hpp`, `ElectroSolver.cpp` (1623 L),
`ElectroSolver.hpp`, `Monowave.cpp`, `Monowave.hpp`. Total: ~6,457 lines. All files are
already extracted — this analysis is pure inspection.

**TL;DR.** There is no separate "hyperbolic-hyperbolic" code path. The enum value
`ParabolicParabolicHyperbolic` is set by default and the single code path for the
hyperbolic bidomain works for all τ_i, τ_e ≥ 0 — including the degenerate cases. The
τ_i ≠ τ_e difference lives in exactly **6 algebraic contributions** that silently vanish
when τ_i = τ_e. Translating to FDM or LBM means re-expressing those 6 contributions;
everything else is shared with the existing parabolic-elliptic solver.

---

## 1. The single branch point (one if-else)

`Bidomain.cpp:248-252` and `BidomainWithBath.cpp:310-314`:

```cpp
M_equationType = EquationType::ParabolicParabolicHyperbolic;   // default
if (tau_e == tau_i)
    M_equationType = EquationType::ParabolicEllipticHyperbolic;
if (tau_i == tau_e && 0 == tau_i)
    M_equationType = EquationType::ParabolicEllipticBidomain;
```

**That enum is set but never switched on** anywhere else in the assembly or RHS. It is
purely diagnostic (for log output). The actual equations are always assembled with the
**ParabolicParabolicHyperbolic** form, and degeneracy is achieved when the coefficients
`(τ_i − τ_e)` multiply to zero.

This means one solver class handles all three regimes:
- `tau_i = tau_e = 0` → standard parabolic-elliptic bidomain
- `tau_i = tau_e > 0` → hyperbolic Vm + elliptic φ_e (telegraph Vm only)
- **`tau_i ≠ tau_e > 0` → hyperbolic Vm + hyperbolic φ_e (our target regime)**

---

## 2. The six PP-hyperbolic-specific contributions

Every place the `τ_i − τ_e` factor appears in BeatIt. Anywhere else, the code is identical
across the three regimes.

### 2.1  Matrix block VeQ — mass contribution (`Bidomain.cpp:680-682`)

```cpp
// Block VeQ : (tau_i-tau_e) / cdt * Cm * M + dt * Ki
Ke(i + n_Q_dofs, i) += (tau_i - tau_e) / cdt * Cm * JxW_qp1[qp] * (phi_qp1[i][qp] * phi_qp1[j][qp]);
Ke(i + n_Q_dofs, j) += cdt * JxW_qp1[qp] * DigradV * dphi_qp1[j][qp];
```

The diagonal (lumped-mass) contribution `(τ_i − τ_e)/cdt · C_m` is the **only** mass term
in the Ve-row that carries the asymmetry. The stiffness part `cdt · K_i` is always
present regardless of τ.

**Effect of τ_i = τ_e:** the mass term vanishes → VeQ reduces to `cdt · K_i` → the system
is (upper triangular in mass) the same 2×2 block as the standard PE solve, just scaled.

**FDM translation:** In our code, `K_i` on Q is `-L_i @ Q` (our L_i is negative-semidefinite).
For τ_i ≠ τ_e, we add the diagonal contribution `(τ_i − τ_e)/cdt · C_m · I` to the VeQ
operator. Because this coefficient is nonzero, the VeVe block alone is no longer sufficient —
the Gauss-Seidel decoupling requires folding this term into the Ve RHS after Q is known.

### 2.2  RHS Ve — ionic current time-derivative source (`Bidomain.cpp:894, 911`)

```cpp
// SBDF2 (line 894):
rhsve =  (tau_e - tau_i) * ( 2 * dIion - dIion_old ) - 0 * (stim_i + stim_e);

// SBDF1 (line 911):
rhsve = (tau_e - tau_i) * dIion - 0 * (stim_i + stim_e);
```

When τ_i = τ_e, the Ve-row ionic source is identically zero (up to stimulus terms, which
are also zero here). This is the source term you would EXPECT from the PDE:
`(τ_i − τ_e)·∂I_ion/∂t` in eq 14 of Rossi & Griffith — with the opposite sign as written
because of BeatIt's "we store in iion −I and in diion −dI" convention (noted at
`Bidomain.cpp:820`).

**Sign convention trap.** The comment at line 820 says ionic values are stored negated;
the rhsve expression uses `(τ_e − τ_i)` (flipped sign) — and then the RHS vector is
multiplied by the lumped mass, which is also absorbed. Net effect: the continuous-PDE
equivalent is `+(τ_i − τ_e)·∂I_ion/∂t`. Trace carefully when porting.

### 2.3  RHS Ve — Q history (old-solution) term (`Bidomain.cpp:898, 915`)

```cpp
// SBDF2:
rhs_oldve = (tau_i - tau_e) / cdt * Cm * ( 4 * Qn - Q_nm1 ) / 3.0;

// SBDF1:
rhs_oldve = (tau_i - tau_e) / cdt * Cm * Qn;
```

This is the Q-dependent history that needs to move to the RHS of the Ve equation. When
τ_i = τ_e it vanishes — the standard elliptic φ_e equation emerges with `rhs_Ve = L_i·V_m^n`
only.

**Implementation note.** This history is multiplied by the *lumped* mass matrix on line
936 (`bidomain_system.rhs->add_vector(bidomain_system.get_vector("old_solution"), bidomain_system.get_matrix("lumped_mass"));`) while the ionic source is multiplied by
the *consistent* mass (line 934). For FDM (identity mass), there's no lumped-vs-consistent
distinction — both become pointwise multiplication by the cell volume.

### 2.4  BidomainWithBath: same six contributions, wrapped in tissue check (`BidomainWithBath.cpp:1213-1256`)

The bath version has identical `(τ_i − τ_e)` expressions but gated by the `n_var == n_dofs`
test (only tissue nodes have a Q variable; bath nodes have only Ve):

```cpp
if (n_var == n_dofs)   // tissue node
{
    ...
    // same rhsve = (tau_e - tau_i) * dIion etc.
    ...
}
// else: bath node, only Ve, no Cattaneo terms
rhsve += -0 * (stim_i + stim_e);
bidomain_system.get_vector("ionic_currents").set(dof_indices_Ve[0], rhsve);
```

### 2.5 Block QVe — NOT asymmetric (`Bidomain.cpp:678`)

```cpp
// Block QVe : Ki   (always Ki, no tau factor)
Ke(i, j + n_Q_dofs) += JxW_qp1[qp] * DigradVe * dphi_qp1[j][qp];
```

`QVe` is always `K_i` (intracellular stiffness). No τ dependence. This is the term that
couples Q to Ve — identical across all three regimes.

### 2.6 Block VeVe — NOT asymmetric (`Bidomain.cpp:687`)

```cpp
// Block VeVe : Kie   (always K_i + K_e, the standard elliptic)
Ke(i + n_Q_dofs, j + n_Q_dofs) += JxW_qp1[qp] * DiegradVe * dphi_qp1[j][qp];
```

The Ve-Ve block is always the bulk conductivity tensor `K_{ie} = K_i + K_e`. No τ
dependence. This is what gets factored by the elliptic (or spectral) preconditioner.

---

## 3. What does NOT change when going to hyperbolic-hyperbolic

All of the following are **identical** across `ParabolicParabolicHyperbolic`,
`ParabolicEllipticHyperbolic`, and `ParabolicEllipticBidomain`:

1. **Block QQ** (`Bidomain.cpp:674`): `(1 + τ_i/cdt)·C_m·M_L + cdt·K_i`. This is the
   "parabolic-like" upper-left operator. It depends on τ_i alone, not on the difference.
2. **Block QVe** (line 678): `K_i`.
3. **Block VeQ stiffness** (line 683): `cdt·K_i`.
4. **Block VeVe** (line 687): `K_{ie}`.
5. **V update** (lines 1018–1029): `V^{n+1} = V^n + dt·Q^{n+1}` (SBDF1) or
   `V^{n+1} = (4/3)V^n − (1/3)V^{n−1} + (2/3)dt·Q^{n+1}` (SBDF2).
6. **`-K_i·V^n` contribution to both Q-row and Ve-row RHS** (lines 952–955):
   `bidomain_system.rhs->add(dof_indices_Q[0], -KiVn)` and similarly for Ve. Identical
   on both rows because the coupling is through `V^n` only.
7. **Ionic current computation** (`ElectroSolver.cpp:1498-1503`): `Iion` and `dIion`
   via the ionic model's `evaluateIonicCurrent()` and `evaluateIonicCurrentTimeDerivative()`.
   Same for all regimes.
8. **Mass matrices, lumped/consistent distinction, matrix closing, boundary-condition
   assembly** (Robin, Dirichlet ground): all regime-independent.
9. **Linear solver call** (`Bidomain.cpp:987`): the full 2×2 block system is handed to
   PETSc KSP; BeatIt does NOT do Gauss-Seidel decoupling. (This is a BeatIt choice, not
   a hyperbolic-specific choice.)

---

## 4. The analytic `dIion/dt` subtlety (matters for us)

`ElectroSolver.cpp:1498-1503`:

```cpp
double Iion = ionicModelPtr->current_scaling() * ionicModelPtr->evaluateIonicCurrent(values, istim, dt);
if (ionicModelPtr->isSecondOrderImplemented())
    dIion = ionicModelPtr->evaluateIonicCurrentTimeDerivative(values, gating_rhs, dt, M_meshSize);
else
    dIion = ionicModelPtr->evaluateIonicCurrentTimeDerivative(values, old_values, dt, M_meshSize);
```

**Critical observation.** `dIion/dt` is not computed as a finite difference of `Iion^n −
Iion^{n−1}`. Each ionic model implements `evaluateIonicCurrentTimeDerivative()` —
typically an **analytic chain-rule** over the gating RHS. For Hodgkin-Huxley-style
currents, `∂I_ion/∂t = Σ_k (∂I_ion/∂g_k)·(∂g_k/∂t)` where `∂g_k/∂t` is the gating RHS
that Rush-Larsen uses.

If the ionic model does NOT implement the second-order variant (`isSecondOrderImplemented()
== false`), BeatIt falls back to a two-point FD over the stored `old_values`. This is
acceptable first-order accuracy but loses the second-order consistency that SBDF2 needs.

**Implications for Bidomain V1 port:**
- Our TTP06 and ORd ionic models don't currently expose `evaluateIonicCurrentTimeDerivative()`.
- **Option A (cheap):** FD fallback `dIion = (Iion^n − Iion^{n−1})/dt`. Works for SBDF1.
- **Option B (correct):** Implement analytic chain-rule in `IonicModel.compute_dIion_dt(V, states, dt)`. Worth it only if SBDF2 is needed; for a first hyperbolic experiment, A is fine.
- **Option C (hybrid):** Analytic for the fast currents (I_Na, I_to) during upstroke where
  `∂I_ion/∂t` is largest; FD for the slow currents. Probably overkill.

---

## 5. Bath coupling — how BeatIt handles τ_i ≠ τ_e at the tissue-bath interface

The bath version (`BidomainWithBath.cpp`) introduces two architectural pieces that are
essential for any boundary-artifact experiment with finite extracellular propagation:

### 5.1 Block-variable detection

```cpp
auto n_var = nn->n_vars(bidomain_system.number());
auto n_dofs = nn->n_dofs(bidomain_system.number());
if (n_var == n_dofs)    // tissue node (has both Q and Ve)
```

`libMesh` supports variables that exist only on specific element blocks (subdomains).
`Q` exists only on tissue elements; `Ve` exists everywhere (tissue + bath). A node in the
bath has `n_var == 1` (just Ve), tissue nodes have `n_var == 2` (Q + Ve).

**FDM translation.** Our grid is structured, so we do this via a **tissue mask** —
a boolean array the same shape as the grid. At bath nodes, we skip the Q equation entirely
and solve only the Ve elliptic (no Cattaneo term because the bath doesn't have an `I_ion`
or an intracellular compartment). At tissue-bath interface nodes, Q lives on the tissue
side only, and the Ve conductivity jumps from (D_i + D_e) to D_bath.

### 5.2 Bath matrix assembly (lines 870-890)

```cpp
if (! in_tissue)   // bath elements
{
    // Only one variable: Ve. Only one block: VeVe.
    DiegradVe = D0e * dphi_qp1[i][qp];   // NB: uses D0e as bath conductivity
    Ke(i, j) += JxW_qp1[qp] * DiegradVe * dphi_qp1[j][qp];
    // No Q, no Cattaneo terms, no mass matrix for the Cattaneo time derivative.
}
```

The bath contributes **only** to the Ve-Ve block. No Q. No hyperbolic terms. This is
exactly what one would expect physically — the bath is a passive conductor with no
excitable membrane.

**Critical insight.** In the dual-evolving hyperbolic-hyperbolic regime, φ_e still solves
an **elliptic** problem over the union of tissue and bath (because the bath has no
∂V_e/∂t). The "dual-evolving" character is internal to the tissue; the bath remains
instantaneous. This is physically correct: the bath is a quasi-static volume conductor
with no membrane capacitance to slow things down.

So the Kleber boundary artifact experiment under the hyperbolic-hyperbolic bidomain
reduces to: **tissue φ_e has finite propagation speed; bath φ_e is instantaneous.** The
interface is where interesting differences from the standard PE bidomain would appear.

### 5.3 Symmetric-operator variant (`BidomainWithBath.cpp:1209, 1218`)

```cpp
if (!M_symmetricOperator)
    rhsq = - Chi * (2*Iion - Iion_old + 2 * tau_i * dIion  - tau_i * dIion_old + istim);
else
    rhsq = - cdt * Chi * (2*Iion - Iion_old + 2 * tau_i * dIion  - tau_i * dIion_old + istim);
```

BeatIt supports a "symmetric" vs "asymmetric" operator mode: the symmetric mode multiplies
the RHS by `cdt` (and presumably scales the matrix rows correspondingly) to produce an
SPD block system. The asymmetric mode is faster but needs a non-SPD solver.

For a first port this is inessential — we can use whichever is convenient for our
spectral/PCG infrastructure. **Recommendation**: asymmetric (faster, simpler).

---

## 6. What's MISSING from BeatIt (gaps to fill for our project)

1. **No LBM implementation of any regime.** BeatIt is pure P1 FEM on libMesh. The LBM–
   Cattaneo correspondence (KNOWLEDGE.md §"LBM-Cattaneo Correspondence") has no reference
   code — we are on our own.
2. **No systematic (τ_i, τ_e) parameter sweep in the code.** The parameters are read once
   from the input file and never varied within a run. No built-in infrastructure for
   sweeps; we'd need a driver script.
3. **No diagnostic for "φ_e finite-propagation frontier".** There is no code that measures
   whether the φ_e field actually has finite support at each timestep — a diagnostic
   crucial for our artifact-resolution hypothesis. We would need to add it (e.g., track
   `max{x : |φ_e(x, t)| > ε}` per timestep and compare to the causality cone `c_s·t`).
4. **No Kleber boundary-speedup test.** The only bidomain test in the paper is virtual
   electrode (Fig 12), not a propagating wave against a bath. We would be first.
5. **No analytic ∂I_ion/∂t for TTP06 or ORd.** Only for simpler ionic models (McKean,
   Aliev-Panfilov, Fenton-Karma). Their `isSecondOrderImplemented()` returns false for
   TTP06, so SBDF2 accuracy relies on the FD fallback for the derivative.
6. **No stability analysis for the hyperbolic-hyperbolic regime.** Rossi & Griffith do a
   linear stability analysis for the monodomain reduction (τ_i = τ_e) only. The stability
   of the τ_i ≠ τ_e case has not been analyzed, even in the paper that introduced it.

---

## 7. Mapping the six PP-specific contributions to our codebase

| # | BeatIt (FEM, libMesh) | Our FDM (Bidomain V1) | Our LBM (dual-lattice target) |
|---|-----------------------|------------------------|------------------------------|
| 1 | `Ke(i+n_Q,i) += (τ_i−τ_e)/cdt · C_m · M_L` | Add diagonal term `(τ_i−τ_e)/cdt · C_m · I` to A_VeQ | **Free** — different lattice relaxations automatically give τ_i ≠ τ_e |
| 2 | `rhsve = (τ_e−τ_i)·dIion` (SBDF1) | Add `(τ_e−τ_i)·dIion` to Ve RHS | Coupling via source terms in the extracellular lattice |
| 3 | `rhsve += (τ_e−τ_i)·(2dI^n − dI^{n−1})` (SBDF2) | Extend history to SBDF2 | TBD — LBM is naturally first-order in time; SBDF2 may not be needed |
| 4 | `rhs_oldve = (τ_i−τ_e)/cdt · C_m · Q^n` (SBDF1) | Add term to Ve RHS using Q^n | Handled implicitly in lattice evolution |
| 5 | `rhs_oldve = (τ_i−τ_e)/cdt · C_m · (4Q^n − Q^{n−1})/3` (SBDF2) | Extend with Q^{n−1} history | TBD |
| 6 | Bath detection via `n_var == n_dofs` | Tissue mask array | LBM "tissue lattice" distinct from "bath lattice", or excluded nodes |

---

## 8. Recommended implementation sequence

Progression in complexity. Each step is independently testable against the previous.

1. **Validate parity with PE** — port the **hyperbolic-hyperbolic** assembly from
   Bidomain.cpp with `τ_i = τ_e = 0`. Must match current Bidomain V1 PE output bit-for-bit
   (within SBDF1's inherent accuracy). This retires the "does the scaffolding work" risk
   without touching any new physics.
2. **τ_i = τ_e > 0 (Wave-Vm only)** — enable the telegraph Vm; keep φ_e elliptic.
   Reproduces Rossi & Griffith's monodomain hyperbolic results. Sanity check: CV enhancement
   at τ ≈ 0.1–0.2 ms with TTP06.
3. **τ_i ≠ τ_e** — the target regime. Flip on the six `(τ_i − τ_e)` contributions.
   Compare Kleber-boundary wavefronts against standard PE. Expected signature: smoother
   curvature, longer rise time in φ_e at the tissue-bath interface.
4. **Bath coupling with τ_i ≠ τ_e** — extend to the BidomainWithBath case. Same
   tissue-interior physics, plus the quasi-static bath. This is the Kleber-speedup-with-
   finite-propagation experiment.
5. **LBM dual-lattice analogue** — implement the LBM equivalent using the correspondence
   in KNOWLEDGE.md §"LBM-Cattaneo Correspondence". This is the "novel numerical method"
   of our research program — not directly derivable from BeatIt, but the RHS/source-term
   structure of the six PP-hyperbolic contributions is the same.

---

## 9. Key file/line cross-reference

| What | Where | Lines |
|------|-------|-------|
| Enum, regime selection | `Bidomain.cpp` | 248-252 |
| Matrix assembly (tissue) | `Bidomain.cpp` | 652-695 |
| Matrix assembly (bath) | `BidomainWithBath.cpp` | 870-890 |
| RHS SBDF1 (tissue) | `Bidomain.cpp` | 904-920 |
| RHS SBDF2 (tissue) | `Bidomain.cpp` | 879-903 |
| RHS SBDF1/2 (bath) | `BidomainWithBath.cpp` | 1203-1272 |
| Symmetric-operator variant | `BidomainWithBath.cpp` | 1209, 1218 |
| Linear solve + V update | `Bidomain.cpp` | 967-1032 |
| Ionic current + dIion analytic | `ElectroSolver.cpp` | 1498-1520 |
| dIion FD fallback | `ElectroSolver.cpp` | 1503 |
| SBDF2 gate update | `ElectroSolver.cpp` | 1480-1494 |
| Equation type enum definition | `ElectroSolver.hpp` | 59-63 |
| Ground-node handling | `Bidomain.cpp` | 722-729 |
| Robin BC assembly | `BidomainWithBath.cpp` | 893-934 |

---

## 10. One-sentence summary

The hyperbolic-hyperbolic bidomain is not a new algorithm — it is the **same algorithm**
as the hyperbolic Vm + elliptic φ_e solver, with six additional `(τ_i − τ_e)`-scaled
terms that carry the φ_e time dynamics; everything else (matrix structure, ionic current
handling, bath coupling, V update) is unchanged.
