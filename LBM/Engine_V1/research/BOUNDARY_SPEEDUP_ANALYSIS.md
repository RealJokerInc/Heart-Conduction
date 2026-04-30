# Boundary Speedup Effect — Theoretical Analysis

## Summary

The Kleber boundary speedup (increased CV at tissue edges) is a **bidomain phenomenon**
caused by asymmetric boundary conditions on the intracellular and extracellular domains.
It **cannot** be captured by the monodomain equation with any boundary condition, because
the monodomain lumps both domains into one. However, it can be:

1. **Exactly captured** by a bidomain simulation with tissue-bath BCs
2. **Approximated in monodomain** via spatially varying diffusion coefficient D(x,y)

**Important caveat:** A D2Q9 LBM with Neumann (bounce-back) BC on the monodomain equation
will produce a small O(dx^2) artifact at boundaries due to diagonal distribution bounce-back.
This artifact is a numerical discretization error — it is NOT the Kleber boundary speedup
and does NOT represent the correct physiology. The Kleber effect arises from intracellular/
extracellular boundary condition asymmetry, which requires bidomain equations. Any apparent
boundary CV variation from a monodomain D2Q9 simulation with uniform D is a lattice artifact
that vanishes with mesh refinement and should not be interpreted as physiological boundary
speedup. See Section 8 for the full analysis.

This document derives the correct approaches and provides the experimental parameters.

---

## 1. The Physics of Kleber's Boundary Speedup

At a tissue-bath interface:
- **Intracellular** space terminates at the tissue edge (no-flux)
- **Extracellular** space is continuous with the bath (low-resistance volume conductor)

The bath "shorts out" the extracellular resistance near the boundary. With the
extracellular return path made easy, only intracellular resistance limits current
flow. Less total resistance -> more diffusive current -> faster wavefront.

### Quantitative Prediction

Interior effective conductivity (standard monodomain):
```
sigma_eff = sigma_i * sigma_e / (sigma_i + sigma_e)
```

Boundary effective conductivity (bath shorts extracellular):
```
sigma_boundary = sigma_i
```

Since sigma_i > sigma_eff (harmonic mean is always less than either component):
```
CV_boundary / CV_interior = sqrt(sigma_i / sigma_eff)
                          = sqrt((sigma_i + sigma_e) / sigma_e)
```

For human ventricular tissue (longitudinal):
```
sigma_i = 1.74 mS/cm
sigma_e = 6.25 mS/cm
Ratio   = sqrt(1 + 1.74/6.25) = sqrt(1.278) = 1.13
```

**Predicted: ~13% CV increase at tissue-bath boundary.** This matches experimental data.

---

## 2. Bidomain FDM: The Exact Approach

### Governing Equations (parabolic-elliptic formulation)

```
Parabolic:  chi*Cm*dV/dt = div(sigma_i*grad(V)) + div(sigma_i*grad(phi_e)) - chi*I_ion
Elliptic:   div((sigma_i + sigma_e)*grad(phi_e)) = -div(sigma_i*grad(V))
```

### Boundary Conditions at Tissue-Bath Interface

| Domain | BC type | Condition | Physical meaning |
|--------|---------|-----------|-----------------|
| Intracellular | Neumann | n*sigma_i*grad(phi_i) = 0 | Cell network terminates |
| Extracellular | Dirichlet | phi_e = 0 | Continuous with bath |
| V (derived) | Modified Neumann | dV/dn = -d(phi_e)/dn | From intracellular no-flux |

**Critical:** The BC on V is NOT the standard monodomain dV/dn = 0. It is:
```
dV/dn = -d(phi_e)/dn
```
This is nonzero and time-varying, determined by the elliptic solve. During wave
propagation, extracellular current flows into the bath (d(phi_e)/dn != 0),
so V has a nonzero normal gradient at the boundary.

This is precisely why monodomain (which always uses dV/dn = 0) cannot capture the effect.

### How It Works

Consider a strip, wavefront propagating in x. The elliptic equation for phi_e has
a particular solution (y-independent):
```
phi_e_particular = -sigma_i / (sigma_i + sigma_e) * V(x)
```

The Dirichlet BC (phi_e = 0 at bath) requires a homogeneous correction that decays
exponentially from the walls into the interior. The result:

- **Interior** (y >> lambda): phi_e ~ -sigma_i/(sigma_i+sigma_e) * V
  -> Parabolic equation reduces to: sigma_eff * laplacian(V) = chi*(Cm*dV/dt + I_ion)
  -> Standard monodomain with sigma_eff

- **Near boundary** (y ~ 0): phi_e ~ 0 (forced by bath BC)
  -> Parabolic equation reduces to: sigma_i * laplacian(V) = chi*(Cm*dV/dt + I_ion)
  -> Enhanced diffusion with sigma_i > sigma_eff

The transition occurs over the electrotonic space constant lambda ~ 0.5-1 mm.

### FDM Algorithm (per timestep)

```
1. Solve elliptic for phi_e:
     (sigma_i + sigma_e) * laplacian(phi_e) = -sigma_i * laplacian(V)
     BC: phi_e = 0 at bath, Neumann elsewhere
     Method: CG, multigrid, or direct solve

2. Compute coupling term: div(sigma_i * grad(phi_e)) via FDM stencil

3. Advance V (explicit or semi-implicit):
     chi*Cm*(V_new - V)/dt = sigma_i*laplacian(V) + coupling_term - chi*I_ion
     BC: dV/dn = -d(phi_e)/dn at bath

4. Advance ionic states (Rush-Larsen)
```

The elliptic solve (step 1) is the expensive part -- a global linear system at every
timestep. This is why bidomain costs ~10x more than monodomain.

---

## 3. Monodomain Approximation: Spatially Varying D

Since the bidomain analysis shows the effective conductivity transitions from sigma_eff
(interior) to sigma_i (boundary), we can approximate this in monodomain by making D
position-dependent.

### Diffusion Profile

```python
D_interior = sigma_i * sigma_e / ((sigma_i + sigma_e) * chi * Cm)
D_boundary = sigma_i / (chi * Cm)

# Smooth transition over electrotonic space constant
d = distance_to_nearest_boundary(x, y)
lam = sqrt(D_interior / G_m_rest)   # ~ 0.5 mm for cardiac tissue
D(x, y) = D_interior + (D_boundary - D_interior) * exp(-d / lam)
```

Enhancement factor:
```
D_boundary / D_interior = (sigma_i + sigma_e) / sigma_e = 1 + sigma_i/sigma_e
```

For typical values: 1.28 (28% higher D, giving 13% higher CV).

### BC: Standard Neumann (bounce-back in LBM)

The physically correct monodomain BC is no-flux. The spatially varying D handles
the boundary effect. No special BC needed.

### Why LBM Is Good at This

In FDM with spatially varying D, the Laplacian becomes div(D*grad(V)), requiring
modified stencils at every node where D varies. In LBM with MRT, each node simply
gets its own relaxation rate tau from its local D:
```
tau(x,y) = 0.5 + D(x,y) * dt / (cs2 * dx^2)
```
No stencil modification. The collision operator is the same everywhere; only the
relaxation rates change.

---

## 4. Why the Original Hypothesis (Dirichlet BC on V) Is Wrong

The original plan was: D2Q9 + Dirichlet BC (V = V_rest at boundary) -> speedup.

This is incorrect because:

1. **Dirichlet V = V_rest acts as a current SINK.** During depolarization (V > V_rest),
   the boundary drains current from adjacent nodes. In LBM anti-bounce-back:
   ```
   f_opp = -f_i* + 2*w_i*V_rest
   ```
   The returning distributions carry less energy than sent out. Net: current drain.

2. **A current sink INCREASES loading, not decreases it.** The boundary-adjacent cell
   must charge itself AND supply current to the clamped boundary. This SLOWS conduction.

3. **The Kleber effect is about reduced intracellular loading, not fixed-voltage boundaries.**
   The bath doesn't clamp V at V_rest -- it clamps phi_e at phi_bath. These are different
   variables (V = phi_i - phi_e). Clamping phi_e reduces extracellular resistance, while
   clamping V creates an artificial current sink.

4. **For a planar wave with Neumann BC in monodomain:** V is uniform in y, the boundary
   is invisible, CV is the same everywhere. No mechanism for speedup exists.

---

## 5. Experimental Parameters

### Tissue Parameters (human ventricular)

```python
# Conductivities
sigma_i_long  = 1.74    # mS/cm (intracellular, fiber direction)
sigma_i_trans = 0.19    # mS/cm (intracellular, cross-fiber)
sigma_e_long  = 6.25    # mS/cm (extracellular, fiber direction)
sigma_e_trans = 2.36    # mS/cm (extracellular, cross-fiber)

# Membrane properties
chi = 1400.0    # cm^-1 (surface-to-volume ratio)
Cm  = 1.0       # uF/cm^2 (membrane capacitance)

# Derived monodomain diffusion (longitudinal)
D_interior = sigma_i_long * sigma_e_long / ((sigma_i_long + sigma_e_long) * chi * Cm)
           = 1.74 * 6.25 / (7.99 * 1400 * 1.0)
           = 0.000972 cm^2/ms

D_boundary = sigma_i_long / (chi * Cm)
           = 1.74 / (1400 * 1.0)
           = 0.00124 cm^2/ms

# Enhancement
D_boundary / D_interior = (sigma_i + sigma_e) / sigma_e = 7.99 / 6.25 = 1.278

# Predicted CV ratio
CV_ratio = sqrt(1.278) = 1.131  # ~13% speedup

# Space constant
G_m_rest = 0.05  # mS/cm^2 (resting membrane conductance, approximate)
lam = sqrt(D_interior / (G_m_rest / Cm))  # = sqrt(0.000972/0.05) = 0.139 cm = 1.39 mm
```

### Grid Parameters

```python
dx = 0.025   # cm (250 um)
dt = 0.01    # ms
Nx = 200     # nodes in propagation direction (5 cm)
Ny = 80      # nodes in transverse direction (2 cm, enough for W >> lambda)

# Space constant in grid units: lambda/dx = 1.39/0.025 = 55 nodes
# The transition zone is well resolved with this grid
```

### Experimental Configs

| Config | D profile | BC (monodomain) | Expected CV_boundary/CV_interior |
|--------|-----------|-----------------|----------------------------------|
| A | Uniform D_eff | Neumann | 1.00 (no effect) |
| B | Enhanced D near boundary | Neumann | ~1.13 (Kleber approx) |
| C | Uniform D_eff | Dirichlet V_rest | < 1.00 (slowdown) |
| D | Bidomain FDM | Intra-Neumann + Extra-Dirichlet | ~1.13 (exact Kleber) |

---

## 6. Implications for LBM_V1 Architecture

The spatially varying D approach requires:

1. **Per-node relaxation rates** -- already supported by MRT collision. Each node's
   tau is computed from its local D(x,y) during initialization.

2. **A D-field initialization function** -- takes the domain geometry and
   sigma_i, sigma_e values, computes D(x,y) with the exponential transition.

3. **Standard Neumann BC** (bounce-back) -- no need for Dirichlet or special BCs.

4. **D2Q9 vs D2Q5 comparison** -- both should give the speedup (it's a D-field
   effect, not a lattice effect), but D2Q9 should resolve the transition zone
   more accurately due to better isotropy and diagonal gradient resolution.

### New files needed:
- `src/diffusion.py` gains: `create_boundary_enhanced_D_field(Nx, Ny, dx, sigma_i, sigma_e, chi, Cm)`
- Experiment configs updated to use spatially varying tau

---

## 7. Bidomain LBM: Future Extension

The most principled approach would be a full bidomain LBM:
- Two LBM lattices: one for phi_i (or V), one for phi_e
- Intracellular lattice: Neumann BC at tissue edge
- Extracellular lattice: Dirichlet BC at tissue-bath boundary (phi_e = 0)
- Coupled through transmembrane current: R = chi*(Cm*dV/dt + I_ion)

This captures the exact physics without approximation. The LBM is well-suited because:
- Each domain has its own distribution functions
- The coupling is through the source term (already implemented as source-in-collision)
- No elliptic solve needed (LBM handles diffusion through local collision+streaming)

The elimination of the global elliptic solve is a major advantage of LBM over FDM
for the bidomain. In FDM, the elliptic equation requires a full linear system solve
at every timestep. In LBM, the equivalent information propagates naturally through
the streaming step -- one lattice site per timestep.

---

## 8. D2Q9 Bounce-Back Artifact Analysis

Does D2Q9 LBM with Neumann (bounce-back) BC on a monodomain equation produce any
boundary speedup, even as a lattice artifact?

### D2Q5: Zero Artifact

At a flat wall, the only bounced distribution (f_south -> f_north) has velocity (0, -1).
For a planar wave with dV/dy = 0, this distribution is at equilibrium (f = w*V).
Bounce-back returns exactly the correct value. The planar wave remains exactly planar.
**No speedup, no slowdown, zero artifact.**

### D2Q9: O(dx^2) Artifact

The diagonal distributions (SW, SE) bounce to (NE, NW). These have x-velocity components,
so their non-equilibrium parts are nonzero even for a planar wave (because dV/dx != 0):

```
f_7^(1)(x) = +(tau-0.5)*dt*w/cs2 * dV/dx    [SW, e=(-1,-1)]
f_5^(1)(x) = -(tau-0.5)*dt*w/cs2 * dV/dx    [NE, e=(+1,+1)]
```

Bounce-back sets f_5(x,0) = f_7*(x,0), but interior streaming would provide f_5*(x-1,0).
These differ by:
```
delta_f = w*[V(x) - V(x-1)] + (1-1/tau)*(tau-0.5)*dt*w/cs2 * [dV/dx|_x + dV/dx|_{x-1}]
```

The two diagonal errors (f_5 and f_6) partially cancel in the voltage sum, leaving:
```
V(x, 0) - V(x, interior) ~ -w*dx^2 * d^2V/dx^2 = O(dx^2)
```

### Sign and Magnitude

At the wavefront rising edge: d^2V/dx^2 > 0, so V_boundary < V_interior.
The boundary activates LATER -> tiny SLOWDOWN (~0.1-0.5%).

This is a second-order lattice artifact, consistent with bounce-back's known O(dx^2)
accuracy on flat walls. It vanishes with mesh refinement and has no physical meaning.

### Conclusion

Monodomain + Neumann BC cannot produce boundary speedup in either D2Q5 or D2Q9.
The monodomain PDE admits a 1D solution for planar waves — any consistent discretization
must converge to this. Deviations are numerical artifacts, not physics.

---

## References

- Kleber & Rudy, Physiol Rev 84(2), 2004 -- boundary speedup in cardiac tissue
- Bhatt et al. -- experimental measurements of boundary CV enhancement
- Roth, J Math Biol 1991 -- bidomain boundary effects analysis
- Henriquez, Crit Rev Biomed Eng 1993 -- monodomain vs bidomain comparison
