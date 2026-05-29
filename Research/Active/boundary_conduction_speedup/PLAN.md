# PLAN — Diagnose horizontal-redirect edge-row depolarization

> Created 2026-05-28. Cold-start agent-executable. Each step is independent
> and produces one artifact + a one-line text summary.
>
> Previous PLAN.md (2026-04-30 Moore-8 stencil work, fully delivered) is
> archived at `plans/2026-04-30_moore8_stencil_extension.md`.

## Question

Under the **horizontal-redirect** BC (`diag_lbm_specular.py --bc horizontal`),
the top and bottom rows (`j=0`, `j=NY-1`) become a fast "wall channel" of
depolarized cells, AND the rows immediately interior (`j=1`, `j=NY-2`) dip
**below** V_rest (≈ −92 mV). Two known facts:

- `V_sum(t=25 ms)` is 9 % (canonical) to 18 % (uniform_8) higher than the
  HBB baseline → mass leak somewhere (IDEALOG 2026-05-14, cases 13-14).
- Likely culprit: corner cells receive redirect deposits without being
  zeroed out by the non-corner redirect loop, AND HBB still places their
  own bounced mass back at the corner.

Goal: separate **DESIGNED behavior** (wall channel is the intended effect)
from **IMPLEMENTATION ARTIFACT** (mass leak inflating magnitudes) from
**SECONDARY ARTIFACT** (sub-edge dip caused by the leak's gradient).

## Inputs

- Existing HDF5: `data/case13_lbm_d2q9_canonical_horizontal_natural.h5`
  (regenerated 2026-05-28 with col-0 stim).
- Reference HDF5: `data/case10_lbm_d2q9_canonical_hbb_natural.h5` (HBB
  baseline) and `data/case9_lbm_d2q9_canonical_specular_natural.h5`.
- Existing horizontal-redirect implementation:
  `diag_lbm_specular.py::apply_horizontal_redirect_top_bottom_d2q9`
  (lines ~164-210). Donor range `1:NX-1`, dest range `2:NX` and `:NX-2`
  — destinations INCLUDE the corner indices `NX-1` and `0`, but the HBB
  zero-out on those corners does NOT happen → suspected leak site.

## Steps

### Step 1 — Mass-conservation audit

**Artifact**: `figures/horizontal_mass_audit.png` (4 subplots).

**Script**: `diag_horizontal_mass.py` (NEW).

What it does:
1. Load case13 (horizontal) and case10 (HBB) HDF5s.
2. For each time t in saved trajectory, compute:
   - `V_sum(t) = sum over all cells` of `V(t)`
   - `V_sum_corners(t)` (four corner cells only)
   - `V_sum_wall_noncorner(t)` (top + bottom rows excluding corners)
   - `V_sum_interior(t)` (rows j ∈ [1, NY-2])
3. Plot all four as time series, horizontal vs HBB side by side.
4. Compute leak rate `dV_sum/dt` per time step, identify if leak grows
   with wave-front passage at corners.

**Pass/fail**: explicit numerical answer — what % of total V_sum at t=25 ms
is concentrated at the four corner cells vs the rest. If corners dominate
the excess, leak is corner-localized as suspected.

### Step 2 — V(y) profile traces

**Artifact**: `figures/horizontal_vy_profiles.png` (3×4 grid: 3 BCs × 4 times).

**Script**: `diag_horizontal_vyprofile.py` (NEW).

What it does:
1. Load case10 (HBB), case9 (specular), case13 (horizontal).
2. For each BC, plot V vs y at columns 3, 10, 20, 38, at t = 5, 10, 15, 25 ms.
3. On the horizontal panels, mark the j=0 / j=NY-1 (wall channel) and
   j=1 / j=NY-2 (suspected dip) with annotations.
4. Look for: (a) does j=1 dip appear BEFORE or AFTER the wall channel forms?
   (b) is the dip uniform in x or worse near corners?

**Pass/fail**: qualitative — does the dip pattern match a "mass deficit
diffusing inward from over-supplied wall" picture, or does it appear in a
mass-conserving way?

### Step 3 — Counterfactual: mass-conserved horizontal redirect

**Artifact**: `figures/horizontal_fixed_vs_buggy.png` (side-by-side wave
snapshots at t = 5, 15, 25 ms) + `data/case_horiz_fixed_canonical.h5`.

**Script**: edit `diag_lbm_specular.py` to add a new BC mode
`horizontal_fixed`. Mechanism:

Original buggy logic at top wall:
```python
f[7, 1:NX-1, NY-1] = 0                       # zero non-corner HBB diag
f[8, 1:NX-1, NY-1] = 0
f[1, 2:NX,  NY-1] += f_star[5, 1:NX-1, NY-1] # dest 2:NX INCLUDES east corner
f[2, :NX-2, NY-1] += f_star[6, 1:NX-1, NY-1] # dest :NX-2 INCLUDES west corner
```

Fixed logic (corner-aware):
```python
f[7, 1:NX-1, NY-1] = 0                       # same
f[8, 1:NX-1, NY-1] = 0
# DEST excludes corners — they keep their HBB-self-bounce
f[1, 2:NX-1, NY-1] += f_star[5, 1:NX-2, NY-1]  # donor i in [1, NX-3]
f[2, 1:NX-2, NY-1] += f_star[6, 2:NX-1, NY-1]  # donor i in [2, NX-2]
# The orphaned donors (i=NX-2's f_5, i=1's f_6) bounce back at SELF (HBB):
f[7, NX-2, NY-1] = f_star[5, NX-2, NY-1]
f[8, 1,    NY-1] = f_star[6, 1,    NY-1]
# Symmetric for bottom wall.
```

This is mass-conserving: every pre-stream diagonal goes to EXACTLY one
destination (either an adjacent cardinal slot or self via HBB).

Run with `--bc horizontal_fixed --weights canonical` to produce
`case_horiz_fixed_canonical_natural.h5`. Compare to case13 (buggy
horizontal):

- Does the wall channel still appear? Magnitude?
- Does the sub-edge dip persist or vanish?
- Per-column LAT bdry−ctr: still inverse crescent? Same magnitude?

**Pass/fail**: numerical — if wall channel persists with > 50% of the
buggy magnitude AND V_sum is conserved (no growth), the wall channel is
REAL. If wall channel collapses to < 20% magnitude, the buggy version
was leak-driven.

### Step 4 — Diffusion-only horizontal redirect

**Artifact**: `figures/horizontal_diff_vs_ttp06.png` + diag-only HDF5.

**Script**: extend `diag_lbm_specular.py` with `--physics diffusion` flag.

What it does:
1. Replace `R = compute_source_term(I_ion, I_stim, Cm)` with `R = 0`.
2. Keep the IC: V[0, :] = V_STIM = 0 mV (sub-threshold poke).
3. Run 25 ms with the **fixed** horizontal redirect, save trajectory.
4. Compare j=1 / j=NY-2 profile to TTP06 case (case_horiz_fixed) and
   to HBB diffusion-only baseline.

**Pass/fail**: if j=1 dip persists under pure diffusion → it's a
BC/diffusion-coupled artifact, no ionic involvement needed. If it
disappears → TTP06 hyperpolarization at mild depolarization is doing it
(IK1, IKs activation).

### Step 5 — Synthesis

**Artifact**: append a section to `KNOWLEDGE.md` under "Discrete-lattice
boundary effects" with the three-way attribution. Append to `IDEALOG.md`
with a dated entry summarizing the diagnosis.

```
Wall-channel depolarization under horizontal redirect:
  - DESIGNED:   X % of magnitude (from fixed-mass-conserving variant)
  - LEAK:       Y % of magnitude (excess in buggy variant vs fixed)
  - SUB-EDGE DIP:
     * appears under pure-diffusion fixed variant → BC-mechanical artifact
     * vanishes under pure-diffusion fixed variant → ionic-driven
```

## Execution order

Sequential: 1 → 2 → 3 → 4 → 5. Step 3 is the heaviest (needs code edit
+ rerun). Steps 1, 2, 4 are read/plot scripts (~30 s each). Step 5 is
documentation only.

## Out of scope

- Promoting `horizontal_fixed` to default in production code — separate
  decision after diagnosis.
- Re-running the weighted simplex sweep with the fix — depends on outcome.
- 3D extension, anisotropic extension — separate work.
