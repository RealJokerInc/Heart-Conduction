# Boundary-effect amplification — experiment groups A and B

**Created** 2026-04-29 (post BC discretization audit + face_mirror flip).
**Goal.** Test the bridge claim: *the boundary acceleration effect is the same
fewer-neighbor mechanism in John's storage-tank and in monodomain face_mirror;
John's tanks amplify it ~150× via one-way + threshold kinetics; monodomain
dilutes it back to invisibility via Fickian + continuous-HH smoothing.*

If the claim is right, we should be able to:
- (Group A) Make the boundary acceleration **visible in monodomain** by
  dialing the upstroke and resolution toward John's regime.
- (Group B) Make the camel-toe **disappear in John's tanks** by dialing the
  pump rule and threshold toward monodomain's regime.

Both groups produce the same artifact morphing continuously between regimes.

---

## Group A — monodomain experiments

Engine: `Monodomain/Engine_V5.4`. All runs use TTP06 EPI unless noted.
Diffusion solver: forward_euler. Stim: line stim `x < 0.05`, uniform-y, t = 0.

For each run, measure LAT(x, y) at threshold V_th = -40 mV with sub-ms save
resolution, then compute LAT_dev(x) = mean_y[LAT_at_top_edge - LAT_at_bulk_y].

Discriminator across A1–A4: run **both face_mirror AND node_mirror_existing**.
- If face shows shift ≈ node shift → fewer-neighbor topology dominates.
- If node shows shift ≈ 2× face → node_mirror's gradient-amplification dominates.
- If both show ≈0 → effect is below numerical noise; need to dial harder.

### A1. Sub-millisecond LAT sampling

**Read first.** `benchmark_uniform_init.py` (use as base).

**Why.** The 50 µs implied LAT shift is 10× below our default 500 µs
save_every. Just sampling faster might surface the existing effect.

**Spec.** Re-run 4-mode benchmark with `SAVE_EVERY = 0.025 ms` (100×
finer). Use `save_every = max(dt, 0.025)` to avoid sub-step interpolation.
Run for t_end = 25 ms, 41×21 grid, dx = 0.025.

**Test.** Plot LAT_dev(x) for face_mirror and node_mirror. Expect:
face shows ≤ 50 µs shift; node shows ≤ 100 µs shift if 2× amplification holds.

**Verify.**
```
python benchmark_uniform_init.py --save_every 0.025 --modes face_mirror,node_mirror_existing
```
Compare max|LAT_dev| values printed in the summary table.

**Risk.** save_every = 0.025 ms with dt = 0.02 ms means saving every 1.25
steps — round to nearest step. Memory: 1000 frames × 41×21 = ~7 MB float64. OK.

**Exit criteria.** A measurable LAT_dev > 0 (above floating-point noise) for
at least one mode, OR a clear "still below resolution at 25 µs" finding that
motivates A2/A3.

---

### A2. Small-domain sweep

**Read first.** `Monodomain/Engine_V5.4/cardiac_sim/.../monodomain.py`.

**Why.** Boundary cells become a larger fraction of total area as domain
shrinks. At Nx = Ny = 5, half the cells are boundary; effect should be
proportionally bigger.

**Spec.** Sweep grid sizes (Nx × Ny) at fixed dx = 0.025:
- 41 × 21 (1.0 × 0.5 cm)  — current baseline
- 21 × 11 (0.5 × 0.25 cm)
- 11 × 5  (0.25 × 0.1 cm)
- 5 × 5   (0.1 × 0.1 cm)

Same line stim `x < 0.05` for all. t_end auto-scaled so wave traverses
domain.

**Test.** LAT_dev(x) at top edge for each size, both face/node modes.
Expect monotonic increase as N shrinks.

**Verify.** Plot domain_size vs max|LAT_dev| on log-log; should give a
clean power law if the effect is geometric.

**Risk.** Below Nx = 5, line-stim "x < 0.05" only excites 1-2 cells, may
not seed a clean wavefront. Switch to `i == 0` indexing if needed.

**Exit criteria.** Power-law fit confirms scaling; at smallest domain,
LAT_dev should be ms-scale and visible by eye on a colormap.

---

### A3. Sharper Na activation (Mitchell-Schaeffer or FHN)

**Read first.** `Monodomain/Engine_V5.4/cardiac_sim/ionic/` (which models
are pluggable).

**Why.** TTP06's Na current has finite-width activation (m_inf is a smooth
sigmoid centered at -39 mV with slope ~1/9 mV). Steeper threshold should
amplify the LAT shift toward John's binary firing.

**Spec.** Use a 2-variable model with adjustable threshold sharpness:
- Mitchell-Schaeffer (V, h gating) — has a step-like m gate; slope is a
  free parameter.
- OR FitzHugh-Nagumo with eps small (sharp upstroke).

Sweep slope parameter k from "soft" (TTP06-like, ~9 mV) to "sharp" (~1 mV).
Run face_mirror only at each k.

**Test.** LAT_dev(x) at top edge vs k. Expect monotonic increase as k
shrinks (sharper).

**Verify.** Plot k vs max|LAT_dev|. Predict near-zero at k = 9, ms-scale
at k → 0.

**Risk.** TTP06 isn't easily k-tunable; may need to hot-patch m_inf or
swap to a simpler model. Mitchell-Schaeffer is cleanest; not sure if V5.4
ships it.

**Exit criteria.** Demonstrated continuous transition from invisible
(k = 9) to visible (k = 1) LAT shift. Validates the "amplification by
threshold sharpness" mechanism.

---

### A4. Combined: small domain × sharp Na

**Why.** A2 and A3 should compose multiplicatively. Use this as the
"smoking gun" demo run.

**Spec.** 11 × 5 grid + Mitchell-Schaeffer with k = 1 mV. face_mirror
only.

**Test.** Visual: plot V(x, y, t) snapshot mid-upstroke — boundary cells
should be visibly ahead of bulk. LAT(x, y) field — clear camel-toe shape.

**Verify.** Side-by-side video face_mirror vs node_mirror at this
extreme regime. Should show camel-toe in BOTH (with 2× larger amplitude
in node_mirror), confirming fewer-neighbor mechanism is shared.

**Risk.** None significant — diagnostic experiment.

**Exit criteria.** A monodomain run that produces a visible camel-toe.
Closes the bridge from continuum to discrete on the monodomain side.

---

## Group B — John's storage-tank experiments

Harness: `simulation/` at repo root (configs.py + experiment.py).
All runs start from the standard camel-toe-producing configuration:
6×6 tanks, one_way pumps, max_pump = 10, reflect_y BC, line stim left edge.

For each B-experiment, measure LAT shape at the top wall and compute the
camel-toe amplitude (LAT[wall] - LAT[bulk_mid] in steps or pump cycles).

### B1. Face-mirror BC swap

**Read first.** `simulation/configs.py` (BC enum), `simulation/experiment.py`
(boundary handling).

**Why.** Direct test of the original "camel-toe is a BC artifact"
hypothesis. If swapping reflect_y → face_mirror eliminates the camel-toe
even at one_way + binary firing, then the BC dominates. If the camel-toe
survives, the topology + pump rule dominate.

**Spec.** Add `face_mirror` BC option to `simulation/`. Implementation:
ghost tank value = boundary tank's own value (mirroring face, not node).
For pump rules: pump from boundary to ghost = pump(V[i,0], V[i,0]) ≡ 0.

Run baseline (reflect_y) and new (face_mirror) at otherwise identical
config. Compare LAT shapes.

**Test.** Visual comparison + numerical: max|LAT_dev| under each BC.

**Verify.** PNG figure with two panels (reflect_y, face_mirror) showing
LAT(x, y).

**Risk.** "reflect_y" in John's harness may already be the
face-mirror equivalent — need to check what the existing implementation
literally does. May produce equivalent results, in which case the camel-
toe is purely from the pump rule + topology. Either outcome is informative.

**Exit criteria.** Definitive answer to: does face_mirror BC alone
eliminate camel-toe in John's tanks?

---

### B2. Threshold sharpness sweep

**Read first.** `simulation/configs.py` (PumpRule and any threshold
parameter).

**Why.** John's binary firing is the discrete analog of an infinitely
sharp Na activation. Softening it should reduce the camel-toe.

**Spec.** Replace the binary "fire if V > V_th" rule with a smooth
sigmoid: `pump_amount = max_pump * sigmoid((V - V_th) / k)`. Sweep k from
0.1 (near-binary) to 10 (near-linear). All other params at baseline.

**Test.** max|LAT_dev| vs k.

**Verify.** Should show monotonic decrease as k grows. At k = 10, expect
camel-toe to be invisible (matching monodomain's behavior).

**Risk.** Sigmoid with k = 0.1 may be too sharp and cause numerical
issues; use k_min = 0.5. Pump amount scaling may need re-tuning at each k.

**Exit criteria.** Continuous interpolation from "binary, camel-toe
visible" to "smooth, camel-toe invisible," confirming threshold
sharpness as one of the two amplification axes.

---

### B3. Bidirectional pumps

**Read first.** `simulation/configs.py` (PumpRule), specifically `one_way`
vs `two_way`.

**Why.** John's one_way rule prevents back-flow, which is what allows
boundary tanks to accumulate. Adding back-flow makes the dynamics Fickian,
which should suppress the camel-toe.

**Spec.** Three runs at otherwise identical config:
- one_way (baseline)
- two_way (Fickian, pump = max_pump * (V_src - V_dst), no gate)
- gradient (one_way with strict V_src > V_dst gate)

**Test.** max|LAT_dev| for each.

**Verify.** Expect: one_way → camel-toe; two_way → flat or near-flat;
gradient → between.

**Risk.** two_way may oscillate; cap pump amount.

**Exit criteria.** Demonstrates that one-way pumping is the second
amplification axis (the first being threshold sharpness).

---

### B4. Cell-count sweep

**Why.** Boundary fraction shrinks as N grows. Camel-toe should fade in
larger grids — same as A2 in reverse.

**Spec.** Run baseline at N = 6, 12, 24, 48 (square or matched aspect).
All other params identical.

**Test.** max|LAT_dev| / cell-count (relative shift) vs N.

**Verify.** Should decay as 1/N or 1/sqrt(N) depending on the geometry.

**Risk.** Larger N takes more pump cycles to traverse; t_end auto-scales.

**Exit criteria.** Power-law decay of camel-toe with N, completing the
bridge: at small N John's effect is visible; at large N it's invisible
even with one-way + binary, just like monodomain at typical sizes.

---

## Cross-group synthesis

After both groups run, produce a single phase-diagram figure:

```
                ↑ threshold sharpness (k → 0 binary, k → ∞ linear)
                │
   visible      │     visible        invisible
   camel-toe    │     camel-toe      camel-toe
   (sharp,      │     (sharp,        (smooth,
    small)      │     large)         large)
                │
   ─────────────┼──────────────────────────────────→
                │                              domain size N
                │
   visible      │     invisible      invisible
   camel-toe    │     camel-toe      camel-toe
   (sharp/      │     (smooth,       (very smooth,
    small N)    │     small N)       large N)
```

John's storage-tank lives in upper-left (sharp + small). Monodomain
TTP06 lives in lower-right (smooth + large). The bridge is the
diagonal — both groups should sweep along it from opposite ends and meet.

## Backlinks

- IDEALOG.md entry "2026-04-29 (cont): The boundary effect is shared..."
- KNOWLEDGE.md (boundary BC discretization section)
- bc_discretization_math.tex
- video_boundary_modes.py (visual baseline)
- benchmark_uniform_init.py (numerical baseline at coarse LAT resolution)
