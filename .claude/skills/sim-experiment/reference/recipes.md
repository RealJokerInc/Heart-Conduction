# Recipes — map a plain-language goal to cardiac_core parameters

INTERPRET maps the scientist's free-form description to the closest recipe, then fills `run-template.py`.
Each recipe lists the parameters to set and the measurement. Engine rule: **monodomain** unless the
experiment is about the surrounding bath / tissue edge / boundary loading → **bidomain**.

---

## R1 — Conduction velocity in a strip  (DEFAULT)
**Asks like:** "how fast does the signal travel?", "does conduction slow with less coupling / fibrosis?",
"effect of σ / gap-junction reduction on speed?"
- engine `monodomain`; geometry strip (e.g. 2.0 × 0.5 cm, dx=0.01); single stimulus, left edge.
- vary: `SIGMA_I/SIGMA_E` (coupling), `IONIC`.
- measure: `result.cv(x1=round(0.2/DX), x2=round(1.0/DX), y=Ny//2)`.
- `T_END_MS` ≥ time for the front to reach x2 (≈ (x2_cm)/0.05 + margin; 40 ms for ~1 cm).

## R2 — Reentry / spiral wave  (S1–S2)
**Asks like:** "can it form a spiral / reentry?", "does the wave break?", "is this tissue arrhythmogenic?"
- engine `monodomain`; geometry 2-D sheet (e.g. 5 × 5 cm, dx=0.025).
- pacing S1 (plane, left edge) then S2 (a perpendicular/corner region) timed into S1's refractory tail →
  two stimuli in the list, different `start_time` + `region`.
- measure: `cc.phase_singularities(...)` / inspect `result.Vm` for a rotor; longer `T_END_MS` (300–1000 ms),
  finer `SAVE_EVERY_MS` (1–2 ms). Strongly pair with `/sim-media` (video).

## R3 — APD restitution  (regular pacing)
**Asks like:** "how does the action potential change with heart rate / pacing rate?", "restitution curve?"
- engine `monodomain`; small patch (e.g. 0.5 × 0.5 cm) or a point.
- regular pacing: one stimulus with `"bcl"` (cycle length) + `"num_pulses"` (e.g. 8–10 beats), decreasing BCL.
- measure: `result.restitution(ix, iy)` at a central node; `T_END_MS = num_pulses × bcl`.

## R4 — Edge / bath conduction effect  (bidomain)
**Asks like:** "does the tissue edge / surrounding bath speed up conduction?", "boundary loading effect?"
- engine **`cc.bidomain`** (this is the case bidomain exists for); `boundary="bath"`.
- geometry strip; single stimulus; compare edge-row vs centre-row CV.
- measure: `result.cv(...)` at `y=1` (edge) vs `y=Ny//2` (centre).

---

If the description matches no recipe, ask 1–2 clarifying questions to place it, or build the closest
recipe and FLAG the assumption in the manifest. Never invent API — only what's in `API_CHEATSHEET.md`.
