# Lab Notebook — simulation experiments

Master log. `/sim-experiment` appends one row per experiment (on "go"). `/sim-notebook` rebuilds/curates
this table from the per-experiment `MANIFEST.md` files (the source of truth) — don't hand-edit rows;
edit the manifest and re-run `/sim-notebook index`.

| Date | Experiment | Goal | Engine | Status | Result |
|------|-----------|------|--------|--------|--------|
| 2026-06-25 | cv-strip-control | Conduction velocity in a healthy ventricular strip (control) | monodomain | done | CV = 59.3 cm/s |
| 2026-06-25 | cv-strip-knockdown | Conduction velocity with reduced cell coupling (Cx43-knockdown analogue) | monodomain | done | CV = 41.0 cm/s |

## Comparison — CV strip: control vs knockdown (`/sim-notebook compare`)

| experiment | σ_i (mS/cm) | σ_e (mS/cm) | CV (cm/s) |
|---|---|---|---|
| cv-strip-control   | 1.74 | 6.25  | 59.3 |
| cv-strip-knockdown | 0.87 | 3.125 | 41.0 |

Halving coupling slows conduction ~31% (eikonal CV ∝ √D; √0.5 ≈ 0.71).
