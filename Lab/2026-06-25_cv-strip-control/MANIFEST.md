EXPERIMENT MANIFEST — please confirm before I run
──────────────────────────────────────────────────
Goal:         Conduction velocity in a healthy ventricular strip (control)
Engine:       monodomain  (single potential, fast; no bath/boundary effect needed)
Ionic model:  TTP06 (ten Tusscher 2006)
Geometry:     2.0 × 0.5 cm strip,  dx = 0.01 cm   (200 × 50 grid)
Tissue:       σ_i=1.74, σ_e=6.25 mS/cm  (bidomain conductivity; χ=1400, Cm=1)  → D_eff≈0.000972 cm²/ms
Delivery:     single stimulus, left edge (x < 0.05 cm), t=1 ms, −80 µA/µF, 2 ms
Sim length:   t_end = 40 ms,  dt = 0.02 ms (default),  save every 0.5 ms
Measure:      conduction velocity (x = 0.2 → 1.0 cm, mid-row)
Outputs:      CV printout (+ propagation video via /sim-media)
Script:       Lab/2026-06-25_cv-strip-control/run.py
──────────────────────────────────────────────────
Confirm, or tell me what to change.

— Confirmed 2026-06-25. Result: CV = 59.3 cm/s (done).
