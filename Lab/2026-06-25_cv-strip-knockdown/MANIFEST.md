EXPERIMENT MANIFEST — please confirm before I run
──────────────────────────────────────────────────
Goal:         Conduction velocity with reduced cell coupling (Cx43-knockdown analogue)
Engine:       monodomain  (single potential, fast; no bath/boundary effect needed)
Ionic model:  TTP06 (ten Tusscher 2006)
Geometry:     2.0 × 0.5 cm strip,  dx = 0.01 cm   (201 × 51 grid)
Tissue:       σ_i=0.87, σ_e=3.125 mS/cm  (HALF of control — weaker coupling; χ=1400, Cm=1)
Delivery:     single stimulus, left edge (x < 0.05 cm), t=1 ms, −80 µA/µF, 2 ms
Sim length:   t_end = 40 ms,  dt = 0.02 ms (default),  save every 0.5 ms
Measure:      conduction velocity (x = 0.2 → 1.0 cm, mid-row)
Outputs:      CV printout + propagation video / APD figure
Script:       Lab/2026-06-25_cv-strip-knockdown/run.py
──────────────────────────────────────────────────
Confirm, or tell me what to change.

— Confirmed 2026-06-25. Result: CV = 41.0 cm/s (done).
