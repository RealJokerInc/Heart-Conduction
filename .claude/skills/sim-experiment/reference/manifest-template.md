# Manifest template

Render this as PLAIN TEXT and show it to the scientist BEFORE generating/running anything (the
double-check gate). Core fields are required; the *optional* block is included only when the scientist
supplied it. On "go", save this exact text as the experiment's `MANIFEST.md` (the accountability record).

```
EXPERIMENT MANIFEST — please confirm before I run
──────────────────────────────────────────────────
Goal:         {one-line plain-language goal}
Engine:       {monodomain|bidomain|lbm}  ({why — e.g. "single potential, fast"; bidomain only for bath/boundary})
Ionic model:  {ttp06|ord}, {cell type if relevant}
Geometry:     {Lx} × {Ly} cm,  dx = {dx} cm   ({Nx} × {Ny} grid)
Tissue:       σ_i={..}, σ_e={..} mS/cm  ({isotropic|bidomain}; χ={chi}, Cm={Cm})   [or σ_eff for isotropic]
Delivery:     {single | S1-S2 | regular(bcl, n)} stimulus, {region}, t={start} ms, {amp} µA/µF
Sim length:   t_end = {t_end} ms,  dt = {dt} ms,  save every {save_every} ms
Measure:      {cv (x-range, row) | apd | restitution | reentry/rotor}
Outputs:      {printout(s)} {+ propagation video / figures via /sim-media}
Script:       Lab/{date}_{slug}/run.py
{── optional ─────────────────────────────────────
Scientist:    {name/initials}
Hypothesis:   {expected result}
Est. runtime: {rough estimate}
}
──────────────────────────────────────────────────
Confirm, or tell me what to change.
```

**Recompute, don't guess:** `Nx = round(Lx/dx)`, `Ny = round(Ly/dy)`. Flag any assumption you had to make
(e.g. "I assumed a 2 cm strip — change if you meant otherwise"). Never present a manifest you can't build
from `API_CHEATSHEET.md`.
