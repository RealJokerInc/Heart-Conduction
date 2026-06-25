# cardiac_core — API cheatsheet (for generated simulation scripts)

> **Canonical, maintained reference.** This is the ONLY API source the `/sim-*` skills may generate
> against — never invent signatures. Co-located with the code so it can't drift.
> **After any `cardiac_core` API change, re-run `Lab/_validate/smoke.py`** — it's the canary; if it
> breaks, fix this file.

Everything below is verified against the shipped `cardiac_core` (137 tests green, 2026-06-25). CPU +
`float64` by default; deterministic.

```python
import cardiac_core as cc
```

## 1. Geometry — `cc.Grid`

```python
g = cc.Grid(Nx, Ny, dx, dy=None, *, mask=None, device="cpu")   # dy defaults to dx
g.Nx, g.Ny, g.dx, g.dy        # ints / floats
g.Lx, g.Ly                    # dx*(Nx-1), dy*(Ny-1)  (cm)
g.coordinates                 # (x, y) tensors, shape (Nx, Ny), ij indexing  (x[-1,0] == Lx)
g.n_dof                       # Nx*Ny, or mask.sum()
```
A **strip 2 cm × 0.5 cm at dx=0.01**: `cc.Grid(200, 50, 0.01)`  (Nx = round(Lx/dx); 200×0.01 = 2.0 cm).

Optional `mask` (irregular tissue) — `(Nx, Ny)` bool from a helper:
```python
mask = cc.left_edge_mask(Nx, Ny, dx, width)          # left strip
mask = cc.circle_mask(Nx, Ny, dx, center=(cx,cy), radius=r)
mask = cc.rectangle_mask(Nx, Ny, dx, x0, y0, x1, y1)
mask = cc.annulus_mask(Nx, Ny, dx, center, inner_radius, outer_radius)
```

## 2. Conductivity — `cc.ConductivityConfig`

```python
cond = cc.ConductivityConfig.isotropic(sigma, chi=1400.0, Cm=1.0)         # single effective σ
cond = cc.ConductivityConfig.bidomain(sigma_i, sigma_e, chi=1400.0, Cm=1.0)  # most physical
cond = cc.ConductivityConfig.anisotropic(sigma_l, sigma_t, fiber_angle, chi=1400.0, Cm=1.0)
cond.sigma_eff   # effective conductivity (mS/cm)
cond.D_eff       # derived diffusivity = sigma_eff/(chi*Cm)  (cm^2/ms)
```
> ⚠️ **Units trap.** `sigma*` are **raw conductivities in mS/cm**. The diffusivity `D_eff` is DERIVED —
> do NOT pass a pre-divided `D` as `sigma`. Healthy human ventricle: `sigma_i=1.74, sigma_e=6.25`
> → `sigma_eff≈1.36` → `D_eff≈0.000972 cm²/ms` → CV ≈ 54 cm/s. Lower σ = weaker coupling (fibrosis-like).

## 3. Stimulus — a plain dict (or a list of them)

```python
stim = {
    "region":   lambda x, y: x < 0.05,   # callable (x,y)->bool mask, OR an (Nx,Ny) bool array
    "start_time": 1.0,                    # ms
    "duration":   2.0,                    # ms
    "amplitude": -80.0,                   # µA/µF (negative = depolarizing)
}
```
`x`/`y` in the callable are the grid coordinate tensors (cm). For repeated pacing, add `"bcl"` (ms) and
`"num_pulses"`.

## 4. Construct — pick the engine by calling its factory

```python
sim = cc.monodomain(g, "ttp06", cond, stim)     # fast, single potential — DEFAULT choice
sim = cc.bidomain(g, "ttp06", cond, stim)       # ONLY when bath/boundary/edge effects matter
sim = cc.lbm(g, "ttp06", cond, stim, dt=0.005)  # lattice-Boltzmann (smaller dt)
```
- ionic model: `"ttp06"` (ten Tusscher 2006, default) or `"ord"` (O'Hara–Rudy).
- Engine rule for generation: **monodomain** unless the experiment is about the surrounding bath / tissue
  edge / boundary loading → then **bidomain**.
- Optional kwargs: `dt=0.02`, `splitting="strang"`, `device="cpu"` (mono); `boundary="bath"|"insulated"` (bidomain).

## 5. Run — eager `SimulationResult`

```python
r = sim.run(t_end, save_every=1.0)              # eager: one SimulationResult
for chunk in sim.run(t_end, save_every, batch=50):  # streamed chunks (large runs)
    ...
r.Vm        # (T, Nx, Ny) float64 voltage history
r.times     # (T,) ms
r.dx, r.dy  # spacing
r.phi_e     # (T,Nx,Ny) bidomain only, else None
```

## 6. Measure — hooks on the result (+ `cc.<analysis>`)

```python
cv  = r.cv(x1, x2, y)          # conduction velocity (cm/s) between x-indices x1<x2 at row y
apd = r.apd()                  # (Nx,Ny) APD90 map (ms)
lat = r.lat()                  # (Nx,Ny) local activation-time map (ms)
rst = r.restitution(ix, iy)    # APD restitution at a node (multi-beat run)
# direct analysis (same functions): cc.conduction_velocity(Vm,times,dx,x1,x2,y), cc.apd_map(Vm,times),
# cc.activation_time(Vm,times), cc.restitution_curve(Vm,times,ix,iy), cc.phase_singularities(...)
```
CV indices: choose `x1,x2` well inside the tissue and give the front time to reach `x2`
(front ≈ 50 cm/s = 0.05 cm/ms; 1 cm ≈ 20 ms — set `t_end` accordingly).

## 7. Media (path helper now; standardized figures/video land in Phase 4 `cc.viz`)

```python
from cardiac_core.media import media_path
path = media_path("lab", "videos", "my-experiment-slug")   # dated, sequence-suffixed canonical path
# Phase 4 adds: cc.propagation_video(r, slug), cc.apd_map_figure(r, slug), cc.activation_isochrones(r, slug)
```

## 8. Full example — measure conduction velocity in a strip (the smoke pattern)

```python
import cardiac_core as cc

g    = cc.Grid(200, 50, 0.01)                                  # 2.0 × 0.5 cm strip
cond = cc.ConductivityConfig.bidomain(1.74, 6.25, chi=1400.0)  # healthy ventricle σ
stim = {"region": lambda x, y: x < 0.05, "start_time": 1.0, "duration": 2.0, "amplitude": -80.0}

sim = cc.monodomain(g, "ttp06", cond, stim)
r   = sim.run(t_end=40.0, save_every=0.5)
cv  = r.cv(x1=20, x2=100, y=25)                                # ≈59 cm/s (dx=0.01, Crank-Nicolson/PCG)
print(f"conduction velocity = {cv:.1f} cm/s")
```
