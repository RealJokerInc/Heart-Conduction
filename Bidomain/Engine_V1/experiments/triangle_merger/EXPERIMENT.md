# Triangle Merger — Kleber Boundary Speedup Wavefront Characterization

**Research question**: [boundary_conduction_speedup](../../../Research/Active/boundary_conduction_speedup/README.md)
**Master index**: [MASTER.md](../../../MASTER.md)
**Engine**: Bidomain V1
**Created**: 2026-03-14
**Status**: Complete

## Hypothesis
Bath-coupled (Dirichlet) boundaries produce faster CV at tissue edges than insulated (Neumann) boundaries, creating a triangular wavefront "merger" pattern. The Mehrstellen 9-point stencil should produce a sharper triangle than the standard 5-point stencil due to reduced numerical dispersion.

## Method
- **Engine**: Bidomain V1, decoupled Gauss-Seidel
- **Grid**: 50cm × 8cm, dx=dy=0.05cm, NX=1001, NY=161
- **Ionic model**: TTP06 (EPI)
- **Conductivity**: sigma_i=1.74, sigma_e=6.25 mS/cm, chi=1400, Cm=1.0
- **Three configurations**:
  1. Monodomain Mehrstellen — Neumann BCs (flat wavefront reference)
  2. Bidomain 5-point — bath_tb BCs (baseline comparison)
  3. Bidomain Mehrstellen — bath_tb BCs (primary result)
- **Measurement**: Activation time maps, wavefront profiles at 200ms intervals

## Parameters
```yaml
grid:
  Lx: 50.0
  Ly: 8.0
  Nx: 1001
  Ny: 161
  dx: 0.05

conductivity:
  sigma_i: 1.74
  sigma_e: 6.25
  chi: 1400.0
  Cm: 1.0

simulation:
  dt: 0.01
  t_end: 800.0
  save_every: 25.0
  threshold: -30.0
```

## Run
```bash
cd Bidomain/Engine_V1/experiments
python triangle_merger.py          # Full run (~45 min)
python run_pipeline.py --quick     # Quick validation (~1 min)
```

## Results
- Monodomain Mehrstellen: flat wavefront (no boundary speedup), CV ~54 cm/s
- Bidomain 5-point: triangular wavefront, boundary CV faster by ~7%
- Bidomain Mehrstellen: sharper triangle, boundary CV faster by ~7%, cleaner isochrones

## Conclusion
**Hypothesis confirmed.** Bath-coupled boundaries produce a clear triangular wavefront merger. Mehrstellen stencil produces visually sharper results than 5-point with the same boundary CV ratio.

## Outputs
Saved to `Research/Active/boundary_conduction_speedup/triangle_merger/`:
- `bidomain_5pt_*.pt` — Vm snapshots and activation times (5-point stencil)
- `bidomain_mehrstellen_*.pt` — Vm snapshots and activation times (Mehrstellen)
- `monodomain_mehrstellen_*.pt` — Vm snapshots and activation times (reference)
- `*_times.json` — Timing data for each configuration
