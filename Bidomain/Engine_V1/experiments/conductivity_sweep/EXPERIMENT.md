# Conductivity Sweep — Edge Lead Scaling

**Research question**: [boundary_conduction_speedup](../../../Research/Active/boundary_conduction_speedup/README.md)
**Master index**: [MASTER.md](../../../MASTER.md)
**Engine**: Bidomain V1
**Created**: 2026-03-15
**Status**: Complete

## Hypothesis
Steady-state edge lead scales as ~1/sqrt(D_eff) for isotropic conductivity scaling. Increasing sigma_i only (not sigma_e) changes the Kleber ratio and produces a disproportionately larger lead.

## Method
- **Engine**: Bidomain V1, Mehrstellen stencil, bath_tb BCs
- **Grid**: 50cm × 8cm, dx=dy=0.05cm
- **Five configurations**:
  1. 0.5x isotropic: sigma scaled 0.5x → slower CV, same Kleber ratio
  2. 1x isotropic: baseline
  3. 2x isotropic: sigma scaled 2x → faster CV, same Kleber ratio
  4. 4x isotropic: sigma scaled 4x → fastest CV, same Kleber ratio
  5. 4x sigma_i only: sigma_i 4x → faster CV, HIGHER Kleber ratio

## Parameters
```yaml
grid:
  Lx: 50.0
  Ly: 8.0
  Nx: 1001
  Ny: 161
  dx: 0.05

conductivity_base:
  sigma_i: 1.74
  sigma_e: 6.25

simulation:
  dt: 0.01
  t_end: 500.0
  threshold: -30.0
```

## Run
```bash
cd Bidomain/Engine_V1/experiments
python conductivity_sweep.py
```

## Results
Isotropic scaling confirms 1/sqrt(D_eff) relationship. The 4x sigma_i case shows a larger Kleber ratio and disproportionate edge lead.

## Conclusion
**Hypothesis confirmed.** Edge lead scales with 1/sqrt(D_eff) for isotropic changes. Changing sigma_i alone increases the Kleber ratio (sqrt((D_i+D_e)/D_e)), producing a larger boundary speedup effect.

## Outputs
Saved to `Research/Active/boundary_conduction_speedup/conductivity_sweep/`
