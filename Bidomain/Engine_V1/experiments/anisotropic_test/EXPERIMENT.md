# Anisotropic Boundary Speedup Test

**Research question**: [boundary_conduction_speedup](../../../Research/Active/boundary_conduction_speedup/README.md)
**Master index**: [MASTER.md](../../../MASTER.md)
**Engine**: Bidomain V1
**Created**: 2026-03-15
**Status**: Complete

## Hypothesis
With 2:1 anisotropic conductivity and lower longitudinal CV, the halved transverse diffusivity should produce a LARGER boundary lead (sharper triangle), not a bowl shape. The eikonal prediction: lead ~ sqrt(D_long) / D_trans.

## Method
- **Engine**: Bidomain V1, decoupled Gauss-Seidel
- **Grid**: 50cm × 8cm, dx=dy=0.05cm, NX=1001, NY=161
- **Ionic model**: TTP06 (EPI)
- **Two configurations**:
  1. Isotropic baseline: D_i=0.001243, D_e=0.004464 (CV ~47 cm/s)
  2. Anisotropic 2:1: D_i_fiber=0.0008, D_i_cross=0.0004, D_e_fiber=0.003, D_e_cross=0.0015 (CV_long ~38 cm/s, Kleber ~1.126)

## Parameters
```yaml
grid:
  Lx: 50.0
  Ly: 8.0
  Nx: 1001
  Ny: 161
  dx: 0.05

simulation:
  dt: 0.01
  t_end: 500.0
  threshold: -30.0
```

## Run
```bash
cd Bidomain/Engine_V1/experiments
python anisotropic_test.py
```

## Results
Anisotropic case produces sharper triangle as predicted by eikonal analysis.

## Conclusion
**Hypothesis confirmed.** Lower transverse diffusivity increases the relative boundary lead, producing a sharper triangle despite lower overall CV.

## Outputs
Saved to `Research/Active/boundary_conduction_speedup/anisotropic_test/`
