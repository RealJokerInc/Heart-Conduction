# Gonzalez-Rajal et al. 2018 — Geometry-Dependent Arrhythmias in Excitable Tissues

**Citation**: Published in *Physical Review X*, 2018.
**PMC**: PMC6204347

## Key Contribution
Demonstrated that identical cells in different tissue geometries produce completely different electrical dynamics — regular spiking, alternans, irregular firing, or conduction block — purely from geometry, not ion channel heterogeneity.

## Experimental Setup
- **Cell types**: (1) iOS-HEK cells (engineered excitable HEK cells with NaV1.5 + Kir2.1), (2) hiPSC-derived cardiomyocytes for validation
- **Geometries**:
  - 0D islands: 100, 200, 500 µm squares
  - 1D serpentine tracks: widths 100, 200, 500 µm; lengths up to 7 cm
- **Fabrication**: Microcontact printing of fibronectin on cytophobic polyacrylamide

## Computational Model
- **iOS-HEK model**: Hodgkin-Huxley style with NaV1.5 (m, h gates), Kir2.1 (instantaneous), slow recovery variable j (isradipine state-dependent block)
- **Noble model** (cardiac Purkinje fiber): Standard sodium, potassium, chloride currents
- **Spatial discretization**: 10 µm (one cell length)
- **Numerical method**: Fourth-order centered finite difference for Laplacian

## Key Results
| Geometry | Behavior |
|---|---|
| 0D islands | Regular oscillations only, no alternans |
| 1D near-field (paced end) | Alternans at >3 Hz, irregular at 10 Hz |
| 1D far-field (conducting region) | Suppressed alternans, 2:1 conduction block above 3.85 Hz |
| Serpentine corners | **Second-degree conduction block** from increased electrotonic loading at sharp turns |

## Scaling Relationships
- Alternans transition frequency insensitive to coupling strength: varies <2% over 100-fold g_cxn range
- CV ∝ √g_cxn; action potential length λ ∝ √g_cxn
- Scaling tissue size by factor k while scaling g_cxn by k² preserves overall dynamics
- hiPSC-CM validation: CV = 7.2 cm/s, APD = 360 ms, λ = 2.6 cm

## hiPSC-CM Parameters (for our simulations)
These measured values from real hiPSC-CMs provide validation targets:
- CV = 7.2 cm/s
- APD = 360 ms
- Action potential length λ = 2.6 cm
- Alternans decay length = 535 µm

## Relevance to Our Work
1. Proves geometry-dependent behavior exists in hiPSC-CMs (not just theoretical)
2. Provides experimental validation parameters for our PHAS13 model
3. The serpentine corner conduction block is directly related to our sharp-tip pacemaking (reversed sign: block at outward corners, pacing at inward tips)
4. Demonstrates that 10 µm resolution is sufficient for geometry-dependent dynamics
