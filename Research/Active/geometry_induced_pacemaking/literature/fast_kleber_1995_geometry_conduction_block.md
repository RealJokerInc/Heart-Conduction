# Fast & Kléber 1995 — Geometry as Determinant of Conduction Block

**Citation**: Fast VG, Kléber AG. "Cardiac tissue geometry as a determinant of unidirectional conduction block: assessment of microscopic excitation spread by optical mapping in patterned cell cultures and in a computer model." *Cardiovascular Research* 29(5):697-707, 1995.
**PubMed**: 7606760

## Key Contribution
First systematic demonstration that tissue geometry alone (strand width at an abrupt expansion) determines unidirectional conduction block, with exact critical dimensions.

## Experimental Setup
- **Cells**: Neonatal rat heart cells in patterned cultures
- **Geometry**: Narrow strands of variable width emerging into a large cell area (abrupt expansion)
- **Measurement**: Voltage-sensitive dye RH-237, linear array of 10 photodiodes, 15 µm spatial resolution

## Critical Dimensions
- **Block**: Strand width = 15 ± 4 µm (1-2 cells wide, n=7)
- **No block**: Strand width = 31 ± 8 µm (n=9, p<0.001)
- **Mechanism**: Current disperses at expansion, reducing current density below activation threshold

## Computational Extension (Wang & Rudy 1995)
- **2D critical strand diameter**: 175-200 µm (varies by ionic model: Ebihara-Johnson, Beeler-Reuter, Luo-Rudy)
- **3D critical strand diameter**: 472-540 µm (2.7× larger than 2D)
- Block explained by both impedance mismatch at transition AND critical curvature beyond transition

## Dual Mechanism of Block
1. **Impedance mismatch**: Current from small source insufficient to charge large downstream capacitance
2. **Critical curvature**: Expanding wavefront has curvature exceeding critical value κ_c = c₀/D

## Key Insight for Pacemaking
This is the **reverse** of the pacemaking effect: strand-to-expansion blocks conduction because the source is too small for the sink. But a **pointed tip** with gradual expansion *facilitates* pacemaking by reducing the sink seen by spontaneously-beating cells.

## Relevance to Our Work
Foundational paper establishing the biophysics we need to model. The critical dimensions (15-31 µm for block, 175-540 µm for computational models) inform our spatial resolution requirements.
