# Lee et al. 2022 — Biohybrid Fish G-Node Pacemaker

**Citation**: Lee KY, Park SJ, Matthews DG, Kim SL, Marquez CA, Zimmerman JF, Ardoña HAM, Kléber AG, Lauder GV, Parker KK. "An autonomously swimming biohybrid fish designed with human cardiac biophysics." *Science* 375(6581):639-647, 2022.
**DOI**: 10.1126/science.abh0474
**PMC**: PMC8939435

## Key Contribution
First demonstration that **geometry alone** (without specialized pacemaker ion channels) can create a functional autonomous pacemaker in engineered cardiac tissue.

## G-Node Design
- **Architecture**: Geometrically insulated node (G-node) — a small cluster of CMs structurally isolated with a single exit pathway to surrounding muscle
- **Cell counts**: ~600 cells (small pointed node) or ~1700 cells (large node)
- **Substrate**: Micromolded gelatin thin film, 200 µm thick; line groove features (25 µm ridge, 4 µm groove width, 5 µm groove depth)
- **Cells**: hiPSC-CMs (73,000 live CMs total in fish body) or neonatal rat ventricular CMs
- **Total fish body**: ~15 mm length

## Mechanism
1. Reflection of intracellular currents at G-node perimeter synchronizes spontaneous activity
2. Acute-angled anterior corners increase probability of activation at that site by decreasing downstream cell count
3. Source-sink impedance reduction: small activating CMs (source) drive large downstream muscle (sink)
4. **Perimeter-to-area ratio** more important than corner angle specifically
5. The G-node corner design did not affect activation probability — the insulation and exit pathway mattered more

## Muscular Bilayer (Antagonistic Pair)
- Shortening on one side directly translates to axial stretching on the opposite side
- CMs electrically connected within each side, mechanically coupled across sides
- Creates closed-loop control: stretch → stretch-activated channel activation → contraction
- Self-sustaining for >100 days, with speed increasing during first month (CM maturation)

## Key Quotes
- "A pacemaker may be defined by its geometry and source-sink relationships as well as its ion channel expression"
- "The reflection of intracellular currents at the perimeter of the G-node would synchronize the spontaneous activity and initiate coordinated pacemaking"

## What They Did NOT Do
- No computational modeling — purely experimental
- No parameter sweeps on geometry
- No quantification of critical tip angle or minimum node size for pacemaking

## Relevance to Our Work
This is the inspiration paper. We aim to computationally reproduce and characterize the geometry-dependent pacemaking effect using our PHAS13 immature hiPSC-CM model. We can do the systematic geometry parameter sweeps they didn't do.
