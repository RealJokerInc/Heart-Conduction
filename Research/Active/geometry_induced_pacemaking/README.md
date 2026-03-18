# Geometry-Induced Pacemaking in Immature hiPSC-CM Models

## Question
Can immature spontaneously-beating hiPSC-CM ionic models (PHAS13) recreate the geometry-induced pacemaking effect — where sharp-tipped tissue geometry causes spontaneous beating to originate from geometric features (tips, corners) rather than uniformly, purely through source-sink impedance modulation without specialized pacemaker ion channels?

## Status: Active

## Why It Matters
The biohybrid fish paper (Lee et al. 2022, Science) demonstrated that a **geometrically insulated node (G-node)** with a sharp-tipped architecture can act as an autonomous pacemaker using ordinary hiPSC-CMs — no specialized pacemaker channels required. The mechanism relies on reduced electrotonic loading at tissue boundaries/tips (high perimeter-to-area ratio), which synchronizes spontaneous activity and creates coordinated pacing. This is the same source-sink impedance principle studied in our boundary conduction speedup research (Kléber effect), but applied to *pacemaking* rather than conduction velocity. If our PHAS13 model can reproduce this, it opens a path to studying engineered biological pacemakers computationally, and validates that geometry alone can organize spontaneous activity in immature cardiomyocyte tissue.

## Engines
- **Monodomain V5.4**: Has PHAS13 (immature, spontaneously-beating hiPSC-CM model). Primary simulation engine.
- **LBM V1**: Boundary effects are its research focus. Could capture geometry-dependent pacing with different numerics.
- **Bidomain V1**: For validation and bath-loading effects at tissue boundaries.
- **Builder**: For generating sharp-tipped mesh geometries (PNG/SVG -> StructuredGrid).

## Completion Criteria
- [ ] Literature review: G-node mechanism, SAN architecture, Fast & Kléber geometry effects
- [ ] Reproduce spontaneous beating with PHAS13 on simple rectangular geometry (baseline)
- [ ] Demonstrate that sharp-tipped geometry causes pacing to originate from the tip
- [ ] Show uniform (non-tipped) geometry produces disorganized/non-localized spontaneous activity
- [ ] Characterize critical geometry parameters (tip angle, node size, exit pathway width)
- [ ] Compare with non-spontaneous (MHAS13) model as negative control
- [ ] Validate across at least two engines (V5.4 + one other)

## Sub-Questions

| Sub-Question | Status | Key Finding |
|-------------|--------|-------------|
| — | — | — |

## Key Findings So Far
*Starting investigation. Paper review complete (see KNOWLEDGE.md).*

## Inspiration Paper
**Lee et al. (2022)** "An autonomously swimming biohybrid fish designed with human cardiac biophysics." *Science* 375(6581):639-647. DOI: 10.1126/science.abh0474

Key design features of the G-node:
- ~600-1700 hiPSC-CMs structurally isolated with a single exit pathway
- Sharp-tipped (acute-angled) corners increase probability of activation at that site
- Mechanism: reflection of intracellular currents at the perimeter synchronizes spontaneous activity
- Source-sink impedance reduction: small activating CMs (source) drive large downstream quiescent muscle (sink)
- **Geometry alone defines the pacemaker** — no specialized ion channel expression required

Co-author: **Andre G. Kléber** — the foundational researcher on cardiac tissue geometry, source-sink mismatch, and conduction block (Fast & Kléber 1995).

## Literature

| Paper | Summary File | Key Insight |
|-------|-------------|-------------|
| Lee et al. 2022 (Science) | [biohybrid fish](literature/lee_2022_biohybrid_fish.md) | G-node: geometry alone creates pacemaker via source-sink impedance |
| Fast & Kléber 1995 | [geometry conduction block](literature/fast_kleber_1995_geometry_conduction_block.md) | Strand-to-bulk expansion causes UCB at 15 µm; critical diameter 175-540 µm |
| Rohr, Kucera, Kléber 1997 (Science) | [paradoxical uncoupling](literature/rohr_kucera_kleber_1997_paradoxical_uncoupling.md) | Partial uncoupling paradoxically restores conduction at expansions |
| Gonzalez-Rajal et al. 2018 (Phys Rev X) | [geometry arrhythmias](literature/gonzalez_rajal_2018_geometry_dependent_arrhythmias.md) | Same cells, different shapes → different dynamics; hiPSC-CM validation |
| Ryzhii & Ryzhii 2022 (PLoS ONE) | [simplified pacemaker models](literature/ryzhii_2022_simplified_pacemaker_models.md) | 2-variable pAP/pCN models: proven pacemaker-excitable coupling in 2D tissue |
| Ye & Bhatt (PMC4244803) | — | SAN architecture: insulation + exit pathways + low-conductance connexins |
| Grijalva et al. 2019 (PMC6864514) | — | TBX18 pacemaker spheroids: single boundary contact >> embedded; 2-3mm exits |
| Joyner & van Capelle 1986 | — | SAN funnel model: gradual resistance transition needed; SAN 5× too small without uncoupling |
| Cabo et al. 1994 | — | Isthmus critical width: 200 µm longitudinal, 600 µm transverse (Luo-Rudy) |
| Kadota et al. 2017 | — | hiPSC-CM island size (50-1000 µm): beat rate independent, but maturation scales with size |
| Hoang et al. 2024 | — | Cardiac organoid shape library (circles, rectangles, pentagrams): ML classifies geometry from physiology |
| Zemlin et al. 2018 | — | Curvature-dependent ectopy: excitation originates at maximal curvature (paradoxical) |
| Maltsev & Lakatta 2009 | — | Coupled-clock SAN model (29 ODEs): clustered cells rescue pacing; strong coupling kills automaticity |
| Fabbri 2017 | — | Gold-standard human SAN model (~30 ODEs): pace-and-drive in 3D with openCARP |
| Inada et al. 2014 | — | Both coupling gradient AND cell-type gradient essential for SAN function |
| Verheijck et al. 1998 | — | Single SAN cell suppressed at g_j > 0.55 nS; 2-3 channels suffice for entrainment |

## Experiments

| Experiment | Engine | Result | Location |
|-----------|--------|--------|----------|
| — | — | — | — |

## Engine References

| File | What it tells you |
|------|-------------------|
| `Monodomain/Engine_V5.4/cardiac_sim/ionic/phas13/model.py` | PHAS13 (immature, spontaneous) ionic model |
| `Monodomain/Engine_V5.4/cardiac_sim/ionic/mhas13/model.py` | MHAS13 (matured, quiescent) — negative control |
| `Monodomain/Engine_V5.4/cardiac_sim/ionic/base.py` | IonicModel ABC interface |
| `Builder/` | Mesh generation from geometry images |
| `Research/Active/boundary_conduction_speedup/` | Related: Kléber boundary speedup effect |
| `Research/Active/mature_hipsc_cm_models/` | Related: PHAS13 -> MHAS13 maturation pathway |

## Future Work
{No deferred items yet.}

## Connected Research
- **boundary_conduction_speedup** — Same source-sink impedance physics, applied to CV rather than pacemaking
- **mature_hipsc_cm_models** — PHAS13 base model; MHAS13 as negative control
- **ionic_model_optimization** — May need to tune PHAS13 parameters for accurate spontaneous beating
