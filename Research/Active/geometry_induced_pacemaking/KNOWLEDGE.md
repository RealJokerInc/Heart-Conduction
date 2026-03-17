# Geometry-Induced Pacemaking — Knowledge File

> This file is a running synthesis. Updated as findings accumulate.
> When the question is complete, a copy is promoted to `Research/Knowledge/`.

## Current Understanding

### The Core Phenomenon

Immature hiPSC-CMs beat spontaneously due to high funny current (If) and low inward rectifier (IK1). When seeded on a uniform substrate, all cells beat independently at slightly different intrinsic rates, producing disorganized activity. However, when the tissue geometry includes **sharp tips or narrow exit pathways**, the cells at these geometric features become the dominant pacemaker. This happens without any specialized pacemaker ion channels — geometry alone organizes the spontaneous activity.

This is the same source-sink impedance physics studied in our boundary conduction speedup research (Kléber effect), but applied to *pacemaking origin* rather than conduction velocity.

---

## 1. Detailed Mechanism

### Step 1 — Electrotonic Suppression Is the Default State

All cardiomyocytes with automaticity (phase 4 diastolic depolarization) are suppressed when embedded in a large, well-coupled tissue mass. The governing equation for a coupled pacemaker cell:

```
Cm * dVm/dt = I_ion(Vm, states) - g_j * (Vm - V_neighbor)
```

During phase 4, the pacemaker cell sits at ~-60 mV while coupled quiescent cells sit at ~-80 mV. The coupling current `g_j * (Vm - V_neighbor)` creates a persistent 20 mV driving force for outward current that directly opposes If-driven depolarization.

**Quantitative thresholds** (Verheijck et al. 1998, Wilders et al. 1993):
- A single rabbit SAN cell ceases spontaneous activity at coupling conductances above ~0.55 nS to a single atrial cell
- As few as 2-3 gap junctional channels (~0.15-0.23 nS total) suffice for 1:1 frequency entrainment between pacemaker cells
- Fewer than 10 coupled ventricular cells can suppress a single pacemaker cell's automaticity

**The scaling problem**: In a 2D tissue sheet, the electrotonic sink scales as lambda² (space constant squared). The SAN contains <5,000 pacemaker cells in rabbit, yet must drive an atrial mass orders of magnitude larger. Without geometric protection, coupling extinguishes all pacemaking.

### Step 2 — Geometric Features Reduce the Load

A cell's electrotonic load is proportional to the number of downstream neighbors it must supply current to. Geometry modulates this:

**Tissue boundaries (sealed ends)**: At a no-flux boundary (dVm/dx = 0), axial current that would normally flow downstream is reflected back. Edge cells have effectively halved electrotonic load vs. interior cells. This produces enhanced upstroke velocity and increased local CV near boundaries (the Kléber boundary speedup: +7-13% CV).

**Sharp tips and convex boundaries**: A cell at a tissue tip has neighbors in only one direction instead of all directions. For a generic 2D interior cell with ~4-6 electrical neighbors, a tip cell might have only 1-2. This dramatically reduces the current sink. The perimeter-to-area ratio quantifies the effect.

**Narrow strands and isthmuses**: In a narrow strand of width w, cells are bounded on two sides, reducing their neighbor count. For strands narrower than ~2*lambda (lambda ≈ 300-430 µm in SAN tissue), boundary effects dominate throughout the cross-section.

### Step 3 — Below the Suppression Threshold, Automaticity Emerges

When the effective coupling load drops below the critical Gc (~0.5 nS per neighbor for rabbit SAN cells), diastolic depolarization proceeds to threshold and spontaneous firing occurs. Cells at geometric extremities are the first to escape suppression.

### Step 4 — The Pacemaker Cluster Synchronizes

Once a small group of cells at a geometric extremity begin firing, they synchronize through mutual entrainment at coupling conductances as low as 0.15 nS (2-3 gap junction channels). The common frequency is dominated by the fastest cell.

### Step 5 — Source-Sink Matching at the Exit

The geometric boundary that protects the pacemaker from suppression also controls impedance matching for wave exit. The shape must provide:
- Sufficient insulation to prevent suppression (few downstream neighbors)
- Sufficient exit pathway area to exceed the liminal area (~0.3 mm diameter, Fozzard & Schoenberg 1972)
- Gradual enough expansion to keep wavefront curvature below critical (κ < c₀/D)

**Eikonal-curvature relationship**:
```
c_n = c_0 - D * κ
```
where c₀ is planar wave speed, D is diffusion coefficient, κ is local curvature. Propagation fails when κ exceeds κ_c = c₀/D.

### Step 6 — Current Reflection Enhances the Wavefront

At sealed boundaries flanking the exit pathway, reflected current augments the propagating wavefront, increasing local CV by ~7-13% and boosting the safety factor above 1.0 at the critical expansion point.

**Safety factor** (Shaw & Rudy 1997):
```
SF = Q_generated / Q_required
```
Normal planar propagation: SF ≈ 1.17-1.77. At geometric expansions, SF drops sharply and can transiently fall below 1.0 (conduction block). At tissue boundaries: SF rises above bulk value due to current reflection.

### The Paradox of Partial Uncoupling (Rohr, Kucera, Fast, Kléber 1997, Science)

In strand-to-expansion geometries (25-70 µm strands into 2.2×2.2 mm monolayers), partial gap junction uncoupling via palmitoleic acid paradoxically **restored** conduction that was previously blocked. Mechanism: uncoupling affects source and sink **asymmetrically** — reducing coupling in the large mass (sink) decreases its current demand more than it reduces the source's current supply.

### Biological Precedent: The Sinoatrial Node

The native SAN uses all these principles simultaneously:
- ~10,000 pacemaker cells must drive billions of atrial CMs (extreme source-sink mismatch)
- Connective tissue barriers insulate the SAN from atrial loading
- Discrete exit pathways (2-3 mm) limit coupling
- Low-conductance connexins (Cx45, 20-40 pS single-channel) vs atrial Cx43 (60-100 pS)
- Na⁺ channel upregulation at SAN periphery boosts source current at exit points
- Coupling gradient: D increases 1000-fold from SAN center (0.00035 mm²/ms) to atrium (0.35 mm²/ms)

Neither a gradient in coupling alone nor in cell type alone suffices — **both are essential** for SAN function (Inada et al. 2014).

---

## 2. Key Geometries from Literature

### 2.1 Pointed Tip / G-Node (Lee et al. 2022, Science)

The biohybrid fish G-node: a geometrically insulated cardiac tissue node with a sharp pointed tip and single exit pathway.
- **Dimensions**: ~600 cells (small pointed) or ~1700 cells (large node)
- **Substrate**: Micromolded gelatin thin film, 200 µm thick; groove features 25 µm ridge, 4 µm groove, 5 µm depth
- **Cells**: hiPSC-CMs or neonatal rat ventricular CMs
- **Key result**: Perimeter-to-area ratio more important than corner angle. Fish body ~15 mm total length. G-node acted as dominant pacemaker. "A pacemaker may be defined by its geometry and source-sink relationships as well as its ion channel expression."

### 2.2 Strand-to-Expansion (Fast & Kléber 1995)

Narrow cell strands of variable width emerging into a large cell area.
- **Critical dimensions**: Block at 15±4 µm (1-2 cells); no block at 31±8 µm (3+ cells)
- **Cells**: Neonatal rat heart cells
- **Measurement**: Dye RH-237, 15 µm spatial resolution
- **Computational (Wang & Rudy 1995)**: Critical diameter 175-200 µm (2D), 472-540 µm (3D, 2.7× larger)

### 2.3 Serpentine Tracks (Gonzalez-Rajal et al. 2018, Phys Rev X)

1D serpentine tracks of varying width with sharp turns.
- **Dimensions**: Widths 100, 200, 500 µm; lengths up to 7 cm
- **Cells**: iOS-HEK (engineered excitable) + hiPSC-CMs for validation
- **Computational**: 10 µm spatial discretization, Noble model comparison at 0.1 mm grid
- **Key result**: Second-degree conduction block at serpentine corners; same cells in different shapes show regular spiking, alternans, or conduction block. hiPSC-CMs showed qualitatively similar geometry-dependent bifurcations.
- **hiPSC-CM parameters**: CV = 7.2 cm/s, APD = 360 ms, λ = 2.6 cm

### 2.4 Isthmus/Constriction (Cabo et al. 1994)

Narrow isthmus in a 2D sheet, experimental + computational.
- **Critical widths (Luo-Rudy model)**: 200 µm longitudinal, 600 µm transverse
- **Experimental (sheep)**: Rate-dependent, 1-3.5 mm depending on propagation direction and rate

### 2.5 Branching Strands (Kucera, Kléber & Rohr 1998)

Strands with branch points.
- **Dimensions**: 70-80 µm (narrow) to 230-270 µm (wide)
- **Key result**: CV maximally slowed 63% at branch points; in elevated K⁺, slowed 93% (from 15.7 to 1.1 cm/s)
- **"Pull and push" mechanism**: Branches act as sinks approaching ("pull"), then sources once excited ("push")

### 2.6 Spheroid-Contact Geometries (Grijalva et al. 2019)

TBX18-induced pacemaker spheroids with different contact patterns to ventricular monolayers.
- **Spheroids**: 1000 cells each, 150 spheroids per source region
- **Contact designs**: Embedded (8-17% pacing), single boundary 2-3 mm (78% pacing, optimal), elongated sink (33%)
- **Key result**: Physical separation + single boundary contact critical; 2-3 mm contact length derived from human SAN exit pathway dimensions

### 2.7 Micropatterned Islands (Kadota et al. 2017)

Square hiPSC-CM islands of different sizes.
- **Sizes**: 50, 100, 250, 500, 1000 µm edges
- **Key result**: Beat rate largely independent of island size, but voltage amplitude increased 1.81× from 50 µm to 1 mm. Island size drove 58% of changes in functional maturation gene expression.

### 2.8 Cardiac Organoid Shape Library (Hoang et al. 2024)

hiPSC organoids in circles, rectangles, and pentagrams.
- **Shapes**: Circles (200-1000 µm), rectangles (1:1 to 1:4 aspect ratio), stars/pentagrams (blunt to sharp vertices)
- **Key result**: ML classifiers achieved 76-79% accuracy classifying geometry from physiology alone, confirming geometry-function coupling. Sharp pentagram geometries produced distinct functional profiles.

### 2.9 SAN Funnel Model (Joyner & van Capelle 1986)

Radially symmetric 2D model with gradual resistance transition from SAN to atrium.
- **Key result**: Driving the atrium required SAN 5× larger than physiologically reported, unless partial uncoupling was present. Gradual funnel geometry is necessary.

### 2.10 Curvature-Dependent Ectopy (Zemlin et al. 2018)

Square, Pacman, elliptic, and disc geometries with depolarized regions.
- **Models**: Majumder-Korhonen (neonatal rat), TNNP (adult human), Aliev-Panfilov, FHN
- **Key result**: Primary ectopic excitation originates at **areas of maximal curvature** — paradoxically where stimulating electrotonic currents are minimal. Oscillatory instability mechanism at the boundary.

---

## 3. Ionic Models Demonstrated for This Problem

### Tier 1: Rapid Prototyping (2-3 variables, proven geometry behavior)

**Pacemaker Aliev-Panfilov (pAP)** — Ryzhii & Ryzhii 2022, PLoS ONE
- 2 variables (u, v). Parameter b_AP = -a converts excitable → oscillatory via Hopf bifurcation
- Proven in 2D SAN tissue (10×10 mm, 200×200 mesh) and 3D intestine tube
- Frequency range: 0.007-7.6 Hz (0.4-450 BPM)
- Code: github.com/mryzhii/Simplified-pacemaker-cell-models (MATLAB + CellML)
- **Key tissue finding**: Strong coupling → synchronization at slow rate; weak coupling → 2:5 block; over-coupling → quiescence

**Pacemaker Corrado-Niederer (pCN)** — Same paper
- 2 variables (u, h). Wider synchronization area, lower sensitivity to coupling strength than pAP
- More rectangular AP shape. Same code repository.

**FitzHugh-Nagumo (oscillatory regime)** — 2 variables, massive literature, but unrealistic AP shape.

### Tier 2: Physiological Validation (18-23 variables)

**Paci 2013** — Already implemented in our project as PHAS13 (18 state variables, 12 currents)
- Spontaneous beating via If/low IK1 interplay
- Used in tissue simulations by Botti et al. (2017, 2025) with monodomain/bidomain
- Code: Already at `Monodomain/Engine_V5.4/cardiac_sim/ionic/phas13/model.py`
- **Natural choice** — no new implementation needed

**Paci 2018/2020** — 23 ODEs, improved Ca²⁺ handling with RyR gating
- Can reproduce DADs and calcium-related abnormalities
- Used in 2D MEA bidomain tissue (Botti et al. 2025, J Physiol)
- Worth upgrading to if calcium-dependent effects matter

**Kernik-Clancy 2019** — ~20-30 variables, 13 currents including ICaT
- Beats even during total INa and If block (calcium-driven automaticity)
- Built-in population variability framework
- Code: github.com/ClancyLabUCD/IPSC-model
- Used in tissue strand excitation study (Shetty 2026)

### Tier 3: Gold Standard SAN Models (29-30 variables)

**Fabbri 2017** (Human SAN) — ~30 variables, 11+ currents
- The most validated human SAN model. If, ICaL, ICaT, INaCa automaticity.
- Proven pace-and-drive in 3D with openCARP (788 simulations, Loewe 2022)
- SAN-SEP-RA tissue geometry: pace-and-drive requires SAN 2.85× wider than atrial
- Code: CellML models.physiomeproject.org/e/568

**Maltsev-Lakatta 2009** (Rabbit SAN, coupled-clock) — 29 ODEs
- Most physiologically complete: dual Ca²⁺ clock + membrane clock
- Proven in 2D heterogeneous tissue (25×25 grid, Campana/Maltsev 2022)
- **Key tissue finding**: Clustered spontaneous cells rescue pacing in heterogeneous tissue; weak coupling required (ρ=10⁴ MΩ·m); strong coupling kills automaticity
- CUDA/GPU implementation exists

### Tier 4: Modified Ventricular Models

**TTP06 with IK1 suppression** — 19 variables, our standard model modified
- GK1 reduced to <19% of normal → spontaneous beating
- 2D tissue (100×50 cells): 25 pacemaker columns + 75 ventricular columns
- Critical: D must be reduced to 9% of normal for successful driving

---

## Recommended Implementation Path

1. **Prototype with pAP** (2 variables) — identify geometry regimes quickly, sweep tip angles, node sizes, exit widths. Can run thousands of simulations.
2. **Validate with PHAS13** (already implemented, 18 variables) — physiologically grounded hiPSC-CM model in Monodomain V5.4.
3. **Negative control with MHAS13** (matured, quiescent) — should show no geometry-dependent pacemaking.
4. If SAN-specific behavior needed, implement **Fabbri 2017** using CellML code.

---

## Key Decisions
*None yet — starting investigation.*

## Open Questions
1. Can PHAS13 spontaneous beating rate be captured accurately enough, or do parameters need tuning?
2. What spatial resolution (dx) is needed to resolve sharp tips with ~600 cells?
3. Does the LBM engine handle spontaneous (no-stimulus) simulations correctly?
4. What is the minimum tip angle that produces localized pacing?
5. How does tissue anisotropy interact with geometry-induced pacing?
6. Is there a critical tissue size below which all cells synchronize regardless of geometry?
7. Should we implement the pAP model first for rapid prototyping, or go straight to PHAS13?

## Connections
- **Engines**: Monodomain V5.4 (PHAS13), LBM V1 (boundary effects), Bidomain V1 (bath loading), Builder (mesh generation)
- **Related research**: boundary_conduction_speedup (same source-sink physics), mature_hipsc_cm_models (PHAS13 base)
- **Pipelines**: Builder (geometry generation), possibly Optimizer (PHAS13 parameter tuning)
