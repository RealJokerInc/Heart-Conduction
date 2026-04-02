# hiPSC-CM Electrophysiology Target Values

Ground truth values for the Engine Tuner V1 optimizer, derived from experimental literature on human induced pluripotent stem cell-derived cardiomyocytes (hiPSC-CMs).

**V1 approach**: Targets are aligned to TTP06's natural operating range. We achieve hiPSC-CM-like conduction (slow CV, moderate anisotropy) primarily through diffusion coefficient reduction (representing immature gap junctions), with moderate ionic conductance scaling. Parameters that would require extreme perturbation (V_rest, dvdt_max) are kept at TTP06's native values to avoid pushing the model outside its validated regime.

## Selected Targets

```
┌────────────────────────────────────┬───────────┬────────────────────────────────────┐
│ Parameter                          │ Value     │ Rationale                          │
├────────────────────────────────────┼───────────┼────────────────────────────────────┤
│ CV longitudinal                    │ 25 cm/s   │ Matured hiPSC-CM, achievable       │
│                                    │           │ via D reduction (not extreme GNa)  │
│ CV transverse                      │ 12.5 cm/s │ AR = 2.0                           │
│ Anisotropy ratio (LCV/TCV)         │ 2.0       │ Wang 2013, aligned hiPSC-CM        │
│ Single-cell APD90                  │ 250 ms    │ Midrange mature hiPSC-CM           │
│ Resting membrane potential (V_rest)│ -85 mV    │ TTP06 native (don't fight the      │
│                                    │           │ model — GK1 stays near published)  │
│ dvdt_max (upstroke velocity)       │ 150 V/s   │ Matured hiPSC-CM in 3D EHT,        │
│                                    │           │ achievable with GNa ~0.5           │
└────────────────────────────────────┴───────────┴────────────────────────────────────┘

Tissue APD90: estimated ~225 ms (25 ms shorter than single-cell
due to electrotonic loading — to be determined by the optimizer).
```

## Why These Values (Not the Extreme hiPSC-CM Values)

TTP06 is an adult ventricular model. Forcing it to reproduce deeply immature hiPSC-CM behavior (V_rest = -60 mV, dvdt_max = 10 V/s) would require scaling GNa to 0.05 and GK1 to 0.1 — far outside the model's validated regime. The gating kinetics, reversal potentials, and current interactions were all fitted at adult values. Extreme perturbation produces numerical artifacts, not meaningful biology.

Instead, V1 targets represent **matured hiPSC-CM** tissue — the upper end of what's experimentally achievable with maturation protocols — which falls comfortably within TTP06's capacity:

```
    WHAT CHANGES FROM DEFAULT TTP06:

    CV:       65 → 25 cm/s     Achieved primarily via D reduction
                                (immature gap junctions = less Cx43)
                                with moderate GNa reduction (~0.5×)

    APD:      280 → 250 ms     Moderate GKr/GCaL adjustment
                                Well within TTP06's range

    dvdt_max: 300 → 150 V/s    GNa ~0.5 (not extreme)
                                V_rest stays at -85 → full Na availability

    V_rest:   -85 → -85 mV     NO CHANGE. GK1 stays near published.
                                Avoids destabilizing resting potential.

    AR:       1.0 → 2.0        Pure D_long/D_trans ratio.
                                No ionic parameter involvement.
```

## Parameter Budget Estimate

```
    To hit these targets, approximate scaling factors needed:

    GNa  ≈ 0.4–0.6     (150 V/s upstroke, reduced CV)
    PCa  ≈ 0.7–1.0     (slightly shorter plateau for APD=250)
    GKr  ≈ 1.2–1.8     (faster repolarization for shorter APD)
    GKs  ≈ 0.8–1.2     (minor adjustment)
    GK1  ≈ 0.9–1.1     (V_rest stays near -85, minimal change)
    Gto  ≈ 0.5–1.5     (phase 1 morphology)

    D_long  ≈ 0.0003–0.0006 cm²/ms  (vs adult ~0.001–0.002)
    D_trans ≈ 0.00015–0.0003 cm²/ms  (AR = 2.0)

    All within [0.3, 2.0] — no extreme scaling needed.
    GNa lower bound of 0.3 is sufficient (not 0.05).
```

## Selection Logic

**CV = 25 cm/s longitudinal, 12.5 cm/s transverse (AR = 2.0)**

Upper range of matured hiPSC-CM values. The literature reports CV = 5–50 cm/s depending on maturation, with most studies finding 10–25 cm/s for well-differentiated monolayers. We target 25 cm/s because:
- Achievable with moderate D reduction (gap junction immaturity) + moderate GNa reduction
- Not so low that we need extreme GNa scaling
- Represents the practical range for engineered cardiac tissues

AR = 2.0 from Wang 2013 ([DOI](https://doi.org/10.1016/j.biomaterials.2013.07.039)) on aligned hiPSC-CM substrates.

**Single-cell APD90 = 250 ms**

Midrange of mature hiPSC-CM values (200–300 ms). Adult TTP06/epi default is ~280 ms at CL=1000, so this requires moderate GKr increase (faster repolarization) and/or GCaL decrease (shorter plateau). Well within the model's capacity.

**V_rest = -85 mV (TTP06 native)**

Keep at published value. Forcing V_rest to -80 mV requires reducing GK1 to 0.1–0.3×, which then reduces Na channel availability (Boltzmann inactivation), requiring even more GNa reduction for the same upstroke. This cascade pushes the model into an extreme regime. Since V_rest = -85 mV is within the physiological range and our focus is CV/APD tuning, we don't fight this.

**dvdt_max = 150 V/s**

Matured hiPSC-CM in 3D EHT (Lemme 2018, [DOI](https://doi.org/10.1038/s41598-017-05600-w): 219±15 V/s). Achievable with GNa ≈ 0.4–0.6 at V_rest = -85 mV (full Na channel availability). Immature values (10–50 V/s) would require GNa < 0.15, which is outside TTP06's comfortable operating range.

## Literature Sources

| Citation | DOI | Key Values |
|----------|-----|------------|
| MacQueen 2019 (Parker lab) | [10.1038/s41551-018-0271-5](https://doi.org/10.1038/s41551-018-0271-5) | CV = 5.2 cm/s (hiPSC-CM ventricle construct) |
| Sheehy 2014 (Parker lab) | [10.1016/j.stemcr.2014.01.015](https://doi.org/10.1016/j.stemcr.2014.01.015) | Quality metrics, optical mapping CV/APD |
| Wang 2013 | [10.1016/j.biomaterials.2013.07.039](https://doi.org/10.1016/j.biomaterials.2013.07.039) | AR = 1.8-2.0 (aligned hiPSC-CM) |
| Park 2019 (Parker lab) | [10.1161/CIRCULATIONAHA.119.039711](https://doi.org/10.1161/CIRCULATIONAHA.119.039711) | CPVT engineered tissue EP |
| Lemme 2018 | [10.1038/s41598-017-05600-w](https://doi.org/10.1038/s41598-017-05600-w) | 3D EHT dvdt_max = 219 V/s (matured) |
| Paci 2018 | [10.3389/fphys.2018.00080](https://doi.org/10.3389/fphys.2018.00080) | In-silico: APD90 = 365→174 ms (d30→d70) |
| Feyen 2023 (Cells) | [PMC10487143](https://pmc.ncbi.nlm.nih.gov/articles/PMC10487143/) | CV = 9-11 cm/s (monolayer, optical mapping) |
| Martinez 2022 | [10.1016/j.hrthm.2022.12.034](https://doi.org/10.1016/j.hrthm.2022.12.034) | APD90 = 578 ms (VSD) / 237 ms (MEA) |

## Default TuningTargets

```python
HIPSC_CM_TARGETS = TuningTargets(
    apd_90=250.0,               # ms, midrange mature hiPSC-CM
    cv_longitudinal=25.0,       # cm/s, matured aligned tissue
    cv_transverse=12.5,         # cm/s, AR = 2.0
    tissue_apd_90=225.0,        # ms, estimated (cell APD - 25 ms loading)
    restitution=[               # estimated from literature trends
        (50, 150),              # DI=50ms → APD=150ms
        (100, 190),             # DI=100ms → APD=190ms
        (200, 230),             # DI=200ms → APD=230ms
        (500, 248),             # DI=500ms → APD=248ms (near steady-state)
    ],
    v_rest=-85.0,               # mV, TTP06 native
    dvdt_max=150.0,             # V/s, matured hiPSC-CM in 3D EHT
)
```

## Comparison Table

```
                        V1 Targets           hiPSC-CM (lit.)      Adult human ventricle
                        ──────────           ───────────────      ─────────────────────
CV longitudinal         25 cm/s              5–50 cm/s            65 cm/s
CV transverse           12.5 cm/s            3–25 cm/s            25 cm/s
Anisotropy ratio        2.0                  1.8–2.0              2.5–3.5
APD90 (single cell)     250 ms               200–500 ms           280 ms
Tissue APD90            ~225 ms              (not well reported)  ~260 ms
V_rest                  -85 mV               -60 to -85 mV        -85 mV
dvdt_max                150 V/s              10–250 V/s           300 V/s

V1 targets sit at the intersection of:
  ✓ Within TTP06's validated operating range
  ✓ Within experimental hiPSC-CM literature range
  ✓ Achievable without extreme parameter scaling
```
