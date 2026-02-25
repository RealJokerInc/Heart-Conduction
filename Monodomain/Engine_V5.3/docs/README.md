# Engine V5.3 Documentation

## Overview

Engine V5.3 is a GPU-accelerated cardiac electrophysiology simulation engine supporting:
- O'Hara-Rudy 2011 (ORd) ventricular myocyte model
- ten Tusscher-Panfilov 2006 (TTP06) ventricular model
- Monodomain tissue simulation with FEM
- Lookup table (LUT) acceleration

---

## Document Index

### Implementation Guides

| Document | Description | Status |
|----------|-------------|--------|
| [ORd_LUT_Research.md](./ORd_LUT_Research.md) | Research on implementing LUT for ORd model | Complete |
| [LUT_Regeneration_Protocol.md](./LUT_Regeneration_Protocol.md) | Protocol for regenerating LUT when parameters change | Complete |

### Quick Reference

#### LUT Configuration (Recommended)

```python
LUTConfig(
    V_min=-100.0,      # mV
    V_max=80.0,        # mV
    n_points=2001,     # 0.09 mV resolution
)
```

#### LUT-Affecting Parameters

Parameters that require LUT regeneration when changed:

| Parameter | Model | Affects |
|-----------|-------|---------|
| `tau_hs_scale` | ORd | INa hs time constant |
| `tau_hsp_scale` | ORd | INa hsp time constant |
| `celltype` | Both | Cell-type specific tables |

#### Usage Example

```python
from ionic import ORdModel, CellType

# With LUT acceleration
model = ORdModel(
    celltype=CellType.ENDO,
    device='cuda',
    use_lut=True  # Enable LUT
)

# After modifying LUT-affecting parameters
model.params.tau_hs_scale = 0.8
model.rebuild_lut()  # Regenerate LUT
```

---

## Directory Structure

```
Engine_V5.3/
├── docs/                    # Documentation
│   ├── README.md            # This file
│   ├── ORd_LUT_Research.md  # LUT implementation research
│   └── LUT_Regeneration_Protocol.md
├── ionic/                   # Ionic models
│   ├── lut.py               # LUT base classes (TTP06LUT)
│   ├── ord/                 # ORd model
│   │   ├── model.py
│   │   ├── gating.py
│   │   ├── currents.py
│   │   ├── calcium.py
│   │   ├── camkii.py
│   │   └── parameters.py
│   └── ttp06/               # TTP06 model
│       └── ...
├── fem/                     # Finite element methods
├── solver/                  # Linear and time solvers
├── tissue/                  # Tissue simulation
└── tests/                   # Validation tests
```

---

## References

1. O'Hara T, et al. (2011). PLoS Comput Biol. [PMC3102752](https://pubmed.ncbi.nlm.nih.gov/21637795/)
2. ten Tusscher KH, Panfilov AV (2006). Am J Physiol Heart Circ Physiol.
3. Sherwin SJ, et al. (2022). Frontiers in Physiology. [10.3389/fphys.2022.904648](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2022.904648/full)
4. OpenCarp Documentation. [opencarp.org](https://opencarp.org/)
