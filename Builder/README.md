# Builder

Input construction tools for **engine_V5.4** pipeline. Converts image-based designs into simulation-ready data.

## Documentation

- **[BACKEND.md](./BACKEND.md)** - Full backend API documentation with usage examples

## Quick Start

```python
from Builder.MeshBuilder import MeshBuilderSession
from Builder.StimBuilder import StimBuilderSession, StimType

# MeshBuilder
mesh = MeshBuilderSession()
mesh.image_array = arr  # numpy array from image
mesh.image_size = size
mesh.threshold_transparency(128)
mesh.configure_group(color=(0,0,0,255), label="Tissue", cell_type="ventricular", D_xx=0.001, D_yy=0.0003)

# StimBuilder
stim = StimBuilderSession()
stim.image_array = arr
stim.image_size = size
stim.threshold_transparency(128)
stim.configure_current_injection(color=(255,0,0,255), label="S1", amplitude=52.0, bcl=500.0, num_pulses=10)
```

## Architecture

```
Builder/
├── common/              # Shared image processing
│   ├── image.py        # load, detect, threshold, filter
│   └── utils.py        # color conversion
│
├── MeshBuilder/         # Image → Tissue mesh
│   ├── models.py       # CellGroup (conductivity tensor)
│   ├── session.py      # MeshBuilderSession
│   └── export.py       # (future)
│
└── StimBuilder/         # Image → Stimulus map
    ├── models.py       # StimRegion, StimProtocol
    ├── session.py      # StimBuilderSession
    └── export.py       # (future)
```

## Workflow

1. **Load image** (PNG, or SVG via cairosvg)
2. **Clean anti-aliasing**
   - `threshold_transparency()` - binarize alpha channel
   - `filter_small_groups()` - remove RGB blending artifacts
3. **Configure groups** - label colors, assign properties
4. **Export** - get masks/configs for simulation

## Image Requirements

For best results with draw.io SVG exports:

| Setting | Value |
|---------|-------|
| Background | Transparent |
| Stroke | None |
| Border | 0 |

## Modules

### MeshBuilder

Generates tissue meshes with:
- Cell type labels
- Full 2x2 conductivity tensor per group
- Physical dimensions and resolution

### StimBuilder

Generates stimulation maps with:
- Stimulus type (current injection / voltage clamp)
- Amplitude and timing protocol
- BCL input → frequency storage
- Pulse time generation

---

## Integration with v5.4

```
cardiac_sim/
├── Builder/                    # This module
│   ├── common/
│   ├── MeshBuilder/
│   └── StimBuilder/
│
├── tissue/
│   └── mesh.py                # ← MeshBuilder outputs
│
└── simulation/
    └── stimulus.py            # ← StimBuilder outputs
```

## TODO

- [x] Design MeshBuilder core (color detection, session labeling, conductivity)
- [x] Design StimBuilder core (color detection, timing protocol)
- [x] Determine discretization support → Both FEM + structured grids
- [ ] Finalize v5.4 integration points
- [ ] Design root Builder coordinator class
- [ ] Design MeshBuilder scaling (size-up interpolation)
- [ ] Define session save/load schema
- [ ] Design CLI interface
- [ ] Design UI interface
