# Builder Backend Documentation

## Overview

The Builder package provides tools for converting images into simulation-ready data structures for cardiac simulations. It consists of two main components:

- **MeshBuilder**: Converts images to tissue meshes with conductivity properties
- **StimBuilder**: Converts images to stimulation region maps with timing protocols

Both builders share common image processing utilities and follow the same workflow pattern.

---

## Package Structure

```
Builder/
├── __init__.py              # Package exports
├── requirements.txt         # pillow, numpy, cairosvg
├── BACKEND.md              # This documentation
│
├── common/                  # Shared utilities
│   ├── __init__.py
│   ├── image.py            # Image processing functions
│   └── utils.py            # Color utilities
│
├── MeshBuilder/
│   ├── __init__.py
│   ├── models.py           # CellGroup dataclass
│   ├── session.py          # MeshBuilderSession
│   └── export.py           # (future)
│
└── StimBuilder/
    ├── __init__.py
    ├── models.py           # StimRegion, StimProtocol, enums
    ├── session.py          # StimBuilderSession
    └── export.py           # (future)
```

---

## Installation

```bash
pip install pillow numpy cairosvg
```

Or using the requirements file:
```bash
pip install -r Builder/requirements.txt
```

---

## Common Image Processing

### Workflow

1. **Load image** → numpy array
2. **threshold_transparency()** → Remove alpha-based anti-aliasing
3. **filter_small_groups()** → Remove RGB blending artifacts (if needed)
4. **Configure groups** → Assign labels and properties
5. **Export** → Get masks/configs for simulation

### Functions (Builder.common)

```python
from Builder.common import (
    load_image,              # Load image file → (array, size)
    detect_colors,           # Find unique colors → {color: info}
    threshold_transparency,  # Binarize alpha channel
    filter_small_groups,     # Remove small artifact groups
    color_to_hex,           # (255, 0, 0) → "#ff0000"
    hex_to_color,           # "#ff0000" → (255, 0, 0)
    is_background_color,    # Check if white/transparent
)
```

### Anti-Aliasing Handling

| Function | Purpose | When to Use |
|----------|---------|-------------|
| `threshold_transparency()` | Binarize alpha (0 or 255) | RGBA images with alpha anti-aliasing |
| `filter_small_groups()` | Mode filter small groups | Overlapping shapes with RGB blending |

---

## MeshBuilder

### Data Model

```python
@dataclass
class CellGroup:
    color: Tuple[int, ...]      # RGB(A) tuple
    pixel_count: int
    label: str                  # e.g., "Atria", "Ventricle"
    cell_type: str              # e.g., "atrial", "ventricular"
    is_background: bool
    conductivity_tensor: np.ndarray  # 2x2 [[Dxx, Dxy], [Dxy, Dyy]]
```

### Usage

```python
from Builder.MeshBuilder import MeshBuilderSession

# Create session
session = MeshBuilderSession()

# Load and process image
session.image_array = arr  # numpy array from SVG/PNG
session.image_size = (width, height)

# Clean anti-aliasing
session.threshold_transparency(alpha_threshold=128)

# Filter RGB artifacts (if overlapping shapes)
if len(session.tissue_groups) > expected:
    session.filter_small_groups(min_percent=0.1)

# Configure tissue groups
black = (0, 0, 0, 255)
session.configure_group(
    color=black,
    label="Ventricle",
    cell_type="ventricular",
    D_xx=0.001,  # cm²/ms - longitudinal
    D_yy=0.0003, # cm²/ms - transverse
    D_xy=0.0     # cm²/ms - off-diagonal
)

# Set physical dimensions
session.set_dimensions(
    tissue_width=5.0,   # cm
    tissue_height=5.0,  # cm
    dx=0.01            # cm
)

# Check status
print(session.summary())
print(f"All configured: {session.all_groups_configured}")
```

### Properties

| Property | Returns |
|----------|---------|
| `tissue_groups` | List of non-background CellGroups |
| `background_groups` | List of background CellGroups |
| `all_groups_configured` | True if all tissue groups configured |
| `unconfigured_groups` | List of groups needing configuration |

---

## StimBuilder

### Data Models

```python
class StimType(Enum):
    CURRENT_INJECTION = "current_injection"
    VOLTAGE_CLAMP = "voltage_clamp"

class StimTarget(Enum):
    INTRACELLULAR = "intracellular"
    EXTRACELLULAR = "extracellular"

@dataclass
class StimProtocol:
    duration: float      # ms - pulse duration
    start_time: float    # ms - first pulse start
    frequency: float     # Hz - stored internally
    num_pulses: int      # None = continuous

    # Properties
    bcl: float          # ms - computed from frequency

    # Methods
    set_bcl(bcl_ms)              # Set frequency from BCL
    get_pulse_times(max_time)    # Get list of pulse times

@dataclass
class StimRegion:
    color: Tuple[int, ...]
    pixel_count: int
    label: str
    is_background: bool
    stim_type: StimType
    amplitude: float      # μA/cm² or mV
    target: StimTarget
    protocol: StimProtocol
```

### Usage

```python
from Builder.StimBuilder import StimBuilderSession, StimType

# Create session
session = StimBuilderSession()

# Load and process image
session.image_array = arr
session.image_size = (width, height)
session.threshold_transparency(128)

# Configure S1 pacing
black = (0, 0, 0, 255)
session.configure_current_injection(
    color=black,
    label="S1_pacing",
    amplitude=52.0,      # μA/cm²
    duration=1.0,        # ms
    start_time=0.0,      # ms
    bcl=500.0,           # ms (user input)
    num_pulses=10
)

# Configure S2 premature stimulus
red = (255, 0, 0, 255)
session.configure_current_injection(
    color=red,
    label="S2_premature",
    amplitude=52.0,
    duration=1.0,
    start_time=350.0,    # Coupled to S1
    bcl=1000.0,
    num_pulses=1
)

# Or use voltage clamp
session.configure_voltage_clamp(
    color=red,
    label="Clamp_region",
    voltage=-80.0,       # mV
    duration=100.0,
    start_time=0.0,
    bcl=1000.0,
    num_pulses=None      # Continuous
)

# Get summary
print(session.summary())
```

### Export for Simulation

```python
# Get full configuration as list of dicts
configs = session.get_stim_config()

# Example output:
# {
#     'label': 'S1_pacing',
#     'color': (0, 0, 0, 255),
#     'pixel_count': 45216,
#     'stim_type': 'current_injection',
#     'amplitude': 52.0,
#     'amplitude_unit': 'uA/cm2',
#     'target': 'intracellular',
#     'protocol': {
#         'duration': 1.0,
#         'start_time': 0.0,
#         'frequency': 2.0,
#         'bcl': 500.0,
#         'num_pulses': 10
#     }
# }

# Get boolean masks for simulation
masks = session.get_all_masks()
# {'S1_pacing': np.array([[True, False, ...], ...]),
#  'S2_premature': np.array([...])}

# Get pulse times
region = session.get_region_by_label("S1_pacing")
times = region.protocol.get_pulse_times()
# [0.0, 500.0, 1000.0, 1500.0, ...]
```

---

## SVG Loading (draw.io)

For draw.io SVG files, use cairosvg with preprocessing:

```python
import cairosvg
import re
from PIL import Image
from io import BytesIO
import numpy as np

def clean_svg_for_cairo(svg_content: str) -> str:
    """Remove CSS that cairosvg can't parse."""
    cleaned = re.sub(r'light-dark\([^)]+\)', '#000000', svg_content)
    cleaned = re.sub(r'style="[^"]*fill:\s*light-dark[^"]*"', '', cleaned)
    cleaned = re.sub(r'style="[^"]*var\(--[^"]*"', '', cleaned)
    return cleaned

def load_svg(path: str, width: int = 1000, height: int = 1000):
    """Load draw.io SVG and convert to numpy array."""
    with open(path, 'r') as f:
        svg_content = f.read()

    cleaned = clean_svg_for_cairo(svg_content)
    png_data = cairosvg.svg2png(
        bytestring=cleaned.encode(),
        output_width=width,
        output_height=height
    )

    img = Image.open(BytesIO(png_data))
    return np.array(img), img.size
```

### draw.io Export Settings

For best results when exporting from draw.io:

| Setting | Value | Why |
|---------|-------|-----|
| Background | Transparent | Enables alpha-based anti-aliasing cleanup |
| Border | 0 | No extra padding |
| Stroke | None | Avoids RGB blending at edges |

---

## Complete Example

```python
import cairosvg
import re
from PIL import Image
from io import BytesIO
import numpy as np

from Builder.MeshBuilder import MeshBuilderSession
from Builder.StimBuilder import StimBuilderSession

# --- Helper ---
def load_svg(path):
    with open(path, 'r') as f:
        content = f.read()
    cleaned = re.sub(r'light-dark\([^)]+\)', '#000000', content)
    cleaned = re.sub(r'style="[^"]*fill:\s*light-dark[^"]*"', '', cleaned)
    png = cairosvg.svg2png(bytestring=cleaned.encode(), output_width=1000, output_height=1000)
    img = Image.open(BytesIO(png))
    return np.array(img), img.size

# --- MeshBuilder ---
arr, size = load_svg("tissue.drawio.svg")

mesh_session = MeshBuilderSession()
mesh_session.image_array = arr
mesh_session.image_size = size
mesh_session.threshold_transparency(128)
mesh_session.filter_small_groups(0.1)

# Configure tissue
mesh_session.configure_group(
    color=(0, 0, 0, 255),
    label="Myocardium",
    cell_type="ventricular",
    D_xx=0.001, D_yy=0.0003
)

mesh_session.set_dimensions(5.0, 5.0, 0.01)
print(mesh_session.summary())

# --- StimBuilder ---
arr, size = load_svg("stim.drawio.svg")

stim_session = StimBuilderSession()
stim_session.image_array = arr
stim_session.image_size = size
stim_session.threshold_transparency(128)

# Configure pacing
stim_session.configure_current_injection(
    color=(255, 0, 0, 255),
    label="Pacing_site",
    amplitude=52.0,
    bcl=500.0,
    num_pulses=20
)

print(stim_session.summary())

# Export for simulation
stim_configs = stim_session.get_stim_config()
stim_masks = stim_session.get_all_masks()
```

---

## API Reference

### MeshBuilderSession

| Method | Description |
|--------|-------------|
| `load_image(path)` | Load image file |
| `detect_colors()` | Find unique colors |
| `threshold_transparency(threshold=128)` | Binarize alpha |
| `filter_small_groups(min_percent=0.1)` | Remove artifacts |
| `configure_group(color, label, cell_type, D_xx, D_yy, D_xy=0)` | Configure tissue |
| `set_dimensions(width, height, dx)` | Set physical size |
| `mark_as_background(color)` | Exclude from mesh |
| `mark_as_tissue(color)` | Include in mesh |
| `get_color_groups()` | Get all groups |
| `get_mesh_resolution()` | Get (nx, ny) |
| `summary()` | Get text summary |

### StimBuilderSession

| Method | Description |
|--------|-------------|
| `load_image(path)` | Load image file |
| `detect_colors()` | Find unique colors |
| `threshold_transparency(threshold=128)` | Binarize alpha |
| `filter_small_groups(min_percent=0.1)` | Remove artifacts |
| `configure_region(color, label, stim_type, amplitude, ...)` | Full config |
| `configure_current_injection(color, label, amplitude, ...)` | Current stim |
| `configure_voltage_clamp(color, label, voltage, ...)` | Voltage clamp |
| `mark_as_background(color)` | Exclude from stim |
| `get_stim_regions()` | Get all regions |
| `get_region_mask(color)` | Get boolean mask |
| `get_all_masks()` | Get all masks |
| `get_stim_config()` | Export for simulation |
| `get_region_by_label(label)` | Find region |
| `summary()` | Get text summary |
