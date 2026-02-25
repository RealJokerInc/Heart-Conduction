# Builder UI - Design Document

## Overview

A Flask-based web interface for the MeshBuilder pipeline. Converts tissue design images into simulation-ready mesh configurations through an intuitive dark-mode UI.

**Tech Stack:** Flask (Python) + HTML/CSS/JS (vanilla)

**Run Command:**
```bash
cd "/Users/catecholamines/Documents/Heart Conduction"
./venv/bin/python -m Builder.ui.server
# Opens at http://localhost:5001
```

---

## User Flow

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────────┐
│  Start  │ ──▶ │ Upload  │ ──▶ │ Loading │ ──▶ │  Workspace  │
│  Page   │     │  Page   │     │  Screen │     │    Page     │
└─────────┘     └─────────┘     └─────────┘     └─────────────┘
```

---

## Design Choices

### 1. Dark Mode Theme

**Decision:** Dark background (#1a1a2e) with purple accent (#9d4edd)

**Rationale:**
- Reduces eye strain during extended use
- Better contrast for viewing medical/scientific images
- Modern, professional appearance
- Blueprint-style grid background evokes engineering/CAD tools

### 2. Tissue Classification: Conductive vs Non-Conductive

**Decision:** Binary classification system instead of multiple preset categories

**Rationale:**
- Simplifies mental model for users
- Maps directly to simulation physics (conducts electricity or doesn't)
- Conductive tissues share visual representation (dashed white pattern)
- Non-conductive tissues retain their original colors for visual distinction

**Categories:**
| Type | Presets | Visual | Conductivity |
|------|---------|--------|--------------|
| **Conductive** | Myocardial, Endocardial, Epicardial | Dashed white | D > 0 |
| **Non-Conductive** | User-named (default: "Infarct Group N") | Original color | D = 0 |

### 3. Auto-Detection of Background

**Decision:** White (255,255,255) and transparent (alpha=0) pixels auto-initialize as "Myocardial"

**Rationale:**
- Most tissue designs use white/transparent for the main conductive medium
- Reduces manual configuration steps
- Myocardial is the most common default tissue type
- Users can still re-classify if needed

### 4. Dashed White Pattern for Conductive Tissue

**Decision:** All conductive tissues display as diagonal white stripes (45°) instead of their original color

**Rationale:**
- Visually unifies all conductive regions
- Clearly distinguishes from non-conductive (colored) regions
- Pattern suggests "flow" or "conductivity"
- Consistent with engineering/schematic conventions

**Implementation:**
```css
background: repeating-linear-gradient(
    45deg,
    #ffffff,
    #ffffff 2px,
    #9a8c98 2px,
    #9a8c98 4px
);
```

### 5. Group Numbering Persistence

**Decision:** Each detected color group receives a fixed `group_number` at detection time

**Rationale:**
- "Unlabeled Group 3" always defaults to "Infarct Group 3"
- Prevents confusion when configuring groups out of order
- Number persists regardless of configuration sequence
- Makes it easy to reference specific groups

### 6. Two-Button Dropdown UI

**Decision:** Click group → [Conductive] [Non-Conductive] buttons → reveal options

**Rationale:**
- Forces explicit classification decision
- Prevents accidental mis-classification
- Green/red color coding provides instant visual feedback
- Expandable design keeps UI clean when collapsed

### 7. Blueprint-Style Image Background

**Decision:** Dark blue (#0a1628) with grid overlay for the image panel

**Rationale:**
- Evokes engineering blueprints / CAD software
- Grid helps visualize scale and alignment
- Dark background provides contrast for light tissue images
- Professional, technical appearance

---

## File Structure

```
Builder/ui/
├── server.py              # Flask application & API endpoints
├── DESIGN.md              # This document
├── REQUIREMENTS.md        # Original requirements spec
├── templates/
│   ├── base.html          # Base template with dark theme
│   ├── start.html         # Landing page
│   ├── upload.html        # Drag-drop file upload
│   ├── loading.html       # Processing progress screen
│   └── workspace.html     # Main configuration interface
└── static/
    ├── css/
    │   └── style.css      # Dark theme styles
    └── js/
        └── app.js         # Utility functions
```

---

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Start page |
| GET | `/upload` | Upload page |
| GET | `/loading` | Loading/processing page |
| GET | `/workspace` | Main workspace |
| POST | `/api/upload` | Handle file upload (PNG, JPG, SVG) |
| POST | `/api/process` | Process image, detect colors |
| GET | `/api/session` | Get current session data (groups, dimensions) |
| POST | `/api/configure` | Configure a group (conductive/non-conductive) |
| GET | `/api/image` | Get processed image with patterns applied |
| GET | `/api/image/highlight/<index>` | Get image with specific group highlighted |
| POST | `/api/reset` | Clear session, start over |

---

## Data Model

### Group Object
```python
{
    'index': int,              # Position in detection order
    'color': [r, g, b, a],     # Original RGBA color
    'hex': '#RRGGBB',          # Hex representation
    'pixel_count': int,        # Number of pixels
    'label': str,              # Display name
    'preset': str | None,      # 'myocardial', 'endocardial', etc.
    'configured': bool,        # Has been classified
    'is_conductive': bool,     # Conducts electricity
    'is_non_conductive': bool, # Explicitly non-conductive
    'group_number': int | None # Fixed number for naming
}
```

### Conductive Presets
```python
{
    'myocardial': {
        'label': 'Myocardial',
        'cell_type': 'myocardial',
        'D_xx': 0.001,    # cm²/ms
        'D_yy': 0.0003,
        'D_xy': 0.0,
    },
    'endocardial': {
        'label': 'Endocardial',
        'cell_type': 'endocardial',
        'D_xx': 0.002,
        'D_yy': 0.0006,
        'D_xy': 0.0,
    },
    'epicardial': {
        'label': 'Epicardial',
        'cell_type': 'epicardial',
        'D_xx': 0.0012,
        'D_yy': 0.00036,
        'D_xy': 0.0,
    },
}
```

### Non-Conductive Configuration
```python
{
    'cell_type': 'non_conductive',
    'D_xx': 0.0,
    'D_yy': 0.0,
    'D_xy': 0.0,
}
```

---

## Color Detection Logic

1. **Load image** → Convert to numpy array (RGBA)
2. **Detect unique colors** → Group pixels by exact color match
3. **Filter noise** → Remove groups < 0.1% of total pixels
4. **Auto-classify:**
   - White `(255,255,255,*)` → Myocardial (conductive)
   - Transparent `(*,*,*,0)` → Myocardial (conductive)
   - All others → Unlabeled Group N

---

## Image Rendering

### Base Image (`/api/image`)
1. Start with original image
2. For each `is_conductive` group:
   - Create mask matching that color (full RGBA comparison)
   - Apply diagonal stripe pattern to masked pixels
   - Set alpha to 255 (fully opaque)
3. Return as PNG

### Highlighted Image (`/api/image/highlight/<index>`)
1. Start with original image
2. Apply conductive patterns (same as base)
3. Create highlight mask for target group
4. Dim all non-highlighted pixels to 30% brightness
5. Return as PNG

**Key Detail:** Color matching uses full RGBA comparison for RGBA images to prevent transparent pixels `(0,0,0,0)` from matching opaque black `(0,0,0,255)`.

---

## UI Components

### Left Panel: Tissue Groups
- Lists all detected color groups
- Shows: color swatch + label + status (✓ if configured)
- Conductive groups: dashed white swatch
- Non-conductive groups: original color swatch
- Click to expand configuration dropdown
- Hover to highlight in image

### Right Panel: Image Visualization
- Blueprint-style grid background
- Displays tissue image with patterns applied
- Dimensions shown above: W, H, dx, Grid resolution
- Hover highlighting synced with left panel

### Configuration Dropdown
```
┌─────────────────────────────────┐
│ [Conductive]  [Non-Conductive]  │  ← Two-button selector
├─────────────────────────────────┤
│ ○ Myocardial                    │  ← Conductive options
│ ○ Endocardial                   │
│ ○ Epicardial                    │
├─────────────────────────────────┤
│ Name: [Infarct Group 1____]     │  ← Non-conductive input
│ [Confirm]                       │
└─────────────────────────────────┘
```

---

## Theme Colors

```css
--bg-primary: #1a1a2e;      /* Main background */
--bg-secondary: #16213e;    /* Panels */
--bg-dark: #0f0f1a;         /* Darkest elements */
--accent: #9d4edd;          /* Purple accent */
--accent-hover: #7b2cbf;    /* Darker purple */
--text-primary: #e8e8e8;    /* Main text */
--text-secondary: #9a8c98;  /* Muted text */
--border: #4a4e69;          /* Borders */
--success: #22c55e;         /* Green (conductive) */
--error: #ef4444;           /* Red (non-conductive) */
```

---

## Known Limitations

1. **Session Storage:** In-memory only, lost on server restart
2. **Export:** Not yet implemented (placeholder)
3. **Dimensions:** Display only, not yet editable in UI
4. **Multi-user:** Sessions stored by cookie, no authentication

---

## Future Enhancements

1. [ ] Export to .npz format
2. [ ] Editable tissue dimensions dialog
3. [ ] Session save/load to file
4. [ ] Undo/redo support
5. [ ] Stimulus map builder (StimBuilder integration)
6. [ ] Custom conductivity values per group
