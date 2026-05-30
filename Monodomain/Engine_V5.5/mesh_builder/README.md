# MeshBuilder

A mesh generation tool that converts image-based designs into simulation-ready meshes for **engine_V5.4**.

## Overview

MeshBuilder provides an intuitive workflow for creating cardiac tissue meshes by reading pixel data from images and generating meshes with:
- Defined borders/boundaries
- Fiber alignment information

## Components

### 1. Image-to-Mesh Builder
- **Input**: Image files (.jpg, .png, or other standard formats)
- **Process**: Reads pixel data and interprets colors based on defined protocols
- **Output**: Mesh data compatible with engine_V5.4 pipeline
  - FEM triangular mesh (nodes + triangle connectivity)
  - Structured grid (for FDM/FVM solvers)

#### Dimension Handling

The engine accepts arbitrary input dimensions. The builder translates images to meshes based on:
- **Tissue dimensions**: Physical size of the target tissue (user-specified)
- **dx**: Spatial resolution / grid spacing (user-specified)
- **Value customization**: Fine-tune parameters after initial generation

#### Scaling Behavior

| Scenario | Input Resolution | vs | Tissue Dim Resolution | Behavior |
|----------|------------------|----|-----------------------|----------|
| Size up  | Higher           | >  | Lower                 | Supported - interpolate to target resolution |
| Size down| Lower            | <  | Higher                | *Not implemented* - preserve original coarse borders |

**Current scope**: Sizing up only. When input resolution exceeds target mesh resolution, the builder downsamples appropriately. The reverse (upsampling from coarse input) is deferred to preserve rigid border integrity.

### 2. Color Protocols

Each distinct color in the input image represents a unique cell group.

#### Workflow
1. **Identify**: Builder auto-detects all distinct colors in the image
2. **Label**: User assigns meaningful names to each color group
3. **Specify**: User defines properties per group:
   - **Conductivity tensor** (full 2x2 tensor per cell)
   - **Cell types**
   - *(additional properties as needed)*

#### Conductivity Tensor Format

Stored as **full 2x2 conductivity tensor D per cell**:
```
D = | D_xx  D_xy |
    | D_xy  D_yy |
```

#### Example Session

Builder detects 3 distinct colors in image → user assigns in session:

| Detected Color | User Label | Cell Type | D_xx | D_xy | D_yy |
|----------------|------------|-----------|------|------|------|
| Group 1        | Atria      | Atrial    | 0.3  | 0.0  | 0.1  |
| Group 2        | Ventricle  | Ventricular | 0.5 | 0.0 | 0.15 |
| Group 3        | Border     | Non-conducting | 0.0 | 0.0 | 0.0 |

*No predefined color mappings - colors are arbitrary based on user's input image.*

## Interfaces

- **CLI**: Command-line interface for scripting and batch processing
- **UI**: Visual interface for interactive mesh design and preview

Both interfaces will run in parallel, sharing the same underlying engine.

## Status

Project initialization - architecture and protocols under development.
