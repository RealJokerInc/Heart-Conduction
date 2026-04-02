# Q1: How do I discretize the cardiac PDEs on a computational mesh?

## Short Answer

Three methods: **FEM** (unstructured meshes, complex geometries), **FDM** (structured grids, 9-point stencils with harmonic averaging), and **FVM** (cell-centered, conservative). For isotropic tissue on rectangular domains, FDM is simplest and fastest. For patient-specific geometries, FEM on triangular/tetrahedral meshes. FVM bridges the gap with natural conservation properties.

Boundary conditions are face-based: Neumann (no-flux/insulated) at tissue edges, Dirichlet (bath-coupled) where extracellular space connects to a grounded bath.

## Key Files in This Folder

| File | Contents |
|------|----------|
| `BIDOMAIN_DISCRETIZATION.md` | Comprehensive FEM/FDM/FVM guide (1230 lines, 28 references) |
| `01_FDM_Stencils_and_Implementation.md` | FDM stencil construction for cardiac monodomain |
| `02_openCARP_FDM_FVM_Architecture.md` | openCARP system design and data structures |
| `05_Bidomain_Discretization.md` | FEM/FDM/FVM specifically for bidomain equations |
| `Summary_02_Discretization_Methods.md` | Comparison table of spatial methods |
| `00_START_HERE.txt` | Navigation guide with reading paths by role |
| `00_Research_Summary.md` | Overview of downloaded reference repos |

## Relevant Papers

See `../papers/` for PDFs. Key references:
- Plank et al. — openCARP FDM/FVM architecture
- Sundnes et al. — FEM for cardiac EP

## Connected Questions

- **Q2** — After discretization, how to solve the resulting linear system
- **Q5** — Boundary discretization affects conduction velocity near tissue edges
