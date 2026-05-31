# Mesh Builder

## Question
How do we let a user quickly sketch tissue geometry in a graphical tool and convert the drawing into a simulation-ready mesh — with labeled boundaries, per-region conductivity, and stimulus regions — that feeds directly into all active engines (Bidomain V1, Monodomain V5.4, LBM V1)?

## Status: Active

## Why It Matters
Geometry authoring is currently the slowest step in the draw→mesh→simulate loop. Every new tissue shape (hiPSC patterns for pacemaking, Kleber-boundary geometries, scar layouts, optimizer/surrogate training inputs) needs a mesh with labeled boundaries and per-region properties. The current `Builder/` package handles image→mesh at a low level, but there's no fast drawing front-end and no uniform handoff into the three active engines. A good mesh-builder pipeline unblocks all geometry-driven research questions and is prerequisite for data generation at scale.

## Engines
- **Build in isolation first** — standalone tool, not engine-coupled.
- **Bidomain V1** — consume mesh + boundary labels (Neumann / Dirichlet / bath-coupled, possibly mixed).
- **Monodomain V5.4** — consume mesh + conductivity regions + stimulus regions via existing Builder integration.
- **LBM V1** — consume mesh + lattice-compatible boundary tags (bounce-back / anti-bounce / absorbing).

Engine flow modifications are expected — the goal is *one* canonical mesh format that all three engines accept.

## Completion Criteria
- [ ] Drawing tool UX decided (extend current `Builder/` UI, reuse external software, or build new)
- [ ] Canonical mesh format specified (geometry + boundary labels + region properties + stimulus)
- [ ] Boundary labeling strategy decided (stacking vs. alternatives — see sub-questions)
- [ ] Prototype pipeline: draw → export → load into **one** engine end-to-end
- [ ] Pipeline extended to all three active engines
- [ ] Engine flow modifications documented and landed
- [ ] Round-trip validation: authored geometry reproduces expected simulation behavior (CV, boundary artifacts, stimulus firing)

## Sub-Questions

| Sub-Question | Status | Key Finding |
|-------------|--------|-------------|
| Boundary labeling (stacking strategy) | Pending — to be scaffolded when work starts | — |

## Key Findings So Far
*Empty — scaffolding only.*

## Literature
| Paper | Summary | Key Insight |
|-------|---------|-------------|
| *empty* | | |

## Engine References

| File | Purpose |
|------|---------|
| `Builder/README.md` | Existing image→mesh package (V5.4-targeted) |
| `Builder/BACKEND.md` | Backend API reference |
| `Builder/MeshBuilder/session.py` | Current MeshBuilderSession implementation |
| `Builder/StimBuilder/session.py` | Current StimBuilderSession implementation |
| `Builder/MeshLibrary/` | Existing geometry assets |
| `Monodomain/Engine_V5.4/` | Target engine with existing Builder integration |
| `Bidomain/Engine_V1/` | Target engine (boundary-condition sensitive) |
| `LBM/Engine_V1/` | Target engine (lattice-boundary sensitive) |
| `Research/Active/engine_consolidation/README.md` | Related — canonical mesh format is a shared-code concern |

## Future Work
*No deferred items yet.*
