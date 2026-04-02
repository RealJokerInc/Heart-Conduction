# Scar Boundary Condition Validity

## Question
Are Dirichlet boundary conditions at scar boundaries physically valid?

## Status: Complete (2026-03-16)

## Key Answer
**No.** Scar tissue is electrically inert (no ion channels, no gap junctions to viable myocardium). The correct BC is **Neumann (no-flux)**, not Dirichlet (voltage clamping). Dirichlet at scar implies the scar acts as a voltage source, creating unphysical current injection that confounds CV measurements and optimization objectives.

Key distinction: tissue-bath interfaces DO have asymmetric BCs (Neumann intracellular, Dirichlet extracellular) which produces the Kleber speedup. Tissue-scar interfaces have symmetric Neumann BCs on both domains — NO speedup.

## Engines
- **Bidomain V1**: Validates that Neumann at scar gives no speedup
- **LBM V1**: Bounce-back BC implements no-flux correctly

## Literature
See `literature/` for detailed analysis. Key files:
- `PROBLEM_STATEMENT.md` — ML-DO framework and artifact identification
- `ARGUMENT_AGAINST_AND_EMI.md` — Physics critique
- `WHY_NEUMANN_DIRICHLET.md` — Mathematical justification
- `BC_COMBINATION_ANALYSIS.md` — Dirichlet/Neumann/Robin evaluation
- `KLEBER_ARGUMENT_CHAIN.md` — Connection to Kleber safety factor

## Connected Research
- **boundary_conduction_speedup** — The speedup that Dirichlet incorrectly produces at scar
- **lbm_cardiac** — Bounce-back BC implementation for no-flux
