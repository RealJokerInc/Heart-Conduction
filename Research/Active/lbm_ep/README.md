# LBM-EP

## Question
Build the LBM cardiac electrophysiology engine into a robust, correctly-tuned solver that handles anisotropy and reproduces boundary artifacts (Kleber-style speedup) faithfully.

## Status: Active (reopened 2026-04-19)

> **History.** Previously completed as `lbm_cardiac` on 2026-03-16 (8 phases, 34 tests, D2Q5/D2Q9, BGK/MRT — see KNOWLEDGE.md). Reopened to push past initial validation into engine maturation: anisotropy correctness, boundary artifact modeling, and tuning.

## Why It Matters
LBM V1 currently passes its baseline tests but has known gaps: ~35% CV offset vs FDM, anisotropy validated only on simple cases, boundary speedup unverified on D2Q9 + Dirichlet. To use LBM as a serious cardiac solver (and as a candidate for the Surrogate's diffusion stage), it needs to match Bidomain ground truth across anisotropic and boundary-driven regimes.

## Engines
- **LBM V1** (`LBM/Engine_V1/`) — primary engine under development
- **Bidomain V1** (`Bidomain/Engine_V1/`) — ground truth for CV / boundary comparisons
- **Monodomain V5.4** — secondary FDM ground truth

## Completion Criteria
- [ ] (TBD — refined as work progresses; user opted to start without locking criteria)

Working set:
- [ ] Anisotropy validation: D2Q9 + MRT recovers a known anisotropic conductivity tensor
- [ ] Boundary speedup: reproduce theoretical Kleber ratio (~1.131) within tolerance
- [ ] Tuning protocol: documented procedure for choosing tau / MRT relaxation rates given target conductivity
- [ ] Test count growth from 34 → target TBD

## Sub-Questions

| Sub-Question | Status | Key Finding |
|-------------|--------|-------------|
| (none yet) | — | — |

## Key Findings So Far
See KNOWLEDGE.md for the foundation built during the original lbm_cardiac question (D2Q5/D2Q9 architecture, BGK vs MRT, bidomain feasibility).

## Literature
See `literature/` for paper summaries. Key files:
- `LBM_BIDOMAIN.md` — 3 bidomain architectures, coupling strategies
- `04_LBM_EP_Implementation.md` — LBM-EP algorithm details
- `SUMMARY.md` — initial findings

| Paper | Summary | Key Insight |
|-------|---------|-------------|
| Rapaka 2012 | LBM-EP Siemens framework | 3D cardiac, Mitchell-Schaeffer ionic |
| Campos 2016 | GPU LBM monodomain | 10-45x over FEM |
| Belmiloudi 2015-2019 | Coupled LBM bidomain | Dual-lattice + hybrid LBM-FV |
| Chai & Zhao 2012 | Multigrid LBM elliptic | Pseudo-time acceleration |

## Future Work
- Bidomain LBM (architecture A: dual-lattice) — deferred until monodomain LBM is fully tuned
- 3D extension (D3Q7 / D3Q19) once 2D anisotropy is validated

## Connected Research
- **[bidomain_parabolic_parabolic](../bidomain_parabolic_parabolic/)** — **endgame target.** The "dual-evolving bidomain LBM" direction is the Cattaneo hyperbolic bidomain of Rossi & Griffith 2017 (τ_i ≠ τ_e). The LBM–Cattaneo correspondence (BGK's Chapman-Enskog naturally recovers Cattaneo flux dynamics) is documented there. See [rossi_2017_hyperbolic_bidomain](../bidomain_parabolic_parabolic/literature/rossi_2017_hyperbolic_bidomain.md) for the formulation and [KNOWLEDGE.md §"LBM–Cattaneo Correspondence"](../bidomain_parabolic_parabolic/KNOWLEDGE.md) for why dual-lattice LBM is the natural discretization.
- **boundary_conduction_speedup** — uses LBM as a tool to study Kleber effect
- **engine_consolidation** — LBM V1 chosen as canonical LBM
- **geometry_induced_pacemaking** — uses LBM in geometry experiments
