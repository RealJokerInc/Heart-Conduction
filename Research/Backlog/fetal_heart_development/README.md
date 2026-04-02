# Fetal Heart Development

## Question
How does the cardiac conduction system develop in the fetal heart, and how do developmental principles inform our simulation models?

## Status: Backlog

## Why It Matters
The transition from immature to mature cardiomyocytes (gap junction expression changes, IK1 upregulation) parallels the hiPSC-CM maturation pathway (PHAS13 → MHAS13). Understanding developmental biology validates our maturation approach and informs how tissue geometry affects conduction during growth.

## Trigger to Activate
When boundary speedup is validated in 3D geometry.

## Literature
See `literature/` for detailed reviews:
- `fetal_heart_development_literature.md` — 14-category review (200+ PubMed citations)
- `ml_in_fetal_heart_development.md` — ML applications in developmental cardiology

## Connected Research
- **boundary_conduction_speedup** — Same boundary physics applies during tissue growth
- **scar_bc_validity** — Gap junction absence at scar mirrors developmental patterns
- **mature_hipsc_cm_models** — hiPSC-CMs recapitulate immature/fetal phenotype
