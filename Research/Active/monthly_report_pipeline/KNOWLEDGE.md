# Monthly Report Pipeline — Knowledge File

> Running synthesis. Updated as findings accumulate.
> When this question is complete, a copy is promoted to `Research/Knowledge/`.

## Current Understanding
Starting investigation. Goal: an AI-assisted pipeline that consolidates monthly project activity into a Zimmerman-format deck.

The user has many parallel work streams (11+ active research questions, 3 engines, 3 pipelines, surrogate sessions, cardiac_ml harness, textbook). Manually tracking what got done across all of them every month is the bottleneck this pipeline is designed to remove.

## John Zimmerman's Deck Guidelines

### Source artifacts (local copies in `MonthlyReport/`)
- **Spec PDF**: `MonthlyReport/Lab Progress Reports.pdf` — "Progress Report Format -V1 – 4/20/2026". Authoritative format document (3 pages).
- **Slide template PPTX**: `MonthlyReport/ZimmermanLab_DefaultSlides.pptx` — lab default slide aesthetic.
- **Source email**: Gmail thread `19dac19bec01e5b3`, "Zimmerman Lab Progress Reports Format", 2026-04-20. Sent to all lab members.
- **Canonical SharePoint folder**: `ZimmermanLab/Shared Documents/Protocols/Lab Welcome Documents/` (Cornell SharePoint — open in browser).
- **Submission target SharePoint folder**: separate "Sharepoint link" referenced in the spec for uploading completed reports under each month — exact URL not captured yet, lives in the spec PDF as a hyperlink.

### Purpose (per the spec — quote key)
Reports serve a **dual purpose**:
1. **Administrative accountability** — formal monthly reporting to track project milestones and researcher status. Required for compliance with sponsored programs (NIH etc.).
2. **Professional & analytical development** — pedagogical tool: translating data into written summaries and slide decks builds skills for formal talks/defenses. Forces deeper interpretation of results, not just running more experiments.

### Cadence (HARD CONSTRAINT)
- **Due**: last Thursday of each month
- **Submitted by**: before working hours the following Friday
- **April 2026**: Thursday 2026-04-30, submit before working hours Friday 2026-05-01
- **(Today 2026-04-28 — first report due in 2 days.)**

### Slide template
Reports must use the lab default template `ZimmermanLab_DefaultSlides.pptx`.
- Slide size: **10.00 × 7.50 in (standard 4:3, NOT widescreen 16:9)**
- 4 layouts available:
  - `Title and Content` — generic content (2 body placeholders)
  - `7_Title Slide` — title-style with 1 body + 1 content placeholder
  - `8_Title Slide` — minimal/cover (no placeholders — likely section divider)
  - `9_Title Slide` — multi-region layout (1 title + content + 5 body placeholders, likely summary-style)
- Placeholders are empty in the template — content is filled in per report.

### Deck structure (REQUIRED)
**Length**: 7–14 slides total. Contains key findings from the last reporting period (typically one month).

| # | Slide | Required content |
|---|-------|------------------|
| 1 | **Title slide** | Researcher's name + specific month and year of reporting period (for archival/longitudinal tracking) |
| 2 | **Summary slide** | High-level overview of the month. Each active project = primary bullet, with 1-4 sub-bullets of key takeaways previewing the detail slides. Optional "general lab activities" section for non-project work that took significant time (site visits, peer training, grant prep) but doesn't warrant its own slide. |
| 3..N-1 | **Research slides** (core of the deck) | One per experiment/task. See "Research slide rules" below. |
| N | **Future Outlook slide** | Strategic mirror of the Summary slide. Primary bullets per ongoing project. Upcoming milestones, experimental goals, specific outcomes hoped for. Aligns immediate tasks with long-term objectives. |

### Research slide rules (the strict ones)
- **Each slide MUST have a clear stated objective** for the experiment/task
- **Standalone-readable**: no verbal explanation should be needed. Concise bullet/caption for every image, chart, diagram.
- **Every dataset gets ≥1 sentence of significance/interpretation** — analytical rigor
- **Encouraged**: "main takeaway" in a **grey highlighted box at the bottom** of the slide for scannability
- **Videos**: must be explicitly indicated as video (won't play in PDF)
- **Visual data**: must include clear scale bars
- **Notional/expected data MUST BE CLEARLY MARKED AS SUCH** to distinguish from empirical findings (the spec capitalizes this, treat as hard rule)

### Submission protocol (two parallel actions)
1. **Upload** completed PPTX to the SharePoint folder under the appropriate month
2. **Email** the PI (`john.f.zimmerman@cornell.edu`) with this exact format:

| Field | Format |
|-------|--------|
| Subject | `LastName – Progress report – Month, Year` (e.g. `Chang – Progress Report – April, 2026`) |
| Body | Hello John,<br>Attached PDF copy of progress report for Month Year. In summary:<br>• 3-4 bullets — most important progress this month<br>• 2-3 bullets — future research plans<br>(Sharepoint link to PPTX)<br>Best Regards,<br>Name |
| Attachment | `.pdf` of the report (mobile-friendly companion to the PPTX) |

### Open questions (not in the spec)
- How to surface side projects (textbook, cardiac_ml harness) — under "general lab activities" on summary slide, or as own research slides if substantive that month?
- Does `8_Title Slide` (no-placeholder layout) function as a section divider between projects in long reports?
- Submission SharePoint URL — captured as link in the PDF, not as raw text. Need to extract.

## Pipeline Architecture (planned, not finalized)
Working sketch — refined in IDEALOG before being committed here.

```
/monthly  (entry-point, reason-style interactive orchestrator)
   ├── /monthly-scout      — sweep git log + PROGRESS + IDEALOG, per active question
   ├── /monthly-consolidate — uniform per-question monthly summary
   ├── /monthly-assemble    — produce deck matching Zimmerman format
   └── /monthly-imagery     — flag missing figures/proofs
```

Output lives at `Monthly_Notebook/{YYYY-MM}/`:
- `scout.md` — raw activity dump
- `consolidated.md` — organized per-question summary
- `deck.{ext}` — final deliverable in Zimmerman format
- `imagery_gaps.md` — figures we should make but didn't

## Key Decisions
None yet — to be recorded here as the pipeline design settles.

## Open Questions
- Does Zimmerman want progress on side projects (textbook, cardiac_ml) included or only "headline" research questions?
- How does git history map to "what I did" when most work is across many small commits?
- Should the deck assembly step write directly to a slide format, or produce a markdown intermediate that the user manually pastes into slides?
- Where do imagery gaps get tracked across months — does the gap list persist or reset monthly?

## Connections
- **Engines**: All (consumed read-only via PROGRESS.md and git log)
- **Related research**: `research_environment_optimization` (Claude Code workflow tooling — adjacent but distinct: that's about AI-assisted development; this is about reporting deliverables)
- **Pipelines**: None (this is a new pipeline of its own)
- **Tooling**: Gmail MCP (for Zimmerman's email), git, file system scans
