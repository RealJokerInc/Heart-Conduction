# Monthly Report Pipeline — Idea Log

> Thinking trail: how our understanding evolved, what we tried, what failed.
> Scan in 30 seconds to remember where we are and how we got here.
> Not promoted on completion — archived for historical record.

## Current Direction
Build a small pipeline of skills that consolidates monthly project activity (git history per active question, PROGRESS, IDEALOG, KNOWLEDGE) into a Zimmerman-format presentation deck. Entry point is reason-style interactive — the user invokes `/monthly` and it walks them through scout → consolidate → assemble → imagery-audit. Tool output at `MonthlyReport/` at repo root (folder already exists with the source spec PDF and template PPTX).

## Next Step
April test run **CLOSED** (submitted 2026-05-01, PI replied 2026-05-02 with "Good job overall" + 11 numbered comments). 24 friction observations captured (last 6 from PI feedback round-trip on 2026-04-30 session).

Before May report (due 2026-05-28): design `/monthly` pipeline architecture from the 24 friction observations as concrete requirements, then `/blueprint` to generate PLAN.md. Calibration anchor: April deck = floor for what /monthly should produce.

Open PI-supplied V2 direction (John's comment #11): train v3/v4 surrogate on ORd or richer model, evaluate whether simple surrogate retains rich-model features at lower cost. Belongs in `Surrogate/` IDEALOG, not this one.

## Thread

**2026-04-28** — Question scaffolded. User has 11+ active research questions and many parallel side projects; manual monthly reporting is the friction point. Pipeline is small/short but high-leverage for AI-assisted workflow. Confirmed top-level question, not a sub of `research_environment_optimization` because deliverable audience is the PI, not the Claude Code workflow.

## Failed Approaches
None yet.

## Session Log

**2026-04-28** — Scaffolded the question. Identified first step: extract Zimmerman's deck guidelines from email.

**2026-04-28** — Located canonical email via Gmail MCP: thread `19dac19bec01e5b3` "Zimmerman Lab Progress Reports Format" (2026-04-20). Email body is a cover note — the actual format spec is the **PPTX attachment** ("progress report format slide") and the lab default template `ZimmermanLab_DefaultSlides.pptx` on Cornell SharePoint. The Gmail MCP available here does not expose an attachment-download tool, and SharePoint is not in the Drive MCP scope. **Blocker**: need user to export the attached slide content (paste text + describe layout, or save as image and share via the `Heart-Conduction-Claude-Shared` Drive folder). Recorded source pointers in KNOWLEDGE.md under "John Zimmerman's Deck Guidelines".

**2026-04-28** — User dropped both files locally in `MonthlyReport/` at repo root. Format turned out to be a **3-page PDF** ("Lab Progress Reports.pdf"), not a single slide as I'd assumed from the email phrasing. Read it directly. Spec is comprehensive and prescriptive: 7-14 slides, fixed structure (Title / Summary / Research / Future Outlook), strict per-slide rules (objective, captions, ≥1 sentence interpretation, optional grey takeaway box, scale bars, notional-data marking), specific email submission format. Cadence: **last Thursday of month**, submit before working hours next Friday. PPTX template: 4 layouts at 10×7.5in (4:3, not widescreen) — empty placeholders, just provides aesthetic. Full spec captured in KNOWLEDGE.md. **Step 1 of completion criteria DONE**.

**2026-04-28** — Discovered urgent timing: today is Tuesday 2026-04-28. Last Thursday of April is **2026-04-30 (2 days from now)**. Submit before working hours Friday 2026-05-01. So the very first monthly report under this format is due this week. Pipeline can't be built in 2 days — need to decide whether to rush a manual April report using the spec as a checklist, or treat May as the first pipeline-produced report.

**2026-04-28** — Decision: rush April manually, use it as test run for the pipeline. User asked me to log workflow friction in this document as we go.

### Test-run scout pass (manual `/monthly-scout` simulation)

Executed three parallel Bash queries: (1) `git log --since/--until --pretty=format` for April commits, (2) `grep -B1 -A3 "2026-04"` over every `Research/Active/*/IDEALOG.md`, (3) recently-modified `PROGRESS.md` finder.

**Friction observation 1 — IDEALOG grep is too coarse**: A simple month-string grep pulled ~250 lines of mixed-resolution content. Some entries are full multi-paragraph essays (e.g., surrogate Session 29 hybrid pivot rationale); others are one-line scaffold notes. The pipeline needs a structured IDEALOG section parser, not regex — IDEALOG entries already use date-headed `### YYYY-MM-DD` blocks; pipeline should split on those headers and tag each block with its parent question.

**Friction observation 2 — git log alone misses non-code work**: cardiac_ml has 80 tests and 11 commits in April but the *narrative* (4 audit rounds, blueprint-revise passes, parity threshold raised after Step 4.0 reality check) lives in IDEALOG. Conversely, boundary_conduction_speedup has heavy IDEALOG activity but commits are sparse (`simulation/` lives at repo root, not in a research folder). Lesson: **git log and IDEALOG are complementary inputs, not redundant** — pipeline must consume both, deduplicate by date.

**Friction observation 3 — auto-summarization is risky**: Key insights are nuanced and easy to corrupt by paraphrase ("v4 failed because it overfits" loses "v3 at 1,444 params worked on same split"). Pipeline should preserve user-written headlines from IDEALOG `## Current Direction` and dated session-summary blocks verbatim, not re-summarize.

**Friction observation 4 — inactive question detection useful**: 5 of 11 questions had zero April activity. The pipeline auto-flagging dormant questions is itself valuable — surfaces ones that should be parked or closed. Currently I noted this manually; should be automated.

**Friction observation 5 — repo-root code belongs to projects but isn't auto-discoverable**: `simulation/` belongs to boundary_conduction_speedup but git log doesn't tell you that. Pipeline needs a manual or convention-based "this folder belongs to question X" mapping. Project memory entry exists (`project_storage_tank_harness.md`) — pipeline should consume the `memory/` directory as a third input alongside git + IDEALOG.

**Friction observation 6 — many parallel projects → triage step is mandatory**: 7 candidate projects emerged from scout. Zimmerman wants 7-14 slides total, Summary slide bullets each project. Need a curation gate where the user picks which become "active projects" vs "general lab activities" vs omitted. This is judgment-heavy and probably stays human-in-the-loop.

Scout written to `MonthlyReport/2026-04/scout.md`. Next: present curated inventory to user → user decides project priority → draft slide outline.

**2026-04-28** — User extended reporting period to March on the grounds that "we can cheat a little" — reasonable since the spec says "typically one month" not "exactly one month" and this is the first report under V1 format.

**Friction observation 7 — period extension changes everything**: Re-running the scout for March nearly doubled the project list (added 4 substantial projects: Optimizer V1 + MHAS13, Bidomain V1 hardening / audit / Mehrstellen, Research environment optimization 16-skill build, Engine consolidation Phase 0). Pipeline must support periods as a flag, not hardcode "last month." Also — March work that's already "done" (research_environment_optimization, MHAS13) competes for slide space with April's still-evolving work. Triage criterion can't be "did anything happen?" — it has to be "is this PI-relevant and does it warrant a slide vs a bullet?"

**Friction observation 8 — completed-this-period vs ongoing-direction split**: Some projects (cardiac_ml harness, MHAS13, Bidomain V1 audit) finished within the period — these become "completed work" framing. Others (surrogate, boundary speedup, bidomain PP) are pivoting/evolving — these are "ongoing direction" framing. The Future Outlook slide treats them differently. Pipeline should categorize each project on this axis automatically.

**Friction observation 9 — PI-visibility weighting matters**: I went back through Zimmerman's emails this period (paper discussions, "Structure Dependent Speed Up", "Geometry Dependent Action Potential Conduction", "Diffusion Speed Up Simulation", "Storage Tank Code", "V?", "Ionic Surrogate Architecture") to weight which projects to foreground. The boundary speedup is his anchor; the storage tank model is his Apr-24 contribution. This contextual signal is hard to extract from pure git/IDEALOG — pipeline should consume the PI's email thread metadata as a third input alongside git + IDEALOG. Already have Gmail MCP; just need to filter by sender + period.

**Friction observation 10 — gitignored work is invisible**: The textbook lives outside git tracking (or in a separate repo). Without explicit user input, the pipeline cannot know it exists. Pipeline must include a "anything else this period?" prompt to the user as a safety net before finalizing.

**2026-04-28** — User triaged inventory. Ranking: **C > D > A > F > B**. Reasoning given:
- C top priority because **PI's anchor interest** (boundary speedup, storage tank)
- D 100% because grant material context
- A pushed back because **it failed** ("the surrogate pipeline basically failed") — keep on deck for honest reporting but demote slide allocation
- F "maybe consider" — borderline
- B kept but lower priority — completed deliverable, not currently load-bearing for PI

**Friction observation 11 — Three pipeline signals from user's ranking**:
1. **PI-visibility weighting beats commit-volume weighting**. Surrogate has the highest commit count and longest arc but ranked third because it failed; boundary speedup ranks first because John cares about it.
2. **Failures get DEMOTED, not omitted**. User explicitly kept A because Zimmerman's spec demands "deeper interpretation of results and critical evaluation" — failure-with-honest-narrative is a feature of the format, not a bug. Pipeline must offer a "demote but keep" tier, not just "include / exclude."
3. **Ranking is ordinal, not cardinal**. User said "C, D, A, F, B" without %weights. Pipeline should ask for an ordering, then the slide-allocation algorithm maps order → slide count.

**Slide allocation algorithm (proposed)**: rank-1 gets 2 slides, ranks 2-3 get 1 each, ranks 4+ get 1 if they fit budget else become Summary bullets. With 5 ranks ⟹ 2+1+1+1+1 = 6 research slides + 1 title + 1 summary + 1 future = **9 slides** ✓ in 7-14 budget. Demotion of A from "headline arc" to 1 slide saves 1 slide vs my original recommendation.

**2026-04-28** — User reweighted D toward MHAS13 (model as headline, optimizer as means). Proposed split into 2 slides (D1 model + validation, D2 optimizer + degeneracy). User accepted (implicitly via going forward). Total now 10 slides. Folder renamed `MonthlyReport/2026-04` → `MonthlyReport/April` per user preference.

**Friction observation 12 — figure generation is slow and brittle**: First D1 figure (`D1_F1_ap_comparison_4models.png`) required:
- 30 min to write the script (`make_d1_f1.py` — discovering ABC, fixing 1D shape bug in PHAS13 model.py via 2D-batch workaround per Optimizer convention, testing)
- 6 min runtime (PHAS13 10s freerun + 3 paced runs at dt=0.05ms, all on CPU)
- Latent issue surfaced: baseline MHAS13 APD is much longer than the *fitted* APD=347ms claim from the slide narrative. The optimizer's tier-2 fit substantially shortens APD; raw MHAS13 isn't representative of the validated model.
Pipeline implications:
1. Figure-generation skills must be model-aware: there's a real gap between what's in `cardiac_sim/ionic/{mhas13,...}` (raw) vs what the lab actually claims as "validated MHAS13" (post-fit).
2. The optimizer's outputs should be **persisted as artifacts** — not re-derivable on demand. We currently have no saved tier-2 fitted theta on disk; the run script `run_mhas13.py` produces console-only output. This breaks reproducibility and forces re-fitting whenever a downstream slide wants to show "validated" results.
3. Plotting from raw-baseline models is faster but tells the wrong story; plotting from fitted models requires resurrecting parameters from logs. Pipeline needs a `figures/registry.yaml` or similar mapping `figure_id → {script, parameters_source, last_run, last_pass}`.
4. The PHAS13 1D shape bug should be fixed upstream rather than worked around — but that's outside this pipeline's scope. Logged as an aside.

**Friction observation 13 — user-asked questions surface real bugs in the workflow** (2026-04-29). The user asked "did the fit tamper with the spontaneous-beating nature of MHAS13?" — a question I had not thought to ask. The fitted theta reduced `g_K1` by 70%, which would normally re-introduce automaticity, but post-hoc check showed the model remained quiescent (V_max = -83.7 mV in 10s free-run with no stimulus) due to incidental compensation by other parameters (+143% kNaCa, +66% g_Kr, etc.).

The deeper finding: **the optimizer's fit objective and constraint set do not include a quiescence check.** `fit_cell` scores models on paced-AP biomarkers only (APD, dV/dt, V_peak, V_rest range) and tests them under stimulation. A quiescent matured-cell model could silently lose its maturation property in any future fit if random seed / targets change.

Pipeline implications:
1. **For matured-cell model fits, an explicit quiescence constraint is mandatory** — `V_max < -40 mV` under T_no-stim ≥ 10s. This belongs in `Optimizer/V1/tuner/cell_fitter.py` constraint logic, not as a post-hoc check in the monthly-report pipeline.
2. **Slide content should note the post-hoc verification** — turns a methodological gap into a strength (we noticed, we checked). The grey takeaway on D1 should mention this. D2 should flag the open issue. Future Outlook should include the constraint addition.
3. **The pipeline's responsibility extends to interpretive questions, not just data extraction** — a pure scout/consolidate/assemble pipeline would have produced a slide claiming "MHAS13 validates as quiescent matured model" without checking that the *fitted* version was quiescent. Need a "what did we assume that's actually contingent?" prompt step for matured-cell or otherwise-constrained models.

Saved post-hoc check at `MonthlyReport/April/check_fitted_quiescence.py` for reproducibility.

**Friction observation 14 — animation toolchain assumptions** (2026-04-29). User requested an animated quiescence comparison. First pass used `matplotlib.animation.PillowWriter` to write GIF directly — produced a working 4.3 MB GIF but the user pushed back: "I would instead render the video and then turn it into a GIF." Workflow change to **MP4-first → GIF conversion** (FFMpegWriter for libx264 MP4, then ffmpeg subprocess with palettegen+paletteuse for high-quality GIF). System didn't have ffmpeg installed; resolved via `pip install imageio-ffmpeg` (29 MB wheel containing a bundled binary, pointed `matplotlib.rcParams['animation.ffmpeg_path']` at it). Resulting MP4 = 1.8 MB / GIF = 2.6 MB, both higher quality than the direct-PillowWriter GIF.

Pipeline implications:
1. **Animations are a slide deliverable, but PPTX embedding prefers MP4 over GIF** (smaller, smoother, native player). PDF readers see the still cover frame — must label "(video)" on the slide explicitly per Zimmerman spec. Pipeline should default to producing both and let the assemble step pick.
2. **ffmpeg is a hard dependency** for the video path. Pipeline setup script should `pip install imageio-ffmpeg` (no system-package needed, env-isolated).
3. **The palette-method conversion (`palettegen` + `paletteuse=dither=bayer`) gives much higher-quality GIFs** than PillowWriter's direct render at similar size. Worth standardizing in the pipeline's animation utility.

**Friction observation 15 — "use monodomain" for visual demos** (2026-04-29). User asked for the F2 animation to show "a 2D cardiac chip using monodomain" — i.e. a 3x3 grid of cells, not a single cell. Implementation note: with **identical cells and uniform parameters**, the monodomain PDE reduces per-node to the single-cell ODE — every well evolves identically (no spatial gradients → no coupling currents). The single-cell free-run trace IS the monodomain solution on a uniform 3x3 patch. Tiled the existing single-cell trace into 9 wells and called it monodomain (caption explains the equivalence).

Pipeline implication: **the assemble step should know which figures need real tissue physics vs which can use the uniform-equivalence shortcut**. Spontaneous firing / quiescence on a uniform substrate = uniform per-cell visualization is honest. Boundary speedup or wave propagation requires real PDE solve. Pipeline figure-registry should tag each animation as `[uniform | gradient | propagation]` to choose the right backend.

**Friction observation 16 — coupled monodomain regen cost vs visual payoff** (2026-04-29). User followed up with a sharper request: real coupled monodomain on the 3x3 chip with stochastic ICs, V trace of center well only, center well highlighted in red. Implemented via `tuner.batch_step` with M=9 cells + `I_stim = -D∇²V` injection (5-point Laplacian, Neumann BC). PHAS13 wells diverge briefly from perturbed ICs, then synchronize within ~5 ms via diffusion (D=0.001 cm²/ms, dx=0.025 cm — fast coupling on this length scale). MHAS13 chip stays dark with V_max = -72.6 mV (well below the -40 mV AP threshold despite IC noise). Runtime: 132 s per chip × 2 = ~4.5 min total CPU.

Pipeline implications:
1. **Real-tissue regen is a multi-minute cost** — pipeline must cache results, not regenerate on every slide tweak. Save `figures/*_traces.npz` alongside MP4/GIF; downstream rendering reads the .npz, not re-runs the sim.
2. **For visualizing tissue-level coupling, dx and D matter for the visual story**. Realistic dx=0.025 + D=0.001 means coupling syncs cells on millisecond timescale, so chip beats look ~uniform anyway. To make coupling effects *visible*, need either bigger dx (weaker coupling), larger grid (longer L for waves), or an asymmetric perturbation (point stim, parameter heterogeneity). For the current "spontaneous synchronization" story, realistic params work — caption explains the synchronization happens too fast to see.
3. **Center-well highlighting is a useful visual idiom** — single red ring on the central cell + plot only that cell's V in the trace strip. Pipeline should support this as a standard "tissue-with-probe" animation template.

**Friction observation 17 — explaining the optimizer to a reader** (2026-04-29). User asked for visual to explain the tuning algorithm; I built `make_d2_f1_schematic.py` (matplotlib boxes-and-arrows), then walked through GP + qLogNEHVI math, including a clarifying detour ("why GP instead of Bayesian optimization?" — they're complementary, not alternatives). The math walkthrough was useful for slide narration but didn't go into the deck.

Pipeline implication: **bundle each algorithm-explainer figure with a "narration" markdown sidecar** containing the math + plain-language version. The slide bullets are extracted from the narration; the speaker notes can pull the math directly. Avoids re-deriving the same explanation each month.

**Friction observation 18 — user's actual slide layout choices** (2026-04-29). User shared the live deck (`Chang Progress Report - April 2026.pptx` on Drive). Choices that diverged from my proposed deck_outline:

1. **Per-slide template**: 5-zone layout — `PROBLEM / CONCEPT / ACHIEVEMENTS / IMPACT / OUTCOME` (likely from Zimmerman's `ZimmermanLab_DefaultSlides.pptx`, not my custom outline). This template forces a specific narrative structure: setup → idea → what we did → why it matters → next step. Different from my proposed (objective + figure caption + bullets + grey takeaway), but compatible with Zimmerman spec.
2. **Slide consolidation**: my D1+D2 (2 slides per ionic-engine arc) became user's slide 3 (BO algorithm + tuned-MHAS13 example combined) and slide 4 (Non-Spontaneous hiPSC Engine = quiescence). Combined per-arc footprint: ~2 slides, not 4. Trade-off: each slide is denser but the arc reads more cohesively.
3. **Self-built schematic**: user drew their own simplified BO flow rather than embedding `D2_F1_optimizer_schematic.png`. Cleaner for the slide template but introduced naming collisions ("Surrogate" → ambiguous w/ Surrogate Pipeline; "Output Parameters / Gating Parameters" → ambiguous w/ ionic gates) and content gaps (no fitted-vs-baseline numbers). Worth catching with a "domain-vocabulary lint" step.
4. **Summary-vs-body ordering inconsistency**: summary slide listed *Immature → BO → Quiescence*; research slides went *BO → Quiescence*. I flagged it during review.
5. **Forward-looking pointer**: "Consider using a MLDO Approach for V2" — pending: I don't know the acronym; user will clarify or it stays as-is for the report.

Pipeline implications:
1. **The 5-zone template (`PROBLEM/CONCEPT/ACHIEVEMENTS/IMPACT/OUTCOME`) is the slide-content schema** — pipeline output should map directly to those five fields, not free-form prose. Each project entry should produce 5 short paragraphs, one per zone.
2. **Schematic regeneration**: my matplotlib schematic was useful for *the conversation* but the user redrew theirs natively in PowerPoint for stylistic fit. Pipeline should treat my generated figures as **drafts / source-of-truth references**, not final embeddable assets — the user (or future skill) will reformat them in PPT/Keynote/Illustrator. Save SVG alongside PNG for editability.
3. **Domain-vocabulary lint step**: cross-check slide text against project glossary (Surrogate, Gating Parameters, etc.) for collisions or imprecise terms. Cheap LLM pass.
4. **TOC ↔ body consistency check**: pipeline should auto-verify summary slide ordering matches research slide order. Trivial check, easy to miss manually.

The April deck represents the **first manual-with-pipeline-help iteration** — accept that this output is not optimal, but it surfaces concrete pipeline requirements (template, lint, ordering, asset format) that I'd have missed designing in the abstract. Specifically logged here as the calibration anchor for what /monthly should produce: the pipeline's output should not be more elaborate than this; it should be cleaner, faster, and have the lint+consistency checks built in.

### 2026-04-30 Session — Submission and PI feedback round-trip

**Deck submitted** (2026-05-01 01:13 EDT) and **John replied** (2026-05-02 15:40 EDT, dated in the future relative to 2026-04-30 work but already received in the inbox by this session). Net: **"Good job overall, John."** — positive review. No demands for re-work, no scoping pushback. Comments are all clarity / pedagogical / literature-pointer.

**Final deck shape (11 slides, materially different from my proposed outline)**:
- Slide 1 — Title
- Slide 2 — Summary (TOC) listing: Wavefront Curvature Analysis · Monodomain/LBM Modification · Inverse Crescent Formation · BayesOpt V1 · Quiescence Tuning · Ionic Surrogate Modeling · Reporting Period in Summary
- Slide 3 — **Wavefront Curvature Analysis** (Cardinal vs Moore connectivity, diagonal-weight effect)
- Slide 4 — **Monodomain/LBM Simulation Modification** (mirror padding + 9-pt stencils to recreate "crescent shaped artifact")
- Slide 5 — **Inverse Crescent Formation** (drain-less vs receive-less pumping; transition crescent → camel toe → inverse crescent)
- Slide 6 — **Inverse Crescent Formation Supplemental** (original-pump vs Fickian-modified comparison)
- Slide 7 — **Bayesian Optimizer for Ionic Engine** (BO schematic + MHAS13 example: baseline 541 ms / 132 V/s → fitted 334 ms / 101 V/s, V_rest = −87 mV)
- Slide 8 — **Non-Spontaneous hiPSC Ionic Engine** (quiescence tuning narrative)
- Slide 9 — **Ionic Surrogate Modeling** (sample data + simplified v3 schematic: Dense GeLU 17→16, 16→16, 16→16, Linear 16→14 + Forward Euler; 600 GB data)
- Slide 10 — **Ionic Surrogate Modeling Cont.** (training curve, classical-TTP06 benchmark, "consider hybrid approach" pivot)
- Slide 11 — **Future Goals** (Boundary Speedup proofs · Surrogate FNO literature · LBM hyperbolic derivation)

Notable **deviations from my proposed outline**:
- **Bidomain Parabolic-Parabolic (slide F)** dropped entirely — folded into Slide 11's "LBM hyperbolic derivation" bullet.
- **cardiac_ml harness (slide B)** dropped entirely — not surfaced to PI.
- **Storage tank work expanded** to 4 slides (3, 4, 5, 6) using new internal vocabulary ("crescent / camel toe / inverse crescent" — these come from John's storage-tank visualizations, the user adopted his framing).
- **D1+D2 collapsed** to slides 7+8 with the Optimizer + MHAS13 + quiescence story split across them; matches earlier observation 18.
- **Surrogate** rebalanced from 1 slide to **2 slides** (9+10) — given the schematic + training-curve + benchmark each need space, the user found 1 slide too tight.

**John's 11 numbered comments (verbatim excerpts + my synthesis)**:

| # | Slide | Comment | Synthesis |
|---|------|---------|-----------|
| 1 | global | "I cannot read the labels on your graphs. Makes them difficult to interpret." | Hard fail on figure-label readability. Across multiple slides. |
| 2 | global | "Would be good to have a schematic setting up the discussion/simulation." | Each result needs an upstream setup figure. |
| 3 | 4 | "'crescent shaped artifact' is a bit aggressive language. I think it's likely that cells handle themselves in this way, which would actually make the smooth diffusion an 'artifact'. ('John Artifact')." | **Reframe**: the curved wavefront may be physiology, not numerics. Ground-truth flip. |
| 4 | 4 | "Without labels it's hard to understand what the conditions you are testing." | Test-condition labels missing. |
| 5 | 5 | "The physiology handbook that Andre wrote suggests that some of these charge effects could occur due to membrane charge / membrane surface area." | Lit pointer: Andre's physiology handbook for charge / SA mechanism. |
| 6 | 5 | (Long comment): in LBM, fluid velocity must be ≤ dx of one unit per step; faster fluid would pump past the square but discrete streaming masks that. Same applies to diffusion in discrete systems — diffusion is a less-sensitive knob in our simulations than it is in the universe (which doesn't apply physics in discrete steps). | **Substantive scientific point**: discretization-imposed insensitivity to diffusion. Worth a follow-up investigation; affects how we interpret negative diffusion-knob results in slide 5/6. |
| 7 | 7 | "Is this to figure out the correct metrics for hiPSCs? We can probably look up many of these values if we hunt in the literature." | Lit pointer: hiPSC AP-target values exist in literature; don't infer-from-scratch. |
| 8 | 8 | "You need to define your terms (e.g. high I_f + low I_K1) — what do you mean in this context? In general acronyms should be defined the first time they are used (like you did for cardinal vs moore)." | **Acronym/term lint** is mandatory. User did it for cardinal/Moore but not I_f / I_K1. |
| 9 | 8 | "In general, stem cells and neonatal rat ventricular myocytes (NRVMs) will spontaneously beat, especially early on. If they become very mature this can become limited." | Domain context: NRVM as a baseline reference for spontaneous beating. |
| 10 | 9 | "What's your take away here?" | **Missing takeaway sentence on slide 9.** Multiple slides may have this issue — the grey takeaway box that Zimmerman's spec mandates was inconsistently applied. |
| 11 | 10 | "You are using a basic model to train your surrogate model here, but what happens if you train on a more advanced model? Can you learn things in your simple surrogate that the more advanced model contains, but then also save time in comparison?" | **Strategic suggestion**: train v3/v4 surrogate on ORd or richer model, see if simple-surrogate captures rich-model features at lower cost. Concrete future-work prompt. |

**Friction observation 19 — figure-label readability is the #1 PI-visible failure mode** (2026-04-30). Comments #1 and #4 both flag unreadable graph labels — the most common complaint in the entire reply. The pipeline must enforce a **minimum label font size relative to slide dimensions** (e.g., axis labels ≥ 14pt at 10×7.5in slide scale, ≥ 18pt for title text on each panel). Specifically: my matplotlib defaults (`fontsize=8`-ish for tick labels) render unreadable when figures are scaled down to fit a 5×3in PPT placeholder. Pipeline should either (a) generate figures at PPT-target dimensions with PPT-target fonts, not "research-paper" defaults, or (b) include a post-hoc lint that flags any figure where label text would be < 10pt at final embed size.

**Friction observation 20 — the "John Artifact" reframing** (2026-04-30). John's comment #3 inverts the framing: what we've been calling the "crescent-shaped artifact" may actually be physiological reality, and *smooth diffusion* may be the simulation artifact. This is a **scientific reframing**, not just a wording change — it affects how we prosecute the boundary-speedup question in May going forward. Update the boundary_conduction_speedup KNOWLEDGE.md with this PI-supplied alternative hypothesis. Pipeline implication: future drafts should default to neutral framing for novel observations ("emergent feature X" or "phenomenon Y"), not "artifact," and flag any "artifact"-style language for PI review before submission.

**Friction observation 21 — discrete-step diffusion insensitivity** (2026-04-30). John's comment #6 is a **substantive physical insight**: in any discrete simulation (LBM streaming OR finite-difference diffusion), there's an implicit cap on how strongly the discrete operator can respond — a fluid velocity > 1 cell/step gets aliased as 1 cell/step, and diffusion has the analogous issue. This means the negative result on slide 5/6 ("modulating diffusion rate does not cause meaningful push toward inverse crescent") may be an **artifact of the numerical scheme**, not a true negative. Action: investigate whether continuum-limit diffusion (sub-grid resolution) recovers the missing knob. Pipeline implication: when reporting "knob X has no effect," include a discretization-sensitivity check before stating the negative result.

**Friction observation 22 — every slide needs an explicit takeaway sentence** (2026-04-30). John's comment #10 ("What's your takeaway here?") on slide 9 indicates the grey takeaway box — which Zimmerman's V1 spec mandates — was inconsistently applied across the deck. Pipeline must enforce this as a hard rule: every research slide produces exactly one takeaway sentence. The /monthly auditor should reject any draft where a research slide is missing the takeaway.

**Friction observation 23 — acronym lint must be enforced first-use, not just in glossary** (2026-04-30). John's comment #8 is concrete: the user defined "cardinal vs Moore" inline (good) but used "I_f", "I_K1" without definition (bad). The pipeline lint isn't just "are all acronyms in a glossary somewhere?" — it's **"is each acronym defined at first use within slide body text?"** This is harder than a static glossary lint and probably needs a per-slide LLM check.

**Friction observation 24 — surrogate-on-richer-model is a free-from-PI research direction** (2026-04-30). John's comment #11 supplies a specific V2 direction: train v3/v4 surrogate on ORd or another richer model, evaluate whether the simple surrogate retains rich-model features at lower cost. This is exactly the "honest negative result → redirect" loop done well — PI engaged with the pivot and offered next steps. Pipeline implication: the Future Outlook slide should prompt the PI for direction by listing 2-3 candidate continuations per project, not a single locked-in next-step. PI-supplied steering becomes free input for the next month's plan.

**Pipeline scoring recap** (April test run, post-feedback):
- Net PI sentiment: **positive** ("Good job overall")
- Hard quality issues: 1 systemic (label readability), 1 missing-takeaway (slide 9 only), 1 acronym lint (slide 8 only)
- Reframings flagged: 1 ("artifact" vocabulary)
- Scientific points raised: 2 (discretization insensitivity, membrane-charge mechanism)
- Free literature pointers: 2 (Andre's physiology handbook, hiPSC AP-target values)
- Free V2 direction supplied: 1 (surrogate on richer model)

**Updated Next Step**: April test run is **closed**. May report (due 2026-05-28, last Thursday of May) is the next pipeline test. Before May, design `/monthly` pipeline architecture from the now-24 friction observations, then `/blueprint` to PLAN.md. Open V2 direction (John's comment #11) belongs in the Surrogate Pipeline IDEALOG, not here.
