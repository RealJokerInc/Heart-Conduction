# PLAN: Textbook Website Refresh — Interactive Figures + Themeable Redesign

Created: 2026-07-03
Engine(s): None (website/ front-end + build tooling)
Research question: [textbook](README.md)
Source: [IDEALOG.md](IDEALOG.md) — 2026-07-03 "Website UI audit + interactive-refresh direction"; design brief in [REFRESH_PLAN.md](REFRESH_PLAN.md); audit in [audits/UI_AUDIT_2026-07-03.md](audits/UI_AUDIT_2026-07-03.md); proof-of-direction artifact `a514e90d`.
(Prior completed plan archived → [plans/2026-07-02_audit-remediation.md](plans/2026-07-02_audit-remediation.md).)
**Durable prototype (widget source):** [plans/2026-07-03_refresh_prototype.html](plans/2026-07-03_refresh_prototype.html) (in-repo copy of the session's `refresh_pitch.html`; a cold-start agent ports `rk4`/`fit`/`cvar`/`simulateMS`/`apd90` + the FHN/AP widgets from here — do NOT rely on the ephemeral scratchpad path). Also durable: artifact `a514e90d` (WebFetch to view).
**Revised 2026-07-03 after adversarial /audit** (2 HIGH + 4 MED + 8 LOW) — see Mutation Log.

## Objective
Refresh the textbook website (`Research/Active/textbook/website/`) from an anonymous static docs template into a themeable "explorable explanation": fix the dark-mode figure bug at its root, give figures a theme-token color system, wire up the orphaned cover/part pages, and add ~6–8 flagship interactive figure widgets — all as progressive enhancement over static SVG so the Playwright PDF pipeline keeps producing the 195-pp PDF unchanged.

## Success Criteria
- [ ] Dark mode: every figure keeps legible, semantic color (no grey-collapse); confirmed by screenshots of ch2/ch5/ch10 in dark.
- [ ] Every figure-color hex in `chapters/*.html` SVGs resolves to a `var(--fig-*)` token OR is a documented per-figure exception in the census map (target: 0 un-migrated hues; greys collapse to the ink/grid scale, which is NOT a semantic change); blanket `svg path/text` override (style.css:810-812) deleted and replaced by a scoped themeable default.
- [ ] Cover (`title.html`) + 5 part dividers reachable and in reading-order prev/next flow in the SPA.
- [ ] ≥6 interactive figure widgets live in the SPA, each keyboard-operable + `prefers-reduced-motion`-aware, each with a static SVG print fallback.
- [ ] `python website/build/html_to_pdf.py` still builds a ~195-pp PDF; no widget/canvas leaks into print; no raw-LaTeX leak.
- [ ] Light + dark both render clean with zero console errors on every chapter.
- [ ] All existing behavior preserved (search, scroll-spy, font/theme toggles, keyboard nav, MathJax-SVG).

## Architecture Changes
- MOD: `website/style.css` — add an expanded `--fig-*` figure-token block to the site's **TWO** theme scopes only (`:root` = light defaults, line 6; `[data-theme="dark"]` = dark overrides, line 62). **The site has NO `@media (prefers-color-scheme)` block and no `[data-theme="light"]` token scope** — dark mode is driven entirely by the JS-set `data-theme` attribute (`index.html` defaults `data-theme="light"`; `app.js:initTheme/toggleTheme`). Do NOT add an `@media` token block (it would make a dark-OS user who toggles to light get dark figure colors). Also: add a scoped themeable text default (`.figure text, figure.fig text { fill: var(--fig-axis) }`) to REPLACE deleted line 810; `@media print` widget-hide belt; **delete** the blanket override `style.css:810-812`; rebuild type scale (serif×mono), deepen crimson accent, drop `justify`, fix equation-number placement, add cover/part-opener styles.
- MOD: `website/app.js` — include `title.html` + `part-*.html` in the flat reading order (`buildChapterList`), render clickable part rows + a Cover entry (`renderSidebar`), let `navigateTo`/bottom-nav flow through them; after chapter load, invoke the figure loader.
- NEW: `website/figures.js` — SPA-only figure-widget loader: scans `[data-widget]`, mounts the matching widget over the static SVG fallback, re-mounts on theme change; **not** referenced by `html_to_pdf.py`.
- NEW: `website/figures/*.js` — one module per widget (`fhn.js`, `ap.js`, `rk.js`, `cfl.js`, `nernst.js`, `wave.js`) + shared `_canvas.js`; RK4/`fit`/`cvar`/`simulateMS`/`apd90` ported from the durable prototype `plans/2026-07-03_refresh_prototype.html`.
- NEW: `website/build/migrate_figure_colors.py` — one-shot sweep: hardcoded hex → `style="…:var(--fig-*)"` across all chapter SVGs, with an unmapped-hex report.
- NEW: `website/build/verify_site.py` — reusable Playwright harness: screenshots key chapters in light+dark, asserts zero console errors, asserts the PDF-assembly HTML contains no mounted widget canvas.
- MOD: `website/chapters/*.html` — figure SVGs re-colored to tokens (Phase 1); ~6 figures additionally wrapped in the `<figure data-widget>` convention (Phases 2–3).

## Known anti-patterns (do NOT reintroduce — from IDEALOG/audit)
- **Blanket `[data-theme="dark"] svg line, svg path { stroke: … }`** — the root cause: it repaints every figure curve one grey and adds a stray outline to MathJax glyphs. Never reintroduce a global SVG stroke/fill override; color figures via tokens.
- **Hardcoded hex inside SVG** (`fill="#e94560"`, `fill="#16213e"`, …) — the reason the blanket override existed. All figure color must resolve from `--fig-*`.
- **`fill="var(--x)"` as a presentation attribute** — NOT reliably supported. `var()` only resolves inside a `style="…"` attribute or a stylesheet. Migration must emit `style="fill:var(--fig-crimson)"`, never `fill="var(--fig-crimson)"`.
- **Canvas-only figures** — a figure that exists only as a widget prints blank. Every `[data-widget]` must retain a static SVG fallback.
- **Injecting the widget loader into the PDF build** — `html_to_pdf.py` must keep injecting only MathJax + PRINT_CSS; never add `figures.js`, or widgets would mount in print.
- **Editing content while re-coloring** — no prose/equation/caption changes in the color sweep; monolithic-solver framing and appendix-recombination remain forbidden (content anti-patterns).

---

## Phase 1: Fix & Re-shell (CSS + nav; no widgets, no content edits)

**Goal**: Kill the dark-mode figure bug at the root, make every existing static figure themeable, refresh the visual shell/typography, and wire the orphaned cover/part pages into the SPA. Fully deliverable on its own — the site is already much better after this phase, with zero interactivity added yet.
**Tier**: large
**Estimated scope**: 1 new build script + 1 verify harness; heavy edits to `style.css` and `app.js`; scripted edits across ~24 chapter files (color only, never prose).

### Phase Context
- Canonical source: `website/chapters/*.html` (self-contained fragments) + `website/toc.json` + `website/index.html` + `website/app.js` + `website/style.css`. These fragments ALSO build the PDF via `website/build/html_to_pdf.py` (assembles in toc order, wraps with `style.css` + PRINT_CSS + MathJax, `data-theme="light"`, Playwright → A4 PDF).
- **Never edit prose/equations/captions in this phase** — color/markup only. A content diff should show only attribute/style changes inside `<svg>` and structural nav wiring.
- **The figures use ~61 distinct hexes** (census: `grep -rhoiE '(fill|stroke)="#[0-9a-f]{3,6}"' website/chapters | …`). An 8-token set cannot express them, so the token system is a **categorical + neutral + tint** palette. Collapsing multiple *greys* into the ink/grid scale is NOT a semantic change; collapsing distinct *hues* that encode different things within one figure IS — the census step (1.2) flags those for a per-figure decision. Token families (exact light/dark values finalized in Step 1.1 against the census + the `dataviz` skill's colorblind/contrast validator):
  | family | tokens | absorbs (examples from census) |
  |---|---|---|
  | categorical hues | `--fig-crimson --fig-blue --fig-teal --fig-green --fig-amber --fig-orange --fig-purple` | crimson: `#e94560 #d32f2f #c62828 #c0392b`; blue: `#0f3460 #42a5f5 #1565c0 #2980b9 #0072b2`; green: `#2d6a4f #2e7d32 #1b5e20`; amber: `#8a6d00 #b36b00 #e69f00`; orange: `#d55e00`; purple: `#6a1b9a #4a3d7a` |
  | ink / neutral scale | `--fig-ink --fig-muted --fig-faint --fig-grid` | ink: `#16213e #333`; muted: `#5a6175 #666`; faint: `#888 #999 #b0b8d4`; grid/hairline: `#d0d5e0 #e2e5ed #e0e0e0 #ccc #e8e8e8` |
  | surfaces | `--fig-stage` (fig bg) + panel tints `--fig-tint-{blue,teal,red,amber,green,purple}` | stage: `#fff #fafafa #f8f8f8`; tints: `#f0f4ff #e8f4f8 #fce4ec #e3f2fd #fff3e0 #f3e5f5 #e8f5e9` |
  Note: `--fig-axis` is an alias of `--fig-ink` for stroke use (axes). The authoritative hex→token map is produced by the census in Step 1.2, not hand-guessed here.
- Verification is visual, not pytest. Use `conda run -n heart-conduction python website/build/verify_site.py` (built in Step 1.0) and the `--html-only` PDF assembly.
- Env: `conda run -n heart-conduction` (has playwright + chromium).

### Step 1.0: Verification harness
**Model**: opus
#### Read First
- This session's screenshot pattern (scratchpad `shoot.py`): serve `website/` on a port, screenshot `index.html#<chapter>` in light/dark, capture console errors.
- `website/app.js:43-82` — how the SPA boots + `#hash` navigation, so screenshots land on the right chapter.
#### Why
Every later step's "Verify" depends on a repeatable way to see the site in both themes and to prove no console errors / no widget leak into print. Build it first so it exists for 1.1+.
#### Implementation Spec
**Files to create:** `website/build/verify_site.py` — CLI: `--chapters ch2,ch5,ch10 --themes light,dark --out <dir>`; starts `http.server` on `website/`, screenshots each chapter×theme (full-page), records `console`/`pageerror` events, writes a JSON summary `{chapter, theme, errors[]}`. Exit non-zero if any console error.
**Print-safety check (meaningful version — the trivial `'fig-widget' in assemble_html()` string check is USELESS because `canvas.fig-widget` is created at runtime by `figures.js`, which the assembled source never contains):** the real guarantee is that the PDF path never loads the widget loader. Assert BOTH: (a) `assemble_html()` (imported from `html_to_pdf.py`) contains no reference to `figures.js`/`mountFigures`; (b) render the assembled print HTML headless (as the PDF build does) and assert `document.querySelectorAll('canvas.fig-widget').length === 0` in the loaded DOM. (a) is vacuously green until Phase 2 but guards against a future regression that wires the loader into the build; (b) is the true end-state test.
#### Pseudocode
```
serve website/ :PORT
for ch in chapters, th in themes: goto index.html#ch; set data-theme=th; wait; screenshot; collect errors
from html_to_pdf import assemble_html; src = assemble_html()
assert 'figures.js' not in src and 'mountFigures' not in src        # loader not injected into print
render src headless (file://) → assert page.eval('document.querySelectorAll("canvas.fig-widget").length')==0
write summary.json; nonzero exit on any console error or either print-safety failure
```
#### Test Spec
- Manual: run against current site → baseline screenshots + a known-empty error list (current site has 0 console errors per this session's screenshots).
#### Checklist
- [ ] Screenshots land on the correct chapter (hash nav works headless)
- [ ] Console + pageerror captured per page
- [ ] Print-assembly widget-leak assertion wired (import `assemble_html`)
- [ ] JSON summary + non-zero exit on failure
#### Verify
```
conda run -n heart-conduction python website/build/verify_site.py --chapters ch1,ch2,ch5,ch10 --themes light,dark --out /tmp/verify_baseline
```
#### Exit Criteria
- [x] Baseline run: 0 console errors, screenshots produced for all 8 chapter×theme combos. **DONE 2026-07-06** — `verify_site.py` built; baseline green (`/tmp/verify_baseline`).
#### Risk
Headless hash-nav races the async chapter fetch — mitigation: `wait_for_selector('.chapter-content .chapter, .chapter-title')` + a fixed settle delay before screenshot.

### Step 1.1: Figure-token system + themeable defaults (CSS only; keep blanket override for now)
**Model**: opus
#### Read First
- `website/style.css:6` (`:root` = light tokens) and `website/style.css:62` (`[data-theme="dark"]` = dark tokens). **These are the ONLY two theme scopes.** Verify: `grep -n 'prefers-color-scheme' website/style.css` → NONE; `grep -n '\[data-theme="light"\]' website/style.css` → only line 202 (icon display, not tokens). Dark mode is JS-toggled (`app.js:372-386` initTheme/toggleTheme; `index.html` root defaults `data-theme="light"`).
- `website/style.css:716-723` — `.figure` / `.figure-caption` rules.
- `website/style.css:789-812` — inline-bg dark overrides + the blanket SVG override (**do not delete yet**).
- `dataviz` skill `references/palette.md` — to finalize colorblind-safe / AA-contrast values for the 7 categorical hues on both grounds.
#### Why
Introduce the (expanded) color system the migration (1.2) targets, without breaking the current hardcoded figures — the site must stay working between 1.1 and 1.2. Because migrated figures use inline `style="…:var(--fig-*)"` (specificity 1,0,0,0), they already win over the stylesheet blanket override even before it's deleted → no broken intermediate state. The blanket override is deleted in 1.2 once migration is complete.
#### Implementation Spec
**Files to modify:** `website/style.css`
- Add the full `--fig-*` block (families from Phase Context: 7 categorical hues + `--fig-ink/--fig-axis(=ink alias)/--fig-muted/--fig-faint/--fig-grid` + `--fig-stage` + 6 `--fig-tint-*`) to **BOTH** scopes only: light values under `:root`, dark values under `[data-theme="dark"]`. **Do NOT add an `@media (prefers-color-scheme)` block** — it has no matching `[data-theme="light"]` reset here, so it would leak dark figure colors into a dark-OS user's light-toggled view. Finalize hue values via the `dataviz` palette (e.g. light crimson `#C31D38`/blue `#1F6FB2`/teal `#0C7480`/green `#2E7D57`/amber `#B7791F`/orange `#C2410C`/purple `#6D4AA8`; dark lifts `#FF5468/#5AA9E6/#34C6D2/#57C48A/#E7A23C/#F97316/#B69BE0`; tints = pale fills light, ~12%-alpha-over-stage dark).
- Add a scoped themeable **default** (load-bearing — replaces the about-to-be-deleted line 810 and covers `<text>` with no own fill, incl. group-inherited): `.figure text, figure.fig text { fill: var(--fig-axis); }` and `.figure, figure.fig { color: var(--fig-axis); }` (so any `currentColor`/unstyled stroke themes too). Scope to figures so MathJax `mjx-container` SVGs are untouched.
- (Optional, for hand-authored SVGs only) a minimal `.figure .s-cr{stroke:var(--fig-crimson)}`-style utility set — NOTE the 1.2 migration uses inline `style`, not these classes, so keep the set small; the load-bearing piece is the themeable default above.
- Do NOT yet remove `style.css:810-812`.
#### Pseudocode
```
:root { --fig-crimson:#C31D38; --fig-blue:#1F6FB2; … --fig-ink:#1E2438; --fig-axis:var(--fig-ink); --fig-grid:#E7E9F2; --fig-stage:#FBFBFE; --fig-tint-blue:#F0F4FF; … }
[data-theme="dark"] { --fig-crimson:#FF5468; --fig-blue:#5AA9E6; … --fig-ink:#D4D8E8; --fig-grid:#232838; --fig-stage:#10131F; --fig-tint-blue:#16203A; … }
/* --fig-axis aliases --fig-ink → define once in :root; it re-resolves per theme via --fig-ink */
.figure text, figure.fig text { fill: var(--fig-axis); }   /* themeable default; replaces line 810 */
.figure, figure.fig { color: var(--fig-axis); }
```
#### Test Spec
- `verify_site.py` light+dark on ch5 → figures still render (tokens added, override still present); 0 console errors.
#### Checklist
- [ ] Confirmed only 2 scopes (`:root`, `[data-theme="dark"]`); no `@media` token block added
- [ ] All token families present in BOTH scopes
- [ ] Hue values validated colorblind-safe + AA via `dataviz` palette
- [ ] `.figure text`/`figure.fig text` themeable default added
- [ ] Blanket override still present (removed in 1.2)
#### Verify
```
for t in fig-crimson fig-blue fig-teal fig-green fig-amber fig-orange fig-purple fig-ink fig-muted fig-faint fig-grid fig-stage; do echo -n "$t="; grep -c "$t" website/style.css; done   # each hue/neutral token expect 2 (one per scope; --fig-axis alias may be 1)
grep -c 'prefers-color-scheme' website/style.css   # expect 0
conda run -n heart-conduction python website/build/verify_site.py --chapters ch5 --themes light,dark --out /tmp/verify_1_1
```
#### Exit Criteria
- [ ] Every token defined in both scopes (each hue token count == 2); no `@media` block; site unchanged visually; 0 console errors.
#### Risk
Defining a token in only one scope → undefined in the other mode. Mitigation: `grep -c` each token == 2. Adding an `@media` block by habit → light-mode leak; the checklist forbids it explicitly.

### Step 1.2: Migrate figure colors → tokens, then delete the blanket override
**Model**: opus
#### Read First
- `website/chapters/ch5.html` (the phase×current grid — densest hardcoded-hex figure) and `website/chapters/ch2.html` (Fig 2.1) — representative markup (`fill="#hex"`, `stroke="#hex"`, existing `style="…"`, `font-family="EB Garamond,serif"`).
- `website/style.css:810-812` — the blanket override to delete at the end.
#### Why
This is the actual fix for the HIGH bug. Converting SVG color to tokens lets a single theme swap recolor every figure correctly, and removes the need for the destructive blanket override.
#### Implementation Spec
**Files to create:** `website/build/migrate_figure_colors.py` — TWO modes:
- `--census` (run FIRST): scan all `chapters/*.html` + `appendix-*.html`, list every distinct color VALUE — `(fill|stroke)="#hex"` AND **named colors** (`white` ×~42 e.g. ch18 `<g fill="white">` knockouts, `black`, any CSS color keyword) — incl. on `<g>` group elements, with counts and a PROPOSED token, write `website/build/figure_color_map.json` (the authoritative, human-reviewable map). Greys → ink/muted/faint/grid by luminance; hues → nearest categorical token; panel fills → `--fig-tint-*`; **`white` → `--fig-stage` (so a knockout/background follows the theme) but FLAG for review (a white *label on a colored fill* must instead map to `--fig-stage` only if that reads; some belong to `--fig-ink`)**; `black` → `--fig-ink`; unknowns → `null` (flag for manual decision — do NOT guess). `fill="none"`/`stroke="none"` are ignored (not colors).
- `--apply` (run after the map is reviewed): operate ONLY inside `<svg …>…</svg>` spans (skip any `mjx-container`; source chapters have none, but guard anyway). For each element (including `<g fill=…>`), move any `fill=`/`stroke=` whose value is a mapped hex OR mapped named color → `style` as `fill:var(--fig-TOKEN)` / `stroke:var(--fig-TOKEN)` per `figure_color_map.json` (case-insensitive; expand 3-digit hex; leave `none` alone). Merge into an existing `style="…"` (append `;`), else add `style`. A `null`-mapped value is left untouched and re-reported. Idempotent.
**Files to modify:** `website/style.css` — after `--apply` verified in dark, DELETE the blanket override `810-812` (BOTH the `svg text{fill:var(--text)}` line 810 AND the `svg line,svg path{stroke:…}` lines 811-812). Line 810's job (keeping figure `<text>` legible in dark) is taken over by the `.figure text{fill:var(--fig-axis)}` default added in Step 1.1. Keep the inline-bg overrides (790-808) but re-point to `--fig-stage` where relevant.
#### Pseudocode
```
census: for file, for each <svg>..</svg>: collect (fill|stroke)="#hex" (incl <g>); propose token by hue/luminance; write figure_color_map.json
# human/agent reviews figure_color_map.json; resolves any null (add token OR document per-figure exception)
apply: for file, for each <svg>..</svg> region, for each color attr with mapped token:
   remove attr; merge "fill|stroke:var(--fig-tok)" into element style=
write files; re-report any remaining null-mapped hex
# then: delete style.css 810-812; re-verify dark (defaults + tokens now carry all figure color)
```
#### Test Spec
- After `--apply`: `grep -RInE '(fill|stroke)="#' website/chapters website/chapters/appendix-*.html` → only `null`-mapped exceptions remain (documented); target 0.
- `verify_site.py` ch2,ch5,ch10,**ch18**,appendix-b in DARK → figures show distinct hues (crimson/blue/teal/green/amber/purple), and **ch18's group-inherited `<text>` labels are legible (not dark-on-dark)** — the specific MED risk. Manual screenshot review.
#### Checklist
- [ ] `--census` writes `figure_color_map.json`; all 61 hexes have a proposed token or explicit `null`
- [ ] Review the map; resolve every `null` (assign token or document a per-figure exception)
- [ ] `--apply`; grep confirms no un-migrated `(fill|stroke)="#"` in SVGs (besides documented exceptions)
- [ ] Delete blanket override `style.css:810-812`
- [ ] Dark screenshots: ch2 FHN nullclines, ch5 phase grid, **ch18 labels**, App-B bowls all colored + legible
- [ ] MathJax equations unaffected in dark (no stray glyph outline) — spot-check ch1/ch10
#### Verify
```
conda run -n heart-conduction python website/build/migrate_figure_colors.py --census   # writes figure_color_map.json
# → review figure_color_map.json, resolve nulls, then:
conda run -n heart-conduction python website/build/migrate_figure_colors.py --apply
grep -RInE '(stroke|fill)="#' website/chapters/*.html || echo "no un-migrated hex color attrs"
conda run -n heart-conduction python website/build/verify_site.py --chapters ch2,ch5,ch10,ch18,appendix-b --themes dark --out /tmp/verify_1_2
```
#### Exit Criteria
- [ ] Zero un-migrated color values inside chapter SVGs (besides documented `null`-mapped exceptions); blanket override gone; dark-mode figures legible + semantic; equations clean; 0 console errors.
#### Risk
(a) `var()` in a presentation attribute → won't resolve. Mitigation: script emits `style="…:var()"` only; grep to confirm no `="var(`. (b) Malformed `style` merge on elements with an existing `style`. Mitigation: idempotent re-run + screenshot review. (c) Over-broad regex touches text outside SVGs. Mitigation: operate only within matched `<svg>…</svg>` spans; content-diff review (prose untouched).

### Step 1.3: Shell + typography refresh
**Model**: opus
#### Read First
- `website/style.css:104-133` (base/body), `526-655` (title/part/chapter/equation), `633` (`p{justify}`), `650-655` (equation-label float).
- REFRESH_PLAN.md "Design system" (fonts, scale, crimson deepen).
#### Why
Turn the anonymous template into the chosen identity (serif×mono, deepened crimson, breathing room) without changing chapter markup/classes (so fragments + PDF are unaffected).
#### Implementation Spec
**Files to modify:** `website/style.css`
- Add `--font-serif`/`--font-sans`/`--font-mono` (system stacks from REFRESH_PLAN); apply serif to body/headings, mono to `.eyebrow`/`.chapter-number`/`.figure-label`/`.equation-label`/nav labels.
- Deepen `--highlight`/accent to arterial crimson (`#C31D38` light / `#FF5468` dark); keep the variable NAME `--highlight` (change values only to avoid churn); cool-bias `--accent`/`--accent2`.
- `p{ text-align:left }` (drop `justify`+hyphens or keep hyphens but left-align); tighten measure to ~64ch.
- Equation number: position within `.equation-block` without floating over tall displays (absolute top-right with reserved padding, or a caption row).
- Add cover/part-opener styles (`.cover`, `.part-opener`) — ready before 1.4 markup.
#### Pseudocode
```
:root { --font-serif:"Iowan Old Style",…; --font-sans:system-ui,…; --font-mono:ui-monospace,…; }
body,h1,h2,h3,h4 { font-family:var(--font-serif); }
.eyebrow,.chapter-number,.figure-label,.equation-label,.toc-part,.nav-direction { font-family:var(--font-mono); }
:root{ --highlight:#C31D38 } [data-theme="dark"]{ --highlight:#FF5468 }   /* values only; name kept */
p { text-align:left; } .chapter-content{ max-width:64ch; }
.equation-block{ position:relative; padding-right:3.5rem } .equation-label{ position:absolute; top:.5rem; right:.8rem; float:none }
.cover{…} .part-opener{…}
```
#### Test Spec
- `verify_site.py` sample (ch1, ch8, ch11, ch13, ch18, appendix-a) light+dark: legible, no overflow, equation numbers not overlapping equations.
#### Checklist
- [ ] Font tokens applied by role (serif body, mono labels)
- [ ] Crimson deepened both themes; text contrast AA
- [ ] Justify dropped; measure ~64ch
- [ ] Equation-number placement fixed (check a tall display eq, e.g. ch11 BDF2)
- [ ] Existing chapter classes unchanged (no fragment edits)
#### Verify
```
conda run -n heart-conduction python website/build/verify_site.py --chapters ch1,ch8,ch11,ch13,ch18,appendix-a --themes light,dark --out /tmp/verify_1_3
```
#### Exit Criteria
- [ ] Identity visibly shifted (serif×mono, crimson), both themes clean, equations well-placed, 0 console errors, chapters unedited.
#### Risk
Changing `--accent`/`--highlight` values ripples widely. Mitigation: values-only change, keep names; screenshot-sweep several chapters.

### Step 1.4: Wire cover + part pages into the SPA
**Model**: opus
#### Read First
- `website/app.js:84-141` (`buildChapterList`, `renderSidebar`, `renderChapterTocEntry`), `164-265` (`navigateTo`, `updateBottomNav`, `updateRightMargin`), `104-116` (`toc-part` rendered as non-clickable div).
- `website/build/html_to_pdf.py:36-42,95-109` — PDF reading order (title → for each part: divider → chapters). Mirror THIS order in the SPA.
- `website/chapters/title.html`, `part-i.html` — fragments to load.
#### Why
Give the website a cover + part-divider moments (currently PDF-only, orphaned on the web), with prev/next in reading order — closes the "cover & part pages orphaned" finding.
#### Implementation Spec
**Files to modify:** `website/app.js`
- `buildChapterList`: build `allChapters` as `[{id:'title',kind:'cover',num:'',title:'Cover'}, {id:'part-i',kind:'part',num:'Part I',title:'Single Cell Dynamics'}, …chapters (kind:'chapter')…, {id:'part-ii',kind:'part',title:'Tissue-Level Monodomain'}, …, {id:'appendices',kind:'part',title:'Reference Material'}, …]` in PDF order. **Pull `num`/`title` verbatim from the `toc.json` part entries** (e.g. appendices is `num:"Appendices", title:"Reference Material"` — the example must match toc, not be hand-typed). **Every entry MUST carry a `title`** — `updateBottomNav` sets the prev/next label from `.title` (app.js:237,246); an entry without a title would show `undefined`.
- `renderSidebar`: render each part header as a clickable link to its divider (keep the `toc-part` label styling) + a "Cover" link at top.
- `navigateTo`: works for `title`/`part-*` ids (files exist); skip `addSectionAnchors` + right-margin logic when `kind!=='chapter'`.
- `updateSidebarActive`: currently matches only `.toc-chapter[data-id]` (app.js:149) → part/cover links never highlight. Extend it to also set active on the part/Cover link whose id matches (so the promised active-state fires for non-chapter entries).
- `updateBottomNav`/`updateRightMargin`: tolerate entries without subsections (guard on `kind`).
#### Pseudocode
```
buildChapterList: allChapters=[cover]; for part in toc: allChapters.push({id:part.id,kind:'part',num:part.num,title:part.title}); for ch in part.chapters: allChapters.push({...ch,kind:'chapter'})
renderSidebar: prepend Cover link; render toc-part as <a href=#part-id> instead of <div>
navigateTo(id): fetch chapters/id.html (works for title/part-*); if kind!=='chapter' skip addSectionAnchors+updateRightMargin
updateBottomNav: label = entry.title (now always defined)
```
#### Test Spec
- `#title`, `#part-i`, `#part-iii` render the correct fragment (no ch1 fallback), with a defined prev/next label. Bottom-nav Cover → Part I → Ch 1; last-of-Part-I → Part II divider.
#### Checklist
- [ ] `allChapters` includes cover + 5 dividers in PDF order
- [ ] Sidebar: clickable parts + Cover; active-state works
- [ ] `#title`/`#part-*` load (no fallback to ch1)
- [ ] Prev/next flows through dividers; no crash on missing subsections
- [ ] PDF build unaffected (doesn't use app.js) — `--html-only` still assembles
#### Verify
```
# SPA nav is the thing under test → verify by screenshot: #part-i / #title must LAND on part/cover content (not ch1)
conda run -n heart-conduction python website/build/verify_site.py --chapters title,part-i,part-iii,ch1 --themes light,dark --out /tmp/verify_1_4
# (review /tmp/verify_1_4: title.png shows the cover, part-i.png shows the Part I divider — NOT ch1)
# separately confirm the PDF assembly path is UNCHANGED by this JS-only step (regression guard, not the nav test):
conda run -n heart-conduction python website/build/html_to_pdf.py --html-only -o /tmp/assemble_check.html && grep -c 'part-page\|title-page' /tmp/assemble_check.html
```
#### Exit Criteria
- [ ] Cover + parts reachable and in prev/next order with defined labels; `#title`/`#part-i` screenshots show cover/part content (no ch1 fallback); PDF assembly unchanged.
#### Risk
`addSectionAnchors`/scroll-spy assume chapter structure. Mitigation: guard on `kind==='chapter'`.

### Phase 1 Verification
```
conda run -n heart-conduction python website/build/verify_site.py --chapters title,part-i,ch1,ch2,ch5,ch8,ch10,ch11,ch13,ch18,appendix-a,appendix-b --themes light,dark --out /tmp/verify_p1
conda run -n heart-conduction python website/build/html_to_pdf.py -o /tmp/ccm_p1.pdf   # PDF still builds (~195pp)
```
### Phase 1 Exit Criteria — ✅ DONE 2026-07-06
- [x] Dark-mode figures legible + semantic across sampled chapters (verified ch2 FHN, ch5 current-grid, ch18 lattice stencils in dark)
- [x] No raw figure hex (0 residual; 62 values migrated via census map); blanket override deleted; equations clean in dark
- [x] Cover + parts reachable, correct prev/next (cover renders, "Next → Single Cell Dynamics" flow)
- [x] PDF builds, **exactly 195 pages** (no regression), no raw-LaTeX leak
- [x] 0 console errors across 24 chapter×theme combos, both themes; print-safety green
- Steps: 1.0 verify_site.py ✅ · 1.1 tokens ✅ · 1.2 census-migration + override delete ✅ · 1.3 serif×mono/crimson/justify/eqn ✅ · 1.4 cover+part nav ✅
- **NOT yet committed** — awaiting user go on the commit (branch: engine-tuner-cardiac-core has unrelated in-flight changes).
### Phase 1 Cleanup
- [ ] No stray debug logs in app.js / build scripts
- [ ] `migrate_figure_colors.py` idempotent (re-run = no diff)
- [ ] V5.3 untouched; no engine/`cardiac_core` files touched
- [ ] Content diff review: only SVG color attrs + nav wiring changed, zero prose/equation edits

**-> Commit point: `git commit` after Phase 1 passes** (`feat(textbook-web): themeable figures + shell refresh + cover/part nav`)

---

## Phase 2: Figure-Widget Framework

**Goal**: A tiny, print-safe convention + loader that mounts an interactive canvas over a static SVG fallback, proven end-to-end on ONE figure (FHN, Fig 2.1). De-risks all of Phase 3.
**Tier**: large
**Estimated scope**: 1 new loader, 1 new widget module + shared helpers, 1 chapter figure converted, PDF-leak test.

### Phase Context
- Widget contract: `export function mount(canvas, params) -> { redraw(), destroy() }`. Widgets read colors via `getComputedStyle(document.documentElement).getPropertyValue('--fig-*')` and redraw on theme change. All physics in-browser, no deps.
- Markup convention:
  ```html
  <figure class="fig" data-widget="fhn" data-params='{"I":0.5,"a":0.7,"b":0.8,"eps":0.08}'>
    <div class="fig-fallback"><svg …themeable static…></svg></div>
    <div class="fig-controls" hidden>…progressively revealed control rail…</div>
    <figcaption class="figure-caption"><span class="figure-label">Figure 2.1.</span> …</figcaption>
  </figure>
  ```
- Loader mounts ONLY in the SPA (imported by `index.html`/`app.js`), never by `html_to_pdf.py`. Print CSS also hides `canvas.fig-widget` and shows `.fig-fallback` as a double guard.
- Reuse prototype code: `rk4`, `simulateMS`, `apd90`, themeable-canvas `fit()`/`cvar()` from the durable prototype `plans/2026-07-03_refresh_prototype.html`.

### Step 2.1: Loader + `<figure data-widget>` convention + FHN reference widget
**Model**: opus
#### Read First
- **`plans/2026-07-03_refresh_prototype.html`** (durable in-repo prototype) — FHN block (`drawPhase`, `rk4`, pointer→IC inversion, slider wiring, reduced-motion branch) and `fit()`/`cvar()`. Port these. (Do NOT depend on the ephemeral scratchpad copy.)
- `website/index.html:127` — **`app.js` is loaded as a CLASSIC script and is an IIFE** (`(function(){…})()`). A classic-script IIFE CANNOT reference an ES-module `export`. This dictates the loader's module strategy below.
- `website/app.js:164-216` (`navigateTo`) — call the loader after `MathJax.typesetPromise`.
- `website/style.css:718` — `.figure svg{max-width:100%;height:auto}` (the new `.fig` wrapper must re-provide this for the fallback SVG).
- `website/chapters/ch2.html:109` — current `<div class="figure">` Fig 2.1 (now themeable) to keep as fallback.
#### Why
One reference implementation end-to-end (markup → loader → widget → theme swap → print fallback) sets the pattern every Phase-3 widget copies; getting the print gate AND the module boundary right here protects the PDF and avoids a broken import for all later widgets.
#### Implementation Spec
**Files to create:**
- `website/figures.js` — a **CLASSIC script** (NOT an ES module) that defines `window.mountFigures(root)`. Internally it uses **dynamic `import()`** (allowed from classic scripts) to load each widget ES module from `figures/`. This reconciles the boundary: `app.js` (classic IIFE) calls the global `window.mountFigures`; widget files stay clean ES modules. Keep mounted instances; `destroy()` them on re-navigation; theme hook (MutationObserver on `<html>` `data-theme` + `matchMedia` change) calls each instance's `redraw()`.
- `website/figures/_canvas.js` — ES module exporting shared `fit`, `cvar`, `rk4`, `simulateMS`, `apd90`.
- `website/figures/fhn.js` — ES module `export function mount(canvas, params) -> {redraw, destroy}`, porting the prototype's phase-plane widget.
**Files to modify:**
- `website/index.html` — add `<script src="figures.js"></script>` (classic, alongside `app.js`; order doesn't matter since `app.js` calls `window.mountFigures` only at navigate time).
- `website/app.js` — after typeset in `navigateTo`, `if (window.mountFigures) window.mountFigures(contentEl)`; destroy prior instances on renavigation (loader tracks them).
- `website/style.css` — `.fig`, `.fig-fallback`, `.fig-controls`, `canvas.fig-widget` styles **including `.fig svg, .fig-fallback svg{max-width:100%;height:auto}`** (restores the `.figure svg` rule lost by the `div.figure`→`figure.fig` reclass). **Fallback-visibility contract (screen):** `.fig-fallback` visible by DEFAULT (so no-JS / failed-mount shows the static SVG); the loader adds `.has-widget` to the `<figure>` only on SUCCESSFUL mount, and `.fig.has-widget .fig-fallback{display:none}` hides the SVG once the canvas is live (no duplicate figure). Reduced-motion still MOUNTS the widget (it renders a static integrated frame on the canvas, per the prototype) → fallback hidden, canvas shown. **Print contract:** `@media print{ canvas.fig-widget{display:none!important} .fig.has-widget .fig-fallback,.fig-fallback{display:block!important} .fig-controls{display:none!important} }` (print always shows the SVG regardless of `.has-widget`).
- `website/chapters/ch2.html` — wrap Fig 2.1 in `<figure class="fig" data-widget="fhn" …>` with the themeable SVG as `.fig-fallback`. (Confirm the Step-1 `.figure text` default also covers `figure.fig text` — it does, per 1.1.)
#### Pseudocode
```
// figures.js — classic script
window.mountFigures = async function(root){
  destroyAll();
  for (el of root.querySelectorAll('[data-widget]')){
    try {
      const mod = await import(`./figures/${el.dataset.widget}.js`);   // dynamic import OK from classic script
      const canvas = el-inserts <canvas class="fig-widget">;
      const inst = mod.mount(canvas, JSON.parse(el.dataset.params||'{}'));
      el.querySelector('.fig-controls').hidden = false;
      el.classList.add('has-widget');       // ← hides .fig-fallback on screen (CSS); success only
      instances.push(inst);
    } catch(e){ /* leave .fig-fallback visible; no .has-widget; console.warn */ }
  }
};
// theme hook: new MutationObserver(...).observe(documentElement,{attributes:true,attributeFilter:['data-theme']}) → instances.forEach(i=>i.redraw?.())
// app.js navigateTo(): after MathJax.typesetPromise([contentEl]) → window.mountFigures && window.mountFigures(contentEl)
```
#### Test Spec
- SPA: ch2 shows the live, draggable FHN widget; theme toggle keeps colors correct; reduced-motion → static integrated trajectory.
- Print: `verify_site.py` widget-leak assertion passes; `html_to_pdf.py` build shows the static SVG for Fig 2.1.
#### Checklist
- [ ] `figures.js` + `_canvas.js` + `fhn.js` created; dynamic import works
- [ ] ch2 Fig 2.1 wrapped; static SVG remains as fallback
- [ ] Widget mounts in SPA, drag+sliders+play work, theme-aware
- [ ] On successful mount `.fig-fallback` is hidden on screen (no duplicate SVG+canvas); forcing a mount failure leaves the static SVG visible
- [ ] `@media print` hides canvas / shows fallback (even with `.has-widget`)
- [ ] `html_to_pdf.py` unchanged; PDF Fig 2.1 = static SVG
- [ ] Re-navigation destroys prior instance (no leak / double-mount)
#### Verify
```
conda run -n heart-conduction python website/build/verify_site.py --chapters ch2 --themes light,dark --out /tmp/verify_2_1
conda run -n heart-conduction python website/build/html_to_pdf.py -o /tmp/ccm_p2.pdf
```
#### Exit Criteria
- [ ] FHN widget live + theme-correct in SPA; static fallback prints; leak assertion green; 0 console errors.
#### Risk
(a) Module boundary: making `figures.js` an ES module with `export` would break the classic-IIFE `app.js` call → mitigation: `figures.js` is a classic script exposing `window.mountFigures`, widget files are ES modules loaded via dynamic `import()`. (b) Pointer→data inversion must match the draw transform exactly (see prototype `place()` vs `X/Y`) → reuse the prototype's proven inversion. (c) Dynamic `import()` + `file://` fails → SPA already requires an http server; verify over http.

### Phase 2 Verification / Exit / Cleanup
```
conda run -n heart-conduction python website/build/verify_site.py --chapters ch2 --themes light,dark --out /tmp/verify_p2
conda run -n heart-conduction python website/build/html_to_pdf.py -o /tmp/ccm_p2.pdf
```
- [ ] One widget fully working + print-safe; leak assertion green; PDF ~195pp
- [ ] Cleanup: no console logs; instance destroy on renavigation verified; V5.3/engines untouched

**-> Commit point: `git commit` after Phase 2** (`feat(textbook-web): figure-widget framework + FHN reference widget`)

---

## Phase 3: Flagship Interactive Widgets

**Goal**: 5–7 more hero widgets, each following the Phase-2 pattern. Every widgetized figure keeps its Phase-1 themeable SVG as fallback.
**Tier**: large
**Estimated scope**: one module per widget; each a self-contained copy of the 2.1 pattern with different physics.

### Phase Context
- Per widget: (0) **first `grep`/read the target chapter to identify the EXACT figure by its caption/`figure-label`** — several chapters have multiple figures (e.g. ch10 has both an RK-slope diagram *and* a stability/CFL figure; 3.2 wraps the former, 3.3 the latter). Record the specific `Figure N.M` each widget replaces; (1) create `website/figures/<name>.js` (ES module) exporting `mount(canvas,params)->{redraw,destroy}`; (2) wrap THAT figure with `<figure class="fig" data-widget="<name>">` keeping its Phase-1-themed SVG as `.fig-fallback`; (3) validate the physics against a known value / engine; (4) keyboard-operable controls + `prefers-reduced-motion` static branch; (5) verify SPA live + PDF static.
- Reuse `figures/_canvas.js` helpers (`rk4`/`fit`/`cvar`/`simulateMS`/`apd90`) — do NOT copy them per widget.
- Physics correctness is non-negotiable (textbook). Cross-check each model against a cited value or the engine.
- Already ported/correct: FHN (2.1), Mitchell–Schaeffer AP + APD90 (durable prototype `plans/2026-07-03_refresh_prototype.html`).

### Step 3.1: Action-potential shaper (`ap.js`, Ch 3)
**Model**: opus · Mitchell–Schaeffer, sliders τ_close/τ_out/τ_in, live APD₉₀. Port from prototype. Validate: default params → APD₉₀ ~250–300 ms; larger τ_close ⇒ longer APD (monotone). Fallback: static AP-trace SVG.
### Step 3.2: RK step-size explorer (`rk.js`, Ch 10)
**Model**: opus · Integrate a test ODE (`y'=-y` or the AP upstroke) with Euler/RK2/RK4 at slider `h`; plot true vs numerical + global error. Validate: error slopes ~O(h),O(h²),O(h⁴) on a log-log check; RK4 ≈ exact at small h. Fallback: existing Ch 10 RK diagram.
### Step 3.3: CFL stability (`cfl.js`, Ch 10)
**Model**: opus · 1-D explicit diffusion of a Gaussian; slider Δt across `Δt ≤ h²/2D`; show bounded decay vs oscillatory blow-up. Validate: crossing the bound flips stability; matches Ch 10 CFL text. Fallback: static stability-region SVG.
### Step 3.4: Nernst / GHK (`nernst.js`, Ch 1/3)
**Model**: opus · Sliders for [K⁺]ᵢ/ₒ, [Na⁺], …; live E_ion (RT/zF·ln) + resting Vm (GHK). Validate: default concentrations → E_K ≈ −90 mV, E_Na ≈ +65 mV (matches Ch 1). Fallback: static bar/number SVG.
### Step 3.5: Propagating wavefront (`wave.js`, Ch 7/8)
**Model**: opus · 1-D monodomain cable (FHN or MS reaction + explicit diffusion), play button, live CV readout. Validate: CV scales ~√D (double D ⇒ ~1.41× CV); qualitative monodomain match. Fallback: static wavefront SVG. Highest effort — may split.
### Step 3.6 (stretch): HH gating (`hh.js`, Ch 1) and/or restitution (`restitution.js`, Ch 3/6)
**Model**: opus · Only if budget allows; same pattern.

Per-step Verify (template): `conda run -n heart-conduction python website/build/verify_site.py --chapters <chapter> --themes light,dark --out /tmp/verify_3_x`
Per-step Exit: widget live + theme-correct; physics validation value met; static fallback prints; 0 console errors.
Per-step Risk: model instability at slider extremes → clamp params to a validated range; blow-up handled without NaN spam.

### Phase 3 Verification / Exit / Cleanup
```
conda run -n heart-conduction python website/build/verify_site.py --chapters ch1,ch2,ch3,ch7,ch10 --themes light,dark --out /tmp/verify_p3
conda run -n heart-conduction python website/build/html_to_pdf.py -o /tmp/ccm_p3.pdf
```
- [ ] ≥6 widgets total live + print-safe; each physics-validated
- [ ] PDF still ~195pp, all fallbacks static; leak assertion green
- [ ] Cleanup: shared helpers live in `_canvas.js` only (no per-widget copy of rk4/fit); no console logs

**-> Commit point: `git commit` after Phase 3** (`feat(textbook-web): flagship interactive figure widgets`)

---

## Phase 4: Identity, Polish & PDF-Regression Verify

**Goal**: Cover moment + part openers, motion/a11y pass, and a hard confirmation the PDF didn't regress.
**Tier**: medium
**Estimated scope**: cover/part-opener markup+CSS, a11y sweep, final PDF diff.

### Step 4.1: Cover + part openers
**Model**: opus · Style `title.html` as a real cover (`.cover` from 1.3) and give each `part-*.html` a considered opener (big part number, one-line thesis). Content exists; presentation only. Verify: light+dark; prints as cover/part page.
### Step 4.2: Motion, a11y, reduced-motion, focus
**Model**: opus · Every widget: visible focus states, ARIA labels on sliders/buttons, full `prefers-reduced-motion` static branch; ambient motion respects it. Verify: keyboard-only operate one widget end-to-end; reduced-motion screenshot shows static figures.
### Step 4.3: PDF regression gate
**Model**: opus · Re-run full build; compare page count + spot content vs the pre-refresh 195-pp PDF; confirm 0 raw-LaTeX leak, 0 widget canvas, all figures present as static SVG. Update INDEX/KNOWLEDGE/README with the new website state.

### Phase 4 Verification / Exit / Cleanup
```
conda run -n heart-conduction python website/build/html_to_pdf.py -o Cardiac_Computational_Modeling.pdf
conda run -n heart-conduction python website/build/verify_site.py --chapters title,part-i,part-ii,part-iii,part-iv,ch1 --themes light,dark --out /tmp/verify_p4
```
- [ ] Cover + part openers land in both web + PDF
- [ ] a11y: keyboard + reduced-motion pass
- [ ] PDF ~195pp, no regressions, figures all static in print
- [ ] README completion criteria + KNOWLEDGE updated

**-> Commit point: `git commit` after Phase 4** (`feat(textbook-web): cover/part identity + a11y + verified PDF`)

---

## Final Cleanup
- [ ] `migrate_figure_colors.py` + `verify_site.py` documented (module docstring, usage) and idempotent
- [ ] `_canvas.js` is the single home for rk4/fit/cvar/simulateMS/apd90 — no duplication across widgets
- [ ] No hardcoded hex anywhere in `chapters/*.html` SVGs; no blanket SVG override in `style.css`
- [ ] `html_to_pdf.py` still injects only MathJax + PRINT_CSS (never `figures.js`)
- [ ] V5.3 untouched; no `cardiac_core`/engine files touched; no code duplication into engines
- [ ] Any figures/screenshots kept for the record saved under `media/textbook/...` (not next to scripts)
- [ ] Update `README.md` completion criteria + `KNOWLEDGE.md` "Current State" + INDEX with refreshed-website status
- [ ] Archive this plan:
```bash
mkdir -p Research/Active/textbook/plans
cp Research/Active/textbook/PLAN.md "Research/Active/textbook/plans/$(date +%Y-%m-%d)_website-refresh-interactive-figures.md"
```

## Mutation Log
**REVISED 2026-07-03 (adversarial /audit round 1 — 2 HIGH, 4 MED, 8 LOW; all addressed):**
- HIGH-1 (token palette too small): census confirms 61 distinct hexes. Expanded the token system to categorical(7)+ink/neutral(4)+stage+tint(6); Step 1.2 is now census-first (`--census`→`figure_color_map.json`→review→`--apply`); Success Criterion #2 relaxed to "resolves to a token or documented per-figure exception; greys collapse (not semantic)".
- HIGH-2 (wrong theme-scope model): site has ONLY `:root` + `[data-theme="dark"]` (JS-toggled), no `@media prefers-color-scheme`/`[data-theme="light"]`. Corrected all scope references to TWO scopes; forbade adding an `@media` token block (would leak dark colors into a light-toggled dark-OS user); Risk mitigation now "grep == 2".
- MED-1 (delete line 810 ⇒ ch18 group-inherited `<text>` goes dark-on-dark): added scoped themeable default `.figure text,figure.fig text{fill:var(--fig-axis)}` in Step 1.1 to replace line 810; migration now handles `<g fill=…>`; added ch18 to the dark spot-check.
- MED-2 (verify leak-assertion trivially true): replaced with (a) assert `assemble_html()` never references `figures.js`/`mountFigures`, (b) headless-render the print HTML and assert 0 `canvas.fig-widget`.
- MED-3 (app.js IIFE vs figures.js ES-module boundary): `figures.js` is now a CLASSIC script exposing `window.mountFigures`, using dynamic `import()` for widget ES modules.
- MED-4 (Steps 1.1/1.3/1.4 missing Pseudocode): added.
- LOWs: durable prototype persisted → `plans/2026-07-03_refresh_prototype.html` (refs repointed off scratchpad); `.fig svg{max-width}` restored; part/cover `allChapters` entries carry `title` (nav-label fix); Step 1.4 Verify relabeled (screenshot = the real nav test); Phase 3 now requires identifying the exact target figure; utility classes de-emphasized in favor of the themeable default; migration scoped to `<svg>` spans + skips `mjx-container`.

**REVISED 2026-07-03 (adversarial /audit round 2 — CONVERGED: 0 crit, 0 high, 2 med, 7 low; all 6 round-1 fixes verified genuinely reflected, incl. independent re-derivation that the MED-1 text default is sound). Folded in the round-2 gaps:**
- MED-B1 (fallback-visibility): Step 2.1 now specifies `.fig-fallback` visible by default; loader adds `.has-widget` on SUCCESSFUL mount → `.fig.has-widget .fig-fallback{display:none}` on screen (no duplicate SVG+canvas); try/catch keeps the SVG on failure; print always shows the SVG.
- MED-B2 (named colors): Step 1.2 census now also captures `white`(×42)/`black`/keywords → `--fig-stage`/`--fig-ink` (flagged), not just `#hex`.
- LOWs: repointed the two lingering scratchpad refs (lines 27, 302) → durable prototype; fixed the appendices example title (`Reference Material`, from toc.json) + "pull verbatim from toc"; added `updateSidebarActive` extension for part/cover active-state; defined the `--fig-axis`→`--fig-ink` alias in pseudocode; Step 1.1 Verify now loops all tokens (not just `fig-crimson`); Step 1.2 Exit wording reconciled with the "documented exception" criterion.
- Round-2 LOWs left as-is (non-blocking polish): grep-count-vs-alias nuance noted inline.

**Plan is CONVERGED and ready to execute** (no CRITICAL/HIGH across two audit rounds). Awaiting user "go" before Phase 1.

_(execution mutations appended below: `**MUTATED {date}**: Step X.Y {SKIPPED|SPLIT|INSERTED} — {reason}`)_
