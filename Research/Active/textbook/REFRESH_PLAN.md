# Website Refresh — Rebuild Plan (2026-07-03)

> Companion to `audits/UI_AUDIT_2026-07-03.md`. Interactive proof-of-direction (live figures + the
> dark-mode fix, rendered in both themes): **artifact** `a514e90d` →
> https://claude.ai/code/artifact/a514e90d-e538-4b88-9387-1b9290636bdb
> Source of the prototype widgets: scratchpad `refresh_pitch.html` (FHN + Mitchell–Schaeffer, RK4,
> themeable canvas — directly reusable in the real build).

## Direction (decided, pending user confirmation — set while user was away)
- **Aesthetic:** *explorable explanation* — figure-forward, figures break out of the reading column into a
  wider "instrument stage," interactivity is the star. Keep the book's serif reading voice where it earns it.
- **Interactivity:** *flagship subset* — 6–8 hero interactive widgets on the highest-value figures; every
  remaining figure upgraded to clean **themeable static SVG** (no widget).
- **Scope/stack:** *enhance vanilla in place, no build step.* Keep the SPA + `toc.json` + `chapters/*.html`.
  Interactivity is progressive enhancement **over a static SVG fallback** so the Playwright PDF assembly
  (`website/build/html_to_pdf.py`) keeps producing the 195-pp PDF unchanged.

> These three are the forks that shape everything. If the user wants a different call (e.g. all-figures
> interactive, or a build step, or the more conservative "elevated book" look), re-scope Phases 2–4.

## Design system (from the prototype — carry into `style.css`)
**Color** — one accent, arterial crimson (deepened descendant of the current `#e94560`):
```
light  --crimson #C31D38   dark --crimson #FF5468
figure system (themeable, reads on BOTH grounds) — EXPANDED per /audit to cover the 61 real figure hexes:
  categorical hues: --fig-crimson --fig-blue --fig-teal --fig-green --fig-amber --fig-orange --fig-purple
  ink/neutral:      --fig-ink(=--fig-axis) --fig-muted --fig-faint --fig-grid --fig-stage
  panel tints:      --fig-tint-{blue,teal,red,amber,green,purple}
  (exact light/dark values finalized against the `dataviz` colorblind/contrast palette in PLAN Step 1.1)
always-dark "instrument" panel (hero monitor, deliberate single-theme): --fig-stage #0A0C14 …
```
Cool-biased neutrals (not default grey). **This site's dark mode is JS-toggled** (`data-theme` on `<html>`,
defaulting `light`); it has NO `@media prefers-color-scheme` block. So define tokens in exactly TWO scopes:
`:root` (light defaults) + `[data-theme="dark"]` (dark overrides). Do NOT add an `@media` block (no
`[data-theme="light"]` reset exists → it would leak dark figure colors into a light-toggled dark-OS view).
Kill the root-cause bug: **delete the blanket override `style.css:810-812`** (both the `svg text{fill}` line 810
AND `svg line,path{stroke}` 811-812) and replace with a scoped themeable default `.figure text{fill:var(--fig-axis)}`.

**Type** — the identity is a deliberate contrast, **literary serif × instrument mono**:
- serif (reading + display): `"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif`
- mono (eyebrows, axis labels, control readouts, captions): `ui-monospace,"SF Mono","Cascadia Code",Menlo,…`
- Use system stacks (no webfont CDN → no silent fallback / no CSP issue). `tabular-nums` on all data.

**Layout** — serif reading column ~64ch; figures break out to a wider stage + control rail; quiet left nav;
the hero is a **live figure**, not a title.

## Figure architecture (the core new convention)
One markup shape, print-safe by construction:
```html
<figure class="fig" data-widget="fhn-phaseplane" data-params='{"I":0.5,"a":0.7,"b":0.8,"eps":0.08}'>
  <svg class="fig-fallback" ...>…themeable static SVG (what the PDF prints)…</svg>
  <figcaption><b>Figure 2.1</b> …</figcaption>
</figure>
```
- A small loader (`figures.js`) scans `[data-widget]` after chapter load, and **replaces** `.fig-fallback`
  with a live canvas widget when JS + not-print. No `data-widget` → the SVG is the figure, done.
- PDF path: Playwright runs with JS but we gate on `@media print` / a `?print` flag so widgets never mount;
  the static SVG is what renders. **Every figure keeps a static representation** — hard requirement.
- Widgets are plain ES modules in `website/figures/` (e.g. `fhn.js`, `ap.js`), each exporting
  `mount(canvas, params)`. They read colors from CSS `--fig-*` and redraw on theme change.
- Kill the root-cause bug: **delete** `style.css:810-812` (`svg text{fill}` + `svg line,path{stroke}`) and give
  every static SVG themeable colors via inline `style="…:var(--fig-*)"` (from the census map) + a scoped
  `.figure text{fill:var(--fig-axis)}` default (`var()` only resolves in `style=`, never a presentation attribute).

## Flagship interactive widgets (6–8; prototype-proven ones marked ✓)
1. ✓ **FHN phase plane** (Ch 2, Fig 2.1) — drag initial condition, sliders a/b/ε/I, live limit-cycle detection.
2. ✓ **Action-potential shaper** (Ch 3) — Mitchell–Schaeffer, τ_close/τ_out/τ_in sliders, live APD₉₀ readout.
3. **RK step-size explorer** (Ch 10) — slider on h; show true vs Euler vs RK2/RK4 error shrinking.
4. **CFL stability** (Ch 10) — slider on Δt across the stability bound; watch a diffusing pulse stay bounded vs blow up.
5. **Nernst / GHK calculator** (Ch 1/3) — concentration sliders → live E_ion, driving the resting potential.
6. **Propagating wavefront** (Ch 7/8) — 1-D monodomain cable, play button, CV readout; curvature if time allows.
7. *(stretch)* **HH gating** (Ch 1) — voltage clamp, watch m/h/n relax; conductance = ḡ·m³h.
8. *(stretch)* **Restitution curve** (Ch 3/6) — pace at decreasing BCL, trace APD-restitution.

## Phases
**Phase 1 — Fix & re-shell (CSS only, no content risk).**
Delete the dark-mode SVG override; introduce the `--fig-*` token system + `.fig-crimson|teal|amber` classes;
rebuild `style.css` shell + type scale to the design system; wire the orphaned `title.html` / `part-*.html`
into the SPA (clickable part rows + a real cover/landing); tighten typography (drop `justify`, fix eq-number
placement). Verify: light+dark, existing static figures still legible in dark.

**Phase 2 — Figure framework (JS, progressive enhancement).**
`figures.js` loader + the `<figure data-widget>` convention + one reference widget (`fhn.js`) ported from the
prototype. Confirm the PDF build still renders the static SVG (no widget mount under print). This de-risks everything after.

**Phase 3 — Hero widgets + static sweep (the headline work).**
Build widgets 2–6 (7–8 stretch). In parallel, convert every remaining static figure to the themeable format
(replace hardcoded hex; add fallback class). Each widget: correct physics (validate against engine/known
values), keyboard-operable, reduced-motion aware.

**Phase 4 — Identity & polish + verify.**
Cover moment, part openers, motion pass, a11y + `prefers-reduced-motion`, focus states; then **re-run
`/textbook-compile`** and diff the PDF to confirm zero print regression. Update INDEX/KNOWLEDGE.

## Guardrails
- Never let a figure exist only as a canvas — static SVG fallback is mandatory (PDF).
- No build step / no framework / no external runtime deps (CSP + PDF + "just works" on a static host).
- Physics must be correct (this is a textbook): validate each widget's model against the engines / literature.
- Keep the good parts: insight boxes, search, scroll-spy, font/theme toggles, keyboard nav, MathJax-SVG.
```
Pipeline note: this doc is the pre-blueprint design brief. Run /blueprint to expand Phases 1–4 into a
cold-start PLAN.md with per-file steps once direction is confirmed.
```
