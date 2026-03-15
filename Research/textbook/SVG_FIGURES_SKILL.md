# SVG Figure Generation — Research Publication Quality

## Purpose
Generate SVG diagrams suitable for inclusion in research papers, textbooks, and technical PDF documents rendered via HTML + MathJax + Playwright.

## Design Principles

### 1. Publication Standards
- **Minimum viewBox**: 400×300 for simple diagrams, 600×400 for complex ones. Never use 200×200 — it produces blurry, cramped results.
- **Font sizes**: Minimum 14px for labels, 12px for annotations, 16px for titles. In a 400×300 viewBox, 14px is the floor.
- **Line widths**: Minimum 1.5px for primary lines (axes, arrows, connections), 0.75px for grid/secondary lines. Thin hairlines disappear in PDF.
- **Margins**: Leave at least 10% padding on all sides of the viewBox for labels and breathing room.

### 2. Color Palette — Colorblind-Safe (Okabe-Ito)

| Role | Color | Hex |
|------|-------|-----|
| Primary / Active node | Blue | `#0072B2` |
| Secondary / Neighbor | Orange | `#E69F00` |
| Tertiary / Diagonal | Vermillion | `#D55E00` |
| Accent / Highlight | Sky Blue | `#56B4E9` |
| Neutral / Grid | Gray | `#999999` |
| Background grid | Light Gray | `#E0E0E0` |
| Text / Labels | Dark Gray | `#333333` |

Never rely on color alone to convey information — also use shape, line style, or label.

### 3. Arrow and Marker Definitions
Always define clean arrowheads in a `<defs>` block with unique IDs (prefix with figure name to avoid conflicts):

```svg
<defs>
  <marker id="fig17-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L8,3 L0,6 Z" fill="#333333"/>
  </marker>
</defs>
```

- `markerUnits="strokeWidth"` ensures arrows scale with line thickness.
- `refX` should be close to the marker width so the arrowhead tip touches the target.

### 4. Text and Labels
- **Never use MathJax/LaTeX syntax** (`$i=1$`) inside SVG `<text>` elements — it will NOT render. Use plain text or Unicode.
- For subscripts: use `<tspan>` with `dy` and smaller font-size, or Unicode subscript characters (₀₁₂₃₄₅₆₇₈₉).
- For bold: use `font-weight="600"`.
- For italic: use `font-style="italic"`.
- Always set `font-family="'Helvetica Neue', Helvetica, Arial, sans-serif"` for clean rendering.
- Use `text-anchor="middle"` for centered labels, `"start"` or `"end"` as needed.

### 5. Node Representation
- **Filled circles** for active/labeled nodes: `r="6"` to `r="10"` depending on viewBox size.
- **Open circles** (stroke only) for inactive/background: `stroke-width="1.5"`, `fill="white"`.
- **Center/special nodes**: larger radius (r="8"–"10") with distinct fill.
- **Labels**: placed outside the node (offset 15–20 units), never overlapping.

### 6. Grid Lines
- Use `stroke-dasharray="4,4"` for background grids — dashed lines recede visually.
- Grid color: `#E0E0E0` or lighter — grids must not compete with the diagram.
- Grid spacing should match the lattice spacing.

### 7. Layout for Multiple Sub-Figures
When showing related diagrams side by side (e.g., D2Q5, D2Q9, D3Q7):
- Use a single wide SVG with sub-figures separated by whitespace.
- Sub-figure labels: **(a)**, **(b)**, **(c)** in bold 16px, positioned below each sub-figure.
- Maintain consistent scale — same grid spacing, same node sizes.
- Recommended viewBox: 900×350 for 3-panel, 600×350 for 2-panel.

### 8. 3D Perspective Diagrams (Isometric)
For 3D lattices (D3Q7, D3Q19, D3Q27):
- Use isometric projection with consistent transform.
- Depth cues: nodes further back have lower opacity (0.4–0.6) and slightly smaller radius.
- Hidden edges: dashed (`stroke-dasharray="3,3"`) or omitted.
- Front-facing edges: solid, full opacity.

### 9. Figure Caption Convention
Captions go OUTSIDE the SVG in a `<div class="figure-caption">`:
```html
<div class="figure-caption"><strong>Figure N.X:</strong> Description.</div>
```

### 10. Rendering Context
SVGs are embedded in HTML → PDF via Playwright (headless Chromium):
- `viewBox` controls aspect ratio; `style="max-width:Xpx"` controls rendered size.
- Single figures: `max-width: 400px` to `500px`.
- Multi-panel: `max-width: 700px` to `900px`.
- Always: `display:block; margin:1.5em auto;` for centering.

## Anti-Patterns (DO NOT)
- Do NOT use viewBox smaller than 300×200
- Do NOT use font-size below 12px
- Do NOT put `$LaTeX$` inside SVG text elements
- Do NOT use pure primary colors (#FF0000, #0000FF) — harsh and inaccessible
- Do NOT overlap labels with nodes or arrows
- Do NOT use more than 5 colors in one figure
- Do NOT omit arrowhead definitions
- Do NOT make grid lines the same weight as foreground elements

## Checklist Before Committing
1. [ ] viewBox ≥ 400×300 (wider for multi-panel)
2. [ ] All text plain (no LaTeX), ≥ 14px
3. [ ] Arrowheads in `<defs>` with unique IDs
4. [ ] Colorblind-safe palette
5. [ ] Grid lines dashed and light
6. [ ] Labels clear of nodes and arrows
7. [ ] Correct `max-width` for intended size
8. [ ] Caption outside SVG in `figure-caption` div
