# Textbook Website Plan — Cardiac Computational Modeling

## Overview

Transform the single-file `Bidomain_Textbook.html` into a polished, multi-page interactive website with modern textbook UI conventions. The site will be **fully static** (no server needed) — just open `index.html` in a browser.

---

## Architecture

### Single-Page Application (SPA) Approach
- **One HTML file** (`index.html`) containing the full navigation shell
- **One content file** (`content.js`) exporting each chapter as an HTML string fragment
- **MathJax loaded from CDN** (no npm dependency for end users)
- Chapter content is swapped into the main viewport dynamically
- URL hash routing: `#ch7`, `#ch17-4` for deep-linking

### Why SPA over Multi-Page?
The textbook is already a single HTML file. Splitting into 19+ files adds complexity without benefit for a static site. The SPA approach gives us instant navigation, reading position persistence, and a smoother UX — all without a build step.

---

## UI Components

### 1. Left Sidebar — Table of Contents
- Collapsible four-level tree: Part → Chapter → Section → Subsection
- Current section highlighted with accent color
- Sections auto-expand when reading that chapter
- Collapse/expand toggle button (hamburger icon on mobile)
- Width: 280px desktop, full-screen overlay on mobile
- Scroll position preserved independently from content

### 2. Top Bar
- **Title**: "Cardiac Computational Modeling" (truncated on mobile)
- **Search**: Full-text search across all chapters with highlighted results
- **Dark/Light toggle**: Theme switch with system preference detection
- **PDF download button**: Direct link to the generated PDF
- **Progress bar**: Thin line showing reading progress through current chapter

### 3. Main Content Area
- Max-width ~52em (matching the current PDF layout)
- All existing CSS classes preserved: `.insight-*`, `.equation-block`, `.chapter-intro`, etc.
- MathJax renders in-place with `tex-svg.js`
- SVG figures scale responsively
- Tables remain horizontally scrollable on mobile

### 4. Right Margin (Desktop Only, ≥1400px)
- **Equation quick-reference**: Pinned list of key equations for current chapter
- **"On this page"** mini-TOC showing sections of current chapter
- Fades in/out based on viewport width

### 5. Bottom Navigation
- **Previous / Next chapter** buttons with chapter titles
- Keyboard navigation: ← / → arrow keys

---

## Features

### Reading Experience
| Feature | Implementation |
|---------|---------------|
| **Dark mode** | CSS custom properties swap; prefers-color-scheme detection |
| **Font size control** | Three sizes (small/medium/large) via CSS `font-size` on root |
| **Reading progress** | Thin progress bar per chapter + overall % in sidebar |
| **Bookmark last position** | localStorage saves chapter + scroll offset |
| **Smooth scroll to section** | `scrollIntoView({ behavior: 'smooth' })` on TOC click |

### Search
- Client-side full-text index built at load time (lightweight; ~100KB index for 9000 lines)
- Results show chapter, section, and matching text snippet
- Click result → navigate to chapter, scroll to match, highlight

### Mobile
- Sidebar becomes hamburger-triggered overlay
- Right margin hidden
- Equations scale with `overflow-x: auto`
- Touch-friendly navigation buttons

### Accessibility
- Semantic HTML (`<nav>`, `<article>`, `<section>`, `<aside>`)
- ARIA labels on all interactive elements
- Keyboard navigable (Tab, Enter, Escape, arrows)
- High-contrast dark mode option
- Skip-to-content link

---

## Visual Design

### Color Palette (Light Mode)
```
--bg:        #ffffff
--text:      #1a1a2e
--sidebar:   #f8f9fc
--accent:    #16213e (headings)
--accent2:   #0f3460 (subheadings)
--highlight: #e94560 (chapter numbers, active TOC)
--light-bg:  #f4f6fb (equation boxes)
```

### Color Palette (Dark Mode)
```
--bg:        #1a1a2e
--text:      #e4e6eb
--sidebar:   #16213e
--accent:    #8eafc8
--accent2:   #a3c4db
--highlight: #e94560
--light-bg:  #232340
```

### Typography
- **Body**: EB Garamond (serif) — already used in PDF
- **Code**: Source Code Pro (mono) — already used in PDF
- **UI elements**: Inter or system sans-serif — clean, modern sidebar/nav
- **Math**: MathJax SVG output (matches PDF exactly)

---

## File Structure
```
website/
├── index.html          ← Main shell: nav, sidebar, theme, search
├── style.css           ← All styles (light + dark + responsive)
├── app.js              ← SPA router, search, theme, progress tracking
├── chapters/           ← One .html fragment per chapter
│   ├── ch01.html
│   ├── ch02.html
│   ├── ...
│   └── appendix-a.html
└── assets/
    └── favicon.svg
```

### Build Process
A Python script (`build_website.py`) will:
1. Parse `Bidomain_Textbook.html`
2. Split into chapter fragments (one per `<div class="chapter">`)
3. Extract the TOC structure
4. Generate `index.html` with the shell + inline chapter loader
5. Copy CSS, preserving all existing classes
6. Output to `website/` directory

---

## Implementation Phases

### Phase 1 — Core Shell (this session)
- [ ] Write `build_website.py` to split the textbook into chapter fragments
- [ ] Create `index.html` with sidebar TOC, main content area, bottom nav
- [ ] Create `style.css` with light/dark themes, responsive layout
- [ ] Create `app.js` with hash routing, chapter loading, TOC highlighting
- [ ] MathJax CDN integration with identical config to PDF version

### Phase 2 — Polish
- [ ] Full-text search
- [ ] Reading progress persistence (localStorage)
- [ ] Font size controls
- [ ] Right margin mini-TOC
- [ ] Keyboard navigation
- [ ] Mobile hamburger menu

### Phase 3 — Extras (future)
- [ ] Equation cross-reference tooltips (hover eq number → see equation)
- [ ] Print stylesheet (matches PDF layout)
- [ ] Service worker for offline reading
- [ ] Chapter-level PDF download

---

## Technical Notes

- **MathJax**: Use CDN `https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js` with identical config
- **No build tools required**: Pure HTML/CSS/JS — open in any browser
- **Chapter loading**: Fetch individual chapter HTML fragments → inject into `<main>` → trigger MathJax typeset
- **Performance**: Lazy-load chapters on navigation (not all at once); MathJax typesets only the visible chapter
- **Deployment**: Can be hosted on GitHub Pages, Netlify, or just opened as local files
