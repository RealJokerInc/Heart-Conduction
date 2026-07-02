/* ══════════════════════════════════════════════════════════
   Cardiac Computational Modeling — Textbook SPA
   ══════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  // ─── State ───
  let tocData = [];
  let allChapters = [];   // flat list of {id, num, title, subsections}
  let currentChapter = null;
  let searchIndex = [];   // [{chapterId, chapterTitle, text, element}]
  let fontSizeLevel = 1;  // 0=small, 1=base, 2=large

  // ─── DOM refs ───
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => [...document.querySelectorAll(sel)];

  const sidebarEl = $('#sidebar');
  const sidebarTocEl = $('#sidebar-toc');
  const sidebarOverlay = $('#sidebar-overlay');
  const sidebarToggle = $('#sidebar-toggle');
  const contentEl = $('#chapter-content');
  const bottomNav = $('#bottom-nav');
  const navPrev = $('#nav-prev');
  const navNext = $('#nav-next');
  const navPrevLabel = $('#nav-prev-label');
  const navNextLabel = $('#nav-next-label');
  const progressBar = $('#progress-bar');
  const themeToggle = $('#theme-toggle');
  const fontToggle = $('#font-toggle');
  const searchToggle = $('#search-toggle');
  const searchPanel = $('#search-panel');
  const searchInput = $('#search-input');
  const searchResults = $('#search-results');
  const rightMargin = $('#right-margin');
  const rightMarginToc = $('#right-margin-toc');

  // ══════════════════════════════════════════════════════════
  //  INITIALIZATION
  // ══════════════════════════════════════════════════════════

  async function init() {
    // Load TOC data
    try {
      const resp = await fetch('toc.json');
      tocData = await resp.json();
    } catch (e) {
      console.error('Failed to load toc.json:', e);
      contentEl.innerHTML = '<p style="color:red;">Failed to load table of contents. Make sure toc.json is present.</p>';
      return;
    }

    // Build flat chapter list
    buildChapterList();

    // Build sidebar TOC
    renderSidebar();

    // Restore theme
    initTheme();

    // Restore font size
    initFontSize();

    // Setup event listeners
    setupEvents();

    // Navigate to hash or first chapter
    const hash = window.location.hash.slice(1);
    if (hash && allChapters.find(c => c.id === hash)) {
      navigateTo(hash);
    } else {
      // Try to restore last position
      const last = localStorage.getItem('ccm-last-chapter');
      if (last && allChapters.find(c => c.id === last)) {
        navigateTo(last);
      } else {
        navigateTo(allChapters[0]?.id || 'ch1');
      }
    }
  }

  function buildChapterList() {
    allChapters = [];
    for (const entry of tocData) {
      if (entry.type === 'part') {
        for (const ch of entry.chapters) {
          allChapters.push(ch);
        }
      } else {
        allChapters.push(entry);
      }
    }
  }

  // ══════════════════════════════════════════════════════════
  //  SIDEBAR
  // ══════════════════════════════════════════════════════════

  function renderSidebar() {
    let html = '';

    for (const entry of tocData) {
      if (entry.type === 'part') {
        html += `<div class="toc-part">${entry.num} — ${entry.title}</div>`;
        for (const ch of entry.chapters) {
          html += renderChapterTocEntry(ch);
        }
      } else {
        html += renderChapterTocEntry(entry);
      }
    }

    sidebarTocEl.innerHTML = html;
  }

  function renderChapterTocEntry(ch) {
    // Extract chapter number (e.g., "1" from "Chapter 1", "A" from "Appendix A", "B" from "Appendix B")
    // Also handle bare "Appendix" (used for References section)
    const numMatch = ch.num.match(/(\d+|[A-Z])(?:\s|$)/);
    const shortNum = numMatch ? numMatch[1] : '';

    let html = `<a class="toc-chapter" data-id="${ch.id}" href="#${ch.id}">`;
    if (shortNum) {
      html += `<span class="toc-chapter-num">${shortNum}.</span>`;
    }
    html += `${escapeHtml(ch.title)}</a>`;

    if (ch.subsections && ch.subsections.length > 0) {
      html += `<div class="toc-sections" data-parent="${ch.id}">`;
      for (const sub of ch.subsections) {
        // Clean the title: remove &ensp; entities
        const cleanTitle = sub.title.replace(/&ensp;/g, ' ').replace(/\s+/g, ' ').trim();
        html += `<a class="toc-section" data-id="${ch.id}" data-anchor="${sub.anchor}" href="#${ch.id}">${escapeHtml(cleanTitle)}</a>`;
      }
      html += `</div>`;
    }

    return html;
  }

  function updateSidebarActive(chapterId) {
    // Remove all active states
    $$('.toc-chapter.active').forEach(el => el.classList.remove('active'));
    $$('.toc-sections.expanded').forEach(el => el.classList.remove('expanded'));

    // Set active chapter
    const activeLink = $(`.toc-chapter[data-id="${chapterId}"]`);
    if (activeLink) {
      activeLink.classList.add('active');
      // Expand its sections
      const sections = $(`.toc-sections[data-parent="${chapterId}"]`);
      if (sections) sections.classList.add('expanded');
      // Scroll sidebar to show active item
      activeLink.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }

  // ══════════════════════════════════════════════════════════
  //  NAVIGATION
  // ══════════════════════════════════════════════════════════

  async function navigateTo(chapterId, scrollAnchor) {
    if (currentChapter === chapterId && !scrollAnchor) return;

    // Load chapter HTML
    try {
      const resp = await fetch(`chapters/${chapterId}.html`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      let html = await resp.text();

      // Add id anchors to h3 elements for scroll targeting
      html = addSectionAnchors(html);

      contentEl.innerHTML = html;
      currentChapter = chapterId;

      // Update URL hash
      history.replaceState(null, '', `#${chapterId}`);

      // Save position
      localStorage.setItem('ccm-last-chapter', chapterId);

      // Typeset MathJax
      if (window.MathJax && window.MathJax.typesetPromise) {
        await MathJax.typesetPromise([contentEl]);
      }

      // Update sidebar
      updateSidebarActive(chapterId);

      // Update bottom nav
      updateBottomNav(chapterId);

      // Update right margin
      updateRightMargin(chapterId);

      // Build search index for this chapter
      addToSearchIndex(chapterId);

      // Scroll to top or anchor
      if (scrollAnchor) {
        const target = document.getElementById(`sec-${scrollAnchor}`);
        if (target) {
          setTimeout(() => target.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
        }
      } else {
        window.scrollTo({ top: 0, behavior: 'instant' });
      }

    } catch (e) {
      console.error(`Failed to load chapter ${chapterId}:`, e);
      contentEl.innerHTML = `<p style="color: var(--highlight);">Failed to load chapter. (${e.message})</p>`;
    }
  }

  function addSectionAnchors(html) {
    // Add id="sec-X-Y" to h3 elements based on their section number
    return html.replace(/<h3([^>]*)>([\s\S]*?)<\/h3>/g, (match, attrs, inner) => {
      const numMatch = inner.match(/(\d+\.\d+|[A-Z]\.\d+)/);
      if (numMatch) {
        const anchor = numMatch[1].replace('.', '-');
        return `<h3${attrs} id="sec-${anchor}">${inner}</h3>`;
      }
      return match;
    });
  }

  function updateBottomNav(chapterId) {
    const idx = allChapters.findIndex(c => c.id === chapterId);

    if (idx > 0) {
      const prev = allChapters[idx - 1];
      navPrev.classList.remove('hidden');
      navPrev.dataset.id = prev.id;
      navPrevLabel.textContent = prev.title;
    } else {
      navPrev.classList.add('hidden');
    }

    if (idx < allChapters.length - 1) {
      const next = allChapters[idx + 1];
      navNext.classList.remove('hidden');
      navNext.dataset.id = next.id;
      navNextLabel.textContent = next.title;
    } else {
      navNext.classList.add('hidden');
    }
  }

  function updateRightMargin(chapterId) {
    const ch = allChapters.find(c => c.id === chapterId);
    if (!ch || !ch.subsections || ch.subsections.length === 0) {
      rightMarginToc.innerHTML = '';
      return;
    }

    let html = '';
    for (const sub of ch.subsections) {
      const cleanTitle = sub.title.replace(/&ensp;/g, ' ').replace(/\s+/g, ' ').trim();
      html += `<li><a href="#" data-anchor="${sub.anchor}">${escapeHtml(cleanTitle)}</a></li>`;
    }
    rightMarginToc.innerHTML = html;
  }

  // ══════════════════════════════════════════════════════════
  //  SEARCH
  // ══════════════════════════════════════════════════════════

  const searchIndexCache = new Map();

  function addToSearchIndex(chapterId) {
    if (searchIndexCache.has(chapterId)) return;

    const ch = allChapters.find(c => c.id === chapterId);
    if (!ch) return;

    // Extract text from the loaded content (strip HTML tags)
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = contentEl.innerHTML;
    const paragraphs = tempDiv.querySelectorAll('p, h3, h4, .insight, li, td');

    const entries = [];
    paragraphs.forEach(el => {
      const text = el.textContent.trim();
      if (text.length > 20) {
        entries.push({
          chapterId: ch.id,
          chapterTitle: `${ch.num}: ${ch.title}`,
          text: text.substring(0, 300)
        });
      }
    });

    searchIndexCache.set(chapterId, entries);
  }

  // Build full search index by loading all chapters in background
  async function buildFullSearchIndex() {
    for (const ch of allChapters) {
      if (searchIndexCache.has(ch.id)) continue;
      try {
        const resp = await fetch(`chapters/${ch.id}.html`);
        if (!resp.ok) continue;
        const html = await resp.text();
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = html;
        const paragraphs = tempDiv.querySelectorAll('p, h3, h4, .insight, li, td');
        const entries = [];
        paragraphs.forEach(el => {
          const text = el.textContent.trim();
          if (text.length > 20) {
            entries.push({
              chapterId: ch.id,
              chapterTitle: `${ch.num}: ${ch.title}`,
              text: text.substring(0, 300)
            });
          }
        });
        searchIndexCache.set(ch.id, entries);
      } catch (e) { /* skip */ }
    }
  }

  function performSearch(query) {
    if (query.length < 2) {
      searchResults.innerHTML = '';
      return;
    }

    const terms = query.toLowerCase().split(/\s+/).filter(t => t.length > 1);
    const results = [];

    for (const [chId, entries] of searchIndexCache) {
      for (const entry of entries) {
        const textLower = entry.text.toLowerCase();
        const matches = terms.every(t => textLower.includes(t));
        if (matches) {
          results.push(entry);
          if (results.length >= 20) break;
        }
      }
      if (results.length >= 20) break;
    }

    if (results.length === 0) {
      searchResults.innerHTML = '<div class="search-no-results">No results found</div>';
      return;
    }

    let html = '';
    for (const r of results) {
      // Highlight matches
      let snippet = r.text.substring(0, 150);
      for (const term of terms) {
        const regex = new RegExp(`(${escapeRegex(term)})`, 'gi');
        snippet = snippet.replace(regex, '<mark>$1</mark>');
      }
      html += `<div class="search-result-item" data-id="${r.chapterId}">
        <div class="search-result-chapter">${escapeHtml(r.chapterTitle)}</div>
        <div class="search-result-text">${snippet}...</div>
      </div>`;
    }
    searchResults.innerHTML = html;
  }

  // ══════════════════════════════════════════════════════════
  //  THEME
  // ══════════════════════════════════════════════════════════

  function initTheme() {
    const saved = localStorage.getItem('ccm-theme');
    if (saved) {
      document.documentElement.dataset.theme = saved;
    } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
      document.documentElement.dataset.theme = 'dark';
    }
  }

  function toggleTheme() {
    const current = document.documentElement.dataset.theme;
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('ccm-theme', next);
  }

  // ══════════════════════════════════════════════════════════
  //  FONT SIZE
  // ══════════════════════════════════════════════════════════

  function initFontSize() {
    const saved = localStorage.getItem('ccm-font-size');
    if (saved !== null) fontSizeLevel = parseInt(saved);
    applyFontSize();
  }

  function cycleFontSize() {
    fontSizeLevel = (fontSizeLevel + 1) % 3;
    localStorage.setItem('ccm-font-size', fontSizeLevel);
    applyFontSize();
  }

  function applyFontSize() {
    document.documentElement.classList.remove('font-small', 'font-large');
    if (fontSizeLevel === 0) document.documentElement.classList.add('font-small');
    if (fontSizeLevel === 2) document.documentElement.classList.add('font-large');
  }

  // ══════════════════════════════════════════════════════════
  //  PROGRESS BAR
  // ══════════════════════════════════════════════════════════

  function updateProgress() {
    const scrollTop = window.scrollY;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const progress = docHeight > 0 ? Math.min(scrollTop / docHeight * 100, 100) : 0;
    progressBar.style.width = `${progress}%`;
  }

  // ══════════════════════════════════════════════════════════
  //  SCROLL SPY (right margin TOC)
  // ══════════════════════════════════════════════════════════

  function updateScrollSpy() {
    const headings = contentEl.querySelectorAll('h3[id]');
    if (headings.length === 0) return;

    let activeId = null;
    const offset = 120; // topbar + some padding

    for (const heading of headings) {
      if (heading.getBoundingClientRect().top <= offset) {
        activeId = heading.id;
      }
    }

    // Update right margin
    $$('.right-margin-toc a').forEach(a => a.classList.remove('active'));
    if (activeId) {
      const anchor = activeId.replace('sec-', '');
      const activeLink = $(`.right-margin-toc a[data-anchor="${anchor}"]`);
      if (activeLink) activeLink.classList.add('active');
    }

    // Update sidebar section links
    $$('.toc-section.active').forEach(el => el.classList.remove('active'));
    if (activeId) {
      const anchor = activeId.replace('sec-', '');
      const sidebarLink = $(`.toc-section[data-anchor="${anchor}"]`);
      if (sidebarLink) sidebarLink.classList.add('active');
    }
  }

  // ══════════════════════════════════════════════════════════
  //  EVENTS
  // ══════════════════════════════════════════════════════════

  function setupEvents() {
    // Sidebar toggle (mobile)
    sidebarToggle.addEventListener('click', () => {
      sidebarEl.classList.toggle('open');
      sidebarOverlay.classList.toggle('active');
    });
    sidebarOverlay.addEventListener('click', () => {
      sidebarEl.classList.remove('open');
      sidebarOverlay.classList.remove('active');
    });

    // Sidebar chapter clicks
    sidebarTocEl.addEventListener('click', (e) => {
      const chapterLink = e.target.closest('.toc-chapter');
      const sectionLink = e.target.closest('.toc-section');

      if (sectionLink) {
        e.preventDefault();
        const id = sectionLink.dataset.id;
        const anchor = sectionLink.dataset.anchor;
        navigateTo(id, anchor);
        closeMobileSidebar();
      } else if (chapterLink) {
        e.preventDefault();
        navigateTo(chapterLink.dataset.id);
        closeMobileSidebar();
      }
    });

    // Bottom nav clicks
    navPrev.addEventListener('click', (e) => {
      e.preventDefault();
      if (navPrev.dataset.id) navigateTo(navPrev.dataset.id);
    });
    navNext.addEventListener('click', (e) => {
      e.preventDefault();
      if (navNext.dataset.id) navigateTo(navNext.dataset.id);
    });

    // Right margin clicks
    rightMarginToc.addEventListener('click', (e) => {
      const link = e.target.closest('a');
      if (link) {
        e.preventDefault();
        const anchor = link.dataset.anchor;
        const target = document.getElementById(`sec-${anchor}`);
        if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });

    // Theme toggle
    themeToggle.addEventListener('click', toggleTheme);

    // Font size toggle
    fontToggle.addEventListener('click', cycleFontSize);

    // Search
    searchToggle.addEventListener('click', () => {
      searchPanel.classList.toggle('active');
      if (searchPanel.classList.contains('active')) {
        searchInput.focus();
        // Build full index on first search open
        if (searchIndexCache.size < allChapters.length) {
          buildFullSearchIndex();
        }
      }
    });

    searchInput.addEventListener('input', () => {
      performSearch(searchInput.value.trim());
    });

    searchResults.addEventListener('click', (e) => {
      const item = e.target.closest('.search-result-item');
      if (item) {
        navigateTo(item.dataset.id);
        searchPanel.classList.remove('active');
        searchInput.value = '';
        searchResults.innerHTML = '';
      }
    });

    // Close search on click outside
    document.addEventListener('click', (e) => {
      if (!e.target.closest('#search-container')) {
        searchPanel.classList.remove('active');
      }
    });

    // Escape key closes search/sidebar
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        searchPanel.classList.remove('active');
        sidebarEl.classList.remove('open');
        sidebarOverlay.classList.remove('active');
      }
      // Arrow key navigation
      if (e.key === 'ArrowLeft' && !e.target.closest('input')) {
        const idx = allChapters.findIndex(c => c.id === currentChapter);
        if (idx > 0) navigateTo(allChapters[idx - 1].id);
      }
      if (e.key === 'ArrowRight' && !e.target.closest('input')) {
        const idx = allChapters.findIndex(c => c.id === currentChapter);
        if (idx < allChapters.length - 1) navigateTo(allChapters[idx + 1].id);
      }
    });

    // Hash change
    window.addEventListener('hashchange', () => {
      const hash = window.location.hash.slice(1);
      if (hash && hash !== currentChapter) {
        navigateTo(hash);
      }
    });

    // Scroll events
    let scrollTicking = false;
    window.addEventListener('scroll', () => {
      if (!scrollTicking) {
        requestAnimationFrame(() => {
          updateProgress();
          updateScrollSpy();
          scrollTicking = false;
        });
        scrollTicking = true;
      }
    });
  }

  function closeMobileSidebar() {
    sidebarEl.classList.remove('open');
    sidebarOverlay.classList.remove('active');
  }

  // ══════════════════════════════════════════════════════════
  //  UTILITIES
  // ══════════════════════════════════════════════════════════

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  // ─── Start ───
  init();

})();
