/* Figure-widget loader — CLASSIC script (NOT an ES module), so app.js (a classic
   IIFE) can call window.mountFigures. Widget files ARE ES modules, loaded here via
   dynamic import(). This file is loaded only by index.html and is NEVER injected
   into the PDF build (html_to_pdf.py injects only MathJax + PRINT_CSS) — so widgets
   never mount in print; the static SVG fallback is what prints. See PLAN.md Phase 2. */
(function () {
  'use strict';
  var instances = [];

  function destroyAll() {
    instances.forEach(function (i) { try { i && i.destroy && i.destroy(); } catch (e) {} });
    instances = [];
  }

  function redrawAll() {
    instances.forEach(function (i) { try { i && i.redraw && i.redraw(); } catch (e) {} });
  }

  // Mount every [data-widget] figure inside `root`. Destroys any previously
  // mounted widgets first (called on each chapter navigation).
  window.mountFigures = function (root) {
    destroyAll();
    var els = (root || document).querySelectorAll('[data-widget]');
    Array.prototype.forEach.call(els, function (el) {
      var name = el.dataset.widget;
      var stage = el.querySelector('.fig-stage') || el;
      // dynamic import() from a classic script resolves relative to the document
      // base URL (index.html at website/), so './figures/<name>.js' is correct.
      import('./figures/' + name + '.js').then(function (mod) {
        if (!mod || !mod.mount) throw new Error('widget "' + name + '" exports no mount()');
        var canvas = document.createElement('canvas');
        canvas.className = 'fig-widget';
        canvas.setAttribute('role', 'img');
        stage.appendChild(canvas);
        var params = {};
        try { params = JSON.parse(el.dataset.params || '{}'); } catch (e) {}
        var inst = mod.mount(canvas, params);
        var controls = el.querySelector('.fig-controls');
        if (controls) controls.hidden = false;
        el.classList.add('has-widget');    // CSS then hides .fig-fallback on screen
        instances.push(inst);
      }).catch(function (e) {
        // On any failure leave the static .fig-fallback visible (no .has-widget).
        console.warn('figure widget "' + name + '" failed to mount:', e);
      });
    });
  };

  // Redraw widgets when the theme changes — via the data-theme attribute (toggle)
  // or the OS preference (only when no explicit data-theme is set).
  new MutationObserver(redrawAll).observe(document.documentElement,
    { attributes: true, attributeFilter: ['data-theme'] });
  window.matchMedia('(prefers-color-scheme:dark)').addEventListener('change', function () {
    if (!document.documentElement.getAttribute('data-theme')) redrawAll();
  });
})();
