/* Shared helpers for interactive figure widgets (ES module).
   Widgets read colors from the themeable --fig-* tokens so they stay correct
   in both light and dark mode. See PLAN.md Phase 2/3. */

/** Size a canvas to its CSS box × devicePixelRatio and return a ready 2D context. */
export function fit(cv) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const r = cv.getBoundingClientRect();
  cv.width = Math.max(1, Math.round(r.width * dpr));
  cv.height = Math.max(1, Math.round(r.height * dpr));
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: r.width, h: r.height };
}

/** Resolve a CSS custom property (e.g. '--fig-crimson') from an element (default :root). */
export function cvar(name, el) {
  return getComputedStyle(el || document.documentElement).getPropertyValue(name).trim();
}

export const reducedMotion = () =>
  window.matchMedia('(prefers-reduced-motion:reduce)').matches;

/** Build a labelled range slider into `parent`; returns the <input>. onInput(value). */
export function slider(parent, { label, min, max, step, value, fmt }, onInput) {
  const wrap = document.createElement('div');
  wrap.className = 'fw-ctrl';
  const row = document.createElement('div');
  row.className = 'fw-row';
  const name = document.createElement('span');
  name.className = 'fw-name';
  name.textContent = label;
  const val = document.createElement('span');
  val.className = 'fw-val';
  val.textContent = fmt ? fmt(value) : value;
  const input = document.createElement('input');
  input.type = 'range';
  input.min = min; input.max = max; input.step = step; input.value = value;
  input.setAttribute('aria-label', label);
  input.addEventListener('input', () => {
    const v = parseFloat(input.value);
    val.textContent = fmt ? fmt(v) : v;
    onInput(v);
  });
  row.append(name, val);
  wrap.append(row, input);
  parent.appendChild(wrap);
  return input;
}

/** Build a button into `parent`. */
export function button(parent, label, onClick, primary) {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'fw-btn' + (primary ? ' fw-btn-primary' : '');
  b.textContent = label;
  b.addEventListener('click', onClick);
  parent.appendChild(b);
  return b;
}
