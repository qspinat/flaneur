// Shared by the local pages (see localpage.page): zoom on a picture within its box.

/** Zoom `target` within `box`: the wheel zooms towards the cursor (1 to 8 times), dragging
 *  moves it once zoomed (never past its own edges; the click ending a move is swallowed),
 *  a double click puts it back (unless `doubleClick` is false). `onZoom(scale)` is told each
 *  new scale, `onMove()` each change of the view. Returns `reset()` and `zoomBy(factor)`
 *  (towards the box's middle). */
function zoomable(box, target, { onZoom = () => {}, onMove = () => {}, doubleClick = true } = {}) {
  const MAX_ZOOM = 8;
  let scale = 1, x = 0, y = 0, drag = null, moved = false;
  target.style.transformOrigin = '0 0';
  const apply = () => {
    const w = target.offsetWidth, h = target.offsetHeight;  // as laid out, unscaled
    x = Math.min(0, Math.max(w - w * scale, x));
    y = Math.min(0, Math.max(h - h * scale, y));
    target.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
    box.classList.toggle('zoomed', scale > 1);
    onMove();
  };
  // the target's point at (cx, cy) on screen stays there
  const zoomAt = (factor, cx, cy) => {
    const rect = target.getBoundingClientRect();
    const px = (cx - rect.left) / scale, py = (cy - rect.top) / scale;
    const next = Math.min(MAX_ZOOM, Math.max(1, scale * factor));
    x += px * (scale - next); y += py * (scale - next); scale = next;
    apply(); onZoom(scale);
  };
  const reset = () => { scale = 1; x = 0; y = 0; apply(); onZoom(1); };
  box.addEventListener('wheel', (event) => {
    event.preventDefault();
    zoomAt(Math.exp(-event.deltaY / 600), event.clientX, event.clientY);
  }, { passive: false });
  box.addEventListener('mousedown', (event) => {
    moved = false;
    if (scale === 1 || event.button !== 0) return;
    event.preventDefault();  // no picture dragged out of the page
    drag = { x: event.clientX - x, y: event.clientY - y, fromX: event.clientX, fromY: event.clientY };
  });
  window.addEventListener('mousemove', (event) => {
    if (!drag) return;
    if (!moved && Math.hypot(event.clientX - drag.fromX, event.clientY - drag.fromY) < 4) return;
    moved = true;  // a move, not a click
    box.classList.add('panning');
    x = event.clientX - drag.x; y = event.clientY - drag.y;
    apply();
  });
  window.addEventListener('mouseup', () => { drag = null; box.classList.remove('panning'); });
  box.addEventListener('click', (event) => {
    if (moved) { event.stopPropagation(); event.preventDefault(); moved = false; }
  }, true);
  if (doubleClick) box.addEventListener('dblclick', reset);
  const zoomBy = (factor) => {
    const rect = box.getBoundingClientRect();
    zoomAt(factor, rect.left + rect.width / 2, rect.top + rect.height / 2);
  };
  return { reset, zoomBy };
}
