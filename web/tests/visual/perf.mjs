// Where the site lags: a phone-like Chromium (390x844 at 2.6x, CPU 4x slower), measuring
// frames per second while the plan pans and while photos are stepped through, the tasks
// blocking the page (> 50 ms) in each phase, and what stepping fetches. Usage:
//   node tests/visual/perf.mjs [base-url]
import { chromium } from 'playwright';
const base = process.argv[2] ?? 'http://localhost:4174/flaneur/';
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2.6, hasTouch: false, bypassCSP: true });
const page = await ctx.newPage();
const cdp = await ctx.newCDPSession(page);
await cdp.send('Emulation.setCPUThrottlingRate', { rate: 4 });
await page.addInitScript(() => {
  window.__long = [];
  new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__long.push(e.duration); }).observe({ type: 'longtask', buffered: true });
  window.__frames = (ms) => new Promise((done) => {
    let n = 0; const t0 = performance.now(); const gaps = []; let last = t0;
    const f = (t) => { n++; gaps.push(t - last); last = t; if (t - t0 < ms) requestAnimationFrame(f); else done({ fps: Math.round((n * 1000) / (t - t0)), worst: Math.round(Math.max(...gaps)) }); };
    requestAnimationFrame(f);
  });
});
const phase = async (name, run) => {
  await page.evaluate(() => (window.__long = []));
  const t0 = Date.now();
  const extra = await run();
  const long = await page.evaluate(() => window.__long);
  console.log(`${name}: ${Date.now() - t0} ms; blocking tasks ${long.length}, total ${Math.round(long.reduce((a, b) => a + b, 0))} ms, worst ${Math.round(Math.max(0, ...long))} ms` + (extra ? `; ${JSON.stringify(extra)}` : ''));
};
await phase('plan opens', async () => {
  await page.goto(base);
  await page.waitForSelector('#map canvas');
  await page.waitForLoadState('networkidle');
});
await page.click('#welcome-close');
await phase('plan pans (2 s drag)', async () => {
  const frames = page.evaluate(() => window.__frames(2000));
  await page.mouse.move(200, 400);
  await page.mouse.down();
  for (let i = 0; i < 40; i++) { await page.mouse.move(200 - i * 4, 400 - i * 3); await page.waitForTimeout(40); }
  await page.mouse.up();
  return await frames;
});
await phase('plan zooms (wheel)', async () => {
  const frames = page.evaluate(() => window.__frames(1500));
  for (let i = 0; i < 6; i++) { await page.mouse.wheel(0, -200); await page.waitForTimeout(150); }
  return await frames;
});
const walk = await (await fetch(new URL('data/walk.json', base))).json();
const start = walk.photos.find((p) => p.links.forward) ?? walk.photos[0];
await phase('a photo opens', async () => {
  await page.goto(`${base}#/${encodeURIComponent(start.id)}`);
  await page.waitForFunction(() => { const i = document.getElementById('photo'); return i.complete && i.naturalWidth > 0; });
  await page.waitForLoadState('networkidle');
});
await phase('5 steps forward', async () => {
  const bodies = [];
  const count = (r) => bodies.push(r.body().then((b) => b.length, () => 0));
  page.on('response', count);
  const frames = page.evaluate(() => window.__frames(5000));
  for (let i = 0; i < 5; i++) { await page.keyboard.press('ArrowUp'); await page.waitForTimeout(900); }
  const seen = await frames;
  await page.waitForLoadState('networkidle'); // what was fetched ahead too
  page.off('response', count);
  const bytes = (await Promise.all(bodies)).reduce((a, b) => a + b, 0);
  return { ...seen, fetched: `${Math.round(bytes / 1024)} KB` };
});
await phase('back to the plan', async () => {
  const frames = page.evaluate(() => window.__frames(1500));
  await page.click('#to-plan');
  return await frames;
});
await phase('into the photo from its print', async () => {
  const frames = page.evaluate(() => window.__frames(1500));
  await page.click('#enter');
  return await frames;
});
await browser.close();
