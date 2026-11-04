// The walk, checked in a real browser (Chromium, headless): open a photo, step forward and
// back, turn, open "today" and the plan; screenshots at phone and laptop sizes; the credit
// and licence of every photo shown. Usage:
//   node tests/visual/walk.mjs [base-url] [screenshot-folder]
// Exits 1 when a check fails.
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';

const base = process.argv[2] ?? 'http://localhost:5173/';
const out = process.argv[3] ?? 'screenshots';
mkdirSync(out, { recursive: true });

const failures = [];
const check = (ok, what) => {
  if (!ok) failures.push(what);
  console.log(`${ok ? 'ok ' : 'NOT'} ${what}`);
};

const walk = await (await fetch(new URL('data/walk.json', base))).json();
const byId = new Map(walk.photos.map((p) => [p.id, p]));
// a start from which every way can be tested: forward, then back from there
const canGoAndComeBack = (p) => p.links.forward && byId.get(p.links.forward.id)?.links.back;
const start = walk.photos.find((p) => canGoAndComeBack(p) && p.today?.image) ?? walk.photos.find(canGoAndComeBack);
if (!start) {
  console.log('NOT a photo from which one can step forward and back');
  process.exit(1);
}

const browser = await chromium.launch();
for (const [name, viewport] of [['portable', { width: 1440, height: 900 }], ['telephone', { width: 390, height: 844 }]]) {
  const page = await browser.newPage({ viewport, reducedMotion: 'reduce' });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  const t0 = Date.now();
  await page.goto(`${base}#/${encodeURIComponent(start.id)}`);
  await page.waitForFunction(() => {
    const img = document.getElementById('photo');
    return img.complete && img.naturalWidth > 0 && getComputedStyle(img).opacity === '1';
  });
  console.log(`${name}: first photo shown in ${Date.now() - t0} ms`);
  await page.waitForTimeout(1500); // map tiles
  await page.screenshot({ path: `${out}/${name}-photo.png` });

  const hash = () => page.evaluate(() => decodeURIComponent(location.hash.slice(2)));
  check((await hash()) === start.id, `${name}: the URL names the photo shown`);
  check((await page.textContent('#credit')).includes('CC0'), `${name}: credit and licence shown`);

  await page.keyboard.press('ArrowUp');
  await page.waitForFunction((id) => decodeURIComponent(location.hash.slice(2)) === id, start.links.forward.id);
  check(true, `${name}: forward leads to ${start.links.forward.id}`);
  await page.keyboard.press('ArrowDown');
  const back = byId.get(start.links.forward.id).links.back.id;
  await page.waitForFunction((id) => decodeURIComponent(location.hash.slice(2)) === id, back);
  check(true, `${name}: back leads to ${back}`);

  await page.goto(`${base}#/${encodeURIComponent(start.id)}`);
  await page.waitForTimeout(500);
  if (start.today?.image) {
    await page.click('.era.today');
    await page.waitForFunction(() => document.getElementById('photo').src.includes('/today/'));
    check((await page.textContent('#credit')).includes('Panoramax'), `${name}: today shows its credit`);
    await page.waitForTimeout(400);
    await page.screenshot({ path: `${out}/${name}-today.png` });
    await page.click('.era.today');
  }

  await page.click('#minimap');
  await page.waitForTimeout(1500);
  check(await page.isVisible('#plan-map'), `${name}: the plan opens`);
  await page.screenshot({ path: `${out}/${name}-plan.png` });
  await page.keyboard.press('Escape');
  check(!(await page.isVisible('#plan-map')), `${name}: Escape closes the plan`);

  const sizes = await page.evaluate(() =>
    Object.fromEntries(['#photo', '#plate', '.step[data-way="forward"]', '#when'].map((s) => {
      const r = document.querySelector(s)?.getBoundingClientRect();
      return [s, r ? [Math.round(r.width), Math.round(r.height)] : null];
    })),
  );
  console.log(`${name}: sizes`, JSON.stringify(sizes));
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  check(!overflow, `${name}: no sideways scrolling`);
  check(errors.length === 0, `${name}: no script error ${errors.join(' | ')}`);
  await page.close();
}

// the first photo on a slow 4G connection (Lighthouse's mobile profile: 1.6 Mbit/s, 150 ms)
{
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Network.enable');
  await cdp.send('Network.emulateNetworkConditions', {
    offline: false, latency: 150, downloadThroughput: (1.6 * 1024 * 1024) / 8, uploadThroughput: (750 * 1024) / 8,
  });
  await page.goto(`${base}#/${encodeURIComponent(start.id)}`);
  // measured inside the page, from the start of navigation to the photo shown
  await page.waitForFunction(() => performance.getEntriesByName('first-photo').length, null, { timeout: 30000 });
  const ms = Math.round(await page.evaluate(() => performance.getEntriesByName('first-photo')[0].startTime));
  check(ms < 2000, `first photo on slow 4G in ${ms} ms (goal < 2000)`);
  await page.close();
}

// every photo of the walk can be opened, with its credit
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, reducedMotion: 'reduce' });
let shown = 0;
for (const photo of walk.photos) {
  await page.goto(`${base}#/${encodeURIComponent(photo.id)}`);
  await page.waitForFunction(() => document.getElementById('photo').naturalWidth > 0);
  const credit = await page.textContent('#credit');
  if (credit.includes(photo.credit) && credit.includes('CC0')) shown += 1;
}
check(shown === walk.photos.length, `every photo opens with its credit (${shown}/${walk.photos.length})`);
await browser.close();
process.exit(failures.length ? 1 : 0);
