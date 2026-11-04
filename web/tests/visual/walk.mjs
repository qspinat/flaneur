// The walk, checked in a real browser (headless Chromium, or Firefox with BROWSER=firefox):
// the plan the site opens on and its welcome; open a photo, step forward and back, turn, open
// "today", go back to the plan and in again from the photo's print; screenshots at phone
// and laptop sizes; the credit and licence of every photo shown; nothing the pages' security
// policy refuses. Usage:
//   [BROWSER=firefox] node tests/visual/walk.mjs [base-url] [screenshot-folder]
// Exits 1 when a check fails.
import { chromium, firefox } from 'playwright';
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

const engine = process.env.BROWSER === 'firefox' ? firefox : chromium;
const browser = await engine.launch();
console.log(`browser: ${engine.name()}`);
for (const [name, viewport] of [['portable', { width: 1440, height: 900 }], ['telephone', { width: 390, height: 844 }]]) {
  // the checks below evaluate code in the page, which its policy refuses: checked apart, after
  const errors = [];

  // the site opens on its plan, with a welcome that leads into a photograph (a page of its own:
  // the photograph's own first load is measured below, from cold)
  {
    const home = await browser.newPage({ viewport, reducedMotion: 'reduce', bypassCSP: true });
    home.on('pageerror', (e) => errors.push(e.message));
    await home.goto(base);
    await home.waitForSelector('#map canvas');
    check(await home.isVisible('#welcome'), `${name}: the plan opens with its welcome`);
    await home.waitForTimeout(1500); // map tiles
    await home.screenshot({ path: `${out}/${name}-plan.png` });
    await home.$eval('.years-min', (e) => { e.value = '1900'; e.dispatchEvent(new Event('input', { bubbles: true })); });
    check((await home.textContent('#plan-years .years-label')) === '1900 – 1950', `${name}: the plan's years can be narrowed`);
    await home.$eval('.years-min', (e) => { e.value = '1850'; e.dispatchEvent(new Event('input', { bubbles: true })); });
    await home.click('#start');
    await home.waitForFunction(() => location.hash.length > 2 && !document.getElementById('walk').hidden);
    check(true, `${name}: the welcome leads into a photograph`);
    await home.close();
  }

  const page = await browser.newPage({ viewport, reducedMotion: 'reduce', bypassCSP: true });
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

  const ways = await page.$$eval('.onward', (els) => els.map((e) => e.dataset.way));
  const expected = Object.entries(start.links).filter(([, l]) => l).map(([w]) => w);
  check(expected.every((w) => ways.includes(w)), `${name}: each way shows where it leads (${ways.join(', ')})`);
  await page.click('.onward[data-way="forward"]');
  await page.waitForFunction((id) => decodeURIComponent(location.hash.slice(2)) === id, start.links.forward.id);
  check(true, `${name}: a way leads to the photo its arrow leads to`);
  await page.goto(`${base}#/${encodeURIComponent(start.id)}`);
  await page.waitForFunction((id) => decodeURIComponent(location.hash.slice(2)) === id, start.id);

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
  await page.waitForSelector('#print', { state: 'visible' });
  check((await page.textContent('#print-title')) === start.title, `${name}: the plan shows the photo's print`);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: `${out}/${name}-print.png` });
  await page.click('#enter');
  await page.waitForFunction((id) => decodeURIComponent(location.hash.slice(2)) === id && !document.getElementById('walk').hidden, start.id);
  check(true, `${name}: the print leads back into the photograph`);
  await page.keyboard.press('m');
  await page.waitForSelector('#print', { state: 'visible' });
  await page.keyboard.press('Escape');
  check(!(await page.isVisible('#print')), `${name}: Escape sets the print aside`);
  await page.goBack();
  await page.waitForFunction(() => !document.getElementById('walk').hidden);
  check(true, `${name}: the browser's back returns to the photograph`);

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

// with the pages' security policy in force: a photo, the plan, the 3D walk load, and nothing
// they need is refused (clicks and waits only: nothing evaluated in the page)
{
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  // each refusal said on the console from inside the page (Firefox reports them only to its
  // own web console); an init script runs whatever the page's policy
  await page.addInitScript(() =>
    document.addEventListener('securitypolicyviolation', (e) =>
      console.error(`Content Security Policy refused ${e.violatedDirective} ${e.blockedURI}`),
    ),
  );
  const refused = [];
  const errors = [];
  page.on('console', (m) => /Content.Security.Policy|violates/i.test(m.text()) && refused.push(m.text()));
  page.on('pageerror', (e) => errors.push(e.message));
  // waits for the network to rest (no evaluation in the page), then a moment for late refusals
  const settled = () => page.waitForLoadState('networkidle').then(() => page.waitForTimeout(500));
  await page.goto(base);
  await settled();
  await page.goto(`${base}#/${encodeURIComponent(start.id)}`);
  await settled();
  await page.click('#minimap');
  await page.waitForSelector('#print', { state: 'visible' });
  await settled();
  await page.goto(new URL('3d.html', base).href);
  await settled();
  await page.screenshot({ path: `${out}/policy-3d.png` });
  check(refused.length === 0, `security policy: nothing refused ${refused.slice(0, 3).join(' | ')}`);
  check(errors.length === 0, `security policy: no script error ${errors.slice(0, 3).join(' | ')}`);
  await page.close();
}

// the first photo on a slow 4G connection (Lighthouse's mobile profile: 1.6 Mbit/s, 150 ms);
// throttled through Chromium's own protocol, so measured in Chromium only
if (engine === chromium) {
  const page = await browser.newPage({ viewport: { width: 390, height: 844 }, bypassCSP: true }); // its waits evaluate code
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
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, reducedMotion: 'reduce', bypassCSP: true });
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
