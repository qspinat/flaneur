// Renders the local review page (uv run flaneur review) and saves a screenshot.
// Usage: node tests/visual/review_screenshot.mjs <out.png> [width] [height] [item index]
// The item index only changes what the page shows: no decision is recorded.
import { chromium } from 'playwright';

const [out, width = '1440', height = '900', skip = '0'] = process.argv.slice(2);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: +width, height: +height } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
await page.goto('http://127.0.0.1:8765/');
await page.waitForFunction(() => document.getElementById('progress').textContent.includes('/'));
// Show the item at position `skip` without recording anything: only the page state changes.
await page.evaluate((n) => { index = n; show(); }, +skip);
await page.waitForTimeout(2500); // map tiles
const sizes = await page.evaluate(() => {
  const box = (id) => document.getElementById(id).getBoundingClientRect();
  return { photo: box('photo'), map: box('map'), title: box('title') };
});
await page.screenshot({ path: out });
console.log(JSON.stringify({ errors, progress: await page.textContent('#progress'), sizes }));
await browser.close();
